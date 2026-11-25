"""Chest X-ray sentence alignment of block 24 (Methods, "Sentence alignment", X-ray).

Same text side, loss, step, schedule (including ``epochs_schedule``) and checkpoints as ``train_ct``. Image side, computed online (no cache):
* each film of the report (at most 8): content crop, percentile stretch to [-1, 1], native resolution, cut into
  1,024-px tiles with token ownership (``fora.encoder.cxr.film_tiles``);
* frozen X-ray teacher blocks 1-23 once per step; trainable block 24 + norm (initialized from the teacher) recomputed
  tile chunk by tile chunk in both passes -> (y - mean) / rms ++ film-grid position features -> projection + view kind
  (0 frontal: PA / AP / frontal, 1 lateral, 2 other).

Subcommands
    index          --config CFG --prep DIR         training reports with statements and valid films -> DIR/cases_train.jsonl
    normalization  --config CFG --films 200        (torchrun) per-dimension mean and scalar rms of the initial block-24
                                                   tokens of every token of 200 random training films -> CFG normalization
    train          --config CFG --output OUT       (torchrun) training
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.distributed as dist

from fora.encoder.cxr import CXRSplitEncoder, film_tiles, valid_film_shape
from fora.geometry.xray import prepare_film, series_meta

from .common import (RESUME_FREE, Stopper, ThreadLoader, alignment_step, build_model, horizon_epochs, init_distributed,
                     lr_fraction, save_state, size_bucketed_order, write_json)

PATCH_PX = 16
VIEW_KIND = {"frontal": 0, "lateral": 1, "other": 2}
_VIEW_OF = {"pa": 0, "ap": 0, "frontal-?": 0, "lateral": 1}


# ------------------------------------------------------------------------------------------------------------- index
def _index_shard(path: str) -> list[dict]:
    out, h5_id = [], Path(path).stem
    with h5py.File(path, "r") as f:
        for patient in f:
            pg = f[patient]
            if not hasattr(pg, "keys"):
                continue
            for rid in pg:
                try:
                    rg = pg[rid]
                    if not hasattr(rg, "keys") or "report" not in rg or "there_is_statements" not in rg["report"]:
                        continue
                    try:
                        v = rg["report"]["there_is_statements"][()]
                        raw = json.loads(v.decode() if isinstance(v, bytes) else v)
                    except Exception:
                        raw = []
                    sents = [s.strip() for s in raw if isinstance(s, str) and s.strip()] if isinstance(raw, list) else []
                    if not sents:
                        continue
                    films = []
                    for acc in rg:
                        if acc == "report":
                            continue
                        ag = rg[acc]
                        if not hasattr(ag, "keys") or ag.attrs.get("kind") == "report":
                            continue
                        for sid in ag:
                            sg = ag[sid]
                            if not hasattr(sg, "keys") or "data" not in sg or not valid_film_shape(sg["data"].shape):
                                continue
                            vc = str(series_meta(sg).get("view_class", "")).lower()
                            h, w = (int(v) for v in sg["data"].shape[:2])
                            films.append([acc, sid, _VIEW_OF.get(vc, 2), h, w])
                    if films:
                        out.append(dict(id=f"{h5_id}/{patient}/{rid}", h5=h5_id, patient=patient, report=rid,
                                        sentences=sents, films=films))
                except Exception as e:
                    print(json.dumps(dict(event="index_skip", shard=h5_id, report=rid, error=repr(e)[:200])), flush=True)
    return out


def build_index(m: dict, prep: Path, workers: int = 16) -> None:
    from multiprocessing import Pool
    shards = [r["h5_id"] for r in csv.DictReader(open(m["splits_csv"])) if r["split"].strip() == "train"]
    with Pool(workers) as p:
        res = p.map(_index_shard, [f"{m['h5_folder']}/{s}.h5" for s in shards])
    cases = [c for r in res for c in r]
    prep.mkdir(parents=True, exist_ok=True)
    with open(prep / "cases_train.jsonl", "w") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")
    print(json.dumps(dict(event="index", shards=len(shards), reports=len(cases),
                          films=sum(len(c["films"]) for c in cases))), flush=True)


# -------------------------------------------------------------------------------------------------------------- data
def prepare(case: dict, cfg: dict, h5_folder: str, fallback: dict | None = None) -> dict:
    """All films of one report -> tiles, owned-token selections, position features and per-token view kind."""
    T, freqs = cfg.get("tile", 1024), cfg.get("position_freqs", 4)
    tiles, sel, pf, kind = [], [], [], []
    for film in case["films"][:cfg.get("max_films", 8)]:
        acc, sid = film[:2]
        try:
            with h5py.File(f"{h5_folder}/{case['h5']}.h5", "r") as f:
                u = prepare_film(f[case["patient"]][case["report"]][acc][sid]["data"][()])
        except Exception as e:
            print(json.dumps(dict(event="film_skip", case=case["id"], film=sid, error=repr(e)[:200])), flush=True)
            continue
        t, s, p, _ = film_tiles(u, T, freqs)
        tiles.append(t); sel += s; pf.append(p); kind.append(np.full(len(p), film[2], np.int64))
    if not tiles:
        if fallback is None or fallback is case:
            raise RuntimeError(f"no loadable film in {case['id']}")
        return prepare(fallback, cfg, h5_folder)
    return dict(id=case["id"], tiles=torch.cat(tiles), sel=sel, pos=torch.from_numpy(np.concatenate(pf)),
                kind=torch.from_numpy(np.concatenate(kind)))


def tile_chunks(enc, h23, bag, chunk, mean, rms, pscale, device):
    """(x, kind) per chunk of tiles, in a fixed order (both passes must see identical chunks)."""
    o = 0
    for i in range(0, len(h23), chunk):
        y = enc.b24(h23[i:i + chunk])
        f = torch.cat([y[b][torch.as_tensor(s, device=device)] for b, s in enumerate(bag["sel"][i:i + chunk])]).float()
        n = len(f)
        pf, kd = bag["pos"][o:o + n].to(device).float(), bag["kind"][o:o + n].to(device)
        o += n
        yield torch.cat(((f - mean) / rms, pf * pscale), 1), kd


def read_cases(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def encoder_for(m, device) -> CXRSplitEncoder:
    return CXRSplitEncoder(m["teacher"], torch.load(m["block24_state"], map_location="cpu", weights_only=False), device)


# ----------------------------------------------------------------------------------------------------- normalization
def normalization(m: dict, n_films: int) -> None:
    """Every token of the first film of ``n_films`` random training reports (rng seed 12345), initial block 24 output."""
    rank, world, device, barrier = init_distributed()
    cfg = m["training"]
    torch.set_num_threads(4)
    enc = encoder_for(m, device)
    cases = read_cases(m["cases"])
    pick = np.random.default_rng(12345).permutation(len(cases))[:n_films]
    films = [dict(cases[i], films=cases[i]["films"][:1]) for i in pick]
    sums = torch.zeros(1024, dtype=torch.float64, device=device)
    sq, n = torch.zeros_like(sums), torch.zeros((), dtype=torch.float64, device=device)
    used = torch.zeros((), dtype=torch.float64, device=device)
    for case in films[rank::world]:
        try:
            bag = prepare(case, cfg, m["h5_folder"])
        except Exception as e:
            print(json.dumps(dict(event="norm_skip", case=case["id"], error=repr(e)[:200])), flush=True)
            continue
        h = enc.h23(bag["tiles"], cfg.get("frozen_batch", 8))
        with torch.no_grad():
            for i in range(len(h)):
                f = enc.b24(h[i:i + 1])[0][torch.as_tensor(bag["sel"][i], device=device)].double()
                sums += f.sum(0); sq += f.square().sum(0); n += len(f)
        used += 1
    for v in (sums, sq, n, used):
        dist.all_reduce(v)
    mean = sums / n
    rms = float((sq / n - mean.square()).clamp_min(0).mean().sqrt().clamp_min(1e-8))
    if rank == 0:
        save_state(Path(m["normalization"]), mean=mean.float().cpu(), rms=rms, tokens=int(n), films=int(used),
                   teacher_step=enc.step)
        print(json.dumps(dict(event="normalization", films=int(used), tokens=int(n), rms=rms)), flush=True)
    barrier()
    dist.destroy_process_group()


# ------------------------------------------------------------------------------------------------------------- train
def train(m: dict, out: Path, max_steps: int) -> None:
    cfg, spec = m["training"], m["training"]["model"]
    out.mkdir(parents=True, exist_ok=True)
    rank, world, device, barrier = init_distributed()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(cfg["seed"])
    src = torch.load(m["normalization"], map_location=device, weights_only=False)
    mean, rms = src["mean"].to(device).float(), float(src["rms"])
    enc = encoder_for(m, device)
    block = enc.top
    vproj, text, tok, temp, pool_temp, opt, params = build_model(block, spec, cfg.get("position_freqs", 4), len(VIEW_KIND), device)
    model = dict(vproj=vproj, text=text, tok=tok, temp=temp, pool_temp=pool_temp, optimizer=opt, params=params)

    allc = read_cases(m["cases"])                    # one fixed global shuffle; rank r takes every world-th report
    perm = np.random.default_rng(cfg["seed"]).permutation(len(allc))
    mine = [allc[i] for i in perm[rank::world]]
    del allc
    cases = {c["id"]: c for c in mine}
    train_ids = [c["id"] for c in mine]
    counts = [None] * world
    dist.all_gather_object(counts, len(train_ids))
    steps = min(counts)
    fallback = cases[train_ids[0]]
    loader = ThreadLoader(lambda case: prepare(case, cfg, m["h5_folder"], fallback), cfg.get("prefetch_depth", 4),
                          cfg.get("loader_workers", 3))
    stopper = Stopper()
    statepath, epoch0, batch0, step = out / "latest.pt", 0, 0, 0
    if statepath.exists():
        s = torch.load(statepath, map_location=device, weights_only=False)
        same = lambda c: {k: v for k, v in c.items() if k not in RESUME_FREE}
        if same(s["training"]) != same(cfg) or s["world"] != world:
            raise ValueError("latest.pt was written with a different recipe or world size")
        block.load_state_dict(s["block"]); vproj.load_state_dict(s["vproj"]); text.load_state_dict(s["text"])
        opt.load_state_dict(s["optimizer"])
        with torch.no_grad():
            temp.copy_(s["temp"]); pool_temp.copy_(s["pool_temp"])
        epoch0, batch0, step = s["epoch"], s["batch"], s["step"]
        del s

    def save(epoch, batch, name="latest.pt"):
        if rank == 0:
            save_state(out / name, training=cfg, world=world, epoch=epoch, batch=batch, step=step, block=block.state_dict(),
                       vproj=vproj.state_dict(), text=text.state_dict(), temp=temp.detach(), pool_temp=pool_temp.detach(),
                       optimizer=opt.state_dict(), teacher_step=enc.step)
        barrier()

    E = min(int(cfg["epochs"]), int(cfg.get("stop_after_epoch", cfg["epochs"])))
    warmup, start = cfg["warmup_steps"], time.time()
    chunk, fb, pscale = cfg.get("tiles_per_chunk", 8), cfg.get("frozen_batch", 8), spec.get("position_scale", 1.)
    snapshot = cfg.get("snapshot_every", 2000)
    if rank == 0:
        write_json(out / "setup.json", dict(train_reports=sum(counts), steps_per_epoch=steps, epochs=E, world=world,
                                            model=spec, teacher_step=enc.step, tile=cfg.get("tile", 1024)))
    size = lambda c: sum(-(-f[3] // PATCH_PX) * -(-f[4] // PATCH_PX) for f in cases[c]["films"])
    for epoch in range(epoch0, E):
        order = size_bucketed_order(train_ids, size, steps, cfg["seed"], epoch, rank, cfg.get("bucket_size", 8))
        first, DP = (batch0 if epoch == epoch0 else 0), loader.depth
        loader.pending = []
        for b_ in range(first, min(first + DP, len(order))):
            loader.submit(cases[order[b_]])
        for batch in range(first, steps):
            began = time.perf_counter()
            frac = lr_fraction(step, warmup, steps * horizon_epochs(cfg, step))
            for g in opt.param_groups:
                g["lr"] = g["base"] * frac
            opt.zero_grad(set_to_none=True)
            bag = loader.get()
            if batch + DP < len(order):
                loader.submit(cases[order[batch + DP]])
            h23 = enc.h23(bag["tiles"], fb)
            r = alignment_step(lambda: tile_chunks(enc, h23, bag, chunk, mean, rms, pscale, device),
                               cases[bag["id"]]["sentences"], model=model, cfg=cfg,
                               sentence_seed=cfg["seed"] * 7919 + step * 131 + rank, world=world, device=device)
            if r["tokens"] != len(bag["pos"]):
                raise AssertionError("every owned film token must be pooled exactly once")
            step += 1
            del bag, h23
            if rank == 0 and (step % 10 == 0 or step <= 20):
                r.update(epoch=epoch + 1, batch=batch + 1, batches=steps, step=step, lr_fraction=frac,
                         batch_seconds=time.perf_counter() - began, elapsed=time.time() - start)
                write_json(out / "progress.json", r)
                print(json.dumps(r), flush=True)
            stop = stopper.agreed(bool(max_steps and step >= max_steps), device)
            if step % 200 == 0 or stop:
                save(epoch, batch + 1)
            if step % snapshot == 0:
                save(epoch, batch + 1, f"step_{step:06d}.pt")
            if stop:
                loader.close()
                dist.destroy_process_group()
                return
        save(epoch + 1, 0)
        save(epoch + 1, 0, f"epoch_{epoch + 1:02d}.pt")
    if rank == 0:
        write_json(out / "COMPLETE.json", dict(epochs=E, step=step))
    loader.close()
    barrier()
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("index", "normalization", "train"))
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--prep", type=Path, default=None, help="index: output directory")
    ap.add_argument("--films", type=int, default=200, help="normalization: number of training films")
    ap.add_argument("--output", type=Path, default=None, help="train: run directory")
    ap.add_argument("--max-steps", type=int, default=0)
    a = ap.parse_args()
    m = json.loads(a.config.read_text())
    if a.command == "index":
        build_index(m, a.prep or Path(m["cases"]).parent)
    elif a.command == "normalization":
        normalization(m, a.films)
    else:
        train(m, a.output, a.max_steps)


if __name__ == "__main__":
    main()

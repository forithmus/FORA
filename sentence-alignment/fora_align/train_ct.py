"""CT sentence alignment of block 24 on the cached block-23 tokens of all views of every report (Methods, "Sentence
alignment").

Image side: frozen blocks 1-23 (int8 cache, ``fora.encoder.cache``) -> trainable block 24 + norm initialized from the
pre-trained teacher (``block24_state.pt``) -> normalization + position features -> projection + view kind
(full / native window / lung MIP). Text side: the report's ``there_is_statements``. Loss: ``loss.contrastive_ce``.

    torchrun --nnodes 32 --nproc-per-node 4 ... -m fora_align.train_ct --config configs/ct_alignment.json --output OUT
    torchrun ... -m fora_align.train_ct --config configs/ct_alignment_continuation.json --output OUT2 --init OUT/epoch_03.pt

``--init`` starts a new run (own schedule from step 0) from the weights and AdamW moments of a checkpoint; learning rates
and weight decay come from the new config. A run resumes from OUT/latest.pt; its config may differ from the saved one
only in ``epochs``, ``epochs_schedule``, ``stop_after_epoch`` and ``continuation``.

Schedule: the cosine spans ``epochs`` x steps per epoch. With ``epochs_schedule`` = [[step, epochs], ...] the epoch count
used for the cosine horizon at a step is the last entry whose step is <= the current step (the horizon is stretched from
that step on). ``stop_after_epoch`` ends the run after saving that epoch.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.distributed as dist

from fora.encoder.blocks import build_block24
from fora.encoder.cache import CachedBag, load_case

from .common import (RESUME_FREE, Stopper, ThreadLoader, alignment_step, build_model, horizon_epochs, init_distributed,
                     lr_fraction, save_state, size_bucketed_order, write_json)

VIEW_KIND = {"full": 0, "native": 1, "lung_mip_10mm": 2}



def read_cases(path: Path, ids: set) -> dict:
    cases = {}
    with open(path) as f:
        for line in f:
            c = json.loads(line)
            if c["id"] in ids:
                cases[c["id"]] = c
    return cases


def my_reports(m: dict, rank: int, world: int, all_ids) -> list[str]:
    """This rank's reports: its entry of the rank assignment (train + val lists, filtered by the split later), or a
    round-robin share of the sorted case ids when no assignment file is configured."""
    if m.get("assign"):
        A = json.loads(Path(m["assign"]).read_text())
        if A["ranks"] != world:
            raise ValueError(f"assignment is for {A['ranks']} ranks, world is {world}")
        return [c for s in ("train", "val") for c in A["assignments"][s][rank]]
    return sorted(all_ids)[rank::world]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--init", default="", help="start a new run from this checkpoint (weights + optimizer moments)")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (smoke runs)")
    a = ap.parse_args()
    m = json.loads(a.config.read_text())
    cfg, spec, out = m["training"], m["training"]["model"], a.output
    out.mkdir(parents=True, exist_ok=True)
    rank, world, device, barrier = init_distributed()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(cfg["seed"])
    src = torch.load(m["normalization"], map_location=device, weights_only=False)
    mean, rms = src["mean"].to(device).float(), float(src["rms"])
    block = build_block24(torch.load(m["block24_state"], map_location="cpu", weights_only=False)).to(device).train()
    freqs = cfg.get("position_freqs", 4)
    vproj, text, tok, temp, pool_temp, opt, params = build_model(block, spec, freqs, len(VIEW_KIND), device)
    model = dict(vproj=vproj, text=text, tok=tok, temp=temp, pool_temp=pool_temp, optimizer=opt, params=params)

    split = json.loads(Path(m["split"]).read_text())
    S = json.loads(Path(m["sentences"]).read_text())
    mine = my_reports(m, rank, world, split.keys())
    train = [c for c in mine if split.get(c) == "train" and S.get(c)]
    cases = read_cases(Path(m["cases"]), set(train))
    train = [c for c in train if c in cases]
    counts = [None] * world
    dist.all_gather_object(counts, len(train))
    steps = min(counts)                               # equal steps: every rank takes part in every gather

    cache_root = Path(m["cache_root"])
    loader = ThreadLoader(lambda case: load_case(case, cache_root, freqs), cfg.get("prefetch_depth", 4),
                          cfg.get("loader_workers", 3))
    stopper = Stopper()
    statepath, epoch0, batch0, step = out / "latest.pt", 0, 0, 0

    def restore(s, keep_hparams):
        hyper = [{k: g[k] for k in ("lr", "base", "weight_decay")} for g in opt.param_groups]
        block.load_state_dict(s["block"]); vproj.load_state_dict(s["vproj"]); text.load_state_dict(s["text"])
        opt.load_state_dict(s["optimizer"])
        if keep_hparams:
            for g, h in zip(opt.param_groups, hyper):
                g.update(h)
        with torch.no_grad():
            temp.copy_(s["temp"]); pool_temp.copy_(s["pool_temp"])

    if statepath.exists():
        s = torch.load(statepath, map_location=device, weights_only=False)
        same = lambda c: {k: v for k, v in c.items() if k not in RESUME_FREE}
        if same(s["training"]) != same(cfg) or s["world"] != world:
            raise ValueError("latest.pt was written with a different recipe or world size")
        if s["training"]["epochs"] != cfg["epochs"] and rank == 0:
            print(json.dumps(dict(event="extended", epochs_from=s["training"]["epochs"], epochs_to=cfg["epochs"])), flush=True)
        restore(s, keep_hparams=False)
        epoch0, batch0, step = s["epoch"], s["batch"], s["step"]
        del s
    elif a.init:
        s = torch.load(a.init, map_location=device, weights_only=False)
        restore(s, keep_hparams=True)
        if rank == 0:
            print(json.dumps(dict(event="init", source=a.init, source_step=s["step"], source_epoch=s["epoch"])), flush=True)
        del s

    def save(epoch, batch, name="latest.pt"):
        if rank == 0:
            save_state(out / name, training=cfg, world=world, epoch=epoch, batch=batch, step=step, block=block.state_dict(),
                       vproj=vproj.state_dict(), text=text.state_dict(), temp=temp.detach(), pool_temp=pool_temp.detach(),
                       optimizer=opt.state_dict())
        barrier()

    warmup, start = cfg["warmup_steps"], time.time()
    last_epoch = min(int(cfg["epochs"]), int(cfg.get("stop_after_epoch", cfg["epochs"])))
    if rank == 0:
        write_json(out / "setup.json", dict(train_reports=sum(counts), steps_per_epoch=steps, epochs=cfg["epochs"],
                                            world=world, model=spec))
    kinds = {c: [VIEW_KIND.get(v["window"]["kind"], 0) for v in cases[c]["views"]] for c in train}
    size = lambda c: sum(v["n_tokens"] for v in cases[c]["views"])
    pscale = spec.get("position_scale", 1.)
    for epoch in range(epoch0, last_epoch):
        order = size_bucketed_order(train, size, steps, cfg["seed"], epoch, rank, cfg.get("bucket_size", 8))
        first, DP = (batch0 if epoch == epoch0 else 0), loader.depth
        for b_ in range(first, min(first + DP, len(order))):
            loader.submit(cases[order[b_]])
        for batch in range(first, steps):
            began = time.perf_counter()
            cid = order[batch]
            frac = lr_fraction(step, warmup, steps * horizon_epochs(cfg, step))
            for g in opt.param_groups:
                g["lr"] = g["base"] * frac
            opt.zero_grad(set_to_none=True)
            bag = CachedBag(loader.get(), device, cfg.get("gpu_bag_gib", 20.))
            if batch + DP < len(order):
                loader.submit(cases[order[batch + DP]])
            kd = torch.tensor(kinds[cid], device=device)

            def chunks():
                for y, pf, k in bag.chunks(block, cfg["crops_per_chunk"]):
                    yield torch.cat(((y - mean) / rms, pf.float() * pscale), 1), kd[k]

            r = alignment_step(chunks, S[cid], model=model, cfg=cfg, sentence_seed=cfg["seed"] * 7919 + step * 131 + rank,
                               world=world, device=device)
            step += 1
            del bag
            if rank == 0 and (step % 10 == 0 or step <= 20):
                r.update(epoch=epoch + 1, batch=batch + 1, batches=steps, step=step, lr_fraction=frac,
                         batch_seconds=time.perf_counter() - began, elapsed=time.time() - start)
                write_json(out / "progress.json", r)
                print(json.dumps(r), flush=True)
            stop = stopper.agreed(bool(a.max_steps and step >= a.max_steps), device)
            if step % 200 == 0 or stop:
                save(epoch, batch + 1)
            if step % 1000 == 0:
                save(epoch, batch + 1, f"step_{step:06d}.pt")
            if stop:
                loader.close()
                dist.destroy_process_group()
                return
        save(epoch + 1, 0)
        save(epoch + 1, 0, f"epoch_{epoch + 1:02d}.pt")
    if rank == 0:
        write_json(out / "COMPLETE.json", dict(epochs=last_epoch, step=step))
    loader.close()
    barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

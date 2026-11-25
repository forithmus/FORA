"""Train the chest X-ray finding classifier (Methods, "Chest X-ray").

Same recipe as the CT classifier (``train_ct.py``) on the X-ray encoder, online (no token cache):
  * frozen X-ray teacher blocks 1-23 on the 1,024-px tiles of all films of a report (at most ``max_films``), once per step;
  * trainable (block 24, norm) initialised from the exported X-ray sentence-alignment block (``init_block_from``,
    ``fora_align.export_block24``), view-kind
    embedding, head of ``fora.mil.model`` (every token routed); EMA 0.998 of every trainable parameter;
  * targets 1 = asserted or uncertain, 0 = denied or unmentioned (no masked entries), masked BCE / number of classes;
  * AdamW head lr 1e-3, route lr 1e-2, block lr 2e-5, warm-up 0.25 epoch then cosine over 6 epochs, clip 1.0;
  * one report per rank per step; reports are split over ranks by one fixed shuffle of all training reports.
Training films are augmented towards external acquisition styles before the content box (``augment``; validation never):
  gamma U(0.7, 1.4) + contrast stretch to a random percentile window (p 0.5); global histogram equalisation (p 0.3);
  downsampling so the long side is U(1024, 2048) px (p 0.5) with bilinear upsampling back to the original grid; 8-bit
  quantisation of a random percentile window (p 0.5, applied at the low resolution when downsampled); JPEG quality
  U{60..95} (p 0.3). Order: gamma/stretch -> equalisation -> downsample -> 8-bit -> upsample -> JPEG.
After every epoch raw and EMA weights are evaluated on the validation reports (macro AUROC / AUPRC over classes with
>= 10 positives and negatives); ``model_epoch_XX.pt`` and ``model_best_{raw,ema}.pt`` are written.

  torchrun --nproc-per-node 4 ... train_xray.py --config configs/xray.json --output runs/xray
"""
from __future__ import annotations

import argparse
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from common import (allreduce_gradients, bucketed_order, init_distributed, load_config, lr_fraction, save_npz,
                    save_torch, swap_ema, write_json)
from fora.encoder.cxr import PATCH_PX, CXRSplitEncoder
from fora.mil.model import auroc, backward, forward, masked_bce
from fora.mil.xray import build_xray_model, make_xray_factory, prepare_study, unit_film


# ------------------------------------------------------------------------------------------------------- augmentation
def _pct(x, q):
    return np.percentile(x[::4, ::4], q)


def _to8(x, lo, hi):
    return np.rint(np.clip((x - lo) / max(float(hi - lo), 1e-6), 0., 1.) * 255.).astype(np.uint8)


def augment(img: np.ndarray, rng: np.random.Generator, A: dict) -> np.ndarray:
    """Raw film -> augmented raw film (float32, same shape)."""
    from PIL import Image
    x = np.asarray(img, dtype=np.float32)
    u8 = False
    H, W = x.shape
    do = {k: bool(rng.random() < A["p_" + k]) for k in ("gamma", "histeq", "resample", "quant8", "jpeg")}
    if do["gamma"]:
        lo, hi = float(x.min()), float(x.max())
        n = np.power((x - lo) / max(hi - lo, 1e-6), float(rng.uniform(*A["gamma"])), dtype=np.float32)
        wl, wh = float(rng.uniform(0, A["stretch_lo_pct_max"])), float(rng.uniform(A["stretch_hi_pct_min"], 100))
        a, b = _pct(n, (wl, wh))
        x = np.clip((n - a) / max(float(b - a), 1e-6), 0., 1.).astype(np.float32)
    if do["histeq"]:
        B = 4096
        lo, hi = float(x.min()), float(x.max())
        idx = ((x - lo) * ((B - 1) / max(hi - lo, 1e-6))).astype(np.int32)
        cdf = np.cumsum(np.bincount(idx.ravel(), minlength=B)).astype(np.float64)
        cdf = (cdf - cdf[0]) / max(cdf[-1] - cdf[0], 1.)
        x = (cdf[idx] * 255.).astype(np.float32)
    small = None
    if do["resample"]:
        s = float(rng.uniform(*A["resample_long"])) / max(H, W)
        if s < 1:
            small = (max(16, round(H * s)), max(16, round(W * s)))
            x = F.interpolate(torch.from_numpy(np.ascontiguousarray(x))[None, None], size=small, mode="area")[0, 0].numpy()
    if do["quant8"]:
        wl, wh = float(rng.uniform(0, A["quant_lo_pct_max"])), float(rng.uniform(A["quant_hi_pct_min"], 100))
        lo, hi = _pct(x, (wl, wh))
        x = _to8(x, lo, hi).astype(np.float32)
        u8 = True
    if small is not None:
        x = F.interpolate(torch.from_numpy(np.ascontiguousarray(x))[None, None], size=(H, W), mode="bilinear",
                          align_corners=False)[0, 0].numpy()
        u8 = False
    if do["jpeg"]:
        q = int(rng.integers(A["jpeg_quality"][0], A["jpeg_quality"][1] + 1))
        b = x.astype(np.uint8) if u8 else _to8(x, float(x.min()), float(x.max()))
        buf = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(b), "L").save(buf, "JPEG", quality=q)
        buf.seek(0)
        x = np.asarray(Image.open(buf).convert("L"), dtype=np.float32)
    assert x.shape == (H, W)
    return x


# -------------------------------------------------------------------------------------------------------------- data
def read_raw(h5_root: str, case: dict, film) -> np.ndarray:
    """Raw 2-D film of an internal X-ray shard: <h5_root>/<h5>.h5 [patient][report][acc][sid]['data']."""
    import h5py
    with h5py.File(f"{h5_root}/{case['h5']}.h5", "r") as f:
        data = f[case["patient"]][case["report"]][film[0]][film[1]]["data"][()]
    return np.asarray(data[:, :, 0] if data.ndim == 3 else data, dtype=np.float32)


def prepare(case, cfg, h5_root, fallback=None, A=None, seed=None):
    """All films of a report (optionally augmented) -> prepared study; a report without a loadable film is replaced by
    ``fallback``."""
    rng = np.random.default_rng(seed) if A else None
    films = []
    for film in case["films"][:cfg.get("max_films", 8)]:
        try:
            img = read_raw(h5_root, case, film)
            if A:
                img = augment(img, rng, A)
            films.append((unit_film(img), int(film[2])))
        except Exception as e:
            print(json.dumps(dict(event="film_skip", case=case["id"], error=repr(e)[:200])), flush=True)
    if not films:
        if fallback is None or fallback is case:
            raise RuntimeError(f"no loadable film in {case['id']}")
        return prepare(fallback, cfg, h5_root, None, A, seed)
    return prepare_study(case["id"], films, cfg.get("tile", 1024), cfg.get("position_freqs", 4))


class ThreadLoader:
    """Prefetch prepared studies in threads; a load that fails or exceeds ``timeout`` seconds is retried synchronously."""

    def __init__(self, cfg, h5_root, fallback, depth=4, workers=3, timeout=900.):
        self.cfg, self.h5, self.fb, self.depth, self.timeout = cfg, h5_root, fallback, depth, timeout
        self.pool = ThreadPoolExecutor(workers)
        self.pending = []

    def submit(self, case, A=None, seed=None):
        self.pending.append((case, A, seed, self.pool.submit(prepare, case, self.cfg, self.h5, self.fb, A, seed)))

    def get(self):
        case, A, seed, fut = self.pending.pop(0)
        try:
            return fut.result(timeout=self.timeout)
        except Exception as e:
            print(json.dumps(dict(event="loader_retry", case=case.get("id"), error=repr(e)[:200])), flush=True)
            return prepare(case, self.cfg, self.h5, self.fb, A, seed)

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)


def read_cases(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def average_precision(y, s):
    o = np.argsort(-s, kind="mergesort")
    y = y[o]
    prec = np.cumsum(y) / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / max(1, y.sum()))


def metrics(Y, X, classes, min_pos):
    per, au, ap = {}, [], []
    for k, name in enumerate(classes):
        y = Y[:, k].astype(np.int64)
        yp = int(y.sum())
        e = dict(val_pos=yp, val_neg=len(y) - yp)
        if yp >= min_pos and len(y) - yp >= min_pos:
            e["auroc"], e["auprc"] = auroc(y, X[:, k]), average_precision(y, X[:, k])
            au.append(e["auroc"])
            ap.append(e["auprc"])
        per[name] = e
    return dict(n_reports=int(len(Y)), macro_auroc=float(np.mean(au)) if au else None,
                macro_auprc=float(np.mean(ap)) if ap else None, n_scored=len(au), per_class=per)


# -------------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--eval-limit", type=int, default=0, help="evaluate only this many validation reports per rank")
    a = ap.parse_args()
    m = load_config(a.config)
    cfg = m["training"]
    spec = cfg["model"]
    out = a.output
    out.mkdir(parents=True, exist_ok=True)
    rank, world, device, barrier = init_distributed()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(cfg["seed"])
    src = torch.load(m["normalization"], map_location=device, weights_only=False)
    mean, rms = src["mean"].to(device).float(), float(src["rms"])
    enc = CXRSplitEncoder(m["teacher_backbone"], torch.load(m["block24_state"], map_location="cpu", weights_only=False), device)
    block = enc.top
    Lz = np.load(m["labels"])
    classes = [str(c) for c in Lz["classes"]]
    C = len(classes)
    tgt = {**{str(i): y for i, y in zip(Lz["train_ids"], Lz["train_Y"])}, **{str(i): y for i, y in zip(Lz["val_ids"], Lz["val_Y"])}}
    model = build_xray_model(spec, C, block, device, 1024 + 6 * cfg.get("position_freqs", 4))
    loss_fn = masked_bce(C)
    allc = read_cases(m["cases_train"])
    perm = np.random.default_rng(cfg["seed"]).permutation(len(allc))
    train = [allc[i] for i in perm[rank::world]]
    val = read_cases(m["cases_val"])[rank::world]
    del allc
    if a.eval_limit:
        val = val[:a.eval_limit]
    cases = {c["id"]: c for c in train + val}
    train, val = [c["id"] for c in train], [c["id"] for c in val]
    counts = [None] * world
    dist.all_gather_object(counts, len(train))
    steps = max(counts)
    h5 = m["h5_root"]
    loader = ThreadLoader(cfg, h5, cases[train[0]], cfg.get("prefetch_depth", 4), cfg.get("loader_workers", 3))
    aug = cfg["augment"]
    aseed = lambda ep, b: (cfg["seed"], 7741, ep * steps + b, rank)
    fb, chunk = cfg.get("frozen_batch", 8), cfg.get("tiles_per_chunk", 8)

    statepath = out / "latest.pt"
    epoch0 = batch0 = step = 0
    best = {}
    if statepath.exists():
        s = torch.load(statepath, map_location=device, weights_only=False)
        if not (s["training"] == cfg and s["world"] == world and s["classes"] == classes):
            raise SystemExit("latest.pt is from a different recipe, world size or class list")
        model["head"].load_state_dict(s["head"])
        block.load_state_dict(s["block"])
        model["view"].load_state_dict(s["view"])
        model["optimizer"].load_state_dict(s["optimizer"])
        if model["ema"] is not None:
            for e, v in zip(model["ema"], s["ema"]):
                e.copy_(v)
        epoch0, batch0, step, best = s["epoch"], s["batch"], s["step"], s.get("best", {})
        del s
    else:
        s = torch.load(m["init_block_from"], map_location=device, weights_only=False)
        block.load_state_dict(s["block"])
        if int(s.get("teacher_step", enc.step)) != int(enc.step):
            raise SystemExit("the block-24 checkpoint was trained on a different X-ray teacher")
        if model["ema"] is not None:
            for e, p in zip(model["ema"], model["params"]):
                e.copy_(p.detach())
        del s

    def snapshot():
        return dict(head=model["head"].state_dict(), block=block.state_dict(), view=model["view"].state_dict(),
                    ema=model["ema"], classes=classes, step=step, teacher_step=enc.step, training=cfg)

    def save(epoch, batch):
        if rank == 0:
            save_torch(statepath, dict(snapshot(), world=world, epoch=epoch, batch=batch,
                                       optimizer=model["optimizer"].state_dict(), best=best))
        barrier()

    def evaluate(epoch):
        preds = {"raw": {}, "ema": {}} if model["ema"] is not None else {"raw": {}}
        model["head"].eval()
        el = ThreadLoader(cfg, h5, None, cfg.get("prefetch_depth", 4), cfg.get("loader_workers", 3))
        for c in val[:el.depth]:
            el.submit(cases[c])
        for i, cid in enumerate(val):
            try:
                bag = el.get()
            except Exception as e:
                print(json.dumps(dict(event="eval_skip", case=cid, error=repr(e)[:200])), flush=True)
                bag = None
            if i + el.depth < len(val):
                el.submit(cases[val[i + el.depth]])
            if bag is None:
                continue
            h23 = enc.h23(bag["tiles"], fb)
            f = make_xray_factory(enc, h23, bag, model, mean, rms, chunk, device, C)
            with torch.no_grad():
                preds["raw"][cid] = forward(model["head"], f)[0].float().cpu().numpy()
                if model["ema"] is not None:
                    swap_ema(model["params"], model["ema"])
                    try:
                        preds["ema"][cid] = forward(model["head"], f)[0].float().cpu().numpy()
                    finally:
                        swap_ema(model["params"], model["ema"])
            del bag, h23
        el.close()
        model["head"].train()
        gathered = [None] * world
        dist.all_gather_object(gathered, preds)
        if rank == 0:
            summary, arrays, new_best = {}, {}, []
            for var in preds:
                merged = {c: v for p in gathered for c, v in p[var].items()}
                ids = sorted(merged)
                X = np.stack([merged[c] for c in ids]).astype(np.float64)
                Y = np.stack([tgt[c] for c in ids])
                summary[var] = metrics(Y, X, classes, cfg.get("min_eval_pos", 10))
                arrays[f"{var}__logits"] = X.astype(np.float32)
                arrays[f"{var}__ids"] = np.array(ids)
                arrays[f"{var}__targets"] = Y
                sc = summary[var]["macro_auroc"]
                if sc is not None and sc > best.get(var, {}).get("macro_auroc", -1.):
                    best[var] = dict(epoch=epoch, macro_auroc=sc, macro_auprc=summary[var]["macro_auprc"])
                    new_best.append(var)
            res = out / "results.json"
            hist = json.loads(res.read_text())["by_epoch"] if res.exists() else []
            hist = [h for h in hist if h["epoch"] != epoch] + [dict(epoch=epoch, step=step, **summary)]
            write_json(res, dict(classes=classes, best=best, by_epoch=hist))
            mdl = dict(snapshot(), epoch=epoch)
            save_npz(out / f"predictions_epoch_{epoch:02d}.npz", **arrays)
            save_torch(out / f"model_epoch_{epoch:02d}.pt", mdl)
            for var in new_best:
                save_npz(out / f"predictions_best_{var}.npz", **arrays)
                save_torch(out / f"model_best_{var}.pt", dict(mdl, variant=var))
            print(json.dumps(dict(event="epoch_complete", epoch=epoch, new_best=new_best,
                                  **{v: dict(macro_auroc=d["macro_auroc"], macro_auprc=d["macro_auprc"]) for v, d in summary.items()})), flush=True)
        barrier()

    E = int(cfg["epochs"])
    total, warmup = steps * E, int(steps * cfg["warmup_epochs"])
    if rank == 0:
        write_json(out / "setup.json", dict(classes=C, train_reports=sum(counts), steps_per_epoch=steps, epochs=E, world=world))
    size = lambda c: sum(-(-f[3] // PATCH_PX) * -(-f[4] // PATCH_PX) for f in cases[c]["films"])
    start = time.time()
    for epoch in range(epoch0, E):
        order = bucketed_order(train, size, cfg["seed"], epoch, rank, cfg["bucket_size"], min(counts))
        first = batch0 if epoch == epoch0 else 0
        DP = loader.depth
        loader.pending = []
        for b in range(first, min(first + DP, len(order))):
            loader.submit(cases[order[b]], aug, aseed(epoch, b))
        for batch in range(first, steps):
            active = batch < len(order)
            count = sum(batch < n for n in counts)
            frac = lr_fraction(step, warmup, total)
            model["optimizer"].zero_grad(set_to_none=True)
            for g in model["optimizer"].param_groups:
                g["lr"] = g["base"] * frac
            row_loss = 0.
            if active:
                bag = loader.get()
                if batch + DP < len(order):
                    loader.submit(cases[order[batch + DP]], aug, aseed(epoch, batch + DP))
                t = torch.from_numpy(tgt[bag["id"]].astype(np.float32)).to(device)
                h23 = enc.h23(bag["tiles"], fb)
                f = make_xray_factory(enc, h23, bag, model, mean, rms, chunk, device, C,
                                      seed=(cfg["seed"] * 1000003 + step) * 64 + rank * 7919)
                loss, _ = backward(model["head"], f, t, loss_scale=world / count, loss_fn=loss_fn)
                row_loss = float(loss) * count / world
                del bag, h23, f
            allreduce_gradients(model["params"], world)
            norm = float(torch.nn.utils.clip_grad_norm_(model["params"], 1., error_if_nonfinite=True))
            model["optimizer"].step()
            if model["ema"] is not None:
                with torch.no_grad():
                    torch._foreach_lerp_(model["ema"], [p.detach() for p in model["params"]], 1 - spec["ema"])
            step += 1
            meter = torch.tensor([row_loss, float(active)], device=device, dtype=torch.float64)
            dist.all_reduce(meter)
            if rank == 0 and (step % 10 == 0 or step <= 20):
                r = dict(epoch=epoch + 1, batch=batch + 1, batches=steps, step=step, lr_fraction=frac,
                         loss=float(meter[0] / max(1., float(meter[1]))), gradient_norm=norm, elapsed=time.time() - start)
                write_json(out / "progress.json", r)
                print(json.dumps(r), flush=True)
            if step % 200 == 0:
                save(epoch, batch + 1)
        evaluate(epoch + 1)
        save(epoch + 1, 0)
    if rank == 0:
        write_json(out / "COMPLETE.json", dict(epochs=E, step=step, best=best, finished=time.time()))
    loader.close()
    barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

"""Export CT evidence, foci and evidence images for requested (report, finding) pairs (Methods, "Finding localization";
input of the report writer's evidence tokens and evidence images).

Per report, block 24 + head run once over all tokens of all views (``fora.mil.evidence.study_evidence``); for every
requested finding the script stores the classifier outputs, the pooled head input, evidence-weighted region shares,
left/right share, cranio-caudal position, the evidence peak, the foci decomposition (``fora.mil.evidence.foci``) and,
with ``--render 1``, up to six JPEG evidence images (axial, coronal, sagittal at the peak + one axial per further site with
>= 5 % of the evidence; ``fora.mil.render``).

Targets JSON: {"targets": {case id: [class index, ...]}, "case2rep": {case id: report uid}}.
Plan reports JSONL (for the renders): one line per report with ``id``, ``h5``, ``patient``, ``report`` and ``series``
(list of {acc, sid, sub, frame}); views refer to series by index.

Output: ``<out>/extent_r<shard>_p0_c<chunk>.npz`` every ``--chunk`` reports with one row per (report, finding):
  pair_case, pair_rep, cls, logit, prob, feat [n, 1048] f16, reg3 / frac3 (top-3 specific regions and shares), regall [n, 63]
  f16, left, right, zfrac, peak (z, y, x), view, fo_<key> for every key of ``fora.mil.evidence.FOCI_KEYS``,
  img_blob uint8 + img_off [n, 7] (byte offsets of the six images); plus ids and classes.
Reports already present in the shard's chunks are skipped (resume).

  export_evidence_ct.py --config configs/ct_stage3.json --ckpt runs/ct_stage3/model_epoch_04.pt --targets targets.json \
      --reports plan_reports.jsonl --h5-root CT_H5 --out evidence/ --shard 0 --shards 128 --crops 32 --chunk 200 --render 1
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from common import load_config
from fora.encoder.cache import local_view_file
from fora.mil import render as RF
from fora.mil.bag import RoutedBag, load_study
from fora.mil.evidence import FOCI_KEYS, foci, study_evidence, view_info
from fora.mil.model import class_bits, load_classifier


def load_ct(h5_root, rep, s) -> np.ndarray:
    """Resampled, body-cropped series volume (S, R, A) as int16 HU clipped to [-1024, 3071]."""
    import h5py
    from fora.geometry.series import read_series
    from fora.geometry.tiling import body_frame, resample_series
    with h5py.File(f"{h5_root}/{rep['h5']}.h5", "r") as f:
        data, affine = read_series(f, rep["patient"], rep["report"], s["acc"], s["sid"], s.get("sub") or "", s.get("frame"))
    vol, _ = resample_series(data, affine)
    y0, y1, x0, x1, _, _ = body_frame(vol)
    return np.clip(vol[:, y0:y1, x0:x1], -1024, 3071).astype(np.int16)


def render_pairs(res, vols, load_more, class_names):
    """Up to six JPEG images per requested finding (empty bytes where nothing was rendered); ``vols`` holds the prefetched
    series volumes, ``load_more`` loads any other series."""
    def load_volume(sidx):
        if sidx not in vols:
            vols[sidx] = load_more(sidx)
        if vols[sidx] is None:
            raise RuntimeError(f"series {sidx} could not be loaded")
        return vols[sidx]
    return RF.study_finding_images(res["E"], res["vinfo"], res["foci"], class_names, load_volume,
                                   log=lambda d: print(json.dumps(d), flush=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--targets", type=Path, required=True)
    ap.add_argument("--reports", type=Path, help="plan reports JSONL (required with --render 1)")
    ap.add_argument("--h5-root", help="CT H5 shard directory (required with --render 1)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--crops", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=200)
    ap.add_argument("--render", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    m = load_config(a.config)
    cfg = m["training"]
    dev = torch.device(a.device)
    norm = torch.load(m["normalization"], map_location=dev, weights_only=False)
    mean, rms = norm["mean"].to(dev).float(), float(norm["rms"])
    block, head, classes = load_classifier(m, a.ckpt, dev)
    cb = class_bits(json.loads(Path(m["class_region_bits"]).read_text()), classes, dev)
    TG = json.loads(a.targets.read_text())
    tg, c2r = TG["targets"], TG["case2rep"]
    mine = sorted(tg)[a.shard::a.shards]
    done, chunk_no = set(), 0
    for f in sorted(glob.glob(str(a.out / f"extent_r{a.shard:03d}_p0_c*.npz"))):
        with np.load(f) as z:
            done.update(z["ids"].tolist())
        chunk_no = max(chunk_no, int(f.rsplit("_c", 1)[1][:4]) + 1)
    want = {c for c in mine if c not in done}
    cases = {}
    with open(m["cases"]) as f:
        for line in f:
            i = line.find('"id": "')
            cid = line[i + 7:line.index('"', i + 7)]
            if cid in want:
                cases[cid] = json.loads(line)
    todo = [c for c in mine if c in cases]
    reps = {}
    if a.render:
        with open(a.reports) as f:
            for line in f:
                r = json.loads(line)
                if r["id"] in want:
                    reps[r["id"]] = r

    def prep(cid):
        case = cases[cid]
        views = load_study(case, m["cache_root"], m["region_root"], cfg.get("position_freqs", 4))
        vinfo = [view_info(local_view_file(vw, m["cache_root"]), vw, v["bits"]) for vw, v in zip(case["views"], views)]
        return views, vinfo

    def load1(cid, sidx):
        try:
            return load_ct(a.h5_root, reps[cid], reps[cid]["series"][sidx])
        except Exception as ex:
            print(json.dumps(dict(case=cid, ct_error=str(ex)[:200])), flush=True)
            return None

    def ctload(cid):          # prefetch the series with the most views; others are loaded on demand
        cnt = {}
        for vw in cases[cid]["views"]:
            cnt[vw["series"]] = cnt.get(vw["series"], 0) + 1
        s = sorted(cnt, key=lambda k: (-cnt[k], k))[0]
        return {s: load1(cid, s)}

    buf = {k: [] for k in ("ids", "pair_case", "pair_rep", "cls", "logit", "prob", "feat", "reg3", "frac3", "regall", "left",
                           "right", "zfrac", "peak", "view", "img", *("fo_" + k for k in FOCI_KEYS))}

    def flush():
        nonlocal chunk_no
        if not buf["ids"]:
            return
        z = {k: (np.concatenate(v) if k == "feat" else np.stack(v) if k.startswith("fo_") else np.array(v))
             for k, v in buf.items() if k != "img" and v}
        z["feat"] = z["feat"].astype(np.float16)
        z["regall"] = z["regall"].astype(np.float16)
        if a.render:
            imgs = buf["img"]
            off = np.zeros((len(imgs), 7), np.int64)
            o = 0
            for i, t in enumerate(imgs):
                off[i, 0] = o
                for j, b in enumerate(t):
                    o += len(b)
                    off[i, j + 1] = o
            z["img_blob"] = np.frombuffer(b"".join(b for t in imgs for b in t), np.uint8)
            z["img_off"] = off
        name = f"extent_r{a.shard:03d}_p0_c{chunk_no:04d}.npz"
        tmp = a.out / f".{name}.tmp.npz"
        np.savez(tmp, classes=np.array(classes), **z)
        os.replace(tmp, a.out / name)
        chunk_no += 1
        for v in buf.values():
            v.clear()

    pool = ThreadPoolExecutor(2)
    fut = {c: pool.submit(prep, c) for c in todo[:2]}
    ctpool = ThreadPoolExecutor(3)
    ctfut = {c: ctpool.submit(ctload, c) for c in todo[:3]} if a.render else {}
    t0 = time.time()
    for n_, cid in enumerate(todo):
        views, vinfo = fut.pop(cid).result()
        if n_ + 2 < len(todo):
            fut[todo[n_ + 2]] = pool.submit(prep, todo[n_ + 2])
        if a.render and n_ + 3 < len(todo):
            ctfut[todo[n_ + 3]] = ctpool.submit(ctload, todo[n_ + 3])
        clist = tg[cid]
        names = [classes[c] for c in clist]
        cl = torch.tensor(clist, device=dev)
        res = None
        for crops in (a.crops, 32, 8):
            try:
                bag = RoutedBag(views, dev, cb, cfg.get("min_route_tokens", 8), cfg.get("gpu_bag_gib", 20.))
                res = study_evidence(block, head, bag, vinfo, mean, rms, cl, crops)
                break
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
        if res is None:
            print(json.dumps(dict(case=cid, skipped="oom")), flush=True)
            ctfut.pop(cid, None)
            continue
        res["vinfo"] = vinfo
        res["foci"] = foci(res["E"], vinfo, names, res["XA"])
        tl = lambda t: t.float().cpu().numpy().tolist()
        buf["ids"].append(cid)
        buf["pair_case"] += [cid] * len(clist)
        buf["pair_rep"] += [c2r[cid]] * len(clist)
        buf["cls"] += clist
        buf["logit"] += tl(res["logit"])
        buf["prob"] += tl(res["prob"])
        buf["feat"].append(res["feat"].half().cpu().numpy())
        buf["reg3"] += res["ri"].tolist()
        buf["frac3"] += tl(res["fr"])
        buf["regall"] += tl(res["reg"])
        buf["left"] += tl(res["side"][:, 0])
        buf["right"] += tl(res["side"][:, 1])
        buf["zfrac"] += tl(res["zfrac"])
        buf["peak"] += res["peak"][:, 1:].tolist()
        buf["view"] += res["peak"][:, 0].tolist()
        for R in res["foci"]:
            for k in FOCI_KEYS:
                buf["fo_" + k].append(np.asarray(R[k]))
        if a.render:
            buf["img"] += render_pairs(res, ctfut.pop(cid).result(), lambda s: load1(cid, s), names)
        del res
        if n_ % 25 == 0:
            print(json.dumps(dict(shard=a.shard, done=n_ + 1, of=len(todo), sec=round(time.time() - t0))), flush=True)
        if len(buf["ids"]) >= a.chunk:
            flush()
    flush()
    pool.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    main()

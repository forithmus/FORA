"""Normal-tissue reference of the X-ray evidence (Methods, "Finding localization", chest X-ray).

For every class c: b_c = token-count-weighted mean token logit sL = l_t,c * s_c over 2,000 random training study bags
(seed 0) that are negative for c. ``fora.mil.evidence.xray_token_evidence`` subtracts b_c before taking the positive
part of the attention-weighted evidence.

  xray_evidence_reference.py run   --config configs/xray.json --model runs/xray/model_best_raw.pt --out ref/ --shard 0 --shards 8
  xray_evidence_reference.py merge --config configs/xray.json --out ref/ --json localization_reference_bc.json
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import torch

from common import load_config, save_npz, write_json
from fora.mil.xray import make_xray_factory
from predict_xray import load_xray_classifier
from train_xray import prepare, read_cases

N_STUDIES = 2000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("run", "merge"))
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--model", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    m = load_config(a.config)
    cfg = m["training"]
    if a.mode == "merge":
        Z = [dict(np.load(f)) for f in sorted(glob.glob(str(a.out / "ref_s*.npz")))]
        ids = np.concatenate([z["ids"] for z in Z])
        mean, n = np.concatenate([z["sl_mean"] for z in Z]).astype(np.float64), np.concatenate([z["ntok"] for z in Z]).astype(np.float64)
        lab = np.load(m["labels"])
        row = {k: i for i, k in enumerate(lab["train_ids"])}
        Y = lab["train_Y"][[row[k] for k in ids]]
        classes = [str(c) for c in lab["classes"]]
        b = {}
        for j, c in enumerate(classes):
            neg = Y[:, j] == 0
            b[c] = float((mean[neg, j] * n[neg]).sum() / n[neg].sum())
        write_json(a.json, dict(rule=f"b_c = token-count-weighted mean token logit (token_logit x output_scale) over "
                                     f"{len(ids)} random training study bags negative for c", b_mean=b))
        return
    dev = torch.device(a.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    src = torch.load(m["normalization"], map_location=dev, weights_only=False)
    mu, rms = src["mean"].to(dev).float(), float(src["rms"])
    enc, model, classes = load_xray_classifier(m, a.model, dev)
    head = model["head"]
    C = len(classes)
    allc = torch.arange(C, device=dev)
    cases = read_cases(m["cases_train"])
    sel = np.sort(np.random.default_rng(0).choice(len(cases), N_STUDIES, replace=False))
    ids, means, ntok = [], [], []
    for j in sel[a.shard::a.shards]:
        c = cases[j]
        try:
            bag = prepare(c, cfg, m["h5_root"])
        except Exception as e:
            print(f"skip {c['id']}: {e!r}", flush=True)
            continue
        h23 = enc.h23(bag["tiles"], cfg.get("frozen_batch", 8))
        with torch.no_grad():
            f = make_xray_factory(enc, h23, bag, model, mu, rms, cfg.get("tiles_per_chunk", 8), dev, C)
            sl = torch.cat([head._chunk_outputs(x, allc)[1].float() for x, _ in f()]) * head.output_scale.float()[None]
        ids.append(c["id"])
        means.append(sl.mean(0).cpu().numpy())
        ntok.append(len(sl))
    a.out.mkdir(parents=True, exist_ok=True)
    save_npz(a.out / f"ref_s{a.shard:04d}.npz", ids=np.array(ids), sl_mean=np.array(means, np.float32).reshape(-1, C),
             ntok=np.array(ntok, np.int64))


if __name__ == "__main__":
    main()

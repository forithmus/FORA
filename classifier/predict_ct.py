"""Study-bag logits of the CT classifier for every report (input of calibration and of the rule-A cut-offs).

Per report, block 24 + head run once over all tokens of all views (soft routing, raw weights, no token or feature drop),
exactly the evaluation path of ``train_ct.py``.

  predict_ct.py run   --config configs/ct_stage3.json --ckpt runs/ct_stage3/model_epoch_04.pt --out preds/ --shard 0 --shards 64
  predict_ct.py merge --config configs/ct_stage3.json --out preds/ --dense dense_predictions.npz

Shard files ``logits_s<shard>.npz``: ids, logits [n, C] float32, classes. The merged file holds ``fora_id``,
``report_uid``, ``logit`` [N, C] float32, ``label`` (class names) and ``class_index``.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

from common import load_config, save_npz
from fora.mil.bag import RoutedBag, load_study, make_factory
from fora.mil.model import class_bits, forward, load_classifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("run", "merge"))
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--crops", type=int, default=96)
    ap.add_argument("--dense", type=Path)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    m = load_config(a.config)
    cfg = m["training"]
    if a.mode == "merge":
        ids, X, classes = [], [], None
        for f in sorted(glob.glob(str(a.out / "logits_s*.npz"))):
            with np.load(f) as z:
                ids += [str(i) for i in z["ids"]]
                X.append(z["logits"])
                classes = [str(c) for c in z["classes"]]
        report = {}
        with open(m["cases"]) as f:
            for line in f:
                c = json.loads(line)
                report[c["id"]] = c["report"]
        save_npz(a.dense, fora_id=np.array(ids), report_uid=np.array([report[i] for i in ids]),
                 logit=np.concatenate(X).astype(np.float32), label=np.array(classes), class_index=np.arange(len(classes)))
        return
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device(a.device)
    norm = torch.load(m["normalization"], map_location=dev, weights_only=False)
    mean, rms = norm["mean"].to(dev).float(), float(norm["rms"])
    block, head, classes = load_classifier(m, a.ckpt, dev)
    cb = class_bits(json.loads(Path(m["class_region_bits"]).read_text()), classes, dev)
    with open(m["cases"]) as f:
        cases = [json.loads(line) for line in f][a.shard::a.shards]
    a.out.mkdir(parents=True, exist_ok=True)
    ids, X = [], []
    for case in cases:
        bag = RoutedBag(load_study(case, m["cache_root"], m["region_root"], cfg.get("position_freqs", 4)), dev, cb,
                        cfg.get("min_route_tokens", 8), cfg.get("gpu_bag_gib", 20.))
        X.append(forward(head, make_factory(bag, block, cfg["model"], mean, rms, a.crops))[0].float().cpu().numpy())
        ids.append(case["id"])
        del bag
    save_npz(a.out / f"logits_s{a.shard:04d}.npz", ids=np.array(ids), logits=np.stack(X).astype(np.float32)
             if X else np.zeros((0, len(classes)), np.float32), classes=np.array(classes))


if __name__ == "__main__":
    main()

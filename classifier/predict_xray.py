"""Study-bag logits of the X-ray classifier (all films of a report, no augmentation) for calibration.

  predict_xray.py run   --config configs/xray.json --model runs/xray/model_best_raw.pt --cases cases_train.jsonl --out preds_train/ --shard 0 --shards 16
  predict_xray.py merge --out preds_train/ --npz preds_train.npz

Output npz: ids, logits [n, C] float32, classes.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import torch

from common import load_config, save_npz
from fora.encoder.cxr import CXRSplitEncoder
from fora.mil.model import forward
from fora.mil.xray import build_xray_model, make_xray_factory
from train_xray import prepare, read_cases


def load_xray_classifier(m, model_path, device):
    cfg = m["training"]
    enc = CXRSplitEncoder(m["teacher_backbone"], torch.load(m["block24_state"], map_location="cpu", weights_only=False), device)
    s = torch.load(model_path, map_location=device, weights_only=False)
    classes = list(s["classes"])
    if int(s["teacher_step"]) != int(enc.step):
        raise SystemExit("checkpoint teacher step differs from the exported X-ray teacher")
    model = build_xray_model(cfg["model"], len(classes), enc.top, device, 1024 + 6 * cfg.get("position_freqs", 4))
    model["head"].load_state_dict(s["head"])
    enc.top.load_state_dict(s["block"])
    model["view"].load_state_dict(s["view"])
    model["head"].eval()
    return enc, model, classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("run", "merge"))
    ap.add_argument("--config", type=Path)
    ap.add_argument("--model", type=Path)
    ap.add_argument("--cases", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--npz", type=Path)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.mode == "merge":
        Z = [dict(np.load(f)) for f in sorted(glob.glob(str(a.out / "logits_s*.npz")))]
        ids = np.concatenate([z["ids"] for z in Z])
        o = np.argsort(ids, kind="stable")
        save_npz(a.npz, ids=ids[o], logits=np.concatenate([z["logits"] for z in Z])[o], classes=Z[0]["classes"])
        return
    m = load_config(a.config)
    cfg = m["training"]
    dev = torch.device(a.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    src = torch.load(m["normalization"], map_location=dev, weights_only=False)
    mean, rms = src["mean"].to(dev).float(), float(src["rms"])
    enc, model, classes = load_xray_classifier(m, a.model, dev)
    ids, X = [], []
    for c in read_cases(a.cases)[a.shard::a.shards]:
        try:
            bag = prepare(c, cfg, m["h5_root"])
        except Exception as e:
            print(f"skip {c['id']}: {e!r}", flush=True)
            continue
        h23 = enc.h23(bag["tiles"], cfg.get("frozen_batch", 8))
        with torch.no_grad():
            f = make_xray_factory(enc, h23, bag, model, mean, rms, cfg.get("tiles_per_chunk", 8), dev, len(classes))
            X.append(forward(model["head"], f)[0].float().cpu().numpy())
        ids.append(c["id"])
    a.out.mkdir(parents=True, exist_ok=True)
    save_npz(a.out / f"logits_s{a.shard:04d}.npz", ids=np.array(ids), logits=np.stack(X).astype(np.float32)
             if X else np.zeros((0, len(classes)), np.float32), classes=np.array(classes))


if __name__ == "__main__":
    main()

"""Fit the calibration used by report generation (Methods, "Finding classifier", calibration and plan tiers).

Per finding, on the TRAINING study-bag logits only:
  * Platt curve p = sigmoid(a * logit + b), Newton with backtracking, ridge toward (a, b) = (1, 0) with l2 = 1.0,
    slope floor 0.05;
  * Youden cut-off on the raw sigmoid (positive if score >= cut; 0.5 for a finding without training positives);
  * probable tier = calibrated p >= 0.5, never below the Youden cut; possible tier = the Youden cut, never above the
    probable cut; both stored on the raw sigmoid and on the calibrated scale;
  * base rate = training positives / training studies (keys ``class_val_positive`` / ``n_calibration_studies``, read by
    the report-generation plan builder).
Validation studies are only used for the printed tier statistics.

CT targets: labels npz state 1/2 -> 1, 0/3 -> 0, -1 masked. X-ray targets: labels npz train_Y / val_Y (no masked entries).

  fit_calibration.py ct   --dense dense_predictions.npz --labels labels.npz --split split.json --out calibration_ct.json
  fit_calibration.py xray --preds-train preds_train.npz --preds-val preds_val.npz --labels labels.npz --out calibration_xray.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from common import write_json


def label_order_sha256(labels) -> str:
    return hashlib.sha256("\n".join(map(str, labels)).encode("utf-8")).hexdigest()


def fit_platt(logits, targets, mask, l2=1.0, iterations=50):
    """Per-class Platt (vectorised Newton with backtracking), ridge toward (1, 0); returns (max(a, 0.05), b)."""
    x = logits.astype(np.float64)
    y = targets.astype(np.float64)
    w = mask.astype(np.float64)
    C = x.shape[1]
    a = np.ones(C)
    b = np.zeros(C)

    def objective(a_, b_):
        z = x * a_ + b_
        loss = (np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))) * w
        return loss.sum(0) + 0.5 * l2 * ((a_ - 1.0) ** 2 + b_ ** 2)

    current = objective(a, b)
    for _ in range(iterations):
        z = x * a + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))
        r = (p - y) * w
        s = p * (1 - p) * w
        ga = (r * x).sum(0) + l2 * (a - 1.0)
        gb = r.sum(0) + l2 * b
        haa = (s * x * x).sum(0) + l2
        hab = (s * x).sum(0)
        hbb = s.sum(0) + l2
        det = haa * hbb - hab * hab
        det = np.where(np.abs(det) < 1e-12, 1e-12, det)
        da = (hbb * ga - hab * gb) / det
        db = (haa * gb - hab * ga) / det
        step = np.ones(C)
        accepted = np.zeros(C, bool)
        for _h in range(30):
            trial = objective(a - step * da, b - step * db)
            better = (trial <= current) & ~accepted
            a = np.where(better, a - step * da, a)
            b = np.where(better, b - step * db, b)
            current = np.where(better, trial, current)
            accepted |= better
            if accepted.all():
                break
            step = np.where(accepted, step, step * 0.5)
        if max(np.abs(step * da).max(), np.abs(step * db).max()) < 1e-7:
            break
    return np.maximum(a, 0.05), b


def youden(s, y) -> float:
    """Cut-off maximising sensitivity + specificity - 1 (positive if score >= cut), at distinct-score boundaries."""
    o = np.argsort(-s, kind="mergesort")
    s, y = s[o], y[o]
    tp, fp = np.cumsum(y), np.cumsum(1 - y)
    last = np.r_[s[1:] != s[:-1], True]
    j = tp / y.sum() - fp / (len(y) - y.sum())
    j[~last] = -np.inf
    return float(s[int(np.argmax(j))])


def logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def tiers(slope, intercept, yd):
    """Raw and calibrated probable / possible cut-offs from the Platt curves and the Youden cuts."""
    raw_p50 = 1 / (1 + np.exp(intercept / np.maximum(slope, 1e-6)))
    raw_plan = np.maximum(raw_p50, yd)
    raw_poss = np.minimum(raw_plan, yd)
    plan_cal = 1 / (1 + np.exp(-(slope * logit(raw_plan) + intercept)))
    poss_cal = np.minimum(1 / (1 + np.exp(-(slope * logit(raw_poss) + intercept))), plan_cal)
    return raw_plan, raw_poss, plan_cal, poss_cal


def tier_stats(p, raw_plan, raw_poss, target, mask):
    """Per-study tier sizes and micro precision / recall against the report labels (masked entries excluded)."""
    conf = p >= raw_plan
    poss = (p >= raw_poss) & ~conf
    y = target & mask
    out = dict(n_studies=int(p.shape[0]), label_positives_per_study=float(y.sum(1).mean()))
    for k, sel in (("probable", conf), ("possible", poss), ("union", conf | poss)):
        n = sel.sum(1)
        tp = int((sel & y).sum())
        out[k] = dict(mean_per_study=float(n.mean()), p95=float(np.percentile(n, 95)),
                      micro_precision=tp / max(int((sel & mask).sum()), 1), micro_recall=tp / max(int(y.sum()), 1))
    return out


def calibration_doc(classes, slope, intercept, yd, n_pos, n_studies, target_policy, sources, summary):
    raw_plan, raw_poss, plan_cal, poss_cal = tiers(slope, intercept, yd)
    return {
        "format": "fora_mil_train_platt_tiers", "format_version": 1, **sources,
        "source_split": "train", "n_calibration_studies": int(n_studies), "calibration_report_uids": [],
        "exclude_from_report_training": False, "target_policy": target_policy,
        "platt": {"l2_toward_identity": 1.0, "min_slope": 0.05},
        "threshold_rule": {"plan": "platt50_train", "possible": "youden_train"},
        "labels": classes, "label_order_sha256": label_order_sha256(classes), "canonical_plan_labels": classes,
        "class_val_positive": np.asarray(n_pos).astype(int).tolist(),
        "class_slope": slope.tolist(), "class_intercept": intercept.tolist(),
        "class_raw_youden_threshold": yd.tolist(),
        "class_raw_plan_threshold": raw_plan.tolist(), "class_raw_possible_threshold": raw_poss.tolist(),
        "class_plan_threshold": plan_cal.tolist(), "class_possible_threshold": poss_cal.tolist(),
        "summary": summary,
    }


def fit_ct(a):
    d = np.load(a.dense, allow_pickle=False)
    classes = [str(c) for c in d["label"]]
    C = len(classes)
    fid = d["fora_id"].astype(str)
    X = np.asarray(d["logit"], np.float32)
    split = json.loads(a.split.read_text())
    Z = np.load(a.labels, allow_pickle=False)
    node_col = {n: j for j, n in enumerate(Z["nodes"].tolist())}
    cols = np.array([node_col[c] for c in classes])
    case_row = {c: j for j, c in enumerate(Z["cases"].tolist())}
    rows = {s: np.array([i for i, f in enumerate(fid.tolist()) if split.get(f) == s and f in case_row]) for s in ("train", "val")}
    state = {s: Z["state"][[case_row[f] for f in fid[k]]][:, cols] for s, k in rows.items()}
    Xs = {s: X[k] for s, k in rows.items()}
    tgt = {s: np.isin(state[s], (1, 2)) for s in rows}
    msk = {s: state[s] >= 0 for s in rows}
    slope, intercept = np.ones(C), np.zeros(C)
    for c0 in range(0, C, 128):          # class chunks bound the memory of the Newton fit
        sl = slice(c0, min(C, c0 + 128))
        slope[sl], intercept[sl] = fit_platt(Xs["train"][:, sl], tgt["train"][:, sl], msk["train"][:, sl])
    ptr = 1 / (1 + np.exp(-Xs["train"].astype(np.float64)))
    yd, n_pos = np.empty(C), np.zeros(C, int)
    for j in range(C):
        m = msk["train"][:, j]
        yy = tgt["train"][m, j].astype(np.float64)
        n_pos[j] = int(yy.sum())
        yd[j] = youden(ptr[m, j], yy) if 0 < yy.sum() < len(yy) else 0.5
    del ptr
    raw_plan, raw_poss, _, _ = tiers(slope, intercept, yd)
    pva = 1 / (1 + np.exp(-Xs["val"].astype(np.float64)))
    summary = dict(val_tiers=tier_stats(pva, raw_plan, raw_poss, tgt["val"], msk["val"]),
                   n_classes_no_train_positive=int((n_pos == 0).sum()))
    return calibration_doc(classes, slope, intercept, yd, n_pos, len(rows["train"]),
                           "state asserted(1) or uncertain(2) = 1; denied(0) or unmentioned(3) = 0; -1 masked",
                           {"source_predictions": str(a.dense)}, summary)


def fit_xray(a):
    L = np.load(a.labels, allow_pickle=False)
    classes = [str(c) for c in L["classes"]]
    C = len(classes)

    def aligned(path, want):
        z = np.load(path, allow_pickle=False)
        if [str(c) for c in z["classes"]] != classes:
            raise SystemExit(f"{path}: class order differs from the labels")
        pos = {str(i): k for k, i in enumerate(z["ids"])}
        missing = [i for i in want if i not in pos]
        if missing:
            raise SystemExit(f"{path}: {len(missing)} label rows without predictions")
        return np.asarray(z["logits"], np.float32)[[pos[i] for i in want]]
    Ytr, Yva = L["train_Y"].astype(bool), L["val_Y"].astype(bool)
    Xtr = aligned(a.preds_train, [str(i) for i in L["train_ids"]])
    Xva = aligned(a.preds_val, [str(i) for i in L["val_ids"]])
    slope, intercept = fit_platt(Xtr, Ytr, np.ones_like(Ytr, bool))
    ptr = 1 / (1 + np.exp(-Xtr.astype(np.float64)))
    yd = np.array([youden(ptr[:, j], Ytr[:, j].astype(np.float64)) if Ytr[:, j].any() else 0.5 for j in range(C)])
    raw_plan, raw_poss, _, _ = tiers(slope, intercept, yd)
    pva = 1 / (1 + np.exp(-Xva.astype(np.float64)))
    summary = dict(val_tiers=tier_stats(pva, raw_plan, raw_poss, Yva, np.ones_like(Yva, bool)))
    return calibration_doc(classes, slope, intercept, yd, Ytr.sum(0), len(Ytr),
                           "1 = asserted or uncertain; 0 = denied or unmentioned (no masked entries)",
                           {"source_predictions": str(a.preds_train)}, summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("modality", choices=("ct", "xray"))
    ap.add_argument("--dense", type=Path, help="CT: dense study-bag logits (predict_ct.py merge)")
    ap.add_argument("--split", type=Path, help="CT: patient-level split JSON {fora_id: train|val|test}")
    ap.add_argument("--preds-train", type=Path)
    ap.add_argument("--preds-val", type=Path)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    doc = fit_ct(a) if a.modality == "ct" else fit_xray(a)
    write_json(a.out, doc)
    print(json.dumps(doc["summary"], indent=1))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Frozen X-ray MIL calibration and dense study-bag logits for the chest X-ray writer plan (paper section "Grounded
report generation", diagnostic plan tiers).

  * Platt per class, p = sigmoid(a * logit + b), ridge prior toward (1, 0) (l2 = 1.0), slope floor 0.05, fitted on the
    TRAIN study-bag logits against the train labels (1 = asserted or uncertain, 0 = denied or unmentioned);
  * possible tier = per-class Youden cutoff on the TRAIN raw probabilities (sigmoid(logit));
  * probable tier = calibrated probability >= 0.5, never below the Youden cutoff;
  * thresholds are chosen in raw probability space and mapped through the class Platt curve (monotone);
  * base rate shown in the plan = train positives / train studies (key ``class_val_positive`` /
    ``n_calibration_studies``).
Outputs: --out calibration json (schema of ``fora_report.plan.calibration_tiers``) and --dense npz (report_uid,
full_token_count, logit [n, C] f32, label, class_index, split; train + val rows; the trainers check
calibration['source_predictions'] == this path).

    python -m fora_report.cxr.calibration --preds-train preds_train.npz --preds-val preds_val.npz --labels labels.npz \\
        --cases cases_train.jsonl cases_val.jsonl --mil-checkpoint mil.pt --out calibration_train_platt.json \\
        --dense dense_predictions.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fora_report.cxr.data import case_tokens
from fora_report.plan.calibration_tiers import (class_youden, fit_platt, label_order_sha256, tier_thresholds,
                                              write_json_atomic)


def load_preds(path, classes):
    z = np.load(path, allow_pickle=False)
    if list(map(str, z['classes'].tolist())) != classes:
        raise SystemExit(f'{path}: class order differs from the labels')
    return [str(i) for i in z['ids'].tolist()], np.asarray(z['logits'], np.float32)


def align(ids, X, want_ids, name):
    pos = {i: k for k, i in enumerate(ids)}
    missing = [i for i in want_ids if i not in pos]
    if missing:
        raise SystemExit(f'{name}: {len(missing)} label rows without predictions (first {missing[0]})')
    return X[[pos[i] for i in want_ids]]


def build_calibration(Xtr, Ytr, Xva, Yva, l2: float = 1.0):
    """Train Platt + train Youden tiers -> (slope, intercept, tiers dict, summary dict)."""
    slope, intercept = fit_platt(Xtr, Ytr, np.ones_like(Ytr, bool), l2=l2)
    ptr = 1 / (1 + np.exp(-Xtr.astype(np.float64)))
    pva = 1 / (1 + np.exp(-Xva.astype(np.float64)))
    yd, _ = class_youden(ptr, Ytr, mask=None, require_negative=False)
    tiers = tier_thresholds(slope, intercept, yd)
    n_conf = (pva >= tiers['raw_plan']).sum(1)
    n_poss = ((pva >= tiers['raw_possible']) & (pva < tiers['raw_plan'])).sum(1)
    cal = 1 / (1 + np.exp(-(Xva * slope + intercept)))
    rule_used = ['youden_train' if yd[j] < tiers['raw_plan'][j] else 'none(=plan)' for j in range(len(yd))]
    summary = dict(
        val_plan_sizes=dict(confident_mean=float(n_conf.mean()), confident_median=float(np.median(n_conf)),
                            confident_p95=float(np.percentile(n_conf, 95)), confident_zero_frac=float((n_conf == 0).mean()),
                            possible_mean=float(n_poss.mean()), possible_median=float(np.median(n_poss)),
                            possible_p95=float(np.percentile(n_poss, 95))),
        val_brier=dict(raw=float(((pva - Yva) ** 2).mean()), platt=float(((cal - Yva) ** 2).mean())),
        possible_rule_counts={k: rule_used.count(k) for k in sorted(set(rule_used))},
        val_label_positives_mean=float(Yva.sum(1).mean()))
    return slope, intercept, tiers, summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--preds-train', type=Path, required=True, help='MIL study-bag predictions on TRAIN (ids, classes, logits)')
    ap.add_argument('--preds-val', type=Path, required=True)
    ap.add_argument('--labels', type=Path, required=True, help='labels npz: classes, train_ids/val_ids, train_Y/val_Y')
    ap.add_argument('--cases', nargs=2, type=Path, required=True, help='train and val study lists (token counts)')
    ap.add_argument('--mil-checkpoint', required=True, help='recorded in the calibration file')
    ap.add_argument('--platt-l2', type=float, default=1.0)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--dense', type=Path, required=True)
    a = ap.parse_args(argv)

    L = np.load(a.labels, allow_pickle=False)
    classes = [str(c) for c in L['classes'].tolist()]
    C = len(classes)
    tr_ids = [str(i) for i in L['train_ids'].tolist()]
    va_ids = [str(i) for i in L['val_ids'].tolist()]
    Ytr = L['train_Y'].astype(bool)
    Yva = L['val_Y'].astype(bool)
    Xtr = align(*load_preds(a.preds_train, classes), tr_ids, 'train')
    Xva = align(*load_preds(a.preds_val, classes), va_ids, 'val')
    print(f'train {Xtr.shape} val {Xva.shape} classes {C}', flush=True)
    slope, intercept, tiers, summary = build_calibration(Xtr, Ytr, Xva, Yva, l2=a.platt_l2)
    print(json.dumps(summary, indent=1), flush=True)

    case_tok = {}
    for path in a.cases:
        with open(path) as f:
            for line in f:
                c = json.loads(line)
                case_tok[c['id']] = case_tokens(c)
    uids = np.array(tr_ids + va_ids)
    a.dense.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.dense.with_name(a.dense.name + '.tmp')
    with open(tmp, 'wb') as f:
        np.savez(f, report_uid=uids, full_token_count=np.array([max(1, case_tok.get(u, 1)) for u in uids], np.int64),
                 logit=np.concatenate([Xtr, Xva]).astype(np.float32), label=np.array(classes),
                 class_index=np.arange(C, dtype=np.int64), split=np.array(['train'] * len(tr_ids) + ['val'] * len(va_ids)))
    tmp.replace(a.dense)

    calibration = {
        "format": "cxr_frozen_mil_train_platt", "format_version": 1,
        "mil_source": "precomputed", "mil_checkpoint": a.mil_checkpoint,
        "source_predictions": str(a.dense.resolve()),
        "calibration_logits_source": str(a.preds_train.resolve()),
        "threshold_train_logits_source": str(a.preds_train.resolve()),
        "source_split": "train", "n_calibration_studies": len(tr_ids), "calibration_report_uids": [],
        "exclude_from_report_training": False,
        "target_policy": "1 = asserted or uncertain; 0 = denied or unmentioned (no masked entries)",
        "platt": {"l2_toward_identity": a.platt_l2, "min_slope": 0.05},
        "threshold_rule": {"plan": "platt50_train", "possible": "youden_train"},
        "threshold_policy": {
            "plan": "calibrated probability (train Platt curve) >= 0.5; never below the train Youden cutoff",
            "possible": "train Youden cutoff, never above the plan threshold; mapped through Platt"},
        "labels": classes, "label_order_sha256": label_order_sha256(classes), "canonical_plan_labels": classes,
        "class_val_positive": Ytr.sum(0).astype(int).tolist(),
        "class_slope": slope.tolist(), "class_intercept": intercept.tolist(),
        "class_raw_plan_threshold": tiers['raw_plan'].tolist(),
        "class_raw_possible_threshold": tiers['raw_possible'].tolist(),
        "class_plan_threshold": tiers['plan'].tolist(), "class_possible_threshold": tiers['possible'].tolist(),
        "summary": summary,
    }
    write_json_atomic(a.out, calibration)
    print(f'wrote {a.out} and {a.dense}', flush=True)


if __name__ == '__main__':
    main()

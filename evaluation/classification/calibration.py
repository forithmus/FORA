"""Calibration of the Platt-scaled probabilities (paper Methods 4.9, Supplementary Fig. 7).

Platt parameters are fitted per finding on training study bags (classifier stage) and applied unchanged:
``p = sigmoid(slope * logit + intercept)``. Findings with at least 10 validation positives are assessed.

* ``ece``: expected calibration error with 15 equal-count bins per finding (bins = ``np.array_split`` of the
  probability order), macro-averaged over findings.
* ``reliability_equal_width``: 10 equal-width bins, bins with fewer than 20 studies left empty (per-finding panels).
* ``reliability_pooled_log``: all assessed findings pooled, 16 log-spaced bins between 1e-4 and 1 (pooled panel).

CLI::

    python -m evaluation.classification.calibration --pred val_predictions.npz --labels val_labels.npz \
        --platt platt.json --out calibration/

``platt.json``: ``{"labels": [finding, ...], "class_slope": [...], "class_intercept": [...]}``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

N_BINS = 15
MIN_POS = 10


def apply_platt(logits: np.ndarray, slope, intercept) -> np.ndarray:
    return 1 / (1 + np.exp(-(np.asarray(slope)[None] * logits + np.asarray(intercept)[None])))


def ece(p: np.ndarray, y: np.ndarray, nb: int = N_BINS) -> float:
    o = np.argsort(p)
    return float(sum(len(b) / len(p) * abs(p[b].mean() - y[b].mean()) for b in np.array_split(o, nb)))


def reliability_equal_width(p, y, nb: int = 10, minn: int = 20):
    idx = np.minimum((p * nb).astype(int), nb - 1)
    x = np.array([p[idx == b].mean() if (idx == b).sum() >= minn else np.nan for b in range(nb)])
    o = np.array([y[idx == b].mean() if (idx == b).sum() >= minn else np.nan for b in range(nb)])
    return x, o, np.array([(idx == b).sum() for b in range(nb)])


def reliability_pooled_log(P: np.ndarray, Y: np.ndarray, keep: np.ndarray):
    pp = P[:, keep].ravel(); yy = Y[:, keep].ravel()
    edges = np.r_[0, np.logspace(-4, 0, 17)[1:-1], 1.0001]
    idx = np.digitize(pp, edges) - 1
    nb = len(edges) - 1
    mp = np.array([pp[idx == b].mean() if (idx == b).any() else np.nan for b in range(nb)])
    ob = np.array([yy[idx == b].mean() if (idx == b).any() else np.nan for b in range(nb)])
    return mp, ob, np.array([(idx == b).sum() for b in range(nb)])


def calibration_report(P: np.ndarray, Y: np.ndarray, findings: list[str]) -> dict:
    """Per-finding ECE for findings with >= 10 positives (Y bool, studies with unknown labels removed upstream)."""
    npos = Y.sum(0); keep = np.flatnonzero(npos >= MIN_POS)
    E = np.array([ece(P[:, j], Y[:, j]) for j in keep])
    rows = [dict(finding=findings[j], n_pos=int(npos[j]), mean_pred=float(P[:, j].mean()), observed_rate=float(Y[:, j].mean()), ECE=float(e))
            for j, e in zip(keep, E)]
    mp, ob, cnt = reliability_pooled_log(P, Y, keep)
    return dict(findings=len(keep), macro_ECE=float(E.mean()), median_ECE=float(np.median(E)), per_finding=rows,
                pooled_curve=dict(mean_pred=mp.tolist(), observed=ob.tolist(), n=cnt.tolist()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pred', required=True); ap.add_argument('--labels', required=True)
    ap.add_argument('--platt', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    from evaluation.classification.heldout import align
    L = dict(np.load(a.labels, allow_pickle=False)); F = [str(f) for f in L['findings']]
    S = align(dict(np.load(a.pred, allow_pickle=False)), L)['scores']
    cal = json.load(open(a.platt)); ci = {f: i for i, f in enumerate(cal['labels'])}
    sl = np.array([cal['class_slope'][ci[f]] for f in F]); ic = np.array([cal['class_intercept'][ci[f]] for f in F])
    P = apply_platt(S, sl, ic); Y = L['labels'] == 1
    rep = calibration_report(P, Y, F)
    with open(os.path.join(a.out, 'ece_per_finding.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, list(rep['per_finding'][0])); w.writeheader(); w.writerows(rep['per_finding'])
    json.dump({k: v for k, v in rep.items() if k != 'per_finding'}, open(os.path.join(a.out, 'calibration.json'), 'w'), indent=1)
    print('findings', rep['findings'], 'macro ECE', round(rep['macro_ECE'], 4))


if __name__ == '__main__':
    main()

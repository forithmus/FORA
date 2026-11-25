"""Decision thresholds of FORA (paper Methods 4.9, "training Youden cut-off"), fitted on training predictions only.

CT: per finding and scan type, the cut-off that maximises Youden's J on the training
study bags of that scan type, placed half-way between the best score and the next lower score. If the scan type has
fewer than 50 training positives or fewer than 50 training negatives, the cut-off fitted on all training studies is
used. A study is positive when ``logit > cut-off``.

For a benchmark finding that maps to several FORA classes the score is the maximum member logit and the training
target is positive when any member is asserted or uncertain, negative when all known members are absent and
unknown when every member is unknown. External benchmarks use the cut-off of the scan types listed
in ``benchmarks/configs/ct_mappings.json`` (chest for CT-RATE and RAD-ChestCT, abdomen + CAP for Merlin and
KiTS23, chest + CAP for RibFrac).

X-ray: Youden cut-off on the training composite score (max sigmoid over mapped
classes), set to the best score itself; a case is positive when ``score >= cut-off``.

CLI::

    python -m evaluation.classification.thresholds ct --train train_predictions.npz --out youden_cutoffs.json \
        [--mappings evaluation/benchmarks/configs/ct_mappings.json]

``train_predictions.npz`` (FORA training split): ``case_ids``, ``findings`` (FORA class names), ``scores``
(float logits, n x K), ``labels`` (int8, 1 asserted/uncertain, 0 absent, -1 unknown) and ``scan_type`` (str per study).
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np

SCAN_TYPES = ('chest', 'abdomen', 'headneck', 'CAP', 'other')
MIN_SCAN_TYPE_POS = 50
MIN_SCAN_TYPE_NEG = 50
MIN_TRAIN_POS = 10


def scan_type_from_protocols(protocols: list[str]) -> str:
    """FORA scan type of a study from its report protocol names ."""
    p = ' | '.join(protocols or [])
    if 'Chest, Abdomen and Pelvis' in p:
        return 'CAP'
    if re.search(r'CT Chest - |CTA Chest|Coronary CTA', p):
        return 'chest'
    if re.search(r'Abdomen|Urography', p):
        return 'abdomen'
    if re.search(r'CT Head|CTA Head|Sinus|Temporal|CT Neck|CTA Neck|Dental', p):
        return 'headneck'
    return 'other'


def youden_midpoint(z: np.ndarray, y: np.ndarray) -> float:
    """CT cut-off: argmax of TPR - FPR in descending order, midpoint to the next score."""
    o = np.argsort(-z)
    sv = z[o]; yy = y[o]
    tp = np.cumsum(yy == 1); fp = np.cumsum(yy == 0)
    b = tp / (yy == 1).sum() - fp / (yy == 0).sum()
    i = int(np.argmax(b))
    return float((sv[i] + sv[min(i + 1, len(sv) - 1)]) / 2)


def youden_cutoff(z, y, scan_type, groups=None, min_pos=MIN_SCAN_TYPE_POS, min_neg=MIN_SCAN_TYPE_NEG) -> float:
    """Youden on the training studies of ``groups`` (scan types), fallback to all training studies."""
    z = np.asarray(z, np.float64); y = np.asarray(y)
    m = y >= 0
    t = youden_midpoint(z[m], y[m])
    if groups is None:
        return t
    k = m & np.isin(np.asarray(scan_type), groups)
    if (y[k] == 1).sum() >= min_pos and (y[k] == 0).sum() >= min_neg:
        return youden_midpoint(z[k], y[k])
    return t


def composite(scores: np.ndarray, labels: np.ndarray | None, findings: list[str], members: list[str]):
    """Max-logit composite of member classes and its OR training target ."""
    cols = [findings.index(m) for m in members]
    z = np.asarray(scores[:, cols], np.float64).max(1)
    if labels is None:
        return z, None
    a = np.asarray(labels[:, cols])
    y = np.where((a < 0).all(1), -1, (a == 1).any(1).astype(int))
    return z, y


def youden_xray(s: np.ndarray, y: np.ndarray) -> float:
    """X-ray cut-off: positive if score >= returned value."""
    o = np.argsort(-s, kind='mergesort')
    s = s[o]; y = y[o]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    last = np.r_[s[1:] != s[:-1], True]
    j = tp / y.sum() - fp / (len(y) - y.sum())
    j[~last] = -np.inf
    return float(s[int(np.argmax(j))])


def fit_ct_cutoffs(train: dict, mappings: dict | None = None) -> dict:
    """Youden cut-offs for every FORA finding x scan type and, optionally, for every benchmark finding.

    Output layout:
    ``{"fora_per_scan_type": {finding: {scan_type: cut}}, "external": {dataset: {finding: {...}}}}``.
    Findings with fewer than 10 training positives get no cut-off (they are never scored).
    """
    S = train['scores']; Y = train['labels']; st = np.asarray(train['scan_type']).astype(str)
    F = [str(f) for f in train['findings']]
    out: dict = {'fora_per_scan_type': {}, 'external': {}}
    for j, f in enumerate(F):
        z, y = S[:, j].astype(np.float64), np.asarray(Y[:, j])
        if (y == 1).sum() < MIN_TRAIN_POS:
            continue
        out['fora_per_scan_type'][f] = {g: youden_cutoff(z, y, st, [g]) for g in SCAN_TYPES}
    for ds, spec in (mappings or {}).items():
        if ds.startswith('_'):
            continue
        for name, members in spec['findings'].items():
            z, y = composite(S, Y, F, members)
            out['external'].setdefault(ds, {})[name] = dict(
                members=members, scan_types=spec['scan_types'], cutoff=youden_cutoff(z, y, st, spec['scan_types']))
    return out


def cutoff_matrix(cutoffs: dict, findings: list[str], scan_type: np.ndarray) -> np.ndarray:
    """(n, K) matrix of per-study cut-offs; +inf where a finding has no cut-off (never positive)."""
    scan_type = np.asarray(scan_type).astype(str)
    T = np.full((len(scan_type), len(findings)), np.inf, np.float64)
    for j, f in enumerate(findings):
        for g, t in cutoffs.get(f, {}).items():
            T[scan_type == g, j] = t
    return T


def _load_npz(path: str) -> dict:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('modality', choices=['ct'])
    ap.add_argument('--train', required=True)
    ap.add_argument('--mappings')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    maps = json.load(open(a.mappings)) if a.mappings else None
    res = fit_ct_cutoffs(_load_npz(a.train), maps)
    json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()

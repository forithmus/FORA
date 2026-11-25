"""Subgroup analyses of the held-out set by age, sex, scanner manufacturer and tube voltage (paper Methods 4.9,
Supplementary Fig. 9).

For each attribute the macro AUROC and macro balanced accuracy are taken over one common set of (scan type,
finding) cells: those with at least 10 positives and at least 50 negatives in every level of the attribute.
Studies with unknown values and the small "other manufacturer" group are not shown as levels.

Levels:
    age (years)        <18, 18-39, 40-59, 60-74, 75+   (ages outside 0-110 are unknown)
    sex                F, M
    manufacturer       GE, Philips, Siemens
    tube voltage (kVp) <=100, 110-120, >120

CLI::

    python -m evaluation.classification.subgroups --pred fora_val.npz --labels val_labels.npz \
        --cutoffs youden_cutoffs.json --meta val_meta.csv --out subgroups.json

``val_meta.csv`` columns: ``case_id, age, sex, manufacturer, kvp`` (empty = unknown).
"""
from __future__ import annotations

import argparse
import csv
import json

import numpy as np

from evaluation.classification.heldout import MIN_NEG, MIN_POS
from evaluation.classification.metrics import auroc, balanced_accuracy
from evaluation.classification.thresholds import SCAN_TYPES, cutoff_matrix


def age_band(a) -> str:
    if a is None or not a == a or a < 0 or a > 110:
        return 'unknown'
    return '<18' if a < 18 else '18-39' if a < 40 else '40-59' if a < 60 else '60-74' if a < 75 else '75+'


def manufacturer_group(v: str) -> str:
    return v if v in ('GE', 'Philips', 'Siemens') else ('unknown' if v in ('', 'unknown', None) else 'Other')


def kvp_band(k) -> str:
    if k is None or not k == k:
        return 'unknown'
    return '<=100' if k <= 100 else '110-120' if k <= 120 else '>120'


LEVELS = {'Age (years)': ['<18', '18-39', '40-59', '60-74', '75+'], 'Sex': ['F', 'M'],
          'Scanner manufacturer': ['GE', 'Philips', 'Siemens'], 'Tube voltage (kVp)': ['<=100', '110-120', '>120']}


def common_cells(Y, scan_type, vals, levels):
    cells = []
    for g in SCAN_TYPES:
        ok = np.ones(Y.shape[1], bool)
        for lv in levels:
            y = Y[(scan_type == g) & (vals == lv)]
            ok &= ((y == 1).sum(0) >= MIN_POS) & ((y == 0).sum(0) >= MIN_NEG)
        cells += [(g, int(j)) for j in np.flatnonzero(ok)]
    return cells


def subgroup_table(S, D, Y, scan_type, attributes: dict) -> list[dict]:
    rows = []
    for name, vals in attributes.items():
        levels = LEVELS[name]; cells = common_cells(Y, scan_type, vals, levels)
        if not cells:
            continue
        for lv in levels:
            au, ba = [], []
            for g, j in cells:
                r = (scan_type == g) & (vals == lv)
                au.append(auroc(S[r, j], Y[r, j])); ba.append(balanced_accuracy(D[r, j], Y[r, j]))
            rows.append(dict(attribute=name, level=lv, n_studies=int((vals == lv).sum()), n_cells=len(cells),
                             macro_AUROC=float(np.nanmean(au)), macro_BA=float(np.mean(ba))))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pred', required=True); ap.add_argument('--labels', required=True)
    ap.add_argument('--cutoffs', required=True); ap.add_argument('--meta', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    from evaluation.classification.heldout import align
    L = dict(np.load(a.labels, allow_pickle=False)); F = [str(f) for f in L['findings']]; st = L['scan_type'].astype(str)
    S = align(dict(np.load(a.pred, allow_pickle=False)), L)['scores']; Y = L['labels'].astype(int)
    D = S > cutoff_matrix(json.load(open(a.cutoffs))['fora_per_scan_type'], F, st)
    meta = {r['case_id']: r for r in csv.DictReader(open(a.meta))}
    m = [meta.get(str(u), {}) for u in L['case_ids']]
    num = lambda x: float(x) if x not in (None, '') else float('nan')
    attrs = {'Age (years)': np.array([age_band(num(r.get('age'))) for r in m]),
             'Sex': np.array([r.get('sex') or 'U' for r in m]),
             'Scanner manufacturer': np.array([manufacturer_group(r.get('manufacturer') or 'unknown') for r in m]),
             'Tube voltage (kVp)': np.array([kvp_band(num(r.get('kvp'))) for r in m])}
    json.dump(subgroup_table(S, D, Y, st, attrs), open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()

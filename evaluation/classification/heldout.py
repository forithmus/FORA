"""Held-out (internal validation) finding recognition (paper Methods 4.9, Fig. 2a-b, Supplementary Fig. 4).

Cells are (finding, CT type) pairs with at least 10 positive and at least 50 negative validation studies. The CT
types are the 11 examination types plus "mixed" (``data-preprocessing/exam_types.py``: a study belongs to the
type whose protocol classes contain all of its protocol classes, otherwise to "mixed"); on the FORA validation set
this gives 2,132 cells over 854 findings.
AUROC, AUPRC and balanced accuracy are averaged over cells; CRG pools every decision of the decision matrix.
FORA decisions are ``logit > training Youden cut-off`` of the study's scan type; any other model is
scored with its own native decisions. The same scorer is applied to every model.

Prediction file (one per model, ``.npz``)::

    case_ids   (n,)  str   study identifiers (must match the label file; order free)
    findings   (K,)  str   finding names
    scores     (n,K) float continuous score (logit, probability, yes-minus-no margin, ...)
    decision   (n,K) bool  optional native decisions; required unless --cutoffs is given

Label file (``.npz``): ``case_ids``, ``findings``, ``labels`` (int8, 1/0/-1), ``exam_type`` (str, one of the 12 CT
types; defines the cells), ``scan_type`` (str, chest / abdomen / headneck / CAP / other; selects the FORA cut-off)
and optionally ``patient_ids`` (str, used by the confidence intervals). Without ``exam_type`` the cells use
``scan_type``.

CLI::

    python -m evaluation.classification.heldout --labels val_labels.npz --pred fora=fora_val.npz \
        --pred nv_reason=nv_val.npz --cutoffs fora=youden_cutoffs.json --out results/heldout [--ci --draws 200000]
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

from evaluation.classification.metrics import auroc, average_precision, crg, operating_point
from evaluation.classification.thresholds import cutoff_matrix

MIN_POS, MIN_NEG = 10, 50


def eligible_cells(Y: np.ndarray, group: np.ndarray, min_pos=MIN_POS, min_neg=MIN_NEG, groups=None):
    """List of (group, column, row indices) with >= min_pos positives and >= min_neg negatives."""
    cells = []
    for g in (sorted(set(group.tolist())) if groups is None else groups):
        rows = np.flatnonzero(group == g)
        if not len(rows):
            continue
        y = Y[rows]
        ok = ((y == 1).sum(0) >= min_pos) & ((y == 0).sum(0) >= min_neg)
        cells += [(g, int(j), rows) for j in np.flatnonzero(ok)]
    return cells


def align(pred: dict, labels: dict) -> dict:
    """Reorder a prediction file to the label file's studies and findings (missing findings -> NaN / False)."""
    idx = {str(u): i for i, u in enumerate(pred['case_ids'])}
    rows = np.array([idx[str(u)] for u in labels['case_ids']])
    pf = [str(f) for f in pred['findings']]
    cols = [pf.index(f) if f in pf else -1 for f in labels['findings']]
    n, K = len(rows), len(cols)
    S = np.full((n, K), np.nan); D = np.zeros((n, K), bool)
    for k, c in enumerate(cols):
        if c < 0:
            continue
        S[:, k] = pred['scores'][rows, c]
        if 'decision' in pred:
            D[:, k] = pred['decision'][rows, c]
    out = dict(scores=S, covered=np.array([c >= 0 for c in cols]))
    if 'decision' in pred:
        out['decision'] = D
    return out


def score_model(S: np.ndarray, D: np.ndarray, Y: np.ndarray, findings: list[str], cells) -> tuple[list[dict], dict]:
    rows_out = []
    for g, j, r in cells:
        s, d, y = S[r, j], D[r, j], Y[r, j]
        if not np.isfinite(s).any():
            continue
        row = dict(finding=findings[j], group=g, n_pos=int((y == 1).sum()), n_neg=int((y == 0).sum()),
                   prevalence=float((y == 1).sum() / max((y >= 0).sum(), 1)), AUROC=auroc(s, y), AUPRC=average_precision(s, y))
        row.update(operating_point(d, y))
        rows_out.append(row)
    summary = dict(cells=len(rows_out), findings=len({r['finding'] for r in rows_out}),
                   macro_AUROC=float(np.nanmean([r['AUROC'] for r in rows_out])),
                   macro_AUPRC=float(np.nanmean([r['AUPRC'] for r in rows_out])),
                   macro_BA=float(np.mean([r['ba'] for r in rows_out])),
                   macro_sens=float(np.mean([r['sens'] for r in rows_out])),
                   macro_spec=float(np.mean([r['spec'] for r in rows_out])),
                   macro_PPV=float(np.mean([r['ppv'] for r in rows_out])),
                   CRG=crg(D, Y)[0])
    return rows_out, summary


def per_group(rows: list[dict]) -> list[dict]:
    out = []
    for g in sorted({r['group'] for r in rows}):
        rr = [r for r in rows if r['group'] == g]
        if rr:
            out.append(dict(group=g, cells=len(rr), macro_AUROC=float(np.nanmean([r['AUROC'] for r in rr])),
                            macro_AUPRC=float(np.nanmean([r['AUPRC'] for r in rr])), macro_BA=float(np.mean([r['ba'] for r in rr]))))
    return out


def _write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, keys); w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--labels', required=True)
    ap.add_argument('--pred', action='append', required=True, help='NAME=predictions.npz')
    ap.add_argument('--cutoffs', action='append', default=[], help='NAME=youden_cutoffs.json (FORA-style decisions: score > cut)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--ci', action='store_true', help='add linearization intervals and paired tests vs the first model')
    ap.add_argument('--draws', type=int, default=200_000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    L = dict(np.load(a.labels, allow_pickle=False))
    F = [str(f) for f in L['findings']]; Y = L['labels'].astype(int); st = L['scan_type'].astype(str)
    cuts = dict(x.split('=', 1) for x in a.cutoffs)
    cells = eligible_cells(Y, L['exam_type'].astype(str) if 'exam_type' in L else st)
    models = {}
    for spec in a.pred:
        name, path = spec.split('=', 1)
        P = align(dict(np.load(path, allow_pickle=False)), L)
        if name in cuts:
            C = json.load(open(cuts[name]))['fora_per_scan_type']
            D = P['scores'] > cutoff_matrix(C, F, st)
        else:
            D = P['decision']
        models[name] = (P['scores'], D)
    summary = {}
    for name, (S, D) in models.items():
        rows, summ = score_model(S, D, Y, F, cells)
        _write_csv(os.path.join(a.out, f'{name}_per_cell.csv'), rows)
        summ['per_group'] = per_group(rows)
        summary[name] = summ
    if a.ci:
        from evaluation.statistics.linearization import heldout_intervals
        pid = L['patient_ids'].astype(str) if 'patient_ids' in L else L['case_ids'].astype(str)
        summary['intervals'] = heldout_intervals(models, Y, cells, pid, draws=a.draws)
    json.dump(summary, open(os.path.join(a.out, 'summary.json'), 'w'), indent=1)
    print(json.dumps({k: {m: v[m] for m in ('cells', 'macro_AUROC', 'macro_BA', 'CRG')} for k, v in summary.items() if k != 'intervals'}, indent=1))


if __name__ == '__main__':
    main()

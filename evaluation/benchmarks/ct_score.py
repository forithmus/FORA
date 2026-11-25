"""Scoring of any model on the public CT benchmarks (paper Methods 4.9, Fig. 2c-d, Supplementary Fig. 5).

Every model is scored on the full finding set of the benchmark with the same code: per-finding AUROC, AUPRC,
sensitivity, specificity, balanced accuracy and precision, their macro means, and CRG pooled over all decisions.
Decisions are the model's own operating point:

* FORA: ``score > cut-off``, with the training Youden cut-off of each benchmark finding
  (``thresholds.py ct --mappings benchmarks/configs/ct_mappings.json`` -> ``external``), or with per-finding
  cut-offs passed directly.
* Any other model: its native decision (``decision`` array of the prediction file, or ``score > 0`` with
  ``--native-zero``, i.e. a yes-minus-no margin or a released threshold already subtracted).

Prediction file (``.npz``): ``case_ids`` (str), ``findings`` (benchmark finding names), ``scores`` (float n x K),
optional ``decision`` (bool n x K). Missing findings are allowed (NaN scores); they are left out of that model's
macro means. Truth file: written by ``ct_readers`` (``Truth.save``).

Intervals (``--draws``): paired cluster bootstrap of each model against the first model (patients for CT-RATE,
studies or cases elsewhere), percentile 95% intervals and two-sided paired p-values (``--test normal`` for the
normal approximation used on KiTS23 and RibFrac). Holm correction is applied over all comparisons of the call.

CLI::

    python -m evaluation.benchmarks.ct_score --truth ctrate_truth.npz --dataset CT-RATE \
        --pred FORA=fora_ctrate.npz --fora-cutoffs youden_cutoffs.json --pred NV-Reason-CT=nv.npz --native-zero NV-Reason-CT \
        --draws 200000 --out ctrate_scores.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from evaluation.classification.metrics import auroc, average_precision, crg, operating_point


def score_matrix(S: np.ndarray, D: np.ndarray, Y: np.ndarray, findings: list[str]) -> dict:
    per = {}
    for j, f in enumerate(findings):
        if not np.isfinite(S[:, j]).any():
            continue
        y = Y[:, j]
        if (y == 1).sum() == 0 or (y == 0).sum() == 0:
            continue
        per[f] = dict(n_pos=int((y == 1).sum()), n_neg=int((y == 0).sum()), AUROC=auroc(S[:, j], y), AUPRC=average_precision(S[:, j], y),
                      **operating_point(D[:, j], y))
    cols = [findings.index(f) for f in per]
    mac = {f'macro_{k}': float(np.nanmean([per[f][k] for f in per])) for k in ('AUROC', 'AUPRC', 'sens', 'spec', 'ba', 'ppv')}
    mac['CRG'] = crg(D[:, cols], Y[:, cols])[0]
    mac['findings'] = len(per)
    return dict(macro=mac, per_finding=per)


def load_pred(path: str, case_ids, findings) -> tuple[np.ndarray, np.ndarray | None]:
    z = np.load(path, allow_pickle=False)
    idx = {str(u): i for i, u in enumerate(z['case_ids'])}
    rows = np.array([idx[str(u)] for u in case_ids])
    pf = [str(f) for f in z['findings']]
    S = np.full((len(rows), len(findings)), np.nan)
    D = np.zeros((len(rows), len(findings)), bool) if 'decision' in z.files else None
    for k, f in enumerate(findings):
        if f in pf:
            S[:, k] = z['scores'][rows, pf.index(f)]
            if D is not None:
                D[:, k] = z['decision'][rows, pf.index(f)]
    return S, D


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--truth', required=True); ap.add_argument('--dataset', required=True)
    ap.add_argument('--pred', action='append', required=True, help='NAME=predictions.npz (first = reference model of the tests)')
    ap.add_argument('--fora-cutoffs', help='cut-off JSON with external[dataset][finding].cutoff, applied to the model named FORA')
    ap.add_argument('--native-zero', action='append', default=[], help='model names whose decision is score > 0')
    ap.add_argument('--draws', type=int, default=0); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--test', choices=['percentile', 'normal'], default='percentile')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    T = dict(np.load(a.truth, allow_pickle=False))
    ids = [str(u) for u in T['case_ids']]; F = [str(f) for f in T['findings']]; Y = T['labels'].astype(int)
    models = {}
    for spec in a.pred:
        name, path = spec.split('=', 1)
        S, D = load_pred(path, ids, F)
        if name == 'FORA' and a.fora_cutoffs:
            C = json.load(open(a.fora_cutoffs))['external'][a.dataset]
            D = S > np.array([C[f]['cutoff'] for f in F])[None]
        elif name in a.native_zero or D is None:
            D = np.nan_to_num(S, nan=-1e9) > 0
        models[name] = (S, D)
    out = dict(dataset=a.dataset, n_cases=len(ids), models={n: score_matrix(S, D, Y, F) for n, (S, D) in models.items()})
    if a.draws:
        from evaluation.statistics.bootstrap import macro_auroc_stat, macro_ba_stat, paired_bootstrap
        from evaluation.statistics.multiple_testing import holm
        names = list(models); ref = names[0]; comps = []
        cl = T['patient_ids'].astype(str) if 'patient_ids' in T else None
        for other in names[1:]:
            cols = [F.index(f) for f in out['models'][other]['per_finding']]
            for metric, mk in (('AUROC', macro_auroc_stat), ('BA', macro_ba_stat)):
                arg = (lambda m: models[m][0]) if metric == 'AUROC' else (lambda m: models[m][1])
                r = paired_bootstrap(mk(arg(ref), Y, cols), mk(arg(other), Y, cols), len(ids), a.draws, a.seed, cl, test=a.test)
                comps.append(dict(model=ref, versus=other, metric=metric, findings=len(cols), **r))
        for c, p in zip(comps, holm([c['p_two_sided'] for c in comps])):
            c['p_holm'] = p
        out['comparisons'] = comps
    json.dump(out, open(a.out, 'w'), indent=1)
    for n, r in out['models'].items():
        m = r['macro']; print(f"{a.dataset:12s} {n:18s} AUROC {m['macro_AUROC']:.4f} AUPRC {m['macro_AUPRC']:.4f} BA {m['macro_ba']:.4f} CRG {m['CRG']:.4f} ({m['findings']} findings)")


if __name__ == '__main__':
    main()

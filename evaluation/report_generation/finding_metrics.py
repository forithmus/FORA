"""Finding-level scoring of generated reports against reference reports (paper Methods 4.11, Fig. 4).

The same extractor labels the reference Findings and every generated Findings text (``labels/report_scoring``;
asserted or uncertain = positive). This module takes the resulting per-report positive class sets and computes,
per model:

* macro recall, specificity, balanced accuracy and precision over findings with >= 10 reference positives
  (``ge10``, the paper metric set) and over findings with >= 1 positive and >= 1 negative (``all``);
* CRG pooled over the same findings; micro precision / recall; findings per report;
* coverage (share of studies with a label record for the model; a missing report or label record counts as a
  report without findings) and loop rate (a sentence of more than 8 characters repeated at least 3 times, or the
  generation cap reached);
* 95% intervals from 1,000 bootstrap resamples (seed 0) of patients on CT-RATE (``patient`` field) and of studies
  elsewhere, and, with ``--paired-draws``, paired tests of the first model against every other model from 2,000
  resamples on the same clusters, Holm-corrected over the comparisons of the call.

Inputs::

    --studies   studies.jsonl  {uid, patient?}                 the evaluated set (all studies with a reference)
    --labels    labels.jsonl   {source, uid, positive_classes}  source = "ref" or "gen:<model>"
    --classes   classes.json   [class names]                   (CT: 1,028 findings; X-ray: see xray_labels.py)
    --texts     NAME=gens.jsonl {uid, text, hit_cap?}          optional, for the loop rate

CLI::

    python -m evaluation.report_generation.finding_metrics --studies ctrate_studies.jsonl --labels labels.jsonl \
        --classes classes.json --out scores_ctrate.json --paired-draws 2000
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict

import numpy as np

MIN_REF_POS = 10


def looped(text: str, hit_cap=False) -> bool:
    s = [x.strip().lower() for x in re.split(r'(?<=[.!?])\s+', text or '') if len(x.strip()) > 8]
    return (max(Counter(s).values()) >= 3 if s else False) or bool(hit_cap)


def summarise(P: np.ndarray, Y: np.ndarray, cls: np.ndarray, w: np.ndarray) -> dict:
    M = [np.asarray(x, np.float32) for x in (P & Y, P & ~Y, ~P & Y, ~P & ~Y)]
    TP, FP, FN, TN = ((w.astype(np.float32) @ x).astype(np.float64)[cls] for x in M)
    pos, neg = TP + FN, FP + TN; ok = (pos > 0) & (neg > 0)
    TP, FP, FN, TN, pos, neg = TP[ok], FP[ok], FN[ok], TN[ok], pos[ok], neg[ok]
    prec = np.where(TP + FP > 0, TP / np.maximum(TP + FP, 1e-12), 0.0); rec = TP / pos; spec = TN / neg
    T_, A = (pos + neg).sum(), pos.sum(); tp, fn, fp = TP.sum(), FN.sum(), FP.sum()
    wcr = (T_ - A) / (2 * A); s = wcr * tp - wcr * fn - fp; sm = A * wcr
    return dict(macro_recall=rec.mean(), macro_specificity=spec.mean(), macro_BA=((rec + spec) / 2).mean(), macro_precision=prec.mean(),
                CRG=sm / (2 * sm - s), micro_precision=tp / max(tp + fp, 1e-12), micro_recall=tp / max(tp + fn, 1e-12),
                findings_per_report=float((w @ P[:, cls].sum(1)) / w.sum()), n_classes=int(ok.sum()))


def cluster_resamples(clusters, draws: int, seed: int = 0, n: int | None = None) -> list[np.ndarray]:
    """Bootstrap multiplicities per study; ``clusters=None`` resamples studies (rows) directly."""
    if clusters is None:
        rng = np.random.default_rng(seed)
        return [np.bincount(rng.integers(0, n, n), minlength=n).astype(float) for _ in range(draws)]
    cu, inv = np.unique(np.asarray(clusters), return_inverse=True); inv = inv.ravel(); rng = np.random.default_rng(seed)
    return [np.bincount(rng.integers(0, len(cu), len(cu)), minlength=len(cu))[inv].astype(float) for _ in range(draws)]


def score_models(Y: np.ndarray, preds: dict, clusters, boot: int = 1000, paired_draws: int = 0) -> dict:
    n = len(Y); ones = np.ones(n); Wb = cluster_resamples(clusters, boot, 0, n) if boot else []
    sets = {'all': np.arange(Y.shape[1]), 'ge10': np.flatnonzero(Y.sum(0) >= MIN_REF_POS)}
    out = {}
    for name, P in preds.items():
        m = {}
        for cname, cls in sets.items():
            pt = summarise(P, Y, cls, ones); bs = [summarise(P, Y, cls, w) for w in Wb]
            m[cname] = {k: (dict(value=float(v), ci95=[float(np.percentile([b[k] for b in bs], 2.5)), float(np.percentile([b[k] for b in bs], 97.5))])
                            if bs else dict(value=float(v))) if k != 'n_classes' else v for k, v in pt.items()}
        out[name] = m
    comps = []
    if paired_draws and len(preds) > 1:
        from evaluation.statistics.bootstrap import two_sided_p
        from evaluation.statistics.multiple_testing import holm
        Wp = cluster_resamples(clusters, paired_draws, 1, n); cls = sets['ge10']; names = list(preds)
        for other in names[1:]:
            for k in ('macro_recall', 'macro_precision', 'macro_BA', 'CRG'):
                d = [summarise(preds[names[0]], Y, cls, w)[k] - summarise(preds[other], Y, cls, w)[k] for w in Wp]
                pt = summarise(preds[names[0]], Y, cls, ones)[k] - summarise(preds[other], Y, cls, ones)[k]
                comps.append(dict(model=names[0], versus=other, metric=k, diff=float(pt), p_two_sided=two_sided_p(d)))
        for c, p in zip(comps, holm([c['p_two_sided'] for c in comps])):
            c['p_holm'] = p
    return dict(models=out, comparisons=comps)


def load_matrices(studies: list[dict], label_rows: list[dict], classes: list[str]):
    ci = {c: j for j, c in enumerate(classes)}; ids = [s['uid'] for s in studies]; ix = {u: i for i, u in enumerate(ids)}
    by = defaultdict(dict)
    for r in label_rows:
        if r['uid'] in ix:
            by[r['source']][r['uid']] = r['positive_classes']
    ref = by.pop('ref'); keep = [u for u in ids if u in ref]; kx = {u: i for i, u in enumerate(keep)}

    def mat(d):
        M = np.zeros((len(keep), len(classes)), bool)
        for u, cs in d.items():
            if u in kx:
                for c in cs:
                    M[kx[u], ci[c]] = True
        return M
    preds = {src.split(':', 1)[1]: mat(d) for src, d in sorted(by.items())}
    coverage = {src.split(':', 1)[1]: sum(u in d for u in keep) / max(len(keep), 1) for src, d in by.items()}
    return keep, mat(ref), preds, coverage


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--studies', required=True); ap.add_argument('--labels', required=True); ap.add_argument('--classes', required=True)
    ap.add_argument('--texts', action='append', default=[]); ap.add_argument('--first', help='model listed first (paired tests reference)')
    ap.add_argument('--boot', type=int, default=1000); ap.add_argument('--paired-draws', type=int, default=0); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    studies = [json.loads(l) for l in open(a.studies)]; rows = [json.loads(l) for l in open(a.labels)]
    keep, Y, preds, cov = load_matrices(studies, rows, json.load(open(a.classes)))
    if a.first:
        preds = {a.first: preds.pop(a.first), **preds}
    pat = {s['uid']: s.get('patient', s['uid']) for s in studies}
    res = score_models(Y, preds, [pat[u] for u in keep], a.boot, a.paired_draws)
    for name in res['models']:
        res['models'][name]['coverage'] = cov.get(name)
    for spec in a.texts:
        name, p = spec.split('=', 1); ks = set(keep)
        lp = [looped(r.get('text', ''), r.get('hit_cap')) for r in map(json.loads, open(p)) if r['uid'] in ks]
        res['models'].setdefault(name, {})['loop_rate'] = float(np.mean(lp)) if lp else None
    res['n_reference_studies'] = len(keep)
    json.dump(res, open(a.out, 'w'), indent=1)
    for name, m in res['models'].items():
        g = m.get('ge10', {})
        if g:
            print(f"{name:24s} rec {g['macro_recall']['value']:.3f} spec {g['macro_specificity']['value']:.3f} BA {g['macro_BA']['value']:.3f} "
                  f"prec {g['macro_precision']['value']:.3f} CRG {g['CRG']['value']:.3f}")


if __name__ == '__main__':
    main()

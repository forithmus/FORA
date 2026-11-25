"""Scoring of any model on the chest X-ray classification benchmarks (paper Methods 4.9).

Metrics per benchmark: macro AUROC, precision, recall, specificity, balanced accuracy over labels with at least one
positive and one negative case, CRG pooled over the same labels, macro AUPRC (scikit-learn definition), and 95%
intervals from 1,000 case resamples (seed 0; weighted Mann-Whitney AUROC and weighted confusion counts on shared
draws). AUPRC intervals use a per-label stratified bootstrap so every label stays in every replicate.

FORA scores of a benchmark label are the maximum sigmoid over the mapped FORA classes
(``configs/xray_mappings.json``); its cut-off is the Youden cut-off of the same composite on the FORA X-ray
training predictions (training target = OR of the mapped classes); a case is positive when ``score >= cut-off``.
On Open-I every image of a study receives the study-bag score. Other models are scored from their own scores
and either their own cut-offs or their native decisions.

Inputs::

    truth        labels.json {names, ids, Y} (xray_readers)
    FORA         --fora val_logits.npz (ids, classes, logits) --fora-train train_logits.npz (ids, classes, logits, labels)
    other model  --pred NAME=predictions.npz (case_ids, findings, scores, and decision or cutoff)

CLI::

    python -m evaluation.benchmarks.xray_score --truth chestxray14_labels.json --mapping chestxray14 \
        --fora fora_cxr14.npz --fora-train fora_train.npz --pred CARZero=carzero.npz --out cxr14_scores.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from evaluation.classification.metrics import average_precision_tied
from evaluation.classification.thresholds import youden_xray

CONFIG = os.path.join(os.path.dirname(__file__), 'configs', 'xray_mappings.json')


def sig(x):
    return 1 / (1 + np.exp(-x))


def _auc(y, x):
    _, inv, c = np.unique(x, return_inverse=True, return_counts=True)
    r = (np.cumsum(c) - (c - 1) / 2)[inv]; n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _crg(P, Y):
    m = Y >= 0; P = P[m]; Y = Y[m] == 1; T = Y.size; A = Y.sum()
    TP = (P & Y).sum(); FN = (~P & Y).sum(); FP = (P & ~Y).sum()
    w = (T - A) / (2 * A); s = w * TP - w * FN - FP; sm = A * w
    return float(sm / (2 * sm - s))


def point_metrics(S, P, Y, min_pos=1):
    a, pr, re, sp, ba, used = [], [], [], [], [], []
    for j in range(Y.shape[1]):
        k = Y[:, j] >= 0; y = Y[k, j] == 1; p = P[k, j]
        if y.sum() < min_pos or (~y).sum() < min_pos:
            continue
        a.append(_auc(y.astype(int), S[k, j])); tp = (p & y).sum(); se = tp / y.sum(); spc = (~p & ~y).sum() / (~y).sum()
        re.append(se); sp.append(spc); ba.append((se + spc) / 2); pr.append(tp / p.sum() if p.sum() else 0.); used.append(j)
    pt = dict(AUROC=np.mean(a), Precision=np.mean(pr), Recall=np.mean(re), Specificity=np.mean(sp), BA=np.mean(ba), CRG=_crg(P[:, used], Y[:, used]))
    return pt, used, dict(AUROC=a, BA=ba, Recall=re, Specificity=sp, Precision=pr)


def boot_metrics(S, P, Y, used, B, seed=0):
    """(B, 6) bootstrap draws of AUROC, Precision, Recall, Specificity, BA, CRG (case multiplicity weights)."""
    n = len(Y); rng = np.random.default_rng(seed); prep = []
    for j in used:
        k = np.where(Y[:, j] >= 0)[0]; o = k[np.argsort(S[k, j], kind='mergesort')]; s = S[o, j]; y = (Y[o, j] == 1).astype(float)
        st = np.r_[0, np.flatnonzero(s[1:] != s[:-1]) + 1]
        prep.append((o, y, st, k, Y[k, j] == 1, P[k, j]))
    out = np.full((B, 6), np.nan)
    for b in range(B):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        A, PR, RE, SP, BA = [], [], [], [], []; TP = FN = FP = TT = AA = 0.
        for o, y, st, k, yy, pp in prep:
            wo = w[o]; pw = np.add.reduceat(wo * y, st); nw = np.add.reduceat(wo * (1 - y), st); Pn = pw.sum(); Nn = nw.sum()
            if Pn == 0 or Nn == 0:
                continue
            A.append(float((pw * (np.cumsum(nw) - nw + .5 * nw)).sum() / (Pn * Nn)))
            wk = w[k]; tp = (wk * (pp & yy)).sum(); fp = (wk * (pp & ~yy)).sum(); fn = (wk * (~pp & yy)).sum(); tn = (wk * (~pp & ~yy)).sum()
            se = tp / (tp + fn); spc = tn / (tn + fp); RE.append(se); SP.append(spc); BA.append((se + spc) / 2); PR.append(tp / (tp + fp) if tp + fp else 0.)
            TP += tp; FN += fn; FP += fp; TT += wk.sum(); AA += tp + fn
        if not A:
            continue
        wc = (TT - AA) / (2 * AA); sm = AA * wc; sc = wc * TP - wc * FN - FP
        out[b] = [np.mean(A), np.mean(PR), np.mean(RE), np.mean(SP), np.mean(BA), sm / (2 * sm - sc)]
    return out


def macro_auprc(S, Y, B=1000, seed=0):
    """Macro AUPRC with a per-label stratified bootstrap interval."""
    rng = np.random.default_rng(seed); pt = []; bs = []
    for j in range(Y.shape[1]):
        k = np.where(Y[:, j] >= 0)[0]; y = Y[k, j] == 1
        if y.sum() < 1 or (~y).sum() < 1:
            continue
        o = k[np.argsort(-S[k, j], kind='mergesort')]; s_ = S[o, j]; yy = (Y[o, j] == 1).astype(np.float32)
        st = np.r_[0, np.flatnonzero(s_[1:] != s_[:-1]) + 1]

        def ap(w):
            pw = np.add.reduceat(w * yy, st, axis=-1); nw = np.add.reduceat(w * (1 - yy), st, axis=-1); tp = np.cumsum(pw, -1); fp = np.cumsum(nw, -1)
            return (pw * tp / np.maximum(tp + fp, 1e-12)).sum(-1) / np.maximum(pw.sum(-1), 1e-12)
        pt.append(float(ap(np.ones(len(o), np.float32))))
        ip = np.flatnonzero(yy == 1); ineg = np.flatnonzero(yy == 0); W = np.zeros((B, len(o)), np.float32)
        W[:, ip] = rng.multinomial(len(ip), np.full(len(ip), 1 / len(ip)), size=B)
        W[:, ineg] = rng.multinomial(len(ineg), np.full(len(ineg), 1 / len(ineg)), size=B)
        bs.append(ap(W))
    m = np.mean(np.stack(bs, 1), 1)
    return float(np.mean(pt)), [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def evaluate(S, Y, cut, names, min_pos=1, B=1000):
    P = S >= cut[None]
    point, used, per = point_metrics(S, P, Y, min_pos)
    D = boot_metrics(S, P, Y, used, B)
    ci = {k: [float(np.nanpercentile(D[:, n], 2.5)), float(np.nanpercentile(D[:, n], 97.5))] for n, k in enumerate(point)}
    res = dict(n_cases=int(len(Y)), n_findings=len(used), **{k: float(v) for k, v in point.items()}, ci95=ci,
               per_finding={names[j]: dict(pos=int((Y[:, j] == 1).sum()), cut=float(cut[j]), **{m: float(per[m][n]) for m in per}) for n, j in enumerate(used)})
    res['AUPRC'], res['ci95']['AUPRC'] = macro_auprc(S, Y, B)
    return res


def composite_probs(logits, classes, mapping, names):
    ix = {c: i for i, c in enumerate(classes)}; p = sig(np.asarray(logits, np.float64))
    return np.stack([p[:, [ix[c] for c in mapping[k]]].max(1) if mapping[k] else np.zeros(len(p)) for k in names], 1)


def fora_scores(val: dict, train: dict, mapping: dict, names, ids, study_of=None):
    """Composite scores of FORA on ``ids`` and training Youden cut-offs per benchmark label."""
    classes = [str(c) for c in val['classes']]
    lg = val['logits']; vid = [str(i) for i in val['ids']]
    if study_of is not None:   # study bag: one prediction per study, assigned to each image
        d = dict(zip(vid, lg)); lg = np.stack([d[study_of(i)] for i in ids])
    else:
        r = {u: i for i, u in enumerate(vid)}; lg = lg[[r[i] for i in ids]]
    S = composite_probs(lg, classes, mapping, names)
    St = composite_probs(train['logits'], [str(c) for c in train['classes']], mapping, names)
    ix = {c: i for i, c in enumerate(classes)}; Ytr = np.asarray(train['labels']).astype(np.int64)
    cut = np.array([youden_xray(St[:, j], Ytr[:, [ix[c] for c in mapping[k]]].max(1)) if mapping[k] else np.inf for j, k in enumerate(names)])
    return S, cut


def mapping_for(key: str) -> dict:
    m = json.load(open(CONFIG))[key]
    if key == 'padchest_gr24':
        return {k: v['fora_classes'] for k, v in m.items()}
    if key == 'padchest_gr45':
        return m['mapping']
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--truth', required=True); ap.add_argument('--mapping', required=True, help='key of configs/xray_mappings.json')
    ap.add_argument('--fora'); ap.add_argument('--fora-train'); ap.add_argument('--study-bag', action='store_true', help='Open-I study bag ids')
    ap.add_argument('--pred', action='append', default=[]); ap.add_argument('--boot', type=int, default=1000); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    T = json.load(open(a.truth)); names, ids, Y = T['names'], T['ids'], np.array(T['Y'], np.int64)
    out = dict(n_cases=len(ids), models={})
    if a.fora:
        from evaluation.benchmarks.xray_readers import openi_study
        val = dict(np.load(a.fora, allow_pickle=False)); tr = dict(np.load(a.fora_train, allow_pickle=False))
        if a.study_bag:
            val['ids'] = np.array([openi_study(str(i)) if '_IM' in str(i) else str(i).split('/')[-1] for i in val['ids']])
        S, cut = fora_scores(val, tr, mapping_for(a.mapping), names, ids, openi_study if a.study_bag else None)
        out['models']['FORA'] = evaluate(S, Y, cut, names, 1, a.boot)
    for spec in a.pred:
        name, path = spec.split('=', 1); z = np.load(path, allow_pickle=False)
        r = {str(u): i for i, u in enumerate(z['case_ids'])}; rows = [r[i] for i in ids]
        pf = [str(f) for f in z['findings']]; cols = [pf.index(k) for k in names]
        S = z['scores'][rows][:, cols].astype(np.float64)
        cut = z['cutoff'][cols] if 'cutoff' in z.files else np.full(len(names), 0.5)
        if 'decision' in z.files:   # native yes/no: decision is the score, cut 0.5
            S = z['decision'][rows][:, cols].astype(np.float64); cut = np.full(len(names), 0.5)
        out['models'][name] = evaluate(S, Y, cut, names, 1, a.boot)
    json.dump(out, open(a.out, 'w'), indent=1)
    for n, r in out['models'].items():
        print(n, {k: round(r[k], 4) for k in ('AUROC', 'AUPRC', 'BA', 'Recall', 'Specificity', 'Precision', 'CRG')})


if __name__ == '__main__':
    main()

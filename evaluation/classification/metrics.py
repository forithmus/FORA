"""Finding-recognition metrics (paper Methods 4.9): AUROC, AUPRC, operating-point metrics, balanced accuracy and CRG.

All functions take a score vector ``s`` and a label vector ``y`` with 1 = positive, 0 = negative and -1 = not
evaluable (excluded).

* ``auroc``: Mann-Whitney statistic with mid-ranks for ties.
* ``average_precision``: mean precision at the ranks of the positives in descending score order, stable sort, no
  tie grouping (used for CT).
* ``average_precision_tied``: the scikit-learn definition with ties grouped at distinct thresholds (used for X-ray).
* ``operating_point``: sensitivity, specificity, balanced accuracy and precision of binary decisions.
* ``crg``: CRG pooled over all decisions of a (studies x findings) matrix (the same formula is used for X-ray and
  for report generation).
"""
from __future__ import annotations

import numpy as np


def _valid(s: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(s, np.float64)
    y = np.asarray(y)
    m = (y >= 0) & np.isfinite(s)
    return s[m], (y[m] == 1)


def auroc(s, y) -> float:
    """AUROC with ties counted 1/2. NaN when one class is empty."""
    s, y = _valid(s, y)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float('nan')
    o = np.argsort(s, kind='mergesort')
    _, inv, cnt = np.unique(s[o], return_inverse=True, return_counts=True)
    r = np.empty(len(s))
    r[o] = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def average_precision(s, y) -> float:
    """CT package AUPRC: mean over positives of precision at their rank (descending score, stable order)."""
    s, y = _valid(s, y)
    if y.sum() == 0 or (~y).sum() == 0:
        return float('nan')
    o = np.argsort(-s, kind='mergesort')
    yy = y[o]
    tp = np.cumsum(yy)
    return float((tp[yy] / np.arange(1, len(yy) + 1)[yy]).mean())


def average_precision_tied(s, y) -> float:
    """X-ray package AUPRC (scikit-learn definition, ties grouped). NaN when one class is empty."""
    s, y = _valid(s, y)
    if y.sum() == 0 or y.all():
        return float('nan')
    o = np.argsort(-s, kind='mergesort')
    s = s[o]; yy = y[o].astype(float)
    st = np.r_[0, np.flatnonzero(s[1:] != s[:-1]) + 1]
    pw = np.add.reduceat(yy, st); nw = np.add.reduceat(1 - yy, st)
    tp = np.cumsum(pw); fp = np.cumsum(nw)
    return float((pw * tp / (tp + fp)).sum() / pw.sum())


def operating_point(decision, y) -> dict:
    """Confusion counts and rates of binary decisions; label -1 excluded."""
    y = np.asarray(y)
    m = y >= 0
    p = np.asarray(decision, bool)[m]; t = y[m] == 1
    tp = int((p & t).sum()); fp = int((p & ~t).sum()); fn = int((~p & t).sum()); tn = int((~p & ~t).sum())
    se = tp / max(tp + fn, 1); sp = tn / max(tn + fp, 1); ppv = tp / max(tp + fp, 1)
    return dict(sens=se, spec=sp, ba=(se + sp) / 2, ppv=ppv, tp=tp, fp=fp, fn=fn, tn=tn)


def balanced_accuracy(decision, y) -> float:
    return operating_point(decision, y)['ba']


def crg(P, Y) -> tuple[float, int, int]:
    """CRG pooled over every evaluable decision of the matrices ``P`` (bool) and ``Y`` (1/0/-1).

    Returns (CRG, T = number of decisions, A = number of positive decisions). All-negative and all-positive
    decisions give 1/3.
    """
    Y = np.asarray(Y)
    m = Y >= 0
    P = np.asarray(P, bool)[m]; Y = Y[m] == 1
    T = Y.size; A = int(Y.sum())
    TP = int((P & Y).sum()); FN = int((~P & Y).sum()); FP = int((P & ~Y).sum())
    w = (T - A) / (2 * A)
    s = w * TP - w * FN - FP
    smax = A * w
    return float(smax / (2 * smax - s)), int(T), A


def crg_from_counts(tp: float, fn: float, fp: float, total: float, positives: float) -> float:
    """CRG from pooled (possibly weighted) counts; same formula as ``crg``."""
    w = (total - positives) / (2 * positives)
    s = w * tp - w * fn - fp
    smax = positives * w
    return float(smax / (2 * smax - s))


def macro(fn, S, Y, cols=None) -> float:
    """Mean of ``fn(S[:, j], Y[:, j])`` over columns (NaN columns ignored)."""
    cols = range(Y.shape[1]) if cols is None else cols
    return float(np.nanmean([fn(S[:, j], Y[:, j]) for j in cols]))

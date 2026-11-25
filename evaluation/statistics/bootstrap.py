"""Resampling intervals and paired tests (paper Methods 4.9-4.11).

* Paired cluster bootstrap: patients (CT-RATE) or studies (other sets) are drawn with replacement; every model is
  scored on the same draws, so differences are paired. Used with 200,000 draws for the public CT benchmarks.
* Case bootstrap: clusters = cases (KiTS23, RibFrac, the RADAR comparison on Merlin: 2,000 draws), with a
  two-sided normal-approximation test of the difference.
* Report-generation bootstrap: clusters = patients on CT-RATE and studies elsewhere (1,000 draws for intervals,
  2,000 for paired tests).

A resample is a vector of multiplicities (``weights``) over the original rows. Statistics are functions
``stat(W) -> (B,)`` or ``(B, m)`` of a weight matrix ``W`` of shape (B, n), which lets AUROC, balanced accuracy
and the report-generation metrics be computed for many resamples at once (``weighted_auroc``, ``weighted_ba``).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import norm


def cluster_weights(n: int, draws: int, seed: int = 0, clusters=None) -> np.ndarray:
    """(draws, n) multiplicities. Each draw resamples rows (``clusters=None``) or whole clusters."""
    rng = np.random.default_rng(seed)
    if clusters is None:
        return np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(draws)]).astype(np.float32)
    _, inv = np.unique(np.asarray(clusters), return_inverse=True)
    inv = inv.ravel(); k = inv.max() + 1
    return np.stack([np.bincount(rng.integers(0, k, k), minlength=k)[inv] for _ in range(draws)]).astype(np.float32)


def iter_weights(n: int, draws: int, seed: int = 0, clusters=None, chunk: int = 1000):
    """Chunks of ``cluster_weights`` from one random stream (memory-bounded for 200,000 draws)."""
    rng = np.random.default_rng(seed)
    if clusters is not None:
        _, inv = np.unique(np.asarray(clusters), return_inverse=True); inv = inv.ravel(); k = inv.max() + 1
    for a in range(0, draws, chunk):
        b = min(draws, a + chunk)
        if clusters is None:
            yield np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(b - a)]).astype(np.float32)
        else:
            yield np.stack([np.bincount(rng.integers(0, k, k), minlength=k)[inv] for _ in range(b - a)]).astype(np.float32)


def weighted_auroc(s, y, W=None):
    """AUROC (ties 1/2) for every row of weights W (B, n); returns (point, boot or None)."""
    s = np.asarray(s, np.float64); y = np.asarray(y).astype(bool)
    order = np.argsort(s, kind='mergesort'); ss = s[order]; yo = y[order]
    starts = np.flatnonzero(np.r_[True, np.diff(ss) != 0])

    def _auc(w):
        P = np.add.reduceat(w * yo, starts, axis=1); N = np.add.reduceat(w * ~yo, starts, axis=1)
        cumN = np.cumsum(N, axis=1) - N
        num = (P * (cumN + 0.5 * N)).sum(1); den = P.sum(1) * N.sum(1)
        with np.errstate(invalid='ignore', divide='ignore'):
            return num / den
    pt = float(_auc(np.ones((1, len(s)), np.float32))[0])
    if W is None:
        return pt, None
    out = np.empty(W.shape[0])
    for i in range(0, W.shape[0], 250):
        out[i:i + 250] = _auc(W[i:i + 250][:, order])
    return pt, out


def weighted_ba(d, y, W=None):
    """Balanced accuracy of decisions d for every row of weights W; returns (point, boot or None)."""
    d = np.asarray(d).astype(bool); y = np.asarray(y).astype(bool)

    def _ba(w):
        tp = w @ (y & d).astype(np.float32); p = w @ y.astype(np.float32)
        tn = w @ (~y & ~d).astype(np.float32); n = w @ (~y).astype(np.float32)
        with np.errstate(invalid='ignore', divide='ignore'):
            return 0.5 * (tp / p + tn / n)
    pt = float(_ba(np.ones((1, len(y)), np.float32))[0])
    return pt, (None if W is None else _ba(W))


def percentile_ci(boot, level: float = 0.95) -> list[float]:
    b = np.asarray(boot, float); b = b[np.isfinite(b)]
    q = 100 * (1 - level) / 2
    return [float(np.percentile(b, q)), float(np.percentile(b, 100 - q))] if len(b) else [float('nan')] * 2


def two_sided_p(diff_boot) -> float:
    """2 * min(P(D* <= 0), P(D* >= 0)) over bootstrap draws of a difference."""
    d = np.asarray(diff_boot, float); d = d[np.isfinite(d)]
    return float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))


def normal_p(diff: float, diff_boot) -> float:
    """Two-sided normal approximation: z = diff / SD(bootstrap differences)."""
    sd = float(np.nanstd(diff_boot, ddof=1))
    return float(2 * norm.sf(abs(diff) / sd)) if sd > 0 else (0.0 if diff else 1.0)


def paired_bootstrap(stat_a: Callable, stat_b: Callable, n: int, draws: int, seed: int = 0, clusters=None,
                     chunk: int = 1000, test: str = 'percentile') -> dict:
    """Paired bootstrap of two statistics on shared draws.

    ``stat_x(W)`` returns one value per row of W; ``stat_x(None)`` returns the point estimate. ``test`` is
    'percentile' (two-sided bootstrap p) or 'normal' (two-sided normal approximation).
    """
    a = []; b = []
    for W in iter_weights(n, draws, seed, clusters, chunk):
        a.append(np.asarray(stat_a(W), float)); b.append(np.asarray(stat_b(W), float))
    a = np.concatenate(a); b = np.concatenate(b)
    pa, pb = float(stat_a(None)), float(stat_b(None)); d = a - b
    p = two_sided_p(d) if test == 'percentile' else normal_p(pa - pb, d)
    return dict(a=dict(value=pa, ci95=percentile_ci(a)), b=dict(value=pb, ci95=percentile_ci(b)),
                diff=dict(value=pa - pb, ci95=percentile_ci(d)), p_two_sided=p, draws=draws)


def bootstrap_ci(stat: Callable, n: int, draws: int, seed: int = 0, clusters=None, chunk: int = 1000) -> dict:
    v = np.concatenate([np.asarray(stat(W), float) for W in iter_weights(n, draws, seed, clusters, chunk)])
    return dict(value=float(stat(None)), ci95=percentile_ci(v), draws=draws)


def macro_auroc_stat(S: np.ndarray, Y: np.ndarray, cols=None) -> Callable:
    """stat(W) = macro AUROC over columns (label -1 excluded, NaN when a resample lacks one class)."""
    cols = list(range(Y.shape[1])) if cols is None else list(cols)
    prep = []
    for j in cols:
        k = (Y[:, j] >= 0) & np.isfinite(S[:, j])
        prep.append((np.flatnonzero(k), S[k, j], Y[k, j] == 1))

    def stat(W):
        if W is None:
            return float(np.nanmean([weighted_auroc(s, y)[0] for _, s, y in prep]))
        return np.nanmean(np.stack([weighted_auroc(s, y, W[:, r])[1] for r, s, y in prep], 1), 1)
    return stat


def macro_ba_stat(D: np.ndarray, Y: np.ndarray, cols=None) -> Callable:
    cols = list(range(Y.shape[1])) if cols is None else list(cols)
    prep = [(np.flatnonzero(Y[:, j] >= 0), D[Y[:, j] >= 0, j], Y[Y[:, j] >= 0, j] == 1) for j in cols]

    def stat(W):
        if W is None:
            return float(np.nanmean([weighted_ba(d, y)[0] for _, d, y in prep]))
        return np.nanmean(np.stack([weighted_ba(d, y, W[:, r])[1] for r, d, y in prep], 1), 1)
    return stat

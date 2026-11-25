"""First-order linearization intervals with patient-level multiplier draws (paper Methods 4.9).

Every metric is written as a function of study weights ``w`` (all ones for the observed data). Its first-order
expansion around ``w = 1`` is ``theta(w) ~ theta + sum_i (w_i - 1) phi_i`` with ``phi_i = d theta / d w_i``.
Studies of one patient share one multiplier, so the patient influence is ``phi_p = sum_{i in p} phi_i``.
A multiplier draw is ``theta* = theta + sum_p xi_p phi_p`` with independent standard normal ``xi_p`` (200,000
draws, seed 0 by default); the 95%
interval is given by the 2.5% and 97.5% quantiles of the draws. A paired comparison of two models uses the same
``xi`` for both, so the difference has influence ``phi_A - phi_B``; the two-sided p-value is
``2 * min(P(D* <= 0), P(D* >= 0))`` over the draws of the difference.

Influence functions are exact derivatives of the weighted estimators:

* AUROC: weighted Mann-Whitney statistic with ties counted 1/2.
* Average precision (``average_precision`` definition: precision at each positive, descending order).
* Ratio metrics (sensitivity, specificity, balanced accuracy, precision) and pooled CRG through the gradient of a
  smooth function of weighted sums.

A macro average over findings (or (finding, scan type) cells) has the mean of the per-cell influences, each cell
contributing only through its own studies.
"""
from __future__ import annotations

import numpy as np

DEFAULT_DRAWS = 200_000


# ----------------------------------------------------------------------------- per-metric influence values
def auroc_influence(s: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """AUROC and d AUROC / d w_i for every study (y in {0, 1})."""
    s = np.asarray(s, np.float64); y = np.asarray(y).astype(bool)
    P = y.sum(); N = (~y).sum()
    o = np.argsort(s, kind='mergesort'); ss = s[o]; yo = y[o]
    starts = np.flatnonzero(np.r_[True, np.diff(ss) != 0])
    gid = np.cumsum(np.r_[True, np.diff(ss) != 0]) - 1
    gp = np.add.reduceat(yo.astype(float), starts); gn = np.add.reduceat((~yo).astype(float), starts)
    neg_below = np.cumsum(gn) - gn; pos_above = P - np.cumsum(gp)
    h_pos = neg_below[gid] + 0.5 * gn[gid]          # for a positive: negatives below + 1/2 ties
    h_neg = pos_above[gid] + 0.5 * gp[gid]          # for a negative: positives above + 1/2 ties
    u = (h_pos * yo).sum()
    auc = u / (P * N)
    phi_o = np.where(yo, h_pos / (P * N) - auc / P, h_neg / (P * N) - auc / N)
    phi = np.empty_like(phi_o); phi[o] = phi_o
    return float(auc), phi


def ap_influence(s: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Average precision (no tie grouping, stable descending order) and its weight derivative.

    With the rank order fixed, AP(w) = sum_k w_k y_k TP_k / D_k / P, where TP_k and D_k are the weighted positives
    and studies ranked at or above k.
    """
    s = np.asarray(s, np.float64); y = np.asarray(y).astype(float)
    o = np.argsort(-s, kind='mergesort'); yo = y[o]
    D = np.arange(1, len(yo) + 1, dtype=float); TP = np.cumsum(yo); P = yo.sum()
    prec = TP / D
    A = (yo * prec).sum()
    ap = A / P
    # d A / d w_i = y_i prec_i + sum_{k >= i in rank} y_k (y_i - prec_k) / D_k
    t1 = yo / D; t2 = yo * prec / D
    c1 = np.cumsum(t1[::-1])[::-1]; c2 = np.cumsum(t2[::-1])[::-1]
    dA = yo * prec + yo * c1 - c2
    phi_o = dA / P - A * yo / P ** 2
    phi = np.empty_like(phi_o); phi[o] = phi_o
    return float(ap), phi


def smooth_influence(g, X: np.ndarray) -> tuple[float, np.ndarray]:
    """theta = g(S) with S = X.sum(0) (weighted sums); phi_i = grad g(S) . X_i (central differences)."""
    X = np.asarray(X, np.float64); S = X.sum(0)
    grad = np.zeros(len(S))
    for k in range(len(S)):
        h = 1e-6 * max(abs(S[k]), 1.0)
        e = np.zeros(len(S)); e[k] = h
        grad[k] = (g(S + e) - g(S - e)) / (2 * h)
    return float(g(S)), X @ grad


def ba_influence(d: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    d = np.asarray(d, bool); y = np.asarray(y, bool)
    X = np.stack([y & d, y, ~y & ~d, ~y], 1).astype(float)
    return smooth_influence(lambda S: 0.5 * (S[0] / S[1] + S[2] / S[3]), X)


def sens_influence(d, y):
    d = np.asarray(d, bool); y = np.asarray(y, bool)
    return smooth_influence(lambda S: S[0] / S[1], np.stack([y & d, y], 1).astype(float))


def spec_influence(d, y):
    d = np.asarray(d, bool); y = np.asarray(y, bool)
    return smooth_influence(lambda S: S[0] / S[1], np.stack([~y & ~d, ~y], 1).astype(float))


def precision_influence(d, y):
    d = np.asarray(d, bool); y = np.asarray(y, bool)
    return smooth_influence(lambda S: S[0] / max(S[1], 1e-12), np.stack([y & d, d], 1).astype(float))


def _crg(S):
    tp, fn, fp, T, A = S
    w = (T - A) / (2 * A); s = w * tp - w * fn - fp; sm = A * w
    return sm / (2 * sm - s)


def crg_influence(P: np.ndarray, Y: np.ndarray) -> tuple[float, np.ndarray]:
    """Pooled CRG over a (studies x findings) decision matrix; influence per study (row)."""
    P = np.asarray(P, bool); Y = np.asarray(Y); m = Y >= 0; Yb = Y == 1
    X = np.stack([(P & Yb & m).sum(1), (~P & Yb & m).sum(1), (P & ~Yb & m).sum(1), m.sum(1), (Yb & m).sum(1)], 1).astype(float)
    return smooth_influence(_crg, X)


METRICS = {'AUROC': ('score', auroc_influence), 'AUPRC': ('score', ap_influence), 'BA': ('decision', ba_influence),
           'sensitivity': ('decision', sens_influence), 'specificity': ('decision', spec_influence),
           'precision': ('decision', precision_influence)}


def macro_influence(metric: str, S: np.ndarray, D: np.ndarray, Y: np.ndarray, cells) -> tuple[float, np.ndarray]:
    """Macro mean of ``metric`` over ``cells`` = [(column, row indices)], with its per-study influence."""
    kind, fn = METRICS[metric]
    n = Y.shape[0]; phi = np.zeros(n); vals = []
    for j, rows in cells:
        y = Y[rows, j]; ok = y >= 0
        r = np.asarray(rows)[ok]
        x = (S if kind == 'score' else D)[r, j]
        if kind == 'score' and not np.isfinite(x).all():
            continue
        v, f = fn(x, y[ok] == 1)
        if not np.isfinite(v):
            continue
        vals.append(v); np.add.at(phi, r, f)
    k = len(vals)
    return float(np.mean(vals)), phi / max(k, 1)


# ----------------------------------------------------------------------------- multiplier draws
def cluster_influence(phi: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    _, inv = np.unique(np.asarray(clusters), return_inverse=True)
    return np.bincount(inv.ravel(), weights=phi)


def multiplier_draws(phis: np.ndarray, draws: int = DEFAULT_DRAWS, seed: int = 0, chunk: int = 2000) -> np.ndarray:
    """``(draws, m)`` draws of sum_p xi_p phi_p for ``phis`` of shape (n_clusters, m), shared xi across columns."""
    phis = np.atleast_2d(np.asarray(phis, np.float64).T).T
    rng = np.random.default_rng(seed)
    out = np.empty((draws, phis.shape[1]))
    for a in range(0, draws, chunk):
        b = min(draws, a + chunk)
        out[a:b] = rng.standard_normal((b - a, phis.shape[0])) @ phis
    return out


def interval(theta: float, delta: np.ndarray, level: float = 0.95) -> list[float]:
    q = (1 - level) / 2
    return [float(theta + np.quantile(delta, q)), float(theta + np.quantile(delta, 1 - q))]


def paired_p(diff: float, delta: np.ndarray) -> float:
    d = diff + delta
    return float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))


def linearized_ci(theta: float, phi: np.ndarray, clusters: np.ndarray, draws: int = DEFAULT_DRAWS, seed: int = 0) -> dict:
    """Interval of one metric from its per-study influence values."""
    dlt = multiplier_draws(cluster_influence(phi, clusters)[:, None], draws, seed)[:, 0]
    return dict(value=theta, ci95=interval(theta, dlt), se=float(np.sqrt((cluster_influence(phi, clusters) ** 2).sum())))


def paired_comparison(theta_a, phi_a, theta_b, phi_b, clusters, draws: int = DEFAULT_DRAWS, seed: int = 0) -> dict:
    """Intervals of A, B and A - B from shared multipliers, with the two-sided paired p-value."""
    C = np.stack([cluster_influence(phi_a, clusters), cluster_influence(phi_b, clusters)], 1)
    dl = multiplier_draws(np.c_[C, C[:, 0] - C[:, 1]], draws, seed)
    diff = theta_a - theta_b
    return dict(a=dict(value=theta_a, ci95=interval(theta_a, dl[:, 0])), b=dict(value=theta_b, ci95=interval(theta_b, dl[:, 1])),
                diff=dict(value=diff, ci95=interval(diff, dl[:, 2])), p_two_sided=paired_p(diff, dl[:, 2]))


def heldout_intervals(models: dict, Y: np.ndarray, cells, clusters: np.ndarray, draws: int = DEFAULT_DRAWS, seed: int = 0,
                      metrics=('AUROC', 'AUPRC', 'BA', 'CRG')) -> dict:
    """Intervals for every model and paired tests of the first model against each other model.

    ``models`` = {name: (scores, decisions)} aligned to ``Y``; ``cells`` = [(scan_type, column, rows)].
    All intervals and differences use one shared set of multiplier draws. P-values are Holm-adjusted across all
    comparisons returned here.
    """
    from evaluation.statistics.multiple_testing import holm
    cl = [(j, r) for _, j, r in cells]
    names = list(models); ref = names[0]
    theta, cols, keys = {}, [], []
    for name, (S, D) in models.items():
        for m in metrics:
            v, phi = crg_influence(D, Y) if m == 'CRG' else macro_influence(m, S, D, Y, cl)
            theta[(name, m)] = v; cols.append(cluster_influence(phi, clusters)); keys.append((name, m))
    base = len(cols)
    for other in names[1:]:
        for m in metrics:
            cols.append(cols[keys.index((ref, m))] - cols[keys.index((other, m))]); keys.append((ref, other, m))
    C = np.stack(cols, 1)
    dl = multiplier_draws(C, draws, seed)
    out = {'models': {n: {} for n in names}, 'comparisons': [], 'draws': draws}
    for k, key in enumerate(keys):
        if k < base:
            out['models'][key[0]][key[1]] = dict(value=theta[key], ci95=interval(theta[key], dl[:, k]), se=float(np.sqrt((C[:, k] ** 2).sum())))
        else:
            d = theta[(key[0], key[2])] - theta[(key[1], key[2])]
            out['comparisons'].append(dict(model=key[0], versus=key[1], metric=key[2], diff=dict(value=d, ci95=interval(d, dl[:, k])),
                                           p_two_sided=paired_p(d, dl[:, k])))
    for c, p in zip(out['comparisons'], holm([c['p_two_sided'] for c in out['comparisons']])):
        c['p_holm'] = p
    return out

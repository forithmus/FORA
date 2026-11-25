"""Multiple-testing correction and exact binomial intervals (paper Methods 4.9-4.10).

* ``holm``: Holm step-down adjusted p-values (Holm 1979), applied across a family of comparisons (for example the
  internal comparisons of FORA against each model, all public-benchmark comparisons, or one figure panel).
* ``clopper_pearson``: exact binomial 95% interval of a hit rate (Clopper and Pearson 1934).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import beta


def holm(pvalues) -> list[float]:
    """Holm-adjusted p-values in the input order."""
    p = np.asarray(pvalues, float); m = len(p)
    if m == 0:
        return []
    o = np.argsort(p, kind='mergesort')
    adj = np.maximum.accumulate((m - np.arange(m)) * p[o])
    out = np.empty(m); out[o] = np.minimum(adj, 1.0)
    return out.tolist()


def clopper_pearson(hits: int, n: int, level: float = 0.95) -> list[float]:
    a = (1 - level) / 2
    lo = float(beta.ppf(a, hits, n - hits + 1)) if hits else 0.0
    hi = float(beta.ppf(1 - a, hits + 1, n - hits)) if hits < n else 1.0
    return [lo, hi]

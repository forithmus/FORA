"""Statistics library: influence functions, interval coverage on synthetic data, Holm and exact binomial."""
import numpy as np
import pytest
from scipy.stats import binomtest

from evaluation.statistics.bootstrap import macro_auroc_stat, paired_bootstrap, weighted_auroc
from evaluation.statistics.linearization import (ap_influence, auroc_influence, ba_influence, crg_influence, linearized_ci,
                                                  multiplier_draws, paired_comparison)
from evaluation.statistics.multiple_testing import clopper_pearson, holm


def _weighted_ap(s, y, w):
    o = np.argsort(-s, kind='mergesort'); yo = y[o].astype(float); wo = w[o]
    tp = np.cumsum(wo * yo); d = np.cumsum(wo)
    return (wo * yo * tp / d).sum() / (wo * yo).sum()


def _fd(f, n, i, h=1e-6):
    w = np.ones(n); w[i] += h; a = f(w); w[i] -= 2 * h; return (a - f(w)) / (2 * h)


def test_influences_match_finite_differences():
    rng = np.random.default_rng(0); n = 60; y = rng.integers(0, 2, n).astype(bool); s = np.round(rng.normal(y, 1), 1)
    _, phi = auroc_influence(s, y)
    f = lambda w: weighted_auroc(s, y, w[None].astype(np.float64))[1][0]
    for i in range(0, n, 7):
        assert phi[i] == pytest.approx(_fd(f, n, i), abs=1e-6)
    s2 = rng.normal(y, 1); _, phi = ap_influence(s2, y)
    for i in range(0, n, 7):
        assert phi[i] == pytest.approx(_fd(lambda w: _weighted_ap(s2, y, w), n, i), abs=1e-6)
    d = rng.random(n) < 0.5; _, phi = ba_influence(d, y)
    ba = lambda w: 0.5 * ((w * (y & d)).sum() / (w * y).sum() + (w * (~y & ~d)).sum() / (w * ~y).sum())
    for i in range(0, n, 7):
        assert phi[i] == pytest.approx(_fd(ba, n, i), abs=1e-6)


def test_crg_influence_point():
    Y = np.array([[1, 0], [0, 0], [0, 1], [1, 1]]); P = np.array([[1, 0], [0, 1], [0, 1], [1, 0]], bool)
    v, phi = crg_influence(P, Y)
    from evaluation.classification.metrics import crg
    assert v == pytest.approx(crg(P, Y)[0]) and phi.shape == (4,)


def test_linearization_coverage_auroc():
    """95% intervals cover the true AUROC of a binormal model in roughly 95% of synthetic datasets."""
    from scipy.stats import norm
    rng = np.random.default_rng(1); true = norm.cdf(1 / np.sqrt(2)); hit = 0; R = 200
    for r in range(R):
        y = rng.random(300) < 0.3; s = rng.normal(y.astype(float), 1)
        v, phi = auroc_influence(s, y)
        lo, hi = linearized_ci(v, phi, np.arange(300), draws=2000, seed=r)['ci95']
        hit += lo <= true <= hi
    assert 0.88 <= hit / R <= 0.99


def test_patient_clustering_widens_intervals():
    rng = np.random.default_rng(2); n_p = 150
    pid = np.repeat(np.arange(n_p), 2); yp = rng.random(n_p) < 0.4; y = yp[pid]; s = rng.normal(yp.astype(float), 1)[pid] + rng.normal(0, .1, 2 * n_p)
    v, phi = auroc_influence(s, y)
    se_cl = linearized_ci(v, phi, pid, draws=4000)['se']; se_iid = linearized_ci(v, phi, np.arange(2 * n_p), draws=4000)['se']
    assert se_cl > 1.2 * se_iid


def test_linearization_agrees_with_bootstrap():
    rng = np.random.default_rng(3); y = rng.random(400) < 0.3; s = rng.normal(y.astype(float), 1)
    v, phi = auroc_influence(s, y)
    se_lin = linearized_ci(v, phi, np.arange(400), draws=20000)['se']
    stat = macro_auroc_stat(s[:, None], y[:, None].astype(int))
    from evaluation.statistics.bootstrap import iter_weights
    b = np.concatenate([stat(W) for W in iter_weights(400, 2000, 0)])
    assert se_lin == pytest.approx(b.std(), rel=0.15)


def test_paired_tests():
    rng = np.random.default_rng(4); y = rng.random(300) < 0.3; s = rng.normal(y.astype(float), 1)
    va, pa = auroc_influence(s, y); vb, pb = auroc_influence(rng.normal(0, 1, 300), y)
    c = paired_comparison(va, pa, vb, pb, np.arange(300), draws=5000)
    assert c['p_two_sided'] < 0.01 and c['diff']['ci95'][0] > 0
    same = paired_comparison(va, pa, va, pa, np.arange(300), draws=2000)
    assert same['p_two_sided'] == 1.0
    st = macro_auroc_stat(s[:, None], y[:, None].astype(int))
    r = paired_bootstrap(st, st, 300, 200, seed=0)
    assert r['diff']['value'] == 0 and r['p_two_sided'] == 1.0


def test_multiplier_draws_gaussian():
    phi = np.array([[0.3], [0.4]]); d = multiplier_draws(phi, 100000, seed=0)[:, 0]
    assert d.std() == pytest.approx(0.5, rel=0.02)


def test_holm_hand_example():
    assert holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])   # step-down monotonicity


@pytest.mark.parametrize('k,n', [(0, 10), (3, 10), (10, 10), (314, 1139)])
def test_clopper_pearson(k, n):
    ci = binomtest(k, n).proportion_ci(method='exact')
    assert clopper_pearson(k, n) == pytest.approx([ci.low, ci.high], abs=1e-9)

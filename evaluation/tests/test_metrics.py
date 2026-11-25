"""Classification metrics and cut-offs against scikit-learn and hand examples."""
import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from evaluation.classification.calibration import ece
from evaluation.classification.metrics import auroc, average_precision, average_precision_tied, crg, operating_point
from evaluation.classification.thresholds import composite, youden_cutoff, youden_midpoint, youden_xray


@pytest.mark.parametrize('seed', range(5))
def test_auroc_matches_sklearn(seed):
    rng = np.random.default_rng(seed); y = rng.integers(0, 2, 300); s = np.round(rng.normal(y, 1.0), 1)   # rounding -> ties
    assert auroc(s, y) == pytest.approx(roc_auc_score(y, s), abs=1e-12)


def test_auroc_ignores_unknown():
    y = np.array([1, 0, -1, 1, 0]); s = np.array([.9, .1, 5., .8, .2])
    assert auroc(s, y) == 1.0


@pytest.mark.parametrize('seed', range(5))
def test_average_precision_matches_sklearn(seed):
    rng = np.random.default_rng(seed); y = rng.integers(0, 2, 400); s = rng.normal(y, 1.0)
    assert average_precision(s, y) == pytest.approx(average_precision_score(y, s), abs=1e-12)   # no ties
    st = np.round(s, 1)
    assert average_precision_tied(st, y) == pytest.approx(average_precision_score(y, st), abs=1e-12)


def test_operating_point_and_crg():
    y = np.array([1, 1, 0, 0, 0, -1]); d = np.array([1, 0, 1, 0, 0, 1], bool)
    op = operating_point(d, y)
    assert (op['tp'], op['fn'], op['fp'], op['tn']) == (1, 1, 1, 2)
    assert op['ba'] == pytest.approx(0.5 * (0.5 + 2 / 3))
    Y = np.array([[1, 0], [0, 0], [0, 1]])
    assert crg(np.zeros_like(Y, bool), Y)[0] == pytest.approx(1 / 3)
    assert crg(np.ones_like(Y, bool), Y)[0] == pytest.approx(1 / 3)
    assert crg(Y == 1, Y)[0] == pytest.approx(1.0)


def test_youden_cutoffs():
    z = np.array([5., 4., 3., 2., 1.]); y = np.array([1, 1, 0, 0, 0])
    assert youden_midpoint(z, y) == pytest.approx(3.5)
    assert youden_xray(z, y) == 4.0                       # positive if score >= cut
    st = np.array(['chest'] * 5)
    assert youden_cutoff(z, y, st, ['abdomen']) == pytest.approx(3.5)   # no abdomen studies -> all-train fallback


def test_composite_or_target():
    S = np.array([[1., 3.], [2., 0.]]); L = np.array([[-1, 1], [-1, -1]])
    z, y = composite(S, L, ['a', 'b'], ['a', 'b'])
    assert z.tolist() == [3., 2.] and y.tolist() == [1, -1]


def test_ece_perfect_and_biased():
    rng = np.random.default_rng(0); p = rng.random(30000); y = rng.random(30000) < p
    assert ece(p, y) < 0.02
    assert ece(np.clip(p + 0.2, 0, 1), y) > 0.1

"""Localization measures on toy grids."""
import numpy as np
import pytest

from evaluation.localization.ct_metrics import aggregate, instance_metrics, uniform_route_chance
from evaluation.localization.ct_regions import CELL, equal_size_overlap, regions, tie_expected_hit, top_n
from evaluation.localization import xray as X


def _grid(n=6):
    c = np.stack(np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing='ij'), -1).reshape(-1, 3)
    return c, (c + .5) * CELL


def test_single_peak_regions_and_hit():
    cells, world = _grid(); e = np.zeros(len(cells)); k = int(np.flatnonzero((cells == [3, 3, 3]).all(1))[0]); e[k] = 1.
    r = regions(e, world)
    assert r['core'][np.flatnonzero((r['cells'] == [3, 3, 3]).all(1))[0]]
    assert np.allclose(r['focus'], world[k])
    # 1 mm voxels with RAS affine; mask covers the cell (S=3, R=3, A=3) -> voxels x in [24,32), y in [24,32), z in [9,12)
    mask = np.zeros((48, 48, 18), bool); mask[24:32, 24:32, 9:12] = True; aff = np.eye(4)
    m = instance_metrics(mask, aff, r['focus'], r['cells'], r['score'], r['core'], r['hdr80'])
    assert m['first_focus_center_inside_exact_mask'] and m['HDR50_intersects_GT_cells'] and m['HDR80_intersects_GT_cells']
    far = np.zeros_like(mask); far[0:4, 0:4, 0:2] = True
    m2 = instance_metrics(far, aff, r['focus'], r['cells'], r['score'], r['core'], r['hdr80'])
    assert not m2['first_focus_center_inside_exact_mask'] and not m2['HDR50_intersects_GT_cells']


def test_no_evidence_and_aggregate():
    cells, world = _grid(3)
    assert regions(np.zeros(len(cells)), world) is None
    rows = [dict(GT_positive=True, first_focus_center_inside_exact_mask=h, HDR50_intersects_GT_cells=h) for h in [1, 1, 0, 0]] + [dict(GT_positive=False)]
    a = aggregate(rows)
    assert a['annotated_positive_cases'] == 4 and a['first_focus_hit_rate'] == 0.5 and a['mean_HDR50_intersects_GT_cells'] == 0.5
    assert 0 < a['exact_binomial95CI'][0] < 0.5 < a['exact_binomial95CI'][1] < 1


def test_hdr_mass():
    cells, world = _grid(); rng = np.random.default_rng(0); e = rng.random(len(cells)) ** 8
    r = regions(e, world)
    sc = r['score']; tot = sc[r['allowed']].sum()
    assert sc[r['core']].sum() >= 0.5 * tot - 1e-6 and sc[r['hdr80']].sum() >= 0.8 * tot - 1e-6 and r['core'].sum() <= r['hdr80'].sum()


def test_top_n_ties_and_equal_size():
    score = np.array([3., 2., 2., 2., 1.]); allow = np.ones(5, bool)
    sel, tie = top_n(score, allow, 2)
    assert sel.tolist() == [True, True, False, False, False] and tie['boundary_tie']
    lesion = np.array([False, False, False, True, False])
    assert tie_expected_hit(score, allow, sel, tie, lesion) == pytest.approx(1 / 3)
    cells, world = _grid(4); e = np.zeros(len(cells)); e[0] = 1.
    r = regions(e, world); mask = np.zeros((32, 32, 12), bool); mask[0:2, 0:2, 0:2] = True
    o = equal_size_overlap(r, mask, np.eye(4), 1)
    assert o['n'] == 1 and o['hit']


def test_route_chance():
    c = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0], bool); routed = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0], bool)
    assert uniform_route_chance(c, routed)['uniform_routing_peak_exact_hit_expectation'] == pytest.approx(2 / 8)
    assert uniform_route_chance(c, routed[:3].tolist() + [0] * 7)['native_route_tokens'] == 10   # < 8 routed -> all tokens


def test_xray_point_dice_chance():
    E = np.zeros((8, 8)); E[2, 5] = 1.0; box = (0, 128, 0, 128); boxes = np.array([[72, 24, 100, 56]])
    r = X.score_instance(E, (128, 128), (8, 8), box, boxes)
    assert r['hit'] and r['dice'] > 0 and r['chance'] == pytest.approx(29 * 33 / 128 ** 2)
    r2 = X.score_instance(E, (128, 128), (8, 8), box, np.array([[0, 100, 20, 127]]))
    assert not r2['hit']
    assert not X.score_instance(np.zeros((8, 8)), (128, 128), (8, 8), box, boxes)['hit']
    s = X.summarise([dict(image='a', finding='f', hit=1, dice=.5, chance=.1), dict(image='b', finding='f', hit=0, dice=.1, chance=.1),
                     dict(image='a', finding='g', hit=1, dice=.3, chance=.2)], nb=50)
    assert s['pointing'] == pytest.approx(0.75)

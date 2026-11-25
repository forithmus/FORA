"""CPU checks of the pure-geometry measurement code (RECIST caliper, 2-D caliper, CADS organ / vessel / midline helpers)."""
import math

import numpy as np
import pytest

from fora.reportgen.measure import geometry as CG


def _grid(Z=21, N=121, sp=(1.5, .5, .5)):
    return np.meshgrid(np.arange(Z) * sp[0], np.arange(N) * sp[1], np.arange(N) * sp[2], indexing='ij')


def test_caliper3d_rotated_ellipsoid():
    sp = (1.5, .5, .5); zz, yy, xx = _grid(); c = np.array([15., 30., 30.])
    th = math.radians(30); y_, x_ = yy - c[1], xx - c[2]; a_ = y_ * math.cos(th) + x_ * math.sin(th); b_ = -y_ * math.sin(th) + x_ * math.cos(th)
    E = ((zz - c[0]) / 6) ** 2 + (a_ / 12) ** 2 + (b_ / 5) ** 2 <= 1
    r = CG.caliper3d(E, sp)
    assert abs(r['long_mm'] - 24) < 1.0 and abs(r['short_mm'] - 10) < 1.0
    assert r['recist'] == 'measurable'


def test_caliper3d_rectangle_diagonal_and_node_flag():
    sp = (1.5, .5, .5); B = np.zeros((21, 121, 121), bool); B[10, 20:40, 30:36] = True
    r = CG.caliper3d(B, sp)
    assert abs(r['long_mm'] - math.sqrt(109)) < 1e-6
    zz, yy, xx = _grid(); S = ((zz - 15) ** 2 + (yy - 30) ** 2 + (xx - 30) ** 2) <= 16
    rn = CG.caliper3d(S, sp, is_node=True)
    assert rn['recist'] == 'node_nonpathological' and rn['reportable_mm'] == rn['short_mm']


def test_pick_component_prefers_centre():
    p = np.zeros((10, 40, 40), np.float32); p[4:6, 5:9, 5:9] = .9; p[4:6, 18:22, 18:22] = .8
    m = CG.pick_component(p, (5, 20, 20), (1.5, .5, .5))
    assert m[5, 20, 20] and not m[5, 6, 6]


def test_measure2d_disk():
    from fora.reportgen.measure.caliper_model import measure2d
    yy, xx = np.mgrid[:96, :96]; p = (((yy - 47.5) ** 2 + (xx - 47.5) ** 2) <= 20 ** 2).astype(np.float32)
    r = measure2d(p, 0.5)
    assert abs(r['long'] - 20.5) < 1.0 and abs(r['short'] - 20.5) < 1.0 and abs(r['ins'] - 20) < 1.5


def test_organ_feats_sphere():
    from fora.reportgen.measure.cads.organ import organ_feats, orient
    A = np.diag([1., 1., 2., 1.]); m = np.zeros((60, 60, 40), bool)
    ii, jj, kk = np.meshgrid(np.arange(60), np.arange(60), np.arange(40) * 2., indexing='ij'); m[((ii - 30) ** 2 + (jj - 30) ** 2 + (kk - 40) ** 2) <= 15 ** 2] = True
    zax, sp = orient(A, m.shape); f, _ = organ_feats(m, A, zax, sp, m.shape[zax])
    assert zax == 2 and abs(f['cc_mm'] - 32) < 3 and abs(f['max3d_mm'] - 30) < 3 and f['z_touch'] == 0


def test_vessel_variant_readout():
    from fora.reportgen.measure.cads.vessel import seg_values, pa_value
    n = 30; a = dict(seg=[1] * 10 + [2] * 10 + [3] * 10, s=list(np.arange(n) * 2.), comp=[0] * n, touch=[0] * n,
                     fmax=list(np.linspace(30, 40, n)), fmin=list(np.linspace(28, 38, n)), deq=list(np.linspace(29, 39, n)), ins=list(np.linspace(27, 37, n)))
    r = seg_values(a, 'fmax', 1, 0)
    assert r[1] == pytest.approx(a['fmax'][9]) and r[7] == pytest.approx(40.)
    p = dict(fmax=[25.] * 20, fmin=[20.] * 20, deq=[22.] * 20, ins=[19.] * 20, touch=[0] * 20, s=list(np.arange(20) * 2.), bif=15)
    assert pa_value(p, 'deq', 8, 8, 'med') == pytest.approx(22.)


def test_midline_plane_fit_symmetric_points():
    from fora.reportgen.measure.cads.midline import fit_plane
    rng = np.random.default_rng(0); P = rng.normal(size=(3000, 3)) * np.array([40., 60., 50.])
    P = np.concatenate([P, P * np.array([-1., 1., 1.])]) + np.array([5., 0., 0.])   # mirror-symmetric about x = 5
    n, d, cost, x = fit_plane(P)
    assert abs(abs(n[0]) - 1) < 1e-2 and abs(d * np.sign(n[0]) - 5) < 1.0


def test_cads_records_from_raw():
    import json
    from pathlib import Path
    from fora.reportgen.measure.cads import measure as M
    cal = json.loads((Path(__file__).resolve().parents[1] / 'data' / 'measure' / 'cads_calibration.json').read_text())
    c = dict(case='c1', report='r1', classes=[279, 110, 561])
    raw = dict(organ=dict(liver=dict(cc_mm=180., max3d_mm=200., inplane_feret_mm=150., vol_ml=1500., z_touch=0, series='a/b', dz=1.),
                          kidney_R=dict(cc_mm=100., max3d_mm=110., inplane_feret_mm=60., vol_ml=150., z_touch=0, series='a/b', dz=1.)),
               ctr=dict(same=0.52, z_touch=0, heart_mm=130., thorax_mm=250.))
    recs, ref = M.records(c, raw, cal)
    kinds = {r['cls']: r for r in recs}
    assert kinds[110]['value'] == round(0.52 + M.CTR_ADD, 2) and ref['cardiothoracic_ratio'] == kinds[110]['value']
    assert kinds[279]['value_mm'] == round(180. / np.exp(cal['organ']['279']['factor']), 1)
    assert kinds[561]['L_mm'] is None and kinds[561]['value_side'] == 'R'

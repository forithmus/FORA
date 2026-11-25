"""CPU smoke tests of the lesion caliper networks, crop geometry, scale calibration and inference tables (toy shapes)."""
import json

import numpy as np
import torch

from fora.reportgen.measure import caliper_model as CM
from fora.reportgen.measure import crops as CR
from fora.reportgen.measure import encoder as FD
from fora.reportgen.measure.scale_calibration import top_peak_calibration
from fora.reportgen.measure.tables import tables_from_extent


def _batch(B=2, ctx=True):
    N = int(np.prod(FD.TOK)); g = torch.Generator().manual_seed(0)
    b = dict(f=torch.randn(B, N, 1024, generator=g), tokc=torch.randn(B, N, 3) * 10, tok=torch.full((B,), 1.5), fov=torch.full((B,), 36.),
             hu=torch.randn(B, *CM.RG) * 200, c=torch.tensor([0, 3]), ct=torch.tensor([0, 2]), k=torch.tensor([5, -1]), mt=torch.tensor([0, 1]),
             evt=torch.rand(B, N, 4), evr=torch.rand(B, 4, *CM.RG), huc=torch.rand(B, *CM.RG) * 2 - 1, cx=None)
    if ctx:
        b.update(cx=torch.randn(B, 125, 1048), coff=torch.randn(B, 125, 3) * 8, cmask=torch.ones(B, 125, dtype=torch.bool), evk=torch.rand(B, 125, 4))
    return b


def test_caliper_head_forward_and_zero_init():
    head = CM.CaliperHead(nk=10).eval()
    with torch.no_grad():
        lt, lr = head(_batch())
        lt0, lr0 = head(_batch(ctx=False))
    assert lt.shape == (2, int(np.prod(FD.TOK))) and lr.shape == (2, *CM.RG) and torch.isfinite(lr).all()
    up = torch.nn.functional.interpolate(lt.view(2, 1, *FD.TOK), size=CM.RG, mode='trilinear', align_corners=False)[:, 0]
    assert torch.allclose(lr, up, atol=1e-5)   # zero-initialised refinement output conv
    assert lr0.shape == lr.shape


def test_dec2d_residual_and_slice_pick():
    dec = CM.Dec2D()
    lt3 = torch.randn(2, 96, 96)
    out = dec(torch.randn(2, 1024, 32, 32), lt3, torch.rand(2, 3, 96, 96), torch.rand(2, 4, 96, 96))
    assert torch.allclose(out, lt3)   # zero-initialised last conv: residual on the 3-D logit slice
    m3 = torch.zeros(2, 32, 96, 96); m3[0, 20, 40:60, 40:60] = 1; m3[0, 14, 45:50, 45:50] = 1
    z = CM.pick_slice(m3)
    assert int(z[0]) == 20 and int(z[1]) == 16


def test_view_sampling_identity_and_crop_grid():
    crop = torch.arange(20 * 40 * 40, dtype=torch.float32).view(20, 40, 40)
    origin = np.zeros(3); stride = np.ones(3); cvox = np.array([10., 20., 20.])
    mm = torch.zeros(1, 1, 1, 3)
    v = CM.sample_crop(crop, origin, stride, cvox, mm)
    assert abs(float(v) - float(crop[10, 20, 20])) < 1e-2
    vmm, tmm, rmm = CM.view_coords(1.5, 0., np.zeros(3), 'cpu')
    assert vmm.shape == (*FD.IN_VOX, 3) and tmm.shape == (*FD.TOK, 3) and rmm.shape == (*CM.RG, 3)
    g = CM.to_view_grid(rmm.reshape(-1, 3), 1.5, 0., np.zeros(3), CM.RG)
    assert g.abs().max() <= 1 + 1e-5


def test_crop_box_and_fov():
    assert CR.fov_of(5.) == 32. and CR.fov_of(100., 1.5) == 240.
    n, st = CR.box_of(36.)
    assert (n * st * np.array([1.5, .5, .5]) >= np.array([min(220, 1.25 * 36 + 8), min(300, 2.1 * 36 + 8), min(300, 2.1 * 36 + 8)]) - 1e-6).all()
    bits = np.array([1, 2, 3, 0], np.uint64); g = CR.route_grid(bits, (1, 2, 2), np.arange(4), 2)
    assert g.tolist() == [[[False, True], [True, False]]]


def test_scale_calibration_top_peak():
    rng = np.random.default_rng(0); nT = 80
    T = dict(key=np.array([f'k{i}' for i in range(nT)]), cls=np.full(nT, 546), mt=np.zeros(nT, int), split=np.array(['train'] * 60 + ['val'] * 20),
             y_long=rng.uniform(5, 20, nT), keep=np.ones(nT, bool), ycmp=np.zeros(nT, int))
    C = dict(t=np.arange(nT), rank=np.zeros(nT, int), row=np.arange(nT))
    import tempfile, os
    d = tempfile.mkdtemp()
    np.savez(os.path.join(d, 'cal_r000_c0000.npz'), t=np.arange(nT), prank=np.zeros(nT, int), long=T['y_long'] * 1.2, short=T['y_long'], ins=T['y_long'], ext3d=T['y_long'])
    cal = top_peak_calibration(d, C, T, K=16)
    assert abs(cal['t0'] - np.log(1.2)) < 1e-6 and abs(cal['546_0'] - np.log(1.2)) < 1e-6 and cal['t1'] == 0.


def test_tables_from_extent(tmp_path):
    schema = {'546': dict(mt=0, grp=0, kidx=3, bits=4, node=False, ycmp=3, prior=6.0)}
    n = 2
    np.savez(tmp_path / 'extent_r000_p0_c0000.npz', cls=np.array([546, 7]), pair_case=np.array(['a', 'a']), peak=np.array([[1, 2, 3], [0, 0, 0]]),
             prob=np.array([.9, .8]), logit=np.array([2., 1.]), blk_view=np.zeros((n, 2), int), blk_mask=np.ones((n, 2, 125), bool),
             blk_x=np.zeros((n, 2, 125, 1048), np.float16), blk_off=np.zeros((n, 2, 125, 3), np.float32))
    T, C = tables_from_extent(sorted(tmp_path.glob('extent_*.npz')), schema, tmp_path / 'context.npz')
    assert T['key'].tolist() == ['a|546'] and C['peak'].tolist() == [[1, 2, 3]] and float(T['fov_size_mm'][0]) == 6.0
    assert np.load(tmp_path / 'context.npz')['cb_x'].shape == (1, 125, 1048)


def test_data_files_consistent():
    from pathlib import Path
    D = Path(__file__).resolve().parents[1] / 'data' / 'measure'
    groups = json.loads((D / 'caliper_groups.json').read_text()); schema = json.loads((D / 'measurement_schema.json').read_text())
    assert sorted(groups) == ['air', 'cyst', 'node', 'nodule', 'solid', 'stone', 'thick']
    for g, spec in groups.items():
        assert (D / f'caliper_scale_{g}.json').exists()
        ex = {tuple(int(v) for v in p.split(':')) for p in spec['exclude'].split(',')}
        assert not any(tuple(p) in ex for p in spec['pairs'])
    assert all({'mt', 'grp', 'kidx', 'bits', 'node', 'ycmp', 'prior'} <= set(v) for v in schema.values())

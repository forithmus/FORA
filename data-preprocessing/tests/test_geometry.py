"""CPU tests of fora.geometry: resampling, body crop, windows and the lung-MIP rule, tokens, world, X-ray."""
import numpy as np

from conftest import synthetic_ct
from fora.geometry import protocols, tiling, tokens, windows, world, xray


def test_resample_and_body_crop():
    ct, _ = synthetic_ct()
    vol, geom = tiling.resample_series(ct, np.diag([1.0, 1.0, 3.0, 1.0]))
    assert vol.shape == (48, 96, 80)                   # (S, R, A) at 1.5 x 0.5 x 0.5 mm
    assert geom[0][1] == 1.5 and geom[1][1] == 0.5
    cropped, box = tiling.crop_to_body(vol)
    assert cropped.dtype == np.float32 and cropped.shape[0] == 48
    assert box[1] - box[0] < 96 and (cropped > -300).any()


def test_windows_and_mip_rule():
    w = windows.windows_from_meta(dict(WindowCenter='40\\-600\\40', WindowWidth='400\\1500\\400'))
    assert w == [(-160, 240), (-1350, 150)]
    views, mip = windows.plan_views(w)
    assert views[0] == [-1000, 1000] and mip == [-1350, 150]
    _, mip_inf = windows.plan_views(w, lung_protocol=False, mip_rule='inference')
    assert mip_inf is None
    _, mip_inf = windows.plan_views(w, lung_protocol=True, mip_rule='inference')
    assert mip_inf == [-1350, 150]
    v = windows.view_list(views, mip)
    assert [x['kind'] for x in v] == ['full', 'native', 'native', 'lung_mip_10mm']


def test_slab_mip_and_normalize():
    vol = np.zeros((20, 4, 4), np.float32)
    vol[10] = 100
    m = windows.slab_mip(vol)                          # 10 mm / 1.5 mm -> 7-slice window
    assert m[7:14].min() == 100 and m[6].max() == 0 and m[14].max() == 0
    t = windows.window_normalize(np.array([[[-2000., 0., 2000.]]]), -1000, 1000)
    assert t.shape == (1, 1, 1, 3) and t.float().flatten().tolist() == [-1.0, 0.0, 1.0]


def test_tokens_and_world():
    vol = np.full((6, 40, 40), -1000, np.float32)
    vol[:, 8:30, 8:30] = 40
    grid = tokens.token_grid(vol.shape)
    assert grid == (3, 3, 3)
    body = tokens.body_mask(tokens.token_mean_hu(vol))
    assert body.shape == grid and body[:, 1, 1].all()
    prov = dict(affine=np.diag([1., 1., 3., 1.]).tolist(), orig_shape=[48, 40, 24], body_crop_yxyx=[4, 90, 6, 76])
    c = world.token_centres_mm(prov, np.array([[0, 0, 0], [1, 0, 0]]))
    assert np.allclose(c[1] - c[0], [3.0, 0, 0])


def test_protocols():
    assert protocols.study_type_index('ct chest - iv contrast') == protocols.PROTOCOL_CLASSES.index('CT Chest - IV Contrast')
    assert protocols.study_type_index('unknown') == protocols.UNK_STUDY_TYPE == 29
    assert protocols.is_lung_study([0]) and not protocols.is_lung_study([1])


def test_xray_film():
    img = np.zeros((300, 400), np.uint16)
    img[50:250, 60:340] = np.linspace(100, 4000, 280, dtype=np.uint16)[None]
    r0, r1, c0, c1 = xray.content_bbox(img.astype(np.float32))
    assert r0 >= 45 and r1 <= 255 and c0 >= 55
    u = xray.prepare_film(img)
    assert u.min() == -1 and u.max() == 1
    assert xray.native_spacing({'pixel_spacing_mm': [0.2, 0.2]}) == 0.2

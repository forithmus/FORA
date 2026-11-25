"""Evidence foci, descriptions, focus-token store, finding store and finding-image batching on synthetic data (CPU)."""
import io

import numpy as np
import torch
from PIL import Image

from fora.reportgen.evidence import describe as D
from fora.reportgen.evidence.finding_store import FindingLocStore, write_study_loc_dir
from fora.reportgen.evidence.focus_tokens import build_store
from fora.mil.evidence import FOCI_KEYS as FO_KEYS, N_TOKENS as FT, foci
from fora.mil.extents import extents
from fora.mil.render import N_VIEWS, render_finding, series_map
from fora.reportgen.evidence.zoom_images import fimg_batch

REGIONS = ['lungs', 'lung_rul', 'lung_lll', 'liver', 'kidneys', 'body']
LABELS = ['Lung nodule', 'Liver lesion']


def _view(grid=(6, 8, 8), lung_left=True):
    zyx = np.stack(np.meshgrid(*[np.arange(g) for g in grid], indexing='ij'), -1).reshape(-1, 3)
    tok = np.array([3., 8., 8.]); world = (zyx + .5) * tok
    bits = np.zeros(len(zyx), np.int64)
    lung = zyx[:, 0] >= 3; bits[lung] |= 1 << 0
    bits[lung & (zyx[:, 1] < 4)] |= 1 << 1; bits[lung & (zyx[:, 1] >= 4)] |= 1 << 2
    bits[~lung] |= 1 << 3; bits |= 1 << 5
    return dict(zyx=zyx, world=world, tok=tok, nat=(1.5, .7), grid=grid, series=0, window='full', bits=bits)


def _evidence(v):
    c = np.array([4, 2, 3]); d = np.linalg.norm((v['zyx'] - c) * v['tok'], axis=1)
    e1 = np.exp(-(d / 10.) ** 2); c2 = np.array([1, 5, 5]); e2 = np.exp(-(np.linalg.norm((v['zyx'] - c2) * v['tok'], axis=1) / 12.) ** 2)
    return np.stack([e1, e2], 1)


def test_foci_and_description():
    v = _view(); E = _evidence(v); x = torch.randn(len(v['zyx']), 1048)
    fo = foci(E, [v], LABELS, x, REGIONS)
    assert len(fo) == 2 and set(FO_KEYS) <= set(fo[0])
    assert fo[0]['tva'].sum() >= 1 and fo[0]['tx'].shape == (FT, 1048) and fo[0]['restricted'] == 1
    txt = D.describe_finding(LABELS[0], fo[0], REGIONS, measurement='about 6.0 mm')
    assert txt.startswith('focal') and 'measured lesion about 6.0 mm' in txt
    merged, rows, pview = extents(E, [v])
    assert merged.shape == (2, 48) and len(rows) == 2


def test_wording():
    assert D.measurement_phrase(0, 3, False, 6.4, 5.1, 3.0) == 'about 6.4 x 5.1 mm (mean diameter 5.8 mm)'
    assert D.measurement_phrase(0, 0, True, 18, 11, 8) == 'short axis about 11 mm'
    assert D.location_phrase('Renal cyst', ['kidneys'], [0.8], 0.92, 0.08) == 'left kidney'
    assert D.family_of('Mediastinal lymph node enlargement')[0] == 'ln'


def _jpeg():
    b = io.BytesIO(); Image.new('RGB', (32, 32), (90, 90, 90)).save(b, format='JPEG'); return b.getvalue()


def test_store_roundtrip(tmp_path):
    v = _view(); E = _evidence(v); x = torch.randn(len(v['zyx']), 1048)
    fo = foci(E, [v], LABELS, x, REGIONS); j = _jpeg()
    z = dict(pair_rep=np.array(['study0001', 'study0001']), pair_case=np.array(['c0', 'c0']), cls=np.array([0, 1]), prob=np.array([.9, .4]),
             reg3=np.array([[1, 0, 2], [3, 4, 0]]), frac3=np.array([[.7, .2, .1], [.9, .05, .05]]), left=np.array([.1, .5]), right=np.array([.9, .5]),
             zfrac=np.array([.8, .2]), view=np.array([0, 0]), feat=np.random.randn(2, 1048).astype(np.float16),
             img_blob=np.frombuffer(j * 3, np.uint8), img_off=np.array([[0, len(j), 2 * len(j), 3 * len(j), 3 * len(j), 3 * len(j), 3 * len(j)]] * 2))
    for k in FO_KEYS: z['fo_' + k] = np.stack([np.asarray(r[k]) for r in fo])
    ext = tmp_path / 'extent'; ext.mkdir(); np.savez(ext / 'extent_r000_p0_c0000.npz', **z)
    ft = tmp_path / 'ftok'; s = build_store(ext, ft, LABELS, REGIONS, {'c0|0': 'about 6.0 mm'}, log=lambda *_: None)
    assert s['rows'] == 2
    loc = tmp_path / 'loc'; write_study_loc_dir(loc, ft, 'study0001', 2, LABELS, REGIONS)
    st = FindingLocStore(loc, LABELS, descriptions=ft / 'desc.npy', ftok=ft)
    rows = st.rows('study0001'); assert rows == {0: 0, 1: 1}
    tc = st.token_classes(rows); assert tc[0] == 1 + int(fo[0]['tva'].sum())
    feats = st.features([rows[0], rows[1]], 'cpu'); assert feats.shape == (tc[0] + tc[1], 1064)
    assert 'measured lesion about 6.0 mm' in st.descriptions('study0001')[0]
    ims = st.images([0]); assert len(ims[0]) == N_VIEWS and ims[0][0] == j and ims[0][3] == b''


def test_render_finding():
    vol = np.random.randint(-1000, 200, (24, 128, 128)).astype(np.int16)
    grid = (12, 8, 8); zyx = np.stack(np.meshgrid(*[np.arange(g) for g in grid], indexing='ij'), -1).reshape(-1, 3)
    ev = np.exp(-np.linalg.norm(zyx - np.array([6, 4, 4]), axis=1) ** 2).astype(np.float32)
    out = render_finding(vol, series_map(grid, [(zyx, ev)]), (12, 64, 64), 'lung', [(4, 30, 30)])
    assert len(out) == 4 and all(b[:2] == b'\xff\xd8' for b in out)


class _StubProcessor:
    def __call__(self, images, return_tensors):
        return {'pixel_values': torch.zeros(len(images), 3, 4, 4), 'image_position_ids': torch.zeros(len(images), 2, dtype=torch.long)}


def test_fimg_batch():
    j = _jpeg()
    pv, pos, mask = fimg_batch(_StubProcessor(), [(j, j, b'', b'', b'', b''), (b'',) * 6])
    assert pv.shape[0] == 3 and mask.tolist() == [[True, True, False, False, False, False], [True] + [False] * 5]
    pv, pos, mask = fimg_batch(_StubProcessor(), [])
    assert pv.shape[0] == 1 and mask.shape == (0, 0)


def test_merge_and_extract_import(tmp_path):
    import importlib
    importlib.import_module('fora.reportgen.evidence.extract')
    from fora_report.evidence.merge_store import merge_parts, merge_tokens
    v = _view(); E = _evidence(v); fo = foci(E, [v], LABELS, torch.randn(len(v['zyx']), 1048), REGIONS)
    for n, uid in enumerate(['study0002', 'study0001']):
        z = dict(pair_rep=np.array([uid]), pair_case=np.array([f'c{n}']), cls=np.array([0]), prob=np.array([.9]), reg3=np.array([[1, 0, 2]]),
                 frac3=np.array([[.7, .2, .1]]), left=np.array([.1]), right=np.array([.9]), zfrac=np.array([.8]), view=np.array([0]),
                 feat=np.random.randn(1, 1048).astype(np.float16))
        for k in FO_KEYS: z['fo_' + k] = np.stack([np.asarray(fo[0][k])])
        ext = tmp_path / f'ext{n}'; ext.mkdir(); np.savez(ext / 'extent_r000_p0_c0000.npz', **z)
        build_store(ext, tmp_path / 'parts' / f'node_{n:02d}', LABELS, REGIONS, log=lambda *_: None)
    ref = tmp_path / 'ref.npz'; np.savez(ref, region_names=np.array(REGIONS), labels=np.array(LABELS))
    r = merge_parts(tmp_path / 'parts', tmp_path / 'store.npz', tmp_path / 'desc.npy', ref)
    assert r['reports'] == 2 and list(np.load(tmp_path / 'store.npz')['report_uid']) == ['study0001', 'study0002']
    s = merge_tokens(str(tmp_path / 'parts/node_00'), str(tmp_path / 'parts/node_01'), str(tmp_path / 'ftok'))
    assert s == dict(rows=2, from_old=1, from_new=1)

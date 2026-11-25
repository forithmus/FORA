"""Tiny end-to-end run on a synthetic shard: plan -> block-23 cache (fake encoder) -> region bits -> bags -> split."""
import json

import h5py
import numpy as np
import torch

import exam_types
import make_split
import make_study_bags
import plan_extraction
import token_regions
from extract_block23 import encode_series, load_series
from fora.cache.encode import Block23Encoder
from fora.cache.format import dequantize, quantize, read_view
from fora.encoder.blocks import crop_layout
from fora.regions import REGION_NAMES, clear_pelvis_without_hips, family_of, load_class_region_bits, token_region_bits

ENCODING = dict(crop=[64, 192, 192], stride=[64, 192, 192], blend='none')


def test_quantize_roundtrip():
    flat = np.array([0, 5, 300, 4700])
    grid = (40, 12, 12)
    order, ids, starts, local, n = crop_layout(flat, grid)
    h = torch.randn(len(ids), 4613, 1024).to(torch.bfloat16)
    c = quantize(h.view(torch.int16).numpy().view(np.uint16), flat, grid)
    rows = np.repeat(np.arange(len(starts) - 1), np.diff(starts))
    x = dequantize(c, rows)
    ref = h.float()[torch.from_numpy(rows), torch.from_numpy(local + 5)].numpy()
    assert np.abs(x - ref).max() <= np.abs(ref).max() / 127 + 1e-3
    assert np.array_equal(x[:, c['outlier_idx']], ref[:, c['outlier_idx']])


def test_end_to_end(ct_tree, fake_block23):
    tmp = ct_tree['tmp']
    rows = plan_extraction.scan(f"{ct_tree['h5_dir']}/batch_00001.h5")
    assert len(rows) == 1 and len(rows[0]['series']) == 1
    s = rows[0]['series'][0]
    assert s['windows'][0] == [-1000, 1000] and s['mip'] == [-1350, 150]
    rows[0]['split'] = 'train'
    (tmp / 'plan').mkdir()
    plan_extraction.write_plan(rows, str(tmp / 'plan'), workers=1)
    rep = json.loads(open(tmp / 'plan/reports_train.jsonl').readline())

    vol, mip, views, prov = load_series(ct_tree['h5_dir'], rep, rep['series'][0])
    enc = Block23Encoder(None, 'cpu')
    out = tmp / 'local/train' / rep['id'] / 'series_000.h5'
    n = encode_series(enc, ENCODING, vol, mip, views, prov, out, 'id0', 'fp0')
    assert n > 0
    with h5py.File(out, 'r') as f:
        assert sorted(k for k in f if k.startswith('v')) == ['v00', 'v01', 'v02', 'v03']
        v = read_view(f, 'v03')
        assert json.loads(f['v03'].attrs['window_json'])['kind'] == 'lung_mip_10mm'
        assert v['q'].shape == (n, 1024) and v['prefix'].shape[1:] == (5, 1024)

    bits, names, info = token_region_bits(out, ct_tree['h5_dir'], ct_tree['cads_dir'])
    assert names == REGION_NAMES and info['cads_source'] == 'own' and info['mapping'] == 'index' and info['geom_ok']
    li = REGION_NAMES.index('liver')
    bi = REGION_NAMES.index('body')
    assert ((bits >> np.uint64(bi)) & np.uint64(1)).all()
    assert 0 < info['counts']['liver'] < len(bits)
    assert info['counts']['lungs'] == 0

    # study bags, region pack / unpack round trip
    store = tmp / 'store'
    store.mkdir()
    view = dict(file=str(out), offset=0, file_size=out.stat().st_size, relative_file=f"train/{rep['id']}/series_000.h5")
    case = dict(id=rep['id'], split='train', report=rep['report'], complete=True, series=[view], views=[dict(view, group='v00')])
    json.dump(dict(fingerprint='fp0', cases={rep['id']: case}, failed={}), open(store / 'index_worker_000.json', 'w'))
    cases, _ = make_study_bags.collect([str(store)])
    a = make_study_bags.assign(cases, 4)
    assert a['train'][0] == [rep['id']]
    (tmp / 'bags').mkdir()
    with open(tmp / 'bags/cases.jsonl', 'w') as f:
        f.write(json.dumps(case) + '\n')
    json.dump(dict(fingerprint='fp0', ranks=4, assignments=a), open(tmp / 'bags/assign.json', 'w'))

    class A:
        cases = str(tmp / 'bags/cases.jsonl')
        assign = str(tmp / 'bags/assign.json')
        ranks = '0-3'
        local = str(tmp / 'local')
        h5_dir = ct_tree['h5_dir']
        cads_dir = ct_tree['cads_dir']
        procs = 1
        out = str(tmp / 'regions.h5')
        pack = str(tmp / 'regions.h5')
    token_regions.compute(A)
    token_regions.pack(A)
    rel = f"{A.local}/regions/train/{rep['id']}/series_000.h5.npz"
    before = np.load(rel)['bits']
    token_regions.unpack(A)
    assert np.array_equal(np.load(rel)['bits'], before) and np.array_equal(before, bits)


def test_class_bits_and_families(tmp_path):
    p = tmp_path / 'bits.json'
    p.write_text(json.dumps({'Hepatic steatosis': 1 << REGION_NAMES.index('liver')}))
    cb = load_class_region_bits(p, ['Hepatic steatosis', 'Unrouted finding'])
    assert cb[1] == np.uint64((1 << 64) - 1)
    pel = REGION_NAMES.index('pelvis')
    b = np.array([1 << pel, 0], np.int64)
    assert (clear_pelvis_without_hips(b) == 0).all()
    assert family_of('Hepatic steatosis')[0] == 'liver'
    assert family_of('Mediastinal lymphadenopathy')[0] == 'ln'
    assert family_of('Pulmonary nodule')[0] == 'lung'
    assert family_of('Pulmonary nodule')[1][-1] == 'lungs'


def test_split_and_exam_types():
    rep2case = {'r1': 'train_000000', 'r2': 'train_000001', 'r3': 'val_000000', 'r4': 'val_000001'}
    pat = {'r1': 'p1', 'r2': 'p1', 'r3': 'p2', 'r4': 'p3'}
    prot = {'r1': {6}, 'r2': {7}, 'r3': {7}, 'r4': {7}}
    new, summary = make_split.make_split(rep2case, pat, prot, {'r1'})
    assert new['train_000000'] == new['train_000001'] == 'val'
    assert summary['moved_val_to_train'] == 2 and list(new.values()).count('val') == 2
    assert exam_types.exam_type({6, 7}) == 'Chest CT'
    assert exam_types.exam_type({6, 13}) == 'mixed' and exam_types.exam_type({29}) == 'mixed'


def test_protocol_classification_helpers():
    import classify
    import post_process
    import study_features
    classes = classify.load_keep_classes()
    title_by_n = {int(c['n']): c['title'] for c in classes}
    titles = {c['title'] for c in classes}
    out = classify.parse_protocols('[{"n": 13, "title": "x"}, {"n": 0, "title": "other"}]', title_by_n, titles)
    assert out == ['CT Head - Non-Contrast']
    assert classify.parse_protocols('garbage', title_by_n, titles) == ['other']
    assert '{class_menu}' in classify.cfg.CLS_SYSTEM_PROMPT_TEMPLATE
    assert post_process.apply_rules(['CT Liver - Four-Phase', 'CT Abdomen and Pelvis - Multiphase'], post_process.MERGE_RULES) == \
        ['CT Abdomen and Pelvis - Multiphase']
    f = study_features.compute_features([{'Modality': 'CT', 'Contrast/BolusAgent': 'X', 'Contrast/BolusStartTime': '1',
                                          'AcquisitionNumber': '1'}])
    assert f['n_bolus_times'] == 1

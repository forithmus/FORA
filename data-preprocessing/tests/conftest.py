"""Synthetic CT shard, CADS masks and a fake encoder for the CPU smoke tests."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

STAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE))
sys.path.insert(0, str(STAGE / 'protocol_classification'))


def synthetic_ct(shape=(48, 40, 24)):
    """(X, Y, Z) HU volume: air, an elliptic body (40 HU), a 'liver' blob (60 HU) on the patient's right."""
    X, Y, Z = np.meshgrid(*(np.arange(n) for n in shape), indexing='ij')
    ct = np.full(shape, -1000, np.int16)
    body = ((X - 24) / 18.) ** 2 + ((Y - 20) / 14.) ** 2 <= 1
    ct[body] = 40
    liver = ((X - 14) / 6.) ** 2 + ((Y - 20) / 6.) ** 2 + ((Z - 12) / 6.) ** 2 <= 1
    ct[liver] = 60
    return ct, liver


@pytest.fixture
def ct_tree(tmp_path):
    """CT H5 shard batch_00001.h5 with one report and one series + its CADS mask file."""
    h5_dir = tmp_path / 'h5'
    cads_dir = tmp_path / 'cads'
    h5_dir.mkdir()
    cads_dir.mkdir()
    ct, liver = synthetic_ct()
    affine = np.diag([1.0, 1.0, 3.0, 1.0])
    meta = dict(WindowCenter='40\\-600', WindowWidth='400\\1500')
    key = 'pat-1/rpt-1/acc-1/series_20200101120000_a'
    with h5py.File(h5_dir / 'batch_00001.h5', 'w') as f:
        g = f.create_group(key)
        g.create_dataset('data', data=ct)
        g.create_dataset('affine', data=affine)
        g.create_dataset('pacs_metadata', data=json.dumps(meta))
        g.create_dataset('dcm2niix_metadata', data=json.dumps({}))
        g.attrs['classified_protocols'] = json.dumps(['CT Chest - Non-Contrast (incl. HRCT / Low-Dose Screening)'])
        r = f.create_group('pat-1/rpt-1/report')
        r.create_dataset('structured', data='{}')
    with h5py.File(cads_dir / 'cads_batch_00001.h5', 'w') as f:
        g = f.create_group(key)
        g.attrs['image_shape'] = ct.shape
        g.create_dataset('image_affine', data=affine)
        g.create_dataset('part_551', data=(liver * 5).astype(np.uint8))
    return dict(h5_dir=str(h5_dir), cads_dir=str(cads_dir), key=key, tmp=tmp_path)


@pytest.fixture
def fake_block23(monkeypatch):
    """Replace blocks 1-23 by a deterministic function of the input (no encoder weights needed)."""
    import fora.cache.encode as E

    def fake(bb, image):
        b = image.shape[0]
        g = torch.Generator().manual_seed(0)
        base = torch.randn(1, 4613, 1024, generator=g)
        h = base + image.float().mean(dim=(1, 2, 3, 4)).view(b, 1, 1)
        return h.to(torch.bfloat16), None

    monkeypatch.setattr(E, 'block23_tokens', fake)
    return fake

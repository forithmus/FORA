"""Put the stage folder on sys.path so `import fora_pretrain` works without installation; shared tiny fixtures."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

STAGE = Path(__file__).resolve().parents[1]
if str(STAGE) not in sys.path:
    sys.path.insert(0, str(STAGE))


def tiny_backbone(**kw):
    from fora.encoder.model import DinoVisionTransformer3D
    m = DinoVisionTransformer3D(embed_dim=64, depth=kw.pop("depth", 2), num_heads=1, ffn_ratio=2, **kw)
    m.init_weights()
    return m


@pytest.fixture
def ct_h5(tmp_path):
    """One shard, one patient, one report with statements and two series (one with a lung window)."""
    import h5py
    path = tmp_path / "batch_00001.h5"
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        rep = f.create_group("pat-A/rpt-A")
        rep.create_group("report").create_dataset("there_is_statements", data=json.dumps(["There is a nodule.", "There is no effusion."]))
        for k, (wc, ww) in enumerate([(-600, 1500), (40, 400)]):
            sg = rep.create_group(f"acc-1/series-{k}")
            vol = rng.integers(-1000, 400, size=(400, 400, 70)).astype(np.int16)
            vol[100:300, 100:300, :] = 50
            sg.create_dataset("data", data=vol)
            sg.create_dataset("affine", data=np.diag([0.5, 0.5, 1.5, 1.0]))
            sg.create_dataset("pacs_metadata", data=json.dumps({"WindowCenter": wc, "WindowWidth": ww, "BodyPartExamined": "CHEST"}))
    return path


@pytest.fixture
def cxr_h5(tmp_path):
    import h5py
    path = tmp_path / "cxr_00001.h5"
    rng = np.random.default_rng(1)
    with h5py.File(path, "w") as f:
        rep = f.create_group("pat-B/rpt-B")
        rep.create_group("report").create_dataset("there_is_statements", data=json.dumps(["There is no pneumothorax."]))
        img = rng.integers(0, 4095, size=(300, 280, 1)).astype(np.uint16)
        sg = rep.create_group("acc-1/film-0")
        sg.create_dataset("data", data=img)
        sg.create_dataset("dcm2niix_metadata", data=json.dumps({"view_class": "pa", "pixel_spacing_mm": [0.15, 0.15]}))
    return path

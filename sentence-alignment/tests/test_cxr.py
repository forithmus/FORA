"""X-ray alignment data path: index, film tiling, split teacher (frozen 1-23 | block 24) equals the plain forward."""
import json

import h5py
import numpy as np
import torch


def test_cxr_index_prepare_and_split_encoder(tmp_path):
    from fora.encoder.backbone import TEACHER_FORMAT, build_backbone
    from fora.encoder.blocks import block24_state
    from fora.encoder.cxr import CXRSplitEncoder
    from fora_align.train_cxr import _index_shard, prepare, tile_chunks
    rng = np.random.default_rng(0)
    with h5py.File(tmp_path / "s1.h5", "w") as f:
        rep = f.create_group("pat-B/rpt-B")
        rep.create_group("report").create_dataset("there_is_statements", data=json.dumps(["There is no effusion."]))
        for k, view in enumerate(["pa", "lateral"]):
            sg = rep.create_group(f"acc/film-{k}")
            sg.create_dataset("data", data=rng.integers(0, 4095, size=(300, 270, 1)).astype(np.uint16))
            sg.create_dataset("dcm2niix_metadata", data=json.dumps({"view_class": view}))
    cases = _index_shard(str(tmp_path / "s1.h5"))
    assert len(cases) == 1 and [f[2] for f in cases[0]["films"]] == [0, 1]
    bag = prepare(cases[0], dict(tile=128), str(tmp_path))
    bb = build_backbone("large")
    bb.init_weights()
    torch.save(dict(format=TEACHER_FORMAT, backbone=bb.state_dict(), metadata={}, step=1), tmp_path / "t.pt")
    enc = CXRSplitEncoder(tmp_path / "t.pt", block24_state(bb), "cpu")
    h23 = enc.h23(bag["tiles"][:1], 1)
    x = bag["tiles"][:1][:, None, None].expand(-1, 1, 2, -1, -1)
    with torch.no_grad(), enc.autocast():
        ref = enc.backbone.forward_features(x)["x_norm_patchtokens"].float()
        got = enc.b24(h23).float()
    assert torch.nn.functional.cosine_similarity(got, ref, dim=-1).min() > 0.999
    n = 0
    with torch.no_grad():
        for xc, kind in tile_chunks(enc, enc.h23(bag["tiles"], 4), bag, 3, torch.zeros(1024), 1.0, 1.0, "cpu"):
            assert xc.shape[1] == 1024 + 24
            n += len(xc)
    assert n == len(bag["pos"]) == 2 * 19 * 17

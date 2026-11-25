"""CPU tests of the writer's visual stream helpers, cache extraction and FSDP helpers."""
from __future__ import annotations

import json
import os
import socket
import sys

import numpy as np
import pytest
import torch

from fora_report.writer import dense_cache
from fora.reportgen.writer import dino_stream as D
from fora_report.writer import fsdp as FS


def test_placement_helpers():
    assert D.data_rank_for(3, 2) == 14
    with pytest.raises(ValueError):
        D.data_rank_for(0, 4)
    D.check_unique_data_ranks([0, 1, 2])
    with pytest.raises(RuntimeError):
        D.check_unique_data_ranks([0, 1, 1])
    assert [D.view_kind_id(k) for k in ("full", "native", "lung_mip_10mm", "bone")] == [0, 1, 2, 3]
    assert D.token_dim({"training": {"position_freqs": 4}}) == 1048


def test_detect_data_node(tmp_path):
    (tmp_path / "fp").mkdir()
    (tmp_path / "fp" / "STAGED_node7.json").write_text(json.dumps({"node": 7}))
    assert D.detect_data_node("fp", tmp_path) == 7
    assert D.detect_data_node("fp", tmp_path, data_node="2") == 2
    with pytest.raises(RuntimeError):
        D.detect_data_node("missing", tmp_path)


def test_subsample_mask_deterministic():
    a = D.subsample_mask(1000, 100, "42|uid|0", "cpu")
    b = D.subsample_mask(1000, 100, "42|uid|0", "cpu")
    c = D.subsample_mask(1000, 100, "42|uid|1", "cpu")
    assert int(a.sum()) == 100 and torch.equal(a, b) and not torch.equal(a, c)
    assert D.subsample_mask(50, 100, "x", "cpu") is None
    assert D.subsample_mask(50, 0, "x", "cpu") is None


def test_shared_roundtrip():
    views = [dict(q=np.arange(6, dtype=np.int8), outlier=np.arange(4, dtype=np.uint16), window={"kind": "full"})]
    back = D.from_shared(D.to_shared(views))
    assert back[0]["outlier"].dtype == np.uint16
    assert np.array_equal(back[0]["q"], views[0]["q"]) and back[0]["window"] == {"kind": "full"}


def _fake_view(n_crops, rng, dim=8, prefix=5):
    lens = rng.integers(1, 5, n_crops)
    starts = np.concatenate([[0], np.cumsum(lens)])
    n = int(starts[-1])
    return dict(prefix=np.zeros((n_crops, prefix, dim), np.uint16),
                q=rng.integers(-127, 127, (n, dim)).astype(np.int8), scale=np.ones((n_crops, dim), np.float16),
                oidx=np.array([0, 1], np.int16), outlier=np.zeros((n, 2), np.uint16), starts=starts,
                local=np.zeros(n, np.int64), pos=rng.standard_normal((n, 24)).astype(np.float32))


def test_visual_stream_streams_every_token():
    pytest.importorskip("fora.encoder.cache")
    rng = np.random.default_rng(0)
    views = [_fake_view(5, rng), _fake_view(3, rng)]
    total = int(sum(v["starts"][-1] for v in views))
    stream = D.DinoVisualStream.__new__(D.DinoVisualStream)
    stream.device = torch.device("cpu")
    stream.mean = torch.zeros(8)
    stream.rms = 2.0
    stream.position_scale = 1.0
    stream.crops_per_chunk = 2
    stream.gpu_bag_gib = 1.0
    stream.block = lambda x, pos, ok: x                     # identity stands in for block 24
    image = dict(views=D.to_shared(views), view_kind=torch.tensor([0, 2]), view_series=torch.tensor([0, 1]),
                 case_id="c")
    bag = stream.encode(image)
    assert bag.tokens.shape == (total, 8 + 24)
    assert bag.view_kind.tolist() == [0] * int(views[0]["starts"][-1]) + [2] * int(views[1]["starts"][-1])
    capped = stream.encode(image, cap=7, seed_key="k")
    assert capped.tokens.shape[0] == 7 and capped.total_tokens == total


def test_dense_cache_extract_and_reuse(tmp_path, monkeypatch, capsys):
    source = tmp_path / "dense.npz"
    np.savez(source, report_uid=np.array(["a", "b", "c"]), full_token_count=np.array([5, 6, 7], np.int64),
             logit=np.zeros((3, 4), np.float32), label=np.array(["w", "x", "y", "z"]),
             class_index=np.arange(4))
    out = tmp_path / "cache"
    for _ in range(2):
        monkeypatch.setattr(sys, "argv", ["dense_cache", "--source", str(source), "--output", str(out),
                                          "--expected-studies", "3", "--expected-classes", "4"])
        dense_cache.main()
    log = capsys.readouterr().out
    assert "created" in log and "reuse" in log
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["n_studies"] == 3 and manifest["n_classes"] == 4


def test_fsdp_flag():
    assert FS.fsdp_requested("gemma4") is True
    assert FS.fsdp_requested("gemma4", "0") is False
    assert FS.fsdp_requested("other", "1") is True
    with pytest.raises(ValueError):
        FS.fsdp_requested("gemma4", "maybe")


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def single_rank_group():
    import torch.distributed as dist
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(_free_port())
    dist.init_process_group("gloo", rank=0, world_size=1)
    yield
    dist.destroy_process_group()


def test_fsdp_state_roundtrip_and_reference_swap(single_rank_group):
    from fora_report.writer import fsdp_dpo as FD
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 6), torch.nn.Linear(6, 3))
    model[0].bias.requires_grad_(False)
    mesh = FS.build_mesh("cpu", 1, 1)
    FS.shard_report_model(model, [model[0]], mesh)
    state = FS.full_trainable_state(model, True)
    assert set(state) == {"0.weight", "1.weight", "1.bias"}
    reference = {k: torch.zeros_like(v) for k, v in state.items()}
    ref = FD.FSDPReferenceWeights(model, reference, "cpu")
    assert ref.max_abs_difference() > 0
    x = torch.randn(2, 4)
    with torch.no_grad():
        policy_out = model(x).clone()
        with ref.swapped():
            ref_out = model(x).clone()
        again = model(x)
    assert torch.allclose(policy_out, again)
    assert not torch.allclose(policy_out, ref_out)
    FS.load_full_trainable_state(model, reference)
    assert ref.max_abs_difference() == 0.0
    assert FD.load_exact_warm_start(model, state) == []
    checksum = FS.trainable_checksum(model, torch.device("cpu"))
    assert torch.isclose(checksum[0], sum(v.double().sum() for v in state.values()))

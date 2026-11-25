"""End-to-end CPU run of the CT alignment trainer on a synthetic block-23 cache (2 gloo ranks, 2 steps)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

BERT = os.environ.get("FORA_TEST_CXR_BERT", "")
STAGE = Path(__file__).resolve().parents[1]


def write_series(path: Path, rng, n_views=2, grid=(34, 13, 12)):
    from fora.encoder.blocks import crop_layout
    flat = np.sort(rng.choice(int(np.prod(grid)), 400, replace=False))
    order, ids, starts, local, _ = crop_layout(flat, grid)
    rows = np.repeat(np.arange(len(ids)), np.diff(starts))
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("flat_indices", data=flat)
        for v in range(n_views):
            g = f.create_group(f"v{v:02d}")
            g.attrs["metadata_json"] = json.dumps({"grid": list(grid)})
            x = torch.randn(len(flat), 1024)
            scale = torch.zeros(len(ids), 1024).index_reduce_(0, torch.from_numpy(rows), x.abs(), "amax") / 127
            g.create_dataset("q", data=torch.round(x / scale[rows]).clamp(-127, 127).to(torch.int8).numpy())
            g.create_dataset("scale", data=scale.half().numpy())
            g.create_dataset("prefix", data=torch.randn(len(ids), 5, 1024).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16))
            oidx = np.arange(8, dtype=np.int16)
            g.create_dataset("outlier_idx", data=oidx)
            g.create_dataset("outlier", data=x[:, :8].to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16))
            g.create_dataset("crop_ids", data=ids.astype(np.int32))
    return int(path.stat().st_size), len(flat)


@pytest.mark.skipif(not BERT, reason="set FORA_TEST_CXR_BERT to a local CXR-BERT directory")
def test_ct_alignment_two_ranks(tmp_path):
    from fora.encoder.backbone import build_backbone
    from fora.encoder.blocks import block24_state
    rng = np.random.default_rng(0)
    cache = tmp_path / "cache"
    cases, split, sents = [], {}, {}
    for i in range(4):
        cid = f"train_{i:06d}"
        rel = f"train/{cid}/series_000.h5"
        size, n = write_series(cache / rel, rng)
        kinds = ["full", "lung_mip_10mm"]
        cases.append(dict(id=cid, views=[dict(relative_file=rel, file_size=size, group=f"v{v:02d}", n_tokens=n,
                                              window=dict(kind=kinds[v])) for v in range(2)]))
        split[cid] = "train"
        sents[cid] = [f"There is finding {j}." for j in range(i + 2)]
    (tmp_path / "cases.jsonl").write_text("\n".join(json.dumps(c) for c in cases) + "\n")
    (tmp_path / "split.json").write_text(json.dumps(split))
    (tmp_path / "sentences.json").write_text(json.dumps(sents))
    torch.save(dict(mean=torch.zeros(1024), rms=1.0), tmp_path / "norm.pt")
    bb = build_backbone("large")
    bb.init_weights()
    torch.save(block24_state(bb), tmp_path / "b24.pt")
    cfg = json.loads((STAGE / "configs" / "ct_alignment.json").read_text())
    cfg.update(cases=str(tmp_path / "cases.jsonl"), assign=None, split=str(tmp_path / "split.json"),
               sentences=str(tmp_path / "sentences.json"), normalization=str(tmp_path / "norm.pt"),
               block24_state=str(tmp_path / "b24.pt"), cache_root=str(cache))
    cfg["training"].update(max_sentences=3, warmup_steps=1, bucket_size=1, crops_per_chunk=1)
    cfg["training"]["model"]["text_model"] = BERT
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))
    out = tmp_path / "run"
    env = dict(os.environ, PYTHONPATH=f"{STAGE}:{os.environ.get('PYTHONPATH', '')}", OMP_NUM_THREADS="2")
    r = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--nproc-per-node", "2", "--master-port", "29731",
                        "-m", "fora_align.train_ct", "--config", str(tmp_path / "cfg.json"), "--output", str(out),
                        "--max-steps", "2"], env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    s = torch.load(out / "latest.pt", weights_only=False)
    assert s["step"] == 2 and set(s["block"]) and all(torch.isfinite(v).all() for v in s["block"].values())
    progress = json.loads((out / "progress.json").read_text())
    assert np.isfinite(progress["loss"])
    subprocess.run([sys.executable, "-m", "fora_align.export_block24", str(out / "latest.pt"), str(tmp_path / "b.pt")],
                   env=env, check=True)
    assert "block.attn.qkv.weight" in torch.load(tmp_path / "b.pt", weights_only=False)["block"]

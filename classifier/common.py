"""Small helpers shared by the classifier scripts: config loading, distributed setup, atomic JSON / npz writes, the
learning-rate schedule and the manual gradient all-reduce."""
from __future__ import annotations

import json
import math
import os
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


def _expand(v):
    if isinstance(v, str):
        return os.path.expandvars(v)
    if isinstance(v, dict):
        return {k: _expand(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_expand(x) for x in v]
    return v


def load_config(path) -> dict:
    """JSON config with ${VAR} environment references in string values expanded (``training`` is kept verbatim)."""
    m = json.loads(Path(path).read_text())
    return {k: (v if k == "training" else _expand(v)) for k, v in m.items()}


def write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def save_npz(path, **arrays) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    tmp.replace(path)


def save_torch(path, obj) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def init_distributed():
    """torchrun environment -> (rank, world, device, barrier). NCCL on GPU, gloo on CPU (tests)."""
    rank, world, local = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
        dist.init_process_group("nccl", device_id=device, timeout=timedelta(minutes=30))
        barrier = lambda: dist.barrier(device_ids=[local])
    else:
        device = torch.device("cpu")
        dist.init_process_group("gloo", timeout=timedelta(minutes=30))
        barrier = dist.barrier
    return rank, world, device, barrier


def lr_fraction(step: int, warmup: int, total: int) -> float:
    """Linear warm-up to 1, then cosine to 0 at ``total`` steps."""
    return min(1., (step + 1) / max(1, warmup)) * (.5 + .5 * math.cos(math.pi * max(0, step - warmup) / max(1, total - warmup)))


def allreduce_gradients(params, world: int) -> None:
    """Mean of the gradients over ranks (missing gradients count as zero), one flat all-reduce."""
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    flat = torch.cat([p.grad.reshape(-1) for p in params])
    dist.all_reduce(flat)
    flat.div_(world)
    o = 0
    for p in params:
        n = p.numel()
        p.grad.copy_(flat[o:o + n].reshape_as(p))
        o += n


def bucketed_order(train, size, seed: int, epoch: int, rank: int, bucket: int, min_count: int):
    """Epoch order of a rank's reports: sorted by size, cut into buckets of ``bucket`` reports, buckets in an order shared
    by all ranks (so the ranks process bags of similar size together), shuffled within each bucket."""
    rng = np.random.default_rng(seed + 100 + epoch + 10000 * rank)
    ordered = sorted(train, key=lambda c: (-size(c), c))
    bins = min_count // bucket
    bin_order = np.random.default_rng(seed + epoch).permutation(bins)
    return ([c for b in bin_order for c in rng.permutation(ordered[b * bucket:(b + 1) * bucket]).tolist()]
            + rng.permutation(ordered[bins * bucket:]).tolist())


def swap_ema(params, ema) -> None:
    """Exchange the live parameters with their EMA copies (call twice to restore)."""
    if ema is None:
        return
    for p, e in zip(params, ema):
        tmp = p.detach().clone()
        p.data.copy_(e)
        e.copy_(tmp)

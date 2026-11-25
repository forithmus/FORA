"""Distributed runtime of the encoder pre-training: process group, shard-to-rank assignment and node-local staging,
FSDP2 wrapping of student / teacher / Gram anchor, DCP checkpoints with exact resume, and a signal-triggered stop.
"""
from __future__ import annotations

import csv
import json
import os
import random
import shutil
import signal
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

OFFICIAL_DINOV3_COMMIT = "6876159a11b4df116f30f667f8c9888617df0751"
CHECKPOINT_FORMAT = "fora_fsdp_checkpoint"


def setup_distributed() -> tuple[int, int, int, torch.device]:
    rank, world, local = (int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    dist.init_process_group("nccl", timeout=timedelta(hours=2), device_id=device)
    return rank, world, local, device


def read_train_files(csv_path: str, h5_folder: str) -> list[str]:
    """Training shards: rows of the split CSV with split == train (columns h5_id, split) that exist on disk."""
    out = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["split"].strip() == "train":
                path = os.path.join(h5_folder, row["h5_id"].strip() + ".h5")
                if os.path.exists(path):
                    out.append(path)
    return sorted(out)


def balanced_file_assignment(files: list[str], bins_n: int, weights: dict[str, int] | None = None) -> list[list[str]]:
    """Greedy deterministic (LPT) assignment of whole shards by workload (default: bytes), bytes as tie-break."""
    bins: list[list[str]] = [[] for _ in range(bins_n)]
    primary, byte_loads = [0] * bins_n, [0] * bins_n
    sized = [(int(weights[p]) if weights is not None else os.path.getsize(p), os.path.getsize(p), p) for p in files]
    sized.sort(key=lambda x: (-x[0], -x[1], x[2]))
    for load, size, path in sized:
        r = min(range(bins_n), key=lambda i: (primary[i], byte_loads[i], i))
        bins[r].append(path)
        primary[r] += load
        byte_loads[r] += size
    return bins


def reconstruction_balanced_file_assignment(files, world, rank, cache_path, count_fn, min_slices=16):
    """Balance CT shards over ranks by their number of usable reconstructions.

    Counts are computed collectively (``count_fn(path) -> int``) and cached in ``cache_path`` keyed by size and mtime."""
    cache: dict = {"version": 1, "min_slices": min_slices, "files": {}}
    if rank == 0 and cache_path and Path(cache_path).exists():
        try:
            loaded = json.loads(Path(cache_path).read_text())
            if loaded.get("version") == 1 and loaded.get("min_slices") == min_slices:
                cache = loaded
        except Exception:
            pass
    payload = [cache]
    dist.broadcast_object_list(payload, src=0)
    cache = payload[0]
    counts, missing = {}, []
    for path in files:
        stat, record = os.stat(path), cache["files"].get(path)
        if record is not None and int(record.get("size", -1)) == stat.st_size and int(record.get("mtime_ns", -1)) == stat.st_mtime_ns:
            counts[path] = int(record["reconstructions"])
        else:
            missing.append(path)
    local_counts = {p: count_fn(p) for p in balanced_file_assignment(missing, world)[rank]}
    gathered: list = [None] * world
    dist.all_gather_object(gathered, local_counts)
    for part in gathered:
        counts.update(part or {})
    if set(counts) != set(files):
        raise RuntimeError(f"Failed to count reconstructions in {len(set(files) - set(counts))} H5 files")
    if rank == 0 and cache_path and missing:
        for path in files:
            stat = os.stat(path)
            cache["files"][path] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "reconstructions": counts[path]}
        target = Path(cache_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".partial")
        tmp.write_text(json.dumps(cache, sort_keys=True))
        os.replace(tmp, target)
    bins = balanced_file_assignment(files, world, weights=counts)
    if rank == 0:
        loads = [sum(counts[p] for p in b) for b in bins]
        print(f"[assignment] reconstructions per rank min={min(loads)} max={max(loads)}", flush=True)
    return bins


def stage_files(files: list[str], root: str, slot: int) -> list[str]:
    """Copy shards to ``root/rank_<slot>/`` (node-local disk); no-op when ``root`` is empty."""
    if not root:
        return files
    target = Path(root) / f"rank_{slot}"
    target.mkdir(parents=True, exist_ok=True)
    out = []
    for source in files:
        dest = target / Path(source).name
        if not dest.exists() or dest.stat().st_size != os.path.getsize(source):
            tmp = dest.with_suffix(dest.suffix + ".partial")
            shutil.copyfile(source, tmp)
            os.replace(tmp, dest)
        out.append(str(dest))
    return out


def _parallelize_network(network, mesh, mp_policy) -> None:
    from torch.distributed._composable.fsdp import fully_shard
    blocks = network.backbone.blocks
    for i, block in enumerate(blocks):
        blocks[i] = fully_shard(block, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=True)
    for previous, following in zip(blocks, blocks[1:]):
        previous.set_modules_to_forward_prefetch([following])
        following.set_modules_to_backward_prefetch([previous])
    fully_shard(network.backbone, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=True)
    network.dino_head = fully_shard(network.dino_head, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=True)
    network.ibot_head = fully_shard(network.ibot_head, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=True)


def fsdp_wrap(learner, device, compile_blocks: bool) -> None:
    """Compile each block (optional), shard every block and head (bf16 params, fp32 reduction), materialize, initialize."""
    from torch.distributed._composable.fsdp import MixedPrecisionPolicy
    from torch.distributed.device_mesh import init_device_mesh
    if compile_blocks:
        for network in (learner.student, learner.teacher, learner.gram_anchor):
            if network is not None:
                for block in network.backbone.blocks:
                    block.compile()
    mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("dp",))
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for network in (learner.student, learner.teacher, learner.gram_anchor):
        if network is not None:
            _parallelize_network(network, mesh, mp_policy)
    learner.to_empty(device=device)
    learner.initialize_student_teacher()


def checkpoint_root(output: Path) -> Path:
    return output / "checkpoints"


def resolve_checkpoint(output: Path, resume: str) -> Path | None:
    """'none' -> None; 'latest' -> the last complete checkpoint of this run (or None); else the given directory."""
    if not resume or resume.lower() == "none":
        return None
    if resume != "latest":
        return Path(resume)
    marker = checkpoint_root(output) / "latest.txt"
    if not marker.exists():
        return None
    path = checkpoint_root(output) / marker.read_text().strip()
    return path if (path / "COMPLETE").exists() else None


def save_checkpoint(output: Path, learner, optimizer, args, step: int, sampler_offset: int, rank: int) -> None:
    """DCP checkpoint (model + optimizer) + per-rank sampler offset and RNG states; keeps the last ``keep_checkpoints``."""
    from dinov3.checkpointer import save_checkpoint as dcp_save
    root = checkpoint_root(output)
    path = root / f"step_{step:08d}"
    dcp_save(path, iteration=step, model=learner, optimizer=optimizer, overwrite=True)
    tmp = path / f"runtime_rank{rank:04d}.pt.partial"
    torch.save({"sampler_offset": int(sampler_offset), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(),
                "numpy": np.random.get_state(), "python": random.getstate()}, tmp)
    os.replace(tmp, path / f"runtime_rank{rank:04d}.pt")
    if rank == 0:
        (path / "metadata.json").write_text(json.dumps({
            "format": CHECKPOINT_FORMAT, "step": step, "stage": args.stage, "sampler_offset_per_rank": sampler_offset,
            "world_size": dist.get_world_size(), "args": vars(args), "official_dinov3_commit": OFFICIAL_DINOV3_COMMIT,
        }, indent=2, sort_keys=True))
    dist.barrier()
    if rank == 0:
        (path / "COMPLETE").touch()
        latest_tmp = root / "latest.txt.partial"
        latest_tmp.write_text(path.name + "\n")
        os.replace(latest_tmp, root / "latest.txt")
        old = sorted(p for p in root.glob("step_*") if p.is_dir() and (p / "COMPLETE").exists())
        for stale in old[:-args.keep_checkpoints]:
            shutil.rmtree(stale)
        print(f"[checkpoint] saved step {step}: {path}", flush=True)
    dist.barrier()


def load_checkpoint(path: Path | None, learner, optimizer, sampler, args, rank: int) -> int:
    """Restore model, optimizer, sampler offset and RNG states; returns the step to continue from.

    Loading a checkpoint of another stage (phase 1 -> ``--stage gram``) is non-strict (the Gram anchor is new), restarts
    the schedule at step 0 and sets the Gram anchor to the loaded EMA teacher. Optimizer moments are carried over."""
    if path is None:
        if rank == 0:
            print("[resume] no checkpoint; starting from scratch", flush=True)
        return 0
    from dinov3.checkpointer import load_checkpoint as dcp_load
    if not (path / "COMPLETE").exists():
        raise RuntimeError(f"Incomplete checkpoint: {path}")
    source_stage = json.loads((path / "metadata.json").read_text())["stage"]
    same = source_stage == args.stage
    loaded_step = int(dcp_load(path, model=learner, optimizer=optimizer,
                               strict_loading=same and not args.resume_allow_partial))
    state = torch.load(path / f"runtime_rank{rank:04d}.pt", map_location="cpu", weights_only=False)
    sampler.set_offset(int(state["sampler_offset"]))
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    torch.cuda.set_rng_state(state["cuda"])
    step = loaded_step if same else 0
    if not same and learner.gram_anchor is not None:
        learner.reset_gram_anchor_from_teacher()
    if rank == 0:
        note = "" if same else f"; stage {source_stage} -> {args.stage}, schedule restarts at 0"
        print(f"[resume] loaded {path} at step {step}{note}", flush=True)
    return step


class StopController:
    """SIGTERM / SIGUSR1 request a checkpoint and exit at the next step boundary (agreed across ranks)."""

    def __init__(self) -> None:
        self.requested = False
        for sig in (signal.SIGTERM, signal.SIGUSR1):
            signal.signal(sig, self._signal)

    def _signal(self, signum, frame) -> None:
        self.requested = True

    def should_stop(self, device) -> bool:
        flag = torch.tensor(int(self.requested), device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

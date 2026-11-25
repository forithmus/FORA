"""Distributed runtime, RNG state and resumable checkpoints of the CT writer trainer.

Checkpoint format (``latest.pt``, ``step_NNNNNN.pt``, ``checkpoint_epochNNN.pt``): a dict with ``format``,
``metadata`` (run metadata incl. ``fingerprint``), ``epoch``, ``microstep``, ``optimizer_step``, ``trainable_model``
(full, unsharded tensors of every trainable parameter: LoRA, connector, vision tower), ``optimizer``, ``scheduler``,
``rng_states`` (one per rank) and ``world_size``. Sharded (FSDP2) tensors are gathered to rank 0 before saving.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

CHECKPOINT_FORMAT = "fora_writer_resume"


@dataclass(frozen=True)
class Runtime:
    rank: int
    local_rank: int
    world_size: int
    local_world_size: int
    device: torch.device

    @property
    def main(self) -> bool:
        return self.rank == 0


def init_runtime(pg_timeout_s: int) -> Runtime:
    local = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", timeout=timedelta(seconds=int(pg_timeout_s)), device_id=device)
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    return Runtime(dist.get_rank(), local, dist.get_world_size(), local_world, device)


def worker_init(threads: int):
    """DataLoader worker initializer: per-worker Python/NumPy seeds and a fixed torch thread count."""
    def init(_worker_id: int):
        value = torch.initial_seed() % (2**32)
        random.seed(value)
        np.random.seed(value)
        torch.set_num_threads(int(threads))
    return init


def atomic_save(value, path: Path):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def rng_state(runtime: Runtime) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(runtime.device),
    }


def restore_rng_state(state: dict, runtime: Runtime) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"], runtime.device)


def synchronize_checkpoint_decision(local_due: bool, local_stop: bool, vote: torch.Tensor) -> tuple[bool, bool]:
    """One globally agreed (checkpoint, stop) decision so every rank enters the same collectives."""
    if vote.shape != (2,) or vote.dtype != torch.uint8:
        raise ValueError("checkpoint vote must be a two-element uint8 tensor")
    vote[0] = int(local_due)
    vote[1] = int(local_stop)
    dist.all_reduce(vote, op=dist.ReduceOp.MAX)
    return bool(vote[0].item()), bool(vote[1].item())


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch, microstep, optimizer_step, runtime: Runtime,
                    metadata: dict):
    from fora_report.writer import fsdp as writer_fsdp
    rngs = [None] * runtime.world_size
    dist.all_gather_object(rngs, rng_state(runtime))
    trainable = writer_fsdp.full_trainable_state(model, runtime.main)
    optimizer_state = writer_fsdp.full_optimizer_state(optimizer, runtime.main)
    if runtime.main:
        atomic_save(
            {
                "format": CHECKPOINT_FORMAT,
                "format_version": 1,
                "metadata": metadata,
                "epoch": int(epoch),
                "microstep": int(microstep),
                "optimizer_step": int(optimizer_step),
                "trainable_model": trainable,
                "optimizer": optimizer_state,
                "scheduler": scheduler.state_dict(),
                "rng_states": rngs,
                "world_size": runtime.world_size,
            },
            path,
        )
        print(f"[checkpoint] {path} epoch={epoch + 1} next_microstep={microstep} optimizer_step={optimizer_step}",
              flush=True)
    dist.barrier(device_ids=[runtime.local_rank])

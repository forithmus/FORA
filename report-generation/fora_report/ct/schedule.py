"""Study order and learning-rate schedule of the CT writer trainer (paper: "Grounded report generation", training).

* :class:`TokenBalancedEpochSampler`: every rank sorts its studies by exact encoder token count and all ranks consume
  the same shuffled quantile order, so similarly expensive studies run in the same distributed step. Each epoch is a
  permutation of the rank's studies plus only the padding repeats needed to give every rank the same step count.
* :func:`cosine_scheduler`: linear warm-up then cosine decay to zero, per optimizer step. ``initial_horizon_steps``
  / ``initial_horizon_until_step`` define a two-horizon schedule: before
  ``initial_horizon_until_step`` the cosine runs over the initial horizon, afterwards over the final horizon, with the
  warm-up length of the initial horizon throughout (used by the stage-1 SFT config).
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch.utils.data import Sampler


class TokenBalancedEpochSampler(Sampler[int]):
    """Shuffle length-balanced global batches without changing epoch membership."""

    def __init__(self, token_counts: Sequence[int], steps: int, epoch: int, start: int, seed: int, *,
                 local_rank: int, local_world_size: int) -> None:
        self.token_counts = tuple(map(int, token_counts))
        self.size = len(self.token_counts)
        self.steps = int(steps)
        self.epoch = int(epoch)
        self.start = int(start)
        self.seed = int(seed)
        self.local_rank = int(local_rank)
        self.local_world_size = int(local_world_size)
        if self.size <= 0:
            raise ValueError("token-balanced sampling requires a non-empty dataset")
        if self.steps < self.size:
            raise ValueError(f"token-balanced steps cannot omit local studies: steps={self.steps} studies={self.size}")
        if not (0 <= self.start <= self.steps):
            raise ValueError("token-balanced sampler start must be within its step range")
        if any(count <= 0 for count in self.token_counts):
            raise ValueError("encoder token counts must all be positive")
        if self.local_world_size < 2 or self.local_world_size % 2 or not 0 <= self.local_rank < self.local_world_size:
            raise ValueError("token-balanced sampling requires an even local world size and a valid local rank")

    def _complete_order(self) -> list[int]:
        # Random tie-breaking keeps equal-size studies from retaining index adjacency across epochs.
        tie_generator = torch.Generator().manual_seed(self.seed + 1_000_003 * self.epoch + 10_007 * self.local_rank)
        randomized = torch.randperm(self.size, generator=tie_generator).tolist()
        sorted_indices = sorted(randomized, key=lambda index: self.token_counts[index])
        padding = self.steps - self.size
        if padding:
            sorted_indices.extend(randomized[position % self.size] for position in range(padding))
            sorted_indices.sort(key=lambda index: self.token_counts[index])
        # One shuffled quantile permutation shared by every rank (matched quantiles per step).
        batch_generator = torch.Generator().manual_seed(self.seed + 1_000_003 * self.epoch + 53)
        quantiles = torch.randperm(self.steps, generator=batch_generator).tolist()
        return [sorted_indices[position] for position in quantiles]

    def __iter__(self):
        yield from self._complete_order()[self.start:]

    def __len__(self) -> int:
        return self.steps - self.start

    def summary(self) -> dict:
        ordered = sorted(self.token_counts)
        return {
            "mode": "token_balanced_shuffled",
            "studies": self.size,
            "steps": self.steps,
            "ddp_padding_repeats": self.steps - self.size,
            "minimum_encoder_tokens": ordered[0],
            "median_encoder_tokens": ordered[len(ordered) // 2],
            "maximum_encoder_tokens": ordered[-1],
            "local_world_size": self.local_world_size,
            "rank_alignment": "matched_quantiles",
            "batch_order": "epoch_shuffled",
            "coverage": "one_pass_plus_only_ddp_padding",
        }


def cosine_factor(step: int, total_steps: int, warmup: int) -> float:
    if warmup and step < warmup:
        return float(step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def cosine_scheduler(optimizer, total_steps: int, warmup_fraction: float, *, initial_horizon_steps: int = 0,
                     initial_horizon_until_step: int = 0, last_epoch: int = -1):
    """LambdaLR; ``step`` is the number of optimizer updates already taken."""
    horizon = int(initial_horizon_steps) if initial_horizon_steps else int(total_steps)
    warmup = int(horizon * warmup_fraction)
    if not 0 <= warmup < total_steps:
        raise ValueError(f"warmup steps must be in [0, {total_steps}), got {warmup}")
    until = int(initial_horizon_until_step) if initial_horizon_steps else 0

    def factor(step):
        return cosine_factor(step, horizon if step < until else int(total_steps), warmup)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor, last_epoch=last_epoch)

"""View placement, student augmentation, 3-D block masks and batching shared by the CT and X-ray pre-training data
(Methods, "Image encoder pre-training").

Two global crops and 8 local crops per sample:
* global 0 is placed in the volume; global 1 overlaps it by at least ``global_overlap`` (25 %) of the crop size per axis and
  its offset is snapped to a multiple of the patch (2, 16, 16), so every token in the overlap has an exact twin;
* every local crop lies inside one of the two global crops (chosen at random);
* no rescaling and no flips: physical size and laterality stay meaningful;
* student views get intensity gain/bias, Gaussian noise and a mild in-plane blur; teacher views are left unchanged.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Sampler

DEFAULT_GLOBAL_OVERLAP = 0.25
PATCH = (2, 16, 16)


@dataclass(frozen=True)
class CropSpec:
    global_shape: tuple[int, int, int] = (64, 192, 192)
    local_shape: tuple[int, int, int] = (32, 96, 96)
    local_crops: int = 8


class InfiniteReportSampler(Sampler):
    """Infinite deterministic per-rank stream with exact mid-stream resume; yields (index, local epoch).

    Every rank performs the same number of steps although shard sizes differ; within a local epoch every item is
    visited once."""

    def __init__(self, n: int, seed: int, offset: int = 0) -> None:
        if n < 1:
            raise ValueError("Each rank must own at least one indexed item")
        self.n, self.seed, self.offset = int(n), int(seed), int(offset)

    def set_offset(self, offset: int) -> None:
        self.offset = int(offset)

    def __iter__(self):
        while True:
            epoch, pos = divmod(self.offset, self.n)
            order = torch.randperm(self.n, generator=torch.Generator().manual_seed(self.seed + epoch)).tolist()
            for j in range(pos, self.n):
                self.offset += 1
                yield (order[j], epoch)

    def __len__(self) -> int:
        return 2 ** 31


def sample_rng(seed: int, idx: int, epoch: int) -> random.Random:
    """Per-(item, epoch) RNG: rendering is deterministic and resumable with persistent DataLoader workers."""
    value = seed + 0x9E3779B1 * int(idx) + 0x85EBCA77 * int(epoch)
    return random.Random(value & ((1 << 63) - 1))


def crop_at_start(x: np.ndarray, shape, start, pad_value: float) -> np.ndarray:
    """Axis-preserving crop at an explicit start; the part outside ``x`` is filled with ``pad_value``."""
    out = np.full(shape, pad_value, dtype=np.float32)
    d, h, w = (min(have - begin, want) for have, want, begin in zip(x.shape, shape, start))
    z, y, xx = start
    out[:d, :h, :w] = x[z:z + d, y:y + h, xx:xx + w]
    return out


def shifted_start(origin, container_shape, crop_shape, overlap: float, rng: random.Random):
    """Second-crop start whose intersection with ``origin`` spans at least ``overlap`` x crop per axis (inside the volume)."""
    if not 0.0 <= overlap <= 1.0:
        raise ValueError("overlap must be in [0, 1]")
    out = []
    for have, want, o in zip(container_shape, crop_shape, origin):
        if have <= want:
            out.append(0)
            continue
        max_shift = int((1.0 - overlap) * want)
        lo, hi = max(0, o - max_shift), min(have - want, o + max_shift)
        out.append(rng.randint(lo, hi) if hi > lo else lo)
    return tuple(out)


def snap_second_start(origin, second, container_shape, crop_shape, patch=PATCH):
    """Move ``second`` so that (second - origin) is a multiple of the patch per axis, staying inside the container."""
    out = []
    for o, s, have, want, p in zip(origin, second, container_shape, crop_shape, patch):
        k = int(round((s - o) / p))
        lo = -(o // p)
        hi = (max(have - want, 0) - o) // p
        hi = max(hi, lo)
        out.append(o + min(max(k, lo), hi) * p)
    return tuple(out)


def start_inside(outer_start, outer_shape, inner_shape, container_shape, rng: random.Random):
    """Start of an inner crop inside the outer crop and, where possible, inside the volume."""
    out = []
    for o, big, small, have in zip(outer_start, outer_shape, inner_shape, container_shape):
        if small > big:
            raise ValueError("local crop larger than the global crop")
        hi = min(o + big, have) - small
        out.append(rng.randint(o, hi) if hi > o else o)
    return tuple(out)


def student_distortion(x: torch.Tensor, rng: random.Random) -> torch.Tensor:
    """Intensity gain/bias (p 0.8), Gaussian noise (p 0.5), in-plane 3 x 3 box blur (p 0.25); never changes orientation."""
    y = x.float()
    if rng.random() < 0.8:
        y = y * rng.uniform(0.92, 1.08) + rng.uniform(-0.04, 0.04)
    if rng.random() < 0.5:
        sigma = rng.uniform(0.005, 0.035)
        generator = torch.Generator().manual_seed(rng.randrange(2 ** 63 - 1))
        y = y + torch.randn(y.shape, generator=generator, dtype=y.dtype) * sigma
    if rng.random() < 0.25:
        y = F.avg_pool3d(y[None], kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))[0]
    return y.clamp_(-1, 1).to(torch.bfloat16)


def make_block_mask_3d(grid, ratio: float, generator: torch.Generator | None = None) -> torch.Tensor:
    """Exact-cardinality 3-D block mask on the patch lattice (mildly anisotropic cuboids), flattened."""
    total = math.prod(grid)
    target = min(total - 1, max(1, int(round(total * float(ratio)))))
    mask = torch.zeros(grid, dtype=torch.bool)
    tries = 0
    while int(mask.sum()) < target and tries < 64:
        remain = target - int(mask.sum())
        volume = max(1, min(remain, int(torch.randint(4, max(5, remain + 1), (1,), generator=generator))))
        dz = min(grid[0], max(1, round(volume ** (1 / 3) * 0.7)))
        dy = min(grid[1], max(1, round(math.sqrt(volume / dz))))
        dx = min(grid[2], max(1, math.ceil(volume / (dz * dy))))
        z = int(torch.randint(0, grid[0] - dz + 1, (1,), generator=generator))
        y = int(torch.randint(0, grid[1] - dy + 1, (1,), generator=generator))
        x = int(torch.randint(0, grid[2] - dx + 1, (1,), generator=generator))
        mask[z:z + dz, y:y + dy, x:x + dx] = True
        tries += 1
    flat = mask.flatten()
    count = int(flat.sum())
    if count < target:
        choices = (~flat).nonzero().flatten()
        flat[choices[torch.randperm(len(choices), generator=generator)[:target - count]]] = True
    elif count > target:
        choices = flat.nonzero().flatten()
        flat[choices[torch.randperm(len(choices), generator=generator)[:count - target]]] = False
    return flat


def overlap_pairs(offset, grid):
    """Flat token indices (in crop A, in crop B) of every token that covers the same voxels in both crops."""
    ranges = []
    for d, g in zip(offset, grid):
        lo, hi = max(0, d), min(g, g + d)
        if hi <= lo:
            return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
        ranges.append(torch.arange(lo, hi))
    z, y, x = torch.meshgrid(*ranges, indexing="ij")
    gz, gy, gx = grid
    a = (z * gy + y) * gx + x
    b = ((z - offset[0]) * gy + (y - offset[1])) * gx + (x - offset[2])
    return a.flatten(), b.flatten()


def token_offset(g0, g1, patch=PATCH):
    out = []
    for a, b, p in zip(g0, g1, patch):
        if (b - a) % p:
            raise ValueError("global crops are not aligned on the patch lattice")
        out.append((b - a) // p)
    return tuple(out)


def collate(samples: list[dict], patch_size=PATCH, mask_ratio=(0.1, 0.5)) -> dict:
    """Stack views as [views, B, 1, D, H, W]; one block mask per student global crop (ratio ~ U(mask_ratio)) with
    per-mask token weights 1 / count; twin-token indices of the two global crops for the cross-crop term."""
    teacher = torch.stack([s["teacher_global"] for s in samples], dim=1)
    student = torch.stack([s["student_global"] for s in samples], dim=1)
    local = torch.stack([s["student_local"] for s in samples], dim=1)
    _, _, _, d, h, w = student.shape
    grid = (d // patch_size[0], h // patch_size[1], w // patch_size[2])
    masks = []
    for crop in range(2):
        for sample in samples:
            generator = torch.Generator().manual_seed(int(sample["mask_seed"]) + crop * 1_000_003)
            ratio = float(torch.empty(1).uniform_(*mask_ratio, generator=generator))
            masks.append(make_block_mask_3d(grid, ratio, generator=generator))
    masks = torch.stack(masks)
    indices = masks.flatten().nonzero().flatten()
    weights = (1.0 / masks.sum(-1).clamp(min=1)).unsqueeze(-1).expand_as(masks)[masks]
    n = math.prod(grid)
    pa, pb, ps = [], [], []
    for s, sample in enumerate(samples):
        a, b = overlap_pairs(token_offset(*sample["global_starts"], patch_size), grid)
        pa.append(a + s * n)
        pb.append(b + s * n)
        ps.append(torch.full((a.numel(),), s, dtype=torch.long))
    return {
        "teacher_global": teacher, "student_global": student, "student_local": local,
        "masks": masks, "mask_indices": indices, "mask_weights": weights,
        "sample_weights": torch.tensor([float(s.get("reconstruction_weight", 1.0)) for s in samples], dtype=torch.float32),
        "cross_a": torch.cat(pa), "cross_b": torch.cat(pb), "cross_sample": torch.cat(ps),
    }

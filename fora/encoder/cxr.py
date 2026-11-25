"""X-ray side of the encoder: film selection, 1,024-px tiling with token ownership, and the split (frozen 1-23 | block 24)
teacher used online by X-ray sentence alignment, X-ray MIL and inference (Methods, "Chest X-ray").

A prepared film (content crop, percentile stretch to [-1, 1], native resolution; ``fora.geometry.xray``) is padded with
air to the 16-px lattice and cut into T x T tiles (T = 1,024 = the X-ray DINO global crop). The last tile of an axis is
snapped back onto the lattice, so tiles overlap; every film-grid token is owned by exactly one tile (overlaps are split at
their midpoint). A film of H x W pixels therefore gives ceil(H/16) * ceil(W/16) tokens, each computed with the full
context of one tile. Each tile enters the encoder as a two-slice volume (2, T, T).
"""
from __future__ import annotations

import numpy as np
import torch

from .backbone import CXR_ROPE_MIN_PERIOD_MM, CXR_VOXEL_SPACING_MM, load_teacher
from .blocks import block23_tokens, final_block, split_top
from .positions import position_features

PATCH_PX = 16
TILE_PX = 1024
AIR = -1.0
MIN_FILM_DIM = 256          # films with a side below this are skipped
MAX_FILM_ASPECT = 3.0       # and films more elongated than this


def valid_film_shape(shape) -> bool:
    """2-D film with sane dimensions (drops slivers and extreme aspect ratios)."""
    if len(shape) < 2:
        return False
    h, w = int(shape[0]), int(shape[1])
    return min(h, w) >= MIN_FILM_DIM and max(h, w) / max(1, min(h, w)) <= MAX_FILM_ASPECT


def axis_tiles(n: int, T: int = TILE_PX):
    """Tile starts (px, on the 16-px lattice), owned token ranges [lo, hi) per tile, and the token count of an axis."""
    g = -(-n // PATCH_PX)
    G, tt = g * PATCH_PX, T // PATCH_PX
    starts = sorted({min(s, max(G - T, 0)) for s in range(0, G, T)})
    a = [s // PATCH_PX for s in starts]
    own = []
    for i, ai in enumerate(a):
        lo = 0 if i == 0 else (ai + a[i - 1] + tt) // 2
        hi = g if i == len(a) - 1 else (a[i + 1] + ai + tt) // 2
        own.append((lo, min(hi, ai + tt, g)))
    return starts, own, g


def film_tiles(u: np.ndarray, T: int = TILE_PX, freqs: int = 4):
    """Prepared film (H, W) -> (tiles [n, T, T] bf16, owned local token ids per tile, film-grid position features, grid)."""
    H, W = u.shape
    ys, oy, gy = axis_tiles(H, T)
    xs, ox, gx = axis_tiles(W, T)
    tt = T // PATCH_PX
    tiles, sel, flat = [], [], []
    for y0, (ylo, yhi) in zip(ys, oy):
        for x0, (xlo, xhi) in zip(xs, ox):
            t = np.full((T, T), AIR, np.float32)
            sub = u[y0:y0 + T, x0:x0 + T]
            t[:sub.shape[0], :sub.shape[1]] = sub
            tiles.append(t)
            ty, tx = np.meshgrid(np.arange(ylo, yhi), np.arange(xlo, xhi), indexing="ij")
            a, b = y0 // PATCH_PX, x0 // PATCH_PX
            sel.append(((ty - a) * tt + (tx - b)).ravel().astype(np.int64))
            flat.append((ty * gx + tx).ravel().astype(np.int64))
    allflat = np.concatenate(flat)
    if len(allflat) != gy * gx or len(np.unique(allflat)) != gy * gx:
        raise AssertionError("tile ownership must partition the film grid")
    pf = position_features(allflat, (1, gy, gx), freqs)
    return torch.from_numpy(np.stack(tiles)).to(torch.bfloat16), sel, pf, (gy, gx)


class CXRSplitEncoder:
    """Frozen X-ray teacher blocks 1-23 plus a trainable copy of (block 24, norm) = ``self.top``.

    Both halves are called exactly as the DINO backbone calls them (autocast bf16; the RoPE of the tile grid is computed
    once and reused). ``top`` is kept in eval mode: block 24 then runs without stochastic depth while its parameters train.
    """

    def __init__(self, teacher_path, block24_state: dict | None, device):
        self.device = torch.device(device)
        self.backbone = load_teacher(teacher_path, "cpu", voxel_spacing_mm=CXR_VOXEL_SPACING_MM,
                                     rope_min_period_mm=CXR_ROPE_MIN_PERIOD_MM)
        self.step = self.backbone.teacher_step
        self.top = split_top(self.backbone, block24_state).to(self.device).eval()
        self.top.requires_grad_(True)
        self.backbone = self.backbone.to(self.device)
        self.rope = None

    def autocast(self):
        return torch.autocast(self.device.type, dtype=torch.bfloat16)

    @torch.no_grad()
    def h23(self, tiles: torch.Tensor, batch: int = 8) -> torch.Tensor:
        """tiles [n, T, T] bf16 -> block-23 residual stream [n, 5 + (T/16)^2, 1024] fp32."""
        out = []
        for i in range(0, len(tiles), batch):
            x = tiles[i:i + batch].to(self.device, non_blocking=True)[:, None, None].expand(-1, 1, 2, -1, -1)
            with self.autocast():
                grid = tuple(s // p for s, p in zip(x.shape[2:], self.backbone.patch_size))
                rope = self.rope[1] if self.rope is not None and self.rope[0] == grid else None
                tok, rope = block23_tokens(self.backbone, x, rope)
                self.rope = (grid, rope)
            out.append(tok.float())
        return torch.cat(out)

    def b24(self, h: torch.Tensor) -> torch.Tensor:
        """(block 24, norm) on block-23 tokens of tiles -> patch tokens [n, (T/16)^2, 1024]."""
        with self.autocast():
            return final_block(self.top.block, self.top.norm, h, self.rope[1])

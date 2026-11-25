"""Encode one view of a CT series into block-23 body tokens with the frozen encoder (Methods, "Block-23 token cache").

The resampled, body-cropped HU volume is cut into non-overlapping 64 x 192 x 192 crops (air-padded at the far edges),
each crop is window-normalized to [-1, 1] and run through blocks 1-23 of the teacher backbone; their output is
kept. Only crops that contain at least one body token are kept. The body-token mask is computed
from the FULL-window volume (`body_volume`) so all views of a series share the same token set.

The backbone is the ViT-L teacher returned by `fora.encoder.backbone.load_teacher`; blocks 1-23 are run by
`fora.encoder.blocks.block23_tokens` (bf16 autocast and flash attention on CUDA).
"""
from __future__ import annotations

import itertools
import time

import numpy as np
import torch

from ..encoder.blocks import block23_tokens
from ..geometry.tokens import PATCH, body_mask, token_mean_hu
from ..geometry.windows import window_normalize
from .format import CROP_TOKENS, PREFIX, WIDTH


class Block23Encoder:
    """Block-23 residual stream of a teacher backbone for whole volumes."""

    def __init__(self, backbone, device, brick_batch: int = 4):
        self.bb = backbone
        self.device = torch.device(device)
        self.brick_batch = int(brick_batch)

    @torch.no_grad()
    def encode(self, vol: np.ndarray, hu_window, body_volume: np.ndarray, encoding: dict, patch=PATCH):
        """-> (h23 uint16 [K, 4613, 1024] bf16 bits, kept crop ids int32 [K], body flat indices, metadata dict)."""
        started = time.time()
        assert body_volume.shape == vol.shape
        body = body_mask(token_mean_hu(body_volume, patch=patch))
        grid = body.shape
        lo, hi = map(float, hu_window)
        crop = tuple(encoding['crop'])
        stride = tuple(encoding['stride'])
        extent = tuple(c // p for c, p in zip(crop, patch))
        assert encoding['blend'] == 'none' and stride == crop and int(np.prod(extent)) == CROP_TOKENS
        starts = list(itertools.product(*(range(0, n, s) for n, s in zip(vol.shape, stride))))
        counts = [-(-g // e) for g, e in zip(grid, extent)]
        assert len(starts) == int(np.prod(counts))
        keep = []
        for cid, origin in enumerate(starts):
            o = tuple(v // q for v, q in zip(origin, patch))
            if body[tuple(slice(v, v + e) for v, e in zip(o, extent))].any():
                keep.append(cid)
        indices = np.flatnonzero(body.ravel())
        if not len(indices):
            raise ValueError('Empty body mask')
        out = np.empty((len(keep), PREFIX + CROP_TOKENS, WIDTH), np.uint16)
        cuda = self.device.type == 'cuda'
        for b in range(0, len(keep), self.brick_batch):
            ids = keep[b:b + self.brick_batch]
            crops = []
            for cid in ids:
                origin = starts[cid]
                c = np.full(crop, -1000, np.float32)
                size = tuple(min(n - o, k) for n, o, k in zip(vol.shape, origin, crop))
                c[tuple(slice(0, n) for n in size)] = vol[tuple(slice(o, o + n) for o, n in zip(origin, size))]
                crops.append(torch.as_tensor(window_normalize(c, lo, hi)))
            image = torch.stack(crops).to(self.device)
            if cuda:
                from torch.nn.attention import SDPBackend, sdpa_kernel
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION), torch.autocast('cuda', dtype=torch.bfloat16):
                    h = block23_tokens(self.bb, image)[0].detach()
            else:
                h = block23_tokens(self.bb, image)[0].detach().to(torch.bfloat16)
            assert h.dtype == torch.bfloat16 and h.shape == (len(ids), PREFIX + CROP_TOKENS, WIDTH)
            if not bool(torch.isfinite(h).all()):
                raise ValueError('Nonfinite block-23 tokens')
            out[b:b + len(ids)] = h.contiguous().view(torch.int16).cpu().numpy().view(np.uint16)
            del h, image
        meta = dict(grid=list(grid), volume_shape=list(vol.shape), n_grid_tokens=int(np.prod(grid)), n_tokens=len(indices),
                    crops=len(starts), kept_crops=len(keep), seconds=time.time() - started)
        return out, np.asarray(keep, np.int32), indices, meta

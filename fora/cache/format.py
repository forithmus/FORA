"""Block-23 token cache format `fora_int8_tokens` (Methods, "Block-23 token cache").

The frozen encoder runs on non-overlapping 64 x 192 x 192 voxel crops (32 x 12 x 12 = 4,608 tokens + 5 prefix tokens,
CLS and 4 registers). The residual stream after block 23 is stored for every crop that contains a body token, and only
for body tokens. Per view (H5 group `vNN`) of a series file:

  prefix       [K, 5, 1024]  uint16  exact bf16 bits of the prefix tokens of each kept crop
  q            [N, 1024]     int8    body tokens in crop-major order (fora.encoder.blocks.crop_layout), per-crop per-channel absmax / 127
  scale        [K, 1024]     float16 the quantization scale (unused for the outlier channels)
  outlier_idx  [8]           int16   the view's 8 channels with the largest |x|, stored exactly
  outlier      [N, 8]        uint16  exact bf16 bits of those channels
  crop_ids     [K]           int32   z-major index of each kept crop

Series-file attributes: `identity`, `fingerprint`, `format`, `provenance_json`, `complete`; dataset `flat_indices`
(body tokens, grid order). Group attributes: `metadata_json` (grid, volume shape, crop counts), `window_json` (view
kind and HU window), `complete`. Block 24 later attends over the prefix + body tokens of each crop.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path

import numpy as np
import torch

FORMAT = 'fora_int8_tokens'
CROP = (64, 192, 192)          # voxels; stride = crop, no blending
EXTENT = (32, 12, 12)          # tokens per crop for the (2, 16, 16) patch
CROP_TOKENS = 4608
PREFIX = 5
N_OUTLIER = 8
WIDTH = 1024


def quantize(h23: np.ndarray, flat: np.ndarray, grid) -> dict:
    """h23 [K, 4613, 1024] uint16 (bf16 bits) of the body-containing crops -> the compact per-view arrays."""
    from ..encoder.blocks import crop_layout
    order, ids, starts, local, _ = crop_layout(flat, grid)
    x = torch.from_numpy(h23.view(np.int16)).view(torch.bfloat16)
    rows = np.repeat(np.arange(len(starts) - 1), np.diff(starts))
    body = x[torch.from_numpy(rows), torch.from_numpy(local + PREFIX)].float()             # [N, 1024] crop-major
    amax = body.abs().amax(0)
    oidx = torch.argsort(amax, descending=True)[:N_OUTLIER].sort().values
    keep = torch.ones(body.shape[1], dtype=torch.bool)
    keep[oidx] = False
    r = torch.from_numpy(rows)
    scale = torch.zeros(len(starts) - 1, body.shape[1])
    scale.index_reduce_(0, r, body.abs(), 'amax', include_self=True)
    scale = (scale / 127).clamp_min(1e-8)
    q = torch.round(body / scale[r]).clamp(-127, 127).to(torch.int8)
    q[:, ~keep] = 0
    out = body[:, oidx].to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
    prefix = np.ascontiguousarray(h23[:, :PREFIX])
    return dict(prefix=prefix, q=q.numpy(), scale=scale.half().numpy(), outlier_idx=oidx.short().numpy(), outlier=out,
                crop_ids=ids.astype(np.int32))


def dequantize(v: dict, rows: np.ndarray) -> np.ndarray:
    """Inverse of `quantize` on the CPU: body tokens [N, 1024] float32 (rows = crop index of each token)."""
    x = v['q'].astype(np.float32) * v['scale'][rows].astype(np.float32)
    x[:, v['outlier_idx']] = torch.from_numpy(v['outlier'].view(np.int16)).view(torch.bfloat16).float().numpy()
    return x


def write_group(hf, name: str, arrays: dict, metadata: dict, window: dict):
    """Write one quantized view group with the chunking used for the training cache."""
    g = hf.create_group(name)
    for k, x in arrays.items():
        g.create_dataset(k, data=x, chunks=(min(4096, len(x)),) + x.shape[1:] if k in ('q', 'outlier') and len(x) else None)
    g.attrs['metadata_json'] = json.dumps(metadata)
    g.attrs['window_json'] = json.dumps(window)
    g.attrs['complete'] = 1
    return g


def read_view(hf, group: str) -> dict:
    g = hf[group]
    return dict(prefix=g['prefix'][:], q=g['q'][:], scale=g['scale'][:], outlier_idx=g['outlier_idx'][:],
                outlier=g['outlier'][:], crop_ids=g['crop_ids'][:])


def valid_series_file(path: Path, identity: str) -> bool:
    """True if a node-local series file exists, is complete and carries the expected identity hash."""
    import h5py
    if not path.exists():
        return False
    try:
        with h5py.File(path, 'r') as f:
            return f.attrs.get('identity') == identity and f.attrs.get('complete', 0) == 1
    except OSError:
        return False


def write_json_atomic(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


class PackSlice(io.RawIOBase):
    """Read-only file object over one series H5 embedded in a pack (file, offset, size); h5py can open it directly."""

    def __init__(self, fn, off, size):
        self.f = open(fn, 'rb')
        self.off = off
        self.size = size
        self.pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, p, whence=0):
        self.pos = p if whence == 0 else (self.pos + p if whence == 1 else self.size + p)
        return self.pos

    def readinto(self, b):
        n = max(0, min(len(b), self.size - self.pos))
        self.f.seek(self.off + self.pos)
        d = self.f.read(n)
        b[:len(d)] = d
        self.pos += len(d)
        return len(d)

    def close(self):
        self.f.close()
        super().close()


def open_series(path):
    """h5py.File over a local series file path or a (pack_file, offset, file_size) triple."""
    import h5py
    return h5py.File(PackSlice(*path) if isinstance(path, (tuple, list)) else os.fspath(path), 'r')

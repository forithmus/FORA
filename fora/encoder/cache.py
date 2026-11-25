"""Reader for the CT block-23 token cache (format ``fora_int8_tokens``), the input of sentence alignment and MIL.

The cache is written by ``data-preprocessing`` (frozen encoder, blocks 1-23, every 64 x 192 x 192 crop that contains a body
token). One HDF5 file per series:

    flat_indices  [N]            body-token indices in the view's (gz, gy, gx) token grid
    v<k>/         one group per view (full window, each distinct native window, 10 mm lung MIP)
        attrs metadata_json      JSON with at least {"grid": [gz, gy, gx]} (v00 holds the grid of the series)
        prefix       [K, 5, 1024]  uint16  exact bf16 bits of CLS + 4 registers per kept crop
        q            [N, 1024]     int8    body tokens in crop-major order (``blocks.crop_layout``)
        scale        [K, 1024]     float16 per crop and channel: absmax / 127
        outlier_idx  [8]           int16   the view's 8 largest-|x| channels
        outlier      [N, 8]        uint16  exact bf16 bits of those channels
        crop_ids     [K]           int32   kept crop ids (z-major)

Series files are appended into pack files; a case manifest line (``cases.jsonl``) lists for every view the pack ``file``,
``offset``, ``file_size``, ``relative_file``, ``group``, ``n_tokens`` and ``window`` ({"kind": full|native|lung_mip_10mm, ...}).
``restore_from_pack`` copies a series file out of its pack into a local cache directory.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch

from .blocks import PREFIX, crop_layout
from .positions import position_features

FORMAT = "fora_int8_tokens"
N_OUTLIERS = 8


def restore_from_pack(view: dict, dest: Path) -> None:
    """Copy bytes [offset, offset + file_size) of ``view['file']`` to ``dest`` (atomic rename)."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".copying")
    remaining = int(view["file_size"])
    with Path(view["file"]).open("rb") as source, tmp.open("wb") as target:
        source.seek(int(view.get("offset", 0)))
        while remaining:
            data = source.read(min(8 * 1024 ** 2, remaining))
            if not data:
                raise IOError("Truncated feature pack")
            target.write(data)
            remaining -= len(data)
    if tmp.stat().st_size != int(view["file_size"]):
        raise IOError(f"Restored size mismatch for {dest}")
    tmp.replace(dest)


def local_view_file(view: dict, cache_root: Path) -> Path:
    """Path of a view's series file under ``cache_root``; restored from its pack when absent or truncated."""
    path = Path(cache_root) / view["relative_file"]
    if not path.exists() or path.stat().st_size != int(view["file_size"]):
        restore_from_pack(view, path)
    return path


def load_case(case: dict, cache_root: Path, position_freqs: int = 4) -> list[dict]:
    """All views of one report -> host arrays per view (crop-major body tokens) + Fourier position features."""
    out, layouts = [], {}
    for view in case["views"]:
        path = local_view_file(view, cache_root)
        with h5py.File(path, "r") as f:
            if str(path) not in layouts:
                flat = f["flat_indices"][:]
                grid = json.loads(f["v00"].attrs["metadata_json"])["grid"]
                order, ids, starts, local_pos, _ = crop_layout(flat, grid)
                layouts[str(path)] = (ids, starts, local_pos, position_features(flat[order], grid, position_freqs))
            ids, starts, local_pos, pf = layouts[str(path)]
            g = f[view["group"]]
            if not np.array_equal(g["crop_ids"][:], ids):
                raise ValueError(f"{path}:{view['group']}: stored crops are not the body-containing crops")
            out.append(dict(prefix=g["prefix"][:], q=g["q"][:], scale=g["scale"][:], oidx=g["outlier_idx"][:],
                            outlier=g["outlier"][:], starts=starts, local=local_pos, pos=pf))
    return out


def dequantize(view: dict) -> np.ndarray:
    """Reference CPU dequantization of one view's body tokens -> float32 [N, 1024] (``CachedBag`` does it on device)."""
    rows = np.repeat(np.arange(len(view["starts"]) - 1), np.diff(view["starts"]))
    x = view["q"].astype(np.float32) * view["scale"][rows].astype(np.float32)
    x[:, view["oidx"]] = torch.from_numpy(view["outlier"].view(np.int16)).view(torch.bfloat16).float().numpy()
    return x


class CachedBag:
    """The views of one report on the device (int8 tokens), or streamed from host RAM when larger than ``gpu_gib``."""

    def __init__(self, views: list[dict], device, gpu_gib: float = 20.0):
        self.device = torch.device(device)
        self.host = sum(v["q"].nbytes for v in views) > gpu_gib * 1024 ** 3
        dev = lambda t: t if self.host else t.to(self.device, non_blocking=False)
        self.views = []
        for v in views:
            starts = v["starts"]
            rows = np.repeat(np.arange(len(starts) - 1), np.diff(starts))
            self.views.append(dict(
                starts=starts, pos=torch.from_numpy(v["pos"]).to(self.device),
                prefix=torch.from_numpy(v["prefix"].view(np.int16)).view(torch.bfloat16).to(self.device),
                q=dev(torch.from_numpy(v["q"])), scale=torch.from_numpy(v["scale"]).to(self.device),
                oidx=torch.from_numpy(v["oidx"].astype(np.int64)).to(self.device),
                outlier=dev(torch.from_numpy(v["outlier"].view(np.int16)).view(torch.bfloat16)),
                local=torch.from_numpy(v["local"].astype(np.int64)).to(self.device),
                rows=torch.from_numpy(rows).to(self.device)))

    def _autocast(self):
        return torch.autocast(self.device.type, dtype=torch.bfloat16)

    def view_chunks(self, index: int, block, crops: int):
        """Block 24 over ``crops`` crops at a time of view ``index`` -> (tokens [n, C] fp32, position features [n, F])."""
        v = self.views[index]
        K = len(v["starts"]) - 1
        for i in range(0, K, crops):
            j = min(K, i + crops)
            s, e = int(v["starts"][i]), int(v["starts"][j])
            body = v["q"][s:e].to(self.device).float() * v["scale"][v["rows"][s:e]].float()
            body[:, v["oidx"]] = v["outlier"][s:e].to(self.device).float()
            lens = np.diff(v["starts"][i:j + 1])
            L, B = int(lens.max()), j - i
            x = torch.zeros(B, PREFIX + L, body.shape[1], dtype=torch.bfloat16, device=self.device)
            x[:, :PREFIX] = v["prefix"][i:j]
            pos = torch.zeros(B, L, dtype=torch.long, device=self.device)
            ok = torch.zeros(B, PREFIX + L, dtype=torch.bool, device=self.device)
            ok[:, :PREFIX] = True
            o = 0
            for b, n in enumerate(lens):
                n = int(n)
                x[b, PREFIX:PREFIX + n] = body[o:o + n].to(torch.bfloat16)
                pos[b, :n] = v["local"][s + o:s + o + n]
                ok[b, PREFIX:PREFIX + n] = True
                o += n
            with self._autocast():
                y = block(x, pos, ok)
            yield y[:, PREFIX:][ok[:, PREFIX:]].float(), v["pos"][s:e]

    def chunks(self, block, crops: int):
        """All views in order -> (tokens, position features, view index)."""
        for k in range(len(self.views)):
            for y, pf in self.view_chunks(k, block, crops):
                yield y, pf, k

"""CT study bags for the MIL head: cached block-23 tokens, per-token region bits and per-finding route masks
(Methods, "Finding classifier").

A study bag is every view of every series of one report. ``load_study`` reads the views from the block-23 cache
(``fora.encoder.cache.load_case``) and the per-token CADS region bits (uint64, one bit per routed region, written by
data-preprocessing as ``<region_root>/<relative_file>.npz`` with key ``bits``, aligned with ``flat_indices``).

``RoutedBag`` holds the views on the device and builds the route mask of every finding: token t is inside the route of
finding c if its region bits intersect the finding's route bits. If fewer than ``min_route`` (8) tokens of a view fall
inside a finding's route, the whole view counts as inside for that finding. If a view has no hip-bone token, its
``pelvis`` bit is cleared (without hip bones the pelvis region of the region builder covers the whole lower abdomen).

``make_factory`` turns a bag into the token factory of ``fora.mil.model.forward/backward``: block 24 over crops of the
cache, standardisation (y - mean) / rms, training-time token drop (rows, p = 0.2) and feature drop (entries, p = 0.1,
inverted scaling) from a seeded generator so that both passes of the streamed gradient see identical inputs, and the
Fourier position features appended.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch

from fora.encoder.blocks import crop_layout
from fora.encoder.cache import CachedBag, load_case, local_view_file
from fora.regions import REGION_NAMES

PELVIS_BIT = REGION_NAMES.index("pelvis")
HIP_BIT = REGION_NAMES.index("pelvic_bones_hips")


def load_region_bits(view: dict, cache_root: Path, region_root: Path) -> np.ndarray:
    """Region bits of one view's series in crop-major token order (int64 view of the uint64 bits)."""
    path = local_view_file(view, cache_root)
    with h5py.File(path, "r") as f:
        flat = f["flat_indices"][:]
        grid = json.loads(f["v00"].attrs["metadata_json"])["grid"]
    rp = Path(region_root) / (view["relative_file"] + ".npz")
    with np.load(rp) as z:
        bits = z["bits"].view(np.int64)
    if len(bits) != len(flat):
        raise ValueError(f"region bits of {view['relative_file']} do not align with flat_indices")
    return bits[crop_layout(flat, grid)[0]]


def load_study(case: dict, cache_root: Path, region_root: Path, position_freqs: int = 4) -> list[dict]:
    """All views of one report -> host arrays (block-23 cache) + region bits per token."""
    views = load_case(case, cache_root, position_freqs)
    bits = {}
    for v, view in zip(views, case["views"]):
        key = view["relative_file"]
        if key not in bits:
            bits[key] = load_region_bits(view, cache_root, region_root)
        v["bits"] = bits[key]
    return views


class RoutedBag(CachedBag):
    """``CachedBag`` plus the per-view route masks [n_tokens, C] of every finding."""

    def __init__(self, views: list[dict], device, class_bits: torch.Tensor, min_route: int = 8, gpu_gib: float = 20.):
        super().__init__(views, device, gpu_gib)
        self.masks = []
        for v in views:
            bits = torch.from_numpy(v["bits"]).to(self.device)
            if not bool(((bits >> HIP_BIT) & 1).any()):
                bits = bits & ~(1 << PELVIS_BIT)
            mask = (bits[:, None] & class_bits[None, :]) != 0
            mask[:, mask.sum(0) < min_route] = True
            self.masks.append(mask)

    def routed_chunks(self, block, crops: int, views=None):
        """(block-24 tokens, route mask, position features, view index) per chunk of ``crops`` crops."""
        for k in (range(len(self.views)) if views is None else views):
            o = 0
            for y, pf in self.view_chunks(k, block, crops):
                yield y, self.masks[k][o:o + len(y)], pf, k
                o += len(y)


def make_factory(bag: RoutedBag, block, spec: dict, mean: torch.Tensor, rms: float, crops: int, seed=None):
    """Token factory over a routed bag; ``seed`` given = training (token and feature drop), None = evaluation."""
    drop_t = spec.get("token_drop", 0.) if seed is not None else 0.
    drop_f = spec.get("feature_drop", 0.) if seed is not None else 0.
    pscale = spec.get("position_scale", 1.)

    def factory():
        g = torch.Generator(device=bag.device)
        if seed is not None:
            g.manual_seed(int(seed))
        for y, mk, pf, _ in bag.routed_chunks(block, crops):
            x = (y - mean) / rms
            if drop_t:
                keep = torch.rand(len(x), generator=g, device=bag.device) >= drop_t
                x, mk, pf = x[keep], mk[keep], pf[keep]
                if len(x) == 0:
                    continue
            if drop_f:
                x = x * ((torch.rand(x.shape, generator=g, device=bag.device) >= drop_f).float() / (1 - drop_f))
            yield torch.cat((x, pf.float() * pscale), 1), mk
    return factory


def _to_shared(views):
    """numpy -> torch tensors (moved to shared memory by a multiprocessing queue); uint16 travels as int16."""
    out = []
    for v in views:
        d = {}
        for k, x in v.items():
            if isinstance(x, np.ndarray):
                u16 = x.dtype == np.uint16
                d[k] = ("u16" if u16 else "arr", torch.from_numpy(np.ascontiguousarray(x.view(np.int16) if u16 else x)))
            else:
                d[k] = ("obj", x)
        out.append(d)
    return out


def _from_shared(views):
    return [{k: (t.numpy().view(np.uint16) if kind == "u16" else t.numpy() if kind == "arr" else t)
             for k, (kind, t) in v.items()} for v in views]


def _loader_main(args, inq, outq):
    torch.set_num_threads(2)
    while True:
        case = inq.get()
        if case is None:
            break
        try:
            outq.put(_to_shared(load_study(case, *args)))
        except Exception as e:
            outq.put(RuntimeError(f"{case.get('id')}: {e!r}"))


class Loader:
    """FIFO prefetch of study bags in ``workers`` spawned processes (round-robin keeps the submission order)."""

    def __init__(self, cache_root, region_root, position_freqs=4, depth=4, workers=3):
        import torch.multiprocessing as tmp
        ctx = tmp.get_context("spawn")
        self.depth, self.n, self.sent, self.recv = depth, workers, 0, 0
        self.q = [(ctx.Queue(), ctx.Queue()) for _ in range(workers)]
        args = (Path(cache_root), Path(region_root), position_freqs)
        self.procs = [ctx.Process(target=_loader_main, args=(args, i, o), daemon=True) for i, o in self.q]
        for p in self.procs:
            p.start()

    def submit(self, case):
        self.q[self.sent % self.n][0].put(case)
        self.sent += 1

    def get(self):
        r = self.q[self.recv % self.n][1].get()
        self.recv += 1
        if isinstance(r, Exception):
            raise r
        return _from_shared(r)

    def close(self):
        for i, _ in self.q:
            try:
                i.put(None)
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()

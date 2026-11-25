"""CT pre-training samples (Methods, "Image encoder pre-training", CT).

One sample = one reconstruction (a usable series, or one timepoint of a 4-D series) of a report with sentence statements.
Every reconstruction is visited once per local epoch and weighted 1 / (reconstructions of its report), so reports count
equally. Rendering:

1. resample to 1.5 x 0.5 x 0.5 mm (RAS), crop in-plane to the body box;
2. pick uniformly one of all 64 x 384 x 384 bricks with at least 2 % tissue (HU > -900);
3. global crop 0 (64 x 192 x 192) at a random position inside that brick, in the full window [-1000, 1000] HU;
   global crop 1 overlapping it (``views``), in a random window among full + the series' distinct native DICOM windows;
4. 8 local crops (32 x 96 x 96), each inside one of the globals, each in a random window;
5. lung CTs that have a native lung window (centre below -250 HU): global crop 1 and every local crop are replaced, each
   with probability 0.25, by the same crop of the 10 mm slab MIP shown in the narrowest-centre lung window.
A CT is a lung CT if a series protocol class is a lung-CT class or a lung keyword (English or Turkish) appears in its
classification or DICOM metadata.
"""
from __future__ import annotations

import json

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from fora.geometry.protocols import LUNG_CT_CLASSES, norm_class
from fora.geometry.series import dec, pick_source, sane_affine, valid_frames
from fora.geometry.tiling import AIR_HU, body_frame, resample_series
from fora.geometry.windows import FULL_WINDOW, LUNG_WINDOW_MAX_CENTER, slab_mip, window_normalize, windows_from_meta

from .views import (DEFAULT_GLOBAL_OVERLAP, PATCH, CropSpec, crop_at_start, sample_rng, shifted_start, snap_second_start,
                    start_inside, student_distortion)

BRICK = (64, 384, 384)
TISSUE_HU = -900.0
TISSUE_MIN_FRAC = 0.02
LUNG_KEYWORDS = ("lung", "pulmon", "thorac", "thorax", "chest", "hrct", "toraks", "akcig", "akciğ", "gogus", "göğüs")


def _overlap_starts(L: int, T: int) -> list[int]:
    if L <= T:
        return [0]
    n = -(-L // T)
    return [int(round(i * (L - T) / (n - 1))) for i in range(n)]


def brick_candidates(vol: np.ndarray) -> list[tuple[float, int, int, int]]:
    """(tissue fraction, z0, y0, x0) of every brick of the volume (z tiled without overlap, y / x with overlap)."""
    D, H, W = vol.shape
    tissue = vol > TISSUE_HU
    out = []
    for z0 in [k * BRICK[0] for k in range(max(1, -(-D // BRICK[0])))]:
        for y0 in _overlap_starts(H, BRICK[1]):
            for x0 in _overlap_starts(W, BRICK[2]):
                sub = tissue[z0:z0 + BRICK[0], y0:y0 + BRICK[1], x0:x0 + BRICK[2]]
                out.append((float(sub.mean()) if sub.size else 0.0, z0, y0, x0))
    return out


def is_lung_ct(sg, meta) -> bool:
    cls = [dec(c) for c in sg.attrs.get("classified_protocols", [])]
    if any(norm_class(c) in LUNG_CT_CLASSES for c in cls):
        return True
    blob = " ".join(list(cls) + ([str(v) for v in meta.values()] if meta else [])).lower()
    return any(k in blob for k in LUNG_KEYWORDS)


def read_native_windows(sg) -> tuple[list[tuple[int, int]], dict | None]:
    """Native windows (pacs_metadata first, else dcm2niix_metadata) and the metadata dict they were read from."""
    meta, native = None, []
    for key in ("pacs_metadata", "dcm2niix_metadata"):
        if key not in sg:
            continue
        try:
            meta = json.loads(dec(sg[key][()]))
            native = windows_from_meta(meta)
        except Exception:
            meta, native = None, []
        if native:
            break
    return native, meta


def iter_report_index(h5_path: str, min_slices: int = 16):
    """Reports of one CT shard with non-empty ``there_is_statements`` and at least one usable reconstruction.

    entry = {h5_path, patient, report_uid, series: [(acc, series_id, derivative subpath, frame or None), ...]}"""
    try:
        f = h5py.File(h5_path, "r")
    except Exception:
        return
    with f:
        for patient in f:
            pg = f[patient]
            if not hasattr(pg, "keys"):
                continue
            for report_uid in pg:
                rg = pg[report_uid]
                if not hasattr(rg, "keys") or "report" not in rg or "there_is_statements" not in rg["report"]:
                    continue
                try:
                    sentences = json.loads(dec(rg["report"]["there_is_statements"][()]))
                except Exception:
                    continue
                if not [s for s in sentences if isinstance(s, str) and s.strip()]:
                    continue
                series = []
                for acc in rg:
                    if acc == "report":
                        continue
                    ag = rg[acc]
                    if not hasattr(ag, "keys") or ag.attrs.get("kind") == "report":
                        continue
                    for sid in ag:
                        sg = ag[sid]
                        if not hasattr(sg, "keys"):
                            continue
                        sub, dd = pick_source(sg)
                        if dd is None:
                            continue
                        frames = valid_frames(dd.shape, min_slices)
                        if not frames:
                            continue
                        src = sg[sub] if sub else sg
                        if "affine" not in src or not sane_affine(src["affine"][()]):
                            continue
                        series.extend((acc, sid, sub, fr) for fr in frames)
                if series:
                    yield {"h5_path": h5_path, "patient": patient, "report_uid": report_uid, "series": series}


def count_reconstructions(h5_path: str) -> int:
    return sum(len(e["series"]) for e in iter_report_index(h5_path))


class CTDinoDataset(Dataset):
    def __init__(self, h5_files: list[str], crop_spec: CropSpec = CropSpec(), *, global_overlap=DEFAULT_GLOBAL_OVERLAP,
                 mip_probability: float = 0.25, mip_slab_mm: float = 10.0, seed: int = 3407, patch=PATCH,
                 min_slices: int = 16, verbose: bool = False) -> None:
        if not 0.0 <= float(global_overlap) <= 1.0:
            raise ValueError("global_overlap must be in [0, 1]")
        self.crop_spec = crop_spec
        self.global_overlap = float(global_overlap)
        self.mip_probability = float(mip_probability)
        self.mip_slab_mm = float(mip_slab_mm)
        self.seed = int(seed)
        self.patch = tuple(int(v) for v in patch)
        self.reports = []
        for i, path in enumerate(h5_files):
            self.reports.extend(iter_report_index(path, min_slices=min_slices))
            if verbose and (i % 20 == 0 or i + 1 == len(h5_files)):
                print(f"[data] indexed {i + 1}/{len(h5_files)} shards -> {len(self.reports)} reports", flush=True)
        self.index = [(r, k) for r, entry in enumerate(self.reports) for k in range(len(entry["series"]))]
        if not self.index:
            raise RuntimeError("No usable CT reconstructions were indexed")
        self.n_reports = len(self.reports)

    def __len__(self) -> int:
        return len(self.index)

    def _render(self, entry: dict, k: int, rng) -> dict:
        acc, sid, sub, frame = entry["series"][k]
        with h5py.File(entry["h5_path"], "r") as f:
            sg = f[entry["patient"]][entry["report_uid"]][acc][sid]
            src = sg[sub] if sub else sg
            data = src["data"][:, :, :, frame] if frame is not None else src["data"][()]
            affine = src["affine"][()]
            native, meta = read_native_windows(sg)
            lung_ct = is_lung_ct(sg, meta)
        vol, _ = resample_series(data, affine)
        y0, y1, x0, x1, _, _ = body_frame(vol)
        cvol = vol[:, y0:y1, x0:x1]
        candidates = [c for c in brick_candidates(cvol) if c[0] >= TISSUE_MIN_FRAC] or [(0.0, 0, 0, 0)]
        _, bz, by, bx = rng.choice(candidates)
        gshape, lshape = tuple(self.crop_spec.global_shape), tuple(self.crop_spec.local_shape)
        inner = tuple(rng.randrange(max(1, b - g + 1)) for b, g in zip(BRICK, gshape))
        g0 = (bz + inner[0], by + inner[1], bx + inner[2])
        g1 = snap_second_start(g0, shifted_start(g0, cvol.shape, gshape, self.global_overlap, rng), cvol.shape, gshape,
                               self.patch)
        starts = (g0, g1)
        windows = [FULL_WINDOW] + [w for w in native if w != FULL_WINDOW]
        lung_windows = [w for w in native if (w[0] + w[1]) / 2 < LUNG_WINDOW_MAX_CENTER]
        mip_vol = mip_window = None
        if lung_ct and lung_windows:
            mip_window = min(lung_windows, key=lambda w: (w[0] + w[1]) / 2)
            mip_vol = slab_mip(cvol, slab_mm=self.mip_slab_mm)
        teacher_globals, student_globals = [], []
        for gi in range(2):
            use_mip = gi == 1 and mip_vol is not None and rng.random() < self.mip_probability
            window = mip_window if use_mip else (FULL_WINDOW if gi == 0 else rng.choice(windows))
            teacher = window_normalize(crop_at_start(mip_vol if use_mip else cvol, gshape, starts[gi], AIR_HU), *window)
            teacher_globals.append(teacher)
            student_globals.append(student_distortion(teacher, rng))
        locals_ = []
        for _ in range(self.crop_spec.local_crops):
            parent = rng.randrange(2)
            use_mip = mip_vol is not None and rng.random() < self.mip_probability
            window = mip_window if use_mip else rng.choice(windows)
            start = start_inside(starts[parent], gshape, lshape, cvol.shape, rng)
            local = window_normalize(crop_at_start(mip_vol if use_mip else cvol, lshape, start, AIR_HU), *window)
            locals_.append(student_distortion(local, rng))
        return {"teacher_global": torch.stack(teacher_globals), "student_global": torch.stack(student_globals),
                "student_local": torch.stack(locals_), "reconstruction_weight": 1.0 / len(entry["series"]),
                "global_starts": starts, "mask_seed": rng.randrange(2 ** 63 - 1)}

    def __getitem__(self, key) -> dict:
        idx, epoch = key if isinstance(key, tuple) else (key, 0)
        rng = sample_rng(self.seed, idx, epoch)
        r, k = self.index[idx]
        error = None
        for attempt in range(4):        # a corrupt series must not stop all ranks: retry other reconstructions
            if attempt:
                r, k = self.index[rng.randrange(len(self.index))]
            try:
                return self._render(self.reports[r], k, rng)
            except Exception as exc:
                error = exc
        raise RuntimeError(f"Failed to render a CT sample after 4 attempts: {error}") from error

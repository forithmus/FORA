"""Chest X-ray pre-training samples (Methods, "Chest X-ray encoder"). Same trainer, objective, masks and cross-crop term
as CT; the film is a two-slice volume (2, H, W), i.e. one token layer of the (2, 16, 16) patch.

One sample = one film of a report, weighted 1 / (films of its report). Rendering:
1. content crop (black margins removed), native resolution, percentile [0.5, 99.5] -> [-1, 1] once per film; films
   smaller than a global crop are padded with air (-1);
2. global crop 0 (1,024 x 1,024 px): up to 20 random positions, the first with >= 50 % non-background pixels, else the best;
   global crop 1 overlapping it on the patch lattice; 8 local crops (512 x 512 px) inside a global crop;
3. teacher views keep the full intensity range. Every student view gets a linear window drawn uniformly from
   ``student_windows`` (on the percentile axis 0..1), then a random gamma in [0.7, 1.4] (p 0.5), then the CT student
   distortion.
"""
from __future__ import annotations

import json

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from fora.encoder.cxr import AIR, valid_film_shape
from fora.geometry.xray import content_bbox, native_spacing, series_meta, to_unit

from .views import (DEFAULT_GLOBAL_OVERLAP, CropSpec, crop_at_start, sample_rng, shifted_start, snap_second_start,
                    start_inside, student_distortion)


def parse_windows(spec: str) -> list[tuple[float, float]]:
    """'lo:hi,lo:hi,...' -> [(lo, hi), ...]; '' -> [] (no student windows)."""
    return [tuple(float(v) for v in w.split(":")) for w in spec.split(",") if w.strip()] if spec.strip() else []


def crop_tensor(crop: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(crop))[None].to(torch.bfloat16)


def gamma_jitter(x: torch.Tensor, rng, lo: float = 0.7, hi: float = 1.4) -> torch.Tensor:
    if rng.random() < 0.5:
        return x
    g = rng.uniform(lo, hi)
    return (((x.float() + 1.0) * 0.5).clamp_(0, 1).pow_(g) * 2.0 - 1.0).to(x.dtype)


def student_window(x: torch.Tensor, rng, windows) -> torch.Tensor:
    """y = clip((x01 - lo) / (hi - lo), 0, 1) on the [0, 1] intensity axis, mapped back to [-1, 1]; no-op without windows."""
    if not windows:
        return x
    lo, hi = windows[rng.randrange(len(windows))]
    if lo <= 0.0 and hi >= 1.0:
        return x
    return ((((x.float() + 1.0) * 0.5 - lo) / (hi - lo)).clamp_(0, 1) * 2.0 - 1.0).to(x.dtype)


def resample_2d(img: np.ndarray, native_sp: float, target_sp: float) -> np.ndarray:
    scale = float(native_sp) / float(target_sp)
    H, W = img.shape
    nh, nw = max(8, int(round(H * scale))), max(8, int(round(W * scale)))
    if (nh, nw) == (H, W):
        return img
    t = torch.from_numpy(np.ascontiguousarray(img))[None, None]
    mode = "area" if scale < 1.0 else "bilinear"
    out = F.interpolate(t, size=(nh, nw), mode=mode, **({} if mode == "area" else {"align_corners": False}))
    return out[0, 0].numpy()


def iter_report_index(h5_path: str):
    """Reports of one X-ray shard with a report group and at least one valid film: {h5_path, patient, report_uid, series}."""
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
                if not hasattr(rg, "keys") or "report" not in rg:
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
                        if hasattr(sg, "keys") and "data" in sg and valid_film_shape(sg["data"].shape):
                            series.append((acc, sid))
                if series:
                    yield {"h5_path": h5_path, "patient": patient, "report_uid": report_uid, "series": series}


class CXRDinoDataset(Dataset):
    def __init__(self, h5_files: list[str], crop_spec: CropSpec, *, global_overlap=DEFAULT_GLOBAL_OVERLAP,
                 spacing_mm: float = 0.0, student_windows=(), seed: int = 3407, verbose: bool = False) -> None:
        if crop_spec.global_shape[0] != 2 or crop_spec.local_shape[0] != 2:
            raise ValueError("X-ray crops must be 2 slices deep (one token layer with the (2, 16, 16) patch)")
        self.crop_spec = crop_spec
        self.global_overlap = float(global_overlap)
        self.spacing_mm = float(spacing_mm)
        self.student_windows = [tuple(w) for w in student_windows]
        self.seed = int(seed)
        self.reports = []
        for i, path in enumerate(h5_files):
            self.reports.extend(iter_report_index(path))
            if verbose and (i % 10 == 0 or i + 1 == len(h5_files)):
                print(f"[data] indexed {i + 1}/{len(h5_files)} shards -> {len(self.reports)} reports", flush=True)
        self.index = [(r, k) for r, e in enumerate(self.reports) for k in range(len(e["series"]))]
        if not self.index:
            raise RuntimeError("No usable X-ray films were indexed")
        self.n_reports = len(self.reports)

    def __len__(self) -> int:
        return len(self.index)

    def _load(self, entry: dict, k: int) -> np.ndarray:
        acc, sid = entry["series"][k]
        with h5py.File(entry["h5_path"], "r") as f:
            sg = f[entry["patient"]][entry["report_uid"]][acc][sid]
            data = sg["data"][()]
            native_sp = native_spacing(series_meta(sg))
        img = np.asarray(data[:, :, 0] if data.ndim == 3 else data, dtype=np.float32)
        r0, r1, c0, c1 = content_bbox(img)
        img = img[r0:r1, c0:c1]
        if self.spacing_mm > 0:
            img = resample_2d(img, native_sp, self.spacing_mm)
        u = to_unit(img)
        gh, gw = self.crop_spec.global_shape[1:]
        if u.shape[0] < gh or u.shape[1] < gw:
            pad = np.full((max(gh, u.shape[0]), max(gw, u.shape[1])), AIR, np.float32)
            pad[:u.shape[0], :u.shape[1]] = u
            u = pad
        return np.stack([u, u], 0)

    def _student(self, x: torch.Tensor, rng) -> torch.Tensor:
        return student_distortion(gamma_jitter(student_window(x, rng, self.student_windows), rng), rng)

    def _render(self, entry: dict, k: int, rng) -> dict:
        cvol = self._load(entry, k)
        gshape, lshape = tuple(self.crop_spec.global_shape), tuple(self.crop_spec.local_shape)
        img, best = cvol[0], None
        for _ in range(20):
            cand = (0, rng.randrange(cvol.shape[1] - gshape[1] + 1), rng.randrange(cvol.shape[2] - gshape[2] + 1))
            frac = float((img[cand[1]:cand[1] + gshape[1], cand[2]:cand[2] + gshape[2]] > -0.95).mean())
            if best is None or frac > best[1]:
                best = (cand, frac)
            if frac >= 0.5:
                break
        g0 = best[0]
        g1 = snap_second_start(g0, shifted_start(g0, cvol.shape, gshape, self.global_overlap, rng), cvol.shape, gshape)
        starts = (g0, g1)
        teacher_globals, student_globals = [], []
        for gi in range(2):
            teacher = crop_tensor(crop_at_start(cvol, gshape, starts[gi], AIR))
            teacher_globals.append(teacher)
            student_globals.append(self._student(teacher, rng))
        locals_ = []
        for _ in range(self.crop_spec.local_crops):
            parent = rng.randrange(2)
            start = start_inside(starts[parent], gshape, lshape, cvol.shape, rng)
            locals_.append(self._student(crop_tensor(crop_at_start(cvol, lshape, start, AIR)), rng))
        return {"teacher_global": torch.stack(teacher_globals), "student_global": torch.stack(student_globals),
                "student_local": torch.stack(locals_), "reconstruction_weight": 1.0 / len(entry["series"]),
                "global_starts": starts, "mask_seed": rng.randrange(2 ** 63 - 1)}

    def __getitem__(self, key) -> dict:
        idx, epoch = key if isinstance(key, tuple) else (key, 0)
        rng = sample_rng(self.seed, idx, epoch)
        r, k = self.index[idx]
        error = None
        for attempt in range(4):
            if attempt:
                r, k = self.index[rng.randrange(len(self.index))]
            try:
                return self._render(self.reports[r], k, rng)
            except Exception as exc:
                error = exc
        raise RuntimeError(f"Failed to render an X-ray sample after 4 attempts: {error}") from error

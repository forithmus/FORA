"""Chest X-ray film reading shared by the X-ray encoder, sentence alignment, MIL and inference (Methods, "Chest X-ray").

A film is read at native resolution, cropped to its content box (black margins removed) and stretched once from its
[0.5, 99.5] percentile range to [-1, 1]. No HU, no windows. The X-ray encoder sees a film as a two-slice volume
(2, H, W), i.e. one token layer of the (2, 16, 16) patch.

Two sources:
  * the internal CT-FORA-style H5 shards used for training (`{patient}/{report}/{acc}/{series}/data`, uint16, with
    `dcm2niix_metadata` JSON);
  * the public release (Hugging Face `Forithmus/FORA`): lossless JPEG-LS 16-bit images inside per-batch zip archives,
    addressed by the metadata column `file` = "<zip path>::<member>". Decoding needs `imagecodecs`.
"""
from __future__ import annotations

import json

import numpy as np

PCT_LO, PCT_HI = 0.5, 99.5          # percentile contrast stretch
AIR = -1.0                          # pad value in the [-1, 1] range
DEFAULT_SPACING_MM = 0.15           # fallback pixel spacing (dataset median)


def content_bbox(img: np.ndarray):
    """Tight content box (r0, r1, c0, c1): rows / columns with > 3 % pixels above 0.12 after the percentile stretch."""
    lo, hi = np.percentile(img, (PCT_LO, PCT_HI))
    n = (np.clip(img, lo, hi) - lo) / max(1e-6, hi - lo)
    m = n > 0.12
    H, W = img.shape
    rows = np.where(m.sum(1) > 0.03 * W)[0]
    cols = np.where(m.sum(0) > 0.03 * H)[0]
    if rows.size == 0 or cols.size == 0:
        return 0, H, 0, W
    return int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1


def to_unit(img: np.ndarray) -> np.ndarray:
    """X-ray pixels -> [-1, 1] by the [0.5, 99.5] percentile stretch (bit-depth agnostic)."""
    x = np.asarray(img, dtype=np.float32)
    lo, hi = np.percentile(x, (PCT_LO, PCT_HI))
    if hi <= lo:
        hi = lo + 1.0
    return (np.clip(x, lo, hi) - lo) / (hi - lo) * 2.0 - 1.0


def prepare_film(pixels: np.ndarray) -> np.ndarray:
    """Raw 2-D (or (H, W, 1)) film -> content crop -> [-1, 1] image (H', W')."""
    img = np.asarray(pixels[:, :, 0] if pixels.ndim == 3 else pixels, dtype=np.float32)
    r0, r1, c0, c1 = content_bbox(img)
    return to_unit(img[r0:r1, c0:c1])


def native_spacing(meta: dict) -> float:
    """Mean pixel spacing (mm / px) from the film metadata (`pixel_spacing_mm`), else DEFAULT_SPACING_MM."""
    v = meta.get("pixel_spacing_mm")
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        try:
            a, b = float(v[0]), float(v[1])
            if a > 0 and b > 0:
                return (a + b) / 2.0
        except Exception:
            pass
    return DEFAULT_SPACING_MM


def series_meta(series_group) -> dict:
    """Parsed `dcm2niix_metadata` JSON of an X-ray H5 series group ({} if absent or unreadable)."""
    d = series_group.get("dcm2niix_metadata")
    if d is None:
        return {}
    try:
        v = d[()]
        return json.loads(v.decode() if isinstance(v, (bytes, bytearray)) else v)
    except Exception:
        return {}


def read_h5_film(h5_path: str, patient: str, report: str, acc: str, sid: str) -> np.ndarray:
    """One film from an X-ray H5 shard -> prepared [-1, 1] image."""
    import h5py
    with h5py.File(h5_path, "r") as f:
        data = f[patient][report][acc][sid]["data"][()]
    return prepare_film(data)


def read_release_image(root: str, file_field: str) -> np.ndarray:
    """Decode one image of the public release: metadata `file` = '<zip path>::<member>' -> uint16 (rows, columns)."""
    import zipfile

    import imagecodecs
    archive, member = file_field.split("::")
    with zipfile.ZipFile(f"{root}/{archive}") as z:
        return imagecodecs.jpegls_decode(z.read(member))


def read_release_film(root: str, file_field: str) -> np.ndarray:
    """One film of the public release -> prepared [-1, 1] image (same preparation as the training films)."""
    return prepare_film(read_release_image(root, file_field))

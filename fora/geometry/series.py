"""Selection and reading of usable CT series in the CT-FORA H5 shards (Methods, "CT preprocessing").

A series is usable if
  * a geometry-corrected derivative is preferred when present: Tilt_Eq_1 > Tilt_1 > Eq_1 > the raw series (ROI*
    derivatives are ignored);
  * it has at least 16 slices and at least 16 px in-plane; a 4-D (X, Y, Z, T) series gives one volume per timepoint;
  * its affine is finite, every axis spacing is in (0, 30] mm and the orientation is resolvable.
"""
from __future__ import annotations

import numpy as np

DERIV_PRIORITY = ("Tilt_Eq_1", "Tilt_1", "Eq_1")
MIN_SLICES = 16
MIN_INPLANE = 16
MAX_SANE_SPACING_MM = 30.0   # larger spacing means a corrupt affine (runaway resampling); coarse 10-15 mm z is kept


def dec(x):
    return x.decode() if isinstance(x, (bytes, bytearray)) else x


def attr_list(grp, name: str) -> list[str]:
    v = grp.attrs.get(name, [])
    if isinstance(v, (bytes, str)):
        v = [v]
    return [dec(t) for t in v]


def pick_source(series_grp):
    """(subpath, data dataset) of the volume to use: a geometry-corrected derivative if present, else the raw series.

    subpath is "" or "derivative_<tag>"; (None, None) if the group holds no volume.
    """
    tags = attr_list(series_grp, "derivatives")
    for t in DERIV_PRIORITY:
        if t in tags and f"derivative_{t}" in series_grp:
            d = series_grp[f"derivative_{t}"]
            if "data" in d and "affine" in d:
                return f"derivative_{t}", d["data"]
    if "data" in series_grp and "affine" in series_grp:
        return "", series_grp["data"]
    return None, None


def sane_affine(affine) -> bool:
    """True if the affine is finite, has per-axis spacing in (0, MAX_SANE_SPACING_MM] and a resolvable orientation."""
    aff = np.asarray(affine, dtype=np.float64)
    if aff.shape != (4, 4) or not np.all(np.isfinite(aff)):
        return False
    sp = np.linalg.norm(aff[:3, :3], axis=0)
    if not (np.all(sp > 0) and np.all(sp <= MAX_SANE_SPACING_MM)):
        return False
    try:
        from nibabel.orientations import io_orientation
        return bool(np.all(np.isfinite(io_orientation(aff))))
    except Exception:
        return False


def valid_frames(shape, min_slices: int = MIN_SLICES) -> list:
    """Frame indices that each give a usable 3-D volume: [None] for 3-D, range(T) for 4-D, [] if unusable."""
    shp = tuple(shape)
    if len(shp) not in (3, 4):
        return []
    if shp[0] < MIN_INPLANE or shp[1] < MIN_INPLANE or shp[2] < min_slices:
        return []
    return [None] if len(shp) == 3 else list(range(shp[3]))


def read_series(h5file, patient: str, report: str, acc: str, sid: str, sub: str = "", frame=None):
    """Read one planned series volume (X, Y, Z) and its affine from an open CT-FORA H5 shard."""
    sg = h5file[patient][report][acc][sid]
    src = sg[sub] if sub else sg
    data = src['data'][:, :, :, frame] if frame is not None else src['data'][()]
    return data, src['affine'][()]

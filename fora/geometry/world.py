"""World (patient mm) coordinates of cached tokens, from the provenance stored in each cache series file.

The cache stores per series `provenance_json` with the source affine, the source array shape (`orig_shape`), the
in-plane body crop (`body_crop_yxyx`) and the target spacing. Token (z, y, x) of the token grid then has the world
centre (S, R, A order, mm)

    world = origin + ((zyx + 0.5) * PATCH - 0.5 + crop_offset) * step

with origin / step from the RAS-reoriented affine (as `fora.geometry.tiling.resample_series`). The evidence maps and
foci (fora.mil.evidence) pool tokens of all views of a study on a token-sized grid in this frame.
"""
from __future__ import annotations

import numpy as np

from .tiling import TARGET_SPACING
from .tokens import PATCH


def token_geometry(prov: dict):
    """-> (origin[3], step[3], crop_offset[3], token_mm[3], native_spacing (z, mean in-plane)) in (S, R, A) order."""
    from nibabel.orientations import axcodes2ornt, inv_ornt_aff, io_orientation, ornt_transform
    aff = np.asarray(prov['affine'], float)
    shp = tuple(prov['orig_shape'][:3])
    nat = np.linalg.norm(aff[:3, :3], axis=0)
    aff = aff @ inv_ornt_aff(ornt_transform(io_orientation(aff), axcodes2ornt(('R', 'A', 'S'))), shp)
    sg = lambda v: 1. if v >= 0 else -1.
    ts = np.asarray(prov.get('target_spacing', list(TARGET_SPACING)), float)
    origin = np.array([aff[2, 3], aff[0, 3], aff[1, 3]])
    step = ts * np.array([sg(aff[2, 2]), sg(aff[0, 0]), sg(aff[1, 1])])
    y0, y1, x0, x1 = prov['body_crop_yxyx']
    p = np.array(PATCH, float)
    return origin, step, np.array([0, y0, x0], float), p * ts, (float(nat[2]), float(nat[:2].mean()))


def token_centres_mm(prov: dict, zyx: np.ndarray) -> np.ndarray:
    """World centres (S, R, A mm) of tokens with grid indices zyx [n, 3]."""
    origin, step, offset, _, _ = token_geometry(prov)
    return origin[None] + ((np.asarray(zyx) + .5) * np.array(PATCH, float) - .5 + offset[None]) * step[None]


def world_cells(world_mm: np.ndarray, token_mm: np.ndarray) -> np.ndarray:
    """Snap world positions to a token-sized world grid -> integer cell indices [n, 3]."""
    return np.floor(np.asarray(world_mm) / np.asarray(token_mm)[None]).astype(np.int64)

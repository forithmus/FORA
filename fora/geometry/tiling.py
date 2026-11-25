"""CT volume geometry: RAS reorientation, resampling to 1.5 x 0.5 x 0.5 mm and the body crop.

Implements the resampling and body-crop step of the CT preprocessing (Methods). Every series is reoriented to RAS,
transposed to (S, R, A) = (slice, left-right, anterior-posterior), resampled trilinearly to TARGET_SPACING and cropped
in-plane to the bounding box of voxels above -300 HU. The z extent is not cropped.

Axis convention: the stored H5 / NIfTI array is (X, Y, Z) with a 4 x 4 affine. After `resample_series` the volume is
(D, H, W) = (S, R, A) and `geom[axis] = (origin_mm, step_mm)` gives the world coordinate of voxel index i as
origin + i * step.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

TARGET_SPACING = (1.5, 0.5, 0.5)       # (z, y, x) mm
BODY_HU_THRESH = -300.0                # solid-tissue threshold of the in-plane body box
AIR_HU = -1000.0                       # pad value for crops that extend past the volume


def resize_array(array: torch.Tensor, current_spacing, target_spacing) -> np.ndarray:
    """Trilinear resample of a (B, C, D, H, W) tensor from current to target spacing."""
    orig = array.shape[2:]
    scale = [current_spacing[i] / target_spacing[i] for i in range(len(orig))]
    new_shape = [max(1, int(orig[i] * scale[i])) for i in range(len(orig))]
    return F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False).cpu().numpy()


def to_ras(data: np.ndarray, affine: np.ndarray):
    """Reorient a volume and its (4, 4) affine to RAS+ (axis 0 -> Right, 1 -> Anterior, 2 -> Superior).

    Pure axis permutation and flips (nibabel orientation utilities), no interpolation.
    """
    from nibabel.orientations import apply_orientation, axcodes2ornt, inv_ornt_aff, io_orientation, ornt_transform
    t = ornt_transform(io_orientation(affine), axcodes2ornt(("R", "A", "S")))
    new_affine = affine @ inv_ornt_aff(t, data.shape)
    return apply_orientation(data, t), new_affine


def resample_series(data, affine):
    """(X, Y, Z) HU array + (4, 4) affine -> RAS-reoriented, resampled (D, H, W) float32 volume and its geometry.

    Spacing is taken from the affine column norms, so any acquisition orientation works, and each axis is resampled by
    its own ratio. Integer input is reoriented losslessly and converted to float32 once. Floating input is copied and
    NaN / Inf are set to 0.
    """
    raw = np.asarray(data)
    if np.issubdtype(raw.dtype, np.floating):
        img = raw.astype(np.float32, copy=True)
        np.nan_to_num(img, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        img = raw
    affine = np.asarray(affine, dtype=np.float64)
    img, affine = to_ras(img, affine)
    sp = [float(s) if s > 0 else 1.0 for s in np.linalg.norm(affine[:3, :3], axis=0)]   # (R, A, S)
    img = np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32)                 # (R, A, S) -> (S, R, A)
    cur = (sp[2], sp[0], sp[1])
    t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    vol = resize_array(t, cur, TARGET_SPACING)[0, 0]
    tz, txy, _ = TARGET_SPACING
    sgn = lambda v: 1.0 if v >= 0 else -1.0
    geom = {
        0: (float(affine[2, 3]), tz * sgn(affine[2, 2])),
        1: (float(affine[0, 3]), txy * sgn(affine[0, 0])),
        2: (float(affine[1, 3]), txy * sgn(affine[1, 1])),
    }
    return vol, geom


def body_frame(vol: np.ndarray):
    """In-plane body bounding box and z extent (resampled voxel indices): (y0, y1, x0, x1, z0, z1).

    Body = voxels above BODY_HU_THRESH on a 2x in-plane subsampled grid. Callers crop in-plane only.
    """
    D, H, W = vol.shape
    sx = 2
    small = vol[:, ::sx, ::sx]
    body = small > BODY_HU_THRESH
    ys, xs = np.where(body.any(axis=0))
    if ys.size:
        y0, y1 = int(ys.min()) * sx, min(int(ys.max() + 1) * sx, H)
        x0, x1 = int(xs.min()) * sx, min(int(xs.max() + 1) * sx, W)
    else:
        y0, y1, x0, x1 = 0, H, 0, W
    zz = np.where(body.any(axis=(1, 2)))[0]
    z0, z1 = (int(zz.min()), int(zz.max()) + 1) if zz.size else (0, D)
    return y0, y1, x0, x1, z0, z1


def crop_to_body(vol: np.ndarray):
    """In-plane body crop of a resampled volume -> (contiguous float32 volume, [y0, y1, x0, x1])."""
    y0, y1, x0, x1, _, _ = body_frame(vol)
    return np.ascontiguousarray(vol[:, y0:y1, x0:x1], dtype=np.float32), [int(y0), int(y1), int(x0), int(x1)]


def world(geom, axis: int, idx):
    """World coordinate (mm) of a resampled voxel index along one axis."""
    o, s = geom[axis]
    return o + idx * s

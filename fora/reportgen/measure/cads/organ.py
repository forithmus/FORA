"""Organ geometry from CADS label volumes (liver, spleen, kidneys): largest 3-D component, craniocaudal span, 3-D max extent,
in-plane Feret diameter (paper: Methods, "Measurements")."""
import numpy as np
from scipy import ndimage as ndi


def largest_cc(m):
    lab, n = ndi.label(m)
    if n <= 1: return m
    s = ndi.sum_labels(m, lab, np.arange(1, n + 1)); return lab == (1 + int(np.argmax(s)))


def feret2d(pts):
    if len(pts) < 3: return 0.
    from scipy.spatial import ConvexHull
    try: h = pts[ConvexHull(pts).vertices]
    except Exception: h = pts
    d = h[:, None, :] - h[None, :, :]; return float(np.sqrt((d ** 2).sum(-1)).max())


def orient(A, shape):
    """index axes -> which is z (largest |S| component), spacing per axis, sign of S/A along axes."""
    M = A[:3, :3]; sp = np.linalg.norm(M, axis=0)
    zax = int(np.argmax(np.abs(M[2]) / sp)); return zax, sp


def organ_feats(m, A, zax, sp, nz):
    m = largest_cc(m); idx = np.nonzero(m)
    if len(idx[0]) == 0: return None
    z = idx[zax]; f = dict(n_vox=int(len(z)), vol_ml=float(len(z) * np.prod(sp) / 1000))
    f['cc_mm'] = float((z.max() - z.min() + 1) * sp[zax]); f['z_touch'] = int(z.min() == 0 or z.max() == nz - 1)
    ax = [a for a in range(3) if a != zax]
    # max in-plane Feret over axial slices (subsample slices), world in-plane extents of the whole organ
    best = 0.; zs = np.unique(z)
    for k in zs[::max(1, len(zs) // 40)]:
        sel = z == k; p = np.stack([idx[ax[0]][sel] * sp[ax[0]], idx[ax[1]][sel] * sp[ax[1]]], 1).astype(float)
        if len(p) > 4000: p = p[np.random.default_rng(0).choice(len(p), 4000, replace=False)]
        best = max(best, feret2d(p))
    f['inplane_feret_mm'] = best
    P = np.stack(idx, 1)[::max(1, len(z) // 20000)].astype(float) * sp
    from scipy.spatial import ConvexHull
    try: h = P[ConvexHull(P).vertices]; d = h[:, None] - h[None]; f['max3d_mm'] = float(np.sqrt((d ** 2).sum(-1)).max())
    except Exception: f['max3d_mm'] = float('nan')
    W = (A[:3, :3] @ np.stack(idx, 0)[:, ::max(1, len(z) // 20000)]).T
    f['ext_R_mm'], f['ext_A_mm'] = float(np.ptp(W[:, 0])), float(np.ptp(W[:, 1]))
    return f, m

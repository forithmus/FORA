"""Pure-geometry RECIST 1.1 caliper on a binary lesion mask (no learning; paper: Methods, "Measurements").
Used identically for predicted and reference masks.

Conventions
- A voxel is a box of its spacing; the in-plane lesion shape of one axial slice is the UNION OF PIXEL SQUARES. Its convex hull is the hull
  of the pixel corners (only boundary pixels needed).
- Per slice only the largest 8-connected 2-D piece is measured (a 3-D-connected lesion can split into pieces on one slice).
- LONG axis (RECIST 1.1): the longest in-plane diameter on the AXIAL slice where it is maximal = max Feret diameter of that slice
  (exact: max distance between hull vertices; equivalent to rotating calipers). Ties -> slice with the larger area.
- SHORT axis: longest diameter PERPENDICULAR to the long axis in the same slice = the longest chord of the hull along the normal
  direction. The chord length is concave piecewise linear in the offset along the long axis -> its maximum is attained at a hull vertex
  offset (exact; all vertex offsets evaluated).
- 3-D maximal extent: diameter of the convex hull of the voxel corners (boundary voxels). Volume = voxel count x voxel volume.
- RECIST flags: non-nodal lesion measurable iff long >= max(10 mm, 2 x slice thickness); lymph node: short >= 15 mm pathological
  (measurable), 10-15 mm pathological non-measurable, < 10 mm non-pathological.
Coordinates: arrays are indexed (z, a1, a2); endpoints are returned as continuous voxel indices (pixel centres at integers)."""
import math
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull

_STRUCT26 = np.ones((3, 3, 3), bool)


def _hull_pts(pts):
    if len(pts) < 3: return pts
    try: return pts[ConvexHull(pts).vertices]
    except Exception: return pts   # degenerate (collinear) -> all points


def slice_caliper(m2, sp):
    """m2 bool [n1, n2]; sp (d1, d2) mm. -> dict(long, short, long_pts, short_pts) with points in continuous pixel index (i1, i2)."""
    if not m2.any(): return None
    lab, n = ndi.label(m2, structure=np.ones((3, 3), bool))   # RECIST measures ONE in-plane lesion: the largest 8-connected 2-D piece
    if n > 1: m2 = lab == (1 + int(np.argmax(np.bincount(lab[lab > 0])[1:])))
    b = m2 & ~ndi.binary_erosion(m2)            # boundary pixels carry all hull corners
    p = np.argwhere(b).astype(np.float64)
    c = (p[:, None, :] + np.array([[-.5, -.5], [-.5, .5], [.5, -.5], [.5, .5]])[None]).reshape(-1, 2)
    c = np.unique(c, axis=0); s = np.asarray(sp, float); cm = c * s
    hv = _hull_pts(cm)
    d2 = ((hv[:, None] - hv[None]) ** 2).sum(-1); i, j = np.unravel_index(np.argmax(d2), d2.shape)
    L = math.sqrt(d2[i, j]); A, B = hv[i], hv[j]
    u = (B - A) / max(L, 1e-12); n = np.array([-u[1], u[0]])
    # hull polygon in order (ConvexHull vertices are ordered counter-clockwise in 2-D)
    poly = hv; tv = poly @ u; best = (0., A, A)
    E = np.stack([poly, np.roll(poly, -1, 0)], 1)   # edges
    ta, tb = E[:, 0] @ u, E[:, 1] @ u
    for t in np.unique(tv):
        lo_ok = ((ta - t) * (tb - t) <= 0) & (np.abs(ta - tb) > 1e-12)
        if not lo_ok.any(): continue
        a, bb = E[lo_ok, 0], E[lo_ok, 1]; w = ((t - ta[lo_ok]) / (tb[lo_ok] - ta[lo_ok]))[:, None]; q = a + w * (bb - a)
        on = np.abs(poly @ u - t) < 1e-9
        if on.any(): q = np.concatenate([q, poly[on]])
        pr = q @ n; k0, k1 = int(np.argmin(pr)), int(np.argmax(pr)); ch = pr[k1] - pr[k0]
        if ch > best[0]: best = (ch, q[k0], q[k1])
    return dict(long=L, short=float(best[0]), long_pts=[A / s, B / s], short_pts=[best[1] / s, best[2] / s], area=float(m2.sum() * s[0] * s[1]))


def caliper3d(mask, sp, is_node=False, slice_thickness=None, fast=False):
    """mask bool [Z, n1, n2] (one lesion); sp (dz, d1, d2) mm. -> RECIST dict or None."""
    if not mask.any(): return None
    sp = np.asarray(sp, float); best = None
    for z in np.flatnonzero(mask.any((1, 2))):
        r = slice_caliper(mask[z], sp[1:])
        if r is None: continue
        if best is None or r['long'] > best['long'] + 1e-9 or (abs(r['long'] - best['long']) <= 1e-9 and r['area'] > best['area']): best = dict(r, z=int(z))
    z = best['z']
    if fast: return dict(long_mm=best['long'], short_mm=best['short'], slice=z, long_vox=[[z, *best['long_pts'][0].tolist()], [z, *best['long_pts'][1].tolist()]],
                         short_vox=[[z, *best['short_pts'][0].tolist()], [z, *best['short_pts'][1].tolist()]])
    vb = mask & ~ndi.binary_erosion(mask); p = np.argwhere(vb).astype(np.float64)
    if len(p) > 4000: p = p[np.random.default_rng(0).choice(len(p), 4000, replace=False)]
    cor = (p[:, None, :] + (np.stack(np.meshgrid(*[[-.5, .5]] * 3, indexing='ij'), -1).reshape(-1, 3))[None]).reshape(-1, 3) * sp
    hv = _hull_pts(np.unique(cor, axis=0)); d2 = ((hv[:, None] - hv[None]) ** 2).sum(-1) if len(hv) < 3000 else None
    ext3 = math.sqrt(d2.max()) if d2 is not None else float(np.linalg.norm(cor.max(0) - cor.min(0)))
    st = float(slice_thickness if slice_thickness is not None else sp[0])
    out = dict(long_mm=best['long'], short_mm=best['short'], slice=z,
               long_vox=[[z, *best['long_pts'][0].tolist()], [z, *best['long_pts'][1].tolist()]],
               short_vox=[[z, *best['short_pts'][0].tolist()], [z, *best['short_pts'][1].tolist()]],
               area_mm2=best['area'], extent3d_mm=ext3, volume_mm3=float(mask.sum() * np.prod(sp)),
               z_extent_mm=float((np.ptp(np.flatnonzero(mask.any((1, 2)))) + 1) * sp[0]))
    if is_node:
        s_ = out['short_mm']; out['recist'] = 'node_pathological_measurable' if s_ >= 15 else 'node_pathological_nonmeasurable' if s_ >= 10 else 'node_nonpathological'
        out['reportable_mm'] = s_
    else:
        out['recist'] = 'measurable' if out['long_mm'] >= max(10., 2 * st) else 'non_measurable'
        out['reportable_mm'] = out['long_mm']
    return out


def pick_component(prob, centre_idx, sp, thr=.5, sigma_mm=None):
    """prob float [Z, n1, n2]; keep the 26-connected component of prob > thr with the largest prob mass weighted by proximity to the
    centre (Gaussian of the distance from the centre to the component's nearest voxel, sigma = sigma_mm). -> bool mask (maybe empty)."""
    hard = prob > thr
    if not hard.any(): return hard
    lab, n = ndi.label(hard, structure=_STRUCT26)
    if n == 1: return hard
    sp = np.asarray(sp, float); c = np.asarray(centre_idx, float)
    sig = sigma_mm if sigma_mm is not None else 0.25 * float(np.min(np.array(prob.shape) * sp))
    idx = np.argwhere(hard); d = np.sqrt((((idx - c[None]) * sp[None]) ** 2).sum(1)); l_ = lab[hard]
    dmin = np.full(n + 1, np.inf); np.minimum.at(dmin, l_, d)
    mass = np.bincount(l_, weights=prob[hard], minlength=n + 1)
    score = mass * np.exp(-0.5 * (dmin / sig) ** 2); score[0] = -1
    return lab == int(np.argmax(score))

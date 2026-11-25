"""Evidence regions on the 3 x 8 x 8 mm world-cell grid for any model's CT map (paper Methods 4.10).

Input is a non-negative evidence value per point (FORA: positive part of attention x token logit x output scale
per token; another model: its map reduced to cells by the maximum per cell, see ``dense_map_to_cells``) with the
world position of each point in (S, R, A) mm. The geometry is the one of FORA's evidence maps:

1. cell = floor(world / cell size); a cell takes the maximum evidence of its points (the representative point);
2. 4 mm Gaussian smoothing on the cell grid (zero outside);
3. allowed cells = all occupied cells, or the cells of the finding's anatomical family when given and non-empty;
4. HDR50 / HDR80 = the smallest sets of allowed cells holding 50% / 80% of the smoothed mass;
5. foci = 26-connected components of HDR50 ranked by smoothed mass; the top focus centre is the smoothed-value
   weighted centroid of the representative points of its cells.

``regions`` returns ``None`` when the map has no positive value (no focus: every measure scores 0).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import gaussian_filter, label

CELL = np.array([3., 8., 8.])   # (S, R, A) mm
SIGMA_MM = 4.0


def regions(e: np.ndarray, world: np.ndarray, tok: np.ndarray = CELL, in_family: np.ndarray | None = None) -> dict | None:
    """Cells, smoothed values, HDR50 / HDR80 and the first-focus centre of one evidence map.

    e: (n,) evidence >= 0; world: (n, 3) S, R, A mm; tok: cell size (mm); in_family: optional (n,) bool.
    Returns dict(cells (m,3) int, score (m,), allowed, core = HDR50, hdr80, focus = (S, R, A) mm, restricted).
    """
    e = np.asarray(e, np.float64); W = np.asarray(world, np.float64); tok = np.asarray(tok, np.float64)
    if e.max() <= 0:
        return None
    cell = np.floor(W / tok[None]).astype(np.int64)
    o_ = np.lexsort((-e, cell[:, 2], cell[:, 1], cell[:, 0])); cs = cell[o_]
    newc = np.r_[True, np.any(cs[1:] != cs[:-1], 1)]; starts = np.flatnonzero(newc); rep = o_[starts]
    uc = cs[starts]; lo = uc.min(0); shp = tuple(uc.max(0) - lo + 1); I = tuple((uc - lo).T)
    G = np.zeros(shp, np.float32); G[I] = e[rep]; HT = np.zeros(shp, bool); HT[I] = True
    sm = gaussian_filter(G, sigma=SIGMA_MM / tok, mode='constant')
    nzv = sm[sm > 0]
    if not len(nzv) or float(np.quantile(nzv, .995)) <= 0:
        return None
    allow = HT.copy(); restricted = False
    if in_family is not None:
        fam = np.logical_or.reduceat(np.asarray(in_family, bool)[o_], starts)
        F = np.zeros(shp, bool); F[I] = fam; a2 = HT & F
        if a2.any() and sm[a2].sum() > 0:
            allow = a2; restricted = True
    vals_ = sm[allow]; o2 = np.argsort(-vals_); cu = np.cumsum(vals_[o2]) / max(vals_.sum(), 1e-30)
    kcut = int(np.searchsorted(cu, .5)) + 1; thr_ = vals_[o2[min(kcut, len(o2)) - 1]]; msk = allow & (sm >= thr_)
    k80 = int(np.searchsorted(cu, .8)) + 1; thr80 = vals_[o2[min(k80, len(o2)) - 1]]; m80 = (allow & (sm >= thr80))[I]
    if not msk.any():
        msk = np.zeros(shp, bool); msk[np.unravel_index(int(np.argmax(sm * HT)), shp)] = True
    lab, nl = label(msk, structure=np.ones((3, 3, 3), bool)); cid = lab[I] - 1; val = sm[I]
    cm = np.bincount(cid[cid >= 0], weights=val[cid >= 0], minlength=nl); order = np.argsort(-cm)
    mc = cid == order[0]; ti = rep[mc]; vv = val[mc]; oo = np.argsort(-vv, kind='stable'); ti, vv = ti[oo], vv[oo]
    focus = (vv[:, None] * W[ti]).sum(0) / vv.sum()
    return dict(cells=uc.astype(np.int32), score=val.astype(np.float32), allowed=allow[I], core=msk[I], hdr80=m80, focus=focus,
                restricted=restricted)


def dense_map_to_cells(values: np.ndarray, affine: np.ndarray, keep: np.ndarray | None = None):
    """Reduce a dense voxel map (any model) to world cells by the maximum per cell.

    values: (X, Y, Z) map on the image grid with RAS ``affine``; keep: optional voxel mask (e.g. body).
    Returns (cells (m, 3) S,R,A ints, value per cell). Cells are then passed to ``regions`` with their centres
    ``(cells + 0.5) * CELL`` as world positions.
    """
    shape = values.shape
    corners = np.array([[i, j, k] for i in (0, shape[0] - 1) for j in (0, shape[1] - 1) for k in (0, shape[2] - 1)], float)
    cc = np.floor((corners @ affine[:3, :3].T + affine[:3, 3])[:, [2, 0, 1]] / CELL).astype(np.int64)
    lo = cc.min(0); dims = cc.max(0) - lo + 1; N = int(np.prod(dims))
    best = np.full(N, -np.inf)
    jj, kk = np.meshgrid(np.arange(shape[1]), np.arange(shape[2]), indexing='ij'); jj = jj.ravel(); kk = kk.ravel()
    for i in range(shape[0]):
        xyz = np.stack([np.full(len(jj), i), jj, kk], 1) @ affine[:3, :3].T + affine[:3, 3]
        idx = np.ravel_multi_index((np.floor(xyz[:, [2, 0, 1]] / CELL).astype(np.int64) - lo).T, dims)
        v = values[i].ravel().astype(np.float64)
        if keep is not None:
            sel = keep[i].ravel(); idx, v = idx[sel], v[sel]
        np.maximum.at(best, idx, v)
    nz = np.flatnonzero(np.isfinite(best))
    return np.stack(np.unravel_index(nz, dims), 1) + lo, best[nz]


def lesion_cells(mask: np.ndarray, affine: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Per cell: does any annotated voxel centre fall in it."""
    vox = np.argwhere(mask); tc = set()
    for first in range(0, len(vox), 200000):
        xyz = vox[first:first + 200000] @ affine[:3, :3].T + affine[:3, 3]
        for row in np.unique(np.floor(xyz[:, [2, 0, 1]] / CELL).astype(np.int32), axis=0):
            tc.add(tuple(row))
    return np.array([tuple(c) in tc for c in np.asarray(cells).tolist()], bool)


def top_n(score: np.ndarray, allow: np.ndarray, n: int):
    """Top-n allowed cells by smoothed value (ties broken by cell order); returns mask and tie information."""
    idx = np.flatnonzero(allow); n = int(min(max(n, 0), len(idx)))
    m = np.zeros(len(score), bool)
    if n == 0:
        return m, dict(n=0, boundary_tie=False, tied_in=0, tied_out=0)
    o = idx[np.lexsort((idx, -score[idx]))]; m[o[:n]] = True
    v = score[o[n - 1]]; tin = int((score[o[:n]] == v).sum()); tout = int((score[o[n:]] == v).sum())
    return m, dict(n=n, boundary_tie=tout > 0, tied_in=tin, tied_out=tout, boundary_value=float(v))


def tie_expected_hit(score, allow, sel, tie: dict, lesion) -> float:
    """P(selection intersects the lesion) when the cells tied at the cut-off are chosen uniformly (hypergeometric)."""
    if tie['n'] == 0:
        return 0.
    v = tie['boundary_value']; strict = sel & (score > v)
    if (strict & lesion).any():
        return 1.
    tied = allow & (score == v); T = int(tied.sum()); g = int((tied & lesion).sum()); k = tie['tied_in']
    if g == 0:
        return 0.
    if T - g < k:
        return 1.
    return 1. - math.exp(math.lgamma(T - g + 1) - math.lgamma(T - g - k + 1) - math.lgamma(T + 1) + math.lgamma(T - k + 1))


def equal_size_overlap(reg: dict, mask: np.ndarray, affine: np.ndarray, n_cells: int) -> dict:
    """Evidence overlap of a model's map at a fixed size: the ``n_cells`` highest-valued allowed cells (e.g. as many
    as FORA's HDR80 for the same instance), both deterministic and with tied cells counted at their expected value."""
    if reg is None:
        return dict(hit=False, expected_hit=0., n=0)
    sc = reg['score'].astype(np.float64); les = lesion_cells(mask, affine, reg['cells'])
    sel, tie = top_n(sc, reg['allowed'], n_cells)
    return dict(hit=bool((sel & les).any()), expected_hit=tie_expected_hit(sc, reg['allowed'], sel, tie, les), n=tie['n'],
                boundary_tie=tie['boundary_tie'])

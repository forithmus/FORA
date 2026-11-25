"""CT localization measures (paper Methods 4.10) for any model whose evidence is given as world cells.

Per annotated instance (``instance_metrics``):
    first_focus_center_inside_exact_mask   top-focus hit: the focus centre's nearest voxel lies in the annotation
    HDR50_intersects_GT_cells / HDR80_...  evidence overlap: an annotated voxel centre falls in a 50% / 80% cell
    HDR50_GT_cell_coverage / HDR80_...     share of annotated cells inside the region
    coarse_world_cell_AUPRC                AP of the smoothed cell values against annotated cells
A map without positive evidence has no focus and scores 0 on every measure.

Chance (``uniform_route_chance``): probability that a uniformly drawn token centre inside the finding's route
(the CADS regions of the finding; all tokens when the route has fewer than 8 tokens) lies inside the annotation,
and the AP of the binary route indicator against annotation-intersecting tokens.

Cohort summary (``aggregate``): hit rate with an exact binomial (Clopper-Pearson) 95% interval and means of the
other measures over annotated instances.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score

from evaluation.localization.ct_regions import CELL
from evaluation.statistics.multiple_testing import clopper_pearson

MIN_ROUTE_TOKENS = 8
FIELDS = ['HDR50_intersects_GT_cells', 'HDR80_intersects_GT_cells', 'HDR50_GT_cell_coverage', 'HDR80_GT_cell_coverage', 'coarse_world_cell_AUPRC']


def instance_metrics(mask: np.ndarray, affine: np.ndarray, focus, cells, score, core, hdr80) -> dict:
    """mask: annotation on the image grid (RAS ``affine``); focus: (S, R, A) mm or None; cells (m, 3) S,R,A ints."""
    vox = np.argwhere(mask); n = len(vox)
    result = dict(GT_positive=bool(n), GT_voxels=n, GT_volume_mm3=n * abs(float(np.linalg.det(affine[:3, :3]))))
    if not n:
        return result
    center = np.array([focus[1], focus[2], focus[0]]) if focus is not None else None
    if center is not None:
        q = np.linalg.inv(affine) @ np.r_[center, 1.]; nearest = np.rint(q[:3]).astype(int)
        inside = bool(np.all(nearest >= 0) and np.all(nearest < mask.shape) and mask[tuple(nearest)])
    else:
        inside = False
    truth = set(); d2 = float('inf')
    for first in range(0, n, 200000):
        xyz = vox[first:first + 200000] @ affine[:3, :3].T + affine[:3, 3]
        if center is not None:
            d2 = min(d2, float(np.square(xyz - center).sum(1).min()))
        for row in np.unique(np.floor(xyz[:, [2, 0, 1]] / CELL).astype(np.int32), axis=0):
            truth.add(tuple(row))
    cells = np.asarray(cells); core = np.asarray(core, bool); hdr80 = np.asarray(hdr80, bool)
    overlap = np.array([tuple(c) in truth for c in cells], bool)
    p50 = {tuple(c) for c in cells[core]}; p80 = {tuple(c) for c in cells[hdr80]}
    result.update(first_focus_center_inside_exact_mask=inside, nearest_GT_voxel_center_mm=float(np.sqrt(d2)) if center is not None else None,
                  HDR50_intersects_GT_cells=bool(p50 & truth), HDR80_intersects_GT_cells=bool(p80 & truth),
                  GT_occupied_world_cells=len(truth), HDR50_GT_cell_coverage=len(p50 & truth) / len(truth),
                  HDR80_GT_cell_coverage=len(p80 & truth) / len(truth), HDR50_cell_count=int(core.sum()), HDR80_cell_count=int(hdr80.sum()),
                  coarse_world_cell_AUPRC=float(average_precision_score(overlap, score)) if overlap.any() else None)
    return result


def empty_instance(mask: np.ndarray, affine: np.ndarray) -> dict:
    """Instance of a model without output for the target: every measure 0."""
    return instance_metrics(mask, affine, None, np.empty((0, 3), np.int32), np.empty(0), np.empty(0, bool), np.empty(0, bool))


def uniform_route_chance(centres_in_gt: np.ndarray, routed: np.ndarray, token_overlaps_gt: np.ndarray | None = None) -> dict:
    """centres_in_gt: per token, its centre lies in the annotation; routed: per token, inside the finding's route."""
    r = np.asarray(routed, bool).copy()
    if r.sum() < MIN_ROUTE_TOKENS:
        r[:] = True
    c = np.asarray(centres_in_gt, bool)
    out = dict(uniform_routing_peak_exact_hit_expectation=float(c[r].mean()), native_route_tokens=int(r.sum()))
    if token_overlaps_gt is not None and np.asarray(token_overlaps_gt).any():
        out['routing_only_coarse_token_AUPRC'] = float(average_precision_score(np.asarray(token_overlaps_gt, bool), r.astype(float)))
    return out


def aggregate(rows: list[dict], key: str = 'first_focus_center_inside_exact_mask') -> dict:
    pos = [r for r in rows if r.get('GT_positive')]; n = len(pos); hits = sum(bool(r.get(key)) for r in pos)
    m = dict(annotated_positive_cases=n, first_focus_hits=hits, first_focus_hit_rate=hits / n if n else None,
             exact_binomial95CI=clopper_pearson(hits, n) if n else None)
    for field in FIELDS + ['uniform_routing_peak_exact_hit_expectation', 'routing_only_coarse_token_AUPRC']:
        vals = [r[field] for r in pos if r.get(field) is not None]
        if vals:
            m['mean_' + field] = float(np.mean(vals))
    return m

"""Chest X-ray localization against boxes (paper Methods 4.10, last paragraph; Extended Data localization panels).

Map (FORA): referenced evidence per class ``E_c = max(w (s - b_c), 0)`` on the film token grid, where ``w`` is the
bag attention (normalised by the number of tokens), ``s`` the token logit and ``b_c`` the token-weighted mean
token logit of training studies negative for c. The maps of the classes mapped to a finding are combined with
weights given by their probabilities (``combine_pwsum``). Any other model can supply its own non-negative map on
a token grid.

Per instance (image x finding with boxes):
* ``foci``: 4 mm Gaussian smoothing (pixel spacing = 430 mm over the long side of the image as fed, token = 16 px),
  scale = 99.5% quantile of the positive smoothed values, foci = 8-connected components of the cells
  >= 0.3 x scale ranked by smoothed mass; the point is the peak of the top-mass focus. No positive evidence = miss.
* hit: the point (token centre, clipped to the film content box) lies inside any box of the finding.
* region Dice: the smoothed map is smoothed again with sigma 4 tokens and upsampled x4 (cubic); the region is
  ``U >= 0.1 max U`` (relative rule, ChestX-Det10 / ChestX-ray14 / RSNA) or ``U >= tau`` with a fixed absolute
  threshold chosen on development data (PadChest-GR); Dice uses the exact box coverage of each sub-cell.
* chance: the share of the film content box covered by the boxes.
Summary: mean over instances per finding, then mean over findings; 95% intervals from 2,000 image resamples
(seed 0).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, label, zoom

FIELD_MM = 430.; PATCH = 16; SUB = 4
RELATIVE_THRESHOLD = 0.1
PADCHEST_ABS_THRESHOLD = 0.0011789099433248812


def referenced_map(w: np.ndarray, s: np.ndarray, b: float, grid) -> np.ndarray:
    """E = max(w (s - b), 0) on the token grid; w = attention / number of tokens."""
    return np.maximum(np.asarray(w, np.float64) * (np.asarray(s, np.float64) - b), 0).reshape(grid)


def combine_pwsum(maps: list[np.ndarray], logits) -> tuple[np.ndarray, float]:
    p = 1 / (1 + np.exp(-np.asarray(logits, np.float64)))
    return sum(pc * e for pc, e in zip(p, maps)) / p.sum(), float(p.max())


def sigma_tokens(image_shape) -> float:
    return 4. / (PATCH * FIELD_MM / max(image_shape))


def foci(E: np.ndarray, sg: float) -> dict:
    if E.max() <= 0:
        return dict(cov=False)
    sm = gaussian_filter(E, sigma=sg, mode='constant'); nz = sm[sm > 0]
    scale = float(np.quantile(nz, .995)) if len(nz) else 0.
    if scale <= 0:
        return dict(cov=False)
    msk = sm >= 0.3 * scale
    lab, nl = label(msk, structure=np.ones((3, 3), bool)); cm = np.bincount(lab.ravel(), sm.ravel(), nl + 1)[1:]
    top = int(np.argmax(cm)) + 1; pk = int(np.argmax(np.where(lab == top, sm, -1)))
    tot = cm.sum()
    return dict(cov=True, sm=sm, point=divmod(pk, sm.shape[1]), mask=msk, n_foci=int(nl), n5=int((cm >= .05 * tot).sum()))


def hit_point(ty, tx, box, P, boxes) -> bool:
    """box = film content (r0, r1, c0, c1) in pixels of the image as fed; boxes = (k, 4) x0, y0, x1, y1."""
    r0, r1, c0, c1 = [int(v) for v in box]
    x = min(max(c0 + (tx + .5) * P - .5, c0), c1 - 1); y = min(max(r0 + (ty + .5) * P - .5, r0), r1 - 1)
    B = np.asarray(boxes, float)
    return bool(((B[:, 0] <= x) & (x <= B[:, 2]) & (B[:, 1] <= y) & (y <= B[:, 3])).any())


def sub_cover(grid, box, P, boxes):
    """Per sub-cell (token / 4): fraction covered by boxes and fraction inside the film content box."""
    gy, gx = [int(v) for v in grid]; r0, r1, c0, c1 = [int(v) for v in box]
    vv = r0 + np.arange(gy * P); uu = c0 + np.arange(gx * P); inc = (vv < r1)[:, None] & (uu < c1)[None, :]
    m = np.zeros((gy * P, gx * P), bool)
    for b in boxes:
        m[np.ix_((vv >= b[1]) & (vv <= b[3]), (uu >= b[0]) & (uu <= b[2]))] = True
    m &= inc; k = P // SUB
    return m.reshape(gy * SUB, k, gx * SUB, k).mean((1, 3)), inc.reshape(gy * SUB, k, gx * SUB, k).mean((1, 3))


def upmap(sm, sg=4):
    m = gaussian_filter(sm, sg, mode='constant') if sg else sm
    return np.maximum(zoom(m, SUB, order=3), 0)


def chance(box, boxes) -> float:
    r0, r1, c0, c1 = [int(v) for v in box]; m = np.zeros((r1 - r0, c1 - c0), bool); uu = np.arange(c0, c1); vv = np.arange(r0, r1)
    for b in boxes:
        m[np.ix_((vv >= b[1]) & (vv <= b[3]), (uu >= b[0]) & (uu <= b[2]))] = True
    return float(m.mean())


def score_instance(E, image_shape, grid, box, boxes, P=PATCH, rule='relative', tau=PADCHEST_ABS_THRESHOLD) -> dict:
    F = foci(E, sigma_tokens(image_shape))
    hit = hit_point(*F['point'], box, P, boxes) if F['cov'] else False
    U = upmap(F['sm'] if F['cov'] else np.zeros_like(E), 4); cov, cc = sub_cover(grid, box, P, boxes); g = float(cov.sum())
    t = RELATIVE_THRESHOLD * U.max() if rule == 'relative' else tau
    R = (U >= t) & (U > 0); a = float((R * cc).sum()); it = float((R * cov).sum())
    return dict(hit=hit, dice=2 * it / max(a + g, 1e-9), covered=F['cov'], chance=chance(box, boxes))


def class_mean(fi: np.ndarray, v: np.ndarray, w=None) -> np.ndarray:
    w = np.ones(len(fi)) if w is None else w; nF = fi.max() + 1
    v = v[:, None] if v.ndim == 1 else v
    s = np.stack([np.bincount(fi, w * v[:, j], nF) for j in range(v.shape[1])], 1); n = np.bincount(fi, w, nF)[:, None]
    ok = n[:, 0] > 0
    return (s[ok] / n[ok]).mean(0)


def image_bootstrap(fi, ii, v, nb=2000, seed=0):
    rng = np.random.default_rng(seed); nI = ii.max() + 1; out = []
    for _ in range(nb):
        w = np.bincount(rng.integers(0, nI, nI), minlength=nI).astype(float)[ii]; out.append(class_mean(fi, v, w))
    out = np.array(out)
    return np.percentile(out, 2.5, 0), np.percentile(out, 97.5, 0)


def summarise(records: list[dict], nb: int = 2000) -> dict:
    """records: dicts with image, finding, hit, dice, chance -> class-mean pointing / Dice / chance with intervals."""
    Fs = sorted({r['finding'] for r in records}); ims = sorted({r['image'] for r in records})
    fi = np.array([Fs.index(r['finding']) for r in records]); ii = np.array([ims.index(r['image']) for r in records])
    V = np.array([[r['hit'], r['dice'], r['chance']] for r in records], float)
    lo, hi = image_bootstrap(fi, ii, V, nb); mm = class_mean(fi, V)
    return dict(instances=len(records), images=len(ims), pointing=float(mm[0]), pointing_ci=[float(lo[0]), float(hi[0])],
                dice=float(mm[1]), dice_ci=[float(lo[1]), float(hi[1])], chance=float(mm[2]),
                per_finding={F: dict(n=int((fi == a).sum()), pointing=float(V[fi == a, 0].mean()), dice=float(V[fi == a, 1].mean())) for a, F in enumerate(Fs)})

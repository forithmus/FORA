"""Extent statistics of a finding's positive evidence and the token blocks around its evidence peak (Methods, "Finding
localization", "Measurements").

Input per requested class: the positive evidence E [N, K] of every token of every view of the study (``study_evidence``)
and the per-view token geometry (``view_info``).

``extents`` gives, per view whose peak reaches ``view_min_rel`` x the study peak and for the merged study (all views
snapped to a token-sized world grid, cell value = max over views), 36 threshold statistics (at 50 / 25 / 10 % of the
peak: token count, volume, extents in S / R / A, maximal extent, principal lengths, size of the connected component at
the peak) plus peak, mass, positive-token count, effective token count, token count, token size, native spacing, peak
relative to the study peak and mass share.

``peak_blocks`` returns the (2r + 1)^3 token block (head inputs, evidence, mask, offsets in mm) around the evidence peak
of the best view and of the best other view (a different series is preferred); the lesion calipers start from block 0.
"""
from __future__ import annotations

import numpy as np
import torch

THR = (0.5, 0.25, 0.1)
_PER = ('n', 'vol', 'ext_z', 'ext_x', 'ext_y', 'maxext', 'pca1', 'pca2', 'pca3', 'cc_n', 'cc_vol', 'cc_maxext')
STAT_NAMES = [f'{k}_{int(t * 100)}' for t in THR for k in _PER] + ['peak', 'mass', 'n_pos', 'n_eff', 'n_tokens', 'tok_z',
                                                                    'tok_x', 'tok_y', 'native_z', 'native_xy', 'rel_peak',
                                                                    'mass_share']
NS = len(STAT_NAMES)


def extent_stats(idx, world, vals, tok):
    """idx [n, 3] token-grid indices, world [n, 3] token centres mm, vals [n] positive evidence, tok [3] token size mm
    -> the 36 threshold statistics + peak, mass, n_pos, n_eff (40 floats)."""
    from scipy.ndimage import label
    out = np.full(40, np.nan, np.float64)
    vmax = float(vals.max()) if len(vals) else 0.
    if vmax <= 0:
        out[36:40] = (0., 0., 0., 0.)
        return out
    pk = int(vals.argmax())
    tv = float(np.prod(tok))
    st = np.ones((3, 3, 3), bool)
    for ti, t in enumerate(THR):
        m = vals >= t * vmax
        n = int(m.sum())
        P, w, I = world[m], vals[m], idx[m]
        ext = P.max(0) - P.min(0) + tok
        mu = (w[:, None] * P).sum(0) / w.sum()
        X = P - mu
        cov = (w[:, None, None] * (X[:, :, None] * X[:, None, :])).sum(0) / w.sum()
        ev = np.sort(np.clip(np.linalg.eigvalsh(cov), 0, None))[::-1]
        pl = np.sqrt(12 * ev)
        lo, hi = I.min(0), I.max(0)
        g = np.zeros(hi - lo + 1, bool)
        g[tuple((I - lo).T)] = True
        lab, _ = label(g, structure=st)
        c = lab[tuple(idx[pk] - lo)]
        cm = lab[tuple((I - lo).T)] == c
        cn = int(cm.sum())
        cext = (P[cm].max(0) - P[cm].min(0) + tok).max()
        out[ti * 12:(ti + 1) * 12] = (n, n * tv, ext[0], ext[1], ext[2], ext.max(), pl[0], pl[1], pl[2], cn, cn * tv, cext)
    s = float(vals.sum())
    out[36:40] = (vmax, s, float((vals > 0).sum()), s * s / float((vals ** 2).sum()))
    return out


def extents(E, vinfo, view_min_rel=0.05):
    """E [N, K] positive evidence -> merged [K, NS], per-view rows (q, view, series, window, stats [NS]), peak view [K]."""
    K = E.shape[1]
    offs = np.cumsum([0] + [len(v['zyx']) for v in vinfo])
    merged = np.full((K, NS), np.nan)
    rows = []
    pview = np.zeros(K, np.int64)
    W_all = np.concatenate([v['world'] for v in vinfo])
    tok = vinfo[0]['tok']
    for q in range(K):
        e = E[:, q]
        g_peak, g_mass = float(e.max()), float(e.sum())
        pview[q] = int(np.searchsorted(offs, int(e.argmax()), 'right') - 1)
        for k, v in enumerate(vinfo):
            ev = e[offs[k]:offs[k + 1]]
            vp = float(ev.max()) if len(ev) else 0.
            if g_peak <= 0 or vp < view_min_rel * g_peak:
                continue
            st = extent_stats(v['zyx'], v['world'], ev, v['tok'])
            rows.append((q, k, v['series'], v['window'],
                         np.concatenate([st, [len(ev), *v['tok'], *v['nat'], vp / g_peak, float(ev.sum()) / max(g_mass, 1e-30)]])))
        pos = e > 0
        if not pos.any():
            merged[q] = np.concatenate([extent_stats(np.zeros((0, 3), int), np.zeros((0, 3)), np.zeros(0), tok),
                                        [len(e), *tok, *vinfo[pview[q]]['nat'], 1., 1.]])
            continue
        cell = np.floor(W_all[pos] / tok[None]).astype(np.int64)
        val, wd = e[pos], W_all[pos]
        order = np.argsort(-val, kind='stable')
        cell, val, wd = cell[order], val[order], wd[order]
        _, first = np.unique(cell, axis=0, return_index=True)
        cell, val, wd = cell[first], val[first], wd[first]
        merged[q] = np.concatenate([extent_stats(cell - cell.min(0), wd, val, tok),
                                    [len(e), *tok, *vinfo[pview[q]]['nat'], 1., 1.]])
    return merged, rows, pview


def peak_blocks(E, vinfo, XA, r_, view_min_rel=0.05):
    """(2r + 1)^3 token block around the evidence peak of the best view and of the best other view (another series
    preferred) -> dict of x [K, 2, B^3, 1048] f16, ev, mask, off (mm), view, ctr (mm from the main peak), rel (view peak /
    study peak); grid order (z, axis 1, axis 2)."""
    K = E.shape[1]
    B = 2 * r_ + 1
    d = np.stack(np.meshgrid(*[np.arange(-r_, r_ + 1)] * 3, indexing='ij'), -1).reshape(-1, 3)
    offs = np.cumsum([0] + [len(v['zyx']) for v in vinfo])
    luts = {}
    BX = np.zeros((K, 2, B ** 3, 1048), np.float16)
    BE = np.zeros((K, 2, B ** 3), np.float32)
    BM = np.zeros((K, 2, B ** 3), bool)
    BO = np.zeros((K, 2, B ** 3, 3), np.float32)
    BV = np.full((K, 2), -1, np.int16)
    BC = np.zeros((K, 2, 3), np.float32)
    BR = np.zeros((K, 2), np.float32)
    for q in range(K):
        e = E[:, q]
        gp = float(e.max())
        if gp <= 0:
            continue
        vp = np.array([e[offs[k]:offs[k + 1]].max() for k in range(len(vinfo))])
        k1 = int(vp.argmax())
        other = [k for k in range(len(vinfo)) if k != k1 and vp[k] >= view_min_rel * gp]
        diff = [k for k in other if vinfo[k]['series'] != vinfo[k1]['series']]
        ks = [k1] + ([max(diff, key=lambda k: vp[k])] if diff else [max(other, key=lambda k: vp[k])] if other else [])
        for b, k in enumerate(ks):
            v = vinfo[k]
            ev = e[offs[k]:offs[k + 1]]
            lp = int(ev.argmax())
            if k not in luts:
                lut = np.full(v['grid'], -1, np.int64)
                lut[tuple(v['zyx'].T)] = np.arange(len(v['zyx']))
                luts[k] = lut
            lut = luts[k]
            P = v['zyx'][lp][None] + d
            inb = np.all((P >= 0) & (P < np.array(v['grid'])), 1)
            li = np.full(len(P), -1, np.int64)
            li[inb] = lut[tuple(P[inb].T)]
            mk = li >= 0
            gi = offs[k] + li[mk]
            BX[q, b][mk] = XA[torch.from_numpy(gi).to(XA.device)].cpu().numpy()
            BE[q, b][mk] = E[gi, q]
            BM[q, b] = mk
            BO[q, b] = d * v['tok'][None]
            BV[q, b] = k
            BR[q, b] = vp[k] / gp
            BC[q, b] = v['world'][lp] - vinfo[k1]['world'][int(e[offs[k1]:offs[k1 + 1]].argmax())]
    return dict(x=BX, ev=BE, mask=BM, off=BO, view=BV, ctr=BC, rel=BR)

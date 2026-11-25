"""Region masks on the work grid from sampled CADS label maps (the semantics of `fora.regions.definitions.REGIONS`).

Arrays are indexed (x = R, y = A, z = S): +x patient right, +y anterior, +z superior. The work grid has 1.5 mm in z
and WORK_MM = 2 mm in-plane, aligned to the token grid, so one 3 x 8 x 8 mm token = 2 x 4 x 4 work voxels. A token is
in a region if at least COVER = 25 % of its work voxels are (`token_cover`).

Two proxies are used because the CT itself is not re-read when the bits are computed for the cache:
  * body = the series file's body tokens (`flat_indices`), the exact token set the MIL head sees;
  * bone inside 'groin' = CADS bone labels (554:7-11, 559:5) instead of CT > 300 HU.
Passing `ct=` reproduces the CT-based body / bone (validation only).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from . import definitions as D

TARGET = (1.5, .5, .5)
PATCH = (2, 16, 16)
COVER = .25
WORK_MM = 2.0
BONE_IDS = ['554:7', '554:8', '554:9', '554:10', '554:11', '559:5']


class Builder:
    """Region masks for one series; `parts` maps CADS task id -> uint8 label map on the work grid."""

    def __init__(s, parts, sp, body, ct=None):
        s.parts = parts
        s.sp = np.asarray(sp, float)
        s.cache = {}
        s.ccache = {}
        s.ct = ct
        s.shape = body.shape
        if ct is not None:
            b = ct > -500
            for z in range(b.shape[2]):
                b[:, :, z] = ndi.binary_fill_holes(b[:, :, z])
            lab, n = ndi.label(b)
            if n > 1:
                sz = ndi.sum(b, lab, range(1, n + 1))
                b = lab == (1 + int(np.argmax(sz)))
            body = b
        s.body = body
        s.zmm = np.arange(s.shape[2]) * s.sp[2]
        s.xmm = np.arange(s.shape[0]) * s.sp[0]

    def cads(s, ids):
        key = tuple(sorted(ids))
        if key in s.ccache:
            return s.ccache[key]
        m = np.zeros(s.shape, bool)
        by = {}
        for c in ids:
            t, l = c.split(':')
            by.setdefault(int(t), []).append(int(l))
        for t, ls in by.items():
            p = s.parts.get(t)
            if p is None:
                continue
            lut = np.zeros(256, bool)
            lut[ls] = True
            m |= lut[p]
        s.ccache[key] = m
        return m

    def _box(s, m, mm):
        nz = [np.nonzero(m.any(tuple(a for a in range(3) if a != d)))[0] for d in range(3)]
        pad = [int(np.ceil(mm / sp)) + 2 for sp in s.sp]
        return tuple(slice(max(0, v[0] - p), min(n, v[-1] + 1 + p)) for v, p, n in zip(nz, pad, s.shape))

    def dil(s, m, mm):
        if not mm or not m.any():
            return m
        b = s._box(m, mm)
        out = np.zeros_like(m)
        out[b] = ndi.distance_transform_edt(~m[b], sampling=s.sp) <= mm
        return out

    def ero(s, m, mm):
        if not m.any():
            return m
        b = s._box(m, 0)
        out = np.zeros_like(m)
        out[b] = ndi.distance_transform_edt(m[b], sampling=s.sp) > mm
        return out

    def zr(s, m):
        z = np.nonzero(m.any((0, 1)))[0]
        return (s.zmm[z.min()], s.zmm[z.max()]) if len(z) else None

    def landmark(s, name):
        if name == 'lungs_base':
            m = s.cads(list(D.LUNG_LOBES.values()))
            z = np.nonzero(m.any((0, 1)))[0]
            if not len(z):
                return None
            w = m.sum((0, 1))[z]
            c = np.cumsum(w) / w.sum()
            return s.zmm[z[np.searchsorted(c, .02)]]
        if name == 'kidney_right_centre':
            m = s.cads(['551:2'])
            return s.zmm[int(np.nonzero(m)[2].mean())] if m.any() else None
        if name == 'hip_top':
            r = s.zr(s.cads(['554:9', '554:10']))
            return r and r[1]
        if name == 'lowest_rib':
            r = s.zr(s.cads(D.RIBS))
            return r and r[0]
        if name == 'pubis':
            r = s.zr(s.cads(['554:9', '554:10']))
            return r and r[0]
        raise KeyError(name)

    def zmask(s, lo=None, hi=None):
        z = np.ones(s.shape[2], bool)
        if lo is not None:
            z &= s.zmm >= lo
        if hi is not None:
            z &= s.zmm <= hi
        return np.broadcast_to(z[None, None], s.shape)

    def region(s, k):
        if k in s.cache:
            return s.cache[k]
        R = D.REGIONS[k]
        if R.get('special') == 'groin':
            m = s.groin()
        elif R.get('all_body'):
            m = s.body.copy()
        elif 'slab_of' in R:
            r = s.zr(s.cads(R['slab_of']))
            m = (s.body & s.zmask(*r)) if r else np.zeros_like(s.body)
        elif 'erode_mm' in R:
            m = s.ero(s.cads(R['cads']), R['erode_mm'])
        elif 'shell_mm' in R:
            c = s.cads(R['cads'])
            a, b = R['shell_mm']
            m = s.dil(c, a) & ~s.ero(c, b)
        else:
            m = s.cads(R.get('cads', [])).copy()
            for r in R.get('regions', []):
                m = m | s.region(r)
            m = s.dil(m, R.get('dilate_mm'))
        if R.get('within') == 'body':
            m = m & s.body
        for r in R.get('and_', []):
            m = m & s.region(r)
        for r in R.get('minus', []):
            m = m & ~s.region(r)
        if 'z_above' in R:
            v = s.landmark(R['z_above'])
            m = m & s.zmask(lo=v) if v is not None else np.zeros_like(m)
        if 'z_below' in R:
            v = s.landmark(R['z_below'])
            if v is not None:
                m = m & s.zmask(hi=v)
        if 'z_between' in R:
            a, b = (s.landmark(x) for x in R['z_between'])
            if a is not None and b is not None:
                m = m & s.zmask(min(a, b), max(a, b))
        if R.get('anterior_half'):
            ys = np.nonzero(s.body.any((0, 2)))[0]
            ymid = ys.mean() if len(ys) else s.shape[1] / 2
            m = m & np.broadcast_to((np.arange(s.shape[1]) > ymid)[None, :, None], m.shape)
        if R.get('right_half'):
            xs = np.nonzero(s.body.any((1, 2)))[0]
            mid = s.xmm[xs].mean() if len(xs) else s.xmm.mean()
            m = m & np.broadcast_to((s.xmm > mid)[:, None, None], m.shape)
        m = m & (s.body | (k in ('intracranial',)))
        s.cache[k] = m
        return m

    def groin(s):
        m = np.zeros_like(s.body)
        fem = s.cads(['554:7', '554:8'])
        pub = s.landmark('pubis')
        fz = s.zr(fem)
        if pub is None or fz is None or fem.sum() * np.prod(s.sp) < 20000:
            return m
        hip = s.cads(['554:9', '554:10'])
        X = s.xmm[:, None, None]
        Y = (np.arange(s.shape[1]) * s.sp[1])[None, :, None]
        Z = s.zmm[None, None, :]
        mid = s.xmm[np.nonzero(hip)[0]].mean()
        fx = s.xmm[np.nonzero(fem)[0]]
        half = max(abs(fx.max() - mid), abs(fx.min() - mid)) + 20
        fy = np.nonzero(fem)[1].mean() * s.sp[1]
        cav = s.cads(['559:3', '553:17', '553:16', '553:14', '556:5', '556:6'])
        bone = s.cads(['554:7', '554:8', '554:9', '554:10', '554:11']) | (s.ct > 300 if s.ct is not None else s.cads(BONE_IDS))
        m = s.body & ~cav & ~bone & (Y >= fy) & (np.abs(X - mid) <= half) & (Z >= pub - 100) & (Z <= fz[1] + 50)
        thigh = (Z < pub + 10) & (np.abs(X - mid) > 50)
        return m & ~thigh


def token_cover(m: np.ndarray, r: int) -> np.ndarray:
    """(R, A, S) work mask -> token grid (gz, gy, gx) bool: at least COVER of the token's work voxels are set."""
    X, Y, Z = m.shape
    f = (PATCH[1] // r, PATCH[2] // r, PATCH[0])
    frac = m.reshape(X // f[0], f[0], Y // f[1], f[1], Z // f[2], f[2]).mean((1, 3, 5), dtype=np.float32)
    return frac.transpose(2, 0, 1) >= COVER


def work_ratio() -> int:
    """Token-to-work-voxel in-plane ratio r: one token = PATCH[1] // r work voxels per in-plane axis."""
    return int(round(PATCH[1] * TARGET[1] / WORK_MM))

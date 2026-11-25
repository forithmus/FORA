"""Lesion-centred native CT crops for the lesion calipers (paper: Methods, "Measurements").

For each candidate (MIL evidence peak token of one finding) the peak view's series is read from the CT-FORA H5 only in the needed
sub-block, reoriented to RAS and linearly resampled to the DINO cache frame (1.5 x 0.5 x 0.5 mm; same arithmetic as the encoder's
series resampling: F.interpolate(size=int(n*cur/target), align_corners=False) == separable linear interpolation with source coordinate
(o + 0.5) * n_in / n_out - 0.5, clamped) and expressed in the body-cropped frame of the cache (body_crop_yxyx).
Crop box (cache frame, centred on the peak token centre): FOV = clip(3 x size prior x mult, 32, 240) mm (mult 1.5 for thickness / lumen
types); side in-plane = min(300, 2.1 FOV + 8) mm, side z = min(220, 1.25 FOV + 8) mm; stride k = ceil(side/0.5/256) in-plane,
kz = ceil(side_z/1.5/128); int16 HU, -1000 outside the volume.  Also returned: crop origin / stride (cache-frame voxels), HU window of the
view, the class organ route on the coarse token grid (sub-window covering the crop) and the peak token.

Case lines (one JSON object per study, JSONL): {"id": study id, "views": [{"relative_file", "file", "offset", "file_size",
"window": {"lo", "hi"}}, ...]} — the staged feature-pack view files of the study (`file` / `offset` / `file_size` locate the view's H5
inside a pack).  Token-region bits per view come from either a directory of `<relative_file>.npz` files (key `bits`, uint64, flat
token order) or per-node region packs (`regions_node_XX.h5` with `relative_files` / `offsets` / `bits`).
"""
from __future__ import annotations

import io
import json
import math
import os
from pathlib import Path

import numpy as np

TSP = np.array([1.5, 0.5, 0.5]); P = np.array([2, 16, 16])


class PackSlice(io.RawIOBase):
    """Read-only file object over one view H5 embedded in a feature pack (file, offset, size)."""
    def __init__(s, path, offset, size): s.f = open(path, 'rb', buffering=0); s.o = int(offset); s.n = int(size); s.p = 0
    def readable(s): return True
    def seekable(s): return True
    def tell(s): return s.p
    def seek(s, pos, whence=0): s.p = pos if whence == 0 else s.p + pos if whence == 1 else s.n + pos; return s.p

    def readinto(s, b):
        k = max(0, min(len(b), s.n - s.p))
        if k == 0: return 0
        s.f.seek(s.o + s.p); r = s.f.readinto(memoryview(b)[:k]); s.p += r; return r

    def close(s):
        try: s.f.close()
        finally: super().close()


def box_of(fov):
    side_xy = min(300., 2.1 * fov + 8.); side_z = min(220., 1.25 * fov + 8.)
    k = max(1, math.ceil(side_xy / 0.5 / 256)); kz = max(1, math.ceil(side_z / 1.5 / 128))
    n_xy = int(math.ceil(side_xy / (0.5 * k))); n_z = int(math.ceil(side_z / (1.5 * kz)))
    return np.array([n_z, n_xy, n_xy]), np.array([kz, k, k])


def fov_of(size_mm, mult=1.):
    return float(np.clip(3. * size_mm * mult, 32., 240.))   # min 32, max 240 mm; k = 3 x mult (4.5 for thickness / lumen)


class Series:
    """Lazy RAS-reoriented, cache-resampled view of one H5 series; reads only the requested sub-blocks."""
    def __init__(self, pv, h5_root):
        import h5py
        from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform, inv_ornt_aff
        self.f = h5py.File(f"{h5_root}/{pv['h5']}.h5", 'r'); sg = self.f[pv['patient']][pv['report']][pv['acc']][pv['sid']]
        src = sg[pv['sub']] if pv['sub'] else sg; self.ds = src['data']; self.frame = pv['frame']; aff = np.asarray(src['affine'][()], float)
        shp = tuple(self.ds.shape[:3]); t = ornt_transform(io_orientation(aff), axcodes2ornt(('R', 'A', 'S'))); self.t = t; self.shp = shp
        self.aff_ras = aff @ inv_ornt_aff(t, shp)
        ras_shape = [0, 0, 0]
        for i in range(3): ras_shape[int(t[i, 0])] = shp[i]
        sp = [float(s) if s > 0 else 1. for s in np.linalg.norm(self.aff_ras[:3, :3], axis=0)]   # (R, A, S)
        self.n_in = np.array([ras_shape[2], ras_shape[0], ras_shape[1]])                       # (S, R, A)
        cur = np.array([sp[2], sp[0], sp[1]]); self.n_out = np.array([max(1, int(n * c / tt)) for n, c, tt in zip(self.n_in, cur, TSP)])
        y0, y1, x0, x1 = pv['body_crop_yxyx']; self.off = np.array([0, y0, x0]); self.body_shape = np.array(pv['resampled_body_shape'])
        assert self.body_shape[0] == self.n_out[0] and y1 - y0 == self.body_shape[1] and x1 - x0 == self.body_shape[2], (self.body_shape, self.n_out, pv['body_crop_yxyx'])

    def src_coord(self, ax, o):   # resampled (full frame) index -> RAS-array (S,R,A) continuous index, as F.interpolate(align_corners=False)
        return np.clip((o + .5) * self.n_in[ax] / self.n_out[ax] - .5, 0, None)

    def read_ras(self, lo, hi):   # RAS-array (S,R,A) integer box [lo, hi) clipped -> float32 HU
        box = [None] * 3
        ras_lo = {2: lo[0], 0: lo[1], 1: lo[2]}; ras_hi = {2: hi[0], 0: hi[1], 1: hi[2]}   # ras axis j: 0=R,1=A,2=S
        flips = []
        for i in range(3):
            j = int(self.t[i, 0]); n = self.shp[i]
            if self.t[i, 1] < 0: box[i] = slice(n - ras_hi[j], n - ras_lo[j]); flips.append(i)
            else: box[i] = slice(ras_lo[j], ras_hi[j])
        a = self.ds[box[0], box[1], box[2], self.frame] if self.frame is not None else self.ds[box[0], box[1], box[2]]
        a = np.asarray(a, np.float32)
        for i in flips: a = np.flip(a, i)
        perm = [0, 0, 0]
        for i in range(3): perm[int(self.t[i, 0])] = i
        a = a.transpose(perm)                 # (R, A, S)
        return np.ascontiguousarray(a.transpose(2, 0, 1))   # (S, R, A)

    def resampled_block(self, r0, n, stride=(1, 1, 1)):
        """Cache-frame (body-cropped) voxel box: output index i <-> cache voxel r0 + stride*i + (stride-1)/2 (stride 1 = the exact
        1.5x0.5x0.5 mm resampling of the cache; stride > 1 = direct linear sampling at the block centres); -1000 outside the volume."""
        st = np.asarray(stride, np.int64); n = np.asarray(n, np.int64)
        out = np.full(tuple(n), -1000., np.float32); full0 = r0 + self.off
        pos = [full0[ax] + st[ax] * np.arange(n[ax]) + (st[ax] - 1) / 2. for ax in range(3)]
        okm = [(p_ >= 0) & (p_ <= self.n_out[ax] - 1) for ax, p_ in enumerate(pos)]
        if not all(m_.any() for m_ in okm): return out
        lo_k = [int(np.flatnonzero(m_)[0]) for m_ in okm]; hi_k = [int(np.flatnonzero(m_)[-1]) + 1 for m_ in okm]
        coords = [self.src_coord(ax, pos[ax][lo_k[ax]:hi_k[ax]]) for ax in range(3)]
        i0 = [np.floor(c).astype(np.int64) for c in coords]; lo_i = np.array([c.min() for c in i0]); hi_i = np.array([min(c.max() + 2, self.n_in[ax]) for ax, c in enumerate(i0)])
        a = self.read_ras(lo_i, hi_i)
        for ax in range(3):   # separable linear interpolation (== trilinear F.interpolate)
            c = coords[ax]; f0 = np.floor(c).astype(np.int64); lam = (c - f0).astype(np.float32); f1 = np.minimum(f0 + 1, self.n_in[ax] - 1)
            a0 = np.take(a, f0 - lo_i[ax], axis=ax); a1 = np.take(a, f1 - lo_i[ax], axis=ax); sh = [1, 1, 1]; sh[ax] = -1
            a = a0 * (1 - lam.reshape(sh)) + a1 * lam.reshape(sh)
        out[tuple(slice(l, h) for l, h in zip(lo_k, hi_k))] = a
        return out


# ---------------------------------------------------------------------------------------------------- region bits providers
class RegionDir:
    """Per-view token-region bits as <dir>/<relative_file>.npz (key `bits`, flat token order)."""
    def __init__(self, root): self.root = Path(root)

    def __call__(self, rel, case_id):
        p = self.root / (rel + '.npz')
        return np.load(p)['bits'].astype(np.uint64) if p.exists() else None


class RegionPacks:
    """Per-node region packs `regions_node_{node:02d}.h5` (datasets relative_files, offsets, bits); node = case_rank[case_id] // 4."""
    def __init__(self, pattern, case_rank):
        self.pattern = pattern; self.case_rank = case_rank; self.packs = {}

    def __call__(self, rel, case_id):
        import h5py
        node = self.case_rank[case_id] // 4
        if node not in self.packs:
            f = h5py.File(self.pattern.format(node), 'r'); rl = [x.decode() if isinstance(x, bytes) else x for x in f['relative_files'][:]]
            self.packs[node] = (f, {r: k for k, r in enumerate(rl)}, f['offsets'][:])
        f, ix, off = self.packs[node]; k = ix.get(rel)
        if k is None: return None
        return f['bits'][off[k]:off[k + 1]].astype(np.uint64)


def route_grid(bits, grid, flat, cls_bits):
    if bits is None: return None
    assert len(bits) == len(flat)
    g = np.zeros(int(np.prod(grid)), bool); g[flat] = (bits & np.uint64(cls_bits)) != 0
    return g.reshape(grid)


# ---------------------------------------------------------------------------------------------------- worker state + reader
_W = {}


def winit(cases_jsonl, h5_root, regions):
    """Pool initializer: byte-offset index of the case lines, H5 root and the region-bits provider."""
    off = {}; pos = 0
    with open(cases_jsonl, 'rb') as f:
        for l in f:
            i = l.find(b'"id": "'); off[l[i + 7:l.index(b'"', i + 7)].decode()] = pos; pos += len(l)
    _W.update(cases=cases_jsonl, off=off, h5_root=h5_root, regions=regions)


def case_line(cid):
    with open(_W['cases'], 'rb') as f:
        f.seek(_W['off'][cid]); return json.loads(f.readline())


def crops_for_case(job):
    """job = (case id, [(row, view, peak[3], fov_size_mm, bits_int64, fov_mult), ...]) -> list of item dicts (or ('err', row, msg))."""
    import h5py
    cid, rows = job; out = []
    try: case = case_line(cid)
    except Exception as ex: return [('err', r[0], repr(ex)[:200]) for r in rows]
    sc = {}
    for row, view, peak, fsz, bits, fm in rows:
        try:
            vw = case['views'][int(view)]
            if vw['relative_file'] not in sc:
                with h5py.File(PackSlice(vw['file'], vw.get('offset', 0), vw['file_size']), 'r') as f:
                    pv = json.loads(f.attrs['provenance_json']); flat = f['flat_indices'][:]; grid = json.loads(f['v00'].attrs['metadata_json'])['grid']
                sc[vw['relative_file']] = (Series(pv, _W['h5_root']), pv, flat, grid)
            S, pv, flat, grid = sc[vw['relative_file']]
            fov = fov_of(float(fsz), fm); nbox, stride = box_of(fov); pk = np.asarray(peak, float); c = (pk + .5) * P - .5
            ext = nbox * stride; r0 = np.round(c - (ext - 1) / 2.).astype(np.int64)
            crop = np.clip(np.round(S.resampled_block(r0, nbox, stride)), -32768, 32767).astype(np.int16); origin = r0 + (stride - 1) / 2.
            t_lo = np.floor(r0 / P).astype(int); t_hi = np.floor((r0 + ext - 1) / P).astype(int) + 1
            sub = np.zeros(tuple(t_hi - t_lo), bool)
            if int(bits) != -1:
                ub = int(np.int64(bits).view(np.uint64))
                rg = route_grid(_W['regions'](vw['relative_file'], cid), grid, flat, ub)
                if rg is not None:
                    a_lo = np.maximum(t_lo, 0); a_hi = np.minimum(t_hi, np.array(grid))
                    if np.all(a_hi > a_lo): sub[tuple(slice(l - t, h - t) for l, h, t in zip(a_lo, a_hi, t_lo))] = rg[tuple(slice(l, h) for l, h in zip(a_lo, a_hi))]
            out.append(dict(row=row, crop=crop, origin=origin.astype(np.float64), stride=stride.astype(np.float64), window=(vw['window']['lo'], vw['window']['hi']),
                            route=np.packbits(sub.ravel()), route_t0=np.concatenate([t_lo, t_hi]), peak=pk.astype(np.int64), fov=fov, grid=np.array(grid)))
        except Exception as ex:
            out.append(('err', row, repr(ex)[:200]))
    for S_, *_ in sc.values(): S_.f.close()
    return out

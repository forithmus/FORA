"""Per-token region bits for cached CT series (Methods, "Organ routing").

token_region_bits(series file) -> (bits uint64[n_tokens], REGION_NAMES, info). Bit i of bits[t] is set if body token
t (order of the file's `flat_indices`) has at least 25 % of its volume inside region REGION_NAMES[i].

Geometry. Regions are built on a token-aligned work grid in the extraction frame (RAS -> (S, R, A) -> resampling to
1.5 x 0.5 x 0.5 mm -> body crop -> padding to grid x patch), at 1.5 mm in z and WORK_MM in-plane. CADS labels are
sampled (nearest) at the work-voxel centres:
  * own CADS of the same array (no derivative sub-volume, CADS image_shape == orig_shape): a separable index map
    (RAS permutation / flips + the align_corners=False centre mapping of the trilinear resampling);
  * own CADS of the parent series (the series used a derivative, e.g. derivative_Tilt_Eq_1) or the CADS of another
    series of the same accession (borrowed, the series with the best coverage >= 10 %, ties to the finest voxels):
    world mapping through the affines stored in the CT-FORA H5;
  * no CADS: only 'body' and 'body_minus_lungs' (= body) are set.

CADS mask files (`cads_<h5>.h5`, one per CT shard) hold, at the same group path as the CT series, uint8 datasets
`part_551` ... `part_559` in the array order of the CT `data` and the attribute `image_shape`.
"""
from __future__ import annotations

import json
import os
import time

import h5py
import numpy as np

from ..cache.format import open_series
from .builder import PATCH, TARGET, WORK_MM, Builder, token_cover, work_ratio
from .definitions import REGION_NAMES

BORROW_MIN_COVER = .10


def ras_orientation(affine, shape):
    """-> (RAS index -> original index homogeneous transform, RAS array shape)."""
    from nibabel.orientations import axcodes2ornt, inv_ornt_aff, io_orientation, ornt_transform
    t = ornt_transform(io_orientation(affine), axcodes2ornt(('R', 'A', 'S')))
    ras_shape = [0, 0, 0]
    for d in range(3):
        ras_shape[int(t[d, 0])] = shape[d]
    return inv_ornt_aff(t, shape), ras_shape


def work_axes(prov: dict, grid):
    """1-D RAS-array coordinates of the work-voxel centres along R, A, S, the RAS -> original transform, the work
    spacing (R, A, S mm) and a geometry consistency flag (the recomputed resampled shape matches the provenance)."""
    A = np.asarray(prov['affine'], float)
    shape = list(prov['orig_shape'])[:3]
    ras2orig, rs = ras_orientation(A, shape)
    ras_aff = A @ ras2orig
    sp = np.linalg.norm(ras_aff[:3, :3], axis=0)
    rb = prov['resampled_body_shape']
    y0, y1, x0, x1 = prov['body_crop_yxyx']
    gz, gy, gx = grid
    r = work_ratio()
    assert PATCH[1] % r == 0

    def ax(n_ras, s_ras, t, n_work, step, off):
        new = max(1, int(n_ras * s_ras / t))
        c = np.arange(n_work) * step + (step - 1) / 2 + off          # target-voxel centre (resampled frame)
        return (c + .5) * n_ras / new - .5, new

    R, newR = ax(rs[0], sp[0], TARGET[1], gy * PATCH[1] // r, r, y0)
    Aa, newA = ax(rs[1], sp[1], TARGET[2], gx * PATCH[2] // r, r, x0)
    S, newS = ax(rs[2], sp[2], TARGET[0], gz * PATCH[0], 1, 0)
    ok = (newS == rb[0] and y1 - y0 == rb[1] and x1 - x0 == rb[2] and y1 <= newR and x1 <= newA)
    wsp = np.array([TARGET[1] * r, TARGET[2] * r, TARGET[0]])
    return (R, Aa, S), ras2orig, wsp, ok


def sample_identity(ds, coords, ras2orig) -> np.ndarray:
    """Nearest sample of an array in its own (original) index space at RAS 1-D centres -> (R, A, S) uint8."""
    shape = ds.shape
    idx = [None] * 3
    perm = [None] * 3
    M = ras2orig[:3, :3]
    for d in range(3):                       # original axis d depends on exactly one RAS axis j
        j = int(np.argmax(np.abs(M[d])))
        perm[d] = j
        idx[d] = np.rint(M[d, j] * coords[j] + ras2orig[d, 3]).astype(np.int64)
    valid = [(c >= 0) & (c < shape[d]) for d, c in enumerate(idx)]
    arr = ds[()] if hasattr(ds, 'shape') and not isinstance(ds, np.ndarray) else ds
    out = arr[np.ix_(*[np.clip(c, 0, shape[d] - 1) for d, c in enumerate(idx)])]
    for d in range(3):
        if not valid[d].all():
            sl = [slice(None)] * 3
            sl[d] = ~valid[d]
            out[tuple(sl)] = 0
    inv = [perm.index(j) for j in range(3)]
    return np.ascontiguousarray(out.transpose(inv))


def world_index_map(coords, ras2orig, A_src, A_cads, cads_shape) -> np.ndarray:
    """Per work voxel the flat CADS array index through world coordinates (-1 = outside)."""
    R, Aa, S = coords
    n = (len(R), len(Aa), len(S))
    T = np.linalg.inv(A_cads) @ A_src @ ras2orig
    g = np.meshgrid(R, Aa, S, indexing='ij')
    out = np.full(n, -1, np.int64)
    lin = np.zeros(n, np.int64)
    ok = np.ones(n, bool)
    for d in range(3):
        c = np.rint(T[d, 0] * g[0] + T[d, 1] * g[1] + T[d, 2] * g[2] + T[d, 3]).astype(np.int64)
        ok &= (c >= 0) & (c < cads_shape[d])
        lin = lin * cads_shape[d] + np.clip(c, 0, cads_shape[d] - 1)
    out[ok] = lin[ok]
    return out


def sample_world(arr, flat_idx) -> np.ndarray:
    out = np.zeros(flat_idx.shape, np.uint8)
    m = flat_idx >= 0
    out[m] = arr.reshape(-1)[flat_idx[m]]
    return out


def _cads_group(f, p, sid):
    k = f"{p['patient']}/{p['report']}/{p['acc']}/{sid}"
    return f[k] if k in f else None


def _src_geom(hf, p, sid, sub=None):
    g = hf[p['patient']][p['report']][p['acc']][sid]
    if sub:
        g = g[sub]
    return np.asarray(g['affine'][()], float), tuple(g['data'].shape[:3])


def token_body_work(flat, grid, r):
    """Body tokens on the token grid (z, R, A) and their (R, A, S) work-grid expansion."""
    tb = np.zeros(int(np.prod(grid)), bool)
    tb[flat] = True
    tb = tb.reshape(grid)
    body = np.repeat(np.repeat(np.repeat(tb.transpose(1, 2, 0), PATCH[1] // r, 0), PATCH[2] // r, 1), PATCH[0], 2)
    return tb, body


def bits_from_builder(builder: Builder, flat, r, names=REGION_NAMES, only=None):
    """uint64 bits per body token and per-region token counts from a Builder (`only`: restrict to these names)."""
    bits = np.zeros(len(flat), np.uint64)
    counts = {}
    for i, k in enumerate(names):
        if only is not None and k not in only:
            counts[k] = 0
            continue
        tok = token_cover(builder.region(k), r).ravel()[flat]
        counts[k] = int(tok.sum())
        if counts[k]:
            bits[tok] |= np.uint64(1) << np.uint64(i)
    return bits, counts


def _load_parts(p, coords, ras2orig, tb, wshape, ct_h5_dir, cads_dir, info):
    """Sample the CADS parts of a series (own or borrowed) on the work grid; updates info in place."""
    parts = {}
    r = work_ratio()
    cfn = f"{cads_dir}/cads_{p['h5']}.h5"
    A_src = np.asarray(p['affine'], float)
    if not os.path.exists(cfn):
        return parts
    with h5py.File(cfn, 'r') as cf:
        g = _cads_group(cf, p, p['sid'])
        src = None
        if g is not None:
            cs = tuple(int(v) for v in g.attrs['image_shape'])
            if not p['sub'] and cs == tuple(p['orig_shape'][:3]):
                info.update(cads_source='own', mapping='index', cads_sid=p['sid'])
            else:
                with h5py.File(f"{ct_h5_dir}/{p['h5']}.h5", 'r') as hf:
                    A_c, sh = _src_geom(hf, p, p['sid'])
                if sh == cs:
                    info.update(cads_source='own', mapping='world', cads_sid=p['sid'])
                    src = (A_c, cs)
                else:
                    g = None
        if g is None:
            acc = f"{p['patient']}/{p['report']}/{p['acc']}"
            cands = [s for s in (cf[acc].keys() if acc in cf else []) if s != p['sid']]
            if cands:
                fi = np.flatnonzero(tb.transpose(1, 2, 0).ravel())
                fi = fi[::max(1, len(fi) // 3000)]
                ti = np.stack(np.unravel_index(fi, tb.transpose(1, 2, 0).shape), 1).astype(float)
                k = PATCH[1] // r
                cz = [np.interp(ti[:, 0] * k + (k - 1) / 2, np.arange(wshape[0]), coords[0]),
                      np.interp(ti[:, 1] * k + (k - 1) / 2, np.arange(wshape[1]), coords[1]),
                      np.interp(ti[:, 2] * PATCH[0] + .5, np.arange(wshape[2]), coords[2])]
                pts = np.stack(cz + [np.ones(len(fi))])
                best = None
                with h5py.File(f"{ct_h5_dir}/{p['h5']}.h5", 'r') as hf:
                    for s in cands:
                        cs = tuple(int(v) for v in cf[acc][s].attrs['image_shape'])
                        try:
                            A_c, sh = _src_geom(hf, p, s)
                        except KeyError:
                            continue
                        if sh != cs:
                            continue
                        q = (np.linalg.inv(A_c) @ A_src @ ras2orig @ pts)[:3]
                        cov = float(np.mean(np.all((q > -.5) & (q < np.array(cs)[:, None] - .5), 0)))
                        vox = float(np.prod(np.linalg.norm(A_c[:3, :3], axis=0)))
                        if best is None or (cov, -vox) > (best[0], -best[1]):
                            best = (cov, vox, s, A_c, cs)
                if best is not None and best[0] >= BORROW_MIN_COVER:
                    g = cf[acc][best[2]]
                    src = (best[3], best[4])
                    info.update(cads_source='borrowed', mapping='world', cads_sid=best[2], borrow_cover=round(best[0], 3))
        if g is not None:
            fidx = world_index_map(coords, ras2orig, A_src, src[0], src[1]) if src is not None else None
            for k in g:
                if not k.startswith('part_'):
                    continue
                arr = g[k][()]
                parts[int(k[5:])] = sample_identity(arr, coords, ras2orig) if src is None else sample_world(arr, fidx)
                del arr
    return parts


def token_region_bits(path, ct_h5_dir: str, cads_dir: str):
    """Region bits of one cache series file (local path, or (pack_file, offset, file_size) inside a pack)."""
    t0 = time.time()
    with open_series(path) as f:
        p = json.loads(f.attrs['provenance_json'])
        flat = f['flat_indices'][()]
        grid = tuple(json.loads(f['v00'].attrs['metadata_json'])['grid'])
    coords, ras2orig, wsp, geom_ok = work_axes(p, grid)
    r = work_ratio()
    wshape = tuple(len(c) for c in coords)
    tb, body = token_body_work(flat, grid, r)
    info = dict(cads_source='none', mapping=None, cads_sid=None, geom_ok=bool(geom_ok), grid=list(grid),
                n_tokens=int(len(flat)), work_mm=WORK_MM)
    parts = _load_parts(p, coords, ras2orig, tb, wshape, ct_h5_dir, cads_dir, info)
    t1 = time.time()
    if info['cads_source'] == 'none':
        bits, counts = bits_from_builder(Builder({}, wsp, body), flat, r, only={'body', 'body_minus_lungs'})
    else:
        bits, counts = bits_from_builder(Builder(parts, wsp, body), flat, r)
    info.update(counts=counts, seconds=round(time.time() - t0, 2), read_seconds=round(t1 - t0, 2))
    return bits, REGION_NAMES, info

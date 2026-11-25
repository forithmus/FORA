"""Whole-volume CADS vessel measurer (CPU, no learning; paper: Methods, "Measurements").

Per case: series = first of the case's series (axial first, thinnest first, up to 4 tried) whose CADS group has the structure
(same rule as organ.py).  Masks are taken in the series' native index space, physical frame = index * spacing.
  aorta 551:7 (largest 3-D component)
    centreline: 1 mm isotropic copy of the mask -> 26-neighbour voxel graph -> geodesic distance from the aortic valve
      (aorta voxels within 3 mm of the LV 553:5; fallback: top cut plane when the aorta is truncated at the volume top, else
      the geodesic extreme end nearest the heart) -> 2 mm geodesic shells -> per shell the connected component that continues
      the previous one -> centroid -> Gaussian smoothing (centre 4 mm, tangent 6 mm).
    cross-section every shell: plane perpendicular to the smoothed tangent, sampled at 0.5 mm from the native mask
      (trilinear, > 0.5) -> component containing the centre -> max Feret ('fmax'), min Feret ('fmin'), area-equivalent
      diameter ('deq'), max inscribed diameter ('ins').
    segments along arc length s: root = first 30 mm; ascending = valve -> start of the arch; arch = part of the centreline
      above z_split (highest axial level at which the ascending and descending centrelines are separate by > sum of radii);
      descending thoracic = after the arch down to the lung bases (lowest lung voxel z, 551:13-17); abdominal = below.
  pulmonary artery 553:8: the same geodesic shells from the pulmonary valve (PA voxels within 3 mm of the RV 553:7); the
    bifurcation = first shell (persisting 2 shells) with >= 2 large components; trunk sections at 2..20 mm proximal of it.

`seg_values` / `pa_value` read one measurement variant (section measure, median-filter width, ascending skip / PA window)
out of a measured profile; the variant per class is chosen on the training split (calibration file `cads_calibration.json`)."""
import os
import time
import numpy as np, h5py
from scipy import ndimage as ndi
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, connected_components
from scipy.spatial import ConvexHull, cKDTree

from . import io

ISO = 1.0      # mm, centreline grid
BIN = 2.0      # mm, geodesic shell width
H = 0.5        # mm, cross-section sampling


def largest_cc(m):
    lab, n = ndi.label(m)
    if n <= 1: return m
    s = ndi.sum_labels(m, lab, np.arange(1, n + 1)); return lab == (1 + int(np.argmax(s)))


class Frame:
    """native crop of a mask: physical (mm, crop index * sp) <-> world."""
    def __init__(s, m, A, sp, pad=4):
        bb = ndi.find_objects(m.astype(np.uint8))[0]
        s.lo = np.array([max(b.start - pad, 0) for b in bb]); s.hi = np.array([min(b.stop + pad, n) for b, n in zip(bb, m.shape)])
        s.sl = tuple(slice(a, b) for a, b in zip(s.lo, s.hi)); s.A = A; s.sp = sp
        s.F = m[s.sl].astype(np.float32)

    def world(s, P):   # P: (n, 3) physical in crop frame
        idx = P / s.sp + s.lo
        return idx @ s.A[:3, :3].T + s.A[:3, 3]


def iso_graph(fr):
    """1 mm isotropic mask of the crop -> (node phys coords (n,3), csr graph)."""
    z = fr.sp / ISO
    if np.prod(np.array(fr.F.shape) * z) > 6e7 or fr.F.sum() * np.prod(fr.sp) > 2.5e6: raise ValueError('mask too large (> 2.5 l or huge bbox)')
    I = ndi.zoom(fr.F, z, order=1) > 0.5
    I = largest_cc(I)
    shp = I.shape; idx = np.flatnonzero(I.ravel()); n = len(idx)
    nid = np.full(I.size, -1, np.int32); nid[idx] = np.arange(n, dtype=np.int32)
    co = np.stack(np.unravel_index(idx, shp), 1)
    rows, cols, w = [], [], []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if (dx, dy, dz) <= (0, 0, 0): continue
                c2 = co + np.array([dx, dy, dz]); ok = np.all((c2 >= 0) & (c2 < shp), 1)
                j = np.full(n, -1, np.int32); j[ok] = nid[np.ravel_multi_index(c2[ok].T, shp)]
                k = j >= 0
                rows.append(np.flatnonzero(k)); cols.append(j[k]); w.append(np.full(k.sum(), ISO * np.sqrt(dx * dx + dy * dy + dz * dz)))
    r = np.concatenate(rows); c = np.concatenate(cols); ww = np.concatenate(w)
    G = csr_matrix((np.concatenate([ww, ww]), (np.concatenate([r, c]), np.concatenate([c, r]))), shape=(n, n))
    P = co * ((np.array(fr.F.shape) - 1) / np.maximum(np.array(shp) - 1, 1)) * fr.sp   # ndi.zoom grid: output i <-> input i*(n_in-1)/(n_out-1)
    return P.astype(np.float64), G


def shells(P, G, seeds):
    """geodesic shells from the seed nodes -> list of dicts (g, centroid, ncomp_big, size)."""
    d = dijkstra(G, directed=False, indices=seeds, min_only=True)
    ok = np.isfinite(d); kb = np.where(ok, (d // BIN).astype(np.int64), -1)
    order = np.argsort(kb, kind='stable'); kbs = kb[order]
    starts = np.searchsorted(kbs, np.arange(kbs.max() + 2)) if ok.any() else np.array([0])
    out = []; prev = None
    for k in range(len(starts) - 1):
        nodes = order[starts[k]:starts[k + 1]]
        if len(nodes) < 3: continue
        sub = G[nodes][:, nodes]; nc, lab = connected_components(sub, directed=False)
        sz = np.bincount(lab, minlength=nc); big = np.flatnonzero(sz >= max(min(20, sz.max()), 0.15 * sz.max()))
        cents = [P[nodes[lab == b]].mean(0) for b in big]
        if prev is None: j = int(np.argmax(sz[big]))
        else: j = int(np.argmin([np.linalg.norm(c - prev) for c in cents]))
        prev = cents[j]
        out.append(dict(g=k * BIN + BIN / 2, c=cents[j], nbig=len(big), size=int(sz[big[j]]),
                        cents=[c for i, c in enumerate(cents) if i != j], sizes=[int(sz[b]) for i, b in enumerate(big) if i != j]))
    return out, d


def smooth_line(C, sc=2., st=3.):
    C = np.asarray(C, float)
    if len(C) < 3: return C, np.tile([0, 0, 1.], (len(C), 1))
    Cs = ndi.gaussian_filter1d(C, sc, axis=0, mode='nearest')
    T = np.gradient(ndi.gaussian_filter1d(C, st, axis=0, mode='nearest'), axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-9
    return Cs, T


def feret(pts):
    if len(pts) < 3: return H, H
    try: h = pts[ConvexHull(pts).vertices]
    except Exception: h = pts
    d = h[:, None] - h[None]; fmax = float(np.sqrt((d ** 2).sum(-1)).max()) + H
    ang = np.linspace(0, np.pi, 90, endpoint=False); pr = h @ np.stack([np.cos(ang), np.sin(ang)])
    fmin = float((pr.max(0) - pr.min(0)).min()) + H
    return fmax, fmin


def section(fr, c, t, R):
    a = np.array([1., 0, 0]) if abs(t[0]) < .9 else np.array([0, 1., 0])
    u = np.cross(t, a); u /= np.linalg.norm(u); v = np.cross(t, u)
    g = np.arange(-R, R + 1e-6, H); aa, bb = np.meshgrid(g, g, indexing='ij')
    Pp = c[None] + aa.reshape(-1, 1) * u + bb.reshape(-1, 1) * v
    val = ndi.map_coordinates(fr.F, (Pp / fr.sp).T, order=1, cval=0.).reshape(aa.shape) > 0.5
    if not val.any(): return None
    lab, n = ndi.label(val); m0 = len(g) // 2
    l0 = lab[m0, m0]
    if l0 == 0:
        e = ndi.distance_transform_edt(lab == 0); pts = np.argwhere(lab > 0)
        dd = np.linalg.norm(pts - m0, axis=1); j = int(np.argmin(dd))
        if dd[j] * H > 5: return None
        l0 = lab[tuple(pts[j])]
    M = lab == l0
    touch = bool(M[0].any() or M[-1].any() or M[:, 0].any() or M[:, -1].any())
    area = M.sum() * H * H; deq = 2 * np.sqrt(area / np.pi)
    edge = M & ~ndi.binary_erosion(M); pts = np.argwhere(edge) * H
    fmax, fmin = feret(pts)
    ins = 2 * ndi.distance_transform_edt(np.pad(M, 1)).max() * H
    return dict(fmax=fmax, fmin=fmin, deq=float(deq), ins=float(ins), touch=int(touch))


def profile(fr, Cs, T, Rloc):
    rows = []
    for c, t, r in zip(Cs, T, Rloc):
        R = float(np.clip(2.2 * r + 12, 25, 70)); s = section(fr, c, t, R)
        rows.append(s if s is not None else dict(fmax=np.nan, fmin=np.nan, deq=np.nan, ins=np.nan, touch=0))
    return {k: np.array([r_[k] for r_ in rows], float) for k in ('fmax', 'fmin', 'deq', 'ins', 'touch')}


def local_radius(P, G, Cs):
    tr = cKDTree(P); d, _ = tr.query(Cs, k=1)
    return d


def seeds_near(P, fr, other, mm=3.):
    """iso nodes within mm of a native mask 'other' (bool, full volume)."""
    if other is None or not other.any(): return np.array([], int)
    o = other[fr.sl]
    if not o.any(): return np.array([], int)
    surf = o & ~ndi.binary_erosion(o)
    q = np.argwhere(surf).astype(float) * fr.sp
    if len(q) > 300000: q = q[np.random.default_rng(0).choice(len(q), 300000, replace=False)]
    dv, _ = cKDTree(q).query(P, k=1, distance_upper_bound=mm + 1e-6)
    return np.flatnonzero(dv <= mm)


def zsup(fr, P, ssign):
    return fr.world(P)[:, 2] * ssign


def _ends(G, zs):
    d0 = dijkstra(G, directed=False, indices=int(np.argmax(zs))); d0[~np.isfinite(d0)] = -1
    e1 = int(np.argmax(d0)); d1 = dijkstra(G, directed=False, indices=e1); d1[~np.isfinite(d1)] = -1; e2 = int(np.argmax(d1))
    return e1, e2


def _aorta_comp(m, A, sp, parts, ssign, ztop_vol, zlb):
    """one connected component of the aorta mask -> profile dict (seg 1 asc, 2 arch, 3 desc thoracic, 4 abdominal, 0 = role decided later)."""
    fr = Frame(m, A, sp)
    P, G = iso_graph(fr)
    if len(P) < 50: return None
    lv = parts[553] == 5
    sd = seeds_near(P, fr, lv, 5.); how = 'lv'
    near_lv = len(seeds_near(P, fr, lv, 8.)) >= 5
    zs = zsup(fr, P, ssign)
    if len(sd) < 5:
        e1, e2 = _ends(G, zs)
        if zs.max() > ztop_vol - 6:    # truncated at the volume top: no arch in this component
            if near_lv: sd = np.array([[e1, e2][int(np.argmin(zs[[e1, e2]]))]]); how = 'low'
            else: sd = np.flatnonzero(zs >= zs.max() - 3); how = 'top'
        else:
            heart = np.isin(parts[553], [3, 4, 5, 6, 7])
            if heart.any():
                hc = np.argwhere(heart[::4, ::4, ::4]).mean(0) * 4; hcw = hc @ A[:3, :3].T + A[:3, 3]
                ew = fr.world(P[[e1, e2]]); e = [e1, e2][int(np.argmin(np.linalg.norm(ew - hcw, axis=1)))]
            else: e = [e1, e2][int(np.argmax(zs[[e1, e2]]))]
            sd = np.array([e]); how = 'end'
    sh, gd = shells(P, G, sd)
    if len(sh) < 3 or len(sh) > 450: return None
    Cs, T = smooth_line([s_['c'] for s_ in sh])
    s = np.array([s_['g'] for s_ in sh]); W = fr.world(Cs); z = W[:, 2] * ssign
    r_loc = np.sqrt(np.array([s_['size'] for s_ in sh]) * ISO ** 3 / BIN / np.pi)   # shell area ~ size * ISO^3 / BIN
    pr = profile(fr, Cs, T, r_loc)
    n = len(s); seg = np.zeros(n, np.int8)
    itop = int(np.argmax(z)); out = dict(how=how, n=n, near_lv=int(near_lv), s=s, z=z, y=W[:, 1], nvox=int(m.sum()))
    has_arch = how not in ('top', 'low') and 0 < itop < n - 1 and (z[itop] - z[0] > 20) and (z[itop] - z[-1] > 20)
    out['arch'] = int(has_arch)
    if has_arch:
        # z_split: highest level where the ascending (s < s_top) and descending (s > s_top) centrelines are separate
        za = z[:itop + 1]; zd = z[itop:]; wa = W[:itop + 1]; wd = W[itop:]; ra = r_loc[:itop + 1]; rd = r_loc[itop:]
        zsplit = None
        for zz in np.arange(np.floor(z[itop]), max(za.min(), zd.min()), -1.):
            ia = np.flatnonzero(za <= zz); idd = np.flatnonzero(zd <= zz)
            if not len(ia) or not len(idd): continue
            i1 = ia[-1]; i2 = idd[0]
            if np.linalg.norm((wa[i1] - wd[i2])[:2]) > (ra[i1] + rd[i2]) * 1.05: zsplit = zz; break
        if zsplit is None: zsplit = z[itop] - 25
        ia = np.flatnonzero(za > zsplit); idd = np.flatnonzero(zd > zsplit)
        a0 = ia[0] if len(ia) else itop; d1 = itop + (idd[-1] if len(idd) else 0)
        seg[:a0] = 1; seg[a0:d1 + 1] = 2; seg[d1 + 1:] = 3
        if zlb is not None: seg[(seg == 3) & (z < zlb)] = 4
        out['zsplit'] = float(zsplit)
    out['seg'] = seg; out.update(pr)
    out['nbig'] = np.array([s_['nbig'] for s_ in sh])
    return out


def vol_geom(shape, A, sp, parts, ssign):
    """S of the volume top, S extent of the volume, S of the lung bases (None without lungs) - from the full volume."""
    cor = np.array(np.meshgrid(*[[0, n - 1] for n in shape], indexing='ij')).reshape(3, -1).T
    zc = (cor @ A[:3, :3].T + A[:3, 3])[:, 2] * ssign
    zax = int(np.argmax(np.abs(A[2, :3]) / sp)); zlb = None
    prof = np.zeros(shape[zax], np.int64)
    lab = parts[551]
    for k in range(shape[zax]):
        sl = [slice(None)] * 3; sl[zax] = k; a = lab[tuple(sl)]; prof[k] = int(((a >= 13) & (a <= 17)).sum())
    if prof.sum() > 1000:
        zi = np.flatnonzero(prof > 0); c2 = np.zeros((2, 3)); c2[:, zax] = [zi.min(), zi.max()]
        zlb = float(((c2 @ A[:3, :3].T + A[:3, 3])[:, 2] * ssign).min())
    return float(zc.max()), float(zc.max() - zc.min()), zlb


def crop_to(m, A, sp, parts, margin_mm=40.):
    ii = [np.flatnonzero(m.any(axis=tuple(b for b in range(3) if b != a))) for a in range(3)]
    mg = np.ceil(margin_mm / sp).astype(int)
    lo = np.array([max(int(i.min()) - g, 0) for i, g in zip(ii, mg)])
    hi = np.array([min(int(i.max()) + 1 + g, n) for i, g, n in zip(ii, mg, m.shape)])
    sl = tuple(slice(a, b) for a, b in zip(lo, hi)); Ac = A.copy(); Ac[:3, 3] = A[:3, :3] @ lo + A[:3, 3]
    return np.ascontiguousarray(m[sl]), Ac, {k: v[sl] for k, v in parts.items()}


def measure_aorta(m, A, sp, parts, ssign, ztop_vol, zlb):
    """all aorta components >= 15 % of the largest (cardiac / truncated FOVs split ascending and descending)."""
    lab, nl = ndi.label(m)
    sz = ndi.sum_labels(m, lab, np.arange(1, nl + 1)); order = np.argsort(-sz)
    keep = [int(j) + 1 for j in order[:3] if sz[j] >= max(0.15 * sz[order[0]], 3000. / np.prod(sp))]
    comps = []
    for k in keep:
        r = _aorta_comp(lab == k, A, sp, parts, ssign, ztop_vol, zlb)
        if r is not None: comps.append(r)
    if not comps: return None
    # roles of the components without an arch
    any_asc = any(c['arch'] for c in comps)
    free = [c for c in comps if not c['arch']]
    for c in free:
        if c['near_lv'] and not any_asc: c['role'] = 1
    if len(free) >= 2 and not any(c.get('role') == 1 for c in free) and not any_asc:
        ant = max(free, key=lambda c: np.mean(c['y'])); ant['role'] = 1
    for c in free:
        if c.get('role') == 1:
            c['seg'][:] = 1
            if len([1 for c2 in free if c2.get('role') == 1]) > 1 and c is not max([c2 for c2 in free if c2.get('role') == 1], key=lambda c2: c2['nvox']):
                c['seg'][:] = 3
        else:
            c['seg'][:] = 3
            if zlb is not None: c['seg'][c['z'] < zlb] = 4
            elif not (np.isin(parts[553], [3, 4, 5, 6, 7]).sum() > 1000): c['seg'][:] = 4
    out = dict(zlb=zlb, ncomp=len(comps), how=[c['how'] for c in comps], arch=[c['arch'] for c in comps], zsplit=[c.get('zsplit') for c in comps])
    for k in ('s', 'z', 'y', 'seg', 'fmax', 'fmin', 'deq', 'ins', 'touch', 'nbig'):
        out[k] = np.round(np.concatenate([np.asarray(c[k], float) for c in comps]), 2).tolist()
    out['comp'] = np.concatenate([np.full(c['n'], i) for i, c in enumerate(comps)]).tolist()
    return out


def measure_pa(m, A, sp, parts, ssign):
    m = largest_cc(m); fr = Frame(m, A, sp)
    P, G = iso_graph(fr)
    rv = parts[553] == 7
    sd = seeds_near(P, fr, rv, 4.); how = 'rv'
    if len(sd) < 5:
        zs = zsup(fr, P, ssign)
        d0 = dijkstra(G, directed=False, indices=int(np.argmax(zs))); d0[~np.isfinite(d0)] = -1
        e1 = int(np.argmax(d0)); d1 = dijkstra(G, directed=False, indices=e1); d1[~np.isfinite(d1)] = -1; e2 = int(np.argmax(d1))
        # trunk origin: the most anterior-inferior extreme (world y = A, minus z)
        ew = fr.world(P[[e1, e2, int(np.argmin(zs))]]); sc = ew[:, 1] - ew[:, 2] * ssign
        sd = np.array([[e1, e2, int(np.argmin(zs))][int(np.argmax(sc))]]); how = 'end'
    sh, gd = shells(P, G, sd)
    if len(sh) < 4: return None
    nb = np.array([s_['nbig'] for s_ in sh]); g = np.array([s_['g'] for s_ in sh])
    ib = None
    for i in range(2, len(sh) - 1):
        if nb[i] >= 2 and nb[i + 1] >= 2 and g[i] >= 10: ib = i; break
    out = dict(how=how, n=len(sh), bif=-1 if ib is None else int(ib))
    last = ib if ib is not None else min(len(sh), int(50 / BIN))
    C = [s_['c'] for s_ in sh[:last + 1]]
    Cs, T = smooth_line(C)
    r_loc = np.sqrt(np.array([s_['size'] for s_ in sh[:last + 1]]) * ISO ** 3 / BIN / np.pi)
    pr = profile(fr, Cs, T, r_loc)
    out['s'] = g[:last + 1].tolist(); out['z'] = zsup(fr, Cs, ssign).tolist()
    out.update({k: np.round(v, 2).tolist() for k, v in pr.items()})
    out['nbig'] = nb.tolist()
    return out


def orient(A):
    M = A[:3, :3]; sp = np.linalg.norm(M, axis=0); zax = int(np.argmax(np.abs(M[2]) / sp)); return zax, sp


def process_case(c, need):
    cand = []
    for p, s in io.case_series(c):
        cfn = io.cads_file(p['h5'])
        if not os.path.exists(cfn): continue
        try:
            with h5py.File(cfn, 'r') as cf:
                g = io.cads_group(cf, p, p['sid'])
                if g is None: continue
                A = np.asarray(g['image_affine'][()], float); shp = tuple(int(v) for v in g.attrs['image_shape'])
        except Exception: continue
        zax, sp = orient(A); thick = float(sp.max()); axial = int(zax == int(np.argmax(sp)) or thick < 1.25 * sp.min())
        cand.append(((1 - axial, thick), p, s, A, shp, zax, sp))
    cand.sort(key=lambda t: t[0]); todo = set(need); out = {}
    for dz, p, s, A, shp, zax, sp in cand[:4]:
        if not todo: break
        with h5py.File(io.cads_file(p['h5']), 'r') as cf:
            g = io.cads_group(cf, p, p['sid']); parts = {pt: g[f'part_{pt}'][()] for pt in (551, 553) if f'part_{pt}' in g}
        if 551 not in parts or 553 not in parts: continue
        # superior sign check: heart above liver
        ssign = 1.
        hv = np.isin(parts[553], [3, 4, 5, 6, 7]); lv_ = parts[551] == 5
        if hv.sum() > 500 and lv_.sum() > 500:
            zh = (np.argwhere(hv[::4, ::4, ::4]).mean(0) * 4) @ A[:3, :3].T; zl = (np.argwhere(lv_[::4, ::4, ::4]).mean(0) * 4) @ A[:3, :3].T
            if zh[2] < zl[2]: ssign = -1.
        meta = dict(series=s['relative_file'], dz=dz[1], axial=1 - dz[0], sp=[float(v) for v in sp], zax=zax, h5=p['h5'], sid=p['sid'], ssign=ssign)
        for o in sorted(todo):
            m = parts[551] == 7 if o == 'aorta' else parts[553] == 8
            if m.sum() < 200: continue
            t0 = time.time()
            if 'geom' not in meta: meta['geom'] = vol_geom(m.shape, A, sp, parts, ssign)
            ztop_vol, zext, zlb = meta['geom']
            if zext > 1200: out.setdefault('skipped', []).append(f'{o}: z extent {zext:.0f} mm'); continue
            mc, Ac, pc = crop_to(m, A, sp, parts)
            try: r = measure_aorta(mc, Ac, sp, pc, ssign, ztop_vol, zlb) if o == 'aorta' else measure_pa(mc, Ac, sp, pc, ssign)
            except ValueError as ex: out.setdefault('skipped', []).append(f'{o}: {ex}'[:120]); continue
            if r is None: continue
            r.update({k: v for k, v in meta.items() if k != 'geom'}); r['sec'] = round(time.time() - t0, 1); out[o] = r; todo.discard(o)
    return out


# ---------------------------------------------------------------- variant read-outs
def seg_values(a, meas, w, asc_skip):
    """aorta record -> {segment code: value} for one variant."""
    seg = np.array(a['seg']); s = np.array(a['s']); comp = np.array(a['comp']); tch = np.array(a['touch']) > 0
    if meas == 'mid': v = (np.array(a['fmax'], float) + np.array(a['fmin'], float)) / 2
    else: v = np.array(a[meas], float)
    v = np.where(tch, np.nan, v)
    if w > 1:
        out = np.full_like(v, np.nan)
        for c in np.unique(comp):
            k = comp == c; x = v[k]; ok = np.isfinite(x)
            if ok.sum() >= 1:
                xf = np.where(ok, x, np.nanmedian(x)); out[k] = np.where(ok, ndi.median_filter(xf, size=w, mode='nearest'), np.nan)
        v = out
    def mx(k):
        x = v[k]; x = x[np.isfinite(x)]; return float(x.max()) if len(x) else np.nan
    # the ascending's distance from the valve: s of the first ascending section of the component
    s0 = np.zeros_like(s)
    for c in np.unique(comp):
        k = (comp == c) & (seg == 1)
        if k.any(): s0[comp == c] = s[k].min()
    rel = s - s0
    r = {1: mx((seg == 1) & (rel >= asc_skip)), 2: mx(seg == 2), 3: mx(seg == 3), 4: mx(seg == 4), 5: mx((seg == 1) & (rel <= 30))}
    if not np.isfinite(r[1]): r[1] = mx(seg == 1)
    r[6] = np.nanmax([r[1], r[2], r[3], -1]); r[6] = r[6] if r[6] > 0 else np.nan
    r[7] = np.nanmax([r[6], r[4], -1]); r[7] = r[7] if r[7] > 0 else np.nan
    return r


def pa_value(p, meas, a_, w_, agg):
    if meas == 'mid': v = (np.array(p['fmax'], float) + np.array(p['fmin'], float)) / 2
    else: v = np.array(p[meas], float)
    v = np.where(np.array(p['touch']) > 0, np.nan, v); s = np.array(p['s'])
    if p['bif'] >= 0: sb = s[-1]; k = (s >= sb - a_ - w_) & (s <= sb - a_)
    else: k = (s >= 10) & (s <= 40)
    x = v[k]; x = x[np.isfinite(x)]
    if not len(x): x = v[np.isfinite(v)]
    if not len(x): return np.nan
    return float(x.max() if agg == 'max' else np.median(x))

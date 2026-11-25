"""Mid-sagittal plane fit and plane-frame resampling of the CADS brain labels (part 557) and the CT for the midline-shift and
Evans-index measurers (paper: Methods, "Measurements").

Per case: series = axial-ish, thinnest first (or closest to a ~4 mm recon with pref='thick'), part_557 with an intracranial cavity
>= MINVOL ml.  Ideal midline = the reflection-symmetry plane of the intracranial-cavity surface (WM + GM + CSF + blood = inner table),
Nelder-Mead over (yaw, roll, offset) minimising the mean truncated distance between the reflected and the original surface points
(KD-tree), started from a small yaw/roll grid.  Labels (nearest) and CT (linear) are resampled into the plane frame: u = signed
distance from the plane (0.5 mm), v = A-P (1 mm), w = S-I (1.5 mm).  The septum / ventricle analysis is in measure.py."""
import numpy as np, h5py
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy.optimize import minimize

from . import io
from .organ import orient

CAV = (1, 2, 3, 8); MINVOL = 500.; DEEP = 12.; DU, DV, DW = 0.5, 1.0, 1.5; CROP = 45.


def largest_cc(m):
    lab, n = ndi.label(m)
    if n <= 1: return m
    s = ndi.sum_labels(m, lab, np.arange(1, n + 1)); return lab == (1 + int(np.argmax(s)))


def normal_of(yaw, roll):
    # unit normal starting from world x, yaw about world z, roll about world y (degrees)
    a, b = np.deg2rad(yaw), np.deg2rad(roll)
    return np.array([np.cos(a) * np.cos(b), np.sin(a) * np.cos(b), np.sin(b)])


def fit_plane(P):
    """P: (N,3) world surface points of the cavity. Returns unit normal n, offset d (plane n.x = d), cost."""
    tree = cKDTree(P); rng = np.random.default_rng(0)
    Q = P[rng.choice(len(P), min(len(P), 6000), replace=False)]
    c0 = P.mean(0)

    def cost(x):
        n = normal_of(x[0], x[1]); d = x[2]
        R = Q - 2 * ((Q @ n) - d)[:, None] * n[None]
        dd, _ = tree.query(R, distance_upper_bound=15.)
        return float(np.minimum(dd, 15.).mean())
    best = None
    for yaw in range(-30, 31, 6):
        for roll in (-12, -6, 0, 6, 12):
            n = normal_of(yaw, roll); x = np.array([yaw, roll, float(c0 @ n)]); cst = cost(x)
            if best is None or cst < best[0]: best = (cst, x)
    r = minimize(cost, best[1], method='Nelder-Mead', options=dict(xatol=0.05, fatol=1e-4, maxiter=400,
                                                                     initial_simplex=best[1] + np.array([[0, 0, 0], [3, 0, 0], [0, 3, 0], [0, 0, 2]])))
    n = normal_of(r.x[0], r.x[1]); return n, float(r.x[2]), float(r.fun), [float(v) for v in r.x]


def frame_resample(L, A, n, d, Pw, order=0, cval=0):
    """resample index-space volume L into the plane frame. Returns Lf[w,v,u], frame dict."""
    ap = np.array([0., 1., 0.]) - n[1] * n; ap /= np.linalg.norm(ap); si = np.cross(n, ap)
    if si[2] < 0: si = -si
    o = Pw.mean(0); o = o - ((o @ n) - d) * n
    rel = Pw - o; uu, vv, ww = rel @ n, rel @ ap, rel @ si
    um = float(np.abs(uu).max()) + 4
    u = np.arange(-um, um + 1e-6, DU); v = np.arange(vv.min() - 4, vv.max() + 4, DV); w = np.arange(ww.min() - 3, ww.max() + 3, DW)
    Ai = np.linalg.inv(A)
    out = np.zeros((len(w), len(v), len(u)), L.dtype)
    UU, VV = np.meshgrid(u, v)
    base = o[None, None] + UU[..., None] * n + VV[..., None] * ap
    for k, wk in enumerate(w):
        X = base + wk * si; I = (Ai[:3, :3] @ X.reshape(-1, 3).T) + Ai[:3, 3:4]
        out[k] = ndi.map_coordinates(L, I, order=order, mode='constant', cval=cval).reshape(UU.shape)
    return out, dict(o=o, n=n, ap=ap, si=si, u0=float(u[0]), v0=float(v[0]), w0=float(w[0]), nu=len(u), nv=len(v), nw=len(w))


def process_case(c, want_panel=False, crop=CROP, pref='thin'):
    """pref = 'thin' (thinnest axial series first) or 'thick' (series closest to a ~4 mm recon first); crop = |u| half width (mm) kept."""
    cand = []
    for p, s in io.case_series(c):
        cfn = io.cads_file(p['h5'])
        if not os.path.exists(cfn): continue
        try:
            with h5py.File(cfn, 'r') as cf:
                g = io.cads_group(cf, p, p['sid'])
                if g is None or 'part_557' not in g: continue
                A = np.asarray(g['image_affine'][()], float); shp = tuple(int(v) for v in g.attrs['image_shape'])
        except Exception: continue
        zax, sp = orient(A, shp); thick = float(sp.max()); axial = int(zax == int(np.argmax(sp)) or thick < 1.25 * sp.min())
        cand.append(((1 - axial, thick), p, s, A, shp, zax, sp))
    if pref == 'thick': cand.sort(key=lambda t: (t[0][0], abs(t[0][1] - 4.0)))   # standard ~3-5 mm recon (less noise)
    else: cand.sort(key=lambda t: t[0])
    if not cand: return dict(fail='no_557'), None
    last = None
    for dz, p, s, A, shp, zax, sp in cand[:3]:
        with h5py.File(io.cads_file(p['h5']), 'r') as cf:
            L = io.cads_group(cf, p, p['sid'])['part_557'][()]
        C = np.isin(L, CAV)
        vol = C.sum() * np.prod(sp) / 1000
        if vol < MINVOL: last = dict(fail='small_cavity', cav_native_ml=float(vol)); continue
        C = largest_cc(C)
        B = C & ~ndi.binary_erosion(C)
        idx = np.stack(np.nonzero(B), 1)
        if len(idx) > 60000: idx = idx[np.random.default_rng(1).choice(len(idx), 60000, replace=False)]
        Pw = idx @ A[:3, :3].T + A[:3, 3]
        n, d, cst, x = fit_plane(Pw)
        Lf, fr = frame_resample(L, A, n, d, Pw)
        with h5py.File(io.h5_file(p['h5']), 'r') as hf:
            sg = hf[p['patient']][p['report']][p['acc']][p['sid']]; sg = sg[p['sub']] if p['sub'] else sg
            d3 = sg['data']; vol3 = np.asarray(d3[()] if d3.ndim == 3 else d3[..., p['frame']], np.float32)
            Ah = np.asarray(sg['affine'][()], float) if 'affine' in sg else A
        CTf, _ = frame_resample(vol3, Ah, n, d, Pw, order=1, cval=-1024)
        del vol3
        C = np.isin(Lf, CAV)
        for k in range(C.shape[0]): C[k] = ndi.binary_fill_holes(C[k])
        C = largest_cc(C)
        edt = ndi.distance_transform_edt(np.pad(C, 1), sampling=(DW, DV, DU))[1:-1, 1:-1, 1:-1]
        u = fr['u0'] + DU * np.arange(fr['nu']); cen = np.abs(u) <= crop
        Uu = u[None, None, :]
        par = np.isin(Lf, (1, 2)) & C; nl, nr = float((par & (Uu < 0)).sum()), float((par & (Uu > 0)).sum())
        cl, cr = float((C & (Uu < 0)).sum()), float((C & (Uu > 0)).sum())
        f = dict(series=s['relative_file'], h5=p['h5'], sid=p['sid'], sp=[float(v) for v in sp], axial=1 - dz[0],
                 plane_cost=cst, plane_x=x, cav_native_ml=float(vol), cav_ml=float(C.sum() * DU * DV * DW / 1000),
                 par_asym=(nr - nl) / max(nl + nr, 1), cav_asym=(cr - cl) / max(cl + cr, 1), tilt_affine=int(not np.allclose(Ah, A)),
                 shape=list(Lf.shape), u0=float(u[cen][0]), w0=fr['w0'], v0=fr['v0'],
                 cav_area=[float(v) for v in C.sum((1, 2)) * DU * DV])
        ctq = np.clip(np.round(CTf + 20), 0, 100).astype(np.uint8)          # HU -20..80 -> 0..100
        lab = (Lf + 16 * C).astype(np.uint8)
        arr = dict(ctq=ctq[:, :, cen], lab=lab[:, :, cen], edt=np.clip(edt[:, :, cen], 0, 255).astype(np.uint8))
        if want_panel: arr['ctq_full'] = ctq; arr['u0_full'] = np.float32(fr['u0'])
        return f, arr
    return last or dict(fail='none'), None

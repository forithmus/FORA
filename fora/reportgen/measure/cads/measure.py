"""Image-only CT anatomy measurements for the report generator from CADS anatomical segmentations (CPU, no learning;
paper: Methods, "Measurements").

Input : case list (caselist.py): one JSON line per case {case, report, h5, patient, series [[acc, sid]...], classes [...]}.
Output: one JSON line per case {case, report, classes, records: [per (case, class) measurement], ref: {always-available
        reference measurements}, raw: {...}, sec, err}.

Methods (all on the CADS label volume of ONE series per task: axial first, then thinnest; up to 4 series tried):
  vessels  centreline measurer (vessel.py): CADS aorta 551:7 / pulmonary artery 553:8 ->
           1 mm geodesic centreline from the valve -> cross-sections perpendicular to the smoothed tangent every 2 mm (max/min
           Feret, area-equivalent and inscribed diameter) -> segments root / ascending / arch / descending / abdominal; PA trunk
           window before the bifurcation.  Per class the training-split section measure x median filter
           and its training calibration (mult = ratio, add = lumen -> outer-wall offset) are applied (calibration file 'vessel').
  liver    CADS 551:5 largest 3-D component: craniocaudal span (z extent of the series' S axis, mm) = primary, 3-D max extent;
           calibrated with the training median log ratio vs report sizes (calibration file 'organ').
  spleen   CADS 551:1: 3-D max extent (= train-chosen 'length') primary, craniocaudal span; train calibration as above.
  kidneys  CADS 551:2 right / 551:3 left: 3-D max extent (bipolar length) per side, craniocaudal span.
           (series rule: the organ must not touch the z border of the volume if another series has it whole.)
  midline  see MIDLINE below; CADS has no ventricle / falx labels (part 557 = 1 WM, 2 GM, 3 CSF, 4-7 scalp / eyes / bone, 8 blood),
           so a symmetry-based method on the intracranial cavity (inner table) + HU ventricles is used.

MIDLINE (midline shift, mm)
  1. ideal midline: 3-D mid-sagittal plane = reflection-symmetry plane of the intracranial-cavity surface (CADS 557 WM+GM+CSF+
     blood = inner table of the skull): Nelder-Mead over (yaw, roll, offset) minimising the mean truncated distance between the
     reflected surface points and the surface (KD-tree), from a yaw/roll grid (midline.py).
  2. CT + labels resampled into the plane frame (u = signed distance to the plane 0.5 mm, v = A-P 1 mm, w = S-I 1.5 mm).
  3. ventricles = cavity & smoothed HU < 13 & cavity-EDT > 10 mm & |u| < 32 mm (HU-based: the CADS CSF label misses
     effaced / dilated ventricles); band = 35-75 mm below the vertex (fallback: slices with >= 30 % of the max ventricle area).
  4. septum pellucidum: per A-P row of a band slice, the gap (<= 14 mm) between a left and a right ventricular run (each >= 2 mm)
     -> septum point = gap centre; slice septum = median of its row points (>= 3 rows).
  5. foramen-of-Monro level: the septum spans from the callosal body down to the fornix / foramen of Monro; the Monro level is
     approximated by the 3 most inferior slices that still have a septum (frontal horns separated by the septum, just above the
     third ventricle).  monro = |median signed septum u| over these slices = perpendicular distance from the ideal midline.
     Other candidates: median over all septum slices, slice with the most septum rows, top-3-rows slices.  The feature, the
     smoothing set and the calibration (ratio / log-linear) are chosen on TRAIN by 5-fold CV median APE (calibration file 'midline').
  sign: u > 0 = towards world +x of the RAS affine = patient RIGHT; direction = side the septum is displaced to.

EVANS (Evans index) - reuses the MIDLINE plane frame, cavity and HU ventricles (frame crop widened to |u| <= 90 mm
  so the whole inner table is in the frame; the midline features only use |u| < 32 mm and are unchanged).
  ventricles V as in MIDLINE but |u| <= 45 mm; on every band slice, every A-P row with a septum gap (left run | gap <= 14 mm |
  right run, as in step 4) gives a frontal-horn width = outer edge of the left run -> outer edge of the right run; frontal horns =
  the anterior half (v >= median v) of the slice's septum rows; slice horn width = 2nd-largest anterior row width (robust max).
  inner skull diameter = max L-R (u) extent of the filled intracranial cavity on the same slice (inner table to inner table).
  evans = horn / skull at the slice with the largest horn width ('same'); 'maxskull' = horn / max cavity width over the band.
CTR (cardiothoracic ratio) - axial CT practice: maximal transverse cardiac diameter (outer cardiac border to outer
  border on one axial slice) / internal thoracic diameter (inner rib margin to inner rib margin) on the SAME axial slice.
  heart = largest 3-D component of CADS 553:3-7 (myocardium + 4 chambers) | 556:3 (heart); thorax = largest 3-D component of
  CADS 559:4 thoracic cavity | 559:9 mediastinum | 559:7 pericardium | heart.  Widths = extent along the voxel axis closest to
  world L-R.  Series: axial first, thinnest, heart >= 150 ml not touching the z border of the volume (else fallback).
  Also kept: 'max' = max heart width / max thoracic width over the heart z-range, '_p' = heart incl. pericardium (559:7).
  Training-split validation (reports stating the value): CTR 'same' n 116 medAE 0.025 raw / 0.020 calibrated (-0.022), r 0.83;
  Evans 'same' n 33 medAE 0.018 raw / 0.010 calibrated (+0.014; < 0.25 reported as '< 0.25'), r 0.63-0.72 (positives only).
usage: python -m fora.reportgen.measure.cads.measure --cases cases.jsonl --out out.jsonl --calib data/measure/cads_calibration.json
         --h5-root H5_DIR --cads-root CADS_DIR [--node-idx i --nnodes n] [--procs 64] [--tasks vessel,organ,midline,ctr]
  study-level lines + per-finding Evans / CTR texts: study.py
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')
import sys, json, time, argparse, traceback, signal
import numpy as np
from scipy import ndimage as ndi

from . import io
from . import vessel as mv
from . import organ as mc
from . import midline as ml
from .classes import VESSEL, LIVER, SPLEEN, KIDNEY, MIDLINE, EVANS, CTR

SEGC = {'asc': 1, 'arch': 2, 'desc': 3, 'abd': 4, 'root': 5, 'thor': 6, 'all': 7}
# vessel variants always computed (aorta: measure, median-filter width, ascending skip mm; pa: measure, a, w, agg)
AORTA_VARIANTS = [('fmax', 7, 0), ('mid', 7, 0), ('fmax', 7, 15), ('deq', 3, 0), ('fmax', 1, 0), ('deq', 7, 0), ('fmin', 1, 15), ('mid', 3, 0)]
PA_VARIANTS = [('deq', 28, 12, 'med'), ('deq', 8, 8, 'med'), ('fmax', 8, 8, 'med')]
MID_PARAMS = {
    'b2': dict(T=13., DEEP=10., VTOP=35., VBOT=75., GAPMAX=14., RUNMIN=2., UMAX=32., SEPMAX=25., SIG=3., SIGW=2., GAPSEL=.3, OPEN=1, SIGU=.5),
    'a1': dict(T=13., DEEP=10., VTOP=35., VBOT=75., GAPMAX=14., RUNMIN=2., UMAX=32., SEPMAX=25., SIG=2., SIGW=2., GAPSEL=.3, OPEN=1, SIGU=.5),
}
LIMITS = dict(vessel=90, organ=60, midline=150, ctr=60)
EVANS_UMAX = 45.; MID_CROP = 90.
DU, DV, DW = ml.DU, ml.DV, ml.DW


class Timeout(Exception): pass


def _alarm(sig, frm): raise Timeout()


def timed(fn, sec, *a):
    signal.signal(signal.SIGALRM, _alarm); signal.alarm(sec); t0 = time.time()
    try: return fn(*a), None, time.time() - t0
    except Timeout: return None, f'timeout {sec} s', time.time() - t0
    except Exception as e: return None, repr(e)[:200], time.time() - t0
    finally: signal.alarm(0)


def fin(x, nd=2):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


# ---------------------------------------------------------------- vessels
def vessel_raw(c):
    out = mv.process_case(c, ['aorta', 'pa'])
    raw = dict(meta={})
    a = out.get('aorta'); p = out.get('pa')
    if out.get('skipped'): raw['meta']['skipped'] = out['skipped']
    if a is not None:
        raw['meta']['aorta'] = dict(series=a['series'], dz=a['dz'], ncomp=a['ncomp'], how=a['how'], arch=a['arch'], sec=a['sec'])
        for v in AORTA_VARIANTS:
            r = mv.seg_values(a, *v); raw['aorta|' + '|'.join(map(str, v))] = {k: fin(r[SEGC[k]]) for k in SEGC}
    if p is not None:
        raw['meta']['pa'] = dict(series=p['series'], dz=p['dz'], how=p['how'], bif=p['bif'], sec=p['sec'])
        for v in PA_VARIANTS: raw['pa|' + '|'.join(map(str, v))] = fin(mv.pa_value(p, *v))
    return raw


def vessel_value(raw, seg, ch):
    """class segment + vessel calibration entry -> (calibrated, raw, segment used); falls back to a wider segment when missing."""
    var = ch['variant']
    if var[0] == 'pa':
        x = raw.get('pa|' + '|'.join(var[1:])); used = 'pa'
    else:
        r = raw.get('aorta|' + '|'.join(var[1:]))
        if r is None: return None, None, None
        x = r.get(seg); used = seg
        for alt in ({'asc': ['thor'], 'root': ['thor'], 'thor': [], 'abd': ['desc'], 'all': []}[seg] + ['all']):
            if x is not None: break
            x = r.get(alt); used = alt
    if x is None or x <= 0: return None, None, None
    y = x / np.exp(ch['factor']) if ch['cal'] == 'mult' else x - ch['factor']
    return fin(y, 1), x, used


def vessel_segments(raw, ch):
    """all aorta segments with the class's measure variant + calibration (lets the report pick the named segment)."""
    if ch['variant'][0] == 'pa': return None
    r = raw.get('aorta|' + '|'.join(ch['variant'][1:]))
    if not r: return None
    cv = lambda x: None if x is None else fin(x / np.exp(ch['factor']) if ch['cal'] == 'mult' else x - ch['factor'], 1)
    return {k: cv(r.get(k)) for k in ('root', 'asc', 'arch', 'desc', 'abd')}


# ---------------------------------------------------------------- organs
ORGANS = {'liver': 5, 'spleen': 1, 'kidney_R': 2, 'kidney_L': 3}   # CADS part 551 labels


def organ_raw(c):
    """Series rule: axial-thinnest first (<= 4 series); an organ touching the z border of the volume is kept
    only if no other series has it whole."""
    import h5py
    cand = []
    for p, s in io.case_series(c):
        cfn = io.cads_file(p['h5'])
        if not os.path.exists(cfn): continue
        try:
            with h5py.File(cfn, 'r') as cf:
                g = io.cads_group(cf, p, p['sid'])
                if g is None or 'part_551' not in g: continue
                A = np.asarray(g['image_affine'][()], float); shp = tuple(int(v) for v in g.attrs['image_shape'])
        except Exception: continue
        zax, sp = mc.orient(A, shp); thick = float(sp.max()); axial = int(zax == int(np.argmax(sp)) or thick < 1.25 * sp.min())
        cand.append(((1 - axial, thick), p, s, A, shp, zax, sp))
    cand.sort(key=lambda t: t[0]); todo = set(ORGANS); r = {}; touch = {}
    for dz, p, s, A, shp, zax, sp in cand[:4]:
        if not todo: break
        with h5py.File(io.cads_file(p['h5']), 'r') as cf:
            L = io.cads_group(cf, p, p['sid'])['part_551'][()]
        for o in sorted(todo):
            m = L == ORGANS[o]
            if m.sum() < 200: continue
            res = mc.organ_feats(m, A, zax, sp, shp[zax])
            if res is None: continue
            f = res[0]
            d = dict(cc_mm=fin(f['cc_mm'], 1), max3d_mm=fin(f['max3d_mm'], 1), inplane_feret_mm=fin(f['inplane_feret_mm'], 1),
                     vol_ml=fin(f['vol_ml'], 1), z_touch=f['z_touch'], series=s['relative_file'], dz=dz[1])
            if f['z_touch']:
                touch.setdefault(o, d); continue
            r[o] = d; todo.discard(o)
        del L
    for o in todo:
        if o in touch: r[o] = touch[o]
    return r


# ---------------------------------------------------------------- midline
def _runs(row):
    d = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0] - 1))


def septum_feats(f, ctq, lab, edt, p):
    """Septum-pellucidum offset features incl. the Monro-level feature; returns dict of features (mm, signed where noted)."""
    out = {}
    hu = ctq.astype(np.float32) - 20
    hu = ndi.gaussian_filter(hu, (p['SIGW'] / DW, p['SIG'] / DV, (p['SIG'] if p['SIGU'] < 0 else p['SIGU']) / DU))
    C = lab >= 16
    u = f['u0'] + DU * np.arange(ctq.shape[2])
    V = C & (hu < p['T']) & (edt > p['DEEP']); V[:, :, np.abs(u) > p['UMAX']] = False
    if p['OPEN']: V = ndi.binary_opening(V, structure=np.ones((1, 2, 3), bool))
    area = np.asarray(f['cav_area']); nz = len(area); top = np.nonzero(area > 300)[0]; trunc = 1
    if len(top):
        kt = top.max(); trunc = int(kt >= nz - 3 and area[kt] > 0.3 * area.max())
    vk = V.sum((1, 2)) * DU * DV
    if not trunc:
        dmm = (kt - np.arange(nz)) * DW; band = np.nonzero((dmm >= p['VTOP']) & (dmm <= p['VBOT']))[0]
    else: band = np.nonzero(vk >= 0.3 * max(vk.max(), 1))[0]
    out['trunc'] = trunc
    sl = []
    for k in band:
        rp = []
        for j in range(V.shape[1]):
            row = V[k, j]
            if row.sum() < 4: continue
            rr = [(s, e) for s, e in _runs(row) if (e - s + 1) * DU >= p['RUNMIN']]; best = None
            for a, b in zip(rr[:-1], rr[1:]):
                gap = (b[0] - a[1] - 1) * DU
                if gap <= 0 or gap > p['GAPMAX']: continue
                cu = 0.5 * (u[a[1]] + u[b[0]])
                if abs(cu) > p['SEPMAX']: continue
                m = min(a[1] - a[0], b[1] - b[0]) * DU - p['GAPSEL'] * abs(cu)
                if best is None or m > best[0]: best = (m, cu)
            if best is not None: rp.append(best[1])
        if len(rp) >= 3: sl.append((int(k), float(np.median(rp)), len(rp)))
    out['n_sl'] = len(sl)
    if sl:
        sg = np.array([s[1] for s in sl]); a_ = np.abs(sg)
        out['med'] = float(np.median(a_)); out['max'] = float(a_.max())
        kw = max(sl, key=lambda s: s[2]); out['rows'] = abs(kw[1]); out['rows_signed'] = kw[1]
        t3 = sorted(sl, key=lambda s: -s[2])[:3]; out['top3'] = float(abs(np.median([s[1] for s in t3])))
        lo = sorted(sl, key=lambda s: s[0])[:3]                     # k increases superiorly (w axis = +S)
        ms = float(np.median([s[1] for s in lo])); out['monro'] = abs(ms); out['monro_signed'] = ms
        out['monro_w_mm'] = float(f['w0'] + DW * np.median([s[0] for s in lo]))
        out['med_signed'] = float(np.median(sg))
    if len(band) and V[band].any():
        Uu = np.broadcast_to(u, V.shape); Vb = np.zeros_like(V); Vb[band] = V[band]
        out['vent_cu'] = float(abs(Uu[Vb].mean())); out['vent_cu_signed'] = float(Uu[Vb].mean())
    return out


def evans_feats(f, ctq, lab, edt, p):
    """Evans index from the plane-frame arrays (see EVANS)."""
    hu = ctq.astype(np.float32) - 20
    hu = ndi.gaussian_filter(hu, (p['SIGW'] / DW, p['SIG'] / DV, (p['SIG'] if p['SIGU'] < 0 else p['SIGU']) / DU))
    C = lab >= 16
    u = f['u0'] + DU * np.arange(ctq.shape[2])
    V = C & (hu < p['T']) & (edt > p['DEEP']); V[:, :, np.abs(u) > EVANS_UMAX] = False
    if p['OPEN']: V = ndi.binary_opening(V, structure=np.ones((1, 2, 3), bool))
    area = np.asarray(f['cav_area']); nz = len(area); top = np.nonzero(area > 300)[0]; trunc = 1
    if len(top):
        kt = top.max(); trunc = int(kt >= nz - 3 and area[kt] > 0.3 * area.max())
    vk = V.sum((1, 2)) * DU * DV
    if not trunc:
        dmm = (kt - np.arange(nz)) * DW; band = np.nonzero((dmm >= p['VTOP']) & (dmm <= p['VBOT']))[0]
    else: band = np.nonzero(vk >= 0.3 * max(vk.max(), 1))[0]
    sl = []
    for k in band:
        rows = []
        for j in range(V.shape[1]):
            row = V[k, j]
            if row.sum() < 4: continue
            rr = [(s_, e_) for s_, e_ in _runs(row) if (e_ - s_ + 1) * DU >= p['RUNMIN']]; best = None
            for a, b in zip(rr[:-1], rr[1:]):
                gap = (b[0] - a[1] - 1) * DU
                if gap <= 0 or gap > p['GAPMAX']: continue
                cu = 0.5 * (u[a[1]] + u[b[0]])
                if abs(cu) > p['SEPMAX']: continue
                m = min(a[1] - a[0], b[1] - b[0]) * DU - p['GAPSEL'] * abs(cu)
                if best is None or m > best[0]: best = (m, (b[1] - a[0] + 1) * DU)
            if best is not None: rows.append((j, best[1]))
        if len(rows) < 3: continue
        js = np.array([r[0] for r in rows]); ws = np.array([r[1] for r in rows])
        ant = ws[js >= np.median(js)]
        hw = float(np.sort(ant)[-2]) if len(ant) >= 2 else float(ant.max())
        ck = C[k]; cols = np.nonzero(ck.any(0))[0]
        sk = float((cols.max() - cols.min() + 1) * DU) if len(cols) else 0.
        sl.append((int(k), hw, sk))
    out = dict(n_sl=len(sl))
    if not sl: return out
    cw = [float((np.ptp(np.nonzero(C[k].any(0))[0]) + 1) * DU) for k in band if C[k].any()]
    kb = max(sl, key=lambda t: t[1])
    out.update(horn_mm=kb[1], skull_mm=kb[2], skull_max_mm=max(cw), same=kb[1] / kb[2] if kb[2] > 0 else None,
               maxskull=kb[1] / max(cw), w_mm=float(f['w0'] + DW * kb[0]),
               skull_edge=int(np.abs(u[[0, -1]]).min() - kb[2] / 2 < 3))      # cavity reaches the frame edge (should not happen)
    return out


def midline_raw(c):
    """thinnest axial series (S1) + the series closest to a ~4 mm recon ('_t' keys, only if it is a different series)."""
    f, arr = ml.process_case(c, False, crop=MID_CROP)   # whole inner table in the frame (Evans); midline features use |u| < 32 mm only
    if arr is None: return dict(fail=f.get('fail', 'none'))
    r = dict(series=f['series'], plane_cost=fin(f['plane_cost'], 3), cav_ml=fin(f['cav_ml'], 0), par_asym=fin(f['par_asym'], 4),
             normal=[round(float(x), 4) for x in ml.normal_of(f['plane_x'][0], f['plane_x'][1])])
    for nm, p in MID_PARAMS.items():
        r[nm] = {k: (fin(v) if isinstance(v, float) else v) for k, v in septum_feats(f, arr['ctq'], arr['lab'], arr['edt'], p).items()}
    r['evans'] = {k: (fin(v, 4) if isinstance(v, float) else v) for k, v in evans_feats(f, arr['ctq'], arr['lab'], arr['edt'], MID_PARAMS['b2']).items()}
    del arr
    f2, arr2 = ml.process_case(c, False, crop=MID_CROP, pref='thick')
    if arr2 is not None and f2['series'] != f['series']:
        r['series_t'] = f2['series']
        for nm, p in MID_PARAMS.items():
            r[nm + '_t'] = {k: (fin(v) if isinstance(v, float) else v) for k, v in septum_feats(f2, arr2['ctq'], arr2['lab'], arr2['edt'], p).items()}
        r['evans_t'] = {k: (fin(v, 4) if isinstance(v, float) else v) for k, v in evans_feats(f2, arr2['ctq'], arr2['lab'], arr2['edt'], MID_PARAMS['b2']).items()}
    return r


def _mget(raw, prm, key):
    """prm 'b2' = thinnest series; 'b2m' = mean over the thinnest and the ~4 mm series (whichever exist)."""
    if not prm.endswith('m'): return raw.get(prm, {}).get(key)
    v = [raw.get(b, {}).get(key) for b in (prm[:-1], prm[:-1] + '_t')]; v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


def midline_value(raw, cal):
    """calibration entry {feature, params, fallback, kind, a, b, r_cu} -> (calibrated mm, raw mm, signed raw, source)."""
    if raw is None or 'fail' in raw: return None, None, None, None
    ft = cal['feature']; x = _mget(raw, cal['params'], ft); sgn = _mget(raw, cal['params'], ft + '_signed'); src = 'S1'
    if x is None:
        x = _mget(raw, cal['fallback'], ft); sgn = _mget(raw, cal['fallback'], ft + '_signed'); src = 'S2'
    if x is None:
        v = _mget(raw, cal['params'], 'vent_cu')
        if v is None: return None, None, None, None
        x = v * cal['r_cu']; sgn = _mget(raw, cal['params'], 'vent_cu_signed'); src = 'S3'
    y = np.exp(cal['a'] + cal['b'] * np.log(max(x, cal.get('eps', 0.5))))
    return fin(y, 1), fin(x), sgn, src


# ---------------------------------------------------------------- cardiothoracic ratio
def _lcc(m):
    lab, n = ndi.label(m)
    if n <= 1: return m
    return lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)


def _widths(M, zax, lr, sp_lr):
    """per slice along zax: extent (mm) of M along voxel axis lr."""
    Ms = np.moveaxis(M, (zax, lr), (0, 1)); prof = Ms.any(2)            # [z, lr]
    w = np.zeros(prof.shape[0])
    for k in np.nonzero(prof.any(1))[0]:
        ix = np.nonzero(prof[k])[0]; w[k] = (ix[-1] - ix[0] + 1) * sp_lr
    return w


def ctr_raw(c):
    import h5py
    cand = []
    for p, s in io.case_series(c):
        cfn = io.cads_file(p['h5'])
        if not os.path.exists(cfn): continue
        try:
            with h5py.File(cfn, 'r') as cf:
                g = io.cads_group(cf, p, p['sid'])
                if g is None or 'part_556' not in g or 'part_553' not in g or 'part_559' not in g: continue
                A = np.asarray(g['image_affine'][()], float); shp = tuple(int(v) for v in g.attrs['image_shape'])
        except Exception: continue
        zax, sp = mc.orient(A, shp); thick = float(sp.max()); axial = int(zax == int(np.argmax(sp)) or thick < 1.25 * sp.min())
        cand.append(((1 - axial, thick), p, s, A, shp, zax, sp))
    cand.sort(key=lambda t: t[0]); fb = None
    for dz, p, s, A, shp, zax, sp in cand[:4]:
        with h5py.File(io.cads_file(p['h5']), 'r') as cf:
            g = io.cads_group(cf, p, p['sid'])
            H = g['part_556'][()] == 3
            L3 = g['part_553'][()]
            H |= (L3 >= 3) & (L3 <= 7); del L3
            zz = np.nonzero(np.moveaxis(H, zax, 0).any((1, 2)))[0]
            if not len(zz): continue
            z0, z1 = max(zz[0] - 2, 0), min(zz[-1] + 3, shp[zax])
            sl = [slice(None)] * 3; sl[zax] = slice(z0, z1); sl = tuple(sl)
            L9 = g['part_559'][sl]
        H = _lcc(H[sl]); vol = float(H.sum() * np.prod(sp) / 1000)
        if vol < 150: continue
        T = _lcc(np.isin(L9, (4, 7, 9)) | H); Hp = _lcc(H | (L9 == 7)); del L9
        lr = max((ax for ax in range(3) if ax != zax), key=lambda ax: abs(A[0, ax]) / np.linalg.norm(A[:3, ax]))
        hw, tw, hpw = (_widths(M, zax, lr, sp[lr]) for M in (H, T, Hp))
        hz = np.nonzero(np.moveaxis(H, zax, 0).any((1, 2)))[0]
        touch = int(hz[0] + z0 == 0 or hz[-1] + z0 == shp[zax] - 1)
        k = int(np.argmax(hw)); kp = int(np.argmax(hpw))
        r = dict(series=s['relative_file'], dz=dz[1], heart_ml=fin(vol, 0), z_touch=touch, heart_mm=fin(hw[k], 1), thorax_mm=fin(tw[k], 1),
                 same=fin(hw[k] / tw[k], 4) if tw[k] > 0 else None, max=fin(hw.max() / tw[hz].max(), 4) if tw[hz].max() > 0 else None,
                 same_p=fin(hpw[kp] / tw[kp], 4) if tw[kp] > 0 else None, thorax_max_mm=fin(tw[hz].max(), 1), heart_p_mm=fin(hpw[kp], 1),
                 lr_axis=int(lr))
        if not touch: return r
        fb = fb or r
    return fb or dict(fail='no_heart')


# training-split validated variant + additive calibration (training reports stating the value, leave-one-out medAE):
#   Evans 'same' S1: n 33, medAE 0.018 raw (bias -0.014), r 0.63 -> +0.014 (LOO medAE 0.015)
#   CTR 'same'     : n 116, medAE 0.025 raw (bias +0.022), r 0.84 -> -0.022 (LOO medAE 0.019)
#   Evans < EVANS_MIN (calibrated) is reported without a number ('< 0.25, ventricles not enlarged'): on unselected head CTs the
#   HU ventricle mask under-measures normal-calibre (slit) frontal horns (median 0.13), but low values are reliable negatives
#   (report positive rate by band on 35k head CTs: <0.15 2 %, 0.20-0.25 14 %, 0.25-0.30 29 %, 0.35-0.55 63 %, >0.55 84 %).  Values > EVANS_FAIL are failures (dropped).  A looser HU mask (< 18-25 HU, 1 mm smoothing) leaked into sulci /
#   caudate (medAE 0.16-0.19 on the positives) and was rejected.
EVANS_KEY = 'same'; CTR_KEY = 'same'; EVANS_ADD = 0.014; CTR_ADD = -0.022; EVANS_MIN = 0.25; EVANS_FAIL = 0.9


def evans_value(M):
    if not M or 'fail' in M: return None, None
    e = M.get('evans') or {}
    if e.get(EVANS_KEY) is None: e = M.get('evans_t') or {}
    if e.get(EVANS_KEY) is None: return None, None
    if not e.get('n_sl'): return None, None
    v = e[EVANS_KEY] + EVANS_ADD
    if not 0 < v <= EVANS_FAIL: return None, None
    return fin(v, 2), fin(e['horn_mm'], 0)


def ctr_value(Cr):
    if not Cr or 'fail' in Cr or Cr.get(CTR_KEY) is None: return None, None
    return fin(Cr[CTR_KEY] + CTR_ADD, 2), Cr['z_touch']


# ---------------------------------------------------------------- per case
def records(c, raw, cal):
    recs = []; ref = {}
    V = raw.get('vessel') or {}; O = raw.get('organ') or {}; M = raw.get('midline')
    for nm, cls, seg in (('ascending_aorta_mm', 55, 'asc'), ('aortic_root_mm', 43, 'root'), ('thoracic_aorta_max_mm', 678, 'thor'),
                         ('abdominal_aorta_mm', 1, 'abd'), ('pulmonary_trunk_mm', 549, 'pa')):
        ch = cal['vessel'].get(str(cls))
        if ch and V:
            y, x, used = vessel_value(V, seg, ch)
            if y is not None and (used == seg or seg == 'pa'): ref[nm] = y
    lv = O.get('liver'); sp_ = O.get('spleen')
    if lv and lv.get('cc_mm') is not None: ref['liver_cc_mm'] = fin(lv['cc_mm'] / np.exp(cal['organ']['279']['factor']), 1); ref['liver_max_mm'] = lv['max3d_mm']
    if sp_ and sp_.get('max3d_mm') is not None: ref['spleen_length_mm'] = fin(sp_['max3d_mm'] / np.exp(cal['organ']['633']['factor']), 1); ref['spleen_cc_mm'] = sp_['cc_mm']
    for s in ('R', 'L'):
        k = O.get('kidney_' + s)
        if k and k.get('max3d_mm') is not None: ref[f'kidney_{s}_length_mm'] = fin(k['max3d_mm'] / np.exp(cal['organ']['561']['factor']), 1)
    trunc = sorted(o for o, d in O.items() if d.get('z_touch'))
    if trunc: ref['organs_truncated_by_fov'] = trunc      # organ touches the z border of the volume: length is a lower bound
    if M and 'fail' not in M and 'midline' in cal:
        y, x, sgn, src = midline_value(M, cal['midline']['336'])
        if y is not None:
            ref['midline_shift_raw_mm'] = x
            if sgn is not None: ref['midline_shift_direction'] = 'right' if sgn > 0 else 'left'
    ev_, horn_ = evans_value(M)
    if ev_ is not None: ref['evans_index'] = ev_; ref['frontal_horn_mm'] = horn_
    Cr = raw.get('ctr'); ct_, ctt_ = ctr_value(Cr)
    if ct_ is not None:
        ref['cardiothoracic_ratio'] = ct_
        if ctt_: ref.setdefault('organs_truncated_by_fov', []).append('heart')
    for cls in c['classes']:
        r = dict(cls=cls)
        if cls in VESSEL:
            seg, cc = VESSEL[cls]; ch = cal['vessel'].get(cc); seg = (ch or {}).get('segment', seg)
            y, x, used = vessel_value(V, seg, ch) if (ch and V) else (None, None, None)
            r.update(kind='vessel_diameter', structure='pulmonary trunk' if seg == 'pa' else 'aorta', segment=seg, segment_used=used,
                     value_mm=y, raw_mm=x, segments_mm=vessel_segments(V, ch) if ch else None, method='centreline ' + '/'.join(ch['variant']) + f" {ch['cal']} {ch['factor']:+.3f}" if ch else None)
        elif cls in LIVER or cls in SPLEEN:
            for org, tab, fk, cc in (('liver', LIVER, 'cc_mm', '279'), ('spleen', SPLEEN, 'max3d_mm', '633')):
                if cls not in tab: continue
                o = O.get(org); rr = dict(r)
                rr.update(kind='organ_size', structure=org, measure='craniocaudal length' if org == 'liver' else 'length (3-D max extent)')
                if o and o.get(fk) is not None:
                    rr.update(value_mm=fin(o[fk] / np.exp(cal['organ'][cc]['factor']), 1), raw_mm=o[fk], cc_mm=o['cc_mm'], max3d_mm=o['max3d_mm'],
                              vol_ml=o['vol_ml'], truncated=o['z_touch'])
                else: rr.update(value_mm=None)
                recs.append(rr)
            continue
        elif cls in KIDNEY:
            r.update(kind='organ_size', structure='kidneys', measure='bipolar length (3-D max extent)')
            for s in ('R', 'L'):
                k = O.get('kidney_' + s)
                r[f'{s}_mm'] = fin(k['max3d_mm'] / np.exp(cal['organ']['561']['factor']), 1) if k and k.get('max3d_mm') is not None else None
                r[f'{s}_vol_ml'] = k['vol_ml'] if k else None
                r[f'{s}_truncated'] = k['z_touch'] if k else None
            ok = [v for v in (r['R_mm'], r['L_mm']) if v is not None]
            r['value_mm'] = min(ok) if ok else None; r['value_side'] = None if not ok else ('R' if r['value_mm'] == r['R_mm'] else 'L')
        elif cls in MIDLINE:
            ce = cal.get('midline', {}).get(str(cls)) or cal.get('midline', {}).get('336')
            y, x, sgn, src = midline_value(M, ce) if ce else (None, None, None, None)
            r.update(kind='midline_shift', measure='septum pellucidum offset from the mid-sagittal symmetry plane at the Monro level',
                     value_mm=y, raw_mm=x, source=src, direction=None if sgn is None else ('right' if sgn > 0 else 'left'))
        elif cls in EVANS:
            r.update(kind='evans_index', measure='max frontal-horn width / inner skull diameter on the same slice',
                     value=ev_, frontal_horn_mm=horn_, below_min=None if ev_ is None else bool(ev_ < EVANS_MIN))
        elif cls in CTR:
            r.update(kind='cardiothoracic_ratio', measure='max transverse heart diameter / inner thoracic diameter on the same axial slice',
                     value=ct_, truncated=ctt_, heart_mm=(Cr or {}).get('heart_mm'), thorax_mm=(Cr or {}).get('thorax_mm'))
        else: continue
        recs.append(r)
    return recs, ref


def work(arg):
    line, cal, ref_mid, only, roots = arg
    io.set_roots(*roots)
    t0 = time.time(); c = json.loads(line); res = dict(case=c['case'], report=c['report'], classes=c['classes'], raw={}, err={}, sec={})
    try:
        tasks = [('vessel', vessel_raw), ('organ', organ_raw)]
        if ref_mid or any(k in MIDLINE or k in EVANS for k in c['classes']): tasks.append(('midline', midline_raw))
        tasks.append(('ctr', ctr_raw))
        if only: tasks = [t for t in tasks if t[0] in only]
        for nm, fn in tasks:
            r, e, dt = timed(fn, LIMITS[nm], c); res['sec'][nm] = round(dt, 1)
            if e: res['err'][nm] = e
            if r is not None: res['raw'][nm] = r
        res['records'], res['ref'] = records(c, res['raw'], cal)
    except Exception as e:
        res['err']['case'] = repr(e)[:300]; res['tb'] = traceback.format_exc()[-600:]
    res['sec']['total'] = round(time.time() - t0, 1)
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cases', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--node-idx', type=int, default=0); ap.add_argument('--nnodes', type=int, default=1); ap.add_argument('--procs', type=int, default=8)
    ap.add_argument('--calib', required=True, help='data/measure/cads_calibration.json'); ap.add_argument('--ref-midline', type=int, default=1)
    ap.add_argument('--h5-root', required=True, help='CT-FORA H5 directory (<h5_id>.h5)'); ap.add_argument('--cads-root', required=True, help='CADS directory (cads_<h5_id>.h5)')
    ap.add_argument('--limit', type=int, default=0); ap.add_argument('--tasks', default='', help='comma list subset of vessel,organ,midline,ctr'); ap.add_argument('--deadline', type=float, default=0)
    a = ap.parse_args(); cal = json.load(open(a.calib)); io.set_roots(a.h5_root, a.cads_root)
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try: done.add(json.loads(l)['case'])
            except Exception: pass
    todo = []
    with open(a.cases) as f:
        for i, l in enumerate(f):
            if i % a.nnodes != a.node_idx: continue
            k = l.find('"case": "'); cid = l[k + 9:l.index('"', k + 9)]
            if cid not in done: todo.append(l)
    if a.limit: todo = todo[:a.limit]
    print(f'node {a.node_idx}/{a.nnodes}: todo {len(todo)} done {len(done)} procs {a.procs}', flush=True)
    import multiprocessing as mp
    t0 = time.time(); nerr = 0
    with open(a.out, 'a') as fo, mp.Pool(a.procs, maxtasksperchild=50) as pool:
        for i, r in enumerate(pool.imap_unordered(work, ((l, cal, a.ref_midline, set(a.tasks.split(',')) - {''}, (a.h5_root, a.cads_root)) for l in todo), chunksize=1)):
            fo.write(json.dumps(r, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else (float(o) if isinstance(o, np.floating) else int(o))) + '\n'); fo.flush()
            nerr += bool(r['err'])
            if i < 5 or i % 200 == 0:
                print(i, len(todo), r['case'], r['sec'], sorted(r['err']), f'{time.time() - t0:.0f}s', flush=True)
            if a.deadline and time.time() > a.deadline: print('deadline reached', flush=True); pool.terminate(); break
    print(f'finished {len(todo)} cases, with any error {nerr}, {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()

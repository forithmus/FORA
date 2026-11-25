"""Evidence images of a finding for the report writer (Methods, "Report generation", evidence images).

Axial, coronal and sagittal CT slices through the finding's evidence peak with the positive evidence map overlaid as in
the paper figures: the positive evidence of all views of one series summed on its token grid, 4 mm Gaussian smoothing,
token-centred trilinear sampling, scale = 99.5 % quantile of the smoothed map, red fill (35 % maximum opacity) and a contour
at 0.3 x scale. Radiological orientation (patient right on image left; anterior up on axial, superior up on coronal and
sagittal, anterior left on sagittal). Window by finding family (lung / bone / soft tissue). Each view is cropped around the
peak with a square field of view of max(120 mm, 1.5 x the in-plane extent of the connected evidence region at the peak),
letter-boxed to 448 x 448 and JPEG-encoded; further evidence sites get one zoomed axial view each (at most 3).

Volume axes = (S, R, A) as produced by ``fora.geometry.tiling.resample_series``, body-cropped; token (2, 16, 16) voxels
= 3 x 8 x 8 mm.
"""
import io
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from PIL import Image

PATCH = np.array([2, 16, 16]); SPACING = np.array([1.5, 0.5, 0.5]); SIZE = 448
WIN = {'lung': (-1200., 300.), 'bone': (-450., 1050.), 'soft': (-160., 240.)}
N_VIEWS = 6   # 3 views at the peak + up to 3 further sites


def series_map(grid, views):
    """sum the POSITIVE evidence (negative tokens dropped) of all views of one series on its token grid, then 4 mm Gaussian."""
    g = np.zeros(int(np.prod(grid)), np.float32)
    for zyx, ev in views:
        g[np.ravel_multi_index(tuple(zyx.T), grid)] += ev
    g = g.reshape(grid)
    return gaussian_filter(g, sigma=4. / (PATCH * SPACING), mode='constant')


def _plane(sm, fixed_axis, fixed_vox, shape):
    """sample the smoothed token map on one CT plane (token-centred coordinates)."""
    axes = [a for a in range(3) if a != fixed_axis]
    u = (np.arange(shape[axes[0]]) + .5) / PATCH[axes[0]] - .5; v = (np.arange(shape[axes[1]]) + .5) / PATCH[axes[1]] - .5
    U, V = np.meshgrid(u, v, indexing='ij'); f = np.full_like(U, (fixed_vox + .5) / PATCH[fixed_axis] - .5)
    c = [None, None, None]; c[fixed_axis] = f; c[axes[0]] = U; c[axes[1]] = V
    return map_coordinates(sm, c, order=1, mode='nearest').astype(np.float32)


def _overlay(ct, hm, lo, hi, scale, row_stretch):
    """CT + positive evidence: 35 % max opacity fill + solid 1-px contour at 0.3 x scale (lesion stays visible)."""
    from scipy.ndimage import binary_erosion, gaussian_filter as _gf
    hm = _gf(hm, sigma=(2.5 / row_stretch, 2.5))   # pixel-space smoothing: round outlines instead of token-grid blocks
    g = np.clip((ct - lo) / (hi - lo), 0, 1); v = np.clip(hm / scale, 0, 1); al = (v * .35)[..., None]
    rgb = np.stack([g] * 3, -1) * (1 - al) + np.array([1., .2, 0.]) * al
    m = v >= .3; edge = m & ~binary_erosion(m, iterations=1)
    rgb[edge] = np.array([1., .25, 0.])
    im = Image.fromarray((255 * rgb).astype(np.uint8))
    if row_stretch != 1: im = im.resize((im.width, int(round(im.height * row_stretch))), Image.BILINEAR)
    s_ = SIZE / max(im.width, im.height); im = im.resize((max(1, int(im.width * s_)), max(1, int(im.height * s_))), Image.BICUBIC)
    out = Image.new('RGB', (SIZE, SIZE)); out.paste(im, ((SIZE - im.width) // 2, (SIZE - im.height) // 2)); return out


def _crop(c, n, half):
    lo_ = max(0, int(round(c - half))); hi_ = min(n, int(round(c + half)))
    return slice(lo_, max(hi_, lo_ + 1))


def render_finding(vol, sm, peak_vox, family, extra_sites=(), allow=None):
    """vol (S,R,A) HU; sm smoothed positive token map of the same series; peak_vox (s,r,a) -> 3 + len(extra_sites) JPEGs:
    zoomed axial, coronal, sagittal at the main peak, then a zoomed axial through each further site (max 3; sites 2..4).
    allow: optional token-grid mask of the finding's anatomy; used when it contains evidence."""
    lo, hi = WIN['lung' if family == 'lung' else 'bone' if family == 'bones' else 'soft']
    if allow is not None and (sm * allow).max() > 0:   # evidence inside the finding's anatomy only
        sm = sm * allow
    s, r, a = [int(np.clip(p, 0, n - 1)) for p, n in zip(peak_vox, vol.shape)]
    nz = sm[sm > 0]; scale = float(np.quantile(nz, .995)) if len(nz) else 1.
    scale = scale or 1.
    from scipy.ndimage import label as _label
    pk_tok = tuple(int(np.clip(int(p // P), 0, n - 1)) for p, P, n in zip((s, r, a), PATCH, sm.shape))
    lab_, _n = _label(sm >= .3 * scale, structure=np.ones((3, 3, 3), bool))
    reg = np.argwhere(lab_ == lab_[pk_tok]) if lab_[pk_tok] > 0 else np.zeros((0, 3), int)   # connected evidence region around the PEAK only
    if len(reg): ext = (reg.max(0) - reg.min(0) + 1) * PATCH * SPACING      # mm
    else: ext = np.zeros(3)
    fov = lambda e1, e2: max(120., 1.5 * max(e1, e2))                       # mm, square field of view
    half = lambda f, sp: f / 2. / sp                                        # voxels
    f_ax = fov(ext[1], ext[2]); f_co = fov(ext[0], ext[1]); f_sa = fov(ext[0], ext[2])
    rs = _crop(r, vol.shape[1], half(f_ax, SPACING[1])); as_ = _crop(a, vol.shape[2], half(f_ax, SPACING[2]))
    ax_ct = vol[s][rs, as_].astype(np.float32); ax_hm = _plane(sm, 0, s, vol.shape)[rs, as_]
    ss = _crop(s, vol.shape[0], half(f_co, SPACING[0])); rs2 = _crop(r, vol.shape[1], half(f_co, SPACING[1]))
    co_ct = vol[ss, rs2, a].astype(np.float32); co_hm = _plane(sm, 2, a, vol.shape)[ss, rs2]
    ss3 = _crop(s, vol.shape[0], half(f_sa, SPACING[0])); as3 = _crop(a, vol.shape[2], half(f_sa, SPACING[2]))
    sa_ct = vol[ss3, r, as3].astype(np.float32); sa_hm = _plane(sm, 1, r, vol.shape)[ss3, as3]
    ims = (_overlay(ax_ct.T[::-1, ::-1], ax_hm.T[::-1, ::-1], lo, hi, scale, 1.),
           _overlay(co_ct[::-1, ::-1], co_hm[::-1, ::-1], lo, hi, scale, SPACING[0] / SPACING[2]),
           _overlay(sa_ct[::-1, ::-1], sa_hm[::-1, ::-1], lo, hi, scale, SPACING[0] / SPACING[2]))
    for sv in list(extra_sites)[:3]:   # zoomed axial through each further evidence site (sites 2..4)
        s2, r2, a2 = [int(np.clip(p, 0, n - 1)) for p, n in zip(sv, vol.shape)]
        rs5 = _crop(r2, vol.shape[1], half(f_ax, SPACING[1])); as5 = _crop(a2, vol.shape[2], half(f_ax, SPACING[2]))
        ims = ims + (_overlay(vol[s2][rs5, as5].astype(np.float32).T[::-1, ::-1], _plane(sm, 0, s2, vol.shape)[rs5, as5].T[::-1, ::-1], lo, hi, scale, 1.),)
    out = []
    for im in ims:
        b = io.BytesIO(); im.save(b, format='JPEG', quality=88); out.append(b.getvalue())
    return out


def family_allow(vinfo, k, sidx, keys, region_names):
    """Token-grid mask of series ``sidx``: cells whose region bits hit the finding family's regions (dilated by 1 token)."""
    from scipy.ndimage import binary_dilation
    rnames = list(region_names)
    fb = np.int64(0)
    for kk in keys:
        if kk in rnames:
            fb |= np.int64(1) << np.int64(rnames.index(kk))
    grid = tuple(vinfo[k]['grid'])
    gb = np.zeros(int(np.prod(grid)), np.int64)
    for j in range(len(vinfo)):
        if vinfo[j]['series'] == sidx:
            np.bitwise_or.at(gb, np.ravel_multi_index(tuple(vinfo[j]['zyx'].T), grid), vinfo[j]['bits'])
    return binary_dilation(((gb & fb) != 0).reshape(grid), iterations=1).astype(np.float32)


def study_finding_images(E, vinfo, foci, class_names, load_volume, region_names=None, log=print):
    """Images of every requested class of one study: list (per class) of N_VIEWS JPEG byte strings (b'' when absent).

    E [N, K] positive evidence, vinfo per-view geometry (``fora.mil.evidence.view_info``), foci = ``fora.mil.evidence.foci``
    records (or None), load_volume(series index) -> int16 (S, R, A) volume of that series (or raises). Volumes are loaded on
    first use and kept for the study."""
    from fora.regions import REGION_NAMES, family_of
    rnames = list(REGION_NAMES if region_names is None else region_names)
    offs_ = np.cumsum([0] + [len(v['zyx']) for v in vinfo]); smc = {}; amc = {}; vols = {}; out = []
    for q in range(E.shape[1]):
        ims = [b''] * N_VIEWS
        try:
            e_ = E[:, q]; gk = int(e_.argmax()); k = int(np.searchsorted(offs_, gk, 'right') - 1); sidx = vinfo[k]['series']
            if e_[gk] > 0 and sidx not in vols:
                try: vols[sidx] = load_volume(sidx)
                except Exception as ex: log(dict(ct_error=str(ex)[:200]))
            if e_[gk] > 0 and sidx in vols:
                key = (sidx, q)
                if key not in smc:
                    smc[key] = series_map(tuple(vinfo[k]['grid']), [(vinfo[j]['zyx'], E[offs_[j]:offs_[j + 1], q].astype(np.float32))
                                                                   for j in range(len(vinfo)) if vinfo[j]['series'] == sidx])
                fam, fk = family_of(class_names[q]); pv = vinfo[k]['zyx'][gk - offs_[k]] * PATCH + PATCH // 2
                xs = []; F_ = foci[q] if foci is not None else None
                if F_ is not None:   # further evidence sites (>= 5 % of the evidence) -> nearest token of this series
                    W_ = np.concatenate([vinfo[j]['world'] for j in range(len(vinfo)) if vinfo[j]['series'] == sidx])
                    Z_ = np.concatenate([vinfo[j]['zyx'] for j in range(len(vinfo)) if vinfo[j]['series'] == sidx])
                    for fi in range(1, min(4, len(F_['share']))):
                        if F_['share'][fi] < 0.05: continue
                        cand = Z_[int(np.argmin(((W_ - np.asarray(F_['ctr'][fi])[None]) ** 2).sum(1)))] * PATCH + PATCH // 2
                        # skip a site already visible in an earlier view (same slice +/- 10 mm, within the 120 mm field)
                        if any(abs(int(cand[0]) - int(p_[0])) * 1.5 < 10 and abs(int(cand[1]) - int(p_[1])) * .5 < 60
                               and abs(int(cand[2]) - int(p_[2])) * .5 < 60 for p_ in [pv] + xs): continue
                        xs.append(cand)
                allow = None
                if fk:
                    akey = (sidx, tuple(fk))
                    if akey not in amc:
                        amc[akey] = family_allow(vinfo, k, sidx, fk, rnames)
                    allow = amc[akey]
                ims = list(render_finding(vols[sidx], smc[key], pv, fam, xs, allow)); ims = ims + [b''] * (N_VIEWS - len(ims))
        except Exception as ex:
            log(dict(cls=str(class_names[q]), render_error=str(ex)[:200]))
        out.append(ims)
    return out

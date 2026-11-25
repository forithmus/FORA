"""Signed evidence, evidence maps and foci of the classifier (Methods, "Finding localization").

CT. For finding c and token t of a study bag (all tokens of all views, same pass as the classifier logit):
    e_t,c = softmax_t(a_t,c - [t outside route] softplus(r_c)) * l_t,c * s_c        (signed evidence; sum_t e_t,c + o_c = logit)
The positive part max(e, 0) is the evidence map. ``foci`` pools the tokens of all views on a token-sized world grid
(cell value = max over views), smooths it with a 4 mm Gaussian, restricts it to the finding's anatomical family
(``fora.regions.family_of``; only if the family holds evidence), keeps the smallest set of cells holding 50 % of the smoothed
mass (HDR50), splits it into 26-connected components and orders the foci by mass (up to 8 are stored; at most 4 are used
by the report writer) with the mass-weighted centroid as the focus position. Per focus it also records the dominant
regions, left share, organ-relative position, organ extent and involvement, distance to the organ (and lung) boundary,
and 16 evidence tokens spread over the foci (head input x, evidence / peak, focus id, offsets).

X-ray. The token evidence is referenced to normal tissue: for finding c the reference b_c is the token-count-weighted mean
token logit sL = l_t,c * s_c over 2,000 random training studies negative for c, and the film evidence is
    E_t,c = max(w_t,c (sL_t,c - b_c), 0)          (w = attention softmax over the study bag)
smoothed with a 4 mm Gaussian on the film grid (token size = 16 px x 430 mm / long side of the content box). Foci are the
8-connected components of the smoothed map >= 0.3 x its 99.5 % quantile; the focus of a film is the component with the
largest mass, located at its peak.
"""
from __future__ import annotations

import json

import h5py
import numpy as np
import torch

from fora.encoder.blocks import crop_layout
from fora.geometry.world import token_centres_mm, token_geometry
from fora.regions import REGION_NAMES, family_of

N_FOCI, N_TOKENS = 8, 16
BROAD = {"body", "body_minus_lungs", "peritoneum", "soft_tissue", "bones", "vessels", "pelvis", "chest_wall",
         "abdominal_wall", "neck"}
PAIRED = {"kidneys", "adrenals", "pleura", "lungs", "hila", "ureters", "breast", "axilla", "groin", "ribs", "orbit",
          "temporal_bone", "salivary", "shoulder", "pelvic_bones_hips", "neck_vessels", "chest_wall"}
FOCI_KEYS = ("tally", "cov", "gshare", "gshare80", "left80", "ncell", "restricted", "left_all", "n5", "n2", "share", "ntok",
             "ctr", "ext", "reg", "regf", "regall", "ml", "lbdist", "inv10", "left", "rel", "orgext", "inv", "bdist", "view",
             "spread", "neff", "tx", "tev", "tfo", "trel", "toff", "tgo", "tva")


# ------------------------------------------------------------------------------------------------------------ CT
def view_info(series_path, view: dict, bits: np.ndarray) -> dict:
    """Token geometry of one view (crop-major order): grid indices, world centres (S, R, A mm), token / native size."""
    with h5py.File(series_path, "r") as f:
        flat = f["flat_indices"][:]
        grid = json.loads(f["v00"].attrs["metadata_json"])["grid"]
        prov = json.loads(f.attrs["provenance_json"])
    zyx = np.stack(np.unravel_index(flat[crop_layout(flat, grid)[0]], grid), 1)
    _, _, _, tok, nat = token_geometry(prov)
    return dict(zyx=zyx, world=token_centres_mm(prov, zyx), tok=tok, nat=nat, grid=grid, series=view["series"],
                window=view["window"]["kind"], bits=bits.view(np.uint64).astype(np.int64))


@torch.no_grad()
def study_evidence(block, head, bag, vinfo, mean, rms, cl: torch.Tensor, crops: int, keep_signed: bool = False) -> dict:
    """One pass over a routed study bag for the requested classes ``cl``.

    Returns logits / probabilities, the attention-pooled 1048-d head input ``feat``, evidence-weighted region shares
    (``reg``, top-3 specific regions ``ri``/``fr``), left/right share, cranio-caudal position ``zfrac`` (0 inferior,
    1 superior within the view), the same summaries attention-weighted (``reg_a``, ``ri_a``/``fr_a``, ``side_a``,
    ``zf_a``), the positive-evidence mass relative to the attention mass ``evm``, the peak token (view, z, y, x), the
    positive evidence E [N, K] (float64, all tokens in bag order), optionally the signed evidence, and the head inputs
    XA [N, 1048] (float16, CPU) for the focus tokens."""
    dev = mean.device
    C = head.n_classes
    R = len(REGION_NAMES)
    allc = torch.arange(C, device=dev)
    spec = torch.tensor([n not in BROAD for n in REGION_NAMES], device=dev)
    pen = head.penalty()
    m_ = torch.full((C,), -1e30, device=dev)
    s_, al, az, aze, se = (torch.zeros(C, device=dev) for _ in range(5))
    ar, are = torch.zeros(C, R, device=dev), torch.zeros(C, R, device=dev)
    alr, alre = torch.zeros(C, 2, device=dev), torch.zeros(C, 2, device=dev)
    best = torch.full((C,), -1e30, device=dev)
    bpk = torch.zeros(C, 4, dtype=torch.long, device=dev)
    ax = None
    SQ, LQ, XQ = [], [], []
    for k, v in enumerate(vinfo):
        zyx = v["zyx"]
        bits = torch.from_numpy(v["bits"]).to(dev)
        mid = np.median(zyx[:, 1])
        z0, z1 = zyx[:, 0].min(), max(zyx[:, 0].max(), zyx[:, 0].min() + 1)
        lr_all = torch.from_numpy(np.stack([zyx[:, 1] < mid, zyx[:, 1] > mid], 1).astype(np.float32)).to(dev)
        zf_all = torch.from_numpy(((zyx[:, 0] - z0) / (z1 - z0)).astype(np.float32)).to(dev)
        pk_all = torch.from_numpy(np.concatenate([np.full((len(zyx), 1), k), zyx], 1)).to(dev)
        o = 0
        for y, mk, pf, _ in bag.routed_chunks(block, crops, views=[k]):
            x = torch.cat(((y - mean) / rms, pf.float()), 1)
            sc, lg = head._chunk_outputs(x, allc)
            n = len(sc)
            sc = sc.float() + (~mk).float() * (-pen)[None]
            lg = lg.float()
            SQ.append(sc[:, cl].clone())
            LQ.append(lg[:, cl].clone())
            XQ.append(x.half().cpu())
            mn = torch.maximum(m_, sc.max(0).values)
            sf = torch.exp(m_ - mn)
            e = torch.exp(sc - mn[None])
            oh = ((bits[o:o + n, None] >> torch.arange(R, device=dev)[None]) & 1).float()
            s_ = s_ * sf + e.sum(0)
            ax = (e.T @ x.float()) if ax is None else ax * sf[:, None] + e.T @ x.float()
            al = al * sf + (e * lg).sum(0)
            ar = ar * sf[:, None] + e.T @ oh
            alr = alr * sf[:, None] + e.T @ lr_all[o:o + n]
            az = az * sf + e.T @ zf_all[o:o + n]
            ee = e * lg.clamp_min(0.)
            se = se * sf + ee.sum(0)
            are = are * sf[:, None] + ee.T @ oh
            alre = alre * sf[:, None] + ee.T @ lr_all[o:o + n]
            aze = aze * sf + ee.T @ zf_all[o:o + n]
            m_ = mn
            ev = sc + torch.log(lg.clamp_min(1e-6))
            bv, bi = ev.max(0)
            upd = bv > best
            best = torch.where(upd, bv, best)
            bpk[upd] = pk_all[o + bi[upd]]
            o += n
        assert o == len(zyx)
    lgt = al / s_ * head.output_scale + head.output_bias
    reg_a = ar[cl] / s_[cl, None]
    side_a, zf_a = alr[cl] / s_[cl, None], az[cl] / s_[cl]
    ok = (se[cl] > 1e-12 * s_[cl])[:, None]
    den = se[cl].clamp_min(1e-30)
    reg = torch.where(ok, are[cl] / den[:, None], reg_a)            # evidence-weighted, attention-weighted if no evidence
    fr, ri = torch.topk(torch.where(spec[None], reg, torch.full_like(reg, -1.)), 3, 1)
    side = torch.where(ok, alre[cl] / den[:, None], side_a)
    zfrac = torch.where(ok[:, 0], aze[cl] / den, zf_a)
    fr_a, ri_a = torch.topk(torch.where(spec[None], reg_a, torch.full_like(reg_a, -1.)), 3, 1)
    S, L = torch.cat(SQ), torch.cat(LQ)
    signed = torch.softmax(S, 0) * L * head.output_scale[cl][None]
    return dict(logit=lgt[cl], prob=torch.sigmoid(lgt[cl]), feat=ax[cl] / s_[cl, None], reg=reg, fr=fr, ri=ri, side=side,
                zfrac=zfrac, peak=bpk[cl], reg_a=reg_a, fr_a=fr_a, ri_a=ri_a, side_a=side_a, zf_a=zf_a,
                evm=se[cl] / s_[cl], E=signed.clamp_min(0.).cpu().numpy().astype(np.float64),
                signed=signed.float().cpu().numpy() if keep_signed else None, XA=torch.cat(XQ))


def foci(E: np.ndarray, vinfo: list, class_names: list, XA: torch.Tensor, region_names=None) -> list:
    """Foci decomposition of the positive evidence map of every requested class (see the module docstring).
    ``region_names``: bit order of the token region bits (default ``fora.regions.REGION_NAMES``)."""
    from scipy.ndimage import gaussian_filter, label
    from scipy.spatial import cKDTree
    RN = list(REGION_NAMES if region_names is None else region_names)
    paired = np.array([n in PAIRED or n.startswith("lung_") for n in RN])
    lungs = RN.index("lungs")
    specific = np.array([n not in ("body", "body_minus_lungs") for n in RN])
    K = E.shape[1]
    offs = np.cumsum([0] + [len(v["zyx"]) for v in vinfo])
    W = np.concatenate([v["world"] for v in vinfo])
    tok = vinfo[0]["tok"]
    B = np.concatenate([v["bits"] for v in vinfo])
    VI = np.concatenate([np.full(len(v["zyx"]), k) for k, v in enumerate(vinfo)])
    left_tok = np.concatenate([v["zyx"][:, 1] < np.median(v["zyx"][:, 1]) for v in vinfo])
    Rb = np.arange(len(RN), dtype=np.int64)
    st = np.ones((3, 3, 3), bool)
    mids = [np.median(v["zyx"][:, 1]) for v in vinfo]
    cache, vcount = {}, {}

    def organ_geom(k, r, sd=-1):
        """Organ r in view k (paired organs: only side sd, 1 left / 0 right): 2-98 % bbox, boundary tree, mask, midline range."""
        if (k, r, sd) in cache:
            return cache[(k, r, sd)]
        v = vinfo[k]
        m = ((v["bits"] >> r) & 1).astype(bool)
        if sd >= 0:
            m &= ((v["zyx"][:, 1] < mids[k]) == bool(sd))
        if m.sum() < 4:
            cache[(k, r, sd)] = None
            return None
        P = v["world"][m]
        lo, hi = np.percentile(P, 2, 0), np.percentile(P, 98, 0)
        g = np.zeros(v["grid"], bool)
        g[tuple(v["zyx"][m].T)] = True
        zz = v["zyx"][m]
        bd = np.zeros(len(zz), bool)
        for ax_ in range(3):
            for sg in (-1, 1):
                q = zz.copy()
                q[:, ax_] += sg
                inb = np.all((q >= 0) & (q < np.array(v["grid"])), 1)
                nb = np.zeros(len(q), bool)
                nb[inb] = g[tuple(q[inb].T)]
                bd |= ~nb
        tree = cKDTree(P[bd]) if bd.any() else None
        dm = np.abs(v["zyx"][m, 1] - mids[k])
        cache[(k, r, sd)] = (lo, hi, tree, m, np.percentile(dm, 2), np.percentile(dm, 98))
        return cache[(k, r, sd)]

    def rsize(k):
        if k not in vcount:
            vcount[k] = np.array([int(((vinfo[k]["bits"] >> r) & 1).sum()) for r in range(len(RN))])
        return vcount[k]

    NF, FT = N_FOCI, N_TOKENS
    out = []
    for q in range(K):
        e = E[:, q]
        gp = float(e.max())
        R = dict(n5=0, n2=0, share=np.zeros(NF), ntok=np.zeros(NF), ctr=np.zeros((NF, 3)), ext=np.zeros((NF, 3)),
                 reg=np.full((NF, 3), -1), regf=np.zeros((NF, 3)), left=np.full(NF, np.nan),
                 regall=np.zeros((NF, len(RN)), np.float16), ml=np.full(NF, np.nan), lbdist=np.full(NF, np.nan),
                 inv10=np.full(NF, np.nan), rel=np.full((NF, 3), np.nan), orgext=np.full((NF, 3), np.nan),
                 inv=np.full(NF, np.nan), bdist=np.full(NF, np.nan), view=np.full(NF, -1), spread=np.zeros(3), neff=0.,
                 tx=np.zeros((FT, 1048), np.float16), tev=np.zeros(FT), tfo=np.full(FT, -1), trel=np.full((FT, 3), np.nan),
                 toff=np.zeros((FT, 3)), tgo=np.zeros((FT, 3)), tva=np.zeros(FT, bool), tally=np.zeros(len(RN), np.int16),
                 cov=np.zeros(len(RN), np.float16), gshare=np.zeros(len(RN), np.float16),
                 gshare80=np.zeros(len(RN), np.float16), left80=np.nan, ncell=0, restricted=0, left_all=np.nan)
        if gp <= 0:
            out.append(R)
            continue
        pos = e > 0
        w = e[pos]
        mu = (w[:, None] * W[pos]).sum(0) / w.sum()
        R["spread"] = np.sqrt((w[:, None] * (W[pos] - mu) ** 2).sum(0) / w.sum())
        R["neff"] = float(w.sum() ** 2 / (w ** 2).sum())
        # world token-cell grid over all tokens: cell value = max evidence over views, bits = OR over the cell's tokens
        cell = np.floor(W / tok[None]).astype(np.int64)
        o_ = np.lexsort((-e, cell[:, 2], cell[:, 1], cell[:, 0]))
        cs = cell[o_]
        starts = np.flatnonzero(np.r_[True, np.any(cs[1:] != cs[:-1], 1)])
        rep = o_[starts]
        cbits = np.bitwise_or.reduceat(B[o_], starts)
        uc = cs[starts]
        lo = uc.min(0)
        shp = tuple(uc.max(0) - lo + 1)
        I = tuple((uc - lo).T)
        G = np.zeros(shp, np.float32)
        G[I] = e[rep]
        HT = np.zeros(shp, bool)
        HT[I] = True
        sm = gaussian_filter(G, sigma=4. / tok, mode="constant")
        nzv = sm[sm > 0]
        if not len(nzv) or float(np.quantile(nzv, .995)) <= 0:
            out.append(R)
            continue
        fam = family_of(class_names[q])[1] or None
        allow = HT.copy()
        if fam:
            fb = np.int64(0)
            for kk in fam:
                if kk in RN:
                    fb |= np.int64(1) << np.int64(RN.index(kk))
            CB = np.zeros(shp, np.int64)
            CB[I] = cbits
            a2 = HT & ((CB & fb) != 0)
            if a2.any() and sm[a2].sum() > 0:
                allow = a2
                R["restricted"] = 1
        # HDR50: smallest set of allowed cells holding 50 % of the smoothed evidence mass
        vals_ = sm[allow]
        o2 = np.argsort(-vals_)
        cu = np.cumsum(vals_[o2]) / max(vals_.sum(), 1e-30)
        thr_ = vals_[o2[min(int(np.searchsorted(cu, .5)) + 1, len(o2)) - 1]]
        msk = allow & (sm >= thr_)
        thr80 = vals_[o2[min(int(np.searchsorted(cu, .8)) + 1, len(o2)) - 1]]
        m80 = (allow & (sm >= thr80))[I]
        oh80 = ((cbits[m80][:, None] >> Rb[None]) & 1).astype(np.float64)
        v80 = sm[I][m80]
        R["gshare80"] = ((v80[:, None] * oh80).sum(0) / max(v80.sum(), 1e-30)).astype(np.float16)
        R["left80"] = float((v80 * left_tok[rep[m80]]).sum() / max(v80.sum(), 1e-30))
        if not msk.any():
            msk = np.zeros(shp, bool)
            msk[np.unravel_index(int(np.argmax(sm * HT)), shp)] = True
        R["ncell"] = int(msk.sum())
        lab, nl = label(msk, structure=st)
        cid_all = lab[I] - 1
        val = sm[I]
        cm = np.bincount(cid_all[cid_all >= 0], weights=val[cid_all >= 0], minlength=nl)
        tot = cm.sum()
        order = np.argsort(-cm)
        R["n5"], R["n2"] = int((cm >= 0.05 * tot).sum()), int((cm >= 0.02 * tot).sum())
        fam_idx = [RN.index(k_) for k_ in (fam or []) if k_ in RN]
        for c in np.flatnonzero(cm >= 0.02 * tot):        # every site >= 2 % of the evidence, by its dominant region
            mc_ = cid_all == c
            sh = (val[mc_][:, None] * ((cbits[mc_][:, None] >> Rb[None]) & 1)).sum(0)
            if fam_idx:
                j_ = fam_idx[int(np.argmax(sh[fam_idx]))] if sh[fam_idx].max() > 0 else -1
            else:
                sh = sh.astype(float)
                sh[~specific] = -1
                j_ = int(np.argmax(sh))
            if j_ >= 0:
                R["tally"][j_] += 1
        oh_all = ((cbits[:, None] >> Rb[None]) & 1).astype(bool)
        inm = cid_all >= 0
        R["cov"] = (oh_all[inm].sum(0) / np.maximum(oh_all.sum(0), 1)).astype(np.float16)
        R["gshare"] = ((val[inm][:, None] * oh_all[inm]).sum(0) / max(val[inm].sum(), 1e-30)).astype(np.float16)
        R["left_all"] = float((val[inm] * left_tok[rep[inm]]).sum() / max(val[inm].sum(), 1e-30))
        gpk = int(e.argmax())
        toks = []
        for f, c in enumerate(order[:NF]):
            if cm[c] < 0.02 * tot and f > 0:
                break
            mc = cid_all == c
            ti, vv = rep[mc], val[mc]
            oo = np.argsort(-vv, kind="stable")
            ti, vv = ti[oo], vv[oo]
            P = W[ti]
            R["share"][f] = cm[c] / tot
            R["ntok"][f] = mc.sum()
            ctr = (vv[:, None] * P).sum(0) / vv.sum()
            R["ctr"][f] = ctr
            R["ext"][f] = P.max(0) - P.min(0) + tok
            rs = (vv[:, None] * ((B[ti][:, None] >> Rb[None]) & 1).astype(np.float64)).sum(0) / vv.sum()
            R["regall"][f] = rs
            rs = rs.copy()
            rs[~specific] = -1
            k = int(VI[ti[0]])
            tr = np.argsort(-(np.round(rs, 1) * 1e9 - rsize(k)))[:3]
            R["reg"][f], R["regf"][f] = tr, rs[tr]
            R["left"][f] = float((vv * left_tok[ti]).sum() / vv.sum())
            R["view"][f] = k
            r0 = int(tr[0])
            sd = int(R["left"][f] >= 0.5)
            og = organ_geom(k, r0, sd if paired[r0] else -1) if rs[r0] > 0 else None
            if og is not None:
                olo, ohi, tree, om, d0, d1 = og
                R["rel"][f] = np.clip((ctr - olo) / np.maximum(ohi - olo, 1e-3), 0, 1)
                R["orgext"][f] = ohi - olo
                R["inv"][f] = float((e[offs[k]:offs[k + 1]][om] >= 0.25 * gp).mean())
                R["inv10"][f] = float(R["cov"][r0])
                if tree is not None:
                    R["bdist"][f] = float(tree.query(P)[0].min())
                inv_ = VI[ti] == k
                cz = vinfo[k]["zyx"][ti[inv_] - offs[k]]
                dcol = float((vv[inv_] * np.abs(cz[:, 1] - mids[k])).sum() / max(vv[inv_].sum(), 1e-30))
                R["ml"][f] = float(np.clip((dcol - d0) / max(d1 - d0, 1e-3), 0, 1))
            if rs[lungs] > 0.5 or RN[r0].startswith("lung_"):
                lg_ = organ_geom(k, lungs, sd)
                if lg_ is not None and lg_[2] is not None:
                    R["lbdist"][f] = float(lg_[2].query(P)[0].min())
            toks.append((f, ti, vv, ctr, og))
        # 16 tokens over the evidence region: allocation by focus mass (>= 1 each), farthest-point sampling (sqrt value)
        sh = np.array([R["share"][f] for f, *_ in toks])
        alloc = np.maximum(1, np.floor(sh / sh.sum() * FT)).astype(int)
        while alloc.sum() > FT:
            alloc[np.argmax(alloc)] -= 1
        while alloc.sum() < FT:
            alloc[np.argmax(sh / alloc)] += 1
        pick = []
        for (f, ti, vv, ctr, og), na in zip(toks, alloc):
            P = W[ti]
            chosen = [0]
            dmin = np.linalg.norm(P - P[0], axis=1)
            while len(chosen) < min(na, len(ti)):
                j = int(np.argmax(dmin * np.sqrt(vv / vv[0])))
                if dmin[j] <= 0:
                    break
                chosen.append(j)
                dmin = np.minimum(dmin, np.linalg.norm(P - P[j], axis=1))
            for j in chosen:
                pick.append((f, int(ti[j]), float(vv[j]), ctr, og))
        pick = pick[:FT]
        if pick:
            R["tx"][:len(pick)] = XA[torch.tensor([t for _, t, _, _, _ in pick])].float().numpy()
        vmax = max(v_ for _, _, v_, _, _ in pick) if pick else 1.
        for j, (f, t, v_, ctr, og) in enumerate(pick):
            R["tev"][j], R["tfo"][j] = v_ / vmax, f
            R["toff"][j], R["tgo"][j], R["tva"][j] = W[t] - ctr, W[t] - W[gpk], True
            if og is not None:
                R["trel"][j] = np.clip((W[t] - og[0]) / np.maximum(og[1] - og[0], 1e-3), 0, 1)
        out.append(R)
    return out


# --------------------------------------------------------------------------------------------------------- X-ray
def xray_token_evidence(w: torch.Tensor, token_logits: torch.Tensor, output_scale: torch.Tensor, b_ref: torch.Tensor):
    """Referenced positive evidence [N, K]: max(w (l * s - b), 0) for attention w, token logits l, scale s, reference b."""
    return torch.clamp(w * (token_logits * output_scale[None] - b_ref[None]), min=0)


def xray_film_focus(E: np.ndarray, content_box_rc, quantile: float = .995, rel: float = 0.3) -> dict:
    """Smoothed film evidence map and its top focus. E [gy, gx] positive evidence on the film grid; content box (r0, r1,
    c0, c1) in pixels. Returns the map and, if covered, the peak token / pixel, top-focus pixel box, focus count and the
    area share of the 80 % highest-density region."""
    from scipy.ndimage import gaussian_filter, label
    r0, r1, c0, c1 = content_box_rc
    gy, gx = E.shape
    sg = 4.0 / (16 * 430.0 / max(r1 - r0, c1 - c0))
    rec = dict(evidence_mass=float(E.sum()), covered=False, sigma_tokens=float(sg))
    if E.max() <= 0:
        return rec
    sm = gaussian_filter(E, sigma=sg, mode="constant")
    nz = sm[sm > 0]
    q = float(np.quantile(nz, quantile))
    if q <= 0:
        return rec
    lab, nl = label(sm >= rel * q, structure=np.ones((3, 3), bool))
    cm = np.bincount(lab.ravel(), sm.ravel(), nl + 1)[1:]
    top = int(np.argmax(cm)) + 1
    pr, pc = divmod(int(np.argmax(np.where(lab == top, sm, -1))), gx)
    v = np.sort(sm.ravel())[::-1]
    cu = np.cumsum(v) / v.sum()
    thr80 = v[min(int(np.searchsorted(cu, .8)) + 1, len(v)) - 1]
    ys, xs = np.nonzero(lab == top)
    rec.update(covered=True, map=sm, n_foci=int(nl), n_foci_ge5pct=int((cm >= .05 * cm.sum()).sum()),
               peak_token_rc=[int(pr), int(pc)], peak_pixel_rc=[int(r0 + 16 * pr + 8), int(c0 + 16 * pc + 8)],
               top_focus_pixel_box_rc=[int(r0 + 16 * ys.min()), int(r0 + 16 * (ys.max() + 1)),
                                       int(c0 + 16 * xs.min()), int(c0 + 16 * (xs.max() + 1))],
               hdr80_area_share=float((sm >= thr80).mean()))
    return rec

"""Lesion caliper networks and their geometry (paper: Methods, "Measurements").

Segment-then-measure: the size is measured geometrically on a predicted mask (geometry.py, RECIST 1.1), never regressed.

3-D head (`CaliperHead`, frozen at the axial-2-D stage): on the frozen block-24 tokens of a lesion-centred view (16 x 24 x 24 tokens of
size tok = FOV / 24) a token trunk (token projection + position MLP + 2 residual 3-D conv blocks + cross-attention to the 125 coarse
MIL tokens of the candidate's 5 x 5 x 5 block + embeddings of the trunk class, finding class, measurement type and HU group, + the
frozen-MIL class evidence of the block) predicts a token mask logit, upsampled to the refine grid RG = (32, 96, 96) voxels (spacing
tok/2, tok/4, tok/4 mm) and refined by a light HU-guided residual 3-D conv net (input: upsampled logit, 8 token-feature channels,
3 HU windows, + 4 evidence channels and the raw HU in the class window).

Axial-2-D stage (`Dec2D`): the axial slice where the 3-D mask cross-section is largest (the RECIST slice) is encoded by a frozen 2-D
DINOv2-L (3 windows, 448 px -> 32 x 32 tokens over the view's field of view) and a small decoder predicts a 2-D mask on the 96 x 96
refine grid (pixel = tok / 4 mm) as a zero-initialised residual on the 3-D logit slice.  Inputs to the decoder: 2-D DINO tokens, the
3-D logit slice, 3 HU windows, 4 MIL-evidence channels.  Measurement on the 2-D mask: long axis (max Feret), short axis (extent
perpendicular to it), max inscribed diameter.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage as ndi

from . import encoder as FD
from . import geometry as CG

RG = (32, 96, 96)
HUWIN = ((-1000., 200.), (-160., 240.), (-1000., 1000.))
SP = FD.CACHE_SP
TRUNK = {0: 0, 1: 1, 2: 2, 3: 2, 4: 0, 5: 2, 6: 2, 7: 2}   # HU group -> trunk class embedding (0 nodule / 1 cyst / 2 lymph node)
FM = {0: 1., 1: 1.5, 2: 1., 3: 1.5}   # FOV multiplier per measurement type (lesion, thickness, vessel, lumen)

# clinical windows (lo, hi HU): lung L-600 W1500 | mediastinum/abdomen L40 W400 | liver L80 W200 | brain L40 W80 |
# blood/subdural L60 W160 | bone L400 W1800 | stone L400 W1200 | pancreas L40 W260 | angio L150 W500 | metal
W_LUNG, W_SOFT, W_LIVER, W_BRAIN, W_BLOOD, W_BONE, W_STONE, W_PANC, W_ANGIO, W_METAL = (-1350., 150.), (-160., 240.), (-20., 180.), (0., 80.), (-20., 140.), (-500., 1300.), (-200., 1000.), (-90., 170.), (-100., 400.), (-1000., 3000.)
GRP_WIN = {0: W_LUNG, 1: W_SOFT, 2: W_SOFT, 3: W_STONE, 4: W_LUNG, 5: W_SOFT, 6: W_BLOOD, 7: W_SOFT}   # lung fluid soft dense air vessel brain fat
CLS_WIN = {**{c: W_LUNG for c in (158, 191, 254, 374, 376, 473, 544, 546, 551, 666, 105, 341, 476, 532, 533, 535, 537, 692)},
           **{c: W_LIVER for c in (604, 269, 272, 370)}, **{c: W_BRAIN for c in (344, 335, 339, 49)}, **{c: W_BLOOD for c in (328, 259)},
           590: W_ANGIO, 458: W_PANC, 403: W_METAL, **{c: W_BONE for c in (82, 441, 442, 467)},
           **{c: W_STONE for c in (78, 552, 603, 639, 715, 721)}, 104: W_SOFT}


# ----------------------------------------------------------------------------------------------------------- 3-D head
class FineHead(nn.Module):
    """token trunk (state-dict key prefix `trunk.`)."""
    def __init__(s, h=128):
        super().__init__()
        s.inp = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, h)); s.pos = nn.Sequential(nn.Linear(4, h), nn.GELU(), nn.Linear(h, h))
        s.cemb = nn.Embedding(3, h)
        s.convs = nn.ModuleList([nn.Sequential(nn.Conv3d(h, h, 3, padding=1), nn.GroupNorm(8, h), nn.GELU(), nn.Conv3d(h, h, 3, padding=1), nn.GroupNorm(8, h)) for _ in range(2)])
        s.cin = nn.Sequential(nn.LayerNorm(1048), nn.Linear(1048, h)); s.cpos = nn.Linear(4, h)
        s.xattn = nn.MultiheadAttention(h, 4, batch_first=True); s.xnorm = nn.LayerNorm(h)
        s.out = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))


class CaliperHead(nn.Module):
    """Class-conditioned 3-D caliper head with MIL-evidence inputs; forward(b) -> (token logit [B, 16*24*24], refined logit [B, *RG])."""
    def __init__(s, nk, h=128, fh=8):
        super().__init__()
        s.trunk = FineHead(h); s.proj = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, fh))
        s.ref = nn.Sequential(nn.Conv3d(1 + fh + 3, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.GELU(), nn.Conv3d(16, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.GELU(),
                              nn.Conv3d(16, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.GELU(), nn.Conv3d(16, 1, 3, padding=1))
        nn.init.zeros_(s.ref[-1].weight); nn.init.zeros_(s.ref[-1].bias)
        s.register_buffer('pca_mean', torch.zeros(1024)); s.register_buffer('pca_P', torch.eye(1024)[:, :64].clone())
        s.kemb = nn.Embedding(nk, 128); s.temb = nn.Embedding(4, 128); s.gemb = nn.Embedding(8, 128)
        for e in (s.kemb, s.temb, s.gemb): nn.init.zeros_(e.weight)
        s.evt = nn.Linear(4, 128); s.evk = nn.Linear(4, 128); s.ref_ev = nn.Conv3d(5, 16, 3, padding=1)
        for m_ in (s.evt, s.evk, s.ref_ev): nn.init.zeros_(m_.weight); nn.init.zeros_(m_.bias)

    def forward(s, b):
        t = s.trunk; f, cent, tok, fov = b['f'], b['tokc'], b['tok'], b['fov']
        B, N, _ = f.shape; g = torch.cat([cent / fov[:, None, None], (tok / 10.)[:, None, None].expand(B, N, 1)], -1)
        h = t.inp(f) + t.pos(g) + (t.cemb(b['ct']) + s.kemb(b['k'].clamp_min(0)) * (b['k'] >= 0).float()[:, None] + s.temb(b['mt']) + s.gemb(b['c']))[:, None]
        h = h + s.evt(b['evt'])
        v = h.transpose(1, 2).reshape(B, -1, *FD.TOK)
        for blk in t.convs: v = F.gelu(v + blk(v))
        h = v.reshape(B, -1, N).transpose(1, 2)
        if b['cx'] is not None:
            kv = t.cin(b['cx'].float()) + t.cpos(torch.cat([b['coff'] / fov[:, None, None], torch.ones_like(b['coff'][..., :1])], -1))
            kv = kv + s.evk(b['evk'])
            h = h + t.xattn(t.xnorm(h), kv, kv, key_padding_mask=~b['cmask'], need_weights=False)[0]
        lt = t.out(h)[..., 0]
        up = F.interpolate(lt.view(B, 1, *FD.TOK), size=RG, mode='trilinear', align_corners=False)
        hf = F.interpolate(s.proj(h).transpose(1, 2).reshape(B, -1, *FD.TOK), size=RG, mode='trilinear', align_corners=False)
        hu = torch.stack([((b['hu'].clamp(lo, hi) - lo) / (hi - lo)) * 2 - 1 for lo, hi in HUWIN], 1)
        r0 = s.ref[0](torch.cat([up, hf, hu], 1)) + s.ref_ev(torch.cat([b['evr'], b['huc'][:, None]], 1))
        return lt, (up + s.ref[1:](r0))[:, 0]


# ----------------------------------------------------------------------------------------------------------- view geometry
def view_coords(tok, th, sh, dev):
    """Cache-frame mm (rel. to the crop centre voxel) of the DINO input voxels, token centres and refine-grid centres for one view."""
    D, H, W = FD.IN_VOX; c, s = math.cos(th), math.sin(th); R = torch.tensor([[c, -s], [s, c]], device=dev, dtype=torch.float32)
    sh = torch.as_tensor(sh, device=dev, dtype=torch.float32)
    def grid(nz, ny, nx, dz, dy):
        z = (torch.arange(nz, device=dev, dtype=torch.float32) - (nz - 1) / 2) * dz
        y = (torch.arange(ny, device=dev, dtype=torch.float32) - (ny - 1) / 2) * dy; x = (torch.arange(nx, device=dev, dtype=torch.float32) - (nx - 1) / 2) * dy
        Y, X = torch.meshgrid(y, x, indexing='ij'); p = torch.stack([Y, X], -1) @ R.T
        return torch.stack([(z[:, None, None] + sh[0]).expand(nz, ny, nx), (p[None, ..., 0] + sh[1]).expand(nz, ny, nx), (p[None, ..., 1] + sh[2]).expand(nz, ny, nx)], -1)
    return grid(D, H, W, tok / 2, tok / 16), grid(*FD.TOK, tok, tok), grid(*RG, tok / 2, tok / 4)


def sample_crop(crop, origin, stride, cvox, mm, fill=-1000.):
    """trilinear sample of the crop (HU) at cache mm positions [..., 3] (rel. to cvox)."""
    sz = torch.tensor(crop.shape, device=crop.device, dtype=torch.float32)
    k = [(cvox[i] + mm[..., i] / SP[i] - origin[i]) / stride[i] for i in range(3)]
    g = torch.stack([k[2] / (sz[2] - 1) * 2 - 1, k[1] / (sz[1] - 1) * 2 - 1, k[0] / (sz[0] - 1) * 2 - 1], -1)[None]
    return F.grid_sample((crop.float() - fill)[None, None], g, mode='bilinear', padding_mode='zeros', align_corners=True)[0, 0] + fill


def to_view_grid(mm, tok, th, sh, shape):
    """cache mm [..., 3] -> normalised grid_sample coords of a view grid of `shape` with spacing (tok/2, tok*24/shape[1], ...)."""
    c, s = math.cos(th), math.sin(th); q = mm - torch.as_tensor(sh, device=mm.device, dtype=torch.float32)
    y = c * q[..., 1] + s * q[..., 2]; x = -s * q[..., 1] + c * q[..., 2]   # R^T (a1, a2)
    dz = tok * FD.TOK[0] / shape[0]; dy = tok * FD.TOK[1] / shape[1]
    iz = q[..., 0] / dz + (shape[0] - 1) / 2; iy = y / dy + (shape[1] - 1) / 2; ix = x / dy + (shape[2] - 1) / 2
    return torch.stack([ix / (shape[2] - 1) * 2 - 1, iy / (shape[1] - 1) * 2 - 1, iz / (shape[0] - 1) * 2 - 1], -1)


def crop_grid_prob(lr_j, it, b, j, dev):
    """Refined probability resampled onto the crop grid (cache frame, native) within the view's FOV box -> (prob, crop index offset)."""
    tok = float(b['tok'][j]); cvox = b['cvox'][j]; half = np.array([tok * FD.TOK[0] / 2, tok * FD.TOK[1] / 2, tok * FD.TOK[2] / 2])
    o, st, shp = it['origin'], it['stride'].astype(np.float64), np.array(it['shape'])
    lo = np.maximum(np.floor((cvox - half / SP - o) / st).astype(int), 0); hi = np.minimum(np.ceil((cvox + half / SP - o) / st).astype(int) + 1, shp)
    idx = [torch.arange(lo[i], hi[i], device=dev, dtype=torch.float32) for i in range(3)]
    Z, Y, X = torch.meshgrid(*idx, indexing='ij'); mm = torch.stack([(o[i] + [Z, Y, X][i] * st[i] - cvox[i]) * SP[i] for i in range(3)], -1)
    g = to_view_grid(mm, tok, b['th'][j], b['sh'][j], RG)
    p = F.grid_sample(torch.sigmoid(lr_j.float())[None, None], g[None], mode='bilinear', padding_mode='zeros', align_corners=True)[0, 0]
    return p.cpu().numpy(), lo


def measure_crop(prob, lo, it, is_node, centre_vox):
    """component pick + RECIST caliper on the crop-grid probability; endpoints in crop index and cache-frame voxels."""
    st = it['stride'].astype(np.float64); sp = SP * st
    cidx = (np.asarray(centre_vox) - it['origin']) / st - lo
    C = CG.pick_component(prob, cidx, sp)
    if not C.any(): return None, C
    r = CG.caliper3d(C, sp, is_node=is_node)
    for k in ('long_vox', 'short_vox'):
        ci = np.array(r[k]) + lo[None]; r[k.replace('_vox', '_crop_idx')] = ci.tolist()
        r[k.replace('_vox', '_cache_vox')] = (it['origin'][None] + ci * st[None]).tolist()
        r.pop(k)
    zz, yy, xx = np.nonzero(C); r['centroid_cache_vox'] = (it['origin'] + (np.array([zz.mean(), yy.mean(), xx.mean()]) + lo) * st).tolist()
    r['touches_fov_border'] = bool(C[0].any() or C[-1].any() or C[:, 0].any() or C[:, -1].any() or C[:, :, 0].any() or C[:, :, -1].any())
    return r, C


def ins_diam(Cm, sp):
    """max inscribed diameter (mm) on axial slices of a boolean component: 2 x max in-plane EDT - pixel (pixel-centre convention)."""
    if Cm is None or not Cm.any(): return np.nan, -1, None
    zz, yy, xx = np.nonzero(Cm); lo_ = np.array([zz.min(), yy.min(), xx.min()]); hi_ = np.array([zz.max(), yy.max(), xx.max()]) + 1
    sub = np.pad(Cm[lo_[0]:hi_[0], lo_[1]:hi_[1], lo_[2]:hi_[2]], ((0, 0), (1, 1), (1, 1)))
    d = ndi.distance_transform_edt(sub, sampling=(1e4, sp[1], sp[2])); d[d > 5e3] = 0
    k = np.unravel_index(int(np.argmax(d)), d.shape); r = float(d[k])
    return max(2 * r - float(sp[1]), float(sp[1])), int(k[0] + lo_[0]), [int(k[0] + lo_[0]), int(k[1] - 1 + lo_[1]), int(k[2] - 1 + lo_[2])]


# ----------------------------------------------------------------------------------------------------------- axial-2-D stage
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
SOFT = (-160., 240.); LUNG = (-1350., 150.)


class Dec2D(nn.Module):
    def __init__(s, d=1024, c=128):
        super().__init__()
        s.proj = nn.Sequential(nn.Conv2d(d, c, 1), nn.GroupNorm(8, c), nn.GELU())
        s.body = nn.Sequential(nn.Conv2d(c + 8, c, 3, padding=1), nn.GroupNorm(8, c), nn.GELU(), nn.Conv2d(c, c, 3, padding=1), nn.GroupNorm(8, c), nn.GELU(),
                               nn.Conv2d(c, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(), nn.Conv2d(64, 1, 3, padding=1))
        nn.init.zeros_(s.body[-1].weight); nn.init.zeros_(s.body[-1].bias)

    def forward(s, tok, lt3, hu3, ev):
        """tok (B,1024,32,32) DINOv2 patch tokens; lt3 (B,96,96) 3-D logit slice; hu3 (B,3,96,96) windows in [-1,1]; ev (B,4,96,96)."""
        h = F.interpolate(s.proj(tok), size=lt3.shape[-2:], mode='bilinear', align_corners=False)
        return lt3 + s.body(torch.cat([h, lt3[:, None], hu3, ev], 1))[:, 0]


def win(x, lo, hi): return ((x.clamp(lo, hi) - lo) / (hi - lo))


def dino2d(enc2d, xs, wins):
    """xs (B,384,384) raw HU slices, wins list of (lo,hi) class windows -> (B,1024,32,32) normalised patch tokens."""
    B = xs.shape[0]; dev = xs.device
    cw = torch.stack([win(xs[j], *wins[j]) for j in range(B)])
    img = torch.stack([cw, win(xs, *SOFT), win(xs, *LUNG)], 1)
    img = F.interpolate(img, size=(448, 448), mode='bilinear', align_corners=False)
    img = (img - MEAN.to(dev)) / STD.to(dev)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=dev.type == 'cuda'):
        o = enc2d(pixel_values=img).last_hidden_state[:, 1:].float()
    return F.normalize(o, dim=-1).transpose(1, 2).reshape(B, -1, 32, 32)


def pick_slice(m3, max_off=8):
    """m3 (B,Z,96,96) 3-D mask probabilities -> index of the axial slice with the largest cross-section near the crop centre."""
    Z = m3.shape[1]; c = Z // 2; area = (m3 > .5).float().sum((2, 3))
    zz = torch.arange(Z, device=m3.device)[None]; area = torch.where((zz - c).abs() <= max_off, area, torch.full_like(area, -1.))
    z = area.argmax(1); return torch.where(area.max(1).values > 0, z, torch.full_like(z, c))


def measure2d(p, px, centre_r=0.3):
    """exact 2-D measurement of a probability map (96x96 numpy) -> dict(long, short, ins, u, n, comp) or None.
    component = the one containing the most probability mass within centre_r of the centre (else the largest)."""
    hard = p > .5
    if hard.sum() < 2: return None
    lab, nl = ndi.label(hard)
    H, W = p.shape; yy, xx = np.mgrid[:H, :W]; cen = ((yy - H / 2 + .5) ** 2 + (xx - W / 2 + .5) ** 2) <= (centre_r * H) ** 2
    mass = ndi.sum_labels(p * cen, lab, range(1, nl + 1))
    k = int(np.argmax(mass)) + 1 if mass.max() > 0 else int(np.argmax(ndi.sum_labels(hard, lab, range(1, nl + 1)))) + 1
    C = lab == k; pts = np.argwhere(C).astype(np.float64)
    if len(pts) < 2: return None
    try:
        from scipy.spatial import ConvexHull
        hp = pts[ConvexHull(pts).vertices] if len(pts) >= 4 else pts
    except Exception: hp = pts
    d2 = ((hp[:, None] - hp[None]) ** 2).sum(-1); i, j = np.unravel_index(int(np.argmax(d2)), d2.shape)
    L = math.sqrt(d2[i, j]) * px + px; u = (hp[j] - hp[i]) / max(np.linalg.norm(hp[j] - hp[i]), 1e-9); n = np.array([-u[1], u[0]])
    pr = pts @ n; S = (pr.max() - pr.min()) * px + px
    edt = ndi.distance_transform_edt(np.pad(C, 1))[1:-1, 1:-1]; ins = max(2 * float(edt.max()) * px - px, px)
    return dict(long=L, short=S, ins=ins, u=u, n=n, comp=C, ends=(hp[i], hp[j]))


def soft_len(p, comp, direction, px):
    """soft 4-sigma extent of the mask probability along `direction` inside the dilated exact component (torch, differentiable)."""
    reg = torch.from_numpy(ndi.binary_dilation(comp, iterations=2).astype(np.float32)).to(p.device)
    ms = p * reg; W = ms.sum().clamp_min(1e-3); H, Wd = p.shape
    yy, xx = torch.meshgrid(torch.arange(H, device=p.device, dtype=torch.float32) * px, torch.arange(Wd, device=p.device, dtype=torch.float32) * px, indexing='ij')
    pr = yy * float(direction[0]) + xx * float(direction[1]); mu = (ms * pr).sum() / W
    return 4 * torch.sqrt(((ms * (pr - mu) ** 2).sum() / W).clamp_min(1e-6))


def chan_vese(p, I, roi):
    """region term: p (H,W) prob, I (C,H,W) or (H,W) image/features, roi (H,W) float -> mean squared deviation from the in/out means."""
    if I.dim() == 2: I = I[None]
    w1 = p * roi; w0 = (1 - p) * roi
    c1 = (I * w1).sum((1, 2)) / w1.sum().clamp_min(1e-3); c0 = (I * w0).sum((1, 2)) / w0.sum().clamp_min(1e-3)
    e = (w1 * ((I - c1[:, None, None]) ** 2).sum(0) + w0 * ((I - c0[:, None, None]) ** 2).sum(0)).sum() / roi.sum().clamp_min(1.)
    return e


def roi_of(p, r_dil=12):
    """ring ROI: (hard mask U centre disk) dilated by r_dil px (detached)."""
    H, W = p.shape; hard = (p.detach() > .5).cpu().numpy()
    yy, xx = np.mgrid[:H, :W]; disk = ((yy - H / 2 + .5) ** 2 + (xx - W / 2 + .5) ** 2) <= 36
    return torch.from_numpy(ndi.binary_dilation(hard | disk, iterations=r_dil).astype(np.float32)).to(p.device)

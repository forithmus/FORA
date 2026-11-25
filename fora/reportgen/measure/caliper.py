"""Lesion caliper: inference and training of the axial-2-D decoder (paper: Methods, "Measurements").

One run handles one finding group (air, cyst, node, nodule, solid, stone, thickness; data/measure/caliper_groups.json); the (class,
type) pairs of the other groups and of the CADS-measured classes are passed with --exclude.

Inputs: candidate table (--cand npz: per candidate t, rank, row, case, view, peak, fov_size_mm, tsplit, cbf / cbj = index of its coarse
5x5x5 MIL token block in cb_files, cb_files) and target table (--targets npz: per finding key, cls, mt, grp, kidx, bits, is_node,
ycmp, lat, y_long / y_short / eps_long / eps_short (report sizes, training only), keep), see tables.py.
Per candidate: lesion-centred crop (crops.py), frozen 3-D ViT-L block-24 tokens of a view of FOV = clip(3 x size prior x type
multiplier, 32, 240) mm, the 3-D head (frozen), frozen-MIL class evidence of the candidate's token block, the axial-2-D decoder.

infer: deterministic view (zoom 1, no rotation); crop-grid caliper on the 3-D mask; small lesions (long < 10 mm) re-measured from a
zoomed view (FOV2 = clip(2.5 x long, 12, 24) mm) centred on the mask centroid; a mask touching the view border is re-run at 1.5x / 2x
FOV centred on the mask; the 2-D axial measurement replaces the 3-D one (and the inscribed diameter for thickness / lumen types).
Output OUT/cal_r<rank>_c<chunk>.npz rows: t, prank, long, short, ins, empty, ... (mm, before the per-class scale calibration).
train: the 3-D head and block 24 are frozen; only the 2-D decoder is trained (AdamW, one-cycle LR) with report-size supervision on
the most report-consistent candidate of each finding (straight-through soft axis lengths), a size-prior hinge on the other
candidates, Chan-Vese terms on HU and on DINOv2 features inside a ring ROI, total variation, a decaying anchor to the 3-D slice
mask, and the report content / shape / multiplicity terms of --info.
Distributed: torchrun (RANK / WORLD_SIZE / LOCAL_RANK); inference shards cases by rank, training all-reduces the gradients.

usage:
  python -m fora.reportgen.measure.caliper infer --ckpt HEAD.pt --cand cand.npz --targets targets.npz --exclude "$EXCL" --out OUT \\
      --cases cases.jsonl --h5-root H5 --regions-dir REG --encoder-ckpt ENC.pt --mil-ckpt MIL.pt --mil-config MIL.json \\
      --normalization norm.pt --dino2d DINOV2_L_DIR [--k 1]
  python -m fora.reportgen.measure.caliper train --init INIT_HEAD.pt --info info.npz --ln-short-axis ... (same inputs, --k 8 --epochs 2)
"""
import argparse
import collections
import glob
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from . import crops as CR


def bounded(pool, fn, jobs, ahead):
    q = collections.deque(); it = iter(jobs)
    for _ in range(ahead):
        try: q.append(pool.apply_async(fn, (next(it),)))
        except StopIteration: break
    while q:
        r = q.popleft().get()
        try: q.append(pool.apply_async(fn, (next(it),)))
        except StopIteration: pass
        yield r


def npz_memmap(path, key):
    """Read-only memmap of one array stored UNCOMPRESSED inside an .npz (np.savez)."""
    import zipfile, numpy.lib.format as fmt
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(key + '.npy'); assert info.compress_type == 0
        with open(path, 'rb') as f:
            f.seek(info.header_offset); h = f.read(30); n1 = int.from_bytes(h[26:28], 'little'); n2 = int.from_bytes(h[28:30], 'little')
            start = info.header_offset + 30 + n1 + n2; f.seek(start); ver = fmt.read_magic(f)
            shape, fo, dt = (fmt.read_array_header_1_0(f) if ver == (1, 0) else fmt.read_array_header_2_0(f)); off = f.tell()
    return np.memmap(path, dtype=dt, mode='r', offset=off, shape=shape, order='F' if fo else 'C')


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('infer', 'train')); ap.add_argument('--out', required=True)
    ap.add_argument('--ckpt', default=None, help='infer: caliper checkpoint (head, dec2d, b24, nk)')
    ap.add_argument('--init', default=None, help='train: checkpoint the 3-D head, block 24 and 2-D decoder start from')
    ap.add_argument('--cand', required=True); ap.add_argument('--targets', required=True)
    ap.add_argument('--k', type=int, default=8, help='candidates per finding (rank < k)'); ap.add_argument('--exclude', default='', help='cls:mt,... pairs not trained / inferred')
    ap.add_argument('--split', default='all', help='infer: candidate tsplit to measure (all = every candidate)')
    ap.add_argument('--epochs', type=int, default=1); ap.add_argument('--bs', type=int, default=4); ap.add_argument('--lr-dec', type=float, default=3e-4)
    ap.add_argument('--w-meas', type=float, default=1.0); ap.add_argument('--w-prior2', type=float, default=0.3); ap.add_argument('--w-cv', type=float, default=0.5)
    ap.add_argument('--w-feat', type=float, default=0.5); ap.add_argument('--w-tv', type=float, default=0.02); ap.add_argument('--w-anchor3d', type=float, default=0.3)
    ap.add_argument('--w-content', type=float, default=0.5); ap.add_argument('--w-shape', type=float, default=0.2); ap.add_argument('--w-multi', type=float, default=0.3)
    ap.add_argument('--info', default='', help='per-target report content / margin / shape / multiplicity (npz: CONT, content, hu, SHAPE, shape, margin, largest, other)')
    ap.add_argument('--ln-short-axis', action='store_true', help='lymph-node classes with ONE reported size: compare it to the mask SHORT axis')
    ap.add_argument('--chunk', type=int, default=4000); ap.add_argument('--workers', type=int, default=12); ap.add_argument('--ahead', type=int, default=24)
    ap.add_argument('--cases', required=True, help='case lines JSONL (crops.py)'); ap.add_argument('--h5-root', required=True)
    ap.add_argument('--regions-dir', default='', help='<relative_file>.npz token-region bits'); ap.add_argument('--region-packs', default='', help='regions_node_{:02d}.h5 pattern')
    ap.add_argument('--case-rank', default='', help='with --region-packs: JSON {case id: MIL data rank}')
    ap.add_argument('--encoder-ckpt', required=True); ap.add_argument('--mil-ckpt', required=True); ap.add_argument('--mil-config', required=True)
    ap.add_argument('--normalization', required=True); ap.add_argument('--dino2d', required=True, help='DINOv2-L (facebook/dinov2-large) directory')
    a = ap.parse_args(argv)
    if a.mode == 'infer' and not a.ckpt: ap.error('infer needs --ckpt')
    if a.mode == 'train' and not a.init: ap.error('train needs --init')
    return a


def select_jobs(a, C, T, rank, world, out):
    """candidate selection (rank < k, kept targets, excluded pairs) -> per-case crop jobs."""
    FM = {0: 1., 1: 1.5, 2: 1., 3: 1.5}
    sel = C['rank'] < a.k
    if 'keep' in T: sel &= T['keep'][C['t']]
    if a.exclude:
        ex_ = {tuple(int(v) for v in p.split(':')) for p in a.exclude.split(',') if p}
        bad_t = np.array([(int(c_), int(m_)) in ex_ for c_, m_ in zip(T['cls'], T['mt'])]); sel &= ~bad_t[C['t']]
    job = lambda i, t: (int(C['row'][i]), int(C['view'][i]), C['peak'][i], float(C['fov_size_mm'][i]), int(T['bits'][t]), FM[int(T['mt'][t])])
    if a.mode == 'infer':
        if a.split != 'all': sel &= C['tsplit'] == a.split
        done = set()
        for f in glob.glob(str(out / 'cal_r*_c*.npz')):   # resume: all ranks' outputs
            with np.load(f) as z: done.update((z['t'].astype(np.int64) * 64 + z['prank']).tolist())
        cases = sorted(set(C['case'][sel].tolist()))[rank::world]; by = {}
        for i in np.flatnonzero(sel & np.isin(C['case'], cases)):
            t = int(C['t'][i])
            if t * 64 + int(C['rank'][i]) in done: continue
            by.setdefault(str(C['case'][i]), []).append(job(i, t))
        return sorted(by.items()), None, None
    trs = np.flatnonzero(sel & (C['tsplit'] == 'train')); byt = {}
    for i in trs: byt.setdefault(int(C['t'][i]), []).append(i)
    tr = sorted(byt)[rank::world]
    return None, tr, byt


def main(argv=None):
    a = parse_args(argv); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get('RANK', 0)); world = int(os.environ.get('WORLD_SIZE', 1)); local = int(os.environ.get('LOCAL_RANK', 0))
    log = lambda *x: print(f'[{time.strftime("%H:%M:%S")} r{rank}]', *x, flush=True)
    C = dict(np.load(a.cand, allow_pickle=True)); T = dict(np.load(a.targets, allow_pickle=True))
    jobs, tr, byt = select_jobs(a, C, T, rank, world, out)
    if a.mode == 'infer': log('infer cases todo', len(jobs), 'cands', sum(len(v) for _, v in jobs))
    else: log('train targets (rank)', len(tr), 'of', len(byt))
    if a.region_packs: regions = CR.RegionPacks(a.region_packs, json.load(open(a.case_rank)))
    else: regions = CR.RegionDir(a.regions_dir)
    import multiprocessing as mp
    pool = mp.get_context('fork').Pool(a.workers, initializer=CR.winit, initargs=(a.cases, a.h5_root, regions))   # forks before torch / CUDA init
    import torch, torch.nn as nn, torch.nn.functional as F
    from . import encoder as FD
    from . import caliper_model as CM
    SP = FD.CACHE_SP; RG = CM.RG
    dist = world > 1 and a.mode == 'train'
    if dist:
        import torch.distributed as tdist, datetime
        torch.cuda.set_device(local); dev = torch.device('cuda', local); tdist.init_process_group('nccl', timeout=datetime.timedelta(minutes=10))
    else:
        dev = torch.device('cuda', local) if torch.cuda.is_available() else torch.device('cpu')
        if dev.type == 'cuda': torch.cuda.set_device(dev)
    ck_path = a.ckpt if a.mode == 'infer' else a.init
    ck = torch.load(ck_path, map_location='cpu', weights_only=False)
    NK = int(ck['nk']) if a.mode == 'infer' and 'nk' in ck else int(T['kidx'].max()) + 1

    # ---- coarse context (candidate MIL token blocks, uncompressed npz -> memmap)
    mm = {}
    def coarse(row):
        fi, j = int(C['cbf'][row]), int(C['cbj'][row])
        if fi not in mm: f = str(C['cb_files'][fi]); z = np.load(f); mm[fi] = (npz_memmap(f, 'cb_x'), z['cb_mask'], z['cb_off'])
        m_ = mm[fi]; return np.asarray(m_[0][j]), m_[2][j].astype(np.float32), m_[1][j]

    def route_of(it):
        t0, t1 = it['route_t0'][:3], it['route_t0'][3:]; n = int(np.prod(t1 - t0))
        return np.unpackbits(it['route'])[:n].reshape(tuple(t1 - t0)).astype(bool), t0

    def decorate(it):
        r = it['row']; t = int(C['t'][r]); it['t'] = t; it['prank'] = int(C['rank'][r]); it['ctx'] = int(C['cbf'][r]) >= 0; it['shape'] = it['crop'].shape
        it['grp'] = int(T['grp'][t]); it['kidx'] = int(T['kidx'][t]); it['mt'] = int(T['mt'][t]); it['lat'] = int(T['lat'][t]); it['node'] = bool(T['is_node'][t])
        it['win0'] = tuple(float(v) for v in it['window'])
        it['window'] = CM.CLS_WIN.get(int(T['cls'][t]), CM.GRP_WIN.get(it['grp'], (-160., 240.)))   # clinical window per class, group default otherwise
        return it

    def encode(its, enc, train, g, use_ctx, fixed=None):
        bb, mean, rms = enc; o = {k: [] for k in ('f', 'tokc', 'refc', 'tok', 'fov', 'route', 'side', 'hu', 'th', 'sh', 'cvox', 'scale', 'x')}
        for j, it in enumerate(its):
            fov = it['fov']; crop = torch.from_numpy(it['crop']).to(dev); sc = 1.
            cvox = (it['peak'].astype(np.float64) + .5) * np.array(FD.PATCH) - .5
            if fixed is not None: fov, cvox = fixed[j]
            if train:
                z = math.exp(g.uniform(math.log(.8), math.log(1.25)))
                if fov <= 30 and g.uniform() < 0.3: z = g.uniform(.5, 1.)
                th = g.uniform(0, 2 * math.pi); sh = np.array([g.uniform(-.05, .05) * fov, g.uniform(-.1, .1) * fov, g.uniform(-.1, .1) * fov])
            else: z, th, sh = 1., 0., np.zeros(3)
            tok = fov * z / 24.; lo, hi = it['window']
            vmm, tmm, rmm = CM.view_coords(tok, th, sh, dev)
            x = CM.sample_crop(crop, it['origin'], it['stride'], cvox, vmm / sc)
            img = FD.window(x, lo, hi)[None, None]
            o['f'].append(FD.encode(bb, mean, rms, img, (tok / 2, tok / 16, tok / 16))[0])
            o['hu'].append(F.avg_pool3d(x[None, None], (1, 4, 4))[0, 0]); o['x'].append(x)
            o['tokc'].append(tmm.reshape(-1, 3)); o['refc'].append(rmm); o['tok'].append(tok); o['fov'].append(fov); o['th'].append(th); o['sh'].append(sh); o['cvox'].append(cvox)
            rg, t0 = route_of(it); tc = tmm.reshape(-1, 3)
            vox = torch.from_numpy(cvox).to(dev).float() + tc / sc / torch.tensor(SP, device=dev, dtype=torch.float32)
            ti = torch.floor(vox / torch.tensor(FD.PATCH, device=dev, dtype=torch.float32)).long() - torch.from_numpy(t0).to(dev)
            ok = ((ti >= 0) & (ti < torch.tensor(rg.shape, device=dev))).all(1); rt = torch.zeros(len(ti), device=dev)
            if ok.any(): rt[ok] = torch.from_numpy(rg).to(dev)[ti[ok, 0], ti[ok, 1], ti[ok, 2]].float()
            o['route'].append(rt if rg.any() else torch.ones_like(rt))
            o['side'].append((cvox[1] - it['grid'][1] * 16 / 2.) * 0.5 * sc); o['scale'].append(sc)
        L_ = lambda k, dt=torch.long: torch.tensor([it[k] for it in its], device=dev, dtype=dt)
        b = dict(f=torch.stack(o['f']), tokc=torch.stack(o['tokc']), refc=torch.stack(o['refc']), hu=torch.stack(o['hu']), route=torch.stack(o['route']),
                 tok=torch.tensor(o['tok'], device=dev, dtype=torch.float32), fov=torch.tensor(o['fov'], device=dev, dtype=torch.float32),
                 side=torch.tensor(o['side'], device=dev, dtype=torch.float32), th=o['th'], sh=o['sh'], cvox=o['cvox'], scale=o['scale'],
                 c=L_('grp'), ct=torch.tensor([CM.TRUNK[it['grp']] for it in its], device=dev), k=L_('kidx'), mt=L_('mt'), lat=L_('lat'), cx=None)
        if use_ctx and all(it['ctx'] for it in its):
            co = [coarse(it['row']) for it in its]
            b['cx'] = torch.from_numpy(np.stack([c[0] for c in co])).to(dev); b['coff'] = torch.from_numpy(np.stack([c[1] for c in co])).to(dev); b['cmask'] = torch.from_numpy(np.stack([c[2] for c in co])).to(dev)
        add_evidence(b, its)
        b['x'] = torch.stack(o['x'])
        return b

    def add_evidence(b, its):
        """frozen-MIL class evidence on the candidate's 5x5x5 MIL token block -> 4 channels per block token (positive evidence
        softmax(score) x relu(logit) / block max, attention / block max, tanh(scale x logit / 4), in-block flag), trilinearly sampled at
        the head token centres (evt) and refine voxels (evr); evk = the same channels on the coarse context tokens; huc = raw HU in the
        class clinical window, [-1, 1].  Items without a block -> zeros."""
        B = len(its); N = b['tokc'].shape[1]; ok = [bool(it['ctx']) for it in its]
        evt = torch.zeros(B, N, 4, device=dev); evr = torch.zeros(B, 4, *RG, device=dev); evk = torch.zeros(B, 125, 4, device=dev)
        idx = [j for j in range(B) if ok[j]]
        if idx:
            with torch.no_grad():
                MB = mil_block([its[j] for j in idx])
                for q, j in enumerate(idx):
                    it = its[j]; v = MB['valid'][q]; w = torch.softmax(MB['sc'][q], 0) * v.float(); lg = MB['lg'][q]
                    ep = w * lg.clamp_min(0.); ch = torch.stack([ep / ep.max().clamp_min(1e-12), w / w.max().clamp_min(1e-12),
                                                                 torch.tanh(MB['scale'][q] * lg / 4.) * v.float(), v.float()], -1)   # 125 x 4
                    evk[j] = ch; vol = ch.t().reshape(1, 4, 5, 5, 5)
                    def samp(cmm):   # mm offsets in the view -> fractional block index -> trilinear (zeros outside)
                        vox = torch.from_numpy(np.asarray(b['cvox'][j], np.float64)).to(dev).float() + cmm.reshape(-1, 3) / b['scale'][j] / SP_T
                        ti = (vox + .5) / PATCH_T - .5 - torch.from_numpy(it['peak']).to(dev).float()[None] + 2.
                        gr = (ti / 4. * 2. - 1.).flip(-1).view(1, 1, 1, -1, 3)
                        return F.grid_sample(vol, gr, mode='bilinear', padding_mode='zeros', align_corners=True)[0, :, 0, 0]   # 4 x M
                    evt[j] = samp(b['tokc'][j]).t(); evr[j] = samp(b['refc'][j]).view(4, *RG)
        b['evt'] = evt; b['evr'] = evr; b['evk'] = evk
        W_ = torch.tensor([it['window'] for it in its], device=dev, dtype=torch.float32)
        b['huc'] = ((b['hu'].float().clamp(W_[:, 0].view(-1, 1, 1, 1), W_[:, 1].view(-1, 1, 1, 1)) - W_[:, 0].view(-1, 1, 1, 1)) / (W_[:, 1] - W_[:, 0]).view(-1, 1, 1, 1)) * 2 - 1

    # ---- frozen encoder, 3-D head, 2-D stage
    enc = FD.load_encoder(dev, a.encoder_ckpt, a.mil_ckpt, a.normalization); head = CM.CaliperHead(NK).to(dev)
    miss = head.load_state_dict(ck['head'], strict=False); log('loaded', ck_path, 'missing', miss.missing_keys, 'unexpected', miss.unexpected_keys)
    if 'b24' in ck: enc[0].blocks[-1].load_state_dict(ck['b24']['block']); enc[0].norm.load_state_dict(ck['b24']['norm']); log('loaded block 24 from ckpt')
    from transformers import Dinov2Model
    enc2d = Dinov2Model.from_pretrained(a.dino2d).to(dev).eval()
    for p_ in enc2d.parameters(): p_.requires_grad_(False)
    dec = CM.Dec2D().to(dev)
    if 'dec2d' in ck: dec.load_state_dict(ck['dec2d']); log('loaded 2-D decoder from ckpt')
    RP = (torch.randn(1024, 64, generator=torch.Generator().manual_seed(0)) / 8.).to(dev)

    def axial_2d_run(b, lr, its):
        B = lr.shape[0]; ar = torch.arange(B, device=dev); lr = lr.float()
        z = CM.pick_slice(torch.sigmoid(lr)); lt3 = lr[ar, z]; xs = b['x'][ar, z].float()
        wins = [tuple(it['window']) for it in its]; tok2 = CM.dino2d(enc2d, xs, wins)
        xp = F.avg_pool2d(xs[:, None], 4)[:, 0]
        I = torch.stack([CM.win(xp[j], *wins[j]) for j in range(B)])
        hu3 = torch.stack([I, CM.win(xp, *CM.SOFT), CM.win(xp, *CM.LUNG)], 1) * 2 - 1
        ev = b['evr'][ar, :, z] if 'evr' in b else torch.zeros(B, 4, *lt3.shape[-2:], device=dev)
        l2 = dec(tok2, lt3, hu3, ev)
        fe = F.normalize(F.interpolate(torch.einsum('bchw,cd->bdhw', tok2, RP), size=lt3.shape[-2:], mode='bilinear', align_corners=False), dim=1)
        return dict(l2=l2, z=z, lt3=lt3, I=I, fe=fe, xs=xs, xp=xp)

    ln_single = lambda t_: bool(a.ln_short_axis and T['is_node'][t_] and not np.isfinite(T['y_short'][t_]))
    g = np.random.default_rng(1000 + rank)

    # ---- frozen classifier head (class evidence of the candidate block)
    from fora.mil.model import build_head
    mcfg = json.load(open(a.mil_config))['training']; mck = torch.load(a.mil_ckpt, map_location='cpu', weights_only=False); NCL = len(mck['classes'])
    milh = build_head(NCL, dim=1024 + 6 * mcfg.get('position_freqs', 4)).to(dev)
    milh.load_state_dict(mck['head']); milh.eval()
    for p_ in milh.parameters(): p_.requires_grad_(False)
    MPEN = F.softplus(milh.route_bias_raw.float()); del mck
    PATCH_T = torch.tensor(FD.PATCH, device=dev, dtype=torch.float32); SP_T = torch.tensor(SP, device=dev, dtype=torch.float32)

    def mil_block(its):
        """frozen MIL per-token class score (attention logit, + route penalty outside the class route) and token logit on the
        candidate's 5x5x5 MIL token block (peak-centred, the candidate's view); valid = in-volume body tokens."""
        co = [coarse(it['row']) for it in its]
        X = torch.from_numpy(np.stack([c[0] for c in co])).to(dev).float(); V = torch.from_numpy(np.stack([c[2] for c in co])).to(dev).bool()
        cls = [int(T['cls'][it['t']]) for it in its]; uc = sorted(set(cls)); ci = torch.tensor([uc.index(c) for c in cls], device=dev)
        with torch.no_grad():
            sc, lg = milh._chunk_outputs(X.reshape(-1, X.shape[-1]), torch.tensor(uc, device=dev))
            B = len(its); sc = sc.float().view(B, -1, len(uc)); lg = lg.float().view(B, -1, len(uc))
            sc = sc.gather(2, ci[:, None, None].expand(B, sc.shape[1], 1))[..., 0]; lg = lg.gather(2, ci[:, None, None].expand(B, lg.shape[1], 1))[..., 0]
            d = np.stack(np.meshgrid(*[np.arange(-2, 3)] * 3, indexing='ij'), -1).reshape(-1, 3)
            for j, it in enumerate(its):   # route penalty (as the MIL head: score - softplus(route_bias) outside the class route)
                rg, t0 = route_of(it)
                if not rg.any(): continue
                P_ = it['peak'][None] + d - t0[None]; inb = np.all((P_ >= 0) & (P_ < np.array(rg.shape)), 1); r_ = np.ones(len(d), np.float32)
                r_[inb] = rg[tuple(P_[inb].T)]
                sc[j] = sc[j] - (1. - torch.from_numpy(r_).to(dev)) * MPEN[cls[j]]
        sc = sc.masked_fill(~V, -1e4)
        osc = milh.output_scale[torch.tensor(cls, device=dev)].float(); obi = milh.output_bias[torch.tensor(cls, device=dev)].float()
        return dict(sc=sc, lg=lg, valid=V, scale=osc, bias=obi)

    if a.mode == 'infer':
        head.eval(); rows = []; chunk = len(glob.glob(str(out / f'cal_r{rank:03d}_c*.npz'))); t0 = time.time(); nerr = 0; ndone = 0

        def flush():
            nonlocal rows, chunk
            if not rows: return
            z = {k: np.stack([np.asarray(r_[k]) for r_ in rows]) for k in rows[0]}
            tmp = out / f'.cal_r{rank:03d}_c{chunk:04d}.tmp.npz'; np.savez(tmp, **z); os.replace(tmp, out / f'cal_r{rank:03d}_c{chunk:04d}.npz'); chunk += 1; rows = []

        def run(grp_, fixed=None):
            b = encode(grp_, enc, False, g, True, fixed); _, lr = head(b); m = torch.sigmoid(lr.float())
            b['ax'] = axial_2d_run(b, lr, grp_)
            mu = ((m[..., None] * b['refc']).sum((1, 2, 3)) / m.sum((1, 2, 3)).clamp_min(1e-3)[:, None])
            return b, lr, m, mu

        def touch(Cm, mz=1, mxy=3):   # component within mz / mxy voxels of the view box faces (the box has a 1-voxel zero-padded rim)
            if Cm is None or not Cm.any(): return False
            return bool(Cm[:mz + 1].any() or Cm[-mz - 1:].any() or Cm[:, :mxy + 1].any() or Cm[:, -mxy - 1:].any() or Cm[:, :, :mxy + 1].any() or Cm[:, :, -mxy - 1:].any())

        def process(batch):
            nonlocal ndone
            with torch.no_grad():
                for grp_ in ([x for x in batch if x['ctx']], [x for x in batch if not x['ctx']]):
                    if not grp_: continue
                    b, lr, m, mu = run(grp_); mun = mu.cpu().numpy(); res = {}; zoom = []
                    for j, it in enumerate(grp_):
                        mtj = F.avg_pool3d(m[j][None, None], (2, 4, 4)).flatten(); W = mtj.sum().clamp_min(1e-3)
                        rf = float((mtj * b['route'][j]).sum() / W); side = float(b['side'][j] + mun[j][1])
                        fin = ((mtj[:, None] * b['f'][j].float()).sum(0) / W).half().cpu().numpy()
                        mh = m[j] > .5; dil = F.max_pool3d(mh[None, None].float(), (5, 9, 9), stride=1, padding=(2, 4, 4))[0, 0] > 0; shl = dil & ~mh
                        hin = float(b['hu'][j][mh].mean()) if mh.any() else np.nan; hsh = float(b['hu'][j][shl].mean()) if shl.any() else np.nan
                        prob, lo = CM.crop_grid_prob(lr[j], it, b, j, dev); mc, C_ = CM.measure_crop(prob, lo, it, it['node'], b['cvox'][j] + mun[j] / SP)
                        res[j] = dict(mc=mc, C=C_, lo=lo, zoom=False, rf=rf, side=side, fin=fin, hin=hin, hsh=hsh, tok=float(b['tok'][j]), msum=float(m[j].sum() * float(b['tok'][j]) ** 3 / 32.))
                        res[j]['m2'] = CM.measure2d(torch.sigmoid(b['ax']['l2'][j]).cpu().numpy(), float(b['tok'][j]) / 4)
                        if it['mt'] == 0 and mc is not None and mc['long_mm'] < 10:
                            f2 = float(np.clip(2.5 * mc['long_mm'], 12., 24.))
                            if f2 / 24. < float(b['tok'][j]) - 1e-6: zoom.append((j, f2, np.array(mc['centroid_cache_vox'])))
                    if zoom:   # small lesions: zoomed view centred on the pass-1 mask centroid
                        zg = [grp_[j] for j, _, _ in zoom]; b2, lr2, m2, mu2 = run(zg, [(f2, c) for _, f2, c in zoom]); mu2 = mu2.cpu().numpy()
                        for q, (j, f2, c) in enumerate(zoom):
                            it = grp_[j]; prob, lo = CM.crop_grid_prob(lr2[q], it, b2, q, dev); mc, C_ = CM.measure_crop(prob, lo, it, it['node'], b2['cvox'][q] + mu2[q] / SP)
                            res[j]['zoom'] = True
                            if mc is not None: res[j].update(mc=mc, C=C_, lo=lo, tok=float(b2['tok'][q]))
                            m2z = CM.measure2d(torch.sigmoid(b2['ax']['l2'][q]).cpu().numpy(), float(b2['tok'][q]) / 4)
                            if m2z is not None: res[j]['m2'] = m2z
                    for j, it in enumerate(grp_):   # multi-scale expansion: mask touching the view border -> re-run at 1.5x / 2x FOV centred on the mask
                        R = res[j]; R['touch0'] = int(touch(R['C'])) if R['mc'] else 0; R['expand'] = 1.
                        if not R['touch0']: continue
                        fov0 = R['tok'] * 24.
                        for sc_ in (1.5, 2.0):
                            bx, lrx, mx, mux = run([it], [(fov0 * sc_, np.array(R['mc']['centroid_cache_vox']))]); mux = mux.cpu().numpy()
                            prob, lo = CM.crop_grid_prob(lrx[0], it, bx, 0, dev); mcx, Cx = CM.measure_crop(prob, lo, it, it['node'], bx['cvox'][0] + mux[0] / SP)
                            if mcx is None: break
                            R.update(mc=mcx, C=Cx, lo=lo, expand=sc_)
                            m2x = CM.measure2d(torch.sigmoid(bx['ax']['l2'][0]).cpu().numpy(), float(bx['tok'][0]) / 4)
                            if m2x is not None: R['m2'] = m2x
                            if not touch(Cx): break
                        R['touch1'] = int(touch(R['C'])) if R['mc'] else 0
                    for j, it in enumerate(grp_):
                        R = res[j]; mc = R['mc']; st = it['stride'].astype(np.float64); sp = SP * st
                        ins, iz, ip = CM.ins_diam(R['C'] if mc is not None else None, sp)
                        r = dict(row=it['row'], t=it['t'], prank=it['prank'], long=np.nan, short=np.nan, ins=ins, ext3d=np.nan, vol=np.nan, empty=1, zoom=int(R['zoom']), border=0, route=R['rf'], side=R['side'],
                                 dcen=np.nan, hin=R['hin'], hsh=R['hsh'], msum=R['msum'], tok=R['tok'], touch0=R['touch0'], touch1=R.get('touch1', R['touch0']), expand=R['expand'], f_in=R['fin'])
                        if mc is not None:
                            pk = (it['peak'].astype(np.float64) + .5) * np.array(FD.PATCH) - .5
                            r.update(long=float(mc['long_mm']), short=float(mc['short_mm']), ext3d=float(mc.get('extent3d_mm', np.nan)), vol=float(mc.get('volume_mm3', np.nan)), empty=0,
                                     border=int(mc.get('touches_fov_border', False)), dcen=float(np.linalg.norm((np.array(mc['centroid_cache_vox']) - pk) * SP)))
                        r['ax_ok'] = 0
                        if R.get('m2') is not None:   # the 2-D axial measurement replaces the 3-D one
                            m2_ = R['m2']; r.update(long=float(m2_['long']), short=float(m2_['short']), empty=0, ax_ok=1)
                            if it['mt'] in (1, 3): r['ins'] = float(m2_['ins'])
                        rows.append(r); ndone += 1
            if len(rows) >= a.chunk: flush()

        pend = []
        for res_ in bounded(pool, CR.crops_for_case, jobs, a.ahead):
            for it in res_:
                if isinstance(it, tuple): nerr += 1; log('crop fail', it[1], it[2]); continue
                pend.append(decorate(it))
            while len(pend) >= a.bs:
                process(pend[:a.bs]); pend = pend[a.bs:]
                if ndone % 2000 < a.bs: log(f'{ndone} cands {(time.time() - t0) / max(ndone, 1):.3f}s/cand errs {nerr}')
        if pend: process(pend)
        flush(); pool.terminate(); log('finished', ndone, 'errs', nerr, f'{time.time() - t0:.0f}s'); return

    train_decoder(a, C, T, tr, byt, pool, decorate, encode, enc, head, dec, axial_2d_run, ln_single, g, ck, NK, dev, dist, rank, world, out, log)


def train_decoder(a, C, T, tr, byt, pool, decorate, encode, enc, head, dec, axial_2d_run, ln_single, g, ck, NK, dev, dist, rank, world, out, log):
    """axial-2-D decoder training (3-D head + block 24 frozen)."""
    import torch, torch.nn as nn, torch.nn.functional as F
    from scipy import ndimage as ndi
    from . import caliper_model as CM
    if dist: import torch.distributed as tdist
    FM = CM.FM
    for p_ in head.parameters(): p_.requires_grad_(False)
    head.eval(); params = list(dec.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr_dec, weight_decay=0.01)
    nsteps = max(1, len(tr) // a.bs)
    if dist:
        t_ = torch.tensor([nsteps], device=dev); tdist.all_reduce(t_, op=tdist.ReduceOp.MIN); nsteps = int(t_)
        for p_ in params: tdist.broadcast(p_.data, 0)
    total = a.epochs * nsteps; sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr_dec, total_steps=max(total, 2), pct_start=0.05)
    log('steps/epoch', nsteps, 'epochs', a.epochs, 'targets/rank', len(tr))
    perm_t = [tr[i] for i in g.permutation(len(tr))]
    def tjobs():
        for ep_ in range(a.epochs):
            for s_ in range(nsteps):
                for t in perm_t[s_ * a.bs:(s_ + 1) * a.bs]:
                    rr = sorted(byt[t], key=lambda i: C['rank'][i])
                    yield (str(C['case'][rr[0]]), [(int(C['row'][i]), int(C['view'][i]), C['peak'][i], float(C['fov_size_mm'][i]), int(T['bits'][t]), FM[int(T['mt'][t])]) for i in rr])
    stream = bounded(pool, CR.crops_for_case, tjobs(), a.ahead); step = 0; t0 = time.time(); agg = collections.defaultdict(float); apes = []; nerr = 0
    INFO = dict(np.load(a.info, allow_pickle=True)) if a.info else None
    # report content -> HU range of the lesion interior; unknown / mixed -> no content term
    HUR = {'fluid': (-25., 30.), 'dense_fluid': (15., 60.), 'blood': (35., 95.), 'soft_tissue': (0., 180.), 'fat': (-180., -30.), 'calcified': (130., 3000.),
           'air': (-1100., -600.), 'ground_glass': (-800., -250.), 'part_solid': (-800., 150.)}
    def content_terms(pj, hu, rng, ysz):
        """partial-volume aware: core = mask pixels >= 1 px inside the edge.  (a) p-weighted mean HU of the core within the stated
        range; (b) lesions >= 10 mm only: per-pixel penalty for core pixels > 100 HU outside the range, and pull-in of clearly
        matching tissue in a 4-px ring outside the mask.  Small lesions (nodules, stones) get only (a)."""
        lo_, hi_ = rng; hard = (pj.detach() > .5).cpu().numpy()
        core = torch.from_numpy(ndi.binary_erosion(hard, iterations=1)).to(dev).float()
        if core.sum() < 1: core = torch.from_numpy(hard).to(dev).float()
        if core.sum() < 1: return torch.zeros((), device=dev)
        w_ = pj * core; m_ = (w_ * hu).sum() / w_.sum().clamp_min(1e-3)
        res = ((torch.relu(lo_ - m_) + torch.relu(m_ - hi_)) / 50.).clamp_max(3.)
        if ysz >= 10.:
            d_ = ((torch.relu(lo_ - 100. - hu) + torch.relu(hu - hi_ - 100.)) / 50.).clamp_max(3.)
            res = res + (w_ * d_).sum() / w_.sum().clamp_min(1.)
            ring = torch.from_numpy(ndi.binary_dilation(hard, iterations=4) & ~hard).to(dev)
            inr = (hu >= lo_ + 10.) & (hu <= hi_ - 10.) & ring
            res = res + 0.3 * ((1 - pj) * inr.float()).sum() / ring.float().sum().clamp_min(1.)
        return res
    def compact(pj, thr=1.3):
        gy = pj[1:, :-1] - pj[:-1, :-1]; gx = pj[:-1, 1:] - pj[:-1, :-1]; P = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6).sum(); A_ = pj.sum().clamp_min(1.)
        return torch.relu(P ** 2 / (4 * math.pi * A_) - thr) / 5.
    for ep in range(a.epochs):
        for s in range(nsteps):
            byk = {}
            for _try in range(20):
                for _ in range(min(a.bs, len(perm_t[s * a.bs:(s + 1) * a.bs])) if _try == 0 else 1):
                    for it in next(stream):
                        if isinstance(it, tuple): nerr += 1; continue
                        it = decorate(it); byk.setdefault(it['t'], []).append(it)
                if byk: break
            its = [it for t in sorted(byk) for it in sorted(byk[t], key=lambda q: q['prank'])]
            with torch.no_grad():
                b = encode(its, enc, True, g, True); _, lr = head(b)
            ax = axial_2d_run(b, lr, its); p2 = torch.sigmoid(ax['l2']); frac = step / max(total - 1, 1)
            L = collections.defaultdict(list); cand = collections.defaultdict(list)
            for j, it in enumerate(its):
                t = it['t']; px = float(b['tok'][j]) / 4; pj = p2[j]; roi = CM.roi_of(pj)
                L['cv'].append(CM.chan_vese(pj, ax['I'][j], roi)); L['feat'].append(CM.chan_vese(pj, ax['fe'][j], roi))
                L['tv'].append((pj[1:] - pj[:-1]).abs().mean() + (pj[:, 1:] - pj[:, :-1]).abs().mean())
                L['anchor'].append(F.binary_cross_entropy_with_logits(ax['l2'][j], torch.sigmoid(ax['lt3'][j]).detach()))
                if INFO is not None:
                    cn = str(INFO['CONT'][INFO['content'][t]]); hv = float(INFO['hu'][t])
                    rng = (hv - 25., hv + 25.) if np.isfinite(hv) else HUR.get(cn)
                    if rng is not None and not np.isfinite(hv):   # partial volume: lung lesions and small lesions read lower / blurred
                        lung_ = int(T['grp'][t]) == 0 or ('ycmp' in T and int(T['ycmp'][t]) == 3)
                        if lung_ and cn in ('soft_tissue', 'part_solid'): rng = (-400., rng[1])
                        if lung_ and cn == 'calcified': rng = (50., rng[1])
                        if float(T['y_long'][t]) < 10.: rng = (rng[0] - 150., rng[1] + 150.)
                    if rng is not None: L['content'].append(content_terms(pj, ax['xp'][j], rng, float(T['y_long'][t])))
                    shp = str(INFO['SHAPE'][INFO['shape'][t]]); mrg = int(INFO['margin'][t])
                    if mrg == 1 or shp in ('round', 'oval'): L['shape'].append(compact(pj))   # report: smooth / round / oval -> strong
                    elif it['mt'] == 0 and mrg in (0,) and shp in ('unknown', 'elongated'): L['shape'].append(0.3 * compact(pj, 2.0))   # unstated -> mild
                m2 = CM.measure2d(pj.detach().cpu().numpy(), px)
                if m2 is None: cand[t].append((j, None, None, None)); continue
                Lv = CM.soft_len(pj, m2['comp'], m2['u'], px); Lv = Lv * (m2['long'] / Lv).detach()
                Sv = CM.soft_len(pj, m2['comp'], m2['n'], px); Sv0 = Sv * (m2['short'] / Sv).detach()
                yc_ = int(T['ycmp'][t]) if 'ycmp' in T else 0
                if yc_ == 4: cand[t].append((j, None, None, None)); continue   # craniocaudal stone sizes are not measurable in 2-D
                if it['mt'] == 0 and yc_ == 3: v, vex, sv = (Lv + Sv0) / 2, (m2['long'] + m2['short']) / 2, None   # nodules = mean diameter
                elif it['mt'] == 0 and yc_ == 1: v, vex, sv = Lv, m2['long'], None
                elif it['mt'] == 0 and yc_ == 2: v, vex, sv = Sv0, m2['short'], None
                elif it['mt'] == 0:
                    if ln_single(t): v, vex, sv = Sv0, m2['short'], None
                    else: v, vex, sv = Lv, m2['long'], (Sv0 if np.isfinite(T['y_short'][t]) else None)
                else: v, vex, sv = Sv * (m2['ins'] / Sv).detach(), m2['ins'], None
                cand[t].append((j, v, vex, sv))
            for t, cs in cand.items():
                y = float(T['y_long'][t]); eps = float(T['eps_long'][t]) if np.isfinite(T['eps_long'][t]) else 1.; prior = float(T['fov_size_mm'][t])
                ok = [c for c in cs if c[1] is not None]
                if not ok:
                    j0 = cs[0][0]; L['empty'].append(0.2 * F.softplus(-ax['l2'][j0][44:52, 44:52]).clamp_max(5.).mean()); continue
                kk = min(ok, key=lambda c: abs(math.log(max(c[2], 1e-3) / y)))
                L['meas'].append(torch.relu((kk[1] - y).abs() - eps) / y); apes.append(abs(kk[2] - y) / y)
                if kk[3] is not None:
                    ys = float(T['y_short'][t]); es = float(T['eps_short'][t]) if np.isfinite(T['eps_short'][t]) else 1.
                    L['meas'].append(0.5 * torch.relu((kk[3] - ys).abs() - es) / ys)
                for c in ok:
                    if c is not kk: L['prior'].append(torch.relu(c[1] - 1.5 * prior) / prior)
                if INFO is not None:
                    others = [c for c in ok if c is not kk]
                    if INFO['largest'][t]:   # 'multiple, the largest of which ...': other candidates must not exceed the reported size
                        for c in others: L['multi'].append(torch.relu(c[1] - 1.1 * y) / y)
                    os_ = sorted(INFO['other'][t], reverse=True); used = set()
                    for s_ in os_:   # other listed sizes -> nearest unmatched candidate (plausible matches only)
                        cand_ = [(abs(math.log(max(c[2], 1e-3) / s_)), q) for q, c in enumerate(others) if q not in used]
                        if not cand_: break
                        dl, q = min(cand_)
                        if dl < 0.4: used.add(q); L['multi'].append(torch.relu((others[q][1] - s_).abs() - 1.) / s_)
            mean_ = lambda k_: torch.stack(L[k_]).mean() if L[k_] else torch.zeros((), device=dev)
            wa = a.w_anchor3d * max(0.15, 1 - frac)
            loss = a.w_content * mean_('content') + a.w_shape * mean_('shape') + a.w_multi * mean_('multi') + a.w_meas * mean_('meas') + a.w_prior2 * mean_('prior') + a.w_cv * mean_('cv') + a.w_feat * mean_('feat') + a.w_tv * mean_('tv') + wa * mean_('anchor') + mean_('empty')
            opt.zero_grad(); loss.backward()
            if dist:
                gs = [p_.grad if p_.grad is not None else torch.zeros_like(p_) for p_ in params]; flat = torch.cat([x_.flatten() for x_ in gs]); tdist.all_reduce(flat); flat /= world; o_ = 0
                for p_ in params: n_ = p_.numel(); p_.grad = flat[o_:o_ + n_].view_as(p_).clone(); o_ += n_
            nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); step += 1
            for k_ in ('meas', 'prior', 'cv', 'feat', 'tv', 'anchor', 'empty', 'content', 'shape', 'multi'): agg[k_] += float(mean_(k_).detach())
            if not math.isfinite(float(loss)): log('stop: non-finite loss'); break
            if step % 10 == 0 or step <= 3:
                a_ = torch.tensor([float(np.median(apes)) if apes else float('nan'), float(np.mean(np.array(apes) <= .3)) if apes else float('nan')], device=dev)
                if dist: tdist.all_reduce(a_); a_ /= world
                n_ = 10 if step % 10 == 0 else 1
                log(f'ep {ep} step {step}/{total}', ' '.join(f'{k_}={v_ / n_:.4f}' for k_, v_ in agg.items()), f'| matched APE med {float(a_[0]):.3f} within30 {float(a_[1]):.3f} | anchor w {wa:.3f} | {(time.time() - t0) / step:.2f}s/step errs {nerr}')
                agg = collections.defaultdict(float); apes = []
            if rank == 0 and step % 100 == 0:
                torch.save(dict(head=head.state_dict(), dec2d=dec.state_dict(), step=step, args=vars(a), nk=NK, **({'b24': ck['b24']} if 'b24' in ck else {})), out / f'head_step{step}.pt')
        if rank == 0:
            torch.save(dict(head=head.state_dict(), dec2d=dec.state_dict(), epoch=ep, step=step, args=vars(a), nk=NK, **({'b24': ck['b24']} if 'b24' in ck else {})), out / f'head_ep{ep}.pt'); log('saved', f'head_ep{ep}.pt')
    log('training finished', step, 'steps', f'{time.time() - t0:.0f}s'); pool.terminate()
    if dist: tdist.barrier(); tdist.destroy_process_group()


if __name__ == '__main__':
    main()

"""Per-finding signed evidence, foci and zoomed evidence images of the frozen CT MIL classifier
(paper sections "Signed evidence maps" and "Grounded report generation").

For a given list of (study, class) targets, block 24 + the organ-routed MIL head run once over all body tokens of all views
of the study (streaming online softmax, soft route penalty, identical to the classifier forward). For the requested classes:

    evidence_t = attention_t * token_logit_t * output_scale   (attention = softmax over ALL tokens of all views)
    positive evidence = max(evidence, 0)

From the positive evidence the extractor writes, per (study, class): the attention-pooled 1048-d head input (``feat``, the
writer's pooled finding token), logit / probability, region / side / cranio-caudal summaries, extent statistics, the foci
decomposition with 16 evidence tokens (``fo_*``, see ``fora.mil.evidence``), optionally the 5x5x5 token block around the
evidence peak of the best two views (``blk_*``, ``--block 2``; consumed by the lesion calipers at inference) and up to six JPEG
evidence images (``img_blob`` / ``img_off``, see ``fora.mil.render``).

The same code builds the training stores (``--rank`` over the staged cohort, ``--block -1``) and runs at inference
(``--cases`` with one study, ``--block 2``, CT volumes from ``--ct-dir``; ``extract_ct_evidence``). Output: ``<out>/extent_r<rank>_p<part>_c<chunk>.npz``
every ``--chunk`` studies; studies already present in this rank's chunks are skipped (resume).

Paper runs: training stores ``--crops 32 --block -1 --render 1 --chunk 200`` (final CT MIL classifier checkpoint);
inference ``--crops 32 --block 2 --render 1 --chunk 1``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from fora.encoder.cache import local_view_file
from fora.mil.bag import RoutedBag, load_study
from fora.mil.evidence import FOCI_KEYS, foci, study_evidence, view_info
from fora.mil.extents import STAT_NAMES, extents, peak_blocks
from fora.mil.model import class_bits, load_classifier
from fora.mil.render import N_VIEWS, study_finding_images


def load_series_volume(h5_root, rep, s):
    """CT volume of one series of a study plan entry: resampled to the target spacing, body-cropped, int16 HU in [-1024, 3071]
    (the same volume the cached views were encoded from)."""
    import h5py

    from fora.geometry.tiling import body_frame, resample_series
    with h5py.File(f"{h5_root}/{rep['h5']}.h5", 'r') as f:
        sg = f[rep['patient']][rep['report']][s['acc']][s['sid']]; src = sg[s['sub']] if s['sub'] else sg
        data = src['data'][:, :, :, s['frame']] if s['frame'] is not None else src['data'][()]; affine = src['affine'][()]
    vol, _ = resample_series(data, affine); del data
    y0, y1, x0, x1, _, _ = body_frame(vol)
    return np.clip(np.ascontiguousarray(vol[:, y0:y1, x0:x1], dtype=np.float32), -1024, 3071).astype(np.int16)


def _expand(m: dict) -> dict:
    """${VAR} references in the string values of a classifier config (the ``training`` section is kept verbatim)."""
    return {k: (os.path.expandvars(v) if isinstance(v, str) and k != 'training' else v) for k, v in m.items()}


class EvidenceExtractor:
    """Frozen block 24 + organ-routed MIL head; ``study()`` returns the evidence record of one study for a list of classes."""

    def __init__(self, config: Path, ckpt: Path, device: str = 'cuda', cache_root: Path | None = None,
                 region_root: Path | None = None):
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = False
        self.m = m = _expand(json.loads(Path(config).read_text())); self.cfg = m['training']; self.dev = dev = torch.device(device)
        self.cache_root = Path(cache_root or m['cache_root']); self.region_root = Path(region_root or m['region_root'])
        norm = torch.load(m['normalization'], map_location=dev, weights_only=False); self.mean = norm['mean'].to(dev).float(); self.rms = float(norm['rms'])
        self.block, self.head, self.classes = load_classifier(m, ckpt, dev)
        self.cb = class_bits(json.loads(Path(m['class_region_bits']).read_text()), self.classes, dev)

    def prepare(self, case):
        """Host arrays of all cached views of one study (crop-major body tokens + per-token region bits) and their geometry."""
        views = load_study(case, self.cache_root, self.region_root, self.cfg.get('position_freqs', 4))
        vinfo = [view_info(local_view_file(vw, self.cache_root), vw, v['bits']) for vw, v in zip(case['views'], views)]
        return views, vinfo

    def study(self, views, vinfo, cl, crops, block_radius=-1):
        """One streaming pass over all views (``fora.mil.evidence.study_evidence``) + foci, optional peak token blocks."""
        bag = RoutedBag(views, self.dev, self.cb, self.cfg.get('min_route_tokens', 8), self.cfg.get('gpu_bag_gib', 20.))
        res = study_evidence(self.block, self.head, bag, vinfo, self.mean, self.rms, cl, crops)
        del bag
        res['vinfo'] = vinfo
        res['blk'] = peak_blocks(res['E'], vinfo, res['XA'], block_radius) if block_radius >= 0 else None
        res['fo'] = foci(res['E'], vinfo, [self.classes[int(c)] for c in cl.tolist()], res['XA'])
        del res['XA']
        return res


class ChunkWriter:
    """Accumulates per-(study, class) records and writes ``extent_r<rank>_p<part>_c<chunk>.npz`` files."""

    KEYS = ('ids', 'pair_case', 'pair_rep', 'cls', 'logit', 'prob', 'feat', 'reg3', 'frac3', 'regall', 'left', 'right', 'zfrac', 'peak', 'view',
            'att_reg3', 'att_frac3', 'att_regall', 'att_left', 'att_right', 'att_zfrac', 'ev_mass', 'merged', 'peak_view',
            'v_pair', 'v_view', 'v_series', 'v_window', 'v_stats', 'blk_x', 'blk_ev', 'blk_mask', 'blk_off', 'blk_view', 'blk_ctr', 'blk_rel')

    def __init__(self, out: Path, rank: int, part_label: int, classes, chunk_no: int = 0, render: bool = True):
        self.out = out; self.rank = rank; self.plabel = part_label; self.classes = classes; self.chunk_no = chunk_no; self.render = render
        self.buf = {k: [] for k in self.KEYS}; self.buf.update({'fo_' + k: [] for k in FOCI_KEYS}); self.img = []

    def add(self, cid, rep_uid, clist, res, view_min_rel=0.05, images=None):
        merged, vrows, pview = extents(res['E'], res['vinfo'], view_min_rel)
        buf = self.buf; b = len(buf['pair_case']); buf['ids'].append(cid)
        tl = lambda t: t.float().cpu().numpy().tolist()  # noqa: E731
        buf['pair_case'] += [cid] * len(clist); buf['pair_rep'] += [rep_uid] * len(clist); buf['cls'] += clist; buf['logit'] += tl(res['logit']); buf['prob'] += tl(res['prob'])
        buf['feat'].append(res['feat'].half().cpu().numpy()); buf['reg3'] += res['ri'].tolist(); buf['frac3'] += tl(res['fr']); buf['regall'] += tl(res['reg'])
        buf['left'] += tl(res['side'][:, 0]); buf['right'] += tl(res['side'][:, 1]); buf['zfrac'] += tl(res['zfrac']); buf['peak'] += res['peak'][:, 1:].tolist(); buf['view'] += res['peak'][:, 0].tolist()
        buf['att_reg3'] += res['ri_a'].tolist(); buf['att_frac3'] += tl(res['fr_a']); buf['att_regall'] += tl(res['reg_a']); buf['att_left'] += tl(res['side_a'][:, 0]); buf['att_right'] += tl(res['side_a'][:, 1])
        buf['att_zfrac'] += tl(res['zf_a']); buf['ev_mass'] += tl(res['evm']); buf['merged'] += merged.tolist(); buf['peak_view'] += pview.tolist()
        if self.render:
            self.img += images
        for R_ in res['fo']:
            for k_ in FOCI_KEYS: buf['fo_' + k_].append(np.asarray(R_[k_]))
        if res['blk'] is not None:
            for n_k in ('x', 'ev', 'mask', 'off', 'view', 'ctr', 'rel'): buf['blk_' + n_k].append(res['blk'][n_k])
        for q, k, s, w, st in vrows: buf['v_pair'].append(b + q); buf['v_view'].append(k); buf['v_series'].append(s); buf['v_window'].append(w); buf['v_stats'].append(st)

    def __len__(self):
        return len(self.buf['ids'])

    def flush(self):
        if not self.buf['ids']: return
        z = {k: (np.concatenate(v) if (k == 'feat' or k.startswith('blk_')) and v else np.stack(v) if k.startswith('fo_') and v else np.array(v)) for k, v in self.buf.items()}
        if self.render:
            imgs = self.img; blob = b''.join(b for t in imgs for b in t); off = np.zeros((len(imgs), N_VIEWS + 1), np.int64); o = 0
            for i_, t in enumerate(imgs):
                off[i_, 0] = o
                for j_, b in enumerate(t): o += len(b); off[i_, j_ + 1] = o
            z['img_blob'] = np.frombuffer(blob, np.uint8); z['img_off'] = off
        for k in ('regall', 'att_regall', 'feat', 'blk_off'): z[k] = z[k].astype(np.float16) if len(z[k]) else z[k]
        name = f'extent_r{self.rank:03d}_p{self.plabel}_c{self.chunk_no:04d}'
        tmp = self.out / f'.{name}.tmp.npz'; np.savez(tmp, stat_names=np.array(STAT_NAMES), classes=np.array(self.classes), **z)
        os.replace(tmp, self.out / f'{name}.npz'); self.chunk_no += 1
        for v in self.buf.values(): v.clear()
        self.img = []


def _read_cases(path, want):
    cases = {}
    with open(path) as f:
        for line in f:
            i = line.find('"id": "'); cid = line[i + 7:line.index('"', i + 7)]
            if cid in want: cases[cid] = json.loads(line)
    return cases


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--config', type=Path, required=True, help='CT classifier config JSON (training section, normalization, block24_state, class_region_bits, cases, assign, cache_root, region_root)')
    ap.add_argument('--ckpt', type=Path, required=True, help='classifier checkpoint (classes, block, head)')
    ap.add_argument('--targets', type=Path, required=True, help='{"targets": {case_id: [cls, ...]}, "case2rep": {case_id: report_uid}}')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--cases', type=Path, default=None, help='explicit case ids (one per line); otherwise the rank assignment of the config is used')
    ap.add_argument('--rank', type=int, default=0, help='classifier data rank (4 * node + gpu)')
    ap.add_argument('--part', type=int, default=0); ap.add_argument('--parts', type=int, default=1)
    ap.add_argument('--plabel', type=int, default=-1, help='part number used in file names (default --part)')
    ap.add_argument('--resume-rank', type=int, default=0, help='1: done = studies in ANY chunk of this rank (re-partitioning)')
    ap.add_argument('--crops', type=int, default=32); ap.add_argument('--chunk', type=int, default=200)
    ap.add_argument('--device', default='cuda'); ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--cache-root', type=Path, default=None, help='directory with the block-23 series files (default: config cache_root; e.g. the node-local staged copy)')
    ap.add_argument('--region-root', type=Path, default=None, help='directory with the token region bits (default: config region_root)')
    ap.add_argument('--cases-jsonl', default=None, help='override config cases file; "{rank}" is replaced by the rank')
    ap.add_argument('--block', type=int, default=-1, help='dump the (2r+1)^3 token block around the evidence peak; -1 = off (training), 2 = inference')
    ap.add_argument('--view-min-rel', type=float, default=0.05, help='store a per-view row only if the view peak >= this x global peak')
    ap.add_argument('--render', type=int, default=1, help='render the zoomed evidence images per finding')
    ap.add_argument('--plan-reports', type=Path, nargs='*', default=(), help='study plan JSONL(s) with h5 location + series per id (training renders)')
    ap.add_argument('--h5-root', type=Path, default=None, help='root of the CT h5 files (training renders)')
    ap.add_argument('--ct-dir', type=Path, default=None, help='inference: directory with ct_<series:03d>.npy volumes of the single study')
    return ap


def run(a: argparse.Namespace) -> None:
    a.out.mkdir(parents=True, exist_ok=True)
    if a.threads: torch.set_num_threads(a.threads)
    ex = EvidenceExtractor(a.config, a.ckpt, a.device, a.cache_root, a.region_root); m = ex.m; rank = a.rank
    TG = json.loads(Path(a.targets).read_text()); tg = TG['targets']; c2r = TG['case2rep']
    if a.cases: mine = [c for c in Path(a.cases).read_text().split() if c in tg]
    else:
        A = json.loads(Path(m['assign']).read_text()); mine = [c for c in (A['assignments']['val'][rank] + A['assignments']['train'][rank]) if c in tg]
    mine = mine[a.part::a.parts]
    done = set(); chunk_no = 0
    if a.plabel < 0: a.plabel = a.part
    for f in sorted(glob.glob(str(a.out / f'extent_r{rank:03d}_p*_c*.npz'))):
        own = f.rsplit('_p', 1)[1].split('_c')[0] == str(a.plabel)
        if not (own or a.resume_rank): continue
        with np.load(f) as z: done.update(z['ids'].tolist())
        if own: chunk_no = max(chunk_no, int(f.rsplit('_c', 1)[1][:4]) + 1)
    todo = [c for c in mine if c not in done]
    log = lambda d: print(json.dumps(dict(rank=rank, **d)), flush=True)  # noqa: E731
    log(dict(part=a.part, mine=len(mine), done=len(done), todo=len(todo)))
    if not todo: log(dict(part=a.part, finished=True, sec=0)); return
    cases = _read_cases(str(a.cases_jsonl).replace('{rank}', f'{rank:03d}') if a.cases_jsonl else m['cases'], set(todo))
    reps = {}
    for p in a.plan_reports: reps.update(_read_cases(p, set(todo)))

    def volume_loader(cid):
        if a.ct_dir is not None:
            return lambda sidx: np.load(Path(a.ct_dir) / ('ct_%03d.npy' % sidx), mmap_mode='r')
        return lambda sidx: load_series_volume(a.h5_root, reps[cid], reps[cid]['series'][sidx])
    writer = ChunkWriter(a.out, rank, a.plabel, ex.classes, chunk_no, bool(a.render)); t0 = time.time()
    todo = [c for c in todo if c in cases]; pool = ThreadPoolExecutor(2); fut = {c: pool.submit(ex.prepare, cases[c]) for c in todo[:2]}
    for n_, cid in enumerate(todo):
        views, vinfo = fut.pop(cid).result()
        if n_ + 2 < len(todo): fut[todo[n_ + 2]] = pool.submit(ex.prepare, cases[todo[n_ + 2]])
        clist = tg[cid]; cl = torch.tensor(clist, device=ex.dev); res = None
        for crops in (a.crops, 32, 8):
            try:
                res = ex.study(views, vinfo, cl, crops, a.block)
                break
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache(); log(dict(cid=cid, oom_retry=crops))
        if res is None: log(dict(cid=cid, skipped='oom')); continue
        images = None
        if a.render:
            images = study_finding_images(res['E'], res['vinfo'], res['fo'], [ex.classes[c] for c in clist], volume_loader(cid),
                                          log=lambda d: log(dict(cid=cid, **d)))
        writer.add(cid, c2r[cid], clist, res, a.view_min_rel, images)
        del res
        if ex.dev.type == 'cuda' and n_ % 20 == 0: torch.cuda.empty_cache()
        if n_ % 25 == 0 or a.cases: log(dict(part=a.part, done=n_ + 1, of=len(todo), sec=round(time.time() - t0)))
        if len(writer) >= a.chunk: writer.flush()
    writer.flush(); pool.shutdown(wait=False, cancel_futures=True)
    log(dict(part=a.part, finished=True, sec=round(time.time() - t0)))


def extract_ct_evidence(config, checkpoint, targets, out, **options) -> None:
    """Library entry of the extractor (inference): ``options`` are the CLI options by their argument names, e.g.
    ``cases=Path("case_ids.txt"), cases_jsonl=..., cache_root=..., region_root=..., ct_dir=..., block=2, render=1,
    crops=32, chunk=1, device="cuda"``."""
    a = build_parser().parse_args(['--config', str(config), '--ckpt', str(checkpoint), '--targets', str(targets), '--out', str(out)])
    unknown = set(options) - set(vars(a))
    if unknown:
        raise TypeError(f'unknown extractor options {sorted(unknown)}')
    vars(a).update(options)
    a.out = Path(a.out)
    run(a)


def main(argv=None):
    run(build_parser().parse_args(argv))


if __name__ == '__main__':
    main()

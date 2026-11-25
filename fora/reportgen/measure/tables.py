"""Candidate / target tables for the lesion calipers without report information (paper: Methods, "Measurements").

One candidate (the strongest MIL evidence peak) per (study, predicted measurable class).  Class-level fields (measurement type mt,
HU group grp, scope class kidx, class region bits, is_node, compared value ycmp) are the modal values of that class among the
training measurement targets, and the crop size prior is the class's training median report size (15 mm if none) — the class
schema (data/measure/measurement_schema.json).  y_long / y_short are NaN (no report).

Builders:
  schema      measurement targets npz (cls, mt, keep, split, grp, kidx, bits, is_node, ycmp, y_long) + measurable classes {cls: mt}
              -> class schema JSON
  peaks       per-node MIL peak shards (peaksN_r*.npz: pair_case, cls, prob, logit, pk3 [n,1,5] = (view, z, y, x, evidence),
              pk3_world, pkbits, pkx, cb_case / cb_cls / cb_rank + cb_x / cb_mask / cb_off token blocks) -> targets_local.npz,
              cand_local.npz (training-set measurement text)
  extent      evidence-extractor extent files (extent_*.npz: cls, pair_case, peak, prob, logit, blk_view, blk_mask, blk_x, blk_off)
              -> targets.npz, cand.npz, context.npz (inference; lesion_measure.py)
usage: python -m fora.reportgen.measure.tables schema --targets T.npz --measurable M.json --out measurement_schema.json
       python -m fora.reportgen.measure.tables peaks --schema S.json PEAK_DIR OUT_DIR
"""
import argparse
import collections
import glob
import json
import os

import numpy as np

XN = ['inroute', 'lmass', 'n25', 'd_pleura', 'd_hila', 'd_mediastinum', 'd_outroute', 'd_nonlung', 'off_R', 'off_S', 'off_A', 'top_ev']


def class_schema(T2, measurable):
    """targets npz dict + {cls: mt} -> {cls: dict(mt, grp, kidx, bits, node, ycmp, prior)}."""
    info = {}
    for c, m in measurable.items():
        c, m = int(c), int(m)
        s = (T2['cls'] == c) & (T2['mt'] == m) & T2['keep']; tr = s & (T2['split'] == 'train')
        if not s.any(): continue
        md = lambda a: collections.Counter(a[s].tolist()).most_common(1)[0][0]
        y = T2['y_long'][tr].astype(float); y = y[np.isfinite(y)]
        info[c] = dict(mt=m, grp=int(md(T2['grp'])), kidx=int(md(T2['kidx'])), bits=int(md(T2['bits'])), node=bool(md(T2['is_node'])),
                       ycmp=int(md(T2['ycmp'])), prior=float(np.median(y)) if len(y) else 15.)
    return info


def _target(T, case, c, I, prob, logit):
    for k, v in (('key', f'{case}|{c}'), ('split', 'all'), ('cls', c), ('mt', I['mt']), ('case', case), ('y_long', np.nan), ('y_short', np.nan), ('fov_size_mm', I['prior']),
                 ('sh_pred', np.nan), ('sh_src', 0), ('lat', 0), ('eps_long', 1.), ('eps_short', 1.), ('grp', I['grp']), ('is_node', I['node']), ('kidx', I['kidx']),
                 ('bits', I['bits']), ('tsplit', 'train'), ('prob', float(prob)), ('logit', float(logit)), ('has_peaks', True), ('keep', True), ('ycmp', I['ycmp']), ('seg', 0)):
        T[k].append(v)


def tables_from_peaks(peak_dir, info):
    """training-time tables from the MIL peak shards (one candidate per (study, class): the top peak)."""
    files = sorted(glob.glob(peak_dir + '/peaksN_r*.npz')); rows = collections.defaultdict(list); T = collections.defaultdict(list)
    for fi, f in enumerate(files):
        z = np.load(f); CB = {(str(c), int(k), int(r)): j for j, (c, k, r) in enumerate(zip(z['cb_case'], z['cb_cls'], z['cb_rank']))}
        for j, (case, c) in enumerate(zip(z['pair_case'], z['cls'])):
            c = int(c); case = str(case)
            if c not in info or z['pk3'][j][0, 0] < 0: continue
            I = info[c]; t = len(T['key']); pk = z['pk3'][j][0]
            _target(T, case, c, I, z['prob'][j], z['logit'][j])
            cbj = CB.get((case, c, 0), -1)
            for k, v in (('cbf', fi if cbj >= 0 else -1), ('cbj', cbj), ('row', len(rows['row'])), ('t', t), ('rank', 0), ('case', case), ('view', int(pk[0])), ('peak', pk[1:4].astype(np.int64)),
                         ('world', z['pk3_world'][j][0]), ('ev', pk[4]), ('ev_rel', 1.), ('bits', z['pkbits'][j][0]), ('x', z['pkx'][j][0]), ('fov_size_mm', I['prior']), ('cls', c),
                         ('tsplit', 'train'), ('mt', I['mt']), ('grp', I['grp']), ('kidx', I['kidx'])):
                rows[k].append(v)
    Tn = {k: np.array(v) for k, v in T.items()}; Cn = {k: np.array(v) for k, v in rows.items()}; Cn['xnames'] = np.array(XN); Cn['cb_files'] = np.array(files)
    return Tn, Cn


def tables_from_extent(files, schema, context_path):
    """inference tables from the evidence-extractor extent files: block 0 (strongest focus) of every measurable class.
    Writes the candidate token blocks to context_path (uncompressed npz) and returns (targets, candidates)."""
    target = collections.defaultdict(list); cand = collections.defaultdict(list); blocks = []; masks = []; offsets = []
    for fn in files:
        with np.load(fn) as z:
            for j, c in enumerate(z['cls']):
                c = int(c)
                if str(c) not in schema: continue
                info = schema[str(c)]; bidx = 0
                if int(z['blk_view'][j, bidx]) < 0 or not z['blk_mask'][j, bidx].any(): continue
                i = len(target['cls']); case = str(z['pair_case'][j]); vi = int(z['blk_view'][j, bidx]); peak = z['peak'][j].astype(np.int64)
                _target(target, case, c, info, z['prob'][j], z['logit'][j])
                for k, v in dict(cbf=0, cbj=i, row=i, t=i, rank=0, case=case, view=vi, peak=peak, fov_size_mm=info['prior'], cls=c,
                                 tsplit='train', mt=info['mt'], grp=info['grp'], kidx=info['kidx']).items(): cand[k].append(v)
                blocks.append(z['blk_x'][j, bidx]); masks.append(z['blk_mask'][j, bidx]); offsets.append(z['blk_off'][j, bidx])
    if not blocks: return None, None
    np.savez(context_path, cb_x=np.stack(blocks), cb_mask=np.stack(masks), cb_off=np.stack(offsets))
    C = {k: np.array(v) for k, v in cand.items()}; C['cb_files'] = np.array([str(context_path)])
    T = {k: np.array(v) for k, v in target.items()}
    return T, C


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('schema'); s.add_argument('--targets', required=True); s.add_argument('--measurable', required=True, help='JSON {"measurable": {cls: mt}}'); s.add_argument('--out', required=True)
    p = sub.add_parser('peaks'); p.add_argument('--schema', required=True); p.add_argument('peak_dir'); p.add_argument('out_dir')
    a = ap.parse_args()
    if a.cmd == 'schema':
        T2 = np.load(a.targets, allow_pickle=True); MC = json.load(open(a.measurable))['measurable']
        json.dump(class_schema(T2, MC), open(a.out, 'w'), indent=2); return
    info = {int(k): v for k, v in json.load(open(a.schema)).items()}
    Tn, Cn = tables_from_peaks(a.peak_dir, info)
    os.makedirs(a.out_dir, exist_ok=True); np.savez(a.out_dir + '/targets_local.npz', **Tn); np.savez(a.out_dir + '/cand_local.npz', **Cn)
    print(json.dumps(dict(targets=len(Tn.get('key', [])), cands=len(Cn.get('row', [])), ctx=float((Cn['cbf'] >= 0).mean()) if len(Cn.get('row', [])) else 0)))


if __name__ == '__main__':
    main()

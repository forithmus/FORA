"""Extraction plan: every report x every usable series x views, balanced over encoding workers (Methods, "CT preprocessing").

For each report group of every CT H5 shard that has a `report/` group, every usable series is listed
(fora.geometry.series: derivative preference, 3-D / 4-D frames, sane affine) with its views
(fora.geometry.windows.plan_views: FULL + distinct native windows + the 10 mm lung MIP, training rule by default).
Reports get ids `<split>_<NNNNNN>` in (h5, patient, report) order within each split of the shard-level split file.
Reports are assigned to workers by longest-processing-time balancing of an encoding cost
(number of 96 mm crops x views).

Usage
  one pass:        python plan_extraction.py --h5-dir H5 --splits splits.csv --out OUT --workers 128
  scan in parts:   python plan_extraction.py ... scan --part K --nparts N --procs P     (one call per part)
  merge the parts: python plan_extraction.py ... merge --nparts N
Outputs OUT/reports_{train,val}.jsonl and OUT/assign_{train,val}.json.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import h5py
import numpy as np

from fora.geometry.series import dec, pick_source, sane_affine, valid_frames
from fora.geometry.windows import native_windows, plan_views

COST_CROP_MM = 96


def scan(fn: str, mip_rule: str = "training") -> list[dict]:
    """All planned reports of one CT H5 shard."""
    out = []
    try:
        f = h5py.File(fn, 'r')
    except Exception:
        return out
    with f:
        for pat in f:
            pg = f[pat]
            if not hasattr(pg, 'keys'):
                continue
            for rpt in pg:
                rg = pg[rpt]
                if not hasattr(rg, 'keys') or 'report' not in rg:
                    continue
                series = []
                for acc in rg:
                    if acc == 'report':
                        continue
                    ag = rg[acc]
                    if not hasattr(ag, 'keys') or ag.attrs.get('kind') == 'report':
                        continue
                    for sid in ag:
                        sg = ag[sid]
                        if not hasattr(sg, 'keys'):
                            continue
                        sub, dd = pick_source(sg)
                        if dd is None:
                            continue
                        frames = valid_frames(dd.shape)
                        if not frames:
                            continue
                        src = sg[sub] if sub else sg
                        if 'affine' not in src:
                            continue
                        A = src['affine'][()]
                        if not sane_affine(A):
                            continue
                        protocols = dec(str(sg.attrs.get('classified_protocols', '')))
                        views, mip = plan_views(native_windows(sg), mip_rule=mip_rule,
                                                lung_protocol=_lung_protocol(protocols) if mip_rule == "inference" else None)
                        sp = np.linalg.norm(A[:3, :3], axis=0)
                        ext = np.array(dd.shape[:3]) * sp
                        crops = int(np.prod([-(-e // COST_CROP_MM) for e in ext]))
                        for fr in frames:
                            series.append(dict(acc=acc, sid=sid, sub=sub, frame=fr, windows=views, mip=mip,
                                               cost=crops * (len(views) + (mip is not None)), protocols=protocols))
                if series:
                    out.append(dict(h5=os.path.basename(fn)[:-3], patient=pat, report=rpt, series=series))
    return out


def _lung_protocol(protocols: str) -> bool:
    """Series attribute `classified_protocols` (JSON list or a single title) -> any lung-CT protocol."""
    from fora.geometry.protocols import LUNG_CT_CLASSES, norm_class
    try:
        titles = json.loads(protocols)
        titles = titles if isinstance(titles, list) else [titles]
    except (ValueError, TypeError):
        titles = [protocols]
    return any(norm_class(t) in LUNG_CT_CLASSES for t in titles)


def write_plan(reports: list[dict], out: str, workers: int):
    for sp in ['train', 'val']:
        rs = [r for r in reports if r['split'] == sp]
        for k, r in enumerate(rs):
            r['id'] = f"{sp}_{k:06d}"
        with open(f'{out}/reports_{sp}.jsonl', 'w') as fo:
            for r in rs:
                fo.write(json.dumps(r) + '\n')
        order = sorted(range(len(rs)), key=lambda k: -sum(s['cost'] for s in rs[k]['series']))
        load = np.zeros(workers)
        assign = [[] for _ in range(workers)]
        for k in order:
            w = int(np.argmin(load))
            assign[w].append(rs[k]['id'])
            load[w] += sum(s['cost'] for s in rs[k]['series'])
        json.dump(dict(workers=workers, assignments=assign, load=load.tolist()), open(f'{out}/assign_{sp}.json', 'w'))
        v = sum(len(s['windows']) + (s['mip'] is not None) for r in rs for s in r['series'])
        print(json.dumps(dict(split=sp, reports=len(rs), series=sum(len(r['series']) for r in rs), views=v,
                              cost=float(load.sum()), max_over_mean=float(load.max() / max(load.mean(), 1e-9)))), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--h5-dir', required=True, help='directory of CT H5 shards batch_*.h5')
    ap.add_argument('--splits', required=True, help='CSV with columns h5_id, split (train / val) per shard')
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=128)
    ap.add_argument('--mip-rule', choices=['training', 'inference'], default='training')
    ap.add_argument('mode', nargs='?', choices=['all', 'scan', 'merge'], default='all')
    ap.add_argument('--part', type=int, default=0)
    ap.add_argument('--nparts', type=int, default=1)
    ap.add_argument('--procs', type=int, default=24)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    split = {r['h5_id']: r['split'].strip() for r in csv.DictReader(open(a.splits))}
    files = sorted(glob.glob(a.h5_dir + '/batch_*.h5'))
    fn = partial(scan, mip_rule=a.mip_rule)
    reports = []
    if a.mode == 'scan':
        os.makedirs(a.out + '/parts', exist_ok=True)
        tmp = f'{a.out}/parts/part_{a.part:03d}.tmp'
        with ProcessPoolExecutor(a.procs) as ex, open(tmp, 'w') as fo:
            for rows in ex.map(fn, files[a.part::a.nparts]):
                for r in rows:
                    r['split'] = split.get(r['h5'], '?')
                    fo.write(json.dumps(r) + '\n')
        os.replace(tmp, f'{a.out}/parts/part_{a.part:03d}.jsonl')
        return
    if a.mode == 'merge':
        for k in range(a.nparts):
            for line in open(f'{a.out}/parts/part_{k:03d}.jsonl'):
                reports.append(json.loads(line))
        reports.sort(key=lambda r: (r['h5'], r['patient'], r['report']))
    else:
        with ProcessPoolExecutor(a.procs) as ex:
            for i, rows in enumerate(ex.map(fn, files, chunksize=2)):
                for r in rows:
                    r['split'] = split.get(r['h5'], '?')
                    reports.append(r)
                if i % 500 == 0:
                    print('scanned', i, len(files), len(reports), flush=True)
    write_plan(reports, a.out, a.workers)


if __name__ == '__main__':
    main()

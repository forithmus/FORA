"""Scan-level matching of CT-RATE validation volumes to FORA series by DICOM metadata (Methods, "Data splits").

Key: StudyDate, PatientSex, PatientAge, Manufacturer (lower-case first word), ConvolutionKernel, ReconstructionDiameter
(0.1 mm) and the in-plane origin (ImagePositionPatient x, y, 0.1 mm). Tube current and exposure are not in the key
(they vary per slice under dose modulation). Only FORA series whose name carries a study date that occurs among the
CT-RATE validation study dates are read. Output `match.json`: per CT-RATE volume the matching FORA series as
[plan split, report id, ReportUID, series id], plus volume- and case-level summaries.

Usage
  python match_ctrate.py --h5-dir H5 --plan PLAN_DIR --ctrate-metadata validation_metadata.csv --out OUT_DIR
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import re
from functools import partial
from multiprocessing import Pool

import h5py


def num(x):
    try:
        return round(float(str(x).strip().strip('"')), 1)
    except Exception:
        return None


def ipp_xy(s):
    try:
        v = json.loads(s) if isinstance(s, str) else s
        return (round(float(v[0]), 1), round(float(v[1]), 1))
    except Exception:
        return None


def key(date, sex, age, manu, kernel, rdiam, xy):
    return (str(date).strip(), str(sex).strip().upper(), str(age).strip().upper(), (str(manu).strip().split() or [''])[0].lower(),
            str(kernel).strip().upper(), num(rdiam), xy)


def ctrate_keys(path: str) -> dict:
    K = {}
    for r in csv.DictReader(open(path)):
        K[r['VolumeName']] = key(r['StudyDate'], r['PatientSex'], r['PatientAge'], r['Manufacturer'], r['ConvolutionKernel'],
                                 r['ReconstructionDiameter'], ipp_xy(r['ImagePositionPatient']))
    return K


def scan_file(args, h5_dir):
    h5, items = args
    out = []
    try:
        with h5py.File(f'{h5_dir}/{h5}.h5', 'r') as f:
            for split, rid, rep, pat, acc, sid in items:
                try:
                    g = f[pat][rep][acc][sid]
                    p = json.loads(g['pacs_metadata'][()])
                    d = json.loads(g['dcm2niix_metadata'][()])
                    k = key(p.get('StudyDate'), p.get('PatientSex'), p.get('PatientAge'), p.get('Manufacturer'),
                            p.get('ConvolutionKernel'), p.get('ReconstructionDiameter'), ipp_xy(d.get('ImagePositionPatient')))
                    out.append((k, split, rid, rep, sid))
                except Exception:
                    out.append((None, split, rid, rep, sid))
    except Exception:
        pass
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--h5-dir', required=True)
    ap.add_argument('--plan', required=True, help='directory with reports_{train,val}.jsonl from plan_extraction.py')
    ap.add_argument('--ctrate-metadata', required=True, help='CT-RATE validation_metadata.csv')
    ap.add_argument('--out', required=True)
    ap.add_argument('--procs', type=int, default=24)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    K = ctrate_keys(a.ctrate_metadata)
    dates = {k[0] for k in K.values()}
    print('CT-RATE valid volumes', len(K), 'distinct study dates', len(dates), flush=True)
    by = collections.defaultdict(list)
    for split in ('train', 'val'):
        for line in open(f'{a.plan}/reports_{split}.jsonl'):
            r = json.loads(line)
            for s in r['series']:
                m = re.search(r'_(\d{8})\d{6}_', s['sid'])
                if m and m.group(1) in dates:
                    by[r['h5']].append((split, r['id'], r['report'], r['patient'], s['acc'], s['sid']))
    idx = collections.defaultdict(list)
    with Pool(a.procs) as pool:
        for res in pool.imap_unordered(partial(scan_file, h5_dir=a.h5_dir), list(by.items()), chunksize=4):
            for k, split, rid, rep, sid in res:
                if k is not None:
                    idx[k].append((split, rid, rep, sid))
    res = {}
    c = collections.Counter()
    for v, k in K.items():
        m = idx.get(k, [])
        res[v] = m
        c['matched' if m else 'unmatched'] += 1
        if m:
            c['in_train' if any(x[0] == 'train' for x in m) else 'in_val_only'] += 1
    cases = collections.defaultdict(list)
    for v, m in res.items():
        cases[re.sub(r'_[a-z]_\d+\.nii\.gz$', '', v)].append(m)
    cc = collections.Counter()
    for _, ms in cases.items():
        s = {x[0] for m in ms for x in m}
        cc['case_in_fora_train' if 'train' in s else ('case_in_fora_val_only' if s else 'case_clean')] += 1
    json.dump(dict(volumes=res, summary_volumes=c, summary_cases=cc), open(f'{a.out}/match.json', 'w'), indent=0)
    print(json.dumps(dict(volumes=c, cases=cc)), flush=True)


if __name__ == '__main__':
    main()

"""Case list for the CADS anatomy measurer (paper: Methods, "Measurements"): one JSON line per study.

Line: {"case": fora_id, "report": report_uid, "split", "h5": batch id, "patient", "series": [[acc, sid], ...], "classes": [cls, ...]}
  classes  = measurement classes predicted for the study: calibrated prob = sigmoid(logit * class_slope + class_intercept)
             >= class_possible_threshold (dense MIL predictions + MIL calibration), restricted to the CADS measurement classes.
  series   = provenance of the study's CT series (needed to open the CADS group patient/report/acc/sid and the H5 volume):
             index sqlite `samples.series_json` (union over protocol rows), fallback = the CT ingestion manifest (acc, basename(img_path)).
usage:
  python -m fora.reportgen.measure.cads.caselist --dense dense.npz --calibration calib.json --index index.sqlite [--manifest m.csv] --out cases_pred.jsonl
  ... --all                     # every study (reference measurements for all; used for the study measurement line)
  ... --items items.json        # explicit (case, classes) list: JSON [[case, [cls, ...]], ...]
The dense prediction file holds `logit` [n_studies, n_classes], `fora_id` and `report_uid`.
"""
import argparse
import collections
import csv
import json
import os
import sqlite3

import numpy as np

from .classes import RG_CLASSES


def provenance(reports, index_db, manifest=None):
    """report_uid -> dict(h5, patient, series)"""
    db = sqlite3.connect(f'file:{index_db}?mode=ro', uri=True)
    P = {}
    for rid, pat, h5, sj in db.execute('select report_uid, patient_uid, h5_id, series_json from samples'):
        if rid not in reports: continue
        d = P.setdefault(rid, dict(h5=h5, patient=pat, series=[]))
        for acc, sid in json.loads(sj):
            if [acc, sid] not in d['series']: d['series'].append([acc, sid])
    miss = reports - set(P)
    if miss and manifest:
        csv.field_size_limit(1 << 30)
        with open(manifest, newline='') as f:
            for row in csv.DictReader(f):
                rid = row['ReportUID']
                if rid not in miss: continue
                d = P.setdefault(rid, dict(h5=row['h5_id'], patient=row['PatientUID'], series=[], src='manifest'))
                s = [row['AccessionNumber'], os.path.basename(row['img_path'].rstrip('/'))]
                if s not in d['series']: d['series'].append(s)
    return P


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--all', action='store_true')
    ap.add_argument('--items', default=''); ap.add_argument('--dense', required=True); ap.add_argument('--calibration', default='')
    ap.add_argument('--index', required=True); ap.add_argument('--manifest', default='')
    a = ap.parse_args()
    Z = np.load(a.dense, allow_pickle=True)
    fid = Z['fora_id']; rid = Z['report_uid']
    if a.items:
        want = {c: sorted(set(int(x) for x in cl)) for c, cl in json.load(open(a.items))}
        k = np.flatnonzero(np.isin(fid, list(want)))
        cls_of = {fid[i]: want[fid[i]] for i in k}
    else:
        C = json.load(open(a.calibration))
        cl = np.array(sorted(RG_CLASSES)); sl = np.array(C['class_slope'])[cl]; ic = np.array(C['class_intercept'])[cl]
        th = np.array(C['class_possible_threshold'])[cl]
        L = Z['logit'][:, cl].astype(np.float64); pr = 1 / (1 + np.exp(-(L * sl + ic))) >= th
        k = np.arange(len(fid)) if a.all else np.flatnonzero(pr.any(1))
        cls_of = {fid[i]: cl[pr[i]].tolist() for i in k}
    P = provenance({rid[i] for i in k}, a.index, a.manifest or None)
    n = 0; nmiss = 0; src = collections.Counter()
    with open(a.out, 'w') as fo:
        for i in k:
            p = P.get(rid[i])
            if p is None: nmiss += 1; continue
            src[p.get('src', 'sqlite')] += 1
            fo.write(json.dumps(dict(case=str(fid[i]), report=str(rid[i]), split=str(fid[i]).split('_')[0], h5=p['h5'], patient=p['patient'],
                                     series=p['series'], classes=[int(c) for c in cls_of[fid[i]]])) + '\n'); n += 1
    print('cases', n, 'no provenance', nmiss, dict(src), '->', a.out)


if __name__ == '__main__':
    main()

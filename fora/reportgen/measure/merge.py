"""Per-finding measurement text of the training cohort: {"<case>|<cls>": text} (paper: Methods, "Measurements").

Inputs: PULL_DIR/node_XX/{targets_local.npz (tables.py peaks), cal_<group>/cal_r*.npz (caliper.py infer per finding group),
cads_*.jsonl (cads/measure.py)}.
Lesion / thickness / lumen value = the finding group's caliper on the MIL top candidate, divided by the group's per-class scale
calibration exp(offset) (data/measure/caliper_scale_<group>.json); value form by class (nodule mean diameter, node short axis,
long x short, thickness, lumen diameter) from the finding-description module.  Vessel / organ / midline values come from the CADS
records (calibrated by the CADS measurer) and replace a caliper value of the same (case, class).
usage: python -m fora.reportgen.measure.merge PULL_DIR --scales data/measure --out finding_measurements.json
"""
import argparse
import collections
import glob
import json
import math

import numpy as np

GROUPS = ('nodule', 'solid', 'stone', 'cyst', 'node', 'thick', 'air')
SEGMENT_NAME = {'asc': 'ascending aorta', 'root': 'aortic root', 'thor': 'thoracic aorta (maximum)', 'abd': 'abdominal aorta', 'pa': 'main pulmonary artery',
                'arch': 'aortic arch', 'desc': 'descending thoracic aorta'}


def cads_record_text(rec: dict):
    """Writer text of one CADS measurement record (vessel diameter, organ size, midline shift), None for other kinds or
    a missing value."""
    from fora.reportgen.evidence.describe import fmt_mm, measurement_phrase
    v = rec.get('value_mm')
    if v is None or not math.isfinite(v): return None
    kind = rec.get('kind')
    if kind == 'vessel_diameter':
        seg = SEGMENT_NAME.get(rec.get('segment_used') or rec.get('segment'), 'aorta')
        return measurement_phrase(2, 0, False, v, float('nan'), float('nan'), vessel_segment=seg)
    if kind == 'organ_size' and rec.get('structure') == 'kidneys':
        return '; '.join(x for x in (f"right kidney length about {fmt_mm(rec['R_mm'])} mm" if rec.get('R_mm') else '',
                                     f"left kidney length about {fmt_mm(rec['L_mm'])} mm" if rec.get('L_mm') else '') if x)
    if kind == 'organ_size':
        return f"{rec['structure']} {rec.get('measure', 'length')} about {fmt_mm(v)} mm" + (' (organ cut off by the scan range: lower bound)' if rec.get('truncated') else '')
    if kind == 'midline_shift':
        return f"midline shift about {fmt_mm(v)} mm" + (f" toward the {rec['direction']}" if rec.get('direction') else '') + ' (approximate)'
    return None


def merge(pull_dir, scales_dir):
    from fora.reportgen.evidence.describe import measurement_phrase
    CAL = {g: json.load(open(f'{scales_dir}/caliper_scale_{g}.json')) for g in GROUPS}
    meas = {}; nlesion = collections.Counter()
    for nd in sorted(glob.glob(pull_dir + '/node_*')):
        T = np.load(nd + '/targets_local.npz', allow_pickle=True)
        for g in GROUPS:
            for f in glob.glob(f'{nd}/cal_{g}/cal_r*.npz'):
                z = np.load(f)
                for t, L, S_, I, emp in zip(z['t'], z['long'], z['short'], z['ins'], z['empty']):
                    if emp: continue
                    c = int(T['cls'][t]); mt = int(T['mt'][t]); yc = int(T['ycmp'][t]); node = bool(T['is_node'][t])
                    off = CAL[g].get(f'{c}_{mt}', CAL[g].get(f't{mt}', 0.)); k = math.exp(off)
                    txt = measurement_phrase(mt, yc, node, float(L) * k, float(S_) * k, float(I) * k)
                    if txt: meas[(str(T['case'][t]), c)] = txt; nlesion[g] += 1
        for f in glob.glob(nd + '/cads_*.jsonl'):
            for l in open(f):
                r = json.loads(l)
                for rec in r.get('records', []):
                    txt = cads_record_text(rec)
                    if txt is None: continue
                    kind = rec.get('kind'); c = int(rec['cls'])
                    meas[(r['case'], c)] = txt; nlesion['cads_' + kind] += 1
    print('measurements', len(meas), dict(nlesion), flush=True)
    return {f'{k[0]}|{k[1]}': v for k, v in meas.items()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('pull_dir'); ap.add_argument('--scales', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--extra', action='append', default=[], help='further {case|cls: text} JSON whose keys are added when not already present (the Evans / CTR texts of cads/study.py)')
    a = ap.parse_args()
    m = merge(a.pull_dir, a.scales)
    for e in a.extra:
        for k, v in json.load(open(e)).items(): m.setdefault(k, v)
    json.dump(m, open(a.out, 'w'))


if __name__ == '__main__':
    main()

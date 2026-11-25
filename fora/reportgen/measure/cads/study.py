"""Study-level reference measurement line and per-finding Evans-index / cardiothoracic-ratio texts from the CADS measurer output
(paper: Methods, "Measurements").

(1) study line per report: {report_uid: "ascending aorta 34 mm, pulmonary trunk 25 mm, cardiothoracic ratio 0.48, ..."}
    (plausibility ranges RNG; organs cut off by the scan range are left out), given to the writer as the study measurement line;
(2) per-finding texts {case|cls: text} for the Evans-index and cardiothoracic-ratio classes.
Records / reference values are recomputed from the raw measurer output with measure.records().
Several raw outputs may be given; later files win per task (e.g. a re-run of only midline + ctr: --raw "new_*.jsonl:midline,ctr").
usage: python -m fora.reportgen.measure.cads.study --cases cases_all.jsonl --raw "cads_*.jsonl" --calib data/measure/cads_calibration.json
         --out-study study_measurements.json --out-findings study_finding_measurements.json"""
import argparse, glob, json, collections, os
import numpy as np

from . import measure as M

RNG = dict(ascending_aorta_mm=(15, 70), pulmonary_trunk_mm=(12, 50), abdominal_aorta_mm=(8, 80), liver_cc_mm=(80, 300),
           spleen_length_mm=(40, 250), kidney_R_length_mm=(50, 180), kidney_L_length_mm=(50, 180), cardiothoracic_ratio=(0.3, 0.85),
           evans_index=(0.0, 0.9))


def study_texts(C, RAW, cal, mid_none=2.5):
    """C {case: case-list line}, RAW {case: raw measurer output} -> (study lines {report: text}, per-finding Evans / CTR
    texts {case|cls: text}, coverage counter, number of cases with results)."""
    def ok(ref, k):
        v = ref.get(k); lo, hi = RNG[k]
        return v is not None and np.isfinite(v) and lo <= v <= hi
    fm = lambda v: f'{v:.0f}'
    study = {}; meas = {}; cov = collections.Counter(); ncase = 0
    for case, c in C.items():
        raw = RAW.get(case)
        if raw is None: continue
        ncase += 1
        try: recs, ref = M.records(c, raw, cal)
        except Exception as e:
            cov["records_error"] += 1; continue
        tr = set(ref.get('organs_truncated_by_fov', []))
        parts = []
        if ok(ref, 'ascending_aorta_mm'): parts.append(f"ascending aorta {fm(ref['ascending_aorta_mm'])} mm"); cov['ascending_aorta'] += 1
        if ok(ref, 'pulmonary_trunk_mm'): parts.append(f"pulmonary trunk {fm(ref['pulmonary_trunk_mm'])} mm"); cov['pulmonary_trunk'] += 1
        if ok(ref, 'cardiothoracic_ratio') and 'heart' not in tr: parts.append(f"cardiothoracic ratio {ref['cardiothoracic_ratio']:.2f}"); cov['cardiothoracic_ratio'] += 1
        if ok(ref, 'abdominal_aorta_mm'): parts.append(f"abdominal aorta {fm(ref['abdominal_aorta_mm'])} mm"); cov['abdominal_aorta'] += 1
        if ok(ref, 'liver_cc_mm') and 'liver' not in tr: parts.append(f"liver {fm(ref['liver_cc_mm'])} mm (craniocaudal)"); cov['liver'] += 1
        if ok(ref, 'spleen_length_mm') and 'spleen' not in tr: parts.append(f"spleen {fm(ref['spleen_length_mm'])} mm"); cov['spleen'] += 1
        for s, nm in (('R', 'right'), ('L', 'left')):
            if ok(ref, f'kidney_{s}_length_mm') and f'kidney_{s}' not in tr:
                parts.append(f"{nm} kidney {fm(ref[f'kidney_{s}_length_mm'])} mm"); cov[f'{nm}_kidney'] += 1
        if ok(ref, 'evans_index'):
            parts.append(f"Evans index {ref['evans_index']:.2f}" if ref['evans_index'] >= M.EVANS_MIN else 'Evans index < 0.25 (ventricles not enlarged)')
            cov['evans_index'] += 1; cov['evans_index_ge025'] += ref['evans_index'] >= M.EVANS_MIN
        Mid = raw.get('midline')
        if Mid and 'fail' not in Mid:
            y, x, sgn, src = M.midline_value(Mid, cal['midline']['336'])
            if y is not None and np.isfinite(y):
                parts.append('midline shift none' if x < mid_none else f"midline shift about {np.round(y, 0):.0f} mm" + (f" toward the {'right' if sgn > 0 else 'left'}" if sgn is not None else ''))
                cov['midline_shift'] += 1; cov['midline_shift_present'] += x >= mid_none
        if parts:
            study[c['report']] = ', '.join(parts); cov['nonempty'] += 1
        for r in recs:
            k = r.get('kind'); key = f"{case}|{r['cls']}"
            if k == 'evans_index' and r.get('value') is not None and RNG['evans_index'][0] < r['value'] <= RNG['evans_index'][1]:
                meas[key] = (f"Evans index about {r['value']:.2f} (frontal horns {r['frontal_horn_mm']:.0f} mm)" if r['value'] >= M.EVANS_MIN
                             else 'Evans index < 0.25 (ventricles not enlarged)'); cov['rec_evans'] += 1; cov['rec_evans_ge025'] += r['value'] >= M.EVANS_MIN
            elif k == 'cardiothoracic_ratio' and r.get('value') is not None and RNG['cardiothoracic_ratio'][0] <= r['value'] <= RNG['cardiothoracic_ratio'][1]:
                meas[key] = f"cardiothoracic ratio about {r['value']:.2f}" + (' (heart partly outside the scan range)' if r.get('truncated') else ''); cov['rec_ctr'] += 1
    return study, meas, cov, ncase


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', required=True); ap.add_argument('--raw', action='append', required=True, help='GLOB[:task,task] (later wins per task)')
    ap.add_argument('--calib', required=True); ap.add_argument('--out-study', required=True); ap.add_argument('--out-findings', required=True)
    ap.add_argument('--mid-none', type=float, default=2.5, help="raw septum offset (mm) below which the study line says 'none' (the calibration floors at 2.6 mm)")
    a = ap.parse_args()
    OUT_S, OUT_M, MID_NONE = a.out_study, a.out_findings, a.mid_none
    cal = json.load(open(a.calib))
    C = {}
    for l in open(a.cases):
        r = json.loads(l); C[r['case']] = r
    RAW = {}
    def load(pat, only=None):
        n = 0
        for f in sorted(glob.glob(pat)):
            for l in open(f):
                try: r = json.loads(l)
                except Exception: continue
                d = RAW.setdefault(r['case'], {})
                for k, v in r['raw'].items():
                    if only is None or k in only: d[k] = v
                n += 1
        print(pat, n, flush=True)
    for spec in a.raw:
        pat, _, tasks = spec.partition(':')
        load(pat, set(tasks.split(',')) if tasks else None)
    study, meas, cov, ncase = study_texts(C, RAW, cal, MID_NONE)
    nrep = len({c['report'] for c in C.values()})
    json.dump(study, open(OUT_S + '.tmp', 'w')); os.replace(OUT_S + '.tmp', OUT_S)
    json.dump(meas, open(OUT_M, 'w'))
    st = dict(reports_total=nrep, cases_with_results=ncase, reports_nonempty=cov['nonempty'], coverage={k: round(v / nrep, 4) for k, v in cov.items()}, counts=dict(cov))
    print(json.dumps(st, indent=1)); json.dump(st, open(OUT_S.replace('.json', '_stats.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()

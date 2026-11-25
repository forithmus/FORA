"""Score CT evidence maps of any model against annotations (paper Methods 4.10).

Per-case evidence file ``<maps>/<case_id>.npz``, one of two forms per finding ``j`` (``findings`` lists the names):

* points: ``e_<j>`` (n,) non-negative evidence and ``world_<j>`` (n, 3) S, R, A mm (optional ``family_<j>`` bool);
  the regions are computed here (``ct_regions.regions``). FORA's token evidence and any model's cell-reduced map
  (``ct_regions.dense_map_to_cells``, world = cell centres) use this form;
* regions: ``cells_<j>``, ``score_<j>``, ``core_<j>`` (HDR50), ``hdr80_<j>``, ``focus_<j>`` (S, R, A mm).
A finding without an entry (or without positive evidence) scores 0 on every measure.

Optionally ``--match-size other_per_instance.jsonl`` adds equal-size overlap: as many of this model's highest
cells as the other model's HDR80 region of the same instance, with tied cells counted at their expected value.

CLI::

    python -m evaluation.localization.score_ct --cohort kits23 --annotations KiTS23/ --maps maps/ --out kits23_loc/
    (cohorts: fumpe <truth dir>, kits23 <root>, ribfrac <labels zip>, rexg <dataset.json> --segmentations <dir>)
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from evaluation.localization import ct_readers as R
from evaluation.localization.ct_metrics import aggregate, empty_instance, instance_metrics
from evaluation.localization.ct_regions import equal_size_overlap, regions


def case_regions(path: str, finding: str):
    if not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=False); names = [str(f) for f in z['findings']]
    if finding not in names:
        return None
    j = names.index(finding)
    if f'cells_{j}' in z.files:
        return dict(cells=z[f'cells_{j}'], score=z[f'score_{j}'], allowed=np.ones(len(z[f'score_{j}']), bool), core=z[f'core_{j}'].astype(bool),
                    hdr80=z[f'hdr80_{j}'].astype(bool), focus=z[f'focus_{j}'])
    return regions(z[f'e_{j}'], z[f'world_{j}'], in_family=z[f'family_{j}'] if f'family_{j}' in z.files else None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cohort', required=True, choices=['fumpe', 'kits23', 'ribfrac', 'rexg'])
    ap.add_argument('--annotations', required=True); ap.add_argument('--segmentations')
    ap.add_argument('--maps', required=True); ap.add_argument('--match-size'); ap.add_argument('--out', required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    it = {'fumpe': lambda: R.fumpe(a.annotations), 'kits23': lambda: R.kits23(a.annotations), 'ribfrac': lambda: R.ribfrac(a.annotations),
          'rexg': lambda: R.rexg(a.annotations, a.segmentations)}[a.cohort]()
    other = {}
    if a.match_size:
        for l in open(a.match_size):
            r = json.loads(l); other[(r['case_id'], r['finding'], r['instance'])] = r.get('HDR80_cell_count', 0)
    rows = []
    for case, finding, inst, mask, aff in it:
        reg = case_regions(os.path.join(a.maps, case + '.npz'), finding)
        m = instance_metrics(mask, aff, reg['focus'], reg['cells'], reg['score'], reg['core'], reg['hdr80']) if reg else empty_instance(mask, aff)
        if a.match_size and m['GT_positive']:
            m['equal_size_HDR80'] = equal_size_overlap(reg, mask, aff, other.get((case, finding, inst), 0))
        rows.append(dict(case_id=case, finding=finding, instance=inst, **m))
    with open(os.path.join(a.out, 'per_instance.jsonl'), 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    summ = {fd: aggregate([r for r in rows if r['finding'] == fd]) for fd in sorted({r['finding'] for r in rows})}
    summ['all'] = aggregate(rows)
    json.dump(summ, open(os.path.join(a.out, 'metrics.json'), 'w'), indent=1)
    print(json.dumps({k: (v['annotated_positive_cases'], v['first_focus_hit_rate']) for k, v in summ.items()}))


if __name__ == '__main__':
    main()

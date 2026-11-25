"""Lesion measurements of one or more studies at inference time (paper: Methods, "Measurements").

For every planned finding of a measurable class, the strongest evidence focus of the evidence extractor (extent files) is measured by
the caliper of the class's finding group (seven groups: air, cyst, node, nodule, solid, stone, thick), the raw caliper value is divided
by the group's per-class scale calibration exp(offset), and the measurement phrase is produced by the finding-description module.
Outputs OUT/measurements.json (one record per measured finding) and returns {"<case>|<cls>": measurement text}.

usage: python -m fora.reportgen.measure.lesion_measure --extent-dir EXT --out OUT --calipers DIR(caliper_<group>.pt) --data data/measure \\
         --cases cases.jsonl --h5-root H5 --regions-dir REG --encoder-ckpt ENC.pt --mil-ckpt MIL.pt --mil-config MIL.json \\
         --normalization norm.pt --dino2d DINOV2_L_DIR
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from .tables import tables_from_extent


def run(extent_dir, out, *, schema: dict, groups: dict, heads: dict, scales: dict, common_args, workers=2):
    """schema / groups: measurement_schema.json / caliper_groups.json content; heads: {group: caliper checkpoint};
    scales: {group: caliper_scale_<group>.json content}; common_args: CLI arguments shared by all caliper runs (--cases,
    --h5-root, --regions-dir / --region-packs, --encoder-ckpt, --mil-ckpt, --mil-config, --normalization, --dino2d,
    --ln-short-axis). Returns {"<case>|<cls>": measurement text} and writes OUT/measurements.json."""
    from fora.reportgen.evidence.describe import measurement_phrase
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    T, C = tables_from_extent(sorted(Path(extent_dir).glob('extent_*.npz')), schema, out / 'context.npz')
    if T is None:
        (out / 'measurements.json').write_text('[]\n'); return {}
    np.savez(out / 'cand.npz', **C); np.savez(out / 'targets.npz', **T)
    for group, spec in groups.items():
        if not any([int(c), int(mt)] in spec['pairs'] for c, mt in zip(T['cls'], T['mt'])): continue
        cmd = [sys.executable, '-m', 'fora.reportgen.measure.caliper', 'infer', '--out', str(out / group), '--ckpt', str(heads[group]),
               '--cand', str(out / 'cand.npz'), '--targets', str(out / 'targets.npz'), '--exclude', spec['exclude'], '--split', 'all', '--k', '1',
               '--workers', str(workers), *map(str, common_args)]
        with open(out / (group + '.log'), 'w') as log: subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True, cwd=str(Path(__file__).resolve().parents[3]))
    res = {}; records = []
    finite = lambda x: float(x) if np.isfinite(x) else None
    for group in groups:
        cal = scales[group]
        for f in sorted((out / group).glob('cal_r*.npz')):
            with np.load(f) as z:
                for t, long, short, ins, empty in zip(z['t'], z['long'], z['short'], z['ins'], z['empty']):
                    t = int(t); c = int(T['cls'][t]); mt = int(T['mt'][t]); factor = np.exp(cal.get(f'{c}_{mt}', cal.get(f't{mt}', 0.)))
                    records.append(dict(index=c, group=group, source='caliper', empty=bool(empty),
                                        long_mm=finite(long * factor) if not empty else None, short_mm=finite(short * factor) if not empty else None,
                                        inscribed_diameter_mm=finite(ins * factor) if not empty else None, calibration_multiplier=float(factor),
                                        view_index=int(C['view'][t]), peak_token_sra=C['peak'][t].tolist(), candidate_rank=0))
                    if empty: continue
                    text = measurement_phrase(mt, int(T['ycmp'][t]), bool(T['is_node'][t]), float(long) * factor, float(short) * factor, float(ins) * factor)
                    if text: res[str(T['case'][t]) + '|' + str(c)] = text
    (out / 'measurements.json').write_text(json.dumps(records, indent=2, allow_nan=False) + '\n')
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--extent-dir', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--calipers', required=True, help='directory with caliper_<group>.pt'); ap.add_argument('--data', required=True, help='data/measure')
    ap.add_argument('--workers', type=int, default=2)
    a, rest = ap.parse_known_args()
    data = Path(a.data)
    groups = json.loads((data / 'caliper_groups.json').read_text())
    r = run(a.extent_dir, a.out, schema=json.loads((data / 'measurement_schema.json').read_text()), groups=groups,
            heads={g: Path(a.calipers) / f'caliper_{g}.pt' for g in groups},
            scales={g: json.loads((data / f'caliper_scale_{g}.json').read_text()) for g in groups}, common_args=rest, workers=a.workers)
    (Path(a.out) / 'measurement_text.json').write_text(json.dumps(r, indent=1) + '\n'); print(len(r), 'measurements')


if __name__ == '__main__':
    main()

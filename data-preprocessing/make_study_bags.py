"""Study-bag manifest and byte-balanced rank assignment (Methods, "Study bags").

Reads the extraction journals of one or more store directories and keeps every complete report with at least one view.
One report is one study bag: all its series and all their views. Writes

  cases.jsonl   one JSON line per report: id, split, report, fingerprint, series[] (pack file, offset, file_size,
                relative_file, ...), views[] (series, group, window, n_tokens, n_crops, grid, ...)
  assign.json   {fingerprint, ranks, ranks_per_node, assignments: {train: [[ids] x ranks], val: [...]}}

Reports are assigned to ranks per split by greedy byte balancing (largest bag first to the least-loaded rank). The
`split` stored here is the shard-level split of the plan; trainers re-partition each rank's reports with split.json
(make_split.py).

Usage
  python make_study_bags.py --manifest manifest.json --store STORE [--store STORE2 ...] --out OUT_DIR
         [--ranks 128 --ranks-per-node 4]
"""
from __future__ import annotations

import argparse
import glob
import heapq
import json
from pathlib import Path


def collect(stores) -> tuple[dict, int]:
    cases = {}
    failed = 0
    for d in stores:
        for f in sorted(glob.glob(f'{d}/index_worker_*.json')):
            j = json.load(open(f))
            failed += len(j.get('failed', {}))
            for cid, c in j['cases'].items():
                if c.get('complete') and c['views']:
                    cases[cid] = c
    return cases, failed


def assign(cases: dict, ranks: int) -> dict:
    out = {}
    for split in ('train', 'val'):
        cs = sorted((c for c in cases.values() if c['split'] == split), key=lambda c: -sum(s['file_size'] for s in c['series']))
        h = [(0, r) for r in range(ranks)]
        a = [[] for _ in range(ranks)]
        for c in cs:
            b, r = heapq.heappop(h)
            a[r].append(c['id'])
            heapq.heappush(h, (b + sum(s['file_size'] for s in c['series']), r))
        out[split] = a
        loads = sorted(b for b, _ in h)
        print(split, len(cs), 'reports; per-rank TB min/max %.2f/%.2f' % (loads[0] / 1e12, loads[-1] / 1e12), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--store', action='append', required=True, help='store directory with index_worker_*.json (repeatable)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--ranks', type=int, default=128)
    ap.add_argument('--ranks-per-node', type=int, default=4)
    a = ap.parse_args()
    fp = json.load(open(a.manifest))['fingerprint']
    cases, failed = collect(a.store)
    print('stores', len(a.store), 'cases', len(cases), 'failed series', failed, flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'cases.jsonl', 'w') as fo:
        for c in cases.values():
            fo.write(json.dumps(c) + '\n')
    json.dump(dict(fingerprint=fp, ranks=a.ranks, ranks_per_node=a.ranks_per_node, assignments=assign(cases, a.ranks)),
              open(out / 'assign.json', 'w'))


if __name__ == '__main__':
    main()

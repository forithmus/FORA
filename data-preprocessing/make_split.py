"""Patient-level train / validation split with every CT-RATE validation study in FORA validation (Methods, "Data splits").

Start: the shard-level split (the `train_` / `val_` prefix of the report ids). Every FORA report that matches a CT-RATE
validation volume (match_ctrate.py), together with ALL reports of the same patient, goes to validation. To keep the
split sizes, the same number of reports moves from validation to training: whole validation patients none of whose
reports match CT-RATE validation and all of whose reports are non-contrast chest CT (protocol class 7 of the protocol
classifier), drawn in random order (seed 0).

Inputs
  --cases             cases.jsonl (make_study_bags.py): report id `id` and ReportUID `report`
  --labeled           optional text file with one report id per line: restrict to reports in the label set; its
                      order is the report order used for the seeded draw (default: the order of cases.jsonl)
  --report-protocols  CSV with ReportUID, PatientUID, protocol_class (one row per report x class;
                      report_protocol_classes.py)
  --match             match.json (match_ctrate.py)
Outputs OUT/split.json {report id: 'train' | 'val'} and OUT/summary.json.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import random
from pathlib import Path

CHEST_NON_CONTRAST = 7
SEED = 0


def make_split(rep2case: dict, pat: dict, prot: dict, hit: set, seed: int = SEED):
    reps = list(rep2case)
    case2split = {c: c.split('_')[0] for c in rep2case.values()}
    hit = {r for r in hit if r in rep2case}
    P = collections.defaultdict(list)
    for r in reps:
        P[pat.get(r, '?' + r)].append(r)
    mixed = [p for p, rs in P.items() if len({case2split[rep2case[r]] for r in rs}) > 1]
    hit_pat = {pat.get(r, '?' + r) for r in hit}
    to_val = [r for p in hit_pat for r in P[p] if case2split[rep2case[r]] == 'train']
    pool = [p for p, rs in P.items() if p not in hit_pat and all(case2split[rep2case[r]] == 'val' for r in rs)
            and all(prot[r] == {CHEST_NON_CONTRAST} for r in rs)]
    random.Random(seed).shuffle(pool)
    back = []
    n = 0
    for p in pool:
        if n >= len(to_val):
            break
        back += P[p]
        n += len(P[p])
    new = dict(case2split)
    for r in to_val:
        new[rep2case[r]] = 'val'
    for r in back:
        new[rep2case[r]] = 'train'
    assert all(new[rep2case[r]] == 'val' for r in hit)
    for p, rs in P.items():
        assert len({new[rep2case[r]] for r in rs}) == 1 or p in mixed, p
    summary = dict(before=dict(collections.Counter(case2split.values())), after=dict(collections.Counter(new.values())),
                   moved_train_to_val=len(to_val), moved_val_to_train=len(back), ctrate_valid_matched_reports=len(hit),
                   patients_matched=len(hit_pat), patients_split_across_before=len(mixed), seed=seed,
                   to_val=[rep2case[r] for r in to_val], to_train=[rep2case[r] for r in back])
    return new, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cases', required=True)
    ap.add_argument('--labeled')
    ap.add_argument('--report-protocols', required=True)
    ap.add_argument('--match', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    case2rep = {}
    with open(a.cases) as f:
        for line in f:
            c = json.loads(line)
            case2rep[c['id']] = c['report']
    order = Path(a.labeled).read_text().split() if a.labeled else list(case2rep)
    rep2case = {case2rep[c]: c for c in order if c in case2rep}     # report order fixes the seeded draw
    csv.field_size_limit(10 ** 9)
    pat = {}
    prot = collections.defaultdict(set)
    for r in csv.DictReader(open(a.report_protocols)):
        pat[r['ReportUID']] = r['PatientUID']
        prot[r['ReportUID']].add(int(r['protocol_class']))
    hit = {rid for lst in json.load(open(a.match))['volumes'].values() for _, _, rid, _ in lst}
    new, summary = make_split(rep2case, pat, prot, hit)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(new, open(out / 'split.json', 'w'))
    json.dump(summary, open(out / 'summary.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('to_val', 'to_train')}))


if __name__ == '__main__':
    main()

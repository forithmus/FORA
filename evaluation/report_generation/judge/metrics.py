"""LLM-judge summary per dataset and model (paper Methods 4.11, Fig. 4).

* score: mean judge score per report (``prompts.report_score``), with a 95% percentile interval from 1,000 study
  resamples; significant errors per report (total and per kind), matched findings per report, parse rate.
  Unparsable items are left out of that model's means.
* pairwise: per study and order an outcome for ``ours`` (win 1, tie 0.5, loss 0); win / tie / loss rates are
  means over studies of the mean over both orders; ``consistent`` = share of studies whose two orders agree.

CLI::

    python -m evaluation.report_generation.judge.metrics --items judge_inputs/items.jsonl --pairs judge_inputs/pairs.jsonl \
        --outputs 'judge_out/gemma/shard_*.jsonl' --ours ours --out judge_metrics.json
"""
from __future__ import annotations

import argparse
import glob
import json
import random
from collections import defaultdict

import numpy as np

from evaluation.report_generation.judge.prompts import CATS

B, SEED = 1000, 0


def mean(x):
    return sum(x) / len(x) if x else float('nan')


def boot(per_study, rng: random.Random):
    v = np.array([mean(x) if isinstance(x, list) else x for x in per_study], dtype=float)
    idx = np.random.default_rng(rng.randrange(2 ** 31)).integers(0, len(v), size=(B, len(v)))
    m = np.sort(v[idx].mean(1))
    return [float(m[int(.025 * B)]), float(m[int(.975 * B) - 1])]


def summarise(items, pairs, outs, ours=None) -> dict:
    rng = random.Random(SEED)
    by = defaultdict(lambda: defaultdict(dict))
    for r in items:
        by[r['dataset']][r['model']][r['uid']] = r['item_id']
    pr = defaultdict(lambda: defaultdict(dict))
    for r in pairs:
        pr[r['dataset']][r['uid']][r['item_id'].rsplit('|', 1)[1]] = r['item_id']
    res = {}
    for ds in by:
        D = res[ds] = {'single': {}, 'pairwise': None}
        for m in sorted(by[ds]):
            recs = [outs.get(i) for i in by[ds][m].values() if i in outs]; ok = [p for p in recs if p is not None]
            if not ok:
                continue
            sc = [p['score'] for p in ok]; se = [p['sig_total'] for p in ok]
            D['single'][m] = dict(n=len(ok), judged=len(recs), expected=len(by[ds][m]), parse=len(ok) / max(1, len(recs)),
                                  score=mean(sc), score_ci=boot(sc, rng), sig_err=mean(se), sig_err_ci=boot(se, rng),
                                  matched=mean([p['matched'] for p in ok]), sig_by_cat={c: mean([p['sig'][c] for p in ok]) for c in CATS},
                                  insig_err=mean([sum(p['insig'].values()) for p in ok]))
        per = []
        for u, od in pr[ds].items():
            oc = []
            for o, iid in od.items():
                p = outs.get(iid)
                if p:
                    mine = 'A' if o == 'AB' else 'B'
                    oc.append(1.0 if p['better'] == mine else 0.5 if p['better'] == 'tie' else 0.0)
            if oc:
                per.append(oc)
        if per:
            flat = lambda S: mean([mean(x) for x in S])
            D['pairwise'] = dict(ours=ours, studies=len(per), ours_score=flat(per), ours_score_ci=boot(per, rng),
                                 win=mean([mean([x == 1.0 for x in s]) for s in per]), tie=mean([mean([x == 0.5 for x in s]) for s in per]),
                                 loss=mean([mean([x == 0.0 for x in s]) for s in per]),
                                 consistent=mean([len(set(s)) == 1 for s in per if len(s) == 2]) if any(len(s) == 2 for s in per) else None)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--items', required=True); ap.add_argument('--pairs'); ap.add_argument('--outputs', nargs='+', required=True)
    ap.add_argument('--ours'); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    outs = {}
    for g in a.outputs:
        for f in glob.glob(g):
            for l in open(f):
                try:
                    r = json.loads(l)
                except json.JSONDecodeError:
                    continue
                outs[r['item_id']] = r['parsed']
    items = [json.loads(l) for l in open(a.items)]; pairs = [json.loads(l) for l in open(a.pairs)] if a.pairs else []
    res = summarise(items, pairs, outs, a.ours)
    json.dump(res, open(a.out, 'w'), indent=1)
    for ds, D in res.items():
        for m, s in sorted(D['single'].items(), key=lambda kv: -kv[1]['score']):
            print(f"{ds:14s} {m:16s} score {s['score']:.3f} [{s['score_ci'][0]:.3f},{s['score_ci'][1]:.3f}] sig_err {s['sig_err']:.2f} n {s['n']}")


if __name__ == '__main__':
    main()

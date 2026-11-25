#!/usr/bin/env python3
"""TRAIN studies whose own generated reports are edited into PU-DPO pairs (paper section "Grounded report generation",
X-ray PU-DPO). Rare-class stratified: for every class (rarest first) take up to --per_class TRAIN studies positive for
it, then fill with random TRAIN studies up to --n. Output: a subset of the train study list (same rows).

    python -m fora_report.cxr.pudpo_subset --labels labels.npz --cases cases_train.jsonl --out cases_pudpo.jsonl
"""
import argparse
import json
import random

import numpy as np


def select(tid, Y, ok, n: int, per_class: int, rng: random.Random) -> tuple[list[str], int]:
    freq = Y[ok].sum(0)
    chosen, seen = [], set()
    for c in np.argsort(freq):
        idx = [i for i in np.flatnonzero(Y[:, c] & ok) if tid[i] not in seen]
        rng.shuffle(idx)
        have = sum(1 for i in np.flatnonzero(Y[:, c] & ok) if tid[i] in seen)
        for i in idx[: max(0, per_class - have)]:
            seen.add(tid[i])
            chosen.append(tid[i])
    strat = len(chosen)
    rest = [u for u, o in zip(tid, ok) if o and u not in seen]
    rng.shuffle(rest)
    chosen += rest[: max(0, n - len(chosen))]
    rng.shuffle(chosen)
    return chosen, strat


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--labels', required=True, help='labels npz: train_ids, train_Y')
    ap.add_argument('--cases', required=True, help='train study list jsonl')
    ap.add_argument('--n', type=int, default=24000)
    ap.add_argument('--per_class', type=int, default=200)
    ap.add_argument('--seed', type=int, default=20261102)
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)
    L = np.load(a.labels, allow_pickle=False)
    tid = [str(u) for u in L['train_ids'].tolist()]
    Y = np.asarray(L['train_Y']).astype(bool)
    cases = {json.loads(line)['id']: line for line in open(a.cases)}
    ok = np.array([u in cases for u in tid])
    chosen, strat = select(tid, Y, ok, a.n, a.per_class, rng)
    with open(a.out, 'w') as f:
        for u in chosen:
            f.write(cases[u])
    row = {u: i for i, u in enumerate(tid)}
    S = Y[[row[u] for u in chosen]]
    print(json.dumps(dict(n=len(chosen), stratified=strat, classes=int(Y.shape[1]),
                          class_pos_min=int(S.sum(0).min()), class_pos_median=float(np.median(S.sum(0))),
                          classes_lt50=int((S.sum(0) < 50).sum()), positives_per_study=float(S.sum(1).mean()),
                          abnormal_frac=float((S.sum(1) > 0).mean()))))


if __name__ == '__main__':
    main()

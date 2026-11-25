#!/usr/bin/env python3
"""PU-DPO pair construction for the chest X-ray writer (paper section "Grounded report generation", X-ray PU-DPO;
positive-unlabelled DPO after Kobayashi et al., arXiv 2608.05341).

Base response b = the greedy report of the start (DPO) checkpoint on a TRAIN study (on-policy).
Findings in b (G) = the X-ray label extractor run on b (asserted or uncertain).
Reference positives (Rf) = the train labels of the study (labelled positives). Everything else is unlabelled.
Evidence for unlabelled findings = frozen MIL calibrated p (train Platt): probable tier p >= class_plan_threshold,
possible tier p >= class_possible_threshold (train Youden), unsupported p < possible threshold.
For one finding j per pair, the mention variant and the omit variant differ only in j; one of them is b itself, the
other an LLM edit of b (``fora_report.cxr.pudpo_edit``: insert = mention j with the MIL location and plain or hedged
wording; remove = stop discussing j):
  a_insert  j in Rf, j not in G : chosen = insert(b, j, plain)  rejected = b      (labelled positive, model omitted)
  a_keep    j in Rf, j in G     : chosen = b                    rejected = remove(b, j)
  b_insert  j unlabelled, in plan, not in G: chosen = insert(b, j, plain if probable else hedged)  rejected = b
  b_keep    j unlabelled, in plan, in G    : chosen = b         rejected = remove(b, j)
  c_insert  j unlabelled, p < possible, not in G: chosen = b    rejected = insert(b, j, plain)   (unsupported insertion)
  c_remove  j unlabelled, p < possible, in G    : chosen = remove(b, j)  rejected = b            (unsupported mention)
Verification (finalize): the extractor on every edit; insert must add j, remove must drop j, at most one other class
may change, and |len(edit) - len(b)| <= tau * max(len(b), 200 chars) with tau = 0.5.
Rare-class balance: pair weight w = clip(sqrt(median_f / f_j), 0.5, 4) (f = TRAIN positives), study repeat =
clip(round(max w over its pairs), 1, 4); the trainer samples pairs proportional to w and repeats studies.

Steps (label CSVs are read and written by the labels-stage X-ray extractor):
  pudpo_pairs gen_csv   --gen G.jsonl --out label_input_G.csv
  pudpo_pairs requests  --gen G.jsonl --labels labels_G.csv --out requests.jsonl  (+ common data args)
  pudpo_pairs label_csv --edits edits.jsonl --out label_input_V.csv
  pudpo_pairs finalize  --gen G.jsonl --labels labels_G.csv --requests requests.jsonl --edits edits.jsonl
                        --vlabels labels_V.csv --out pairs_pudpo.sqlite                   (+ common data args)
"""
import argparse
import csv
import json
import math
import os
import random
import re
import sqlite3
import time
from collections import Counter, defaultdict

import numpy as np

from fora.reportgen.cxr import location as cxr_location

EXCLUDE_GROUPS = {'Technical & image quality', 'Extractor extras'}
EXCLUDE_CLASSES = {'Support Devices (generic umbrella)'}
TAU = 0.5
STORE_SCHEMA = 'pudpo_pairs_v1'
csv.field_size_limit(10 ** 9)


def normalize(v):
    v = v.casefold().replace('–', '-').replace('—', '-')
    return re.sub(r'[^a-z0-9]+', ' ', v).strip()


def strip_findings(t):
    return re.sub(r'^\s*findings\s*:\s*', '', t, flags=re.I).strip()


class Vocab:
    """Classes + train labels (labels npz) and groups / synonyms (label database vocabulary)."""

    def __init__(self, labels_npz: str, label_db: str):
        L = np.load(labels_npz, allow_pickle=False)
        self.classes = [str(c) for c in L['classes'].tolist()]
        self.cidx = {c: j for j, c in enumerate(self.classes)}
        self.train_ids = [str(u) for u in L['train_ids'].tolist()]
        self.Y = np.asarray(L['train_Y']).astype(bool)
        self.freq = self.Y.sum(0)
        voc = json.loads(sqlite3.connect(f'file:{label_db}?mode=ro', uri=True).execute(
            "select value from metadata where key='vocabulary'").fetchone()[0])
        self.groups = voc['groups']
        self.syn = voc['synonyms']
        self.alias = defaultdict(set)
        for lab in voc['labels']:
            self.alias[normalize(lab)].add(lab)
            for s in (voc['synonyms'].get(lab) or '').split(';'):
                if s.strip():
                    self.alias[normalize(s)].add(lab)
        self.excluded = {j for j, c in enumerate(self.classes)
                         if self.groups.get(c) in EXCLUDE_GROUPS or c in EXCLUDE_CLASSES}

    def read_labels(self, path):
        """ReportUID -> set of class indices (asserted or uncertain), only rows with verdict 'pass'."""
        out, bad = {}, Counter()
        with open(path, newline='', encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                v = r['cxr_pathology_extraction_verdict'].strip()
                if v != 'pass':
                    bad[v] += 1
                    continue
                pos = set()
                for col in ('cxr_pathology_asserted', 'cxr_pathology_uncertain'):
                    for it in json.loads(r[col] or '[]'):
                        if isinstance(it, dict) and it.get('leaf'):
                            m = self.alias.get(normalize(str(it['leaf'])), set())
                            if len(m) == 1:
                                lab = next(iter(m))
                                if lab in self.cidx:
                                    pos.add(self.cidx[lab])
                out[r['ReportUID']] = pos
        return out, bad


def calibrated(calibration: dict, logits):
    return 1 / (1 + np.exp(-(np.asarray(logits, np.float64) * np.asarray(calibration['class_slope'])
                             + np.asarray(calibration['class_intercept']))))


def plan_probs(calibration_path, uids, labels):
    cal = json.loads(open(calibration_path).read())
    assert list(cal['labels']) == labels
    z = np.load(cal['source_predictions'], allow_pickle=False)
    zr = {str(u): i for i, u in enumerate(z['report_uid'].tolist())}
    p = calibrated(cal, np.asarray(z['logit'], np.float64)[[zr[u] for u in uids]])
    return p, np.asarray(cal['class_plan_threshold']), np.asarray(cal['class_possible_threshold'])


def study_items(g: set, ref: set, pi, prob, poss, allowed, freq, prev, rr: random.Random):
    """Edit requests of one study as (type, j, op, mode). g = classes in the base report, ref = labelled positives,
    pi = calibrated p, prob / poss = tier masks, allowed = class mask, freq = train positives, prev = sqrt(freq)."""
    C = len(pi)
    items = []
    rare = lambda js: sorted(js, key=lambda j: freq[j])
    a_ins = [j for j in rare(ref - g) if allowed[j]][:3]
    items += [('a_insert', j, 'insert', 'plain') for j in a_ins]
    a_keep = [j for j in rare(ref & g) if allowed[j]][:1]
    items += [('a_keep', j, 'remove', '') for j in a_keep]
    unl = [j for j in range(C) if allowed[j] and j not in ref]
    b_prob = [j for j in unl if prob[j] and j not in g]
    b_poss = [j for j in unl if poss[j] and j not in g]
    pick = rare(b_prob)[:2]
    if b_poss:
        w = np.array([pi[j] for j in b_poss])
        w = w / w.sum()   # possible-tier item drawn proportional to p
        pick.append(b_poss[int(np.searchsorted(np.cumsum(w), rr.random()))])
    items += [('b_insert', j, 'insert', 'plain' if prob[j] else 'hedged') for j in pick]
    b_keep = [j for j in rare([j for j in unl if (prob[j] or poss[j]) and j in g])][:1]
    items += [('b_keep', j, 'remove', '') for j in b_keep]
    c_pool = [j for j in unl if not (prob[j] or poss[j]) and j not in g]
    if c_pool:
        w = prev[c_pool] / prev[c_pool].sum()
        items.append(('c_insert', c_pool[int(np.searchsorted(np.cumsum(w), rr.random()))], 'insert', 'plain'))
    c_rem = [j for j in unl if not (prob[j] or poss[j]) and j in g]
    if c_rem:
        items.append(('c_remove', rr.choice(c_rem), 'remove', ''))
    return items


def class_name_synonyms(vocab: Vocab, j: int):
    c = vocab.classes[j]
    nm = re.sub(r'\s*\((?:general|NOS|general/composite label|simple/free-flowing|shadow on CXR)\)', '', c)
    return nm, '; '.join(s.strip() for s in (vocab.syn.get(c) or '').split(';')[:3] if s.strip())


def requests(a):
    V = Vocab(a.labels_npz, a.label_db)
    rng = random.Random(a.seed)
    gens = [json.loads(line) for line in open(a.gen)]
    base = {g['id']: strip_findings(g['generated']) for g in gens}
    G, bad = V.read_labels(a.labels)
    uids = [u for u in base if f'gen|{u}' in G and len(base[u]) >= 40]
    print('studies', len(base), 'with base labels', len(uids), 'non-pass', dict(bad), flush=True)
    row = {u: i for i, u in enumerate(V.train_ids)}
    p, thr_plan, thr_pos = plan_probs(a.calibration, uids, V.classes)
    loc = cxr_location.open_store(a.locations, V.classes, a.label_db)
    prev = np.sqrt(np.maximum(V.freq, 1.0))
    allowed = np.array([j not in V.excluded for j in range(len(V.classes))])
    reqs = []
    cnt = Counter()
    for n, u in enumerate(uids):
        rr = random.Random(f'{a.seed}|{u}')
        g = G[f'gen|{u}']
        ref = set(np.flatnonzero(V.Y[row[u]]).tolist())
        pi = p[n]
        prob = pi >= thr_plan
        poss = (pi >= thr_pos) & ~prob
        locs = loc.texts(u) if loc else {}
        merged = {}
        for t, j, op, mode in study_items(g, ref, pi, prob, poss, allowed, V.freq, prev, rr):
            merged.setdefault((int(j), op, mode), []).append(t)
        for (j, op, mode), ts in merged.items():
            nm, syn = class_name_synonyms(V, j)
            reqs.append(dict(rid=f'{u}#{op}#{j}#{mode}', uid=u, types=ts, j=int(j), op=op, mode=mode, name=nm, syn=syn,
                             location=locs.get(int(j)) if op == 'insert' else None, p=round(float(pi[j]), 5),
                             base=base[u]))
            for t in ts:
                cnt[t] += 1
    rng.shuffle(reqs)
    with open(a.out, 'w') as f:
        for r in reqs:
            f.write(json.dumps(r) + '\n')
    print('requests', len(reqs), dict(cnt), flush=True)


def verify_edit(r: dict, edited, base: str, edit_labels, base_labels: set, tau: float = TAU):
    """None if the edit is usable, else the rejection reason."""
    j = r['j']
    if not edited:
        return 'no_edit'
    if edited.strip() == base.strip():
        return 'unchanged'
    if abs(len(edited) - len(base)) > tau * max(len(base), 200):
        return 'length'
    if edit_labels is None:
        return 'no_label'
    if r['op'] == 'insert' and j not in edit_labels:
        return 'j_not_added'
    if r['op'] == 'remove' and j in edit_labels:
        return 'j_not_removed'
    if len((edit_labels ^ base_labels) - {j}) > 1:
        return 'side_effects'
    return None


def orient(pair_type: str, edited: str, base: str) -> tuple[str, str]:
    """(chosen, rejected) of a verified edit."""
    if pair_type in ('a_insert', 'b_insert', 'c_remove'):
        return edited, base
    return base, edited


def pair_weight(freq_j: float, median: float) -> float:
    return float(np.clip(math.sqrt(median / max(freq_j, 1)), 0.5, 4.0))


def study_repeats(keep) -> dict:
    wmax = defaultdict(float)
    for u, _t, _c, w, *_ in keep:
        wmax[u] = max(wmax[u], w)
    return {u: int(np.clip(round(w), 1, 4)) for u, w in wmax.items()}


def finalize(a):
    V = Vocab(a.labels_npz, a.label_db)
    gens = {json.loads(line)['id']: strip_findings(json.loads(line)['generated']) for line in open(a.gen)}
    G, _ = V.read_labels(a.labels)
    req = {json.loads(line)['rid']: json.loads(line) for line in open(a.requests)}
    edits = {json.loads(line)['rid']: json.loads(line)['edited'] for line in open(a.edits)}
    VL, vbad = V.read_labels(a.vlabels)
    med = float(np.median(V.freq[[j for j in range(len(V.classes)) if j not in V.excluded]]))
    keep, why, by_type, kept_cls = [], Counter(), Counter(), defaultdict(Counter)
    for rid, r in req.items():
        b = gens[r['uid']]
        j = r['j']
        t = '+'.join(r['types'])
        e = edits.get(rid)
        reason = verify_edit(r, e, b, VL.get(f'edit|{rid}'), G[f'gen|{r["uid"]}'])
        if reason:
            why[f'{t}:{reason}'] += 1
            continue
        for t in r['types']:
            chosen, rejected = orient(t, e, b)
            keep.append((r['uid'], t, V.classes[j], pair_weight(V.freq[j], med), chosen, rejected, r['mode'] or '',
                         r['p']))
            by_type[t] += 1
            kept_cls[t][V.classes[j]] += 1
    rep = study_repeats(keep)
    write_store(a.out, keep, rep, dict(schema=STORE_SCHEMA, arm='pudpo_mil', pairs=len(keep), pairs_by_type=dict(by_type),
                                       studies=len(rep), study_visits_with_repeats=int(sum(rep.values())),
                                       repeat_hist=dict(Counter(rep.values())),
                                       classes_per_type={t: len(v) for t, v in kept_cls.items()}, requests=len(req),
                                       rejected_reasons=dict(why), verify_nonpass=dict(vbad), tau=TAU,
                                       median_class_freq=med, built=time.time()))
    print('rejected reasons', json.dumps(dict(why), indent=1))


def write_store(out, keep, rep, summary):
    tmp = out + '.tmp'
    if os.path.exists(tmp):
        os.remove(tmp)
    con = sqlite3.connect(tmp)
    con.execute('CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
    con.execute('CREATE TABLE pairs (pair_id INTEGER PRIMARY KEY, report_uid TEXT NOT NULL, pair_type TEXT NOT NULL, '
                'finding TEXT, weight REAL NOT NULL, chosen_text TEXT NOT NULL, rejected_text TEXT NOT NULL, mode TEXT, p REAL)')
    con.execute('CREATE TABLE studies (report_uid TEXT PRIMARY KEY, repeat INTEGER NOT NULL)')
    for i, (u, t, c, w, ch, rj, mode, pp) in enumerate(keep):
        con.execute('INSERT INTO pairs VALUES (?,?,?,?,?,?,?,?,?)', (i, u, t, c, w, ch, rj, mode, pp))
    con.executemany('INSERT INTO studies VALUES (?,?)', sorted(rep.items()))
    con.execute('CREATE INDEX pairs_uid ON pairs(report_uid)')
    con.execute('INSERT INTO metadata VALUES (?,?)', ('summary', json.dumps(summary)))
    con.commit()
    con.close()
    os.replace(tmp, out)
    print(json.dumps({k: v for k, v in summary.items() if k != 'rejected_reasons'}, indent=1))


def _patient(uid: str) -> str:
    return uid.split('/')[1]


def label_csv(a):
    """Label-extractor input for the edits: ReportUID = 'edit|<rid>'."""
    edits = [json.loads(line) for line in open(a.edits)]
    with open(a.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['PatientUID', 'ReportUID', 'findings', 'impression'])
        for e in edits:
            w.writerow([_patient(e['rid']), f"edit|{e['rid']}", e['edited'], ''])
    print('wrote', a.out, len(edits))


def gen_csv(a):
    """Label-extractor input for the base reports: ReportUID = 'gen|<study id>'."""
    with open(a.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['PatientUID', 'ReportUID', 'findings', 'impression'])
        n = 0
        for line in open(a.gen):
            g = json.loads(line)
            w.writerow([_patient(g['id']), f"gen|{g['id']}", strip_findings(g['generated']), ''])
            n += 1
    print('wrote', a.out, n)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['gen_csv', 'requests', 'label_csv', 'finalize'])
    ap.add_argument('--gen')
    ap.add_argument('--labels', help='extractor CSV on the base reports')
    ap.add_argument('--requests')
    ap.add_argument('--edits')
    ap.add_argument('--vlabels', help='extractor CSV on the edits')
    ap.add_argument('--labels_npz', help='train labels npz (classes, train_ids, train_Y)')
    ap.add_argument('--label_db', help='X-ray label sqlite (vocabulary, groups, synonyms)')
    ap.add_argument('--calibration', help='X-ray calibration json')
    ap.add_argument('--locations', default='', help='locations_train.npz')
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=20261102)
    a = ap.parse_args(argv)
    dict(gen_csv=gen_csv, requests=requests, label_csv=label_csv, finalize=finalize)[a.cmd](a)


if __name__ == '__main__':
    main()

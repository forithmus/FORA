"""Chest X-ray report labels -> scored finding sets (paper Methods 4.11, chest radiograph part).

Input: the X-ray extractor CSVs (``labels/report_scoring``; one row per report, ``ReportUID`` = ``<source>|<study>``,
``source`` = ``ref`` for the reference Findings or the model name, verdict column and JSON lists of asserted and
uncertain items with a ``leaf`` name and a ``quote``). Only rows with verdict ``pass`` are used; a missing or failed
row is a report without findings.

Resolution, applied identically to references and to every model:
1. a leaf is mapped to a vocabulary label by its normalised name or a unique synonym (vocabulary JSON
   ``{"labels": [...], "synonyms": {label: "a;b"}}``);
2. hierarchy credit: a label also counts as each of its scored ancestors (``configs/xray_hierarchy.json``);
3. a label that is not one of the 129 classes counts as its scored parent (``configs/xray_rollup.json``);
4. a cerclage-wire or suture-material item whose quote mentions the sternum also counts as median sternotomy wires;
5. synonym groups (``configs/xray_synonym_groups.json`` plus the hierarchy file's ``same_groups``) are merged by OR
   into the group's first member.

CLI (writes the scores directly; X-ray intervals resample studies)::

    python -m evaluation.report_generation.xray_labels --csv labels_*.csv --vocabulary vocab.json \
        --classes classes.json --studies studies.jsonl --out scores_xray.json [--labels-out labels.jsonl]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

from evaluation.report_generation.finding_metrics import score_models

CFG = os.path.join(os.path.dirname(__file__), 'configs')
STERNOTOMY = 'Median sternotomy / sternal wires (normal)'
STERNAL_SOURCES = {'Cerclage wires (rib/sternal fracture fixation)', 'Surgical clips / vascular clips / suture material'}


def normalize(v: str) -> str:
    v = v.casefold().replace('–', '-').replace('—', '-')
    return re.sub(r'[^a-z0-9]+', ' ', v).strip()


class Resolver:
    def __init__(self, vocabulary: dict, classes: list[str], cfg_dir: str = CFG):
        self.classes = list(classes); self.cidx = {c: j for j, c in enumerate(classes)}
        self.alias = defaultdict(set)
        for lab in vocabulary['labels']:
            self.alias[normalize(lab)].add(lab)
            for s in (vocabulary['synonyms'].get(lab) or '').split(';'):
                if s.strip():
                    self.alias[normalize(s)].add(lab)
        self.roll = {m['child']: m['parent'] for m in json.load(open(os.path.join(cfg_dir, 'xray_rollup.json')))['map']}
        h = json.load(open(os.path.join(cfg_dir, 'xray_hierarchy.json')))
        self.ancestors = {}
        for c in h['parent']:
            seen, todo = set(), list(h['parent'][c])
            while todo:
                p = todo.pop()
                if p not in seen:
                    seen.add(p); todo += h['parent'].get(p, [])
            self.ancestors[c] = seen
        self.groups = [g['members'] for g in json.load(open(os.path.join(cfg_dir, 'xray_synonym_groups.json')))['groups']] + h['same_groups']

    def positives(self, row: dict) -> set:
        pos = set()
        for col in ('cxr_pathology_asserted', 'cxr_pathology_uncertain'):
            for it in json.loads(row[col] or '[]'):
                if not (isinstance(it, dict) and it.get('leaf')):
                    continue
                m = self.alias.get(normalize(str(it['leaf'])), set())
                if len(m) != 1:
                    continue
                lab = next(iter(m))
                pos |= {self.cidx[a] for a in self.ancestors.get(lab, ()) if a in self.cidx}
                if lab in self.cidx:
                    pos.add(self.cidx[lab])
                elif lab in self.roll:
                    pos.add(self.cidx[self.roll[lab]])
                if lab in STERNAL_SOURCES and re.search(r'stern', str(it.get('quote', '')), re.I):
                    pos.add(self.cidx[STERNOTOMY])
        return pos

    def merge(self, mats: list[np.ndarray]) -> tuple[list[np.ndarray], list[str]]:
        drop = []
        for g in self.groups:
            js = [self.cidx[m] for m in g]
            for M in mats:
                M[:, js[0]] = M[:, js].any(1)
            drop += js[1:]
        keep = np.array([j for j in range(len(self.classes)) if j not in set(drop)])
        return [M[:, keep] for M in mats], [self.classes[j] for j in keep]


def read_rows(paths: list[str], resolver: Resolver) -> dict:
    out = defaultdict(dict)
    for p in paths:
        with open(p, newline='', encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                tag, uid = r['ReportUID'].split('|', 1)
                if r['cxr_pathology_extraction_verdict'].strip() == 'pass':
                    out[tag][uid] = resolver.positives(r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--csv', nargs='+', required=True); ap.add_argument('--vocabulary', required=True); ap.add_argument('--classes', required=True)
    ap.add_argument('--studies', required=True); ap.add_argument('--first'); ap.add_argument('--boot', type=int, default=1000)
    ap.add_argument('--paired-draws', type=int, default=0); ap.add_argument('--out', required=True); ap.add_argument('--labels-out')
    a = ap.parse_args()
    R = Resolver(json.load(open(a.vocabulary)), json.load(open(a.classes)))
    rows = read_rows([p for g in a.csv for p in glob.glob(g)], R)
    ids = [json.loads(l)['uid'] for l in open(a.studies)]; n = len(ids)

    def mat(d):
        M = np.zeros((n, len(R.classes)), bool)
        for i, u in enumerate(ids):
            for j in d.get(u, ()):
                M[i, j] = True
        return M
    gens = sorted(t for t in rows if t != 'ref')
    mats, names = R.merge([mat(rows['ref'])] + [mat(rows[t]) for t in gens])
    preds = dict(zip(gens, mats[1:]))
    if a.first:
        preds = {a.first: preds.pop(a.first), **preds}
    res = score_models(mats[0], preds, None, a.boot, a.paired_draws)
    res.update(n_studies=n, n_classes_after_merge=len(names), missing_rows={t: sum(u not in rows[t] for u in ids) for t in rows})
    json.dump(res, open(a.out, 'w'), indent=1)
    if a.labels_out:
        with open(a.labels_out, 'w') as f:
            for src, M in [('ref', mats[0])] + [('gen:' + t, preds[t]) for t in preds]:
                for i, u in enumerate(ids):
                    f.write(json.dumps(dict(source=src, uid=u, positive_classes=[names[j] for j in np.flatnonzero(M[i])])) + '\n')
    for t, m in res['models'].items():
        g = m['ge10']; print(f"{t:24s} BA {g['macro_BA']['value']:.4f} CRG {g['CRG']['value']:.4f} rec {g['macro_recall']['value']:.4f}")


if __name__ == '__main__':
    main()

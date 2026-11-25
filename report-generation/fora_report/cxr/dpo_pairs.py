#!/usr/bin/env python3
"""Synthetic DPO pairs for the chest X-ray writer from TRAIN reports (paper section "Grounded report generation",
X-ray DPO pair synthesis).

chosen  = the writer's target text of the study (``fora_report.cxr.data.findings_from``), unprefixed (the trainer
          prepends 'Findings:\\n'); its sha256 is stored per pair and re-checked by the trainer.
rejected (synthetic, all relative to the written report, not image truth):
  omission           one abnormal sentence deleted (abnormal = holds an asserted/uncertain evidence quote of the label
                     extractor, label database columns asserted/uncertain_evidence_json)
  multi_omission     2-4 abnormal sentences deleted
  insertion_plan_fp  a frozen-MIL plan item (calibrated p >= class_possible_threshold, i.e. listed in the prompt) that
                     the report does not assert or hedge is asserted by a donor sentence of that class from another
                     TRAIN patient (the donor sentence asserts only classes of one anatomical group and none of them is
                     positive in the report); inserted after the last sentence of the class's organ, or replacing a
                     sentence whose only denials are of that class. At most 3% of the type per class.
  insertion_random   the same insertion for a random non-positive class (sqrt-prevalence weights, 3% cap)
  normalised         all abnormal sentences of one organ -> the first replaced by one frequent normal-template sentence
                     of that organ (chosen by co-occurrence with the report's own template sentences), the rest deleted
  repetition         a sentence duplicated, or the last 1-2 sentences looped 2-3 times
Insertion excludes incidental, degenerative and technical classes (EXCLUDE_INSERT, EXCLUDE_GROUPS).
Output: one sqlite (tables metadata, pairs).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sqlite3
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np

from fora_report.cxr.data import findings_from

TYPES = ('omission', 'multi_omission', 'insertion_plan_fp', 'insertion_random', 'normalised', 'repetition')

# incidental / degenerative / technical classes: never inserted
EXCLUDE_INSERT = {
    'Chronic bronchitis changes', 'Aortic calcification', 'Unfolded aorta (aortic tortuosity/elongation)',
    'Vertebral degenerative changes / spondylosis', 'Glenohumeral (shoulder) osteoarthritis', 'Kyphosis (hyperkyphosis)',
    'Osteopenia / osteoporosis', 'Scoliosis', 'Calcified granuloma', 'Apical pleural cap', 'Pulmonary parenchymal bands',
    'Rib fracture, healed (with callus)', 'Hiatal hernia (sliding, Type I)', 'Azygos lobe / azygos fissure', 'Cervical rib',
    'Costal cartilage calcification / costochondral junction hypertrophy', 'Nipple shadow',
    'Thoracic cage deformation (generic/other)', 'Accessory fissure (superior accessory / inferior accessory / azygos variant)',
}
EXCLUDE_GROUPS = {'Technical & image quality', 'Normal / No finding', 'Extractor extras', 'Anatomical variants'}

GROUP_ORGAN = {
    'Cardiac & pericardial': 'heart', 'Great vessels & pulmonary vasculature': 'aorta', 'Mediastinum & hila': 'mediastinum',
    'Pulmonary parenchyma & airspace': 'lung', 'Airways & chronic lung disease': 'lung',
    'Pulmonary edema & vascular congestion': 'lung', 'Pleura': 'pleura', 'Diaphragm & upper abdomen': 'diaphragm',
    'Bones & chest wall': 'bone', 'Soft tissues & breast': 'softtissue',
}
ORGANS = {
    'heart': r'heart|cardi\w*|pericard\w*',
    'aorta': r'aort\w*',
    'mediastinum': r'mediastin\w*|hil(?:a|ar|um|i|us)\b|trache\w*|paratracheal',
    'lung': r'lungs?\b|parenchym\w*|pulmonary|bronch\w*|opacit\w*|infiltrat\w*|consolidat\w*|nodul\w*|aerat\w*|atelecta\w*|'
            r'interstiti\w*|reticul\w*|zones?\b|lob(?:e|es|ar)\b|emphysem\w*|pneumon\w*',
    'pleura': r'pleura\w*|costophrenic|costodiaphragmatic|sinus(?:es)?\b|pneumothorax|effusion\w*',
    'diaphragm': r'diaphragm\w*|hemidiaphragm\w*|subdiaphragm\w*',
    'bone': r'bon(?:e|es|y)\b|osseous|vertebr\w*|spin(?:e|al)\b|ribs?\b|clavic\w*|scapul\w*|scolio\w*|spondyl\w*|degenerat\w*|'
            r'humer\w*|lytic|sclerotic|fractur\w*|thoracic cage',
    'softtissue': r'soft tissue\w*|subcutaneous|breast\w*|axilla\w*',
}
ORGAN_RE = {k: re.compile(r'\b(?:' + v + r')', re.I) for k, v in ORGANS.items()}
NORMAL_RE = re.compile(r'\b(?:normal|natural|unremarkable|within (?:normal|physiological) limits|regular|clear|preserved|open|'
                       r'midline|no\b|not (?:detected|observed|seen|present)|were not|was not)\b', re.I)
NEXT_DEPENDENCY = re.compile(r'^(?:it|its|this|these|they|there|also|additionally|in addition|the latter|the former|'
                             r'which|same|similar(?:ly)?|both of)\b', re.I)
SENT_RE = re.compile(r'\S.*?(?:[.!?](?=\s|$)|$)', re.S)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def norm(text: str) -> str:
    return re.sub(r'\s+', ' ', text.lower()).strip().rstrip('.').strip()


def organs_of(text: str) -> frozenset:
    return frozenset(k for k, r in ORGAN_RE.items() if r.search(text))


def locate(quotes: list, sentences_norm: list) -> set:
    """Sentence indices holding an evidence quote (whole quote, else its >= 12-char sentence pieces)."""
    hit = set()
    for q in quotes:
        qn = norm(q)
        if not qn:
            continue
        found = [i for i, s in enumerate(sentences_norm) if qn in s]
        if not found:
            for piece in re.split(r'(?<=[.!?])\s+', q):
                pn = norm(re.sub(r'^\(|\)$', '', piece.strip()))
                if len(pn) >= 12:
                    found += [i for i, s in enumerate(sentences_norm) if pn in s]
        hit.update(found)
    return hit


def shard_pass1(job):
    """One h5 shard: findings text, sentence spans, per-sentence positive/denied vocab classes."""
    h5_root, h5_id, items = job
    out = []
    with h5py.File(f'{h5_root}/{h5_id}.h5', 'r') as handle:
        for uid, patient, report_id, row in items:
            try:
                text = findings_from(handle[patient][report_id]['report'])
            except Exception:
                continue
            spans = [(m.start(), m.end()) for m in SENT_RE.finditer(text)]
            sn = [norm(text[a:b]) for a, b in spans]
            pos = [set() for _ in spans]
            den = [set() for _ in spans]
            if row is not None:
                for key, target in (('asserted', pos), ('uncertain', pos), ('denied', den)):
                    for c, quotes in json.loads(row[key]).items():
                        for i in locate(quotes, sn):
                            target[i].add(int(c))
            out.append(dict(uid=uid, patient=patient, text=text, spans=spans,
                            pos=[sorted(p) for p in pos], den=[sorted(d) for d in den],
                            positive=row['positive'] if row is not None else [],
                            denied_all=row['denied_all'] if row is not None else []))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--label_db', required=True, help='X-ray label sqlite (labels stage)')
    ap.add_argument('--cases', required=True, help='train study list jsonl')
    ap.add_argument('--h5_root', required=True)
    ap.add_argument('--workers', type=int, default=48)
    ap.add_argument('--seed', type=int, default=20261031)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--calibration', required=True)
    ap.add_argument('--dense', default='', help='default: the calibration source_predictions')
    ap.add_argument('--target', default='omission=55000,multi_omission=25000,insertion_plan_fp=35000,'
                                        'insertion_random=35000,normalised=40000,repetition=12000')
    a = ap.parse_args()
    t0 = time.time()
    log = lambda *x: print(f'[cxrpairs {time.time() - t0:7.0f}s]', *x, flush=True)
    rng = random.Random(a.seed)
    target = {k: int(v) for k, v in (kv.split('=') for kv in a.target.split(','))}

    db = sqlite3.connect(f'file:{a.label_db}?mode=ro', uri=True)
    vocab = json.loads(db.execute("select value from metadata where key='vocabulary'").fetchone()[0])
    vnames = list(vocab['labels']); vgroup = vocab['groups']
    calib = json.loads(Path(a.calibration).read_text())
    labels = list(calib['labels'])
    vidx = {n: i for i, n in enumerate(vnames)}
    mil_v = [vidx[n] for n in labels]                       # MIL class k -> vocab index
    v2mil = {v: k for k, v in enumerate(mil_v)}
    excluded_v = {vidx[n] for n in vnames if n in EXCLUDE_INSERT or vgroup.get(n) in EXCLUDE_GROUPS}
    organ_v = {vidx[n]: GROUP_ORGAN.get(vgroup.get(n)) for n in vnames}
    group_v = {vidx[n]: vgroup.get(n) for n in vnames}

    cases = [json.loads(l) for l in open(a.cases)]
    if a.limit:
        cases = cases[: a.limit]
    rows = {}
    for r in db.execute("select report_uid,h5_id,split,positive_json,asserted_evidence_json,uncertain_evidence_json,"
                        "denied_evidence_json,denied_json from samples where split='train'"):
        rows[(r[1], r[0])] = dict(positive=json.loads(r[3]), asserted=r[4], uncertain=r[5], denied=r[6],
                                  denied_all=json.loads(r[7]))
    by_shard = defaultdict(list)
    for c in cases:
        by_shard[c['h5']].append((c['id'], c['patient'], c['report'], rows.get((c['h5'], c['report']))))
    log(f'train cases={len(cases)} with label row={sum(1 for c in cases if (c["h5"], c["report"]) in rows)} shards={len(by_shard)}')
    with Pool(a.workers) as pool:
        reports = [r for part in pool.imap_unordered(shard_pass1, [(a.h5_root, k, v) for k, v in sorted(by_shard.items())], chunksize=1) for r in part]
    reports.sort(key=lambda r: r['uid'])
    log(f'pass1 reports={len(reports)} abnormal-sentence reports={sum(1 for r in reports if any(r["pos"]))}')

    # plan (frozen MIL, all_classes): calibrated p >= possible threshold
    z = np.load(a.dense or calib['source_predictions'], allow_pickle=False)
    assert [str(x) for x in z['label'].tolist()] == labels, 'dense labels != calibration labels'
    zrow = {str(u): i for i, u in enumerate(z['report_uid'].tolist())}
    zlogit = np.asarray(z['logit'], np.float32)   # load once (NpzFile re-reads per access)
    slope = np.asarray(calib['class_slope'], np.float32); icpt = np.asarray(calib['class_intercept'], np.float32)
    possible = np.asarray(calib['class_possible_threshold'], np.float32)

    def plan_of(uid):
        i = zrow.get(uid)
        if i is None:
            return []
        p = 1 / (1 + np.exp(-(zlogit[i] * slope + icpt)))
        sel = np.flatnonzero(p >= possible)
        return [(int(k), float(p[k])) for k in sel[np.argsort(-p[sel])]]

    # donors (class -> [(patient, sentence)]) and normal templates
    donors = defaultdict(list)
    normal_count = Counter()
    for r in reports:
        for (s0, s1), P, D in zip(r['spans'], r['pos'], r['den']):
            s = r['text'][s0:s1]
            if P:
                groups = {group_v[c] for c in P}
                if len(groups) == 1 and 15 <= len(s) <= 240 and not NEXT_DEPENDENCY.match(s):
                    for c in P:
                        donors[c].append((r['patient'], s, tuple(P)))
            elif NORMAL_RE.search(s) and len(s) <= 200:
                normal_count[s] += 1
    templates = defaultdict(list)       # organ -> frequent normal sentences
    for s, n in normal_count.most_common():
        if n < 20:
            break
        o = organs_of(s)
        if len(o) == 1:
            (organ,) = o
            if len(templates[organ]) < 25:
                templates[organ].append(s)
    tset = {s: o for o, ss in templates.items() for s in ss}
    cooc = defaultdict(Counter)
    for r in reports:
        present = [r['text'][a0:a1] for a0, a1 in r['spans'] if r['text'][a0:a1] in tset]
        for s in present:
            for t in present:
                if s != t:
                    cooc[t][s] += 1
    log(f'donor classes={len(donors)} templates={ {o: len(v) for o, v in templates.items()} }')

    def text_without(r, drop: set, replace: dict | None = None) -> str:
        parts = []
        for i, (s0, s1) in enumerate(r['spans']):
            if i in (replace or {}):
                parts.append(replace[i])
            elif i not in drop:
                parts.append(r['text'][s0:s1])
        return ' '.join(parts)

    def insert_text(r, c, donor):
        """Insert donor sentence asserting vocab class c; None if that would be ill-formed."""
        positive = set(r['positive'])
        _patient, sentence, dclasses = donor
        if positive & set(dclasses) or norm(sentence) in {norm(r['text'][a0:a1]) for a0, a1 in r['spans']}:
            return None
        denial_rows = [i for i, D in enumerate(r['den']) if set(D) & set(dclasses)]
        other_denials = set(r['denied_all']) & set(dclasses)
        if denial_rows:
            if len(denial_rows) > 1:
                return None
            i = denial_rows[0]
            if not set(r['den'][i]) <= set(dclasses) or r['pos'][i]:
                return None              # the denial sentence also denies other things: no clean replacement
            return text_without(r, set(), {i: sentence})
        if other_denials:
            return None                  # denied in a sentence we could not locate
        organ = organ_v.get(c)
        anchor = None
        if organ:
            hits = [i for i, (a0, a1) in enumerate(r['spans']) if organ in organs_of(r['text'][a0:a1])]
            anchor = hits[-1] if hits else None
        if anchor is None:
            anchor = len(r['spans']) - 1
        parts = [r['text'][a0:a1] for a0, a1 in r['spans']]
        if anchor + 1 < len(parts) and NEXT_DEPENDENCY.match(parts[anchor + 1]):
            return None
        parts.insert(anchor + 1, sentence)
        return ' '.join(parts)

    def pick_donor(c, patient, rr):
        pool = donors.get(c)
        if not pool:
            return None
        for _ in range(6):
            d = pool[rr.randrange(len(pool))]
            if d[0] != patient:
                return d
        return None

    cand = defaultdict(list)    # type -> [(uid, attribute, rejected)]
    mil_weight = {}
    for k, n in enumerate(labels):
        v = mil_v[k]
        if v in excluded_v or v not in donors:
            continue
        mil_weight[v] = math.sqrt(len(donors[v]))
    rand_classes = list(mil_weight); rand_w = np.array([mil_weight[v] for v in rand_classes]); rand_w /= rand_w.sum()
    for r in reports:
        rr = random.Random(f'{a.seed}|{r["uid"]}')
        n = len(r['spans'])
        if n == 0:
            continue
        abn = [i for i, P in enumerate(r['pos']) if P]
        ok_drop = [i for i in abn if n > 1 and not (i + 1 < n and NEXT_DEPENDENCY.match(r['text'][r['spans'][i + 1][0]:]))]
        pref = [i for i in ok_drop if not set(r['pos'][i]) <= excluded_v]
        # omission
        pool_om = pref if pref else (ok_drop if rr.random() < 0.3 else [])
        for i in rr.sample(pool_om, min(2, len(pool_om))):
            cand['omission'].append((r['uid'], '|'.join(vnames[c] for c in r['pos'][i]), text_without(r, {i})))
        # multi omission
        if len(ok_drop) >= 2 and len(ok_drop) < n:
            m = rr.randint(2, min(4, len(ok_drop)))
            drop = set(rr.sample(ok_drop, m))
            cand['multi_omission'].append((r['uid'], f'n={m}', text_without(r, drop)))
        # normalised
        by_organ = defaultdict(list)
        for i in abn:
            for c in r['pos'][i]:
                if organ_v.get(c):
                    by_organ[organ_v[c]].append(i)
        organs = [o for o in by_organ if templates.get(o)]
        if organs:
            o = rr.choice(sorted(organs))
            idx = sorted(set(by_organ[o]))
            present = [r['text'][a0:a1] for a0, a1 in r['spans'] if r['text'][a0:a1] in tset]
            existing = {norm(r['text'][a0:a1]) for a0, a1 in r['spans']}
            scored = sorted(templates[o], key=lambda t: (-sum(cooc[t][s] for s in present), -normal_count[t]))
            scored = [t for t in scored if norm(t) not in existing]
            if scored and len(idx) < n + 1:
                tpl = scored[0]
                cand['normalised'].append((r['uid'], o, text_without(r, set(idx[1:]), {idx[0]: tpl})))
        # insertion_plan_fp (up to 2 plan items per report, highest p first)
        positive = set(r['positive'])
        got = 0
        for k, p in plan_of(r['uid']):
            v = mil_v[k]
            if got >= 2 or v in positive or v in excluded_v:
                continue
            d = pick_donor(v, r['patient'], rr)
            t = insert_text(r, v, d) if d else None
            if t:
                cand['insertion_plan_fp'].append((r['uid'], f'{vnames[v]}|p={p:.3f}', t)); got += 1
        # insertion_random (1 per report)
        for _ in range(4):
            v = rand_classes[int(np.searchsorted(np.cumsum(rand_w), rr.random()))]
            if v in positive:
                continue
            d = pick_donor(v, r['patient'], rr)
            t = insert_text(r, v, d) if d else None
            if t:
                cand['insertion_random'].append((r['uid'], vnames[v], t))
                break
        # repetition: duplicate one sentence, or loop the tail
        if rr.random() < 0.15:
            parts = [r['text'][a0:a1] for a0, a1 in r['spans']]
            if rr.random() < 0.5 or len(parts) < 2:
                i = rr.choice(abn) if abn else rr.randrange(len(parts))
                rep = parts[: i + 1] + [parts[i]] + parts[i + 1:]
                cand['repetition'].append((r['uid'], 'duplicate', ' '.join(rep)))
            else:
                tail = parts[-rr.randint(1, 2):]
                cand['repetition'].append((r['uid'], 'loop', ' '.join(parts + tail * rr.randint(2, 3))))
    log('candidates', {k: len(v) for k, v in cand.items()})

    # caps: insertion types <= 3% per class; then subsample each type to its target
    final = {}
    for kind in TYPES:
        items = cand[kind]
        rng.shuffle(items)
        if kind.startswith('insertion'):
            cap = max(1, int(0.03 * min(target[kind], len(items))))
            per = Counter(); kept = []
            for it in items:
                cls = it[1].split('|')[0]
                if per[cls] < cap:
                    per[cls] += 1; kept.append(it)
            items = kept
        final[kind] = items[: target[kind]]
    log('final', {k: len(v) for k, v in final.items()})

    text_by_uid = {r['uid']: r['text'] for r in reports}
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + '.tmp'); tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    con.execute('CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
    con.execute('CREATE TABLE pairs (pair_id INTEGER PRIMARY KEY, report_uid TEXT NOT NULL, pair_type TEXT NOT NULL, '
                'attribute_type TEXT, chosen_sha256 TEXT NOT NULL, rejected_text TEXT NOT NULL)')
    pid = 0; by_type = Counter(); bad = 0
    for kind in TYPES:
        for uid, attr, rej in final[kind]:
            chosen = text_by_uid[uid]
            if not rej.strip() or rej == chosen:
                bad += 1; continue
            con.execute('INSERT INTO pairs VALUES (?,?,?,?,?,?)', (pid, uid, kind, attr, sha(chosen), rej))
            pid += 1; by_type[kind] += 1
    con.execute('CREATE INDEX pairs_uid ON pairs(report_uid)')
    summary = dict(schema='cxr_report_preference_index_v1', pairs=pid,
                   pairs_by_type=dict(by_type), reports_with_pairs=len({u for k in final for u, _a, _r in final[k]}),
                   train_reports=len(reports), dropped_identical=bad, excluded_insert=sorted(EXCLUDE_INSERT),
                   excluded_groups=sorted(EXCLUDE_GROUPS), seed=a.seed, target=target, built=time.time(),
                   calibration=str(Path(a.calibration).resolve()), dense=str(Path(a.dense or calib['source_predictions']).resolve()))
    con.execute('INSERT INTO metadata VALUES (?,?)', ('summary', json.dumps(summary)))
    con.commit(); con.close(); tmp.replace(out)
    log('wrote', out, json.dumps(summary)[:600])


if __name__ == '__main__':
    main()

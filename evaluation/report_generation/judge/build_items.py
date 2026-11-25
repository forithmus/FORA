"""Build LLM-judge items (paper Methods 4.11) from references and generated reports of any models.

Every model of a dataset is judged on the same studies (studies with a non-empty reference). A model without a
report for a study is judged on an empty report (every abnormal reference finding missed). Text rule: CT keeps
the text before an "Impression:" header minus a leading "Findings:" and removes any ``<think>...</think>``
reasoning; X-ray removes a leading "Findings:". Pairwise items compare ``--ours`` with ``--opponent`` in both
orders (AB and BA) to cancel position bias.

Inputs::

    --references refs.jsonl   {uid, reference, region?}
    --gens NAME=gens.jsonl    {uid, text}     (one per model)

Output: ``items.jsonl`` {item_id, kind:'single', dataset, modality, region, uid, model, ref, gen} and
``pairs.jsonl`` {item_id, kind:'pair', dataset, modality, region, uid, a_model, b_model, ref, a, b}.

CLI::

    python -m evaluation.report_generation.judge.build_items --dataset ct_ctrate --modality CT --region chest \
        --references refs.jsonl --gens ours=ours.jsonl --gens nv_reason=nv.jsonl --ours ours --opponent nv_reason --out judge_inputs/
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re

_IMP = re.compile(r'\n?\s*\**\s*impressions?\s*\**\s*:', re.I)
_FIND = re.compile(r'^\s*\**\s*findings?\s*\**\s*:\s*', re.I)
_THINK = re.compile(r'^.*?</think>', re.S)


def ct_text(t: str) -> str:
    t = _THINK.sub('', t or '')
    return _FIND.sub('', _IMP.split(t, maxsplit=1)[0]).strip()


def xray_text(t: str) -> str:
    return re.sub(r'^\s*findings\s*:\s*', '', t or '', flags=re.I).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', required=True); ap.add_argument('--modality', required=True, choices=['CT', 'chest x-ray'])
    ap.add_argument('--region', default='chest'); ap.add_argument('--references', required=True)
    ap.add_argument('--gens', action='append', required=True); ap.add_argument('--ours'); ap.add_argument('--opponent')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    clean = ct_text if a.modality == 'CT' else xray_text
    refs = {r['uid']: r for r in map(json.loads, open(a.references))}
    ids = [u for u, r in refs.items() if (r.get('reference') or '').strip()]
    g = {}
    for spec in a.gens:
        name, p = spec.split('=', 1); g[name] = {r['uid']: clean(r.get('text', '')) for r in map(json.loads, open(p))}
    recs = []
    with open(os.path.join(a.out, 'pairs.jsonl'), 'a') as hp:
        for u in ids:
            ref = refs[u]['reference'].strip(); reg = refs[u].get('region') or a.region
            for m in sorted(g):
                recs.append(dict(item_id=f'{a.dataset}|{u}|{m}', kind='single', dataset=a.dataset, modality=a.modality, region=reg, uid=u,
                                 model=m, ref=ref, gen=g[m].get(u, '') or ''))
            if a.ours and a.opponent:
                for order in ('AB', 'BA'):
                    x, y = (a.ours, a.opponent) if order == 'AB' else (a.opponent, a.ours)
                    hp.write(json.dumps(dict(item_id=f'{a.dataset}|{u}|{order}', kind='pair', dataset=a.dataset, modality=a.modality, region=reg,
                                             uid=u, a_model=x, b_model=y, ref=ref, a=g[x].get(u, '') or '', b=g[y].get(u, '') or ''), ensure_ascii=False) + '\n')
    random.Random(7).shuffle(recs)
    with open(os.path.join(a.out, 'items.jsonl'), 'a') as hi:
        for r in recs:
            hi.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(a.dataset, len(ids), 'studies', len(g), 'models')


if __name__ == '__main__':
    main()

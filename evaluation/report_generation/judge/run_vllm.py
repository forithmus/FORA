"""Run the LLM judge (Gemma-4-31B-it, ``google/gemma-4-31B-it``) with vLLM on one shard of the judge items
(paper Methods 4.11).

Greedy decoding (temperature 0, at most 900 new tokens), single-report items with the abnormal-findings prompt and
pairwise items with the pairwise prompt. An unparsable answer is retried once with sampling (temperature 0.7,
top-p 0.95, seed 1). The run is resumable: item ids already in the output file are skipped.
Output: ``<out>/shard_<k>.jsonl`` {item_id, judge, raw, parsed (None = unparsable)[, raw_retry]}.

CLI (one GPU per shard)::

    python -m evaluation.report_generation.judge.run_vllm --model google/gemma-4-31B-it --items judge_inputs/items.jsonl \
        --items judge_inputs/pairs.jsonl --shard 0 --nshards 16 --out judge_out/gemma
"""
from __future__ import annotations

import argparse
import json
import os

from evaluation.report_generation.judge import prompts as P


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True); ap.add_argument('--judge', default='gemma')
    ap.add_argument('--items', action='append', required=True); ap.add_argument('--shard', type=int, default=0); ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--tp', type=int, default=1); ap.add_argument('--max-len', type=int, default=16384); ap.add_argument('--util', type=float, default=0.88)
    ap.add_argument('--chunk', type=int, default=1024); ap.add_argument('--out', required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, f'shard_{a.shard:03d}.jsonl')
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)['item_id'] for l in open(out)}
    items = []
    for f in a.items:
        for i, l in enumerate(open(f)):
            if i % a.nshards == a.shard:
                it = json.loads(l)
                if it['item_id'] not in done:
                    items.append(it)
    if not items:
        return
    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, tensor_parallel_size=a.tp, max_model_len=a.max_len, gpu_memory_utilization=a.util, seed=42,
              enable_prefix_caching=True, trust_remote_code=True, limit_mm_per_prompt={'image': 0, 'audio': 0})
    tok = llm.get_tokenizer()
    render = lambda it: tok.apply_chat_template(P.messages(it), tokenize=False, add_generation_prompt=True)
    sp = SamplingParams(temperature=0.0, max_tokens=900)
    sp_retry = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=900, seed=1)
    with open(out, 'a') as fo:
        for c in range(0, len(items), a.chunk):
            ch = items[c:c + a.chunk]
            raws = [o.outputs[0].text for o in llm.generate([render(it) for it in ch], sp, use_tqdm=False)]
            parsed = [P.parse(it, r) for it, r in zip(ch, raws)]
            bad = [i for i, p in enumerate(parsed) if p is None]; retry = {}
            if bad:
                for i, o in zip(bad, llm.generate([render(ch[i]) for i in bad], sp_retry, use_tqdm=False)):
                    retry[i] = o.outputs[0].text; parsed[i] = P.parse(ch[i], retry[i])
            for i, it in enumerate(ch):
                rec = {'item_id': it['item_id'], 'judge': a.judge, 'raw': raws[i], 'parsed': parsed[i]}
                if i in retry:
                    rec['raw_retry'] = retry[i]
                fo.write(json.dumps(rec, ensure_ascii=False) + '\n')
            fo.flush()


if __name__ == '__main__':
    main()

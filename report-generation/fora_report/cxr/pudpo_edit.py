#!/usr/bin/env python3
"""PU-DPO report editor (paper section "Grounded report generation", X-ray PU-DPO pair synthesis).

An instruction-tuned LLM (Gemma-4-31B-it via vLLM, greedy) applies one 'insert' (mention a finding) or 'remove'
(stop discussing a finding) edit per request to a writer-generated Findings section. The prompt text lives in
prompts/cxr_pudpo_editor.txt.

Input: requests jsonl (``fora_report.cxr.pudpo_pairs requests``): {rid, base, op, name, syn, location, mode, ...}.
Output: OUT_DIR/edits_rank_XXX.jsonl ({rid, edited}); one vLLM engine per process (one GPU each), requests sharded by
--rank / --world (defaults from RANK/WORLD_SIZE or SLURM_PROCID/SLURM_NTASKS).

    python -m fora_report.cxr.pudpo_edit --requests requests.jsonl --out_dir edits --model google/gemma-4-31B-it
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

PROMPT_FILE = Path(__file__).resolve().parents[2] / 'prompts' / 'cxr_pudpo_editor.txt'


def load_prompts(path: str | Path = PROMPT_FILE) -> dict:
    """'### name' sections; each section's text without its final newline."""
    sections, name, lines = {}, None, []
    for line in Path(path).read_text().splitlines(keepends=True):
        if line.startswith('### '):
            if name is not None:
                sections[name] = ''.join(lines)[:-1] if lines else ''
            name, lines = line[4:].strip(), []
        else:
            lines.append(line)
    if name is not None:
        sections[name] = ''.join(lines).rstrip('\n')
    return sections


def _fill(template: str, **values) -> str:
    for k, v in values.items():
        template = template.replace('{' + k + '}', v)
    return template


def messages(r: dict, prompts: dict) -> list[dict]:
    alias = _fill(prompts['alias'], syn=r['syn']) if r.get('syn') else ''
    head = _fill(prompts['head'], base=r['base'])
    if r['op'] == 'insert':
        loc = _fill(prompts['location'], location=r['location']) if r.get('location') else ''
        how = prompts['hedged'] if r['mode'] == 'hedged' else prompts['plain']
        body = _fill(prompts['insert'], name=r['name'], alias=alias, location=loc, how=how)
    else:
        body = _fill(prompts['remove'], name=r['name'], alias=alias)
    return [{'role': 'system', 'content': prompts['system']}, {'role': 'user', 'content': head + body}]


def clean(text: str) -> str:
    t = text.strip()
    t = re.sub(r'^```\w*\s*|\s*```$', '', t).strip()
    t = re.sub(r'^<<<\s*|\s*>>>$', '', t).strip()
    t = re.sub(r'^\s*\**findings\**\s*:\s*', '', t, flags=re.I).strip()
    return t


class Editor:
    """vLLM engine with the model's chat template, greedy decoding."""

    def __init__(self, model: str, *, max_model_len: int = 6144, gpu_memory_utilization: float = 0.88, seed: int = 42,
                 max_tokens: int = 1024):
        from vllm import LLM, SamplingParams
        self.llm = LLM(model=model, tensor_parallel_size=1, max_model_len=max_model_len,
                       gpu_memory_utilization=gpu_memory_utilization, dtype='auto', seed=seed,
                       enable_prefix_caching=True, limit_mm_per_prompt={'image': 0, 'audio': 0})
        self.tokenizer = self.llm.get_tokenizer()
        self.sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens, seed=seed)

    def chat(self, batch: list[list[dict]]) -> list[str]:
        if not batch:
            return []
        prompts = [self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch]
        return [o.outputs[0].text for o in self.llm.generate(prompts, self.sampling)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--requests', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--model', required=True, help='HF id or local snapshot of the editor LLM')
    ap.add_argument('--prompts', default=str(PROMPT_FILE))
    ap.add_argument('--batch', type=int, default=4000)
    ap.add_argument('--rank', type=int, default=int(os.environ.get('RANK', os.environ.get('SLURM_PROCID', 0))))
    ap.add_argument('--world', type=int, default=int(os.environ.get('WORLD_SIZE', os.environ.get('SLURM_NTASKS', 1))))
    a = ap.parse_args(argv)
    rank, world = a.rank, a.world
    out = Path(a.out_dir) / f'edits_rank_{rank:03d}.jsonl'
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f'[rank {rank}] exists, skip', flush=True)
        return
    prompts = load_prompts(a.prompts)
    rows = [json.loads(line) for i, line in enumerate(open(a.requests)) if i % world == rank]
    editor = Editor(a.model)
    t0 = time.time()
    res = []
    for i in range(0, len(rows), a.batch):
        part = rows[i:i + a.batch]
        texts = editor.chat([messages(r, prompts) for r in part])
        res += [dict(rid=r['rid'], edited=clean(t)) for r, t in zip(part, texts)]
        print(f'[rank {rank}] {len(res)}/{len(rows)} {time.time() - t0:.0f}s', flush=True)
    tmp = out.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        for r in res:
            f.write(json.dumps(r) + '\n')
    os.replace(tmp, out)
    print(f'[rank {rank}] done {len(res)} in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()

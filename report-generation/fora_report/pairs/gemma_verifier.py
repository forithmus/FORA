"""Gemma-4-31B-it attribute corruption and judge for the CT attribute DPO pairs (paper: "Preference optimization").

Four steps, run in this order (``corrupt`` and ``judge`` need one GPU per process, vLLM):

  corrupt      every CPU-preselected candidate sentence (``base build`` -> attribute_candidates.jsonl.gz) is sent
               to Gemma, which writes ONE minimal, clinically WRONG edit of ONE attribute, or {"applicable": false}.
  judge-input  join candidates with the applicable corruptions (edits identical to the source are dropped).
  judge        Gemma grades each edit on five booleans (grammatical, consistent, anatomy_valid,
               only_target_changed, attribute_changed).
  filter       keep the corruption records whose judgement has all five booleans true.

The filtered rank files then go through the CPU verifier (``python -m fora_report.pairs.attribute``).
Greedy decoding (temperature 0, top-p 1), seed 0, bf16, max_model_len 16384, 512 new tokens, chunks of 4,096
prompts. Prompts are read from ``prompts/pairs_attribute_{corruption,verifier}_{system,user}.txt``.

Sharding: start one process per GPU with ``--rank r --world W`` (item n goes to rank n % W). Output is
``<out_dir>/rank_XXX.jsonl``; already written ids are skipped, so every step is resumable.

  python -m fora_report.pairs.gemma_verifier corrupt --input attribute_candidates.jsonl.gz --class_names classes.json \
      --out_dir gemma_attr --rank 0 --world 4 --model google/gemma-4-31B-it
  python -m fora_report.pairs.gemma_verifier judge-input --candidates attribute_candidates.jsonl.gz \
      --gemma_dir gemma_attr --output judge_input.jsonl.gz
  python -m fora_report.pairs.gemma_verifier judge --input judge_input.jsonl.gz --class_names classes.json \
      --out_dir gemma_judge --rank 0 --world 4
  python -m fora_report.pairs.gemma_verifier filter --gemma_dir gemma_attr --judge_dir gemma_judge --out_dir gemma_attr_judged
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
from pathlib import Path
import sys
import time

PROMPTS = Path(__file__).resolve().parents[2] / "prompts"
JUDGE_KEYS = ("grammatical", "consistent", "anatomy_valid", "only_target_changed", "attribute_changed")
GREEDY_TEMPERATURE = 0.0
GREEDY_TOP_P = 1.0
CONTEXT_LENGTH_SAFETY_TOKENS = 64
MODEL_TAG = "gemma-4-31b-it"


def load_prompt(name: str) -> str:
    return (PROMPTS / name).read_text(encoding="utf-8")


def parse(raw):
    s = (raw or "").strip()
    i = s.find("{")
    if i < 0:
        return None
    try:
        return json.JSONDecoder().raw_decode(s[i:])[0]
    except Exception:
        return None


def read_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


class VLLMChat:
    """Minimal vLLM chat wrapper: chat-template rendering, prompt-length check and batched greedy generation."""

    def __init__(self, model: str, *, max_model_len: int, gpu_memory_utilization: float = 0.90, seed: int = 0,
                 dtype: str = "bfloat16", tensor_parallel_size: int = 1):
        from vllm import LLM
        self.max_model_len = max_model_len
        self.seed = seed
        self.llm = LLM(model=model, tensor_parallel_size=tensor_parallel_size, max_model_len=max_model_len,
                       gpu_memory_utilization=gpu_memory_utilization, dtype=dtype, seed=seed,
                       enable_prefix_caching=True, limit_mm_per_prompt={"image": 0, "audio": 0})
        self.tokenizer = self.llm.get_tokenizer()

    def prompt_fits(self, messages, max_new_tokens: int, safety_tokens: int = 0) -> bool:
        n = len(self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
        return n <= self.max_model_len - max_new_tokens - safety_tokens

    def sampling(self, *, temperature: float, top_p: float, max_tokens: int):
        from vllm import SamplingParams
        return SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=self.seed)

    def chat(self, batch, sampling) -> list[str]:
        if not batch:
            return []
        prompts = [self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch]
        return [o.outputs[0].text for o in self.llm.generate(prompts, sampling)]


def finding_namer(class_names_path: str):
    names = json.loads(Path(class_names_path).read_text()) if class_names_path else []

    def finding_name(rec):
        found = [names[int(c)] for c in rec.get("class_ids", []) if names and 0 <= int(c) < len(names)]
        return "; ".join(found) if found else "(the abnormal finding described in the sentence)"
    return finding_name


def build_messages(task: str, rec: dict, finding_name) -> list[dict]:
    if task == "corrupt":
        kind = rec["attribute_type"]
        spans = ", ".join(json.dumps(s, ensure_ascii=False) for s in rec.get("attribute_spans", []))
        user = load_prompt("pairs_attribute_corruption_user.txt").format(
            attribute=kind, attribute_upper=kind.upper(), sentence=rec["sentence"], spans=spans,
            finding=finding_name(rec))
        return [{"role": "system", "content": load_prompt("pairs_attribute_corruption_system.txt")},
                {"role": "user", "content": user}]
    user = load_prompt("pairs_attribute_verifier_user.txt").format(
        attribute=rec["attribute_type"], finding=finding_name(rec), sentence=rec["sentence"],
        edited=rec["corrupted_sentence"])
    return [{"role": "system", "content": load_prompt("pairs_attribute_verifier_system.txt")},
            {"role": "user", "content": user}]


def output_row(task: str, rec: dict, parsed: dict, raw: str) -> dict:
    if task == "corrupt":
        return {"cand_id": rec["cand_id"], "attribute_type": rec["attribute_type"],
                "applicable": bool(parsed.get("applicable", False)) and not parsed.get("unparsed"),
                "original_span": parsed.get("original_span"), "new_span": parsed.get("new_span"),
                "corrupted_sentence": parsed.get("corrupted_sentence"), "model": MODEL_TAG,
                "raw": None if parsed.get("corrupted_sentence") else (raw or "")[:400]}
    return {"item_id": rec["item_id"], "parsed": parsed, "raw": None if not parsed.get("unparsed") else (raw or "")[:2000]}


def cmd_generate(a):
    """``corrupt`` / ``judge``: one process per GPU, records to <out_dir>/rank_XXX.jsonl."""
    log = lambda m: print(f"[rank {a.rank}] {m}"[:880], file=sys.stderr, flush=True)  # noqa: E731
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"rank_{a.rank:03d}.jsonl"
    done = set()
    if out_path.exists():
        for rec in read_jsonl(out_path):
            done.add(rec.get("cand_id") or rec.get("item_id"))
    items = []
    for n, rec in enumerate(read_jsonl(a.input)):
        if a.limit and n >= a.limit:
            break
        if n % a.world != a.rank:
            continue
        if (rec.get("cand_id") or rec.get("item_id")) in done:
            continue
        items.append(rec)
    log(f"task={a.cmd} items={len(items)} already_done={len(done)} world={a.world} out={out_path}")
    if not items:
        return
    engine = VLLMChat(a.model, max_model_len=a.max_model_len)
    samp = engine.sampling(temperature=GREEDY_TEMPERATURE, top_p=GREEDY_TOP_P, max_tokens=a.max_tokens)
    finding_name = finding_namer(a.class_names)
    t0, stats = time.time(), {"written": 0, "unparsed": 0, "too_long": 0}
    with out_path.open("a", encoding="utf-8") as stream:
        for start in range(0, len(items), a.chunk):
            msgs, metas = [], []
            for rec in items[start:start + a.chunk]:
                m = build_messages(a.cmd, rec, finding_name)
                if not engine.prompt_fits(m, a.max_tokens, safety_tokens=CONTEXT_LENGTH_SAFETY_TOKENS):
                    stats["too_long"] += 1
                    continue
                msgs.append(m)
                metas.append(rec)
            for rec, raw in zip(metas, engine.chat(msgs, samp)):
                p = parse(raw)
                if not isinstance(p, dict):
                    stats["unparsed"] += 1
                    p = {"applicable": False, "unparsed": True}
                stream.write(json.dumps(output_row(a.cmd, rec, p, raw), ensure_ascii=False) + "\n")
                stats["written"] += 1
            stream.flush()
            log(f"progress {min(start + a.chunk, len(items))}/{len(items)} {stats} {time.time() - t0:.0f}s")
    log(f"done {stats} {time.time() - t0:.0f}s")


def cmd_judge_input(a):
    """Join candidates with applicable Gemma corruptions -> judge input (edits equal to the source dropped)."""
    corrupted = {}
    for f in glob.glob(a.gemma_dir + "/rank_*.jsonl"):
        for r in read_jsonl(f):
            if r.get("applicable") and r.get("corrupted_sentence"):
                corrupted[r["cand_id"]] = r
    n = kept = 0
    with gzip.open(a.output, "wt") as out:
        for c in read_jsonl(a.candidates):
            g = corrupted.get(c["cand_id"])
            if not g:
                continue
            n += 1
            if g["corrupted_sentence"].strip() == c["sentence"].strip():
                continue
            kept += 1
            out.write(json.dumps(dict(item_id=c["cand_id"], attribute_type=c["attribute_type"],
                                      class_ids=c.get("class_ids", []), sentence=c["sentence"],
                                      corrupted_sentence=g["corrupted_sentence"], original_span=g.get("original_span"),
                                      new_span=g.get("new_span")), ensure_ascii=False) + "\n")
    print("joined", n, "judge items", kept, "from", len(corrupted), "applicable outputs")


def judge_accepts(parsed) -> bool:
    return isinstance(parsed, dict) and all(parsed.get(k) is True for k in JUDGE_KEYS)


def cmd_filter(a):
    """Keep the corruption records whose Gemma judgement is true on all five criteria (one output file)."""
    accepted = set()
    for f in glob.glob(a.judge_dir + "/rank_*.jsonl"):
        for r in read_jsonl(f):
            if judge_accepts(r.get("parsed")):
                accepted.add(r["item_id"])
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with (out_dir / "rank_000.jsonl").open("w", encoding="utf-8") as out:
        for f in sorted(glob.glob(a.gemma_dir + "/rank_*.jsonl")):
            for r in read_jsonl(f):
                if r.get("cand_id") in accepted:
                    out.write(json.dumps(r, ensure_ascii=False) + "\n")
                    n += 1
    print(f"[filter] judge-accepted={len(accepted)} kept corruption records={n} -> {out_dir}/rank_000.jsonl")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("corrupt", "judge"):
        g = sub.add_parser(name)
        g.add_argument("--input", required=True)
        g.add_argument("--out_dir", required=True)
        g.add_argument("--class_names", required=True, help="JSON list of the 1,028 class names (vocabulary order)")
        g.add_argument("--model", default="google/gemma-4-31B-it")
        g.add_argument("--max_model_len", type=int, default=16384)
        g.add_argument("--max_tokens", type=int, default=512)
        g.add_argument("--chunk", type=int, default=4096)
        g.add_argument("--limit", type=int, default=0, help="first N items overall (smoke)")
        g.add_argument("--rank", type=int, default=0)
        g.add_argument("--world", type=int, default=1)
    j = sub.add_parser("judge-input")
    j.add_argument("--candidates", required=True)
    j.add_argument("--gemma_dir", required=True)
    j.add_argument("--output", required=True)
    f = sub.add_parser("filter")
    f.add_argument("--gemma_dir", required=True)
    f.add_argument("--judge_dir", required=True)
    f.add_argument("--out_dir", required=True)
    a = ap.parse_args(argv)
    {"corrupt": cmd_generate, "judge": cmd_generate, "judge-input": cmd_judge_input, "filter": cmd_filter}[a.cmd](a)


if __name__ == "__main__":
    main()

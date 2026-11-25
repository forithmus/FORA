"""X-ray report finding extraction with Gemma-4-31B-it, one process per GPU (Methods, X-ray labels).

Three passes per report against the single closed X-ray vocabulary (304 leaves, ``data/cxr/vocabulary.json``):
(1) extraction with verbatim quotes, (2) verification against the report, (3) resolution with a second look for the
time-critical leaves listed in ``second_look``. Pass-3 lists are final; an unparsable pass 3 falls back to the pass-1 lists
(verdict ``p3_unparsed``). Eligible rows: findings >= 40 characters. Rows are sharded ``eligible[rank::world]`` and written
in chunks (``cxr_pathology_extracted_shard_rank_RR_chunk_CCCC.csv``, atomic, resumable). Merge with ``merge_shards.py``.

The same runner labels generated and reference reports for report scoring (``labels/report_scoring``).

    python -m labels.cxr.extract_findings --input cxr_input.csv --output-dir shards --log-dir logs --model <snapshot>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

from labels.ct.three_pass import KEYS_P1, KEYS_P3, STATUS_P3_UNPARSED, _lists, parse_json
from labels.llm import (CONTEXT_LENGTH_SAFETY_TOKENS, GREEDY_TEMPERATURE, GREEDY_TOP_P, STATUS_FAIL, STATUS_PASS,
                        STATUS_TOO_LONG, ChatMessages, inferer_from_config, load_config, rank_world, read_prompt, stage_path,
                        vocab_lines)

MIN_REPORT_CHARS = 40
COLS = ["ReportUID", "cxr_pathology_asserted", "cxr_pathology_denied", "cxr_pathology_uncertain", "cxr_pathology_other",
        "cxr_pathology_extraction_verdict"]


def bullets(xs) -> str:
    return "\n".join(f"- {x}" for x in xs) if xs else "- (none)"


class CXRThreePass:
    def __init__(self, cfg: dict):
        d = stage_path(cfg["prompt_dir"])
        self.P = {k: read_prompt(d / f"{k}.txt") for k in ("p1_system", "p1_user", "p2_system", "p2_user", "p3_system", "p3_user")}
        self.voc = json.loads(stage_path(cfg["vocabulary"]).read_text("utf-8"))
        self.vocab = vocab_lines(self.voc["leaves"])
        self.hints = bullets(self.voc["hints"])
        self.second_look = bullets(self.voc["second_look"])
        names = {lf["name"] for lf in self.voc["leaves"]}
        missing = [s for s in self.voc["second_look"] if s not in names]
        if missing:
            raise ValueError(f"second_look names that are not vocabulary leaves: {missing}")
        self.t1, self.t2, self.t3 = cfg["p1_max_tokens"], cfg["p2_max_tokens"], cfg["p3_max_tokens"]

    def p1(self, findings, impression) -> ChatMessages:
        v, P = self.voc, self.P
        return [{"role": "system", "content": P["p1_system"].format(title=v["title"], region=v["region"], hints=self.hints,
                                                                    vocabulary=self.vocab)},
                {"role": "user", "content": P["p1_user"].format(title=v["title"], findings=findings,
                                                                impression=impression or "(none)")}]

    def p2(self, findings, impression, labelling) -> ChatMessages:
        return [{"role": "system", "content": self.P["p2_system"].format(title=self.voc["title"], vocabulary=self.vocab)},
                {"role": "user", "content": self.P["p2_user"].format(findings=findings, impression=impression or "(none)",
                                                                     labelling=labelling)}]

    def p3(self, findings, impression, p1_json, p2_json) -> ChatMessages:
        v = self.voc
        return [{"role": "system", "content": self.P["p3_system"].format(title=v["title"], region=v["region"],
                                                                         second_look=self.second_look, vocabulary=self.vocab)},
                {"role": "user", "content": self.P["p3_user"].format(findings=findings, impression=impression or "(none)",
                                                                     p1_json=p1_json, p2_json=p2_json)}]

    def process(self, inferer, rows: list[tuple]) -> dict:
        mk = lambda t: inferer.make_sampling(temperature=GREEDY_TEMPERATURE, top_p=GREEDY_TOP_P, max_tokens=t)  # noqa: E731
        reserve = self.t1 + self.t2 + self.t3
        results, msgs1, live = {}, [], []
        for idx, f, imp in rows:
            m = self.p1(f, imp)
            if not inferer.prompt_fits(m, reserve, safety_tokens=CONTEXT_LENGTH_SAFETY_TOKENS)[0]:
                results[idx] = {"verdict": STATUS_TOO_LONG}
            else:
                msgs1.append(m)
                live.append((idx, f, imp))
        if not live:
            return results
        r1 = inferer.chat(msgs1, mk(self.t1))
        r2 = inferer.chat([self.p2(f, i, a) for (_, f, i), a in zip(live, r1)], mk(self.t2))
        r3 = inferer.chat([self.p3(f, i, a, b) for (_, f, i), a, b in zip(live, r1, r2)], mk(self.t3))
        for (idx, _, _), a, c in zip(live, r1, r3):
            p3p, p1p = parse_json(c), parse_json(a)
            other = p1p.get("other", []) if isinstance(p1p, dict) else []
            if p3p is not None:
                f = _lists(p3p, KEYS_P3)
                results[idx] = {"verdict": STATUS_PASS, "asserted": f["final_asserted"], "denied": f["final_denied"],
                                "uncertain": f["final_uncertain"], "other": other}
            elif p1p is not None:
                f = _lists(p1p, KEYS_P1)
                results[idx] = {"verdict": STATUS_P3_UNPARSED, "asserted": f["asserted"], "denied": f["denied"],
                                "uncertain": f["uncertain"], "other": other}
            else:
                results[idx] = {"verdict": STATUS_FAIL, "asserted": [], "denied": [], "uncertain": [], "other": []}
        return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/cxr_extract.json")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--log-dir", required=True, type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--chunk-size", type=int, help="override the config chunk size")
    ap.add_argument("--rank", type=int)
    ap.add_argument("--world", type=int)
    args = ap.parse_args()
    cfg = load_config(args.config)
    chunk_size = args.chunk_size or cfg["chunk_size"]
    rank, world = rank_world(args.rank, args.world)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    logf = open(args.log_dir / f"cxr_pathology_extraction_rank_{rank:02d}.log", "w")

    def log(m):
        logf.write(f"{m}\n")
        logf.flush()
        print(f"[rank {rank}] {m}", flush=True)

    df = pd.read_csv(args.input, low_memory=False)
    for col in ("ReportUID", "findings"):
        if col not in df.columns:
            log(f"ERROR: no '{col}' column in {args.input}")
            sys.exit(2)
    df = df.reset_index(drop=True)
    eligible = [i for i in range(len(df)) if isinstance(df.at[i, "findings"], str)
                and len(df.at[i, "findings"].strip()) >= MIN_REPORT_CHARS]
    shard = eligible[rank::world]
    log(f"rank {rank}/{world}: {len(shard)} of {len(eligible)} eligible rows")
    engine = CXRThreePass(cfg)
    inferer = inferer_from_config(args.model, cfg, log)
    chunks = [shard[i:i + chunk_size] for i in range(0, len(shard), chunk_size)]
    part = lambda c: args.output_dir / f"cxr_pathology_extracted_shard_rank_{rank:02d}_chunk_{c:04d}.csv"  # noqa: E731
    t0 = time.time()
    for c, idxs in enumerate(chunks):
        if part(c).exists():
            continue
        rows = []
        for i in idxs:
            imp = df.at[i, "impression"] if "impression" in df.columns else ""
            rows.append((i, str(df.at[i, "findings"]), imp if isinstance(imp, str) else ""))
        res = engine.process(inferer, rows)
        out = []
        for i, _, _ in rows:
            r = res.get(i, {"verdict": STATUS_FAIL})
            out.append({"ReportUID": df.at[i, "ReportUID"],
                        "cxr_pathology_asserted": json.dumps(r.get("asserted", []), ensure_ascii=False),
                        "cxr_pathology_denied": json.dumps(r.get("denied", []), ensure_ascii=False),
                        "cxr_pathology_uncertain": json.dumps(r.get("uncertain", []), ensure_ascii=False),
                        "cxr_pathology_other": json.dumps(r.get("other", []), ensure_ascii=False),
                        "cxr_pathology_extraction_verdict": r.get("verdict", STATUS_FAIL)})
        tmp = part(c).with_suffix(".csv.tmp")
        pd.DataFrame(out, columns=COLS).to_csv(tmp, index=False)
        os.replace(tmp, part(c))
        log(f"chunk {c + 1}/{len(chunks)}: {len(out)} rows | {dict(Counter(r['cxr_pathology_extraction_verdict'] for r in out))} | "
            f"{time.time() - t0:.0f}s")
    log(f"rank {rank} done in {time.time() - t0:.1f}s")
    logf.close()


if __name__ == "__main__":
    main()

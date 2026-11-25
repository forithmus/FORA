"""CT report finding extraction, one process per GPU (Methods, report labels).

Input CSV (``prepare_input.py``): ReportUID, findings, impression, protocol_class (1..29); a report may appear once per
protocol class it spans. Eligible rows: findings >= 100 characters and a protocol class in 1..29. Rows are sharded
``eligible[rank::world]``, grouped by protocol class, and each class is written to its own part file
(``pathology_extracted_shard_rank_RR_class_NN.csv``) via tmp + atomic rename, so a restarted process skips finished parts.

Output columns: ReportUID, findings, protocol_class, pathology_asserted / _denied / _uncertain (JSON lists of
{leaf, quote}), pathology_other (JSON list of {finding, quote}), pathology_extraction_verdict.

    python -m labels.ct.extract_findings --config labels/configs/ct_extract_global.json \
        --input global_input.csv --output-dir out/shards --log-dir out/logs --model <gemma-4-31B-it snapshot>
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

from labels.ct.three_pass import N_PROTOCOLS, ThreePass
from labels.llm import STATUS_FAIL, inferer_from_config, load_config, rank_world

MIN_REPORT_CHARS = 100
SHARD_COLS = ["ReportUID", "findings", "protocol_class", "pathology_asserted", "pathology_denied", "pathology_uncertain",
              "pathology_other", "pathology_extraction_verdict"]


def protocol_class(v) -> int | None:
    try:
        n = int(float(v))
        return n if 1 <= n <= N_PROTOCOLS else None
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--log-dir", required=True, type=Path)
    ap.add_argument("--model", required=True, help="local Gemma-4-31B-it snapshot directory")
    ap.add_argument("--rank", type=int)
    ap.add_argument("--world", type=int)
    args = ap.parse_args()
    cfg = load_config(args.config)
    rank, world = rank_world(args.rank, args.world)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    logf = open(args.log_dir / f"pathology_extraction_rank_{rank:02d}.log", "w")

    def log(m):
        logf.write(f"{m}\n")
        logf.flush()
        print(f"[rank {rank}] {m}", flush=True)

    df = pd.read_csv(args.input, low_memory=False)
    for col in ("protocol_class", "ReportUID", "findings"):
        if col not in df.columns:
            log(f"ERROR: no '{col}' column in {args.input}")
            sys.exit(2)
    df = df.reset_index(drop=True)
    eligible = [i for i in range(len(df)) if isinstance(df.at[i, "findings"], str)
                and len(df.at[i, "findings"].strip()) >= MIN_REPORT_CHARS and protocol_class(df.at[i, "protocol_class"])]
    shard = eligible[rank::world]
    log(f"rank {rank}/{world}: {len(shard)} of {len(eligible)} eligible rows")
    engine = ThreePass(cfg)
    inferer = inferer_from_config(args.model, cfg, log)
    by_class: dict[int, list] = {}
    for i in shard:
        imp = df.at[i, "impression"] if "impression" in df.columns else ""
        by_class.setdefault(protocol_class(df.at[i, "protocol_class"]), []).append(
            (i, str(df.at[i, "findings"]), imp if isinstance(imp, str) else ""))
    part = lambda n: args.output_dir / f"pathology_extracted_shard_rank_{rank:02d}_class_{n:02d}.csv"  # noqa: E731
    done = {n for n in range(1, N_PROTOCOLS + 1) if part(n).exists()}
    t0 = time.time()
    for n, rows in sorted(by_class.items()):
        if n in done:
            continue
        mod = engine.protocols.load(n)
        log(f"class {n} ({mod['slug']}): {len(rows)} reports, {len(mod['leaves'])} leaves")
        results = engine.process_modality(inferer, mod, rows)
        out = []
        for i, _, _ in rows:
            r = results.get(i, {"verdict": STATUS_FAIL})
            out.append({"ReportUID": df.at[i, "ReportUID"], "findings": df.at[i, "findings"], "protocol_class": n,
                        "pathology_asserted": json.dumps(r.get("asserted", []), ensure_ascii=False),
                        "pathology_denied": json.dumps(r.get("denied", []), ensure_ascii=False),
                        "pathology_uncertain": json.dumps(r.get("uncertain", []), ensure_ascii=False),
                        "pathology_other": json.dumps(r.get("other", []), ensure_ascii=False),
                        "pathology_extraction_verdict": r.get("verdict", STATUS_FAIL)})
        tmp = part(n).with_suffix(".csv.tmp")
        pd.DataFrame(out, columns=SHARD_COLS).to_csv(tmp, index=False)
        os.replace(tmp, part(n))
        log(f"class {n}: {len(out)} rows | {dict(Counter(r['pathology_extraction_verdict'] for r in out))} | "
            f"{time.time() - t0:.0f}s")
    log(f"rank {rank} done in {time.time() - t0:.1f}s")
    logf.close()


if __name__ == "__main__":
    main()

"""Label generated and reference CT reports with the scoring extractor, one process per GPU (Methods, report evaluation).

Generated and reference reports go through the identical three-pass extractor (``labels/ct/three_pass.py`` with the
scoring prompts and per-protocol vocabularies of ``configs/ct_scoring_extract.json``) on findings text only. Tasks
(``make_tasks.py``) are sorted by task_id and sharded ``tasks[rank::world]``; tasks with findings shorter than 100
characters get verdict ``too_short``. Results are appended per chunk (grouped by protocol class inside a chunk) to the
output JSONL; a restart skips task_ids already present.

Output row: task_id, source, uid, cohort, protocol_class, verdict, asserted, denied, uncertain, other.
Map rows to the 1,028 classes with ``ct_states.Resolver`` (group the rows of one (source, cohort, uid)).

    python -m labels.report_scoring.label_reports --tasks tasks.jsonl --output labels_rank00.jsonl --model <snapshot>
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

from labels.ct.three_pass import ThreePass
from labels.llm import inferer_from_config, load_config, rank_world

MIN_REPORT_CHARS = 100


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="report_scoring/configs/ct_scoring_extract.json")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--rank", type=int)
    ap.add_argument("--world", type=int)
    a = ap.parse_args()
    cfg = load_config(a.config)
    rank, world = rank_world(a.rank, a.world)
    log = lambda m: print(f"[rank {rank}] {m}", flush=True)  # noqa: E731
    done = set()
    if Path(a.output).exists():
        for line in open(a.output, encoding="utf-8"):
            try:
                done.add(json.loads(line)["task_id"])
            except json.JSONDecodeError:
                pass
    tasks = sorted((json.loads(line) for line in open(a.tasks, encoding="utf-8")), key=lambda t: t["task_id"])
    shard = [t for t in tasks[rank::world] if t["task_id"] not in done]
    log(f"{len(shard)} tasks to do (of {len(tasks)})")
    if not shard:
        return
    engine = ThreePass(cfg)
    inferer = None
    out = open(a.output, "a", encoding="utf-8")
    t0 = time.time()
    for s in range(0, len(shard), cfg["chunk_size"]):
        part = shard[s:s + cfg["chunk_size"]]
        results, by_class = {}, defaultdict(list)
        for i, t in enumerate(part):
            f = t["findings"] if isinstance(t["findings"], str) else ""
            if len(f.strip()) < MIN_REPORT_CHARS:
                results[i] = {"verdict": "too_short", "asserted": [], "denied": [], "uncertain": [], "other": []}
                continue
            by_class[int(t["protocol_class"])].append((i, f, t["impression"] or ""))
        if by_class and inferer is None:
            inferer = inferer_from_config(a.model, cfg, log)
        for n, rows in sorted(by_class.items()):
            results.update(engine.process_modality(inferer, engine.protocols.load(n), rows))
        for i, t in enumerate(part):
            r = results.get(i, {"verdict": "fail"})
            out.write(json.dumps({"task_id": t["task_id"], "source": t["source"], "uid": t["uid"], "cohort": t["cohort"],
                                  "protocol_class": t["protocol_class"], "verdict": r.get("verdict", "fail"),
                                  "asserted": r.get("asserted", []), "denied": r.get("denied", []),
                                  "uncertain": r.get("uncertain", []), "other": r.get("other", [])}, ensure_ascii=False) + "\n")
        out.flush()
        os.fsync(out.fileno())
        log(f"{s + len(part)}/{len(shard)} {dict(Counter(r.get('verdict') for r in results.values()))} {time.time() - t0:.0f}s")
    out.close()


if __name__ == "__main__":
    main()

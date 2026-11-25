"""DPO pair stores read by the CT DPO trainer (paper: "Preference optimization").

Store schema (sqlite):
  pairs(pair_id TEXT PRIMARY KEY, report_uid TEXT NOT NULL, pair_type TEXT NOT NULL, attribute_type TEXT,
        chosen_sha256 TEXT NOT NULL, rejected_text TEXT NOT NULL)          + INDEX pairs_uid ON pairs(report_uid)
  metadata(key TEXT PRIMARY KEY, value TEXT)   key 'summary' = JSON {schema: "ct_report_preference_index_v1",
        sources, pairs_by_type, reports, target_prefix: "Findings:\\n"[, policy]}
  provenance(pair_id, record)                  optional (plan-pair store only), full JSON record without texts

The chosen text is not stored: the trainer rebuilds it from the report and checks ``chosen_sha256``.

  python -m fora_report.pairs.merge index --pairs base_pairs.jsonl.gz attribute_pairs.jsonl.gz --output base_all.sqlite
  python -m fora_report.pairs.merge merge --base base_all.sqlite --new plan_pairs.sqlite --output dpo_pairs.sqlite
(``plan_pairs`` runs ``merge_stores`` itself when given ``--base_store``.)
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import sqlite3

from .text import TARGET_PREFIX, jdump

INDEX_SCHEMA = "ct_report_preference_index_v1"
PAIRS_DDL = ("CREATE TABLE pairs(pair_id TEXT PRIMARY KEY, report_uid TEXT NOT NULL, pair_type TEXT NOT NULL,"
             " attribute_type TEXT, chosen_sha256 TEXT NOT NULL, rejected_text TEXT NOT NULL)")


def create_store(path: Path):
    """New store at ``path.partial`` (renamed by the caller after commit)."""
    tmp = path.with_suffix(".partial")
    if tmp.exists():
        raise FileExistsError(tmp)
    db = sqlite3.connect(tmp)
    db.execute(PAIRS_DDL)
    db.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT)")
    return db, tmp


def cmd_index(args):
    """Pair jsonl(s) -> one sqlite store (train pairs only)."""
    out = Path(args.output)
    if out.exists():
        if not args.overwrite:
            raise FileExistsError(out)
        out.unlink()
    tmp = out.with_suffix(".partial")
    if tmp.exists():
        tmp.unlink()
    db, tmp = create_store(out)
    counts, reports = Counter(), set()
    for path in args.pairs:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                p = json.loads(line)
                if p["split"] != "train":
                    raise ValueError("non-train pair in " + path)
                if args.exclude_types and p["pair_type"] in args.exclude_types:
                    continue
                db.execute("INSERT OR IGNORE INTO pairs VALUES(?,?,?,?,?,?)",
                           (p["pair_id"], p["report_uid"], p["pair_type"], p.get("attribute_type"),
                            p["chosen_sha256"], p["rejected_text"]))
                counts[p["pair_type"]] += 1
                reports.add(p["report_uid"])
    db.execute("CREATE INDEX pairs_uid ON pairs(report_uid)")
    db.execute("INSERT INTO metadata VALUES(?,?)", ("summary", jdump(dict(
        schema=INDEX_SCHEMA, sources=[Path(p).name for p in args.pairs],
        pairs_by_type=dict(counts), reports=len(reports), target_prefix=TARGET_PREFIX))))
    db.commit()
    db.close()
    os.replace(tmp, out)
    print(f"[index] {out} pairs={sum(counts.values())} by_type={dict(counts)} reports={len(reports)}")


def merge_stores(base_store: str, new_rows, new_source: str, merged_store: Path, policy: str) -> dict:
    """merged = base rows (unchanged) + new rows (INSERT OR IGNORE on pair_id); returns pairs_by_type and reports."""
    if merged_store.exists():
        raise FileExistsError(merged_store)
    db, tmp = create_store(merged_store)
    base = sqlite3.connect(f"file:{base_store}?mode=ro", uri=True)
    base_summary = json.loads(base.execute("SELECT value FROM metadata WHERE key='summary'").fetchone()[0])
    cur = base.execute("SELECT pair_id, report_uid, pair_type, attribute_type, chosen_sha256, rejected_text FROM pairs")
    while True:
        chunk = cur.fetchmany(200000)
        if not chunk:
            break
        db.executemany("INSERT OR IGNORE INTO pairs VALUES(?,?,?,?,?,?)", chunk)
    db.executemany("INSERT OR IGNORE INTO pairs VALUES(?,?,?,?,?,?)", new_rows)
    db.execute("CREATE INDEX pairs_uid ON pairs(report_uid)")
    by_type = dict(db.execute("SELECT pair_type, COUNT(*) FROM pairs GROUP BY 1"))
    reports = db.execute("SELECT COUNT(DISTINCT report_uid) FROM pairs").fetchone()[0]
    db.execute("INSERT INTO metadata VALUES(?,?)", ("summary", jdump(dict(
        schema=INDEX_SCHEMA, sources=base_summary.get("sources", []) + [new_source],
        pairs_by_type=by_type, reports=reports, target_prefix=TARGET_PREFIX, policy=policy))))
    db.commit()
    db.close()
    os.replace(tmp, merged_store)
    return dict(pairs_by_type=by_type, reports=reports)


def cmd_merge(args):
    src = sqlite3.connect(f"file:{args.new}?mode=ro", uri=True)
    meta = json.loads(src.execute("SELECT value FROM metadata WHERE key='summary'").fetchone()[0])
    rows = src.execute("SELECT pair_id, report_uid, pair_type, attribute_type, chosen_sha256, rejected_text FROM pairs")
    r = merge_stores(args.base, rows, Path(args.new).name, Path(args.output), meta.get("policy"))
    print(f"[merge] {args.output} by_type={r['pairs_by_type']} reports={r['reports']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("index")
    i.add_argument("--pairs", nargs="+", required=True)
    i.add_argument("--output", required=True)
    i.add_argument("--exclude_types", nargs="*", default=[])
    i.add_argument("--overwrite", action="store_true")
    m = sub.add_parser("merge")
    m.add_argument("--base", required=True)
    m.add_argument("--new", required=True)
    m.add_argument("--output", required=True)
    args = ap.parse_args(argv)
    {"index": cmd_index, "merge": cmd_merge}[args.cmd](args)


if __name__ == "__main__":
    main()

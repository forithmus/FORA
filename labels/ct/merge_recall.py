"""Merge recall-pass outputs: first run + retry run -> one REC line per report.

Reports of the first run with status ok or too_long are kept; every other report takes its retry-run record (the retry
reprocesses the missing and unparsed reports with a larger generation budget). Several REC lines of one report are unioned.
``--retry-ids`` writes the ReportUIDs that need the retry run.

    python -m labels.ct.merge_recall --input global_input.csv --first run1/*.recs --retry-ids retry_ids.txt
    python -m labels.ct.merge_recall --first run1/*.recs --retry run2/*.recs --output recall_merged.recs
"""
from __future__ import annotations

import argparse
import csv
import json


def read(files: list[str]) -> dict:
    out = {}
    for f in files:
        for line in open(f, encoding="utf-8"):
            if not line.startswith("REC\t"):
                continue
            try:
                d = json.loads(line[4:])
            except Exception:
                continue
            o = out.setdefault(d["ReportUID"], dict(d, add=[]))
            o["add"] += d["add"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--first", nargs="+", required=True)
    ap.add_argument("--retry", nargs="*", default=[])
    ap.add_argument("--input", help="extraction input CSV (for --retry-ids)")
    ap.add_argument("--retry-ids")
    ap.add_argument("--output")
    a = ap.parse_args()
    first = {u: d for u, d in read(a.first).items() if d["status"] in ("ok", "too_long")}
    if a.retry_ids:
        csv.field_size_limit(10 ** 9)
        want = set()
        for r in csv.DictReader(open(a.input, encoding="utf-8")):
            try:
                pc = int(float(r["protocol_class"]))
            except Exception:
                continue
            if len(r["findings"].strip()) >= 20 and 1 <= pc <= 29 and r["ReportUID"] not in first:
                want.add(r["ReportUID"])
        open(a.retry_ids, "w").write("\n".join(sorted(want)) + "\n")
        print("retry", len(want))
        return
    merged = dict(first)
    for u, d in read(a.retry).items():
        merged.setdefault(u, d)
    with open(a.output, "w", encoding="utf-8") as f:
        f.writelines("REC\t" + json.dumps(d, ensure_ascii=False) + "\n" for d in merged.values())
    print(len(merged), "reports,", sum(len(d["add"]) for d in merged.values()), "additions")


if __name__ == "__main__":
    main()

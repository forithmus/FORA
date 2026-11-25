"""Merge the per-(rank, chunk) X-ray extraction parts into one CSV, one row per ReportUID.

Refuses to merge when a rank produced no parts (``--expect-ranks``) or a part lacks columns; duplicate ReportUIDs keep
the first row; malformed label cells are counted.

    python -m labels.cxr.merge_shards --shard-dir shards --out cxr_extracted.csv --expect-ranks 16
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from labels.cxr.extract_findings import COLS

LIST_COLS = COLS[1:5]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect-ranks", type=int, default=0)
    a = ap.parse_args()
    parts = sorted(Path(a.shard_dir).glob("cxr_pathology_extracted_shard_rank_*_chunk_*.csv"))
    if not parts:
        sys.exit(f"no shard parts in {a.shard_dir}")
    ranks = {int(m.group(1)) for p in parts if (m := re.search(r"_rank_(\d+)_", p.name))}
    if a.expect_ranks and (missing := set(range(a.expect_ranks)) - ranks):
        sys.exit(f"refusing to merge: ranks {sorted(missing)} produced no parts")
    frames = []
    for p in parts:
        d = pd.read_csv(p, low_memory=False)
        if miss := [c for c in COLS if c not in d.columns]:
            sys.exit(f"refusing to merge: {p.name} lacks columns {miss}")
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["ReportUID"].notna()].drop_duplicates("ReportUID", keep="first")
    bad = Counter()
    for col in LIST_COLS:
        for v in df[col]:
            try:
                ok = isinstance(v, str) and isinstance(json.loads(v), list)
            except Exception:
                ok = False
            bad[col] += not ok
    print(f"rows {len(df):,} | verdicts {dict(Counter(df['cxr_pathology_extraction_verdict'].astype(str)))} | malformed {dict(bad)}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()

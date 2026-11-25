#!/usr/bin/env python3
"""Flatten the filtered per-series metadata parquet shards into one CSV with the classification columns.

Usage
  python prepare_metadata.py -i FILTERED_PARQUET_DIR -o classification_metadata.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from config import CLASSIFICATION_COLUMNS


def main() -> None:
    p = argparse.ArgumentParser(
        description="Extract classification columns from filtered metadata shards into one CSV."
    )
    p.add_argument("-i", "--input", required=True, help="directory with the filtered metadata *.parquet shards")
    p.add_argument("-o", "--output", required=True, help="output CSV path")
    args = p.parse_args()

    files = sorted(Path(args.input).glob("*.parquet"))
    if not files:
        print(f"[!] No parquet files found in {args.input}", file=sys.stderr)
        sys.exit(1)

    frames: list[pd.DataFrame] = []
    for path in tqdm(files, desc="Reading", unit="shard"):
        table = pq.read_table(path)
        cols = [c for c in CLASSIFICATION_COLUMNS if c in table.schema.names]
        frames.append(table.select(cols).to_pandas())
        del table

    df = pd.concat(frames, ignore_index=True)
    df = df.reindex(columns=CLASSIFICATION_COLUMNS)

    dup_mask = df["SeriesInstanceUID"].duplicated(keep=False)
    n_dup = int(dup_mask.sum())
    if n_dup:
        n_uids = df.loc[dup_mask, "SeriesInstanceUID"].nunique()
        print(f"[!] {n_dup:,} rows share a duplicated SeriesInstanceUID ({n_uids:,} unique UID(s))")
    else:
        print("[*] No duplicated SeriesInstanceUID values found")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[✓] Wrote {len(df):,} rows × {len(df.columns)} columns → {out_path}")


if __name__ == "__main__":
    main()

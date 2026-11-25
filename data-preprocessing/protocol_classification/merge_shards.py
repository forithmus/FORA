#!/usr/bin/env python3
"""Merge the per-rank shards of classify.py onto every per-series row of the classification metadata.

Steps: concatenate the shard CSVs (unique StudyInstanceUID -> classification columns), check that every non-empty study
of the input is covered, left-join on StudyInstanceUID (every series row gains classified_protocols,
classification_verdict, classification_qc_issues; rows without a study UID get blanks) and write the CSV.

Usage
  python merge_shards.py --input classification_metadata.csv --shard-dir SHARDS --output classified_protocols.csv
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import pandas as pd

from config import CLS_SHARD_GLOB, PROTOCOL_COL, QC_ISSUES_COL, QC_STATUS_FAIL, STATUS_OTHER, STUDY_COL, VERDICT_COL

# The three columns each shard carries back onto every original row.
MERGE_COLS = [PROTOCOL_COL, VERDICT_COL, QC_ISSUES_COL]


# ─────────────────────────────────────────────────────────────────────────────
# Shard CSV merge
# ─────────────────────────────────────────────────────────────────────────────
def _read_shards(shard_dir: Path, shard_glob: str) -> pd.DataFrame:
    """Load and concatenate per-rank shard CSVs; dedupe on the study UID."""
    files = sorted(glob.glob(str(shard_dir / shard_glob)))
    if not files:
        raise SystemExit(
            f"No shard files matching {shard_glob} under {shard_dir}"
        )
    print(f"Found {len(files)} shard CSV files in {shard_dir}")
    frames = []
    for f in files:
        d = pd.read_csv(f, encoding="utf-8-sig", low_memory=False)
        print(f"  {os.path.basename(f)}: {len(d)} studies")
        frames.append(d)
    merged = pd.concat(frames, ignore_index=True)

    if STUDY_COL in merged.columns:
        before = len(merged)
        merged = merged.drop_duplicates(subset=[STUDY_COL], keep="last")
        if before != len(merged):
            print(
                f"  Cross-shard dedup: {before} -> {len(merged)} "
                f"(shards should be disjoint; duplicates would indicate "
                f"a sharding bug or a resumed run)"
            )
    return merged.reset_index(drop=True)


def _verdict_label(value) -> str:
    """Bucket a ``classification_verdict`` cell for summary counts."""
    if pd.isna(value) or value == "":
        return "empty"
    return str(value)  # "pass" / "fail" / "too_long"


def _redup_onto_original(
    original_csv: Path,
    dedup_cls: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Left-merge shard classifications back onto every per-series row.

    Join key is ``StudyInstanceUID``. Rows with an empty study UID (NaN or "")
    get blank classification columns. The final CSV is the original columns plus
    ``classified_protocols``, ``classification_verdict``, ``classification_qc_issues``.
    """
    orig = pd.read_csv(original_csv, low_memory=False)
    if STUDY_COL not in orig.columns:
        raise SystemExit(
            f"Input CSV {original_csv} must contain '{STUDY_COL}' "
            f"column; got {list(orig.columns)}"
        )

    missing_shard_cols = [c for c in MERGE_COLS if c not in dedup_cls.columns]
    if missing_shard_cols:
        raise SystemExit(
            f"Shard CSVs must contain {missing_shard_cols} column(s); "
            f"got {list(dedup_cls.columns)}"
        )

    empty_mask = orig[STUDY_COL].isna() | (orig[STUDY_COL] == "")

    # Every unique non-empty study UID in the input must appear in the shards.
    input_uniques = set(orig.loc[~empty_mask, STUDY_COL].tolist())
    shard_uniques = set(dedup_cls[STUDY_COL].tolist())
    missing_uniques = input_uniques - shard_uniques
    if missing_uniques:
        sample = list(missing_uniques)[:3]
        raise SystemExit(
            f"Shard coverage incomplete: {len(missing_uniques)} unique "
            f"non-empty {STUDY_COL!r} values in input are absent from "
            f"shards ({len(input_uniques)} unique in input, "
            f"{len(shard_uniques)} unique in shards). "
            f"Sample missing keys: {sample!r}"
        )
    print(
        f"Shard coverage OK: all {len(input_uniques)} unique non-empty "
        f"{STUDY_COL} values present in shards."
    )

    n_orig = len(orig)
    merged = orig.merge(
        dedup_cls[[STUDY_COL] + MERGE_COLS],
        on=STUDY_COL,
        how="left",
    )
    if len(merged) != n_orig:
        raise SystemExit(
            f"Merge changed row count ({n_orig} -> {len(merged)}). "
            f"Duplicate {STUDY_COL!r} keys in shard CSVs would inflate "
            f"rows — check cross-shard dedup and shard outputs."
        )

    # Non-empty study rows unexpectedly missing from shards (shouldn't happen —
    # coverage check above would have caught it; mark fail defensively).
    missing_mask = (~empty_mask) & merged[PROTOCOL_COL].isna()
    n_missing = int(missing_mask.sum())
    if n_missing:
        print(
            f"WARNING: {n_missing} non-empty study rows had no matching "
            f"classification in shards — marking {STATUS_OTHER}/{QC_STATUS_FAIL}."
        )
        merged.loc[missing_mask, PROTOCOL_COL] = STATUS_OTHER
        merged.loc[missing_mask, VERDICT_COL] = QC_STATUS_FAIL

    # Empty-study rows (no inference): blank all classification columns.
    for col in MERGE_COLS:
        merged.loc[empty_mask, col] = ""
        # CSV round-trips blank cells back as NaN — normalize to "".
        merged[col] = merged[col].fillna("")

    n_total_rows = len(merged)
    n_empty_rows = int(empty_mask.sum())
    n_studies = len(dedup_cls)

    verdict_labels = merged[VERDICT_COL].map(_verdict_label)
    row_verdict_counts = verdict_labels.value_counts(dropna=False).to_dict()
    # Per-study (not per-row) verdict + protocol distributions.
    study_verdict_counts = (
        dedup_cls[VERDICT_COL].map(_verdict_label)
        .value_counts(dropna=False).to_dict()
    )
    top_protocols = dedup_cls[PROTOCOL_COL].value_counts().head(15).to_dict()

    stats = {
        "input_rows_total": n_total_rows,
        "input_rows_empty_study_skipped": n_empty_rows,
        "input_rows_with_study": n_total_rows - n_empty_rows,
        "unique_studies_classified": n_studies,
        "row_verdict_counts": row_verdict_counts,
        "study_verdict_counts": study_verdict_counts,
        "top_protocols_by_study": top_protocols,
    }
    return merged, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path, help="per-series classification metadata CSV")
    ap.add_argument("--shard-dir", required=True, type=Path, help="directory with the per-rank shard CSVs")
    ap.add_argument("--output", required=True, type=Path, help="merged CSV")
    ap.add_argument("--shard-glob", default=CLS_SHARD_GLOB)
    args = ap.parse_args()
    dedup = _read_shards(args.shard_dir, args.shard_glob)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged, stats = _redup_onto_original(args.input, dedup)
    merged.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Final merged CSV: {args.output}  ({len(merged)} rows)")
    print(f"  unique studies classified: {stats['unique_studies_classified']}")
    print(f"  study verdict_counts: {stats['study_verdict_counts']}")


if __name__ == "__main__":
    sys.exit(main())

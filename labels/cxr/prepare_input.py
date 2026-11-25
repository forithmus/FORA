"""Build the X-ray extraction input: one row per ReportUID with structured findings and impression (Methods, X-ray labels).

Rows with findings shorter than 40 characters (the extractor's eligibility gate) are dropped; duplicate ReportUID rows
(accession-grained source) are collapsed after checking that their findings text is identical.

    python -m labels.cxr.prepare_input --source cxr_structured_reports.csv --out cxr_input.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

MIN_REPORT_CHARS = 40
KEEP = ["PatientUID", "ReportUID", "findings", "impression"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = pd.read_csv(a.source, low_memory=False, usecols=lambda c: c in KEEP or c == "structuring_verdict")
    fnd = df["findings"].astype(str).str.strip()
    has_text = ~fnd.isin(("", "nan", "None"))
    df = df[has_text & (fnd.str.len() >= MIN_REPORT_CHARS)].copy()
    dup = df[df.duplicated("ReportUID", keep=False)]
    if len(dup):
        conflicts = (dup.groupby("ReportUID")["findings"].nunique() > 1).sum()
        print(f"duplicate-ReportUID rows: {len(dup):,} | findings conflicts: {conflicts}")
    df = df.drop_duplicates("ReportUID", keep="first")
    df = df[[c for c in KEEP if c in df.columns]]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out} ({len(df):,} reports)")


if __name__ == "__main__":
    main()

"""Report-level protocol classes: one row per (ReportUID, protocol class) of the H5 cohort (Methods, "Protocol classes").

The protocol classifier labels each study (StudyInstanceUID) with a list of protocol titles; the ingestion manifest
copies that list onto every series row. A report can span several studies, so it is emitted once per distinct protocol
class n (1..29 of protocol_classification/protocol_classes.json) over its studies. "CT Liver - Four-Phase" maps to
class 3 ("CT Abdomen and Pelvis - Multiphase"), "other" is dropped. Only manifest rows that are in an H5 shard
(h5_id != "None") with an attached report (enriched_with_report == True) count. Reports without findings text or with
findings shorter than --min-chars are dropped. The output feeds the label extraction (per-protocol vocabularies), the
split (make_split.py) and the exam-type grouping (exam_types.py).

Usage
  python report_protocol_classes.py --manifest ingestion_manifest.csv --reports structured_reports.csv --output OUT.csv
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

CLASSES = json.loads((Path(__file__).resolve().parent / 'protocol_classification/protocol_classes.json').read_text())
PROTOCOL_TITLES = [c['title'] for c in sorted(CLASSES, key=lambda c: c['n'])]
TITLE_TO_N = {t: i + 1 for i, t in enumerate(PROTOCOL_TITLES)}
TITLE_TO_N["CT Liver - Four-Phase"] = 3
MIN_REPORT_CHARS = 100
MANIFEST_COLS = ["ReportUID", "StudyInstanceUID", "h5_id", "enriched_with_report", "classified_protocols"]


def build_reportuid_classes(manifest: Path, chunksize: int = 500_000) -> dict[str, list[int]]:
    """ReportUID -> every distinct protocol class (1..29) it spans, over the H5 cohort."""
    study_labels: dict[str, tuple] = {}
    report_studies: dict[str, set] = defaultdict(set)
    for chunk in pd.read_csv(manifest, usecols=MANIFEST_COLS, chunksize=chunksize, dtype=str):
        for rpt, study, h5, enriched, raw in zip(chunk["ReportUID"], chunk["StudyInstanceUID"], chunk["h5_id"],
                                                 chunk["enriched_with_report"], chunk["classified_protocols"]):
            if not (isinstance(h5, str) and h5 != "None") or str(enriched) != "True":
                continue
            rpt, study = str(rpt), str(study)
            report_studies[rpt].add(study)
            if study not in study_labels:
                try:
                    study_labels[study] = tuple(json.loads(raw)) if isinstance(raw, str) else ()
                except Exception:
                    study_labels[study] = ()
    out: dict[str, list[int]] = {}
    for rpt, studies in report_studies.items():
        classes = {TITLE_TO_N[label] for study in studies for label in study_labels.get(study, ()) if label in TITLE_TO_N}
        if classes:
            out[rpt] = sorted(classes)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path, help="ingestion manifest with h5_id, enriched_with_report, classified_protocols")
    ap.add_argument("--reports", required=True, type=Path, help="structured reports CSV")
    ap.add_argument("--report-uid-col", default="ReportUID")
    ap.add_argument("--findings-col", default="findings")
    ap.add_argument("--keep-cols", default="ReportUID,PatientUID,findings,impression")
    ap.add_argument("--min-chars", type=int, default=MIN_REPORT_CHARS)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--chunksize", type=int, default=200_000)
    args = ap.parse_args()

    rpt2classes = build_reportuid_classes(args.manifest)
    print(f"cohort: {len(rpt2classes):,} ReportUIDs -> {sum(len(c) for c in rpt2classes.values()):,} (ReportUID, class) pairs")
    keep = [c.strip() for c in args.keep_cols.split(",") if c.strip()]
    seen: set = set()
    parts: list = []
    n_in = n_dropped = 0
    uid, fnd = args.report_uid_col, args.findings_col
    for chunk in pd.read_csv(args.reports, usecols=keep, chunksize=args.chunksize, dtype=str):
        n_in += len(chunk)
        txt = chunk[fnd].astype(str).str.strip()
        has_text = chunk[fnd].notna() & (txt.str.len() >= args.min_chars)
        n_dropped += int((~has_text).sum())
        chunk = chunk[has_text]
        chunk = chunk[~chunk[uid].isin(seen)].drop_duplicates(subset=[uid])
        seen.update(chunk[uid])
        chunk = chunk.assign(protocol_class=chunk[uid].astype(str).map(rpt2classes))
        chunk = chunk[chunk["protocol_class"].notna()].explode("protocol_class")
        if len(chunk):
            parts.append(chunk)
    df = pd.concat(parts, ignore_index=True)
    df["protocol_class"] = df["protocol_class"].astype("Int64")
    per_class = Counter(df["protocol_class"].tolist())
    print(f"rows in: {n_in:,} (dropped {n_dropped:,}) | reports out: {df[uid].nunique():,} | rows out: {len(df):,}")
    for n in range(1, 30):
        if per_class.get(n):
            print(f"   class {n:>2} {PROTOCOL_TITLES[n - 1][:52]:52} : {per_class[n]:>8,}")
    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()

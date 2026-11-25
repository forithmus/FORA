"""Build the CT extraction input: one row per (ReportUID, protocol class) of the imaging cohort (Methods, report labels).

Reads the structured reports (one or more rows per ReportUID; the structured text is identical across a report's rows) and
the ingestion manifest (one row per series: ReportUID, StudyInstanceUID, h5_id, enriched_with_report,
classified_protocols as a JSON list of protocol titles). Cohort = manifest rows with an H5 shard and an attached report.
A report is emitted once per distinct protocol class (1..29) over its studies; class 14 (CT Liver four-phase) is mapped to
class 3. Rows with findings shorter than 100 characters are dropped (the extractor's eligibility gate).

    python -m labels.ct.prepare_input --manifest manifest.csv --reports structured_reports.csv --output input.csv
    python -m labels.ct.prepare_input ... --retry-from out/shards --retry-output retry_input.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

MIN_REPORT_CHARS = 100
PROTOCOL_TITLES = [
    "CT Abdomen - Renal Stone Protocol", "CT Abdomen and Pelvis - IV Contrast",
    "CT Abdomen and Pelvis - Multiphase", "CT Abdomen and Pelvis - Non-Contrast",
    "CT Cervical Spine", "CT Chest - IV Contrast",
    "CT Chest - Non-Contrast (incl. HRCT / Low-Dose Screening)",
    "CT Chest, Abdomen and Pelvis - IV Contrast", "CT Chest, Abdomen and Pelvis - Multiphase",
    "CT Chest, Abdomen and Pelvis - Non-Contrast", "CT Extremity",
    "CT Head - IV Contrast", "CT Head - Non-Contrast", "CT Liver - Four-Phase",
    "CT Lumbar Spine", "CT Neck - IV Contrast", "CT Neck - Non-Contrast",
    "CT Sinus / Facial Bones / Orbits", "CT Temporal Bones", "CT Thoracic Spine",
    "CT Urography", "CTA Abdomen", "CTA Chest / PE / Aorta", "CTA Extremity Runoff",
    "CTA Head", "CTA Head and Neck", "CTA Neck", "Coronary CTA / Calcium Score",
    "Dental Cone-Beam CT",
]
TITLE_TO_N = {t: i + 1 for i, t in enumerate(PROTOCOL_TITLES)}
TITLE_TO_N["CT Liver - Four-Phase"] = 3
MANIFEST_COLS = ["ReportUID", "StudyInstanceUID", "h5_id", "enriched_with_report", "classified_protocols"]


def report_classes(manifest: Path, chunksize: int = 500_000) -> dict[str, list[int]]:
    """ReportUID -> every distinct protocol class it spans (protocol labels de-duplicated per study)."""
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
    out = {}
    for rpt, studies in report_studies.items():
        classes = {TITLE_TO_N[label] for s in studies for label in study_labels.get(s, ()) if label in TITLE_TO_N}
        if classes:
            out[rpt] = sorted(classes)
    return out


def build_input(manifest: Path, reports: Path, output: Path, keep_cols: list[str], chunksize: int = 200_000) -> None:
    classes = report_classes(manifest)
    print(f"cohort: {len(classes):,} reports -> {sum(map(len, classes.values())):,} (report, class) pairs")
    seen: set = set()
    parts = []
    for chunk in pd.read_csv(reports, usecols=keep_cols, chunksize=chunksize, dtype=str):
        txt = chunk["findings"].astype(str).str.strip()
        chunk = chunk[chunk["findings"].notna() & (txt.str.len() >= MIN_REPORT_CHARS)]
        chunk = chunk[~chunk["ReportUID"].isin(seen)].drop_duplicates(subset=["ReportUID"])
        seen.update(chunk["ReportUID"])
        chunk = chunk.assign(protocol_class=chunk["ReportUID"].astype(str).map(classes))
        chunk = chunk[chunk["protocol_class"].notna()].explode("protocol_class")
        if len(chunk):
            parts.append(chunk)
    df = pd.concat(parts, ignore_index=True)
    df["protocol_class"] = df["protocol_class"].astype("Int64")
    print(f"reports: {df['ReportUID'].nunique():,} | rows: {len(df):,} | per class {dict(sorted(Counter(df['protocol_class']).items()))}")
    df.to_csv(output, index=False)


def build_retry_input(shard_dir: Path, input_csv: Path, output: Path) -> None:
    """Reports with any non-pass row in a finished run -> their input rows (input of the next retry run)."""
    csv.field_size_limit(10 ** 9)
    bad = set()
    for f in glob.glob(str(shard_dir / "*.csv")):
        for r in csv.DictReader(open(f)):
            if r["pathology_extraction_verdict"] != "pass":
                bad.add(r["ReportUID"])
    rd = csv.DictReader(open(input_csv))
    w = csv.DictWriter(open(output, "w"), fieldnames=rd.fieldnames)
    w.writeheader()
    n = 0
    for r in rd:
        if r["ReportUID"] in bad:
            w.writerow(r)
            n += 1
    print("retry rows", n)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--reports", type=Path)
    ap.add_argument("--keep-cols", default="ReportUID,PatientUID,findings,impression")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--retry-from", type=Path, help="shard dir of a finished run")
    ap.add_argument("--retry-input", type=Path, help="input CSV of that run")
    ap.add_argument("--retry-output", type=Path)
    a = ap.parse_args()
    if a.retry_from:
        build_retry_input(a.retry_from, a.retry_input, a.retry_output)
    else:
        build_input(a.manifest, a.reports, a.output, [c.strip() for c in a.keep_cols.split(",") if c.strip()])


if __name__ == "__main__":
    main()

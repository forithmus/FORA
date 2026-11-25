"""Study manifests for CT report generation: FORA val, CT-RATE val and Merlin test in one format.

A manifest is a JSON-lines file with one study per line:

    {"uid": <report uid>, "case": <MIL case dict: id, fingerprint, views[...] of the block-23 cache>,
     "study_type_ids": [int, ...], "reference_findings": <str or null>, "protocol_class": <int, optional>}

`case` is the case record of the MIL cache (classifier stage); its block-23 tokens and region bits must be on local
disk under `<cache_root>/<fingerprint>/`. Builders:

  fora    report-generation index (split column of the patient-level split) + MIL cases jsonl -> all studies of one split; study types
          and reference findings come from the index / H5 report group (DinoReportDataset).
  ctrate  CT-RATE cohort json {uid: {study_type_id, labeler_protocol_class, reference_findings}} + cases jsonl.
  merlin  Merlin reports csv (`study id`, `Findings`) + labeller protocol csv (`ReportUID`, `protocol_class`) +
          cases jsonl; protocol class 2 (CT A/P IV contrast) -> study type 2, 4 (CT A/P non-contrast) -> 7.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

# labeller protocol class -> FORA study-type id (protocol classes of the FORA index)
MERLIN_PROTOCOL_TO_STUDY_TYPE = {2: 2,   # CT Abdomen and Pelvis - IV Contrast
                                 4: 7}   # CT Abdomen and Pelvis - Non-Contrast


def findings_section(text: str) -> str:
    """Merlin 'Findings' cell -> findings only: after FINDINGS: up to IMPRESSION: (or the first numbered '1.' line)."""
    t = str(text).replace("\r", "")
    m = re.search(r"FINDINGS\s*:", t)
    body = t[m.end():] if m else t
    e = re.search(r"IMPRESSION\s*:", body) or re.search(r"(?m)^\s*1\.", body)
    if e:
        body = body[:e.start()]
    return " ".join(body.split()).rstrip(" :")


def read_cases(path, wanted_reports=None) -> dict:
    """report uid -> MIL case (cases jsonl of the classifier cache)."""
    out = {}
    with open(path) as handle:
        for line in handle:
            case = json.loads(line)
            uid = str(case["report"])
            if wanted_reports is None or uid in wanted_reports:
                out[uid] = case
    return out


def read_manifest(path) -> list[dict]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_manifest(path, studies) -> int:
    studies = sorted(studies, key=lambda s: str(s["uid"]))
    with open(path, "w") as handle:
        for s in studies:
            handle.write(json.dumps(s, ensure_ascii=False) + "\n")
    return len(studies)


def verify_staged(case: dict, cache_root: Path) -> None:
    """Block-23 cache + region bits of every view must be on local disk with the recorded size."""
    root = Path(cache_root) / case["fingerprint"]
    for view in case["views"]:
        path = root / view["relative_file"]
        if not path.is_file() or path.stat().st_size != int(view["file_size"]):
            raise RuntimeError(f"{case['id']}: block-23 cache not staged: {path}")
        if not (root / "regions" / (view["relative_file"] + ".npz")).is_file():
            raise RuntimeError(f"{case['id']}: region bits not staged for {view['relative_file']}")


def fora_studies(index_path, mil_config_path, h5_root, cache_root, split="val") -> list[dict]:
    from fora_report.ct.report_data import DinoReportDataset
    from fora.reportgen.writer.dino_stream import read_index_splits
    config = json.loads(Path(mil_config_path).read_text())
    splits = read_index_splits(index_path)
    wanted = {u for u, s in splits.items() if s == split}
    cases = read_cases(config["cases"], wanted)
    dataset = DinoReportDataset(cases_by_uid=cases, mil_config=config, cache_root=cache_root, index_path=index_path,
                                h5_root=h5_root, split=split)
    return [{"uid": str(s["report_uid"]), "case": cases[str(s["report_uid"])],
             "study_type_ids": [int(t) for t in s["study_type_ids"]], "reference_findings": dataset.report_text(s)}
            for s in dataset.index]


def ctrate_studies(cohort_json, cases_path) -> list[dict]:
    meta = json.loads(Path(cohort_json).read_text())
    cases = read_cases(cases_path, set(meta))
    return [{"uid": uid, "case": cases[uid], "study_type_ids": [int(meta[uid]["study_type_id"])],
             "reference_findings": meta[uid].get("reference_findings"),
             "protocol_class": int(meta[uid]["labeler_protocol_class"])} for uid in sorted(cases)]


def merlin_studies(reports_csv, protocol_csv, cases_path) -> list[dict]:
    csv.field_size_limit(1 << 30)
    with open(reports_csv, newline="") as handle:
        reports = {r["study id"]: r for r in csv.DictReader(handle)}
    with open(protocol_csv, newline="") as handle:
        protocols = {r["ReportUID"]: int(r["protocol_class"]) for r in csv.DictReader(handle)}
    cases = read_cases(cases_path)
    out = []
    for uid in sorted(cases):
        protocol = protocols[uid]
        out.append({"uid": uid, "case": cases[uid], "study_type_ids": [MERLIN_PROTOCOL_TO_STUDY_TYPE[protocol]],
                    "reference_findings": findings_section(reports[uid]["Findings"]) if uid in reports else None,
                    "protocol_class": protocol})
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cohort", required=True)
    f = sub.add_parser("fora")
    f.add_argument("--index", required=True)
    f.add_argument("--mil-config", required=True)
    f.add_argument("--h5-root", required=True)
    f.add_argument("--cache-root", required=True, help="local root of the block-23 caches")
    f.add_argument("--split", default="val")
    c = sub.add_parser("ctrate")
    c.add_argument("--cohort-json", required=True)
    c.add_argument("--cases", required=True)
    m = sub.add_parser("merlin")
    m.add_argument("--reports-csv", required=True)
    m.add_argument("--protocol-csv", required=True)
    m.add_argument("--cases", required=True)
    for s in (f, c, m):
        s.add_argument("--output", required=True)
    a = p.parse_args(argv)
    if a.cohort == "fora":
        studies = fora_studies(a.index, a.mil_config, a.h5_root, a.cache_root, a.split)
    elif a.cohort == "ctrate":
        studies = ctrate_studies(a.cohort_json, a.cases)
    else:
        studies = merlin_studies(a.reports_csv, a.protocol_csv, a.cases)
    print(f"{a.output}: {write_manifest(a.output, studies)} studies")


if __name__ == "__main__":
    main()

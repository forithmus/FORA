#!/usr/bin/env python3
"""Build the X-ray label index (SQLite) from the merged extraction CSV (Methods, X-ray labels).

Vocabulary = the 304 closed leaves plus every unresolved extractor spelling kept as an explicit extra label (7 in the
paper run, 311 labels). Emitted names resolve to a leaf by normalised canonical name or unique synonym. Per report
(verdict pass): positive = asserted OR uncertain, negative = denied AND NOT positive; asserted > uncertain > denied.
Reports are routed to H5 shard and split by the ingestion manifest and the split CSV. The paper run used
``--include_empty_targets``.

    python -m labels.cxr.build_index --manifest manifest.csv --labels cxr_extracted.csv --splits cxr_splits.csv \
        --vocabulary labels/data/cxr/vocabulary.json --h5_root <h5 dir> --output cxr_labels_index.sqlite --include_empty_targets
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--vocabulary", required=True, type=Path)
    parser.add_argument("--h5_root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--include_empty_targets",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Retain pass-verdict reports with no extracted states as all-unmentioned.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(value: str) -> str:
    value = value.casefold().replace("–", "-").replace("—", "-")
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def parse_records(value: str) -> list[dict[str, str]]:
    if not value.strip():
        return []
    parsed = json.loads(value)
    records = []
    for item in parsed:
        if not isinstance(item, dict) or not item.get("leaf"):
            continue
        records.append(
            {
                "leaf": str(item["leaf"]),
                "quote": str(item.get("quote", "")).strip(),
            }
        )
    return records


def parse_items(value: str) -> list[str]:
    return [record["leaf"] for record in parse_records(value)]


def atomic_json(path: Path, value) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    for path in (args.manifest, args.labels, args.splits, args.vocabulary, args.h5_root):
        if not path.exists():
            raise FileNotFoundError(path)

    leaves = json.loads(args.vocabulary.read_text(encoding="utf-8"))["leaves"]
    if not isinstance(leaves, list) or not leaves:
        raise ValueError("CXR vocabulary LEAVES must be a nonempty list")
    labels = [str(item["name"]) for item in leaves]
    if len(labels) != len(set(labels)):
        raise ValueError("Duplicate CXR vocabulary leaves")
    groups = {str(item["name"]): str(item.get("group", "")) for item in leaves}
    types = {str(item["name"]): str(item.get("type", "")) for item in leaves}
    synonyms = {str(item["name"]): str(item.get("syn", "")) for item in leaves}

    aliases: dict[str, set[str]] = defaultdict(set)
    for label in labels:
        aliases[normalize(label)].add(label)
        for synonym in synonyms[label].split(";"):
            synonym = synonym.strip()
            if synonym:
                aliases[normalize(synonym)].add(label)

    # Determine every emitted spelling first. Unique configured aliases map to
    # their canonical leaf. True extractor extras remain explicit, auditable heads.
    observed = Counter()
    verdicts = Counter()
    with args.labels.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            verdict = row["cxr_pathology_extraction_verdict"].strip()
            verdicts[verdict] += 1
            if verdict != "pass":
                continue
            for polarity in ("asserted", "denied", "uncertain"):
                for raw in parse_items(row[f"cxr_pathology_{polarity}"]):
                    observed[raw] += 1

    resolver: dict[str, str] = {}
    unresolved: dict[str, Counter] = defaultdict(Counter)
    resolver_audit = Counter()
    for raw, count in observed.items():
        matches = aliases.get(normalize(raw), set())
        if len(matches) == 1:
            resolver[raw] = next(iter(matches))
            resolver_audit["canonical_or_synonym_instances"] += count
        else:
            unresolved[normalize(raw)][raw] += count
    extras = []
    for variants in unresolved.values():
        canonical = min(variants, key=lambda raw: (-variants[raw], raw))
        extras.append(canonical)
        for raw, count in variants.items():
            resolver[raw] = canonical
            resolver_audit["extractor_extra_instances"] += count
    extras.sort(key=lambda value: (normalize(value), value))
    for label in extras:
        labels.append(label)
        groups[label] = "Extractor extras"
        types[label] = "extractor_extra"
        synonyms[label] = ""
    label_to_index = {label: index for index, label in enumerate(labels)}

    with args.splits.open(newline="", encoding="utf-8-sig") as handle:
        split_by_h5 = {
            row["h5_id"].strip(): row["split"].strip()
            for row in csv.DictReader(handle)
        }
    routes: dict[str, tuple[str, str]] = {}
    route_conflicts = Counter()
    manifest_rows = 0
    with args.manifest.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            manifest_rows += 1
            h5_id = row["h5_id"].strip()
            report_uid = row["ReportUID"].strip()
            if (
                not report_uid
                or row["enriched_with_report"].strip() != "True"
                or h5_id not in split_by_h5
            ):
                continue
            route = (h5_id, split_by_h5[h5_id])
            previous = routes.setdefault(report_uid, route)
            if previous != route:
                route_conflicts[report_uid] += 1
    if route_conflicts:
        raise ValueError(f"Report-to-shard conflicts: {len(route_conflicts)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    if temporary.exists():
        temporary.unlink()
    connection = sqlite3.connect(temporary)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.executescript(
        """
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE samples (
            report_uid TEXT PRIMARY KEY,
            h5_id TEXT NOT NULL,
            split TEXT NOT NULL,
            positive_json TEXT NOT NULL,
            negative_json TEXT NOT NULL,
            asserted_json TEXT NOT NULL,
            uncertain_json TEXT NOT NULL,
            denied_json TEXT NOT NULL,
            asserted_evidence_json TEXT NOT NULL,
            uncertain_evidence_json TEXT NOT NULL,
            denied_evidence_json TEXT NOT NULL,
            extraction_verdict TEXT NOT NULL
        );
        CREATE INDEX samples_lookup ON samples (split, h5_id);
        """
    )
    positive_counts = [0] * len(labels)
    negative_counts = [0] * len(labels)
    asserted_counts = [0] * len(labels)
    uncertain_counts = [0] * len(labels)
    denied_counts = [0] * len(labels)
    counts = Counter()
    missing_manifest = 0
    label_rows = 0
    with args.labels.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            label_rows += 1
            if row["cxr_pathology_extraction_verdict"].strip() != "pass":
                continue
            report_uid = row["ReportUID"].strip()
            route = routes.get(report_uid)
            if route is None:
                missing_manifest += 1
                continue
            def resolved_evidence(column: str) -> dict[str, list[str]]:
                evidence: dict[str, list[str]] = defaultdict(list)
                for record in parse_records(row[column]):
                    label = resolver[record["leaf"]]
                    quote = record["quote"]
                    if quote and quote not in evidence[label]:
                        evidence[label].append(quote)
                    else:
                        evidence.setdefault(label, [])
                return dict(evidence)

            asserted_evidence = resolved_evidence("cxr_pathology_asserted")
            uncertain_evidence = resolved_evidence("cxr_pathology_uncertain")
            denied_evidence = resolved_evidence("cxr_pathology_denied")
            asserted = set(asserted_evidence)
            # Resolve extraction conflicts deterministically. A definite
            # assertion dominates uncertainty, and either positive state
            # dominates a denial of the same canonical leaf.
            uncertain = set(uncertain_evidence) - asserted
            denied = set(denied_evidence) - asserted - uncertain
            positive = asserted | uncertain
            negative = denied
            if not positive and not negative:
                counts["empty_target"] += 1
                if not args.include_empty_targets:
                    continue
            positive_indices = sorted(label_to_index[label] for label in positive)
            negative_indices = sorted(label_to_index[label] for label in negative)
            asserted_indices = sorted(label_to_index[label] for label in asserted)
            uncertain_indices = sorted(label_to_index[label] for label in uncertain)
            denied_indices = sorted(label_to_index[label] for label in denied)
            asserted_quote_map = {
                str(label_to_index[label]): asserted_evidence[label]
                for label in sorted(asserted)
            }
            uncertain_quote_map = {
                str(label_to_index[label]): uncertain_evidence[label]
                for label in sorted(uncertain)
            }
            denied_quote_map = {
                str(label_to_index[label]): denied_evidence[label]
                for label in sorted(denied)
            }
            h5_id, split = route
            connection.execute(
                "INSERT INTO samples VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    report_uid,
                    h5_id,
                    split,
                    json.dumps(positive_indices, separators=(",", ":")),
                    json.dumps(negative_indices, separators=(",", ":")),
                    json.dumps(asserted_indices, separators=(",", ":")),
                    json.dumps(uncertain_indices, separators=(",", ":")),
                    json.dumps(denied_indices, separators=(",", ":")),
                    json.dumps(
                        asserted_quote_map,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    json.dumps(
                        uncertain_quote_map,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    json.dumps(
                        denied_quote_map,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    "pass",
                ),
            )
            counts[f"sample|{split}"] += 1
            if split == "train":
                for index in positive_indices:
                    positive_counts[index] += 1
                for index in negative_indices:
                    negative_counts[index] += 1
                for index in asserted_indices:
                    asserted_counts[index] += 1
                for index in uncertain_indices:
                    uncertain_counts[index] += 1
                for index in denied_indices:
                    denied_counts[index] += 1

    vocabulary = {
        "slug": "cxr_all_pathologies",
        "title": "Chest radiograph pathology",
        "labels": labels,
        "groups": groups,
        "types": types,
        "synonyms": synonyms,
    }
    metadata = {
        "format": "fora_cxr_label_index",
        "format_version": 4,
        "label_policy": {
            "positive": "asserted OR uncertain",
            "negative": "denied AND NOT positive",
            "unmentioned": "unknown/masked",
            "four_state_priority": (
                "asserted > uncertain > denied > unmentioned"
            ),
            "state_evidence": (
                "extractor label-to-source-report quotes retained for "
                "output-side disease-to-token alignment"
            ),
            "verdicts_included": ["pass"],
            "empty_targets_included": bool(args.include_empty_targets),
        },
        "vocabulary": vocabulary,
        "label_counts": {"positive": positive_counts, "negative": negative_counts},
        "four_state_label_counts": {
            "asserted": asserted_counts,
            "uncertain": uncertain_counts,
            "denied": denied_counts,
            "unmentioned": [
                int(counts["sample|train"])
                - asserted_counts[index]
                - uncertain_counts[index]
                - denied_counts[index]
                for index in range(len(labels))
            ],
        },
        "observed_extractor_extras": extras,
        "label_resolver_audit": dict(resolver_audit),
        "counts": dict(counts),
        "input": {
            "manifest_rows": manifest_rows,
            "label_rows": label_rows,
            "routed_reports": len(routes),
            "missing_manifest": missing_manifest,
            "verdicts": dict(verdicts),
        },
        "sources": {
            "manifest": {"path": str(args.manifest.resolve()), "sha256": sha256_file(args.manifest)},
            "labels": {"path": str(args.labels.resolve()), "sha256": sha256_file(args.labels)},
            "splits": {"path": str(args.splits.resolve()), "sha256": sha256_file(args.splits)},
            "vocabulary": {"path": str(args.vocabulary.resolve()), "sha256": sha256_file(args.vocabulary)},
            "h5_root": str(args.h5_root.resolve()),
        },
    }
    for key, value in metadata.items():
        connection.execute(
            "INSERT INTO metadata VALUES (?, ?)",
            (key, json.dumps(value, ensure_ascii=False, sort_keys=True)),
        )
    connection.commit()
    connection.execute("PRAGMA optimize")
    connection.close()
    os.replace(temporary, args.output)
    atomic_json(args.output.with_suffix(args.output.suffix + ".json"), metadata)
    print(f"Wrote {args.output} ({args.output.stat().st_size / 2**20:.1f} MiB)")
    print(f"labels={len(labels)} extras={len(extras)} train={counts['sample|train']:,} val={counts['sample|val']:,}")
    print(f"missing_manifest={missing_manifest:,} empty_target={counts['empty_target']:,}")


if __name__ == "__main__":
    main()

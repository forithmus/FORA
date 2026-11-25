#!/usr/bin/env python3
"""Add anatomy-scoped negative evidence to the X-ray label index (Methods, X-ray labels).

    python -m labels.cxr.build_scoped_supervision --base-index cxr_labels_index.sqlite --h5-root <h5 dir> \
        --output cxr_labels_scoped.sqlite --audit-output cxr_labels_scoped_audit.json


The input SQLite is copied into a new versioned artifact; it is never modified.
Original report sentences are read from the source H5 shards.  Added evidence is
stored separately from direct denials with its quote, rule, scope, and weight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from labels.cxr.scoped_normality import extract_scoped_evidence, rule_contract


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-index", required=True, type=Path)
    parser.add_argument("--h5-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    parser.add_argument("--sample-per-rule", type=int, default=20)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if hasattr(value, "item"):
        value = value.item()
        if isinstance(value, bytes):
            return value.decode("utf-8", "replace")
    return str(value)


def report_sentences(report) -> list[str]:
    for key in ("extracted_sentences", "there_is_statements"):
        if key not in report:
            continue
        try:
            values = json.loads(decode(report[key][()]))
        except Exception:
            continue
        result = [
            " ".join(str(item).split())
            for item in values
            if isinstance(item, str) and item.strip()
        ]
        if result:
            return result
    return []


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def main() -> None:
    args = arguments()
    if args.sample_per_rule < 0:
        raise ValueError("--sample-per-rule must be nonnegative")
    for path in (args.base_index, args.h5_root):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output.resolve() == args.base_index.resolve():
        raise ValueError("Output must differ from the immutable base index")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # sqlite3.Connection objects cannot carry arbitrary attributes on all
    # Python versions, so retain the path explicitly after constructing it.
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    if temporary.exists():
        temporary.unlink()
    source = sqlite3.connect(f"file:{args.base_index.resolve()}?mode=ro", uri=True)
    connection = sqlite3.connect(temporary)
    source.backup(connection)
    source.close()
    connection.execute(
        "ALTER TABLE samples ADD COLUMN scoped_denied_json TEXT NOT NULL DEFAULT '[]'"
    )
    connection.execute(
        "ALTER TABLE samples ADD COLUMN scoped_denied_weight_json TEXT NOT NULL DEFAULT '{}'"
    )
    connection.execute(
        "ALTER TABLE samples ADD COLUMN scoped_denied_evidence_json TEXT NOT NULL DEFAULT '[]'"
    )
    connection.commit()
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")

    metadata = {
        key: json.loads(value)
        for key, value in connection.execute("SELECT key,value FROM metadata")
    }
    labels = list(metadata["vocabulary"]["labels"])
    n_classes = len(labels)
    if n_classes != 311:
        raise ValueError(f"Expected the canonical 311-label CXR index, got {n_classes}")

    rows_by_h5: dict[str, dict[str, tuple[set[int], set[int], set[int], str]]] = (
        defaultdict(dict)
    )
    for uid, h5_id, split, asserted, uncertain, denied in connection.execute(
        "SELECT report_uid,h5_id,split,asserted_json,uncertain_json,denied_json "
        "FROM samples"
    ):
        rows_by_h5[str(h5_id)][str(uid)] = (
            set(map(int, json.loads(asserted))),
            set(map(int, json.loads(uncertain))),
            set(map(int, json.loads(denied))),
            str(split),
        )

    import h5py

    state_counts = {
        split: {
            "asserted": [0] * n_classes,
            "uncertain": [0] * n_classes,
            "direct_denied": [0] * n_classes,
            "scoped_denied": [0] * n_classes,
            "unmentioned": [0] * n_classes,
        }
        for split in ("train", "val")
    }
    scoped_weight_sums = {
        split: [0.0] * n_classes for split in ("train", "val")
    }
    rule_hits = Counter()
    scope_hits = Counter()
    report_counts = Counter()
    audit_samples: dict[str, list[dict[str, object]]] = defaultdict(list)

    for shard_number, h5_id in enumerate(sorted(rows_by_h5), 1):
        expected = rows_by_h5[h5_id]
        found: set[str] = set()
        h5_path = args.h5_root / f"{h5_id}.h5"
        if not h5_path.is_file():
            report_counts["missing_h5_shard"] += len(expected)
            continue
        with h5py.File(h5_path, "r") as handle:
            for patient in handle:
                patient_group = handle[patient]
                if not hasattr(patient_group, "keys"):
                    continue
                for uid in patient_group:
                    if uid not in expected:
                        continue
                    found.add(uid)
                    asserted, uncertain, direct_denied, split = expected[uid]
                    report_group = patient_group[uid]
                    sentences = (
                        report_sentences(report_group["report"])
                        if hasattr(report_group, "keys") and "report" in report_group
                        else []
                    )
                    if not sentences:
                        report_counts["missing_report_sentences"] += 1
                    candidates = extract_scoped_evidence(sentences, labels)
                    blocked = asserted | uncertain | direct_denied
                    candidates = [
                        item for item in candidates if item.label_index not in blocked
                    ]
                    grouped: dict[
                        tuple[str, str, str, float, str], set[int]
                    ] = defaultdict(set)
                    weights: dict[int, float] = {}
                    for item in candidates:
                        evidence_key = (
                            item.scope,
                            item.quote,
                            item.rule_id,
                            item.weight,
                            item.qualifier,
                        )
                        grouped[evidence_key].add(item.label_index)
                        weights[item.label_index] = max(
                            weights.get(item.label_index, 0.0), item.weight
                        )
                        rule_hits[item.rule_id] += 1
                        scope_hits[item.scope] += 1
                        if len(audit_samples[item.rule_id]) < args.sample_per_rule:
                            audit_samples[item.rule_id].append(
                                {
                                    "report_uid": uid,
                                    "h5_id": h5_id,
                                    **item.to_dict(),
                                }
                            )
                    scoped = set(weights)
                    compact_evidence = [
                        {
                            "scope": scope,
                            "quote": quote,
                            "rule_id": rule_id,
                            "weight": weight,
                            "qualifier": qualifier,
                            "label_indices": sorted(label_indices),
                        }
                        for (
                            scope,
                            quote,
                            rule_id,
                            weight,
                            qualifier,
                        ), label_indices in sorted(grouped.items())
                    ]
                    connection.execute(
                        "UPDATE samples SET scoped_denied_json=?,"
                        "scoped_denied_weight_json=?,scoped_denied_evidence_json=? "
                        "WHERE report_uid=?",
                        (
                            json.dumps(sorted(scoped), separators=(",", ":")),
                            json.dumps(
                                {str(k): weights[k] for k in sorted(weights)},
                                separators=(",", ":"),
                            ),
                            json.dumps(
                                compact_evidence,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                            uid,
                        ),
                    )
                    report_counts[f"reports|{split}"] += 1
                    report_counts[f"reports_with_scoped|{split}"] += int(bool(scoped))
                    for index in range(n_classes):
                        if index in asserted:
                            state = "asserted"
                        elif index in uncertain:
                            state = "uncertain"
                        elif index in direct_denied:
                            state = "direct_denied"
                        elif index in scoped:
                            state = "scoped_denied"
                        else:
                            state = "unmentioned"
                        state_counts[split][state][index] += 1
                        if state == "scoped_denied":
                            scoped_weight_sums[split][index] += weights[index]
        missing = set(expected).difference(found)
        report_counts["reports_not_found_in_h5"] += len(missing)
        # Missing H5 reports retain empty scoped evidence but must still enter
        # the state counts so the artifact totals match the SQLite index.
        for uid in missing:
            asserted, uncertain, direct_denied, split = expected[uid]
            for index in range(n_classes):
                state = (
                    "asserted"
                    if index in asserted
                    else "uncertain"
                    if index in uncertain
                    else "direct_denied"
                    if index in direct_denied
                    else "unmentioned"
                )
                state_counts[split][state][index] += 1
        if shard_number % 10 == 0:
            connection.commit()
            print(
                f"[scoped] shards={shard_number}/{len(rows_by_h5)} "
                f"reports={sum(v for k,v in report_counts.items() if k.startswith('reports|')):,}",
                flush=True,
            )

    contract = {
        "format": "cxr_scoped_normality_supervision",
        "format_version": 2,
        "base_index": {
            "path": str(args.base_index.resolve()),
            "sha256": sha256(args.base_index),
        },
        "h5_root": str(args.h5_root.resolve()),
        "extractor": rule_contract(),
        "report_counts": dict(report_counts),
        "rule_hits": dict(sorted(rule_hits.items())),
        "scope_hits": dict(sorted(scope_hits.items())),
        "state_counts": state_counts,
        "scoped_weight_sums": scoped_weight_sums,
    }
    for key, value in (
        ("scoped_normality", contract),
        ("five_state_label_counts", state_counts["train"]),
        ("scoped_denied_weight_sums", scoped_weight_sums["train"]),
    ):
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)",
            (key, json.dumps(value, ensure_ascii=False, sort_keys=True)),
        )
    connection.commit()
    connection.execute("PRAGMA optimize")
    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    if integrity != "ok":
        raise RuntimeError(f"SQLite integrity check failed: {integrity}")
    connection.close()
    os.replace(temporary, args.output)

    audit = {
        "format": "cxr_scoped_normality_audit",
        "format_version": 2,
        "index": str(args.output.resolve()),
        "index_sha256": sha256(args.output),
        "report_counts": dict(report_counts),
        "rule_hits": dict(sorted(rule_hits.items())),
        "scope_hits": dict(sorted(scope_hits.items())),
        "samples": dict(audit_samples),
    }
    atomic_json(args.audit_output, audit)
    print(
        f"wrote {args.output} ({args.output.stat().st_size / 2**20:.1f} MiB) "
        f"and {args.audit_output}",
        flush=True,
    )


if __name__ == "__main__":
    main()

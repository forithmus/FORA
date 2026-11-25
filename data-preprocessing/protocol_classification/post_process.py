#!/usr/bin/env python3
"""Post-process the protocol classifications.

Loads the per-series classified-protocols CSV produced by the classification
pipeline (``classified_protocols.csv``), applies a set of deterministic
clean-up rules to the ``classified_protocols`` list of each row, and writes a
slim CSV with only the identifier columns plus the cleaned class list.

Rules are intentionally simple and live in ``MERGE_RULES`` below so new ones can
be added without touching the logic. Each rule renames one protocol class to
another; when a row already contains the target class, the rename collapses to a
single occurrence (the list is de-duplicated, order preserved).

Currently:
  - ``CT Liver - Four-Phase``  ->  ``CT Abdomen and Pelvis - Multiphase``

Usage::

    python post_process.py --input classified_protocols.csv --output classified_protocols_post_processed.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from config import PROTOCOL_CLASSES_JSON, PROTOCOL_COL

# ─────────────────────────────────────────────────────────────────────────────
# Rules — add new class-merge rules here (source class -> target class).
# Both endpoints are validated against config_protocol_classes.json at startup.
# ─────────────────────────────────────────────────────────────────────────────
MERGE_RULES: dict[str, str] = {
    "CT Liver - Four-Phase": "CT Abdomen and Pelvis - Multiphase",
}

# Output: identifiers + the cleaned class list, nothing else.
OUTPUT_COLS: tuple[str, ...] = (
    "AccessionNumber",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    PROTOCOL_COL,
)

def load_valid_titles(classes_json: Path) -> set[str]:
    """Return the set of canonical protocol-class titles from the class config."""
    data = json.loads(Path(classes_json).read_text(encoding="utf-8"))
    return {entry["title"] for entry in data}


def validate_rules(rules: dict[str, str], valid_titles: set[str]) -> None:
    """Fail fast if any rule endpoint is not a known protocol-class title."""
    bad = {
        cls
        for pair in rules.items()
        for cls in pair
        if cls not in valid_titles
    }
    if bad:
        raise ValueError(
            "MERGE_RULES reference unknown protocol class(es) "
            f"(not in config_protocol_classes.json): {sorted(bad)}"
        )


def apply_rules(classes: list[str], rules: dict[str, str]) -> list[str]:
    """Map each class through ``rules`` and de-duplicate, preserving order."""
    out: list[str] = []
    for c in classes:
        mapped = rules.get(c, c)
        if mapped not in out:
            out.append(mapped)
    return out


def post_process(
    input_path: Path, output_path: Path, classes_json: Path, rules: dict[str, str]
) -> None:
    valid_titles = load_valid_titles(classes_json)
    validate_rules(rules, valid_titles)

    print(f"Loading {input_path} …")
    df = pd.read_csv(
        input_path,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
        usecols=list(OUTPUT_COLS),
        low_memory=False,
    )
    n_rows = len(df)
    print(f"  {n_rows:,} rows.")

    # Fast path: only rows whose raw cell mentions a rule source need parsing;
    # all others are passed through verbatim (original formatting preserved).
    sources = tuple(rules)
    raw = df[PROTOCOL_COL]
    touched_mask = raw.apply(lambda s: any(src in s for src in sources))

    n_changed = 0
    per_rule = {src: 0 for src in rules}
    n_collapsed = 0

    def transform(cell: str) -> str:
        nonlocal n_changed, n_collapsed
        classes = json.loads(cell)
        present = [c for c in classes if c in rules]
        if not present:
            return cell
        for c in present:
            per_rule[c] += 1
        new_classes = apply_rules(classes, rules)
        n_changed += 1
        if len(new_classes) < len(classes):
            n_collapsed += 1
        return json.dumps(new_classes, ensure_ascii=False)

    df.loc[touched_mask, PROTOCOL_COL] = raw[touched_mask].map(transform)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df[list(OUTPUT_COLS)].to_csv(output_path, index=False)

    # ── Summary ──
    print("Rules applied:")
    for src, dst in rules.items():
        print(f"  {per_rule[src]:>9,} rows  {src}  ->  {dst}")
    print(f"Rows with a class list change : {n_changed:,} "
          f"({100 * n_changed / max(n_rows, 1):.3f}%)")
    print(f"  of which de-duplicated (source+target both present): {n_collapsed:,}")
    print(f"Wrote {output_path}  ({n_rows:,} rows, {len(OUTPUT_COLS)} columns)")


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Apply clean-up rules to classified_protocols and write a slim CSV "
            "(AccessionNumber, StudyInstanceUID, SeriesInstanceUID, "
            "classified_protocols)."
        )
    )
    p.add_argument("--input", required=True, help="classified-protocols CSV from merge_shards.py")
    p.add_argument("--output", required=True, help="output CSV")
    p.add_argument("--classes-json", default=PROTOCOL_CLASSES_JSON,
                   help="Protocol-class definitions JSON (for rule validation).")
    args = p.parse_args()

    post_process(
        input_path=Path(args.input),
        output_path=Path(args.output),
        classes_json=Path(args.classes_json),
        rules=MERGE_RULES,
    )


if __name__ == "__main__":
    sys.exit(main())

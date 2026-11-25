"""Per-protocol vocabularies of the CT report-scoring labeler and their label-name resolution (Methods, report evaluation).

Each protocol class has a closed leaf vocabulary (``data/ct_protocols_scoring.json``; class 14 is folded into class 3).
An emitted leaf resolves, within its protocol, to the unique leaf whose normalised name or synonym equals the normalised
emitted name. Emitted names that resolve to no leaf in the reference-report extraction corpus were kept as explicit
"extractor extra" labels (``data/ct_scoring_extras.json``, written by ``build_extras.py``); together with the leaves they
form the 1,106-label scoring vocabulary. ``other``/``Other: ...`` names are invalid.
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from labels.llm import stage_path

DEFAULT_PROTOCOLS = "report_scoring/data/ct_protocols_scoring.json"
DEFAULT_EXTRAS = "report_scoring/data/ct_scoring_extras.json"


def normalize_label(value: str) -> str:
    value = value.casefold().replace("–", "-").replace("—", "-")
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def load_vocabularies(path: str | Path = DEFAULT_PROTOCOLS) -> dict[int, dict]:
    prot = json.loads(stage_path(path).read_text("utf-8"))["protocols"]
    out = {}
    for n, m in sorted(prot.items(), key=lambda kv: int(kv[0])):
        if int(n) == 14:
            continue
        labels = [str(lf["name"]) for lf in m["leaves"]]
        if len(labels) != len(set(labels)):
            raise ValueError(f"duplicate leaves in protocol class {n}")
        out[int(n)] = {"labels": labels, "synonyms": {str(lf["name"]): str(lf.get("syn", "")) for lf in m["leaves"]}}
    return out


def alias_tables(vocabs: dict[int, dict]) -> dict[int, dict[str, set[str]]]:
    out = {}
    for pc, voc in vocabs.items():
        al: dict[str, set[str]] = defaultdict(set)
        for lab in voc["labels"]:
            al[normalize_label(lab)].add(lab)
            for syn in voc["synonyms"].get(lab, "").split(";"):
                if syn.strip():
                    al[normalize_label(syn.strip())].add(lab)
        out[pc] = al
    return out


def observed_extras(labels_csv: Path, vocabs: dict[int, dict]) -> tuple[dict[int, list[str]], Counter]:
    """Extractor extras of a merged CT extraction CSV (column pathology_labels = JSON list of per-protocol results)."""
    csv.field_size_limit(10 ** 9)
    aliases = alias_tables(vocabs)
    observed: dict[int, Counter] = defaultdict(Counter)
    with open(labels_csv, newline="", encoding="utf-8-sig") as h:
        for row in csv.DictReader(h):
            if row["pathology_extraction_verdict"].strip() not in {"pass", "partial"}:
                continue
            for m in json.loads(row["pathology_labels"]):
                pc = 3 if int(m["protocol_class"]) == 14 else int(m["protocol_class"])
                for pol in ("asserted", "uncertain", "denied"):
                    for item in m.get(pol, []):
                        observed[pc][str(item["leaf"])] += 1
    extras, audit = {}, Counter()
    for pc, counts in observed.items():
        unresolved: dict[str, Counter] = defaultdict(Counter)
        for raw, count in counts.items():
            norm = normalize_label(raw)
            if len(aliases[pc].get(norm, set())) == 1:
                audit["canonical_or_synonym_instances"] += count
            elif norm == "other" or norm.startswith("other "):
                audit["invalid_other_instances"] += count
            else:
                unresolved[norm][raw] += count
        chosen = []
        for variants in unresolved.values():
            # spelling variants of one normalised name merge under the most frequent raw spelling
            chosen.append(min(variants, key=lambda raw: (-variants[raw], raw)))
            audit["extractor_extra_instances"] += sum(variants.values())
        extras[pc] = sorted(chosen, key=lambda v: (normalize_label(v), v))
    return extras, audit


def load_extras(path: str | Path = DEFAULT_EXTRAS) -> dict[int, list[str]]:
    d = json.loads(stage_path(path).read_text("utf-8"))
    return {int(k): v for k, v in d["extras"].items()}


def scoring_vocabulary(vocabs: dict[int, dict], extras: dict[int, list[str]]) -> list[str]:
    """Sorted 1,106-label scoring vocabulary (leaves of all protocols + their extractor extras)."""
    return sorted({lab for pc, v in vocabs.items() for lab in v["labels"] + extras.get(pc, [])})

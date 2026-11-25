"""Map X-ray extractor output of generated / reference reports to the 129 X-ray classifier classes (Methods, report
evaluation).

X-ray reports are labelled with the training extractor itself (``labels/cxr/extract_findings.py``, chunk size 2000) on
findings text. A row with verdict pass contributes its asserted and uncertain leaves; a leaf resolves to the scoped label
index vocabulary (311 labels) by normalised canonical name or unique synonym; non-pass or missing rows have no positive
finding. Positives restricted to the classifier classes give the per-report positive set.

    from labels.report_scoring.cxr_states import CXRResolver
    pos = CXRResolver(label_db, classes).positives(row)   # row = one CSV row of the merged extraction output
"""
from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter, defaultdict


def normalize(v: str) -> str:
    v = v.casefold().replace("–", "-").replace("—", "-")
    return re.sub(r"[^a-z0-9]+", " ", v).strip()


class CXRResolver:
    def __init__(self, label_db: str, classes: list[str]):
        voc = json.loads(sqlite3.connect(f"file:{label_db}?mode=ro", uri=True).execute(
            "select value from metadata where key='vocabulary'").fetchone()[0])
        self.alias: dict[str, set[str]] = defaultdict(set)
        for lab in voc["labels"]:
            self.alias[normalize(lab)].add(lab)
            for s in (voc["synonyms"].get(lab) or "").split(";"):
                if s.strip():
                    self.alias[normalize(s)].add(lab)
        self.classes = list(classes)
        self.cidx = {c: j for j, c in enumerate(self.classes)}
        self.unresolved = Counter()

    def labels(self, row: dict) -> set[str]:
        if row["cxr_pathology_extraction_verdict"].strip() != "pass":
            return set()
        pos = set()
        for col in ("cxr_pathology_asserted", "cxr_pathology_uncertain"):
            for it in json.loads(row[col] or "[]"):
                if isinstance(it, dict) and it.get("leaf"):
                    m = self.alias.get(normalize(str(it["leaf"])), set())
                    if len(m) == 1:
                        pos.add(next(iter(m)))
                    else:
                        self.unresolved[str(it["leaf"])] += 1
        return pos

    def positives(self, row: dict) -> set[int]:
        return {self.cidx[x] for x in self.labels(row) if x in self.cidx}

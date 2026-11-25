"""Build the CT scoring-labeler task file from reference and generated reports (Methods, report evaluation).

Inputs:
* ``--studies`` JSONL: {cohort, uid, protocol_classes: [int], reference: findings text};
* ``--generations`` JSONL (any number): {model, cohort, uid, text}; the findings part of a generated report is the text
  before an "Impression:" header, without a leading "Findings:" header.
Protocol classes per cohort: internal FORA studies use every protocol class of the report; Merlin uses its labelled class
(2 or 4); CT-RATE uses ``ctrate_protocol`` on the Technique field (6 = CT chest with IV contrast, else 7 = non-contrast).
One task per (source, cohort, uid, protocol class): {task_id, source ('ref' | 'gen:<model>'), uid, cohort,
protocol_class, findings, impression: ''}.

    python -m labels.report_scoring.make_tasks --studies studies.jsonl --generations gen_a.jsonl gen_b.jsonl --output tasks.jsonl
"""
from __future__ import annotations

import argparse
import json
import re

_IMP = re.compile(r"\n?\s*\**\s*impressions?\s*\**\s*:", re.I)
_FIND = re.compile(r"^\s*\**\s*findings?\s*\**\s*:\s*", re.I)


def findings_part(text: str) -> str:
    t = _IMP.split(text or "", maxsplit=1)[0]
    return _FIND.sub("", t).strip()


def ctrate_protocol(technique: str) -> int:
    t = (technique or "").lower()
    contrast = "contrast" in t and not re.search(
        r"non[- ]?contrast|without (iv |intravenous )?contrast|no contrast|not (given|administered)", t)
    return 6 if contrast else 7


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--studies", required=True)
    ap.add_argument("--generations", nargs="*", default=[])
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    studies = {(d["cohort"], d["uid"]): d for d in map(json.loads, open(a.studies, encoding="utf-8"))}
    gens = {}
    for f in a.generations:
        for d in map(json.loads, open(f, encoding="utf-8")):
            if (d["cohort"], d["uid"]) in studies:
                gens[(d["model"], d["cohort"], d["uid"])] = d
    n = 0
    with open(a.output, "w", encoding="utf-8") as h:
        def emit(source, c, u, text):
            nonlocal n
            for pc in studies[(c, u)]["protocol_classes"]:
                h.write(json.dumps({"task_id": f"{source}|{c}|{u}|{pc}", "source": source, "uid": u, "cohort": c,
                                    "protocol_class": pc, "findings": text, "impression": ""}, ensure_ascii=False) + "\n")
                n += 1
        for (c, u), d in studies.items():
            emit("ref", c, u, d.get("reference") or "")
        for (m, c, u), d in sorted(gens.items()):
            emit(f"gen:{m}", c, u, findings_part(d["text"]))
    print(n, "tasks")


if __name__ == "__main__":
    main()

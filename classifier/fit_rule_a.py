"""Rule-A operating points of the CT classifier: the cut-offs behind the reported balanced accuracy and CRG
(Methods, "Evaluation", finding recognition).

Rule A, per finding and scan type, on the raw study-bag logits of TRAINING studies only: the cut-off maximising
sensitivity + specificity (Youden) among training studies of that scan type, or among all training studies when the scan
type has fewer than 50 positive or 50 negative training studies. Masked labels are excluded. A study is called positive if
logit >= cut-off (``fora.mil.calibration.RuleA``). Scan types (``fora.mil.calibration.scan_type``): chest, abdomen,
head/neck, chest-abdomen-pelvis (CAP), other, from the protocol titles of the study.

External benchmarks map a benchmark label to one or more classifier findings (``members``; study score = max logit over
the members) and to the scan types of the benchmark (e.g. CT-RATE: chest); the cut-off is fitted on the training studies
of those scan types with the same 50-positive / 50-negative fallback.

  fit_rule_a.py --dense dense_predictions.npz --labels labels.npz --split split.json --protocols protocols.json \
                [--external external_mapping.json] --out ruleA_cutoffs.json

protocols.json: {report_uid: [protocol title, ...]}. external_mapping.json: {dataset: {label: {"members": [...],
"scan_types": [...]}}}. Output: {"fora_per_scan_type": {finding: {scan type: cut}}, "external": {dataset: {label:
{"members", "scan_types", "ruleA_cutoff"}}}}.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import write_json
from fora.mil.calibration import SCAN_TYPES, scan_type

MIN_POS = MIN_NEG = 50


def youden_midpoint(z, y) -> float:
    """Youden cut-off on scores z (y in {0, 1}), placed midway between the maximising score and the next lower one."""
    o = np.argsort(-z)
    sv, yy = z[o], y[o]
    b = np.cumsum(yy == 1) / (yy == 1).sum() - np.cumsum(yy == 0) / (yy == 0).sum()
    i = int(np.argmax(b))
    return float((sv[i] + sv[min(i + 1, len(sv) - 1)]) / 2)


def cut_a(z, y, types, wanted) -> float:
    """Rule-A cut-off for the scan types ``wanted`` (y = -1 masked)."""
    m = y >= 0
    t = youden_midpoint(z[m], y[m])
    k = m & np.isin(types, wanted)
    if (y[k] == 1).sum() >= MIN_POS and (y[k] == 0).sum() >= MIN_NEG:
        return youden_midpoint(z[k], y[k])
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--protocols", type=Path, required=True)
    ap.add_argument("--external", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    d = np.load(a.dense, allow_pickle=False)
    labs = [str(c) for c in d["label"]]
    split = json.loads(a.split.read_text())
    trn = np.array([split.get(x) == "train" for x in d["fora_id"].astype(str)])
    fid, uid, LG = d["fora_id"].astype(str)[trn], d["report_uid"].astype(str)[trn], d["logit"][trn]
    P = json.loads(a.protocols.read_text())
    types = np.array([scan_type(P.get(u, [])) for u in uid])
    Z = np.load(a.labels, allow_pickle=False)
    nodes = list(Z["nodes"])
    row = {c: i for i, c in enumerate(Z["cases"].tolist())}
    st = Z["state"][[row[f] for f in fid]]

    def score_target(members):
        z = LG[:, [labs.index(m) for m in members]].astype(float).max(1)
        s = st[:, [nodes.index(m) for m in members]]
        y = np.where((s < 0).all(1), -1, np.isin(s, (1, 2)).any(1).astype(int))
        return z, y
    out = {"fora_per_scan_type": {}, "external": {}}
    for c in labs:
        z, y = score_target([c])
        out["fora_per_scan_type"][c] = {g: cut_a(z, y, types, [g]) for g in SCAN_TYPES}
    if a.external:
        for ds, table in json.loads(a.external.read_text()).items():
            out["external"][ds] = {}
            for name, e in table.items():
                z, y = score_target(e["members"])
                out["external"][ds][name] = dict(members=e["members"], scan_types=e["scan_types"],
                                                 ruleA_cutoff=cut_a(z, y, types, e["scan_types"]))
    write_json(a.out, out)


if __name__ == "__main__":
    main()

"""Label DB build on a synthetic two-report corpus, and the recall merge."""
import csv
import json
from argparse import Namespace

import numpy as np

from labels.ct.build_label_db import build
from labels.ct.merge_recall import read

H = {"nodes": {
    "Lung disease": {"kind": "umbrella", "ancestors": [], "aliases": [], "parents": []},
    "Pneumonia": {"kind": "leaf", "ancestors": ["Lung disease"], "aliases": [], "parents": ["Lung disease"]},
    "Lung oedema": {"kind": "leaf", "ancestors": ["Lung disease"], "aliases": [], "parents": ["Lung disease"]},
    "Cardiomegaly": {"kind": "leaf", "ancestors": [], "aliases": [], "parents": []},
    "Breast cancer": {"kind": "leaf", "ancestors": [], "aliases": [], "parents": []}}, "redirect": {}}
COLS = ["ReportUID", "findings", "protocol_class", "pathology_asserted", "pathology_denied", "pathology_uncertain",
        "pathology_other", "pathology_extraction_verdict"]


def shard(d, rows):
    d.mkdir()
    with open(d / "s.csv", "w", newline="") as f:
        w = csv.DictWriter(f, COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()})


def row(u, a=(), d=(), un=(), other=(), verdict="pass"):
    L = lambda xs: [{"leaf": x, "quote": "q"} for x in xs]  # noqa: E731
    return dict(ReportUID=u, findings="f", protocol_class=7, pathology_asserted=L(a), pathology_denied=L(d),
                pathology_uncertain=L(un), pathology_other=[{"finding": x, "quote": "q"} for x in other],
                pathology_extraction_verdict=verdict)


def test_build(tmp_path):
    (tmp_path / "h.json").write_text(json.dumps(H))
    (tmp_path / "al.json").write_text(json.dumps({"Big heart": "Cardiomegaly"}))
    (tmp_path / "cases.jsonl").write_text('{"id": "train_0", "report": "r1"}\n{"id": "val_0", "report": "r2"}\n')
    shard(tmp_path / "main", [row("r1", a=["Pneumonia", "Breast cancer"], d=["Lung oedema"]),
                              row("r2", a=["Cardiomegaly denied"], d=["Lung disease"], other=["Big heart"], verdict="fail")])
    shard(tmp_path / "retry", [row("r2", d=["Lung disease"], a=["Cardiomegaly denied"])])
    (tmp_path / "rec").write_text('REC\t{"ReportUID": "r2", "protocol_class": 7, "add": [{"leaf": "Pneumonia", "polarity": "uncertain"}]}\n')
    (tmp_path / "can").write_text('CAN\t{"ReportUID": "r1", "label": "Breast cancer", "verdict": "history_only"}\n')
    a = Namespace(shards=str(tmp_path / "main"), retry_shards=[str(tmp_path / "retry")], union_shards=None,
                  hierarchy=str(tmp_path / "h.json"), aliases=str(tmp_path / "al.json"), cases=str(tmp_path / "cases.jsonl"),
                  recall=str(tmp_path / "rec"), malignancy=str(tmp_path / "can"), adjudication=None,
                  dump_items=str(tmp_path / "items"), output=str(tmp_path / "db"))
    s = build(a)
    z = np.load(tmp_path / "db/labels.npz")
    st = {(r, n): z["state"][i, j] for i, r in enumerate(z["reports"]) for j, n in enumerate(z["nodes"])}
    assert st[("r1", "Pneumonia")] == 1 and st[("r1", "Lung disease")] == 1 and st[("r1", "Lung oedema")] == 0
    assert st[("r1", "Breast cancer")] == 3 and st[("r2", "Cardiomegaly")] == 0
    assert st[("r2", "Pneumonia")] == 2 and st[("r2", "Lung disease")] == 2 and st[("r2", "Lung oedema")] == 3
    assert s["pos_denied_conflicts"] == 1 and s["flip_cardiomegaly_denied"] == 1
    items = (tmp_path / "items").read_text().splitlines()
    assert any(x.startswith("JUDGE\t") and "Lung disease" in x for x in items)
    assert any(x.startswith("MAP\t") and "Cardiomegaly denied" in x for x in items)


def test_merge_recall(tmp_path):
    p = tmp_path / "a.recs"
    p.write_text('REC\t{"ReportUID": "r", "status": "ok", "add": [{"leaf": "A"}]}\nREC\t{"ReportUID": "r", "status": "ok", "add": [{"leaf": "B"}]}\n')
    assert [x["leaf"] for x in read([str(p)])["r"]["add"]] == ["A", "B"]

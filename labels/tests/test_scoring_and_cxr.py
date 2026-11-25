"""Report-scoring resolvers, task building, X-ray target selection and scoped normality on synthetic data."""
import json
import sqlite3

from labels.cxr.scoped_normality import extract_scoped_evidence
from labels.cxr.select_targets import select
from labels.report_scoring.ct_states import Resolver
from labels.report_scoring.cxr_states import CXRResolver
from labels.report_scoring.make_tasks import ctrate_protocol, findings_part


def test_ct_resolver():
    r = Resolver()
    assert len(r.labels) == 1106 and len(r.classes) == 1028
    st, raw, unm = r.report_states([{"protocol_class": 7, "asserted": [{"leaf": "Pleural effusion"}],
                                     "uncertain": [], "denied": [{"leaf": "Pleural effusion"}, {"leaf": "Cardiomegaly"}]}])
    assert st["asserted"] == ["Pleural effusion"] and "Pleural effusion" not in st["denied"] and not unm


def test_tasks():
    assert findings_part("Findings: lungs clear.\nImpression: normal") == "lungs clear."
    assert ctrate_protocol("Non-contrast CT") == 7 and ctrate_protocol("with IV contrast") == 6


def make_db(path, labels, rows):
    c = sqlite3.connect(path)
    c.execute("create table metadata (key text, value text)")
    c.execute("create table samples (report_uid text, h5_id text, split text, positive_json text, extraction_verdict text)")
    c.execute("insert into metadata values ('vocabulary', ?)", (json.dumps({"labels": labels, "synonyms": {"A": "alpha"}}),))
    c.executemany("insert into samples values (?,?,?,?,?)", rows)
    c.commit()


def test_cxr_targets_and_states(tmp_path):
    db = str(tmp_path / "x.sqlite")
    labels = ["A", "B", "No Finding / Normal study"]
    rows = [(f"r{i}", "h", "train", json.dumps([0, 2] if i % 2 else [1]), "pass") for i in range(6)]
    make_db(db, labels, rows + [("v0", "h", "val", "[0]", "pass")])
    cases = {"train": [dict(id=f"h/p/r{i}", h5="h", report=f"r{i}") for i in range(6)], "val": [dict(id="h/p/v0", h5="h", report="v0")]}
    r = select(db, cases, 3, ["No Finding / Normal study"])
    assert [c["name"] for c in r["classes"]] == ["A", "B"] and r["Yt"].sum() == 6 and r["Yv"].tolist() == [[1, 0]]
    res = CXRResolver(db, ["A", "B"])
    row = {"cxr_pathology_extraction_verdict": "pass", "cxr_pathology_asserted": json.dumps([{"leaf": "Alpha"}]),
           "cxr_pathology_uncertain": json.dumps([{"leaf": "b"}])}
    assert res.positives(row) == {0, 1}


def test_scoped_normality():
    from labels.cxr.scoped_normality import SCOPE_LABEL_NAMES
    labels = sorted({n for v in SCOPE_LABEL_NAMES.values() for n in v})
    ev = extract_scoped_evidence(["Bony structures are intact."], labels)
    assert ev and all(e.scope == "visible_skeleton" for e in ev)

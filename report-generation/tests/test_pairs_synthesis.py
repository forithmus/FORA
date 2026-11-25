"""CPU smoke tests for the CT DPO pair synthesis (fora_report.pairs) on a synthetic report index."""
import gzip
import json
import sqlite3

import numpy as np
import pytest

from fora_report.pairs import attribute, base, gemma_verifier, merge, plan_pairs
from fora_report.pairs.text import attribute_spans, sha, split_sentences

LABELS = ["Pleural effusion", "Lung nodule", "Hepatic steatosis", "Splenomegaly"]
EFF, NOD = 0, 1
EFF_S = "Pleural effusion is seen in the right hemithorax."
NOD_S = "A nodule measuring 6 mm is seen in the right lower lobe of the lung."
DENY_S = "There is no pleural effusion."
REPORT_A = (f"The liver is normal in size. {EFF_S} The kidneys are normal in appearance. {NOD_S} "
            "The spleen is normal.")
REPORT_B = "The liver is normal in size. The pleural cavities are clear. The kidneys are normal in appearance. The lungs are clear."


def _studies():
    out = []
    for i in range(10):
        out.append(dict(uid=f"a{i:02d}", patient=f"pa{i:02d}", text=REPORT_A, asserted=[EFF, NOD],
                        ev={"asserted": {str(EFF): [EFF_S.rstrip(".")], str(NOD): [NOD_S.rstrip(".")]},
                            "uncertain": {}, "denied": {}}, denied=[]))
    for i in range(6):
        deny = i < 3
        text = REPORT_B + (" " + DENY_S if deny else "")
        out.append(dict(uid=f"b{i:02d}", patient=f"pb{i:02d}", text=text, asserted=[],
                        ev={"asserted": {}, "uncertain": {}, "denied": {str(EFF): [DENY_S]} if deny else {}},
                        denied=[EFF] if deny else []))
    return out


@pytest.fixture()
def toy(tmp_path):
    studies = _studies()
    idx = sqlite3.connect(tmp_path / "index.sqlite")
    idx.execute("CREATE TABLE samples(report_uid TEXT, patient_uid TEXT, h5_id TEXT, split TEXT, protocol_class INT,"
                " study_type_ids_json TEXT, asserted_json TEXT, uncertain_json TEXT, denied_json TEXT,"
                " asserted_evidence_json TEXT, uncertain_evidence_json TEXT, denied_evidence_json TEXT,"
                " extraction_verdict TEXT)")
    idx.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT)")
    idx.execute("INSERT INTO metadata VALUES('global', ?)", (json.dumps({"labels": LABELS}),))
    fnd = sqlite3.connect(tmp_path / "findings.sqlite")
    fnd.execute("CREATE TABLE findings(report_uid TEXT PRIMARY KEY, patient_uid TEXT, h5_id TEXT, split TEXT,"
                " findings TEXT NOT NULL, source TEXT NOT NULL, sha256 TEXT NOT NULL)")
    for s in studies:
        idx.execute("INSERT INTO samples VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (s["uid"], s["patient"], "h0", "train", 0, "[0]", json.dumps(s["asserted"]), "[]",
                     json.dumps(s["denied"]), json.dumps(s["ev"]["asserted"]), "{}", json.dumps(s["ev"]["denied"]),
                     "pass"))
        fnd.execute("INSERT INTO findings VALUES(?,?,?,?,?,?,?)",
                    (s["uid"], s["patient"], "h0", "train", s["text"], "structured_findings", sha(s["text"])))
    idx.commit(); fnd.commit()
    logits = np.full((len(studies), len(LABELS)), -6.0, np.float32)
    for j, s in enumerate(studies):
        if s["uid"].startswith("b"):
            logits[j, EFF] = 3.0
    np.savez(tmp_path / "dense.npz", report_uid=np.array([s["uid"] for s in studies]),
             fora_id=np.array([s["patient"] for s in studies]), label=np.array(LABELS), logit=logits)
    n = len(LABELS)
    (tmp_path / "cal.json").write_text(json.dumps(dict(labels=LABELS, class_slope=[1.0] * n, class_intercept=[0.0] * n,
                                                       class_plan_threshold=[0.5] * n,
                                                       class_possible_threshold=[0.2] * n)))
    (tmp_path / "split.json").write_text(json.dumps({s["patient"]: "train" for s in studies}))
    (tmp_path / "heads.json").write_text(json.dumps({"attached": []}))
    (tmp_path / "hier.json").write_text(json.dumps({"nodes": {}}))
    return tmp_path


def _read(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f]


def test_split_and_attribute_spans():
    assert [t for t, _, _ in split_sentences("A b c. D e f.")] == ["A b c.", "D e f."]
    found = attribute_spans(NOD_S, "asserted")
    assert found["size"] == ["6 mm"] and "laterality" in found and found["certainty"] == ["<definite statement>"]


@pytest.mark.parametrize("kind,original,edited,ok", [
    ("laterality", "Pleural effusion is seen on the right.", "Pleural effusion is seen on the left.", True),
    ("size", "A nodule measuring 6 mm is seen.", "A nodule measuring 25 mm is seen.", True),
    ("size", "A nodule measuring 6 mm is seen.", "A nodule measuring 7 mm is seen.", False),
    ("severity", "There is mild atelectasis.", "There is severe atelectasis.", True),
    ("number", "A single nodule is seen.", "Multiple nodules are seen.", True),
    ("certainty", "Findings are suspicious for pneumonia.", "Findings are diagnostic of pneumonia.", True),
    ("location", "A nodule is seen in the upper lobe.", "A nodule is seen in the lower lobe.", True),
    ("location", "A nodule is seen in the upper lobe.", "A cyst is seen in the upper lobe.", False),
    ("laterality", "A nodule is seen in the right middle lobe.", "A nodule is seen in the left middle lobe.", False),
])
def test_verify_attribute_edit(kind, original, edited, ok):
    assert attribute.verify_attribute_edit(original, edited, kind)[0] is ok


def test_base_build_and_attribute_pipeline(toy):
    out = toy / "out"
    base.main(["build", "--index", str(toy / "index.sqlite"), "--findings", str(toy / "findings.sqlite"),
               "--output_dir", str(out)])
    pairs = _read(out / "base_pairs.jsonl.gz")
    kinds = {p["pair_type"] for p in pairs}
    assert {"omission", "repetition", "explicit_denial"} <= kinds
    for p in pairs:
        assert p["chosen_sha256"] == sha(p["chosen_text"]) and p["rejected_text"] != p["chosen_text"]
    om = next(p for p in pairs if p["pair_type"] == "omission")
    assert om["changed_sentence"] not in om["rejected_text"]
    den = next(p for p in pairs if p["pair_type"] == "explicit_denial")
    assert DENY_S not in den["rejected_text"] and EFF_S in den["rejected_text"]
    cands = _read(out / "attribute_candidates.jsonl.gz")
    size = next(c for c in cands if c["attribute_type"] == "size")
    # Gemma stand-in: one corruption per candidate type, the judge accepts everything except laterality
    gdir, jdir = out / "gemma_attr", out / "gemma_judge"
    gdir.mkdir(); jdir.mkdir()
    rows = [dict(cand_id=size["cand_id"], attribute_type="size", applicable=True, original_span="6 mm",
                 new_span="25 mm", corrupted_sentence=size["sentence"].replace("6 mm", "25 mm"), model="m", raw=None)]
    lat = next((c for c in cands if c["attribute_type"] == "laterality"), None)
    if lat:
        rows.append(dict(cand_id=lat["cand_id"], attribute_type="laterality", applicable=True,
                         corrupted_sentence=lat["sentence"].replace("right", "left", 1), model="m", raw=None))
    (gdir / "rank_000.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    gemma_verifier.main(["judge-input", "--candidates", str(out / "attribute_candidates.jsonl.gz"),
                         "--gemma_dir", str(gdir), "--output", str(out / "judge_input.jsonl.gz")])
    items = _read(out / "judge_input.jsonl.gz")
    assert len(items) == len(rows)
    yes = {k: True for k in gemma_verifier.JUDGE_KEYS}
    judged = [dict(item_id=i["item_id"], parsed=dict(yes, grammatical=i["attribute_type"] != "laterality"))
              for i in items]
    (jdir / "rank_000.jsonl").write_text("".join(json.dumps(r) + "\n" for r in judged))
    gemma_verifier.main(["filter", "--gemma_dir", str(gdir), "--judge_dir", str(jdir),
                         "--out_dir", str(out / "gemma_attr_judged")])
    attribute.main(["--candidates", str(out / "attribute_candidates.jsonl.gz"),
                    "--gemma_outputs", str(out / "gemma_attr_judged/rank_*.jsonl"),
                    "--findings", str(toy / "findings.sqlite"), "--output", str(out / "attribute_pairs.jsonl.gz")])
    attr = _read(out / "attribute_pairs.jsonl.gz")
    assert [p["attribute_type"] for p in attr] == ["size"] and "25 mm" in attr[0]["rejected_text"]
    merge.main(["index", "--pairs", str(out / "base_pairs.jsonl.gz"), str(out / "attribute_pairs.jsonl.gz"),
                "--output", str(out / "base_all.sqlite")])
    db = sqlite3.connect(out / "base_all.sqlite")
    assert db.execute("SELECT COUNT(*) FROM pairs").fetchone()[0] == len(pairs) + 1
    meta = json.loads(db.execute("SELECT value FROM metadata WHERE key='summary'").fetchone()[0])
    assert meta["schema"] == "ct_report_preference_index_v1" and meta["target_prefix"] == "Findings:\n"


def test_plan_candidates_cap():
    p = np.array([0.9, 0.4, 0.45, 0.3, 0.1])
    got = plan_pairs.plan_candidates(p, np.full(5, 0.5), np.full(5, 0.25), possible_cap=2)
    assert got.tolist() == [0, 1, 2]
    assert plan_pairs.plan_candidates(p, np.full(5, 0.5), np.full(5, 0.25), possible_cap=0).tolist() == [0, 1, 2, 3]


def test_plan_pairs_end_to_end(toy):
    out = toy / "out"
    base.main(["build", "--index", str(toy / "index.sqlite"), "--findings", str(toy / "findings.sqlite"),
               "--output_dir", str(out)])
    merge.main(["index", "--pairs", str(out / "base_pairs.jsonl.gz"), "--output", str(out / "base_all.sqlite")])
    summary = plan_pairs.build(plan_pairs.parse_args([
        "--index", str(toy / "index.sqlite"), "--findings", str(toy / "findings.sqlite"),
        "--dense", str(toy / "dense.npz"), "--calibration", str(toy / "cal.json"), "--split", str(toy / "split.json"),
        "--heads", str(toy / "heads.json"), "--hierarchy", str(toy / "hier.json"),
        "--base_store", str(out / "base_all.sqlite"), "--out_dir", str(out),
        "--template_min_reports", "2", "--plan_class_cap_frac", "1.0", "--random_keep_with_plan", "1.0"]))
    by_type = summary["pairs_by_type"]
    assert all(by_type.get(k, 0) >= 1 for k in plan_pairs.NEW_TYPES)
    db = sqlite3.connect(out / "plan_pairs.sqlite")
    for uid, kind, rej in db.execute("SELECT report_uid, pair_type, rejected_text FROM pairs"):
        if kind == "insertion_plan_fp":
            assert uid.startswith("b") and EFF_S in rej
        if kind == "multi_omission":
            assert EFF_S not in rej and NOD_S not in rej
    merged = sqlite3.connect(out / "dpo_pairs.sqlite")
    n_base = sqlite3.connect(out / "base_all.sqlite").execute("SELECT COUNT(*) FROM pairs").fetchone()[0]
    n_new = db.execute("SELECT COUNT(*) FROM pairs").fetchone()[0]
    assert merged.execute("SELECT COUNT(*) FROM pairs").fetchone()[0] == n_base + n_new

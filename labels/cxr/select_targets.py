"""Select the X-ray classifier targets from the scoped label index (Methods, X-ray classifier targets).

Rule: target 1 = ``positive_json`` (asserted OR uncertain), 0 = denied or unmentioned. Reports with verdict pass and at
least one valid film (``--cases``: JSONL with id, h5, report, one line per such report, written by the X-ray case index of
the classifier stage) whose label-DB split matches the split CSV. Classes with >= 50 positive TRAIN reports, minus
"No Finding / Normal study" -> 129 classes in the paper run.

Outputs: targets.npz (train_ids, train_Y, val_ids, val_Y, classes, vocab_index), classes.json, summary.json.

    python -m labels.cxr.select_targets --label-db cxr_labels_scoped.sqlite --cases-train cases_train.jsonl \
        --cases-val cases_val.jsonl --config labels/configs/cxr_targets.json --out-dir cxr_targets
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np

from labels.llm import load_config


def select(label_db: str, cases: dict[str, list[dict]], min_train_pos: int, drop: list[str]) -> dict:
    db = sqlite3.connect(f"file:{label_db}?mode=ro", uri=True)
    vocab = json.loads(db.execute("select value from metadata where key='vocabulary'").fetchone()[0])["labels"]
    lab = {(h5, uid): (sp, json.loads(pos), verdict) for uid, h5, sp, pos, verdict in
           db.execute("select report_uid,h5_id,split,positive_json,extraction_verdict from samples")}
    assert all(d in vocab for d in drop)
    by = {"train": [], "val": []}
    miss = {"train": 0, "val": 0}
    for s in by:
        for c in cases[s]:
            k = (c["h5"], c["report"])
            if k not in lab:
                miss[s] += 1
                continue
            sp, pos, verdict = lab[k]
            if sp != s:
                raise ValueError(f"label-DB split differs for {c['id']}")
            if verdict == "pass":
                by[s].append(dict(id=c["id"], pos=sorted(set(pos))))
        by[s].sort(key=lambda c: c["id"])
    tp, vp = np.zeros(len(vocab), np.int64), np.zeros(len(vocab), np.int64)
    for c in by["train"]:
        tp[c["pos"]] += 1
    for c in by["val"]:
        vp[c["pos"]] += 1
    keep = [i for i in range(len(vocab)) if tp[i] >= min_train_pos and vocab[i] not in drop]
    col = {v: j for j, v in enumerate(keep)}

    def mat(cs):
        Y = np.zeros((len(cs), len(keep)), np.uint8)
        for r, c in enumerate(cs):
            for i in c["pos"]:
                if i in col:
                    Y[r, col[i]] = 1
        return Y

    nt, nv = len(by["train"]), len(by["val"])
    classes = [dict(name=vocab[i], vocab_index=int(i), train_pos=int(tp[i]), val_pos=int(vp[i]),
                    train_prevalence=float(tp[i] / nt), val_prevalence=float(vp[i] / max(1, nv))) for i in keep]
    return dict(by=by, keep=keep, vocab=vocab, Yt=mat(by["train"]), Yv=mat(by["val"]), classes=classes, miss=miss)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label-db", required=True)
    ap.add_argument("--cases-train", required=True)
    ap.add_argument("--cases-val", required=True)
    ap.add_argument("--config", default="configs/cxr_targets.json")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    cfg = load_config(a.config)
    cases = {s: [json.loads(line) for line in open(p)] for s, p in (("train", a.cases_train), ("val", a.cases_val))}
    r = select(a.label_db, cases, cfg["min_train_pos"], cfg["drop_classes"])
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = [c["name"] for c in r["classes"]]
    np.savez_compressed(out / "targets.npz", train_ids=np.array([c["id"] for c in r["by"]["train"]]), train_Y=r["Yt"],
                        val_ids=np.array([c["id"] for c in r["by"]["val"]]), val_Y=r["Yv"], classes=np.array(names),
                        vocab_index=np.array(r["keep"]))
    policy = (f"target 1 = positive_json (asserted OR uncertain); 0 = denied OR unmentioned; classes with >= "
              f"{cfg['min_train_pos']} positive train reports, minus {cfg['drop_classes']}")
    json.dump(dict(policy=policy, classes=r["classes"]), open(out / "classes.json", "w"), indent=1)
    json.dump(dict(train_reports=len(r["by"]["train"]), val_reports=len(r["by"]["val"]), no_label_row=r["miss"],
                   classes=len(names)), open(out / "summary.json", "w"), indent=1)
    print(f"{len(names)} classes")


if __name__ == "__main__":
    main()

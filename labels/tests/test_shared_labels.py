"""fora.labels: states, canonicalisation, propagation, head tiers on toy data and on the shipped hierarchy."""
import json

import numpy as np

from fora.labels import (Hierarchy, assign_heads, canonicalizer, exclude_heads, load_heads, load_hierarchy, report_state,
                         to_target)
from labels.llm import stage_path

TOY = {
    "Lung disease": {"kind": "umbrella", "ancestors": [], "aliases": []},
    "Pneumonia": {"kind": "leaf", "ancestors": ["Lung disease"], "aliases": ["Lung infection"]},
    "Lung oedema": {"kind": "leaf", "ancestors": ["Lung disease"], "aliases": []},
}


def toy():
    return Hierarchy(nodes=TOY, redirect={"Pulmonary infection": "Pneumonia"})


def test_states_to_target():
    t = to_target(np.array([0, 1, 2, 3, -1]))
    assert t[:4].tolist() == [0, 1, 1, 0] and np.isnan(t[4])


def test_canon():
    c = canonicalizer(toy(), {"Chest infection": "Pneumonia", "Junk": None})
    assert c("pneumonia") == "Pneumonia" and c("Lung infection") == "Pneumonia"
    assert c("Pulmonary infection") == "Pneumonia" and c("Chest infection") == "Pneumonia" and c("Junk") is None
    assert c("Lung edema") == "Lung oedema" and c("Pneumonias") == "Pneumonia" and c("Right pneumonia") == "Pneumonia"
    assert c("Pneumonia — community acquired") == "Pneumonia" and c("nothing") is None


def test_propagation():
    h = toy()
    i = h.index
    rows = [{"asserted": {i["Pneumonia"]}, "uncertain": set(), "denied": {i["Lung oedema"]}},
            {"asserted": set(), "uncertain": set(), "denied": {i["Lung disease"], i["Pneumonia"]}}]
    s, conf = report_state(rows, h.ancestors, h.descendants, len(h))
    assert s[i["Pneumonia"]] == 1 and s[i["Lung disease"]] == 1 and s[i["Lung oedema"]] == 0
    assert sorted(conf) == sorted([i["Lung disease"], i["Pneumonia"]])
    s, _ = report_state([{"asserted": set(), "uncertain": {i["Pneumonia"]}, "denied": set()}], h.ancestors, h.descendants, len(h))
    assert s.tolist() == [2, 3, 2]


def test_heads(tmp_path):
    names = ["A", "B", "C", "D"]
    anc = {"A": [], "B": ["A"], "C": ["A"], "D": []}
    tp, tn = np.array([500, 300, 50, 10]), np.array([1000, 1000, 1000, 1000])
    h = assign_heads(names, {n: "leaf" for n in names}, anc, tp, tn, tp, tn)
    assert [x["name"] for x in h["own"]] == ["A", "B"] and h["attached"][0]["parent"] == "A" and h["no_head"] == ["D"]
    h2 = exclude_heads(h, ["B"])
    assert [x["name"] for x in h2["own"]] == ["A"] and h2["removed"] == ["B"]
    p = tmp_path / "h.json"
    p.write_text(json.dumps(h2))
    classes, parent = load_heads(p)
    assert classes == ["A", "C"] and parent == {1: 0}


def test_shipped_hierarchy_and_heads():
    h = load_hierarchy(stage_path("data/ct/hierarchy.json"))
    assert len(h) == 1459 and sum(h.kind(n) == "leaf" for n in h.names) == 1286
    classes, parent = load_heads(stage_path("data/ct/heads_1028.json"))
    assert len(classes) == 1028 and len(parent) == 279 and set(classes) <= set(h.names)
    assert "Crystalluria" not in classes and "Gastroesophageal reflux" not in classes

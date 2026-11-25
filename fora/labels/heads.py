"""Classifier head assignment from train-split label counts (Methods, classifier targets).

Tiers (positives = asserted or uncertain, negatives = denied or unmentioned, TRAIN reports only):
* own head: >= 200 positives and >= 200 negatives;
* attached head: 20-199 positives; attached to the own-head ancestor with the FEWEST train positives (the most specific
  one). Its parameters start as a copy of the parent's and are pulled towards them during training;
* rare orphan: 20-199 positives but no own-head ancestor; no head;
* no head: < 20 positives (still propagates to its ancestors in the label DB).
Excluded classes (a config list) are removed from the own heads after tiering.

The classifier consumes the heads file as ``classes = own + attached`` (in file order) and ``parent_of``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

OWN_MIN_POS = 200
OWN_MIN_NEG = 200
ATTACH_MIN_POS = 20


def assign_heads(names: Sequence[str], kinds: dict[str, str], ancestors: dict[str, list[str]],
                 train_pos: np.ndarray, train_neg: np.ndarray, val_pos: np.ndarray, val_neg: np.ndarray,
                 own_min_pos: int = OWN_MIN_POS, own_min_neg: int = OWN_MIN_NEG,
                 attach_min_pos: int = ATTACH_MIN_POS) -> dict:
    idx = {n: i for i, n in enumerate(names)}
    tier = {}
    for n in names:
        i = idx[n]
        tier[n] = "own" if train_pos[i] >= own_min_pos and train_neg[i] >= own_min_neg else (
            "rare" if train_pos[i] >= attach_min_pos else "none")
    own = [n for n in names if tier[n] == "own"]
    attached = []
    for n in names:
        if tier[n] != "rare":
            continue
        cands = [a for a in ancestors[n] if tier[a] == "own"]
        if cands:
            attached.append((n, min(cands, key=lambda a: train_pos[idx[a]])))
        else:
            tier[n] = "rare_orphan"

    def counts(n):
        i = idx[n]
        return dict(train_pos=int(train_pos[i]), train_neg=int(train_neg[i]), val_pos=int(val_pos[i]), val_neg=int(val_neg[i]))

    return dict(own=[dict(name=n, **counts(n), kind=kinds[n]) for n in own],
                attached=[dict(name=n, parent=p, **counts(n)) for n, p in attached],
                rare_orphans=[n for n in names if tier[n] == "rare_orphan"],
                no_head=[n for n in names if tier[n] == "none"])


def exclude_heads(heads: dict, exclude: Sequence[str], note: str = "") -> dict:
    """Remove excluded own heads (no attached head may depend on them)."""
    exclude = list(exclude)
    blocked = [a["name"] for a in heads["attached"] if a["parent"] in exclude or a["name"] in exclude]
    if blocked:
        raise ValueError(f"excluded classes are parents of / are attached heads: {blocked}")
    out = dict(heads)
    out["own"] = [h for h in heads["own"] if h["name"] not in exclude]
    out["removed"] = [n for n in exclude if any(h["name"] == n for h in heads["own"])]
    if note:
        out["note"] = note
    return out


def load_heads(path: str | Path) -> tuple[list[str], dict[int, int]]:
    """(class names = own + attached, {attached class index: parent class index})."""
    h = json.loads(Path(path).read_text(encoding="utf-8"))
    classes = [x["name"] for x in h["own"]] + [x["name"] for x in h["attached"]]
    cidx = {n: i for i, n in enumerate(classes)}
    return classes, {cidx[x["name"]]: cidx[x["parent"]] for x in h["attached"]}

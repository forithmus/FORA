"""Finding hierarchy and label-name canonicalisation (Methods, finding hierarchy; Supplementary Data 1).

A hierarchy file (``labels/data/ct/hierarchy.json``) holds ``nodes`` (name -> kind leaf|umbrella, parents, ancestors,
aliases, definition) and ``redirect`` (merged names -> surviving node). Nodes are indexed in sorted-name order, which is the
column order of the label DB.

``canonicalizer`` maps a free label name emitted by an extractor to a node: exact name or alias (case-insensitive) after
following redirects; then an optional curated alias table; then spelling variants (British/American, plural, side words).
This is the canonicalisation used by the label DB build.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

_SPELLING = (
    ("ischemi", "ischaemi"), ("Ischemi", "Ischaemi"), ("hemorrh", "haemorrh"), ("edema", "oedema"), ("tumor", "tumour"),
    ("esophag", "oesophag"), ("Esophag", "Oesophag"), ("Ascending aorta ", "Ascending aortic "),
    ("Aortic arch dilatation", "Thoracic aortic dilatation"),
)


@dataclass
class Hierarchy:
    nodes: dict
    redirect: dict
    names: list[str] = field(init=False)
    index: dict[str, int] = field(init=False)
    ancestors: list[np.ndarray] = field(init=False)
    descendants: dict[int, set[int]] = field(init=False)

    def __post_init__(self) -> None:
        self.names = sorted(self.nodes)
        self.index = {n: i for i, n in enumerate(self.names)}
        self.ancestors = [np.array([self.index[a] for a in self.nodes[n]["ancestors"]], np.int64) for n in self.names]
        self.descendants = defaultdict(set)
        for n in self.names:
            for a in self.nodes[n]["ancestors"]:
                self.descendants[self.index[a]].add(self.index[n])

    def __len__(self) -> int:
        return len(self.names)

    def kind(self, name: str) -> str:
        return self.nodes[name]["kind"]

    def node_ancestors(self, name: str) -> list[str]:
        return list(self.nodes[name]["ancestors"])


def load_hierarchy(path: str | Path) -> Hierarchy:
    h = json.loads(Path(path).read_text(encoding="utf-8"))
    return Hierarchy(nodes=h["nodes"], redirect=h.get("redirect", {}))


def load_alias_table(path: str | Path | None) -> dict[str, str | None]:
    """Curated extractor-name -> hierarchy-name table (value None = drop the label)."""
    if path is None:
        return {}
    table = json.loads(Path(path).read_text(encoding="utf-8"))
    table.pop("_note", None)
    return table


def canonicalizer(h: Hierarchy, alias_table: dict[str, str | None] | None = None) -> Callable[[str], str | None]:
    low = {n.lower(): n for n in h.nodes}
    al = {a.lower(): n for n, v in h.nodes.items() for a in v["aliases"]}
    table = alias_table or {}

    def exact(x: str) -> str | None:
        seen = set()
        while x in h.redirect and x not in seen:
            seen.add(x)
            x = h.redirect[x]
        return low.get(x.lower()) or al.get(x.lower())

    def variants(x: str) -> str | None:
        x = x.strip()
        if not x:
            return None
        c = exact(x)
        if c:
            return c
        x = x.split(" — ")[0].split(" (")[0].strip()
        if not x:
            return None
        br = x
        for a, b in _SPELLING:
            br = br.replace(a, b)
        for y in (br, br.capitalize(), br[0] + br[1:].lower(), x, x[:-1] if x.endswith("s") else x + "s",
                  x.replace("Right ", "").replace("Left ", "").replace("Bilateral ", "").capitalize(),
                  x.replace("carcinomatosis", "carcinomatosa"), x.replace("cancer", "carcinoma")):
            c = exact(y)
            if c:
                return c
        return None

    def canon(x: str) -> str | None:
        x0 = x.strip()
        if x0 in table:
            return exact(table[x0]) if table[x0] else None
        return variants(x)

    return canon

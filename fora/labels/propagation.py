"""Hierarchy propagation of one report's extracted findings into a full state row (Methods, report labels).

Rules (applied per report after all label sources are unioned):
* asserted -> the node and all its ancestors are asserted;
* uncertain -> the node and its ancestors are uncertain, unless asserted;
* denied -> the node and all its descendants are denied, but a denial never overrides a positive (asserted or uncertain)
  state of the same node; such a denial is a positive/denial conflict and is dropped (the positive wins);
* every other node is unmentioned (state 3, a negative with full weight).
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from .states import ASSERTED, DENIED, UNCERTAIN, UNMENTIONED


def report_state(rows: Sequence[dict[str, set[int]]], ancestors: Sequence[np.ndarray],
                 descendants: dict[int, set[int]], n_nodes: int) -> tuple[np.ndarray, list[int]]:
    """rows: one dict {asserted, uncertain, denied} of node-index sets per label source of the report.

    Returns (int8 state row, list of denied node indices that conflicted with a positive; one entry per source row)."""
    a = set().union(*[r["asserted"] for r in rows])
    u = set().union(*[r["uncertain"] for r in rows]) - a
    pos = set(a)
    for i in a:
        pos.update(ancestors[i].tolist())
    unc = set(u)
    for i in u:
        unc.update(ancestors[i].tolist())
    unc -= pos
    neg: set[int] = set()
    conflicts: list[int] = []
    for r in rows:
        for d in r["denied"]:
            if d in pos or d in unc:
                conflicts.append(d)
                continue
            neg.add(d)
            neg.update(descendants[d])
    neg -= pos | unc
    s = np.full(n_nodes, UNMENTIONED, np.int8)
    s[list(neg)] = DENIED
    s[list(unc)] = UNCERTAIN
    s[list(pos)] = ASSERTED
    return s, conflicts


def positive_closure(nodes: Iterable[int], ancestors: Sequence[np.ndarray]) -> set[int]:
    """Nodes plus all their ancestors."""
    out = set()
    for i in nodes:
        out.add(i)
        out.update(ancestors[i].tolist())
    return out

"""Map raw CT scoring-extractor output of one report to states of the 1,028 classifier classes (Methods, report evaluation).

1. Per protocol: emitted leaf -> scoring-vocabulary label (``ct_vocab``), else extractor extra by normalised name.
2. Per report: union over its protocol rows with asserted > uncertain > denied (class 14 read as 3).
3. Scoring label -> hierarchy node (``fora.labels.canonicalizer`` with hierarchy_base.json + curated alias table) ->
   class: asserted/uncertain map to the node itself if it is a class, else to its deepest ancestor that is a class
   (positive evidence for a child is positive evidence for its parent); denied maps only to the node itself. A positive
   wins over a denial of the same class.

    from labels.report_scoring.ct_states import Resolver
    states_1028, states_1106, unmapped = Resolver().report_states(rows)
"""
from __future__ import annotations

from collections import Counter

from fora.labels import canonicalizer, load_alias_table, load_heads, load_hierarchy
from labels.llm import stage_path
from labels.report_scoring.ct_vocab import (DEFAULT_EXTRAS, DEFAULT_PROTOCOLS, alias_tables, load_extras,
                                            load_vocabularies, normalize_label, scoring_vocabulary)

DEFAULT_HIERARCHY = "data/ct/hierarchy_base.json"
DEFAULT_ALIASES = "data/ct/unmapped_aliases.json"
DEFAULT_HEADS = "data/ct/heads_1028.json"


def class_maps(labels: list[str], classes: list[str], canon, nodes) -> tuple[dict, dict, Counter]:
    """up[i]: positive mapping (node or deepest class ancestor); exact[i]: denial mapping (node only)."""
    cidx = {c: j for j, c in enumerate(classes)}
    depth = {n: len(v.get("ancestors", [])) for n, v in nodes.items()}
    up, exact, how = {}, {}, Counter()
    for i, name in enumerate(labels):
        node = canon(name)
        if node is None:
            how["unmapped"] += 1
        elif node in cidx:
            up[i] = exact[i] = cidx[node]
            how["exact_class"] += 1
        else:
            anc = [a for a in nodes[node].get("ancestors", []) if a in cidx]
            if anc:
                up[i] = cidx[max(anc, key=lambda a: depth.get(a, 0))]
                how["ancestor_class"] += 1
            else:
                how["node_without_class"] += 1
    return up, exact, how


class Resolver:
    def __init__(self, protocols=DEFAULT_PROTOCOLS, extras=DEFAULT_EXTRAS, hierarchy=DEFAULT_HIERARCHY,
                 aliases=DEFAULT_ALIASES, heads=DEFAULT_HEADS):
        vocabs = load_vocabularies(protocols)
        self.base_aliases = alias_tables(vocabs)
        ex = load_extras(extras)
        self.extra_norm = {pc: {normalize_label(e): e for e in v} for pc, v in ex.items()}
        self.labels = scoring_vocabulary(vocabs, ex)
        self.label_index = {lab: i for i, lab in enumerate(self.labels)}
        self.classes, _ = load_heads(stage_path(heads))
        H = load_hierarchy(stage_path(hierarchy))
        canon = canonicalizer(H, load_alias_table(stage_path(aliases)))
        self.up, self.exact, self.how = class_maps(self.labels, self.classes, canon, H.nodes)
        self.stats = Counter()

    def leaf(self, pc: int, raw) -> str | None:
        norm = normalize_label(str(raw))
        m = self.base_aliases.get(pc, {}).get(norm, set())
        if len(m) == 1:
            self.stats["alias"] += 1
            return next(iter(m))
        if norm == "other" or norm.startswith("other "):
            self.stats["invalid_other"] += 1
            return None
        if norm in self.extra_norm.get(pc, {}):
            self.stats["extra"] += 1
            return self.extra_norm[pc][norm]
        self.stats["novel"] += 1
        return None

    def report_states(self, modalities: list[dict]):
        """modalities: [{protocol_class, asserted, uncertain, denied}] with [{leaf, quote}] items.

        Returns (class-name lists per state over the 1,028 classes, scoring-label lists, unmapped positive labels)."""
        ga, gu, gd = set(), set(), set()
        for m in modalities:
            pc = 3 if int(m["protocol_class"]) == 14 else int(m["protocol_class"])
            st = {}
            for pol in ("asserted", "uncertain", "denied"):
                st[pol] = set()
                for item in m.get(pol, []) or []:
                    c = self.leaf(pc, item.get("leaf") if isinstance(item, dict) else item)
                    if c is not None:
                        st[pol].add(c)
            a = st["asserted"]
            u = st["uncertain"] - a
            ga |= a
            gu |= u
            gd |= st["denied"] - a - u
        gu -= ga
        gd -= ga | gu
        old = {k: sorted(self.label_index[x] for x in v) for k, v in (("asserted", ga), ("uncertain", gu), ("denied", gd))}
        asserted = sorted({self.up[i] for i in old["asserted"] if i in self.up})
        uncertain = [j for j in sorted({self.up[i] for i in old["uncertain"] if i in self.up}) if j not in asserted]
        positive = set(asserted) | set(uncertain)
        denied = [j for j in sorted({self.exact[i] for i in old["denied"] if i in self.exact}) if j not in positive]
        unmapped = sorted(self.labels[i] for i in old["asserted"] + old["uncertain"] if i not in self.up)
        names = lambda idx: sorted(self.classes[j] for j in idx)  # noqa: E731
        return ({"asserted": names(asserted), "uncertain": names(uncertain), "denied": names(denied)},
                {"asserted": sorted(ga), "uncertain": sorted(gu), "denied": sorted(gd)}, unmapped)

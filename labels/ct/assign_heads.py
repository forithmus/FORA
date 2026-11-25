"""Assign classifier heads from a label DB and apply the exclusion list (Methods, classifier targets).

Tiers come from ``fora.labels.assign_heads`` on train-split counts; the classes listed in the config's ``exclude`` are
then removed from the own heads (classes that are not visible on CT).

    python -m labels.ct.assign_heads --labels labels_db/labels.npz --hierarchy labels/data/ct/hierarchy_base.json \
        --config labels/configs/ct_heads.json --output heads.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from fora.labels import assign_heads, exclude_heads, load_hierarchy
from labels.llm import load_config


def heads_from_db(labels_path: str, hierarchy_path: str, cfg: dict) -> dict:
    z = np.load(labels_path)
    H = load_hierarchy(hierarchy_path)
    names = [str(n) for n in z["nodes"]]
    assert names == H.names, "label DB node order differs from the hierarchy"
    S, sp = z["state"], z["split"]
    tr, va = sp == "train", sp != "train"
    pos = lambda m: ((S[m] == 1) | (S[m] == 2)).sum(0)  # noqa: E731
    neg = lambda m: ((S[m] == 0) | (S[m] == 3)).sum(0)  # noqa: E731
    heads = assign_heads(names, {n: H.kind(n) for n in names}, {n: H.node_ancestors(n) for n in names},
                         pos(tr), neg(tr), pos(va), neg(va), own_min_pos=cfg["own_min_pos"], own_min_neg=cfg["own_min_neg"],
                         attach_min_pos=cfg["attach_min_pos"])
    return exclude_heads(heads, cfg["exclude"], note=cfg.get("note", ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--hierarchy", required=True)
    ap.add_argument("--config", default="configs/ct_heads.json")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    h = heads_from_db(a.labels, a.hierarchy, load_config(a.config))
    json.dump(h, open(a.output, "w"), indent=1, ensure_ascii=False)
    print(f"own {len(h['own'])} + attached {len(h['attached'])} = {len(h['own']) + len(h['attached'])}; removed {h['removed']}")


if __name__ == "__main__":
    main()

"""Finding -> organ-region routing tables of the classifier (Methods, "Finding classifier", organ routing).

A route is a set of the 63 CADS-derived regions of ``fora.regions.REGION_NAMES``; the route bits of a finding are the OR of
its regions' bits (an empty route, or the marker ``ALL``, means every token). Three tables are used in training:

  table1   from the audited explicit finding -> region table (``routes/route_table.json``, two LLM reviewers):
           every finding = own route U all descendants' routes; a finding touching the lungs, a lobe or the pleura also
           gets lungs + pleura; image-audit additions for the finding and its descendants (EXTRA_TABLE1).
  table2   table1 + context regions for findings judged against a neighbouring structure (CONTEXT_TABLE2), added to the
           finding and to all its ancestors.
  table3   table2 with routing removed (route = ALL) for the findings in NO_ROUTING_TABLE3: findings where the classifier
           trained without routing was better on validation (AUPRC difference >= +0.03 with a bootstrap 95% CI above 0,
           or >= +0.03 for findings whose meaning makes the route ambiguous; also large findings with n+ >= 300 and a CI
           above 0, and organ-ectopia findings with < 10 validation positives). It also lists the findings with < 50
           training positives (and routing kept), whose route penalty is restarted at 1.0 in training stage 3.

Outputs per table: ``class_routes_<table>.json`` {finding: [regions]} and ``class_region_bits_<table>.json``
{finding: int bitmask}; table3 also writes ``route_penalty_init_table3.json``.

  build_routes.py table1 --route-table routes/route_table.json --hierarchy HIERARCHY --out routes/
  build_routes.py table2 --hierarchy HIERARCHY --out routes/
  build_routes.py table3 --heads heads.json --out routes/
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from fora.regions import REGION_NAMES

ALL_BITS = (1 << 64) - 1
LUNG_OR_PLEURA = {"lungs", "pleura", "lung_rul", "lung_rml", "lung_rll", "lung_lul", "lung_lll"}
EXTRA_TABLE1 = {"Bowel obstruction": ["stomach", "peritoneum"], "Acute pancreatitis": ["duodenum"],
                "Pancreatic neoplasms": ["duodenum"], "Colonic diverticulitis": ["pelvis"], "Appendicitis": ["pelvis"],
                "Mediastinal mass": ["hila"], "Pericardial effusion": ["mediastinum"], "Hiatal hernia": ["mediastinum"]}
CONTEXT_TABLE2 = {
    "Cardiomegaly": ["mediastinum", "lungs", "pleura", "chest_wall", "ribs"], "Hepatic steatosis": ["spleen"],
    "Pulmonary artery enlargement": ["aorta_thoracic", "mediastinum"], "Pulmonary trunk dilatation": ["aorta_thoracic", "mediastinum"],
    "Pulmonary hypertension": ["aorta_thoracic", "mediastinum"], "Mediastinal shift": ["lungs", "pleura", "heart"],
    "Pectus excavatum": ["heart", "mediastinum"], "Barrel chest": ["lungs", "pleura"], "Pericardial fat pad": ["mediastinum"],
    "Pericardial thickening": ["mediastinum"], "Tracheal deviation": ["mediastinum", "neck", "thyroid"], "Ascites": ["pelvis"],
    "Prostatomegaly": ["bladder"], "Pacemaker": ["vessels", "mediastinum"], "Sternotomy": ["ribs"], "Os trigonum": ["bones"]}
NO_ROUTING_TABLE3 = [
    "Cavum vergae", "Cavum septum pellucidum", "Os fabella", "Os trigonum", "Status post mastectomy", "Status post nephrectomy",
    "Status post orchiectomy", "Pectus excavatum", "Pericardial fat pad", "Bilateral renal enlargement", "Renal artery stenosis",
    "Chronic otomastoiditis with ossicular erosions", "Aortic valve replacement", "Liver ablation site", "Malleolar fracture",
    "Ectopic kidney", "Posterior cerebral artery hypoplasia", "Thyroid hyperplasia", "Pulmonary cavity", "Femur fracture",
    "Intrasinus calcification", "Vertebral metastases", "Ventriculoperitoneal shunt", "Status post hepatectomy",
    "Status post pneumonectomy", "Renal transplant", "Mosaic attenuation pattern", "Mediastinal mass", "Renal agenesis",
    "Crossed fused renal ectopia"]
RARE_PENALTY_MAX_TRAIN_POS = 50
RARE_PENALTY = 1.0


def route_bits(regions) -> int:
    """Regions -> int bitmask; empty or ['ALL'] -> every bit."""
    if list(regions) == ["ALL"]:
        return ALL_BITS
    bit = {n: i for i, n in enumerate(REGION_NAMES)}
    return sum(1 << bit[g] for g in regions) or ALL_BITS


def nodes_of(hierarchy: Path) -> dict:
    """Hierarchy JSON {"nodes": {name: {"ancestors": [...], ...}}} -> nodes."""
    return json.loads(Path(hierarchy).read_text())["nodes"]


def table1(route_table: dict, nodes: dict) -> dict:
    bad = [(n, g) for n, r in route_table.items() for g in r if g not in REGION_NAMES]
    missing = [n for n in nodes if n not in route_table]
    if bad or missing:
        raise SystemExit(f"route table: unknown regions {bad[:5]}, findings without a route {missing[:5]}")
    desc = collections.defaultdict(set)
    for n, v in nodes.items():
        for a in v["ancestors"]:
            desc[a].add(n)
    routes = {}
    for n in nodes:
        s = set(route_table[n])
        for d in desc[n]:
            s.update(route_table[d])
        if s & LUNG_OR_PLEURA:
            s.update(("lungs", "pleura"))
        for c, add in EXTRA_TABLE1.items():
            if n == c or c in nodes[n]["ancestors"]:
                s.update(add)
        routes[n] = sorted(s)
    return routes


def table2(routes1: dict, nodes: dict) -> dict:
    routes = {k: list(v) for k, v in routes1.items()}
    for c, add in CONTEXT_TABLE2.items():
        for n in [c] + nodes[c]["ancestors"]:
            routes[n] = sorted(set(routes[n]) | set(add))
    return routes


def table3(routes2: dict, heads: dict):
    names = {h["name"] for h in heads["own"] + heads["attached"]}
    missing = [c for c in NO_ROUTING_TABLE3 if c not in names]
    if missing:
        raise SystemExit(f"no-routing findings without a head: {missing}")
    routes = {k: list(v) for k, v in routes2.items()}
    for c in NO_ROUTING_TABLE3:
        routes[c] = ["ALL"]
    penalty = {h["name"]: RARE_PENALTY for h in heads["own"] + heads["attached"]
               if h.get("train_pos", 0) < RARE_PENALTY_MAX_TRAIN_POS and h["name"] not in NO_ROUTING_TABLE3}
    return routes, penalty


def write(out: Path, name: str, routes: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / f"class_routes_{name}.json").write_text(json.dumps(routes, indent=0))
    (out / f"class_region_bits_{name}.json").write_text(json.dumps({n: route_bits(r) for n, r in routes.items()}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table", choices=("table1", "table2", "table3"))
    ap.add_argument("--route-table", type=Path, default=Path(__file__).parent / "routes/route_table.json")
    ap.add_argument("--hierarchy", type=Path, help="finding hierarchy JSON (labels stage)")
    ap.add_argument("--heads", type=Path, help="heads JSON of the classifier (table3)")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "routes")
    a = ap.parse_args()
    if a.table == "table1":
        write(a.out, "table1", table1(json.loads(a.route_table.read_text()), nodes_of(a.hierarchy)))
    elif a.table == "table2":
        write(a.out, "table2", table2(json.loads((a.out / "class_routes_table1.json").read_text()), nodes_of(a.hierarchy)))
    else:
        routes, penalty = table3(json.loads((a.out / "class_routes_table2.json").read_text()), json.loads(a.heads.read_text()))
        write(a.out, "table3", routes)
        (a.out / "route_penalty_init_table3.json").write_text(json.dumps(penalty, indent=1))
    print(json.dumps(dict(table=a.table, out=str(a.out))))


if __name__ == "__main__":
    main()

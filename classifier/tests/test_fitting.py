"""Calibration, rule-A cut-offs, routing tables and evidence foci on synthetic data."""
import numpy as np
import torch

from fit_calibration import fit_platt, tiers, youden
from fit_rule_a import cut_a, youden_midpoint
from fora.mil.calibration import RuleA, scan_type


def test_platt_recovers_curve():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 2, (20000, 2))
    p = 1 / (1 + np.exp(-(np.array([0.5, 2.0]) * x + np.array([-1.0, 0.5]))))
    y = rng.random(p.shape) < p
    a, b = fit_platt(x, y, np.ones_like(y), l2=1.0)
    assert np.allclose(a, [0.5, 2.0], atol=0.08) and np.allclose(b, [-1.0, 0.5], atol=0.08)
    raw_plan, raw_poss, plan_cal, poss_cal = tiers(a, b, np.array([0.3, 0.9]))
    assert (raw_poss <= raw_plan).all() and (poss_cal <= plan_cal).all()


def test_youden_rules():
    z = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.95])
    y = np.array([0, 0, 0, 1, 1, 1])
    assert youden(z, y.astype(float)) == 0.8
    assert abs(youden_midpoint(z, y) - 0.55) < 1e-12
    types = np.array(["chest"] * 6)
    assert abs(cut_a(z, y, types, ["chest"]) - 0.55) < 1e-12      # < 50 positives: all-train fallback


def test_scan_type_and_rule_a(tmp_path):
    assert scan_type(["CT Chest, Abdomen and Pelvis"]) == "CAP"
    assert scan_type(["CT Chest - routine"]) == "chest"
    assert scan_type(["CT Head"]) == "headneck"
    import json
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"fora_per_scan_type": {"a": {s: 0.0 for s in ("chest", "abdomen", "headneck", "CAP", "other")}}}))
    r = RuleA(p, ["a"])
    assert r.decide(np.array([[0.0], [-1.0]]), ["chest", "other"]).ravel().tolist() == [True, False]


def test_route_bits():
    from build_routes import ALL_BITS, route_bits
    assert route_bits(["ALL"]) == ALL_BITS and route_bits([]) == ALL_BITS and route_bits(["body"]) > 0


def test_foci_single_blob():
    from fora.mil.evidence import foci
    from fora.regions import REGION_NAMES
    g = np.stack(np.meshgrid(np.arange(6), np.arange(10), np.arange(10), indexing="ij"), -1).reshape(-1, 3)
    tok = np.array([3., 8., 8.])
    bits = np.full(len(g), 1 << REGION_NAMES.index("liver"), np.int64)
    vinfo = [dict(zyx=g, world=g * tok, tok=tok, grid=[6, 10, 10], bits=bits)]
    d = np.linalg.norm(g - np.array([3, 5, 5]), axis=1)
    E = np.exp(-d ** 2)[:, None]
    XA = torch.zeros(len(g), 1048, dtype=torch.float16)
    R = foci(E, vinfo, ["Liver lesion"], XA)[0]
    assert R["share"][0] > 0.9 and np.allclose(R["ctr"][0], np.array([3, 5, 5]) * tok, atol=4)
    assert R["tva"].sum() >= 1


def test_xray_film_focus():
    from fora.mil.evidence import xray_film_focus
    E = np.zeros((40, 40))
    E[10:13, 20:23] = 1.
    r = xray_film_focus(E, (0, 640, 0, 640))
    assert r["covered"] and abs(r["peak_token_rc"][0] - 11) <= 1 and abs(r["peak_token_rc"][1] - 21) <= 1

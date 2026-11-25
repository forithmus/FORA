"""CPU tests of the diagnostic plan builder and the calibration tiers on synthetic data."""

import hashlib
import json

import numpy as np
import pytest
import torch

from fora_report.plan import calibration_tiers as CT
from fora.reportgen.plan import diagnostic_plan as DP


class CharTokenizer:
    """apply_chat_template stub: one id per character of the user content."""

    def apply_chat_template(self, messages, **_):
        return {"input_ids": [ord(c) for c in messages[0]["content"]]}


def decode(ids):
    return "".join(map(chr, ids.tolist()))


def write_calibration(path, C=8, plan=0.6, possible=0.2, positives=None, n=1000):
    labels = [f"finding {i}" for i in range(C)]
    cal = dict(labels=labels, canonical_plan_labels=labels,
               label_order_sha256=hashlib.sha256("\n".join(labels).encode()).hexdigest(),
               class_slope=[1.0] * C, class_intercept=[0.0] * C,
               class_plan_threshold=[plan] * C, class_possible_threshold=[possible] * C,
               class_val_positive=positives if positives is not None else [100] * C, n_calibration_studies=n)
    path.write_text(json.dumps(cal))
    return labels


def p_to_logit(p):
    p = torch.tensor(p, dtype=torch.float32)
    return torch.log(p / (1 - p))


def test_templates_parse():
    T = DP.load_templates()
    assert T["findings_instruction"].endswith("Conclusion.\n\n")
    assert T["plan_footer"].endswith("below threshold.")
    assert T["ct_evidence_token"].startswith(" A finding")
    assert T["cxr_probability"].startswith("\nParentheses")


def test_tiers_and_cap_by_probability(tmp_path):
    labels = write_calibration(tmp_path / "c.json", C=8)
    pb = DP.DiagnosticPlanBuilder(tmp_path / "c.json", labels, CharTokenizer(), dataset_kind="ct", plan_possible_cap=2)
    logits = p_to_logit([0.9, 0.3, 0.5, 0.25, 0.1, 0.59, 0.21, 0.65])
    sel = pb.plan_selection(logits)
    assert sel.confident == (0, 7)
    # possible = {1, 2, 3, 5, 6}; cap 2 by calibrated p keeps 5 (0.59) and 2 (0.5), returned in class order
    assert sel.possible == (2, 5)
    assert pb.planned_finding_order(logits, [7, 5, 0]) == [0, 7, 5]


def test_cap_by_fold_change(tmp_path):
    labels = write_calibration(tmp_path / "c.json", C=4, positives=[900, 10, 900, 900])
    pb = DP.DiagnosticPlanBuilder(tmp_path / "c.json", labels, CharTokenizer(), dataset_kind="ct",
                                  plan_possible_cap=1, plan_cap_rank="fold")
    sel = pb.plan_selection(p_to_logit([0.5, 0.3, 0.55, 0.1]))
    assert sel.possible == (1,)   # 0.3 / 0.0105 beats 0.55 / 0.9


def test_ct_defaults_and_text(tmp_path):
    labels = write_calibration(tmp_path / "c.json", C=3, positives=[99, 0, 0], n=999)
    pb = DP.DiagnosticPlanBuilder(tmp_path / "c.json", labels, CharTokenizer(), dataset_kind="ct")
    assert (pb.plan_possible_cap, pb.plan_cap_rank, pb.fimg_tokens, pb.evidence_descriptions) == (20, "prob", 6, True)
    logits = p_to_logit([0.8, 0.3, 0.05])
    text = decode(pb.plan_ids(logits, torch.device("cpu"), location_descriptions={0: "right upper lobe", -1: "aorta 30 mm"},
                              finding_token_classes={0: 3}))
    # base rate (99 + 0.5) / 1000 = 9.95% (printed 10.0%) -> 0.80 / 0.0995 = 8.0x
    assert "Confident findings: finding 0 (calibrated probability 0.80, 8.0x the base rate of 10.0%; right upper lobe) " \
           + "<|video|>" * 3 + "<|audio|>" * 6 + "." in text
    assert "Possible findings (lower-confidence classifier signal; verify carefully): finding 1 (calibrated probability 0.30, " in text
    assert text.startswith("Generate only the complete English radiology Findings section from this CT study.")
    assert "Automatic image measurements (verify against the images): aorta 30 mm.\n" in text
    assert "zoomed CT images" in text


def test_cxr_wording(tmp_path):
    labels = write_calibration(tmp_path / "c.json", C=2)
    pb = DP.DiagnosticPlanBuilder(tmp_path / "c.json", labels, CharTokenizer(), dataset_kind="cxr")
    text = decode(pb.plan_ids(p_to_logit([0.7, 0.01]), torch.device("cpu"), location_descriptions={0: "left lower zone"},
                              finding_token_classes=[0]))
    assert "Probable findings (calibrated probability at least 0.5): finding 0 (left lower zone; calibrated probability 0.70" in text
    assert "Possible findings (lower-confidence classifier signal; verify carefully): none." in text
    assert "<|audio|>" not in text and "radiograph feature" in text


def test_rejects_non_final_modes(tmp_path):
    labels = write_calibration(tmp_path / "c.json", C=2)
    pb = DP.DiagnosticPlanBuilder(tmp_path / "c.json", labels, CharTokenizer(), dataset_kind="ct")
    with pytest.raises(ValueError):
        pb.plan_ids(torch.zeros(2), torch.device("cpu"), report_writer_mode="abnormal")


def test_consistency_targets(tmp_path):
    labels = write_calibration(tmp_path / "c.json", C=4)
    pb = DP.DiagnosticPlanBuilder(tmp_path / "c.json", labels, CharTokenizer(), dataset_kind="ct")
    t = pb.consistency_targets(torch.tensor([1]), torch.device("cpu"), negative_class_indices=torch.tensor([2]))
    assert t.tolist() == [-1.0, 1.0, 0.0, -1.0]


def test_platt_recovers_curve_and_tiers():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20000, 2)) * 2
    true_a, true_b = np.array([2.0, 0.5]), np.array([-1.0, 0.5])
    y = rng.random(x.shape) < 1 / (1 + np.exp(-(x * true_a + true_b)))
    a, b = CT.fit_platt(x, y, np.ones_like(y), l2=1.0)
    assert np.allclose(a, true_a, atol=0.1) and np.allclose(b, true_b, atol=0.1)
    raw = 1 / (1 + np.exp(-x))
    yd, npos = CT.class_youden(raw, y, np.ones_like(y))
    t = CT.tier_thresholds(a, b, yd)
    assert (t["raw_plan"] >= yd - 1e-12).all() and (t["raw_possible"] <= t["raw_plan"]).all()
    assert (t["plan"] >= 0.5 - 1e-9).all() and (t["possible"] <= t["plan"]).all()
    assert npos.tolist() == y.sum(0).tolist()


def test_slope_floor():
    x = np.linspace(-3, 3, 200)[:, None]
    y = (x[:, 0] < 0)[:, None]          # inverted relation -> slope would be negative
    a, _ = CT.fit_platt(x, y, np.ones_like(y), l2=0.0)
    assert a[0] == pytest.approx(0.05)


def test_ct_calibration_end_to_end(tmp_path):
    rng = np.random.default_rng(1)
    C, n = 3, 400
    labels = np.array([f"c{i}" for i in range(C)])
    fid = np.array([f"case{i}" for i in range(n)])
    logit = rng.normal(size=(n, C)).astype(np.float32)
    state = np.where(rng.random((n, C)) < 1 / (1 + np.exp(-2 * logit)), 1, 3)
    state[:5, 0] = -1
    np.savez(tmp_path / "dense.npz", report_uid=fid, fora_id=fid, logit=logit, label=labels)
    np.savez(tmp_path / "labels.npz", nodes=labels, cases=fid, state=state)
    (tmp_path / "split.json").write_text(json.dumps({f: ("train" if i < 300 else "val") for i, f in enumerate(fid)}))
    cal, _ = CT.build_ct_calibration(tmp_path / "dense.npz", tmp_path / "labels.npz", tmp_path / "split.json")
    assert cal["n_calibration_studies"] == 300
    assert cal["class_val_positive"] == [int(np.isin(state[:300, j], (1, 2)).sum()) for j in range(C)]
    CT.write_json_atomic(tmp_path / "cal.json", cal)
    pb = DP.DiagnosticPlanBuilder(tmp_path / "cal.json", list(labels), CharTokenizer(), dataset_kind="ct")
    sel = pb.plan_selection(torch.tensor([5.0, 0.0, -5.0]))
    assert 0 in sel.confident and 2 not in sel.confident + sel.possible

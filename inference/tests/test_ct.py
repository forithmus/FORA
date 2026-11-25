"""CT inference smoke tests: argument parsing, input discovery, decoding helpers, foci reading and the result schema
with mocked models."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import ct_inputs
import ct_pipeline
from conftest import load_entry

ct_infer = load_entry("ct")


def test_parse_args_defaults(tmp_path):
    a = ct_infer.parse_args(["--study", "s.json", "--output", "o.json", "--work", "w", "--weights", "wt"])
    assert (a.measurements, a.opener, a.ban, a.token_cap, a.no_report, a.mil_only) == ("on", "auto", "default", 0, False, False)
    a = ct_infer.parse_args(["--study", "s.json", "--output", "o.json", "--work", "w", "--weights", "wt",
                             "--measurements", "off", "--no-report", "--ban", "ctrate"])
    assert a.measurements == "off" and a.no_report and a.ban == "ctrate"
    with pytest.raises(SystemExit):
        ct_infer.parse_args(["--study", "s.json", "--output", "o.json", "--work", "w", "--weights", "wt", "--ban", "x"])


def test_seed_and_opener():
    from fora.reportgen.decoding.opener import OPENERS, auto_opener
    from fora.reportgen.decoding.sampling import stable_seed
    # per-study seed of a public CT-RATE validation study
    assert stable_seed("valid_508_a") == 908182996
    assert auto_opener([0, 4]) == "chest" and OPENERS["chest"].startswith("Findings:\n")
    assert auto_opener([2]) == "abdomen"
    assert auto_opener([0, 2]) == "none" and OPENERS["none"] is None


def test_ban_lists():
    from fora.reportgen.decoding import bans
    assert set(ct_pipeline.BAN_CHOICES) == {"default", "ctrate", "ctrate_alt", "none"}
    for name in ct_pipeline.BAN_LISTS.values():
        bad, info = bans.load_ban_list(name)
        assert bad and info["sequences"] == len(bad)


def test_study_validation():
    with pytest.raises(ValueError):
        ct_inputs.validate_study({"study_id": "bad id!", "series": [{"path": "x"}]})
    with pytest.raises(ValueError):
        ct_inputs.validate_study({"study_id": "s1", "series": []})
    with pytest.raises(ValueError):
        ct_inputs.validate_study({"study_id": "s1", "series": [{"path": "x"}] * 33})
    assert ct_inputs.validate_study({"study_id": "s-1.a_b", "series": [{"path": "x"}]}) == "s-1.a_b"
    assert ct_inputs.study_type_ids({"study_types": [4, 0, 0]})[0] == [0, 4]
    ids, warnings = ct_inputs.study_type_ids({})
    assert ids == [29] and warnings
    ids, warnings = ct_inputs.study_type_ids({"study_types": ["CT Chest - IV Contrast", "no such protocol"]})
    assert ids == [4, 29] and len(warnings) == 1


def test_load_npz_series(tmp_path):
    data = np.full((32, 32, 20), -1000, np.int16)
    np.savez(tmp_path / "ct.npz", data=data, affine=np.diag([-0.8, -0.8, 1.5, 1.0]))
    s = ct_inputs.load_series({"path": "ct.npz", "windows": [[-160, 240], [-1000, 1000], [-1350, 150]]}, tmp_path)
    assert s.windows == ((-1000.0, 1000.0), (-160.0, 240.0), (-1350.0, 150.0))
    assert s.data.shape == (32, 32, 20)
    s = ct_inputs.load_series({"path": "ct.npz", "window_center": "40\\-600", "window_width": "400\\1500"}, tmp_path)
    assert s.windows == ((-1000.0, 1000.0), (-160.0, 240.0), (-1350.0, 150.0))
    with pytest.raises(ValueError):
        ct_inputs.load_series({"path": "ct.npz", "windows": [[10, 5]]}, tmp_path)
    (tmp_path / "x.mha").write_text("")
    with pytest.raises(ValueError):
        ct_inputs.load_series({"path": "x.mha"}, tmp_path)


def test_series_views():
    item = {"windows": [[-1000.0, 1000.0], [-160.0, 240.0], [-1350.0, 150.0]], "add_mip": True}
    kinds = [v["kind"] for v in ct_pipeline._series_views(item, True, "inference")]
    assert kinds == ["full", "native", "native", "lung_mip_10mm"]
    assert [v["kind"] for v in ct_pipeline._series_views(item, False, "inference")] == ["full", "native", "native"]
    assert [v["kind"] for v in ct_pipeline._series_views(dict(item, add_mip=False), True, "inference")][-1] == "native"


def _extent(work, region_names):
    nf = 8
    z = dict(cls=np.array([1]), prob=np.array([0.7], np.float32), fo_share=np.zeros((1, nf)), fo_ntok=np.zeros((1, nf)),
             fo_ctr=np.zeros((1, nf, 3)), fo_ext=np.zeros((1, nf, 3)), fo_reg=np.full((1, nf, 3), -1),
             fo_regf=np.zeros((1, nf, 3)), fo_left=np.full((1, nf), np.nan), fo_view=np.full((1, nf), -1))
    z["fo_share"][0, :2] = [0.8, 0.2]
    z["fo_ntok"][0, :2] = [12, 3]
    z["fo_ctr"][0, 0] = [10.0, -20.0, 5.0]
    z["fo_reg"][0, 0] = [0, 1, -1]
    z["fo_regf"][0, 0] = [0.9, 0.1, 0]
    z["fo_left"][0, 0] = 0.25
    z["fo_view"][0, :2] = [0, 1]
    (work / "extent").mkdir()
    np.savez(work / "extent/extent_r000_p0_c0000.npz", **z)


def test_evidence_foci(tmp_path):
    names = ["lung_left", "lung_right"]
    _extent(tmp_path, names)
    out = ct_pipeline.evidence_foci(tmp_path, ["a", "b"], names, [{"series": 0}, {"series": 1}])
    assert out[0]["name"] == "b" and len(out[0]["foci"]) == 2
    f0 = out[0]["foci"][0]
    assert f0["regions"] == [{"name": "lung_left", "fraction": 0.9}, {"name": "lung_right", "fraction": 0.1}]
    assert f0["centroid_mm_sra"] == [10.0, -20.0, 5.0] and f0["left_share"] == 0.25 and out[0]["foci"][1]["series"] == 1


def test_predict_schema_mocked(tmp_path, monkeypatch):
    labels = ["Finding A", "Finding B", "Finding C"]
    weights = tmp_path / "weights"
    (weights / "ct").mkdir(parents=True)
    (weights / "ct/labels.json").write_text(json.dumps(labels))
    (weights / "ct/region_names.json").write_text(json.dumps(["lung_left", "lung_right"]))
    (weights / "ct/calibration.json").write_text(json.dumps(dict(
        labels=labels, class_plan_threshold=[0.5] * 3, class_possible_threshold=[0.2, 0.3, 0.4])))
    study = tmp_path / "study.json"
    study.write_text(json.dumps({"study_id": "s1", "study_types": [0], "series": [{"path": "ct.npz"}]}))
    work = tmp_path / "work"

    def prepare_input(path, w):
        return dict(study_id="s1", study_type_ids=[0], warnings=[], needs_cads=False,
                    series=[dict(index=0, raw="raw.nii.gz", windows=[[-1000.0, 1000.0]], add_mip=True, cads_parts={})])
    views = [dict(series=0, window=dict(kind="full", lo=-1000.0, hi=1000.0), n_tokens=100)]
    monkeypatch.setattr(ct_inputs, "prepare_input", prepare_input)
    monkeypatch.setattr(ct_pipeline, "mil_config", lambda assets, w: (w / "config.json", {}))
    monkeypatch.setattr(ct_pipeline, "extract", lambda *a, **k: dict(id="s1", views=views))
    monkeypatch.setattr(ct_pipeline, "classify", lambda *a, **k: (torch.tensor([2.0, -1.0, 0.1]), labels))
    monkeypatch.setattr(ct_pipeline, "plan_selection",
                        lambda assets, labels, logits: (SimpleNamespace(confident=(0,), possible=(2,)), [0.9, 0.1, 0.45]))
    monkeypatch.setattr(ct_pipeline, "set_numerics", lambda *a: None)
    monkeypatch.setattr(ct_pipeline, "anatomy_measurements", lambda *a: dict(records=[], err=[]))
    monkeypatch.setattr(ct_pipeline, "evidence", lambda w, *a, **k: _extent(w, None))
    monkeypatch.setattr(ct_pipeline, "report", lambda *a, **k: dict(text="Findings:\nNormal.", optimizer_step=300,
                                                                    prompt_text="plan", localizations=[]))
    a = ct_infer.parse_args(["--study", str(study), "--output", str(tmp_path / "out.json"), "--work", str(work),
                             "--weights", str(weights), "--device", "cpu"])
    result = ct_infer.predict(a)
    saved = json.loads((tmp_path / "out.json").read_text())
    assert saved == json.loads(json.dumps(result))
    for key in ("schema_version", "study_id", "findings", "plan", "evidence", "report", "windows", "measurements_output"):
        assert key in saved
    assert [f["confident"] for f in saved["findings"]] == [True, False, False]
    assert [f["possible"] for f in saved["findings"]] == [False, False, True]
    assert set(saved["findings"][0]) == {"index", "name", "raw_logit", "probability", "threshold", "possible_threshold",
                                         "confident", "possible"}
    assert saved["plan"] == {"confident": ["Finding A"], "possible": ["Finding C"]}
    assert saved["evidence"][0]["foci"][0]["regions"][0]["name"] == "lung_left"
    assert saved["report"]["text"].startswith("Findings:")
    with pytest.raises(ValueError):
        ct_infer.predict(a)          # the work directory is no longer empty

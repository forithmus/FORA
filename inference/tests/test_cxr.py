"""Chest X-ray inference smoke tests: argument parsing, film reading, signed-evidence localization and the result
schema with a mocked visual stream and plan builder."""
import json
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import cxr_inputs
import cxr_localize
from conftest import load_entry

cxr_infer = load_entry("cxr")
CFG = json.loads((cxr_infer.DEFAULT_CONFIG).read_text())


def test_parse_args():
    a = cxr_infer.parse_args(["--study", "s.json", "--output", "o.json", "--work", "w", "--weights", "wt"])
    assert not a.no_report and a.device == "cuda:0"
    assert cxr_infer.parse_args(["--study", "s", "--output", "o", "--work", "w", "--weights", "x", "--no-report"]).no_report
    assert CFG["decoding"] == {"greedy": True, "num_beams": 1, "max_new_tokens": 1024} and CFG["plan"]["promote_k"] == 8


def test_read_pixels_and_case(tmp_path):
    import h5py
    from PIL import Image
    film = (np.arange(300 * 200) % 4096).reshape(300, 200).astype(np.uint16)
    np.save(tmp_path / "a.npy", film)
    Image.fromarray((film % 256).astype(np.uint8)).save(tmp_path / "b.png")
    with h5py.File(tmp_path / "store.h5", "w") as f:
        f.create_dataset("p/r/a/s/data", data=film[..., None])
    assert cxr_inputs.read_pixels({"path": "a.npy"}, tmp_path).dtype == np.float32
    assert cxr_inputs.read_pixels({"path": "b.png"}, tmp_path).shape == (300, 200)
    np.testing.assert_array_equal(cxr_inputs.read_pixels({"h5": "store.h5", "key": ["p", "r", "a", "s"]}, tmp_path), film)
    case = cxr_inputs.make_case({"study_id": "x1", "films": [{"path": "a.npy", "view": "PA"},
                                                            {"path": "b.png", "view": "lateral"}]}, tmp_path)
    assert case["films"] == [["0", "0", 0, 0, 0], ["1", "1", 1, 0, 0]]
    with pytest.raises(ValueError):
        cxr_inputs.make_case({"study_id": "x1", "films": []}, tmp_path)
    with pytest.raises(ValueError):
        cxr_inputs.make_case({"study_id": "x1", "films": [{"path": "a.npy", "view": "oblique"}]}, tmp_path)


def _toy_bag():
    gy, gx = 4, 6
    n, c = gy * gx, 2
    rows, cols = np.divmod(np.arange(n), gx)
    geometry = [torch.tensor(rows, dtype=torch.float32), torch.tensor(cols, dtype=torch.float32)]
    L = torch.zeros(n, c)
    L[9, 0] = 5.0                       # one strongly positive token for class 0 at (row 1, col 3)
    w = torch.full((n, c), 1.0 / n)
    return [(gy, gx)], geometry, L, w


def test_localize_peak(tmp_path):
    grids, geometry, L, w = _toy_bag()
    films = [["0", "0", 0, 0, 0]]
    boxes = {"0": dict(content_box_rc=[10, 74, 20, 116], shape=[100, 140])}
    out = cxr_localize.localize(L, w, torch.ones(2), geometry, grids, films, boxes, [0, 1], {"a": 0.0, "b": 0.0},
                                ["a", "b"], tmp_path, CFG["localization"])
    assert out[0]["covered"] and out[0]["peak_token_rc"] == [1, 3]
    assert out[0]["peak_pixel_rc"] == [10 + 16 + 8, 20 + 48 + 8]
    assert out[1]["covered"] is False and out[1]["evidence_mass"] == 0.0
    assert (tmp_path / "evidence_maps.npz").exists()


def test_predict_schema_mocked(tmp_path, monkeypatch):
    labels = ["Cardiomegaly", "Pleural effusion"]
    weights = tmp_path / "weights"
    (weights / "cxr").mkdir(parents=True)
    for name, value in [("mil_config.json", {"training": {}}), ("calibration.json", {"labels": labels}),
                        ("class_groups.json", {}), ("device_classes.json", []),
                        ("highauc_classes.json", {"high": ["Cardiomegaly"]}),
                        ("localization_reference_bc.json", {"b_mean": {"Cardiomegaly": 0.0, "Pleural effusion": 0.0}})]:
        (weights / "cxr" / name).write_text(json.dumps(value))
    np.save(tmp_path / "pa.npy", np.ones((64, 96), np.float32))
    study = tmp_path / "study.json"
    study.write_text(json.dumps({"study_id": "x1", "films": [{"path": "pa.npy", "view": "frontal"}]}))
    grids, geometry, L, w = _toy_bag()

    class Stream:
        classes = labels
        head = SimpleNamespace(output_scale=torch.ones(2))

        def __init__(self, *a, **k):
            pass

        def prepare(self, case, load_film):
            self.boxes_loader = load_film
            load_film(case["films"][0])
            return dict(grids=grids)

        def encode(self, bag, location, keep_scores):
            return SimpleNamespace(logits=torch.tensor([1.5, -2.0]), total_tokens=24, location=np.zeros((2, 5)),
                                   token_logits=L, weights=w, pooled=torch.zeros(2, 1048))

        def film_geometry(self, bag):
            return geometry

    class Builder:
        class_threshold = torch.tensor([0.5, 0.5])
        class_possible_threshold = torch.tensor([0.2, 0.2])

        def plan_selection(self, logits):
            return SimpleNamespace(confident=(0,), possible=())

        def probabilities(self, logits):
            return torch.sigmoid(logits)

    import fora.reportgen.cxr.stream as stream_module
    monkeypatch.setattr(stream_module, "CXRVisualStream", Stream)
    xray = types.ModuleType("fora.geometry.xray")
    xray.content_bbox = lambda img: (0, img.shape[0], 0, img.shape[1])
    xray.to_unit = lambda img: img
    monkeypatch.setitem(sys.modules, "fora.geometry.xray", xray)
    monkeypatch.setattr(cxr_infer, "plan_builder", lambda assets, labels, tok: Builder())
    monkeypatch.setattr(cxr_infer, "location_texts", lambda labels, groups, loc: {0: "enlarged cardiac silhouette"})
    monkeypatch.setattr(cxr_infer, "write_report", lambda *a: (dict(text="Findings:\nCardiomegaly.", optimizer_step=200,
                                                                    prompt_text="plan"), Builder()))
    out = tmp_path / "out.json"
    a = cxr_infer.parse_args(["--study", str(study), "--output", str(out), "--work", str(tmp_path / "work"),
                              "--weights", str(weights), "--device", "cpu"])
    cxr_infer.predict(a)
    saved = json.loads(out.read_text())
    for key in ("schema_version", "study_id", "films", "findings", "plan", "localization", "report"):
        assert key in saved
    assert saved["films"] == [{"index": 0, "content_box_rc": [0, 64, 0, 96], "shape": [64, 96]}]
    assert saved["findings"][0]["confident"] and saved["findings"][0]["location_text"] == "enlarged cardiac silhouette"
    assert saved["plan"] == {"confident": ["Cardiomegaly"], "possible": []}
    assert saved["localization"][0]["covered"] and saved["report"]["optimizer_step"] == 200
    a.no_report = True
    cxr_infer.predict(a)
    assert "report" not in json.loads(out.read_text())

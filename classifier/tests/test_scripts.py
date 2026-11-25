"""Every classifier script imports; X-ray augmentation keeps the film shape; CT warm start copies rows by name."""
import importlib

import numpy as np
import torch


def test_imports():
    for name in ("train_ct", "train_xray", "predict_ct", "predict_xray", "fit_calibration", "fit_rule_a", "build_routes",
                 "export_evidence_ct", "xray_evidence_reference", "fora.mil.render", "fora.mil.xray"):
        importlib.import_module(name)


def test_augment_shape():
    from train_xray import augment
    A = dict(p_gamma=1, gamma=[0.7, 1.4], stretch_lo_pct_max=3., stretch_hi_pct_min=97., p_histeq=1, p_resample=1,
             resample_long=[64, 96], p_quant8=1, quant_lo_pct_max=2., quant_hi_pct_min=98., p_jpeg=1, jpeg_quality=[60, 95])
    img = np.random.default_rng(0).random((150, 120)).astype(np.float32) * 4000
    out = augment(img, np.random.default_rng(1), A)
    assert out.shape == img.shape and np.isfinite(out).all()


def test_warm_start_and_attach_decay():
    from fora.mil.model import build_head
    from train_ct import attach_decay, warm_start
    src_head = build_head(3, dim=16)
    with torch.no_grad():
        src_head.output_bias.copy_(torch.tensor([1., 2., 3.]))
    blk = torch.nn.Linear(2, 2)
    model = dict(head=build_head(4, dim=16), block=blk, child=torch.tensor([3]), par=torch.tensor([0]))
    src = dict(block=blk.state_dict(), head=src_head.state_dict(), classes=["b", "a", "c"])
    classes = ["a", "b", "c", "d"]
    warm_start(model, src, classes, {n: i for i, n in enumerate(classes)}, {3: 0}, {"route_penalty_init": {"c": 1.0}}, 0)
    ob = model["head"].output_bias.detach()
    assert ob.tolist() == [2., 1., 3., 2.]
    assert abs(float(torch.nn.functional.softplus(model["head"].route_bias_raw[2])) - 1.0) < 1e-6
    with torch.no_grad():
        model["head"].output_bias[3] = 10.
    attach_decay(model, 0.5)
    assert abs(float(model["head"].output_bias[3]) - 6.) < 1e-6

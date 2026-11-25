"""CPU smoke tests of the pre-training data, views, objective and one optimizer step (no process group, tiny network)."""
import math

import torch

from conftest import tiny_backbone


def test_views_twins_cover_same_voxels():
    import random
    from fora_pretrain.views import overlap_pairs, shifted_start, snap_second_start, token_offset
    rng = random.Random(0)
    for _ in range(50):
        container, crop = (80, 300, 280), (64, 192, 192)
        g0 = (rng.randrange(17), rng.randrange(109), rng.randrange(89))
        g1 = snap_second_start(g0, shifted_start(g0, container, crop, 0.25, rng), container, crop)
        off = token_offset(g0, g1)
        a, b = overlap_pairs(off, (32, 12, 12))
        za, zb = a // 144, b // 144
        assert torch.all(g0[0] + 2 * za == g1[0] + 2 * zb)


def test_binned_sinkhorn_is_per_bin():
    from fora_pretrain.objective import binned_sinkhorn, distributed_sinkhorn, position_bin_ids
    logits = torch.randn(40, 16)
    idx = torch.randperm(64)[:40]
    ids = position_bin_ids(idx, 8, (2, 2, 2), (2, 2, 2))
    q = binned_sinkhorn(logits, 0.07, ids, 8)
    for b in range(8):
        rows = (ids == b).nonzero().flatten()
        if len(rows):
            assert torch.allclose(q[rows], distributed_sinkhorn(logits[rows], 0.07))


def _tiny_learner(grid, gram=False):
    from fora_pretrain.objective import DINO3DLearner, LossWeights, normalize_prototypes
    learner = DINO3DLearner(tiny_backbone(), dino_prototypes=32, ibot_prototypes=32, dino_head_hidden_dim=32,
                            ibot_head_hidden_dim=32, dino_bottleneck_dim=16, ibot_bottleneck_dim=16, position_grid=grid,
                            position_bins=(2, 2, 2), ibot_loss_chunk_size=7, cross_chunk_size=5, with_gram_anchor=gram,
                            loss_weights=LossWeights(gram=1.0 if gram else 0.0))
    normalize_prototypes(learner)
    return learner


def _step(learner, batch):
    from fora_pretrain.objective import METRIC_KEYS
    from fora_pretrain.optim import build_param_groups, GroupedAdamW
    opt = GroupedAdamW(build_param_groups(learner.student, layerwise_decay=0.9, patch_embed_lr_mult=0.2, n_blocks=2), lr=1e-3)
    learner.train()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss, metrics = learner(batch, teacher_temperature=0.04, step=0)
    assert torch.isfinite(loss) and set(METRIC_KEYS) <= set(metrics)
    assert float(metrics["cross_pairs"]) > 0
    loss.backward()
    opt.step()
    learner.update_teacher(0.992)
    return metrics


def test_ct_dataset_and_step(ct_h5):
    from fora_pretrain.data_ct import CTDinoDataset, count_reconstructions
    from fora_pretrain.views import CropSpec, collate
    assert count_reconstructions(str(ct_h5)) == 2
    spec = CropSpec((8, 32, 32), (4, 16, 16), 2)
    ds = CTDinoDataset([str(ct_h5)], spec, mip_probability=1.0)
    samples = [ds[(i % len(ds), 0)] for i in range(2)]
    s = samples[0]
    assert s["teacher_global"].shape == (2, 1, 8, 32, 32) and s["student_local"].shape == (2, 1, 4, 16, 16)
    assert s["reconstruction_weight"] == 0.5
    batch = collate(samples)
    batch["loss_weights"] = batch["sample_weights"] * 2.0
    _step(_tiny_learner((4, 2, 2)), batch)


def test_gram_stage_step(ct_h5):
    from fora_pretrain.data_ct import CTDinoDataset
    from fora_pretrain.views import CropSpec, collate
    ds = CTDinoDataset([str(ct_h5)], CropSpec((8, 32, 32), (4, 16, 16), 2))
    learner = _tiny_learner((4, 2, 2), gram=True)
    learner.reset_gram_anchor_from_teacher()
    metrics = _step(learner, collate([ds[(0, 0)], ds[(1, 0)]]))
    assert float(metrics["gram"]) >= 0


def test_cxr_dataset_and_step(cxr_h5):
    from fora_pretrain.data_cxr import CXRDinoDataset, parse_windows
    from fora_pretrain.views import CropSpec, collate
    windows = parse_windows("0:1,0:0.65,0.2:0.8,0.45:1")
    assert windows[1] == (0.0, 0.65)
    ds = CXRDinoDataset([str(cxr_h5)], CropSpec((2, 64, 64), (2, 32, 32), 2), student_windows=windows)
    s = ds[(0, 0)]
    assert s["teacher_global"].shape == (2, 1, 2, 64, 64)
    assert float(s["teacher_global"].float().min()) >= -1 and float(s["teacher_global"].float().max()) <= 1
    _step(_tiny_learner((1, 4, 4)), collate([s, ds[(0, 1)]]))


def test_schedules_match_recipe():
    from fora_pretrain.train import parse_args
    from fora_pretrain.optim import schedules
    from pathlib import Path
    args_file = Path(__file__).resolve().parents[1] / "configs" / "ct_phase1.args"
    args = parse_args(["--h5-folder", "x", "--splits-csv", "y", "--output-dir", "z"] + args_file.read_text().split())
    assert args.stop_after_step == 20000 and args.steps == 125000
    peak = args.lr * math.sqrt(1024 / 1024)
    s = schedules(args, 20000, peak)
    assert 0.9 < s["lr"] / peak < 0.97
    assert schedules(args, 0, peak)["teacher_temperature"] == 0.04
    gram = parse_args(["--h5-folder", "x", "--splits-csv", "y", "--output-dir", "z"] +
                      (args_file.parent / "ct_phase2_gram.args").read_text().split())
    assert gram.stage == "gram" and gram.stop_after_step == 5000 and schedules(gram, 0, 1)["teacher_temperature"] == 0.07

"""CPU tests of the CT writer trainer's data side: prompts, schedule, logit cache, index dataset, configs."""
from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

import numpy as np
import pytest
import torch

from fora_report.ct import prompts
from fora_report.ct.args import parse_args
from fora_report.ct.mil_logits import CACHE_FORMAT, DenseMILLogitStore
from fora_report.ct.report_data import CTReportIndexDataset, read_index_metadata, shard_counts
from fora_report.ct.schedule import TokenBalancedEpochSampler, cosine_factor, cosine_scheduler

STAGE = Path(__file__).resolve().parents[1]


class ToyTokenizer:
    eos_token_id = 1

    def __call__(self, text, add_special_tokens=False):
        class Out:
            input_ids = [2 + (ord(ch) % 50) for ch in text]
        return Out()

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, return_dict=True):
        ids = [100] + self(messages[0]["content"]).input_ids + [101]
        return {"input_ids": ids}


def test_trainer_imports_and_writer_signature():
    import inspect
    import fora_report.ct.train as train
    from fora.reportgen.writer.model import CTReportGenerator
    accepted = set(inspect.signature(CTReportGenerator.__init__).parameters)
    passed = {"visual_dim", "llm_dim", "num_visual_tokens", "resampler_depth", "resampler_heads",
              "semantic_embeddings", "eligible_class_indices", "finding_feature_dim", "finding_placeholder_id",
              "calibration_slope", "calibration_intercept", "diagnostic_consistency_classes", "fimg_tokens",
              "fimg_placeholder_id"}
    assert passed <= accepted
    forward = set(inspect.signature(CTReportGenerator.forward).parameters)
    assert {"visual_tokens", "modality_tokens", "prompt_ids", "target_ids", "positive_class_indices",
            "precomputed_mil_logits", "diagnostic_plan_ids", "diagnostic_consistency_targets",
            "finding_features"} <= forward
    assert callable(train.main)


def test_prompt_and_target_ids():
    tok = ToyTokenizer()
    instruction = prompts.writer_instruction()
    assert instruction.startswith("Generate only the complete English radiology Findings section")
    ids = prompts.prompt_ids(tok, torch.device("cpu"))
    assert ids.tolist() == [100] + tok(instruction).input_ids + [101]
    target = prompts.target_ids(tok, "No effusion.", 2048, torch.device("cpu"))
    assert target.tolist() == tok("Findings:\nNo effusion.").input_ids + [1]
    short = prompts.target_ids(tok, "x" * 100, 8, torch.device("cpu"))
    assert short.numel() == 8 and short[-1] == 1
    with pytest.raises(ValueError):
        prompts.prompt_ids(tok, torch.device("cpu"), "cxr")


def test_token_balanced_sampler_covers_and_aligns_quantiles():
    counts = [[5, 1, 9, 3, 7], [2, 8, 6, 4]]
    steps = 5
    orders = [list(TokenBalancedEpochSampler(c, steps, 0, 0, 42, local_rank=r, local_world_size=2))
              for r, c in enumerate(counts)]
    assert sorted(orders[0]) == list(range(5))
    assert set(orders[1]) == set(range(4)) and len(orders[1]) == 5      # one padding repeat
    # Matched quantiles: both ranks run their k-th cheapest study at the same step.
    rank_of = [{i: sorted(range(len(c)), key=lambda j: c[j]).index(i) for i in range(len(c))} for c in counts]
    assert [rank_of[0][i] for i in orders[0]].index(0) == [rank_of[1][i] for i in orders[1]].index(0)
    resumed = list(TokenBalancedEpochSampler(counts[0], steps, 0, 2, 42, local_rank=0, local_world_size=2))
    assert resumed == orders[0][2:]
    other_epoch = list(TokenBalancedEpochSampler(counts[0], steps, 1, 0, 42, local_rank=0, local_world_size=2))
    assert sorted(other_epoch) == list(range(5))


def test_cosine_schedule_and_shortened_horizon():
    param = torch.nn.Parameter(torch.zeros(1))
    plain = cosine_scheduler(torch.optim.SGD([param], lr=1.0), 100, 0.05)
    assert plain.get_last_lr()[0] == pytest.approx(1 / 5)
    assert cosine_factor(100, 100, 5) == pytest.approx(0.0)
    opt = torch.optim.SGD([param], lr=1.0)
    sched = cosine_scheduler(opt, 4666, 0.05, initial_horizon_steps=6999, initial_horizon_until_step=1358)
    lrs = []
    for _ in range(1360):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step(); sched.step()
    warmup = int(6999 * 0.05)
    assert warmup == 349
    assert lrs[100] == pytest.approx(101 / warmup)
    assert lrs[1357] == pytest.approx(cosine_factor(1357, 6999, warmup))
    assert lrs[1358] == pytest.approx(cosine_factor(1358, 4666, warmup))


def _dense_cache(root: Path, uids, labels, source="/data/dense.npz"):
    root.mkdir()
    np.save(root / "report_uid.npy", np.array(uids))
    np.save(root / "logit.npy", np.arange(len(uids) * len(labels), dtype=np.float32).reshape(len(uids), -1))
    np.save(root / "full_token_count.npy", np.full(len(uids), 10, dtype=np.int64))
    np.save(root / "label.npy", np.array(labels))
    np.save(root / "class_index.npy", np.arange(len(labels)))
    (root / "manifest.json").write_text(json.dumps({"format": CACHE_FORMAT, "source": source}))


def test_dense_logit_store(tmp_path):
    labels = ["a", "b", "c"]
    _dense_cache(tmp_path / "dense", ["u1", "u2"], labels)
    store = DenseMILLogitStore(tmp_path / "dense", labels, calibration={"source_predictions": "/data/dense.npz"})
    assert store.get("u2", torch.device("cpu")).tolist() == [[3.0, 4.0, 5.0]]
    with pytest.raises(ValueError):
        store.require_coverage(["u3"])
    with pytest.raises(ValueError):
        DenseMILLogitStore(tmp_path / "dense", labels, calibration={"source_predictions": "/other.npz"})


def test_index_dataset_reads_findings(tmp_path):
    h5py = pytest.importorskip("h5py")
    index = tmp_path / "index.sqlite"
    c = sqlite3.connect(index)
    c.execute("CREATE TABLE metadata(key TEXT, value TEXT)")
    c.execute("INSERT INTO metadata VALUES ('global', ?)", (json.dumps({"labels": ["a", "b"]}),))
    c.execute("CREATE TABLE samples(report_uid TEXT, patient_uid TEXT, h5_id TEXT, split TEXT, protocol_class INT, "
              "series_json TEXT, study_type_ids_json TEXT, positive_json TEXT, negative_json TEXT)")
    c.execute("INSERT INTO samples VALUES ('r1','p1','s0','train',0,'[]','[3]','[1]','[0]')")
    c.execute("INSERT INTO samples VALUES ('r2','p1','s0','val',0,'[]','[3]','[]','[]')")
    c.commit(); c.close()
    with h5py.File(tmp_path / "s0.h5", "w") as f:
        f.create_dataset("p1/r1/report/structured/findings", data="Small left effusion.")
    assert read_index_metadata(index)["global"]["labels"] == ["a", "b"]
    assert shard_counts(index, split="train") == {"s0": 1}

    class Toy(CTReportIndexDataset):
        def render_report(self, sample):
            return {"uid": sample["report_uid"]}, None, None

    ds = Toy(index_path=index, h5_root=tmp_path, split="train", h5_ids=["s0"], n_classes=2)
    image, _m, _c, study_types, pos, neg, report, uid = ds[0]
    assert len(ds) == 1 and uid == "r1" and report == "Small left effusion."
    assert pos.tolist() == [1] and neg.tolist() == [0] and study_types.tolist() == [3]


@pytest.mark.parametrize("name", ["ct_sft1", "ct_sft2", "ct_dpo"])
def test_configs_parse(name, tmp_path):
    required = ["--index", "i", "--h5_root", "h", "--dino_mil_config", "c", "--dino_mil_checkpoint", "k",
                "--dino_cache_root", "d", "--mil_calibration", "m", "--precomputed_mil_predictions", "p",
                "--finding_store", "f", "--llm_path", "l", "--results_dir", str(tmp_path)]
    extra = ["--dpo_pairs", "x", "--dpo_reference", "r"] if name == "ct_dpo" else []
    args = parse_args(["--config", str(STAGE / "configs" / f"{name}.json"), *required, *extra])
    if name == "ct_sft1":
        assert args.epochs == 2 and args.initial_horizon_epochs == 3 and args.plan_possible_cap == 0
        assert math.isclose(args.connector_lr, 1e-4) and math.isclose(args.vision_lr, 1e-5)
    if name == "ct_sft2":
        assert args.limit_train_per_rank == 1750 and math.isclose(args.vision_lr, 5e-6)
    if name == "ct_dpo":
        assert args.limit_train_per_rank == 300 and args.dpo_beta == 0.1 and args.dpo_pairs_per_visit == 2
        assert math.isclose(args.warmup_fraction, 0.02) and math.isclose(args.llm_lr, 1e-5)

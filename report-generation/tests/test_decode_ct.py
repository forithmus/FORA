"""CPU smoke tests for CT decoding: per-study seed, phrase-ban lists, forced opener, sampling settings, configs."""
import hashlib
import json
from pathlib import Path

import pytest
import torch

from fora.reportgen.decoding import bans
from fora_report.decoding import records
from fora.reportgen.decoding.generator import caps_to_try, decode
from fora.reportgen.decoding.opener import OPENERS, append_opener, opener_text, prepend_opener
from fora_report.decoding.run import arguments
from fora.reportgen.decoding.sampling import SamplingConfig, stable_seed

STAGE = Path(__file__).resolve().parents[1]


def test_stable_seed_matches_definition():
    uid = "study_0001"
    expected = int.from_bytes(hashlib.sha256(b"temperature_v1|study_0001|42").digest()[:8], "big") % (2 ** 31)
    assert stable_seed(uid) == expected == stable_seed(uid, 42)
    assert 0 <= expected < 2 ** 31
    assert stable_seed(uid, 43) != expected


def test_sampling_config():
    t08 = SamplingConfig.named("t08")
    assert t08.generate_kwargs() == {"do_sample": True, "temperature": 0.8, "top_p": 0.9, "top_k": 0, "num_beams": 1}
    assert t08.record("u")["per_study_seed"] == stable_seed("u")
    greedy = SamplingConfig.named("greedy")
    assert greedy.generate_kwargs() == {"do_sample": False, "num_beams": 1} and greedy.seed_for("u") is None


class ToyTokenizer:
    """Whitespace/character tokenizer: each character is one token id (ord + 1)."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) + 1 for c in text]


def test_variants_and_bad_words_ids():
    assert bans.variants("prior") == sorted({" prior", "prior", " Prior", "Prior"})
    enc = ToyTokenizer().encode
    ids = bans.build_bad_words_ids(["prior"], ["Comparison"], "Findings:\n", [1999, 2000], enc)
    assert [ord(c) + 1 for c in "prior"] in ids and [ord(c) + 1 for c in " Prior"] in ids
    assert enc("Findings:\nComparison") in ids and enc("1999") in ids and enc("2000") in ids
    assert ids == sorted(ids) and len(ids) == 4 + 1 + 2


@pytest.mark.parametrize("name,count", [("prior_comparison", 626), ("old_imaging", 702), ("comparative", 734)])
def test_shipped_ban_lists(name, count):
    ids, info = bans.load_ban_list(name)
    assert len(ids) == count == info["sequences"]
    data = json.loads(bans.resolve(name).read_text())
    assert data["header"] == "Findings:\n" and all(min(s) > 0 for s in ids)
    assert bans.load_ban_list("none") == (None, None)


def test_opener_append_and_prepend():
    table = torch.nn.Embedding(50, 8)
    emb = torch.randn(1, 5, 8)
    att = torch.ones(1, 5, dtype=torch.long)
    types = torch.ones(1, 5, dtype=torch.long)
    pre = [3, 4, 7]
    e2, a2, t2 = append_opener(emb, att, types, pre, table)
    assert e2.shape == (1, 8, 8) and torch.equal(e2[:, :5], emb)
    assert torch.allclose(e2[0, 5:], table(torch.tensor(pre)))
    assert a2.tolist() == [[1] * 8] and t2.tolist() == [[1] * 5 + [0] * 3]
    out = prepend_opener(torch.tensor([[9, 10]]), pre)
    assert out.tolist() == [[3, 4, 7, 9, 10]]
    assert opener_text("chest") == OPENERS["chest"] and opener_text("none") is None


class FakeLLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.table = torch.nn.Embedding(20, 4)
        self.calls = []

    def get_input_embeddings(self):
        return self.table

    def generate(self, **kwargs):
        kwargs["rand"] = torch.rand(1).item()   # RNG state right after the per-study seed
        self.calls.append(kwargs)
        return torch.tensor([[11, 12]])


def test_decode_order_of_operations():
    model = torch.nn.Module()
    model.llm = FakeLLM()
    emb, att, types = torch.zeros(1, 3, 4), torch.ones(1, 3, dtype=torch.long), torch.zeros(1, 3, dtype=torch.long)
    ids = decode(model, emb, att, types, max_new_tokens=10, stop_ids=[1, 2], uid="u1", sampling=SamplingConfig(),
                 opener_ids=[5, 6], bad_words_ids=[[7]])
    call = model.llm.calls[-1]
    assert ids.tolist() == [[5, 6, 11, 12]]
    assert call["max_new_tokens"] == 8 and call["inputs_embeds"].shape[1] == 5 and call["bad_words_ids"] == [[7]]
    assert call["do_sample"] and call["temperature"] == 0.8 and call["top_p"] == 0.9 and call["top_k"] == 0
    assert call["eos_token_id"] == [1, 2] and call["pad_token_id"] == 1 and call["logits_to_keep"] == 1
    torch.manual_seed(stable_seed("u1"))
    assert call["rand"] == torch.rand(1).item()


def test_caps_and_records(tmp_path):
    assert caps_to_try(0, [2097152, 1048576]) == [0, 2097152, 1048576]
    assert caps_to_try(1048576, [2097152, 524288]) == [1048576, 524288]
    for shard, uid in ((0, "a"), (1, "b")):
        with records.shard_path(tmp_path, shard).open("a") as h:
            h.write(json.dumps({"uid": uid, "checkpoint": "c", "generation_max_new_tokens": 2048, "optimizer_step": 3}) + "\n")
    summary, missing = records.merge_outputs(tmp_path, ["b", "a", "z"], 2, "c", 2048)
    assert missing == ["z"] and summary["count"] == 2
    assert [json.loads(l)["uid"] for l in open(tmp_path / "predictions.jsonl")] == ["b", "a"]


@pytest.mark.parametrize("name", ["decode_ct_fora_val.json", "decode_ct_ctrate.json", "decode_ct_merlin.json"])
def test_configs_parse(name):
    args = arguments(["--config", str(STAGE / "configs" / name), "--output", "x"])
    assert args.temperature == 0.8 and args.top_p == 0.9 and args.plan_possible_cap == 20
    assert args.opener in OPENERS and (args.ban_list == "none" or bans.resolve(args.ban_list).is_file())


def test_constructor_signatures_match_writer_and_plan():
    """The keyword arguments build_generator / run pass must bind to the writer, plan and store constructors."""
    import inspect
    from fora_report.ct.mil_logits import DenseMILLogitStore
    from fora.reportgen.evidence.finding_store import FindingLocStore
    from fora.reportgen.plan.diagnostic_plan import DiagnosticPlanBuilder
    from fora.reportgen.writer.dino_stream import DinoCTReportGenerator
    from fora.reportgen.writer.llm import build_gemma_lora
    from fora.reportgen.writer.model import CTReportGenerator
    inspect.signature(build_gemma_lora).bind("p", "cpu", lora_r=16, lora_alpha=32, fimg_tokens=6)
    inspect.signature(DiagnosticPlanBuilder).bind("c", [], None, "all_classes", dataset_kind="ct", plan_possible_cap=20,
                                                  plan_cap_rank="prob", plan_prob=True, fimg_tokens=6)
    inspect.signature(FindingLocStore).bind("r", [], descriptions="d", ftok="f", study_meas="s", fimg_views=6)
    inspect.signature(DenseMILLogitStore).bind("r", [], calibration={})
    base = dict(semantic_embeddings=None, eligible_class_indices=None, calibration_slope=None, calibration_intercept=None,
                diagnostic_consistency_classes=0, diagnostic_consistency_weight=0.0, finding_feature_dim=1064,
                finding_placeholder_id=1, fimg_tokens=6, fimg_views=6, fimg_placeholder_id=2)
    inspect.signature(DinoCTReportGenerator.__init__).bind(None, None, visual_dim=1048, llm_dim=8, resampler_dim=1024,
                                                            n_study_types=32, num_visual_tokens=512, resampler_depth=2,
                                                            resampler_heads=8)
    inspect.signature(CTReportGenerator.__init__).bind(None, None, visual_dim=1048, llm_dim=8, num_visual_tokens=512,
                                                       resampler_depth=2, resampler_heads=8, **base)

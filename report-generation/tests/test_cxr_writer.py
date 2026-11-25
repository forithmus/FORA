"""CPU smoke test of the X-ray writer wiring: plan builder + shared writer with X-ray view kinds on a tiny random
Gemma-4, one teacher-forced forward/backward and the generation prompt."""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("peft")
pytest.importorskip("transformers")

from fora.reportgen.cxr import model as M  # noqa: E402
from fora.reportgen.writer.llm import attach_text_lora, route_token_types_to_mm  # noqa: E402
from fora.reportgen.writer.model import generation_inputs  # noqa: E402

VOCAB = 512


def tiny_gemma4():
    from transformers import AutoModelForImageTextToText, Gemma4Config
    text = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=2,
                num_key_value_heads=1, num_global_key_value_heads=1, head_dim=32, global_head_dim=32,
                sliding_window=64, layer_types=["sliding_attention", "full_attention"], vocab_size=VOCAB,
                vocab_size_per_layer_input=VOCAB, hidden_size_per_layer_input=0, attention_k_eq_v=True,
                tie_word_embeddings=True)
    vision = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                  num_key_value_heads=2, head_dim=16, global_head_dim=16, patch_size=16, pooling_kernel_size=3,
                  position_embedding_size=1024)
    cfg = Gemma4Config(text_config=text, vision_config=vision, audio_config=None, image_token_id=500,
                       video_token_id=501, audio_token_id=502, boi_token_id=503, eoi_token_id=504,
                       boa_token_id=505, eoa_token_id=506)
    torch.manual_seed(0)
    model = AutoModelForImageTextToText.from_config(cfg)
    route_token_types_to_mm(model)
    return model


class Tok:
    unk_token_id = 3
    eos_token_id = 1

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[10 + (ord(c) % 400) for c in str(text)])

    def apply_chat_template(self, messages, **_):
        return {"input_ids": [10 + (ord(c) % 400) for c in messages[0]["content"]]}


def test_xray_writer_forward_and_generation_prompt(tmp_path):
    labels = [f"finding {i}" for i in range(6)]
    C = len(labels)
    cal = dict(labels=labels, canonical_plan_labels=labels,
               label_order_sha256=hashlib.sha256("\n".join(labels).encode()).hexdigest(),
               class_slope=[1.0] * C, class_intercept=[0.0] * C, class_plan_threshold=[0.5] * C,
               class_possible_threshold=[0.1] * C, class_val_positive=[10] * C, n_calibration_studies=100)
    (tmp_path / "cal.json").write_text(json.dumps(cal))
    tok = Tok()
    builder = M.make_plan_builder(tmp_path / "cal.json", labels, tok)
    llm = attach_text_lora(tiny_gemma4(), torch.device("cpu"), lora_r=4, lora_alpha=8, fimg_tokens=0)
    args = SimpleNamespace(finding_tokens=False, dino_resampler_dim=1024, dino_study_types=32, num_visual_tokens=8,
                           resampler_depth=2, resampler_heads=8)
    model = M.build_report_model(llm, tok, 64, labels, builder, torch.device("cpu"), args).float()
    assert model.dino_view_kind_embed.num_embeddings == 4 and model.dino_view_series_embed.num_embeddings == 32
    n = 300
    tokens = torch.randn(n, M.VISUAL_DIM)
    kind = torch.zeros(n, dtype=torch.long)
    series = torch.zeros(n, dtype=torch.long)
    logits = torch.tensor([[2.0, -1.0, -3.0, 0.5, -5.0, -0.5]])
    plan, features = M.plan_inputs(builder, logits, torch.randn(C, M.VISUAL_DIM), torch.device("cpu"), False, None)
    assert features is None and plan.ndim == 1 and plan.numel() > 10
    target = M.target_ids(tok, "Heart normal.", 2048, torch.device("cpu"))
    assert int(target[-1]) == 1 and target.numel() == len("Findings:\nHeart normal.") + 1
    study_types = torch.tensor([M.STUDY_TYPE_CXR])
    out = model(tokens, torch.empty(0, 1024), plan, target, torch.zeros(0, dtype=torch.long), logits, plan,
                builder.consistency_targets(torch.zeros(0, dtype=torch.long), torch.device("cpu")),
                finding_features=features, dino_view_kind=kind, dino_view_series=series, dino_study_types=study_types)
    out["loss"].backward()
    assert torch.isfinite(out["loss"])
    assert model.resampler.parameters().__next__().grad is not None
    bag = SimpleNamespace(tokens=tokens, view_kind=kind, view_series=series)
    with torch.no_grad():
        prefix = model.visual_prefix_from_bag(bag, study_types)
        emb, att, types = generation_inputs(model, prefix, logits, plan, features)
    assert emb.shape[1] == prefix.shape[1] + C + plan.numel() and int(types.sum()) == prefix.shape[1] + C


def test_stop_ids_and_scheduler(tmp_path):
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": [1, 106]}))

    class T:
        eos_token_id = 1
        unk_token_id = 3

        def convert_tokens_to_ids(self, t):
            return 107
    assert M.stop_ids(T(), tmp_path) == [1, 106, 107]
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sch = M.cosine_scheduler(opt, 100, 0.05)
    lrs = []
    for _ in range(100):
        lrs.append(opt.param_groups[0]["lr"]); opt.step(); sch.step()
    assert lrs[0] == pytest.approx(0.2) and lrs[4] == pytest.approx(1.0) and lrs[-1] < 0.01

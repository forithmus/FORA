"""CPU smoke tests of the report writer with a tiny random Gemma-4 (no pretrained weights, no tokenizer files)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")

from fora.reportgen.writer import dino_stream as D  # noqa: E402
from fora.reportgen.writer.conditioning import semantic_label_embeddings  # noqa: E402
from fora.reportgen.writer.llm import attach_text_lora, lora_target_modules, route_token_types_to_mm  # noqa: E402
from fora.reportgen.writer.model import generation_inputs, report_cross_entropy  # noqa: E402
from fora.reportgen.writer.resampler import QueryResampler  # noqa: E402

VOCAB = 512
FINDING_ID = 501     # stands for the single-token finding placeholder
IMAGE_ID = 502       # stands for the single-token evidence-image placeholder
TOKEN_DIM = 1048
N_CLASSES = 12


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
                       video_token_id=FINDING_ID, audio_token_id=IMAGE_ID, boi_token_id=503, eoi_token_id=504,
                       boa_token_id=505, eoa_token_id=506)
    torch.manual_seed(0)
    model = AutoModelForImageTextToText.from_config(cfg)
    route_token_types_to_mm(model)
    return model


class StubTokenizer:
    unk_token_id = 3

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[10 + (ord(c) % 400) for c in str(text)[:4]])


def build_writer(fimg_tokens: int = 0, consistency: bool = True):
    llm = attach_text_lora(tiny_gemma4(), torch.device("cpu"), lora_r=4, lora_alpha=8, fimg_tokens=fimg_tokens)
    labels = [f"finding {i}" for i in range(N_CLASSES)]
    semantics = semantic_label_embeddings(llm, StubTokenizer(), labels, torch.device("cpu"))
    torch.manual_seed(0)
    model = D.DinoCTReportGenerator(
        llm, visual_dim=TOKEN_DIM, llm_dim=64, num_visual_tokens=16, resampler_depth=2, resampler_heads=8,
        resampler_dim=1024, n_study_types=32, semantic_embeddings=semantics,
        calibration_slope=torch.ones(N_CLASSES), calibration_intercept=torch.zeros(N_CLASSES),
        diagnostic_consistency_classes=N_CLASSES if consistency else 0, diagnostic_consistency_weight=0.0,
        finding_feature_dim=1064, finding_placeholder_id=FINDING_ID, fimg_tokens=fimg_tokens,
        fimg_placeholder_id=IMAGE_ID,
    ).float()
    return model


def study(n_tokens=700, n_findings=2, fimg=0):
    torch.manual_seed(1)
    tokens = torch.randn(n_tokens, TOKEN_DIM)
    kind = torch.randint(0, D.N_VIEW_KINDS, (n_tokens,))
    series = torch.randint(0, 4, (n_tokens,))
    plan = [20, 21]
    for _ in range(n_findings):
        plan += [30, 31] + [FINDING_ID] * 3 + [IMAGE_ID] * fimg + [32]
    plan = torch.tensor(plan + [40])
    features = torch.randn(3 * n_findings, 1064)
    targets = torch.tensor([50, 51, 52, 53, 54, 1])
    logits = torch.randn(N_CLASSES) * 2 - 1
    consistency = torch.full((N_CLASSES,), -1.0)
    consistency[2], consistency[5] = 1.0, 0.0
    return tokens, kind, series, plan, features, targets, logits, consistency


def run(model, s, **kw):
    tokens, kind, series, plan, features, targets, logits, consistency = s
    return model(tokens, tokens.new_empty((0, 1024)), plan, targets, None, logits, plan, consistency,
                 finding_features=features, dino_view_kind=kind, dino_view_series=series,
                 dino_study_types=torch.tensor([5, 7]), **kw)


def test_lora_targets_language_model_only():
    model = tiny_gemma4()
    pattern = lora_target_modules(model)
    assert pattern.startswith("model[.]language_model[.]")
    for name in ("q_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        assert name in pattern
    llm = attach_text_lora(model, torch.device("cpu"), lora_r=4, lora_alpha=8)
    lora = [n for n, p in llm.named_parameters() if p.requires_grad]
    assert lora and all("lora_" in n and ".language_model." in n for n in lora)
    assert llm.get_base_model().model.vision_tower is None


def test_forward_backward_reaches_connector_and_lora():
    model = build_writer()
    model.train()
    out = run(model, study())
    assert torch.isfinite(out["loss"])
    assert torch.allclose(out["loss"].detach(), out["report_loss"])          # consistency weight 0
    out["loss"].backward()
    grads = {n: p.grad for n, p in model.named_parameters() if p.requires_grad}
    for name in ("resampler.input_projection.weight", "resampler.latents", "visual_projection.1.weight",
                 "dino_view_kind_embed.weight", "dino_study_type_embed.weight", "image_start",
                 "finding_token_projection.projection.1.weight", "label_conditioner.probability_projection.0.weight"):
        assert grads[name] is not None and float(grads[name].abs().sum()) > 0, name
    assert any(g is not None and float(g.abs().sum()) > 0 for n, g in grads.items() if "lora_B" in n)
    assert tuple(model.resampler.input_projection.weight.shape) == (1024, TOKEN_DIM)


def test_prefix_layout_and_zero_view_embedding():
    model = build_writer(consistency=False)
    tokens = torch.randn(300, TOKEN_DIM)
    visual, modality = model.dino_inputs(tokens, torch.randint(0, 4, (300,)), torch.zeros(300, dtype=torch.long),
                                         torch.tensor([1, 2, 3]))
    assert torch.equal(visual, tokens)
    assert tuple(modality.shape) == (3, 1024)
    with torch.no_grad():
        prefix = model._visual_prefix(visual, modality)
    assert prefix.shape[1] == 1 + 16 + 3 + 1
    with pytest.raises(ValueError):
        model.dino_inputs(tokens[:, :1024], torch.zeros(300, dtype=torch.long), torch.zeros(300, dtype=torch.long),
                          torch.tensor([1]))


def test_generation_inputs_match_training_prefix():
    model = build_writer(consistency=False)
    model.eval()
    s = study()
    tokens, kind, series, plan, features, targets, logits, _ = s
    with torch.no_grad():
        out = run(model, s)
        bag = SimpleNamespace(tokens=tokens, view_kind=kind, view_series=series)
        prefix = model.visual_prefix_from_bag(bag, torch.tensor([5, 7]))
    embeddings, attention, types = generation_inputs(model, prefix, logits, plan, features)
    assert embeddings.shape[1] == int(out["prefix_length"])
    assert int(types.sum()) == prefix.shape[1] + N_CLASSES
    assert attention.shape == types.shape


def test_placeholder_count_mismatch_raises():
    model = build_writer(consistency=False)
    s = list(study())
    s[4] = s[4][:-1]
    with pytest.raises(ValueError):
        run(model, s)


def test_evidence_images_through_vision_tower():
    import numpy as np
    from transformers.models.gemma4.image_processing_gemma4 import Gemma4ImageProcessor
    model = build_writer(fimg_tokens=6, consistency=False)
    model.train()
    assert any(".vision_tower." in n and p.requires_grad for n, p in model.named_parameters())
    proc = Gemma4ImageProcessor(max_soft_tokens=70)
    rng = np.random.default_rng(0)
    images = [(rng.random((48, 48, 3)) * 255).astype(np.uint8) for _ in range(5)]
    batch = proc(images=images, return_tensors="pt")
    vmask = torch.zeros(2, 6, dtype=torch.bool)
    vmask[0, :3] = True
    vmask[1, :2] = True
    model.set_finding_images(batch["pixel_values"].float(), batch["image_position_ids"], vmask, 2)
    out = run(model, study(fimg=6))
    out["loss"].backward()
    pool_grad = model.fimg_pool.query.grad
    assert pool_grad is not None and float(pool_grad.abs().sum()) > 0


def test_chunked_cross_entropy_equals_plain():
    logits = torch.randn(1, 150, 40, requires_grad=True)
    targets = torch.randint(0, 40, (150,))
    loss = report_cross_entropy(logits, targets, 64)
    ref = torch.nn.functional.cross_entropy(logits[0], targets)
    assert torch.allclose(loss, ref, atol=1e-6)
    loss.backward()
    assert logits.grad is not None


def test_streaming_cross_attention_is_exact():
    torch.manual_seed(0)
    fused = QueryResampler(input_dim=32, dim=32, num_queries=4, depth=1, heads=4, stream_threshold=0, kv_chunk=0)
    streamed = QueryResampler(input_dim=32, dim=32, num_queries=4, depth=1, heads=4, stream_threshold=10,
                              stream_chunk=7)
    streamed.load_state_dict(fused.state_dict())
    tokens = torch.randn(50, 32)
    with torch.no_grad():
        assert torch.allclose(fused(tokens), streamed(tokens), atol=1e-5)

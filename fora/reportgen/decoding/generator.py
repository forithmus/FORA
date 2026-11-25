"""Build the trained CT writer and decode one report (paper section "Grounded report generation").

The prompt is

    [512 visual latents + study-type tokens] [frozen-Platt label tokens] [diagnostic plan text with evidence focus
    tokens and zoomed evidence images inserted at their placeholders] (+ forced opener)

and `decode` runs one `generate` call on it. Every setting of the model (LoRA rank, resampler, study types, finding
feature width) comes from the checkpoint metadata, exactly as trained.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .opener import append_opener, prepend_opener
from .sampling import SamplingConfig

# Plan and evidence-image settings of the paper writer: possible tier capped at 20 items ranked by p, probabilities
# shown in the plan, 6 zoomed evidence images per finding (6 views), 140 Gemma-4 soft tokens per image.
FINAL_SWITCHES = {"plan_possible_cap": 20, "plan_cap_rank": "prob", "plan_prob": True,
                  "fimg_tokens": 6, "fimg_views": 6, "fimg_soft": 140}
DEFAULT_FALLBACK_CAPS = (2097152, 1048576, 524288)
FIMG_PLACEHOLDER = "<|audio|>"


def build_generator(checkpoint_path, llm_path, labels, device, calibration_override="", *, visual_dim=1048,
                    dtype=torch.bfloat16, package=None, switches=None):
    """The trained DinoCTReportGenerator (all model settings from the checkpoint metadata).

    Returns (model, tokenizer, optimizer_step, plan_builder, metadata, train_args, calibration_path)."""
    from fora.reportgen.plan import diagnostic_plan
    from fora.reportgen.plan.diagnostic_plan import DiagnosticPlanBuilder
    from fora.reportgen.writer.conditioning import semantic_label_embeddings
    from fora.reportgen.writer.dino_stream import DinoCTReportGenerator, is_dino_checkpoint
    from fora.reportgen.writer.llm import build_gemma_lora

    sw = dict(FINAL_SWITCHES, **(switches or {}))
    if package is None:
        package = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = package.get("metadata", {})
    train_args = metadata.get("arguments", {})
    if not is_dino_checkpoint(package):
        raise RuntimeError(f"{checkpoint_path}: not a DINO block-24 writer checkpoint")
    llm, tokenizer, llm_dim = build_gemma_lora(
        llm_path, device, lora_r=int(train_args.get("lora_r", 16)), lora_alpha=int(train_args.get("lora_alpha", 32)),
        fimg_tokens=int(sw["fimg_tokens"]))
    finding_placeholder = diagnostic_plan.set_finding_placeholder(llm.fora_llm_family)
    calibration_path = calibration_override or metadata.get("mil_calibration")
    plan_builder = DiagnosticPlanBuilder(
        calibration_path, labels, tokenizer, "all_classes", dataset_kind="ct",
        plan_possible_cap=int(sw["plan_possible_cap"]), plan_cap_rank=str(sw["plan_cap_rank"]),
        plan_prob=bool(sw["plan_prob"]), fimg_tokens=int(sw["fimg_tokens"]))
    semantics = semantic_label_embeddings(llm, tokenizer, labels, device)
    finding_dim = int(metadata.get("finding_feature_dim") or 0)
    placeholder = int(tokenizer.convert_tokens_to_ids(finding_placeholder)) if finding_dim else -1
    model = DinoCTReportGenerator(
        llm, visual_dim=int(visual_dim), llm_dim=llm_dim,
        resampler_dim=int(train_args.get("dino_resampler_dim", 1024)),
        n_study_types=int(train_args.get("dino_study_types", 32)),
        num_visual_tokens=int(train_args.get("num_visual_tokens", 512)),
        resampler_depth=int(train_args.get("resampler_depth", 2)), resampler_heads=int(train_args.get("resampler_heads", 8)),
        semantic_embeddings=semantics, eligible_class_indices=torch.arange(len(labels), device=device),
        calibration_slope=plan_builder.slope.to(device), calibration_intercept=plan_builder.intercept.to(device),
        diagnostic_consistency_classes=0, diagnostic_consistency_weight=0.0,
        finding_feature_dim=finding_dim, finding_placeholder_id=placeholder,
        fimg_tokens=int(sw["fimg_tokens"]), fimg_views=int(sw["fimg_views"]),
        fimg_placeholder_id=int(tokenizer.convert_tokens_to_ids(FIMG_PLACEHOLDER)),
    ).to(device=device, dtype=dtype)
    load_trained_tensors(model, package["trainable_model"])
    model.eval()
    model.llm.config.use_cache = True
    return (model, tokenizer, int(package.get("optimizer_step", -1)), plan_builder, metadata, train_args, calibration_path)


def load_trained_tensors(model, trainable: dict) -> None:
    """Copy the checkpoint's trainable tensors into the model; every trainable parameter must be present."""
    known = dict(model.named_parameters())
    # The full writer trains a diagnostic-consistency head (loss weight 0) that generation never uses.
    trainable = {k: v for k, v in trainable.items() if k in known or not k.startswith("diagnostic_consistency_head.")}
    unknown = sorted(set(trainable) - set(known))
    if unknown:
        raise RuntimeError(f"checkpoint has unknown trainable keys: {unknown[:10]}")
    expected = {n for n, p in model.named_parameters() if p.requires_grad}
    absent = sorted(expected - set(trainable))
    if absent:
        raise RuntimeError(f"checkpoint misses trainable tensors (would stay random): {absent[:10]}")
    for name, value in trainable.items():
        if tuple(known[name].shape) != tuple(value.shape):
            raise RuntimeError(f"shape mismatch for {name}: ckpt{tuple(value.shape)} model{tuple(known[name].shape)}")
        known[name].data.copy_(value.to(device=known[name].device, dtype=known[name].dtype))


def generation_stop_ids(tokenizer, llm_path) -> list[int]:
    """Gemma-4 stop ids: tokenizer EOS + the snapshot's generation_config.json eos_token_id list
    ([1 <eos>, 106 <turn|>, 50 <|tool_response>]) + <turn|> (also if the config file is absent)."""
    eos = tokenizer.eos_token_id
    ids = [int(e) for e in (eos if isinstance(eos, (list, tuple)) else [eos]) if e is not None]
    config = Path(llm_path) / "generation_config.json"
    if config.is_file():
        extra = json.loads(config.read_text()).get("eos_token_id")
        ids += [int(e) for e in (extra if isinstance(extra, list) else [extra]) if e is not None]
    turn = tokenizer.convert_tokens_to_ids("<turn|>")
    if isinstance(turn, int) and turn != tokenizer.unk_token_id:
        ids.append(int(turn))
    out = list(dict.fromkeys(ids))
    if not out:
        raise RuntimeError("no stop token ids")
    return out


@torch.inference_mode()
def generation_inputs(model, visual_prefix, logits, plan_ids, finding_features):
    """Visual prefix + label tokens + plan prompt with finding tokens and images (the training layout)."""
    from fora.reportgen.writer.model import generation_inputs as inputs
    return inputs(model, visual_prefix, logits, plan_ids, finding_features)


@torch.inference_mode()
def dino_prefix(model, bag, study_types):
    from fora.reportgen.writer.model import amp
    with amp(bag.tokens.device):
        return model.visual_prefix_from_bag(bag, study_types.to(bag.tokens.device))


def caps_to_try(cap: int, fallback) -> list[int]:
    """Requested block-24 token cap first (0 = all tokens), then strictly smaller fallback caps (after a CUDA OOM)."""
    out = [int(cap)]
    for value in fallback:
        value = int(value)
        if value > 0 and (int(cap) == 0 or value < int(cap)) and value not in out:
            out.append(value)
    return out


@torch.inference_mode()
def decode(model, embeddings, attention, token_types, *, max_new_tokens: int, stop_ids, uid: str,
           sampling: SamplingConfig, opener_ids=None, bad_words_ids=None) -> torch.Tensor:
    """One generate() call; returns ids of the report (opener + new tokens).

    Order of operations: optional opener embeddings appended (max_new_tokens reduced by its
    length), then torch.manual_seed(per-study seed) immediately before generate(), then the opener ids prepended.
    `logits_to_keep=1` is passed explicitly because the wrapped Gemma-4 forward has a (*args, **kwargs) signature."""
    table = model.llm.get_input_embeddings()
    embeddings = embeddings.to(dtype=table.weight.dtype)
    maximum = int(max_new_tokens)
    if opener_ids:
        embeddings, attention, token_types = append_opener(embeddings, attention, token_types, opener_ids, table)
        maximum -= len(opener_ids)
    kwargs = dict(inputs_embeds=embeddings, attention_mask=attention, token_type_ids=token_types, max_new_tokens=maximum,
                  use_cache=True, eos_token_id=list(stop_ids), pad_token_id=int(stop_ids[0]), logits_to_keep=1,
                  **sampling.generate_kwargs())
    if bad_words_ids is not None:
        kwargs["bad_words_ids"] = bad_words_ids
    seed = sampling.seed_for(uid)
    if seed is not None:
        torch.manual_seed(seed)
    ids = model.llm.generate(**kwargs)
    if opener_ids:
        ids = prepend_opener(ids, opener_ids)
    return ids


def set_finding_images(model, finding_store, rows, order, fimg_soft: int) -> int:
    """Zoomed evidence images of the planned findings -> model (Gemma-4 image processor, `fimg_soft` soft tokens).
    Returns the number of non-empty images."""
    if getattr(model, "fimg_pool", None) is None:
        return 0
    from fora.reportgen.evidence.zoom_images import fimg_batch, make_fimg_processor
    if not hasattr(model, "_fimg_proc"):
        model._fimg_proc = make_fimg_processor(int(fimg_soft))
    images = finding_store.images([rows[c] for c in order])
    model.set_finding_images(*fimg_batch(model._fimg_proc, images), len(order))
    return sum(1 for views in images for b in views if b) if order else 0

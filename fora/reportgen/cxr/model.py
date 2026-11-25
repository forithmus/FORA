"""Chest X-ray writer model (paper section "Grounded report generation", X-ray arm).

The shared writer (``fora.reportgen.writer.dino_stream.DinoCTReportGenerator``: Gemma-4-31B-it, LoRA r16/a32 on the
language model's attention and MLP projections, frozen-Platt label tokens, plan text with one evidence token per
planned finding) with its encoder-token prefix: QueryResampler 1048 -> 1024 (depth 2, 8 heads) -> 512 latents -> LLM width,
zero-initialised per-token view embeddings (film view kind 0 frontal / 1 lateral / 2 other, and film ordinal in the
study) and a trainable study-type token (one type: chest radiograph). X-ray writers carry no CT location descriptions,
measurement line or zoomed evidence images (``fimg_tokens=0``).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from fora.reportgen.cxr.stream import MAX_SERIES, N_VIEW_KINDS

STUDY_TYPE_CXR = 0
VISUAL_DIM = 1048
TARGET_PREFIX = 'Findings:\n'


def _plan():
    from fora.reportgen.plan import diagnostic_plan
    return diagnostic_plan


def make_plan_builder(calibration: str, labels, tokenizer, builder_class=None):
    """X-ray plan builder: all_classes mode, chest-radiograph wording, calibrated probability + base-rate text
    no possible-tier cap, no CT descriptions or finding images."""
    builder_class = builder_class or _plan().DiagnosticPlanBuilder
    return builder_class(calibration, labels, tokenizer, 'all_classes', dataset_kind='cxr', plan_prob=True,
                         plan_possible_cap=0, fimg_tokens=0, evidence_descriptions=False)


def load_llm(llm_path: str, device, lora_r: int, lora_alpha: int):
    """Gemma-4-31B-it + text-only LoRA (vision tower dropped: no finding images on X-ray)."""
    from fora.reportgen.writer.llm import build_gemma_lora
    llm, tokenizer, llm_dim = build_gemma_lora(llm_path, device, lora_r=lora_r, lora_alpha=lora_alpha, fimg_tokens=0)
    _plan().set_finding_placeholder(llm.fora_llm_family)
    return llm, tokenizer, llm_dim


def build_report_model(llm, tokenizer, llm_dim, labels, plan_builder, device, args):
    """The shared DINO-prefix writer with X-ray view kinds (frontal / lateral / other) and film ordinals, every
    auxiliary loss off and frozen-Platt label conditioning."""
    from fora.reportgen.writer.conditioning import semantic_label_embeddings
    from fora.reportgen.writer.dino_stream import DinoCTReportGenerator
    DP = _plan()
    semantics = semantic_label_embeddings(llm, tokenizer, labels, device)
    placeholder = -1
    if args.finding_tokens:
        placeholder = int(tokenizer.convert_tokens_to_ids(DP.FINDING_PLACEHOLDER))
        probe = tokenizer(DP.FINDING_PLACEHOLDER, add_special_tokens=False).input_ids
        if probe != [placeholder] or placeholder in (tokenizer.unk_token_id, None):
            raise RuntimeError(f'{DP.FINDING_PLACEHOLDER} is not a single reserved token: {probe}')
    C = len(labels)
    return DinoCTReportGenerator(
        llm, resampler_dim=args.dino_resampler_dim, n_study_types=args.dino_study_types, visual_dim=VISUAL_DIM,
        llm_dim=llm_dim, num_visual_tokens=args.num_visual_tokens, resampler_depth=args.resampler_depth,
        resampler_heads=args.resampler_heads, n_view_kinds=N_VIEW_KINDS, max_series=MAX_SERIES,
        semantic_embeddings=semantics, eligible_class_indices=torch.arange(C, device=device),
        calibration_slope=plan_builder.slope.to(device), calibration_intercept=plan_builder.intercept.to(device),
        diagnostic_consistency_classes=C, diagnostic_consistency_weight=0.0,
        finding_feature_dim=VISUAL_DIM if args.finding_tokens else 0, finding_placeholder_id=placeholder,
        fimg_tokens=0,
    ).to(device, dtype=torch.bfloat16)


def load_trainable(model, trainable: dict, device) -> None:
    """Copy a writer checkpoint's trainable tensors into the model; every trainable parameter must be present."""
    known = dict(model.named_parameters())
    unknown = sorted(set(trainable) - set(known))
    absent = sorted({n for n, p in known.items() if p.requires_grad} - set(trainable))
    if unknown or absent:
        raise RuntimeError(f'checkpoint/model mismatch unknown={unknown[:5]} absent={absent[:5]}')
    with torch.no_grad():
        for name, value in trainable.items():
            known[name].data.copy_(value.to(device=known[name].device, dtype=known[name].dtype))


def build_writer(checkpoint, llm_path: str, labels, device, calibration: str, builder_class=None):
    """The trained X-ray writer for generation (LoRA attached, not merged; every model setting from the checkpoint
    metadata) -> (model, tokenizer, optimizer step, training arguments, plan builder)."""
    package = torch.load(checkpoint, map_location='cpu', weights_only=False)
    meta = package['metadata']
    targs = argparse.Namespace(**meta['arguments'])
    if not meta.get('plan_probability', True):
        raise RuntimeError('the X-ray writer is trained with probability plan text')
    llm, tokenizer, llm_dim = load_llm(str(llm_path), device, targs.lora_r, targs.lora_alpha)
    plan_builder = make_plan_builder(str(calibration), labels, tokenizer, builder_class)
    model = build_report_model(llm, tokenizer, llm_dim, labels, plan_builder, device, targs)
    load_trainable(model, package['trainable_model'], device)
    step = int(package.get('optimizer_step', -1))
    del package
    model.eval()
    model.llm.config.use_cache = True
    return model, tokenizer, step, targs, plan_builder


def plan_inputs(plan_builder, logits, pooled, device, finding_tokens: bool, locations=None):
    """Plan ids (full writer, probability-style plan text) + the pooled finding features in placeholder order."""
    C = logits.shape[-1]
    token_classes = list(range(C)) if finding_tokens else None
    plan = plan_builder.plan_ids(logits, device, location_descriptions=locations or None, report_writer_mode='full',
                                 finding_token_classes=token_classes)
    features = None
    if finding_tokens:
        order = plan_builder.planned_finding_order(logits, token_classes)
        features = pooled[torch.as_tensor(order, dtype=torch.long, device=pooled.device)] if order else pooled[:0]
    return plan, features


def target_ids(tokenizer, text: str, maximum: int, device):
    """'Findings:\\n' + text + EOS, truncated to `maximum` tokens."""
    ids = tokenizer(TARGET_PREFIX + text, add_special_tokens=False).input_ids
    eos = tokenizer.eos_token_id
    if isinstance(eos, (list, tuple)):
        eos = eos[0]
    ids = ids[: max(1, maximum - 1)]
    if not ids or ids[-1] != eos:
        ids.append(int(eos))
    return torch.tensor(ids, dtype=torch.long, device=device)


def cosine_scheduler(optimizer, total_steps: int, warmup_fraction: float):
    """Linear warm-up then cosine decay to 0."""
    warmup = int(total_steps * warmup_fraction)
    if not 0 <= warmup < total_steps:
        raise ValueError(f'warmup steps must be in [0, {total_steps}), got {warmup}')

    def factor(step):
        if warmup and step < warmup:
            return float(step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def stop_ids(tokenizer, llm_path):
    """Gemma-4 stop ids: tokenizer EOS + generation_config eos list + <turn|>."""
    eos = tokenizer.eos_token_id
    ids = [int(e) for e in (eos if isinstance(eos, (list, tuple)) else [eos]) if e is not None]
    config = Path(llm_path) / 'generation_config.json'
    if config.is_file():
        extra = json.loads(config.read_text()).get('eos_token_id')
        ids += [int(e) for e in (extra if isinstance(extra, list) else [extra]) if e is not None]
    turn = tokenizer.convert_tokens_to_ids('<turn|>')
    if isinstance(turn, int) and turn != tokenizer.unk_token_id:
        ids.append(int(turn))
    return list(dict.fromkeys(ids))


def split_parameters(model):
    """Trainable parameters: connector (resampler, projections, label conditioner, finding-token projection) and the
    LoRA adapters of the language model."""
    connector, adapters = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (adapters if name.startswith('llm.') else connector).append(parameter)
    return connector, adapters

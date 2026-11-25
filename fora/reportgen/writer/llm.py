"""Gemma-4-31B-it language model with LoRA for the report writer (Methods, "Report generation").

LoRA (rank 16, alpha 32) is attached to every attention and MLP projection of the language model
(q/k/v/o, gate/up/down); embeddings, LM head and base weights stay frozen. The native audio tower is removed.
The vision tower and its projector are kept and fully trainable only when zoomed evidence images are used
(``fimg_tokens > 0``); otherwise they are removed as well. Gradient checkpointing is enabled.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

LORA_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
LLM_FAMILY = "gemma4"
LORA_DROPOUT = 0.05   # configured value; train_mode_for_checkpointing sets it to 0 for training (writer.fsdp)


def check_gemma4(model_path: str) -> None:
    config = json.loads((Path(model_path) / "config.json").read_text())
    if not str(config.get("model_type", "")).startswith("gemma4"):
        raise ValueError(f"expected a Gemma-4 snapshot, got model_type={config.get('model_type')!r} in {model_path}")


def llm_family_id(model_path: str, family: str = LLM_FAMILY) -> str:
    """Stable identity of the LLM for resume fingerprints."""
    config = json.loads((Path(model_path) / "config.json").read_text())
    text = config.get("text_config", config)
    return f"google/{config.get('model_type')}:{text.get('hidden_size')}x{text.get('num_hidden_layers')}"


def language_layers(llm: nn.Module):
    """Decoder blocks of the (PEFT-wrapped) Gemma language model."""
    root = llm.get_base_model() if hasattr(llm, "get_base_model") else llm
    inner = getattr(root, "model", root)
    language = getattr(inner, "language_model", inner)
    layers = getattr(language, "layers", None)
    if layers is None:
        raise RuntimeError("cannot locate the language-model decoder layers")
    return layers


def route_token_types_to_mm(model: nn.Module) -> None:
    """Pass ``token_type_ids`` (1 = bidirectional visual/label prefix) to Gemma-4 as ``mm_token_type_ids``.

    If the caller sets ``mm_token_type_ids`` explicitly (also to None, as cached decoding steps do), it is kept.
    """
    base_forward = model.forward

    def forward(*args, token_type_ids=None, **kwargs):
        if token_type_ids is not None and "mm_token_type_ids" not in kwargs:
            kwargs["mm_token_type_ids"] = token_type_ids
        return base_forward(*args, **kwargs)

    model.forward = forward


def lora_target_modules(model: nn.Module) -> str:
    """Regex over the language model's attention / MLP projections that exist in this model."""
    prefix = "model.language_model."
    present = sorted({
        name.rsplit(".", 1)[-1]
        for name, module in model.named_modules()
        if name.startswith(prefix) and isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in LORA_PROJECTIONS
    })
    if not present:
        raise RuntimeError("no LoRA projection found in the language model")
    return "model[.]language_model[.].*[.](" + "|".join(present) + ")"


def attach_text_lora(model: nn.Module, device, *, lora_r: int = 16, lora_alpha: int = 32, fimg_tokens: int = 0):
    """Freeze ``model`` (a Gemma-4 ImageTextToText model), drop unused towers and attach the text LoRA."""
    from peft import LoraConfig, TaskType, get_peft_model

    keep_vision = int(fimg_tokens) > 0
    for attribute in ("vision_tower", "multi_modal_projector", "audio_tower", "embed_vision", "embed_audio"):
        if keep_vision and attribute in ("vision_tower", "embed_vision"):
            continue
        if getattr(model.model, attribute, None) is not None:
            setattr(model.model, attribute, None)
    model.requires_grad_(False)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = LoraConfig(r=int(lora_r), lora_alpha=int(lora_alpha), lora_dropout=LORA_DROPOUT,
                      target_modules=lora_target_modules(model), bias="none", task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora, adapter_name="default")
    model = model.to(device)
    if keep_vision:
        for name, parameter in model.named_parameters():
            if ".vision_tower." in name or ".embed_vision." in name:
                parameter.requires_grad_(True)
    model.fora_llm_family = LLM_FAMILY
    return model


def build_gemma_lora(model_path: str, device: torch.device, *, lora_r: int = 16, lora_alpha: int = 32,
                     fimg_tokens: int = 0):
    """Load Gemma-4 from a local snapshot and attach the text LoRA -> (model, tokenizer, hidden size)."""
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    check_gemma4(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa", low_cpu_mem_usage=True)
    route_token_types_to_mm(model)
    model = attach_text_lora(model, device, lora_r=lora_r, lora_alpha=lora_alpha, fimg_tokens=fimg_tokens)
    config = model.config
    text_config = config.get_text_config() if hasattr(config, "get_text_config") else config.text_config
    return model, tokenizer, int(text_config.hidden_size)


def activate_lora_adapter(model: nn.Module, adapter_name: str = "default") -> None:
    """Select a PEFT adapter and keep every LoRA tensor trainable."""
    model.set_adapter(str(adapter_name))
    for name, parameter in model.named_parameters():
        if "lora_" in name:
            parameter.requires_grad_(True)

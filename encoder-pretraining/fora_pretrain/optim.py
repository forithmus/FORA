"""Optimizer of the encoder pre-training (Methods, "Image encoder pre-training").

DINOv2 parameter groups: layer-wise learning-rate decay (block i of n gets decay^(n + 1 - (i + 1)), patch embedding and
tokens decay^(n + 1), the final norms and both heads 1), patch embedding learning rate x 0.2, no weight decay on biases,
1-D parameters and tokens. The prototype layers of both heads are frozen during the first steps. The trainer sets one
scheduled lr / weight decay per step; ``GroupedAdamW`` applies the per-group multipliers inside ``step``.
"""
from __future__ import annotations

import json
import math
import os

import torch

TOKEN_NAMES = ("cls_token", "mask_token", "storage_tokens")


def _block_index(name: str):
    parts = name.split(".")
    for i, part in enumerate(parts[:-1]):
        if part == "blocks" and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


def build_param_groups(student: torch.nn.Module, *, layerwise_decay: float, patch_embed_lr_mult: float, n_blocks: int):
    """Groups of (lr_mult, wd_mult) for the student network (backbone + two heads); equal multipliers are merged."""
    if not 0.0 < layerwise_decay <= 1.0 or patch_embed_lr_mult <= 0 or n_blocks < 1:
        raise ValueError("Require 0 < layerwise_decay <= 1, patch_embed_lr_mult > 0 and n_blocks >= 1")
    merged: dict[tuple[float, float], dict] = {}
    for name, param in student.named_parameters():
        if not param.requires_grad:
            continue
        lr_mult, wd_mult = 1.0, 1.0
        if name.startswith("backbone."):
            block = _block_index(name)
            if block is not None:
                layer_id = block + 1
            elif "patch_embed" in name or any(tok in name for tok in TOKEN_NAMES) or "rope" in name:
                layer_id = 0
            else:
                layer_id = n_blocks + 1
            lr_mult = float(layerwise_decay) ** (n_blocks + 1 - layer_id)
            if "patch_embed" in name:
                lr_mult *= float(patch_embed_lr_mult)
        if param.ndim <= 1 or name.endswith(".bias") or any(tok in name for tok in TOKEN_NAMES):
            wd_mult = 0.0
        key = (round(lr_mult, 12), wd_mult)
        group = merged.setdefault(key, dict(params=[], lr_mult=float(lr_mult), wd_mult=float(wd_mult), names=[]))
        group["params"].append(param)
        group["names"].append(name)
    groups = [merged[k] for k in sorted(merged, key=lambda k: (-k[0], -k[1]))]
    if os.environ.get("RANK", "0") == "0":
        print(json.dumps(dict(event="param_groups", groups=[dict(lr_mult=g["lr_mult"], wd_mult=g["wd_mult"],
                                                                 n=len(g["params"]), example=g["names"][0]) for g in groups])),
              flush=True)
    return groups


class GroupedAdamW(torch.optim.AdamW):
    """AdamW whose ``step`` scales each group's scheduled lr / weight decay by its ``lr_mult`` / ``wd_mult``."""

    @torch.no_grad()
    def step(self, closure=None):
        saved = [(group["lr"], group["weight_decay"]) for group in self.param_groups]
        for group in self.param_groups:
            group["lr"] = group["lr"] * float(group.get("lr_mult", 1.0))
            group["weight_decay"] = group["weight_decay"] * float(group.get("wd_mult", 1.0))
        try:
            return super().step(closure)
        finally:
            for group, (lr, wd) in zip(self.param_groups, saved):
                group["lr"], group["weight_decay"] = lr, wd


def build_optimizer(learner, args) -> GroupedAdamW:
    groups = build_param_groups(learner.student, layerwise_decay=args.layerwise_decay,
                                patch_embed_lr_mult=args.patch_embed_lr_mult, n_blocks=len(learner.student.backbone.blocks))
    return GroupedAdamW(groups, lr=args.lr, betas=(args.adamw_beta1, args.adamw_beta2), weight_decay=args.weight_decay,
                        fused=torch.cuda.is_available())


def cancel_last_layer_gradients(student: torch.nn.Module) -> int:
    """Drop the prototype-layer gradients of both heads (``freeze_last_layer`` of DINO / DINOv2)."""
    count = 0
    for name, param in student.named_parameters():
        if "last_layer" in name and param.grad is not None:
            param.grad = None
            count += 1
    return count


def cosine(start: float, end: float, step: int, total: int) -> float:
    if total <= 1:
        return end
    x = min(max(step / (total - 1), 0.0), 1.0)
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * x))


def schedules(args, step: int, peak_lr: float) -> dict:
    """lr (linear warm-up, then cosine to min_lr), weight decay and EMA momentum (cosine over ``args.steps``), and the
    teacher temperature (linear warm-up, then constant)."""
    if step < args.warmup_steps:
        lr = peak_lr * min(1.0, (step + 1) / max(1, args.warmup_steps))
    else:
        lr = cosine(peak_lr, args.min_lr, step - args.warmup_steps, args.steps - args.warmup_steps)
    if step >= args.teacher_warmup_steps:
        temperature = float(args.teacher_temperature)
    else:
        alpha = step / max(1, args.teacher_warmup_steps)
        temperature = args.teacher_warmup_temperature + alpha * (args.teacher_temperature - args.teacher_warmup_temperature)
    return dict(lr=lr, weight_decay=cosine(args.weight_decay, args.weight_decay_end, step, args.steps),
                teacher_momentum=cosine(args.teacher_momentum, 1.0, step, args.steps), teacher_temperature=temperature)

"""Offline DPO under FSDP2/HSDP for the report writer (Methods, "Report generation", preference optimisation).

Objective (same as ``fora_report.ct.dpo_loss.dpo_study_backward``):

    paired study (K >= 1 pairs):  dpo_weight * mean_k[-log sigmoid(beta * ((pc - pr_k) - (rc - rr_k)))]
                                  + chosen_ce_weight * token-mean CE(chosen)
    unpaired study:               unpaired_ce_weight * token-mean CE(chosen)

with summed log-probabilities p (policy) and r (frozen reference), and the exact sequential-graph chain rule
(rejected policy scores first without grad, the chosen graph with them as constants, then every rejected graph
with the detached chosen score).

FSDP specifics:

* ``FSDPReferenceWeights``: the frozen reference trainable tensors are stored as LOCAL SHARDS laid out exactly like
  each FSDP parameter's DTensor (``distribute_tensor(..., src_data_rank=None)``, no communication). ``swapped()``
  reshards every FSDP unit, copies the reference into ``param.to_local()`` under ``no_grad``, and on exit reshards
  again and copies the policy shard back (bit-exact ``copy_``).
* every FSDP forward/backward is a collective of the node's shard group, so all ranks of a shard group run the same
  schedule per micro-step: ``slots = max K over the group`` (``agree_slots``); if ``slots == 0`` the caller runs a
  plain CE micro-step; otherwise every rank runs (1 + slots) reference no-grad forwards, slots policy no-grad
  forwards, 1 chosen grad pass and slots rejected grad passes (the last one gradient-synchronised at a boundary).
  A rank with K < slots fills the missing slots with dummy passes on a short prefix of its chosen target (no-grad:
  result dropped; grad: ``0 * score`` backward). A rank with K = 0 in a paired group puts the unpaired CE on its
  chosen pass.

FSDP's reduce-scatter (shard) + all-reduce (replicate) average the gradient over the world.
"""
from __future__ import annotations

import math
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch.nn import functional as F

from .fsdp import load_full_trainable_state

DUMMY_TARGET_TOKENS = 16


def _is_dtensor(value) -> bool:
    from torch.distributed.tensor import DTensor
    return isinstance(value, DTensor)


def fsdp_modules(model: torch.nn.Module) -> list:
    from torch.distributed.fsdp import FSDPModule
    return [m for m in model.modules() if isinstance(m, FSDPModule)]


def reshard_all(model_or_modules) -> None:
    """Free every unsharded FSDP parameter (local, no collective); registers the sharded DTensors again."""
    modules = model_or_modules if isinstance(model_or_modules, (list, tuple)) else fsdp_modules(model_or_modules)
    for module in modules:
        module.reshard()


def _local(parameter: torch.Tensor) -> torch.Tensor:
    return parameter.to_local() if _is_dtensor(parameter) else parameter.data


class FSDPReferenceWeights:
    """Frozen reference trainable tensors as local shards of each FSDP parameter; swapped in for reference scores.

    Interface of ``fora_report.ct.dpo_loss.ReferenceWeights`` (items, numel, max_abs_difference, swapped).
    ``max_abs_difference`` is a collective (MAX over ``group``, default world) so every rank decides alike.
    """

    def __init__(self, model: torch.nn.Module, state: dict, device, group=None):
        self.modules = fsdp_modules(model)
        reshard_all(self.modules)          # named_parameters() must return the sharded DTensors
        self.group = group
        parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
        missing = sorted(set(parameters) - set(state))
        unknown = sorted(set(state) - set(parameters))
        if missing or unknown:
            raise RuntimeError(
                f"DPO reference must cover exactly the trainable tensors: missing={missing[:5]} "
                f"unknown={unknown[:5]} (n_missing={len(missing)} n_unknown={len(unknown)})")
        self.items = []
        numel = local_numel = 0
        with torch.no_grad():
            for name, parameter in parameters.items():
                value = state[name]
                if tuple(value.shape) != tuple(parameter.shape):
                    raise RuntimeError(f"DPO reference shape mismatch for {name}")
                local = _local(parameter)
                if _is_dtensor(parameter):
                    from torch.distributed.tensor import distribute_tensor
                    reference = distribute_tensor(
                        value.to(device=local.device, dtype=parameter.dtype), parameter.device_mesh,
                        parameter.placements, src_data_rank=None).to_local()
                else:
                    reference = value.to(device=local.device, dtype=parameter.dtype)
                reference = reference.contiguous().clone()
                if tuple(reference.shape) != tuple(local.shape):
                    raise RuntimeError(f"{name}: reference shard {tuple(reference.shape)} != "
                                       f"local shard {tuple(local.shape)}")
                self.items.append((parameter, reference, torch.empty_like(local)))
                numel += parameter.numel()
                local_numel += local.numel()
        self.numel = numel
        self.local_numel = local_numel

    @torch.no_grad()
    def max_abs_difference(self) -> float:
        value = torch.zeros((), dtype=torch.float64,
                            device=self.items[0][1].device if self.items else torch.device("cpu"))
        for parameter, reference, _ in self.items:
            local = _local(parameter)
            if local.numel():
                value = torch.maximum(value, (local.double() - reference.double()).abs().max())
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(value, op=dist.ReduceOp.MAX, group=self.group)
        return float(value)

    @contextmanager
    def swapped(self):
        if torch.is_grad_enabled():
            raise RuntimeError("reference swap must run under torch.no_grad()")
        reshard_all(self.modules)
        for parameter, reference, backup in self.items:
            local = _local(parameter)
            backup.copy_(local)
            local.copy_(reference)
        try:
            yield
        finally:
            reshard_all(self.modules)      # the root keeps reference values unsharded after a forward
            for parameter, _reference, backup in self.items:
                _local(parameter).copy_(backup)


def load_exact_warm_start(model: torch.nn.Module, state: dict) -> list[str]:
    """Warm start under FSDP: a checkpoint's trainable tensors -> the sharded parameters, exactly.

    Every checkpoint key must be a trainable parameter of the same shape; returns the trainable names the
    checkpoint lacks (the trainer refuses any for DPO).
    """
    reshard_all(model)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    unknown = sorted(set(state) - trainable)
    if unknown:
        raise RuntimeError(f"Warm-start checkpoint has unknown trainable keys: {unknown[:10]}")
    load_full_trainable_state(model, state)
    return sorted(trainable - set(state))


def slot_group(mesh, scope: str = "shard"):
    """Process group over which the DPO schedule is agreed: the shard group (paper runs) or the world."""
    scope = str(scope).strip().lower()
    if scope == "world":
        return None
    if scope != "shard":
        raise ValueError(f"DPO slot scope must be shard or world, got {scope!r}")
    return mesh.get_group("shard")


def agree_slots(n_pairs: int, group, device) -> int:
    """max(K) over ``group`` (collective, once per micro-step on every rank)."""
    value = torch.tensor([int(n_pairs)], dtype=torch.int64, device=device)
    dist.all_reduce(value, op=dist.ReduceOp.MAX, group=group)
    return int(value.item())


def fsdp_dpo_study_backward(*, score, chosen_ids, rejected, slots, reference, ddp_context, beta, dpo_weight,
                            chosen_ce_weight, unpaired_ce_weight, accumulation, stats,
                            recompute_tolerance=0.5, mismatch_abort=5.0,
                            dummy_tokens: int = DUMMY_TARGET_TOKENS):
    """One micro-step of the fixed FSDP DPO schedule (see module docstring).

    score(kind, target_ids) -> (sum_logprob tensor, output dict); kind in {"reference", "policy", "train"}.
    ddp_context(is_last) -> gradient-sync context of a grad pass (sync only for the last at a boundary).
    rejected: list of (pair_type, target_ids) with len(rejected) <= slots, slots >= 1.
    Returns (combined_loss_value, dpo_mean_value, chosen_output_dict, paired).
    """
    from fora_report.ct.dpo_loss import dpo_terms

    K = len(rejected)
    if slots < 1 or K > slots:
        raise ValueError(f"invalid DPO schedule: K={K} slots={slots}")
    n_chosen = int(chosen_ids.numel())
    dummy = chosen_ids[: max(1, min(int(dummy_tokens), n_chosen))]
    with torch.no_grad():
        with reference.swapped():
            ref_c = score("reference", chosen_ids if K else dummy)[0].detach().float()
            ref_r = [score("reference", ids)[0].detach().float() for _t, ids in rejected]
            for _ in range(slots - K):
                score("reference", dummy)
        pol_r = [score("policy", ids)[0].detach().float() for _t, ids in rejected]
        for _ in range(slots - K):
            score("policy", dummy)
    # chosen graph (never the last pass: slots >= 1)
    with ddp_context(False):
        pol_c_live, chosen_output = score("train", chosen_ids)
        pol_c_live = pol_c_live.float()
        if K:
            losses = [dpo_terms(pol_c_live, pol_r[k], ref_c, ref_r[k], beta)[0] for k in range(K)]
            dpo_mean = torch.stack(losses).mean()
            chosen_ce = -pol_c_live / n_chosen
            total = dpo_weight * dpo_mean + chosen_ce_weight * chosen_ce
        else:
            dpo_mean = None
            total = unpaired_ce_weight * chosen_output["loss"].float()
        if not torch.isfinite(total).item():
            raise FloatingPointError("non-finite DPO chosen objective")
        (total / accumulation).backward()
    pol_c = pol_c_live.detach()
    combined = float(total.detach())
    dpo_value = float(dpo_mean.detach()) if dpo_mean is not None else 0.0
    for j in range(slots):
        with ddp_context(j == slots - 1):
            if j < K:
                pair_type, ids = rejected[j]
                pol_rk_live, _out = score("train", ids)
                pol_rk_live = pol_rk_live.float()
                error = abs(float(pol_rk_live.detach()) - float(pol_r[j]))
                if error > mismatch_abort:
                    raise RuntimeError(f"rejected policy score changed on recompute by {error:.3f} nats")
                if error > recompute_tolerance:
                    stats.misc[3] += 1
                loss_k, _ = dpo_terms(pol_c, pol_rk_live, ref_c, ref_r[j], beta)
                ((dpo_weight / K) * loss_k / accumulation).backward()
                margin = float((pol_c - pol_r[j]) - (ref_c - ref_r[j]))
                stats.add(pair_type, dpo_loss=float(-F.logsigmoid(torch.tensor(beta * margin))),
                          correct=float(margin > 0), margin=margin,
                          policy_margin=float(pol_c - pol_r[j]), reference_margin=float(ref_c - ref_r[j]),
                          chosen_ce=float(chosen_ce.detach()), recompute_abs_error=error)
                del pol_rk_live, _out
            else:
                dummy_live, _out = score("train", dummy)
                (dummy_live.float() * 0.0).backward()
                del dummy_live, _out
    stats.misc[0 if K else 1] += 1
    if not math.isfinite(combined):
        raise FloatingPointError("non-finite DPO objective")
    return combined, dpo_value, chosen_output, bool(K)

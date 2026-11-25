"""Offline DPO objective for the CT report writer (paper: "Preference optimisation", CT DPO stage).

Inputs are identical to SFT (DINO tokens, diagnostic plan with evidence text, focus tokens, zoomed evidence images);
only the response differs between the chosen (reference report) and the rejected (synthetic corruption) target.

* The frozen reference is the stage-2 SFT checkpoint. Its trainable tensors (LoRA, connector, vision tower) are kept
  on the GPU and swapped into the policy under ``torch.no_grad`` for the reference scores of the current study
  (:class:`ReferenceWeights`; the FSDP variant is ``fora_report.writer.fsdp_dpo.FSDPReferenceWeights``).
* Sequence score = SUM log-probability of every target token including EOS. With every auxiliary loss at zero the
  writer's ``loss`` is the token-mean report NLL, so sum log p = -loss * len(target).
* Objective per paired study (K pairs share the chosen response)::

      dpo_weight * mean_k[-log sigmoid(beta * ((pc - pr_k) - (rc - rr_k)))] + chosen_ce_weight * CE(chosen)

  and ``unpaired_ce_weight`` * CE for studies without a usable pair.
* Exact chain rule with one response graph resident at a time: rejected policy scores are first computed without
  gradient; the chosen graph is differentiated with them as constants, then every rejected graph with the detached
  chosen score. Each recomputed rejected score is compared with its no-grad value (dropout disabled).
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from contextlib import contextmanager

import torch
from torch.nn import functional as F

PREFERENCE_OBJECTIVE = "ct_offline_dpo_sum_logprob"
PREFERENCE_SCHEMA = "ct_report_preference_index_v1"
# Row order of DPOStatistics (logging only). The final pool contains all eight types.
PAIR_TYPES = ("omission", "repetition", "explicit_denial", "attribute",
              "insertion_plan_fp", "insertion_random", "normalised", "multi_omission")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class PreferencePairStore:
    """Read-only sqlite pair index (schema ``ct_report_preference_index_v1``, written by ``fora_report.pairs.merge``).

    Table ``pairs(pair_id, report_uid, pair_type, attribute_type, chosen_sha256, rejected_text, ...)`` and table
    ``metadata(key, value)`` with a JSON ``summary`` row.
    """

    def __init__(self, path: str, pair_types=None):
        self.path = str(path)
        self.connection = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, check_same_thread=False)
        row = self.connection.execute("SELECT value FROM metadata WHERE key='summary'").fetchone()
        self.summary = json.loads(row[0]) if row else {}
        if self.summary.get("schema") != PREFERENCE_SCHEMA:
            raise ValueError(f"not a CT preference index: {self.path}")
        self.pair_types = tuple(pair_types) if pair_types else None
        if self.pair_types and set(self.pair_types) - set(PAIR_TYPES):
            raise ValueError(f"unknown pair types {self.pair_types}")

    def pairs(self, report_uid: str):
        rows = self.connection.execute(
            "SELECT pair_id, pair_type, attribute_type, chosen_sha256, rejected_text "
            "FROM pairs WHERE report_uid=? ORDER BY pair_id", (str(report_uid),)).fetchall()
        if self.pair_types:
            rows = [r for r in rows if r[1] in self.pair_types]
        return rows

    def count_uids(self, uids) -> int:
        found = 0
        for uid in uids:
            if self.connection.execute("SELECT 1 FROM pairs WHERE report_uid=? LIMIT 1", (str(uid),)).fetchone():
                found += 1
        return found


def select_pairs(rows, *, uid: str, epoch: int, seed: int, k: int):
    """Deterministic per-visit selection that cycles through a study's pairs over epochs.

    Pairs are ordered by a seeded hash, interleaving pair types so the first visits cover different kinds;
    visit ``epoch`` takes the next ``k`` pairs of that cycle.
    """
    if not rows or k <= 0:
        return []
    by_type = {}
    for row in sorted(rows, key=lambda r: hashlib.sha256(f"{seed}|{uid}|{r[0]}".encode()).hexdigest()):
        by_type.setdefault(row[1], []).append(row)
    kinds = sorted(by_type, key=lambda t: hashlib.sha256(f"{seed}|{uid}|{t}".encode()).hexdigest())
    order = []
    depth = 0
    while len(order) < len(rows):
        for kind in kinds:
            if depth < len(by_type[kind]):
                order.append(by_type[kind][depth])
        depth += 1
    start = (epoch * k) % len(order)
    return [order[(start + i) % len(order)] for i in range(min(k, len(order)))]


class ReferenceWeights:
    """Frozen SFT trainable tensors kept on device and swapped in for reference scores (unsharded model)."""

    def __init__(self, model: torch.nn.Module, state: dict, device):
        parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
        missing = sorted(set(parameters) - set(state))
        unknown = sorted(set(state) - set(parameters))
        if missing or unknown:
            raise RuntimeError(
                f"DPO reference must cover exactly the trainable tensors: missing={missing[:5]} "
                f"unknown={unknown[:5]} (n_missing={len(missing)} n_unknown={len(unknown)})")
        self.items = []
        numel = 0
        for name, parameter in parameters.items():
            value = state[name]
            if tuple(value.shape) != tuple(parameter.shape):
                raise RuntimeError(f"DPO reference shape mismatch for {name}")
            self.items.append((parameter, value.to(device=device, dtype=parameter.dtype).contiguous(),
                               torch.empty_like(parameter.data)))
            numel += parameter.numel()
        self.numel = numel

    @torch.no_grad()
    def max_abs_difference(self) -> float:
        return max(float((p.data - r).abs().max()) for p, r, _ in self.items) if self.items else 0.0

    @contextmanager
    def swapped(self):
        if torch.is_grad_enabled():
            raise RuntimeError("reference swap must run under torch.no_grad()")
        for parameter, reference, backup in self.items:
            backup.copy_(parameter.data)
            parameter.data.copy_(reference)
        try:
            yield
        finally:
            for parameter, _reference, backup in self.items:
                parameter.data.copy_(backup)


def load_reference_state(path: str) -> tuple[dict, dict]:
    package = torch.load(path, map_location="cpu", weights_only=False)
    state = package.get("trainable_model")
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"{path} has no trainable_model state")
    meta = dict(epoch=package.get("epoch"), microstep=package.get("microstep"),
                optimizer_step=package.get("optimizer_step"))
    return state, meta


def disable_preference_dropout(model) -> dict:
    """Dropout off (exact recompute of rejected scores); activation checkpointing is kept (model stays in train)."""
    changes = {}
    attributes = ("dropout", "attention_dropout", "activation_dropout", "hidden_dropout_prob",
                  "attention_probs_dropout_prob", "token_dropout", "block_dropout",
                  "mil_proposal_dropout", "diagnostic_image_dropout")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.dropout._DropoutNd):
            if module.p:
                changes[f"{name}.p"] = float(module.p)
            module.p = 0.0
        for attribute in attributes:
            value = getattr(module, attribute, None)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if value:
                    changes[f"{name}.{attribute}"] = float(value)
                setattr(module, attribute, 0.0)
        config = getattr(module, "config", None)
        for attribute in attributes:
            value = getattr(config, attribute, None)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if value:
                    changes[f"{name}.config.{attribute}"] = float(value)
                try:
                    setattr(config, attribute, 0.0)
                except Exception:
                    pass
    return changes


def dpo_terms(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta):
    """Sigmoid DPO loss on summed log-probabilities; returns (loss, margin)."""
    margin = (policy_chosen - policy_rejected) - (reference_chosen - reference_rejected)
    return -F.logsigmoid(beta * margin), margin


class DPOStatistics:
    """Per-type running sums, all-reduced and printed at log steps."""

    FIELDS = ("pairs", "dpo_loss", "correct", "margin", "policy_margin", "reference_margin",
              "chosen_ce", "recompute_abs_error")

    def __init__(self, device):
        self.types = PAIR_TYPES
        self.values = torch.zeros(len(self.types), len(self.FIELDS), device=device, dtype=torch.float64)
        self.misc = torch.zeros(4, device=device, dtype=torch.float64)  # paired_visits, unpaired, skipped, mismatch

    def add(self, pair_type, **values):
        row = self.types.index(pair_type)
        self.values[row, 0] += 1
        for key, value in values.items():
            self.values[row, self.FIELDS.index(key)] += float(value)

    def line(self, step, world_reduce):
        values, misc = self.values.clone(), self.misc.clone()
        world_reduce(values)
        world_reduce(misc)
        total = values.sum(0)
        n = max(float(total[0]), 1.0)
        parts = [f"[dpo] step={step} pairs={int(total[0])} loss={total[1]/n:.4f} acc={total[2]/n:.3f} "
                 f"margin={total[3]/n:.3f} pol_margin={total[4]/n:.2f} ref_margin={total[5]/n:.2f} "
                 f"chosen_ce={total[6]/n:.4f} paired_visits={int(misc[0])} unpaired={int(misc[1])} "
                 f"skipped_pairs={int(misc[2])} recompute_mismatch={int(misc[3])}"]
        for i, kind in enumerate(self.types):
            k = float(values[i, 0])
            if k:
                parts.append(f"{kind}:n={int(k)},acc={values[i, 2]/k:.3f},margin={values[i, 3]/k:.2f}")
        self.values.zero_()
        self.misc.zero_()
        return " ".join(parts)


def dpo_study_backward(*, score, chosen_ids, rejected, reference, ddp_context, beta, dpo_weight,
                       chosen_ce_weight, accumulation, stats, recompute_tolerance=0.5, mismatch_abort=5.0):
    """Exact sequential-graph DPO backward for one study (unsharded model).

    ``score(module_kind, target_ids) -> (sum_logprob, output)`` with module_kind "reference" (no grad, swapped
    weights), "policy" (no grad) or "train" (grad). ``ddp_context(is_last)`` returns the gradient-sync context for
    a gradient pass. ``rejected`` is a list of (pair_type, target_ids). Returns (combined_loss, dpo_mean, output).
    Training uses the FSDP twin ``fora_report.writer.fsdp_dpo.fsdp_dpo_study_backward`` with the same algebra.
    """
    K = len(rejected)
    n_chosen = int(chosen_ids.numel())
    with torch.no_grad():
        with reference.swapped():
            ref_c = score("reference", chosen_ids)[0].detach().float()
            ref_r = [score("reference", ids)[0].detach().float() for _t, ids in rejected]
        pol_r = [score("policy", ids)[0].detach().float() for _t, ids in rejected]
    passes = 1 + K
    with ddp_context(passes == 1):
        pol_c_live, chosen_output = score("train", chosen_ids)
        pol_c_live = pol_c_live.float()
        losses = []
        for k in range(K):
            loss_k, _m = dpo_terms(pol_c_live, pol_r[k], ref_c, ref_r[k], beta)
            losses.append(loss_k)
        dpo_mean = torch.stack(losses).mean()
        chosen_ce = -pol_c_live / n_chosen
        total = dpo_weight * dpo_mean + chosen_ce_weight * chosen_ce
        if not torch.isfinite(total).item():
            raise FloatingPointError("non-finite DPO chosen objective")
        (total / accumulation).backward()
    pol_c = pol_c_live.detach()
    dpo_sum = float(dpo_mean.detach())
    combined = float(total.detach())
    for k, (pair_type, ids) in enumerate(rejected):
        with ddp_context(k == K - 1):
            pol_rk_live, _out = score("train", ids)
            pol_rk_live = pol_rk_live.float()
            error = abs(float(pol_rk_live.detach()) - float(pol_r[k]))
            if error > mismatch_abort:
                raise RuntimeError(f"rejected policy score changed on recompute by {error:.3f} nats")
            if error > recompute_tolerance:
                stats.misc[3] += 1
            loss_k, _ = dpo_terms(pol_c, pol_rk_live, ref_c, ref_r[k], beta)
            ((dpo_weight / K) * loss_k / accumulation).backward()
        margin = float((pol_c - pol_r[k]) - (ref_c - ref_r[k]))
        stats.add(pair_type, dpo_loss=float(-F.logsigmoid(torch.tensor(beta * margin))),
                  correct=float(margin > 0), margin=margin,
                  policy_margin=float(pol_c - pol_r[k]), reference_margin=float(ref_c - ref_r[k]),
                  chosen_ce=float(chosen_ce.detach()), recompute_abs_error=error)
        del pol_rk_live, _out
    stats.misc[0] += 1
    if not math.isfinite(combined):
        raise FloatingPointError("non-finite DPO objective")
    return combined, dpo_sum, chosen_output

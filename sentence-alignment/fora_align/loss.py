"""VL-CABS multi-positive contrastive loss and its exact streamed computation (Methods, "Sentence alignment").

score(sentence s, report r) = temp * cos(l2norm(p_s,r), t_s), where p_s,r = sum_k a_k z_k pools ALL tokens z_k of report r
with a_k = softmax_k(pool_temp * cos(t_s, z_k)) (sentence-conditioned attention pooling, VL-CABS).

The loss is a symmetric InfoNCE over the global batch (one report per GPU):
* sentence -> report: cross-entropy over all reports, the sentence's own report is the target;
* report -> sentence: every sentence of a report is a positive; each positive is scored against the OTHER reports'
  sentences only, so a report's positives never compete with each other.

A report has up to ~10^6 tokens, so the pooling is streamed: pass 1 (no grad) keeps an online softmax per sentence;
pass 2 recomputes each chunk and back-propagates a surrogate whose gradient equals the exact gradient of the pooling.
"""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F


class GatherLayer(torch.autograd.Function):
    """All-gather with a deterministic backward: one blocking all_reduce(SUM) of the stacked gradients, own slice returned."""

    @staticmethod
    def forward(ctx, x):
        x = x.contiguous()
        out = [torch.empty_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(out, x)
        return tuple(out)

    @staticmethod
    def backward(ctx, *grads):
        grad_stack = torch.stack(grads, dim=0)
        dist.all_reduce(grad_stack, op=dist.ReduceOp.SUM)
        return grad_stack[dist.get_rank()]


def contrastive_ce(logits: torch.Tensor, num_sentences_per_image: int, mask_all: torch.Tensor | None) -> torch.Tensor:
    """logits [N_T sentences, B reports] (sentence-major blocks of ``num_sentences_per_image``); mask_all [N_T] valid rows."""
    device = logits.device
    logits = logits.float()
    B, N_T = logits.shape[1], logits.shape[0]
    sentence_to_image = torch.arange(B, device=device).repeat_interleave(num_sentences_per_image)
    membership = F.one_hot(sentence_to_image, num_classes=B).bool()
    rows = torch.arange(N_T, device=device)
    if mask_all is not None:
        n_valid = mask_all.float().sum().clamp(min=1.0)
        loss_t = torch.where(mask_all, F.cross_entropy(logits, sentence_to_image, reduction="none"),
                             torch.zeros(N_T, device=device)).sum() / n_valid
        pos = logits[rows, sentence_to_image]
        valid = mask_all.view(N_T, 1).expand(N_T, B)
        neg = logits.clone()
        neg[~(~membership & valid)] = -1e4
        log_denom = torch.logsumexp(torch.stack([pos, torch.logsumexp(neg, dim=0)[sentence_to_image]], dim=1), dim=1)
        loss_i = torch.where(mask_all, -pos + log_denom, torch.zeros(N_T, device=device)).sum() / n_valid
    else:
        loss_t = F.cross_entropy(logits, sentence_to_image)
        pos = logits[rows, sentence_to_image]
        neg = logits.clone()
        neg[membership] = -1e4
        log_denom = torch.logsumexp(torch.stack([pos, torch.logsumexp(neg, dim=0)[sentence_to_image]], dim=1), dim=1)
        loss_i = (-pos + log_denom).mean()
    return loss_t + loss_i


class StreamPool:
    """Exact VL-CABS pooling over a token stream with memory independent of the report size.

    Pass 1 (``add``): online softmax over chunks -> per-sentence max m, normalizer S and pooled vector p.
    Pass 2 (``surrogate``): for a recomputed chunk z, given g = dL/dp, the surrogate
        J = sum_ik a_ik.detach() (g_i . z_k) + sum_ik s_ik [a_ik (g_i . z_k - g_i . p_i)].detach()
    has dJ/dz = dL/dz and dJ/dt, dJ/dpool_temp equal to the attention-path gradients."""

    def __init__(self, text: torch.Tensor, pool_temp):
        self.t = text.detach()
        n, d = self.t.shape
        self.pt = float(pool_temp)
        self.m = torch.full((n,), -torch.inf, device=self.t.device, dtype=self.t.dtype)
        self.S = torch.zeros(n, device=self.t.device, dtype=self.t.dtype)
        self.P = torch.zeros(n, d, device=self.t.device, dtype=self.t.dtype)

    @torch.no_grad()
    def add(self, z: torch.Tensor) -> None:
        s = self.pt * (self.t @ z.T) * z.norm(dim=-1).clamp_min(1e-12).reciprocal()[None]
        mn = torch.maximum(self.m, s.amax(1))
        c, e = torch.exp(self.m - mn), torch.exp(s - mn[:, None])
        self.S = self.S * c + e.sum(1)
        self.P = self.P * c[:, None] + e @ z
        self.m = mn

    def pooled(self) -> torch.Tensor:
        return self.P / self.S[:, None]

    def surrogate(self, t_leaf, pool_temp, z, p, g):
        s = pool_temp * (t_leaf @ z.T) * z.norm(dim=-1).clamp_min(1e-12).reciprocal()[None]
        with torch.no_grad():
            a = torch.exp(s - self.m[:, None]) / self.S[:, None]
            ds = a * (g @ z.T - (g * p).sum(1, keepdim=True))
        return (a * (g @ z.T)).sum() + (s * ds).sum()


def pooled_logits_reference(text, z, pool_temp, temp):
    """Plain autograd VL-CABS score of all sentences against one report's tokens z (the definition; used in tests)."""
    w = torch.softmax(pool_temp * ((text @ z.T) * z.norm(dim=-1).clamp_min(1e-12).reciprocal()[None]), dim=1)
    return (F.normalize(w @ z, dim=-1) * text).sum(1) * temp

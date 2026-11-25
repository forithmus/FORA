"""Organ-routed attention MIL head and its exact streamed forward / backward (Methods, "Finding classifier").

Head (one per finding c, shared token layers):
  * gated attention  a_t,c = w_c . (tanh(V x_t) * sigmoid(U x_t)) + b_c, hidden 512;
  * token classifier l_t,c = Linear_c(GELU(Linear(x_t))), hidden 384;
  * soft organ routing: tokens outside the finding's route get -softplus(r_c) added to their attention score (a learned
    per-finding penalty, initialised at 5; never a hard exclusion);
  * study logit z_c = s_c * sum_t softmax_t(a_t,c) l_t,c + o_c with per-finding scale s_c and bias o_c.
The softmax runs over every token of every view of the study bag (all series, windows and the lung MIP).

Inputs x_t are block-24 tokens standardised with fixed statistics, (y - mean) / rms, concatenated with 24 Fourier position
features (``position_features``), 1024 + 24 = 1048 dims.

A study bag can hold millions of tokens. ``forward`` and ``backward`` stream it through a token factory (an iterable of
(x, route mask) chunks that is identical on every call): pass 1 computes the online-softmax statistics without a graph,
pass 2 recomputes each chunk with a graph and back-propagates the exact gradient of the pooled logit. Parameters must not
change between the passes (checked).
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fora.encoder.positions import position_features

__all__ = ["SparseClassifyThenAggregate", "OrganRoutedHead", "build_head", "route_scores", "statistics", "forward",
           "backward", "class_bits", "masked_bce", "auroc", "position_features", "load_classifier", "CLASS_ROWS", "HEAD_DIM",
           "ALL_BITS"]

POSITION_FREQS = 4
HEAD_DIM = 1024 + 6 * POSITION_FREQS        # 1048
ALL_BITS = (1 << 64) - 1                    # route = every token (no routing)
# per-finding parameter rows of the head (a rare finding's rows are tied to its parent's, see classifier/train_ct.py)
CLASS_ROWS = ("attention_w.weight", "attention_w.bias", "patch_out.weight", "patch_out.bias", "output_scale",
              "output_bias", "route_bias_raw")


class SparseClassifyThenAggregate(nn.Module):
    """Class-selective gated-attention MIL (classify every token, then aggregate) with chunked online softmax."""

    def __init__(self, dim: int, n_classes: int, hidden_dim: int = 512, mlp_hidden_dim: int = 384,
                 token_chunk_size: int = 32768, checkpoint_chunks: bool = True) -> None:
        super().__init__()
        self.dim, self.n_classes = int(dim), int(n_classes)
        self.hidden_dim, self.mlp_hidden_dim = int(hidden_dim), int(mlp_hidden_dim)
        self.token_chunk_size = int(token_chunk_size)
        self.checkpoint_chunks = bool(checkpoint_chunks)
        self.attention_v = nn.Linear(self.dim, self.hidden_dim)
        self.attention_u = nn.Linear(self.dim, self.hidden_dim)
        self.attention_w = nn.Linear(self.hidden_dim, self.n_classes)
        self.patch_hidden = nn.Linear(self.dim, self.mlp_hidden_dim)
        self.patch_out = nn.Linear(self.mlp_hidden_dim, self.n_classes)
        self.output_scale = nn.Parameter(torch.ones(self.n_classes))
        self.output_bias = nn.Parameter(torch.zeros(self.n_classes))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _chunk_attention_scores(self, tokens: torch.Tensor, class_indices: torch.Tensor) -> torch.Tensor:
        """Attention scores [n, k] of the selected classes (no routing)."""
        hidden = torch.tanh(self.attention_v(tokens)) * torch.sigmoid(self.attention_u(tokens))
        return F.linear(hidden, self.attention_w.weight.index_select(0, class_indices),
                        self.attention_w.bias.index_select(0, class_indices))

    def _chunk_outputs(self, tokens: torch.Tensor, class_indices: torch.Tensor):
        """(attention scores [n, k], token logits [n, k]) of the selected classes."""
        scores = self._chunk_attention_scores(tokens, class_indices)
        hidden = F.gelu(self.patch_hidden(tokens))
        logits = F.linear(hidden, self.patch_out.weight.index_select(0, class_indices),
                          self.patch_out.bias.index_select(0, class_indices))
        return scores, logits

    def forward(self, tokens: torch.Tensor, class_indices: torch.Tensor) -> torch.Tensor:
        """Unrouted bag logits [1, k] of one in-memory bag."""
        from torch.utils.checkpoint import checkpoint
        class_indices = class_indices.to(device=tokens.device, dtype=torch.long)
        running_max = torch.full((len(class_indices),), -torch.inf, dtype=tokens.dtype, device=tokens.device)
        den = torch.zeros_like(running_max)
        num = torch.zeros_like(running_max)
        for start in range(0, tokens.shape[0], self.token_chunk_size):
            chunk = tokens[start:start + self.token_chunk_size]
            if self.checkpoint_chunks and self.training and torch.is_grad_enabled():
                scores, logits = checkpoint(self._chunk_outputs, chunk, class_indices, use_reentrant=False)
            else:
                scores, logits = self._chunk_outputs(chunk, class_indices)
            new_max = torch.maximum(running_max, scores.max(dim=0).values)
            old = torch.exp(running_max - new_max)
            w = torch.exp(scores - new_max[None])
            num = num * old + (w * logits).sum(0)
            den = den * old + w.sum(0)
            running_max = new_max
        z = num / den.clamp_min(torch.finfo(den.dtype).tiny)
        return (z * self.output_scale.index_select(0, class_indices) + self.output_bias.index_select(0, class_indices))[None]


class OrganRoutedHead(SparseClassifyThenAggregate):
    """The MIL head with soft organ routing: ``route_bias_raw`` [C] gives the out-of-route penalty softplus(raw)."""

    def __init__(self, dim: int, n_classes: int, hidden_dim: int = 512, mlp_hidden_dim: int = 384,
                 token_chunk_size: int = 8192, checkpoint_chunks: bool = True, soft_init: float = 5.0) -> None:
        super().__init__(dim, n_classes, hidden_dim, mlp_hidden_dim, token_chunk_size, checkpoint_chunks)
        self.route_bias_raw = nn.Parameter(torch.full((self.n_classes,), float(np.log(np.expm1(soft_init)))))

    def penalty(self) -> torch.Tensor:
        """Out-of-route attention penalty softplus(raw) >= 0 per class."""
        return F.softplus(self.route_bias_raw.float())

    def forward(self, tokens: torch.Tensor, class_indices: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Routed bag logits [1, k] of one in-memory bag; mask [n, k] bool = token inside the class route."""
        class_indices = class_indices.to(device=tokens.device, dtype=torch.long)
        maximum = torch.full((len(class_indices),), -torch.inf, device=tokens.device, dtype=torch.float32)
        den = torch.zeros_like(maximum)
        num = torch.zeros_like(maximum)
        pen = self.penalty()[class_indices]
        for start in range(0, len(tokens), self.token_chunk_size):
            chunk = tokens[start:start + self.token_chunk_size]
            keep = mask[start:start + self.token_chunk_size]
            scores, logits = self._chunk_outputs(chunk, class_indices)
            scores = scores.float() + (~keep).float() * (-pen)
            new = torch.maximum(maximum, scores.max(0).values)
            old = torch.exp(maximum - new)
            w = torch.exp(scores - new)
            num = num * old + (w * logits.float()).sum(0)
            den = den * old + w.sum(0)
            maximum = new
        z = num / den.clamp_min(torch.finfo(den.dtype).tiny)
        return (z * self.output_scale[class_indices] + self.output_bias[class_indices])[None]


def build_head(n_classes: int, soft_init: float = 5.0, dim: int = HEAD_DIM) -> OrganRoutedHead:
    """The FORA head: 1048-d input, attention hidden 512, token MLP hidden 384, chunks of 8,192 tokens."""
    return OrganRoutedHead(dim, n_classes, hidden_dim=512, mlp_hidden_dim=384, token_chunk_size=8192,
                           checkpoint_chunks=True, soft_init=soft_init)


def load_classifier(m: dict, ckpt, device, ema: bool = False):
    """(block 24, head, classes) of a CT classifier checkpoint (raw weights unless ``ema``); ``m`` is the classifier
    config (``block24_state``: the block-24 construction state the checkpoint's block weights are loaded into)."""
    from fora.encoder.blocks import build_block24
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    classes = list(ck["classes"])
    block = build_block24(torch.load(m["block24_state"], map_location="cpu", weights_only=False))
    block.load_state_dict(ck["block"])
    head = build_head(len(classes), dim=HEAD_DIM)
    head.load_state_dict(ck["head"])
    if ema:
        params = [p for n, p in head.named_parameters() if n != "route_bias_raw"] + [head.route_bias_raw]
        params += [p for p in block.parameters() if p.ndim > 1] + [p for p in block.parameters() if p.ndim <= 1]
        with torch.no_grad():
            for p, e in zip(params, ck["ema"]):
                p.copy_(e)
    return block.to(device).eval(), head.to(device).eval(), classes


def route_scores(head: OrganRoutedHead, scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Attention scores after soft routing: outside-route tokens get -softplus(raw) <= 0."""
    return scores.float() + (~mask).float() * (-F.softplus(head.route_bias_raw.float()))


@torch.no_grad()
def statistics(head, factory):
    """Pass 1 over the bag: (running max, softmax denominator, pooled token logit, token count) for all classes."""
    device = next(head.parameters()).device
    classes = torch.arange(head.n_classes, device=device)
    maximum = torch.full((head.n_classes,), -torch.inf, device=device)
    den = torch.zeros_like(maximum)
    num = torch.zeros_like(maximum)
    n = 0
    for x, mask in factory():
        assert mask.dtype == torch.bool and mask.shape == (len(x), head.n_classes)
        scores, logits = head._chunk_outputs(x, classes)
        scores = route_scores(head, scores, mask)
        new = torch.maximum(maximum, scores.amax(0))
        safe = torch.where(torch.isfinite(new), new, torch.zeros_like(new))
        old = (maximum - safe).exp()
        weights = (scores - safe).exp()
        den = den * old + weights.sum(0)
        num = num * old + (weights * logits.float()).sum(0)
        maximum = new
        n += len(x)
    assert n > 0 and bool((den > 0).all()) and bool(torch.isfinite(num).all())
    return maximum, den, num / den, n


def forward(head, factory) -> torch.Tensor:
    """Study logits [1, C] of a streamed bag."""
    with torch.no_grad():
        _, _, pooled, _ = statistics(head, factory)
        return (pooled * head.output_scale + head.output_bias)[None]


def backward(head, factory, target: torch.Tensor, loss_scale: float = 1., loss_fn=None):
    """Exact two-pass gradient of loss(study logits, target) w.r.t. the head and everything upstream of the factory."""
    versions = tuple(p._version for p in head.parameters())
    maximum, den, pooled, n = statistics(head, factory)
    leaf = pooled.detach().requires_grad_(True)
    logits = (leaf * head.output_scale + head.output_bias)[None]
    objective = loss_fn or F.binary_cross_entropy_with_logits
    loss = objective(logits, target.reshape(1, -1)) * loss_scale
    loss.backward()
    gradient = leaf.grad.detach()
    classes = torch.arange(head.n_classes, device=pooled.device)
    seen = 0
    for x, mask in factory():
        scores, patch = head._chunk_outputs(x, classes)
        routed = route_scores(head, scores, mask)       # the graph also reaches the soft-route penalty
        with torch.no_grad():
            weights = (routed.detach() - maximum).exp() / den
            dl = weights * gradient
            ds = dl * (patch.float() - pooled)
        torch.autograd.backward((routed, patch), (ds, dl))
        seen += len(x)
    assert n == seen and versions == tuple(p._version for p in head.parameters())
    return loss.detach(), logits.detach()


def class_bits(region_bits: dict, classes, device) -> torch.Tensor:
    """{class: int route bitmask} -> int64 tensor [C] (classes absent from the table route to every token)."""
    bits = np.array([int(region_bits.get(n, ALL_BITS)) for n in classes], dtype=np.uint64).view(np.int64)
    return torch.from_numpy(bits).to(device)


def masked_bce(n_classes: int, pos_weight=None):
    """Sum of BCE over the known (finite-target) classes divided by the number of head classes."""
    def fn(logits, target):
        known = torch.isfinite(target)
        t = torch.where(known, target, torch.zeros_like(target))
        loss = F.binary_cross_entropy_with_logits(logits, t, reduction="none", pos_weight=pos_weight)
        return (loss * known).sum() / n_classes
    return fn


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUROC with average ranks for ties; y in {0, 1}."""
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s))
    ss = s[o]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    P = y.sum()
    N = len(y) - P
    return float((r[y == 1].sum() - P * (P + 1) / 2) / (P * N))


def softplus_inverse(v: float) -> float:
    """raw with softplus(raw) = v (route penalty initialisation)."""
    return math.log(math.expm1(v))

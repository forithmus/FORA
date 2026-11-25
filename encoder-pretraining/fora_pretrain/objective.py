"""Self-distillation objective of the encoder pre-training (Methods, "Image encoder pre-training").

Student and EMA teacher are the 3-D ViT-L with a DINO (CLS) head and an iBOT (masked patch) head, 16,384 prototypes each.
Per step the loss is

    L = L_DINO + L_iBOT + 0.1 L_KoLeo (+ L_Gram in phase 2) + L_cross

* DINO: the two global crops of the teacher give Sinkhorn-Knopp targets for the CLS token of the two student global crops
  (off-diagonal) and of the 8 local crops, weighted as in DINOv3 (2 global terms vs 2 x 8 local terms);
* iBOT: masked patch tokens of the student global crops predict the teacher's targets of the same tokens. The teacher's
  Sinkhorn for the patch head runs separately in 2 x 2 x 2 spatial bins of the crop's token grid ("position-binned
  Sinkhorn", after CAPI, arXiv 2502.08769), so the balancing cannot be met by encoding position;
* KoLeo on the student global CLS tokens, with a detached global nearest-neighbour search;
* Gram anchoring (phase 2): MSE between the student's and a frozen anchor's patch Gram matrices on a shared random subset of
  1,024 tokens; the anchor is the phase-1 EMA teacher at the start of phase 2;
* cross-crop patch term: the second global crop is placed on the patch lattice relative to the first, so every token in
  their overlap has a twin covering the same voxels. Each token's student prediction must match the teacher target of its
  twin in the other crop, in both directions (VICRegL / PixPro / DenseCL idea).

Both heads compute cosine logits (the prototype layer weight is L2-normalized per prototype, DINOv2 style), and every
Sinkhorn uses the actual global token count as its denominator. Each sample carries an importance weight
(1 / number of reconstructions of its report) x (reconstructions / reports), so every report has the same expected weight.
The teacher-target entropy H and KL = CE - H of each head are returned as monitoring metrics.
"""
from __future__ import annotations

import copy
import math
import types
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

from dinov3.layers.dino_head import DINOHead

MONITOR_KEYS = ("dino_target_entropy", "ibot_target_entropy", "dino_target_top1", "ibot_target_top1",
                "dino_kl", "ibot_kl", "dino_prototype_usage", "ibot_prototype_usage")
CROSS_KEYS = ("cross", "cross_target_entropy", "cross_kl", "cross_pairs")
METRIC_KEYS = ("loss", "dino", "ibot", "koleo", "gram", "dino_global", "dino_local") + MONITOR_KEYS + CROSS_KEYS


class SSLNetwork3D(nn.Module):
    def __init__(self, backbone, dino_prototypes: int, ibot_prototypes: int, dino_hidden_dim: int, ibot_hidden_dim: int,
                 dino_bottleneck_dim: int, ibot_bottleneck_dim: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.dino_head = DINOHead(in_dim=backbone.embed_dim, out_dim=dino_prototypes, hidden_dim=dino_hidden_dim,
                                  bottleneck_dim=dino_bottleneck_dim, nlayers=3)
        self.ibot_head = DINOHead(in_dim=backbone.embed_dim, out_dim=ibot_prototypes, hidden_dim=ibot_hidden_dim,
                                  bottleneck_dim=ibot_bottleneck_dim, nlayers=3)

    def init_weights(self) -> None:
        self.backbone.init_weights()
        self.dino_head.init_weights()
        self.ibot_head.init_weights()


def _cosine_prototypes(self, x):
    return F.linear(x, F.normalize(self.weight, dim=1, eps=1e-12))


def normalize_prototypes(learner) -> None:
    """Cosine logits on the prototype layer of both heads of student and teacher; parameters keep the plain-linear layout."""
    for network in (learner.student, learner.teacher):
        for head in ("dino_head", "ibot_head"):
            layer = getattr(network, head).last_layer
            if not isinstance(layer, nn.Linear) or layer.bias is not None:
                raise ValueError("Expected a bias-free linear prototype layer")
            layer.forward = types.MethodType(_cosine_prototypes, layer)


@torch.no_grad()
def distributed_sinkhorn(logits: Tensor, temperature: float, iterations: int = 3) -> Tensor:
    """Sinkhorn-Knopp over WORLD with the SUM of the ranks' actual row counts as batch size (ranks may differ, even 0)."""
    if logits.ndim != 2 or logits.shape[1] < 1:
        raise ValueError("logits must have shape [local_tokens, positive_prototypes]")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    distributed = dist.is_initialized()
    global_count = torch.tensor(logits.shape[0], dtype=torch.int64, device=logits.device)
    if distributed:
        dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
    global_batch = int(global_count.item())
    if global_batch == 0:
        raise ValueError("Sinkhorn requires at least one token globally")
    max_logit = (logits.detach().float().max() if logits.shape[0]
                 else torch.full((), -torch.inf, dtype=torch.float32, device=logits.device))
    if distributed:
        dist.all_reduce(max_logit, op=dist.ReduceOp.MAX)
    q = torch.exp((logits.float() - max_logit) / temperature).t()
    prototypes = q.shape[0]
    total = q.sum()
    if distributed:
        dist.all_reduce(total)
    q /= total.clamp_min(1e-12)
    for _ in range(iterations):
        rows = q.sum(dim=1, keepdim=True)
        if distributed:
            dist.all_reduce(rows)
        q /= rows.clamp_min(1e-12)
        q /= prototypes
        q /= q.sum(dim=0, keepdim=True).clamp_min(1e-12)
        q /= global_batch
    return (q * global_batch).t()


def position_bin_ids(mask_indices: Tensor, n_tokens: int, grid, bins) -> Tensor:
    """Spatial bin id in [0, prod(bins)) of every masked token (row order of the patch logits)."""
    gz, gy, gx = (int(v) for v in grid)
    if gz * gy * gx != int(n_tokens):
        raise ValueError(f"position grid {grid} does not match {n_tokens} tokens per crop")
    bz, by, bx = (int(v) for v in bins)
    pos = mask_indices % int(n_tokens)
    z = pos // (gy * gx)
    y = (pos // gx) % gy
    x = pos % gx
    return ((z * bz) // gz) * (by * bx) + ((y * by) // gy) * bx + (x * bx) // gx


def binned_sinkhorn(logits: Tensor, temperature: float, bin_ids: Tensor, n_bins: int, iterations: int = 3) -> Tensor:
    """``distributed_sinkhorn`` once per spatial bin; every rank visits every globally non-empty bin."""
    if bin_ids.shape[0] != logits.shape[0]:
        raise ValueError("bin ids must align with the logit rows")
    out = torch.empty(logits.shape, dtype=torch.float32, device=logits.device)
    counts = torch.bincount(bin_ids, minlength=n_bins).to(torch.int64)
    if dist.is_initialized():
        counts = counts.to(logits.device)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    for b in range(n_bins):
        if int(counts[b]) == 0:
            continue
        rows = (bin_ids == b).nonzero().flatten()
        out.index_copy_(0, rows, distributed_sinkhorn(logits.index_select(0, rows), temperature, iterations))
    return out


def dino_cross_entropy(student: Tensor, teacher: Tensor, student_temp: float, ignore_diagonal: bool) -> Tensor:
    """One DINO loss per sample for [views, batch, prototypes]."""
    logp = F.log_softmax(student.float() / student_temp, dim=-1)
    losses = -torch.einsum("sbk,tbk->stb", logp, teacher)
    if ignore_diagonal:
        n = min(student.shape[0], teacher.shape[0])
        view_mask = torch.ones(losses.shape[:2], dtype=torch.bool, device=losses.device)
        idx = torch.arange(n, device=losses.device)
        view_mask[idx, idx] = False
        return losses[view_mask].reshape(-1, student.shape[1]).mean(0)
    return losses.mean((0, 1))


def sampled_gram_loss(student: Tensor, target: Tensor, max_tokens: int, seed: int) -> Tensor:
    """Gram anchoring on the same deterministic random subset of at most ``max_tokens`` tokens of student and anchor."""
    n = student.shape[1]
    if n > max_tokens:
        g = torch.Generator(device=student.device).manual_seed(seed)
        idx = torch.randperm(n, generator=g, device=student.device)[:max_tokens]
        student, target = student[:, idx], target[:, idx]
    student = F.normalize(student.float(), dim=-1)
    target = F.normalize(target.float(), dim=-1)
    return F.mse_loss(student @ student.transpose(-1, -2), target @ target.transpose(-1, -2), reduction="none").mean((-1, -2))


def distributed_koleo(x: Tensor, eps: float = 1e-8) -> Tensor:
    """KoLeo with a detached global nearest-neighbour lookup (the gather happens in the forward only)."""
    x = F.normalize(x.float(), dim=-1)
    if dist.is_initialized():
        gathered = [torch.empty_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, x.detach())
        all_x = torch.cat(gathered)
        offset = dist.get_rank() * x.shape[0]
    else:
        all_x, offset = x.detach(), 0
    scores = x.detach() @ all_x.t()
    rows = torch.arange(x.shape[0], device=x.device)
    scores[rows, rows + offset] = -torch.inf
    if all_x.shape[0] == 1:
        return x.new_zeros((x.shape[0],))
    nearest = all_x[scores.argmax(dim=-1)]
    return -torch.log(torch.linalg.vector_norm(x - nearest, dim=-1).clamp_min(eps))


def padded_head(head, tokens: Tensor, chunk: int):
    """Run an FSDP-sharded head on a rank-variable token count with rank-identical call shapes.

    Every rank must issue the same number of head forwards (FSDP all-gathers), so tokens are processed in chunks up to the
    global maximum count and zero-padded. Returns (logits of the local tokens, a zero-valued graph anchor that keeps the
    backward hooks of padded-only calls)."""
    local = int(tokens.shape[0])
    maximum = torch.tensor(local, device=tokens.device)
    if dist.is_initialized():
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    maximum = int(maximum.item())
    parts, out = [], None
    anchor = tokens.sum() * 0.0
    for start in range(0, maximum, chunk):
        stop = min(start + chunk, maximum)
        valid = max(0, min(stop, local) - start)
        x = tokens[start:min(stop, local)]
        if valid < stop - start:
            x = F.pad(x, (0, 0, 0, stop - start - valid))
        out = head(x)
        if valid == 0:
            anchor = anchor + out.sum() * 0.0
            continue
        parts.append(out[:valid])
    if parts:
        logits = torch.cat(parts)
    elif maximum:
        logits = out[:0]
    else:
        logits = tokens.new_zeros((0, head.last_layer.weight.shape[0]))
    return logits, anchor


def _entropy(q: Tensor) -> Tensor:
    return -(q * q.clamp_min(1e-12).log()).sum(-1)


@dataclass
class LossWeights:
    dino: float = 1.0
    ibot: float = 1.0
    koleo: float = 0.1
    gram: float = 0.0
    cross: float = 1.0


class DINO3DLearner(nn.Module):
    """Student, EMA teacher, optional frozen Gram anchor, and all pre-training losses."""

    def __init__(self, backbone, *, dino_prototypes: int, ibot_prototypes: int, dino_head_hidden_dim: int,
                 ibot_head_hidden_dim: int, dino_bottleneck_dim: int, ibot_bottleneck_dim: int, position_grid,
                 position_bins, student_temperature: float = 0.1, loss_weights: LossWeights = LossWeights(),
                 gram_max_tokens: int = 1024, ibot_loss_chunk_size: int = 2048, cross_chunk_size: int = 4096,
                 with_gram_anchor: bool = False, initialize_weights: bool = True) -> None:
        super().__init__()
        self.student = SSLNetwork3D(backbone, dino_prototypes, ibot_prototypes, dino_head_hidden_dim,
                                    ibot_head_hidden_dim, dino_bottleneck_dim, ibot_bottleneck_dim)
        if initialize_weights:
            self.student.init_weights()
        self.teacher = copy.deepcopy(self.student).requires_grad_(False)
        self.teacher.eval()
        self.gram_anchor = copy.deepcopy(self.teacher).requires_grad_(False) if with_gram_anchor else None
        self.student_temperature = float(student_temperature)
        self.weights = loss_weights
        self.gram_max_tokens = int(gram_max_tokens)
        self.ibot_loss_chunk_size = int(ibot_loss_chunk_size)
        self.cross_chunk_size = int(cross_chunk_size)
        self.position_grid = [int(v) for v in position_grid]
        self.position_bins = [int(v) for v in position_bins]
        if len(self.position_bins) != 3 or any(b < 1 for b in self.position_bins):
            raise ValueError("position_bins must be three positive integers")

    @torch.no_grad()
    def initialize_student_teacher(self) -> None:
        """Initialize a materialized (FSDP2) student, then copy its local shards to teacher and Gram anchor."""
        self.student.init_weights()
        for target, source in zip(self.teacher.parameters(), self.student.parameters()):
            target.copy_(source)
        for target, source in zip(self.teacher.buffers(), self.student.buffers()):
            target.copy_(source)
        if self.gram_anchor is not None:
            for target, source in zip(self.gram_anchor.parameters(), self.teacher.parameters()):
                target.copy_(source)
            for target, source in zip(self.gram_anchor.buffers(), self.teacher.buffers()):
                target.copy_(source)

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        if self.gram_anchor is not None:
            self.gram_anchor.eval()
        return self

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        for target, source in zip(self.teacher.parameters(), self.student.parameters()):
            target.lerp_(source.detach(), 1.0 - momentum)
        for target, source in zip(self.teacher.buffers(), self.student.buffers()):
            target.copy_(source.detach())

    @torch.no_grad()
    def reset_gram_anchor_from_teacher(self) -> None:
        if self.gram_anchor is None:
            self.gram_anchor = copy.deepcopy(self.teacher).requires_grad_(False)
        else:
            self.gram_anchor.load_state_dict(self.teacher.state_dict())
        self.gram_anchor.eval()

    @staticmethod
    def _record(q: Tensor) -> dict | None:
        if q.shape[0] == 0:
            return None
        return dict(entropy_rows=_entropy(q), top1=q.max(-1).values.mean(),
                    usage=torch.tensor(q.argmax(-1).unique().numel() / q.shape[1], device=q.device))

    def forward(self, batch: dict, teacher_temperature: float, step: int) -> tuple[Tensor, dict[str, Tensor]]:
        teacher_images = batch["teacher_global"]          # [2, B, 1, D, H, W]
        student_images = batch["student_global"]
        local_images = batch["student_local"]
        masks, mask_indices, mask_weights = batch["masks"], batch["mask_indices"], batch["mask_weights"]
        n_global, batch_size = teacher_images.shape[:2]
        n_local = local_images.shape[0]
        sample_weights = batch.get("loss_weights")
        if sample_weights is None:
            sample_weights = torch.ones(batch_size, device=teacher_images.device)
        sample_weights = sample_weights.float()
        if sample_weights.shape != (batch_size,):
            raise ValueError(f"loss_weights must have shape ({batch_size},), got {tuple(sample_weights.shape)}")

        def weighted_mean(per_sample: Tensor) -> Tensor:
            # Importance weights have population mean one: never renormalize by the minibatch weight sum.
            return (per_sample * sample_weights).mean()

        bin_ids = position_bin_ids(mask_indices, int(masks.shape[-1]), self.position_grid, self.position_bins)
        n_bins = math.prod(self.position_bins)
        tg, sg, sl = teacher_images.flatten(0, 1), student_images.flatten(0, 1), local_images.flatten(0, 1)

        with torch.no_grad():
            tout = self.teacher.backbone(tg, is_training=True)
            t_cls_logits = self.teacher.dino_head(tout["x_norm_clstoken"])
            t_masked = tout["x_norm_patchtokens"].flatten(0, 1).index_select(0, mask_indices)
            t_patch_logits = self.teacher.ibot_head(t_masked)
            q_cls = distributed_sinkhorn(t_cls_logits, teacher_temperature)
            q_patch = (binned_sinkhorn(t_patch_logits, teacher_temperature, bin_ids, n_bins)
                       if t_patch_logits.shape[0] > 0 else distributed_sinkhorn(t_patch_logits, teacher_temperature))
            rec_cls, rec_patch = self._record(q_cls), self._record(q_patch)
            t_cls_prob = q_cls.unflatten(0, (n_global, batch_size))
            del t_patch_logits, t_masked

        global_out, local_out = self.student.backbone([sg, sl], masks=[masks, None], is_training=True)
        s_g_cls = self.student.dino_head(global_out["x_norm_clstoken"]).unflatten(0, (n_global, batch_size))
        s_l_cls = self.student.dino_head(local_out["x_norm_clstoken"]).unflatten(0, (n_local, batch_size))
        s_masked = global_out["x_norm_patchtokens"].flatten(0, 1).index_select(0, mask_indices)

        dino_global = dino_cross_entropy(s_g_cls, t_cls_prob, self.student_temperature, True)
        dino_local = dino_cross_entropy(s_l_cls, t_cls_prob, self.student_temperature, False)
        global_scale = 2.0 / (2.0 + 2.0 * n_local)
        dino = weighted_mean(global_scale * dino_global + (1.0 - global_scale) * dino_local)

        # iBOT: the head is FSDP-sharded, so every rank runs the same number of identically shaped head calls (padded to
        # the global maximum masked count); the cross-entropy is computed per chunk without a full log-softmax tensor.
        local_masked = int(s_masked.shape[0])
        maximum_masked = torch.tensor(local_masked, device=s_masked.device)
        if dist.is_initialized():
            dist.all_reduce(maximum_masked, op=dist.ReduceOp.MAX)
        maximum_masked = int(maximum_masked.item())
        ibot_parts = []
        ibot_anchor = s_masked.sum() * 0.0
        for start in range(0, maximum_masked, self.ibot_loss_chunk_size):
            common_stop = min(start + self.ibot_loss_chunk_size, maximum_masked)
            local_stop = min(common_stop, local_masked)
            valid = max(0, local_stop - start)
            tokens = s_masked[start:local_stop]
            if valid < common_stop - start:
                tokens = F.pad(tokens, (0, 0, 0, common_stop - start - valid))
            padded_logits = self.student.ibot_head(tokens)
            if valid == 0:
                ibot_anchor = ibot_anchor + padded_logits.sum() * 0.0
                continue
            scaled = padded_logits[:valid].float() / self.student_temperature
            targets = q_patch[start:local_stop]
            expected = torch.bmm(targets.unsqueeze(1), scaled.unsqueeze(2)).flatten()
            ibot_parts.append(torch.logsumexp(scaled, dim=-1) * targets.sum(-1) - expected)
        ibot_each = torch.cat(ibot_parts)
        mask_counts = masks.sum(-1)
        mask_ids = torch.arange(masks.shape[0], device=masks.device).repeat_interleave(mask_counts)
        ibot_per_mask = torch.zeros(masks.shape[0], device=ibot_each.device, dtype=ibot_each.dtype)
        ibot_per_mask.scatter_add_(0, mask_ids, ibot_each * mask_weights)
        ibot = weighted_mean(ibot_per_mask.unflatten(0, (n_global, batch_size)).mean(0)) + ibot_anchor

        koleo = weighted_mean(torch.stack([distributed_koleo(x) for x in
                                           global_out["x_norm_clstoken"].unflatten(0, (n_global, batch_size))]).mean(0))

        gram = dino.new_zeros(())
        if self.gram_anchor is not None and self.weights.gram > 0:
            with torch.no_grad():
                anchor = self.gram_anchor.backbone(tg, is_training=True)["x_norm_patchtokens"]
            gram = weighted_mean(sampled_gram_loss(global_out["x_norm_patchtokens"], anchor, self.gram_max_tokens,
                                                   seed=step).unflatten(0, (n_global, batch_size)).mean(0))

        total = self.weights.dino * dino + self.weights.ibot * ibot + self.weights.koleo * koleo + self.weights.gram * gram
        metrics = {"loss": total.detach(), "dino": dino.detach(), "dino_global": weighted_mean(dino_global).detach(),
                   "dino_local": weighted_mean(dino_local).detach(), "ibot": ibot.detach(), "koleo": koleo.detach(),
                   "gram": gram.detach()}
        metrics.update(self._monitor(metrics, rec_cls, rec_patch, batch, sample_weights, total.device))
        cross, cross_metrics = self._cross_view(batch, tout, global_out, teacher_temperature, sample_weights)
        metrics.update(cross_metrics)
        return total + self.weights.cross * cross, metrics

    @staticmethod
    def _monitor(metrics, cls, patch, batch, weights, device) -> dict:
        """Target entropy / top-1 / prototype usage and KL = CE - H per head, H aggregated exactly like its loss."""
        masks = batch["masks"]
        n_rows, batch_size = masks.shape[0], batch["teacher_global"].shape[1]
        zero = torch.zeros((), device=device)
        h_cls = h_patch = None
        if cls is not None and cls["entropy_rows"].shape[0] == n_rows:
            h_cls = (cls["entropy_rows"].view(-1, batch_size).mean(0) * weights).mean()
        if patch is not None and patch["entropy_rows"].shape[0] == batch["mask_indices"].shape[0]:
            ids = torch.arange(n_rows, device=masks.device).repeat_interleave(masks.sum(-1))
            per_mask = torch.zeros(n_rows, device=masks.device, dtype=torch.float32)
            per_mask.scatter_add_(0, ids, patch["entropy_rows"] * batch["mask_weights"].float())
            h_patch = (per_mask.view(-1, batch_size).mean(0) * weights).mean()
        extra = {
            "dino_target_entropy": cls["entropy_rows"].mean() if cls else zero,
            "ibot_target_entropy": patch["entropy_rows"].mean() if patch else zero,
            "dino_target_top1": cls["top1"] if cls else zero,
            "ibot_target_top1": patch["top1"] if patch else zero,
            "dino_kl": (metrics["dino_global"] - h_cls) if h_cls is not None else zero,
            "ibot_kl": (metrics["ibot"] - h_patch) if h_patch is not None else zero,
            "dino_prototype_usage": cls["usage"] if cls else zero,
            "ibot_prototype_usage": patch["usage"] if patch else zero,
        }
        return {k: v.detach().float() for k, v in extra.items()}

    def _cross_view(self, batch, tout, global_out, teacher_temperature, weights):
        """Cross-crop patch term over the twin tokens listed in batch['cross_a'] / ['cross_b'] (flat [2B, N] layout)."""
        s_out = global_out["x_norm_patchtokens"]         # [2B, N, C], view-major
        t_out = tout["x_norm_patchtokens"]
        zero = torch.zeros((), device=s_out.device)
        if self.weights.cross <= 0:
            return zero, dict(cross=zero, cross_target_entropy=zero, cross_kl=zero, cross_pairs=zero)
        batch_size = batch["teacher_global"].shape[1]
        N = s_out.shape[1]
        s_flat, t_flat = s_out.flatten(0, 1), t_out.flatten(0, 1)
        a_idx = batch["cross_a"].to(s_out.device)
        b_idx = batch["cross_b"].to(s_out.device) + batch_size * N    # view 1 follows view 0 in the flat layout
        n_pairs = int(a_idx.numel())
        with torch.no_grad():
            # targets of the crop-A tokens come from their crop-B twins, and vice versa
            t_tokens = torch.cat([t_flat.index_select(0, b_idx), t_flat.index_select(0, a_idx)])
            t_logits, _ = padded_head(self.teacher.ibot_head, t_tokens, self.cross_chunk_size)
            count = torch.tensor(t_logits.shape[0], device=t_logits.device, dtype=torch.int64)
            if dist.is_initialized():
                dist.all_reduce(count)
            q = distributed_sinkhorn(t_logits, teacher_temperature) if count.item() else t_logits.float()
        s_tokens = torch.cat([s_flat.index_select(0, a_idx), s_flat.index_select(0, b_idx)])
        s_logits, anchor = padded_head(self.student.ibot_head, s_tokens, self.cross_chunk_size)
        if n_pairs == 0:
            return anchor, dict(cross=anchor.detach(), cross_target_entropy=zero, cross_kl=zero, cross_pairs=zero)
        ce = -(q * F.log_softmax(s_logits.float() / self.student_temperature, dim=-1)).sum(-1)     # [2P]
        sample_ids = torch.cat([batch["cross_sample"], batch["cross_sample"]]).to(s_out.device)
        counts = torch.bincount(sample_ids, minlength=batch_size).clamp_min(1).float()
        per_sample = torch.zeros(batch_size, device=s_out.device).scatter_add_(0, sample_ids, ce) / counts
        cross = (per_sample * weights).mean() + anchor
        h = _entropy(q)
        per_sample_h = torch.zeros(batch_size, device=s_out.device).scatter_add_(0, sample_ids, h) / counts
        h_w = (per_sample_h * weights).mean()
        return cross, dict(cross=cross.detach(), cross_target_entropy=h.mean().detach(), cross_kl=(cross - h_w).detach(),
                           cross_pairs=torch.tensor(float(n_pairs), device=s_out.device))

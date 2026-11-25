"""Report writer: visual prefix + classifier tokens + diagnostic plan -> Gemma-4 (Methods, "Report generation").

Input sequence of one study (all positions before the report are a bidirectional prefix, token type 1 for the
visual and classifier parts):

    [image_start] [512 resampled visual latents] [study-type tokens] [image_end]
    [one Platt-calibrated classifier token per vocabulary class]
    [prompt / diagnostic plan; each planned finding's placeholders carry its pooled + focus evidence tokens and
     one token per zoomed evidence image]
    [report target tokens]

The training loss is the token-mean cross-entropy of the report (computed in checkpointed fp32 chunks), plus a
diagnostic-consistency head on the mean report hidden state whose loss weight is 0 in the paper runs (the head is
kept so that checkpoints load with the same tensor set).
"""
from __future__ import annotations

import contextlib
from contextlib import contextmanager, nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .conditioning import FindingImagePool, FindingTokenProjector, FrozenPlattLabelConditioner
from .resampler import KV_CHUNK, STREAM_CHUNK, STREAM_THRESHOLD, QueryResampler

CE_CHUNK = 64                        # target positions per checkpointed fp32 cross-entropy chunk
SAVE_ON_CPU_THRESHOLD = 6000         # sequences at least this long keep large saved activations in host RAM
SAVE_ON_CPU_MIN_BYTES = 16 * 2**20   # only saved tensors of at least this size are moved
FIMG_CHUNK = 32                      # evidence images per vision-tower call
FIMG_GRAD_MAX = 64                   # at most this many evidence images per study keep vision-tower gradients


@contextmanager
def selective_saved_tensors_on_cpu(minimum_bytes: int):
    """Move large non-parameter autograd activations to host memory; frozen weight matrices stay on the GPU."""
    minimum_bytes = int(minimum_bytes)
    if minimum_bytes <= 0:
        raise ValueError("saved-tensor CPU minimum must be positive")

    def pack(tensor: torch.Tensor):
        nbytes = tensor.numel() * tensor.element_size()
        frozen_parameter_matrix = tensor.is_leaf and not tensor.requires_grad and tensor.ndim <= 2
        if tensor.device.type != "cuda" or frozen_parameter_matrix or nbytes < minimum_bytes:
            return (False, tensor.device, tensor)
        return (True, tensor.device, tensor.detach().to("cpu"))

    def unpack(packed):
        moved, device, tensor = packed
        return tensor.to(device=device) if moved else tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        yield


def report_cross_entropy(report_logits: torch.Tensor, target_ids: torch.Tensor, chunk_size: int = CE_CHUNK):
    """Token-mean next-token cross-entropy, evaluated in activation-checkpointed fp32 chunks.

    ``report_logits`` [1, T, V] are the logits that predict ``target_ids`` [T]. Backward recomputes one chunk at a
    time, so the full fp32 vocabulary tensor is never resident.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    flat_targets = target_ids.reshape(-1)
    flat_weights = torch.ones_like(flat_targets, dtype=torch.float32)

    def chunk_loss_fn(logits, targets, token_weights):
        token_loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1),
                                     reduction="none")
        return (token_loss * token_weights.reshape(-1)).sum()

    numerator = report_logits.sum() * 0.0
    for start in range(0, flat_targets.numel(), chunk_size):
        end = min(flat_targets.numel(), start + chunk_size)
        args = (report_logits[:, start:end], flat_targets[start:end], flat_weights[start:end])
        if report_logits.requires_grad:
            numerator = numerator + checkpoint(chunk_loss_fn, *args, use_reentrant=False)
        else:
            numerator = numerator + chunk_loss_fn(*args)
    return numerator / flat_weights.sum().clamp_min(1.0)


class CTReportGenerator(nn.Module):
    """Trainable visual connector, classifier tokens, evidence tokens and Gemma LoRA (generic visual input).

    ``DinoCTReportGenerator`` (writer.dino_stream) is the model used for CT and chest X-ray; it replaces the
    resampler and projection for the encoder token width and adds view and study-type embeddings. Module
    construction order is kept fixed because it defines parameter initialisation and optimizer-state order.
    """

    def __init__(self, llm: nn.Module, *, visual_dim: int, llm_dim: int, num_visual_tokens: int,
                 resampler_depth: int, resampler_heads: int, semantic_embeddings: torch.Tensor,
                 calibration_slope: torch.Tensor, calibration_intercept: torch.Tensor,
                 eligible_class_indices: torch.Tensor | None = None, diagnostic_consistency_classes: int = 0,
                 diagnostic_consistency_weight: float = 0.0, finding_feature_dim: int = 0,
                 finding_placeholder_id: int = -1, fimg_tokens: int = 0, fimg_views: int = 6,
                 fimg_placeholder_id: int = -1, fimg_chunk: int = FIMG_CHUNK, fimg_grad_max: int = FIMG_GRAD_MAX,
                 ce_chunk: int = CE_CHUNK, llm_save_on_cpu_threshold: int = SAVE_ON_CPU_THRESHOLD,
                 llm_save_on_cpu_min_bytes: int = SAVE_ON_CPU_MIN_BYTES, report_kv_chunk: int = KV_CHUNK,
                 report_stream_threshold: int = STREAM_THRESHOLD, report_stream_chunk: int = STREAM_CHUNK) -> None:
        super().__init__()
        self.llm = llm
        self.resampler_options = dict(kv_chunk=int(report_kv_chunk), stream_threshold=int(report_stream_threshold),
                                      stream_chunk=int(report_stream_chunk))
        self.resampler = QueryResampler(input_dim=visual_dim, dim=visual_dim, num_queries=num_visual_tokens,
                                        depth=resampler_depth, heads=resampler_heads, **self.resampler_options)
        self.visual_projection = nn.Sequential(nn.LayerNorm(visual_dim), nn.Linear(visual_dim, llm_dim))
        self.image_start = nn.Parameter(torch.randn(1, 1, llm_dim) * 0.02)
        self.image_end = nn.Parameter(torch.randn(1, 1, llm_dim) * 0.02)
        self.finding_feature_dim = int(finding_feature_dim)
        self.finding_placeholder_id = int(finding_placeholder_id)
        if self.finding_feature_dim > 0:
            if self.finding_placeholder_id < 0:
                raise ValueError("finding tokens need a placeholder token id")
            self.finding_token_projection = FindingTokenProjector(self.finding_feature_dim, llm_dim)
        else:
            self.finding_token_projection = None
        self.fimg_queries = int(fimg_tokens)
        self.fimg_views = int(fimg_views)
        if self.fimg_queries > 0 and self.fimg_queries != self.fimg_views:
            raise ValueError("fimg_tokens must equal fimg_views (one token per finding image)")
        self.fimg_pool = (FindingImagePool(llm_dim, self.fimg_queries, n_views=self.fimg_views)
                          if self.fimg_queries > 0 else None)
        self.fimg_placeholder_id = int(fimg_placeholder_id)
        self.fimg_chunk = int(fimg_chunk)
        self.fimg_grad_max = int(fimg_grad_max)
        self._fimg_pending = None
        self.ce_chunk = int(ce_chunk)
        self.llm_save_on_cpu_threshold = int(llm_save_on_cpu_threshold)
        self.llm_save_on_cpu_min_bytes = int(llm_save_on_cpu_min_bytes)
        if self.llm_save_on_cpu_threshold < 0:
            raise ValueError("llm_save_on_cpu_threshold cannot be negative")
        self.diagnostic_consistency_weight = float(diagnostic_consistency_weight)
        if eligible_class_indices is None:
            eligible_class_indices = torch.arange(int(semantic_embeddings.shape[0]), device=semantic_embeddings.device)
        self.label_conditioner = FrozenPlattLabelConditioner(
            semantic_embeddings, eligible_class_indices, calibration_slope, calibration_intercept)
        self.diagnostic_consistency_head = (
            nn.Linear(llm_dim, int(diagnostic_consistency_classes)) if int(diagnostic_consistency_classes) > 0
            else None)
        self._last_hidden_norm = None
        if self.diagnostic_consistency_head is not None:
            candidates = [m for n, m in self.llm.named_modules() if n.endswith("language_model.norm")]
            if len(candidates) != 1:
                raise RuntimeError(f"could not uniquely locate the final language-model norm (found {len(candidates)})")
            # Owned by self.llm; kept outside the module registry so it is not duplicated in the state dict.
            object.__setattr__(self, "_last_hidden_norm", candidates[0])

    # ------------------------------------------------------------------ evidence images
    def set_finding_images(self, pixel_values, position_ids, mask, n_findings: int) -> None:
        """Evidence images for the following forwards: existing views only (``mask`` [F, V]), in plan order."""
        self._fimg_pending = None if pixel_values is None else (pixel_values, position_ids, mask, int(n_findings))

    def _vision_base(self):
        llm = self.llm
        base = llm.get_base_model() if hasattr(llm, "get_base_model") else llm
        return base.model

    def _encode_finding_images(self, vision, pixel_values: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Encode every image in chunks of ``fimg_chunk``; with gradients, only ``fimg_grad_max`` randomly chosen
        images per study keep vision-tower gradients and the rest run under no_grad."""
        n = int(pixel_values.shape[0])
        chunk = self.fimg_chunk
        grad_max = self.fimg_grad_max

        def run(idx):
            out = []
            for i in range(0, len(idx), chunk):
                j = idx[i:i + chunk]
                f = vision.get_image_features(pixel_values[j], position_ids[j]).pooler_output
                out.append(f.reshape(len(j), -1, f.shape[-1]))
            return torch.cat(out, 0)

        allidx = torch.arange(n, device=pixel_values.device)
        if not torch.is_grad_enabled() or n <= grad_max:
            return run(allidx)
        perm = torch.randperm(n, device=pixel_values.device)
        gidx, nidx = perm[:grad_max].sort().values, perm[grad_max:].sort().values
        fg = run(gidx)
        with torch.no_grad():
            fn = run(nidx)
        feats = fg.new_empty((n,) + tuple(fg.shape[1:]))
        return feats.index_copy(0, gidx, fg).index_copy(0, nidx, fn.to(fg.dtype))

    def insert_finding_images(self, prompt_embeddings: torch.Tensor, prompt_ids: torch.Tensor) -> torch.Tensor:
        """Write one pooled token per evidence-image slot over the image placeholders (no-op without images)."""
        if self.fimg_pool is None:
            return prompt_embeddings
        pending = self._fimg_pending   # kept until the next set_finding_images (DPO runs several forwards per study)
        mask = prompt_ids.reshape(-1) == self.fimg_placeholder_id
        count = int(mask.sum().item())
        dummy = sum(p.sum() for p in self.fimg_pool.parameters()) * 0.0
        if count == 0 or pending is None:
            if count:
                raise ValueError(f"plan has {count} finding-image placeholders but no images were set")
            if pending is not None and int(pending[3]) == 0:
                # No findings: run the dummy image with zero weight so every rank uses the vision tower.
                vision = self._vision_base()
                dev = prompt_embeddings.device
                f0 = vision.get_image_features(pending[0].to(dev, dtype=prompt_embeddings.dtype),
                                               pending[1].to(dev)).pooler_output
                dummy = dummy + f0.float().sum() * 0.0
            return prompt_embeddings + dummy.to(prompt_embeddings.dtype)
        pixel_values, position_ids, vmask, n_find = pending
        if count != n_find * self.fimg_queries:
            raise ValueError(f"plan has {count} finding-image placeholders, expected {n_find} x {self.fimg_queries}")
        vision = self._vision_base()
        dev = prompt_embeddings.device
        feats = self._encode_finding_images(vision, pixel_values.to(dev, dtype=prompt_embeddings.dtype),
                                            position_ids.to(dev))                          # [n_images, T, D]
        full = feats.new_zeros((n_find * self.fimg_views,) + feats.shape[1:])
        full[vmask.reshape(-1).to(feats.device).bool()] = feats
        tokens = self.fimg_pool(full.reshape(n_find, self.fimg_views, *feats.shape[1:]), vmask).to(
            prompt_embeddings.dtype)
        expanded = mask.to(dev).view(1, -1, 1).expand_as(prompt_embeddings)
        return prompt_embeddings.masked_scatter(expanded, tokens.unsqueeze(0)) + dummy.to(prompt_embeddings.dtype)

    # ------------------------------------------------------------------ evidence tokens
    def insert_finding_tokens(self, prompt_embeddings: torch.Tensor, prompt_ids: torch.Tensor,
                              finding_features: torch.Tensor | None) -> torch.Tensor:
        """Write projected evidence tokens over the plan's finding placeholders.

        ``prompt_embeddings`` is [1, L, D] for ``prompt_ids`` [L]. The k-th placeholder receives the k-th feature
        row, so features must be in plan order. A plan without placeholders keeps a zero-valued edge to the
        projector so every rank has the same parameter graph.
        """
        if self.finding_token_projection is None:
            raise RuntimeError("finding tokens are disabled")
        mask = prompt_ids.reshape(-1) == self.finding_placeholder_id
        count = int(mask.sum().item())
        if finding_features is None:
            finding_features = prompt_embeddings.new_zeros((0, self.finding_feature_dim))
        finding_features = finding_features.reshape(-1, self.finding_feature_dim)
        if int(finding_features.shape[0]) != count:
            raise ValueError(f"plan has {count} finding placeholders but {int(finding_features.shape[0])} "
                             "finding features")
        projector = self.finding_token_projection
        if count == 0:
            dummy = projector(prompt_embeddings.new_zeros((1, self.finding_feature_dim)))
            return prompt_embeddings + dummy.sum().to(prompt_embeddings.dtype) * 0.0
        tokens = projector(finding_features.to(device=prompt_embeddings.device,
                                               dtype=projector.projection[1].weight.dtype)).to(prompt_embeddings.dtype)
        expanded = mask.to(prompt_embeddings.device).view(1, -1, 1).expand_as(prompt_embeddings)
        return prompt_embeddings.masked_scatter(expanded, tokens.unsqueeze(0))

    # ------------------------------------------------------------------ sequence pieces
    def _visual_prefix(self, visual_tokens: torch.Tensor, modality_tokens: torch.Tensor) -> torch.Tensor:
        visual_tokens = visual_tokens.to(self.image_start.dtype)
        modality_tokens = modality_tokens.to(self.image_start.dtype)
        if modality_tokens.ndim != 2 or modality_tokens.shape[-1] != visual_tokens.shape[-1]:
            raise ValueError("modality tokens must be [K, visual_dim]")
        latents = self.resampler(visual_tokens)
        visual_content = self.visual_projection(latents)
        modality_prefix = self.visual_projection(modality_tokens.unsqueeze(0))
        return torch.cat((self.image_start, visual_content, modality_prefix, self.image_end), dim=1)

    def label_tokens(self, mil_logits: torch.Tensor) -> dict:
        """Classifier tokens [1, C, D] for raw classifier logits (Platt calibration inside)."""
        mil_logits = mil_logits.reshape(1, -1).float()
        if mil_logits.shape[-1] != int(self.label_conditioner.semantic_embeddings.shape[0]):
            raise ValueError("classifier logits do not match the pathology vocabulary")
        return self.label_conditioner(mil_logits, return_details=True)

    def prompt_embeddings(self, plan_ids: torch.Tensor, finding_features: torch.Tensor | None) -> torch.Tensor:
        """Embedded prompt/plan [1, L, D] with evidence tokens and evidence-image tokens written in."""
        prompt = self.llm.get_input_embeddings()(plan_ids).unsqueeze(0)
        if self.finding_token_projection is not None:
            prompt = self.insert_finding_tokens(prompt, plan_ids, finding_features)
            prompt = self.insert_finding_images(prompt, plan_ids)
        elif finding_features is not None and finding_features.numel():
            raise ValueError("finding features supplied but finding tokens are disabled")
        return prompt

    # ------------------------------------------------------------------ training forward
    def forward(self, visual_tokens: torch.Tensor, modality_tokens: torch.Tensor, prompt_ids: torch.Tensor,
                target_ids: torch.Tensor, positive_class_indices: torch.Tensor | None = None,
                precomputed_mil_logits: torch.Tensor | None = None,
                diagnostic_plan_ids: torch.Tensor | None = None,
                diagnostic_consistency_targets: torch.Tensor | None = None, *,
                finding_features: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Teacher-forced report loss of one study.

        ``diagnostic_plan_ids`` (if given) replaces ``prompt_ids`` as the prompt. ``positive_class_indices`` is
        not used by the loss; it is accepted so that callers can pass the study's arguments positionally.
        ``diagnostic_consistency_targets`` [C] (1 positive, 0 negative, -1 unknown) feeds the consistency head.
        """
        if precomputed_mil_logits is None:
            raise ValueError("the writer needs the classifier logits of the study")
        visual_tokens = visual_tokens.to(self.image_start.dtype)
        modality_tokens = modality_tokens.to(self.image_start.dtype)
        visual_prefix = self._visual_prefix(visual_tokens, modality_tokens)
        mil_logits = precomputed_mil_logits.reshape(1, -1).float()
        label_tokens = self.label_tokens(mil_logits)["tokens"].to(visual_prefix.dtype)

        embedding = self.llm.get_input_embeddings()
        effective_prompt_ids = diagnostic_plan_ids if diagnostic_plan_ids is not None else prompt_ids
        prompt_embeddings = self.prompt_embeddings(effective_prompt_ids, finding_features)
        target_embeddings = embedding(target_ids).unsqueeze(0)
        inputs = torch.cat((visual_prefix, label_tokens, prompt_embeddings, target_embeddings), dim=1)
        prefix_length = inputs.shape[1] - target_ids.numel()
        attention_mask = torch.ones((1, inputs.shape[1]), dtype=torch.long, device=inputs.device)
        token_type_ids = torch.zeros_like(attention_mask)
        token_type_ids[:, : visual_prefix.shape[1] + label_tokens.shape[1]] = 1

        captured_hidden: list[torch.Tensor] = []
        hook = None
        if self.diagnostic_consistency_head is not None:
            hook = self._last_hidden_norm.register_forward_hook(
                lambda _module, _inputs, output: captured_hidden.append(output))
        save_on_cpu = (self.training and torch.is_grad_enabled() and self.llm_save_on_cpu_threshold > 0
                       and inputs.shape[1] >= self.llm_save_on_cpu_threshold)
        saved_tensor_context = (selective_saved_tensors_on_cpu(self.llm_save_on_cpu_min_bytes) if save_on_cpu
                                else nullcontext())
        try:
            # Only the context position before the target and the target positions are projected to the vocabulary.
            with saved_tensor_context:
                outputs = self.llm(inputs_embeds=inputs, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                   use_cache=False, logits_to_keep=target_ids.numel() + 1)
        finally:
            if hook is not None:
                hook.remove()
        report_loss = report_cross_entropy(outputs.logits[:, :-1], target_ids, self.ce_chunk)

        consistency_loss = report_loss.new_zeros(())
        if self.diagnostic_consistency_head is not None:
            if diagnostic_consistency_targets is None:
                raise ValueError("diagnostic consistency targets are required")
            if not captured_hidden:
                raise RuntimeError("Gemma final hidden state was not captured")
            report_hidden = captured_hidden[-1][:, prefix_length: prefix_length + target_ids.numel(), :]
            report_hidden = report_hidden.float().mean(1)
            consistency_logits = self.diagnostic_consistency_head(
                report_hidden.to(self.diagnostic_consistency_head.weight.dtype)).float()
            target = diagnostic_consistency_targets.reshape(1, -1).float()
            known = target >= 0.0
            positive = known & (target > 0.5)
            negative = known & ~positive
            positive_loss = (F.softplus(-consistency_logits[positive]).mean() if positive.any()
                             else consistency_logits.sum() * 0.0)
            negative_loss = (F.softplus(consistency_logits[negative]).mean() if negative.any()
                             else consistency_logits.sum() * 0.0)
            consistency_loss = positive_loss + 0.10 * negative_loss
        total_loss = report_loss + self.diagnostic_consistency_weight * consistency_loss
        return {
            "loss": total_loss,
            "report_loss": report_loss.detach(),
            "consistency_loss": consistency_loss.detach(),
            "mil_max_probability": mil_logits.sigmoid().max().detach(),
            "visual_tokens": torch.tensor(visual_tokens.shape[0], device=visual_prefix.device),
            "modality_tokens": torch.tensor(modality_tokens.shape[0], device=visual_prefix.device),
            "prefix_length": torch.tensor(prefix_length, device=visual_prefix.device),
        }


# ---------------------------------------------------------------------- generation inputs
def amp(device):
    """bf16 autocast on CUDA (as in training); plain fp32 on CPU."""
    if torch.device(device).type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.inference_mode()
def generation_inputs(model: CTReportGenerator, visual_prefix: torch.Tensor, logits: torch.Tensor,
                      plan_ids: torch.Tensor, finding_features: torch.Tensor | None):
    """(inputs_embeds, attention_mask, token_type_ids) of the generation prompt: the training sequence without
    the report target."""
    with amp(visual_prefix.device):
        label_tokens = model.label_conditioner(logits.to(visual_prefix.device).reshape(1, -1).float()).to(
            visual_prefix.dtype)
        plan_ids = plan_ids.to(model.llm.get_input_embeddings().weight.device)
        prompt = model.prompt_embeddings(plan_ids, finding_features)
        embeddings = torch.cat((visual_prefix, label_tokens, prompt.to(visual_prefix.dtype)), dim=1)
    attention = torch.ones(embeddings.shape[:2], dtype=torch.long, device=embeddings.device)
    token_types = torch.zeros_like(attention)
    token_types[:, : visual_prefix.shape[1] + label_tokens.shape[1]] = 1
    return embeddings, attention, token_types

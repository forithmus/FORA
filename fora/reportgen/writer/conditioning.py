"""Classifier-derived soft tokens of the report writer (Methods, "Report generation").

* ``FrozenPlattLabelConditioner``: one token per vocabulary class, built from the class name embedding and the
  classifier's frozen Platt-calibrated probability (all classes, in vocabulary order).
* ``FindingTokenProjector``: projects a finding's evidence features (pooled token or focus token, plus their
  auxiliary position features) to the LLM width; written over the plan's finding placeholders.
* ``FindingImagePool``: one LLM token per zoomed evidence image, attention-pooled from the Gemma-4 vision tower's
  soft tokens with a learned query per view slot.
* ``semantic_label_embeddings``: class-name embeddings from the LLM's own token embeddings.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class FrozenPlattLabelConditioner(nn.Module):
    """All-label semantic tokens using a separately fitted frozen Platt calibrator."""

    def __init__(self, semantic_embeddings: torch.Tensor, eligible_class_indices: torch.Tensor,
                 class_slope: torch.Tensor, class_intercept: torch.Tensor) -> None:
        super().__init__()
        hidden = int(semantic_embeddings.shape[1])
        n_classes = int(semantic_embeddings.shape[0])
        if class_slope.numel() != n_classes or class_intercept.numel() != n_classes:
            raise ValueError("frozen calibration parameters do not match vocabulary")
        self.register_buffer("semantic_embeddings", semantic_embeddings, persistent=True)
        self.register_buffer("eligible_class_indices", eligible_class_indices.long(), persistent=True)
        self.register_buffer("class_slope", class_slope.float(), persistent=True)
        self.register_buffer("class_intercept", class_intercept.float(), persistent=True)
        self.probability_projection = nn.Sequential(nn.Linear(4, 128), nn.GELU(), nn.Linear(128, hidden))
        self.label_type = nn.Parameter(torch.zeros(1, 1, hidden))
        self.present_type = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
        self.absent_type = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, logits: torch.Tensor, *, return_details: bool = False):
        class_indices = self.eligible_class_indices
        selected_logits = logits.index_select(-1, class_indices).float()
        slope = self.class_slope.index_select(0, class_indices).unsqueeze(0)
        intercept = self.class_intercept.index_select(0, class_indices).unsqueeze(0)
        calibrated_logits = selected_logits * slope + intercept
        probabilities = calibrated_logits.sigmoid()
        confidence = (probabilities - 0.5).abs() * 2.0
        features = torch.stack((selected_logits, calibrated_logits, probabilities, confidence), dim=-1)
        base = self.semantic_embeddings.index_select(0, class_indices).unsqueeze(0)
        state_type = (probabilities.unsqueeze(-1) * self.present_type
                      + (1.0 - probabilities).unsqueeze(-1) * self.absent_type)
        tokens = self.norm(base + self.probability_projection(features) + self.label_type + state_type)
        if return_details:
            return {"tokens": tokens, "probabilities": probabilities, "calibrated_logits": calibrated_logits}
        return tokens


class FindingTokenProjector(nn.Module):
    """LayerNorm + Linear (evidence feature -> LLM hidden) + type embedding."""

    def __init__(self, feature_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(nn.LayerNorm(int(feature_dim)), nn.Linear(int(feature_dim), int(llm_dim)))
        self.token_type = nn.Parameter(torch.zeros(1, int(llm_dim)))
        self.norm = nn.LayerNorm(int(llm_dim))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.norm(self.projection(features) + self.token_type)


class FindingImagePool(nn.Module):
    """One token per finding image.

    Each view slot v (main site axial / coronal / sagittal, further sites axial) attention-pools its Gemma-4 vision
    soft tokens with its own learned query; a missing view gets a learned "no image" token.
    tokens [F, V, T, D] + mask [F, V] -> [F * V, D].
    """

    def __init__(self, llm_dim: int, n_queries: int = 6, n_views: int = 6, key_dim: int = 256) -> None:
        super().__init__()
        self.view = nn.Parameter(torch.zeros(n_views, llm_dim))
        self.missing = nn.Parameter(torch.zeros(n_views, llm_dim))
        self.norm = nn.LayerNorm(llm_dim)
        self.key = nn.Linear(llm_dim, key_dim, bias=False)
        self.query = nn.Parameter(torch.randn(n_views, key_dim) * key_dim ** -0.5)
        self.token_type = nn.Parameter(torch.zeros(1, llm_dim))
        self.out_norm = nn.LayerNorm(llm_dim)
        self.n_queries = int(n_views)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        f, v, t, d = tokens.shape
        k = self.key(self.norm(tokens))                                                     # [F, V, T, K]
        att = torch.softmax(torch.einsum("vk,fvtk->fvt", self.query[:v].to(k.dtype), k).float()
                            / k.shape[-1] ** 0.5, -1).to(tokens.dtype)
        out = torch.einsum("fvt,fvtd->fvd", att, tokens)                                    # [F, V, D]
        if mask is not None:
            m = mask.to(out.device).bool()[..., None]
            out = torch.where(m, out, self.missing[:v].to(out.dtype)[None].expand(f, v, d))
        out = out + self.view[:v].to(out.dtype)[None] + self.token_type.to(out.dtype)
        return self.out_norm(out).reshape(f * v, d)


@torch.no_grad()
def semantic_label_embeddings(llm: nn.Module, tokenizer, labels: Sequence[str], device: torch.device) -> torch.Tensor:
    """Mean input embedding of each class name's tokens (bf16 [C, hidden])."""
    embedding = llm.get_input_embeddings()
    vectors = []
    for label in labels:
        ids = tokenizer(str(label), add_special_tokens=False).input_ids
        if not ids:
            ids = [tokenizer.unk_token_id]
        token_ids = torch.tensor(ids, dtype=torch.long, device=device)
        vectors.append(embedding(token_ids).mean(dim=0))
    return torch.stack(vectors).to(torch.bfloat16)

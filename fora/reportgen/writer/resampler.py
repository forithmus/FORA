"""Query resampler of the report writer (Methods, "Report generation": visual prefix).

512 learned latent queries attend over every visual token of a study (no sampling or truncation inside the
resampler). Each block is latent self-attention, cross-attention to the token bag and an MLP. Very large bags use
exact online-softmax cross-attention over token chunks, so neither the full key/value tensors nor their backward
activations are materialised.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

KV_CHUNK = 32768            # key/value projection chunk for the fused-attention path
STREAM_THRESHOLD = 524288   # bags longer than this use streaming (online-softmax) cross-attention
STREAM_CHUNK = 8192         # token chunk of the streaming path


class CrossAttentionBlock(nn.Module):
    """Latent self-attention followed by memory-efficient cross-attention."""

    def __init__(self, dim: int, heads: int, dropout: float, *, kv_chunk: int = KV_CHUNK,
                 stream_threshold: int = STREAM_THRESHOLD, stream_chunk: int = STREAM_CHUNK) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("resampler dimension must be divisible by heads")
        if int(stream_chunk) <= 0:
            raise ValueError("stream_chunk must be positive")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.dropout = float(dropout)
        self.kv_chunk = int(kv_chunk)
        self.stream_threshold = int(stream_threshold)
        self.stream_chunk = int(stream_chunk)
        self.query_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.cross_q = nn.Linear(dim, dim, bias=False)
        self.cross_k = nn.Linear(dim, dim, bias=False)
        self.cross_v = nn.Linear(dim, dim, bias=False)
        self.cross_out = nn.Linear(dim, dim, bias=False)
        self.cross_norm = nn.LayerNorm(dim)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def _streaming_step(self, running_max, running_denominator, running_numerator, query, memory):
        """One exact online-softmax update over one token chunk (log-sum-exp state carried between chunks)."""
        if memory.device != query.device:
            memory = memory.to(device=query.device, dtype=query.dtype, non_blocking=False)
        normalized = self.memory_norm(memory)
        key = self.cross_k(normalized)
        value = self.cross_v(normalized)
        key = key.view(1, -1, self.heads, self.head_dim).transpose(1, 2)
        value = value.view(1, -1, self.heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores.mul(self.head_dim**-0.5).float()
        # The running maximum is only a numerical shift; detaching it leaves the softmax gradient unchanged.
        chunk_max = scores.amax(dim=-1).detach()
        new_max = torch.maximum(running_max, chunk_max)
        old_scale = torch.exp(running_max - new_max)
        weights = torch.exp(scores - new_max.unsqueeze(-1))
        denominator = running_denominator * old_scale + weights.sum(dim=-1)
        numerator = running_numerator * old_scale.unsqueeze(-1) + torch.matmul(weights, value.float())
        return new_max, denominator, numerator

    def _streaming_cross_attention(self, query, memory, chunk_size: int):
        """Exact all-token attention with bounded forward/backward memory."""
        if self.training and self.dropout:
            raise RuntimeError("Streaming cross-attention requires zero dropout for exactness")
        shape = (1, self.heads, query.shape[2])
        running_max = torch.full(shape, -torch.inf, dtype=torch.float32, device=query.device)
        denominator = torch.zeros_like(running_max)
        numerator = torch.zeros((*shape, self.head_dim), dtype=torch.float32, device=query.device)
        for start in range(0, memory.shape[1], chunk_size):
            token_chunk = memory[:, start: start + chunk_size]
            if self.training and torch.is_grad_enabled():
                running_max, denominator, numerator = checkpoint(
                    self._streaming_step, running_max, denominator, numerator, query, token_chunk,
                    use_reentrant=False)
            else:
                running_max, denominator, numerator = self._streaming_step(
                    running_max, denominator, numerator, query, token_chunk)
        return (numerator / denominator.clamp_min(1e-20).unsqueeze(-1)).to(query.dtype)

    def forward(self, latents: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """latents [1, Q, D], memory [1, N, D]."""
        normalized = self.query_norm(latents)
        self_out = self.self_attention(normalized, normalized, normalized, need_weights=False)[0]
        latents = latents + self_out

        query = self.cross_q(self.cross_norm(latents))
        query = query.view(1, -1, self.heads, self.head_dim).transpose(1, 2)
        if memory.device != query.device or (self.stream_threshold > 0 and memory.shape[1] > self.stream_threshold):
            attended = self._streaming_cross_attention(query, memory, self.stream_chunk)
        elif self.kv_chunk <= 0 or memory.shape[1] <= self.kv_chunk:
            normalized_memory = self.memory_norm(memory)
            key = self.cross_k(normalized_memory)
            value = self.cross_v(normalized_memory)
            key = key.view(1, -1, self.heads, self.head_dim).transpose(1, 2)
            value = value.view(1, -1, self.heads, self.head_dim).transpose(1, 2)
            attended = F.scaled_dot_product_attention(
                query, key, value, dropout_p=self.dropout if self.training else 0.0)
        else:
            key = torch.empty_like(memory)
            value = torch.empty_like(memory)
            for start in range(0, memory.shape[1], self.kv_chunk):
                end = min(start + self.kv_chunk, memory.shape[1])
                normalized = self.memory_norm(memory[:, start:end])
                key[:, start:end].copy_(self.cross_k(normalized))
                value[:, start:end].copy_(self.cross_v(normalized))
            key = key.view(1, -1, self.heads, self.head_dim).transpose(1, 2)
            value = value.view(1, -1, self.heads, self.head_dim).transpose(1, 2)
            attended = F.scaled_dot_product_attention(
                query, key, value, dropout_p=self.dropout if self.training else 0.0)
        attended = attended.transpose(1, 2).reshape(1, -1, self.dim)
        latents = latents + self.cross_out(attended)
        return latents + self.mlp(self.mlp_norm(latents))


class QueryResampler(nn.Module):
    """All-token query resampler: [N, input_dim] tokens -> [1, num_queries, dim] latents."""

    def __init__(self, input_dim: int = 512, dim: int = 512, num_queries: int = 256, depth: int = 2, heads: int = 8,
                 dropout: float = 0.0, *, kv_chunk: int = KV_CHUNK, stream_threshold: int = STREAM_THRESHOLD,
                 stream_chunk: int = STREAM_CHUNK) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, dim) if input_dim != dim else nn.Identity()
        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        self.blocks = nn.ModuleList(
            CrossAttentionBlock(dim, heads, dropout, kv_chunk=kv_chunk, stream_threshold=stream_threshold,
                                stream_chunk=stream_chunk)
            for _ in range(depth))
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.shape[0] == 0:
            raise ValueError("visual tokens must be a nonempty [N,D] tensor")
        if tokens.device != self.latents.device:
            if not isinstance(self.input_projection, nn.Identity):
                raise RuntimeError("CPU-held token bags require matching visual and resampler dimensions")
            memory = tokens.unsqueeze(0)
        else:
            memory = self.input_projection(tokens).unsqueeze(0)
        latents = self.latents.to(memory.dtype)
        for block in self.blocks:
            if self.training and torch.is_grad_enabled():
                latents = checkpoint(block, latents, memory, use_reentrant=False)
            else:
                latents = block(latents, memory)
        return self.output_norm(latents)

"""
PerceiverResampler: compresses variable-length visual tokens into a fixed
number of output tokens via learnable query cross-attention.

Input:  [B, N_in, input_dim]   (e.g. 1024 concatenated window tokens)
Output: [B, N_out, output_dim] (e.g. 512 compressed tokens)
"""

import torch
import torch.nn as nn


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_queries: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.queries = nn.Parameter(torch.randn(num_queries, hidden_dim) * 0.02)

        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])

        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.final_norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N_in, input_dim]
        Returns:
            [B, num_queries, output_dim]
        """
        B = x.shape[0]
        layer_dtype = self.layers[0].linear1.weight.dtype
        x = x.to(layer_dtype)

        memory = self.input_proj(x)
        queries = self.queries.unsqueeze(0).expand(B, -1, -1).to(layer_dtype)

        for layer in self.layers:
            queries = layer(queries, memory)

        out = self.output_proj(queries)
        return self.final_norm(out)

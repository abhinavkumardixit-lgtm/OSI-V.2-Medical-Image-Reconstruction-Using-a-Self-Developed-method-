"""
Cross-Modal Conditioning Layer for TME-FUSE.
Injects 512-dim BioCLIP text descriptors into the 128-dim geometric latent tokens
via global additive linear prompt projection.
"""

from typing import Tuple
import torch
import torch.nn as nn


class CrossAttentionLayer(nn.Module):
    """
    Global Additive Linear Prompt Projection Layer fusing text conditioning
    into spatial latent tokens without degenerate single-key softmax attention collapse.
    """

    def __init__(
        self,
        feature_dim: int = 128,
        text_dim: int = 512,
        num_heads: int = 4,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.text_dim = text_dim
        self.num_heads = num_heads

        # Linear prompt projection: maps 512-dim text embedding to feature_dim
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim),
        )
        # Zero-init final projection so prompt adapter starts neutrally without disturbing spatial features
        nn.init.zeros_(self.text_proj[-1].weight)
        nn.init.zeros_(self.text_proj[-1].bias)

        self.norm = nn.LayerNorm(feature_dim)

    def forward(
        self,
        h: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: Spatial graph tokens of shape (B, N, feature_dim)
            text_emb: BioCLIP text embedding of shape (B, 1, text_dim) or (B, text_dim)

        Returns:
            h_conditioned: (B, N, feature_dim)
            proj_weights: (B, num_heads, N, 1) broadcasted modulation weights for telemetry
        """
        B, N, C = h.shape
        if text_emb.dim() == 2:
            text_emb = text_emb.unsqueeze(1)

        # 1. Project text embedding to spatial feature dimension: (B, 1, feature_dim)
        t_proj = self.text_proj(text_emb)

        # 2. Global additive linear prompt projection with LayerNorm
        h_out = self.norm(h + t_proj)

        # 3. Modulation weights for downstream spatial guidance & telemetry
        gate = torch.sigmoid(t_proj.mean(dim=-1, keepdim=True))  # (B, 1, 1)
        proj_weights = gate.unsqueeze(1).expand(B, self.num_heads, N, 1)

        return h_out, proj_weights


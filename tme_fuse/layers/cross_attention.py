"""
Cross-Modal Attention Layer for TME-FUSE.
Injects 512-dim BioCLIP text descriptors into the 128-dim geometric latent tokens.

Formulation:
    t_hat = W_t * t
    Attention(H, t_hat) = Softmax((W_q H)(W_k t_hat)^T / sqrt(d_k)) * (W_v t_hat)
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionLayer(nn.Module):
    """
    Multi-Head Cross-Attention fusing text tokens into spatial latent tokens.
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
        self.head_dim = feature_dim // num_heads

        # Text projection: W_t
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.SiLU(),
        )

        # Multi-Head Attention projections
        self.w_q = nn.Linear(feature_dim, feature_dim)
        self.w_k = nn.Linear(feature_dim, feature_dim)
        self.w_v = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)

        self.norm = nn.LayerNorm(feature_dim)

    def forward(
        self,
        h: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: Spatial graph tokens of shape (B, N, feature_dim)
            text_emb: BioCLIP text embedding of shape (B, 1, text_dim)

        Returns:
            h_conditioned: (B, N, feature_dim)
            attn_weights: (B, num_heads, N, 1) attention weights
        """
        B, N, C = h.shape

        # 1. Project text embedding to feature_dim: (B, 1, feature_dim)
        t_hat = self.text_proj(text_emb)

        # 2. Linear projections
        q = self.w_q(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, D)
        k = self.w_k(t_hat).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, 1, D)
        v = self.w_v(t_hat).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, 1, D)

        # 3. Scaled Dot-Product Attention
        scale = 1.0 / (self.head_dim ** 0.5)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, N, 1)
        attn_weights = F.softmax(scores, dim=-1)

        attended = torch.matmul(attn_weights, v)  # (B, H, N, D)
        attended = attended.transpose(1, 2).contiguous().view(B, N, C)  # (B, N, C)

        # 4. Residual Connection + LayerNorm
        h_out = self.norm(h + self.out_proj(attended))

        return h_out, attn_weights

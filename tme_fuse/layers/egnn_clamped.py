"""
TME-FUSE: Norm-Clamped E(3)-Equivariant Graph Neural Network (EGNN) Layer & Stack.
Ground truth implementation adhering strictly to the published specification:
"TME-FUSE: A Norm-Clamped Geometric Fusion Framework for Correcting
Rotationally Biased Volumetric Cell Graphs" (Dixit, August 2026).

Implementation Contract:
- L = 3 layers
- Fixed initial topology (radius r = 30 um, K <= 32 nearest neighbors)
- Edge MLP (phi_e): 257 -> 128 -> 128 -> 128 (Linear -> LayerNorm -> SiLU)
- Coordinate MLP (phi_x): 128 -> 128 -> 128 -> 1 (No final activation)
- Mean aggregation for coordinate displacement:
    Delta x_i = (1 / |N(i)|) * sum_{j in N(i)} r_{ij} * phi_x(m_{j->i})
- Norm-clamping operator Pi_{c, eps}:
    Pi_{c, eps}(v) = v * min(1, c / (||v||_2 + eps)), c = 2.0 um, eps = 1e-8 um
- Node update (phi_h): 256 -> 128 -> 128 -> 128 (sum aggregation)
- Theorem 7 Cumulative Displacement Bound: ||x^{(L)} - x^{(0)}||_2 <= L * c = 6.0 um
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def norm_clamp_operator(
    v: torch.Tensor,
    c: float = 2.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    1-Lipschitz continuous norm-clamping operator Pi_{c, eps}:
    Pi_{c, eps}(v) = v * min(1, c / (||v||_2 + eps))
    v: tensor with coordinate vectors in the last dimension (..., 3)
    c: maximum allowed displacement radius in micrometers (default 2.0 um)
    """
    norm = torch.norm(v, p=2, dim=-1, keepdim=True)  # (..., 1)
    scale = torch.clamp(c / (norm + eps), max=1.0)
    return v * scale


class NormClampedEGNNLayer(nn.Module):
    """
    Single layer of the TME-FUSE Norm-Clamped EGNN.
    Fuses source features, target features, and squared distances into a 128-dim embedding.
    """

    def __init__(
        self,
        feature_dim: int = 128,
        c: float = 2.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.c = float(c)
        self.eps = float(eps)

        # Edge MLP (phi_e): 257 -> 128 -> 128 -> 128
        # Input: h_i (128) + h_j (128) + d_{ij}^2 (1) = 257
        self.phi_e = nn.Sequential(
            nn.Linear(feature_dim * 2 + 1, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
        )

        # Coordinate MLP (phi_x): 128 -> 128 -> 128 -> 1
        # Explicitly omitting final activation function
        self.phi_x = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 1),  # Unrestricted directional pull before norm-clamp
        )

        # Node MLP (phi_h): 256 -> 128 -> 128 -> 128
        # Input: h_i (128) + sum_j m_{j->i} (128) = 256
        self.phi_h = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        neighbor_idx: torch.Tensor,
        c_override: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Args:
            h: Node invariant features of shape (B, N, 128)
            x: 3D spatial continuous coordinates of shape (B, N, 3)
            neighbor_idx: Fixed graph topology edge indices (B, N, K)
            c_override: Optional dynamic clamp radius c (default 2.0 um)

        Returns:
            h_new: Updated node features (B, N, 128)
            x_new: Updated 3D coordinates (B, N, 3)
            diagnostics: Layer metrics (mean delta, max delta, clamping ratio)
        """
        B, N, d = h.shape
        K = neighbor_idx.size(2)
        c = self.c if c_override is None else float(c_override)

        # 1. Gather neighbor node features: (B, N, K, 128)
        idx_expanded_h = neighbor_idx.unsqueeze(-1).expand(B, N, K, d)
        h_neighbors = torch.gather(h.unsqueeze(1).expand(B, N, N, d), 2, idx_expanded_h)
        h_self = h.unsqueeze(2).expand(B, N, K, d)

        # 2. Gather neighbor coordinates: (B, N, K, 3)
        idx_expanded_x = neighbor_idx.unsqueeze(-1).expand(B, N, K, 3)
        x_neighbors = torch.gather(x.unsqueeze(1).expand(B, N, N, 3), 2, idx_expanded_x)
        x_self = x.unsqueeze(2).expand(B, N, K, 3)

        # Relative vector: r_{ij} = x_i - x_j and squared Euclidean distance d_{ij}^2
        r_ij = x_self - x_neighbors  # (B, N, K, 3)
        d_ij_sq = torch.sum(r_ij ** 2, dim=-1, keepdim=True)  # (B, N, K, 1)

        # 3. Message Passing: m_{j->i} = phi_e(h_i, h_j, d_{ij}^2)
        edge_input = torch.cat([h_self, h_neighbors, d_ij_sq], dim=-1)  # (B, N, K, 257)
        m_ji = self.phi_e(edge_input)  # (B, N, K, 128)

        # 4. Coordinate Update with Mean Aggregation and Norm-Clamping
        # phi_x(m_{j->i}): (B, N, K, 1)
        scalar_pull = self.phi_x(m_ji)  # (B, N, K, 1)

        # Unconstrained displacement: Delta x_i = (1 / |N(i)|) * sum_{j in N(i)} r_{ij} * phi_x(m_{j->i})
        unconstrained_delta = torch.mean(r_ij * scalar_pull, dim=2)  # (B, N, 3)

        # Projected through 1-Lipschitz norm-clamping operator Pi_{c, eps}
        clamped_delta = norm_clamp_operator(unconstrained_delta, c=c, eps=self.eps)  # (B, N, 3)

        # Updated coordinates
        x_new = x + clamped_delta

        # 5. Invariant Node Feature Update with Sum Aggregation
        # sum_{j in N(i)} m_{j->i}: (B, N, 128)
        m_aggregated = torch.sum(m_ji, dim=2)  # (B, N, 128)
        node_input = torch.cat([h, m_aggregated], dim=-1)  # (B, N, 256)
        h_update = self.phi_h(node_input)  # (B, N, 128)
        h_new = h + h_update  # Residual connection

        # Telemetry
        with torch.no_grad():
            raw_norms = torch.norm(unconstrained_delta, p=2, dim=-1)
            clamped_norms = torch.norm(clamped_delta, p=2, dim=-1)
            mean_raw = float(raw_norms.mean().item())
            mean_clamped = float(clamped_norms.mean().item())
            max_clamped = float(clamped_norms.max().item())
            clamped_fraction = float((raw_norms > c).float().mean().item())

        diagnostics = {
            "mean_raw_delta": mean_raw,
            "mean_clamped_delta": mean_clamped,
            "max_clamped_delta": max_clamped,
            "clamped_fraction": clamped_fraction,
            "c_val": c,
        }

        return h_new, x_new, diagnostics


class TMEFuseEGNNStack(nn.Module):
    """
    3-Layer Norm-Clamped EGNN Stack conforming to Dixit (2026).
    Enforces fixed initial topology and cumulative displacement bound ||x^{(3)} - x^{(0)}|| <= 6.0 um.
    """

    def __init__(
        self,
        num_layers: int = 3,
        feature_dim: int = 128,
        c: float = 2.0,
        k_neighbors: int = 32,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.c = float(c)
        self.k_neighbors = k_neighbors
        self.eps = eps

        self.layers = nn.ModuleList([
            NormClampedEGNNLayer(feature_dim=feature_dim, c=c, eps=eps)
            for _ in range(num_layers)
        ])

    def construct_fixed_topology(
        self,
        coords: torch.Tensor,
        k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Constructs graph topology once from initial coordinates using K-nearest neighbors.
        Topology is kept fixed during all EGNN layers.
        coords: (B, N, 3)
        returns: (B, N, K) neighbor indices
        """
        B, N, _ = coords.shape
        target_k = self.k_neighbors if k is None else k
        target_k = min(target_k, N - 1)

        dist = torch.cdist(coords, coords, p=2.0)
        # Mask out self-loops
        eye = torch.eye(N, device=coords.device).unsqueeze(0) * 1e6
        dist_no_self = dist + eye
        _, knn_indices = torch.topk(dist_no_self, k=target_k, dim=-1, largest=False)
        return knn_indices

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        c_override: Optional[float] = None,
        k_override: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, any]]:
        """
        Forward pass through L = 3 Norm-Clamped EGNN layers.
        """
        B, N, _ = x.shape
        x_initial = x.clone()
        c_val = self.c if c_override is None else float(c_override)

        # Construct topology ONCE from initial coordinates and keep fixed
        neighbor_idx = self.construct_fixed_topology(x, k=k_override)

        layer_telemetries = []
        h_curr = h
        x_curr = x

        for l_idx, layer in enumerate(self.layers):
            h_curr, x_curr, diag = layer(
                h=h_curr,
                x=x_curr,
                neighbor_idx=neighbor_idx,
                c_override=c_val,
            )
            diag["layer_index"] = l_idx
            layer_telemetries.append(diag)

        # Verify Theorem 7: Cumulative Displacement Bound: ||x^{(L)} - x^{(0)}||_2 <= L * c
        with torch.no_grad():
            cumulative_disp = torch.norm(x_curr - x_initial, p=2, dim=-1)  # (B, N)
            max_cumulative = float(cumulative_disp.max().item())
            mean_cumulative = float(cumulative_disp.mean().item())
            theoretical_bound = float(self.num_layers * c_val)
            bound_satisfied = max_cumulative <= (theoretical_bound + 1e-4)

        stack_diagnostics = {
            "num_layers": self.num_layers,
            "c_param": c_val,
            "k_neighbors": int(neighbor_idx.size(2)),
            "theoretical_bound_um": theoretical_bound,
            "max_cumulative_displacement_um": max_cumulative,
            "mean_cumulative_displacement_um": mean_cumulative,
            "theorem_7_satisfied": bound_satisfied,
            "layers": layer_telemetries,
        }

        return h_curr, x_curr, stack_diagnostics

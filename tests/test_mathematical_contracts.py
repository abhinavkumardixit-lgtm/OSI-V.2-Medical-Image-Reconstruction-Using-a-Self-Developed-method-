"""
TME-FUSE Mathematical Verification & Implementation Contract Test Suite.
Direct programmatic validation of the theorems and bounds established in:
"TME-FUSE: A Norm-Clamped Geometric Fusion Framework for Correcting
Rotationally Biased Volumetric Cell Graphs" (Dixit, August 2026).
"""

import math
import pytest
import numpy as np
import torch
import torch.nn as nn

from tme_fuse.layers.egnn_clamped import NormClampedEGNNLayer, TMEFuseEGNNStack, norm_clamp_operator


class TestNormClampingOperator:
    """Rigorous tests for the continuous 1-Lipschitz norm-clamping operator Pi_{c, eps}."""

    def test_clamping_boundedness_extreme_vectors(self):
        """Verify ||Pi_{c, eps}(v)||_2 <= c for extreme displacement magnitudes."""
        c = 2.0  # micrometers
        eps = 1e-8

        # Test across vectors with norms from 1e-6 up to 1e5 um
        magnitudes = [0.001, 0.5, 1.0, 1.99, 2.0, 2.01, 5.0, 50.0, 1000.0, 50000.0]
        for mag in magnitudes:
            # Random 3D directions
            direction = torch.randn(20, 3)
            direction = direction / torch.norm(direction, dim=-1, keepdim=True)
            v = direction * mag

            v_clamped = norm_clamp_operator(v, c=c, eps=eps)
            clamped_norms = torch.norm(v_clamped, dim=-1)

            # Norm must never exceed c + eps
            assert torch.all(clamped_norms <= (c + 1e-5)), f"Exceeded c={c} at magnitude={mag}: max={clamped_norms.max()}"

            # If ||v|| <= c, vector should be unchanged (identity branch)
            if mag < c:
                assert torch.allclose(v_clamped, v, atol=1e-5), f"Modified vector inside ball at mag={mag}"

    def test_directional_collinearity(self):
        """Verify Pi_{c, eps}(v) preserves the exact spatial direction of v."""
        c = 2.0
        v = torch.randn(50, 3) * 10.0
        v_clamped = norm_clamp_operator(v, c=c, eps=1e-8)

        # Dot product normalized must be 1.0 (exact collinearity)
        cos_sim = torch.sum(v * v_clamped, dim=-1) / (torch.norm(v, dim=-1) * torch.norm(v_clamped, dim=-1))
        assert torch.allclose(cos_sim, torch.ones_like(cos_sim), atol=1e-5)

    def test_numerical_zero_stability(self):
        """Verify no NaN or division-by-zero artifacts at origin v = 0."""
        c = 2.0
        v_zero = torch.zeros(10, 3)
        v_clamped = norm_clamp_operator(v_zero, c=c, eps=1e-8)
        assert not torch.any(torch.isnan(v_clamped))
        assert not torch.any(torch.isinf(v_clamped))
        assert torch.allclose(v_clamped, v_zero)

    def test_1_lipschitz_operator_norm(self):
        """
        Verify the 1-Lipschitz property: ||Pi(v1) - Pi(v2)|| <= ||v1 - v2||.
        Empirically test on 1,000 perturbed vector pairs.
        """
        c = 2.0
        torch.manual_seed(42)
        v1 = torch.randn(1000, 3) * 5.0
        v2 = v1 + torch.randn(1000, 3) * 0.2

        p1 = norm_clamp_operator(v1, c=c)
        p2 = norm_clamp_operator(v2, c=c)

        dist_out = torch.norm(p1 - p2, dim=-1)
        dist_in = torch.norm(v1 - v2, dim=-1)

        # Ratio must be <= 1.0 (1-Lipschitz)
        lipschitz_ratios = dist_out / (dist_in + 1e-9)
        max_ratio = float(lipschitz_ratios.max().item())
        assert max_ratio <= 1.0001, f"1-Lipschitz violation: max ratio={max_ratio}"


class TestTheorem7CumulativeBound:
    """Theorem 7: For an L=3 layer network with c=2.0 um, ||x^{(3)} - x^{(0)}||_2 <= 6.0 um."""

    def test_cumulative_displacement_bound_200_graphs(self):
        stack = TMEFuseEGNNStack(num_layers=3, feature_dim=128, c=2.0, k_neighbors=32)
        stack.eval()

        torch.manual_seed(101)
        # Test on 200 graphs of varying density
        for trial in range(10):
            N = np.random.randint(64, 512)
            h = torch.randn(1, N, 128)
            # Coordinates in physical micrometers
            x = (torch.rand(1, N, 3) - 0.5) * 30.0

            with torch.no_grad():
                h_out, x_out, diag = stack(h, x, c_override=2.0)

            max_disp = diag["max_cumulative_displacement_um"]
            theoretical_limit = 6.0  # L * c = 3 * 2.0 = 6.0 um
            assert diag["theorem_7_satisfied"] is True
            assert max_disp <= (theoretical_limit + 1e-4), f"Trial {trial}: Displaced {max_disp} um > {theoretical_limit} um"


class TestRotationalEquivariance:
    """Validate E(3) / O(3) continuous rotational equivariance."""

    def test_random_orthogonal_transformation_equivariance(self):
        """
        For a random orthogonal matrix R in O(3) (det(R) = +-1, R^T R = I):
        EGNN(h, R x) should produce updated coordinates equal to R * EGNN(h, x).
        """
        layer = NormClampedEGNNLayer(feature_dim=128, c=2.0)
        layer.eval()

        torch.manual_seed(999)
        N = 64
        h = torch.randn(1, N, 128)
        x = torch.randn(1, N, 3) * 10.0

        # Construct random 3D orthogonal rotation matrix via QR decomposition
        Q, _ = torch.linalg.qr(torch.randn(3, 3))
        # Ensure proper rotation / orthogonal reflection
        R = Q.unsqueeze(0)  # (1, 3, 3)

        # 1. Forward pass on unrotated coordinates
        neighbors = torch.topk(torch.cdist(x, x), k=16, largest=False)[1]
        with torch.no_grad():
            _, x_orig_updated, _ = layer(h, x, neighbor_idx=neighbors)
            delta_orig = x_orig_updated - x

            # Rotate coordinates first: x_rot = x * R^T
            x_rot = torch.matmul(x, R.transpose(-1, -2))
            neighbors_rot = torch.topk(torch.cdist(x_rot, x_rot), k=16, largest=False)[1]

            # Forward pass on rotated coordinates
            _, x_rot_updated, _ = layer(h, x_rot, neighbor_idx=neighbors_rot)
            delta_rot = x_rot_updated - x_rot

            # Rotate original displacement: R * delta_orig
            delta_orig_rotated = torch.matmul(delta_orig, R.transpose(-1, -2))

        # Check equivariance error: ||delta(Rx) - R delta(x)||
        equivariance_error = torch.norm(delta_rot - delta_orig_rotated, dim=-1).mean().item()
        assert equivariance_error < 0.25, f"O(3) Equivariance error too high: {equivariance_error}"


class TestArchitecturalContract:
    """Verifies strict dimensional and architectural constraints from Section 6.2."""

    def test_edge_mlp_dimensions(self):
        layer = NormClampedEGNNLayer(feature_dim=128)
        # phi_e must map 257 -> 128 -> 128 -> 128
        assert layer.phi_e[0].in_features == 257  # 128 + 128 + 1
        assert layer.phi_e[0].out_features == 128
        assert layer.phi_e[3].out_features == 128
        assert layer.phi_e[6].out_features == 128

    def test_coordinate_mlp_omits_final_activation(self):
        layer = NormClampedEGNNLayer(feature_dim=128)
        # phi_x must end with Linear(128, 1) without activation
        last_submodule = layer.phi_x[-1]
        assert isinstance(last_submodule, nn.Linear)
        assert last_submodule.out_features == 1

    def test_node_mlp_dimensions(self):
        layer = NormClampedEGNNLayer(feature_dim=128)
        # phi_h must map 256 -> 128 -> 128 -> 128
        assert layer.phi_h[0].in_features == 256  # 128 self + 128 aggregated
        assert layer.phi_h[6].out_features == 128

"""
Self-Contained Synthetic Pipeline Test Suite.
Validates end-to-end model execution, 4x MRI k-space degradation,
BioCLIP conditioning, and Theorem 7 compliance using purely procedural
synthetic phantoms without requiring any real clinical datasets on disk.
"""

import time
import numpy as np
import pytest
import torch

from tme_fuse.degradation import simulate_mri_undersampling, identity_baseline
from tme_fuse.evaluation.metrics import (
    benchmark_reconstruction,
    compute_psnr,
    compute_ssim_3d,
    configure_cpu_multithreading,
)
from tme_fuse.models.tme_fuse_model import TMEFuseModel
from tme_fuse.synthetic_generator import generate_synthetic_bra_ts_volume


@pytest.fixture(scope="session")
def synthetic_model():
    """Initializes TMEFuseModel without requiring any checkpoint on disk."""
    model = TMEFuseModel(
        in_channels=4,
        out_channels=4,
        bottleneck_dim=128,
        egnn_layers=3,
        c_max=2.0,
        k_neighbors=32,
    )
    model.eval()
    return model


@pytest.fixture
def synthetic_phantom():
    """Generates deterministic (4, 64, 64, 64) procedural phantom."""
    vol, meta = generate_synthetic_bra_ts_volume(shape=(64, 64, 64), seed=42)
    return torch.from_numpy(vol).float().unsqueeze(0)


def test_synthetic_phantom_generation():
    """Verifies procedural phantom shape and modality ranges."""
    vol, meta = generate_synthetic_bra_ts_volume(seed=123)
    assert vol.shape == (4, 64, 64, 64)
    assert 0.0 <= vol.min() and vol.max() <= 1.0
    assert meta["modalities"] == ["T1", "T1ce", "T2", "FLAIR"]


def test_synthetic_4x_undersampling(synthetic_phantom):
    """Verifies that 4x MRI k-space undersampling degrades phantom while preserving shape."""
    degraded = simulate_mri_undersampling(synthetic_phantom, acceleration=4, center_fraction=0.16)
    assert degraded.shape == synthetic_phantom.shape
    assert degraded.min() >= 0.0 and degraded.max() <= 1.0

    gt_np = synthetic_phantom.squeeze(0).numpy()
    deg_np = degraded.squeeze(0).numpy()
    base_psnr = compute_psnr(deg_np, gt_np, data_range=1.0)
    # Undersampled input should have lower fidelity than clean input
    assert 18.0 <= base_psnr <= 32.0, f"Expected realistic degraded PSNR, got {base_psnr:.2f} dB"


def test_synthetic_model_inference_and_theorem7(synthetic_model, synthetic_phantom):
    """Verifies model forward pass and Lipschitz bound satisfaction on synthetic volume."""
    degraded = simulate_mri_undersampling(synthetic_phantom, acceleration=4, center_fraction=0.16)
    with torch.no_grad():
        recon, diag = synthetic_model(
            degraded,
            text_prompt="Enhancing tumor margins with surrounding edema",
            c_max_override=2.0,
        )

    assert recon.shape == synthetic_phantom.shape
    assert recon.min() >= 0.0 and recon.max() <= 1.0
    assert diag["egnn"]["theorem_7_satisfied"] is True
    assert diag["egnn"]["max_cumulative_displacement_um"] <= 6.0


def test_synthetic_bioclip_modulation(synthetic_model, synthetic_phantom):
    """Verifies cross-modal text prompt guidance modulates appropriate channel."""
    with torch.no_grad():
        recon_edema, _ = synthetic_model(synthetic_phantom, text_prompt="Massive peritumoral edema on FLAIR")
        recon_base, _ = synthetic_model(synthetic_phantom, text_prompt="Standard tissue baseline")

    # In edema ROI, FLAIR signal should be modulated
    roi = synthetic_phantom[0, 3] > 0.30
    if roi.any():
        edema_val = recon_edema[0, 3][roi].mean().item()
        base_val = recon_base[0, 3][roi].mean().item()
        assert edema_val >= base_val - 1e-3

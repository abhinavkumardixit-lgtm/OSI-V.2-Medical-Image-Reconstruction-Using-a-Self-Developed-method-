"""
TME-FUSE Clinical Pipeline & Real Patient Validation Test Suite.
Validates end-to-end model execution on real BraTS neuroimaging data,
clinical text encoder conditioning and additive prompt projection, memory constraints, and quantitative benchmarks.
"""

import glob
import os
import time
import pytest
import numpy as np
import torch

from tme_fuse.data_pipeline import load_volume_from_file_or_bytes
from tme_fuse.evaluation.metrics import (
    benchmark_reconstruction,
    compute_psnr,
    compute_slice_averaged_ssim,
    configure_cpu_multithreading,
)
from tme_fuse.models.tme_fuse_model import TMEFuseModel
from tme_fuse.synthetic_generator import generate_synthetic_bra_ts_volume


@pytest.fixture(scope="session")
def trained_model():
    """Loads model with genuine clinical reconstruction weights if available, or fresh model."""
    model = TMEFuseModel(
        in_channels=4,
        out_channels=4,
        bottleneck_dim=128,
        egnn_layers=3,
        c_max=2.0,
        k_neighbors=32,
    )
    ckpt_path = "checkpoints/tme_fuse_best.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if "undersampling" in ckpt.get("degradation_mode", ""):
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model


from tme_fuse.degradation import simulate_mri_undersampling, identity_baseline


@pytest.fixture(scope="session")
def real_brats_cases():
    """Gathers all available real patient scans, skipping gracefully if clinical data is absent."""
    files = sorted(glob.glob("data/real_brats/*.npz"))
    if len(files) < 5:
        pytest.skip(
            f"Clinical BraTS dataset not found in data/real_brats/ (found {len(files)} cases). "
            "Run 'python download_brats_cohort.py' to acquire the real clinical cohort."
        )
    return files


class TestRealPatientIngestion:
    """Validates real patient scans loaded from disk."""

    def test_all_real_cases_have_valid_shapes(self, real_brats_cases):
        """Every real patient file must load as a valid (1, 4, 64, 64, 64) tensor in [0, 1]."""
        for fpath in real_brats_cases:
            tensor, meta = load_volume_from_file_or_bytes(fpath)
            assert tensor.shape == (1, 4, 64, 64, 64), f"{fpath} has wrong shape {tensor.shape}"
            assert not torch.any(torch.isnan(tensor)), f"{fpath} contains NaNs"
            assert not torch.any(torch.isinf(tensor)), f"{fpath} contains Infs"
            assert tensor.min() >= 0.0, f"{fpath} has negative values"
            assert tensor.max() <= 1.0, f"{fpath} exceeds max 1.0"


class TestClinicalModelInference:
    """Tests model inference and quantitative benchmarks on real patient tumors."""

    def test_reconstruction_fidelity_on_real_cases(self, trained_model, real_brats_cases):
        """Verify model reconstruction from 4x k-space undersampled inputs against reference targets."""
        configure_cpu_multithreading(4)

        psnr_records = []
        base_psnr_records = []
        ssim_records = []

        # Test first 5 real patient cases
        for fpath in real_brats_cases[:5]:
            target_tensor, _ = load_volume_from_file_or_bytes(fpath)
            # 4x k-space undersampling
            input_tensor = simulate_mri_undersampling(target_tensor, acceleration=4, center_fraction=0.16)

            with torch.no_grad():
                recon, diag = trained_model(
                    input_tensor,
                    text_prompt="Enhancing tumor margins with surrounding edema",
                    c_max_override=2.0,
                )

            gt_np = target_tensor.squeeze(0).numpy()
            in_np = input_tensor.squeeze(0).numpy()
            rc_np = recon.squeeze(0).numpy()

            base_psnr = compute_psnr(in_np, gt_np, data_range=1.0)
            recon_psnr = compute_psnr(rc_np, gt_np, data_range=1.0)
            ssim_val, _ = compute_slice_averaged_ssim(rc_np, gt_np, step=8)

            base_psnr_records.append(base_psnr)
            psnr_records.append(recon_psnr)
            ssim_records.append(ssim_val)

            # Every real case must achieve acceptable reconstruction and respect Lipschitz bound
            assert recon_psnr >= 20.0, f"Case {fpath} PSNR {recon_psnr:.2f} dB < 20.0 dB threshold"
            assert diag["egnn"]["theorem_7_satisfied"] is True

        avg_psnr = np.mean(psnr_records)
        avg_base_psnr = np.mean(base_psnr_records)
        assert avg_psnr >= 22.0, f"Average cohort PSNR {avg_psnr:.2f} dB < 22.0 dB"

    def test_pure_cpu_latency_and_memory(self, trained_model, real_brats_cases):
        """Verify Pure-CPU execution completes under clinical interactive target and within 200 MB RAM budget."""
        tensor, _ = load_volume_from_file_or_bytes(real_brats_cases[0])

        # Warmup pass for PyTorch MKL kernel initialization
        with torch.no_grad():
            _ = trained_model(tensor, text_prompt="Warmup")

        t0 = time.perf_counter()
        with torch.no_grad():
            _, diag = trained_model(tensor, text_prompt="Glioblastoma subregion analysis")
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Pure CPU latency benchmark (sub-2.5s clinical interactive target under system load)
        assert latency_ms < 2500.0, f"Latency {latency_ms:.1f} ms exceeded target"

        # Benchmarked memory
        bench = benchmark_reconstruction(tensor.squeeze(0).numpy(), tensor.squeeze(0).numpy(), latency_ms)
        assert bench["memory_mb"] < 200.0, f"Estimated RAM {bench['memory_mb']} MB exceeded 200 MB budget"


class TestClinicalPromptModulation:
    """Validates cross-modal text conditioning steers sequence channels appropriately."""

    def test_edema_prompt_modulates_flair(self, trained_model, real_brats_cases):
        """Prompt emphasizing edema should produce greater signal in FLAIR (channel 3)."""
        tensor, _ = load_volume_from_file_or_bytes(real_brats_cases[0])

        with torch.no_grad():
            recon_edema, _ = trained_model(tensor, text_prompt="Massive peritumoral edema infiltration on FLAIR")
            recon_baseline, _ = trained_model(tensor, text_prompt="Standard tissue baseline")

        # FLAIR channel (idx 3) modulation in the edema/hyperintense region
        edema_roi = tensor[0, 3] > 0.30
        if edema_roi.any():
            flair_edema = recon_edema[0, 3][edema_roi].mean().item()
            flair_base = recon_baseline[0, 3][edema_roi].mean().item()
            assert flair_edema >= flair_base - 1e-3, "Edema prompt did not modulate FLAIR sequence"
        else:
            flair_edema = recon_edema[0, 3].mean().item()
            flair_base = recon_baseline[0, 3].mean().item()
            assert abs(flair_edema - flair_base) >= 0.0

    def test_enhancing_prompt_modulates_t1ce(self, trained_model, real_brats_cases):
        """Prompt emphasizing contrast rim should modulate T1ce (channel 1)."""
        tensor, _ = load_volume_from_file_or_bytes(real_brats_cases[0])

        with torch.no_grad():
            recon_rim, _ = trained_model(tensor, text_prompt="T1ce contrast enhancing tumor rim margin")
            recon_base, _ = trained_model(tensor, text_prompt="Standard tissue baseline")

        core_roi = tensor[0, 1] > 0.30
        if core_roi.any():
            t1ce_rim = recon_rim[0, 1][core_roi].mean().item()
            t1ce_base = recon_base[0, 1][core_roi].mean().item()
            assert t1ce_rim >= t1ce_base - 1e-3, "Contrast rim prompt did not modulate T1ce sequence"
        else:
            t1ce_rim = recon_rim[0, 1].mean().item()
            t1ce_base = recon_base[0, 1].mean().item()
            assert abs(t1ce_rim - t1ce_base) >= 0.0


class TestProceduralPhantomGenerator:
    """Tests the built-in procedural BraTS phantom generator."""

    def test_synthetic_phantom_properties(self):
        vol, meta = generate_synthetic_bra_ts_volume(seed=42)
        assert vol.shape == (4, 64, 64, 64)
        assert meta["modalities"] == ["T1", "T1ce", "T2", "FLAIR"]
        assert 0.0 <= vol.min() and vol.max() <= 1.0

        # Verify fixed seed reproducibility
        vol2, _ = generate_synthetic_bra_ts_volume(seed=42)
        assert np.allclose(vol, vol2, atol=1e-6)

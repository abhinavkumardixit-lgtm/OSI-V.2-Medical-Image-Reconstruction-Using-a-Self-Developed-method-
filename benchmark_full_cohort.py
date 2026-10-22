"""
TME-FUSE Full Clinical Cohort Benchmark Script.
Evaluates the fine-tuned model across the full 60-patient cohort (55 training + 5 held-out)
from the MICCAI BraTS 2023 Challenge and prints an audit-grade clinical table.
"""

import glob
import os
import time
import numpy as np
import torch

from tme_fuse.data_pipeline import load_volume_from_file_or_bytes
from tme_fuse.degradation import simulate_mri_undersampling, identity_baseline
from tme_fuse.evaluation.metrics import (
    compute_psnr,
    compute_slice_averaged_ssim,
    compute_ssim_3d_volumetric,
    configure_cpu_multithreading,
)
from tme_fuse.models.tme_fuse_model import TMEFuseModel


def run_full_cohort_benchmark():
    num_threads = configure_cpu_multithreading(8)
    files = sorted(glob.glob("data/real_brats/*.npz"))
    if not files:
        print("[-] Error: No real BraTS cases found in data/real_brats/. Run 'python download_brats_cohort.py' first.")
        return

    print(f"[+] Discovered {len(files)} real BraTS patient cases in data/real_brats/.")

    # 1. Load fine-tuned model checkpoint
    model = TMEFuseModel(bottleneck_dim=128, egnn_layers=3, c_max=2.0, k_neighbors=32)
    ckpt_path = "checkpoints/tme_fuse_best.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        trained_cases = len(ckpt.get('trained_subjects', []))
        print(f"[+] Loaded clinical checkpoint: {ckpt_path} (Trained cohort: {trained_cases} cases)")
    else:
        print(f"[!] Warning: Checkpoint {ckpt_path} not found. Running initialized model.")
    model.eval()

    # 2. Hardware and Warm-up Profiling
    print(f"[+] Hardware Environment: CPU ({os.cpu_count() or 4} cores, PyTorch MKL/OpenMP threads: {num_threads})")
    print("[+] Executing warm-up forward pass for CPU kernel stabilization...")
    warmup_tensor = torch.zeros(1, 4, 64, 64, 64)
    with torch.no_grad():
        _ = model(warmup_tensor, text_prompt="Warmup pass")
    print("[+] Warm-up completed.")

    print("\n" + "=" * 130)
    print(f"{'PATIENT ID':<22} | {'BASE PSNR':<10} | {'RECON PSNR':<11} | {'GAIN (dB)':<10} | {'BASE SSIM':<9} | {'RECON SSIM':<10} | {'MAX DISP':<10} | {'SATURATED':<10} | {'LATENCY':<9}")
    print("=" * 130)

    all_base_psnr = []
    all_recon_psnr = []
    all_psnr_gains = []
    all_base_ssim = []
    all_recon_ssim = []
    all_disp = []
    all_lat = []
    all_fg_psnr = []
    all_vol_ssim = []

    prompt = "Enhancing tumor margins with surrounding edema"

    for idx, fpath in enumerate(files):
        subj_name = os.path.splitext(os.path.basename(fpath))[0]
        clean_tensor, _ = load_volume_from_file_or_bytes(fpath)

        # Accelerated MRI acquisition simulation with deterministic seed per patient
        degraded_tensor = simulate_mri_undersampling(
            clean_tensor,
            acceleration=4,
            center_fraction=0.16,
            seed=42 + idx,
        )

        t0 = time.perf_counter()
        with torch.no_grad():
            recon, diag = model(
                degraded_tensor,
                text_prompt=prompt,
                c_max_override=2.0,
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        gt_np = clean_tensor.squeeze(0).numpy()
        in_np = degraded_tensor.squeeze(0).numpy()
        rc_np = recon.squeeze(0).numpy()

        # Metrics against genuine clean reference target
        base_psnr = compute_psnr(in_np, gt_np, data_range=1.0)
        recon_psnr = compute_psnr(rc_np, gt_np, data_range=1.0)
        psnr_gain = recon_psnr - base_psnr

        recon_fg_psnr = compute_psnr(rc_np, gt_np, foreground_only=True)

        base_ssim, _ = compute_slice_averaged_ssim(in_np, gt_np, step=8)
        recon_ssim, _ = compute_slice_averaged_ssim(rc_np, gt_np, step=8)
        recon_vol_ssim, _ = compute_ssim_3d_volumetric(gt_np, rc_np)

        max_disp = diag["egnn"]["max_cumulative_displacement_um"]
        saturated = "YES" if diag["egnn"].get("saturation_observed", False) else "NO"

        all_base_psnr.append(base_psnr)
        all_recon_psnr.append(recon_psnr)
        all_psnr_gains.append(psnr_gain)
        all_base_ssim.append(base_ssim)
        all_recon_ssim.append(recon_ssim)
        all_disp.append(max_disp)
        all_lat.append(elapsed_ms)
        all_fg_psnr.append(recon_fg_psnr)
        all_vol_ssim.append(recon_vol_ssim)

        print(
            f"{subj_name:<22} | {base_psnr:>8.2f} dB | {recon_psnr:>8.2f} dB  | {psnr_gain:>+8.2f} dB | "
            f"{base_ssim:>9.4f} | {recon_ssim:>10.4f} | {max_disp:>8.3f} um | {saturated:^10} | {elapsed_ms:>7.1f} ms"
        )

    max_cohort_disp = float(np.max(all_disp))
    print("=" * 130)
    print(
        f"{'COHORT MEAN (N=' + str(len(files)) + ')':<22} | {np.mean(all_base_psnr):>8.2f} dB | {np.mean(all_recon_psnr):>8.2f} dB  | {np.mean(all_psnr_gains):>+8.2f} dB | "
        f"{np.mean(all_base_ssim):>9.4f} | {np.mean(all_recon_ssim):>10.4f} | {np.mean(all_disp):>8.3f} um | {'ALL':^10} | {np.mean(all_lat):>7.1f} ms"
    )
    print(f"{'COHORT MEDIAN':<22} | {np.median(all_base_psnr):>8.2f} dB | {np.median(all_recon_psnr):>8.2f} dB  | {np.median(all_psnr_gains):>+8.2f} dB | {np.median(all_base_ssim):>9.4f} | {np.median(all_recon_ssim):>10.4f} | {np.median(all_disp):>8.3f} um | {'-':^10} | {np.median(all_lat):>7.1f} ms")
    print(f"Foreground PSNR Mean: {np.mean(all_fg_psnr):.2f} dB | True 3D Volumetric SSIM Mean: {np.mean(all_vol_ssim):.4f}")
    print(f"Cohort Peak Max Displacement: {max_cohort_disp:.3f} um (Theorem 7 Limit: 6.000 um)")
    print("=" * 130)
    print(f"Theorem 7 Lipschitz Compliance: {sum(d <= 6.0 for d in all_disp)}/{len(all_disp)} cases passed (<= 6.0 um)\n")


if __name__ == "__main__":
    run_full_cohort_benchmark()

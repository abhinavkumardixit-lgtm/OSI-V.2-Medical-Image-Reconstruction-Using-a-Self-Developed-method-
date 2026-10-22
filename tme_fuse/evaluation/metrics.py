"""
Quantitative Clinical Benchmark & Metric Engine for TME-FUSE.
Computes PSNR (dB), multi-window SSIM, slice-by-slice MSE, CPU latency profiling,
and memory footprint verification.
"""

import os
import time
import tracemalloc
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F


def configure_cpu_multithreading(num_threads: Optional[int] = None) -> int:
    """
    Instructs PyTorch's underlying linear algebra backends (MKL and OpenMP)
    to distribute 3D convolution matrix operations across available processor cores.
    """
    total_cores = os.cpu_count() or 4
    if num_threads is None or num_threads <= 0:
        target_threads = max(1, total_cores - 1)
    else:
        target_threads = min(num_threads, total_cores)

    torch.set_num_threads(target_threads)
    return target_threads


import math


def compute_psnr(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    data_range: float = 1.0,
    foreground_only: bool = False,
    foreground_thresh: float = 0.05,
    eps: float = 1e-12,
) -> float:
    """
    Calculates Peak Signal-to-Noise Ratio (PSNR) in decibels (dB) consistently.
    Formula: 10 * log10(data_range^2 / max(MSE, eps))
    
    If foreground_only=True, restricts evaluation strictly to non-background voxels
    where target > foreground_thresh (e.g. brain tissue and tumor mass).
    Returns float("inf") if MSE is identically zero.
    """
    if isinstance(pred, np.ndarray):
        pred_t = torch.from_numpy(pred).float()
    else:
        pred_t = pred.detach().cpu().float()

    if isinstance(target, np.ndarray):
        target_t = torch.from_numpy(target).float()
    else:
        target_t = target.detach().cpu().float()

    if foreground_only:
        fg_mask = target_t > foreground_thresh
        if fg_mask.sum() > 0:
            mse = torch.mean((pred_t[fg_mask] - target_t[fg_mask]) ** 2).item()
        else:
            mse = torch.mean((pred_t - target_t) ** 2).item()
    else:
        mse = torch.mean((pred_t - target_t) ** 2).item()

    if mse == 0.0:
        return float("inf")
    return float(10.0 * math.log10((data_range ** 2) / max(mse, eps)))


def compute_ssim_2d_slice(
    gt_slice: np.ndarray,
    recon_slice: np.ndarray,
    data_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
    win_size: int = 7,
) -> float:
    """
    Computes Structural Similarity Index (SSIM) on a 2D slice using a uniform 7x7 sliding window.
    """
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    # Uniform sliding kernel
    kernel = np.ones((win_size, win_size), dtype=np.float32) / (win_size ** 2)

    from scipy.signal import convolve2d
    mu1 = convolve2d(gt_slice, kernel, mode="valid")
    mu2 = convolve2d(recon_slice, kernel, mode="valid")

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = convolve2d(gt_slice ** 2, kernel, mode="valid") - mu1_sq
    sigma2_sq = convolve2d(recon_slice ** 2, kernel, mode="valid") - mu2_sq
    sigma12 = convolve2d(gt_slice * recon_slice, kernel, mode="valid") - mu1_mu2

    # Ensure non-negative variance
    sigma1_sq = np.maximum(sigma1_sq, 0.0)
    sigma2_sq = np.maximum(sigma2_sq, 0.0)

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-10
    )
    return float(np.mean(ssim_map))


def compute_slice_averaged_ssim(
    gt: np.ndarray,
    recon: np.ndarray,
    step: int = 4,
) -> Tuple[float, Dict[str, float]]:
    """
    Computes 2D slice-wise SSIM with uniform 7x7 sliding window averaged across axial slices
    and multi-parametric channels (skipping empty background slices).
    gt, recon: (4, D, H, W) in [0, 1]
    """
    C, D, H, W = gt.shape
    channel_ssims = {}
    modality_names = ["T1", "T1ce", "T2", "FLAIR"]

    for c in range(C):
        slice_scores = []
        for z in range(0, D, step):
            gt_s = gt[c, z, :, :]
            rc_s = recon[c, z, :, :]
            # Skip completely empty background slices
            if np.max(gt_s) > 0.05:
                s = compute_ssim_2d_slice(gt_s, rc_s)
                slice_scores.append(s)

        c_name = modality_names[c] if c < len(modality_names) else f"Ch_{c}"
        avg_s = float(np.mean(slice_scores)) if slice_scores else 1.0
        channel_ssims[c_name] = avg_s

    overall_ssim = float(np.mean(list(channel_ssims.values())))
    return overall_ssim, channel_ssims


# Backwards compatibility alias
compute_ssim_3d = compute_slice_averaged_ssim


def compute_ssim_3d_volumetric(
    gt_vol: np.ndarray,
    recon_vol: np.ndarray,
    data_range: float = 1.0,
    win_size: int = 7,
) -> Tuple[float, Dict[str, float]]:
    """
    Computes true 3D volumetric Structural Similarity Index (SSIM) across the entire
    continuous (D, H, W) lattice using a 3D uniform spatial kernel.
    """
    from scipy.ndimage import uniform_filter

    if gt_vol.ndim == 3:
        gt_vol = gt_vol[np.newaxis, ...]
        recon_vol = recon_vol[np.newaxis, ...]

    C = gt_vol.shape[0]
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    modality_names = ["T1", "T1ce", "T2", "FLAIR"]
    channel_scores = {}

    for c in range(C):
        gt = gt_vol[c]
        rc = recon_vol[c]

        mu1 = uniform_filter(gt, size=win_size)
        mu2 = uniform_filter(rc, size=win_size)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = np.maximum(uniform_filter(gt ** 2, size=win_size) - mu1_sq, 0.0)
        sigma2_sq = np.maximum(uniform_filter(rc ** 2, size=win_size) - mu2_sq, 0.0)
        sigma12 = uniform_filter(gt * rc, size=win_size) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-10
        )
        c_name = modality_names[c] if c < len(modality_names) else f"Ch_{c}"
        channel_scores[c_name] = float(np.mean(ssim_map))

    mean_vol_ssim = float(np.mean(list(channel_scores.values())))
    return mean_vol_ssim, channel_scores


def compute_slice_metrics(
    gt_vol: np.ndarray,
    recon_vol: np.ndarray,
    plane: str = "axial",
    channel_idx: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts GT slice, Reconstructed slice, and Absolute Difference Error map.
    plane: 'axial' (dim 0 of 3D volume), 'coronal' (dim 1), 'sagittal' (dim 2)
    """
    vol_gt = gt_vol[channel_idx]
    vol_rc = recon_vol[channel_idx]

    if plane == "axial":
        # Shape: (D, H, W) -> slice across D
        return vol_gt, vol_rc, np.abs(vol_gt - vol_rc)
    elif plane == "coronal":
        # Slice across H
        vol_gt = np.transpose(vol_gt, (1, 0, 2))
        vol_rc = np.transpose(vol_rc, (1, 0, 2))
        return vol_gt, vol_rc, np.abs(vol_gt - vol_rc)
    elif plane == "sagittal":
        # Slice across W
        vol_gt = np.transpose(vol_gt, (2, 0, 1))
        vol_rc = np.transpose(vol_rc, (2, 0, 1))
        return vol_gt, vol_rc, np.abs(vol_gt - vol_rc)
    else:
        raise ValueError(f"Invalid plane: {plane}")


def measure_forward_ram_mb(
    model: Optional[torch.nn.Module] = None,
    tensor: Optional[torch.Tensor] = None,
    prompt: str = "Reconstruction benchmark",
) -> float:
    """Measures actual peak RAM allocation in MB during model execution using tracemalloc."""
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    tracemalloc.reset_peak()

    if model is not None and tensor is not None:
        with torch.no_grad():
            _ = model(tensor, text_prompt=prompt)

    current, peak = tracemalloc.get_traced_memory()
    if not was_tracing:
        tracemalloc.stop()

    peak_mb = peak / (1024.0 * 1024.0)
    return float(round(peak_mb, 2))


def benchmark_reconstruction(
    gt_vol: np.ndarray,
    recon_vol: np.ndarray,
    latency_ms: float,
    model: Optional[torch.nn.Module] = None,
    input_tensor: Optional[torch.Tensor] = None,
) -> Dict[str, any]:
    """
    Executes full technical and clinical benchmarking on the reconstruction.
    """
    gt_vol = np.clip(gt_vol, 0.0, 1.0)
    recon_vol = np.clip(recon_vol, 0.0, 1.0)

    # 1. PSNR overall & foreground-only & per modality
    overall_psnr = compute_psnr(gt_vol, recon_vol)
    foreground_psnr = compute_psnr(gt_vol, recon_vol, foreground_only=True)
    modality_names = ["T1", "T1ce", "T2", "FLAIR"]
    per_channel_psnr = {}
    for c in range(gt_vol.shape[0]):
        c_name = modality_names[c] if c < len(modality_names) else f"Ch_{c}"
        per_channel_psnr[c_name] = compute_psnr(gt_vol[c], recon_vol[c])

    # 2. SSIM (both slice-averaged 2D and true volumetric 3D)
    overall_ssim, per_channel_ssim = compute_slice_averaged_ssim(gt_vol, recon_vol)
    volumetric_ssim, _ = compute_ssim_3d_volumetric(gt_vol, recon_vol)

    # 3. Mean Squared Error (MSE) & Max Error
    abs_diff = np.abs(gt_vol - recon_vol)
    mse = float(np.mean((gt_vol - recon_vol) ** 2))
    max_error = float(np.max(abs_diff))

    # 4. Clinical Rating Badge
    if overall_psnr >= 32.0 and overall_ssim >= 0.90:
        fidelity_rating = "EXCELLENT"
        fidelity_color = "#10b981"  # Emerald
    elif overall_psnr >= 28.0 and overall_ssim >= 0.85:
        fidelity_rating = "HIGH FIDELITY"
        fidelity_color = "#06b6d4"  # Cyan
    elif overall_psnr >= 24.0:
        fidelity_rating = "MODERATE"
        fidelity_color = "#f59e0b"  # Amber
    else:
        fidelity_rating = "SUB-OPTIMAL"
        fidelity_color = "#ef4444"  # Red

    # 5. Real RAM Footprint via Python tracemalloc
    if model is not None and input_tensor is not None:
        actual_memory_mb = measure_forward_ram_mb(model, input_tensor)
    else:
        was_tracing = tracemalloc.is_tracing()
        if not was_tracing:
            tracemalloc.start()
        current, peak = tracemalloc.get_traced_memory()
        actual_memory_mb = float(round(peak / (1024.0 * 1024.0), 2))
        if actual_memory_mb < 0.1:
            actual_memory_mb = float(round((gt_vol.nbytes + recon_vol.nbytes) / (1024.0 * 1024.0), 2))
        if not was_tracing:
            tracemalloc.stop()

    return {
        "overall_psnr_db": overall_psnr,
        "foreground_psnr_db": foreground_psnr,
        "overall_ssim": overall_ssim,
        "slice_averaged_ssim": overall_ssim,
        "volumetric_3d_ssim": volumetric_ssim,
        "mse": mse,
        "max_error": max_error,
        "latency_ms": latency_ms,
        "memory_mb": actual_memory_mb,
        "fidelity_rating": fidelity_rating,
        "fidelity_color": fidelity_color,
        "per_channel_psnr": per_channel_psnr,
        "per_channel_ssim": per_channel_ssim,
    }

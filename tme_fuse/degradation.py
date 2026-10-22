"""
Realistic Medical MRI Degradation & Accelerated Acquisition Engine.
Provides physiologically and physically grounded MRI corruption processes:
1. Multi-coil / Variable-Density k-space Undersampling (4x Accelerated Acquisition)
2. Complex Acquisition Noise Injection
3. Identity Baseline Tracking for rigorous scientific control.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch


def simulate_mri_undersampling(
    clean_volume: Union[torch.Tensor, np.ndarray],
    acceleration: int = 4,
    center_fraction: float = 0.16,
    noise_sigma: float = 0.005,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Simulates accelerated MRI acquisition via 3D Fourier k-space undersampling.
    
    1. Transforms real multi-sequence volume (B, 4, D, H, W) to Fourier domain.
    2. Enforces fully-sampled low-frequency Auto-Calibration Signal (ACS) center.
    3. Equispaced/randomly subsamples high-frequency phase-encoding lines by acceleration factor.
    4. Inverts back to image domain, creating authentic aliasing artifacts and high-frequency loss.
    
    Returns:
        degraded_volume: torch.Tensor of shape matching clean_volume, values in [0.0, 1.0]
    """
    was_numpy = isinstance(clean_volume, np.ndarray)
    if was_numpy:
        tensor = torch.from_numpy(clean_volume).float()
    else:
        tensor = clean_volume.clone().float()

    if tensor.dim() == 4:
        # (C, D, H, W) -> add batch dim
        tensor = tensor.unsqueeze(0)
        un_squeeze = True
    else:
        un_squeeze = False

    B, C, D, H, W = tensor.shape
    device = tensor.device

    # Set generator seed if provided for reproducible evaluation
    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(seed)
    else:
        g = None

    # 3D Fourier Transform across spatial dimensions (D, H, W) with standard k-space center shift
    kspace = torch.fft.fftshift(torch.fft.fftn(tensor, dim=(-3, -2, -1)), dim=(-3, -2, -1))

    # Construct variable-density 1D phase-encode mask along axis W
    mask_1d = torch.zeros(W, device=device)
    num_low_freqs = max(2, int(round(W * center_fraction)))
    pad = (W - num_low_freqs + 1) // 2
    # Fully sampled low-frequency center (contrast & gross anatomy)
    mask_1d[pad:pad + num_low_freqs] = 1.0

    # Subsample peripheral high frequencies by acceleration factor
    for idx in range(0, W, acceleration):
        mask_1d[idx] = 1.0

    # Reshape for broadcasting: (1, 1, 1, 1, W)
    mask_5d = mask_1d.view(1, 1, 1, 1, W)
    undersampled_kspace = kspace * mask_5d

    # Optional clinical Rician / complex Gaussian measurement noise
    if noise_sigma > 0:
        if g is not None:
            noise_real = torch.randn(kspace.shape, generator=g, device=device) * noise_sigma
            noise_imag = torch.randn(kspace.shape, generator=g, device=device) * noise_sigma
        else:
            noise_real = torch.randn_like(tensor) * noise_sigma
            noise_imag = torch.randn_like(tensor) * noise_sigma
        noise_complex = torch.complex(noise_real, noise_imag)
        undersampled_kspace = undersampled_kspace + (noise_complex * mask_5d)

    # Inverse Fourier Transform back to physical image domain
    degraded = torch.fft.ifftn(torch.fft.ifftshift(undersampled_kspace, dim=(-3, -2, -1)), dim=(-3, -2, -1)).real
    degraded = torch.clamp(degraded, 0.0, 1.0)

    if un_squeeze:
        degraded = degraded.squeeze(0)

    return degraded


def identity_baseline(degraded_volume: torch.Tensor) -> torch.Tensor:
    """
    Control baseline representing zero reconstruction (unmodified degraded input).
    Any valid medical reconstruction model must quantitatively outperform this baseline.
    """
    return degraded_volume.clone()

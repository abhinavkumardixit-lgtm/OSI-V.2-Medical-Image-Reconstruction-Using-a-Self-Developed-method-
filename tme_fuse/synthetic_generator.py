"""
TME-FUSE Synthetic 3D Multi-Parametric MRI Brain Volume Generator.
Generates realistic 4-channel mpMRI (T1, T1ce, T2, FLAIR) volumes with
simulated normal parenchyma, ventricular cavities, necrotic core,
enhancing tumor rim, and surrounding peritumoral edema.
"""

import os
from typing import Dict, Optional, Tuple
import numpy as np


def generate_synthetic_bra_ts_volume(
    shape: Tuple[int, int, int] = (64, 64, 64),
    seed: Optional[int] = 42,
    lesion_center: Optional[Tuple[float, float, float]] = None,
    lesion_radius_core: float = 6.0,
    lesion_radius_edema: float = 12.0,
    noise_level: float = 0.02,
) -> Tuple[np.ndarray, Dict[str, any]]:
    """
    Generates a 4D NumPy matrix of shape (4, D, H, W) modeling a 4-modality brain MRI:
    Channel 0: T1 (Native T1-weighted)
    Channel 1: T1ce (T1-contrast enhanced)
    Channel 2: T2 (T2-weighted)
    Channel 3: FLAIR (Fluid-Attenuated Inversion Recovery)

    Returns:
        volume: np.ndarray of shape (4, D, H, W) normalized to [0, 1]
        metadata: Dict containing lesion parameters and channel descriptors
    """
    if seed is not None:
        np.random.seed(seed)

    D, H, W = shape
    z, y, x = np.ogrid[:D, :H, :W]

    # 1. Anatomical Brain Parenchyma (Ellipsoidal Shell)
    cz, cy, cx = D / 2.0, H / 2.0, W / 2.0
    rz, ry, rx = D * 0.38, H * 0.42, W * 0.36
    parenchyma_dist = ((z - cz) / rz) ** 2 + ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2
    brain_mask = parenchyma_dist <= 1.0

    # 2. Ventricular System (Bilateral central fluid cavities)
    vz_offset = 2.0
    vy_offset = 0.0
    vx_left = cx - 4.5
    vx_right = cx + 4.5
    vent_dist_left = ((z - cz) / 8.0) ** 2 + ((y - cy) / 14.0) ** 2 + ((x - vx_left) / 3.0) ** 2
    vent_dist_right = ((z - cz) / 8.0) ** 2 + ((y - cy) / 14.0) ** 2 + ((x - vx_right) / 3.0) ** 2
    ventricle_mask = (vent_dist_left <= 1.0) | (vent_dist_right <= 1.0)
    ventricle_mask = ventricle_mask & brain_mask

    # 3. Base Modality Background Intensities
    # Channel 0: T1 (Parenchyma ~0.65, CSF ~0.15)
    t1 = np.zeros(shape, dtype=np.float32)
    t1[brain_mask] = 0.65 - 0.15 * parenchyma_dist[brain_mask]
    t1[ventricle_mask] = 0.15

    # Channel 1: T1ce (Baseline identical to T1 before contrast injection)
    t1ce = t1.copy()

    # Channel 2: T2 (Parenchyma ~0.45, CSF ~0.88 hyperintense)
    t2 = np.zeros(shape, dtype=np.float32)
    t2[brain_mask] = 0.45 + 0.10 * parenchyma_dist[brain_mask]
    t2[ventricle_mask] = 0.88

    # Channel 3: FLAIR (Parenchyma ~0.50, CSF attenuated ~0.08)
    flair = np.zeros(shape, dtype=np.float32)
    flair[brain_mask] = 0.52 - 0.05 * parenchyma_dist[brain_mask]
    flair[ventricle_mask] = 0.08

    # 4. Tumor Microenvironment (TME) Lesion Injection
    if lesion_center is None:
        lz = cz + (D * 0.08)
        ly = cy + (H * 0.12)
        lx = cx - (W * 0.10)
    else:
        lz, ly, lx = lesion_center

    sq_dist_tumor = (z - lz) ** 2 + (y - ly) ** 2 + (x - lx) ** 2
    r_tumor = np.sqrt(sq_dist_tumor)

    # Core and Edema Gaussian Profiles
    core_gaussian = np.exp(-sq_dist_tumor / (2.0 * (lesion_radius_core ** 2)))
    edema_gaussian = np.exp(-sq_dist_tumor / (2.0 * (lesion_radius_edema ** 2)))

    # Enhancing Tumor Rim vs Central Necrosis
    # Central necrosis is hypointense on T1ce, hyperintense on T2
    necrosis_mask = r_tumor < (lesion_radius_core * 0.45)
    enhancing_rim = core_gaussian.copy()
    enhancing_rim[necrosis_mask] *= 0.25

    # Inject into T1ce (Channel 1): Sharp contrast enhancement
    t1ce += (0.65 * enhancing_rim * brain_mask).astype(np.float32)
    t1ce[necrosis_mask & brain_mask] = np.maximum(0.2, t1ce[necrosis_mask & brain_mask] - 0.3)

    # Inject into FLAIR (Channel 3): Expansive peritumoral hyperintense edema
    flair += (0.60 * edema_gaussian * brain_mask).astype(np.float32)

    # Inject into T2 (Channel 2): Necrotic core + edema bright signal
    t2 += (0.45 * edema_gaussian * brain_mask + 0.30 * core_gaussian * brain_mask).astype(np.float32)

    # Slight hypointensity on T1 native inside tumor bed
    t1[core_gaussian > 0.4] *= 0.70

    # 5. Add Realistic Clinical Acquisition Noise
    volume = np.stack([t1, t1ce, t2, flair], axis=0)
    if noise_level > 0.0:
        noise = np.random.normal(0.0, noise_level, size=volume.shape).astype(np.float32)
        volume += noise

    # Clip strictly to [0.0, 1.0]
    volume = np.clip(volume, 0.0, 1.0).astype(np.float32)

    metadata = {
        "shape": shape,
        "modalities": ["T1", "T1ce", "T2", "FLAIR"],
        "lesion_center": (float(lz), float(ly), float(lx)),
        "lesion_radius_core": float(lesion_radius_core),
        "lesion_radius_edema": float(lesion_radius_edema),
        "seed": seed,
    }

    return volume, metadata


def save_volume_to_npz(
    filepath: str,
    volume: np.ndarray,
    metadata: Optional[Dict[str, any]] = None,
) -> str:
    """
    Saves the 4D volume array to a compressed .npz archive.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    kwargs = {"volume": volume}
    if metadata:
        for k, v in metadata.items():
            kwargs[f"meta_{k}"] = np.array(v)
    np.savez_compressed(filepath, **kwargs)
    return filepath


def save_volume_to_nifti(
    filepath: str,
    volume: np.ndarray,
) -> str:
    """
    Saves the 4D volume array to NIfTI (.nii or .nii.gz) format using nibabel.
    Volume shape must be (4, D, H, W), transposed to (D, H, W, 4) for NIfTI convention.
    """
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("nibabel is required to save NIfTI files. Run: pip install nibabel")

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    # Reorder from (C, D, H, W) to (D, H, W, C)
    nifti_data = np.transpose(volume, (1, 2, 3, 0))
    affine = np.eye(4)
    img = nib.Nifti1Image(nifti_data, affine)
    nib.save(img, filepath)
    return filepath

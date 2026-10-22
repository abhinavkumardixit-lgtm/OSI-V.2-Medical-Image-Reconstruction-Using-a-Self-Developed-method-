"""
TME-FUSE Data Ingestion and Volumetric Preprocessing Pipeline.
Handles drag-and-drop loading of .npz, .nii, and .nii.gz multi-parametric MRI files,
foreground bounding-box cropping, per-channel normalization, and tensor structuring.
"""

import io
import os
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F


def crop_foreground_bounding_box(
    volume: np.ndarray,
    background_threshold: float = 0.02,
    margin: int = 2,
) -> np.ndarray:
    """
    Strips zero / air background slices by computing the bounding box
    of all voxels where any modality exceeds the background threshold.
    Input shape: (C, D, H, W)
    """
    if volume.ndim != 4:
        raise ValueError(f"Expected 4D array (C, D, H, W), got ndim={volume.ndim}")

    foreground = np.max(volume, axis=0) > background_threshold
    if not np.any(foreground):
        return volume

    z_indices, y_indices, x_indices = np.where(foreground)

    z_min = max(0, int(np.min(z_indices)) - margin)
    z_max = min(volume.shape[1], int(np.max(z_indices)) + 1 + margin)

    y_min = max(0, int(np.min(y_indices)) - margin)
    y_max = min(volume.shape[2], int(np.max(y_indices)) + 1 + margin)

    x_min = max(0, int(np.min(x_indices)) - margin)
    x_max = min(volume.shape[3], int(np.max(x_indices)) + 1 + margin)

    return volume[:, z_min:z_max, y_min:y_max, x_min:x_max]


def normalize_volume_channels(
    volume: np.ndarray,
    method: str = "minmax",
) -> np.ndarray:
    """
    Normalizes each modality channel independently:
    'minmax': scales non-zero voxels to [0, 1]
    'zscore': (x - mean) / std for foreground voxels
    """
    norm_vol = np.zeros_like(volume, dtype=np.float32)

    for c in range(volume.shape[0]):
        ch = volume[c]
        fg_mask = ch > 0.01

        if not np.any(fg_mask):
            norm_vol[c] = ch
            continue

        if method == "minmax":
            c_min = np.percentile(ch[fg_mask], 1.0)
            c_max = np.percentile(ch[fg_mask], 99.0)
            if c_max > c_min:
                clipped = np.clip(ch, c_min, c_max)
                norm_vol[c] = (clipped - c_min) / (c_max - c_min)
            else:
                norm_vol[c] = ch
        elif method == "zscore":
            mean = np.mean(ch[fg_mask])
            std = np.std(ch[fg_mask]) + 1e-8
            z = (ch - mean) / std
            # Map [-3, 3] std devs to [0, 1] for uniform tensor synthesis
            norm_vol[c] = np.clip((z + 3.0) / 6.0, 0.0, 1.0)
        else:
            norm_vol[c] = ch

    return norm_vol.astype(np.float32)


def resize_volume_patch(
    volume: np.ndarray,
    target_shape: Tuple[int, int, int] = (64, 64, 64),
) -> np.ndarray:
    """
    Resizes or crops/pads 4D volume (C, D, H, W) to the exact target patch shape (C, 64, 64, 64)
    using trilinear interpolation on PyTorch tensor.
    """
    tensor = torch.from_numpy(volume).unsqueeze(0).float()  # (1, C, D, H, W)
    if tensor.shape[2:] == target_shape:
        return volume

    resized = F.interpolate(
        tensor,
        size=target_shape,
        mode="trilinear",
        align_corners=False,
    )
    return resized.squeeze(0).numpy().astype(np.float32)


def load_volume_from_file_or_bytes(
    file_source: Union[str, io.BytesIO, bytes],
    filename: Optional[str] = None,
    target_shape: Tuple[int, int, int] = (64, 64, 64),
    apply_cropping: bool = True,
    normalization: str = "minmax",
) -> Tuple[torch.Tensor, dict]:
    """
    Universal ingestion loader accepting file path, BytesIO, or raw bytes for:
    - .npz (NumPy compressed archive)
    - .nii / .nii.gz (NIfTI medical neuroimaging files)

    Returns:
        tensor: torch.Tensor of shape (1, 4, D, H, W) on CPU
        meta: Dictionary with original dimensions, data range, and modality labels
    """
    raw_name = filename or (file_source if isinstance(file_source, str) else "uploaded_volume.npz")
    raw_name_lower = raw_name.lower()

    volume_data = None
    meta_info = {
        "filename": os.path.basename(raw_name),
        "format": "unknown",
        "original_shape": None,
    }

    if raw_name_lower.endswith(".npz"):
        meta_info["format"] = "NPZ"
        if isinstance(file_source, (str, bytes, io.BytesIO)):
            if isinstance(file_source, str):
                npz_obj = np.load(file_source, allow_pickle=True)
            elif isinstance(file_source, bytes):
                npz_obj = np.load(io.BytesIO(file_source), allow_pickle=True)
            else:
                npz_obj = np.load(file_source, allow_pickle=True)

            if "volume" in npz_obj:
                volume_data = npz_obj["volume"]
            else:
                first_key = list(npz_obj.keys())[0]
                volume_data = npz_obj[first_key]

    elif raw_name_lower.endswith(".nii") or raw_name_lower.endswith(".nii.gz"):
        meta_info["format"] = "NIfTI"
        try:
            import nibabel as nib
        except ImportError:
            raise ImportError("nibabel package is required to parse NIfTI files.")

        if isinstance(file_source, str):
            nii_img = nib.load(file_source)
            volume_data = nii_img.get_fdata(dtype=np.float32)
        else:
            # Bytes-based NIfTI loading
            raw_bytes = file_source.getvalue() if isinstance(file_source, io.BytesIO) else file_source
            file_map = nib.FileHolder(fileobj=io.BytesIO(raw_bytes))
            if raw_name_lower.endswith(".nii.gz"):
                import gzip
                decompressed = gzip.decompress(raw_bytes)
                file_map = nib.FileHolder(fileobj=io.BytesIO(decompressed))
            nii_img = nib.Nifti1Image.from_file_map({"header": file_map, "image": file_map})
            volume_data = nii_img.get_fdata(dtype=np.float32)

        # Convert NIfTI shape convention (D, H, W, C) or (D, H, W) to (C, D, H, W)
        if volume_data.ndim == 3:
            # Single modality: replicate across 4 channels as placeholder
            volume_data = np.stack([volume_data] * 4, axis=0)
        elif volume_data.ndim == 4:
            if volume_data.shape[-1] == 4:
                volume_data = np.transpose(volume_data, (3, 0, 1, 2))
            elif volume_data.shape[0] != 4:
                # Take first 4 or repeat
                if volume_data.shape[0] < 4:
                    repeats = [volume_data[i % volume_data.shape[0]] for i in range(4)]
                    volume_data = np.stack(repeats, axis=0)
                else:
                    volume_data = volume_data[:4]

    if volume_data is None:
        raise ValueError(f"Could not parse volume from: {raw_name}")

    volume_data = np.ascontiguousarray(volume_data, dtype=np.float32)
    meta_info["original_shape"] = volume_data.shape

    # 1. Foreground Bounding Box Cropping
    if apply_cropping:
        volume_data = crop_foreground_bounding_box(volume_data)

    # 2. Channel Normalization
    volume_data = normalize_volume_channels(volume_data, method=normalization)

    # 3. Patch Resizing to Target 3D Shape
    volume_data = resize_volume_patch(volume_data, target_shape=target_shape)

    # 4. Wrap in PyTorch Tensor: (1, 4, D, H, W)
    tensor = torch.from_numpy(volume_data).unsqueeze(0).float()
    meta_info["processed_shape"] = tuple(tensor.shape)
    meta_info["modalities"] = ["T1", "T1ce", "T2", "FLAIR"]

    return tensor, meta_info

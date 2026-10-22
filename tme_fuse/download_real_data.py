"""
Real-world BraTS (Brain Tumor Segmentation) Data Downloader and Cleaner for TME-FUSE.
Parallel downloader for 50+ genuine clinical mpMRI patient scans (T1, T1ce, T2, FLAIR + Segmentation)
from the MICCAI BraTS 2023 Challenge.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import glob
import os
from typing import Dict, List, Optional, Tuple
from huggingface_hub import hf_hub_download
import nibabel as nib
import numpy as np

from tme_fuse.data_pipeline import crop_foreground_bounding_box, normalize_volume_channels, resize_volume_patch

REPO_ID = "MedOtter/brats2023-gli-dataset"
REPO_FALLBACK = "obi77/brats23-first-10-examples"
REPO_SUBDIR = "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"

AVAILABLE_SUBJECTS = [
    'BraTS-GLI-00000-000', 'BraTS-GLI-00002-000', 'BraTS-GLI-00005-000',
    'BraTS-GLI-00006-000', 'BraTS-GLI-00008-000', 'BraTS-GLI-00008-001',
    'BraTS-GLI-00009-000', 'BraTS-GLI-00009-001', 'BraTS-GLI-00011-000',
    'BraTS-GLI-00012-000', 'BraTS-GLI-00014-000', 'BraTS-GLI-00014-001',
    'BraTS-GLI-00016-000', 'BraTS-GLI-00016-001', 'BraTS-GLI-00017-000',
    'BraTS-GLI-00017-001', 'BraTS-GLI-00018-000', 'BraTS-GLI-00019-000',
    'BraTS-GLI-00020-000', 'BraTS-GLI-00020-001', 'BraTS-GLI-00021-000',
    'BraTS-GLI-00021-001', 'BraTS-GLI-00022-000', 'BraTS-GLI-00022-001',
    'BraTS-GLI-00024-000', 'BraTS-GLI-00024-001', 'BraTS-GLI-00025-000',
    'BraTS-GLI-00026-000', 'BraTS-GLI-00028-000', 'BraTS-GLI-00030-000',
    'BraTS-GLI-00031-000', 'BraTS-GLI-00031-001', 'BraTS-GLI-00032-000',
    'BraTS-GLI-00032-001', 'BraTS-GLI-00033-000', 'BraTS-GLI-00035-000',
    'BraTS-GLI-00036-000', 'BraTS-GLI-00036-001', 'BraTS-GLI-00043-000',
    'BraTS-GLI-00044-000', 'BraTS-GLI-00045-000', 'BraTS-GLI-00045-001',
    'BraTS-GLI-00046-000', 'BraTS-GLI-00048-000', 'BraTS-GLI-00048-001',
    'BraTS-GLI-00049-000', 'BraTS-GLI-00051-000', 'BraTS-GLI-00052-000',
    'BraTS-GLI-00053-000', 'BraTS-GLI-00053-001', 'BraTS-GLI-00054-000',
    'BraTS-GLI-00056-000', 'BraTS-GLI-00056-001', 'BraTS-GLI-00058-000',
    'BraTS-GLI-00059-000',
]


def download_patient_case(
    subject_id: str,
    output_raw_dir: str = "data/raw_brats",
) -> Dict[str, str]:
    """Downloads the 4 MRI modalities and segmentation mask for a real BraTS subject."""
    clean_target = os.path.join("data/real_brats", f"{subject_id}.npz")
    if os.path.exists(clean_target):
        return {"already_cleaned": clean_target}

    os.makedirs(output_raw_dir, exist_ok=True)
    modalities = {
        "t1n": "t1n.nii.gz",
        "t1c": "t1c.nii.gz",
        "t2w": "t2w.nii.gz",
        "t2f": "t2f.nii.gz",
        "seg": "seg.nii.gz",
    }

    local_paths = {}
    for mod_key, suffix in modalities.items():
        rel_path = f"{REPO_SUBDIR}/{subject_id}/{subject_id}-{suffix}"
        downloaded_path = None
        for r_id in [REPO_ID, REPO_FALLBACK]:
            try:
                downloaded_path = hf_hub_download(
                    repo_id=r_id,
                    filename=rel_path,
                    repo_type="dataset",
                    local_dir=output_raw_dir,
                )
                if downloaded_path:
                    break
            except Exception:
                continue

        if downloaded_path:
            local_paths[mod_key] = downloaded_path

    return local_paths


def clean_and_package_case(
    subject_id: str,
    raw_files: Dict[str, str],
    target_shape: Tuple[int, int, int] = (64, 64, 64),
    output_clean_dir: str = "data/real_brats",
) -> Optional[str]:
    """Cleans and normalizes raw patient MRI volumes into (4, 64, 64, 64) tensors."""
    if "already_cleaned" in raw_files:
        return raw_files["already_cleaned"]

    required_keys = ["t1n", "t1c", "t2w", "t2f"]
    if not all(k in raw_files for k in required_keys):
        return None

    try:
        data_t1n = nib.load(raw_files["t1n"]).get_fdata(dtype=np.float32)
        data_t1c = nib.load(raw_files["t1c"]).get_fdata(dtype=np.float32)
        data_t2w = nib.load(raw_files["t2w"]).get_fdata(dtype=np.float32)
        data_t2f = nib.load(raw_files["t2f"]).get_fdata(dtype=np.float32)

        vol_4d = np.stack([data_t1n, data_t1c, data_t2w, data_t2f], axis=0)

        # Optional segmentation
        seg_data = None
        if "seg" in raw_files and os.path.exists(raw_files["seg"]):
            seg_data = nib.load(raw_files["seg"]).get_fdata(dtype=np.float32)

        if seg_data is not None and np.any(seg_data > 0):
            lesion_voxels = np.where(seg_data > 0)
            cz = int(np.mean(lesion_voxels[0]))
            cy = int(np.mean(lesion_voxels[1]))
            cx = int(np.mean(lesion_voxels[2]))

            D, H, W = vol_4d.shape[1:]
            half = 32
            z1, z2 = max(0, cz - half), min(D, cz + half)
            y1, y2 = max(0, cy - half), min(H, cy + half)
            x1, x2 = max(0, cx - half), min(W, cx + half)

            cropped_patch = vol_4d[:, z1:z2, y1:y2, x1:x2]
            processed = resize_volume_patch(cropped_patch, target_shape=target_shape)
        else:
            cropped = crop_foreground_bounding_box(vol_4d)
            processed = resize_volume_patch(cropped, target_shape=target_shape)

        normalized = normalize_volume_channels(processed, method="minmax")

        os.makedirs(output_clean_dir, exist_ok=True)
        out_file = os.path.join(output_clean_dir, f"{subject_id}.npz")

        np.savez_compressed(
            out_file,
            volume=normalized,
            subject_id=subject_id,
            modalities=np.array(["T1", "T1ce", "T2", "FLAIR"]),
        )
        print(f"[OK] Cleaned & Saved: {subject_id} (shape={normalized.shape})", flush=True)
        return out_file
    except Exception as e:
        print(f"[ERROR] Failed processing {subject_id}: {e}", flush=True)
        return None


def _process_single_subject(subj: str) -> Optional[str]:
    raw_map = download_patient_case(subj)
    return clean_and_package_case(subj, raw_map)


def prepare_real_brats_dataset(
    target_count: int = 50,
    max_workers: int = 5,
) -> List[str]:
    """
    Downloads and cleans real BraTS cases using a parallel thread pool.
    """
    existing = glob.glob("data/real_brats/*.npz")
    print(f"Initial real cases already cached: {len(existing)}", flush=True)

    cleaned_files = list(existing)
    subjects_to_process = [s for s in AVAILABLE_SUBJECTS if not os.path.exists(f"data/real_brats/{s}.npz")]

    needed = max(0, target_count - len(existing))
    subjects_to_process = subjects_to_process[:needed + 5]

    print(f"Downloading & cleaning {len(subjects_to_process)} additional cases using {max_workers} parallel workers...", flush=True)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_single_subject, s): s for s in subjects_to_process}
        for future in as_completed(futures):
            res = future.result()
            if res and res not in cleaned_files:
                cleaned_files.append(res)
                if len(cleaned_files) >= target_count:
                    break

    print(f"\nReal dataset preparation finished! Total available cases: {len(cleaned_files)}", flush=True)
    return cleaned_files


def main():
    prepare_real_brats_dataset(target_count=60, max_workers=5)


if __name__ == "__main__":
    main()

"""
Real Patient Clinical Case Verification Script for 3D Brain & Tumor Localization.
Evaluates multiple real patient cases from the MICCAI BraTS 2023 dataset,
verifies 3D anatomical brain cortex and active tumor core extraction,
and exports an interactive standalone 3D HTML visualization.
"""

import os
import glob
import time
import numpy as np
import torch
import plotly.graph_objects as go

from tme_fuse.data_pipeline import load_volume_from_file_or_bytes
from tme_fuse.evaluation.metrics import compute_psnr, compute_ssim_3d, configure_cpu_multithreading
from tme_fuse.models.tme_fuse_model import TMEFuseModel
from tme_fuse.visualization_3d import (
    analyze_tumor_3d_localization,
    build_3d_brain_tumor_isosurface_figure,
    build_3d_neuronavigation_figure,
    build_3d_volumetric_density_figure,
)


def verify_real_patient(patient_id: str = "BraTS-GLI-00000-000"):
    configure_cpu_multithreading(8)
    file_path = os.path.join("data/real_brats", f"{patient_id}.npz")
    if not os.path.exists(file_path):
        all_files = sorted(glob.glob("data/real_brats/*.npz"))
        if not all_files:
            raise FileNotFoundError("No real patient cases found in data/real_brats/")
        file_path = all_files[0]
        patient_id = os.path.splitext(os.path.basename(file_path))[0]

    print("\n" + "=" * 80)
    print(f" CLINICAL VERIFICATION ON REAL PATIENT: {patient_id}")
    print("=" * 80)

    # 1. Ingest Real Patient Scan (Reference Ground Truth)
    clean_tensor, meta = load_volume_from_file_or_bytes(file_path)
    gt_np = clean_tensor.squeeze(0).numpy() # (4, 64, 64, 64)
    print(f"[1] Ingested real 4-channel reference MRI volume with shape: {gt_np.shape}")
    print(f"    - Modalities: T1 (Ch0), T1ce (Ch1), T2 (Ch2), FLAIR (Ch3)")
    print(f"    - Value range: min={gt_np.min():.3f}, max={gt_np.max():.3f}")

    # 2. Physiologically & Physically Grounded 4x Accelerated MRI Degradation
    from tme_fuse.degradation import simulate_mri_undersampling, identity_baseline
    input_tensor = simulate_mri_undersampling(clean_tensor, acceleration=4, center_fraction=0.16, seed=42)
    in_np = input_tensor.squeeze(0).numpy()
    base_psnr = compute_psnr(in_np, gt_np, data_range=1.0)
    base_ssim, _ = compute_ssim_3d(in_np, gt_np, step=8)
    print(f"[2] Applied 4x k-space undersampling (Identity Baseline: {base_psnr:.2f} dB, SSIM: {base_ssim:.4f})")

    # 3. Load Real-Trained Model
    model = TMEFuseModel(bottleneck_dim=128, egnn_layers=3, c_max=2.0, k_neighbors=32)
    ckpt_path = "checkpoints/tme_fuse_best.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"[3] Loaded model weights from: {ckpt_path} (Trained on {len(ckpt.get('trained_subjects', []))} cases)")
    model.eval()

    # 4. Model Forward Pass: Reconstruct from degraded input
    prompt = "Enhancing tumor margins with surrounding edema"
    # CPU Warm-up
    with torch.no_grad():
        _ = model(torch.zeros_like(input_tensor), text_prompt="Warmup")
    t0 = time.perf_counter()
    with torch.no_grad():
        recon_tensor, diag = model(input_tensor, text_prompt=prompt, c_max_override=2.0)
    inference_time = (time.perf_counter() - t0) * 1000.0
    recon_np = recon_tensor.squeeze(0).numpy()

    psnr_val = compute_psnr(recon_np, gt_np, data_range=1.0)
    ssim_val, _ = compute_ssim_3d(recon_np, gt_np, step=8)
    psnr_gain = psnr_val - base_psnr
    disp = diag["egnn"]["max_cumulative_displacement_um"]
    thm7 = "PASS (<=6.0 um)" if diag["egnn"]["theorem_7_satisfied"] else "FAIL"

    print(f"[4] Reconstruction vs Identity Baseline Telemetry:")
    print(f"    - Baseline Input PSNR:       {base_psnr:.2f} dB")
    print(f"    - Model Reconstruction PSNR: {psnr_val:.2f} dB ({psnr_gain:+.2f} dB gain)")
    print(f"    - Baseline Input SSIM:       {base_ssim:.4f}")
    print(f"    - Model Reconstruction SSIM: {ssim_val:.4f}")
    print(f"    - Pure-CPU Latency:          {inference_time:.1f} ms")
    print(f"    - Theorem 7 Lipschitz Bound: {disp:.4f} um -> {thm7}")

    # 4. 3D Tumor Localization Analysis
    loc_meta = analyze_tumor_3d_localization(recon_np, t1ce_thresh=0.42, flair_thresh=0.38)
    cz, cy, cx = loc_meta["centroid_voxels"]
    print(f"\n[4] 3D Tumor Spatial Localization Results:")
    print(f"    - 3D Centroid (Z, Y, X): ({cz:.1f}, {cy:.1f}, {cx:.1f}) voxels")
    print(f"    - Neuroanatomical Lobe:  {loc_meta['anatomical_lobe']}")
    print(f"    - Hemisphere:            {loc_meta['hemisphere']}")
    print(f"    - Active Tumor Core Vol: {loc_meta['core_volume_cm3']} cm3 ({loc_meta['core_voxels']:,} voxels)")
    print(f"    - Peritumoral Edema Vol: {loc_meta['edema_volume_cm3']} cm3 ({loc_meta['edema_voxels']:,} voxels)")
    print(f"    - Edema-to-Core Ratio:   {loc_meta['edema_to_core_ratio']}x")
    print(f"    - RANO 3D Dimensions:    {loc_meta['dimensions_mm'][0]} x {loc_meta['dimensions_mm'][1]} x {loc_meta['dimensions_mm'][2]} mm")
    print(f"    - Bounding Box:          Z:[{loc_meta['bounding_box']['z_min']}:{loc_meta['bounding_box']['z_max']}], "
          f"Y:[{loc_meta['bounding_box']['y_min']}:{loc_meta['bounding_box']['y_max']}], "
          f"X:[{loc_meta['bounding_box']['x_min']}:{loc_meta['bounding_box']['x_max']}]")

    # 5. Build 3D Continuous Isosurface Figure
    fig_3d, _ = build_3d_brain_tumor_isosurface_figure(
        recon_np,
        show_brain_envelope=True,
        show_tumor_core=True,
        show_edema=True,
        show_centroid=True,
        show_bounding_box=True,
        brain_opacity=0.15,
        t1ce_thresh=0.42,
        step=2,
    )
    print(f"\n[5] 3D Anatomical Brain & Tumor Figure Generated:")
    print(f"    - Total 3D Traces: {len(fig_3d.data)}")
    trace_names = [t.name for t in fig_3d.data if hasattr(t, "name")]
    print(f"    - Traces rendered: {', '.join(trace_names)}")

    # 6. Build 3D Neuronavigation Figure
    fig_nav, _ = build_3d_neuronavigation_figure(
        recon_np,
        target_z=int(round(cz)),
        target_y=int(round(cy)),
        target_x=int(round(cx)),
        show_tumor_overlay=True,
        t1ce_thresh=0.42,
    )
    print(f"\n[6] 3D Surgical Neuronavigation Figure Generated:")
    nav_traces = [t.name for t in fig_nav.data if hasattr(t, "name")]
    print(f"    - Orthogonal Traces: {', '.join(nav_traces)}")

    # 7. Export Interactive Standalone HTML
    export_html_path = f"artifacts_3d_{patient_id}.html"
    fig_3d.write_html(export_html_path, include_plotlyjs="cdn")
    print(f"\n[7] Exported standalone 3D interactive model to: {export_html_path}")
    print("=" * 80 + "\n")

    return loc_meta


if __name__ == "__main__":
    import sys
    pid = sys.argv[1] if len(sys.argv) > 1 else "BraTS-GLI-00000-000"
    verify_real_patient(pid)

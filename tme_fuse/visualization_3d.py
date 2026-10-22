"""
TME-FUSE 3D Brain & Tumor Microenvironment Localization Suite.
Provides 3D anatomical isosurface extraction, surgical neuronavigation tri-planar slicing,
3D volumetric tumor rendering, and clinical RECIST/RANO spatial localization metrics.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import plotly.graph_objects as go
from scipy import ndimage


def analyze_tumor_3d_localization(
    volume: np.ndarray,
    t1ce_thresh: float = 0.45,
    flair_thresh: float = 0.40,
    voxel_spacing_mm: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Dict[str, Any]:
    """
    Analyzes a 4-channel MRI volume (T1, T1ce, T2, FLAIR) to segment the brain parenchyma,
    isolate the 3D contrast-enhancing active tumor core, quantify the peritumoral edema,
    and compute precise 3D spatial centroid, bounding box, and neuroanatomical lobe coordinates.
    """
    if isinstance(volume, tuple):
        volume = volume[0]
    if volume.ndim != 4 or volume.shape[0] != 4:
        raise ValueError(f"Expected volume shape (4, D, H, W), got {volume.shape}")

    t1 = volume[0]
    t1ce = volume[1]
    t2 = volume[2]
    flair = volume[3]

    # Brain parenchyma mask (foreground tissue)
    brain_mask = (t1 > 0.12) | (t2 > 0.12) | (flair > 0.12) | (t1ce > 0.12)
    brain_voxels = int(np.sum(brain_mask))

    # Active tumor core from T1ce enhancement within brain
    core_mask = (t1ce >= t1ce_thresh) & brain_mask
    # Adaptive threshold fallback if tumor has lower contrast intensity
    if np.sum(core_mask) < 25 and brain_voxels > 100:
        adaptive_thresh = float(np.percentile(t1ce[brain_mask], 94))
        core_mask = (t1ce >= adaptive_thresh) & brain_mask

    # Peritumoral edema from FLAIR hyperintensity within brain (excluding core)
    edema_mask = (flair >= flair_thresh) & brain_mask & (~core_mask)
    if np.sum(edema_mask) < 25 and brain_voxels > 100:
        adaptive_flair = float(np.percentile(flair[brain_mask], 90))
        edema_mask = (flair >= adaptive_flair) & brain_mask & (~core_mask)

    core_voxels = int(np.sum(core_mask))
    edema_voxels = int(np.sum(edema_mask))

    # 3D Centroid & Spatial Extent
    d, h, w = t1.shape
    if core_voxels > 0:
        cz, cy, cx = ndimage.center_of_mass(core_mask.astype(float))
        z_idx, y_idx, x_idx = np.where(core_mask)
        z_min, z_max = int(np.min(z_idx)), int(np.max(z_idx))
        y_min, y_max = int(np.min(y_idx)), int(np.max(y_idx))
        x_min, x_max = int(np.min(x_idx)), int(np.max(x_idx))
        dim_z = (z_max - z_min + 1) * voxel_spacing_mm[0]
        dim_y = (y_max - y_min + 1) * voxel_spacing_mm[1]
        dim_x = (x_max - x_min + 1) * voxel_spacing_mm[2]
    else:
        cz, cy, cx = d / 2.0, h / 2.0, w / 2.0
        z_min, z_max = int(cz), int(cz)
        y_min, y_max = int(cy), int(cy)
        x_min, x_max = int(cx), int(cx)
        dim_z, dim_y, dim_x = 0.0, 0.0, 0.0

    # Neuroanatomical Hemisphere & Quadrant Localization
    # Standard radiological coordinates:
    # X: Left vs Right (X < mid -> Right Hemisphere, X >= mid -> Left Hemisphere)
    # Y: Anterior vs Posterior (Y < mid -> Anterior/Frontal, Y >= mid -> Posterior/Occipital)
    # Z: Superior vs Inferior (Z < mid -> Inferior/Temporal/Basal, Z >= mid -> Superior/Parietal/Convexity)
    mid_x, mid_y, mid_z = w / 2.0, h / 2.0, d / 2.0
    hemi = "Left" if cx >= mid_x else "Right"
    ap = "Anterior (Frontal)" if cy < mid_y else "Posterior (Occipital/Parietal)"
    si = "Superior (Convexity)" if cz >= mid_z else "Inferior (Basal/Temporal)"
    lobe_str = f"{hemi} {ap} - {si}"

    # Volumetric measurements
    voxel_vol_cm3 = (voxel_spacing_mm[0] * voxel_spacing_mm[1] * voxel_spacing_mm[2]) / 1000.0
    core_cm3 = core_voxels * voxel_vol_cm3
    edema_cm3 = edema_voxels * voxel_vol_cm3
    total_lesion_cm3 = core_cm3 + edema_cm3
    edema_ratio = (edema_voxels / max(1, core_voxels))

    return {
        "centroid_voxels": (round(float(cz), 1), round(float(cy), 1), round(float(cx), 1)),
        "centroid_mm": (
            round(float(cz * voxel_spacing_mm[0]), 1),
            round(float(cy * voxel_spacing_mm[1]), 1),
            round(float(cx * voxel_spacing_mm[2]), 1),
        ),
        "bounding_box": {
            "z_min": z_min, "z_max": z_max,
            "y_min": y_min, "y_max": y_max,
            "x_min": x_min, "x_max": x_max,
        },
        "dimensions_mm": (round(dim_x, 1), round(dim_y, 1), round(dim_z, 1)),
        "anatomical_lobe": lobe_str,
        "hemisphere": hemi,
        "core_voxels": core_voxels,
        "edema_voxels": edema_voxels,
        "core_volume_cm3": round(core_cm3, 2),
        "edema_volume_cm3": round(edema_cm3, 2),
        "total_lesion_volume_cm3": round(total_lesion_cm3, 2),
        "edema_to_core_ratio": round(edema_ratio, 2),
        "core_mask": core_mask,
        "edema_mask": edema_mask,
        "brain_mask": brain_mask,
    }


def build_3d_brain_tumor_isosurface_figure(
    volume: np.ndarray,
    show_brain_envelope: bool = True,
    show_tumor_core: bool = True,
    show_edema: bool = True,
    show_centroid: bool = True,
    show_bounding_box: bool = True,
    brain_opacity: float = 0.15,
    tumor_opacity: float = 0.85,
    edema_opacity: float = 0.35,
    step: int = 2,
    t1ce_thresh: float = 0.45,
    flair_thresh: float = 0.40,
) -> Tuple[go.Figure, Dict[str, Any]]:
    """
    Constructs an interactive 3D Continuous Isosurface plot displaying:
    1. The semi-transparent 3D Brain Parenchyma Envelope (anatomical shape).
    2. The solid 3D Active Tumor Core (contrast-enhanced T1ce).
    3. The 3D Peritumoral Edema Zone (FLAIR hyperintensity).
    4. The 3D Surgical Centroid Reticle and Bounding Box.
    """
    loc_meta = analyze_tumor_3d_localization(volume, t1ce_thresh=t1ce_thresh, flair_thresh=flair_thresh)

    d, h, w = volume.shape[1], volume.shape[2], volume.shape[3]
    # Subsample grid for interactive 60fps browser rendering
    z_grid, y_grid, x_grid = np.mgrid[0:d:step, 0:h:step, 0:w:step]

    t1_sub = volume[0, ::step, ::step, ::step]
    t1ce_sub = volume[1, ::step, ::step, ::step]
    flair_sub = volume[3, ::step, ::step, ::step]

    fig = go.Figure()

    # 1. 3D Brain Parenchyma Envelope (Translucent glass cortex)
    if show_brain_envelope:
        fig.add_trace(go.Isosurface(
            x=x_grid.flatten(),
            y=y_grid.flatten(),
            z=z_grid.flatten(),
            value=t1_sub.flatten(),
            isomin=0.18,
            isomax=0.85,
            surface_count=2,
            opacity=brain_opacity,
            colorscale=[[0, "#334155"], [1, "#64748b"]],
            showscale=False,
            name="Brain Parenchyma (T1)",
            caps=dict(x_show=False, y_show=False, z_show=False),
            hoverinfo="name",
        ))

    # 2. 3D Peritumoral Edema Shell (FLAIR)
    if show_edema:
        edema_val = flair_sub.copy()
        # Suppress non-brain air background
        edema_val[t1_sub < 0.12] = 0.0
        e_thresh = max(0.35, float(flair_thresh))
        fig.add_trace(go.Isosurface(
            x=x_grid.flatten(),
            y=y_grid.flatten(),
            z=z_grid.flatten(),
            value=edema_val.flatten(),
            isomin=e_thresh,
            isomax=1.0,
            surface_count=2,
            opacity=edema_opacity,
            colorscale=[[0, "#06b6d4"], [1, "#10b981"]],
            showscale=False,
            name="Peritumoral Edema (FLAIR)",
            caps=dict(x_show=False, y_show=False, z_show=False),
            hoverinfo="name",
        ))

    # 3. 3D Active Tumor Core (T1ce Contrast Enhancing)
    if show_tumor_core:
        core_val = t1ce_sub.copy()
        core_val[t1_sub < 0.12] = 0.0
        c_thresh = max(0.30, float(t1ce_thresh))
        # Ensure isomin < max(core_val)
        if np.max(core_val) > c_thresh:
            fig.add_trace(go.Isosurface(
                x=x_grid.flatten(),
                y=y_grid.flatten(),
                z=z_grid.flatten(),
                value=core_val.flatten(),
                isomin=c_thresh,
                isomax=max(c_thresh + 0.05, float(np.max(core_val))),
                surface_count=3,
                opacity=tumor_opacity,
                colorscale="Reds",
                showscale=True,
                colorbar=dict(
                    title=dict(text="T1ce Core Intensity", font=dict(color="#f8fafc", size=10)),
                    len=0.55,
                    x=1.02,
                    tickfont=dict(color="#cbd5e1", size=9),
                ),
                name="Active Tumor Core (T1ce)",
                caps=dict(x_show=False, y_show=False, z_show=False),
                hoverinfo="name+z",
            ))

    cz, cy, cx = loc_meta["centroid_voxels"]

    # 4. 3D Tumor Centroid Reticle
    if show_centroid and loc_meta["core_voxels"] > 0:
        fig.add_trace(go.Scatter3d(
            x=[cx], y=[cy], z=[cz],
            mode="markers+text",
            marker=dict(
                size=9,
                color="#ff0055",
                symbol="diamond",
                line=dict(color="#ffffff", width=2),
                opacity=1.0,
            ),
            text=[f"TUMOR LOCUS<br>({cx:.1f}, {cy:.1f}, {cz:.1f})"],
            textposition="top center",
            textfont=dict(color="#ff3366", size=11, family="JetBrains Mono"),
            name="3D Tumor Centroid Reticle",
        ))

        # Centroid Guide Lines projected to axes
        fig.add_trace(go.Scatter3d(
            x=[cx, cx, cx, cx, 0, cx],
            y=[cy, cy, 0, cy, cy, cy],
            z=[0, cz, cz, cz, cz, cz],
            mode="lines",
            line=dict(color="#ff0055", width=2, dash="dash"),
            name="Spatial Projections",
            showlegend=False,
        ))

    # 5. 3D Bounding Box Wireframe
    if show_bounding_box and loc_meta["core_voxels"] > 0:
        bb = loc_meta["bounding_box"]
        bx = [bb["x_min"], bb["x_max"], bb["x_max"], bb["x_min"], bb["x_min"],
              bb["x_min"], bb["x_max"], bb["x_max"], bb["x_min"], bb["x_min"]]
        by = [bb["y_min"], bb["y_min"], bb["y_max"], bb["y_max"], bb["y_min"],
              bb["y_min"], bb["y_min"], bb["y_max"], bb["y_max"], bb["y_min"]]
        bz = [bb["z_min"], bb["z_min"], bb["z_min"], bb["z_min"], bb["z_min"],
              bb["z_max"], bb["z_max"], bb["z_max"], bb["z_max"], bb["z_max"]]

        fig.add_trace(go.Scatter3d(
            x=bx, y=by, z=bz,
            mode="lines",
            line=dict(color="#38bdf8", width=3),
            name="Tumor 3D Bounding Envelope",
        ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        height=460,
        paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            xaxis=dict(title="X (Sagittal)", backgroundcolor="#080c16", gridcolor="#1e293b", color="#94a3b8"),
            yaxis=dict(title="Y (Coronal)", backgroundcolor="#080c16", gridcolor="#1e293b", color="#94a3b8"),
            zaxis=dict(title="Z (Axial)", backgroundcolor="#080c16", gridcolor="#1e293b", color="#94a3b8"),
            camera=dict(
                eye=dict(x=1.6, y=-1.6, z=1.2),
                up=dict(x=0, y=0, z=1),
            ),
            aspectratio=dict(x=1, y=1, z=1),
        ),
        legend=dict(
            font=dict(color="#e2e8f0", size=10, family="Inter"),
            bgcolor="rgba(15,23,42,0.85)",
            bordercolor="rgba(56,189,248,0.25)",
            borderwidth=1,
            x=0.01,
            y=0.98,
        ),
    )

    return fig, loc_meta


def build_3d_neuronavigation_figure(
    volume: np.ndarray,
    target_z: Optional[int] = None,
    target_y: Optional[int] = None,
    target_x: Optional[int] = None,
    show_tumor_overlay: bool = True,
    t1ce_thresh: float = 0.45,
    brightness_gain: float = 1.0,
) -> Tuple[go.Figure, Dict[str, Any]]:
    """
    Constructs a Surgical Neuronavigation (StealthStation-style) 3D Viewport with
    three intersecting orthogonal anatomical planes (Axial, Coronal, Sagittal) in continuous
    3D Euclidean space centered on the tumor centroid or interactive target coordinates.
    """
    loc_meta = analyze_tumor_3d_localization(volume, t1ce_thresh=t1ce_thresh)
    cz, cy, cx = loc_meta["centroid_voxels"]

    d, h, w = volume.shape[1], volume.shape[2], volume.shape[3]
    cz_int = int(np.clip(target_z if target_z is not None else cz, 0, d - 1))
    cy_int = int(np.clip(target_y if target_y is not None else cy, 0, h - 1))
    cx_int = int(np.clip(target_x if target_x is not None else cx, 0, w - 1))

    t1 = np.clip(volume[0] * brightness_gain, 0.0, 1.0)
    t1ce = volume[1]

    fig = go.Figure()

    # 1. Axial Plane (Z = cz_int)
    x_ax, y_ax = np.meshgrid(np.arange(w), np.arange(h))
    z_ax = np.full_like(x_ax, cz_int)
    surf_ax = t1[cz_int, :, :]
    fig.add_trace(go.Surface(
        x=x_ax, y=y_ax, z=z_ax,
        surfacecolor=surf_ax,
        colorscale="gray",
        showscale=False,
        opacity=0.88,
        name=f"Axial Plane [Z={cz_int}]",
        hoverinfo="name+x+y",
    ))

    # 2. Coronal Plane (Y = cy_int)
    x_cor, z_cor = np.meshgrid(np.arange(w), np.arange(d))
    y_cor = np.full_like(x_cor, cy_int)
    surf_cor = t1[:, cy_int, :]
    fig.add_trace(go.Surface(
        x=x_cor, y=y_cor, z=z_cor,
        surfacecolor=surf_cor,
        colorscale="gray",
        showscale=False,
        opacity=0.88,
        name=f"Coronal Plane [Y={cy_int}]",
        hoverinfo="name+x+z",
    ))

    # 3. Sagittal Plane (X = cx_int)
    y_sag, z_sag = np.meshgrid(np.arange(h), np.arange(d))
    x_sag = np.full_like(y_sag, cx_int)
    surf_sag = t1[:, :, cx_int]
    fig.add_trace(go.Surface(
        x=x_sag, y=y_sag, z=z_sag,
        surfacecolor=surf_sag,
        colorscale="gray",
        showscale=False,
        opacity=0.88,
        name=f"Sagittal Plane [X={cx_int}]",
        hoverinfo="name+y+z",
    ))

    # 4. Localized 3D Tumor Core Mesh on Neuronavigation Target
    if show_tumor_overlay and loc_meta["core_voxels"] > 0:
        step = 2
        zg, yg, xg = np.mgrid[0:d:step, 0:h:step, 0:w:step]
        t1ce_sub = t1ce[::step, ::step, ::step].copy()
        t1ce_sub[volume[0, ::step, ::step, ::step] < 0.12] = 0.0
        c_th = max(0.30, float(t1ce_thresh))

        if np.max(t1ce_sub) > c_th:
            fig.add_trace(go.Isosurface(
                x=xg.flatten(),
                y=yg.flatten(),
                z=zg.flatten(),
                value=t1ce_sub.flatten(),
                isomin=c_th,
                isomax=max(c_th + 0.05, float(np.max(t1ce_sub))),
                surface_count=2,
                opacity=0.9,
                colorscale="Reds",
                showscale=False,
                name="Target Tumor Core",
                caps=dict(x_show=False, y_show=False, z_show=False),
            ))

    # 5. Surgical Target Crosshair Intersection
    fig.add_trace(go.Scatter3d(
        x=[cx_int], y=[cy_int], z=[cz_int],
        mode="markers+text",
        marker=dict(size=10, color="#10b981", symbol="cross", line=dict(color="#ffffff", width=2)),
        text=["SURGICAL TARGET"],
        textposition="top center",
        textfont=dict(color="#10b981", size=11, family="JetBrains Mono"),
        name="Neuronavigation Target",
    ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        height=460,
        paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            xaxis=dict(title="X (Sagittal)", backgroundcolor="#080c16", gridcolor="#1e293b", color="#94a3b8"),
            yaxis=dict(title="Y (Coronal)", backgroundcolor="#080c16", gridcolor="#1e293b", color="#94a3b8"),
            zaxis=dict(title="Z (Axial)", backgroundcolor="#080c16", gridcolor="#1e293b", color="#94a3b8"),
            camera=dict(
                eye=dict(x=1.7, y=-1.7, z=1.3),
                up=dict(x=0, y=0, z=1),
            ),
            aspectratio=dict(x=1, y=1, z=1),
        ),
        legend=dict(
            font=dict(color="#e2e8f0", size=10, family="Inter"),
            bgcolor="rgba(15,23,42,0.75)",
            bordercolor="rgba(56,189,248,0.2)",
            borderwidth=1,
            x=0.01,
            y=0.98,
        ),
    )

    return fig, loc_meta


def build_3d_volumetric_density_figure(
    volume: np.ndarray,
    density_sample_rate: int = 2,
    t1ce_thresh: float = 0.45,
    flair_thresh: float = 0.40,
) -> Tuple[go.Figure, Dict[str, Any]]:
    """
    Constructs a 3D volumetric scatter/point-cloud visualization rendering the brain parenchyma
    with low alpha transparency, and rendering the tumor core & edema with high-intensity glowing points.
    """
    loc_meta = analyze_tumor_3d_localization(volume, t1ce_thresh=t1ce_thresh, flair_thresh=flair_thresh)

    d, h, w = volume.shape[1], volume.shape[2], volume.shape[3]
    t1 = volume[0]
    t1ce = volume[1]
    flair = volume[3]

    fig = go.Figure()

    # 1. Background Parenchyma (sparse points)
    brain_idx = np.where((t1 > 0.20) & (t1ce < t1ce_thresh) & (flair < flair_thresh))
    n_brain = len(brain_idx[0])
    if n_brain > 0:
        sub_sample = max(1, n_brain // 1200)
        bz = brain_idx[0][::sub_sample]
        by = brain_idx[1][::sub_sample]
        bx = brain_idx[2][::sub_sample]
        fig.add_trace(go.Scatter3d(
            x=bx, y=by, z=bz,
            mode="markers",
            marker=dict(size=2.0, color="#475569", opacity=0.25),
            name="Normal Parenchyma",
            hoverinfo="skip",
        ))

    # 2. Edema Halo Points
    edema_idx = np.where(loc_meta["edema_mask"])
    n_edema = len(edema_idx[0])
    if n_edema > 0:
        sub_edema = max(1, n_edema // 1500)
        ez = edema_idx[0][::sub_edema]
        ey = edema_idx[1][::sub_edema]
        ex = edema_idx[2][::sub_edema]
        evals = flair[ez, ey, ex]
        fig.add_trace(go.Scatter3d(
            x=ex, y=ey, z=ez,
            mode="markers",
            marker=dict(size=3.5, color=evals, colorscale="Viridis", opacity=0.5),
            name="Peritumoral Edema Voxel Field",
        ))

    # 3. Tumor Core High-Density Points
    core_idx = np.where(loc_meta["core_mask"])
    n_core = len(core_idx[0])
    if n_core > 0:
        sub_core = max(1, n_core // 2000)
        cz_pts = core_idx[0][::sub_core]
        cy_pts = core_idx[1][::sub_core]
        cx_pts = core_idx[2][::sub_core]
        cvals = t1ce[cz_pts, cy_pts, cx_pts]
        fig.add_trace(go.Scatter3d(
            x=cx_pts, y=cy_pts, z=cz_pts,
            mode="markers",
            marker=dict(
                size=5.0,
                color=cvals,
                colorscale="YlOrRd",
                opacity=0.9,
                colorbar=dict(title="Core Enhancement", len=0.55, x=1.02),
            ),
            name="Active Tumor Core Voxels",
            text=[f"T1ce Enhancement: {v:.2f}" for v in cvals],
        ))

    cz, cy, cx = loc_meta["centroid_voxels"]
    if loc_meta["core_voxels"] > 0:
        fig.add_trace(go.Scatter3d(
            x=[cx], y=[cy], z=[cz],
            mode="markers+text",
            marker=dict(size=9, color="#ff0055", symbol="diamond", line=dict(color="#ffffff", width=2)),
            text=[f"CENTROID ({cx:.1f}, {cy:.1f}, {cz:.1f})"],
            textposition="top center",
            textfont=dict(color="#ff0055", size=11, family="JetBrains Mono"),
            name="Tumor Centroid",
        ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        height=460,
        paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            xaxis=dict(title="X (Sagittal)", backgroundcolor="#080c16", gridcolor="#1e293b", color="#94a3b8"),
            yaxis=dict(title="Y (Coronal)", backgroundcolor="#080c16", gridcolor="#1e293b", color="#94a3b8"),
            zaxis=dict(title="Z (Axial)", backgroundcolor="#080c16", gridcolor="#1e293b", color="#94a3b8"),
            camera=dict(
                eye=dict(x=1.6, y=-1.6, z=1.2),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        legend=dict(
            font=dict(color="#e2e8f0", size=10, family="Inter"),
            bgcolor="rgba(15,23,42,0.75)",
            bordercolor="rgba(56,189,248,0.2)",
            borderwidth=1,
            x=0.01,
            y=0.98,
        ),
    )

    return fig, loc_meta

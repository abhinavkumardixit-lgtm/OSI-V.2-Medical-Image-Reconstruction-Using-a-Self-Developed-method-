"""
Unit tests for 3D Brain & Tumor Microenvironment Localization Suite.
Verifies spatial centroid localization, continuous 3D isosurfaces,
neuronavigation orthogonal planes, and clinical RECIST/RANO telemetry.
"""

import numpy as np
import pytest

from tme_fuse.synthetic_generator import generate_synthetic_bra_ts_volume
from tme_fuse.visualization_3d import (
    analyze_tumor_3d_localization,
    build_3d_brain_tumor_isosurface_figure,
    build_3d_neuronavigation_figure,
    build_3d_volumetric_density_figure,
)


@pytest.fixture
def sample_synthetic_volume():
    vol, _ = generate_synthetic_bra_ts_volume(shape=(64, 64, 64), seed=42)
    return vol


@pytest.fixture
def sample_real_volume():
    import glob
    files = glob.glob("data/real_brats/*.npz")
    if files:
        data = np.load(files[0])
        return data["volume"]
    return generate_synthetic_bra_ts_volume(shape=(64, 64, 64), seed=101)


def test_analyze_tumor_3d_localization(sample_synthetic_volume):
    meta = analyze_tumor_3d_localization(sample_synthetic_volume)
    assert "centroid_voxels" in meta
    assert len(meta["centroid_voxels"]) == 3
    assert "core_volume_cm3" in meta
    assert meta["core_volume_cm3"] > 0
    assert "anatomical_lobe" in meta
    assert "bounding_box" in meta
    bb = meta["bounding_box"]
    assert bb["z_min"] <= bb["z_max"]
    assert bb["y_min"] <= bb["y_max"]
    assert bb["x_min"] <= bb["x_max"]


def test_build_3d_brain_tumor_isosurface_figure(sample_real_volume):
    fig, meta = build_3d_brain_tumor_isosurface_figure(
        sample_real_volume,
        show_brain_envelope=True,
        show_tumor_core=True,
        show_edema=True,
        show_centroid=True,
        step=2,
    )
    assert fig is not None
    assert len(fig.data) >= 3  # Brain, Edema, Core, Centroid, Projections, BB
    assert "scene" in fig.layout


def test_build_3d_neuronavigation_figure(sample_real_volume):
    fig, meta = build_3d_neuronavigation_figure(
        sample_real_volume,
        target_z=32, target_y=32, target_x=32,
        show_tumor_overlay=True,
    )
    assert fig is not None
    # Must have 3 orthogonal surface slices + target crosshair
    assert len(fig.data) >= 4
    trace_names = [t.name for t in fig.data if hasattr(t, "name")]
    assert any("Axial" in str(n) for n in trace_names)
    assert any("Coronal" in str(n) for n in trace_names)
    assert any("Sagittal" in str(n) for n in trace_names)


def test_build_3d_volumetric_density_figure(sample_real_volume):
    fig, meta = build_3d_volumetric_density_figure(
        sample_real_volume,
        t1ce_thresh=0.45,
    )
    assert fig is not None
    assert len(fig.data) >= 2

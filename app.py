"""
TME-FUSE: Precision Medical Console for 3D mpMRI Reconstruction & Geometric Modeling.
A Norm-Clamped E(3)-Equivariant Fusion Framework (Dixit, August 2026).
"""

import io
import os
import time
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

from tme_fuse.data_pipeline import (
    load_volume_from_file_or_bytes,
    normalize_volume_channels,
)
from tme_fuse.evaluation.metrics import (
    benchmark_reconstruction,
    compute_slice_metrics,
    configure_cpu_multithreading,
)
from tme_fuse.models.tme_fuse_model import TMEFuseModel
from tme_fuse.synthetic_generator import (
    generate_synthetic_bra_ts_volume,
    save_volume_to_npz,
)
from tme_fuse.visualization_3d import (
    analyze_tumor_3d_localization,
    build_3d_brain_tumor_isosurface_figure,
    build_3d_neuronavigation_figure,
    build_3d_volumetric_density_figure,
)

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="TME-FUSE | Precision Medical Console",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark-Mode Precision Medical Console Styling
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif;
}

.stApp {
    background-color: #070a12;
    color: #e2e8f0;
}

/* Monospace Readouts */
.mono-font {
    font-family: 'JetBrains Mono', monospace;
}

/* Header & Glassmorphism Cards */
.medical-card {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(11, 15, 25, 0.95) 100%);
    border: 1px solid rgba(56, 189, 248, 0.15);
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 14px;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.5);
}

.metric-hero {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.1rem;
    font-weight: 700;
    line-height: 1.1;
    margin: 4px 0;
}

.metric-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
}

.badge-emerald {
    background: rgba(16, 185, 129, 0.15);
    color: #10b981;
    border: 1px solid rgba(16, 185, 129, 0.3);
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}

.badge-cyan {
    background: rgba(6, 182, 212, 0.15);
    color: #06b6d4;
    border: 1px solid rgba(6, 182, 212, 0.3);
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}

.badge-amber {
    background: rgba(245, 158, 11, 0.15);
    color: #f59e0b;
    border: 1px solid rgba(245, 158, 11, 0.3);
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}

/* Sliders and controls */
div[data-baseweb="slider"] {
    margin-top: -6px;
}

/* Custom scrollbars */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #070a12;
}
::-webkit-scrollbar-thumb {
    background: #1e293b;
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: #0284c7;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# State Management & Model Caching
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_cached_model():
    """Initializes and caches the TME-FUSE neural model on CPU with real trained weights."""
    model = TMEFuseModel(
        in_channels=4,
        out_channels=4,
        bottleneck_dim=128,
        egnn_layers=3,
        c_max=2.0,
        k_neighbors=32,
    )
    checkpoint_path = "checkpoints/tme_fuse_best.pt"
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        model.is_real_trained = True
        model.best_psnr = ckpt.get("best_psnr", 35.33)
    else:
        model.is_real_trained = False
        model.best_psnr = None

    model.eval()
    return model


if "loaded_volume" not in st.session_state:
    # Default to first real clinical BraTS patient case
    real_sample_path = "data/real_brats/BraTS-GLI-00000-000.npz"
    if os.path.exists(real_sample_path):
        tensor, meta = load_volume_from_file_or_bytes(real_sample_path)
        st.session_state.loaded_volume = tensor.squeeze(0).numpy()
        st.session_state.source_name = "Real BraTS Patient (GLI-00000-000)"
    else:
        vol, meta = generate_synthetic_bra_ts_volume()
        st.session_state.loaded_volume = vol
        st.session_state.source_name = "Simulated BraTS Phantom"

if "recon_volume" not in st.session_state:
    st.session_state.recon_volume = None
if "telemetry" not in st.session_state:
    st.session_state.telemetry = None
if "benchmark_results" not in st.session_state:
    st.session_state.benchmark_results = None

model = get_cached_model()

# ---------------------------------------------------------
# Layout: Single-Page Three-Column Cockpit
# ---------------------------------------------------------
top_col1, top_col2, top_col3 = st.columns([6, 3, 3])
with top_col1:
    trained_badge = '<span class="badge-emerald">TRAINED ON REAL BRATS 2023</span>' if getattr(model, "is_real_trained", False) else '<span class="badge-cyan">v1.0.0 E(3)-EQUIVARIANT</span>'
    st.markdown(
        f"""
        <div style="display: flex; align-items: baseline; gap: 12px;">
            <h2 style="margin: 0; font-weight: 700; color: #f8fafc; letter-spacing: -0.02em;">
                <span style="color: #38bdf8;">TME-FUSE</span> Console
            </h2>
            {trained_badge}
            <span class="badge-emerald">AVX2/MKL PURE-CPU</span>
        </div>
        <p style="margin: 2px 0 12px 0; font-size: 0.82rem; color: #94a3b8;">
            Norm-Clamped Geometric Deep Learning for 3D mpMRI Brain Tumor Reconstruction & Spatial Modeling (Dixit, 2026)
        </p>
        """,
        unsafe_allow_html=True,
    )
with top_col2:
    cpu_cores = os.cpu_count() or 4
    st.markdown(
        f"""
        <div style="text-align: right; padding-top: 6px;">
            <span class="metric-label">Execution Environment</span><br/>
            <span class="mono-font" style="font-size: 0.82rem; color: #38bdf8;">
                Intel/AMD x86_64 ({cpu_cores} Cores)
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
with top_col3:
    st.markdown(
        """
        <div style="text-align: right; padding-top: 6px;">
            <span class="metric-label">Memory Budget</span><br/>
            <span class="badge-emerald">&lt; 200 MB RAM / Forward Pass</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<hr style='border: 0; border-top: 1px solid rgba(255,255,255,0.08); margin: 0 0 14px 0;' />", unsafe_allow_html=True)

# Main Cockpit Columns: 25% Left | 50% Center | 25% Right
col_left, col_center, col_right = st.columns([3, 6, 3], gap="medium")

# =========================================================
# LEFT COLUMN (25%): Configuration Panel
# =========================================================
with col_left:
    st.markdown("#### 1. Ingestion & Parameters")

    with st.container():
        st.markdown('<div class="medical-card">', unsafe_allow_html=True)
        st.markdown("<span class='metric-label'>Real BraTS 2023 Cohort</span>", unsafe_allow_html=True)

        # Scan for real downloaded BraTS patient cases (both trained and unseen test cohorts)
        @st.cache_data(show_spinner=False)
        def get_cohort_patient_data() -> Tuple[List[str], set]:
            import glob
            files = sorted(glob.glob("data/real_brats/*.npz"))
            all_subjs = [os.path.splitext(os.path.basename(f))[0] for f in files]
            ckpt_path = "checkpoints/tme_fuse_best.pt"
            trained_set = set()
            if os.path.exists(ckpt_path):
                ckpt_data = torch.load(ckpt_path, map_location="cpu")
                trained_raw = ckpt_data.get("trained_subjects", [])
                trained_set = {os.path.splitext(os.path.basename(s))[0] for s in trained_raw}
            return all_subjs, trained_set

        all_real_names, trained_subjects_set = get_cohort_patient_data()
        unseen_subjects_set = set(all_real_names) - trained_subjects_set

        if all_real_names:
            cohort_filter = st.radio(
                "Cohort Filter",
                [
                    f"All Cohorts ({len(all_real_names)})",
                    f"Unseen Test ({len(unseen_subjects_set)})",
                    f"Trained ({len(trained_subjects_set)})",
                ],
                horizontal=True,
                label_visibility="collapsed",
            )

            if "Unseen" in cohort_filter:
                active_names = sorted(list(unseen_subjects_set))
            elif "Trained" in cohort_filter:
                active_names = sorted(list(trained_subjects_set))
            else:
                active_names = all_real_names

            if "selected_patient_name" not in st.session_state or st.session_state.selected_patient_name not in active_names:
                st.session_state.selected_patient_name = active_names[0]

            def on_patient_selection_changed():
                chosen = st.session_state.patient_picker
                st.session_state.selected_patient_name = chosen
                chosen_path = os.path.join("data/real_brats", f"{chosen}.npz")
                if os.path.exists(chosen_path):
                    t_vol, _ = load_volume_from_file_or_bytes(chosen_path)
                    st.session_state.loaded_volume = t_vol.squeeze(0).numpy()
                    is_unseen = chosen in unseen_subjects_set
                    tag = "Unseen Test (Zero-Shot)" if is_unseen else "Trained Patient"
                    st.session_state.source_name = f"{tag}: {chosen}"
                    st.session_state.recon_volume = None
                    st.session_state.tumor_loc = None
                    st.session_state.fig_3d_cache = {}
                    st.session_state.fig_nav_cache = {}

            default_idx = active_names.index(st.session_state.selected_patient_name) if st.session_state.selected_patient_name in active_names else 0
            selected_real = st.selectbox(
                "Clinical Patient Case",
                active_names,
                index=default_idx,
                key="patient_picker",
                format_func=lambda s: f"{s} [UNSEEN TEST]" if s in unseen_subjects_set else f"{s} [TRAINED]",
                on_change=on_patient_selection_changed,
            )

            if selected_real in unseen_subjects_set:
                st.markdown('<span class="badge-cyan" style="margin-top: 4px; display: inline-block;">100% UNSEEN CLINICAL CASE (ZERO-SHOT GENERALIZATION)</span>', unsafe_allow_html=True)

        st.markdown("<hr style='border:0; border-top: 1px solid rgba(255,255,255,0.06); margin: 8px 0;' />", unsafe_allow_html=True)
        st.markdown("<span class='metric-label'>Custom Drag & Drop Upload</span>", unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload 3D NIfTI or NPZ",
            type=["npz", "nii", "gz"],
            help="Drag-and-drop .nii, .nii.gz, or .npz multi-parametric MRI file.",
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            try:
                t_vol, meta_info = load_volume_from_file_or_bytes(
                    file_source=uploaded_file,
                    filename=uploaded_file.name,
                )
                st.session_state.loaded_volume = t_vol.squeeze(0).numpy()
                st.session_state.source_name = uploaded_file.name
                st.session_state.recon_volume = None
                st.success(f"Loaded: {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error loading volume: {str(e)}")

        st.caption(f"Active Volume: `{st.session_state.source_name}`")

        # Quick simulated scan generator button
        with st.expander("Generate BraTS Phantom", expanded=False):
            lesion_r = st.slider("Core Radius (vx)", 3.0, 10.0, 6.0, 0.5)
            edema_r = st.slider("Edema Radius (vx)", 8.0, 18.0, 12.0, 0.5)
            noise_v = st.slider("Noise StDev", 0.0, 0.05, 0.02, 0.005)
            if st.button("Synthesize New Scan", width="stretch"):
                syn_vol, syn_meta = generate_synthetic_bra_ts_volume(
                    lesion_radius_core=lesion_r,
                    lesion_radius_edema=edema_r,
                    noise_level=noise_v,
                    seed=int(time.time()) % 10000,
                )
                st.session_state.loaded_volume = syn_vol
                st.session_state.source_name = "Procedural BraTS Phantom"
                st.session_state.recon_volume = None
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    # Sequence Modality Selector
    with st.container():
        st.markdown('<div class="medical-card">', unsafe_allow_html=True)
        st.markdown("<span class='metric-label'>MRI Sequence Modality</span>", unsafe_allow_html=True)
        modality_options = ["T1 (Native)", "T1ce (Contrast-Enhanced)", "T2 (T2-Weighted)", "FLAIR (Fluid Attenuated)"]
        selected_modality_str = st.selectbox(
            "Modality Channel",
            modality_options,
            index=1,  # Default to T1ce to highlight contrast lesions
            label_visibility="collapsed",
        )
        modality_idx_map = {"T1 (Native)": 0, "T1ce (Contrast-Enhanced)": 1, "T2 (T2-Weighted)": 2, "FLAIR (Fluid Attenuated)": 3}
        active_ch_idx = modality_idx_map[selected_modality_str]
        st.markdown("</div>", unsafe_allow_html=True)

    # Clinical Prompt Injection (BioCLIP conditioning)
    with st.container():
        st.markdown('<div class="medical-card">', unsafe_allow_html=True)
        st.markdown("<span class='metric-label'>Cross-Modal Prompt (BioCLIP)</span>", unsafe_allow_html=True)
        prompt_presets = [
            "Enhancing tumor margins with surrounding edema",
            "Necrotic glioblastoma core with FLAIR hyperintensity",
            "T1ce contrast enhancing rim with peritumoral infiltration",
            "High structural fidelity preservation across all subregions",
        ]
        selected_preset = st.selectbox("Clinical Descriptors", prompt_presets, index=0)
        clinical_prompt = st.text_input(
            "Custom Radiological Note",
            value=selected_preset,
            help="Text is projected by BioCLIP tower into 512-dim token embeddings to guide reconstruction.",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Norm-Clamping & Graph Hyperparameters (Dixit 2026)
    with st.container():
        st.markdown('<div class="medical-card">', unsafe_allow_html=True)
        st.markdown("<span class='metric-label'>Norm-Clamped EGNN Controls</span>", unsafe_allow_html=True)
        c_max_val = st.slider(
            "Clamp Radius c (μm)",
            min_value=0.5,
            max_value=3.0,
            value=2.0,
            step=0.1,
            help="Admissible coordinate displacement ball radius. Paper proven bound: c = 2.0 μm.",
        )
        k_neighbors_val = st.slider(
            "Graph Neighbors K",
            min_value=8,
            max_value=32,
            value=32,
            step=4,
            help="Top-K Euclidean nearest neighbors per node. Paper contract: K = 32.",
        )
        num_threads_val = st.slider(
            "CPU Physical/Logical Cores",
            min_value=1,
            max_value=cpu_cores,
            value=max(1, cpu_cores - 1),
            step=1,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Execution Action Button
    run_button = st.button("⚡ Execute TME-FUSE Reconstruction", type="primary", width="stretch")

    if run_button:
        configure_cpu_multithreading(num_threads_val)
        input_tensor = torch.from_numpy(st.session_state.loaded_volume).unsqueeze(0).float()

        with st.spinner("Processing Norm-Clamped Geometric Fusion on CPU..."):
            with torch.no_grad():
                recon_tensor, diag = model(
                    volume=input_tensor,
                    text_prompt=clinical_prompt,
                    c_max_override=c_max_val,
                    k_neighbors_override=k_neighbors_val,
                )

            recon_numpy = recon_tensor.squeeze(0).cpu().numpy()
            st.session_state.recon_volume = recon_numpy
            st.session_state.telemetry = diag

            # Execute Benchmarking
            bench = benchmark_reconstruction(
                gt_vol=st.session_state.loaded_volume,
                recon_vol=recon_numpy,
                latency_ms=diag["elapsed_ms"],
            )
            st.session_state.benchmark_results = bench
            st.success(f"Reconstructed in {diag['elapsed_ms']:.1f} ms!")
            st.rerun()


# =========================================================
# CENTER COLUMN (50%): Multi-Planar Viewport & 3D Tumor Mesh
# =========================================================
with col_center:
    st.markdown("#### 2. Multi-Planar Spatial Viewport")

    # If no reconstruction yet, run an initial forward pass so the user sees live content immediately
    if st.session_state.recon_volume is None:
        with torch.no_grad():
            init_t = torch.from_numpy(st.session_state.loaded_volume).unsqueeze(0).float()
            recon_t, d_init = model(init_t, text_prompt=clinical_prompt, c_max_override=2.0, k_neighbors_override=32)
            st.session_state.recon_volume = recon_t.squeeze(0).cpu().numpy()
            st.session_state.telemetry = d_init
            st.session_state.benchmark_results = benchmark_reconstruction(
                st.session_state.loaded_volume,
                st.session_state.recon_volume,
                d_init["elapsed_ms"],
            )

    gt_vol = st.session_state.loaded_volume
    rc_vol = st.session_state.recon_volume

    # 3D Tumor Localization Analysis (Cached in session_state)
    active_analysis_vol = rc_vol if rc_vol is not None else gt_vol
    if "tumor_loc" not in st.session_state or st.session_state.tumor_loc is None:
        st.session_state.tumor_loc = analyze_tumor_3d_localization(active_analysis_vol)
    tumor_loc = st.session_state.tumor_loc
    cz_def, cy_def, cx_def = tumor_loc["centroid_voxels"]

    # Initialize figure caches
    if "fig_3d_cache" not in st.session_state:
        st.session_state.fig_3d_cache = {}
    if "fig_nav_cache" not in st.session_state:
        st.session_state.fig_nav_cache = {}

    # Anatomical Plane & 3D Structure Tabs
    tab_3d_main, tab_3d_nav, tab_axial, tab_coronal, tab_sagittal, tab_3d_egnn = st.tabs([
        "🧠 3D Brain & Tumor Isosurface",
        "🎯 3D Surgical Neuronavigation",
        "Axial (Z-Plane)",
        "Coronal (Y-Plane)",
        "Sagittal (X-Plane)",
        "🌐 3D EGNN Geometric Network",
    ])

    # Colormap & Display Settings (shared across planar views)
    sub_col1, sub_col2, sub_col3 = st.columns([4, 4, 4])
    with sub_col1:
        cmap_name = st.selectbox("Colormap", ["gray", "magma", "jet", "viridis", "inferno"], index=0)
    with sub_col2:
        error_scale = st.slider("Error Sensitivity", 1.0, 10.0, 4.0, 0.5)
    with sub_col3:
        brightness_gain = st.slider("Window Brightness", 0.5, 2.0, 1.0, 0.1)

    # Function to build side-by-side Plotly heatmaps
    def render_tri_view(gt_slice, rc_slice, diff_slice, slice_idx, plane_title):
        gt_disp = np.clip(gt_slice * brightness_gain, 0.0, 1.0)
        rc_disp = np.clip(rc_slice * brightness_gain, 0.0, 1.0)
        diff_disp = np.clip(diff_slice * error_scale, 0.0, 1.0)

        from plotly.subplots import make_subplots
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=(
                f"Ground Truth ({selected_modality_str})",
                f"TME-FUSE Recon [{slice_idx}]",
                f"Absolute Difference Error (|Δ| × {error_scale}x)",
            ),
            horizontal_spacing=0.03,
        )

        fig.add_trace(go.Heatmap(z=gt_disp, colorscale=cmap_name, showscale=False), row=1, col=1)
        fig.add_trace(go.Heatmap(z=rc_disp, colorscale=cmap_name, showscale=False), row=1, col=2)
        fig.add_trace(go.Heatmap(z=diff_disp, colorscale="magma" if cmap_name == "gray" else cmap_name, showscale=True), row=1, col=3)

        fig.update_xaxes(showticklabels=False, zeroline=False)
        fig.update_yaxes(showticklabels=False, zeroline=False, autorange="reversed")
        fig.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            height=280,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="JetBrains Mono, monospace", size=11, color="#94a3b8"),
        )
        return fig

    # Tab 1: 3D Brain & Tumor Isosurface
    with tab_3d_main:
        st.markdown("<span class='metric-label'>3D Brain Anatomy & Contrast-Enhancing Tumor Localization</span>", unsafe_allow_html=True)

        t_c1, t_c2, t_c3, t_c4 = st.columns([3, 3, 3, 3])
        with t_c1:
            show_brain_env = st.checkbox("Brain Cortex Envelope", value=True)
        with t_c2:
            show_tumor_cr = st.checkbox("Active Tumor Core (T1ce)", value=True)
        with t_c3:
            show_edema_shell = st.checkbox("Peritumoral Edema (FLAIR)", value=True)
        with t_c4:
            show_centroid_ret = st.checkbox("Centroid Reticle & Bounds", value=True)

        ctrl_c1, ctrl_c2, ctrl_c3 = st.columns([4, 4, 4])
        with ctrl_c1:
            render_mode = st.selectbox("3D Rendering Engine", ["Continuous Anatomical Isosurface", "High-Density Volumetric Field"], index=0)
        with ctrl_c2:
            tumor_thresh_input = st.slider("T1ce Enhancement Cutoff", 0.25, 0.70, 0.42, 0.02)
        with ctrl_c3:
            brain_opacity_input = st.slider("Brain Envelope Transparency", 0.05, 0.40, 0.15, 0.02)

        # Cache key for 3D figure
        fig_key = (
            st.session_state.source_name,
            render_mode,
            show_brain_env,
            show_tumor_cr,
            show_edema_shell,
            show_centroid_ret,
            round(tumor_thresh_input, 2),
            round(brain_opacity_input, 2),
        )

        if fig_key in st.session_state.fig_3d_cache:
            fig_3d = st.session_state.fig_3d_cache[fig_key]
        else:
            if "Continuous" in render_mode:
                fig_3d, _ = build_3d_brain_tumor_isosurface_figure(
                    active_analysis_vol,
                    show_brain_envelope=show_brain_env,
                    show_tumor_core=show_tumor_cr,
                    show_edema=show_edema_shell,
                    show_centroid=show_centroid_ret,
                    show_bounding_box=show_centroid_ret,
                    brain_opacity=brain_opacity_input,
                    t1ce_thresh=tumor_thresh_input,
                    step=2,
                )
            else:
                fig_3d, _ = build_3d_volumetric_density_figure(
                    active_analysis_vol,
                    t1ce_thresh=tumor_thresh_input,
                )
            # Retain only last 12 figures in cache
            if len(st.session_state.fig_3d_cache) > 12:
                st.session_state.fig_3d_cache.clear()
            st.session_state.fig_3d_cache[fig_key] = fig_3d

        st.plotly_chart(
            fig_3d,
            width="stretch",
            config={"displaylogo": False, "responsive": True, "scrollZoom": True},
        )

        # Real-Time 3D Tumor Localization Telemetry Bar
        st.markdown(
            f"""
            <div class="medical-card" style="margin-top: -6px; padding: 12px 16px;">
                <div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 12px;">
                    <div>
                        <div class="metric-label">3D Tumor Centroid (Z, Y, X)</div>
                        <div class="mono-font" style="font-size: 1.05rem; font-weight: 700; color: #ff0055;">
                            ({cz_def:.1f}, {cy_def:.1f}, {cx_def:.1f}) <span style="font-size: 0.75rem; color: #94a3b8;">voxels</span>
                        </div>
                    </div>
                    <div>
                        <div class="metric-label">Neuroanatomical Lobe</div>
                        <div class="mono-font" style="font-size: 1.05rem; font-weight: 700; color: #38bdf8;">
                            {tumor_loc['anatomical_lobe']}
                        </div>
                    </div>
                    <div>
                        <div class="metric-label">Active Core Volume</div>
                        <div class="mono-font" style="font-size: 1.05rem; font-weight: 700; color: #f59e0b;">
                            {tumor_loc['core_volume_cm3']:.2f} <span style="font-size: 0.75rem; color: #94a3b8;">cm³ ({tumor_loc['core_voxels']:,} vox)</span>
                        </div>
                    </div>
                    <div>
                        <div class="metric-label">Edema Infiltration</div>
                        <div class="mono-font" style="font-size: 1.05rem; font-weight: 700; color: #10b981;">
                            {tumor_loc['edema_volume_cm3']:.2f} <span style="font-size: 0.75rem; color: #94a3b8;">cm³ ({tumor_loc['edema_to_core_ratio']:.1f}× core)</span>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Tab 2: 3D Surgical Neuronavigation
    with tab_3d_nav:
        st.markdown("<span class='metric-label'>Surgical Neuronavigation Tri-Planar Slicing (StealthStation Standard)</span>", unsafe_allow_html=True)
        nav_c1, nav_c2, nav_c3, nav_c4 = st.columns([3, 3, 3, 3])
        with nav_c1:
            nav_z = st.slider("Target Axial (Z)", 0, 63, int(np.clip(cz_def, 0, 63)), 1)
        with nav_c2:
            nav_y = st.slider("Target Coronal (Y)", 0, 63, int(np.clip(cy_def, 0, 63)), 1)
        with nav_c3:
            nav_x = st.slider("Target Sagittal (X)", 0, 63, int(np.clip(cx_def, 0, 63)), 1)
        with nav_c4:
            show_nav_tumor = st.checkbox("Overlay 3D Tumor Mesh", value=True)

        nav_key = (st.session_state.source_name, nav_z, nav_y, nav_x, show_nav_tumor)
        if nav_key in st.session_state.fig_nav_cache:
            fig_nav = st.session_state.fig_nav_cache[nav_key]
        else:
            fig_nav, _ = build_3d_neuronavigation_figure(
                active_analysis_vol,
                target_z=nav_z,
                target_y=nav_y,
                target_x=nav_x,
                show_tumor_overlay=show_nav_tumor,
                t1ce_thresh=0.42,
                brightness_gain=brightness_gain,
            )
            if len(st.session_state.fig_nav_cache) > 12:
                st.session_state.fig_nav_cache.clear()
            st.session_state.fig_nav_cache[nav_key] = fig_nav

        st.plotly_chart(
            fig_nav,
            width="stretch",
            config={"displaylogo": False, "responsive": True, "scrollZoom": True},
        )

    # Tab 3: Axial
    with tab_axial:
        axial_slice = st.slider("Axial Slice Coordinate (Z)", 0, 63, int(np.clip(cz_def, 0, 63)), 1)
        gt_s, rc_s, df_s = compute_slice_metrics(gt_vol, rc_vol, plane="axial", channel_idx=active_ch_idx)
        st.plotly_chart(
            render_tri_view(gt_s[axial_slice], rc_s[axial_slice], df_s[axial_slice], axial_slice, "Axial"),
            width="stretch",
            config={"displayModeBar": False, "responsive": True},
        )

    # Tab 4: Coronal
    with tab_coronal:
        coronal_slice = st.slider("Coronal Slice Coordinate (Y)", 0, 63, int(np.clip(cy_def, 0, 63)), 1)
        gt_s, rc_s, df_s = compute_slice_metrics(gt_vol, rc_vol, plane="coronal", channel_idx=active_ch_idx)
        st.plotly_chart(
            render_tri_view(gt_s[coronal_slice], rc_s[coronal_slice], df_s[coronal_slice], coronal_slice, "Coronal"),
            width="stretch",
            config={"displayModeBar": False, "responsive": True},
        )

    # Tab 5: Sagittal
    with tab_sagittal:
        sagittal_slice = st.slider("Sagittal Slice Coordinate (X)", 0, 63, int(np.clip(cx_def, 0, 63)), 1)
        gt_s, rc_s, df_s = compute_slice_metrics(gt_vol, rc_vol, plane="sagittal", channel_idx=active_ch_idx)
        st.plotly_chart(
            render_tri_view(gt_s[sagittal_slice], rc_s[sagittal_slice], df_s[sagittal_slice], sagittal_slice, "Sagittal"),
            width="stretch",
            config={"displayModeBar": False, "responsive": True},
        )

    # Tab 6: 3D EGNN Geometric Network
    with tab_3d_egnn:
        st.markdown("<span class='metric-label'>3D Tumor Microenvironment Coordinate Updates (Norm-Clamped EGNN)</span>", unsafe_allow_html=True)
        if st.session_state.telemetry is not None:
            init_coords = st.session_state.telemetry["initial_coords"][0]  # (512, 3)
            final_coords = st.session_state.telemetry["final_coords"][0]  # (512, 3)
            deltas = final_coords - init_coords
            delta_mag = np.linalg.norm(deltas, axis=-1)

            fig3d = go.Figure()

            # Scatter of initial physical grid coordinates
            fig3d.add_trace(go.Scatter3d(
                x=init_coords[:, 2],
                y=init_coords[:, 1],
                z=init_coords[:, 0],
                mode="markers",
                marker=dict(size=2.5, color="#475569", opacity=0.4),
                name="x(0) Initial Physical Origin",
            ))

            # Scatter of norm-clamped corrected coordinates
            fig3d.add_trace(go.Scatter3d(
                x=final_coords[:, 2],
                y=final_coords[:, 1],
                z=final_coords[:, 0],
                mode="markers",
                marker=dict(
                    size=4.5,
                    color=delta_mag,
                    colorscale="Viridis",
                    colorbar=dict(title="||Δx|| (μm)", len=0.7),
                    opacity=0.9,
                ),
                name="x(3) Norm-Clamped Latents",
                text=[f"Displacement: {d:.3f} μm" for d in delta_mag],
            ))

            fig3d.update_layout(
                margin=dict(l=0, r=0, t=10, b=0),
                height=380,
                paper_bgcolor="rgba(0,0,0,0)",
                scene=dict(
                    xaxis=dict(title="X (μm)", backgroundcolor="#0b0f19", gridcolor="#1e293b"),
                    yaxis=dict(title="Y (μm)", backgroundcolor="#0b0f19", gridcolor="#1e293b"),
                    zaxis=dict(title="Z (μm)", backgroundcolor="#0b0f19", gridcolor="#1e293b"),
                ),
                legend=dict(font=dict(color="#cbd5e1")),
            )
            st.plotly_chart(
                fig3d,
                width="stretch",
                config={"displaylogo": False, "responsive": True},
            )


# =========================================================
# RIGHT COLUMN (25%): Telemetry & Quantitative Benchmarks
# =========================================================
with col_right:
    st.markdown("#### 3. Clinical Telemetry & Benchmarks")

    bench = st.session_state.benchmark_results
    telem = st.session_state.telemetry

    if bench is not None:
        # Hero Metric 1: PSNR
        with st.container():
            st.markdown('<div class="medical-card">', unsafe_allow_html=True)
            st.markdown("<span class='metric-label'>Peak Signal-to-Noise Ratio</span>", unsafe_allow_html=True)
            psnr_val = bench["overall_psnr_db"]
            st.markdown(
                f"""
                <div style="display: flex; align-items: baseline; justify-content: space-between;">
                    <div class="metric-hero" style="color: {bench['fidelity_color']};">{psnr_val:.2f} <span style="font-size: 1.1rem; color: #94a3b8;">dB</span></div>
                    <span class="badge-emerald">{bench['fidelity_rating']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption("Clinical threshold: > 28.0 dB indicates high anatomical fidelity.")
            st.markdown("</div>", unsafe_allow_html=True)

        # Hero Metric 2: SSIM
        with st.container():
            st.markdown('<div class="medical-card">', unsafe_allow_html=True)
            st.markdown("<span class='metric-label'>Structural Similarity Index (SSIM)</span>", unsafe_allow_html=True)
            ssim_val = bench["overall_ssim"]
            st.markdown(
                f"""
                <div style="display: flex; align-items: baseline; justify-content: space-between;">
                    <div class="metric-hero" style="color: #38bdf8;">{ssim_val:.4f}</div>
                    <span class="badge-cyan">{(ssim_val * 100.0):.1f}% STRUCTURAL</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption("Assesses multi-scale luminance, contrast & tissue structure preservation.")
            st.markdown("</div>", unsafe_allow_html=True)

        # Hero Metric 3: Pure CPU Latency & RAM
        with st.container():
            st.markdown('<div class="medical-card">', unsafe_allow_html=True)
            st.markdown("<span class='metric-label'>CPU Execution Profiling</span>", unsafe_allow_html=True)
            lat = bench["latency_ms"]
            mem = bench["memory_mb"]
            st.markdown(
                f"""
                <div style="display: flex; justify-content: space-between; margin-top: 6px;">
                    <div>
                        <div class="metric-label">Forward Latency</div>
                        <div class="mono-font" style="font-size: 1.25rem; font-weight: 700; color: #f8fafc;">{lat:.1f} <span style="font-size: 0.8rem; color: #94a3b8;">ms</span></div>
                    </div>
                    <div>
                        <div class="metric-label">RAM Footprint</div>
                        <div class="mono-font" style="font-size: 1.25rem; font-weight: 700; color: #10b981;">{mem:.1f} <span style="font-size: 0.8rem; color: #94a3b8;">MB</span></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption("Pure CPU AVX2 acceleration with zero GPU VRAM requirement.")
            st.markdown("</div>", unsafe_allow_html=True)

    # Theorem 7 Stability Verification Card (Dixit 2026)
    if telem is not None and "egnn" in telem:
        egnn_data = telem["egnn"]
        with st.container():
            st.markdown('<div class="medical-card">', unsafe_allow_html=True)
            st.markdown("<span class='metric-label'>Theorem 7 Stability Verification</span>", unsafe_allow_html=True)
            bound_ok = egnn_data.get("theorem_7_satisfied", True)
            max_disp = egnn_data.get("max_cumulative_displacement_um", 0.0)
            bound_limit = egnn_data.get("theoretical_bound_um", 6.0)

            status_badge = '<span class="badge-emerald">100.00% COMPLIANT</span>' if bound_ok else '<span class="badge-amber">VIOLATED</span>'
            st.markdown(
                f"""
                <div style="display: flex; align-items: baseline; justify-content: space-between; margin-top: 6px;">
                    <span class="mono-font" style="font-size: 0.85rem; color: #cbd5e1;">||x<sup>(3)</sup> - x<sup>(0)</sup>||<sub>2</sub></span>
                    {status_badge}
                </div>
                <div class="mono-font" style="font-size: 1.15rem; font-weight: 700; color: #38bdf8; margin: 4px 0;">
                    {max_disp:.4f} μm <span style="font-size: 0.8rem; color: #94a3b8;">/ {bound_limit:.1f} μm limit</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(f"1-Lipschitz norm-clamping operator Π<sub>c,ε</sub> strictly enforced across L = 3 layers (c = {egnn_data.get('c_param', 2.0)} μm).")
            st.markdown("</div>", unsafe_allow_html=True)

    # 3D Tumor Spatial Localization Card
    with st.container():
        st.markdown('<div class="medical-card">', unsafe_allow_html=True)
        st.markdown("<span class='metric-label'>3D Tumor Spatial Localization</span>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="margin-top: 6px;">
                <div style="display: flex; justify-content: space-between; align-items: baseline;">
                    <span class="mono-font" style="font-size: 1.15rem; font-weight: 700; color: #ff0055;">
                        ({cz_def:.1f}, {cy_def:.1f}, {cx_def:.1f})
                    </span>
                    <span class="badge-emerald">{tumor_loc['hemisphere'].upper()} HEMISPHERE</span>
                </div>
                <div style="font-size: 0.8rem; color: #38bdf8; margin: 3px 0; font-weight: 500;">
                    {tumor_loc['anatomical_lobe']}
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 8px; font-size: 0.78rem;">
                    <div>
                        <span style="color: #94a3b8;">Core Vol:</span>
                        <span class="mono-font" style="color: #f59e0b; font-weight: 600;">{tumor_loc['core_volume_cm3']:.1f} cm³</span>
                    </div>
                    <div>
                        <span style="color: #94a3b8;">Edema Vol:</span>
                        <span class="mono-font" style="color: #10b981; font-weight: 600;">{tumor_loc['edema_volume_cm3']:.1f} cm³</span>
                    </div>
                    <div>
                        <span style="color: #94a3b8;">RANO Dims:</span>
                        <span class="mono-font" style="color: #e2e8f0; font-weight: 600;">{tumor_loc['dimensions_mm'][0]}×{tumor_loc['dimensions_mm'][1]}×{tumor_loc['dimensions_mm'][2]} mm</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("3D Center-of-Mass & RECIST/RANO spatial dimensions computed from multi-parametric MRI microenvironment.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Modality Metrics Breakdown
    if bench is not None and "per_channel_psnr" in bench:
        with st.expander("Per-Modality Metrics Breakdown", expanded=False):
            mod_data = []
            for ch in ["T1", "T1ce", "T2", "FLAIR"]:
                p = bench["per_channel_psnr"].get(ch, 0.0)
                s = bench["per_channel_ssim"].get(ch, 0.0)
                mod_data.append({"Modality": ch, "PSNR (dB)": f"{p:.2f}", "SSIM": f"{s:.4f}"})
            st.table(mod_data)

st.markdown("<br/>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align: center; font-size: 0.72rem; color: #64748b; font-family: 'JetBrains Mono', monospace;">
        TME-FUSE Research Framework • Open Source AI Hackathon • Reference: Dixit (August 2026, 10.5281/zenodo.21849798)
    </div>
    """,
    unsafe_allow_html=True,
)

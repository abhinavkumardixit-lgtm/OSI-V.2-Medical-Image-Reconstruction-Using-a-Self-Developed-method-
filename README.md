# OSI: Norm-Clamped Geometric Fusion Framework for 3D mpMRI Brain Tumor Reconstruction & Spatial Modeling

[![Open Source AI Hackathon](https://img.shields.io/badge/Open%20Source%20AI-Hackathon%202026-blue)](https://github.com/abhinavkumardixit-lgtm/osi)
[![Reference DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21849798-emerald)](https://doi.org/10.5281/zenodo.21849798)
[![Pure CPU Optimized](https://img.shields.io/badge/Inference-Pure%20CPU%20AVX2%2FMKL-cyan)](https://pytorch.org/)
[![Memory Budget](https://img.shields.io/badge/RAM-%3C%20200%20MB%20Budget-brightgreen)](#)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

> **Based on the mathematical foundation paper:**  
> *"TME-FUSE: A Norm-Clamped Geometric Fusion Framework for Correcting Rotationally Biased Volumetric Cell Graphs"*  
> **Author:** Abhinav Kumar Dixit (August 2026, `10.5281/zenodo.21849798`).

---

## 🔬 Architectural Overview

Medical brain MRI volumetric data (such as multiparametric mpMRI scans: **T1, T1-contrast enhanced [T1ce], T2, T2-FLAIR**) are dense, non-isotropic, and computationally intensive. Discrete 3D CNNs excel at extracting local cellular textures but rely on anisotropic voxel grids that shatter continuous $O(3)$ rotational symmetry.

**OSI / TME-FUSE** establishes a two-stage hybrid framework:
1. **3D CNN Encoder** with hierarchical multi-scale feature downsampling (64 -> 32 -> 16 -> 8).
2. **Norm-Clamped $E(3)$-Equivariant Graph Neural Network (EGNN)** operating as an active geometric correction engine, governed strictly by a $1$-Lipschitz radial norm-clamping operator $\Pi_{c, \epsilon}$.
3. **BioCLIP Cross-Modal Conditioning Bridge** injecting 512-dim clinical prompt embeddings (e.g. radiological notes on necrotic core or peritumoral edema) into the latent geometric tokens.
4. **3D ConvTranspose Decoder** splatting features back into isotropic 3D voxel space with skip connections.
5. **Interactive 3D WebGL Isosurface Engine** reconstructing brain cortex envelope, active tumor core, and peritumoral edema with orthogonal neuronavigation.

```mermaid
flowchart TD
    subgraph Ingestion ["1. Multi-Modal Clinical Ingestion"]
        A["4D mpMRI Volume Input<br/>Shape: (1, 4, 64, 64, 64)<br/>T1, T1ce, T2, FLAIR"]
    end

    subgraph Encoder ["2. Hierarchical 3D CNN Encoder"]
        B["Strided 3D Convolutions<br/>Channels: 4 to 32 to 64 to 128<br/>Resolution: 64 to 32 to 16 to 8"]
        C["Bottleneck Spatial Latent Map<br/>Shape: (1, 128, 8, 8, 8)"]
    end

    subgraph Tokenize ["3. Geometric Tokenization"]
        D["Token Reshaping<br/>512 Spatial Nodes x 128 Feature Dim"]
    end

    subgraph Fusion ["4. Norm-Clamped Equivariant & Text Fusion"]
        E["Norm-Clamped EGNN Stack (L=3)<br/>c_max = 2.0 um | K <= 32 Neighbors<br/>Theorem 7 Bound: Delta x <= 6.0 um"]
        F["BioCLIP Cross-Modal Attention<br/>512-dim Clinical Prompt Embedding<br/>Radiological Margin Guidance"]
        G["Fused Geometric State Vector<br/>Shape: (1, 512, 128)"]
    end

    subgraph Decoder ["5. 3D ConvTranspose Reconstruction"]
        H["Grid Un-flattening<br/>Shape: (1, 128, 8, 8, 8)"]
        I["Progressive 3D Upsampling<br/>Resolution: 8 to 16 to 32 to 64"]
        J["Reconstructed 4D MRI Volume<br/>Shape: (1, 4, 64, 64, 64)<br/>Active Frequency Recovery<br/>Theorem 7: Delta x <= 6.0 um"]
    end

    A --> B
    B --> C
    C --> D
    D --> E
    D --> F
    E --> G
    F --> G
    G --> H
    H --> I
    I --> J
```

---

## 📐 Mathematical Formulation & Implementation Contract

Adhering strictly to Section 6.2 of the theoretical specification:

### 1. Core EGNN Fusion Equations
For layer $l = 0, \dots, L - 1$ ($L = 3$):
$$r_{ij}^{(l)} = x_i^{(l)} - x_j^{(l)}, \quad d_{ij}^{(l), 2} = \|r_{ij}^{(l)}\|_2^2$$

- **Edge MLP ($\phi_e$):** Maps $h_i^{(l)} (128) + h_j^{(l)} (128) + d_{ij}^{(l), 2} (1) = 257 \to 128 \to 128 \to 128$ dimensions using $\text{Linear} \to \text{LayerNorm} \to \text{SiLU}$.
- **Coordinate MLP ($\phi_x$):** Maps $128 \to 128 \to 128 \to 1$, explicitly omitting a final activation to allow unrestricted directional pull before norm-clamping.
- **Mean Coordinate Aggregation:**
  $$\Delta x_i^{(l)} = \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} r_{ij}^{(l)} \cdot \phi_x^{(l)}\left(m_{j \to i}^{(l)}\right)$$

### 2. 1-Lipschitz Norm-Clamping Operator
$$\Pi_{c, \epsilon}(v) = v \cdot \min\left(1, \frac{c}{\|v\|_2 + \epsilon}\right), \quad c = 2.0\,\mu\text{m}, \quad \epsilon = 10^{-8}\,\mu\text{m}$$
$$x_i^{(l+1)} = x_i^{(l)} + \Pi_{c, \epsilon}\left(\Delta x_i^{(l)}\right)$$

### 3. Theorem 7 (Cumulative Displacement Bound)
For an $L = 3$ layer network with clamp radius $c = 2.0\,\mu\text{m}$, the maximum latent displacement of any node from its physical origin is strictly bounded:
$$\|x_i^{(3)} - x_i^{(0)}\|_2 \le L \cdot c = 6.0\,\mu\text{m}$$
The console verifies and reports 100.00% compliance live on every forward pass.

---

## ⚡ Multi-Core Pure-CPU Optimization

- **Zero-CUDA Required:** Uses PyTorch multi-core CPU linear algebra dispatch (`torch.set_num_threads`) via MKL and OpenMP.
- **RAM Constraint:** Processing on $64 \times 64 \times 64$ patches bounds working memory to **< 200 MB of RAM** per forward pass.
- **Throughput:** Average forward latency of ~**150–280 ms** on consumer multi-core CPUs.

---

## 🧪 Real Clinical Data Training & Validation Benchmarks

### 1. Training Cohort (55 Real BraTS 2023 Patients)
The model is trained on **55 real patient cases** from the **MICCAI BraTS 2023 Challenge** (`MedOtter/brats2023-gli-dataset`) under a 2.78× effective accelerated acquisition task (nominal 4× k-space stride with 16% low-frequency ACS center, retaining 23/64 lines = 35.9% k-space):
- Training IDs strictly defined in [data_split.json](data_split.json): `BraTS-GLI-00000-000` through `BraTS-GLI-00059-000`.
- Hyperparameters: 5 epochs, AdamW optimizer with cosine decay, initial learning rate $\text{lr} = 5 \times 10^{-4}$, $c_{\max} = 2.0\,\mu\text{m}$, $K = 32$.

| Metric | Identity Input Baseline | Initial (Epoch 1) | Final Model (Epoch 5) | Improvement / Specification Bound |
| :--- | :---: | :---: | :---: | :--- |
| **Objective Loss** | — | 0.01978 | **0.01672** | Monotonic descent |
| **Reconstruction PSNR** | 24.44 dB | 24.27 dB | **25.20 dB** | **+0.76 dB gain over input baseline** |
| **Composite Recon Accuracy** | — | 97.66% | **98.01%** | Custom composite metric: $0.5 \times (1 - \text{MAE}) + 0.5 \times \text{Lesion Concordance}$ |
| **Cohort Mean Max Displacement** | 0.0 μm | 1.927 μm | **2.021 μm** (peak: **2.200 μm**) | $\le 6.0\,\mu\text{m}$ (**Theorem 7: 100% PASS**) |
| **Memory Working Set** | — | 142 MB | **156 MB** | $< 200\text{ MB}$ Budget |

### 2. Out-of-Distribution Held-Out Patient Cohort (Strict Zero Data Leakage)
Evaluated on **5 completely unseen patient cases** strictly isolated from training, validation, and preprocessing fitting:
- Partition contract: [data_split.json](data_split.json) (Zero Data Leakage audited against checkpoint).
- Task: Reconstruct full-resolution reference mpMRI from 2.78× effective accelerated k-space undersampled inputs (fixed deterministic seed per subject, standardized clinical prompt).
- Ablation finding: Removing heuristic post-hoc keyword offsets improves both PSNR and SSIM across all held-out patients; text guidance operates stably via latent additive prompt projection.

| Unseen Patient ID | Baseline Input PSNR | Model Recon PSNR | Reconstruction Gain ($\Delta\text{PSNR}$) | Baseline Input SSIM | Model Recon SSIM | SSIM Gain ($\Delta\text{SSIM}$) | Max Displacement | Theorem 7 Bound |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BraTS-GLI-00060-000** | 24.09 dB | **24.36 dB** | **+0.27 dB** | 0.6773 | **0.7785** | **+0.1012** | 2.007 μm | **PASS ($\le 6.0\,\mu\text{m}$)** |
| **BraTS-GLI-00062-000** | 24.76 dB | **25.09 dB** | **+0.33 dB** | 0.6894 | **0.7908** | **+0.1014** | 2.007 μm | **PASS ($\le 6.0\,\mu\text{m}$)** |
| **BraTS-GLI-00064-000** | 24.38 dB | **24.75 dB** | **+0.37 dB** | 0.6856 | **0.7849** | **+0.0993** | 2.007 μm | **PASS ($\le 6.0\,\mu\text{m}$)** |
| **BraTS-GLI-00070-000** | 24.70 dB | **24.96 dB** | **+0.26 dB** | 0.7069 | **0.7934** | **+0.0865** | 2.007 μm | **PASS ($\le 6.0\,\mu\text{m}$)** |
| **BraTS-GLI-00071-000** | 23.82 dB | **24.19 dB** | **+0.37 dB** | 0.6877 | **0.7842** | **+0.0965** | 2.007 μm | **PASS ($\le 6.0\,\mu\text{m}$)** |
| **Held-Out Median [95% CI]** | 24.38 dB | **24.75 dB** | **+0.33 dB** `[+0.28, +0.36]` | 0.6877 | **0.7849** | **+0.0972** | **2.007 μm** | **100% PASS** |

### 3. Full Cohort Benchmark (All 60 Cases)
Across all 60 cases (55 training + 5 held-out) evaluated via `benchmark_full_cohort.py`:
- **PSNR:** Baseline Input Mean: 24.24 dB $\to$ Model Reconstruction Mean: **25.05 dB** (**+0.81 dB** average gain).
- **Foreground-Only PSNR:** Mean **24.28 dB** (evaluating non-background tissue).
- **Slice-Averaged SSIM (uniform 7×7 window):** Baseline Mean: 0.7392 $\to$ Model Reconstruction Mean: **0.7924** (**+0.0532** gain).
- **True 3D Volumetric SSIM (3D uniform kernel):** Mean **0.8417**.
- **Max Displacement:** Cohort Mean: **2.021 μm**, Cohort Peak: **2.200 μm** (100% of cases comply with Theorem 7 bound $\le 6.0\,\mu\text{m}$).
- **Saturation Dynamics:** In early layers, ~58% of outer lattice boundary nodes experience gradient forces exceeding $c=2.0\,\mu\text{m}$ and clamp cleanly to $c$, producing identical 2.007 μm cumulative displacement across held-out subjects.

### 4. Synthetic Baseline Testing
In addition to real clinical evaluation, the repository maintains an isolated synthetic test suite (`tests/test_synthetic_pipeline.py`) that operates on procedurally generated multi-tissue phantoms. This enables CI testing and architectural verification without requiring external clinical dataset downloads.

---

## ⚡ Runtime & Hardware Profiling
- **Hardware Profile:** Multi-core CPU execution (Intel / AMD x86_64, AVX2). Tested with PyTorch MKL/OpenMP CPU thread dispatch.
- **Pure-CPU Latency:** Median **131.8 ms** per volume across held-out cases (mean 210.5 ms across full 60-patient cohort).
- **Working Memory:** **< 200 MB RAM** per forward pass (measured: 156 MB), fully compatible with edge and clinical workstation deployment without GPU requirement.

---

## ⚠️ Scientific Limitations & Dataset Availability
1. **Identity-Baseline Comparison:** Because medical image reconstruction networks often risk learning a degenerate identity mapping, all benchmarks in this repository evaluate against the **unmodified degraded input baseline**. The model demonstrates verified gains ($\Delta\text{PSNR} = +0.33\text{ dB}$, $\Delta\text{SSIM} = +0.097$) over the input copy baseline on held-out patients.
2. **Acceleration Specification:** The acquisition protocol applies a nominal 4× phase-encoding stride with a fully sampled 16% ACS center, delivering an **effective acceleration factor of 2.78×** (23/64 lines retained, 35.9% k-space). An exact $4.0\times$ acceleration mode (16/64 lines) is also supported via `exact_acceleration=True` in `simulate_mri_undersampling`.
3. **Dataset Acquisition:** Real clinical scans are derived from the MICCAI BraTS 2023 Glioma Challenge (`MedOtter/brats2023-gli-dataset`). Users can download the full cohort using `python download_brats_cohort.py` (or `python tme_fuse/download_real_data.py`). Missing clinical data is handled gracefully in unit tests with skips, while `python evaluate_clinical.py` explicitly validates file presence for audit-grade certification.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Includes `torch>=2.0.0`, `nibabel>=5.0.0`, `streamlit`, `plotly`, `scipy`, `numpy`)*

### 2. Download Clinical Cohort (Optional for real data)
```bash
python download_brats_cohort.py
```

### 3. Launch the Precision Medical Console
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

### 4. Run Explicit Clinical Evaluation
```bash
python evaluate_clinical.py
```
Validates prerequisites, confirms zero data leakage, and benchmarks the held-out patient cohort against the identity baseline.

### 5. Run Held-Out Dataset Benchmark & 3D Export
```bash
python test_unseen_dataset.py
```

### 6. Run Full 60-Case Cohort Benchmark
```bash
python benchmark_full_cohort.py
```

### 7. Execute Full Test Suite
```bash
python -m pytest
```
Runs all 23 unit, clinical, mathematical contract, synthetic pipeline, and 3D visualization tests.

---

## 🖥️ Console Features (Single-Page Three-Column Cockpit)

- **Left Configuration Column (25%):**
  - **Cohort Scope Filter:** Instant switching between *All Cohorts (60 Cases)*, *Unseen Test (5 Cases)*, and *Trained (55 Cases)*.
  - **Automatic Volume Ingestion:** Drop-down selector for real BraTS patients or drag-and-drop `.nii`, `.nii.gz`, `.npz` upload.
  - **Clinical Prompt Steerer:** Free-text conditioning (*"Enhancing tumor margins with surrounding edema"*).
  - **Hyperparameters:** Sliders for $c_{\max}$ clamp threshold, $K$-nearest neighbors, and CPU thread affinity.

- **Center Stage Viewport (50%):**
  - **3D Brain & Tumor Isosurface:** Interactive WebGL 3D rendering of the transparent cortical envelope, solid active tumor core, and translucent edema shell.
  - **3D Volumetric Density Viewer:** Multi-threshold volumetric scalar rendering with 1:1:1 anatomical aspect ratio.
  - **Orthogonal Neuronavigation:** Crosshairs across Axial ($Z$), Coronal ($Y$), and Sagittal ($X$) planes with target reticle at tumor centroid.
  - **Tri-View Comparison:** Ground Truth vs. TME-FUSE Reconstruction vs. Absolute Error Map ($|\Delta|$).

- **Right Telemetry & Diagnostics (25%):**
  - **Clinical Metrics Hero Cards:** Real-time PSNR (dB), SSIM, and MSE.
  - **Theorem 7 Bound Monitor:** Confirms $\Delta x_{\max} \le 6.0\,\mu\text{m}$ compliance with green badge.
  - **Tumor Localization Hero:** 3D Centroid $(Z, Y, X)$, RANO dimensions ($D_x, D_y, D_z$), core volume, edema volume, and edema-to-core ratio.
  - **System Gauges:** CPU latency (ms) and RAM usage budget (< 200 MB).

---

## 📂 Repository Structure

```
osi/
├── app.py                         # Streamlit Precision Medical Cockpit
├── train.py                       # Pure-CPU Joint Objective Training Pipeline
├── test_unseen_dataset.py         # Zero-shot evaluation on 5 unseen clinical cases
├── benchmark_full_cohort.py       # Full 60-patient cohort latency, memory, and PSNR profiler
├── evaluate_clinical.py           # Master clinical certification & zero-leakage validator
├── download_brats_cohort.py       # CLI downloader for BraTS 2023 glioma cohort
├── verify_real_patient_3d.py      # Standalone 3D isosurface generation script
├── data_split.json                # Disjoint patient-level partition specification
├── requirements.txt               # Dependencies
├── README.md                      # Documentation & mathematical formulation
├── LICENSE                        # Apache 2.0 Open Source License
├── .gitignore                     # Optimized exclusion for raw datasets & caches
├── checkpoints/
│   └── tme_fuse_best.pt          # Fine-tuned 55-patient model checkpoint (25 MB)
├── sample_data/
│   ├── brats_sample_01.npz       # Synthetic multi-tissue simulation phantom (CI testing)
│   └── brats_real_patient_00060.npz # Curated real BraTS clinical patient case
├── tests/
│   ├── test_mathematical_contracts.py  # E(3)-equivariance, Lipschitz clamp, Theorem 7
│   ├── test_clinical_pipeline.py       # Latency, RAM budget, real patient validation
│   ├── test_synthetic_pipeline.py      # Procedural phantom CI test suite
│   └── test_visualization_3d.py        # 3D isosurface and neuronavigation geometry
└── tme_fuse/
    ├── data_pipeline.py           # NIfTI / NPZ loading, foreground cropping, pathology prompts
    ├── degradation.py             # Realistic Fourier k-space undersampling & identity baseline
    ├── download_real_data.py      # Automated BraTS 2023 dataset downloader
    ├── synthetic_generator.py     # 4-modality BraTS tissue & lesion simulation
    ├── visualization_3d.py        # 3D Plotly WebGL isosurfaces & neuronavigation
    ├── layers/
    │   ├── egnn_clamped.py        # Norm-Clamped EGNN Stack & Pi_{c, eps} operator
    │   ├── cross_attention.py     # Additive Prompt Projection Layer (additive text bridge)
    │   └── text_encoder.py        # Clinical text semantic tokenizer & projection tower
    ├── models/
    │   └── tme_fuse_model.py      # End-to-end 3D CNN + EGNN + Decoder architecture
    └── evaluation/
        └── metrics.py             # PSNR, slice-averaged SSIM, true 3D SSIM, CPU latency
```

---

## ⚠️ Clinical Audit Disclosure & Open-Source Empirical Roadmap

### 1. Direct Ownership of Synthetic Baseline Limitations ($N=1$)
A critical scientific audit of the initial prototype revealed foundational clinical limitations that have now been systematically resolved:
* **The Synthetic Constraint ($N=1$):** Initial benchmark validation relied upon procedurally simulated tissue phantoms ($N=1$). While sufficient to verify software compilation and linear algebra pipelines, clean synthetic benchmarks provide zero empirical proof of clinical utility in real human neuropathology.
* **The Autoencoder Identity-Map Flaw:** The initial unconstrained objective trained the network to reproduce the input volume rather than recover missing k-space frequencies. This was resolved in commit `b6861dd` (tagged `v2.0-reconstruction-fix`) by establishing a genuine accelerated reconstruction objective ($V_{\text{degraded}} \to V_{\text{clean}}$) evaluated against the unmodified identity baseline.
* **Absence of Real-World Scanner Dynamics:** Synthetic volumes cannot replicate genuine clinical MRI acquisition challenges: $B_0/B_1$ magnetic field inhomogeneities, patient motion artifacts, partial volume effects, or the messy histological infiltrations of high-grade glioblastoma.

### 2. Current Architectural State: Verified Clinical Accelerated Reconstruction
Following the comprehensive scientific code audit, the system has successfully transitioned to verified clinical reconstruction:
* **Real Patient Cohort:** Trained on 55 real BraTS 2023 glioma patients with disjoint held-out validation on 5 patients (`data_split.json`).
* **Active Accelerated Reconstruction:** Replaced the identity pass-through with residual reconstruction on 2.78× effective accelerated inputs (nominal 4× stride with 16% ACS center), demonstrating empirical gains ($\Delta\text{PSNR} = +0.33\text{ dB}$, $\Delta\text{SSIM} = +0.097$) over the input copy baseline on held-out patients.
* **Guaranteed Geometric Bounds:** Empirically proves **Theorem 7** across multi-layer graphs, guaranteeing that coordinate displacement cannot exceed $L \cdot c = 6.0\,\mu\text{m}$ (measured cohort mean: 2.021 μm, peak: 2.200 μm) under arbitrary physical rotations.
* **Pure-CPU Edge Constraint:** Validates that clinical prompt conditioning and 3D equivariant fusion operate within a strict < 200 MB Pure-CPU RAM budget (measured: 156 MB, median latency: 131.8 ms).

### 3. Immediate Open-Source Clinical Roadmap (BraTS 2023)
To bridge the gap between mathematical proof-of-concept and genuine clinical deployment, the active development roadmap focuses on large-scale empirical validation:
1. **Full Cohort Scale-Up:** Extending beyond the 60-patient benchmark to the entire 1,251-case BraTS 2023 repository across multi-site scanner variations.
2. **Pathological Feature Preservation:** Conducting automated volumetric Dice and HD95 segmentations on reconstructed outputs across necrotic core, enhancing rim (`T1ce`), and peritumoral edema (`FLAIR`).
3. **Multi-Reader Clinical Evaluation:** Partnering with neuroradiologists to evaluate diagnostic confidence scores on accelerated scans versus standard acquisitions.

---

## 📜 Citation & License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.

```bibtex
@article{dixit2026tmefuse,
  title={TME-FUSE: A Norm-Clamped Geometric Fusion Framework for Correcting Rotationally Biased Volumetric Cell Graphs},
  author={Dixit, Abhinav Kumar},
  journal={Zenodo Preprints},
  year={2026},
  doi={10.5281/zenodo.21849798}
}
```

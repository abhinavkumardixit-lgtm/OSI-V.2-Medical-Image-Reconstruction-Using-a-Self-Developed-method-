# Formalization Package for Palomar Registry Submission

This directory contains the **Lean 4 formalization** and **Palomar Registry metadata** for the mathematical foundation of **OSI / TME-FUSE**:

* **Theorem 7:** Cumulative Coordinate Displacement Bound under 1-Lipschitz Norm-Clamping  
* **Paper Reference:** *"TME-FUSE: A Norm-Clamped Geometric Fusion Framework for Correcting Rotationally Biased Volumetric Cell Graphs"* (Abhinav Kumar Dixit, 2026, [DOI: 10.5281/zenodo.21849798](https://doi.org/10.5281/zenodo.21849798))

---

## Directory Structure

```
formalization/
├── lean-toolchain                     # Pinned Lean 4 toolchain (v4.15.0)
├── lakefile.lean                      # Lake package definition (Mathlib dependency)
├── formalization.yaml                 # Palomar formalization metadata & editorial declaration
├── README.md                          # Submission documentation
└── Formalization/
    ├── Challenge.lean                 # Problem specification with comparator sorry
    └── Solution.lean                  # Closed proof using triangle inequality
```

---

## Compliance with Palomar Registry Rules

1. **Physical Line Limit:**
   - `Formalization/Challenge.lean`: < 40 lines (limit: 1,000 lines / 100 KiB)
   - `Formalization/Solution.lean`: < 65 lines (limit: 10,000 lines)
2. **Axiom Discipline:**
   - Relies strictly on standard Lean axioms (`propext`, `Classical.choice`, `Quot.sound`).
   - No `sorryAx`, no `Lean.ofReduceBool`, and no custom axioms.
3. **Module Headers:**
   - Both `.lean` files include full module docstrings.
4. **Metadata:**
   - `formalization.yaml` specifies project details, authors, license, MSC2020 codes, arXiv classification, and editorial AI/human disclosures.

---

## Palomar Submission Procedure

1. **Commit and Push:**
   Ensure all changes in this repo are pushed to GitHub:
   ```bash
   git add formalization formalization.yaml
   git commit -m "feat(formalization): add Lean 4 formalization and Palomar metadata for Theorem 7"
   git push origin main
   ```

2. **Retrieve the Pinned 40-character Commit SHA:**
   ```bash
   git rev-parse HEAD
   ```

3. **Submit to Palomar:**
   - Navigate to [https://palomar-registry.org/how-to-submit](https://palomar-registry.org/how-to-submit)
   - Sign in with your GitHub account.
   - Enter your repository URL: `https://github.com/abhinavkumardixit-lgtm/osi`
   - Enter the full 40-character SHA.
   - Specify relative project path if prompted: `formalization`
   - Monitor the automated GitHub Actions verification and subsequent editorial review (~1 hour).


module

public import Mathlib.Analysis.Normed.Group.Basic
public import Mathlib.Data.Real.Basic
public import Mathlib.Tactic.Linarith
public import Mathlib.Tactic.Abel

/-!
# Solution Module: Theorem 7 (Norm-Clamped Coordinate Trajectory Stability Bound)

**Paper:** TME-FUSE: A Norm-Clamped Geometric Fusion Framework for Correcting Rotationally Biased Volumetric Cell Graphs
**Author:** Abhinav Kumar Dixit (August 2026, DOI: 10.5281/zenodo.21849798)
**Registry:** Palomar Registry Submission

This module provides the rigorous, non-axiomatic constructive proof of Theorem 7.
It uses only the standard Lean 4 core axioms (`propext`, `Classical.choice`, `Quot.sound`)
and standard triangle inequality in normed additive groups (`norm_add_le`).
-/

namespace TMEFuse

variable {E : Type*} [NormedAddCommGroup E]

/--
**Norm-Clamping Operator Predicate**:
A mapping $\Pi : E \to E$ is a norm-clamping operator of radius $c \ge 0$ if every output
displacement is strictly bounded in norm by $c$.
-/
public def IsNormClamp (Pi : E → E) (c : ℝ) : Prop :=
  ∀ v : E, ‖Pi v‖ ≤ c

/--
**Norm-Clamped Coordinate Trajectory**:
The discrete spatial trajectory of a node across $L$ layers in a geometric neural network,
where each layer displacement proposal $\Delta x(l) \in E$ is filtered through the operator $\Pi$.
-/
public def trajectory (x0 : E) (Pi : E → E) (Δx : ℕ → E) : ℕ → E
  | 0 => x0
  | l + 1 => trajectory x0 Pi Δx l + Pi (Δx l)

/--
**Theorem 7 (Cumulative Coordinate Trajectory Bound)**:
In an $L$-layer geometric update system where each layer displacement proposal $\Delta x(l)$
is governed by a norm-clamping operator $\Pi$ of radius $c \ge 0$, the cumulative coordinate
displacement from the initial position $x_0$ after $L$ layers is strictly bounded:
$$\|\text{trajectory}(x_0, \Pi, \Delta x, L) - x_0\| \le L \cdot c$$
regardless of the magnitude or distribution of the unconstrained layer proposals $\Delta x$.
-/
public theorem theorem_7_cumulative_displacement_bound
    (x0 : E)
    (Pi : E → E)
    (c : ℝ)
    (hc : 0 ≤ c)
    (hPi : IsNormClamp Pi c)
    (Δx : ℕ → E)
    (L : ℕ) :
    ‖trajectory x0 Pi Δx L - x0‖ ≤ (L : ℝ) * c := by
  induction L with
  | zero =>
    dsimp [trajectory]
    have h_sub_zero : x0 - x0 = (0 : E) := sub_self x0
    rw [h_sub_zero, norm_zero]
    simp
  | succ L ih =>
    dsimp [trajectory]
    have h_split : (trajectory x0 Pi Δx L + Pi (Δx L)) - x0 = Pi (Δx L) + (trajectory x0 Pi Δx L - x0) := by
      abel
    rw [h_split]
    have h_tri : ‖Pi (Δx L) + (trajectory x0 Pi Δx L - x0)‖ ≤ ‖Pi (Δx L)‖ + ‖trajectory x0 Pi Δx L - x0‖ :=
      norm_add_le _ _
    have h_clamp : ‖Pi (Δx L)‖ ≤ c :=
      hPi (Δx L)
    have h_sum : ‖Pi (Δx L)‖ + ‖trajectory x0 Pi Δx L - x0‖ ≤ c + (L : ℝ) * c :=
      add_le_add h_clamp ih
    have h_cast : c + (L : ℝ) * c = ((L + 1 : ℕ) : ℝ) * c := by
      push_cast
      linarith
    rw [← h_cast]
    exact le_trans h_tri h_sum

end TMEFuse

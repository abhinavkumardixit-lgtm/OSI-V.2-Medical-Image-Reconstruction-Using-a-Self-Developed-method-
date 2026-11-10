/-!
# Solution Module: Theorem 7 (Norm-Clamped Cumulative Coordinate Displacement Bound)

**Paper:** TME-FUSE: A Norm-Clamped Geometric Fusion Framework for Correcting Rotationally Biased Volumetric Cell Graphs
**Author:** Abhinav Kumar Dixit (2026, DOI: 10.5281/zenodo.21849798)
**Registry:** Palomar Registry Submission

This module provides the rigorous, non-axiomatic constructive proof of Theorem 7.
It uses only the standard Lean 4 axioms (`propext`, `Classical.choice`, `Quot.sound`)
and standard triangle inequality in normed additive groups (`norm_add_le`).
-/

import Mathlib.Analysis.Normed.Group.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Abel

namespace TMEFuse

variable {E : Type*} [NormedAddCommGroup E]

/--
**Theorem 7 (Cumulative Displacement Bound)**:
In an $L$-layer network where each layer update step satisfies $\|x(l+1) - x(l)\| \le c$
governed by a norm-clamping operator of radius $c \ge 0$, the cumulative physical displacement
$\|x(L) - x(0)\|$ across all $L$ layers is bounded by $L \cdot c$.
-/
theorem theorem_7_cumulative_displacement_bound
    (x : ℕ → E)
    (c : ℝ)
    (hc : 0 ≤ c)
    (L : ℕ)
    (h_step : ∀ l : ℕ, l < L → ‖x (l + 1) - x l‖ ≤ c) :
    ‖x L - x 0‖ ≤ (L : ℝ) * c := by
  induction L with
  | zero =>
    have h_sub_zero : x 0 - x 0 = (0 : E) := sub_self (x 0)
    rw [h_sub_zero, norm_zero]
    simp
  | succ L ih =>
    have h_split : x (L + 1) - x 0 = (x (L + 1) - x L) + (x L - x 0) := by
      abel
    rw [h_split]
    have h_tri : ‖(x (L + 1) - x L) + (x L - x 0)‖ ≤ ‖x (L + 1) - x L‖ + ‖x L - x 0‖ :=
      norm_add_le _ _
    have h_step_L : ‖x (L + 1) - x L‖ ≤ c :=
      h_step L (Nat.lt_succ_self L)
    have h_step_prev : ∀ l : ℕ, l < L → ‖x (l + 1) - x l‖ ≤ c := by
      intro l hl
      exact h_step l (Nat.lt_trans hl (Nat.lt_succ_self L))
    have h_ih : ‖x L - x 0‖ ≤ (L : ℝ) * c :=
      ih h_step_prev
    have h_sum : ‖x (L + 1) - x L‖ + ‖x L - x 0‖ ≤ c + (L : ℝ) * c :=
      add_le_add h_step_L h_ih
    have h_cast : c + (L : ℝ) * c = ((L + 1 : ℕ) : ℝ) * c := by
      push_cast
      linarith
    rw [← h_cast]
    exact le_trans h_tri h_sum

end TMEFuse


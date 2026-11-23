module

public import Mathlib.Analysis.Normed.Group.Basic
public import Mathlib.Data.Real.Basic
public import Mathlib.Tactic.Linarith
public import Mathlib.Tactic.Abel

namespace TMEFuse

variable {E : Type*} [NormedAddCommGroup E]

public def IsNormClamp (Pi : E → E) (c : ℝ) : Prop :=
  ∀ v : E, ‖Pi v‖ ≤ c

public def trajectory (x0 : E) (Pi : E → E) (Δx : ℕ → E) (L : ℕ) : E :=
  Nat.recOn (motive := fun _ => E) L x0 (fun l acc => acc + Pi (Δx l))

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
    have h_sub_zero : x0 - x0 = (0 : E) := sub_self x0
    have h_traj_zero : trajectory x0 Pi Δx 0 = x0 := rfl
    rw [h_traj_zero, h_sub_zero, norm_zero]
    simp
  | succ L ih =>
    have h_traj_succ : trajectory x0 Pi Δx (L + 1) = trajectory x0 Pi Δx L + Pi (Δx L) := rfl
    rw [h_traj_succ]
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

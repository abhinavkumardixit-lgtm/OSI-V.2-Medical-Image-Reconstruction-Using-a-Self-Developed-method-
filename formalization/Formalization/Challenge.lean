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
  sorry

end TMEFuse

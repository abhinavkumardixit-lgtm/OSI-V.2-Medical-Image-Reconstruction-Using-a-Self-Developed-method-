import Lake
open Lake DSL

package "formalization" where
  version := v!"0.1.0"
  keywords := #["mathematics", "geometric-deep-learning", "lipschitz", "equivariant-gnn"]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.35.0-rc2"

@[default_target]
lean_lib "Formalization" where
  roots := #[`Formalization.Challenge, `Formalization.Solution]

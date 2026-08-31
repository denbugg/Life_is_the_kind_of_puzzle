# Solver step 79 — fixed focal-gated tail fresh pair confirmation

Parent: step 42. The unchanged logit-zero/tail96 consumer is the new local pair
leader on fresh32:

- pairs **`348.34375`** vs **`346.06250`**, delta **`+2.28125`**, source-cluster
  CI95 **`[+0.875,+3.59375]`**, W/T/L `19/7/6`;
- recall **`0.315528759`** vs `0.313462409`;
- exact `1.03125` vs `1.15625`, delta `-0.125`, CI95
  `[-0.3125,+0.0625]`.

Verdict: `pair-candidate-confirmed`, not exact-oriented and not production yet.
The panel is now opened and historically model-selection-exposed. Freeze the
rule and obtain one new unchanged roster confirmation before promotion; do not
sweep the focal threshold or tail budget.

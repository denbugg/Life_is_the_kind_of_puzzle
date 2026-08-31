# Solver step 83 — focal-gated tail independent fresh16 confirmation

Parent: step 79. The candidate was unchanged: target350 matcher,
raw/logistic/focal-top5/nonlinear all-bond selector, and tail96 with protection
limited to frozen recovered focal `train_exact_top5` logits `>=0.0`.

One preregistered current-lineage-disjoint `16 sources × 2 draws` panel gave:

- pairs **`354.750`** vs all-edge control `352.875`, delta **`+1.875`**,
  source-cluster CI95 **`[-0.1875,+3.84375]`**, case W/T/L `24/0/8`;
- recall **`0.321331522`** vs `0.319633152`;
- exact **`2.50000`** vs `2.28125`, delta `+0.21875`, CI95
  `[-0.09375,+0.53125]`.

The preregistered pair gate required delta mean `>=+0.5` and CI95 lower
`>=-0.25`; both passed. Verdict: focal-logit-zero protection is confirmed for
pair-default promotion. This step did not change production code and does not
claim universal/model freshness. Nearby threshold or tail-budget sweep is
closed.

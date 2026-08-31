# Solver step 94 — full-resolution union + focal tail independent confirmation

Parent: step 88 in both Weco tracks. The candidate was frozen before scoring:
the same-pass current four-arm portfolio, one full-resolution-denoiser union
arm, original all-1104-edge selector, and the focal-logit-zero protected
tail96. Restored pixels were matcher-only; every output remained a strict
permutation of the 576 original upright tiles.

One preregistered lineage-disjoint `16 sources × 2 draws` panel gave:

- combo **`356.3125` pairs**, recall **`0.322746830`**, exact `8.0`;
- same-pass control `348.40625`, `0.315585371`, exact `8.0`;
- combo-control pair delta **`+7.90625`**, source-cluster CI95
  **`[+3.53125,+12.96875]`**, case W/T/L `24/0/8`;
- full-resolution arm alone added `+5.4375` pairs over control, CI95
  `[+1.25,+10.6875]`; focal-gated tail then added another `+2.46875` over
  that arm, CI95 `[+0.34375,+4.6875]`.

The preregistered gate required combo-control pair delta mean `>=+2.0` and
CI95 lower `>=0.0`; both passed. Exact placement was neutral on this roster.
Verdict: the full-resolution matcher supply and focal-gated tail are jointly
confirmed for the pair-focused lineage. No competition test or postprocessing
was used, and the official submission baseline was not replaced.

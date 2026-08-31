# Solver step 90 — selective target500, local32

Parent in pair and exact Weco runs: step 83.

One target500 matcher pass supplied a same-pass current350 subset.  Only new
edges with frozen recovered focal top5 logit `>=0` entered one fifth arm; the
original all-bond selector and focal-gated tail96 were unchanged.

- pairs `314.40625 -> 323.62500`, delta **`+9.21875`**, CI95
  **`[+4.34375,+14.93750]`**;
- recall `0.284788270 -> 0.293138587`;
- exact `1.28125 -> 1.56250`, delta `+0.28125`, CI95
  `[-0.1875,+0.9375]`.

The nonnegative local pair gate passed. Same-pass focal-gated control matched
the historical control on `32/32` layouts.

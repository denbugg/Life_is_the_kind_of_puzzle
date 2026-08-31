# Solver step 55 — focal-feature stacker is exact-positive but pair-negative on held

Parent in both Weco runs: step 54.

The unchanged held32 replay used only previously frozen TASKA matrices,
candidate edges, and recovered focal logits/features. No matcher, training, or
parameter choice was repeated after the local gate.

The standalone stacker reached **331.84375 pairs**, recall **0.300583107**, and
**3.40625 exact tiles**. The five-arm selector plus tail96 reached **337.03125
pairs**, recall **0.305281929**, and **3.28125 exact tiles**, versus the retained
four-arm-tail96 **337.5625 / 0.305763134 / 3.0625**.

Five-minus-four was **-0.53125 pairs**, source-cluster CI95
`[-2.4375,+1.375]`, case W/T/L `2/25/5`. Exact was **+0.21875**, CI95
`[+0.03125,+0.46875]`, case W/T/L `4/28/0`. Thus this is a reproducible
exact-only signal, not a pair improvement. The preregistered held pair gate
failed, so the pair-promotion path stopped. Its strictly positive held exact
interval later triggered the explicitly labelled, unchanged exact-oriented
override recorded in step 56; the current four-arm-tail96 pair default remains
retained.

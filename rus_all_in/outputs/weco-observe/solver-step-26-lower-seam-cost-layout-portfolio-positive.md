# Solver step 26: lower-seam-cost layout portfolio transfers

The fixed train256 edge calibrator and raw fused-cost ordering sometimes make
different legal TASKA layouts.  A target-free selector now computes the sum of
the original TASKA costs on all 1,104 realised right/down board bonds and keeps
the lower-cost strict permutation (raw wins exact cost ties).

On opened32 the selector produced **336.8125 pairs**, recall
**0.305083786**, and **4.75 exact tiles**, versus 334.71875 / 0.303187274 /
4.46875 for raw.  The pair delta was +2.09375 with source-clustered CI95
`[+0.25, +4.03125]`.

On held300 it produced **335.25 pairs**, recall **0.303668478**, and
**2.9375 exact tiles**, versus 329.625 / 0.298573370 / 2.90625 for raw and
333.90625 / 0.302451313 / 2.71875 for calibrated.  Pair gain versus raw was
+5.625 with CI95 `[-0.3125, +14.96875]`; exact was essentially neutral.

The positive direction transfers across panels and the opened pair interval is
strictly positive.  This is retained as a cheap portfolio primitive, but the
held panel remains historically model-selection-exposed and cannot establish a
fresh promotion by itself.

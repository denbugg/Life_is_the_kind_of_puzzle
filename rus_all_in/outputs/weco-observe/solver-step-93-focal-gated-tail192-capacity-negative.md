# Solver step 93 — focal-gated tail192 capacity is pair-negative

Parent: step 83 in both Weco tracks.

One preregistered capacity step compared the confirmed focal-logit-zero
non-adjacent tail96 with an otherwise identical tail192. A new deterministic
source16×draw2 roster was SHA-reserved before candidate code/scoring and had
zero overlap with signed TASKA train256/extension/focal-train,
local/held/fresh/fresh16, fullres-denoiser, and active-panel rosters.

Tail192 scored **322.78125 pairs**, recall `0.292374321`, exact `2.18750`;
tail96 scored **323.09375**, `0.292657382`, `2.03125`. Pair delta was
**−0.31250**, source-cluster CI95 `[-1.28125,+0.68750]`, case W/T/L
`11/6/15`. The fixed `mean>=+0.5` and `CI lower>=−0.25` pair gate failed.
Exact secondary delta was `+0.15625`, CI95 `[0,+0.28125]`.

The extra budget was active: mean accepted swaps increased by `54.47`, and
tail192 hit its own cap on 9/32 cases. More minimization of the original seam
objective nevertheless reduced true pairs. Retain focal-gated tail96; do not
promote or sweep nearby budgets/thresholds on this panel.

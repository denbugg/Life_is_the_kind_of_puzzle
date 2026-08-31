# Solver step 52: structural-border cyclic origin is negative

Starting only from recovered focal top-5, recompute the audited TASKA v3+local
structural border unary (`slack=6`, 20 Sinkhorn iterations), enumerate all 576
whole-board cyclic rolls, and maximise unary on physical border positions with
stable row-major ties. No seam/border mixing or target-derived inference input
is used.

Opened32 changed all 32 origins and reached **323.625 pairs**, recall
**0.293138587**, and **3.84375 exact tiles**, versus focal's 335.5 /
0.303894928 / 4.34375. Deltas are -11.875 pairs and -0.5 exact; pair CI95 is
[-14.25, -9.375]. The preregistered exact-positive and pair-loss-at-most-two
gate failed, so held300 is not opened and step 53 is intentionally absent.


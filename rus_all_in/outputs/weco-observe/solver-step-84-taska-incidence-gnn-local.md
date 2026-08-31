# Solver step 84 — TASKA incidence-GNN local gate

Parent: step 54 in both Weco tracks.

One fixed width-64, two-block permutation-equivariant incidence-GNN was trained
for exactly 400 one-board AdamW steps only on source-disjoint train256 indices
128:256. It aggregated mean/max current edge states by outgoing source and
incoming target within each board/axis, then added a bounded `2*tanh` residual
to the frozen recovered-focal logit. No architecture or epoch sweep.

On excluded local32, standalone scored `307.21875` pairs, recall
`0.278277853`, exact `1.50000`. Five-arm+tail96 scored **`314.71875 /`
`0.285071332 / 1.40625`** against four-arm **`314.37500 / 0.284759964 /
1.37500`**. Pair delta `+0.34375`, CI95 `[-1.125,+2.125]`; exact delta
`+0.03125`. The nonnegative pair gate passed and opened held32.

All candidates were frozen before reference reconstruction and remained strict
permutations of the original upright tiles.

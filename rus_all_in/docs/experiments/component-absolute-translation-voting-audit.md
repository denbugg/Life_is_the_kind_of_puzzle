# Component-level absolute translation voting: duplication and oracle audit

Status: **design-only conditional go for an independent, board-conditioned
component placer; no-go for a simple shared global-roll vote or another
component-only shift head.** No source was selected or opened, no model/code
was created, and no production artifact changed for this audit.

## What has already been tested

| Existing family | Contract and result | Duplication boundary |
|---|---|---|
| Absolute-coordinate sorter | Per-tile d64/set-context row/column/slot logits are summed over each rigid component and feasible translation. Fresh exact128 gave `+0.3125` exact tile/board but CI crossed zero; row signal was reliable, column was not. | Summing independently produced tile→absolute scores over a component is already tested. A new model must jointly encode the actual component and current board. |
| Component-translation CE scale-up | Trained the same tile logits with exact truth-consistent predicted-component shift CE. Shift top1 was `0.679%` vs `0.184%` chance; only `48.320` tiles in `18.531` pure nontrivial components were supervised per board. Material exact gate failed. | More steps, width or a new scale on this summed-logit objective is a duplicate. |
| Explicit component-shift head | Directly mean/max-pooled d32 member tokens, relative coordinates, component geometry/confidence and board mean, then predicted row/column shift. Train-only joint accuracy was `0.409%` vs `0.206%` chance; predicted support `1.83` vs dominant-mode oracle `417.17` tiles/board. Column NLL gain was `0.013%`. | A component set head over the same d32/member/shape contract is already closed. Full-resolution rendered component pixels only count as new evidence if paired with joint current-board conditioning and an inference-matched purity head. |
| Coordinate cyclic origin | Aggregated per-tile absolute profiles over one common roll. Best equal coordinate/Socket blend improved cyclic5 by only `+0.0547` tile/board. | Any component scores reduced to one shared 24×24 roll vote are still a coordinate-cyclic origin scorer. They cannot independently correct component offsets. |
| Whole-layout origin CNN / frame-side origin | Whole-grid learned roll ranking had R@1/R@5 `0/0%`; marginal true frame sets reached only `1.156` exact while the same layout's best-roll oracle had `13.031`. | A single whole-board origin head, marginal border vote, face/centre or background heuristic is closed. |
| Foundation semantic component proposal | A frozen DINO/population-field score summed isolated tile votes over components. It was stopped as a duplicate of absolute positional probes and their fit/transfer gap. | A rendered component alone mapped to population absolute position remains the same risky family. The candidate must be conditioned on all components of the current board, not a train-population atlas. |

## Measured support ceilings from already persisted reports

These numbers are **oracle tile-support ceilings, not collision-aware strict
layout scores**. They do not authorize a panel.

### Exact synthetic d64 panel: pure components

The already-opened source64×draw2 component-translation report has exact
synthetic shuffle truth. It found on average:

- `18.53125` truth-consistent nontrivial predicted components per board;
- `48.3203125` tiles inside them (`8.389%` of the board);
- maximum pure-component size `5.59375`, mean size `2.5363`.

Thus an oracle that identifies every pure component and its independent legal
absolute translation could directly support at most **48.32 tiles/board before
packing/collision loss** on this panel. This is materially above the current
one-tile exact regime, but the pieces are numerous and small; it is not evidence
that a shared global shift can realize that support.

Report SHA-256:
`894cf97731fcdb5df05f4409b93f6821fcd05a4f9d282612aa2b8999075c5505`.

### Frozen recovered-reference dev24: top-k component slices

The older component-anchor report persists every nontrivial component with
size and target-assisted translation purity. Re-aggregating only that JSON,
without decoding another target, gives:

| component selection per board | k=1 | k=2 | k=4 | k=8 |
|---|---:|---:|---:|---:|
| largest decoder components: total tiles moved | `42.42` | `67.04` | `101.88` | `144.25` |
| oracle dominant-shift correct support within those components | `5.42` | `10.75` | `18.42` | `31.04` |
| internally exact tiles among those largest components | `0.00` | `0.00` | `0.00` | `0.00` |
| oracle-selected largest *pure* components | `2.83` | `4.79` | `7.75` | `11.25` |

Across all nontrivial pure components the same recovered-reference ceiling is
`18.17` tiles/board. The nominal largest/high-confidence component averaged
`42.42` tiles but only `17.31%` trusted translation purity, and `0/24` largest
components were internally exact. Therefore size/current decoder confidence is
not a usable purity gate: top-k large voting moves much more incorrect than
correct geometry, even under an oracle absolute offset. The pure top-k row is
an optimistic target-assisted selector, not an inference method.

This panel uses recovered rather than authoritative permutation truth, so it
is supporting mechanism evidence only. Report SHA-256:
`01b1e803c18aa75c40763afb3941a9b2c1cb945cc2a24a8c0a588d2c38aaba23`.

The newer opened source40 conversion audit reports `295.1` tiles in internally
exact components, but that count includes trivially pure singletons. It is not
used as the nontrivial component ceiling. Its useful fact is instead the
unchanged origin gap: best global cyclic roll `14.25` exact versus dirty origin
`0.75`.

## Materially distinct design, if activated later

A justified candidate must make **independent component placement decisions**,
not collapse them into one global roll:

1. Build the exact inference-time decoder components, retaining impure false
   bridges rather than filtering by target truth.
2. Render each component as a masked native-resolution mosaic of original
   upright tiles. Encode raw + normalized + optional matcher-only fullres
   restored boundary sequences without spatial downsampling.
3. Feed all component tokens through a permutation-equivariant board-level set
   model. Each component token must be conditioned on the other components and
   current-board global context; no source ID, population field, face/centre or
   background rule is allowed.
4. Predict two calibrated outputs per component: probability of internal
   translation purity, and a categorical distribution over feasible absolute
   `(row, column)` translations. Supervise impure components with their
   dominant-support distribution and purity, not by silently dropping them.
5. Use a strict collision-aware component-to-shift assignment/packing solver.
   Freeze a small top-k/cost budget from train-only calibration. Fill unresolved
   tiles with the unchanged decoder policy and retain all 576 original upright
   tiles exactly once.

This differs from the failed explicit head only through two substantive new
information paths: joint native-pixel component rendering and current-board
component-to-component conditioning. Removing either makes it a duplicate.

## Go/no-go

- **No-go** for “each component votes, average votes, choose one 24×24 global
  roll”: it duplicates coordinate cyclic origin/whole-layout origin and cannot
  realize the 48.32-tile independent-component ceiling.
- **No-go** for a larger d32 component shift MLP, isolated rendered-component
  absolute classifier, confidence=component-size rule, or DINO/population
  position field.
- **Conditional go** only for the joint board-conditioned rendered-component
  model above, and only after a cheap train-only signal probe. Before any exact
  panel, it must materially beat the explicit head on both column NLL and
  purity-ranked supported tiles; otherwise stop without evaluation.

The decisive risk is not capacity but purity selection: existing largest
components are false-bridge mixtures, while genuinely pure components are
small. A future preregistration should therefore make top-k purity precision
and collision-free oracle-realization rate primary local gates, with exact and
adjacency assessed only after they pass.

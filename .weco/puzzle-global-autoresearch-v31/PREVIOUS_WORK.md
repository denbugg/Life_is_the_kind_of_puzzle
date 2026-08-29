# Previous work audit

## Current baseline

- Commit: `1a714e115d2be1aa21813f4a6300771af61bf159` (V30).
- Fixed 15-scene development benchmark: mean adjacency `0.1057367150`, aligned
  placement `0.0212962963`, composite `0.1110607890`, tile coverage `1.0`.
- Same-run V30 baseline adjacency was `0.0971618357`; V30 improved it by 8.83%.
- V30 won adjacency on 10/15 scenes, tied one, and lost four.

This 15-scene set has already influenced development. It is evidence for relative
progress, not an untouched test set. V31 must select changes on the separate
6981--6988 validation scenes and report the fixed 15 only after selection.

## What V29/V30 established

- V29's candidate oracle composite was `0.1117779` versus its fixed candidate
  composite `0.1012077`: an oracle gap of about 10.54%.
- V30's selected composite `0.1110608` consumes roughly 93% of that old oracle
  gap. Selector-only tuning on the same candidate family is therefore nearly
  exhausted.
- The V30 portfolio is useful: packed1-unfreeze was selected for nine scenes,
  packed2 for three, packed4 for two, and unfreeze for one.
- The learned candidate objective is misaligned with adjacency. On V29 candidates
  its Pearson correlation with true adjacency was about `-0.070`.
- Sinkhorn refinement, the transferred V27 edge calibrator, full-matrix
  calibration, older QAP/SA/genetic variants, and earlier relative/GNN fusion did
  not improve the fixed benchmark and should not be repeated unchanged.

## Structural defects to fix

1. The coordinate/border GNN is trained on V27 matrices but used on fused V28
   matrices. This domain shift invalidates calibration.
2. The LNS destroy set uses `np.unique(... )[:width]`; sorting cell indices biases
   repair toward the top-left and may discard the actual weakest/random cells.
3. Hungarian repair scores all movable cells against stale neighbours. It is a
   linear surrogate for a coupled QAP, not an exact local objective.
4. Search is shallow: six seeds, one run per method, 18x96 LNS, no component moves,
   no adaptive restarts, and no multiscale repair.
5. The graph is fixed top-8 pairwise evidence. It lacks reciprocal rank/margin,
   cycle/2x2 consistency, component membership, and current-board feedback.
6. Checkpoint selection overweights border F1. Step 100 was selected, although
   step 700 had better row/column accuracy; selection should use downstream
   assembly quality.
7. Unary weight was selected from only four scenes, so `0.5` is weakly identified.

## Promising unfinished ideas

- Rank-normalized mutual confidence plus 2x2 weakest-link loop consensus. A prior
  plan exists but was never executed.
- Reciprocal-margin seed islands. Earlier diagnostics found about 95.4% precision
  at margin >0.7, though only about 49 edges/image.
- Multi-context frontier scoring: retrieval R@1 rose from 19.7% with one neighbour
  to 45.3% with four in prior diagnostics.
- Confidence-gated components, component-level multi-start, and multiscale LNS.
- Train heads/calibrators directly on the fused matrix domain and select them by
  held-out assembly metrics.
- A layout-level candidate critic trained with out-of-fold pairwise ranking.

## V31 implication

The first V31 generation should create a genuinely stronger candidate family:
fix the destroy bias, add reciprocal rank and loop-consensus energy, then use
component-aware multi-start and multiscale LNS. A learned board critic and
same-domain fused heads are second-stage improvements, not substitutes for better
search. Every candidate must remain a permutation of all 576 tiles.

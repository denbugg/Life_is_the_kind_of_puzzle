# Solver step 7: learned Union-hard priority improves pairs, not exact

Status: preregistered local gate failed only on the exact nonnegative clause,
while the pair-level objectives improved clearly on the frozen source-disjoint
16-source x 2-draw evaluation panel.

Treatment: a target-free 340D DeepSets residual head reorders only the frozen
1104 Union-v2 hard-edge identities.  It cannot introduce a new edge.  Features
combine Union geometry and confidence, raw/twin evidence, identity-matched
Direct evidence, and matcher-only full-resolution denoiser evidence.  The
unchanged decoder emits a strict permutation of the 576 original upright
20x20 tiles.

Matched 32-case result:

- exact tiles per board: `1.1875 -> 1.09375` (`-0.09375`, source-clustered
  95% CI `[-0.65625,+0.4375]`);
- adjacency recall: `0.1454936594 -> 0.1485790308` (`+0.0030853714`, 95% CI
  `[+0.0008491848,+0.0055480072]`);
- satisfied adjacent pairs: `160.625 -> 164.03125` of 1104
  (`+3.40625` pairs per board);
- correct fixed top288 hard edges: `150.0625 -> 153.6875` (`+3.625`, 95% CI
  `[+2.59375,+4.625]`);
- fixed-top288 wins/ties/losses: `15 / 0 / 1` by source;
- all emitted layouts strict: yes.

The edge-ranking signal replicated from fit (`+3.2109375` fixed-top288) to
held-out evaluation (`+3.625`) and is therefore worth retaining as a pair-level
building block.  It is not promoted as the exact-layout winner: the exact
delta is slightly negative and noisy.  The next experiment must test whether
a target-free decoder/composition can convert the extra correct local edges
into globally anchored layouts; do not tune a selector on this opened panel.

Frozen report:
`outputs/union-hard-edge-priority/pilot-v1-final/report.json`
(`sha256 c4cf10f37f10a709e5390f2bd05555ecf0304ab958f7ca6ebde713cbb9f17e5e`).

# Order-2 Growing Consensus audit

Decision: `closed_no_promotion`

This branch is a faithful, bounded pilot of the missing-edge inference step in
Son et al.'s Growing Consensus method. The implementation is correct and
materially different from the existing reciprocal, soft-cycle, two-side, and
translation-consensus heuristics, but the first frozen exposed-data gate is a
strong negative. It must not be used in production or a submission.

## Primary-source audit

- Son et al., *Solving Small-Piece Jigsaw Puzzles by Growing Consensus*,
  CVPR 2016: <https://openaccess.thecvf.com/content_cvpr_2016/papers/Son_Solving_Small-Piece_Jigsaw_CVPR_2016_paper.pdf>
- Vardi et al., *A Multi-Phase Relaxation Labeling Framework for Solving
  Jigsaw Puzzles*, VISAPP 2023: <https://www.scitepress.org/Papers/2023/116228/116228.pdf>
- Official Vardi implementation:
  <https://github.com/BenVr/multi-phase-rl-for-square-puzzles>

The existing `faithful_multi_phase_relaxation_solver` already implements the
substantive MPRL mechanics: the uniform barycenter, adaptive similarity,
multiplicative row-normalized relaxation, alpha=0.7 single-frontier anchoring,
uniform phase reset, and ALC convergence. The documented omission is the
boundary translation/four-translation branch. Translation cannot repair the
observed adjacency failure because neighbour relations are invariant to a
global translation. Existing exact exposed reports give faithful MPRL
adjacency 0.007246 / SSIM 0.188461 on primary2 and adjacency 0.009511 / SSIM
0.182898 on independent2, versus soft-cycle component adjacency 0.128623 and
0.129982 respectively. On real4, faithful MPRL denoised-render SSIM is 0.141865
versus 0.183733 for QAP. MPRL is therefore closed without another implementation
or tuning run.

Son's method contains one inference operation not present in the existing
solvers: a directed edge absent from the candidate graph can be proposed when
multiple distinct three-edge 2x2 configurations imply it, independently of
the missing edge's score magnitude. `discover_order2_consensus` implements
exactly this minimal order-2 operation. It emits proposals only; it neither
hard-locks edges nor constructs a layout.

## Frozen pilot protocol

- Data: exposed `edge_development`, offset 0, limit 2; source IDs
  `img_005666.png` and `img_003853.png`.
- Corruption panels: `primary_kornia` and `independent_libjpeg`.
- View: selected TileNAF denoiser output.
- Candidate graph: the existing denoised `C1 + L1w4` score bank, top-k 8 in
  each of the right/down directions.
- Proposal gate: at least 13 distinct incomplete-square witnesses.
- Use in solver: bounded soft cost bonus 0.05, support strength capped at 2;
  never a hard lock.
- Comparator: the existing `component_l1_softcycle_k8_p1` seed followed by
  identical QAP settings (25 iterations, 2 restarts, boundary weight 0.05,
  8 swap-refinement passes, identical deterministic seed).
- The support threshold was fixed before evaluating exact target relations. On
  an input-only iid-null diagnostic with N=576 and k=8, support q99/q99.9/max
  was 6/8/12; the Poisson approximation gives about 0.036 support>=13 false
  proposals across all directed edges.
- No sealed targets, real-target inspection, oracle labels, Kaggle, or
  label-driven threshold/bonus tuning were used.

## Exact exposed results

| Panel | Variant | R@1 | R@5 | R@10 | R@32 | MRR | QAP adjacency | Layout SSIM |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| primary2 | baseline | 0.161232 | 0.332428 | 0.432065 | 0.660779 | 0.252526 | 0.068388 | 0.255208 |
| primary2 | order-2 | 0.088315 | 0.254529 | 0.368659 | 0.660326 | 0.182052 | 0.059783 | 0.253970 |
| primary2 | delta | -0.072917 | -0.077899 | -0.063406 | -0.000453 | -0.070474 | -0.008605 | -0.001238 |
| independent2 | baseline | 0.145380 | 0.319293 | 0.423913 | 0.674819 | 0.242601 | 0.072464 | 0.253043 |
| independent2 | order-2 | 0.081522 | 0.247283 | 0.374094 | 0.667572 | 0.177820 | 0.065217 | 0.250423 |
| independent2 | delta | -0.063859 | -0.072011 | -0.049819 | -0.007246 | -0.064781 | -0.007246 | -0.002620 |

The structured top-k graphs invalidate the iid intuition: support>=13 still
produces 14,983-20,085 proposals per image. Proposal precision is only
0.702%-0.839%, around two orders of magnitude below the project's 99.5%
hard-lock precision gate.

| Panel / source | Proposals | Precision | Complete loops | True-edge coverage | Support mean / max | Adjacency delta | SSIM delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| primary / `img_005666.png` | 20,085 | 0.7020% | 43,352 | 0.127717 | 21.36 / 115 | -0.013587 | -0.006257 |
| primary / `img_003853.png` | 14,983 | 0.7475% | 31,816 | 0.101449 | 20.36 / 102 | -0.003623 | +0.003781 |
| independent / `img_005666.png` | 18,481 | 0.8387% | 45,937 | 0.140399 | 21.70 / 109 | -0.015399 | -0.006316 |
| independent / `img_003853.png` | 16,774 | 0.8227% | 32,807 | 0.125000 | 20.49 / 98 | +0.000906 | +0.001076 |

The result is not a marginal miss: retrieval collapses on both independent
corruption panels, while macro QAP adjacency and SSIM both worsen. The branch
is closed at the first cheap gate. Trying nearby support thresholds or bonuses
after reading these labels would be post-hoc tuning and was intentionally not
done. Reopening is justified only if a fundamentally higher-precision candidate
graph becomes available.

## Correctness and artifacts

- Deterministic core: `src/puzzle_assembly/growing_consensus.py`.
- Synthetic tests: `tests/test_growing_consensus.py`.
- Evaluation integration: `scripts/evaluate_assembly_baselines.py`.
- Canonical reports: `exact_primary2.json` and `exact_independent2.json`.
- Timing-accounting-only predecessor reports are retained as
  `*_v0_timing_accounting.json`; all scientific fields are identical to the
  canonical reruns, and only timing measurements differ.
- Verification: 10 focused unit tests plus an independent exhaustive
  enumerator comparison on 50 random seven-node graphs; final related suite is
  22 passed. Python compilation also passed.

Artifact hashes are recorded in `SHA256SUMS.txt`.

# SGT1 Evidence Report — Sparse Candidate-Graph Transformer

**Experiment family:** ORBIT-24 SGT1  
**Status:** **Rejected before cache expansion, buddies solve, or real-input SSIM run**

## Mechanism tested

SGT1 is a **1,080,255-parameter** edge-aware graph Transformer. It consumes a frozen rank96-like directed candidate graph, not absolute tile slots. Each candidate edge carries a frozen score, its row-normalized value, reciprocal score/presence, rank, direction, and a validity mask. The model alternates global permutation-equivariant tile attention with directed sparse edge-to-node/node-to-edge message updates, then predicts a residual correction to the frozen edge score.

The candidate source itself is incomplete: a read-only audit of 20 cached boards at the rank96 budget found mean true directed-neighbour coverage **68.44%** (minimum **56.16%**). Frozen top-1 is **17.59%** over all valid edges and **25.61%** conditional on coverage. Thus the experiment can only test reranking precision among existing candidates; it cannot repair missing edges.

## Numerical validation

The first training attempt exposed padded `-inf` scores entering normalization. It was stopped immediately rather than interpreted as model evidence. The cache loader was repaired to compute finite-only statistics and to mask invalid candidates out of both loss and decode. A one-step GPU smoke was finite and preserved the frozen baseline up to the initial learned residual.

## Fixed-board capacity control

Two cached FIT boards (`img_000000`, `img_000001`) were trained for 1,600 steps with K=96, 128-dimensional nodes, 3 graph blocks and learning rate 0.001. The registered capacity criterion was ≥95% top-1 **conditional on covered true edges** for both boards.

| Metric | Frozen candidate score | SGT1 after capacity training |
|---|---:|---:|
| Board 0 covered top-1 | 31.36% | **100.00%** |
| Board 1 covered top-1 | 27.14% | **100.00%** |
| Mean delta | — | **+70.75 pp** |

The capacity gate passed. This establishes that the architecture and loss can express a reranking function over the frozen graph.

## Source-disjoint pilot

The same model/training contract then fit cached **FIT** boards `img_000000`, `img_000001` and evaluated the two cached source-disjoint **DEV** boards `img_000014`, `img_000020`, whose membership was verified against the pinned 5,360/670/670/300 split manifest. This is an informative two-board pilot, not a full eight-board DEV gate.

| DEV board | Frozen covered top-1 | SGT1 covered top-1 | Delta |
|---|---:|---:|---:|
| `img_000014` | 21.10% | 16.18% | **−4.93 pp** |
| `img_000020` | 25.02% | 21.58% | **−3.43 pp** |
| Mean | 23.06% | 18.88% | **−4.18 pp** |

The negative delta on both independent DEV boards falsifies the declared mechanism for the present training regime. The model learned two static graphs but made their local structural score pattern less transferable, despite its successful capacity control.

## Decision

**Reject SGT1 v1.** Do not generate further candidate caches, do not invoke the buddies solver on its scores, and do not run real-input SSIM. The negative source-disjoint ranking pilot is sufficient to stop this route before more compute is spent. This result also narrows the interpretation of the capacity pass: raw score-pattern message passing can memorize a small fixed graph but lacks a transferable visual or scene-conditioned representation.

The canonical benchmark remains `submission_rank96_v1.zip` at user-confirmed SSIM **0.2161981413457065**. SGT1 does not improve, or compare against, that score.

## Artifacts

| Artifact | Path |
|---|---|
| Candidate ceiling audit | `E:\pazzle_work\pazzle_fixed_orientation_20260813\SGT1_sparse_graph\candidate_coverage_k96_probe.log` |
| Capacity report | `E:\pazzle_work\pazzle_fixed_orientation_20260813\SGT1_sparse_graph\sgt1_capacity_finite_report.json` |
| Source-disjoint pilot | `E:\pazzle_work\pazzle_fixed_orientation_20260813\SGT1_sparse_graph\sgt1_source_disjoint_pilot.json` |
| Harness | `src/train_sgt1_sparse_graph.py` |

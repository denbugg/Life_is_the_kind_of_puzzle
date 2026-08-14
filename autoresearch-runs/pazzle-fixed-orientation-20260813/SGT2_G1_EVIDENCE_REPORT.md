# SGT2-V G1 Evidence Report — Rejected

**Experiment:** `SGT2_visual_sparse_candidate_ranker`  
**Decision:** **Rejected before any global-layout or submission use.**  
**Scope:** A frozen rank96 candidate list was retained exactly. The model consumed only corrupted train-input tile pixels plus pre-existing candidate graph caches; supervision evaluated only neighbours present in the candidate list. No test image, target image, candidate expansion, layout solver, rotation, or submission artifact was used.

## Question

Could a small direction-aware visual patch encoder make sparse candidate graph reranking transfer across source-disjoint scenes, correcting SGT1's failure mode of score-only graph message passing?

## Protocol

The G0 adapter materialized visual caches for 20 frozen candidate graphs with strict source alignment: 17 FIT, 1 CAL and 2 DEV. Each cache preserved 24Û24 upright tile indexing, candidate IDs, finite-score ordering, and covered-neighbour labels. G1 trained a 32-width directional strip CNN plus candidate-pair residual scorer for 600 CUDA steps on FIT boards `image_0000_k64.npz` and `image_0001_k64.npz`, then evaluated on source-disjoint DEV boards `image_0014_k64.npz` and `image_0020_k64.npz` at K=96.

| Gate quantity | Frozen candidate score | SGT2-V | Delta |
|---|---:|---:|---:|
| Mean covered-edge top-1 | 0.23060 | 0.15924 | **−0.07136** |
| DEV image_0014 covered top-1 | 0.21103 | 0.15809 | −0.05294 |
| DEV image_0020 covered top-1 | 0.25017 | 0.16040 | −0.08977 |
| Mean candidate coverage | 0.65104 | 0.65104 | 0.00000 |

The pre-registered local gate required a positive source-disjoint delta; expansion to a larger DEV cache required at least +1.0 pp. SGT2-V fails both conditions decisively.

## Mechanism audit

Training loss fell monotonically from **3.0791** to **1.8059**, while DEV covered top-1 deteriorated monotonically from approximately the frozen baseline at step 1 to **−7.14 pp** at step 600. Thus the visual encoder learned FIT-scene-specific compatibility shortcuts rather than a transferable edge representation. The mechanism is **refuted** under this data scale and objective.

The fixed candidate coverage is important: failure is not due to hiding true edges or changing rank96 candidate mining. It is a reranking generalization failure. SGT2 must not be composed with the current rank96 solver, and no SGT2 submission variant should be produced.

## Next solver lever

The next lever must avoid supervised source-specific edge residual fitting. A candidate route is **assignment-time structural self-consistency without learned score residuals**, evaluated only if it introduces new information beyond rejeced C1 cycles and SGT1/SGT2 rerankers. In parallel, broad-scene R6 remains a restoration lever, not a solver improvement; it is not a substitute for measurable placement accuracy.

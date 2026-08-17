# P31 Pre-Registration: BHCS-24

> **Status:** PRE-REGISTERED BEFORE IMPLEMENTATION — 2026-08-17.

**Experiment:** BHCS-24 — Boundary Hard-Contrastive Scorer.

## Evidence and rationale

P29 shows that DINOv2 features improve **proposal coverage** but cannot be converted to reliable adjacency with shallow score fusion. P30 shows that reciprocal retrieval ranks do not repair that limitation. P26 rejected a lightweight scorer over full tiles; P24/P25 established that unbounded all-source cross-reranking is operationally infeasible. P31 therefore isolates the information source that defines an adjacency: a small raw-pixel strip on both sides of the candidate seam.

This design follows the separation of pairwise compatibility from global assembly used by learned compatibility pipelines. JigsawNet explicitly applies CNN computation to the stitching region and combines the pairwise measure with global composition.[1] Edge2Vec similarly targets the efficiency/accuracy gap for pairwise compatibility through embeddings and hard-batch triplet learning.[2] These sources motivate a seam-only, hard-negative ranking loss rather than another full-tile model, alpha fusion, or rank transformation.

## Locked model and score construction

A directed candidate `i -> j` is represented by a canonical 8×20 RGB seam tensor: the four boundary-adjacent pixels from tile `i` followed by the four opposite-boundary pixels from tile `j`. Vertical seams are transposed to the same canonical orientation. A compact FP32 2-D CNN directly scores this seam. It uses no DINO descriptor, frozen rank96 score, P8 artifact, target pixels, or absolute coordinates as model input.

Training uses the fixed 96 FIT-train sources only. Every valid directed ground-truth adjacency is a positive. For each positive, the deterministic hard-negative pool is the raw RGB seam candidates with the closest baseline seam mismatch plus a deterministic source-keyed random subset; its true neighbor is removed. The loss is a fixed margin-ranking objective with one positive and eight negative seams per anchor. The model remains strictly FP32; AMP/FP16 is prohibited.

At inference, P31 streams all 576×576 directed seams direction-by-direction in batches capped at 16,384 pairs. It materializes only a 576×576 FP32 score matrix per direction on CPU and passes the learned right/down scores to the canonical `solve_buddies_from_scores` API. It does not use a frozen score fusion or a candidate union; the learned seam score is the entire solver field.

## Leakage, split, and resource lock

G0/G1 are input-only. G2 training may read cached labels solely for the fixed 96 FIT-train sources. FIT-selection is the distinct fixed 32-source subset. CAL, DEV, held, test, all target PNGs, and the single exceptional CAL target are forbidden until a gate explicitly authorizes them. P8 checkpoints, score caches, labels, imports, filenames, and derivatives are forbidden. All large artifacts must remain under `E:\pazzle_work\pazzle_fixed_orientation_20260813\P31_bhcs`.

Training is capped at 40 epochs, 2.5 million sampled seams per epoch, and 45 GPU minutes. Feature/pair caches must checkpoint every four sources. Inference is stopped if one board exceeds 90 seconds or 10 GB process memory. Any failed cap is a fast-futility rejection, not a reason to expand capacity.

## Staged gates

| Gate | Permitted data | Locked pass criterion | Failure action |
|---|---|---|---|
| G0 | Synthetic seam tensors only | Finite directional scores, valid 24×24 fixed-orientation bijection, and exact recovery of an unambiguous synthetic seam graph | Reject before real data |
| G1 | 16 FIT inputs only, no labels | Canonical horizontal/vertical seam construction is shape-correct, direction-sensitive, deterministic, and batch streaming stays within the resource cap | Reject before labels |
| G2 | 96 FIT-train cached labels only | Learned full-score recall@20 improves by **>= +1.0 pp** over frozen rank96 baseline recall@20 on the 96 FIT-train sources; zero invalid score matrices | Reject before FIT-selection |
| G3 | Fixed 32 FIT-selection cached labels only | Frozen checkpoint improves recall@20 by **>= +1.0 pp** over frozen rank96; canonical solver produces zero invalid boards; placement is non-inferior to frozen rank96 placement | Reject before held |
| Held (only after G3) | Exactly pinned held-32 cached labels only | Recall@20 gain >= +2.0 pp, placement >= 0.03189887152777778, and zero invalid boards | Preserve evidence; only then consider a submission |

## Falsification

If a seam-only hard-contrastive CNN cannot exceed the source-disjoint gate, the problem is not recoverable by local raw-boundary compatibility at this capacity. The next lever must instead introduce independent global/absolute-position evidence or a substantially different global assignment model; no further seam-CNN hyperparameter sweep is authorized.

## References

[1] Le and Li, *JigsawNet: Shredded Image Reassembly Using Convolutional Neural Network and Loop-Based Composition*, IEEE Transactions on Image Processing, 2019. https://ieeexplore.ieee.org/abstract/document/8661593

[2] Rika et al., *Edge2Vec: A High Quality Embedding for the Jigsaw Puzzle Problem*, arXiv:2211.07771, 2022. https://arxiv.org/abs/2211.07771

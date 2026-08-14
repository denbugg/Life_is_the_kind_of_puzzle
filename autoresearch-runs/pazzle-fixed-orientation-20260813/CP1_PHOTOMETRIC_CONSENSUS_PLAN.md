# CP1 — Candidate-Conditioned Photometric Consensus

**Hypothesis.** Independently corrupted tile brightness/contrast can be estimated jointly at inference time from high-confidence input-only provisional rank96 adjacencies. Per-tile affine RGB corrections should reduce nuisance seam discontinuity and improve the ordering of *already covered* K=96 candidates on source-disjoint boards.

**Mechanism.** Provisional adjacency bands provide noisy sparse correspondences → robust regularized per-tile color transforms estimate a common photometric frame → corrected continuity/derivative seam score reranks only existing candidates → more true neighbours become top-ranked.

**Expected delta.** Greater than +1.0 pp source-disjoint covered top-1 at unchanged K=96; no candidate coverage change.

**Falsification.** Reject on any non-finite/color-clamp instability, delta ≤0 on source-disjoint local gate, candidate-set change, or non-positive shared-layout paired SSIM gate.

**Protocol.** Fixed upright orientation; no test/target/permutation use in provisional calibration; raw rank96 candidate universe frozen; only CAL selects score fusion weights. CP1 is distinct from C1, SGT1, SGT2, F1P and PN2 because it jointly solves deterministic board-level per-tile corrections from an input-only provisional graph rather than learning a residual.

**Sources.** Park et al., CVPR 2016, https://www.microsoft.com/en-us/research/publication/efficient-color-consistency-for-community-photos/; Kovalsky et al., SIIMS 2015, https://doi.org/10.1137/140987869.

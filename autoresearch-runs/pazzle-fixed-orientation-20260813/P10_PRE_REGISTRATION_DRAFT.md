# ORBIT-24 P10 — Conditional Pre-registration: Layout-Conditioned Absolute Position Sinkhorn Refiner

**Status:** *Conditional and not started.* P9 G1 is the active experiment. This document authorizes P10 implementation only if P9 is rejected before CAL. No P10 model, cache, target read beyond FIT, CAL/DEV/test access, or GPU run has occurred.

## Motivation

P5's direct set-to-grid Transformer failed because it had no explicit positional state. P6 introduced noisy positions and yielded a small positive signal, but its coordinate-diffusion decoder was too weak. P10 retains the orthogonal global-assignment idea but conditions every tile on a concrete canonical rank96+buddies layout. This gives the model an explicit spatial hypothesis to correct, rather than asking it to infer the 24×24 coordinate system from an unordered set.

The design follows the assignment framing of square-puzzle assembly and differentiable doubly-stochastic permutation learning.[1][2] It is deliberately distinct from P8: it will **not** import P8 cache labels, P8 scorer weights, candidate ordering, or any learned P8 score.

## Fixed scope

| Aspect | Locked P10 choice |
|---|---|
| Orientation | Fixed; no rotation branch |
| Inputs | Corrupted shuffled FIT tiles, frozen canonical rank96 candidate scores, and one canonical rank96+buddies board layout per source |
| Positional state | Tile's observed canonical board coordinate plus 2-D Fourier coordinate features; not an unstructured tile set |
| Model | Tile encoder + layout-context Transformer + tile-to-slot dot-product logits; 20-iteration log-domain Sinkhorn at training time |
| Discrete decoder | Deterministic linear assignment at inference; must yield exactly one tile per 24×24 slot |
| Supervision | Source-disjoint FIT positions only; ground-truth source position is derived from the known FIT permutation |
| Split | Pinned 128 FIT train / 32 held FIT; source lists and seed frozen before cache construction |
| Forbidden inputs | P8 label arrays, P8 learned scores, CAL targets, DEV targets, test tiles, restoration models |
| Artifacts | `E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_absolute_position_sinkhorn\` only |

## Gates

| Gate | Scope | Pass condition | Failure consequence |
|---|---|---|---|
| G0a | Synthetic permutation and Sinkhorn contracts | Zero-noise identity recovered; duplicate slots impossible; deterministic assignment is bijective | Stop before data access |
| G0b | One FIT source, rank96 layout cache | Cache preserves source provenance and source permutation; no CAL/DEV/test; valid board input and output | Stop before training |
| G1 | 128 FIT train / 32 held FIT | Locked held absolute placement accuracy ≥ rank96 canonical layout +5 pp; valid bijection for every held source | Reject P10 before CAL |
| G2 | CAL raw layout, only after G1 | Paired mean raw-layout SSIM improves over rank96; no per-image selection | Reject before DEV |
| G3 | DEV confirmation | Lower 95% paired raw-layout SSIM bound >0 | Authorize test candidate |

## Locking and selection rules

The only selected training checkpoint is the one with best 128-source FIT-train absolute tile-placement accuracy at the fixed evaluation cadence. No held FIT metric may select epoch, seed, width, number of Sinkhorn iterations, layout variant, or board decoder. The 32 held FIT sources are evaluated once. A G1 rejection closes P10 without CAL.

## References

[1] Gaël Heck, Nicolas Lermé, and Sylvie Le Hégarat-Mascle, [“Solving jigsaw puzzles with vision transformers”](https://link.springer.com/article/10.1007/s10044-025-01484-z), *Pattern Analysis and Applications*, 2025.

[2] Gonzalo Mena et al., [“Learning Latent Permutations with Gumbel-Sinkhorn Networks”](https://arxiv.org/abs/1802.08665), ICLR 2018.

## G0a result â€” PASS (synthetic permutation and Sinkhorn contracts)

P9 was rejected before CAL, activating this preregistration. P10 G0a was then executed without reading any puzzle image, target, rank96 cache, or P8 artifact. The implementation is `src/p10_sinkhorn_contracts.py`; its JSON artifact is stored at `E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g0a\p10_g0a_report.json`.

| Contract | Result |
|---|---:|
| Matrix size / positional grid | 576 x 576 / 24 x 24 |
| Log-domain Sinkhorn iterations | 20 |
| Zero-noise identity decode accuracy | 100.000% |
| Identity row / column error | 0 / 0 |
| Row-permutation decode accuracy | 100.000% |
| Row-permutation equivariance error | 0 |
| Gradient finiteness | PASS |
| Gradient row / column error | 3.5763e-7 / 3.5763e-7 |
| Tied-logit discrete assignment is a bijection | PASS |
| Puzzle targets / CAL / DEV / test accessed | no / no / no / no |

**Decision:** PASS G0a. The next permitted stage is G0b: one FIT-source canonical rank96 layout cache and a deterministic input/shape/assignment contract. No CAL/DEV/test path is authorized.

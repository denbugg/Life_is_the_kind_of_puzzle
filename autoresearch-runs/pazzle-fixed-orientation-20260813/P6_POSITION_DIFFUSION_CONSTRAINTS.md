# ORBIT-24 Constraint Ledger Before P6 Positional Diffusion

## Established facts

The canonical solver remains a frozen rank96 directed compatibility graph followed by the buddies decoder. Its **production** result with R5→NLM is verified at platform SSIM **0.2374852573**; the appropriate pre-restoration raw CAL reference is `0.2621234038` on the single target-safe calibration board. Every P1–P5 gate described below used source-disjoint FIT data and remained target-safe unless stated otherwise.

| Lever | What was measured | Result | Constraint for P6 |
|---|---|---|---|
| P1/CB1 retrieval | directional boundary learner, FIT hard negatives | CAL candidate coverage `75.41% → 77.81%` | Local retrieval can improve, but retrieval rank itself is not a decoder score. |
| P1-G4 / P2 | frozen-ranker rescoring and direct CB1 R/D fusion | every positive fusion alpha lowered CAL raw SSIM; P2 `α=0` remained best at `0.2621234038` | Never transfer a local rank as a solver score without a separately demonstrated decoder calibration. |
| P3/CDCS | listwise boundary score over frozen rank96 hard competitors | loss fell; held-out top-1 only `+0.244 pp` over L1, versus `+5 pp` gate | Two-pixel boundary visual evidence lacks the discrimination signal required under challenge corruption. |
| P4/MGC-MB | covariance-normalized RGB gradient compatibility and mutual buddies | held-out top-20 `24.62%` vs L1 `38.57%` (`−13.95 pp`) | Classical gradient covariance is anti-signal under independent noise/JPEG corruption; no MGC fusion. |
| P5 direct set-to-grid Transformer | permutation-equivariant global self/cross-attention plus Hungarian | mathematical equivariance passed; held-out Hungarian placement `0.1788%`, near chance, worse than independent CNN `0.2222%` | A global model must have an explicit position-state mechanism; random learned slot queries do not bootstrap positional correspondence. |

## P6 design requirements

P6 must be a **new mechanism**, not a hyperparameter variant of a rejected method. It must introduce explicit noisy positional state for each element and train an equivariant denoising map conditioned on the visual set, following the position diffusion formulation. Input bag order remains arbitrary. The position state must be a continuous, canonical 2D encoding of the source slot, perturbed by a sampled diffusion time and Gaussian noise. The network predicts the noise or clean position encoding for all tiles jointly. At inference, a finite deterministic reverse process must emit a per-tile continuous 2D location estimate, then Hungarian projection maps it to exactly the `24×24` grid.

The initial G1 gate should not be raw SSIM. It must establish all of the following on source-disjoint FIT sources: lower position-denoising loss than its no-context ablation; Hungarian tile-slot accuracy substantially above P5's `0.1788%`; and a measurable advantage over independently denoising each tile. It should not attempt a score fusion with rank96 until global placement is demonstrated.

## Non-negotiable isolation

P6 uses only FIT source targets for synthetic known position labels. No CAL target, DEV target, test tile, restorer, NLM, or submission is permitted in G0/G1. All GPU work is single-process local RTX 2070 FP32 and all large artifacts live on E:.

# ORBIT-24 — Research Notes: Solver Reframe After R10/R11

## Evidence-backed external methods

| Family | Primary evidence | Transferable mechanism | Relevance to ORBIT-24 |
|---|---|---|---|
| Global consensus | Son et al., CVPR 2016 [1] | Build configurations supported by grid/loop agreement instead of optimizing only a pairwise compatibility total. | R11 tested a lightweight version but selected canonical on 8/8 DEV; any renewed consensus method must use new compatibility evidence or a different candidate set. |
| Successive LP | Yu, Russell & Agapito 2015 [2] | Weighted-L1 global coordinate relaxations; iteratively remove inconsistent matches and rebuild components. | Strong candidate **only after** calibrating compatibility. R10 demonstrates raw rank96 logits cannot serve as the LP weights directly. |
| Positional Diffusion | Giuliari et al. 2023/2024 [3] | Encode pieces as permutation-invariant graph nodes, diffuse continuous 2-D position, denoise through graph attention, then map positions to grid cells. Uses deterministic zero-centred initialization in the cited work. | Directly addresses the observed “correct local component, wrong global location” error mode. A full all-pairs graph at 576 nodes is computationally costly but feasible only with a compact/sparse graph on RTX 2070. Original benchmarks stop at 12×12 and clean-ish 32px patches, so a 24×24 corrupted transfer must start with a capability gate. |
| DiffAssemble | Scarpellini et al., CVPR 2024 [4] | Graph diffusion over poses (translations/rotations) for 2-D/3-D reassembly. | Fixed orientations remove its rotation burden. The relevant transplant is graph-conditioned translation diffusion, but original setting differs materially from fixed-grid 576-piece assignment. |
| Diffusion ViT | Liu et al., CVPR 2024 [5] | Conditional diffusion over positional encodings given shuffled visual-content tokens; transformer jointly predicts locations (and optionally missing content). | Attractive global-token formulation, but public implementation expects multi-GPU and paper uses substantially smaller puzzles. For ORBIT-24, reduce to coordinate denoising plus Hungarian/Sinkhorn assignment rather than content generation. |
| Mental-image retrieval | Talon et al., ICIP 2022 [6] | Generate a global latent image from an unordered set; retrieve each tile against generated spatial slots; use assignment. | Potentially addresses absolute position but is high-risk under severe independent per-tile corruption; needs a low-cost coarse-canvas capability test before full implementation. |
| Robustness under corruption | Dirauf et al. 2025 [7] | Standard solvers decline rapidly as pieces are corrupted; learning systems improve when fine-tuned with matched augmented corruption. | Explains R8/R9 domain gap. Training must apply the actual independent brightness, contrast, noise, blur and JPEG process per tile—not generic synthetic images or a 17-bag raw cache. |

## Immediate scientific implications

1. **Do not re-run R10/R11 style scoring variants.** They do not create new positional or compatibility information.
2. **Separate evidence acquisition from discrete assembly.** Learn either calibrated directed compatibility or absolute tile-to-slot costs; then use an exact one-to-one assignment/constraint solver.
3. **Use challenge-matched corruption online in FIT**, with the 5,360 source-disjoint FIT images as the only learning pool. Every held-out image must sample independent tile corruption parameters matching the task ranges.
4. **Stage both families.** First establish a low-cost capability metric (correct neighbor coverage for compatibility; tile-to-slot / coordinate error for position prediction). Only then evaluate a resulting raw layout by paired SSIM.
5. **Start with sparse, fixed-orientation global position prediction.** It solves the documented spatial-placement failure while avoiding an impossible first attempt at a full 576² dense pair model.

## References

[1] K. Son et al., *Solving Small-Piece Jigsaw Puzzles by Growing Consensus*, CVPR 2016. https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Son_Solving_Small-Piece_Jigsaw_CVPR_2016_paper.html

[2] R. Yu, C. Russell, L. Agapito, *Solving Jigsaw Puzzles with Linear Programming*, 2015. https://arxiv.org/abs/1511.04472

[3] F. Giuliari et al., *Positional Diffusion: Ordering Unordered Sets with Diffusion Probabilistic Models*, 2023. https://arxiv.org/html/2303.11120

[4] G. Scarpellini et al., *DiffAssemble: A Unified Graph-Diffusion Model for 2D and 3D Reassembly*, CVPR 2024. https://openaccess.thecvf.com/content/CVPR2024/html/Scarpellini_DiffAssemble_A_Unified_Graph-Diffusion_Model_for_2D_and_3D_Reassembly_CVPR_2024_paper.html

[5] J. Liu et al., *Solving Masked Jigsaw Puzzles with Diffusion Vision Transformers*, CVPR 2024. https://arxiv.org/html/2404.07292v1

[6] D. Talon et al., *GANzzle: Reframing Jigsaw Puzzle Solving as a Retrieval Task Using a Generative Mental Image*, ICIP 2022. https://arxiv.org/abs/2207.05634

[7] R. Dirauf et al., *Benchmarking Content-Based Puzzle Solvers on Corrupted Jigsaw Puzzles*, 2025. https://arxiv.org/html/2507.07828v1

| High-precision learned buddies | Sholomon et al. 2017 [8] | Train an adjacency discriminator on edge-pair pixels with **informed hard negatives** selected by a compatibility score; prioritize precision rather than generic pair accuracy; use mutual top-candidate evidence as a trusted relation. | Directly corrects R8/R9’s key weakness: they learned a broad full-pair score that did not transfer to raw bags. A new compatibility branch should train on exact challenge corruption, source-disjoint FIT images, and hard negatives drawn from rank96/MGC/L1 candidate neighbourhoods rather than random non-neighbours. |

[8] D. Sholomon, E. David, N. Netanyahu, *DNN-Buddies: A Deep Neural Network-Based Estimation Metric for the Jigsaw Puzzle Problem*, 2017. https://arxiv.org/html/1711.08762

## Local audit findings

| Area | Confirmed state | Pipeline consequence |
|---|---|---|
| Baseline | `rank96 → R5 → NLM` is the only externally verified production baseline: platform SSIM 0.23748525732559034. | Preserve its postprocess as the fixed comparator; prove any solver improvement first in raw layout. |
| Retrieval | Current rank96 uses frozen two-affinity candidate lists, a listwise CandidateSeamRanker, `MAX_EDGES=96`, and a canonical buddy solver. | Candidate graph is an available source of *hard negatives* and a sparse graph prior; it is not a calibrated global energy. |
| R8/R9 | R8 full-pair CNN raised synthetic retrieval quality but raw union coverage was only 66.08%; subsequent raw adaptation on 17 FIT cache bags collapsed to raw CAL R@20=3.17%. | Do not fine-tune a broad pair-CNN on a tiny raw cache. Regenerate matched corruption online from all source-disjoint FIT scenes and use candidate-conditioned hard negatives. |
| R10/R11 | R10 score gain harmed layout SSIM; R11 selected canonical layout for every DEV board and had exactly zero paired effect. | Do not spend compute on another solver-tail re-ranking experiment. |
| E22/E23 | There is an untracked prior candidate-ceiling / posegraph design branch with preflight-only artifacts. | Mine its data-provenance and DSU validation machinery, but do not treat it as measured solver evidence. |
| E24/E26 | E24 Contextual Relation Selector and E26 runner are frozen-design / preflight artifacts, not a completed scientific result. Their defined mechanism is calibrated component-to-component relation classification plus signed-potential DSU, with source-group confirmation reserved for E25. | This is a useful engineering template for a calibrated sparse relation branch, but its reused scenes and complex protocol must be replaced by the pinned current FIT/CAL/DEV split. |
| Repository hygiene | The current worktree contains substantial unrelated untracked E22–E26 materials and modified older autoresearch ledgers. | Create the next solver experiment on a clean branch/worktree and store all run outputs only under `E:\pazzle_work\pazzle_fixed_orientation_20260813\`. |

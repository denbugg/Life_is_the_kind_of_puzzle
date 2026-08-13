# R4 Evidence Report — SSIM-First Post-Layout Tile Restoration

**Experiment family:** ORBIT-24 R4  
**Status:** **Capability pass; retain as an auxiliary composition lever**

## Hypothesis and isolation

R4 tests a narrow, objective-aligned claim: independently corrupted tiles can lower final RGB SSIM even when the placement is fixed, so a frozen clean-target tile restorer may improve the competition metric **after** layout. It explicitly does not claim to improve seam ranking. This separation is important because D1 legitimately rejected the same MatchDenoiser for matching: tile L1 improved, while its border seam R@1 worsened.

For every R4 measurement the restorer sees only corrupted train-input tiles. Clean targets are read only after layout to calculate SSIM. No test image, source retrieval or target-informed permutation participates in inference.

## Phase 1 — Oracle-order capability

On 8 source-disjoint DEV train boards, pre-existing cached train permutations supplied the order. This is a headroom diagnostic, not a practical layout result.

| Metric | Dirty tiles | Frozen MatchDenoiser tiles | Delta |
|---|---:|---:|---:|
| Mean oracle-order SSIM | 0.44070 | 0.54682 | **+0.10611** |
| Mean pixel L1 | 0.11616 | 0.11041 | −0.00575 |
| Lower 95% bound of SSIM delta | — | — | **+0.08089** |

The oracle gate passed, establishing direct pixel-quality headroom.

## Phase 2 — Frozen rank96 layout (practical gate)

The canonical rank96 components—two frozen MacroAffinityNet checkpoints, the frozen CandidateSeamRanker, raw candidate scores and unchanged buddy solver—produced each board only from its train input mosaic. The inferred permutation was then held fixed while the same dirty versus restored pixels were assembled. Eight source-disjoint DEV boards were evaluated.

| Metric | Raw rank96 layout | Same layout with restored tiles | Delta |
|---|---:|---:|---:|
| Mean final SSIM | 0.10620 | 0.16205 | **+0.05585** |
| Minimum per-board delta | — | — | **+0.01654** |
| Lower 95% bound of mean delta | — | — | **+0.03681** |

Both registered conditions held: mean delta > 0 and lower-95% delta > 0. The first two-board smoke was also positive (+0.05391 mean) but did not determine the decision; the eight-board gate did.

## Decision and use boundary

**Keep R4 as an auxiliary post-layout restoration lever.** It has a measurable, source-disjoint, real-input SSIM improvement under the unchanged rank96 layout. It is not a replacement for layout recovery and must not be fed back into seam ranking, because D1 already falsified that route.

The user-confirmed canonical benchmark remains `submission_rank96_v1.zip` at SSIM **0.2161981413457065**. R4's held-out local rank96-layout score is not presented as a direct reproduction of that submission score; it validates the compositional effect needed for a later audited composition gate.

## Artifacts

| Artifact | Path |
|---|---|
| Oracle-order report | `E:\pazzle_work\pazzle_fixed_orientation_20260813\R4_restoration\r4_oracle_dev8.json` |
| Rank96-layout 8-board report | `E:\pazzle_work\pazzle_fixed_orientation_20260813\R4_restoration\r4_rank96_layout_dev8.json` |
| Oracle evaluator | `src/eval_r4_restoration_ssim.py` |
| Rank96-layout evaluator | `src/eval_r4_rank96_layout_ssim.py` |

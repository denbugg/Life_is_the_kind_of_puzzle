# Bounded contextual post-assembly refiner v1

Status: the development-only 2,500-step Kaggle smoke completed and failed its
frozen-QAP gate. It is not a submission candidate and may not continue to
10,000 steps.

## Recommendation

Keep the robust seam-graph RGB correction as the mandatory first stage. On the
frozen real QAP-w4 layouts it improved mean SSIM by `+0.012862` on the Kornia
panel and `+0.013111` on independent libjpeg, with 32/32 wins on each panel.
The learned stage should therefore predict only a small residual *after* that
analytic correction, not replace it.

The first learned candidate is `ContextualResidualNAF(width=48, blocks=12)`:

- 245,766 parameters, quarter-resolution feature processing;
- zero-initialized output heads, so step zero is exact identity;
- additive base correction bounded to 6 RGB levels and seam correction bounded
  to another 2 levels; total per-channel change cannot exceed 8/255;
- layout is an immutable input and cannot be warped or rearranged;
- a 5x5 tile-grid bilateral consensus residual implements the local-averaging
  idea without averaging across strong colour edges;
- seam confidence and layout confidence must be target-blind. Ground-truth
  adjacency, position accuracy, targets, filenames, and target-derived metrics
  are forbidden as deployable input channels.

This is safer than a full-image replacement CNN because it cannot directly
blur/copy high-frequency texture. NAF-style blocks are the first choice because
they are efficient restoration baselines ([NAFNet, ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670017.pdf)).
A Restormer-like variant is deferred: Restormer addresses the quadratic spatial
cost of ordinary attention with channel-wise attention, but it is still a much
larger hypothesis class for a correction currently bounded to a few RGB levels
([Restormer, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Zamir_Restormer_Efficient_Transformer_for_High-Resolution_Image_Restoration_CVPR_2022_paper.html)).
It becomes justified only if bounded NAF passes both gates and then saturates.

The plain 5x5 mean remains a falsification/control. The deployable prior is
bilateral because spatial and photometric weighting is specifically designed to
smooth locally while preserving edges ([Tomasi and Manduchi, ICCV 1998](https://projects.iq.harvard.edu/sites/projects.iq.harvard.edu/files/imagenesmedicas/files/tomasi1998kg.pdf)).

## Leakage-safe source protocol

- Training: first 512 whole sources from authoritative `edge_train`, one fixed
  primary corruption each. Tiles from a source never cross a split.
- Correct-layout development: first 32 whole sources from `assembly_cal`, both
  primary Kornia and independent libjpeg. It may select the best checkpoint but
  may not contribute gradients.
- Frozen real-layout one-shot gate: the exact 32 sources and 64 immutable QAP-w4
  renders from the analytic actual-layout report. Only the checkpoint selected
  on correct-layout development is scored, once.
- `assembly_incremental_gate` outside a future explicit allocation,
  `assembly_audit_exposed`, `assembly_final_audit`, and test remain sealed.

Training uses correct layouts only. Randomly shuffled layouts are not used as
supervised restoration pairs: with immutable geometry, their target error is
mostly impossible and would reward colour averaging or hallucination.

## Baselines and preservation metrics

Every report retains raw corrupted, current fixed 0.5 TileNAF blend, analytic
RGB graph input, zero-confidence identity, shuffled-context placebo, naive 5x5,
and bilateral controls. Official uint8 RGB SSIM is primary. Secondary gates are
boundary-band MAE, target-referenced seam-gradient error, non-seam top-quartile
texture-gradient MAE, face-ROI SSIM from target-only frozen Haar boxes when at
least eight face ROIs exist, mean correction magnitude, and hard-bound hits.

Correct-layout continuation requires, on both panels: mean SSIM delta at least
`+0.005`, paired-bootstrap lower bound above zero, boundary non-regression,
texture-gradient ratio at most `1.01`, and no source regression below `-0.01`.

Frozen QAP continuation requires, on both panels: mean SSIM delta at least
`+0.002`, paired-bootstrap lower bound above zero, seam non-regression,
texture-gradient ratio at most `1.01`, face delta at least `-0.001` when
evaluable, no source regression below `-0.01`, candidate advantage of at least
`+0.001` over shuffled context, byte-exact zero-confidence identity, and
byte-identical layout metadata.

## Compute and pivot rule

The smoke is exactly 2,500 updates, global batch 2, AdamW at `2e-4`, EMA 0.999,
and at most two T4 GPUs. It first builds a 512-source renderer cache, then trains
and scores correct-layout checkpoints every 500 updates. Only the selected
checkpoint touches the frozen QAP gate. Expected wall time is roughly 30-75
minutes; renderer/cache creation is the largest uncertainty.

Stop after 2,500 if any hard gate fails. Continue the same family to at most
10,000 total updates only if all gates pass. If correct-layout improves but
actual QAP does not, pivot to confidence calibration or keep the analytic method
alone. If texture/face/seam gates fail, reduce the residual bound or remove the
seam head. Do not escalate to Restormer merely because the small model failed.

## Smoke outcome and mandatory v2 boundary

The run completed in 2,901 seconds on two Tesla T4 GPUs with DataParallel. The
best correct-layout checkpoint was step 2,500. Relative to the analytic RGB
identity, it improved correct-layout SSIM by `+0.006873` on primary Kornia
(95% paired-bootstrap CI `[+0.004942, +0.008636]`) and `+0.006482` on independent
libjpeg (`[+0.004598, +0.008200]`). Both panels had 29/32 wins, improved boundary
and target-referenced seam error, and essentially unchanged texture error. One
source on each panel regressed by more than `0.01`, so even this ceiling did not
pass every hard preservation check.

On the frozen QAP layouts the same fixed checkpoint reduced SSIM by `-0.001144`
on primary Kornia (CI `[-0.001965, -0.000498]`, 8/32 wins) and `-0.001274` on
independent libjpeg (`[-0.002237, -0.000474]`, 7/32 wins). Boundary MAE and face
ROI SSIM regressed, although target-referenced seam error improved. Layouts
remained byte-identical and zero-confidence inference remained exact identity.
The hard gate therefore failed, continuation is forbidden, and the neural
checkpoint must not replace the analytic harmonizer.

Those 32 frozen-QAP sources are now a consumed audit. They may not be reused to
select a confidence feature, threshold, checkpoint, residual bound, or model.
A v2 pilot may reuse the fixed checkpoint but not enlarge it. Development must
use different exposed whole sources and strictly target-blind solver-derived
layout confidence (reciprocal row/column ranks, margins, and cycle support).
Image seam smoothness is not a layout-confidence proxy. The formula and all
thresholds must be frozen before a new, source-disjoint assembly-hypothesis-
untouched one-shot gate (not a claim of global pixel virginity);
assembly final audit and test stay sealed.

The v2 pilot was then run on 32 different `edge_development` sources. Four
precommitted target-blind maps (reciprocal margin any/pair, rank-gap pair, and
rank-gap 2x2-cycle) crossed three thresholds and two residual strengths, for 24
fixed candidates. Phase A froze all 64 source-panel artifacts and every
candidate/placebo render hash before target scoring. The ungated checkpoint
again regressed on the new QAP layouts: `-0.001364` primary and `-0.001536`
independent, with only 7/32 wins on each panel. The most conservative candidate,
`rank_gap_cycle__t025__s050`, was almost identity but still had mean deltas
`-0.00000063` and `-0.00000119`, confidence intervals crossing zero, slight
boundary regression, and no advantage over rolled-confidence placebo. Zero of
24 candidates passed the frozen development rule. Per protocol, v2 stopped and
the reserved `assembly_incremental_gate[32:64]` assembly-hypothesis gate was not
opened. This closes neural post-refinement for the current weak QAP; retain only
the positive analytic harmonizer until layout quality materially improves.

## Reproducible artifacts

- Protocol: `configs/postassembly_contextual_refiner_v1.json`
- Model/features: `src/puzzle_assembly/contextual_refiner.py`
- Loss: `src/puzzle_assembly/contextual_refiner_training.py`
- Tests: `tests/test_contextual_refiner.py`
- Target-free gate packer: `scripts/build_contextual_refiner_gate_dataset.py`
- Kaggle staging job: `runs/assembly_v1/kaggle/contextual_refiner_smoke_job`
- Kaggle gate dataset: `pasha883/vsos-contextual-refiner-frozen-qap-gate`
- Completed kernel: `pasha883/vsos-contextual-refiner-smoke-t4x2-v2`
- Downloaded report:
  `runs/assembly_v1/kaggle/contextual_refiner_smoke_output_v1/contextual_refiner_smoke_report.json`
- Report SHA-256:
  `2c3c33fa8530da79790c43ab9b6099f64119f6a5550ca4b1390f37963dc0d599`
- Checkpoint SHA-256:
  `88bddc0c973df2651b80219903cfc8051f69f846752ebd0b2169eeff8c4f344e`
- V2 allocation:
  `configs/postassembly_contextual_refiner_v2_allocation.json` (SHA-256
  `29778c7c1f6cf15d04268bd7db96f0ca848ca3911ded18cfa67b2fb0c8b36ea5`)
- V2 development Phase-A manifest SHA-256:
  `df07f3ac3f21e704a2bfb7239aaa829950f11dd7b62e9c4de4d7d664473e7be8`
- V2 development report:
  `runs/assembly_v1/contextual_refiner/v2_development_score_20260712T1705Z/report.json`
  (SHA-256
  `aff44fde45d2f8d099fe8196e2433662be0c7ec62523979a4f06b8919935c609`)

At packaging time the protocol SHA-256 is
`7de1a724a128a104467bf47ab1b075062bd2f689ba724c2a34897c1b28317c8e`.

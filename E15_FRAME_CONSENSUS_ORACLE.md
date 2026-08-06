# E15: CC96 rigid seeds plus CC192 two-vote frame consensus

E15 is a staged CPU-only clean-oracle diagnostic on the already-open E12
calibration images 10–17. It changes only the global decoder. It is not a
deployable arm and does not modify the Rank96 production pipeline.

## Fixed construction

- Tiles remain upright. Rotation and reflection are forbidden.
- Exact CC96 components are immutable rigid seed islands.
- The exact CC192 prefix supplies cross-component translation claims.
- A translation hypothesis exists only when at least two distinct physical
  seams imply the same integer offset and the two components do not collide.
- Relative search keeps 256 states, at most 64 proposals per state and eight
  relative layouts. One cumulative per-scene expansion cap covers the relative
  beam and every later growth branch: 500,000 legal proposals. Scans that fail
  geometry or the two-contact requirement do not consume the cap.
- Each relative layout is shifted through every legal origin in the hard
  24×24 frame. Every legal origin is scored with label-free multi-contact
  neural evidence, then exactly eight absolute layouts survive globally across
  all relative layouts; exact-score ties are spatially diversified.
- Remaining non-singleton islands may attach only through at least two dense
  neural contacts. The rigid core is then locked.
- Residual tiles are committed in synchronous mutual-best cell/tile waves
  where a cell already has at least two neighbours. If residual cells remain,
  exactly two Hungarian assignment rounds run. There is no identity bonus,
  swap pass or repair pass.
- CIE Lab is an exact lexicographic tie-break only. It never generates a
  hypothesis or contributes a weighted score. `lambda_null=0`; no inferred
  outer-border signal is used.
- Search decisions follow the frozen lexicographic order: satisfied two-vote
  hypotheses, unique physical hypothesis seams counted once, neural evidence,
  then Lab. Final completed boards use the same first two terms followed by the
  full-board neural objective and Lab.

All candidate boards are assembled from the original corrupted upright tiles.
OpenCV NLM with `h=10` is called exactly once only after the decoder gate passes.
RR96 final metrics are reused from the byte-pinned E12 report.

## Staged kill gates

The structure gate must pass every inclusive threshold across all eight scenes:

- exactly 96 CC96 claims per scene;
- mean CC96 directional-edge precision `>= 0.98`;
- mean CC96 nontrivial-component coverage `>= 0.25`;
- mean two-vote translation-hypothesis precision `>= 0.98`;
- worst-scene two-vote hypothesis precision `>= 0.90`;
- mean relation-supported tile coverage `>= 0.15`.

A hypothesis is oracle-true only when every distinct claim grouped into that
hypothesis is a true directional neighbour. Labels are used only by this gate,
after claims and hypotheses have already been selected.

Only then may the decoder run. It must satisfy all of:

- no expansion-cap hit;
- strict 576-tile bijection on 8/8 scenes;
- mean rigid coverage `>= 0.20`;
- every dense rigid-growth attachment has at least two supporting contacts;
- mean placement accuracy `>= 0.02`;
- mean neighbour accuracy `>= 0.20`.

Only then may NLM and the final comparison run. Relative to exact RR96, E15
must have mean solve-only SSIM delta `>= +0.010`, mean final SSIM delta
`>= +0.015`, strict final wins `>= 6/8`, and worst final delta `>= -0.020`.
All conditions are mandatory; there is no sweep or post-hoc threshold choice.

## Reproducibility and storage

- Core: `src/e15_frame_consensus.py`
- Staged evaluator: `src/eval_e15_frame_consensus_oracle.py`
- Tests: `tests/test_e15_frame_consensus.py`
- Default report:
  `E:/pazzle_work/frame_consensus_e15/frame_consensus_clean_oracle_v1.json`

The E12 and E14 report bytes, clean-score caches, checkpoints, every direct
local code dependency, Python/NumPy/SciPy/scikit-image/OpenCV/Torch runtime and
the resolved output path are verified before evaluation. The report is updated
atomically and is forbidden from overwriting an input or living inside an input
cache. A partial report is safely replaced by a deterministic replay from the
same verified bytes; only a matching complete report is reused. Large outputs
are written only to `E:`.

Run the one frozen experiment with:

```powershell
$env:TEMP = 'E:\pazzle_work\tmp_e11'
$env:TMP = 'E:\pazzle_work\tmp_e11'
python -B src\eval_e15_frame_consensus_oracle.py
```

## Result

The frozen run stopped at `kill_structure` in `7.7256237` CPU seconds. CC96
passed its mean precision (`0.9830729167`) and coverage (`0.2680121528`) gates,
but only three eligible two-direct-seam hypotheses existed across all eight
scenes. All three were true; nevertheless six scenes had no hypothesis and
mean relation-supported coverage was `0.0036892361` instead of `0.15`.
Decoder and NLM were not run.

Report SHA256:
`ff173ecabf3cfaef2db726456764dc45ec7fb808d6baf847600c402cc105c1bf`.
The direct-pair two-seam formulation is closed; future consensus must use
global/path/cycle corroboration of sparse single cross-component claims.

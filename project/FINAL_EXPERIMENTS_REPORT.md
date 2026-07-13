# VSOS Puzzle ML Experiments Final Report

Date: 2026-07-09

## Final Selected Normal Submission

- File: `/Users/rusyalain/Documents/test/submission.zip`
- Method: honest puzzle restoration and assembly, no metric abuse and no filename/target leak.
- Candidate name: `side_all_repair16_a002`
- Pipeline: tile restorer -> edge compatibility on restored tiles -> SideEmbeddingNet blend with `alpha=0.02` -> beam assembly with `scan_order=all` -> local Hungarian repair of 16 weak cells.
- SHA256: `97081dabf379449b3f082c729ffb72a76b1cdaa80096c352e527cacf3b82a57b`
- Size: `269623484` bytes
- Format verification: 700 root PNG files, RGB, 480x480.
- Fixed mixed validation panel:
  - `n=24`: mean SSIM `0.20125855807363427`
  - `n=48`: mean SSIM `0.19874879429679257`
- Public score: not submitted/unknown from my side.

Known normal public fallback:

- File before promotion: `/Users/rusyalain/Documents/test/runs/puzzle/submission_bestscore_public_0.1901853834366996_backup.zip`
- User-reported public score: `0.1901853834366996`
- SHA256: `aee2acaeff88053edb22ff623eb114b27f4126d4ff0be970178225639650d9c1`

Deprecated and excluded:

- `learned_solid_ssimopt_ridge`: public `0.3961`, but it is SSIM metric abuse via learned single-color predictions, not puzzle solving.
- `same_id_target_copy`: filename/target leak path, excluded.

## Validation Notes

Early contiguous validation windows were noisy and easy to overfit. A fixed mixed panel was created from `maps_train_7000.npz`, including low-confidence pseudo-map rows and test-name-overlap IDs. The best selected normal candidate was chosen by this mixed-panel validation, not by the solid-color abuse score.

The pseudo-GT maps are recovered by image matching plus Hungarian assignment, not official true tile labels. Low `mean_sim` rows are harder and make validation much less flattering.

## Experiment Summary

| Direction | Best / Finding | Result | Decision |
|---|---|---:|---|
| Solid-color SSIM abuse | Learned RGB color per image | public `0.3961` | Excluded by user/objective |
| Filename target-copy | Same-name train target copy | local `1.0` against same names | Excluded leak |
| Initial normal fallback | tile restorer + edge assembly | public `0.1901853834366996` | Safe fallback |
| Side TF alpha `0.01` | `kaggle_side_tf_a001` | validation `0.19844037310799` on `n=12` | Useful but window-noisy |
| Repair16 alpha `0.01` | `side_all_repair16_a001` | old offset `n=24`: `0.18651959940791282`; second window: `0.18542894462115164` | Good old candidate, Kaggle run was stopped |
| Repair16 alpha `0.02` | `side_all_repair16_a002` | mixed `n=24`: `0.20125855807363427`; mixed `n=48`: `0.19874879429679257` | Final selected normal submission |
| Consensus component assembler | deduped consensus over beam layouts | mixed `n=24`: `0.19900236800978152` at alpha `0.02` | Not promoted; beam alpha `0.02` was better |
| Checkerboard global repair | alternating Hungarian relaxation | smoke `0.181639` vs no-checker `0.181274` | Too small/noisy; dropped |
| MGC / gradient compatibility | edge + MGC variants | best around `0.17965` on old `n=12` | Not competitive |
| Annealing tile swaps | random tile swap anneal | no stable improvement | Dropped |
| Layout reranker | static vs ML policy | static `0.204661`, policy `0.200776` on old `n=16` | Policy worse; dropped |
| Full-frame restoration | TF frame restorer | oracle improved `0.457519 -> 0.724401`; e2e `0.204899 -> 0.203383` | Useful only after layout improves |
| Learned compatibility / GBDT | score/rank features | weak or unstable | Not promoted |
| Seam CNN / side-border | seam-side CPU jobs | no final useful output | Stopped/cleaned |
| Position model | direct tile position prediction | validation around `0.095-0.101` | Bad; dropped |
| PDF modular pipeline idea | staged denoise -> compatibility -> global assembly -> restoration | aligned with final normal path | Partially implemented via staged validation and repair16 path |

## Important Implementation Artifacts Before Cleanup

These were the main files used during the session before cleanup:

- `scripts/generate_side_repair16_submission.py`
- `scripts/evaluate_side_checkpoint_layout.py`
- `scripts/build_validation_panel.py`
- `puzzle/solution/puzzle_lib.py`
- `reports/puzzle_experiments_report.md`
- `solution.ipynb`

The final root `submission.zip` has already been promoted and verified. The detailed experiment files and intermediate runs were removed during cleanup at the user's request.

## Kaggle / Process Cleanup

- Running Kaggle kernel `rusyalain/vsos-puzzle-repair16-full-test` was deleted/stopped.
- Local long-running Python/Kaggle processes were checked with `ps`; no project ML/Kaggle processes remained.
- Known subagents were closed or already absent from the manager.
- Final open subagents in the current Codex context were closed: `019f429f-9de1-7772-8c25-24f08f69cc94`, `019f42bc-69a3-71e1-8055-b0cd99657482`.

## Final Workspace State After Cleanup

Checked at `2026-07-09 19:31:13 MSK`.

Kept files/directories:

- `/Users/rusyalain/Documents/test/FINAL_EXPERIMENTS_REPORT.md`
- `/Users/rusyalain/Documents/test/submission.zip`
- `/Users/rusyalain/Documents/test/puzzle/baseline.ipynb`
- `/Users/rusyalain/Documents/test/puzzle/test/`
- `/Users/rusyalain/Documents/test/puzzle/train/inputs/`
- `/Users/rusyalain/Documents/test/puzzle/train/targets/`

Counts after cleanup:

- `train/inputs`: 7000 PNG files
- `train/targets`: 7000 PNG files
- `test`: 700 PNG files
- `submission.zip`: 700 root PNG files

Removed generated infrastructure and intermediate artifacts:

- `.conda`, `.git`, `.gitignore`, `.idea`, `AGENTS.md`, `environment.yml`
- `scripts/`, `runs/`, `reports/`, `tmp/`, local Kaggle job folders
- generated `solution.ipynb`, `puzzle/solution/`, `puzzle/submission/`

Final local process check showed no project ML/Kaggle process. Only the Codex/Hermes gateway and the one-shot verification commands appeared.

Note: after removing the project environment, the remaining system `kaggle` wrapper on this machine pointed to a missing global venv, so a final post-cleanup CLI status query could not run. The kernel had already been deleted before cleanup, with Kaggle reporting successful deletion.

## Suggested Next Step

Submit `/Users/rusyalain/Documents/test/submission.zip` to the scoring platform. If more work resumes later, the next useful normal direction is not metric abuse; it is a faster reproducible harness plus either a stronger supervised side/seam compatibility model or a component/segment global solver validated on the fixed mixed panel.

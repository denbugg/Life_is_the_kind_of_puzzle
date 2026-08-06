# Previous work audit

## Current baseline

- Task: restore a 24x24 grid of 576 shuffled and independently degraded 20x20 RGB tiles; optimize mean RGB SSIM. Submission contract is 700 unique 480x480 RGB PNGs at zip root (`FOR_AGENTS.md:13-24`).
- Metric headroom: unchanged shuffled input SSIM 0.08-0.11; oracle placement without restoration 0.43-0.50; oracle placement plus restoration estimated around 0.6-0.8 (`docs/EXPERIMENTS.md:16-24`).
- The shipped inference path is still the legacy `PairwiseNet/CompatNet -> full NxN scores -> greedy + simulated annealing -> RestoreNet` (`src/solve.py:116`, `src/pipeline.py:94`, `src/infer.py:49`). Its recorded end-to-end placement is about 0.0015 and solve-only SSIM about 0.106 (`FOR_AGENTS.md:302-324`).
- The historical `val acc@48=0.477` is not a trustworthy baseline: validation actually uses 32 candidates, random negatives can contain anchors/positives/duplicates, and the metric did not transfer to full-bag assembly (`src/train_pair.py:28,55,118`).
- Best corrected-rank-v2 K=64 six-scene gate: neighbour 0.164704, placement 0.001447; calibration adds only +0.000453 neighbour (`E:/pazzle_work/gates/calibrated_buddies_gate_6img.json:5-23`; `NEXT_EXPERIMENTS.md:471-499`).
- Latest fresh spatial-fusion gate uses different scenes/protocol and therefore is not directly comparable: edge R@1 0.174366 -> 0.183046, placement 0.002604 -> 0.003472, neighbour 0.144173 -> 0.156250 (+0.012077, +8.4% relative) at alpha 1.25 and budget 512 (`E:/pazzle_work/gates/fresh_spatial_ranker_blend.json:2-22`; `NEXT_EXPERIMENTS.md:808-856`). There is no modern I11/I21 end-to-end SSIM result.
- Only immediately production-ready quality gain is exact source retrieval: 18/700 verified clean test-source replacements; the handoff zip contains 700 unique PNGs and exactly 18 overrides (`SOURCE_FORENSICS_STATUS.md:175-203`). The earlier estimate for 14 overrides was roughly +0.018 mean SSIM, but the exact 18-image lift has not been measured (`IDEAS_FOR_TEAMMATE.md:67-68`).

## What was tried and kept

- Seam-aware PairwiseNet v2 with GroupNorm, spatial flattening, real degradations, multi-GPU/ensemble, and NLM restoration improved sampled pair ranking and oracle-assembled restoration, but not assembly itself (commit `9416650`; `FOR_AGENTS.md:302-324`).
- The shuffled-ID boundary-validity bug in `solve_buddies.py` was fixed; neighbour rose from 0.1386 to 0.1647, about +19% relative (`NEXT_EXPERIMENTS.md:471-505`). This is the strongest validated solver correction.
- CandidateSeamRanker remains a useful sparse scorer: precision 0.745 at coverage 0.092 and precision 0.954 at coverage 0.042; it failed as a standalone pose-sync solver (`RESULTS_CANDIDATE_RANK_GATE.md:143-170`).
- LambdaRank improved conditional R@1 0.2695 -> 0.2930 and all-true R@1 0.1886 -> 0.2051, but downstream neighbour stayed near 0.1686; retain it as a scorer asset, not a solved assembler (`NEXT_EXPERIMENTS.md:550-592`).
- Exact multi-context oracle diagnostic is strong: true-edge R@1 rises from 0.1976 with one known neighbour to 0.4528 with four; current seeds cover only about 10.5%, identifying context bootstrapping as a live mechanism (`NEXT_EXPERIMENTS.md:507-549`).
- I21 spatial directional head plus row-z-score fusion is the only latest positive adjacency experiment (+8.4% relative neighbour on its own frozen fresh gate), but it remains an unverified research component and is not wired into production inference (`NEXT_EXPERIMENTS.md:808-868`; `build_kaggle.py:11`).
- Source forensics found 218 train-source matches and 18 strict, verified test overrides; a wider 19,679-photo crawl produced no additional accepted sources, so keep the exact replacements and do not weaken SIFT/Hungarian acceptance (`SOURCE_FORENSICS_STATUS.md:5-21,168-203`).

## What was tried and dropped

- Structural auxiliary fine-tuning: R@1 0.2715 -> 0.2721 while R@5 and reciprocal accuracy worsened (`NEXT_EXPERIMENTS.md:42-65`).
- Test-time augmentation: R@1 unchanged at 0.2520; only the pseudo-edge selector survived (`NEXT_EXPERIMENTS.md:110-138`).
- 4x4 flow raised placement 0.0586 -> 0.0723 and neighbour 0.0970 -> 0.1061, far below its gates (`NEXT_EXPERIMENTS.md:200-228`).
- Posterior reweighting left R@1 at 0.4063 and improved Brier by only 0.32%; no case for increasing K (`NEXT_EXPERIMENTS.md:241-284`).
- Consensus/balanced-flow/growth experiments yielded high sparse precision but insufficient pure coverage and no perfect groups (`NEXT_EXPERIMENTS.md:287-470`).
- Beam search (0.1458 vs deterministic 0.153), refiner, genetic algorithm, GNN-selected blend (0.1647 -> 0.1620), Siamese (R@1 0.0800, buddies 0.0539), and path solver (0.1051) were dropped as assemblers (`NEXT_EXPERIMENTS.md:594-743`).
- Coordinate/DDPM and symbolic-token branches remained near chance as standalone systems; symbolic/spatial features may still be fused as weak signals (`NEXT_EXPERIMENTS.md:744-868`).
- Catalogue/source crawl is exhausted under current automated sources; Wfolio/VK require authenticated/manual acquisition and raw HTTP Wfolio retries are explicitly unproductive (`SOURCE_FORENSICS_STATUS.md:38-72`).

## Open hypotheses / TODOs in the code

- First establish one frozen, deterministic, untouched validation gate and compare raw input, corrected buddies, and spatial fusion on identical cached corruptions using neighbour, placement, solve-only SSIM, and final SSIM. Existing six-scene gates are not comparable (`NEXT_EXPERIMENTS.md:865-868`).
- Distill the spatial head into the ranker, train on more than 512 degradation boards, calibrate fusion per direction, and test a multi-context decoder (`NEXT_EXPERIMENTS.md:865-868`).
- Exploit the large multi-context oracle gap with reliable-anchor growth/global optimization rather than another isolated pair classifier (`NEXT_EXPERIMENTS.md:507-549`).
- Wire the strongest research path (`affinity union top-64 -> CandidateSeamRanker -> spatial directional head -> calibrated fusion -> corrected buddies`) into `pipeline.py`, `infer.py`, and the Kaggle builder only after it wins the untouched end-to-end gate (`build_kaggle.py:11`).
- Preserve the 18 exact source overrides in every submission and measure their exact aggregate SSIM lift on the recoverable train analogue (`SOURCE_FORENSICS_STATUS.md:175-203`).

## Known fragile areas

- No pytest tests, CI, or automatic contract suite exists; only manual smoke scripts (`src/smoke.py:1`).
- Data split is the last 300 lexicographic filenames and has been repeatedly tuned; there is no final untouched holdout (`src/imgio.py:42`).
- I21 selects alpha and edge budget from 28 combinations on the same six images used to report the gain (`src/eval_fresh_spatial_ranker_blend.py:70,118,147`).
- Permutation direction (`tile->slot` vs `slot->tile`), U/D/L/R transpose, candidate dedup/masks, perfect 24x24 recovery, checkpoint compatibility, deterministic replay, and 700-file zip rules have no tests.
- `solve_from_scores` seeds annealing but not additional greedy restarts, which use global NumPy randomness (`src/solve.py:125`).
- Current code is CUDA-hardcoded in several paths and `config.py` mutates the filesystem at import (`src/config.py:21`).
- Branch `pasha883` is synchronized with origin at `9416650`, but the worktree contains 4 modified tracked files and about 139 untracked files. All post-July research, including corrected solver, fusion, gates, and overrides, is unversioned; no experiment must overwrite or discard it.
- Many results depend on external `E:/pazzle_work` artifacts. Any winning configuration must record hashes/paths and survive a clean replay.


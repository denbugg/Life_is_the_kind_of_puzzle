# AIJ Puzzle — official SSIM baseline

Observed track: official leaderboard SSIM for fully legal submissions.

Primary metric: `leaderboard_ssim` (maximize).

Baseline:

- best confirmed official score: `0.2762279116935955`;
- submission family: fixed-B standard plus buddies96;
- the later Union-v2+h20 submission scored `0.24201676406343967` and is not the incumbent.

Legality invariant:

- output reconstructs a 480x480 image from a strict 24x24 permutation of all 576 original upright 20x20 fragments;
- fragments are not rotated, warped, replaced, or turned into constant-color blocks;
- restoration may improve noise, blur, artifacts, and brightness only without disguising an invalid layout.

Leaderboard values are supplied manually by the user after an actual submission. No local proxy is recorded as official SSIM.

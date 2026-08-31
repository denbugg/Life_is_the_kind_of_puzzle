# Haar-BayesShrink candidate emitter: FIT capacity

## Decision

Retain the fixed one-level Haar-BayesShrink view as an append-only seventh
candidate-supply emitter. It passes both the preregistered raw-coverage gate and
the stricter target-blind volume-matched null gate. It must not replace or
directly average scores with any existing emitter.

This is organizer-train FIT cache-capacity evidence only. It is not ranking,
solver, source-disjoint DEV, leaderboard, or submission evidence.

## No-repeat audit and fixed recipe

The prior-research ledger and current code contain median, bilateral, Gaussian,
NLM, guided, Wiener, local-rank/Census, contour, unsharp, neural denoiser, and
restoration matcher views. No pixel-domain wavelet shrinkage matcher result was
found. Historical spectral shrinkage operated on a similarity matrix and is a
different mechanism.

Before any label access, exactly one recipe was fixed:

- upright `20x20` dirty tiles and RGB channels are processed independently;
- one orthonormal 2-D Haar level on top-left-aligned non-overlapping `2x2`
  blocks;
- diagonal-detail MAD divided by `0.67448975` estimates noise per tile/channel;
- standard BayesShrink soft threshold is computed independently for each of the
  three detail bands;
- the low-pass band is unchanged;
- reconstructed pixels are matcher-only and original upright tile pixels remain
  the only renderable pixels.

No wavelet, level, phase, threshold, strength, or fusion sweep was run.

## Freeze and score separation

The target-free config
`configs/wavelet_candidate_fit_preregistered_v1.json` (SHA-256
`bc4640887a0ab0517e2773bc1430ef1d7414df4e7d1ae311447c9f6674b7fb11`)
fixed the transform, immutable FIT32 x two-draw roster, raw `+0.3 pp` gate, and
volume-matched-null gate before labels. All 64 all6+wavelet top-32 identity
rosters were written before a label archive was opened. The freeze took
`34.51 s` CPU and is recorded in
`outputs/wavelet-candidate-emitter/fit32-draw2-v1/pre-label-freeze.json`
(SHA-256
`cd1c72b462b56ebcfeb6f10ccf10cb70d52102103fc5493b9beb340df5e09adb`).

Only afterwards, the separately signed binding
`configs/wavelet_candidate_fit_score_binding_v1.json` (SHA-256
`3c2f05afcd5b2bf6ad7fd42fbe693806b91182f441cc17177a4c0e94667ff9a6`)
authorised one aggregate FIT label pass. It forbids fitting, parameter
selection, and DEV/local/terminal/test/submission access.

## Raw incremental result

There are `70,656 = 64 x 1,104` directed right/down true-neighbour queries.

| Candidate supply | Exact hits | Coverage |
|---|---:|---:|
| Frozen all6 union | 57,593 | 81.5118% |
| Haar-BayesShrink top-32 alone | 31,696 | 44.8596% |
| All6 + Haar-BayesShrink | **58,048** | **82.1558%** |
| Unique wavelet recovery over all6 | **455** | **+0.6440 pp** |

The direction split is `+244` right and `+211` down. Raw incremental recovery
is positive on `64/64` cases and `32/32` two-draw source groups, with no zero or
negative group. Per-case min/median/max is `3/7/17`, mean `7.109`. The fixed
source-group bootstrap CI95 is `[6.203, 8.063]` additional neighbours per case,
or `[+0.5619,+0.7303] pp` absolute coverage.

## Volume-matched null correction

Candidate unions grow even when a new emitter is random, so raw unique coverage
is not sufficient evidence. The null was fixed before labels:

For each eligible query whose truth is absent from all6, let `m` be the number
of wavelet identities absent from that row's all6 union. Uniformly drawing the
same `m` identities without replacement from the remaining
`575 - |union6|` identities has exact hit probability
`m / (575 - |union6|)`. This matches the actual target-blind candidate volume
row by row; no Monte Carlo candidate draw is used.

| Conditional all6 misses | Actual wavelet | Matched uniform null | Specific excess |
|---|---:|---:|---:|
| Right (`6,779`) | 244 | 55.310 | **+188.690** |
| Down (`6,284`) | 211 | 50.854 | **+160.146** |
| Pooled (`13,063`) | **455** | **106.164** | **+348.836** |

Actual available-miss hit rate is `3.4831%`, versus `0.8127%` under the
volume-matched null, an excess of `+2.6704 pp` conditional on a miss. Relative
to all eligible edges, specific excess is **+0.4937 pp**, above the fixed
`+0.1 pp` minimum.

Excess is positive on `64/64` cases and `32/32` source groups. The source-group
bootstrap CI95 is `[4.597, 6.337]` excess hits per case, or
`[+0.4164,+0.5740] pp` over all eligible edges. Both the size gate and strictly
positive lower-CI gate pass.

## Boundary and next use

The wavelet view supplies genuinely non-random exact-neighbour identities, but
coverage does not show that its scores can be ranked safely. Keep it only as a
seventh append-only identity supply for a separately frozen joint verifier or
selector that retains all six existing arms. Do not directly fuse its scores,
replace raw pixels, open a solver run from this result alone, or infer official
S-Team improvement.

Primary report:
`outputs/wavelet-candidate-emitter/fit32-draw2-v1/capacity-report.json`, SHA-256
`0e57b96d7bf3e6549795b8fb916001e9120e0d6ca8643ff358899f3b6731f0db`.

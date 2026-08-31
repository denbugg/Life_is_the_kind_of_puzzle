# Local-Wiener candidate emitter: FIT capacity

## Decision

Retain the fixed local-Wiener view as a fifth append-only candidate-supply
emitter for a future jointly scored model.  It must not replace raw, guided, or
any existing legacy emitter, and fixed score averaging is not authorised.

This is organizer-train FIT cache-capacity evidence only.  It is not a model,
solver, source-disjoint DEV, promotion, leaderboard, or submission result.

## Why this is not a repeated denoiser screen

The no-repeat ledger already closes median, colored NLM, mild/strong bilateral,
Gaussian, unsharp, guided filtering, contour views, DRUNet, DualNAF, and similar
hard replacement paths.  BM3D was tested as an output tail, not as a tile
matcher view.  No prior local-Wiener matcher implementation or result was found.

The fixed new transform is a parameter-free classical local Wiener shrinker:

- reflected `3x3` window;
- local mean and variance per upright `20x20` tile and RGB channel;
- noise variance equal to mean local variance of that same tile/channel;
- no cross-tile or cross-channel pixels;
- filtered pixels are matcher-only and are never rendered.

The choice was fixed from the known corruption mechanism and standard Wiener
estimator before FIT labels were read.  No window or strength sweep was run.

## Protocol separation

Config
`configs/wiener_candidate_emitter_fit_preregistered_v1.json` (SHA-256
`66bb44649219a5ced0f8474a02b4e6f6d5a02d3b63327325953ff1725d948945`)
authorised only a target-free freeze on the immutable FIT32 x two-draw roster.
The runner safely loaded the five label-free legacy arrays and never
materialised legacy `target_slots`.

All 64 Wiener top-32 rosters were saved before a label file was opened.  The
pre-label freeze is
`outputs/wiener-candidate-emitter/fit32-draw2-v1/pre-label-freeze.json`, SHA-256
`ee54db7420f74c6654e85a691591dd0b01a60ae0738162a0e4b0de5d1391895c`.
The complete CPU freeze took `31.52 s`.

Only afterwards, the separate binding
`configs/wiener_candidate_emitter_fit_score_binding_v1.json` (SHA-256
`025b0f8744b1143e0a110885896535cbf899edbc2e72ce76890a595d202518ac`)
authorised aggregate FIT coverage counting against the already physically
separated exact-neighbour labels.  It forbids fitting, parameter selection, and
all DEV/local/terminal/test/submission access.

## Standalone and legacy-union result

There are `70,656 = 64 x 1,104` directed right/down true neighbours.

| Supply | Exact hits | Coverage |
|---|---:|---:|
| Raw top-32 | 46,779 | 66.2067% |
| Wiener top-32 | 32,224 | 45.6069% |
| Legacy raw+adapter+DINO union | 53,333 | 75.4826% |
| Legacy+Wiener union | 55,832 | 79.0195% |
| Wiener unique over legacy | **2,499** | **+3.5369 pp** |

All `32/32` source groups have positive unique recovery.  Mean gain is `39.047`
true neighbours per case; a fixed source-group bootstrap gives CI95
`[36.391, 41.734]` per case.

## Incremental value beyond the guided fourth emitter

Wiener and guided are almost tied when each is appended to the legacy union,
but they are not duplicates:

| Supply | Exact hits | Coverage |
|---|---:|---:|
| Legacy+guided | 55,839 | 79.0294% |
| Legacy+Wiener | 55,832 | 79.0195% |
| Legacy+guided+Wiener | **56,813** | **80.4079%** |

Wiener recovers `974` neighbours absent from the full legacy+guided union:
`+1.3785 pp`, positive on `32/32` source groups, mean `15.219` per case, with
source-group bootstrap CI95 `[13.734, 16.781]`.  Symmetrically, guided adds
`981` neighbours beyond legacy+Wiener.  The two filters therefore have nearly
equal capacity and material complementary tails.

## Recommendation and boundary

Keep Wiener only as an append-only fifth identity emitter.  The next legitimate
consumer would be a separately signed joint verifier that can select among the
larger frozen union while hard-retaining raw candidates.  Coverage alone does
not show that Wiener identities can be ranked safely, so there is no basis for
direct fusion, solver replacement, or a real-data run yet.

Primary report:
`outputs/wiener-candidate-emitter/fit32-draw2-v1/capacity-report.json`, SHA-256
`162cdd818bed9550de0e633a60ec84395ee3fab0ae4ff211a656046f7de72f19`.
Five-emitter supplement:
`outputs/wiener-candidate-emitter/fit32-draw2-v1/supplemental-five-emitter-coverage.json`.

# Joint reciprocal v2: unsigned FIT fixed-head audit

Status: **unsigned post-hoc FIT diagnostic only**. This audit describes the
already frozen and already scored v2 fixed 5% reciprocal head. It did not train
a model, change a threshold, select a checkpoint, run Weco, or access
DEV/local/terminal/competition-test/submission material. Every signed input is
unchanged.

The result is in-sample: the endpoint was fitted on these same 32 organizer
sources and two corruption draws. It is useful for detecting collapse,
direction imbalance and source concentration, but it is not promotion evidence
and must not be presented as expected unseen precision.

## Fixed contract and aggregate replay

Each of the 64 cases contributes exactly 29 selected right edges and 29 selected
down edges: `3,712` selected relations in total. Reconstructing correctness from
the immutable FIT caches exactly reproduces the signed aggregate score:

| Axis | Correct / selected | Precision |
|---|---:|---:|
| Right | 1,633 / 1,856 | 87.9849% |
| Down | 1,708 / 1,856 | 92.0259% |
| Pooled | 3,341 / 3,712 | 90.0054% |

Coverage is complete on every case and axis. Down is about `+4.04 pp` higher
than right, so the pooled headline should not hide the directional gap.

## Distribution, not only the mean

Quantiles use NumPy's fixed linear convention. Counts at 80% and 90% are
descriptive; neither value is used as a cutoff.

### Per case

| Axis | Mean | Median | Q25 | Q75 | Min | Max | >=80% | >=90% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Right | 87.985% | 89.655% | 82.759% | 93.103% | 72.414% | 100% | 56/64 | 19/64 |
| Down | 92.026% | 93.103% | 86.207% | 100% | 75.862% | 100% | 57/64 | 36/64 |
| Pooled | 90.005% | 91.379% | 86.207% | 94.828% | 74.138% | 98.276% | 61/64 | 33/64 |

### Paired source, combining both draws

| Axis | Mean | Median | Q25 | Q75 | Min | Max | >=80% | >=90% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Right | 87.985% | 89.655% | 84.483% | 93.103% | 74.138% | 98.276% | 29/32 | 12/32 |
| Down | 92.026% | 94.828% | 87.069% | 96.552% | 77.586% | 100% | 31/32 | 21/32 |
| Pooled | 90.005% | 90.948% | 86.638% | 93.966% | 77.586% | 97.414% | 30/32 | 16/32 |

A fixed 20,000-draw bootstrap resampling the 32 sources, while keeping their
two corruption draws paired, gives:

| Axis | Source-bootstrap CI95 for mean precision |
|---|---:|
| Right | [85.938%, 89.926%] |
| Down | [89.709%, 94.181%] |
| Pooled | [88.227%, 91.676%] |

These intervals quantify FIT source heterogeneity, not generalisation.

## Lower tail and source influence

The three lowest individual cases are:

| Source / draw | Right | Down | Pooled |
|---|---:|---:|---:|
| `img_006141.png` / 1 | 21/29 | 22/29 | 43/58 = 74.138% |
| `img_002898.png` / 1 | 21/29 | 23/29 | 44/58 = 75.862% |
| `img_001988.png` / 0 | 24/29 | 22/29 | 46/58 = 79.310% |

The weakest paired source is `img_006141.png`: `90/116 = 77.586%`. It contains
26 of all 371 pooled errors, or 7.01%. Removing it raises pooled precision only
from 90.0054% to 90.4060%, a `+0.4006 pp` leave-one-source effect. The source
with the largest correct-numerator contribution is `img_004562.png`, with
113/116 correct and 3.382% of all correct selections; removing it lowers pooled
precision by `0.2390 pp`.

Axis-specific maximum absolute leave-one effects are `+0.4467 pp` for right
after removing `img_005357.png`, and `+0.4658 pp` for down after removing
`img_006141.png`. Thus no single source creates the 90% mean, although a real
lower tail remains.

## Descriptive confidence calibration

Bins were fixed numerically as
`[-inf,-8,-4,-2,-1,0,1,2,4,8,inf]` and were not used to choose a cutoff. Only
five pooled bins are nonempty:

| Joint confidence | Count | Correct | Precision |
|---|---:|---:|---:|
| [-1, 0) | 70 | 42 | 60.00% |
| [0, 1) | 1,122 | 896 | 79.857% |
| [1, 2) | 1,441 | 1,343 | 93.199% |
| [2, 4) | 970 | 951 | 98.041% |
| [4, 8) | 109 | 109 | 100% |

The relationship is monotone on this FIT set. This is descriptive calibration
of already selected edges, not evidence for changing the fixed 5% head or
introducing a confidence threshold.

## Comparator availability

An exact raw/prior-tri comparison is **unavailable**. No target-free raw or
prior-tri head was frozen on this exact FIT64 roster with the same contract of
29 selected reciprocal edges per axis per case. Existing prior diagnostics use
native/unequal coverage or another panel. Reconstructing a new baseline now
would introduce a post-hoc head policy, so this audit deliberately does not do
it.

## Artifacts and boundary

Machine-readable audit:
`outputs/joint-reciprocal-tri-emitter-verifier/real-fit32-draw2-dev32-development-v2/fit/unsigned-fixed-head-distribution-audit.json`.

The audit verified these immutable inputs:

- signed score SHA-256
  `d7601a30fe008c05b331b71a6ddd2eb3d110fb39bd25291a1a8de2e22a80d81c`;
- target-free archive SHA-256
  `f2a60ccfc5d8d09d25e4648bf7226e1364b3bd2575275be6bb6a7b9ac562257c`;
- target-free metadata SHA-256
  `1f1e6784fd61b913b560d6149af2ef5fea3a2473455744e82bca6bfa75f84153`;
- pre-score freeze SHA-256
  `5921efae286e97da430a2bca3cc1fe4dcd711db412a70fa488472158bdc98448`;
- endpoint SHA-256
  `66244123312b794ea6c1ae077f608653db99a122473a47d25712a374e3fe7747`;
- signed config SHA-256
  `c8ffae9c11d5d101f92f0b769b0d5f6e6bfc68f771239bc18c83af0b2b401880`.

No result here authorises a solver, decoder, model promotion, production change
or submission.

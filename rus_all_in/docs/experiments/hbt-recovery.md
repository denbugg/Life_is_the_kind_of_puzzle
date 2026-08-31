# HBT side-embedding recovery under the current protocol

Status: **closed after the 256-source pilot; do not scale to 2048**.

The bounded recovery answered a narrow question: does the strongest historical
full-board hard-negative side embedding improve the current, artifact-free,
bijective atlas decoder on real dirty calibration inputs?  It does not.  The
learned edge matrices are retained as research artifacts, but none passed the
predeclared downstream gate.

## What was recovered, rather than reinvented

The architecture and loss are a mathematical port from the read-only branch
`origin/таска-говно`:

- commit `d6a82f82ceefa109ef706402712d03805bc9e880`;
- `source/src/puzzle_assembly/learned.py` blob
  `fa6209701c06667526bc609158874df96618dc47`;
- `source/scripts/train_side_embeddings.py` blob
  `8277cb961e9bedd8e41e0b0bade2615f757b6db5`.

The selected historical configuration was kept exactly: pooled
`SideEmbeddingNet`, 64 channels, 320-dimensional query/key vectors, 4-pixel
side band, 10 tangent bins, `rgb_sobel`, temperature 0.07, hardest negative
among all other 575 tiles, margin 0.2, CE weight 0.25, raw-embedding L2
`1e-4`, outside-edge weight 0.2, AdamW `3e-4` / `1e-4`, and gradient clipping
at 1.  The historical 2048-source results reported local R@1 of 0.1790 on raw
tiles and 0.2238 after a separate TileNAF denoiser, but those checkpoints were
not present in the fetched repository and had no current-protocol downstream
evaluation.

The new code is intentionally only protocol glue and an exact recovery:

- `src/aiijc_puzzle/hbt_recovery.py` — labels, historical network and loss,
  synthetic panel, guarded views and dense-score extraction;
- `scripts/run_hbt_recovery.py` — deterministic paired training, bound
  checkpoints, target-free freeze, strict decoder and post-freeze metrics;
- `tests/test_hbt_recovery.py` — grid labels, finite full-board gradients,
  deterministic corruption and the exact `rgb_sobel`/d320 score contract.

## Frozen protocol and leakage boundary

- Train membership: first 256 shared-selector records from manifest `train`.
- Training data: clean target only, then two deterministic replicas of the
  reverse-engineered organizer corruption, one per epoch.  Each replica applies
  independent per-tile contrast 0.70–1.30, brightness ±30, noise sigma 40–55,
  separable blur3, JPEG quality 35–50, and a fresh 576-tile permutation.
- Paired arms start from identical weights: raw tiles and guarded bilateral
  tiles (`d=5`, sigma color 25, sigma space 5).  Raw remains an explicit arm.
- The available R6 restorer was not used as a third view: it is a post-layout
  full-canvas network, not a frozen tile denoiser.  Applying it before layout
  would violate its semantics.
- Authoritative evaluation: untouched shared-selector calibration records
  84:96, 12 boards.  The trained checkpoints and complete variant roster were
  frozen before any of those 12 targets were loaded.
- Every prediction uses all 576 input fragments exactly once in a 24×24 grid.
  The decoder is buddies96 plus the train-only population atlas at weight 0.03;
  five colored NLM h=10 passes happen only after the raw permutation audit.
- Targets are used after freezing only for approximate Hungarian label recovery,
  adjacency/placement diagnostics, and SSIM.  Holdout and test were not opened.

The first smoke mistakenly used calibration offset 72, so the first 72:84
pilot panel was not fully fresh.  Its result is retained as a non-authoritative
diagnostic.  No weights or roster were changed after seeing it; the authoritative
checkpoint-only run moved to the wholly untouched 84:96 slice.

Selection digests:

- train-256: `4e407402ad5c81fd1698c65a22da5c5b8d12ea886608eed337051473e54348dc`;
- calibration 84:96: `67ba6d10606a65f58cb587ef20b9c4ba814337ec94b2a3918812401ee313db2f`.

## Commands and runtime

Focused verification:

```bash
uv run ruff format --check src/aiijc_puzzle/hbt_recovery.py \
  scripts/run_hbt_recovery.py tests/test_hbt_recovery.py
uv run ruff check src/aiijc_puzzle/hbt_recovery.py \
  scripts/run_hbt_recovery.py tests/test_hbt_recovery.py
uv run pytest tests/test_hbt_recovery.py
```

Training run (its 72:84 evaluation is non-authoritative for freshness):

```bash
uv run python scripts/run_hbt_recovery.py \
  --output-dir outputs/hbt-recovery/pilot-train256-cal12-offset72 \
  --train-limit 256 --epochs 2 --eval-offset 72 --eval-limit 12 \
  --device mps --views raw bilateral
```

Authoritative checkpoint-only evaluation:

```bash
uv run python scripts/run_hbt_recovery.py \
  --output-dir outputs/hbt-recovery/pilot-train256-fresh-cal12-offset84 \
  --checkpoint-dir outputs/hbt-recovery/pilot-train256-cal12-offset72 \
  --train-limit 256 --epochs 2 --eval-offset 84 --eval-limit 12 \
  --device mps --views raw bilateral
```

MPS training took 79.07 s.  The authoritative dirty-only freeze took 51.35 s
and target-assisted evaluation 2.39 s.  Four focused tests passed in 0.64 s.

## Results on untouched calibration 84:96

Synthetic train R@1 rose from 3.43% to 4.89% for raw and from 3.40% to 4.93%
for bilateral.  The actual dirty calibration retrieval remained below the
classical bilateral matcher:

| dense compatibility | pooled R@1 | R@5 | R@32 |
|---|---:|---:|---:|
| classical bilateral | 0.06280 | 0.16018 | 0.39138 |
| HBT raw | 0.03223 | 0.09715 | 0.26910 |
| HBT bilateral | 0.03910 | 0.11051 | 0.28570 |
| mean(HBT raw, HBT bilateral) | 0.03842 | 0.10771 | 0.27959 |
| mean(HBT pair, classical) | **0.06416** | **0.16244** | 0.39070 |

Full-board results use the identical atlas+buddies96 decoder for every row:

| variant | raw SSIM | NLM5 SSIM | adjacency | right | down |
|---|---:|---:|---:|---:|---:|
| classical bilateral baseline | 0.110144 | 0.251702 | 0.035553 | 0.032609 | 0.038496 |
| HBT raw | 0.111738 | **0.265192** | 0.011398 | 0.010266 | 0.012530 |
| HBT bilateral | 0.109142 | 0.264531 | 0.011549 | 0.010870 | 0.012228 |
| HBT pair mean | 0.108290 | 0.261139 | 0.012228 | 0.011322 | 0.013134 |
| HBT pair + classical | 0.109434 | 0.254057 | **0.036458** | 0.032609 | **0.040308** |

The scale gate required, against the baseline, strictly positive raw SSIM,
NLM5 SSIM and pooled adjacency, plus non-negative right and down adjacency.
No arm passed:

- HBT raw gained `+0.001594` raw and `+0.013490` NLM5 SSIM, but lost
  `−0.024155` adjacency (`−0.022343` right, `−0.025966` down).
- HBT bilateral lost `−0.001001` raw SSIM and `−0.024004` adjacency despite
  `+0.012829` NLM5 SSIM.
- the fixed HBT-pair/classical fusion recovered `+0.000906` adjacency and
  `+0.002356` NLM5 SSIM, but lost `−0.000710` raw SSIM.

Because the quantitative prerequisite already failed, no visual-coherence
promotion sheet was used and no 2048-source training was run.  The large NLM5
gain of the pure learned arms is not evidence of a better puzzle: their exact
retrieval and adjacency collapse.  It is safest to treat that gain as a
post-restoration metric interaction, not as recovered geometry.

## Artifacts and hashes

- authoritative report:
  `outputs/hbt-recovery/pilot-train256-fresh-cal12-offset84/report.json`,
  SHA-256 `c3f2a519fc4af35515b325de7b427402dddb9cfd5b6301b7d001231c3d5f8599`;
- pre-target frozen manifest: `frozen-inference.json`, SHA-256
  `e8f4a0fdf3b34a071415aa35455095765d86e5e52dec96c7a2ed4685b24565d8`;
- raw checkpoint: SHA-256
  `ae109f549ab40107d92256da31738a3beb7af214add14c603d58fca42a760645`;
- bilateral checkpoint: SHA-256
  `fad240b8818f176f905a7a7314870a0531a12604abc7a1e4b0d5aa8b04035320`;
- `hbt_recovery.py` training/runtime hash:
  `44cf575686c06eab9555a901ad1db282d58f884ee1b172e898f3be9289b0bebb`;
- runner training/runtime hash:
  `f4d0ec96c14bf0f2c88e915d83288bf452d4b1c89adf6969214daf58386ac972`.

## Verdict and limitations

Close this 256-source recovery as a global layout candidate.  The exact
historical mechanism generalizes only weakly from the synthetic corruptor to
current real dirty edges at this scale, and pure HBT actively destroys the
local geometry that the strict decoder needs.  The fixed classical fusion is a
useful auxiliary observation, but its tiny adjacency gain does not compensate
for the raw-SSIM regression and does not justify the requested 2048 scale.

The recovered exact labels themselves depend on target-assisted Hungarian
matching and can contain noise; that is a limitation of diagnostics, not an
inference leak.  The conclusion is nevertheless robust to that caveat because
the same recovered mapping evaluates every arm and the pure-HBT adjacency loss
is about 2.4 percentage points, far larger than the fusion-level noise.

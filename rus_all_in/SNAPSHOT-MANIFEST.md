# Rus all-in snapshot manifest

Срез: 2026-08-31. Этот каталог добавлен поверх legacy ветки Pasha883; legacy
файлы не удалялись и не переписывались. Competition test, submission и
prediction outputs в snapshot не входят.

## Current result and metric contract

- Official leaderboard best остаётся `0.2762279116935955`; это число сообщено
  пользователем, соответствующий прежний submission не пересобирался.
- Confirmed pair leader — `aiijc-taska-relation-selector`: formal
  source-disjoint `16×2`, `332.21875→338.06250` pairs/board,
  delta `+5.84375`, source-CI95 `[+3.000,+9.12578]`, exact delta `-0.15625`.
- Socket cyclic origin transfer — полезный exact signal
  (`5.9375→12.875` exact), но pair gate провален (`-3.34375`), поэтому он
  сохранён как trade-off evidence, не production arm.
- Solver metrics: absolute exact/radius0 primary; absolute mean Manhattan
  secondary; radius2 и satisfied pairs companions; dirty/restored SSIM final;
  cyclic-aligned distance diagnostic-only.

Главные документы:

- [`docs/BEST-current.md`](docs/BEST-current.md);
- [`docs/NEXT-solver-roadmap.md`](docs/NEXT-solver-roadmap.md), SHA-256
  `bd0484bd6770cecf2a00cbdaa6d7e8bc03970653d86cb9469f39afd6380a1459`;
- [`docs/experiments/taska-relation-truth-selector.md`](docs/experiments/taska-relation-truth-selector.md);
- [`docs/experiments/tile-position-distance-metric-validation.md`](docs/experiments/tile-position-distance-metric-validation.md);
- [`docs/experiments/tri-emitter-edge-verifier.md`](docs/experiments/tri-emitter-edge-verifier.md).

## Included runtime/evidence chain

Promoted relation selector может fail-closed проверить весь frozen runtime
chain по тем же paths, которые зафиксированы в source:

| Artifact | SHA-256 |
|---|---|
| `artifacts/prior-taska/ckpt/seam_embed_v3.pt` | `6f0917d66d908f6cc0f4c1fcb949d3bcbadcba2490a6f7b5a12596e61de9730e` |
| `artifacts/prior-taska/ckpt/seam_embed_local.pt` | `5932853a73961d261b494368a4db04633fecc5996771c14d64f49ef00c7cfe73` |
| `artifacts/prior-taska/ckpt/verify_pair_best.pt` | `3bcc89a12e7b539304484b441688b4b9fb1c3711e918befed9cdef7c17f776e7` |
| `outputs/taska-edge-calibrator/train256-v1/calibrator.npz` | `adc76ee87fc112d4ca3eeb676cdec6b7d103c596d62a9848ba65ee5ef384b1ac` |
| `outputs/taska-nonlinear-calibrator/train256-v1/calibrator.npz` | `2a5f95bd9d8e08e57b8bd02e242e25ef4661036ed3b1985fda1d70ee1bf9d2a6` |
| `outputs/fullres-boundary-denoiser/pilot-train32-s400-eval16-auto/fullres_boundary_denoiser.pt` | `a6dfc3e264e97d93ad678f3ee97e070067357c2a6f6875e7b7432f880aa1492c` |
| `outputs/taska-relation-truth-selector/fixed-v1/model-local32-held32/frozen-relation-classifier.pkl` | `ec4eca99243cdc6be20104d789b9e5d5598b79fa0d1b7e69bc37314375ad8c6b` |
| `outputs/taska-relation-truth-selector/formal-confirmation-v1/frozen-target-free-eval.npz` | `4cd0346333813cea3576f6db40ea517dcc45fdd5aa81a432a351cf4afdd73131` |
| formal relation report | `d260872251077e1515251b6c7afc316af25df75045c8119112dff4f36c68ea23` |
| distance validation report | `57308b3bc944226022fcba0a52a55fa2ffd50391f0aa41b368f28c1bb9957ad6` |
| relation distance bridge report | `2f14336e91ca889e9c8777f90ee596a7f390cfeacb7a82378a140b42a9781104` |

Also included: required fusion reports, model/freezing metadata, Socket exact
trade-off reports and frozen layouts, distance frozen roster, all compact Weco
notes, and the non-promoted tri-emitter final/capacity weights plus reports.
Tri final checkpoint SHA-256 is
`e7afa13a5090369bb407e3cb9f48f4592a78a190f32cfbc04d0e390a8a7f1d8c`;
its signed local matched-precision gate failed, so it is research/handoff only.
Scale3200 is represented only by the signed config, runner/tests/docs and
`ABORTED_BEFORE_CHECKPOINT_OR_SCORING.json`: no checkpoint existed.

## Deliberately omitted

- all `data/raw`, `data/interim`, `data/processed`, competition test and targets;
- all submission ZIPs, predictions, PNG outputs and temporary caches;
- tri fit caches and 28 MiB frozen prediction array;
- non-required fusion 72 MiB evaluation archive and fullres 7 MiB prediction cache;
- DINOv2 and DRUNet checkpoints; neither is part of the promoted relation arm.

For optional research reproduction, download only from the official locations
and verify before use:

- DINOv2 ViT-S/14:
  `https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth`,
  SHA-256 `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`;
- KAIR DRUNet color:
  `https://github.com/cszn/KAIR/releases/download/v1.0/drunet_color.pth`,
  SHA-256 `479abe3c5327dfd10ff54a80ec7d4098ca80752a5c9492cdff31cee430bec4b4`.

## Run and verification status

```bash
cd rus_all_in
uv sync
uv run aiijc-taska-relation-selector tiles.npy \
  --output-layout layout.npy --diagnostics-json receipt.json --device mps
```

`tiles.npy` must be one legal `uint8[576,20,20,3]` bag. The layout output is
only a strict `int32[576]` permutation of original upright tile IDs; restored
pixels are matcher-only.

At snapshot time:

- in the canonical workspace, `uv run ruff check .` passed and the current
  solver slice passed `24/24` tests (relation truth/pipeline, ranked-union
  runner+module, distance, tri verifier and scale3200);
- in this portable copy, Ruff passed, the data-independent subset passed
  `20/20`, and the promoted relation adapter verified all `22/22` pinned
  source/model/report/archive SHA records. Four canonical slice tests require
  deliberately omitted raw target pixels or non-promoted ranked-union archives;
  they were not represented as portable failures or silently claimed green;
- full pytest collection still has 13 known failures: 12 pre-existing immutable
  legacy/hash-binding drift cases and one MPS deterministic-backward limitation.
  Therefore this snapshot does **not** claim the historical full suite is green.

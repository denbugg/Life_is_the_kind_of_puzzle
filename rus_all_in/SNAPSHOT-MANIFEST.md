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
  `614f08e530eae4331a2d0b270adf529ccc282ea5b0b61d3bff0ddfaeb811bc28`;
- [`docs/experiments/taska-relation-truth-selector.md`](docs/experiments/taska-relation-truth-selector.md);
- [`docs/experiments/tile-position-distance-metric-validation.md`](docs/experiments/tile-position-distance-metric-validation.md);
- [`docs/experiments/tri-emitter-edge-verifier.md`](docs/experiments/tri-emitter-edge-verifier.md);
- [`docs/experiments/joint-reciprocal-tri-emitter-verifier.md`](docs/experiments/joint-reciprocal-tri-emitter-verifier.md);
- [`docs/experiments/six-emitter-joint-consumer-design-unsigned.md`](docs/experiments/six-emitter-joint-consumer-design-unsigned.md);
- [`docs/experiments/joint-native-head-arm-fit.md`](docs/experiments/joint-native-head-arm-fit.md).

## Joint-v2, decoder and scale update

- Signed joint reciprocal tri-v2 прошёл source-disjoint DEV32 retrieval gate.
  Относительно raw pooled R@1/R@5 выросли на `+0.7133/+1.1690 pp`, fixed 5%
  head precision — `77.7478%→88.0927%` (`+10.3448 pp`). Read-only
  source-bootstrap audit дал положительные CI для всех девяти retrieval/head
  метрик и не обнаружил single-image domination. Это matcher/verifier signal,
  не готовый solver/layout gain.
- Отдельный FIT64 fixed-head score подтвердил precision
  `3341/3712 = 90.0054%` (right `87.9849%`, down `92.0259%`). Но
  compatibility-aware structured oracle не имел нужной ёмкости: mean true-edge
  headroom `1.047` и realised gain `0.891` при gate `>=8`; новый model не
  разрешён.
- Same-case relation-selector bridge изменил `10/32` layouts и провалился:
  pairs `353.344→348.406` (`-4.938`), Manhattan хуже на `0.0554`; exact
  вырос только на `+0.094`. Не продвигать и не подбирать threshold рядом.
- Joint-native head-first rebuild также закрыт. На единственном frozen FIT64
  score 58 anchors дали pairs `349.484→69.063`, delta `-280.422`, case/source
  W/T/L `0/0/64` и `0/0/32`; exact delta `-1.781`, Manhattan benefit
  `-0.584`. Все пять gates fail. Reciprocal head остаётся только локальным
  repair/constraint signal поверх сильного layout.
- FIT256×2 target-free feature cache завершён `512/512`: `3,588,352,000`
  bytes (`3.342 GiB`), mean feature time `2.090 s/case`; reserved DEV64 в
  cache stage не открывался. В snapshot входит только signed protocol, runner,
  tests и текстовый cache report; сами 512 NPZ caches не входят.
- Fixed Haar-BayesShrink прошёл volume-matched supply gate: `+455` raw unique
  true-neighbour hits, matched-null specific excess `+348.836` (`+0.4937 pp`),
  source-CI95 `[+0.4164,+0.5740] pp`, `64/64` cases positive. Поэтому
  default-six roster фиксирует raw+adapter+DINO+guided+Wiener+Haar и исключает
  local-rank. Default-six пока **unsigned design only**: training/DEV/solver
  result не заявляется.
- Guided-four real-roster audit сохранён как отдельный blocked result: recursive
  inventory исключает все `5600/5600` organizer-train sources, поэтому свежий
  source-disjoint DEV32 невозможен. В snapshot входят стабильные roster module,
  audit runner, unit tests и компактный audit report; target-free cache,
  pre-label metadata и separated FIT labels намеренно не входят.

Ключевые новые текстовые bindings/evidence:

| Artifact | SHA-256 |
|---|---|
| joint-v2 signed config | `c8ffae9c11d5d101f92f0b769b0d5f6e6bfc68f771239bc18c83af0b2b401880` |
| joint-v2 DEV score | `9548487b73481d5ec01963911a75c62d117ae634d7105df708edad1802be5274` |
| joint DEV robustness audit | `74145b170c085bcf675aa7d07167185a6b535a58fdf7b805daa6f6922c5fb36d` |
| structured capacity report | `bc3ad541df586911d14d90d74d599e2528085c9c62d2e71c66bdafe0bdb88621` |
| relation bridge score | `468bcc4e5ac1c512b2a5fcc1d6047af8df1925693ffb4575556d990896303d9c` |
| joint-native one-score report | `33277e20097c47a33037188b4db98f547565b9aaa8b4a17854899fe39d061971` |
| joint-native compact protocol audit | `5f8202f69434a0777e1c22d5a0d8d414c413bb188759beeb233b066cbf6cd6b1` |
| FIT256 cache signed config | `3e397e7ff3a565de2b1ab412f71f8e5b7d500649b40a51be2ed97805ecc7344e` |
| FIT256 cache report | `04c0bf7edebc8809f854a95bf97b361ad28fec5b12eb26e54d4c780b2c8d823d` |
| signed FIT256→DEV64 transition config | `e3fe4aa5c594b149c4dab93960aabf220754d82bf938edef852658356fc9bf3b` |
| default-six unsigned template | `5db0860c7dd0b14c1cf36894f1a3c4e2bc3ccc32d43df8042059cd0f34f11647` |

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
- DINOv2 and DRUNet checkpoints; neither is part of the promoted relation arm;
- all new joint NPZ/PT endpoints, target-free prediction/layout archives and
  the complete 3.342 GiB FIT256 cache. Signed configs retain their canonical
  paths/hashes, so artifact-bound replay checks deliberately fail closed here;
- guided-four target-free cache, pre-label metadata and separated FIT labels;
  их hashes остаются в unsigned blocked template, но сами label/cache artifacts
  не нужны для воспроизведения roster-inventory blocker;
- the in-progress `scale-fit256-draw2-dev64-real-v1` live directory/log. This
  snapshot makes no claim about its unfinished training or DEV result;
- all in-progress BasinCycle Stage-B code/config/report files. They were
  intentionally left in the canonical workspace until that work is stable.

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
- the synchronized joint/default-six/scale/roster-audit text slice passes Ruff.
  The exact artifact-independent portable invocation passes **`100/100`**.
  The complete synchronized staged slice collects `111` tests: `100` pass and
  `11` fail closed on deliberately omitted artifacts. Seven are non-scorer
  bindings: one omitted default-six endpoint, one guided-four cache/label chain,
  two Socket-parent-report checks plus one validation-manifest check in the
  scale-cache materializer, one omitted frozen guided config and one scale-real
  validation-manifest check. Four scorer checks require omitted frozen NPZs:
  one v2 reciprocal-head proof and three v3 frozen-layout overlays. The second
  v2 manifest-roster unit test is artifact-independent and is included in the
  passing `100/100`. In the canonical workspace the joint-native v1/v2/v3 slice
  passed `10/10` before the one authorised score;
- in this portable copy, the earlier promoted-relation subset passed `20/20`,
  and the promoted relation adapter verified all `22/22` pinned
  source/model/report/archive SHA records. Four canonical slice tests require
  deliberately omitted raw target pixels or non-promoted ranked-union archives;
  they were not represented as portable failures or silently claimed green;
- the pre-joint full collection had 13 known legacy/hash/MPS failures. The
  expanded portable tree also contains the explicitly omitted artifact-bound
  checks above, so this snapshot does **not** claim the full suite is green.
- `git diff --cached --check` retains one intentional EOF-blank warning for
  `scripts/freeze_joint_reciprocal_fit_heads_target_free.py`: its exact bytes are
  bound by signed config/report SHA
  `0a9332e8871c83bafa5408d3774bc25c6455e62c46e5b0cc10fc6973bd129524`.
  The two unbound EOF warnings were removed; the frozen script was not rewritten
  or re-signed for whitespace-only cleanup.

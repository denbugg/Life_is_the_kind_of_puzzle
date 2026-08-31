# Ultimate legal stack: h28 + cap08 fusion + DualNAF alpha=0.125

## Решение

**Reject as tested. Confirmation `444:468`, holdout, competition test и frozen
production не открывать и не изменять.** Единственный promotable arm D получил
`0.250107`, не достиг absolute gate `0.27` и проиграл strong legal control B с
bilateral layout и h28 (`0.252481`). Geometry gates conservative fusion тоже не
прошли. Все detail/grid safety bounds и relative manual-artifact gate прошли,
но они не компенсируют quantitative failure.

## Fixed hypothesis без sweep

В один preregistered stack были сложены только ранее измеренные компоненты:

- h28 — maximum manual-safe dense single-pass NLM strength;
- non-destructive k16 fusion `cap08-v0-c050`;
- same-index DualNAF residual `alpha=0.125`;
- frozen RGB seam offsets и bounded luminance gains.

Проверялись ровно четыре заранее заданных arms:

| Arm | Layout | Tile pixels | Tail | Роль |
|---|---|---|---|---|
| A | bilateral buddies96 | original, alpha=0 | RGB+luma -> h20 x1 | primary control |
| B | bilateral buddies96 | original, alpha=0 | RGB+luma -> h28 x1 | strong legal control |
| C | fixed cap08 fusion -> buddies96 | original, alpha=0 | RGB+luma -> h28 x1 | layout attribution |
| D | fixed cap08 fusion -> buddies96 | same-index DualNAF alpha=.125 | RGB+luma -> h28 x1 | only promotable candidate |

Никаких alpha, h, confidence, edge-count или blend sweep не было. D использует
`round(.875 * original_tile + .125 * rendered_same_index_tile)` внутри каждого
upright 20x20 tile. Cross-board pixels, templates, rotations, warp и target
inference отсутствуют.

## Preregistration и data boundary

Authoritative config:
`configs/ultimate_stack_preregistered_v1.json`, SHA-256
`4857fe1e67be9c56cad06f0eb651215250e5bbd8e6e80c91a6d095a7cdd1de63`.
До target decode он зафиксировал checkpoints, source/config hashes, arms,
selector, gates и confirmation policy.

Primary — shared-selector calibration `420:444`, 24 records:

- newline filename digest
  `e36b5628855d547c821761f3a0db2e700ac0fc8e52ac9c65102daeb3463d6dc3`;
- input-roster digest
  `76ffc2d09e6dbc4c54615847925e81504f229a68a666051c14f97db8dbeceb08`.

Conditional confirmation — disjoint `444:468`, digest
`14f422d1c25fada1928867e4b975e2838edf4680a8f23be00bb28e66436ba84b`.

Обе панели historically exposed: legacy calibration-700 уже декодировал все
700 calibration targets. Exact primary panel ранее не открывался этим stack-run,
но результат остаётся reused-calibration evidence, не untouched holdout и не
оценка generalization.

## Freeze-before-target и compliance

До чтения primary target bytes были сохранены все 96 PNG, обе score-matrix
пары, обе layouts, exact raw hashes, audits, target-free safety metrics и runtime
provenance. Commitment:

- file SHA-256
  `3d810b5cd5ed2e7355307eda491ff73afb7c9a8fee1380f3150b0b0ee2bd2ddd`;
- canonical self-hash
  `9f75677e2ffd42fb526c27561c875d564d89ddb9d48eff34acb7c95c8cabc769`;
- permissions `0444`;
- 48/48 strict raw permutation audits PASS;
- 96/96 frozen prediction PNG verified before first target decode.

Каждая raw assembly использует все 576 original upright tiles ровно один раз.
DualNAF работает one-to-one до сборки; raw audit выполняется до любого renderer,
harmonizer или NLM.

## Frozen primary gate

Для promotion D обязан был одновременно выполнить:

1. mean final SSIM `>=0.27`;
2. paired CI lower `>0` и не меньше 18/24 wins против A;
3. paired CI lower `>0` и не меньше 15/24 wins против B;
4. fused adjacency CI lower `>=0` и mean translation placement delta `>=0`;
5. D/A mean/min within-tile gradient retention `>=.80/.70`;
6. D/A mean/min Laplacian retention `>=.72/.60`;
7. D/A mean/max relative grid ratio `<=1.05/1.12`;
8. severe new manual artifacts `=0`.

Bootstrap — paired percentile, 20 000 replicates, fixed seeds beginning at
`20260920`.

## Primary result

| Arm | Mean final SSIM | Delta vs A | Delta vs B |
|---|---:|---:|---:|
| A bilateral/h20 | 0.241007 | — | -0.011474 |
| **B bilateral/h28** | **0.252481** | **+0.011474** | — |
| C fused/h28 | 0.249653 | +0.008646 | -0.002828 |
| D fused/DualNAF.125/h28 | 0.250107 | +0.009100 | **-0.002374** |

Paired results:

- B vs A: `+0.011474`, CI95 `[+0.010483,+0.012575]`, 24/24 wins;
- D vs A: `+0.009100`, CI95 `[+0.004924,+0.012944]`, 20/24 wins;
- D vs B: `-0.002374`, CI95 `[-0.006704,+0.001661]`, 10/24 wins;
- C vs B: `-0.002828`, CI95 `[-0.007014,+0.001005]`, 8/3/13;
- D vs C: `+0.000454`, CI95 `[-0.000165,+0.001239]`, 11/24 wins.

Таким образом, h28 снова дал устойчивый legal tail gain. На этой панели fusion
ухудшил B больше, чем DualNAF смог вернуть; DualNAF contribution остался мал и
статистически неопределён.

Geometry:

| Metric | Bilateral | Fused | Delta / gate evidence |
|---|---:|---:|---:|
| adjacency | 0.036345 | 0.037553 | +0.001208, CI lower **-0.000491 FAIL** |
| translation-aligned placement | 0.009187 | 0.009115 | **-0.000072 FAIL** |
| direct placement | 0.002459 | 0.001519 | -0.000940 |

D/A safety полностью прошла:

- gradient mean/min `0.8623/0.8257`;
- Laplacian mean/min `0.7221/0.6144`;
- relative grid ratio mean/max `0.7756/0.9512`.

Manual review всех четырёх sheets: D не добавляет severe hallucination,
clipping, geometric distortion или material local failure относительно A/B/C;
count `0`. Но все 24 boards остаются очевидными мозаиками, лица и цельные сцены
не восстанавливаются. Это relative safety PASS, не puzzle-quality PASS.

Из 13 quantitative conditions провалены пять: absolute D, D-vs-B CI, D-vs-B
wins, adjacency CI и translation placement. Поэтому общий gate FAIL независимо
от manual result.

## Artifacts и команды

```bash
uv run python scripts/run_ultimate_stack.py freeze --stage primary --device mps --run
uv run python scripts/run_ultimate_stack.py score --stage primary --run
uv run python scripts/run_ultimate_stack.py record-manual --stage primary \
  --severe-artifacts 0 --review-note '<review>' --run
```

Authoritative files:

- primary report SHA-256
  `34eed0eb8a5a442bc3697a0606ef56cab97234d0d07ed7ebad1d8537f5ddd37d`;
- manual review SHA-256
  `5babfb2cb70ff755f58d9996f359f9fca4edc5d749ecac476951542bba5ca745`;
- runner SHA-256
  `c07f4edf1349c3e853efbb835e2e20ed1f90396fdf728ec12b2e002cdba6f0e6`;
- implementation SHA-256
  `8fd4d4f0639e9972bdb963f5b16a1404bb84d86bae17651685e844c96cb933d1`.

Confirmation freeze проверен fail-closed: runner завершился до создания
confirmation directory с `primary quantitative gate failed; confirmation
forbidden`. Holdout, competition test и production не затрагивались.

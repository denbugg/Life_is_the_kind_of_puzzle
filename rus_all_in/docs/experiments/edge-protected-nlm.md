# Edge- and grid-protected flat-region NLM

## Решение

**Reject for promotion; production не менять.** Заранее зафиксированный
`E_flat_h40_t40` прошёл primary numeric и manual gates на historically exposed
reused calibration `120:144`: mean RGB SSIM `0.274239`, gain к `h20`
`+0.004040`, paired 95% t CI `[+0.003437,+0.004643]`, `24/24` wins и `0`
новых severe manual artifacts. Но неизменённый E на disjoint confirmation
`144:168` получил только `0.249786<0.27`. Это был единственный failed
confirmation check; положительный relative gain и все structure/grid bounds
сохранились. По frozen gate experiment окончательно не promoted.

Обе панели исторически target-exposed через
`outputs/legacy-upgrade/calibration700-champion/report.json`. Это не fresh
calibration, не holdout и не независимая generalization estimate. Primary и
confirmation не пересекаются только между собой.

## Проверенная гипотеза и legal boundary

После общего bilateral `solve_buddies(max_edges=96)` strict layout и frozen
RGB-offset -> bounded-luminance harmonizer независимо вычислялись proper colored
NLM single-pass outputs. `h20` служил safe source, а `h35`/`h40` — aggressive
source только для target-blind spatial blend:

1. Sobel magnitude считался на single-pass `h20` RGB image;
2. к edge mask добавлялись все строки и столбцы у границ исходных 20-pixel
   fragments;
3. mask расширялся `3x3` dilation и смягчался Gaussian `sigma=1`;
4. protected pixels брались из `h20`, flat pixels — из независимого aggressive
   single pass;
5. geometry, layout и полный набор 576 upright source tiles не менялись.

Таким образом, ни один candidate не был global `h>=30` output и ни один pixel
не проходил NLM последовательно дважды. Filename, target, cross-board pixels,
routing, tile substitution, rotation и warp не использовались. Эксперимент не
реабилитирует ранее отклонённые global `h>=30` или multi-pass regimes.

Frozen arms:

| Arm | Output | Роль |
|---|---|---|
| A | global `h20 x1` | control |
| B | global `h28 x1` | diagnostic, не может победить |
| C | protected `h20` + flat `h35`, Sobel threshold 30 | candidate |
| D | protected `h20` + flat `h40`, threshold 30 | candidate |
| E | protected `h20` + flat `h40`, threshold 40 | candidate |

## Preregistration и prediction commitments

Immutable config:
`configs/edge_protected_nlm_reused_calibration_preregistered_v1.json`, SHA-256
`63713f0da78940daae3738626cbb33e6d76f9b35daf1f6689842b5e40956c62b`.
Shared selector — namespace `aiijc-puzzle-experiments-v1`, seed `20260829`:

- primary ranked `120:144`, filename digest
  `6d737b42cabaa4fd97d3e595652450482b623c05850d20df3eb8d39ea1ab6db2`,
  filename+input digest
  `d21d838b35967df4792ac5529172f99ee6279a2dde3c505d1e15e95a89673f30`;
- confirmation ranked `144:168`, filename digest
  `3159c48f5cf8c8dc6fe54f0a43dc323e478e3f682f4333121aad1b94ffaaa691`,
  filename+input digest
  `d9cc1449115a0f0c8b44afe9a2db2eb96b7e3db6ad6a257094be16a0f5da329d`;
- exact primary/confirmation filename overlap: `0`.

До target decode для каждого board были записаны read-only NPZ с dirty input,
layout, raw assembly, harmonized canvas, всеми независимыми NLM images, final
predictions, dilated masks и soft masks. Commitment связывает точный roster,
каждый array digest, artifact-file digest и hashes всего scoring/inference
source. Все strict-permutation audits прошли.

| Panel | Commitment file SHA-256 | Commitment payload SHA-256 | Target receipt SHA-256 |
|---|---|---|---|
| Primary | `fbce0eee0d27c39e4fe50df91556f9b896add1d1836fddd76037bd4a0a798b76` | `a93a0377ea7ec00bada160c4bbea18dbcab5af71b597599efb706c59c1b305f7` | `9f39c385a63153f654c7cb5e835e1e4e03cbcbf5ef3d348d565e9c3889b8f77d` |
| Confirmation | `de1fce886384836c304f9362c4b25ccf47495ab908e1b58f60e5e5e563ef894f` | `4465d7f4e40bc260c53853e18d01736bf119864d711ce0247067afe150e341c8` | `b0037d65677e9a36aed42f07ba113530e6dce507a30e737295d8c77e021a4469` |

## Frozen gate

Promotion требовал одновременно mean RGB SSIM `>=0.27`, paired t CI lower
`>0` против A, не менее `18/24` wins, distinct output на каждом board, bounded
clipping, protected fraction и target-free safety:

- luma-gradient retention mean/min `>=0.90/0.80`;
- chroma-gradient retention mean/min `>=0.80/0.65`;
- luminance-Laplacian retention mean/min `>=0.90/0.80`;
- relative grid ratio mean/max `<=1.08/1.15`;
- protected-pixel fraction mean `[0.40,0.75]`, every board `[0.30,0.85]`;
- manual severe artifacts `0`.

Primary winner выбирался только среди C/D/E по максимальному mean; exact ties
разрешались меньшим aggressive h, затем threshold и arm name. Confirmation
разрешалась только для неизменённого primary winner после explicit root manual
PASS.

## Primary result: numeric + manual PASS

| Arm | Mean RGB SSIM | Gain vs A | Paired 95% t CI | Wins vs A | Numeric gate |
|---|---:|---:|---:|---:|---|
| A h20 | 0.270200 | — | — | — | control |
| B h28 | **0.281557** | +0.011358 | diagnostic only | 24/24 | ineligible |
| C h35/t30 | 0.273117 | +0.002917 | `[+0.002502,+0.003332]` | 24/24 | PASS |
| D h40/t30 | 0.273081 | +0.002882 | `[+0.002389,+0.003374]` | 24/24 | PASS |
| **E h40/t40** | **0.274239** | **+0.004040** | **`[+0.003437,+0.004643]`** | **24/24** | **PASS / selected** |

E safety mean/min: luma gradient `0.991916/0.967361`, chroma gradient
`1.013171/0.995411`, Laplacian `1.023060/1.001835`; relative grid mean/max
`1.003586/1.028635`; protected fraction mean/min/max
`0.532125/0.449397/0.629063`; clipped increase `0`.

Root independently inspected all 24 full canvases and fixed board-index
`0,7,15,23` center zooms. Manual verdict: PASS, severe new artifacts `0`.
No new smooth blob crossed protected content/tile edges, no identifiable
text/face/object edge was materially lost relative to A, no mask halo appeared,
and fragment-boundary integrity was not visibly worse. Existing wrong-layout
mosaics are pre-existing layout failures, not denoise artifacts. Bound read-only
manual review SHA-256:
`5053ecb2f081fedb1666ff9277282a1385fb695a9a685b4a813a011160351ec2`.

## Confirmation result: FAIL

| Arm | Mean RGB SSIM |
|---|---:|
| A h20 | 0.245129 |
| B h28 diagnostic | 0.256969 |
| **E h40/t40 unchanged** | **0.249786** |

E improved A on `24/24`: mean gain `+0.004657`, paired 95% t CI
`[+0.003993,+0.005322]`. Every relative and safety check passed:
luma gradient mean/min `0.976528/0.921685`, chroma
`1.002698/0.971982`, Laplacian `1.018524/0.981897`, grid mean/max
`1.019339/1.079146`, protected fraction mean/min/max
`0.532636/0.418064/0.623442`, clipped increase `0`. Но absolute check
`0.249786>=0.27` был false, поэтому `provisional_winner=null`, manual
confirmation PASS не создавался и production запрещён.

На обеих панелях global h28 diagnostic заметно лучше E. Edge protection
сохраняет ручную структуру и даёт стабильный lift к h20, но возвращение h20 у
маски отбрасывает значительную часть metric benefit h28 и не решает layout.
Повторять те же `h35/h40, threshold 30/40` arms без нового механизма не нужно.

## Artifacts и QA

- Primary report:
  `outputs/edge-protected-nlm/primary-calibration-offset120-count24/report.json`,
  SHA-256
  `eb74df7c97cf37b7fc8b8913fa2ad0265eea48b2009a5d5a4702f42604dd0d12`.
- Primary full/zoom sheets SHA-256:
  `b736e482d180b2e9024b7a913de8e500ede616fa6865bd12742e389dff51105a` /
  `f113713b89fe0a54a36bfb6bf845b2baf84f2aaf4b36b6db0370c961bc309442`.
- Confirmation report:
  `outputs/edge-protected-nlm/confirmation-calibration-offset144-count24/report.json`,
  SHA-256
  `23e9c3545b62b010fe083adcdba5508ef3dc3a70ed49ddca342d12245824ea23`.
- Confirmation full/zoom sheets SHA-256:
  `082332630e262b0e1acac46688c9ee0f4ffeff2756d04b291fa20eb2fc50d7de` /
  `7516e07d2e6fa41ce11e18af2f935df472727062e9f467506995830ec91f0feb`.

The archival runner deliberately separates target-free prepare from single-use
score:

```bash
uv run python scripts/run_edge_protected_nlm.py --mode primary --phase prepare
uv run python scripts/run_edge_protected_nlm.py --mode primary --phase score
uv run python scripts/run_edge_protected_nlm.py --mode confirmation --phase prepare
uv run python scripts/run_edge_protected_nlm.py --mode confirmation --phase score
```

Existing directories, commitments and receipts make all four completed actions
fail closed on rerun. Competition holdout, test and production ZIP were not
accessed or modified.

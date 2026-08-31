# Pretrained discriminative DRUNet как tile-preserving restoration tail

Статус: **strong relative component signal, reject for absolute gate**. Один
официальный pretrained DRUNet tail выиграл у `h20` и `h28` на всех 24 primary
boards с узкими положительными paired CI, но дал только `0.251061 < 0.27`.
Confirmation не открывалась, production не менялся.

## Что именно проверено

Это отдельная проверка discriminative denoiser-а после уже зафиксированной
раскладки, а не denoise-before-matcher и не повтор DualNAF:

1. bilateral-only score и ORBIT buddies96 задают строгую биекцию 576 tiles;
2. raw canvas проходит permutation audit;
3. frozen RGB seam offsets и bounded luma gains применяются к тем же tiles;
4. официальный colour DRUNet получает каждый upright `20x20` tile независимо;
5. tile отражается только из собственных пикселей справа/снизу на 4 px до
   `24x24`, затем output обрезается обратно ровно до `20x20`;
6. все 576 outputs возвращаются на те же layout indices без resize, warp,
   rotation, flip, generation, reference pixels или substitution;
7. после exact reassembly выполняется ровно один full-canvas colored NLM `h28`.

Neural model не видит соседний tile или другую board. Batch равен 144, в модели
нет BatchNorm, поэтому batching не является источником cross-tile context.

## Provenance и license

Использован официальный [KAIR](https://github.com/cszn/KAIR) commit
`fc1732f4a4514e42ce15e5b3a1e18c828af47a1e` и его официальный release
[`drunet_color.pth`](https://github.com/cszn/KAIR/releases/download/v1.0/drunet_color.pth).
KAIR опубликован под MIT license. DRUNet здесь является обычным
discriminatively trained Gaussian denoiser-ом, а не diffusion/GAN/generative
моделью.

Frozen hashes:

- checkpoint: `479abe3c5327dfd10ff54a80ec7d4098ca80752a5c9492cdff31cee430bec4b4`;
- MIT LICENSE: `448e69b705d64f21bf8cb86562301e0edd99ac79026064ddd75af8242b067be5`;
- official `network_unet.py`:
  `8043b6350f1589d5f08892e3be0b4d12c5a502058014285107b7360696d12bf5`;
- official `basicblock.py`:
  `48406db8867394ac5ae233ebeec7711ac10acfc3a6bbf0072c33aa77d659b6fd`.

Локальная минимальная реализация загрузила все 64 tensors строго и дала
bit-exact CPU output относительно vendored official implementation на случайном
probe (`max_abs=0`). Parameter count равен `32,640,960`. Submission может
содержать только итоговые PNG, не checkpoint или vendor code.

## Train-only выбор до calibration

До чтения любых новых calibration targets был выполнен фиксированный screen на
shared-ranked TRAIN records `512:528`: первые 8 только для выбора, последние 8
как disjoint train verification. Сравнивались одна architecture и заранее
заданные sigma/tail/blend композиции. Победитель первых 8 был заморожен как
`DRUNet sigma=40 -> one NLM h28`.

| Train panel | C mean | vs h20 | vs h28 |
|---|---:|---:|---:|
| selection 8 | 0.290882 | +0.015854, 8/8 | +0.005682, 8/8 |
| verification 8 | 0.258175 | +0.017997, 8/8 | +0.006353, 8/8 |

Train report SHA-256:
`7c64c96661c82e0a684a89e696763efcb35c818c32e7ad0c668d9fdd0a4ce6b0`.
Полный 37-arm train-only runtime на MPS составил `63.69 s`; target-blind
primary freeze с тремя arms — `30.85 s` на 24 boards.

## Preregistration и historical exposure

До primary target decode зафиксированы ровно три arms:

- A: `NLM h20`;
- B: strong reference `NLM h28`;
- C: independent DRUNet `sigma=40`, exact assembly, один `NLM h28`.

Config:
`configs/pretrained_drunet_tile_tail_preregistered_v1.json`, SHA-256
`43f62344d9b2302323780a92edadc2b5122d2ab97d6830c9a900cda504df3fdb`.

Primary использовал reused calibration `384:408`; confirmation была заранее
назначена на disjoint `600:624`. Панели нельзя называть fresh: legacy
`calibration700-champion/report.json` уже содержал все calibration targets.
Кроме него primary имел один current-experiment overlap
(`img_005968.png`, compliant-atlas ablation); confirmation имела два
(`img_005816.png`, candidate-supply smoke; `img_006808.png`, novel-analog
diagnostic). Эти сведения записаны в config до score. Primary и confirmation
не пересекаются с активными reservations `0:120` и `192:240`.

## Target-blind commitment

Все 72 primary PNG были созданы до target decode. `24/24` raw permutation
audits прошли; A/B/C были distinct на каждой board. Commitment SHA-256:
`0cabee5cfc3ef58a38260a78a6a10ab388d86476603c0d89cc362ddfc3f036dc`;
uncompressed prediction-roster SHA-256:
`a4e19e3169b126dd9035dc4b279eaf1a31102bd1a6c39ec7aa557e5b461c34d7`.

Commitment связывает checkpoint, license, vendor sources, final local sources,
MPS runtime, `sigma=40`, batch 144, padding и crop. Поле
`targets_decoded_during_freeze` равно `false`.

## Primary result

| Arm | Mean SSIM | Delta C vs arm | Paired 95% CI | C wins |
|---|---:|---:|---:|---:|
| A NLM h20 | 0.234361 | **+0.016700** | `[+0.015197,+0.018266]` | **24/24** |
| B NLM h28 | 0.245407 | **+0.005654** | `[+0.005064,+0.006254]` | **24/24** |
| C DRUNet40 -> NLM28 | **0.251061** | — | — | — |

Relative gates убедительно прошли, но absolute gate `mean >= 0.27` провален.
Authoritative report SHA-256:
`f94746753f18a2b67fb8dcdf68164e9c72f3e2628ea59c9b2fefd854c28c0740`.

## Target-free safety

Все заранее заданные bounds прошли:

| Diagnostic C/B | Mean | Board min | Board max |
|---|---:|---:|---:|
| within-tile gradient ratio | 0.9089 | 0.8418 | 0.9303 |
| within-tile Laplacian ratio | 0.9063 | 0.8511 | 0.9296 |
| grid seam ratio | 0.9195 | 0.8693 | 0.9708 |
| global RGB std ratio | 0.9899 | 0.9799 | 0.9952 |

Maximum absolute global channel-mean shift был `1.198/255`, maximum board mean
absolute pixel change — `2.223/255`, clipping increase —
`1.45e-6`. Это доказывает bounded pixel behaviour, но не заменяет manual scene
review. По preregistration root manual gate требовался только перед confirmation;
quantitative FAIL остановил experiment раньше него.

## Решение

Confirmation `600:624`, holdout и competition test **не открывать** для этой
preregistration. Не повторять тот же sweep DRUNet sigma10/20/30/40, post-h28
sigma5/10/20 или те же fixed blends на том же buddies96 endpoint: train screen
уже измерил их, а лучший вариант прошёл чистую primary проверку.

Практический вывод: DRUNet40 -> NLM28 — сильный legal restoration
component (`+0.00565` к h28, 24/24), но он не исправляет layout и сам по себе не
достигает требуемого диапазона. Его можно рассматривать только в новом заранее
зафиксированном эксперименте поверх действительно более сильного независимого
layout-а; текущий result не разрешает менять frozen production или заявлять
submission с `0.27`.

## Воспроизведение

```bash
uv run python scripts/run_pretrained_tile_denoiser_train_screen.py \
  --run --device mps

uv run python scripts/run_pretrained_tile_denoiser_calibration.py \
  --phase freeze --stage primary --device mps --run

uv run python scripts/run_pretrained_tile_denoiser_calibration.py \
  --phase score --stage primary --device mps --run

uv run ruff check src/aiijc_puzzle/pretrained_tile_denoiser.py \
  scripts/run_pretrained_tile_denoiser_train_screen.py \
  scripts/run_pretrained_tile_denoiser_calibration.py \
  tests/test_pretrained_tile_denoiser.py

uv run pytest tests/test_pretrained_tile_denoiser.py
```

Freeze/score команды fail-closed и откажутся перезаписывать существующие
artifacts. Confirmation дополнительно требует quantitative primary PASS и
отдельный `manual-review.json` от reviewer `root`; такого файла нет.

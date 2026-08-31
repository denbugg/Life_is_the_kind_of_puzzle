# DRUNet-protected production v1

Статус: **workflow подготовлен, production не запускался и сейчас не
авторизован**. Primary прошла, но disjoint confirmation получила absolute mean
`0.262817<0.27` и провалила frozen gate. Поэтому confirmation manual PASS и
immutable production authorization отсутствуют. Старый
`outputs/compliant-submission/` и его h20x1 runtime не менялись.

Целевой release root:
`outputs/compliant-drunet-protected-submission-v1/`:

- `predictions/` — ровно 700 root-level RGB PNG `480x480`;
- `submission.zip` — deterministic root-only archive тех же 700 PNG;
- `compliance-attestation.json` — per-board layout/raw/restoration evidence;
- `independent-validation.json` — повторный полный MPS пересчёт всех 700 boards.

Attestation намеренно имеет ограниченный статус
`METHOD_COMPLIANT_LAYOUT_ACCURACY_UNPROVEN`. Он доказывает соответствующий input,
строгую биекцию upright fragments, raw reassembly и exact frozen restoration,
но не доказывает hidden ground-truth permutation или ручную приёмку.

## Exact legal pipeline

Для каждого test board независимо:

1. split соответствующего dirty RGB480 input на 576 upright `20x20` tiles;
2. bilateral directional scores и `solve_buddies(max_edges=96)`;
3. strict permutation `0..575`, raw reassembly и два pre-restoration audits:
   declared-layout byte equality и sorted tile-byte multiset equality;
4. frozen RGB seam offsets, затем bounded luminance gains;
5. official KAIR colour DRUNet `sigma=40` отдельно на каждом ordered tile:
   same-tile reflect pad right/bottom `4` до `24x24`, batch 144, exact crop
   top-left обратно до `20x20`;
6. neural tiles собираются на тех же layout positions;
7. из одного DRUNet canvas независимо вычисляются single-pass colored NLM
   `h/hColor=20`, `28`, `40`, template 7, search 21;
8. exact t40 mask строится только из DRUNet+h20: grayscale, Sobel magnitude
   `>=40`, защита всех 20px grid boundaries, `3x3` dilation, Gaussian sigma 1;
9. final = `rint(soft*h28 + (1-soft)*h40)`, uint8 clip.

Нет target/reference/template pixels, source lookup, filename routing,
cross-board/cross-tile neural context, substitution, duplicate tile use,
rotation/flip, resize, warp или geometry change. DRUNet — discriminative Gaussian
denoiser, не diffusion/GAN/generative renderer.

## Promotion evidence gate

Production требует read-only
`configs/compliant_drunet_protected_submission_v1.json`. Пока файла нет, dry run
возвращает `BLOCKED_AWAITING_IMMUTABLE_PROMOTION_AUTHORIZATION`, а `--run`
останавливается до чтения competition boards.

Authorization должен exact-hash bind:

- combined preregistration;
- primary prediction commitment, external receipt, report и root manual review;
- confirmation prediction commitment, external receipt, report и root manual
  review.

Loader повторно читает каждый JSON, требует read-only mode, проверяет SHA-256,
preregistration binding, stage/offset/count, `quantitative_pass=true`, frozen
winner `D_drunet_sigma40_protected_h28_h40_t40`, все quantitative checks и root
manual PASS на всех 120 full-canvas triplets с severe artifacts `0`. Report с
holdout/test access или другой pipeline отклоняется.

Известные primary hashes уже закреплены:

- preregistration
  `6e6db6d4becb22a5fb70a9ce20474c6350f7e55ff391d65d763699938958e5a8`;
- primary commitment
  `d50994d5102015cf806914a02820ed7f0611c507a72b801325d13242c560aba5`;
- primary external receipt
  `8ca791ee95f785d0803b8479a0259ab2184c75d7e3321b52dfdf1e9d5e76572e`;
- primary report
  `bc75cfc69d4ad7323b24a2cce1da52f592dbef9b6c1895442e0f0348b3b89b90`;
- primary root review
  `03d13139ca9d4607a69c8ee369302e672764c9a47b0f788167780a496a8a72c2`.

Полученный confirmation failure нельзя превратить в authorization: hashes можно
связать только с report, который сам декларирует полный quantitative PASS, и
отдельным root manual verdict. Placeholder, игнорирование absolute gate или
post-hoc pipeline change запрещены.

Фактические read-only confirmation artifacts сохранены только как отрицательное
evidence:

- commitment
  `cf4251ca271f6cc3540d71d0fafb245e89bd8f0ecbaa04124ce45a3a59a72101`;
- external receipt
  `2f491f089f459651ec96782b574f2f6696013704b6a015da5c8076e54d99be0f`;
- report
  `0d1afd6be942f52cbea2f72297bb433414641b63aab413f3d7010af212f64385`.

Report фиксирует `quantitative_pass=false`, `selected_passing_winner=null` и
единственный failed check `D_mean_ssim_at_least_0_27`. Confirmation manual review
не создавался, что является правильным fail-closed состоянием.

## Model provenance и canonical bytes

- official repository: `https://github.com/cszn/KAIR`;
- commit: `fc1732f4a4514e42ce15e5b3a1e18c828af47a1e`;
- license: MIT, local license SHA
  `448e69b705d64f21bf8cb86562301e0edd99ac79026064ddd75af8242b067be5`;
- checkpoint URL:
  `https://github.com/cszn/KAIR/releases/download/v1.0/drunet_color.pth`;
- checkpoint SHA
  `479abe3c5327dfd10ff54a80ec7d4098ca80752a5c9492cdff31cee430bec4b4`;
- strict parameter count: `32,640,960`.

Canonical production и validator требуют Apple MPS. CUDA/CPU могут отличаться
на 1 LSB из-за backend floating-point kernels/rounding и поэтому являются только
noncanonical numerical reproduction; они не могут заменять hash-bound MPS PNG.

## Independent validation

Validator не вызывает production prediction/blend wrapper. Для каждого из 700
official inputs он отдельно:

- пересчитывает bilateral buddies96 layout и layout digest;
- заново собирает raw и проверяет exact tile multiset;
- пересчитывает оба harmonizers;
- загружает новый экземпляр official checkpoint и независимо реализует
  per-tile reflect-pad/crop inference;
- пересчитывает h20/h28/h40, binary/soft t40 mask и final blend;
- сверяет hashes каждого intermediate, decoded RGB480 output и PNG bytes;
- проверяет exact ordered root-only ZIP roster.

Проверка не использует кеш между boards: report обязан иметь
`boards_fully_recomputed=700`.

Schema:
`configs/compliant-drunet-protected-submission-v3.schema.json`, SHA-256
`a5581b56604671cee44747ede095a2666f2375ebd74962b8ffaa5484fcc5bf69`.
Она отдельна от исторической h20x1 schema и запрещает изменение sigma, h roster,
threshold, pad/crop/batch, provenance booleans и limited method status.

## Команды

До root promotion PASS безопасен только dry run:

```bash
uv run python scripts/run_compliant_drunet_protected_submission.py
```

После создания и проверки immutable authorization root может явно разрешить:

```bash
uv run python scripts/run_compliant_drunet_protected_submission.py --run
```

Production сам выполняет отдельный полный prepublish validation pass перед
публикацией. Повторный внешний запуск:

```bash
uv run python scripts/validate_compliant_drunet_protected_submission.py
```

Ни один CLI не предоставляет knobs для device, sigma, h, threshold, mask,
layout, checkpoint, target или routing. Existing outputs и все target/source
paths защищены от overlap/overwrite.

## Colab

`notebooks/reproduce_compliant_drunet_protected_colab.ipynb`:

- проверяет exact official `test.zip` и filename roster;
- скачивает checkpoint только с official KAIR release URL и проверяет SHA;
- запускает ровно fixed pipeline на CUDA;
- создаёт отдельный `NONCANONICAL-CUDA-MANIFEST.json` и root-only 700 PNG ZIP;
- повторяет 1-LSB disclosure до и после inference.

Notebook не создаёт canonical attestation и не называет CUDA bytes submission
authority.

## QA до production

Focused tests проверяют promotion tamper/fail-closed semantics, strict schema,
отсутствие public tuning knobs, raw bijection/multiset, independent tile
pad/crop, exact mask math, 700-record full-recompute orchestration, intermediate
tampering, dry-run no-write и отделение от старого fallback.

```bash
uv run ruff check src/aiijc_puzzle/compliant_drunet_protected_submission.py \
  src/aiijc_puzzle/compliant_drunet_protected_validation.py \
  scripts/run_compliant_drunet_protected_submission.py \
  scripts/validate_compliant_drunet_protected_submission.py \
  tests/test_compliant_drunet_protected_submission.py

uv run pytest -q tests/test_compliant_drunet_protected_submission.py
```

До confirmation/root PASS не запускать `--run`, validator или competition-test
inference. Старый `outputs/compliant-submission/` остаётся самостоятельным
immutable fallback release.

# DRUNet40 + protected h28/h40 NLM

Статус: **stable legal relative gain; reject for confirmation absolute gate**.
Единственная заранее зафиксированная композиция достигла `0.271644` на primary
reused-calibration-120, прошла root manual review, а затем без изменений
повторила relative gains на disjoint confirmation-120. Но confirmation mean
равен лишь `0.262817 < 0.27`, поэтому full gate провален, promotion и production
запрещены.

## Legal dataflow

Для каждой board используется только её собственный input:

1. bilateral dirty-tile scores и `solve_buddies(max_edges=96)` задают одну
   строгую перестановку всех 576 upright `20x20` fragments;
2. exact raw assembly проходит permutation audit до restoration;
3. frozen RGB seam offsets и bounded luma gains применяются к тем же tiles;
4. официальный colour DRUNet `sigma=40` независимо обрабатывает каждый tile:
   same-tile reflection pad `20x20 -> 24x24`, batch 144, exact crop обратно до
   `20x20`, без контекста соседнего tile;
5. neural tiles собираются в прежние layout indices без geometry change;
6. из одного DRUNet canvas независимо вычисляются colored NLM `h20`, `h28` и
   `h40`;
7. exact v1 `t40` Sobel+grid mask строится только из DRUNet+h20; итог D берёт
   DRUNet+h28 в protected regions и DRUNet+h40 в flat regions.

Нет resize, warp, rotation, flip, reference/template pixels, cross-board
pixels, tile substitution или generation. DRUNet — discriminatively trained
Gaussian denoiser, не diffusion/GAN/generative model.

## Fixed arms и train-only sanity

- A: original harmonized canvas -> one NLM h20;
- B: original harmonized canvas -> one NLM h28;
- C: tilewise DRUNet40 -> exact assembly -> one NLM h28;
- D: C canvas -> independent h20/h28/h40 -> exact t40 protected blend.

Перед calibration exact composition была sanity-checked без изменения
параметров на ранее использованных train `512:528`: B `0.268511`, C `0.274528`,
D `0.276525`; D выиграл у B и C на `16/16`. Train report SHA-256:
`9833bedcdcf8e317e0724aa09dbef34c268e0e89adde4b60a90bf610236e9ad9`.

## Preregistration, exposure и integrity

Primary заранее назначен на shared-ranked calibration `264:384`, confirmation
на disjoint `408:528`. Панели нельзя называть fresh: legacy calibration-700
уже исторически открывал все calibration targets. Полный basename-overlap
ledger сохранён отдельно; primary и confirmation не пересекаются друг с
другом.

- preregistration SHA-256:
  `6e6db6d4becb22a5fb70a9ce20474c6350f7e55ff391d65d763699938958e5a8`;
- overlap ledger SHA-256:
  `c2a793a2155cd7ffadc7b7ddf93141756683239dc864f6b55f736ab7c5535a71`;
- primary filename digest:
  `11d0dc7ce4d7a797a93b81c13766ad8990b6e60d84f101a80b94f05016c5ab94`;
- primary input-roster digest:
  `0f8ff23503dcbb8a2414e28ced46e5eb2179ad7bacb27a38d94295f66df2be05`.

Config, sidecar и ledger были сделаны read-only до clean freeze. Все 480 PNG
созданы через exclusive create и сделаны read-only. После 120/120 boards были
записаны immutable commitment и внешний receipt; только затем открыты primary
targets.

- commitment SHA-256:
  `d50994d5102015cf806914a02820ed7f0611c507a72b801325d13242c560aba5`;
- external receipt SHA-256:
  `8ca791ee95f785d0803b8479a0259ab2184c75d7e3321b52dfdf1e9d5e76572e`;
- prediction-roster SHA-256:
  `04764f144f80bb22eac8cefaedb708f171c6b61d1e2aa030169197ba2ff606f6`;
- bound runner SHA-256:
  `43c27e52ba76b3a7588a9a0f874892bea5d20546c9c0d62cafe977679adac2a1`.

All `120/120` raw permutation audits passed, all four arms were distinct on
every board, and `targets_decoded_during_freeze=false`. Clean primary MPS freeze
took `180.70 s`.

## Primary result

| Arm | Mean SSIM | D delta | Paired bootstrap 95% CI | D wins |
|---|---:|---:|---:|---:|
| A original h20 | 0.253079 | — | — | — |
| B original h28 | 0.264038 | **+0.007606** | `[+0.007237,+0.007967]` | **120/120** |
| C DRUNet40 -> h28 | 0.269736 | **+0.001908** | `[+0.001777,+0.002037]` | **119/120** |
| D protected stack | **0.271644** | — | — | — |

Все preregistered quantitative checks прошли, включая absolute
`mean >= 0.27`. Authoritative read-only report SHA-256:
`bc75cfc69d4ad7323b24a2cce1da52f592dbef9b6c1895442e0f0348b3b89b90`.

## Target-free safety

| Diagnostic D/B | Mean | Board min | Board max |
|---|---:|---:|---:|
| luma gradient retention | 0.9086 | 0.8424 | — |
| chroma gradient retention | 0.8260 | 0.6632 | — |
| luma Laplacian retention | 0.9156 | 0.8433 | — |
| grid ratio | 1.0191 | — | 1.1023 |
| protected fraction | 0.5120 | 0.3982 | 0.6583 |
| RGB std ratio | 0.9842 | 0.9654 | 0.9914 |

Maximum absolute RGB channel-mean shift равен `1.203/255`, maximum board mean
absolute pixel change — `2.661/255`, clipping increase — `0`. Все 16 strict
safety checks прошли.

## Manual gate

60 read-only sheets покрывают все 120 triplets B/C/D:
`outputs/pretrained-drunet-protected-stack/primary-calibration-offset264-count120/root-manual-review-sheets/`.
Sheet-roster SHA-256:
`10aea21892236982bb9614d082be4ddd3cfe83419b2fda670b3752476cceb9b6`.

Root просмотрел все 60 sheets / 120 full-canvas triplets и зафиксировал PASS:
`severe_artifacts=0`, `material_face_text_or_object_loss=false`,
`mask_halo_or_boundary_damage=false`. Read-only review SHA-256:
`03d13139ca9d4607a69c8ee369302e672764c9a47b0f788167780a496a8a72c2`.
Review связывает preregistration, primary report, commitment, external receipt,
prediction roster и sheet roster; runtime fail-closed authorization прошла.

## Unchanged confirmation result

После manual PASS неизменённый D был сначала target-blind frozen на disjoint
calibration `408:528` на canonical MPS. Все 480 PNG read-only, все `120/120` raw
permutation audits прошли, четыре arms distinct на каждой board, пересечение с
primary roster равно нулю. Confirmation freeze занял `172.12 s`.

- commitment SHA-256:
  `cf4251ca271f6cc3540d71d0fafb245e89bd8f0ecbaa04124ce45a3a59a72101`;
- external receipt SHA-256:
  `2f491f089f459651ec96782b574f2f6696013704b6a015da5c8076e54d99be0f`;
- prediction-roster SHA-256:
  `3cd6436cb1f9d0cccd868ecfa6e53959664a6d540cf547fd976b7321adc36cec`.

| Arm | Mean SSIM | D delta | Paired bootstrap 95% CI | D wins |
|---|---:|---:|---:|---:|
| A original h20 | 0.244001 | — | — | — |
| B original h28 | 0.254928 | **+0.007888** | `[+0.007532,+0.008243]` | **120/120** |
| C DRUNet40 -> h28 | 0.260812 | **+0.002005** | `[+0.001894,+0.002116]` | **120/120** |
| D protected stack | **0.262817** | — | — | — |

Все relative gates и все 16 target-free safety bounds снова прошли. В частности,
luma retention `0.9112/min 0.8239`, chroma `0.8246/min 0.6315`, Laplacian
`0.9173/min 0.8430`, grid ratio `1.0145/max 1.0841`, clipping increase `0`.
Единственный FAIL — absolute `D mean >= 0.27`. Поэтому selected winner равен
`null`, confirmation manual sheets не создавались, production не менялся.
Authoritative report SHA-256:
`0d1afd6be942f52cbea2f72297bb433414641b63aab413f3d7010af212f64385`.

Только как post-hoc descriptive summary, без изменения gate: на объединённых
240 boards D mean равен `0.267230`; gain к B `+0.007747`, CI
`[+0.007487,+0.008006]`, 240/240, а gain к C `+0.001956`, CI
`[+0.001870,+0.002043]`, 239/240. Это подтверждает стабильность bounded tail,
но не отменяет preregistered confirmation FAIL.

## Model provenance и устройства

Использован официальный KAIR commit
`fc1732f4a4514e42ce15e5b3a1e18c828af47a1e` под MIT license. Checkpoint
[`drunet_color.pth`](https://github.com/cszn/KAIR/releases/download/v1.0/drunet_color.pth)
имеет SHA-256
`479abe3c5327dfd10ff54a80ec7d4098ca80752a5c9492cdff31cee430bec4b4`;
parameter count `32,640,960`, strict state-dict load обязателен.

Canonical frozen bytes preregistered на MPS. CPU/CUDA поддерживаются моделью и
полезны для noncanonical воспроизведения, но могут отличаться на 1 LSB из-за
backend rounding и не должны заменять hash-bound predictions. В Colab сначала
нужно скачать официальный checkpoint, проверить приведённый SHA-256, затем
использовать `cuda` только как numerical reproduction; authoritative результат
остаётся сохранённым MPS roster.

## Проверки

```bash
uv run ruff check src/aiijc_puzzle/pretrained_drunet_protected_stack.py \
  scripts/run_pretrained_drunet_protected_stack.py \
  tests/test_pretrained_drunet_protected_stack.py

uv run pytest -q tests/test_pretrained_drunet_protected_stack.py
```

Не менять bound source после freeze: score проверяет source hashes. Не
повторять sweep `sigma/h/threshold/mask`; exact composition уже убедительно
подтвердила relative gain на 240/240 boards против h28 и 239/240 против C, но
не удержала absolute threshold на disjoint confirmation. Production и прежний
frozen C experiment не изменены.

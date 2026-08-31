# DRUNet-protected D stack: full calibration-700 measurement

Статус: **broad target range not achieved; holdout fail-closed**.

Уже замороженный legal candidate D был без нового выбора параметров измерен на
всех 700 records manifest calibration. Exact mean RGB SSIM равен
`0.268270469063251`: это ниже заранее зафиксированного диапазона `[0.27, 0.28]`
на `0.001729530936749`. Следовательно, результат отдельных primary-панелей выше
`0.27` не переносится на calibration split целиком. Holdout-700 не открывался.

## Что именно измерено

Candidate не менялся относительно исходной preregistration SHA-256
`6e6db6d4becb22a5fb70a9ce20474c6350f7e55ff391d65d763699938958e5a8`:

1. dirty-only bilateral edge scores;
2. `solve_buddies(max_edges=96)` и строгая перестановка 576 upright tiles;
3. raw permutation/multiset audit до restoration;
4. frozen RGB seam offsets и bounded luma gains;
5. official colour DRUNet `sigma=40` независимо на каждом `20x20` tile,
   reflection pad `+4` справа/снизу, exact crop и прежние layout indices;
6. независимые colored NLM `h20/h28/h40` из одного DRUNet canvas;
7. exact t40 mask из h20, protected source h28, flat source h40.

Target-scored output только один: D. Original-h28 сохранялся исключительно как
старый target-free safety reference, не как новый candidate arm. Нет sweep,
post-hoc выбора, target/reference/template pixels при inference, cross-board
pixels, resize, rotation, warp, substitution или generation.

## Freeze-before-score и историческая exposure

Calibration-700 нельзя называть fresh: все targets уже исторически открывались
в legacy calibration-700. Тем не менее именно этот запуск сначала создал все
700 target-blind predictions, повторно проверил каждый input hash, layout hash,
PNG hash и raw reconstruction, затем записал immutable commitment и receipt;
только после полной проверки был записан target-access receipt и декодированы
targets.

- immutable V2 config SHA-256:
  `ce8d8ff3c0d30c8da264a22813aa863f9b83416e709c61473523539363ff4458`;
- calibration filename roster SHA-256:
  `8c384af37afa3db09480feedd3d4bd7b7f4d2edbf132b29c9941551cdf68083d`;
- input roster SHA-256:
  `c0f47c07776d008ea0323cee889fccfd7ccde120b1fa79cfbc04cb6987201893`;
- prediction commitment SHA-256:
  `495d4f11d00d00fa0df577ec0acb183b4cfcaf4dca83a6e816f21a956715e88a`;
- commitment receipt SHA-256:
  `9e3ba5e902496d29f424cc52f6dde7d5211658bff4fb72ce1d9f567936caf58a`;
- candidate roster SHA-256:
  `084e5bb761a37854c3c3ab492aca7a651fc6da6652b1f94b2562ac0a95afca22`;
- target-access receipt SHA-256:
  `d6f58c8885a59ab0a475eefca494c8d775acf6bfe783835aa785970d1ef26347`;
- final report SHA-256:
  `2d590f0f7ba33de65ea8e4b8b40c75d56fe804a4f065297e87ed9d7ee4c85404`.

V1 протокол и source остаются отдельным immutable failure evidence. Он
остановился после первой target-blind board до commitment и до любого target
decode: tuple-поля audit после JSON round-trip стали lists и семантически
равные evidence-словари не прошли прямое Python equality. V2 исправил только
каноническое JSON-представление обеих сторон сравнения; pipeline и thresholds
не менялись. V2 использует новый output root.

## Exact calibration-700 результат

- mean: `0.268270469063251`;
- sample standard deviation: `0.081691808872841`;
- standard error: `0.003087660148979`;
- boards `SSIM >= 0.27`: `306/700`;
- boards `SSIM >= 0.28`: `267/700`.

| Quantile | SSIM |
|---:|---:|
| min | 0.076630 |
| 5% | 0.153819 |
| 10% | 0.173185 |
| 25% | 0.215265 |
| 50% | 0.260070 |
| 75% | 0.310184 |
| 90% | 0.365679 |
| 95% | 0.422901 |
| max | 0.651835 |

Заранее заданные contiguous folds в shared-ranked order:

| Fold | Ranked records | Mean SSIM |
|---:|---:|---:|
| 0 | 0:70 | 0.275814 |
| 1 | 70:140 | 0.286401 |
| 2 | 140:210 | 0.258602 |
| 3 | 210:280 | 0.256055 |
| 4 | 280:350 | 0.276703 |
| 5 | 350:420 | 0.262068 |
| 6 | 420:490 | 0.264632 |
| 7 | 490:560 | 0.265214 |
| 8 | 560:630 | 0.267385 |
| 9 | 630:700 | 0.269831 |

Только 3/10 folds выше `0.27`; диапазон fold means равен
`[0.256055, 0.286401]`. Это прямо показывает panel sensitivity прежних
primary/confirmation оценок.

## Provenance и target-free safety

Strict raw provenance прошёл `700/700`: каждая layout — bijection всех 576
indices, raw canvas точно пересобирается по declared layout, а input/output tile
multisets и raw pixels совпадают до restoration.

Из 15 frozen safety checks прошли 14. Единственный FAIL — minimum board chroma
gradient retention `0.5964254932 < 0.60` на `img_001643.png`. Остальные ключевые
значения:

| Diagnostic D / original-h28 | Mean | Min | Max |
|---|---:|---:|---:|
| luma gradient retention | 0.909362 | 0.816238 | — |
| chroma gradient retention | 0.825411 | 0.596425 | — |
| luma Laplacian retention | 0.916143 | 0.816657 | — |
| grid ratio | 1.017064 | — | 1.115147 |
| protected fraction | 0.515771 | 0.360551 | 0.739783 |
| RGB std ratio | 0.984060 | 0.965417 | 0.992830 |

Maximum absolute channel-mean shift `1.289744/255`, maximum board mean absolute
pixel change `2.987671/255`, clipping increase `0`.

## Frozen decision

До calibration score было зафиксировано: только если mean принадлежит closed
interval `[0.27,0.28]` **и** все provenance/safety checks проходят, разрешается
один unchanged holdout-700 prepare -> commitment -> score. Calibration провалил
и mean, и один safety bound. Поэтому broad gate `FAIL`, directory holdout700 не
создан, targets holdout не декодировались этим measurement, production и
submission outputs не менялись.

## Воспроизведение и проверки

```bash
PYTHONPATH=src .venv/bin/python \
  scripts/run_pretrained_drunet_protected_stack_all700_v2.py \
  --phase prepare --stage calibration --device mps --run

PYTHONPATH=src .venv/bin/python \
  scripts/run_pretrained_drunet_protected_stack_all700_v2.py \
  --phase score --stage calibration --run

.venv/bin/ruff check \
  scripts/run_pretrained_drunet_protected_stack_all700.py \
  scripts/run_pretrained_drunet_protected_stack_all700_v2.py \
  tests/test_pretrained_drunet_protected_stack_all700.py

PYTHONPATH=src .venv/bin/pytest -q \
  tests/test_pretrained_drunet_protected_stack_all700.py
```

Authoritative artifacts находятся в
`outputs/pretrained-drunet-protected-stack/all700-measurement-v2/calibration700/`.
Per-board records и обе PNG на каждой board read-only; полный output занимает
около `331 MB`.

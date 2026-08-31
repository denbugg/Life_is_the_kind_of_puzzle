# Fixed-layout pixel-tail bakeoff

Дата: 2026-08-29. Вердикт: **цветной OpenCV NLM `h=9` — уверенный
победитель этого frozen roster на calibration48 и independent holdout48;
E18b gray guard снижает contest SSIM и не нужен без отдельного
safety-требования.**

## Что проверялось

Цель эксперимента — отделить качество пиксельного postprocess от качества
решателя пазла. Использованы две непересекающиеся панели по 48 пар из frozen
manifest: `calibration` и `holdout`. Во всех экспериментальных направлениях
они выбираются общим `select_manifest_records`, namespace
`aiijc-puzzle-experiments-v1`, seed `20260829`. Hash manifest и каждой PNG
проверяется перед вычислением. `data/raw/test` и любые test-derived артефакты
явно исключены.

Roster и параметры были зафиксированы историческим E18 и первым exploratory
smoke до открытия manifest holdout. После calibration48 они не менялись;
holdout48 запущен ровно один раз тем же кодом.

Для каждой доски построены две target-assisted раскладки:

1. `lowres_hungarian`: нормализованные RGB block-mean descriptors 5×5 и
   one-to-one Hungarian — воспроизведение исторического train-permutation
   recovery;
2. `fullres_reference_hungarian`: нормализованные RGB pixels 20×20 и тот же
   Hungarian — более сильный, но более чувствительный к шуму независимый
   reference.

Обе раскладки являются oracle diagnostics: target используется при assignment,
поэтому ни одна не deployable на test. На каждой неизменной раскладке сравнены:

- raw assembly;
- `cv2.fastNlMeansDenoisingColored`, `h=hColor=3/5/7/9`, template 7, search 21;
- каждый colored-NLM с точным E18b no-new-gray-cell guard;
- gray-only NLM h=9 с репликацией канала в RGB;
- luminance-only NLM h=9 с сохранением Cr/Cb;
- bilateral 5×5, `sigma=15/25`;
- Gaussian `sigma=0.5/1.0`;
- non-deployable cellwise oracle, выбирающий raw или colored NLM h=9 по
  target-cell SSIM.

Метрика везде ровно contest RGB SSIM:
`structural_similarity(target, output, channel_axis=2, data_range=255)`.
CI ниже — paired Student 95% CI для gain относительно raw на тех же 48 досках.

Воспроизведение:

```bash
uv run python scripts/run_pixel_tail_bakeoff.py \
  --split calibration \
  --limit 48 \
  --output-dir outputs/pixel-tail-bakeoff/calibration48

uv run python scripts/run_pixel_tail_bakeoff.py \
  --split holdout \
  --limit 48 \
  --output-dir outputs/pixel-tail-bakeoff/holdout48
```

Полные per-board и aggregate JSON:

- `outputs/pixel-tail-bakeoff/calibration48/{per_board,summary}.json`;
- `outputs/pixel-tail-bakeoff/holdout48/{per_board,summary}.json`.

Старые `smoke8` и `run48` были выбраны random permutation всех 7 000 stems.
Поздний membership audit показал для `run48`: 35 train, 7 calibration и 6
holdout; smoke8 включал две holdout-доски. Они сохранены как exploratory
reproducibility artifacts, но полностью исключены из promotion headline.

## Calibration48 и holdout48: low-resolution Hungarian

| Variant | Calibration SSIM / gain | Holdout SSIM / gain | Holdout wins | CPU s/image |
|---|---:|---:|---:|---:|
| raw | 0.439085 / — | 0.430621 / — | — | 0 |
| colored NLM h3 | 0.465928 / +0.026843 | 0.456280 / +0.025659 | 47/48 | 0.1068 |
| colored NLM h5 | 0.508867 / +0.069783 | 0.495520 / +0.064898 | 48/48 | 0.1057 |
| colored NLM h7 | 0.546310 / +0.107225 | 0.531604 / +0.100982 | 48/48 | 0.1053 |
| **colored NLM h9** | **0.571616 / +0.132531** | **0.557442 / +0.126821** | **48/48** | **0.1051** |
| colored NLM h9 + gray guard | 0.567194 / +0.128109 | 0.553853 / +0.123231 | 48/48 | 0.1099 |
| gray-only NLM h9 | 0.555768 / +0.116683 | 0.544787 / +0.114166 | 48/48 | 0.0410 |
| luma-only NLM h9 | 0.529255 / +0.090170 | 0.519805 / +0.089184 | 48/48 | 0.0412 |
| bilateral sigma25 | 0.478578 / +0.039493 | 0.470089 / +0.039468 | 48/48 | 0.0004 |
| Gaussian sigma1.0 | 0.511811 / +0.072727 | 0.502785 / +0.072163 | 48/48 | 0.0002 |
| target-leaking h9 cell oracle | 0.573751 / +0.134666 | 0.560931 / +0.130309 | 48/48 | 0.2438 |

Paired Student 95% CI для gain colored NLM h9:

- calibration: `[+0.123724, +0.141338]`;
- holdout: `[+0.114821, +0.138820]`.

Calibration и holdout robust score
`mean - 0.5 × std(four interleaved folds)` тоже выбирает colored NLM h9:
`0.569665` и `0.550600` соответственно.

### Подробный independent holdout

| Variant | Mean RGB SSIM | Gain vs raw | Paired 95% CI gain | Wins | CPU s/image |
|---|---:|---:|---:|---:|---:|
| raw | 0.430621 | — | — | — | 0 |
| colored NLM h3 | 0.456280 | +0.025659 | [+0.019583, +0.031734] | 47/48 | 0.1068 |
| colored NLM h5 | 0.495520 | +0.064898 | [+0.054476, +0.075321] | 48/48 | 0.1057 |
| colored NLM h7 | 0.531604 | +0.100982 | [+0.089700, +0.112265] | 48/48 | 0.1053 |
| **colored NLM h9** | **0.557442** | **+0.126821** | **[+0.114821, +0.138820]** | **48/48** | **0.1051** |
| colored NLM h9 + gray guard | 0.553853 | +0.123231 | [+0.111528, +0.134934] | 48/48 | 0.1099 |
| gray-only NLM h9 | 0.544787 | +0.114166 | [+0.103789, +0.124542] | 48/48 | 0.0410 |
| luma-only NLM h9 | 0.519805 | +0.089184 | [+0.081767, +0.096601] | 48/48 | 0.0412 |
| bilateral sigma15 | 0.450861 | +0.020239 | [+0.018900, +0.021579] | 48/48 | 0.0004 |
| bilateral sigma25 | 0.470089 | +0.039468 | [+0.036874, +0.042062] | 48/48 | 0.0004 |
| Gaussian sigma0.5 | 0.457243 | +0.026621 | [+0.024743, +0.028500] | 48/48 | 0.0001 |
| Gaussian sigma1.0 | 0.502785 | +0.072163 | [+0.064498, +0.079828] | 48/48 | 0.0002 |
| target-leaking h9 cell oracle | 0.560931 | +0.130309 | [+0.118786, +0.141832] | 48/48 | 0.2438 |

Colored h3/h5/h7 gray-guarded значения также сохранены в JSON. Каждый guard
немного хуже соответствующего unguarded output; они опущены из таблицы для
компактности.

Exploratory gray-only победа на smoke-8 была маловыборочным эффектом и не
перенеслась ни на calibration, ни на holdout.

## Контроль ошибки раскладки

| Diagnostic | Calibration | Holdout |
|---|---:|---:|
| low/full assignment agreement | 81.27% | 80.02% |
| minimum per-board agreement | 54.86% | 57.29% |
| low-res raw SSIM | 0.439085 | 0.430621 |
| full-res raw SSIM | 0.446063 | 0.438270 |
| full minus low raw SSIM | +0.006978 | +0.007649 |
| low-res mutual-nearest fraction | 0.589120 | 0.572519 |
| full-res mutual-nearest fraction | 0.593533 | 0.577510 |

Две inferred permutations совпадают лишь примерно на 80% ячеек. Это
подтверждает, что matching-based layouts нельзя безусловно называть ground
truth: текстурно похожие клетки и degradation создают неоднозначность.

Главный вывод о tail от этого почти не меняется:

| Split/layout | Raw | Colored NLM h9 | Gain | Paired 95% CI |
|---|---:|---:|---:|---:|
| calibration low-res | 0.439085 | 0.571616 | +0.132531 | [+0.123724, +0.141338] |
| calibration full-res | 0.446063 | 0.580375 | +0.134312 | [+0.125287, +0.143337] |
| holdout low-res | 0.430621 | 0.557442 | +0.126821 | [+0.114821, +0.138820] |
| holdout full-res | 0.438270 | 0.566744 | +0.128474 | [+0.116617, +0.140331] |

То есть разница layout recovery меняет абсолютный SSIM, но не ranking tails и
на `0.0017–0.0018` меняет оценку gain h9.

## Решения по вариантам

### Promote: colored NLM h9

- лучший deployable mean и robust SSIM в обоих layout-треках;
- 48/48 побед относительно raw на calibration и holdout;
- нижняя граница paired 95% CI намного выше нуля;
- около `0.104 s/image` на CPU MacBook с OpenCV 5.0.0;
- monotonic improvement h3 → h5 → h7 → h9 внутри заранее заданного roster.

Последний пункт не доказывает, что h9 — глобальный optimum: h>9 не входил в
этот эксперимент. Он доказывает только победу h9 среди параметров,
зафиксированных в задаче и прежнем E18 sweep.

### Reject for contest metric: E18b gray guard

Guard откатил 727 newly-gray cells на calibration и 621 на holdout, сохранив
invariant «gray count не выше raw». При этом он проиграл unguarded h9 на 46
calibration-досках, выиграл на одной и совпал на одной; на independent holdout
проиграл **48/48**, в среднем на `−0.003589` SSIM. Исторический E18b PASS
относился к дополнительному safety gate, а не к максимуму leaderboard SSIM.
Если такого внешнего требования нет, использовать guard не следует.

### Latency fallbacks, не новые champions

- Gray-only h9 примерно в 2.56 раза быстрее colored h9, но на holdout в
  среднем хуже на `0.012655`; он выиграл 14/48 и проиграл 34/48, поэтому
  безопасного общего routing rule пока нет.
- Gaussian sigma1.0 почти бесплатный и даёт holdout gain `+0.072163`, 48/48,
  но заметно уступает NLM h9. Это разумный extreme-latency fallback.
- Bilateral sigma25 стабильно положителен, но holdout gain всего `+0.039468`.
- Luma-only NLM сохраняет цвет, однако существенно проигрывает и colored, и
  gray-only NLM. Повторять этот вариант не нужно.

### Oracle headroom мал

Target-cell chooser откатывает 2 831 из 27 648 клеток на calibration и 3 028
на holdout. Его holdout headroom относительно h9 — лишь `+0.003489`. Он
non-deployable и не является строгим upper bound:
локальный cell SSIM не учитывает окна SSIM, пересекающие границы клеток, поэтому
на 7/48 holdout canvas score даже слегка снизился. Практический вывод — сложный
raw-vs-NLM cell router имеет маленький доступный headroom относительно h9.

## Ограничения

- Это pixel-tail experiment, а не solver benchmark. Target-assisted layouts
  намеренно недоступны на test и не должны попадать в inference.
- Full-resolution reference — более сильная inferred assignment, но не
  опубликованная организаторами истинная permutation.
- Calibration/holdout membership, protocol digest, selection digest и image
  hashes записаны в JSON. Старые исследования не имели общего manifest,
  поэтому `holdout` означает independent split текущего workspace, а не
  гарантию, что эти stems никогда не встречались во всей старой git-истории.
- Gain измерен на почти правильных layout (`raw≈0.43–0.44`). На слабом solver
  абсолютный gain может быть другим, хотя исторический E18 показал тот же знак
  на плохих layouts.
- Runtime относится только к transform на локальном CPU; чтение PNG, layout
  recovery, SSIM и запись submission в него не входят. Полный диагностический
  run с двумя layouts и oracle chooser занял 89.9 s на calibration и 89.2 s
  на holdout.

## Проверки

```bash
uv run pytest tests/test_pixel_tails.py
uv run ruff check \
  src/aiijc_puzzle/pixel_tails.py \
  scripts/run_pixel_tail_bakeoff.py \
  tests/test_pixel_tails.py
```

Тесты покрывают split/assemble round trip, exact recovery синтетической
affine-corrupted permutation, descriptor validation, E18b guard, target-leaking
cell fallback, aggregate paired metrics и лёгкий публичный
`apply_nlm_color(image, h=9)` без запуска полного диагностического roster.

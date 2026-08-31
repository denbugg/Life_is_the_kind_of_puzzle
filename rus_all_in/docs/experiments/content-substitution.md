# Content substitution: строгая проверка M420

Статус: **завершён; content slack подтверждён при one-to-one ограничении, а clean
oracle отделён от realistic dirty-pixel proxy**.

Главный confirmatory результат на frozen holdout-48:

- clean-target Hungarian derangement: RGB SSIM `0.533053`;
- та же перестановка соответствующих recovered dirty input tiles: `0.258824`;
- тот же dirty canvas после frozen colored NLM h=9: **`0.383725`**.

Во всех трёх случаях точная индексная расстановка равна нулю, а Hungarian использует
каждый из 576 source tiles ровно один раз. Значение `0.533` доказывает геометрию
метрики, но не является ожидаемым submission score. Более реалистичная оценка
pixel-path — `0.384`, и даже она остаётся target-assisted oracle diagnostic, потому
что clean target участвует в выборе substitute и восстановлении train permutation.

## Проверяемый вопрос

Исторический M420 показал высокий SSIM при замене каждого фрагмента визуально похожим,
но выбирал substitute независимо по клеткам и мог многократно использовать один
source tile. Этот эксперимент отвечает на три вопроса:

1. сохраняется ли эффект при допустимой биекции всех 576 фрагментов;
2. сколько результата создаёт reuse;
3. сколько clean-target ceiling переживает переход к реальным искажённым input pixels
   и утверждённому postprocess.

## Три уровня рендера

### 1. `clean_oracle`

Стоимость `C[i,j]` — точная RMSE всех 1200 RGB-значений между clean target tiles `i`
и `j`. Source pixels тоже берутся из clean target. Это чистая проверка metric slack,
полностью недоступная на test.

Матрица 576×576 вычисляется через float64 Gram identity без пятиразмерного тензора
попарных разностей. Float64 нужен, чтобы не потерять малые расстояния при вычитании
норм порядка `1e8`.

### 2. `raw_dirty_proxy`

Substitute по-прежнему выбирается clean RMSE, но рендерится соответствующий corrupted
tile из `train/inputs`. Так как организаторы не публикуют permutation labels, shuffled
input предварительно сопоставляется target-позициям нормализованным 5×5 block
descriptor + one-to-one Hungarian. Это воспроизводит исторический train-pair recovery.

Alignment нельзя называть ground truth: на holdout средняя descriptor correlation
`0.81297`, а назначенный target является независимым row-wise best только для `75.28%`
input tiles. Поэтому absolute dirty score включает ошибку recovery.

### 3. `nlm_h9_dirty_proxy`

К каждому собранному raw-dirty canvas применяется публичная реализация победителя
pixel-tail bakeoff: `apply_nlm_color(..., h=9)`, OpenCV colored NLM, template 7,
search 21. Сам tail deployable и не видит target; выбор substitute и диагностическая
раскладка всё ещё target-assisted.

## Варианты назначения

- `identity`: контроль extraction, recovered alignment и pixel tail;
- `nearest_other`: ближайший другой tile независимо для каждой клетки, reuse разрешён;
- `bijective_derangement`: минимальная по суммарной clean RMSE перестановка через
  SciPy Hungarian, диагональ запрещена;
- `random_other`: независимый случайный другой tile;
- `random_k3_nearest`, `random_k10_nearest`: случайный выбор из 3/10 ближайших других.

Все неidentity-варианты запрещают диагональ. Случайные назначения воспроизводимы:
seed выводится SHA-256 от assignment seed `420`, имени доски и названия варианта.

## Frozen протокол

- Manifest: `data/interim/validation_manifest.json`, protocol digest
  `2a9e3b74f7defa8c00846a05eb598fd263fd16c2787c70e77d3b7a4b585bfbf4`.
- Общий selector: `protocol.select_manifest_records`, namespace
  `aiijc-puzzle-experiments-v1`, seed `20260829`.
- Development: calibration-48, selection digest
  `5b4ff9b7e14b8fbb3e6522a4398c912d477e5ec7c877ad8242e5f8c7c3b0e8eb`.
- Confirmatory: независимый holdout-48, selection digest
  `941f272377dad2aa3edb9092b89582bd4fa04f6db1ac80aa254b6edefb781e40`.
- Roster и параметры были зафиксированы до calibration-48 и не менялись перед
  holdout-48.
- Test-файлы, test-derived labels и исторические test refs не используются.
- Метрика: `structural_similarity(target, output, channel_axis=2, data_range=255)`.

## Confirmatory holdout-48

| Назначение | Clean oracle | Raw dirty | Dirty + NLM h9 | Reuse | Clean RMSE mean / q50 / q90 |
|---|---:|---:|---:|---:|---:|
| `identity` | 1.000000 | 0.430621 | 0.557442 | 0.0 | 0.00 / 0.00 / 0.00 |
| `nearest_other` | **0.572959** | 0.252676 | **0.384960** | 262.9 | 21.33 / 16.95 / 46.94 |
| `bijective_derangement` | **0.533053** | 0.258824 | **0.383725** | **0.0** | 24.57 / 19.59 / 54.26 |
| `random_k3_nearest` | 0.538275 | 0.235023 | 0.365310 | 279.1 | 23.11 / 18.94 / 50.17 |
| `random_k10_nearest` | 0.483721 | 0.208963 | 0.334979 | 281.6 | 26.25 / 22.19 / 55.12 |
| `random_other` | 0.148637 | 0.093911 | 0.153302 | 211.0 | 82.22 / 76.16 / 146.41 |

`exact_placement_fraction=0` у каждого неidentity-варианта. Clean nearest превышает
clean bijection на `0.03991` SSIM в среднем, paired SD `0.00797`; reuse немного
завышает clean ceiling, но не создаёт сам эффект. После dirty render + NLM разница
сжимается до `0.00123 ± 0.00999` и nearest выигрывает лишь 24 из 48 досок.

NLM повышает bijective dirty proxy с `0.258824` до `0.383725`: paired gain
**`+0.124901`**, 95% Student CI **`[+0.113457, +0.136345]`**, wins `48/48`.
Но clean oracle остаётся выше NLM proxy на `0.14933` в среднем. Только 24/48
NLM-bijective досок достигают SSIM 0.38; для clean ceiling это 45/48. Поэтому clean
числа нельзя переносить на submission.

Полезный sanity check: NLM identity равен `0.557442`, почти совпадая по масштабу с
историческим M420 true-layout shipping score `0.5599`, хотя выборки и детали recovery
различаются.

## Calibration → holdout

| Метрика | Calibration-48 | Holdout-48 | Δ holdout−calibration |
|---|---:|---:|---:|
| Clean bijective | 0.526729 | 0.533053 | +0.006325 |
| Raw-dirty bijective | 0.259284 | 0.258824 | −0.000460 |
| NLM-dirty bijective | 0.388504 | 0.383725 | −0.004779 |
| NLM-dirty nearest | 0.390555 | 0.384960 | −0.005595 |
| NLM-dirty random | 0.149747 | 0.153302 | +0.003555 |

Вывод устойчив: все ключевые movement меньше `0.0064` SSIM и ни один знак или ranking
решения не меняется.

## Решение

M420 выдерживает устранение duplicate-use confounder. Exact index placement,
adjacency, top-k recall и bond count являются более строгими целями, чем contest SSIM.
Практическое продолжение:

1. считать content-aware candidate recall и bond/placement ceilings рядом с index
   метриками;
2. сохранять one-to-one constraint в oracle и solver-анализе;
3. обучать selector/verifier на visually equivalent positives, но подтверждать выигрыш
   только по full-canvas RGB SSIM;
4. применять frozen colored NLM h9 после любого собранного dirty canvas.

Не следует делать вывод, что target-free solver уже может получить `0.384`: текущий
эксперимент не показывает, как находить визуальные substitutes без clean target. Он
показывает, что такой сигнал ценен и что допустимая биекция не уничтожает payoff.

## Воспроизведение

```bash
uv run python scripts/run_content_substitution.py \
  --targets-dir data/raw/train/targets \
  --inputs-dir data/raw/train/inputs \
  --manifest data/interim/validation_manifest.json \
  --split calibration \
  --output-dir outputs/content-substitution/calibration-48 \
  --count 48 --assignment-seed 420 --nearest-k 3 10 \
  --apply-nlm-h9 --no-save-images

uv run python scripts/run_content_substitution.py \
  --targets-dir data/raw/train/targets \
  --inputs-dir data/raw/train/inputs \
  --manifest data/interim/validation_manifest.json \
  --split holdout \
  --output-dir outputs/content-substitution/holdout-48 \
  --count 48 --assignment-seed 420 --nearest-k 3 10 \
  --apply-nlm-h9 --no-save-images
```

Calibration занял 44.88 s, holdout — 45.77 s на локальном arm64 CPU. Один colored
NLM h9 занимает около 0.105 s на canvas; весь run включает шесть вариантов на каждую
доску. В `outputs/content-substitution/{calibration-48,holdout-48}/` записаны:

- `per_board.json` — alignment diagnostics, SSIM, exact placement, reuse, RMSE
  quantiles, SHA-256 назначения и tail runtime;
- `aggregate.json` — manifest provenance, выбранные имена, версии окружения, runtime и
  все агрегаты.

В `outputs/content-substitution/manifest-smoke-8/images/` сохранены фактические PNG
clean, raw dirty и NLM dirty вариантов для визуального аудита.

Код: `src/aiijc_puzzle/content_substitution.py` и
`scripts/run_content_substitution.py`. Проверки: `tests/test_content_substitution.py`.

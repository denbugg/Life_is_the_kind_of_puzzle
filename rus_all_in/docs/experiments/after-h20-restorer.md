# Train-only residual restorer after the frozen h20 tail

Статус: **decisive train-only reject; evaluation targets unopened**.

Primary predictions были target-free заморожены, но по stop-правилу после
отрицательной train-only диагностики primary targets не декодировались.
Confirmation, holdout, competition test и production не открывались и не
менялись.

## Почему это не повтор старого R6

Старый `restoration_r6.py` обучал full-canvas DualNAF на spatial crops и подавал
ему raw canvas вместе с NLM h10. На неверной раскладке receptive field такой
сети пересекал границы фрагментов и смешивал ложных соседей. Поздний
tile-wise запуск старого checkpoint убрал cross-tile context, но всё ещё решал
другую задачу: модель работала **до** harmonizer/final NLM и её output затем
снова сильно фильтровался.

Здесь впервые проверена ровно предложенная residual-after-h20 формулировка:

```text
dirty board only
  -> bilateral scores -> solve_buddies(max_edges=96)
  -> strict raw 576-tile permutation audit
  -> exact frozen RGB offsets -> bounded luminance gains
  -> full-canvas colored NLM h20
  -> shared model independently on each upright 20x20 tile
       input = same-position pre-h20 RGB + post-h20 RGB (6 channels)
       output = bounded residual around post-h20
  -> fixed convex blend around h20
```

Модель не получает соседние tiles, coordinates, reference/template, другое
изображение или target. Layout и identity каждого из 576 фрагментов остаются
неизменными; rotation, warp, resampling, substitution и generation отсутствуют.

## Train-only data contract

Использован общий deterministic selector
`aiijc-puzzle-experiments-v1`, seed `20260829`:

- ranked train `0:96` — fit;
- ranked train `96:120` — disjoint diagnostic;
- calibration targets на этом этапе не открывались.

Для каждой fit-доски clean target использовался только как train supervision:
Hungarian bijection из `candidate_supply.recover_layout` сопоставляла dirty tile
его clean identity. Затем clean identities переставлялись в ту же предсказанную,
в том числе неверную, bilateral-buddies96 раскладку. В train loss допускались
только пары с assignment margin `>=1.0` и dirty/clean RGB RMSE `<=80`: всего
23 464 пары, в среднем 244.4 на доску (range 57–456). Это target-assisted
фильтр только training data; inference API его не содержит.

Архитектура — 33 859-parameter MPS-efficient NAF-style residual network,
8 blocks, width 32, residual cap `64/255`. Ending layer инициализирован нулём,
поэтому исходная сеть byte-exact воспроизводит h20. Обучение: 2 500 steps,
batch 256, AdamW, cosine LR `2e-3 -> 2e-5`, Charbonnier pixel loss + gradient
loss + residual penalty. Предобработка заняла 65.63 s, обучение — 637.18 s.
Отдельный architecture-only MPS benchmark перед запуском дал 14.68 steps/s и
3 757 tiles/s на batch 256; полный loss/data pipeline был медленнее.

## Train-only diagnostic result

Все candidate alphas были фиксированы до diagnostic: `.125`, `.25`, `.5`, а
pure model разрешался только при более строгой target-free safety. Результат на
24 disjoint train boards:

| Arm | Mean RGB SSIM | Delta к h20 | Wins vs h20 | Mean grid-ratio / h20 |
|---|---:|---:|---:|---:|
| A: h20 | 0.253924 | 0 | — | 1.000 |
| B: h28 | **0.265213** | **+0.011289** | 24/24 | — |
| residual alpha=.125 | 0.245621 | −0.008303 | 0/24 | 1.169 |
| residual alpha=.25 | 0.234992 | −0.018932 | 0/24 | 1.334 |
| residual alpha=.5 | 0.210985 | −0.042939 | 0/24 | 1.580 |
| pure model | 0.165231 | −0.088693 | 0/24 | 1.830 |

Даже минимальный blend проиграл h20 на всех 24 досках и нарушил grid safety:
mean/max relative grid ratio `1.169/1.310` при допустимых `1.05/1.12`.
Усиление alpha монотонно ухудшало и SSIM, и сеточный артефакт. Значит, сеть
научилась слишком агрессивной независимой per-tile коррекции, которая создаёт
brightness/edge discontinuities; причинная формулировка этой интерпретации не
доказана, но measured failure однозначен.

Train-only selection rule заморозил наименее агрессивный `alpha=.125`, поскольку
ни один arm не прошёл safety. Это не рекомендация: frozen candidate нужен был
лишь для воспроизводимого fail-closed receipt.

## Freeze-before-target и stop decision

После checkpoint и train diagnostic создан immutable prereg для historically
exposed calibration panels:

- primary ranked `192:216`, count 24, names digest
  `dcca17ffed3c85326b94da6af89dc93d7d4c0add3cbc3e20e292814014c10185`;
- conditional confirmation `216:240`, count 24, names digest
  `5812189771d84be0bda10208db7fb2131926643a9ef58923fff258b6630111af`;
- primary и confirmation disjoint, но **не fresh**: legacy report ранее открыл
  все 700 calibration records.

Config SHA-256:
`9c6e232f02df7694fb0fc3631bf009a92ae87684f352fee2404d82c20086d447`.
Checkpoint SHA-256:
`0c33ea2729d799141370eec613f1e24c72f5eb8e204db176e00b0aef1e685b6b`.
Train report SHA-256:
`e98af03cabb4f85392ebda76b266594417d2c2e236c578f8dc8d68ba0378c49f`.

Для primary были построены только target-free raw/layout/pre-h20/h20/h28/model
predictions; strict permutation audit прошёл 24/24. Commitment SHA-256:
`b766aff93c76f791263fc8f3ece3b5d80b4382aa589935d70ae903676c91a82f`.
После train-only результата root применил более консервативный stop: **не
открывать даже primary targets**. Поэтому `primary/report.json` отсутствует,
target receipt отсутствует, manual sheet не нужен, confirmation directory
отсутствует.

## Решение и правило «не повторять»

Не использовать и не масштабировать этот exact post-h20 independent-tile
checkpoint или его alpha blends. Наблюдаемый провал существенно больше любого
ожидаемого малого tail gain, а минимальный arm уже нарушает grid safety. Не
считать отсутствие calibration score незавершённым экспериментом: stop до
target decode — намеренное решение, полностью определённое train-only 0/24.

Повтор имеет смысл только при материально другой постановке, заранее
устраняющей per-tile discontinuity (например, residual с exact zero boundary и
board-consistent color constraint), и лишь после нового train-only gate. Простое
уменьшение alpha, больше steps/blocks или повтор текущего clean-identity loss не
является новым направлением.

## Артефакты и QA

- source: `src/aiijc_puzzle/after_h20_restorer.py`;
- runner: `scripts/run_after_h20_restorer.py`;
- tests: `tests/test_after_h20_restorer.py`;
- checkpoint/report:
  `outputs/after-h20-restorer/train96-diagnostic24-v1/`;
- immutable prereg:
  `configs/after_h20_restorer_reused_calibration_v1.json`;
- target-free commitment:
  `outputs/after-h20-restorer/evaluation/9c6e232f02df7694fb0fc3631bf009a92ae87684f352fee2404d82c20086d447/primary/prediction-commitment.json`.

```bash
uv run ruff check src/aiijc_puzzle/after_h20_restorer.py \
  scripts/run_after_h20_restorer.py tests/test_after_h20_restorer.py
uv run pytest tests/test_after_h20_restorer.py
```

Тесты проверяют exact identity initialization, same-index tile batching,
детерминированный convex blend, train-only clean-identity alignment и сохранение
upright 20x20 geometry.

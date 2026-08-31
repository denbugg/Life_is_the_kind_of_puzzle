# TASKA exact cyclic row-phase DP

Дата: 2026-08-31. Статус: **отклонён на local gate**.

## Зачем открывали направление

Перед реализацией повторно просмотрены handoff 2026-08-31 и TaskA no-repeat
ledger. Уже закрыты individual-swap tails, block-Hungarian, global board rolls,
component origin/centering, multistart portfolios и расширение arm roster под
тот же all-bond selector. Здесь проверялось другое ограниченное пространство:
membership и циклический порядок фрагментов внутри каждой из 24 строк
фиксированы, но каждая полная строка независимо выбирает одну из 24 циклических
фаз. Это `24^24` layouts, но точный Viterbi DP решает задачу за
`24 * 24` states и `23 * 24 * 24` transitions.

## Target-assisted диагностика до фиксации кандидата

На одном ранее открытом local32 panel измерялся только потолок пространства
ходов. Truth-pair oracle по фазам строк дал:

- mean pair delta `+1.0625`, W/T/L `13/19/0`;
- mean exact delta `+0.125`;
- mean changed rows `3.96875`.

Аналогичный column-phase oracle дал `+0.750` pairs, но `-2.000` exact и был
отброшен до preregistration. Диагностика показала небольшой, но реальный
структурный headroom; она не использовалась внутри deployable solver-а.

## Единственный зафиксированный кандидат

Preregistration SHA-256:
`4f19bdea5caa3115980329a90c844e85d2516ad4d5d2f31be5bff5ff484524fd`.

- control: точный frozen final layout confirmed six-arm fusion;
- moves: только независимые циклические фазы фиксированных строк;
- objective: исходная raw TASKA seam-cost по всем 1,104 реализованным bonds;
- optimizer: точный Viterbi минимум, затем минимальное число изменённых строк
  как tie-break;
- gates: local pair mean `>=0`, затем held `>=0`, затем fresh;
- без weights, thresholds, budgets, iterations, arm selection, learned guard,
  column fallback или sweep.

Target-free layouts и provenance freeze записывались до восстановления exact
reference. Все layouts — строгие перестановки 576 исходных upright fragments;
pixels, production, competition test, submission и postprocess не затрагивались.

## Результат

| local32 | Confirmed fusion | Row-phase DP | Delta, source CI95 | W/T/L |
|---|---:|---:|---:|---:|
| Satisfied pairs | `326.78125` | `326.68750` | **`-0.09375 [-0.28125,0]`** | `0/31/1` |
| Adjacency recall | `0.2959975` | `0.2959126` | `-0.0000849 [-0.0002548,0]` | `0/31/1` |
| Exact tiles | `5.9375` | `5.9375` | `0.0000 [0,0]` | `0/32/0` |

Raw objective остался неизменным на 31/32 boards. На единственном сработавшем
board DP сдвинул все 24 строки, уменьшил собственную seam-cost на `6.50566`, но
потерял три истинные пары. В среднем изменилось `0.75` rows/board, а raw-cost
уменьшилась на `0.20330`; objective-monotonicity выполнена на 32/32.

Local gate провален, поэтому held32 и fresh32 не открывались.

## Вывод и no-repeat

Confirmed fusion уже почти всегда является фазовым минимумом исходной raw
seam-cost. Небольшой truth-oracle headroom существует, но эта objective его не
идентифицирует: единственный найденный ею descent вреден. Не повторять
row/column cyclic-phase DP поверх final fusion с тем же raw all-1104 objective,
не подбирать штраф за фазу, порог улучшения или mixture на открытом local32.
Возвращаться к этому пространству имеет смысл лишь с новым независимым
target-blind сигналом, который непосредственно определяет место разрыва строки.

## Артефакты

- report: `outputs/taska-row-phase-dp/fixed-v1/report.json`, SHA-256
  `5e787ea9b45d7f37cf2c69350514533f297e35204ca2e7e48fd729c18ecf5438`;
- target-free smoke: `outputs/taska-row-phase-dp/smoke-preregistered-v1/`;
- preregistration: `configs/taska_row_phase_dp_v1.json` и `.sha256`;
- solver: `src/aiijc_puzzle/taska_row_phase_dp.py`;
- runner: `scripts/run_taska_row_phase_dp.py`;
- tests: `tests/test_taska_row_phase_dp.py` (exact brute-force optimum,
  identity optimum, strict-permutation rejection);
- Weco Observe pair+exact: step `113`, parent `102`.

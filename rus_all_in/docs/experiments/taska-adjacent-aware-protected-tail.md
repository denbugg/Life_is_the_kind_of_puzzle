# TASKA adjacent-aware protected tail96

Дата: 2026-08-31. Статус: **положительный opened/held signal, но fresh sign
не перенёсся; current tail96 остаётся default**.

## Fixed hypothesis

Текущий protected tail исключал swaps соседних board-позиций не из-за
содержательной гипотезы, а потому, что его vectorized placement-unary delta
точна только для позиций без общей грани. Этот эксперимент изменил ровно это:

- старт — тот же seed-0 raw/logistic/focal/nonlinear layout, выбранный по
  original TASKA cost на всех 1104 board bonds;
- tiles, участвующие в уже realised harvested edge, остаются неподвижными;
- objective — те же original `cost_right/cost_down`;
- `max_swaps=96`, `minimum_gain=1e-9`, без budget/weight/threshold sweep;
- для non-adjacent позиций используется прежняя vectorized формула;
- для horizontal/vertical adjacent позиций delta считается exact before/after
  суммой по union всех затронутых directed board bonds. Общая грань входит в
  union один раз;
- на каждом шаге берётся глобальный minimum delta, exact ties разрешаются
  стабильным row-major порядком.

Unit tests перебирают обе ориентации каждой horizontal/vertical adjacent пары
на `4×4`, сравнивают delta с brute-force полным board cost, а также проверяют
все non-adjacent пары, монотонность, preservation protected relations, strict
permutation, read-only output и stable tie.

## Legality и evaluation protocol

Каждый layout — строгая перестановка всех 576 исходных upright `20×20` tiles;
пиксели не меняются, не вращаются и не подменяются. Matcher matrices и
harvested edges уже были frozen target-free. Candidate layouts, metadata и их
SHA были записаны до восстановления exact synthetic references. Targets
использовались только в offline scoring; competition test не открывался.

Preregistered gates:

1. opened32 pair delta `>=0` открывает unchanged held diagnostic32;
2. held pair delta `>0` открывает unchanged fresh32;
3. fresh служит confirmation, без последующей подстройки.

## Results

| Panel | Control pairs / recall / exact | Adjacent pairs / recall / exact | Pair delta, CI95 | Exact delta, CI95 |
|---|---:|---:|---:|---:|
| opened32 | 341.3125 / 0.309159873 / 4.75000 | 341.5625 / 0.309386322 / 4.62500 | +0.2500 `[-0.59375,+0.875]` | -0.12500 `[-0.3125,+0.03125]` |
| held diagnostic32 | 337.5625 / 0.305763134 / 3.06250 | 338.2500 / 0.306385870 / 3.03125 | +0.6875 `[-0.09375,+1.46875]` | -0.03125 `[-0.125,+0.0625]` |
| fresh32 | **346.0625 / 0.313462409 / 1.15625** | 345.9375 / 0.313349185 / 1.03125 | -0.1250 `[-0.78125,+0.5625]` | -0.12500 `[-0.3125,+0.03125]` |

Mean accepted adjacent swaps per board были `4.53`, `5.16`, `5.47` на трёх
панелях соответственно; candidate действительно активировал новый путь на
каждом case. Он монотонно уменьшал objective относительно собственного
pre-tail старта и сохранял все initially realised relations. В сравнении с
control его greedy path иногда приходил к другому local minimum, что ожидаемо:
расширение множества шагов не гарантирует лучший финальный minimum после
ограниченного greedy budget.

## Decision и no-repeat boundary

Это реальный слабый pair signal, поскольку знак был положителен на opened и
held, а не no-op. Но fresh mean развернулся до `-0.125` пары, все pair CI
пересекают ноль, и exact mean отрицателен на каждой панели. Поэтому adjacent
tail **не заменяет** текущий four-arm+tail96 pair default и не входит в
submission pipeline.

Не повторять этот exact adjacent-delta extension с теми же start/protection /
objective/budget. Материально новый следующий вопрос может использовать обе
trajectories как candidate supply с независимым target-free selector либо
менять search method, но не подбирать `max_swaps` или gain threshold на уже
открытых panels.

Weco Observe: opened step 68 (parent 42), held step 69, fresh step 70 — в pair
и exact tracks; primary metric pairs/1104, secondary recall и exact.

## Reproduction и artifacts

```bash
.venv/bin/python scripts/run_taska_adjacent_tail.py --panel opened32 --workers 4
.venv/bin/python scripts/run_taska_adjacent_tail.py --panel held300 --workers 4
.venv/bin/python scripts/run_taska_adjacent_tail.py --panel fresh32 --workers 4
```

- runtime: `src/aiijc_puzzle/taska_adjacent_tail.py`;
- runner: `scripts/run_taska_adjacent_tail.py`;
- tests: `tests/test_taska_adjacent_tail.py`,
  `tests/test_run_taska_adjacent_tail.py`;
- frozen layouts, pre-score provenance and CI reports:
  `outputs/taska-adjacent-tail/{opened32,held300,fresh32}-v1/`.

Frozen raw solver remained byte-identical at
`97859e1f4ff6ceadf56ebccf429f2944ae7a5726631731d9f776224a667eb486`.

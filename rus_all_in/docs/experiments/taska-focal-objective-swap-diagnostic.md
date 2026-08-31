# One-swap focal-objective TASKA diagnostic

Дата фиксации: 2026-08-31. Статус: **negative opened diagnostic; do not
promote**.

## Гипотеза

Подтверждённый focal-gated tail защищает уже реализованные candidate edges с
focal logit `>=0`, но оставшиеся tiles переставляет только по original dense
seam cost. Этот screen заменил objective ровно для одного глобально лучшего
non-adjacent swap свободных tiles:

- `positive_softplus`: sparse cost `-softplus(logit)` только для candidate
  edges с `logit>=0`;
- `signed_logit`: sparse cost `-logit` для всех selected-supply candidate
  edges, чтобы одновременно создавать положительные и разрывать отрицательные
  связи.

Endpoints всех исходно реализованных focal-positive edges неизменяемы. Target,
filename semantics, clean pixels и canonical coordinates в выборе swap не
используются; результат всегда является strict permutation исходных upright
tiles. Это меняет именно objective, поэтому не является повтором tail96/192,
adjacent-tail или raw-log-equivalence screens.

## Результат на уже открытом local32

Control — frozen confirmed six-arm fusion: `326.78125` satisfied pairs,
`5.93750` exact tiles/board.

| Objective | Changed boards | Pairs | Pair delta | Pair W/T/L | Exact delta |
|---|---:|---:|---:|---:|---:|
| positive softplus | 2/32 | 326.81250 | +0.03125 | 1/31/0 | 0.00000 |
| signed logit | 32/32 | 326.46875 | -0.31250 | 0/23/9 | 0.00000 |

У positive-only варианта текущая сборка уже почти локально насыщает sparse
objective: найдено лишь два допустимых swap, один добавил правильную пару,
второй был neutral. Это слишком редкий signal для отдельного held gate.
Signed objective менял каждый board, но каждый swap сильно ухудшал original
raw objective в среднем на `+72.14255`; девять boards потеряли суммарно десять
правильных пар и ни один board не выиграл. Средний focal objective gain
`3.64469` практически не коррелировал с pair delta (`r=0.0387`).

## Решение

Ветку закрыть без iterative swap, weight/threshold/budget sweep и без held или
fresh replay. Отрицательный focal logit над sparse harvested set надёжно
говорит, что конкретная relation сомнительна, но не сообщает, куда безопасно
переставить два освободившихся tile: такой swap ломает unrelated true seams.
Positive-only gain не отрицателен, однако coverage почти нулевой. Следующий
полезный consumer должен jointly repack component/free-region либо получить
более плотный независимый scorer; одиночный sparse-objective swap не годится.

Воспроизводимый target-free primitive:
`src/aiijc_puzzle/taska_focal_objective_swap.py`; unit tests:
`tests/test_taska_focal_objective_swap.py`. Weco pair/exact step `127`, parent
`102`.

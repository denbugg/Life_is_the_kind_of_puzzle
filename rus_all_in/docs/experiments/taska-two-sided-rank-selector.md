# Two-sided log-rank selector over six TASKA layouts

Дата фиксации: 2026-08-31. Статус: **strong negative opened diagnostic**.

## Гипотеза

Current six-arm fusion выбирает layout по сумме original all-1104 raw costs.
Один parameter-free альтернативный proxy устранил несопоставимый scale разных
source rows: каждый реализованный seam получил outgoing row rank и incoming
column rank в соответствующей `576×576` right/down matrix. Layout score —
сумма `log1p(rank)` по четырём rank-вкладам каждого из 1104 bonds; выбирался
минимум, deterministic tie — порядок frozen six-arm roster.

Это target-free правило не обучается, не использует thresholds/top-k и не
является прежним raw-log tail: оно выбирает между шестью полностью готовыми
post-tail layouts по двухсторонним scale-free ranks.

## Opened local32

Frozen confirmed fusion: `326.78125` pairs, `5.93750` exact. Two-sided rank:

- `324.12500` pairs, delta `-2.65625`, W/T/L `6/14/12`;
- adjacency recall `0.293591486`;
- `1.34375` exact, delta `-4.59375`, W/T/L `4/20/8`.

Selector choices raw/logistic/focal/nonlinear/selective/combined были
`4/4/2/8/9/5`, то есть proxy активно менял решения, но не находил pair-oracle.
Особенно опасны arm-specific global translations: rank score может улучшиться,
одновременно сдвинув десятки exact tiles.

## Решение

Ветка закрыта без held/fresh и без rank transform/top-k/blend sweep. Row/column
normalisation полезна для individual edge evidence, но сумма ranks всех 1104
реализованных bonds остаётся тем же correlated seam-proxy family и усиливает
winner's curse между whole layouts. Воспроизводимый primitive:
`src/aiijc_puzzle/taska_two_sided_rank_selector.py`; tests:
`tests/test_taska_two_sided_rank_selector.py`. Weco pair/exact step `131`,
parent `102`.

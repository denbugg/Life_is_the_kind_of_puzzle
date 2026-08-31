# TASKA: one-component relation anchor over confirmed six-arm fusion

Статус: **exact gate прошёл только на local32 и не перенёсся; слабый pair gain
перенёс знак на held32/fresh32, но не является новым подтверждённым default**.
Confirmed six-arm fusion остаётся формальным лидером. Competition test,
production, official best, postprocess и submission не затрагивались.

## Аудит новизны

Перед запуском проверены exact/origin/component документы и no-repeat ledger.
Новый кандидат не повторяет:

- global cyclic roll, coordinate/whole-layout origin или frame-side origin;
- largest-component-to-center, monotone component placement и coordinate sorter;
- absolute component-shift MLP или independent absolute placer;
- pre-build component relation reorder и unique-fullres priority promotion.

Вместо повторной сборки layout он работает **после** frozen confirmed six-arm
solver-а, не меняет внутреннюю геометрию уже реализованных компонент и допускает
ровно одно relation-implied rigid translation с локальным bijective fill.

## Oracle/diagnostic на уже открытом local32

Диагностический скрипт использует organizer-train exact targets только после
восстановления frozen control. Это target-assisted потолок, не deployable
selector. Компоненты определены target-blind: connected components из
selected-supply edges с focal logit `>=0`, уже реализованных final six-arm
layout.

| local32, tiles/pairs per board | Значение |
|---|---:|
| frozen six-arm exact | `5.9375` |
| frozen six-arm pairs | `326.78125` |
| translation-aligned tiles | `71.0625` |
| oracle best global cyclic exact | `71.9375` |
| oracle best global cyclic pairs | `318.8125` |
| oracle best one-component local-fill exact | `52.3750` |
| oracle best one-component local-fill pairs | `302.9375` |
| pure nontrivial component tiles | `75.71875` |
| sum of dominant feasible shift support over nontrivial components | `198.125` |

В среднем есть `39.44` nontrivial components, largest содержит `75.63` tiles.
Oracle one-component move даёт `+46.44` exact tiles, но теряет `23.84` pairs.
Следовательно, translation headroom большой, но original seam objective обычно
предпочитает неправильную абсолютную gauge. Нужен inference-visible сигнал,
который различает translation компонент независимо от уже оптимизированных
локальных seams. Pure/dominant ceilings нельзя складывать в готовый layout:
они не учитывают взаимные collision и packing.

Полный отчёт:
`outputs/taska-six-arm-component-shift-diagnostic/local32-v1/report-v2.json`,
SHA-256 `e39acd60288852f64678412a413936ef7ceb20634c5a199174698ff1a1c86121`.

## Единственное зафиксированное target-blind правило

Правило было подписано до любого held32/fresh32 candidate construction или
scoring:

1. В final six-arm layout строятся компоненты только по selected-supply edges
   с focal logit `>=0`, которые layout уже реализует.
2. Каждый оставшийся selected-supply edge между разными компонентами предлагает
   единственный integer shift одного endpoint component, который реализовал бы
   этот edge.
3. Для `(component, shift)` складывается `softplus(focal_logit)`; support, более
   низкий all-bond cost и стабильный порядок используются только как tie-break.
4. Проверяются только relation-implied, nonzero, board-feasible shifts
   non-singleton component. Разрешено переместить ровно одну компоненту.
5. Tiles, непосредственно вытесненные новым footprint, row-major переносятся в
   освобождённые cells. Получается строгая upright permutation всех 576 исходных
   fragments.
6. Move принимается только если original TASKA all-1104 raw seam cost строго
   ниже control. Иначе возвращается control bit-for-bit.

Ни targets, ни source identity, ни filename, ни clean image, ни absolute tile
ID feature в inference API отсутствуют. Threshold, score, fill и guard не
sweep-ились. Signed config:
`configs/taska_component_relation_anchor_v1.json`, SHA-256
`97d7b328672ca57605ed810d395c6518dc34615ec862f19d8ac33d36eaa46288`.

Gates: local должен иметь `exact delta > 0` и `pair delta >= 0`; held —
`exact delta >= 0` и `pair delta >= 0`, чтобы открыть fresh. Оба прошли.

## Fixed local/held/fresh результат

| Panel | Control exact | Candidate exact | Delta | Control pairs | Candidate pairs | Delta | Changed |
|---|---:|---:|---:|---:|---:|---:|---:|
| local32 | `5.93750` | `6.00000` | `+0.06250` | `326.78125` | `327.15625` | `+0.37500` | `8/32` |
| held32 | `1.90625` | `1.90625` | `0.00000` | `345.31250` | `345.62500` | `+0.31250` | `3/32` |
| fresh32 | `0.93750` | `0.93750` | `0.00000` | `355.62500` | `355.78125` | `+0.15625` | `5/32` |

Exact W/T/L: local `1/31/0`, held `0/32/0`, fresh `0/32/0`. Pair W/T/L:
`4/25/3`, `3/29/0`, `3/28/1`. Target-free construction заняла примерно
`0.42 s` на каждую 32-board panel; весь 96-board build+evaluation command —
`6.6 s` wall-clock.

Переносимый pair знак показывает, что strict seam veto полезен как
fail-closed primitive. Но exact-primary evidence состоит из одного local win и
нулевого held/fresh эффекта. Кандидат поэтому не promoted и не заменяет
formal-confirmed six-arm fusion.

## Вывод и no-repeat

Закрыт exact-путь «relation votes + один rigid move + original seam veto» в
этой фиксированной форме. Не повторять softplus/support/minimum-cost-gain,
component-size или focal-threshold sweep на открытых panels: это nearby tuning.
Также не переносить oracle shift напрямую — он target-assisted и обычно теряет
много pairs.

Следующая materially new exact formulation должна jointly согласовывать
несколько component translations/cycles либо использовать независимый
board-conditioned absolute signal. Ключевой критерий: inference score должен
уметь предпочесть exact-correct large-component translation даже тогда, когда
raw local seam cost временно ухудшается. Уже подтверждённый d64 component
relation score можно переиспользовать как primitive, но не через прежний
hard-edge ordering или этот single-move seam guard.

## Артефакты и проверки

- final report:
  `outputs/taska-component-relation-anchor/fixed-v1/report.json`, SHA-256
  `693f26d4851c72b0d80812392521df85b93570d18e513b761e348abd74b17316`;
- local/held/fresh target-free archives SHA-256:
  `ead42cd95baefdac26adcb887698e24ff395a401414a2735d14110e094f9120e`,
  `ee26714b7cac45a3a8eaae393a2bf444574b97d518b8c6b537b7a5c5904b09c6`,
  `89dbebc632fe61ee9ce2feafcf90efd60d70df4a2c9c5986c2774dbef5b069bc`;
- solver module:
  `src/aiijc_puzzle/taska_component_relation_anchor.py`, SHA-256
  `d16cce8452d9d5ce827ca96d544d300a74361e9c8d75a3a6f346f8b2b7b3d06f`;
- runner:
  `scripts/run_taska_component_relation_anchor.py`, SHA-256
  `bf4266b1ef6af0162c3b49d9ccf9dd615caf0b689b5d6aba9561acf1dd278b38`;
- oracle diagnostic:
  `scripts/diagnose_taska_six_arm_component_shifts.py`, SHA-256
  `5b26367b3801c76c31a57c28719ea82f4180f1c5d095ab6e624785f90c50d85b`;
- component-anchor + parent fusion tests: `8 passed`; target-free smoke, ruff
  and py_compile passed;
- Weco Observe exact+pair steps `120/121/122`, parent `102`.

Все frozen outputs проверены как strict permutations. Organizer competition
test не открывался, pixels не менялись, production/submission не запускались.

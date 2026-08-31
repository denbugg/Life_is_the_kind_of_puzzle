# BorderPointer-24: full-resolution field и causal pointer

Дата запуска: **2026-08-30**. Итог: **standalone free-run и единственный
baseline-guided rescue не promoted**. Full-resolution field/checkpoint остаются
research artifact для будущей calibrated relation fusion или sparse QAP; новый
fresh panel не открывался.

## Короткий результат

На source-disjoint exact `train128 / eval16×1` initial greedy BorderPointer дал
`1.4375` exact tile/board и `5.6273%` adjacency против matched frozen d64
`decoder144` с `4.7500` и `13.6549%`. Row/column также не улучшились. Gate
провален широко, поэтому ни scale-up, ни fresh `source64×draw2`, ни competition
test не разрешены.

Однако чувствительный анализ подтвердил, что conditional signal не полностью
нулевой. На том же уже открытом exact16 deployable d64-baseline prefix поднял
eligible pointer R@1 `0.333→1.267%` и R@5 `1.644→3.022%` против reset/no-prefix.
Единственный заранее зафиксированный causal repair с четырьмя swaps сохранил
exact буквально `76→76` и потерял лишь `0.0453 pp` adjacency. Это почти
нейтральная конверсия, но не score gain; более широкий budget16 потерял один
exact tile total и `0.2434 pp` adjacency. Поэтому decoder formulation закрыта,
а conditional evidence сохранён для другого global optimizer-а.

## Что было исправлено code review-ом

Исходный `border_pointer_sorter.py` уже имел правильные 20×20 perimeter features,
permutation-equivariant board attention и used-mask, но runnable pilot ещё не
соответствовал immediate brief в четырёх материальных местах:

1. Pointer logits не использовали выбранных left/up соседей, хотя pair heads
   вычислялись.
2. Не было learned border unary и четырёх distance-to-border признаков.
3. Causal state состоял из одного `GRUCell`, а frozen brief требовал четыре
   causal layers.
4. 2-D residual field использовал zero padding, создавая искусственную чёрную
   рамку tile.

Исправленный модуль:

- сохраняет lattice `20×20` во всех convolution blocks и использует replication
  padding;
- кодирует 76 уникальных clockwise perimeter samples и четыре side summaries;
- добавляет frozen d64 tile/side summary и directional score как retained raw
  evidence;
- имеет четыре recurrent causal layers, fixed row/column slot queries,
  normalised top/left/bottom/right distances, learned border unary и явные
  right(left,candidate)/down(up,candidate) logit terms;
- возвращает только strict permutation исходных upright tile IDs.

Unit tests проверяют 76 уникальных perimeter positions, отсутствие spatial
downsample, gradient flow, left/up contribution, input-permutation equivariance,
strict greedy/beam permutations и strict baseline repair.

Важно не расширять вывод этого pilot-а на все возможные PuzLM-like модели.
Именно измеренная v1 использовала raw+tile-normalised RGB, а не отдельные fixed
gradient channels; adjacency auxiliary, но не paired-corruption field
consistency; один cached corruption draw/source; четыре GRU layers, а не
Transformer decoder. Это честные frozen design deviations, а не knobs для
ретроспективного sweep на exact16.

## Mechanical gate и compute

4×4 capacity использовала тот же strict pointer invariant и один fixed
corrupted/shuffled board. За 120 MPS updates:

- greedy exact `16/16`;
- rows/columns `16/16`;
- adjacency `24/24`;
- teacher-forced pointer NLL `0.00507`;
- runtime `10.28 s`.

Full-grid one-update benchmark:

| Device | Seconds/update |
|---|---:|
| MPS | 3.265 |
| CPU | 3.653 |

MPS был выбран до pilot. Модель содержит `1,617,899` trainable parameters и
`2,193,907` total вместе с frozen d64, то есть далеко ниже preregistered 10M cap.
Фактические 400 updates заняли `725.66 s`.

## Initial exact16

Frozen config:
[border_pointer_preregistered_v1.json](../../configs/border_pointer_preregistered_v1.json).
Fit/eval используют только manifest `train`, exact inverse deterministic shuffle
и `restoration_r6.distort_tiles`. Eval исключена из d64 и current pointer fit
lineage. Dirty-only greedy и matched decoder layouts записаны до открытия exact
references; frozen artifact не содержит clean pixels или labels.

| Metric, mean/board | d64 decoder144 | BorderPointer greedy | Delta |
|---|---:|---:|---:|
| Exact tiles | **4.7500** | 1.4375 | −3.3125 |
| Correct rows | **31.875** | 31.8125 | −0.0625 |
| Correct columns | **31.000** | 25.625 | −5.375 |
| Adjacency | **13.6549%** | 5.6273% | −8.0276 pp |
| Translation-aligned tiles | **16.750** | 6.3125 | −10.4375 |

Teacher-forced exact16 diagnostics были намного оптимистичнее free-run:

- pointer NLL `2.2735`;
- pointer accuracy `42.1875%`;
- right/down R@1 `16.293/17.505%`.

Это не end-to-end result: teacher forcing даёт истинный prefix, а used-mask
удаляет всё больше кандидатов поздно в raster sequence. Greedy cascade показал,
что этот oracle conditional ceiling не переносится автоматически.

Matched baseline на этой маленькой panel имеет один outlier: на
`img_003229.png` decoder144 получил 53 exact tiles против 2 у pointer. Но провал
не сводится только к нему: без этой board totals всё ещё `21` у pointer против
`23` у baseline. По board-wise exact delta получилось `6 wins / 4 ties / 6
losses`, median delta `0`; это нестабильное complementarity, а не основание
выбирать модель.

Primary report:
`outputs/border-pointer/pilot-d64-train128-s400-exact16-mps/report.json`, SHA-256
`7ae2c770e09ed61e0ffa67cc11fa836f09570c2f91ba7f22d751f404c7f9ce44`.
Checkpoint SHA-256
`c404c76f732d85554dd1f5bc7db17a6fe0275f356f39dffe60f54a2e8a2dcb5d`.
Frozen predictions SHA-256
`a7422bf13eb10103b0df229f6db4d4a82cc86e222720da1190732ac797e6b7ee`.

## Единственный baseline-guided rescue

После initial fail, но до новых rescue predictions, был frozen один development
contract:
[border_pointer_baseline_repair_preregistered_v1.json](../../configs/border_pointer_baseline_repair_preregistered_v1.json).
Он использует ту же уже открытую exact16 panel и не является fresh confirmation.

Алгоритм начинает с d64 decoder144 permutation. В raster slot оставляется
текущий baseline tile, кроме случая, когда лучший unused pointer candidate
выигрывает минимум `1.0` logit. Тогда candidate меняется местами со своей
единственной будущей позицией. Frozen Socket-OT top-16 guard считает только
grid edges, затронутые swap, и запрещает уменьшать число поддержанных seams.
Проверены ровно budgets 4 и 16 без margin/weight sweep.

| Metric, mean/board | Baseline | Budget 4 | Δ4 | Budget 16 | Δ16 |
|---|---:|---:|---:|---:|---:|
| Exact tiles | 4.7500 | 4.7500 | 0.0000 | 4.6875 | −0.0625 |
| Correct rows | 31.875 | 31.875 | 0.000 | 32.625 | **+0.750** |
| Correct columns | 31.000 | 30.875 | −0.125 | 31.1875 | +0.1875 |
| Adjacency | 13.6549% | 13.6096% | −0.0453 pp | 13.4115% | −0.2434 pp |

На 16 boards margin прошли 650 proposals: guard принял 256 и vetoed 394.
Каждая arm сохранила 16/16 strict permutations. Ни low exact, ни low geometry
gate не прошёл.

Deployable conditional diagnostic использует d64 baseline identities как prefix
и исключает из denominator только позиции, где точный tile уже был consumed
ошибочным prefix. Eligible coverage составила `4500/9216 = 48.828%`:

| Candidate score | R@1 | R@5 |
|---|---:|---:|
| Reset/no-prefix | 0.333% | 1.644% |
| d64 baseline prefix | **1.267%** | **3.022%** |
| Delta | **+0.933 pp** | **+1.378 pp** |

То есть причинный conditional signal есть, но его абсолютной точности не хватило
для безопасного исправления decoder144. Большие oracle-prefix 42% нельзя
использовать как оценку deployable качества.

Rescue report:
`outputs/border-pointer/baseline-guided-repair-margin1-b4-b16-top16-exact16/report.json`,
SHA-256
`ff28f4d767c5630b35a86eed3733aa25ae64fef61c7810fdc90d6d2d84d3e054`.

## Решение и следующий допустимый compose

- Не менять default d64 decoder144.
- Не открывать fresh/test и не делать margin, beam, size или augmentation sweep
  этой standalone/free-run formulation.
- Сохранить full-resolution field checkpoint и additive right/down evidence
  interface. Relation v1.1/new-edge arms уже дали отдельный небольшой exact или
  geometry signal; их можно позже подать как calibrated left/up evidence, не
  меняя legal strict-permutation renderer.
- Если новый field/reranker независимо улучшит high-confidence edge precision,
  использовать sparse QAP/baseline component optimizer. Текущий результат
  показывает именно cascade/gauge problem: полезнее сохранить d64 geometry и
  обучать constrained component/origin decision, чем снова строить raster
  layout с нуля.
- Любой такой compose начинает с нового preregistration и development panel;
  exact16 из этого отчёта больше не является fresh.


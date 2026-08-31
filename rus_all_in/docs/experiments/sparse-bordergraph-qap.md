# Rank2 Sparse BorderGraph-QAP

## Статус

**Bounded discovery gate fail; stop/no default.** Experiment реализован по
[preregistration](../../configs/sparse_bordergraph_qap_preregistered_v1.json),
но source-disjoint exact16 QAP после одинакового cyclic-border5 буквально
совпал с matched decoder144 на всех шестнадцати boards:

| Exact16 mean | Decoder144 + cyclic5 | Sparse QAP + cyclic5 | Delta |
|---|---:|---:|---:|
| Correct tiles / board | 2.500 | 2.500 | 0.000 |
| Direct placement | 0.4340% | 0.4340% | 0.0000 pp |
| Correct rows / board | 24.813 | 24.813 | 0.000 |
| Correct columns / board | 30.125 | 30.125 | 0.000 |
| Adjacency | 13.9946% | 13.9946% | 0.0000 pp |

Решающий prerequisite внутри decoder-а тоже отсутствует: mean **pure
quadratic** truth-minus-decoder energy равен `−77.475`; unary difference был
`+5.842`, joint difference `−149.108`. То есть learned sparse energy лучше
оценивает baseline layout, чем exact truth. Gate fail не вызван одной только
слишком сильной Hungarian anchor.

Tiny local signal есть, но он ниже уровня продолжения этой formulation:
conditional-on-supplied top-8 edge R@1 `47.172% → 47.588%` (`+0.416 pp`), при
top-8 truth coverage `35.927%`. Его сохранить как диагностику fusion, но не
ослаблять после результата quadratic/exact gate и не sweep-ить anchor/top-k.

## Почему запуск теперь разрешён

В literature memo Rank2 был условным: новый QAP запрещалось запускать на старой
слабой edge matrix. Activation требовал `+3 pp` high-confidence precision и
`+1` correct top-32 attachment на board, либо positive BorderPointer exact.
Frozen fullres-relation fusion дал существенно больше:

- top-32 confidence precision `20.508% → 34.961%`, то есть `+14.453 pp`;
- correct top-32 attachments `6.563 → 11.188/board`, то есть `+4.625`;
- union candidate coverage `54.129% → 61.587%`, то есть `+7.459 pp`;
- relation R@1/R@5 дополнительно выросли на `+1.157/+1.438 pp`.

Это representation evidence, а не layout claim. Оно только открывает bounded
decoder pilot; promotion всё ещё требует отдельного fresh source64×draw2 gate.

## Distinction audit: что именно не повторяется

### Не SocketPermutationFlow

Flow v1 получал sparse Socket graph, но каждый refinement сворачивал его в
bounded coordinate proposal, independent row/column logits и Hungarian. На
24×24 это подняло row accuracy, одновременно разрушив adjacency
`15.670% → 1.291%`. Новый arm вообще не предсказывает coordinate displacement:
каждый из двух unrolled steps дифференцирует одну и ту же directional quadratic
energy

`sum_(i→j,right/down) w_ij sum_s P[i,s] P[j,neighbour(s)]`.

Поэтому правильная component edge остаётся фактором objective на каждом шаге,
а не теряется после separable projection.

### Не historical solver-only QAP / seeded QAP

P17 проверил exact-delta arithmetic, P33/P36 остановились по runtime, historical
Russian HBT/QAP получил production adjacency около `6.24%`, а старый seeded QAP
был закрыт в своей реализации. Общая проблема: solver работал поверх прежних
score-ов, не прошедших precision knee. Здесь запуск произошёл только после
source-disjoint fusion activation выше. Edge MLP обучается на exact organizer
shuffles, но frozen raw/restored/relation supply и candidate roster не получают
truth.

### Не LP/pose synchronization

Historical LP translation synchronization оживал около selected-edge precision
`0.9`; P13 robust pose sync оставался около chance direct и имел translation
gauge. Новый decoder не утверждает, что относительная component map сама задаёт
origin. Он сразу match-ит tile graph к **фиксированному абсолютному** 24×24 grid;
dirty-visible tile-to-slot unary и frozen decoder anchor входят в joint energy,
а final Hungarian возвращает одну absolute bijection.

### Не standalone Sinkhorn / dense affinity

Sinkhorn здесь — только шесть нормализаций continuous tile-to-slot relaxation
внутри двух quadratic mean-field steps. Это не standalone linear assignment
objective. Dense `(N²)²` affinity не строится: top-8 graph содержит `9,216`
directed edges, сообщения имеют сложность `O(E·N)`. Матрица `P` размером
`576×576` допустима; four-index QAP tensor отсутствует.

## Frozen architecture

- Tile graph: для каждого tile отдельно top-8 right и top-8 down из union raw
  d64, full-resolution-restored d64, всех frozen fusion relation contacts и
  decoder-component internal edges.
- Edge features: raw/restored Socket и OT z-scores, reciprocal ranks, frozen
  fusion relation rank/query confidence, raw/restored supply flags и internal
  component flag. Truth не входит в candidate construction.
- Unary: raw/restored d64 context, восемь raw/restored border logits и dirty RGB
  mean/std против learned fixed-grid row/column/analytic slot token. Нет input
  tile-ID embedding и нет BorderPointer checkpoint/memory.
- Initial state: frozen decoder144 layout как target-free soft anchor.
- Optimizer: два sparse quadratic message steps, log-Sinkhorn, final Hungarian.
- Tail: одинаковый frozen cyclic-border5 применяется и к comparator, и к QAP.
- Output: только permutation original upright tile identities. Restored pixels
  остаются matcher-only.

## Mechanical и device gate

На 4×4 same-case mechanical capacity QAP восстановил `16/16` identities и
сохранил strict permutation. На полном random `N=576`, `E=9,216`, два
forward/backward steps заняли после warm-up median около `0.079 s` CPU и
`0.036 s` MPS. Это mechanics/runtime evidence, не quality.

## Source protocol

Final preregistration SHA-256: `79314268efc9e0268afefba56b97848adedeeac323e2106f01e744822ef70bba`.
До target access записан
`outputs/sparse-bordergraph-qap/pilot-fit64-s240-eval16-top8-mps/selection_commitment.json`
(SHA-256 `bbe66d8474b838823a02ca5817b2f19007b8f5cfd0e16d1a02d736277f0a84de`).
Fit64 digest — `0aa8005e…`, exact16 digest — `60259a51…`. Полный roster имеет
нулевое пересечение с frozen D2 exact40 (`058bec96…`) и whole-layout origin
fit256/eval16 (`d1d615ce…` / `feb6a3ae…`). Calibration, holdout и competition
test не открывались.

Discovery gate остаётся описательным: mean truth quadratic energy должна быть выше
energy frozen decoder layout и одновременно exact delta должен быть positive
либо adjacency loss не хуже `−1 pp`. Даже pass не меняет production default.

## Artifacts и решение

- Main report:
  `outputs/sparse-bordergraph-qap/pilot-fit64-s240-eval16-top8-mps/report.json`,
  SHA-256 `a12ca65775facb0717e81fbeaa3735a6fde2ca186f93b428e2985180825ce115`.
- Frozen strict predictions:
  `frozen_predictions.npz`, SHA-256
  `a2c1ea5e3c0f1eeef35da744d7e1b723dc66ebfcc39ad23427a416a0c31f281f`.
- Pure quadratic decomposition:
  `quadratic_energy_analysis.json`, SHA-256
  `f493c7162337ea24b3fc00590314e0f5eedd55335df862c66d67400193bd42df`.
- Checkpoint SHA-256:
  `1731b9d70228b2c76d58dbcc23429d9608b1845645dc552c336bb4832400b1db`.

Все 16 outputs — strict permutations original upright identities. Restored
view не рендерился. Calibration, organizer holdout и competition test не
открывались. Rank2 закрывается в этой bounded форме; fullres-relation fusion
остаётся полезным local/context evidence для materially другого decoder-а.

# TASKA six-arm whole-layout adjacency consensus

Дата: 2026-08-31. Статус: **rejected on local32**.

## Fixed target-blind rule

Все шесть arms (`raw`, `logistic`, `focal_top5`, `nonlinear`, `selective`,
`combined`) независимо проходят тот же focal-gated tail96. Для каждого strict
layout строятся его 1,104 directed right/down adjacency. Score arm равен сумме
числа других arms, повторяющих каждое его adjacency. Максимальный score
выбирает целый layout; при точном tie сохраняется frozen confirmed-six-arm
layout, если он присутствует среди tied arms, иначе действует стабильный
roster order.

Правило не использует target, filename, tile id semantics или absolute
coordinate. Оно не смешивает tiles между layouts и не меняет pixels. До
scoring были выбраны ровно этот overlap score и tie rule; вариантов веса,
margin или minimum support не было.

## Local32 result

Контроль — confirmed selective+unique-fullres six-arm fusion. Consensus
выбрал в основном слабые, но взаимно похожие current arms и резко проиграл:

| Metric | Confirmed fusion | Consensus | Delta |
|---|---:|---:|---:|
| satisfied pairs / board | `326.78125` | `311.78125` | **`-15.00000`** |
| adjacency recall | `0.295997509` | `0.282410553` | `-0.013586957` |
| exact tiles / board | `5.93750` | `1.37500` | **`-4.56250`** |

Pair W/T/L: `4/5/23`. Held32 и fresh32 не открывались для candidate rule.

## Interpretation and no-repeat

Agreement measures shared inductive bias, not truth. The four current arms
often reproduce the same wrong relations, so a majority-style whole-layout
selector suppresses the more diverse selective/fullres gains. Do not sweep an
overlap exponent, support threshold, arm weight or tie margin. The separately
measured oracle across the same six post-tail arms remains large
(`+4.875/+9.531/+6.688` pairs on local/held/fresh), so a future selector must
learn board-relative quality from independent target-visible training and
inference-visible confidence features rather than arm agreement alone.

Implementation retained only for reproducibility:
`src/aiijc_puzzle/taska_six_arm_consensus_selector.py`.

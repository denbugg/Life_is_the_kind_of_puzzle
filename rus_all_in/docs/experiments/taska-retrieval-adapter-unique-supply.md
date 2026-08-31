# TASKA retrieval-adapter unique supply

Дата: 2026-08-31. Статус: **local gate fail; held/fresh не открывались**.

## Fixed question и no-repeat

Эксперимент проверил, может ли положительный top32 candidate-supply signal
[full-resolution retrieval adapter](fullres-retrieval-adapter.md) конвертироваться
в confirmed six-arm TASKA fusion. Это отдельный nominator-only consumer:

- старый NAF fullres union использовал четыре restored v3/local scorers и
  support `>=3/4`;
- Twin unique пересекал direct Twin top32 с Union-v2 hard top144;
- DRUNet unique использовал width6 descriptor mutual-top1 и оказался резко OOD
  для старого focal verifier-а;
- здесь используется только adapter-step400 → frozen d64 top32, без score
  replacement, blend или нового standalone arm-а.

До target decode был подписан
`configs/taska_retrieval_adapter_unique_supply_preregistered_v1.json`, SHA-256
`eb4b2426d4fa6f7f6d3364d1886e5bed9026ddd08ef88001249b125955ebbcfe`.
Единственный natural reciprocity contract строит mutual row-top32 / column-top32
graph. Каждый source и target независимо выбирает edge по фиксированному ключу
`(max rank, rank sum, row rank, column rank, ids)`; номинируются только взаимные
choices. Затем удаляются current, selective accepted и confirmed fullres
accepted edges. Оставшиеся proposals проходят старый dirty-visible focal
`logit>=0` и расширяют только прежний `combined_union_focal` arm. Six-arm
roster, original all-1104 raw seam cost и focal-gated tail96 неизменны.

## Target-free feasibility до labels

Полный local32 сначала был заморожен без восстановления reference:

- control replay `32/32`;
- nominated `733.78` edges/board;
- unique после parent dedup `554.09`;
- focal accepted `18.34` (case range `4–42`);
- runtime `52.4 s` на MPS.

Заранее заданный target-free range `1..64` accepted и минимум два unique
proposal прошли. Archive/metadata/pre-score freeze SHA-256:
`53d466c6...875f / 19aaa15a...4371 / 8acea345...a95a`. Freeze report явно
содержит `reference_reconstructed=false`; только после этого был выполнен
единственный local decode.

## Local32 result

| Arm | Satisfied pairs | Recall | Exact tiles |
|---|---:|---:|---:|
| Confirmed fusion control | `326.78125` | `0.295997509` | `5.93750` |
| Adapter unique candidate | `325.37500` | `0.294723732` | `1.65625` |
| Delta | **`−1.40625`** | `−0.001273777` | **`−4.28125`** |

Pair source-bootstrap CI95 был `[-4.032,+1.063]`, W/T/L `2/23/7`. Exact CI95
`[-10.844,+0.125]`, W/T/L `3/25/4`. Fixed local gates требовали pair delta
`>=0` и exact delta `>=−1`; оба провалены.

Главный supply diagnostic объясняет результат:

| Stage | Edges/board | True/board | Precision |
|---|---:|---:|---:|
| all reciprocal-rank nominations | `733.78` | `169.28` | `23.07%` |
| unique после strong-parent dedup | `554.09` | `27.38` | `4.94%` |
| focal `>=0` accepted unique | `18.34` | `3.22` | `17.55%` |

Focal полезен относительно ungated unique pool (`4.94→17.55%`), но не прошёл
заранее заданную minimum calibration precision `25%`. То есть adapter видит
много хороших соседей в целом, однако почти весь сильный сигнал уже покрыт
confirmed parents; complement остаётся слишком слабым для rigid combined
consumer-а. Это не опровергает top32 adapter как candidate generator, но
закрывает именно direct unique-suffix extension.

## Решение, legality и no-repeat

Held32 и fresh32 не открывались. Не подбирать на local top-k, reciprocal key,
focal threshold, candidate cap, arm roster или tail budget. Следующий consumer
adapter evidence должен быть context-aware/calibrated до parent dedup либо
jointly выбирать relation evidence; простое добавление unique suffix в rigid
union повторять не надо.

Adapter/restored pixels использовались только matcher-ом. Raw d64 top32 сохранён
параллельно, dense TASKA costs не изменены, все layouts — строгие перестановки
576 original upright tiles. Adapter fit sources не пересекаются с
local/held/fresh. Competition test, postprocess, production, official best и
submission не затрагивались.

Артефакты:

- final report SHA-256
  `4f36424d04180438d8098cd0a154b9d8553b13b3f26b8748be8adb69dd191c0b`;
- module / runner SHA-256
  `e38d41cca1d0180afc94b92b59161e1de984c079262fee72df71cf984af18b3b` /
  `f0aee709b0b60fa98be2aadd494d453af512732ce65d50fc7b7ee56a6c25c903`;
- checkpoint SHA-256
  `00ca56f1be2c8e99bc8ef19b0d9190862d6bd5e4fb8b36fbe926087cd3945cb0`;
- tests: `tests/test_taska_retrieval_adapter_unique_supply.py`.

Weco Observe pair+exact step `132`, parent `102`. Steps `133/134` не
логировались, потому что held/fresh корректно не выполнялись.

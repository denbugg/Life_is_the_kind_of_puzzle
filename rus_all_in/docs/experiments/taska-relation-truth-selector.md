# TASKA relation-level truth selector

Дата: 2026-08-31. Статус: **formally confirmed for pair metric**.

## Что изменено

Предыдущий six-arm learned selector сводил каждый layout к 16 aggregate
признакам и обучал additive Ridge. Он дал `-0.844` pairs на local OOF. Новый
selector не фильтрует tail и не строит новый layout: для каждого из шести уже
готовых post-tail layouts он размечает **все 1,104 realised seams**, оценивает
вероятность правильности каждого seam и выбирает ровно один целый strict arm по
сумме ожидаемых правильных пар.

Inference-visible relation features:

- raw outgoing/incoming ranks, margins, signed best gaps and z-scores;
- axis, pre/post six-arm support, survival/creation by tail;
- focal supply membership/logit and current/selective/unique-fullres provenance;
- whether the relation occurs in the confirmed control;
- fixed control/arm identity. Arm identity is provenance of six fixed solver
  mechanisms, not target information.

Единственная заранее зафиксированная модель —
`HistGradientBoostingClassifier`: learning rate `0.05`, `160` iterations,
`31` leaves, `min_samples_leaf=128`, `L2=1`, без early stopping. Ни параметр,
ни feature subset, ни score threshold не подбирались. Все arms независимо
получают один и тот же focal-logit-`>=0` protected non-adjacent tail96. При
точном score tie остаётся confirmed control.

Development contract SHA-256:
`5fac92c3a2c6c562f18a6e38065d1d2dcc13131a74e29ab5cf079213d1b6bacd`.

## Development separability

Local32 использовался для первого fit; held32 был source-disjoint signal gate.
После его прохождения модель один раз refit-нулась на local32+held32 и была
заморожена до existing fresh32. Fresh32 исторически model-selection-exposed и
не выдаётся за confirmation.

| Panel | Control pairs / exact | Selector pairs / exact | Pair delta, source CI95 | W/T/L | Edge ROC-AUC |
|---|---:|---:|---:|---:|---:|
| local32 in-sample | `326.781/5.938` | `326.750/5.969` | `-0.031 [-0.625,+0.594]` | `2/28/2` | `0.9746` |
| held32 | `345.312/1.906` | `350.281/1.531` | **`+4.969 [+1.656,+8.781]`** | `6/26/0` | `0.9650` |
| fresh32 development | `355.625/0.938` | `358.781/0.812` | **`+3.156 [+0.188,+6.345]`** | `10/16/6` | `0.9691` |

Fresh eligibility gate (`pairs >= +0.5`, `exact >= -1`) прошёл. Frozen final
model SHA-256:
`ec4eca99243cdc6be20104d789b9e5d5598b79fa0d1b7e69bc37314375ad8c6b`.

## Formal source-disjoint confirmation

До генерации dirty cases и scoring были подписаны:

- complete semantic explicit-source exclusion snapshot: `2,054` уже
  использованных sources; SHA-256
  `46d3acccc593998b4afcdf0e53459649aa8cd17cfbf0145631162b6ebca4661f`;
- exact source16 x draw2 roster from organizer-train `img_006400..006999`, with
  `227/483` train sources excluded and zero collisions;
- confirmation config SHA-256
  `3d903eb595d1c0d152a8b53c7c9fa578b5b012227eeb03ab629a7dd24d5ce4e9`;
- gate: pair mean `>= +1`, source-bootstrap CI95 lower `>= 0`, exact mean
  `>= -1`, strict permutations `32/32`.

Для каждого case один unchanged selective-target500 + unique-fullres parent
inference создал шесть pre-tail arms и matcher evidence. Затем каждый arm
независимо прошёл tail96. До reconstruction references в одном SHA-frozen
archive были записаны все шесть post-tail layouts, relation features, expected
scores и final whole-arm choice. Только после freeze были восстановлены exact
synthetic references.

| Metric | Confirmed fusion | Relation selector | Delta, source CI95 | Case W/T/L | Source W/T/L |
|---|---:|---:|---:|---:|---:|
| Satisfied pairs | `332.21875` | **`338.06250`** | **`+5.84375 [+3.000,+9.12578]`** | **`13/19/0`** | **`11/5/0`** |
| Adjacency recall | `30.0923%` | **`30.6216%`** | **`+0.5293 pp [+0.2717,+0.8322]`** | `13/19/0` | `11/5/0` |
| Exact tiles | `1.21875` | `1.06250` | `-0.15625 [-0.40625,+0.06250]` | `2/25/5` | `2/10/4` |

Selector изменил control на `15/32` cases. Не было ни одного pair loss; все
`32/32` outputs — strict permutations 576 исходных upright tiles. Formal gate
прошёл полностью. Это новый подтверждённый pair-solver component; exact остаётся
secondary trade-off и не следует переименовывать в exact improvement.

### Descriptive distance/SSIM bridge

Отдельный post-hoc replay на **этом же frozen archive** добавил новую metric
view без повторного inference, отбора или gate. `64/64` control/candidate
layouts strict, а все pair/exact counts точно совпали с formal report.

| Mean по 32 | Control | Selector | Delta |
|---|---:|---:|---:|
| absolute mean Manhattan, cells | `14.9034` | **`14.7269`** | **`-0.1765`** |
| radius0 / exact recall | `.2116%` | `.1845%` | `-0.0271 pp` |
| radius2 recall | `4.0907%` | **`5.3331%`** | **`+1.2424 pp`** |
| clean layout SSIM | `.17324` | **`.17471`** | **`+.00148`** |
| dirty SSIM | `.10633` | **`.10748`** | **`+.00115`** |
| restored h20 SSIM | `.24887` | **`.24973`** | **`+.00085`** |

То есть pair gain сопровождается лучшим smooth position signal, radius2 и
всеми тремя SSIM views, но не radius0/exact. SSIM здесь только evaluation на
organizer-train targets после freeze; это не новое formal confirmation и не
проверка competition score. Полный bridge report:
`outputs/tile-position-distance-validation/relation-selector-bridge-v1/report.json`,
SHA-256
`2f14336e91ca889e9c8777f90ee596a7f390cfeacb7a82378a140b42a9781104`.

## Legality и границы вывода

- denoised full-resolution view использован только matcher-ом до solver;
- output не меняет pixels, orientation или состав tiles;
- relation model не видит target/reference при inference;
- официальный production/default, competition test и submission не затрагивались;
- confirmation проверяет переносимость на organizer-train synthetic cases, а не
  официальный leaderboard SSIM.

После confirmation добавлен отдельный production-ready layout adapter
`aiijc-taska-relation-selector`. Он до deserialization fail-closed SHA-gate-ит
frozen six-arm parent, relation model, development/confirmation configs,
reports и target-free archive; принимает только original
`uint8[576,20,20,3]` tile bag и записывает только strict `int32[576]` layout.
Restored full-resolution pixels остаются matcher-only и не могут попасть в
output. CLI не открывает model/arm/threshold/top-k/tail tuning и не перезаписывает
существующие файлы. End-to-end MPS replay formal case 0 совпал побитово; layout
SHA-256 `96d4349fb75a4fd6e3ad38c9ae1d5820700d030339a63141c3d74b80a4e1bd66`.

```bash
uv run aiijc-taska-relation-selector tiles.npy \
  --output-layout layout.npy \
  --diagnostics-json receipt.json \
  --device mps
```

Не повторять aggregate Ridge, тот же HGB parameter/threshold/feature sweep на
открытых panels или post-hoc exact guard. Следующий materially distinct consumer
может использовать complementary adapter/DINO candidate supply, но DINO direct
R@1/reciprocal precision слишком низки для прямой замены scorer-а; это отдельная
заранее подписанная verifier/fusion ветка, не изменение подтверждённой модели.
Безусловный [HGB-ranked union всех realised relations](taska-relation-ranked-union.md)
уже проверен отдельно и резко провалил local pair gate (`-127.25` pairs):
relation AUC нельзя превращать в all-edge decoder без joint compatibility.

## Артефакты

- development report:
  `outputs/taska-relation-truth-selector/fixed-v1/report.json`, SHA-256
  `022739dec8a47465f588a3ad9e45660ffbfa327a6f1647bd1517134c01420c39`;
- confirmation report:
  `outputs/taska-relation-truth-selector/formal-confirmation-v1/report.json`,
  SHA-256
  `d260872251077e1515251b6c7afc316af25df75045c8119112dff4f36c68ea23`;
- frozen confirmation archive SHA-256
  `4cd0346333813cea3576f6db40ea517dcc45fdd5aa81a432a351cf4afdd73131`;
- pre-score freeze SHA-256
  `97a6d2344669ff6f18ece5085e001de3b6b1f04db89e3154510df42b501757b7`;
- source: `src/aiijc_puzzle/taska_relation_truth_selector.py`;
- production adapter:
  `src/aiijc_puzzle/taska_relation_selector_pipeline.py`, SHA-256
  `1020ebc28777ba02872a82613bbb433d802e9e2b3e6fc04a5cbd2b81e49e7976`;
- production CLI shim:
  `scripts/run_taska_relation_selector_pipeline.py`, SHA-256
  `c43e0941c45c74d69ed5bfe13c4920b376e34c1934e139be6af272e9a1b119cc`;
- runners: `scripts/run_taska_relation_truth_selector.py` and
  `scripts/run_taska_relation_truth_selector_confirmation.py`;
- post-hoc metric bridge runner:
  `scripts/describe_relation_selector_distance_bridge.py`;
- tests: relation-model + production-adapter suites (`7 passed`), including
  the real formal case-0 end-to-end replay;
- Weco Observe pair+exact: steps `141` held, `142` fresh development, `143`
  formal confirmation; parent step `102`.

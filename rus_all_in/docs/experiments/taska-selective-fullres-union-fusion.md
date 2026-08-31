# TASKA selective target500 + unique fullres union fusion

Дата фиксации: 2026-08-31.

## Вердикт

Один заранее фиксированный replay успешно объединил два уже подтверждённых
candidate-supply источника, дал новый pair-oriented leader на существующих
local/held/fresh panels и затем **прошёл отдельное preregistered source16×draw2
подтверждение**. По отношению к точному frozen selective-target500 control
combined candidate получил:

| Panel | Control pairs / exact | Fusion pairs / exact | Pair delta, source CI95 | Exact delta, source CI95 |
|---|---:|---:|---:|---:|
| local32 | `323.625 / 1.563` | `326.781 / 5.938` | `+3.156 [-0.063,6.938]` | `+4.375 [-0.031,11.000]` |
| held32 | `343.094 / 1.563` | `345.313 / 1.906` | `+2.219 [-0.219,5.094]` | `+0.344 [-0.031,0.875]` |
| fresh32 | `354.094 / 0.969` | **`355.625 / 0.938`** | **`+1.531 [0.125,3.094]`** | `-0.031 [-0.094,0.000]` |

Local gate `pair mean >=0` и held gate `pair mean >=+0.5` выполнены, поэтому
fresh32 был открыт. На fresh pair W/T/L равен **`5/27/0`**: combined arm
изменил только шесть boards, улучшил пять и ни один не ухудшил. Fresh
source-bootstrap lower bound тоже положителен. Exact остаётся secondary:
local-выигрыш создаётся несколькими шумными boards и его CI пересекает ноль,
а fresh знак слегка отрицателен. Production/default и submission этим
экспериментом не менялись.

## Формальное независимое подтверждение

До нового inference был подписан config
`configs/taska_selective_fullres_union_fusion_fresh32_confirmation_v1.json`
(SHA-256 `11b713d0475306d8e1e1397f8563132d74ef5b8957e85e1e58a5e4f57f018190`).
Его 16 sources × 2 draws не пересекаются с историческими TASKA panels,
selective formal confirmation, fullres confirmation или tail192 reservation.
Ни один threshold, support, budget, arm или roster не подбирался.

| Metric | Frozen selective control | Confirmed fusion | Delta, source CI95 |
|---|---:|---:|---:|
| satisfied pairs | `330.03125` | **`333.12500`** | **`+3.09375 [0.84375,5.75000]`** |
| adjacency recall | `0.298941350` | **`0.301743659`** | `+0.002802310 [0.000764,0.005208]` |
| exact tiles | `0.71875` | `1.56250` | `+0.84375 [-0.09375,2.18750]` |

Preregistered pair gate требовал mean `>=+1.0` и CI95 lower `>=0`; обе части
выполнены. Case W/T/L `7/23/2`, source W/T/L `6/9/1`. Механический selective
final control совпал на `32/32` boards. Unique fullres supply добавил в среднем
`12.0625` edges, из них `6.0` true (`49.74%` precision); combined arm выиграл
selector на `9/32` boards. Это формально подтверждает fusion как pair-oriented
solver. Exact знак положительный, но остаётся secondary из-за CI через ноль.

## Fixed contract

Matcher и denoiser не запускались повторно. Для каждого board были
SHA-проверены и identity-aligned:

- current four layouts и исходные dense `cost_right/cost_down`;
- frozen selective-target500 current/accepted edges и focal logits;
- frozen fullres accepted edges и соответствующие proposal focal logits;
- source filename, draw index, dirty SHA и row prefix во всех lineage caches.

Дальше применялся ровно один candidate:

1. `selective_union = current + selective_accepted` в исходном frozen order;
2. из fullres accepted удалялись все edges, уже присутствующие в current или
   selective accepted;
3. оставшиеся fullres edges с выровненными logits добавлялись в их frozen
   порядке: `combined = current + selective + unique_fullres`;
4. selective union arm был механически пересобран frozen raw solver-ом;
5. его five-arm selector и focal-gated tail96 обязаны были побитово
   воспроизвести frozen selective final control;
6. единственный candidate selector roster был
   `raw/logistic/focal_top5/nonlinear/selective_vote500_focal/combined_union_focal`;
7. standalone fullres arm намеренно не добавлялся, чтобы не создавать ещё один
   portfolio winner-choice;
8. selector использовал исходную сумму costs на всех 1,104 board bonds;
9. focal-gated tail96 защищал candidate set выбранного arm-а. Если combined
   не выигрывал, final candidate буквально совпадал с frozen control.

Threshold, support, arm roster, tail budget и gates не подбирались. Fullres
родитель уже зафиксировал `support>=3/4` и focal logit `>=0`; selective
родитель уже зафиксировал target500-minus-current и focal logit `>=0`.

## Identity и control audit

- layout/base/selective/fullres row identity: `96/96`;
- current edge order selective/base/fullres: `96/96`;
- selective selector choice replay: `96/96`;
- exact final selective control replay: `96/96`;
- все candidate/control layouts — строгие перестановки `0..575` исходных
  upright fragments;
- frozen raw solver SHA-256:
  `97859e1f4ff6ceadf56ebccf429f2944ae7a5726631731d9f776224a667eb486`.

Held/fresh historical focal caches имеют небольшой ожидаемый MPS numerical
drift, поэтому replay корректно использовал именно focal logits из selective
freeze. Это не повлияло на identity edges; механический final control всё
равно совпал побитово на всех 96 boards.

## Overlap и действительно новый supply

Overlap/unique rosters были записаны в target-free archive до восстановления
references. Только после pre-score freeze были посчитаны true-edge counts:

| Panel | Fullres accepted / board | Overlap selective / board | Unique fullres / board | Unique true / board | Unique precision |
|---|---:|---:|---:|---:|---:|
| local32 | `26.188` | `14.469` | `11.719` | `5.281` | `45.07%` |
| held32 | `32.875` | `18.063` | `14.813` | `7.781` | `52.53%` |
| fresh32 | `29.250` | `16.563` | `12.688` | `6.031` | `47.54%` |

Fullres overlap с current равен нулю, как и требует parent contract. Более
половины fullres accepted уже встречались среди selective accepted; precision
этого overlap высокая: `66.95% / 65.74% / 67.36%`. Unique остаток слабее, но
всё ещё добавляет в среднем `5.28 / 7.78 / 6.03` новых true edges. Combined
candidate-supply recall local/held/fresh составил
`0.256369 / 0.271230 / 0.280231`.

Combined arm выиграл selector на `9/32`, `8/32`, `6/32` boards. Final pair
W/T/L local/held/fresh: `8/23/1`, `5/24/3`, `5/27/0`. Это показывает, что
новый сигнал — complementary accepted-edge supply, а не ещё один порог или
budget sweep.

## Freeze, legality и воспроизведение

Каждый panel сначала создал `frozen-target-free-eval.npz`, target-free JSON и
`pre-score-freeze.json`; только потом были восстановлены organizer-train
references для offline pair/exact оценки. Competition test не открывался.
Пиксели не денойзились повторно, не заменялись, не поворачивались, не
деформировались и не postprocess-ились.

Запуск:

```bash
.venv/bin/python scripts/run_taska_selective_fullres_fusion.py
```

Артефакты:

- report:
  `outputs/taska-selective-fullres-union-fusion/fixed-v1/report.json`, SHA-256
  `1f9d84c99eae6ba1f03a668163f6e19321e20292e31dd5e51ec00282587517af`;
- local NPZ / metadata / pre-score freeze:
  `1b17c4a5...f7df / 106ac31d...efa1 / 3b35db32...9720`;
- held NPZ / metadata / pre-score freeze:
  `6cfb766c...e469 / f37d23bd...0bf4 / aa5b53ab...78bc`;
- fresh NPZ / metadata / pre-score freeze:
  `75a9359e...0cb / c65d7e33...6794 / 7fca88f9...5e93`;
- module SHA-256:
  `13ba0e8f5c09c84dfef8c25711805e334a7afd5f0e9e80db749415f566ed6348`;
- runner SHA-256:
  `173a9f44870a4e60b2e5199ff28e03e42afa0bf34d8fe970806f762e2dba4ab5`.

Tests:
`tests/test_taska_selective_fullres_fusion.py` и
`tests/test_run_taska_selective_fullres_fusion.py`.

Weco Observe pair+exact: steps `95/96/97` for local/held/fresh, lineage
`92 -> 95 -> 96 -> 97`.

Formal confirmation artifacts:

- report:
  `outputs/taska-selective-fullres-union-fusion/fresh32-formal-confirmation-v1/report.json`,
  SHA-256 `4d0ea850e101cb56a4f70dc6ff164201c09af047dcb669e3c81e19488661e555`;
- target-free NPZ / metadata / pre-score freeze:
  `1cbc3b38...43e0 / 2b66d8ff...1a83 / 0a213c08...43c5`;
- confirmation runner SHA-256:
  `f38356446da7283d42fb8c14ce2c024a74c5f57a1c9133ca3f107f54bdd5a654`;
- Weco Observe pair+exact: step `102`, parent `97`.

## Решение

**Confirmed** как новый ведущий pair-oriented TASKA solver. Не подбирать
support, focal threshold, vote target, selector roster или tail budget на уже
открытых данных. Production/default и submission автоматически не менялись.

Отдельный layout-only adapter добавлен как
`src/aiijc_puzzle/taska_best_pair_fusion_pipeline.py` и CLI
`aiijc-taska-best-pair-fusion`. Он SHA-gate-ит selective/fusion/fullres/raw
sources, все pair model artifacts, denoiser checkpoint, parent report,
confirmation config и confirmation report до deserialization/inference.
Пиксели restored view используются только matcher-ом; output — только строгая
перестановка 576 исходных upright tile ids. Старый `aiijc-taska-best-pair`
сохранён как отдельный selective fallback, legacy и official best не заменены.

```bash
uv run aiijc-taska-best-pair-fusion tiles.npy \
  --output-layout layout.npy --diagnostics-json receipt.json
```

Frozen confirmation case 0 был воспроизведён end-to-end побитово. Adapter / CLI
shim / test SHA-256:
`2760708d...e0c / 4daaefb9...42f / 81350695...319`.

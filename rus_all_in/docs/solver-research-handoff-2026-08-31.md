# Handoff исследования solver-а — 2026-08-31

Этот handoff продолжает
[снимок 2026-08-30](solver-research-handoff-2026-08-30.md) и фиксирует новый
TASКA-трек: сначала измеряем правильные соседние пары, затем exact placement.
Подробные протоколы и no-repeat ledger находятся в
[experiments/README.md](experiments/README.md).

## Главное

- Лучший официальный submission **не менялся**: `fixed-B standard + buddies96`,
  score **`0.2762279116935955`**.
- Competition test в этом цикле не открывался; submission не собирался.
- Все новые layouts — строгие перестановки 576 исходных upright `20×20` tiles.
  Restored views используются только matcher-ом; пиксели output не заменяются,
  не поворачиваются и не деформируются.
- Pair-метрика — число правильно восстановленных горизонтальных и вертикальных
  соседств из **1104**; recall равен `pairs / 1104`.
- В Weco Observe теперь ведутся **два параллельных lineage**:
  - exact: run `c2876967-cca7-44a6-83dd-1fca125c237e`, metric
    `exact_tiles_per_board`;
  - pairs: run `6bf52932-d716-4959-bee4-d652d7286cba`, primary
    `satisfied_adjacent_pairs_per_board`, secondary `adjacency_recall`.
  Реестр лежит в `outputs/weco-observe/runs.json`.

## Подтверждённые результаты

### Selective target500 — production-ready pair solver

[Selective target500](experiments/taska-selective-vote500.md) делает один
matcher-pass с vote target 500, восстанавливает same-pass current350 и добавляет
только новые lower-vote edges с recovered focal top5 logit `>=0`. Один fifth
arm конкурирует с четырьмя прежними по исходному all-1104 seam cost; затем
применяется focal-gated tail96.

На прежних local/held/fresh панелях pair delta к same-pass control была
`+9.219/+5.000/+5.750`; fresh дал `354.094` пары, recall `0.320737`.

[Формальное независимое подтверждение](experiments/taska-selective-vote500-formal-confirmation.md)
на новом source-disjoint `16 sources × 2 draws` roster:

| Вариант | pairs | recall | exact |
|---|---:|---:|---:|
| same-pass current350 + focal tail96 | `348.78125` | `0.315925045` | `2.59375` |
| selective target500 + focal tail96 | **`354.28125`** | **`0.320906929`** | `2.81250` |

Pair delta **`+5.50000`**, source-bootstrap CI95
**`[+0.81250,+11.31250]`**. Exact delta `+0.21875`, CI пересекает ноль.
Gate `mean>=+2`, `CI lower>=0` пройден. Weco step `98` в обеих метриках.

После gate добавлен отдельный SHA-gated layout-only adapter
`src/aiijc_puzzle/taska_best_pair_pipeline.py` и CLI
`aiijc-taska-best-pair`. Legacy pipeline и официальный best не заменялись.

### Full-resolution denoiser supply + focal-gated tail

[Full-resolution voter](experiments/taska-fullres-union-voter.md) не заменяет
raw scores восстановленными. Stride-one NAF view только номинирует absent
edges; требуется support `>=3/4` restored scorers и focal logit `>=0`.

На исходных local/held/fresh панелях five-arm gain был
`+4.781/+4.219/+4.031` пары. Отдельная композиция с focal-gated tail имела
слабый marginal held-сигнал и поэтому не открывала прежний fresh panel.

[Новое независимое подтверждение](experiments/taska-fullres-focal-gated-tail-fresh32-confirmation.md)
было заранее зарегистрировано на другом source-disjoint `16×2` roster:

| Вариант | pairs | recall | exact |
|---|---:|---:|---:|
| four-arm control + focal tail96 | `348.40625` | `0.315585371` | `8.0` |
| fullres union + focal tail96 | **`356.31250`** | **`0.322746830`** | `8.0` |

Total pair delta **`+7.90625`**, source-CI95
**`[+3.53125,+12.96875]`**. Fullres arm дал `+5.4375`, а focal protection
добавила ещё `+2.46875`; обе нижние границы CI положительны. Exact нейтрален.
Gate пройден; Weco step `94` в обеих метриках.

### Selective + unique fullres fusion

[Fixed fusion](experiments/taska-selective-fullres-union-fusion.md) удаляет
overlap между selective и fullres supplies и добавляет только уникальные
fullres edges. Standalone fullres arm не добавляется: selector видит прежние
четыре arms, selective arm и один combined-union arm.

| Panel | selective control | fused candidate | pair delta |
|---|---:|---:|---:|
| local32 | `323.625` | `326.781` | `+3.156` |
| held32 | `343.094` | `345.313` | `+2.219` |
| fresh32 | `354.094` | **`355.625`** | **`+1.531`** |

На fresh source-CI95 равен **`[+0.125,+3.094]`**, W/T/L `5/27/0`; exact
delta `-0.03125`. Control replay совпал `96/96`. Weco steps `95→96→97`.

Формальное независимое подтверждение на новом preregistered `16×2` roster
также прошло gate:

| Вариант | pairs | recall | exact |
|---|---:|---:|---:|
| selective control | `330.03125` | `0.298941350` | `0.71875` |
| confirmed fusion | **`333.12500`** | **`0.301743659`** | `1.56250` |

Pair delta **`+3.09375`**, source-CI95 **`[+0.84375,+5.75000]`**,
W/T/L `7/23/2`; gate `mean>=+1`, `CI lower>=0` пройден. Exact delta
`+0.84375`, но CI пересекает ноль. Unique fullres supply добавлял в среднем
`6.0` правильных ребра/board с precision `49.74%`. Weco step `102`.

После confirmation добавлен отдельный SHA-gated layout-only adapter
`src/aiijc_puzzle/taska_best_pair_fusion_pipeline.py` и CLI
`aiijc-taska-best-pair-fusion`. Он gate-ит solver sources, parent/config/report,
все TASKA model resources и fullres denoiser; frozen confirmation case replay
совпал bit-for-bit. Selective fallback и legacy pipeline не менялись.

### Relation-level whole-arm selector

[Confirmed relation selector](experiments/taska-relation-truth-selector.md)
оценивает все `1,104` realised seams каждого из шести frozen post-tail arms
одним fixed HGB и выбирает whole layout по сумме expected-correct probabilities.
На новой заранее подписанной source16×draw2 панели он поднял confirmed fusion
`332.219→338.063` pairs/board: delta **`+5.844`**, source-CI95
**`[+3.000,+9.126]`**, W/T/L `13/19/0`; exact delta `-0.156`. Все `32/32`
layout — strict permutations. Это текущий подтверждённый pair leader, не
leaderboard-SSIM promotion.

Добавлен отдельный fail-closed layout-only adapter/CLI
`aiijc-taska-relation-selector`: SHA gates frozen six-arm parent,
model/config/report/evidence, target-free inference, denoised view matcher-only.
Formal case 0 воспроизведён end-to-end bit-for-bit. Official default, test,
submission и оба прежних fallback CLI не менялись.

## Закрытые ветки этого цикла

- [Incidence GNN](experiments/taska-incidence-gnn.md): local/held дали слабый
  плюс, fresh `-0.313` пары; не переносится.
- [Tail96→tail192](experiments/taska-focal-gated-tail192-capacity.md):
  `-0.3125` пары при `+54.47` swaps; больше оптимизации того же seam objective
  не помогает.
- Старый безусловный target500 supply остаётся отрицательным; работает именно
  selective focal-filtered consumer.
- Fixed fullres+focal replay на прежнем held panel не прошёл marginal gate на
  `0.09375`, но последующая независимая confirmation доказала общий эффект;
  не повторять nearby threshold/budget sweep.
- FullResolutionTwin-only supply поверх confirmed fusion дал local
  `+1.625` пары, CI95 `[+0.375,+3.125]`, но held перенёс только `+0.219`,
  CI95 `[-1.781,+2.625]`, ниже fixed gate `+0.5`; fresh не открывался.
  Retain как weak auxiliary signal, не подбирать top-k/focal threshold.
- DRUNet restored-descriptor unique supply провалился уже на local32:
  `326.781→323.625`, delta `-3.156`, CI95 `[-6.906,+0.125]`, exact
  `5.938→1.563`. Focal `>=0` оказался резко miscalibrated для этого OOD
  emitter-а: accepted precision лишь `0.397%`. Held/fresh не открывались;
  не ослаблять и не sweep-ить этот contract.
- Learned post-tail guard selective-vs-fusion остановлен до fit: oracle gain
  над fusion был лишь `+0.688/+0.469/0/+0.188` на
  local/held/fresh/formal, а local+held содержали только 4 selective-win против
  13 fusion-win и 47 ties. Preregistered минимум 8 примеров каждого класса не
  выполнен; Weco steps107/108 намеренно отсутствуют.
- Unique-fullres edge calibrator действительно поднял held/fresh precision
  `52.53→64.44%` и `47.54→69.51%`, но retaining лишь около 59% true edges
  оказалось слишком агрессивно. Pair delta была `+1.844` на held и `-0.688`
  на fresh; exact `-0.281/+0.063`. Formal confirmation не открывалась.
  Сохранять как calibration diagnostic, не заменять confirmed unfiltered
  fusion и не sweep-ить `C`/threshold/features на этих panels.
- Seven-arm confirmed portfolio добавил standalone fullres arm к шести fusion
  arms. Он дал `+1.000/+1.313/+1.000` пары на local/held/fresh и прошёл staged
  gates, но на новой preregistered `16×2` панели ухудшил
  `314.344→313.438`: delta `-0.906`, source-CI95 `[-2.750,+0.625]`, exact
  `-0.094`. Это ещё один реальный winner's-curse counterexample: не расширять
  подтверждённый six-arm roster без нового robust selector.

## Ближайшая очередь

1. Считать relation-level whole-arm selector текущим подтверждённым pair
   leader; использовать `aiijc-taska-relation-selector`, сохраняя
   `aiijc-taska-best-pair-fusion` и selective solver как отдельные fallbacks.
2. Не продолжать Twin/DRUNet nearby sweeps: обе bounded ветки закрыты на held
   или local gate.
3. Следующий matcher step должен давать materially new calibrated evidence,
   а следующий solver step — новый objective/search, не больше swaps того же
   original seam objective.
4. Exact/origin и leaderboard SSIM возвращать в приоритет только после
   стабилизации pair solver-а; один и тот же frozen restoration tail обязателен
   для честного layout A/B.

## Быстрое возобновление

1. Прочитать этот handoff и три linked positive experiment reports.
2. Проверить `outputs/weco-observe/runs.json` и последние Weco steps.
3. Использовать `aiijc-taska-relation-selector` как текущий SHA-gated pair
   leader, `aiijc-taska-best-pair-fusion` как confirmed fusion fallback и
   `aiijc-taska-best-pair` как selective fallback. Они не переписывают друг
   друга.
4. Не трогать competition test и не заменять официальный `0.2762279` best без
   отдельного compliant same-tail leaderboard результата.

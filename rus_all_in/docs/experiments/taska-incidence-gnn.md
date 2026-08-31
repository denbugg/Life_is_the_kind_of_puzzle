# Fixed TASKA incidence-GNN

Дата: 2026-08-31. Вердикт: `fresh pair gate failed; do not promote`.

## Гипотеза и отличие от уже проверенного

Прежние linear pairwise и HGB/logistic stacker-ы оценивали каждую harvested
relation независимо. Здесь проверен один заранее фиксированный
context-aware consumer: edge state видит конкурентов того же board и axis,
имеющих тот же outgoing source или incoming target. Это также не повторяет
Union-hard DeepSets: graph, candidate roster и downstream solver взяты из
текущего TASKA pipeline.

Модель использует только существующие candidate edges и 22 уже аудированных
target-free признака (`15 TASKA + recovered focal logit + 6 focal top-5`).
Архитектура зафиксирована до оценки:

- global StandardScaler по extension128;
- input MLP ширины 64;
- два permutation-equivariant incidence block-а;
- в каждом block-е mean/max по outgoing-source и incoming-target отдельно для
  right/down axis;
- zero-init head и residual `2*tanh(head)`, добавляемый к неизменному recovered
  focal logit.

Fit использовал только frozen train256 indices `128:256`, 48 100 candidate
edges и 34 170 positives. Local32 `96:128` полностью исключён по indices и
source filenames. Ровно 400 deterministic one-board AdamW steps:
`lr=3e-4`, `wd=1e-4`, seed `2026083184`; balanced per-board BCE плюс
`1e-3 * residual²`. Epoch/hyperparameter selection не было. Training loss
снизился с `0.45761` до `0.37382`.

## Solver contract

GNN меняет только priority уже harvested edges. Его strict layout добавляется
пятым arm к frozen `raw/logistic/focal_top5/nonlinear`; selector всё так же
минимизирует original all-1104-bond TASKA cost, затем выполняется unchanged
protected tail96. Все layout-ы были записаны и SHA-frozen до восстановления
evaluation reference.

Gate protocol:

1. local32: five-minus-four pairs `>=0` открывает held32;
2. held32: five-minus-four pairs `>=+0.5` открывает fresh32;
3. fresh32 — unchanged final confirmation; exact всегда secondary.

## Результат

| Panel | Standalone pairs / exact | Four-arm+tail96 | Five-arm+tail96 | Δ pairs | Δ exact |
|---|---:|---:|---:|---:|---:|
| local32 | `307.219 / 1.500` | `314.375 / 1.375` | `314.719 / 1.406` | `+0.344` | `+0.031` |
| held32 | `332.750 / 3.094` | `337.563 / 3.063` | `338.281 / 2.813` | `+0.719` | `−0.250` |
| fresh32 | `340.625 / 1.250` | `346.063 / 1.156` | `345.750 / 1.063` | `−0.313` | `−0.094` |

Local pair CI95 `[-1.125,+2.125]`; held `[-0.219,+1.875]`; fresh
`[-1.313,+0.625]`. Five-arm выбирал GNN на `6/32` boards на каждой панели.
Таким образом чувствительные local и held gates прошли, но оба headline
показателя на frozen fresh32 развернули знак. Модель не становится новым pair
default и не добавляется в production.

Это полезный, но слабый signal: текущая incidence competition иногда меняет
выбор arm, однако original-cost selector не отличает переносимый gain от
winner's curse. Не повторять nearby width/block/step/residual-bound sweep на
этих уже открытых panels. Следующий context-aware опыт должен менять либо
training objective так, чтобы он соответствовал realised component quality,
либо robust selector, и потребует новой confirmation roster.

## Воспроизводимость и legality

Команда:

```bash
.venv/bin/python scripts/run_taska_incidence_gnn.py
```

Основной report:
`outputs/taska-incidence-gnn/extension128-v1/report.json`.

Frozen artifacts:

- weights-only state SHA `1cd226b9...f051`;
- mean/scale SHA `80863e6a...73c3`;
- SHA-gated contract `65d52b62...a68`;
- raw solver сохранил SHA `97859e1f...486`.

Tests проверяют tile-relabeling equivariance, bounded residual, weights-only
round-trip/SHA refusal и strict 576-tile layouts. Competition test не читался,
пиксели не менялись, labels использовались только offline fit/scoring, каждый
выход — перестановка всех исходных upright 20×20 tiles.

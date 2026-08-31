# Conservative mutual-edge fusion для k16 ranker

## Решение

**Reject as tested. Confirmation, holdout, test и production integration не
запускать.** Новый target-free fusion действительно извлёк часть локального
signal из k16 ranker-а без регрессии полной замены: лучший conservative arm дал
`+0.005748` mean final SSIM и положительный adjacency CI. Однако absolute final
SSIM остался `0.247861 < 0.27`, а paired SSIM CI всё ещё пересёк ноль. Ни один
из пяти заранее зафиксированных arms не прошёл все пять gates.

## Что было новым

Предыдущий k16 experiment заменял bilateral scores на learned residual во всём
candidate union. Он повышал adjacency `0.032684 -> 0.062689`, но ухудшал h20
endpoint на `-0.009386`. Здесь реализован **non-destructive mutual-edge union**:

1. frozen k16 model предлагает только learned row/column mutual-best edges с
   положительным residual;
2. source и target предложения не должны быть заняты ни одним bilateral
   mutual-best edge того же направления;
3. confidence равен минимуму нормированных row/column top-one margins;
4. можно требовать corroboration top-4 от четырёх dirty-only views
   (`raw`, `tile_z`, `bilateral`, `gray`);
5. не более фиксированного числа предложений получают минимальный float32 score,
   строго превышающий текущие row/column maxima;
6. проверяемый инвариант гарантирует: каждый исходный bilateral mutual-best edge
   остаётся mutual-best после fusion.

Это не простая E2/E14 score mixture и не full k16 replacement. Метод не принимает
target, абсолютные позиции, внешние изображения или данные другой test-board.
Дальше применяется прежний `buddies96`, строгая перестановка 576 upright tiles,
RGB seam offsets, bounded luma и ровно один coloured NLM `h=20`.

## Train-only выбор arms

Checkpoint fit использовал shared-selector `train[0:256]`. Пороговая диагностика
проводилась только на disjoint `train[256:280]`; calibration/holdout targets для
выбора arms не читались. Baseline final SSIM был `0.313945`.

| Arm | Новых edges | View gate | Confidence | Adjacency delta | Final SSIM delta |
|---|---:|---:|---:|---:|---:|
| `cap32-v2-c000` | 32 | >=2 | 0.0 | **+0.004944** | **+0.005723** |
| `cap08-v0-c050` | 8 | >=0 | 0.5 | +0.002227 | +0.004086 |
| `cap08-v0-c000` | 8 | >=0 | 0.0 | +0.002264 | +0.003978 |
| `cap16-v2-c000` | 16 | >=2 | 0.0 | +0.003736 | +0.003583 |
| `cap16-v2-c050` | <=16 | >=2 | 0.5 | +0.003359 | +0.003009 |

Эти пять arms и единственный winner rule были зафиксированы в
`configs/edge_ranker_conservative_fusion_preregistered_v1.json`, SHA-256
`6acf7d813cc1609b10bbff8651cef447bbe9f8bb1771551d966bdf895690614f`.

## Leakage boundary

Calibration нельзя называть globally fresh: старый
`outputs/legacy-upgrade/calibration700-champion/report.json` уже перечислял весь
calibration manifest. Preregistration явно фиксирует этот historical exposure.
Primary `360:384` и условный confirmation `444:468` disjoint только между собой.

Для текущего primary run все right/down score matrices, layouts, raw assemblies,
RGB+luma и NLM20 images были вычислены из inputs, заморожены в памяти и
content-addressed в commitment до чтения target bytes. Первый phase-one attempt
fail-closed до target access, потому что image-only hasher отказался принимать
float32 score matrices. Был добавлен только deterministic
`dtype + shape + bytes` matrix digest, после чего весь freeze повторён; immutable
preregistration и fusion policy не менялись.

Commitment self-hash:
`7a30164ec4eb72effa28697985fa2b966e6ec707f7e809d4bac11af2cb2eb173`.
Все `24 boards x 6 variants = 144` raw audits подтвердили точную перестановку,
576 уникальных исходных tiles и сохранение raw pixels.

## Frozen primary gate

Каждый arm обязан был одновременно выполнить:

1. mean final SSIM delta `>= +0.004`;
2. absolute mean final SSIM `>= 0.27`;
3. paired final-SSIM bootstrap CI95 lower `> 0`;
4. paired adjacency bootstrap CI95 lower `>= 0`;
5. mean translation-aligned placement delta `>= 0`.

Использованы 20 000 paired percentile-bootstrap replicates и заранее заданные
seeds. Winner выбирался только среди полностью прошедших arms по final SSIM,
затем adjacency и preregistered priority.

## Результат primary calibration 360:384

| Arm | Final SSIM | Delta | SSIM CI95 | Adjacency | Adj. delta CI95 lower | Translation delta | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| bilateral baseline | 0.242113 | — | — | 0.037704 | — | — | control |
| `cap32-v2-c000` | 0.244347 | +0.002235 | [-0.002588,+0.007499] | **0.043893** | +0.003963 | +0.000940 | FAIL |
| `cap08-v0-c050` | **0.247861** | **+0.005748** | [-0.000221,+0.012380] | 0.041289 | +0.001585 | **+0.001013** | FAIL |
| `cap08-v0-c000` | **0.247861** | **+0.005748** | [-0.000246,+0.012637] | 0.041289 | +0.001585 | **+0.001013** | FAIL |
| `cap16-v2-c000` | 0.245068 | +0.002956 | [-0.001858,+0.007864] | 0.041742 | +0.001774 | +0.000289 | FAIL |
| `cap16-v2-c050` | 0.245947 | +0.003834 | [-0.001187,+0.008898] | 0.041289 | +0.001585 | +0.000217 | FAIL |

У всех arms adjacency CI и mean translation delta неотрицательны: заявленный
non-destructive local signal подтвердился. Но ни один arm не достиг `0.27`, ни у
одного SSIM CI lower не стал положительным. `cap08` прошёл только mean-gain,
adjacency и translation условия, но провалил absolute и CI gates.

Manual sheet подтверждает ограничение: sparse union иногда меняет крупные
цветовые полосы, но все 24 outputs остаются явными мозаиками, а цельные лица и
сцены не восстанавливаются. Метод законен структурно, но не достигает manual
review качества.

## Артефакты и воспроизведение

```bash
uv run python scripts/run_edge_ranker_conservative_fusion_train_diagnostic.py
uv run python scripts/run_edge_ranker_conservative_fusion.py --mode primary --device mps
```

Authoritative artifacts:

- train diagnostic `outputs/edge-ranker/conservative-fusion-train-diagnostic-256-280/report.json`,
  SHA-256 `2abfff1ff6ac681b2806815d9986767502f8be07cbf2bbf3079db0cf65e62021`;
- primary report `outputs/edge-ranker/conservative-fusion-k16/primary-cal360-count24/report.json`,
  SHA-256 `461409da22e60cf8d41abaeb65c3ba48c887cf8bc187ebb4e451ccc40768c6e8`;
- prediction commitment, file SHA-256
  `4da4c405bee312ca6b374d80c4fc8977ad5c362c2fdf52c97a46107d4c154cd0`;
- manual sheet, SHA-256
  `145ea2f69f1d2011358d2d5f1ebcfb81d555444adf4477c084cf0d688c79df76`.

`--mode confirmation` проверен fail-closed и завершается до создания output dir:
primary winner равен `null`. Поэтому confirmation, holdout, test и frozen
production не открывались и не изменялись.

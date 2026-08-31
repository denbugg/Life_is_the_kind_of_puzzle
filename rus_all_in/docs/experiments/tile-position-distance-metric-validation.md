# Tile-to-true-position distance as a solver metric

Дата: 2026-08-31. Статус: **validated as a useful smooth secondary metric;
not a replacement for absolute exact placement**.

## Вопрос и preregistration

Нужно было проверить, даёт ли расстояние от каждого tile до true cell более
информативный solver progress signal, чем только exact match, и связано ли оно
с реальным SSIM. До любого scoring был подписан один fixed protocol:

- config: `configs/tile_position_distance_metric_validation_v1.json`;
- SHA-256: `635c4c6e437c9173d0dbf37cbe4f09f4e613f39d487ae1d1ea9a2e35e69e1622`;
- только уже открытые organizer-train local32 sources;
- 8 frozen solver layouts + 11 deterministic reference controls + 5
  deterministic perturbations confirmed control = `24 layouts/source`, всего
  `768` rows;
- сначала materialize/freeze всех strict layouts, затем считать metrics/SSIM;
- никакого model, threshold, roster или perturbation sweep;
- terminal/held/fresh/competition test и Weco steps `149–154` не открывать.

Candidate archive был записан и hash-frozen до первого distance/SSIM score.
Reference-based perturbations — явно target-assisted metric controls, а не
deployable solver outputs.

## Точные определения

Обе layout задаются как `tile_at_position`. Для каждого tile identity
инвертируются predicted и exact layouts и вычисляются ошибки в клетках сетки:

- mean, median и p90 (`method=higher`) Manhattan;
- normalized mean L1 = `mean Manhattan / 46`;
- mean Euclidean;
- recall within Manhattan radius `0/1/2`;
- radius0 в точности равен exact tile recall, рядом хранится integer exact
  tile count.

Отдельно, с явным именем `cyclic_aligned`, перебираются все 576 whole-board
cyclic rolls и выбирается минимальный absolute mean Manhattan. Zero-roll —
incumbent, tie-break stable row-major. После выбранного roll считается та же
полная metric suite. Этот вариант специально удаляет global-origin error и
поэтому не может подменять absolute metric.

Reusable implementation:
`src/aiijc_puzzle/tile_position_distance.py`.

## Что сравнивалось

Для каждого из 768 layouts считались три RGB SSIM одним contest-compatible
`channel_axis=2, data_range=255` implementation:

1. `layout_only_clean`: pristine tiles в shuffled identity order, собранные
   данным layout;
2. `production_like_dirty`: original corrupted shuffled tiles;
3. `production_like_restored_h20`: ровно один frozen historical
   RGB/luminance/colored-NLM `h20` pass после assembly.

Correlations были зафиксированы заранее в четырёх views: pooled, within-source
centered, distribution per-source и means 24 layout families. Для distance
headline знак развёрнут, поэтому в таблицах higher correlation = better
geometry/SSIM agreement.

## Главный результат

### Pooled Pearson / Spearman

| Absolute metric, quality-oriented | Clean layout | Dirty | Restored h20 |
|---|---:|---:|---:|
| mean Manhattan | `.786 / .787` | `.739 / .764` | `.752 / .728` |
| median Manhattan | `.761 / .800` | `.713 / .777` | `.735 / .745` |
| p90 Manhattan | `.754 / .734` | `.704 / .717` | `.717 / .665` |
| mean Euclidean | **`.802 / .792`** | **`.756 / .769`** | **`.767 / .736`** |
| radius0 / exact | **`.943 / .451`** | **`.951 / .475`** | **`.836 / .463`** |
| radius1 | `.787 / .715` | `.747 / .719` | `.745 / .667` |
| radius2 | `.755 / .789` | `.705 / .768` | `.724 / .732` |
| cyclic-aligned mean Manhattan | `.711 / .629` | `.656 / .567` | `.683 / .537` |

### Within-source centered Pearson / Spearman

| Metric | Clean layout | Dirty | Restored h20 |
|---|---:|---:|---:|
| absolute mean Manhattan | `.804 / .868` | `.745 / .863` | `.828 / .863` |
| absolute mean Euclidean | `.818 / .871` | `.761 / .868` | `.842 / .866` |
| radius0 / exact | **`.965 / .412`** | **`.958 / .404`** | **`.927 / .393`** |
| radius2 | `.771 / .849` | `.709 / .830` | `.798 / .836` |
| cyclic-aligned mean Manhattan | `.731 / .761` | `.664 / .753` | `.759 / .764` |

Per-source result переносится на все 32 sources: для absolute mean Manhattan
median Spearman равен `.867/.869/.863` для clean/dirty/h20 соответственно.
Family-mean Spearman ещё выше: `.953/.954/.956`.

Интерпретация не “distance победил exact”, а более тонкая:

- exact/radius0 имеет самый сильный **linear Pearson** relation, особенно на
  controlled layouts, где доля совершенно правильных pixels напрямую
  определяет SSIM;
- exact имеет слабый **rank Spearman**, потому что на сложных solver layouts
  почти все варианты имеют очень мало exact tiles и много ties;
- mean Manhattan/Euclidean и radius2 дают гораздо более гладкий ranking и
  устойчиво связаны со всеми тремя SSIM views;
- normalized mean L1 корреляционно полностью избыточен mean Manhattan — это
  только удобный scale `[0,1]`;
- p90 слабее mean/median и годится как tail diagnostic, не headline.

Mean Euclidean численно чуть сильнее Manhattan, но разница мала. Manhattan
лучше подходит основным secondary metric: он целочисленно интерпретируем,
раскладывается по row/column и согласован с radius1/2.

## Почему cyclic alignment нельзя использовать как primary

Контролируемый pure origin error показывает проблему напрямую:

| Variant, mean по 32 | Absolute L1 | Aligned L1 | Exact recall | Clean SSIM | h20 SSIM |
|---|---:|---:|---:|---:|---:|
| exact reference | `0.000` | `0.000` | `1.000` | `1.000` | `.684` |
| exact board, row roll `+1` | `1.917` | **`0.000`** | `0.000` | `.385` | `.395` |
| exact board, diagonal `+1,+1` | `3.833` | **`0.000`** | `0.000` | `.302` | `.338` |

Aligned metric объявляет обе сильно испорченные картинки perfect. Поэтому её
correlation хуже absolute во всех SSIM views.

Тот же вывод виден на реальном exact-направлении:

| Frozen solver | Absolute L1 | Aligned L1 | Exact recall | radius2 | Clean SSIM | h20 SSIM |
|---|---:|---:|---:|---:|---:|---:|
| confirmed six-arm | `14.426` | `12.629` | `.0103` | `.0443` | `.2066` | `.2763` |
| + Socket cyclic border5 | `14.469` | `12.629` | **`.0224`** | **`.0481`** | **`.2109`** | **`.2776`** |

Socket roll больше чем удвоил exact recall и слегка поднял SSIM, но absolute
mean L1 чуть ухудшился, а aligned L1 полностью связал два layouts. Это
конкретный counterexample против замены exact одним средним distance и против
использования aligned distance для promotion.

## Post-hoc bridge к formal relation-selector

После formal source-disjoint confirmation pair-лидера был сделан отдельный
descriptive replay **только двух уже frozen layouts** на тех же 32 cases. Это не
новый gate, selector, inference или панель. Все `64/64` layouts оказались strict,
а pair/exact counts побитово воспроизвели formal report; targets были открыты
только после freeze для расчёта метрик и evaluation-only SSIM.

| Metric, mean по 32 | Confirmed six-arm | Relation selector | Delta |
|---|---:|---:|---:|
| satisfied pairs | `332.2188` | **`338.0625`** | **`+5.8438`** |
| absolute mean Manhattan | `14.9034` | **`14.7269`** | **`-0.1765`** |
| radius0 / exact recall | `.2116%` | `.1845%` | `-0.0271 pp` |
| radius2 recall | `4.0907%` | **`5.3331%`** | **`+1.2424 pp`** |
| clean layout SSIM | `.17324` | **`.17471`** | **`+.00148`** |
| dirty SSIM | `.10633` | **`.10748`** | **`+.00115`** |
| restored h20 SSIM | `.24887` | **`.24973`** | **`+.00085`** |

На 15 реально изменённых cases эффект ожидаемо больше: pairs `+12.467`, mean
Manhattan `-0.377`, radius2 `+2.650 pp`, clean/dirty/h20 SSIM
`+.00315/+.00245/+.00182`; exact при этом `-0.333 tile`. Это полезная
согласованность независимых diagnostics: pair selector не просто увеличил
локальные связи, но в среднем приблизил tiles и слегка поднял все три SSIM
views. Однако exact trade-off остаётся настоящим, а post-hoc SSIM не является
новым confirmation claim или основанием менять primary metric.

Bridge report:
`outputs/tile-position-distance-validation/relation-selector-bridge-v1/report.json`,
SHA-256
`2f14336e91ca889e9c8777f90ee596a7f390cfeacb7a82378a140b42a9781104`.

## Рекомендованный metric contract

Для exact-oriented solver runs:

1. **Primary:** absolute `exact_tiles_per_board` / radius0 recall, maximize.
2. **Secondary smooth progress:** absolute `mean_manhattan_cells`, minimize.
3. **Near-placement companion:** absolute radius2 recall, maximize; p90
   Manhattan хранить как tail diagnostic.
4. **Orthogonal structure:** satisfied adjacent pairs / pair recall продолжать
   логировать отдельно — position distance не заменяет relational geometry.
5. **End-to-end:** dirty/restored SSIM остаётся отдельным final outcome metric.
6. **Origin diagnostic only:** cyclic-aligned exact/distance всегда показывать
   рядом с absolute, но никогда не использовать отдельно как gate, selector
   objective или leaderboard claim.

То есть tile distance нужно **добавить**, а exact не заменять. В early-stage
coordinate/placement learning mean Manhattan может быть более чувствительным
capacity signal; promotion всё равно требует absolute exact non-regression и
отдельного pair/SSIM контроля.

## Ограничения

- Это train-only validation на уже открытом local32, не fresh confirmation.
- Pooled suite намеренно содержит controlled reference perturbations; они
  расширяют error range и помогают проверить metric semantics, но не являются
  естественным распределением будущих solver outputs.
- Within-source и per-source views уменьшают content/source confounding, но не
  устраняют dependence от выбранного perturbation roster.
- SSIM зависит от content, corruption и restoration; никакая layout-only
  distance metric не может объяснить его полностью.

## Артефакты

- report:
  `outputs/tile-position-distance-validation/fixed-v1/report.json`, SHA
  `57308b3bc944226022fcba0a52a55fa2ffd50391f0aa41b368f28c1bb9957ad6`;
- frozen layouts:
  `outputs/tile-position-distance-validation/fixed-v1/frozen-layout-roster.npz`,
  SHA `a789d49fd65e4ba6556c235217817b3e9635a35a9a6e719b307d8c5070392fe0`;
- pre-score freeze SHA
  `6399730fd7f168e52e6226cfe8d60e80e28fa7d7296f1f412828e117863484cf`;
- runner: `scripts/validate_tile_position_distance_metric.py`;
- relation-selector descriptive bridge runner:
  `scripts/describe_relation_selector_distance_bridge.py`;
- tests: `tests/test_tile_position_distance.py`.

Фактический runtime: `1.0 s` freeze + `179.2 s` scoring/correlations.

```bash
PYTHONPATH=src:. .venv/bin/python \
  scripts/validate_tile_position_distance_metric.py
```

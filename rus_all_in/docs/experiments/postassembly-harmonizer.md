# Target-blind postassembly harmonizer

Статус: **promote on calibration as the preferred postassembly order**. Это
результат только на calibration; holdout и competition test для этого
эксперимента не открывались.

> Production freeze после отдельного manual-safety аудита: strict **no-atlas**
> bilateral buddies96 → эти же RGB offsets → bounded luma → ровно один colored
> NLM `h=20/hColor=20`. Исследовательские h10 multi-pass таблицы ниже сохраняются
> как provenance, но production CLI их больше не принимает.

## Вывод

Для строгой сборки `bilateral atlas w=0.03 + buddies96` лучший проверенный
порядок такой:

```text
576 input tiles
  -> strict one-to-one raw assembly
  -> additive RGB seam-graph offsets
  -> bounded luminance seam-graph gains
  -> proper RGB OpenCV NLM h=10, repeated 20 times
```

На свежей панели calibration `offset=48, count=24` этот pipeline дал
`0.3150448153` mean SSIM против `0.3034585499` у той же сборки с одним лишь
20-кратным NLM: gain `+0.0115862654`, paired bootstrap 95% CI
`[+0.0096737498, +0.0135225281]`, wins `24/24`.

Гармонизацию следует применять **до NLM**. Для RGB+luma разница порядка
`harmonize -> NLM` минус `NLM -> harmonize` положительна на каждом из 24
изображений для каждого числа проходов:

| NLM passes | Разница порядка | Paired 95% CI | Wins |
|---:|---:|---:|---:|
| 1 | +0.008168 | `[+0.006482, +0.009979]` | 24/24 |
| 5 | +0.010146 | `[+0.008045, +0.012379]` | 24/24 |
| 10 | +0.010054 | `[+0.008005, +0.012233]` | 24/24 |
| 20 | +0.009987 | `[+0.008208, +0.011844]` | 24/24 |

Вывод независимо подтвердился на следующей непересекающейся панели
calibration 72:96 и на **true no-atlas** `solve_buddies(max_edges=96)`. Там
RGB+luma -> NLM20 дал `0.2890087002` против `0.2817576665`, gain
`+0.0072510337`, 95% CI `[+0.0059598949, +0.0085700755]`, 24/24 wins. Таким
образом, эффект не является артефактом population atlas или одной панели.

## Что именно перенесено

Алгоритмы перенесены без изменения численной логики из read-only research
ветки `origin/таска-говно`:

- commit: `d6a82f82ceefa109ef706402712d03805bc9e880`;
- source: `source/src/puzzle_assembly/postassembly_harmonizer.py`;
- Git blob: `9d8d01c0f48d0e1473c1ff48285b06ab786a5dd8`;
- package port: `src/aiijc_puzzle/postassembly_harmonizer.py`;
- frozen configs: `configs/postassembly_rgb_offset_v1.json` и
  `configs/postassembly_luminance_gain_v1.json`.

RGB-этап строит граф всех 1104 соседних пар в уже собранной решётке, линейно
экстраполирует три крайних ряда/столбца к центру шва и решает robust IRLS-задачу
для ограниченных additive per-tile RGB offsets. Глобальная калибровка фиксируется
нулевой медианой offset по каждому каналу; абсолютный offset ограничен 12.

Отдельный luminance-этап работает на результате RGB-коррекции, решает аналогичную
задачу в log-gain и ограничивает множитель диапазоном `[0.96, 1.04]`. Пиксели с
яркостью вне `[12, 243]` исключаются из оценки шва. Ни один этап не получает
target, clean tile, source identity или layout label.

Порт был проверен на одном и том же детерминированном массиве против модуля,
загруженного непосредственно через `git show` из указанного blob. Получены
побитово одинаковые RGB offsets, RGB render, luminance gains, luminance render и
одинаковые diagnostics. Изменены только package imports, docstrings и форма
default singleton, требуемая текущими lint-правилами.

## Frozen protocol и compliance

- Manifest protocol digest:
  `2a9e3b74f7defa8c00846a05eb598fd263fd16c2787c70e77d3b7a4b585bfbf4`.
- Split: только `calibration`, records 48:72 общего seeded selector-а.
- Selection digest:
  `77b4a65efa310f93d82fba0e1a54cd390a2e32185ec9fec53035258d50f03990`.
- Train-only atlas:
  `artifacts/low-frequency-prior/train5600-v1.npz`, SHA-256
  `d92fa19cd66f557028e576e4ad4c4a864553882b2ae52766d24072d0312267fe`.
- Layout для всех вариантов один и тот же: bilateral position atlas `w=0.03`,
  strict buddies с budget 96.
- Все 576 исходных фрагментов каждого board проверены до любой коррекции:
  каждый raw canvas — точная биективная перестановка input pixels.
- Нет template render, подмены/дублирования tiles или spatial warp.
- Полный roster из 23 predictions для всех 24 boards был построен и
  захеширован **до первого decode target**. Frozen roster digest:
  `11263ba1ec417f0da076dca450007b8171f8bc574a32c46945e0cbd9ebfad471`.
- NLM использует правильное RGB -> BGR -> OpenCV -> RGB преобразование,
  `h=10`, template window 7, search window 21.
- Holdout access: false. Test access: false.

Гармонизация меняет только значения пикселей уже размещённого tile и не может
скрыть ошибку permutation audit: audit выполняется на raw assembly до
restoration. Калибровочный win делает этот tail активным кандидатом, но не
отменяет финальный manual visual/compliance gate перед submission, особенно для
20 повторов NLM.

## Полные результаты

### До NLM

| Вариант | Mean SSIM | Gain vs raw | Paired 95% CI | Wins |
|---|---:|---:|---:|---:|
| raw assembly | 0.117480 | — | — | — |
| RGB offsets | 0.121916 | +0.004435 | `[+0.003688, +0.005202]` | 24/24 |
| RGB offsets + bounded luma | **0.122707** | **+0.005227** | `[+0.004278, +0.006229]` | 24/24 |

### Гармонизация перед repeated NLM

| Passes | NLM baseline | RGB -> NLM | Gain | RGB+luma -> NLM | Gain | Wins RGB+luma |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.219814 | 0.231523 | +0.011709 | **0.233518** | **+0.013704** | 24/24 |
| 5 | 0.272445 | 0.282981 | +0.010536 | **0.284940** | **+0.012496** | 24/24 |
| 10 | 0.289291 | 0.299329 | +0.010037 | **0.301080** | **+0.011789** | 24/24 |
| 20 | 0.303459 | 0.313564 | +0.010106 | **0.315045** | **+0.011586** | 24/24 |

Bounded luma даёт дополнительный выигрыш поверх RGB для всех четырёх tails:
`+0.001995`, `+0.001960`, `+0.001751`, `+0.001480` при 1/5/10/20 проходах;
wins соответственно 24/24, 23/24, 24/24, 23/24.

Reverse order работает заметно слабее. После NLM20 применение RGB+luma даёт
только `0.305057` (`+0.001599` к NLM20, 21/24 wins), тогда как применение тех же
этапов до NLM20 даёт `0.315045`.

### Независимое no-atlas подтверждение

| Passes | NLM baseline | RGB+luma -> NLM | Gain | Paired 95% CI | Wins |
|---:|---:|---:|---:|---:|---:|
| 5 | 0.257878 | **0.267160** | **+0.009281** | `[+0.007637, +0.010956]` | 24/24 |
| 10 | 0.271788 | **0.279875** | **+0.008087** | `[+0.006676, +0.009556]` | 24/24 |
| 20 | 0.281758 | **0.289009** | **+0.007251** | `[+0.005960, +0.008570]` | 24/24 |

Для RGB+luma разница `harmonize -> NLM` минус reverse order равна
`+0.007363`, `+0.006430`, `+0.005294` на 5/10/20 проходах. Все bootstrap CI
строго положительны; wins 24/24, 24/24, 23/24. Selection digest панели:
`3b3f1f3b0c4aa5642a83956da8bcd6d7e5ee101a69fb171fbaf65e2444059cc5`.
Все predictions были глобально frozen до первого target decode; atlas не
загружался, permutation audits прошли 24/24.

## Воспроизведение

Полный target-blind render занимает основную часть времени из-за 20 проходов NLM:

```bash
uv run python scripts/run_postassembly_harmonizer.py --run

uv run python scripts/run_postassembly_harmonizer.py --run \
  --layout no-atlas --offset 72 --count 24 --nlm-h 10 \
  --passes 5 10 20 \
  --output outputs/postassembly-harmonizer/no-atlas-calibration-offset72-count24-h10.json
```

Authoritative machine-readable report:
`outputs/postassembly-harmonizer/calibration24.json`. Он содержит per-board SSIM,
prediction/layout hashes, diagnostics обоих seam solvers, каждый permutation
audit, source/config hashes и 20 000-replicate paired bootstrap intervals.
No-atlas confirmation сохранён в
`outputs/postassembly-harmonizer/no-atlas-calibration-offset72-count24-h10.json`.

Проверки:

```bash
uv run ruff check src/aiijc_puzzle/postassembly_harmonizer.py \
  scripts/run_postassembly_harmonizer.py tests/test_postassembly_harmonizer.py
uv run pytest tests/test_postassembly_harmonizer.py \
  tests/test_compliant_atlas_decoder.py
```

## Решение

Не повторять перебор порядка: на этой frozen панели `harmonize -> NLM` доминирует
reverse order 24/24 во всех восьми сравнениях RGB/RGB+luma x 1/5/10/20.
Default исследовательского pipeline: RGB offsets, затем bounded luminance, затем
RGB NLM. Среди проверенного roster максимум достигается на 20 проходах; выбирать
меньший pass count можно только как явный runtime/visual-safety trade-off.

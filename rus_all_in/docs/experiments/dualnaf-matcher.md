# Tile-wise DualNAF как pre-denoiser только для matcher-а

Статус: **reject-as-tested; fail-fast после pilot, confirmation не открывался**.
Holdout и competition test не использовались.

## Гипотеза и граница легальности

Проверена предложенная цепочка `pre-denoise -> matcher`, но без переноса
сгенерированных моделью пикселей в ответ:

```text
576 original upright dirty tiles
  -> frozen DualNAF independently on each 20x20 tile
  -> DualNAF pixels are used only for E14 right/down scores
  -> strict no-atlas buddies96 layout
  -> assemble ORIGINAL dirty tiles exactly once
  -> RGB seam offsets -> bounded luma -> colored NLM h20 exactly once
```

Таким образом, модель могла изменить только решение о соседстве. Все четыре
выходных canvas-а перед harmonizer-ом были точными перестановками исходных 576
тайлов: без rotation, resampling, warp, substitution и model-rendered pixels.

## Preregistered roster и gate

До первого target decode были зафиксированы четыре arm-а:

1. `baseline_dirty_bilateral`: true dirty bilateral E14 control;
2. `dualnaf_match_raw`: E14 на raw output независимого per-tile DualNAF;
3. `dualnaf_match_bilateral`: bilateral E14 на output DualNAF;
4. `fusion_dirty_bilateral_dualnaf_raw_50_50`: единственный fixed fusion —
   среднее уже row-normalized log-probability scores control и DualNAF-raw.

Последний arm был единственным primary. Разрешение на disjoint confirmation
требовало одновременно mean tail gain не меньше `+0.002`, положительную нижнюю
границу paired 95% CI, минимум 8/12 wins, положительную adjacency и отсутствие
регрессии в обоих направлениях и translation-aligned placement. Чистые DualNAF
arm-ы заранее объявлены диагностическими и не могли стать winner постфактум.

## Frozen protocol

- checkpoint:
  `outputs/restoration-r6/compliant-r6-medium-train256-step2000-h10.pt`;
- checkpoint SHA-256:
  `331322460c8af87e5d4760b075726979f0574a23209889c1e95b6b90f2eac1a9`;
- model conditioning: independent per-tile NLM `h=10`;
- fresh calibration records `192:204`, 12 boards;
- selection digest:
  `a1ee0cd89730918d571336a6e910fe562e8d6c46f8e610c29dcc421649e6ec6d`;
- decoder: no-atlas buddies, `max_edges=96` для всех arm-ов;
- output tail: RGB offsets -> bounded luma -> full-canvas colored NLM `h20 x1`;
- все 48 layouts/images заморожены до первого target decode;
- frozen sidecar SHA-256:
  `7451729d9808b1468e4f540e3fb00fa8f6fb1dd7192a0dcd92640250fd6b9e98`;
- raw exact-permutation/pixel-multiset audits: 48/48 pass;
- holdout access: false; test access: false.

Target использовался только после freeze для SSIM и approximate train-label
диагностик через one-to-one `recover_layout`.

## Результат

Все дельты ниже paired относительно `baseline_dirty_bilateral` на одних и тех
же 12 boards.

| Matcher scores | Tail SSIM | Delta | Paired 95% CI | Wins | Adjacency | Delta adjacency |
|---|---:|---:|---:|---:|---:|---:|
| Dirty bilateral baseline | **0.227034** | — | — | — | 0.035326 | — |
| DualNAF raw | 0.228133 | +0.001099 | `[−0.007584, +0.009064]` | 7/12 | 0.032760 | −0.002566 |
| DualNAF bilateral | 0.224402 | −0.002632 | `[−0.010944, +0.004969]` | 3/12 | 0.032458 | −0.002868 |
| 50/50 primary fusion | 0.225512 | **−0.001522** | `[−0.008599, +0.005119]` | 7/12 | 0.037666 | +0.002340 |

У чистого DualNAF-raw точное размещение формально выросло с `0.001736` до
`0.002749`, но right adjacency упала на `−0.005133` с полностью отрицательным
95% CI `[−0.009662, −0.000755]`. Это не восстановление геометрии: SSIM gain мал,
неопределён и сопровождается ухудшением соседства.

Primary fusion дал противоположный размен: слабый и статистически неопределённый
рост adjacency (`+0.002340`, CI пересекает ноль) при снижении full-image tail
SSIM. Три обязательных SSIM-условия gate провалены.

## Решение

Confirmation calibration `204:228` не открывался. Не масштабировать и не
добавлять текущий frozen DualNAF checkpoint в production matcher — ни как raw
E14 view, ни после bilateral, ни в проверенном 50/50 fusion.

Отрицательный вывод ограничен именно текущим checkpoint и E14+buddies96:
специально обученный edge model или denoiser с loss на boundary compatibility
будет новой гипотезой. Простое повторение того же checkpoint на большем числе
boards или перебор fusion weight без нового preregistration не нужен.

## Код и воспроизведение

- matcher-only helpers: `src/aiijc_puzzle/dualnaf_matcher.py`;
- runner: `scripts/run_dualnaf_matcher.py`;
- tests: `tests/test_dualnaf_matcher.py`;
- report:
  `outputs/dualnaf-matcher/pilot-calibration-offset192-count12.json`;
- pre-target sidecar:
  `outputs/dualnaf-matcher/pilot-calibration-offset192-count12.frozen.json`.

```bash
uv run python scripts/run_dualnaf_matcher.py --run --stage pilot \
  --device mps --batch-size 144

uv run ruff check src/aiijc_puzzle/dualnaf_matcher.py \
  scripts/run_dualnaf_matcher.py tests/test_dualnaf_matcher.py
uv run pytest tests/test_dualnaf_matcher.py tests/test_tilewise_renderer.py
```

# Tile-wise DualNAF faithful-renderer pilot

Статус: **rejected-for-final в композиции с repeated NLM; no scale-up**.
Holdout и competition test не открывались.

## Гипотеза

Предыдущий R6 запуск применял DualNAF ко всему неверно собранному 480x480 canvas,
поэтому convolutional receptive field смешивал пиксели разных, часто ложных
соседей. Исторически сильный TileNAF, напротив, обрабатывал каждый upright tile
независимо. Этот pilot проверяет максимально близкую доступную формулировку на
замороженном текущем checkpoint:

```text
dirty input
  -> bilateral true no-atlas solve_buddies(max_edges=96)
  -> exact 576-tile permutation audit
  -> frozen DualNAF independently on each upright 20x20 tile
  -> reassemble with the unchanged strict layout
  -> RGB seam offsets
  -> bounded luminance gains
  -> proper RGB NLM h=10, passes 5 / 10 / 20
```

NLM conditioning самого DualNAF также вычисляется отдельно внутри каждого 20x20
tile. Batch dimension используется только для ускорения: модель не получает
пиксели, контекст или координаты других tiles. Один input tile порождает ровно
один output tile того же размера и индекса; rotation, resampling, warp,
substitution и template render отсутствуют.

## Frozen protocol

- Checkpoint:
  `outputs/restoration-r6/compliant-r6-medium-train256-step2000-h10.pt`;
- checkpoint SHA-256:
  `331322460c8af87e5d4760b075726979f0574a23209889c1e95b6b90f2eac1a9`;
- architecture: DualNAF, base 24, depth 3, blocks 2, 347 715 parameters;
- checkpoint training: manifest-train 256, 2 000 steps, NLM conditioning h=10;
- evaluation: fresh calibration records 96:108, 12 boards;
- selection digest:
  `b6c9f3133ad8e0719f54cbc0d3ce20323a99e9937c3118d5b7272d41238b5c8f`;
- layout: true no-atlas bilateral buddies96, одинаковый для control и model;
- tails: RGB+luma before proper RGB NLM h=10, frozen passes 5/10/20;
- все predictions всех 12 boards построены до первого target decode, frozen
  prediction digest:
  `a2161d92e68303c2d0304f62889449450ba6c513850e6ee3c089e17bb42265a7`;
- raw permutation/pixel-multiset audit: 12/12 pass;
- holdout access: false; test access: false.

Панель не пересекается с calibration 0:12, сохранённой в checkpoint metadata, с
первым harmonizer panel 48:72 и с no-atlas confirmation 72:96.

## Результат

### Без repeated canvas NLM

Tile-wise применение модели само по себе действительно устраняет прежний
full-canvas failure:

| Вариант | Raw-tile control | Tile-wise DualNAF | Gain | Paired 95% CI | Wins |
|---|---:|---:|---:|---:|---:|
| До harmonizer | 0.141186 | **0.209974** | **+0.068788** | `[+0.054838, +0.083358]` | 12/12 |
| После RGB+luma | 0.146173 | **0.228405** | **+0.082232** | `[+0.066026, +0.098574]` | 12/12 |

То есть независимая per-tile геометрия — существенно более подходящий способ
использовать этот checkpoint, чем обработка всего ошибочного canvas.

### В обязательном сравнении с сильным tail

При добавлении сильного repeated NLM результат меняется на противоположный:

| NLM passes | Raw tiles -> RGB+luma -> NLM | DualNAF tiles -> RGB+luma -> NLM | Delta | Paired 95% CI | Wins |
|---:|---:|---:|---:|---:|---:|
| 5 | **0.337324** | 0.320167 | **−0.017158** | `[−0.026052, −0.008595]` | 1/12 |
| 10 | **0.351736** | 0.321965 | **−0.029771** | `[−0.046467, −0.015669]` | 1/12 |
| 20 | **0.364465** | 0.312388 | **−0.052077** | `[−0.087106, −0.024818]` | 1/12 |

Ухудшение растёт вместе с числом проходов. На NLM20 две наиболее сильные для
control доски теряют `−0.15284` и `−0.18616`; только одна из 12 выигрывает.
Предварительно заданный downstream gate поэтому провален.

## Pixel-fidelity diagnostics

Средние target-free изменения per-tile renderer-а:

- mean absolute change от dirty tile: 15.02 levels;
- mean absolute change самого per-tile NLM conditioning: 7.62;
- дополнительный model residual относительно conditioning: 12.94;
- mean q99 абсолютного изменения: 43 levels;
- mean q99 model residual относительно conditioning: 30.83 levels;
- unchanged pixels: 1.71%; clipped output pixels: 4.59%.

Эти числа не нарушают geometry contract, но показывают, что checkpoint не
является слабой correction поверх NLM: его residual крупнее самого conditioning
transform. Наблюдаемая зависимость от pass count согласуется с
over-restoration/texture-loss конфликтом, хотя это объяснение — inference, а не
отдельно идентифицированная причинность.

## Решение

Не масштабировать текущую композицию и не открывать holdout. Tile-wise DualNAF
может быть полезен только как **замена** слабому/отсутствующему pixel tail: без
repeated canvas NLM он выигрывает 12/12. В текущем сильнейшем pipeline он не
добавляется: raw ordered tiles -> RGB offsets -> bounded luma -> NLM20 остаётся
лучше на `+0.052077` SSIM в среднем.

Не повторять этот exact checkpoint перед 5/10/20x NLM на большем split. Новый
renderer имеет смысл возвращать в roster лишь после обучения специально под
per-tile input и остаток **после** зафиксированного final NLM budget либо после
отдельного замещения NLM с новым frozen gate.

## Код и воспроизведение

- independent renderer: `src/aiijc_puzzle/tilewise_renderer.py`;
- runner: `scripts/run_tilewise_dualnaf_harmonizer.py`;
- tests: `tests/test_tilewise_renderer.py`;
- report:
  `outputs/tilewise-dualnaf/no-atlas-calibration-offset96-count12-h10.json`.

```bash
uv run python scripts/run_tilewise_dualnaf_harmonizer.py --run \
  --device mps --offset 96 --count 12 --nlm-h 10 \
  --passes 5 10 20 --batch-size 144

uv run ruff check src/aiijc_puzzle/tilewise_renderer.py \
  scripts/run_tilewise_dualnaf_harmonizer.py tests/test_tilewise_renderer.py
uv run pytest tests/test_tilewise_renderer.py
```

Unit tests проверяют сохранение формы/dtype/roster, отсутствие влияния изменения
одного tile на остальные 575 и инвариантность результата к разбиению batch.

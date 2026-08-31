# Dense legal single-pass colored-NLM tail

## Решение

**Reject as tested; не открывать confirmation и не менять frozen production.**
Плотный target-free sweep `h=21..29`, где каждый вариант применяет ровно один
proper RGB colored-NLM pass, дал устойчивый монотонный прирост относительно
`h20`, но не достиг заранее заданного абсолютного порога `0.27`. Максимальный
manual-safe arm `h28 x1` получил `0.257032`; `h29 x1` получил `0.257957`, но
дополнительно нарушил preregistered Laplacian-retention bound.

Это **reused calibration**, не fresh calibration и не holdout. Старый
`outputs/legacy-upgrade/calibration700-champion/report.json` уже содержит те же
filename, input SHA-256 и target SHA-256 для всех 700 records текущего
calibration split. Поэтому результат годится только как контролируемый screen и
не является независимой оценкой generalization.

## Preregistration и selector correction

Первая конфигурация ошибочно трактовала `300:336` как прямой slice массива
manifest. Ошибка была найдена до декодирования target pixels и до создания
prediction artifacts. Ошибочная конфигурация сохранена read-only как
`configs/dense_safe_tail_reused_calibration_preregistered_v1.json`, SHA-256
`cb29c68c8f1aa38ba63ca5ff0906a140d694c251c0f7366c1a55851c35dbe8d4`, но
не использовалась.

Authoritative v2 использует общий
`aiijc_puzzle.protocol.select_manifest_records` с namespace
`aiijc-puzzle-experiments-v1`, seed `20260829`:

- config:
  `configs/dense_safe_tail_reused_calibration_preregistered_v2.json`;
- config SHA-256:
  `6e1ed5840bf77f9ce5ef7f3a83cdfb84232fa34813f700bda11a56eae2b8fa3c`;
- primary ranked slice `300:336`, NUL-join digest
  `f58ceb741d57e2df7e9b2645c7e2b50cc5944dfc8123bbd26135dffc93839be0`;
- gated confirmation ranked slice `384:420`, NUL-join digest
  `814f211d5771deca4a5dd9343430c248c90e22dd3f276b4bda48b1670325c7f0`;
- взаимное пересечение панелей: `0`;
- историческое exposure через calibration-700: `36/36` и `36/36`.

До target decode были сохранены и захешированы 36 layouts, raw assemblies,
RGB+luma harmonized canvases и все 12 arms. Все strict permutation audits
прошли. Commitment SHA-256:
`431f92ae46dad5c95a78f0fb1569742d0b5aec5dafe72cdef64fd9b4599f6a79`.
После записи read-only commitment создан отдельный target-open receipt, SHA-256
`8210dd563f877589e65fe8c2a6bcc2b8d6e267da6cd7ecc806ae6e756a3e1e16`.

## Frozen arms и gate

Проверялись:

- baseline `h20 x1`;
- `h21..29 x1`, где `hColor=h`, template window 7, search window 21;
- два заранее заданных convex blend независимых single-pass outputs:
  `0.75*h20 + 0.25*h28` и `0.50*h20 + 0.50*h28`.

Последовательного или повторного NLM не было. Для promotion одновременно
требовались mean RGB SSIM `>=0.27`, paired t CI lower `>0`, не менее 24 wins из
36 и все detail/grid safety bounds относительно `h20`:

- mean/min within-tile gradient retention `>=0.80/0.70`;
- mean/min Laplacian retention `>=0.72/0.60`;
- mean/max relative grid ratio `<=1.05/1.12`;
- 36 distinct board outputs.

## Primary result

| Arm | Mean RGB SSIM | Gain vs h20 | Paired 95% t CI | Wins | Safety | Absolute 0.27 |
|---|---:|---:|---:|---:|---|---|
| h20 x1 | 0.246740 | — | — | — | baseline | FAIL |
| h21 x1 | 0.248420 | +0.001680 | `[+0.001548,+0.001813]` | 36/36 | PASS | FAIL |
| h22 x1 | 0.249950 | +0.003211 | `[+0.002954,+0.003467]` | 36/36 | PASS | FAIL |
| h23 x1 | 0.251358 | +0.004619 | `[+0.004242,+0.004995]` | 36/36 | PASS | FAIL |
| h24 x1 | 0.252665 | +0.005925 | `[+0.005436,+0.006414]` | 36/36 | PASS | FAIL |
| h25 x1 | 0.253872 | +0.007132 | `[+0.006536,+0.007729]` | 36/36 | PASS | FAIL |
| h26 x1 | 0.254987 | +0.008247 | `[+0.007549,+0.008945]` | 36/36 | PASS | FAIL |
| h27 x1 | 0.256047 | +0.009307 | `[+0.008509,+0.010105]` | 36/36 | PASS | FAIL |
| **h28 x1** | **0.257032** | **+0.010292** | **`[+0.009397,+0.011187]`** | **36/36** | **PASS** | **FAIL** |
| h29 x1 | 0.257957 | +0.011217 | `[+0.010227,+0.012208]` | 36/36 | FAIL: mean Laplacian `0.7081<0.72` | FAIL |
| h20/h28 75/25 | 0.249404 | +0.002664 | `[+0.002428,+0.002900]` | 36/36 | PASS | FAIL |
| h20/h28 50/50 | 0.252183 | +0.005443 | `[+0.004994,+0.005892]` | 36/36 | PASS | FAIL |

`h28` сохранил mean/min within-tile gradient `0.8701/0.7845`, mean/min
Laplacian `0.7305/0.6349` и mean/max relative grid ratio `0.7803/0.8658`.
`h29` прошёл остальные safety bounds, но его mean Laplacian retention равен
`0.7081`, ниже фиксированного `0.72`.

Визуальный лист подтверждает интерпретацию: увеличение `h` сглаживает шум и
мелкие детали, но не восстанавливает глобальную сцену. Все четыре просмотренные
раскладки остаются очевидными мозаиками. Это tail ceiling, а не решение layout
problem.

## Artifacts и воспроизведение

Authoritative report:
`outputs/dense-safe-tail/6e1ed5840bf77f9ce5ef7f3a83cdfb84232fa34813f700bda11a56eae2b8fa3c/primary/report.json`,
SHA-256
`b48971fcb4646fca914f021547c80d290895a81bdb7366edf31d4c21df262330`.

Contact sheet:
`outputs/dense-safe-tail/6e1ed5840bf77f9ce5ef7f3a83cdfb84232fa34813f700bda11a56eae2b8fa3c/primary/manual-safety-contact-sheet.png`,
SHA-256
`d531dfd23074161e587f393828db55a7ac8494a732ec2d2b0d1e1ca6bb6c74a4`.

```bash
uv run python scripts/run_dense_safe_tail.py audit
uv run python scripts/run_dense_safe_tail.py prepare-primary
uv run python scripts/run_dense_safe_tail.py score-primary
```

Последние две команды intentionally single-use и не перезаписывают существующий
commitment/receipt/report. Поскольку primary gate не прошёл,
`prepare-confirmation` fail-closed и confirmation targets не открывались.


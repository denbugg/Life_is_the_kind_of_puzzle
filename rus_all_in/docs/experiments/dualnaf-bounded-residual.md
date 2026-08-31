# Bounded DualNAF residual перед frozen h20x1 tail

Статус: **два положительных paired signal, но оба absolute/gate FAIL**.
Confirmation `420:444` и `588:636`, holdout и competition test не открывались.

## Новая проверенная гипотеза

Предыдущий tile-wise pilot проверял полную замену original tiles на DualNAF и
repeated NLM `h10`; matcher pilot использовал DualNAF только для edge scores.
Здесь впервые проверена ограниченная convex correction при неизменном layout:

```text
original upright tile и frozen DualNAF output того же tile
  -> round((1-alpha) * original + alpha * DualNAF), alpha <= 0.5
  -> RGB seam offsets -> bounded luma -> colored NLM h20 ровно один раз
```

Все arm-ы используют один no-atlas bilateral `solve_buddies(max_edges=96)`
layout. Raw canvas до restoration является точной перестановкой всех 576 input
tiles; renderer работает one-to-one внутри каждого upright 20x20 tile. Нет
rotation, resampling, warp, external/template/reference pixels или cross-board
data.

## Preregistration и честная граница данных

До target decode были заморожены пять arm-ов: `alpha=0`, `0.125`, `0.25`,
`0.375`, `0.5`. Selection rule выбирал candidate с наибольшим mean SSIM, tie —
меньший alpha. Его gate требовал одновременно:

- mean final SSIM `>=0.27`;
- paired 95% CI gain vs `alpha=0` строго выше нуля;
- не меньше 18/24 wins;
- target-free и ручной preservation pass.

Config:
`configs/dualnaf_bounded_residual_preregistered_v1.json`, SHA-256
`73e0553efd0a5f6f2610f97595149cbfdeb9cdc57c5a7f516b86fe93cbbd95b3`.
Primary — calibration `336:360`, 24 boards, selection digest
`76c93b31e0732342257852dca9f8bad456cbb6dadc044458c9488dee053d05a2`.

Эта панель **не является fresh/untouched**: historical
`legacy-upgrade/calibration700-champion` уже оценивал все 700 calibration names.
Она лишь не пересекается с прежними DualNAF pilots и planned confirmation этого
опыта. Это reused calibration evidence, не holdout estimate.

120 final predictions были сохранены до первого target decode текущего запуска.
Prediction commitment SHA-256:
`107914c532c29742d6d09c96b21bd559b2155947aa721fcdc92614a72d898f16`;
frozen pixel-roster digest:
`d5d1668f05bd051e625a666dc42de6dbd2348521cfa9d2a1b9ffb003bbddadf4`.
Permutation audit прошёл 24/24.

## Результат

Baseline `alpha=0` равен `0.282739`.

| Alpha | Mean SSIM | Gain | Paired 95% CI | W/T/L |
|---:|---:|---:|---:|---:|
| 0.125 | 0.284503 | +0.001763 | `[+0.000816,+0.002861]` | 20/0/4 |
| 0.25 | 0.285773 | +0.003034 | `[+0.001212,+0.005148]` | 18/0/6 |
| 0.375 | 0.286624 | +0.003884 | `[+0.001215,+0.006968]` | 18/0/6 |
| **0.5** | **0.286933** | **+0.004194** | `[+0.000781,+0.008160]` | **16/0/8** |

По frozen selection rule победил `alpha=0.5`, но он улучшил лишь 16/24 boards
при требовании 18/24. Поэтому общий quantitative gate — **FAIL**, хотя absolute
SSIM и CI условия прошли. Нельзя задним числом заменить winner на `0.25` или
`0.375`: их более стабильные результаты остаются diagnostics.

Manual review всех 24 side-by-side пар не нашёл новых тяжёлых артефактов или
material local-structure regression. Gradient-energy ratio candidate/control
лежит в `[0.9281,0.9960]`, mean `0.9692`; максимальный рост clipped fraction —
`0.000433`. Candidate немного сглаживает детали. Обе стороны всё ещё сильно
мозаичны из-за общего слабого layout, поэтому review подтверждает только
относительную pixel preservation, не правильность пазла.

## Решение

Confirmation `420:444` не открывать: preregistered winner провалил wins gate.
Результат показывает реальный, статистически положительный bounded-restoration
signal и исключает прежний вывод, будто frozen DualNAF полезен только как full
replacement без final tail. Следующий опыт должен быть новой preregistration с
одним заранее выбранным безопасным alpha на отдельной reused-calibration панели,
без post-hoc выбора по этой таблице.

Artifacts:

- `outputs/dualnaf-bounded-residual/primary-calibration-offset336-count24/prediction-commitment.json`;
- `outputs/dualnaf-bounded-residual/primary-calibration-offset336-count24/report.json`,
  SHA-256
  `1199901443221721c55201b7d4581cb687753108f36cd18c3c437cb7f852eda1`;
- `outputs/dualnaf-bounded-residual/primary-calibration-offset336-count24/manual-review.json`;
- четыре sheets в `manual-review-sheets/`.

Код и тесты:

- `src/aiijc_puzzle/dualnaf_bounded_residual.py`;
- `scripts/run_dualnaf_bounded_residual.py`;
- `tests/test_dualnaf_bounded_residual.py`.

```bash
uv run python scripts/run_dualnaf_bounded_residual.py \
  --phase freeze --stage primary --device mps --batch-size 144 --run
uv run python scripts/run_dualnaf_bounded_residual.py \
  --phase score --stage primary --run
uv run pytest tests/test_dualnaf_bounded_residual.py tests/test_tilewise_renderer.py
```

Runner fail-closed откажется перезаписывать существующий frozen experiment.

## Single-alpha follow-up: alpha=0.125

После FAIL первого winner заранее выбран единственный безопасный вариант
`alpha=0.125`: именно он имел максимальную стабильность 20/24 и минимальное
изменение pixels. Новый config запрещал другие alpha и фиксировал primary
calibration `540:588` (48 boards), confirmation `588:636` (48 boards):

- config:
  `configs/dualnaf_alpha0125_followup_preregistered_v1.json`;
- preregistration SHA-256:
  `a11b246fc14b3c04b1e995892f66b87a895f73142d47f8afc886ec3a1c99e162`;
- primary selection digest:
  `182fd97bd02a6d5ad1f86ca16ae60e606b1d09e08805a8d94c216b4d3212d183`;
- confirmation selection digest:
  `580aa4e6d24e3394cb07f7b732d806d3c30ad155bcf0c4cf98236e5fe84334b9`.

Follow-up является adaptive reused-calibration evidence: все имена уже входили
в historical calibration700, а по два имени в каждой панели встречались в
других современных diagnostics. Primary и confirmation взаимно непересекаются,
но ни один из них нельзя называть untouched holdout.

96 primary predictions были заморожены до target decode. Commitment SHA-256:
`0f9c7e2472437a95d4ba44423a4ab05b25bd51155d5a34265b11cbc34229070c`;
pixel-roster digest:
`3b62ebab5b55d8b7ef5486be111eed76401ed53030650d950ffa517e9a5ea5d7`.
Все 48 raw permutation audits прошли.

| Arm | Mean SSIM | Gain | Paired 95% CI | W/T/L |
|---|---:|---:|---:|---:|
| `alpha=0` | 0.250113 | — | — | — |
| `alpha=0.125` | **0.251208** | **+0.001096** | `[+0.000688,+0.001555]` | **38/0/10** |

Effect устойчив: CI строго положителен, wins gate 32/48 пройден. Но absolute
gate `>=0.27` провален (`0.251208`), поэтому общий primary status — **FAIL** и
confirmation `588:636` не открывался.

Manual review всех 48 пар прошёл relative safety: новых тяжёлых артефактов и
material regressions не обнаружено; gradient-energy ratio лежит в
`[0.9737,0.9973]`, mean `0.9892`, maximum clipped-fraction increase всего
`0.0000246`. Это почти незаметная коррекция и она не исправляет общий мозаичный
layout.

Follow-up artifacts:

- `outputs/dualnaf-alpha0125-followup/primary-calibration-offset540-count48/prediction-commitment.json`;
- `outputs/dualnaf-alpha0125-followup/primary-calibration-offset540-count48/report.json`,
  SHA-256
  `563c055c68dc5fd6d6714ff76827860499dac5a80e4f1636ca5c4b1afeaede7f`;
- `outputs/dualnaf-alpha0125-followup/primary-calibration-offset540-count48/manual-review.json`,
  SHA-256
  `20915dcac2785a947782da1ef129dc685bf546374535b38ca8388a3ea8e9b08e`;
- `scripts/run_dualnaf_alpha0125_followup.py`.

Итог bounded-route: small convex DualNAF residual статистически улучшает frozen
baseline на обеих reused-calibration панелях, но измеренный прирост
`+0.0011..+0.0018` слишком мал, чтобы переносить слабый panel из `0.25` в
требуемый диапазон. Не масштабировать этот checkpoint/alpha дальше без нового
layout signal или специально обученного residual-after-h20 checkpoint.

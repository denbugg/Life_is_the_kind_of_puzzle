# TASKA structural-border cyclic origin

Дата фиксации: 2026-08-31. Статус: **opened gate fail; held300 не открыт**.

## Fixed hypothesis

Стартовый layout — только recovered focal `train_exact_top5`, без portfolio и
post-tail. Для каждого dirty tile bag заново вычисляется ранее audited
`structural_border_unary` по SHA-locked TASKA v3+local raw logits, slack
Sinkhorn `slack=6`, `20` iterations. Затем перебираются ровно все `24×24`
global cyclic rolls одного и того же layout и выбирается максимум суммы unary
на физических border positions. Exact ties разрешаются стабильным row-major
порядком `(row_roll, column_roll)`.

Никаких seam-cost/border-weight смесей, thresholds, target ids, filenames,
content labels или exact references selector не получает. Выбор меняет только
origin/cuts и сохраняет все относительные связи. Каждый результат — строгая
перестановка всех 576 исходных upright tiles; selected layouts и SHA roster
записаны до воссоздания exact reference.

## Почему это не повтор

- Закрытый `taska-exact-portfolio-proxy` использовал тот же structural unary,
  но выбирал **между focal и pair-leader layouts**; origin внутри layout не
  менялся.
- Закрытый `taska-all-bond-cyclic-origin` перебирал rolls, но ранжировал их
  **original seam cost**, а не structural border evidence, и стартовал с
  four-arm+tail96.

Следовательно, текущий тест — новый фиксированный composition этих двух
примитивов, а не повтор или weight sweep.

## Preregistered gate

Сначала открывается только исторический opened32. Unchanged held300 допускается
ровно если относительно focal одновременно:

- exact delta `> 0` tiles/board;
- pair delta `>= -2` satisfied pairs/board.

## Opened32 result

| Arm | Satisfied pairs | Recall | Exact tiles |
|---|---:|---:|---:|
| Focal top-5 unchanged | 335.50000 | 0.303894928 | 4.34375 |
| Structural origin | 323.62500 | 0.293138587 | 3.84375 |
| Delta | -11.87500 | -0.010756341 | -0.50000 |

Source-cluster bootstrap pair CI95: `[-14.25, -9.375]`. Selector изменил
origin на всех `32/32` cases, но ухудшил обе gate-метрики. Gate провален по
обоим условиям, поэтому held300 не запускался и Weco step 53 не создавался.

## Verdict

`closed-negative; no held transfer; no nearby sweep`.

Structural border unary не является пригодным absolute-origin scorer для
recovered focal components. Не следует подбирать slack, Sinkhorn iterations,
blend weight или roll threshold на opened32/held300. Нужен новый источник
translation-consistent absolute evidence.

Artifacts:

- `outputs/taska-structural-origin/opened32-mps-v1/report.json`;
- `outputs/taska-structural-origin/opened32-mps-v1/frozen-target-free-eval.npz`;
- `src/aiijc_puzzle/taska_structural_origin.py`;
- `scripts/run_taska_structural_origin.py`;
- `tests/test_taska_structural_origin.py`.


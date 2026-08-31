# TASKA confirmed-arm seven-layout portfolio

Дата: 2026-08-31. Статус: **отклонён независимым подтверждением**.

## Фиксированный вопрос

Проверена одна заранее зарегистрированная интеграция без learned selector и
без перебора: к подтверждённому six-arm fusion control добавлен отдельный уже
подтверждённый `fullres_union_focal` pre-tail arm. Итоговый roster ровно такой:

`raw / logistic / focal_top5 / nonlinear / selective_vote500_focal /
combined_union_focal / fullres_union_focal`.

Selector — неизменный минимум исходной суммы TASKA raw seam-cost по всем 1,104
реализованным bonds. После выбора каждого arm применён winner-aligned
focal-logit-`>=0` non-adjacent tail96. Для current arms используется current
supply, для selective/combined/fullres — соответствующий frozen union supply.
Six-arm control механически воспроизведён побитово на всех common `96/96`
cases. Threshold, weight, arm, budget, seed и roster не sweep-ились.

## Common-panel replay

| Panel | Six-arm control pairs / exact | Seven-arm pairs / exact | Pair delta, source CI95 | W/T/L |
|---|---:|---:|---:|---:|
| local32 | `326.781 / 5.938` | `327.781 / 3.969` | `+1.000 [-0.188,+2.406]` | `3/28/1` |
| held32 | `345.313 / 1.906` | `346.625 / 2.000` | `+1.313 [+0.030,+3.125]` | `4/27/1` |
| fresh32 | `355.625 / 0.938` | `356.625 / 1.000` | `+1.000 [-0.250,+2.469]` | `3/26/3` |

Local gate `>=0` и held gate `>=+0.5` выполнены, поэтому был открыт только
один заранее зарезервированный source16×draw2 formal panel. Положительные
common цифры считаются discovery evidence, но не promotion evidence: local
exact потерял `-1.969` tiles/board, а pair CI local/fresh пересекают ноль.

## Независимое подтверждение

Новый roster был подписан до inference в
`configs/taska_confirmed_arm_portfolio_v1.json`: 16 ранее не встречавшихся в
signed configs sources из `5700:6399`, каждый с draws `0/1`. На нём знак
развернулся:

| Metric | Six-arm control | Seven-arm candidate | Delta, source CI95 |
|---|---:|---:|---:|
| satisfied pairs | `314.34375` | `313.43750` | **`-0.90625 [-2.750,+0.625]`** |
| adjacency recall | `0.284731658` | `0.283910779` | `-0.000820879 [-0.002491,+0.000566]` |
| exact tiles | `1.25000` | `1.15625` | `-0.09375 [-0.34375,+0.15625]` |

Case W/T/L по pairs — `3/25/4`. Preregistered confirmation gate требовал
pair mean `>=+0.5` и source-CI lower `>=0`; обе части провалены. Это ещё одно
прямое проявление winner's curse all-bond selector-а: независимо хорошие arms
не становятся лучше от простого расширения final-layout portfolio.

## Решение и no-repeat

- Seven-arm portfolio не подтверждён и не должен заменять confirmed six-arm
  selective+unique-fullres fusion.
- Не добавлять к fusion ещё standalone layouts под тот же минимум raw
  all-bond cost. Исторический seed16 multistart уже показал тот же failure mode.
- Полезная fullres информация должна объединяться на уровне accepted-edge
  evidence до solve, как в confirmed combined fusion, либо требовать materially
  нового robust selector и новой панели.
- Не подбирать arm subset, cost weights, threshold или tail budget на открытых
  common/formal panels.

Все layouts — строгие permutations 576 исходных upright fragments. Restored
pixels использованы только matcher-ом; targets появились только после
target-free freeze. Competition test, postprocess, production и submission не
затрагивались.

## Артефакты

- common report: `outputs/taska-confirmed-arm-portfolio/fixed-v1/report.json`,
  SHA-256 `35905df43f6769d2dd75751ee88bff4e535397e07f521c51c79a748b796e103a`;
- formal report:
  `outputs/taska-confirmed-arm-portfolio/formal-source16-draw2-v1/report.json`,
  SHA-256 `7d1024b26afe4a2e031818fe19f09b7ca48cc249655e020805f33effb7fa2f1a`;
- module: `src/aiijc_puzzle/taska_confirmed_arm_portfolio.py`;
- common/formal runners: `scripts/run_taska_confirmed_arm_portfolio.py` и
  `scripts/run_taska_confirmed_arm_portfolio_formal.py`;
- preregistration SHA-256:
  `968c86e4d9dc6ad4fb67cee4ddd4064f4ebb7c2ad060e36b67bb1b38d177f3cc`;
- Weco Observe pair+exact: steps `109/110/111/112`, parent `102`.


# TASKA six-arm board-relative learned selector

Дата: 2026-08-31. Статус: **rejected on source-grouped local OOF gate**.

## Отличие от закрытых selector-веток

Прежний two-layout post-tail guard остановился до fit: selective выигрывал у
fusion лишь на четырёх local+held rows. Здесь использован полный roster из
шести independently focal-tail96-polished arms. Его pair oracle имеет заметный
потолок над confirmed fusion: `+4.875/+9.531/+6.688` pairs на
local/held/fresh. Перед этим простая adjacency-consensus эвристика дала
`-15.0` local pairs, поэтому agreement не использовался как самостоятельный
score; он вошёл лишь как два признака среди inference-visible confidence
features.

## Signed fixed contract

Preregistration SHA-256:
`8a7eaa63d3ab11fdb657f87c63852be28bf8dc4f3606ee8322b6ceb19ec25834`.

Roster фиксирован:

`raw / logistic / focal_top5 / nonlinear / selective_vote500_focal /
combined_union_focal`.

Каждый pre-tail layout независимо проходит неизменный focal-logit-`>=0`
non-adjacent tail96. Current4 используют current supply, selective и combined —
свои aligned union supplies. Control механически воспроизводит frozen confirmed
fusion: original raw-cost winner плюс соответствующий tail.

Для каждого arm извлекаются только target-free board-relative признаки:

- original all-1104 pre/post cost и tail gain;
- swaps, protected/free tiles, realised focal-positive edge counts;
- kept/realised focal-logit sums and means;
- среднее совпадение tile positions и directed adjacencies с пятью другими
  layouts.

Continuous features центрируются по шести arms одного board, затем добавляется
фиксированный six-way arm contrast. Единственная модель —
`StandardScaler + Ridge(alpha=1, fit_intercept=False)` на всех 30 ordered arm
differences каждого fit board; label — разность satisfied pairs. Никакого
alpha/feature/threshold/margin/roster sweep.

Local32 сначала полностью замораживает target-free arms/features. Evaluation —
фиксированный source-grouped 8-fold OOF. Gate для final all-local fit и held:
pair delta `>=0`, exact delta `>=-1.0`. Final model был сериализован до held, но
OOF gate не прошёл, поэтому held/fresh не открывались.

## Local32 OOF result

| Metric | Confirmed fusion | Learned selector | Delta, source CI95 | W/T/L |
|---|---:|---:|---:|---:|
| Satisfied pairs | `326.78125` | `325.93750` | **`-0.84375 [-3.3125,+1.46875]`** | `5/20/7` |
| Adjacency recall | `0.295997509` | `0.295233243` | `-0.000764266 [-0.002972147,+0.001302083]` | `5/20/7` |
| Exact tiles | `5.93750` | `5.65625` | `-0.28125 [-1.1875,+0.3125]` | `4/24/4` |

OOF choice counts: combined `15`, selective `11`, logistic `4`, nonlinear `1`,
raw `1`. То есть модель не повторила consensus collapse к четырём похожим
current arms, но всё равно не выбрала переносимые exceptions.

Pair-oracle воспроизведён точно: `331.65625`, то есть `+4.875` над control.
Однако exact у pair-oracle только `3.59375`, поэтому даже идеальный pair
selector не является автоматически exact-safe.

## Диагноз и no-repeat

Знаки восьми OOF folds нестабильны: `-1.0/+3.25/+0.25/-3.75/+3.0/+1.5/-5/-5`
pairs. При этом финальная all-local модель даже in-sample дала `-0.125` pair
delta против control. Значит, провал не сводится только к variance cross-fit:
фиксированный additive Ridge score и эти aggregate board features не выражают
arm-specific редкие выигрыши достаточно хорошо для argmax.

Не повторять на local32:

- alpha, feature-subset, arm-weight или confidence-margin sweep;
- logistic вместо Ridge на тех же aggregate rows;
- adjacency agreement как главный score;
- refit с held/fresh или post-hoc exact guard.

Возвращаться к learned whole-layout selection имеет смысл только с новым
заранее подписанным source-disjoint fit roster существенно больше 32 boards и
materially новым evidence/learning contract, например features, локализующие
конкретные спорные relations, а не ещё одной линейной комбинацией board sums.
Confirmed six-arm fusion остаётся лидером.

Все layouts — строгие permutations 576 исходных upright tiles. Pixels,
competition test, production, postprocess и submission не затрагивались.

## Артефакты

- report: `outputs/taska-six-arm-learned-selector/fixed-v1/report.json`,
  SHA-256 `9fb6354d3fe6588e90ca52cc3c950a366874c1151ea031e1631b4990be19fb62`;
- frozen model: `outputs/taska-six-arm-learned-selector/fixed-v1/model/`
  (`frozen-selector.npz` SHA-256
  `1b8aa5cb2f3b0321c4d9deab4d09baf7d55cc34197bcec47cea5e4c18cd15a39`);
- config: `configs/taska_six_arm_learned_selector_v1.json` и `.sha256`;
- module: `src/aiijc_puzzle/taska_six_arm_learned_selector.py`;
- runner: `scripts/run_taska_six_arm_learned_selector.py`;
- tests: `tests/test_taska_six_arm_learned_selector.py` (`4 passed`);
- Weco Observe pair+exact: step `116`, parent `102`; steps `117/118` не
  использованы.


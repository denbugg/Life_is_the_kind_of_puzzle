# TASKA focal nonlinear stacker: local gate negative

Status: **закрыт после disjoint local32; held/fresh не открывались**.

## Проверенная гипотеза

Один fixed `HistGradientBoostingClassifier` получил объединённый dirty-visible
контракт из 22 признаков:

- 15 признаков текущего TASKA harvest;
- recovered focal top-5 logit;
- шесть handcrafted top-5 focal-признаков.

Это изолирует ровно один вопрос: может ли тот же nonlinear contract, который
используется 15-feature TASKA arm, извлечь полезные взаимодействия из focal
evidence. Параметры скопированы без изменений и без sweep:

```text
loss=log_loss, learning_rate=0.05, max_iter=100,
max_leaf_nodes=15, min_samples_leaf=100,
l2_regularization=1.0, random_state=0
```

Модель обучена на первых 96 boards прежнего train256 roster: 36 022 harvested
edges, positive fraction `0.682389`. Независимые TASKA/focal caches совпали по
source order, board offsets и всем binary edge labels. Labels использовались
только при offline fit.

На inference модель меняет только порядок уже harvested edges. Candidate
membership, component placement, Hungarian fill, portfolio selection и
protected-tail polish продолжают использовать исходные TASKA costs.

## Local32 результат

Панель — 32 уникальных organizer-train sources из прежнего fixed slice
`96:128`, не пересекающиеся с train96 и с ранее открытыми opened/held/fresh
rosters. Все candidate layouts были сохранены и SHA-frozen до восстановления
exact references.

| Arm | Pairs / board | Recall / 1104 | Exact tiles / board |
|---|---:|---:|---:|
| nonlinear focal standalone | 309.40625 | 0.280259284 | 1.62500 |
| current four-arm + tail96 control | **314.37500** | **0.284759964** | 1.37500 |
| five-arm + tail96 candidate | 314.34375 | 0.284731658 | **1.40625** |

Candidate minus control:

- pairs: **−0.03125**, source bootstrap CI95
  `[−1.21875,+1.15625]`, W/T/L `2/26/4`;
- recall: `−0.0000283062`, CI95
  `[−0.001075634,+0.001075634]`;
- exact: `+0.03125`, CI95 `[−0.0625,+0.125]`, W/T/L `2/29/1`.

Original-cost selector выбрал новый arm на 6/32 boards, но после общего
tail96 это не дало положительной pair delta. Preregistered gate требовал
`five-arm pairs - four-arm pairs >= 0`; он провален буквально на одну пару за
всю панель. Поэтому held step 59 и fresh step 60 не создавались.

## Вердикт

Fixed HGB-вариант **не добавлять** в current pair pipeline. Результат почти
нейтрален, но чувствительный gate намеренно сохраняет отрицательный знак:
открывать held или подбирать depth/leaves/l2 на уже scored local32 означало бы
превратить bounded comparison в sweep.

Не повторять этот же 22-feature HGB contract с nearby hyperparameters на этой
панели. Если возвращаться к learned fusion, нужен materially другой training
objective, который ближе к downstream component/portfolio loss, либо новый
закрытый roster. Отдельный linear 22-feature stacker дал небольшой local pair
плюс и held exact signal, но его pair gain не перенёсся; это не основание
спасать HGB через подбор.

## Legality и воспроизводимость

- каждый layout — строгая перестановка всех 576 исходных upright tiles;
- target/exact permutation не входят в inference;
- denoised/generated/replaced output fragments отсутствуют;
- recovered focal читает только dirty boundary patches и matcher-derived
  top-5 features;
- competition test не открывался;
- frozen raw solver не менялся, SHA-256
  `97859e1f4ff6ceadf56ebccf429f2944ae7a5726631731d9f776224a667eb486`.

Главные артефакты:

- report: `outputs/taska-focal-nonlinear-stacker/train96-v1/report.json`,
  SHA-256 `59b6cc84a900e8f19e3555b7a1b8285a72d0959c70d6462d4470d6e0101febf2`;
- portable HGB NPZ:
  `outputs/taska-focal-nonlinear-stacker/train96-v1/focal-nonlinear-stacker.npz`,
  SHA-256 `fa971c70d38b07f102deffdfe51f29b0fd24a7ad118aa0e8818b046844686df9`;
- aligned train22 cache:
  `outputs/taska-focal-nonlinear-stacker/train96-v1/training-stacked-features.npz`,
  SHA-256 `6fa305c391779cc6ac76b93ce70c382205114e80a42ad2a03d4dba0927de33ac`;
- target-free local layouts:
  `outputs/taska-focal-nonlinear-stacker/train96-v1/local32-target-free.npz`,
  SHA-256 `870d31c2d53803550d02a468efe25dd8cdef13de55aa3b3dc976e642905d71cc`.

Код:

- `src/aiijc_puzzle/taska_focal_nonlinear_stacker.py`;
- `scripts/run_taska_focal_nonlinear_stacker.py`;
- `tests/test_taska_focal_nonlinear_stacker.py`.

Повтор в новом output directory:

```bash
.venv/bin/python scripts/run_taska_focal_nonlinear_stacker.py \
  --device mps \
  --output-dir outputs/taska-focal-nonlinear-stacker/replay-v1
```

Weco Observe: step 58 в pair run
`6bf52932-d716-4959-bee4-d652d7286cba` и exact run
`c2876967-cca7-44a6-83dd-1fca125c237e`. Steps 59/60 намеренно отсутствуют.

# TASKA selective target500 supply on focal-gated tail

Дата фиксации: 2026-08-31.

## Вердикт

Один фиксированный selective target500 consumer прошёл все pair gates и стал
новым сильнейшим pair-oriented TASKA вариантом на существующих
local/held/fresh panels. Production/default в этом эксперименте не менялся.

Относительно уже подтверждённого focal-gated current350 control pair delta
составила:

- local32: **`+9.21875`**, CI95 **`[+4.34375,+14.93750]`**;
- held32: **`+5.00000`**, source-cluster CI95
  **`[+1.31250,+9.62578]`**;
- fresh32: **`+5.75000`**, source-cluster CI95
  **`[+2.28125,+9.62500]`**.

Local gate `>=0` и held gate `>=+0.5` выполнены с запасом. Fresh candidate
достиг **`354.09375`** satisfied pairs и recall **`0.320737092`** против
`348.34375 / 0.315528759` у focal-gated control.

Exact не переносится как устойчивый gain: delta local/held/fresh равна
`+0.28125 / -1.43750 / -0.06250`; все exact CI пересекают ноль. Held mean
заметно отрицателен, поэтому arm сохраняется именно как pair-oriented solver,
не как exact/default решение.

## Фиксированный contract

На каждом board выполнен ровно **один** matcher pass с `vote_target=500`.
Из его vote records без нового neural forward выделен same-pass current350
subset. Дальше:

1. `new = target500 - current350`, с сохранением matcher edge order;
2. recovered focal verifier один раз считает `train_exact_top5` logits на всём
   target500 roster;
3. в fifth arm допускаются только новые edges с logit `>=0.0`;
4. union order — сначала current edges, затем принятые new edges; priorities —
   строго выровненные focal logits;
5. union образует один `selective_vote500_focal` raw-tail layout;
6. исходный all-1104-bond TASKA cost selector сравнивает его с неизменными
   `raw/logistic/focal_top5/nonlinear` layouts current350;
7. если выигрывает union arm, focal-gated tail96 получает union candidates;
   иначе он получает current candidates;
8. protection всегда сохраняет только edges с focal logit `>=0`.

Control строится из того же matcher pass: current four-arm winner плюс тот же
focal-gated non-adjacent tail96. Threshold, vote target, focal mode, arm roster,
tail budget и gates не подбирались.

## Mechanical smoke и контроль воспроизводимости

До scoring был выполнен target-free smoke1. Его same-pass current control
побитово совпал с известным focal-gated layout; exact reference не
восстанавливался.

На полном запуске same-pass control совпал с историческим focal-gated layout на
**96/96** случаях: `32/32` local, held и fresh. Соответственно, delta не вызвана
MPS replay drift или изменившимся control.

Для каждого panel matcher evidence, focal logits, edge rosters, strict layouts
и provenance записаны в NPZ/JSON и SHA-frozen до восстановления exact
references. Frozen raw solver SHA остался
`97859e1f4ff6ceadf56ebccf429f2944ae7a5726631731d9f776224a667eb486`.

## Метрики

| Panel | Focal-gated control pairs / exact | Selective target500 pairs / exact | Pair delta (CI95) | Exact delta (CI95) |
|---|---:|---:|---:|---:|
| local32 | `314.406 / 1.281` | **`323.625 / 1.563`** | **`+9.219 [4.344,14.938]`** | `+0.281 [-0.188,0.938]` |
| held32 | `338.094 / 3.000` | **`343.094 / 1.563`** | **`+5.000 [1.313,9.626]`** | `-1.438 [-5.938,1.250]` |
| fresh32 | `348.344 / 1.031` | **`354.094 / 0.969`** | **`+5.750 [2.281,9.625]`** | `-0.063 [-0.313,0.188]` |

Pair case W/T/L для local/held/fresh: `14/16/2`, `10/18/4`, `12/17/3`.
Source-cluster W/T/L для held/fresh: `8/6/2` и `9/5/2`.

Fifth arm выиграл original-cost selector на `16/32`, `14/32`, `15/32`
случаях соответственно. На остальных candidate буквально совпадает с current
four-arm focal-gated control.

## Candidate supply

| Panel | Proposed new / board | Accepted new / board | Accepted precision | Current recall | Union recall |
|---|---:|---:|---:|---:|---:|
| local32 | `156.906` | `39.844` | `62.27%` | `22.911%` | `25.159%` |
| held32 | `157.313` | `43.250` | `63.44%` | `23.933%` | `26.418%` |
| fresh32 | `161.813` | `46.500` | `65.12%` | `24.734%` | `27.477%` |

Прямой target500 ранее провалился, потому что добавлял весь lower-vote pool и
терял pair quality. Здесь focal intersection поднимает precision новых edges с
`26–30%` до `62–65%`; union recall растёт на `+2.25/+2.49/+2.74 pp` без
безусловного включения широкого шума. Это подтверждает исходную гипотезу
selective consumer-а, а не опровергает предыдущий negative target500 result.

## Legality

- Solver видит только dirty upright `20×20` tiles и target-free model evidence.
- Каждый scored layout — строгая перестановка всех 576 исходных fragments.
- Пиксели не восстанавливаются, не заменяются, не поворачиваются и не
  деформируются; postprocessing отсутствует.
- Clean organizer-train targets используются только после candidate freeze для
  offline scoring.
- Competition test не открывался.

## Решение и следующий шаг

Retain как ведущий pair-oriented current-TASKA candidate. Не подбирать
vote_target 400/450, focal threshold или tail budget на открытых panels.
Следующий materially distinct bounded check может объединить этот
independently confirmed lower-vote supply с confirmed full-resolution restored
supply, сохранив по отдельности фиксированные acceptance contracts.

Weco Observe pair+exact: local `90`, held `91`, fresh `92`; lineage начинается
от step `83`.

Artifacts:

- report: `outputs/taska-selective-vote500/fixed-v1/report.json`, SHA-256
  `2d9f328159c3c80280a0112f3fb663765c9b94fe650cd4639d8cc4ae5c1a0d18`;
- module: `src/aiijc_puzzle/taska_selective_vote500.py`, SHA-256
  `8bb23f6ff6402bfde3a2ec8701ea8ddffff86711fbd71e48993eb6d29a8e1fbc`;
- runner: `scripts/run_taska_selective_vote500.py`, SHA-256
  `4cc5e978da75889a952e07976c8339ebf83487fac31f24fd05384026eb70be13`;
- tests: `tests/test_taska_selective_vote500.py` and
  `tests/test_run_taska_selective_vote500.py`.

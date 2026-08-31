# TASKA focal и four-arm+tail96 на current-disjoint fresh32

Статус: **pair pipeline подтверждён; focal сохраняется отдельным
exact-oriented arm**. Это follow-up на уже открытой панели, а не fresh
promotion и не выбор параметров.

## Что проверялось

Без sweep были перенесены ровно два текущих лидера:

1. recovered focal verifier с training-exact `top_k=5`, который только меняет
   порядок уже собранных candidate edges;
2. четыре строгие layout — raw, train256 logistic, focal top5 и portable
   nonlinear — с target-free выбором минимальной суммы исходных TASKA costs на
   всех 1104 board bonds, затем fixed protected-tail `max_swaps=96`.

Каждый arm возвращает перестановку всех 576 исходных upright 20×20 tiles. Ни
один tile не вращается, не деформируется, не заменяется и не синтезируется.
Original TASKA costs неизменно используются для component placement, fill,
portfolio selector и tail objective.

## Freeze discipline и ограничение панели

Signed fresh32 recipe (`9854ef20…`) детерминированно восстановил те же dirty
bags. Matcher был повторно запущен только для отсутствующих в parent archive
`minimum_margin` и `vote_count`; edge roster совпал точно, а максимальная
разница с frozen cost matrices была `1.89e-6` (raw-log matrices совпали точно).

До реконструкции exact references в текущем процессе были записаны и
hash-frozen:

- focal logits/features/layout;
- logistic и nonlinear feature logits/layouts;
- raw, four-arm portfolio и tail96 layouts;
- portfolio choices/costs и protected-tail diagnostics.

Важно: exact targets этой панели уже были открыты прежним protected-tail
confirmation. Поэтому текущий результат только independent-current-process
target-free replay на source-disjoint roster, но **не формально свежая
валидация**. Весь last-300 range также исторически model-selection-exposed.

## Результаты

| Arm | Pairs / board | Recall | Exact tiles / board |
|---|---:|---:|---:|
| Raw TASKA | 339.75000 | 0.307744565 | 1.21875 |
| Logistic train256 | 341.34375 | 0.309188179 | 0.62500 |
| Focal top5 | **342.65625** | **0.310377038** | **1.62500** |
| Portable nonlinear | 341.46875 | 0.309301404 | 1.31250 |
| Four-arm portfolio | 345.12500 | 0.312613225 | 1.12500 |
| Four-arm portfolio + tail96 | **346.06250** | **0.313462409** | 1.15625 |

Source-clustered дельты относительно того же frozen raw:

| Candidate | Pair delta, CI95 | Source W/T/L | Exact delta, CI95 |
|---|---:|---:|---:|
| Focal top5 | +2.90625 `[-0.78125,+7.21875]` | 9/2/5 | +0.40625 `[-0.25000,+1.15625]` |
| Portfolio + tail96 | **+6.31250 `[+2.28125,+10.06250]`** | **13/0/3** | -0.06250 `[-0.96875,+0.96875]` |

Portfolio выбрал raw/logistic/focal/nonlinear `5/14/6/7` раз. Tail96
насытил cap в 21/32 cases. Этот saturation не разрешает подбирать больший cap
на уже открытой панели.

## Решение

- Для pair objective retained pipeline теперь имеет ещё одно положительное
  current-disjoint подтверждение: four-arm all-bond selector + tail96.
- Для exact objective focal top5 остаётся отдельным arm: на этой панели он дал
  лучший exact среди проверенных fixed layouts и положительный, но шумный,
  exact delta.
- Pair pipeline не заменяет focal в exact-oriented выборе; один selector не
  оптимизирует обе цели одновременно.

## Артефакты

- runner: `scripts/run_taska_fresh32_focal_portfolio_confirmation.py`, SHA-256
  `25dca478508a9db32402581027d44312363525803fe5429ea183f3f0900444fc`;
- tests: `tests/test_run_taska_fresh32_focal_portfolio_confirmation.py`, SHA-256
  `61fc91bb9e8cf394bfd71993a3c7d14f56c613593675ff5aa191ff92b9e93638`;
- target-free NPZ: `f3710cc3b00aaf2e75cb4127c280bc95eeeedf237f51a76ca234bac079c6f75f`;
- target-free metadata: `311a1b3dc42bfb317a2c5cde1cee319de86ceba85622cb376fe4bfb83e2b53b1`;
- pre-score freeze: `d88bae41096261c7ba97126977048db18e2dd4d31bf292ce73a693220ff66696`;
- report: `4db1e100f674613308f2cbca1e4be60c6687be3e9b3b446df16e442bc6cfbb7d`.

Machine-readable artifacts лежат в
`outputs/taska-fresh32-leader-confirmation/fresh-held32-mps-v1/`. Все 192
scored layouts (6 arms × 32 cases) strict. Targeted suite: 37 tests green;
Ruff green. Frozen `raw_tail_global_solver.py` остался SHA-256 `97859e1f…`.

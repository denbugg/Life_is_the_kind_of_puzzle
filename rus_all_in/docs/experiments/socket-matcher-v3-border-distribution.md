# SocketMatcher v3: border по распределению кандидатов — continuation rejected

Status: **the v3 implementation remains valid and opt-in, but the trained d64
continuation is rejected; keep d64 v2 as the default checkpoint**.

## Зачем нужен новый head

У v2 четыре отдельных border head видят только embedding одного socket-а.
На двух real train-dev панелях их exact-top-24 recall был всего `6.03–7.42%`,
тогда как border mass после partial OT давала `10.11%`. Это ожидаемый разрыв:
край изображения часто определяется не локальным видом границы, а отсутствием
хорошего партнёра среди остальных 575 tiles.

V3 добавляет к каждому из right-out, left-in, bottom-out и top-in logits шесть
дешёвых статистик соответствующей строки или колонки raw score matrix:

- лучший score и разницу top-1/top-2;
- `logmeanexp` и нормированную entropy;
- mean и standard deviation.

Статистики нормируются относительно остальных socket-ов той же board. Поэтому
head использует ровно известную относительную border cardinality, но не видит
индекс tile, координату slot, clean target или recovered label. При любом
одновременном переименовании tiles logits переименовываются тем же образом.

## Совместимость

Это opt-in архитектура `board-conditioned-partial-socket-matcher-v3` с флагом
`--border-head-version score_stats_v3`. Без флага остаётся
`embedding_v2`: default module tree и `state_dict` v2 не изменились.

Четыре новых линейных head-а создаются только для v3 и инициализируются нулём.
За счёт этого warm-start из v2 сначала полностью воспроизводит его outputs, а
новые параметры затем могут обучиться border loss-ом. Runner разрешает и
проверяет только объявленные переходы `v1→v2`, `v1→v3`, `v2→v2`, `v2→v3` и
`v3→v3`; лишние или недостающие state keys приводят к ошибке. Exact synthetic
evaluator также строго читает v3 contract и соответствующий module tree.

Фактически выполненный bounded continuation:

```bash
.venv/bin/python scripts/run_socket_matcher.py \
  --output-dir outputs/socket-matcher/v3-d64-continue-s500-r125-dev32 \
  --warmstart-in outputs/socket-matcher/v2-d64-train1024-s1600-r400-dev32/socket_matcher.pt \
  --border-head-version score_stats_v3 --raw-rank-weight 0.15 \
  --train-limit 1024 --eval-offset 3584 --eval-limit 32 \
  --synthetic-steps 500 --real-steps 125 --synthetic-grid 24 \
  --dimension 64 --heads 4 --board-layers 1 --socket-layers 1 \
  --sinkhorn-iterations 12 --learning-rate 0.0001 \
  --border-weight 0.5 --real-loss-weight 0.25 --schedule interleave
```

Он warm-started d64 v2 и добавил 500 synthetic + 125 real steps.  Checkpoint
v3 SHA-256 —
`a43b87116877f889c0a7c0997b0029985160a5ad740103e3af7f10f0cdfca263`;
родительский v2 —
`0e9df49a503c65aac7f1468e9acd6a074a5e658ae8b61f8954be086272c49670`.
Center/texture prior был выключен.

Важное ограничение атрибуции: run одновременно включил новый
score-distribution border head и raw-rank auxiliary `0.15`. Поэтому результат
оценивает весь continuation recipe; он не позволяет приписать дельту одному
border head.

## Строго paired exact-synthetic comparison

V2 и v3 были отдельно заморожены на **одних и тех же** 24 source-файлах × 2
draws, seed `20260903`, source digest
`0d67acd74ca44dbc9b5e26de73f84fc908a6a5f9cd3a0eb430b81eb000209e66`.
Все 48 case IDs, source/draw pairs и SHA-256 corrupted tiles совпадают; per-board
bilateral controls также совпадают. Источники исключены из полной exposure
lineage обоих checkpoints (`1056` v2 и `1088` v3 filenames). Predictions были
записаны и хешированы до exact inverse-shuffle scoring.

Paired intervals ниже — descriptive 20,000-sample bootstrap по 24 source
clusters (оба draws одного source ресемплируются вместе), seed `20260904`.

Authoritative reports:

- `outputs/socket-matcher/paired-v2-d64-vs-v3-source24-draw2/report.json`;
- `outputs/socket-matcher/paired-v3-d64-vs-v2-source24-draw2/report.json`.

### Local retrieval

Все значения ниже pooled right+down; дельта дана в процентных пунктах.

| Score | Metric | d64 v2 | d64 v3 | v3 − v2 |
|---|---|---:|---:|---:|
| raw | R@1 | 14.9287% | **15.2778%** | **+0.3491 pp** |
| raw | R@5 | 32.8087% | **33.5579%** | **+0.7492 pp** |
| raw | R@16 | 50.1019% | **50.8265%** | **+0.7246 pp** |
| raw | R@32 | 62.5981% | **63.2775%** | **+0.6793 pp** |
| partial OT | R@1 | 16.8799% | **17.1252%** | **+0.2453 pp** |
| partial OT | R@5 | 34.7184% | **35.0770%** | **+0.3585 pp** |
| partial OT | R@16 | 52.1984% | **52.4306%** | **+0.2321 pp** |
| partial OT | R@32 | 64.4192% | **64.8438%** | **+0.4246 pp** |
| fused OT | R@1 | 17.0460% | **17.3253%** | **+0.2793 pp** |
| fused OT | R@5 | 34.8845% | **35.1978%** | **+0.3133 pp** |
| fused OT | R@16 | 52.4475% | **52.6249%** | **+0.1774 pp** |
| fused OT | R@32 | 64.6418% | **64.9626%** | **+0.3208 pp** |

Source-clustered bootstrap 95% intervals are positive for every row. For the
main partial-OT R@1 delta the interval is `[+0.0887,+0.4001] pp`; the local
gain is therefore reproducible, but small.

### Global decoder144

Primary metric remains literal absolute tile identity, not adjacency.

| Metric, per board unless noted | d64 v2 | d64 v3 | v3 − v2 |
|---|---:|---:|---:|
| Correct absolute tiles, total / 27,648 | **59** | 44 | **−15 (−25.4%)** |
| Correct absolute tiles / board | **1.2292** | 0.9167 | −0.3125 |
| Direct placement | **0.2134%** | 0.1591% | −0.0543 pp |
| Correct rows | 25.2917 | **25.5000** | +0.2083 |
| Correct columns | **25.1250** | 24.6042 | −0.5208 |
| Translation-aligned tiles | 13.0833 | 13.0833 | 0.0000 |
| Right adjacency | 11.8131% | **12.0811%** | +0.2680 pp |
| Down adjacency | 13.4398% | **13.7455%** | +0.3057 pp |
| Pooled adjacency | 12.6264% | **12.9133%** | **+0.2868 pp** |

Adjacency gain is paired-positive: source-clustered 95% interval
`[+0.1057,+0.4831] pp`, 32/48 board wins. Exact-tile delta has interval
`[-0.9167,+0.3125]` tile/board and W/T/L `15/18/15`: its uncertainty crosses
zero, but the observed primary total falls by 15 while aligned placement is
exactly flat. Thus stronger local ranking converted to a small relative-graph
gain, not to better absolute coordinates.

## Fresh cyclic-border5 result

The post-decoder global-origin arm was also evaluated on a separate fresh v3
panel (24 sources × 2 draws, seed `20260902`):

| v3 arm | Exact total | Exact / board | Adjacency |
|---|---:|---:|---:|
| decoder144 | 43 | 0.8958 | 12.8642% |
| + cyclic border5 | 51 | 1.0625 | 12.8416% |

The exact gain is `+8` tiles / `+0.1667` per board, but its source-clustered
95% interval is `[-0.1250,+0.4792]`; W/T/L is `8/36/4`. This does not confirm
cyclic anchoring for v3. The earlier v2 cyclic experiment had a positive
`40→58` result and CI above zero, but used a different fresh panel, so the two
cyclic totals are not a direct paired model comparison. The two paired reports
above did not include cyclic post-processing.

Artifact:

- `outputs/socket-matcher/v3-d64-global-cyclic-fresh-source24-draw2/report.json`.

## Decision

**Reject `v3-d64-continue-s500-r125-dev32` as the next default checkpoint.**
It is not materially better on the primary exact-coordinate objective: exact
tiles fall `59→44`, aligned placement is flat, and the separate cyclic arm is
inconclusive. The small, well-supported local/adjacency gains are retained as a
research result but do not compensate for the primary regression.

Keep `v2-d64-train1024-s1600-r400-dev32` as Socket production/default input.
Do not tune v3 further on either opened panel. A future return to v3 would need
to isolate border-distribution training from raw-rank auxiliary and pass a new
source-disjoint **exact placement** gate, not merely local R@k or adjacency.

## Проверки реализации

Focused tests фиксируют shape и finite gradient статистик, permutation
equivariance, неизменность v2 `state_dict`, точное совпадение v2 и нулевого v3
после warm-start, gradient новых четырёх head-ов, runner transition и strict
load в exact synthetic evaluator. Эти implementation guarantees остаются
валидными; отрицательный verdict относится к обученному continuation recipe и
его promotion, а не к корректности кода v3.

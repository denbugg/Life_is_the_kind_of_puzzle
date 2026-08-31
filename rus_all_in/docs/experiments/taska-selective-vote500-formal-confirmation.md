# TASKA selective target500: formal disjoint confirmation

Дата фиксации: 2026-08-31.

## Вердикт

Неизменный `solve_selective_vote500` прошёл отдельную preregistered
source16 × draw2 confirmation. Pair delta против same-pass current350
focal-gated control составила **`+5.500`** пары/board, source-cluster CI95
**`[+0.813,+11.313]`**. Заранее заданный gate `mean >= +2.0` и
`CI95 lower >= 0` выполнен.

Candidate достиг **`354.28125`** satisfied pairs и recall **`0.320906929`**
против `348.78125 / 0.315925045` у control. Exact вырос
`2.59375 → 2.81250`, delta `+0.21875`, но CI95
`[-0.125,+0.625]` пересекает ноль. Результат подтверждает solver как
pair-oriented pipeline; он не является доказательством exact или official-SSIM
улучшения.

Поскольку pair gate прошёл, добавлен отдельный production-ready layout-only
adapter `taska_best_pair_pipeline.py`. Legacy `taska_pair_pipeline.py` не
заменён и не изменён.

## Пререгистрация и roster

До inference и scoring были materialized:

- config `configs/taska_selective_vote500_fresh32_confirmation_v1.json`,
  SHA-256
  `181d562e2d3cc337404608508e8fca6c25bbebf89d584e67744d30a961656628`;
- его `.sha256` sidecar;
- conservative exclusion snapshot
  `configs/taska_selective_vote500_fresh32_confirmation_v1.exclusions.json`,
  SHA-256
  `286000ad6f6d1e47932bee8e4e672c2bd7a856bf019346e28e95660a22ab561e`;
- отдельный sidecar snapshot-а.

Snapshot прочитал `178` существовавших TASKA config/output JSON artifacts и
собрал union всех явных `img_XXXXXX.png`: `356` sources, digest
`0da687471cc49a7ad75f4c5029994a869c30cd9ff5dae54933bba539b18ee61f`.
В union явно входят source rosters подписанных tail192 и fullres-combo
confirmation configs.

Universe намеренно не повторяет диапазон `6400:6699`: использованы только
organizer-train filenames `6700:6999`. В manifest это `242` sources; после
исключения `28` collisions осталось `214`. Из них SHA-256 ranking одного
фиксированного namespace/seed выбрал:

```text
006909 006976 006857 006771 006757 006982 006739 006924
006886 006722 006991 006952 006749 006899 006752 006988
```

Для каждого source использованы draws `0/1`, всего `32` cases. Это честная
current-TASKA-lineage disjointness; universal/model freshness не заявляется.

## Неизменный contract

На каждом board вызван ровно существующий
`aiijc_puzzle.taska_selective_vote500.solve_selective_vote500`:

1. один matcher pass с `vote_target=500`;
2. same-pass current350 subset без второго neural forward;
3. `new = target500 - current350`;
4. только new edges с recovered `train_exact_top5` focal logit `>=0` входят в
   `selective_vote500_focal` fifth arm;
5. original all-1104-bond selector сравнивает fifth arm с неизменными
   `raw/logistic/focal_top5/nonlinear` current arms;
6. control получает current four-arm winner + focal-gated non-adjacent tail96;
7. candidate получает five-arm winner + тот же tail96 с winner-aligned
   current/union roster.

Threshold, arm roster, tail budget, matcher target и gate не менялись. Код
подтверждённого solver-а остался с SHA
`8bb23f6ff6402bfde3a2ec8701ea8ddffff86711fbd71e48993eb6d29a8e1fbc`.

## Target-free freeze

До восстановления exact references были записаны оба final layouts,
current/proposed/accepted/union edge rosters, выровненные focal logits,
four/five-arm costs, choices, tail diagnostics и полный runtime/model
provenance:

- NPZ SHA
  `4c9b0967d381c90d80aabfd765a25bbf62aceeb04144f29b32303883127e8d62`;
- metadata SHA
  `da22056b0a82eaa16ff4d2be72d327b757f5c54c783b6e406cb847d7ce949668`;
- pre-score freeze SHA
  `371b4bcf29e96eba2ca815ad647b54ed35e221561820477a538f6a73a6927118`.

Все `64/64` scored layouts являются строгими перестановками 576 исходных
upright fragments.

## Метрики

| Arm | Pairs / board | Recall | Exact tiles / board |
|---|---:|---:|---:|
| same-pass current350 focal-gated tail96 | `348.78125` | `0.315925045` | `2.59375` |
| selective target500 focal-gated tail96 | **`354.28125`** | **`0.320906929`** | **`2.81250`** |

| Delta | Mean | Source-cluster CI95 | Case W/T/L | Source W/T/L |
|---|---:|---:|---:|---:|
| pairs | **`+5.50000`** | **`[+0.81250,+11.31250]`** | `7/23/2` | `6/9/1` |
| recall | `+0.004981884` | `[+0.000707654,+0.010246830]` | `7/23/2` | `6/9/1` |
| exact | `+0.21875` | `[-0.12500,+0.62500]` | `5/25/2` | `4/10/2` |

Fifth arm выиграл на `9/32` cases; на `23/32` candidate буквально совпал с
control. Средний current roster содержал `372.844` edges, additional proposal
`161.469`, accepted new `52.813`. Focal filter поднял precision proposed pool
с `34.12%` до **`69.17%`**; candidate recall вырос
`24.425% → 27.734%`.

## Production adapter

Новый модуль `src/aiijc_puzzle/taska_best_pair_pipeline.py`:

- byte-gates selective solver SHA;
- переиспользует существующий SHA-gated resource loader для v3/local matchers,
  logistic/focal/nonlinear artifacts и frozen raw solver;
- возвращает только read-only strict `tile_at_position` layout;
- прикладывает target-free diagnostics и confirmation receipt;
- не меняет pixels и не содержит postprocess.

Adapter SHA-256:
`89ed978bd90938cee27bd713d3524443071e5713daf1ba68401042e2c695cdc3`.

CLI принимает один `.npy` tile bag и создаёт layout исключительно:

```bash
uv run aiijc-taska-best-pair tiles.npy \
  --output-layout layout.npy \
  --diagnostics-json diagnostics.json \
  --device mps
```

CLI не имеет параметров vote target, threshold, arm или tail budget. Отдельный
shim: `scripts/run_taska_best_pair_pipeline.py`.

## Legality и Weco

- Candidate inference видит только dirty upright tiles и target-free model
  evidence.
- Competition test не открывался; submission не строился.
- Postprocess отсутствует; pixels не заменялись, не поворачивались и не
  деформировались.
- Weco Observe pair и exact: step `98`, parent `92` в обоих runs.

Authoritative report:
`outputs/taska-selective-vote500/fresh32-formal-confirmation-v1/report.json`,
SHA-256
`981d2ac218671bee4faaae090e24ebddaf7f075d1129ad9e562d218eec12bfc4`.

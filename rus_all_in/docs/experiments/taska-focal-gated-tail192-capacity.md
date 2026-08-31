# TASKA focal-gated tail96 → tail192: fixed capacity step

Дата: 2026-08-31. Вердикт: `pair gate failed; retain tail96`.

## Зачем был нужен этот эксперимент

На fresh16 confirmation focal-gated tail96 дошёл до лимита на `30/32` cases.
Поэтому до нового scoring был зарегистрирован ровно один capacity step:
тот же target350 matcher, четыре arm-а, original all-1104-bond selector,
focal protection `logit >= 0`, non-adjacent greedy swaps и minimum gain
`1e-9`; control ограничен 96 swaps, candidate — 192. Оба arm-а начинают с
одного и того же strict pre-tail layout и защищают побитово одно множество
realised edges. Threshold, budget, arm roster и panel не подбирались.

## Preregistration и collision audit

Config и SHA sidecar были материализованы до создания candidate-кода и до
scoring:

- `configs/taska_focal_gated_tail192_fresh16_capacity_v1.json`;
- config SHA `6ab12dfa...01db`;
- source-order digest `46818ecf...dac4`.

Новый deterministic SHA-ranked roster содержит 16 sources × draws `(0,1)`.
Проверка fail-closed закрепила SHA каждого входного roster/metadata и исключила:

- весь TASKA train256, extension128 и focal train224;
- local/opened/held/fresh и подтверждающий fresh16;
- full-resolution denoiser train32/eval16/terminal16;
- все filenames текущих active TASKA panels для focal-gate, fullres-union,
  fullres-focal и incidence-GNN веток.

Signed exclusion union содержит 382 sources; пересечение выбранных 16 с каждым
из 25 зарегистрированных roster-ов равно нулю. Это доказанная свежесть против
перечисленного current workspace lineage, но не утверждение universal
historical-model freshness.

Оба layout-а и provenance заморожены до reference reconstruction:

- frozen NPZ SHA `1be73be8...4ebb`;
- target-free metadata SHA `9ed7eab9...77a9`;
- pre-score freeze SHA `d509434c...a474`;
- raw solver SHA `97859e1f...486` остался неизменным.

## Результат

| Arm | Pairs/board | Adjacency recall | Exact tiles/board |
|---|---:|---:|---:|
| focal-gated tail96 | **323.09375** | **0.292657382** | 2.03125 |
| focal-gated tail192 | 322.78125 | 0.292374321 | **2.18750** |
| tail192 − tail96 | **−0.31250** | **−0.000283062** | **+0.15625** |
| source-cluster CI95 | `[-1.28125,+0.68750]` | `[-0.001160553,+0.000622736]` | `[0,+0.28125]` |

Primary pair W/T/L: cases `11/6/15`, sources `6/1/9`. Gate требовал pair mean
`>=+0.5` и CI lower `>=−0.25`; оба условия провалены. Exact secondary вырос,
но не может отменить pair-primary rejection.

Capacity действительно была использована: control достиг 96 swaps на `30/32`,
candidate достиг 192 на `9/32`; среднее число принятых swaps выросло
`95.03→149.50`, то есть на `+54.47`. При этом greedy original-cost продолжение
не переносится в true adjacency. Значит, cap96 — полезная регуляризация, а не
просто незавершённая оптимизация.

Absolute pair level этой новой панели ниже прежнего fresh16, поэтому сравнивать
между roster-ами `323` и `355` причинно нельзя; authoritative результат — только
paired tail192-minus-tail96 на одних случаях.

## Решение

Сохранить подтверждённый focal-gated tail96 как pair-default primitive.
Tail192 не добавлять в production и не sweep-ить рядом budgets или threshold на
открытом roster. Положительный exact signal можно хранить только как secondary
diagnostic; для exact-oriented применения нужен отдельный заранее
зарегистрированный selector, а не post-hoc смена primary metric.

Команда воспроизведения:

```bash
.venv/bin/python \
  scripts/run_taska_focal_gated_tail192_fresh16_capacity.py \
  --device mps --allow-nondeterministic-mps
```

Report:
`outputs/taska-focal-gated-tail-capacity/tail192-fresh16-v1/report.json`
(SHA `0b34cd73...16e0`). Weco Observe: step 93, parent 83, в pair и exact
tracks.

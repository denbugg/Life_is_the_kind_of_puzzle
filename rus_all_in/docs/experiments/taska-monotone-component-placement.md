# TASKA coordinate-only monotone component placement

Дата: 2026-08-31. Статус: **closed on opened32; held32 и fresh32 не
открывались**.

## Relation to prior step 18

Эта гипотеза близка к уже закрытому Weco step 18, поэтому до запуска был
проведён отдельный scope audit. Step 18 сохранял двухкомпонентные relocations,
но коммитил их только под objective guard, и измерял один raw arm:
`334.5625` pairs против raw control `334.71875` (`-0.15625`), exact delta
`+0.125`.

Текущий эксперимент materially отличается только в двух фиксированных
аспектах:

- unconditional two-component relocation loop полностью отсутствует, то есть
  число pair relocation attempts равно нулю;
- новый placement применяется ко всем current raw/logistic/focal/nonlinear
  arms, затем используется current original all-1104-bond selector и
  protected tail96. Контроль уже сильнее: `341.3125` pairs.

Matcher и focal inference не перезапускались: использованы frozen target-free
matrices, edge membership, 15-feature priorities и focal logits. Поэтому эта
проверка была дешёвой (`15.1 s` target-free на opened32) и не повторяла дорогой
matcher experiment.

## Fixed hypothesis

Frozen `raw_tail_global_solver.py` после largest-first initial placement делает
coordinate-wise best relocation каждого component, а затем без проверки
objective коммитит feasible random relocation двух components. Fixed candidate
меняет ровно последнюю часть:

- те же translation-consistent components и stable per-arm edge order;
- тот же largest-first initial placement;
- те же `6` rounds, `seed=0`, baseline quantile `0.15`;
- в каждом round остаётся только historical coordinate-wise
  single-component best relocation с strict row-major tie rule;
- pair relocation loop пропущен полностью;
- та же one-round seed-0 Hungarian fill;
- четыре arms `raw/logistic/focal/nonlinear`;
- minimum original TASKA cost по всем 1104 realised board bonds;
- тот же protected tail, `max_swaps=96`, `minimum_gain=1e-9`.

Никакого round/seed/threshold sweep не было. Preregistered gate требовал
nonnegative pair delta на opened32, чтобы открыть held32. Для открытия fresh32
требовалось ещё `>=+0.5` пары/board на held32 и отсутствие opened collapse.

## Legality и protocol

Каждый output — строгая перестановка всех 576 исходных upright `20x20` tiles;
пиксели не меняются и не рендерятся. Candidate membership, original costs и
component construction не изменены. Все layouts и metadata с SHA были frozen
до восстановления exact synthetic references. Targets использовались только
после freeze для offline scoring; competition test не открывался.

Frozen raw solver остался побайтно неизменным:
`97859e1f4ff6ceadf56ebccf429f2944ae7a5726631731d9f776224a667eb486`.

## Opened32 result

Panel: `16 sources x 2 draws`, pair denominator `1104`.

| Layout | Pairs / board | Recall | Exact tiles / board |
|---|---:|---:|---:|
| current four-arm + tail96 control | **341.31250** | **0.309159873** | **4.75000** |
| coordinate-only four-arm portfolio, pre-tail | 338.09375 | 0.306244339 | 3.71875 |
| coordinate-only four-arm + tail96 | 340.03125 | 0.307999321 | 3.81250 |

Final candidate minus control:

- pairs: `-1.28125`, source-cluster CI95 `[-4.0625,+1.53125]`, source
  W/T/L `7/0/9`;
- recall: `-0.001160553`, CI95 `[-0.003651495,+0.001387002]`;
- exact: `-0.93750`, CI95 `[-2.40625,0.0]`.

До tail разница была ещё хуже: `-3.21875` pairs, CI95
`[-6.5,+0.34375]`, и `-1.03125` exact, CI95 `[-2.53125,-0.09375]`.
Tail вернул в среднем `1.9375` пары, но не догнал control. Candidate selector
выбрал logistic/raw/nonlinear/focal на `11/8/7/6` boards соответственно, то
есть изменение было активно, а не no-op.

## Decision и no-repeat boundary

Opened pair gate провален, exact также ухудшился, поэтому held32 и fresh32 по
протоколу не запускались. Candidate не добавляется в production pipeline;
current four-arm+tail96 остаётся pair default.

Вместе со step 18 закрыты обе прямые трактовки исторического pair-relocation
дефекта на прежних components/objective: objective-guarded pair moves и полное
удаление pair moves. Не повторять nearby seed/round/guard sweep на этих
opened labels. Материально новый placement experiment должен менять сам search
method или использовать независимый selector/evidence, а не ещё один вариант
коммита тех же pair relocations.

Weco Observe: step 71 от parent step 42 в pair и exact tracks. Primary
pair metric — satisfied adjacent pairs per board, secondary — recall; exact
залогирован одновременно.

## Reproduction и artifacts

```bash
.venv/bin/python scripts/run_taska_monotone_components.py --panel opened32 --workers 4
```

- runtime: `src/aiijc_puzzle/taska_monotone_components.py`;
- runner: `scripts/run_taska_monotone_components.py`;
- tests: `tests/test_taska_monotone_components.py`,
  `tests/test_run_taska_monotone_components.py`;
- frozen target-free layouts, pre-score provenance and report:
  `outputs/taska-monotone-components/opened32-v1/`.


# TASKA focal-gated tail: new fresh16 confirmation

Дата фиксации: 2026-08-31.

## Вердикт

Фиксированная focal-защита **прошла независимый confirmation gate** и теперь
может быть promoted как pair-default вариант tail protection. Production-код в
этом эксперименте намеренно не менялся.

На новой current-lineage-disjoint панели `16 sources × 2 draws` control получил
`352.875` правильных соседних пар на доску, focal-gated candidate — **`354.750`**.
Paired delta равна **`+1.875`**, source-cluster CI95
**`[-0.1875,+3.84375]`**. Заранее заданные ворота `mean >= +0.5` и
`CI95 lower >= -0.25` выполнены. Case W/T/L — `24/0/8`, source W/T/L —
`11/1/4`.

Exact был только secondary metric и тоже вырос: `2.28125 -> 2.50000`, delta
`+0.21875`, CI95 `[-0.09375,+0.53125]`. Это не отдельное статистически
подтверждённое exact-улучшение, но и exact regression на этой панели не
наблюдается.

## Неизменный кандидат

До выбора новой панели были зафиксированы:

- current TASKA matcher `vote_target=350` на raw/median/bilateral;
- четыре layout arm-а `raw/logistic/focal_top5/nonlinear`;
- selector по минимальной сумме исходных TASKA costs на всех 1104 связях;
- один и тот же non-adjacent tail96 и исходный all-bond objective;
- control защищает все harvested candidate edges;
- candidate защищает только harvested edges с frozen recovered focal
  `train_exact_top5` logit `>= 0.0`;
- никаких изменений threshold, budget или roster arms.

Оба arm-а стартовали с побитово одного pre-tail layout на каждом случае.
Candidate менял только множество protected edges. Выход каждого arm-а — строгая
перестановка всех 576 исходных upright `20×20` tiles; пиксели не
перерисовывались, не заменялись, не поворачивались и не деформировались.

## Preregistration и freshness

Preregistration JSON и SHA sidecar созданы **до** scoring выбранных references:

- `configs/taska_focal_gated_protected_tail_fresh16_confirmation_v1.json`;
- config SHA-256 `d07744a88e4155e3a4f157cb896221d79d137c12379b7ae79bd9820f6024953c`.

Roster детерминированно выбран из `img_006400..img_006699` через
`sha256(namespace\0seed\0filename)` после исключения как минимум:

- полного TASKA train256 и отдельно local32 positions `96:128`;
- focal/current training224 positions `0:96 + 128:256`;
- opened32, held32 и предыдущего fresh32;
- full-resolution boundary denoiser train32/eval16/terminal16.

Пересечение выбранных 16 sources с каждым из этих roster-ов равно нулю. Это
честно обозначено как **current-lineage-disjoint**, а не universal/model fresh:
исторические эксперименты всего репозитория могли использовать часть того же
organizer-train universe в других целях.

Matcher, focal logits, оба строгих layout-а и provenance были заморожены до
восстановления exact references:

- frozen NPZ SHA `8dda17481929818e6f7cb9169c1bb9f7541e95f3ec61994776688c4425c9fb2f`;
- target-free metadata SHA `eb2e6881b6bec62fc42629b061fd7e8aba8d911e2b1fd935f5a28e93c83cb75b`;
- pre-score freeze SHA `0c11be99191bcd6699ecb9047baf34c637486dd6d90445535aaff822623ed198`;
- frozen raw solver SHA остался
  `97859e1f4ff6ceadf56ebccf429f2944ae7a5726631731d9f776224a667eb486`.

## Полные метрики

| Arm | Pairs/board | Adjacency recall | Exact tiles/board |
|---|---:|---:|---:|
| all-edge protection tail96 | 352.875 | 0.319633152 | 2.28125 |
| focal-logit-zero protection tail96 | **354.750** | **0.321331522** | **2.50000** |
| candidate − control | **+1.875** | **+0.001698370** | **+0.21875** |
| source-cluster CI95 | [-0.1875,+3.84375] | [-0.000169837,+0.003509964] | [-0.09375,+0.53125] |

Portfolio choices на 32 случаях: focal-top5 `10`, logistic `9`, nonlinear `7`,
raw `6`. Средний рост не является следствием одного единственного arm-а.

## Интерпретация

Предыдущий current-disjoint fresh32 уже дал `+2.28125` pairs с полностью
положительным CI, но тот roster был открыт в parent-run, а правило логит-ноль
было разработано после target-assisted local diagnostic. Новый неизменный
replay переносит положительный знак и проходит более мягкий заранее заданный
confirmation gate. Вместе две панели дают достаточное evidence для promotion
focal-gated protection как pair-default tail primitive.

Не следует после этого результата подбирать рядом threshold или swap budget на
новой панели. Следующий содержательный эксперимент — композиция уже
подтверждённой focal-защиты с independently confirmed additional matcher supply,
а не ещё один sweep того же gate.

## Воспроизведение

```bash
.venv/bin/python \
  scripts/run_taska_focal_gated_protected_tail_fresh16_confirmation.py \
  --device mps --allow-nondeterministic-mps
```

Machine-readable report:
`outputs/taska-focal-gated-protected-tail/fresh16-confirmation-v1/report.json`
(SHA-256 `850a4a8034d3d8f0b783391ad7cfade9661afd4ec695753a9fb07737b562edd3`).

Weco Observe: step `83`, parent `79`, в pair и exact runs.

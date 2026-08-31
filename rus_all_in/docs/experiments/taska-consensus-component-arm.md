# TASKA majority-bond consensus как component supply

Статус: **opened gate резко провален; held300 и fresh32 не открывались; ветка
закрыта без sweep**.

## Чем это отличается от protection

Это не повтор прежнего protected-tail/consensus-protection механизма. Там
layout сначала уже выбран, а polish лишь запрещает двигать tiles из
реализованных harvested edges. Здесь четыре независимо построенных pre-tail
layout — raw, train256 logistic, focal top5 и portable nonlinear — сначала
голосуют за все свои directed right/down board bonds. Связи с support `>=2`
становятся **новым supply самого translation-consistent component builder**.

Заранее зафиксирован единственный порядок, без threshold/weight sweep:

1. support `4 → 3 → 2`;
2. original TASKA priority `-cost`;
3. stable identity `(right before down, source tile id, target tile id)`.

После component build исторический raw-tail placer и Hungarian fill получают
неизменённые original `cost_right/cost_down`. Затем один раз применяется
protected tail с `max_swaps=96`, защищающий majority bonds, которые реализовал
consensus layout.

API не принимает target, clean image, filename или source-grid coordinate.
Все три frozen/scored layout каждого case — строгие перестановки 576 исходных
upright tiles.

## Freeze protocol

Для opened32 повторно использованы byte-pinned target-free TASKA matrices и
candidate membership, а также frozen focal-top5 layouts. Logistic и nonlinear
layouts воспроизведены неизменными portable calibrators. Consensus bonds,
component layouts, tail96 layouts и current four-arm-selector+tail96 baseline
записаны в NPZ до реконструкции exact references.

- target-free NPZ SHA-256:
  `993c2883b60ff337f02dd221e1be47a5ff531d976ecc0dbc59298d3aa80ce82d`;
- target-free metadata SHA-256:
  `006bb813e56a274344087b2e0ec73d009531b37e671c2b5e2d5e7850efedc481`;
- pre-score freeze SHA-256:
  `35e0ba36f7ed71ee88c04a4a114c11e6e0c7a51d98074876577c437f0764184a`;
- report SHA-256:
  `a84e8e41f528f269ab04134eeab15fa731f80eb8c8fb27d86974ee1b2179d4aa`.

## Opened32 результат

| Arm | Pairs / board | Recall | Exact tiles / board |
|---|---:|---:|---:|
| current four-arm selector + tail96 | **341.31250** | **0.309159873** | **4.75000** |
| consensus component, pre-tail | 318.21875 | 0.288241621 | 2.00000 |
| consensus component + tail96 | 322.84375 | 0.292430933 | 1.90625 |

Consensus-tail minus current-tail:

- pairs `−18.46875`, source-cluster CI95
  `[−26.06250,−10.96875]`, case W/T/L `7/0/25`;
- recall `−0.016728940`, CI95
  `[−0.023550725,−0.009963768]`;
- exact `−2.84375`, CI95 `[−7.75000,−0.12500]`, case W/T/L `8/9/15`.

В среднем retained supply содержал `825.44` bonds: `457.31` support-4,
`106.09` support-3 и `262.03` support-2. Builder принимал `759.53` edges и
пытался разместить `460.88` component tiles; tail защищал `462.78` tiles.
То есть majority rule не было слишком sparse. Наоборот, agreement четырёх
коррелированных solvers закрепляет крупные общие, но ошибочные геометрии. Tail
возвращает `+4.625` pairs относительно consensus pre-tail, однако не может
сломать защищённую wrong consensus core.

## Решение

Nonnegative opened gate провален с CI, целиком лежащим ниже нуля. Поэтому:

- held300 не запускался;
- fresh32 не открывался;
- support cutoff, arm weights, support-only tail protection и nearby priority
  варианты на этой панели не подбираются;
- текущий pair leader остаётся four-arm all-bond selector + tail96.

Материально новый возврат к consensus возможен только с независимым
edge-level evidence или cycle/purity gate, который отбрасывает correlated
shared errors до component build; простое layout agreement закрыто.

## Код и проверка

- module: `src/aiijc_puzzle/taska_consensus_component_arm.py`, SHA-256
  `af79386aedb8e6bcd959c99c736996826acdb4601f9009fb9c95e2bcb85bd43f`;
- runner: `scripts/run_taska_consensus_component_arm.py`, SHA-256
  `9d4e7ebf4314e918c574d324921ff5a233437e154773c41dbe8eec03bfc391ac`;
- tests: `tests/test_taska_consensus_component_arm.py`, SHA-256
  `31c632e369f3853c71e90e301ab2e235bb0ba9f8aaa0de1d7d1c244dae59707c`;
- artifacts: `outputs/taska-consensus-component-arm/opened32-v1/`.

Focused tests: 4/4 green; Ruff green. Frozen raw solver сохранил SHA-256
`97859e1f4ff6ceadf56ebccf429f2944ae7a5726631731d9f776224a667eb486`.

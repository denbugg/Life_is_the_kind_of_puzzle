# TASKA dense-top8 reciprocal translation consensus feasibility

Статус: **precision gate резко провален; solver не строился**.
Confirmed selective+unique-fullres six-arm fusion остаётся default.

## Новизна и fixed rule

Это pair-first feasibility continuation после joint component-pose pilot, а
не ещё один pose model. Он не повторяет unique-fullres translation
consensus: тот использовал редкий accepted unique-fullres suffix и нашёл
всего `8` edges на `4/32` boards. Здесь emitter — тот же frozen raw
TASKA top-8 contact roster, который дал высокий candidate coverage в joint-pose
cache.

До target scoring было подписано ровно одно правило:

1. Для каждого tile берутся frozen TASKA top-8 в outgoing/incoming
   right/down направлениях между разными realised focal-positive
   components final six-arm layout.
2. Physical edge считается только если он reciprocal: одно и то же
   `(source,target,axis)` есть и в outgoing, и в incoming top-8 после
   того же board-feasibility filter, что в joint cache.
3. Реализованные control-layout contacts отбрасываются. Остальные
   канонизируются по unordered component pair и одному implied integer
   relative translation.
4. Group остаётся, если есть минимум два distinct physical contacts и
   вместе они покрывают обе axis: right и down.

Top-k, support, focal threshold, reciprocity и axis rule не sweep-ились.
Diagnostic preregistration:
`configs/taska_dense_contact_consensus_feasibility_v1.json`, SHA-256
`ca1c04123e9a463d1f6176bee0a808a6aef01a772761ddc2b7b2f731020f6d62`.

## Freeze protocol

Target-free contacts для fit32 и source-disjoint local32 были записаны до
восстановления organizer-train references:

- archive SHA-256
  `d4ac06e1cb8cd67853dbc513968fa7e51f51cbf266e7691f4fc743e561e7b677`;
- metadata SHA-256
  `a77157fdd3318904ed434bf00df048c3211f357a05b8378b780e7353ecd7f069`;
- pre-score freeze SHA-256
  `96c6eecbec5adf115c1ce327de925eee431b1885f36f50157bec1b4390123537`.

Competition test, fresh panels, pixels, production и submission не открывались.

## Результат

Frequency gate требовал на каждой panel хотя бы `8/32` touched boards
и `2` emitted edges/board; precision gate — pooled edge precision строго
выше `60%`.

| Fixed feasibility metric | Fit32 | Local32 |
|---|---:|---:|
| boards with signal | `32/32` | `32/32` |
| groups / board | `87.344` | `81.656` |
| emitted edges / board | `210.094` | `195.375` |
| true emitted edges | `730/6723` | `726/6252` |
| pooled edge precision | `10.858%` | `11.612%` |
| true-missing edge coverage | `2.959%` | `2.990%` |
| all-true group precision | `7.871%` | `8.802%` |

Сигнал очень частый, но не high precision: даже двустороннее top-8
retrieval, agreement одного rigid translation и evidence обеих axes оставляют
почти девять ложных contacts на один true. Это объясняет, почему
joint-pose roster имел coverage/R@5, но почти не имел R@1.

## Decision / no-repeat

Precision gate провален более чем в пять раз на обеих panels. По
заранее записанному contract consensus-supply solver и layout не
строились; Weco step `126` не использован.

Не повторять nearby support/top-k/focal/reciprocity/axis варианты на этих
уже открытых panels. Dense top-8 contacts можно сохранить как broad
candidate roster, но не как independent hard-edge supply. Нужен materially
independent matcher/calibrator или learned context с реальным source-disjoint
top-1 signal.

## Артефакты и checks

- report:
  `outputs/taska-dense-contact-consensus-feasibility/fixed-v1/report.json`,
  SHA-256 `e201bdb2121360128cdef993990c2a8b7c56bb8a41b8614b6ff262e0a68bcfbf`;
- module SHA-256
  `9f6db260df286ec3bc110d019c6cd48363d4663e8d8ace29ece7a4a2c9b20b62`;
- diagnostic runner SHA-256
  `8a975408f47471a62a1c7101ee8ac1c8d92277af195d6ae9ce58ac5480051d72`;
- tests SHA-256
  `927cb31f7ffca8360e35039710696a7902d98a127a8888682046bdfdaa6d4897`;
- `3` focused tests, ruff and pycompile passed;
- Weco Observe feasibility step `125`, parent `102`; no solver metrics были
  приписаны несуществующему layout.


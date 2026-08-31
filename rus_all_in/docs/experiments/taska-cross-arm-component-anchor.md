# TASKA cross-arm absolute component anchor

Статус: **local exact/pair gate провален; terminal/fresh не открывались**.
Confirmed six-arm fusion остаётся default.

## Materially distinct hypothesis

Это не повтор relation-anchor или joint-pose:

- one-component relation anchor голосовал selected-supply edges и принимал
  move только при raw-seam improvement;
- dense consensus группировал raw TASKA top-8 contacts по relative translation;
- whole-arm consensus выбирал один готовый layout по adjacency overlap.

Здесь новый inference-visible evidence — **absolute coordinate agreement
нескольких independently assembled post-tail layouts** на всей геометрии
одной control component. Ни raw seam, ни DINO/population absolute unary, ни
center/background/frame prior не используются.

## Frozen rule

До target scoring подписан config
`configs/taska_cross_arm_component_anchor_v1.json`, SHA-256
`c39f5ee550fc4be854b91142e369285e70508f9501049f392582f58029feae4e`.

1. Control components строятся из realised selected-supply edges с focal
   logit `>=0`.
2. Каждый из fixed six post-tail arms голосует за component shift, только
   если **все** member tiles в этом arm равны control positions плюс
   один и тот же rigid integer shift.
3. Shift должен быть nonzero и поддержан минимум двумя distinct
   arms.
4. Двигается не более одной component. Fixed lexicographic ranking:
   `size*arm_support`, support, size, minimum L1 shift, stable identity.
5. Вытесненные tiles переносятся existing local bijective fill;
   выход — strict permutation всех 576 original upright tiles.

Ровно один support/ranking/fill rule, без sweep. Local gate: exact delta
строго `>0`, pair delta `>=-1`.

## Local32

Candidate и control были заморожены до organizer-train reference reconstruction.
Rule был частым: он нашёл в среднем `19.06` hypotheses/board и
изменил `32/32` layouts. Selected component имела в среднем `19.41`
tiles и `2.94` supporting arms.

| Local32 metric | Control | Candidate | Delta | W/T/L |
|---|---:|---:|---:|---:|
| exact tiles / board | `5.9375` | `5.6875` | **`-0.2500`** | `4/23/5` |
| satisfied pairs / board | `326.7813` | `318.4063` | **`-8.3750`** | `0/3/29` |
| adjacency recall | `29.5998%` | `28.8411%` | `-0.7586 pp` | `0/3/29` |

Оба gate провалены. Все 64 frozen control/candidate layouts прошли
independent strict-permutation audit.

## Вывод / no-repeat

Совпадение абсолютных component positions у двух-четырёх TASKA arms
не является independent origin evidence. Arms наследуют correlated
packing/gauge errors; full-component agreement точно переносит wrong islands. Без
pair-preserving guard one-component move также ломает много boundary bonds.

Не повторять support=3/4, component-size cap, weighted arm vote, alternate
ranking, multi-component pack или fill на той же local panel. Raw-seam guard для
одного component уже отдельно закрыт relation-anchor experiment. Нужен
independent absolute evidence, а не ещё одна consensus форма correlated arms.

## Артефакты

- report:
  `outputs/taska-cross-arm-component-anchor/local32-v1/report.json`, SHA-256
  `2ddc08db57592b5d3daf45f2a43904c39651fcbb53a2dcecad9aeede8a21d222`;
- frozen target-free archive SHA-256
  `03e51c0f46f942196f9b9a0b25528ef4bd230de343fabeea77b16b939c23e2a9`;
- pre-score freeze SHA-256
  `6fdbac6ffa0f39acfcfcf3736dcc1ee8dda1aab19b2e0e73f5be5c1aadc744fc`;
- module SHA-256
  `9c2ff32757d5753f2f8ffb4dbd4b929d370bddf721d2c75dbc82f3bcc8af5d3c`;
- runner SHA-256
  `b82ca6cd6a60571dbf37e8ed505ac53fbf84e866956ee2758483af3bdfd3a25b`;
- tests SHA-256
  `c6e91325d3f6f3718690fcace2d95faf783e5f2cb41cd40278d03cf25b066bb7`;
- 8 focused anchor tests and ruff passed;
- Weco pair+exact step `133`, parent `102`; step `134` не использован.

Terminal/fresh, competition test, pixels, production и submission не затронуты.


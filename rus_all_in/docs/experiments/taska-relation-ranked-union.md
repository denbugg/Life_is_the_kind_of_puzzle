# TASKA HGB-ranked six-arm relation union

Дата: 2026-08-31. Статус: **rejected at preregistered local gate**.

## Fixed вопрос

Подтверждённый relation HGB хорошо различает truth для realised seams и полезен
как selector целого layout. Этот опыт проверил materially другой consumer:
можно ли из его локальных вероятностей собрать **новый** layout, а не выбрать
один из шести готовых.

До candidate construction/scoring был подписан contract SHA-256
`2f0beb7fc071f4aef673267fab348baaeccdb664048e2ac2154edf45fa2723a7`:

1. Frozen HGB оценивает все `6×1,104` occurrences в шести post-tail arms.
2. Одинаковые `(axis,source,target)` deduplicate-ятся по maximum `p`; точный
   tie сохраняет первый fixed arm/relation occurrence.
3. **Все** unique edges сортируются по убыванию `p` и подаются в unchanged
   `solve_prioritized_raw_tail_global` с исходными cost matrices, placement,
   Hungarian fill и `SOLVER_CONFIG`.
4. Threshold, top-k, weight, model, tail и solver parameter отсутствуют; sweep
   запрещён.
5. Comparator — final frozen whole-arm relation selector. Local/held заранее
   помечены как HGB-in-sample mechanical panels, fresh — как opened
   model-selection-exposed development, не confirmation.

Gates были staged: local pair delta `>=0`, held `>=0`; только после них fresh
требовал pair delta `>=+1`, source-CI lower `>=0`, exact `>=-1`. Новый
source16×draw2 confirmation разрешался только после прохождения всех gates.

## No-repeat audit

Точного повтора не найдено. Focal nonlinear stacker ранжировал другой
pre-layout harvested supply другим 22-feature HGB; Union-hard priority работал
в Union-v2 decoder lineage; component-relation confidence использовал capped
Socket queries и bonus. Здесь впервые объединялись realised relations всех
шести post-tail TASKA layouts именно confirmed relation-truth HGB. Общая идея
learned edge ordering старая, consumer и evidence lineage новые.

## Local32 результат и stop

Frozen final HGB обучался в том числе на local32, поэтому результат —
оптимистичный in-sample/mechanical upper screen, не переносимость.

| Metric | Whole-arm selector | All-edge union | Delta, source CI95 | W/T/L |
|---|---:|---:|---:|---:|
| Satisfied pairs | `326.750` | `199.500` | **`-127.250 [-145.625,-108.530]`** | `1/0/31` |
| Adjacency recall | `29.597%` | `18.071%` | **`-11.526 pp [-13.219,-9.845]`** | `1/0/31` |
| Exact tiles | `5.688` | `2.219` | `-3.469 [-9.313,+0.750]` | `10/10/12` |

Все `64/64` comparator/candidate layouts были strict permutations исходных
upright tile IDs. Но primary gate провален с огромным отрицательным запасом.
Held32, fresh32 и formal confirmation **не создавались и не оценивались**.

Механика провала соответствует заранее отмеченному риску max-over-contexts:
из `6,624` occurrences оставалось в среднем `3,991.8` unique edges
(`3,361..4,631`). HGB был обучен оценивать correctness relations внутри
конкретного arm/context, а не калибровать многотысячный pooled tail. Даже
низкоранговые relations обязаны войти из-за all-edge contract и создают много
взаимно противоречивых rigid constraints. Это не отрицание edge AUC или
whole-arm selector; провален именно безусловный all-edge consumer.

## Решение и no-repeat

Не спасать этот result post-hoc подбором threshold/top-k, probability transform,
arm weight или tail: это превратило бы один signed вопрос в sweep на scored
in-sample panel. Если возвращаться к layout synthesis, нужен отдельно
пререгистрированный objective/decoder, который моделирует совместимость набора
relations, а не независимый max `p` каждого edge.

Competition test, matcher, denoiser, pixel output, production, default и
submission не затрагивались. Weco pair+exact step `155`; reserved `156/157/158`
не создавались из-за local stop.

## Артефакты

- report SHA-256:
  `cee46746aac04e19e4c008d46092611cdd0cc32dfbd1eea6951eab9988d4cca4`;
- target-free local archive SHA-256:
  `f40b337ffef717d767862b39b74ced97fae689bf30c6f8cf5df376e94a17d2a9`;
- pre-score freeze SHA-256:
  `0c0c39649a6e098f2ec282c928c3e448ce714c8286c672e44132b99e0d7f87f8`;
- solver source SHA-256:
  `426e7ab5ff207109343f0d158e2555bb7a9d243f5638b255ec393c8c98345a90`;
- runner SHA-256:
  `d27ebe82e00ced7db8c3e4c366492a61b101ae2eb90cb9bfe0736cc18c23b048`;
- candidate/runner tests: `7 passed`; Ruff passed.

# Joint verifier → frozen relation-selector consumer

Дата: 2026-08-31. Статус: **implementation ready; intentionally unsigned and
blocked before any DEV pixel/reference access**.

## Зачем нужен отдельный consumer

Joint reciprocal verifier выдаёт новый target-free edge signal, но сам по себе
не является global layout solver. Текущий подтверждённый pair leader — frozen
relation-level HGB selector одного из шести целых TASKA post-tail layouts. Его
formal confirmation дала `+5.844` satisfied pairs/board без pair losses, хотя
exact снизился на `0.156` tile/board.

No-repeat аудит перед реализацией зафиксировал два важных отрицательных
результата:

- безусловный HGB-ranked union realised relations уже дал `-127.25` pairs на
  local32: высокий relation AUC нельзя превращать в новый all-edge decoder;
- six-arm aggregate Ridge и adjacency consensus также провалились, поэтому
  nearby linear/consensus selector или weight sweep не повторяются.

Поэтому consumer не синтезирует ни одного edge или layout. Он может выбрать
только один из шести уже frozen strict layouts:

`raw / logistic / focal_top5 / nonlinear / selective_vote500_focal /
combined_union_focal`.

## Fixed target-free mapping

Joint archive обязан сохранить exact v2 schema для:

- immutable `union_candidates`, `union_valid`, `emitter_topk` и их SHA-256
  identity digest;
- `learned_logits__right/down`;
- `learned_joint_confidence__right/down`;
- ровно 29 selected fixed-5% reciprocal edges на каждую ось с source, target и
  confidence.

Каждый head edge проверяется как уникальная `(source_tile_id, target_tile_id)`
пара в immutable union и должен побитово соответствовать сохранённому joint
confidence slot. Unknown arrays, `target_slots`, truth/reference/label arrays и
изменённый digest отвергаются.

Same-case relation-roster sibling freeze имеет минимальный normalized schema:

- шесть `relation_arm_*_layout`, каждый strict `int[576]` permutation;
- frozen incumbent `relation_truth_selector_layout` и выбранный arm;
- исходные HGB `relation_features[6,1104,29]` и
  `relation_expected_correct_scores[6]`.

Joint и roster rows обязаны точно совпасть по
`case_id/source_filename/draw_index/dirty_sha256`. Это одновременно фиксирует
tile-bag identity: source/target IDs joint verifier-а и IDs layout solver-а не
могут быть незаметно переставлены.

Для каждого arm считаются только inference-visible diagnostics:

- сколько его 1,104 realised edges присутствуют в joint union;
- сумма их learned logits;
- число realised fixed-head edges отдельно справа и вниз;
- суммы confidence и learned logits этих head edges.

Switch с incumbent разрешён только если arm имеет не меньше fixed-head hits на
**каждой** оси и строго больше hits суммарно. Среди eligible arms fixed
lexicographic order максимизирует total hits, minimum per-axis delta, summed
joint confidence, summed head logits, frozen HGB score; последний tie-break —
неизменный arm order. Если eligible arm нет, сохраняется incumbent.

Такой rule не утверждает, что head hit является true pair. Он лишь не позволяет
обменять right signal на down signal и ограничивает evaluation уже существующим
whole-layout roster. Output всегда буквально один входной layout, без
orientation/pixel/content changes.

## Fail-closed sequencing

Runner: `scripts/run_joint_relation_selector_consumer.py`.

1. `freeze` сначала SHA-проверяет joint pre-score freeze и отдельный
   relation-roster pre-score freeze, отрицательные label/pixel flags, schemas,
   source order и все case identities.
2. После mapping он эксклюзивно записывает только integer control/candidate
   layouts и target-free arm diagnostics.
3. Новый `pre-score-freeze.json` связывает оба input bundle и output
   archive/metadata.
4. `score` сначала повторно проверяет hashes. Только затем отдельный callback
   может восстановить exact organizer-train reference.
5. Scoring считает satisfied directed pairs из 1,104, absolute exact tiles,
   absolute mean Manhattan и radius-2 recall. Cyclic-aligned distance не
   используется.

Focused test портит frozen archive и доказывает, что reference loader не
вызывается. Другие tests проверяют candidate/head identity schema, запрет
скрытого `opaque_y`, strict permutation, успешный dominance switch и отказ от
right↔down trade.

## Один bounded preregistered evaluation

Unsigned template:
`configs/joint_relation_selector_consumer_unsigned_template_v1.json`.
Он уже фиксирует exact v2 DEV source32×draw0 roster, corruption seed, arm order,
mapping и один Pareto gate, но содержит pending hashes двух будущих target-free
siblings. Template нельзя подписывать in-place: после появления архивов reviewer
должен проверить их lineage и создать отдельный signed config.

Предлагается ровно один run без threshold/weight/roster/tie-break tuning:

- control: frozen HGB relation-selector incumbent;
- candidate: fixed axiswise-head-dominance whole-arm consumer;
- требуется хотя бы один changed case и хотя бы одно strict aggregate
  improvement;
- mean pair delta `>=0`;
- mean exact delta `>=0`;
- mean absolute Manhattan delta `<=0`;
- mean radius-2 delta `>=0`.

Это намеренно строгий Pareto discovery gate. Если он не пройден, branch
останавливается: нельзя по открытым labels ослаблять safety bounds, менять
5%-coverage, добавлять scalar weights или собирать layout из selected edges.
Если пройден, результат всё равно остаётся organizer-train solver evidence, а
не leaderboard/submission claim.

## Текущее отсутствие доступа

На момент реализации joint v2 FIT не менялся и не перезапускался. Joint DEV,
same-case relation roster, DEV references, прежний local, terminal16 и
competition test не открывались. Никакой final config, sidecar, endpoint,
selection archive или score report не создавался.

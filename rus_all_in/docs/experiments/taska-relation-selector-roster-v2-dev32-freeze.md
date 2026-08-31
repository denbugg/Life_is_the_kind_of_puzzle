# Target-free six-arm roster freeze for joint-v2 DEV32

Дата: 2026-08-31. Статус: **implemented, unsigned and blocked before first DEV
pixel access**.

## Назначение

Joint verifier DEV archive хранит edge evidence, но не содержит pixels или
готовые global layouts. Консервативному joint→relation consumer поэтому нужен
same-case sibling: шесть уже существующих TASKA post-tail layouts и frozen HGB
relation-selector evidence в том же shuffled tile-bag ID space.

Freezer не создаёт новый solver. Он повторно использует ровно текущий
production `aiijc-taska-relation-selector`, который SHA-gates:

- selective-target500 + unique-fullres six-arm parent;
- matcher/focal/calibrator/denoiser resources;
- relation HGB model `ec4eca99…`;
- development и formal-confirmation evidence.

Это принципиально сохраняет no-repeat решение: rejected HGB-ranked all-edge
union, aggregate Ridge, adjacency consensus и новый threshold/weight sweep не
возвращаются. Freezer только экспортирует уже вычисленные whole-arm candidates.

## До-пиксельный fixed contract

Unsigned template:
`configs/taska_relation_selector_roster_v2_dev32_unsigned_template_v1.json`.

Rule commitment `723455a54dc29708b91689544e8bb996c69138ae58a0a8de728bd80adff4cf97`
фиксирует независимо от будущего output hash:

- exact joint-v2 DEV source32 order/digest;
- draw `0`, corruption seed `20260908`;
- MPS и inference batch `576`;
- неизменный SHA-gated pipeline, без model/threshold tuning;
- six-arm order, все 29 HGB relation feature names и 1,104 rows/arm;
- exact normalized NPZ key allowlist;
- только strict permutations 576 original upright tile IDs, без pixels или
  labels/references.

Template не исполняется и не подписывается in-place. Reviewer должен создать
отдельный signed config + sidecar до первого обращения к DEV target pixels.

## Target-free generation

Runner:
`scripts/freeze_taska_relation_selector_roster_target_free.py`.

После проверки config sidecar и всех runtime SHA он:

1. проверяет pinned validation manifest и exact v2 source order;
2. читает только эти 32 organizer-train source images и через тот же
   `make_target_free_synthetic_case` создаёт dirty shuffled bag;
3. не вызывает `make_exact_synthetic_case` и не строит inverse-shuffle
   reference;
4. запускает existing matcher/fullres/six-arm/HGB inference;
5. отбрасывает все промежуточные parent arrays и сохраняет только:
   `relation_features`, `relation_expected_correct_scores`, frozen incumbent и
   шесть `relation_arm_*_layout`;
6. заново проверяет каждый arm как strict permutation и соответствие incumbent;
7. эксклюзивно пишет archive, metadata и `pre-score-freeze.json` с
   `case_id/source/draw/dirty_sha256`.

Archive не содержит RGB, raw/denoised tiles, costs, edge labels, truth,
`target_slots`, exact reference или score. Dirty hash позволяет downstream
consumer fail-closed связать roster с independently frozen joint case.

## Tests и границы

Focused tests проверяют exact allowlist, удаление лишних target-free parent
arrays, отказ от скрытого `target_slots`, invalid permutation, incumbent drift,
изменение rule commitment и overwrite. Они используют только synthetic 3×3
arrays и не открывают ни один organizer source.

На момент документа DEV pixels, labels/references, старый local, terminal16 и
competition test не открывались; pipeline inference и scoring не запускались.

# Scale256 DEV64 target-free six-arm roster freeze

Дата: 2026-08-31. Статус: **unsigned and blocked before DEV64 pixel access**.
Organizer pixels, labels/references, MPS inference, terminal и competition test
не открывались и не запускались при подготовке этого protocol scaffold.

## Назначение

[`freeze_taska_relation_selector_roster_scale256_target_free.py`](../../scripts/freeze_taska_relation_selector_roster_scale256_target_free.py)
— тонкая fail-closed обёртка над неизменённым
[`freeze_taska_relation_selector_roster_target_free.py`](../../scripts/freeze_taska_relation_selector_roster_target_free.py).
Она не реализует новый solver и не меняет inference: после проверки protocol
делегирует непосредственно `freeze_target_free_roster`.

Обёртка нужна из-за различия двух уже signed scale contracts:

- scale256 real protocol хранит только `source_contract` с count/digest и
  фиксированными `draw=0`, `case_seed=20260908`;
- scale-cache protocol хранит явный ordered `source_protocol`, включая все 64
  reserved DEV filenames.

Перед будущим запуском wrapper требует точное совпадение обоих contracts:
`64` уникальных sources, digest
`5c6cb5b9b204a38c78e79936ff34235dae9896cfc13d6edaf12dfad635bcdb8e`,
draw `0`, seed `20260908`, одинаковую scale-cache lineage и отсутствие
FIT/DEV overlap.

## Frozen runtime и output

Runtime остаётся существующим SHA-gated MPS relation-selector pipeline:
inference batch `576`, без tuning и без нового model. Output содержит ровно
шесть неизменённых whole-layout arms, frozen HGB relation features/scores и
incumbent. Каждый layout — строгая перестановка всех original upright tile
identities. Pixels, restored views, exact references и labels не сохраняются.

Config
[`taska_relation_selector_roster_scale256_dev64_unsigned_template_v1.json`](../../configs/taska_relation_selector_roster_scale256_dev64_unsigned_template_v1.json)
фиксирует signed scale и scale-cache configs, validation manifest, board loader,
production pipeline, relation model, confirmation config/report/runners, joint
case runner, а также byte hashes base freezer и wrapper. Base freezer сам пишет
hash preregistration config в pre-score freeze; поэтому wrapper bytes связаны с
output ещё и транзитивно через config, а base bytes записываются напрямую.

## Stop boundary

Checked-in template намеренно не имеет sidecar и немедленно отказывает со
статусом `unsigned-template-blocked-before-dev64-pixel-access`. Его нельзя
подписывать или запускать in place. После review можно создать отдельный signed
config, заново проверить все frozen hashes, заменить только status и создать
sidecar **до** первого DEV64 pixel access. Freeze должен завершиться до любой
реконструкции reference или score; изменение roster, draw, seed, runtime,
allowlist либо любого frozen artifact требует нового review.

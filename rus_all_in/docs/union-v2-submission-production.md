# Union-v2 submission production contract

Статус на 2026-08-30: **production run запущен после frozen fresh64 promotion**.
Metadata-only MPS dry-run завершён успешно; итоговые ZIP/attestation hashes будут
добавлены после встроенной независимой проверки полного roster-а. Старый
`outputs/compliant-submission/` не изменяется: Union-v2 публикуется в отдельный
`outputs/union-v2-submission/`.

Успешный validation status намеренно не называется доказательством правильной
раскладки. `METHOD_COMPLIANT_LAYOUT_ACCURACY_UNPROVEN` доказывает exact input,
строгую перестановку исходных fragments, frozen model/tail lineage и повторное
исполнение pipeline, но не hidden ground truth, leaderboard score или решение
ручной комиссии.

## Единственный production arm

Layout использует frozen Socket d64, full-resolution Twin и Union-v2 selector:

1. соответствующий RGB input 480×480 делится на 576 исходных upright tiles
   20×20 без rotation, resize или warp;
2. `raw32 ∪ twin32 ∪ frozen-raw-hard-projection` проходит frozen Union-v2 head;
3. outside-union edges запрещены, restricted partial OT и exact hard projection
   подаются в неизменённый `decoder144`, затем применяется `cyclic-border5`;
4. layout обязан быть строгой перестановкой 0…575; до restoration runner
   пересобирает raw canvas только из соответствующих original tiles и проверяет
   pixel/tile multiset audit;
5. после audit выполняются ровно additive RGB seam offsets, bounded luminance
   gains и один proper-colored OpenCV NLM pass `h=hColor=20`, windows `7/21`.

Запрещены targets, source lookup, clean references, filename/board overrides,
templates, tile substitution (включая constant/near-flat substitution), чужие
board pixels и любые geometric tile transformations. Restoration не меняет
layout.

## Frozen lineage

| Artifact | SHA-256 |
|---|---|
| official `test.zip` | `62d365c45fe85c3da06e96f83390e7bb056935036a9b5dee7a99d32f11483c89` |
| sorted official filenames | `312e8c46b2ccfa27e525d607d046d0e3676688f8c71533b8498c377d71805376` |
| Socket d64 checkpoint | `0e9df49a503c65aac7f1468e9acd6a074a5e658ae8b61f8954be086272c49670` |
| fullres Twin checkpoint | `c5b44901e8da459e3c48b6e7af7153c5d7eed26f1c1b52c8712c4fa0dc4ea8ae` |
| Union-v2 checkpoint | `a5f882ab3c827e4e3779be3372c62d2a8fb9cd95d3558fd30cc566a9c3137f79` |
| Union-v2 preregistration | `6741e92e832a630f1b83bde6edc8a341a348f52daa82313c40a8f32c7c1173d4` |
| Union-v2 selection commitment | `71ae4f5095489613857fcd25c541fe496da0d6861f6ff604850147dd04b91cd2` |
| production config | `0d58b59915a5797db0ec4ac956fb2180fea2aed4df8d8f19bc02795924311aad` |
| attestation JSON Schema | `6bcf12ab940e64aeb7afe954f5c316e6c3c767062d517afdab877443103a18bb` |

Frozen production/runtime entrypoints:

| File | SHA-256 |
|---|---|
| `src/aiijc_puzzle/union_v2_submission.py` | `b8211336c23805a6a9733a163dd6feaa1262018f30bbdbc63e44fe3cf0eba99f` |
| `src/aiijc_puzzle/union_v2_submission_validation.py` | `9958f5cd929d5a9f31bfaccd0252e3ca098869fa3ba9626e007af61588f28e14` |
| `src/aiijc_puzzle/raw_twin_union_production.py` | `965c63fdb42457a4ceb297c3f6cfd9cf048596849640417123851e0cfa73287a` |
| `scripts/run_union_v2_submission.py` | `f59224a717928ca2a7f6efb9ad8cd5738d3fdd3a279ffef9e64c4056a14e626d` |
| `scripts/validate_union_v2_submission.py` | `0217fa81e2813b75f9bc141a747e011252c0adaf3b47f94014321fc9e54afaca` |

В production config жёстко задан полный runtime source allowlist; его нельзя
подменить sidecar-ом или сократить. Attestation записывает SHA каждого файла из
этого списка и версии Python, PyTorch, NumPy, OpenCV и Pillow. MPS device check
нормализует совместимые `mps`/`mps:0`, но сохраняет fail-closed mismatch для
другого backend/index.

Tests: `tests/test_union_v2_submission.py` SHA
`03f520fc57689b7f95d77f4a09063b6a741b34eb5ea0f3848677b8e862b4dcab`;
`tests/test_raw_twin_union_production.py` SHA
`46076524a61888baefa230aaea9db89e9cf57d9e2d83b7f5da9b1ee6f24eacd0`.
Focused run: Ruff PASS, `10 passed`. Он проверяет SHA/Schema, независимую raw
assembly и tail equivalence, strict-layout rejection, material PNG tamper,
foreign files и MPS device normalization.

## Dry-run и production

Dry-run по умолчанию проверяет archive/roster metadata, все frozen hashes и
загружает модели, но не открывает test PNG и ничего не пишет:

```bash
.venv/bin/python scripts/run_union_v2_submission.py \
  --source-dir data/raw/test \
  --source-archive data/raw/archives/test.zip \
  --output-dir outputs/union-v2-submission/predictions \
  --output-zip outputs/union-v2-submission/submission-union-v2.zip \
  --attestation outputs/union-v2-submission/attestation.json \
  --validation-state outputs/union-v2-submission/validation-progress.json \
  --device mps --allow-nondeterministic-mps
```

Dry-run status: `DRY_RUN_METADATA_ONLY_NO_TEST_PIXELS_OPENED`, exact 700 roster,
`writes_performed=false`; frozen pipeline digest
`cb21bd0319d450e7cdcaa4cfc83d960ebb65cf65688144e8d5de1704a74e7e20`.

Полный run/resume — та же команда с `--run`. При совпадающих PNG+record resume
не запускает layout model: он строго читает layout, заново собирает raw canvas,
повторяет frozen tail и сверяет material hashes/PNG. Односторонний PNG/record,
неизвестный файл или другой run identity приводят к fail-closed остановке.

После 700 boards packager создаёт детерминированный flat ZIP с sorted official
names и отдельную attestation. До публикации независимый validator заново
запускает Union-v2 layout, отдельно реализует raw assembly/tile multiset audit,
отдельно повторяет оба harmonizer-а и colored NLM и сравнивает decoded RGB,
directory PNG, ZIP bytes, board records и attestation. Validation receipts
позволяют безопасно продолжить после interruption; tail всё равно повторяется
для каждого board на каждом проходе. Отдельный повторный entrypoint:

```bash
.venv/bin/python scripts/validate_union_v2_submission.py \
  --source-dir data/raw/test \
  --source-archive data/raw/archives/test.zip \
  --output-dir outputs/union-v2-submission/predictions \
  --submission-zip outputs/union-v2-submission/submission-union-v2.zip \
  --attestation outputs/union-v2-submission/attestation.json \
  --validation-state outputs/union-v2-submission/validation-progress.json \
  --device mps --allow-nondeterministic-mps
```

## Expected publication paths

- `outputs/union-v2-submission/predictions/` — 700 RGB PNG + records/run state;
- `outputs/union-v2-submission/submission-union-v2.zip` — flat official roster;
- `outputs/union-v2-submission/attestation.json` — full immutable evidence;
- `outputs/union-v2-submission/validation-progress.json` — resumable independent
  validation receipts.

## Фактический production outcome

MPS inference завершил все `700/700` boards и записал строгие layouts, raw
audits и h20 PNG. Предварительно задуманный validator с повторным model replay
остановился на втором board: MPS `index_add_` в Union grouped reduction дал
микроскопически другой residual и другой, хотя также строгий, decoder layout.
Это численная недетерминированность PyTorch MPS, а не нарушение tile bijection.
Повторный full-700 inference после этого не выполнялся.

Финальный flat ZIP был собран один раз из уже записанных 700 PNG после проверки
точного official roster, всех записанных strict `0..575` layouts/raw-audits и
RGB480 geometry:

- path: `outputs/union-v2-submission/submission-union-v2.zip`;
- SHA-256: `8866e060cae32d56277470f565779cd68826d9a766513e3e81eed2165f6d9725`;
- size: `186744053` bytes;
- user-reported official score: **`0.24201676406343967`**.

Предыдущий user-reported `fixed-B standard + buddies96` score равен
`0.2762279116935955`; поэтому этот submitted `Union-v2+h20` arm имеет статус
**combined-pipeline reject / do not mark best**. Delta нельзя приписывать только
solver-у: одновременно был заменён сильный fixed-B restoration на h20. Текущий
attestation/full-replay contract следует считать невыполненным; будущая версия
должна либо валидировать сохранённые MPS layouts без ложного заявления о
bit-exact model replay, либо полностью генерироваться и повторяться на CPU.

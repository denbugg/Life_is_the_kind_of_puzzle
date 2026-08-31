# Joint reciprocal tri-emitter: roster audit

Дата аудита: 2026-08-31. Это только target-blind аудит метаданных и
рекомендация для discovery-протокола. Изображения, labels, содержимое NPZ,
модели и solver не открывались и не запускались. Финальный training config этим
документом не подписывается: сначала должен быть отдельно проверен capacity
gate joint-verifier.

## Вывод

Практичный source-disjoint split для следующего дешёвого real discovery:

- **FIT32 × draw2** — прежние 32 источника tri-emitter, 64 уже созданных
  immutable fit-cache cases. Ordered digest:
  `6c0d605b60d9f437a9676dbee653185e62ffb44c42e012e05228b8f3901a0d1c`.
- **DEV32 × draw1** — уже открытый `Socket v2 eval32`, но на одном новом
  deterministic exact-synthetic draw (`case_seed=20260908`, `draw_index=0`).
  Ordered digest:
  `93112f89096f8e9555172f10f6934fd8dd5abf48a8029b86a8803d507e79e87e`.
  Все 32 детерминированных `synthetic-*` case ID были вычислены только из
  filename/draw/seed; ни один не найден в существующих text metadata. Это не
  возвращает source freshness, но исключает точный case replay по доступному
  ledger.

FIT и DEV имеют нулевое source overlap. DEV также не пересекается с открытым
tri/adapter/DINO local16 и с защищённым adapter3200 terminal16. Однако DEV32 уже
использовался и target-assisted оценивался в Socket v2, поэтому результат на нём
может быть только **development signal**. Его нельзя называть fresh confirmation
или использовать для promotion без новой source-disjoint CONFIRM.

## Почему не прежний terminal16

Прежний terminal16 с digest
`2a39d853772aa2c6d23d8b7dbc59f726e2f3a3ecfe098e96ad065c1bbd6d65a6`
не выделяется joint-verifier:

1. `configs/fullres_retrieval_adapter_scale3200_preregistered_v1.json` уже
   подписал этот exact panel за adapter3200. Если его local gate пройдёт, только
   этот runner имеет право открыть terminal16.
2. Adapter1600 и прежний tri-emitter его не открывали: оба отчёта содержат
   `terminal16.status = skipped_by_local_gate`; aborted adapter3200-run также
   фиксирует `terminal_metrics_computed=false`.
3. При этом глобально source filenames уже нельзя считать невиданными: пять из
   них были scored в component-relation `confirm24`, остальные одиннадцать — в
   `decoder40`. То есть adapter-specific exact terminal draw ещё защищён, но
   source-level universal freshness отсутствует.

Решение fail-closed: terminal16 не резервировать, не генерировать и не читать в
joint discovery.

## Точные roster-ы

FIT32 в сохранённом порядке:

```text
img_000934.png img_001317.png img_001988.png img_006170.png
img_002442.png img_000789.png img_003117.png img_002434.png
img_004562.png img_000867.png img_002813.png img_006141.png
img_006402.png img_004204.png img_006205.png img_002727.png
img_004951.png img_001111.png img_000043.png img_003181.png
img_000413.png img_001093.png img_005873.png img_004418.png
img_002898.png img_004595.png img_001345.png img_001996.png
img_000694.png img_005357.png img_005123.png img_005951.png
```

DEV32 в сохранённом порядке:

```text
img_004604.png img_006179.png img_004153.png img_001598.png
img_006463.png img_000896.png img_000968.png img_001364.png
img_004060.png img_003856.png img_005219.png img_004516.png
img_006404.png img_004233.png img_005956.png img_001000.png
img_006967.png img_004934.png img_005632.png img_005309.png
img_004280.png img_001953.png img_006794.png img_001490.png
img_006879.png img_006563.png img_000928.png img_005550.png
img_003769.png img_004729.png img_006705.png img_003595.png
```

Оба digest — SHA-256 от имён в указанном порядке, соединённых `\n`, без
конечного перевода строки.

## Lineage overlap

| Lineage | Count / digest | FIT overlap | DEV overlap | Состояние |
|---|---:|---:|---:|---|
| Socket v2 train | 1024 / `d9071055…` | 0 | 0 | checkpoint training |
| Socket v2 eval | 32 / `93112f89…` | 0 | **32** | opened and scored; это выбранный DEV |
| adapter fit | 32 / `6c0d605b…` | **32** | 0 | выбранный FIT |
| adapter local | 16 / `25ea956a…` | 0 | 0 | opened |
| adapter3200 terminal | 16 / `2a39d853…` | 0 | 0 | protected reservation |
| DINO candidate-screen local | 16 / `25ea956a…` | 0 | 0 | opened; тот же local16 |
| tri-emitter fit | 32 / `6c0d605b…` | **32** | 0 | выбранный FIT |
| tri-emitter local | 16 / `25ea956a…` | 0 | 0 | opened; replay запрещён |
| tri-emitter terminal declaration | 16 / `2a39d853…` | 0 | 0 | не открывать; принадлежит adapter3200 |

Полные hashes, списки и machine-readable overlap находятся в
`outputs/joint-reciprocal-tri-emitter-verifier/roster-audit-v1/report.json`.

## Artifact audit

- Все 64 FIT cache files существуют; каждый byte SHA-256 совпал со значением в
  прежнем tri report. Digest нормализованного inventory:
  `d39e07d90a18cf923fc14e11366ed53553dd972a9985cafa1a0f7ad3b3c71f65`.
- Frozen emitter inputs существуют и совпадают с заявленными hashes: Socket v2
  `0e9df49a…`, adapter1600 `51beee8d…`, official DINOv2-S/14 `b938bf1b…`.
- Старый tri checkpoint `e7afa13a…` сохранён как evidence, но его learned head
  не должен быть warm-start для новой objective. Повторно используются только
  frozen emitter checkpoints и immutable candidate/content cache.
- Содержимое NPZ не проверялось. До подписания runner обязан fail closed, если
  cache schema не содержит всех candidate identities/content, нужных для
  column grouping и `NONE`. В таком случае допустима только регенерация тех же
  64 FIT cases с теми же источниками/draws/emitters — не смена roster-а.

## Один рекомендуемый discovery-протокол

После reviewed capacity pass подписать один endpoint, не sweep:

- model/training seed: `20260913`;
- FIT endpoint: ровно 3 эпохи / 1,752 optimizer updates над теми же 64 cache
  cases; from scratch, без checkpoint selection;
- DEV: перечисленный Socket eval32, `case_seed=20260908`, `draw_index=0`, один
  target-free candidate freeze до exact scoring;
- candidate identities: неизменный raw + adapter1600 + DINO top32 union, raw
  top32 всегда сохранён;
- objective: `L_row + L_col + 0.25 L_conf + 1e-3 mean(delta^2)`, learned row и
  column `NONE`, differentiable minimum `tau=0.25`;
- deployment head: reciprocal row/column top-1, затем ровно top 5% confidence
  на axis/board (`ceil(.05 × 576) = 29`), без threshold/coverage sweep.

Sensitive discovery gate из `NEXT-solver-roadmap.md`, все условия вместе:

- pooled R@1 `>= raw + 0.5 pp`;
- pooled R@5 `>= raw`;
- two-sided reciprocal precision на фиксированных 5% `>= raw + 2 pp`;
- ни одна axis не имеет отрицательного delta;
- union identities/coverage не уменьшены.

Только supply gain сохраняет модель как emitter, но decoder не открывает. Даже
полный pass на этом DEV разрешает лишь следующий source-disjoint CONFIRM; local16,
terminal16, competition test, submission и output pixels остаются закрыты.

## Audit inputs

- `docs/NEXT-solver-roadmap.md` — `bd0484bd…`;
- tri report — `29686696…`;
- Socket v2 report — `ff461744…`;
- adapter1600 report — `47ce8b17…`;
- DINO local report — `6e5d0481…`;
- adapter3200 preregistration — `792dc330…`;
- component confirm24 / decoder40 reports — `7c0874c2…` / `bc3a51cc…`.

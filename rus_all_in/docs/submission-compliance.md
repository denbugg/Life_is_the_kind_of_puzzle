# Frozen-method submission integrity gate

Статус: **обязательный fail-closed gate**. Уточнение организаторов требует не
просто высокий SSIM, а восстановление правильного расположения всех 576
фрагментов в сетке 24×24 и их качества. Фрагменты нельзя подменять, дублировать,
пропускать или заменять синтетическим canvas. Поэтому target-free и высокий
offline SSIM сами по себе не означают допустимость submission.

Machine-readable evidence contract находится в
[`configs/submission-compliance.schema.json`](../configs/submission-compliance.schema.json).
Production runner формирует attestation, проходящий эту JSON Schema, а
независимый validator пересчитывает все утверждения из immutable input snapshot
и готового ZIP. Успешный статус намеренно называется
`METHOD_COMPLIANT_LAYOUT_ACCURACY_UNPROVEN`: он доказывает provenance,
биекцию/геометрию и точное исполнение frozen pipeline, но **не** доказывает
скрытую правильную перестановку, качество реконструкции или ручную приёмку.

## Обязательные инварианты

Для каждой из 700 test-досок runner должен:

1. читать только соответствующий официальный dirty input; roster и SHA-256
   должны быть связаны с исходным архивом организаторов, symlink запрещён;
2. разделить input на ровно 576 исходных RGB-фрагментов 20×20;
3. записать `tile_at_position` как перестановку всех целых чисел 0…575: каждый
   input tile использован ровно один раз в одной из 576 позиций;
4. собрать `raw_assembly` только как эту перестановку исходных tiles и записать
   его SHA-256 до pixel restoration;
5. применять tail только после layout и только как image-to-image восстановление
   RGB-качества; tail не может менять геометрию, использовать чужие pixels,
   templates, clean references или подменять fragment identity;
6. не использовать train/test targets, historical 18 clean references, source
   retrieval, filename/board overrides, content substitutions или pixels других
   досок;
7. сформировать ZIP из ровно 700 root-level RGB PNG 480×480 с именами,
   совпадающими с официальным test snapshot.

Отсутствие любого обязательного evidence, невалидная перестановка, несовпадение
recomputed hash или неизвестный источник pixels автоматически дают
`NONCOMPLIANT_DO_NOT_SUBMIT`.

## Текущий статус pipelines

| Pipeline / artifact | Manual compliance | Измеренный результат | Решение |
|---|---|---:|---|
| Constant median/mean/gray canvas | **NONCOMPLIANT** | holdout-700 metric diagnostic 0.388832 | **DO NOT SUBMIT** |
| SSIM-parametric constant RGB | **NONCOMPLIANT** | calibration-48 best 0.409374; gate fail | diagnostic only |
| Population atlas / low-frequency-only canvas | **NONCOMPLIANT** | calibration-48 best nonbaseline 0.403517; gate fail | diagnostic only |
| M420 content substitution | **NONCOMPLIANT as output** | target-assisted diagnostic only | never package |
| Bilateral E14 scores → ORBIT buddies96 → raw tile assembly | structurally compliant in current code | calibration-4 0.108139 | insufficient evidence |
| То же → colored NLM `h=9` restoration tail | structurally compliant historical smoke | calibration-4 0.200029 | superseded as production tail; historical value unchanged |
| No-atlas buddies96 → RGB offsets → bounded luma → NLM `h20 x1` | structurally audited; hidden layout unproven | frozen calibration-48 0.257664; one-time holdout-96 0.253128 | 700-image artifact built; two validator PASS; manual risk critical |

Две historical smoke-строки воспроизведены target-free на четырёх
manifest-calibration досках: solver возвращает точную перестановку 576 tiles, а
prediction hashes совпали с сохранённым report. Frozen fallback позднее прошёл
более сильный evaluator и одноразовый holdout, но это всё ещё не доказывает
скрытую правильную перестановку. Исторический test packager в
`run_legacy_upgrade.py` жёстко привязан к запрещённому constant canvas и не
пригоден для отправки. Новый strict packager завершил production и дважды
проверил артефакт. В workspace есть технически целостный ZIP, однако его
ограниченный статус не доказывает правильный скрытый layout или ручную
допустимость.

## Реализованный production gate

Для нового Socket layout-кандидата отдельно подготовлен
[production-safe resumable runner](socket-sorter-production.md). Он пока не
запускался на 700 test и не заменяет frozen fallback: сначала нужно выбрать
checkpoint и legal pixel tail. Scaffold уже запрещает targets/manifests,
source lookup, centre/background shortcuts, warp/resize и constant canvas;
decoder144 + opt-in cyclic-border5 собирают строгую перестановку исходных tiles,
а default tail — identity.

`scripts/run_compliant_submission.py` — fail-closed packager для одного заранее
замороженного варианта. В production CLI больше нет atlas/strength/pass-count
развилок: единственный разрешённый layout — true no-atlas bilateral buddies96,
а единственный tail — исторические additive RGB seam offsets, затем bounded
luminance gains, затем ровно один proper-RGB OpenCV NLM проход с
`h=20, hColor=20, template=7, search=21`. Любая попытка передать прежние
`--atlas-weight` или `--nlm-passes` завершается ошибкой argument parser-а.

Pipeline evidence зафиксирован в
`configs/frozen_submission_h20x1_fallback_v1.json` (SHA-256
`7609987c9d9b817c48cc893d58f2a77fc37b8c1a2911574bed0013e01e38a042`). Это
не ретюнинг после aspirational gate: численная цепочка не изменилась, новый
config только фиксирует разрешённый пользователем fallback threshold `>=0.25`.

```bash
uv run python scripts/run_compliant_submission.py --run
```

Runner принимает только официальный `test.zip` с SHA-256
`62d365c45fe85c3da06e96f83390e7bb056935036a9b5dee7a99d32f11483c89` и roster
digest `312e8c46b2ccfa27e525d607d046d0e3676688f8c71533b8498c377d71805376`.
Он побайтово связывает извлечённые 700 inputs с этим архивом, запрещает
symlink и пересечение input/output путей, пишет PNG/ZIP/JSON через staging и
публикует только после полного self-validation. Для отдельной повторной проверки
есть независимый entrypoint:

```bash
uv run python scripts/validate_compliant_submission.py
```

Public validator не принимает override схемы или отключение recomputation. Он
проверяет закреплённую Draft 2020-12 schema (SHA-256
`9e1b046a7484b20c6883a8b0322500e8230cb66a8b4ca8edd7370af05584a8ac`), полный policy object,
content-addressed runtime manifest (production/validator source files,
конфигурации, `uv.lock` и версии численных библиотек), roster и SHA-256 исходного
ZIP, все 700 input/ZIP PNG, каждую перестановку 0…575, raw assembly и его hash.
Затем он **заново запускает no-atlas bilateral buddies96 solver из каждого
соответствующего input** и требует точного совпадения `tile_at_position`, после чего
проверяет SHA-256 обоих target-blind harmonizer configs и общего frozen-pipeline
config. Для каждой доски он независимо повторяет RGB offsets, bounded luma и
ровно один h20/hColor20 NLM проход, сверяя hash промежуточного harmonized canvas
и decoded RGB output. ZIP обязан содержать ровно официальные root-level имена,
regular mode `0644`, deflated entries и RGB PNG 480×480. Такой PASS подтверждает
только соответствие замороженному методу и целостность артефакта; правильность
скрытого layout остаётся `correct_hidden_layout_proven=false`.

Production создал 700 predictions в
`outputs/compliant-submission/predictions/` и
`outputs/compliant-submission/submission.zip`, SHA-256
`7c36307af0ea821c8a5fbf3139323ece332744dcf59a413198dd96d5a2f619bf`.
Attestation SHA-256:
`5323d05b71b56645a7ad2acab5276187035c4e1e9de07c3fb34821b60c688c8f`;
runtime-manifest digest:
`15c88d3def7bccc9c0fd0fe082ae848e9e768af89fadf363b8bb6ae4f31d3d6f`.
Встроенная проверка production runner-а и отдельный повторный запуск validator-а
оба дали `METHOD_COMPLIANT_LAYOUT_ACCURACY_UNPROVEN`; сохранённый независимый
отчёт — `outputs/compliant-submission/independent-validation.json`.

Target-free visual audit
`outputs/compliant-submission/visual-audit/REPORT.md` просмотрел 24
детерминированно выбранных результата и не нашёл ни одной уверенно целостной
сцены. Median detail retention равен `0.321`, median 20×20 grid ratio — `2.94`.
Это **critical manual-review risk**: технический PASS нельзя повышать до
утверждения о правильной раскладке или manual compliance.

## Quarantine

Сохранённые constant/parametric/low-frequency результаты полезны только как
диагностика рассогласования метрики с предметной задачей. Их machine-readable
quarantine overlays лежат рядом с артефактами:

- `outputs/legacy-upgrade/QUARANTINE.json`;
- `outputs/ssim-parametric/QUARANTINE.json`;
- `outputs/low-frequency-prior/QUARANTINE.json`.

Quarantine не удаляет результаты и не меняет их численные выводы. Она запрещает
называть их champion-ами решения задачи и использовать для submission.

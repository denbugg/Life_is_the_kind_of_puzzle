# Финальное решение: frozen fallback и передача в production

Дата фиксации: 2026-08-30.

Текущий итог — замороженный target-blind fallback, который прошёл внутренний
одноразовый holdout с mean RGB SSIM `0.253128`. Это **не официальный leaderboard
score** и не доказательство правильной раскладки. Production завершён: построены
700 test predictions и submission ZIP, а встроенный и повторный независимый
validator оба дали `METHOD_COMPLIANT_LAYOUT_ACCURACY_UNPROVEN`. Это подтверждает
целостность frozen метода, но не manual compliance.

## Замороженный pipeline

Единственный разрешённый production-вариант задан в
[`configs/frozen_submission_h20x1_fallback_v1.json`](../configs/frozen_submission_h20x1_fallback_v1.json),
SHA-256
`7609987c9d9b817c48cc893d58f2a77fc37b8c1a2911574bed0013e01e38a042`:

1. Соответствующий dirty input делится на 576 upright RGB-фрагментов 20×20.
2. Bilateral E14 scores и `best_buddies(max_edges=96)` без atlas/unary prior
   строят строгую перестановку `0..575`.
3. Raw canvas собирается только перестановкой исходных fragment pixels. Каждый
   fragment используется ровно один раз; rotation, warp, resize, substitution и
   pixels других досок отсутствуют.
4. Уже после layout применяются исторические target-blind additive RGB seam
   offsets, затем bounded luminance gains с пределом ±4%.
5. Последний шаг — ровно один OpenCV colored NLM pass: `h=20`, `hColor=20`,
   `templateWindowSize=7`, `searchWindowSize=21`.

Хэши двух неизменяемых harmonizer-конфигураций:

- `configs/postassembly_rgb_offset_v1.json` —
  `4adfd9b614e8556b7de5c1f527d759d15d29c0f74e20aa26ff87900dd773ec9a`;
- `configs/postassembly_luminance_gain_v1.json` —
  `7488cad2ae7cc75792d6ff0ff2ea0a38fa778979083ffd5c161c857b68fd550f`.

В metadata RGB-конфига поле `input` исторически говорит
`strict atlas_w0.03+buddies96 ordered dirty uint8 tiles`. Это описание входа в
эксперименте-первоисточнике, а не зависимость алгоритма: RGB seam solver работает
с любой уже упорядоченной сеткой 24×24. Production применяет тот же неизменённый
конфиг и hash к **no-atlas** layout. Менять metadata задним числом нельзя, потому
что это разрушит frozen hash.

## Зафиксированные calibration и single-use holdout

Общий manifest:

- `data/interim/validation_manifest.json`, file SHA-256
  `4781e370e092ad272c63e6d5165b25951aaf93fae5fde74c75d534a9e8efc9da`;
- protocol digest
  `2a9e3b74f7defa8c00846a05eb598fd263fd16c2787c70e77d3b7a4b585bfbf4`.

Aspirational config
`configs/frozen_final_h20x1_v1.json` имеет SHA-256
`83443d7500ae98b6e4f33dd1ef26c2300e2a5228117d6d9cc7d6536c70c2e5e8`:
его calibration result `0.257664` не прошёл заранее заданный gate `0.28`, поэтому
его holdout не открывался. Frozen fallback не меняет pipeline после этого
неуспеха; он фиксирует отдельно разрешённый порог `>=0.25` после серии
отрицательных экспериментов.

| Split | Выборка | Raw | NLM h20×1 | RGB+luma→NLM h20×1 | Gain final−control | Paired 95% CI | W/T/L |
|---|---|---:|---:|---:|---:|---:|---:|
| calibration | offset 120, count 48 | 0.1117105411 | 0.2473083412 | **0.2576641709** | +0.0103558297 | `[+0.0090735261, +0.0117008262]` | 48/0/0 |
| holdout | offset 96, count 96 | 0.1121401300 | 0.2433203925 | **0.2531282915** | +0.0098078990 | `[+0.0090095315, +0.0106259073]` | 96/0/0 |

Calibration filename digest:
`5b1a8dcd358c87191d1c0ced0253ec66f45566568e7126c76259ff13f9289bbf`.
Артефакты frozen-evaluator-а:

- `outputs/postassembly-harmonizer/no-atlas-calibration-offset120-count48-h20x1-frozen.json` —
  `e201b3cc8b5a51b047349a991712576502272657e4fd7ca216c5ca566c42bb74`;
- `calibration-report.json` —
  `36b405e0c616337319a768b1266b740e6f20d2ac0780d61390ce09bfa896122f`;
- `calibration-prediction-commitment.json` —
  `44176b7de72307ca0864394ab1cdc00e18db95e648603f9eb936948aba1d37c5`.

Holdout filename digest:
`a8d840c30a15419852bbd748b06d3985b390d069cbed8ed39964ac6f4cc8c175`.
Он был открыт **ровно один раз** после target-blind prediction commitment:

- `holdout-report.json` —
  `715a4ed2d5c2c1b2ef7f12254219f5b0c6e153a6ab64dc0c852d38d9cbcaae5d`;
- `holdout-prediction-commitment.json` —
  `c2077a1e5677f49d67a8ac55d249c6d7513e1a1553d5d6ceecf264b7867c0349`;
- read-only `HOLDOUT_OPENED.receipt.json` —
  `8e4ca3a74139d20b5e8fbfba7bba25013f391e3f3bd55d3563cc993bd432e997`.

Все файлы находятся в
`outputs/frozen-final-evaluations/7609987c9d9b817c48cc893d58f2a77fc37b8c1a2911574bed0013e01e38a042/`.
Команду с `--mode holdout --allow-holdout` повторять нельзя. Любое дальнейшее
исследование обязано получить новый versioned config и новую calibration;
использовать этот holdout для ретюнинга запрещено.

## Что означают и чего не означают числа

Исторический `0.2374852573` — официальный score другого S1 pipeline на скрытом
competition test. Новый `0.2531282915` получен на внутреннем manifest-holdout из
train-пар организаторов. Распределение, protocol и pipeline различаются, поэтому
разность `+0.015643` не является честным paired improvement и не гарантирует
аналогичный leaderboard score. У текущего pipeline официального результата нет,
хотя 700-image ZIP уже построен: в этой документации нет факта его отправки или
оценки competition platform.

Статус успешной attestation намеренно называется
`METHOD_COMPLIANT_LAYOUT_ACCURACY_UNPROVEN`. Его scope узок: validator доказывает
соответствие каждого output своему dirty input, строгую биекцию всех 576 tiles,
сохранение raw geometry до restoration, отсутствие запрещённых references и
битовое исполнение frozen tail. Он **не** доказывает, что `tile_at_position`
совпадает со скрытой истинной перестановкой, что изображение восстановлено с
достаточным качеством или что эксперты примут решение вручную.

Точные machine-readable поля:
`scope = provenance_bijection_geometry_and_tail_only` и
`correct_hidden_layout_proven = false`. Их нельзя заменять более сильной
формулировкой даже после технического PASS.

## Критический риск ручной проверки

Это главное нерешённое ограничение. На calibration-диагностиках тот же
no-atlas `buddies96` layout давал лишь примерно `0.1–0.2%` direct placement и
`3.5–3.8%` adjacency по target-assisted приближённой разметке. Manual sheets
показывают преимущественно мозаичные canvases без стабильно восстановленных лиц,
людей, текста и целых объектов. Рост `0.112140 -> 0.253128` на holdout в основном
даёт restoration tail; он не превращает слабую глобальную раскладку в доказанно
правильный пазл.

Target-free production visual audit
[`outputs/compliant-submission/visual-audit/REPORT.md`](../outputs/compliant-submission/visual-audit/REPORT.md)
подтвердил этот риск на 24 детерминированно выбранных outputs: **24/24 не имеют
уверенно читаемой целостной сцены**. Медианное сохранение среднего внутриточного
градиента — `0.321`, а медианное отношение яркостного перепада на сетке 20×20 к
среднему внутреннему перепаду — `2.94`. Иными словами, детали сильно сглажены, а
стыки fragments остаются заметными. Это critical manual-review limitation, а не
доказательство ground-truth ошибки: скрытые targets в аудите не использовались.

Следовательно, pipeline честно сохраняет и использует все fragments, но может
не удовлетворить содержательной части письма организаторов о **правильном
расположении** всех 576 фрагментов. Ни PASS validator-а, ни внутренний SSIM нельзя
описывать как гарантию manual-compliance или выхода в финал.

## Production и независимая проверка

Команды ниже уже были выполнены из корня проекта. Production runner намеренно не
перезаписывает существующие артефакты, поэтому повторный запуск с default paths
теперь завершится fail-closed; команда сохранена как точный provenance запуска.

Production-команда:

```bash
uv run python scripts/run_compliant_submission.py --run
```

Она принимает только официальный `data/raw/archives/test.zip` с SHA-256
`62d365c45fe85c3da06e96f83390e7bb056935036a9b5dee7a99d32f11483c89` и roster
digest `312e8c46b2ccfa27e525d607d046d0e3676688f8c71533b8498c377d71805376`.
Публикация происходит только после встроенной полной проверки.

После встроенной проверки была выполнена повторная read-only проверка отдельным
entrypoint:

```bash
uv run python scripts/validate_compliant_submission.py
```

Опубликованные итоговые пути:

- `outputs/compliant-submission/predictions/` — 700 RGB PNG 480×480;
- `outputs/compliant-submission/submission.zip` — ровно те же 700 PNG в корне,
  SHA-256
  `7c36307af0ea821c8a5fbf3139323ece332744dcf59a413198dd96d5a2f619bf`;
- `outputs/compliant-submission/compliance-attestation.json` — roster,
  per-board permutations/hashes, method и runtime manifest; SHA-256
  `5323d05b71b56645a7ad2acab5276187035c4e1e9de07c3fb34821b60c688c8f`;
- `outputs/compliant-submission/independent-validation.json` — сохранённый
  результат второй проверки; SHA-256
  `ee1a7c7566f643ae0c0be6325a96c0db0b5be08e032ce8a083cb4ad82bbf0c28`.

Обе проверки вернули `METHOD_COMPLIANT_LAYOUT_ACCURACY_UNPROVEN`, проверили все
700 boards и один submission SHA. Runtime-manifest digest:
`15c88d3def7bccc9c0fd0fe082ae848e9e768af89fadf363b8bb6ae4f31d3d6f`.

Технический PASS не является рекомендацией отметить решение «Лучшим» и не
подменяет решение пользователя об отправке. При таком решении нужно учитывать
отдельный критический visual-audit verdict: 24/24 просмотренных canvases не
выглядят как целостно восстановленные сцены.

## Сохранение точного source snapshot

После завершения всех экспериментов и последней правки документации создать или
пересобрать детерминированный allowlist-only снимок. После этого не менять
sources, docs, production scripts, configs или `uv.lock` в сохранённом release:

```bash
uv run python scripts/build_source_snapshot.py --check-reproducible
uv run python scripts/build_source_snapshot.py --verify-only
```

Builder не обходит дерево рекурсивно: explicit allowlist включает production
runtime, конфиги, lockfile, evaluator/validation tests и ключевые compliance,
holdout/manual-safety документы, включая эту страницу. `data/`, прежние
`outputs/`, `.git`, path traversal и symlinks запрещены fail-closed. Архив имеет
sorted entries, fixed timestamp, mode `0644` и deterministic DEFLATE; первая
команда требует byte-identical double build, вторая только читает и сверяет
архив с текущим workspace.

Ожидаемые snapshot-артефакты рядом с submission ZIP:

- `outputs/compliant-submission/source-snapshot.zip`;
- `outputs/compliant-submission/source-snapshot-manifest.json`;
- `outputs/compliant-submission/source-snapshot.sha256`.

Attestation дополнительно содержит content-addressed production runtime manifest
с хэшами реально исполняемых файлов и версиями численных библиотек. После PASS
нужно сохранить вместе как единый release: эти три snapshot-файла,
`submission.zip`, `compliance-attestation.json`, hash официального `test.zip` и
stdout validator-а. Любая последующая правка закономерно меняет snapshot hash и
требует пересборки; старый ZIP следует валидировать в восстановленном
snapshot/environment, а не ослаблять validator.

## Идеи, которые не следует повторять без новой гипотезы

| Идея | Наблюдение | Решение |
|---|---|---|
| Constant/median, SSIM-parametric, population/low-frequency canvas | Высокий proxy SSIM достигается подменой содержания, а не перестановкой 576 input tiles | **NONCOMPLIANT / DO NOT SUBMIT** |
| Pairwise edge CNN, финальный k16/train256 scale | Adjacency `0.032684→0.062689`, gain `+0.030005`, CI `[+0.027325,+0.032646]`, 24/24; translation-aligned placement `+0.002170`. Но exact h20x1 endpoint `0.247168→0.237782`, delta `−0.009386`, CI `[−0.016116,−0.002948]`, лишь 8/24 wins | Gate 3/5 FAIL; confirmation, holdout, test и production integration не открывать |
| 8.1M ordered-seam Transformer с сильными augmentations | Exact R@1 `+0.00623`, но NLM-h10 SSIM `−0.007113` | Reject-as-tested; больший Transformer без topology signal не оправдан |
| Historical HBT side embeddings | Pure HBT дал NLM5 `+0.013490`, одновременно разрушив adjacency на `−2.416 п.п.`; fixed fusion не прошёл gate | Не масштабировать до 2048 |
| Content-RMSE listwise verifier | Scale улучшил exact, но inference-relevant all-row content recall ухудшился на `−4.789 п.п.` | Закрыть текущую multi-positive formulation |
| Tile-wise DualNAF | До сильного tail RGB+luma gain `+0.082232`, но после NLM20 проигрыш `−0.052077` | Не добавлять к frozen tail; только новая replacement-гипотеза |
| Multi-pass / high-strength NLM | Metric maximum `h120×10 = 0.412984`, но gradient ratio `0.083`, tile identity `0.017`, детали схлопываются в blobs | Все multi-pass и `h>=30` **REJECT / DO NOT SUBMIT** |
| Generic population atlas / global Hungarian | Pure assignment `−0.009819` SSIM и `−3.302 п.п.` adjacency; большие weights незначимы или портят geometry | Production остаётся no-atlas |

Авторитетный отчёт финального k16/train256 опыта:
`outputs/edge-ranker/scale-raw-k16-train256-cal24-offset228/final-tail-primary/report.json`,
SHA-256
`6fe6790f470c3e39a28d3c5c050feac1cd08623b7db3677d4f2102d7028ddad9`.
Сильный рост local adjacency не достиг заранее требуемого absolute level `0.08`
и одновременно статистически ухудшил основной full-image endpoint; это
закрывает простое масштабирование той же pairwise formulation.

Diffusion/generative restoration не включалась: без отдельного доказательства
fragment/pixel provenance она создаёт высокий риск hallucination и ручного
отклонения. Наиболее содержательный новый research route — board-specific
semantic/multi-tile signal, который улучшает глобальную геометрию и проходит
одновременно SSIM, adjacency и manual-coherence gates. Он должен жить в новой
versioned ветке оценки и не менять уже открытый fallback задним числом.

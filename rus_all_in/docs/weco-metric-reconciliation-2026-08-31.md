# Сверка Weco и GitHub — 2026-08-31

## Итог

Расхождение оказалось не потерей лучшего solver-а, а смешением трёх разных
понятий:

1. абсолютного значения метрики на конкретной панели;
2. эффекта относительно control на той же панели;
3. статуса эксперимента: development, independent confirmation или
   failed joint gate.

Ветка `rus-all-in` на момент коммита `5c50ee58cfbc4e4abf0fc0ed6bfe0a1da4a013ae`
содержит актуальные на тот snapshot solver-код, веса и frozen evidence.
Проверено `21` ключевое локальное/запушенное соответствие: `20` совпадают
побайтово, отличается только `outputs/weco-observe/runs.json`, потому что два
новых специализированных Weco-run и reconciliation exact step были созданы уже
после push. Нужен follow-up push, но исходные solver-результаты в GitHub не
утрачены.

## Почему Weco показывает 358.78125, а GitHub — 338.0625

Обе цифры верны, но получены на разных frozen-панелях и потому не сравнимы как
абсолютные уровни.

| Weco explicit step | Панель | Control pairs | Candidate pairs | Same-panel delta | Статус |
|---:|---|---:|---:|---:|---|
| `142` | уже открытая fresh32 development, участвовала в model selection | `355.62500` | `358.78125` | `+3.15625` | positive development, не formal confirmation |
| `143` | заранее подписанная новая source16×draw2 | `332.21875` | `338.06250` | `+5.84375` | formal confirmation, source CI95 `[+3.000,+9.12578]` |

Raw-pairs run `6bf52932-d716-4959-bee4-d652d7286cba` механически выбирает
максимальное абсолютное число и поэтому объявляет best `358.78125@142`. Он не
знает, что панели имеют разную сложность. GitHub намеренно приводит
`338.0625@143`, потому что это независимый formal result и именно он доказывает
переносимость frozen relation selector-а.

Чтобы больше не смешивать эти величины, создан отдельный Weco-run
`84896552-8f11-434a-9350-357b8e3d3feb` с метрикой
`pair_delta_vs_same_panel_control`:

- step `1`: relation development `+3.15625`;
- step `2`: relation formal confirmation `+5.84375`;
- step `3`: independent fullres-union + focal-tail confirmation `+7.90625`.

Здесь raw pairs остаются secondary context, а primary metric — только дельта
от идентичного same-pass control.

## Почему exact 12.875 не был best

Первоначальная read-only сверка exact-run
`c2876967-cca7-44a6-83dd-1fca125c237e` дала:

- run summary: best `8.0@94`;
- explicit step `147`: `12.875`, `status=evaluated`, но `is_buggy=true`.

Step `147` — валидное exact-наблюдение frozen Socket cyclic-origin transfer:
`5.9375→12.8750`, delta `+6.9375`. Однако одновременно pairs упали
`326.78125→323.4375`, delta `-3.34375`, что хуже заранее заданного joint pair
floor `-2`. Поэтому step был корректно закрыт как непригодный к совместному
promotion, а Weco исключил `is_buggy=true` node из run-level best summary.

Во время аудита был добавлен explicit step `159`: это только ledger replay
того же frozen evidence, без нового inference. После него exact dashboard
правильно показывает best `12.875@159`, но описание сохраняет pair-gate failure.
Это **не** делает Socket roll production-default и не заменяет promoted
pair-solver.

Следующий explicit step `160` прикрепил к неизменному exact node только
post-hoc absolute-distance replay. Run остался best `12.875@159` при
`current_step=160`. На all32 Socket roll ухудшил mean absolute Manhattan
`14.42578125→14.469292534722221` (`+0.04351128472222224`, lower is better),
хотя radius2 вырос на `0.3798 pp`. Вместе с pair delta `-3.34375` это ещё раз
подтверждает: exact-сигнал реален, но joint promotion запрещён.

После замечания о heavy-tail был добавлен failed audit step `161`. У
`12.875` median равна `1`, `24/32` boards имеют не больше одного exact tile,
а один lineage-overlap sample с `256` exact создаёт `83.93%` всей positive
mass. После удаления только крупнейшего positive paired mean delta становится
`-1.0968`. Поэтому step `159` остаётся исторической арифметической записью,
но больше не считается устойчивым exact leader; dashboard best без
distribution diagnostics нельзя интерпретировать как solver quality.

| Explicit step | Exact | `is_buggy` | Смысл |
|---:|---:|---|---|
| `94` | `8.0` | `false` | independent pair-confirmed fullres+tail run; exact neutral на своей панели |
| `147` | `12.875` | `true` | сильный exact signal, joint pair floor failed |
| `159` | `12.875` | `false` | audit replay для exact-only ledger; не новый experiment и не joint promotion |
| `160` | `12.875` | `false` | post-hoc Manhattan attachment; no new inference, Manhattan и pairs хуже control |
| `161` | `12.875` | `true` | failed robustness audit: median `1`, outlier-dominated positive mass |

Weco code payload для steps `94`, `142`, `143`, `147` и `159` побайтово
совпал с соответствующими локальными файлами по SHA-256.

## Что означают UI «113» и «80»

Канонический идентификатор — это `run_id + node_id + explicit step`, который
возвращают `weco run show/results`. Число в UI может быть display ordinal в
отфильтрованном списке и не обязано совпадать с server-allocated explicit
step: часть номеров пропущена, зарезервирована или относится к rows без
метрики.

Это видно напрямую: exact explicit step `94` находился на zero-based ordinal
`80` в полном chronological node-list. Для explicit step `147` текущий server
после последующих логов даёт ordinal `116` среди всех rows и `111` среди
scored rows; показанный ранее UI ordinal `113` лежит между ними и зависит от
UI-фильтра/момента snapshot. Поэтому `80/113` нельзя использовать как номера
для CLI, документации или воспроизведения; правильные explicit steps —
`94/147` (и reconciliation `159`).

## Manhattan теперь есть в Weco

Создан отдельный minimization-run
`effeda86-4312-45bd-8e6b-b420aa96c4d1` с primary metric
`mean_absolute_manhattan_cells`:

| Step | Layout | Mean absolute Manhattan | Delta |
|---:|---|---:|---:|
| `1` | formal same-panel control | `14.903428819444446` | — |
| `2` | frozen relation selector | `14.726888020833334` | `-0.176540798611112` (лучше) |

Тот же frozen bridge дал radius2 `+1.2424 pp`, pairs `+5.84375`, exact
`-0.15625`. Absolute Manhattan — smooth secondary metric; exact/radius0
остаётся primary placement metric, а cyclic-aligned distance — только
diagnostic, потому что она скрывает ошибку глобального origin.

## GitHub SHA-аудит

Удалённая ветка и локальный checkout GitHub-репозитория указывали на один
commit:
`5c50ee58cfbc4e4abf0fc0ed6bfe0a1da4a013ae`. Tracked diff отсутствовал;
единственный посторонний файл в checkout — незатреканный `.DS_Store`.

Ключевые совпавшие SHA-256:

| Путь | SHA-256 |
|---|---|
| `src/aiijc_puzzle/taska_relation_truth_selector.py` | `1c91b3f18d2fe08dce59217bbdf446a4638fabf4eec19b91cacd988de8cd48e2` |
| `src/aiijc_puzzle/taska_relation_selector_pipeline.py` | `1020ebc28777ba02872a82613bbb433d802e9e2b3e6fc04a5cbd2b81e49e7976` |
| `outputs/taska-relation-truth-selector/fixed-v1/model-local32-held32/frozen-relation-classifier.pkl` | `ec4eca99243cdc6be20104d789b9e5d5598b79fa0d1b7e69bc37314375ad8c6b` |
| `outputs/taska-relation-truth-selector/formal-confirmation-v1/report.json` | `d260872251077e1515251b6c7afc316af25df75045c8119112dff4f36c68ea23` |
| `src/aiijc_puzzle/taska_socket_cyclic_origin_transfer.py` | `0ef0c0314f3c805926c1ffa786692a94df59a343c5ac455545c078feb20c23f5` |
| `outputs/taska-socket-cyclic-origin-transfer/local32-v1/report.json` | `9317254528490fdd1a8b09ddac0feffd97981d04ef3076910b2657974acb0d10` |
| `src/aiijc_puzzle/tile_position_distance.py` | `5c916249c84f0d1ac3d5c0788921b2fa2356e2c6c1b513f7f07a571c2c21c171` |
| `outputs/tile-position-distance-validation/fixed-v1/report.json` | `57308b3bc944226022fcba0a52a55fa2ffd50391f0aa41b368f28c1bb9957ad6` |
| `outputs/tile-position-distance-validation/relation-selector-bridge-v1/report.json` | `2f14336e91ca889e9c8777f90ee596a7f390cfeacb7a82378a140b42a9781104` |

Полный список `21` проверенного пути и обе стороны SHA находится в
`outputs/weco-metric-audit/v1/report.json`.

После snapshot появились и пока не были в GitHub:

- обновлённый `outputs/weco-observe/runs.json`, SHA-256
  `57a3ded51f3dd45bfdb141585fd55af62b95b0047f70726f2e8a1036ac402212`;
- `outputs/weco-observe/solver-step-159-exact-ledger-reconciliation.md`,
  SHA-256
  `bbd842816f8a1e66ad259a4b61eb85c062e44f88d2a81fc772c1729f60b5988d`;
- Socket distance attachment: report
  `46613d8b05aca57df1391b712f07d858760349c031edeab0c8ef16c7df54a6cc`,
  compact journal
  `b40b490971ba40f7d71b930023d576c5204c2195702030ced08275f15c55a0d3`,
  runner
  `d9eeab61a10dfada48871828bbeb8f4c177b31ee5844893d46c7cd08277ea2bb`
  и test
  `d8e8c7d2dfb9dac37f32d485691f5fc06129f1862f34bb9bfc16627c000eb5a9`;
- этот reconciliation document и машинный audit report.

## Правило дальнейшего учёта

- Для pairs оптимизировать и ранжировать `same-panel candidate-control delta`;
  raw абсолютное число хранить только как secondary panel context.
- Для exact хранить сам exact отдельно, но всегда указывать joint-promotion
  constraints и не превращать exact-only ledger best в production best.
- Manhattan вести отдельным minimize-run; сравнивать только на одной frozen
  панели либо через same-panel delta.
- В документах ссылаться на explicit step и node ID, а не UI ordinal.
- Legal invariant неизменен: strict permutation всех `576` исходных upright
  tiles; denoised/restored pixels используются matcher-only; competition test
  и official submission этим аудитом не затрагивались.

## Выполненные read-only команды

```text
weco run status <run-id>
weco run results <run-id> --top ... --format json
weco run show <run-id> --step <explicit-step>
weco run show ... | jq -j '.code[...]' | shasum -a 256
WecoClient.list_nodes(..., include_code=False)  # ordinal audit, без записи
git fetch origin --quiet
git rev-parse HEAD
git rev-parse origin/rus-all-in
git diff --exit-code -- rus_all_in
shasum -a 256 <local-current> <github-checkout>
```

Некоторые `status` вызовы напечатали transient timeout/HTTP-500 только на
дополнительном `GET /runs`; основной `node-list` ответ в том же вызове вернул
валидный JSON и exit `0`. Результаты повторены через `results`, `show` и прямой
read-only `list_nodes`, поэтому выводы не опираются на один нестабильный запрос.

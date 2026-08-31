# Socket d64: единый legal full-cycle scoreboard

Статус: **воспроизводимый protocol-v2 matched development report; не новый
default и не доказательство решённого layout**.

Цель этого запуска — перестать смешивать local retrieval, global geometry и
pixel restoration с разных панелей. На уже открытых exact-synthetic
`source16 × draw2 = 32` случаях одним runner-ом измерена цепочка:

```text
challenge-like dirty tiles
→ frozen SocketMatcher d64 partial OT
→ decoder144
→ optional frozen cyclic-border5 absolute anchor
→ audit strict permutation of all 576 original upright tiles
→ historical target-blind RGB offsets + bounded luma + NLM h20 once
```

Clean source и точная inverse permutation не передаются predictor-у. Метрики
геометрии/SSIM вычисляются только после атомарной записи и readback label-free
freeze. Competition test не открывался. Параметры **в этом runner-е** не
выбирались: checkpoint, decoder144, cyclic weight `5.0` и pixel tail были
зафиксированы прежними экспериментами. Панель reused и не считается fresh
confirmation.

## Результат

Matched local signal frozen checkpoint-а: pooled OT R@1 `17.7649%`, R@5
`35.7337%`.

| Pipeline | Exact tiles/board | Direct | Adjacency | Raw SSIM | Final SSIM | Matcher s | Layout inference s | Tail s | Full cycle s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| decoder144 + identity | 1.1563 | 0.2007% | 13.1737% | 0.10406 | 0.10406 | 0.908 | 1.058 | 0 | 1.058 |
| decoder144 + cyclic5 + legal h20 tail | **1.4063** | **0.2441%** | 13.1029% | 0.10399 | **0.25536** | 0.912 | 1.074 | 0.192 | 1.266 |

Cyclic anchor поднял total exact `37→45` на этой reused panel, потеряв
`0.071 п.п.` adjacency. Это согласуется по направлению с отдельным fresh
cyclic-confirmation (`40→58`, source-clustered CI gain выше нуля), но текущий
matched full-cycle запуск сам по себе не является новой confirmation.

Pixel tail увеличил SSIM примерно на `+0.1514`, не меняя layout. Поэтому число
`0.25536` нельзя выдавать за качество сортировки: exact остаётся всего около
`1.4/576` тайла на доску. Оно показывает только то, что теперь текущий solver и
легальная restoration-цепочка измеряются совместно и готовы к одинаковой
оценке будущих layout-кандидатов.

## Compliance boundary

- evaluator напрямую сверяет все `1056` actual checkpoint exposure filenames с
  roster-ом панели: overlap равен нулю; exposed digest
  `a6178391ac0ede9f0c8e8bc74260094e59e5b533fd9f2ea2df6966c528e34720`;
- source-report binding проверяет checkpoint/architecture, manifest, ordered
  source digest/count/case count, target hashes, decoder144 contract и хэши
  исходных frozen NPZ/metadata; доверия одному boolean source-disjoint больше
  нет;
- freeze v2 связывает roster/seed/checkpoint/lineage/configs/code/device,
  атомарно записывается и читается обратно до первого вызова scoring;
- до tail raw canvas является точной биекцией всех 576 фрагментов исходного
  dirty input;
- rotation, resize, warp, duplication, source retrieval, atlas, centre/face и
  background shortcuts отсутствуют;
- tail видит только уже собранный RGB canvas, не принимает targets, filenames,
  tile identities или pixels других boards;
- RGB/luma configs pinned к историческим hashes и provenance; bitwise
  equivalence с frozen production RGB→luma→NLM реализацией проверена на
  non-constant fixture; NLM применяется ровно один раз с `h=hColor=20`,
  template `7`, search `21`;
- высокий SSIM не компенсирует слабую абсолютную раскладку и не снимает риск
  ручной проверки.

## Artifacts

- runner: `scripts/evaluate_socket_full_cycle.py`;
- reusable tail: `src/aiijc_puzzle/socket_pixel_tails.py`;
- production registry: `src/aiijc_puzzle/socket_sorter_production.py`;
- report:
  `outputs/socket-full-cycle/v2-d64-decoder144-cyclic5-h20-matched-source16-draw2-protocol-v2/report.json`,
  SHA-256 `932206de447f325bdb95466762cf3e6c32e94eb70cf370351dedd5500396fe62`;
- label-free freeze SHA-256
  `e019144bc7eb3c155bc414822df39101206362f15cd95b5c4e21b9bf1008d3ba`.

Старый v1 artifact сохранён как исторический, но superseded из-за неполной
самостоятельной проверки lineage и слишком сильной freeze-формулировки. Во
вторичном аудите `32/32` layouts, dirty/raw/final hashes, permutation audits,
geometry и SSIM у v2 побитово/численно совпали с v1; изменились только runtime и
protocol metadata.

Focused full-cycle/Socket production tests: 9 passed; Ruff clean.

# TASKA unique-fullres translation consensus inside confirmed fusion

Статус: **закрытый отрицательный/no-op эксперимент**. Confirmed six-arm
selective + unique-fullres fusion остаётся pair leader; production, official
best и submission не менялись.

## Что именно проверялось

Перед реализацией были сверены близкие отрицательные и положительные ветки:
fullres union voter, selective/fullres union fusion, focal-feature stacker,
fullres relation fusion, unique-edge calibrator, seven-arm portfolio и tail
budget experiments. Поэтому этот опыт не обучал ещё один edge classifier, не
добавлял седьмой arm и не менял tail budget.

На уже открытом `local32` был сделан один target-assisted diagnostic. Его
задача — разделить потери между качеством accepted-edge supply, rigid component
build и фактически выбранным layout. Все числа ниже агрегированы по 32 boards:

| Стадия unique-fullres suffix | Edges | True | Precision |
|---|---:|---:|---:|
| accepted focal/fullres verifier | `375` | `169` | `45.07%` |
| accepted rigid component builder-ом | `269` | `141` | `52.42%` |
| realised в combined pre-tail layout | `258` | `134` | `51.94%` |
| realised в final selected six-arm layouts | `139` | `88` | `63.31%` |

Средний combined union supply содержит `283.031` true relations/board. Final
confirmed fusion реализует `264.906` true union relations, ещё `61.875` true
relations вне union и одновременно `80.438` false union relations; итог —
`326.781` satisfied pairs/board. Это показывает сразу три ограничения:

- verifier всё ещё пропускает ложные edges;
- даже истинные accepted edges могут конфликтовать при rigid build/packing;
- итоговый pair score нельзя свести к edge precision: глобальная укладка
  создаёт много правильных соседств, которых не было в accepted union.

Для масштаба: selective pre-tail даёт `319.625` pairs/board, его final control
после selector/tail — `323.625`; combined pre-tail — `315.094`, а final
six-arm fusion — `326.781`. Последнее сравнение включает выбор между arms,
поэтому это не чистый эффект tail и не трактуется как causal tail gain.

## Один зафиксированный inference-visible signal

Diagnostic выделил единственный новый механистический сигнал: несколько
unique-fullres edges между одной парой уже построенных selective components
могут независимо требовать один и тот же rigid translation.

До candidate inference был зафиксирован ровно один rule:

1. Неизменённые current + accepted-selective edges строят backbone components
   в исходном recovered-focal порядке.
2. Для каждого unique-fullres edge между двумя разными components считается
   требуемый сдвиг. Пара components и знак сдвига канонизируются.
3. Edge получает consensus support, только если одинаковый сдвиг предложили
   как минимум два unique edges.
4. Только эти edges переносятся строго выше старого максимума priority через
   `nextafter(max, +inf)`; их исходный focal порядок сохраняется. Все остальные
   priorities остаются побитово теми же.
5. Новый component build заменяет прежний combined arm. Portfolio по-прежнему
   содержит ровно шесть arms; original all-1104 selector и focal-gated tail96
   не меняются. Tail получает исходные, не boosted logits.

Support, weight, threshold, arm roster и tail budget не sweep-ились. Candidate
использует только frozen parent evidence; matcher/denoiser не перезапускался.
Config: `configs/taska_fullres_translation_consensus_v1.json`, SHA-256
`d67c31d2f19c4d44acbf0a3fa039f05304ce5021bc1ed3c952732204ee99ace8`.

## Результат local32 и gate

Consensus нашёл всего `8` edges на `4` boards; `6/8` были истинными
(`75%` precision). Но на всех четырёх затронутых boards unchanged selector
выбрал другой существующий arm: дважды `nonlinear`, один раз `selective` и один
раз `logistic`. На девяти boards, где выиграл новый combined arm, consensus
edges не было, поэтому этот arm совпал с прежним combined layout.

| Метрика local32 | Confirmed fusion | Candidate | Delta |
|---|---:|---:|---:|
| satisfied pairs/board | `326.78125` | `326.78125` | `0.00000` |
| adjacency recall | `0.295997509` | `0.295997509` | `0.000000000` |
| exact tiles/board | `5.93750` | `5.93750` | `0.00000` |

Pair CI95 равен `[0, 0]`, W/T/L — `0/32/0`. Preregistered gate требовал
строго положительный local pair delta, поэтому `held32` и `fresh32` не
открывались.

Первый preliminary freeze обнаружил QA-дефект: после float64 `nextafter`
вектор приводился к float32, и минимальная promoted priority могла округлиться
обратно к старому максимуму. Этот прогон не логировался, помечен как
`outputs/taska-fullres-translation-consensus/preliminary-float32-bug-v0` и не
является валидным результатом. После сохранения priorities в float64 regression
test подтвердил строгое `> old max`; новый freeze и полный local32 rerun дали
тот же no-op результат.

## Вывод и no-repeat

Repeated rigid translation — реальный high-precision signal, но слишком
редкий (`0.25` edge/board) и невидимый нынешнему all-bond selector-у там, где
он действительно появляется. Не повторять support/priority threshold sweep на
`local32`: это будет подгонка уже открытой панели. Следующая materially new
ветка должна менять target-blind consumer/selection signal либо находить более
частый независимый consistency evidence; простого priority promotion внутри
того же six-arm consumer недостаточно.

## Артефакты, Weco и проверки

- final report:
  `outputs/taska-fullres-translation-consensus/fixed-v1/report.json`, SHA-256
  `e05383a56fe3e138b11c77f33b17581fdced129f1595561a573d450c1dab1613`;
- frozen target-free archive SHA-256
  `3ecf0f03386ea3de92abfb2575fc47fb3aa5a7f3775662a3fe9f8f2cfc6b5023`;
- module SHA-256
  `370d2f4a5e6a4284c3a0e916e3393f573d573e82c4659410dc19026742bae81b`;
- runner SHA-256
  `108dea7286442c9b900ff6b018e73e6d54126aeca1f5712901fcb0efa8eda5ea`;
- unchanged raw solver SHA-256
  `97859e1f4ff6ceadf56ebccf429f2944ae7a5726631731d9f776224a667eb486`;
- Weco Observe pair + exact step `114`, parent `102`; логировался только
  фактически выполненный corrected local32 result.

Проверки: `2` новых targeted tests плюс `7` parent fusion regressions —
`9 passed`; `ruff` для module, runner и tests — passed. Все 32 control replays
совпали с frozen confirmed fusion. Candidate и
control — строгие перестановки 576 original upright tiles. Restored pixels
использовались только matcher-ом в frozen parent evidence; competition test,
postprocess, production и submission не затрагивались.

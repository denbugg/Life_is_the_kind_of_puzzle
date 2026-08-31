# TASKA official-DRUNet unique restored-descriptor supply

Дата: 2026-08-31. Статус: **local gate fail; held/fresh не открыты**.

## Вердикт

Один fixed nominator-only replay проверил supply-positive исторический
official DRUNet40 emitter поверх frozen
`selective-target500 + unique-fullres` solver. Он не перенёсся:

| local32 | Control | DRUNet candidate | Delta, source CI95 | W/T/L |
|---|---:|---:|---:|---:|
| pairs / 1104 | **326.781** | 323.625 | **-3.156 [-6.906,+0.125]** | 1/23/8 |
| adjacency recall | **0.295998** | 0.293139 | -0.002859 [-0.006341,+0.000113] | 1/23/8 |
| exact tiles | **5.938** | 1.563 | **-4.375 [-11.000,+0.031]** | 2/26/4 |

Local gate требовал pair mean `>=0` и провален. Held32 и fresh32 по
preregistration не запускались. Production, официальный best, test и submission
не менялись. Weco Observe pair/exact: для этой ветки логировался только успешно
завершённый local step103; held/fresh продолжений не было. Глобальные steps
104-105 позднее занял отдельный unique-fullres calibrator experiment.

## Что было новым и что осталось неизменным

Это не повтор rejected direct DRUNet matcher fusion и не rejected restored
BorderRanker:

1. official KAIR colour DRUNet sigma40 независимо обрабатывал каждый upright
   dirty `20x20` tile через reflect-pad `24x24` и crop обратно;
2. restored pixels использовались только существующим normalized grayscale
   width-six border descriptor;
3. nominator оставлял только depth-one row/column mutual top-1 отдельно для
   right/down;
4. до focal scoring удалялись все edges из current, selective-target500 и
   confirmed fullres supplies;
5. оставшиеся proposals принимались только frozen dirty-visible
   `train_exact_top5` focal verifier с logit `>=0`;
6. новый standalone arm не добавлялся: accepted edges расширяли только старый
   `combined_union_focal`;
7. raw dense matrices, six-arm roster, all-1104-cost selector и focal-gated
   tail96 оставались прежними.

Все layouts — строгие перестановки `0..575` исходных upright tiles. Targets
восстанавливались только после target-free NPZ/metadata/pre-score freeze.

## Почему supply не сработал

DRUNet reciprocal nominations сами по себе имели `9.91%` precision, но почти
весь правильный сигнал уже находился в сильных frozen parents. После
дедупликации осталось в среднем `269.22` proposals/board и всего `4.56` новых
true edges (`1.69%`). Dirty focal gate принял слишком широкий out-of-distribution
набор — `118.13` edges/board, но только `0.469` true, то есть **`0.397%`
precision**. Extended union вырос `426.0→544.1` edges, а true count лишь
`283.03→283.50`.

В результате изменённый combined arm ни разу не выиграл six-arm selector на
local32. На boards, где frozen fullres-combined control был полезен, он иногда
заменился weaker unchanged arm-ом; отсюда отрицательные pair/exact deltas.

## Правило «не повторять»

Не повторять на этих panels nearby sweep sigma, descriptor width, reciprocal
depth/top-k, focal threshold, candidate cap, arm roster или tail budget. Также
не использовать rejected restored BorderRanker checkpoint и не возвращаться к
direct restored-score replacement/blend.

Материально новый follow-up допустим только с verifier-ом, обученным именно на
restored-emitter proposals и проверенным source-disjoint до decoder-а, либо с
другим independent supply, уже имеющим высокую **unique** precision после
deduplication. Текущий dirty focal verifier для DRUNet proposals закрыт.

## Артефакты

- preregistration:
  `configs/taska_drunet_unique_supply_fixed_v1.json`, SHA-256
  `6090dd715604b333f0e0df37d4673dd30099ef49946fb93dc871b67c24292aa2`;
- report:
  `outputs/taska-drunet-unique-supply/fixed-v1/report.json`, SHA-256
  `8e15139f297553529ff94242d4aa18139233c8c678dbdfebd85f70823aab9590`;
- module: `src/aiijc_puzzle/taska_drunet_unique_supply.py`, SHA-256
  `dd578720db901db8f8e3ff296b706193d96355f7573d8450b4a16dec37a8223a`;
- runner: `scripts/run_taska_drunet_unique_supply.py`, SHA-256
  `41dd1e1f88b2ea4d5a437bba1160af2040696b1ae30375675a8541176cfd48f9`;
- tests: `tests/test_taska_drunet_unique_supply.py` (`5 passed`).

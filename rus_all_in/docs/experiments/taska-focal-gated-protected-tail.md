# TASKA focal-gated protected tail

Дата фиксации: 2026-08-31.

## Решение

Сохранить вариант как **pair-candidate-confirmed**, но пока не менять
production/default. На current-disjoint fresh32 фиксированная focal-защита
подняла four-arm+tail96 с `346.0625` до **`348.34375`** правильных соседних
пар на доску: delta **`+2.28125`**, source-cluster CI95
**`[+0.875,+3.59375]`**. Exact снизился с `1.15625` до `1.03125`:
`-0.125`, CI95 `[-0.3125,+0.0625]`.

Это подтверждённый pair signal, а не exact improvement. Перед promotion нужен
один новый неизменный replay на roster-е, который не использовался при
разработке этого правила. Порог и swap budget больше не подбирать.

**Обновление:** этот replay завершён на новой preregistered fresh16 панели и
прошёл заранее заданный pair gate (`+1.875`, CI95 `[-0.1875,+3.84375]`), exact
также не снизился (`+0.21875`). См.
[отдельный confirmation report](taska-focal-gated-protected-tail-fresh16-confirmation.md).
Focal-logit-zero protection теперь подтверждён как pair-default tail primitive;
production-интеграция остаётся отдельным изменением.

## Фиксированная гипотеза

Текущий tail96 защищает все harvested edges, уже реализованные в выбранном
pre-tail layout. Новый consumer оставляет неизменными:

- four-arm roster `raw/logistic/focal-top5/nonlinear`;
- выбор arm с минимальной суммой исходных TASKA right/down costs по всем 1104
  связям;
- те же исходные cost matrices;
- только non-adjacent two-tile swaps;
- `max_swaps=96`, `minimum_gain=1e-9`;
- строгую перестановку всех 576 исходных upright 20x20 fragments.

Меняется только множество защищаемых связей: в protection передаются harvested
edges с frozen recovered focal `train_exact_top5` logit `>= 0`. Ноль —
естественная decision boundary бинарного verifier-а. В API намеренно нет
настраиваемых threshold/max-swaps параметров.

Важно: до этого bounded run на уже открытом local32 была выполнена
target-assisted диагностика порогов `-1/0/1/2/3`. При пороге `0` она показывала
примерно `258.44` реализованных kept edges с precision `88.48%` и `81.56`
dropped edges с precision `20.42%`. Поэтому local32 считается touched discovery,
а не fresh evidence. Порог после просмотра метрик не менялся и nearby sweep не
проводился.

## Gating protocol

Использованы только ранее замороженные target-free artifacts; matcher не
перезапускался. На каждом panel candidate layouts сначала записывались в NPZ,
metadata и SHA-roster, и лишь затем synthetic exact reference восстанавливался
для scoring.

1. Touched local32: pair delta `>= 0` открывает held32.
2. Held32: pair delta `>= +0.5` при отсутствии local collapse открывает fresh32.
3. Fresh32: один неизменный confirmation, без нового выбора параметров.

Local прошёл буквально на `+0.03125`, held — на `+0.53125`; следовательно,
fresh был открыт по заранее заданным воротам. Held CI пересекает ноль, поэтому
сам по себе он был только gate, а не подтверждение.

## Результаты

| Panel | Control pairs | Gated pairs | Pair delta | Pair CI95 | Control recall | Gated recall | Exact delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| local32, touched | 314.37500 | 314.40625 | +0.03125 | [-1.78125,+1.84375] | 0.284759964 | 0.284788270 | -0.09375 |
| held32 | 337.56250 | 338.09375 | +0.53125 | [-1.87500,+3.00000] | 0.305763134 | 0.306244339 | -0.06250 |
| fresh32 | 346.06250 | **348.34375** | **+2.28125** | **[+0.87500,+3.59375]** | 0.313462409 | **0.315528759** | -0.12500 |

Fresh source-cluster wins/ties/losses по pair delta: `19/7/6`. Exact CI95 на
fresh равен `[-0.3125,+0.0625]`; exact W/T/L — `3/23/6`.

Target-free mechanism действительно освободил tail: на fresh среднее число
protected tiles снизилось `384.66 -> 307.41`, а accepted swaps выросло
`87.06 -> 94.50`. Среднее число harvested edges было `378.00`, из них focal
gate сохранял `294.22`; реализованных защищённых edges в pre-tail layout —
`284.44`. Во всех 96 случаях сохранена каждая исходно реализованная gated
связь, итоговая исходная all-bond cost не выросла, каждый layout остался строгой
перестановкой.

## Ограничения интерпретации

- Local32 явно touched target-assisted threshold diagnostic и годится только
  как discovery gate.
- Held32 исторически model-selection-exposed, а его CI пересекает ноль.
- Fresh32 был current-iteration-disjoint на момент исходного roster selection,
  но last-300 range исторически model-selection-exposed; после этого run он
  также открыт. Результат сильный и paired, но не formal untouched promotion.
- Pair gain не означает официальный SSIM gain. Official best submission
  `0.2762279116935955` этим экспериментом не заменяется.
- Exact secondary слегка отрицателен; consumer нельзя выдавать за
  exact-oriented улучшение.

Следующий допустимый шаг — повторить **ровно тот же** logit-zero/tail96 consumer
на новом заранее SHA-зафиксированном roster-е. Нельзя подбирать порог,
swap-budget, focal mode или выбирать panel после просмотра результата.

## Легальность

Inference принимает только frozen dirty-derived layouts, costs, harvested
edges и focal logits. Targets/labels используются только offline после
candidate freeze для evaluation. Candidate membership не меняется. Нет
rotation, warp, replacement, constant tiles, synthesis, postprocessing или
competition-test access. Output содержит каждый исходный upright tile ровно
один раз.

## Артефакты и воспроизведение

Команда:

```bash
.venv/bin/python scripts/run_taska_focal_gated_protected_tail.py
```

Канонический report:

- `outputs/taska-focal-gated-protected-tail/logit0-v1/report.json` — SHA-256
  `80c43ed8c8dd090315b2a1e3b45572debb7de5139d70817bd6acc9d2aaab4a6e`;
- local/held/fresh target-free NPZ SHA-256:
  `180699bb5b6fdd1e20d1487c43f8c76a96b0e236802cd523d5104631948bab47`,
  `e76a72c57bfc00cd38f7113aa4be3c6f814381ae4ee563aac9831ed83ddd86c6`,
  `f4c02c1b30e118e9ce8be583ca1021612bea65950fcc53e2b558164eee1cadb0`;
- module SHA-256:
  `33d64d7202a3b65b925d12c77d10e00429968ab70cfdd4b47a52d738dc1224c1`;
- runner SHA-256:
  `5a8c0c1999b1744e369d1d7250dcb22275f622d7a56dd6fa569a26e291043963`;
- frozen raw solver остался
  `97859e1f4ff6ceadf56ebccf429f2944ae7a5726631731d9f776224a667eb486`.

Weco Observe: шаги `77/78/79` в adjacency-pair и exact runs, все ветвятся от
шага 42. Primary pair metrics дополнены `adjacency_recall`; exact сохранён как
secondary diagnostic.

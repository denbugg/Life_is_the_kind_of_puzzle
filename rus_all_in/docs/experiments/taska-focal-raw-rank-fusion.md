# TASKA raw/focal axis-rank fusion

Дата фиксации: 2026-08-31.

## Решение

Один заранее зафиксированный parameter-free fusion между raw TASKA priority и
recovered focal top-5 logits оказался отрицательным уже на opened32.  Его
единственная естественная композиция с ранее зафиксированным protected-tail
polish почти восстановила raw pair mean, но всё ещё не превзошла его.
Preregistered gate остановил эксперимент до held300; никаких alpha/threshold
или rank-formula sweep после просмотра метрик не было.

## Frozen contract

Для каждого board и отдельно для `right`/`down` candidate edges:

1. raw priority равен отрицательной исходной TASKA cost данного edge;
2. raw priorities переводятся в percentile average-midranks `[0,1]`;
3. focal top-5 logits независимо переводятся в такие же midranks;
4. итоговый larger-is-better priority — их среднее с весами `0.5/0.5`;
5. точные aggregate ties сохраняют frozen candidate order.

Midrank нормируется как `(rank-1)/(n-1)`; singleton axis group получает `0.5`.
Candidate membership, original cost matrices, rigid component builder,
placement и Hungarian fill не менялись.  Focal checkpoint/feature contract —
тот же `train_exact_top5`, который уже был отдельно проверен.  Конфиг записан
до scoring:
`configs/taska_focal_raw_rank_fusion_v1.json`, SHA-256
`7b0deaa73a0224491513726fd4805a5750b053c606ea90375b928c2aa4ee83b2`.

Одновременно до exact-reference recreation были заморожены два layout arm:

- `fusion` — только новый component-build order;
- `fusion_then_protected_tail` — предыдущий layout плюс неизменный fixed
  `max_swaps=24`, `minimum_gain=1e-9` polish.

## Opened32 result

| Arm | Pairs | Recall | Exact | Pair delta vs raw, source-cluster CI95 | Exact delta |
|---|---:|---:|---:|---:|---:|
| raw | 334.71875 | 0.303187274 | 4.46875 | reference | reference |
| focal top-5 | 335.50000 | 0.303894928 | 4.34375 | +0.78125 `[-1.15625,+2.65625]` | -0.125 |
| equal axis-rank fusion | 334.12500 | 0.302649457 | 3.90625 | **-0.59375** `[-3.96875,+2.46875]` | -0.5625 |
| fusion + protected tail | 334.62500 | 0.303102355 | 4.00000 | **-0.09375** `[-3.53203125,+3.15625]` | -0.46875 |

Fusion проиграл focal на 1.375 пары/board.  Protected tail дал +0.5 пары над
fusion, но не изменил отрицательный verdict относительно raw.  Case W/T/L
против raw: `15/0/17` для fusion и `17/1/14` после tail; source-cluster W/T/L:
`8/0/8` и `9/0/7` соответственно.

Gate требовал неотрицательную pair delta самого fusion arm на opened32.  Он не
пройден, поэтому held300 не вычислялся.  Это особенно важно: слабый tail arm
нельзя использовать как повод открыть ещё одну already-exposed панель или
подбирать blending weight.

## Legality and artifacts

Оба layout — строгие перестановки всех 576 исходных upright tile ids; pixels,
ориентация и candidate membership не меняются.  Fusion использует только
current-board costs и focal logits.  Target, exact permutation, filename и
source-grid coordinates не входят в inference.

Artifacts:

- frozen target-free NPZ:
  `outputs/taska-focal-rank-fusion/opened32-v1/frozen-target-free-eval.npz`,
  SHA `f5a774efa1cf5dd393305c851b8c07e85616bbf9bac4ebc7bf83ed9f7a833dac`;
- frozen metadata:
  `outputs/taska-focal-rank-fusion/opened32-v1/frozen-target-free-eval.json`,
  SHA `2e63cb4e6e4258d9b35923fb3fb62d9a995fe807003ea71789c49eaa88494ca1`;
- pre-score freeze:
  `outputs/taska-focal-rank-fusion/opened32-v1/pre-score-freeze.json`,
  SHA `4a04e882200b0c2e2d63b30370aff41bc172e017f51b876bd58cb38756e803e2`;
- scored report:
  `outputs/taska-focal-rank-fusion/opened32-v1/report.json`,
  SHA `cf0283c8ecc58bf60c23ad23a0664f627801d3826daf0013e1a4b73979adef27`.

Weco Observe steps 32/33 зафиксированы в exact и adjacency-pair runs.  Так как
opened gate провален, production module и новые tests намеренно не добавлялись:
это не deployable primitive.  Frozen raw solver остался byte-identical, SHA
`97859e1f4ff6ceadf56ebccf429f2944ae7a5726631731d9f776224a667eb486`.

## Closed direction

Не повторять equal percentile-rank mean, axis pooling, alpha sweep или nearby
protected-tail budget sweep на этих panels.  Focal top-5 полезен как отдельный
ordering arm и в target-free portfolio, но его signal не складывается с raw
ordering простой усреднённой rank aggregation.

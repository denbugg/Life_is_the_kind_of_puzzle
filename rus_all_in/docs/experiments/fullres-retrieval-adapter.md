# Full-resolution retrieval adapter through frozen SocketMatcher

Дата: 2026-08-31. Статус: **положительный candidate-supply diagnostic,
decoder gate не пройден**. Terminal panel, global decoder, competition test,
production и submission не открывались.

## Зачем был нужен отдельный эксперимент

Это не повтор прежних restoration/ranker веток:

- full-resolution NAF20×20 раньше обучался в основном по boundary-pixel loss и
  ухудшал direct d64 retrieval;
- E13 уже оптимизировал exact-neighbour retrieval под blur/noise/JPEG, но был
  слабым standalone four-pixel border CNN;
- DRUNet/DualNAF были generic restoration views с фиксированным score fusion;
- restored BorderRanker использовал downsampling DRUNet view и отдельный
  shortlist cross-ranker.

Здесь zero-initialised stride-one NAF adapter сохраняет разрешение `20×20` на
каждом блоке и обучается **end-to-end через frozen d64 SocketMatcher** по exact
horizontal/vertical neighbour ranking. Слабый boundary reconstruction guard
ограничивает adversarial score-space residual. Raw d64 evidence измеряется и
сохраняется параллельно; adapter pixels являются только matcher view.

## Frozen protocol

До benchmark, training и local scoring был зафиксирован
`configs/fullres_retrieval_adapter_preregistered_v1.json`, SHA-256
`74bc2f356a5750bd13f19a0911b639831f771522e258313f765027b5a6d0fc95`.

- Source-disjoint rosters: fit32; уже открытый родительской веткой local16;
  заранее зарезервированный и не открытый terminal16.
- Adapter: width 32, 8 stride-one NAF blocks, bounded residual `32/255`, без
  pooling/resampling.
- Frozen matcher: d64 SocketMatcher v2, eval mode, SHA-256
  `0e9df49a503c65aac7f1468e9acd6a074a5e658ae8b61f8954be086272c49670`.
- Один детерминированный training stream, checkpoints 100 и 400 нужны только
  для scaling slope; выбор checkpoint/hyperparameter по local запрещён.
- Независимые per-tile train corruptions: contrast `0.7–1.3`, brightness
  `±30`, Gaussian noise sigma `40–55`, reflect blur3, JPEG quality `35–50`,
  новый bag shuffle на каждом update.
- Loss: frozen Socket partial-OT bidirectional neighbour NLL + raw row/column
  CE (`raw_rank_weight=0.25`, `border_weight=0.1`) + boundary guard с весом
  `0.25`.
- Local decoder gate на step400 был зафиксирован как: directional raw∪adapter
  top32 coverage gain не меньше `+1 pp` по обеим осям и дополнительно либо
  pooled R@1 `>=+0.5 pp` при неотрицательном R@5, либо reciprocal precision
  `>=+3 pp` при coverage `>=3%`.

## Runtime benchmark

Один полный train update, включая forward frozen d64, backward и optimizer:

| Device | Cold full update | Projected 400 steps |
|---|---:|---:|
| CPU | `19.695 s` | `131.3 min` |
| MPS | `2.078 s` | `13.85 min` |

Initial loss совпал (`5.470553875`); MPS был примерно `9.5×` быстрее. PyTorch
предупредил о nondeterministic MPS implementations для scatter/index-put.
Seeds, roster, corruptions и shuffle фиксированы, но bitwise repeatability
checkpoint-а не заявляется. Фактические 400 шагов заняли `327.02 s` после
warm-up.

## Local16 retrieval result

Все candidate lists и reciprocal sets были target-free frozen до reference
scoring. В таблице проценты; `union gain` — дополнительная coverage true
neighbour в raw-top32 ∪ adapter-top32 относительно raw-top32.

| View | pooled R@1 | pooled R@5 | pooled R@32 | right union gain | down union gain | matched reciprocal precision gain |
|---|---:|---:|---:|---:|---:|---:|
| raw d64 | `19.565` | `38.887` | `69.724` | — | — | — |
| adapter step100 | `19.565` | `38.904` | `69.956` | `+2.095` | `+1.540` | `+0.119` |
| adapter step400 | `19.622` | `39.159` | `70.414` | `+3.385` | `+2.389` | `−0.072` |

Step400 против raw дал pooled R@1 `+0.057 pp` и R@5 `+0.272 pp`. Scaling
step100→400 был положительным для pooled R@1 (`+0.057 pp`), pooled R@5
(`+0.255 pp`) и особенно union supply, но не симметричным: right R@1 вырос на
`+0.249 pp`, down R@1 снизился на `−0.136 pp`.

## Gate и решение

Directional supply gate пройден: `+3.385/+2.389 pp`. Ranking gate провален:
R@1 gain всего `+0.057 pp`, ниже fixed `+0.5 pp`. Reciprocal precision gate
тоже провален (`−0.072 pp` при `47.14%` matched coverage). Поэтому общий local
decoder gate = false; terminal16 остался закрыт, decoder не запускался.

Вердикт: этот objective создаёт **materially new и растущий candidate supply**,
но step400 не является самостоятельной заменой raw matcher-а. Сохранять
raw∪adapter top32 как research primitive для materially new context-aware
consumer; не выполнять nearby threshold/fusion/checkpoint sweep на открытом
local16. Более длинное обучение допустимо только как заранее зафиксированный
новый fit/holdout protocol, а не как подбор по этим числам.

## Артефакты и проверки

- report: `outputs/fullres-retrieval-adapter/fixed-s100-s400-local16-v1/report.json`,
  SHA-256 `5fafb0307586669c7b7c9eaa4699fda1a3bd1250ca921fc48dd7e86af0bdefbb`;
- frozen target-free archive: `local16/frozen-target-free-retrieval.npz`,
  SHA-256 `b64cb6e5c649f85f3169495f3b576d7eae4d89d6891ab6316cd789b216459343`;
- step100 / step400 checkpoints:
  `1d06fb191886525d7f54fc1d1ac2f8b979b8947cbd6c65fc5c17d992fc83d0bf` /
  `00ca56f1be2c8e99bc8ef19b0d9190862d6bd5e4fb8b36fbe926087cd3945cb0`;
- module: `src/aiijc_puzzle/fullres_retrieval_adapter.py`, SHA-256
  `fc28b6c361a2e637ae23fcff1d1b0c03fc85aada8076c06035c899a491be35b6`;
- runner: `scripts/run_fullres_retrieval_adapter.py`, SHA-256
  `a12caabb759803b54d90e6df435c49eda962c30efe53ee447292dc868388b19d`;
- benchmark report SHA-256
  `567882147dddc0d10bb7feddc7e23291d38eec7df547750e6d72337f01ab3dd0`.

Targeted tests проверяют отсутствие spatial downsampling/identity при zero-init,
gradient только в adapter и fail-closed frozen Socket eval contract. `ruff` и
`3` targeted tests прошли. Weco Observe steps `119` (step100) и `123`
(step400) содержат только retrieval metrics; exact/pair layout metrics не
заявляются, поскольку decoder по preregistered gate не запускался.

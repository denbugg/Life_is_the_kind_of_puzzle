# TASKA six-arm + frozen Socket cyclic-border5 origin

Дата: 2026-08-31. Статус: **unconditional transfer rejected by the fixed
pair floor; exact signal preserved; first conservative-selector feasibility is
negative**.

## Зачем это materially new

Confirmed TASKA six-arm fusion хорошо восстанавливает относительные связи, но
его opened-local exact равен лишь `5.9375` tiles/board, тогда как oracle best
global cyclic roll ранее показывал `71.9375`. Предыдущие TASKA origin branches
уже закрыли raw all-bond cyclic cost, structural border, row-phase raw DP,
largest-component centre, coordinate sorting и learned whole-layout/frame
heads.

Здесь использован другой абсолютный сигнал: неизменённые right/down OT
assignments frozen d64 SocketMatcher, включая четыре dustbin-border heads.
Именно `select_global_cyclic_translation` с ранее независимо подтверждёнными
`border_weight=5`, `minimum_gain=1e-9`, all-576 row-major rolls и stable zero
tie был применён не к Socket decoder layout, а к frozen confirmed TASKA
six-arm final layout. Raw TASKA seam veto, semantic/centre/background prior,
target/reference и новый learned origin head отсутствуют.

Signed preregistration создана **до** Socket inference и exact scoring:

- `configs/taska_socket_cyclic_origin_transfer_v1.json`;
- SHA-256 `776414e1849da9eb1c77c23200aa826bcb9f28f1ae328b57953f97aa471da414`;
- local gate: exact delta strictly positive **и** pair delta `>=-2.0`;
- fail означает stop без weight/gain/roll/selector sweep;
- terminal/fresh можно открыть только после root review.

## Lineage caveat

Socket checkpoint содержит `1056` exposed sources. Из 32 уже открытых local
sources ровно шесть входят в его lineage:
`img_000405`, `img_001872`, `img_002346`, `img_003499`, `img_004443`,
`img_006343`. Поэтому headline приводится вместе с отдельным срезом 26
source-disjoint boards. Это development diagnostic, не confirmation.

## Fixed local32 result

Все candidate layouts были сохранены до восстановления exact references и
остались строгими перестановками 576 original upright tiles.

| Срез | Control exact / pairs | Rolled exact / pairs | Delta exact / pairs |
|---|---:|---:|---:|
| all32 | `5.9375 / 326.7813` | `12.8750 / 323.4375` | **`+6.9375 / -3.3438`** |
| Socket-lineage-disjoint26 | `3.9231 / 321.0000` | `5.5769 / 318.0385` | **`+1.6538 / -2.9615`** |

Selector changed `17/32` boards. Exact W/T/L was `4/22/6`; pair W/T/L
`0/16/16`. Формальный gate **failed** только по pair floor: `-3.3438 < -2`.
Weco step `147` записан failed в exact и pair lineage, parent `102`; step
`148`, terminal/fresh и competition test не открывались.

Сигнал при этом не является только aggregate noise:

- disjoint `img_003742`: exact `3 -> 45` (`+42`), pairs `-4`;
- disjoint `img_002819`: exact `0 -> 6`, pairs `-2`;
- lineage-overlap `img_006343`: exact `0 -> 256`, pairs `-2`;
- lineage-overlap `img_004443`: exact `74 -> 0`, pairs `-9`.

Иными словами, Socket border head иногда действительно находит абсолютный
origin сильной TASKA component structure, но unconditional rule также делает
ложные rolls и режет wrap-boundary pairs.

## Conservative selector feasibility на уже открытых labels

Ни одного нового reference или panel не открывалось. Для тех же 32 cases были
пересчитаны только inference-visible features:

- Socket total/row/column gain и runner-up margins per axis;
- original TASKA all-bond cost change как cut-loss proxy;
- toroidal roll distance и число ненулевых axes;
- agreement выбранного roll и positive objective gain среди всех шести arms.

Hard-safe positive был зафиксирован как `exact_delta>0` и per-board
`pair_delta>=-2`. Oracle ceiling:

| Срез | Safe positives | Oracle exact delta | Oracle pair delta |
|---|---:|---:|---:|
| all32 | `2` | `+8.1875` | `-0.1250` |
| disjoint26 | `1` | `+0.2308` | `-0.0769` |

All32 ceiling в основном создаёт lineage-overlap gain `+256`; source-disjoint
safe ceiling — только один `+6/-2` случай. Fixed exploratory
source-LeaveOneOut `StandardScaler -> LogisticRegression(C=1,
class_weight=balanced)`, threshold `0.5`, на 17 changed boards имел всего два
positive labels и дал:

- OOF ROC-AUC `0.2333`;
- выбрано пять false positives, precision/recall `0/0`;
- portfolio delta на all32 `-2.4063` exact, `-1.5000` pairs.

Лучший univariate orientation AUC (`0.80` для column margin, `0.75` для roll
distance) при `n_positive=2` не является достаточным evidence. Поэтому этот
compact feature family нельзя обучать/применять по local32; новый FIT/CONFIRM
не открыт.

## Decision / no-repeat

- Не применять unconditional Socket roll к confirmed fusion: pair floor
  нарушен.
- Не подбирать border weight, minimum gain, objective blend, roll distance,
  support threshold или classifier threshold на local32.
- Не выдавать all32 `+6.94` как source-disjoint confirmation: существенная доля
  gain связана с checkpoint-lineage case.
- Сохранять независимый Socket absolute-origin signal как перспективный, но
  следующий consumer требует большой заранее выбранный source-disjoint fit и
  отдельный confirm. До такого signed contract текущий selector-as-tested
  закрыт.

Если capacity будет отдельно разрешена, proposed bounded protocol:
`FIT=256 sources x 1 draw`, `CONFIRM=64 new sources x 2 draws`, исключить весь
Socket lineage и все prior exact/eval sources. Fit OOF должен иметь минимум 20
hard-safe positives, precision `>=50%`, recall `>=15%`, selected exact delta
`>+0.25` и pair delta `>=-0.5`; только затем один frozen confirmation с exact
`>0`, pairs `>=-0.5`, без model/feature/threshold sweep.

## Артефакты и воспроизведение

- module: `src/aiijc_puzzle/taska_socket_cyclic_origin_transfer.py`;
- fixed runner: `scripts/run_taska_socket_cyclic_origin_transfer.py`;
- selector diagnostic:
  `scripts/diagnose_taska_socket_cyclic_selector_feasibility.py`;
- fixed report:
  `outputs/taska-socket-cyclic-origin-transfer/local32-v1/report.json`, SHA
  `9317254528490fdd1a8b09ddac0feffd97981d04ef3076910b2657974acb0d10`;
- feasibility report:
  `outputs/taska-socket-cyclic-origin-transfer/local32-selector-feasibility-v1/report.json`,
  SHA `8a18ea2f95fc4eeab6dd951054660c03d41f174c7f4bd714bf109049219e9525`.

```bash
PYTHONPATH=src:. .venv/bin/python \
  scripts/run_taska_socket_cyclic_origin_transfer.py --device mps

PYTHONPATH=src:. .venv/bin/python \
  scripts/diagnose_taska_socket_cyclic_selector_feasibility.py --device mps
```

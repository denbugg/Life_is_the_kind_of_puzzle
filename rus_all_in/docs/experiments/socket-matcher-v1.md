# SocketMatcher v1: board-conditioned partial matching

## Вердикт

`SocketMatcher v1` подтвердил новый **локальный** сигнал, но не решил глобальную
раскладку. На основном source-disjoint train-dev-16 прогоне contextual OT
поднял pooled neighbor retrieval относительно bilateral control:

- `R@1: 7.9144% -> 10.1279%` (`+2.2135 pp`);
- `R@32: 42.0686% -> 52.5419%` (`+10.4733 pp`).

При переводе scores в строгую перестановку лучший v1 arm по adjacency дал
`4.0874% -> 6.4368%`, но direct placement остался около случайных
`1/576 = 0.1736%`, а raw SSIM снизился `0.113324 -> 0.104368`. Поэтому v1 —
**research evidence, не production/submission model**. Повторять тот же scalar
dustbin вариант на большем бюджете без исправления border/global decoder не
нужно.

## Что именно проверялось

Модель имела 150 148 параметров: `d=32`, 4 attention heads, один whole-board
Transformer layer, один SocketGNN layer и 10 Sinkhorn iterations. Она не
получала positional embedding от случайного входного индекса tile. Для каждой
оси выполнялся partial `576 x 576` OT с массой dustbin, равной 24: в правильной
сетке ровно 552 исходящих и 552 входящих соседних socket-а имеют пару.

Обучение объединяло:

- synthetic puzzles с известной точной перестановкой;
- real train boards с восстановленными по clean target labels и маской
  наиболее уверенной половины tiles по margin.

Оценочные predictions строились только из dirty input и замораживались до
чтения target. Оба evaluation-набора принадлежат manifest `train`, не
пересекаются со своими train sources; calibration, holdout и competition test
не открывались. При этом reference permutation восстановлена из dirty/clean
пары и **не является organizer ground truth**, поэтому абсолютные layout-числа
остаются диагностическими.

## Pilot: train-64, synthetic-100 + real-100, dev-8

Артефакт:
[report.json](../../outputs/socket-matcher/pilot-train64-s100-r100-dev8-v1/report.json)
(`SHA-256 77122bd627b559417f817c81bf0dd97665905b46de38ec45f7465f5117a6e6b6`).
Checkpoint:
`outputs/socket-matcher/pilot-train64-s100-r100-dev8-v1/socket_matcher.pt`,
SHA-256 `9197ab20f651d9c3f16475868b6a8b12e18cb764772cf6b02079d36c55964dfe`.

Конфигурация: 64 real train boards, 100 synthetic steps на grid 12, затем 100
real steps; evaluation offset 128, count 8; CPU. Pooled socket retrieval был:

| Scores | R@1 | R@5 | R@16 | R@32 |
|---|---:|---:|---:|---:|
| Raw contextual logits | 4.8800% | 16.1798% | 31.6576% | 44.8936% |
| Partial-OT logits | **5.9556%** | **18.4896%** | **34.3184%** | **47.1128%** |

В pilot report bilateral local-retrieval control ещё не записывался. Глобальные
результаты:

| Layout arm | Direct | Translation-aligned | Adjacency | Raw SSIM |
|---|---:|---:|---:|---:|
| `bilateral_buddies96` | 0.1085% | **1.0200%** | 3.5779% | **0.101718** |
| `socket_ot_buddies96` | 0.1519% | 0.8247% | 3.0005% | 0.099115 |
| `fused_ot_buddies96` | **0.1736%** | 0.8681% | 3.9402% | 0.099572 |
| `fused_ot_relax_border` | 0.1519% | 0.8898% | **4.0761%** | 0.094163 |

Pilot уже показывал, что fusion/relaxation может перенести часть signal в
adjacency, но не давал основания заявлять улучшение координат или SSIM.

## Scale continuation: train-256, real-400, fresh dev-16

Артефакт:
[report.json](../../outputs/socket-matcher/scale-train256-continue-r400-dev16-v1/report.json)
(`SHA-256 9fbcb98392b83a8e12d21adab43363047374898daf68458238495498fae4d754`).
Checkpoint:
`outputs/socket-matcher/scale-train256-continue-r400-dev16-v1/socket_matcher.pt`,
SHA-256 `986d31e4b8e99d33ec479b172cc8af3b293a9a44b8aafed2105504b7b7af39a6`.

Прогон продолжил pilot checkpoint на 256 real boards ещё 400 real steps без
дополнительных synthetic steps. Evaluation использовала свежие 16 sources с
offset 512. Ее pooled local retrieval:

| Scores | R@1 | R@5 | R@16 | R@32 |
|---|---:|---:|---:|---:|
| Bilateral control | 7.9144% | 18.1103% | 31.0349% | 42.0686% |
| Raw contextual logits | 9.1089% | 23.0469% | 38.5813% | 50.5605% |
| Partial-OT logits | **10.1279%** | **25.2831%** | **40.5118%** | **52.5419%** |

То есть выигрыш существует до decoder-а и усиливается OT-нормализацией. Но
глобальные метрики показывают оставшийся bottleneck:

| Layout arm | Direct | Translation-aligned | Adjacency | Raw SSIM |
|---|---:|---:|---:|---:|
| `bilateral_buddies96` | **0.2062%** | 1.0525% | 4.0874% | **0.113324** |
| `socket_ot_buddies96` | 0.1953% | 1.1068% | 5.3555% | 0.108869 |
| `fused_ot_buddies96` | 0.1845% | **1.1719%** | 5.9273% | 0.108665 |
| `fused_ot_relax_border` | 0.1736% | 1.0525% | **6.4368%** | 0.104368 |

Относительно bilateral arm это `+1.2681 pp` adjacency для socket buddies,
`+1.8399 pp` для fused buddies и `+2.3494 pp` для fused relaxation. Одновременно
direct placement изменился на `-0.0109/-0.0217/-0.0326 pp`, а raw SSIM — на
`-0.004455/-0.004659/-0.008956`. Следовательно, v1 научился лучше ранжировать
соседей, но старый conversion не зафиксировал компоненты в абсолютной сетке.

## Важная ошибка старой border-метрики

Поля v1 report `right_border_*`, `down_border_*`, `pooled_border_*` нельзя
цитировать как качество определения границы. В v1 один scalar bin score
реплицировался на все sockets, а OT содержал одну **агрегированную** dustbin
строку/колонку с суммарной массой 24. Сравнение этой агрегированной массы с
одной real-real ячейкой через `argmax` не является классификацией конкретного
tile как одного из ровно 24 border tiles; отсюда и бессмысленно большая
`predicted_border_fraction`.

Эта ошибка не отменяет real-real `R@K` и метрики уже собранных строгих layout
перестановок в таблицах выше. Она отменяет только старые aggregate-dustbin
border recall/precision/fraction и не позволяет трактовать scalar bin как
выученный spatial anchor.

## Почему нужен v2

`SocketMatcher v2` устраняет именно этот дефект, не меняя подтверждённый
contextual matching signal:

1. добавляет четыре per-socket border heads: right-out, left-in, bottom-out и
   top-in;
2. обучает их отдельным listwise border loss;
3. оценивает каждую сторону через exact top-24 projection, а decoder может
   сделать жесткую Hungarian-проекцию на 552 парных и 24 непарных socket-а;
4. interleaves exact synthetic boards с low-weight noisy real labels, чтобы
   border/cardinality supervision не растворялась в real continuation.

`outputs/socket-matcher/smoke-v2-border-head/report.json` — только smoke test
исполняемости warm-start и новых метрик, не quality evidence. Этот
содержательный gate теперь выполнен в
[source-disjoint отчёте SocketMatcher v2](socket-matcher-v2.md) вместе со
строгим [component/QAP decoder](socket-decoder-prototype.md). Слабый prior может тянуть
уверенный содержательный компонент ближе к центру; гладкие tiles при этом лишь
не получают центрального притяжения и не объявляются рамкой. Такой сигнал
допустим только как отключаемый component-level unary: прежний isolated-tile
semantic/centre hard rule история уже не поддерживает.

# Efficient Pasha-on-Socket top-32 reranker

## Статус и gate

**Prototype implemented; quality continuation stopped fail-closed; default не
изменён.** До реализации bounded path был завершён единственный matched
full-score diagnostic на уже открытых historical boards `6996–6999`:

| Arm | Pooled R@1 | R@5 | R@25 | Buddies96 adjacency | Aligned placement |
|---|---:|---:|---:|---:|---:|
| Pasha883 | 18.0707% | 34.8732% | 58.4918% | **6.8614%** | **1.0851%** |
| Pasha + Socket-OT rank50 | **19.6784%** | **37.2056%** | **60.6431%** | 5.6159% | 1.0417% |

Fixed fusion улучшил local retrieval, но ухудшил оба заранее требовавшихся
global signals: adjacency на `-1.2455 pp`, translation-aligned placement на
`-0.0434 pp`. Поэтому gate «улучшить pooled R@1 **и** decoder/global adjacency»
не пройден. Новый exact-synthetic label panel не открывался; competition test
не читался; promote/default claim запрещён. Matched report:
[`report.json`](../../outputs/socket-matcher/v2-d64-vs-pasha883-matched-last4/report.json),
SHA-256
`56f4dfe928ff0943f806df0283f9e5cc84d0dcff5e553bd9a32f103425303bbb`.
Все четыре boards отсутствуют в Socket d64 train/eval/exposed lineage, но входят
в historical Pasha validation/model-selection roster. Reference permutation
target-assisted, поэтому результат остаётся Pasha-source-exposed diagnostic.

## Bounded production-safe primitive

[`socket_pasha_topk.py`](../../src/aiijc_puzzle/socket_pasha_topk.py) реализует
строго ограниченный dirty-only путь:

1. d64 Socket partial-OT real block выбирает top-32 non-self candidates отдельно
   для каждого из 576 outgoing sockets и каждой оси;
2. archived C64 Pasha883 получает только эти ordered pairs — `576×32` horizontal
   и `576×32` vertical;
3. vertical contract совпадает с historical training: каждый 20×20 tile сначала
   транспонируется, затем ordered pair конкатенируется в 20×40;
4. внутри каждого candidate row Socket и Pasha независимо переводятся в stable
   descending ordinal percentiles; priority равна фиксированным
   `0.5*Socket-rank + 0.5*Pasha-rank`;
5. self и все неоценённые Pasha pairs получают единый finite mask `-1`, ниже
   диапазона admitted priorities `[0,1]`. Никакого Pasha score им не
   приписывается.

API ограничивает `top_k <= 32`, поэтому на board 24×24 Pasha исполняет ровно
`2×576×32 = 36 864` pair evaluations вместо `2×576² = 663 552`: в 18 раз
меньше примеров и лишь 5.56% full-pair pool. Он не создаёт dense Pasha score
matrix; dense `576×576` создаётся только для дешёвой decoder priority mask.

Интеграция намеренно консервативна. `decode_socket_with_pasha_topk_priority`
передаёт fusion только как `component_edge_priority` существующему decoder144.
Original Socket partial-OT assignment, exact-capacity hard projection, dustbin
mass, border unary и полный QAP/swap objective остаются неизменными. То есть
prototype не подменяет обученное transport distribution несовместимой rank
шкалой.

## Dirty-only benchmark

Скрипт
[`benchmark_socket_pasha_topk.py`](../../scripts/benchmark_socket_pasha_topk.py)
запущен на уже exposed train input `img_006999.png`, MPS, batch 2048. Он не
открывает target, reference permutation или manifest labels.

| Stage | Seconds |
|---|---:|
| Socket d64 matcher | 0.184 |
| Control decoder144 | 0.157 |
| Pasha top-32, обе оси | 4.912 |
| Reranked decoder144 | 0.172 |
| Added rerank + decoder after Socket | 5.084 |

Оба decoder outputs — строгие permutations; layout изменился, но benchmark
намеренно не вычисляет quality. Артефакт:
[`benchmark-img006999-mps.json`](../../outputs/socket-pasha-topk/benchmark-img006999-mps.json),
SHA-256
`65e3130ee8cbc46f63066f034348066a6a3498ace0ebbb4c2f6ac3faf6d5bc9f`.

## Tests и решение

Focused tests фиксируют:

- self masking и жёсткий cap 32;
- отсутствие Pasha imputation вне Socket top-K;
- точную 50/50 rank fusion и tile-permutation equivariance при нетождественных
  scores;
- ровно `2*N*K` model evaluations и exact vertical transpose;
- передачу priority в decoder без замены исходных assignments.

Primitive сохранён для возможного будущего теста после появления независимого
положительного global evidence. Сейчас его нельзя включать в submission или
production default: full-score upper-bound-like diagnostic уже провалил global
gate, а top-K approximation не имеет отдельного quality evidence.

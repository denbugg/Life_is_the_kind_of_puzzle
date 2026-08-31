# Edge-conditioned SocketPermutationFlow v1

## Вердикт

Prototype реализован и прошёл механический 4×4 capacity gate, но **провалил
source-disjoint 24×24 pilot**. Его нельзя включать в default или submission.

На одном 4×4 capacity source fixed decoder layout улучшился с `87.5%` до
`100%` direct placement; adjacency выросла `79.17%→100%`, strict permutation
сохранилась. Это показывает, что sparse GNN, coordinate flow, Sinkhorn и
Hungarian соединены правильно и способны представить нужную коррекцию.

На свежих четырёх 24×24 exact-synthetic sources direct placement, напротив,
снизился `0.3038%→0.2170%`, column accuracy — `5.8160%→3.9063%`, adjacency —
`15.6703%→1.2908%`. Row accuracy выросла `4.0365%→5.5122%`, но этот один
частичный сигнал не компенсирует разрушение правильных socket components.
Все predictions остались строгими перестановками.

## Чем это отличается от уже проваленных absolute heads

Перед реализацией повторно проверены P6, P10, P28, P37 и I21 в historical
ledger:

- P6/I21 предсказывали positional diffusion/directional state и переносились
  слабо;
- P10 использовал current layout, Fourier slots и Sinkhorn, но не имел сильного
  frozen edge graph;
- P28 был edge-conditioned coordinate denoiser, но не прошёл даже 2-board
  continuous-RMSE capacity gate и не обеспечивал строгую permutation на каждом
  шаге;
- P37/P38 были raw position-free relational Transformers с тем же слабым
  direct objective.

Текущий v1 не читает raw RGB и не имеет embedding shuffled tile index. Он
получает только frozen d64 SocketMatcher context + четыре post-GNN socket
embedding-а, partial-OT top-4 right/left/down/top graph, текущую строгую
раскладку и flow time. Три sparse edge-conditioned GNN слоя видят score,
direction и ошибку текущего relative displacement. Head выдаёт bounded
coordinate proposal, из него строятся row/column и joint slot logits; 8
Sinkhorn iterations и Hungarian после каждого из четырёх refinement steps
гарантируют биекцию.

Training states — строгие permutations между random или decoder144 start и
exact truth. Интерполяция последовательно ставит правильный tile swap-ом и не
создаёт coordinate collisions. Loss объединяет balanced assignment NLL,
row/column CE и continuous coordinate loss.

## Протокол

Использован frozen d64 SocketMatcher v2:
`outputs/socket-matcher/v2-d64-train1024-s1600-r400-dev32/socket_matcher.pt`.
Его verified lineage содержит 1 056 sources. Pilot дополнительно исключил
filename-ы из 73 ранее созданных reports, затем выбрал 16 flow-train и 4
flow-eval sources только из manifest `train`; панели не пересекаются. Для
каждого source создана одна deterministic official-like corruption и точная
inverse-shuffle label. Calibration, holdout и competition test не открывались.

4×4 capacity artifact:
[report.json](../../outputs/socket-permutation-flow/capacity-grid4-s400-d96-l3/report.json),
SHA-256 `454ac297e03a6e0b42a284d44d95a39567dfd8d8518e96019d534f1d224868c9`.
Это capacity-only run: его single eval source уже присутствовал в preceding
smoke и не является confirmatory evidence.

24×24 bounded pilot artifact:
[report.json](../../outputs/socket-permutation-flow/pilot-grid24-train16-s300-d96-l3/report.json),
SHA-256 `353819b0727fb9c6708ce697adabd04dc6c8bdfd1a944bed50148072db47ad90`.
Модель имеет 786 598 параметров, обучалась 300 board updates за 15.99 секунды;
frozen matcher evidence для 20 boards готовилось 30.18 секунды. Checkpoint
SHA-256: `0b728e06d0f4c99763b6f51a5ebf02ffe5d0259daa4c423095339b8b7f7da470`.

| Exact metric, eval-4 | Decoder144 | Flow, 4 steps | Delta |
|---|---:|---:|---:|
| Direct placement | 0.3038% | 0.2170% | −0.0868 pp |
| Row accuracy | 4.0365% | 5.5122% | +1.4757 pp |
| Column accuracy | 5.8160% | 3.9063% | −1.9097 pp |
| Translation-aligned | 3.4288% | 0.7812% | −2.6476 pp |
| Adjacency | 15.6703% | 1.2908% | −14.3795 pp |

Train-16 показывает тот же failure mode: direct не изменился, row/column стали
лучше на `+1.7361/+0.9006 pp`, но adjacency упала
`14.9966%→1.0700%`. Значит, проблема не только в source transfer: separable
absolute coordinate projection выбрасывает полезную rigid-component структуру
уже на flow-train boards.

## Зафиксированное решение

- Код и equivariance/capacity mechanics сохранить как prototype.
- Default matcher/decoder не менять.
- Не масштабировать тот же row/column/slot objective большим Transformer-ом:
  это снова повторит P10/P37 после уже наблюдаемого 24×24 failure.
- Если возвращаться к learned refinement, следующий вариант обязан сохранять
  trusted socket components как rigid units или проектировать под совместную
  `absolute + socket-edge` energy. Простое Hungarian по separable coordinate
  logits закрыто этим pilot-ом.


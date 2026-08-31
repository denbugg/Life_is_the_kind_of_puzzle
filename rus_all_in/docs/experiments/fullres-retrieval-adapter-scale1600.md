# Full-resolution retrieval adapter: fixed scale400→1600

Дата: 2026-08-31. Статус: **устойчивый положительный scaling signal, но
terminal gate не пройден**. Terminal16, layout decoder, competition test,
production и submission не открывались.

## Зачем запускался scale-up

Предыдущий fixed step100→400 pilot не дал standalone replacement ranker, но
показал одновременно положительный R@5 slope и растущий raw∪adapter top32
candidate supply. Это оправдало ровно один заранее зафиксированный более
длинный run той же retrieval-oriented full-resolution модели. Архитектура,
loss, frozen SocketMatcher, fit32 roster, augmentations и features не менялись;
checkpoint или threshold по local16 не подбирались.

До запуска подписан
`configs/fullres_retrieval_adapter_scale1600_preregistered_v1.json`, SHA-256
`840a50cba2dea4c7c57300f65ce18613d62ec26f696dd25311a58a25e0605563`.

- Training from scratch, seed `20260911`, один deterministic extended stream
  на `1600` update; первые 400 TrainSpec побитово совпадают с pilot stream.
- Fixed checkpoints `400` и `1600` находятся внутри одной trajectory.
  Cosine scheduler имеет заранее зафиксированный horizon `T_max=1600`, поэтому
  новый checkpoint400 является matched scale-control этой trajectory, а не
  копией старого pilot checkpoint400.
- Model: width32, 8 stride-one NAF blocks, разрешение `20×20` на каждом block,
  bounded residual `32/255`; никаких pooling/downsampling.
- Optimizer: AdamW, learning rate `5e-4`, weight decay `2e-4`; frozen d64
  SocketMatcher остаётся в eval mode.
- Raw d64 evidence неизменно и измеряется параллельно. Adapter pixels являются
  только matcher view и не могут попадать в submission pixels.
- Local gate был подписан до scoring: s1600 против raw обязан одновременно дать
  pooled R@1 `>=+0.5 pp`, R@5 `>=0`, matched reciprocal precision `>=+0.2 pp`
  при coverage `>=3%`. Только после этого разрешалось открыть никогда ранее не
  использованный terminal16.

## Local16 scaling diagnostic

Все top32 candidate matrices и reciprocal evidence были сохранены target-free
до exact-reference scoring. Проценты в таблице.

| View | pooled R@1 | pooled R@5 | pooled R@32 | raw∪view top32 gain | matched reciprocal gain vs raw |
|---|---:|---:|---:|---:|---:|
| raw d64 | `19.565` | `38.887` | `69.724` | — | — |
| adapter step400 | `19.758` | `39.464` | `70.363` | `+3.125 pp` | `+0.181 pp @ 46.98%` |
| adapter step1600 | `19.990` | `39.810` | `70.641` | `+3.844 pp` | `+0.845 pp @ 46.25%` |

Directional raw∪adapter1600 supply gains равны `+4.291 pp` right и
`+3.397 pp` down. Внутри одной trajectory step400→1600 выросли:

- pooled R@1: `+0.232 pp`;
- pooled R@5: `+0.345 pp`;
- pooled R@32: `+0.277 pp`;
- pooled raw-union coverage: `+0.719 pp`;
- checkpoint-matched reciprocal precision: `+0.686 pp` при coverage `46.25%`.

Таким образом, увеличение training budget дало согласованный положительный
retrieval slope, а не только случайный рост одного supply показателя.

## Gate и решение

Step1600 против raw прошёл R@5 (`+0.923 pp`) и reciprocal precision
(`+0.845 pp @ 46.25%`), но pooled R@1 вырос только на **`+0.425 pp`** при
заранее заданном минимуме `+0.5 pp`. Общий gate поэтому false. Terminal16
остался закрыт, Weco step140 не создавался, layout decoder и pair/exact metrics
не запускались.

Положительный step400→1600 slope разрешил создать
`configs/fullres_retrieval_adapter_server_scale1600_v1.json`, SHA-256
`f982f7b49cbbfc3ad1390db9a3491d97d1b387f08c78b1d7941f215130716b6c`.
Это только portable training artifact/config для будущего server run, не solver
promotion и не разрешение использовать adapter pixels в output.

## Следующий допустимый consumer

Standalone suffix уже был отрицательным, а независимый DINO screen показал,
что adapter supply остаётся complementary к raw+DINO. Поэтому следующий
materially new шаг — один vectorized raw+adapter1600+DINO verifier/selector,
обученный на объединённом candidate pool и использующий per-query rank,
reciprocity и cross-view agreement. Не повторять simple suffix, fixed rank
mixing, nearby checkpoint/threshold sweep или decoder до нового signed gate.

Per-query replay не требует повторной adapter inference: target-free archive
содержит для всех 16 local boards массивы top32 формы `576×32` для raw,
adapter400 и adapter1600 по обеим осям, а также reciprocal target/confidence
evidence:

- `outputs/fullres-retrieval-adapter/scale1600-local16-v1/local16/frozen-target-free-retrieval.npz`,
  SHA-256 `985f4953c10255b3194a8f08bb05248333e60271716535c8a833230f47a0d5f0`;
- metadata SHA-256
  `93c0e2159f761ec05faca6038ecf9fb031e1522f9381588638aa6a2ea40f87e2`;
- pre-score freeze SHA-256
  `622c22847f3db40de50d8be75069df42a6f55cf7a8052216fe51af79c64c4884`.

## Runtime, артефакты и проверки

MPS training занял `1119.28 s` (`18.65 min`); prefetch wait `0.030 s`.
Seeds и stream фиксированы, однако из-за известных nondeterministic MPS
scatter/index-put kernels bitwise-repeatability весов не заявляется.

- report: `outputs/fullres-retrieval-adapter/scale1600-local16-v1/report.json`,
  SHA-256 `47ce8b176d2da5b6c278af6bc66be27464d87465cca6507a94e57e569f0ec796`;
- checkpoint400 SHA-256
  `1ec080b8e733e458ddb1eef34a10e7d0c5528addb34a8dbea5e0075ee6802ae0`;
- checkpoint1600 SHA-256
  `51beee8dea615e00440f90737ee537244dcf26934e9e292ac7a33bea235e6a48`;
- runner SHA-256
  `8172f019e47c02111c27268a0e8d7baf187977e310168029e87cc688910d2e8b`;
- targeted test SHA-256
  `8cec8eec7f69dfcf18165245d95cf03ceb2ad44b8164c03f4df94bac045719d6`.

Weco Observe step `138` фиксирует interim checkpoint400, step `139` —
выполненный checkpoint1600 retrieval result в pair и exact research trees с
parent step138. Оба шага явно retrieval-only: pair/exact layout metric не
заявляется. Step140 не использован. `ruff` и targeted tests прошли.

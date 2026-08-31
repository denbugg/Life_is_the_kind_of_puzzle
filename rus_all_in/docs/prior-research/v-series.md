# Аудит neural/contour-линии V10–V30

[К сводному индексу](README.md)

Источник — `origin/codex/contour-normalization` на `1a714e115d`. Ветка имеет
182 коммита, 175 после ранней базы `origin/pasha883`; до `5e36b3b` она делит
историю с M-серией, а восемь последних коммитов публикуют V10–V30 и результаты
V25–V30.

Главный документ ветки — `docs/NEURAL_PIPELINE_V10_V28.md`; несмотря на имя,
на tip в него дописаны V29 и V30. Числа сверены с JSON reports и отдельными
`.weco/*/README.md`/`report.md`. Эта линия использует свои split-ы и метрики,
поэтому её проценты нельзя напрямую сравнивать с M-, E- или P-сериями.

## Результат линии

V-линия построила наиболее последовательную neural retrieval → global solver
цепочку в репозитории:

- лучший retrieval — **V28**, где raw RGB, denoised grayscale и learned
  contours дополняют V27: top-1 15.73%, top-5 29.20%, top-32 51.45%, MRR
  23.02% на 11 свежих сценах;
- лучший global solver — **V30 graph unary + LNS**: adjacency 10.57%,
  direct placement 0.197%, translation-aligned placement 2.13%, composite
  0.11106 на 15 сценах;
- V30 улучшает same-run baseline по adjacency на 8.83% и composite на 8.25%,
  direct placement меняется лишь с 0.150% до 0.197%, а translation-aligned
  снижается 2.18%→2.13%; точная сборка не решена;
- после V28 нет нового untouched terminal split. V29 использует 3-fold OOF на
  15 уже доступных score-cache сценах. V30 обучает свои heads на 52 сценах,
  выбирает параметры на 8 disjoint validation scenes, а финально оценивается
  на тех же 15 уже просмотренных caches. Дальнейший model selection требует
  настоящего CV или нового holdout.

## Полный леджер версий

| Версия | Идея | Честный результат / статус |
|---|---|---|
| V10 | 77.4M dense scene CNN + 14-layer permutation-equivariant Transformer | База архитектуры; V18 — её hard-negative continuation. |
| V11 | Global solver, размещающий все 576 tiles | Служит baseline для поздних V29/V30; global placement остаётся низким. |
| V18 | Natural hard-negative fine-tune V10 | 8 сцен: top-1 9.64%, top-5 19.71%, MRR 15.02%, global placement 1.26%. Frozen backbone V22. |
| V20/V21 | Cycle GNN и calibration graph reranking | Отклонены: маленький top-1 сдвиг сопровождался ухудшением top-5, MRR или assembly. |
| V22 | Multiscale boundary cross-attention reranker поверх V18 top-32 | На тех же 8 сценах top-1 13.38%, top-5 26.21%, MRR 19.56%, global placement 1.61%. Не может восстановить отсутствующего в V18 top-32 соседа. |
| V23 small | Noise-invariant multiscale boundary bi-encoder, 1.06M | 8 сцен: 10.33/21.13/39.27% top-1/5/32. Принят как быстрый candidate generator. |
| V23-XL | Та же идея, 5.76M | 16 сцен: 11.62/23.31/41.79%; сильнейшая одиночная V23. |
| V23 ensemble | small 0.25 + XL 0.75 + handcrafted seam 0.50 | 16 сцен: 12.95/24.98/43.69%, MRR 19.36%. |
| V24 | Cross-attention residual поверх V23 | Отклонён: alpha 0.18 ухудшил 16×16 validation; calibrated alpha 0.025 дал +0.06 pp top-1, но снизил top-5 и top-32. |
| V25 | Linear V22+V23 fusion, alpha 0.55 выбран на отдельной calibration | 16 holdout сцен: 14.32/27.32/45.61%, MRR 21.05%. Union recall 43.01% @32 и 51.91% @64. Принят. |
| V26 | Learned listwise MLP над union top-64, 19 329 params | 14.65/27.85/46.09%, MRR 21.42%; все четыре метрики выше V25. Post-hoc beta-to-3 extension отклонён. |
| V27 | Query-conditioned set-transformer, 1.06M | 15.14/28.17/46.95%, MRR 21.99%; partial gate — top-5 на 0.034 pp ниже V26. Scene 6973 adjacency 7.79→8.97%, translation-aligned placement 1.39→1.22%. |
| V28 | RGB + U-Net denoised gray + soft/binary learned contours, fused с V27 | 11 fresh scenes: 15.73/29.20/51.45%, MRR 23.02%; все retrieval gates пройдены. Scene 6989 adjacency 10.33→11.05%, translation-aligned placement 1.74→2.78%. |
| V29 | Six-layout portfolio, Hungarian/swap refinement, 3-fold candidate selector | 15 cache scenes: baseline composite 0.10112, fixed packed1 0.10653, OOF selector 0.10665 (+5.47%). Oracle portfolio 0.11178; continuous Sinkhorn relaxation отклонена на smoke. |
| V30 | Directed recurrent GNN row/col/border unaries + pairwise/unary LNS | Validation: row 10.35% vs chance 4.17%, col 8.57%, border F1 54.65%. 15 scenes: adjacency 10.57%, direct placement 0.197%, translation-aligned 2.13%, composite 0.11106. Лучший proxy-composite result линии. |

## Почему V28 работает лучше

V28 standalone слабее V27 на top-1, но сильнее на top-32:

| Модель, scenes 6989–6999 | Top-1 | Top-5 | Top-32 | MRR |
|---|---:|---:|---:|---:|
| V27 | 14.38% | 27.07% | 45.75% | 21.09% |
| V28 standalone | 12.01% | 25.36% | 49.77% | 19.32% |
| V27 + V28 | **15.73%** | **29.20%** | **51.45%** | **23.02%** |

То есть новые modalities в первую очередь расширяют candidate supply, а V27
возвращает точный low-rank порядок. Это согласуется с M-серией: разнообразие
input view полезнее ещё одной random initialization той же модели. Именно этот
механизм, а не «contours сами решают пазл», следует переносить.

## V29/V30: что удалось глобальному solver-у

На 15 сценах V29 выбирал одну из шести полных раскладок. Его OOF selector
забрал примерно половину доступного oracle portfolio gain. Непрерывная
576×576 Sinkhorn-релаксация ухудшила smoke objective и adjacency.

V30 добавил слабые, но настоящие absolute unaries. Pairwise edge calibrator
имел AP 0.8127, однако проявил domain shift между V27 support score и fused V28
score, поэтому победитель честно использует `solver_gamma=0`. GNN unaries и LNS
остались:

| Same-run arm | Adjacency | Translation-aligned | Composite |
|---|---:|---:|---:|
| complete baseline | 9.72% | **2.18%** | 0.10260 |
| V30 graph unary + LNS | **10.57%** | 2.13% | **0.11106** |

Direct placement на тех же 15 сценах вырос с 0.150% до 0.197%, несмотря на
небольшое снижение translation-aligned числа. На scene 6989 adjacency выросла
11.23→14.67%. Это реальный proxy/adjacency gain, но не доказательство высокого
end-to-end SSIM и не exact reconstruction.

## Протокольные ограничения

- V18/V22 измерены на 8 сценах, V23–V27 преимущественно на других наборах из
  16 сцен, V28 — на 11 новых сценах, V29/V30 — на 15 cache scenes. Сравнивать
  версии можно только по явно приведённым matched tables.
- V27 формально не прошёл strict all-metrics gate, хотя выиграл 3 из 4 retrieval
  метрик. V26 оставался принятым baseline до fusion с V28.
- V29 — 3-fold OOF на малой выборке. V30 имеет disjoint train/validation для
  своих параметров, но финальная таблица использует те же 15 ранее просмотренных
  caches; ни один из них не является новым terminal test.
- Top-k считает точного индексного соседа. После M420 top-k/top-32 надо
  пересчитать с content-equivalent positives: V28 может быть лучше или хуже,
  чем говорит exact-index label.
- `translation-aligned placement` не равно absolute placement и тем более не
  равно leaderboard SSIM.
- Dataset и большинство upstream checkpoints/secrets не закоммичены; абсолютные
  пути remote runners ведут под `/home/kva`.

## Что не повторять

- V20/V21 graph reranking с прежним objective;
- fixed-alpha V24 и его post-hoc beta/alpha подстройку;
- continuous Sinkhorn layout relaxation из V29;
- V30 edge calibrator в текущем виде без score-domain matching;
- объявление уже просмотренных scenes новым independent test;
- вывод «set-transformer победил» только по top-1: V27 не прошёл собственный
  all-metrics gate и дал mixed assembly outcome.

## Reusable artifacts

В отличие от многих старых веток, поздняя V-линия сохраняет не только prose:

- `.weco/puzzle-transformer-kaggle/train_puzzle_transformer_v10.py`;
- `.weco/puzzle-hard-finetune-v18/train_hard_v18.py`;
- `.weco/puzzle-boundary-reranker-v22/train_boundary_v22.py`;
- `.weco/puzzle-boundary-biencoder-v23-remote/`;
- `.weco/puzzle-v18-v22-v23-fusion-v25-remote/`;
- `.weco/puzzle-union-reranker-v26-remote/`;
- `.weco/puzzle-set-transformer-v27-remote/` с report и audit image;
- `.weco/puzzle-multimodal-boundary-v28-remote/` с report и audit image;
- `.weco/puzzle-global-soft-v29-remote/` с evaluator, 80 KB JSON report и
  audit image;
- `.weco/puzzle-edge-unary-lns-v30-remote/` с trainer, evaluator, report,
  audit image и **committed `solver_v30.pt` размером 728 053 bytes**;
- `docs/neural_pipeline_results.json` и сводный Markdown.

Upstream V18/V22/V23/V28 weights и score caches всё равно нужны извне; один
V30 checkpoint не делает цепочку автономной.

## Как использовать эту линию дальше

Переносить имеет смысл не «V30 как готовое решение», а три проверенных блока:

1. V28 multimodal candidate generator — самый высокий измеренный exact-index
   top-32 в своей линии.
2. V26/V27 union reranking — небольшое, но устойчивое извлечение пользы из
   разнородных кандидатов.
3. V30 unary-aware LNS — лучший global decoder при доступных score matrices.

Перед дорогим retrain нужно объединить их с content-aware target из M420 и
сделать общий source-disjoint CV protocol. Иначе получится ещё одно улучшение
несопоставимой offline-метрики.

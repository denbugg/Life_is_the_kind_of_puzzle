# PAZZLE: neural boundary retrieval and global assembly

Экспериментальный пайплайн для восстановления пазлов 24×24 из 576 зашумлённых тайлов.
Текущая neural-ветка исследует устойчивое представление границ, dense scene-level
Transformer, cross-attention reranking и последующую глобальную сборку.

Актуальная сводка, честные holdout-метрики и статус каждого эксперимента:
[docs/NEURAL_PIPELINE_V10_V28.md](docs/NEURAL_PIPELINE_V10_V28.md).

Основной подтверждённый результат — мультимодальный fusion V27+V28:

- top-1 правильного соседа: **15.73%**;
- top-5: **29.20%**;
- top-32: **51.45%**;
- MRR: **23.02%** на 11 новых полных пазлах 24×24.

На удалённой RTX 4060 также обучен быстрый V23 boundary candidate generator.
Калиброванный V23 ensemble достигает **43.69% recall@32**, fusion V25 — **45.61%**,
а обучаемое ранжирование V26 — **46.09%**.

V28 явно добавляет U-Net denoised grayscale и learned soft/binary contours к RGB.
На свежих сценах 6989–6999 fusion V27+V28 достигает **15.73% top-1**, **29.20% top-5**,
**51.45% top-32** и **23.02% MRR**, улучшая V27 по всем retrieval-метрикам.

Глобальный solver V29 строит несколько полных раскладок: сохраняет главный anchor,
упаковывает дополнительные coordinate-consistent компоненты с разными `top-k`, затем
применяет Hungarian refinement и swap local search. Маленький confidence-ranker выбирает
раскладку по seam-score и согласованности графа. В трёхфолдовой оценке на 15 сценах
adjacency выросла **9.57% → 10.12%**, а composite assembly score — **+5.47%**.

Датасет: [VSOS AI Initiative PAZZLE](https://www.kaggle.com/datasets/pasha883/vsos-ai-initiative-pazzle).
Тяжёлые датасеты и checkpoints намеренно не хранятся в Git.

> Исходная ветка принадлежит Pasha883; экспериментальная работа ведётся в отдельных ветках.

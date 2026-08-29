# PAZZLE: neural boundary retrieval and global assembly

Экспериментальный пайплайн для восстановления пазлов 24×24 из 576 зашумлённых тайлов.
Текущая neural-ветка исследует устойчивое представление границ, dense scene-level
Transformer, cross-attention reranking и последующую глобальную сборку.

Актуальная сводка, честные holdout-метрики и статус каждого эксперимента:
[docs/NEURAL_PIPELINE_V10_V27.md](docs/NEURAL_PIPELINE_V10_V27.md).

Основной подтверждённый результат — V26 learned union reranker поверх V25:

- top-1 правильного соседа: **14.65%**;
- top-5: **27.85%**;
- top-32: **46.09%**;
- MRR: **21.42%** на 16 полных пазлах 24×24.

На удалённой RTX 4060 также обучен быстрый V23 boundary candidate generator.
Калиброванный V23 ensemble достигает **43.69% recall@32**, fusion V25 — **45.61%**,
а обучаемое ранжирование V26 — **46.09%**.

V27 set-transformer проверен на новом test split: он немного улучшает top-1, top-32,
MRR и adjacency сборки относительно V26, но из-за микроскопического снижения top-5
пока оставлен экспериментальным, а не объявлен новым строгим победителем.

Датасет: [VSOS AI Initiative PAZZLE](https://www.kaggle.com/datasets/pasha883/vsos-ai-initiative-pazzle).
Тяжёлые датасеты и checkpoints намеренно не хранятся в Git.

> Исходная ветка принадлежит Pasha883; экспериментальная работа ведётся в отдельных ветках.

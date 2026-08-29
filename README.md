# PAZZLE: neural boundary retrieval and global assembly

Экспериментальный пайплайн для восстановления пазлов 24×24 из 576 зашумлённых тайлов.
Текущая neural-ветка исследует устойчивое представление границ, dense scene-level
Transformer, cross-attention reranking и последующую глобальную сборку.

Актуальная сводка, честные holdout-метрики и статус каждого эксперимента:
[docs/NEURAL_PIPELINE_V10_V25.md](docs/NEURAL_PIPELINE_V10_V25.md).

Основной подтверждённый результат — V25 fusion V22 + V23:

- top-1 правильного соседа: **14.32%**;
- top-5: **27.32%**;
- top-32: **45.61%**;
- MRR: **21.05%** на 16 полных пазлах 24×24.

На удалённой RTX 4060 также обучен быстрый V23 boundary candidate generator.
Калиброванный V23 ensemble достигает **43.69% recall@32**, а его fusion с V22 — **45.61%**.

Датасет: [VSOS AI Initiative PAZZLE](https://www.kaggle.com/datasets/pasha883/vsos-ai-initiative-pazzle).
Тяжёлые датасеты и checkpoints намеренно не хранятся в Git.

> Исходная ветка принадлежит Pasha883; экспериментальная работа ведётся в отдельных ветках.

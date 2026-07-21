# Puzzle restoration and assembly

Решение задачи восстановления изображений 480x480, разбитых на случайно
перемешанные и независимо повреждённые фрагменты 20x20. Итоговая система
восстанавливает содержимое тайлов, оценивает их соседство и позицию, собирает
сетку 24x24 и формирует проверенный Kaggle submission.

## Codex research pipeline

Экспериментальная реализация находится в
[`research/codex_pipeline`](research/codex_pipeline/README.md) и отделена от
основного кода в `src/`.

Pipeline состоит из:

1. supervised residual-модели для очистки фрагментов;
2. `EdgeMatcher` для оценки правых и нижних соседей;
3. `PositionPrior` для оценки строки и колонки;
4. глобальной assignment + local-search сборки;
5. RL actor-critic, предлагающего перестановки фрагментов;
6. guardrail, отклоняющего RL-расклад при ухудшении baseline objective;
7. атомарной сборки и проверки ZIP с 700 PNG.

Последний четырёхкартинный smoke-test новой генеративки вместе с RL:

| Вариант | Mean SSIM |
|---|---:|
| Restorer + heuristic | 0.175991 |
| Restorer + guarded RL | **0.179611** |
| Предыдущий audit-fixed smoke | 0.176371 |

Полный журнал запусков, включая неудачные подходы и причины выбора моделей:
[`EXPERIMENTS.md`](research/codex_pipeline/EXPERIMENTS.md).

## Материалы

- [Описание и запуск pipeline](research/codex_pipeline/README.md)
- [Хронология экспериментов](research/codex_pipeline/EXPERIMENTS.md)
- [Модели, Kaggle kernels и SHA256](research/codex_pipeline/MODEL_MANIFEST.md)
- [Аудит solver-а](research/codex_pipeline/docs/solver_audit_2026-07-20.md)
- [Регрессионные тесты](research/codex_pipeline/test_solver_regressions.py)

Чекпоинты и submission-архивы не хранятся в Git из-за размера. Их источники и
контрольные суммы зафиксированы в model manifest.

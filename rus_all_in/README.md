# AIIJC Puzzle Reconstruction

Рабочее пространство для задачи «Восстановление пазла» основного этапа AIIJC.
Оригинальное условие сохранено в [docs/AI Challenge.pdf](docs/AI%20Challenge.pdf).

> **Portable snapshot note.** Этот каталог — self-contained code/docs/runtime
> срез ветки `rus-all-in`, но намеренно без raw train/test, submission ZIP,
> predictions, временных caches, DINOv2 и DRUNet weights. Promoted relation
> layout pipeline снабжён всеми небольшими runtime weights и полным
> fail-closed formal archive по ожидаемым paths. Полный allowlist, SHA-256,
> omitted assets и честный verification status находятся в
> [SNAPSHOT-MANIFEST.md](SNAPSHOT-MANIFEST.md).

Результаты прежних экспериментов собраны в
[сводном индексе исследований](docs/prior-research/README.md): покрыты все
23 remote-ветки и 477 коммитов, построены поисковая матрица идей, полный реестр
M1–M420 и список перспективных направлений.

Новые воспроизводимые проверки на общем frozen protocol сведены в
[центральном реестре экспериментов](docs/experiments/README.md). Там отдельно
зафиксированы подтверждённый pixel tail, строгая проверка M420, candidate-supply
gate и закрытая после scale-up формулировка content-aware verifier-а. Новый
selective-target500 pair solver прошёл независимый gate; для него добавлен
отдельный SHA-gated layout-only adapter, но официальный submission не менялся.
Короткий указатель на официальный leaderboard best и текущий подтверждённый
pair-solver: [BEST current](docs/BEST-current.md).

Текущее состояние, официальный best, Weco exact/pair lineages и точная очередь
solver-экспериментов сохранены в
[handoff от 2026-08-31](docs/solver-research-handoff-2026-08-31.md). Предыдущий
[снимок от 2026-08-30](docs/solver-research-handoff-2026-08-30.md) оставлен как
историческая точка.
Следующие materially distinct solver-ветки, быстрые stop/gates и
отдельный handoff для 40–60 GB GPU сведены в
[NEXT solver roadmap](docs/NEXT-solver-roadmap.md).

Самый сильный текущий legal restoration candidate — фиксированная
[DRUNet40 + protected h28/h40 композиция](docs/experiments/pretrained-drunet-protected-stack.md):
на reused-calibration-120 она прошла quantitative primary gate с mean SSIM
`0.271644` и root manual gate. На disjoint confirmation relative gains точно
повторились, но absolute mean снизился до `0.262817<0.27`. Поэтому full gate
провален: это подтверждённый bounded component, а не новый production
submission. Последующее [полное calibration-700
измерение](docs/experiments/pretrained-drunet-protected-stack-all700.md) дало
authoritative aggregate `0.268270`, только 3/10 fixed folds выше `0.27`, и
narrow safety FAIL на одной board; fail-closed holdout-700 не открывался.

Для этой композиции подготовлен отдельный
[fail-closed production/validation scaffold](docs/drunet-protected-production-v1.md),
но он не запускался: confirmation не прошла frozen gate, immutable production
authorization отсутствует, старый fallback не изменён.

Параллельно подготовлен новый [fixed-B standard fail-closed
scaffold](docs/fixed-b-standard-production-v1.md) для единственного legal arm
DRUNet50 → h20/h28/h50 → t60 h28-safe/h50-flat. Он также не запускался и не
открывал test: production разрешается только после двух immutable all700 отчётов
(calibration и unchanged holdout) в диапазоне `[0.27, 0.28`, provenance/safety/
flatness pass и ручной проверки root. Promotion config пока намеренно отсутствует.

Текущий честный production artifact описан в
[финальном handoff](docs/final-solution.md): frozen no-atlas strict layout →
RGB+luma harmonization → один colored NLM `h20` pass дал `0.253128` на
одноразовом внутреннем holdout-96. Это proxy, а не официальный leaderboard
score, и правильность скрытой перестановки не доказана. Production завершён:
700 predictions в `outputs/compliant-submission/predictions/` и
`outputs/compliant-submission/submission.zip` построены, а
встроенная и повторная независимая проверки дали
`METHOD_COMPLIANT_LAYOUT_ACCURACY_UNPROVEN`. SHA-256 ZIP:
`7c36307af0ea821c8a5fbf3139323ece332744dcf59a413198dd96d5a2f619bf`.
Это технически проверенный artifact, но не доказанно правильный layout и не
заявление о manual-compliant/submission-ready решении:
[visual audit](outputs/compliant-submission/visual-audit/REPORT.md) не нашёл
целостной сцены ни в одном из 24 просмотренных результатов.

Последний масштабированный pairwise ranker (`k16`, train256) улучшил adjacency
на `+0.030005`, но ухудшил frozen h20x1 SSIM на `−0.009386`; его gate провален,
а confirmation/holdout/test не открывались.

Перед любым packaging обязателен
[manual-compliance gate](docs/submission-compliance.md): output должен содержать
биективную раскладку всех 576 исходных fragments и только последующее
restoration качества. Constant/parametric/low-frequency canvases и существующий
constant-median ZIP помечены **NONCOMPLIANT / DO NOT SUBMIT**.

Для SocketMatcher sorter подготовлен отдельный
[production-safe resumable runner](docs/socket-sorter-production.md): строгий
v2/v3 checkpoint → decoder144 → opt-in cyclic border5 → биективная сборка
исходных tiles, с identity pixel-tail по умолчанию. Это только scaffold:
финальный checkpoint/tail ещё не выбраны, 700 competition test им не запускались.

Для layout выбран [Union-v2](docs/experiments/raw-twin-union-reranker.md):
frozen source-disjoint `64×draw0` confirmation без retrain дал exact
`0.938→1.281` tiles/board (`+0.344`, clustered 95% CI пересекает ноль),
adjacency `13.668→14.419%` (`+0.752 pp`, CI положительный) и
fixed top144 `+5.266` correct edges/board. Predeclared submission gate пройден;
все `128/128` layouts — строгие перестановки исходных upright tiles.
Competition test для этой проверки не открывался.

Для promoted Union-v2 подготовлен отдельный [fail-closed production packager и
независимый validator](docs/union-v2-submission-production.md). Frozen MPS
metadata-only dry-run прошёл exact official roster/hashes без чтения test PNG и
без записи outputs; полный 700-board run запущен после promotion. Он сохраняет
строгую перестановку original upright tiles, проводит raw-pixel audit до
RGB-offset → bounded-luma → single colored NLM `h20` и публикует отдельный ZIP,
не изменяя прежний `outputs/compliant-submission/`.

Фактический `Union-v2+h20` ZIP получил user-reported official score
`0.24201676406343967` против `0.2762279116935955` у предыдущего
`fixed-B standard+buddies96`; submitted combined arm отклонён и не должен быть
отмечен лучшим. Это не чистая solver-ablation, потому что вместе с layout был
заменён restoration tail. MPS full-replay validator также обнаружил
недетерминированный `index_add_`; подробности и корректная граница вывода
зафиксированы в production и Union-v2 experiment docs.

## Задача

Входное RGB-изображение 480×480 состоит из 576 независимо искажённых фрагментов
20×20, перемешанных в сетке 24×24. Нужно одновременно восстановить расположение
фрагментов и качество исходного изображения.

- train: 7 000 пар `inputs` / `targets`;
- test: 700 входных изображений без ответов;
- метрика: средний RGB SSIM (`channel_axis=2`, `data_range=255`, остальные параметры
  `skimage.metrics.structural_similarity` по умолчанию);
- submission: ровно 700 RGB PNG 480×480 с исходными именами, в корне ZIP-архива.

## Быстрый старт

```bash
uv sync
uv run python scripts/prepare_data.py
uv run python scripts/build_validation_manifest.py --run
uv run aiijc-check
uv run pytest
uv run jupyter lab
```

`uv` использует Python 3.11 из `.python-version`, локальную `.venv` и версии из
`uv.lock`. На Apple Silicon smoke-check автоматически выбирает MPS, если backend
доступен.

Команда `build_validation_manifest.py` детерминированно делит 7 000 train-пар на
`5600/700/700`, проверяет SHA-256 каждого input/target и записывает
`data/interim/validation_manifest.json`. Текущий protocol digest:
`2a9e3b74f7defa8c00846a05eb598fd263fd16c2787c70e77d3b7a4b585bfbf4`;
competition test в manifest не входит.

## Данные

Оригинальные файлы организаторов лежат без изменений в `data/raw/archives/`.
`scripts/prepare_data.py` безопасно и идемпотентно проверяет их структуру и распаковывает
рабочие копии:

```text
data/raw/
├── archives/
│   ├── train.zip
│   ├── test.zip
│   └── submission.zip
├── train/
│   ├── inputs/       # 7 000 PNG
│   └── targets/      # 7 000 PNG
└── test/             # 700 PNG
```

Предоставленный `submission.zip` задаёт только формат: его PNG побайтно совпадают с
`test.zip` и не являются ответами. Подробности хранения данных описаны в
[data/README.md](data/README.md).

## Среда

- численные вычисления: NumPy, SciPy, pandas, scikit-learn;
- изображения: Pillow, OpenCV, scikit-image, Albumentations;
- глубокое обучение: PyTorch, torchvision, timm;
- сборка и оптимизация: NetworkX, OR-Tools, Optuna;
- эксперименты: Matplotlib, Seaborn, JupyterLab, tqdm;
- качество кода: pytest, coverage, Ruff.

Для редкой неподдерживаемой MPS-операции можно запустить процесс с
`PYTORCH_ENABLE_MPS_FALLBACK=1`.

## Структура проекта

```text
configs/          конфигурации экспериментов
data/raw/         неизменяемые исходные данные
data/interim/     промежуточные представления
data/processed/   подготовленные выборки и признаки
docs/             условие и документация
notebooks/        baseline организаторов, EDA и быстрые эксперименты
scripts/          служебные воспроизводимые команды
src/              код решения
tests/            автоматические проверки
outputs/          прогнозы, логи и submission-архивы
artifacts/        модели, кэши и крупные артефакты
```

Содержимое `data/raw`, `data/interim`, `data/processed`, `outputs`, `artifacts` и
`tmp` не попадает в Git.

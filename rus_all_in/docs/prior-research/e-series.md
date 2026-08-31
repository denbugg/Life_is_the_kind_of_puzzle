[← Индекс предыдущих исследований](README.md)

# Аудит веток fast-score / E1–E20

Дата аудита: 2026-08-29. Исходный репозиторий изучался read-only:
`/Users/rusyalain/Documents/GitHub/pazzle_will_be_killed`.

## 1. Итог в одном экране

Среди этого семейства экспериментов есть два локальных победителя:

1. **E14 — лучший проверенный layout solver на frozen cache**: learned directional
   log-probabilities + `0.2` raw MGC/SSD, затем sparse relaxation/Hungarian.
   На paired full-128 относительно зафиксированного SA: robust SSIM
   `+0.0011229061`, mean SSIM `+0.0012070220`, adjacency `+0.0171323030`,
   end-to-end runtime `125.0452 s` вместо `428.7263 s` (`3.43x` быстрее).
2. **E18b — лучший локальный pixel postprocess**: тот же E14 layout, full-image
   colored NLM `h=9`, затем возврат только вновь появившихся серых 20x20-ячеек
   к raw-пикселям. На тех же 128 случаях: robust SSIM `+0.0652273999`, mean
   `+0.0678174924`, `128/128` побед, layout/adjacency неизменны.

Но **готового Kaggle-чемпиона из этого ещё не получилось**:

- remote validation self-contained E18b/E14: `0.180304` против названного в
  логе v5 baseline `0.187267`, delta `-0.006963`;
- тестовый проход шёл примерно `13.8–14.2 s/image`, дошёл только до `189/700`
  и был отменён часовым лимитом; submission не создан;
- offline E14 был проверен с `DirectionalTransformer` на raw tiles, а Kaggle
  port получает logits **другого** `EdgeMatcher` на restored tiles. Победа
  frozen-cache поэтому не переносится автоматически на production;
- validation gate в текущем production-коде не выбирает v5 при отрицательной
  дельте: он лишь отключает `relation_guard`, после чего E14 и E18b всё равно
  исполняются на test.

Самое перспективное продолжение — не новый solver ablation, а сначала честно
воспроизвести score source победившего E14 в production, исправить selection
gate и убрать двойной дорогой путь. Из новых/недопроверенных идей наиболее
интересны E15 (положительный smoke, но не прошёл строгий gate и имеет overlap
risk) и E13 (на момент аудита содержательный border encoder ещё не был
обучен; текущий bounded follow-up ниже дал отрицательный результат).

## 2. Покрытие refs и git-графа

В задании ref `fast-score-gen1` фактически находится как
`origin/autoresearch/fast-score-gen1`.

`rev-list --count` включает общий корневой snapshot `8460d6d`.

| ref | tip | reachable commits | что находится на вершине |
|---|---:|---:|---|
| `origin/autoresearch/e1-margin` | `c2c4f96` | 4 | E1 reciprocal-margin |
| `origin/autoresearch/e2-score-fusion` | `63c1456` | 5 | E2 raw MGC/SSD fusion |
| `origin/autoresearch/e3-cache-multistart` | `72a9c3b` | 5 | E3 Cython SA kernel; E9 только остановленный план |
| `origin/autoresearch/e4-bestbuddy` | `44a874a` | 4 | E4 component initializer |
| `origin/autoresearch/e11-relaxation` | `4d67749` | 5 | E11 relaxation solver |
| `origin/autoresearch/e12-cpsat` | `581c8f7` | 6 | E3 + E12 CP-SAT LNS |
| `origin/autoresearch/e13-border-encoder` | `a605814` | 6 | E3 + не запущенный E13 training package |
| `origin/autoresearch/e14-fusion-relaxation` | `2087f8d` | 6 | E11 + E14 evaluator/evidence |
| `origin/autoresearch/e14-kaggle-port` | `2fd08f5` | 9 | интегрированная цепочка до E14 Kaggle port |
| `origin/autoresearch/e15-no-gray-multiplex` | `77496e4` | 10 | E14 port + E15 |
| `origin/autoresearch/e18-nlm-polish` | `0d8a526` | 12 | E14 port + E18/E18b + self-contained bundle |
| `origin/autoresearch/e19-nlm-dual-view` | `0e58675` | 10 | E14 port + E19 |
| `origin/autoresearch/e20-restored-ranker-verifier` | `a877065` | 10 | E14 port + E20 coverage gate |
| `origin/autoresearch/fast-score-gen1` | `c57c126` | 13 | E18b champion chain + общий отчёт и remote follow-up |

Всего по этим refs достижимы **28 разных commit objects**, включая общий root.
Все они учтены ниже. Повторяющиеся cherry-pick-пары имеют одинаковый stable
patch-id, кроме особо отмеченного E11:

| commit object(s) | логическое изменение | покрытие |
|---|---|---|
| `8460d6d` | общий baseline snapshot (152 файла) | root всех 14 refs |
| `5c8bc58` | запуск E8 low-LR continuation | общий предок всех refs |
| `7f602c0` | отрицательный legacy two-side move result | общий предок всех refs |
| `ceea9ca` | artifact отрицательного E8 | все refs, кроме E1/E4 |
| `c2c4f96` | E1 | только E1 ref |
| `44a874a` | E4 | только E4 ref |
| `63c1456` | E2 | только E2 ref |
| `72a9c3b`, `43bc158` | E3, exact same patch-id `fe85b3…` | standalone E3 / интеграционная цепочка |
| `581c8f7` | E12 | только E12 ref |
| `a605814`, `c0c3fec` | E13, exact same patch-id `15b189…` | standalone E13 / интеграционная цепочка |
| `4d67749`, `fed2d61` | E11 | итоговые 9 E11 blobs идентичны; patch-id различается, потому что `fed2d61` также заменяет уже модифицированный E3 solver blob |
| `2087f8d`, `731c6f3` | E14, exact same patch-id `1fc7a5…` | standalone E14 / интеграционная цепочка |
| `2fd08f5`, `e182f68` | E14 Kaggle port, exact same patch-id `688dc3…` | standalone port / интеграционная цепочка |
| `77496e4` | E15 | только E15 ref |
| `0e58675` | E19 | только E19 ref |
| `a877065` | E20 | только E20 ref |
| `5670208`, `e2e028b` | E18 offline evidence, exact same patch-id `059864…` | E18 ref / fast-score ref |
| `01e5bd9`, `f98a6b5` | E18b production port, exact same patch-id `1075b7…` | E18 ref / fast-score ref |
| `0d8a526`, `7628213` | self-contained E18b bundle, exact same patch-id `dfeae3…` | E18 ref / fast-score ref |
| `c57c126` | `AUTORESEARCH_REPORT.md` + remote follow-up | только fast-score ref |

Важная деталь интеграционной ветки: E3 сначала добавляет опциональный Cython
backend в `global_solver_candidate.py`, но поздний `fed2d61` полностью заменяет
этот файл реализацией E11. Поэтому E3-файлы остаются в дереве, однако E3 backend
**не используется** E14/E18 champion solver.

## 3. Общий экспериментальный протокол

Большинство layout ablations использует один frozen NPZ:

- path по документации: `outputs/directional_student_holdout128.npz`;
- SHA-256:
  `74db2b62e9d5eafffae33117c7771512d823b0dcaa0095ef5807adb8e86a25df`;
- 128 grouped real-noisy изображений, каждое — 576 RGB-тайлов 20x20, сетка
  24x24;
- arrays: `right`, `down`, `pos`, raw `tiles`, `target`, `truth`, `stems`;
- selection должна видеть только явно разрешённые input arrays; `target` и
  `truth` читаются evaluator только после фиксации layout;
- основной seed: `20260818 + index * 100`; alt seed добавляет `1_000_003`;
- smoke/tune: indices `0..31`; untouched verification: `32..127`;
- SSIM: `skimage.metrics.structural_similarity(..., channel_axis=2,
  data_range=255)`;
- adjacency: доля истинных горизонтальных/вертикальных связей из 1104;
- `robust SSIM` здесь — не CI и не официальный metric: `mean - 0.5 * std`
  четырёх interleaved fold means (`scores[offset::4]`).

Baseline SA в парных экспериментах:

- Hungarian initializer только по `pos`;
- objective = `right + down + 0.11 * pos`;
- 400,000 swap steps, температура `1.0 -> 0.0001`;
- 10% random swaps, иначе best-neighbour guided swap;
- seed формула выше.

Текущий fetch не содержит самого NPZ, checkpoints, sidecar NPZ или generated
submission archives (`outputs/` и `models/` пусты). В Git сохранены JSON,
скрипты и иногда hashes/provenance. Поэтому численные результаты можно
аудировать, но полностью переисполнить без внешних artifacts нельзя.

## 4. Общий baseline snapshot и shared-history commits

### `8460d6d` — baseline snapshot

Это большой начальный snapshot, не один новый experiment: 152 файла, 24,199
строк, десятки train/evaluate/Kaggle scripts и artifacts. Для текущего семейства
важны следующие зафиксированные компоненты:

- `global_solver_candidate.py`: описанный выше 400k SA;
- `artifacts/directional_student_full576_holdout128_metrics.json`:
  `DirectionalTransformer`, checkpoint epoch 14, `tau=0.1`, right R@1
  `0.155571`, down R@1 `0.167856`; исторический full-128 output robust
  `0.0998174972`, mean `0.1021895386`, adjacency `0.0869423687`;
- `precompute_directional_student_solver_cases.py`: frozen `right/down` были
  получены именно `DirectionalTransformer` из raw input tiles, батчами по 192,
  с dot-product side embeddings / tau и row `log_softmax`;
- `artifacts/post_assembly_denoise_holdout128_metrics.json`: прежний sweep на
  128 случаях уже сравнивал full-image NLM `h=3,5,7,9` и bilateral. `h=9`
  был лучшим: mean `0.159581629`, robust `0.155339069`, gain `+0.062853601`,
  `128/128` wins. Именно отсюда позднее взят E18 `h=9`;
- `artifacts/restored_border_ranker_metrics.json`: 12 epochs, лучший/последний
  R@1 `0.386836`, R@5 `0.688516`; checkpoint этого ranker позже используется
  только в незапущенной части E20;
- `artifacts/full_pipeline_ssim_real_restorer.json`: на 20 случаях restored
  pixels давали mean `0.168405` против raw `0.100013`, но adjacency была лишь
  `0.02962`;
- `artifacts/pair_relation_restorer_continue_metrics.json`: предыдущий
  continuation достиг best restored macro-F1 `0.520624` на epoch 4;
- source-aware и другие более ранние ветки также присутствуют в snapshot, но
  не являются дельтами изучаемых refs и должны рассматриваться в общем аудите
  остальных веток.

Кавеат baseline numerics: исторический E0 (`0.0998175/0.1021895`) не равен
matched full-128 SA из E14 (`0.1003414/0.1027065`). E14 сравнивается со своим
парным baseline на том же evaluator/cache/seed; смешивать эти две строки нельзя.

### `5c8bc58` + `ceea9ca` — E8 low-LR continuation

**Гипотеза.** Ещё одна эпоха relation classifier с меньшим LR улучшит restored
pair relation scores.

**Изменение.** `kaggle_continue_pair_on_restorer.py`: `EPOCHS=1`, `LR=5e-6`,
input checkpoint `pair_relation_restorer_continued_best.pt`; 4,000 train
images, 64 pairs/image, batch 512, clean replay 0.25, confidence floor 0.05,
power 1.5. Private T4 kernel metadata добавлен в `5c8bc58`; результат JSON —
в `ceea9ca`.

**Результат.** На 21-stem strict holdout, 50,000 samples/mode:

- restored macro-F1 `0.503714397 -> 0.500354725`;
- restored adjacency accuracy `0.77126 -> 0.69198`;
- clean macro-F1 вырос `0.795991 -> 0.827584`, но это не целевой restored
  domain;
- epoch 0 остался best (`best_restored_macro_f1=0.503714397`).

**Вердикт.** DROP. Exact one-epoch `5e-6` continuation повторять не нужно.

**Кавеаты.** Artifact не содержит checkpoint SHA, runtime или raw Kaggle log.
Его baseline restored macro-F1 также ниже `0.520624` в старом continuation
artifact, хотя filename checkpoint совпадает; причин расхождения в ветке нет.
Надёжен paired вывод «эта эпоха ухудшила текущую evaluation», но не абсолютное
сравнение двух JSON из разных запусков.

### `7f602c0` — legacy two-side/block move experiments

Это не fast-score E1/E2, а generation 3 старого block-solver run.

На smoke-32:

| метод | robust SSIM | delta | adjacency | delta adjacency |
|---|---:|---:|---:|---:|
| baseline | `0.094709247` | — | `0.087409420` | — |
| `two_side` | `0.094997943` | `+0.000288695` | `0.084324049` | `-0.003085371` |
| `two_side_block2` | `0.095055245` | `+0.000345998` | `0.084352355` | `-0.003057065` |

Обе идеи DROP по dual-metric gate: небольшой SSIM выигран ценой истинных
локальных соседей. Reuse: artifact `artifacts/two_side_smoke32_metrics.json` и
legacy `autoresearch-runs/directional-block-solver-v1/solver_candidate.py`.

## 5. Детальный audit каждой экспериментальной ветки

### E1 — reciprocal-margin confidence bonus (`c2c4f96`)

**Гипотеза.** Увеличить score `A->B` на `beta=0.5`, если B — row top-1 A,
A — column top-1 B, а обе top1-top2 margins не меньше `0.5`. Это должно
сохранять надёжные reciprocal edges в SA basin.

**Протокол.** Неизменный 400k SA; smoke-32 indices 0–31; seed 0 и offset
`1_000_003`; raw learned matrices, diagonal явно исключена. Gate: robust delta
`> +0.0005`, mean `>0`, adjacency `>=0`. Hold96 разрешён после первого smoke.

**Результаты.** Declared seed:

- robust `0.094709247 -> 0.095464233`, `+0.000754986`;
- mean `0.098239138 -> 0.098561346`, `+0.000322208`;
- adjacency `0.087409420 -> 0.090579710`, `+0.003170290`;
- 18/32 SSIM wins, 17/32 adjacency wins;
- runtime `136.5973 -> 137.5388 s`;
- в среднем 66.0 right и 68.21875 down boosted edges.

Alt seed:

- robust `-0.002893613`, mean `-0.002614382`, adjacency `-0.001103940`;
- 10/32 SSIM wins; runtime `162.8071 -> 163.1130 s`.

**Вердикт.** DROP: фиксированный bonus меняет SA basin, но не создаёт
seed-stable signal. Точный `(beta=.5, margin=.5)` повторять не нужно.

**Artifacts/code.** `autoresearch-runs/fast-score-e1-margin/` содержит README,
RESULTS, evaluator, два полных smoke JSON и три run scripts.

**Кавеат.** Hold96 был остановлен после 41/96; метрика правильно не заявлена,
но сам partial JSON/log в commit отсутствует. Следовательно, holdout evidence
доступен только как prose, не как проверяемый artifact.

### E2 — raw MGC+SSD score fusion (`63c1456`)

**Гипотеза.** Classical seam evidence дополняет learned directional score.

**Реализация.** Из raw tiles строятся bidirectional Mahalanobis Gradient
Compatibility и one-pixel SSD. Каждая dissimilarity row-normalized median/MAD,
затем 50/50 MGC/SSD, ещё одна robust calibration и row `log_softmax`.
`fused = 0.8 * learned_logp + 0.2 * classical_logp`; diagonal `-1e4`.

**Протокол.** Тот же smoke-32 и два seeds; output pixels raw; target/truth
evaluation-only. Gate дополнен end-to-end runtime `<=1.1x` baseline.

**Declared seed.** Robust `+0.005158306`, mean `+0.005466989`, adjacency
`+0.001217165`; 27/32 SSIM wins. Right R@1 `0.163610 -> 0.165025`, down
`0.169044 -> 0.172441`. Classical prep `12.6851 s`; e2e ratio `1.103975x`,
по строгому runtime gate уже небольшой FAIL.

**Alt seed.** Robust `+0.003162781`, mean `+0.003060300`, но adjacency
`-0.001330389`; 20/32 SSIM wins. Prep `12.9652 s`; e2e `1.119752x`.

**Вердикт.** DROP standalone по joint gate, но **механизм сохранён**: SSIM
signal положителен на обоих seeds. Этот code не следует снова проверять отдельно
— он уже точно перенесён в E14, где оказался частью победившей композиции.

**Reuse.** Самый удобный frozen implementation:
`autoresearch-runs/e14-fusion-relaxation/e2_raw_fusion.py`; standalone evaluator
даёт rank diagnostics.

### E3 — compiled exact SA hot loop (`72a9c3b` / `43bc158`)

**Гипотеза.** Cython-компиляция неизменного 400k SA освободит wall-clock для
multistart без изменения trajectory.

**Реализация.** `SOLVER_BACKEND=cython` вызывает `global_solver_kernel.pyx`.
Сохранены NumPy `Generator`, порядок RNG calls, Python set для порядка affected
positions, acceptance math, schedule и 400k steps. Build через
`setup_e3_fast.py`.

**Протокол/result.** Warm runtime, build/import исключены; frozen smoke-32,
seed 0:

- exact layouts/SSIM/adjacency `32/32`;
- max objective delta `0.0`, failures 0;
- Python `128.781667 s`, Cython `37.834321 s`;
- ratio `0.293787`, speedup `3.403832x`;
- profile: `solve_layout` 3.526/3.800 s (`92.8%`) на одном case.

**Вердикт.** KEEP как efficiency tool для исходного SA. На metric ничего не
меняет.

**Кавеаты/relation.** Timings получены на конкретной локальной macOS/arm64
сборке и warm extension. E11 позже заменяет solver file, поэтому champion
E14/E18 не использует этот backend. Нельзя суммировать E3 speedup с E14 speedup.

#### E9 equal-wall-clock multistart

В E3 PLAN заявлен E9: потратить выигранное время на независимые SA starts и
выбрать максимум прежнего objective. Run остановлен на 3/32; aggregate metric
не заявлен. В Git нет multistart implementation или partial artifact. Это
**не отрицательный результат**, а незавершённая идея; из-за победы E14 её
приоритет низкий, но считать проверенной нельзя.

### E4 — reciprocal component initializer (`44a874a`)

**Гипотеза.** Reciprocal high-margin edges образуют coordinate-consistent
components; их strongest-first размещение по суммарному `pos` даст SA лучший
initial basin.

**Реализация.** Margin `0.5`, reciprocal row/column top-1, cycle/collision/span
checks; components >=2 занимают клетки раньше, остаток дополняет Hungarian.
400k SA после initializer неизменён.

**Smoke-32 seed0.** Robust `0.094709247 -> 0.093436066`
(`-0.001273181`), mean `-0.001150397`, adjacency `+0.017946105`; 14/32 SSIM
wins, 31/32 adjacency wins; runtime `1.00978x`; все permutations valid.

**Вердикт.** DROP по SSIM, несмотря на сильный topology signal. Alt-seed был
отменён после 9/32 и не используется.

**Вывод.** Жёсткие локальные компоненты реально сохраняют neighbours, но
ухудшают глобальную pixel alignment/origin. Exact initializer повторять не
нужно; компонентный signal полезен только как soft evidence/diagnostic.

### E11 — sparse relaxation labeling (`4d67749` / `fed2d61`)

**Гипотеза.** Глобальная propagation по sparse reciprocal directional graph
лучше локального SA.

**Реализация.** Top-12 joint row/column relative edges, normalized sparse
incoming/outgoing graphs, Sinkhorn 14 steps, Hungarian projection, confidence
freezing и четыре frozen phases:

1. `(temp=.45, edge=1.5, inertia=.10, hard=.55, iter=4, freeze=0)`;
2. `(.28, 3, .08, .70, 5, .03)`;
3. `(.16, 6, .06, .85, 6, .08)`;
4. `(.09, 10, .04, .94, 20, .15)`.

Best intermediate выбирается по исходному cached objective; seed используется
только для tie noise `±1e-7`.

**Smoke-16 seed0.** Robust `+0.001144168`, mean `+0.001580124`, adjacency
`+0.013643569`; 11/16 SSIM и 14/16 adjacency wins; runtime `11.853 s` против
`63.205 s` (`5.33x`).

**Alt seed.** Candidate aggregate bit-identical, но другой SA baseline оказался
лучше: robust `-0.001436797`, mean `-0.001038402`; adjacency всё ещё
`+0.007755888`; 5/16 SSIM wins. Candidate cached objective хуже SA примерно на
`-757`/case (`-6617.75` против `-5860.30`).

**Вердикт.** DROP standalone по alt-seed SSIM. Topology mechanism подтверждён,
SSIM superiority — нет. Полный 128 не запускался.

**Relation/reuse.** Именно solver E11 без изменения принимает E2 fusion в E14
и становится победителем. Переиспользовать `global_solver_candidate.py` на
E11/E14 tree или self-contained `kaggle_e14_solver.py`.

**Кавеат seed gate.** Alt test в основном меняет stochastic baseline; E11
candidate сам почти детерминирован. Это хороший paired stress test, но не
доказательство вариативности E11 trajectory.

### E12 — sparse CP-SAT 4x4 repair (`581c8f7`)

**Гипотеза.** Exact all-different local optimization исправит слабые регионы
SA лучше heuristic swaps.

**Реализация.** После baseline SA выбираются три weakest non-overlapping 4x4
windows. Tile set окна фиксирован, outside board фиксирован. CP-SAT видит
top-16 gains над row floor, `0.11*pos`, internal/boundary adjacency Boolean
links; 1 second/window, 1 worker, integer scale 1000. Candidate принимается
только при росте sparse objective.

**Smoke-16 seed0.** 48 solves: 24 OPTIMAL, 24 FEASIBLE; 44/48 repairs accepted;
sparse gain `+82.114257`, но sum dense learned objective delta
`-692.079590`. Robust SSIM `-0.000289359`, mean `-0.000277039`, adjacency
`+0.000113225`; 6/16 SSIM wins; runtime `60.997 -> 98.910 s`.

**Вердикт.** DROP на первом gate; alt/smoke32/holdout не запускались.

**Вывод.** Exact solver оптимизировал ровно заданный proxy, но truncation top-16
сломала соответствие dense score/SSIM. Повторять CP-SAT с тем же sparse floor
не нужно. Если возвращаться к exact/LNS, acceptance обязана сохранять dense
objective или другой валидированный global score.

### E13 — corruption-aware border encoder (`a605814` / `c0c3fec`)

**Гипотеза.** Shared CNN только по canonicalized 4-pixel borders, обученный
full-576 InfoNCE + batch-hard triplet на corruption curriculum, даст более
чистый neighbor ranking, чем whole-tile embeddings/raw borders.

**Locked package.** `BorderEncoder`: shared conv stack, 96-D side embeddings,
4 directional heads; `tau=.08`, triplet margin `.12`, 8 epochs, 160 steps/epoch,
batch 2, LR `3e-4`; до 7000 target images, grouped val 96, quick 16, final 48;
noise/blur/JPEG/edge erosion/combined; 52-minute internal T4 deadline.
Diagnostics должны считать clean/corrupt R@1/R@5, reciprocal precision/coverage,
Sinkhorn and Hungarian R@1 и сохранить 8 score matrices.

**Что реально выполнено.** По prose прошли compile, shapes, loss/backward,
5 corruption paths и assignment checks. GPU training не был запущен: Kaggle
CLI 2.2.4/2.2.3, legacy API и SaveKernel возвращали HTTP 404; retries остановлены.
Checkpoint, metrics, URL/version и score matrices отсутствуют.

**Исторический вердикт.** DESIGN/BLOCKED, а не failed experiment: в этой ветке
идея не была проверена.

**Reuse.** `kaggle_train_border_encoder.py`, metadata и push script полностью
самодостаточны. Это один из наиболее содержательных незапущенных кандидатов,
но его нужно сначала прогнать на независимом grouped split.

**Кавеат evidence.** Локальные check logs/tests не сохранены отдельным artifact;
доступны code + self-test + prose. Remote blocker исторический и больше не
должен считаться основанием не запускать локально/на другой GPU среде.

**Текущий follow-up (2026-08-30).** Пакет портирован без изменения старого repo
и прошёл bounded source-disjoint pilot: 256 train sources, 400 full-576 updates,
16 fresh exact-synthetic eval sources. Frozen d64 OT дал R@1/R@5
`18.654/37.494%`, E13 — `6.878/19.095%`, fixed 50/50 rank fusion —
`13.026/30.152%`; matched reciprocal precision также сильно снизилась. Gate
`+2 pp R@1` или `+5 pp precision @ >=3% coverage` провален, global decoder не
запускался. Поэтому E13 теперь **measured-negative at bounded gate**, а не OPEN;
полный протокол и artifacts — в [текущем отчёте](../experiments/corruption-aware-border-encoder-e13.md).

### E14 — E2 fusion -> E11 relaxation (`2087f8d` / `731c6f3`)

**Гипотеза.** E2 даёт устойчивый pixel/seam SSIM cue, но слабую topology;
E11 даёт topology, но standalone seed-unstable SSIM. Их ошибки дополняют друг
друга.

**Реализация.** Bit-for-bit E2 classical/fusion (`alpha=.2`) подаётся в
неизменный E11. Parameter sweep не проводился. Layout selection видит raw
tiles, learned `right/down`, `pos`; target/truth только для метрик.

**Пошаговая проверка.** Frozen cache SHA подтверждён. На одном frozen case E2
classical и fused arrays заявлены bit-identical исходному E2.

| split | robust delta | mean delta | adjacency delta | wins SSIM / adj | e2e ratio |
|---|---:|---:|---:|---:|---:|
| smoke16 seed0 | `+0.003054882` | `+0.003094520` | `+0.020380435` | 10/16, 15/16 | `0.28664x` |
| smoke16 alt | `+0.000473917` | `+0.000475994` | `+0.014492754` | 8/16, 13/16 | `0.28530x` |
| smoke32 seed0 | `+0.003121579` | `+0.002890809` | `+0.018851902` | 19/32, 29/32 | `0.29212x` |
| untouched96 | `+0.000631136` | `+0.000645760` | `+0.016559103` | 48/96, 87/96 | `0.29152x` |
| full128 aggregate | `+0.001122906` | `+0.001207022` | `+0.017132303` | 67/128, 116/128 | `0.291667x` |

Full-128 absolute paired metrics:

- SA robust/mean/adj `0.100341443 / 0.102706548 / 0.085512908`;
- E14 `0.101464349 / 0.103913570 / 0.102645211`;
- SA runtime `428.7263 s`;
- fusion prep `48.7808 s`, relaxation `76.2644 s`, e2e `125.0452 s`;
- 128/128 valid layouts, failures 0.

**Вердикт.** PASS layout на frozen cache. E2 и E11 отдельно больше не нужно
повторять; E14 — их проверенная композиция.

**Кавеаты.** Full-128 включает 32 tune/smoke cases; независимая часть — только
untouched96, где SSIM delta существенно меньше. Победа по SSIM не универсальна
(67/128 wins), тогда как adjacency signal гораздо устойчивее. Все metrics
offline; hidden leaderboard не проверен.

#### E14 Kaggle port (`2fd08f5` / `e182f68`)

`kaggle_e14_solver.py` переносит MGC/SSD и relaxation self-contained. В
`kaggle_solve_puzzles.py` он включён по умолчанию, имеет exception/invalid-
permutation fallback к legacy layout.

**Локальная parity evidence (case 0).** 4 tests, 3.113 s:

- classical right/down bit-identical;
- fused right/down bit-identical;
- final layout bit-identical и valid;
- synthetic exception возвращает exact legacy layout;
- hashes сохранены в `parity_result.json` (layout SHA
  `907143189f0137…`).

В самой E14-port ветке remote push не выполнялся из-за перенесённого API-404
blocker.

**Критическое domain mismatch.** Frozen E14 cache был создан
`DirectionalTransformer` epoch14 (`tau=.1`) **из raw input tiles**. Production
port вместо этого делает:

1. `FragmentRestorer`;
2. legacy `EdgeMatcher` на `clean_tiles`/restored tensors;
3. row `log_softmax` его logits;
4. только после этого E14 fusion/relaxation.

Parity test подставляет уже готовые frozen cache matrices с
`learned_are_logp=True`; production path `learned_are_logp=False` и реальная
модельная комбинация metric-parity не проходили. README честно говорит, что
full-128 metrics не являются claim о Kaggle checkpoint combination. Следующий
production run должен либо портировать настоящий DirectionalTransformer, либо
заново валидировать E14 на тех scores, которые реально будут в kernel.

### E15 — raw/guarded-restorer multiplex (`77496e4`)

**Гипотеза.** Сохранять raw E14 graph/objective, но при propagation добавить
независимый classical graph из guarded restored tiles:

`support = .70*raw + .30*guarded - .15*abs(raw-guarded)`.

Guarded layer — чистый MGC+SSD, не learned+classical fusion. Position weight,
top-k, phases, Hungarian, tie-break и best-layout objective остаются E14.

**Artifacts.** Restorer checkpoint epoch8 SHA
`6fcc7de2…`, `FragmentRestorer(base=64)`, 1,670,595 params, residual 0.5.
Sidecar SHA `65c04742…`, 69,387,927 bytes, mean 212.789 reverted tiles/case.
Binary отсутствует в Git, но provenance сохранён.

**Corrected smoke-16 seed0.** Относительно E14:

- raw robust `+0.001289442` (gate требовал `>=+0.002`);
- raw mean `+0.001768682`;
- adjacency `+0.006623641`;
- guarded robust/mean `+0.001936764 / +0.002539610`;
- raw SSIM wins 10/16, adjacency wins 11/16;
- runtime `15.7905 -> 23.0674 s`, `1.46084x`;
- gray delta `-194`, excess images 0, all layouts valid.

**Вердикт.** Формально DROP: единственный failed gate — raw robust threshold.
Alt seed, smoke32 и untouched96 не запускались.

**Сохранённый invalid diagnostic.** Первая ошибочная версия использовала
learned+guarded-classical fused second layer и дала robust/mean/adj
`-0.001898/-0.001375/-0.003057`; JSON помечен
`rejected_wrong_guard_fused_smoke16.json` и не является E15 metric.

**Кавеат.** Frozen 128 и restorer training split используют разные seeds, но
manifest/maps отсутствуют; исключить overlap нельзя. Результат provisional.
Тем не менее это самый сильный незавершённый layout extension в данном наборе:
все направления положительны, а fail — только заранее жёсткий порог на малом
smoke. Повторять стоит только на source-disjoint OOF и с заранее фиксированными
weights, не считать текущие цифры promotion evidence.

### E18/E18b — full-image NLM after fixed layout (`5670208`/`e2e028b`)

**Гипотеза.** Не менять layout вообще; сгладить corruption full-image
postprocess. Frozen OpenCV `fastNlMeansDenoisingColored`: `h=9`, `hColor=9`,
template 7, search 21.

**Safety split.** E18 — unguarded. E18b помечает 20x20 cell gray, если RGB mean
range `<10` и общий std `<25`, и возвращает к raw только cells,
которые стали gray после NLM, но не были gray до него. Gate требует не меньше
90% mean/robust gain E18, неизменный layout/adjacency и ни одного per-image
gray excess.

**Full-128.** Абсолютные значения:

| metric | raw E14 | E18 | E18b |
|---|---:|---:|---:|
| robust SSIM | `0.101464349` | `0.170337274` | `0.166691749` |
| mean SSIM | `0.103913570` | `0.175375886` | `0.171731063` |
| gain robust | — | `+0.068872925` | `+0.065227400` |
| gain mean | — | `+0.071462316` | `+0.067817492` |
| wins | — | 128/128 | 128/128 |
| gray cells | 17,996 | 19,644 | 16,776 |
| gray-excess images | — | 97/128 | 0/128 |

Guard reverted 2,868 newly-gray cells; retained 94.90% mean и 94.71% robust
gain. Layout/adjacency identical 128/128; adjacency `0.102645211`.
Runtime: E14 layout `126.9480 s`, NLM `14.6209 s`, guard `0.6533 s`, total
`142.2222 s` (~3.02x faster than frozen SA).

Smoke32 and untouched96 both дали 100% wins; untouched guarded robust/mean
gains `+0.065633373 / +0.067662376`, 0 gray excess.

**Вердикт.** E18 FAIL safety, E18b PASS pixels offline.

**Reuse/port.** `kaggle_e18b_postprocess.py` имеет bit-identical evaluator
parity и raw fallback. `build_e18b_self_contained.py` детерминированно встраивает
E14/E18b modules в единственный `kaggle_solve_puzzles_e18b.py`; tests проверяют
staleness, isolated import, no sidecar imports, layout identity и fallback.

**Кавеат parameter reuse.** `h=9` не sweep-ился в E18 commit, но ранее был
выбран как лучший среди `h=3/5/7/9` на baseline artifact с теми же 128 cases.
Следовательно, E18 full-128 не является совершенно untouched validation
параметра NLM. Сильные `128/128` gains уменьшают риск случайности, но свежая
OOF/hidden проверка всё равно обязательна.

#### Remote Kaggle follow-up (только prose в `c57c126`)

- private kernel `phoenix0501/pazzle-e18b-guarded-nlm`;
- self-contained imports, E14 и E18b прошли без fallback; guard работал;
- validation solver `0.180304`, reported v5 baseline `0.187267`, delta
  `-0.006963`;
- test ~`13.8–14.2 s/image`, `189/700`, затем
  `KernelWorkerStatus.CANCEL_ACKNOWLEDGED` около 3600 s;
- output files/submission отсутствуют.

Raw remote log/version artifact в Git не сохранён, поэтому это документированное,
но не независимо перепроверяемое из repo evidence.

**Критический control-flow bug.** В `main()` validation delta используется как
`use_relation_guard = delta >= 0`. Если delta отрицательна, test отключает
только relation guard. `solve_one()` всё равно после legacy/RL layout вызывает
E14, затем E18b. То есть фактического «fallback to v5» нет. Более того, даже
на test всегда вычисляется второй `baseline_pred = E18b(raw_v5)`, хотя caller
его выбрасывает. Это лишний NLM pass.

Название remote metric `v5_baseline_ssim` тоже требует точности: в этой версии
это не исторический v5 pixel output из `clean_tiles`, а E18b, применённый к
`raw_v5` layout. Validation delta поэтому в основном сравнивает два layout под
одинаковым postprocess, а не полный новый pipeline с прежним shipping output.
Production default выбирает 64 validation images; raw remote log не сохранён,
поэтому из prose нельзя проверить, был ли этот default переопределён.

Также перед E14 всегда выполняются restorer, legacy 20k optimizer и обычно RL;
E14 затем пересобирает layout. Это объясняет значительную часть remote runtime.

### E19 — per-tile NLM dual classical view (`0e58675`)

**Гипотеза.** NLM полезен не только после assembly, но и как независимый edge
view. Каждый raw tile отдельно denoised (`h=9`, 7/21), его MGC/SSD graph
усредняется 50/50 с raw classical graph; затем неизменный E14 `alpha=.2` и
relaxation. Output pixels остаются raw.

**Smoke-16 seed0.** Robust `+0.000009078`, mean `+0.000157636`, adjacency
`-0.000283062`; 9/16 SSIM wins, 11/16 adjacency wins. E2E `20.246 -> 48.661 s`,
`2.40343x`. Added NLM/classical view занял `28.194 s`; solver runtime почти
не изменился.

**Вердикт.** DROP по трём gates: robust меньше `+0.0005`, adjacency отрицательна,
runtime >2x. Alt/full не запускались. Exact raw+per-tile-NLM 50/50 view
повторять не нужно. Это не противоречит E18: NLM полезен как full-image pixel
operation, но почти бесполезен как independent per-tile edge evidence.

### E20 — restored BorderRanker verifier (`a877065`)

**Гипотеза.** Для каждой row объединить top-32 E14 и top-32 nearest restored
border descriptors, затем sparse-rerank union уже обученным BorderRanker.

**Locked unexecuted layout path.** Unguarded restored descriptors/ranker;
row robust z clipped `[-4,4]`; bonus
`S20 = S14 + .25 * good[i]*good[j]*z`; `good=~bad_mask` frozen no-gray guard;
unchanged E14 solver, raw output metric. Restorer epoch8 hash `6fcc7de2…`;
ranker epoch12 hash `8eb7b7e1…`, 153,745 params, border width6, candidates32;
sidecar hash `65c04742…`.

**Precondition coverage, cases 0–15.** Truth используется только после union
construction:

| direction | E14 top32 | union | delta | gate |
|---|---:|---:|---:|---:|
| right | `0.527060688` (4655/8832) | `0.578011775` (5105/8832) | `+0.050951087` | PASS |
| down | `0.524682971` (4634/8832) | `0.571218297` (5045/8832) | `+0.046535326` | FAIL `<.05` |

Mean union size: 56.884 right, 57.026 down.

**Вердикт.** Pipeline остановился до загрузки/вызова ranker и до любого layout,
SSIM, adjacency или runtime result. Exact coverage gate failed; E20
non-promotable также из-за недоказуемого training-stem overlap.

**Правильная интерпретация.** Проверен и слегка не прошёл только candidate
generator. Нельзя записывать «BorderRanker не работает»: его влияние на layout
вообще не измерено. Возвращаться имеет смысл лишь с OOF artifacts и новым
заранее обоснованным coverage protocol; подгонять top-k/threshold на этих 16
cases нельзя.

**Current source-disjoint resolution.** Точные historical weights/sidecar в
fetch отсутствуют, поэтому current bounded substitute использовал внешний
frozen DRUNet40 и заново обученный exact-synthetic residual ranker. На 16
source-held-out boards restored-descriptor union дал `+2.9778/+2.7627 pp`
top32 coverage, но d64→ranker R@1 изменился `17.9178→17.8442%`, R@5
`35.8016→35.8356%`, matched reciprocal precision упала на `−3.2461 pp`.
Predeclared local gate провален; decoder не запускался. Значит, restored
candidate emitter остаётся полезным primitive, а этот ranker закрыт как
bounded reject. [Полный current report](../experiments/restored-border-ranker-oof.md).

### `fast-score-gen1` final report (`c57c126`)

Commit добавляет `AUTORESEARCH_REPORT.md`, ссылку из root README и remote
follow-up в E18 RESULTS. Новых model/layout metrics или binary artifacts нет.
Он корректно сводит E1–E20 в ledger и выделяет E14/E18b, но несколько формулировок
нужно читать с указанными выше caveats:

- «Kaggle v5 fallback» фактически не реализован control flow;
- E14 offline score source и production score source различны;
- E18 `h=9` ранее выбран на этих же 128;
- remote evidence хранится только prose;
- E15/E20 model overlap не исключён.

## 6. Матрица «проверяли / не проверяли»

| идея | фактически проверено | статус для будущей работы |
|---|---|---|
| exact reciprocal bonus `.5/.5` в SA | 2 seeds × 32 | закрыто: seed-unstable |
| standalone raw MGC/SSD `alpha=.2` + SA | 2 seeds × 32 | не повторять; cue уже вошёл в E14 |
| Cython exact 400k SA | identity/speed, 32 | сохранить как tool |
| equal-wall-clock SA multistart | только 3/32, без code/artifact | не проверено; низкий приоритет |
| hard reciprocal component initializer | 32 seed0, alt 9 incomplete | exact variant закрыт по SSIM |
| low-LR `5e-6`, 1 epoch relation continuation | один remote epoch | закрыто для этой конфигурации |
| E11 standalone relaxation | 16 × 2 seeds | standalone закрыт; solver переиспользован E14 |
| top16 sparse CP-SAT 3×4x4 | 16 seed0 | закрыто: proxy mismatch |
| corruption-aware border CNN | historical: только local checks; current: 256-source/400-step exact pilot | bounded gate **провален**, не повторять standalone E13 |
| E14 E2->E11 | 32 tune + 96 untouched + alt smoke16 | лучший offline layout |
| E14 с production EdgeMatcher/restored scores | remote validation в составе E18b | проиграл v5; это не тот score domain, что offline |
| E15 raw/guarded multiplex | 16 seed0 | promising, но gate fail + overlap; не подтверждён |
| full-image NLM h9 | старый sweep + E18 full128 | сильный offline; fresh OOF нужен |
| E18b gray guard | full128 + parity tests | pass safety offline |
| per-tile NLM edge view | 16 seed0 | закрыто |
| E20 union coverage | historical16 + current exact16 | historical один direction чуть ниже gate; current `+2.978/+2.763 pp`, keep supply |
| E20 BorderRanker layout | current source-disjoint 256/16 local gate | ranker local gate **провален**, decoder/layout не открывать |
| self-contained E18b packaging | local isolated import + remote launch | packaging pass |
| end-to-end hidden submission E14/E18b | не завершён | не проверено |

## 7. Приоритетные направления

### P0 — привести production в соответствие с доказанным E14

Самый большой риск сейчас не solver, а score-domain mismatch. Нужно:

1. восстановить/переобучить и упаковать `DirectionalTransformer` epoch14 (или
   честно построить новый frozen cache для фактического `EdgeMatcher`);
2. сравнить E14 и v5 на source-disjoint local validation до remote run;
3. сохранять hashes score checkpoint и per-case matrix parity;
4. не использовать full-128 E14 дельту как доказательство для другого scorer.

Это имеет наибольшую вероятность объяснить remote `-0.006963`.

### P0 — исправить gate и runtime Kaggle path

Нужен настоящий выбор final pipeline по validation:

- при отрицательной delta использовать сохранённый v5 output/layout, а не
  только отключать relation guard;
- на test не считать `baseline_pred`, если он не нужен;
- если выбран E14, не выполнять legacy optimizer/RL layout path без необходимости;
- не запускать NLM дважды;
- профилировать отдельно restorer, edge scores, legacy solver, E14 MGC/SSD,
  relaxation и PNG encode;
- цель из remote evidence — `<~5 s/image` для 700 изображений/час.

После этого один полный remote run даст больше информации, чем очередной малый
solver ablation.

### P1 — fresh OOF revalidation E18b

Повторить фиксированный `h=9`, без sweep, на новом source-disjoint set и на
фактически выбранном production layout. Сравнить минимум: raw, original v5
pixel output, unguarded E18, guarded E18b. Gray guard оставить как независимый
safety audit. Это проверит, переносится ли огромный offline gain за пределы
ранее использованных 128 cases.

### P1 — E15 на честном OOF

E15 дал положительные robust/mean/adjacency и разумный runtime, проиграв только
строгому `+.002` threshold на 16 cases. Это сильнее большинства rejected
layout ideas. Перед повтором нужно создать restorer predictions строго OOF по
stem, зафиксировать `.70/.30/.15`, затем пройти alt seed и untouched split.

### P1/P2 — реально обучить E13 (resolved by current bounded pilot)

E13 — наиболее серьёзная незапущенная representation idea. Код и diagnostics
уже есть; исторический Kaggle 404 не является научным отрицательным результатом.
При запуске сохранить checkpoint hash, split manifest, raw/learned rank metrics
и downstream E14 ablation. Сначала проверить full-576 R@1/R@5 и reciprocal
precision, только затем layout.

Текущий workspace выполнил именно этот local-first gate в ограничении 400
updates: E13 и fixed fusion сильно проиграли d64, поэтому downstream layout не
открывался. Исходный исторический пакет по-прежнему не имеет своего 1,280-step
run, но этот пункт больше не считается приоритетным OPEN.

### P2 — OOF E20 (resolved by current bounded pilot)

Новый source-disjoint exact pilot сохранил candidate signal (`+2.978/+2.763 pp`
top32 coverage), но residual ranker не улучшил R@1 и проиграл matched reciprocal
precision. Decoder не открывался. Повторять checkpoint или подбирать weight на
открытой панели нельзя; возвращать emitter можно только внутри materially new
multi-view/context-aware ranker с новой панелью.

### Инфраструктурно полезное

- сохранить E3 Cython backend для быстрых matched-SA controls;
- стандартизовать manifest: cache/checkpoint/sidecar SHA, stem split, code SHA,
  seed formula, per-case JSON;
- сохранять remote Kaggle logs/status/files как artifacts, а не только prose;
- всегда сравнивать production score matrices с offline matrices, не только
  solver module на уже готовом cache.

## 8. Что точно не стоит повторять без новой механистической причины

- E1 exact bonus `beta=.5`, margin `.5`;
- legacy two-side/two-side-block moves;
- E4 hard component initializer;
- E8 одна эпоха continuation `LR=5e-6`;
- E11 standalone на исходных learned scores;
- E12 top16 row-floor CP-SAT windows;
- E19 50/50 raw/per-tile-NLM classical view;
- E2 standalone с SA (использовать только как часть E14);
- unguarded E18, если no-gray safety остаётся обязательным.

В рамках только исторического repo нельзя было помечать как failed E9
multistart, E13 border encoder и E20 ranker layout: они не получили metric
evaluation. Текущий bounded follow-up уже измерил и отклонил standalone E13;
source-disjoint E20 follow-up сохранил только candidate emitter и отклонил
ranker до decoder-а. Только E9 остаётся в прежнем неизмеренном статусе.

## 9. Карта reusable code и evidence

- E1: `autoresearch-runs/fast-score-e1-margin/`.
- E2 standalone: `autoresearch-runs/fast-score-e2-fusion/`; reusable frozen
  score code: `autoresearch-runs/e14-fusion-relaxation/e2_raw_fusion.py`.
- E3: `global_solver_kernel.pyx`, `setup_e3_fast.py`,
  `autoresearch-runs/e3-cache-multistart/e3_identity_smoke32.json`.
- E4: `global_solver_candidate.py` на `44a874a`, `smoke32_metrics.json`.
- E11: `global_solver_candidate.py` на `4d67749`, `test_relaxation_solver.py`,
  два smoke JSON.
- E12: `autoresearch-runs/fast-score-e12-cpsat/evaluate_e12_cpsat.py` и
  `results/smoke16_metrics.json`.
- E13 historical: `kaggle_train_border_encoder.py`, metadata/push script;
  metrics в старом repo нет. Current port/report находятся в
  [отдельном эксперименте](../experiments/corruption-aware-border-encoder-e13.md).
- E14: `autoresearch-runs/e14-fusion-relaxation/evaluate_e14.py`,
  `results/full128_aggregate.json`, `results/untouched96_seed0.json`;
  production solver: `kaggle_e14_solver.py`.
- E14 port evidence: `autoresearch-runs/e14-kaggle-port/parity_result.json`,
  `tests/test_kaggle_e14_port.py`.
- E15: `e15_multiplex_solver.py`, corrected `results/smoke16_seed0.json`,
  `sidecar_provenance.json`; binary sidecar отсутствует.
- E18: `evaluate_e18.py`, `full128.json`, `kaggle_e18b_postprocess.py`.
- E18 bundle: `build_e18b_self_contained.py`,
  `kaggle_solve_puzzles_e18b.py`, tests in `tests/`.
- E19: `e19_nlm_dual_view.py`, `results/smoke16_seed0.json`.
- E20: `e20_common.py`, `e20_verifier.py`,
  `results/coverage_smoke16.json`, `artifact_provenance.json`; layout JSON нет
  по дизайну gate.
- Общий индекс: `AUTORESEARCH_REPORT.md` на `c57c126`.

## 10. Evidence-quality caveats, которые надо не потерять при общем merge

1. Frozen NPZ и model checkpoints не находятся в Git; hashes есть не везде.
2. E14 независимая проверка — untouched96; full128 включает tune32.
3. `robust SSIM` — custom four-fold penalty, не статистический interval.
4. E15/E20 restorer/ranker stem overlap нельзя исключить.
5. Исторические E13 local checks и Kaggle blocker представлены prose без raw
   logs; текущий bounded run имеет отдельные hashed artifacts.
6. E1 partial hold96 и E9 partial 3/32 не сохранены как artifacts.
7. E18 `h=9` ранее выбран на тех же 128 cases.
8. Remote E18b evidence не имеет committed raw kernel log/output.
9. Offline E14 scorer (`DirectionalTransformer` raw) не совпадает с production
   scorer (`EdgeMatcher` restored).
10. Текущий remote validation control flow не делает заявленный v5 fallback.

С учётом этих ограничений надёжное заключение таково: **E14/E18b — лучшие
локальные компоненты и правильная отправная точка, но не готовый submission;
первым следующим экспериментом должен быть исправленный, score-matched,
ускоренный end-to-end validation/test pipeline.**

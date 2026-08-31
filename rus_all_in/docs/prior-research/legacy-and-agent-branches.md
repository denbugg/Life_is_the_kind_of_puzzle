# Аудит legacy / ORBIT-веток `pazzle_will_be_killed`

Навигация: [README проекта](../../README.md) ·
[сводная база предыдущих исследований](knowledge-base.md).

Дата аудита: 2026-08-29. Репозиторий-источник:
`/Users/rusyalain/Documents/GitHub/pazzle_will_be_killed`.

Охваченные ref:

- `origin/pasha883`;
- `origin/MAESTRO`;
- `origin/Taska-govna`;
- `origin/agent/research-restorer-rl-pipeline`;
- `origin/agent/ssim-scorer`;
- `origin/таска-говно`.

Аудит выполнен read-only через `git show`, `git diff`, `git ls-tree`,
`git rev-list` и поиск по blob-ам, без checkout и без изменений исследовательского
репозитория. Код не переисполнялся: необходимые train/test данные, checkpoints и
часть заявленных submission-архивов находятся на внешних дисках/Kaggle и в git
отсутствуют. Поэтому ниже строго различаются:

- измерение, для которого в git есть report/JSON;
- утверждение документации о внешнем запуске;
- результат, который в текущем tree воспроизвести невозможно;
- oracle/leaky diagnostic, который нельзя считать честной validation.

## 1. Краткий итог

1. `origin/pasha883` и `origin/MAESTRO` — один и тот же ref, а не две независимые
   линии работы. Это ранний общий фундамент: synthetic distortion, recovered
   train permutations, pairwise seam scorer, greedy/SA solver и NLM. Главный
   доказанный вывод — механика solver работает на oracle scores, но learned
   compatibility слишком слаба: `place_acc≈0.0015`, solve SSIM `≈0.106`,
   best-buddy precision `≈0.48`.
2. `origin/agent/research-restorer-rl-pipeline` добавляет residual restorer, PPO
   swap policy, строгий solver audit и 5-class relation classifier. Restorer
   полезен на tile-level, guarded RL даёт лишь маленький 4-image gain, а 700-image
   ZIP не имеет leaderboard score. Relation classifier после domain adaptation
   плато около accuracy `0.52`; последний graph-greedy solver v2 закоммичен без
   законченного результата.
3. `origin/agent/ssim-scorer` — не solver. В ветке лежат 18 чистых test references
   и browser SSIM tool. Это прямая test-label leakage; имена полностью совпадают
   с 18 source overrides из `origin/Taska-govna`. Скорер полезен только как
   отдельно помеченный forensic tool, не как честная validation.
4. `origin/Taska-govna` — наиболее системный ORBIT/Rank96 архив до E21. Frozen
   Rank96 получил external score `0.2161981413457065`, но submission состоял из
   682 generic outputs + 18 exact source overrides, поэтому score нельзя чисто
   приписать solver. Ветка подробно закрыла generator inversion, classical
   matching, DINO/global critic, Sinkhorn, RL/GA/search, denoise-before-scoring,
   clean-render restoration и несколько component/pose-graph decoders.
5. `origin/таска-говно` — отдельный монолитный full-history bundle. Его лучший
   честно описанный generic pipeline: TileNAF + C1/HBT + directional QAP + RGB
   harmonization + bounded luma. Exact RGB-only leaderboard `0.2167844489529071`,
   luma показан пользователем как округлённые `0.218`; exact luma score неизвестен.
   Главный reusable результат: candidate union уже содержит `72.98%` true edges,
   но production layout сохраняет лишь `~6%` adjacency; truth-filtered ceiling
   достигает SSIM `0.627267`, translation ceiling `0.7091`.
6. Повторять ещё один обычный pairwise seam CNN, denoiser-as-scorer, RL/SA/LNS на
   прежней энергии, one-shot Sinkhorn/set-to-grid, generic global critic,
   cycle/consensus поверх тех же noisy top-k edges или fidelity-restoration на
   неправильной раскладке не следует.
7. Самое перспективное честное направление — не новый renderer, а verifier,
   который видит несколько согласованных тайлов/компонентов и переоценивает
   high-recall candidate graph до packing. Сначала нужен all-emitter oracle
   coverage gate: E21 показал, что текущий nontrivial-component emitter pool
   принципиально не содержит достаточно связей.

## 2. Доказательство покрытия ref и commit

### 2.1. Tips и trees

| Ref | Tip commit | Tree | Reachable commits | Files at tip | Commits exclusive относительно остальных перечисленных ref |
|---|---|---|---:|---:|---:|
| `origin/pasha883` | `94166506b092bc46a03dfd338e85006726bd9097` | `67aa58a5e03b8e1b453fb6a3541094eff159059e` | 7 | 32 | 0, потому что весь ref является общим предком других веток |
| `origin/MAESTRO` | `94166506b092bc46a03dfd338e85006726bd9097` | `67aa58a5e03b8e1b453fb6a3541094eff159059e` | 7 | 32 | 0, точный alias `pasha883` |
| `origin/Taska-govna` | `d28136151f17161ecdf791dfc456ceea2f6e4fa0` | `4f852479d1cc8c39b241e98954cf5d35cc7f8edf` | 8 | 255 | 1 |
| `origin/agent/research-restorer-rl-pipeline` | `f4e72db849a7073d1dfc8d525d75eb101eb38130` | `696f4d906f7d2b11711f3a663509c2ec0e2f3a4b` | 12 | 66 | 5 |
| `origin/agent/ssim-scorer` | `f40e77baf9b643538020582aa2b96927c73ebf93` | `7bfb30278a58d701f4298c13e9e5a43841716d61` | 1 | 25 | 1, unrelated root |
| `origin/таска-говно` | `d6a82f82ceefa109ef706402712d03805bc9e880` | `cab0a4dfa63ba17747a2f071438d02619b02493c` | 8 | 1533 | 1 |

Все ref, кроме orphan `agent/ssim-scorer`, имеют merge-base
`94166506b092bc46a03dfd338e85006726bd9097` с `origin/pasha883`.

Всего в заданном наборе 15 различных commits: 7 общих foundational, 5 в
restorer/RL, по одному в двух архивных ветках и один orphan scorer commit.

### 2.2. Полный ledger уникальных commits

Общий фундамент `pasha883`/`MAESTRO`:

| Commit | Содержание |
|---|---|
| `dd205ed333bcd64e2e6a2a43928e865f10eda5a9` | root README, 1 файл |
| `d4d8e5820c7a4c26716d3394f2176bea3d76a0aa` | условие в `Task.txt`, 1 файл |
| `75541bca3972e601536f205a6e51ef5c33bf9560` | `.gitignore` и `baseline.ipynb`, 2 файла |
| `28cbb0c2edf6ad15692802bc110598f6bd127215` | первый полный solve+restore pipeline, 17 файлов |
| `bf5084cf84d1c3ba51a21d214da3470629f1bca6` | `FOR_AGENTS`, experiment log, PairwiseNet, diagnostics, 10 файлов |
| `1c2660a359dada4881a81f866e072e1a20dabfd8` | `solution.ipynb`, notebook builder, pair scorer in inference, 3 файла |
| `94166506b092bc46a03dfd338e85006726bd9097` | strategy/alternative ideas, score diagnostics, architecture/fixes, 15 файлов |

Exclusive commits `agent/research-restorer-rl-pipeline`:

| Commit | Содержание |
|---|---|
| `b3277a22139c9dfcaf6c6a9a4b0a7a28a89cdadf` | restoration, assembly и PPO research package, docs/scripts, 20 файлов |
| `71331de41d481388b095e8229fc14f9f1fb85f21` | project overview |
| `b576cd90a495f9cfc4b878cf506b98890ce35cc7` | final restorer+RL submission record и manifest updates |
| `3fc1b99dd0be490d0a4b4e524e2600e8f075a7dd` | 5-class relation pipeline и 4 committed metrics JSON, 17 файлов |
| `f4e72db849a7073d1dfc8d525d75eb101eb38130` | relation graph-greedy solver v2 и regression coverage, 6 файлов |

Остальные exclusive commits:

| Commit | Ref | Содержание |
|---|---|---|
| `d28136151f17161ecdf791dfc456ceea2f6e4fa0` | `origin/Taska-govna` | один snapshot «Archive puzzle research through E21»: 227 changed files, 92,097 insertions |
| `f40e77baf9b643538020582aa2b96927c73ebf93` | `origin/agent/ssim-scorer` | orphan Vite scorer и 18 reference PNG: 25 files |
| `d6a82f82ceefa109ef706402712d03805bc9e880` | `origin/таска-говно` | монолитный delivery snapshot: 1,562 changed files, 20,642,611 insertions |

`origin/MAESTRO` не содержит ни одного отдельного commit/tree/blob: это буквальный
alias `origin/pasha883`, и повторный отчёт для него создавал бы фиктивную работу.

## 3. Общий фундамент: `origin/pasha883` = `origin/MAESTRO`

### 3.1. Что реализовано

Задача интерпретирована корректно: вход — 480×480 RGB image, разрезанный на
24×24 = 576 upright tiles по 20×20, каждый tile независимо испорчен и
перемешан; metric — mean RGB SSIM.

Pipeline состоит из:

1. восстановления train permutation: degraded input tiles сопоставляются с
   известными clean target tiles по normalized coarse 5×5 descriptors и
   Hungarian assignment;
2. reverse engineering corruptor:
   affine → Gaussian noise `sigma 40–55` → 3×3 Gaussian blur → JPEG quality
   `35–50`; документация сообщает synthetic-to-real mismatch `ΔSSIM≈0.03`;
3. `CompatNet`: siamese directional edge embeddings для дешёвого all-pairs
   candidate retrieval;
4. `PairwiseNet`: seam cross-encoder по двум tiles, либо reranking top-K, либо
   полный N² scoring;
5. `solve.py`: greedy initialization + simulated annealing QAP;
6. `RestoreNet`: residual U-Net с MS-SSIM+L1;
7. production-friendly альтернативный tail `--nlm`;
8. `diag_compat.py`, `diag_scores.py`, placement/full SSIM evaluators, Kaggle
   builder и generated Colab notebook.

`baseline.ipynb` намного слабее: PairCNN учится на clean target pairs и затем
делает row-major greedy solve. Вертикальные соседи также подаются side-by-side,
то есть physical bottom/top seam не представлен в своей геометрии; вместе с
clean-only training это сильный domain/representation mismatch. В обоих
notebooks сохранено ноль execution outputs, поэтому они являются кодом и
нарративом, а не доказательством результата.

### 3.2. Измеренные результаты

| Измерение | Результат | Вывод |
|---|---:|---|
| shuffled input unchanged | SSIM `0.08–0.11` | тривиальный floor |
| perfect placement, no restoration | `0.43–0.50`, типичный ceiling `~0.447` | placement доминирует |
| stated target perfect placement + restoration | `~0.6–0.8` | ранняя оценка, не production result |
| CompatNet v1 | H@1 `~0.19` at step 2400 | recovered-real labels шумные: `~12%` misplaced tiles, `~25%` corrupted adjacencies |
| CompatNet v2 synthetic-only | R@1 `.16`, R@25 `.52`, R@50 `.63`, median rank `~20/575` | cheap embedding недостаточен |
| PairwiseNet train | 9,000 steps, reported sampled `val acc@48=.477` | поздний audit показал, что это не надёжный metric |
| full score true-neighbor | R@1 `~.20`, median rank `7–18/576`, best-buddy precision `~.48` | локальный scorer слишком слаб для component growth |
| actual solver snippet | `place_acc≈.0015`, solve SSIM `≈.106`, perfect-layout ceiling `≈.447` | раскладка практически случайна |
| classical MGC | best-buddy precision `~.05` | pixel continuity разрушена corruption |
| NLM on correctly placed image | `~.447 → ~.57` | сильный и дешёвый pixel tail при хорошем layout |
| learned RestoreNet at that stage | `0.4385` | хуже no-restore и NLM |

Поздний `origin/Taska-govna` audit обнаружил, что исторический `acc@48=.477`
фактически считался на 32 candidates, а random negatives могли включать anchor,
positive и duplicates. Этот metric не переносился на full-bag assembly и не
должен использоваться как baseline.

### 3.3. Что точно установлено

- Oracle scores дают 100% reconstruction: базовая механика solver не является
  первопричиной провала.
- Даже небольшие ошибки score matrix меняют optimum: неверная layout может иметь
  лучшую сумму pairwise energy, чем truth. Значит недостаточно просто увеличить
  число SA iterations.
- Seam-aware PairwiseNet v2 (GroupNorm, spatial flatten вместо global pooling,
  wider channels), harder negatives, больше real data и ensemble не вывели
  best-buddy precision из диапазона примерно `.41–.53`.
- Denoise-before-match и classical normalized boundary scores не помогли.
- Diffusion для pixel restoration не соответствует SSIM: hallucinated high
  frequency хуже безопасного smoothing.
- Сборка ограничена compatibility, но NLM стоит сохранять как tail, пока layout
  плох.

### 3.4. Незакрытые идеи из `pazzle_alt_ideas`

Документ предлагал DINO/CLIP/SAM features, low-frequency descriptors, joint
photometric normalization, set-to-grid Transformer + Sinkhorn, spectral layout,
row/column decomposition, global critic и RL. Почти все эти семейства позже
были реально проверены в двух архивных ветках и дали отрицательные результаты;
см. сводный реестр ниже. Поэтому их нельзя считать «новыми» только потому, что
в этой ранней ветке они обозначены как roadmap.

### 3.5. Артефакты и ограничения воспроизводимости

- В tree: 21 Python, 3 Markdown, 2 HTML, 2 PDF, 2 notebooks; нет blob >1 MiB.
- Checkpoints, recovered permutation cache, logs, Kaggle outputs и submissions
  лежали на `E:/pazzle_work`/Kaggle и не закоммичены.
- `FOR_AGENTS.md` сообщает, что private Kaggle notebook в тот момент содержал
  W&B key. Сам notebook/key игнорировался и в git-tip отсутствует, но workflow
  нельзя публично реплеить без очистки secrets.
- `solution.ipynb` — generated source-only notebook без outputs и без финального
  scored ZIP.

## 4. `origin/agent/research-restorer-rl-pipeline`

### 4.1. Pipeline

Отдельный package `research/codex_pipeline` реализует:

1. residual `FragmentRestorer`;
2. `EdgeMatcher` и absolute `PositionPrior`;
3. `PairRelationClassifier` с классами
   `not_adjacent/left/right/up/down`;
4. graph/position assignment и local swaps;
5. optional PPO swap policy;
6. objective guard, strict shape/checkpoint checks и atomic 700-PNG ZIP.

В tip 34 Python, 15 JSON и 8 Markdown. В git есть четыре relation metrics JSON,
но weights и примерно 185 MB submission archives намеренно исключены; их Kaggle
locations и SHA-256 перечислены в `MODEL_MANIFEST.md`.

Ключевые внешние артефакты, зафиксированные manifest-ом:

| Artifact | SHA-256 | Статус |
|---|---|---|
| `fragment_restorer_epoch8.pt` | `5db11a55da07d7db9bb51ac9b0f94efe410f330a272566310cc60f88e18b32fe` | selected restorer |
| `rl_swap_actor_critic_epoch1.pt` | `ec56aa1115cf02b3db8dcab094857e9fb13e57b42883bdcff28495550e58b036` | selected RL policy |
| `pair_relation_restorer_continued_best.pt` | `d399c742b64dcf0bcbb9029a32c0706e060fb839397df0b3f1fd9cb753090eb4` | selected relation model |
| `submission_pazzle_solver_audit_fixed.zip` | `617b39ec3983fb74db0761932f5961db5770f50c4d00cd5b6a588c5313fdc29e` | verified fallback |
| `submission_pazzle_solver_restorer_rl.zip` | `f21de3ef38996e9fa7e4f6c914593a2c40b68e799485169d48ed83535776f778` | verified 700-file candidate, no LB |

### 4.2. Restoration и RL

| Ветка | Измерение | Результат/статус |
|---|---|---|
| conditional DDPM, epoch 14 | визуально oversmooth; training/inference objective mismatch | сохранён как history, не выбран |
| `FragmentRestorer`, epoch 8 | tile MSE `.037895→.031750`, SSIM `.569723→.753088`, PSNR `14.214→14.983` | выбранный residual restorer, но может сглаживать/ошибаться |
| ранний heuristic end-to-end | SSIM `.152632` | исторический baseline |
| PPO epoch 1 | RL `.201791` vs heuristic `.197389`; adjacency `.055820` vs `.050045` | Pareto checkpoint selected |
| PPO epoch 4 | SSIM `.204003`, adjacency `.049026` | не выбран: adjacency хуже baseline |
| initial 4-image integration | `.178489` vs `.168155`, delta `+.010334` | очень маленькая validation |
| post-audit smoke | `.176371` vs `.168212`, `3/4` accepted | guard работает по surrogate objective |
| restorer + guarded RL smoke | `.179611` vs `.175991`, delta `+.003620` | один per-image true SSIM loss остаётся возможен |
| full inference v10 | first-two `.160895` vs `.156745`; accepted/rejected `425/277` including 2 validation rows; 700 test PNG, `~9024s` | ZIP сформирован, leaderboard score не записан |

Guard гарантирует неухудшение только observable solver objective, а не hidden
target SSIM. Поэтому небольшое отрицательное per-image SSIM возможно даже после
исправления unconditional RL replacement.

### 4.3. Solver audit: найденные и исправленные проблемы

Audit `solver_audit_2026-07-20.md` зафиксировал:

- optimizer не сохранял initial/best layout;
- submission ZIP мог включать stale PNG;
- RL candidate безусловно заменял baseline, хотя objective отличался;
- model/data/checkpoint resolver мог смешать несколько Kaggle datasets;
- hard-coded checkpoint epoch был неясен (позже закреплён как deliberate
  Pareto choice);
- empty/wrong-shape test set проходил слишком далеко;
- duplicated RL features и неверная geometry local-swap proposals;
- GPU checkpoint loading создавал memory spike;
- validation была мала, слаба и местами потенциально leaky;
- EdgeMatcher использовал лишь top-2 raw seams;
- portability/reproducibility зависели от Kaggle paths.

В tip исправлены best-so-far preservation, geometry-valid proposals,
same-objective RL guard, explicit roots, atomic ZIP, 700-name validation,
fallback validation и safe checkpoint loading. `test_solver_regressions.py`
содержит 8 regression tests, включая consistent-grid recovery, best-so-far,
RL guard, geometry, root confinement, complete model set, clean ZIP и population
std normalization. Тесты в этом read-only аудите не переисполнялись.

### 4.4. Relation classifier и последний solver

Первый relation model показывал clean accuracy `.8943`, macro-F1 `.8944`, noisy
accuracy `.8492`, но строгая source-disjoint проверка на 21 unseen images и
50,000 fixed pairs/condition обнаружила domain gap:

| Tile source | Accuracy | Macro-F1 | Binary adjacency accuracy |
|---|---:|---:|---:|
| clean target | `.8544` | `.8550` | `.9464` |
| raw damaged | `.27362` | `.21494` | `.31104` |
| residual-restored | `.37778` | `.37195` | `.48254` |

Fine-tune на frozen restorer outputs поднял accuracy до `.51592`, macro-F1 до
`.51640`. Продолжение на 4,000 source images, LR `2e-5`, clean replay weight
`.25` дало лишь `.52004/.52062`, сохранив clean accuracy `.84346`: одинаковый
fine-tune практически вышел на плато.

Solver v1 применял directional log-odds к raw-seam top-8 shortlist, но его
Kaggle run ещё выполнялся на момент записи. Solver v2 отключил mismatched legacy
RL, строил coordinate-consistent components, отбрасывал collisions/contradictory
cycles, anchored их PositionPrior и сравнивал greedy vs position-only init.
Commit содержит код и kernel metadata, но не завершённый validation report,
submission или leaderboard result. Статус идеи: **реализована, результат не
получен; не считать победителем**. Более поздние ORBIT/archive experiments дают
намного больше отрицательной информации о похожем graph packing.

## 5. `origin/agent/ssim-scorer`

### 5.1. Что это

Orphan root из одного commit: Vite/JSZip web app, 1 JS, 1 CSS, 2 package JSON,
README и 18 PNG в `public/reference/`. Он читает ZIP в браузере, сопоставляет
basename независимо от вложенной директории, декодирует PNG и считает valid
7×7 SSIM с sample covariance correction `49/48`, стандартными C1/C2 и средним
по RGB. Формула соответствует заявленным параметрам `skimage`.

Reference filenames:

`img_000013`, `000275`, `000313`, `000697`, `000809`, `000840`, `001737`,
`001786`, `001990`, `001997`, `002051`, `002127`, `002198`, `002647`,
`002775`, `002829`, `002948`, `002950`.

### 5.2. Критические caveats

- Это 18 чистых известных test answers, а не source-disjoint validation.
- Набор имён в точности совпадает с 18 verified test overrides, описанными в
  `origin/Taska-govna`.
- Использование mean SSIM на этих images для model/hyperparameter selection —
  прямая test leakage, даже если originals найдены в публичных источниках.
- Mean считается только по `status=ok`; missing/invalid samples исключаются, а
  не получают penalty. Пользователь может получить вводящий в заблуждение mean
  по удобному subset.
- При duplicate basename берётся первый encountered ZIP entry.
- В ветке нет parity regression suite против `skimage`, только собственная
  реализация и README claim.

Разрешённое применение: локальный forensic/UI scorer с явной маркировкой
test-label access. Запрещённое для честного исследования применение: validation,
selection, threshold tuning или сравнение generic solvers.

## 6. `origin/Taska-govna`: ORBIT / Rank96 архив до E21

### 6.1. Состав и reproducibility boundary

Один snapshot commit добавляет 227 files поверх pasha base; tip содержит 255
files: 213 Python, 33 Markdown, 2 notebooks, без committed images/checkpoints/
submission ZIP и без blob >1 MiB. Main reports лежат в root и
`autoresearch-runs/pazzle-solution-20260806`; `README.md` содержит лишь ownership
warning.

Большинство evidence paths указывает на `E:/pazzle_work`. Source, tests,
protocols и reports сохранены, но weights, score caches, frozen artifacts и ZIP
из branch alone не восстановить.

### 6.2. Frozen Rank96 production

Точный contract из `src/infer_rank96.py`:

`upright tiles → affinity r1/r3 top-64 union → raw rank_v2w64 logits → CPU
float32 dense_rd → corrected buddies(max_edges=96,min_margin=0,repair=0) →
upright assembly → OpenCV NLM(h=10)`.

Pinned checkpoint hashes:

- ranker `42685373b1a450a4cb3d7a9b22370dfcfaa2335e9e8ada609f21b7cc64abbfbc`;
- affinity primary `708565329c7661a965215d98e85f462a90930071f36a0f75b4813c0c5797ec4f`;
- affinity secondary `0fceafdb110bde59149fe1ad1e800a69d116041bc627af369aaecd60be53b6c8`.

Frozen evidence:

| Gate | Scenes | Solve delta 96 vs 512 | Final delta | Caveat |
|---|---:|---:|---:|---|
| calibration | 8 | `+.003889` | not used | budget selected здесь |
| reserved confirmation | 4 | `+.002947` | not used | 3/4 wins |
| immutable v1 | 24 | `+.001108` | `+.005104` | bootstrap CI crosses zero |
| untouched v2 | 24 | `+.001908` | `+.000681` | zero-overlap source groups; CI crosses zero |

Equal-weight mean двух 24-scene gates: solve `+.001508`, final `+.002893`.
Конфигурация retained по заранее объявленному positive-mean rule, а не как
conventionally significant result.

Final v1 archive: 700 files, SHA
`9a2eaf962507d11f2cad0caf59af40fe9755a6f092051c9d144a5f6aca10965f`,
external score `0.2161981413457065`, runtime `6798s`. Но состав — **682 Rank96
outputs + 18 exact verified source overrides**. Это одновременно сильнейший
documented score этой ветки и contaminated comparison: нельзя узнать generic
Rank96 leaderboard без overrides.

### 6.3. E1–E21: полный реестр

| ID | Идея | Результат | Статус |
|---|---|---|---|
| E1 | I11/I21 replay | edge R1 `+.009133`, neighbour `-.001359`, solve `-.000746`, final `-.002244` | reject |
| E2 | reciprocal rank transplant | calibration solve `+.003725`, trusted-pair precision `.265625` vs gate `.85`; confirmation sealed | reject |
| E3 | 9 RGB/Lab/MGC donors | edge R1 `.147–.160` ниже raw `.164515` | reject |
| E4 | two-side plaquette growth | at precision≥`.95`, coverage≤`.000868`; at coverage≥`.15`, precision≤`.110132` | reject |
| E5 | corruption-invariant scorer fine-tune | conditional on solver transfer, не запущен | queued, not failed |
| E6 | exact-SSIM restoration fine-tune | conditional on placement transfer, не запущен | queued, not failed |
| E7 | I21 packing transfer | edge R1 `+.013927`, solve `-.000129` | reject |
| E8 | buddies edge-budget sweep | budget 96 confirmation `+.002947`, 3/4 wins | opened immutable gate |
| E9 | Rank96 immutable gate v1 | solve `+.001108`, final `+.005104` | champion under fixed rule |
| E10 | second untouched gate | solve `+.001908`, final `+.000681` | keep Rank96 |
| E11 | label-free Lab selector 96/512 | final `+.000623`, solve `+.000225`; failed strict `>+.001` | reject |
| E12 | unattainable clean score before solve | solve `-.007070`, final `-.016292`, 1/8 wins, worst `-.041554` | kill denoise scoring |
| E13 | toroidal global origin | RR final `+.002975`, only 2/8 wins, worst `-.017817`; CC negative | origin hypothesis insufficient |
| E14 | CC192 clean local oracle | precision `.957`, component coverage `.473`, neighbour `+.14164`, но solve `-.008794`, final `-.014433` | local islands insufficient |
| E15 | CC96 seeds + two-direct-seam frame consensus | 3 eligible hypotheses total; supported coverage `.003689` vs `.15` | kill structure before decoder |
| E16 | exact clean render on same wrong RR96 board | final `-.015296`, 1/8 wins | kill faithful restoration until layout improves |
| E17 | clean CC192 rigid-island prerequisite | added96 precision `.93099`, pure coverage `.42578`, largest pure component 15 | positive prerequisite only |
| E18 | absolute-frame beam | hit 500,000 proposal cap on first scene, no board | kill exact beam |
| E19 | quotient out global translation | still hit 500,000 after 32 rounds, no board | global origin not sole cause |
| E20 | triangle-supported signed-potential DSU | pose coverage `.0360`, relative precision `.2637`, relation `.1325`, seam `.2323`, cycle ratio 0 | kill noisy-path composition |
| E21 | raw CC96-anchor top-8 candidate ceiling | 29,209 hypotheses, 616 true; oracle component only 22.75 tiles, coverage `.0395` | kill this candidate pool before training |

Особенно важное различие: E17 показал полезные clean-oracle islands, но E21
показал, что **production raw candidate pool, где emitters только nontrivial
components, не имеет достаточного oracle coverage**. Learned verifier не может
восстановить отсутствующие relations. Любой следующий verifier сначала должен
изменить candidate generation/all-tile emission и пройти label-only coverage
gate.

### 6.4. Ранние направления A–H

| Семейство | Что проверено | Итог |
|---|---|---|
| A: generator inversion/forensics | 7,000 unique permutations; 77 seed/PRNG schemes; slot lookup `.00346`; row `.0479`, col `.0407`, cell R25 `.0502`; PNG metadata/SIFT same-name tests | seed/filename inversion закрыт |
| B: frontier inpainting pointer | context4 R1 `.0234`, R5 `.0762`; context8 `.0215/.0898` | closed |
| C: JPEG/resampling phase | dirty и clean mod2/mod4 near chance | closed |
| D: GraphGRU/CNSD | exact graph procedural control может работать; noisy placement `.00694`, neighbour `.00543` | global decoder не создаёт signal |
| E: dirty↔clean tile identity | same-image R1 `.759/.772`, pooled 4,608 R1 `.724`; но clean halo 4-neighbour R1 `.029`, 8-neighbour `.054` | отличный source-retrieval asset, не assembly |
| F: radial/optical priors | near chance даже clean | closed |
| G: 4×4 macro-block | oracle tile→block R1 `.2465`, but balanced clustering purity `~.246`, no perfect blocks; Siamese top1 `.2203` | standalone closed; weak macro signal remains |
| H: global critic | learned heldout `.539` vs simple TV `.933`; bounded residual TV `+.0059`, zero target threshold successes | learned critic closed; TV residual лишь слабый feature |

### 6.5. I1–I21: prior neural/global experiments

| ID | Проверка | Результат/решение |
|---|---|---|
| I1 | structural seam auxiliary | R1 `.2715→.2721`, R5 `.5078→.4948`, reciprocal `.5846→.5389`; closed |
| I2 | TTA pseudo edges | conservative R1 unchanged `.252`, reciprocal `+.0042`, pseudo precision `.872`; no solve gain |
| I3 | 4×4 relative flow | mechanics/equivariance pass, train overfit 100%; heldout placement `.0723`, neighbour `.1061`; closed |
| I4 | posterior seam/inpainting | deterministic edge L1 `.0634`, oracle best4 `.0433`; R1 unchanged `.4063`, R5 `+.0313`, Brier only `.32%`; residual follow-up worsened calibration |
| I5 | consensus islands | best precision `.846` at coverage `.087`; dual `.848/.072`; too sparse |
| I6 | balanced partition flow | purity `.2387` before/after, no perfect groups |
| I7 | edge confidence calibration | at 2% coverage precision `.9615`, at 5% `.8906`, at 15% `.6354`; useful seeds only |
| I8–I10 | growth/reciprocal variants | max precision `.9135` coverage `.130`; reciprocal `.933/.135`; fails broad safe growth |
| I11 | corrected shuffled-ID boundary bug | neighbour `.1386→.1647` on six scenes; placement remains near zero |
| I12 | multi-context | with 1/2/3/4 true neighbours known, R1 `.1976/.2995/.3852/.4528`; real seeds cover only ~10.5%; plaquette/path/SA/population fail |
| I13 | full-row LambdaRank | candidate conditional R1 `.2695→.2930`, all-true `.1886→.2051`; buddies `.1686`, no assembly gain |
| I14 | component beam | `.1458` vs deterministic `~.153`; reject |
| I15 | proposal refiner | preserves `.1726`, then degrades |
| I16 | genetic solver | no improvement from `.1522` |
| I17 | GNN rank | R1 `.1886→.1987`, buddies `.163→.1646`; selected blend transfers worse `.1647→.162` |
| I18 | all-pairs Siamese | R1 `.0800`, buddies `.0539`; insufficient even без top-K bottleneck |
| I19 | context repair/path solver | repairs hurt; horizontal `.2011`, vertical `.0091`, overall `.1051` |
| I20 | symbolic/PuzLM tokens | agreement `.4069`; neighbour R1/R5/R64 `.0682/.1191/.3943`, median rank 106; no fusion gain. Rotation audit: upright 97.8%/98.61% high-confidence |
| I21 | positional DDPM/directional head | full-board DDPM overfits: heldout neighbour `~1.27%`; directional fusion fresh6 improves neighbour `.14417→.15625`, but selected among 28 configs on same six scenes, no end-to-end SSIM |

I21 directional fusion — единственный небольшой не полностью закрытый signal в
этом блоке, но он требует новой source-disjoint frozen confirmation и
end-to-end comparison; текущие числа нельзя считать promotion.

### 6.6. Candidate ranker retained asset

`rank_v2w64` достиг:

- confidence `.30`: precision `.745`, coverage `.092`;
- `.50`: `.880/.067`;
- `.70`: `.954/.042`, примерно 49 seed edges/image;
- affinity union direct-neighbour recall `~.67–.69` при фактических `~81`
  candidates/row.

Это хороший seed/candidate generator, но pose-sync neighbour accuracy лишь
`.008`. Не повторять capacity scaling той же listwise architecture; использовать
его как frozen input/control.

### 6.7. Public-source retrieval и leakage boundary

Forensics собрал 19,679 public photos из T-Bank/CU/Telegram и нашёл 218 exact
train sources и 18 exact test sources. Acceptance: 10×10 Hungarian tile
assignment + минимум 5 spatially aligned SIFT matches + identity fraction
`≥.35`. Wider crawl новых test matches не добавил; Wfolio/VK были JS/auth
blocked.

Это сильный capability для image identity/source acquisition, но 18 clean test
crops являются test answers. Их использование в submitted output может зависеть
от правил конкурса; для научной оценки это leakage. Generic baseline, source
override layer и scorer по этим 18 samples должны храниться/оцениваться отдельно.

## 7. `origin/таска-говно`: complete research delivery snapshot

### 7.1. Tree и manifest

Tip tree содержит 1,533 files:

- `history/`: 1,023 paths, из них 1,022 decision evidence;
- `source/`: 494 paths = 137 job definitions, 111 scripts, 97 Kaggle jobs,
  72 tests, 57 source modules, 20 configs;
- `project/`: 10 docs/config files;
- 98 Markdown, 365 Python, 1,015 JSON, 47 TXT;
- `promoted_assets/`: только `denoiser_release_SHA256SUMS.txt`.

`project/COMPLETE_ML_TASK_HISTORY.md` — 707-line authoritative synthesis;
`MANIFEST.json` перечисляет 1,534 release entries и 18 дополнительных
`hash_only_artifacts`.

Критическое расхождение: root README и history утверждают, что текущий delivery
содержит weights и `submission/submission.zip`, но git-tip физически не содержит
шесть manifest payloads:

- `submission/submission.zip`;
- `promoted_assets/hbt_d320_denoised_rgb_sobel.pt`;
- `promoted_assets/seam_denoiser_gpu.pt`;
- `promoted_assets/selected_tilenaf_synth_50k.pt`;
- `masked_gap/masked_gap_gate.pt`;
- `masked_gap/masked_gap_gate_code.zip`.

Tree, напротив, имеет пять wrapper/VCS paths, которых нет среди archive entries:
`.DS_Store`, `.gitignore`, `MANIFEST.json`, `SHA256SUMS.txt`,
`history/.DS_Store`. Следовательно, manifest описывает внешний release ZIP, а
не materialized git tree. Hashes/provenance полезны, но exact inference из этой
ветки без внешних assets невозможен.

### 7.2. Лучший production record

Pipeline:

1. selected TileNAF tile restoration;
2. `0.5` blend с seam-trained TileNAF renderer;
3. C1/HBT directional rank fusion;
4. soft-cycle seed;
5. directional QAP weight 4, boundary `.05`, 25 iterations, 2 restarts;
6. input-only RGB seam-graph harmonization;
7. bounded luma gain.

Scores/provenance:

- exact RGB-only leaderboard: `0.2167844489529071`;
- pre-harmonization QAP render: только user-reported rounded `.203`, exact
  provenance отсутствует;
- luma archive: user-reported rounded `.218`; exact luma score/delta неизвестен;
- canonical luma ZIP hash
  `099d1c5fe69cda8519a4f19750cb3a481ac87999c294a35e19691a849d4c6096`;
- при обычном rounding inferred lower bound к RGB-only `+.000715551`, но это
  inference, не platform measurement.

Pinned external hashes из frozen production record:

- selected TileNAF:
  `77a2d8607c9bc6b80e7aa99ed03329c1fae6bf94ee1d9a654241ab93a05cc734`;
- HBT:
  `c2589bff58573d592227d522954a7ed6c22f6fec7ecbbea2bf0c755d7a720787`;
- seam-renderer checkpoint:
  `f973c7e606a112020c527bb72277b82586df915edc829a22305e587b35aec1b9`;
- decoded final pixel stream:
  `b8fd646bc3cc2071853988cc36d13108c30c5aa7efc0097f6dc4ef91cbc4cc98`.

В отличие от Rank96 archive, production record явно утверждает отсутствие
filename/target leakage и отделяет candidate oracle как
`safe_for_submission=false`. Но final ZIP/weights в git всё равно отсутствуют.

### 7.3. Restoration: что сработало и что закрыто

| Идея | Результат | Статус |
|---|---|---|
| selected TileNAF, 50k steps | synthetic raw `.56302→.80828`; Kornia `.52435→.82267`; libjpeg `.51295→.81928`; boundary MAE `19.19→14.57` | promoted |
| sealed 350-source real-pair gate | raw `.67570`, NLM `.72041`, legacy `.77100`, TileNAF `.81098` | strong source-disjoint win |
| conservative real fine-tune | `+.00183`, ниже prereg `+.003`; rollback bitwise | reject |
| ordered 5×5 context | ordered SSIM `+.0041–.0055`, but tile SSIM/boundary worse | stop |
| seam-trained renderer blend | `+.000502`, CI positive | promoted only as renderer |
| RGB seam-graph harmonization | `+.0118…+.0131`, 32/32 wins на двух panels | главный pixel-side win |
| bounded luma | confirmation `+.001721901`, 32/32 wins | promoted |
| contextual postassembly refiner | oracle-good `+.0065`, actual QAP `-.00114/-.00127` | reject |
| exact clean content on wrong layout | ухудшает относительно NLM | совпадает с Taska E16; не повторять |
| masked-gap clean central-gap model | MRR `-.00121` vs equal control; vs W4 MRR `-.0442`, R1 `-.01664`, 0/4 wins | failed before holdout/QAP |

Вывод: сохранять TileNAF/RGB/luma tail, но не тратить compute на новую fidelity
restoration до заметного улучшения placement.

### 7.4. Compatibility и QAP

Ранние/локальные модели:

| Model | Result | Решение |
|---|---:|---|
| C1 classical real64 | SSIM `.191870` | strong historical baseline |
| L0 seam CNN | R1 `.152231` | retired |
| L1 side embedding | R1 `.219486`, R32 `.698299`, real16 `.172663` | useful feature, not promoted |
| L1-v2 | R1 `.203167` | retired |
| T0 absolute position | accuracy `.002658` | chance, retired |
| X0 reranker | R1 `.200153`, candidate recall `.761096` | not promoted |
| L1+X0+T0 | real64 `.188669 < .191870` | overfit/retired |
| real pseudo-label L1 | exact R1 `~.194 < .219`; real16 `.170359` | self-confirming degradation |
| G0 global matcher | R1 `.217165 < .224072` HBT | retired |
| HBT denoised RGB+Sobel | R1 `.223845` | retained feature |
| HBT RGB-only/raw RGB+Sobel | `.215636/.179008` | weaker |
| Sobel-only/binary-only | `~.034/.015` and `~.0075` | destructive |

Promoted directional QAP, real16:

- soft-cycle seed `.165431140`;
- ordinary QAP `.182329628`;
- heavy 40×4 `.181305114`;
- boundary-QAP `.182819915`;
- selected vs seed `+.017388775`, 16/16 wins, CI
  `[+.012173,+.023101]`.

Сам boundary term не доказан отдельно: increment к ordinary лишь `+.000490`, CI
пересекает zero. Доказана семья/configuration целиком, не причинность каждого
knob.

### 7.5. Search/global/neural branches, которые не надо повторять

Search по той же энергии:

- RL top-k: `.1685–.1728`;
- LNS 64/192: `.171237/.169914`;
- cross-view: `.175156`;
- annealing: `.170495`;
- protected annealing: no-op или negative;
- line-QAP: `.170975`;
- CP-SAT: тот же layout;
- oracle выбора из QAP+RL/LNS/cross/anneal pool: лишь `.188504`.

Значит дополнительный budget/move class на прежнем objective не закрывает gap.

Global/context:

| Branch | Result | Status |
|---|---|---|
| context reorganization | wrong positions `4597→4597`, no SSIM change | scientific zero |
| 2×2 hyperedge | AP `.01593`, precision 6/344, adjacency `.061→.034`, SSIM `.18282→.16122` | retired |
| frozen MAE energy | broad Spearman `.6518`, gain only `+.000730` | not promoted |
| MAE population | competitive Spearman `.0574`, pair `.5202`, SSIM `-.000813` | retired |
| DINO 4×4 | cell accuracy `.0447` | retired |
| LaMa | три infrastructure attempts, no target metric | inconclusive, не научный fail |
| ViT-Sinkhorn | `.175902 < .201440`; holdout `.196005 < .222600`, position chance | retired |
| Pair Transformer | R1 `.1884/.1870 < .1993`, затем nonfinite | retired |
| layout energy | AUC `.9605`, но adjacency `-.00286`; hybrid best `+.000117` | retired |
| positional diffusion | SSIM `-.0598/-.0639`, adjacency `~-.117` | decisive fail |
| HBT continuation | R1 `.223845→.229676`, но prereg gates fail | no promotion |
| dense residual all-pairs | R1 `.182476→.113423` | retired |
| QAP weight 1 | fresh64 `+.001375`, но fails `+.005`/40-win gates | confirmed small, not promoted |

### 7.6. Candidate oracle и structured-solver chain

Самый ценный diagnostic:

- C1/HBT top-32 union true-edge recall `.729789`;
- median true-edge largest component `545.5/576`;
- production-like QAP SSIM `.193591`, adjacency `.062358`;
- truth-filtered SSIM `.627267`, adjacency `.385134`;
- truth-assisted translation ceiling SSIM `.709094`, position `.963677`,
  adjacency `.939750`.

Oracle v1 invalid из-за shape-descriptor bug, v2 stranded без crash-safe journal,
v3 recovery-only; только v4 допустим как exploratory ceiling, всё равно
`safe_for_submission=false`.

Попытки превратить ceiling в input-only solver:

| Attempt | Result | Status |
|---|---|---|
| full-union HGB | AP `.1742`, AUC `.6801`, QAP `+.000030` | not promoted |
| p80 verifier | AP `.26865`, AUC `.82586`, component size 2.95, SSIM `.19059 < QAP` | retired standalone |
| HGB cycle | worst-panel AP `+.000898 < .005` | retired |
| component sync | macro SSIM `-.002084` | retired |
| trust repair | adjacency `~+.001`, SSIM `-.0000175` | retired |
| robust translation sync + OT | SSIM `-.0142`, adjacency `-.045…-.053` | retired |
| PT anneal | `+0.000008` | no value |
| D4/group switching/GNC | slow prerequisite or synthetic fixture failure | retired/gated |
| exact path cover | near-zero adjacency delta, expensive | retired |
| LongSync-4 | retrieval/AP negative, 0/8 wins | retired |
| dual LambdaRank | QAP `.206636→.201832`, 0/8 | retired |
| ContinuationNet-0 | R1 `.05967` vs W4 `.16950` | retired |
| binary edge verifier | remote SIGKILL before valid report/checkpoint | inconclusive infra, not proven fail |
| GANzzle route | DINO prerequisite failed, not run | gated out |
| TileNAF latent edge | vs HBT about `-.01`; vs W4 `+.00776/+.00793`, but gate required `+.008` and coverage `.75` vs `.744` | failed gate, no QAP |

Macro-block counterfactual также показывает, почему локальная coherence важна,
но недостаточна: random tiles `.12147`, exact 2×2 blocks `.1564`, 4×4 `.18982`,
6×6 `.22246`; cyclic 4×4 shift `.21599`; одна перестановка двух 4×4 blocks на
почти правильном board даёт `.95099`. Этот diagnostic не имеет frozen report,
поэтому использовать только качественно.

## 8. Сводный реестр идей: проверено или нет

| Идея | Где проверялась | Итог |
|---|---|---|
| обычный seam PairCNN / Siamese | pasha Pairwise/Compat; Taska I18; Russian L0/L1/HBT/PairTransformer | pair-only precision недостаточна; не повторять без новой информации/context |
| deeper/wider/harder negatives той же модели | pasha v2, Taska rank_v2w64, Russian continuations | локальный metric иногда растёт, assembly не переносится |
| denoise before compatibility | pasha, Taska E12/E14, Russian scorer/restorer comparisons | даже clean oracle может ухудшить final layout; закрыто |
| faithful learned restoration на wrong board | pasha RestoreNet, Taska E16, Russian contextual refiner | хуже NLM/smoothing; закрыто до placement win |
| safe postassembly color correction | Russian RGB harmonizer + luma | устойчиво работает; сохранить |
| more SA/RL/LNS/GA/CP-SAT on same energy | pasha SA; agent PPO; Taska I14–I16; Russian search matrix | не меняет неправильный optimum; закрыто |
| set-to-grid/Sinkhorn/absolute position | pasha idea; Taska GraphGRU/CNSD/I3/I21; Russian T0/ViT-Sinkhorn | heldout near chance; закрыто в tested formulations |
| DINO/foundation semantic layout | Taska G/H/I20; Russian DINO | weak/nearly chance; не повторять тот же probe |
| global critic/MAE/layout energy | Taska H; Russian MAE/layout energy | может различать broad corruptions, не competitive layouts |
| cycles/consensus/pose sync over noisy edges | Taska I5/I8–10/E15–E21; Russian structured chain | false edges corroborate false poses либо coverage слишком мала |
| source image identity | Taska E/source forensics | сильный R1 `~.72–.77`; полезен для retrieval, не adjacency |
| exact public test source override | Taska + SSIM scorer | leakage/compliance route; изолировать от honest ML |
| candidate graph + truth verifier ceiling | Russian oracle v4 | огромный headroom `.1936→.6273/.7091`; главный открытый lever |
| multi-neighbour context | Taska I12 | oracle context даёт R1 до `.4528`; promising, но реальные seeds слишком редки |
| relation graph greedy v2 | agent branch | код есть, завершённого результата нет; superseded evidence делает низким приоритетом |
| LaMa / binary verifier | Russian archive | infrastructure-inconclusive, не следует называть научно опровергнутыми |
| corruption-invariant scorer / exact-SSIM fine-tune E5/E6 | Taska | conditional queued, не запускались; prerequisites потом не прошли |

## 9. Противоречия, leakage и несопоставимость metrics

1. **Alias:** `MAESTRO` не отдельная работа; любые два результата от этих ref
   идентичны byte-for-byte.
2. **Pasha `val acc@48=.477`:** поздний audit доказал фактические 32 candidates
   и contaminated random negatives. Не использовать.
3. **Pasha optimistic restoration range:** `~.6–.8` — target estimate; реально
   ранний learned RestoreNet `0.4385`, NLM `~.57` только при correct placement.
4. **Taska Rank96 leaderboard:** `.2161981413` включает 18 exact test source
   overrides; это не generic solver score.
5. **SSIM scorer:** содержит эти 18 clean test labels и усредняет только найденные
   valid files. Любой gain на нём является leaked subset result.
6. **Russian `.218`:** только rounded user observation. Exact RGB `.2167844489`
   известен, exact luma нет; разность `.00121555` — разность отображений, не
   точный platform delta.
7. **Russian bundle manifest:** заявляет полный release, но шесть critical
   payloads отсутствуют в git. Hash manifest не заменяет runnable weights/ZIP.
8. **Validation scales:** 4-image smoke, 8-scene calibration, 16/24/32/48/64
   source panels и leaderboard — разные protocols. Их абсолютные SSIM нельзя
   ранжировать без одинакового split/corruption/render tail.
9. **Oracle metrics:** clean tiles, truth-filtered graph, exact source crops,
   correct-neighbour context и target-selected candidate pool являются ceilings,
   не deployable methods.
10. **Negative vs inconclusive:** LaMa, binary edge verifier, queued E5/E6 и
    relation-greedy v2 не получили валидной финальной метрики; их нельзя
    записывать в окончательно failed family.
11. **Boundary-QAP attribution:** full configuration доказан, но отдельный
    boundary term имеет CI через zero.
12. **Over-selection:** Taska I21 alpha/config выбран среди 28 вариантов на тех
    же 6 scenes; required fresh confirmation отсутствует.

## 10. Наиболее перспективные направления

### P0. Восстановить сильнейший честный baseline среди legacy refs и отделить его от leakage

Сначала необходимо materialize или переобучить assets русского pipeline:
TileNAF, HBT/C1, QAP, RGB harmonizer, luma. Это единственная линия с exact generic
RGB leaderboard `.2167844489` и rounded `.218` после luma. Сделать отдельные
artifacts:

- `generic_only` без source overrides/test references;
- optional `public_source_overlay`, только если правила явно разрешают;
- frozen source-disjoint validation и exact ZIP provenance.

Без этого новый experiment будет сравниваться с более слабым/contaminated
Rank96 и может дать ложную «победу».

### P1. Изменить candidate pool, затем учить multi-tile relation verifier

Наиболее убедительная совокупность фактов:

- union содержит `72.98%` true edges;
- truth filter даёт SSIM `.6273`, translation ceiling `.7091`;
- 4 correct neighbours поднимают local R1 до `.4528`;
- production adjacency лишь `~.06`;
- E21 current nontrivial-emitter pool имеет oracle coverage только `.0395`.

Следующий bounded experiment:

1. сформировать all-tile-emitter union, включая singleton→component и
   component→singleton claims;
2. до ML посчитать label-only oracle connected coverage и translation ceiling;
3. fail-fast, если graph не содержит достаточной структуры;
4. обучать source-disjoint verifier не на одиночном seam, а на
   `island + candidate tile/component + several boundary/context views`;
5. выход — calibrated relative offset/relation confidence, а не ещё один raw
   seam logit;
6. до packing проверить precision при фиксированной useful coverage;
7. только затем запускать global assignment/QAP и end-to-end SSIM.

Это отличается от проваленных HGB-cycle/triangle/path methods: модель должна
видеть pixels/semantics нескольких тайлов непосредственно, а не составлять
новое правило из тех же noisy scalar edge scores.

### P2. Проверить complementarity small signals на frozen end-to-end gate

Кандидаты только как дополнительные features, не standalone solvers:

- Taska I21 directional spatial fusion (`neighbour +.0121` on fresh6, пока
  over-selected);
- high-precision Rank96 seeds (`p=.954`, 49/image);
- HBT/C1 disagreement/rank/margin;
- simple TV/component health, которые иногда лучше learned global critic.

Нужна одна predeclared source-disjoint confirmation без alpha/config resweep.
Если нет end-to-end delta, ветки закрыть.

### P3. Pixel tail сохранять неизменным до placement improvement

TileNAF + RGB harmonization + bounded luma — доказанный production tail.
NLM остаётся сильной fallback на плохих layouts. Новый renderer оправдан только
после фиксированного улучшения layout и paired test на тех же boards; иначе
faithful detail легко уменьшает SSIM.

### P4. Source retrieval — отдельный compliance track

Dirty↔clean identity embeddings имеют R1 `~.72–.77`, поэтому легитимный путь —
поиск дополнительных train/source data и dedup grouping. Но exact test crops и
18-reference scorer не должны участвовать в model selection. До любой подачи
нужно получить явное толкование правил конкурса.

## 11. Рекомендуемый protocol следующего эксперимента

1. Зафиксировать `generic_only` baseline, source groups, corruption seeds,
   checkpoint hashes и renderer tail.
2. Не открывать 18 reference images и не смешивать их со split selection.
3. Сделать candidate-availability oracle до обучения; отдельно измерить recall,
   connected coverage, component sizes и legal-origin coverage.
4. Пререгистрировать один verifier architecture, один calibration rule и один
   end-to-end gate; никаких post-hoc threshold/alpha sweeps на confirmation.
5. Минимальные метрики: candidate R@1/R@5, precision at fixed coverage,
   neighbour/placement accuracy, solve-only SSIM и final SSIM с **одинаковым**
   TileNAF/RGB/luma tail.
6. Сравнить против обоих controls: strongest generic QAP pipeline и Rank96.
7. Сохранять report JSON, split IDs, exact command, environment, weights и
   final ZIP внутри доступного artifact store; manifest должен описывать
   materialized tree, а не внешний невыданный bundle.

## 12. Что сохранять как reusable assets

| Asset | Откуда | Назначение | Доступность в git |
|---|---|---|---|
| synthetic corruptor + recovered-perm tooling | pasha common | train supervision/data generation | source есть, caches нет |
| `diag_scores`, oracle solver controls | pasha common | scorer/solver decomposition | source есть |
| corrected solver safety fixes + 8 tests | agent branch | reliability/ZIP/checkpoints | source/tests есть |
| 5-class relation metrics JSON | agent branch | domain-gap reference | есть |
| Rank96 contract and tests | Taska | deterministic fallback/control | source/tests есть, weights нет |
| rank_v2w64 high-precision seeds | Taska | candidate seed feature | code/report есть, checkpoint external |
| source-group/dedup/forensics tooling | Taska | leakage-safe splits and retrieval | source есть; crawled assets external |
| TileNAF/HBT/QAP/harmonizer code | Russian archive | strongest generic baseline | canonical source/config/tests есть, weights absent |
| candidate oracle v4 reports | Russian archive | headroom/candidate diagnostics | reports/JSON есть, oracle-only |
| 18-reference SSIM app | scorer branch | forensic UI only | полностью есть, leaky |

## 13. Финальное решение по заданным ref

Каждый перечисленный ref и каждый различный reachable commit учтён. Для alias
`MAESTRO` отдельно доказана идентичность tip/tree. Для обеих snapshot branches
зафиксирована грань между committed source/evidence и внешними отсутствующими
assets. Основные experiment families, их protocols, численные outcomes,
failures, queued/inconclusive статусы и leakage paths сведены выше.

Главный практический вывод для продолжения: брать generic TileNAF + QAP +
RGB/luma как baseline, не повторять локальные scorer/search/restoration dead
ends и тратить следующий исследовательский budget на **all-emitter candidate
availability → source-disjoint multi-tile relation verification → frozen
end-to-end packing gate**.

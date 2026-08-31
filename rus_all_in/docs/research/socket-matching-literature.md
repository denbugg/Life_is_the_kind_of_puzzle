# Contextual socket matching для сборки пазла 24×24

Дата обзора: 2026-08-30.

Это literature-to-experiment memo для следующего поколения layout-модели. Он не
является экспериментальным результатом проекта. Главный предлагаемый эксперимент
ниже — **SocketGlue**: contextual matching всех правых/левых и нижних/верхних
«сокетов» изображения с partial optimal transport, после которого раскладка
декодируется как quadratic assignment. Остальные направления приведены как
экспериментально отличающиеся альтернативы, а не как варианты «сделать тот же
pairwise CNN ещё больше».

## 1. Ограничения задачи, которые определяют модель

- В одной сцене ровно 576 неповёрнутых фрагментов 20×20, исходная сетка — 24×24.
- Для горизонтального направления существуют 552 истинных внутренних ребра и по
  24 незамкнутых сокета на левой и правой границах. Для вертикального направления
  числа те же.
- Итоговая layout-часть обязана быть строгой биекцией: каждый исходный tile
  используется ровно один раз и без геометрического искажения. Matcher-only
  представления могут быть денойзнуты, но это не разрешает подменять ими пиксели
  финальной раскладки.
- Основная метрика для выбора layout-модели — **exact absolute position**.
  Adjacency и translation-aligned placement полезны только как диагностические
  метрики: хороший edge recall сам по себе не гарантирует правильную абсолютную
  раскладку.
- Шум, blur, JPEG и независимые фотометрические искажения делают raw seam distance
  слабым сигналом. При этом слишком сильный денойзер способен стереть или
  выдумать единственный полезный сигнал на границе 20×20 tile.

Отсюда следуют два разных ограничения, которые нельзя смешивать:

1. **Directional socket matching:** правый сосед должен быть выбран один-к-одному
   среди левых сокетов, нижний — среди верхних, с известным числом unmatched
   border sockets.
2. **Tile-to-grid placement:** набор хороших ребер всё ещё надо превратить в одну
   абсолютную перестановку 576 tiles. Это quadratic, а не обычная linear
   assignment задача.

## 2. Что уже известно из литературы

### 2.1. Classical compatibility и global optimization

Классические методы особенно важны здесь не как финальный seam metric, а как
описание правильной структуры decoder-а.

- Gallagher ввёл Mahalanobis Gradient Compatibility (MGC) и global assembly на
  основе локальных границ: [Jigsaw Puzzles with Pieces of Unknown Orientation,
  CVPR 2012](https://chenlab.ece.cornell.edu/people/Andy/Andy_files/Gallagher_cvpr2012_puzzleAssembly.pdf).
  MGC остаётся полезным дешёвым каналом и hard-negative generator, но на 20×20
  независимо повреждённых tiles его цветовая/градиентная стационарность нарушена.
- Son et al. показали ценность consensus нескольких кандидатов именно на малых
  фрагментах: [Solving Small-Piece Jigsaw Puzzles by Growing Consensus,
  CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/html/Son_Solving_Small-Piece_Jigsaw_CVPR_2016_paper.html).
  Это поддерживает multi-start/consensus decoding, но не устраняет каскадную
  ошибку greedy growth.
- Yu, Russell и Agapito формулируют сборку как linear program с глобальными
  ограничениями: [Solving Jigsaw Puzzles with Linear Programming,
  BMVC 2016](https://www.bmva-archive.org.uk/bmvc/2016/papers/paper139/index.html).
- Vardi et al. используют несколько фаз relaxation, уменьшая влияние ранних
  неверных решений: [Multi-Phase Relaxation Labeling for Square Jigsaw Puzzle
  Solving](https://arxiv.org/abs/2303.14793).
- Общая модель пазла как графа и ограничения placement систематизированы в
  [A Fully Automated Greedy Square Jigsaw Puzzle
  Solver](https://pmc.ncbi.nlm.nih.gov/articles/PMC4401723/).
- Если оптимизировать одновременно correspondence и структуру, естественно
  возникает quadratic constraint: [Deep Graph Matching under Quadratic
  Constraint, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Gao_Deep_Graph_Matching_Under_Quadratic_Constraint_CVPR_2021_paper.html).

Практический вывод: улучшенный learned compatibility нужно подавать не только в
greedy/Kruskal growth. Нужен decoder, который видит целиком биекцию и две
решётчатые системы соседства.

### 2.2. Deep pairwise matching и boundary embeddings

- SuperGlue сочетает contextual message passing, self/cross attention и
  differentiable optimal matching с dustbin: [SuperGlue: Learning Feature
  Matching with Graph Neural Networks,
  CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Sarlin_SuperGlue_Learning_Feature_Matching_With_Graph_Neural_Networks_CVPR_2020_paper.html).
  Его ключевая для нас идея — оценивать match не изолированно, а после обмена
  информацией внутри и между двумя множествами кандидатов.
- Bridger et al. учат совместимость даже при утраченной полосе между кусками:
  [Solving Jigsaw Puzzles with Eroded Boundaries,
  CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Bridger_Solving_Jigsaw_Puzzles_With_Eroded_Boundaries_CVPR_2020_paper.pdf).
  Это подтверждает, что matcher не обязан опираться только на один raw pixel seam.
- HardNet показывает практичность batch-hard обучения patch descriptor-ов:
  [Working Hard to Know Your Neighbor's Margins,
  NeurIPS 2017](https://papers.nips.cc/paper/2017/hash/831caa1b600f852b7844499430ecac17-Abstract.html).
  Для пазла наиболее ценные negatives — не случайные tiles, а визуально похожие
  границы из той же сцены и кандидаты текущего solver-а.
- Предсказание относительного положения patches как self-supervised objective
  исследовано Doersch et al.: [Unsupervised Visual Representation Learning by
  Context Prediction,
  ICCV 2015](https://openaccess.thecvf.com/content_iccv_2015/html/Doersch_Unsupervised_Visual_Representation_ICCV_2015_paper.html).
  Здесь это auxiliary loss, а не готовый solver.
- Census transform сравнивает локальный порядок яркости и поэтому менее
  чувствителен к монотонным фотометрическим изменениям: [Non-parametric Local
  Transforms for Computing Visual Correspondence,
  ECCV 1994](https://mlanthology.org/eccv/1994/zabih1994eccv-non/).

Для 20×20 tiles boundary encoder не должен сразу схлопывать сторону в один
вектор. Полезно сохранить последовательность вдоль границы: например, 20
позиционных токенов, каждый из raw, denoised, residual, Sobel/Laplacian и
census/rank каналов. В отличие от чистой фотометрической инвариантности, такое
представление сохраняет касательные контуры и их порядок.

### 2.3. Permutation-equivariant models, Sinkhorn и optimal transport

- Set Transformer даёт permutation-invariant/equivariant attention над
  множеством: [Set Transformer,
  ICML 2019](https://proceedings.mlr.press/v97/lee19d.html).
- DeepPermNet учит распределение над перестановками изображения:
  [DeepPermNet: Visual Permutation Learning,
  CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Santa_Cruz_DeepPermNet_Visual_Permutation_CVPR_2017_paper.html).
- Gumbel-Sinkhorn позволяет дифференцируемо приближать permutation matrices:
  [Learning Latent Permutations with Gumbel-Sinkhorn
  Networks](https://arxiv.org/abs/1802.08665).
- Энтропийно-регуляризованный optimal transport и Sinkhorn iterations описаны в
  [Sinkhorn Distances: Lightspeed Computation of Optimal Transport,
  NeurIPS 2013](https://proceedings.neurips.cc/paper_files/paper/2013/hash/af21d0c97db2e27e13572cbf59eb343d-Abstract.html).
- Jigsaw-specific fully contextual ViT доступен с авторским кодом:
  [FCViT](https://github.com/HiMyNameIsDavidKim/fcvit).

Критическая граница применимости: Sinkhorn решает **линейную** задачу
один-к-одному для уже известных tile-to-slot scores. В нашем случае основной
сигнал — «tile `i` стоит слева от tile `j`», то есть стоимость placement содержит
пары assignments и является quadratic. Поэтому standalone Sinkhorn по слабым
unary logits закономерно может быть почти бесполезен; Sinkhorn полезен внутри
socket matcher-а и как relaxation более полного quadratic decoder-а.

### 2.4. Diffusion и energy-based permutation models

- JPDVT применяет diffusion ViT к masked jigsaw formulation:
  [Solving Masked Jigsaw Puzzles with Diffusion Vision Transformers,
  CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_Solving_Masked_Jigsaw_Puzzles_with_Diffusion_Vision_Transformers_CVPR_2024_paper.pdf).
- PuzzleFlow формулирует сборку как discrete flow matching над
  перестановками: ViT-признаки tiles объединяются с текущими
  position/time embeddings, а модель итеративно предсказывает новую
  строгую раскладку: [The Missing GAP: From Solving Square Jigsaw
  Puzzles to Handling Real World Archaeological Fragments, CVPR
  2026](https://arxiv.org/abs/2605.12077). В авторской постановке
  показаны сетки до 5×5, поэтому это доказательство механизма, а не
  готовая гарантия масштабирования на 576 tiles.
- DiffAssemble — graph-diffusion framework для 2D/3D reassembly:
  [paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Scarpellini_DiffAssemble_A_Unified_Graph-Diffusion_Model_for_2D_and_3D_Reassembly_CVPR_2024_paper.pdf),
  [official code](https://github.com/iit-pavis/diffassemble).
- PuzzleFlow предоставляет авторскую реализацию flow matching для сборки:
  [official code](https://github.com/OfirShahar/puzzle-flow-matching).
- Symmetry-aware discrete diffusion имеет авторскую реализацию:
  [SymmetricDiffusers](https://github.com/DSL-Lab/SymmetricDiffusers).
- Structured Prediction Energy Networks учат дифференцируемую энергию целого
  структурированного ответа: [SPEN,
  ICML 2016](https://proceedings.mlr.press/v48/belanger16.html).

Эти работы доказывают применимость generative/energy inference к перестановкам и
reassembly, но не доказывают, что непрерывные координаты масштабируются до
576 почти неразличимых 20×20 tiles и метрики exact absolute position. Для этой
задачи diffusion должна работать над строгой перестановкой или bistochastic
матрицей с периодической hard projection, а не независимо предсказывать 576
координат.

Для текущей задачи наиболее важно ещё одно отличие: published
PuzzleFlow/JPDVT денойзят позиции из visual tokens, а наш первый
24×24 pilot должен стартовать из уже найденой SocketGlue-раскладки и видеть
frozen top-k directional edge graph. Это сохраняет достигнутый local signal и
делает эксперимент отличным от уже отвергнутых raw absolute heads P6/P10/P37.

### 2.5. Denoise-before-match

- DRUNet — практичный blind/non-blind denoising prior с официальной реализацией:
  [DPIR / DRUNet](https://github.com/cszn/DPIR).
- Noise2Self показывает, как обучать denoising без чистых targets при условии
  подходящей независимости шума: [Noise2Self,
  ICML 2019](https://proceedings.mlr.press/v97/batson19a/batson19a.pdf).

Здесь денойзер разумно применять только как **дополнительный feature view**:
`raw`, `denoised` и `raw − denoised` поступают в matcher одновременно. Pure
denoise-before-seam опасен по трём причинам: blur удаляет высокочастотное
продолжение контура; маленький crop провоцирует padding artifacts; generative
restoration может галлюцинировать разные границы у истинных соседей. Denoiser
лучше заморозить на первом этапе, не использовать соседние tiles как подсказку и
никогда не рендерить его matcher-only output вместо исходного tile.

## 3. Основной эксперимент: SocketGlue

### 3.1. Постановка

Для каждого tile `i` строятся четыре directional socket representation:
`s_i^L`, `s_i^R`, `s_i^U`, `s_i^D`. Горизонтальная задача — partial bipartite
matching между всеми 576 `R` и всеми 576 `L`; вертикальная — между `D` и `U`.
Матрица `M` каждого направления удовлетворяет:

```text
0 <= M_ij <= 1
sum_j M_ij <= 1
sum_i M_ij <= 1
sum_ij M_ij = 552
```

Остаточные массы строк и столбцов суммируются ровно в 24 border sockets с каждой
стороны. Это лучше отражает геометрию, чем 24 различимых фиктивных border-ID:
истинная граница имеет известную capacity, но у unmatched socket пока нет
абсолютного номера строки или столбца.

### 3.2. Encoder

Предлагаемая минимальная конфигурация:

1. Из каждого tile формируются `raw`, frozen-denoised, residual,
   Sobel/Laplacian и census/rank views.
2. Небольшой shared CNN кодирует tile context. Четыре side heads извлекают
   полосы шириной 4–6 px.
3. Каждая сторона остаётся последовательностью из 20 токенов с координатой вдоль
   границы; shallow 1D attention/conv превращает её в socket token размерности
   192 или 256.
4. Board-wide permutation-equivariant self-attention даёт каждому tile контекст
   остальных 575 элементов сцены.
5. Как в SuperGlue, чередуются self-attention внутри socket set и cross-attention
   между `R/L` или `D/U`. Направления используют общую backbone, но отдельные
   side embeddings.
6. Дешёвый factorized dot-product считает все 576² пар. Опциональный cross-encoder
   rerank-ит только top-32 кандидатов, сохраняя тангенциальную структуру границы.

Это не «SuperGlue на keypoints» буквально: keypoint geometry отсутствует, а
dustbin заменяется partial OT с известной unmatched capacity 24.

### 3.3. Training data и augmentations

Если доступны чистые train targets, exact labels создаются без восстановления
сомнительных скрытых перестановок: target режется на 24×24, затем каждый tile
повреждается **независимо**. Для 5600 сцен это даёт примерно 6.18 млн
направленных положительных внутренних ребер
(`5600 × 2 × 24 × 23`). Полный board остаётся единицей train sample.

Augmentation family должна покрывать наблюдаемую corruption family, но не менять
геометрию и orientation:

- независимые brightness/contrast/gamma, channel gain и слабый color matrix;
- Gaussian/Poisson noise, blur с несколькими sigma, JPEG/WebP-like artifacts;
- лёгкая boundary erosion/cutout и различный corruption strength у двух соседей;
- смешивание raw/denoised/residual views и channel dropout;
- **без** rotation, flip, resize или warp tile.

Recovered labels с actual dirty train допустимы лишь как confidence-weighted
добавка: прежний аудит оценивал в них заметную долю ошибочных tiles/adjacencies.
Они не должны перевешивать exact synthetic supervision.

### 3.4. Loss

Базовый objective:

```text
L = L_partial-OT-NLL
  + lambda_rank * L_batch-hard/listwise
  + lambda_border * L_unmatched
  + lambda_aug * L_consistency
  + lambda_topo * L_topology
```

- `L_partial-OT-NLL` максимизирует массу на 552 истинных matches после log-space
  Sinkhorn/partial OT, а не только raw pair logits.
- `L_batch-hard/listwise` различает истинного соседа от top-k похожих границ той
  же сцены; случайные межсценовые negatives слишком просты.
- `L_unmatched` обучает border sockets и калибрует общую unmatched capacity.
- `L_consistency` требует близких распределений матчей при двух независимых
  photometric corruptions одной чистой сцены.
- `L_topology` с небольшим весом штрафует несовместимые короткие циклы/degree
  patterns. Он не должен заменять точный decoder.

Curriculum: сначала factorized boundary pretraining, затем full-board contextual
fine-tuning, затем top-k reranker на hard negatives, которые реально предлагает
solver. Для label-noisy actual data полезны EMA teacher и clipping веса примера,
но не self-training по собственным hard predictions без audit.

### 3.5. Из socket scores в абсолютную раскладку

Пусть `P[i,g]` — assignment tile `i` в grid cell `g`, а `G_R` и `G_D` —
фиксированные adjacency matrices сетки. Decoder минимизирует/максимизирует
quadratic energy вида:

```text
E(P) = <U, P>
     + lambda_R * <C_R, P G_R P^T>
     + lambda_D * <C_D, P G_D P^T>

P is a 576 x 576 permutation matrix.
```

`C_R/C_D` — calibrated SocketGlue scores, `U` — слабые border/absolute unaries.
Практичный decoder:

1. хранит sparse top-k scores и всегда сохраняет fallback ребра;
2. запускает несколько инициализаций из robust LP/consensus components;
3. оптимизирует continuous relaxation projected/mirror descent с log-Sinkhorn и
   annealing температуры;
4. проецирует Hungarian-ом в строгую перестановку;
5. делает bounded 2-swap/3-cycle/block-swap polish по полной quadratic energy;
6. выбирает layout по независимому calibrated energy/verifier, а не по одному
   raw seam score.

Faces/objects могут дать learned global context, но правило «ставить лицо в
центр» нельзя hard-code: композиция сцены не гарантирует такой prior. Более
надёжны border likelihood, продолжение больших контуров и согласие нескольких
направлений.

### 3.6. Compute и критерий остановки

Ниже — инженерная оценка для этой задачи, не числа из статей:

- 10–40 млн параметров, `d=192/256`, 6–8 attention layers;
- batch 1–4 полных board на GPU порядка 24 GB, mixed precision и checkpointing;
- 12–36 GPU-hours до первого содержательного full-board результата;
- all-pairs factorized logits дешёвы относительно dense pair cross-encoder;
  последний должен работать только на shortlist.

Эксперимент следует закрыть, если на замороженной calibration панели он не даёт
одновременно: значимый edge lift, рост translation-aligned placement и рост
**exact absolute position после одного заранее выбранного decoder-а**. Нельзя
открывать новые decoder arms до бесконечности, пока adjacency растёт, а exact
position стоит на месте.

### 3.7. Ожидаемые failure modes

- partial OT может «украсть» capacity у истинных низкотекстурных границ;
- десятки почти одноцветных tiles могут быть информационно неразличимы локально;
- label noise обучит уверенно неправильные сокеты;
- unmatched head может схлопнуться и маркировать удобные, а не граничные tiles;
- top-k pruning необратимо удалит истинное ребро;
- улучшенные edge scores могут не изменить optimum или выбор basin quadratic
  decoder-а;
- denoised branch может доминировать и стереть полезный raw residual;
- семантический global context может переобучиться на типичную композицию train.

Нужные ablations: raw-only против multi-view; independent pair encoder против
board context; unconstrained logits против partial OT; fixed decoder на всех
вариантах; oracle decoder на true edges как верхняя граница.

## 4. Почему это не повтор уже проваленных веток

Локальная база знаний фиксирует следующие результаты:

- ordinary pairwise/listwise scorer улучшал retrieval/adjacency, но не итоговую
  layout: [knowledge base](../prior-research/knowledge-base.md) и свежий
  [candidate-k16 ranker](../experiments/edge-ranker-k16-scale.md);
- greedy/Paikin/Kruskal growth каскадирует раннюю ошибку, а conservative fusion
  даёт лишь малый нестабильный end-to-end lift:
  [conservative edge fusion](../experiments/edge-ranker-conservative-fusion.md);
- standalone continuous Sinkhorn layout relaxation уже отвергнут в V29;
- raw set-to-grid/Set Transformer был около chance, V27 дал лишь небольшой
  retrieval gain, а V30 GNN-unary/LNS почти не решил direct placement:
  [V-series audit](../prior-research/v-series.md);
- positional/global diffusion, plain Set-to-Grid и multi-phase relaxation без
  сильного scorer-а уже закрыты:
  [CB1/ORBIT R/P audit](../prior-research/cb1-orbit-r-p.md);
- denoise-before-match повторялся, а recovered mapping labels оказались шумными:
  [legacy and agent branches](../prior-research/legacy-and-agent-branches.md).

SocketGlue отличается сразу по четырём проверяемым признакам:

1. score каждого ребра зависит от **полного набора сокетов сцены**, а не только
   от пары или одного query shortlist;
2. известные one-to-one и border-cardinality constraints входят в matcher во
   время обучения, а не добавляются постфактум к слабым unaries;
3. side representation сохраняет упорядоченный boundary sequence и несколько
   photometric views;
4. linear socket OT отделён от quadratic tile-to-grid decoder-а, поэтому
   adjacency gain проверяется на exact absolute position без подмены задачи.

Этот пакет изменений достаточно отличен, чтобы оправдать один строгий
эксперимент. Если убрать board context или quadratic decoder, получится уже
проверенная и недостаточная линия «лучше ранжировать отдельные seams».

## 5. Четыре экспериментально отличающиеся альтернативы

### A. Edge-conditioned Set-to-Grid + quadratic Sinkhorn

**Когда пробовать:** только если SocketGlue существенно поднимет true-neighbor
recall/precision, но его quadratic decoder не сможет превратить это в absolute
placement.

**Архитектура:** 576 tile tokens и 576 learned grid queries с 2D positional
encoding; tile↔grid cross-attention получает не только visual token, но и
агрегаты SocketGlue top-k edges. На каждой итерации expected grid adjacency
сравнивается с `C_R/C_D`. Выход — bistochastic `P`, затем Hungarian.

**Loss:** assignment cross-entropy + expected horizontal/vertical adjacency +
border loss + entropy annealing. Exact tile-to-slot labels берутся из synthetic
boards.

**Compute/data:** ориентировочно 30–80 млн параметров и 1–3 GPU-days; нужны полные
exact-labeled boards. Warm-start от SocketGlue обязателен.

**Failure modes:** positional queries снова учат dataset composition вместо
границ; quadratic term нестабилен; Sinkhorn выдаёт гладкую смесь похожих tiles;
576×576 supervision доминируется лёгкими negatives.

**Почему это не прежний Set-to-Grid:** edge warm-start, explicit quadratic
adjacency objective и frozen exact-position gate. Без этих трёх отличий ветку
повторять не следует.

### B. Discrete permutation diffusion / PuzzleFlow

**Когда пробовать:** как high-risk/high-compute ветку после доказанной пользы
socket energy на меньших сетках 8×8 и 12×12.

**Архитектура:** состояние — строгая перестановка или bistochastic assignment, а
не 576 независимых `(x,y)`. Denoiser видит текущих hypothesized grid neighbors,
SocketGlue candidate graph и timestep. Между 8–20 denoising/flow steps выполняется
hard или near-hard projection. Training corruptions — random swaps, block swaps,
row/column shifts и разрушение правильных компонентов, а не только равномерная
случайная permutation.

**Loss/decoder:** denoising score/flow loss + edge energy + border energy;
несколько samples ранжируются frozen full-layout energy.

**Compute/data:** примерно 50–150 млн параметров, многие full-board passes и
2–7 GPU-days на рабочий pilot; нужны exact layouts и curriculum по размеру.

**Failure modes:** continuous coordinate collisions, нарушение биекции,
накопление projection error, высокая дисперсия samples, невозможность отличить
однотонные tiles и несоответствие масштаба опубликованных задач нашим 576 pieces.

### C. SPEN / hard-negative full-layout reranker

**Когда пробовать:** если уже есть несколько конкурентных legal layouts, но
solver выбирает не лучший.

**Архитектура:** небольшая energy network получает весь grid: socket likelihood,
border statistics, длинные контуры, local 2×2 consistency и global tile-context.
Она не генерирует layout с нуля.

**Loss/decoder:** contrastive/ranking loss между true layout и **трудными**
solver layouts: 2×2/4×4 block swaps, row/column shifts, 2-cycles и разные QAP/LNS
basins. Inference — переоценка candidate pool или несколько projected-gradient
steps плюс bounded swaps.

**Compute/data:** 5–20 млн параметров, порядка 6–18 GPU-hours после генерации
hard layouts. Самая дешёвая из альтернатив.

**Failure modes:** random negatives дают ложный высокий AUC; энергия использует
shortcut по corruption; reranker переобучается на один solver; candidate oracle
не содержит хорошей раскладки.

**Отличие от прежней layout-energy линии:** обучение только на
inference-relevant near-solutions и оценка candidate oracle перед запуском.

### D. Hierarchical component transformer

**Когда пробовать:** только если SocketGlue создаёт достаточно чистые mutual
edges и малые компоненты, но global placement разваливается.

**Архитектура:** из нескольких альтернативных high-precision edge thresholds
строятся overlapping 2×2/4×4 компоненты. Transformer кодирует их perimeter
sockets, uncertainty и внутреннюю геометрию, затем предсказывает compatible
translation/merge. Конфликты разрешаются global component-to-grid assignment.

**Loss:** правильность merge/relative translation + cycle consistency +
component-overlap consistency; negatives берутся из конкурирующих компонентов
одной сцены.

**Compute/data:** 20–60 млн параметров и 1–3 GPU-days, плюс нетривиальный
candidate generator.

**Failure modes:** одна неверная связь заражает целый компонент; low-texture
области не дают seed; overlapping candidates взрывают память; абсолютный offset
остаётся неопределённым.

Старые cycles/islands/component-growth ветки упирались именно в недостаточную
edge precision/coverage, поэтому начинать с этой альтернативы нельзя.

## 6. Сравнение и приоритет

| Направление | Главный новый сигнал | Риск | Оценка compute | Решение |
|---|---|---:|---:|---|
| **SocketGlue + quadratic decoder** | board-conditioned sockets + exact partial matching | средний | 12–36 GPU-h | **P0** |
| SPEN hard-layout reranker | различает близкие legal layouts | средний | 6–18 GPU-h после candidates | P1 при candidate headroom |
| Edge-conditioned Set-to-Grid | прямые tile-to-slot logits, усиленные edge graph | высокий | 1–3 GPU-days | P1 только после edge lift |
| Hierarchical components | multi-tile perimeter/context | высокий | 1–3 GPU-days | conditional |
| Discrete diffusion/flow | iterative global permutation correction | очень высокий | 2–7 GPU-days | research-only pilot |

Наиболее сильные и действительно различные предложения: (1) SocketGlue,
(2) SPEN reranker трудных legal layouts, (3) edge-conditioned Set-to-Grid и
(4) discrete permutation diffusion. Hierarchical component transformer стоит
между (1) и (3), но требует сначала получить чистые компоненты.

## 7. Рекомендуемый минимальный protocol

1. Зафиксировать exact synthetic board generator и audit corruption family.
2. Обучить factorized multi-view socket encoder без board context; это дешёвый
   ablation baseline, не самостоятельное новое направление.
3. Добавить full-board context и partial OT с capacity 24; сравнить на том же
   edge protocol.
4. Один раз заморозить socket scores и сравнить текущий solver, robust LP и
   sparse quadratic decoder. После выбора decoder больше не менять его внутри
   модельного ablation.
5. Primary endpoint — exact absolute position. Adjacency, R@k,
   translation-aligned placement, border accuracy и candidate oracle — только
   причины успеха/провала.
6. Передавать дальше лишь строгую перестановку исходных upright tiles. Matcher
   denoising остаётся feature-only; pixel restoration оценивается отдельно на
   уже зафиксированном layout.
7. Открывать альтернативу A или C только по наблюдаемому bottleneck. Diffusion
   сначала обязана пройти capacity test на 8×8/12×12 и лишь затем на 24×24.

Такой порядок проверяет одну новую причинную гипотезу за раз: сначала способен ли
contextual partial matcher найти правильные сокеты, затем способен ли quadratic
decoder превратить их в абсолютную сетку, и только после этого нужен ли более
сложный генератор перестановок.

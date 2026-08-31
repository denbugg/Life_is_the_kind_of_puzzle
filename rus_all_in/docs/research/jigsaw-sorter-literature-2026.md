# Новая архитектура tile-sorter: literature update 2026

Дата среза: **2026-08-30**.

> **Implementation update.** Bounded BorderPointer-24 и единственный
> baseline-guided causal rescue уже выполнены. Standalone decoder провалил
> matched exact/adjacency gate; deployable prefix signal оказался положительным,
> но слишком слабым для score gain. Метрики, hashes и stop/compose decision:
> [BorderPointer experiment](../experiments/border-pointer-sorter.md). Этот memo
> ниже сохраняет исходную preregistration rationale, а не отменяет measured result.

> **Rank2 implementation update.** Fullres-relation fusion позже формально
> прошёл activation (`+14.453 pp` top32 precision, `+4.625` correct/board),
> поэтому Sparse BorderGraph-QAP был реализован и bounded-проверен. Exact16
> layout буквально совпал с decoder144+cyclic5, а pure quadratic
> truth-minus-decoder energy был `−77.475`; gate fail-stop. Не повторять этот
> top8/baseline-anchor/two-step QAP как scale или weight sweep. Полный
> [отчёт](../experiments/sparse-bordergraph-qap.md).

> **Edge2Vec/TEN implementation update.** Full-resolution ordered twin matcher
> проиграл d64 напрямую, но добавил `+7.416 pp` top32 supply. Отдельный
> bidirectional raw/twin union reranker на новой source-disjoint eval24 прошёл
> оба локальных gate: partial-OT R@1/R@5 `+0.476/+0.279 pp`, fixed top144
> `+8.458` correct edges/board и `+2.937 pp` precision. Разрешённый
> decoder144+cyclic5 descriptive дал adjacency `12.847→13.753%` и exact
> `0.792→1.208` tiles/board. Последующий однократный frozen fresh64
> confirmation без retrain прошёл submission gate: exact `+0.344`
> tile/board (при пересекающем ноль CI), adjacency `+0.752 pp` с
> положительным CI, top144 `+5.266` correct/board, `128/128` strict
> original permutations. Детали и protocol caveat — в
> [отчёте](../experiments/raw-twin-union-reranker.md).

Это architecture-selection memo, а не отчёт о новом запуске. Он продолжает
[обзор contextual socket matching](socket-matching-literature.md) и сверяется с
[authoritative layout ledger](../prior-research/layout-sorter-ledger.md). В этом
документе нет обученных моделей и новых метрик.

## Решение в одном абзаце

Первым следует реализовать небольшой **BorderPointer-24**: каждый исходный
20×20 tile переводится без downsample в непрерывное поле
`20×20×48`, его 76 уникальных perimeter-позиций кодируются в упорядоченную
последовательность, а permutation-equivariant encoder видит все 576 tiles.
Autoregressive pointer decoder проходит по **фиксированным абсолютным** raster
slots `0..575`, выбирает один ещё не использованный tile и добавляет к pointer
logit совместимость с уже поставленными left/up соседями. Used-mask гарантирует
строгую перестановку. Фиксированный slot query, четыре distance-to-border
признака и learned unmatched/border logits решают absolute origin внутри самой
задачи, без эвристики «лицо в центр».

Это переносит из PuzLM действительно новую для workspace часть — глобальный
encoder–decoder и условное autoregressive placement — но **не** повторяет его
PCA/k-means tokenizer. Локальные symbolic/PuzLM tokens уже проверялись в I20 и
не дали fusion gain. Непрерывное 48-D поле сохраняет тот слабый fine seam signal,
который жёсткая квантизация здесь потеряла.

Пилот можно начинать до завершения любых дополнительных обзоров. Он bounded:
`128–256` clean organizer-train sources, не более `400` full-board updates,
`d_model=128`, четыре encoder и четыре decoder layers, greedy и beam-4 free-run
на `16×1` source-disjoint exact boards. Это **discovery**, где допустим
описательный сигнал. Default можно менять только после отдельного fresh
`source64×draw2` exact gate с source-clustered confidence interval.

## 1. Контракт задачи и текущая точка отсчёта

- Вход — неупорядоченное множество 576 upright original tiles размером 20×20;
  выход — одна биекция на абсолютную сетку 24×24.
- Matcher может строить denoised/latent views. Raw, restored или generated
  feature нельзя считать разрешением подменить пиксели финального tile.
  Sorter должен вернуть только индексы; legal renderer собирает **исходные
  upright tiles**. Pixel restoration остаётся отдельным downstream этапом.
- Обучающие labels для новых sorter-ов — только точная inverse shuffle известных
  organizer-train изображений. Target-assisted Hungarian labels реальных dirty
  boards годятся лишь как вторичная диагностика.
- Primary metric — число tiles в буквально правильной абсолютной клетке.
  Adjacency, aligned placement и local R@k — диагностические метрики.
- В reference full-cycle d64 SocketMatcher имел local R@1/R@5
  `17.765/35.734%`; `decoder144 + cyclic-border5` дал `1.406` exact tile/board и
  `13.103%` adjacency на своей matched `16×2` панели. Это ориентир, не число для
  сравнения между разными panels. Новый gate всегда должен пересчитать frozen
  comparator на тех же cases.

## 2. Что уже проверено и не должно маскироваться под новую идею

| Уже проверенная линия | Evidence в workspace | Следствие для нового дизайна |
|---|---|---|
| Symbolic/PuzLM-like tokens как local score | I20: agreement `.4069`, neighbour R@1/R@5/R@64 `.0682/.1191/.3943`, median rank 106, fusion gain отсутствует | Не повторять PCA/k-means vocabulary или token distance. Из PuzLM переносить global autoregressive solver, а не старый local tokenizer. |
| HBT: edge encoder → Transformer/Sinkhorn | Exact port: edge R@1 `6.280→3.223%`; global evidence также хуже | Не переносить max-pooling 320-D HBT encoder и simultaneous `N×N` head. |
| Standalone absolute/set-to-grid/Sinkhorn | Absolute head нашёл row signal, но standalone Hungarian уничтожил adjacency; component integration не прошла fresh exact material gate | Slot unary полезен как часть conditional decoder, но ещё один simultaneous row/column head закрыт. |
| Position diffusion / flow | I21 positional DDPM overfit; current [SocketPermutationFlow](../experiments/socket-permutation-flow.md) на 24×24 дал direct `0.3038→0.2170%`, adjacency `15.6703→1.2908%` | Не запускать ещё один coordinate flow, даже с frozen Socket graph. PuzzleFlow остаётся literature context, не roadmap experiment. |
| Component relation reranker | Pair/translation R@1 `20.968→24.335%`, R@5 `66.883→71.817%`, но top-32 precision только `16.797→17.188%` | Multi-component context информативен, но non-sequential residual score недостаточно уверен. Сохранить как auxiliary/candidate evidence. |
| Solver-only QAP, LP, pose sync, cycles | LP sync exact на oracle и лучше clean+blur, но требовал около `0.9` precision; P13 robust pose sync остался около `0.222%` direct | Не запускать новый solver на прежних scores. QAP/sync допустимы только после нового representation gate. |
| Hard denoise-before-match | DRUNet/DualNAF, bilateral HBT и множество historical arms дали mixed/negative geometry | Denoised view только auxiliary; raw skip обязателен. |
| Большой Transformer / больше augmentations | 31M/77M и другие capacity sweeps не перенеслись source-disjoint | Новизна должна быть в target и decoding factorization, не в размере модели. |

Полные ссылки на эти результаты находятся в
[ledger](../prior-research/layout-sorter-ledger.md),
[legacy audit](../prior-research/legacy-and-agent-branches.md),
[absolute sorter report](../experiments/absolute-coordinate-sorter.md) и
[component relation report](../experiments/component-relation-reranker.md).

## 3. Primary-source literature и точный перенос

### 3.1. Global square-jigsaw models

#### PuzLM: border tokens + encoder–decoder sequence prediction

**Источник.** Elkin, Shahar, Ben-Shahar, *PuzLM: Solving Jigsaw Puzzles with
Sequence-to-Sequence Language Models*, arXiv v2, 2026:
[paper](https://arxiv.org/html/2511.06315v2). На дату обзора официальная
реализация авторами не опубликована; paper — единственный primary artifact.

**Механизм.** Tile режется на `B×B` patches, PCA и k-means превращают patches в
discrete vocabulary, сохраняются `4B−4` border tokens в clockwise order. BART
encoder получает весь tokenized puzzle, а autoregressive decoder условно
предсказывает permutation decisions. В paper default `B=4`, PCA dimension 1024,
vocabulary 4096. Global bidirectional input context сочетается с conditional
one-step-at-a-time output. На ImageNet 3×3 авторы сообщают absolute/perfect
`92.2/87.1%`, на eroded JPwLEG-5 — `72.1/32.5%`.

**Что переносится.** Иерархия `ordered border → global encoder → conditional
decoder` и exact permutation cross-entropy. Наш decoder разворачивает задачу в
более удобном направлении: step `s=24r+c` означает фиксированный output slot,
а pointer выбирает identity одного unused input tile. Так origin задан в output
space, а duplicate tiles архитектурно невозможны.

**Почему это materially distinct.** I20 измерял только local similarity между
уже квантизованными symbolic tokens. Он не обучал whole-board bidirectional
encoder и не condition-ил каждый выбор на raster prefix. Absolute head и HBT
предсказывали assignments одновременно; BorderPointer factorizes permutation
условно и в каждом шаге непосредственно видит уже выбранные left/up tiles.

**Compute/data.** Published числа относятся максимум к 5×5 в основной таблице,
а не к 24×24. Feeding `76×576=43,776` pixel tokens в BART непрактично, поэтому
нужен hierarchical local compression до одного tile token и четырёх side
tokens. Engineering estimate bounded версии: 4–8M trainable parameters,
128–256 exact synthetic sources, 400 updates, примерно 2–8 GPU-hours в
зависимости от устройства. Это оценка проекта, не число из paper.

**Legal boundary.** Использовать только organizer-train images и разрешённые
photometric corruptions; не загружать неизвестные clean references. Model
выдаёт индексы, а не pixels. Paper license не является лицензией на отсутствующий
code; архитектуру следует реализовать самостоятельно.

#### Heck et al.: edge similarity as Transformer placement embedding

**Источник.** Heck, Lermé, Le Hégarat-Mascle, *Solving jigsaw puzzles with
vision transformers*, Pattern Analysis and Applications 2025:
[open paper](https://d-nb.info/1375813587/34),
[publisher page](https://link.springer.com/article/10.1007/s10044-025-01484-z).

**Механизм.** Four-side CNN embeddings обучаются contrastive loss; learned
similarity входит в positional representation whole-puzzle Transformer, затем
`N×N` assignment logits проходят Sinkhorn. Paper действительно демонстрирует
70–600 pieces, но pieces имеют 32/64 px и erosion radius 0/2/4. Training использует
450k LAION-art images, 4×A100-40GB, 50M Transformer и 9–38M CNN. CNN применяет
2×2 max-pooling после convolution layers.

**Что переносится.** Только принцип: global placement должен видеть learned
edge evidence, а placement loss может обратно улучшать edge representation.

**Почему не roadmap.** Exact port HBT уже decisively проиграл current d64, а
20×20 crop особенно плохо совместим с повторным max-pooling. Published model
не покрывает независимые brightness/noise/JPEG/blur на каждом tile и слишком
дорог для bounded discovery.

**Legal boundary.** Самостоятельно обученные organizer features допустимы как
matcher-only. LAION training recipe и pretrained payload нельзя автоматически
считать разрешёнными правилами конкурса.

#### PuzzleFlow и FCViT: важные negative controls

**Источники.** Shahar et al., *The Missing GAP*, CVPR 2026:
[paper](https://arxiv.org/abs/2605.12077),
[official code](https://github.com/OfirShahar/puzzle-flow-matching). Kim et al.,
*Solving Jigsaw Puzzles by Predicting Fragment's Coordinate Based on Vision
Transformer*: [official code](https://github.com/HiMyNameIsDavidKim/fcvit),
[paper DOI](https://doi.org/10.1016/j.eswa.2025.126776).

**Механизм.** PuzzleFlow кодирует fragments pretrained ViT-Base и итеративно
refines permutation через discrete flow matching; official release использует
GAP-3/GAP-5 и заявляет `O(N²)` на step. FCViT directly predicts fragment
coordinates с ViT.

**Почему это не новый эксперимент.** Published grids только 3×3/5×5, а current
edge-conditioned strict-permutation 24×24 flow уже разрушил Socket components.
Direct coordinate formulation также совпадает с закрытой absolute-head линией.

**Compute/data и legal boundary.** Полный ViT-Base flow существенно тяжелее
bounded pointer. Position/permutation diffusion сама по себе legal, если
render остаётся original-tile permutation; pixel diffusion или generated
fragments не входят в legal sorter. Эти источники оставлены как explicit
stop-sign, не как третий roadmap arm.

### 3.2. Learned edge compatibility и dense boundary representation

#### Edge2Vec и TEN

**Источники.** Rika et al., *Edge2Vec*, 2022:
[paper](https://arxiv.org/abs/2211.07771). Rika et al., *TEN: Twin Embedding
Networks for the Jigsaw Puzzle Problem with Eroded Boundaries*, 2022:
[paper](https://arxiv.org/abs/2203.06488).

**Механизм.** Оба метода учат быстро сравнимые directional edge embeddings;
Edge2Vec добавляет hard batch triplet objective, TEN специализирует twin
encoders на eroded boundaries. Это подтверждает, что edge embedding и
hard-negative mining масштабируются лучше pair-CNN call на каждую пару.

**Что переносится.** Side-level InfoNCE/triplet auxiliary и hard negatives из
той же board: near-ties d64, reciprocal-but-false edges и похожие однотонные
tiles. Ни один side нельзя сразу сворачивать в один vector: BorderPointer
сохраняет 20 positions и их касательный порядок до local 1-D encoder.

**Почему distinct.** Workspace уже исчерпал standalone side vectors и seam
rankers. Новый 48-D field обучается одновременно corruption-invariance,
directional adjacency и exact autoregressive placement; side distance — лишь
один logit term, не финальный solver.

**Compute/data и legal boundary.** Auxiliary можно считать на sampled in-board
negatives, не на всех `576²` pairs. Все embeddings matcher-only; никакой
inpainting border не рендерится.

> **Independent representation implementation.** После solver-negative QAP
> эта идея отделена от placement decoder и реализована как самостоятельный
> [full-resolution twin side matcher](../experiments/fullres-twin-side-matcher.md):
> `20×20×48` field без downsample, ordered length-20 compatibility, raw skip и
> dual-corruption within/cross-view listwise loss по всем 576 same-board
> candidates. Mechanical 4×4 дала `100%` R@1; fit256/eval24 queued. Этот запуск
> не использует autoregressive placement из первоначального текста раздела и
> поэтому честно проверяет, создаёт ли сама representation новый local signal.

#### DnCNN, SwinIR и HRNet как архитектурные, не pretrained, priors

**Источники.** Zhang et al., *DnCNN*:
[paper](https://arxiv.org/abs/1608.03981),
[official code](https://github.com/cszn/DnCNN). Liang et al., *SwinIR*:
[paper](https://arxiv.org/abs/2108.10257),
[official code](https://github.com/JingyunLiang/SwinIR). Sun et al., *HRNet*:
[paper](https://arxiv.org/abs/1908.07919).

**Механизм.** DnCNN учит same-resolution residual для blind Gaussian denoise и
JPEG deblocking; SwinIR сочетает shallow convolution, residual Swin blocks и
long skip для restoration; HRNet показывает ценность сохранения high-resolution
stream для position-sensitive tasks.

**Что переносится.** Не готовый denoiser, а shape-preserving stem:
`20×20×C → 20×20×48`, только stride 1, без pooling/patch merge/bottleneck,
с raw/residual skip. Это прямо отвечает риску U-Net на 20×20: feature map не
проходит цепочку `20→10→5→...`.

**Почему distinct.** DRUNet/DualNAF проверялись как independent restored image
views. Здесь restoration target не является output вообще; pointwise field
оптимизируется по adjacency/exact placement и paired-corruption consistency.

**Compute/data.** Четыре depthwise-separable residual blocks width 48 меньше
1M parameters. Они применяются batched к `B×576` tiles. Full HRNet или полный
SwinIR на каждом 20×20 tile не нужен.

**Legal boundary.** Raw RGB остаётся доступным model input; latent/restored
каналы не покидают matcher. Финальный canvas использует original tiles. Любой
внешний checkpoint требует отдельного rules/license audit; первый pilot должен
обучаться с нуля на organizer train.

### 3.3. Permutation-equivariant assignment и QAP

#### Set Transformer и entropy-adaptive Gumbel–Sinkhorn

**Источники.** Lee et al., *Set Transformer*, ICML 2019:
[paper](https://proceedings.mlr.press/v97/lee19d.html). Eisenberg, Lindenbaum,
*Learning Permutation from Structure Without Supervision*, ICML 2026:
[paper](https://arxiv.org/html/2605.25551),
[official code](https://github.com/LindenbaumLab/Learning-Permutation-from-Structure-Without-Supervision).

**Механизм.** Set attention даёт permutation-equivariant processing без input
index embeddings. Entropy-adaptive Gumbel–Sinkhorn строит row/column-local
inverse temperature из assignment entropy: confident rows/columns sharp-ятся,
ambiguous остаются diffuse; eval заканчивается Hungarian.

**Критический caveat absolute origin.** Jigsaw experiments второй работы
используют **истинные anchors**: 1 для 5×5, 6 для 6×6 и 12 для 7×7; метрика —
Kendall tau. Это не evidence решения нашего origin. Кроме того, official repo
на дату обзора содержит number-sorting code, но не jigsaw implementation.

**Что переносится.** Set equivariance обязательна для BorderPointer encoder.
Entropy-adaptive Sinkhorn можно использовать только как optimizer primitive в
QAP arm, если новая unary/edge energy уже информативна.

**Почему не standalone experiment.** Set/slot/Sinkhorn heads уже были около
chance; авторы сами отмечают, что entropy control не устраняет слабый или
симметричный objective. Наш exact supervision сильнее unsupervised smoothness,
а absolute origin должен быть learned из border evidence, не раскрыт anchor-ом.

**Compute/data и legal boundary.** `576×576` soft matrix имеет 331,776 entries и
помещается в память, но это ещё не quadratic edge tensor. Hungarian сохраняет
legal bijection. Truth anchors на evaluation использовать нельзя.

#### QC-DGM: differentiable quadratic graph matching

**Источник.** Gao et al., *Deep Graph Matching under Quadratic Constraint*,
CVPR 2021: [paper](https://openaccess.thecvf.com/content/CVPR2021/html/Gao_Deep_Graph_Matching_Under_Quadratic_Constraint_CVPR_2021_paper.html),
[official code](https://github.com/Zerg-Overmind/QC-DGM).

**Механизм.** Assignment matrix обучается с explicit quadratic discrepancy
между adjacency structures; relaxed Koopmans–Beckmann QAP оптимизируется
дифференцируемым modified Frank–Wolfe, после чего следует discretization.

**Что переносится.** Match tile graph, построенный из learned directional
compatibility, с фиксированным 24×24 grid graph. Loss должен штрафовать не
только неправильный tile→slot unary, но и несовпадение `right/down` edges.

**Почему distinct.** Historical QAP получал frozen слабые pair scores. Новый arm
совместно обучает 48-D boundary field и structural energy от exact permutation.
Он также отличается от SocketPermutationFlow: оптимизация сохраняет pairwise
grid energy, а не проецирует каждый refinement в separable coordinate logits.

**Compute/data.** Полный four-index affinity нельзя материализовать. Нужны
top-k (`8–16`) tile edges, sparse fixed grid adjacency и factorized energy;
2–4 unrolled optimization steps в bounded run. Engineering estimate:
3–10M parameters, 256 sources/400 updates, 6–18 GPU-hours.

**Legal boundary.** QAP меняет только permutation indices. Hard Hungarian или
equivalent collision-free projection обязателен перед render.

### 3.4. Relative-pose synchronization

#### Learning2Sync

**Источник.** Huang et al., *Learning Transformation Synchronization*, CVPR
2019: [paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Huang_Learning_Transformation_Synchronization_CVPR_2019_paper.html),
[official code](https://github.com/xiangruhuang/Learning2Sync).

**Механизм.** Model чередует weighted transformation synchronization и neural
re-estimation весов относительных transforms. Weight network видит состояние
текущей synchronization, поэтому может учить data-specific robust loss вместо
фиксированного threshold/cycle rule.

**Что переносится.** Для upright square tiles rotation фиксирован; остаётся 2-D
translation. Edge `(i,j)` предлагает `t_j−t_i∈{(0,1),(1,0)}`. Небольшой
recurrent weight head может видеть 48-D side pair, reciprocal rank, margin,
cycle/sync residual и component confidence, затем перевзвешивать sparse
least-squares synchronization.

**Absolute-origin problem.** Relative synchronization имеет gauge:
`t_i + constant` даёт то же решение. Если одна корректная component spans всю
24×24 board, bounding box фиксирует shift; иначе необходимо перечислить все
feasible integer component translations и выбрать их learned top/left/right/
bottom border unary плюс collision-free grid assignment. Нельзя просто ставить
component «в центр».

**Почему это пока rank 3.** Static/cycle pose sync уже failed, а historical LP
оживал около 0.9 selected-edge precision. Learned residual weighting materially
distinct, но не создаёт отсутствующий edge signal. Запуск разрешён только после
успешного 48-D high-confidence edge gate.

**Compute/data и legal boundary.** Sparse top-8 graph, 3–5 reweight/sync rounds,
менее 3M parameters и ориентировочно 2–8 GPU-hours после cached embeddings.
Output после integer quantization/Hungarian — только strict original-tile
permutation.

## 4. Experiment 1: bounded BorderPointer-24

### 4.1. Full-resolution 48-D field

Для tile `x_i∈R^{20×20×3}`:

```text
raw RGB + fixed raw gradients
    → 1×1 lift to 48 channels
    → 4 stride-1 residual blocks, no pooling/patch merge
    → F_i ∈ R^(20×20×48)
```

Padding должен явно не смешивать tile с выдуманным соседом: valid-mask/partial
convolution или reflection/replication с отдельным valid-border channel.
Нулевой padding без mask опасен, потому что сеть легко выучит искусственную
рамку. Raw RGB/high-pass skip подаётся в compatibility head вместе с `F_i`,
поэтому latent denoise не может полностью стереть seam.

Из `F_i` берутся 76 уникальных perimeter positions в clockwise order:
`top20 + right19 + reversed-bottom19 + reversed-left18`. Shared shallow 1-D
conv/attention с side/within-side position tags возвращает четыре sequences
`20×48`, четыре side summaries и один tile summary. Глобальный Transformer не
видит все 43,776 border positions: только 576 tile summaries; side sequences
остаются у fast compatibility head.

Loss representation:

- pointwise consistency между двумя независимыми legal corruptions одного
  clean tile, после channel normalization;
- directional InfoNCE/triplet на true right/down edges с in-board hard
  negatives;
- raw-preserving auxiliary: clean-vs-corrupted field similarity, но без
  требования генерировать clean RGB;
- основной exact pointer CE.

Photometric corruption curriculum: independent brightness/contrast/gamma,
color cast, Gaussian/Poisson-like noise, blur, JPEG/quantization. Запрещены
rotation, flip, crop, resize, warp и любые изменения геометрии tile.

### 4.2. Board encoder и pointer decoder

Каждый tile token объединяет field summary, четыре side summaries и frozen d64
Socket summary. Четыре self-attention layers width 128 работают без input-order
embedding. Перестановка input tiles должна только переставить memory rows.

Decoder step `s=24r+c` имеет query:

```text
learned row[r] + learned col[c]
+ normalized (r, c, 23-r, 23-c)
+ causal copies of the previously selected tiles' encoded memory vectors
```

Здесь нет embedding table по input tile ID: после pointer choice decoder
gather-ит соответствующую permutation-equivariant memory row. Поэтому
перенумерация входных tiles перенумеровывает logits/choices, но не меняет
физическую раскладку.

Pointer logit для unused candidate `j`:

```text
q_s · key_j
+ λ_left * compat(right(selected[s-1]), left(j))       if c > 0
+ λ_up   * compat(bottom(selected[s-24]), top(j))      if r > 0
+ border_unary(j, r, c)
+ frozen Socket/component evidence
```

`border_unary` использует learned unmatched probability каждой стороны и
distance-to-border query. Это делает первый top-left decision и общий origin
явными. Уже выбранные tiles получают `−∞`; output всегда содержит ровно 576
разных original identities. Train использует parallel teacher forcing;
free-run — greedy и beam-4. Beam нужен в pilot только для диагностики early
seed error, не для выбора вариантов по truth.

### 4.3. Почему это не «ещё один большой Transformer»

- Новый supervised object — conditional tile identity для fixed absolute slot,
  а не independent row/column coordinate.
- Каждый step получает deployable left/up context, то есть operationalizes I12
  multi-neighbour ceiling без oracle neighbours.
- Local 48-D field не downsample-ится и сохраняет per-pixel border order.
- Strict permutation — архитектурный invariant, не надежда на Sinkhorn.
- Absolute origin встроен в slot order и border unary, а не выбирается center/
  background semantic prior.

### 4.4. Bounded implementation brief

| Блок | Frozen первый выбор |
|---|---|
| Field | width 48, 4 stride-1 residual blocks, no pooling, valid-border mask |
| Perimeter | 76 unique positions, 2 shallow 1-D blocks, 4 side + 1 tile summaries |
| Existing evidence | frozen d64 summary и frozen directional scores; raw view не удалять |
| Board encoder | 4 layers, `d=128`, 4 heads, FFN 512, no input-index embedding |
| Pointer decoder | 4 causal layers, fixed raster slots, used-tile mask, left/up compatibility |
| Training | 128–256 sources, max 400 board updates, exact inverse shuffle, paired corruptions |
| Eval | unseen `source16×draw1`, greedy + fixed beam-4, comparator rerun on same boards |
| Hard caps | no test/calibration access, no model >10M, no >400 updates, no render changes |

Implementation можно начинать немедленно. Не нужно сначала строить BART-sized
модель, k-means vocabulary, diffusion process или full QAP solver.

## 5. Ranked three-experiment roadmap

### Rank 1 — BorderPointer-24

**Question.** Даёт ли full-resolution corruption-invariant border field плюс
conditional raster decoding новый exact absolute signal, не уничтожая Socket
adjacency?

**Low discovery gate, не promotion.** После mechanical 4×4 overfit/equivariance
tests открыть один source-disjoint `16×1` panel. Достаточно описательно увидеть
хотя бы один заранее объявленный полезный сигнал:

1. free-run exact delta имеет положительный mean против matched frozen
   `decoder144+cyclic5`; или
2. exact flat, но adjacency не падает больше чем 2 pp и correct row/column count
   растёт; или
3. на prefix decisions с только model-predicted context conditional candidate
   R@1 растёт минимум на 2 pp против того же 48-D pair score без prefix.

Teacher-forced accuracy отдельно публикуется, но **никогда** не считается
end-to-end успехом. Discovery result может быть описательным; CI и material
exact gain здесь не обязательны. Если все три сигнала отсутствуют, остановить
формулировку без capacity sweep.

**Strict promotion.** Один заранее замороженный fresh `source64×draw2` panel,
полностью исключающий lineage. Требовать одновременно:

- mean exact gain не меньше `+0.5 tile/board`;
- source-clustered paired 95% CI exact delta имеет lower bound `>0`;
- adjacency loss не хуже `−0.2 pp`;
- 128/128 strict permutations, input-permutation equivariance и original-upright
  render;
- никакого выбора beam/weights/checkpoint по этой panel.

Только этот gate разрешает изменить default sorter.

### Rank 2 — Sparse BorderGraph-QAP

**Question.** Если field улучшает edge evidence, но autoregressive errors
каскадируются, сможет ли end-to-end quadratic grid energy сохранить хорошие
components и одновременно найти absolute slots?

**Design.** Тот же 48-D field строит top-8/16 directional tile graph. Fixed
24×24 grid — второй graph. Tile→slot unary содержит BorderPointer memory/slot
score; pairwise term связывает learned tile edges с right/down grid edges.
Использовать 2–4 sparse modified Frank–Wolfe/QC steps, optional
entropy-adaptive Sinkhorn и final Hungarian. Не материализовать dense
`(576²)²` affinity.

**Activation prerequisite.** Не запускать, пока field не улучшит source-disjoint
high-confidence local precision минимум на `+3 pp` и correct top-32 attachments
минимум на `+1/board`, либо BorderPointer discovery не покажет positive exact с
явным autoregressive cascade failure. Иначе это повтор solver-only линии.

**Discovery.** 64–128 fit sources, 16 exact held sources, 2 unrolled steps.
Описательно требовать joint energy lower на truth, чем на frozen decoder layout,
и free-run adjacency не хуже comparator более чем на 1 pp. Это ещё не promotion.
Promotion использует тот же fresh exact CI contract, что Rank 1.

### Rank 3 — learned 2-D synchronization + explicit origin

**Question.** Может ли recurrent residual-aware edge weighting выделить
достаточно чистый component graph, когда fixed cycles/thresholds не смогли?

**Design.** Top-8 relative translations → alternating learned weights and
sparse 2-D synchronization → integer components → enumeration of feasible
component shifts → border-unary/collision-free assignment. Никаких semantic
center guesses.

**Activation prerequisite.** Selected edge precision около historical LP knee
(`≈0.9`) либо убедительная calibrated precision curve новой 48-D модели.
Без этого эксперимент не запускать. Low discovery оценивает outlier rejection,
aligned component placement и origin classification отдельно; direct exact
promotion всё равно требует fresh source64×draw2 positive CI. Это rank 3 именно
потому, что synchronization не создаёт edge information и absolute gauge надо
решать отдельно.

## 6. Общий evaluation protocol

### Discovery допускает быстрые отрицательные ответы

- 4×4 capacity проверяет только mechanics, не generalization.
- 24×24 `16×1` source-disjoint panel может давать descriptive exact/local
  deltas без CI.
- Один frozen config; максимум один scheduled checkpoint и greedy/фиксированный
  beam-4.
- Публиковать exact tiles/board, row/column, adjacency, strict-permutation count,
  teacher-forced и free-run отдельно.
- Любой catastrophic adjacency collapse, duplicate identity, input-order leak
  или use of target-assisted label завершает arm.

### Promotion всегда строгий

- Fresh clean sources и два независимых corruption/shuffle draws на source.
- Source-clustered paired bootstrap CI, а не treating 128 boards independent.
- Comparator, corruption draws и exact labels фиксируются до candidate decode.
- Primary — exact absolute delta. Local R@1, aligned, SSIM и row signal не могут
  отдельно promote sorter.
- Real dirty target-assisted panel открывается только после synthetic exact
  gate и остаётся secondary.

## 7. Legal checklist для любого roadmap arm

1. Output sorter-а — длина-576 permutation исходных tile IDs.
2. Никаких rotation, flip, resize, warp, crop или tile deformation.
3. Denoised, residual, Transformer или diffusion tensors — matcher-only.
4. Raw upright tiles собираются без изменения; image-quality tail проверяется
   отдельно по правилам соревнования.
5. Никаких retrieved clean test sources, hidden target images, ручных anchors
   или truth-derived center/background decisions.
6. Training corruptions только photometric/noise/blur/JPEG и документированы.
7. External code/checkpoint/data требуют отдельного license/rules audit; первый
   BorderPointer pilot обучается с нуля на organizer train и frozen audited d64.

## Итоговый приоритет

`BorderPointer-24` — единственный arm, который стоит кодировать немедленно. Он
прямо operationalizes идею 48–50-D per-pixel embeddings, избегает U-Net collapse
на 20×20, использует уже подтверждённый d64 edge signal, получает deployable
multi-neighbour context и решает absolute origin в output parameterization.

Sparse QAP и learned synchronization — условные downstream decoders только
после representation evidence. PuzzleFlow/coordinate diffusion, standalone
Sinkhorn, symbolic token distance, HBT port и ещё один большой Transformer
закрыты имеющимися workspace results и не входят в новый compute budget.

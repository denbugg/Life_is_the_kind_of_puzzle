# NEXT solver roadmap

Дата среза: 2026-08-31. Это bounded handoff для следующих solver-
веток; он не авторизует competition-test, submission или вывод
denoised pixels. Порядок ниже оптимизирован по expected value:
сначала короткий проверяемый сигнал, затем отдельный scale-up на
40–60 GB GPU.

## От какой точки отталкиваемся

- Текущий confirmed pair leader —
  [relation-level selector](experiments/taska-relation-truth-selector.md) одного
  из шести целых TASKA layouts. На новой source-disjoint `16×2`
  confirmation он дал `+5.844` satisfied pairs/board, source-CI95
  `[+3.000,+9.126]`, ни одного pair loss на 32 cases; exact delta `-0.156`.
- [Distance validation](experiments/tile-position-distance-metric-validation.md)
  подтвердила, что absolute mean Manhattan — хороший smooth progress
  signal: within-source Spearman с clean/dirty/h20 SSIM равен
  `.868/.863/.863`. Но exact/radius0 имеет более сильную линейную
  связь с SSIM, а cyclic-aligned distance может счесть неверную глобальную
  фазу perfect. Поэтому metric contract: absolute exact primary,
  absolute Manhattan secondary, radius2 companion, pairs и same-tail SSIM отдельно.
- [Tri-emitter verifier](experiments/tri-emitter-edge-verifier.md) уже
  конвертировал часть raw + adapter1600 + DINO supply: raw-relative
  R@1/R@5 `+1.053/+1.234 pp`. На fixed matched coverage 3/5/10% его
  reciprocal precision выше raw на `+5.094/+6.569/+6.059 pp`. Но native
  tail слишком длинный: на matched 7,329 edges precision хуже raw на
  `-1.774 pp`; поэтому decoder не открывался.
- [HGB-ranked all-edge union](experiments/taska-relation-ranked-union.md)
  обрушился ещё на local32: `326.750→199.500`, delta `-127.250`
  pairs. Высокий relation AUC не разрешает безусловный
  независимый edge synthesis.

## Приоритеты

| Priority | Направление | Быстрый локальный ответ | Роль 40–60 GB GPU |
|---|---|---|---|
| 1 | Joint outgoing/incoming tri-emitter verifier | Вытянули ли exact-neighbour R@1/R@5 и clean reciprocal head | Больше source-disjoint boards, full sparse row+column batches, hard collisions |
| 2 | Raw-preserving full-resolution boundary model | Даёт ли matcher-oriented restoration новый supply, не теряя raw | Широкий stride-one encoder/cross-attention и большой corruption stream |
| 3 | Compatibility-aware structured decoder | Можно ли безопасно добавить только clean verifier head к six-arm layout | Контекстная graph policy/energy поверх многих fit boards |
| 4 | Pair-safe global-origin abstention | Есть ли достаточно safe-positive rolls для exact | Shift-equivariant board model с exact/distance/risk heads на большом fit |

## 1. Joint reciprocal tri-emitter verifier

### Почему это не повтор

Текущий verifier оптимизирует row-listwise exact neighbour, а incoming
rank видит только как frozen scalar feature. Он не учится отличать
хороший row winner от many-to-one collision в column. Новая ветка сохраняет
те же fixed candidate identities, raw residual, ordered seam и DINO tokens, но
оптимизирует две стороны одного sparse assignment. Это не scalar rank
fusion, не confidence-threshold sweep на открытой local16 и не new emitter.

### Implementation-ready objective

Для каждой axis собрать один sparse logit tensor `z[i,j]` на fixed
raw+adapter+DINO union. Не делать отдельные model calls на edges.

1. `L_row`: cross-entropy по candidate targets каждого source socket.
   Добавить learned `NONE` logit для true border и случая, когда exact
   neighbour отсутствует в union; так missing 21.97% не превращаются в
   ложную positive.
2. `L_col`: та же cross-entropy по всем source rows, которые номинировали
   один target socket. Exact predecessor — label; для board border и absent
   predecessor работает тот же `NONE` contract.
3. Для candidate edge вычислить two-sided margins:
   `m_row = z[i,j] - logsumexp(z[i,k], k!=j)` и
   `m_col = z[i,j] - logsumexp(z[k,j], k!=i)`.
   Joint confidence — differentiable minimum
   `c = -tau*log(exp(-m_row/tau)+exp(-m_col/tau))`, fixed `tau=0.25`.
4. `L_conf` — BCE на exact-edge truth от `(c-b)/T`; скаляры `b` и
   positive `T=softplus(t)+1e-3` учатся только на FIT. Hard negatives
   обязательно включают row winners, column winners и many-to-one collisions.
5. Fixed first objective:
   `L = L_row + L_col + 0.25*L_conf + 1e-3*mean(delta**2)`.
   Архитектура, weights, endpoint и seed подписываются до FIT;
   не сравнивать nearby lambdas на одном DEV.

Первый deployable acceptance contract — reciprocal row/column top-1 с fixed
**top 5% `c` per axis per board**, а не post-hoc threshold. Эта точка
зарегистрирована потому, что у frozen diagnostic именно 5% дали
`+6.569 pp` precision; эта цифра не должна использоваться для
подбора ещё 3/10/20/30% variants.

### Быстрый тест и scale-up

- Capacity-only: synthetic `4×4` с column-collision distractors. Пройти только
  если row и column R@1 оба `100%`, positive `c` выше hard-collision
  `c`, а transpose/relabel invariants выполнены. Это plumbing, не quality claim.
- Cheap real discovery: один fixed small FIT и новый source-disjoint DEV,
  без local16 replay. Sensitive gate: R@1 `>= raw +0.5 pp`, R@5 `>= raw`,
  two-sided precision на fixed 5% coverage `>= raw +2 pp`, каждая axis
  nonnegative. Слабый плюс только в union coverage сохраняет model
  как emitter, но не открывает decoder.
- Server: тот же contract с большим FIT, несколькими заранее
  заданными corruption draws и vectorized full sparse row+column batches. Один
  endpoint, один seed, без checkpoint выбора. Final CONFIRM должен быть
  source-disjoint от FIT, всех Socket/adapter/DINO lineages и всех ранее
  scored panels.

### Гипотеза и promotion gate

- Pair: clean 5% reciprocal head должна добавить высокоточные
  anchors без native-tail pollution; это primary expected gain.
- Exact/distance: R@1 и fewer collisions должны слабо поднять exact/radius2
  или как минимум не ухудшать absolute Manhattan.
- SSIM: польза только через layout; denoised pixels в output не идут.
  Decoder confirmation: pairs mean `>=+2`, source-CI lower `>=0`, exact
  `>=-1 tile`, absolute Manhattan `<=0`, clean и frozen-same-tail SSIM
  nonnegative. Сначала freeze layouts, потом references.

## 2. Raw-preserving full-resolution matcher model

### Почему это не ещё один denoiser

Pixel-MSE/PSNR denoisers, independent per-tile NLM, downsampling DRUNet и
denoise-only replacement scores уже проваливались. Положительный сигнал дали
именно full-resolution, raw-preserving views: adapter400→1600 монотонно
улучшил R@1/R@5/union/reciprocal, а DINO добавил independent candidate
supply. Новая model оптимизирует **adjacency retrieval и reciprocal
assignment**, а не pixel fidelity.

Fixed conceptual contract:

- every block keeps `20×20` resolution; no U-Net pooling;
- raw Socket/DINO evidence and raw RGB skip are always available;
- independent-tile encoder produces ordered side tokens, then proposed pair gets
  a small cross-attention/joint-seam head;
- train-only clean tiles are teachers for boundary phase, but inference sees only
  the current corrupted tile bag;
- augmentations cover JPEG, blur, sensor noise, brightness/gamma, colour cast and
  resampling; paired views include both independent tile corruption and
  board-correlated photometric shifts;
- objective is the joint reciprocal objective from direction 1 plus
  clean/dirty side-token consistency. Pixel reconstruction may be an auxiliary
  with a bounded weight, never the primary loss.

### Быстрый тест, server version и gate

- Cheap: freeze raw/adapter/DINO union and train only a narrow side-token residual
  on one small new FIT/DEV. Pass if raw-union coverage is preserved, R@1 or fixed
  5% reciprocal precision improves, and neither axis is negative. A pure supply
  gain is retained as an emitter even when replacement R@1 is flat.
- Server: width `128–192`, deep stride-one NAF/ConvNeXt-style field plus boundary
  cross-attention; many full boards per update, online hard negatives and the
  same pre-signed corruption stream. Capacity comes from diverse sources/views,
  not an ensemble of correlated random seeds. The already signed
  [adapter scale3200](experiments/fullres-retrieval-adapter-scale3200-deferred.md)
  is a low-risk parallel supply measurement, not a substitute for this joint
  model and not a reason to alter its gate.
- Retrieval CONFIRM uses untouched sources and requires R@1/R@5 and fixed-coverage
  reciprocal non-regression plus at least one material positive. Decoder is
  opened only after this. Layout gate is the same as direction 1.

Expected effect: pair/radius2 first, exact modest, layout-only SSIM small but
positive if the solver converts edges. The restored view remains **matcher-only**;
the final image is assembled from all 576 original upright tiles and any output
restoration is evaluated in a separate frozen-tail experiment.

## 3. Compatibility-aware structured decoder

### Почему all-edge union провалился

The `-127.25` result has a precise mechanical cause, not merely a weak HGB:

1. HGB was trained on a relation **inside one realised arm/context**. Its
   probability is not the counterfactual marginal utility of forcing that edge
   into a different layout.
2. Max over up to six occurrences creates winner's-curse calibration and throws
   away which neighbouring relations made the occurrence plausible.
3. Deduplication still left about `3,992` unique edges for only `1,104` output
   slots. The fixed all-edge contract forced the low-confidence tail to
   participate instead of allowing abstention.
4. Independent probabilities ignore outgoing/incoming uniqueness, overlap,
   incompatible component translations, cycle closure and the two board cuts.
   Several individually plausible edges can be jointly impossible.
5. The raw-tail solver is path-dependent: an early false rigid merge welds
   components and changes every later action. Edge ROC-AUC does not measure this
   asymmetric downstream cost.

Therefore do **not** rescue that report with local threshold/top-k, probability
transform or arm-weight sweeps. A future consumer must treat scores as proposals,
include `reject`, and optimise joint compatibility/marginal layout gain.

### Fixed first structured formulation

- Start from the confirmed whole-arm layout, never an empty board.
- Immutable anchor set: realised control relations plus only the fixed 5%
  reciprocal verifier head. A proposed edit may remove a control edge only when
  the same move adds a strictly higher joint-confidence compatible relation.
- State is the current rigid-component graph. An action is one edge-implied
  rigid merge/translation or `STOP`. Reject collisions, duplicate
  outgoing/incoming sockets, inconsistent translation cycles and out-of-frame
  spans before scoring.
- Score an action by a context model over both components, all induced contacts,
  row/column confidence and lost cut relations. Train listwise against its true
  **incremental satisfied-pair delta** on FIT sources, not isolated edge truth.
  Pair loss from welding a false component is therefore visible in the label.
- Apply a bounded beam or best-first search with a preregistered action/beam
  budget. Preserve the untouched control as beam item 0 and return it on a score
  tie or no positive action.

Cheap capacity test: a synthetic `6×6` graph with compatible truth edges,
high-scoring conflicting distractors and a forced `STOP`. It must recover the
truth layout, reject a locally strongest incompatible edge, preserve relabel
equivariance and never score below control. Before a real model, a FIT-only
oracle must show at least `+8` compatible supplied true edges/board and a
pair-safe action ceiling; otherwise the branch stops before training.

The scalable version is a component-conditioned graph Transformer/policy with
batched candidate actions and larger source/augmentation coverage. This is not
the failed joint absolute-pose Transformer: it predicts a local, feasible,
pair-delta action and has a hard control/STOP path, rather than repacking many
components from a weak absolute shift.

Source-disjoint discovery gate: pairs `>=+1`, every axis nonnegative, exact
`>=-1`, absolute Manhattan non-increasing. New CONFIRM: pairs `>=+2`, source-CI
lower `>=0`, exact `>=-1`, Manhattan `<=0`, same-tail SSIM nonnegative, strict
permutations on every arm. The primary hypothesis is pair gain; exact and SSIM
are safety metrics until an origin model is added.

## 4. Pair-safe global-origin abstention

Relative geometry and absolute origin are separable failure modes. On the old
six-arm local layout the oracle best cyclic roll contained `71.94` exact
tiles/board versus `5.94` at the frozen origin. The fixed Socket roll sometimes
found the signal (`+42` exact for `-4` pairs on one disjoint board), but
unconditional transfer lost `-3.34` pairs overall. A selector trained on only
17 changed local boards had two safe positives and correctly failed. The idea is
not closed; the small-data unconditional/simple-feature consumer is.

### Materially new contract

- Input is the current relation-selector 24×24 layout, Socket border logits,
  tri-emitter two-sided confidence, component masks and six-arm agreement. No
  isolated tile-to-population atlas, face/centre/background heuristic or source
  identity.
- Candidate actions are only the 576 whole-board cyclic rolls plus `KEEP`.
  No tile/component repacking is allowed.
- A shift-equivariant board network predicts three quantities for every roll:
  exact/radius2 utility, absolute Manhattan change and pair-loss risk. Training
  labels come only from synthetic FIT references after target-free candidates
  are frozen.
- The policy abstains unless a lower confidence bound on exact utility is
  positive and an upper confidence bound on pair loss is within the signed
  budget. Calibration is source-grouped on FIT; no threshold selection on DEV.

Cheap first step is an **action-space feasibility audit on new FIT sources**, not
a model: under the fixed mean pair-loss budget `0.5`, hard-safe positive actions
must exist on at least 5% of boards and give oracle exact gain at least
`+1 tile/board`. If that ceiling is absent, stop. If present, run one small
shift-equivariant model and require OOF selected precision `>=50%` and positive
exact/distance utility before reserving CONFIRM.

The 40–60 GB version scales source count and corruption draws, uses a circular
CNN/ViT with board-wide receptive field and heteroscedastic/quantile risk heads,
and keeps the same action/abstention contract. It is distinct from the failed
whole-layout CNN by starting from the confirmed relation layout, explicitly
learning pair-loss risk and `KEEP`, and requiring a large source-disjoint fit
rather than promoting a tiny aggregate exact fluctuation.

Final gate: exact `>=+0.5 tile/board` with source-CI lower `>=0`, pair delta
`>=-0.5`, absolute Manhattan `<0`, radius2 `>0`, clean and frozen-same-tail SSIM
nonnegative. Cyclic-aligned distance is diagnostic only. Output remains one
strict roll of all original upright tiles.

## Source-disjoint and legal protocol for every direction

1. Recursively union every prior `*_filenames`/source roster from matcher,
   denoiser, solver and scored-panel lineages before selecting FIT/DEV/CONFIRM.
2. Sign exact sources, draws, corruption specs, architecture, seed, endpoint,
   metrics and gates before cache generation or scoring. One model endpoint and
   one policy per question; no nearby sweep on an opened DEV.
3. Materialise target-free scores/layouts and freeze their hashes before reading
   exact targets. Restore references only for evaluation.
4. Report absolute exact, mean Manhattan, radius2, satisfied pairs/1104 and
   clean/dirty/frozen-same-tail SSIM together. Never promote on cyclic-aligned
   distance or pair metric alone.
5. Every output layout must be an `int32[576]` strict permutation of the 576
   original upright `20×20` tiles. Denoised/restored representations are
   matcher-only in this solver roadmap; rotations, warps, replacement tiles and
   synthetic output pixels are forbidden.
6. Competition test and official submission remain closed until a separately
   authorised compliant same-tail end-to-end promotion.

## Recommended execution order

1. Implement direction 1 capacity test and one new small source-disjoint
   discovery. This is the fastest path to a number and directly addresses the
   current verifier's only failed gate.
2. In parallel on the large GPU, run the already signed adapter3200 supply
   continuation and prepare direction 2 at scale. Do not block direction 1 on
   adapter3200.
3. Open direction 3 only after a reciprocal verifier produces a fixed clean
   edge head; otherwise it would repeat low-precision solver work.
4. Run direction 4's FIT-only oracle audit in parallel. Train the origin model
   only if the pair-safe action ceiling exists.

This ordering keeps weak positive retrieval/supply signals alive while refusing
to spend a fresh confirmation split on an uncalibrated decoder.

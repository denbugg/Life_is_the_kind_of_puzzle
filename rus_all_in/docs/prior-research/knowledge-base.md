# База знаний: какие идеи уже проверяли

[К сводному индексу](README.md) · [Tile-sorter ledger](layout-sorter-ledger.md) · [M1–M420](generated/m-experiments.md)

Этот файл отвечает на практический вопрос: «идею уже пробовали или нет?».
Числа и протоколы находятся в углублённых отчётах по сериям. Метки:

- **keep** — подтверждённый полезный компонент;
- **partial** — механизм или proxy работает, end-to-end выигрыш не доказан;
- **reject-as-tested** — конкретная реализация/протокол дали отрицательный
  результат; повтор оправдан только новой причиной;
- **invalid** — результат нельзя использовать из-за leakage, bug или неверного
  control;
- **resource stop** — качество идеи не измерено;
- **open** — содержательный следующий эксперимент действительно не завершён.

## Поисковая матрица идей

| Идея / семейство | Где проверяли | Что установлено | Статус / условие возврата |
|---|---|---|---|
| Raw one-pixel seam, L1/L2/MSE | ORBIT R0; M1; P3/P31 controls | На dirty tiles low-rank слаб; на clean локальный signal заметен. | **reject-as-tested** как основной matcher. Сохранять как cheap control. |
| Ridge / variance+mean seam | M3, M19; ранний rank pipeline | Лучше plain MSE в отдельных режимах; per-tile normalization ухудшала. | **partial/control**, не самостоятельный путь к assembly. |
| MGC / Mahalanobis Gradient Compatibility | M17–M18, M35, M46, P4, P27, E2/E14 | Почти решает clean puzzle; на dirty alone слаб. Raw classical score полезен малым весом в E14. | **keep** как clean control и auxiliary cue; не повторять как единственный scorer. |
| Phase/derivative/handcrafted boundary families | ORBIT F1P, P20, M58–M59 | Небольшие локальные эффекты, hard-row recall не вырос. | **reject-as-tested**. |
| Photometric normalization / affine correction до matching | ORBIT PN1/PN2, M8, M31, M38, M52, M129–M132, M179, M185, M199, E15 | Простая normalization теряет signal; oracle affine помогает мало; seam levelling после layout полезен. | Pre-match **reject-as-tested**; post-layout offset levelling **keep**. |
| Tile denoising/restoration перед matcher | ORBIT D1/R4/R5 diagnostics; M6–M7, M21–M61, M209, M275–M304; E19 | Pixel error может падать при ухудшении matching; real residual correlated и border-heavy. Restore-input и repeated/NLM-per-tile routes не конвертировали. | **reject-as-tested** для ещё одного похожего denoiser. Новый target/evidence обязателен. |
| Restoration после фиксированного layout | ORBIT R4/R5/S1; M126, M147, M174–M184; E18/E18b; current frozen bakeoff | Официальный rank96→R5→NLM дал 0.237485. На frozen holdout-48 colored NLM `h=9`: 0.430621→0.557442, `+0.126821`, CI `[+0.114821,+0.138820]`, 48/48; gray guard хуже unguarded на 48/48 (`−0.003589`). | Colored NLM `h=9` **keep/promote** как tail. E18b gray guard **reject-as-tested** без внешнего safety-инварианта. Это не solver benchmark. |
| Larger restorer / больше steps / иной clean target | M54–M55, M279–M292; ORBIT R5/R6 proposal | MGC-restorer plateau; target swap/scale не закрыли разрыв. Broad R6 curriculum не выполнен. | Точный repeat **reject**; broad source-disjoint restoration **open** для output metric. |
| Siamese/listwise directional matcher | ORBIT R2/R2L/R3; M79–M87; CB1; V23 | Learned matcher сильно лучше raw analytic signal, но candidate ranking и transfer ограничены. | **keep** как базовая learned family. Менять labels/evidence, не только capacity. |
| Full-pair / cross-encoder reranking | ORBIT R8/R9, P24–P26, M20, M56–M57, M105–M113, V22/V24 | R8 выигрывает на matched synthetic, transfer на raw провален. V22 силён поверх V18 top-32. P24/P25 не дошли до checkpoint из-за memory/runtime. | **partial/open** только при streaming/vectorized raw-domain training и честном split. |
| Joint candidate/listwise chooser | M412/M419, V26/V27; current position-aware content verifier | V26 даёт малый устойчивый gain, V27 mixed gate. M419 почти не извлёк shortlist headroom. Новый verifier на scale128/calibration24 дал all exact `+1.079 pp`, но content≤20 `−3.378 pp` vs ensemble и `−4.789 pp` vs bilateral. Strict trusted exact/content `+3.246/+3.266 pp`, но после confidence filter content почти совпадает с exact и не доказывает content slack. | Глобальная content-multipositive formulation **reject-as-tested**; fresh holdout/decoder не запускались. Exact-edge signal — только **research auxiliary**, не production. Возврат требует нового inference-visible evidence/target, не capacity scaling. |
| Candidate union / multimodel diversity | ORBIT U1/R6U1; CB1; P23/P29; M158–M170/M201–M219; V25/V28; current analytic supply | Coverage растёт от независимых views. На trusted holdout union@5 exact/content≤20 right 0.4738/0.5010, down 0.5170/0.5424; union@32 — 0.7719/0.7970 и 0.7931/0.8182. | Generator diversity **keep** как supply gate; score averaging **reject-as-tested**. `union@32` фактически ~78–79 кандидатов, labels target-assisted, global/fixed-budget win не доказан. |
| Random-seed ensembles одной архитектуры | M208/M309/M363 | Ошибки слишком коррелированы; независимость приходит от входного view/модальности. | **reject-as-tested**. |
| Analytic filtered views | M363–M373 | Analytic filters обошли learned restorers как voters; M371 изменил default. | **keep** как дешёвая diversity axis; roster settings затем насыщены. |
| Contours на fragment scale | M368–M370 | Contours переживают corruption на крупном масштабе, но fragment adjacency отвергнута; layout judge имел signal без conversion. | Alone **reject-as-tested**. |
| Contours как мультимодальный retrieval input | V28 | В fusion V27+V28 улучшены все retrieval metrics; standalone прежде всего расширяет top-32. | **keep**, лучший V retrieval block. |
| Reciprocal/best-buddy/margin edges | ORBIT F1/F2/Q1; E1/E4; M11–M16, M41; P34 | Можно получить очень чистое маленькое ядро, но объёма/связности недостаточно; fixed bonus seed-unstable. | **partial diagnostic**; не использовать как полный solver. |
| 2×2 cycles / loop closure / corroboration | ORBIT C1/G2/G2b; M11–M15, M189/M193, M248–M251, M318–M335, M404; P9/P12/P14/P34 | Иногда резко повышает precision, но starves coverage или проверяет уже сделанную ошибку слишком поздно. | **reject-as-tested** как bootstrap/selector. Closed-loop merge полезен лишь на чистых islands. |
| Greedy / Paikin-Tal / Kruskal growth | M10–M16, M153, M205–M207 | Ошибки каскадируют; growth ниже knee не конвертирует. | **reject-as-tested**. |
| Component packing / islands | ORBIT R10; M148–M181, M213–M251, M378–M418 | Packer впервые двигал placement; island purity предсказывает merge. На реальном seed рост крупного clean block почти не меняет итог. | Packer primitives **keep**; ещё один peripheral growth rule **reject**. |
| RL/DAgger attachment policy | M411–M413 | Policy улучшает precision-volume frontier и умеет не двигаться, но на shipping seed добавляет ~4 correct bonds и не растит block. | Mechanism **partial**, место применения закрыто. |
| Merge policy / STOP / UNDO value | M414/M418 | Первый merge+split run invalid; fixed value забирает выигрыш action policy, undo хуже stop. | **reject-as-tested**. |
| Max-contact вместо mean-contact merge | M415/M416 | На clean stand block 30.9→38.9, но в real pipeline growth ухудшает placement. | **partial diagnostic**, не доказанный shipping gain. |
| Simulated annealing | legacy/V11/E baseline/M360–M361 | Работает как baseline; старый M annealer фактически не annealed — 93% moves invalid, ни одного uphill accepted. E3 Cython даёт exact 3.4× speedup для старого SA. | Baseline **keep**; прежний move class **reject**. |
| Equal-wall-clock multistart SA | E9 | Остановлено на 3/32, implementation/artifact не сохранены. | **resource stop/open**, но низкий приоритет. |
| LP translation synchronization | M44–M53 | Exact на oracle, лучше greedy на clean+blur; требует около 0.9 edge precision и не спасает current dirty scores. | Solver primitive **keep** выше activation threshold. |
| Relaxation labeling | M122; E11/E14; P15/P36 | E11 alone neutral/negative, E14 fusion+relaxation лучший offline E-layout; M122 малый refiner gain. P36 runtime stop. | **keep** как E14 component; не повторять standalone. |
| CP-SAT local repair | E12; P16–P18 related exact/beam work | E12 ухудшил target metrics; exact swap arithmetic P17 корректна, но runs stopped по времени. | E12 **reject**; vectorized exact polish **resource-limited partial**. |
| Genetic solver | M111–M112 | Полезен выше knee, но не двигает сам knee. | **partial**, только с улучшенным score. |
| Belief propagation | M64 | Oracle работает, real scores collapses к chance, uniqueness enforced слишком поздно. | **reject-as-tested**. |
| Sinkhorn assignment/training | P5/P10/P11; M85–M87/M116/M403; V29 | Calibration Sinkhorn полезен форме matcher costs; direct/global relaxations провалены. M403 four-seed training gain = 0. | Calibration **keep**; solver/training add-on **reject-as-tested**. |
| Spectral seriation / diffusion graph | M293–M299 | Oracle mechanism существует, real data слаб; global-averaging family закрыта. | **reject-as-tested**. |
| Portfolio + learned solver selection | V29; M356–M358/M374 | V29 OOF +5.47% composite на 15 cases; generic board judges в M fail. | **partial**, нужен больший CV и content-aware metric. |
| Graph coordinate/unary LNS | V30; P28 | P28 edge-conditioned coordinate denoiser провалил capacity gate. V30 weak absolute unaries + LNS дали лучший V composite; direct placement 0.150→0.197%, но translation-aligned 2.18→2.13%, final caches уже просмотрены, edge calibrator отключён. | V30 **partial/comparator**, не default; нужен новый CV и direct SSIM gate. |
| Set-to-grid / absolute slot Transformer | ORBIT G1/PGA1; P5/P10/P11/P32/P37–P39 | Малые models около chance; raw Transformer не учится; masked pretraining fit-ит train и не переносится source-disjoint. | **reject-as-tested**; не масштабировать тот же CE. |
| DINO/foundation features | P29/P30/P32/P35; Russian DINO 4×4; 2026-08-30 component-field audit | DINO расширяет candidate coverage, но direct score/absolute position не конвертируют и memorise source. Сумма isolated-tile DINO→population-field scores по rigid buddies96 component признана тем же information family и остановлена до target decode; [evidence](../experiments/foundation-semantic-component-stop.md). | Candidate generator **keep**; absolute/global heads **reject-as-tested**. Новый semantic arm обязан jointly видеть actual component/board, а не только суммировать tile-wise absolute votes. |
| Coarse colour field / thumbnail | M134–M142, M161–M177, M387–M391 | Oracle fields имеют payoff; prediction из bag слаба, anchoring/placement не достигается. | **reject-as-tested**. |
| Frame/border prior | M228–M247/M343/M390–M401; V30 border head | Первый слабый absolute cue; M343 сильно помогает oracle failures, но ломает часть boards; generic photo prior закрыт. V30 border unary полезен в joint solver. | **partial/keep only jointly**, не самостоятельное решение. |
| Per-board routing/judges | M344–M358; E14 production gate | Summary features смешивают board texture с quality; M policies fail. E remote gate ещё и реализован неверно. | Generic routing **reject**; direct matched validation selector **open/engineering**. |
| Source-image retrieval and reassembly | ORBIT SA1/SA2/SA3 | При правильном source SA1 tile agreement 84.79%; SA2 retrieval R@1 94.24%, strict held verifier 100% true accept/0% wrong accept. Известно лишь 18/700 test sources, более широкий crawl новых не нашёл. | **conditional high upside**: только после письменной проверки правил, отдельный overlay и неизменный strict verifier. |
| Content-equivalent / visual-twin target | M68, M125, M420; current content substitution/supply/verifier | M125 first implementation была defective. Строгий holdout-48 Hungarian derangement: clean 0.533053, recovered dirty 0.258824, dirty+NLM 0.383725, placement/reuse 0. Candidate content lift при trusted union@32 лишь ~2.5–2.7 pp для RMSE≤20. Первый global content-multipositive verifier регрессировал на all rows. | Metric slack при биекции **confirmed/keep diagnostic**. Target-free recovery остаётся **open**; clean/dirty oracle score не deployable. Текущая verifier formulation **reject-as-tested**. |

## Невалидные и отозванные результаты

Не использовать эти headlines при принятии решений:

| Результат | Почему недействителен | Что считать authoritative |
|---|---|---|
| P8: 100% scorer | True neighbour всегда был candidate slot 0. | Leakage; все P8 artifacts навсегда исключены. |
| P20 attempt 1 | Обратная permutation mapping; один JSON оставил противоречивый boolean. | Только corrected metrics + rejection note. |
| P12: «−0.542535 pp» | Ошибка единиц ×100. | −0.005425 percentage points; reject не меняется. |
| M9 и более ранние raw placement claims | Torus origin/cyclic shift не учтён. | `fix_origin` или best-shift-aware numbers. |
| M141/M303 information impossibility | Label/matching premise был артефактом. | M304: current stack решает perfect input. |
| M258 gain | Не реплицировался. | Withdrawn. |
| M264 Hungarian improvement | Degenerate tie передал ответ. | Полностью отозвано. |
| M296 interim dimension gain | Unmatched control и run noise. | M306 reject. |
| M339 impossibility global energy | M340 построил objective, предпочитающий truth. | Направление возможно, реализация M341 не конвертировала. |
| M403 +3% Sinkhorn | Single seed ниже уже измеренного noise floor. | Four-seed mean +0.0036±0.0033: ноль. |
| M417 doubled placement | 24-board noise. | 48-board correction: parity, не win. |
| E14 «готов для Kaggle» | Offline и production используют разные scorer/domain. | Offline E14 local winner; production требует score-matched revalidation. |
| E18b «fallback to v5» | При отрицательном validation delta код отключает guard, но всё равно запускает E14/E18b. | Fallback не реализован. |

Полный список M-corrections находится в [M-аудите](m-series.md), P/R-corrections
— в [CB1-аудите](cb1-orbit-r-p.md), E-caveats — в
[E-аудите](e-series.md).

## Идеи, которые не провалились, а не были измерены

Их нельзя закрывать отрицательным ярлыком:

- E9 equal-wall-clock multistart — остановлен на 3/32, нет реализации/artifact;
- E13 corruption-aware border encoder больше не open: локальный bounded pilot
  (256 train, 400 full-576 updates, fresh exact eval16) дал R@1 `6.878%` против
  `18.654%` у frozen d64 OT; fixed fusion и reciprocal precision также
  ухудшились, поэтому global gate не открывался;
- E20 больше не open: current source-disjoint exact 256/16 pilot подтвердил
  restored candidate supply (`+2.978/+2.763 pp` top32 coverage), но residual
  ranker не улучшил R@1 и проиграл matched precision; decoder не открывался;
- P14d symmetric topology, P17/P18 exact polish, P24/P25 cross-ranker,
  P33 learned agglomeration, P36 soft propagation — resource/runtime stops;
- SA3 как формальный gated run не выполнен, но более ранний crawl 19 679
  public photos уже не нашёл новых test matches; те же каталоги не повторять;
- R6 broad-scene restoration curriculum — предложен, не выполнен;
- V30 на новом untouched terminal split — такого split после V28 уже нет.

Исторически незакоммиченный M420 `content_top1.py` больше не является открытым
пунктом: one-to-one substitution, content-aware supply и verifier теперь
проверены независимо в текущем workspace. Authoritative итог находится в
[реестре экспериментов](../experiments/README.md).

## Самые перспективные направления

### 1. Выполнено — единый generic-only protocol

Frozen manifest зафиксирован: `train/calibration/holdout = 5600/700/700`, seed
`20260829`, protocol digest
`2a9e3b74f7defa8c00846a05eb598fd263fd16c2787c70e77d3b7a4b585bfbf4`.
Каждый текущий результат хранит selection digest, filenames и image/code hashes.
Competition test и 18 historical clean refs исключены. Это закрывает
инфраструктурный пункт, но не создаёт новый generic solver; platform anchor
по-прежнему S1 `0.237485` с прежними artifact caveats.

### 2. Выполнено — M420 с bijection control

Metric mismatch устойчив к one-to-one ограничению: holdout-48 clean
derangement `0.533053`, dirty+NLM `0.383725`, exact placement и duplicate use
равны нулю. Clean nearest-other выше всего на `0.039906`, несмотря на средний
reuse 262.9. Направление больше не требует повторной проверки биекции.

Открытым остаётся другой вопрос: какой **inference-visible** semantic signal
может находить полезные substitutes без clean target. Пока такого механизма нет,
content metrics — diagnostics/labels, не submission score.

### 3. Выполнено — fixed-layout pixel-tail bakeoff

Colored NLM `h=9` выбран на calibration-48 и подтверждён holdout-48:
`+0.126821`, CI `[+0.114821,+0.138820]`, 48/48. Его следует использовать как
default output tail будущей сборки и повторить paired control на фактической
раскладке solver-а. Gray guard закрыт (`−0.003589` против unguarded, 48/48
losses); Gaussian `sigma=1` остаётся latency fallback.

Новый large restorer имеет смысл только как отдельный source-disjoint output-
metric эксперимент, а не как повтор уже отвергнутого denoise-before-matching.

### 4. Выполнено — analytic supply и первый multi-tile verifier

Supply gate пройден: trusted holdout union@32 достигает примерно 77–79% exact и
80–82% content≤20 recall, но фактический pool равен ~78–79, labels
target-assisted, global consistency не измерена.

Position-aware scale128 verifier не прошёл следующий gate. На свежих 24
calibration boards all exact вырос на `+1.079 pp`, но content≤20 снизился на
`−3.378 pp` против ensemble и `−4.789 pp` против bilateral. Right/down
регрессируют против bilateral на `−3.729/−5.850 pp`. Strict trusted
exact/content `+3.246/+3.266 pp` недоступны как test gate, а content после
confidence filter почти совпадает с exact и не подтверждает content slack.
Поэтому новый holdout, decoder/QAP/LNS и SSIM не запускались; ту же
content-multipositive formulation наращивать нельзя. Следующий возврат возможен
только с новым inference-visible evidence, а exact-edge head сохраняется
research-only.

### 5. P1 conditional — source retrieval как отдельный compliance-трек

SA2/SA1 имеют сильную conditional accuracy, но известно лишь 18/700 test
sources, более широкий crawl новых не нашёл, а чистые references являются test
labels. До работы нужны явное разрешение регламента и provenance/licensing.

Всегда публиковать отдельно `generic_only` и optional `public_source_overlay`,
не ослаблять strict verifier и не использовать 18-reference scorer для model
selection. Высокая accuracy при найденном source не равна высокому coverage.

### 6. P2 — новые scorer-ы только после предыдущих gates

Boundary-only cross-ranker имеет низкий приоритет: V22 помог внутри top-32,
V24 провалился, R8 не перенёсся на raw, P24/P25 остановились по resource, M419
показал насыщение same-seam chooser. Возврат оправдан лишь с действительно
новым evidence: full-tile/multi-tile semantics, target с проверенной полезностью
на inference-relevant `all` rows, streaming/vectorized raw-domain training и
cheap source-disjoint gate. Простое повторение текущих content multi-positives
не считается новым механизмом.

P24/P25 остаются научно неизмеренными, но «не измерено» само по себе не означает
высокий expected value. E13 и E20 ranker отдельно закрыты текущими bounded
source-disjoint local gates как measured-negative; E20 restored emitter остаётся
reusable candidate supply.

## Decision rules для новых экспериментов

1. Сначала назвать целевую метрику: exact identity diagnostic,
   content-equivalent diagnostic или официальный full-image SSIM.
2. Не сравнивать числа разных split/cache/renderer как одну таблицу.
3. Candidate coverage без ranking gain не разрешает solver run автоматически.
4. Proxy gain должен пройти matched whole-pipeline control; M-серия многократно
   отзывала выводы именно на этом переходе.
5. Single-seed gain меньше измеренного noise floor — гипотеза, не результат.
6. Resource stop не превращать в scientific reject.
7. Любой learned component обязан иметь source-disjoint split и anti-leakage
   audit; source memorization уже поймана в P32/P35/P39.
8. Перед новым training искать reusable code/checkpoint/cache по карте серии;
   отсутствие artifact в Git фиксировать, а не молча переобучать «по памяти».

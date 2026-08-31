[← Общий обзор предыдущих исследований](README.md)

> Углублённый аудит ветки `origin/autoresearch/pazzle-fixed-orientation-cb1`: полная история ORBIT, R-серии и P1–P39.

# Аудит `origin/autoresearch/pazzle-fixed-orientation-cb1`

Дата аудита: 2026-08-29. Репозиторий: `/Users/rusyalain/Documents/GitHub/pazzle_will_be_killed`. Ветка исследована read-only, без checkout и без изменения research-репозитория.

## 1. Охват и способ проверки

- Общий предок с `origin/Taska-govna`: `d28136151f17161ecdf791dfc456ceea2f6e4fa0`.
- Диапазон аудита: `d281361..9bd8db1`.
- В диапазоне **338 уникальных коммитов** (в исходном задании было приблизительное «~339»), все по first-parent, merge-коммитов 0, пустых коммитов 0.
- Первый: `a0879e6` от 2026-08-13 18:49 +03; последний: `9bd8db1` от 2026-08-17 22:47 +03.
- Для каждого коммита просмотрены subject и parent-diff/name-status; затем результаты сверены с итоговыми `FINDINGS.md`, `EXPERIMENTS.md`, `PLAN.md`, отдельными pre-registration/evidence/rejection/stop-файлами, JSON-отчётами и кодом на tip.
- Совокупный diff ветки: **339 новых файлов, 31 349 строк**, без удалений относительно базы: 191 файл в `autoresearch-runs/...` и 148 в `src/`; по типам — 136 Markdown, 54 JSON, 1 log, 131 Python, 17 PowerShell.
- Непрерывное покрытие всех 338 коммитов приведено в §12. Диапазоны там не имеют дыр или пересечений и в сумме дают 338.

## 2. Главный вывод

Ветка проверила почти весь естественный набор подходов к fixed-orientation сборке: локальные seam-фичи, pairwise/listwise нейроскореры, candidate retrieval, циклы и reciprocity, component/pose/QAP/relaxation solver-ы, set-to-grid и absolute-coordinate модели, DINO-признаки, raw-image Transformers и masked pretraining.

Устойчивый результат таков:

1. **Сильнейший подтверждённый platform anchor** — pipeline `rank96 → R5 → NLM`: официальный SSIM **0.2374852573**. Зафиксированный S1 runner не задаёт source overrides. Старый rank96 **0.2161981413** содержал 18 exact overrides, поэтому разница **+0.021287116** не является чистым paired generic gain и приводится только как historical comparison.
2. **Самый сильный conditional структурный маршрут** — восстановление по найденной исходной фотографии. SA1 даёт 84.79% tile-to-slot agreement при правильном source; SA2 даёт retrieval R@1 94.24% и строгую held-out верификацию 100% true accept / 0% wrong accept. Ограничения — разрешённость и покрытие корпуса: известно лишь 18/700 test sources, а более ранний crawl 19 679 public photos новых hits не нашёл.
3. **Кандидатов часто удаётся добавить, но не ранжировать.** U1, CB1, P23 и особенно P29 увеличивали candidate coverage, однако score fusion / ranker / solver не превращали это в placement или SSIM.
4. **Solver-only преобразования исчерпаны на текущем сигнале.** Циклы, reciprocity, pose synchronization, topology pruning, relaxation и multistart либо ухудшали метрики, либо были инертны, либо останавливались по времени. Новому solver-у сначала нужен лучший edge signal.
5. **Глобальные semantic/absolute heads переобучаются на source.** P32/P35 хорошо или приемлемо fit-ят FIT-train, но падают почти к случайному уровню на source-disjoint selection.
6. **Raw Transformer без pretraining не обучился вовсе; masked pretraining обучил FIT, но не перенёсся.** P38 после 7 680 updates остался около random; P39 поднял train Top-20 до 38.55%, но selection дал 3.54% при random ≈3.48%.

## 3. Протоколы и важные оговорки о числах

### Данные и разделения

- Геометрия везде фиксирована: 24×24, 576 upright tiles, 20×20 px, rotation search/augmentation запрещены.
- Полный train: 7 000 input/target PNG. Пинованный source split описан как `fit=5360 / cal=670 / dev=670 / reserve=300`, но сам JSON manifest **не закоммичен**; он находился на `E:\pazzle_work\...`.
- В R5/R11 зафиксирован SHA-256 split manifest: `a858a194ceab9976b72069aef6c46481734ce15594f67ae6818b4d7bfe30231a`; без внешнего файла можно проверить ссылочную целостность отчётов, но не восстановить список source names.
- Поздняя P-серия обычно использует из FIT фиксированные `96 train / 32 selection / 32 held`; P9–P13 — `128 train / 32 held`. CAL/DEV/test открывались только по gate-ам.
- Один разрешённый CAL board: `img_000051.png`. Пинованные DEV-8: `000008, 000014, 000020, 000033, 000048, 000057, 000064, 000081`.
- Ранние ORBIT R0–G3 преимущественно измерялись на synthetic corruption/source-disjoint boards; более поздние R8/R9 и P-серия отдельно выявили сильный synthetic→raw domain gap.
- Нельзя напрямую сравнивать coverage около 74% в U1, 65% в raw rank96 DEV и 14% в позднем P10/FIT-cache протоколе: это разные candidate constructions, split-ы, width/valid-mask semantics и определения метрики.

### Вычисления и seeds

- Исходный контракт: RTX 2070, `C:\Python313\python.exe`, артефакты на `E:`, начальный лимит 3 GPU-hours / 6 экспериментов / 30 min.
- Явно зафиксированные seeds: CB1 full train `20260814`; P10 `20260814`; P11 `20260816`; P15 auxiliary starts `20260816/17/18`; P19 `20260817`; P39 `20260817`; ранние G2/G2b code defaults `240815`. Для многих одноразовых gate-ов журнал говорит только «deterministic/source-keyed», без числового seed; выдумывать его нельзя.
- Все P10+ learned gates заявляют FP32/AMP off; R5 тоже был переведён в FP32 после NaN в MS-SSIM под AMP.

### Уровни доказательности

- `G0/G1 PASS` часто означает лишь правильную форму tensor-а, equivariance, deterministic hash или способность fit-нуть 1–2 boards. Это **не** evidence общего качества.
- `STOPPED/ABORTED` по времени/памяти не означает, что гипотеза опровергнута.
- Локальные SSIM на 8 DEV boards не равны leaderboard SSIM. Только S1 имеет официальный platform score.

## 4. Леджер раннего ORBIT: retrieval, pair scoring и consensus

| ID | Гипотеза / протокол | Количественный результат | Вердикт |
|---|---|---|---|
| R0 | Raw one-pixel boundary baseline, 8 source-disjoint boards | R@1 9.8902%, R@5 21.5014%, R@20 35.2468% | Базовая точка |
| R1 | Untrained multi-band handcrafted cosine fusion | R@20 25.9964%, −9.2504 pp к R0 | Reject |
| R2 | Learned directional Siamese, 200 steps | R@1 5.9047%, R@5 18.4556%, R@20 39.7758% | Partial: coverage↑, low-rank↓ |
| R3 | 200-step listwise hard-negative ranker | true-edge candidate coverage 68.8179%; proxy R@1 7.7958%; reciprocal mutual candidate coverage 89.8438% | Retain только как union generator |
| G1a | 6×6 coarse set prior, 200 steps | Hungarian membership 2.95% vs chance 2.78% | Inconclusive short run |
| G1b | То же, 1 200 steps | Hungarian 3.4288%, top64 group coverage 13.1510% | Reject visual-only global prior |
| F1 | Direct/non-direct + direction classifier | mutual candidate coverage 91.2639%, reciprocal precision 3.3840% | Reject as selector |
| F2/F2b | Frozen PairwiseNet + F1 fusion, K sweep | top1 precision 41.84% при recall 10.91%; dense precision 4.01% при recall 66.89%; ни один operating point не прошёл | Reject for assignment |
| C1 | Exact oriented 2×2 cycle availability, 2 boards | true motif coverage 1.51% at 128 / 2.93% at 512 motifs per anchor | Reject до реализации |
| R2L | Directional Siamese scale, 800 steps; best step 600 | R@1 9.81%, R@5 26.06%, R@20 49.88%, b384-neighbour 7.39% | Retain checkpoint only as proposal source |
| U1 | R3 top64 ∪ R2L top8/direction, 8 boards | coverage 69.34→73.95% (+4.61 pp), density +11.16% | **Pass candidate-source gate** |
| U2 | Frozen pair/pose scoring U1, 1-board smoke | top4 precision 18.27%, recall 19.07% vs 35/20% gate | Reject scorer; U1 remains recall-only |
| D1 | MatchDenoiser before simple seam ranking, 2 boards | tile L1 0.07790→0.07104, seam R1 13.7→13.5% | Reject as matching feature |
| R3L | Scaled full-row listwise ranker | no first step in ~10 min; 12.1 GB RSS / 20.6 GB committed | STOPPED, no quality claim |
| P1S | n=4/K=16 hard-negative micro-cache | no cache in 5m29s | Reject implementation/timing path |
| ORBIT-P2 | Existing posterior seam marginalization, 192 held rows | raw R1 17.19%; best posterior/hybrid ≤16.67%; R5 up to 42.19% but R1/Brier fail | Reject reuse |
| E2 | Streaming continuation predictor | valid first step, candidate coverage 76.28%, but 26.57 s/iteration | STOPPED for runtime, efficacy open |
| OH1/OH2 | Online random-reservoir hard PairwiseNet, 200 steps, then U1 smoke | aux best 54.69%; downstream top1 28.82%, top4 18.36%, recall 19.16% | Fast trainer retained; scorer rejected |
| OH3/OH4 | U1-aligned M=16 hard rows | aux 32.81%; downstream top1 30.38% but top4 18.32%, recall 19.11% | Local top1 success; production reject |
| OH5/OH6 | U1-aligned M=64 full row | aux 32.81%; downstream top1 29.17%, top4 17.93%, recall 18.70% | Reject; larger list did not help |
| Q1 | Scene-conditioned confidence, 4 fit /2 cal /2 held | no threshold; reciprocal+both-affinity diagnostic 33.33% precision at 14.06% row acceptance / 4.69% exact-edge coverage | Reject routing; sparse diagnostic only |
| G2 | F1-routed 2×2 closures, K=8/16/32, 2 boards | best precision lift 1.12×, recall 1.20%; thousands of closures but uninformative | Reject |
| PN1/PN2 | Per-tile photometric normalization, matched train/infer | aux 18.75%; downstream top1 23.09%, top4 15.15%, recall 15.81% | Reject; PairwiseNet family retired |
| GC1 | Same-bag whole-board critic, 400 steps / 4 boards | near-swap acc 31.25%, macro 55.56% | Reject global energy |
| G3 | Instance-conditioned latent canvas, 600 steps /4 boards | canvas L1 ≈0.224, placement top1 ≈0.2–0.3%, top20 ≈3.7% | Reject architecture |
| G2b | R2L-native directional 2×2 closure | p 3.08→3.10% at K4; 3.26→3.28% at K8; later K too slow | Reject consensus independently of F1 routing |
| F1P | Phase/derivative/value seams, 4 boards | best norm-value R@20 19.72%; phase reciprocal p 11.95%, recall 1.86% | Reject deterministic family |

Итог этого блока: high-recall graph строится, но directional precision недостаточна. Простые циклы не фильтруют ошибки, PairwiseNet насыщается около 18% top4 precision, а restoration нельзя использовать как seam feature.

## 5. Леджер ORBIT: source retrieval, restoration и сильные R-ветки

| ID | Протокол | Результат | Вердикт |
|---|---|---|---|
| SA1 | Correct public clean source; 218 linked cases, 51 held; robust matching + Hungarian | held tile agreement 84.79% (q10 75.87%); source canvas SSIM 0.9909; true-vs-hard-distractor margin positive 96.08% | **Capability pass**; требует verified source |
| SA2 | Dirty-bag source retrieval, 139 event-held queries; OOF confidence; strict SIFT/Hungarian on 167 cal +51 held true/wrong pairs | retrieval R@1 94.24%, R@50 100%; OOF accepts 92.09% at 97.66% precision; strict held true accept 100%, wrong accept 0% | **Pass, coverage-limited**; пороги не ослаблять |
| SA3 | Расширение законного public-source corpus | Формальный gated run в этой ветке не запускался; было 18 independently verified test overrides. Cross-branch audit: crawl 19 679 public photos новых hits не нашёл | **Conditional open**: только новые разрешённые источники, прежний crawl не повторять |
| PGA1 | 576×576 set-to-slot Transformer + Sinkhorn/Hungarian; two-board overfit controls | stochastic corruption 40.19%, fixed-corruption 11.55% vs required 95%; synthetic SSIM 0.254/0.376 нельзя считать DEV | Reject до DEV |
| SGT1 | 1.08M sparse score-only graph Transformer | covered candidate capacity 100% on 2 FIT; source-disjoint DEV covered top1 −4.93/−3.43 pp; coverage ceiling 68.44% | Reject score-message model |
| R4 | Frozen MatchDenoiser strictly after one frozen rank96 layout, DEV-8 | raw mean 0.10620→0.16205; +0.05585, lower-95 +0.03681 | Retain post-layout only |
| R5 | RestoreNet(base32, depth4), FP32 MS-SSIM+L1; 2-scene capacity + matched DEV-8 | capacity SSIM 0.733509 vs dirty 0.482370/R4 0.575197; matched DEV raw 0.104760, R4 0.160012, R5 0.185030; R5 raw delta +0.080270, lower-95 +0.047606; R5−R4 +0.025018 | **Retain strongest restorer** |
| R5 composition | Same inferred DEV boards: raw/NLM/R5/R5→NLM/NLM→R5 | canonical NLM 0.195530; R5 only 0.185030; NLM→R5 0.213214; **R5→NLM 0.230917**, +0.035387 vs NLM, lower-95 +0.024860 | Select R5→NLM |
| S1 | 700 test images, frozen rank96→R5→canonical NLM, deterministic ZIP, без configured override directory | official SSIM **0.23748525732559034**; historical rank96 comparator **0.2161981413457065** включал 18 overrides | **Production/platform anchor**; абсолютный score valid, delta не paired generic |
| SGT2-V | Visual patch features + sparse graph reranker, 600 CUDA steps | frozen covered top1 23.06% vs model 15.92%, −7.14 pp; coverage 65.10% | Reject supervised visual residual |
| CP1 | Per-tile affine RGB consensus from mutual rank96 edges | CAL chose alpha=0; DEV 0.23060→0.23060; positive alpha worse; gains/offsets saturated | Reject |
| QAP1 | Existing seeded QAP on perfect synthetic R/D | placement 24.83%, oriented neighbours 58.42%, DS error 0.99993 | Reject implementation at synthetic G0 |
| R6U1 | R2L ∪ rank96 at active width 128, pinned DEV | coverage 65.10→66.78% (+1.68 pp), active density 128→105.37; gate 73% | Reject before ranker; earlier adapter runs invalid |
| R7 | 475k-param full-board two-tower InfoNCE; 1 200 FIT steps, 32 CAL | R@20 47.5062% vs matched R2L 47.8346% (−0.3284 pp) | Reject factorized embeddings |
| R8 | Joint full-pair CNN; 2 000 FIT steps, 32 synthetic CAL, then 2 raw frozen DEV caches | synthetic R@20 58.7990%, +10.9644 pp vs R2L; raw R8-only coverage 22.5091%, union 66.0779% vs base 65.1042% (+0.9737 pp), below 73% | Architecture signal real, **raw transfer reject** |
| R9 | Fine-tune R8 on 17 raw FIT caches, evaluate raw CAL-51 | loss 5.510→2.750; R@20 3.1703%, coverage 21.8297% vs 20/50% gates | Reject naive small raw adaptation |
| R10-A | 32 multistart component packings; oracle G0 then DEV-8 | perfect synthetic recovery; frozen edge objective +4.190589, but raw-layout SSIM delta −0.002510, lower-95 −0.006608 | Reject objective, не всю spatial optimization гипотезу |
| R11 | 32-layout rank-normalized loop selector; oracle, CAL-1, DEV-8 | oracle selects exact; CAL picks λ=0 and canonical layout; DEV selects index 0 on all boards, SSIM delta 0 | Reject as no-op |

## 6. Полный леджер numbered P1–P39

Общий поздний протокол: FIT-only до прохождения gate, targets PNG закрыты, P8 запрещён после leakage-аудита, fixed orientation, canonical solver не меняется без отдельной регистрации.

| P | Идея и gate | Результат | Статус / что сохранять |
|---|---|---|---|
| P1 / CB1 | Matched-corruption BoundaryBuddy CNN. G1: 240 steps, 5 356 FIT +4 held hard-list queries; full: 6 000 steps, seed 20260814. G2 CAL candidate union; G4 ranker-rescored layout | G1 R@20 38.54→72.14%; CAL coverage 75.4076→77.8080% (+2.4004 pp). Но G4 C={0,16,32,48} дал одинаковый CAL SSIM 0.248863, tie selects C=0 | Candidate model/caches reusable; **overall rejected**: new edges get no solver support |
| P2 | Direct rank-normalized CB1 score fusion, CAL alpha grid 0–0.4 | alpha0 SSIM 0.262123; every positive alpha worse, до 0.228191 | Reject before DEV |
| P3 / CDCS | 2-pixel boundary listwise scorer on exact rank96 32-way rows; 96 train /32 held, 2 000 steps | top1 8.4961% vs L1 8.2520%, +0.2441 pp vs +5 pp gate | Reject before CAL; hard-list cache reusable |
| P4 / MGC-MB | Mahalanobis gradient compatibility on 128 FIT caches | top20 24.6179% vs L1 38.5657%; mutual precision 5.9039% vs 16.7935% | Reject analytic local gradient family |
| P5 | 6-block width192 permutation-invariant Set-to-Grid, 4 000 steps, 256 train +32 held | loss ≈ln576; Hungarian 0.1788% vs independent CNN 0.2222%, gate >10% | Reject |
| P6 | Conditional positional diffusion, 8 000 steps, 32 held | held placement 0.22786% vs independent 0.13563%, only +0.09223 pp; gate 1% and +0.5 pp | Reject; position state has weak real signal |
| P7 | 12 000-step paired clean/corruption contrastive+reconstruction encoder, 32 source-disjoint held | embedding top20 84.7005% vs RGB L1 69.9653% (+14.7352 pp); decoder L1 0.073938 worse than identity 0.072262 | Registered conjunctive gate reject; **contrastive encoder signal remains useful** |
| P8 | Frozen P7 + context-aware virtual halos over rank96 candidates; 160 FIT cache, 128/32 | context and local-only both reported 100% held top1 | **INVALID / permanently excluded**: true neighbour was candidate slot 0 for 100% rows; trivial index prior gets 100% |
| P9 | Leakage-audited rank96-only 2×2 loop reweight; 128 train selects λ, 32 held once | λ=0.40; held placement 0.189887→0.179036%, −0.010851 pp; 0 invalid | Reject before CAL |
| P10 | Layout-conditioned Fourier slot Transformer + 20-step Sinkhorn; seed 20260814; 12 epochs, 128/32 | machinery/equivariance/bijection pass; held 0.189887→0.173611% | Reject absolute refiner |
| P11 | Conditional generated canvas + adaptive Sinkhorn; seed 20260816; 16 epochs, 128/32 | G0 conditional/bijection pass; held 0.189887→0.168186% | Reject global-canvas assignment |
| P12 | Sparse 2×2 loop support over canonical width128, λ grid on 128/32 | train chooses .05; held 0.189887→0.184462%, 0 invalid | Reject. Correct delta is **−0.005425 pp**; journal text’s “−0.542535 percentage points” has a ×100 unit error |
| P13 | Robust component-pose translation synchronization + Hungarian; threshold grid 128/32 | all thresholds train 0.227865%; held 0.222439%, +0.032552 pp vs 0.189887, far below +3 pp gate | Reject; solver mechanics valid |
| P14a | One-sided iterative 2×2 topology pruning | dangling false removed, but true 2×2 cascades to empty after iter2 | Reject synthetic operator |
| P14b | Bidirectional topology using first K candidate slots | synthetic pass, frozen candidate-order invariance fail; slot is not semantic rank | Reject integration |
| P14c | Score-ranked, order-invariant topology | one FIT recall 75.9058% preserved, only 2 of 73 728 R/D edges removed | Invalidated before G1: omitted reciprocal LEFT/UP physical evidence |
| P14d | Symmetric R/L/U/D topology; G0 then 128-grid | G0 removes 0 edges at K64; first K32/iter1 point took ~2 CPU-hours, train placement 0.208876% | **STOPPED before held; no quality verdict** |
| P15a/b | Multi-phase relaxation labeling around canonical + 3 seeded packings | P15a runtime abort; P15b valid permutation but not planted layout, objective unchanged 4272, 70.50s | P15b reject synthetic correctness |
| P16 | Bounded component beam assembly | exceeded 90s G0 cap; stopped at 133.83 CPU-s | Reject runtime configuration, no data result |
| P17 | Exact affected-edge QAP swap polish | synthetic exact-delta/planted repair pass in 11.227s; 4-board G0b exceeded 60s, stopped 87.36s | Runtime stop; arithmetic reusable |
| P18a/b | Cache canonical seeds, then P17 swaps | Stage A 3/4 seeds at 203.5s vs 180 cap; P18b validates 4 seeds but Stage B >60s, stopped 96.42s | STOPPED; no placement metric |
| P19 | Random contiguous-vs-random strip edge contrastive, 128 sources /12 epochs /seed 20260817 | CNN AUROC 0.859468 vs raw seam 0.986855 | Reject proxy: task too easy/misaligned |
| P20 / DDCC | RGB/tangential/normal derivative logistic calibration on true frozen hard rows | corrected 639 408 samples /39 963 positives; recall20 3.484842→3.455474%, −0.029368 pp | Reject. Attempt 1 invalidated for inverse neighbor mapping |
| P21 / GBLS | Positive-only masked boundary bridge, 2 000 FP32 steps, 96/32 | selection recall20 +0.001415 pp | Reject before held |
| P22 / FCLR | Exact frozen-row listwise boundary CNN, 96/32 | selection recall20 +0.079257 pp vs +1 pp gate | Reject before held |
| P23 / DCTR | Full-tile directional InfoNCE retriever, M grid, 96/32 | M64 coverage +4.180820 pp; recall20 only +0.012738 pp | Reject final score; proposal model remains useful |
| P24 / RCR | P23 retrieved full-pair 128-way cross-reranker | G0/G1 pass; all-source pool >5 min, ~15 GB, no checkpoint | STOPPED; hypothesis untested |
| P25 / SCXR | Streamed bounded revision of P24 | pools pass at ~0.17–0.42s/source; cross-ranker no 250-step checkpoint, ~14.8 GB | STOPPED before metric; hypothesis still open only with redesigned implementation |
| P26 / SHNCS | Lightweight full-pair scorer, 1 positive +15 hard negatives, 2 000 steps | selection recall20 2.740036% vs 3.456182%, −0.716146 pp | Reject lightweight approximation |
| P27 / AMGC | Covariance-aware Mahalanobis local gradients; 481 456 FIT rows | best alpha=0, recall20 3.456182%; nonzero worst −0.072181 pp | Reject |
| P28 / GDCP | Edge-conditioned graph coordinate denoiser, 2-board/600-step capacity | RMSE .294983/.297715 vs random .300965; needed ≤.150482 | Reject before 96-source training |
| P29 / DPCG | Frozen DINOv2 boundary descriptors; union M16/32/64 then logistic fusion | coverage 14.13999→22.25643% at M64, **+8.11644 pp**; best fusion recall20 +0.007077 pp | Candidate generator retained; score fusion reject |
| P30 / DGRS | Dense DINO-only reciprocal rank score | G0 exact synthetic; G1 deterministic; best λ=1 recall20 3.427404→3.430235%, +0.002831 pp | Reject rank algebra |
| P31 / BHCS | Raw 8-pixel seam CNN, hard contrastive; 105 984 examples, 6 epochs /2.60 GPU-min | recall20 3.494395→3.504303%, +0.009907 pp | Reject seam-only capacity |
| P32 / DSCP | DINO mean-tile feature + set Transformer to 576 slots; 96/32 | train top20 13.8636%, placement 1.2080%; selection top20 3.2878%, placement .1682% | Reject source memorization |
| P33 / CVA | Learned verifier + translation-consistent agglomeration on rank96∪DINO | G0/G1 pass; 96-source prep +10 epochs done; evaluator made millions of edgewise Python GPU calls, exceeded 15 min | STOPPED, quality unmeasured; vectorized scorer required |
| P34 / VCLS | Fully vectorized reciprocal+2×2 boolean pruning, 96 boards | mutual correct-edge coverage 60.1808→52.7485%, **−7.4323 pp** | Reject unweighted loop pruning |
| P35 / FCVT | Continuous row/col regressor over frozen DINO; 96/32 | train MAE 4.215 slots, exact .6529%; selection MAE 6.569, exact .2387% | Reject tile-only absolute coordinates |
| P36 / CSRP | Soft confidence-weighted 2×2 matrix propagation, preserves all candidates | G0/G1 pass; G2 only 4/96 in ~9 CPU-min, beyond 15-min cap | STOPPED, quality unmeasured; optimize/screen decoder first |
| P37 / RIT | Raw RGB 8×384 position-free relational Transformer, 4 epochs /384 board updates | train Top20 3.4883%, loss 12.7089 | Reject exact undertrained configuration, not all raw models |
| P38 / SRIT | 31M, 10×512, 80 epochs /7 680 updates | train Top20 3.5043%, loss 12.708713, 783s | Reject more-capacity/same-objective route |
| P39 / MPRT | Conv masked-pixel pretrain on all 5 360 unlabeled FIT inputs; relational fine-tune 30 epochs on 96, test 32 selection | G1 loss .37773→.14257 (−62.26%); G2 train Top20 **38.5549%**; G3 selection **3.5411%** vs 7% gate/random ≈3.48% | Reject source-disjoint transfer; pretraining machinery/diagnosis reusable |

## 7. Невалидные, отозванные и неоднозначные результаты

1. **P8 — жёсткая утечка candidate position.** Истинный neighbour стоял в slot 0 для 100% held rows. Все P8 checkpoints/scores/labels запрещены во всех последующих протоколах. Ни 100% local, ни 100% context не являются результатом модели.
2. **P20 attempt 1 — отозван.** Использовалась обратная `target_tile_to_slot` mapping при neighbor lookup. Run остановлен до завершения, после фикса полный G2 запущен заново. Только corrected G2 отрицателен.
3. **P20 JSON inconsistency.** `P20_G2_REPORT.json` содержит `passes_G2: true`, хотя gain отрицателен и `P20_REJECTION.md`/журнал однозначно фиксируют REJECT. Булево поле нельзя использовать как verdict; authoritative — corrected metrics + rejection note.
4. **P12 reporting unit bug.** Из 0.189887% в 0.184462% следует −0.005425 percentage points, а не −0.542535 pp. Знак/verdict не меняются.
5. **R6U1 ранние adapter checks не являются evidence.** До valid direct-metric run исправлялись schema, direction axes, permutation shape и coverage metric. Финальное допустимое число — +1.68 pp при падении active density.
6. **P14c invalidated, не rejected.** G0 был order-safe, но physical graph не включал reciprocal LEFT/UP; P14d зарегистрирован до G1.
7. **P14d, R3L, E2, P17/P18, P24/P25, P33, P36 — resource stops.** Их нельзя цитировать как отрицательное качество. Можно цитировать только невозможность конкретной реализации/бюджета.
8. **R8 synthetic G1 нельзя переносить на raw.** +10.96 pp на synthetic CAL действительно, но raw candidate coverage почти не выросла; production claim запрещён.
9. **PGA1 synthetic SSIM не сравним с rank96.** Fixed-corruption capacity gate провален; DEV не открыт.
10. **Журнальные артефакты.** В P12/P13 остались буквальные `$report` и PowerShell `$(@{...})`, в P39 Markdown есть control characters, во многих старых строках mojibake. Числа сверены с JSON/отдельными evidence-файлами, где они есть.

### Корректирующие коммиты, которые не являются отдельными экспериментами

- `d7311c7` заменил неверные clean-label assumptions в раннем ORBIT evaluator на distortion-aware labels.
- `15ba77a` исправил чтение metadata в U1 affinity cache.
- `b6f0849`, `613759d`, `ec1bb64`, `9e14b21` последовательно исправили label helper, shape, direction axes и routing G2; учитывать можно только результат после `9e14b21`.
- `5a66088` привёл F1P gate к правильному R0 baseline.
- `bcc80e9`, `011aafc` — синтаксис SA1 summary и parsing SA2 top-k distance.
- `76bfd49`, `b4ec135` — checkpoint provenance/serialization R4.
- `04d4a75`, `87858d0` — per-query indexing и padded-score mask SGT1.
- `b80c9d5`, `7c3a90a`, `3678c37`, `b0439e2`, `c290abb`, `f026388`, `8903356` исправили R5 uint8 conversion, AMP→FP32, device/layout и materialization/parsing отчёта; retained R5 result следует после всей цепочки.
- `cbbf7c8`, `b2b4eb6`, `771ca72` сделали R8 microbatch/resume после внешнего прерывания; source-disjoint checkpoint provenance сохранён.
- `3b95c8a`/G2 sharding обошёл silent termination CB1 после anchor 432; финальный G2 собран из четырёх hashed shards.
- P9 commits `740df4e`–`97540ba` исправили canonical width, duplicate semantics, graph arrays и absent-edge cache semantics до locked result.
- P12 corrections `84f919c`, `5af4b6a` были внесены до accepted G0b: width 128/shared candidate axis и score tensor `[4,576,128]`.
- P20 correctness chain `595a824`, `96926cb`, `b961c88` предшествует единственному допустимому G2 run.
- `c2decac` (P35 frozen features on GPU) и `4b98040` (P36 P13 label helper) — implementation fixes до их final gate/stop.

## 8. Повторять не нужно

Считать закрытыми в текущем виде:

- untrained multi-band/phase/derivative/MGC/AMGC seam features;
- PairwiseNet OH1/OH3/OH5 и photometric normalization;
- narrow boundary CDCS/FCLR/BHCS и lightweight sampled full-pair P26;
- direct CB1/rank/DINO alpha fusion;
- raw score reciprocity, unweighted 2×2 loops, scalar loop reweighting;
- small direct 576-slot Set Transformer, Sinkhorn refiner, conditional canvas и tile-only DINO coordinate heads;
- solver-only pose synchronization/topology/relaxation/multistart на прежних score-ах;
- direct raw-RGB 576-way adjacency CE с простым увеличением depth/epochs;
- текущий seeded-QAP implementation;
- любые P8 artifacts или candidate-slot-as-rank assumptions.

## 9. Внутриветочные направления и cross-series поправка

Изначальный branch-local ranking ставил SA3/R6/cross-ranker первыми. После
аудита legacy crawl и M420 общий порядок изменён: сначала honest generic
baseline и bijective content-aware diagnostic, затем fixed-layout output
restoration и all-emitter oracle supply gate. Authoritative общий ranking — в
[`knowledge-base.md`](knowledge-base.md); ниже сохранено, что именно остаётся
живым из этой ветки.

### Conditional — SA3: только новый разрешённый corpus, не ослабляя SA2

При наличии source SA1/SA2 радикально превосходят neighbour inference, а held verification имеет 100/0 true/wrong acceptance. Но известное покрытие — 18/700, и crawl 19 679 public photos новых matches не добавил. Нужны явное разрешение правил, действительно новые lawful sources, provenance и тот же strict SIFT/Hungarian accept. Порог понижать нельзя; `generic_only` и overlay публиковать отдельно, S1 сохранять для unmatched boards.

### Приоритет 2 — R6: broad-scene restoration и строгая композиция

R5 обучался фактически на двух FIT scenes, но дал официальный прирост после R5→NLM. Отдельно зарегистрированный, но не выполненный R6 — обучить ту же/улучшенную restoration architecture на широком FIT curriculum, затем сравнить на неизменных layouts против текущего R5→NLM с paired mean/lower-95. Это единственный путь, уже доказавший связь с leaderboard metric.

### Более низкий приоритет — raw-domain candidate-conditioned verifier

Комбинируем подтверждённые части:

- proposal diversity: P29 DINO +8.116 pp coverage, P23 +4.181 pp, CB1 +2.400 pp;
- joint-pair interaction: R8 сильно выигрывает на matched synthetic (+10.964 pp);
- failure lesson: synthetic→raw mismatch и малая raw cache выборка уничтожают transfer;
- execution lesson: P24/P25 scientific hypothesis не измерена — implementations упёрлись в 15 GB/нет checkpoint; P26 был слишком слабым приближением.

Новая попытка допустима только после content-aware/all-emitter supply gate. Она должна строить pools streaming/vectorized, видеть full-tile или multi-tile semantics, тренироваться на большом raw-like FIT corpus с multi-positive content labels, иметь cheap source-disjoint recall screen до solver-а и не использовать unbounded resident pairs. Ещё один boundary-only scorer повторит уже закрытые V24/R8/M419 ограничения.

### Приоритет 4 — source-invariant relational pretraining, но не ещё один direct CE

P7 доказал сильный corruption-invariant representation signal; P39 доказал, что masked reconstruction резко улучшает in-sample learnability. Провал — transfer. Следующая модель должна использовать image-specific relational/continuity/contrastive pretraining на всех 5 360 FIT scenes, stronger anti-memorization augmentation и selection gate. Нельзя просто повторять P38 или косметически увеличивать P39.

### Приоритет 5 — вернуть остановленные solver/verification идеи только после ускорения и лучшего score

- P33 learned agglomeration и P36 soft relaxation не имеют quality verdict.
- P17 exact-delta arithmetic корректна.
- Но до нового edge signal их expected value низок. Сначала нужен vectorized evaluator/decoder и дешёвый recall/correlation gate; только затем placement.

## 10. Reusable assets

### В Git и пригодно для переиспользования

- Provenance/split/cache probes: `build_pga1_source_split.py`, `inspect_split_manifest.py`, `inspect_*`, `probe_*`, P9 leakage audit.
- Source-aware route: `eval_sa1_source_aware_assignment.py`, `eval_sa2_retrieval_confidence.py`, `eval_sa2_strict_verification.py`.
- Production restoration/inference: `train_r5_restore_unet.py`, R4/R5 paired evaluators, `eval_r5_nlm_composition.py`, `infer_rank96_r5nlm.py` с deterministic ZIP, hashes и resume contract.
- Candidate generators/diagnostics: R2L/U1, CB1 sharded scorers, P23 DCTR, P29 DINO descriptor/coverage code.
- Hard-list and scorer harnesses: P3, P20–P27; использовать как negative-control infrastructure, не как доказанную модель.
- Solver primitives: P12 loop cache/order audit, P13 pose, P17 exact delta, P34 vectorized witness, P36 matrix propagation.
- Raw model research code: P37–P39; P39 encoder checkpoint format и split controls полезны, но код экспериментальный.
- 17 PowerShell runners отражают рабочий Windows/Task Scheduler execution path.

### Чего в Git нет

В ветке нет `.pt/.pth/.npz/.zip` checkpoints/caches/submission artifacts и нет самого source-disjoint manifest. Почти все canonical artifacts находятся только по путям `E:\pazzle_work\...`, включая R5 checkpoint, rank96 caches, P23/P29/P39 checkpoints, source corpus, 18 overrides и S1 ZIP. Поэтому ветка сохраняет **код + summaries**, но не полностью самодостаточную воспроизводимость.

Также код привязан к Windows paths, CUDA RTX 2070 и внешним baseline modules/checkpoints. В ветке не добавлены автоматические tests; многие P20+ scripts — одноразовые gate harnesses. Перед переносом в текущую macOS-среду нужно сначала найти/скопировать внешние artifacts и перепривязать paths, не переобучать «по памяти».

## 11. Практический decision map для будущих агентов

1. Хотим прямо поднять leaderboard SSIM без нового solver-а → broad R5/R6 restoration, paired against current S1.
2. Нашли public candidate source → только SA2 strict verify → SA1 assignment; иначе fallback S1.
3. Хотим улучшать placement → сначала source-disjoint candidate **ranking**, не solver SSIM.
4. Candidate coverage выросла, recall@20/precision нет → не запускать buddies/loops; нужен cross-ranker.
5. Модель хороша на FIT-train, плоха на selection → считать source memorization; не открывать held/CAL.
6. Глобальный solver медленный → cheap rank/edge-correlation screen, затем vectorized decoder; не повторять P33/P36 wall-clock failure.
7. Любой P8-derived score/cache/checkpoint → немедленно исключить.

## 12. Покрытие всех 338 коммитов

Ordinal — позиция в `git log --reverse d281361..9bd8db1`. Каждая строка соответствует секции леджера выше; вместе строки покрывают 001–338 без пропусков.

| Ordinals (count) | First..last | Леджер / содержание |
|---|---|---|
| 001–045 (45) | `a0879e6..c2ffa1d` | ORBIT R0–R3, G1/F1/F2, C1, R2L/U1/U2, D1, R3L/P1S, posterior P2, E2, OH1–OH6, Q1, G2 |
| 046–058 (13) | `6ba458f..1327b5b` | PN1/PN2, GC1, G3, G2b, F1P |
| 059–066 (8) | `8bea55c..52d5348` | SA1/SA2 |
| 067–071 (5) | `0d5706d..716ce8f` | PGA1 |
| 072–075 (4) | `86536e7..b4ec135` | R4 oracle-order harness/provenance fixes |
| 076–082 (7) | `518aadc..e469bf4` | SGT1 |
| 083–084 (2) | `f733060..d42478e` | R4 fixed-rank96-layout result |
| 085–102 (18) | `b10fd78..d2dca8d` | R5, composition, S1 runner, post-S1 research |
| 103–106 (4) | `0e19ff0..76b37fa` | SGT2-V |
| 107–109 (3) | `7c1f426..086c710` | CP1/QAP1 |
| 110–118 (9) | `f03add5..3cd7cbe` | R6U1 plus schema/metric corrections |
| 119–122 (4) | `5af6dfb..ab1bdfe` | R7 |
| 123–129 (7) | `0df3d96..fdd10fa` | R8, microbatch/resume, raw union rejection |
| 130–133 (4) | `ac1b21f..5372515` | R9 и официальный S1 score |
| 134–138 (5) | `7da8953..ea6a091` | R10-A |
| 139–143 (5) | `2eaf620..a35744e` | R11 |
| 144–144 (1) | `268c2e6` | итоговый ORBIT solver-pipeline research handoff |
| 145–158 (14) | `3ffa3fa..a6f17fb` | P1/CB1 G0–G4 |
| 159–160 (2) | `3937932..c5af5d9` | P2 CB1 score fusion |
| 161–163 (3) | `8450ebd..2c9739b` | P3 CDCS |
| 164–165 (2) | `b12d752..cbc8ed5` | P4 MGC |
| 166–167 (2) | `40efee9..9e719ce` | P5 Set-to-Grid |
| 168–169 (2) | `1e6045d..f92564e` | P6 diffusion |
| 170–171 (2) | `5e2409a..89447b1` | P7 pretraining |
| 172–175 (4) | `b3102cf..c692898` | P8 implementation/result/leakage invalidation |
| 176–187 (12) | `bad1bdf..8c94fbf` | P9, candidate-semantics fixes, P10 contingency prereg handoff |
| 188–194 (7) | `9ad4711..288ea7b` | P10 |
| 195–200 (6) | `4163816..ff9fd1f` | P11 |
| 201–207 (7) | `24ada23..a3b48f3` | P12 plus two pre-code contract corrections |
| 208–211 (4) | `35e7bb2..d794eac` | P13 |
| 212–218 (7) | `23a325e..2c868a0` | P14a–d |
| 219–221 (3) | `11ed2ac..0f43c4f` | P15a/b |
| 222–223 (2) | `fcf5e95..a3bd8ca` | P16 |
| 224–226 (3) | `70b7ec9..ff72667` | P17 |
| 227–230 (4) | `00beba6..77b6cc9` | P18a/b |
| 231–233 (3) | `d430887..b330542` | P19 |
| 234–239 (6) | `f1e8c7c..024dbb7` | P20, three correctness fixes, invalidated attempt + corrected rejection |
| 240–243 (4) | `0af5fe6..2201638` | P21 |
| 244–247 (4) | `bc05d6d..e657ba2` | P22 |
| 248–252 (5) | `d63e8b1..0683902` | P23 |
| 253–256 (4) | `6d0bec9..98c0532` | P24 |
| 257–264 (8) | `f22931c..525f636` | P25 streamed revision |
| 265–269 (5) | `b339e73..8991af4` | P26 |
| 270–273 (4) | `fed9827..1c15ecf` | P27 |
| 274–277 (4) | `b6fb2de..31a8c12` | P28 |
| 278–284 (7) | `32db199..1383854` | P29 |
| 285–288 (4) | `21d8f5c..f55cde2` | P30 |
| 289–293 (5) | `ce3c5a1..ef2c0f8` | P31 |
| 294–300 (7) | `273460f..6daa4ec` | P32 |
| 301–306 (6) | `6601dfe..bc0d93e` | P33 |
| 307–312 (6) | `d44995e..b221044` | P34 |
| 313–318 (6) | `a985703..177ad3f` | P35 |
| 319–323 (5) | `8551bdd..71ce0bb` | P36 и raw-transformer research handoff |
| 324–327 (4) | `448cabf..4bb4b84` | P37 |
| 328–331 (4) | `84f6d8c..6118dc0` | P38 |
| 332–338 (7) | `bcb655b..9bd8db1` | P39 |

## 13. Источники внутри ветки, которым отдавался приоритет

- `autoresearch-runs/pazzle-fixed-orientation-20260813/EXPERIMENTS.md`
- `autoresearch-runs/pazzle-fixed-orientation-20260813/FINDINGS.md`
- `autoresearch-runs/pazzle-fixed-orientation-20260813/PLAN.md`
- `P8_RESULT_P9_PREREGISTRATION.md`
- `SA1_SOURCE_AWARE_REPORT.md`, `SA2_SOURCE_RETRIEVAL_REPORT.md`
- `R4_RESTORATION_EVIDENCE_REPORT.md`, `R5_RESTORATION_EVIDENCE_REPORT.md`, `R5_SUBMISSION_CANDIDATE_PLAN.md`
- все `P*_PRE_REGISTRATION.md`, `P*_REPORT.json`, `P*_REJECTION.md`, `P*_STOPPED.md`
- parent-diffs всех 338 коммитов и 148 добавленных source/runners на tip.

Итоговый статус ветки: **аудит завершён; повторять закрытые подходы не нужно. Из её assets первыми полезны S1 как generic anchor, fresh OOF R6-like output restoration и P29/P23 candidate supply для content-aware/all-emitter gate; SA3 условен правилами и новым corpus, cross-ranker — прохождением этих gates.**

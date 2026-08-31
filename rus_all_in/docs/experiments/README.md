# Реестр воспроизводимых экспериментов

Дата среза: 2026-08-31. Это центральная точка входа для экспериментов,
выполненных уже после [аудита прежних исследований](../prior-research/README.md).
Она отделяет подтверждённый компонент от диагностического oracle и от
отрицательного результата, чтобы следующие запуски не повторяли уже закрытые
гипотезы.

[BasinCycle Stage-B 6x6](basincycle-stage-b-mps-reductions-v3.md) после двух
нулевых MPS-runtime сбоев был механически исправлен и прошёл единственный
подписанный final-only FIT/EVAL цикл. Генератор коротких swap/3-cycle действий
показал сильный supply: `54/58 = 93.1%` состояний с oracle-возможностью покрыты.
Но фиксированный selector выбрал KEEP в `64/64` случаях, поэтому pairs/exact/
radius2 delta равны нулю и результат имеет статус **fail-stop**. Причина — не
баг и не отсутствие кандидатов: q10/risk/q50 головы выучили маргинальный prior
сильно несбалансированного банка и создали пустое feasible set. Не подбирать
пороги и не переключаться post-hoc на policy argmax на открытом EVAL32. Сохранять
proposal generator; следующий materially different consumer должен напрямую
учить paired safe-improvement относительно KEEP либо композиционную changed-edge
energy на новом source-disjoint протоколе.

[Fixed adapter scale3200 continuation](fullres-retrieval-adapter-scale3200-deferred.md)
подписан как единственное продолжение положительного step400→1600 slope, но
локально остановлен на update400 до первого checkpoint и до любого scoring:
GPU был отдан более информативному tri-emitter verifier-у. Это server-ready
deferred protocol, не model failure; local/terminal/test и Weco153/154 не
открывались.

Новые sorter-ветки используют [трёхуровневую gating policy](sorter-gating-policy.md):
низкий чувствительный discovery gate, отдельный небольшой decoder/exact pilot и
строгий fresh confirmation только перед default/submission promotion. Это не
позволяет слабой ранней confidence-метрике закрыть реальный relational/supply
signal, но сохраняет высокий финальный стандарт.

[Tile-to-true-position distance validation](tile-position-distance-metric-validation.md)
зафиксировала reusable absolute mean/median/p90 Manhattan, normalized L1,
Euclidean и radius0/1/2, а также отдельно названный best-cyclic-aligned
diagnostic. На `32 sources x 24` frozen/controlled strict layouts absolute mean
Manhattan имеет within-source Spearman `.868/.863/.863` с clean/dirty/h20 SSIM,
тогда как exact/radius0 даёт более сильный Pearson `.965/.958/.927`, но слабый
Spearman `.412/.404/.393` из-за ties и низкого exact режима. Pure row-roll
контроль имеет aligned L1 `0` при exact `0` и clean SSIM `.385`, поэтому aligned
metric скрывает критичный origin error и не допускается как gate. Contract:
absolute exact остаётся primary, absolute mean Manhattan — smooth secondary,
radius2/p90 — companion diagnostics, pairs и restored SSIM логируются
ортогонально. Distance **добавить, не заменять exact**; Weco `149–154`, новые
panels и production не затрагивались.

[Relation-level truth selector для шести TASKA layouts](taska-relation-truth-selector.md)
заменил провалившийся 16-feature aggregate Ridge на один fixed nonlinear model
по всем `1,104` realised seams каждого arm. На source-disjoint held32 он дал
`+4.969` pairs, на opened fresh32 `+3.156`. Затем полностью новый signed
source16×draw2 confirmation подтвердил **`+5.844` pairs/board**, source-CI95
`[+3.000,+9.126]`, case W/T/L `13/19/0`; exact delta `-0.156`. Все `32/32`
outputs — целые strict permutations исходных upright tiles. Это formally
confirmed pair-solver component; он выбирает один frozen post-tail arm и не
является exact/SSIM improvement сам по себе. Отдельный SHA-gated layout-only
adapter `aiijc-taska-relation-selector` побитово воспроизвёл formal case 0;
official default/submission не менялись. Краткий current-best указатель —
[здесь](../BEST-current.md).

[HGB-ranked all-edge six-arm union](taska-relation-ranked-union.md) проверил,
можно ли теми же сильными relation probabilities собрать новый layout. Fixed
consumer max-p deduplicate-ил все realised relations и без threshold/top-k
подавал весь unique union в unchanged raw-tail solver. Уже на HGB-in-sample
local32 пары обрушились `326.750→199.500`, delta `-127.250`, source-CI95
`[-145.625,-108.530]`; exact `-3.469`. Gate провален, held/fresh/formal не
открывались. Не повторять post-hoc top-k/threshold rescue на этой панели.

[Union top48 rigid-fragment synchronization](union-fragment-synchronizer.md)
проверил reversible `Z24²` equations и rigid exact-cover consumer. На opened8
он ухудшил exact на `−0.75` tile/board и adjacency на `−1.642 pp`; защитный
Socket-objective guard превратил результат в no-op. Аудит fullres-denoise
подключения также выявил несопоставимые меж-query scores, Union-only guard и
тороидальное wrapping. Формулировка закрыта без sweep; эти ошибки не следует
повторять в следующем fusion consumer.

Новая [direct board-listwise модель hard-edge priority](direct-hard-edge-board-priority.md)
напрямую размечает все 1104 frozen d64 hard edges и оптимизирует ровно
decoder-бюджет top-144/axis. Она отличается и от независимой 21-параметрической
confidence calibration, и от component-query v1.1. Её scale-free rank-delta
перенос на Union-v2 теперь повторён на независимой source64 панели: exact
`1.234→1.875`, adjacency `13.934→14.019%`, все 64 layout — строгие
перестановки исходных upright tiles. Простое component-geometry переключение
между Union и rank-delta не подтвердилось (`1.828 < 1.875`) и закрыто без
sweep; always-rank-delta остаётся сильнейшим подтверждённым продолжением.

[Learned Union-hard priority](union-hard-edge-learned-priority.md) независимо
подтвердил сильный локальный signal на frozen eval32: fixed top288
`+3.625` правильных ребра, satisfied adjacency `+3.406` пары/board и recall
`+0.309 pp`, причём clustered CI положительны. Exact при этом статистически
нейтрален и численно чуть хуже (`1.188→1.094`), поэтому exact-gate провален.
Модель сохраняется как pair-level primitive, но не заменяет rank-delta и не
является standalone default; nearby selector/scale sweep на этой панели закрыт.
Единственная parameter-free композиция learned top-144 membership + rank-delta
ordering также закрыта после opened64 replay: пары выросли ещё на `+1.359` и
top288 на `+2.250` относительно rank-delta, но exact упал
`1.906→1.297`. Новый confirm-split на неё не расходуется; нужен другой
глобальный consumer компонентов.
Target-free self-satisfied-edge selector подтвердил тот же trade-off на
opened64 (`+1.516` пары, но `−0.766` exact tile/board) и закрыт. Seed16/factor16
component-MILP остановлен ещё до запуска: обычный builder уже максимизирует
его factor objective на `59/64` boards, rejected weight только `0.115%`.
Следующий bounded check — тот же learned top144 decoder без QAP24 swaps.
Этот check также завершён отрицательно: QAP0 против QAP24 дал `−0.078` пары,
`−0.0071 pp` adjacency и `−0.016` exact tile/board при идентичном top288.
QAP24 остаётся включён; nearby swap-budget sweep закрыт.
Один fixed cutoff-exchange continuation тоже закрыт без sweep: на opened32
он ухудшил learned arm с `164.031` до `152.250` satisfied pairs/board и с
`153.688` до `140.094` правильных top288 edges; все 16 source clusters хуже.
Небольшая exact-дельта `+0.125` статистически неопределённа и не компенсирует
систематический pair loss. Следующий шаг должен менять solver, а не ещё раз
настраивать top-144 loss/cutoff на этой панели.

[BorderPointer-24](border-pointer-sorter.md) реализовал literature-backed
full-resolution `20×20×48` field, ordered perimeter, permutation-equivariant
board encoder и strict causal pointer с left/up и absolute-border evidence.
4×4 capacity прошла `16/16`, но source-disjoint exact16 free-run проиграл
matched d64 decoder144: exact `1.4375→4.7500` в пользу baseline, adjacency
`5.627→13.655%`. Единственный preregistered baseline-guided rescue budget4
сохранил exact `76→76` total с малой потерей adjacency `−0.045 pp`; budget16
также не дал exact gain. Deployable baseline-prefix R@1 всё же вырос
`0.333→1.267%`, поэтому field/checkpoint сохранён для calibrated relation
fusion/QAP, но standalone pointer decoder закрыт без fresh/test.

Следующее новое layout-направление и его отличие от уже закрытых pairwise,
Sinkhorn, Set-to-Grid и diffusion веток описаны в
[обзоре contextual socket matching](../research/socket-matching-literature.md).
Завершённые [pilot и scale-up SocketMatcher v1](socket-matcher-v1.md)
подтвердили рост local retrieval и adjacency, но не exact placement/SSIM;
отчёт также помечает некорректную aggregate-dustbin border-метрику и фиксирует
мотивацию четырёх per-socket border heads в v2.
[SocketMatcher v2 d32](socket-matcher-v2.md) подтвердил этот signal на двух
новых real train-dev панелях: OT R@1 вырос примерно `6.5–6.9%→11.1–11.5%`, а
component/QAP decoder стабильно поднял adjacency `3.44%→7.76–7.97%`. Отчёт
отдельно помечает старый row-renormalized solver artifact, фиксирует
OT-mass-preserving reruns и отвергает texture-centre prior после провала
confirmation; prior остаётся выключенным по умолчанию.
[Fixed SocketMatcher + retained k5 ranker fusion](socket-ranker-fusion.md) на
новой source-fresh train-24 панели дал небольшой прирост decoder144: exact
tiles `14→18`, adjacency `7.74%→8.11%`, raw SSIM `+0.00041`. Но все paired CI
против socket control пересекли ноль, а R@5/16/32 снизились; 50/50 fusion не
promoted и его вес нельзя подбирать на уже открытой панели.
[SocketMatcher v3 border-distribution](socket-matcher-v3-border-distribution.md)
добавляет opt-in permutation-equivariant head, который определяет рамку по
отсутствию хорошего партнёра во всей строке/колонке score matrix. V2 остаётся
совместимым по `state_dict`. Выполненный d64 v2→v3 continuation одновременно
добавил score-statistics head и raw-rank auxiliary `0.15`. На строго paired
exact `24×2` v3 дал малые, но устойчивые gains partial-OT R@1
`16.8799%→17.1252%` и decoder adjacency `12.6264%→12.9133%`; при этом primary
absolute tiles упали `59→44`, а aligned осталось `13.0833`/board. Отдельный
fresh v3 cyclic arm дал `43→51`, но CI пересёк ноль. Continuation rejected;
**d64 v2 остаётся default**, а local/adjacency result хранится только как
research evidence без причинной атрибуции одному head-у.
[Edge-conditioned SocketPermutationFlow](socket-permutation-flow.md) проверил
итеративный current-layout refiner, существенно отличный от raw absolute heads:
frozen d64 socket embeddings, OT top-4 graph, relational GNN, Sinkhorn и
Hungarian. 4×4 capacity достигла 100% exact, но свежий exact 24×24 pilot ухудшил
direct `0.3038%→0.2170%` и adjacency `15.67%→1.29%`; formulation отклонена,
default не менялся.
[Historical E13 corruption-aware border encoder](corruption-aware-border-encoder-e13.md)
теперь материализован и честно проверен на source-disjoint exact `256 train / 16
eval`, 400 full-576 updates. Explicit JPEG/erosion/clean-corrupt consistency не
добавили usable signal: d64 OT R@1/R@5 `18.654/37.494%`, E13
`6.878/19.095%`, fixed 50/50 rank fusion `13.026/30.152%`; matched reciprocal
precision также ниже на `27.246/11.184 pp`. Оба local gate провалены, поэтому
global decoder не открывался и E13 помечен measured-negative at bounded gate.
[Source-disjoint E20 restored BorderRanker](restored-border-ranker-oof.md)
разрешил вторую исторически незавершённую ветку без утечки старых binaries.
DRUNet-descriptor union добавил top-32 coverage `+2.978/+2.763 pp`, поэтому
restored view сохранён как candidate-supply primitive. Но learned residual
cross-ranker дал R@1/R@5 `17.844/35.836%` против d64
`17.918/35.802%`, а matched reciprocal precision снизилась на `−3.246 pp`.
Gate провален; decoder не открывался, checkpoint rejected, default неизменён.
Аудит архитектуры уточняет возможную причину: independent tile pad-ится лишь до
24×24, а три stride-2 уровня DRUNet дают pyramid `24→12→6→3`. Skip connections
не уничтожают full-resolution signal полностью, но глубокий context слишком
груб для точной boundary phase. Поэтому закрыт именно этот downsampling U-Net.
[Full-resolution 20×20 boundary denoiser](fullres-boundary-denoiser.md) затем
проверил materially new альтернативу: 33,859-param zero-init NAF, все восемь
blocks сохраняют 20×20, loss сфокусирован на clean border/gradient/shape, raw
d64 остаётся неизменным. На exact source-disjoint16 restored-d64 top32 union
дал сильный supply `+4.857/+4.755 pp` right/down, descriptor union ещё
`+2.163/+1.981 pp`. Но direct restored d64 ухудшил R@1/R@5 на
`−1.913/−2.836 pp`, 50/50 raw/restored тоже проиграл, reciprocal precision
ниже. Sensitive discovery gate прошёл только по supply; strong decoder gate
провален, поэтому layout не открывался. Checkpoint сохраняется исключительно
как auxiliary emitter для context-aware selector, не replacement scorer.
[Full-resolution ordered twin side matcher](fullres-twin-side-matcher.md)
активирует отличную Edge2Vec/TEN-like representation ветку без RGB
reconstruction: `20×20×48` stride-one field, четыре ordered length-20 side
sequences, directional twin heads и явный raw skip. Два независимых legal
corruption views учатся через within/cross-view full-board listwise retrieval,
hardest same-board negatives и consistency. Procedural 4×4 capacity дала
`100/100%` R@1/R@5; full576 step измерен `56.709 s` CPU против `0.387 s` MPS.
Fit256/eval24 показал, что twin хуже raw как replacement ranker
(`R@1 12.22%` против `16.69%`), но raw32∪twin32 supply вырос на `+7.416 pp`.
[Raw/twin union reranker v2](raw-twin-union-reranker.md) превратил эту
диверсификацию в bidirectional row+incoming-column selector на новой
source-disjoint eval24. Оба gate прошли: partial-OT R@1/R@5 выросли на
`+0.476/+0.279 pp`, fixed top144 correct edges — на `+8.458/board` при
`+2.937 pp` precision. Predeclared decoder144+cyclic5 descriptive дал
exact `0.792→1.208` tiles/board и adjacency `12.847→13.753%`; strict original
permutations, без pixel replacement/holdout/test. Один frozen source-disjoint
fresh64 confirmation затем прошёл submission gate без retrain:
exact `0.938→1.281` (`+0.344`, CI пересекает ноль), adjacency
`13.668→14.419%` (`+0.752 pp`, CI положительный), top144 `+5.266`
correct/board; `128/128` strict original permutations. Verdict:
`frozen-fresh64-submission-candidate-confirmed`; competition test не открывался.
После promotion для этого единственного arm создан отдельный
[Union-v2 production contract](../union-v2-submission-production.md): exact
official-700 roster, SHA-locked Socket/Twin/Union lineage, strict raw tile audit
до historical RGB+luma+single-NLM-h20 tail, resumable packager и независимый
full reexecution validator. Metadata-only MPS dry-run прошёл без чтения test
pixels/writes; production run стартовал только после frozen fresh64 decision и
не меняет старый compliant-submission artifact.
[Learned fullres/component-relation fusion](fullres-relation-fusion.md)
реализовал этот selector без fixed score mixing. На source-disjoint local16
raw∪restored supply вырос `54.13→61.59%`, frozen-relation R@1/R@5 —
`12.01/35.99→13.17/37.43%`, а top32 precision — `20.51→34.96%`.
Но отдельно preregistered source40 exact pilot с translation-consistent forest,
decoder144 и тем же cyclic-border5 дал лишь `+0.025` exact tile/board и
`+0.0068 pp` adjacency; оба D2 material branches провалены. Не повторять
nearby cap/budget score-substitution sweep: selector остаётся research primitive
для materially другого joint graph/QAP consumer-а, default не менялся.
Target-assisted audit на тех же opened40 показал, почему: restored-only winner
ни разу не вошёл в learned top8, correct hard-edge count остался flat; oracle
порядок уже имеющихся raw edges поднял adjacency `13.97→20.23%`, а oracle
cyclic origin — exact `0.675→14.25`. Значит, нужны одновременно более сильный
global edge selection/QAP и новый inference-visible origin signal, а не ещё
один nearby forest cap.
[Sparse BorderGraph-QAP](sparse-bordergraph-qap.md) проверил именно такой
materially different consumer после formal activation: top-8 directional graph,
два explicit quadratic mean-field шага, Sinkhorn и strict Hungarian вместо
separable coordinate flow. Mechanical 4×4 прошла `16/16`, но exact16 QAP
полностью совпал с decoder144+cyclic5: `2.500` exact tile/board и `13.995%`
adjacency у обоих. Pure quadratic truth-minus-decoder energy оказался
`−77.475`; tiny conditional edge R@1 gain `+0.416 pp` не конвертировался.
Bounded gate fail-stop: не sweep-ить anchor/top-k, default/test не менять.
[Архивный Pasha883 PairwiseNet C64](pasha883-pairwise-audit.md) впервые проверен
честным all-576 scoring: на четырёх historical model-selection-exposed boards
pooled R@1/R@5/R@25 равны `18.07/34.87/58.49%`. Это близко к d64 OT R@1
`17.76%`, но buddies direct остался около chance, поэтому artifact сохранён как
local evidence, не новый default. Строго matched d64 diagnostic на тех же boards
дал Socket partial-OT `16.51/34.04/58.70%`; единственный 50/50 row-rank fusion
поднял retrieval до `19.68/37.21/60.64%`, но buddies96 adjacency снизилась
`6.86%→5.62%`, aligned также не вырос. Fusion закрыта как local-only идея без
promotion и без нового fresh panel. Lineage check уточняет: boards source-disjoint
для Socket train1024/eval32, но model-selection-exposed для Pasha, а reference
target-assisted/recovered; это не общий holdout. Старый `val=.4766` также исправлен в
документации: это sampled 32-candidate accuracy на 48 anchors, ошибочно
напечатанный как `acc@48`.
[Bounded Pasha-on-Socket top-32 reranker](socket-pasha-topk.md) после этого
реализован только как fail-closed primitive: `36 864` Pasha pair evaluations
вместо `663 552`, fixed 50/50 within-row rank priority и unchanged Socket
transport/decoder objective. Dirty-only MPS benchmark занял `4.91 s` на Pasha
часть, но новый quality panel не открывался из-за уже проваленного global gate;
default не менялся.
[Recovered TASKA focal seam verifier](taska-focal-verifier.md) теперь имеет
SHA-gated weights-only port и два явно разделённых historical feature contract.
Training-exact top5 дал pair delta `+0.781` на opened32 и `+2.906` на held300;
repository-tip top8 дал `+2.188` на held300. Знак переносится, top5 сильнее, но
все CI пересекают ноль и обе панели model-selection-exposed. Focal сохраняется
как promising ordering primitive, не новый default; K/threshold на этих панелях
не подбирать.
[Один fixed current-harvest focal fine-tune](taska-focal-current-finetune.md)
затем обучил recovered top5 residual на 96 organizer-train boards и проверил
его на 32 disjoint local-gate boards. All-pairs ranking loss снизился
`0.23778→0.20752`, но solver pairs упали `308.719→308.188`, exact
`2.0625→2.0000`; nonnegative gate провален и held не запускался. Этот exact
two-epoch fine-tune закрыт без sweep: уменьшение edge-ranking surrogate не
перенеслось в rigid-component consumer.
[Fixed board/axis focal pairwise ranker](taska-focal-pairwise-ranker.md)
проверил отдельную от BCE формулировку на тех же 22 target-free признаках:
каждый positive сравнивался с четырьмя hardest-focal negative своего board и
axis, а linear head обучался на точных sign-reversed differences. На disjoint
local32 standalone дал лишь `280.594` пары, а включение пятым arm ухудшило
four-arm+tail96 с `314.375` до `311.531` (`−2.844`, CI95
`[−6.5,−0.219]`). Local gate провален, held/fresh не открывались; эту точную
hard-negative/intercept-free формулировку и nearby sweep не повторять.
[Parameter-free raw/focal axis-rank fusion](taska-focal-raw-rank-fusion.md)
проверил один fixed equal-midranks consumer без alpha sweep. Уже на opened32 он
ухудшил raw с `334.719` до `334.125` пары и exact с `4.469` до `3.906`.
Неизменный protected-tail arm восстановил только 0.5 пары и остался ниже raw
(`334.625`). Preregistered gate остановил held300; equal rank averaging и
nearby weight/budget sweep закрыты, focal следует хранить отдельным arm.
[Bounded exact-oriented TASKA portfolio proxies](taska-exact-portfolio-proxy-closure.md)
закрыли ещё два простых selector: realised focal-logit и audited
structural-border score между focal top-5 и three-arm+tail pair leader.
Лучший border-proxy дал opened/held exact `4.469/3.469` против
`4.656/3.156` pair-leader и pairs `336.313/335.250` против
`338.688/337.031`. Held exact частично вернулся, но sign не
перенёсся и full focal exact не сохранился. Production selector не
добавлен; nearby threshold/weight sweep на этих panels закрыт.
[Current-disjoint protected-tail confirmation](taska-protected-tail-fresh32-confirmation.md)
повторил fixed 24-swap polish и заранее зафиксировал единственное расширение до
96 swaps на новых `16 sources × 2 draws`, исключив opened32 и прежний held16.
Arm96 дал `342.094` пары против raw `339.750`: delta `+2.344`, source-bootstrap
CI95 `[+1.000,+3.719]`; против arm24 gain `+1.813`, CI95
`[+0.438,+3.281]`. Exact остался нейтрален (`−0.031`, CI пересекает ноль),
поэтому arm96 promoted только как pair-oriented primitive. Панель всё ещё из
historically model-selection-exposed last-300 range, а 19/32 случаев упёрлись
уже в cap96; nearby budget sweep на открытой панели не разрешён.
[Fixed focal/four-arm leader replay on fresh32](taska-fresh32-leader-confirmation.md)
без sweep перенёс focal top5 и raw/logistic/focal/nonlinear all-bond portfolio
с tail96 на тот же current-disjoint roster. Focal дал `342.656` pairs и
`1.625` exact против raw `339.750/1.219`; exact delta `+0.406`, но CI пересёк
ноль. Portfolio+tail96 дал **`346.063` pairs**, delta **`+6.313`**, clustered
CI95 **`[+2.281,+10.063]`**, при exact delta `-0.063`. Pair pipeline
подтверждён, focal остаётся отдельным exact arm. Targets панели уже были
открыты parent-run, поэтому это не formal fresh promotion.
[Fixed focal-gated protected tail](taska-focal-gated-protected-tail.md) оставил
тот же four-arm all-bond winner, исходные costs и non-adjacent tail96, но
защищал только реализованные harvested edges с recovered focal top5 logit
`>=0`. После touched local gate `+0.031` и held gate `+0.531` неизменный
fresh32 replay дал новый pair leader **`348.344` против `346.063`**:
delta **`+2.281`**, source-cluster CI95 **`[+0.875,+3.594]`**, recall
`0.315529`. Exact снизился на `−0.125` с CI через ноль. Verdict:
`pair-candidate-confirmed`, без production promotion до нового неизменного
roster replay; threshold/budget sweep закрыт.
[Новый preregistered fresh16 replay focal-gated tail](taska-focal-gated-protected-tail-fresh16-confirmation.md)
закрыл этот последний promotion gate без изменения threshold, budget или
arm roster. На current-lineage-disjoint `16 sources × 2 draws` candidate
поднял all-edge control с `352.875/2.281` до **`354.750/2.500`**
pairs/exact: pair delta **`+1.875`**, source-cluster CI95
`[-0.188,+3.844]`, exact delta `+0.219`. Заранее заданные ворота
`mean>=+0.5`, `CI lower>=−0.25` прошли; focal-logit-zero protection теперь
можно promoted как pair-default tail primitive. Production в этом run не
менялся, а nearby threshold/budget sweep на открытой панели закрыт.
[Один preregistered focal-gated tail192 capacity step](taska-focal-gated-tail192-capacity.md)
проверил видимый saturation tail96 уже на новом signed collision-free
`16 sources × 2 draws` roster. Control достиг cap96 на `30/32`, tail192 принял
в среднем ещё `54.47` swaps, но pairs снизились `323.094→322.781`: delta
`−0.313`, source-cluster CI95 `[-1.281,+0.688]`. Exact secondary вырос на
`+0.156`, CI95 `[0,+0.281]`, однако fixed pair gate провален. Tail96 остаётся
default: дополнительная минимизация original seam objective переобучает layout,
а nearby budget/threshold sweep на открытой панели закрыт.
[Monotone block-Hungarian TASKA tail](taska-monotone-hungarian-tail.md)
прошёл намеренно чувствительный opened gate (`+0.438` пары), но unchanged
held перенос развернул знак (`−0.156` пары, exact flat). Ветка закрыта без
round/threshold sweep; код сохранён только для воспроизводимости.
[All-bond cyclic-origin screen](taska-all-bond-cyclic-origin.md) проверил все
`24×24` глобальные rolls pair-лидера по той же original TASKA seam-cost.
Правило изменило лишь 1/32 opened boards и ухудшило pairs/exact на
`−0.250/−0.031`; held не открывался. Для origin нужен новый border/semantic
signal, а не повторное применение all-bond objective.
[Structural-border cyclic origin](taska-structural-border-cyclic-origin.md)
затем использовал materially другой dirty-only score: audited TASKA
structural-border unary (`slack=6`, 20 Sinkhorn iterations) выбирал один из
576 rolls focal layout. Он изменил все 32 origins, но снизил opened pairs
`335.500→323.625` и exact `4.344→3.844`; held не открывался. Этот unary не
является пригодным absolute-origin scorer, nearby slack/blend sweep закрыт.
[Naive largest-component centre roll](taska-largest-component-center-roll.md)
буквально проверил эвристику «самый большой уверенный кусок поставить в
центр». На opened32 raw TASKA упал `334.719→323.531` pairs и
`4.469→0.906` exact, pair W/T/L `0/1/31`. Largest component часто является
фоном/рамкой, доходящей до границы, а не центральным лицом; закрыт именно
whole-layout roll по размеру component, без nearby shift/rounding sweep.
[Majority-bond consensus component arm](taska-consensus-component-arm.md)
проверил materially другой consumer: directed bonds с support `>=2` среди
raw/logistic/focal/nonlinear layouts стали новым supply component builder, а не
только protected set после сборки. Несмотря на `825.44` majority bonds/board,
opened32 резко ухудшился с pair-leader `341.313/4.750` до
`322.844/1.906` pairs/exact; pair delta `−18.469`, clustered CI95
`[−26.063,−10.969]`. Agreement коррелированных solvers закрепило общие wrong
geometries. Gate провален; held/fresh и nearby cutoff/weight sweep не запускались.
[Fixed four-seed TASKA multistart](taska-fixed-multistart-portfolio.md)
возобновил ранее незавершённую nondeterminism-идею уже на сильном TASKA signal:
seeds `(0,1,2,3)` для каждого из четырёх ordering arms, minimum original
all-bond selector и tail96. Расширенный 16-layout portfolio ухудшил opened32 с
`341.313/4.750` до `339.344/4.438` pairs/exact; pair CI95
`[-4.25,0]`. Это winner's curse proxy-selector-а. Held/fresh не открывались;
не увеличивать seed roster без materially нового robust selector.
[Alternate raw-log TASKA tail](taska-rawlog-alternate-tail.md) проверил один
fixed alternate trajectory: те же start/protected edges/tail96, но swaps
оценивались строго по `-right_log/-down_log`, после чего original-cost selector
сравнивал candidate и control. Opened и held дали точные нули: layouts
совпали на всех `64/64` cases, метрики остались `341.313/4.750` и
`337.563/3.063` pairs/exact. Диагностика показала, что frozen TASKA cost на
всех допустимых off-diagonal парах равна `-raw_log` плюс константа с residual
range не больше `1.91e-6`. Это один и тот же objective; fresh не открывался,
raw-log tail отдельно больше не повторять.
[Adjacent-aware protected tail96](taska-adjacent-aware-protected-tail.md)
устранил технический запрет соседних swaps: их delta теперь exact считается по
union затронутых directed bonds, а весь остальной fixed contract сохранён.
Pair delta была `+0.250/+0.688/-0.125` на opened/held/fresh; exact delta
`-0.125/-0.031/-0.125`, все CI пересекли ноль. Это слабый реальный pair signal,
но fresh знак не перенёсся, поэтому current four-arm+tail96 остаётся default;
тот же adjacent search с nearby budget/gain sweep не повторять.
[Coordinate-only monotone component placement](taska-monotone-component-placement.md)
проверил полное удаление historical unconditional two-component relocation
loop уже внутри current four-arm + all-bond selector + tail96. На opened32
candidate ухудшил control `341.313/4.750` до `340.031/3.813` pairs/exact;
pair delta `-1.281`, CI95 `[-4.063,+1.531]`, exact delta `-0.938`. Gate
провален, held/fresh не открывались. Вместе с прежним objective-guarded raw
step 18 закрыты прямые guard/omit варианты этого relocation defect; nearby
seed/round/guard sweep по тем же components и objective не повторять.
[Current candidate/solver bottleneck diagnostic](taska-current-candidate-bottleneck.md)
разложил local32 pair result без изменения solver-а. Из `374.44` harvested
edges истинны `252.94`, а current layout уже реализует `245.31` из них; всего
он получает `314.38` пар, включая `69.06` true noncandidate seams. Значит,
ближайший резерв — complementary matcher supply и отказ от защиты явно слабых
false relations, а не ещё один глобальный search по тем же identities. На
уже touched threshold audit естественная focal-граница `logit>=0` разделила
realised edges на `88.48%` precision kept и `20.42%` precision dropped; она
зафиксирована ровно как один последующий transfer-test, без threshold sweep.
[Fixed dynamic mutual-vote target500](taska-vote500-candidate-supply.md)
проверил ровно одно расширение supply без перебора nearby thresholds. На
disjoint local32 candidate count вырос `374.44→531.34`, supplied true pairs —
`252.94→294.19` (`+41.25`), но realised supplied прибавил лишь `+9.56`, а
true noncandidate seams потеряли `−17.44`. Поэтому итоговые pairs упали
`314.375→306.500` (`−7.875`, CI95 `[-18.906,+2.281]`), хотя exact шумно вырос
`1.375→2.875`. Local gate провален; held/fresh не открывались. Не sweep-ить
target400/450: нужен selective consumer новых low-vote edges, а не их
безусловное включение в rigid components.
[Selective target500 focal consumer](taska-selective-vote500.md) реализовал
ровно этот следующий шаг: один target500 matcher pass, same-pass current350
subset, только новые edges с recovered focal top5 logit `>=0`, один union-focal
fifth arm, original all-bond selector и уже подтверждённый focal-gated tail96.
Pair delta против focal-gated control перенеслась local/held/fresh:
**`+9.219/+5.000/+5.750`**, все три CI lower положительны; fresh достиг нового
pair result **`354.094`**, recall **`0.320737`**. Focal intersection поднял
precision новых lower-vote edges до `62–65%` и дал `+2.25–2.74 pp` candidate
recall. Exact delta `+0.281/−1.438/−0.063` не переносится, поэтому это ведущий
pair-oriented candidate, не exact/default. Same-pass control побитово совпал с
историческим на `96/96`; thresholds/budget не sweep-ить.
[Formal disjoint confirmation selective target500](taska-selective-vote500-formal-confirmation.md)
проверила тот же неизменный solver на новом preregistered source16×draw2
organizer-train roster из диапазона `6700:6999`, после conservative исключения
всех явных prior TASKA sources и отдельных tail192/fullres-combo rosters.
Candidate перенёс gain: **`354.281`** против `348.781` pairs/board, delta
**`+5.500`**, source-cluster CI95 **`[+0.813,+11.313]`**; gate
`mean>=+2, lower>=0` пройден. Exact delta `+0.219`, CI95
`[-0.125,+0.625]`, поэтому claim остаётся pair-oriented. Accepted new precision
составила `69.17%`, union recall `27.734%` против `24.425%` current. После gate
добавлен отдельный SHA-gated layout-only `taska_best_pair_pipeline.py` и CLI;
legacy pipeline не заменён, competition test/submission/postprocess не
использовались.
[Fixed full-resolution restored-view union voter](taska-fullres-union-voter.md)
реализовал именно selective supply consumer без изменения 12-scorer harvest и
raw dense costs. Новое ребро требовало одновременно support `>=3/4` среди
restored v3/local×orientation scorers и recovered focal top5 logit `>=0`.
Five-arm+tail96 перенёс pair gain local/held/fresh
`+4.781/+4.219/+4.031` пары/board; held/fresh clustered CI95 полностью
положительны. Fresh достиг нового pair leader **`350.094`**, recall
**`0.317114`**, против `346.063/0.313462` у unchanged control. Focal gate
поднял precision широкого restored proposal pool с 15–17% до 57–60% и добавил
17.19 correct missing edges/board на fresh. Exact delta
`+0.406/−0.250/−0.063` не переносится; arm retained как подтверждённый
pair-oriented кандидат, production пока не менялся. Не sweep-ить support,
focal threshold, orientations или budget на открытых panels.
[Selective target500 + unique fullres union fusion](taska-selective-fullres-union-fusion.md)
механически объединил два frozen accepted-edge supply без matcher rerun и без
нового threshold/budget sweep. Более половины fullres accepted уже совпадали с
selective accepted; после дедупликации оставалось `11.72/14.81/12.69` unique
edges на board, из них истинны `5.28/7.78/6.03`. Один combined-union arm поверх
current4+selective перенёс pair delta local/held/fresh
`+3.156/+2.219/+1.531`; fresh достиг нового pair leader **`355.625`**,
source-CI95 delta **`[+0.125,+3.094]`**, W/T/L `5/27/0`. Exact не переносится
как устойчивый gain. Frozen selective control воспроизведён `96/96`. Отдельная
preregistered source16×draw2 confirmation затем дала
`330.031→333.125` pairs, delta **`+3.094`**, source-CI95
**`[+0.844,+5.750]`** и прошла gate; exact delta `+0.844`, но CI пересекает
ноль. Fusion теперь confirmed pair leader и доступен отдельным SHA-gated
layout-only CLI `aiijc-taska-best-pair-fusion`; selective fallback, pixels,
legacy, official best и submission не менялись. Weco step `102` parent `97`.
[Confirmed-arm seven-layout portfolio](taska-confirmed-arm-portfolio.md)
проверил ровно одно расширение этого fusion: добавить independently confirmed
standalone fullres pre-tail arm к current4+selective+combined и выбирать минимум
того же raw all-1104 cost, затем winner-aligned focal-tail96. Common
local/held/fresh дали pair delta `+1.000/+1.313/+1.000`, но новая заранее
зарезервированная source16xdraw2 панель развернула знак:
`314.344→313.438`, delta **`-0.906`**, source-CI95 `[-2.750,+0.625]`, exact
также `-0.094`. Confirmation gate провален; seven-arm rejected, confirmed
six-arm fusion остаётся лидером. Это winner's-curse no-repeat: хорошие
standalone layouts нельзя просто добавлять под прежний all-bond proxy; fullres
следует объединять на уровне evidence до solve. Weco steps `109–112`.
[Exact cyclic row-phase DP](taska-row-phase-dp.md) проверил materially другое
`24^24` пространство крупных ходов поверх frozen confirmed fusion: membership и
циклический порядок каждой строки фиксированы, а 24 фазы на строку глобально
оптимизируются точным Viterbi по исходной raw all-1104 seam-cost. Truth oracle
на local32 показывал `+1.063` pairs, но fixed target-blind objective изменила
лишь 1/32 layouts и дала `326.781→326.688`, delta **`-0.094`**, CI95
`[-0.281,0]`, W/T/L `0/31/1`; exact без изменения. Held/fresh не открывались.
Не повторять phase DP с той же raw objective и не sweep-ить phase penalty или
acceptance threshold: для этого пространства нужен новый независимый сигнал
места разрыва строки. Weco pair+exact step `113`, parent `102`.
[Unique-fullres translation consensus](taska-fullres-translation-consensus.md)
проверил один новый inference-visible graph signal внутри неизменного six-arm
fusion: если минимум два unique edges между одной парой selective components
требуют один rigid translation, они строго поднимаются выше старого maximum
priority. На local32 signal имел `6/8 = 75%` edge precision, но был лишь на
4 boards; на всех четырёх all-bond selector выбрал другой старый arm. Поэтому
candidate побитово связал control на `32/32`: pairs `326.781→326.781`, exact
`5.938→5.938`, CI delta `[0,0]`. Gate провален, held/fresh не открывались.
Не sweep-ить support/priority на открытом local; нужен другой target-blind
consumer или более частый independent consistency signal. Weco pair+exact
step `114`, parent `102`; confirmed fusion остаётся лидером.
[One-component relation anchor](taska-component-relation-anchor.md) затем
проверил materially другой post-tail consumer: selected-supply cross-component
edges голосуют за integer shift уже реализованной компоненты; допускается ровно
один rigid move с local bijective fill и только при строгом улучшении original
all-1104 seam cost. Target-assisted local32 diagnostic показал большой exact
headroom: frozen `5.938` exact, oracle best cyclic `71.938`, oracle best
one-component move `52.375`; последний при этом снижает pairs
`326.781→302.938`, то есть seam objective не является absolute anchor signal.
Fixed candidate дал local/held/fresh pair delta `+0.375/+0.313/+0.156`, но exact
`+0.063/0/0`: знак pair переносится, exact — нет. Кандидат не promoted;
threshold/size/weight/cost-gain sweep на открытых panels закрыт. Следующая exact
ветка должна jointly согласовывать несколько translations либо добавлять
независимый board-conditioned absolute signal. Weco exact+pair steps
`120/121/122`, parent `102`.
[Cross-arm absolute component anchor](taska-cross-arm-component-anchor.md)
проверил materially другой exact signal: rigid shift control component
должен быть точно подтверждён всеми её tiles сразу в `>=2` distinct
post-tail arms. Без raw-seam veto и semantic prior двигалась максимум
одна component с local bijective fill. Rule был частым (`19.06`
hypotheses/board, `32/32` changed), но correlated arms подтверждают wrong
absolute gauge: exact `5.938→5.688` (`-0.25`), pairs `326.781→318.406`
(`-8.375`, W/T/L `0/3/29`). Local gate fail-stop; terminal/fresh не
открывались. Не sweep-ить support/arm weights/component cap/ranking/fill.
Weco pair+exact step `133`, parent `102`; step `134` не использован.
[Frozen Socket cyclic-border5 origin transfer](taska-socket-cyclic-origin-transfer.md)
применил independently confirmed d64 Socket absolute-origin primitive ко всему
final six-arm layout без TASKA raw-seam veto, semantic/centre prior или нового
fit. Он изменил `17/32` boards и резко поднял opened all32 exact
`5.938→12.875` (`+6.938`), но пары упали `326.781→323.438` (`-3.344`) и
нарушили preregistered floor `-2`. На 26 sources вне Socket lineage знак exact
сохранился (`+1.654`), pairs `-2.962`; headline all32 частично обусловлен одним
lineage-overlap gain `+256`. Step `147` fail-stop, `148`/terminal/fresh закрыты.
Hard-safe oracle (`exact>0`, per-board pairs `>=-2`) существует, однако fixed
source-LOO conservative selector на Socket margins, TASKA cut-loss proxy, roll
distance и six-arm support получил AUC `0.233`, precision/recall `0/0` при
двух positives. Не sweep-ить border/gain/support/classifier threshold на
local32; сигнал хранить только для заранее подписанного большого disjoint fit.
[Dense-contact joint component pose Transformer](taska-joint-component-pose.md)
сначала закрыл generic set/absolute head как дубль
`component_absolute_placer`, а selected-supply shift roster — из-за fit support
coverage всего `0.627%`. Materially новый dense TASKA top8 roster поднял
coverage до `48.31%` до cap; 212k-param all-component edge-message+attention
model прошёл capacity R@1 `90.9%` и на source-disjoint local поднял
support-weighted shift R@5 `2.70→9.34%`. Но R@1 вырос лишь `0.48→0.75%`.
Four-anchor pack численно дал exact `1.906→2.094`, одновременно разрушив pairs
`345.313→278.844`; forensic нашёл только `1/97` dominant-correct anchors,
поэтому exact delta признана repacking noise. Gate fail-stop, server scale
заблокирован, cache сохранён; top-k/cap/threshold/max-anchor sweep закрыт.
Weco failed step `124`, parent `102`.
[Dense-top8 reciprocal translation-consensus feasibility](taska-dense-contact-consensus-feasibility.md)
проверил pair-first pivot без pose model: тот же frozen dense TASKA top8
emitter, только reciprocal currently-missing contacts, support `>=2` одного
component-pair translation и обязательные right+down axes. Сигнал был
очень частым (`210.1/195.4` edges/board fit/local, `32/32` boards), но
pooled precision всего `10.86/11.61%` при true-missing coverage
`2.96/2.99%`. Fixed `>60%` precision gate резко провален, поэтому
consensus-supply solver не строился. Не sweep-ить support/top-k/focal/
reciprocity/axis на этих panels; dense roster остаётся broad shortlist, но
не hard-edge supply. Weco feasibility step `125`, parent `102`; step `126`
не использован.
[One-swap focal-objective diagnostic](taska-focal-objective-swap-diagnostic.md)
проверил локальный поиск по sparse focal evidence вместо очередного уменьшения
того же raw seam-cost. Positive-only objective изменил лишь `2/32` layouts и
дал `+0.031` пары/board; signed logits изменили все layouts, но дали `-0.313`
пары, W/T/L `0/23/9`, при нулевой exact-дельте. Отрицательный focal edge можно
распознать, но одиночный swap не знает безопасного нового места для tiles и
ломает unrelated seams. Ветка закрыта без iterative/weight/threshold sweep;
нужен joint region/component consumer. Weco pair+exact step `127`, parent
`102`.
[Two-sided log-rank whole-layout selector](taska-two-sided-rank-selector.md)
заменил raw all-bond sum одним parameter-free scale-free proxy: сумма
`log1p` outgoing-row и incoming-column ranks всех реализованных right/down
seams у каждого из шести post-tail layouts. Он активно менял выбор, но local32
ухудшил `326.781→324.125` pairs и `5.938→1.344` exact. Rank-нормализация не
устраняет correlated whole-layout winner's curse; held/fresh и transform/top-k
sweep закрыты. Weco pair+exact step `131`, parent `102`.
[Unique-fullres accepted-edge calibrator](taska-fullres-unique-edge-calibrator.md)
проверил ровно один source-disjoint `StandardScaler -> LogisticRegression(C=1)`
filter на уже accepted unique-fullres suffix. Fixed threshold `0.5` повысил
edge precision local/held/fresh с `45.07/52.53/47.54%` до
`68.03/64.44/69.51%`, но сохранил лишь `58–59%` true edges. Held pair delta
была `+1.844` (CI пересёк ноль), а fresh знак сменился на `-0.688`; exact fresh
шумно `+0.063`. Формальный roster не открывался. Это показывает, что более
чистый, но сильно урезанный supply хуже широкого confirmed fusion для rigid
consumer-а. Feature/C/class-weight/threshold sweep на открытых panels закрыт;
Weco held/fresh steps `104/105`, production неизменён.
[Learned post-tail fusion guard feasibility](taska-fusion-posttail-layout-guard-feasibility.md)
посчитал потолок выбора между final selective и confirmed fusion: oracle gain
над fusion равен `+0.688/+0.469/0.000/+0.188` pairs/board на
local/held/fresh/formal. Local прошёл eligibility `>=+0.5`, но fit
local+held содержит лишь `4` selective-win и `13` fusion-win non-tie rows при
`47` ties; minority class ниже preregistered минимума `8`. Поэтому logistic
guard не fit-ился, fresh/formal candidate evaluation и Weco 107/108 не
запускались. Не oversample-ить эти три selective-win sources и не превращать
уже открытые fresh/formal в fit; confirmed fusion остаётся control.
[Six-arm board-relative learned selector](taska-six-arm-learned-selector.md)
расширил задачу до всех шести independently focal-tail96-polished layouts,
где pair oracle над fusion равен `+4.875/+9.531/+6.688` на
local/held/fresh. Единственный signed contract — 16 target-free per-arm
cost/tail/focal/agreement features, within-board centering, arm contrast и
`StandardScaler + ordered-pair Ridge(alpha=1)`; source-grouped 8-fold local OOF
дал `326.781→325.938`, delta **`-0.844`**, CI95 `[-3.313,+1.469]`, exact
`-0.281`. Local gate провален, held/fresh не открывались. Даже all-local
in-sample selector остался `-0.125`, поэтому не sweep-ить alpha/features/margin
и не заменять Ridge на nearby logistic на этих 32 rows; нужен больший новый
source-disjoint fit roster и relation-local evidence. Weco pair+exact step
`116`, parent `102`; steps117/118 не использованы.
[FullResolutionTwin unique supply поверх fusion](taska-twin-unique-supply.md)
не повторял отрицательный standalone Twin ranker: direct Twin top32 пересекался
с frozen Union-v2 hard-top144, дедуплицировался против combined parent и
проходил focal logit `>=0`. Local дал сильный pair signal
`+1.625 [0.375,3.125]` и noisy exact `+1.719`, но held перенёс лишь
`+0.219 [-1.781,2.625]` pairs при gate `+0.5`, exact `-0.094`. Fresh не
открывался. Focal поднял precision очень слабого proposal pool с `9–10%` до
`41–46%`, но оставшиеся `~5.3` edges/board недостаточно стабильны. Сохранить
как diagnostic/no-repeat; top-k, focal threshold, roster и tail на этих panels
не ослаблять, production не менять.
[Official-DRUNet unique restored-descriptor supply](taska-drunet-unique-supply.md)
проверил ровно один nominator-only follow-up поверх этого fusion. После
reciprocal restored-width6 nomination, дедупликации current/selective/fullres и
старого dirty focal `>=0` оставалось `118.13` accepted edges/board, но их unique
precision была лишь **`0.397%`**. Local control `326.781/5.938` ухудшился до
`323.625/1.563` pairs/exact; pair delta `-3.156`, CI95
`[-6.906,+0.125]`. Gate провален, held/fresh не открыты. Не sweep-ить DRUNet
sigma/descriptor/reciprocity/focal threshold: старый dirty focal verifier
анти-калиброван на этом out-of-distribution emitter-е.
[Full-resolution retrieval adapter](fullres-retrieval-adapter.md) проверил
materially new pre-solver view: stride-one NAF20×20 обучался end-to-end через
frozen d64 SocketMatcher по exact neighbour ranking под независимыми
noise/blur/brightness/JPEG corruptions; raw evidence сохранялся параллельно.
На source-disjoint local16 step400 дал лишь `+0.057 pp` pooled R@1 и
`+0.272 pp` R@5, reciprocal precision `−0.072 pp`, но raw∪adapter top32
coverage выросла на **`+3.385/+2.389 pp`** right/down. Step100→400 R@5 и
union supply росли, поэтому view сохранён как candidate-supply primitive, но
fixed decoder gate провален: terminal16 и decoder не открывались. Не sweep-ить
checkpoint/fusion threshold на local; следующий consumer должен использовать
дополнительные кандидаты context-aware. Weco retrieval-only steps `119/123`.
[Fixed scale400→1600 retrieval adapter](fullres-retrieval-adapter-scale1600.md)
подтвердил, что этот signal продолжает масштабироваться без смены
architecture/loss/features: внутри одной trajectory pooled R@1/R@5 выросли ещё
на `+0.232/+0.345 pp`, raw-union supply — на `+0.719 pp`, matched reciprocal
precision — на `+0.686 pp`. Step1600 против raw дал R@1 `+0.425 pp`, R@5
`+0.923 pp`, reciprocal `+0.845 pp` и raw∪adapter top32 `+3.844 pp`. Signed
terminal gate провален только по R@1 (`+0.425 < +0.5 pp`), поэтому terminal16 и
decoder не открывались. Positive slope сохранил portable server config и
target-free per-query raw/adapter400/adapter1600 candidate matrices. Следующий
consumer — joint vectorized raw+adapter+DINO verifier, не простой suffix или
nearby threshold/checkpoint sweep. Weco retrieval-only steps `138/139`; step140
не использован.
[Adapter-step400 unique suffix](taska-retrieval-adapter-unique-supply.md)
проверил ровно один такой nominator-only consumer: mutual row/column-top32
reciprocal-rank projection, dedup current+selective+fullres, fixed dirty focal
`>=0`, расширение только existing combined arm. До labels target-free local32
freeze дал `18.34` accepted edges/board и `32/32` control replay. После decode
strong-parent dedup обрушил proposal precision `23.07→4.94%`; focal восстановил
только `17.55%`, ниже prereg `25%`. Control `326.781/5.938` ухудшился до
`325.375/1.656` pairs/exact: pair delta `−1.406`, exact `−4.281`. Local gates
провалены, held/fresh не открывались. Не sweep-ить top-k/reciprocity/focal/tail;
top32 adapter остаётся supply primitive только для materially new calibrated or
context-aware consumer. Weco pair+exact step `132`, parent `102`.
[DINOv2 boundary candidate screen](dinov2-boundary-candidate-screen.md)
проверил retained P29-style foundation emitter на том же opened local16:
direct DINO слаб (`R@1 5.135%`, mutual precision `11.35%`), поэтому strong
replacement gate провален. Но raw∪DINO top32 coverage вырос
`69.724→75.385%` (**`+5.661 pp`**), а raw∪adapter1600∪DINO достиг
**`78.029%` (`+8.305 pp`)**. DINO сохраняет `+4.461 pp` unique coverage
поверх raw+adapter, adapter — `+2.644 pp` поверх raw+DINO. Сохранить оба
pre-solver views для одного нового vectorized learned verifier-а; не повторять
fixed rank/logistic fusion и не отправлять сырой DINO supply в rigid solver.
Weco retrieval-only step `144`, competition test/production не затрагивались.
[Vectorized raw + adapter1600 + DINO edge verifier](tri-emitter-edge-verifier.md)
закрыл resource-stopped P33 materially новым relation-local listwise scorer-ом:
он одновременно видит ordered raw 20-pixel seam, inward gradients и ordered
DINO boundary-token relation, а emitter ranks/scores оставляет auxiliaries.
На source-disjoint local16 pooled R@1/R@5/R@32 выросли на
**`+1.053/+1.234/+0.685 pp`**, при неизменном tri-emitter union `78.029%`.
Но строгая matched reciprocal precision упала `35.093→33.320%`
(`−1.774 pp`) при `41.491%` coverage, поэтому terminal16 и decoder не
открывались. Post-hoc fixed-grid diagnostic показывает полезную
high-confidence голову (`+5.094/+6.569/+6.059/+1.670 pp` precision при
3/5/10/20% coverage), но знак меняется к 30%; локальный threshold/coverage
sweep запрещён. Следующая distinct линия — тот же content model с joint
outgoing/incoming reciprocal-consistency/calibration objective, fit и
confirmation только на новых source-disjoint train panels. Weco pair+exact
steps `149/150`, parent `139`; steps `151/152` не использованы.
[Fixed fullres + focal-gated tail96 composition](taska-fullres-focal-gated-tail.md)
без нового sweep применила уже подтверждённый focal logit-zero protection к
frozen пяти-arm pre-tail winner. Local combo добавил ещё `+1.406` пары поверх
fullres и достиг `320.563`; held добавил лишь `+0.406`, чуть ниже заранее
заданного gate `+0.5`, поэтому fresh не открывался. При этом total delta против
four-arm остался сильным: `+6.188/+4.625` local/held с положительными CI lower,
а held exact восстановился `2.813→3.094` (`+0.281`, CI95 положительный) и стал
численно равен four-arm. Verdict: полезная фиксированная композиция, но её
marginal transfer не подтверждён; standalone fullres voter остаётся текущим
pair-кандидатом. Step89 намеренно не логировался, thresholds/budget на этих
panels не подбирать.
[Отдельная preregistered end-to-end fresh32 confirmation](taska-fullres-focal-gated-tail-fresh32-confirmation.md)
затем проверила неизменную композицию на 16 новых sources × 2 draws после
signed исключения TASKA/fullres lineage. Combo дал **`356.313`** pairs/board и
recall **`0.322747`** против `348.406/0.315585` у same-pass four-arm control:
delta **`+7.906`**, source-cluster CI95 **`[+3.531,+12.969]`**, gate
`mean>=+2, lower>=0` пройден. Обе составляющие перенеслись отдельно:
fullres-control `+5.438 [+1.250,+10.688]`, focal-tail поверх fullres
`+2.469 [+0.344,+4.688]`. Focal-filtered new edges имели `62.34%` precision
и добавили `20.28` true missing pairs/board. Mean exact остался ровно `8.0` у
всех arms, поэтому это подтверждённый pair-oriented recipe, не exact или
official-SSIM promotion; production и официальный best не менялись.
[Fixed 22-feature focal logistic stacker](taska-focal-feature-stacker.md)
объединил 15 dirty-visible TASKA признаков, recovered focal logit и шесть
top-5 focal признаков без feature/parameter sweep. Как пятый portfolio arm он
дал five-minus-four deltas local/held/fresh: pairs
`+0.094/−0.531/+1.094`, exact `+0.656/+0.219/+0.188`; held exact CI95
`[+0.031,+0.469]`, а fresh pair/exact intervals едва пересекли ноль.
Descriptive 96-case aggregate составил `+0.219` пары и `+0.354` exact.
Artifact сохранён как promising optional/exact-oriented fifth arm; four-arm
tail96 остаётся pair default из-за held pair sign reversal.
[Fixed train224 focal logistic stacker](taska-focal-feature-stacker-train224.md)
масштабировал ровно тот же arm без смены модели или гиперпараметров на fixed
indices `0:96 + 128:256`, оставив local32 `96:128` полностью вне fit. Все
добавленные 48 100 edge rows побитово совпали по 15 train256 features и labels.
На local32 five-arm+tail96 ухудшился до `313.906/1.813` pairs/exact против
four-arm `314.375/1.375` и retained train96 `314.469/2.031`; pair deltas
`−0.469` и `−0.563`. Оба fixed gate-а провалены, held/fresh не открывались.
Exact unweighted train224 scale-up закрыт; optional train96 artifact сохранён.
[Fixed focal nonlinear stacker](taska-focal-nonlinear-stacker.md) проверил
ровно существующий 100-tree HGB contract на объединённых 15 TASKA + focal
logit + 6 top-5 признаках, без sweep. На disjoint local32 standalone дал
`309.406` pairs, а добавление его пятым arm почти точно воспроизвело текущий
four-arm+tail96 control: `314.344/1.406` против `314.375/1.375` pairs/exact.
Pair delta `−0.03125`, CI95 `[−1.219,+1.156]`; nonnegative gate провален,
held/fresh не открывались. Этот fixed nonlinear fusion закрыт; nearby HGB
hyperparameters на scored local32 не подбирать.
[Fixed TASKA incidence-GNN](taska-incidence-gnn.md) проверил materially другой
context-aware ranker: два permutation-equivariant block-а агрегировали mean/max
edge states по outgoing source и incoming target внутри board/axis, а bounded
residual добавлялся к frozen focal logit. Fit использовал только source-disjoint
indices `128:256`; local32 `96:128` был исключён. Five-arm pair delta прошла
local/held gates (`+0.344/+0.719`), но на unchanged fresh32 развернулась в
`−0.313` pairs и `−0.094` exact против four-arm. Модель не promoted; nearby
width/block/step sweep на открытых panels закрыт. Нужен другой realised-component
objective или robust selector, а не просто больше capacity той же формулировки.
[Source-disjoint exact synthetic evaluator](socket-matcher-exact-synthetic.md)
убирает recovered-label ambiguity: на первых 8 v2 boards fused local R@1 вырос
`8.37%→13.72%`, а component/QAP decoder поднял adjacency `4.05%→8.40%`, но
direct placement остался около chance.
[Fresh component/anchor diagnostic](socket-component-anchor-diagnostic.md)
локализовал следующий bottleneck: largest top-144 component имеет в среднем
`42.42` tiles, но лишь `17.31%` trusted translation purity; ни одна из 24
largest components не была внутренне цельной. Generic centre/texture-centre
anchor не подтвердился, поэтому следующий шаг — precision-first component
graph, а не более сильный center prior.
[Precision-first Socket decoder](socket-precision-first-decoder.md) подтвердил
переносимость dirty-only confidence: trusted edge precision `78.24%→75.15%`, а
component purity на fresh offset-2816 выросла `35.78%→84.91%`. Но sparse
standalone layout резко проиграл default decoder по primary direct placement
`0.001447→0.000940`, adjacency `7.83%→2.51%` и raw SSIM
`0.106081→0.095503`. Variant rejected; pure components стоит использовать как
immutable seeds внутри более полного global solver, не как всю раскладку.
[2x2 commutative-cycle diagnostic](socket-commutative-cycle-diagnostic.md) на
том же уже открытом offset-2304 не нашёл дополнительного usable coverage. K4
поднял precision confidence-подмножества `78.24%→84.40%`, но сохранил лишь
`48.51%` правильных confidence edges; cycle-only edges имели только `22.72%`
precision. K8 почти redundant, K16 saturated chance closures. Новый decoder не
запускался; cycle support можно оставить лишь feature для будущего learned
calibrator.
[Learned hard-edge calibrator](socket-hard-edge-confidence-calibration.md)
реализовал именно этот следующий шаг на exact-synthetic source-disjoint
`fit32/confirm16`. Один frozen logistic threshold перенёс precision
`80.00%→77.95%` и на confirmation дал `25.19` правильных edges/board против
`22.81` при `77.66%` у прежнего fixed heuristic. Fit-precision-matched
confidence top-16 сохранил `82.42%`, но только `13.19` correct edges/board.
Калибратор promoted как edge-selection primitive; layout decoder в этом
эксперименте не запускался.
[Calibrated-order decoder144](socket-calibrated-order-decoder144.md) затем
проверил continuous probability только как порядок greedy constraints внутри
обычного полного decoder. На fresh exact-synthetic-24 adjacency выросла
`9.379%→9.749%`, paired 95% CI delta `[+0.001049,+0.006348]`, 18/24 wins;
correct selected edges выросли `101.83→103.88`, largest component уменьшился
`43.83→34.63`, pairwise purity — `8.97%→11.11%`. Exact tiles и raw SSIM
улучшились лишь descriptively, aligned tiles были flat. Порядок promoted для
следующих Socket decoder experiments, но absolute placement всё ещё не решён.
[d64 hard-edge calibration](socket-d64-hard-edge-confidence-calibration.md)
перенесла тот же frozen protocol на новый d64 checkpoint, исключив его полную
1056-source lineage, `exact-synthetic-v2-d64-source16-draw2` и все прежние
exact panels. Learned threshold перенёс precision `80.00%→78.42%` и дал
`73.13` correct edges/board против `67.75 @ 76.99%` у fit-precision top-88.
Но coverage gain составил лишь `+7.93%`, а paired correct-edge CI пересёк ноль
`[-2.17,+12.92]`; frozen material gate требовал `+15%`. Поэтому новый d64
decoder exact24 panel не открывался и d32 promote-order автоматически на d64
не переносится.
[Global cyclic translation placer](socket-global-cyclic-translation.md) впервые
дал подтверждённый absolute-placement gain поверх d64 decoder144. Он выбирает
общий циклический origin всей готовой доски только по right/down seam cuts и
четырём learned dustbin border probabilities, сохраняя строгую перестановку и
всю component geometry вне двух cuts. На 48 fresh exact-synthetic boards
правильные абсолютные tiles выросли `40→58` (`0.833→1.208`/board),
source-clustered 95% CI прироста `[+0.041,+0.875]`; adjacency снизилась лишь на
`0.060 п.п.`. Frozen weight `5.0` promoted как opt-in post-decoder primitive;
center/background heuristic не используется.
[Единый legal Socket full-cycle](socket-legal-full-cycle.md) затем связал уже
зафиксированные d64 OT, decoder144, cyclic-border5 и target-blind
RGB/luma→NLM-h20 tail на одних и тех же exact-synthetic `16×2` случаях. Local
R@1/R@5 равны `17.765/35.734%`; cyclic layout дал `1.406` exact tile/board и
`13.103%` adjacency, а legal tail поднял raw SSIM `0.10399→0.25536`.
Protocol-v2 runtime: `0.192` с tail и `1.266` с full cycle. Все raw
canvases — строгие перестановки original upright tiles. Protocol-v2 напрямую
перепроверяет actual checkpoint lineage, source-report hashes и атомарную
label-free freeze; все geometry/SSIM числа совпали с v1. Панель reused и
параметры не выбирались; это сопоставимый full-cycle scoreboard, не новая
confirmation и не доказательство правильной раскладки.
[Absolute coordinate sorter](absolute-coordinate-sorter.md) напрямую обучил
permutation-equivariant row/column/slot head на exact synthetic shuffle без
embedding shuffled index. На frozen source32×draw2 direct Hungarian показал
абсолютный signal (`1.578` exact tiles/board, `35.02` correct rows), но уничтожил
adjacency. Практичный component-unary arm сохранил adjacency и descriptively
поднял exact `1.141→1.516`, rows `23.63→27.77`, columns `24.88→26.38`; row CI
положителен, exact CI `[-0.188,+0.984]` пересекает ноль. Поэтому primitive
сохранён для component-aware scale-up, default pipeline не менялся.
[Component-translation scale-up](absolute-coordinate-component-translation-scale.md)
обучил заранее выбранный exact feasible-shift CE на 2048 sources / 1600 steps.
На новой source64×draw2 панели train-consistent unary сохранил adjacency и дал
exact `1.219→1.531` tile/board, но delta `+0.3125`, CI
`[-0.1563,+0.8359]` не прошёл material gate `+0.5` с положительным lower bound.
Rows выросли устойчиво `24.53→29.31`, CI `[+2.34,+7.23]`; component-shift
top-1 `0.679%` против chance `0.184%`, но NLL остался близко к uniform.
Повторять тот же capacity sweep не надо; default не менялся.
Same-panel axis ablation усилил rows у row-only `w=.03` до `30.20`, но exact
остался `1.461`, слабее провалившего gate joint arm; cyclic synergy не было.
Поэтому новая панель на axis-вариант не открывалась.
[Coordinate cyclic origin](coordinate-cyclic-origin.md) отдельно агрегировал
row/column log-probabilities всех tiles только для выбора одного из 576 общих
cyclic origins, не двигая компоненты относительно друг друга. На том же уже
открытом source64×draw2 panel coordinate-only дал `1.070` exact/board,
row-coordinate/Socket-column `1.234`, а лучший equal coordinate+border blend
`1.562` против frozen border5 `1.508`: delta лишь `+0.055`, CI
`[-0.391,+0.508]`, adjacency loss `0.170 п.п.`. Preregistered `+0.25` gate не
пройден, fresh panel не открывался и default не менялся.
[Whole-layout cyclic-origin CNN](whole-layout-cyclic-origin-cnn.md) проверил
materially different nonlinear conversion: 109 dirty-visible channels были
собраны в actual decoder144 grid, а 45,345-параметрическая circular/dilated CNN
оценивала все 576 global rolls без tile ID или position embedding. Capacity
smoke прошёл, fit256/400 использовал exact→mixed→decoder curriculum. На fresh
manifest-train16 learned roll descriptively поднял exact `0.750→0.938`
tile/board (`+0.1875`), но adjacency упала `13.123→12.777%` (`−0.345 pp`).
Главное, model-specific signal отсутствовал: best-roll R@1/R@5 `0/0%` против
uniform `0.195/0.976%`, NLL был хуже uniform на `0.054`. Gate = fail-stop;
16/16 layouts strict, holdout/test не открывались, capacity/blend sweep закрыт.
[Dedicated full-resolution frame-side origin](frame-side-origin.md) затем
изолировал более узкую гипотезу: 51,865-param stride-one 20×20 модель
предсказывает ровно 24 top/bottom/left/right tiles, а неизменный decoder144
только globally roll-ится по integer frame hits с raw-cut tie-break. На
source-disjoint train32 frame F1 ухудшился `7.780→7.259%`, exact
`1.531→0.656` tile/board, adjacency `13.465→13.157%`; gate fail-stop.
Same-opened target-assisted audit показал структурный ceiling: даже истинные
frame sets дают лишь `1.156` exact, хотя oracle best cyclic roll того же layout
даёт `13.031`. Значит, marginal frame membership не определяет origin
фрагментированного layout; nearby capacity/denoiser/weight sweep и fresh64
закрыты, нужен translation-consistent internal component signal.
[Component absolute-translation voting audit](component-absolute-translation-voting-audit.md)
проверил этот следующий кандидат без нового target access и без model code.
Simple shared-roll vote закрыт как дубль coordinate/whole-layout origin.
Independent pure-component oracle на exact source64×draw2 покрывает `48.32`
tiles/board до collision packing, но legacy recovered dev24 показывает, что
largest components крайне impure: top-4 содержат `101.88` tiles, oracle mode
support лишь `18.42`, internally exact support `0`. Conditional go оставлен
только board-conditioned native-pixel component model с отдельными purity и
feasible-shift heads плюс strict collision-aware packing; larger d32 MLP,
size-confidence и population-field варианты не повторять.
[Independent component absolute placer](component-absolute-placer.md) активировал
ровно этот conditional-go: 140,561-param native-pixel/lattice/set model отдельно
предсказывал exact-purity и feasible joint offset, fit-only selector мог
заанкорить максимум один компонент, затем strict pack сохранял все 576 original
upright tiles. Purity signal перенёсся на fresh train32 (`AP 0.4254→0.5937`,
`1.396×`), но не прошёл gate `2×`; pure-offset top1 на fit-cal был лишь
`0.2119%`. Conservative calibration поэтому выбрала fallback-only и 0 anchors:
exact/adjacency treatment полностью совпали с cyclic5 comparator (`1.781`
tile/board, `13.242%`, 32/32 strict). Gate fail-stop; independent absolute-offset
head/one-anchor packing закрыты без fresh64/holdout/test. Положительный purity
ranking не является абсолютным position signal и не отменяет отдельно
подтверждённый relative hard-edge primitive.
[Transpose-equivariant coordinate continuation](coordinate-transpose-equivariance.md)
проверил genuinely distinct перенос сильной row-головы на column через
whole-board transpose как model-only view. На том же opened source64×draw2
panel bounded head-only train192/300 поднял classifier columns на `+1.695`
tile/board, но CI `[-0.516,+3.883]` пересёк ноль и threshold `+2` не пройден;
descriptive exact decoder delta был `−0.328`. Все layouts остались строгими
перестановками original upright tiles. Development gate failed, fresh panel не
открывалась, default не менялся.
[Explicit component-shift fallback](component-shift-head-fallback.md) устраняет
один конкретный mismatch первого scale loss: новый head действительно получает
decoder144 component membership, внутренние relative coordinates,
size/shape/confidence и board context, а не только суммирует независимые
tile→slot logits. Impure components обучаются по dominant feasible translation
с ненулевым purity/size weight; output точно конвертируется в существующий
tile×slot component unary строгого decoder-а. Bounded train-only run обучил
ровно 60,208 параметров на 2,048 source-disjoint synthetic sources за 800
steps. Tail100 дал row `7.152% vs 4.485%` chance, но row NLL gain лишь `1.401%`;
column `5.203% vs 4.465%` и NLL gain `0.0128%`. Supported tiles были
`1.830` против chance `1.085`, тогда как predeclared material delta требовала
`41.608`. Gate = stop, `quality_panel_authorized=false`; checkpoint rejected,
exact panel и default не открывались.
[d64 component-relation reranker](component-relation-reranker.md) проверил
материально иной local target: две реальные decoder144-компоненты,
collision-free relative translation, pooled d64 member tokens/coordinates и
все индуцированные Socket/OT contacts. Learned score дал source-disjoint R@1
`+3.366 pp` и R@5 `+4.934 pp`; отдельный 68-параметрический calibrator
реплицировал на confirm24 R@1 `+3.494 pp`, R@5 `+4.367 pp` и top32 precision
`15.234→32.422%`. Однако conversion в layout не подтвердился: promising
opened40 `v1.1+cyclic5` gain `+0.825` на строгом fresh source64×draw2 стал
`−0.09375` exact tile/board, source-CI `[-0.4531,+0.2500]`, W/T/L
`23/15/26`; adjacency выросла на `+0.0665 pp`. Fresh gate = fail-stop,
128/128 outputs были строгими перестановками, competition test не открывался.
Не повторять hard-edge+cyclic treatment с близкими cap/bonus weights; relation
confidence сохранять только как primitive для материально другого solver-а.

Новый [fail-closed frozen evaluator](frozen-final-evaluation.md) фиксирует ровно
три legal arms, все predictions до target decode и single-use holdout receipt.
Frozen v1 дал `0.257664`, не прошёл свой абсолютный calibration gate `0.28`,
поэтому его holdout не открывался. Отдельный неизменённый fallback прошёл ровно
один свежий holdout `offset=96, count=96`: final `0.253128`, control `0.243320`,
gain `+0.009808`, paired 95% CI `[+0.009010, +0.010626]`, 96/96 wins. Его
read-only receipt запрещает повтор; менять v1 или fallback задним числом нельзя.
Итоговый scope, hashes и production-команды собраны в
[финальном handoff](../final-solution.md).

## Короткий итог

Frozen fallback прошёл production: опубликованы 700 predictions в
`outputs/compliant-submission/predictions/` и
`outputs/compliant-submission/submission.zip` с SHA-256
`7c36307af0ea821c8a5fbf3139323ece332744dcf59a413198dd96d5a2f619bf`.
Attestation SHA-256 равен
`5323d05b71b56645a7ad2acab5276187035c4e1e9de07c3fb34821b60c688c8f`,
runtime-manifest digest —
`15c88d3def7bccc9c0fd0fe082ae848e9e768af89fadf363b8bb6ae4f31d3d6f`.
Встроенный validator и отдельный повторный entrypoint оба дали
`METHOD_COMPLIANT_LAYOUT_ACCURACY_UNPROVEN`; independent report находится в
`outputs/compliant-submission/independent-validation.json`. Точность скрытой
раскладки по-прежнему не доказана, поэтому это технически проверенный artifact,
а не заявление о submission-ready решении, прошедшем ручную приёмку. Новый
[target-blind postassembly harmonizer](postassembly-harmonizer.md) подтверждён на
строгой generic сборке: RGB seam offsets -> bounded luma -> RGB NLM20 дал
`0.315045` против `0.303459` у NLM20, gain `+0.011586`, paired 95% CI
`[+0.009674, +0.013523]`, 24/24 wins. Все predictions были заморожены до target
decode, а все 576 tiles прошли raw permutation audit. Это preferred
postassembly order на calibration; выполненный production visual audit выявил
critical manual risk уже у полной сцены. [Tile-wise DualNAF](tilewise-dualnaf.md) подтвердил, что
independent 20x20 inference исправляет прежний full-canvas failure, но перед
сильным NLM tail проиграл control на `−0.017/−0.030/−0.052` при 5/10/20
проходах; эта композиция имеет статус **rejected-for-final**. Отдельная
[matcher-only проверка DualNAF](dualnaf-matcher.md), где модель не рендерила
выходные пиксели, тоже провалила preregistered gate: 50/50 fusion дал
`−0.001522` tail SSIM при лишь неопределённом `+0.002340` adjacency; disjoint
confirmation не открывался. Новая
[bounded residual проверка](dualnaf-bounded-residual.md) нашла малый, но
повторяемый pixel-tail signal: `alpha=0.125` дал `+0.001763`, CI
`[+0.000816,+0.002861]`, 20/24, затем на adaptive reused-calibration-48
`+0.001096`, CI `[+0.000688,+0.001555]`, 38/48. Однако второй mean равен лишь
`0.251208 < 0.27`; оба preregistered primary gate провалены, обе confirmation
панели остались закрыты. Это safe minor correction, не решение layout. M420 подтвердился при
строгой биекции и меняет правильную постановку diagnostics/labels, но остаётся
target-assisted oracle. Дешёвый candidate pool достаточно широк для исследований,
однако первый content-multipositive listwise verifier провалил калибровочный gate
на inference-relevant `all` rows; эту глобальную формулировку не следует повторять.

Последний [candidate-k16 / train256 pairwise ranker](edge-ranker-k16-scale.md)
закрыл и более широкий learned-edge scale route. На fresh calibration-24 он
улучшил adjacency `0.032684→0.062689`: gain `+0.030005`, paired 95% CI
`[+0.027325,+0.032646]`, 24/24 wins; translation-aligned placement вырос на
`+0.002170`. Но frozen compliant h20x1 endpoint ухудшился
`0.247168→0.237782`: delta `−0.009386`, CI
`[−0.016116,−0.002948]`, 8/24 wins. Gate прошёл только 2 из 5 условий, то есть
3/5 **FAIL**; confirmation, holdout, test и production integration не
открывались. Authoritative report:
`outputs/edge-ranker/scale-raw-k16-train256-cal24-offset228/final-tail-primary/report.json`,
SHA-256
`6fe6790f470c3e39a28d3c5c050feac1cd08623b7db3677d4f2102d7028ddad9`.

Следующий [conservative mutual-edge fusion](edge-ranker-conservative-fusion.md)
проверил, можно ли использовать тот же local k16 signal без полной замены
bilateral scores. Non-destructive union сохранял каждый bilateral mutual-best
edge и добавлял только 8–32 learned mutual proposals. Лучший arm на reused
calibration `360:384` повысил final SSIM `0.242113 -> 0.247861`
(`+0.005748`) и adjacency `0.037704 -> 0.041289`, причём adjacency CI lower был
положительным. Но final paired CI пересёк ноль (`lower=-0.000221`), а absolute
gate `0.27` провален. Ни один из пяти preregistered arms не прошёл полный gate;
confirmation `444:468`, holdout, test и production integration запрещены.

Новый [dense legal single-pass NLM screen](dense-safe-tail.md) проверил весь
допустимый промежуток `h=21..29` после того же RGB+luma harmonizer. Это честно
помеченный reused-calibration experiment: legacy calibration-700 уже открывал
все его targets. На ranked records `300:336` каждый `h=21..29` улучшил `h20` на
36/36 boards; maximum manual-safe `h28` дал `0.257032`, gain `+0.010292`, paired
95% t CI `[+0.009397,+0.011187]`. Однако preregistered absolute gate `0.27` не
прошёл ни один arm; `h29=0.257957` дополнительно нарушил mean Laplacian-retention
bound. Confirmation `384:420` не открывалась, frozen production не менялся.

Первый [legal BM3D screen](bm3d-legal-screen.md) проверил official PyPI
`bm3d==4.0.3` после strict layout и frozen RGB+luma. Все pure BM3D
`sigma=.12/.16/.20`, cascade `.16→NLM10` и 50/50 blend с NLM20 проиграли
baseline на 24/24. Лучший BM3D candidate получил `0.243646` против
`0.253976`, delta `−0.010330`, CI `[−0.011821,−0.008800]`. Outputs сохранили
detail/clipping bounds и были distinct, но ни один SSIM gate не прошёл;
confirmation `252:276` не открывалась. Package использовался ephemeral только
в non-commercial research scope, submission/production не содержат его code
или binaries.

[Pretrained discriminative DRUNet tail](pretrained-drunet-tile-tail.md) дал
первый большой и полностью стабильный neural-restoration gain: train-selected
`sigma40 -> one NLM h28` улучшил h28 на reused calibration `384:408` на
`+0.005654`, paired CI `[+0.005064,+0.006254]`, 24/24 wins; против h20 gain
равен `+0.016700`, также 24/24. Все tile-identity, detail, color, grid и clipping
bounds прошли. Но absolute mean составил только `0.251061<0.27`, поэтому
confirmation `600:624` не открывалась. Компонент перспективен поверх нового
сильного layout/tail, но текущая preregistration отвергнута для promotion.

Новая фиксированная
[DRUNet40 + protected h28/h40 композиция](pretrained-drunet-protected-stack.md)
впервые перенесла этот neural gain через absolute primary gate на широкой
reused-calibration `264:384`: D получил `0.271644`, улучшил original h28 на
`+0.007606` (CI `[+0.007237,+0.007967]`, 120/120) и DRUNet+h28 на `+0.001908`
(CI `[+0.001777,+0.002037]`, 119/120). Все 16 structural/color/grid/clipping
bounds прошли, затем root просмотрел все 120 triplets с severe-artifact count
`0`. Неизменённая confirmation `408:528` повторила relative result: `+0.007888`
к h28 и `+0.002005` к C, оба CI positive, 120/120 wins, все safety bounds PASS.
Но absolute confirmation mean равен `0.262817<0.27`; это единственный failed
gate. Статус — **reject for promotion**, production не менять.

Последующее [измерение того же неизменённого D на всех 700 calibration
records](pretrained-drunet-protected-stack-all700.md) сняло panel uncertainty:
exact mean равен `0.268270`, то есть заранее заданный broad interval
`[0.27,0.28]` не достигнут. Только 3/10 fixed folds по 70 records оказались выше
`0.27`; fold means лежат в `[0.256055,0.286401]`. Raw provenance прошёл
`700/700`, но safety также narrowly failed на одной board: minimum chroma
retention `0.596425<0.60`. Поэтому fail-closed holdout-700 не открывался,
production не менялся. Этот масштабированный результат считать
authoritative aggregate для fixed D и не повторять выборочные панели.

Отдельный [official DRUNet40 matcher-view diagnostic](drunet-matcher.md)
проверил ту же модель строго как источник bilateral edge scores, никогда не как
renderer. Frozen fusion `.50` на disjoint train verification-8 повысил exact
adjacency на `+0.004642`, CI `[+0.000709,+0.008576]`, но ухудшил
translation-aligned placement на `-0.002170`, CI
`[-0.004330,-0.000010]`. Final F gain `+0.006846` имел CI, пересекающий ноль,
и только 4/8 wins; на selection-8 лучший fusion уже проигрывал baseline по F на
`-0.002122`. Robust train gate провален, поэтому calibration не открывалась и
этот exact matcher route закрыт.

Финальный [ultimate legal stack](ultimate-stack.md) без sweep сложил h28,
non-destructive cap08 fusion и same-index DualNAF `alpha=.125`. На historically
exposed calibration `420:444` кандидат D получил `0.250107`: лучше A/h20 на
`+0.009100`, но хуже strong B/h28 на `-0.002374`; D-vs-B CI lower
`-0.006704`, wins лишь 10/24. Fusion adjacency CI lower `-0.000491`, translation
delta `-0.000072`; absolute `0.27` тоже провален. Все detail/grid safety bounds
и severe-artifact=0 прошли, однако все scenes остались мозаичными. Confirmation
`444:468`, holdout, test и production не открывались.

[Frozen DualNAF alpha=.125 + h28 stack](dualnaf-stack.md) достиг `0.267429` на
reused calibration `636:668`, но также провалил gate. D улучшил h20 baseline на
`+0.011977` с положительными t/bootstrap CI и 32/32 wins. Против pure h28
control gain был лишь `+0.000608`: t CI `[-0.000147,+0.001364]`, bootstrap CI
`[-0.000020,+0.001401]`, 17/32 wins. Safety и узкий manual comparison прошли,
но absolute `0.267429<0.27`; confirmation `668:700` не открывалась.

[Edge- and grid-protected flat-region NLM](edge-protected-nlm.md) первым прошёл
absolute primary gate: E (`h40` только в target-blind flat regions, `h20` у
Sobel content edges и всех 20-pixel fragment boundaries) получил `0.274239`,
gain к h20 `+0.004040`, CI `[+0.003437,+0.004643]`, 24/24 wins и root manual
PASS с severe-artifact count `0` на reused calibration `120:144`. Однако
неизменённый E на disjoint confirmation `144:168` дал только `0.249786<0.27`.
Он снова выиграл у A 24/24 и прошёл все relative/safety bounds, но failed
absolute check запрещает promotion. Production не менялся; те же
`h35/h40, threshold 30/40` arms не повторять без нового механизма.

[h28-safe / h40-flat protected NLM v2](edge-protected-nlm-v2.md) исправил
главный недостаток v1: exact h20-derived t40 mask сохранился, но protected
pixels берутся из independent h28, flat pixels — из independent h40. На
reused-calibration primary `60:120` F получил `0.280087`, улучшил h28 на
`+0.002074`, CI `[+0.001894,+0.002254]`, 60/60 wins и прошёл root manual review
с severe-artifact count `0`. На unchanged confirmation `0:60` relative result
повторился (`+0.002220`, CI `[+0.002042,+0.002397]`, 60/60), но absolute mean
был `0.266852<0.27`. Это единственный failed gate: F — подтверждённый stable
legal tail improvement, но не promoted production solution. Descriptive pooled
gain к h28 на 120 boards равен `+0.002147`, 120/120; layout problem он не решает.

[Residual-after-h20 tile restorer](after-h20-restorer.md) впервые обучил
специальную shared 20x20 NAF-style сеть на train-only clean identities в той же
предсказанной/ошибочной buddies96 раскладке, с входом pre-h20+h20. Train-only
diagnostic дал decisive reject: даже минимальный `alpha=.125` снизил mean SSIM
`0.253924 -> 0.245621`, проиграл 24/24 и повысил mean/max grid ratio до
`1.169/1.310` от h20. Primary predictions `192:216` были target-free frozen, но
по stop-rule ни один evaluation target не открывался; confirmation `216:240`,
holdout/test/production также остались закрыты. Exact checkpoint/blends не
повторять.

[Раздельный luma/chroma NLM screen](nlm-luma-chroma.md) закрыл ещё одну ранее
непроверенную вариацию. На reused calibration `468:504` усиление только
`hColor` при фиксированном `h` слегка ухудшало SSIM: `20/20=0.247069`,
`20/24=0.246967`, `20/28=0.246869`. Лучший arm `h24/hColor20=0.253257`
получил весь прирост от общей яркостной силы и не достиг absolute gate `0.27`.
Ни один arm не promoted; confirmation `504:540` не открывалась.

Отдельный [manual-safety audit силы colored NLM](nlm-strength-manual-safety.md)
зафиксировал fail-closed tail: **ровно один проход `h=20`**, с `h=15 x1` как
консервативным fallback. На calibration 96:108 он дал `0.307547` для plain
buddies96 и `0.304610` для atlas layout и прошёл ручной просмотр `6/6`. Все
multi-pass tails и все `h>=30` отклонены для финальной посылки. Metric-only
максимум `h120 x10 = 0.412984` визуально схлопывает детали в blobs и имеет статус
**REJECT / DO NOT SUBMIT**.

Финальный [visual audit](../../outputs/compliant-submission/visual-audit/REPORT.md)
подтвердил критическое содержательное ограничение: ни один из 24
детерминированно выбранных outputs не образует уверенно читаемой целостной сцены.
Медианное сохранение внутриточного градиента равно `0.321`, а медианное отношение
перепада на сетке 20×20 к внутреннему перепаду — `2.94`. Значит, file/provenance
PASS не снимает очень высокий риск ручного отклонения.

## Обязательный manual-compliance gate

Уточнение организаторов требует восстановить правильное расположение и качество
всех 576 фрагментов и запрещает их подмену/уничтожение. Поэтому constant RGB,
SSIM-parametric constant, population-atlas и low-frequency-only canvases имеют
статус **NONCOMPLIANT / DO NOT SUBMIT**, даже если их SSIM выше solver-а. Они
сохраняются только как diagnostics рассогласования метрики и задачи.

Полный fail-closed contract и текущая матрица допустимости находятся в
[manual submission compliance](../submission-compliance.md); machine-readable
attestation schema — в `configs/submission-compliance.schema.json`. Текущий
700-image ZIP предъявлен contract-у и прошёл его дважды, но намеренно получил
ограниченный статус `METHOD_COMPLIANT_LAYOUT_ACCURACY_UNPROVEN`. Для
structurally compliant path bilateral E14
→ ORBIT buddies96 → one-to-one tile assembly новый manual gate разрешает только
single-pass colored NLM `h20` (`h15` fallback); прежний `h10 x20` cap отменён.
В production run ZIP прошёл встроенный и повторный независимый validator; даже
этот двойной PASS подтверждает метод и биекцию, но не правильность скрытого
layout.

## Ранжирование направлений

`Confidence` относится только к сформулированному выводу, а не к ожидаемому
leaderboard score.

| Ранг | Направление | Потенциал | Confidence | Решение |
|---:|---|---|---|---|
| 1 | [RGB+luma harmonizer -> RGB NLM](postassembly-harmonizer.md) | Высокий как target-blind postprocess на actual strict layout | **Высокий** на calibration-24 | **promote order**; final visual audit всё равно critical из-за full-scene layout |
| 2 | [Single-pass colored NLM safety](nlm-strength-manual-safety.md) | Высокий как готовый postprocess | **Высокий** на calibration audit + manual review | **h20 x1**, h15 x1 fallback; multi-pass/h>=30 reject |
| 3 | [Dense legal h21..29 tail](dense-safe-tail.md) | Небольшой монотонный metric lift без layout improvement | **Высокий** для reused-calibration screen; не generalization | **reject for 0.27 target**; h28 safe ceiling, production не менять |
| 4 | [Decoupled luma/chroma NLM](nlm-luma-chroma.md) | Низкий: chroma-only усиление отрицательно | **Высокий** для reused-calibration screen | **reject-as-tested**; не повторять hColor sweep |
| 5 | [Legal BM3D screen](bm3d-legal-screen.md) | Низкий для текущего noisy mosaicked canvas | **Высокий** на reused calibration-24 | **decisive reject-as-tested**; best BM3D `−0.010330`, 0/24 |
| 6 | [Pretrained tile-wise DRUNet](pretrained-drunet-tile-tail.md) | Высокий как bounded restoration component, layout не исправляет | **Высокий**: train verification + reused calibration-24, 24/24 | **reject for absolute gate**; `+0.005654` vs h28, но `0.251061<0.27` |
| 7 | [Official DRUNet40 matcher view](drunet-matcher.md) | Низкий для exact bilateral fusion: local adjacency не переносится в placement | **Средний**: reused train selection8 + disjoint verification8 | **reject-as-tested**; translation `-0.002170`, calibration закрыта |
| 8 | [Content-equivalent target при биекции](content-substitution.md) | Высокая информационная ценность для labels/diagnostics | **Высокий** для metric slack; **низкий** для target-free recovery | **keep diagnostic**, не считать solver-ом |
| 9 | [Dirty-only analytic candidate supply](candidate-supply.md) | Средний: даёт shortlist для принципиально нового evidence | **Средний** из-за target-assisted labels | **keep supply gate**, но не считать fixed-budget/global win |
| 10 | [Position-aware listwise verifier](content-verifier.md) | Средний только как exact-edge research auxiliary | **Средний** на calibration-24 | **reject-as-tested** глобальный content-multipositive ranker; decoder не запускать |
| 11 | [Pairwise edge ranker k16/train256](edge-ranker-k16-scale.md) | Высокий local adjacency, отрицательный final endpoint | **Высокий** на preregistered calibration-24 | **reject-as-tested**; gate 3/5 fail, confirmation не открывать |
| 12 | Gray/cheap pixel tails | Низкий, кроме жёсткого latency budget | **Высокий** для measured roster | gray guard **reject**; Gaussian `sigma=1` — лишь fallback |
| 13 | [Bounded tile-wise DualNAF residual](dualnaf-bounded-residual.md) | Малый безопасный pixel-tail gain, недостаточный для target range | **Средний**: две reused-calibration панели, CI positive | **reject-as-scaled**; `+0.001096`, но `0.251208<0.27`; confirmations закрыты |
| 14 | [Tile-wise DualNAF -> harmonizer -> repeated NLM](tilewise-dualnaf.md) | Высокий без final NLM, отрицательный в сильной композиции | **Высокий** на fresh calibration-12 | **rejected-for-final**; не масштабировать |
| 15 | [DualNAF pre-denoise -> E14 matcher](dualnaf-matcher.md) | Низкий для текущего frozen checkpoint | **Средний** на fresh calibration-12 | **reject-as-tested**; confirmation не открывать |
| 16 | [Full-strength global population assignment](global-population-layout.md) | Низкий для generic train atlas | **Высокий** на fresh calibration-24 + manual canvas review | pure Hungarian decisive reject; weights 0.25/1.0 не масштабировать |

Наиболее перспективный следующий исследовательский путь — не ещё один вариант
того же verifier, а **новый inference-visible semantic/multi-tile сигнал**, который
сможет использовать content slack при глобальной one-to-one сборке. До такого
сигнала candidate supply и exact-edge auxiliary остаются инструментами анализа,
а не разрешением запускать дорогой decoder.

## Общий frozen protocol

- Manifest: `data/interim/validation_manifest.json`.
- Конфигурация: `configs/validation.yaml`.
- Разбиение: `train=5600`, `calibration=700`, `holdout=700` из 7 000 train-пар.
- Protocol digest:
  `2a9e3b74f7defa8c00846a05eb598fd263fd16c2787c70e77d3b7a4b585bfbf4`.
- Общий selector: `aiijc-puzzle-experiments-v1`, seed `20260829`.
- Calibration-48 selection digest:
  `5b4ff9b7e14b8fbb3e6522a4398c912d477e5ec7c877ad8242e5f8c7c3b0e8eb`.
- Holdout-48 selection digest:
  `941f272377dad2aa3edb9092b89582bd4fa04f6db1ac80aa254b6edefb781e40`.
- Verifier scale-up: первые 128 выбранных train boards, selection digest
  `96b3bd33369c43da6ad963f3cba803603f4004f16a8c4ed3f93ebcd6de0b3bda`;
  calibration boards 12:36, digest
  `6e4b988948186c91dd0947da1df99ae988fdb44b1ff1d0ba7ed3dab68a3fa632`.
- Competition test и исторические 18 clean test references не используются.

Manifest строится и проверяется одной командой:

```bash
uv run python scripts/build_validation_manifest.py --run
```

Каждый результат хранит filenames, protocol/selection digest и конфигурацию;
SHA-256 входов закреплены общим content-addressed manifest. Отчёты дополнительно
записывают доступные code/artifact hashes, но ранние pixel-tail и M420 JSON не
дублируют их внутри каждого файла. Название `holdout` относится к независимому
split текущего workspace: прежние ветки не имели единого manifest, поэтому оно не
гарантирует, что stem никогда не появлялся в старой истории.

## 1. Исторический pixel-tail h9 и актуальный final cap

Числа `h=9` ниже сохраняются как результаты первоначального frozen bakeoff и не
являются текущей production-настройкой. Поздний cross-strength/manual audit
заморозил для production ровно `h=20 x1`; исторические значения не
пересчитывались и не подменялись.

На одинаковых target-assisted low-resolution Hungarian layouts:

| Split | Raw RGB SSIM | Colored NLM `h=9` | Gain | Paired 95% CI | Wins |
|---|---:|---:|---:|---:|---:|
| calibration-48 | 0.439085 | **0.571616** | **+0.132531** | `[+0.123724, +0.141338]` | 48/48 |
| holdout-48 | 0.430621 | **0.557442** | **+0.126821** | `[+0.114821, +0.138820]` | 48/48 |

На более сильном full-resolution inferred holdout layout gain почти тот же:
`+0.128474`, 95% CI `[+0.116617, +0.140331]`. CPU runtime самого transform —
примерно `0.105 s/image`.

Строгая граница вывода: это paired postprocess win на почти правильных
target-assisted layouts, а не измерение качества layout solver-а. Позднейший
[cross-strength manual audit](nlm-strength-manual-safety.md) отдельно проверил
`h=10..120` на actual strict layouts и разрешил только single-pass `h20` как
final candidate (`h15` fallback).

### Закрытые pixel-tail варианты

- E18b gray guard хуже unguarded `h=9` на всех 48 holdout boards, средняя разница
  `-0.003589`; без внешнего safety-инварианта его не использовать.
- Gray-only `h=9` быстрее примерно в 2.56 раза, но хуже colored `h=9` на
  `0.012655` в среднем; универсальный router не найден.
- Gaussian `sigma=1.0` дал `+0.072163`, 48/48, и годится только как почти
  бесплатный extreme-latency fallback.
- Target-assisted cell oracle добавил к `h=9` лишь `+0.003489`; сложный
  raw-vs-NLM cell router имеет малый измеренный headroom.

## 2. M420: подтверждённый content slack, не recovery method

Главный confirmatory holdout-48 результат:

| One-to-one назначение | Clean oracle | Recovered raw dirty | Dirty + NLM `h=9` | Exact placement | Duplicate use |
|---|---:|---:|---:|---:|---:|
| Hungarian derangement | **0.533053** | 0.258824 | **0.383725** | 0 | 0 |

Для unconstrained nearest-other clean score равен `0.572959`, но один source tile
используется в среднем 262.9 раза. Цена строгой биекции на clean canvas — только
`0.039906`, поэтому historical M420 не был создан duplicate reuse. На calibration-48
соответствующие значения bijection равны `0.526729`, `0.259284`, `0.388504`.

NLM повышает dirty bijective proxy на `+0.124901`, 95% CI
`[+0.113457, +0.136345]`, 48/48. Однако и `0.533`, и `0.384` остаются
target-assisted: clean target выбирает substitute и помогает восстановить
train permutation. Эксперимент доказывает рассогласование exact identity с SSIM
и оправдывает content-aware diagnostics/labels, но **не показывает, как найти эти
замены на test**.

## 3. Candidate supply: gate достаточности, а не solver

Четыре dirty-only MGC+SSD emitter-а (`raw`, `tile_z`, `bilateral`, `gray`) дают на
строгом mapping-trusted subset:

| Split | Direction | Pool | Фактический mean budget | Exact recall | Content recall, RMSE≤20 |
|---|---|---:|---:|---:|---:|
| calibration-48 | right | union@5 | 14.0 | 0.4740 | 0.4998 |
| calibration-48 | down | union@5 | 14.1 | 0.5016 | 0.5277 |
| holdout-48 | right | union@5 | 14.0 | 0.4738 | 0.5010 |
| holdout-48 | down | union@5 | 14.0 | 0.5170 | 0.5424 |
| calibration-48 | right | union@32 | 78.2 | 0.7691 | 0.7964 |
| calibration-48 | down | union@32 | 78.8 | 0.7866 | 0.8136 |
| holdout-48 | right | union@32 | 78.1 | 0.7719 | 0.7970 |
| holdout-48 | down | union@32 | 78.4 | 0.7931 | 0.8182 |

Результат переносится calibration→holdout, но имеет четыре строгих ограничения:

1. true/content labels восстановлены через clean-target Hungarian assignment;
2. `trusted` — принудительно лёгкая половина target-selected позиций, не
   inference-visible confidence;
3. `union@32` означает до 32 кандидатов **от каждого** emitter-а и фактически
   около 78–79, поэтому fixed-budget win не доказан;
4. локальный edge recall не обеспечивает глобальную биекцию, layout или SSIM.

На всех label-uncertain inferred rows holdout union@32 даёт right/down exact
`0.5760/0.5948` и content≤20 `0.8529/0.8635`; эти числа оставлены только как
companion view и не заменяют строгую таблицу. Здесь проверены лишь четыре
analytic views, а не полный historical V28/P29/learned roster.

## 4. Verifier: какой вариант закрыт

Scale-up был заранее зафиксирован: 128 manifest-train boards, 24 свежие
calibration boards с offset 12, пять эпох, union-top-5, общий learned 5×5 spatial
position encoding. На 26 496 `all` rows и 9 613 target-assisted `trusted` rows:

| Scope | Classical ensemble exact / content≤20 | Bilateral content≤20 | Verifier exact / content≤20 | Дельта exact | Дельта content≤20 |
|---|---:|---:|---:|---:|---:|
| all | 0.084050 / 0.202634 | **0.216750** | **0.094845** / 0.168856 | **+1.079 pp** vs ensemble | **−3.378 pp** vs ensemble; **−4.789 pp** vs bilateral |
| trusted strict | 0.166857 / 0.170290 | 0.152086 | **0.199313 / 0.202954** | **+3.246 pp** vs ensemble | **+3.266 pp** vs ensemble; **+5.087 pp** vs bilateral |

Strict re-eval требует mapping confidence и для выбранного content candidate.
После фильтрации content≤20 почти совпадает с exact, потому что low-margin twins
исключены; положительная trusted дельта поэтому **не доказывает использование
content slack**. Кроме того, `trusted` недоступен на test и не перекрывает
регрессию на inference-relevant `all`. На `all` отдельно по направлениям exact
растёт на `+1.155/+1.004 pp` (right/down) против ensemble, но content≤20 падает
на `−3.729/−5.850 pp` против bilateral. Заранее заданный calibration gate
провален в обоих направлениях.

В соответствии с протоколом новый scale-up holdout **не открывался**, decoder и
full-image SSIM run не запускались. Ранее открытые pilot holdout-12 считаются
touched и не используются как confirmatory panel масштаба.

Вердикт: глобальную content-multipositive формулировку этого verifier-а закрыть
как `reject-as-tested`. Положительный exact signal можно хранить только как
research-only auxiliary; он не является разрешением подмешивать модель в solver
или submission.

## Что делать и чего не повторять

Можно переносить дальше:

- colored NLM **`h=20 x1`** как manual-safe output tail; `h=15 x1` fallback;
- one-to-one content-equivalent metrics рядом с exact-index diagnostics;
- analytic union pool как воспроизводимый supply/control;
- exact verifier checkpoint только для исследования нового evidence, не production.

Без нового механизма не повторять:

- gray guard и per-cell raw/NLM router ради contest SSIM;
- M420 без bijection или с clean score, представленным как deployable result;
- очередной fixed-budget claim для `union@k`, не учитывающий фактический union size;
- ту же global content-multipositive verifier loss/architecture на большем числе
  boards;
- текущий DualNAF checkpoint tile-wise перед repeated NLM 5/10/20: без NLM он
  полезен, но в сильной композиции устойчиво проигрывает;
- тот же DualNAF checkpoint только для E14 matcher scores: raw view дал шумовой
  `+0.001099` SSIM при ухудшении adjacency, а preregistered 50/50 fusion ухудшил
  SSIM; calibration confirmation не запускать;
- pure/global Hungarian по generic train-population atlas и усиление того же
  unary внутри buddies96: pure arm потерял `0.009819` SSIM и почти всю adjacency,
  а веса `0.25/1.0` дали лишь шум менее `0.001` без ручного geometry gain;
- decoder/QAP/LNS только потому, что candidate oracle coverage высок.

## Воспроизведение и артефакты

Подробные команды, параметры и пути к JSON находятся в отчётах:

- [postassembly-harmonizer.md](postassembly-harmonizer.md);
- [nlm-strength-manual-safety.md](nlm-strength-manual-safety.md);
- [tilewise-dualnaf.md](tilewise-dualnaf.md);
- [dualnaf-matcher.md](dualnaf-matcher.md);
- [global-population-layout.md](global-population-layout.md);
- [drunet-goal-cycle2-and-sigma50-broad.md](drunet-goal-cycle2-and-sigma50-broad.md);
- [foundation-semantic-component-stop.md](foundation-semantic-component-stop.md);
- [pixel-tail-bakeoff.md](pixel-tail-bakeoff.md);
- [content-substitution.md](content-substitution.md);
- [candidate-supply.md](candidate-supply.md);
- [content-verifier.md](content-verifier.md).

Основные machine-readable результаты лежат в `outputs/pixel-tail-bakeoff/`,
`outputs/content-substitution/`, `outputs/candidate-supply/`,
`outputs/content-verifier/`, `outputs/postassembly-harmonizer/` и
`outputs/tilewise-dualnaf/`, `outputs/dualnaf-matcher/`,
`outputs/global-population-layout/`. Они намеренно
не коммитятся, но содержат provenance,
code/checkpoint hashes и per-board metrics.

Для scale-up verifier-а authoritative artifact —
`outputs/content-verifier/scale128-calibration24-final.json`. Предшествующий
`scale128-calibration24.json` сохранил старую `trusted_query` семантику, где
confidence выбранного content candidate не фильтровался; его trusted-content
headline использовать нельзя.

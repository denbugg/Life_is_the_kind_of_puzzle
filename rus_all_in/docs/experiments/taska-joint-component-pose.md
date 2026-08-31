# TASKA dense-contact joint component pose Transformer

Статус: **bounded gate fail-stop**. Capacity и conditional shift R@5 показали
signal, но source-disjoint R@1 не прошёл gate, а четырёхкомпонентный pack
разрушил adjacency. Долгий/server run заблокирован; default confirmed six-arm
fusion, production и submission не менялись.

## No-repeat audit и pivot до обучения

Generic component-set/graph Transformer с direct absolute offset не запускался:
это был бы дубль `component_absolute_placer`, где native component pixels,
board-set attention, purity и joint feasible absolute offsets уже дали offset
top1 `0.2119%`, около chance. Также не повторялись whole-layout/global-roll CNN,
DINO/population probes, factorized component-shift MLP, tile-slot
SocketPermutationFlow и independent component-relation reranker.

Первый proposed roster из оставшихся selected-supply cross-component edges был
закрыт до model fit. На fit он покрывал лишь `0.360%` nontrivial components,
`0.627%` dominant exact support и `0.531%` pure-component tiles; на local —
`0.713/2.604/1.279%`. Correct shift почти никогда не был доступен модели.

Единственный активированный новый evidence source — dense TASKA top-8 contacts
для каждого member tile и четырёх направлений. Каждый contact между разными
realised six-arm components однозначно задаёт feasible integer translation.
До cap он покрывал:

| Candidate-space diagnostic | Fit | Local |
|---|---:|---:|
| nontrivial components | `19.568%` | `22.979%` |
| dominant exact support | `48.306%` | `53.164%` |
| pure-component tiles | `24.014%` | `24.928%` |

Это materially отличается от unique-fullres translation-consensus: emitter
здесь — полный dense raw TASKA top-8 roster, не редкий accepted unique-fullres
suffix. Raw seams создают кандидатов/features, но **не** служат post-hoc veto.

## Frozen architecture и protocol

Все реальные components, включая singletons, получают dirty-visible pooled
colour/boundary/gradient descriptors и current geometry. Directed component
pairs получают aggregate contact rank, normalized cost, directions и implied
translations. Два блока выполняют sparse incoming/outgoing edge messages и
global self-attention по полному board component set. Candidate head выдаёт
residual над matched raw ordering для максимум 128 contact-implied shifts на
каждую nontrivial component; отдельные heads оценивают roster coverage и
dominant-shift purity.

Модель имеет `212,483` параметра (`width64`, 2 layers, 4 heads). Training
использовал component-order permutation, standardized feature jitter `0.03`,
15% pair-feature и 10% candidate-feature dropout, а также два независимых
corruption draws на source. Fit — formal organizer-train source16×draw2;
local — disjoint held organizer-train source16×draw2. Fresh/competition test не
открывались.

Decoder target-blind выбирал максимум четыре anchors по fixed
coverage×purity×candidate confidence. Anchors размещались совместно; остальные
components сначала сохранялись на baseline, затем collision components
repack-ились по minimum displacement, а deferred tiles назначались Hungarian.
Raw-seam guard отсутствовал; каждый output — strict permutation 576 original
upright fragments.

Preregistration записан до cache/model initialization/training:
`configs/taska_joint_component_pose_pilot_v1.json`, SHA-256
`3b22094a1e164f39cd90cdebbb17a591ca1f525603a4fa446e7b00d07bf46a1b`.

## Capacity и bounded result

One-board capacity за 80 MPS steps снизила loss `6.158→0.241`; conditional
R@1/R@5 достигли `90.91/100%`, capacity gate прошёл. Full pilot 240 steps занял
`24.06 s`; cache — `24.39 s`, capacity — `8.33 s`.

После fixed cap128 candidate-space coverage стала `27.175%` dominant support
на fit и `33.092%` на local. Conditional retrieval:

| Local shift retrieval | Raw roster | Learned | Delta |
|---|---:|---:|---:|
| support-weighted R@1 | `0.477%` | `0.755%` | `+0.278 pp` |
| support-weighted R@5 | `2.702%` | `9.337%` | `+6.635 pp` |
| unweighted R@1 | `0.467%` | `1.402%` | `+0.935 pp` |
| unweighted R@5 | `2.804%` | `5.607%` | `+2.804 pp` |

R@5 — положительный relational/context signal, но preregistered R@1 gate
требовал `+2 pp` support-weighted и провален.

| Local32 layout metric | Six-arm control | Joint pose | Delta |
|---|---:|---:|---:|
| exact tiles / board | `1.90625` | `2.09375` | `+0.18750` |
| satisfied pairs / board | `345.31250` | `278.84375` | `−66.46875` |
| adjacency recall | `31.2783%` | `25.2576%` | `−6.0207 pp` |

Exact W/T/L `5/23/4`; pairs `0/0/32`. Layout gate требовал exact `>0` и pair
delta `>=−1`; pair condition decisively failed.

## Почему exact delta не является usable signal

Frozen-prediction forensic использовал targets только после scoring freeze.
Selector запросил 97 anchors и разместил в среднем `2.31` на board. Лишь
`1/97` выбранных shifts совпал с dominant exact shift; все anchors напрямую
поддержали всего `12` exact tiles при `2,491` moved component tiles. Pack в
среднем:

- переместил `1375.94` tile-L1;
- repacked `29.22` components;
- deferred/Hungarian-filled `65.94` tiles.

Следовательно, aggregate `+6` exact tiles на 32 boards возникли случайно при
массовом repacking, а не из точного pose выбора. Положительный R@5 говорит, что
board graph помогает coarse shortlist, но top1/packing недостаточны.

## Verdict и no-repeat

Не масштабировать тот же width/layers/steps, не sweep-ить top-k, cap,
thresholds или maximum anchors на открытой панели. Новый серверный run разрешён
только после materially new pair-preserving joint packing objective либо
independent evidence, которое даёт source-disjoint shift R@1 gain `>=2 pp`.
Blocked server contract и готовый cache сохранены в
`configs/taska_joint_component_pose_server_scale_blocked_v1.json` (SHA-256
`78f9d353b66b0b6adceedc5edaccd52cafdd52b8788787d72d51c26d50e4121f`).

Наиболее узкий следующий diagnostic — не pose model: проверить, дают ли
несколько независимых dense top-8 contacts одну и ту же component translation
с высокой precision. Если да, их следует использовать как pair-first supply;
если нет, dense roster остаётся только low-top-k retrieval evidence.

## Артефакты и проверки

- report:
  `outputs/taska-joint-component-pose/pilot-v1/report.json`, SHA-256
  `1f47494059e063b4dccf7d0b7c27b6eb092d905d878321b19c601f69135f83bc`;
- checkpoint SHA-256
  `974c09f1b32d03c271aa3ab91526a214ea754bf2c9215a0503378b28642d9984`;
- frozen predictions SHA-256
  `a6d800005abc043252377b789efa67bb9f619e40e99f9766ae7fc88c2a572c3b`;
- dirty-visible cache SHA-256
  `fcaee42adcc4f07ff00a04ca130194d6f20cda6d61b7f30428eb48054846f0ba`;
- module: `src/aiijc_puzzle/taska_joint_component_pose.py`, SHA-256
  `7bf3f877f797a1bb3f9326e5423c9f3fc12ec844efd19660e819f4dd4e36a512`;
- runner: `scripts/run_taska_joint_component_pose.py`, SHA-256
  `9648b032fa0fb1a9f4fd2e14c0d4089464278235bb9bb24ea4ea280317db93af`;
- `4` focused tests, ruff and pycompile passed; `64` frozen local layouts
  independently checked as strict permutations;
- Weco exact+pair failed step `124`, parent `102`; steps `125–126` not consumed
  by post-hoc pose variants.

Competition test, pixel rendering, production и submission не затрагивались.

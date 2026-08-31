# Полный поисковый реестр M-серии

> Генерируется `scripts/build_prior_research_index.py` из итогового журнала ветки. Интерпретация и поправки к выводам находятся в [`knowledge-base.md`](../knowledge-base.md).

Источник: `origin/autoresearch/pazzle-fixed-orientation-20260813` (`6fb563c4b7`), `autoresearch-runs/pazzle-mgc-restoration-20260818/EXPERIMENTS.md`. В таблице **431** именованных записей; базовые номера M1–M420, пропущено в источнике: **M144**. Варианты `CORRECTION`, `FINAL`, `GATES` и повторные проверки сохранены отдельными строками.

`Вердикт журнала` воспроизводит текст источника в момент записи и сам по себе не является последним словом: строки `IN PROGRESS`, `RUNNING` и ранние `ACCEPTED` нужно читать вместе с одноимёнными `RESULT`/`CORRECTION`/`FINAL` и ручным аудитом.

Искать идею удобнее обычным полнотекстовым поиском, например `rg -ni 'sinkhorn|spectral|restor|chooser' docs/prior-research`.

## M1–M50

| ID | Проверка / идея | Вердикт журнала | Commit | Строка |
|---|---|---|---:|---:|
| `M1` | Seam signal budget: rank the true right-neighbour among all 576 under four input conditions | DIAGNOSTIC / KEY | `e8481ca18f` | 1 |
| `M2` | Degradation sweep: bb_prec versus residual noise and residual brightness error | DIAGNOSTIC | `e8481ca18f` | 2 |
| `M3` | Ridge seam cost: score = var(d) + w*mean(d)^2, sweep w and strip width | KEPT | `e8481ca18f` | 3 |
| `M4` | distort.py fidelity against real (dirty, clean) pairs | PASS | `e8481ca18f` | 4 |
| `M5` | Permutation label accuracy for the restoration dataset | KEPT | `e8481ca18f` | 5 |
| `M6` | Tile restorer trained on pixel L1 | PLATEAU | `e8481ca18f` | 6 |
| `M7` | Tile restorer trained on differentiable ridge seam InfoNCE | KEPT | `e8481ca18f` | 7 |
| `M8` | Brightness calibration from content similarity (no assembly needed) | REJECTED | `e8481ca18f` | 8 |
| `M9` | Torus-origin audit of solve_buddies on ORACLE scores | CRITICAL | `e8481ca18f` | 9 |
| `M10` | solve_buddies robustness on CLEAN tiles | FAIL | `e8481ca18f` | 10 |
| `M11` | Loop verification: keep mutual edges closing a 2x2 cycle | KEPT | `e8481ca18f` | 11 |
| `M12` | solve_loop: loop-verified Kruskal with collision and 24x24 span checks | KEPT | `e8481ca18f` | 12 |
| `M13` | Edge-budget requirement map for the Kruskal builder | DIAGNOSTIC | `e8481ca18f` | 13 |
| `M14` | Achievable edge yield at a precision floor | KEY NEGATIVE | `e8481ca18f` | 14 |
| `M15` | Conflict pruning: drop tiles contradicted by intra-component edges | REJECTED | `e8481ca18f` | 15 |
| `M16` | Paikin-Tal style greedy growth, with score normalisation and best-buddy priority | REJECTED | `e8481ca18f` | 16 |
| `M17` | Mahalanobis Gradient Compatibility (Gallagher CVPR'12) as the seam measure | BREAKTHROUGH | `e8481ca18f` | 17 |
| `M18` | Full chain mgc_cost -> solve_loop -> fix_origin, no oracle | PASS | `e8481ca18f` | 18 |
| `M19` | place_acc -> SSIM payoff map | DIAGNOSTIC / KEY | `e8481ca18f` | 19 |
| `M20` | pair_compat: joint full-pair CNN trained on real pairs, over the restorer | PARTIAL | `e8481ca18f` | 20 |
| `M21` | Context restorer (3x3 tile block + validity mask) | REJECTED | `e8481ca18f` | 21 |
| `M22` | Non-local denoising across the whole shuffled canvas | REJECTED | `e8481ca18f` | 22 |
| `M23` | Proxy-metric audit: residual sigma of every restorer checkpoint | CRITICAL | `e8481ca18f` | 23 |
| `M24` | L1/contrastive balance sweep under the MGC seam loss | REJECTED | `e8481ca18f` | 24 |
| `M25` | FINAL HONEST GATE — actual assembly per restorer checkpoint | FAIL | `e8481ca18f` | 25 |
| `M26` | Information ceiling: candidate recall at depth versus achieved assembly | DIAGNOSTIC / DECISIVE | `e8481ca18f` | 26 |
| `M27` | Structure of the residual error: real restorer versus synthetic Gaussian of similar magnitude | DIAGNOSTIC / KEY | `e8481ca18f` | 27 |
| `M28` | Border-ring oracle: graft the true outer ring onto a restored tile | BREAKTHROUGH | `e8481ca18f` | 28 |
| `M29` | Correct ring target: graft from CLEAN rather than blur3(clean) | KEY | `e8481ca18f` | 29 |
| `M30` | Ring accuracy specification: white noise of controlled sigma on an oracle ring | DIAGNOSTIC | `e8481ca18f` | 30 |
| `M31` | White versus correlated ring error at matched sigma | DECISIVE | `e8481ca18f` | 31 |
| `M32` | High-frequency boosting (unsharp) to whiten the residual | REJECTED | `e8481ca18f` | 32 |
| `M33` | Per-column error profile; JPEG truncated-block asymmetry | DIAGNOSTIC | `e8481ca18f` | 33 |
| `M34` | Matching on an inset strip instead of the tile edge | REJECTED | `e8481ca18f` | 34 |
| `M35` | Whitening the seam difference by the measured error covariance (GLS) | REJECTED | `e8481ca18f` | 35 |
| `M36` | Degradation decomposition: which stage destroys the seam signal | KEY | `e8481ca18f` | 36 |
| `M37` | Iterative re-restoration and noise-TTA | REJECTED | `e8481ca18f` | 37 |
| `M38` | Ring error decomposed into photometry and noise | KEY | `e8481ca18f` | 38 |
| `M39` | Ensembling compatibility matrices (z-normalised) across four restorers, plus the raw ridge view | REJECTED | `e8481ca18f` | 39 |
| `M40` | Noise reduction versus matchability: the two do not agree | KEY NEGATIVE | `e8481ca18f` | 40 |
| `M41` | High-precision edge core: mutual MGC edges ranked by relative margin | KEY | `e8481ca18f` | 41 |
| `M42` | Does SPARSE but CORRECT context help the restorer | REJECTED / CLOSES THE EM LOOP | `e8481ca18f` | 42 |
| `M43` | Gradient checkpointing for the restorer | ENGINEERING | `e8481ca18f` | 43 |
| `M44` | LP global placement (Yu, Russell & Agapito, BMVC 2016) reimplemented as weighted-L1 translation synchronisation | KEPT as the solver | `e8481ca18f` | 44 |
| `M45` | LP outlier tolerance and the role of weights | KEY | `e8481ca18f` | 45 |
| `M46` | Pomeranz (Lp)q sub-additive compatibility with prediction term | REJECTED | `e8481ca18f` | 46 |
| `M47` | Border replaced by linear extrapolation from the cleaner interior (naive Bridger) | REJECTED | `e8481ca18f` | 47 |
| `M48` | Ring information audit: graft observed ring into otherwise CLEAN tiles | KEY | `e8481ca18f` | 48 |
| `M49` | LP activation threshold versus edge precision | KEY | `e8481ca18f` | 49 |
| `M50` | Origin recovery statistics on the toroidal cut | PARTIAL | `e8481ca18f` | 50 |

## M51–M100

| ID | Проверка / идея | Вердикт журнала | Commit | Строка |
|---|---|---|---:|---:|
| `M51` | Hybrid: greedy assembly, origin chosen to agree with the LP layout | KEPT for good scores | `e8481ca18f` | 51 |
| `M52` | Whole-tile appearance statistics as an independent cue, fused with MGC | REJECTED | `e8481ca18f` | 52 |
| `M53` | LP with multiple candidates per edge, re-tested AFTER the axis fix | REJECTED | `e8481ca18f` | 53 |
| `M54` | Scaling the MGC-trained restorer (2.41M params, checkpointed, 20k steps) | PLATEAU | `e8481ca18f` | 54 |
| `M55` | Restoration target A/B under identical settings: clean vs blur3(clean) | NO EFFECT | `e8481ca18f` | 55 |
| `M56` | Seam-inpainting compatibility (Bridger-style, regression core) | PARTIAL / weaker than MGC | `e8481ca18f` | 56 |
| `M57` | Fusing the inpainting score with MGC on a shared shortlist | REJECTED | `e8481ca18f` | 57 |
| `M58` | Colour-space audit: per-channel residual noise after JPEG | KEY | `e8481ca18f` | 58 |
| `M59` | SNR weighting of colour channels inside the measure | REJECTED | `e8481ca18f` | 59 |
| `M60` | Redundancy audit: do similar tiles exist within one board | KEY | `e8481ca18f` | 60 |
| `M61` | Non-local averaging over photometrically normalised twins | REJECTED | `e8481ca18f` | 61 |
| `M62` | Explicit YCrCb input view for the restorer | REJECTED | `e8481ca18f` | 62 |
| `M63` | Hard-negative focusing of the contrastive seam loss (top-32 competitors per row) | REJECTED | `e8481ca18f` | 63 |
| `M64` | Loopy belief propagation over the grid MRF (after Cho et al., CVPR 2010) | REJECTED | `e8481ca18f` | 64 |
| `M65` | Corruption-strength mixing (sample synthetic severity per board) | REJECTED | `e8481ca18f` | 65 |
| `M66` | Cascade: L1 denoiser feeding the MGC-trained matcher | REJECTED | `e8481ca18f` | 66 |
| `M67` | Absolute-position predictability (foundation of positional diffusion) | REJECTED | `e8481ca18f` | 67 |
| `M68` | Label-noise audit of our own metrics | METHODOLOGICAL | `e8481ca18f` | 68 |
| `M69` | SSIM cost of misplacing FLAT tiles versus random ones | KEY | `e8481ca18f` | 69 |
| `M70` | Does coarse absolute position pay at all | REJECTED / CLOSES THE POSITIONAL AXIS | `e8481ca18f` | 70 |
| `M71` | Matching restricted to the textured subset (smaller candidate pool) | PARTIAL | `e8481ca18f` | 71 |
| `M72` | Per-tile photometric z-normalisation before MGC | REJECTED | `e8481ca18f` | 72 |
| `M73` | Attribution of the matching loss to each corruption stage | KEY | `e8481ca18f` | 73 |
| `M74` | Ring-sigma response curve: how clean must the border be | DIAGNOSTIC / CALIBRATION | `e8481ca18f` | 74 |
| `M75` | Audit of the actual corruption, measured instead of assumed | METHODOLOGICAL / KEY | `e8481ca18f` | 75 |
| `M76` | Is the true layout the MINIMUM of the summed seam objective | DECISIVE NEGATIVE / CLOSES THE SOLVER AXIS | `e8481ca18f` | 76 |
| `M77` | Cost-matrix normalisations to remove the per-tile bias | REJECTED | `e8481ca18f` | 77 |
| `M78` | Affine-invariant seam costs (normalised cross-correlation across the seam) | PARTIAL | `e8481ca18f` | 78 |
| `M79` | Learned seam matcher: siamese directional descriptors, InfoNCE over all 576 candidates | BREAKTHROUGH (in progress) | `e8481ca18f` | 79 |
| `M80` | Simulated annealing as a CONSTRUCTOR, tuned on clean boards | REJECTED for construction, KEPT as refiner | `e8481ca18f` | 80 |
| `M81` | Corruption-stage ablation against the LEARNED matcher | KEY | `e8481ca18f` | 81 |
| `M82` | Objective soundness under the LEARNED cost (re-test of M76) | BREAKTHROUGH / UNBLOCKS THE SOLVER AXIS | `e8481ca18f` | 82 |
| `M83` | Twin ambiguity: how much of the ranking loss is impossible | KEY / METHODOLOGICAL | `e8481ca18f` | 83 |
| `M84` | Activation curve of the LEARNED cost: severity sweep end to end | KEY NEGATIVE | `e8481ca18f` | 84 |
| `M85` | Why recall does not convert: the solver-relevant statistics | DIAGNOSTIC / DECISIVE | `e8481ca18f` | 85 |
| `M86` | Sinkhorn-calibrated cost form, and the high-precision edge core | KEY | `e8481ca18f` | 86 |
| `M87` | Restricting the LP to the trustworthy core | KEY NEGATIVE / STRUCTURAL | `e8481ca18f` | 87 |
| `M88` | Activation curve redone on the Sinkhorn-calibrated cost | KEY / SPLITS THE PROBLEM IN TWO | `e8481ca18f` | 88 |
| `M89` | Whole-board assembler: transformer over all 576 tiles, seam costs as attention bias | REJECTED | `e8481ca18f` | 89 |
| `M90` | Soft-permutation relaxation of the layout QAP (src/solve_soft.py) | REJECTED | `e8481ca18f` | 90 |
| `M91` | The restorer under the learned matcher | REJECTED / RETIRES THE RESTORATION LINE | `e8481ca18f` | 91 |
| `M92` | Loop verification (2x2 closure) on the calibrated learned cost | REJECTED | `e8481ca18f` | 92 |
| `M93` | Cycle consistency between the two assignment matrices | KEPT / FREE GAIN | `e8481ca18f` | 93 |
| `M94` | Assembly under cycle-consistent costs: the solvers switch on | BREAKTHROUGH / SETS THE TARGET | `e8481ca18f` | 94 |
| `M95` | Twin-tolerant edge precision: are our errors real | METHODOLOGICAL | `e8481ca18f` | 95 |
| `M96` | Greedy's layout anchored by the LP's origin (M51 revisited on good costs) | KEPT | `e8481ca18f` | 96 |
| `M97` | Slack Sinkhorn: a 24x24 board has 552 right-neighbour edges, not 576 | NO EFFECT | `e8481ca18f` | 97 |
| `M98` | Re-ranker engineering: the fifth WDDM spill, and fp64 in the hot path | ENGINEERING | `e8481ca18f` | 98 |
| `M99` | Cycle-consistent shortlists widen the re-ranker's ceiling | KEY | `e8481ca18f` | 99 |
| `M100` | End-to-end gate, nothing oracular (src/eval_pipeline.py) | BASELINE | `e8481ca18f` | 100 |

## M101–M150

| ID | Проверка / идея | Вердикт журнала | Commit | Строка |
|---|---|---|---:|---:|
| `M101` | Acyclicity of the right-neighbour matrix | KEPT / small free gain | `e8481ca18f` | 101 |
| `M102` | Definitive activation curve on the deployed pipeline | KEY / SETS THE TARGET IN ONE NUMBER | `e8481ca18f` | 102 |
| `M103` | Assembly payoff measured THROUGH the submission's own post-processing | KEY / RESETS THE TARGET | `e8481ca18f` | 103 |
| `M104` | End-to-end gate with post-processing, and what greedy's components really look like | BASELINE / DIAGNOSTIC | `e8481ca18f` | 104 |
| `M105` | Two-stage re-ranker: what the second stage must see, and what it must not be given | KEY / DESIGN | `e8481ca18f` | 105 |
| `M106` | Shortlist coverage against shortlist size | DESIGN | `e8481ca18f` | 106 |
| `M107` | Why two re-rankers failed, and the design that does not | KEY / DESIGN | `e8481ca18f` | 107 |
| `M108` | Origin recovery, three more attempts | REJECTED / CLOSES THE ORIGIN AXIS BY EVIDENCE | `e8481ca18f` | 108 |
| `M109` | Joint head over the retriever trunk: the training curve | IN PROGRESS / PROMISING | `e8481ca18f` | 109 |
| `M110` | Literature wave: where this problem actually sits | RESEARCH / CALIBRATION | `e8481ca18f` | 110 |
| `M111` | Genetic solver, first gate (src/solve_ga.py) | BREAKTHROUGH | `e8481ca18f` | 111 |
| `M112` | Genetic solver activation curve, and the budget anomaly | KEPT above the knee / DOES NOT MOVE IT | `e8481ca18f` | 112 |
| `M113` | Joint head: final plateau | PARTIAL | `e8481ca18f` | 113 |
| `M114` | Positional diffusion, and the initialisation that decides whether it works | IN PROGRESS / ENGINEERING | `e8481ca18f` | 114 |
| `M115` | Positional diffusion at 4000 steps: where its skill is, and why that is not enough | KEY NEGATIVE (interim) | `e8481ca18f` | 115 |
| `M116` | Training the retriever THROUGH the calibration | REJECTED | `e8481ca18f` | 116 |
| `M117` | Iterative discrete assembly (src/iter_assemble.py) | IN PROGRESS | `e8481ca18f` | 117 |
| `M118` | Iterative discrete assembler: the reveal schedule decides whether it learns anything | KEY / DESIGN | `e8481ca18f` | 118 |
| `M119` | Seeded decoding, and a decode bug worth remembering | PARTIAL / NOT YET COMPETITIVE | `e8481ca18f` | 119 |
| `M120` | Iterative discrete assembler: final verdict | REJECTED | `e8481ca18f` | 120 |
| `M121` | What our assembly is actually worth, against the deployed baseline | KEY / SETTLES A CLAIM | `e8481ca18f` | 121 |
| `M122` | Relaxation labelling over tile-to-position beliefs (src/solve_relax.py) | KEPT as a refiner / small gain | `e8481ca18f` | 122 |
| `M123` | Do the two refiners compound at full corruption | NO / WITHIN NOISE | `e8481ca18f` | 123 |
| `M124` | Fusing the learned matcher with MGC in the CALIBRATED space | REJECTED | `e8481ca18f` | 124 |
| `M125` | Twin-tolerant targets, as first implemented, destroy the model | REJECTED then FIXED | `e8481ca18f` | 125 |
| `M126` | Per-tile restoration for the OUTPUT image rather than for matching | REJECTED / MECHANISM CLEAR | `e8481ca18f` | 126 |
| `M127` | Continuation-prediction auxiliary loss on the retriever | REJECTED | `e8481ca18f` | 127 |
| `M128` | Severity-swept training: generality gained, target regime unmoved | KEY NEGATIVE | `e8481ca18f` | 128 |
| `M129` | The matcher is partly matching on brightness, which carries no information | KEY / DIAGNOSIS | `e8481ca18f` | 129 |
| `M130` | Removing absolute brightness from the input | REJECTED / CORRECTS M129 | `e8481ca18f` | 130 |
| `M131` | SuperGlue-style board context for the descriptors | REJECTED | `e8481ca18f` | 131 |
| `M132` | Photometric remedies, and a re-tune of the calibration | SMALL GAINS | `e8481ca18f` | 132 |
| `M133` | Layout cache for training on our own input distribution | INFRASTRUCTURE | `e8481ca18f` | 133 |
| `M134` | Blur ladder: how much of the post-processing is just smoothing | KEY / REFRAMES THE PROBLEM | `e8481ca18f` | 134 |
| `M135` | What a correct 24x24 thumbnail is worth | KEY | `e8481ca18f` | 135 |
| `M136` | Tolerance of the thumbnail objective to layout error | SUPERSEDED BY M137 | `e8481ca18f` | 136 |
| `M137` | The honest scoreboard: gain over the flat fill | KEY / METHODOLOGY | `e8481ca18f` | 137 |
| `M138` | Resolution ceiling: what coarse structure alone is worth | KEY / OPENS A NEW FRONT | `e8481ca18f` | 138 |
| `M139` | Ridge regression from the palette to the coarse field | PARTIAL / LOWER BOUND | `e8481ca18f` | 139 |
| `M140` | Would a coarse colour field help ASSEMBLY? | KEY NEGATIVE / CLOSES A DOOR | `e8481ca18f` | 140 |
| `M141` | Tile identity under two independent corruptions | KEY / DIAGNOSIS / REFRAMES THE NOISE QUESTION | `e8481ca18f` | 141 |
| `M142` | Set model from the bag of tiles to the coarse colour field | PARTIAL / THE CONTROL IS THE RESULT | `e8481ca18f` | 142 |
| `M143` | Invariance loss trained properly, three arms | REJECTED / CLOSES THE REPRESENTATION FRONT | `e8481ca18f` | 143 |
| `M145` | The exchange rate between visible detail and score | KEY / DECIDES WHAT AN HONEST SUBMISSION CAN LOOK LIKE | `e8481ca18f` | 144 |
| `M146` | The line the user drew, and the test that enforces it | METHODOLOGY | `e8481ca18f` | 145 |
| `M147` | Restorer trained on our own assemblies, and the dead clamp | PARTIAL / TRAP | `e8481ca18f` | 146 |
| `M148` | What a PARTIAL answer is worth | KEY / OPENS THE LIVE ROUTE | `e8481ca18f` | 147 |
| `M149` | Placing an ISLAND rather than a tile | KEY | `e8481ca18f` | 148 |
| `M150` | Matching islands instead of tiles | KEY / MOVES THE WALL | `e8481ca18f` | 149 |

## M151–M200

| ID | Проверка / идея | Вердикт журнала | Commit | Строка |
|---|---|---|---:|---:|
| `M151` | Harvesting correct 2x2 blocks from real costs | KEY / REACHABILITY | `e8481ca18f` | 150 |
| `M152` | Field-anchored restorer | REJECTED | `e8481ca18f` | 151 |
| `M153` | Agglomerative island growth on real boards | PARTIAL / GROWTH HURTS AS BUILT | `e8481ca18f` | 152 |
| `M154` | The partial answer built for real | KEY / THE PROBLEM IS NOW ONE SENTENCE | `e8481ca18f` | 153 |
| `M155` | Joint placement of the trusted blocks | REJECTED / TWO SEPARATE REASONS | `e8481ca18f` | 154 |
| `M156` | Where the island route actually stands | SUMMARY | `e8481ca18f` | 155 |
| `M157` | The joint scorer as a source of costs | REJECTED / AND A CALIBRATION TRAP | `e8481ca18f` | 156 |
| `M158` | Agreement between two matchers as a confidence signal | KEY / BEST EDGE SET SO FAR | `e8481ca18f` | 157 |
| `M159` | Building islands from the trusted edge set directly | PARTIAL | `e8481ca18f` | 158 |
| `M160` | Re-calibrating the costs on the tiles that are LEFT | REJECTED | `e8481ca18f` | 159 |
| `M161` | Every remaining route to ABSOLUTE island placement | KEY NEGATIVE / CLOSES THE ROUTE | `e8481ca18f` | 160 |
| `M162` | A larger, higher-resolution coarse field | REJECTED | `e8481ca18f` | 161 |
| `M163` | Consensus across the whole model roster | KEY NEGATIVE / AN INFORMATION CEILING | `e8481ca18f` | 162 |
| `M164` | The joint scorer with an UNFROZEN trunk | REJECTED / LAST LEVER SPENT | `e8481ca18f` | 163 |
| `M165` | Pasting an island's TEXTURE without its colour | KEY / GIVES A DIAL | `e8481ca18f` | 164 |
| `M166` | Is the 0.155 ceiling the edges or the BUILDER? | KEY / RENAMES THE TARGET | `e8481ca18f` | 165 |
| `M167` | Costs from RESTORED tiles, as a second view | KEY / BEST EDGE PRECISION IN THE PROJECT | `e8481ca18f` | 166 |
| `M168` | Voting across input representations | PARTIAL | `e8481ca18f` | 167 |
| `M169` | Union of several high-precision detectors | REJECTED / THE DETECTORS OVERLAP | `e8481ca18f` | 168 |
| `M170` | Three-view matcher: raw, normalised and RESTORED | REJECTED AT EQUAL BUDGET | `e8481ca18f` | 169 |
| `M171` | Our edges through the earlier team's component packer | KEY / FIRST MOVEMENT IN place_acc | `e8481ca18f` | 170 |
| `M172` | Literature sweep, August 2026 | REFERENCE | `e8481ca18f` | 171 |
| `M173` | Filling the untrusted cells by colour continuity | PARTIAL | `e8481ca18f` | 172 |
| `M174` | Restoration erases the layout | KEY / EXPLAINS EVERY SOLVER RESULT IN THIS PROJECT | `e8481ca18f` | 173 |
| `M175` | Training the restorer under a DETAIL FLOOR | REJECTED / THE BLEND IS THE RIGHT MECHANISM | `e8481ca18f` | 174 |
| `M176` | Restoring hard inside components and gently outside | ACCEPTED, SMALL / AND IT DATES THE COMPONENTS | `e8481ca18f` | 176 |
| `M177` | Placing all 576 fragments to match a colour field, by Hungarian assignment | MECHANISM CONFIRMED, FIELD IS THE BOTTLENECK | `e8481ca18f` | 178 |
| `M178` | The number that reframes the project | MEASUREMENT | `e8481ca18f` | 180 |
| `M179` | Levelling the per-fragment brightness from the seams, and shrinking the field | LEVELLING ACCEPTED / SHRINKAGE REJECTED | `e8481ca18f` | 182 |
| `M180` | Does the object-size effect hold for the components we actually build | REJECTED / THE BOOTSTRAP IS CLOSED | `e8481ca18f` | 184 |
| `M181` | The operating point, on 24 boards, with everything switched on | THE FIRST CONFORMANT ARM ABOVE THE FLAT FILL | `e8481ca18f` | 186 |
| `M182` | At which spatial scale should the surviving texture live | SIGMA 20 CONFIRMED, HYPOTHESIS REJECTED | `e8481ca18f` | 188 |
| `M183` | Which texture source buys the most score per unit of visible detail | EDGE-PRESERVING PASS ACCEPTED | `e8481ca18f` | 190 |
| `M184` | What the seam levelling is worth where it can be checked | STRONG, AND IT SCALES WITH THE ASSEMBLY | `e8481ca18f` | 192 |
| `M185` | Solving for per-fragment GAIN as well as offset | CORRECT ON A CORRECT LAYOUT, DEGENERATE ON OURS -- DO NOT SHIP | `e8481ca18f` | 194 |
| `M186` | Fusing the raw and restored views as CALIBRATED PROBABILITIES rather than as a filter | REJECTED | `e8481ca18f` | 196 |
| `M187` | Recall at depth for the CURRENT matcher: the number M26 never took | KEY / REOPENS THE SOLVER AXIS | `e8481ca18f` | 198 |
| `M188` | The two switched-off knobs in joint_cost.py | REJECTED / M164's VERDICT CONFIRMED TWICE | `e8481ca18f` | 200 |
| `M189` | A 2x2 verifier on pixels, trained BINARY | REJECTED / THE OBJECTIVE WAS THE CEILING | `e8481ca18f` | 202 |
| `M190` | Four matcher checkpoints exist and the deployed one is the weakest | SMALL FREE GAIN / METHODOLOGICAL | `e8481ca18f` | 204 |
| `M191` | Listwise training, and a metric that scored its own trap | ENGINEERING / METHODOLOGICAL | `e8481ca18f` | 206 |
| `M192` | The residual gate that was not a gate | ENGINEERING / TRAP 1 AGAIN | `e8481ca18f` | 208 |
| `M193` | Does the four-corner junction carry anything the four seams do not | REJECTED / CLOSES THE 2x2 ROUTE | `e8481ca18f` | 210 |
| `M194` | Standardising a score destroys the ranking BETWEEN anchors | METHODOLOGICAL | `e8481ca18f` | 212 |
| `M195` | If re-ranking were PERFECT, would the solvers switch on | KEY / VALIDATES THE FRAMING AND EXPOSES A SECOND GAP | `e8481ca18f` | 214 |
| `M196` | Choosing the whole board's origin by the colour field | REJECTED / CLOSES ORIGIN AT BOARD SCALE, AND CORRECTS M195 | `e8481ca18f` | 216 |
| `M197` | Finishing the matcher's own recipe | SMALL GAIN / CLOSES THE SCALING LEVER | `e8481ca18f` | 218 |
| `M198` | The packer's default budget throws away a factor of 58 in the regime we are aiming at | KEY / FREE, AND ONE LANDMINE | `e8481ca18f` | 220 |
| `M199` | What perfect photometry is worth to the LEARNED matcher | REJECTED / CLOSES THE PHOTOMETRIC LINE AND CORRECTS A CARRIED BELIEF | `e8481ca18f` | 222 |
| `M200` | The local head: one descriptor per position along the seam | NO DIFFERENCE / CLOSES THE POSITIONAL HYPOTHESIS | `e8481ca18f` | 224 |

## M201–M250

| ID | Проверка / идея | Вердикт журнала | Commit | Строка |
|---|---|---|---:|---:|
| `M201` | Pessimistic fusion of two matchers of EQUAL strength | ACCEPTED / THE FIRST FUSION HERE THAT DOES NOT LOSE | `e8481ca18f` | 226 |
| `M202` | Total variation as a global objective: ranks layouts, cannot find the origin | KEY / HALF ACCEPTED | `e8481ca18f` | 228 |
| `M203` | Choosing among a bag of layouts, by total variation or by anything else | REJECTED / THE BAG IS EMPTY | `e8481ca18f` | 230 |
| `M204` | What a cell's second, third and fourth neighbour are worth | KEY / QUANTIFIES THE BOOTSTRAP | `e8481ca18f` | 232 |
| `M205` | Most-constrained-first growth, three seedings | REJECTED / AND IT NAMES THE REAL BOTTLENECK | `e8481ca18f` | 234 |
| `M206` | Telling a correct component from a wrong one, without the truth | ACCEPTED / SELF-CONSISTENCY PICKS THE CORE | `e8481ca18f` | 236 |
| `M207` | Gated growth from a self-consistency-verified core | REJECTED / THE OBSTRUCTION IS GEOMETRIC | `e8481ca18f` | 238 |
| `M208` | Does pessimistic fusion accumulate with a third equal member | NO / IT SATURATES AT TWO | `e8481ca18f` | 240 |
| `M209` | Restoration for matching, tested in its OWN domain for the first time | NEUTRAL, NOT HARMFUL / CORRECTS M66 AND M91 | `e8481ca18f` | 242 |
| `M210` | Is there a cut-off at which the field only corrects colour instead of replacing it | NO / THE CHOICE IS BINARY | `e8481ca18f` | 244 |
| `M211` | Filtering components by self-consistency before packing | NO EFFECT / THE HARVEST IS TOO FINE-GRAINED | `e8481ca18f` | 246 |
| `M212` | Agreement across BOTH axes of diversity | KEY / FIRST BREAK OF THE 0.155 CAP | `e8481ca18f` | 248 |
| `M213` | Does the two-axis harvest reach the layout | PARTIALLY / BEST ADJACENCY THE PROJECT HAS HAD | `e8481ca18f` | 250 |
| `M214` | Three architectures times three inputs: the k-of-9 curve | KEY / THE CAP WAS AN ARTEFACT OF ONE AXIS | `e8481ca18f` | 252 |
| `M215` | Which point on the k-of-9 curve the LAYOUT wants | ACCEPTED / BEST ADJACENCY THE PROJECT HAS HAD | `e8481ca18f` | 254 |
| `M216` | The voted harvest end to end, with adjacency finally instrumented | THE TRADE, MEASURED | `e8481ca18f` | 256 |
| `M217` | Absolute placement re-tested on the better harvest | STILL CLOSED / AND IT NAMES WHAT WOULD OPEN IT | `e8481ca18f` | 258 |
| `M218` | Widening the input axis to five | NO / IT IS EXHAUSTED | `e8481ca18f` | 260 |
| `M219` | A third axis, free: score every seam with the OTHER head | ACCEPTED / BEST ADJACENCY THE PROJECT HAS RECORDED | `e8481ca18f` | 262 |
| `M220` | Merging verified components, ranked by contact self-consistency | REJECTED | `e8481ca18f` | 264 |
| `M221` | Component size is BOND PERCOLATION, and that unifies the activation threshold | KEY / THE MECHANISM BEHIND THE KNEE | `e8481ca18f` | 266 |
| `M222` | Concentrating the edge budget instead of spreading it | REJECTED | `e8481ca18f` | 268 |
| `M223` | More CORRECT edges does not mean bigger components -- a correction to M221 | KEY / REFINES THE TARGET | `e8481ca18f` | 270 |
| `M224` | Votes and margin swept together | KEY / PRECISION 1.000 IS REACHABLE, AND STILL TOO FEW | `e8481ca18f` | 272 |
| `M225` | The harvest settings, validated on 24 boards through the shipping path | THE ASSEMBLY-VERSUS-SCORE TABLE | `e8481ca18f` | 274 |
| `M226` | How the leftover cells are filled is worth NOTHING | KEY / CLOSES A WHOLE CLASS OF WORK | `e8481ca18f` | 276 |
| `M227` | The origin by the LEARNED seam cost | REJECTED / SIXTH CLOSURE | `e8481ca18f` | 278 |
| `M228` | The image BORDER is detectable, and it is the first absolute signal | KEY / NEW THREAD | `e8481ca18f` | 280 |
| `M229` | The border signal with proper features, and what it really is | PARTIALLY CLOSED / THE SIGNAL IS MOSTLY CONTENT | `e8481ca18f` | 282 |
| `M230` | The origin, priced and then attacked from the tiles that carry signal | KEY PRICE / SEVENTH CLOSURE | `e8481ca18f` | 284 |
| `M231` | Placement, split into its two halves and each priced | KEY / THE LARGEST PRIZE IN THE PROJECT | `e8481ca18f` | 286 |
| `M232` | The packer is paid to keep components APART | BUG CONFIRMED / FIX DOES NOT PAY | `e8481ca18f` | 288 |
| `M233` | Merge precision against contact length, on components ten times larger | REJECTED / M150's PROMISE IS EMPTY HERE | `e8481ca18f` | 290 |
| `M234` | Positional signal grows strongly with PATCH size -- M67 measured the wrong object | KEY / FIRST LIVE ROUTE TO ABSOLUTE PLACEMENT | `e8481ca18f` | 292 |
| `M235` | The patch predictor on the components we actually have | WORKS IN PROPORTION TO SIZE, AND OUR PIECES ARE TOO SMALL | `e8481ca18f` | 294 |
| `M236` | Eight orientations of the board, not two | ACCEPTED / BEST HARVEST TO DATE | `e8481ca18f` | 296 |
| `M237` | Which diversity axis to spend a fixed budget on | KEY / 24 SCORERS BEAT 72, AND THE BEST 24 DEPEND ON THE GOAL | `e8481ca18f` | 298 |
| `M238` | The 24-scorer records, re-measured on 24 boards | CORRECTION TO M237 / BOARD VARIANCE EXCEEDS THE EFFECT | `e8481ca18f` | 300 |
| `M239` | Rank aggregation instead of vote intersection | REJECTED / THE INFORMATION IS IN UNANIMITY | `e8481ca18f` | 302 |
| `M240` | Three harvest configurations of equal cost, through the shipping path | ACCEPTED / A STRICT IMPROVEMENT, WHICH IS RARE HERE | `e8481ca18f` | 304 |
| `M241` | CORRECTION to M232: the objective is honest at the component level | KEY / REOPENS PLACEMENT | `e8481ca18f` | 306 |
| `M242` | Coordinate descent over component positions | FIRST PLACEMENT GAIN FROM A SOLVER / STILL NOT ENOUGH | `e8481ca18f` | 308 |
| `M243` | Is the true arrangement the OPTIMUM of the placement objective | KEY / MOSTLY YES, AND THE PACKER NEVER LOOKED | `e8481ca18f` | 310 |
| `M244` | Pair moves in the placement search | MARGINAL / FIRST TO BEAT THE PACKER ON BOTH AXES | `e8481ca18f` | 312 |
| `M245` | Annealing the placement objective, and why a better search cannot help | CLOSES THE SEARCH ROUTE / THE OBJECTIVE IS FLAT AT THE TOP | `e8481ca18f` | 314 |
| `M246` | The border IS structural, and it was in a row of a matrix M97 threw away | KEY / M229 OVERTURNED / EIGHTH THREAD OPENED | `e8481ca18f` | 316 |
| `M247` | The border as a PRIOR in the placement search, on 24 boards | ACCEPTED / FIRST ABSOLUTE SIGNAL THAT CONVERTS | `e8481ca18f` | 318 |
| `M248` | Merging islands on a CLOSED LOOP, with confidence and a geometric veto | ACCEPTED / M180 AND M233 OVERTURNED / PRECISION 0.938 | `e8481ca18f` | 320 |
| `M249` | The 72-scorer pool read as a SHARE, and the square as a selector | DIAGNOSTIC / THE VOTE LADDER IS THE FINDING | `e8481ca18f` | 322 |
| `M250` | The layered peel, with the layers above removed by ORACLE | MECHANISM VALID / BASE CASE UNREACHABLE | `e8481ca18f` | 324 |

## M251–M300

| ID | Проверка / идея | Вердикт журнала | Commit | Строка |
|---|---|---|---:|---:|
| `M251` | Growing islands by loop attachment: size against cleanliness | REJECTED / M223 IS THE BINDING CONSTRAINT | `e8481ca18f` | 326 |
| `M252` | Aggregating votes before assigning, and the real shape of the wall | CLOSES GLOBAL ASSIGNMENT / RESTATES THE BOTTLENECK AS RECALL | `e8481ca18f` | 328 |
| `M253` | The recall ceiling, and the fact that TOP-1 is a choice rather than a limit | KEY / THE WALL IS SELECTION AFTER ALL / RESHAPES THE PROJECT | `e8481ca18f` | 330 |
| `M254` | Every hand-written selector on the depth-two pool | REJECTED / THE SUPPLY IS THERE AND NOTHING EXTRACTS IT | `e8481ca18f` | 332 |
| `M255` | A LEARNED edge selector, and what it says the evidence actually is | ACCEPTED / FIRST RULE TO BEAT THE SHIPPING HARVEST ON ITS OWN TERMS | `e8481ca18f` | 334 |
| `M256` | The content border detector, and a better detector that makes a worse board | REJECTED / SECOND INSTANCE OF THE SAME LESSON | `e8481ca18f` | 336 |
| `M257` | Assembly measured as assembly, and the weight SSIM chose wrong | KEY / INSTRUMENTATION / THE TARGET WAS WRONG | `e8481ca18f` | 338 |
| `M258` | Paying the objective for corroboration | MEASURED THEN WITHDRAWN / DID NOT REPLICATE | `e8481ca18f` | 340 |
| `M259` | Staged assembly with the layer defined by CONFIDENCE, not geometry | REJECTED / THE CONFIDENCE SIGNAL DOES NOT EXIST | `e8481ca18f` | 342 |
| `M260` | Two-round selection, so the second round can see the first round's components | REJECTED / THE GEOMETRIC FEATURES DO NOT TRANSFER | `e8481ca18f` | 344 |
| `M261` | Is the TEST corruption harsher than the training corruption | REJECTED / THREE INDEPENDENT CHECKS / distort.py ALSO VALIDATED | `e8481ca18f` | 346 |
| `M262` | The --calibrate training path destroys the model | BUG CONFIRMED / FEATURE UNUSABLE AS WRITTEN | `e8481ca18f` | 348 |
| `M263` | The placement error is SCATTER, and the objective is destroyed by its own context | KEY / M245 CORRECTED / THE CENTRAL DIAGNOSIS | `e8481ca18f` | 350 |
| `M264` | A degenerate tie in the Hungarian fill handed over the answer | BUG CAUGHT / RESULTS RETRACTED | `e8481ca18f` | 352 |
| `M265` | How fast growth from a seed decays, with the seed given by an ORACLE | REJECTED / THE PERCOLATION WALL SEEN FROM THE PLACEMENT SIDE | `e8481ca18f` | 354 |
| `M266` | A beam over the growth | REJECTED / CLOSES THE COORDINATION REMEDIES | `e8481ca18f` | 356 |
| `M267` | Does the cross-encoder widen the CANDIDATE union | REJECTED / A RE-RANKER CANNOT ADD CANDIDATES | `e8481ca18f` | 358 |
| `M268` | What a correct edge is actually worth: the payoff curve by oracle | KEY / THE MOST DECISION-RELEVANT NUMBER IN THE PROJECT | `e8481ca18f` | 360 |
| `M269` | How clean the 552 edges have to be | KEY / THE CLIFF IS TWO-DIMENSIONAL / SETS THE REAL TARGET | `e8481ca18f` | 362 |
| `M270` | The consistency filter, the board bound, and a target that is 100x easier to state | KEY / RESTATES THE TARGET / MECHANISM CONFIRMED | `e8481ca18f` | 364 |
| `M271` | Cycle redundancy as a trust signal for components | REJECTED / IT TRACKS SIZE, AND SIZE TRACKS CONTAMINATION | `e8481ca18f` | 366 |
| `M272` | The fill of the 330 loose cells, measured on PLACEMENT rather than SSIM | ACCEPTED / SMALL AND FREE | `e8481ca18f` | 368 |
| `M273` | "What fits this hole" instead of "what pairs with this", and why it still cannot grow | KEY NEGATIVE / CLOSES THE LAST STRUCTURAL IDEA | `e8481ca18f` | 370 |
| `M274` | How much of the difficulty is the corruption and how much is the picture | KEY / THE TASK IS MODEL-LIMITED, NOT INFORMATION-LIMITED | `e8481ca18f` | 372 |
| `M275` | The severity sweep: a documented fix, implemented, and switched off in every shipped model | BUG / A/B RUNNING | `e8481ca18f` | 374 |
| `M275 RESULT` | The severity sweep, A/B at equal budget | REJECTED ON THE EVIDENCE AVAILABLE | `e8481ca18f` | 392 |
| `M276` | Which stage of the generator does the damage, and why normalising does not undo it | KEY / CORRECTS M274's CEILING | `e8481ca18f` | 376 |
| `M277` | What a perfect per-fragment brightness correction is worth | DIAGNOSTIC / PRICES THE RESTORER'S REAL JOB | `e8481ca18f` | 378 |
| `M278` | The corruption as a 2x2 of oracles: what levelling and denoising each buy, and together | KEY / MAPS THE REMAINING HEADROOM | `e8481ca18f` | 380 |
| `M279` | How much of the oracle ceiling the actual restorers recover | KEY / THE RESTORERS ARE THE UNDER-INVESTED NODE | `e8481ca18f` | 382 |
| `M280` | The restorer payoff curve, and the two landmarks that fall on it | KEY / SETS THE NEXT TARGET / M66 AND M91 OVERTURNED IN PRINCIPLE | `e8481ca18f` | 384 |
| `M281` | Non-local denoising across the fragment collection | REJECTED / THE NOISE DEFEATS PATCH MATCHING TOO | `e8481ca18f` | 386 |
| `M282` | What the restorers are actually trained on, and three independent estimates of the same ceiling | CORRECTION / CALIBRATION | `e8481ca18f` | 388 |
| `M283` | The restorer's `residual` flag is set at training and dropped at inference | BUG FIXED / SMALL BUT FREE | `e8481ca18f` | 390 |
| `M284` | Fixing the residual bug makes the pipeline WORSE, and the reason is diversity | AWKWARD / CORRECTNESS KEPT / FLAGGED | `e8481ca18f` | 394 |
| `M285` | Testing M284's diversity claim instead of leaving it a hypothesis | CONFIRMED, WEAKLY / SUGGESTS A FREE FIX | `e8481ca18f` | 396 |
| `M286` | Process note: three of today's measurements were re-derivations | PROCESS | `e8481ca18f` | 398 |
| `M287` | honest_v4 on the platform, and the local-to-platform rule breaking | CALIBRATION / A STANDING RULE RETIRED | `e8481ca18f` | 400 |
| `M288` | The clean-but-small harvest, re-tested on the current placement stack | M249 CONFIRMED / MY HYPOTHESIS REFUTED | `e8481ca18f` | 402 |
| `M289` | Sweeping the vote threshold for the quantity that actually converts | CONFIRMS THE OPERATING POINT / VALIDATES THE PROXY | `e8481ca18f` | 404 |
| `M290` | Process, second time today: four of this session's headline entries were re-derivations | PROCESS / AND A SHARPER TARGET | `e8481ca18f` | 406 |
| `M291` | A systematic scan of the rejected verdicts, and what it cancelled | PROCESS / SAVED 13 GPU-HOURS / TWO CLAIMS WITHDRAWN | `e8481ca18f` | 408 |
| `M292` | --restore-input: the matcher trained AND evaluated on restored fragments | REJECTED / CLOSES THE LAST UNTESTED KNOB / M66 AND M91 UPHELD | `e8481ca18f` | 410 |
| `M293` | Spectral seriation: recovering the layout from the whole similarity matrix at once | NEW MECHANISM / VALIDATED BY ORACLE / WEAK ON REAL DATA | `e8481ca18f` | 412 |
| `M294` | The spectral embedding as a placement prior | REJECTED / THE EMBEDDING, NOT THE ORIENTATION | `e8481ca18f` | 414 |
| `M295` | Spectral shrinkage of the similarity matrix toward the lattice's known spectrum | REJECTED | `e8481ca18f` | 416 |
| `M296` | The descriptor DIMENSION, the one axis of the matcher never varied | A/B RUNNING | `e8481ca18f` | 418 |
| `M296 INTERIM` | Descriptor dimension moves R@20, which nothing else has | RETRACTED, SEE M306 -- THE CONTROL WAS NOT MATCHED | `e8481ca18f` | 424 |
| `M297` | honest_v5 on the platform: assembly is FREE | KEY / SETTLES THE ARM QUESTION / RETIRES LOCAL SSIM AS A PREDICTOR | `e8481ca18f` | 420 |
| `M298` | Literature sweep, August 2026, beyond the puzzle field | REFERENCE / EXPLAINS M293 FROM THEORY | `e8481ca18f` | 422 |
| `M299` | Diffusion distance on the affinity graph, and what the three global attempts have in common | REJECTED / CLOSES THE GLOBAL-AVERAGING FAMILY | `e8481ca18f` | 426 |
| `M300` | Infrastructure: the cost matrices are cached | ENGINEERING | `e8481ca18f` | 428 |

## M301–M350

| ID | Проверка / идея | Вердикт журнала | Commit | Строка |
|---|---|---|---:|---:|
| `M301` | The information-theoretic ceiling on denoising one fragment | KEY / CLOSES RESTORATION BY A BOUND RATHER THAN A PLATEAU / REVERSES THE PRIORITY | `e8481ca18f` | 430 |
| `M302` | How much of the per-fragment affine is recoverable at all | KEY / CLOSES THE RESTORATION AXIS BY A SECOND BOUND | `e8481ca18f` | 432 |
| `M303` | The position stated completely, from three independent bounds | WRONG, SEE M304 -- ITS PREMISE IS AN ARTEFACT | `e8481ca18f` | 434 |
| `M304` | The current stack on PERFECT input, and why M141's bound was an artefact | KEY / OVERTURNS M141 AND M303 / THE TASK IS SOLVABLE | `e8481ca18f` | 436 |
| `M305` | The conversion between matcher accuracy and assembly, measured end to end | KEY / SETS THE MATCHER TARGET FROM OUR OWN DATA | `e8481ca18f` | 438 |
| `M306` | The dimension A/B against a MATCHED control, and the run-to-run noise floor | REJECTED / RETRACTS M296's INTERIM / M197 HOLDS | `e8481ca18f` | 440 |
| `M307` | A scoring form that is not bilinear: the best of several descriptor modes | A/B RUNNING | `e8481ca18f` | 442 |
| `M307 RESULT` | The hard maximum over descriptor modes | REJECTED / THE SCORING FORM IS NOT THE LEVER EITHER | `e8481ca18f` | 444 |
| `M308` | The soft maximum over modes, and the closure of the scoring FORM | REJECTED / CLOSES THE FORM ALONGSIDE CAPACITY, DEPTH AND DIMENSION | `e8481ca18f` | 446 |
| `M309` | Whether an independently initialised retriever is a new voter | REJECTED / CLOSES DIVERSITY, AND EXPLAINS M253's CEILING | `e8481ca18f` | 448 |
| `M310` | Model limit or data limit: one clean board under eight independent corruptions | KEY / OVERTURNS THE CONTENT-AMBIGUITY READING / NAMES THE WALL EXACTLY | `e8481ca18f` | 450 |
| `M311` | A second look at the same seam, from different pixels | MECHANISM CONFIRMED, DOES NOT CONVERT | `e8481ca18f` | 452 |
| `M312` | Robust aggregation ACROSS the rows of a seam | REJECTED / THE SUM IS ALREADY RIGHT | `e8481ca18f` | 454 |
| `M313` | Segment agreement against the quantity that actually decides, the clean PREFIX | REJECTED / CLOSES THE SEGMENT THREAD | `e8481ca18f` | 456 |
| `M314` | A perfect selector on the pool we actually have | KEY / RE-OPENS SELECTION / RETIRES THE 552 TARGET | `ba45d2bb95` | 458 |
| `M315` | The vote threshold swept and judged by PLACEMENT rather than clean coverage | REJECTED / THE METRIC WAS NOT THE PROBLEM | `ba45d2bb95` | 460 |
| `M316` | How clean a selector must be, measured on the real pool | KEY / SETS THE TARGET THAT REPLACES 552 | `ba45d2bb95` | 462 |
| `M317` | The learned selector rebuilt against M316's target, with the independent feature | REJECTED / PER-EDGE SELECTION IS CLOSED | `17ddb696b2` | 464 |
| `M318` | Consistency as a SEARCH over orderings rather than a cleanup pass | REJECTED / AND DIAGNOSES WHY THE FILTER LEAKS | `17ddb696b2` | 466 |
| `M319` | Corroboration required for every merge | REJECTED / CANNOT BOOTSTRAP | `17ddb696b2` | 468 |
| `M320` | Corroboration demanded only where a false edge is expensive | ACCEPTED ON COVERAGE / REJECTED ON PLACEMENT | `17ddb696b2` | 470 |
| `M321` | Every selection arm carried to placement, on one measure | REJECTED / CLOSES SELECTION / THE PAYOFF IS ONE LARGE COMPONENT | `17ddb696b2` | 472 |
| `M322` | Spending the precision budget in ONE PLACE | MECHANISM CONFIRMED / SATURATES AT 38 | `d200044af5` | 474 |
| `M322-CORRECTION` | PRIORITY: M322 is a re-derivation of M222, and it overwrote its script | PROCESS NOTE | `bbe9882493` | 504 |
| `M323` | Growing one block only while the evidence is confident | REJECTED / 38 IS A PROPERTY OF THE EVIDENCE | `d200044af5` | 476 |
| `M324` | Per-fragment NOISE LEVEL as a reliability signal, tested as an oracle first | REJECTED / IT IS THE REALISATION, NOT THE LEVEL | `d200044af5` | 478 |
| `M325` | Filling a hole by UNANIMITY of its neighbours instead of by their summed cost | KEY / THE STEP M273 NEEDED / 0.996 WITH ORACLE CONTEXT | `563783bf27` | 480 |
| `M326` | Why the rule grew nothing from a compact seed | PROCESS NOTE / A PROPERTY OF THE SEED, NOT THE RULE | `563783bf27` | 482 |
| `M327` | The agreement front on ragged seeds, with and without nucleation | REJECTED / PRECISE AND STARVED, THE SHAPE EVERY GOOD RULE HERE TAKES | `563783bf27` | 484 |
| `M328` | Two matchers TRAINED on disjoint halves of the seam | REJECTED / CORRECTS M311 / INDEPENDENCE COMES FROM DIFFERENT SEAMS, NOT DIFFERENT PIXELS | `3867d185f7` | 486 |
| `M329` | The agreement rule on a raster front, and why growth is closed by arithmetic | KEY NEGATIVE / CLOSES SEEDED GROWTH FROM THE MECHANISM RATHER THAN FROM A FAILED RUN | `1156258581` | 488 |
| `M330` | Ordering ONE row, given its membership by oracle | KEY NEGATIVE / PRICES THE PEEL'S BASE CASE PROPERLY / THE COST, NOT THE SEARCH | `be37ccabe6` | 490 |
| `M331` | Does matching the NEXT ring make the true order win | REJECTED / THE OBJECTIVE RANKS WELL AND ITS OPTIMUM IS STILL WRONG / AND A REUSABLE BOUND ON GROUPING | `b60cfafbea` | 492 |
| `M332` | Six aggregation forms for the ordering objective | REJECTED / THE RAW SUM IS THE BEST OF THEM / CLOSES THE FORM | `9e719492a5` | 494 |
| `M333` | The seam quality at which the true row becomes the optimum | KEY / PRICES THE PEEL THE WAY M305 PRICED THE PIPELINE | `9e719492a5` | 496 |
| `M334` | How fast a self-check catches a wrong placement | KEY / THE CHECK IS SHARP AND SILENT | `6bdc666cec` | 498 |
| `M335` | Confirmations accumulated over a window, and why they verify but do not guide | KEY NEGATIVE / PRICES THE BACKTRACKING SEARCH | `6bdc666cec` | 500 |
| `M336` | Weeding components by their own agreement, on components large enough to test | SIGNAL CONFIRMED AND STRENGTHENED / STILL NO CONVERSION / M211's BLOCKER IS GONE AND A NEW ONE REPLACES IT | `b9656101c2` | 502 |
| `M337` | Is the objective WRONG or merely DEGENERATE | KEY DIAGNOSTIC / IT IS WRONG, AND THAT DECIDES THE REMEDY | `20afb832c7` | 506 |
| `M338` | A JOINT scorer over a cell's whole neighbourhood, trained 576-way | REJECTED / THE SUM IS STILL BETTER | `37d8281408` | 508 |
| `M339` | A GLOBAL learned objective, and why none can have its optimum at the truth | KEY NEGATIVE / CLOSES THE OBJECTIVE FAMILY BY A PROPERTY OF THE DATA | `37d8281408` | 510 |
| `M340` | An energy trained against its OWN optimum | ACCEPTED / M339's CONCLUSION RETRACTED / FIRST OBJECTIVE EVER TO PREFER THE TRUTH | `e4e21c48de` | 512 |
| `M341` | Contrastive ROUNDS, and the practical number | REJECTED AS RUN / THE DIRECTION SURVIVES, THE IMPLEMENTATION DOES NOT | `91622cea31` | 514 |
| `M342` | How the perfect selector's placement is DISTRIBUTED over boards | KEY / THE MEAN DESCRIBES NO BOARD / AND IT SPLITS THE FAILURE IN TWO | `c85b0bf838` | 516 |
| `M343` | The frame prior against the ORIGIN failures M342 found | ACCEPTED / IT MOVES THE MEDIAN FROM ZERO TO 0.45 / AND IT IS NOT FREE | `a638bd6fbb` | 518 |
| `M344` | The REAL pipeline per board with the frame prior, and an empirical hardness model | KEY / THE BIMODALITY IS THE ORACLE'S, NOT OURS / AND VOLUME PREDICTS THE OUTCOME AT 0.955 | `d454b22fab` | 520 |
| `M345` | A vote threshold chosen PER BOARD, to a target volume | ACCEPTED, PENDING ITS CONTROL | `a2d237f557` | 522 |
| `M346` | The matched control: is it adaptivity or just a lower bar | ACCEPTED / IT IS THE ADAPTIVITY / SHIPPED | `a2d237f557` | 524 |
| `M347` | The per-board bar through the WHOLE pipeline | RETRACTS M346's SHIPPING CLAIM / THE ADJACENCY GAIN DOES NOT CONVERT | `5819499e7e` | 526 |
| `M348` | The vote bar and the edge ORDER, swept through the whole pipeline | ACCEPTED: THE DEFAULT MOVES TO TEN / ORDER IS CLOSED | `784a8407e0` | 528 |
| `M349` | Hygiene: every never-validated default, swept through the shipping path | THREE DEFAULTS CONFIRMED, TWO MECHANISMS CLOSED, ONE CANDIDATE | `a2cab5bdc0` | 530 |
| `M350` | The leftover fill, on 48 boards | ACCEPTED / DEFAULT CHANGED / AND IT NARROWS M226 RATHER THAN OVERTURNING IT | `704fee9b64` | 532 |

## M351–M400

| ID | Проверка / идея | Вердикт журнала | Commit | Строка |
|---|---|---|---:|---:|
| `M351` | The margin, the post-processing and the high-pass cut-off, through the shipping path | MARGIN CLOSED / POST-PROCESSING ALL EARNS ITS PLACE / SIGMA IS MIS-SET | `8bb927009b` | 534 |
| `M352` | CORRECTION to M351: sigma is not a free knob, it is the detail trade again | PROCESS NOTE / AND IT PRICES A BETTER EXCHANGE RATE | `602d2e7de1` | 536 |
| `M353` | Small sigma with a raised alpha, at MATCHED visible detail | REJECTED / M182's CHOICE STANDS / AND THE EXCHANGE RATE WAS AN ILLUSION | `0d756f18c2` | 538 |
| `M354` | The user finds a solved board by eye, and M344's reading is corrected | KEY / THE REAL PIPELINE IS BIMODAL AFTER ALL / AND A TRUTH-FREE DETECTOR THAT WORKS | `8e61d91273` | 540 |
| `M355` | A per-board vote bar chosen by that detector | REJECTED BY CONSTRUCTION / AND IT PRICES THE HEADROOM | `8e61d91273` | 542 |
| `M356` | Four judges of the finished LAYOUT, including an unbiased one | ALL REJECTED / THE PER-BOARD OPPORTUNITY IS GATED BY THE SAME DOOR AS EVERYTHING ELSE / AND IT GIVES M340 A CONSUMER | `bd3e4eb7fb` | 544 |
| `M357` | The board described once, the bar chosen from the description | REJECTED / AND IT EXPLAINS WHY EVERY PER-BOARD POLICY HAS FAILED | `e213625988` | 546 |
| `M358` | Why every per-board policy failed, and the tension it uncovers | KEY / DIAGNOSIS / CLOSES THE PER-BOARD THREAD AND NAMES THE PROJECT'S REAL DILEMMA | `70dd9adf11` | 548 |
| `M359` | Dissolving the components we do not trust, instead of placing them as blocks | REJECTED | `70a607aaca` | 550 |
| `M360` | The annealer has never annealed | KEY / A CENTRAL COMPONENT MEASURED AS A NO-OP | `a65f90d0ad` | 552 |
| `M361` | Why: 93% of proposals do not fit, and NONE of the rest is uphill | KEY / THE MOVE CLASS IS WRONG, NOT THE TEMPERATURE | `a65f90d0ad` | 554 |
| `M362` | A depth-TWO harvest, through the shipping path | REJECTED / CLOSES M253's WAY OUT | `7c3ea1849e` | 556 |
| `M363` | What the eighteen scorers are actually made of | KEY / INDEPENDENCE COMES FROM THE INPUT, NOT THE WEIGHTS / MEASURED AT MATCHED COUNT | `2c724ad558` | 558 |
| `M364` | Five views instead of three | REJECTED UNDER THE STANDING RULE / IT IS THE SAME TRADE AGAIN | `2c724ad558` | 560 |
| `M365` | Where the adjacency went, and what the views are really worth | KEY / MY COMPARISON WAS THE FAULT / AND THE VIEWS ARE VERY UNEQUAL | `a91be7a7a4` | 562 |
| `M366` | ANALYTIC views beat every learned restorer as voters | KEY / THE MATERIAL M365 SAID WAS MISSING / FIRST ARM TODAY TO IMPROVE THE ASSEMBLY | `3d24835fb7` | 564 |
| `M367` | Confidence-weighted voting, estimated without labels | ACCEPTED / PARTLY CLOSES THE ADJACENCY GAP | `3d24835fb7` | 566 |
| `M368` | The teammate's idea: contours as adjacency evidence | REJECTED AT FRAGMENT SCALE / AND THE PICTURE SHOWS WHY | `a2bbf54c1c` | 568 |
| `M369` | Do contours survive the corruption, and at what scale | DIAGNOSTIC / THE TEAMMATE IS RIGHT ABOUT SURVIVAL / THE SCALE IS THE PROBLEM | `a2bbf54c1c` | 570 |
| `M370` | Contours as a JUDGE of the assembled board | FIRST TRUTH-FREE JUDGE WITH REAL SIGNAL / DOES NOT YET CONVERT | `a2bbf54c1c` | 572 |
| `M371` | The analytic roster shipped: filters replace the restorers as views | ACCEPTED / DEFAULT CHANGED / FIRST CHANGE IN TWO DAYS THAT IMPROVES THE ASSEMBLY WITHOUT TRADING ANYTHING | `4966b4a737` | 574 |
| `M372` | What makes a good VIEW: not invention, and not over-smoothing | KEY / A SELECTION RULE, MEASURED FROM BOTH SIDES / AND THE ROSTER SATURATES | `634db5d379` | 576 |
| `M373` | The harvest saturates on every remaining setting | CLOSES THE HARVEST AS A KNOB | `d1edac9ab6` | 578 |
| `M374` | What the user's eye is ranking, with the layout held fixed | NEITHER RENDER FACTOR / THE V5 PREFERENCE IS BOARD LUCK | `8ede66a97a` | 580 |
| `M375` | The leftover fill, re-measured on the roster that replaced the one it was chosen on | SHIPPED / REVERSES M350 | `8ede66a97a` | 582 |
| `M376` | Does the roster COVER more of the truth, or only rank it better | THE ROSTER BUYS COVERAGE, BUT ONLY A FIFTH OF IT | `8ede66a97a` | 584 |
| `M377` | The ceiling is the mutual-best filter, not the noise draw | KEY / OVERTURNS THE READING OF M310 THAT HAS GOVERNED THE PROJECT / THE TRUTH IS IN THE EVIDENCE | `8ede66a97a` | 586 |
| `M378` | Island purity, and the statistic that buys it | KEY / THE LEVER THIS PROJECT HAS NEVER PULLED | `8d9e2355f4` | 588 |
| `M379` | M180's island merge, re-run on islands that are correct | OVERTURNS M180 / THE SIZE EFFECT IS REAL AFTER ALL | `8d9e2355f4` | 590 |
| `M380` | Filling a hole whose neighbours are RIGHT | OVERTURNS THE PREMISE OF M329 | `8d9e2355f4` | 592 |
| `M381` | Merging and filling in alternation, carried to the end state | REJECTED AS AN ASSEMBLY / THE MECHANISM IS NOT THE LIMIT | `8d9e2355f4` | 594 |
| `M382` | The margin STATISTIC, which every ordering in this pipeline has read the same way | REJECTED BY THE PIPELINE / THE STAND WAS RIGHT AND IT DID NOT MATTER | `bd6024907e` | 596 |
| `M383` | A selector trained where RANK is a variable | KEY / M317 RESTATED RATHER THAN OVERTURNED / AND THE DEPTH IT REACHES IS TWO | `0cd3f3497c` | 598 |
| `M384` | The selector as a SEED for the island route | ACCEPTED AS THE BETTER SEED / THE CEILING BARELY MOVES | `c8826aad5a` | 600 |
| `M385` | A second selector that can SEE the assembly the first one built | REJECTED / CONTEXT AS A FEATURE IS CLOSED | `4d6a67f476` | 602 |
| `M386` | What the metric actually pays for, measured off the end of M174's table | KEY / RESTATES THE WHOLE TARGET IN ONE NUMBER / AND CORRECTS THE RANKING RULE | `a4a6e63210` | 604 |
| `M387` | The colour route to absolute placement, priced end to end | KEY NEGATIVE / CLOSES THE FIELD AS A PLACEMENT LEVER / AND RETIRES M138'S TARGET | `f50e55f2c1` | 606 |
| `M388` | A matcher trained on the view it will be asked to judge | REJECTED / M292 EXTENDS TO FILTERS | `f50e55f2c1` | 608 |
| `M388-CORRECTION` | The filtered-view matcher was compared against the wrong control | THE VERDICT FLIPS FROM REJECTED TO NEUTRAL | `df6511b324` | 638 |
| `M389` | The selector through the real pipeline, without the leak | SMALL POSITIVE / NOT YET A DEFAULT | `f50e55f2c1` | 610 |
| `M390` | Where the placement is lost: in the block, or in where we hang it | KEY / SPLITS THE FAILURE AND PRICES BOTH HALVES | `241da3f480` | 612 |
| `M391` | Is an 8x8 colour map predictable from the bag at all | KEY NEGATIVE / CLOSES THE ANCHORING SIGNAL AND M138'S PREMISE WITH IT | `241da3f480` | 614 |
| `M392` | A merge search that can take a merge back | REJECTED / RE-DERIVES M318'S PRINCIPLE IN THE ISLAND SETTING | `c49a59c608` | 616 |
| `M393` | What we pay for visible detail, measured on our own layouts | MEASUREMENT / A DECISION FOR THE OWNER, NOT FOR ME | `c49a59c608` | 618 |
| `M394` | The selector and the corroborated merge, applied together | THEY DO NOT COMPOSE / THE BLOCK CEILING IS 42 FROM FOUR ROUTES | `eacaa64461` | 620 |
| `M395` | The target restated as bond percolation, which retires M316's | KEY / THE EXACT REQUIREMENT, AND IT RULES OUT DEPTH ONE ENTIRELY | `0cfddbef42` | 622 |
| `M396` | Volume to exhaustion, then the decoder itself | ACCEPTED / THE BEST BLOCK THIS PROJECT HAS / AND THE CEILING MOVES TO THE SELECTOR | `d2e373214f` | 624 |
| `M397` | The frame prior's weight, swept for the first time since it became the anchoring mechanism | REJECTED / THE DEFAULT IS ALREADY THE OPTIMUM | `d2e373214f` | 626 |
| `M398` | The assignment without a candidate list, the path-cover constraint, and what the pipeline says | REJECTED BY THE PIPELINE / AND ~310 CORRECT BONDS IS THE CEILING FROM SIX DIRECTIONS | `9a2fa76d21` | 628 |
| `M399` | Vote on the ASSIGNMENT rather than on the edge | KEY / MORE TRUTH FROM THE SAME NETWORKS / A NEW KIND OF OPINION | `963170d406` | 630 |
| `M399-CORRECTION` | The assignment vote was compared against the wrong baseline | THE HEADLINE IS WITHDRAWN / IT CONCENTRATES, IT DOES NOT EXPAND | `7c70b18d4e` | 632 |
| `M400` | The assignment vote as a selector feature | REJECTED / THE CANDIDATE SET ALREADY CONTAINED IT | `7c70b18d4e` | 634 |

## M401–M420

| ID | Проверка / идея | Вердикт журнала | Commit | Строка |
|---|---|---|---:|---:|
| `M401` | The generic photograph prior as an anchor | REJECTED / ANCHORING HAS NO SIGNAL LEFT | `056aaf6cea` | 636 |
| `M402` | Where our seven correctly placed fragments actually come from | KEY / BELOW THE KNEE THE METRIC IS NEARLY BLIND TO ASSEMBLY QUALITY | `61c20bd747` | 640 |
| `M403` | Sinkhorn training, mixed instead of substituted | ACCEPTED / M116 REOPENED WITH A MONOTONE GAIN / THE FIRST THING TODAY THAT IMPROVES WITH ITS KNOB | `47d13061c9` | 642 |
| `M403 COMPLETION` | The Sinkhorn weight swept to its optimum | THE KNOB IS EXHAUSTED AT WEIGHT 1.0 / THE GAIN IS 3% | `e5cf15e766` | 644 |
| `M403-CORRECTION` | The Sinkhorn gain is below the noise floor this project already measured | THE RESULT IS UNPROVEN, NOT WRONG / AND THE FAULT IS MINE TWICE OVER | `35848599d4` | 658 |
| `M403 FINAL` | The Sinkhorn A/B across four seeds | THE GAIN IS ZERO / M403's CORRECTION CONFIRMED | `392e68eb48` | 672 |
| `M404` | 2x2 square closure over SHORTLISTS instead of mutual bests | REJECTED / COINCIDENCE AGAIN | `0932455969` | 646 |
| `M405` | The roster re-scored on top-1, which is the currency it was never chosen for | KEY / THE SHIPPING FUSION IS WORSE THAN THE RAW VIEW ALONE | `0932455969` | 648 |
| `M405-CORRECTION` | The pipeline was already doing what M405 recommended | THE ACTION IS VOID / THE MEASUREMENT STANDS | `f9907c0274` | 652 |
| `M406` | The inference calibration re-tuned for top-1 | AT ITS OPTIMUM ALREADY | `0932455969` | 650 |
| `M407` | The knee, on bonds that cluster the way our evidence does | KEY / CORRECTS M395 / THE TARGET IS NEARER THAN 552 | `39a6e566e1` | 654 |
| `M408` | The raw top-1 set against every cleverer edge set, on the pipeline's own matrix | KEY / THE LARGEST BLOCK THIS PROJECT HAS / AND THE SELECTOR LOSES ON THE NEW CURRENCY | `6807c9e6c9` | 656 |
| `M409` | Matrix edge sets through the real pipeline, and what the clean block is worth below the knee | KEY NEGATIVE / THE BLOCK IS THE WRONG PROXY AT OUR OPERATING POINT | `d45926dd76` | 660 |
| `M410` | Choosing partners to close SQUARES instead of to score well | REJECTED / THE GRID'S OWN OBJECTIVE HAS ITS OPTIMUM AWAY FROM THE TRUTH TOO | `598a0bc62b` | 662 |
| `M411` | A policy that ASSEMBLES, trained on the states its own mistakes produce | ACCEPTED / THE FIRST ARM OF THE SESSION TO MOVE BOTH AXES AT ONCE / THE OWNER'S IDEA | `3befcfc24b` | 664 |
| `M411 GATES` | The policy's precision-volume frontier, which no fixed rule offered | ACCEPTED / THE KNOB IS THE RESULT | `c066b2a3ec` | 666 |
| `M412` | The five-candidate chooser, carried to the end | REJECTED / MORE DATA IS THE ONLY THING THAT COULD CHANGE IT | `c066b2a3ec` | 668 |
| `M413` | The policy seeded from the SHIPPING harvest, which is the question that ships | THE MECHANISM WORKS AND THE PLACE IS WRONG / GROWTH IS NOT WHERE THE GAP IS | `be0b1cf239` | 670 |
| `M414` | A policy over MERGES, with a split action | HARNESS BUG CAUGHT BY THE NUMBERS / FIRST RUN VOID | `392e68eb48` | 674 |
| `M415` | Rank a merge by the BEST seam in its contact, not the average | KEY / SHIPPABLE / THE ISLAND RULE HAS RANKED ON THE WRONG STATISTIC SINCE AUGUST | `b8ab896cc0` | 676 |
| `M416` | Growth in the pipeline, from a dirty seed and from a clean one | KEY NEGATIVE / GROWTH IS CLOSED, BY SIX INDEPENDENT MECHANISMS / AND THE CLEAN SEED IS NOT | `877dcd86ac` | 678 |
| `M417` | The clean margin seed, which replaces the harvest | KEY / THE FIRST ARM TO MOVE PLACEMENT / SIMPLER AND CHEAPER THAN WHAT IT REPLACES | `877dcd86ac` | 680 |
| `M417-CORRECTION` | The margin seed does not double placement | THE HEADLINE IS WITHDRAWN / IT IS PARITY, NOT A WIN | `078e37dbe3` | 682 |
| `M418` | The value function, with STOP and UNDO | REJECTED / IT TAKES BACK WHAT THE POLICY EARNS | `078e37dbe3` | 684 |
| `M419` | The chooser at seventeen times the data, and the headroom it recovers | THE DATA LEVER IS REAL AND THE CEILING IS LOW | `492dd2a820` | 686 |
| `M420` | The metric pays for CONTENT, not for identity | KEY / EVERY CEILING THIS PROJECT HAS MEASURED IS AN INDEX CEILING | `6fb563c4b7` | 688 |

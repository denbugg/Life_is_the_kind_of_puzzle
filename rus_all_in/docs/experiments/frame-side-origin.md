# Dedicated full-resolution frame-side origin model

Status: **bounded train-only discovery failed. Stop this frame-membership →
integer-hit cyclic-roll formulation; do not open fresh64 or use it in
production.** The learned side classifier was weaker than the frozen Socket v2
border heads, and an exact-frame oracle proved that perfect frame membership is
still insufficient to choose the useful cyclic origin of a fragmented layout.

## Duplication audit and distinction

The nearest prior work was audited before selected target access.

| Prior family | Established result | Distinction of this experiment |
|---|---|---|
| Socket v2 direct border heads | Exact top24 recall/F1 was only about `6.03–7.42%` in the original local audit, above the `4.17%` random cardinality baseline. | Socket d64 context and four border logits are frozen inputs/baseline. The new nonlinear model also reads raw/restored 20×20 boundary sequences and board context. |
| Socket v3 border-distribution head | Six row/column score statistics gave mixed local changes and regressed exact `59→44` aggregate tiles. | No linear score-statistics head or raw-rank auxiliary is repeated. The unit of prediction is a tile-side membership from pixels, oriented sequences and frozen context. |
| V30 / M247 frame priors | V30 reported row `10.35%`, column `8.57%`, border F1 `54.65%`, but under an obsolete split/upstream; M247 exposed weak board-dependent absolute frame evidence. | Current exact corruption, explicit lineage exclusion and a source-disjoint train-only panel are used. No population atlas, generic photo/centre/face/background prior or semantic placement rule appears. |
| Frozen Socket cyclic-border5 | On its independent source48 discovery, the analytic cut/border scorer gained `+0.375` exact tile/board with a small adjacency loss. | This is the matched comparator. The candidate uses exact top24 learned frame sets as a scale-free integer primary objective, with raw Socket cut evidence only as a deterministic tie-break. |
| Full-resolution boundary denoiser | A stride-one 20×20 restored view improved union candidate supply but was unsafe as a direct replacement scorer. | Restored pixels are only an optional matcher-visible input alongside raw pixels. Output always remains the 576 original upright tiles. |
| Whole-layout cyclic-origin CNN | A 45,345-param circular CNN had no best-roll ranking signal and failed adjacency safety. | No whole-grid coordinate/roll is learned. This model is tile-permutation-equivariant and predicts four inference-visible frame memberships only; the final action is an exhaustive strict `numpy.roll`. |

This is therefore not another Socket v3 score-stat head, downsampling U-Net,
whole-layout CNN, coordinate unary, fixed fusion, or centre/face heuristic.

## Frozen model and legal inference contract

For each original upright `20×20` tile, the 51,865-parameter classifier sees:

- raw RGB and per-tile normalized raw RGB;
- full-resolution-denoiser RGB and its per-tile normalized view;
- frozen permutation-equivariant Socket d64 context;
- all four standardized frozen Socket border logits;
- board mean/max context formed without tile IDs or slot embeddings.

A stride-one stem plus five gated depthwise blocks preserves `20×20` throughout.
The top, bottom, left and right width-5 bands become consistently oriented
length-20 sequences, followed by dilated 1-D blocks `(1,2,4)`. The output is
four logits per tile. Training uses exact 24-positive all-positive listwise
loss, balanced BCE and paired-corruption consistency after aligning the two
shuffles by known synthetic organizer-train positions.

Inference always takes exactly the top 24 tiles for each side. Starting from
the unchanged raw d64 decoder144 strict layout, all 576 global translations
are enumerated. The fixed lexicographic rule maximizes integer predicted-frame
hits, then raw Socket pairwise/cut objective; there is no learned scale, blend
weight or post-hoc arm sweep. Rendering keeps the original upright input tiles.

## Capacity, preregistration and lineage

An 8×8 procedural capacity task reached macro F1 `100%` after 80 MPS updates.
The final loss was `2.079766`; the all-positive listwise term has the nonzero
mathematical lower bound `log(8)=2.079442`, so excess was only `0.000324`.

Before either fit or evaluation target access, the following were frozen:

- selection SHA-256:
  `2cdfca00618671ddeab6fd829f97ce49d2f975d9c977ec248a043872de89f7b4`;
- fit256 order digest:
  `baf8116b636aecd1dee16ee8f3b8f049b2c1b2b1886b9483d4a305c44f7c4a27`;
- evaluation32 order digest:
  `b81a58eb124aea915fdc0fcf08b4378bc0fea63f7d3b112961c05a0fd77cd80e`;
- preregistration SHA-256:
  `1483ea867782f955e650555e71c6626eb02379cfe0124a7006a2c07becdd1a78`.

All 288 sources are manifest-`train`, fit/evaluation-disjoint and disjoint from
the 4,333-name actual-panel/checkpoint exclusion union captured by the
selection receipt. This includes actual Socket lineage and active
relation/fusion/origin/QAP/Pointer rosters. Broad available-pool declarations
are recorded by the snapshot but are not misrepresented as target exposure.
No calibration, holdout, competition test or fresh64 was opened.

Fit used paired draws 0/1 on 256 sources and 600 AdamW updates on MPS. Frozen
feature/restored-view precompute took `39.90 s`; training took `157.10 s`.
The checkpoint and both eval layouts were frozen before exact scoring:

- checkpoint SHA-256:
  `c52b99ede9a8ad93f0a1345fafc5a7c251c8248555943f92b67b67273cfc08ed`;
- frozen-predictions SHA-256:
  `1fb112b60ba2e23744cb609946d6d4b8f018eb3ffd56b161385665d45310851a`.

The low D1 gate required either macro frame F1 `+2 pp` or exact
`+0.1` tile/board, with adjacency loss at most `0.2 pp` and 32 strict
permutations.

## Source-disjoint evaluation32 result

| metric | frozen comparator | learned treatment | delta |
|---|---:|---:|---:|
| frame macro F1 / exact-cardinality recall | `7.7799%` | `7.2591%` | **`−0.5208 pp`** |
| top-side F1 | `15.1042%` | `13.9323%` | `−1.1719 pp` |
| bottom-side F1 | `5.8594%` | `5.5990%` | `−0.2604 pp` |
| left-side F1 | `4.1667%` | `4.2969%` | `+0.1302 pp` |
| right-side F1 | `5.9896%` | `5.2083%` | `−0.7813 pp` |
| exact tiles / board | cyclic-border5 `1.53125` | learned frame roll `0.65625` | **`−0.875`** |
| adjacency | `13.4652%` | `13.1567%` | **`−0.3085 pp`** |

Exact W/T/L was `4/15/13`; all `32/32` outputs were strict rolls of the
original-tile permutation. Both discovery branches and adjacency safety failed.
Report SHA-256 is
`80fe0d28ffcab11bdd2cbb43f9a2f4012d1c38d9034ca8287d55775c70b45c2e`.

## Same-opened-panel bottleneck audit

A target-assisted diagnostic reused only those already-opened 32 cases. It is
not deployable and opened no new source. To avoid CPU/MPS decoder drift, it
reconstructed the exact frozen decoder144 base layout by inversely rolling the
two frozen candidate layouts and asserted that both reconstructions agreed on
all boards.

| origin rule on the same frozen base layout | exact tiles / board | adjacency |
|---|---:|---:|
| frozen cyclic-border5 comparator | `1.53125` | `13.4652%` |
| learned top24 frame sets | `0.65625` | `13.1567%` |
| Socket v2 top24 frame sets + integer-hit rule | `1.09375` | `13.1058%` |
| target-assisted best learned/Socket arm per side | `0.75000` | `13.1539%` |
| **oracle true 24 frame members per side** + integer-hit rule | **`1.15625`** | `13.1765%` |
| oracle cyclic roll maximizing exact | **`13.03125`** | `13.0548%` |

The learned∪Socket top24 union recalls only `20.05/9.24/6.77/9.38%` of true
top/bottom/left/right frame tiles. More importantly, even perfect frame sets
remain worse than cyclic-border5 while an exact-roll oracle contains about 13
correct tiles. The dominant bottleneck is therefore structural: on a
fragmented decoder layout, maximizing marginal frame-membership hits does not
identify the translation that aligns the largest correct internal component.
Classifier quality is also weak, but a larger nearby side classifier cannot
repair this conversion ceiling.

Verdict: **fail-stop**. Do not sweep width, steps, denoiser arms, frame-hit
weights or fresh panels for this formulation. A future materially different
origin mechanism must score translation-consistent internal component/edge
evidence or another inference-visible whole-layout anchor, rather than only
four marginal frame sets.

Diagnostic SHA-256:
`b825d4e2021bbffa99a6bf26001cdee306784c9bc97f4f8363bd68ccb15558ef`.

Artifacts and code:

- `configs/frame_side_origin_preregistered_v1.json`;
- `src/aiijc_puzzle/frame_side_origin.py`;
- `scripts/run_frame_side_origin.py`;
- `scripts/analyze_frame_side_origin.py`;
- `tests/test_frame_side_origin.py` and
  `tests/test_run_frame_side_origin.py`;
- `outputs/frame-side-origin/v1-fit256-s600-eval32/report.json` and
  `bottleneck_diagnostic_v2.json`.

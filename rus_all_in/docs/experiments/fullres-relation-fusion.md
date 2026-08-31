# Learned fullres/component-relation fusion

## Verdict

**Strong local selector, but exact decoder discovery failed; keep only as a
research primitive.**  A target-free learned selector over the raw plus
full-resolution-restored candidate union materially improves relation ranking
and confidence.  Its separately authorised exact source40 decoder pilot,
however, produced only `+0.025` correct absolute tile/board and `+0.00679 pp`
adjacency.  Both preregistered D2 branches failed.  It must not replace the
raw d64 decoder, trigger a fresh promotion panel or touch competition test.

This closes the obvious continuation of the full-resolution denoiser result:
the extra restored-only neighbours are real and learnably selectable, but the
current top-query score-substitution bridge does not convert that local signal
into a material global-layout gain.

## D1: learned selector result

The frozen raw d64 decoder144 components define the relation queries.  Each
query receives the union of raw and fullres-restored d64 top-32 proposals; raw
candidates are inserted first and cannot be displaced by restored-only fill.
The 12,886-parameter head sees only inference-time evidence:

- raw/restored d64 Socket and OT scores, ranks, margins and reciprocity;
- the full-resolution restored boundary descriptor;
- raw/restored member-token context and their drift;
- component size, shape, density, confidence and relative translation;
- the frozen component-relation v1 score and candidate supply origin.

It learns a bounded residual over the frozen relation score plus a separate
candidate-correctness logit.  There is no source identity, shuffled tile index,
absolute target coordinate or fixed score fusion.  Exact synthetic truth is
attached only for train loss and post-freeze metrics.

On the preregistered source-disjoint local16 panel:

| Metric | Frozen relation | Learned fusion | Delta |
|---|---:|---:|---:|
| union candidate coverage | raw roster `54.1289%` | union `61.5874%` | **`+7.4585 pp`** |
| relation R@1 | `12.0112%` | `13.1686%` | **`+1.1573 pp`** |
| relation R@5 | `35.9924%` | `37.4308%` | **`+1.4384 pp`** |
| top-32 correct relations/board | `6.5625` | `11.1875` | **`+4.6250`** |
| top-32 precision | `20.5078%` | `34.9609%` | **`+14.4531 pp`** |
| top-144 correct relations/board | `21.6875` | `34.3750` | **`+12.6875`** |

Supply, ranking and confidence branches all passed the low discovery gate.
The same frozen checkpoint was rerun independently on all 16 boards and
reproduced these metrics bit-for-bit.  D1 itself had no layout decoder and did
not authorise promotion.

## D2: preregistered exact decoder pilot

Only after D1 passed, one exact treatment was frozen before source40 access:

```text
fusion winner + correctness confidence
  -> top-8 component-direction queries
  -> translation-consistent relation forest
  -> deterministic scale-preserving substitution
       max(current, row-best, column-best) + nextafter(+inf)
  -> unchanged decoder144 / max-swap24
  -> same raw-evidence cyclic-border5 as comparator
```

The forest is materially different from a fixed confidence bonus: it can admit
a restored-only contact absent from the raw hard matching, but only after the
whole proposed relation passes outgoing/incoming capacity, coordinate-cycle,
collision and 24×24 span checks.  Dustbins and all unselected score cells remain
bitwise unchanged.  The comparator is raw d64 decoder144 plus cyclic-border5.

The fixed `top8` cap came from a bounded capacity check on the first eight
already-opened D1 fit sources: cap8/16/32 gave mean exact deltas
`+0.375/+0.250/+0.125` tile/board.  No D2 source was used for this choice and
there was no cap sweep on the exact panel.

Both arms were written as strict layouts before exact scoring.  The frozen
artifact contains 40/40 permutations of the original upright 576 tiles; restored
pixels remain matcher-only.

| Exact source40 × draw1 | Raw decoder144 + cyclic5 | Fusion forest + same decoder/cyclic | Delta |
|---|---:|---:|---:|
| correct absolute tiles/board | `0.750` | `0.775` | `+0.025` |
| direct placement | `0.13021%` | `0.13455%` | `+0.00434 pp` |
| translation-aligned tiles/board | `13.325` | `13.325` | `0.000` |
| adjacency | `13.91078%` | `13.91757%` | `+0.00679 pp` |

Exact W/T/L was `9/22/9`; the source bootstrap 95% interval was
`[-0.20,+0.25]` tile/board.  Mechanism diagnostics rule out a no-op:
the forest accepted `5.925` contacts/board, including `3.075` absent from the
original hard matching, and `3.475` accepted contacts/board survived the new
matching.  The failure is conversion strength/generalisation, not an inactive
implementation.

The D2 gate required strict permutations, adjacency loss no worse than
`0.2 pp`, and either:

1. exact gain at least `+0.1` tile/board; or
2. non-negative exact plus adjacency gain at least `+0.05 pp`.

Permutation and loss checks passed, but observed exact `+0.025` and adjacency
`+0.00679 pp` missed both material branches.  Status is `fail-stop`; the later
fresh64×draw2 promotion panel is not authorised.

## Post-hoc conversion audit on the same opened source40

No new target panel was opened.  A clearly target-assisted, non-deployable
audit reused exactly the D2 source40 to locate where the local signal is lost.
It measured truth at each conversion boundary and added two oracle ceilings:
correct ordering of the existing raw hard edges, and correct supplied-union
relations inside the same top8 forest.  It also compared dirty cyclic5 with the
best exact-label cyclic roll.

The attrition is sharp:

| Stage, mean/board | Result |
|---|---:|
| exact component-direction queries | `1,222.18` |
| supplied by raw candidates | `672.20` (`55.00%`) |
| supplied by raw∪restored union | `775.20` (`63.43%`, **`+8.43 pp`**) |
| learned correct top-1 queries | `92.45` (`11.93%` of supplied) |
| learned correct among the top8 forest queries | `3.375 / 8` (`42.19%`) |
| restored-only winners among learned top8 | **`0.0 / 8`** |
| learned new/removed hard edges | `6.625 / 6.625` |
| correct learned new/removed hard edges | **`0.825 / 0.825`** |

Thus the restored union adds real oracle coverage, but the exact top8 treatment
never selects a restored-only winner.  Its score substitution exchanges the
same number of correct hard edges in and out, so the 196.075 correct projected
hard edges/board are unchanged.  Component geometry is consequently flat:
tile-weighted translation purity `72.27→72.21%`, pairwise relative accuracy
`20.55→20.50%`, and largest-component truth purity about `30.6→30.5%`.
The decoder preserves its chosen components (`98.6%` dominant-shift support),
so destructive packing is not the main loss.

Oracle evidence separates two remaining bottlenecks:

| Diagnostic arm | pre-cyclic exact | pre-cyclic adjacency | oracle cyclic exact |
|---|---:|---:|---:|
| raw baseline | `0.675` | `13.972%` | `14.250` |
| learned forest | `0.700` | `13.967%` | `14.225` |
| oracle correct ordering of raw hard edges | `0.825` | **`20.226%`** | **`22.200`** |
| oracle correct-union top8 forest | `0.775` | `14.194%` | `14.200` |

Oracle raw-hard ordering leaves the projected matching itself unchanged but
raises component tile-weighted purity `72.27→81.96%`, pairwise relative
accuracy `20.55→42.24%`, adjacency by `+6.255 pp`, and translation-aligned
tiles `13.50→21.33`.  This says the most valuable relative-layout work is
better global selection/ordering of already available hard edges, not another
nearby restored top8 substitution.

Independently, the best target-assisted cyclic roll raises baseline exact
`0.675→14.250` tiles/board, whereas dirty cyclic5 reaches only `0.750`.
Absolute origin is therefore a major exact-position bottleneck.  It is not the
only bottleneck: even oracle origin leaves roughly `97.5%` of tiles wrong
because adjacency and large-component purity remain low.  A credible route
needs both a materially better relative graph/QAP consumer and a separately
learned, inference-visible origin signal; the oracle numbers are ceilings, not
methods.  The newly completed whole-layout origin CNN independently failed its
fresh gate, so these diagnostics do not authorise another origin panel.

## Lineage and legality

- D1 preregistration SHA-256:
  `b83e819d161524dccb37dabbee6faf7470a27b835a598eded9c8285fe487d7b5`.
- D1 used manifest-calibration fit32/local16 digests
  `2d1f5bf1e1ee5027322807a6e9697838680e4bde552d33efd9e598940e3ebdff`
  and `8a29344b096e970ddf4d91c919dd8161a738165f53313ecfcb5ead22d5df4f72`.
  It excluded the actual component confirm24/decoder40 rosters.  Organizer
  holdout700 and competition test were untouched.
- Some older reports broadly declare an available/source pool and their
  recursive `*_filenames` union covers all 5,600 manifest-train sources.  This
  is not actual exact-target access.  D2 therefore used exact-panel freshness:
  zero overlap with actual Socket/relation/fullres/fusion/pointer/origin
  fit/eval/confirm/decoder panels and every prior exact-layout source panel.
- D2 preregistration SHA-256:
  `46ae0388fee837efbb1bb47f665a442f1badab62bfc6279b6728c1f5a840114a`.
  The source40 order/set digests are
  `058bec96c295db71580436eb254c2b77f600a69a51e2d21c31b9e66bfc8a1bf9`
  and `032b0c7f0bb542f0184827ceaabe683729be7c62fea970e0b924b1af857719a6`.
  PID `82413`, config digest and roster digests were published while the runner
  was paused before target access.
- A concurrently selected component-origin panel explicitly excluded D1 and
  D2 rosters before its own freeze; its committed fit256/eval16 overlap is zero.
- Synthetic corruption/shuffle uses known exact train imagery only.  No
  organizer holdout, decoder40 reuse or competition test was opened for D2.

## Artifacts and guidance

- feature/selector model: `src/aiijc_puzzle/fullres_relation_fusion.py`;
- forest bridge: `src/aiijc_puzzle/fullres_relation_decoder.py`;
- D1 runner: `scripts/run_fullres_relation_fusion.py`;
- D2 runner: `scripts/run_fullres_relation_fusion_decoder_d2.py`;
- D1 checkpoint SHA-256:
  `ebc202d623cf8323319a00ffa09418d9d03ee39e70b96ac9cafdcc3f4003877c`;
- D1 report SHA-256:
  `3afbd75773370a7f80ba95060b0a7f0dfb4a3343fac69019a79170641f7a42c9`;
- D2 frozen-layout SHA-256:
  `74c5bd64e3e34827e7fd130651894c7bc560f13499b8e5ac2d3716c8d12268c6`;
- D2 report SHA-256:
  `8884375df68870b8b06e62e975d26ac2a829294ac99a7fdd0fa665214a71729e`.
- post-hoc conversion-audit report SHA-256:
  `9fd18c091583d0020f88a91b026541c707c9188f13fa0c6ada2170d241cc62e8`.

Do not repeat top-query score substitution with nearby cap/budget values on a
new exact panel.  The reusable evidence is the learned union selector and its
large local confidence gain.  A future continuation needs a materially
different global consumer—e.g. joint sparse graph/QAP optimisation that can use
multiple alternative edges without forcing each selected relation into one
local matching cell—and must earn a new train-only capacity gate before any
fresh exact evaluation.

## Conservative Union-hard priority continuation

A later bounded continuation reused the already-opened D2 source40 without
opening a fresh panel. It kept the Union-v2 hard projection fixed, selected the
top32 component-direction queries by the fusion winner's learned confidence,
considered within-query top5 candidates, and intersected their canonical
contacts with the existing 1,104 Union hard edges. A bounded noisy-or support
boost, scaled by one target-free hard-confidence standard deviation, changed
only component-edge ordering. No restored-only edge entered the matching and
restored pixels remained matcher-only.

This is materially more conservative than D2 forest/substitution, but it also
failed on the full opened40:

| metric | Union-v2 | fusion priority | delta |
|---|---:|---:|---:|
| exact tiles / board | `1.275` | `0.975` | `−0.300` |
| adjacency | `14.83696%` | `14.81658%` | `−0.02038 pp` |
| correct fixed top288 / board | `151.300` | `151.075` | `−0.225` |

Only `25.025` Union hard edges/board were supported on average, so the result
is not caused by a broad supply replacement. The fixed gate failed on all
three metrics; do not sweep the query cap, candidate cap or boost weight on
this panel. Implementation:
`src/aiijc_puzzle/fullres_fusion_union_priority.py` and
`scripts/run_fullres_fusion_union_priority_opened40.py`. Frozen report SHA-256:
`82a11a6267c9538106bf531e62fa3cc6f86844fc33b7b3262aa199908e4f4329`.

```bash
.venv/bin/ruff check src/aiijc_puzzle/fullres_relation_fusion.py \
  src/aiijc_puzzle/fullres_relation_decoder.py \
  scripts/run_fullres_relation_fusion.py \
  scripts/run_fullres_relation_fusion_decoder_d2.py \
  scripts/run_fullres_relation_fusion_conversion_audit.py \
  tests/test_fullres_relation_fusion.py \
  tests/test_fullres_relation_decoder.py \
  tests/test_run_fullres_relation_fusion.py \
  tests/test_run_fullres_relation_fusion_decoder_d2.py \
  tests/test_run_fullres_relation_fusion_conversion_audit.py
PYTHONPATH=. .venv/bin/pytest -q \
  tests/test_fullres_relation_fusion.py \
  tests/test_fullres_relation_decoder.py \
  tests/test_run_fullres_relation_fusion.py \
  tests/test_run_fullres_relation_fusion_decoder_d2.py \
  tests/test_run_fullres_relation_fusion_conversion_audit.py
```

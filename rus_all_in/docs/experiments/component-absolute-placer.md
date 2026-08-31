# Independent component absolute placer

Status: **bounded train-only D1 fail-stop; no promotion or fresh panel.** This
was the only activated continuation of
the conditional-go in
[the component voting audit](component-absolute-translation-voting-audit.md).
It does not change the default solver or authorize competition-test access.

## Why this is not another shared origin vote

The prior audit closed shared component voting, a larger d32 component MLP,
isolated population-position renders and size/confidence purity rules. V1 keeps
that no-go. It instead makes at most one independent placement decision:

1. freeze raw Socket-v2 `decoder144` components;
2. render each nontrivial component as a literal masked mosaic of original
   upright 20×20 tiles, with raw RGB and per-tile-normalized RGB channels;
3. apply stride-one pixel blocks and masked relative-component lattice blocks,
   then jointly condition every component on the permutation-equivariant set of
   all current-board components;
4. predict exact-component purity and a masked joint distribution over feasible
   24×24 absolute offsets; offset CE sees exact-pure components only;
5. select at most one component by purity probability × maximum offset
   probability using one fit-only calibrated threshold;
6. anchor it independently and collision-pack every remaining original upright
   component/tile while minimizing displacement from the baseline layout. An
   anchored board receives no independent cyclic post-roll. If no anchor passes,
   the candidate is exactly raw `decoder144 + cyclic-border5`.

There is no tile ID/slot embedding, source identity, face/centre/background
heuristic, resize, warp, restored output pixel or shared global component roll.
All 576 originals must appear exactly once.

## Frozen protocol

- Architecture: 140,561 parameters; pixel width 24, three stride-one pixel
  blocks, three relative-lattice blocks, two width-64 four-head set layers.
- Fit: source-disjoint manifest-train 224 sources, draws 0/1, one source pair
  per update, at most 600 AdamW updates. Purity uses balanced BCE; feasible
  offset uses size-weighted CE on exact-pure components. Exactly matching
  train-only component geometry across the paired draws receives probability
  consistency loss.
- Calibration: a separate 32 fit sources on draw2. The one frozen rule maximizes
  signed anchor-size utility among board-top score thresholds selecting at least
  four boards, with positive utility and precision ≥0.25 required; otherwise it
  forces fallback-only.
- D1: a new source-disjoint manifest-train32, draw0. Dirty component geometry,
  scores, offsets and both strict layouts are frozen before exact scoring.
- Gate: purity AP at least 2× the strongest of prevalence, inverse size and
  accepted-edge-confidence AP; exact at least +0.25 tile/board; adjacency loss
  no worse than 0.2 percentage point; all 32 layouts strict.

Selection commitment SHA-256:
`2b2b0c90e559ca2a6c7898d8eaccc4f8944a73c35d7d78722e0bb30939e0ebe6`.
Fit/evaluation order digests are `29438da4426243499e22c598887559d1f78215d7ae2e687861ce1c8d6d2d61c9`
and `f3f86ad87dbc42df0ddf6c9c7ec4ac778d20f30a757935ac7337ddbac9b85ca0`.
The exclusion union contains 4,927 declared actual/panel filenames and includes
the direct-hard-edge fresh64 confirmation plus its report and the Edge-v1
selection. Selected overlap is zero.

Static preregistration:
`configs/component_absolute_placer_preregistered_v1.json`, SHA-256
`20a60905b2bc480b440d1181da6e79c5caeb368ca276955d0293c2712ea80b73`.
It was written before any selected fit/evaluation target access.

## Capacity preflight

The grid4 four-component smoke learned both exact-purity classification and two
feasible absolute offsets in 120 MPS updates:

- loss `3.182943 → 0.000397`;
- purity AP `1.0`;
- pure-offset top1 `1.0`.

Artifact SHA-256:
`outputs/component-absolute-placer/capacity-v1/report.json` =
`9ab072ec2791a15f37f064c6aa193668c2a97623dbe07a6907f051ea90077824`.

## D1 result

The fit-only calibration correctly abstained. Purity AP was `0.569762` versus
the strongest simple baseline AP `0.402892` (`1.414×`), but joint feasible
offset top1 on 472 pure components was only `0.2119%`, effectively chance. No
board-top component had enough correct purity × offset confidence for positive
signed size utility under the frozen rule, so threshold `1.000001` selected
zero anchors.

On the frozen evaluation32 the purity signal transferred, but remained below
the material gate:

| Metric | Comparator / simple baseline | Candidate | Gate |
|---|---:|---:|---:|
| exact-component purity AP | `0.425354` | `0.593700` (`1.396×`) | `≥2×` |
| exact tiles / board | `1.78125` | `1.78125` | gain `≥0.25` |
| adjacency | `13.2416%` | `13.2416%` | loss `≤0.2 pp` |
| strict original permutations | — | `32/32` | `32/32` |

Because the fit-only selector abstained, all 32 treatment layouts were exactly
the frozen fallback comparator: wins/ties/losses `0/32/0`. This is a valid
negative result rather than an eval-selected arm. It isolates the remaining
bottleneck: native component pixels and board-set context improve purity
ranking over edge confidence, but do not carry usable absolute-position
information. The independent absolute-offset head plus conservative one-anchor
packing formulation is therefore closed as tested. Do not lower the threshold,
anchor an impure/uncertain component, increase capacity, add a shared vote, or
open fresh64 for a nearby variant. Relative-geometry evidence may still be used
by the independently confirmed direct-hard-edge path; this result does not
invalidate it.

Authoritative artifacts:

- checkpoint SHA-256
  `82ae77adadb37251eb1e6eaeef1ba1a7bc4a8f635c02dea24a4d3d0fc3c2109e`;
- immutable eval preregistration SHA-256
  `9b4c4e64ea4738de88449ed3ff1239df9ca6b656475cb8f149f2f2410547941b`;
- dirty predictions/layout freeze SHA-256
  `aa1ae32597216688572cb274b6cb4be5b9279e69447f1d5a93e1b58e7fccb2d5`;
- report SHA-256
  `9dfe0434541d77122ac16e052405f34c643ebed8317dacfdb422ced8f5166070`.

Organizer holdout and competition test remained unopened. No fresh64 run is
authorized.

## Reproduction

```bash
.venv/bin/python scripts/run_component_absolute_placer.py \
  --mode capacity --device mps --allow-nondeterministic-mps \
  --output-dir outputs/component-absolute-placer/capacity-v1

.venv/bin/python scripts/run_component_absolute_placer.py \
  --mode pilot --device mps --allow-nondeterministic-mps \
  --wait-for-eval-confirmation --log-every 20 \
  --config configs/component_absolute_placer_preregistered_v1.json \
  --output-dir outputs/component-absolute-placer/v1-fit256-s600-eval32
```

Core implementation is in
`src/aiijc_puzzle/component_absolute_placer.py`; the runner is
`scripts/run_component_absolute_placer.py`; focused tests are in
`tests/test_component_absolute_placer.py`.

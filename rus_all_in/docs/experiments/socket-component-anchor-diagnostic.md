# Socket component / absolute-anchor diagnostic

Status: target-assisted diagnostic only; no candidate was promoted.

## Protocol

The d32 SocketMatcher v2 checkpoint was evaluated on 24 fresh manifest-`train`
sources selected with namespace `aiijc-socket-matcher-v1` at offset 2304.  The
panel has zero filename overlap with the checkpoint lineage or any of the 11
pre-existing SocketMatcher reports.

The run was split into two processes:

1. `freeze` read dirty inputs only, froze SocketGlue assignments and the
   border-only / texture-centre decoder layouts, and wrote their SHA-256.
2. `evaluate` verified that hash and only then opened clean targets for a
   target-assisted diagnostic.

Component truth uses only the top 50% of recovered-position margins on each
board.  The organizer's exact permutation labels are unavailable, so even this
is diagnostic rather than hidden-label proof.  Full recovered-reference values
are also retained as a sensitivity check.

Reproduction:

```bash
.venv/bin/python scripts/diagnose_socket_component_anchors.py freeze \
  --output-dir outputs/socket-matcher/component-anchor-diagnostic-offset2304-dev24
.venv/bin/python scripts/diagnose_socket_component_anchors.py evaluate \
  --output-dir outputs/socket-matcher/component-anchor-diagnostic-offset2304-dev24
```

## Result

The main bottleneck occurs before absolute anchoring: fixed top-144-per-axis
constraints percolate through wrong edges and create large, internally invalid
components.

- Exact-edge rate among constraints whose two recovered labels are trusted:
  34.38%; among constraints actually added to the graph: 37.08%.
- Mean largest component: 42.42 tiles, but its mean trusted translation purity
  is only 17.31%; none of 24 largest components is internally consistent.
- Across all nontrivial components, tile-weighted translation purity is 36.92%.
- Under the full recovered reference, exactly rigid nontrivial components cover
  only 436 / 13,824 panel tiles (3.15%), average 18.17 tiles per board.  The
  largest such component is only 6 tiles.

Consequently, a large component is not yet a confidently recognized person,
face, or object.  It is usually several fragments joined through false edges.
Moving it to the centre moves mostly wrong relative geometry as one block.

For the 151 components internally consistent on at least two trusted labels:

| Anchor | Exact shift | Within Manhattan 2 | Mean L1 shift error |
|---|---:|---:|---:|
| Geometric centre | 0.00% | 2.65% | 11.99 cells |
| Learned OT border unary | 0.66% | 1.99% | 17.22 cells |
| Border + texture-centre weight 0.05 | 0.66% | 2.65% | 16.70 cells |

Only 8.61% of those components truly have a centroid within four cells of the
board centre.  For the largest internally consistent component on each of 22
boards, that rate is 0 / 22.  The texture prior therefore does not validate a
generic “confident component goes to the centre” rule.  It can stay a weak,
optional input-only feature, but not a hard placement decision.

At whole-board level, the texture-centre variant moved direct placement only
from 1.083 to 1.125 tiles per board, while reducing translation-aligned count
from 8.792 to 7.958 and adjacency from 8.224% to 8.103%.  This is noise-level
absolute gain with a measurable relative-layout cost.

## Decision

Do not spend the next run on stronger centre/background anchoring.  First build
a high-precision core graph: calibrated confidence or adaptive stopping,
mutual/cycle-consistent edges, and prevention of low-confidence bridge edges.
Once source-disjoint diagnostics produce substantially larger internally pure
components, absolute anchoring becomes a meaningful separate target.  A scene-
level border/background head remains more plausible than forcing every smooth
tile outward or every textured component inward.

Artifacts:

- `outputs/socket-matcher/component-anchor-diagnostic-offset2304-dev24/freeze_metadata.json`
- `outputs/socket-matcher/component-anchor-diagnostic-offset2304-dev24/frozen_predictions.npz`
- `outputs/socket-matcher/component-anchor-diagnostic-offset2304-dev24/report.json`

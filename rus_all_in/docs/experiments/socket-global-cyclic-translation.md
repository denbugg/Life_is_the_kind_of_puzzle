# Socket global cyclic translation: positive exact-placement confirmation

Status: **promote as an opt-in post-decoder absolute-anchor primitive**.  It is
not yet wired into the production submission path.

## Mechanism

The decoder144 layout already contains useful local components but has a weak
absolute origin.  The new placer evaluates all 576 cyclic `(row, column)` rolls
of that strict layout.  Each roll is scored from dirty-only SocketMatcher
outputs:

- real right/down log assignments decide which two horizontal/vertical seam
  sets should become the outer cuts;
- top/bottom/left/right dustbin probabilities score the four board borders;
- the existing component geometry is retained everywhere except at those two
  cyclic cuts.

The candidate never accepts a target/reference, never classifies monochrome
tiles as background, does not use a centre/face heuristic, and never warps,
duplicates, drops, or replaces a tile.  Shift `(0, 0)` is included and wins
ties, so the declared dirty-only objective cannot decrease.

Implementation:

- `src/aiijc_puzzle/socket_translation_placer.py`;
- `scripts/evaluate_socket_translation_placer.py`;
- `tests/test_socket_translation_placer.py`.

## Bounded development decision

On the already-open d64 exact `source16 × draw2` panel, decoder144 had 37
correct absolute tiles.  Border weights `0, 0.05, 0.2, 1.0` all stayed at 37;
the decoder weight `0.2` was too weak to change 31/32 boards.  A single stronger
weight `5.0` changed 12/32 boards and raised exact tiles `37 → 45`.  That weight
was frozen before selecting the confirmation panel; it was not swept again.

## Fresh exact-synthetic confirmation

The confirmation uses 24 new manifest-`train` clean sources and two independent
corruption/shuffle draws per source (48 boards).  Selection namespace is
`aiijc-socket-global-cyclic-translation-v1`, seed `20260902`, source digest
`12e3f7ac36dcee8d392b7d302e605e5bc562bdd8b550a76c6da31f7adaecfa5e`.

All 1056 d64 checkpoint-lineage filenames and 24 sources from prior exact
Socket reports were excluded before selection.  Dirty predictions were frozen
to NPZ/JSON and hashed before exact inverse-shuffle references were scored.

| Metric | decoder144 | + global cyclic border5 | Delta |
|---|---:|---:|---:|
| Correct absolute tiles, total | 40 / 27,648 | **58 / 27,648** | **+18 (+45%)** |
| Correct tiles / board | 0.8333 | **1.2083** | **+0.3750** |
| Direct placement | 0.1447% | **0.2098%** | **+0.0651 pp** |
| Correct rows / board | 23.2917 | 23.5000 | +0.2083 |
| Correct columns / board | 24.7708 | 25.2708 | +0.5000 |
| Translation-aligned / board | 13.8750 | 13.7917 | -0.0833 |
| Adjacency | 14.0436% | 13.9832% | -0.0604 pp |

The source-clustered bootstrap 95% interval for exact-tile gain is
`[+0.0411, +0.8750]` tile/board.  Per-case exact W/T/L is `8/38/2`.
The placer changed only 14/48 layouts.  The small adjacency decrease is
expected: a cyclic origin change preserves all interior geometry but replaces
the bonds crossing the selected global cuts.

This result is the first source-disjoint exact evidence here that a target-free
absolute-anchor operation improves direct placement rather than only adjacency
or translation-aligned placement.  It does not solve absolute placement: the
confirmed mean remains only 1.21 / 576 correct tiles.  It does establish that
the border head contains usable global-origin information when given enough
weight and when applied after, rather than during, component assembly.

## Decision

Keep weight `5.0` frozen and expose the placer as an opt-in tail after d64
decoder144.  Do not strengthen the rejected texture-centre/background prior
and do not reinterpret the +45% relative gain as a high absolute accuracy.
Before making it a final submission default, integrate it into the selected
full pipeline and rerun the strict permutation/compliance audit; no additional
weight tuning on either opened exact panel is allowed.

Reproduction:

```bash
.venv/bin/python scripts/evaluate_socket_translation_placer.py
.venv/bin/python -m pytest -q \
  tests/test_socket_translation_placer.py \
  tests/test_socket_decoder.py \
  tests/test_layout_evaluation.py
.venv/bin/ruff check \
  src/aiijc_puzzle/socket_translation_placer.py \
  scripts/evaluate_socket_translation_placer.py \
  tests/test_socket_translation_placer.py
```

Authoritative artifacts:

- `outputs/socket-matcher/global-cyclic-translation-v1-fresh-source24-draw2/report.json`;
- `outputs/socket-matcher/global-cyclic-translation-v1-fresh-source24-draw2/frozen_predictions.npz`;
- `outputs/socket-matcher/global-cyclic-translation-v1-fresh-source24-draw2/frozen_predictions.json`.

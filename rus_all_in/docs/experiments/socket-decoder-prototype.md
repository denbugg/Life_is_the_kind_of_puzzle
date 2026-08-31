# Socket OT → translation components → bounded QAP

## Status

Research decoder, **not promoted to production**.  It is implemented in
[`socket_decoder.py`](../../src/aiijc_puzzle/socket_decoder.py).  The SocketGlue
runner exposes it as a research evaluation arm; it does not change the model or
the production submission path.

The purpose of this arm is narrow: preserve more of SocketGlue's global
one-to-one structure than `solve_buddies(max_edges=96)`, while always returning
a legal 24×24 permutation and keeping runtime bounded.

## Decoder contract

Input is the pair of `(577, 577)` horizontal/vertical log assignments returned
by SocketGlue, including the aggregated dustbin row and column.  For each axis:

1. the dustbin is expanded to 24 interchangeable dummy sockets;
2. one Hungarian projection extracts exactly 552 real one-to-one edges, 24
   unmatched outgoing sockets, and 24 unmatched incoming sockets;
3. edges are ranked by two-sided log odds and inserted into one shared 2D
   relative-coordinate graph;
4. coordinate contradictions, tile collisions and components wider/taller than
   24 are rejected;
5. rigid components are shifted and packed using SocketGlue's four dustbin
   probabilities as a weak border unary;
6. remaining tiles are filled by a one-to-one Hungarian assignment;
7. at most 24 topology-guided two-tile swaps are accepted using an exact
   affected-edge QAP delta.  The reported energy therefore cannot decrease.

The result contains the strict `tile_at_position` permutation, its SHA-256, all
edge/cardinality/component counters, the initial/final objective, accepted swap
count and runtime.  Oracle and noisy unit tests cover exact cardinality, a case
where global matching rejects a mutual-top-1 distractor, full-grid recovery,
strict permutation, monotone polish, and input validation.

## Optional semantic/centrality anchor

`decode_socket_assignments(..., component_shift_unary=...)` accepts an optional
input-only `tile × slot` score matrix.  It is disabled by default because
`component_shift_unary_weight=0`.  When enabled, the decoder sums the score over
**all members of a rigid component** for every legal component shift and keeps
the same term in QAP polish.  This is the intended hook for a weak
high-texture/semantic-centre prior; no face or object category is hard-coded in
the decoder.

`texture_centrality_unary()` now provides a deterministic, dirty-input-only
implementation of that hook.  Its exact `texture-centrality-v1` formula is:

1. normalize RGB to `[0, 1]` and apply a spatial Gaussian with `sigma=1` for
   feature estimation only (submitted pixels are unchanged);
2. for tile `i`, compute
   `T_i = gradient_rms(luma) + 0.5 std(luma) + 0.25 chroma_structure`, where
   `chroma_structure = sqrt((var(R-G) + var(B-G)) / 2)`;
3. robustly standardize within the board as
   `z_i = (T_i - median(T)) / max(1.4826 MAD(T), 1e-6)` and use the positive-only
   gate `g_i = clip(z_i / 2, 0, 1)`;
4. build a smooth radial Gaussian slot field with normalized coordinates,
   `centre_sigma=0.55`, then zero-center and unit-scale it;
5. return `U[i, p] = g_i * centre[p]`.

Consequently a tile at or below median texture has an all-zero row.  It is not
labelled as a border tile and is not hard-placed anywhere.  Only above-median
structured tiles get a gradual central preference, and the decoder consumes it
as a whole-component sum.  In `run_socket_matcher.py`, the base
`socket_ot_decoder144` is always evaluated; the paired
`socket_ot_decoder144_texture_centre` arm exists only when
`--component-prior-weight > 0` (default `0.0`).

## Reused dev-8 sanity check

No new calibration or holdout records were opened.  The only empirical check
reused the eight already-exposed boards and the checkpoint from
`outputs/socket-matcher/pilot-train64-s100-r100-dev8-v1`.  References remain
target-assisted recovered permutations, not organizer labels.

| Decoder | Direct placement | Translation-aligned | Adjacency | Raw SSIM |
|---|---:|---:|---:|---:|
| Socket OT + buddies96 | 0.1519% | 0.8247% | 3.0005% | 0.099115 |
| Components 48/axis + QAP-24 | 0.1953% | 0.8030% | 2.6268% | 0.096572 |
| Components 144/axis + QAP-24 | **0.2387%** | 0.8247% | **3.5779%** | 0.095571 |
| Components 276/axis + QAP-24 | 0.1519% | **0.8681%** | 3.4760% | **0.097658** |
| Components 552/axis + QAP-24 | **0.2821%** | 0.8030% | 2.3777% | 0.095789 |

Interpretation:

- the decoder does extract additional exact-position signal from the same
  checkpoint: full edges add about `+0.1302 pp` direct placement versus Socket
  OT + buddies96, while the 144-edge arm adds both `+0.0868 pp` direct and
  `+0.5774 pp` adjacency;
- none of the arms improves raw SSIM, and dev-8 is far too small/noisy for
  promotion;
- the 144/axis arm is the sensible fixed candidate for the next
  source-disjoint decoder comparison; 552/axis demonstrates direct-position
  headroom but admits too many false constraints;
- this does not overturn the historical result that a decoder cannot rescue a
  weak matcher.  It establishes a runnable, sub-second-per-board conversion
  primitive for a materially stronger future SocketGlue checkpoint.

Mean decoder runtime in this check was 0.28–0.47 seconds per board (excluding
SocketGlue inference).  Every arm hit the 24-swap cap, so a later experiment may
compare 24 versus 48 steps, but only after local socket quality improves.

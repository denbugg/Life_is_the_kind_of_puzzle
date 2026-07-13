# Frozen LongSync-4 retrieval diagnostic

This job is a retrieval-only prerequisite for any new assembly experiment. It
does not build layouts, read assembly-gate targets, or create a submission.

## Frozen data and assets

- Development slice: `edge_development[316:324]`.
- Whole-source records: 8 sources x 2 panels = 16.
- Source-name hash contract: `sha256(("\n".join(names) + "\n").encode())`.
- Source-name hash: `c0f9548268a4e72a07e987cfdedf98047313e61967758401b297ff60f82ff7c7`.
- Panels: `primary_kornia`, `independent_libjpeg`.
- Frozen HGB, TileNAF denoiser, and HBT embedding hashes are enforced by both
  the evaluator and Kaggle wrapper.

The slice is disjoint from HGB fit/calibration and from the authoritative
validation/audit-derived assembly splits. No parameter or model is fit here.

## Frozen method

For each outgoing `(direction, source)` query, take HGB top-2. Canonicalize
each hypothesis to one undirected pair carrying a 2-D translation. Conflicting
hypotheses for the same unordered pair are represented only by the
deterministic highest-HGB occurrence. Run the sparse translation-group
LongSync-4 update for 10 iterations with `beta_t=min(2**t,20)`.

LongSync may only swap the original HGB top-1/top-2 score values when both
hypotheses own their canonical measurement, both have simple 4-cycle support,
and top-2 has strictly smaller final corruption. Unsupported/conflict-dropped
queries and every non-top-2 candidate remain byte-identical to HGB. There is no
mixing coefficient and no sweep.

## Frozen continuation gate

Both panels independently must satisfy every condition:

- mean delta AP >= `+0.005`;
- mean delta R1 >= `+0.010`;
- mean delta MRR >= `0`;
- mean delta R5 >= `0`;
- AP wins on at least `6/8` sources;
- R1 wins on at least `6/8` sources.

Failure permanently retires this LongSync-4 branch without opening an assembly
target. Passing only authorizes a separately frozen, source-disjoint actual
assembly Phase-A/Phase-B gate.

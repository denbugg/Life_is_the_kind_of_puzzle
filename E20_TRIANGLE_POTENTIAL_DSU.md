# E20: triangle-supported signed-potential DSU

E20 is the fixed-cost successor to the exact E18/E19 beam route. It is
authorized by the complete E19 relative-cap KILL report SHA256
`9a881793cbbfaa7f4da616e5a283d9f4cb4ad28a13e5605ff88aa05939bc3314`
and uses the same byte-pinned E12 clean-score scenes 10--17. It constructs no
absolute board and does not run residual completion, placement, neighbour,
SSIM or NLM.

The input graph is unchanged: exact CC192 rigid components and the positive
dense top-eight U/D/L/R cross-component claims from E18/E19. Every claim is
converted into the signed equation

`translation[v] - translation[u] = (dr, dc)`.

Component IDs are canonicalized so `u < v`. Claims with the same exact
`(u,v,dr,dc)` are one pose hypothesis. Physical seams are canonicalized and
deduplicated; a reverse top-eight observation of the same physical boundary
is recorded as reciprocity but is not counted as a second independent path.
The direct evidence fields are unique physical-seam count, reciprocal-seam
count, once-only neural sum and maximum neural score.

Triangle support is bounded before execution. For each component, retain its
top eight incident pose hypotheses by unique physical seams, reciprocal
seams, direct once-only neural sum, direct maximum, then the other component
and canonical hypothesis key, with no preliminary one-offset-per-component-
pair collapse. Through an intermediate component,
enumerate at most `8*7/2 = 28` pairs of incident legs. When the exact composed
signed offset equals an existing direct hypothesis between the two outer
components, the intermediate component is one triangle witness for that
hypothesis. Multiple leg pairs through the same intermediate count once, with
the best deterministic bottleneck evidence retained. A strong witness is one
whose two legs each contain at least two unique physical seams. Its bottleneck
is `min(leg1.direct_sum, leg2.direct_sum)`; competing paths through one
intermediate choose maximum bottleneck, then maximum minimum direct maximum,
then canonical leg IDs.

A hypothesis may merge two pose trees only when it has at least two
independent support paths:

`unique_physical_seams + distinct_triangle_intermediates >= 2`.

Reciprocity is useful rank evidence but never increases that independent-path
count. This prevents the forward and reverse records of one physical seam
from satisfying the gate by themselves.

Eligible hypotheses are processed once in this immutable order, evidence
descending and IDs/offset ascending:

1. independent support paths;
2. distinct triangle intermediates;
3. strong triangle intermediates;
4. unique physical seams;
5. reciprocal physical seams;
6. triangle bottleneck neural sum;
7. direct once-only neural sum;
8. direct maximum neural score;
9. `(u,v,dr,dc)`.

The signed-potential DSU stores `translation[node] - translation[parent]`.
For different roots, one hypothesis uniquely aligns the two aggregates. A
merge is accepted only if all supporting physical seams become the claimed
cardinal contacts, tile coordinates do not collide, and the merged height and
width are each at most 24. For an already-connected pair, an exactly implied
offset is accepted as cycle evidence; a different offset is a pose conflict.
There is no rollback, alternative state, beam, reranking or parameter sweep.
Weak hypotheses that fail the independent-path rule are diagnostic only and
cannot merge clusters or add selected cycle/seam evidence.

Every final cluster is normalized by subtracting its minimum occupied row and
column. Exactly one cluster is selected without labels by rigid-tile count,
component-cycle rank, accepted physical-seam count, accepted once-only neural
sum, minimum tile ID, and finally canonical translations. The output remains
sparse: component translations, relative tile entries, bbox/legal-origin
algebra, accepted tree/cycle hypothesis IDs and physical seams. No 24x24 board
object is constructed.

Only after this selection may the evaluator use the known permutation. For
the selected sparse cluster, the modal translation bin of
`truth_coordinate - relative_coordinate` defines exact posed tiles. Exact
relative-pose precision is modal tiles divided by selected tiles; exact pose
coverage is modal tiles divided by 576; tied modal bins are resolved by the
lexicographically smallest signed offset. A pose relation is true only when both
whole CC192 components have exact truth translations and their difference is
the selected signed offset. Empty relation/seam sets have precision zero.

All inclusive PASS checks are required:

- completed invariant-clean scenes and legal-origin scenes: `8/8`;
- rigid coverage mean/worst: at least `0.35/0.25`;
- exact pose coverage mean/worst: at least `0.30/0.20`;
- exact relative-pose precision mean/worst: at least `0.90/0.80`;
- accepted relation truth precision mean/worst: at least `0.85/0.70`;
- accepted physical cross-seam precision mean/worst: at least `0.85/0.70`;
- mean component-cycle-rank ratio: at least `0.05`.

PASS alone opens a separately frozen E21 that evaluates legal absolute origins
for this one sparse cluster and materializes one board/residual completion.
FAIL closes this exact top-eight triangle-potential route without weakening
support, changing thresholds or adding a beam.

All execution is CPU-only and all reports/temp stay on `E:`. The report is
`E:/pazzle_work/triangle_pose_e20/cc192_triangle_potential_viability_v1.json`.

Planned files:

- `src/e20_triangle_potential_dsu.py`;
- `src/eval_e20_triangle_potential_viability.py`;
- `tests/test_e20_triangle_potential_dsu.py`;
- `tests/test_e20_triangle_potential_viability.py`.

## Frozen result

`KILL`: the complete eight-scene run produced legal sparse clusters but failed
every quality check. Mean rigid coverage was `0.13671875`, mean exact pose
coverage `0.03602431`, mean exact relative-pose precision `0.26367252`, mean
accepted-relation precision `0.13252388`, mean physical-seam precision
`0.23226369`, and selected cycle-rank ratio `0.0`. No board or downstream
solver/restoration metric was opened. Report SHA256:
`4538e35825bdfae86aa7bda252d7a7a5aa2b8e933ffc6deaab74ebade8f557be`.

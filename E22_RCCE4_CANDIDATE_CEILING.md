# E22: RCCE-4 full-union all-emitter candidate ceiling

E22 is a CPU-only, label-after-core discovery ceiling for one redesigned
production candidate-generator module. It is authorized by the exact E21 KILL
artifact SHA256
`0c43099860c7a16f5e968a8ea6cf637293cd639d9b86e342797ef68c5d53e724`
and stage `kill_raw_CC96_anchor_top8_candidate_pool`. E21 proved that a learned
verifier cannot recover missing relations from the nontrivial-only directional
top-eight pool. E22 therefore asks whether the complete affinity support that
Rank96 already computes can supply a sufficiently connected oracle graph.

The core signature accepts exactly two label-free arrays:

- `candidate_ids`: contiguous `int64[576,128]` from the frozen ordered,
  deduplicated K64+K64 affinity union;
- `raw_logits`: contiguous `float32[4,576,128]` in literal `U,D,L,R` order.

The finite mask must be identical in all four directions, every valid row must
be nonempty, and valid candidates must be unique, non-self IDs in `0..575`.
The core derives the exact frozen CPU-float32 Rank96 dense matrices and exact
raw CC96 partition (`max_edges=96`, `min_margin=0`) internally. The partition
includes every residual singleton. No components, labels, pixels, permutation,
prebuilt dense matrices or target data enter the core.

## RCCE-4 lift

All 576 tiles are emitters. Directed candidate membership is used only to form
one canonical unordered affinity pair `a<b`: the pair exists when either
`a -> b` or `b -> a` is a valid affinity-union membership. Each pair emits
exactly these four canonical physical seam claims, in this exact order:

1. `(a,b,R)` — `b` is right of `a`;
2. `(b,a,R)` — `a` is right of `b`;
3. `(a,b,D)` — `b` is below `a`;
4. `(b,a,D)` — `a` is below `b`.

These four claims are **upright adjacency orderings**, not four tile
orientations. Every tile keeps its original `0 degree` orientation; rotations
by 90/180/270 degrees and every reflection are forbidden throughout the core
and evaluator.

Their raw-logit metadata is literal and never averaged for admission:

- `(a,b,R)`: `RIGHT[a,b]` and, when reverse membership exists, `LEFT[b,a]`;
- `(b,a,R)`: `RIGHT[b,a]` and, when reverse membership exists, `LEFT[a,b]`;
- `(a,b,D)`: `DOWN[a,b]` and, when reverse membership exists, `UP[b,a]`;
- `(b,a,D)`: `DOWN[b,a]` and, when reverse membership exists, `UP[a,b]`.

Missing directed membership is explicit metadata. The raw values are listwise
row logits and are not comparable across rows: every finite slot is preserved
separately, and logits are never averaged, summed or ranked for RCCE-4
admission. Same-component seam claims
are removed because their relative geometry is already fixed by CC96. Every
cross-component claim induces one exact signed canonical relation
`(u<v,v,dr,dc)` from component-local coordinates. Alternative offsets remain
distinct. Physical seams are unique; reverse observations are metadata, not
new seams or relations.

The only relation filter is exact label-free pair geometry: all supporting seam
endpoints must be adjacent under the induced translation, the two component
coordinate sets must not collide, and their union bounding box must be at most
24 by 24. Incidental component contacts neither add evidence nor cause
rejection. A whole-pure ground-truth relation must survive this filter; E22
reports and requires exact survival after labels are revealed.

The frozen structural bounds are theoretical, never truncation controls:

- directed valid memberships `<= 576*128 = 73728`;
- unordered affinity pairs `<= 73728`;
- stored finite directional logit observations `<= 4*73728 = 294912`;
- RCCE-4 claims `== 4 * unordered_pairs <= 294912`;
- geometry-valid grouped hypotheses `<= 294912`.

Exceeding any bound is a contract failure. E22 adds no truncation to the
already-frozen upstream ordered dual-affinity K64+K64 union (storage width at
most 128). There is no E22 score threshold, rank cutoff, learned shortlist,
triangle rule, iterative growth or sweep.

## Label-only ceiling and decision

Only after the core returns, a component is whole-pure when every member has
one common `truth_coordinate - local_coordinate` translation. A hypothesis is
true only when both whole components are pure and its signed translation is
exact. All true hypotheses are unioned once in canonical order with an
independent signed-potential DSU; pure components with no relation remain
singleton clusters. Collision and 24x24 span are independently revalidated,
the largest exact cluster is selected deterministically, and legal origins are
counted analytically. No absolute board is constructed.

The primary contact denominator is frozen independently of candidate output:
every canonical undirected ground-truth horizontal/vertical seam whose endpoints
belong to two distinct whole-pure CC96 components. It must be positive on every
scene. A hit means that unordered tile pair exists in the affinity-pair OR.
E22 also reports the same recall over every cross-component true seam regardless
of component purity. Post-filter eligible-true survival is the fraction of hit
eligible seams whose exact RCCE-4 hypothesis survives pair geometry; it must be
exactly one.

The all-inclusive PASS rule is:

- completed invariant-clean scenes: `8/8`;
- emitter tiles: exactly `576` on every scene;
- all theoretical membership/pair/claim/hypothesis bounds pass on `8/8`;
- at least one true hypothesis and at least one legal selected origin: `8/8`;
- eligible denominator positive and post-filter eligible-true survival `1.0`:
  `8/8`;
- eligible pure cross-component contact recall mean/worst at least `0.90/0.80`;
- largest exact-connected tile coverage mean/worst at least `0.30/0.20`;
- selected component-cycle-rank ratio mean/worst at least `0.05/0.01`.

PASS proves candidate availability only and opens a separately frozen E23
source-group-disjoint confirmation of the identical generator. It does not
authorize GPU training. Any later learned shortlist needs its own retention
ceiling before packing. FAIL closes this exact existing-affinity full-union
generator without a K, threshold, cap, component-budget or filter resweep.

Excluded: clean scores or pixels, labels inside the core, board/residual,
placement, neighbour, SSIM, NLM, absolute-origin choice, rotation, reflection,
GPU, diffusion and target/test submission data. The fixed discovery scenes are
the already-open byte-pinned raw E12 IDs `10..17`. All report/temp output stays
on `E:` at
`E:/pazzle_work/posegraph_e22/cc96_all_emitter_full_union_candidate_ceiling_v1.json`.

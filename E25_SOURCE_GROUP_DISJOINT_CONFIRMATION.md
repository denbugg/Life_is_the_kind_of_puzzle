# E25: frozen source-group-disjoint CRS-v1 confirmation

## Status and authority

E25 is the one-shot confirmation of the exact E24 CRS-v1 recipe on 48
previously sealed, source-group-disjoint validation scenes.  This document,
the ordered scene identities, all scientific gates, resource limits, process
boundaries and the scene-226 runtime canary are frozen before any E25 pixel,
corrupted tile, raw/spatial logit, feature, prediction, permutation, target,
label, board or metric is opened or created.

E25 is neither a development set nor a sweep.  No E25 result may select or
change a feature, checkpoint, threshold, cap, decoder, packer, restoration
setting, corruption replay, seed or gate.  A failed gate closes exact CRS-v1;
no diagnostic can rescue it.

Real E25 access requires one authenticated upstream chain:

1. the audited E24 generation-3 preflight ledger and run contract;
2. the exact E24 structural PASS plus its authenticated orchestration receipt;
3. the exact staged board/SSIM/NLM PASS report, recomputed by its own validator;
4. the single final-all-eight model (`final/model.txt`) and
   `final/final_all8_manifest.json`, authenticated by their owner validator;
5. this E25 source/protocol seal, created only after 1--4 pass.

An absent, malformed, non-canonical, hash-mismatched or failing member is a
hard stop before E25 data.  A structural report, staged report or model file
alone has no authority.

## Sealed manifest

The ordered validation-relative IDs are:

`226,262,242,123,103,231,286,296,230,134,118,110,239,269,146,187,183,151,148,247,191,186,193,106,220,274,125,117,115,265,165,257,210,213,132,143,152,137,177,225,113,259,101,178,202,141,273,111`.

Each ID `i` maps only to the manifest identity
`img_{6700+i:06d}.png`.  The SHA256 of these 48 names, in this order with a
newline after every name, is
`407a6326ceeec2e8cc78106b74c2f10c46a55143ea488a30f7bac66e2b373caa`.
The SHA256 of canonical ASCII JSON for the ordered list of exact
`{name,source_group,target_sha256}` records (sorted keys, separators
`(',',':')`, no final newline) is
`76e6b9431de41388e4aebef525ff4a5fd8354f789cf0a5913c1e29d8db148e2e`.

The metadata source is the exact manifest
`E:/pazzle_work/rank96_e11_v4/source_groups_v4.json`, SHA256
`fa142c5f9c4fa17671b60d72b9acedff0eafcad4e77afac2b17a9649adfbfbd9`.
The E25 broker must authenticate that file before parsing it, reproduce both
hashes above, prove 48 distinct E25 source groups, and prove no source-group
overlap with the 6,700 training images, validation-relative IDs `0..99`, or
the E24 scenes `10..17`.  This operation reads metadata only, never a target.

## Fixed model, inputs and orientation

The final E24 all-eight checkpoint is used unchanged.  Each E25 scene receives
the exact CRS-v1 227-feature extractor, canonical component-pair offset-plus-
NONE queries, LambdaRank checkpoint, positive-margin selector, cap
`min(survivors,2*(C-1))`, rollback-safe signed-potential DSU, raw-R/D
`solve_components_from_scores` packer with `repair_passes=0`, assembly of the
original corrupted tiles, and NLM10.  RR96 is the unchanged paired baseline.

Tiles are upright.  Orientation is exactly zero and reflection is false; no
rotation or reflection path exists.

## Enforced process boundary

E25 runs in separate, freshly invoked capabilities on `E:`:

1. **Manifest/source broker.**  It sees only the pinned metadata manifest and
   emits the canonical 48-record seal.
2. **Label-free feature/inference workers.**  They may receive only the exact
   corrupted upright tiles, candidate IDs, raw U/D/L/R scores, authenticated
   spatial scores and final E24 checkpoint required by frozen CRS-v1.  They
   cannot open a permutation, target, clean pixel or metric.  Each scene is an
   atomic feature/prediction commit.
3. **Global 48/48 label-free barrier.**  It authenticates every ordered commit,
   exact final checkpoint and resource receipt before label access.  A partial
   barrier cannot open the metric broker.
4. **Trusted structural-label broker.**  Only after the global label-free
   barrier, a separately invoked broker may open the exact permutation for the
   same 48 identities and atomically publish the structural report.  It cannot
   open a clean target.
5. **Structural PASS barrier and staged image broker.**  Only after the exact
   structural report recomputes to PASS may a new process open the exact
   permutation and clean target and evaluate the already-frozen boards against
   RR96.  It may not train, rescore, choose an artifact or write back into the
   label-free cache.

The initial runtime canary is scene `226`, chosen solely because it is the
first sealed manifest ID; no pixel, candidate-count or metric informed the
choice.  A canary PASS permits the remaining 47 scenes but exposes no label or
metric.

The current runner intentionally implements only metadata/authority sealing
and data-free contract validation.  Until the concrete lineage, label-free
worker and trusted metric-broker adapters receive a separate review, all real
E25 feature, inference and metric modes fail before opening an E25 member.

## Structural confirmation gates

All checks are required over exactly 48 ordered scenes:

- provenance, input, orientation, query canonicality, finite output, DSU
  algebra and legal-origin checks pass on `48/48`;
- proposal and accepted-relation denominators are nonempty on `48/48`;
- proposed relation precision mean/worst is at least `0.70/0.60`;
- true-relation recall mean/worst is at least `0.65/0.50`;
- exact-connected tile coverage mean/worst is at least `0.50/0.35`;
- mean accepted-graph cycle-rank ratio is at least `0.05`; and
- every scene has at most `450,000` geometry-valid hypotheses.

The definitions are identical to E24, including recall relations constructed
before candidate presence and inclusive numeric comparisons.  A zero required
denominator, missing scene, duplicate scene, NaN or infinity fails.

## Staged paired confirmation gates

Only after structural PASS may the trusted broker evaluate the exact candidate
against exact RR96.  All gates are inclusive except that a win itself means a
strictly positive per-scene final-SSIM delta:

- mean solve-only SSIM delta at least `+0.003`;
- mean final SSIM delta at least `+0.002`;
- strict positive final-SSIM wins on at least `30/48` scenes;
- worst per-scene final-SSIM delta at least `-0.020`; and
- mean neighbour-accuracy delta at least `+0.005`.

The `30/48` count is the exact scale-up `ceil(48*5/8)` of the frozen E24
`5/8` rule.  Bootstrap intervals, placement, right/down accuracy, loss counts
and worst-scene diagnostics are report-only.

## Resource and storage envelope

All generated E25 artifacts, temporary files, logs and bytecode live under
`E:/pazzle_work/posegraph_e25_confirmation/`.  Import and `smoke` create no
directory.  `TEMP`, `TMP`, `TMPDIR`, `JOBLIB_TEMP_FOLDER`,
`LIGHTGBM_TMPDIR` and `PYTHONPYCACHEPREFIX` must resolve inside that root before
a target worker starts.  No generated scientific artifact is written on `C:`.

Hard caps are:

- label-free feature cache at most `24 GiB`;
- all E25 artifacts and temporary payloads at most `48 GiB`;
- peak resident RAM at most `16 GiB`; and
- total E25 CPU time at most `48 hours`.

Exceeding a cap fails E25 and never authorizes truncation, sampling or recipe
changes.  E25 PASS is required production evidence, but submission generation
still requires the separately authenticated production-parity scene-17 replay
and production runner authority.

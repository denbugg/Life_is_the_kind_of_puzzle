# S1 historical anchor recovery

## Verdict

The exact externally scored S1 (`0.23748525732559034`) is **not runnable from
the downloaded repositories alone**. The production code and three rank96
checkpoint hashes survive, but all four weights and the S1 ZIP/manifest are
absent. In particular, Git never recorded the SHA-256 of
`r5_capacity_fp32.pt`, so a same-named retrain cannot be presented as the
platform-scored R5.

This recovery adds a runnable, input-only replay of the exact post-layout
`R5 -> NLM` arm plus a fail-closed artifact audit. It does not use targets,
clean tile references, source overrides, or test labels.

## Authoritative history

| Item | Evidence |
|---|---|
| S1 runner introduced | commit `c4d2bb2` |
| canonical test-directory fix / exact runner | commit `3c2f0b8` |
| official platform score recorded | commit `d7826aa` |
| branch | `origin/autoresearch/pazzle-fixed-orientation-cb1` |
| archived pre-S1 rank96 sources | commit `d281361` |

The exact production path is:

```text
upright 20x20 tiles
 -> affinity R1 top-64 union affinity R3 top-64
 -> rank_v2w64 raw listwise logits
 -> CPU-float32 dense_rd
 -> corrected buddies(max_edges=96, min_margin=0, repair_passes=0)
 -> raw 480x480 upright layout
 -> RestoreNet(base=32, depth=4), FP32, uint8 round-to-nearest
 -> cv2.fastNlMeansDenoisingColored(h=10, hColor=10, template=7, search=21)
```

The historical `fixed_nlm` passed the repository RGB ndarray directly to the
OpenCV BGR-oriented function. The port intentionally preserves that byte-level
call instead of silently inserting channel swaps.

## Missing artifact proof

Required files and the integrity information retained by Git:

| Role | Historical filename | Expected SHA-256 |
|---|---|---|
| ranker | `rank_v2w64_best.pt` | `42685373b1a450a4cb3d7a9b22370dfcfaa2335e9e8ada609f21b7cc64abbfbc` |
| affinity primary | `affinity_r1_1200_best.pt` | `708565329c7661a965215d98e85f462a90930071f36a0f75b4813c0c5797ec4f` |
| affinity secondary | `affinity_r3_1000_best.pt` | `0fceafdb110bde59149fe1ad1e800a69d116041bc627af369aaecd60be53b6c8` |
| R5 | `r5_capacity_fp32.pt` | **not recorded** |

A filename search over `/Users/rusyalain` found none of these checkpoints,
`submission_rank96_r5nlm_s1.zip`, `s1_manifest.json`, or the original
`source_disjoint_split_v1.json`. The research branch itself contains no
`.pt/.pth/.npz/.zip` production artifacts. The reports point only to the
former Windows paths under `E:\pazzle_work\...`.

Run the repeatable local audit:

```bash
uv run python scripts/run_s1_anchor.py --audit-only
```

If the original files are recovered, place them below `artifacts/s1_anchor/`.
Supply the R5 digest recovered from the original `s1_manifest.json` via
`--r5-sha256`; only then can all four artifacts be called exact.

## Runnable recovered component

`src/aiijc_puzzle/s1_anchor.py` ports the exact R5 architecture, uint8
conversion, historical NLM call, upright board convention, checkpoint audit,
and deterministic ZIP tail. `scripts/run_s1_anchor.py` accepts one input-only
`<image-stem>.npy` slot-to-input permutation per image:

```bash
uv run python scripts/run_s1_anchor.py \
  --boards-dir outputs/rank96/boards \
  --r5-checkpoint artifacts/s1_anchor/r5_capacity_fp32.pt \
  --limit 1 --no-zip
```

This is a tail replay, not a claimed reconstruction of rank96. It becomes the
full historical S1 only when the supplied boards come from the exact pinned
rank96 weights and contract.

## RGB/BGR production discrepancy

The historical composition evaluator and production runner did not apply the
same coloured-NLM channel contract:

- `eval_r5_nlm_composition.py` explicitly converted RGB -> BGR -> RGB;
- production `infer_rank96.fixed_nlm`, called by S1, passed repository RGB
  directly to OpenCV's BGR-oriented API.

This was measured on the already-established frozen holdout-48 panel using the
same target-assisted layouts as the pixel-tail bakeoff. It is a pixel-only
diagnostic, **not** an inference-only solver score and not a fresh holdout:

| Tail on unchanged layout | Mean SSIM |
|---|---:|
| historical direct-array NLM h10 | 0.563769 |
| correct RGB/BGR NLM h10 | **0.565243** |
| correct RGB/BGR NLM h9 | 0.557442 |

Correct channel handling at h10 improved the historical call by `+0.001474`,
95% CI `[+0.000085,+0.002863]`, with 37/48 wins. Conversely, h9 lost
`-0.006327` against historical h10. Thus the previously promoted h9 result
must not be transplanted onto S1 without an R5-layout paired gate. The only
supported follow-up is a small h10 channel-fix ablation after recovering an
input-only layout/R5 path. Full rows are generated at
`outputs/s1-nlm-channel-audit/holdout48.json` by:

```bash
uv run python scripts/run_s1_nlm_channel_audit.py --split holdout --limit 48
```

There is currently no honest inference-only validation SSIM for this port:
without the missing rank96 weights it cannot infer boards, and without the R5
checkpoint it cannot reproduce the production pixels.

## Minimum honest retrain path

Exact byte reproduction requires retrieving the four original files. If that
is impossible, the closest retrain must be labelled **S1-retrained**, and is a
new candidate rather than evidence for the historical `0.237485` score:

1. Check out/extract source at `3c2f0b8` without modifying the archive repo.
2. Train two `MacroAffinityNet` models on exact synthetic distortions:
   radius 1 for 1,200 steps (`affinity_r1_1200`) and radius 3 for 1,000 steps
   (`affinity_r3_1000`). The surviving code is `src/train_macro_affinity.py`.
3. Train `CandidateSeamRanker(width=64, candidate_k=64)` over the ordered union
   using `src/train_candidate_rank.py`; the historical best was step 1,600 and
   training plateaued by 2,800.
4. Train `RestoreNet(base=32, depth=4)` in FP32 with MS-SSIM + L1 using
   `src/train_r5_restore_unet.py`. The old checkpoint was a two-FIT-scene
   capacity model; because its source-disjoint manifest is missing, use the
   current frozen train split and record the chosen filenames/digest rather
   than inventing the old split.
5. Validate all four retrained hashes and compare end-to-end on current
   calibration/holdout panels. Never use clean references or recovered target
   mappings at inference, and never compare a target-assisted layout score to
   the leaderboard anchor.

Retraining all four models is substantial and cannot recreate the exact
platform artifact because the original random split and R5 bytes are missing.
Artifact recovery is therefore the shortest and highest-fidelity route.

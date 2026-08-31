# DRUNet goal cycle 2 and fixed sigma50 broad measurement

## Outcome

The formal 16-board train reproduction rejected the post-h28 tilewise DRUNet30 arm and its half blend because they introduced severe 20×20 grid and near-flat-tile artifacts. The simpler `B_drunet50_protected_h28_h50_t60` arm improved the frozen DRUNet40 stack on all 16 records and had much safer target-free diagnostics. It is therefore the only method carried into a separate fixed-candidate all-700 protocol.

The broad protocol was frozen before producing any of its calibration predictions or decoding any calibration target. It subsequently passed both exact, unchanged 700-board stages: calibration mean SSIM `0.2706107975` and holdout mean SSIM `0.2753562560`. Every frozen provenance, structural, flatness, and absolute-score gate passed. This is a measurement result only; production and competition-test outputs were not changed.

## Scope and exposure

- Train roster: ranked train offset 528, count 16; selection first 8 and verification last 8.
- The train records had already been decoded in prior diagnostics. This run is a formal reproduction, not fresh discovery.
- The all-700 calibration and holdout rosters are also historically exposed in the workspace. No freshness or source-family independence is claimed.
- Every measurement stage is target-blind during rendering, freezes all 700 prediction records and a content-addressed commitment, and writes an external receipt before it may decode a target.
- Competition-test and production/submission outputs are out of scope.

## Corrected train protocol

The initial v1 implementation mistakenly fed the frozen original-h28 image into the post-DRUNet30 arm. It was stopped before commitment or target access. Its 48 partial prediction PNGs are retained read-only under `outputs/drunet-goal-cycle2/aborted-precommit-v1-original-h28-no-target-access/`, with `ABORTED_NO_TARGET_ACCESS.json`. The superseded preregistration remains at `configs/drunet_goal_cycle2_train_preregistered_v1_SUPERSEDED_BEFORE_TARGET_ACCESS.json`.

The corrected v2 roster was fixed as:

- A: frozen DRUNet40 → independent h20/h28/h40 → t40 protected blend.
- B: tilewise DRUNet50 → one canvas → independent h20/h28/h50 → t60 protected blend.
- C: the same DRUNet50-h28 tiles → tilewise DRUNet30 → fixed alpha 0.5 around that h28 image.
- D: exact uint8 half-up blend `0.5 * B + 0.5 * C`.

All methods use the same dirty-only bilateral buddies96 layout, strict raw 576-tile bijection audit, and frozen RGB/luminance harmonization. Neural inference is upright, per 20×20 tile, same-board only, with no resize, rotation, substitution, template, target, or cross-board pixels.

## Train score

| Arm | Selection mean | Delta vs A | Wins | Verification mean | Delta vs A | Wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A current D | 0.299736 | — | — | 0.285445 | — | — |
| B sigma50 protected | 0.302408 | +0.002672 | 8/8 | 0.287571 | +0.002127 | 8/8 |
| C post-h28 DRUNet30 | 0.301475 | +0.001739 | 7/8 | 0.288486 | +0.003041 | 8/8 |
| D half B / half C | 0.303182 | +0.003446 | 8/8 | 0.289373 | +0.003928 | 8/8 |

D beat B on the selection half by `+0.000774` with 6/8 wins, so the immutable selector chose D. The unchanged preservation gate then rejected it. Calibration access under the cycle2 protocol remained forbidden.

## Preservation and manual review

| Arm vs current D | Luma mean/min | Chroma mean/min | Laplacian mean/min | Grid mean/max | Near-flat std<2 delta mean/max |
| --- | --- | --- | --- | --- | --- |
| B | 0.9781 / 0.9621 | 0.9795 / 0.9594 | 1.0128 / 0.9987 | 1.0092 / 1.0162 | +0.875 / +5 |
| C | 0.8661 / 0.8450 | 0.8675 / 0.8226 | 1.0445 / 0.9688 | 1.8940 / 2.5028 | +20.875 / +37 |
| D | 0.9102 / 0.8979 | 0.9913 / 0.9427 | 0.9770 / 0.9034 | 1.4433 / 1.7441 | +10.5625 / +23 |

Exact spatially constant tiles did not increase for B, C, or D. That alone was insufficient: C and D produced many near-flat tiles and visible tile-grid blocking. Manual inspection of `img_003017`, the worst D grid-ratio record, showed obvious 20×20 blocking in C and D while B remained visually close to A. `img_006760` was the worst B near-flat increase at +5 tiles; it did not show the severe C/D failure mode.

The cycle2 gate is not weakened or reinterpreted. C and D are rejected exactly. B failed only the deliberately stronger relative-to-current-D conditions requiring mean luma and chroma retention of at least 1.0; its diagnostics remain inside the older established h28-safe envelope.

## Fixed-B broad protocol

The separate broad experiment contains no roster or selector. Its only scored candidate is exactly cycle2 B:

1. Infer bilateral buddies96 from the dirty board.
2. Audit all 576 original upright 20×20 tiles in a one-to-one raw permutation.
3. Apply frozen RGB offsets and luminance gains.
4. Apply official MIT-licensed KAIR colour DRUNet at sigma 50 independently to each tile, with same-tile reflect padding and exact 20×20 crop.
5. Assemble those 576 restored identities once.
6. Compute independent single-pass colored NLM h20, h28, and h50 images.
7. Derive the fixed t60 softened protected mask from h20 and blend h28-safe with h50-flat pixels.

The target-free safety reference is the same DRUNet50 canvas followed by independent h28, avoiding a confounded comparison to DRUNet40. The preregistered gate uses the older edge-protected-v2 thresholds: luma mean/min `0.80/0.70`, chroma `0.80/0.70`, Laplacian `0.72/0.60`, grid mean/max `1.05/1.12`, protected fraction mean `[0.40, 0.75]` and every board `[0.30, 0.85]`, and clipping increase at most `0.01`. It additionally requires no increase in total exact-constant tiles and near-flat std<2 increases no greater than `+2` mean and `+6` on any board.

Calibration promotion requires candidate mean SSIM in the closed interval `[0.27, 0.28]`, all 700 provenance audits, and the entire target-free safety gate. Only then may exact unchanged holdout700 be prepared, committed, and scored. Otherwise holdout remains sealed.

## Broad results

Both stages rendered and committed all 700 candidate predictions before any target for that stage was decoded. Calibration passed first, thereby authorizing the unchanged holdout stage.

| Stage | Mean SSIM | Absolute gate | Luma mean/min | Chroma mean/min | Laplacian mean/min | Grid mean/max |
| --- | ---: | --- | --- | --- | --- | --- |
| Calibration700 | 0.2706107975 | PASS | 0.99182 / 0.96388 | 1.03098 / 0.99510 | 1.05243 / 1.00407 | 0.99731 / 1.03103 |
| Holdout700 | 0.2753562560 | PASS | 0.99277 / 0.95911 | 1.03257 / 0.99610 | 1.05126 / 0.98315 | 0.99621 / 1.03595 |

The calibration protected-mask fraction was `0.47009` on average with per-board range `0.35028` to `0.66861`; holdout was `0.46875`, range `0.35030` to `0.60525`. Maximum absolute RGB mean shifts were `0.75117` and `0.69362`, respectively, and clipping increase was zero on both stages. All 700 raw permutation/provenance audits passed in each stage and all 700 candidate pixel hashes were distinct.

Exact spatially constant tiles changed from 30 to 29 on calibration and remained 14 to 14 on holdout. Near-flat std<2 counts changed from a per-board mean/max of `20.4986/198` to `19.8029/198` on calibration (delta mean/max `-0.6957/+3`) and from `23.1729/194` to `22.2986/189` on holdout (delta mean/max `-0.8743/+2`). These are inside the frozen `+2/+6` limits.

Target-free review sheets were generated from dirty inputs and committed candidate pixels after each prediction commitment but before scoring: seven lexically sorted 100-board pages covering every record exactly once, plus a high-resolution worst-24 page selected by frozen source-only safety metrics. The holdout manifest and all eight PNG hashes were independently rechecked after scoring. An independent visual pass over all seven holdout pages and the worst-24 page found zero severe restoration-induced artifacts, zero discernible face-, text-, or object-specific destruction, zero new halo/ringing cases, zero systematic restoration-induced 20×20 grid failures, and zero visual indicators of constant/template substitution. A direct same-layout h28-reference versus candidate comparison for the worst 24 supported the attribution: visible blur and tile discontinuities were already present in the reference rather than introduced by B. This is an independent engineering verdict, not the formal root manual-review JSON, and it makes no claim about hidden layout correctness.

## Immutable bindings

- Corrected train preregistration: `configs/drunet_goal_cycle2_train_preregistered_v2.json`, SHA-256 `b163d1060c2ce8a88890a7f671971f2603564d360bdf7f88940968d0665a00db`.
- Train commitment: `outputs/drunet-goal-cycle2-v2/train-offset528-count16/prediction-commitment.json`, SHA-256 `d8eb8314a6a281ad62371bd884267247376fa3d2ee09b71840742dfec4177120`.
- External train receipt: `outputs/drunet-goal-cycle2-v2/train-offset528-count16.commitment-receipt.json`, SHA-256 `4ccb02425cd9538606e54d17cc08127af4a3a9eaff37657a68d61a73ba5ddbc1`.
- Train selection decision: `outputs/drunet-goal-cycle2-v2/train-offset528-count16/selection-decision.json`, SHA-256 `5b26f02878cb502b5e4650d3a87cce20d56c8cd87d5e7e3f6c7d88fa85ceb866`.
- Train report: `outputs/drunet-goal-cycle2-v2/train-offset528-count16/report.json`, SHA-256 `9898716bceb11892c0f7955ceebfeced573b7e9b9ae4bdda76781cb9e78d8f99`.
- Fixed-B all700 preregistration: `configs/drunet_sigma50_protected_all700_measurement_v1.json`, SHA-256 `a402fc682b0db96b60004fa2c33ea70baf06035cb4971b8ee0778ceb1b7f05ac`.
- Fixed-B runner: `scripts/run_drunet_sigma50_protected_all700.py`, SHA-256 `64563820beec56fccf00fc11f899411c175274d46c50a124154ca37bd857bde6`.
- Fixed-B renderer: `src/aiijc_puzzle/drunet_sigma50_protected_broad.py`, SHA-256 `b0b73fb30787394f839a8449a7e7c898e92269a65cef613b5152c90b53db5a9f`.
- Target-free manual-sheet generator: `scripts/generate_drunet_sigma50_manual_sheets_v1.py`, SHA-256 `50aaa9dab190b8a6952651548d5fc0613e6dcd643daa95fcfc9eea39fdfc5879`.
- Calibration commitment / external receipt: SHA-256 `13a61821d616ad01eee79f45db51dc223d1c041fe502ff84525e6259e623853c` / `4c5f689efc26a75348e218704fb32de8594e40b893457a2d07e6b7c6398fa0ab`.
- Calibration report: `outputs/drunet-sigma50-protected/all700-measurement-v1/calibration700/report.json`, SHA-256 `06934d5a2450ef3752a88c0d9f8ab90b17b5f0a3d859ef8cc67dbb4da3392590`.
- Calibration target-free sheet manifest / report binding: SHA-256 `d07fbe1f70172a0f06544b7847d36cabd752325dfea3c704fb5a82b7f4f5d468` / `b261928857a68a457dcdadafe472ce7d21ae438ff5b43ba8781c7e0c1fc4a40d`.
- Holdout commitment / external receipt: SHA-256 `6df1f7b6d8d318a6dce6d81d6c35890cdb05e1c6481c1351b92d2393341298f2` / `94bb83b4e1f8ada29e05ee378cd890c849e4ab070b1266dba9c52b45069edf4f`.
- Holdout report: `outputs/drunet-sigma50-protected/all700-measurement-v1/holdout700/report.json`, SHA-256 `05a4f2fab4d0dd07624e5a54c143004302cf4df78a15460388775fa0b1013d07`.
- Holdout target-free sheet manifest / report binding: SHA-256 `724ff6667e46d7c54a4066eb5f89e7a7427b1e9973e4c56f4276ffc9ff30a372` / `d73500398b262b1059c45ea209be34da0f4ebee76adff2647f1d5c70d11b67bf`.

## Verification

The focused tests cover the one-call tilewise sigma50 contract, the three independent h20/h28/h50 calls, strict 576×20×20×3 input shape, exact output geometry, use of the corrected renderer return contract, JSON-normalized audit roundtrip, and boundary values of the frozen structural/flatness gate.

```bash
.venv/bin/ruff check scripts/run_drunet_sigma50_protected_all700.py \
  src/aiijc_puzzle/drunet_sigma50_protected_broad.py \
  tests/test_drunet_sigma50_protected_broad.py \
  tests/test_run_drunet_sigma50_protected_all700.py
.venv/bin/pytest -q tests/test_drunet_sigma50_protected_broad.py \
  tests/test_run_drunet_sigma50_protected_all700.py
```

At freeze time Ruff passed and all four focused tests passed. The calibration context loaded with all data, source, checkpoint, license, and selector hashes verified. Before calibration passed, the holdout context correctly failed closed because no calibration report existed; after the immutable calibration pass it admitted the exact unchanged holdout protocol.

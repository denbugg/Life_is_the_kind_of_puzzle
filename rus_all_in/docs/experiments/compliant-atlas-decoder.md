# Compliant population-atlas unary decoder

Preregistration: 2026-08-30, before this decoder's calibration run.

## Mandatory output contract

The organizer's clarification requires an actual 24×24 placement of all 576
dirty fragments. Constant canvases, low-frequency-only canvases, fragment
substitution and pre-layout pixel modification are not eligible puzzle
solutions. The earlier low-frequency roster is therefore research-only and
cannot be promoted regardless of SSIM.

Every variant in this experiment obeys a stronger machine-audited contract:

1. inference returns a `tile_at_position` permutation of integers `0..575`;
2. the raw 480×480 canvas is byte-for-byte `assemble_tiles(input_tiles[layout])`;
3. input/output tile SHA-256 multisets are equal;
4. no target is available to the score, unary, decoder or renderer;
5. only after the raw permutation audit passes, colored full-canvas NLM `h=9`
   may restore image quality.

Calibration targets are decoded only after layouts, raw canvases, NLM outputs,
hashes and compliance audits have all been frozen. Targets then provide SSIM
and approximate placement/adjacency diagnostics only. Holdout is not opened.

## New signal

The 5,600-train population atlas from
`artifacts/low-frequency-prior/train5600-v1.npz` is not rendered. Its mean clean
tile feature at each of 576 positions produces a robust tile→position unary for
the current dirty bag. Bilateral E14 MGC+SSD remains the local adjacency score.
The new unary can only choose where real dirty tiles are placed.

## Frozen calibration-48 roster

Historical controls:

- bilateral ORBIT best-buddies, 96-edge component budget;
- bilateral E14 relaxation without position unary;
- bilateral E14 relaxation with its inference-visible border unary.

New variants:

- best-buddy component packing/fill with population unary weights
  `0.01/0.03/0.06/0.12`;
- E14 relaxation with the same four population unary weights;
- E14 border unary plus `0.5 × population unary`, at historical position
  weight `0.11`.

All 12 layouts emit both exact raw reassembly and colored NLM `h=9`. The runner
reports mean raw/restored RGB SSIM, direct and translation-aligned placement,
right/down/combined adjacency, and paired bootstrap intervals against the
matching historical decoder. The exploratory selected new variant is the one
with maximum mean restored SSIM on the fixed shared calibration-48 panel. No
holdout gate is authorized in this task.

Authoritative output:
`outputs/compliant-atlas-decoder/calibration48.json`.

## Result

The frozen calibration-48 run completed in 328.95 s. Every one of the 576 raw
tile assignments for all 12 variants and all 48 boards passed the bijection,
exact-render and input/output tile-multiset audits.

| Decoder | Raw SSIM | NLM h9 SSIM | Gain vs matched control | 95% paired CI | Adjacency |
|---|---:|---:|---:|---:|---:|
| buddies96 control | 0.110590 | 0.196057 | — | — | 0.036439 |
| buddies96 + atlas w=0.03 | **0.111141** | **0.197319** | +0.001262 | [−0.002358,+0.005099] | 0.036383 |
| E14 relaxation control | 0.105402 | 0.182517 | — | — | 0.042421 |
| E14 relaxation + atlas w=0.03 | **0.109715** | **0.192610** | **+0.010092** | **[+0.006208,+0.013964]** | 0.043063 |
| E14 border control | 0.104828 | 0.181550 | — | — | 0.042912 |
| border + 0.5 atlas | 0.107423 | 0.188354 | +0.006804 | [+0.002750,+0.011008] | 0.042478 |

The atlas is a real and statistically stable positional aid for E14 relaxation,
but that decoder remains below the stronger buddies96 control in absolute SSIM.
Inside buddies96, the best atlas weight adds only `+0.001262`, its interval
crosses zero, and exact adjacency is unchanged. Population geometry therefore
cannot replace the missing learned edge signal. The best compliant primary arm
is retained as a weak optional unary, not claimed as a confirmed improvement.

Authoritative selection digest:
`5b4ff9b7e14b8fbb3e6522a4398c912d477e5ec7c877ad8242e5f8c7c3b0e8eb`.
Holdout was not opened.

The runnable target-free production entrypoint is
`predict_compliant_atlas(input_image, generic_tile_template)`. It freezes the
cross-panel decisions `atlas_weight=0.03`, `edge_budget=96`, and proper RGB NLM
`h=10`; its API contains no target parameter and returns the layout, exact raw
canvas, restored canvas and permutation audit together. The weak atlas weight
comes from the primary calibration-48 comparison, while `h=10` and the decision
not to increase the edge budget come from the disjoint fresh-panel ablation.

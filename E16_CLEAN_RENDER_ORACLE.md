# E16: exact clean-render restoration oracle

E16 measures whether post-assembly, content-preserving restoration has enough
headroom to justify a bounded learned pilot. It never changes the Rank96 board,
matching scores, tile order or orientation.

For each byte-pinned E12 scene, RR96 is replayed and its board hash must match.
Pristine target tiles are then restored to corrupted input-ID order by the
verified mapping:

```text
clean_tiles = imgio.to_frags(target_uint8)[permutation]
```

Those upright 20×20 tiles are assembled with the exact same RR96 board. No NLM,
diffusion, smoothing, colour fit, resize, blending or inpainting is applied.
The target-derived render is strictly non-deployable and no oracle image is
written to disk. It is an oracle for faithful content-preserving restoration,
not a mathematical upper bound on arbitrary generative editing.

The candidate is compared with the stored byte-pinned RR96 NLM `h=10` final
SSIM. A restoration pilot opens only if all inclusive checks pass:

- mean final-SSIM delta `>= +0.050`;
- strict wins `8/8`;
- worst-scene delta `>= +0.020`.

The large required gap allows a later realistic model to capture only part of
the oracle benefit. If opened, that separate source-disjoint model gate must
improve mean final SSIM by at least `+0.005` and capture at least `0.20` of the
E16 oracle gap. Otherwise post-assembly diffusion is closed for the current
board and work returns to global placement.

Files:

- evaluator: `src/eval_e16_clean_render_oracle.py`;
- tests: `tests/test_e16_clean_render_oracle.py`;
- report: `E:/pazzle_work/restoration_ceiling_e16/rr96_clean_render_oracle_v1.json`.

The report is atomic/restart-safe and all large output stays on `E:`.

## Result

The frozen oracle was rejected. Exact clean rendering averaged `0.1440086517`
versus the stored RR96 NLM mean `0.1593044531`, for `-0.0152958014`. It won
only `1/8`; the worst delta was `-0.0355985659`. All board hashes reproduced,
and no candidate solver, NLM or other restorer ran.

Report SHA256:
`d61fcf16ec9704c59724330f8a6eb8144ee322268712be1814f0a896c8b9da76`.
Post-assembly content-preserving diffusion is closed for the current board;
placement must improve first.

# Visual QA: harmonized frozen-QAP submission

Compared the old LB-0.203 render and the new harmonized render side by side for:

- `img_000000.png` (first archive member)
- `img_000133.png`
- `img_000488.png`
- `img_001543.png` (first member of the second frozen layout shard)
- `img_002999.png` (last archive member)

Verdict: the layout is visibly byte-for-byte unchanged and remains the dominant error. The new renderer makes modest tile-level colour and seam transitions more consistent. At 480x480 display scale it introduces no visible blur, ringing, clipping, new geometry, or face/texture erasure in these five checks. The change is deliberately subtle and cannot rescue an incorrect permutation.

This is subjective target-free QA, not a score claim. The source-disjoint synthetic validation reports remain the evidence for the expected SSIM improvement.

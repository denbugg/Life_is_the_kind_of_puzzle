# Learned 2x2 hyperedge gate v1

Decision: **do not promote; close this learned 2x2-anchor formulation**.

The run completed normally on 2x Tesla T4 in `2427.64 s`, reproduced the
authoritative boundary-QAP real16 baseline within floating-point precision,
used disjoint whole-source splits, and froze all real layouts before targets
were opened. The failure is scientific, not infrastructure-related.

## Gate results

| Metric | Observed | Required |
|---|---:|---:|
| validation average precision | 0.01593 | diagnostic |
| calibrated exact8 hyperedge precision | 0.01744 (6/344) | >= 0.90 |
| exact8 tile coverage | 0.29861 | >= 0.15 |
| exact8 adjacency gain | -0.02661 | >= +0.03 |
| real16 QAP baseline SSIM | 0.182819915 | exact reproduction |
| real16 hyperedge SSIM | 0.161223830 | — |
| real16 SSIM gain | -0.021596085 | >= +0.015 |

The high calibrated threshold (`0.9283`) did not correspond to real precision:
the verifier accepted 344 anchors but only six were correct. The absolute
anchor placement then reduced exact adjacency from `0.06103` to `0.03442` and
collapsed largest correct components. Both degradation engines failed in the
same direction:

- primary Kornia: exact adjacency `-0.02536`, real8 SSIM `-0.02521`;
- independent libjpeg: exact adjacency `-0.02785`, real8 SSIM `-0.01798`.

This is not a threshold-retuning problem. Achieving 15% coverage at the
observed calibration has only ~2% precision, while increasing the threshold
enough to target 90% precision would leave essentially no usable anchors. The
synthetic-hard-negative verifier did not transfer, and absolute insertion of
false 2x2 anchors destroys the locally coherent QAP fragments. Further epochs
or a looser coverage rule are therefore not justified.

## Integrity

- checkpoint SHA-256: `ee23d6388e93f4e7581bc6184c82a24fd5ca8f8dd755a5a0d1e67e8523d2ebf3`
- training report SHA-256: `74917f3d6b0e16fce81a68c3e7eafd59aa27764e4a34085885a5c40c3961b32e`
- gate report SHA-256: `e72405bee4baee26ff12170afb9141f5064c5dc5f3dbd48acfcb2ff09751f33c`
- primary report SHA-256: `58de6296b27af1bba5aa47255426ee8681b96d475409e45090128d241c6a67b2`
- independent report SHA-256: `4db1711fa591545a808a4f761bed8ead62c95fde63fa87544b44eef45b80b7be`
- authoritative QAP report SHA-256: `cc1b694b1501ba9b02e5618ad838e155ae40af7990bbbf4542b281fc21adec60`

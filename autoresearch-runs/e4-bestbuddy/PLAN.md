# E4 — best-buddy/component initializer

- **Angle:** G, coordinate-consistent component growth (Growing Consensus / Paikin-style best buddies).
- **One variable:** replace the position-only Hungarian initializer with reciprocal, high-confidence local components; keep the existing 400k-step SA unchanged.
- **Mechanism:** reciprocal margin-filtered edges form locally correct chains → coordinate-consistent placement preserves those chains in the initial basin → unchanged SA spends its budget polishing global placement → robust SSIM and adjacency improve.
- **Expected delta:** `+0.001..+0.008` robust SSIM on frozen smoke-32.
- **Falsification:** robust SSIM does not beat the paired Hungarian start, mean SSIM regresses, or adjacency materially regresses.
- **Fixed parameters:** reciprocal top-1 in both row and column, minimum two-sided top1–top2 margin `0.5`; components merged strongest-first and anchored by summed position logits.
- **Gate:** keep only if robust and mean SSIM improve, adjacency does not materially regress, all 32 outputs are valid permutations, no failures occur, and a serial recheck reproduces the result.

# Data

- Status: approved and locally verified on 2026-08-06.
- Train inputs: `E:/pazzle_data/train/inputs` — 7,000 PNG files.
- Train targets: `E:/pazzle_data/train/targets` — 7,000 PNG files.
- Test inputs: `E:/pazzle_data/test` — 700 PNG files.
- Existing gates: `E:/pazzle_work/gates` — 110 files at approval probe.
- Existing checkpoints: `E:/pazzle_work/ckpt` plus research-specific checkpoint folders — 45 files in the main checkpoint directory at approval probe.
- Labels: exact permutations only for newly generated synthetic corruption/shuffle from clean targets. Recovered real-input permutations are pseudo-labels and cannot be used for confirmation truth.
- Confirmation policy: save actual shuffled uint8 tile bytes, clean target, and exact permutation. Never regenerate a reporting scene from an RNG recipe.
- Source leakage policy: build a deterministic exact/near-duplicate source-group manifest before selecting the 24-scene final gate; fail closed on an unknown group.


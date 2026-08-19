## Shared board
- Baseline cache is finite and hash-verified.
- Weight-only calibration did not pass the joint SSIM/adjacency gate.
- Next lever: preserve and relocate components.
- Generation 1: random 2x2 moves won 23/32 but improved robust SSIM by only 0.000184; proposals must be directed toward weak regions.
- Generation 2: guided 2x2 improved mean only 0.000296 and robust 0.000016; block relocation is not the main bottleneck.
- Generation 3: `two_side` and `two_side_block2` improved smoke robust SSIM by `+0.000289` and `+0.000346`, but both materially reduced adjacency (`0.087409` to about `0.08434`). They fail the dual-metric gate and will not consume a full-128 run.
- Mechanism audit: the claimed multi-side consensus is refuted; positive total-objective acceptance can trade away true local neighbors. Next use calibrated ranks/margins or explicit grid/cycle consistency before wider moves.

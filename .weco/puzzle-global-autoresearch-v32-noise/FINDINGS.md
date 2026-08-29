# V32 findings board

## Kept

- N01 exact corruption contract: all geometry, range, determinism, permutation
  and random-name tests pass.
- The main S2/S3 critic has 1,000,924 parameters versus 836,644 for S1; the
  conditional S4 has 1,180,308.

## Preliminary evidence

- Exact challenge corruption cuts fused neighbor Top-1 roughly in half on the
  first real clean target.  It therefore creates a meaningful robustness task.
- Four-scene cache smoke produced 34 boards per scene: 24 real solver candidates
  and 10 bounded near misses.
- A 50-step S3 execution completed end-to-end.  Its one-scene selection result
  is intentionally not treated as evidence because the run is far below budget.

## Running

- Paired clean + two-noisy score cache for 60 scenes.
- After cache completion: full spatial candidate cache, then S1/S2/S3 training
  with scene-group OOF and locked validation.

## Next levers

- If S3 passes, test two replicas versus one and only then the 1.18M S4 scale-up.
- If S3 fails but its local error map is calibrated, use it solely as the LNS
  destroy policy while keeping the handcrafted final selector.

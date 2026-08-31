# Solver step 99 — Twin unique-supply local32

Parent in both Weco pair and exact tracks: step 97.

Frozen FullResolutionTwin top32 was intersected with the existing Union-v2
confidence-sorted hard top144 per axis, deduplicated against the confirmed
selective+fullres combined union, and filtered by the unchanged focal logit
`>=0`. One seventh arm used original TASKA costs and focal-gated tail96.

- pairs `326.78125 -> 328.40625`, delta **`+1.625`**, source CI95
  **`[+0.375,+3.125]`**;
- recall `0.295997509 -> 0.297469429`;
- exact `5.9375 -> 7.65625`, delta `+1.71875`, CI95
  `[-0.6875,+5.8125]`;
- accepted unique Twin supply: `5.3125` edges/board, precision `45.88%`;
- pair W/T/L `5/27/0`, strict parent replay `32/32`.

The preregistered nonnegative local pair gate passed. No competition test or
pixel postprocessing was used.


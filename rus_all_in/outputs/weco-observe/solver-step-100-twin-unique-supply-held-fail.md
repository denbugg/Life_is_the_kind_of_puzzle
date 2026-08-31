# Solver step 100 — Twin unique-supply held32 gate failure

Parent in both Weco pair and exact tracks: step 99. The unchanged step99
contract was transferred to held32.

- pairs `345.3125 -> 345.53125`, delta **`+0.21875`**, source CI95
  `[-1.78125,+2.625]`;
- recall `0.312783062 -> 0.312981205`;
- exact `1.90625 -> 1.8125`, delta `-0.09375`, CI95
  `[-0.46875,+0.1875]`;
- accepted unique Twin supply: `5.34375` edges/board, precision `40.94%`;
- pair W/T/L `1/29/2`, strict parent replay `32/32`.

The preregistered held pair gate required `+0.5` and failed. Fresh was not run,
so step101 is intentionally absent. Do not loosen top144, focal threshold,
selector roster or tail budget on the opened panels.


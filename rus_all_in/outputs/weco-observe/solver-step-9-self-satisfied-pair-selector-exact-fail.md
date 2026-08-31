# Solver step 9: self-satisfied pair selector improves pairs, loses exact

Status: fixed target-free selector failed on the independent opened64 replay.
Close it without a threshold sweep.

Rule discovered on the earlier learned eval32: for rank-delta and learned
layouts, count how many of each arm's own top-144 right plus top-144 down hard
edges are realised by that arm's decoded layout.  Choose learned only when its
count is strictly higher; ties keep rank-delta.  This uses no labels, pixels,
or reference layout.  Opened64 decisions were committed before reading the
scored report (`sha256 756621d9cba3e2551ac265027eba03000c948595b36ae89a0ef01e82f8deed4e`).

The selector chose learned on 43 boards and rank-delta on 21.  Versus always
rank-delta:

- exact tiles per board: `1.90625 -> 1.140625` (`-0.765625`);
- satisfied adjacent pairs: `154.875 -> 156.390625` (`+1.515625`);
- adjacency recall: `0.1402853261 -> 0.1416581748`
  (`+0.0013728487`);
- correct fixed top288: `143.5 -> 145.171875` (`+1.671875`);
- exact wins/ties/losses: `7 / 38 / 19`;
- satisfied-pair wins/ties/losses: `27 / 24 / 13`.

This independently confirms that target-free pair self-consistency can select
more locally coherent layouts but does not select the correct absolute/global
placement.  Do not tune a count margin or mix this with the failed component
selector.  Always-rank-delta remains the exact-oriented arm.


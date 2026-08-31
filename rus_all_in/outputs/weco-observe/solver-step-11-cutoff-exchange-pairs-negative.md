# Solver step 11: cutoff-exchange continuation is strongly pair-negative

Status: the single preregistered development arm failed. Do not sweep the
continuation length, learning rate, margin, or cutoff on the opened eval32.

Treatment: strict-load the frozen 400-step Union hard-edge head and continue it
for exactly 200 AdamW updates with a current-cutoff exchange loss. For each
axis, the loss ranks every false selected top-144 edge below every missed true
edge. It is relative/common-shift invariant and therefore materially differs
from the historical M467 frozen-threshold hinge. The target-free feature cache,
hard identities, decoder144, QAP24, and cyclic5 remain unchanged. Predictions
and strict layouts were hash-frozen before evaluation references were recreated.

Opened eval32, candidate versus the frozen learned-priority parent:

- satisfied adjacent pairs: `164.03125 -> 152.25` (`-11.78125`, source-clustered
  95% CI `[-15.84375,-8.28125]`, source wins/ties/losses `0/0/16`);
- adjacency recall: `14.85790% -> 13.79076%` (`-1.06714 pp`);
- correct fixed top288: `153.6875 -> 140.09375` (`-13.59375`, 95% CI
  `[-17.28125,-10.3125]`);
- exact tiles: `1.09375 -> 1.21875` (`+0.125`, 95% CI
  `[-0.59375,+0.90625]`);
- all three arms produced 32/32 strict original-upright-tile permutations.

The small exact increase is noisy and comes with a large, systematic local-edge
regression, so this is not an exact or pair promotion. Keep the original
learned priority for pair research and the confirmed Direct rank-delta arm for
exact research. The next pair direction should replay a materially different
historical global solver rather than tune this loss on the opened panel.

Frozen report:
`outputs/union-hard-edge-cutoff-continuation/opened32-v1/report.json`
(`sha256 27638ac08c013c65b8a0cdf6611c94eba873a2dee8bdc3b32d672b7ec38567f9`).

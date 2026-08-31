# Solver step 5: rank-delta source-disjoint64 confirmation

Status: independently replicated positive engineering result on a newly
committed organizer-train source64 panel.  This panel is disjoint from the
opened Union/rank-delta development panel and from the learned-priority pilot.

Treatment: transfer the frozen Direct model's learned-minus-raw percentile-rank
displacement only onto identity-matched Union-v2 hard edges.  Preserve the
Union confidence multiset and decode with unchanged decoder144+cyclic5.

Matched fresh64 result:

- exact tiles per board: `1.234375 -> 1.875` (`+0.640625`);
- adjacency recall: `0.1393370697 -> 0.1401862545` (`+0.0008491848`);
- satisfied adjacent pairs: `153.828125 -> 154.765625` of 1104;
- correct fixed top288 hard edges: `143.046875 -> 143.5` (`+0.453125`);
- strict original-upright-tile layouts: `64 / 64`.

All mean metrics improved again.  Source-clustered intervals still cross zero
(exact `[-0.109375,+1.390625]`, pairs/adjacency likewise), so this is robust
replication evidence rather than a statistical guarantee.  It is now the best
confirmed solver continuation among the tested Union-hard conversions.

Frozen report:
`outputs/direct-rank-delta-component-selector/fresh64-v1/report.json`
(`sha256 b50c6df0df62d7f0f89f97d28dd87594d5656d31983b2f49a95e38792b87b46e`).

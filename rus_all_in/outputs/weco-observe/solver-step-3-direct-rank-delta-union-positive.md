# Solver step 3: direct rank-delta transferred to Union hard edges

Status: completed positive engineering result on the established opened Union
fresh64 panel.  This is not fresh promotion evidence.

Treatment: convert the frozen direct hard-edge model's raw and learned scores
to per-axis empirical percentile ranks.  Transfer only the learned-minus-raw
rank displacement to the same `(axis, source, target)` Union-v2 hard edge.
Unmatched Union edges receive zero displacement.  Reassign the original
Union-confidence multiset according to the adjusted order, then run unchanged
decoder144 and cyclic-border5.  The Union scale and spread are preserved
exactly; there is no scalar transfer weight or label-fitted calibration.

Results versus regenerated bit-identical Union-v2 baseline:

- exact tiles per board: `1.28125 -> 1.484375` (`+0.203125`);
- adjacency: `0.1441915761 -> 0.1444604846` (`+0.0002689085`);
- correct fixed top288 hard edges per board: `146.984375 -> 147.609375`
  (`+0.625`);
- mean identity overlap: `870.109375 / 1104` edges per board;
- strict original-upright-tile layouts: `64 / 64`.

All three mean metrics improved and the fixed engineering gate passed.  The
clustered intervals still cross zero (`exact [-0.34375,+0.734375]`, top288
`[-0.015625,+1.3125]`), so keep this as a promising non-default solver arm and
seek a disjoint confirmation before submission promotion.

Frozen report:
`outputs/direct-residual-union-priority/rank-delta-opened64-v1/report.json`.


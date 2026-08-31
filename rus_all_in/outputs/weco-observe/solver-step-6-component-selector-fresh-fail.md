# Solver step 6: component selector fresh64 did not beat always rank-delta

Status: preregistered fresh confirmation gate failed.

The target-blind whole-layout selector chose rank-delta on 42/64 boards and
Union-v2 on 22/64 using only redundant component consistency and largest
component size.  It remained legal and improved materially over Union-v2, but
did not improve over always applying the now-confirmed rank-delta arm.

Fresh64 result:

- Union-v2 exact: `1.234375`;
- always rank-delta exact: `1.875`;
- component selector exact: `1.828125` (`-0.046875` vs rank-delta);
- component selector adjacency recall: `0.1402711730`;
- component selector satisfied adjacent pairs: `154.859375 / 1104`;
- strict layouts across three arms: `192 / 192`.

Close this selector rule without a threshold or tie-break sweep.  Retain the
always-rank-delta arm.

Frozen report:
`outputs/direct-rank-delta-component-selector/fresh64-v1/report.json`.

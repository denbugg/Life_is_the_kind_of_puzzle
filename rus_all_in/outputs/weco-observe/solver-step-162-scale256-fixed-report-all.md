# Solver step 162 — scale256 fixed DEV64 report-all

This is one preregistered post-freeze measurement, not a candidate selection or
promotion.  All 11 layouts were frozen target-free before the 64 exact synthetic
references were reconstructed.  The report path was claimed before reference
access and cannot be replayed under the same signed protocol.

- signed evaluator config SHA-256:
  `4fa2cea71287afb6cccc5c8ed28d0487c58c88a9ac9032519025502431763318`;
- immutable report SHA-256:
  `f1dd1694470923c47cbba4a27f56deb4cd152b0b85f81a21198aae2d3af776de`;
- panel: `64` source-disjoint organizer-train sources, one fixed draw each;
- every output: strict permutation of all `576` original upright tile IDs;
- statistics: source-clustered bootstrap `20,000`, fixed seed `20260901`.

Canonical incumbent:

- exact tiles: mean `1.875`, median `0`, Q1/Q3 `0/1`, range `0..84`;
- satisfied pairs: mean `352.640625`, median `353`, Q1/Q3 `310.5/407`;
- absolute mean Manhattan: `14.8847114` cells;
- radius-1 / radius-2 recall: `0.0179579 / 0.0411513`.

The existing `selective_vote500_focal` arm is the only declared arm with a
positive source-bootstrap lower bound for exact and radius recall:

- exact mean `6.671875`, median `1`, Q1/Q3 `0/1`, range `0..166`;
- exact delta `+4.796875`, CI95 `[+0.375,+11.453516]`, W/T/L `20/34/10`;
- radius-1 delta `+0.0166016`, CI95 `[+0.002007,+0.036025]`;
- radius-2 delta `+0.0181749`, CI95 `[+0.002089,+0.038900]`;
- satisfied-pair delta `-5.734375`, CI95 `[-9.953125,-1.90625]`,
  W/T/L `17/18/29`;
- absolute-Manhattan delta `-0.413411` cells, CI95
  `[-0.922148,+0.050080]` (negative is better).

Its exact gain remains heavy-tailed: the largest positive source contributes
`51.39%` of positive exact mass.  It is useful Pareto evidence, but the robust
pair loss makes it ineligible to replace the incumbent solver.

The three non-KEEP portfolio rules also fail to establish a new joint leader:

| portfolio rule | changed cases | exact delta mean (CI95) | pair delta mean (CI95) | pair W/T/L |
|---|---:|---:|---:|---:|
| fixed-head comparator | 21 | `+0.5625 [-0.171875,+1.828125]` | `-2.34375 [-4.546875,-0.453125]` | `6/44/14` |
| union+dense dominance | 1 | `-0.015625 [-0.046875,0]` | `+0.015625 [0,+0.046875]` | `1/63/0` |
| source-normalized dominance | 31 | `+3.234375 [-0.0625,+9.125391]` | `-6.921875 [-11.5625,-2.8125]` | `11/33/20` |

The union+dense pair gain is exactly one positive source and therefore has
largest-positive-source share `1.0`; it is a no-op-like diagnostic, not robust
evidence.  Source-normalized exact evidence is also concentrated (`76.85%` in
one positive source) while its pair loss is decisive.

Conclusion: keep the canonical incumbent as the pair solver.  Preserve the
selective arm as an exact/radius Pareto comparator and pursue a separately
registered local/contextual action verifier on FIT-only data.  DEV64 labels
must not tune, rank, or select that future policy.  These absolute means are
not directly comparable with earlier Weco steps measured on different frozen
source panels.

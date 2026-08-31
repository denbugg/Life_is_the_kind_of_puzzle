# Weco Observe — solver step 135

Parent: solver adjacency pair run `6bf52932-d716-4959-bee4-d652d7286cba`,
step 102.

One fixed local32 OOF intervention: use p<0.5 only to cut an already realised
weak bridge; keep all p>=0.5 core positions immutable; apply at most one
raw-cost-best feasible rigid weak-subtree move. Result: `326.75000` pairs per
board versus confirmed six-arm `326.78125` (`Δ=-0.03125`), exact unchanged at
`5.93750`. Local gate failed, so Weco 136/137 were not scored/logged.

Artifact: `outputs/taska-context-bridge-relocator/fixed-v1/report.json`.

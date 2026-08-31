# Solver step 47: majority-bond consensus component is strongly negative

Four frozen pre-tail layouts (raw/logistic/focal-top5/nonlinear) voted over
all directed board bonds.  Fixed support>=2 bonds were ordered by support,
untouched TASKA raw cost, then stable identity and used as the component-builder
supply; the resulting layout received protected tail96.

Opened32 reached **322.84375 pairs**, recall **0.292430933**, and **1.90625
exact tiles**, versus the current four-arm selector+tail96 at
341.3125 / 0.309159873 / 4.75. Pair delta was **-18.46875**, source-cluster
CI95 `[-26.0625,-10.96875]`; exact delta was -2.84375, CI95
`[-7.75,-0.125]`.

The consensus supply was not sparse (825.44 retained bonds and 460.88 placed
component tiles per board on average). The failure is consistent with four
correlated solvers sharing large wrong geometries, which majority voting then
locks into the rigid component core. The opened nonnegative gate failed;
held300/fresh32 and nearby threshold/weight variants were not run.

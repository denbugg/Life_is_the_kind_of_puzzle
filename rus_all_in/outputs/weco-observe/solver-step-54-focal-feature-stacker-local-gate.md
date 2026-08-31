# Solver step 54 — fixed focal-feature stacker passes local gate narrowly

Parent in exact run: step 29. Parent in adjacency-pairs run: step 42.

One fixed supervised fusion arm was trained on the first 96 source-aligned
boards from the existing TASKA edge cache. Its exact 22 inputs are the 15
dirty-visible edge features, the SHA-gated recovered focal checkpoint logit,
and the six recovered top-5 handcrafted focal features. The estimator is
exactly unweighted `StandardScaler -> LogisticRegression(C=1, max_iter=1000,
random_state=0)`, with no sweep or feature selection.

On the disjoint local32 gate, the standalone stacker reached **310.15625
pairs**, recall **0.280938632**, and **2.53125 exact tiles**. Adding it as the
fifth arm to the raw/logistic/focal/nonlinear all-bond selector followed by
tail96 reached **314.46875 / 0.284844882 / 2.03125**, versus the fixed four-arm
baseline **314.375 / 0.284759964 / 1.375**.

The five-minus-four delta was **+0.09375 pairs** (CI95
`[-1.59375,+1.9375]`) and **+0.65625 exact** (CI95
`[-0.25,+2.21875]`). The preregistered nonnegative pair gate passed, so the
unchanged held32 panel was opened. All candidate layouts and inputs were
SHA-frozen before exact-reference reconstruction.

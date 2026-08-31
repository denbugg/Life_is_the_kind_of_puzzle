# Solver step 81 — fullres restored-view union voter, held32

Parent in both Weco runs: step 80.

The unchanged step-80 contract scored **341.78125 pairs**, recall
**0.309584466**, and **2.8125 exact** on held32, versus control
**337.5625 / 0.305763134 / 3.0625**.  Pair delta was **+4.21875**, clustered
CI95 `[+1.656,+6.875]`, so the preregistered `>= +0.5` gate opened fresh32.

Exact delta was `-0.25`, CI95 `[-0.5625,-0.03125]`; this is explicitly a
pair/exact tradeoff, not an exact promotion.  Accepted restored edges had
59.79% precision and raised candidate recall `23.933→25.713%`.

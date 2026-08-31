# Solver step 74 — target500 adds supply but hurts pair assembly

Parent in both Weco runs: step 42.

Only the dynamic mutual-vote target changed from 350 to 500.  The exact
same-pass target350 control reproduced the frozen control on all local32
cases.  Candidate count increased `374.438→531.344`, and supplied true pairs
increased `252.938→294.188` (`+41.250`).

The unchanged four-arm all-bond selector plus tail96 nevertheless fell from
`314.375` to `306.500` satisfied pairs (`−7.875`, CI95
`[-18.906,+2.281]`), and recall fell `0.284760→0.277627`.  Realised supplied
true pairs increased only `+9.563`, while true noncandidate seams fell
`−17.438`.  Exact rose `1.375→2.875`, but its delta CI `[-1.0,+5.625]` is
noisy and cannot override the pair gate.

The local pair gate failed; held steps 75 and fresh steps 76 were not created.
Close indiscriminate low-vote expansion and retain the result as evidence that
selective consumption—not raw supply alone—is the bottleneck.

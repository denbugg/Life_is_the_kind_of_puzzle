# P14b G0b Rejection - Raw Candidate-Slot Top-K Selection

| Contract | Result |
|---|---|
| Synthetic G0a | PASS: bidirectional 2x2 support preserved the true cell and removed dangling false edge. |
| Frozen cache source | First pinned FIT source only. |
| Candidate-order invariance | **FAIL**. |
| Baseline true directed adjacency recall at K=64 | 0.4012681; measured only after frozen scores loaded. |
| CAL / DEV / test / P8 | closed / closed / closed / not imported. |

P14b sliced the first K cache slots before applying topology propagation. The P12 score cache preserves a canonical candidate union but the slot order is not a semantic rank under an explicit shuffle; therefore the pre-registered axis-shuffle contract correctly failed. This is an integration/data-representation failure, not a held result. **Decision: REJECT at G0b; no G1 or CAL.**

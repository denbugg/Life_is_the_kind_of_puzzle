# P14c G0 Evidence - Score-Ranked Bidirectional Grid-Topology Propagation

| Gate | Result |
|---|---|
| G0a true isolated 2x2 edges retained | PASS |
| G0a dangling score-matched false edge removed | PASS |
| G0a candidate-order invariance | PASS |
| G0b frozen source | `img_003194.png` only |
| G0b score-ranked candidate-order invariance | PASS |
| G0b unpruned true directed-adjacency recall, K=64 | 0.759058 |
| G0b retained true directed-adjacency recall | 0.759058 |
| G0b retained fraction | 1.000000 |
| G0b strict 576-way decode | PASS |
| G0b removed physical edges | 2 RIGHT, 0 DOWN from 36,864 per direction |
| Labels / CAL / DEV / test / P8 | cached FIT only after frozen scores / closed / closed / closed / not imported |

P14c satisfies its safety and invariance contracts. The evidence also reveals a decisive efficiency diagnostic: on the first FIT cache, bidirectional 2x2 propagation is nearly inert, pruning only two of 73,728 score-ranked physical RIGHT/DOWN candidate edges while preserving all true candidate recall. The pre-registered G1 grid remains necessary to measure placement outcome, but large gains are not expected from this sparse alteration.

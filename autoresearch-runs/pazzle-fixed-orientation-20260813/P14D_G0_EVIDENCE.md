# P14d G0 Evidence - Symmetric Score-Ranked Grid Topology

| Gate | Result |
|---|---|
| G0a isolated true cell / dangling edge / order invariance | PASS |
| G0b frozen source | `img_003194.png` only |
| G0b physical graph invariance | PASS |
| G0b filtered score tensor invariance | PASS |
| G0b unpruned symmetric true directed-adjacency recall | 0.828804 |
| G0b retained recall / fraction | 0.828804 / 1.000000 |
| G0b strict 576-way decode | PASS |
| G0b physical RIGHT edges before / after | 50,407 / 50,407 |
| G0b physical DOWN edges before / after | 50,611 / 50,611 |
| G0b selected finite directional scores before / after | 147,456 / 147,456 |
| Labels / CAL / DEV / test / P8 | cached FIT only after frozen scores / closed / closed / closed / not imported |

P14d satisfies every G0 safety contract, including score-tensor invariance after restoring shuffled axes. The topology operator removed no edge on the representative frozen FIT graph. This means the score-ranked symmetric candidate graph is saturated with 2x2 completions and gives the filter no leverage at K=64. G1 remains pre-registered and must run before a formal gate decision, but any improvement is unlikely unless other K settings induce nontrivial pruning.

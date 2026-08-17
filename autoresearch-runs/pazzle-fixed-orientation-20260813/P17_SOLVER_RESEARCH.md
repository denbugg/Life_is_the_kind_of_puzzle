# P17 Solver Research - Exact-Delta Sparse QAP Polish

## External evidence

Exact swap-neighborhood methods are standard for QAP. Paul describes robust tabu search for sparse QAP using adjacency lists and priority queues to reduce per-iteration complexity from O(N²) to O(N log N), with practical O(N) behavior. [1] Podolsky and Zorin further derive O(1) delta component updates and report up to 25% speedup for robust tabu style methods. [2] The core relevance to ORBIT-24 is computational: a tile-position swap changes only directed puzzle edges incident to the two cells, so a candidate swap can be evaluated from a constant affected-edge set instead of recomputing a whole 24×24 objective.

## Local audit

Canonical `_repair` repeatedly picks only the 96 worst local-agreement cells, tries swaps against all 576 positions, and calls the full-board `objective` for every candidate. This can miss a beneficial exchange whose endpoints are not both in the preselected pool, and it causes the P15/P16 runtime failures when embedded in broader searches. P17 will differ by exact affected-edge deltas over the entire swap neighborhood, with no tabu/negative moves and a fixed small number of greedy improvements.

## Selected P17 mechanism

Start from the canonical rank96 strict board. In each of exactly 24 rounds, enumerate all unordered cell pairs, compute each swap delta from the unique horizontal/vertical directed edges incident to either cell, select the largest strictly positive delta (tie by lexicographic cell pair), then apply that swap. Stop early if no positive delta. Since only a constant number of edges changes per swap, every candidate delta is O(1). The final board is a strict permutation and its full frozen objective must equal the accumulated delta check.

## Fast-futility expectation

G0a can verify the exact-delta identity against brute-force full-objective recomputation on small synthetic boards and verify a planted nonlocal swap is recovered. G0b uses four frozen FIT score-cache boards with no labels, requiring objective non-decrease on all and strict increase on at least one under a 60-second total cap. Failure stops before labels/held.

## References

[1] Gerald Paul. An Efficient Implementation of the Robust Tabu Search Heuristic for Sparse Quadratic Assignment Problems. https://arxiv.org/abs/1009.4880
[2] Sergey Podolsky and Yuri Zorin. O(1) Delta Component Computation Technique for the Quadratic Assignment Problem. https://arxiv.org/abs/1206.0580
[3] zeman412. Tabu Search QAP reference implementation. https://github.com/zeman412/Tabu_Search_QAP_20

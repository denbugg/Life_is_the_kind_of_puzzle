# Solver step 17: extra seam-fill rounds do not improve adjacency

The full legal TASKA harvest and component placement were held fixed while the
Hungarian seam tail was iterated using its previous assignment as context.
This is permutation-equivariant and target-free; only the number of fill
rounds changed.  The opened32 exploratory results were:

| Fill rounds | Pairs / 1104 | Recall | Exact / 576 |
|---:|---:|---:|---:|
| 1 (parent) | 334.71875 | 0.303187274 | 4.46875 |
| 2 | 331.5 | 0.300271739 | 4.46875 |
| 3 | 333.9375 | 0.302479620 | 4.53125 |
| 5 | 333.53125 | 0.302111639 | 4.65625 |

Five rounds gained only 0.1875 exact tiles while losing 1.1875 pairs.  Since
the primary adjacency metric did not improve and the exact change is tiny on
an already opened panel, the arm is closed without source-held replay.

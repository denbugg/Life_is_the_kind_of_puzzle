| exp_id | angle | change | status | metric | delta | verified | note |
|---|---|---|---|---|---|---|---|
| E0 | — | baseline | passed | 0.09470925 | 0 | n/a | reproduced on smoke-32 |
| E1 | block | 2x2 swaps | queued | — | — | no | |
| E2 | segment | length-4 swaps | queued | — | — | no | |
| E3 | consensus | mixed moves | queued | — | — | no | |
| E4 | two-side | weak-region proposals | queued | — | — | no | |
| E5 | guided-LNS | weak 2x2 + best-of-24 destinations | queued | — | — | no | generation 2 |
| E6 | two-side | weak-position all-neighbor relocation | dropped | 0.09499794 | +0.00028870 | no | SSIM gain but adjacency fell 0.08740942→0.08432405; 16/32 wins |
| E7 | hybrid | two-side then block polish | dropped | 0.09505525 | +0.00034600 | no | adjacency fell to 0.08435236; 17/32 wins, fails gate |

# E19 results — rejected at smoke-16 gate

## Change

Relative to verified E14, E19 adds one score view: each raw 20x20 tile is
independently denoised with OpenCV colored NLM (`h=9`, template window 7,
search window 21). Raw and NLM MGC+SSD log-probabilities are averaged 50/50,
then passed through the unchanged E14 learned/classical fusion (`alpha=0.2`)
and unchanged relaxation solver. Layout quality is scored using raw tiles.

## Frozen smoke-16, seed offset 0

Cache SHA-256:
`74db2b62e9d5eafffae33117c7771512d823b0dcaa0095ef5807adb8e86a25df`.

| metric | E14 | E19 | delta |
|---|---:|---:|---:|
| mean SSIM | 0.1057701490 | 0.1059277853 | +0.0001576363 |
| robust SSIM | 0.0999884937 | 0.0999975719 | +0.0000090782 |
| mean adjacency | 0.1064311594 | 0.1061480978 | -0.0002830616 |
| end-to-end runtime | 20.246 s | 48.661 s | 2.403x |
| valid permutations | 16/16 | 16/16 | — |

SSIM wins: 9/16. Adjacency wins: 11/16. Failures: 0.

Runtime decomposition: shared raw classical preprocessing `7.905 s`; additional
NLM plus second classical graph `28.194 s`; E14 solver `12.341 s`; E19 solver
`12.561 s`. Thus the regression is caused by the added NLM/classical view rather
than the unchanged relaxation solver.

## Decision

Reject and stop. E19 fails three predeclared gates: robust delta is below
`+0.0005`, adjacency is negative, and runtime exceeds `2x`. Only mean SSIM is
positive. Per the staged protocol, alternate-seed, smoke-32, and untouched-96
runs were not launched.

Selection consumed only raw tiles, learned directional matrices, position
scores, and the seed. Target/truth were accessed only after each valid layout
was fixed for reporting.

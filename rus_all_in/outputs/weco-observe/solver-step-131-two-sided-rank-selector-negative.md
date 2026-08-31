# Solver step 131: two-sided log-rank layout selector is negative

Parent: confirmed selective+unique-fullres six-arm fusion, step 102.

A parameter-free selector summed `log1p` outgoing-row and incoming-column
ranks for all 1104 realised seams of each post-tail layout. On opened local32
it reduced pairs `326.78125→324.12500` and exact `5.93750→1.34375`.

The proxy actively changed arms but amplified whole-layout winner's curse and
global-origin errors. Held/fresh and rank transform/top-k/blend sweeps were not
run. Source: `src/aiijc_puzzle/taska_two_sided_rank_selector.py`.

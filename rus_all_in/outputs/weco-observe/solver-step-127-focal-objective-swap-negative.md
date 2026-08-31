# Solver step 127: sparse focal-objective swap is negative

Parent: confirmed selective+unique-fullres six-arm fusion, step 102.

On opened local32, one globally best non-adjacent swap under the signed frozen
focal-logit objective changed all 32 layouts but reduced satisfied pairs from
`326.78125` to `326.46875`; exact stayed `5.93750`. Pair W/T/L was `0/23/9`.
The positive-softplus-only diagnostic changed just 2/32 layouts and yielded
`+0.03125` pairs with exact flat, so it was too sparse for a held gate.

The branch is closed without iterative, threshold, weight, or budget sweep.
Source: `src/aiijc_puzzle/taska_focal_objective_swap.py`.

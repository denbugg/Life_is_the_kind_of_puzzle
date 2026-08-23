# Plan and current state

## Verified chain

    tiles -> restorer (MGC contrastive loss) -> mgc_cost -> solve_lp -> assemble -> NLM

End-to-end on CLEAN tiles: place_acc **0.9965**. At clean_blur tile quality:
0.5590 with the LP against 0.3585 greedy. Code: `mgc.py`, `solve_lp.py`,
`solve_loop.py`, `torus_origin.py`, `restore_tile.py`, `infer_mgc_submission.py`.

## The single number that decides everything

Seam R@1 on real held-out boards:

| stage | R@1 |
|---|---:|
| raw tiles | 0.056 |
| best restorer | **0.171** |
| greedy assembly needs | ~0.60 |
| LP needs | ~0.79 |
| ceiling if noise+JPEG removed | 0.774 |

Payoff if we get there (M19): place_acc 0.30 beats the 0.23749 submission,
0.64 matches the leader's 0.40.

## What is exhausted, with numbers

* **Compatibility measure** — MGC beats plain L2, whitened L2, Pomeranz (Lp)q,
  inset strips, extrapolated borders, global tile statistics and every fusion.
* **Solver** — LP (Yu et al.) beats greedy and has no torus ambiguity; mutual
  top-1 edges are optimal, extra candidates always hurt.
* **Model scale** — 6x capacity and 5x length plateau at R@1 0.158 / bb_prec
  0.217; residual noise does not fall.
* **Restoration target** — `clean` and `blur3(clean)` give identical curves.
* **Post-processing** — sharpening, iteration, TTA, ensembling all negative.
* **Context** — needs >=35% correct neighbours; the reliable core supplies 1%.

## Open

1. **Seam inpainting scorer** (`seam_inpaint.py`, training now). Predict the
   removed join strip from both pieces' interiors and score by agreement with
   the observation, instead of comparing two noisy borders. This is the only
   untested idea that uses the interior constructively.
2. **Positional diffusion** (JPDVT). Bypasses pairwise compatibility entirely.
   Demonstrated to 150 pieces against our 576, under milder corruption.

## Excluded by the owner

External source photographs, and metric gaming such as constant fill.

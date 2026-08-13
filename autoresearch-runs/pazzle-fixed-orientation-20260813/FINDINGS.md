## Champion by candidate coverage
- R2 is provisional R@20 champion: 0.397758 versus R0 0.352468 (+0.045290).
- It is not a scientific pass: R@1=0.059047, R@5=0.184556, and b128 neighbour=0.040308 all remain below gate.
- R1 refuted naive untrained multi-band cosine fusion: R@20=0.259964 (-0.092504 vs R0).
## Next lever
- Train a pairwise cross-encoder with same-image hard negatives, then use it as a reranker of R2 top-K retrieval.

## R3 mechanism audit
- Predicted: listwise hard-negative training would preserve a high-recall union while raising local ranks.
- Observed: candidate coverage=0.688179 and reciprocal mutual coverage=0.898438; local all-true proxy R@1=0.077958 remains insufficient.
- Conclusion: mechanism is confirmed only for candidate coverage. Keep R3 as sparse union generator; global slot evidence must arbitrate ambiguous rows.

## G1a mechanism audit
- At 200 steps the coarse 6x6 set prior remains near chance (Hungarian membership=0.0295 versus 1/36≈0.0278).
- This is an undertrained preflight, not evidence that global context is useless: default trainer budget is 8000 steps and loss has not plateaued.
- Next: extend same no-rotation hypothesis at a bounded 1200 steps before changing architecture.

## G1b mechanism audit
- Predicted: longer set-based global context would learn image-semantic macrocell positions and improve macro assignment.
- Observed after 1200 steps: macro Hungarian=0.034288 and top64 group coverage=0.131510; both are only marginally above random baselines 0.027778 and 0.111111.
- Conclusion: drop visual-only global prior. Use the high-coverage R3 relation graph itself as global evidence in a graph-conditioned fusion/assignment stage.

## F1 mechanism audit
- Predicted: hierarchical direct/non-direct plus direction classifier would convert R3 candidate coverage into calibrated reciprocal edges.
- Observed: mutual-direct coverage=0.912639 but reciprocal precision=0.033840. The model still scores too many false direct edges.
- Conclusion: do not assemble. The next lever is calibration/selection over frozen candidate scores, not another uncalibrated direct classifier.

## F2 mechanism audit
- Fusion creates a high-precision sparse top1 signal (direct precision=0.4184), but coverage remains 0.1091; the dense graph retains high recall but 0.0401 precision.
- Conclusion: simple score fusion cannot itself select a usable graph. Test a constrained assignment/repair mechanism that can exploit one-use and grid constraints without accepting dense false edges.

## C1 cycle-consistency pre-gate (2026-08-13)
On two fresh held-out boards, R3 union top64 has 75.77% symmetrized true-direct-edge coverage and 40.83% all-four-edge C4 availability, but exact true 2x2 motif coverage is only 1.51% at 128 and 2.93% at 512 retained motifs per anchor. A graph-only C1 reranker would touch too few correct local structures to plausibly shift global precision by the required 5 pp; reject before implementing/training.


## R2L directional Siamese scale (2026-08-13)
The 800-step run materially improved directional retrieval: best step 600 produced R@1 9.81%, R@5 26.06%, R@20 49.88%, median rank 22.875, and b384-neighbour 7.39% on 8 held-out boards. The previous R2 R@20 was 39.78%, so scale improves candidate recall (+10.10 pp) but row-top1 and local-neighbour gate values plateau far below 25%/18%. Retain best.pt only as a candidate-graph union component; do not treat it as a direct assignment scorer.


# E11 — sparse relaxation-labeling global solver

- **Angle:** G, structural cross-domain transplant.
- **Source:** [ACCV 2024 relaxation labeling](https://arxiv.org/abs/2410.16857).
- **One structural change:** replace 400,000-step stochastic local-swap SA with
  tile→grid probability relaxation using sparse top-12 reciprocal directional
  support, Sinkhorn/Hungarian projections, phased sharpening, confidence
  freezing, and Hungarian finalization.
- **Inputs permitted for selection:** frozen `right`, `down`, and `pos` arrays.
- **Inputs forbidden for selection:** `tiles`, `target`, `truth`, SSIM, and
  adjacency. Those are read only by the paired evaluator after a layout exists.
- **Mechanism:** global support propagation reinforces mutually consistent
  directional neighborhoods over the whole board, escaping local-move basins.
- **Expected delta:** `+0.005..+0.020` robust SSIM with adjacency moving in the
  same direction.
- **Falsification:** alternate-seed adjacency or SSIM fails to improve.
- **Gate:** frozen smoke-16 on declared seed and offset `1,000,003`; every output
  must be a permutation. Run holdout/full only if both seeds pass dual metrics.

The four-phase schedule and top-k value were frozen before the aggregate
smoke-16 metric was read. Within the solver, best-so-far selection uses only the
cached directional/position objective.

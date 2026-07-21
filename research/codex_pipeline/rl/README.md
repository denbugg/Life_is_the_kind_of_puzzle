# RL Puzzle Assembly Experiment

Isolated experiment for a fully connected actor-critic policy that improves a
24x24 puzzle through swap actions.

Promotion gate: held-out SSIM must exceed the same-split heuristic baseline of
`0.182062`, and RL adjacency accuracy must beat the heuristic policy evaluated
on the same shuffled boards. The historical end-to-end solver score is
`0.152632`.

Reward components:

- correct adjacent tile relations;
- exact tile positions;
- visual continuity of touching denoised edges;
- weak position-prior improvement;
- small step penalty.

Training uses a reward-guided warm start followed by PPO and a curriculum from
6x6 crops to complete 24x24 boards.

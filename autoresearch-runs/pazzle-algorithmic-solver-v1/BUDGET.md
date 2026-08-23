# BUDGET.md — Experiment Budget and Configuration

metric                 = solve_ssim / neighbour_accuracy / final_ssim
direction              = higher
seed_experiments       = 5
seconds_per_experiment = 300
parallelism            = 1
compute_cap            = 12.0 # hours
max_generations        = 6
hypotheses_per_gen     = 3
proposers              = 3
critics                = 2
stagnation             = 3

--- spent ---
generations_run = 0
experiments_run = 0
gpu_min_used    = 0
best_metric     = 0.1652 (neighbour) / 0.1913 (final_ssim baseline on held-out)
champion        = baseline_rank96_buddies

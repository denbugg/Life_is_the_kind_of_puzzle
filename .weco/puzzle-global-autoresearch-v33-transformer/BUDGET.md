# Budget

```text
metric                 = group_oof_selected_adjacency
direction              = higher
seed_experiments       = 3
seconds_per_experiment = 1200
parallelism            = 1
compute_cap            = 3 GPU-h
max_generations        = 2
hypotheses_per_gen     = 3
proposers               = 3
critics                 = 2
stagnation              = 2
```

Baseline within V32 cache: OOF `0.3134580627`, validation `0.3776041865`.

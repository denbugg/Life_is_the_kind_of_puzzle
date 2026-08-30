# V33 results

The verified winner remains the handcrafted baseline selector.

| Model | Parameters | OOF selected | Locked validation | Clean/noisy agreement | Decision |
|---|---:|---:|---:|---:|---|
| Baseline | - | 0.3134581 | 0.3776042 | - | keep |
| Transformer-S | 3.11M | 0.3134581 | 0.3776042 | 7.7% OOF / 0% val | reject; fallback only |
| Transformer-M | 8.77M | 0.3143464 | 0.3716033 | 11.5% / 25% | reject |
| Transformer-MC | 8.77M | 0.3138413 | 0.3766984 | 7.7% / 0% | reject |

All experiments completed, produced finite losses and checkpoints, and retained
the frozen scene-group split. No transformer passed both OOF and locked gates.

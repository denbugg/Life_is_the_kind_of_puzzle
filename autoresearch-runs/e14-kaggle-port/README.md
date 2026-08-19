# E14 reviewable Kaggle port

This package ports the verified E14 winner into the existing Kaggle inference
script without changing checkpoint discovery, restoration, image assembly,
submission ZIP creation, validation reporting, CUDA-to-CPU fallback, or the
legacy relation/RL path.

## Inference-only path

`kaggle_e14_solver.py` accepts only raw shuffled tiles plus learned right/down
and position matrices. It reproduces the frozen experiment components:

- raw 20-pixel tile boundaries only for classical MGC+SSD;
- diagonal excluded from both robust row normalization and neighbor choice;
- `classical_logp = log_softmax(-(d - row_median) / max(row_MAD, 1e-6))`;
- `fused = 0.8 * learned_logp + 0.2 * classical_logp` with alpha locked;
- unchanged E11 top-12 sparse, four-phase relaxation and final Hungarian solve;
- fixed E11 position weight `0.11` and deterministic seed tie-breaking.

The production `EdgeMatcher` emits directional logits, so its matrices receive
one row-wise `log_softmax` before the frozen fusion. Classical scores always use
the original `load_tiles()` result, never restored tiles, targets, truth layouts,
SSIM, or adjacency.

`USE_E14=1` is the default. If E14 raises or returns a non-permutation,
`E14_FALLBACK_ON_ERROR=1` preserves the already-computed legacy layout. Setting
`E14_FALLBACK_ON_ERROR=0` makes failures fatal for debugging.

## Regression evidence

Run:

```bash
E14_TEST_CACHE=/path/to/directional_student_holdout128.npz \
  python -m unittest tests.test_kaggle_e14_port -v
```

The local frozen cache run passed four tests in 3.11 seconds. On cached case 0:

- raw classical right/down matrices are bit-identical to verified E14;
- alpha-0.2 fused right/down matrices are bit-identical;
- the final relaxation/Hungarian layout is bit-identical and a valid 0..575 permutation;
- a synthetic E14 exception returns the exact legacy fallback layout.

The already-verified full-128 experiment remains the acceptance evidence:

| metric | baseline | E14 | delta |
| --- | ---: | ---: | ---: |
| robust SSIM | 0.1003414429 | 0.1014643490 | +0.0011229061 |
| mean SSIM | 0.1027065484 | 0.1039135704 | +0.0012070220 |
| adjacency | 0.0855129076 | 0.1026452106 | +0.0171323030 |

E14 won 67/128 SSIM cases and 116/128 adjacency cases; its measured end-to-end
runtime was 125.0452 s versus 428.7263 s (0.2916668x). Those metrics are from
the frozen experimental directional-score cache, not a claim about this Kaggle
checkpoint combination.

## Kaggle package

Private slug: `phoenix0501/pazzle-e14-fusion-relaxation`.

The push package contains `kaggle_solve_puzzles.py`, `kaggle_e14_solver.py`, and
the new metadata. `push_e14_kaggle.sh` refuses network access unless the caller
explicitly sets `KAGGLE_404_BLOCKER_CLEARED=1`. No E14 upload was attempted while
the known 404 blocker remained active.

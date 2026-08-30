# V33 findings

## Verified implementation

- Transformer-S: 3,113,643 parameters.
- Transformer-M/MC: 8,767,739 parameters.
- Both return one global score plus a `3x24x24` local map and complete a
  train/backward smoke on the RTX 4060.
- M forward smoke with two boards peaked near 268 MB allocated CUDA memory.

## Results

- T-S was rejected by OOF calibration, so fallback reproduced the baseline.
- T-M improved OOF by `0.0008884` but lost `0.0060009` on locked validation.
- T-MC improved OOF by `0.0003832` and lost `0.0009058` on validation.
- Clean/noisy agreement stayed between 0% and 11.5%, falsifying the robustness
  mechanism. Larger capacity did not solve the selector-identifiability issue.

## Next levers

- Change the lever to a model over the *set of candidate boards*, where the task
  is explicitly comparative, or use local error predictions only to guide LNS.
- Improve candidate oracle before another final-selector experiment; locked
  validation currently offers only `0.00147` headroom.

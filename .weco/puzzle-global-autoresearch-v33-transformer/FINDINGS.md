# V33 findings

## Verified implementation

- Transformer-S: 3,113,643 parameters.
- Transformer-M/MC: 8,767,739 parameters.
- Both return one global score plus a `3x24x24` local map and complete a
  train/backward smoke on the RTX 4060.
- M forward smoke with two boards peaked near 268 MB allocated CUDA memory.

## Running

- Full order: T-S, T-M, T-MC.
- Each variant uses four scene-group CV folds, then a final fit and locked
  validation. OOF margins calibrate fallback to the handcrafted baseline.

## Next levers

- Promote only a group-OOF winner.
- If all cell-token transformers fail, change the lever to a transformer over
  the *set of candidate boards* or use the local transformer map only to guide
  LNS; do not scale parameters blindly.

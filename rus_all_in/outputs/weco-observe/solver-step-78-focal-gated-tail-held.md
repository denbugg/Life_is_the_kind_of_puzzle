# Solver step 78 — fixed focal-gated protection, held32 gate

Parent: step 42. The unchanged logit-zero/tail96 rule transferred weakly:

- pairs `338.09375` vs all-edge control `337.56250`, delta `+0.53125`, CI95
  `[-1.875,+3.000]`;
- recall `0.306244339` vs `0.305763134`;
- exact `3.00000` vs `3.06250`, delta `-0.06250`.

The fixed held gate was pair delta at least `+0.5`; it passed by `0.03125` even
though its CI crosses zero. No parameter changed, so the one fresh32 replay
opened.

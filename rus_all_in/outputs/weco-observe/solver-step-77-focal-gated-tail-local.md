# Solver step 77 — fixed focal-gated protection, touched local32

Parent: step 42.  One frozen rule only: protect realised harvested edges when
the recovered `train_exact_top5` focal logit is `>=0`, then run the unchanged
original-cost non-adjacent tail96.

Local32 is touched by an earlier target-assisted `-1/0/1/2/3` threshold
diagnostic, so this is discovery only. Candidate versus all-edge control:

- pairs `314.40625` vs `314.37500`, delta `+0.03125`, CI95
  `[-1.78125,+1.84375]`;
- recall `0.284788270` vs `0.284759964`;
- exact `1.28125` vs `1.37500`, delta `-0.09375`.

The preregistered nonnegative pair gate passed narrowly. No threshold or budget
was changed; held32 opened.

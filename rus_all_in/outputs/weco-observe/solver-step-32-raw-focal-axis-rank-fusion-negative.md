# Solver step 32: raw/focal axis-rank fusion is negative

One parameter-free ordering was frozen before scoring: independently inside
each board and axis, convert raw TASKA priority and recovered focal top-5 logit
to average-midranks on `[0,1]`, then use their equal arithmetic mean.  Candidate
membership, matcher costs, placement, and Hungarian fill were unchanged.

On opened32 the fusion produced **334.125 pairs**, recall
**0.302649457**, and **3.90625 exact tiles**, versus raw TASKA
334.71875 / 0.303187274 / 4.46875.  Pair delta was **-0.59375** with
source-cluster CI95 `[-3.96875, +2.46875]`; exact delta was -0.5625.

The preregistered nonnegative-pair gate failed.  Held300 was therefore not
opened, and no alpha, rank, axis, or threshold sweep was run.  This equal-rank
fusion is closed.

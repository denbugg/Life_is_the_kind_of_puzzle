# Solver step 51: raw-log alternate tail closes as duplicate objective

The unchanged held300 diagnostic32 replay again produced byte-identical
control and raw-log layouts on all 32 cases.  Both reached **337.5625 pairs**,
recall **0.305763134**, and **3.0625 exact tiles**; the original-cost selector
retained control 32/32.

For every legal off-diagonal pair in opened and held matrices, original TASKA
cost is `-raw_log` plus an axis/case constant within at most `1.90735e-6` range;
the minimum measured correlation exceeded `0.99999999999999`.  Constants cancel
from equal-bond layout and swap comparisons, explaining exact identity.

The fresh gate required a strictly positive held pair delta, so fresh32 was not
opened.  This branch is closed without a scale/blend/budget sweep: a future
alternate tail needs a materially different source of evidence.

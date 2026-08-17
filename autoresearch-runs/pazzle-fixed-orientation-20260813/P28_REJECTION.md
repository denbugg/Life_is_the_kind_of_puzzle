# P28 GDCP-24 — REJECTED at G2a capacity gate

**Status:** Rejected before 96-source training, held-32, CAL, DEV, test, or submission.

The two-board, edge-conditioned coordinate denoiser was trained for the registered 600 FP32 steps. Its coordinate RMSE was 0.294983 and 0.297715 versus a 0.300965 random-coordinate baseline. This is far short of the pre-registered 50% RMSE reduction gate (<=0.150482) and offers no useful global-pose capacity.

G0/G1 passed. P10 labels were accessed only at G2a. Targets remained unopened; held/CAL/DEV/test and P8 were excluded.

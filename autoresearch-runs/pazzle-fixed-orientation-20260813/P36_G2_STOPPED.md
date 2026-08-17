# P36 CSRP-24 G2 stopped — fast futility

P36 G2 was deliberately terminated without a quality decision. The paired canonical solver evaluation completed only 4 of 96 FIT-train boards after roughly nine CPU-minutes, implying a projected run far beyond the pre-registered 15-minute cap. The task was stopped at that boundary; no G3, CAL, DEV, held, test, target PNG, submission, or P8 artifact was accessed.

The measured bottleneck is repeated solve_buddies_from_scores calls, not candidate propagation. Any future relaxation test must first use an inexpensive rank/coverage gate or batch/accelerated decoder before full paired placement evaluation. Evidence: P36_G2_STOPPED_TASK.log.

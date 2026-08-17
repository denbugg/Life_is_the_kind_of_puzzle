# P29 DPCG-24 — G2 PASS

Dense frozen DINOv2 boundary descriptors achieved a non-duplicate candidate-generation lift on the locked 128 FIT source set. With the fixed width-128 union, M=64 yielded true-neighbor coverage **22.256425%** versus **14.139988%** for frozen rank96 candidates, a gain of **+8.116437 pp**. M=16 also passed (+2.344104 pp), and M=32 gave +4.807801 pp.

This passes the registered +2.0 pp coverage gate and authorizes only P29 G3: a bounded FIT-only dense/rank fusion and source-disjoint selection recall gate. No target PNG, held, CAL, DEV, or test data was opened.

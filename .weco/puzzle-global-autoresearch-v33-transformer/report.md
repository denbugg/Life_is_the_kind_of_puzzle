# V33 interim report

V33 replaces the failed 1.00M spatial CNN selector with transformers over all
576 board cells. The main 8.77M model alternates shifted 6x6 window attention
with three full-board attention layers, uses fixed 2-D position plus learned
window-relative bias, and predicts global rank along with directional seam and
cell correctness.

The full T-S -> T-M -> T-MC experiment is running on the RTX 4060. It reuses all
V32 caches and performs group-disjoint OOF evaluation before locked validation.

# V33 interim report

V33 replaces the failed 1.00M spatial CNN selector with transformers over all
576 board cells. The main 8.77M model alternates shifted 6x6 window attention
with three full-board attention layers, uses fixed 2-D position plus learned
window-relative bias, and predicts global rank along with directional seam and
cell correctness.

The full T-S -> T-M -> T-MC experiment completed on the RTX 4060. T-M produced
a small OOF gain (`0.3143464` vs `0.3134581`) but regressed on locked validation
(`0.3716033` vs `0.3776042`). T-MC was closer but still below baseline at
`0.3766984`; its OOF gain was only `0.0003832`. Clean/noisy selection agreement
remained extremely low, so none of the transformers is promoted.

The result falsifies the claim that more global capacity alone fixes board
selection. The current candidate pool also has very little locked-validation
headroom (`0.3790761` oracle). The next useful lever is comparative modelling
across the candidate set or using learned local errors to generate better boards,
not a still larger cell-token transformer.

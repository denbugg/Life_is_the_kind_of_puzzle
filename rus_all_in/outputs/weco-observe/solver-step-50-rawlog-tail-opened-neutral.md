# Solver step 50: raw-log alternate tail is exactly neutral on opened32

One fixed alternate protected-tail trajectory used exactly
`-right_log/-down_log` from the frozen matcher, with the same four-arm pre-tail
start, protected edges, `max_swaps=96`, and minimum gain as the retained
original-cost control.  A target-free selector compared the two outputs by
original TASKA cost over all 1,104 bonds and retained control on ties.

Control, raw-log candidate, and selected output all reached **341.3125 pairs**,
recall **0.309159873**, and **4.75 exact tiles** on opened32.  Every candidate
layout was identical to control and the selector chose control 32/32.  The
nonnegative opened gate therefore permits one unchanged held replay, but this
is already evidence that raw log may be only a reparameterisation of TASKA
cost.

# TASKA all-bond cyclic-origin screen

Status: **closed on opened32**.

This bounded exact-oriented screen started from the retained four-arm
all-bond selector plus protected tail96.  It enumerated all `24×24` global
cyclic rolls and selected the layout with the smallest original TASKA seam
cost over the 1,104 non-wrapping board bonds.  The rule is target-free, has no
weight or threshold, preserves a strict permutation of all original upright
tiles, and changes only the global origin/cuts.

On opened32 it selected the unchanged origin on 31/32 boards.  The only changed
board rolled two columns and lost eight satisfied pairs and one exact tile.
The panel means changed as follows:

| Metric | Four-arm + tail96 | Cyclic all-bond choice | Delta |
|---|---:|---:|---:|
| satisfied pairs | 341.3125 | 341.0625 | -0.2500 |
| adjacency recall | 0.309159873 | 0.308933424 | -0.000226449 |
| exact tiles | 4.75000 | 4.71875 | -0.03125 |

The sensitive opened gate failed on both pair and exact metrics, so held/fresh
were not opened.  Do not sweep seam weights or nearby cut penalties.  A future
origin experiment needs genuinely new border/semantic evidence rather than the
same all-bond objective already used by layout selection and tail polishing.

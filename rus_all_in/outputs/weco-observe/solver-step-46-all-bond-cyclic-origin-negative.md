# Solver step 46: original-cost cyclic origin is negative

Enumerate every global 24x24 cyclic roll of the fixed four-arm+tail96 layout
and choose the minimum original TASKA all-1104-bond seam cost, with stable
row-major ties.  This target-free origin rule changed only 1/32 opened boards.

Opened32 reached **341.0625 pairs**, recall **0.308933424**, and **4.71875 exact
tiles**, versus 341.3125 / 0.309159873 / 4.75 for the unchanged origin.  The
only changed board lost eight pairs and one exact tile.  The gate failed; no
held transfer or parameter sweep follows.

# V33 decision card

The CNN failure was not a lack of local signal: seam loss improved while board
selection regressed. V33 therefore allocates capacity to interactions between
distant board regions. The main candidate is a 6.4M hybrid transformer with
shifted 6x6 windows and sparse full-board layers. It receives the same 32 planes
and the same clean/noisy views, so the comparison isolates architecture.

Success remains defined by selected adjacency, not training loss. Relative 2-D
position, group CV, no tile IDs, candidate shuffling and baseline fallback limit
small-data shortcuts.

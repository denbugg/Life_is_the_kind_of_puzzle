# Task

Strongly improve complete 24×24 puzzle assembly beyond V30. The primary metric is
mean true adjacency on held-out scenes; coverage must remain 1.0 and the secondary
metric is `adjacency + 0.25 × translation-aligned placement`. V30 is the frozen
baseline (`10.5737%` adjacency, `0.111061` composite on 15 scenes). The run is an
isolated Assistant Scientist autoresearch batch. No production solver changes are
allowed before a named hypothesis exists in `PLAN.md` and passes evaluation.

The main unknown is whether the next gain comes from a better learned objective,
better candidate generation, or a stronger discrete search/selection policy.

## Champion by candidate coverage
- R2 is provisional R@20 champion: 0.397758 versus R0 0.352468 (+0.045290).
- It is not a scientific pass: R@1=0.059047, R@5=0.184556, and b128 neighbour=0.040308 all remain below gate.
- R1 refuted naive untrained multi-band cosine fusion: R@20=0.259964 (-0.092504 vs R0).
## Next lever
- Train a pairwise cross-encoder with same-image hard negatives, then use it as a reranker of R2 top-K retrieval.

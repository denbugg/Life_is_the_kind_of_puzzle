# Deep research

## Paired corruption consistency

- Mean Teacher trains a student on a perturbed view against an EMA teacher on a
  second view.  Transfer: clean teacher/noisy student side embeddings and match
  logits, with a ramped consistency weight to avoid early confirmation error.
  Source: https://arxiv.org/abs/1703.01780
- AugMix aligns clean and augmented predictions with Jensen-Shannon divergence.
  Transfer only photometric/noise/blur/JPEG operations; geometry-changing ops are
  invalid for directional 20x20 seams. Source: https://arxiv.org/abs/1912.02781
- VICReg explicitly preserves feature variance while aligning paired views.
  Transfer as a small side-embedding anti-collapse term if simple cosine
  consistency suppresses discriminative seam texture.
  Source: https://arxiv.org/abs/2105.04906
- Corruption augmentation tends to transfer best to perceptually similar
  corruptions.  The supplied challenge ranges should therefore be the main
  distribution, with one unseen corruption family reserved for robustness QA.
  Source: https://arxiv.org/abs/2102.11273

## Spatial global assessment

- ERL-MPP combines local puzzle-status heads with a global discriminator and
  learned swap evaluation.  Transfer: a 24x24 board tensor, local seam-error
  heads, global adjacency/ranking head, and use the error map only to guide the
  existing permutation-safe LNS. Source: https://arxiv.org/abs/2504.09608

## Concrete implications

- Train the true pixel pipeline on paired clean/noisy bytes; do not fake the
  main experiment with score-matrix jitter.
- Keep the supervised clean loss and add noisy supervision plus EMA consistency.
- Generate near-miss boards to give the spatial critic local supervision, but
  cap synthetic boards at 50% of a batch.
- Validate selector quality with group-disjoint OOF selected adjacency,
  pairwise accuracy, Spearman correlation, and clean/noisy selection agreement.

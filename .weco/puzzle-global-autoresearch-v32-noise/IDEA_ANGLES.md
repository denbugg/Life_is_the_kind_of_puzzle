# V32 idea angles

- **A — optimization:** AdamW `3e-4`, AMP, EMA ramp, scene-balanced batches.
- **B — regularization:** clean/noisy consistency, optional VICReg anti-collapse.
- **C — architecture:** 32-plane 24x24 residual CNN with local and global heads.
- **D — data:** exact per-tile challenge corruption and reproducible replicas.
- **E — objective:** RankNet + adjacency Huber + seam/cell supervision.
- **F — efficiency:** cache each pixel-scored clean/noisy matrix once; train the
  critic from compact tensors with AMP.
- **G — cross-domain:** teacher/student corruption consistency from robust vision.
- **H — scaling:** 0.82M -> 0.98M -> conditional 1.16M critic.
- **I — open source:** reuse the repository's calibrated `distort_frags` rather
  than maintain a second degradation implementation.
- **J — antithesis:** more parameters alone may hurt; require a size-specific
  gain before promoting 1.16M.
- **K — scale-first:** two noisy replicas per training scene before increasing
  critic width further.

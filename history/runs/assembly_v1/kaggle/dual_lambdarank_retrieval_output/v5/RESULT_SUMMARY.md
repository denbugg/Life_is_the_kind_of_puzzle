# Dual-sided seven-origin LambdaRank retrieval gate v5

Decision: **stop without assembly or V4 access**.

The source-disjoint retrieval screen completed successfully on two Tesla T4s
with LightGBM 4.6.0.  The ranker produced a useful but panel-uneven signal.

| Panel | Delta R1 vs strongest baseline | Delta R5 | Delta MRR | Gate |
|---|---:|---:|---:|---|
| primary Kornia | +0.00600091 | -0.00135870 | +0.00479858 | fail |
| independent libjpeg | +0.01143569 | +0.00486866 | +0.00882118 | pass |

Absolute dual-ranker metrics were R1 `0.215353`, R5 `0.389832`, MRR
`0.304404` on Kornia and R1 `0.212070`, R5 `0.386096`, MRR `0.300422` on
libjpeg.  Candidate recall remained `0.740716` / `0.735281` and destination
collisions fell substantially, but the frozen gate required at least `+0.01`
R1 and non-regressing R5 on **both** panels.

The external assembly gate was therefore not opened; no V4 path was
constructed or read, and the models are not safe for submission.

- report SHA256: `ff9db95ea02aa8529415eae09ba7dead325f93d6f37f99bb476753bca6294ee3`
- wrapper SHA256: `36cdc6800afa15c39a8532298fd1b1ed67a62cc02a7491b6a337bc581129fea5`
- outgoing model SHA256: `e7c53b80c100c3705e465a45e70bcee0eee2f72d40c44a6992b479eb19be1963`
- incoming model SHA256: `267abde975348b194eb1b2eb9c61f57c524abcbc518e34c0cc1874bbcb1dd9c1`
- pinned code-tree SHA256: `0832c395363b7795913f8af3362f37e84d7bf966d3eff14f81ba537e758d66cd`

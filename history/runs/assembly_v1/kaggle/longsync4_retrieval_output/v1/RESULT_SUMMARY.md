# LongSync-4 retrieval diagnostic — final result

Status: **retired; decisive negative retrieval signal**.

The frozen sparse translation-group LongSync-4 diagnostic completed on Kaggle
kernel `pasha883/vsos-longsync4-retrieval-t4x2`, version 1, using two Tesla T4
devices. The evaluator processed exactly 8 whole-source images from
`edge_development[316:324]` under both `primary_kornia` and
`independent_libjpeg` (16 source-panel records). The source-name fingerprint is
`c0f9548268a4e72a07e987cfdedf98047313e61967758401b297ff60f82ff7c7`.

No assembly target was opened, no layout was built, and this output is not a
submission.

## Frozen result

| Panel | Mean delta AP | Mean delta R1 | Mean delta MRR | R1 wins/ties/losses |
|---|---:|---:|---:|---:|
| `primary_kornia` | -0.028526 | -0.014431 | -0.007216 | 0 / 0 / 8 |
| `independent_libjpeg` | -0.034395 | -0.018555 | -0.009277 | 0 / 0 / 8 |

The mean R5 delta was exactly zero on both panels, while every other primary
gate condition failed. The frozen decision is `stop_no_retrieval_signal`.

This was not a coverage failure. LongSync had 4-cycle support for about 76.5%
of canonical edges on Kornia and 78.0% on libjpeg, and could compare about
58.1% and 59.3% of outgoing query groups. It made roughly 265 and 275 top-2
swaps per source respectively. Those swaps were systematically harmful:
high-confidence accepted-edge precision also fell by about 0.127 and 0.159.

The result supports the predicted failure mode: in the noisy top-2 hypothesis
graph, accidental translation-closed 4-cycles are common enough that plain
cycle consistency is anti-correlated with the correct neighbour choice. Do not
repeat this branch with larger top-k, beta sweeps, or a mixing coefficient on
the current candidate graph. Reconsider LongSync only if a future producer
first delivers a substantially higher-precision multi-hypothesis graph.

## Reproducibility

- Report: `longsync4_retrieval_report.json`
- Report SHA-256: `16e4e3f1b95eb4be5c7c6b03c301786a4458796c69f870892d072d7eadb29dee`
- Wrapper: `longsync4_retrieval_wrapper.json`
- Wrapper SHA-256: `4ccd30e8c39b9e5bed652f7a9c41deca1ad037839c629290029598513131b91a`
- Evaluator SHA-256: `78a3e7019d84d669bf514b0343da9109a226c10fca1c6c1e7680d3d60dede6d3`
- LongSync core SHA-256: `7592d1e5c986d430aba5139abdfb5774804e4c63b8d05ae2ba97dbb56744871f`
- Frozen HGB SHA-256: `c5929a76c843f7541119f622bf1c5b6774006ad79e3811407e36edfe60bd0f10`

Kaggle emitted a scikit-learn persistence warning because the HGB artifact was
created with 1.9.0 and read with 1.6.1. The comparison remains paired within a
single prediction vector, and the negative effect is large, consistent on
16/16 records, and accompanied by a large accepted-precision loss. This warning
does not justify spending another target or GPU gate on the branch.

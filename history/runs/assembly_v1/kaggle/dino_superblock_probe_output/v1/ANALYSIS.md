# Frozen-DINOv2 4x4 superblock probe v1

Decision: **development kill-gate failed; do not open real16 and do not start
fragment positional diffusion or GANzzle-style retrieval**.

The self-contained Kaggle job completed normally on 2x Tesla T4 in `905.38 s`.
It trained only the small set-to-position head on 512 whole sources, evaluated
64 disjoint development sources and eight further exact sources, then stopped
before reading any real16 target.

## Development64 gate

| Metric | QAP blocks | DINO assignment | Required |
|---|---:|---:|---:|
| coarse-cell accuracy | 0.028646 | 0.044705 | DINO >= 0.10 |
| mean coarse Manhattan | 3.816840 | 3.519097 | — |
| aggregate Manhattan reduction | — | 0.078008 | >= 0.25 |

Chance coarse-cell accuracy is `1/36 = 0.027778`. DINO is above chance, but
only weakly and far below the fixed 10% transfer gate. The mean of per-source
Manhattan reductions is `0.068423`; the gate's ratio of aggregate means is
`0.078008`. Both are far below 25%.

## Exact8 transfer

| Metric | QAP blocks | DINO assignment |
|---|---:|---:|
| coarse-cell accuracy | 0.052083 | 0.062500 |
| mean coarse Manhattan | 3.736111 | 3.520833 |
| aggregate Manhattan reduction | — | 0.057621 |
| mean per-source wrong-position reduction | — | -0.002626 |

The head's training token accuracy rose monotonically to `0.2400` by epoch 12,
while development assignment accuracy stayed at `0.0447`. This is strong scene
overfit: larger 4x4 fragments contain a little transferable top/bottom semantic
information, but not enough image-specific absolute-position signal to move
QAP fragments safely. Exact tile placement became slightly worse despite the
small coarse Manhattan gain.

The predeclared stop-rule therefore applies: do not scale feature extraction,
do not train positional diffusion on these block features, and do not build a
GANzzle-style latent mental image whose retrieval prerequisite has failed.

## Integrity

- status: `development_gate_failed`
- accepted: `false`
- real16 targets opened: `false`
- probe report SHA-256: `0d1e95b7ff5635642907936c26b1f4055decebc645c7c9d8c4aad816b0969555`
- wrapper SHA-256: `7ffb53bad127e696caf68b298164774efcf085982e9807d3cab1d44972708d26`
- embedded payload SHA-256: `64045be1bcb26926704d10daf34b0668baf2db54ed9d0fc91da56aebfd4b96f3`
- authoritative compact manifest SHA-256: `92233fc5343aac3049ce0327417b645998bf477c6db91a4a852659312949ced6`

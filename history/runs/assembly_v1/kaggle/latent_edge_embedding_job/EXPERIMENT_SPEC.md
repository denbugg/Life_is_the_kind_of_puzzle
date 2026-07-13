# TileNAF latent edge Stage-1

## Hypothesis

The selected frozen TileNAF decoder exposes a 48-channel full-resolution
representation immediately before its RGB residual head. A compact side
Transformer can learn a more useful edge space from these latents plus raw RGB,
restored RGB, residual RGB, and restored Sobel features than the current HBT
learns from RGB/Sobel alone.

The model remains factorized and cheap: four normalized directional embeddings,
about 1.68M trainable parameters, and one dense dot-product score per direction.
It is not the failed 26.5M full-pair Transformer or the failed all-pairs residual
CNN.

## Training contract

- frozen restorer: `selected_tilenaf_synth_50k.pt`;
- frozen proposal model: `hbt_d320_denoised_rgb_sobel.pt`;
- train: `edge_train[4096:4352]`, 256 whole source images, two epochs;
- primary and independent-libjpeg panels are interleaved inside every epoch;
- primary loss: listwise CE inside frozen HBT top-64 proposals;
- auxiliary loss: all-575 hard-triplet/listwise objective for uncovered truth
  and global embedding regularity;
- exact 2-process DDP on 2x Tesla T4 sm75, fp32 after bounded fp16 smoke showed
  non-finite Transformer gradients.

## Evaluation contract

- smoke only: previously exposed `edge_development[96]` and `[112]`;
- alpha selection: fresh `assembly_incremental_gate[192:208]`;
- holdout, opened only after selection passes:
  `assembly_incremental_gate[208:224]`;
- candidates: frozen C1 top-32, HBT top-32, and W4 top-32 union capped at 64;
- outside that union, W4 is copied bit-for-bit;
- `alpha=0` is an exact identity baseline.

Each panel must independently pass mean R@1/MRR deltas, R@5/R@32 safety,
candidate coverage, paired source bootstrap lower bounds, win fraction, and
worst-source regression. Stage-1 never runs QAP. QAP/SSIM is a separate stage
that is authorized only by a passed selection and holdout retrieval gate.

## Launch audit

- v1: no training; Kaggle expanded the uploaded overlay ZIP and the wrapper
  incorrectly searched for the container filename.
- v2: overlay/tests/DDP reached the first optimizer step; fp16 Transformer
  gradients were non-finite, so the smoke failed closed before the pilot.
- v3: hash-pinned candidate-aligned fp32 job.

The overlay is the private dataset
`pasha883/vsos-tilenaf-latent-edge-overlay-v1`; the Kaggle kernel is
`pasha883/vsos-tilenaf-latent-edge-stage1-t4x2`.

## Terminal results

Version 3 completed on two Tesla T4 GPUs in fp32. The pilot took
`318.705843 s`; the wrapper completed in about `408 s`. Training stayed finite
and the full-ranking metrics improved from epoch 1 to epoch 2:

- epoch 1: R@1 `0.046044`, MRR `0.097606`;
- epoch 2: R@1 `0.158667`, MRR `0.254240`.

On the precommitted selection slice, `alpha=0.1` was the best blend but did not
beat the strongest HBT comparator. Candidate-minus-HBT deltas were:

| Panel | R@1 delta | MRR delta |
|---|---:|---:|
| primary_kornia | `-0.009907` | `-0.006658` |
| independent_libjpeg | `-0.010813` | `-0.007396` |

The formal strongest-baseline gate therefore failed and the pilot stopped with
`status=stop_selection_retrieval`. It did, however, show a consistent gain over
the actual production W4 anchor, so one source-disjoint diagnostic was allowed
with `alpha=0.1` frozen and no retuning.

On untouched `assembly_incremental_gate[208:224]`, candidate-minus-W4 deltas
were positive on both panels:

| Panel | R@1 | MRR | R@5 | R@32 | R@1 wins |
|---|---:|---:|---:|---:|---:|
| primary_kornia | `+0.007756` | `+0.010650` | `+0.012908` | `+0.013757` | `13/16` |
| independent_libjpeg | `+0.007926` | `+0.012171` | `+0.017437` | `+0.016304` | `13/16` |

Both R@1 bootstrap lower bounds were positive (`+0.004472` and `+0.004189`),
but the frozen R@1 threshold was `+0.008` and candidate coverage was required
to be at least `0.75`. The observed coverages were `0.744452` and `0.744735`.
The terminal diagnostic status is therefore `stop_no_w4_holdout_signal`.

No QAP, SSIM evaluation, submission rendering, or production promotion was
performed. Authoritative artifacts:

- `runs/assembly_v1/kaggle/latent_edge_embedding_output/v3_complete/latent_edge_pilot/latent_edge_embedding_report.json`;
- `runs/assembly_v1/kaggle/latent_edge_embedding_output/v3_complete/production_anchor_holdout_208_224.json`;
- pilot report SHA-256
  `16279686ed179a5a0f9ceb8764ecc027500560f9d11962e4dca1727e8e14b05f`;
- checkpoint SHA-256
  `01a63b08e4f1dabb6e475b690ea6f4b57aa19f87e24d25c3d0f641972e634e9f`;
- production-anchor report SHA-256
  `87321f9e75468a39e1198b62a91979dde70661694df147f2bb7174a93afc40e1`.

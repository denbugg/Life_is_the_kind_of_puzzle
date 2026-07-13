# Pair Transformer Pilot v2

Decision: close without promotion. The branch has no positive selection signal and the full training run hit its bounded AMP safety limit.

## Configuration

- 26,507,009 trainable parameters
- 512 whole-source images per epoch
- 48 queries per source, 31 negatives, four groups per optimizer step
- HBT/current-layout candidate union, cached 576-tile CNN features
- iterative neural rescoring plus equal-budget QAP controls
- two Tesla T4 GPUs

## Completed epochs

| Epoch | Train loss | HBT recall@1 | Neural recall@1 | Delta | AMP skips |
|---|---:|---:|---:|---:|---:|
| 1 | 3.778381 | 0.199275 | 0.188406 | -0.010870 | 0 |
| 2 | 3.725445 | 0.199275 | 0.187047 | -0.012228 | 2 |

Epoch 3 reached source 192/256. Dynamic loss scaling had already recovered four bounded non-finite updates; the fifth triggered the intended fail-closed abort. No final calibration/QAP report was produced.

## Preserved artifacts

- `pair_transformer_pilot/pair_transformer_best.pt`: epoch-1 inference checkpoint, SHA256 `3c76213fc9ccb960cb7d3171584232af53edd5c00b287e3bef08b03a6a280050`, 106,169,147 bytes.
- `pair_transformer_pilot/pair_transformer_latest.pt`: completed epoch-2 exact-resume checkpoint, SHA256 `f94704a185645e76cad3b2e5eeb63ef90cfafb116d1fcab822aaf7cabccc4070`, 318,381,091 bytes.
- `pair_transformer_pilot_wrapper.json`: SHA256 `62e3311a02a1702b82dc92e73d44aee61a4448b6bcaef532409a7ea4c27022a6`.
- `vsos-pair-transformer-pilot-t4x2.log`: SHA256 `11ea9e00817d363072ff3f379ccc70e862d2aa4556c7c4e2fe89bc20fdacf8de`.

The frozen best checkpoint technically supports `--action evaluate`, but the unchanged full gate requires up to 3.74 million pair scores and 128 QAP invocations, with a 30–150 minute estimate on one T4. Running it after two negative selection epochs and an AMP abort would violate the no-signal pivot rule, so no remote evaluation was launched.

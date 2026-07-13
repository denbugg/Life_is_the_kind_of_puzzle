# Frozen Pair Transformer Evaluation Specification

Status: specified but deliberately not launched under the no-signal pivot rule.

## Frozen checkpoint

- path: `pair_transformer_pilot/pair_transformer_best.pt`
- SHA256: `3c76213fc9ccb960cb7d3171584232af53edd5c00b287e3bef08b03a6a280050`
- size: `106169147` bytes
- selected epoch: `1`
- selection metric: quick exact primary recall@1 delta over HBT
- observed delta: `-0.0108695652`
- schema/kind: `1` / `puzzle_full_tile_pair_transformer`
- `safe_for_submission=false`
- strict current-model load: passed, no missing or unexpected keys

## Evaluation-only command

Run on one visible T4 after mounting the frozen checkpoint as a read-only Kaggle dataset. This command performs no training or resume:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/kaggle/working/pair_transformer_code/src \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/usr/bin/python3 \
  /kaggle/working/pair_transformer_code/scripts/train_evaluate_pair_transformer.py \
  --action evaluate \
  --checkpoint /kaggle/input/datasets/pasha883/vsos-pair-transformer-best-epoch1/pair_transformer_best.pt \
  --data-root /kaggle/input/datasets/pasha883/vsos-ai-initiative-pazzle \
  --manifest /kaggle/working/pair_transformer_code/configs/denoise_splits_seed20260710.json \
  --quarantine /kaggle/working/pair_transformer_code/configs/denoise_validation_quarantine_v1.json \
  --denoiser /kaggle/input/datasets/pasha883/vsos-assembly-v1-runtime/selected_tilenaf_synth_50k.pt \
  --hbt-checkpoint /kaggle/input/datasets/pasha883/vsos-assembly-v1-runtime/hbt_d320_denoised_rgb_sobel.pt \
  --pseudo-gold /kaggle/input/datasets/pasha883/vsos-real-gold-512/real_gold_train_512.npz \
  --seed 20260711 \
  --train-sources 512 --quick-val-sources 2 \
  --calibration-sources 4 --validation-sources 8 --validation-replicas 2 \
  --solver-sources 4 --panels primary_kornia,independent_libjpeg \
  --model-dim 512 --layers 8 --heads 8 --feedforward-dim 2048 \
  --cnn-channels 128 --patch-grid 5 --side-band 6 --band-tokens 10 \
  --dropout 0.10 --candidate-top-k 48 --candidate-reverse-top-k 8 \
  --pair-batch-size 512 --neural-blend 0.75 --iterative-passes 2 \
  --qap-iterations 12 --qap-restarts 1 \
  --denoise-batch-size 512 --chunk-size 64 \
  --output-dir /kaggle/working/pair_transformer_eval_epoch1
```

## Cost and decision

The unchanged gate comprises 56 cached-tile neural passes, up to about 3.74 million pair scores, and 128 QAP invocations. The defensible runtime reserve is 30–150 minutes, potentially up to 2.5 hours on one T4. Because epoch 1 and epoch 2 already underperform HBT and epoch 3 hit the bounded AMP safety limit, this evaluation was not launched.

If a future user explicitly chooses to spend that quota, treat the run as a falsification gate only. A passing synthetic report would still require an untouched seed and frozen real-layout gate before promotion.

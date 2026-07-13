# Denoise V2 unified release bundle

This compact hand-off contains the selected synthetic-50k checkpoint,
leakage-safe split/provenance, source code, tests, Kaggle entrypoints, derived
real-pair supervision, baseline, decision-bearing reports/logs, visual QA and a
full-frame integration example.

Final frozen-gate result (350 untouched sources, 2800 pairs per panel):

- primary source-macro RGB SSIM: 0.81097747 vs legacy 0.77100406;
- primary delta: +0.03997341, 95% CI [+0.03810371, +0.04184709];
- sensitivity SSIM: 0.79937019 vs legacy 0.75684379;
- all six precommitted lower-bound checks passed.

`artifacts/model/selected_tilenaf_synth_50k.pt` is the only selected model.
`artifacts/not_promoted/real_finetune_rollback_safe_step0.pt` is audit evidence:
the fine-tune did not meet its precommitted +0.003 threshold. Unsafe/latest,
candidate and duplicate checkpoints are omitted. Raw `puzzle/` images, `.conda`
and unrelated submission/assembly outputs are also excluded. No assembly model
was trained.

Verify after extraction from the directory containing this README:

    shasum -a 256 -c SHA256SUMS

Recreate the environment from `reproducibility/environment.yml`, then run tests
from the extracted bundle root:

    PYTHONPATH=code:code/src python -m pytest -q code/tests

Example inference:

    PYTHONPATH=code/src python code/scripts/apply_denoise_v2.py \
      --checkpoint artifacts/model/selected_tilenaf_synth_50k.pt \
      --input /path/to/input.png \
      --output /path/to/restored.png

See `docs/DENOISE_V2.md` for the protocol and `MANIFEST.json` for every file's
role, byte size, source-relative path and SHA256.

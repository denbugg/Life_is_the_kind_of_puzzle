"""Build an independent frozen holdout, excluding every stem in an earlier cache."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

import evaluate_submit_border_pipeline as pipeline

COUNT = int(os.getenv("CASE_COUNT", "128"))
OUTPUT = Path(os.getenv("CASE_FILE", "source_aware_holdout_128.npz"))
EXCLUDE = Path(os.getenv("EXCLUDE_CASE_FILE", "border_solver_ssim_cases_64.npz"))
HOLDOUT_SEED = int(os.getenv("HOLDOUT_SEED", "20260819"))


def main() -> None:
    device = torch.device("cuda")
    models = pipeline.load_models(device)
    maps_data = np.load(pipeline.MAP_FILE)
    stems, maps = maps_data["stems"], maps_data["maps"]
    excluded = set(map(str, np.load(EXCLUDE)["stems"]))

    order = np.arange(len(stems))
    np.random.default_rng(pipeline.SEED).shuffle(order)
    grouped_validation = order[-max(100, len(order) // 10):]
    available = np.asarray([index for index in grouped_validation if str(stems[index]) not in excluded])
    rng = np.random.default_rng(HOLDOUT_SEED)
    chosen = rng.choice(available, size=min(COUNT, len(available)), replace=False)

    rights, downs, positions = [], [], []
    restored_tiles, targets, truths, names = [], [], [], []
    for index, source_index in enumerate(chosen):
        stem = str(stems[source_index])
        raw = pipeline.split(pipeline.DATA_ROOT / "inputs" / f"{stem}.png")
        restored = pipeline.restore(models[0], raw, device)
        target = np.asarray(Image.open(pipeline.DATA_ROOT / "targets" / f"{stem}.png").convert("RGB"), np.uint8)
        rights.append(pipeline.ranker_matrix(models[1], restored, 0, device))
        downs.append(pipeline.ranker_matrix(models[1], restored, 1, device))
        positions.append(pipeline.position_matrix(models[2], restored, device))
        restored_tiles.append(restored)
        targets.append(target)
        truths.append(maps[source_index].astype(np.int32))
        names.append(stem)
        print({"done": index + 1, "total": len(chosen), "stem": stem}, flush=True)

    np.savez_compressed(
        OUTPUT,
        right=np.asarray(rights, np.float32),
        down=np.asarray(downs, np.float32),
        pos=np.asarray(positions, np.float32),
        restored=np.asarray(restored_tiles, np.uint8),
        target=np.asarray(targets, np.uint8),
        truth=np.asarray(truths, np.int32),
        stems=np.asarray(names),
    )
    print({"output": str(OUTPUT), "cases": len(names), "excluded": len(excluded)}, flush=True)


if __name__ == "__main__":
    from PIL import Image

    main()

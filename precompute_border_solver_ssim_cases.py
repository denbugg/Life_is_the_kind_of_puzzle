"""Freeze restored tiles, model scores, and clean targets for solver SSIM tuning."""
import os
from pathlib import Path

import numpy as np
import torch

import evaluate_submit_border_pipeline as pipeline

COUNT = int(os.getenv("CASE_COUNT", "12"))
OUTPUT = Path(os.getenv("CASE_FILE", "border_solver_ssim_cases.npz"))


def main():
    device = torch.device("cuda")
    models = pipeline.load_models(device)
    maps_data = np.load(pipeline.MAP_FILE)
    stems, maps = maps_data["stems"], maps_data["maps"]
    order = np.arange(len(stems))
    np.random.default_rng(pipeline.SEED).shuffle(order)
    validation = order[-max(100, len(order) // 10):]
    chosen = validation[
        np.linspace(7, len(validation) - 8, min(COUNT, len(validation)), dtype=int)
    ]

    rights, downs, positions = [], [], []
    restored_tiles, targets, truths, names = [], [], [], []
    for index, source_index in enumerate(chosen):
        stem = str(stems[source_index])
        raw = pipeline.split(pipeline.DATA_ROOT / "inputs" / f"{stem}.png")
        restored = pipeline.restore(models[0], raw, device)
        target = np.asarray(
            pipeline.Image.open(
                pipeline.DATA_ROOT / "targets" / f"{stem}.png"
            ).convert("RGB"),
            np.uint8,
        )
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
    print({"output": str(OUTPUT), "cases": len(names)}, flush=True)


if __name__ == "__main__":
    main()

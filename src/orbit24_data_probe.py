from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

GRID = 24
TILE = 20
TILES = GRID * GRID


def tiles(path: Path) -> np.ndarray:
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    if image.shape != (480, 480, 3):
        raise ValueError(f"{path}: expected 480x480x3, found {image.shape}")
    return np.ascontiguousarray(image.reshape(GRID, TILE, GRID, TILE, 3).transpose(0, 2, 1, 3, 4).reshape(TILES, TILE, TILE, 3))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images", type=int, default=2)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import train_e26_contextual_edge as e26

    sources = e26.load_authenticated_training_sources(args.source_manifest)
    split = e26.split_source_groups(sources.names, sources.group_for_name, mapping_sha256=sources.mapping_sha256)
    result = {"schema": "orbit24-data-probe-v1", "records": []}
    for name in split.development_names[: args.images]:
        x = tiles(args.input_root / name)
        y = tiles(args.target_root / name)
        x_flat = x.reshape(TILES, -1).astype(np.int16)
        y_flat = y.reshape(TILES, -1).astype(np.int16)
        exact = int(sum(1 for index in range(TILES) if np.any(np.all(y_flat == x_flat[index], axis=1))))
        # Exact global RGB inventory and nearest-tile MSE diagnostic. This is read-only and bounded.
        candidates = []
        for index in range(min(8, TILES)):
            mse = np.mean((y_flat - x_flat[index]) ** 2, axis=1)
            candidates.append({"input_tile": index, "best_target_tile": int(np.argmin(mse)), "min_mse": float(np.min(mse))})
        result["records"].append({
            "name": name,
            "input_global_mean": [float(v) for v in x.mean(axis=(0, 1, 2))],
            "target_global_mean": [float(v) for v in y.mean(axis=(0, 1, 2))],
            "input_global_std": [float(v) for v in x.std(axis=(0, 1, 2))],
            "target_global_std": [float(v) for v in y.std(axis=(0, 1, 2))],
            "exact_input_tiles_found_in_target": exact,
            "first_eight_nearest_mse": candidates,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Reuse a prior submission's layout while restoring the original noisy test tiles."""
import json
import os
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
import torch

from train_real_tile_restorer import FragmentRestorer, GRID, TILE, N

RAW_DIR = Path(os.environ["RAW_DIR"])
LAYOUT_ZIP = Path(os.environ["LAYOUT_ZIP"])
CHECKPOINT = Path(os.environ["CHECKPOINT"])
OUT_DIR = Path(os.getenv("OUT_DIR", "submission_real_restorer"))
OUT_ZIP = Path(os.getenv("OUT_ZIP", "submission_real_restorer.zip"))


def split_array(image):
    return image.reshape(GRID, TILE, GRID, TILE, 3).transpose(0, 2, 1, 3, 4).reshape(N, TILE, TILE, 3)


def assemble(tiles):
    return tiles.reshape(GRID, GRID, TILE, TILE, 3).transpose(0, 2, 1, 3, 4).reshape(480, 480, 3)


def features(tiles):
    x = tiles.astype(np.float32) / 255.0
    low = x.reshape(N, 5, 4, 5, 4, 3).mean((2, 4))
    gray = low.mean(3)
    gray = (gray - gray.mean((1, 2), keepdims=True)) / (gray.std((1, 2), keepdims=True) + 1e-5)
    dx = np.diff(gray, axis=2, append=gray[:, :, -1:])
    dy = np.diff(gray, axis=1, append=gray[:, -1:, :])
    feat = np.concatenate([gray.reshape(N, -1), 0.35 * dx.reshape(N, -1), 0.35 * dy.reshape(N, -1)], 1)
    return feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-6)


@torch.inference_mode()
def main():
    device = torch.device("cuda")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = FragmentRestorer().to(device).eval()
    model.load_state_dict(checkpoint["model"])
    files = sorted(RAW_DIR.glob("*.png"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    costs = []
    with zipfile.ZipFile(LAYOUT_ZIP) as prior:
        prior_names = set(prior.namelist())
        for index, path in enumerate(files):
            if path.name not in prior_names:
                raise KeyError(f"missing prior layout: {path.name}")
            raw = split_array(np.asarray(Image.open(path).convert("RGB"), np.uint8))
            with prior.open(path.name) as stream:
                layout_reference = split_array(np.asarray(Image.open(stream).convert("RGB"), np.uint8))
            cost = 2.0 - 2.0 * np.clip(features(layout_reference) @ features(raw).T, -1, 1)
            positions, raw_ids = linear_sum_assignment(cost)
            order = np.empty(N, np.int64); order[positions] = raw_ids
            ordered = raw[order]
            tensor = torch.from_numpy(np.ascontiguousarray(ordered.transpose(0, 3, 1, 2))).float().div_(255)
            restored = []
            for start in range(0, N, 256):
                restored.append(model(tensor[start:start + 256].to(device)).cpu())
            tiles = torch.cat(restored).permute(0, 2, 3, 1).mul(255).round().clamp(0, 255).byte().numpy()
            Image.fromarray(assemble(tiles)).save(OUT_DIR / path.name, optimize=False)
            costs.append(float(cost[positions, raw_ids].mean()))
            if (index + 1) % 25 == 0:
                print(json.dumps({"done": index + 1, "total": len(files), "mean_match_cost": float(np.mean(costs))}), flush=True)
    with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(OUT_DIR / path.name, arcname=path.name)
    print(json.dumps({"submission": str(OUT_ZIP), "files": len(files),
                      "mean_match_cost": float(np.mean(costs)),
                      "checkpoint_epoch": checkpoint.get("epoch"),
                      "checkpoint_metrics": checkpoint.get("metrics")}), flush=True)


if __name__ == "__main__":
    main()

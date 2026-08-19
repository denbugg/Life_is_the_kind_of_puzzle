"""Restore all real noisy train tiles and save them in target-position order."""
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from train_real_tile_restorer import FragmentRestorer, split, assemble

DATA_ROOT = Path(os.getenv("DATA_ROOT", "data/real/train"))
MAP_FILE = Path(os.getenv("MAP_FILE", "real_tile_maps.npz"))
CHECKPOINT = Path(os.getenv("CHECKPOINT", "outputs_real_restorer/real_fragment_restorer_best.pt"))
OUT_DIR = Path(os.getenv("OUT_DIR", "data/real/restored_target_order"))
MAX_IMAGES = int(os.getenv("MAX_IMAGES", "0"))
BATCH = int(os.getenv("RESTORE_BATCH", "256"))


@torch.inference_mode()
def main():
    device = torch.device("cuda")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = FragmentRestorer().to(device).eval(); model.load_state_dict(checkpoint["model"])
    z = np.load(MAP_FILE); stems, maps = z["stems"], z["maps"]
    if MAX_IMAGES:
        stems, maps = stems[:MAX_IMAGES], maps[:MAX_IMAGES]
    OUT_DIR.mkdir(parents=True, exist_ok=True); done = 0; skipped = 0
    for index, (stem_value, mapping) in enumerate(zip(stems, maps)):
        stem = str(stem_value); output = OUT_DIR / f"{stem}.png"
        if output.exists():
            skipped += 1; continue
        tiles = split(DATA_ROOT / "inputs" / f"{stem}.png")[mapping]
        tensor = torch.from_numpy(np.ascontiguousarray(tiles.transpose(0, 3, 1, 2))).float().div_(255)
        restored = [model(tensor[i:i + BATCH].to(device)).cpu() for i in range(0, len(tensor), BATCH)]
        restored = torch.cat(restored).permute(0, 2, 3, 1).mul(255).round().clamp(0, 255).byte().numpy()
        Image.fromarray(assemble(restored)).save(output, optimize=False)
        done += 1
        if (index + 1) % 50 == 0:
            print(json.dumps({"processed": index + 1, "total": len(stems), "written": done, "skipped": skipped}), flush=True)
    manifest = {"images": len(stems), "written": done, "skipped": skipped, "checkpoint": str(CHECKPOINT),
                "checkpoint_epoch": checkpoint.get("epoch"), "checkpoint_metrics": checkpoint.get("metrics")}
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2)); print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()

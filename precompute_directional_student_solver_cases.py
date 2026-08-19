"""Cache dense directional-student scores for repeatable solver ablations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from evaluate_directional_student_full576 import score, split
from train_directional_jigsaw_transformer import DirectionalTransformer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--raw-input-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tau", type=float, default=0.10)
    args = parser.parse_args()

    source = np.load(args.cases)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device = torch.device("cuda")
    model = DirectionalTransformer().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    rights, downs, raw_tiles = [], [], []
    stems = source["stems"]
    for index, stem in enumerate(stems):
        image = np.asarray(
            Image.open(args.raw_input_dir / f"{stem}.png").convert("RGB"),
            np.uint8,
        )
        tiles = split(image)
        right, down = score(model, tiles, device, args.tau)
        rights.append(right)
        downs.append(down)
        raw_tiles.append(tiles)
        print(json.dumps({"done": index + 1, "total": len(stems), "stem": str(stem)}), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        right=np.asarray(rights, np.float32),
        down=np.asarray(downs, np.float32),
        pos=source["pos"].astype(np.float32),
        tiles=np.asarray(raw_tiles, np.uint8),
        target=source["target"].astype(np.uint8),
        truth=source["truth"].astype(np.int32),
        stems=stems,
        tau=np.asarray(args.tau, np.float32),
        checkpoint_epoch=np.asarray(checkpoint.get("epoch", -1), np.int32),
    )
    print(json.dumps({"complete": True, "cases": len(stems), "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()

"""Render old/new solver comparisons for selected frozen cases."""
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from evaluate_directional_student_full576 import assemble, score, split
from global_solver_candidate import solve_layout
from train_directional_jigsaw_transformer import DirectionalTransformer

CASES = Path("/home/kva/pazzle_source_aware_ablation/holdout128/cases.npz")
RAW = Path("/home/kva/pazzle_directional_transformer/data/real/train/inputs")
CHECKPOINT = Path("/home/kva/pazzle_directional_transformer/outputs_real_student/best.pt")
OUTPUT = Path("outputs/examples")


def main():
    device = torch.device("cuda")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = DirectionalTransformer().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    data = np.load(CASES)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for index in (21, 120):
        stem = str(data["stems"][index])
        raw = np.asarray(Image.open(RAW / f"{stem}.png").convert("RGB"), np.uint8)
        tiles = split(raw)
        right, down = score(model, tiles, device, 0.10)
        seed = 20260818 + index * 100
        old_layout = solve_layout(data["right"][index], data["down"][index], data["pos"][index], seed)
        new_layout = solve_layout(right, down, data["pos"][index], seed)
        Image.fromarray(assemble(tiles, np.asarray(old_layout))).save(OUTPUT / f"{stem}_old.png")
        Image.fromarray(assemble(tiles, np.asarray(new_layout))).save(OUTPUT / f"{stem}_new.png")
        print(stem, flush=True)


if __name__ == "__main__":
    main()

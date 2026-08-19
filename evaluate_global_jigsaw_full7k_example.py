"""Create an honest held-out example for the full-7k global jigsaw model."""
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

os.environ.setdefault("DIM", "192")
os.environ.setdefault("LAYERS", "6")

from train_global_jigsaw_transformer import (
    GRID,
    N,
    TILE,
    GlobalJigsawTransformer,
    assignment,
    split_tiles,
)

ROOT = Path(os.getenv("ROOT", "/home/kva/pazzle_global_jigsaw"))
SOURCE = Path(os.getenv(
    "SOURCE",
    "/home/kva/pazzle_rl_on_diffusion_v2/clean_targets/img_002950.png",
))
CHECKPOINT = Path(os.getenv(
    "CHECKPOINT",
    ROOT / "validation_full7k_epoch20.pt",
))
OUTPUT = Path(os.getenv(
    "OUTPUT",
    ROOT / "full7k_pipeline_example_img_002950.png",
))
METRICS = Path(os.getenv(
    "METRICS",
    ROOT / "full7k_pipeline_example_img_002950.json",
))
SEED = int(os.getenv("SEED", "20260728"))


def assemble(tiles, layout):
    return tiles[np.asarray(layout)].reshape(
        GRID, GRID, TILE, TILE, 3
    ).transpose(0, 2, 1, 3, 4).reshape(GRID * TILE, GRID * TILE, 3)


def adjacency_accuracy(layout, permutation):
    original_ids = permutation[np.asarray(layout)].reshape(GRID, GRID)
    right = original_ids[:, 1:] == original_ids[:, :-1] + 1
    down = original_ids[1:, :] == original_ids[:-1, :] + GRID
    return float((right.sum() + down.sum()) / (right.size + down.size))


def draw_grid(image):
    result = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(result)
    for value in range(0, GRID * TILE + 1, TILE):
        draw.line((value, 0, value, GRID * TILE), fill=(255, 255, 255), width=1)
        draw.line((0, value, GRID * TILE, value), fill=(255, 255, 255), width=1)
    return np.asarray(result)


def montage(items):
    panel = 420
    margin = 20
    title_h = 42
    canvas = Image.new(
        "RGB",
        (len(items) * panel + (len(items) + 1) * margin, panel + title_h + 2 * margin),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for index, (title, image) in enumerate(items):
        x = margin + index * (panel + margin)
        draw.text((x, margin + 10), title, fill="black")
        canvas.paste(Image.fromarray(image).resize((panel, panel)), (x, margin + title_h))
    return canvas


@torch.inference_mode()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = GlobalJigsawTransformer(
        dim=checkpoint["config"]["dim"],
        layers=checkpoint["config"]["layers"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    target = np.asarray(
        Image.open(SOURCE).convert("RGB").resize((GRID * TILE, GRID * TILE)),
        dtype=np.uint8,
    )
    clean_tiles = split_tiles(target)
    rng = np.random.default_rng(SEED)
    permutation = rng.permutation(N)
    shuffled_tiles = clean_tiles[permutation]
    tensor = torch.from_numpy(
        np.ascontiguousarray(shuffled_tiles.transpose(0, 3, 1, 2))
    ).float().div(127.5).sub(1).unsqueeze(0).to(device)

    row_logits, col_logits = model(tensor)
    predicted_layout = assignment(row_logits[0], col_logits[0])
    truth = np.empty(N, dtype=np.int32)
    truth[permutation] = np.arange(N)

    shuffled = assemble(shuffled_tiles, np.arange(N))
    predicted = assemble(shuffled_tiles, predicted_layout)
    exact = float(np.mean(predicted_layout == truth))
    adjacency = adjacency_accuracy(predicted_layout, permutation)
    row_acc = float(
        (row_logits[0].argmax(1).cpu().numpy() == permutation // GRID).mean()
    )
    col_acc = float(
        (col_logits[0].argmax(1).cpu().numpy() == permutation % GRID).mean()
    )
    report = {
        "source": SOURCE.name,
        "seed": SEED,
        "checkpoint_epoch": checkpoint["epoch"],
        "tile_exact": exact,
        "adjacency": adjacency,
        "row_acc": row_acc,
        "col_acc": col_acc,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    montage([
        ("Clean target", target),
        ("576 shuffled tiles", draw_grid(shuffled)),
        (f"Transformer result | exact {exact:.3%}, adj {adjacency:.3%}", draw_grid(predicted)),
    ]).save(OUTPUT)
    METRICS.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Compare full-7k jigsaw inference on clean, degraded, and DDIM-restored tiles."""
import io
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

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

ROOT = Path("/home/kva/pazzle_global_jigsaw")
DATA_ROOT = Path("/home/kva/pazzle_rl_on_diffusion_v2")
SOURCE_NAME = "img_002950"
VARIANT = 0
SHUFFLE_SEED = 20260728
DATASET_SEED = 27072027


def degrade(tile, rng):
    """Competition corruption, sampled independently for every tile."""
    array = tile.astype(np.float32) + rng.uniform(-30.0, 30.0)
    mean = array.mean(axis=(0, 1), keepdims=True)
    array = (array - mean) * rng.uniform(0.70, 1.30) + mean
    sigma = rng.uniform(40.0, 55.0)
    array += np.asarray(
        [rng.gauss(0, sigma) for _ in range(array.size)], np.float32
    ).reshape(array.shape)
    array = np.clip(array, 0, 255).astype(np.uint8)
    padded = np.pad(array.astype(np.float32), ((1, 1), (1, 1), (0, 0)), mode="reflect")
    horizontal = (padded[:, :-2] + 2 * padded[:, 1:-1] + padded[:, 2:]) * 0.25
    array = np.clip((horizontal[:-2] + 2 * horizontal[1:-1] + horizontal[2:]) * 0.25, 0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, "JPEG", quality=rng.randint(35, 50))
    array = np.asarray(Image.open(io.BytesIO(buffer.getvalue())).convert("RGB"), np.uint8)
    return array


def assemble(tiles, layout=None):
    if layout is not None:
        tiles = tiles[np.asarray(layout)]
    return tiles.reshape(GRID, GRID, TILE, TILE, 3).transpose(
        0, 2, 1, 3, 4
    ).reshape(GRID * TILE, GRID * TILE, 3)


def adjacency_accuracy(layout, permutation):
    ids = permutation[np.asarray(layout)].reshape(GRID, GRID)
    right = ids[:, 1:] == ids[:, :-1] + 1
    down = ids[1:, :] == ids[:-1, :] + GRID
    return float((right.sum() + down.sum()) / (right.size + down.size))


def add_grid(image):
    out = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(out)
    for value in range(0, GRID * TILE + 1, TILE):
        draw.line((value, 0, value, GRID * TILE), fill="white", width=1)
        draw.line((0, value, GRID * TILE, value), fill="white", width=1)
    return np.asarray(out)


def psnr(image, target):
    mse = np.mean((image.astype(np.float32) - target.astype(np.float32)) ** 2)
    return float(-10 * np.log10(max(mse / (255 ** 2), 1e-12)))


def make_montage(rows, output):
    panel, margin, title_h = 360, 18, 40
    cols = 3
    canvas = Image.new(
        "RGB",
        (cols * panel + (cols + 1) * margin,
         len(rows) * (panel + title_h) + (len(rows) + 1) * margin),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for row_i, row in enumerate(rows):
        for col_i, (title, image) in enumerate(row):
            x = margin + col_i * (panel + margin)
            y = margin + row_i * (panel + title_h + margin)
            draw.text((x, y + 10), title, fill="black")
            canvas.paste(
                Image.fromarray(image).resize((panel, panel)),
                (x, y + title_h),
            )
    canvas.save(output)


@torch.inference_mode()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        ROOT / "validation_full7k_epoch20.pt",
        map_location="cpu",
        weights_only=False,
    )
    model = GlobalJigsawTransformer(
        dim=checkpoint["config"]["dim"],
        layers=checkpoint["config"]["layers"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    clean_path = DATA_ROOT / "clean_targets" / f"{SOURCE_NAME}.png"
    restored_path = (
        DATA_ROOT / "restored_rl_targets" / f"{SOURCE_NAME}_v{VARIANT:02d}.png"
    )
    clean_image = np.asarray(
        Image.open(clean_path).convert("RGB").resize((480, 480)), np.uint8
    )
    restored_image = np.asarray(
        Image.open(restored_path).convert("RGB").resize((480, 480)), np.uint8
    )
    clean_tiles = split_tiles(clean_image)
    restored_tiles = split_tiles(restored_image)

    clean_files = sorted((DATA_ROOT / "clean_targets").glob("*.png"))
    image_index = [path.stem for path in clean_files].index(SOURCE_NAME)
    degrade_rng = random.Random(DATASET_SEED + image_index * 1009 + VARIANT)
    noisy_tiles = np.stack([degrade(tile, degrade_rng) for tile in clean_tiles])
    noisy_image = assemble(noisy_tiles)

    permutation = np.random.default_rng(SHUFFLE_SEED).permutation(N)
    truth = np.empty(N, dtype=np.int32)
    truth[permutation] = np.arange(N)
    results = {}
    assembled_results = {}
    shuffled_results = {}
    for name, tiles in (
        ("clean", clean_tiles),
        ("noisy", noisy_tiles),
        ("diffusion_restored", restored_tiles),
    ):
        shuffled = tiles[permutation]
        tensor = torch.from_numpy(
            np.ascontiguousarray(shuffled.transpose(0, 3, 1, 2))
        ).float().div(127.5).sub(1).unsqueeze(0).to(device)
        row_logits, col_logits = model(tensor)
        layout = assignment(row_logits[0], col_logits[0])
        results[name] = {
            "tile_exact": float(np.mean(layout == truth)),
            "adjacency": adjacency_accuracy(layout, permutation),
            "row_acc": float(
                (row_logits[0].argmax(1).cpu().numpy() == permutation // GRID).mean()
            ),
            "col_acc": float(
                (col_logits[0].argmax(1).cpu().numpy() == permutation % GRID).mean()
            ),
        }
        assembled_results[name] = assemble(shuffled, layout)
        shuffled_results[name] = assemble(shuffled)

    results["image_quality"] = {
        "noisy_psnr": psnr(noisy_image, clean_image),
        "diffusion_restored_psnr": psnr(restored_image, clean_image),
    }
    results["source"] = SOURCE_NAME
    results["variant"] = VARIANT
    results["checkpoint_epoch"] = checkpoint["epoch"]

    output = ROOT / "full7k_noise_diffusion_comparison_img_002950.png"
    metrics_path = ROOT / "full7k_noise_diffusion_comparison_img_002950.json"
    make_montage([
        [
            ("Clean target", clean_image),
            (f"Noisy | PSNR {results['image_quality']['noisy_psnr']:.2f}", noisy_image),
            (f"Diffusion restored | PSNR {results['image_quality']['diffusion_restored_psnr']:.2f}", restored_image),
        ],
        [
            (
                f"Clean result | exact {results['clean']['tile_exact']:.2%}, adj {results['clean']['adjacency']:.2%}",
                add_grid(assembled_results["clean"]),
            ),
            (
                f"Noisy result | exact {results['noisy']['tile_exact']:.2%}, adj {results['noisy']['adjacency']:.2%}",
                add_grid(assembled_results["noisy"]),
            ),
            (
                f"Restored result | exact {results['diffusion_restored']['tile_exact']:.2%}, adj {results['diffusion_restored']['adjacency']:.2%}",
                add_grid(assembled_results["diffusion_restored"]),
            ),
        ],
    ], output)
    metrics_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

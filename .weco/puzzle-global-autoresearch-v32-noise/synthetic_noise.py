"""Deterministic challenge-matched clean/noisy puzzle sample generator.

The corruption order and ranges mirror ``src/distort.py``.  This isolated V32
copy additionally records every per-tile draw and creates opaque random names,
which makes exported samples auditable and exactly reproducible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

GRID = 24
TILE = 20
IMAGE = 480
COUNT = 576
BRIGHTNESS = (-30.0, 30.0)
CONTRAST = (0.70, 1.30)
NOISE_SIGMA = (40.0, 55.0)
JPEG_QUALITY = (35, 50)
GRAY = np.asarray((0.299, 0.587, 0.114), np.float32)


def image_to_tiles(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.shape != (IMAGE, IMAGE, 3) or image.dtype != np.uint8:
        raise ValueError(f"expected uint8 RGB {(IMAGE, IMAGE, 3)}, got {image.shape}/{image.dtype}")
    return np.ascontiguousarray(
        image.reshape(GRID, TILE, GRID, TILE, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(COUNT, TILE, TILE, 3)
    )


def tiles_to_image(tiles: np.ndarray) -> np.ndarray:
    tiles = np.asarray(tiles)
    if tiles.shape != (COUNT, TILE, TILE, 3):
        raise ValueError(f"expected {(COUNT, TILE, TILE, 3)}, got {tiles.shape}")
    return np.ascontiguousarray(
        tiles.reshape(GRID, GRID, TILE, TILE, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(IMAGE, IMAGE, 3)
    )


def _blur3(values: np.ndarray) -> np.ndarray:
    padded = np.pad(values, ((0, 0), (1, 1), (0, 0), (0, 0)), mode="reflect")
    values = .25 * padded[:, :-2] + .50 * padded[:, 1:-1] + .25 * padded[:, 2:]
    padded = np.pad(values, ((0, 0), (0, 0), (1, 1), (0, 0)), mode="reflect")
    return .25 * padded[:, :, :-2] + .50 * padded[:, :, 1:-1] + .25 * padded[:, :, 2:]


def corrupt_tiles(clean_tiles: np.ndarray, seed: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    clean_tiles = np.asarray(clean_tiles)
    if clean_tiles.shape != (COUNT, TILE, TILE, 3) or clean_tiles.dtype != np.uint8:
        raise ValueError("clean_tiles must be uint8 [576,20,20,3]")
    rng = np.random.default_rng(seed)
    contrast = rng.uniform(*CONTRAST, size=COUNT).astype(np.float32)
    brightness = rng.uniform(*BRIGHTNESS, size=COUNT).astype(np.float32)
    sigma = rng.uniform(*NOISE_SIGMA, size=COUNT).astype(np.float32)
    x = clean_tiles.astype(np.float32)
    pivot = (x * GRAY).sum(-1, keepdims=True).mean((1, 2), keepdims=True)
    x = contrast[:, None, None, None] * (x - pivot) + pivot + brightness[:, None, None, None]
    x += rng.standard_normal(x.shape).astype(np.float32) * sigma[:, None, None, None]
    x = np.clip(_blur3(np.clip(x, 0, 255)), 0, 255).astype(np.uint8)
    quality = rng.integers(JPEG_QUALITY[0], JPEG_QUALITY[1] + 1, size=COUNT, dtype=np.int16)
    noisy = np.empty_like(x)
    for index, q in enumerate(quality):
        ok, encoded = cv2.imencode(
            ".jpg", np.ascontiguousarray(x[index, ..., ::-1]),
            [int(cv2.IMWRITE_JPEG_QUALITY), int(q)],
        )
        if not ok:
            raise RuntimeError(f"JPEG encoding failed for tile {index}")
        noisy[index] = cv2.imdecode(encoded, cv2.IMREAD_COLOR)[..., ::-1]
    return noisy, {
        "contrast": contrast,
        "brightness": brightness,
        "noise_sigma": sigma,
        "jpeg_quality": quality,
    }


def make_sample(clean_image: np.ndarray, seed: int, source_name: str) -> dict[str, object]:
    clean = image_to_tiles(clean_image)
    noisy, draws = corrupt_tiles(clean, seed)
    rng = np.random.default_rng(seed ^ 0x5A17C9E3)
    permutation = rng.permutation(COUNT).astype(np.int16)
    names = [hashlib.sha256(f"{seed}:{i}:{int(rng.integers(2**63))}".encode()).hexdigest()[:20] + ".png"
             for i in range(COUNT)]
    rows = []
    for observed, canonical in enumerate(permutation):
        rows.append({
            "filename": names[observed],
            "observed_index": observed,
            "canonical_index": int(canonical),
            "row": int(canonical // GRID),
            "column": int(canonical % GRID),
            "contrast": float(draws["contrast"][canonical]),
            "brightness": float(draws["brightness"][canonical]),
            "noise_sigma": float(draws["noise_sigma"][canonical]),
            "jpeg_quality": int(draws["jpeg_quality"][canonical]),
        })
    return {
        "clean_tiles": clean,
        "noisy_tiles": noisy,
        "observed_tiles": np.ascontiguousarray(noisy[permutation]),
        "permutation": permutation,
        "filenames": names,
        "manifest": {
            "schema": "pazzle.synthetic-noise.v1",
            "source": source_name,
            "seed": int(seed),
            "geometry": {"image": [480, 480], "grid": [24, 24], "tile": [20, 20], "count": 576},
            "corruption_order": ["contrast_brightness", "gaussian_noise", "gaussian_blur_3x3", "jpeg"],
            "ranges": {"brightness": [-30, 30], "contrast": [.70, 1.30],
                       "noise_sigma": [40, 55], "blur": [3, 3], "jpeg_quality": [35, 50]},
            "tiles": rows,
        },
    }


def save_sample(sample: dict[str, object], output: Path, export_tiles: bool = False) -> None:
    output.mkdir(parents=True, exist_ok=False)
    manifest = sample["manifest"]
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    np.savez_compressed(output / "sample.npz", clean=sample["clean_tiles"], noisy=sample["noisy_tiles"],
                        observed=sample["observed_tiles"], permutation=sample["permutation"])
    for name, tiles in (("clean", sample["clean_tiles"]), ("noisy_ordered", sample["noisy_tiles"]),
                        ("noisy_shuffled", sample["observed_tiles"])):
        cv2.imwrite(str(output / f"{name}.png"), tiles_to_image(tiles)[..., ::-1])
    if export_tiles:
        tile_dir = output / "tiles"
        tile_dir.mkdir()
        for name, tile in zip(sample["filenames"], sample["observed_tiles"]):
            cv2.imwrite(str(tile_dir / name), tile[..., ::-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=320826)
    parser.add_argument("--export-tiles", action="store_true")
    args = parser.parse_args()
    bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(args.image)
    sample = make_sample(bgr[..., ::-1], args.seed, args.image.name)
    save_sample(sample, args.output, args.export_tiles)
    print(json.dumps({"event": "sample_saved", "output": str(args.output), "seed": args.seed}))


if __name__ == "__main__":
    main()

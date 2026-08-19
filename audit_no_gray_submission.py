"""Independent archive and no-gray audit for the final guarded submission."""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


def gray_count(image: np.ndarray) -> int:
    tiles = image.reshape(24, 20, 24, 20, 3).transpose(0, 2, 1, 3, 4).reshape(576, 20, 20, 3)
    mean = tiles.mean((1, 2))
    std = tiles.std((1, 2, 3))
    return int(((mean.max(1) - mean.min(1) < 10) & (std < 25)).sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    args = parser.parse_args()
    with zipfile.ZipFile(args.zip) as archive:
        names = archive.namelist()
        bad = archive.testzip()
    raw_gray = output_gray = excess_images = 0
    shapes_ok = True
    for name in names:
        raw = np.asarray(Image.open(args.raw / name).convert("RGB"), np.float32)
        output = np.asarray(Image.open(args.images / name).convert("RGB"), np.float32)
        shapes_ok &= raw.shape == (480, 480, 3) and output.shape == (480, 480, 3)
        raw_count = gray_count(raw)
        output_count = gray_count(output)
        raw_gray += raw_count
        output_gray += output_count
        excess_images += int(output_count > raw_count)
    report = {
        "zip_count": len(names),
        "unique": len(set(names)) == len(names),
        "root_png": all("/" not in name and name.endswith(".png") for name in names),
        "testzip": bad,
        "shapes_rgb_480": bool(shapes_ok),
        "raw_gray_tiles": raw_gray,
        "output_gray_tiles": output_gray,
        "gray_delta": output_gray - raw_gray,
        "images_with_gray_excess": excess_images,
    }
    print(json.dumps(report, indent=2))
    if not (
        report["zip_count"] == 700 and report["unique"] and report["root_png"]
        and report["testzip"] is None and report["shapes_rgb_480"]
        and report["gray_delta"] <= 0 and report["images_with_gray_excess"] == 0
    ):
        raise RuntimeError("submission failed independent audit")


if __name__ == "__main__":
    main()

"""Build the frozen E15 restorer sidecar without touching the canonical cache."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from train_real_tile_restorer import FragmentRestorer


EXPECTED_CACHE_SHA256 = "74db2b62e9d5eafffae33117c7771512d823b0dcaa0095ef5807adb8e86a25df"
EXPECTED_CHECKPOINT_SHA256 = "6fcc7de2cf8063b4f2f45d4b96b8999d5eb9c29a071ff2c0031d2703c70d6695"
EXPECTED_PARAMETERS = 1_670_595
GUARD = {
    "min_restored_std": 10.0,
    "relative_std_floor": 0.72,
    "max_mean_shift": 24.0,
    "low_saturation": 10.0,
    "low_texture_std": 25.0,
    "raw_texture_floor": 10.0,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def no_gray_mask(raw: np.ndarray, restored: np.ndarray) -> np.ndarray:
    raw_f = raw.astype(np.float32)
    restored_f = restored.astype(np.float32)
    raw_std = raw_f.std((1, 2, 3))
    restored_std = restored_f.std((1, 2, 3))
    raw_mean = raw_f.mean((1, 2, 3))
    restored_mean = restored_f.mean((1, 2, 3))
    restored_rgb = restored_f.mean((1, 2))
    restored_sat = restored_rgb.max(1) - restored_rgb.min(1)
    return (
        (restored_std < np.maximum(GUARD["min_restored_std"],
                                   GUARD["relative_std_floor"] * raw_std))
        | (np.abs(restored_mean - raw_mean) > GUARD["max_mean_shift"])
        | ((restored_sat < GUARD["low_saturation"])
           & (restored_std < GUARD["low_texture_std"])
           & (raw_std >= GUARD["raw_texture_floor"]))
    )


def select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()

    cache_hash = sha256(args.cache)
    checkpoint_hash = sha256(args.checkpoint)
    if cache_hash != EXPECTED_CACHE_SHA256:
        raise ValueError(f"cache hash mismatch: {cache_hash}")
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(f"checkpoint hash mismatch: {checkpoint_hash}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = FragmentRestorer(base=64)
    model.load_state_dict(checkpoint["model"])
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != EXPECTED_PARAMETERS:
        raise ValueError(f"parameter mismatch: {parameters}")
    device = select_device(args.device)
    model = model.to(device).eval()

    source = np.load(args.cache, mmap_mode="r", allow_pickle=False)
    raw_tiles = source["tiles"]
    restored_cases = np.empty_like(raw_tiles, dtype=np.uint8)
    bad_masks = np.empty(raw_tiles.shape[:2], dtype=np.bool_)
    restored_gray_counts = []
    for case_index in range(len(source["stems"])):
        raw = np.asarray(raw_tiles[case_index], np.uint8)
        tensor = torch.from_numpy(
            np.ascontiguousarray(raw.transpose(0, 3, 1, 2))
        ).float().div_(255.0)
        chunks = []
        for start in range(0, len(tensor), args.batch_size):
            chunks.append(model(tensor[start:start + args.batch_size].to(device)).cpu())
        restored = (
            torch.cat(chunks).permute(0, 2, 3, 1).mul(255).round()
            .clamp(0, 255).byte().numpy()
        )
        bad = no_gray_mask(raw, restored)
        restored_cases[case_index] = restored
        bad_masks[case_index] = bad
        restored_gray_counts.append(int(bad.sum()))
        print(json.dumps({
            "done": case_index + 1,
            "total": len(source["stems"]),
            "stem": str(source["stems"][case_index]),
            "reverted_tiles": int(bad.sum()),
        }), flush=True)

    provenance = {
        "schema_version": 1,
        "source_cache": str(args.cache.resolve()),
        "source_cache_sha256": cache_hash,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "architecture": "train_real_tile_restorer.FragmentRestorer(base=64)",
        "residual_multiplier": 0.5,
        "parameters": parameters,
        "guard": GUARD,
        "device": str(device),
        "dtype": "uint8",
        "cases": int(len(source["stems"])),
        "mean_reverted_tiles": float(np.mean(restored_gray_counts)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        restored=restored_cases,
        bad_mask=bad_masks,
        stems=np.asarray(source["stems"]),
        provenance_json=np.asarray(json.dumps(provenance, sort_keys=True)),
    )
    print(json.dumps({"output": str(args.output), **provenance}, indent=2), flush=True)


if __name__ == "__main__":
    main()

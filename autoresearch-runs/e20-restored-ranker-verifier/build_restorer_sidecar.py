"""Reproduce the exact real-restorer/no-gray sidecar used by E20."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(REPO), str(HERE)]

from e20_common import (EXPECTED_CACHE_SHA256, EXPECTED_RESTORER_SHA256,
                        sha256)
from train_real_tile_restorer import FragmentRestorer

EXPECTED_PARAMETERS = 1_670_595
GUARD = {
    "min_restored_std": 10.0,
    "relative_std_floor": 0.72,
    "max_mean_shift": 24.0,
    "low_saturation": 10.0,
    "low_texture_std": 25.0,
    "raw_texture_floor": 10.0,
}


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


def choose_device(name: str) -> torch.device:
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
    if checkpoint_hash != EXPECTED_RESTORER_SHA256:
        raise ValueError(f"checkpoint hash mismatch: {checkpoint_hash}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = FragmentRestorer(base=64)
    model.load_state_dict(checkpoint["model"])
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != EXPECTED_PARAMETERS:
        raise ValueError(f"restorer parameter mismatch: {parameters}")
    device = choose_device(args.device)
    model = model.to(device).eval()
    source = np.load(args.cache, mmap_mode="r", allow_pickle=False)
    restored_cases = np.empty_like(source["tiles"], dtype=np.uint8)
    bad_masks = np.empty(source["tiles"].shape[:2], dtype=np.bool_)
    reverted = []
    for index in range(len(source["stems"])):
        raw = np.asarray(source["tiles"][index], np.uint8)
        tensor = torch.from_numpy(
            np.ascontiguousarray(raw.transpose(0, 3, 1, 2))
        ).float().div_(255.0)
        chunks = [model(tensor[start:start + args.batch_size].to(device)).cpu()
                  for start in range(0, len(tensor), args.batch_size)]
        restored = (torch.cat(chunks).permute(0, 2, 3, 1).mul(255).round()
                    .clamp(0, 255).byte().numpy())
        bad = no_gray_mask(raw, restored)
        restored_cases[index] = restored
        bad_masks[index] = bad
        reverted.append(int(bad.sum()))
        print(json.dumps({"done": index + 1, "total": len(source["stems"]),
                          "stem": str(source["stems"][index]),
                          "reverted_tiles": int(bad.sum())}), flush=True)
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
        "mean_reverted_tiles": float(np.mean(reverted)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, restored=restored_cases, bad_mask=bad_masks,
                        stems=np.asarray(source["stems"]),
                        provenance_json=np.asarray(json.dumps(provenance, sort_keys=True)))
    print(json.dumps({"output": str(args.output), **provenance}, indent=2), flush=True)


if __name__ == "__main__":
    main()

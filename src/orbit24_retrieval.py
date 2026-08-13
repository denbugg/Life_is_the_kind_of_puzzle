from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image

GRID = 24
TILE = 20
TILES = GRID * GRID
DIRECTIONS = ("right", "left", "down", "up")


@dataclass(frozen=True)
class MetricRow:
    direction: str
    queries: int
    r1: float
    r5: float
    r20: float
    r64: float
    mean_reciprocal_rank: float
    worst_image_r20: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ORBIT-24 fixed-orientation candidate-retrieval evaluator. It does not rotate tiles."
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=("raw1", "multiband"), required=True)
    parser.add_argument("--max-images", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk", type=int, default=96)
    parser.add_argument("--bands", type=int, default=4)
    return parser.parse_args()


def is_e(path: Path) -> bool:
    return str(path.resolve()).lower().startswith("e:\\")


def image_tiles(path: Path) -> np.ndarray:
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    if image.shape != (GRID * TILE, GRID * TILE, 3):
        raise ValueError(f"expected 480x480 RGB image at {path}, got {image.shape}")
    return np.ascontiguousarray(
        image.reshape(GRID, TILE, GRID, TILE, 3).transpose(0, 2, 1, 3, 4).reshape(TILES, TILE, TILE, 3)
    )


def exact_input_to_target_mapping(input_tiles: np.ndarray, target_tiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    buckets: dict[bytes, deque[int]] = defaultdict(deque)
    for index, tile in enumerate(input_tiles):
        buckets[tile.tobytes()].append(index)
    target_to_input = np.full(TILES, -1, dtype=np.int64)
    for target_index, tile in enumerate(target_tiles):
        candidates = buckets[tile.tobytes()]
        if not candidates:
            raise ValueError(f"tile inventory mismatch at target tile {target_index}")
        target_to_input[target_index] = candidates.popleft()
    if any(bucket for bucket in buckets.values()):
        raise ValueError("tile inventory has unmatched shuffled inputs")
    input_to_target = np.empty(TILES, dtype=np.int64)
    input_to_target[target_to_input] = np.arange(TILES, dtype=np.int64)
    return target_to_input, input_to_target


def true_neighbours(target_to_input: np.ndarray) -> dict[str, np.ndarray]:
    result = {direction: np.full(TILES, -1, dtype=np.int64) for direction in DIRECTIONS}
    for row in range(GRID):
        for col in range(GRID):
            source_input = int(target_to_input[row * GRID + col])
            if col + 1 < GRID:
                result["right"][source_input] = int(target_to_input[row * GRID + col + 1])
            if col > 0:
                result["left"][source_input] = int(target_to_input[row * GRID + col - 1])
            if row + 1 < GRID:
                result["down"][source_input] = int(target_to_input[(row + 1) * GRID + col])
            if row > 0:
                result["up"][source_input] = int(target_to_input[(row - 1) * GRID + col])
    return result


def side_vectors(tiles: torch.Tensor, direction: str, bands: int, variant: str) -> tuple[torch.Tensor, torch.Tensor]:
    # `source` is the exposed side of tile i. `candidate` is the side of tile j facing it.
    if direction == "right":
        source = tiles[:, :, :, -bands:]
        candidate = tiles[:, :, :, :bands]
        interior_source = tiles[:, :, :, -bands - 1 : -1]
        interior_candidate = tiles[:, :, :, bands : bands + 1]
    elif direction == "left":
        source = tiles[:, :, :, :bands]
        candidate = tiles[:, :, :, -bands:]
        interior_source = tiles[:, :, :, 1 : bands + 1]
        interior_candidate = tiles[:, :, :, -bands - 1 : -1]
    elif direction == "down":
        source = tiles[:, :, -bands:, :]
        candidate = tiles[:, :, :bands, :]
        interior_source = tiles[:, :, -bands - 1 : -1, :]
        interior_candidate = tiles[:, :, bands : bands + 1, :]
    elif direction == "up":
        source = tiles[:, :, :bands, :]
        candidate = tiles[:, :, -bands:, :]
        interior_source = tiles[:, :, 1 : bands + 1, :]
        interior_candidate = tiles[:, :, -bands - 1 : -1, :]
    else:
        raise ValueError(direction)

    raw_source = source.flatten(1)
    raw_candidate = candidate.flatten(1)
    if variant == "raw1":
        return raw_source[:, : 3 * TILE], raw_candidate[:, : 3 * TILE]

    # Fixed-orientation multi-band features: RGB seam, normal gradient continuation and tangential texture.
    source_gradient = (source - interior_source).flatten(1)
    candidate_gradient = (interior_candidate - candidate).flatten(1)
    if direction in ("right", "left"):
        tangential_source = (source[:, :, 1:, :] - source[:, :, :-1, :]).flatten(1)
        tangential_candidate = (candidate[:, :, 1:, :] - candidate[:, :, :-1, :]).flatten(1)
    else:
        tangential_source = (source[:, :, :, 1:] - source[:, :, :, :-1]).flatten(1)
        tangential_candidate = (candidate[:, :, :, 1:] - candidate[:, :, :, :-1]).flatten(1)
    return (
        torch.cat((raw_source, source_gradient, tangential_source), dim=1),
        torch.cat((raw_candidate, candidate_gradient, tangential_candidate), dim=1),
    )


def ranks_for_direction(tiles: torch.Tensor, truth: np.ndarray, direction: str, bands: int, variant: str, chunk: int) -> list[int]:
    source, candidate = side_vectors(tiles, direction, bands, variant)
    source = torch.nn.functional.normalize(source, dim=1)
    candidate = torch.nn.functional.normalize(candidate, dim=1)
    ranks: list[int] = []
    for start in range(0, TILES, chunk):
        stop = min(TILES, start + chunk)
        score = source[start:stop] @ candidate.T
        rows = torch.arange(stop - start, device=tiles.device)
        score[rows, torch.arange(start, stop, device=tiles.device)] = -float("inf")
        order = torch.argsort(score, dim=1, descending=True).cpu().numpy()
        for offset, query in enumerate(range(start, stop)):
            expected = int(truth[query])
            if expected < 0:
                continue
            rank = int(np.flatnonzero(order[offset] == expected)[0]) + 1
            ranks.append(rank)
    return ranks


def metric(ranks: Iterable[int]) -> dict[str, float | int]:
    values = np.asarray(list(ranks), dtype=np.int64)
    if not len(values):
        raise ValueError("no valid directional queries")
    return {
        "queries": int(len(values)),
        "r1": float(np.mean(values <= 1)),
        "r5": float(np.mean(values <= 5)),
        "r20": float(np.mean(values <= 20)),
        "r64": float(np.mean(values <= 64)),
        "mean_reciprocal_rank": float(np.mean(1.0 / values)),
    }


def main() -> int:
    args = parse_args()
    if args.max_images < 1 or args.chunk < 1 or args.bands < 1 or args.bands > 8:
        raise ValueError("invalid max-images/chunk/bands")
    if not all(is_e(path) for path in (args.input_root, args.target_root, args.output_dir)):
        raise RuntimeError("all mutable/data paths for ORBIT-24 must be on E:")
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for ORBIT-24 retrieval evaluation")
    repo_src = Path(__file__).resolve().parent
    sys.path.insert(0, str(repo_src))
    import train_e26_contextual_edge as e26

    sources = e26.load_authenticated_training_sources(args.source_manifest)
    split = e26.split_source_groups(sources.names, sources.group_for_name, mapping_sha256=sources.mapping_sha256)
    names = tuple(split.development_names[: args.max_images])
    if len(names) != args.max_images:
        raise ValueError("max-images exceeds authenticated development split")
    device = torch.device(args.device)
    per_direction: dict[str, list[int]] = {direction: [] for direction in DIRECTIONS}
    per_image_r20: dict[str, dict[str, float]] = {direction: {} for direction in DIRECTIONS}
    started = time.perf_counter()
    for count, name in enumerate(names, start=1):
        input_tiles = image_tiles(args.input_root / name)
        target_tiles = image_tiles(args.target_root / name)
        target_to_input, _ = exact_input_to_target_mapping(input_tiles, target_tiles)
        truth = true_neighbours(target_to_input)
        tiles = torch.from_numpy(input_tiles).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32).div_(255.0)
        for direction in DIRECTIONS:
            ranks = ranks_for_direction(tiles, truth[direction], direction, args.bands, args.variant, args.chunk)
            per_direction[direction].extend(ranks)
            per_image_r20[direction][name] = float(np.mean(np.asarray(ranks) <= 20))
        if count % 8 == 0:
            print(f"progress={count}/{len(names)}")
    elapsed = time.perf_counter() - started
    result: dict[str, object] = {
        "schema": "orbit24-fixed-orientation-retrieval-v1",
        "variant": args.variant,
        "orientation": "fixed_no_rotation",
        "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_sha256": hashlib.sha256(args.source_manifest.read_bytes()).hexdigest(),
        "split": "authenticated_e26_development",
        "image_count": len(names),
        "image_names_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "bands": args.bands,
        "metrics": {},
        "elapsed_seconds": elapsed,
        "status": "PASS",
    }
    all_ranks: list[int] = []
    for direction in DIRECTIONS:
        row = metric(per_direction[direction])
        row["worst_image_r20"] = min(per_image_r20[direction].values())
        result["metrics"][direction] = row
        all_ranks.extend(per_direction[direction])
    overall = metric(all_ranks)
    overall["worst_image_r20"] = min(min(values.values()) for values in per_image_r20.values())
    result["metrics"]["overall"] = overall
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = args.output_dir / f"orbit24_{args.variant}_report.json"
    report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

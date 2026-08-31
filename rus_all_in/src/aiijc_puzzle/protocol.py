"""Frozen validation protocol and image utilities for AIIJC Puzzle.

This module deliberately has no knowledge of the competition test directory.  A
validation manifest can therefore only be built from paired training inputs and
targets.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from skimage.metrics import structural_similarity

GRID_SIZE = 24
TILE_SIZE = 20
IMAGE_SIZE = GRID_SIZE * TILE_SIZE
TILE_COUNT = GRID_SIZE * GRID_SIZE
RGB_CHANNELS = 3
EXPECTED_TRAIN_PAIRS = 7_000
PROTOCOL_VERSION = 1
SPLIT_ALGORITHM = "sha256(f'{seed}\\0{filename}') ascending, filename tie-break"
EXPERIMENT_SUBSET_NAMESPACE = "aiijc-puzzle-experiments-v1"
EXPERIMENT_SUBSET_SEED = 20260829

# Report schemas evolved across experiments.  These keys all denote panels
# whose exact source membership was intentionally opened for fitting,
# evaluation, or confirmation.  Checkpoint lineage is handled separately: an
# unrelated model's training ancestry is not itself an opened target panel.
DECLARED_SOURCE_PANEL_KEYS = frozenset(
    {
        "train_filenames",
        "eval_filenames",
        "source_filenames",
        "calibration_filenames",
        "holdout_filenames",
    }
)


def collect_declared_source_filenames(
    value: Any,
    *,
    parent_key: str = "",
) -> set[str]:
    """Collect source-panel filenames from any supported experiment report.

    In addition to the older generic keys, any ``*_source_filenames`` key is
    accepted.  This covers newer two-stage reports such as
    ``fit_source_filenames`` and ``confirm_source_filenames`` without silently
    dropping future named source panels.
    """

    names: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            names.update(
                collect_declared_source_filenames(child, parent_key=str(key))
            )
    elif isinstance(value, list):
        is_source_panel = (
            parent_key in DECLARED_SOURCE_PANEL_KEYS
            or parent_key.endswith("_source_filenames")
        )
        if is_source_panel:
            if not all(isinstance(item, str) for item in value):
                raise ValueError(
                    f"declared source panel {parent_key!r} must contain only strings"
                )
            names.update(value)
        else:
            for child in value:
                names.update(
                    collect_declared_source_filenames(child, parent_key=parent_key)
                )
    return names


@dataclass(frozen=True)
class SplitCounts:
    """Number of paired examples allocated to each frozen split."""

    train: int
    calibration: int
    holdout: int

    def __post_init__(self) -> None:
        for name, value in self.as_dict().items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} count must be a non-negative integer, got {value!r}")

    @property
    def total(self) -> int:
        return self.train + self.calibration + self.holdout

    def as_dict(self) -> dict[str, int]:
        return {
            "train": self.train,
            "calibration": self.calibration,
            "holdout": self.holdout,
        }


def contest_ssim(reference: NDArray[Any], prediction: NDArray[Any]) -> float:
    """Compute the organizer's RGB SSIM metric.

    The keyword arguments intentionally match the contest definition exactly.
    In particular, scikit-image's default 7-pixel window and uniform weighting
    are left unchanged.
    """

    reference_array = np.asarray(reference)
    prediction_array = np.asarray(prediction)
    if reference_array.shape != prediction_array.shape:
        raise ValueError(
            "SSIM images must have identical shapes, got "
            f"{reference_array.shape} and {prediction_array.shape}"
        )
    if reference_array.ndim != 3 or reference_array.shape[2] != RGB_CHANNELS:
        raise ValueError(f"SSIM expects an HxWx3 RGB image, got {reference_array.shape}")
    return float(
        structural_similarity(
            reference_array,
            prediction_array,
            channel_axis=2,
            data_range=255,
        )
    )


def split_tiles(image: NDArray[Any]) -> NDArray[Any]:
    """Split one 480x480 RGB image into 576 row-major 20x20 RGB tiles."""

    array = np.asarray(image)
    expected_shape = (IMAGE_SIZE, IMAGE_SIZE, RGB_CHANNELS)
    if array.shape != expected_shape:
        raise ValueError(f"Expected image shape {expected_shape}, got {array.shape}")
    tiles = array.reshape(GRID_SIZE, TILE_SIZE, GRID_SIZE, TILE_SIZE, RGB_CHANNELS)
    return tiles.transpose(0, 2, 1, 3, 4).reshape(TILE_COUNT, TILE_SIZE, TILE_SIZE, RGB_CHANNELS)


def assemble_tiles(tiles: NDArray[Any]) -> NDArray[Any]:
    """Assemble 576 row-major 20x20 RGB tiles into one 480x480 RGB image."""

    array = np.asarray(tiles)
    expected_shape = (TILE_COUNT, TILE_SIZE, TILE_SIZE, RGB_CHANNELS)
    if array.shape != expected_shape:
        raise ValueError(f"Expected tile shape {expected_shape}, got {array.shape}")
    grid = array.reshape(GRID_SIZE, GRID_SIZE, TILE_SIZE, TILE_SIZE, RGB_CHANNELS)
    return grid.transpose(0, 2, 1, 3, 4).reshape(IMAGE_SIZE, IMAGE_SIZE, RGB_CHANNELS)


def discover_training_pairs(
    inputs_dir: Path,
    targets_dir: Path,
    *,
    expected_pairs: int = EXPECTED_TRAIN_PAIRS,
) -> tuple[str, ...]:
    """Return matching PNG basenames from paired training directories.

    Only the two explicit directories are inspected.  This prevents accidental
    test-set access while constructing or updating validation membership.
    """

    if not inputs_dir.is_dir():
        raise FileNotFoundError(f"Training inputs directory does not exist: {inputs_dir}")
    if not targets_dir.is_dir():
        raise FileNotFoundError(f"Training targets directory does not exist: {targets_dir}")
    if isinstance(expected_pairs, bool) or not isinstance(expected_pairs, int):
        raise ValueError("expected_pairs must be an integer")
    if expected_pairs <= 0:
        raise ValueError("expected_pairs must be positive")

    input_names = {
        path.name for path in inputs_dir.iterdir() if path.is_file() and path.suffix == ".png"
    }
    target_names = {
        path.name for path in targets_dir.iterdir() if path.is_file() and path.suffix == ".png"
    }
    if input_names != target_names:
        missing_targets = sorted(input_names - target_names)
        missing_inputs = sorted(target_names - input_names)
        raise ValueError(
            "Training input/target filenames do not match: "
            f"missing_targets={missing_targets[:5]}, missing_inputs={missing_inputs[:5]}"
        )
    if len(input_names) != expected_pairs:
        raise ValueError(f"Found {len(input_names)} matching pairs, expected {expected_pairs}")
    return tuple(sorted(input_names))


def assign_splits(
    filenames: Iterable[str],
    *,
    seed: int,
    counts: SplitCounts,
) -> dict[str, tuple[str, ...]]:
    """Assign unique filenames using a stable SHA-256 ranking."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    names = tuple(filenames)
    if len(names) != len(set(names)):
        raise ValueError("filenames must be unique")
    if len(names) != counts.total:
        raise ValueError(f"Received {len(names)} filenames but split counts total {counts.total}")

    prefix = f"{seed}\0".encode()
    ranked = sorted(
        names,
        key=lambda name: (hashlib.sha256(prefix + name.encode("utf-8")).digest(), name),
    )
    train_end = counts.train
    calibration_end = train_end + counts.calibration
    # Lexical order inside each split makes manifests easy to inspect and diff;
    # membership itself is determined solely by the stable ranking above.
    return {
        "train": tuple(sorted(ranked[:train_end])),
        "calibration": tuple(sorted(ranked[train_end:calibration_end])),
        "holdout": tuple(sorted(ranked[calibration_end:])),
    }


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the lowercase SHA-256 hex digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def compute_protocol_digest(manifest: Mapping[str, Any]) -> str:
    """Digest canonical manifest JSON, excluding the digest field itself."""

    payload = dict(manifest)
    payload.pop("protocol_digest", None)
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def build_validation_manifest(
    inputs_dir: Path,
    targets_dir: Path,
    *,
    seed: int,
    counts: SplitCounts,
    expected_pairs: int = EXPECTED_TRAIN_PAIRS,
) -> dict[str, Any]:
    """Build a portable, content-addressed manifest from training pairs only."""

    filenames = discover_training_pairs(
        inputs_dir,
        targets_dir,
        expected_pairs=expected_pairs,
    )
    if counts.total != expected_pairs:
        raise ValueError(
            f"Split counts total {counts.total}, but expected_pairs is {expected_pairs}"
        )
    assignments = assign_splits(filenames, seed=seed, counts=counts)

    splits: dict[str, list[dict[str, str]]] = {}
    for split_name, split_names in assignments.items():
        splits[split_name] = [
            {
                "filename": filename,
                "input_sha256": sha256_file(inputs_dir / filename),
                "target_sha256": sha256_file(targets_dir / filename),
            }
            for filename in split_names
        ]

    manifest: dict[str, Any] = {
        "schema_version": PROTOCOL_VERSION,
        "protocol": {
            "seed": seed,
            "expected_pairs": expected_pairs,
            "counts": counts.as_dict(),
            "split_algorithm": SPLIT_ALGORITHM,
            "metric": {
                "name": "skimage.metrics.structural_similarity",
                "channel_axis": 2,
                "data_range": 255,
                "win_size": 7,
            },
            "tiling": {
                "grid_rows": GRID_SIZE,
                "grid_columns": GRID_SIZE,
                "tile_height": TILE_SIZE,
                "tile_width": TILE_SIZE,
                "order": "row-major",
            },
            "digest": {
                "algorithm": "sha256",
                "scope": "canonical JSON of this manifest excluding protocol_digest",
            },
        },
        "splits": splits,
    }
    manifest["protocol_digest"] = compute_protocol_digest(manifest)
    return manifest


def write_validation_manifest(manifest: Mapping[str, Any], output_path: Path) -> None:
    """Atomically write a human-readable JSON validation manifest."""

    expected_digest = compute_protocol_digest(manifest)
    if manifest.get("protocol_digest") != expected_digest:
        raise ValueError("Refusing to write a manifest with an invalid protocol digest")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)


def split_name_sets(manifest: Mapping[str, Any]) -> dict[str, set[str]]:
    """Extract filename sets from a manifest for validation and consumers."""

    raw_splits = manifest.get("splits")
    if not isinstance(raw_splits, Mapping):
        raise ValueError("Manifest has no splits mapping")
    result: dict[str, set[str]] = {}
    for split_name in ("train", "calibration", "holdout"):
        records = raw_splits.get(split_name)
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            raise ValueError(f"Manifest split {split_name!r} is not a record sequence")
        names: set[str] = set()
        for record in records:
            if not isinstance(record, Mapping) or not isinstance(record.get("filename"), str):
                raise ValueError(f"Invalid record in manifest split {split_name!r}")
            filename = record["filename"]
            if filename in names:
                raise ValueError(f"Duplicate filename {filename!r} in split {split_name!r}")
            names.add(filename)
        result[split_name] = names
    return result


def select_manifest_records(
    manifest: Mapping[str, Any],
    split: str,
    *,
    limit: int,
    seed: int = EXPERIMENT_SUBSET_SEED,
    namespace: str = EXPERIMENT_SUBSET_NAMESPACE,
) -> tuple[Mapping[str, Any], ...]:
    """Select a shared deterministic experiment subset inside a frozen split.

    The namespace is part of the ranking input, making the rule explicit and
    versionable.  All current experiment families use the default namespace so
    their calibration-48 and holdout-48 panels are directly paired.
    """

    if not isinstance(split, str) or not split:
        raise ValueError("split must be a non-empty string")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("namespace must be a non-empty string")
    raw_splits = manifest.get("splits")
    if not isinstance(raw_splits, Mapping):
        raise ValueError("Manifest has no splits mapping")
    records = raw_splits.get(split)
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError(f"Manifest split {split!r} is not a record sequence")
    if limit > len(records):
        raise ValueError(f"limit must not exceed split size {len(records)}, got {limit}")
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(record.get("filename"), str):
            raise ValueError(f"Invalid record in manifest split {split!r}")

    prefix = f"{namespace}\0{seed}\0".encode()
    ranked = sorted(
        records,
        key=lambda record: (
            hashlib.sha256(prefix + record["filename"].encode("utf-8")).digest(),
            record["filename"],
        ),
    )
    return tuple(ranked[:limit])

"""Strict loading and grouped sampling of high-purity real tile pairs.

The gold NPZ contains metadata and tile indices only.  Pixels are read lazily
from the original train input/target PNGs, so a sampled pair is always:

    corrupted = train/inputs/<source>[input_slot]
    clean     = train/targets/<source>[clean_tile_index]

Sampling is deliberately two-stage: choose a source uniformly, then choose one
of that source's active pairs uniformly.  This prevents high-coverage images
from dominating fine-tuning merely because they contributed more gold pairs.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Sequence

import numpy as np
from PIL import Image
import torch

from .tiles import GRID, IMAGE_SIZE, TILE, split_tiles_numpy


PAIR_ARRAY_KEYS = (
    "source_index",
    "input_slot",
    "clean_tile_index",
    "coarse_cost",
    "structural_cost",
    "coarse_row_margin",
    "coarse_column_margin",
    "structural_row_margin",
    "structural_column_margin",
    "joint_confidence",
    "consensus",
    "coarse_mutual_cycle",
    "structural_mutual_cycle",
)
SOURCE_ARRAY_KEYS = (
    "source_consensus_count",
    "source_both_mutual_count",
    "source_selected_count",
)
REQUIRED_KEYS = {"meta", "source_names", *PAIR_ARRAY_KEYS, *SOURCE_ARRAY_KEYS}
EXPECTED_DTYPES = {
    "source_index": np.dtype(np.uint16),
    "input_slot": np.dtype(np.uint16),
    "clean_tile_index": np.dtype(np.uint16),
    "coarse_cost": np.dtype(np.float32),
    "structural_cost": np.dtype(np.float32),
    "coarse_row_margin": np.dtype(np.float32),
    "coarse_column_margin": np.dtype(np.float32),
    "structural_row_margin": np.dtype(np.float32),
    "structural_column_margin": np.dtype(np.float32),
    "joint_confidence": np.dtype(np.float32),
    "consensus": np.dtype(np.uint8),
    "coarse_mutual_cycle": np.dtype(np.uint8),
    "structural_mutual_cycle": np.dtype(np.uint8),
    "source_consensus_count": np.dtype(np.uint16),
    "source_both_mutual_count": np.dtype(np.uint16),
    "source_selected_count": np.dtype(np.uint16),
}
SOURCE_NAME_PATTERN = re.compile(r"img_\d{6}\.png")


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _readonly(array: np.ndarray) -> np.ndarray:
    output = np.asarray(array).copy()
    output.setflags(write=False)
    return output


def _require_scalar_meta(array: np.ndarray) -> dict:
    if array.shape != () or array.dtype.kind not in {"U", "S"}:
        raise ValueError(f"meta must be a scalar JSON string, got shape={array.shape} dtype={array.dtype}")
    try:
        payload = json.loads(array.item())
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("meta is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("meta JSON must contain an object")
    return payload


@dataclass(frozen=True)
class RealPairTable:
    """Validated, immutable view over active rows in a real-gold NPZ."""

    npz_path: Path
    manifest_path: Path
    data_root: Path
    metadata: dict
    npz_sha256: str
    source_names: tuple[str, ...]
    source_index: np.ndarray
    input_slot: np.ndarray
    clean_tile_index: np.ndarray
    joint_confidence: np.ndarray
    source_selected_count: np.ndarray
    active_pair_indices: np.ndarray
    active_pair_mask: np.ndarray
    active_source_indices: np.ndarray
    rows_by_source: tuple[np.ndarray, ...]
    min_confidence: float

    @property
    def source_count(self) -> int:
        return len(self.source_names)

    @property
    def stored_pair_count(self) -> int:
        return len(self.source_index)

    @property
    def active_pair_count(self) -> int:
        return len(self.active_pair_indices)

    def source_rows(self, source_index: int) -> np.ndarray:
        if source_index < 0 or source_index >= self.source_count:
            raise IndexError(f"source_index {source_index} outside [0, {self.source_count})")
        return self.rows_by_source[source_index]

    @classmethod
    def load(
        cls,
        npz_path: str | Path,
        *,
        manifest_path: str | Path,
        data_root: str | Path,
        expected_split: str,
        min_confidence: float | None = None,
    ) -> "RealPairTable":
        npz_path = Path(npz_path)
        manifest_path = Path(manifest_path)
        data_root = Path(data_root)
        manifest = _read_json(manifest_path)

        with np.load(npz_path, allow_pickle=False) as archive:
            keys = set(archive.files)
            if keys != REQUIRED_KEYS:
                missing = sorted(REQUIRED_KEYS - keys)
                unexpected = sorted(keys - REQUIRED_KEYS)
                raise ValueError(f"unexpected NPZ schema missing={missing} unexpected={unexpected}")
            arrays = {key: np.asarray(archive[key]).copy() for key in archive.files}

        metadata = _require_scalar_meta(arrays["meta"])
        required_meta = {
            "schema_version",
            "kind",
            "manifest_sha256",
            "split",
            "source_count",
            "total_tiles",
            "selected_pairs",
            "selected_coverage",
            "thresholds",
            "source_name_encoding",
            "old_q90_used_as_ground_truth",
        }
        missing_meta = sorted(required_meta - set(metadata))
        if missing_meta:
            raise ValueError(f"NPZ metadata is missing {missing_meta}")
        if metadata["schema_version"] != 1 or metadata["kind"] != "high_purity_real_tile_pairs":
            raise ValueError("unsupported real-pair schema or kind")
        if metadata["old_q90_used_as_ground_truth"] is not False:
            raise ValueError("artifact permits old q90 pseudo-ground-truth")
        if metadata["source_name_encoding"] != "source_names[source_index]":
            raise ValueError("unsupported source-name encoding")
        if metadata["split"] != expected_split:
            raise ValueError(f"artifact split {metadata['split']} != expected {expected_split}")

        manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        if metadata["manifest_sha256"] != manifest_sha256:
            raise ValueError("artifact manifest SHA256 does not match the supplied manifest")
        required_splits = {"train", "val", "audit"}
        if set(manifest.get("splits", {})) != required_splits:
            raise ValueError(f"manifest must contain exactly {sorted(required_splits)}")
        if not manifest.get("policy", {}).get("exclude_all_test_filename_overlaps", False):
            raise ValueError("manifest does not enforce test-filename exclusion")

        source_names_array = arrays["source_names"]
        if source_names_array.ndim != 1 or source_names_array.dtype.kind != "U":
            raise ValueError("source_names must be a one-dimensional Unicode array")
        source_names = tuple(str(name) for name in source_names_array)
        if len(source_names) != len(set(source_names)):
            raise ValueError("source_names contains duplicates")
        if any(SOURCE_NAME_PATTERN.fullmatch(name) is None for name in source_names):
            raise ValueError("source_names contains an invalid filename")
        if int(metadata["source_count"]) != len(source_names):
            raise ValueError("metadata source_count does not match source_names")
        if int(metadata["total_tiles"]) != len(source_names) * GRID * GRID:
            raise ValueError("metadata total_tiles is inconsistent")

        stored_pairs = int(metadata["selected_pairs"])
        if stored_pairs < 1:
            raise ValueError("real-pair artifact is empty")
        for key in PAIR_ARRAY_KEYS:
            array = arrays[key]
            if array.ndim != 1 or len(array) != stored_pairs:
                raise ValueError(f"{key} must have shape ({stored_pairs},), got {array.shape}")
            if array.dtype != EXPECTED_DTYPES[key]:
                raise ValueError(f"{key} dtype {array.dtype} != {EXPECTED_DTYPES[key]}")
        for key in SOURCE_ARRAY_KEYS:
            array = arrays[key]
            if array.shape != (len(source_names),):
                raise ValueError(f"{key} must have shape ({len(source_names)},), got {array.shape}")
            if array.dtype != EXPECTED_DTYPES[key]:
                raise ValueError(f"{key} dtype {array.dtype} != {EXPECTED_DTYPES[key]}")

        expected_coverage = stored_pairs / float(int(metadata["total_tiles"]))
        if not np.isclose(float(metadata["selected_coverage"]), expected_coverage, atol=1e-12):
            raise ValueError("metadata selected_coverage is inconsistent")

        for key in (
            "coarse_cost",
            "structural_cost",
            "coarse_row_margin",
            "coarse_column_margin",
            "structural_row_margin",
            "structural_column_margin",
            "joint_confidence",
        ):
            if not np.isfinite(arrays[key]).all():
                raise ValueError(f"{key} contains non-finite values")
        if (arrays["coarse_cost"] < 0).any() or (arrays["structural_cost"] < 0).any():
            raise ValueError("descriptor costs must be non-negative")
        for flag in ("consensus", "coarse_mutual_cycle", "structural_mutual_cycle"):
            if not np.all(arrays[flag] == 1):
                raise ValueError(f"stored pair failed required gate {flag}")

        thresholds = metadata["thresholds"]
        if not isinstance(thresholds, dict):
            raise ValueError("metadata thresholds must be an object")
        try:
            coarse_floor = float(thresholds["coarse_min_margin"])
            structural_floor = float(thresholds["structural_min_margin"])
            built_confidence_floor = float(thresholds["joint_min_confidence"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("metadata thresholds are incomplete") from error
        if not all(np.isfinite(value) and value >= 0 for value in (coarse_floor, structural_floor, built_confidence_floor)):
            raise ValueError("metadata thresholds must be finite and non-negative")
        if (
            (arrays["coarse_row_margin"] < coarse_floor).any()
            or (arrays["coarse_column_margin"] < coarse_floor).any()
            or (arrays["structural_row_margin"] < structural_floor).any()
            or (arrays["structural_column_margin"] < structural_floor).any()
            or (arrays["joint_confidence"] < built_confidence_floor).any()
        ):
            raise ValueError("stored pair violates metadata thresholds")

        source_index = arrays["source_index"].astype(np.int64)
        input_slot = arrays["input_slot"].astype(np.int64)
        clean_tile_index = arrays["clean_tile_index"].astype(np.int64)
        if source_index.min() < 0 or source_index.max() >= len(source_names):
            raise ValueError("source_index is outside source_names")
        if input_slot.min() < 0 or input_slot.max() >= GRID * GRID:
            raise ValueError("input_slot is outside the 24x24 tile grid")
        if clean_tile_index.min() < 0 or clean_tile_index.max() >= GRID * GRID:
            raise ValueError("clean_tile_index is outside the 24x24 tile grid")
        source_slot_key = source_index * (GRID * GRID) + input_slot
        source_clean_key = source_index * (GRID * GRID) + clean_tile_index
        if len(np.unique(source_slot_key)) != stored_pairs:
            raise ValueError("an input slot is duplicated within a source")
        if len(np.unique(source_clean_key)) != stored_pairs:
            raise ValueError("a clean tile is duplicated within a source")

        selected_counts = np.bincount(source_index, minlength=len(source_names))
        if not np.array_equal(selected_counts, arrays["source_selected_count"].astype(np.int64)):
            raise ValueError("source_selected_count does not match pair rows")
        consensus_counts = arrays["source_consensus_count"].astype(np.int64)
        mutual_counts = arrays["source_both_mutual_count"].astype(np.int64)
        if (
            (consensus_counts < selected_counts).any()
            or (mutual_counts < selected_counts).any()
            or (consensus_counts > GRID * GRID).any()
            or (mutual_counts > GRID * GRID).any()
        ):
            raise ValueError("source-level diagnostic counts are inconsistent")

        split_names = set(manifest["splits"][expected_split])
        excluded_names = set(manifest.get("excluded_test_overlap", []))
        source_name_set = set(source_names)
        if not source_name_set <= split_names:
            raise ValueError("artifact contains names outside the expected manifest split")
        if source_name_set & excluded_names:
            raise ValueError("artifact contains excluded test-overlap names")
        actual_test_names = {path.name for path in (data_root / "test").glob("*.png")}
        if source_name_set & actual_test_names:
            raise ValueError("artifact contains a filename present in the actual test directory")
        missing_files = [
            name
            for name in source_names
            if not (data_root / "train" / "inputs" / name).is_file()
            or not (data_root / "train" / "targets" / name).is_file()
        ]
        if missing_files:
            raise FileNotFoundError(f"missing input/target images for {missing_files[:5]}")

        effective_floor = built_confidence_floor if min_confidence is None else float(min_confidence)
        if not np.isfinite(effective_floor) or effective_floor < built_confidence_floor:
            raise ValueError(
                f"min_confidence must be finite and >= artifact floor {built_confidence_floor}"
            )
        active_rows = np.flatnonzero(arrays["joint_confidence"] >= effective_floor).astype(np.int64)
        if len(active_rows) == 0:
            raise ValueError("confidence threshold removed every real pair")
        active_mask = np.zeros(stored_pairs, dtype=bool)
        active_mask[active_rows] = True
        rows_by_source = tuple(
            _readonly(active_rows[source_index[active_rows] == index])
            for index in range(len(source_names))
        )
        active_sources = np.asarray(
            [index for index, rows in enumerate(rows_by_source) if len(rows)],
            dtype=np.int64,
        )

        return cls(
            npz_path=npz_path,
            manifest_path=manifest_path,
            data_root=data_root,
            metadata=json.loads(json.dumps(metadata)),
            npz_sha256=hashlib.sha256(npz_path.read_bytes()).hexdigest(),
            source_names=source_names,
            source_index=_readonly(source_index),
            input_slot=_readonly(input_slot),
            clean_tile_index=_readonly(clean_tile_index),
            joint_confidence=_readonly(arrays["joint_confidence"]),
            source_selected_count=_readonly(arrays["source_selected_count"].astype(np.int64)),
            active_pair_indices=_readonly(active_rows),
            active_pair_mask=_readonly(active_mask),
            active_source_indices=_readonly(active_sources),
            rows_by_source=rows_by_source,
            min_confidence=effective_floor,
        )


@dataclass(frozen=True)
class RealPairBatch:
    corrupt: torch.Tensor
    clean: torch.Tensor
    source_index: torch.Tensor
    input_slot: torch.Tensor
    clean_tile_index: torch.Tensor
    confidence: torch.Tensor
    pair_row: torch.Tensor

    def __len__(self) -> int:
        return int(self.corrupt.shape[0])


@dataclass(frozen=True)
class CacheInfo:
    hits: int
    misses: int
    size: int
    capacity: int


class _TilePairLRU:
    def __init__(self, table: RealPairTable, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("cache_size must be positive")
        self.table = table
        self.capacity = int(capacity)
        self.cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _load_tiles(path: Path) -> np.ndarray:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if rgb.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
            raise ValueError(f"expected RGB {(IMAGE_SIZE, IMAGE_SIZE)}, got {rgb.shape} at {path}")
        tiles = split_tiles_numpy(rgb)
        tiles.setflags(write=False)
        return tiles

    def get(self, source_index: int) -> tuple[np.ndarray, np.ndarray]:
        if source_index in self.cache:
            self.hits += 1
            self.cache.move_to_end(source_index)
            return self.cache[source_index]
        self.misses += 1
        name = self.table.source_names[source_index]
        train_dir = self.table.data_root / "train"
        value = (
            self._load_tiles(train_dir / "inputs" / name),
            self._load_tiles(train_dir / "targets" / name),
        )
        self.cache[source_index] = value
        self.cache.move_to_end(source_index)
        while len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return value

    def clear(self) -> None:
        self.cache.clear()

    def info(self) -> CacheInfo:
        return CacheInfo(self.hits, self.misses, len(self.cache), self.capacity)


class RealPairSampler:
    """Source-uniform, pair-uniform sampler backed by a bounded image LRU."""

    def __init__(
        self,
        table: RealPairTable,
        *,
        seed: int = 20260710,
        cache_size: int = 8,
    ) -> None:
        if len(table.active_source_indices) == 0:
            raise ValueError("table has no active sources")
        self.table = table
        self.rng = np.random.default_rng(seed)
        self.cache = _TilePairLRU(table, cache_size)

    def draw_pair_rows(
        self,
        batch_size: int,
        *,
        generator: np.random.Generator | None = None,
    ) -> np.ndarray:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if generator is None:
            generator = self.rng
        active_sources = self.table.active_source_indices
        source_draws = active_sources[
            generator.integers(0, len(active_sources), size=batch_size)
        ]
        rows = np.empty(batch_size, dtype=np.int64)
        for source_index in np.unique(source_draws):
            positions = np.flatnonzero(source_draws == source_index)
            group = self.table.source_rows(int(source_index))
            choices = generator.integers(0, len(group), size=len(positions))
            rows[positions] = group[choices]
        return rows

    def draw_grouped_pair_rows(
        self,
        batch_size: int,
        pairs_per_source: int,
        *,
        generator: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Draw a shuffled batch from a small uniform set of source groups.

        Sources are drawn uniformly without replacement when the requested
        number of groups fits in the active source set.  Larger requests use
        independently shuffled source cycles.  Rows within each source are
        sampled uniformly with replacement.  Thus ``256, 32`` produces eight
        distinct source groups of 32 pairs when at least eight sources exist.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if pairs_per_source <= 0:
            raise ValueError("pairs_per_source must be positive")
        if generator is None:
            generator = self.rng

        full_groups, remainder = divmod(batch_size, pairs_per_source)
        group_sizes = [pairs_per_source] * full_groups
        if remainder:
            group_sizes.append(remainder)
        group_count = len(group_sizes)
        active_sources = self.table.active_source_indices

        if group_count <= len(active_sources):
            source_groups = generator.choice(
                active_sources,
                size=group_count,
                replace=False,
            )
        else:
            source_parts = []
            remaining = group_count
            while remaining:
                cycle = generator.permutation(active_sources)
                take = min(remaining, len(cycle))
                source_parts.append(cycle[:take])
                remaining -= take
            source_groups = np.concatenate(source_parts)

        row_parts = []
        for source_index, group_size in zip(source_groups, group_sizes, strict=True):
            group = self.table.source_rows(int(source_index))
            choices = generator.integers(0, len(group), size=group_size)
            row_parts.append(group[choices])
        rows = np.concatenate(row_parts).astype(np.int64, copy=False)
        return rows[generator.permutation(batch_size)]

    def _materialize_rows(self, rows: np.ndarray) -> RealPairBatch:
        rows = np.asarray(rows, dtype=np.int64)
        if rows.ndim != 1 or len(rows) == 0:
            raise ValueError("rows must be a non-empty one-dimensional array")
        if (
            rows.min() < 0
            or rows.max() >= self.table.stored_pair_count
            or not self.table.active_pair_mask[rows].all()
        ):
            raise ValueError("rows contains an inactive or out-of-range pair")

        source_indices = self.table.source_index[rows]
        corrupt = np.empty((len(rows), TILE, TILE, 3), dtype=np.uint8)
        clean = np.empty_like(corrupt)
        for source_index in np.unique(source_indices):
            positions = np.flatnonzero(source_indices == source_index)
            input_tiles, target_tiles = self.cache.get(int(source_index))
            selected_rows = rows[positions]
            corrupt[positions] = input_tiles[self.table.input_slot[selected_rows]]
            clean[positions] = target_tiles[self.table.clean_tile_index[selected_rows]]

        corrupt_tensor = torch.from_numpy(
            np.ascontiguousarray(corrupt.transpose(0, 3, 1, 2))
        ).float().div_(255.0)
        clean_tensor = torch.from_numpy(
            np.ascontiguousarray(clean.transpose(0, 3, 1, 2))
        ).float().div_(255.0)
        return RealPairBatch(
            corrupt=corrupt_tensor,
            clean=clean_tensor,
            source_index=torch.from_numpy(source_indices.astype(np.int64, copy=True)),
            input_slot=torch.from_numpy(self.table.input_slot[rows].astype(np.int64, copy=True)),
            clean_tile_index=torch.from_numpy(
                self.table.clean_tile_index[rows].astype(np.int64, copy=True)
            ),
            confidence=torch.from_numpy(
                self.table.joint_confidence[rows].astype(np.float32, copy=True)
            ),
            pair_row=torch.from_numpy(rows.copy()),
        )

    def sample(
        self,
        batch_size: int,
        *,
        generator: np.random.Generator | None = None,
    ) -> RealPairBatch:
        return self._materialize_rows(
            self.draw_pair_rows(batch_size, generator=generator)
        )

    def sample_grouped(
        self,
        batch_size: int,
        pairs_per_source: int,
        *,
        generator: np.random.Generator | None = None,
    ) -> RealPairBatch:
        return self._materialize_rows(
            self.draw_grouped_pair_rows(
                batch_size,
                pairs_per_source,
                generator=generator,
            )
        )

    def materialize_validation(
        self,
        *,
        source_indices: Sequence[int] | None = None,
        pairs_per_source: int | None = None,
        seed: int = 20260710,
    ) -> RealPairBatch:
        """Materialize a stable panel and retain original source_index values."""
        if source_indices is None:
            selected_sources = self.table.active_source_indices.tolist()
        else:
            selected_sources = [int(index) for index in source_indices]
            if len(selected_sources) != len(set(selected_sources)):
                raise ValueError("source_indices contains duplicates")
            active = set(self.table.active_source_indices.tolist())
            if any(index not in active for index in selected_sources):
                raise ValueError("source_indices contains an inactive or unknown source")
        if not selected_sources:
            raise ValueError("validation source list is empty")
        if pairs_per_source is not None and pairs_per_source <= 0:
            raise ValueError("pairs_per_source must be positive when provided")

        generator = np.random.default_rng(seed)
        row_parts = []
        for source_index in selected_sources:
            rows = self.table.source_rows(source_index)
            if pairs_per_source is not None and len(rows) > pairs_per_source:
                positions = np.sort(
                    generator.choice(len(rows), size=pairs_per_source, replace=False)
                )
                rows = rows[positions]
            row_parts.append(rows)
        return self._materialize_rows(np.concatenate(row_parts))

    def cache_info(self) -> CacheInfo:
        return self.cache.info()

    def clear_cache(self) -> None:
        self.cache.clear()

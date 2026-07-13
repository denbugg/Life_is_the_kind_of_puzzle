"""Frozen-DINO 4x4-superblock utilities for global puzzle placement.

The module deliberately keeps the pretrained backbone separate from the tiny
trainable set-to-position head.  A puzzle layout is partitioned into a 6x6 set
of rigid 4x4-tile superblocks.  The head predicts one coarse destination cell
per block, and a Hungarian projection preserves the one-to-one constraint.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn

from .compatibility import CompatibilityMatrices
from .geometry import GRID, TILE, TILE_COUNT, validate_permutation


SUPERBLOCK_TILES = 4
SUPERBLOCK_GRID = GRID // SUPERBLOCK_TILES
SUPERBLOCK_COUNT = SUPERBLOCK_GRID * SUPERBLOCK_GRID
SUPERBLOCK_PIXELS = SUPERBLOCK_TILES * TILE


def _validate_mapping(values: np.ndarray, *, name: str) -> np.ndarray:
    mapping = np.asarray(values)
    if mapping.shape != (SUPERBLOCK_COUNT,):
        raise ValueError(f"{name} must have shape {(SUPERBLOCK_COUNT,)}")
    if not np.issubdtype(mapping.dtype, np.integer):
        raise TypeError(f"{name} must be integral")
    mapping = mapping.astype(np.int32, copy=False)
    if not np.array_equal(np.sort(mapping), np.arange(SUPERBLOCK_COUNT)):
        raise ValueError(f"{name} must contain every coarse cell exactly once")
    return mapping


def position_tiles_to_superblocks(position_tiles: np.ndarray) -> np.ndarray:
    """Pack position-ordered 20x20 tiles into 36 row-major 80x80 blocks."""
    tiles = np.asarray(position_tiles)
    if tiles.shape != (TILE_COUNT, TILE, TILE, 3) or tiles.dtype != np.uint8:
        raise ValueError("position_tiles must be uint8 576x20x20x3")
    grid = tiles.reshape(GRID, GRID, TILE, TILE, 3)
    blocks = grid.reshape(
        SUPERBLOCK_GRID,
        SUPERBLOCK_TILES,
        SUPERBLOCK_GRID,
        SUPERBLOCK_TILES,
        TILE,
        TILE,
        3,
    ).transpose(0, 2, 1, 4, 3, 5, 6)
    return np.ascontiguousarray(
        blocks.reshape(
            SUPERBLOCK_COUNT, SUPERBLOCK_PIXELS, SUPERBLOCK_PIXELS, 3
        )
    )


def layout_superblocks(
    slot_tiles: np.ndarray, position_to_slot: np.ndarray
) -> np.ndarray:
    """Render the 36 rigid superblocks present in a tile layout."""
    tiles = np.asarray(slot_tiles)
    if tiles.shape != (TILE_COUNT, TILE, TILE, 3) or tiles.dtype != np.uint8:
        raise ValueError("slot_tiles must be uint8 576x20x20x3")
    layout = validate_permutation(position_to_slot, name="position_to_slot")
    return position_tiles_to_superblocks(tiles[layout])


def apply_superblock_mapping(
    position_to_slot: np.ndarray, source_to_destination: np.ndarray
) -> np.ndarray:
    """Move all 36 rigid blocks according to a source-block to cell mapping."""
    layout = validate_permutation(position_to_slot, name="position_to_slot")
    mapping = _validate_mapping(
        source_to_destination, name="source_to_destination"
    )
    source_grid = layout.reshape(GRID, GRID)
    destination_grid = np.empty_like(source_grid)
    for source, destination in enumerate(mapping.tolist()):
        source_row, source_column = divmod(source, SUPERBLOCK_GRID)
        destination_row, destination_column = divmod(
            destination, SUPERBLOCK_GRID
        )
        source_rows = slice(
            source_row * SUPERBLOCK_TILES,
            (source_row + 1) * SUPERBLOCK_TILES,
        )
        source_columns = slice(
            source_column * SUPERBLOCK_TILES,
            (source_column + 1) * SUPERBLOCK_TILES,
        )
        destination_rows = slice(
            destination_row * SUPERBLOCK_TILES,
            (destination_row + 1) * SUPERBLOCK_TILES,
        )
        destination_columns = slice(
            destination_column * SUPERBLOCK_TILES,
            (destination_column + 1) * SUPERBLOCK_TILES,
        )
        destination_grid[destination_rows, destination_columns] = source_grid[
            source_rows, source_columns
        ]
    return validate_permutation(
        destination_grid.reshape(-1), name="superblock_position_to_slot"
    )


@dataclass(frozen=True)
class SuperblockOracle:
    source_to_destination: np.ndarray
    overlap: np.ndarray
    attainable_tile_fraction: float


def oracle_superblock_mapping(
    position_to_slot: np.ndarray, slot_to_target: np.ndarray
) -> SuperblockOracle:
    """Maximum-overlap coarse assignment available to rigid 4x4 block moves."""
    layout = validate_permutation(position_to_slot, name="position_to_slot")
    truth = validate_permutation(slot_to_target, name="slot_to_target")
    layout_grid = layout.reshape(GRID, GRID)
    overlap = np.zeros(
        (SUPERBLOCK_COUNT, SUPERBLOCK_COUNT), dtype=np.int32
    )
    for source in range(SUPERBLOCK_COUNT):
        block_row, block_column = divmod(source, SUPERBLOCK_GRID)
        slots = layout_grid[
            block_row * SUPERBLOCK_TILES : (block_row + 1) * SUPERBLOCK_TILES,
            block_column
            * SUPERBLOCK_TILES : (block_column + 1) * SUPERBLOCK_TILES,
        ].reshape(-1)
        target_positions = truth[slots]
        target_cells = (
            (target_positions // GRID) // SUPERBLOCK_TILES * SUPERBLOCK_GRID
            + (target_positions % GRID) // SUPERBLOCK_TILES
        )
        overlap[source] = np.bincount(
            target_cells, minlength=SUPERBLOCK_COUNT
        )
    sources, destinations = linear_sum_assignment(-overlap)
    mapping = np.empty(SUPERBLOCK_COUNT, dtype=np.int32)
    mapping[sources] = destinations.astype(np.int32, copy=False)
    attainable = float(overlap[sources, destinations].sum() / TILE_COUNT)
    return SuperblockOracle(
        source_to_destination=mapping,
        overlap=overlap,
        attainable_tile_fraction=attainable,
    )


def hungarian_mapping(logits: np.ndarray | torch.Tensor) -> np.ndarray:
    """Project block-to-cell logits to a valid coarse permutation."""
    if isinstance(logits, torch.Tensor):
        values = logits.detach().float().cpu().numpy()
    else:
        values = np.asarray(logits, dtype=np.float32)
    if values.shape == (1, SUPERBLOCK_COUNT, SUPERBLOCK_COUNT):
        values = values[0]
    if values.shape != (SUPERBLOCK_COUNT, SUPERBLOCK_COUNT):
        raise ValueError("logits must be 36x36")
    if not np.isfinite(values).all():
        raise ValueError("logits must be finite")
    sources, destinations = linear_sum_assignment(-values.astype(np.float64))
    mapping = np.empty(SUPERBLOCK_COUNT, dtype=np.int32)
    mapping[sources] = destinations.astype(np.int32, copy=False)
    return _validate_mapping(mapping, name="hungarian_source_to_destination")


def coarse_assignment_metrics(
    predicted: np.ndarray,
    oracle: np.ndarray,
    *,
    baseline: np.ndarray | None = None,
) -> dict[str, float]:
    predicted = _validate_mapping(predicted, name="predicted")
    oracle = _validate_mapping(oracle, name="oracle")
    if baseline is None:
        baseline = np.arange(SUPERBLOCK_COUNT, dtype=np.int32)
    baseline = _validate_mapping(baseline, name="baseline")
    oracle_rows = oracle // SUPERBLOCK_GRID
    oracle_columns = oracle % SUPERBLOCK_GRID

    def manhattan(mapping: np.ndarray) -> np.ndarray:
        return np.abs(mapping // SUPERBLOCK_GRID - oracle_rows) + np.abs(
            mapping % SUPERBLOCK_GRID - oracle_columns
        )

    predicted_distance = manhattan(predicted)
    baseline_distance = manhattan(baseline)
    baseline_mean = float(baseline_distance.mean())
    predicted_mean = float(predicted_distance.mean())
    reduction = (
        (baseline_mean - predicted_mean) / baseline_mean
        if baseline_mean > 0.0
        else 0.0
    )
    return {
        "predicted_cell_accuracy": float(np.mean(predicted == oracle)),
        "baseline_cell_accuracy": float(np.mean(baseline == oracle)),
        "predicted_mean_manhattan": predicted_mean,
        "baseline_mean_manhattan": baseline_mean,
        "manhattan_reduction": float(reduction),
    }


def wrong_position_count(
    position_to_slot: np.ndarray, slot_to_target: np.ndarray
) -> int:
    layout = validate_permutation(position_to_slot, name="position_to_slot")
    truth = validate_permutation(slot_to_target, name="slot_to_target")
    return int(np.count_nonzero(truth[layout] != np.arange(TILE_COUNT)))


def layout_pair_cost(
    position_to_slot: np.ndarray, compatibility: CompatibilityMatrices
) -> float:
    layout = validate_permutation(position_to_slot, name="position_to_slot")
    grid = layout.reshape(GRID, GRID)
    values = np.concatenate(
        [
            compatibility.right[grid[:, :-1], grid[:, 1:]].reshape(-1),
            compatibility.down[grid[:-1, :], grid[1:, :]].reshape(-1),
        ]
    )
    finite = values[np.isfinite(values)]
    if len(finite) != 2 * GRID * (GRID - 1):
        raise RuntimeError("layout seam cost contains non-finite grid edges")
    return float(finite.mean(dtype=np.float64))


def seam_guarded_layout(
    baseline_layout: np.ndarray,
    candidate_layout: np.ndarray,
    compatibility: CompatibilityMatrices,
    *,
    max_ratio: float = 1.02,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    if max_ratio < 1.0 or not np.isfinite(max_ratio):
        raise ValueError("max_ratio must be finite and at least one")
    baseline_cost = layout_pair_cost(baseline_layout, compatibility)
    candidate_cost = layout_pair_cost(candidate_layout, compatibility)
    ratio = candidate_cost / max(baseline_cost, 1e-12)
    accepted = bool(ratio <= max_ratio + 1e-12)
    selected = candidate_layout if accepted else baseline_layout
    return validate_permutation(selected, name="guarded_layout").copy(), {
        "baseline_pair_cost": baseline_cost,
        "candidate_pair_cost": candidate_cost,
        "candidate_to_baseline_ratio": float(ratio),
        "max_ratio": float(max_ratio),
        "candidate_accepted": accepted,
    }


class DinoSetPositionHead(nn.Module):
    """Tiny permutation-equivariant head over 36 frozen DINO block vectors."""

    def __init__(
        self,
        *,
        feature_dim: int,
        model_dim: int = 128,
        layers: int = 2,
        heads: int = 4,
        feedforward_dim: int = 256,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or model_dim <= 0 or model_dim % heads:
            raise ValueError("invalid feature/model/head dimensions")
        if layers <= 0 or feedforward_dim <= 0:
            raise ValueError("layers and feedforward_dim must be positive")
        self.feature_dim = int(feature_dim)
        self.model_dim = int(model_dim)
        self.layers = int(layers)
        self.heads = int(heads)
        self.feedforward_dim = int(feedforward_dim)
        self.input_projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, model_dim),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=feedforward_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context = nn.TransformerEncoder(layer, num_layers=layers)
        self.output_norm = nn.LayerNorm(model_dim)
        self.cell_head = nn.Linear(model_dim, SUPERBLOCK_COUNT)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 2:
            features = features.unsqueeze(0)
        if features.ndim != 3 or features.shape[1:] != (
            SUPERBLOCK_COUNT,
            self.feature_dim,
        ):
            raise ValueError(
                "features must be Bx36xfeature_dim, got "
                f"{tuple(features.shape)}"
            )
        values = self.input_projection(features.float())
        return self.cell_head(self.output_norm(self.context(values)))

    def config(self) -> dict[str, int]:
        return {
            "feature_dim": self.feature_dim,
            "model_dim": self.model_dim,
            "layers": self.layers,
            "heads": self.heads,
            "feedforward_dim": self.feedforward_dim,
        }


class FrozenDinoFeatureAdapter(nn.Module):
    """Normalize hub/Hugging Face DINOv2 outputs to CLS plus mean patch token."""

    def __init__(self, backbone: nn.Module, *, backend: str) -> None:
        super().__init__()
        if backend not in {"torch_hub", "transformers"}:
            raise ValueError("backend must be torch_hub or transformers")
        self.backbone = backbone
        self.backend = backend

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        if self.backend == "torch_hub":
            outputs = self.backbone.forward_features(pixels)
            if isinstance(outputs, Mapping):
                cls = outputs.get("x_norm_clstoken")
                patches = outputs.get("x_norm_patchtokens")
                if cls is None or patches is None:
                    raise RuntimeError(
                        "official DINOv2 forward_features lacks normalized tokens"
                    )
            else:
                raise RuntimeError("unexpected official DINOv2 feature output")
        else:
            outputs = self.backbone(pixel_values=pixels)
            hidden = outputs.last_hidden_state
            cls = hidden[:, 0]
            patches = hidden[:, 1:]
        return torch.cat([cls, patches.mean(dim=1)], dim=1)


def state_dict_sha256(model: nn.Module) -> str:
    """Hash tensor values and metadata without serializing a second checkpoint."""
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def save_superblock_checkpoint(
    path: str | Path,
    model: DinoSetPositionHead,
    *,
    metadata: dict[str, Any],
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        torch.save(
            {
                "schema_version": 1,
                "kind": "puzzle_dino_vits14_superblock_position_head",
                "model_config": model.config(),
                "model_state": model.state_dict(),
                "metadata": metadata,
            },
            temporary,
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def load_superblock_checkpoint(
    path: str | Path, *, device: torch.device | str = "cpu"
) -> tuple[DinoSetPositionHead, dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if (
        payload.get("schema_version") != 1
        or payload.get("kind")
        != "puzzle_dino_vits14_superblock_position_head"
    ):
        raise ValueError("unsupported DINO superblock checkpoint")
    model = DinoSetPositionHead(**payload["model_config"])
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)
    return model, dict(payload.get("metadata", {}))


def synthetic_smoke() -> dict[str, Any]:
    """Small CPU-only invariant check; it never loads DINO weights or data."""
    truth_layout = np.arange(TILE_COUNT, dtype=np.int32)
    slot_to_target = np.arange(TILE_COUNT, dtype=np.int32)
    shuffle = np.roll(np.arange(SUPERBLOCK_COUNT, dtype=np.int32), 7)
    shuffled_layout = apply_superblock_mapping(truth_layout, shuffle)
    oracle = oracle_superblock_mapping(shuffled_layout, slot_to_target)
    restored = apply_superblock_mapping(
        shuffled_layout, oracle.source_to_destination
    )
    if not np.array_equal(restored, truth_layout):
        raise AssertionError("oracle superblock mapping did not restore truth")
    logits = np.full((SUPERBLOCK_COUNT, SUPERBLOCK_COUNT), -4.0, dtype=np.float32)
    logits[np.arange(SUPERBLOCK_COUNT), oracle.source_to_destination] = 4.0
    predicted = hungarian_mapping(logits)
    metrics = coarse_assignment_metrics(predicted, oracle.source_to_destination)
    if metrics["predicted_cell_accuracy"] != 1.0:
        raise AssertionError("Hungarian projection failed synthetic oracle")
    model = DinoSetPositionHead(feature_dim=16, model_dim=32, layers=1, heads=4)
    features = torch.randn(2, SUPERBLOCK_COUNT, 16)
    output = model(features)
    if output.shape != (2, SUPERBLOCK_COUNT, SUPERBLOCK_COUNT):
        raise AssertionError("unexpected set-head output shape")
    return {
        "superblock_count": SUPERBLOCK_COUNT,
        "restored_truth": True,
        "hungarian_accuracy": metrics["predicted_cell_accuracy"],
        "head_output_shape": list(output.shape),
    }


__all__ = [
    "DinoSetPositionHead",
    "FrozenDinoFeatureAdapter",
    "SUPERBLOCK_COUNT",
    "SUPERBLOCK_GRID",
    "SUPERBLOCK_PIXELS",
    "SUPERBLOCK_TILES",
    "SuperblockOracle",
    "apply_superblock_mapping",
    "coarse_assignment_metrics",
    "hungarian_mapping",
    "layout_pair_cost",
    "layout_superblocks",
    "load_superblock_checkpoint",
    "oracle_superblock_mapping",
    "position_tiles_to_superblocks",
    "save_superblock_checkpoint",
    "seam_guarded_layout",
    "state_dict_sha256",
    "synthetic_smoke",
    "wrong_position_count",
]

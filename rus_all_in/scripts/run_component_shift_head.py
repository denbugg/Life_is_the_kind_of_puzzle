#!/usr/bin/env python3
"""Train only the explicit component-shift head; never open a quality panel.

The parent absolute-coordinate checkpoint and its Socket d64 backbone are
strict-loaded and frozen.  Training boards come only from manifest ``train``
sources outside every ``*_filenames`` list in the complete parent checkpoint
and any explicitly supplied report.  Exact labels are synthetic shuffles of
those training sources.  This runner has no evaluation-panel mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import OrderedDict, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from aiijc_puzzle.absolute_coordinate_sorter import AbsoluteCoordinateSorter
from aiijc_puzzle.component_anchor_diagnostic import rebuild_decoder_components
from aiijc_puzzle.component_shift_head import (
    ComponentDescriptor,
    ComponentShiftHead,
    ComponentShiftOutput,
    ComponentShiftTarget,
    component_descriptors_from_decoder,
    component_shift_loss,
    dominant_component_shift_targets,
)
from aiijc_puzzle.protocol import (
    IMAGE_SIZE,
    compute_protocol_digest,
    select_manifest_records,
    sha256_file,
    split_tiles,
)
from aiijc_puzzle.socket_sorter_production import (
    LoadedSocketCheckpoint,
    choose_deterministic_device,
    load_socket_checkpoint,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data/interim/validation_manifest.json"
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
SELECTION_NAMESPACE = "aiijc-component-shift-head-training-v1"
GRID = 24
TILE_COUNT = GRID * GRID
COMPONENT_EDGE_BUDGET = 144
MAX_TRAIN_SOURCES = 2048
MAX_STEPS = 800
HEAD_HIDDEN_DIMENSION = 64
EXPECTED_HEAD_PARAMETERS = 60_208
GATE_TAIL_STEPS = 100
IMPURITY_WEIGHT_FLOOR = 0.10
GATE_MIN_SUPPORT_TILES = 4.0
GATE_ORACLE_HEADROOM_FRACTION = 0.10
GATE_AXIS_ACCURACY_MARGIN = 0.02
GATE_AXIS_NLL_GAIN = 0.02


@dataclass(frozen=True)
class LoadedAbsoluteCheckpoint:
    path: Path
    sha256: str
    payload: dict[str, Any]
    model: AbsoluteCoordinateSorter
    socket: LoadedSocketCheckpoint


class CleanTileCache:
    """Small bounded cache; a 2048-board uint8 preload would exceed 1 GiB."""

    def __init__(self, targets: Path, *, maximum_boards: int = 32) -> None:
        if maximum_boards <= 0:
            raise ValueError("maximum_boards must be positive")
        self.targets = targets
        self.maximum_boards = maximum_boards
        self.values: OrderedDict[str, np.ndarray] = OrderedDict()

    def load(self, record: Mapping[str, Any]) -> np.ndarray:
        filename = str(record["filename"])
        if filename in self.values:
            value = self.values.pop(filename)
            self.values[filename] = value
            return value
        path = self.targets / filename
        expected_hash = record.get("target_sha256")
        if not isinstance(expected_hash, str) or sha256_file(path) != expected_hash:
            raise ValueError(f"manifest target hash mismatch: {filename}")
        with Image.open(path) as image:
            image.load()
            if image.mode != "RGB" or image.size != (IMAGE_SIZE, IMAGE_SIZE):
                raise ValueError(f"expected RGB 480x480 target: {path}")
            tiles = split_tiles(np.asarray(image, dtype=np.uint8))
        self.values[filename] = tiles
        while len(self.values) > self.maximum_boards:
            self.values.popitem(last=False)
        return tiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=512)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--exclude-report",
        type=Path,
        action="append",
        default=[],
        help="additional artifact whose every key ending *_filenames is excluded",
    )
    return parser.parse_args()


def collect_filename_lists(value: Any, *, parent_key: str = "") -> set[str]:
    """Collect every recursively nested list whose key ends in ``_filenames``."""

    names: set[str] = set()
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            if key.endswith("_filenames"):
                if not isinstance(child, (list, tuple)) or not all(
                    isinstance(item, str) and item for item in child
                ):
                    raise ValueError(f"{key} must be a list of non-empty filenames")
                if len(set(child)) != len(child):
                    raise ValueError(f"{key} contains duplicate filenames")
                names.update(child)
            names.update(collect_filename_lists(child, parent_key=key))
    elif isinstance(value, (list, tuple)) and not parent_key.endswith("_filenames"):
        for child in value:
            names.update(collect_filename_lists(child, parent_key=parent_key))
    return names


def _positive_contract_integer(contract: Mapping[str, Any], key: str) -> int:
    value = contract.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"absolute checkpoint contract {key} must be a positive integer")
    return value


def load_absolute_checkpoint(
    path: Path,
    *,
    device: torch.device,
) -> LoadedAbsoluteCheckpoint:
    """Strict-load the frozen coordinate checkpoint and its hashed Socket parent."""

    resolved = path.resolve()
    payload: Any = torch.load(resolved, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("absolute checkpoint payload must be a dictionary")
    contract = payload.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("absolute checkpoint has no contract")
    if contract.get("architecture") != "socket-backed-absolute-coordinate-sorter-v1":
        raise ValueError("unsupported absolute checkpoint architecture")
    grid = _positive_contract_integer(contract, "grid")
    head_dimension = _positive_contract_integer(contract, "head_dimension")
    heads = _positive_contract_integer(contract, "heads")
    set_layers = _positive_contract_integer(contract, "set_layers")
    sinkhorn_iterations = _positive_contract_integer(contract, "sinkhorn_iterations")
    if grid != GRID or head_dimension != 32:
        raise ValueError("component-shift stage is frozen to grid24 final d32 tokens")
    if head_dimension % heads:
        raise ValueError("absolute checkpoint head_dimension is not divisible by heads")
    if contract.get("frozen_socket_backbone") is not True:
        raise ValueError("absolute checkpoint must freeze its Socket backbone")
    if contract.get("input_index_position_embedding") is not False:
        raise ValueError("absolute checkpoint does not prove input-index equivariance")
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("absolute checkpoint has no state_dict")

    socket_metadata = payload.get("socket_checkpoint")
    if not isinstance(socket_metadata, Mapping):
        raise ValueError("absolute checkpoint has no Socket lineage metadata")
    socket_path_value = socket_metadata.get("path")
    socket_hash = socket_metadata.get("sha256")
    if not isinstance(socket_path_value, str) or not isinstance(socket_hash, str):
        raise ValueError("absolute checkpoint Socket path/hash is malformed")
    socket_path = Path(socket_path_value).resolve()
    if sha256_file(socket_path) != socket_hash:
        raise ValueError("Socket checkpoint hash differs from absolute lineage")
    socket = load_socket_checkpoint(socket_path, device=device)
    if socket.sha256 != socket_hash:
        raise RuntimeError("strict Socket loader and absolute lineage hash disagree")
    model = AbsoluteCoordinateSorter(
        socket.model,
        grid=grid,
        head_dimension=head_dimension,
        heads=heads,
        set_layers=set_layers,
        sinkhorn_iterations=sinkhorn_iterations,
        freeze_backbone=True,
    ).to(device)
    model.load_state_dict(dict(state_dict), strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if not callable(getattr(model, "encode_coordinate_tokens", None)):
        raise RuntimeError(
            "AbsoluteCoordinateSorter must expose state-dict-neutral "
            "encode_coordinate_tokens()"
        )

    selection = payload.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("absolute checkpoint has no selection lineage")
    exposed = selection.get("lineage_exposed_filenames")
    if not isinstance(exposed, list) or not all(isinstance(name, str) for name in exposed):
        raise ValueError("absolute checkpoint full exposure lineage is malformed")
    if exposed != sorted(exposed) or len(exposed) != len(set(exposed)):
        raise ValueError("absolute checkpoint full exposure lineage must be sorted and unique")
    lineage_train = selection.get("lineage_train_filenames")
    if not isinstance(lineage_train, list) or not all(
        isinstance(name, str) and name for name in lineage_train
    ):
        raise ValueError("absolute checkpoint full training lineage is malformed")
    if lineage_train != sorted(lineage_train) or len(lineage_train) != len(
        set(lineage_train)
    ):
        raise ValueError("absolute checkpoint full training lineage must be sorted and unique")
    if not set(lineage_train).issubset(exposed):
        raise ValueError("absolute checkpoint training lineage is not a subset of exposure")
    all_declared = collect_filename_lists(payload)
    if not all_declared.issubset(set(exposed)):
        missing = sorted(all_declared - set(exposed))[:8]
        raise ValueError(f"full lineage does not cover every checkpoint filename list: {missing}")
    return LoadedAbsoluteCheckpoint(
        path=resolved,
        sha256=sha256_file(resolved),
        payload=payload,
        model=model,
        socket=socket,
    )


def select_training_records(
    manifest: Mapping[str, Any],
    checkpoint_payload: Mapping[str, Any],
    exclude_reports: list[Path],
    *,
    limit: int,
    checkpoint_sha256: str,
) -> tuple[tuple[Mapping[str, Any], ...], set[str], dict[str, int]]:
    """Select only manifest-train sources outside all declared filename lists."""

    forbidden = collect_filename_lists(checkpoint_payload)
    source_counts = {"absolute_checkpoint": len(forbidden)}
    for path in exclude_reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        names = collect_filename_lists(report)
        source_counts[str(path.resolve())] = len(names)
        forbidden.update(names)
    splits = manifest.get("splits")
    train_split = splits.get("train") if isinstance(splits, Mapping) else None
    if not isinstance(train_split, list):
        raise ValueError("manifest has no train split")
    ranked = select_manifest_records(
        dict(manifest),
        "train",
        limit=len(train_split),
        namespace=f"{SELECTION_NAMESPACE}\0{checkpoint_sha256}",
    )
    records = tuple(
        record for record in ranked if str(record["filename"]) not in forbidden
    )[:limit]
    if len(records) != limit:
        raise ValueError("could not form the requested source-disjoint training roster")
    selected = {str(record["filename"]) for record in records}
    if selected & forbidden:
        raise RuntimeError("component-shift training roster overlaps prior exposure")
    return records, forbidden, source_counts


def _names_digest(records: tuple[Mapping[str, Any], ...]) -> str:
    return _filename_digest(str(record["filename"]) for record in records)


def _filename_digest(names: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(names).encode()).hexdigest()


def _uniform(
    shape: tuple[int, ...],
    low: float,
    high: float,
    *,
    device: torch.device,
) -> torch.Tensor:
    return torch.empty(shape, device=device).uniform_(low, high)


def challenge_augment(clean: torch.Tensor) -> torch.Tensor:
    """Apply the frozen independent per-tile challenge-like corruption."""

    count = len(clean)
    gray = 0.299 * clean[:, :1] + 0.587 * clean[:, 1:2] + 0.114 * clean[:, 2:3]
    pivot = gray.mean(dim=(1, 2, 3), keepdim=True)
    scale = _uniform((count, 1, 1, 1), 0.70, 1.30, device=clean.device)
    offset = _uniform((count, 1, 1, 1), -30 / 255, 30 / 255, device=clean.device)
    value = scale * (clean - pivot) + pivot + offset
    sigma = _uniform((count, 1, 1, 1), 40 / 255, 55 / 255, device=clean.device)
    value = value + sigma * torch.randn_like(value)
    kernel = value.new_tensor([0.25, 0.5, 0.25])
    horizontal = kernel.reshape(1, 1, 1, 3).expand(3, 1, 1, 3)
    vertical = kernel.reshape(1, 1, 3, 1).expand(3, 1, 3, 1)
    value = F.conv2d(F.pad(value, (1, 1, 0, 0), mode="reflect"), horizontal, groups=3)
    value = F.conv2d(F.pad(value, (0, 0, 1, 1), mode="reflect"), vertical, groups=3)
    levels = _uniform((count, 1, 1, 1), 40.0, 72.0, device=clean.device)
    return (torch.round(value.clamp(0, 1) * levels) / levels).clamp(0.0, 1.0)


def synthetic_training_example(
    clean_tiles: np.ndarray,
    *,
    generator: np.random.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    clean = torch.from_numpy(clean_tiles.astype(np.float32)).permute(0, 3, 1, 2).to(device)
    dirty = challenge_augment(clean / 255.0)
    permutation = generator.permutation(TILE_COUNT).astype(np.int64)
    index = torch.from_numpy(permutation).to(device)
    return dirty[index].unsqueeze(0), index.unsqueeze(0)


def _support_table(
    component: ComponentDescriptor,
    input_tile_to_position: np.ndarray,
    *,
    grid: int,
) -> np.ndarray:
    feasible_rows = grid - component.height + 1
    feasible_columns = grid - component.width + 1
    support = np.zeros((feasible_rows, feasible_columns), dtype=np.float64)
    for tile, relative_row, relative_column in zip(
        component.tiles,
        component.relative_rows,
        component.relative_columns,
        strict=True,
    ):
        true_row, true_column = divmod(int(input_tile_to_position[tile]), grid)
        row_shift = true_row - relative_row
        column_shift = true_column - relative_column
        if 0 <= row_shift < feasible_rows and 0 <= column_shift < feasible_columns:
            support[row_shift, column_shift] += 1.0
    return support


def component_observations(
    output: ComponentShiftOutput,
    components: tuple[ComponentDescriptor, ...],
    targets: tuple[ComponentShiftTarget, ...],
    input_tile_to_position: torch.Tensor | np.ndarray,
    *,
    grid: int,
) -> list[dict[str, float | int | str | bool]]:
    """Return train-label diagnostics without producing or scoring a layout."""

    value: Any = input_tile_to_position
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    positions = np.asarray(value, dtype=np.int64)
    if positions.ndim == 2 and positions.shape[0] == 1:
        positions = positions[0]
    if positions.shape != (grid * grid,):
        raise ValueError("input_tile_to_position has the wrong shape")
    if not np.array_equal(np.sort(positions), np.arange(grid * grid)):
        raise ValueError("input_tile_to_position must be one exact permutation")
    records: list[dict[str, float | int | str | bool]] = []
    for index, (component, target) in enumerate(zip(components, targets, strict=True)):
        feasible_rows = output.feasible_row_shifts[index]
        feasible_columns = output.feasible_column_shifts[index]
        row_logits = output.row_logits[index, :feasible_rows]
        column_logits = output.column_logits[index, :feasible_columns]
        row_log_probability = F.log_softmax(row_logits, dim=0)
        column_log_probability = F.log_softmax(column_logits, dim=0)
        predicted_row = int(row_logits.argmax().detach())
        predicted_column = int(column_logits.argmax().detach())
        support = _support_table(component, positions, grid=grid)
        centre_row = (feasible_rows - 1) // 2
        centre_column = (feasible_columns - 1) // 2
        training_weight = component.size * (
            IMPURITY_WEIGHT_FLOOR + (1.0 - IMPURITY_WEIGHT_FLOOR) * target.purity
        )
        purity_bin = (
            "zero"
            if target.purity == 0
            else "low_0_0p5"
            if target.purity < 0.5
            else "majority_0p5_1"
            if target.purity < 1.0
            else "pure_1"
        )
        size_bin = (
            "singleton_1"
            if component.size == 1
            else "small_2_4"
            if component.size <= 4
            else "medium_5_16"
            if component.size <= 16
            else "large_17_plus"
        )
        row_uniform_nll = math.log(feasible_rows)
        column_uniform_nll = math.log(feasible_columns)
        records.append(
            {
                "size": component.size,
                "purity": target.purity,
                "purity_bin": purity_bin,
                "size_bin": size_bin,
                "training_weight": training_weight,
                "feasible_rows": feasible_rows,
                "feasible_columns": feasible_columns,
                "row_informative": feasible_rows > 1,
                "column_informative": feasible_columns > 1,
                "joint_informative": feasible_rows * feasible_columns > 1,
                "row_correct": int(predicted_row == target.target_row_shift),
                "column_correct": int(predicted_column == target.target_column_shift),
                "joint_correct": int(
                    predicted_row == target.target_row_shift
                    and predicted_column == target.target_column_shift
                ),
                "row_chance_accuracy": 1.0 / feasible_rows,
                "column_chance_accuracy": 1.0 / feasible_columns,
                "joint_chance_accuracy": 1.0 / (feasible_rows * feasible_columns),
                "row_nll": float(-row_log_probability[target.target_row_shift].detach()),
                "column_nll": float(
                    -column_log_probability[target.target_column_shift].detach()
                ),
                "row_uniform_nll": row_uniform_nll,
                "column_uniform_nll": column_uniform_nll,
                "joint_nll": float(
                    -row_log_probability[target.target_row_shift].detach()
                    - column_log_probability[target.target_column_shift].detach()
                ),
                "joint_uniform_nll": row_uniform_nll + column_uniform_nll,
                "predicted_support": float(support[predicted_row, predicted_column]),
                "chance_expected_support": float(support.mean()),
                "centre_support": float(support[centre_row, centre_column]),
                "dominant_oracle_support": target.support,
            }
        )
    return records


def _weighted_ratio(
    records: list[dict[str, Any]],
    numerator: str,
    denominator: str,
    *,
    informative: str,
) -> float | None:
    selected = [record for record in records if bool(record[informative])]
    denominator_sum = sum(
        float(record["training_weight"]) * float(record[denominator])
        for record in selected
    )
    if denominator_sum <= 0:
        return None
    return sum(
        float(record["training_weight"]) * float(record[numerator])
        for record in selected
    ) / denominator_sum


def _weighted_mean(
    records: list[dict[str, Any]],
    key: str,
    *,
    informative: str | None = None,
) -> float | None:
    selected = records if informative is None else [r for r in records if bool(r[informative])]
    weight = sum(float(record["training_weight"]) for record in selected)
    if weight <= 0:
        return None
    return sum(
        float(record["training_weight"]) * float(record[key]) for record in selected
    ) / weight


def aggregate_component_observations(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate size/purity-weighted accuracy and uniform-normalised NLL."""

    if not records:
        return {"component_count": 0, "tile_count": 0}
    row_nll_ratio = _weighted_ratio(
        records,
        "row_nll",
        "row_uniform_nll",
        informative="row_informative",
    )
    column_nll_ratio = _weighted_ratio(
        records,
        "column_nll",
        "column_uniform_nll",
        informative="column_informative",
    )
    joint_nll_ratio = _weighted_ratio(
        records,
        "joint_nll",
        "joint_uniform_nll",
        informative="joint_informative",
    )
    return {
        "component_count": len(records),
        "tile_count": int(sum(int(record["size"]) for record in records)),
        "training_weight_sum": float(sum(float(r["training_weight"]) for r in records)),
        "mean_target_purity": _weighted_mean(records, "purity"),
        "row_informative_components": sum(bool(r["row_informative"]) for r in records),
        "column_informative_components": sum(
            bool(r["column_informative"]) for r in records
        ),
        "row_accuracy": _weighted_mean(records, "row_correct", informative="row_informative"),
        "row_chance_accuracy": _weighted_mean(
            records,
            "row_chance_accuracy",
            informative="row_informative",
        ),
        "column_accuracy": _weighted_mean(
            records,
            "column_correct",
            informative="column_informative",
        ),
        "column_chance_accuracy": _weighted_mean(
            records,
            "column_chance_accuracy",
            informative="column_informative",
        ),
        "joint_accuracy": _weighted_mean(
            records,
            "joint_correct",
            informative="joint_informative",
        ),
        "joint_chance_accuracy": _weighted_mean(
            records,
            "joint_chance_accuracy",
            informative="joint_informative",
        ),
        "row_chance_normalized_nll": row_nll_ratio,
        "row_nll_gain_vs_uniform": None if row_nll_ratio is None else 1.0 - row_nll_ratio,
        "column_chance_normalized_nll": column_nll_ratio,
        "column_nll_gain_vs_uniform": (
            None if column_nll_ratio is None else 1.0 - column_nll_ratio
        ),
        "joint_chance_normalized_nll": joint_nll_ratio,
        "joint_nll_gain_vs_uniform": (
            None if joint_nll_ratio is None else 1.0 - joint_nll_ratio
        ),
    }


def aggregate_bins(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    names = sorted({str(record[key]) for record in records})
    return {
        name: aggregate_component_observations(
            [record for record in records if record[key] == name]
        )
        for name in names
    }


def board_support_summary(records: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "predicted_supported_tiles": float(sum(r["predicted_support"] for r in records)),
        "chance_expected_supported_tiles": float(
            sum(r["chance_expected_support"] for r in records)
        ),
        "centre_supported_tiles": float(sum(r["centre_support"] for r in records)),
        "dominant_oracle_supported_tiles": float(
            sum(r["dominant_oracle_support"] for r in records)
        ),
    }


def evaluate_training_only_gate(
    component_metrics: Mapping[str, Any],
    support: Mapping[str, float],
) -> dict[str, Any]:
    """Apply the preregistered capacity gate; passing only permits root review."""

    learned = float(support["predicted_supported_tiles_per_board"])
    chance = float(support["chance_expected_supported_tiles_per_board"])
    centre = float(support["centre_supported_tiles_per_board"])
    oracle = float(support["dominant_oracle_supported_tiles_per_board"])
    baseline = max(chance, centre)
    required_delta = max(
        GATE_MIN_SUPPORT_TILES,
        GATE_ORACLE_HEADROOM_FRACTION * max(oracle - baseline, 0.0),
    )
    row_accuracy = component_metrics.get("row_accuracy")
    row_chance = component_metrics.get("row_chance_accuracy")
    row_nll_gain = component_metrics.get("row_nll_gain_vs_uniform")
    column_accuracy = component_metrics.get("column_accuracy")
    column_chance = component_metrics.get("column_chance_accuracy")
    column_nll_gain = component_metrics.get("column_nll_gain_vs_uniform")
    row_pass = (
        row_accuracy is not None
        and row_chance is not None
        and row_nll_gain is not None
        and row_accuracy >= row_chance + GATE_AXIS_ACCURACY_MARGIN
        and row_nll_gain >= GATE_AXIS_NLL_GAIN
    )
    column_pass = (
        column_accuracy is not None
        and column_chance is not None
        and column_nll_gain is not None
        and column_accuracy >= column_chance + GATE_AXIS_ACCURACY_MARGIN
        and column_nll_gain >= GATE_AXIS_NLL_GAIN
    )
    support_pass = learned >= baseline + required_delta
    return {
        "status": "pass-await-root-review" if support_pass and row_pass and column_pass else "stop",
        "pass": bool(support_pass and row_pass and column_pass),
        "quality_panel_authorized": False,
        "support": {
            "learned": learned,
            "chance": chance,
            "centre": centre,
            "dominant_oracle": oracle,
            "stronger_baseline": baseline,
            "required_delta": required_delta,
            "observed_delta": learned - baseline,
            "pass": support_pass,
        },
        "row": {
            "accuracy": row_accuracy,
            "chance_accuracy": row_chance,
            "minimum_accuracy_margin": GATE_AXIS_ACCURACY_MARGIN,
            "nll_gain_vs_uniform": row_nll_gain,
            "minimum_nll_gain": GATE_AXIS_NLL_GAIN,
            "pass": row_pass,
        },
        "column": {
            "accuracy": column_accuracy,
            "chance_accuracy": column_chance,
            "minimum_accuracy_margin": GATE_AXIS_ACCURACY_MARGIN,
            "nll_gain_vs_uniform": column_nll_gain,
            "minimum_nll_gain": GATE_AXIS_NLL_GAIN,
            "pass": column_pass,
        },
        "interpretation": (
            "training-only capacity gate; passing does not promote the head and does not "
            "authorize this runner to open an exact quality panel"
        ),
    }


def _mean_board_support(boards: list[dict[str, float]]) -> dict[str, float]:
    return {
        f"{key}_per_board": float(np.mean([board[key] for board in boards]))
        for key in boards[0]
    }


def validate_args(args: argparse.Namespace) -> None:
    if not 1 <= args.train_limit <= MAX_TRAIN_SOURCES:
        raise ValueError(f"train-limit must be in [1, {MAX_TRAIN_SOURCES}]")
    if not 1 <= args.steps <= MAX_STEPS:
        raise ValueError(f"steps must be in [1, {MAX_STEPS}]")
    if args.log_every <= 0:
        raise ValueError("log-every must be positive")
    for name in ("learning_rate", "weight_decay"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name.replace('_', '-')} must be finite and non-negative")
    if args.learning_rate == 0:
        raise ValueError("learning-rate must be positive")


def main() -> None:
    args = parse_args()
    validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_deterministic_device(args.device)
    parent = load_absolute_checkpoint(args.checkpoint, device=device)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records, forbidden, exclusion_counts = select_training_records(
        manifest,
        parent.payload,
        args.exclude_report,
        limit=args.train_limit,
        checkpoint_sha256=parent.sha256,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "component_shift_head.pt"
    report_path = output_dir / "report.json"
    if checkpoint_path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite an existing training artifact")

    head = ComponentShiftHead(
        32,
        grid=GRID,
        hidden_dimension=HEAD_HIDDEN_DIMENSION,
    ).to(device)
    trainable_parameters = sum(parameter.numel() for parameter in head.parameters())
    if trainable_parameters != EXPECTED_HEAD_PARAMETERS:
        raise RuntimeError("component-shift head no longer matches its frozen 60k contract")
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        args.steps,
        eta_min=args.learning_rate * 0.08,
    )
    cache = CleanTileCache(args.targets)
    generator = np.random.default_rng(args.seed + 1)
    tail: deque[tuple[list[dict[str, Any]], dict[str, float]]] = deque(
        maxlen=min(GATE_TAIL_STEPS, args.steps)
    )
    log_history: list[dict[str, Any]] = []
    recent_losses: list[float] = []
    runtime = {
        "target_io_and_corruption": 0.0,
        "frozen_coordinate_encoding": 0.0,
        "decoder144_component_rebuild": 0.0,
        "component_head_forward_backward": 0.0,
    }
    started = perf_counter()
    head.train()
    for step in range(args.steps):
        record = records[int(generator.integers(len(records)))]
        phase = perf_counter()
        clean_tiles = cache.load(record)
        tiles, target = synthetic_training_example(
            clean_tiles,
            generator=generator,
            device=device,
        )
        runtime["target_io_and_corruption"] += perf_counter() - phase

        phase = perf_counter()
        with torch.no_grad():
            tile_tokens, socket_output = parent.model.encode_coordinate_tokens(tiles)
        if tile_tokens.shape != (1, TILE_COUNT, 32) or not torch.isfinite(tile_tokens).all():
            raise RuntimeError("public coordinate token contract must be finite Bx576x32")
        runtime["frozen_coordinate_encoding"] += perf_counter() - phase

        phase = perf_counter()
        component_build = rebuild_decoder_components(
            socket_output.right_log_assignment,
            socket_output.down_log_assignment,
            grid=GRID,
            edge_budget_per_axis=COMPONENT_EDGE_BUDGET,
        )
        components = component_descriptors_from_decoder(component_build, grid=GRID)
        targets = dominant_component_shift_targets(components, target, grid=GRID)
        runtime["decoder144_component_rebuild"] += perf_counter() - phase

        phase = perf_counter()
        output = head(tile_tokens.detach(), components)
        loss, loss_diagnostics = component_shift_loss(
            output,
            targets,
            impurity_weight_floor=IMPURITY_WEIGHT_FLOOR,
        )
        observations = component_observations(
            output,
            components,
            targets,
            target,
            grid=GRID,
        )
        support = board_support_summary(observations)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0))
        optimizer.step()
        scheduler.step()
        runtime["component_head_forward_backward"] += perf_counter() - phase
        tail.append((observations, support))
        recent_losses.append(float(loss.detach()))

        if step == 0 or (step + 1) % args.log_every == 0 or step + 1 == args.steps:
            tail_boards = list(tail)[-min(args.log_every, len(tail)) :]
            window_observations = [item for board, _ in tail_boards for item in board]
            window_metrics = aggregate_component_observations(window_observations)
            window_support = _mean_board_support([summary for _, summary in tail_boards])
            row = {
                "step": step + 1,
                "mean_loss": float(np.mean(recent_losses)),
                "gradient_norm": gradient_norm,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "row_accuracy": window_metrics.get("row_accuracy"),
                "column_accuracy": window_metrics.get("column_accuracy"),
                "joint_accuracy": window_metrics.get("joint_accuracy"),
                **window_support,
                "elapsed_seconds": perf_counter() - started,
            }
            log_history.append(row)
            print(json.dumps({"event": "train", **row}), flush=True)
            recent_losses.clear()

    training_seconds = perf_counter() - started
    tail_observations = [item for board, _ in tail for item in board]
    tail_boards = [summary for _, summary in tail]
    component_metrics = aggregate_component_observations(tail_observations)
    support_metrics = _mean_board_support(tail_boards)
    purity_bins = aggregate_bins(tail_observations, "purity_bin")
    size_bins = aggregate_bins(tail_observations, "size_bin")
    gate = evaluate_training_only_gate(component_metrics, support_metrics)

    parent_selection = parent.payload["selection"]
    parent_lineage_train = parent_selection.get(
        "lineage_train_filenames",
        parent_selection.get("train_filenames"),
    )
    if not isinstance(parent_lineage_train, list) or not all(
        isinstance(name, str) and name for name in parent_lineage_train
    ):
        raise ValueError("absolute checkpoint training lineage is malformed")
    selected_names = {str(record["filename"]) for record in records}
    lineage_train = sorted(set(parent_lineage_train) | selected_names)
    lineage_exposed = sorted(forbidden | selected_names)
    selection = {
        "namespace": SELECTION_NAMESPACE,
        "train_filenames": [str(record["filename"]) for record in records],
        "train_digest": _names_digest(records),
        "lineage_train_filenames": lineage_train,
        "lineage_train_digest": _filename_digest(lineage_train),
        "lineage_exposed_filenames": lineage_exposed,
        "lineage_exposed_digest": _filename_digest(lineage_exposed),
        "parent_checkpoint_forbidden_count": len(collect_filename_lists(parent.payload)),
        "total_forbidden_count": len(forbidden),
    }
    head_contract = {
        "architecture": "component-conditioned-feasible-shift-head-v1",
        "grid": GRID,
        "tile_dimension": 32,
        "hidden_dimension": HEAD_HIDDEN_DIMENSION,
        "parameters": trainable_parameters,
        "component_source": "frozen default decoder144 components",
        "member_pooling": "mean+max permutation-invariant",
        "component_features": (
            "normalized relative row/column, size/log-size/height/width/density/"
            "singleton/mean accepted-edge confidence, global mean board token"
        ),
        "target": "dominant feasible exact translation mode, including impure components",
        "loss_weight": "size*(0.10+0.90*purity)",
        "inference_output": "tile-by-slot component_shift_unary",
        "input_index_position_embedding": False,
    }
    torch.save(
        {
            "state_dict": head.state_dict(),
            "contract": head_contract,
            "absolute_checkpoint": {
                "path": str(parent.path),
                "sha256": parent.sha256,
            },
            "socket_checkpoint": {
                "path": str(parent.socket.path),
                "sha256": parent.socket.sha256,
            },
            "selection": selection,
            "training_only_gate": gate,
        },
        checkpoint_path,
    )
    report = {
        "experiment": head_contract["architecture"],
        "status": (
            "training-only-gate-pass-awaiting-root-review"
            if gate["pass"]
            else "training-only-gate-fail-stop"
        ),
        "contract": head_contract,
        "configuration": {
            key: [str(path) for path in value]
            if key == "exclude_report"
            else str(value)
            if isinstance(value, Path)
            else value
            for key, value in vars(args).items()
        }
        | {
            "device_resolved": str(device),
            "component_edge_budget": COMPONENT_EDGE_BUDGET,
            "gate_tail_steps": GATE_TAIL_STEPS,
            "impurity_weight_floor": IMPURITY_WEIGHT_FLOOR,
        },
        "protocol": {
            "manifest_digest": compute_protocol_digest(manifest),
            "manifest_train_split_only": True,
            "dirty_challenge_like_synthetic_inputs": True,
            "exact_truth_source": "known shuffle of selected training sources",
            "absolute_checkpoint_and_socket_backbone_frozen": True,
            "all_recursive_filename_lists_excluded": True,
            "full_parent_checkpoint_lineage_excluded": True,
            "evaluation_panel_selected": False,
            "quality_panel_opened": False,
            "calibration_opened": False,
            "holdout_opened": False,
            "competition_test_opened": False,
            "gate_predeclared_before_first_training_run": True,
        },
        "parents": {
            "absolute_checkpoint": {"path": str(parent.path), "sha256": parent.sha256},
            "socket_checkpoint": {
                "path": str(parent.socket.path),
                "sha256": parent.socket.sha256,
            },
        },
        "selection": selection | {"exclusion_source_counts": exclusion_counts},
        "model": {
            "trainable_parameters": trainable_parameters,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
        },
        "training": {
            "steps": args.steps,
            "log_history": log_history,
            "tail_window_steps": len(tail),
            "tail_component_metrics": component_metrics,
            "tail_chance_normalized_nll_by_purity_bin": purity_bins,
            "tail_chance_normalized_nll_by_size_bin": size_bins,
            "tail_supported_tiles": support_metrics,
        },
        "training_only_gate": gate,
        "runtime_seconds": runtime
        | {
            "training_wall": training_seconds,
            "total": perf_counter() - started,
        },
        "verdict": (
            "Root review is required before any fresh exact quality panel. A gate pass "
            "is capacity evidence only; a gate failure stops this fallback."
        ),
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "event": "complete",
                "report": str(report_path),
                "checkpoint": str(checkpoint_path),
                "gate": gate,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

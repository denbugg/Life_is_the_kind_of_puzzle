#!/usr/bin/env python3
"""Run the preregistered Rank2 sparse BorderGraph-QAP experiment.

The pilot consumes frozen Socket, full-resolution denoiser, component-relation,
and relation-fusion evidence.  Candidate predictions are produced without
exact labels and persisted before the source-disjoint exact16 references are
scored.  This runner cannot access calibration, holdout, or competition test
images and never renders restored matcher pixels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from aiijc_puzzle.component_relation_reranker import (
    ComponentRelationCandidate,
    ComponentRelationReranker,
    extract_frozen_socket_context,
)
from aiijc_puzzle.fullres_boundary_denoiser import (
    FullResolutionBoundaryDenoiser,
    FullResolutionDenoiserConfig,
    restore_matcher_view,
)
from aiijc_puzzle.fullres_relation_fusion import (
    FullresRelationFusion,
    fusion_feature_names,
)
from aiijc_puzzle.layout_evaluation import evaluate_layout
from aiijc_puzzle.protocol import compute_protocol_digest, sha256_file
from aiijc_puzzle.socket_decoder import SocketDecoderConfig, decode_socket_assignments
from aiijc_puzzle.socket_sorter_production import (
    LoadedSocketCheckpoint,
    load_socket_checkpoint,
)
from aiijc_puzzle.socket_translation_placer import (
    CyclicTranslationConfig,
    select_global_cyclic_translation,
)
from aiijc_puzzle.sparse_bordergraph_qap import (
    SparseBorderGraphQAP,
    decode_hungarian,
    qap_training_loss,
    total_layout_energy,
)
from aiijc_puzzle.synthetic_socket_evaluation import (
    names_digest,
    select_source_disjoint_train_records,
)

try:
    from scripts.run_component_relation_reranker import (
        GRID,
        CleanTileCache,
        PreparedCase,
        _tile_tensor,
        prepare_case,
    )
    from scripts.run_fullres_relation_fusion import prepare_fusion_board
except ModuleNotFoundError:  # Direct ``python scripts/run_*.py`` execution.
    from run_component_relation_reranker import (
        GRID,
        CleanTileCache,
        PreparedCase,
        _tile_tensor,
        prepare_case,
    )
    from run_fullres_relation_fusion import prepare_fusion_board


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/sparse_bordergraph_qap_preregistered_v1.json"
DEFAULT_MANIFEST = PROJECT_ROOT / "data/interim/validation_manifest.json"
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/sparse-bordergraph-qap"
TILE_COUNT = GRID * GRID
EDGE_FEATURE_DIMENSION = 16
TILE_FEATURE_DIMENSION = 142


@dataclass(frozen=True)
class FrozenModels:
    socket: LoadedSocketCheckpoint
    relation: ComponentRelationReranker
    denoiser: FullResolutionBoundaryDenoiser
    fusion: FullresRelationFusion
    metadata: dict[str, Any]


@dataclass(frozen=True)
class BlindEvidence:
    case_id: str
    source_filename: str
    dirty_tiles_sha256: str
    tile_features: np.ndarray
    edge_features: np.ndarray
    edge_sources: np.ndarray
    edge_targets: np.ndarray
    edge_directions: np.ndarray
    baseline_layout: np.ndarray
    baseline_cyclic_layout: np.ndarray
    right_log_assignment: np.ndarray
    down_log_assignment: np.ndarray
    runtime_seconds: dict[str, float]


@dataclass(frozen=True)
class TrainingExample:
    evidence: BlindEvidence
    reference_layout: np.ndarray


@dataclass(frozen=True)
class FrozenPrediction:
    evidence: BlindEvidence
    candidate_layout: np.ndarray
    candidate_cyclic_layout: np.ndarray
    candidate_energy: float
    baseline_energy: float
    runtime_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("capacity", "benchmark", "pilot"), required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--allow-nondeterministic-mps", action="store_true")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--inference-batch", type=int, default=576)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema") != "aiijc-sparse-bordergraph-qap-preregistered-v1":
        raise ValueError("unexpected sparse QAP preregistration schema")
    if config.get("status") != "frozen-before-capacity-benchmark-fit-or-exact16-access":
        raise ValueError("sparse QAP preregistration timing contract changed")
    architecture = config["architecture"]
    if int(architecture["unrolled_qap_steps"]) > 2:
        raise ValueError("initial bounded pilot permits at most two QAP steps")
    selection = config["selection"]
    if int(selection["fit_sources"]) > int(config["training"]["maximum_fit_sources"]):
        raise ValueError("fit roster exceeds the bounded Rank2 source cap")
    if selection["manifest_split"] != "train":
        raise ValueError("Rank2 labels are restricted to organizer-train")


def _choose_device(name: str, *, allow_nondeterministic_mps: bool) -> torch.device:
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        if not allow_nondeterministic_mps:
            raise ValueError(
                "MPS fusion inference requires explicit nondeterminism acknowledgement"
            )
        torch.use_deterministic_algorithms(False)
    elif allow_nondeterministic_mps:
        raise ValueError("allow-nondeterministic-mps requires --device mps")
    else:
        torch.use_deterministic_algorithms(True)
    return torch.device(name)


def _validate_sha(path: Path, expected: str, *, name: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{name} checkpoint SHA-256 mismatch: {observed} != {expected}")


def _load_frozen_models(
    config: Mapping[str, Any],
    *,
    device: torch.device,
) -> FrozenModels:
    frozen = config["frozen_inputs"]
    resolved = {
        name: PROJECT_ROOT / str(contract["path"])
        for name, contract in frozen.items()
    }
    for name, path in resolved.items():
        _validate_sha(path, str(frozen[name]["sha256"]), name=name)

    socket = load_socket_checkpoint(resolved["socket_checkpoint"], device=device)
    relation_payload = torch.load(
        resolved["component_relation_checkpoint"],
        map_location="cpu",
        weights_only=True,
    )
    relation_contract = relation_payload["contract"]
    relation = ComponentRelationReranker(
        int(relation_contract["tile_dimension"]),
        grid=int(relation_contract["grid"]),
        hidden_dimension=int(relation_contract["hidden_dimension"]),
    )
    relation.load_state_dict(relation_payload["state_dict"], strict=True)
    relation.to(device).eval().requires_grad_(False)

    denoiser_payload = torch.load(
        resolved["fullres_denoiser_checkpoint"],
        map_location="cpu",
        weights_only=True,
    )
    denoiser_contract = denoiser_payload["contract"]
    denoiser = FullResolutionBoundaryDenoiser(
        FullResolutionDenoiserConfig(**denoiser_contract["model_config"])
    )
    denoiser.load_state_dict(denoiser_payload["state_dict"], strict=True)
    denoiser.to(device).eval().requires_grad_(False)

    fusion_payload = torch.load(
        resolved["fullres_relation_fusion_checkpoint"],
        map_location="cpu",
        weights_only=True,
    )
    fusion_contract = fusion_payload["contract"]
    if int(fusion_contract["feature_dimension"]) != len(fusion_feature_names()):
        raise ValueError("fusion feature contract changed")
    fusion = FullresRelationFusion(
        int(fusion_contract["feature_dimension"]),
        hidden_dimension=int(fusion_contract["hidden_dimension"]),
        residual_limit=float(fusion_contract["residual_limit"]),
    )
    fusion.load_state_dict(fusion_payload["state_dict"], strict=True)
    fusion.to(device).eval().requires_grad_(False)
    metadata = {
        name: {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
        for name, path in resolved.items()
    }
    return FrozenModels(socket, relation, denoiser, fusion, metadata)


def _collect_png_filenames(value: Any, *, parent_key: str = "") -> set[str]:
    names: set[str] = set()
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            names.update(_collect_png_filenames(child, parent_key=key))
    elif isinstance(value, (list, tuple)):
        if "filename" in parent_key:
            names.update(
                Path(item).name
                for item in value
                if isinstance(item, str) and item.lower().endswith(".png")
            )
        for child in value:
            if isinstance(child, (Mapping, list, tuple)):
                names.update(_collect_png_filenames(child, parent_key=parent_key))
    elif (
        isinstance(value, str)
        and "filename" in parent_key
        and value.lower().endswith(".png")
    ):
        names.add(Path(value).name)
    return names


_PANEL_FILENAME_KEYS = frozenset(
    {
        "eval_filenames",
        "reuse_eval_filenames",
        "source_filenames",
        "confirm_source_filenames",
        "local_eval_filenames",
        "confirm_filenames",
        "decoder_filenames",
        "decoder_reserved_filenames",
        "terminal_filenames",
        "evaluation_source_filenames",
        "selection_filenames",
        "selected_filenames",
        "source_filename",
        "filename",
    }
)


def _collect_panel_filenames(value: Any, *, parent_key: str = "") -> set[str]:
    """Collect actual panel/case fields without swallowing all train-only lineage."""

    names: set[str] = set()
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            names.update(_collect_panel_filenames(child, parent_key=key))
    elif isinstance(value, (list, tuple)):
        if parent_key in _PANEL_FILENAME_KEYS:
            names.update(
                Path(item).name
                for item in value
                if isinstance(item, str) and item.lower().endswith(".png")
            )
        for child in value:
            if isinstance(child, (Mapping, list, tuple)):
                names.update(_collect_panel_filenames(child, parent_key=parent_key))
    elif (
        isinstance(value, str)
        and parent_key in _PANEL_FILENAME_KEYS
        and value.lower().endswith(".png")
    ):
        names.add(Path(value).name)
    return names


def _panel_json_exclusion() -> tuple[set[str], dict[str, Any]]:
    paths = sorted(
        set((PROJECT_ROOT / "configs").rglob("*.json"))
        | set((PROJECT_ROOT / "outputs").rglob("*.json"))
    )
    filenames: set[str] = set()
    receipts: list[str] = []
    parsed = 0
    for path in paths:
        try:
            payload = _load_json(path)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            continue
        parsed += 1
        filenames.update(_collect_panel_filenames(payload))
        receipts.append(f"{path.relative_to(PROJECT_ROOT)}\0{sha256_file(path)}")
    return filenames, {
        "json_files_discovered": len(paths),
        "json_files_parsed": parsed,
        "json_snapshot_digest": hashlib.sha256("\n".join(receipts).encode()).hexdigest(),
        "panel_png_count": len(filenames),
        "panel_png_digest": names_digest(sorted(filenames), sort_names=True),
    }


def _validate_required_commitments(config: Mapping[str, Any]) -> list[dict[str, str]]:
    receipts: list[dict[str, str]] = []
    for item in config["selection"].get("required_current_commitments", []):
        path = PROJECT_ROOT / str(item["path"])
        observed = sha256_file(path)
        if observed != item["sha256"]:
            raise ValueError(
                f"required current commitment changed: {path}: {observed} != {item['sha256']}"
            )
        receipts.append({"path": str(path.resolve()), "sha256": observed})
    return receipts


def _actual_checkpoint_filenames(name: str, payload: Mapping[str, Any]) -> set[str]:
    """Return actual model exposure, never a broad exclusion/pool declaration."""

    selection = payload.get("selection")
    if not isinstance(selection, Mapping):
        return set()
    keys = {
        "component_relation_checkpoint": ("fit_filenames", "local_eval_filenames"),
        "fullres_denoiser_checkpoint": ("train_filenames", "eval_filenames"),
        "fullres_relation_fusion_checkpoint": ("fit_filenames", "eval_filenames"),
    }.get(name, ())
    result: set[str] = set()
    for key in keys:
        value = selection.get(key, [])
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError(f"checkpoint selection field {key!r} is malformed")
        result.update(Path(item).name for item in value)
    return result


def _commit_selection(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    models: FrozenModels,
    *,
    output_dir: Path,
) -> tuple[tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...], dict[str, Any]]:
    excluded, audit = _panel_json_exclusion()
    required_commitments = _validate_required_commitments(config)
    excluded.update(models.socket.lineage.exposed_filenames)
    # Binary downstream checkpoints also carry explicit source rosters.
    checkpoint_exposure: dict[str, int] = {
        "socket_checkpoint": len(models.socket.lineage.exposed_filenames)
    }
    for name, contract in config["frozen_inputs"].items():
        if name == "socket_checkpoint":
            continue
        payload = torch.load(
            PROJECT_ROOT / str(contract["path"]),
            map_location="cpu",
            weights_only=False,
        )
        actual = _actual_checkpoint_filenames(name, payload)
        checkpoint_exposure[name] = len(actual)
        excluded.update(actual)
    selection = config["selection"]
    fit_count = int(selection["fit_sources"])
    eval_count = int(selection["evaluation_sources"])
    records = select_source_disjoint_train_records(
        manifest,
        excluded_filenames=tuple(sorted(excluded)),
        limit=fit_count + eval_count,
        seed=int(selection["seed"]),
        namespace=str(selection["namespace"]),
    )
    fit = tuple(records[:fit_count])
    evaluation = tuple(records[fit_count:])
    fit_names = tuple(str(record["filename"]) for record in fit)
    eval_names = tuple(str(record["filename"]) for record in evaluation)
    if set(fit_names) & set(eval_names) or (set(fit_names) | set(eval_names)) & excluded:
        raise RuntimeError("Rank2 source-disjoint roster invariant failed")
    commitment = {
        "schema": "aiijc-sparse-bordergraph-qap-selection-v1",
        "written_before_any_selected_target_access": True,
        "manifest_split": "train",
        "namespace": selection["namespace"],
        "seed": selection["seed"],
        "exclusion_audit": audit,
        "required_current_commitments": required_commitments,
        "actual_checkpoint_exposure_counts": checkpoint_exposure,
        "excluded_filename_count": len(excluded),
        "excluded_filename_digest": names_digest(sorted(excluded), sort_names=True),
        "fit_source_filenames": list(fit_names),
        "fit_source_digest": names_digest(fit_names),
        "evaluation_source_filenames": list(eval_names),
        "evaluation_source_digest": names_digest(eval_names),
        "fit_eval_overlap": 0,
        "excluded_overlap": 0,
        "calibration_opened": False,
        "holdout_opened": False,
        "competition_test_opened": False,
    }
    path = output_dir / "selection_commitment.json"
    path.write_text(
        json.dumps(commitment, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    commitment["path"] = str(path)
    commitment["sha256"] = sha256_file(path)
    print(json.dumps({"event": "selection-committed", **commitment}), flush=True)
    return fit, evaluation, commitment


def _numpy(value: Any, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == len(shape) + 1 and array.shape[0] == 1:
        array = array[0]
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must have finite shape {shape}, got {array.shape}")
    return array


def _standardise_matrix(value: np.ndarray) -> np.ndarray:
    mask = ~np.eye(len(value), dtype=bool)
    selected = value[mask]
    return (value - float(selected.mean())) / max(float(selected.std()), 1e-6)


def _standardise_columns(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    mean = array.mean(axis=0, keepdims=True)
    scale = np.maximum(array.std(axis=0, keepdims=True), 1e-5)
    return np.ascontiguousarray((array - mean) / scale, dtype=np.float32)


def _ranks(value: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(value)
    masked = value.copy()
    masked[np.arange(count), np.arange(count)] = -np.inf
    row_order = np.argsort(-masked, axis=1, kind="stable")
    column_order = np.argsort(-masked, axis=0, kind="stable")
    row_rank = np.empty((count, count), dtype=np.int32)
    column_rank = np.empty((count, count), dtype=np.int32)
    rank = np.arange(count, dtype=np.int32)
    row_rank[np.arange(count)[:, None], row_order] = rank[None, :]
    column_rank[column_order, np.arange(count)[None, :]] = rank[:, None]
    return row_order, row_rank, column_rank


def _component_internal_edges(
    components: Sequence[Any],
) -> set[tuple[int, int, int]]:
    result: set[tuple[int, int, int]] = set()
    for component in components:
        by_coordinate = {
            (int(row), int(column)): int(tile)
            for tile, row, column in zip(
                component.tiles,
                component.relative_rows,
                component.relative_columns,
                strict=True,
            )
        }
        for (row, column), tile in by_coordinate.items():
            right = by_coordinate.get((row, column + 1))
            down = by_coordinate.get((row + 1, column))
            if right is not None:
                result.add((0, tile, right))
            if down is not None:
                result.add((1, tile, down))
    return result


def _canonical_contact(
    candidate: ComponentRelationCandidate,
    source_tile: int,
    target_tile: int,
) -> tuple[int, int, int]:
    if candidate.direction == "right":
        return 0, source_tile, target_tile
    if candidate.direction == "left":
        return 0, target_tile, source_tile
    if candidate.direction == "down":
        return 1, source_tile, target_tile
    if candidate.direction == "up":
        return 1, target_tile, source_tile
    raise ValueError(f"unknown component direction {candidate.direction!r}")


def _relation_contact_evidence(
    candidates: Sequence[ComponentRelationCandidate],
    scores: np.ndarray,
    confidence: np.ndarray,
) -> dict[tuple[int, int, int], tuple[float, float, float]]:
    grouped: dict[tuple[int, str], list[int]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        grouped[candidate.query_key].append(index)
    evidence: dict[tuple[int, int, int], tuple[float, float, float]] = {}
    for indices in grouped.values():
        local = scores[indices]
        local_z = (local - float(local.mean())) / max(float(local.std()), 1e-6)
        best = indices[int(np.argmax(local))]
        query_confidence = math.tanh(float(confidence[best]) / 4.0)
        for position, index in enumerate(indices):
            candidate = candidates[index]
            top = float(index == best)
            for contact in candidate.contacts:
                key = _canonical_contact(
                    candidate,
                    int(contact.source_tile),
                    int(contact.target_tile),
                )
                current = evidence.get(key)
                proposed = (float(local_z[position]), query_confidence, top)
                if current is None or (proposed[2], proposed[1], proposed[0]) > (
                    current[2],
                    current[1],
                    current[0],
                ):
                    evidence[key] = proposed
    return evidence


def build_sparse_tile_graph(
    raw_output: Any,
    restored_output: Any,
    components: Sequence[Any],
    relation_candidates: Sequence[ComponentRelationCandidate],
    fusion_scores: np.ndarray,
    fusion_confidence: np.ndarray,
    *,
    topk: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the frozen target-free top-k right/down tile graph."""

    if not 1 <= topk < TILE_COUNT:
        raise ValueError("topk must be in [1, 575]")
    raw_matrices = (
        _numpy(raw_output.right_raw, shape=(TILE_COUNT, TILE_COUNT), name="raw_right"),
        _numpy(raw_output.down_raw, shape=(TILE_COUNT, TILE_COUNT), name="raw_down"),
    )
    restored_matrices = (
        _numpy(
            restored_output.right_raw,
            shape=(TILE_COUNT, TILE_COUNT),
            name="restored_right",
        ),
        _numpy(
            restored_output.down_raw,
            shape=(TILE_COUNT, TILE_COUNT),
            name="restored_down",
        ),
    )
    raw_ot = (
        _numpy(
            raw_output.right_log_assignment,
            shape=(TILE_COUNT + 1, TILE_COUNT + 1),
            name="raw_right_ot",
        )[:TILE_COUNT, :TILE_COUNT],
        _numpy(
            raw_output.down_log_assignment,
            shape=(TILE_COUNT + 1, TILE_COUNT + 1),
            name="raw_down_ot",
        )[:TILE_COUNT, :TILE_COUNT],
    )
    restored_ot = (
        _numpy(
            restored_output.right_log_assignment,
            shape=(TILE_COUNT + 1, TILE_COUNT + 1),
            name="restored_right_ot",
        )[:TILE_COUNT, :TILE_COUNT],
        _numpy(
            restored_output.down_log_assignment,
            shape=(TILE_COUNT + 1, TILE_COUNT + 1),
            name="restored_down_ot",
        )[:TILE_COUNT, :TILE_COUNT],
    )
    raw_z = tuple(_standardise_matrix(matrix) for matrix in raw_matrices)
    restored_z = tuple(_standardise_matrix(matrix) for matrix in restored_matrices)
    raw_ot_z = tuple(_standardise_matrix(matrix) for matrix in raw_ot)
    restored_ot_z = tuple(_standardise_matrix(matrix) for matrix in restored_ot)
    raw_rank = tuple(_ranks(matrix) for matrix in raw_matrices)
    restored_rank = tuple(_ranks(matrix) for matrix in restored_matrices)
    relation = _relation_contact_evidence(
        relation_candidates,
        fusion_scores,
        fusion_confidence,
    )
    internal = _component_internal_edges(components)
    relation_targets: dict[tuple[int, int], set[int]] = defaultdict(set)
    internal_targets: dict[tuple[int, int], set[int]] = defaultdict(set)
    for direction, source, target in relation:
        relation_targets[(direction, source)].add(target)
    for direction, source, target in internal:
        internal_targets[(direction, source)].add(target)

    sources: list[int] = []
    targets: list[int] = []
    directions: list[int] = []
    features: list[list[float]] = []
    priorities: list[float] = []
    for direction in range(2):
        raw_order, raw_row_rank, raw_column_rank = raw_rank[direction]
        restored_order, restored_row_rank, restored_column_rank = restored_rank[direction]
        for source in range(TILE_COUNT):
            pool = set(int(value) for value in raw_order[source, :topk])
            pool.update(int(value) for value in restored_order[source, :topk])
            pool.update(relation_targets.get((direction, source), ()))
            pool.update(internal_targets.get((direction, source), ()))
            pool.discard(source)
            ranked: list[tuple[float, int, list[float]]] = []
            raw_top = set(int(value) for value in raw_order[source, :topk])
            restored_top = set(int(value) for value in restored_order[source, :topk])
            for target in pool:
                relation_z, query_confidence, relation_top = relation.get(
                    (direction, source, target),
                    (-4.0, -1.0, 0.0),
                )
                is_internal = float((direction, source, target) in internal)
                priority = (
                    max(raw_z[direction][source, target], restored_z[direction][source, target])
                    + 0.35 * relation_z
                    + 0.75 * relation_top
                    + 0.35 * query_confidence
                    + 1.5 * is_internal
                )
                row = [
                    priority,
                    raw_z[direction][source, target],
                    raw_ot_z[direction][source, target],
                    restored_z[direction][source, target],
                    restored_ot_z[direction][source, target],
                    1.0 / (1.0 + int(raw_row_rank[source, target])),
                    1.0 / (1.0 + int(raw_column_rank[source, target])),
                    1.0 / (1.0 + int(restored_row_rank[source, target])),
                    1.0 / (1.0 + int(restored_column_rank[source, target])),
                    relation_z,
                    query_confidence,
                    relation_top,
                    is_internal,
                    float(target in raw_top),
                    float(target in restored_top),
                    float(direction == 1),
                ]
                ranked.append((priority, target, row))
            ranked.sort(key=lambda item: (-item[0], item[1]))
            for priority, target, row in ranked[:topk]:
                sources.append(source)
                targets.append(target)
                directions.append(direction)
                features.append(row)
                priorities.append(priority)
    feature_array = np.asarray(features, dtype=np.float32)
    if feature_array.shape != (2 * TILE_COUNT * topk, EDGE_FEATURE_DIMENSION):
        raise RuntimeError(f"sparse edge contract changed: {feature_array.shape}")
    feature_array[:, 0] = (
        np.asarray(priorities) - float(np.mean(priorities))
    ) / max(float(np.std(priorities)), 1e-6)
    return (
        np.asarray(sources, dtype=np.int64),
        np.asarray(targets, dtype=np.int64),
        np.asarray(directions, dtype=np.int64),
        np.ascontiguousarray(feature_array),
    )


def _border_features(output: Any) -> np.ndarray:
    columns = (
        _numpy(
            output.right_out_border_logits,
            shape=(TILE_COUNT,),
            name="right_border",
        ),
        _numpy(output.left_in_border_logits, shape=(TILE_COUNT,), name="left_border"),
        _numpy(
            output.bottom_out_border_logits,
            shape=(TILE_COUNT,),
            name="bottom_border",
        ),
        _numpy(output.top_in_border_logits, shape=(TILE_COUNT,), name="top_border"),
    )
    return _standardise_columns(np.column_stack(columns))


def _tile_features(
    dirty_tiles: np.ndarray,
    raw_tokens: torch.Tensor,
    restored_tokens: torch.Tensor,
    raw_output: Any,
    restored_output: Any,
) -> np.ndarray:
    raw = _numpy(raw_tokens, shape=(TILE_COUNT, 64), name="raw_tokens")
    restored = _numpy(restored_tokens, shape=(TILE_COUNT, 64), name="restored_tokens")
    pixels = dirty_tiles.astype(np.float32) / 255.0
    moments = np.concatenate(
        (pixels.mean(axis=(1, 2)), pixels.std(axis=(1, 2))),
        axis=1,
    )
    result = np.concatenate(
        (
            _standardise_columns(raw),
            _standardise_columns(restored),
            _border_features(raw_output),
            _border_features(restored_output),
            _standardise_columns(moments),
        ),
        axis=1,
    )
    if result.shape != (TILE_COUNT, TILE_FEATURE_DIMENSION):
        raise RuntimeError(f"tile feature contract changed: {result.shape}")
    return np.ascontiguousarray(result, dtype=np.float32)


@torch.inference_mode()
def prepare_blind_evidence(
    case: PreparedCase,
    models: FrozenModels,
    *,
    device: torch.device,
    inference_batch: int,
    topk: int,
) -> BlindEvidence:
    """Prepare inference evidence without reading ``case.input_tile_to_position``."""

    started = perf_counter()
    board = prepare_fusion_board(
        case,
        socket=models.socket,
        relation=models.relation,
        denoiser=models.denoiser,
        device=device,
        inference_batch=inference_batch,
        raw_topk=32,
        raw_cap=64,
        union_cap=128,
        attach_exact_labels=False,
    )
    if board.union_labels or board.oracle_relations or board.profiles:
        raise RuntimeError("blind fusion preparation attached exact target state")
    fusion_input = torch.from_numpy(board.features).to(device)
    relation_scores = torch.from_numpy(board.frozen_relation_scores).to(device)
    fusion = models.fusion(fusion_input, relation_scores)
    fusion_scores = fusion.scores.float().cpu().numpy()
    fusion_confidence = fusion.confidence_logits.float().cpu().numpy()

    dirty_tensor = _tile_tensor(case.dirty_tiles, device=device)
    raw_tokens, raw_output = extract_frozen_socket_context(
        models.socket.model,
        dirty_tensor,
        grid=GRID,
    )
    restored_tiles = restore_matcher_view(
        models.denoiser,
        case.dirty_tiles,
        device=device,
        batch_size=inference_batch,
    )
    restored_tensor = _tile_tensor(restored_tiles, device=device)
    restored_tokens, restored_output = extract_frozen_socket_context(
        models.socket.model,
        restored_tensor,
        grid=GRID,
    )
    baseline = decode_socket_assignments(
        raw_output.right_log_assignment,
        raw_output.down_log_assignment,
        grid=GRID,
        config=SocketDecoderConfig(
            component_edge_budget_per_axis=144,
            swap_edge_budget_per_axis=144,
            max_swap_steps=24,
        ),
    )
    cyclic = select_global_cyclic_translation(
        baseline.layout,
        raw_output.right_log_assignment,
        raw_output.down_log_assignment,
        grid=GRID,
        config=CyclicTranslationConfig(border_weight=5.0),
    )
    edge_sources, edge_targets, edge_directions, edge_features = build_sparse_tile_graph(
        raw_output,
        restored_output,
        board.components,
        board.union_candidates,
        fusion_scores,
        fusion_confidence,
        topk=topk,
    )
    right_assignment = _numpy(
        raw_output.right_log_assignment,
        shape=(TILE_COUNT + 1, TILE_COUNT + 1),
        name="right_assignment",
    ).astype(np.float32)
    down_assignment = _numpy(
        raw_output.down_log_assignment,
        shape=(TILE_COUNT + 1, TILE_COUNT + 1),
        name="down_assignment",
    ).astype(np.float32)
    return BlindEvidence(
        case_id=case.case_id,
        source_filename=case.source_filename,
        dirty_tiles_sha256=hashlib.sha256(
            np.ascontiguousarray(case.dirty_tiles).tobytes()
        ).hexdigest(),
        tile_features=_tile_features(
            case.dirty_tiles,
            raw_tokens[0],
            restored_tokens[0],
            raw_output,
            restored_output,
        ),
        edge_features=edge_features,
        edge_sources=edge_sources,
        edge_targets=edge_targets,
        edge_directions=edge_directions,
        baseline_layout=np.ascontiguousarray(baseline.layout, dtype=np.int32),
        baseline_cyclic_layout=np.ascontiguousarray(cyclic.layout, dtype=np.int32),
        right_log_assignment=np.ascontiguousarray(right_assignment),
        down_log_assignment=np.ascontiguousarray(down_assignment),
        runtime_seconds={
            **board.runtime_seconds,
            "total_blind_evidence": perf_counter() - started,
        },
    )


def _model_from_config(config: Mapping[str, Any], *, device: torch.device) -> SparseBorderGraphQAP:
    architecture = config["architecture"]
    return SparseBorderGraphQAP(
        TILE_FEATURE_DIMENSION,
        EDGE_FEATURE_DIMENSION,
        hidden_dimension=int(architecture["hidden_dimension"]),
        edge_hidden_dimension=int(architecture["edge_hidden_dimension"]),
        max_grid=GRID,
        unrolled_steps=int(architecture["unrolled_qap_steps"]),
        sinkhorn_iterations=int(architecture["sinkhorn_iterations"]),
        sinkhorn_temperature=float(architecture["sinkhorn_temperature"]),
        baseline_anchor=float(architecture["baseline_decoder_anchor"]),
        pairwise_scale=float(architecture["pairwise_scale"]),
        message_normalizer=float(architecture["message_normalizer"]),
        edge_residual_limit=float(architecture["edge_residual_limit"]),
    ).to(device)


def _tensor_example(
    evidence: BlindEvidence,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    return (
        torch.from_numpy(evidence.tile_features).to(device),
        torch.from_numpy(evidence.edge_features).to(device),
        torch.from_numpy(evidence.edge_sources).to(device),
        torch.from_numpy(evidence.edge_targets).to(device),
        torch.from_numpy(evidence.edge_directions).to(device),
    )


def train_model(
    model: SparseBorderGraphQAP,
    examples: Sequence[TrainingExample],
    config: Mapping[str, Any],
    *,
    device: torch.device,
    log_every: int,
) -> tuple[list[dict[str, float]], float]:
    training = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    steps = int(training["updates"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=steps,
        eta_min=float(training["learning_rate"]) * 0.08,
    )
    generator = np.random.default_rng(int(config["selection"]["seed"]) + 71)
    history: list[dict[str, float]] = []
    started = perf_counter()
    model.train()
    for step in range(steps):
        example = examples[int(generator.integers(len(examples)))]
        tile_features, edge_features, sources, targets, directions = _tensor_example(
            example.evidence,
            device=device,
        )
        output = model(
            tile_features,
            edge_features,
            sources,
            targets,
            directions,
            example.evidence.baseline_layout,
            grid=GRID,
        )
        loss, diagnostics = qap_training_loss(
            output,
            example.reference_layout,
            example.evidence.baseline_layout,
            sources,
            targets,
            directions,
            grid=GRID,
            edge_loss_weight=float(training["edge_loss_weight"]),
            axis_loss_weight=float(training["axis_loss_weight"]),
            energy_margin_weight=float(training["energy_margin_weight"]),
            energy_margin=float(training["energy_margin_per_tile"]),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(training["gradient_clip"]),
            )
        )
        optimizer.step()
        scheduler.step()
        row = {
            "step": float(step + 1),
            **diagnostics,
            "gradient_norm": gradient_norm,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": perf_counter() - started,
        }
        history.append(row)
        if step == 0 or (step + 1) % log_every == 0 or step + 1 == steps:
            recent = history[-min(log_every, len(history)) :]
            print(
                json.dumps(
                    {
                        "event": "train",
                        "step": step + 1,
                        "loss": float(np.mean([item["loss"] for item in recent])),
                        "assignment_nll": float(
                            np.mean([item["assignment_nll"] for item in recent])
                        ),
                        "edge_bce": float(
                            np.mean([item["edge_bce"] for item in recent])
                        ),
                        "elapsed_seconds": perf_counter() - started,
                    }
                ),
                flush=True,
            )
    return history, perf_counter() - started


@torch.inference_mode()
def freeze_predictions(
    model: SparseBorderGraphQAP,
    examples: Sequence[BlindEvidence],
    *,
    device: torch.device,
) -> list[FrozenPrediction]:
    model.eval()
    predictions: list[FrozenPrediction] = []
    for index, evidence in enumerate(examples, start=1):
        tile_features, edge_features, sources, targets, directions = _tensor_example(
            evidence,
            device=device,
        )
        started = perf_counter()
        output = model(
            tile_features,
            edge_features,
            sources,
            targets,
            directions,
            evidence.baseline_layout,
            grid=GRID,
        )
        candidate = decode_hungarian(output.final_logits)
        candidate_cyclic = select_global_cyclic_translation(
            candidate,
            evidence.right_log_assignment,
            evidence.down_log_assignment,
            grid=GRID,
            config=CyclicTranslationConfig(border_weight=5.0),
        ).layout
        predictions.append(
            FrozenPrediction(
                evidence=evidence,
                candidate_layout=candidate,
                candidate_cyclic_layout=np.ascontiguousarray(
                    candidate_cyclic,
                    dtype=np.int32,
                ),
                candidate_energy=total_layout_energy(
                    output,
                    candidate,
                    sources,
                    targets,
                    directions,
                    grid=GRID,
                    pairwise_scale=model.pairwise_scale,
                ),
                baseline_energy=total_layout_energy(
                    output,
                    evidence.baseline_layout,
                    sources,
                    targets,
                    directions,
                    grid=GRID,
                    pairwise_scale=model.pairwise_scale,
                ),
                runtime_seconds=perf_counter() - started,
            )
        )
        print(
            json.dumps(
                {
                    "event": "prediction-frozen-in-memory",
                    "done": index,
                    "total": len(examples),
                    "case_id": evidence.case_id,
                }
            ),
            flush=True,
        )
    return predictions


def write_frozen_predictions(
    predictions: Sequence[FrozenPrediction],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    arrays: dict[str, np.ndarray] = {}
    cases: list[dict[str, Any]] = []
    for index, prediction in enumerate(predictions):
        prefix = f"case_{index:04d}"
        arrays[f"{prefix}__qap"] = prediction.candidate_layout
        arrays[f"{prefix}__qap_cyclic5"] = prediction.candidate_cyclic_layout
        arrays[f"{prefix}__decoder144"] = prediction.evidence.baseline_layout
        arrays[f"{prefix}__decoder144_cyclic5"] = (
            prediction.evidence.baseline_cyclic_layout
        )
        cases.append(
            {
                "array_prefix": prefix,
                "case_id": prediction.evidence.case_id,
                "source_filename": prediction.evidence.source_filename,
                "dirty_tiles_sha256": prediction.evidence.dirty_tiles_sha256,
                "candidate_energy": prediction.candidate_energy,
                "baseline_energy": prediction.baseline_energy,
                "prediction_runtime_seconds": prediction.runtime_seconds,
                "strict_permutations": True,
            }
        )
    arrays_path = output_dir / "frozen_predictions.npz"
    np.savez_compressed(arrays_path, **arrays)
    metadata_path = output_dir / "frozen_predictions.json"
    metadata_path.write_text(
        json.dumps(
            {
                "schema": "aiijc-sparse-bordergraph-qap-frozen-predictions-v1",
                "contains_exact_references": False,
                "contains_clean_pixels": False,
                "original_upright_identities_only": True,
                "cases": cases,
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "arrays_path": str(arrays_path),
        "arrays_sha256": sha256_file(arrays_path),
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
    }


def _numeric_mean(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    keys = [
        key
        for key, value in rows[0].items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    return {key: float(np.mean([float(row[key]) for row in rows])) for key in keys}


@torch.inference_mode()
def score_predictions(
    model: SparseBorderGraphQAP,
    predictions: Sequence[FrozenPrediction],
    references: Mapping[str, np.ndarray],
    *,
    device: torch.device,
) -> dict[str, Any]:
    boards: list[dict[str, Any]] = []
    for prediction in predictions:
        evidence = prediction.evidence
        reference = references[evidence.case_id]
        tile_features, edge_features, sources, targets, directions = _tensor_example(
            evidence,
            device=device,
        )
        output = model(
            tile_features,
            edge_features,
            sources,
            targets,
            directions,
            evidence.baseline_layout,
            grid=GRID,
        )
        truth_energy = total_layout_energy(
            output,
            reference,
            sources,
            targets,
            directions,
            grid=GRID,
            pairwise_scale=model.pairwise_scale,
        )
        baseline_energy = total_layout_energy(
            output,
            evidence.baseline_layout,
            sources,
            targets,
            directions,
            grid=GRID,
            pairwise_scale=model.pairwise_scale,
        )
        boards.append(
            {
                "case_id": evidence.case_id,
                "source_filename": evidence.source_filename,
                "candidate": evaluate_layout(
                    prediction.candidate_cyclic_layout,
                    reference,
                    reference_is_exact=True,
                ).as_dict(),
                "baseline": evaluate_layout(
                    evidence.baseline_cyclic_layout,
                    reference,
                    reference_is_exact=True,
                ).as_dict(),
                "energy": {
                    "truth": truth_energy,
                    "frozen_decoder": baseline_energy,
                    "truth_minus_frozen_decoder": truth_energy - baseline_energy,
                },
            }
        )
    candidate_mean = _numeric_mean([row["candidate"] for row in boards])
    baseline_mean = _numeric_mean([row["baseline"] for row in boards])
    delta = {
        key: candidate_mean[key] - baseline_mean[key]
        for key in (
            "correct_tile_count",
            "direct_placement",
            "correct_row_count",
            "correct_column_count",
            "adjacency_correct",
            "adjacency",
        )
    }
    energy_deltas = [row["energy"]["truth_minus_frozen_decoder"] for row in boards]
    energy_pass = float(np.mean(energy_deltas)) > 0
    exact_pass = delta["correct_tile_count"] > 0
    adjacency_pass = delta["adjacency"] >= -0.01
    return {
        "reference": "exact inverse deterministic shuffle opened only after frozen artifact",
        "case_count": len(boards),
        "candidate_mean": candidate_mean,
        "baseline_mean": baseline_mean,
        "candidate_delta_vs_baseline": delta,
        "mean_truth_minus_frozen_decoder_total_energy": float(np.mean(energy_deltas)),
        "strict_permutation_count": len(boards),
        "discovery_gate": {
            "pass": energy_pass and (exact_pass or adjacency_pass),
            "truth_energy_better": energy_pass,
            "exact_positive": exact_pass,
            "adjacency_loss_at_most_1pp": adjacency_pass,
            "promotion_authorized": False,
        },
        "boards": boards,
    }


def _capacity_graph(
    grid: int,
    reference: np.ndarray,
    *,
    topk: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = grid * grid
    tile_to_slot = np.empty(count, dtype=np.int32)
    tile_to_slot[reference] = np.arange(count, dtype=np.int32)
    rng = np.random.default_rng(seed)
    sources: list[int] = []
    targets: list[int] = []
    directions: list[int] = []
    features: list[list[float]] = []
    for direction in range(2):
        delta = 1 if direction == 0 else grid
        for source in range(count):
            slot = int(tile_to_slot[source])
            valid = (direction == 0 and slot % grid < grid - 1) or (
                direction == 1 and slot < count - grid
            )
            true_target = int(reference[slot + delta]) if valid else None
            candidates = [value for value in range(count) if value != source]
            rng.shuffle(candidates)
            if true_target is not None:
                candidates = [true_target] + [value for value in candidates if value != true_target]
            for rank, target in enumerate(candidates[:topk]):
                correct = float(target == true_target)
                row = np.zeros(EDGE_FEATURE_DIMENSION, dtype=np.float32)
                row[0] = 3.0 if correct else -1.0 - 0.05 * rank
                row[1] = correct
                row[-1] = direction
                sources.append(source)
                targets.append(target)
                directions.append(direction)
                features.append(row.tolist())
    return (
        np.asarray(sources, dtype=np.int64),
        np.asarray(targets, dtype=np.int64),
        np.asarray(directions, dtype=np.int64),
        np.asarray(features, dtype=np.float32),
    )


def run_capacity(
    config: Mapping[str, Any],
    *,
    device: torch.device,
    output_dir: Path,
    log_every: int,
) -> dict[str, Any]:
    grid = int(config["training"]["capacity_grid"])
    count = grid * grid
    seed = int(config["selection"]["seed"])
    rng = np.random.default_rng(seed)
    reference = rng.permutation(count).astype(np.int32)
    baseline = np.roll(reference.reshape(grid, grid), 1, axis=1).reshape(-1).copy()
    topk = min(int(config["sparse_graph"]["topk_per_tile_direction"]), count - 1)
    sources, targets, directions, edge_features = _capacity_graph(
        grid,
        reference,
        topk=topk,
        seed=seed + 1,
    )
    tile_to_slot = np.empty(count, dtype=np.int32)
    tile_to_slot[reference] = np.arange(count)
    tile_features = np.zeros((count, TILE_FEATURE_DIMENSION), dtype=np.float32)
    rows, columns = divmod(tile_to_slot, grid)
    tile_features[:, 0] = rows / (grid - 1)
    tile_features[:, 1] = columns / (grid - 1)
    tile_features[:, 2 + rows] = 1.0
    tile_features[:, 2 + grid + columns] = 1.0
    model = _model_from_config(config, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    started = perf_counter()
    history: list[dict[str, float]] = []
    for step in range(160):
        output = model(
            torch.from_numpy(tile_features).to(device),
            torch.from_numpy(edge_features).to(device),
            torch.from_numpy(sources).to(device),
            torch.from_numpy(targets).to(device),
            torch.from_numpy(directions).to(device),
            baseline,
            grid=grid,
        )
        loss, diagnostics = qap_training_loss(
            output,
            reference,
            baseline,
            torch.from_numpy(sources).to(device),
            torch.from_numpy(targets).to(device),
            torch.from_numpy(directions).to(device),
            grid=grid,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append({"step": float(step + 1), **diagnostics})
        if step == 0 or (step + 1) % log_every == 0:
            print(
                json.dumps({"event": "capacity-train", "step": step + 1, **diagnostics}),
                flush=True,
            )
    with torch.inference_mode():
        output = model(
            torch.from_numpy(tile_features).to(device),
            torch.from_numpy(edge_features).to(device),
            torch.from_numpy(sources).to(device),
            torch.from_numpy(targets).to(device),
            torch.from_numpy(directions).to(device),
            baseline,
            grid=grid,
        )
        layout = decode_hungarian(output.final_logits)
    correct = int(np.sum(layout == reference))
    report = {
        "mode": "capacity",
        "grid": grid,
        "same_case_mechanical_only": True,
        "correct_tiles": correct,
        "tile_count": count,
        "strict_permutation": bool(np.array_equal(np.sort(layout), np.arange(count))),
        "pass": correct == count,
        "runtime_seconds": perf_counter() - started,
        "last_training": history[-1],
    }
    torch.save(
        {"state_dict": model.state_dict(), "capacity_only": True},
        output_dir / "capacity_model.pt",
    )
    return report


def run_benchmark(
    config: Mapping[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    seed = int(config["selection"]["seed"])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    topk = int(config["sparse_graph"]["topk_per_tile_direction"])
    edge_count = 2 * TILE_COUNT * topk
    sources = torch.arange(TILE_COUNT).repeat_interleave(2 * topk)
    directions = torch.arange(2).repeat_interleave(topk).repeat(TILE_COUNT)
    targets = torch.randint(TILE_COUNT, (edge_count,), generator=generator)
    targets[targets == sources] = (targets[targets == sources] + 1) % TILE_COUNT
    tile_features = torch.randn(
        TILE_COUNT,
        TILE_FEATURE_DIMENSION,
        generator=generator,
    )
    edge_features = torch.randn(
        edge_count,
        EDGE_FEATURE_DIMENSION,
        generator=generator,
    )
    model = _model_from_config(config, device=device)
    tensors = tuple(
        value.to(device)
        for value in (tile_features, edge_features, sources, targets, directions)
    )
    layout = np.arange(TILE_COUNT, dtype=np.int32)
    times: list[float] = []
    for _ in range(4):
        started = perf_counter()
        output = model(*tensors, layout, grid=GRID)
        loss = output.probabilities[-1].square().mean()
        model.zero_grad(set_to_none=True)
        loss.backward()
        if device.type == "mps":
            torch.mps.synchronize()
        times.append(perf_counter() - started)
    return {
        "mode": "benchmark",
        "device": str(device),
        "grid": GRID,
        "edge_count": edge_count,
        "unrolled_steps": model.unrolled_steps,
        "forward_backward_seconds": times,
        "median_after_warmup_seconds": float(np.median(times[1:])),
        "strict_hungarian": bool(
            np.array_equal(np.sort(decode_hungarian(output.final_logits)), np.arange(TILE_COUNT))
        ),
    }


def main() -> None:
    args = parse_args()
    if args.log_every <= 0 or args.inference_batch <= 0:
        raise ValueError("logging and inference batch values must be positive")
    config = _load_json(args.config)
    _validate_config(config)
    config_sha256 = sha256_file(args.config)
    random.seed(int(config["selection"]["seed"]))
    np.random.seed(int(config["selection"]["seed"]))
    torch.manual_seed(int(config["selection"]["seed"]))
    device = _choose_device(
        args.device,
        allow_nondeterministic_mps=args.allow_nondeterministic_mps,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    if args.mode == "capacity":
        result = run_capacity(
            config,
            device=device,
            output_dir=output_dir,
            log_every=args.log_every,
        )
        report = {
            "experiment": "sparse-bordergraph-qap-v1",
            "config": {"path": str(args.config.resolve()), "sha256": config_sha256},
            "result": result,
            "competition_test_opened": False,
        }
    elif args.mode == "benchmark":
        report = {
            "experiment": "sparse-bordergraph-qap-v1",
            "config": {"path": str(args.config.resolve()), "sha256": config_sha256},
            "result": run_benchmark(config, device=device),
            "competition_test_opened": False,
        }
    else:
        manifest = _load_json(args.manifest)
        if manifest.get("protocol_digest") != compute_protocol_digest(manifest):
            raise ValueError("validation manifest protocol digest is invalid")
        models = _load_frozen_models(config, device=device)
        fit_records, eval_records, selection = _commit_selection(
            config,
            manifest,
            models,
            output_dir=output_dir,
        )
        cache = CleanTileCache(args.targets)
        topk = int(config["sparse_graph"]["topk_per_tile_direction"])
        fit_examples: list[TrainingExample] = []
        for index, record in enumerate(fit_records, start=1):
            case = prepare_case(
                cache,
                record,
                draw_index=0,
                seed=int(config["selection"]["seed"]),
            )
            evidence = prepare_blind_evidence(
                case,
                models,
                device=device,
                inference_batch=args.inference_batch,
                topk=topk,
            )
            reference = np.empty(TILE_COUNT, dtype=np.int32)
            reference[case.input_tile_to_position] = np.arange(TILE_COUNT, dtype=np.int32)
            fit_examples.append(TrainingExample(evidence, reference))
            print(
                json.dumps(
                    {"event": "fit-evidence", "done": index, "total": len(fit_records)}
                ),
                flush=True,
            )
        model = _model_from_config(config, device=device)
        history, training_seconds = train_model(
            model,
            fit_examples,
            config,
            device=device,
            log_every=args.log_every,
        )
        checkpoint_path = output_dir / "sparse_bordergraph_qap.pt"
        torch.save(
            {
                "schema": "aiijc-sparse-bordergraph-qap-checkpoint-v1",
                "state_dict": model.state_dict(),
                "config_sha256": config_sha256,
                "selection": selection,
                "frozen_inputs": models.metadata,
                "contract": {
                    "strict_original_upright_permutation": True,
                    "dense_four_index_affinity": False,
                    "edge_topk": topk,
                    "unrolled_steps": model.unrolled_steps,
                    "input_tile_identity_embedding": False,
                },
            },
            checkpoint_path,
        )

        # Eval sources are opened only now, after config/roster/model are frozen.
        blind_eval: list[BlindEvidence] = []
        eval_references: dict[str, np.ndarray] = {}
        for index, record in enumerate(eval_records, start=1):
            case = prepare_case(
                cache,
                record,
                draw_index=0,
                seed=int(config["selection"]["seed"]) + 100_000,
            )
            blind_eval.append(
                prepare_blind_evidence(
                    case,
                    models,
                    device=device,
                    inference_batch=args.inference_batch,
                    topk=topk,
                )
            )
            reference = np.empty(TILE_COUNT, dtype=np.int32)
            reference[case.input_tile_to_position] = np.arange(TILE_COUNT, dtype=np.int32)
            eval_references[case.case_id] = reference
            print(
                json.dumps(
                    {"event": "eval-blind-evidence", "done": index, "total": len(eval_records)}
                ),
                flush=True,
            )
        predictions = freeze_predictions(model, blind_eval, device=device)
        frozen_artifact = write_frozen_predictions(predictions, output_dir=output_dir)
        print(json.dumps({"event": "exact16-open-after-freeze", **frozen_artifact}), flush=True)
        evaluation = score_predictions(
            model,
            predictions,
            eval_references,
            device=device,
        )
        report = {
            "experiment": "sparse-bordergraph-qap-v1",
            "status": (
                "bounded-discovery-pass-no-promotion"
                if evaluation["discovery_gate"]["pass"]
                else "bounded-discovery-fail-stop"
            ),
            "config": {"path": str(args.config.resolve()), "sha256": config_sha256},
            "selection": selection,
            "frozen_inputs": models.metadata,
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": sha256_file(checkpoint_path),
            },
            "training": {
                "device": str(device),
                "seconds": training_seconds,
                "history": history,
            },
            "frozen_prediction_artifact": frozen_artifact,
            "evaluation": evaluation,
            "legal": {
                "strict_original_upright_permutations": True,
                "restored_pixels_matcher_only": True,
                "competition_test_opened": False,
                "calibration_opened": False,
                "holdout_opened": False,
            },
        }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"event": "complete", "report": str(report_path), "sha256": sha256_file(report_path)}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

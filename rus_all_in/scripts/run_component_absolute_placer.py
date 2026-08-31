#!/usr/bin/env python3
"""Bounded train-only run for one-anchor independent component placement."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from aiijc_puzzle.component_absolute_placer import (
    ComponentAbsoluteConfig,
    ComponentAbsolutePlacerModel,
    align_components_across_corruptions,
    anchor_confidence,
    average_precision,
    component_absolute_loss,
    component_absolute_targets,
    component_geometry_features,
    paired_component_consistency_loss,
    place_one_component_anchor,
    render_native_component_mosaic,
)
from aiijc_puzzle.component_anchor_diagnostic import rebuild_decoder_components
from aiijc_puzzle.layout_evaluation import evaluate_layout
from aiijc_puzzle.protocol import compute_protocol_digest, sha256_file
from aiijc_puzzle.socket_decoder import SocketDecoderConfig, decode_socket_assignments
from aiijc_puzzle.socket_sorter_production import load_socket_checkpoint
from aiijc_puzzle.socket_translation_placer import (
    CyclicTranslationConfig,
    select_global_cyclic_translation,
)
from aiijc_puzzle.synthetic_socket_evaluation import (
    names_digest,
    select_source_disjoint_train_records,
)

try:
    from scripts.run_component_relation_reranker import (
        GRID,
        CleanTileCache,
        _tile_tensor,
        prepare_case,
    )
    from scripts.run_sparse_bordergraph_qap import _panel_json_exclusion
    from scripts.run_whole_layout_cyclic_origin import collect_declared_filenames
except ModuleNotFoundError:  # Direct ``python scripts/run_*.py`` execution.
    from run_component_relation_reranker import (
        GRID,
        CleanTileCache,
        _tile_tensor,
        prepare_case,
    )
    from run_sparse_bordergraph_qap import _panel_json_exclusion
    from run_whole_layout_cyclic_origin import collect_declared_filenames

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/component_absolute_placer_preregistered_v1.json"
DEFAULT_MANIFEST = PROJECT_ROOT / "data/interim/validation_manifest.json"
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
DEFAULT_SELECTION = PROJECT_ROOT / "outputs/component-absolute-placer/v1-selection"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/component-absolute-placer/v1-fit256-s600-eval32"
SOCKET_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/socket-matcher/v2-d64-train1024-s1600-r400-dev32/socket_matcher.pt"
)
SOCKET_SHA256 = "0e9df49a503c65aac7f1468e9acd6a074a5e658ae8b61f8954be086272c49670"
FIT_SOURCES = 256
TRAIN_SOURCES = 224
CALIBRATION_SOURCES = 32
EVAL_SOURCES = 32
MAX_STEPS = 600
SEED = 20320920
TILE_COUNT = GRID * GRID
ACTIVE_REGISTRY = (
    "configs/direct_hard_edge_fresh64_confirmation_v1.json",
    "outputs/direct-hard-edge-priority/frozen-v1-fresh64-draw0/report.json",
    "outputs/fullres-twin-side-matcher/v1-fit256-s400-eval24/selection-commitment.json",
    "outputs/frame-side-origin/v1-selection/selection_commitment.json",
    "outputs/whole-layout-cyclic-origin/v1-selection/selection_commitment.json",
    "outputs/sparse-bordergraph-qap/pilot-fit64-s240-eval16-top8-mps/selection_commitment.json",
    "configs/direct_hard_edge_board_priority_preregistered_v1.json",
    "configs/fullres_relation_fusion_decoder_d2_preregistered_v1.json",
    "configs/component_relation_cyclic_fresh_gate_v1.json",
)


@dataclass(frozen=True)
class ComponentBoard:
    case_id: str
    source_filename: str
    raw_tiles: np.ndarray
    components: tuple[dict[int, tuple[int, int]], ...]
    nontrivial_indices: tuple[int, ...]
    geometry: np.ndarray
    baseline_layout: np.ndarray
    tile_to_position: np.ndarray | None
    dirty_tiles_sha256: str


@dataclass(frozen=True)
class FrozenEval:
    case_id: str
    source_filename: str
    candidate_layout: np.ndarray
    comparator_layout: np.ndarray
    anchor_selected: bool
    selected_nontrivial_index: int | None
    selected_all_component_index: int | None
    selected_score: float | None
    selected_offset: int | None
    component_scores: np.ndarray
    component_offsets: np.ndarray
    baseline_purity_scores: np.ndarray
    component_sizes: np.ndarray
    nontrivial_components: tuple[dict[int, tuple[int, int]], ...]
    packing: dict[str, Any] | None
    dirty_tiles_sha256: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("capacity", "selection", "pilot"), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--allow-nondeterministic-mps", action="store_true")
    parser.add_argument("--wait-for-eval-confirmation", action="store_true")
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args()


def _device(name: str, acknowledgement: bool) -> torch.device:
    if name == "mps":
        if not torch.backends.mps.is_available() or not acknowledgement:
            raise ValueError("MPS requires availability and explicit acknowledgement")
        torch.use_deterministic_algorithms(False)
    elif acknowledgement:
        raise ValueError("MPS acknowledgement supplied for CPU")
    else:
        torch.use_deterministic_algorithms(True)
    return torch.device(name)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _manifest_lookup(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if manifest.get("protocol_digest") != compute_protocol_digest(dict(manifest)):
        raise ValueError("manifest protocol digest is invalid")
    rows = manifest.get("splits", {}).get("train")
    if not isinstance(rows, list):
        raise ValueError("manifest train split is absent")
    return {str(row["filename"]): row for row in rows}


def freeze_selection(args: argparse.Namespace, device: torch.device) -> None:
    output = (args.output_dir or DEFAULT_SELECTION).resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "selection_commitment.json"
    if path.exists():
        raise FileExistsError("refusing to overwrite component placer selection")
    manifest = _load_json(args.manifest)
    _manifest_lookup(manifest)
    excluded, panel_audit = _panel_json_exclusion()
    registry: list[dict[str, Any]] = []
    for raw_path in ACTIVE_REGISTRY:
        source = PROJECT_ROOT / raw_path
        if not source.exists():
            raise FileNotFoundError(f"required exclusion registry entry is absent: {source}")
        payload = _load_json(source)
        names = collect_declared_filenames(payload)
        excluded.update(names)
        registry.append(
            {
                "path": raw_path,
                "sha256": sha256_file(source),
                "declared_filename_count": len(names),
                "declared_filename_digest": names_digest(sorted(names), sort_names=True),
            }
        )
    if sha256_file(SOCKET_CHECKPOINT) != SOCKET_SHA256:
        raise ValueError("Socket checkpoint changed before selection")
    socket = load_socket_checkpoint(SOCKET_CHECKPOINT, device=device)
    excluded.update(socket.lineage.exposed_filenames)
    excluded_digest = names_digest(sorted(excluded), sort_names=True)
    namespace = f"aiijc-independent-component-anchor-v1:{excluded_digest}"
    records = select_source_disjoint_train_records(
        manifest,
        excluded_filenames=tuple(sorted(excluded)),
        limit=FIT_SOURCES + EVAL_SOURCES,
        seed=SEED,
        namespace=namespace,
    )
    fit = [str(row["filename"]) for row in records[:FIT_SOURCES]]
    evaluation = [str(row["filename"]) for row in records[FIT_SOURCES:]]
    if set(fit) & set(evaluation) or (set(fit) | set(evaluation)) & excluded:
        raise RuntimeError("component placer selection is not source-disjoint")
    commitment = {
        "schema": "aiijc-independent-component-anchor-selection-v1",
        "written_before_any_selected_target_access": True,
        "manifest_split": "train",
        "namespace": namespace,
        "seed": SEED,
        "exclusion": {
            "panel_json_audit": panel_audit,
            "required_registry": registry,
            "socket_lineage_count": socket.lineage.exposed_count,
            "socket_lineage_digest": socket.lineage.exposed_digest,
            "union_count": len(excluded),
            "union_digest": excluded_digest,
        },
        "fit_filenames": fit,
        "train_filenames": fit[:TRAIN_SOURCES],
        "calibration_filenames": fit[TRAIN_SOURCES:],
        "fit_order_digest": names_digest(fit),
        "train_order_digest": names_digest(fit[:TRAIN_SOURCES]),
        "calibration_order_digest": names_digest(fit[TRAIN_SOURCES:]),
        "evaluation_filenames": evaluation,
        "evaluation_order_digest": names_digest(evaluation),
        "evaluation_set_digest": names_digest(evaluation, sort_names=True),
        "fit_evaluation_overlap": 0,
        "excluded_overlap": 0,
        "holdout_opened": False,
        "competition_test_opened": False,
    }
    _write_json(path, commitment)
    print(
        json.dumps(
            {
                "event": "selection-frozen",
                "path": str(path),
                "sha256": sha256_file(path),
                "fit_digest": commitment["fit_order_digest"],
                "evaluation_digest": commitment["evaluation_order_digest"],
                "selected_target_access": False,
            }
        ),
        flush=True,
    )


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    sidecar = path.with_name(f"{path.name}.sha256")
    expected = sidecar.read_text().split()[0]
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError("component placer preregistration hash mismatch")
    config = _load_json(path)
    if config.get("experiment") != "independent-component-absolute-placer-v1":
        raise ValueError("unexpected component placer experiment")
    if not config.get("registered_before_selected_target_access"):
        raise ValueError("component placer timing contract is absent")
    if int(config["training"]["steps"]) > MAX_STEPS:
        raise ValueError("component placer training exceeds bounded cap")
    return config, observed


def _load_rosters(
    config: Mapping[str, Any], manifest: Mapping[str, Any]
) -> tuple[
    list[Mapping[str, Any]],
    list[Mapping[str, Any]],
    list[Mapping[str, Any]],
    dict[str, Any],
    str,
]:
    contract = config["selection_commitment"]
    path = PROJECT_ROOT / str(contract["path"])
    observed = sha256_file(path)
    if observed != contract["sha256"]:
        raise ValueError("component placer selection changed")
    selection = _load_json(path)
    if not selection["written_before_any_selected_target_access"]:
        raise ValueError("selection timing contract failed")
    fit = list(selection["fit_filenames"])
    train = list(selection["train_filenames"])
    calibration = list(selection["calibration_filenames"])
    evaluation = list(selection["evaluation_filenames"])
    if (len(fit), len(train), len(calibration), len(evaluation)) != (
        FIT_SOURCES,
        TRAIN_SOURCES,
        CALIBRATION_SOURCES,
        EVAL_SOURCES,
    ):
        raise ValueError("component placer roster sizes changed")
    for key, names in (
        ("fit_order_digest", fit),
        ("train_order_digest", train),
        ("calibration_order_digest", calibration),
        ("evaluation_order_digest", evaluation),
    ):
        if names_digest(names) != selection[key] or selection[key] != contract[key]:
            raise ValueError(f"component placer roster digest changed: {key}")
    lookup = _manifest_lookup(manifest)
    return (
        [lookup[name] for name in train],
        [lookup[name] for name in calibration],
        [lookup[name] for name in evaluation],
        selection,
        observed,
    )


def _load_socket(device: torch.device) -> Any:
    if sha256_file(SOCKET_CHECKPOINT) != SOCKET_SHA256:
        raise ValueError("Socket checkpoint SHA mismatch")
    return load_socket_checkpoint(SOCKET_CHECKPOINT, device=device)


@torch.inference_mode()
def prepare_board(
    cache: CleanTileCache,
    record: Mapping[str, Any],
    *,
    draw_index: int,
    seed: int,
    socket: Any,
    device: torch.device,
    include_truth: bool,
) -> ComponentBoard:
    case = prepare_case(cache, record, draw_index=draw_index, seed=seed)
    output = socket.model(_tile_tensor(case.dirty_tiles, device=device), grid=GRID)
    decoder = decode_socket_assignments(
        output.right_log_assignment,
        output.down_log_assignment,
        grid=GRID,
        config=SocketDecoderConfig(component_edge_budget_per_axis=144, max_swap_steps=24),
    )
    build = rebuild_decoder_components(
        output.right_log_assignment,
        output.down_log_assignment,
        grid=GRID,
        edge_budget_per_axis=144,
    )
    nontrivial = tuple(index for index, item in enumerate(build.components) if len(item) >= 2)
    if not nontrivial:
        raise RuntimeError("decoder produced no nontrivial component")
    selected = tuple(build.components[index] for index in nontrivial)
    geometry = component_geometry_features(
        selected,
        build.constraints,
        decoder.layout,
        grid=GRID,
    )
    return ComponentBoard(
        case_id=case.case_id,
        source_filename=case.source_filename,
        raw_tiles=np.ascontiguousarray(case.dirty_tiles),
        components=build.components,
        nontrivial_indices=nontrivial,
        geometry=geometry.astype(np.float16),
        baseline_layout=np.ascontiguousarray(decoder.layout, dtype=np.int16),
        tile_to_position=(
            np.ascontiguousarray(case.input_tile_to_position, dtype=np.int16)
            if include_truth
            else None
        ),
        dirty_tiles_sha256=hashlib.sha256(case.dirty_tiles.tobytes()).hexdigest(),
    )


def _forward_board(
    model: ComponentAbsolutePlacerModel,
    board: ComponentBoard,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw = (
        torch.from_numpy(board.raw_tiles)
        .permute(0, 3, 1, 2)
        .to(dtype=torch.float32)
        .div_(255.0)
    )
    components = tuple(board.components[index] for index in board.nontrivial_indices)
    mosaics = [
        render_native_component_mosaic(raw, component).to(device)
        for component in components
    ]
    geometry = torch.from_numpy(board.geometry.astype(np.float32)).to(device)
    return model(mosaics, geometry)


def _board_targets(board: ComponentBoard, device: torch.device) -> tuple[torch.Tensor, ...]:
    if board.tile_to_position is None:
        raise ValueError("component board has no training truth")
    components = tuple(board.components[index] for index in board.nontrivial_indices)
    purity, offsets, support = component_absolute_targets(
        components,
        board.tile_to_position,
        grid=GRID,
    )
    sizes = torch.tensor([len(item) for item in components], dtype=torch.float32)
    return tuple(item.to(device) for item in (purity, offsets, support, sizes))


def run_capacity(args: argparse.Namespace, device: torch.device) -> None:
    torch.manual_seed(SEED)
    config = ComponentAbsoluteConfig(
        grid=4,
        pixel_width=16,
        pixel_blocks=2,
        lattice_blocks=2,
        model_dimension=32,
        set_layers=1,
        set_heads=4,
    )
    model = ComponentAbsolutePlacerModel(config).to(device)
    raw = torch.zeros(16, 3, 20, 20)
    raw[0:4, 0] = torch.tensor([0.15, 0.35, 0.65, 0.85])[:, None, None]
    raw[4:8, 1] = torch.tensor([0.85, 0.65, 0.35, 0.15])[:, None, None]
    components = (
        {0: (0, 0), 1: (0, 1)},
        {2: (0, 0), 3: (0, 1)},
        {4: (0, 0), 5: (1, 0)},
        {6: (0, 0), 7: (1, 0)},
    )
    mosaics = [render_native_component_mosaic(raw, item).to(device) for item in components]
    geometry = torch.zeros(4, 12, device=device)
    geometry[:, 0] = 2 / 16
    geometry[:, 1] = np.log1p(2) / np.log1p(16)
    geometry[:, 2:4] = torch.tensor(
        [[0.25, 0.50], [0.25, 0.50], [0.50, 0.25], [0.50, 0.25]],
        device=device,
    )
    geometry[:, 4] = 1.0
    purity = torch.tensor([1, 0, 1, 0], dtype=torch.bool, device=device)
    offsets = torch.tensor([0, -1, 10, -1], device=device)
    sizes = torch.full((4,), 2.0, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    history: list[float] = []
    for _ in range(120):
        purity_logits, offset_logits = model(mosaics, geometry)
        loss, _ = component_absolute_loss(
            purity_logits,
            offset_logits,
            purity,
            offsets,
            component_sizes=sizes,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    with torch.inference_mode():
        purity_logits, offset_logits = model(mosaics, geometry)
    ap = average_precision(purity.cpu().numpy(), purity_logits.cpu().numpy())
    offset_accuracy = float(
        (offset_logits[purity].argmax(dim=-1) == offsets[purity]).float().mean()
    )
    result = {
        "experiment": "independent-component-anchor-capacity-v1",
        "device": str(device),
        "initial_loss": history[0],
        "final_loss": history[-1],
        "purity_average_precision": ap,
        "pure_offset_top1": offset_accuracy,
        "pass": ap == 1.0 and offset_accuracy == 1.0 and history[-1] < history[0] * 0.1,
    }
    output = (args.output_dir or PROJECT_ROOT / "outputs/component-absolute-placer/capacity-v1")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "report.json", result)
    print(json.dumps({"event": "capacity", **result}), flush=True)


def _purity_metrics(
    labels: Sequence[np.ndarray],
    model_scores: Sequence[np.ndarray],
    baseline_scores: Sequence[np.ndarray],
) -> dict[str, float]:
    truth = np.concatenate(labels).astype(bool)
    learned = np.concatenate(model_scores)
    baseline = np.concatenate(baseline_scores)
    prevalence = float(truth.mean())
    size_ap = average_precision(truth, -baseline[:, 0])
    confidence_ap = average_precision(truth, baseline[:, 1])
    baseline_ap = max(prevalence, size_ap, confidence_ap)
    model_ap = average_precision(truth, learned)
    return {
        "component_count": int(len(truth)),
        "pure_fraction": prevalence,
        "model_average_precision": model_ap,
        "random_prevalence_ap": prevalence,
        "inverse_size_ap": size_ap,
        "edge_confidence_ap": confidence_ap,
        "strongest_baseline_ap": baseline_ap,
        "ap_ratio": model_ap / baseline_ap if baseline_ap else 0.0,
    }


def _fit_threshold(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Choose one conservative fit-only anchor threshold by a frozen rule."""

    top = [row for row in rows if row["top_score"] is not None]
    candidates = sorted({float(row["top_score"]) for row in top}, reverse=True)
    best: tuple[float, float, float, int, dict[str, Any]] | None = None
    for threshold in candidates:
        selected = [row for row in top if float(row["top_score"]) >= threshold]
        if len(selected) < 4:
            continue
        correct = sum(bool(row["top_anchor_correct"]) for row in selected)
        precision = correct / len(selected)
        utility = sum(
            float(row["top_size"]) * (1.0 if row["top_anchor_correct"] else -1.0)
            for row in selected
        ) / CALIBRATION_SOURCES
        payload = {
            "threshold": threshold,
            "selected_boards": len(selected),
            "correct_anchors": correct,
            "anchor_precision": precision,
            "signed_size_utility_per_board": utility,
        }
        key = (utility, precision, threshold, -len(selected), payload)
        if best is None or key[:4] > best[:4]:
            best = key
    if best is None or best[0] <= 0 or best[1] < 0.25:
        return {
            "threshold": 1.000001,
            "selected_boards": 0,
            "correct_anchors": 0,
            "anchor_precision": 0.0,
            "signed_size_utility_per_board": 0.0,
            "fallback_only": True,
        }
    return best[4] | {"fallback_only": False}


def _calibrate(
    model: ComponentAbsolutePlacerModel,
    boards: Sequence[ComponentBoard],
    device: torch.device,
) -> dict[str, Any]:
    labels: list[np.ndarray] = []
    learned_purity: list[np.ndarray] = []
    baselines: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    offset_correct = 0
    pure_count = 0
    model.eval()
    with torch.inference_mode():
        for board in boards:
            purity_logits, offset_logits = _forward_board(model, board, device)
            purity, targets, _, sizes = _board_targets(board, device)
            scores, predicted_offsets = anchor_confidence(purity_logits, offset_logits)
            top_index = int(np.argmax(scores)) if len(scores) else None
            top_correct = bool(
                top_index is not None
                and purity[top_index]
                and int(predicted_offsets[top_index]) == int(targets[top_index])
            )
            rows.append(
                {
                    "source_filename": board.source_filename,
                    "case_id": board.case_id,
                    "top_score": float(scores[top_index]) if top_index is not None else None,
                    "top_size": int(sizes[top_index]) if top_index is not None else 0,
                    "top_anchor_correct": top_correct,
                }
            )
            labels.append(purity.cpu().numpy())
            learned_purity.append(torch.sigmoid(purity_logits).cpu().numpy())
            size = sizes.cpu().numpy()
            baselines.append(np.column_stack((size, board.geometry[:, 5])))
            if purity.any():
                offset_correct += int(
                    (offset_logits[purity].argmax(dim=-1) == targets[purity]).sum()
                )
                pure_count += int(purity.sum())
    return {
        "purity": _purity_metrics(labels, learned_purity, baselines),
        "pure_offset_top1": offset_correct / pure_count if pure_count else 0.0,
        "pure_component_count": pure_count,
        "threshold_selection": _fit_threshold(rows),
        "boards": rows,
    }


def _strict(layout: Any) -> np.ndarray:
    value = np.asarray(layout, dtype=np.int32)
    if value.shape != (TILE_COUNT,) or not np.array_equal(np.sort(value), np.arange(TILE_COUNT)):
        raise ValueError("layout is not a strict original-tile permutation")
    return np.ascontiguousarray(value)


def _layout_summary(rows: Sequence[Mapping[str, Any]], arm: str) -> dict[str, float]:
    return {
        field: float(np.mean([row[arm][field] for row in rows]))
        for field in (
            "correct_tile_count",
            "direct_placement",
            "translation_aligned_count",
            "adjacency_correct",
            "adjacency",
        )
    }


def _evaluate_gate(summary: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    gate = config["discovery_gate"]
    exact_delta = float(summary["candidate"]["correct_tile_count"]) - float(
        summary["comparator"]["correct_tile_count"]
    )
    adjacency_delta = float(summary["candidate"]["adjacency"]) - float(
        summary["comparator"]["adjacency"]
    )
    ap_ratio = float(summary["purity"]["ap_ratio"])
    checks = {
        "purity_ap_ratio": ap_ratio + 1e-12 >= float(gate["minimum_purity_ap_ratio"]),
        "exact": exact_delta + 1e-12 >= float(gate["minimum_exact_gain_per_board"]),
        "adjacency": adjacency_delta + 1e-12
        >= -float(gate["maximum_adjacency_loss_fraction"]),
        "strict": int(summary["strict_permutation_count"])
        == int(gate["strict_original_permutations_required"]),
    }
    passed = all(checks.values())
    return {
        "status": "pass-await-root-review" if passed else "fail-stop",
        "pass": passed,
        "checks": checks,
        "observed": {
            "purity_ap_ratio": ap_ratio,
            "exact_gain_per_board": exact_delta,
            "adjacency_delta": adjacency_delta,
        },
        "promotion_authorized": False,
        "competition_test_authorized": False,
    }


def run_pilot(args: argparse.Namespace, device: torch.device) -> None:
    config, config_sha256 = _load_config(args.config)
    output = (args.output_dir or DEFAULT_OUTPUT).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "component_absolute_placer.pt"
    prereg_path = output / "eval_preregistered.json"
    frozen_path = output / "frozen_predictions.json"
    report_path = output / "report.json"
    if any(path.exists() for path in (checkpoint_path, prereg_path, frozen_path, report_path)):
        raise FileExistsError("refusing to overwrite component placer pilot")
    manifest = _load_json(args.manifest)
    train_records, calibration_records, eval_records, selection, selection_sha256 = (
        _load_rosters(config, manifest)
    )
    socket = _load_socket(device)
    cache = CleanTileCache(args.targets)
    architecture = ComponentAbsoluteConfig(**config["architecture"])
    model = ComponentAbsolutePlacerModel(architecture).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != int(config["expected_parameter_count"]):
        raise ValueError(f"component placer parameter count changed: {parameter_count}")
    training = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        int(training["steps"]),
        eta_min=float(training["learning_rate"]) * 0.08,
    )
    train_boards: list[tuple[ComponentBoard, ComponentBoard]] = []
    precompute_started = perf_counter()
    for index, record in enumerate(train_records):
        pair: list[ComponentBoard] = []
        for draw in training["draw_indices"]:
            pair.append(
                prepare_board(
                    cache,
                    record,
                    draw_index=int(draw),
                    seed=int(training["corruption_seed"]),
                    socket=socket,
                    device=device,
                    include_truth=True,
                )
            )
        if len(pair) != 2:
            raise ValueError("v1 requires exactly two paired corruption draws")
        train_boards.append((pair[0], pair[1]))
        if (index + 1) % 32 == 0:
            print(
                json.dumps(
                    {
                        "event": "fit-precompute",
                        "sources": index + 1,
                        "elapsed_seconds": perf_counter() - precompute_started,
                    }
                ),
                flush=True,
            )
    generator = np.random.default_rng(int(training["seed"]))
    history: list[dict[str, Any]] = []
    train_started = perf_counter()
    model.train()
    for step in range(int(training["steps"])):
        first_board, second_board = train_boards[
            int(generator.integers(len(train_boards)))
        ]
        first_logits = _forward_board(model, first_board, device)
        second_logits = _forward_board(model, second_board, device)
        supervised_losses: list[torch.Tensor] = []
        supervised_diagnostics: list[dict[str, float]] = []
        for board, (purity_logits, offset_logits) in (
            (first_board, first_logits),
            (second_board, second_logits),
        ):
            purity, offsets, _, sizes = _board_targets(board, device)
            item_loss, item_diagnostics = component_absolute_loss(
                purity_logits,
                offset_logits,
                purity,
                offsets,
                component_sizes=sizes,
                offset_weight=float(training["offset_weight"]),
            )
            supervised_losses.append(item_loss)
            supervised_diagnostics.append(item_diagnostics)
        first_components = tuple(
            first_board.components[index]
            for index in first_board.nontrivial_indices
        )
        second_components = tuple(
            second_board.components[index]
            for index in second_board.nontrivial_indices
        )
        first_indices, second_indices = align_components_across_corruptions(
            first_components,
            first_board.tile_to_position,
            second_components,
            second_board.tile_to_position,
            grid=GRID,
        )
        consistency, consistency_diagnostics = paired_component_consistency_loss(
            *first_logits,
            *second_logits,
            first_indices,
            second_indices,
        )
        supervised = torch.stack(supervised_losses).mean()
        loss = supervised + float(training["consistency_weight"]) * consistency
        diagnostics = {
            key: float(np.mean([item[key] for item in supervised_diagnostics]))
            for key in supervised_diagnostics[0]
        }
        diagnostics["supervised_loss"] = float(supervised.detach())
        diagnostics.update(consistency_diagnostics)
        diagnostics["loss"] = float(loss.detach())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip"])
            )
        )
        optimizer.step()
        scheduler.step()
        if step == 0 or (step + 1) % args.log_every == 0 or step + 1 == int(
            training["steps"]
        ):
            row = {
                "step": step + 1,
                **diagnostics,
                "gradient_norm": gradient_norm,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": perf_counter() - train_started,
            }
            history.append(row)
            print(json.dumps({"event": "train", **row}), flush=True)
    training_seconds = perf_counter() - train_started
    model.eval()
    calibration_boards = [
        prepare_board(
            cache,
            record,
            draw_index=int(config["calibration"]["draw_index"]),
            seed=int(config["calibration"]["seed"]),
            socket=socket,
            device=device,
            include_truth=True,
        )
        for record in calibration_records
    ]
    calibration = _calibrate(model, calibration_boards, device)
    checkpoint = {
        "state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "contract": {
            "architecture": "board-conditioned-native-component-anchor-v1",
            "model_config": config["architecture"],
            "parameter_count": parameter_count,
            "original_upright_tiles_only": True,
            "shared_global_roll_vote": False,
            "maximum_anchored_components": 1,
        },
        "selection": {
            "fit_filenames": selection["fit_filenames"],
            "fit_order_digest": selection["fit_order_digest"],
            "evaluation_filenames": selection["evaluation_filenames"],
            "evaluation_order_digest": selection["evaluation_order_digest"],
        },
        "calibration": calibration,
        "preregistration": {"path": str(args.config), "sha256": config_sha256},
    }
    torch.save(checkpoint, checkpoint_path)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    eval_preregistered = {
        "schema": "aiijc-independent-component-anchor-eval-preregistered-v1",
        "written_before_evaluation_target_access": True,
        "pid": os.getpid(),
        "config_sha256": config_sha256,
        "selection_sha256": selection_sha256,
        "fit_digest": selection["fit_order_digest"],
        "evaluation_digest": selection["evaluation_order_digest"],
        "checkpoint_sha256": checkpoint_sha256,
        "threshold": calibration["threshold_selection"],
        "one_inference_arm_only": True,
        "maximum_anchored_components": 1,
        "fallback": "frozen raw decoder144 plus cyclic-border5",
        "anchor_tail": (
            "independent offset plus conservative collision-aware packing; "
            "no cyclic post-shift"
        ),
        "discovery_gate": config["discovery_gate"],
        "target_access_started": False,
    }
    _write_json(prereg_path, eval_preregistered)
    preregistered_sha256 = sha256_file(prereg_path)
    print(
        json.dumps(
            {
                "event": "evaluation-preregistered-before-target-access",
                "pid": os.getpid(),
                "path": str(prereg_path),
                "sha256": preregistered_sha256,
                "checkpoint_sha256": checkpoint_sha256,
                "evaluation_digest": selection["evaluation_order_digest"],
                "threshold": calibration["threshold_selection"],
                "target_access_started": False,
            }
        ),
        flush=True,
    )
    if args.wait_for_eval_confirmation:
        input("Evaluation paused; press Enter after publishing PID/hashes.\n")
    _write_json(
        output / "eval_access_started.json",
        {
            "schema": "aiijc-independent-component-anchor-eval-access-v1",
            "pid": os.getpid(),
            "immutable_preregistration_sha256": preregistered_sha256,
            "target_access_started": True,
        },
    )
    print(json.dumps({"event": "evaluation-target-access-open", "pid": os.getpid()}), flush=True)

    threshold = float(calibration["threshold_selection"]["threshold"])
    blind: list[FrozenEval] = []
    eval_started = perf_counter()
    with torch.inference_mode():
        for index, record in enumerate(eval_records):
            board = prepare_board(
                cache,
                record,
                draw_index=int(config["evaluation"]["draw_index"]),
                seed=int(config["evaluation"]["seed"]),
                socket=socket,
                device=device,
                include_truth=False,
            )
            # Recreate only the frozen pairwise output needed by cyclic5.
            socket_output = socket.model(_tile_tensor(board.raw_tiles, device=device), grid=GRID)
            comparator = select_global_cyclic_translation(
                board.baseline_layout,
                socket_output.right_log_assignment,
                socket_output.down_log_assignment,
                grid=GRID,
                config=CyclicTranslationConfig(border_weight=5.0),
            )
            purity_logits, offset_logits = _forward_board(model, board, device)
            scores, offsets = anchor_confidence(purity_logits, offset_logits)
            top = int(np.argmax(scores))
            selected = bool(float(scores[top]) >= threshold)
            packing: dict[str, Any] | None = None
            all_component_index: int | None = None
            if selected:
                all_component_index = board.nontrivial_indices[top]
                candidate, packing_diagnostics = place_one_component_anchor(
                    board.components,
                    board.baseline_layout,
                    anchor_component_index=all_component_index,
                    anchor_offset=int(offsets[top]),
                    grid=GRID,
                )
                packing = packing_diagnostics.as_dict()
            else:
                candidate = comparator.layout
            sizes = np.asarray(
                [len(board.components[item]) for item in board.nontrivial_indices],
                dtype=np.int16,
            )
            blind.append(
                FrozenEval(
                    case_id=board.case_id,
                    source_filename=board.source_filename,
                    candidate_layout=_strict(candidate),
                    comparator_layout=_strict(comparator.layout),
                    anchor_selected=selected,
                    selected_nontrivial_index=top if selected else None,
                    selected_all_component_index=all_component_index,
                    selected_score=float(scores[top]) if selected else None,
                    selected_offset=int(offsets[top]) if selected else None,
                    component_scores=torch.sigmoid(purity_logits).cpu().numpy(),
                    component_offsets=offsets,
                    baseline_purity_scores=np.column_stack(
                        (sizes.astype(np.float32), board.geometry[:, 5])
                    ),
                    component_sizes=sizes,
                    nontrivial_components=tuple(
                        dict(board.components[item])
                        for item in board.nontrivial_indices
                    ),
                    packing=packing,
                    dirty_tiles_sha256=board.dirty_tiles_sha256,
                )
            )
            if (index + 1) % 4 == 0:
                print(
                    json.dumps(
                        {
                            "event": "eval-freeze",
                            "sources": index + 1,
                            "elapsed_seconds": perf_counter() - eval_started,
                        }
                    ),
                    flush=True,
                )
    frozen_payload = {
        "experiment": config["experiment"],
        "phase": "dirty-component-scores-and-layouts-frozen-before-exact-scoring",
        "config_sha256": config_sha256,
        "eval_preregistered_sha256_before_access": preregistered_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "evaluation_digest": selection["evaluation_order_digest"],
        "threshold": threshold,
        "rows": [
            {
                "case_id": row.case_id,
                "source_filename": row.source_filename,
                "dirty_tiles_sha256": row.dirty_tiles_sha256,
                "anchor_selected": row.anchor_selected,
                "selected_nontrivial_index": row.selected_nontrivial_index,
                "selected_all_component_index": row.selected_all_component_index,
                "selected_score": row.selected_score,
                "selected_offset": row.selected_offset,
                "component_purity_scores": row.component_scores.tolist(),
                "component_offsets": row.component_offsets.tolist(),
                "baseline_purity_scores": row.baseline_purity_scores.tolist(),
                "component_sizes": row.component_sizes.tolist(),
                "nontrivial_components": [
                    [
                        [int(tile), int(position[0]), int(position[1])]
                        for tile, position in sorted(component.items())
                    ]
                    for component in row.nontrivial_components
                ],
                "candidate_tile_at_position": row.candidate_layout.tolist(),
                "comparator_tile_at_position": row.comparator_layout.tolist(),
                "packing": row.packing,
            }
            for row in blind
        ],
        "strict_permutation_count": sum(
            int(
                np.array_equal(np.sort(row.candidate_layout), np.arange(TILE_COUNT))
                and np.array_equal(np.sort(row.comparator_layout), np.arange(TILE_COUNT))
            )
            for row in blind
        ),
    }
    _write_json(frozen_path, frozen_payload)
    frozen_sha256 = sha256_file(frozen_path)
    print(json.dumps({"event": "predictions-frozen", "sha256": frozen_sha256}), flush=True)

    scored: list[dict[str, Any]] = []
    purity_labels: list[np.ndarray] = []
    model_scores: list[np.ndarray] = []
    baseline_scores: list[np.ndarray] = []
    for record, frozen_row in zip(eval_records, blind, strict=True):
        case = prepare_case(
            cache,
            record,
            draw_index=int(config["evaluation"]["draw_index"]),
            seed=int(config["evaluation"]["seed"]),
        )
        dirty_sha256 = hashlib.sha256(case.dirty_tiles.tobytes()).hexdigest()
        if (
            case.case_id != frozen_row.case_id
            or dirty_sha256 != frozen_row.dirty_tiles_sha256
        ):
            raise RuntimeError("deterministic eval case changed after freeze")
        purity, target_offsets, _ = component_absolute_targets(
            frozen_row.nontrivial_components,
            case.input_tile_to_position,
            grid=GRID,
        )
        purity_array = purity.numpy()
        purity_labels.append(purity_array)
        model_scores.append(frozen_row.component_scores)
        baseline_scores.append(frozen_row.baseline_purity_scores)
        reference = np.argsort(case.input_tile_to_position).astype(np.int32)
        candidate = evaluate_layout(
            frozen_row.candidate_layout,
            reference,
            reference_is_exact=True,
        ).as_dict()
        comparator = evaluate_layout(
            frozen_row.comparator_layout,
            reference,
            reference_is_exact=True,
        ).as_dict()
        anchor_correct = None
        if frozen_row.anchor_selected:
            index = int(frozen_row.selected_nontrivial_index)
            anchor_correct = bool(
                purity_array[index]
                and int(frozen_row.selected_offset) == int(target_offsets[index])
            )
        scored.append(
            {
                "case_id": case.case_id,
                "source_filename": case.source_filename,
                "anchor_selected": frozen_row.anchor_selected,
                "anchor_correct": anchor_correct,
                "candidate": candidate,
                "comparator": comparator,
                "exact_delta_tiles": candidate["correct_tile_count"]
                - comparator["correct_tile_count"],
                "adjacency_delta": candidate["adjacency"] - comparator["adjacency"],
            }
        )
    summary = {
        "source_count": len(scored),
        "anchor_selected_boards": sum(row["anchor_selected"] for row in scored),
        "anchor_correct_boards": sum(row["anchor_correct"] is True for row in scored),
        "purity": _purity_metrics(purity_labels, model_scores, baseline_scores),
        "comparator": _layout_summary(scored, "comparator"),
        "candidate": _layout_summary(scored, "candidate"),
        "strict_permutation_count": frozen_payload["strict_permutation_count"],
        "exact_wins_ties_losses": [
            sum(row["exact_delta_tiles"] > 0 for row in scored),
            sum(row["exact_delta_tiles"] == 0 for row in scored),
            sum(row["exact_delta_tiles"] < 0 for row in scored),
        ],
    }
    gate = _evaluate_gate(summary, config)
    report = {
        "experiment": config["experiment"],
        "status": gate["status"],
        "preregistration": {"path": str(args.config.resolve()), "sha256": config_sha256},
        "eval_preregistered": {
            "path": str(prereg_path),
            "sha256_before_access": preregistered_sha256,
        },
        "selection": {
            "commitment_sha256": selection_sha256,
            "fit_digest": selection["fit_order_digest"],
            "evaluation_digest": selection["evaluation_order_digest"],
        },
        "model": {
            "parameter_count": parameter_count,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
        },
        "training": {
            "configuration": training,
            "history": history,
            "precompute_seconds": train_started - precompute_started,
            "training_seconds": training_seconds,
        },
        "calibration": calibration,
        "freeze": {"path": str(frozen_path), "sha256": frozen_sha256},
        "summary": summary,
        "gate": gate,
        "holdout_opened": False,
        "competition_test_opened": False,
        "rows": scored,
    }
    _write_json(report_path, report)
    print(
        json.dumps(
            {
                "event": "complete",
                "report": str(report_path),
                "report_sha256": sha256_file(report_path),
                "gate": gate,
            }
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.log_every <= 0:
        raise ValueError("log-every must be positive")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = _device(args.device, args.allow_nondeterministic_mps)
    if args.mode == "capacity":
        run_capacity(args, device)
    elif args.mode == "selection":
        freeze_selection(args, device)
    else:
        run_pilot(args, device)


if __name__ == "__main__":
    main()

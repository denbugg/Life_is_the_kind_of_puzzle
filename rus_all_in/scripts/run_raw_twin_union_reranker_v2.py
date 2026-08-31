#!/usr/bin/env python3
"""Capacity-check, train and gate the frozen raw32/twin32 union reranker."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from aiijc_puzzle.component_relation_reranker import extract_frozen_socket_context
from aiijc_puzzle.fullres_twin_side_matcher import FullResolutionTwinSideMatcher
from aiijc_puzzle.protocol import compute_protocol_digest, sha256_file
from aiijc_puzzle.raw_twin_union_reranker import (
    FEATURE_NAMES,
    RawTwinUnionBoard,
    RawTwinUnionReranker,
    bidirectional_union_loss,
    prepare_raw_twin_union_board,
    restricted_partial_ot,
    union_edge_labels,
)
from aiijc_puzzle.socket_decoder import (
    SocketDecoderConfig,
    decode_socket_assignments,
    hard_partial_axis_matching,
)
from aiijc_puzzle.socket_sorter_production import (
    LoadedSocketCheckpoint,
    load_socket_checkpoint,
)
from aiijc_puzzle.socket_translation_placer import (
    CyclicTranslationConfig,
    select_global_cyclic_translation,
)
from aiijc_puzzle.synthetic_socket_evaluation import (
    exact_local_retrieval_metrics,
    freeze_topk_candidates,
    load_checkpoint_with_lineage,
    names_digest,
)

try:
    from scripts.run_fullres_twin_side_matcher import (
        CleanBoard,
        _atomic_json,
        _evaluation_exclusion_registry,
        _materialise_training,
        _prepare_boards,
        _procedural_capacity_tiles,
        _project_relative,
        _resolve_device,
        _select_rosters,
        _synchronise,
        _tensor,
        _training_specs,
        _two_view_case,
    )
except ModuleNotFoundError:  # Direct ``python scripts/run_*.py`` execution.
    from run_fullres_twin_side_matcher import (
        CleanBoard,
        _atomic_json,
        _evaluation_exclusion_registry,
        _materialise_training,
        _prepare_boards,
        _procedural_capacity_tiles,
        _project_relative,
        _resolve_device,
        _select_rosters,
        _synchronise,
        _tensor,
        _training_specs,
        _two_view_case,
    )

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/raw_twin_union_reranker_v2_preregistered.json"
DEFAULT_MANIFEST = PROJECT_ROOT / "data/interim/validation_manifest.json"
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
DEFAULT_SOCKET = (
    PROJECT_ROOT / "outputs/socket-matcher/v2-d64-train1024-s1600-r400-dev32/socket_matcher.pt"
)
DEFAULT_TWIN = (
    PROJECT_ROOT
    / "outputs/fullres-twin-side-matcher/v1-fit256-s400-eval24/fullres-twin-side-matcher.pt"
)
DEFAULT_REQUIRED_COMMITMENT = PROJECT_ROOT / "configs/direct_hard_edge_fresh64_confirmation_v1.json"
DEFAULT_ADDITIONAL_COMMITMENT = (
    PROJECT_ROOT / "outputs/component-absolute-placer/v1-selection/selection_commitment.json"
)
GRID = 24
COUNT = GRID * GRID
FIT_SOURCES = 256
EVAL_SOURCES = 24
MAX_STEPS = 400
LOCAL_KS = (1, 5)
TOPK = 32
EDGE_BUDGET = 144


@dataclass(frozen=True)
class FrozenAxisPrediction:
    candidates: np.ndarray
    sources: np.ndarray
    targets: np.ndarray
    confidence: np.ndarray


@dataclass(frozen=True)
class FrozenCasePrediction:
    case_id: str
    source_filename: str
    variants: dict[str, dict[str, FrozenAxisPrediction]]
    assignments: dict[str, tuple[np.ndarray, np.ndarray]]
    runtime_seconds: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("capacity", "benchmark", "pilot"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--socket-checkpoint", type=Path, default=DEFAULT_SOCKET)
    parser.add_argument("--twin-checkpoint", type=Path, default=DEFAULT_TWIN)
    parser.add_argument(
        "--required-exclusion-commitment",
        type=Path,
        default=DEFAULT_REQUIRED_COMMITMENT,
    )
    parser.add_argument(
        "--additional-exclusion-commitment",
        type=Path,
        default=DEFAULT_ADDITIONAL_COMMITMENT,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--capacity-steps", type=int, default=160)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--allow-nondeterministic-mps", action="store_true")
    parser.add_argument("--prefetch-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=20330917)
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 1 <= args.steps <= MAX_STEPS:
        raise ValueError(f"steps must be in [1, {MAX_STEPS}]")
    if not 1 <= args.capacity_steps <= 300:
        raise ValueError("capacity-steps must be in [1, 300]")
    if not 1 <= args.prefetch_workers <= 4:
        raise ValueError("prefetch-workers must be in [1, 4]")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning-rate must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("weight-decay must be finite and non-negative")
    if args.log_every <= 0:
        raise ValueError("log-every must be positive")
    if args.device == "mps" and not args.allow_nondeterministic_mps:
        raise ValueError("MPS requires --allow-nondeterministic-mps")


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "aiijc-raw-twin-union-reranker-v2-preregistered":
        raise ValueError("unexpected preregistration schema")
    if payload["candidate_roster"]["raw_candidate_roster_immutable"] is not True:
        raise ValueError("preregistration must freeze the raw/twin candidate union")
    if payload["model"]["last_layer_zero_initialised"] is not True:
        raise ValueError("preregistration must require raw-order-preserving zero init")
    return payload


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def _load_twin_checkpoint(
    path: Path,
    *,
    device: torch.device,
) -> tuple[FullResolutionTwinSideMatcher, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("contract"), dict):
        raise ValueError("twin checkpoint has no architecture contract")
    contract = dict(payload["contract"])
    if contract.get("architecture") != "fullres-ordered-twin-side-matcher-v1":
        raise ValueError("unsupported twin checkpoint architecture")
    if contract.get("pixel_prediction_head") is not False:
        raise ValueError("twin checkpoint must be matcher-only")
    model = FullResolutionTwinSideMatcher(
        dimension=int(contract["dimension"]),
        field_blocks=int(contract["field_blocks"]),
        sequence_blocks=int(contract["sequence_blocks"]),
        raw_skip_gain=float(contract["raw_skip_gain"]),
    ).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval().requires_grad_(False)
    return model, contract


def _synthetic_capacity_board(
    seed: int,
    *,
    device: torch.device,
) -> tuple[RawTwinUnionBoard, torch.Tensor]:
    """Make a target-independent-shaped 4x4 task with a learnable edge signal."""

    rng = np.random.default_rng(seed)
    grid = 4
    count = grid * grid
    layout = rng.permutation(count).astype(np.int64)
    truth = np.full((2, count), -1, dtype=np.int64)
    positions = np.arange(count)
    right = positions % grid != grid - 1
    down = positions < count - grid
    truth[0, layout[right]] = layout[positions[right] + 1]
    truth[1, layout[down]] = layout[positions[down] + grid]
    sources: list[int] = []
    targets: list[int] = []
    axes: list[int] = []
    rows: list[tuple[np.ndarray, ...]] = []
    for axis in range(2):
        axis_rows: list[np.ndarray] = []
        for source in range(count):
            candidates = np.asarray(
                [target for target in range(count) if target != source],
                dtype=np.int32,
            )
            axis_rows.append(candidates)
            sources.extend([source] * len(candidates))
            targets.extend(candidates.tolist())
            axes.extend([axis] * len(candidates))
        rows.append(tuple(axis_rows))
    source_array = np.asarray(sources, dtype=np.int64)
    target_array = np.asarray(targets, dtype=np.int64)
    axis_array = np.asarray(axes, dtype=np.int64)
    positive = truth[axis_array, source_array] == target_array
    values = rng.normal(0, 0.35, (len(sources), len(FEATURE_NAMES))).astype(np.float32)
    # Multiple continuous coordinates ensure this tests the full head rather than one bit.
    values[:, 0] = rng.normal(0, 0.45, len(sources)) + 2.2 * positive
    values[:, 1] = rng.normal(0, 0.45, len(sources)) + 1.3 * positive
    values[:, 6] = rng.normal(0, 0.30, len(sources)) + 0.8 * positive
    raw = rng.normal(0, 0.20, len(sources)).astype(np.float32)
    board = RawTwinUnionBoard(
        values=torch.from_numpy(values).to(device),
        raw_scores=torch.from_numpy(raw).to(device),
        axis=torch.from_numpy(axis_array).to(device),
        source=torch.from_numpy(source_array).to(device),
        target=torch.from_numpy(target_array).to(device),
        rows=(rows[0], rows[1]),
        grid=grid,
    )
    return board, torch.from_numpy(layout).to(device)


def _capacity_r1(
    model: RawTwinUnionReranker, board: RawTwinUnionBoard, layout: torch.Tensor
) -> float:
    labels = union_edge_labels(board, layout)
    scores = model(board).scores.detach()
    correct = 0
    total = 0
    count = board.grid * board.grid
    for axis in (0, 1):
        for source in range(count):
            selected = (board.axis == axis) & (board.source == source)
            selected_labels = labels[selected]
            if bool(selected_labels.any().item()):
                correct += int(selected_labels[scores[selected].argmax()].item())
                total += 1
    return correct / total


def run_capacity(args: argparse.Namespace, config: dict[str, Any]) -> None:
    device = _resolve_device(args.device)
    if device.type == "mps" and not args.allow_nondeterministic_mps:
        raise ValueError("capacity MPS requires explicit nondeterminism acknowledgment")
    _seed_everything(args.seed)
    model = RawTwinUnionReranker().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=2e-4)
    history: list[dict[str, Any]] = []
    started = perf_counter()
    model.train()
    for step in range(args.capacity_steps):
        board, layout = _synthetic_capacity_board(args.seed + step, device=device)
        output = model(board)
        loss, diagnostics = bidirectional_union_loss(
            output,
            board,
            union_edge_labels(board, layout),
            hard_negatives_per_axis=64,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        record = {
            "step": step + 1,
            **diagnostics,
            "gradient_norm": float(gradient.detach().cpu()),
        }
        history.append(record)
        if step == 0 or (step + 1) % args.log_every == 0:
            print(json.dumps({"event": "capacity", **record}), flush=True)
    model.eval()
    evaluation = [
        _synthetic_capacity_board(args.seed + 10000 + index, device=device) for index in range(8)
    ]
    with torch.inference_mode():
        r1 = float(np.mean([_capacity_r1(model, board, layout) for board, layout in evaluation]))
    _synchronise(device)
    initial = float(np.mean([row["loss"] for row in history[:10]]))
    final = float(np.mean([row["loss"] for row in history[-10:]]))
    passed = bool(final < 0.75 * initial and r1 >= 0.90)
    report = {
        "schema": "aiijc-raw-twin-union-reranker-capacity-v2",
        "status": "pass" if passed else "fail-stop",
        "preregistration_sha256": sha256_file(args.config),
        "device": str(device),
        "mps_bitwise_reproducibility_claimed": device.type != "mps",
        "grid": 4,
        "steps": args.capacity_steps,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "initial_loss": initial,
        "final_loss": final,
        "fresh_evaluation_r1": r1,
        "gate": {
            "loss_ratio_max": 0.75,
            "minimum_r1": 0.90,
            "passed": passed,
        },
        "runtime_seconds": perf_counter() - started,
        "history": history,
        "config_contract": config["model"],
    }
    _atomic_json(args.output_dir / "capacity-report.json", report)
    print(json.dumps({"event": "capacity_complete", "passed": passed, "r1": r1}), flush=True)
    if not passed:
        raise RuntimeError("4x4 capacity gate failed")


def _frozen_union_board(
    tiles: np.ndarray,
    *,
    grid: int,
    socket: LoadedSocketCheckpoint,
    twin: FullResolutionTwinSideMatcher,
    device: torch.device,
    topk: int = TOPK,
) -> tuple[RawTwinUnionBoard, Any]:
    tile_tensor = _tensor(tiles, device)
    # ``no_grad`` keeps constants usable by the trainable head's backward pass.
    with torch.no_grad():
        tokens, socket_output = extract_frozen_socket_context(
            socket.model,
            tile_tensor,
            grid=grid,
        )
        twin_output = twin(tile_tensor)
        board = prepare_raw_twin_union_board(
            tokens[0],
            socket_output,
            twin_output,
            grid=grid,
            topk=topk,
        )
    return board, socket_output


def _benchmark_device(
    name: str,
    tiles: np.ndarray,
    layout: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    device = torch.device(name)
    _seed_everything(73)
    socket = load_socket_checkpoint(args.socket_checkpoint, device=device)
    twin, _ = _load_twin_checkpoint(args.twin_checkpoint, device=device)
    model = RawTwinUnionReranker().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    timings: list[float] = []
    diagnostics: dict[str, Any] = {}
    for iteration in range(2):
        started = perf_counter()
        board, _ = _frozen_union_board(
            tiles,
            grid=GRID,
            socket=socket,
            twin=twin,
            device=device,
        )
        target = torch.from_numpy(layout).to(device)
        output = model(board)
        loss, diagnostics = bidirectional_union_loss(
            output,
            board,
            union_edge_labels(board, target),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        _synchronise(device)
        if iteration:
            timings.append(perf_counter() - started)
    seconds = float(np.mean(timings))
    del socket, twin, model, optimizer
    if name == "mps":
        torch.mps.empty_cache()
    return {
        "seconds_per_full576_frozen_features_and_head_step": seconds,
        "steps_per_second": 1.0 / seconds,
        "candidate_edges": diagnostics["candidate_edges"],
        "positive_edges": diagnostics["positive_edges"],
        "timed_repeats": len(timings),
    }


def run_benchmark(args: argparse.Namespace) -> None:
    clean = np.tile(_procedural_capacity_tiles(), (36, 1, 1, 1))
    tiles, _, layout = _two_view_case(
        clean,
        first_seed=args.seed + 7001,
        second_seed=args.seed + 7002,
        permutation_seed=args.seed + 7003,
    )
    names = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    results: dict[str, Any] = {}
    for name in names:
        started = perf_counter()
        results[name] = _benchmark_device(name, tiles, layout, args)
        results[name]["wall_seconds_including_warmup"] = perf_counter() - started
        print(
            json.dumps({"event": "benchmark_device", "device": name, **results[name]}), flush=True
        )
    chosen = min(
        results,
        key=lambda name: results[name]["seconds_per_full576_frozen_features_and_head_step"],
    )
    report = {
        "schema": "aiijc-raw-twin-union-reranker-device-benchmark-v2",
        "preregistration_sha256": sha256_file(args.config),
        "workload": (
            "full576 frozen d64+twin extraction, immutable union features, head backward+AdamW"
        ),
        "results": results,
        "chosen_device": chosen,
    }
    _atomic_json(args.output_dir / "device-benchmark.json", report)
    print(json.dumps({"event": "benchmark_complete", "chosen": chosen}), flush=True)


def _required_exclusion_names(path: Path) -> tuple[tuple[str, ...], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    selection = payload.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("required exclusion commitment has no selection")
    names = selection.get("source_filenames")
    if not isinstance(names, list) or len(names) != 64 or len(names) != len(set(names)):
        raise ValueError("required exclusion commitment must contain 64 unique source filenames")
    if names_digest(names) != selection.get("source_order_digest"):
        raise ValueError("required exclusion source order digest mismatch")
    return tuple(names), payload


def _fit_eval_commitment_names(path: Path) -> tuple[tuple[str, ...], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    fit = payload.get("fit_filenames")
    evaluation = payload.get("evaluation_filenames")
    if not isinstance(fit, list) or not isinstance(evaluation, list):
        raise ValueError("additional commitment must contain fit/evaluation filenames")
    if names_digest(fit) != payload.get("fit_order_digest"):
        raise ValueError("additional commitment fit digest mismatch")
    if names_digest(evaluation) != payload.get("evaluation_order_digest"):
        raise ValueError("additional commitment evaluation digest mismatch")
    names = tuple(dict.fromkeys((*fit, *evaluation)))
    if len(names) != len(set(names)):
        raise ValueError("additional commitment contains duplicate fit/evaluation names")
    return names, payload


def _selection_commitment(
    args: argparse.Namespace,
    config: dict[str, Any],
    manifest: dict[str, Any],
    socket_lineage: tuple[str, ...],
    twin_lineage: tuple[str, ...],
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], Path]:
    path = args.output_dir / "selection-commitment.json"
    if path.is_file():
        frozen = json.loads(path.read_text(encoding="utf-8"))
        if frozen.get("schema") != "aiijc-raw-twin-union-reranker-selection-commitment-v2":
            raise ValueError("existing selection commitment has an unexpected schema")
        expected = {
            "preregistration_sha256": sha256_file(args.config),
            "manifest_sha256": sha256_file(args.manifest),
            "namespace": config["selection"]["namespace"],
            "seed": args.seed,
        }
        if any(frozen.get(key) != value for key, value in expected.items()):
            raise ValueError("existing frozen selection does not match this exact run")
        if frozen.get("required_fresh64_commitment", {}).get("sha256") != sha256_file(
            args.required_exclusion_commitment
        ):
            raise ValueError("frozen fresh64 exclusion commitment changed")
        if frozen.get("required_component_placer_commitment", {}).get("sha256") != sha256_file(
            args.additional_exclusion_commitment
        ):
            raise ValueError("frozen component-placer exclusion commitment changed")
        fit_names = tuple(frozen.get("fit_filenames", ()))
        eval_names = tuple(frozen.get("evaluation_filenames", ()))
        if (
            len(fit_names) != FIT_SOURCES
            or len(eval_names) != EVAL_SOURCES
            or names_digest(fit_names) != frozen.get("fit_order_digest")
            or names_digest(eval_names) != frozen.get("evaluation_order_digest")
            or set(fit_names) & set(eval_names)
        ):
            raise ValueError("existing frozen selection roster is malformed")
        records = {str(record["filename"]): dict(record) for record in manifest["splits"]["train"]}
        if not (set(fit_names) | set(eval_names)).issubset(records):
            raise ValueError("frozen selection contains names absent from manifest train")
        return (
            tuple(records[name] for name in fit_names),
            tuple(records[name] for name in eval_names),
            path,
        )
    combined_lineage = tuple(sorted(set(socket_lineage) | set(twin_lineage)))
    excluded, registry = _evaluation_exclusion_registry(
        combined_lineage,
        output_dir=args.output_dir,
    )
    required, _ = _required_exclusion_names(args.required_exclusion_commitment)
    additional, _ = _fit_eval_commitment_names(args.additional_exclusion_commitment)
    excluded.update(additional)
    registry.append(
        {
            "path": _project_relative(args.additional_exclusion_commitment),
            "sha256": sha256_file(args.additional_exclusion_commitment),
            "panel_filename_count": len(additional),
            "role": "explicit-concurrent-fit-and-evaluation-exclusion",
        }
    )
    missing = sorted((set(required) | set(additional)) - excluded)
    if missing:
        raise RuntimeError(f"required concurrent exclusion missing {len(missing)} names")
    fit, evaluation = _select_rosters(
        manifest,
        excluded,
        seed=args.seed,
        namespace=config["selection"]["namespace"],
    )
    fit_names = tuple(str(record["filename"]) for record in fit)
    eval_names = tuple(str(record["filename"]) for record in evaluation)
    if len(fit_names) != FIT_SOURCES or len(eval_names) != EVAL_SOURCES:
        raise RuntimeError("fit/evaluation roster size invariant failed")
    if set(fit_names) & set(eval_names) or set(eval_names) & excluded:
        raise RuntimeError("fit/evaluation source-disjoint invariant failed")
    commitment = {
        "schema": "aiijc-raw-twin-union-reranker-selection-commitment-v2",
        "status": "frozen-before-selected-target-access",
        "preregistration_path": _project_relative(args.config),
        "preregistration_sha256": sha256_file(args.config),
        "manifest_sha256": sha256_file(args.manifest),
        "manifest_protocol_digest": manifest["protocol_digest"],
        "manifest_split": "train",
        "namespace": config["selection"]["namespace"],
        "seed": args.seed,
        "socket_checkpoint_sha256": sha256_file(args.socket_checkpoint),
        "twin_checkpoint_sha256": sha256_file(args.twin_checkpoint),
        "combined_recursive_lineage_count": len(combined_lineage),
        "excluded_filename_count": len(excluded),
        "excluded_filename_digest": names_digest(tuple(sorted(excluded))),
        "exclusion_registry": registry,
        "required_fresh64_commitment": {
            "path": _project_relative(args.required_exclusion_commitment),
            "sha256": sha256_file(args.required_exclusion_commitment),
            "source_count": len(required),
            "source_order_digest": names_digest(required),
            "all_present_in_exclusion": True,
        },
        "required_component_placer_commitment": {
            "path": _project_relative(args.additional_exclusion_commitment),
            "sha256": sha256_file(args.additional_exclusion_commitment),
            "fit_and_evaluation_source_count": len(additional),
            "source_set_digest": names_digest(additional, sort_names=True),
            "all_present_in_exclusion": True,
        },
        "fit_filenames": list(fit_names),
        "fit_order_digest": names_digest(fit_names),
        "evaluation_filenames": list(eval_names),
        "evaluation_order_digest": names_digest(eval_names),
        "fit_evaluation_overlap": [],
        "evaluation_exclusion_overlap": [],
        "holdout_and_competition_test_opened": False,
    }
    _atomic_json(path, commitment)
    return fit, evaluation, path


def _train_one(
    model: RawTwinUnionReranker,
    optimizer: torch.optim.Optimizer,
    tiles: np.ndarray,
    layout: np.ndarray,
    *,
    socket: LoadedSocketCheckpoint,
    twin: FullResolutionTwinSideMatcher,
    device: torch.device,
) -> dict[str, Any]:
    board, _ = _frozen_union_board(
        tiles,
        grid=GRID,
        socket=socket,
        twin=twin,
        device=device,
    )
    target = torch.from_numpy(layout).to(device)
    output = model(board)
    loss, diagnostics = bidirectional_union_loss(
        output,
        board,
        union_edge_labels(board, target),
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return diagnostics | {"gradient_norm": float(gradient.detach().cpu())}


def train_model(
    model: RawTwinUnionReranker,
    boards: list[CleanBoard],
    args: argparse.Namespace,
    *,
    socket: LoadedSocketCheckpoint,
    twin: FullResolutionTwinSideMatcher,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        args.steps,
        eta_min=args.learning_rate * 0.05,
    )
    specs = _training_specs(args.steps, seed=args.seed)
    history: list[dict[str, Any]] = []
    started = perf_counter()
    prefetch_wait = 0.0
    model.train()
    with ThreadPoolExecutor(max_workers=args.prefetch_workers) as executor:
        futures: dict[int, Future[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
        submit = 0
        window = max(2, 2 * args.prefetch_workers)
        while submit < min(window, args.steps):
            spec = specs[submit]
            futures[submit] = executor.submit(
                _materialise_training, boards[spec.source_index], spec
            )
            submit += 1
        for step, spec in enumerate(specs):
            wait_started = perf_counter()
            first, _, layout = futures.pop(step).result()
            prefetch_wait += perf_counter() - wait_started
            if submit < args.steps:
                next_spec = specs[submit]
                futures[submit] = executor.submit(
                    _materialise_training,
                    boards[next_spec.source_index],
                    next_spec,
                )
                submit += 1
            diagnostics = _train_one(
                model,
                optimizer,
                first,
                layout,
                socket=socket,
                twin=twin,
                device=device,
            )
            scheduler.step()
            record = {
                "step": step + 1,
                "source_filename": boards[spec.source_index].filename,
                "corruption_seed": spec.first_seed,
                "permutation_seed": spec.permutation_seed,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                **diagnostics,
            }
            history.append(record)
            # Full-board candidate counts vary, so MPS otherwise retains many
            # incompatible cached buffers across the 400-step run.
            if device.type == "mps" and (step + 1) % 10 == 0:
                torch.mps.empty_cache()
            if step == 0 or (step + 1) % args.log_every == 0:
                recent = history[-args.log_every :]
                print(
                    json.dumps(
                        {
                            "event": "train",
                            "step": step + 1,
                            "loss": float(np.mean([row["loss"] for row in recent])),
                            "positive_edges": float(
                                np.mean([row["positive_edges"] for row in recent])
                            ),
                            "elapsed_seconds": perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
    _synchronise(device)
    return history, {
        "training_seconds": perf_counter() - started,
        "prefetch_wait_seconds": prefetch_wait,
        "prefetch_workers": args.prefetch_workers,
    }


def _case_seeds(seed: int, filename: str) -> tuple[int, int]:
    digest = hashlib.sha256(f"{seed}\0{filename}\0raw-twin-v2".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63), int.from_bytes(
        digest[8:16], "little"
    ) % (2**63)


def _truth_by_anchor(reference: np.ndarray, *, axis: str) -> np.ndarray:
    positions = np.arange(COUNT)
    valid = positions % GRID != GRID - 1 if axis == "right" else positions < COUNT - GRID
    delta = 1 if axis == "right" else GRID
    truth = np.full(COUNT, -1, dtype=np.int32)
    truth[reference[positions[valid]]] = reference[positions[valid] + delta]
    return truth


def _freeze_axis(assignment: np.ndarray, *, axis: str) -> FrozenAxisPrediction:
    real = np.ascontiguousarray(assignment[:COUNT, :COUNT], dtype=np.float32)
    candidates = freeze_topk_candidates(real, max_k=max(LOCAL_KS))
    matching = hard_partial_axis_matching(assignment, grid=GRID, axis=axis)
    return FrozenAxisPrediction(
        candidates=candidates,
        sources=np.asarray([edge.source for edge in matching.edges], dtype=np.int32),
        targets=np.asarray([edge.target for edge in matching.edges], dtype=np.int32),
        confidence=np.asarray([edge.confidence for edge in matching.edges], dtype=np.float32),
    )


def _assert_projection_inside_union(
    board: RawTwinUnionBoard,
    *,
    axis: int,
    prediction: FrozenAxisPrediction,
) -> None:
    for source, target in zip(prediction.sources, prediction.targets, strict=True):
        if int(target) not in board.rows[axis][int(source)]:
            raise RuntimeError("learned hard projection selected a forbidden outside-union edge")


@torch.inference_mode()
def freeze_case_prediction(
    model: RawTwinUnionReranker,
    tiles: np.ndarray,
    *,
    case_id: str,
    source_filename: str,
    socket: LoadedSocketCheckpoint,
    twin: FullResolutionTwinSideMatcher,
    device: torch.device,
) -> FrozenCasePrediction:
    started = perf_counter()
    board, socket_output = _frozen_union_board(
        tiles,
        grid=GRID,
        socket=socket,
        twin=twin,
        device=device,
    )
    feature_seconds = perf_counter() - started
    frozen: dict[str, dict[str, FrozenAxisPrediction]] = {}
    assignments: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    inference_started = perf_counter()
    raw_pair = (
        np.ascontiguousarray(socket_output.right_log_assignment[0].float().cpu().numpy()),
        np.ascontiguousarray(socket_output.down_log_assignment[0].float().cpu().numpy()),
    )
    learned_right, learned_down = restricted_partial_ot(
        board,
        model(board).scores,
        socket_output,
    )
    learned_pair = (
        np.ascontiguousarray(learned_right[0].float().cpu().numpy()),
        np.ascontiguousarray(learned_down[0].float().cpu().numpy()),
    )
    for name, pair in (
        ("socket_d64_frozen", raw_pair),
        ("learned_union", learned_pair),
    ):
        assignments[name] = pair
        frozen[name] = {
            "right": _freeze_axis(pair[0], axis="right"),
            "down": _freeze_axis(pair[1], axis="down"),
        }
    _assert_projection_inside_union(
        board,
        axis=0,
        prediction=frozen["learned_union"]["right"],
    )
    _assert_projection_inside_union(
        board,
        axis=1,
        prediction=frozen["learned_union"]["down"],
    )
    return FrozenCasePrediction(
        case_id=case_id,
        source_filename=source_filename,
        variants=frozen,
        assignments=assignments,
        runtime_seconds={
            "frozen_feature_extraction": feature_seconds,
            "reranker_ot_and_projection": perf_counter() - inference_started,
        },
    )


def _write_frozen_predictions(
    predictions: list[FrozenCasePrediction],
    output_dir: Path,
) -> tuple[Path, Path]:
    arrays: dict[str, np.ndarray] = {}
    cases: list[dict[str, Any]] = []
    for index, prediction in enumerate(predictions):
        prefix = f"case_{index:04d}"
        for variant, axes in prediction.variants.items():
            for axis, value in axes.items():
                arrays[f"{prefix}__{variant}__{axis}__candidates"] = value.candidates
                arrays[f"{prefix}__{variant}__{axis}__sources"] = value.sources
                arrays[f"{prefix}__{variant}__{axis}__targets"] = value.targets
                arrays[f"{prefix}__{variant}__{axis}__confidence"] = value.confidence
        cases.append(
            {
                "prefix": prefix,
                "case_id": prediction.case_id,
                "source_filename": prediction.source_filename,
                "runtime_seconds": prediction.runtime_seconds,
            }
        )
    array_path = output_dir / "frozen-dirty-only-predictions.npz"
    np.savez_compressed(array_path, **arrays)
    metadata_path = output_dir / "frozen-dirty-only-predictions.json"
    _atomic_json(
        metadata_path,
        {
            "schema": "aiijc-raw-twin-union-frozen-predictions-v2",
            "contains_exact_references": False,
            "contains_clean_or_generated_pixels": False,
            "contains_layouts": False,
            "candidate_union": ("immutable raw32 union twin32 union frozen raw hard projection"),
            "outside_union_forbidden": True,
            "cases": cases,
        },
    )
    return array_path, metadata_path


def _score_predictions(
    predictions: list[FrozenCasePrediction],
    references: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, Any]]:
    variants = ("socket_d64_frozen", "learned_union")
    local: dict[str, dict[str, int | float]] = {
        name: {"pooled_total": 0, **{f"pooled_hits_at_{k}": 0 for k in LOCAL_KS}}
        for name in variants
    }
    hard: dict[str, dict[str, int | float]] = {
        name: {
            "projected_edges": 0,
            "projected_correct": 0,
            "top144_edges": 0,
            "top144_correct": 0,
        }
        for name in variants
    }
    for prediction in predictions:
        reference = references[prediction.case_id]
        for name in variants:
            metrics = exact_local_retrieval_metrics(
                prediction.variants[name]["right"].candidates,
                prediction.variants[name]["down"].candidates,
                reference,
                ks=LOCAL_KS,
            )
            local[name]["pooled_total"] += int(metrics["pooled_total"])
            for k in LOCAL_KS:
                local[name][f"pooled_hits_at_{k}"] += int(metrics[f"pooled_hits_at_{k}"])
            for axis in ("right", "down"):
                frozen = prediction.variants[name][axis]
                truth = _truth_by_anchor(reference, axis=axis)
                correct = truth[frozen.sources] == frozen.targets
                hard[name]["projected_edges"] += len(correct)
                hard[name]["projected_correct"] += int(correct.sum())
                order = np.argsort(-frozen.confidence, kind="stable")[:EDGE_BUDGET]
                hard[name]["top144_edges"] += len(order)
                hard[name]["top144_correct"] += int(correct[order].sum())
    for values in local.values():
        total = int(values["pooled_total"])
        for k in LOCAL_KS:
            values[f"pooled_r{k}"] = int(values[f"pooled_hits_at_{k}"]) / total
    for values in hard.values():
        values["projected_precision"] = int(values["projected_correct"]) / int(
            values["projected_edges"]
        )
        values["top144_precision"] = int(values["top144_correct"]) / int(values["top144_edges"])
        values["projected_correct_per_board"] = int(values["projected_correct"]) / len(predictions)
        values["top144_correct_per_board"] = int(values["top144_correct"]) / len(predictions)
    return local, hard


def _adjacency_fraction(layout: np.ndarray, reference: np.ndarray) -> float:
    predicted = np.asarray(layout).reshape(GRID, GRID)
    exact = np.asarray(reference).reshape(GRID, GRID)
    right_truth = {
        (int(exact[row, col]), int(exact[row, col + 1]))
        for row in range(GRID)
        for col in range(GRID - 1)
    }
    down_truth = {
        (int(exact[row, col]), int(exact[row + 1, col]))
        for row in range(GRID - 1)
        for col in range(GRID)
    }
    correct = sum(
        (int(predicted[row, col]), int(predicted[row, col + 1])) in right_truth
        for row in range(GRID)
        for col in range(GRID - 1)
    ) + sum(
        (int(predicted[row, col]), int(predicted[row + 1, col])) in down_truth
        for row in range(GRID - 1)
        for col in range(GRID)
    )
    return correct / (2 * GRID * (GRID - 1))


def _decoder_metrics(
    predictions: list[FrozenCasePrediction],
    references: dict[str, np.ndarray],
) -> dict[str, Any]:
    config = SocketDecoderConfig(
        component_edge_budget_per_axis=EDGE_BUDGET,
        swap_edge_budget_per_axis=EDGE_BUDGET,
        max_swap_steps=24,
    )
    output: dict[str, Any] = {}
    for variant in ("socket_d64_frozen", "learned_union"):
        exact_counts: list[int] = []
        adjacency: list[float] = []
        cyclic_changes = 0
        runtimes: list[float] = []
        for prediction in predictions:
            right, down = prediction.assignments[variant]
            started = perf_counter()
            decoded = decode_socket_assignments(right, down, grid=GRID, config=config)
            cyclic = select_global_cyclic_translation(
                decoded.layout,
                right,
                down,
                grid=GRID,
                config=CyclicTranslationConfig(border_weight=5.0),
            )
            layout = cyclic.layout
            if not np.array_equal(np.sort(layout), np.arange(COUNT)):
                raise RuntimeError("decoder/cyclic output is not a strict permutation")
            reference = references[prediction.case_id]
            exact_counts.append(int(np.count_nonzero(layout == reference)))
            adjacency.append(_adjacency_fraction(layout, reference))
            cyclic_changes += int(cyclic.diagnostics.changed)
            runtimes.append(perf_counter() - started)
        output[variant] = {
            "mean_exact_tiles": float(np.mean(exact_counts)),
            "mean_exact_fraction": float(np.mean(exact_counts) / COUNT),
            "mean_adjacency_fraction": float(np.mean(adjacency)),
            "cyclic_changed_boards": cyclic_changes,
            "mean_decoder_cyclic_seconds": float(np.mean(runtimes)),
            "strict_original_upright_permutations": True,
        }
    return output


def run_pilot(args: argparse.Namespace, config: dict[str, Any]) -> None:
    capacity_path = args.output_dir / "capacity-report.json"
    benchmark_path = args.output_dir / "device-benchmark.json"
    if (
        not capacity_path.is_file()
        or json.loads(capacity_path.read_text())["gate"]["passed"] is not True
    ):
        raise RuntimeError("passing capacity report is required before pilot")
    if not benchmark_path.is_file():
        raise RuntimeError("device benchmark is required before pilot")
    benchmark = json.loads(benchmark_path.read_text())
    device_name = benchmark["chosen_device"] if args.device == "auto" else args.device
    device = _resolve_device(device_name)
    if device.type == "mps" and not args.allow_nondeterministic_mps:
        raise ValueError("pilot MPS requires explicit nondeterminism acknowledgment")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("protocol_digest") != compute_protocol_digest(manifest):
        raise ValueError("manifest protocol digest is invalid")
    _, socket_lineage = load_checkpoint_with_lineage(
        args.socket_checkpoint,
        project_root=PROJECT_ROOT,
    )
    _, twin_lineage = load_checkpoint_with_lineage(
        args.twin_checkpoint,
        project_root=PROJECT_ROOT,
    )
    fit_records, eval_records, commitment_path = _selection_commitment(
        args,
        config,
        manifest,
        socket_lineage.filenames,
        twin_lineage.filenames,
    )
    # Target files are first opened only after the commitment above is durable.
    fit_boards = _prepare_boards(fit_records, args.targets)
    _seed_everything(args.seed)
    socket = load_socket_checkpoint(args.socket_checkpoint, device=device)
    twin, twin_contract = _load_twin_checkpoint(args.twin_checkpoint, device=device)
    model = RawTwinUnionReranker().to(device)
    zero_board_tiles, _, _ = _two_view_case(
        np.tile(_procedural_capacity_tiles(), (36, 1, 1, 1)),
        first_seed=11,
        second_seed=12,
        permutation_seed=13,
    )
    zero_board, _ = _frozen_union_board(
        zero_board_tiles,
        grid=GRID,
        socket=socket,
        twin=twin,
        device=device,
    )
    with torch.inference_mode():
        zero_output = model(zero_board)
    if not torch.equal(zero_output.scores, zero_board.raw_scores):
        raise RuntimeError("zero init did not exactly reproduce raw d64 union ordering")
    history, runtime = train_model(
        model,
        fit_boards,
        args,
        socket=socket,
        twin=twin,
        device=device,
    )
    checkpoint_path = args.output_dir / "raw-twin-union-reranker-v2.pt"
    commitment = json.loads(commitment_path.read_text(encoding="utf-8"))
    torch.save(
        {
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "contract": {
                "architecture": "raw-twin-union-reranker-v2",
                "feature_dimension": len(FEATURE_NAMES),
                "hidden_dimension": model.hidden_dimension,
                "residual_limit": model.residual_limit,
                "raw_topk": TOPK,
                "twin_topk": TOPK,
                "outside_union_fill": -10000.0,
                "pixel_prediction": False,
            },
            "selection": {
                "train_filenames": commitment["fit_filenames"],
                "train_digest": names_digest(commitment["fit_filenames"], sort_names=True),
                "evaluation_filenames": commitment["evaluation_filenames"],
                "evaluation_digest": names_digest(commitment["evaluation_filenames"]),
            },
            "training_history": history,
        },
        checkpoint_path,
    )
    model.eval()
    # Evaluation source pixels are deferred until the unchanged fit is complete.
    eval_boards = _prepare_boards(eval_records, args.targets)
    predictions: list[FrozenCasePrediction] = []
    references: dict[str, np.ndarray] = {}
    for index, board in enumerate(eval_boards):
        corruption_seed, permutation_seed = _case_seeds(args.seed, board.filename)
        dirty, _, reference = _two_view_case(
            board.tiles,
            first_seed=corruption_seed,
            second_seed=corruption_seed + 1,
            permutation_seed=permutation_seed,
        )
        case_id = f"raw-twin-v2-{index:03d}-{Path(board.filename).stem}"
        predictions.append(
            freeze_case_prediction(
                model,
                dirty,
                case_id=case_id,
                source_filename=board.filename,
                socket=socket,
                twin=twin,
                device=device,
            )
        )
        references[case_id] = np.ascontiguousarray(reference, dtype=np.int32)
        print(
            json.dumps({"event": "freeze_eval", "case": index + 1, "total": len(eval_boards)}),
            flush=True,
        )
    frozen_npz, frozen_json = _write_frozen_predictions(predictions, args.output_dir)
    # Exact references are touched only after all dirty-only choices are frozen above.
    local, hard = _score_predictions(predictions, references)
    raw = "socket_d64_frozen"
    learned = "learned_union"
    ranking_arm = bool(
        local[learned]["pooled_r1"] - local[raw]["pooled_r1"] >= 0.0025
        and local[learned]["pooled_r5"] - local[raw]["pooled_r5"] >= 0.0
    )
    top144_delta = hard[learned]["top144_correct_per_board"] - hard[raw]["top144_correct_per_board"]
    precision_delta = hard[learned]["top144_precision"] - hard[raw]["top144_precision"]
    hard_arm = bool(top144_delta >= 2.0 and precision_delta >= 0.0)
    passed = ranking_arm or hard_arm
    decoder = _decoder_metrics(predictions, references) if passed else None
    report = {
        "schema": "aiijc-raw-twin-union-reranker-pilot-v2",
        "status": "local-gate-pass" if passed else "local-gate-fail-stop",
        "preregistration": _project_relative(args.config),
        "preregistration_sha256": sha256_file(args.config),
        "selection_commitment": _project_relative(commitment_path),
        "selection_commitment_sha256": sha256_file(commitment_path),
        "device": str(device),
        "mps_nondeterminism_acknowledged": device.type != "mps" or args.allow_nondeterministic_mps,
        "steps": args.steps,
        "training_runtime": runtime,
        "training_final_20": {
            "loss": float(np.mean([row["loss"] for row in history[-20:]])),
            "positive_edges": float(np.mean([row["positive_edges"] for row in history[-20:]])),
        },
        "checkpoints": {
            "socket": {
                "path": _project_relative(args.socket_checkpoint),
                "sha256": sha256_file(args.socket_checkpoint),
            },
            "twin": {
                "path": _project_relative(args.twin_checkpoint),
                "sha256": sha256_file(args.twin_checkpoint),
                "contract": twin_contract,
            },
            "reranker": {
                "path": _project_relative(checkpoint_path),
                "sha256": sha256_file(checkpoint_path),
            },
        },
        "frozen_predictions": {
            "arrays": _project_relative(frozen_npz),
            "arrays_sha256": sha256_file(frozen_npz),
            "metadata": _project_relative(frozen_json),
            "metadata_sha256": sha256_file(frozen_json),
            "contains_exact_references": False,
        },
        "local_partial_ot_retrieval": local,
        "hard_projection": hard,
        "deltas": {
            "pooled_r1": local[learned]["pooled_r1"] - local[raw]["pooled_r1"],
            "pooled_r5": local[learned]["pooled_r5"] - local[raw]["pooled_r5"],
            "projected_correct_per_board": hard[learned]["projected_correct_per_board"]
            - hard[raw]["projected_correct_per_board"],
            "top144_correct_per_board": top144_delta,
            "top144_precision": precision_delta,
        },
        "gate": {
            "ranking_arm": ranking_arm,
            "hard_edge_arm": hard_arm,
            "passed": passed,
            "decoder_ran": decoder is not None,
        },
        "decoder144_cyclic5_descriptive": decoder,
        "distinction_from_fullres_relation_fusion": config[
            "distinction_from_fullres_relation_fusion"
        ],
        "legality": {
            "organizer_train_only": True,
            "target_at_inference": False,
            "holdout_or_competition_test_opened": False,
            "pixel_replacement_or_generation": False,
            "decoder_layouts_strict_original_upright_permutations": decoder is None or True,
        },
    }
    _atomic_json(args.output_dir / "report.json", report)
    print(
        json.dumps({"event": "pilot_complete", "passed": passed, "deltas": report["deltas"]}),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    config = _load_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "capacity":
        run_capacity(args, config)
    elif args.stage == "benchmark":
        run_benchmark(args)
    else:
        run_pilot(args, config)


if __name__ == "__main__":
    main()

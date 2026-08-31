#!/usr/bin/env python3
"""Bounded train-only discovery for the dedicated full-resolution frame model."""

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

from aiijc_puzzle.component_relation_reranker import extract_frozen_socket_context
from aiijc_puzzle.frame_side_origin import (
    SIDES,
    FrameSideClassifier,
    FrameSideConfig,
    frame_side_loss,
    frame_side_targets,
    frame_topk_metrics,
    select_frame_cyclic_translation,
    top_frame_sets,
)
from aiijc_puzzle.fullres_boundary_denoiser import (
    FullResolutionBoundaryDenoiser,
    FullResolutionDenoiserConfig,
    restore_matcher_view,
)
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
DEFAULT_CONFIG = PROJECT_ROOT / "configs/frame_side_origin_preregistered_v1.json"
DEFAULT_MANIFEST = PROJECT_ROOT / "data/interim/validation_manifest.json"
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/frame-side-origin/v1-fit256-s600-eval32"
DEFAULT_SELECTION_OUTPUT = PROJECT_ROOT / "outputs/frame-side-origin/v1-selection"
SOCKET_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/socket-matcher/v2-d64-train1024-s1600-r400-dev32/socket_matcher.pt"
)
SOCKET_SHA256 = "0e9df49a503c65aac7f1468e9acd6a074a5e658ae8b61f8954be086272c49670"
DENOISER_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/fullres-boundary-denoiser/pilot-train32-s400-eval16-auto"
    / "fullres_boundary_denoiser.pt"
)
DENOISER_SHA256 = "a6dfc3e264e97d93ad678f3ee97e070067357c2a6f6875e7b7432f880aa1492c"
TILE_COUNT = GRID * GRID
FIT_SOURCES = 256
EVAL_SOURCES = 32
MAX_STEPS = 800
SEED = 20320914
ACTIVE_FIT_REGISTRY = (
    "outputs/whole-layout-cyclic-origin/v1-selection/selection_commitment.json",
    "outputs/sparse-bordergraph-qap/pilot-fit64-s240-eval16-top8-mps/selection_commitment.json",
    "configs/fullres_relation_fusion_preregistered_v1.json",
    "configs/fullres_relation_fusion_decoder_d2_preregistered_v1.json",
    "configs/component_relation_reranker_preregistered_v1.json",
    "configs/component_relation_confidence_preregistered_v1_1.json",
    "configs/component_relation_cyclic_fresh_gate_v1.json",
    "configs/border_pointer_preregistered_v1.json",
    "outputs/border-pointer/pilot-d64-train128-s400-exact16-mps/selection_commitment.json",
)


@dataclass(frozen=True)
class FrozenView:
    case_id: str
    source_filename: str
    raw_tiles: np.ndarray
    restored_tiles: np.ndarray
    socket_context: np.ndarray
    socket_border_logits: np.ndarray
    tile_to_position: np.ndarray


@dataclass(frozen=True)
class BlindEval:
    case_id: str
    source_filename: str
    candidate_sets: np.ndarray
    socket_sets: np.ndarray
    comparator_layout: np.ndarray
    candidate_layout: np.ndarray
    comparator_report: dict[str, Any]
    candidate_report: dict[str, Any]
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
    parser.add_argument("--inference-batch", type=int, default=576)
    return parser.parse_args()


def _device(name: str, *, allow_nondeterministic_mps: bool) -> torch.device:
    if name == "mps":
        if not torch.backends.mps.is_available() or not allow_nondeterministic_mps:
            raise ValueError(
                "MPS requires availability and explicit nondeterminism acknowledgement"
            )
        torch.use_deterministic_algorithms(False)
    elif allow_nondeterministic_mps:
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


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    expected = path.with_name(f"{path.name}.sha256").read_text().split()[0]
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError("frame-side preregistration hash mismatch")
    config = _load_json(path)
    if config.get("experiment") != "dedicated-frame-side-origin-v1":
        raise ValueError("unexpected frame-side experiment")
    if not config.get("registered_before_evaluation_target_access"):
        raise ValueError("frame-side preregistration timing contract is absent")
    if int(config["training"]["steps"]) > MAX_STEPS:
        raise ValueError("frame-side training exceeds bounded step cap")
    return config, observed


def _manifest_lookup(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if manifest.get("protocol_digest") != compute_protocol_digest(dict(manifest)):
        raise ValueError("manifest protocol digest is invalid")
    rows = manifest.get("splits", {}).get("train")
    if not isinstance(rows, list):
        raise ValueError("manifest train split is absent")
    return {str(row["filename"]): row for row in rows}


def _selection_output(args: argparse.Namespace) -> Path:
    return (args.output_dir or DEFAULT_SELECTION_OUTPUT).resolve()


def freeze_selection(args: argparse.Namespace, device: torch.device) -> None:
    output = _selection_output(args)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "selection_commitment.json"
    if path.exists():
        raise FileExistsError("refusing to overwrite frame-side selection commitment")
    manifest = _load_json(args.manifest)
    _manifest_lookup(manifest)
    excluded, panel_audit = _panel_json_exclusion()
    registry: list[dict[str, Any]] = []
    for raw_path in ACTIVE_FIT_REGISTRY:
        source = PROJECT_ROOT / raw_path
        if not source.exists():
            continue
        payload = _load_json(source)
        names = collect_declared_filenames(payload)
        excluded.update(names)
        registry.append(
            {
                "path": raw_path,
                "sha256": sha256_file(source),
                "actual_roster_count": len(names),
                "actual_roster_digest": names_digest(sorted(names), sort_names=True),
            }
        )
    if sha256_file(SOCKET_CHECKPOINT) != SOCKET_SHA256:
        raise ValueError("Socket checkpoint changed before frame selection")
    socket = load_socket_checkpoint(SOCKET_CHECKPOINT, device=device)
    excluded.update(socket.lineage.exposed_filenames)
    excluded_digest = names_digest(sorted(excluded), sort_names=True)
    namespace = f"aiijc-dedicated-frame-side-origin-v1:{excluded_digest}"
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
        raise RuntimeError("frame-side selection is not source-disjoint")
    commitment = {
        "schema": "aiijc-dedicated-frame-side-selection-v1",
        "written_before_any_selected_target_access": True,
        "manifest_split": "train",
        "namespace": namespace,
        "seed": SEED,
        "exclusion": {
            "panel_json_audit": panel_audit,
            "active_fit_registry": registry,
            "socket_lineage_count": socket.lineage.exposed_count,
            "socket_lineage_digest": socket.lineage.exposed_digest,
            "union_count": len(excluded),
            "union_digest": excluded_digest,
        },
        "fit_filenames": fit,
        "fit_order_digest": names_digest(fit),
        "fit_set_digest": names_digest(fit, sort_names=True),
        "evaluation_filenames": evaluation,
        "evaluation_order_digest": names_digest(evaluation),
        "evaluation_set_digest": names_digest(evaluation, sort_names=True),
        "fit_evaluation_overlap": 0,
        "excluded_overlap": 0,
        "calibration_opened": False,
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


def _load_rosters(
    config: Mapping[str, Any], manifest: Mapping[str, Any]
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], dict[str, Any], str]:
    contract = config["selection_commitment"]
    path = PROJECT_ROOT / str(contract["path"])
    observed = sha256_file(path)
    if observed != contract["sha256"]:
        raise ValueError("frame-side selection commitment changed")
    selection = _load_json(path)
    if not selection["written_before_any_selected_target_access"]:
        raise ValueError("selection was not committed before access")
    fit = list(selection["fit_filenames"])
    evaluation = list(selection["evaluation_filenames"])
    if (
        len(fit) != FIT_SOURCES
        or len(evaluation) != EVAL_SOURCES
        or len(fit) != int(contract["fit_sources"])
        or len(evaluation) != int(contract["evaluation_sources"])
    ):
        raise ValueError("frame-side roster sizes changed")
    if names_digest(fit) != selection["fit_order_digest"] or names_digest(
        evaluation
    ) != selection["evaluation_order_digest"]:
        raise ValueError("frame-side roster digest mismatch")
    if (
        selection["fit_order_digest"] != contract["fit_order_digest"]
        or selection["evaluation_order_digest"] != contract["evaluation_order_digest"]
    ):
        raise ValueError("frame-side roster no longer matches preregistration")
    lookup = _manifest_lookup(manifest)
    return (
        [lookup[name] for name in fit],
        [lookup[name] for name in evaluation],
        selection,
        observed,
    )


def _load_models(device: torch.device) -> tuple[Any, FullResolutionBoundaryDenoiser]:
    if sha256_file(SOCKET_CHECKPOINT) != SOCKET_SHA256:
        raise ValueError("Socket checkpoint SHA mismatch")
    if sha256_file(DENOISER_CHECKPOINT) != DENOISER_SHA256:
        raise ValueError("fullres denoiser SHA mismatch")
    socket = load_socket_checkpoint(SOCKET_CHECKPOINT, device=device)
    payload = torch.load(DENOISER_CHECKPOINT, map_location="cpu", weights_only=True)
    contract = payload["contract"]
    denoiser = FullResolutionBoundaryDenoiser(
        FullResolutionDenoiserConfig(**contract["model_config"])
    )
    denoiser.load_state_dict(payload["state_dict"], strict=True)
    denoiser.to(device).eval().requires_grad_(False)
    return socket, denoiser


def _numpy(value: Any, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    item = value.detach().cpu().numpy() if hasattr(value, "detach") else value
    result = np.asarray(item, dtype=np.float32)
    if result.ndim == len(shape) + 1 and result.shape[0] == 1:
        result = result[0]
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must have finite shape {shape}")
    return np.ascontiguousarray(result)


def _socket_borders(output: Any) -> np.ndarray:
    return np.column_stack(
        (
            _numpy(output.top_in_border_logits, shape=(TILE_COUNT,), name="top"),
            _numpy(output.bottom_out_border_logits, shape=(TILE_COUNT,), name="bottom"),
            _numpy(output.left_in_border_logits, shape=(TILE_COUNT,), name="left"),
            _numpy(output.right_out_border_logits, shape=(TILE_COUNT,), name="right"),
        )
    ).astype(np.float32)


@torch.inference_mode()
def prepare_view(
    cache: CleanTileCache,
    record: Mapping[str, Any],
    *,
    draw_index: int,
    socket: Any,
    denoiser: FullResolutionBoundaryDenoiser,
    device: torch.device,
    inference_batch: int,
) -> FrozenView:
    case = prepare_case(cache, record, draw_index=draw_index, seed=SEED)
    tokens, output = extract_frozen_socket_context(
        socket.model, _tile_tensor(case.dirty_tiles, device=device), grid=GRID
    )
    restored = restore_matcher_view(
        denoiser, case.dirty_tiles, device=device, batch_size=inference_batch
    )
    return FrozenView(
        case_id=case.case_id,
        source_filename=case.source_filename,
        raw_tiles=np.ascontiguousarray(case.dirty_tiles),
        restored_tiles=restored,
        socket_context=np.ascontiguousarray(tokens[0].float().cpu().numpy(), dtype=np.float16),
        socket_border_logits=np.ascontiguousarray(_socket_borders(output), dtype=np.float16),
        tile_to_position=np.ascontiguousarray(case.input_tile_to_position, dtype=np.int16),
    )


def _view_tensors(views: Sequence[FrozenView], device: torch.device) -> tuple[torch.Tensor, ...]:
    raw = torch.from_numpy(np.stack([view.raw_tiles for view in views])).permute(0, 1, 4, 2, 3)
    restored = torch.from_numpy(np.stack([view.restored_tiles for view in views])).permute(
        0, 1, 4, 2, 3
    )
    context = torch.from_numpy(np.stack([view.socket_context for view in views]))
    border = torch.from_numpy(np.stack([view.socket_border_logits for view in views]))
    positions = torch.from_numpy(
        np.stack([view.tile_to_position.astype(np.int64) for view in views])
    )
    return (
        raw.to(device=device, dtype=torch.float32).div_(255.0),
        restored.to(device=device, dtype=torch.float32).div_(255.0),
        context.to(device=device, dtype=torch.float32),
        border.to(device=device, dtype=torch.float32),
        positions.to(device=device),
    )


def _aligned_logits(logits: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    reference = torch.argsort(positions, dim=1)
    return torch.gather(logits, 1, reference[..., None].expand(-1, -1, len(SIDES)))


def run_capacity(args: argparse.Namespace, device: torch.device) -> None:
    torch.manual_seed(SEED)
    grid = 8
    count = grid * grid
    model = FrameSideClassifier(FrameSideConfig(width=24, blocks=3)).to(device)
    positions = torch.arange(count, device=device).repeat(2, 1)
    targets = frame_side_targets(positions, grid=grid).to(device)
    raw = torch.rand(2, count, 3, 20, 20, device=device) * 0.15
    # Direction-specific full-resolution edge patterns; corners carry both labels.
    raw[:, targets[0, :, 0], 0, :5, :] += 0.75
    raw[:, targets[0, :, 1], 1, -5:, :] += 0.75
    raw[:, targets[0, :, 2], 2, :, :5] += 0.75
    raw[:, targets[0, :, 3], :, :, -5:] += 0.75
    raw.clamp_(0, 1)
    restored = raw.clone()
    context = torch.zeros(2, count, 64, device=device)
    border = torch.zeros(2, count, 4, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    history: list[float] = []
    for _ in range(80):
        logits = model(raw, restored, context, border)
        loss, _ = frame_side_loss(logits, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    sets = top_frame_sets(model(raw[:1], restored[:1], context[:1], border[:1])[0], grid=grid)
    metrics = frame_topk_metrics(sets, positions[0].cpu().numpy(), grid=grid)
    listwise_lower_bound = float(np.log(grid))
    excess_loss = history[-1] - listwise_lower_bound
    result = {
        "experiment": "frame-side-synthetic-capacity-v1",
        "device": str(device),
        "initial_loss": history[0],
        "final_loss": history[-1],
        "listwise_lower_bound": listwise_lower_bound,
        "excess_loss": excess_loss,
        "macro_f1": metrics["macro_f1"],
        "pass": excess_loss <= 0.05 and metrics["macro_f1"] >= 0.95,
    }
    output = (args.output_dir or PROJECT_ROOT / "outputs/frame-side-origin/capacity").resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "report.json", result)
    print(json.dumps({"event": "capacity", **result}), flush=True)


def _strict(layout: Any) -> np.ndarray:
    value = np.asarray(layout, dtype=np.int32)
    if value.shape != (TILE_COUNT,) or not np.array_equal(np.sort(value), np.arange(TILE_COUNT)):
        raise ValueError("layout is not a strict original-tile permutation")
    return np.ascontiguousarray(value)


def _side_summary(rows: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    return {
        "macro_f1": float(np.mean([row[arm]["macro_f1"] for row in rows])),
        "per_side_f1": {
            side: float(np.mean([row[arm]["sides"][side]["f1"] for row in rows]))
            for side in SIDES
        },
    }


def _layout_summary(rows: Sequence[Mapping[str, Any]], arm: str) -> dict[str, float]:
    return {
        field: float(np.mean([row[arm]["metrics"][field] for row in rows]))
        for field in (
            "correct_tile_count",
            "direct_placement",
            "translation_aligned_count",
            "translation_aligned_placement",
            "adjacency_correct",
            "adjacency",
        )
    }


def _gate(summary: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    gate = config["discovery_gate"]
    tolerance = 1e-12
    f1_gain = float(summary["frame_candidate"]["macro_f1"]) - float(
        summary["frame_socket_v2"]["macro_f1"]
    )
    exact_gain = float(summary["candidate_layout"]["correct_tile_count"]) - float(
        summary["comparator_layout"]["correct_tile_count"]
    )
    adjacency_delta = float(summary["candidate_layout"]["adjacency"]) - float(
        summary["comparator_layout"]["adjacency"]
    )
    f1_branch = f1_gain + tolerance >= float(gate["minimum_macro_f1_gain"])
    exact_branch = exact_gain + tolerance >= float(
        gate["minimum_exact_tiles_gain_per_board"]
    )
    adjacency_ok = adjacency_delta + tolerance >= -float(
        gate["maximum_adjacency_loss_fraction"]
    )
    strict = int(summary["strict_permutation_count"]) == int(
        gate["strict_original_permutations_required"]
    )
    passed = strict and adjacency_ok and (f1_branch or exact_branch)
    return {
        "status": "pass-preserve-for-fresh64" if passed else "fail-stop",
        "pass": passed,
        "fresh64_authorized": passed,
        "promotion_authorized": False,
        "competition_test_authorized": False,
        "observed": {
            "macro_f1_gain": f1_gain,
            "exact_tiles_gain_per_board": exact_gain,
            "adjacency_delta": adjacency_delta,
        },
        "checks": {
            "macro_f1_branch": f1_branch,
            "exact_branch": exact_branch,
            "adjacency_safety": adjacency_ok,
            "strict_permutations": strict,
        },
    }


def run_pilot(args: argparse.Namespace, device: torch.device) -> None:
    config, config_sha256 = _load_config(args.config)
    output = (args.output_dir or DEFAULT_OUTPUT).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "frame_side_origin.pt"
    frozen_path = output / "frozen_predictions.json"
    report_path = output / "report.json"
    if checkpoint_path.exists() or frozen_path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite frame-side pilot artifacts")
    manifest = _load_json(args.manifest)
    fit_records, eval_records, selection, selection_sha256 = _load_rosters(config, manifest)
    socket, denoiser = _load_models(device)
    architecture = FrameSideConfig(**config["architecture"])
    model = FrameSideClassifier(architecture).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != int(config["expected_parameter_count"]):
        raise ValueError(f"frame-side parameter count changed: {parameter_count}")
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
    cache = CleanTileCache(args.targets)
    fit_pairs: list[tuple[FrozenView, FrozenView]] = []
    precompute_started = perf_counter()
    for index, record in enumerate(fit_records):
        pair = tuple(
            prepare_view(
                cache,
                record,
                draw_index=draw,
                socket=socket,
                denoiser=denoiser,
                device=device,
                inference_batch=args.inference_batch,
            )
            for draw in (0, 1)
        )
        fit_pairs.append((pair[0], pair[1]))
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
    model.train()
    train_started = perf_counter()
    for step in range(int(training["steps"])):
        pair = fit_pairs[int(generator.integers(len(fit_pairs)))]
        raw, restored, context, border, positions = _view_tensors(pair, device)
        logits = model(raw, restored, context, border)
        targets = frame_side_targets(positions, grid=GRID)
        aligned = _aligned_logits(logits, positions)
        loss, diagnostics = frame_side_loss(
            logits,
            targets,
            consistency_logits=(aligned[:1], aligned[1:]),
            consistency_weight=float(training["consistency_weight"]),
            bce_weight=float(training["bce_weight"]),
        )
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
    checkpoint = {
        "state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "contract": {
            "architecture": "fullres-frame-side-classifier-v1",
            "model_config": config["architecture"],
            "parameter_count": parameter_count,
            "side_order": list(SIDES),
            "top_cardinality_per_side": GRID,
            "no_downsample": True,
            "restored_pixels_matcher_only": True,
            "original_upright_tiles_only": True,
        },
        "selection": {
            "fit_filenames": selection["fit_filenames"],
            "fit_order_digest": selection["fit_order_digest"],
            "evaluation_filenames": selection["evaluation_filenames"],
            "evaluation_order_digest": selection["evaluation_order_digest"],
        },
        "preregistration": {"path": str(args.config), "sha256": config_sha256},
    }
    torch.save(checkpoint, checkpoint_path)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    before_eval = {
        "event": "evaluation-preregistered-before-target-access",
        "pid": os.getpid(),
        "config_sha256": config_sha256,
        "selection_sha256": selection_sha256,
        "fit_digest": selection["fit_order_digest"],
        "evaluation_digest": selection["evaluation_order_digest"],
        "checkpoint_sha256": checkpoint_sha256,
        "evaluation_sources": len(eval_records),
        "target_access_started": False,
    }
    _write_json(output / "eval_start.json", before_eval)
    print(json.dumps(before_eval), flush=True)
    if args.wait_for_eval_confirmation:
        input("Evaluation paused; press Enter after publishing PID/hashes.\n")
    before_eval["target_access_started"] = True
    _write_json(output / "eval_start.json", before_eval)
    print(json.dumps({"event": "evaluation-target-access-open", "pid": os.getpid()}), flush=True)

    blind_rows: list[BlindEval] = []
    eval_started = perf_counter()
    with torch.inference_mode():
        for index, record in enumerate(eval_records):
            case = prepare_case(
                cache,
                record,
                draw_index=int(config["evaluation"]["draw_index"]),
                seed=int(config["evaluation"]["seed"]),
            )
            tokens, socket_output = extract_frozen_socket_context(
                socket.model, _tile_tensor(case.dirty_tiles, device=device), grid=GRID
            )
            restored_tiles = restore_matcher_view(
                denoiser,
                case.dirty_tiles,
                device=device,
                batch_size=args.inference_batch,
            )
            raw_tensor = _tile_tensor(case.dirty_tiles, device=device)
            restored_tensor = _tile_tensor(restored_tiles, device=device)
            logits = model(
                raw_tensor,
                restored_tensor,
                tokens,
                torch.from_numpy(_socket_borders(socket_output))
                .unsqueeze(0)
                .to(device),
            )[0]
            candidate_sets = top_frame_sets(logits, grid=GRID)
            socket_sets = top_frame_sets(_socket_borders(socket_output), grid=GRID)
            decoder = decode_socket_assignments(
                socket_output.right_log_assignment,
                socket_output.down_log_assignment,
                grid=GRID,
                config=SocketDecoderConfig(
                    component_edge_budget_per_axis=144,
                    max_swap_steps=24,
                ),
            )
            comparator = select_global_cyclic_translation(
                decoder.layout,
                socket_output.right_log_assignment,
                socket_output.down_log_assignment,
                grid=GRID,
                config=CyclicTranslationConfig(border_weight=5.0),
            )
            candidate = select_frame_cyclic_translation(
                decoder.layout,
                candidate_sets,
                socket_output.right_log_assignment,
                socket_output.down_log_assignment,
                grid=GRID,
            )
            blind_rows.append(
                BlindEval(
                    case_id=case.case_id,
                    source_filename=case.source_filename,
                    candidate_sets=candidate_sets,
                    socket_sets=socket_sets,
                    comparator_layout=_strict(comparator.layout),
                    candidate_layout=_strict(candidate.layout),
                    comparator_report=comparator.report(),
                    candidate_report=candidate.report(),
                    dirty_tiles_sha256=hashlib.sha256(case.dirty_tiles.tobytes()).hexdigest(),
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
        "phase": "dirty-frame-sets-and-layouts-frozen-before-exact-scoring",
        "config_sha256": config_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "evaluation_digest": selection["evaluation_order_digest"],
        "strict_permutation_count": sum(
            int(
                np.array_equal(np.sort(row.comparator_layout), np.arange(TILE_COUNT))
                and np.array_equal(np.sort(row.candidate_layout), np.arange(TILE_COUNT))
            )
            for row in blind_rows
        ),
        "rows": [
            {
                "case_id": row.case_id,
                "source_filename": row.source_filename,
                "dirty_tiles_sha256": row.dirty_tiles_sha256,
                "socket_top24_sets": row.socket_sets.tolist(),
                "candidate_top24_sets": row.candidate_sets.tolist(),
                "comparator_tile_at_position": row.comparator_layout.tolist(),
                "candidate_tile_at_position": row.candidate_layout.tolist(),
                "comparator": row.comparator_report,
                "candidate": row.candidate_report,
            }
            for row in blind_rows
        ],
    }
    _write_json(frozen_path, frozen_payload)
    frozen_sha256 = sha256_file(frozen_path)
    print(json.dumps({"event": "predictions-frozen", "sha256": frozen_sha256}), flush=True)

    # Recreate deterministic cases only after inference outputs are immutable.
    # This makes the target-label/scoring boundary explicit in artifact order.
    eval_truth: list[np.ndarray] = []
    for record, blind in zip(eval_records, blind_rows, strict=True):
        case = prepare_case(
            cache,
            record,
            draw_index=int(config["evaluation"]["draw_index"]),
            seed=int(config["evaluation"]["seed"]),
        )
        if case.case_id != blind.case_id:
            raise RuntimeError("deterministic evaluation case changed after freeze")
        eval_truth.append(
            np.ascontiguousarray(case.input_tile_to_position, dtype=np.int32)
        )

    scored: list[dict[str, Any]] = []
    for row, positions in zip(blind_rows, eval_truth, strict=True):
        reference = np.argsort(positions).astype(np.int32)
        socket_metrics = frame_topk_metrics(row.socket_sets, positions, grid=GRID)
        candidate_metrics = frame_topk_metrics(row.candidate_sets, positions, grid=GRID)
        comparator_metrics = evaluate_layout(
            row.comparator_layout, reference, reference_is_exact=True
        ).as_dict()
        layout_metrics = evaluate_layout(
            row.candidate_layout, reference, reference_is_exact=True
        ).as_dict()
        scored.append(
            {
                "case_id": row.case_id,
                "source_filename": row.source_filename,
                "socket_v2_frame": socket_metrics,
                "candidate_frame": candidate_metrics,
                "comparator_layout": {"metrics": comparator_metrics},
                "candidate_layout": {"metrics": layout_metrics},
                "exact_delta_tiles": layout_metrics["correct_tile_count"]
                - comparator_metrics["correct_tile_count"],
                "adjacency_delta": layout_metrics["adjacency"]
                - comparator_metrics["adjacency"],
            }
        )
    summary = {
        "source_count": len(scored),
        "frame_socket_v2": _side_summary(scored, "socket_v2_frame"),
        "frame_candidate": _side_summary(scored, "candidate_frame"),
        "comparator_layout": _layout_summary(scored, "comparator_layout"),
        "candidate_layout": _layout_summary(scored, "candidate_layout"),
        "strict_permutation_count": frozen_payload["strict_permutation_count"],
        "exact_wins_ties_losses": [
            sum(row["exact_delta_tiles"] > 0 for row in scored),
            sum(row["exact_delta_tiles"] == 0 for row in scored),
            sum(row["exact_delta_tiles"] < 0 for row in scored),
        ],
    }
    gate = _gate(summary, config)
    report = {
        "experiment": config["experiment"],
        "status": gate["status"],
        "preregistration": {"path": str(args.config.resolve()), "sha256": config_sha256},
        "selection": {
            "commitment_sha256": selection_sha256,
            "fit_digest": selection["fit_order_digest"],
            "evaluation_digest": selection["evaluation_order_digest"],
            "fit_evaluation_overlap": selection["fit_evaluation_overlap"],
        },
        "frozen_inputs": {
            "socket_checkpoint": str(SOCKET_CHECKPOINT),
            "socket_sha256": SOCKET_SHA256,
            "fullres_denoiser_checkpoint": str(DENOISER_CHECKPOINT),
            "fullres_denoiser_sha256": DENOISER_SHA256,
        },
        "model": {
            "config": config["architecture"],
            "parameter_count": parameter_count,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
        },
        "training": {
            "configuration": training,
            "history": history,
            "fit_precompute_seconds": train_started - precompute_started,
            "training_seconds": training_seconds,
        },
        "freeze": {
            "path": str(frozen_path),
            "sha256": frozen_sha256,
            "before_exact_scoring": True,
        },
        "summary": summary,
        "gate": gate,
        "fresh64_opened": False,
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
    if args.log_every <= 0 or args.inference_batch <= 0:
        raise ValueError("log-every and inference-batch must be positive")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = _device(args.device, allow_nondeterministic_mps=args.allow_nondeterministic_mps)
    if args.mode == "capacity":
        run_capacity(args, device)
    elif args.mode == "selection":
        freeze_selection(args, device)
    else:
        run_pilot(args, device)


if __name__ == "__main__":
    main()

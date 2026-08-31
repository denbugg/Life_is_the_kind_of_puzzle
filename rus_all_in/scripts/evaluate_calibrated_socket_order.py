#!/usr/bin/env python3
"""Evaluate frozen calibrated ordering inside the ordinary decoder144.

The SocketMatcher-v2 checkpoint, learned edge calibrator, component budget, border packing,
full soft objective and bounded QAP polish are fixed.  The sole candidate
change is the dirty-visible probability used to select and greedily order 144
hard component constraints per axis.  Exact synthetic labels are consumed only
after both layouts and component traces have been frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from PIL import Image
from scipy import stats

from aiijc_puzzle.calibrated_socket_order import (
    ComponentBuildTrace,
    build_component_trace,
    calibrated_priority_matrices,
    edge_set_overlap,
    exact_component_metrics,
)
from aiijc_puzzle.layout_evaluation import evaluate_layout
from aiijc_puzzle.protocol import (
    IMAGE_SIZE,
    assemble_tiles,
    compute_protocol_digest,
    contest_ssim,
    sha256_file,
    split_tiles,
)
from aiijc_puzzle.socket_confidence_calibration import (
    FEATURE_NAMES,
    extract_hard_edge_features,
    frozen_linear_calibrator_from_payload,
)
from aiijc_puzzle.socket_decoder import (
    SocketDecoderConfig,
    SocketDecodeResult,
    decode_socket_assignments,
)
from aiijc_puzzle.socket_matcher import BORDER_HEAD_EMBEDDING_V2, SocketMatcher
from aiijc_puzzle.synthetic_socket_evaluation import (
    ExactSyntheticReference,
    SyntheticSocketInput,
    load_checkpoint_with_lineage,
    make_exact_synthetic_case,
    names_digest,
    select_source_disjoint_train_records,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/socket-matcher/v2-border-train512-s300-r100-dev24/socket_matcher.pt"
)
DEFAULT_CALIBRATOR = (
    PROJECT_ROOT
    / "outputs/socket-confidence-calibration/d32-v2-fit32-confirm16/frozen_calibrator.json"
)
EXPECTED_CALIBRATOR_SHA256 = (
    "a5577a22c96c76e44e2f7735e3912772f182de5c887edba4b806aee1a4c515a5"
)
DEFAULT_MANIFEST = PROJECT_ROOT / "data/interim/validation_manifest.json"
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
DEFAULT_PRIOR_REPORT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs/socket-confidence-calibration/calibrated-order-decoder144-exact24"
)
GRID = 24
TILE_COUNT = GRID * GRID
HARD_EDGES_PER_AXIS = GRID * (GRID - 1)
HARD_EDGES_PER_BOARD = 2 * HARD_EDGES_PER_AXIS
COMPONENT_BUDGET = 144
NAMESPACE = "aiijc-calibrated-socket-order-exact-v1"
STATUS_TO_CODE = {
    "added": 0,
    "consistent": 1,
    "contradiction": 2,
    "collision": 3,
    "span": 4,
}


@dataclass(frozen=True)
class FrozenCase:
    """Dirty-only predictions plus references kept private until freeze."""

    synthetic_input: SyntheticSocketInput
    reference: ExactSyntheticReference
    clean_image: np.ndarray
    target_sha256: str
    base: SocketDecodeResult
    calibrated: SocketDecodeResult
    base_trace: ComponentBuildTrace
    calibrated_trace: ComponentBuildTrace
    hard_source: np.ndarray
    hard_target: np.ndarray
    hard_axis: np.ndarray
    hard_probability: np.ndarray
    threshold_selected: np.ndarray
    overlap: dict[str, float | int]
    runtime_seconds: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--calibrator", type=Path, default=DEFAULT_CALIBRATOR)
    parser.add_argument(
        "--expected-calibrator-sha256",
        default=EXPECTED_CALIBRATOR_SHA256,
        help="Required hash lock for the frozen calibrator JSON.",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--prior-report-root", type=Path, default=DEFAULT_PRIOR_REPORT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-limit", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--torch-threads", type=int, default=1)
    return parser.parse_args()


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        if image.mode != "RGB" or image.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"expected RGB 480x480 image: {path}")
        return np.asarray(image, dtype=np.uint8)


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def load_v2_model(payload: dict[str, Any]) -> tuple[SocketMatcher, dict[str, Any]]:
    contract = payload.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("checkpoint has no architecture contract")
    if contract.get("architecture") != "board-conditioned-partial-socket-matcher-v2":
        raise ValueError("this experiment requires SocketMatcher v2")
    fields: dict[str, Any] = {"border_head_version": BORDER_HEAD_EMBEDDING_V2}
    for key in ("dimension", "heads", "board_layers", "socket_layers", "sinkhorn_iterations"):
        value = contract.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"checkpoint contract {key} must be a positive integer")
        fields[key] = value
    model = SocketMatcher(**fields)
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint has no state_dict")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, contract


def prior_exact_sources(report_root: Path, *, ignore: Path) -> tuple[set[str], list[str]]:
    """Fail closed over every previous exact-synthetic source roster."""

    sources: set[str] = set()
    reports: list[str] = []
    for path in sorted(report_root.rglob("report.json")):
        if path.resolve() == ignore.resolve():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        protocol = payload.get("protocol")
        exact = bool(
            payload.get("experiment")
            in {
                "socket-matcher-source-disjoint-exact-synthetic-v1",
                "socket-hard-edge-confidence-calibration-v1",
                "calibrated-socket-order-decoder144-v1",
            }
            or (
                isinstance(protocol, dict)
                and (
                    protocol.get("exact_synthetic_labels_only") is True
                    or str(protocol.get("permutation_labels", "")).startswith("exact inverse")
                )
            )
        )
        if not exact:
            continue
        selection = payload.get("selection")
        if not isinstance(selection, dict):
            raise ValueError(f"exact report has no selection mapping: {path}")
        rosters: list[list[str]] = []
        for key in (
            "source_filenames",
            "fit_source_filenames",
            "confirm_source_filenames",
        ):
            value = selection.get(key)
            if value is not None:
                if not isinstance(value, list) or not all(isinstance(name, str) for name in value):
                    raise ValueError(f"invalid source roster in {path}: {key}")
                rosters.append(value)
        if not rosters:
            raise ValueError(f"exact report has no recognized source roster: {path}")
        for roster in rosters:
            sources.update(roster)
        reports.append(str(path.resolve()))
    return sources, reports


def _trace_arrays(trace: ComponentBuildTrace) -> dict[str, np.ndarray]:
    return {
        "source": np.asarray([value.edge.source for value in trace.constraints], dtype=np.int16),
        "target": np.asarray([value.edge.target for value in trace.constraints], dtype=np.int16),
        "axis": np.asarray(
            [int(value.edge.axis == "down") for value in trace.constraints],
            dtype=np.int8,
        ),
        "status": np.asarray(
            [STATUS_TO_CODE[value.status] for value in trace.constraints],
            dtype=np.int8,
        ),
    }


def _validate_trace(result: SocketDecodeResult, trace: ComponentBuildTrace) -> None:
    diagnostics = result.diagnostics
    if diagnostics.attempted_constraints != len(trace.constraints):
        raise RuntimeError("component trace constraint count differs from decoder")
    if diagnostics.added_constraints != trace.status_counts["added"]:
        raise RuntimeError("component trace added count differs from decoder")
    if diagnostics.consistent_redundant_constraints != trace.status_counts["consistent"]:
        raise RuntimeError("component trace consistent count differs from decoder")
    if diagnostics.contradiction_rejections != trace.status_counts["contradiction"]:
        raise RuntimeError("component trace contradiction count differs from decoder")
    if diagnostics.collision_rejections != trace.status_counts["collision"]:
        raise RuntimeError("component trace collision count differs from decoder")
    if diagnostics.span_rejections != trace.status_counts["span"]:
        raise RuntimeError("component trace span count differs from decoder")
    sizes = tuple(sorted((len(value) for value in trace.components), reverse=True))
    if sizes != diagnostics.component_sizes:
        raise RuntimeError("component trace sizes differ from decoder")


@torch.inference_mode()
def freeze_case(
    model: SocketMatcher,
    calibrator: Any,
    record: Any,
    *,
    targets_dir: Path,
    seed: int,
) -> FrozenCase:
    filename = str(record["filename"])
    target_path = targets_dir / filename
    target_sha = sha256_file(target_path)
    if target_sha != record.get("target_sha256"):
        raise ValueError(f"manifest target hash mismatch for {filename}")
    clean_image = load_rgb(target_path)
    clean_tiles = split_tiles(clean_image)
    synthetic_input, reference = make_exact_synthetic_case(
        clean_tiles,
        source_filename=filename,
        draw_index=0,
        seed=seed,
    )
    dirty = synthetic_input.tiles
    tensor = torch.from_numpy(dirty.astype(np.float32)).permute(0, 3, 1, 2) / 255.0
    started = perf_counter()
    output = model(tensor.unsqueeze(0), grid=GRID)
    matcher_seconds = perf_counter() - started
    right = output.right_log_assignment[0].float().cpu().numpy()
    down = output.down_log_assignment[0].float().cpu().numpy()
    features = extract_hard_edge_features(
        right_log_assignment=right,
        down_log_assignment=down,
        right_raw=output.right_raw[0],
        down_raw=output.down_raw[0],
        grid=GRID,
    )
    probability = calibrator.predict_probability(features.values)
    priorities = calibrated_priority_matrices(features, probability, grid=GRID)
    config = SocketDecoderConfig(
        component_edge_budget_per_axis=COMPONENT_BUDGET,
        swap_edge_budget_per_axis=COMPONENT_BUDGET,
        max_swap_steps=24,
    )

    started = perf_counter()
    base = decode_socket_assignments(right, down, grid=GRID, config=config)
    base_seconds = perf_counter() - started
    started = perf_counter()
    calibrated = decode_socket_assignments(
        right,
        down,
        grid=GRID,
        config=config,
        component_edge_priority=priorities,
    )
    calibrated_seconds = perf_counter() - started
    base_trace = build_component_trace(
        right,
        down,
        grid=GRID,
        edge_budget_per_axis=COMPONENT_BUDGET,
    )
    calibrated_trace = build_component_trace(
        right,
        down,
        grid=GRID,
        edge_budget_per_axis=COMPONENT_BUDGET,
        component_edge_priority=priorities,
    )
    _validate_trace(base, base_trace)
    _validate_trace(calibrated, calibrated_trace)
    if base.diagnostics.component_edge_priority_used:
        raise RuntimeError("base decoder unexpectedly used external priority")
    if not calibrated.diagnostics.component_edge_priority_used:
        raise RuntimeError("calibrated decoder did not report external priority")

    return FrozenCase(
        synthetic_input=synthetic_input,
        reference=reference,
        clean_image=clean_image,
        target_sha256=target_sha,
        base=base,
        calibrated=calibrated,
        base_trace=base_trace,
        calibrated_trace=calibrated_trace,
        hard_source=features.source,
        hard_target=features.target,
        hard_axis=features.axis,
        hard_probability=probability,
        threshold_selected=probability >= calibrator.threshold,
        overlap=edge_set_overlap(base_trace, calibrated_trace),
        runtime_seconds={
            "matcher": matcher_seconds,
            "base_decoder": base_seconds,
            "calibrated_decoder": calibrated_seconds,
        },
    )


def write_frozen_artifact(
    cases: list[FrozenCase],
    *,
    output_dir: Path,
) -> tuple[Path, str, Path, str]:
    arrays: dict[str, np.ndarray] = {
        "filenames": np.asarray([case.synthetic_input.source_filename for case in cases]),
        "base_layout": np.stack([case.base.layout.astype(np.int16) for case in cases]),
        "calibrated_layout": np.stack(
            [case.calibrated.layout.astype(np.int16) for case in cases]
        ),
        "hard_source": np.stack([case.hard_source.astype(np.int16) for case in cases]),
        "hard_target": np.stack([case.hard_target.astype(np.int16) for case in cases]),
        "hard_axis": np.stack([case.hard_axis for case in cases]),
        "hard_probability": np.stack(
            [case.hard_probability.astype(np.float32) for case in cases]
        ),
        "threshold_selected": np.stack([case.threshold_selected for case in cases]),
    }
    for variant in ("base", "calibrated"):
        traces = [getattr(case, f"{variant}_trace") for case in cases]
        trace_arrays = [_trace_arrays(trace) for trace in traces]
        for name in ("source", "target", "axis", "status"):
            arrays[f"{variant}_constraint_{name}"] = np.stack(
                [value[name] for value in trace_arrays]
            )
    artifact_path = output_dir / "frozen_predictions.npz"
    np.savez_compressed(artifact_path, **arrays)
    metadata_path = output_dir / "frozen_predictions.json"
    metadata_path.write_text(
        json.dumps(
            {
                "schema": "aiijc-calibrated-socket-order-frozen-v1",
                "contains_exact_references": False,
                "contains_labels": False,
                "contains_clean_pixels": False,
                "component_budget_per_axis": COMPONENT_BUDGET,
                "feature_names": list(FEATURE_NAMES),
                "source_filenames": [case.synthetic_input.source_filename for case in cases],
                "dirty_tiles_sha256": [
                    _array_sha256(case.synthetic_input.tiles) for case in cases
                ],
                "target_file_sha256": [case.target_sha256 for case in cases],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return (
        artifact_path,
        sha256_file(artifact_path),
        metadata_path,
        sha256_file(metadata_path),
    )


def _trace_key_set(trace: ComponentBuildTrace) -> set[tuple[int, int, int]]:
    return {
        (int(value.edge.axis == "down"), value.edge.source, value.edge.target)
        for value in trace.constraints
    }


def score_case(case: FrozenCase) -> dict[str, Any]:
    reference = case.reference.tile_at_position
    variants: dict[str, Any] = {}
    for name, result, trace in (
        ("base_decoder144", case.base, case.base_trace),
        ("calibrated_order_decoder144", case.calibrated, case.calibrated_trace),
    ):
        geometry = evaluate_layout(result.layout, reference, reference_is_exact=True).as_dict()
        prediction = assemble_tiles(case.synthetic_input.tiles[result.layout])
        variants[name] = geometry | {
            "raw_ssim": contest_ssim(case.clean_image, prediction),
            "components": exact_component_metrics(trace, reference, grid=GRID),
            "decoder": result.diagnostics.as_dict(),
        }

    threshold_keys = {
        (int(axis), int(source), int(target))
        for source, target, axis, selected in zip(
            case.hard_source,
            case.hard_target,
            case.hard_axis,
            case.threshold_selected,
            strict=True,
        )
        if selected
    }
    base_keys = _trace_key_set(case.base_trace)
    calibrated_keys = _trace_key_set(case.calibrated_trace)
    return {
        "case_id": case.synthetic_input.case_id,
        "source_filename": case.synthetic_input.source_filename,
        "variants": variants,
        "dirty_only_edge_overlap": case.overlap
        | {
            "calibrator_threshold_selected": len(threshold_keys),
            "threshold_selected_in_base_top144": len(threshold_keys & base_keys),
            "threshold_selected_in_calibrated_top144": len(
                threshold_keys & calibrated_keys
            ),
        },
        "runtime_seconds": case.runtime_seconds,
    }


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _paired_summary(candidate: list[float], base: list[float]) -> dict[str, Any]:
    delta = np.asarray(candidate, dtype=np.float64) - np.asarray(base, dtype=np.float64)
    mean = float(delta.mean())
    if len(delta) < 2 or float(delta.std(ddof=1)) == 0.0:
        lower = upper = mean
    else:
        radius = float(stats.t.ppf(0.975, len(delta) - 1) * stats.sem(delta))
        lower, upper = mean - radius, mean + radius
    return {
        "mean_delta": mean,
        "paired_t_95_ci": [lower, upper],
        "wins": int(np.count_nonzero(delta > 0)),
        "ties": int(np.count_nonzero(delta == 0)),
        "losses": int(np.count_nonzero(delta < 0)),
    }


def aggregate(boards: list[dict[str, Any]]) -> dict[str, Any]:
    variant_names = ("base_decoder144", "calibrated_order_decoder144")
    core = (
        "correct_tile_count",
        "translation_aligned_count",
        "adjacency_correct",
        "adjacency",
        "raw_ssim",
    )
    component = (
        "correct_selected_edges",
        "selected_edge_precision",
        "correct_added_constraints",
        "false_added_bridges",
        "added_constraint_precision",
        "largest_component",
        "largest_component_translation_purity",
        "tile_weighted_translation_purity",
        "pairwise_relative_accuracy",
        "fully_exact_component_tiles",
    )
    variants: dict[str, Any] = {}
    for variant in variant_names:
        variants[variant] = {
            **{
                key: _mean([float(board["variants"][variant][key]) for board in boards])
                for key in core
            },
            "components": {
                key: _mean(
                    [
                        float(board["variants"][variant]["components"][key])
                        for board in boards
                    ]
                )
                for key in component
            },
            "decoder": {
                key: _mean(
                    [
                        float(board["variants"][variant]["decoder"][key])
                        for board in boards
                    ]
                )
                for key in (
                    "added_constraints",
                    "contradiction_rejections",
                    "collision_rejections",
                    "span_rejections",
                    "largest_component",
                    "rigid_tiles_packed",
                    "accepted_swaps",
                    "objective_gain",
                )
            },
        }
    paired: dict[str, Any] = {}
    for key in core:
        paired[key] = _paired_summary(
            [float(board["variants"][variant_names[1]][key]) for board in boards],
            [float(board["variants"][variant_names[0]][key]) for board in boards],
        )
    for key in component:
        paired[f"components.{key}"] = _paired_summary(
            [
                float(board["variants"][variant_names[1]]["components"][key])
                for board in boards
            ],
            [
                float(board["variants"][variant_names[0]]["components"][key])
                for board in boards
            ],
        )
    overlap_keys = boards[0]["dirty_only_edge_overlap"].keys()
    overlap = {
        key: _mean([float(board["dirty_only_edge_overlap"][key]) for board in boards])
        for key in overlap_keys
    }
    return {"variants": variants, "paired_candidate_minus_base": paired, "overlap": overlap}


def main() -> None:
    args = parse_args()
    if args.source_limit <= 0:
        raise ValueError("source-limit must be positive")
    if args.torch_threads <= 0:
        raise ValueError("torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)

    output_dir = args.output_dir.resolve()
    report_path = output_dir / "report.json"
    if report_path.exists():
        raise FileExistsError(f"one-shot experiment report already exists: {report_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = args.checkpoint.resolve()
    checkpoint_sha = sha256_file(checkpoint_path)
    checkpoint_payload, lineage = load_checkpoint_with_lineage(
        checkpoint_path,
        project_root=PROJECT_ROOT,
    )
    model, contract = load_v2_model(checkpoint_payload)
    calibrator_path = args.calibrator.resolve()
    calibrator_sha = sha256_file(calibrator_path)
    if args.expected_calibrator_sha256 != calibrator_sha:
        raise ValueError("frozen calibrator hash differs from the confirmed artifact")
    calibrator_payload = json.loads(calibrator_path.read_text(encoding="utf-8"))
    calibrator = frozen_linear_calibrator_from_payload(calibrator_payload)
    declared_checkpoint = calibrator_payload.get("checkpoint")
    if not isinstance(declared_checkpoint, dict) or declared_checkpoint.get(
        "sha256"
    ) != checkpoint_sha:
        raise ValueError("calibrator lineage does not match the requested matcher checkpoint")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("protocol_digest") != compute_protocol_digest(manifest):
        raise ValueError("validation manifest protocol digest is invalid")
    prior_names, prior_reports = prior_exact_sources(
        args.prior_report_root.resolve(),
        ignore=report_path,
    )
    exclusions = set(lineage.filenames) | prior_names
    records = select_source_disjoint_train_records(
        manifest,
        excluded_filenames=tuple(sorted(exclusions)),
        limit=args.source_limit,
        seed=args.seed,
        namespace=NAMESPACE,
    )
    names = tuple(str(record["filename"]) for record in records)
    if set(names) & exclusions:
        raise RuntimeError("selected panel overlaps checkpoint or exact-synthetic lineage")

    cases: list[FrozenCase] = []
    for index, record in enumerate(records, start=1):
        case = freeze_case(
            model,
            calibrator,
            record,
            targets_dir=args.targets.resolve(),
            seed=args.seed,
        )
        cases.append(case)
        print(
            f"froze {index}/{len(records)} {case.synthetic_input.source_filename} "
            f"base_largest={case.base.diagnostics.largest_component} "
            f"cal_largest={case.calibrated.diagnostics.largest_component}",
            flush=True,
        )

    artifact = write_frozen_artifact(cases, output_dir=output_dir)
    print(f"dirty-only artifact frozen: {artifact[0]}", flush=True)
    boards = [score_case(case) for case in cases]
    evaluation = aggregate(boards)
    candidate = evaluation["variants"]["calibrated_order_decoder144"]
    base = evaluation["variants"]["base_decoder144"]
    paired = evaluation["paired_candidate_minus_base"]
    promoted = bool(
        paired["adjacency"]["mean_delta"] > 0
        and paired["adjacency"]["paired_t_95_ci"][0] > 0
        and paired["correct_tile_count"]["mean_delta"] >= 0
        and paired["raw_ssim"]["mean_delta"] >= 0
        and candidate["components"]["false_added_bridges"]
        < base["components"]["false_added_bridges"]
    )
    report = {
        "experiment": "calibrated-socket-order-decoder144-v1",
        "status": "one-shot-source-disjoint-exact-synthetic",
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha,
            "contract": contract,
            "lineage_filenames": list(lineage.filenames),
            "lineage_digest": names_digest(lineage.filenames, sort_names=True),
            "lineage_checkpoint_paths": list(lineage.checkpoint_paths),
        },
        "calibrator": {
            "path": str(calibrator_path),
            "sha256": calibrator_sha,
            "schema": calibrator_payload["schema"],
            "fit_source_digest": calibrator_payload["fit_sources"]["digest"],
            "threshold": calibrator.threshold,
            "refit": False,
            "retuned": False,
        },
        "protocol": {
            "manifest_digest": compute_protocol_digest(manifest),
            "manifest_split": "train",
            "exact_synthetic_labels_only": True,
            "permutation_labels": "exact inverse of deterministic independent shuffle",
            "dirty_predictions_and_component_traces_frozen_before_scoring": True,
            "calibration_files_opened": False,
            "holdout_files_opened": False,
            "competition_test_files_opened": False,
            "source_disjoint_from_checkpoint_lineage": True,
            "source_disjoint_from_all_prior_exact_reports": True,
            "prior_exact_source_count_excluded": len(prior_names),
            "prior_exact_reports_checked": prior_reports,
            "d64_training_output_read_or_modified": False,
        },
        "selection": {
            "namespace": NAMESPACE,
            "seed": args.seed,
            "source_limit": args.source_limit,
            "source_filenames": list(names),
            "source_digest": names_digest(names),
            "draws_per_source": 1,
        },
        "fixed_comparison": {
            "base": "ordinary confidence-order decoder144",
            "candidate": "same decoder, calibrated hard component-edge order only",
            "component_edge_budget_per_axis": COMPONENT_BUDGET,
            "swap_edge_budget_per_axis": COMPONENT_BUDGET,
            "max_qap_swap_steps": 24,
            "border_weight": 0.20,
            "component_shift_unary_weight": 0.0,
            "candidate_changes_soft_pair_scores": False,
            "candidate_changes_border_unary": False,
            "candidate_changes_packing_or_qap_objective": False,
            "candidate_changes_qap_guidance": False,
        },
        "frozen_predictions": {
            "arrays_path": str(artifact[0]),
            "arrays_sha256": artifact[1],
            "metadata_path": str(artifact[2]),
            "metadata_sha256": artifact[3],
        },
        "evaluation": evaluation | {"boards": boards},
        "decision": {
            "promotion_gate": (
                "adjacency mean and CI lower >0; exact tiles and raw SSIM nonnegative; "
                "fewer false added bridges"
            ),
            "promote_calibrated_order": promoted,
            "disposition": "promote-order" if promoted else "reject-as-tested",
        },
        "runtime_seconds": {
            key: _mean([case.runtime_seconds[key] for case in cases])
            for key in ("matcher", "base_decoder", "calibrated_decoder")
        },
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
                "base": base,
                "candidate": candidate,
                "paired": paired,
                "decision": report["decision"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

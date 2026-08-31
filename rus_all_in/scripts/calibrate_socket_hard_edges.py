#!/usr/bin/env python3
"""Fit and once-confirm a bounded hard-Socket-edge confidence calibrator.

The only labels are exact inverse shuffles of independently corrupted clean
manifest-train targets.  Fit and confirmation sources are disjoint from the
complete SocketMatcher-v2 checkpoint lineage, from earlier exact-synthetic
panels, and from one another.  The linear model and its sole probability threshold are
frozen before confirmation targets are scored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from PIL import Image

from aiijc_puzzle.protocol import IMAGE_SIZE, compute_protocol_digest, sha256_file, split_tiles
from aiijc_puzzle.socket_confidence_calibration import (
    FEATURE_NAMES,
    FrozenLinearCalibrator,
    HardEdgeFeatures,
    exact_edge_labels,
    extract_hard_edge_features,
    fit_linear_calibrator,
    fixed_heuristic_selection,
    mutual_top1_selection,
)
from aiijc_puzzle.socket_matcher import BORDER_HEAD_EMBEDDING_V2, SocketMatcher
from aiijc_puzzle.synthetic_socket_evaluation import (
    DEFAULT_SYNTHETIC_NAMESPACE,
    ExactSyntheticReference,
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
DEFAULT_MANIFEST = PROJECT_ROOT / "data/interim/validation_manifest.json"
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
DEFAULT_PRIOR_REPORT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/socket-confidence-calibration/d32-v2-fit32-confirm16"
GRID = 24
TILE_COUNT = GRID * GRID
HARD_EDGES_PER_BOARD = 2 * GRID * (GRID - 1)
FIT_NAMESPACE = "aiijc-socket-confidence-calibration-fit-v1"
CONFIRM_NAMESPACE = "aiijc-socket-confidence-calibration-confirm-v1"


@dataclass(frozen=True)
class FrozenPanel:
    """Feature matrix frozen before exact references are scored."""

    values: np.ndarray
    board_index: np.ndarray
    source: np.ndarray
    target: np.ndarray
    axis: np.ndarray
    references: tuple[ExactSyntheticReference, ...]
    filenames: tuple[str, ...]
    dirty_sha256: tuple[str, ...]
    inference_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--prior-report-root", type=Path, default=DEFAULT_PRIOR_REPORT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fit-sources", type=int, default=32)
    parser.add_argument("--confirm-sources", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--target-fit-precision", type=float, default=0.80)
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


def load_v2_model(
    payload: dict[str, Any],
) -> tuple[SocketMatcher, dict[str, Any]]:
    contract = payload.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("checkpoint has no architecture contract")
    if contract.get("architecture") != "board-conditioned-partial-socket-matcher-v2":
        raise ValueError("this bounded probe requires a SocketMatcher v2 checkpoint")
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


def prior_exact_sources(report_root: Path) -> tuple[set[str], list[dict[str, Any]]]:
    """Collect source names from every earlier exact-synthetic report."""

    names: set[str] = set()
    reports: list[dict[str, Any]] = []
    for path in sorted(report_root.rglob("report.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        protocol = payload.get("protocol")
        experiment = payload.get("experiment")
        exact = bool(
            experiment == "socket-matcher-source-disjoint-exact-synthetic-v1"
            or (
                isinstance(protocol, dict)
                and (
                    str(protocol.get("permutation_labels", "")).startswith("exact inverse")
                    or protocol.get("exact_synthetic_labels_only") is True
                )
            )
        )
        if not exact:
            continue
        selection = payload.get("selection")
        selected: Any = None
        if isinstance(selection, dict):
            selected = selection.get("source_filenames")
            if selected is None:
                fit = selection.get("fit_source_filenames")
                confirm = selection.get("confirm_source_filenames")
                if isinstance(fit, list) and isinstance(confirm, list):
                    selected = [*fit, *confirm]
        if not isinstance(selected, list) or not all(isinstance(name, str) for name in selected):
            raise ValueError(f"exact-synthetic report has invalid source roster: {path}")
        names.update(selected)
        reports.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path.resolve()),
                "source_count": len(selected),
                "source_digest": names_digest(selected),
            }
        )
    return names, reports


@torch.inference_mode()
def freeze_panel(
    model: SocketMatcher,
    records: tuple[Any, ...],
    *,
    targets_dir: Path,
    seed: int,
    panel_name: str,
) -> FrozenPanel:
    """Open clean sources only to synthesize dirty inputs, then freeze features."""

    feature_rows: list[np.ndarray] = []
    board_rows: list[np.ndarray] = []
    source_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    axis_rows: list[np.ndarray] = []
    references: list[ExactSyntheticReference] = []
    filenames: list[str] = []
    dirty_hashes: list[str] = []
    inference_seconds = 0.0
    for board_index, record in enumerate(records):
        filename = str(record["filename"])
        target_path = targets_dir / filename
        expected_hash = record.get("target_sha256")
        if not isinstance(expected_hash, str) or sha256_file(target_path) != expected_hash:
            raise ValueError(f"manifest target hash mismatch for {filename}")
        clean = split_tiles(load_rgb(target_path))
        synthetic_input, reference = make_exact_synthetic_case(
            clean,
            source_filename=filename,
            draw_index=0,
            seed=seed,
        )
        dirty = synthetic_input.tiles
        tensor = torch.from_numpy(dirty.astype(np.float32)).permute(0, 3, 1, 2) / 255.0
        started = perf_counter()
        output = model(tensor.unsqueeze(0), grid=GRID)
        inference_seconds += perf_counter() - started
        features = extract_hard_edge_features(
            right_log_assignment=output.right_log_assignment[0],
            down_log_assignment=output.down_log_assignment[0],
            right_raw=output.right_raw[0],
            down_raw=output.down_raw[0],
            grid=GRID,
        )
        feature_rows.append(features.values)
        board_rows.append(np.full(HARD_EDGES_PER_BOARD, board_index, dtype=np.int16))
        source_rows.append(features.source)
        target_rows.append(features.target)
        axis_rows.append(features.axis)
        references.append(reference)
        filenames.append(filename)
        dirty_hashes.append(_array_sha256(dirty))
        print(
            f"froze {panel_name} {board_index + 1}/{len(records)} {filename}",
            flush=True,
        )
    return FrozenPanel(
        values=np.concatenate(feature_rows),
        board_index=np.concatenate(board_rows),
        source=np.concatenate(source_rows),
        target=np.concatenate(target_rows),
        axis=np.concatenate(axis_rows),
        references=tuple(references),
        filenames=tuple(filenames),
        dirty_sha256=tuple(dirty_hashes),
        inference_seconds=inference_seconds,
    )


def write_frozen_panel(
    panel: FrozenPanel,
    *,
    output_dir: Path,
    stem: str,
) -> tuple[Path, str, Path, str]:
    """Persist features and identities without exact permutations or labels."""

    array_path = output_dir / f"{stem}_dirty_features.npz"
    np.savez_compressed(
        array_path,
        values=panel.values,
        board_index=panel.board_index,
        source=panel.source,
        target=panel.target,
        axis=panel.axis,
    )
    metadata_path = output_dir / f"{stem}_dirty_features.json"
    metadata_path.write_text(
        json.dumps(
            {
                "schema": "aiijc-socket-hard-edge-dirty-features-v1",
                "contains_exact_references": False,
                "contains_labels": False,
                "feature_names": list(FEATURE_NAMES),
                "board_count": len(panel.filenames),
                "hard_edges_per_board": HARD_EDGES_PER_BOARD,
                "source_filenames": list(panel.filenames),
                "dirty_tile_sha256": list(panel.dirty_sha256),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return (
        array_path,
        sha256_file(array_path),
        metadata_path,
        sha256_file(metadata_path),
    )


def panel_labels(panel: FrozenPanel) -> np.ndarray:
    labels: list[np.ndarray] = []
    for board_index, reference in enumerate(panel.references):
        selected = panel.board_index == board_index
        board_features = HardEdgeFeatures(
            values=panel.values[selected],
            source=panel.source[selected],
            target=panel.target[selected],
            axis=panel.axis[selected],
        )
        labels.append(exact_edge_labels(board_features, reference.tile_at_position, grid=GRID))
    return np.concatenate(labels)


def selection_metrics(
    selection: np.ndarray,
    labels: np.ndarray,
    *,
    board_index: np.ndarray,
    board_count: int,
) -> dict[str, float | int]:
    selected = np.asarray(selection, dtype=bool)
    truth = np.asarray(labels, dtype=bool)
    if selected.shape != truth.shape or board_index.shape != truth.shape:
        raise ValueError("selection metric arrays are not aligned")
    selected_count = int(selected.sum())
    correct = int(np.count_nonzero(selected & truth))
    per_board_selected = np.bincount(board_index[selected], minlength=board_count)
    per_board_correct = np.bincount(board_index[selected & truth], minlength=board_count)
    return {
        "selected_edges": selected_count,
        "selected_edges_per_board": selected_count / board_count,
        "correct_selected_edges": correct,
        "correct_selected_edges_per_board": correct / board_count,
        "precision": correct / selected_count if selected_count else math.nan,
        "true_adjacency_recall": correct / (board_count * HARD_EDGES_PER_BOARD),
        "boards_with_any_selected": int(np.count_nonzero(per_board_selected)),
        "selected_per_board_min": int(per_board_selected.min()),
        "selected_per_board_max": int(per_board_selected.max()),
        "correct_per_board_min": int(per_board_correct.min()),
        "correct_per_board_max": int(per_board_correct.max()),
    }


def confidence_top_k_selection(
    values: np.ndarray,
    board_index: np.ndarray,
    *,
    board_count: int,
    k: int,
) -> np.ndarray:
    if not 1 <= k <= HARD_EDGES_PER_BOARD:
        raise ValueError(f"k must be in [1, {HARD_EDGES_PER_BOARD}]")
    confidence_index = FEATURE_NAMES.index("projected_edge_confidence")
    selected = np.zeros(len(values), dtype=bool)
    for board in range(board_count):
        indices = np.flatnonzero(board_index == board)
        order = np.argsort(-values[indices, confidence_index], kind="stable")
        selected[indices[order[:k]]] = True
    return selected


def choose_confidence_top_k_at_precision(
    values: np.ndarray,
    labels: np.ndarray,
    board_index: np.ndarray,
    *,
    board_count: int,
    target_precision: float,
) -> tuple[int, float]:
    """Freeze the broadest per-board confidence prefix meeting fit precision."""

    confidence_index = FEATURE_NAMES.index("projected_edge_confidence")
    correct_by_rank: list[np.ndarray] = []
    for board in range(board_count):
        indices = np.flatnonzero(board_index == board)
        if len(indices) != HARD_EDGES_PER_BOARD:
            raise ValueError("each board must contain the exact hard projection cardinality")
        order = np.argsort(-values[indices, confidence_index], kind="stable")
        correct_by_rank.append(np.asarray(labels[indices[order]], dtype=np.int64))
    cumulative_correct = np.cumsum(np.stack(correct_by_rank), axis=1).sum(axis=0)
    k_values = np.arange(1, HARD_EDGES_PER_BOARD + 1)
    precision = cumulative_correct / (board_count * k_values)
    valid = np.flatnonzero(precision >= target_precision)
    index = int(valid[-1]) if len(valid) else int(np.lexsort((k_values, precision))[-1])
    return index + 1, float(precision[index])


def evaluate_panel(
    panel: FrozenPanel,
    labels: np.ndarray,
    calibrator: FrozenLinearCalibrator,
    *,
    confidence_top_k_fit_coverage: int,
    confidence_top_k_fit_precision: int,
) -> dict[str, Any]:
    board_count = len(panel.filenames)
    probability = calibrator.predict_probability(panel.values)
    selections = {
        "learned_logistic_single_threshold": probability >= calibrator.threshold,
        "fixed_precision_first_heuristic": fixed_heuristic_selection(panel.values),
        "ot_mutual_top1": mutual_top1_selection(panel.values, variant="ot"),
        "raw_mutual_top1": mutual_top1_selection(panel.values, variant="raw"),
        "projected_confidence_top_k_fit_coverage": confidence_top_k_selection(
            panel.values,
            panel.board_index,
            board_count=board_count,
            k=confidence_top_k_fit_coverage,
        ),
        "projected_confidence_top_k_fit_precision": confidence_top_k_selection(
            panel.values,
            panel.board_index,
            board_count=board_count,
            k=confidence_top_k_fit_precision,
        ),
    }
    metrics = {
        name: selection_metrics(
            selection,
            labels,
            board_index=panel.board_index,
            board_count=board_count,
        )
        for name, selection in selections.items()
    }
    metrics["all_hard_projected_edges"] = selection_metrics(
        np.ones(len(labels), dtype=bool),
        labels,
        board_index=panel.board_index,
        board_count=board_count,
    )
    return metrics


def calibrator_payload(
    calibrator: FrozenLinearCalibrator,
    *,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    fit_filenames: tuple[str, ...],
    confidence_top_k_fit_coverage: int,
    confidence_top_k_fit_precision: int,
    confidence_top_k_fit_precision_achieved: float,
) -> dict[str, Any]:
    coefficient_rows = [
        {"feature": name, "standardised_coefficient": float(coefficient)}
        for name, coefficient in zip(
            calibrator.feature_names,
            calibrator.coefficients,
            strict=True,
        )
    ]
    coefficient_rows.sort(key=lambda row: -abs(row["standardised_coefficient"]))
    return {
        "schema": "aiijc-socket-hard-edge-linear-calibrator-v1",
        "dirty_visible_features_only": True,
        "estimator": {
            "type": "standard-scaler plus balanced logistic regression",
            "C": 1.0,
            "solver": "lbfgs",
            "random_state": 0,
            "feature_names": list(calibrator.feature_names),
            "mean": calibrator.mean.tolist(),
            "scale": calibrator.scale.tolist(),
            "coefficients": calibrator.coefficients.tolist(),
            "intercept": calibrator.intercept,
            "coefficient_ranking": coefficient_rows,
        },
        "single_threshold": {
            "probability_greater_equal": calibrator.threshold,
            "target_fit_precision": calibrator.target_fit_precision,
            "achieved_fit_precision": calibrator.achieved_fit_precision,
            "selection_rule": "most inclusive unique fit score meeting target precision",
        },
        "fit_sources": {
            "filenames": list(fit_filenames),
            "digest": names_digest(fit_filenames),
        },
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": checkpoint_sha256,
        },
        "frozen_coverage_control": {
            "projected_confidence_top_k_per_board": confidence_top_k_fit_coverage,
        },
        "frozen_fit_precision_control": {
            "projected_confidence_top_k_per_board": confidence_top_k_fit_precision,
            "achieved_fit_precision": confidence_top_k_fit_precision_achieved,
        },
        "confirmation_data_seen": False,
    }


def main() -> None:
    args = parse_args()
    if args.fit_sources <= 0 or args.confirm_sources <= 0:
        raise ValueError("fit-sources and confirm-sources must be positive")
    if not 0.0 < args.target_fit_precision <= 1.0:
        raise ValueError("target-fit-precision must be in (0, 1]")
    if args.torch_threads <= 0:
        raise ValueError("torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint.resolve()
    checkpoint_sha = sha256_file(checkpoint_path)
    checkpoint_payload, lineage = load_checkpoint_with_lineage(
        checkpoint_path,
        project_root=PROJECT_ROOT,
    )
    model, contract = load_v2_model(checkpoint_payload)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("protocol_digest") != compute_protocol_digest(manifest):
        raise ValueError("validation manifest protocol digest is invalid")

    prior_names, prior_reports = prior_exact_sources(args.prior_report_root.resolve())
    base_exclusions = set(lineage.filenames) | prior_names
    fit_records = select_source_disjoint_train_records(
        manifest,
        excluded_filenames=tuple(sorted(base_exclusions)),
        limit=args.fit_sources,
        seed=args.seed,
        namespace=FIT_NAMESPACE,
    )
    fit_names = tuple(str(record["filename"]) for record in fit_records)
    confirm_records = select_source_disjoint_train_records(
        manifest,
        excluded_filenames=tuple(sorted(base_exclusions | set(fit_names))),
        limit=args.confirm_sources,
        seed=args.seed + 1,
        namespace=CONFIRM_NAMESPACE,
    )
    confirm_names = tuple(str(record["filename"]) for record in confirm_records)
    if set(fit_names) & set(confirm_names):
        raise RuntimeError("fit and confirmation source rosters overlap")

    fit_panel = freeze_panel(
        model,
        fit_records,
        targets_dir=args.targets.resolve(),
        seed=args.seed,
        panel_name="fit",
    )
    fit_artifact = write_frozen_panel(fit_panel, output_dir=output_dir, stem="fit")
    fit_labels = panel_labels(fit_panel)
    calibrator = fit_linear_calibrator(
        fit_panel.values,
        fit_labels,
        target_precision=args.target_fit_precision,
    )
    fit_learned = calibrator.select(fit_panel.values)
    mean_fit_selected = int(fit_learned.sum()) / len(fit_names)
    confidence_top_k_fit_coverage = max(
        1,
        min(HARD_EDGES_PER_BOARD, round(mean_fit_selected)),
    )
    (
        confidence_top_k_fit_precision,
        confidence_top_k_fit_precision_achieved,
    ) = choose_confidence_top_k_at_precision(
        fit_panel.values,
        fit_labels,
        fit_panel.board_index,
        board_count=len(fit_names),
        target_precision=args.target_fit_precision,
    )
    frozen_payload = calibrator_payload(
        calibrator,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha,
        fit_filenames=fit_names,
        confidence_top_k_fit_coverage=confidence_top_k_fit_coverage,
        confidence_top_k_fit_precision=confidence_top_k_fit_precision,
        confidence_top_k_fit_precision_achieved=confidence_top_k_fit_precision_achieved,
    )
    calibrator_path = output_dir / "frozen_calibrator.json"
    calibrator_path.write_text(
        json.dumps(frozen_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    calibrator_sha = sha256_file(calibrator_path)
    fit_evaluation = evaluate_panel(
        fit_panel,
        fit_labels,
        calibrator,
        confidence_top_k_fit_coverage=confidence_top_k_fit_coverage,
        confidence_top_k_fit_precision=confidence_top_k_fit_precision,
    )
    print(
        json.dumps(
            {
                "event": "calibrator_frozen_before_confirmation",
                "path": str(calibrator_path),
                "sha256": calibrator_sha,
                "fit_learned": fit_evaluation["learned_logistic_single_threshold"],
            }
        ),
        flush=True,
    )

    # Confirmation sources are not opened until the estimator, threshold and
    # top-K coverage control above have all been serialized and hash-locked.
    confirm_panel = freeze_panel(
        model,
        confirm_records,
        targets_dir=args.targets.resolve(),
        seed=args.seed + 1,
        panel_name="confirm",
    )
    confirm_artifact = write_frozen_panel(
        confirm_panel,
        output_dir=output_dir,
        stem="confirm",
    )
    if sha256_file(calibrator_path) != calibrator_sha:
        raise RuntimeError("frozen calibrator changed during confirmation feature generation")
    confirm_labels = panel_labels(confirm_panel)
    confirm_evaluation = evaluate_panel(
        confirm_panel,
        confirm_labels,
        calibrator,
        confidence_top_k_fit_coverage=confidence_top_k_fit_coverage,
        confidence_top_k_fit_precision=confidence_top_k_fit_precision,
    )

    learned_confirm = confirm_evaluation["learned_logistic_single_threshold"]
    coverage_control = confirm_evaluation["projected_confidence_top_k_fit_coverage"]
    matched_precision_control = confirm_evaluation[
        "projected_confidence_top_k_fit_precision"
    ]
    precision_floor = args.target_fit_precision - 0.05
    material_gain = bool(
        learned_confirm["precision"] >= precision_floor
        and matched_precision_control["precision"] >= precision_floor
        and abs(
            learned_confirm["precision"] - matched_precision_control["precision"]
        )
        <= 0.05
        and learned_confirm["correct_selected_edges_per_board"]
        >= 1.15 * matched_precision_control["correct_selected_edges_per_board"]
    )
    report = {
        "experiment": "socket-hard-edge-confidence-calibration-v1",
        "status": "exact-synthetic-fit-and-one-shot-confirmation",
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha,
            "architecture_contract": contract,
            "lineage_filenames": list(lineage.filenames),
            "lineage_digest": names_digest(lineage.filenames, sort_names=True),
            "lineage_checkpoint_paths": list(lineage.checkpoint_paths),
        },
        "protocol": {
            "manifest_digest": compute_protocol_digest(manifest),
            "manifest_split": "train",
            "exact_synthetic_labels_only": True,
            "corruption_implementation": "aiijc_puzzle.restoration_r6.distort_tiles",
            "selection_namespace_base": DEFAULT_SYNTHETIC_NAMESPACE,
            "calibration_files_opened": False,
            "holdout_files_opened": False,
            "competition_test_files_opened": False,
            "dirty_features_frozen_before_reference_scoring": True,
            "calibrator_and_single_threshold_frozen_before_confirmation_opened": True,
            "fit_confirm_checkpoint_lineage_disjoint": True,
            "fit_confirm_prior_exact_panel_disjoint": True,
            "fit_confirm_mutually_source_disjoint": True,
            "layout_decoder_run": False,
        },
        "prior_exact_panels": {
            "source_count_excluded": len(prior_names),
            "source_digest": names_digest(tuple(sorted(prior_names)), sort_names=True),
            "reports": prior_reports,
        },
        "selection": {
            "seed": args.seed,
            "fit_namespace": FIT_NAMESPACE,
            "confirm_namespace": CONFIRM_NAMESPACE,
            "fit_source_filenames": list(fit_names),
            "fit_source_digest": names_digest(fit_names),
            "confirm_source_filenames": list(confirm_names),
            "confirm_source_digest": names_digest(confirm_names),
            "draws_per_source": 1,
        },
        "features": {
            "names": list(FEATURE_NAMES),
            "count": len(FEATURE_NAMES),
            "cycle_support": "optional K4 commutative closure statistics",
            "all_features_dirty_visible": True,
            "hard_projection": "expanded-dustbin Hungarian, exactly 552 edges per axis",
        },
        "frozen_artifacts": {
            "fit": {
                "arrays_path": str(fit_artifact[0]),
                "arrays_sha256": fit_artifact[1],
                "metadata_path": str(fit_artifact[2]),
                "metadata_sha256": fit_artifact[3],
            },
            "calibrator": {
                "path": str(calibrator_path),
                "sha256": calibrator_sha,
            },
            "confirm": {
                "arrays_path": str(confirm_artifact[0]),
                "arrays_sha256": confirm_artifact[1],
                "metadata_path": str(confirm_artifact[2]),
                "metadata_sha256": confirm_artifact[3],
            },
        },
        "frozen_calibrator": frozen_payload,
        "controls": {
            "fixed_precision_first_heuristic": {
                "projected_edge_confidence_min": -1.0,
                "ot_row_real_margin_min": 0.0,
                "ot_column_real_margin_min": 0.0,
                "both_dustbin_margins_min": 0.5,
            },
            "mutual_top1": ["raw", "ot"],
            "coverage_matched_projected_confidence_top_k_per_board": (
                confidence_top_k_fit_coverage
            ),
            "fit_precision_matched_projected_confidence_top_k_per_board": (
                confidence_top_k_fit_precision
            ),
            "fit_precision_matched_achieved_fit_precision": (
                confidence_top_k_fit_precision_achieved
            ),
        },
        "fit_evaluation": fit_evaluation,
        "confirm_evaluation": confirm_evaluation,
        "decision": {
            "confirmation_precision_floor": precision_floor,
            "material_gain_definition": (
                "learned and fit-precision-matched top-K confirmation precision both >= "
                "target-5pp and within 5pp; learned has >=15% more correct edges/board"
            ),
            "material_coverage_gain_at_matched_precision": material_gain,
            "layout_decoder_authorized_by_this_probe": material_gain,
            "layout_decoder_run": False,
            "reason": (
                "Calibration met the predeclared matched-coverage gate; a later decoder "
                "experiment may use it."
                if material_gain
                else (
                    "Calibration did not clear the matched-coverage gate; no layout "
                    "decoder was run."
                )
            ),
        },
        "runtime_seconds": {
            "fit_dirty_inference": fit_panel.inference_seconds,
            "confirm_dirty_inference": confirm_panel.inference_seconds,
        },
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "event": "complete",
                "report": str(report_path),
                "fit": fit_evaluation["learned_logistic_single_threshold"],
                "confirm": learned_confirm,
                "coverage_control": coverage_control,
                "matched_precision_control": matched_precision_control,
                "decision": report["decision"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Measure 2x2 commutative-cycle evidence on the opened offset-2304 panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from aiijc_puzzle.candidate_supply import recover_layout
from aiijc_puzzle.layout_evaluation import RECOVERED_REFERENCE_CAVEAT
from aiijc_puzzle.precision_first_socket_decoder import (
    PrecisionFirstDecoderConfig,
    precision_edge_evidence,
)
from aiijc_puzzle.protocol import IMAGE_SIZE, sha256_file, split_tiles
from aiijc_puzzle.socket_cycle_diagnostic import (
    axis_socket_rankings,
    commutative_cycle_support,
)
from aiijc_puzzle.socket_decoder import hard_partial_axis_matching

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FROZEN = (
    PROJECT_ROOT
    / "outputs/socket-matcher/component-anchor-diagnostic-offset2304-dev24"
    / "frozen_predictions.npz"
)
DEFAULT_FREEZE_METADATA = DEFAULT_FROZEN.parent / "freeze_metadata.json"
DEFAULT_CONFIG = PROJECT_ROOT / "configs/socket_precision_first_v1.json"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs/socket-matcher/commutative-cycle-diagnostic-offset2304"
    / "report.json"
)
GRID = 24
TILE_COUNT = GRID * GRID
PANEL_SIZE = 24
TOP_K_VALUES = (4, 8, 16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--freeze-metadata", type=Path, default=DEFAULT_FREEZE_METADATA)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--inputs", type=Path, default=Path("data/raw/train/inputs"))
    parser.add_argument("--targets", type=Path, default=Path("data/raw/train/targets"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        if image.mode != "RGB" or image.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"expected RGB 480x480 image: {path}")
        return np.asarray(image, dtype=np.uint8)


def _names_digest(names: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(names).encode()).hexdigest()


def _positions(reference: np.ndarray) -> np.ndarray:
    result = np.empty(TILE_COUNT, dtype=np.int32)
    result[reference] = np.arange(TILE_COUNT, dtype=np.int32)
    return result


def _edge_correct(edge: Any, position: np.ndarray) -> bool:
    source_row, source_column = divmod(int(position[edge.source]), GRID)
    target_row, target_column = divmod(int(position[edge.target]), GRID)
    return bool(
        target_row - source_row == edge.delta_row
        and target_column - source_column == edge.delta_column
    )


def _metrics(
    rows: list[dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
) -> dict[str, float]:
    selected = [row for row in rows if predicate(row)]
    trusted = [row for row in selected if row["trusted"]]
    correct = sum(row["correct"] for row in trusted)
    return {
        "dirty_selected_edges_per_board": len(selected) / PANEL_SIZE,
        "trusted_selected_edges_per_board": len(trusted) / PANEL_SIZE,
        "trusted_exact_edge_rate": correct / len(trusted) if trusted else math.nan,
        "trusted_correct_edges_per_board": correct / PANEL_SIZE,
    }


def _mean(values: Iterable[float]) -> float:
    array = np.asarray(tuple(values), dtype=np.float64)
    return float(np.mean(array)) if len(array) else math.nan


def _quantiles(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if not len(array):
        return {}
    return {
        name: float(value)
        for name, value in zip(
            ("q00", "q25", "q50", "q75", "q100"),
            np.quantile(array, (0.0, 0.25, 0.5, 0.75, 1.0)),
            strict=True,
        )
    }


def main() -> None:
    args = parse_args()
    freeze_metadata = json.loads(args.freeze_metadata.read_text(encoding="utf-8"))
    if freeze_metadata.get("phase") != "dirty_predictions_frozen_before_target_access":
        raise ValueError("source predictions do not have a dirty-only freeze declaration")
    expected_hash = freeze_metadata["frozen_predictions"]["sha256"]
    if sha256_file(args.frozen) != expected_hash:
        raise ValueError("opened-panel frozen predictions changed")
    if freeze_metadata["selection"]["offset"] != 2304:
        raise ValueError("this diagnostic is restricted to already-open offset 2304")

    config_payload = json.loads(args.config.read_text(encoding="utf-8"))
    fixed_config = PrecisionFirstDecoderConfig(**config_payload["config"])
    stricter_config = PrecisionFirstDecoderConfig(
        minimum_edge_confidence=-0.75,
        minimum_real_row_margin=fixed_config.minimum_real_row_margin,
        minimum_real_column_margin=fixed_config.minimum_real_column_margin,
        minimum_dustbin_margin=fixed_config.minimum_dustbin_margin,
        maximum_component_size=fixed_config.maximum_component_size,
        border_weight=fixed_config.border_weight,
    )
    frozen = np.load(args.frozen, allow_pickle=False)
    names = tuple(str(value) for value in frozen["filenames"].tolist())
    if len(names) != PANEL_SIZE or list(names) != freeze_metadata["selection"]["filenames"]:
        raise ValueError("frozen filename panel differs from offset-2304 metadata")

    rows: list[dict[str, Any]] = []
    for board_index, filename in enumerate(names):
        dirty = split_tiles(load_rgb(args.inputs / filename))
        clean = split_tiles(load_rgb(args.targets / filename))
        recovered = recover_layout(dirty, clean)
        position = _positions(recovered.dirty_at_position)
        trusted_position = recovered.margin_at_position >= np.median(
            recovered.margin_at_position
        )
        trusted_tile = np.zeros(TILE_COUNT, dtype=bool)
        trusted_tile[recovered.dirty_at_position] = trusted_position
        matrices = {
            "right": frozen["right_log_assignment"][board_index],
            "down": frozen["down_log_assignment"][board_index],
        }
        rankings = {
            axis: axis_socket_rankings(matrix, grid=GRID, maximum_k=max(TOP_K_VALUES))
            for axis, matrix in matrices.items()
        }
        for axis in ("right", "down"):
            matching = hard_partial_axis_matching(matrices[axis], grid=GRID, axis=axis)
            for edge in matching.edges:
                fixed = precision_edge_evidence(
                    matrices[axis],
                    edge,
                    grid=GRID,
                    config=fixed_config,
                )
                stricter = precision_edge_evidence(
                    matrices[axis],
                    edge,
                    grid=GRID,
                    config=stricter_config,
                )
                row: dict[str, Any] = {
                    "board": board_index,
                    "axis": axis,
                    "trusted": bool(
                        trusted_tile[edge.source] and trusted_tile[edge.target]
                    ),
                    "correct": _edge_correct(edge, position),
                    "confidence_selected": fixed.eligible,
                    "stricter_confidence_selected": stricter.eligible,
                    "edge_confidence": edge.confidence,
                }
                for top_k in TOP_K_VALUES:
                    support = commutative_cycle_support(
                        edge,
                        right=rankings["right"],
                        down=rankings["down"],
                        top_k=top_k,
                    )
                    row[f"base_rank_{top_k}"] = support.base_rank
                    row[f"support_count_{top_k}"] = support.support_count
                    row[f"best_total_rank_{top_k}"] = support.best_total_rank
                    row[f"best_score_{top_k}"] = support.best_total_conditional_log_score
                rows.append(row)
        print(f"analysed {board_index + 1}/{len(names)} {filename}", flush=True)

    confidence_metrics = _metrics(rows, lambda row: row["confidence_selected"])
    stricter_metrics = _metrics(rows, lambda row: row["stricter_confidence_selected"])
    top_k_reports: dict[str, Any] = {}
    for top_k in TOP_K_VALUES:
        def base(row: dict[str, Any], top_k: int = top_k) -> bool:
            return row[f"base_rank_{top_k}"] is not None

        def supported(row: dict[str, Any], top_k: int = top_k) -> bool:
            return (
                row[f"base_rank_{top_k}"] is not None
                and row[f"support_count_{top_k}"] > 0
            )

        base_metrics = _metrics(rows, base)
        cycle_metrics = _metrics(rows, supported)
        supported_trusted = [row for row in rows if row["trusted"] and supported(row)]
        score_values = np.asarray(
            [float(row[f"best_score_{top_k}"]) for row in supported_trusted],
            dtype=np.float64,
        )
        score_thresholds = np.quantile(score_values, (0.25, 0.5, 0.75))
        score_slices = {
            f"top_{int((1.0 - quantile) * 100)}pct": {
                "threshold": float(threshold),
                "metrics": _metrics(
                    rows,
                    lambda row, threshold=threshold, supported=supported, top_k=top_k: (
                        supported(row)
                        and float(row[f"best_score_{top_k}"]) >= threshold
                    ),
                ),
            }
            for quantile, threshold in zip(
                (0.25, 0.5, 0.75), score_thresholds, strict=True
            )
        }
        confidence_and_cycle = _metrics(
            rows,
            lambda row, supported=supported: row["confidence_selected"] and supported(row),
        )
        confidence_without_cycle = _metrics(
            rows,
            lambda row, supported=supported: row["confidence_selected"]
            and not supported(row),
        )
        cycle_without_confidence = _metrics(
            rows,
            lambda row, supported=supported: supported(row)
            and not row["confidence_selected"],
        )
        true_supported = [row for row in supported_trusted if row["correct"]]
        false_supported = [row for row in supported_trusted if not row["correct"]]
        top_k_reports[str(top_k)] = {
            "base_hard_edges_with_row_rank_at_most_k": base_metrics,
            "commutative_cycle_supported": cycle_metrics,
            "precision_lift_over_same_rank_base_pp": 100.0
            * (
                cycle_metrics["trusted_exact_edge_rate"]
                - base_metrics["trusted_exact_edge_rate"]
            ),
            "correct_edge_coverage_retention_vs_same_rank_base": (
                cycle_metrics["trusted_correct_edges_per_board"]
                / base_metrics["trusted_correct_edges_per_board"]
            ),
            "strong_rank_sum_at_most_2k": _metrics(
                rows,
                lambda row, supported=supported, top_k=top_k: (
                    supported(row) and int(row[f"best_total_rank_{top_k}"]) <= 2 * top_k
                ),
            ),
            "at_least_two_cycle_witnesses": _metrics(
                rows,
                lambda row, top_k=top_k: (
                    row[f"base_rank_{top_k}"] is not None
                    and row[f"support_count_{top_k}"] >= 2
                ),
            ),
            "overlap_with_fixed_confidence": {
                "confidence_and_cycle": confidence_and_cycle,
                "confidence_without_cycle": confidence_without_cycle,
                "cycle_without_confidence": cycle_without_confidence,
                "confidence_correct_edge_retention": (
                    confidence_and_cycle["trusted_correct_edges_per_board"]
                    / confidence_metrics["trusted_correct_edges_per_board"]
                ),
            },
            "best_cycle_score_descriptive_slices": score_slices,
            "aggregate_rank_score": {
                "support_count_mean": _mean(
                    row[f"support_count_{top_k}"] for row in supported_trusted
                ),
                "best_total_rank_true_quantiles": _quantiles(
                    float(row[f"best_total_rank_{top_k}"]) for row in true_supported
                ),
                "best_total_rank_false_quantiles": _quantiles(
                    float(row[f"best_total_rank_{top_k}"]) for row in false_supported
                ),
                "best_score_true_quantiles": _quantiles(
                    float(row[f"best_score_{top_k}"]) for row in true_supported
                ),
                "best_score_false_quantiles": _quantiles(
                    float(row[f"best_score_{top_k}"]) for row in false_supported
                ),
            },
        }

    k4_overlap = top_k_reports["4"]["overlap_with_fixed_confidence"]
    report = {
        "diagnostic": "socket-commutative-cycle-support-v1",
        "scope": "read-only target-assisted diagnostic; no layout or decoder run",
        "selection": {
            "split": "train",
            "namespace": freeze_metadata["selection"]["namespace"],
            "offset": 2304,
            "limit": PANEL_SIZE,
            "filenames": list(names),
            "filenames_digest": _names_digest(names),
            "targets_previously_opened": True,
            "new_target_panel_opened": False,
        },
        "frozen_predictions": {
            "path": str(args.frozen),
            "sha256": expected_hash,
        },
        "reference_caveat": RECOVERED_REFERENCE_CAVEAT,
        "trusted_policy": "both endpoints in per-board top 50% recovered-position margin",
        "fixed_confidence_baseline": confidence_metrics,
        "stricter_scalar_confidence_minus075_comparator": stricter_metrics,
        "top_k": top_k_reports,
        "decision": {
            "substantial_incremental_precision_coverage_lift": False,
            "new_decoder_or_layout_run": False,
            "reason": (
                "K=4 raises precision inside the confidence-selected subset, but retains "
                f"only {100.0 * k4_overlap['confidence_correct_edge_retention']:.2f}% of its "
                "correct edges; cycle-only edges have low precision. K=8 adds little "
                "precision and K=16 is nearly saturated by chance closures. Aggregate "
                "rank/score does not recover confidence-level precision at useful coverage."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["decision"], indent=2), flush=True)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()

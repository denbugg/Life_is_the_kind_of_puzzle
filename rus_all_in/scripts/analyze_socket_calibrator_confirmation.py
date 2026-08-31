#!/usr/bin/env python3
"""Add paired board-level CIs to an already-open Socket calibrator confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import stats

from aiijc_puzzle.protocol import sha256_file, split_tiles
from aiijc_puzzle.socket_confidence_calibration import (
    FEATURE_NAMES,
    HardEdgeFeatures,
    exact_edge_labels,
    fixed_heuristic_selection,
    frozen_linear_calibrator_from_payload,
)
from aiijc_puzzle.synthetic_socket_evaluation import make_exact_synthetic_case, names_digest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "outputs/socket-confidence-calibration/d64-v2-fit32-confirm16"
DEFAULT_REPORT = DEFAULT_ROOT / "report.json"
DEFAULT_CALIBRATOR = DEFAULT_ROOT / "frozen_calibrator.json"
DEFAULT_FEATURES = DEFAULT_ROOT / "confirm_dirty_features.npz"
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
DEFAULT_OUTPUT = DEFAULT_ROOT / "paired_confirmation_analysis.json"
GRID = 24
HARD_EDGES_PER_BOARD = 2 * GRID * (GRID - 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--calibrator", type=Path, default=DEFAULT_CALIBRATOR)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _paired_summary(candidate: np.ndarray, control: np.ndarray) -> dict[str, Any]:
    delta = np.asarray(candidate, dtype=np.float64) - np.asarray(control, dtype=np.float64)
    if delta.ndim != 1 or not len(delta) or not np.isfinite(delta).all():
        raise ValueError("paired metric values must be aligned finite vectors")
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


def _selection_row(selection: np.ndarray, labels: np.ndarray) -> dict[str, float | int]:
    selected = int(selection.sum())
    correct = int(np.count_nonzero(selection & labels))
    return {
        "selected": selected,
        "correct": correct,
        "precision": correct / selected if selected else 0.0,
    }


def main() -> None:
    args = parse_args()
    report_path = args.report.resolve()
    calibrator_path = args.calibrator.resolve()
    features_path = args.features.resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("experiment") != "socket-hard-edge-confidence-calibration-v1":
        raise ValueError("input report is not a Socket hard-edge calibration")
    if report.get("decision", {}).get("layout_decoder_run") is not False:
        raise ValueError("this analysis expects a calibrator-only experiment")
    frozen = report.get("frozen_artifacts", {})
    if sha256_file(features_path) != frozen.get("confirm", {}).get("arrays_sha256"):
        raise ValueError("confirmation feature artifact hash differs from report")
    if sha256_file(calibrator_path) != frozen.get("calibrator", {}).get("sha256"):
        raise ValueError("calibrator hash differs from report")
    calibrator = frozen_linear_calibrator_from_payload(
        json.loads(calibrator_path.read_text(encoding="utf-8"))
    )
    names = tuple(report["selection"]["confirm_source_filenames"])
    seed = int(report["selection"]["seed"]) + 1
    top_k = int(
        report["controls"][
            "fit_precision_matched_projected_confidence_top_k_per_board"
        ]
    )
    arrays = np.load(features_path, allow_pickle=False)
    values = arrays["values"]
    board_index = arrays["board_index"]
    source = arrays["source"]
    target = arrays["target"]
    axis = arrays["axis"]
    if values.shape != (len(names) * HARD_EDGES_PER_BOARD, len(FEATURE_NAMES)):
        raise ValueError("confirmation feature matrix has an unexpected shape")

    confidence_index = FEATURE_NAMES.index("projected_edge_confidence")
    boards: list[dict[str, Any]] = []
    for index, filename in enumerate(names):
        selected = board_index == index
        if int(selected.sum()) != HARD_EDGES_PER_BOARD:
            raise ValueError("confirmation board has the wrong hard-edge cardinality")
        clean = split_tiles(_load_rgb(args.targets.resolve() / filename))
        _, reference = make_exact_synthetic_case(
            clean,
            source_filename=filename,
            draw_index=0,
            seed=seed,
        )
        features = HardEdgeFeatures(
            values=values[selected],
            source=source[selected],
            target=target[selected],
            axis=axis[selected],
        )
        labels = exact_edge_labels(features, reference.tile_at_position, grid=GRID)
        probability = calibrator.predict_probability(features.values)
        learned = probability >= calibrator.threshold
        order = np.argsort(
            -features.values[:, confidence_index],
            kind="stable",
        )
        top_rank = np.zeros(HARD_EDGES_PER_BOARD, dtype=bool)
        top_rank[order[:top_k]] = True
        fixed = fixed_heuristic_selection(features.values)
        boards.append(
            {
                "source_filename": filename,
                "learned": _selection_row(learned, labels),
                "fit_precision_top_k_control": _selection_row(top_rank, labels),
                "fixed_heuristic": _selection_row(fixed, labels),
            }
        )

    def vector(variant: str, metric: str) -> np.ndarray:
        return np.asarray([board[variant][metric] for board in boards], dtype=np.float64)

    reported = report["confirm_evaluation"]
    if sum(board["learned"]["correct"] for board in boards) != reported[
        "learned_logistic_single_threshold"
    ]["correct_selected_edges"]:
        raise RuntimeError("reconstructed learned confirmation metrics differ from report")
    if sum(board["fit_precision_top_k_control"]["correct"] for board in boards) != reported[
        "projected_confidence_top_k_fit_precision"
    ]["correct_selected_edges"]:
        raise RuntimeError("reconstructed top-K confirmation metrics differ from report")

    analysis = {
        "analysis": "socket-hard-edge-confirmation-paired-ci-v1",
        "scope": "read-only analysis of the already-open one-shot confirmation panel",
        "source_report": {
            "path": str(report_path),
            "sha256": sha256_file(report_path),
        },
        "calibrator": {
            "path": str(calibrator_path),
            "sha256": sha256_file(calibrator_path),
            "refit": False,
            "retuned": False,
        },
        "confirmation": {
            "source_filenames": list(names),
            "source_digest": names_digest(names),
            "board_count": len(names),
            "fit_precision_control_top_k": top_k,
        },
        "paired": {
            "learned_minus_fit_precision_top_k": {
                metric: _paired_summary(
                    vector("learned", metric),
                    vector("fit_precision_top_k_control", metric),
                )
                for metric in ("correct", "precision")
            },
            "learned_minus_fixed_heuristic": {
                metric: _paired_summary(
                    vector("learned", metric),
                    vector("fixed_heuristic", metric),
                )
                for metric in ("correct", "precision")
            },
        },
        "boards": boards,
    }
    args.output.resolve().write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(analysis["paired"], indent=2), flush=True)
    print(f"wrote {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Bounded development replay for coordinate-guided global cyclic origin.

This script is intentionally limited to the already-opened absolute-coordinate
``source64 x draw2`` panel.  It freezes dirty-only layouts before scoring the
known synthetic inverse shuffles and must not be pointed at a new panel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from run_absolute_coordinate_sorter import (
    GRID,
    AbsoluteCoordinateSorter,
    load_socket_backbone,
    prepare_clean_boards,
    synthetic_example,
)

from aiijc_puzzle.coordinate_cyclic_placer import (
    CoordinateCyclicConfig,
    select_coordinate_cyclic_translation,
)
from aiijc_puzzle.layout_evaluation import evaluate_layout
from aiijc_puzzle.protocol import compute_protocol_digest, sha256_file
from aiijc_puzzle.socket_decoder import SocketDecoderConfig, decode_socket_assignments
from aiijc_puzzle.socket_translation_placer import (
    CyclicTranslationConfig,
    select_global_cyclic_translation,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "interim" / "validation_manifest.json"
DEFAULT_TARGETS = PROJECT_ROOT / "data" / "raw" / "train" / "targets"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "absolute-coordinate-sorter"
    / "component-translation-scale-d64-head32-train2048-s1600"
    / "absolute_coordinate_sorter.pt"
)
DEFAULT_REPLAY_REPORT = (
    PROJECT_ROOT
    / "outputs"
    / "absolute-coordinate-sorter"
    / "axis-development-source64-draw2-replay"
    / "report.json"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "absolute-coordinate-sorter"
    / "coordinate-cyclic-origin-source64-draw2-development"
)
BASELINE = "decoder144_cyclic_border5"
MATERIAL_EXACT_DELTA = 0.25
MAX_ADJACENCY_LOSS_PERCENTAGE_POINTS = 0.2

ARM_CONFIGS = {
    "coordinate_joint": CoordinateCyclicConfig(),
    "coordinate_row_border_column": CoordinateCyclicConfig(
        row_coordinate_weight=1.0,
        row_socket_weight=0.0,
        column_coordinate_weight=0.0,
        column_socket_weight=1.0,
    ),
    "coordinate_border_equal_axis_blend": CoordinateCyclicConfig(
        row_coordinate_weight=1.0,
        row_socket_weight=1.0,
        column_coordinate_weight=1.0,
        column_socket_weight=1.0,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--reuse-panel-report", type=Path, default=DEFAULT_REPLAY_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--device",
        choices=("cpu",),
        default="cpu",
        help="this cheap replay is CPU-only and must not contend with MPS training",
    )
    return parser.parse_args()


def _names_digest(names: list[str]) -> str:
    return hashlib.sha256("\n".join(names).encode()).hexdigest()


def _numeric_mean(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = [key for key, value in rows[0].items() if isinstance(value, int | float)]
    return {key: float(np.mean([float(row[key]) for row in rows])) for key in keys}


def _load_replay_contract(
    report_path: Path,
    checkpoint_path: Path,
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], tuple[Any, ...], int, int]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_experiment = "socket-backed-absolute-coordinate-sorter-v1-frozen-confirmation"
    if report.get("experiment") != expected_experiment:
        raise ValueError("reuse report is not an absolute-coordinate confirmation")
    selection = report.get("selection", {})
    names = selection.get("eval_filenames")
    if not isinstance(names, list) or len(names) != 64 or not all(
        isinstance(name, str) for name in names
    ):
        raise ValueError("reuse report must declare the opened 64-source panel")
    if selection.get("eval_digest") != _names_digest(names):
        raise ValueError("reuse report source digest is inconsistent")
    draws = selection.get("draws_per_source")
    seed = report.get("configuration", {}).get("seed")
    if draws != 2 or not isinstance(seed, int):
        raise ValueError("reuse report must declare the opened source64 x draw2 seed")
    if report.get("checkpoint", {}).get("sha256") != sha256_file(checkpoint_path):
        raise ValueError("checkpoint differs from the already-opened panel lineage")
    by_name = {str(record["filename"]): record for record in manifest["splits"]["train"]}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ValueError(f"reuse panel contains manifest-unknown sources: {missing[:3]}")
    return report, tuple(by_name[name] for name in names), seed, draws


def _load_coordinate_model(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> tuple[AbsoluteCoordinateSorter, dict[str, Any], Path]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    contract = checkpoint.get("contract", {})
    if contract.get("architecture") != "socket-backed-absolute-coordinate-sorter-v1":
        raise ValueError("unsupported absolute-coordinate checkpoint architecture")
    socket_metadata = checkpoint.get("socket_checkpoint", {})
    socket_path = Path(str(socket_metadata.get("path", "")))
    if not socket_path.is_file():
        raise FileNotFoundError(f"socket checkpoint is unavailable: {socket_path}")
    if socket_metadata.get("sha256") != sha256_file(socket_path):
        raise ValueError("socket checkpoint digest differs from coordinate lineage")
    backbone, _ = load_socket_backbone(socket_path, device)
    model = AbsoluteCoordinateSorter(
        backbone,
        grid=int(contract["grid"]),
        head_dimension=int(contract["head_dimension"]),
        heads=int(contract["heads"]),
        set_layers=int(contract["set_layers"]),
        sinkhorn_iterations=int(contract["sinkhorn_iterations"]),
        freeze_backbone=bool(contract["frozen_socket_backbone"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, contract, socket_path


def _bootstrap_exact_delta(
    boards: list[dict[str, Any]],
    *,
    candidate: str,
    seed: int,
    samples: int = 200_000,
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = {}
    for board in boards:
        delta = float(board["global"][candidate]["correct_tile_count"]) - float(
            board["global"][BASELINE]["correct_tile_count"]
        )
        grouped.setdefault(str(board["source_filename"]), []).append(delta)
    source_delta = np.asarray(
        [float(np.mean(values)) for values in grouped.values()],
        dtype=np.float64,
    )
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(source_delta), size=(samples, len(source_delta)))
    values = source_delta[indices].mean(axis=1)
    return {
        "source_count": len(source_delta),
        "case_count": len(boards),
        "mean_delta_per_board": float(source_delta.mean()),
        "source_cluster_bootstrap_ci95": [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ],
        "bootstrap_samples": samples,
        "seed": seed,
    }


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint_path = args.checkpoint.resolve()
    replay_path = args.reuse_panel_report.resolve()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    replay, records, seed, draws = _load_replay_contract(
        replay_path,
        checkpoint_path,
        manifest,
    )
    model, contract, socket_path = _load_coordinate_model(checkpoint_path, device=device)
    clean_boards = prepare_clean_boards(records, args.targets.resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    predictions: list[dict[str, Any]] = []
    references: list[np.ndarray] = []
    arrays: dict[str, np.ndarray] = {}
    started = perf_counter()
    for source_index, board in enumerate(clean_boards):
        for draw in range(draws):
            case_seed = seed + 100_000 + source_index * draws + draw
            generator = np.random.default_rng(case_seed)
            torch.manual_seed(case_seed)
            tiles, _, reference = synthetic_example(board, generator=generator, device=device)
            case_started = perf_counter()
            output = model(tiles)
            right = output.socket_output.right_log_assignment[0].float().cpu().numpy()
            down = output.socket_output.down_log_assignment[0].float().cpu().numpy()
            base = decode_socket_assignments(
                right,
                down,
                grid=GRID,
                config=SocketDecoderConfig(
                    component_edge_budget_per_axis=144,
                    swap_edge_budget_per_axis=144,
                    max_swap_steps=24,
                ),
            )
            border = select_global_cyclic_translation(
                base.layout,
                right,
                down,
                grid=GRID,
                config=CyclicTranslationConfig(border_weight=5.0),
            )
            row = output.row_logits[0].float().cpu().numpy()
            column = output.column_logits[0].float().cpu().numpy()
            layouts: dict[str, np.ndarray] = {
                "socket_ot_decoder144": np.ascontiguousarray(base.layout),
                BASELINE: np.ascontiguousarray(border.layout),
            }
            diagnostics: dict[str, Any] = {BASELINE: border.report()}
            for name, config in ARM_CONFIGS.items():
                candidate = select_coordinate_cyclic_translation(
                    base.layout,
                    row,
                    column,
                    right_log_assignment=right,
                    down_log_assignment=down,
                    grid=GRID,
                    config=config,
                )
                layouts[name] = candidate.layout
                diagnostics[name] = candidate.report()
            # The Socket-only specialisation of the new profile code must be
            # exactly equivalent to the independently frozen border5 placer.
            socket_profile = select_coordinate_cyclic_translation(
                base.layout,
                row,
                column,
                right_log_assignment=right,
                down_log_assignment=down,
                grid=GRID,
                config=CoordinateCyclicConfig(
                    row_coordinate_weight=0.0,
                    row_socket_weight=1.0,
                    column_coordinate_weight=0.0,
                    column_socket_weight=1.0,
                ),
            )
            if not np.array_equal(socket_profile.layout, border.layout):
                raise RuntimeError("Socket profile decomposition differs from frozen border5")
            prefix = f"case_{len(predictions):04d}"
            for name, layout in layouts.items():
                arrays[f"{prefix}__layout__{name}"] = layout
            references.append(reference)
            predictions.append(
                {
                    "array_prefix": prefix,
                    "source_filename": board.filename,
                    "draw_index": draw,
                    "case_seed": case_seed,
                    "layout_variants": list(layouts),
                    "diagnostics": diagnostics,
                    "runtime_seconds": perf_counter() - case_started,
                }
            )
            print(
                f"froze {len(predictions)}/{len(clean_boards) * draws} "
                f"{board.filename} draw={draw}",
                flush=True,
            )

    artifact_path = output_dir / "frozen_predictions.npz"
    np.savez_compressed(artifact_path, **arrays)
    metadata_path = output_dir / "frozen_predictions.json"
    metadata_path.write_text(
        json.dumps(
            {
                "schema": "aiijc-coordinate-cyclic-origin-development-frozen-v1",
                "contains_exact_references": False,
                "contains_clean_pixels": False,
                "cases": predictions,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    frozen = np.load(artifact_path, allow_pickle=False)
    boards: list[dict[str, Any]] = []
    for index, (prediction, reference) in enumerate(zip(predictions, references, strict=True)):
        prefix = str(prediction["array_prefix"])
        global_metrics: dict[str, Any] = {}
        strict: dict[str, bool] = {}
        for name in prediction["layout_variants"]:
            layout = frozen[f"{prefix}__layout__{name}"]
            strict[name] = bool(np.array_equal(np.sort(layout), np.arange(GRID * GRID)))
            global_metrics[name] = evaluate_layout(
                layout,
                reference,
                reference_is_exact=True,
            ).as_dict()
        if not all(strict.values()):
            raise RuntimeError("a frozen cyclic candidate is not a strict permutation")
        boards.append(
            {
                "case_index": index,
                "source_filename": prediction["source_filename"],
                "draw_index": prediction["draw_index"],
                "case_seed": prediction["case_seed"],
                "global": global_metrics,
                "strict_permutation": strict,
            }
        )

    variant_names = list(boards[0]["global"])
    global_mean = {
        name: _numeric_mean([board["global"][name] for board in boards])
        for name in variant_names
    }
    development: dict[str, Any] = {}
    passing: list[str] = []
    for index, name in enumerate(ARM_CONFIGS):
        exact_delta = (
            global_mean[name]["correct_tile_count"]
            - global_mean[BASELINE]["correct_tile_count"]
        )
        adjacency_loss = 100.0 * (
            global_mean[BASELINE]["adjacency"] - global_mean[name]["adjacency"]
        )
        strict = all(board["strict_permutation"][name] for board in boards)
        passed = bool(
            exact_delta >= MATERIAL_EXACT_DELTA
            and adjacency_loss <= MAX_ADJACENCY_LOSS_PERCENTAGE_POINTS
            and strict
        )
        development[name] = {
            "configuration": as_jsonable_config(ARM_CONFIGS[name]),
            "mean_exact_delta_vs_border5": exact_delta,
            "adjacency_loss_percentage_points_vs_border5": adjacency_loss,
            "strict_permutation": strict,
            "source_clustered_exact_delta": _bootstrap_exact_delta(
                boards,
                candidate=name,
                seed=seed + 501 + index,
            ),
            "passed_material_development_gate": passed,
        }
        if passed:
            passing.append(name)
    selected = (
        sorted(
            passing,
            key=lambda name: (
                -global_mean[name]["correct_tile_count"],
                -global_mean[name]["adjacency"],
                name,
            ),
        )[0]
        if passing
        else None
    )
    report = {
        "experiment": "absolute-coordinate-global-cyclic-translation-v1",
        "status": (
            "opened-panel-development-passed-one-candidate-frozen"
            if selected is not None
            else "opened-panel-development-material-gate-failed-no-fresh-panel"
        ),
        "contract": {
            "primitive": "enumerate all global cyclic translations only",
            "input_layout": "strict decoder144 tile-at-position permutation",
            "coordinate_evidence": (
                "per-tile row/column log-softmax; sum/mean-equivalent score; "
                "one profile-global positive standardisation only"
            ),
            "socket_evidence": "optional independently frozen cut/border objective weight 5",
            "state_dict_changed": False,
            "accepts_original_or_transpose_averaged_logits_in_original_frame": True,
            "tile_replacement_or_warp": False,
        },
        "protocol": {
            "manifest_digest": compute_protocol_digest(manifest),
            "already_opened_panel_only": True,
            "reused_report": str(replay_path),
            "reused_report_sha256": sha256_file(replay_path),
            "reused_source_digest": replay["selection"]["eval_digest"],
            "dirty_only_predictions_frozen_before_reference_scoring": True,
            "frozen_artifact_contains_exact_references": False,
            "development_arm_grid_preregistered_before_replay": list(ARM_CONFIGS),
            "fresh_panel_opened": False,
            "calibration_opened": False,
            "holdout_opened": False,
            "competition_test_opened": False,
        },
        "selection": {
            "eval_filenames": [record["filename"] for record in records],
            "eval_digest": _names_digest([str(record["filename"]) for record in records]),
            "eval_source_count": len(records),
            "draws_per_source": draws,
            "seed": seed,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "socket_path": str(socket_path.resolve()),
            "socket_sha256": sha256_file(socket_path),
            "coordinate_contract": contract,
        },
        "frozen_predictions": {
            "arrays_path": str(artifact_path),
            "arrays_sha256": sha256_file(artifact_path),
            "metadata_path": str(metadata_path),
            "metadata_sha256": sha256_file(metadata_path),
        },
        "development_gate": {
            "baseline": BASELINE,
            "requirements": {
                "mean_exact_delta_per_board_at_least": MATERIAL_EXACT_DELTA,
                "adjacency_loss_percentage_points_at_most": (
                    MAX_ADJACENCY_LOSS_PERCENTAGE_POINTS
                ),
                "strict_tile_permutation": True,
                "freeze_at_most_one_candidate": True,
            },
            "arms": development,
            "selected_candidate": selected,
            "passed": selected is not None,
        },
        "predeclared_fresh_gate": {
            "eligible_only_if_development_gate_passed": True,
            "selected_candidate_must_be_unchanged": True,
            "matched_baseline": BASELINE,
            "requirements": {
                "mean_exact_delta_per_board_at_least": MATERIAL_EXACT_DELTA,
                "source_clustered_exact_ci95_lower_strictly_above": 0.0,
                "adjacency_loss_percentage_points_at_most": (
                    MAX_ADJACENCY_LOSS_PERCENTAGE_POINTS
                ),
                "strict_tile_permutation": True,
            },
            "opened": False,
        },
        "evaluation": {
            "reference": "known synthetic inverse shuffle on reused development panel",
            "case_count": len(boards),
            "global_mean": global_mean,
            "boards": boards,
        },
        "runtime_seconds": {
            "total": perf_counter() - started,
            "mean_per_case": float(np.mean([row["runtime_seconds"] for row in predictions])),
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
                "global_mean": global_mean,
                "selected_candidate": selected,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def as_jsonable_config(config: CoordinateCyclicConfig) -> dict[str, float]:
    return {
        "row_coordinate_weight": config.row_coordinate_weight,
        "row_socket_weight": config.row_socket_weight,
        "column_coordinate_weight": config.column_coordinate_weight,
        "column_socket_weight": config.column_socket_weight,
        "socket_border_weight": config.socket_border_weight,
    }


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate direct-residual ordering on the already-opened Union-v2 fresh64 panel.

This is an engineering conversion test, not fresh promotion evidence.  It adds
the frozen direct-hard-edge model's learned residual to the confidence of the
same edge when that edge survives the Union-v2 hard projection.  New Union
edges retain their own confidence unchanged.  Both layouts and all edge
priorities are persisted before exact synthetic references are scored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from aiijc_puzzle.direct_hard_edge_production import (
    FROZEN_DIRECT_HARD_EDGE_SHA256,
    infer_direct_hard_edge_priorities,
    load_direct_hard_edge_checkpoint,
)
from aiijc_puzzle.direct_residual_union_priority import (
    build_direct_rank_delta_union_priority,
    build_direct_residual_union_priority,
)
from aiijc_puzzle.protocol import compute_protocol_digest, sha256_file
from aiijc_puzzle.raw_twin_union_production import (
    CYCLIC_BORDER_WEIGHT,
    FROZEN_UNION_CHECKPOINT_SHA256,
    infer_raw_twin_union_assignments,
    load_fullres_twin_checkpoint,
    load_raw_twin_union_checkpoint,
)
from aiijc_puzzle.socket_decoder import (
    SocketDecoderConfig,
    decode_socket_assignments,
    hard_partial_axis_matching,
)
from aiijc_puzzle.socket_sorter_production import (
    DECODER_EDGE_BUDGET,
    DECODER_SWAP_STEPS,
    load_socket_checkpoint,
)
from aiijc_puzzle.socket_translation_placer import (
    CyclicTranslationConfig,
    select_global_cyclic_translation,
)

try:
    from scripts.run_fullres_twin_side_matcher import (
        _atomic_json,
        _prepare_boards,
        _two_view_case,
    )
    from scripts.run_raw_twin_union_reranker_fresh64 import (
        DEFAULT_CONFIG,
        DEFAULT_MANIFEST,
        DEFAULT_TARGETS,
        SOCKET_CHECKPOINT,
        TWIN_CHECKPOINT,
        UNION_CHECKPOINT,
        UNION_SELECTION,
        _case_seeds,
        load_config,
        source_clustered_ci,
    )
    from scripts.run_raw_twin_union_reranker_v2 import _adjacency_fraction
except ModuleNotFoundError:  # Direct ``python scripts/run_*.py`` execution.
    from run_fullres_twin_side_matcher import _atomic_json, _prepare_boards, _two_view_case
    from run_raw_twin_union_reranker_fresh64 import (
        DEFAULT_CONFIG,
        DEFAULT_MANIFEST,
        DEFAULT_TARGETS,
        SOCKET_CHECKPOINT,
        TWIN_CHECKPOINT,
        UNION_CHECKPOINT,
        UNION_SELECTION,
        _case_seeds,
        load_config,
        source_clustered_ci,
    )
    from run_raw_twin_union_reranker_v2 import _adjacency_fraction


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UNION_CONFIG = PROJECT_ROOT / "configs/raw_twin_union_reranker_v2_preregistered.json"
DIRECT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/direct-hard-edge-priority/v1-fit256-s600-d1-32-cpu"
    / "direct_hard_edge_priority.pt"
)
FROZEN_PANEL = PROJECT_ROOT / "outputs/raw-twin-union-reranker/frozen-v2-fresh64-draw0"
FROZEN_PREDICTIONS = FROZEN_PANEL / "frozen-target-free-predictions.npz"
FROZEN_METADATA = FROZEN_PANEL / "frozen-target-free-predictions.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/direct-residual-union-priority/opened64-v1"
GRID = 24
COUNT = GRID * GRID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument(
        "--method",
        choices=("additive-residual", "rank-delta"),
        default="additive-residual",
    )
    return parser.parse_args()


def _strict_layout(value: Any) -> np.ndarray:
    layout = np.ascontiguousarray(value, dtype=np.int32)
    if layout.shape != (COUNT,) or not np.array_equal(np.sort(layout), np.arange(COUNT)):
        raise ValueError("layout is not a strict 576-tile permutation")
    return layout


def _cached_layout(archive: Any, prefix: str) -> np.ndarray:
    key = f"{prefix}__learned_union__layout"
    if key not in archive:
        raise KeyError(f"frozen archive is missing {key}")
    return _strict_layout(archive[key])


def _decode_layout(
    right: np.ndarray,
    down: np.ndarray,
    *,
    component_edge_priority: dict[str, np.ndarray] | None,
) -> np.ndarray:
    decoder = decode_socket_assignments(
        right,
        down,
        grid=GRID,
        config=SocketDecoderConfig(
            component_edge_budget_per_axis=DECODER_EDGE_BUDGET,
            swap_edge_budget_per_axis=DECODER_EDGE_BUDGET,
            max_swap_steps=DECODER_SWAP_STEPS,
        ),
        component_edge_priority=component_edge_priority,
    )
    cyclic = select_global_cyclic_translation(
        decoder.layout,
        right,
        down,
        grid=GRID,
        config=CyclicTranslationConfig(border_weight=CYCLIC_BORDER_WEIGHT),
    )
    return _strict_layout(cyclic.layout)


def _edge_arrays(
    right: np.ndarray,
    down: np.ndarray,
    priorities: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for axis_index, (axis, assignment) in enumerate((("right", right), ("down", down))):
        matching = hard_partial_axis_matching(assignment, grid=GRID, axis=axis)
        source = np.asarray([edge.source for edge in matching.edges], dtype=np.int32)
        target = np.asarray([edge.target for edge in matching.edges], dtype=np.int32)
        confidence = np.asarray([edge.confidence for edge in matching.edges], dtype=np.float64)
        treatment = np.asarray(priorities[axis][source, target], dtype=np.float64)
        if (
            source.shape != (GRID * (GRID - 1),)
            or not np.isfinite(confidence).all()
            or not np.isfinite(treatment).all()
        ):
            raise RuntimeError("hard-edge freeze contract changed")
        result[f"axis_{axis_index}_source"] = source
        result[f"axis_{axis_index}_target"] = target
        result[f"axis_{axis_index}_baseline_priority"] = confidence
        result[f"axis_{axis_index}_treatment_priority"] = treatment
    return result


def _edge_is_correct(
    source: np.ndarray,
    target: np.ndarray,
    *,
    axis: int,
    reference: np.ndarray,
) -> np.ndarray:
    position = np.empty(COUNT, dtype=np.int32)
    position[reference] = np.arange(COUNT, dtype=np.int32)
    source_position = position[source]
    target_position = position[target]
    if axis == 0:
        return (target_position == source_position + 1) & (
            source_position % GRID != GRID - 1
        )
    return target_position == source_position + GRID


def _fixed_top144_correct(
    archive: Any,
    prefix: str,
    reference: np.ndarray,
    *,
    arm: str,
) -> int:
    total = 0
    for axis in (0, 1):
        source = np.asarray(archive[f"{prefix}__axis_{axis}_source"], dtype=np.int32)
        target = np.asarray(archive[f"{prefix}__axis_{axis}_target"], dtype=np.int32)
        priority = np.asarray(
            archive[f"{prefix}__axis_{axis}_{arm}_priority"],
            dtype=np.float64,
        )
        order = np.argsort(-priority, kind="stable")[:DECODER_EDGE_BUDGET]
        truth = _edge_is_correct(source, target, axis=axis, reference=reference)
        total += int(np.count_nonzero(truth[order]))
    return total


def _mean(rows: list[dict[str, Any]], arm: str, metric: str) -> float:
    return float(np.mean([float(row[arm][metric]) for row in rows]))


def run(args: argparse.Namespace) -> None:
    if not 1 <= args.limit <= 64:
        raise ValueError("limit must be in [1, 64]")
    config, config_sha = load_config(args.config)
    treatment_name = (
        "residual_transfer"
        if args.method == "additive-residual"
        else "rank_delta_transfer"
    )
    metadata = json.loads(FROZEN_METADATA.read_text(encoding="utf-8"))
    cases = metadata.get("cases")
    if not isinstance(cases, list) or len(cases) != 64:
        raise ValueError("expected the established frozen64 case metadata")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("protocol_digest") != compute_protocol_digest(manifest):
        raise ValueError("manifest protocol digest mismatch")
    names = tuple(config["selection"]["source_filenames"][: args.limit])
    case_names = tuple(str(case["source_filename"]) for case in cases[: args.limit])
    if names != case_names:
        raise ValueError("config and frozen metadata source order differ")
    lookup = {str(record["filename"]): dict(record) for record in manifest["splits"]["train"]}
    records = tuple(lookup[name] for name in names)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "frozen-target-free-layouts.npz"
    metadata_path = output_dir / "frozen-target-free-layouts.json"
    report_path = output_dir / "report.json"
    if any(path.exists() for path in (prediction_path, metadata_path, report_path)):
        raise FileExistsError("refusing to overwrite a residual-transfer run")

    boards = _prepare_boards(records, args.targets)
    device = torch.device("cpu")
    socket = load_socket_checkpoint(SOCKET_CHECKPOINT, device=device)
    twin = load_fullres_twin_checkpoint(TWIN_CHECKPOINT, device=device)
    union = load_raw_twin_union_checkpoint(
        UNION_CHECKPOINT,
        config_path=UNION_CONFIG,
        selection_path=UNION_SELECTION,
        device=device,
    )
    direct = load_direct_hard_edge_checkpoint(DIRECT_CHECKPOINT, device=device)
    if union.sha256 != FROZEN_UNION_CHECKPOINT_SHA256:
        raise ValueError("Union checkpoint identity changed")
    if direct.sha256 != FROZEN_DIRECT_HARD_EDGE_SHA256:
        raise ValueError("direct-hard-edge checkpoint identity changed")

    arrays: dict[str, np.ndarray] = {}
    frozen_rows: list[dict[str, Any]] = []
    started = perf_counter()
    synthetic_seed = int(config["selection"]["synthetic_seed"])
    with np.load(FROZEN_PREDICTIONS) as frozen_archive, torch.inference_mode():
        for index, (case, board) in enumerate(zip(cases[: args.limit], boards, strict=True)):
            corruption_seed, permutation_seed = _case_seeds(synthetic_seed, board.filename)
            dirty, _, _ = _two_view_case(
                board.tiles,
                first_seed=corruption_seed,
                second_seed=corruption_seed + 1,
                permutation_seed=permutation_seed,
            )
            union_inference = infer_raw_twin_union_assignments(
                dirty,
                socket,
                twin,
                union,
                device=device,
            )
            direct_inference = infer_direct_hard_edge_priorities(
                dirty,
                socket,
                direct,
                device=device,
            )
            priority_builder = (
                build_direct_residual_union_priority
                if args.method == "additive-residual"
                else build_direct_rank_delta_union_priority
            )
            priority = priority_builder(
                union_inference.learned_right_log_assignment,
                union_inference.learned_down_log_assignment,
                direct_source=direct_inference.source,
                direct_target=direct_inference.target,
                direct_axis=direct_inference.axis,
                direct_raw_scores=direct_inference.raw_scores,
                direct_learned_scores=direct_inference.learned_scores,
                grid=GRID,
            )
            prefix = str(case["prefix"])
            baseline = _decode_layout(
                union_inference.learned_right_log_assignment,
                union_inference.learned_down_log_assignment,
                component_edge_priority=None,
            )
            cached = _cached_layout(frozen_archive, prefix)
            if not np.array_equal(baseline, cached):
                raise RuntimeError("regenerated Union-v2 baseline differs from frozen layout")
            treatment = _decode_layout(
                union_inference.learned_right_log_assignment,
                union_inference.learned_down_log_assignment,
                component_edge_priority=priority.component_edge_priority,
            )
            arrays[f"{prefix}__union_v2_layout"] = baseline
            arrays[f"{prefix}__{treatment_name}_layout"] = treatment
            for key, value in _edge_arrays(
                union_inference.learned_right_log_assignment,
                union_inference.learned_down_log_assignment,
                priority.component_edge_priority,
            ).items():
                arrays[f"{prefix}__{key}"] = value
            frozen_rows.append(
                {
                    "prefix": prefix,
                    "source_filename": board.filename,
                    "priority": priority.report(),
                }
            )
            print(
                json.dumps(
                    {
                        "event": "freeze",
                        "done": index + 1,
                        "total": args.limit,
                        "matched_direct_edges": priority.diagnostics.matched_edge_count,
                    }
                ),
                flush=True,
            )

    np.savez_compressed(prediction_path, **arrays)
    _atomic_json(
        metadata_path,
        {
            "schema": "aiijc-direct-residual-union-priority-opened64-predictions-v1",
            "panel_role": "already-opened engineering panel; not promotion evidence",
            "contains_exact_references": False,
            "contains_dirty_or_clean_pixels": False,
            "contains_target_free_strict_layouts": True,
            "treatment": (
                "Union hard-edge confidence plus frozen direct learned-minus-raw residual "
                "on identical raw hard-edge identities; zero residual otherwise"
                if args.method == "additive-residual"
                else "direct learned-minus-raw percentile-rank displacement on identical "
                "Union hard-edge identities; original Union confidence multiset preserved"
            ),
            "method": args.method,
            "cases": frozen_rows,
        },
    )
    prediction_sha = sha256_file(prediction_path)
    metadata_sha = sha256_file(metadata_path)
    print(
        json.dumps(
            {
                "event": "predictions_frozen_before_scoring",
                "sha256": prediction_sha,
            }
        ),
        flush=True,
    )

    scored_rows: list[dict[str, Any]] = []
    with np.load(prediction_path) as archive:
        for case, board in zip(cases[: args.limit], boards, strict=True):
            corruption_seed, permutation_seed = _case_seeds(
                synthetic_seed,
                board.filename,
            )
            _, _, reference = _two_view_case(
                board.tiles,
                first_seed=corruption_seed,
                second_seed=corruption_seed + 1,
                permutation_seed=permutation_seed,
            )
            reference = _strict_layout(reference)
            prefix = str(case["prefix"])
            baseline = _strict_layout(archive[f"{prefix}__union_v2_layout"])
            treatment = _strict_layout(archive[f"{prefix}__{treatment_name}_layout"])
            scored_rows.append(
                {
                    "source_filename": str(case["source_filename"]),
                    "union_v2": {
                        "exact_tiles": int(np.count_nonzero(baseline == reference)),
                        "adjacency": float(_adjacency_fraction(baseline, reference)),
                        "top144_correct": _fixed_top144_correct(
                            archive,
                            prefix,
                            reference,
                            arm="baseline",
                        ),
                    },
                    treatment_name: {
                        "exact_tiles": int(np.count_nonzero(treatment == reference)),
                        "adjacency": float(_adjacency_fraction(treatment, reference)),
                        "top144_correct": _fixed_top144_correct(
                            archive,
                            prefix,
                            reference,
                            arm="treatment",
                        ),
                    },
                }
            )

    def deltas(metric: str) -> list[float]:
        return [
            float(row[treatment_name][metric]) - float(row["union_v2"][metric])
            for row in scored_rows
        ]

    metrics = {
        "arms": {
            arm: {
                metric: _mean(scored_rows, arm, metric)
                for metric in ("exact_tiles", "adjacency", "top144_correct")
            }
            for arm in ("union_v2", treatment_name)
        },
        "exact_delta": source_clustered_ci(deltas("exact_tiles"), seed=20330993),
        "adjacency_delta": source_clustered_ci(deltas("adjacency"), seed=20330994),
        "top144_correct_delta": source_clustered_ci(
            deltas("top144_correct"),
            seed=20330995,
        ),
        "strict_boards": args.limit,
        "matched_direct_edges_per_board": float(
            np.mean(
                [
                    row["priority"]["diagnostics"]["matched_edge_count"]
                    for row in frozen_rows
                ]
            )
        ),
    }
    gate = {
        "fixed_top144_gain_at_least_half_edge": float(
            metrics["top144_correct_delta"]["mean"]
        )
        >= 0.5,
        "adjacency_nonnegative": float(metrics["adjacency_delta"]["mean"]) >= 0.0,
        "exact_nonnegative": float(metrics["exact_delta"]["mean"]) >= 0.0,
        "all_strict": metrics["strict_boards"] == args.limit,
        "passed": False,
    }
    gate["passed"] = all(value for key, value in gate.items() if key != "passed")
    _atomic_json(
        report_path,
        {
            "schema": "aiijc-direct-residual-union-priority-opened64-report-v1",
            "method": args.method,
            "status": "engineering-gate-pass" if gate["passed"] else "engineering-gate-fail",
            "panel_role": "already-opened engineering panel; not promotion evidence",
            "frozen_union_config_sha256": config_sha,
            "frozen_union_checkpoint_sha256": union.sha256,
            "frozen_direct_checkpoint_sha256": direct.sha256,
            "predictions": {
                "path": str(prediction_path.relative_to(PROJECT_ROOT)),
                "sha256": prediction_sha,
                "metadata_path": str(metadata_path.relative_to(PROJECT_ROOT)),
                "metadata_sha256": metadata_sha,
                "frozen_before_reference_scoring": True,
                "contains_exact_references": False,
            },
            "metrics": metrics,
            "gate": gate,
            "rows": scored_rows,
            "runtime_seconds": perf_counter() - started,
            "organizer_holdout_or_test_opened": False,
            "original_upright_tile_permutations_only": True,
            "weight_or_budget_sweep": False,
        },
    )
    print(
        json.dumps(
            {
                "event": "complete",
                "report": str(report_path),
                "report_sha256": sha256_file(report_path),
                "metrics": metrics,
                "gate": gate,
            }
        ),
        flush=True,
    )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

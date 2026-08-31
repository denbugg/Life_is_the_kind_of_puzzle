#!/usr/bin/env python3
"""Freeze and score a conservative Union/rank-delta whole-layout selector.

This runner reuses the established rank-delta opened64 layouts and their hard
edge priorities.  It regenerates the two model inferences only to reproduce
the pre-packing translation-component evidence for both arms.  The fixed
target-blind selector lexicographically compares
``(consistent_redundant_constraints, largest_component)`` and retains Union
on exact ties.

All decisions, selected strict layouts and selected edge priorities are saved
and hashed before synthetic references are recreated.  The panel has already
been opened and is therefore engineering evidence only, not a fresh promotion
gate.  There is no selector, threshold, budget or weight sweep in this runner.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
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
from aiijc_puzzle.direct_rank_delta_component_selector import (
    ComponentConsistencyEvidence,
    select_direct_rank_delta_component_arm,
)
from aiijc_puzzle.direct_residual_union_priority import (
    build_direct_rank_delta_union_priority,
)
from aiijc_puzzle.protocol import compute_protocol_digest, sha256_file
from aiijc_puzzle.raw_twin_union_production import (
    FROZEN_UNION_CHECKPOINT_SHA256,
    infer_raw_twin_union_assignments,
    load_fullres_twin_checkpoint,
    load_raw_twin_union_checkpoint,
)
from aiijc_puzzle.socket_decoder import build_translation_components
from aiijc_puzzle.socket_sorter_production import (
    DECODER_EDGE_BUDGET,
    load_socket_checkpoint,
)

try:
    from scripts.run_direct_residual_union_priority_opened64 import (
        COUNT,
        DIRECT_CHECKPOINT,
        FROZEN_METADATA,
        GRID,
        PROJECT_ROOT,
        UNION_CONFIG,
        _fixed_top144_correct,
        _strict_layout,
    )
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
    from run_direct_residual_union_priority_opened64 import (
        COUNT,
        DIRECT_CHECKPOINT,
        FROZEN_METADATA,
        GRID,
        PROJECT_ROOT,
        UNION_CONFIG,
        _fixed_top144_correct,
        _strict_layout,
    )
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


RANK_DELTA_PANEL = PROJECT_ROOT / "outputs/direct-residual-union-priority/rank-delta-opened64-v1"
RANK_DELTA_PREDICTIONS = RANK_DELTA_PANEL / "frozen-target-free-layouts.npz"
RANK_DELTA_METADATA = RANK_DELTA_PANEL / "frozen-target-free-layouts.json"
RANK_DELTA_REPORT = RANK_DELTA_PANEL / "report.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/direct-rank-delta-component-selector/opened64-v1"
ARM_NAMES = ("union_v2", "rank_delta_transfer", "component_selector")
METRIC_NAMES = ("exact_tiles", "adjacency", "top144_correct")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=64)
    return parser.parse_args()


def _component_evidence(
    status_counts: Mapping[str, int],
    components: Sequence[Mapping[int, tuple[int, int]]],
    *,
    tile_count: int,
) -> ComponentConsistencyEvidence:
    """Convert one pre-packing component build into selector evidence."""

    if "consistent" not in status_counts:
        raise ValueError("component status counts are missing 'consistent'")
    sizes = [len(component) for component in components]
    if not sizes or sum(sizes) != tile_count:
        raise ValueError("translation components must partition every tile")
    return ComponentConsistencyEvidence(
        consistent_redundant_constraints=int(status_counts["consistent"]),
        largest_component=max(sizes),
        tile_count=tile_count,
    )


def _mean(rows: list[dict[str, Any]], arm: str, metric: str) -> float:
    return float(np.mean([float(row[arm][metric]) for row in rows]))


def _report_path(path: Path) -> str:
    """Prefer a project-relative artifact path but support temporary smoke runs."""

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _deltas(
    rows: list[dict[str, Any]],
    *,
    treatment: str,
    baseline: str,
    metric: str,
) -> list[float]:
    return [float(row[treatment][metric]) - float(row[baseline][metric]) for row in rows]


def _win_tie_loss(values: Sequence[float]) -> dict[str, int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("values must be a finite non-empty vector")
    return {
        "wins": int(np.count_nonzero(array > 0)),
        "ties": int(np.count_nonzero(array == 0)),
        "losses": int(np.count_nonzero(array < 0)),
    }


def _validate_rank_delta_artifact(*, limit: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report = json.loads(RANK_DELTA_REPORT.read_text(encoding="utf-8"))
    metadata = json.loads(RANK_DELTA_METADATA.read_text(encoding="utf-8"))
    if report.get("method") != "rank-delta":
        raise ValueError("established artifact is not the rank-delta arm")
    predictions = report.get("predictions", {})
    if predictions.get("sha256") != sha256_file(RANK_DELTA_PREDICTIONS):
        raise ValueError("rank-delta predictions SHA mismatch")
    if predictions.get("metadata_sha256") != sha256_file(RANK_DELTA_METADATA):
        raise ValueError("rank-delta metadata SHA mismatch")
    cases = metadata.get("cases")
    if not isinstance(cases, list) or len(cases) < limit:
        raise ValueError("rank-delta metadata does not cover the requested panel")
    return report, [dict(case) for case in cases[:limit]]


def _comparison_metrics(
    rows: list[dict[str, Any]],
    *,
    treatment: str,
    baseline: str,
    seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for offset, metric in enumerate(METRIC_NAMES):
        values = _deltas(
            rows,
            treatment=treatment,
            baseline=baseline,
            metric=metric,
        )
        result[f"{metric}_delta"] = source_clustered_ci(values, seed=seed + offset)
        result[f"{metric}_win_tie_loss"] = _win_tie_loss(values)
    return result


def run(args: argparse.Namespace) -> None:
    if not 1 <= args.limit <= 64:
        raise ValueError("limit must be in [1, 64]")

    config, config_sha = load_config(args.config)
    established_metadata = json.loads(FROZEN_METADATA.read_text(encoding="utf-8"))
    established_cases = established_metadata.get("cases")
    if not isinstance(established_cases, list) or len(established_cases) != 64:
        raise ValueError("expected the established frozen64 Union metadata")
    rank_report, rank_cases = _validate_rank_delta_artifact(limit=args.limit)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("protocol_digest") != compute_protocol_digest(manifest):
        raise ValueError("manifest protocol digest mismatch")
    names = tuple(config["selection"]["source_filenames"][: args.limit])
    union_names = tuple(str(case["source_filename"]) for case in established_cases[: args.limit])
    rank_names = tuple(str(case["source_filename"]) for case in rank_cases)
    if names != union_names or names != rank_names:
        raise ValueError("config, Union artifact and rank-delta artifact source order differ")
    lookup = {str(record["filename"]): dict(record) for record in manifest["splits"]["train"]}
    records = tuple(lookup[name] for name in names)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "frozen-target-free-layouts.npz"
    metadata_path = output_dir / "frozen-target-free-layouts.json"
    report_path = output_dir / "report.json"
    if any(path.exists() for path in (prediction_path, metadata_path, report_path)):
        raise FileExistsError("refusing to overwrite a component-selector run")

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
    with np.load(RANK_DELTA_PREDICTIONS) as established, torch.inference_mode():
        for index, (case, rank_case, board) in enumerate(
            zip(established_cases[: args.limit], rank_cases, boards, strict=True)
        ):
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
            rank_priority = build_direct_rank_delta_union_priority(
                union_inference.learned_right_log_assignment,
                union_inference.learned_down_log_assignment,
                direct_source=direct_inference.source,
                direct_target=direct_inference.target,
                direct_axis=direct_inference.axis,
                direct_raw_scores=direct_inference.raw_scores,
                direct_learned_scores=direct_inference.learned_scores,
                grid=GRID,
            )
            if rank_priority.report() != rank_case.get("priority"):
                raise RuntimeError("regenerated rank-delta priority differs from frozen artifact")

            baseline_build = build_translation_components(
                union_inference.learned_right_log_assignment,
                union_inference.learned_down_log_assignment,
                grid=GRID,
                edge_budget_per_axis=DECODER_EDGE_BUDGET,
            )
            treatment_build = build_translation_components(
                union_inference.learned_right_log_assignment,
                union_inference.learned_down_log_assignment,
                grid=GRID,
                edge_budget_per_axis=DECODER_EDGE_BUDGET,
                component_edge_priority=rank_priority.component_edge_priority,
            )
            decision = select_direct_rank_delta_component_arm(
                _component_evidence(
                    baseline_build.status_counts,
                    baseline_build.components,
                    tile_count=COUNT,
                ),
                _component_evidence(
                    treatment_build.status_counts,
                    treatment_build.components,
                    tile_count=COUNT,
                ),
            )

            prefix = str(case["prefix"])
            baseline = _strict_layout(established[f"{prefix}__union_v2_layout"])
            treatment = _strict_layout(established[f"{prefix}__rank_delta_transfer_layout"])
            selected = treatment if decision.treatment_selected else baseline
            arrays[f"{prefix}__union_v2_layout"] = baseline
            arrays[f"{prefix}__rank_delta_transfer_layout"] = treatment
            arrays[f"{prefix}__component_selector_layout"] = selected
            arrays[f"{prefix}__selected_arm_index"] = np.asarray(
                1 if decision.treatment_selected else 0,
                dtype=np.int8,
            )
            for axis in (0, 1):
                source = np.asarray(
                    established[f"{prefix}__axis_{axis}_source"],
                    dtype=np.int32,
                )
                target = np.asarray(
                    established[f"{prefix}__axis_{axis}_target"],
                    dtype=np.int32,
                )
                baseline_priority = np.asarray(
                    established[f"{prefix}__axis_{axis}_baseline_priority"],
                    dtype=np.float64,
                )
                treatment_priority = np.asarray(
                    established[f"{prefix}__axis_{axis}_treatment_priority"],
                    dtype=np.float64,
                )
                selected_priority = (
                    treatment_priority if decision.treatment_selected else baseline_priority
                )
                arrays[f"{prefix}__axis_{axis}_source"] = source
                arrays[f"{prefix}__axis_{axis}_target"] = target
                arrays[f"{prefix}__axis_{axis}_union_v2_priority"] = baseline_priority
                arrays[f"{prefix}__axis_{axis}_rank_delta_transfer_priority"] = treatment_priority
                arrays[f"{prefix}__axis_{axis}_component_selector_priority"] = selected_priority

            frozen_rows.append(
                {
                    "prefix": prefix,
                    "source_filename": board.filename,
                    "selection": decision.report(),
                }
            )
            print(
                json.dumps(
                    {
                        "event": "freeze",
                        "done": index + 1,
                        "total": args.limit,
                        "selected_arm": decision.selected_arm,
                        "reason": decision.reason,
                    }
                ),
                flush=True,
            )

    np.savez_compressed(prediction_path, **arrays)
    selection_counts = {
        arm: sum(row["selection"]["selected_arm"] == arm for row in frozen_rows)
        for arm in ("union_v2", "rank_delta_transfer")
    }
    reason_counts = {
        reason: sum(row["selection"]["reason"] == reason for row in frozen_rows)
        for reason in (
            "more_consistent_redundant_constraints",
            "consistent_tie_larger_component",
            "union_conservative_fallback",
        )
    }
    _atomic_json(
        metadata_path,
        {
            "schema": "aiijc-direct-rank-delta-component-selector-opened64-predictions-v1",
            "panel_role": "already-opened engineering panel; not promotion evidence",
            "contains_exact_references": False,
            "contains_dirty_or_clean_pixels": False,
            "contains_target_free_strict_layouts": True,
            "selector": (
                "lexicographically maximize (consistent_redundant_constraints, "
                "largest_component); exact ties retain union_v2"
            ),
            "arm_granularity": "one complete strict layout per board",
            "selection_counts": selection_counts,
            "reason_counts": reason_counts,
            "cases": frozen_rows,
        },
    )
    prediction_sha = sha256_file(prediction_path)
    metadata_sha = sha256_file(metadata_path)
    print(
        json.dumps(
            {
                "event": "selector_decisions_and_layouts_frozen_before_scoring",
                "predictions_sha256": prediction_sha,
                "metadata_sha256": metadata_sha,
            }
        ),
        flush=True,
    )

    scored_rows: list[dict[str, Any]] = []
    with np.load(prediction_path) as archive:
        for case, board in zip(established_cases[: args.limit], boards, strict=True):
            corruption_seed, permutation_seed = _case_seeds(synthetic_seed, board.filename)
            _, _, reference = _two_view_case(
                board.tiles,
                first_seed=corruption_seed,
                second_seed=corruption_seed + 1,
                permutation_seed=permutation_seed,
            )
            reference = _strict_layout(reference)
            prefix = str(case["prefix"])
            row: dict[str, Any] = {"source_filename": str(case["source_filename"])}
            for arm in ARM_NAMES:
                layout = _strict_layout(archive[f"{prefix}__{arm}_layout"])
                row[arm] = {
                    "exact_tiles": int(np.count_nonzero(layout == reference)),
                    "adjacency": float(_adjacency_fraction(layout, reference)),
                    "top144_correct": _fixed_top144_correct(
                        archive,
                        prefix,
                        reference,
                        arm=arm,
                    ),
                }
            row["selected_arm"] = (
                "rank_delta_transfer"
                if int(archive[f"{prefix}__selected_arm_index"]) == 1
                else "union_v2"
            )
            scored_rows.append(row)

    arms = {
        arm: {metric: _mean(scored_rows, arm, metric) for metric in METRIC_NAMES}
        for arm in ARM_NAMES
    }
    versus_union = _comparison_metrics(
        scored_rows,
        treatment="component_selector",
        baseline="union_v2",
        seed=20331001,
    )
    versus_rank = _comparison_metrics(
        scored_rows,
        treatment="component_selector",
        baseline="rank_delta_transfer",
        seed=20331011,
    )
    exact_oracle_mean = float(
        np.mean(
            [
                max(row["union_v2"]["exact_tiles"], row["rank_delta_transfer"]["exact_tiles"])
                for row in scored_rows
            ]
        )
    )
    oracle_gain = exact_oracle_mean - arms["union_v2"]["exact_tiles"]
    captured_gain = arms["component_selector"]["exact_tiles"] - arms["union_v2"]["exact_tiles"]
    metrics = {
        "arms": arms,
        "component_selector_vs_union_v2": versus_union,
        "component_selector_vs_rank_delta_transfer": versus_rank,
        "two_arm_exact_oracle_mean_descriptive_only": exact_oracle_mean,
        "fraction_of_two_arm_exact_oracle_gain_captured": (
            captured_gain / oracle_gain if oracle_gain > 0 else None
        ),
        "strict_boards": args.limit,
        "selection_counts": selection_counts,
        "reason_counts": reason_counts,
    }
    gate = {
        "exact_gain_vs_rank_delta_at_least_tenth_tile": float(
            versus_rank["exact_tiles_delta"]["mean"]
        )
        >= 0.10,
        "exact_gain_vs_union_at_least_quarter_tile": float(
            versus_union["exact_tiles_delta"]["mean"]
        )
        >= 0.25,
        "adjacency_nonnegative_vs_union": float(versus_union["adjacency_delta"]["mean"]) >= 0.0,
        "all_strict": metrics["strict_boards"] == args.limit,
        "passed": False,
    }
    gate["passed"] = all(value for key, value in gate.items() if key != "passed")
    _atomic_json(
        report_path,
        {
            "schema": "aiijc-direct-rank-delta-component-selector-opened64-report-v1",
            "status": "engineering-gate-pass" if gate["passed"] else "engineering-gate-fail",
            "panel_role": "already-opened engineering panel; not promotion evidence",
            "frozen_union_config_sha256": config_sha,
            "frozen_union_checkpoint_sha256": union.sha256,
            "frozen_direct_checkpoint_sha256": direct.sha256,
            "reused_rank_delta_report_sha256": sha256_file(RANK_DELTA_REPORT),
            "reused_rank_delta_predictions_sha256": rank_report["predictions"]["sha256"],
            "predictions": {
                "path": _report_path(prediction_path),
                "sha256": prediction_sha,
                "metadata_path": _report_path(metadata_path),
                "metadata_sha256": metadata_sha,
                "selector_decisions_and_layouts_frozen_before_reference_scoring": True,
                "contains_exact_references": False,
            },
            "metrics": metrics,
            "gate": gate,
            "rows": scored_rows,
            "runtime_seconds": perf_counter() - started,
            "organizer_holdout_or_test_opened": False,
            "original_upright_tile_permutations_only": True,
            "whole_layout_arm_selection_only": True,
            "weight_or_budget_sweep": False,
            "selector_threshold_sweep": False,
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

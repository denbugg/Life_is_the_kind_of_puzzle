#!/usr/bin/env python3
"""Evaluate one fixed coordinate-only TASKA component-placement candidate.

The candidate keeps the frozen candidate edges, edge priorities, component
builder, original TASKA right/down costs, largest-first initialization, six
seed-0 coordinate-wise relocation rounds, Hungarian fill, four-arm all-bond
selector, and protected tail96.  It omits only the historical unconditional
random two-component relocation loop.

Every layout is hash-frozen before exact synthetic references are recreated.
Targets are used only by the later offline scoring pass.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np

from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.taska_edge_calibrator import TaskaEdgeCalibrator
from aiijc_puzzle.taska_layout_portfolio import select_lowest_taska_seam_cost_layout
from aiijc_puzzle.taska_monotone_components import (
    MONOTONE_ARM_NAMES,
    MONOTONE_TAIL_MAX_SWAPS,
    solve_taska_monotone_component_portfolio,
)
from aiijc_puzzle.taska_nonlinear_calibrator import TaskaNonlinearCalibrator
from aiijc_puzzle.taska_protected_tail_polish import polish_unprotected_taska_tail

try:
    from scripts import run_taska_rawlog_tail as replay
except ModuleNotFoundError:  # Direct ``python scripts/run_*.py`` execution.
    import run_taska_rawlog_tail as replay

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRID = 24
COUNT = GRID * GRID
CASE_COUNT = 32
SEARCH_ROUNDS = 6
TAIL_MAX_SWAPS = 96
TAIL_MINIMUM_GAIN = 1e-9
BOOTSTRAP_SEED = 3_201_887_031
RAW_SOLVER_SHA256 = replay.RAW_SOLVER_SHA256
REPLAY_RUNNER_SHA256 = "181ae0c24bdd4c2ed2f0459bd51ca4cfded2001f143efa45e11b2cd89fd73383"

FROZEN_SCHEMA = "aiijc-taska-monotone-components-target-free-v1"
FREEZE_SCHEMA = "aiijc-taska-monotone-components-pre-score-freeze-v1"
REPORT_SCHEMA = "aiijc-taska-monotone-components-report-v1"

PanelName = Literal["opened32", "held32", "fresh32"]
PARENT_PANEL: dict[PanelName, replay.PanelName] = {
    "opened32": "opened32",
    "held32": "held300",
    "fresh32": "fresh32",
}
PANEL_SPECS = {
    panel: replay.PANEL_SPECS[parent_panel]
    for panel, parent_panel in PARENT_PANEL.items()
}
DEFAULT_TARGETS = replay.DEFAULT_TARGETS
DEFAULT_OUTPUTS: dict[PanelName, Path] = {
    panel: PROJECT_ROOT / f"outputs/taska-monotone-components/{panel}-v1"
    for panel in PANEL_SPECS
}
EXPECTED_CONTROL_MEANS: dict[PanelName, tuple[float, float]] = {
    "opened32": (341.3125, 4.75),
    "held32": (337.5625, 3.0625),
    "fresh32": (346.0625, 1.15625),
}
SCORED_ARMS = ("control_tail96", "monotone_portfolio", "monotone_tail96")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=tuple(PANEL_SPECS), required=True)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke-one", action="store_true")
    return parser.parse_args(argv)


def _runtime_sources() -> dict[str, Path]:
    return {
        "monotone_runner": Path(__file__).resolve(),
        "monotone_module": (
            PROJECT_ROOT / "src/aiijc_puzzle/taska_monotone_components.py"
        ),
        "protected_tail": (
            PROJECT_ROOT / "src/aiijc_puzzle/taska_protected_tail_polish.py"
        ),
        "layout_portfolio": (
            PROJECT_ROOT / "src/aiijc_puzzle/taska_layout_portfolio.py"
        ),
        "four_arm_replay_helpers": PROJECT_ROOT / "scripts/run_taska_rawlog_tail.py",
        "frozen_raw_solver": (
            PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py"
        ),
    }


def _solve_target_free_case(
    task: tuple[PanelName, str],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    panel, prefix = task
    parent_panel = PARENT_PANEL[panel]
    spec = PANEL_SPECS[panel]
    helpers = replay.parent
    logistic = TaskaEdgeCalibrator.load_npz(replay.DEFAULT_LOGISTIC)
    nonlinear = TaskaNonlinearCalibrator.load_npz(replay.DEFAULT_NONLINEAR)
    with (
        np.load(PROJECT_ROOT / spec.parent_archive, allow_pickle=False) as base,
        np.load(PROJECT_ROOT / spec.priority_archive, allow_pickle=False) as priorities,
    ):
        cost_right = helpers._finite_matrix(base, f"{prefix}__cost_right")
        cost_down = helpers._finite_matrix(base, f"{prefix}__cost_down")
        edges = helpers._edges_from_archive(base, prefix)
        priority_values = helpers._case_priorities(
            parent_panel,
            prefix,
            base,
            priorities,
            cost_right,
            cost_down,
            edges,
            logistic,
            nonlinear,
        )
        candidate_priorities = {
            "logistic": priority_values["logistic"],
            "focal": priority_values["focal"],
            "nonlinear": priority_values["nonlinear"],
        }
        if tuple(candidate_priorities) != MONOTONE_ARM_NAMES[1:]:
            raise RuntimeError("candidate priority arm order changed")

        control_layouts = replay._four_arm_layouts(
            parent_panel,
            prefix,
            base,
            priorities,
            cost_right,
            cost_down,
            edges,
        )
        control_start = select_lowest_taska_seam_cost_layout(
            control_layouts,
            cost_right,
            cost_down,
            grid=GRID,
        )
        control = polish_unprotected_taska_tail(
            control_start.layout,
            cost_right,
            cost_down,
            edges,
            grid=GRID,
            max_swaps=TAIL_MAX_SWAPS,
            minimum_gain=TAIL_MINIMUM_GAIN,
        )
        candidate = solve_taska_monotone_component_portfolio(
            cost_right,
            cost_down,
            edges,
            candidate_priorities,
            grid=GRID,
            solver_config=replay.SOLVER_CONFIG,
        )
        if panel == "fresh32":
            frozen_control = helpers._strict_layout(
                priorities[f"{prefix}__portfolio_tail96_layout"]
            )
            if not np.array_equal(control.layout, frozen_control):
                raise RuntimeError("fresh32 control tail96 did not replay")

        arrays = {
            "control_tail96": helpers._strict_layout(control.layout),
            "monotone_portfolio": helpers._strict_layout(candidate.selection.layout),
            "monotone_tail96": helpers._strict_layout(candidate.polish.layout),
            **{
                f"monotone_arm_{name}": helpers._strict_layout(layout)
                for name, layout in candidate.layouts
            },
        }
        row = {
            "prefix": prefix,
            "candidate_edge_count": len(edges),
            "control_choice": control_start.choice,
            "control_original_total_costs": dict(control_start.total_costs),
            "control_tail_diagnostics": asdict(control.diagnostics),
            "monotone_choice": candidate.selection.choice,
            "monotone_original_total_costs": dict(candidate.selection.total_costs),
            "monotone_tail_diagnostics": asdict(candidate.polish.diagnostics),
            "placement_traces": {
                name: asdict(trace) for name, trace in candidate.placement_traces
            },
            "all_pair_relocation_attempt_counts_zero": all(
                trace.pair_relocation_attempts == 0
                for _, trace in candidate.placement_traces
            ),
            "all_layouts_strict": all(
                np.array_equal(np.sort(layout), np.arange(COUNT))
                for layout in arrays.values()
            ),
        }
    return arrays, row


def _freeze_target_free(
    *,
    panel: PanelName,
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    workers: int,
) -> tuple[Path, Path, Path, float]:
    output_dir.mkdir(parents=True, exist_ok=False)
    frozen_path = output_dir / "frozen-target-free-eval.npz"
    metadata_path = output_dir / "frozen-target-free-eval.json"
    freeze_path = output_dir / "pre-score-freeze.json"
    tasks = [(panel, str(row["prefix"])) for row in rows]
    started = perf_counter()
    executor: ProcessPoolExecutor | None = None
    if workers == 1:
        results = map(_solve_target_free_case, tasks)
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        results = executor.map(_solve_target_free_case, tasks)
    arrays: dict[str, np.ndarray] = {}
    frozen_rows: list[dict[str, Any]] = []
    try:
        for index, (parent_row, (case_arrays, case_row)) in enumerate(
            zip(rows, results, strict=True),
            start=1,
        ):
            prefix = str(parent_row["prefix"])
            for name, value in case_arrays.items():
                arrays[f"{prefix}__{name}"] = value
            frozen_rows.append(
                {
                    "prefix": prefix,
                    "case_id": parent_row["case_id"],
                    "source_filename": parent_row["source_filename"],
                    "draw_index": int(parent_row["draw_index"]),
                    "dirty_sha256": parent_row["dirty_sha256"],
                    **case_row,
                }
            )
            print(
                json.dumps(
                    {
                        "event": "taska_monotone_components_target_free_case_ready",
                        "panel": panel,
                        "case": index,
                        "case_count": len(rows),
                        "control_choice": case_row["control_choice"],
                        "monotone_choice": case_row["monotone_choice"],
                        "strict": case_row["all_layouts_strict"],
                    }
                ),
                flush=True,
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    helpers = replay.parent
    helpers._write_npz_exclusive(frozen_path, arrays)
    helpers._write_json_exclusive(
        metadata_path,
        {
            "schema": FROZEN_SCHEMA,
            "panel": panel,
            "contains_exact_references_or_labels": False,
            "contains_target_ids_or_source_grid_coordinates": False,
            "candidate_membership_and_edge_priorities_unchanged": True,
            "component_build_unchanged": True,
            "placement_initialization": "historical stable largest-first",
            "placement_search": (
                "six seed-0 coordinate-wise single-component best-relocation rounds"
            ),
            "unconditional_two_component_relocation_loop_omitted": True,
            "pair_relocation_attempts": 0,
            "fill": "unchanged seed-0 one-round Hungarian assignment",
            "selector": "minimum original TASKA cost over all 1104 board bonds",
            "protected_tail_max_swaps": TAIL_MAX_SWAPS,
            "no_sweep": True,
            "rows": frozen_rows,
        },
    )
    spec = PANEL_SPECS[panel]
    artifacts = {
        "parent_archive": PROJECT_ROOT / spec.parent_archive,
        "parent_metadata": PROJECT_ROOT / spec.parent_metadata,
        "priority_archive": PROJECT_ROOT / spec.priority_archive,
        "priority_metadata": PROJECT_ROOT / spec.priority_metadata,
        "logistic_calibrator": replay.DEFAULT_LOGISTIC,
        "nonlinear_calibrator": replay.DEFAULT_NONLINEAR,
        "frozen_candidate_archive": frozen_path,
        "frozen_candidate_metadata": metadata_path,
        **_runtime_sources(),
    }
    helpers._write_json_exclusive(
        freeze_path,
        {
            "schema": FREEZE_SCHEMA,
            "panel": panel,
            "created_before_exact_reference_recreation": True,
            "contains_evaluation_references_or_labels": False,
            "artifacts": {
                name: helpers._record(path) for name, path in artifacts.items()
            },
        },
    )
    return frozen_path, metadata_path, freeze_path, perf_counter() - started


def _score_after_freeze(
    *,
    panel: PanelName,
    rows: Sequence[Mapping[str, Any]],
    targets: Path,
    frozen_path: Path,
    metadata_path: Path,
    freeze_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    helpers = replay.parent
    helpers._validate_freeze(freeze_path)
    candidate = json.loads(metadata_path.read_text(encoding="utf-8"))
    candidate_rows = candidate.get("rows")
    if not isinstance(candidate_rows, list) or len(candidate_rows) != len(rows):
        raise RuntimeError("frozen candidate row roster changed")
    lookup = helpers.focal_parent._load_manifest_lookup()
    cache = helpers.focal_parent.CleanTileCache(targets.resolve())
    scored: list[dict[str, Any]] = []
    with np.load(frozen_path, allow_pickle=False) as archive:
        for parent_row, candidate_row in zip(rows, candidate_rows, strict=True):
            identity = (
                "prefix",
                "case_id",
                "source_filename",
                "draw_index",
                "dirty_sha256",
            )
            if any(parent_row[field] != candidate_row[field] for field in identity):
                raise RuntimeError("parent and candidate frozen row identities differ")
            prefix = str(parent_row["prefix"])
            source = str(parent_row["source_filename"])
            draw = int(parent_row["draw_index"])
            dirty, reference = helpers.make_exact_synthetic_case(
                cache.load(lookup[source]),
                source_filename=source,
                draw_index=draw,
                seed=helpers.focal_parent.SYNTHETIC_SEED,
            )
            parent_panel = PARENT_PANEL[panel]
            if (
                dirty.case_id != parent_row["case_id"]
                or reference.case_id != dirty.case_id
                or helpers._dirty_sha256(parent_panel, dirty.tiles)
                != parent_row["dirty_sha256"]
            ):
                raise RuntimeError("scoring recreated a different synthetic case")
            exact = helpers._strict_layout(reference.tile_at_position)
            metrics = {
                arm: helpers._layout_metrics(
                    helpers._strict_layout(archive[f"{prefix}__{arm}"]),
                    exact,
                )
                for arm in SCORED_ARMS
            }
            scored.append(
                {
                    "prefix": prefix,
                    "case_id": dirty.case_id,
                    "source_filename": source,
                    "draw_index": draw,
                    "control_choice": candidate_row["control_choice"],
                    "monotone_choice": candidate_row["monotone_choice"],
                    **metrics,
                }
            )

    metric_names = ("satisfied_adjacent_pairs", "adjacency_recall", "exact_tiles")
    sources = [str(row["source_filename"]) for row in scored]
    summary: dict[str, Any] = {
        "pair_denominator": helpers.PAIR_DENOMINATOR,
        "arms": {
            arm: {
                metric: float(np.mean([row[arm][metric] for row in scored]))
                for metric in metric_names
            }
            for arm in SCORED_ARMS
        },
        "control_choice_counts": dict(
            Counter(str(row["control_choice"]) for row in scored)
        ),
        "monotone_choice_counts": dict(
            Counter(str(row["monotone_choice"]) for row in scored)
        ),
    }
    for candidate_arm, label in (
        ("monotone_portfolio", "monotone_portfolio_minus_control"),
        ("monotone_tail96", "monotone_tail96_minus_control"),
    ):
        comparison: dict[str, Any] = {}
        for index, metric in enumerate(metric_names):
            deltas = [
                float(row[candidate_arm][metric])
                - float(row["control_tail96"][metric])
                for row in scored
            ]
            comparison[metric] = (
                helpers._clustered_ci(
                    deltas,
                    sources,
                    seed=(
                        BOOTSTRAP_SEED
                        + index
                        + (10 if candidate_arm.endswith("tail96") else 0)
                    ),
                )
                if len(scored) == CASE_COUNT
                else {
                    "mean": float(np.mean(deltas)),
                    "ci95_lower": None,
                    "ci95_upper": None,
                    "smoke_only": True,
                }
            )
        summary[label] = comparison
    return scored, summary


def _validate_dependencies(panel: PanelName) -> None:
    replay._validate_dependencies(PARENT_PANEL[panel])
    replay.parent._require_hash(
        PROJECT_ROOT / "scripts/run_taska_rawlog_tail.py",
        REPLAY_RUNNER_SHA256,
        name="frozen four-arm replay helpers",
    )
    replay.parent._require_hash(
        PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py",
        RAW_SOLVER_SHA256,
        name="frozen raw solver",
    )


def run(args: argparse.Namespace) -> Path:
    panel: PanelName = args.panel
    if isinstance(args.workers, bool) or not 1 <= int(args.workers) <= 8:
        raise ValueError("workers must be an integer in [1, 8]")
    if MONOTONE_TAIL_MAX_SWAPS != TAIL_MAX_SWAPS:
        raise RuntimeError("monotone module tail budget changed")
    if replay.SOLVER_CONFIG.search_rounds != SEARCH_ROUNDS:
        raise RuntimeError("solver search-round contract changed")
    _validate_dependencies(panel)
    targets = args.targets.resolve()
    if not targets.is_dir():
        raise ValueError(f"organizer-train target directory is absent: {targets}")
    rows = replay.parent._load_rows(
        PARENT_PANEL[panel],
        smoke_one=bool(args.smoke_one),
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else DEFAULT_OUTPUTS[panel].resolve()
    )
    started = perf_counter()
    frozen, metadata, freeze, inference_seconds = _freeze_target_free(
        panel=panel,
        rows=rows,
        output_dir=output_dir,
        workers=int(args.workers),
    )
    print(
        json.dumps(
            {
                "event": "taska_monotone_components_all_layouts_frozen",
                "panel": panel,
                "case_count": len(rows),
                "frozen_archive_sha256": sha256_file(frozen),
                "reference_reconstructed_yet": False,
            }
        ),
        flush=True,
    )
    scored, metrics = _score_after_freeze(
        panel=panel,
        rows=rows,
        targets=targets,
        frozen_path=frozen,
        metadata_path=metadata,
        freeze_path=freeze,
    )
    if not args.smoke_one:
        expected_pairs, expected_exact = EXPECTED_CONTROL_MEANS[panel]
        control = metrics["arms"]["control_tail96"]
        if (
            control["satisfied_adjacent_pairs"] != expected_pairs
            or control["exact_tiles"] != expected_exact
        ):
            raise RuntimeError("retained four-arm tail96 control mean did not replay")
    pair_delta = metrics["monotone_tail96_minus_control"][
        "satisfied_adjacent_pairs"
    ]["mean"]
    gate = {
        "opened_allows_held": panel == "opened32" and pair_delta >= 0.0,
        "held_allows_fresh": panel == "held32" and pair_delta >= 0.5,
        "criterion": (
            "opened pair delta >= 0 opens held32; held pair delta >= +0.5 "
            "and no opened collapse opens fresh32"
        ),
    }
    report = {
        "schema": REPORT_SCHEMA,
        "status": "completed",
        "panel": panel,
        "case_count": len(rows),
        "candidate": {
            "component_build": "unchanged per arm",
            "initial_placement": "unchanged stable largest-first",
            "search": "six seed-0 coordinate-wise best-relocation rounds",
            "removed_operation": "unconditional random two-component relocation loop",
            "pair_relocation_attempt_count": 0,
            "fill": "unchanged seed-0 one-round Hungarian assignment",
            "arms": list(MONOTONE_ARM_NAMES),
            "selector": "minimum original TASKA all-1104-bond cost",
            "protected_tail_max_swaps": TAIL_MAX_SWAPS,
            "no_sweep": True,
        },
        "legality": {
            "all_outputs_strict_original_upright_tile_permutations": all(
                row[arm]["strict_permutation"] for row in scored for arm in SCORED_ARMS
            ),
            "candidate_membership_and_original_costs_unchanged": True,
            "targets_used_only_after_target_free_layout_freeze": True,
            "competition_test_access": False,
            "pixels_emitted": False,
        },
        "metrics": metrics,
        "gate": gate,
        "rows": scored,
        "artifacts": {
            "frozen_target_free": replay.parent._record(frozen),
            "frozen_metadata": replay.parent._record(metadata),
            "pre_score_freeze": replay.parent._record(freeze),
        },
        "runtime_seconds": {
            "target_free": inference_seconds,
            "total": perf_counter() - started,
        },
    }
    report_path = output_dir / "report.json"
    replay.parent._write_json_exclusive(report_path, report)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "panel": panel,
                "control": metrics["arms"]["control_tail96"],
                "monotone_pre_tail": metrics["arms"]["monotone_portfolio"],
                "monotone_tail96": metrics["arms"]["monotone_tail96"],
                "pair_delta": pair_delta,
                "monotone_choice_counts": metrics["monotone_choice_counts"],
                "gate": gate,
            },
            indent=2,
        ),
        flush=True,
    )
    return report_path


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

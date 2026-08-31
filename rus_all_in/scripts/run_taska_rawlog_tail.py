#!/usr/bin/env python3
"""Evaluate one fixed raw-log protected-tail trajectory for TASKA.

The start layout is the retained seed-0 raw/logistic/focal/nonlinear all-bond
portfolio before tail polishing.  The control applies the established 96-swap
protected tail under original TASKA costs.  The candidate applies the same
protected set, budget, and gain threshold under exactly ``-right_log`` and
``-down_log``.  A final target-free selector retains whichever tail has lower
original TASKA cost across all 1,104 board bonds, with stable control ties.

Every candidate layout is frozen before exact synthetic references are
recreated.  Targets are used only by the later offline scoring pass.
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
from aiijc_puzzle.taska_edge_calibrator import (
    TaskaEdgeCalibrator,
    solve_prioritized_raw_tail_global,
)
from aiijc_puzzle.taska_layout_portfolio import select_lowest_taska_seam_cost_layout
from aiijc_puzzle.taska_nonlinear_calibrator import TaskaNonlinearCalibrator
from aiijc_puzzle.taska_rawlog_tail import select_taska_rawlog_tail

try:
    from scripts import run_taska_multistart_portfolio as parent
except ModuleNotFoundError:  # Direct ``python scripts/run_*.py`` execution.
    import run_taska_multistart_portfolio as parent

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRID = 24
COUNT = GRID * GRID
CASE_COUNT = 32
TAIL_MAX_SWAPS = 96
TAIL_MINIMUM_GAIN = 1e-9
BOOTSTRAP_SEED = 1_947_553_101
RAW_SOLVER_SHA256 = parent.RAW_SOLVER_SHA256
LOGISTIC_SHA256 = parent.LOGISTIC_SHA256
NONLINEAR_SHA256 = parent.NONLINEAR_SHA256
PARENT_RUNNER_SHA256 = "51daa16b6787690fa3de6644f09bc48e4f0e0d844d434576b8b6e9a79549f6bc"

FROZEN_SCHEMA = "aiijc-taska-rawlog-tail-target-free-v1"
FREEZE_SCHEMA = "aiijc-taska-rawlog-tail-pre-score-freeze-v1"
REPORT_SCHEMA = "aiijc-taska-rawlog-tail-report-v1"

PanelName = Literal["opened32", "held300", "fresh32"]
PANEL_SPECS = parent.PANEL_SPECS
DEFAULT_TARGETS = parent.DEFAULT_TARGETS
DEFAULT_LOGISTIC = parent.DEFAULT_LOGISTIC
DEFAULT_NONLINEAR = parent.DEFAULT_NONLINEAR
SOLVER_CONFIG = parent.SOLVER_CONFIG
DEFAULT_OUTPUTS: dict[PanelName, Path] = {
    panel: PROJECT_ROOT / f"outputs/taska-rawlog-tail/{panel}-v1"
    for panel in PANEL_SPECS
}
EXPECTED_CONTROL_MEANS: dict[PanelName, tuple[float, float]] = {
    "opened32": (341.3125, 4.75),
    "held300": (337.5625, 3.0625),
    "fresh32": (346.0625, 1.15625),
}
SCORED_ARMS = ("control_tail96", "rawlog_tail96", "selected_tail96")


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
        "rawlog_tail_runner": Path(__file__).resolve(),
        "rawlog_tail_module": PROJECT_ROOT / "src/aiijc_puzzle/taska_rawlog_tail.py",
        "protected_tail": PROJECT_ROOT
        / "src/aiijc_puzzle/taska_protected_tail_polish.py",
        "layout_portfolio": PROJECT_ROOT / "src/aiijc_puzzle/taska_layout_portfolio.py",
        "parent_replay_helpers": PROJECT_ROOT / "scripts/run_taska_multistart_portfolio.py",
        "frozen_raw_solver": PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py",
    }


def _four_arm_layouts(
    panel: PanelName,
    prefix: str,
    parent_archive: Any,
    priority_archive: Any,
    cost_right: np.ndarray,
    cost_down: np.ndarray,
    edges: Sequence[Any],
) -> dict[str, np.ndarray]:
    raw = parent._strict_layout(
        parent_archive[f"{prefix}__{PANEL_SPECS[panel].raw_layout_key}"]
    )
    focal = parent._strict_layout(priority_archive[f"{prefix}__focal_layout"])
    if panel == "fresh32":
        layouts = {
            "raw": raw,
            "logistic": parent._strict_layout(
                priority_archive[f"{prefix}__logistic_layout"]
            ),
            "focal": focal,
            "nonlinear": parent._strict_layout(
                priority_archive[f"{prefix}__nonlinear_layout"]
            ),
        }
        frozen_raw = parent._strict_layout(priority_archive[f"{prefix}__raw_layout"])
        if not np.array_equal(raw, frozen_raw):
            raise RuntimeError("fresh32 raw layouts differ between frozen parents")
        return layouts

    logistic = TaskaEdgeCalibrator.load_npz(DEFAULT_LOGISTIC)
    nonlinear = TaskaNonlinearCalibrator.load_npz(DEFAULT_NONLINEAR)
    priorities = parent._case_priorities(
        panel,
        prefix,
        parent_archive,
        priority_archive,
        cost_right,
        cost_down,
        edges,
        logistic,
        nonlinear,
    )
    logistic_solved = solve_prioritized_raw_tail_global(
        cost_right,
        cost_down,
        edges,
        priorities["logistic"],
        border_unary=None,
        grid=GRID,
        config=SOLVER_CONFIG,
    )
    nonlinear_solved = solve_prioritized_raw_tail_global(
        cost_right,
        cost_down,
        edges,
        priorities["nonlinear"],
        border_unary=None,
        grid=GRID,
        config=SOLVER_CONFIG,
    )
    return {
        "raw": raw,
        "logistic": parent._strict_layout(logistic_solved.layout),
        "focal": focal,
        "nonlinear": parent._strict_layout(nonlinear_solved.layout),
    }


def _solve_target_free_case(
    task: tuple[PanelName, str],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    panel, prefix = task
    spec = PANEL_SPECS[panel]
    with (
        np.load(PROJECT_ROOT / spec.parent_archive, allow_pickle=False) as base,
        np.load(PROJECT_ROOT / spec.priority_archive, allow_pickle=False) as priorities,
    ):
        cost_right = parent._finite_matrix(base, f"{prefix}__cost_right")
        cost_down = parent._finite_matrix(base, f"{prefix}__cost_down")
        right_log = parent._finite_matrix(base, f"{prefix}__right_log")
        down_log = parent._finite_matrix(base, f"{prefix}__down_log")
        edges = parent._edges_from_archive(base, prefix)
        layouts = _four_arm_layouts(
            panel,
            prefix,
            base,
            priorities,
            cost_right,
            cost_down,
            edges,
        )
        start = select_lowest_taska_seam_cost_layout(
            layouts,
            cost_right,
            cost_down,
            grid=GRID,
        )
        if panel == "fresh32":
            frozen_start = parent._strict_layout(priorities[f"{prefix}__portfolio_layout"])
            if not np.array_equal(start.layout, frozen_start):
                raise RuntimeError("fresh32 four-arm pre-tail layout did not replay")

        result = select_taska_rawlog_tail(
            start.layout,
            cost_right,
            cost_down,
            right_log,
            down_log,
            edges,
            grid=GRID,
            max_swaps=TAIL_MAX_SWAPS,
            minimum_gain=TAIL_MINIMUM_GAIN,
        )
        if panel == "fresh32":
            frozen_control = parent._strict_layout(
                priorities[f"{prefix}__portfolio_tail96_layout"]
            )
            if not np.array_equal(result.control.layout, frozen_control):
                raise RuntimeError("fresh32 control tail96 did not replay")

        arrays = {
            "portfolio_pre_tail": parent._strict_layout(start.layout),
            "control_tail96": parent._strict_layout(result.control.layout),
            "rawlog_tail96": parent._strict_layout(result.rawlog_tail.layout),
            "selected_tail96": parent._strict_layout(result.selection.layout),
        }
        row = {
            "prefix": prefix,
            "candidate_edge_count": len(edges),
            "four_arm_choice": start.choice,
            "four_arm_original_total_costs": dict(start.total_costs),
            "tail_choice": result.selection.choice,
            "tail_original_total_costs": dict(result.selection.total_costs),
            "control_diagnostics": asdict(result.control.diagnostics),
            "rawlog_diagnostics": asdict(result.rawlog_tail.diagnostics),
            "same_protected_tile_count": (
                result.control.diagnostics.protected_tile_count
                == result.rawlog_tail.diagnostics.protected_tile_count
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
                        "event": "taska_rawlog_tail_target_free_case_ready",
                        "panel": panel,
                        "case": index,
                        "case_count": len(rows),
                        "four_arm_choice": case_row["four_arm_choice"],
                        "tail_choice": case_row["tail_choice"],
                        "strict": case_row["all_layouts_strict"],
                    }
                ),
                flush=True,
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    parent._write_npz_exclusive(frozen_path, arrays)
    parent._write_json_exclusive(
        metadata_path,
        {
            "schema": FROZEN_SCHEMA,
            "panel": panel,
            "contains_exact_references_or_labels": False,
            "contains_target_ids_or_source_grid_coordinates": False,
            "start": "seed-0 four-arm minimum original all-1104-bond TASKA cost",
            "control_objective": "original TASKA cost_right/cost_down",
            "candidate_objective": "exactly -right_log/-down_log",
            "candidate_is_alternate_search_trajectory_not_score_blend": True,
            "protected_edges_and_start_layout_identical_between_tails": True,
            "max_swaps": TAIL_MAX_SWAPS,
            "minimum_gain": TAIL_MINIMUM_GAIN,
            "final_selector": "minimum original all-1104-bond TASKA cost; stable control tie",
            "rows": frozen_rows,
        },
    )
    spec = PANEL_SPECS[panel]
    artifacts = {
        "parent_archive": PROJECT_ROOT / spec.parent_archive,
        "parent_metadata": PROJECT_ROOT / spec.parent_metadata,
        "priority_archive": PROJECT_ROOT / spec.priority_archive,
        "priority_metadata": PROJECT_ROOT / spec.priority_metadata,
        "logistic_calibrator": DEFAULT_LOGISTIC,
        "nonlinear_calibrator": DEFAULT_NONLINEAR,
        "frozen_candidate_archive": frozen_path,
        "frozen_candidate_metadata": metadata_path,
        **_runtime_sources(),
    }
    parent._write_json_exclusive(
        freeze_path,
        {
            "schema": FREEZE_SCHEMA,
            "panel": panel,
            "created_before_exact_reference_recreation": True,
            "contains_evaluation_references_or_labels": False,
            "artifacts": {name: parent._record(path) for name, path in artifacts.items()},
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
    parent._validate_freeze(freeze_path)
    candidate = json.loads(metadata_path.read_text(encoding="utf-8"))
    candidate_rows = candidate.get("rows")
    if not isinstance(candidate_rows, list) or len(candidate_rows) != len(rows):
        raise RuntimeError("frozen candidate row roster changed")
    lookup = parent.focal_parent._load_manifest_lookup()
    cache = parent.focal_parent.CleanTileCache(targets.resolve())
    scored: list[dict[str, Any]] = []
    with np.load(frozen_path, allow_pickle=False) as archive:
        for parent_row, candidate_row in zip(rows, candidate_rows, strict=True):
            identity = ("prefix", "case_id", "source_filename", "draw_index", "dirty_sha256")
            if any(parent_row[field] != candidate_row[field] for field in identity):
                raise RuntimeError("parent and candidate frozen row identities differ")
            prefix = str(parent_row["prefix"])
            source = str(parent_row["source_filename"])
            draw = int(parent_row["draw_index"])
            dirty, reference = parent.make_exact_synthetic_case(
                cache.load(lookup[source]),
                source_filename=source,
                draw_index=draw,
                seed=parent.focal_parent.SYNTHETIC_SEED,
            )
            if (
                dirty.case_id != parent_row["case_id"]
                or reference.case_id != dirty.case_id
                or parent._dirty_sha256(panel, dirty.tiles) != parent_row["dirty_sha256"]
            ):
                raise RuntimeError("scoring recreated a different synthetic case")
            exact = parent._strict_layout(reference.tile_at_position)
            metrics = {
                arm: parent._layout_metrics(
                    parent._strict_layout(archive[f"{prefix}__{arm}"]),
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
                    "tail_choice": candidate_row["tail_choice"],
                    **metrics,
                }
            )

    metric_names = ("satisfied_adjacent_pairs", "adjacency_recall", "exact_tiles")
    sources = [str(row["source_filename"]) for row in scored]
    summary: dict[str, Any] = {
        "pair_denominator": parent.PAIR_DENOMINATOR,
        "arms": {
            arm: {
                metric: float(np.mean([row[arm][metric] for row in scored]))
                for metric in metric_names
            }
            for arm in SCORED_ARMS
        },
        "selection_counts": dict(Counter(str(row["tail_choice"]) for row in scored)),
    }
    for candidate_arm, label in (
        ("rawlog_tail96", "rawlog_minus_control"),
        ("selected_tail96", "selected_minus_control"),
    ):
        comparison: dict[str, Any] = {}
        for index, metric in enumerate(metric_names):
            deltas = [
                float(row[candidate_arm][metric])
                - float(row["control_tail96"][metric])
                for row in scored
            ]
            comparison[metric] = (
                parent._clustered_ci(
                    deltas,
                    sources,
                    seed=(
                        BOOTSTRAP_SEED
                        + index
                        + (10 if candidate_arm.startswith("selected") else 0)
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
    spec = PANEL_SPECS[panel]
    parent._require_hash(
        PROJECT_ROOT / spec.parent_archive,
        spec.parent_archive_sha256,
        name="parent archive",
    )
    parent._require_hash(
        PROJECT_ROOT / spec.parent_metadata,
        spec.parent_metadata_sha256,
        name="parent metadata",
    )
    parent._require_hash(
        PROJECT_ROOT / spec.priority_archive,
        spec.priority_archive_sha256,
        name="priority archive",
    )
    parent._require_hash(
        PROJECT_ROOT / spec.priority_metadata,
        spec.priority_metadata_sha256,
        name="priority metadata",
    )
    parent._require_hash(DEFAULT_LOGISTIC, LOGISTIC_SHA256, name="logistic calibrator")
    parent._require_hash(DEFAULT_NONLINEAR, NONLINEAR_SHA256, name="nonlinear calibrator")
    parent._require_hash(
        PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py",
        RAW_SOLVER_SHA256,
        name="frozen raw solver",
    )
    parent._require_hash(
        PROJECT_ROOT / "scripts/run_taska_multistart_portfolio.py",
        PARENT_RUNNER_SHA256,
        name="frozen parent replay helpers",
    )


def run(args: argparse.Namespace) -> Path:
    panel: PanelName = args.panel
    if isinstance(args.workers, bool) or not 1 <= int(args.workers) <= 8:
        raise ValueError("workers must be an integer in [1, 8]")
    _validate_dependencies(panel)
    targets = args.targets.resolve()
    if not targets.is_dir():
        raise ValueError(f"organizer-train target directory is absent: {targets}")
    rows = parent._load_rows(panel, smoke_one=bool(args.smoke_one))
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
                "event": "taska_rawlog_tail_all_layouts_frozen",
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
    pair_delta = metrics["selected_minus_control"]["satisfied_adjacent_pairs"]["mean"]
    gate = {
        "opened_allows_held": panel == "opened32" and pair_delta >= 0.0,
        "held_allows_fresh": panel == "held300" and pair_delta > 0.0,
        "criterion": (
            "opened: selected pair delta >= 0; held: selected pair delta > 0 for fresh"
        ),
    }
    report = {
        "schema": REPORT_SCHEMA,
        "status": "completed",
        "panel": panel,
        "case_count": len(rows),
        "candidate": {
            "start": "current fixed four-arm all-bond selected pre-tail layout",
            "control": "original TASKA protected tail, max_swaps=96",
            "rawlog": "protected tail under exactly -right_log/-down_log, max_swaps=96",
            "selector": "lower original TASKA all-1104-bond cost; stable control tie",
            "no_sweep": True,
        },
        "legality": {
            "all_outputs_strict_original_upright_tile_permutations": all(
                row[arm]["strict_permutation"] for row in scored for arm in SCORED_ARMS
            ),
            "targets_used_only_after_target_free_layout_freeze": True,
            "competition_test_access": False,
        },
        "metrics": metrics,
        "gate": gate,
        "rows": scored,
        "artifacts": {
            "frozen_target_free": parent._record(frozen),
            "frozen_metadata": parent._record(metadata),
            "pre_score_freeze": parent._record(freeze),
        },
        "runtime_seconds": {
            "target_free": inference_seconds,
            "total": perf_counter() - started,
        },
    }
    report_path = output_dir / "report.json"
    parent._write_json_exclusive(report_path, report)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "panel": panel,
                "control": metrics["arms"]["control_tail96"],
                "rawlog": metrics["arms"]["rawlog_tail96"],
                "selected": metrics["arms"]["selected_tail96"],
                "pair_delta": pair_delta,
                "selection_counts": metrics["selection_counts"],
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

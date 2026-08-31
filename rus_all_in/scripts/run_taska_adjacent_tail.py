#!/usr/bin/env python3
"""Evaluate one fixed adjacent-aware protected TASKA tail.

The control and candidate start from the same retained seed-0
raw/logistic/focal/nonlinear all-bond portfolio layout.  Both freeze every tile
in an already realised harvested edge, minimize the original TASKA right/down
cost, use ``max_swaps=96`` and ``minimum_gain=1e-9``, and take the stable
row-major global-best improving swap at every step.  The candidate differs
only by allowing adjacent free positions, whose deltas are recomputed exactly
over the union of affected directed board bonds.

Layouts are hash-frozen before exact synthetic references are recreated.
Targets are used only in the later offline scoring pass.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np

from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.taska_adjacent_tail import (
    polish_unprotected_taska_tail_with_adjacent_swaps,
)
from aiijc_puzzle.taska_layout_portfolio import select_lowest_taska_seam_cost_layout
from aiijc_puzzle.taska_protected_tail_polish import polish_unprotected_taska_tail

try:
    from scripts import run_taska_rawlog_tail as replay
except ModuleNotFoundError:  # Direct ``python scripts/run_*.py`` execution.
    import run_taska_rawlog_tail as replay

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRID = 24
COUNT = GRID * GRID
CASE_COUNT = 32
TAIL_MAX_SWAPS = 96
TAIL_MINIMUM_GAIN = 1e-9
BOOTSTRAP_SEED = 2_608_684_301
RAW_SOLVER_SHA256 = replay.RAW_SOLVER_SHA256
REPLAY_RUNNER_SHA256 = "181ae0c24bdd4c2ed2f0459bd51ca4cfded2001f143efa45e11b2cd89fd73383"

FROZEN_SCHEMA = "aiijc-taska-adjacent-tail-target-free-v1"
FREEZE_SCHEMA = "aiijc-taska-adjacent-tail-pre-score-freeze-v1"
REPORT_SCHEMA = "aiijc-taska-adjacent-tail-report-v1"

PanelName = Literal["opened32", "held300", "fresh32"]
PANEL_SPECS = replay.PANEL_SPECS
DEFAULT_TARGETS = replay.DEFAULT_TARGETS
DEFAULT_OUTPUTS: dict[PanelName, Path] = {
    panel: PROJECT_ROOT / f"outputs/taska-adjacent-tail/{panel}-v1"
    for panel in PANEL_SPECS
}
EXPECTED_CONTROL_MEANS: dict[PanelName, tuple[float, float]] = {
    "opened32": (341.3125, 4.75),
    "held300": (337.5625, 3.0625),
    "fresh32": (346.0625, 1.15625),
}
SCORED_ARMS = ("control_tail96", "adjacent_tail96")


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
        "adjacent_tail_runner": Path(__file__).resolve(),
        "adjacent_tail_module": (
            PROJECT_ROOT / "src/aiijc_puzzle/taska_adjacent_tail.py"
        ),
        "protected_tail_control": (
            PROJECT_ROOT / "src/aiijc_puzzle/taska_protected_tail_polish.py"
        ),
        "layout_portfolio": (
            PROJECT_ROOT / "src/aiijc_puzzle/taska_layout_portfolio.py"
        ),
        "four_arm_replay_helpers": (
            PROJECT_ROOT / "scripts/run_taska_rawlog_tail.py"
        ),
        "frozen_raw_solver": (
            PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py"
        ),
    }


def _solve_target_free_case(
    task: tuple[PanelName, str],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    panel, prefix = task
    spec = PANEL_SPECS[panel]
    base_helpers = replay.parent
    with (
        np.load(PROJECT_ROOT / spec.parent_archive, allow_pickle=False) as base,
        np.load(PROJECT_ROOT / spec.priority_archive, allow_pickle=False) as priorities,
    ):
        cost_right = base_helpers._finite_matrix(base, f"{prefix}__cost_right")
        cost_down = base_helpers._finite_matrix(base, f"{prefix}__cost_down")
        edges = base_helpers._edges_from_archive(base, prefix)
        layouts = replay._four_arm_layouts(
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
            frozen_start = base_helpers._strict_layout(
                priorities[f"{prefix}__portfolio_layout"]
            )
            if not np.array_equal(start.layout, frozen_start):
                raise RuntimeError("fresh32 four-arm pre-tail layout did not replay")

        control = polish_unprotected_taska_tail(
            start.layout,
            cost_right,
            cost_down,
            edges,
            grid=GRID,
            max_swaps=TAIL_MAX_SWAPS,
            minimum_gain=TAIL_MINIMUM_GAIN,
        )
        adjacent = polish_unprotected_taska_tail_with_adjacent_swaps(
            start.layout,
            cost_right,
            cost_down,
            edges,
            grid=GRID,
            max_swaps=TAIL_MAX_SWAPS,
            minimum_gain=TAIL_MINIMUM_GAIN,
        )
        if panel == "fresh32":
            frozen_control = base_helpers._strict_layout(
                priorities[f"{prefix}__portfolio_tail96_layout"]
            )
            if not np.array_equal(control.layout, frozen_control):
                raise RuntimeError("fresh32 control tail96 did not replay")

        protection_fields = (
            "protected_tile_count",
            "free_tile_count",
            "initial_realised_edge_count",
        )
        if any(
            getattr(control.diagnostics, field)
            != getattr(adjacent.diagnostics, field)
            for field in protection_fields
        ):
            raise RuntimeError("control and adjacent tails used different protection")
        arrays = {
            "portfolio_pre_tail": base_helpers._strict_layout(start.layout),
            "control_tail96": base_helpers._strict_layout(control.layout),
            "adjacent_tail96": base_helpers._strict_layout(adjacent.layout),
        }
        row = {
            "prefix": prefix,
            "candidate_edge_count": len(edges),
            "four_arm_choice": start.choice,
            "four_arm_original_total_costs": dict(start.total_costs),
            "control_diagnostics": asdict(control.diagnostics),
            "adjacent_diagnostics": asdict(adjacent.diagnostics),
            "same_protected_set_measurements": True,
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
                        "event": "taska_adjacent_tail_target_free_case_ready",
                        "panel": panel,
                        "case": index,
                        "case_count": len(rows),
                        "accepted_adjacent_swaps": case_row[
                            "adjacent_diagnostics"
                        ]["accepted_adjacent_swap_count"],
                        "strict": case_row["all_layouts_strict"],
                    }
                ),
                flush=True,
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    base_helpers = replay.parent
    base_helpers._write_npz_exclusive(frozen_path, arrays)
    base_helpers._write_json_exclusive(
        metadata_path,
        {
            "schema": FROZEN_SCHEMA,
            "panel": panel,
            "contains_exact_references_or_labels": False,
            "contains_target_ids_or_source_grid_coordinates": False,
            "start": "seed-0 four-arm minimum original all-1104-bond TASKA cost",
            "control": "original TASKA protected tail excluding adjacent swaps",
            "candidate": (
                "same protected tail with exact union-of-directed-bonds deltas "
                "for adjacent free positions"
            ),
            "objective": "original TASKA cost_right/cost_down only",
            "max_swaps": TAIL_MAX_SWAPS,
            "minimum_gain": TAIL_MINIMUM_GAIN,
            "global_best_tie_rule": "stable row-major first minimum",
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
    base_helpers._write_json_exclusive(
        freeze_path,
        {
            "schema": FREEZE_SCHEMA,
            "panel": panel,
            "created_before_exact_reference_recreation": True,
            "contains_evaluation_references_or_labels": False,
            "artifacts": {
                name: base_helpers._record(path) for name, path in artifacts.items()
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
    base_helpers = replay.parent
    base_helpers._validate_freeze(freeze_path)
    candidate = json.loads(metadata_path.read_text(encoding="utf-8"))
    candidate_rows = candidate.get("rows")
    if not isinstance(candidate_rows, list) or len(candidate_rows) != len(rows):
        raise RuntimeError("frozen candidate row roster changed")

    lookup = base_helpers.focal_parent._load_manifest_lookup()
    cache = base_helpers.focal_parent.CleanTileCache(targets.resolve())
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
            dirty, reference = base_helpers.make_exact_synthetic_case(
                cache.load(lookup[source]),
                source_filename=source,
                draw_index=draw,
                seed=base_helpers.focal_parent.SYNTHETIC_SEED,
            )
            if (
                dirty.case_id != parent_row["case_id"]
                or reference.case_id != dirty.case_id
                or base_helpers._dirty_sha256(panel, dirty.tiles)
                != parent_row["dirty_sha256"]
            ):
                raise RuntimeError("scoring recreated a different synthetic case")
            exact = base_helpers._strict_layout(reference.tile_at_position)
            metrics = {
                arm: base_helpers._layout_metrics(
                    base_helpers._strict_layout(archive[f"{prefix}__{arm}"]),
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
                    **metrics,
                }
            )

    metric_names = ("satisfied_adjacent_pairs", "adjacency_recall", "exact_tiles")
    sources = [str(row["source_filename"]) for row in scored]
    summary: dict[str, Any] = {
        "pair_denominator": base_helpers.PAIR_DENOMINATOR,
        "arms": {
            arm: {
                metric: float(np.mean([row[arm][metric] for row in scored]))
                for metric in metric_names
            }
            for arm in SCORED_ARMS
        },
    }
    comparison: dict[str, Any] = {}
    for index, metric in enumerate(metric_names):
        deltas = [
            float(row["adjacent_tail96"][metric])
            - float(row["control_tail96"][metric])
            for row in scored
        ]
        comparison[metric] = (
            base_helpers._clustered_ci(
                deltas,
                sources,
                seed=BOOTSTRAP_SEED + index,
            )
            if len(scored) == CASE_COUNT
            else {
                "mean": float(np.mean(deltas)),
                "ci95_lower": None,
                "ci95_upper": None,
                "smoke_only": True,
            }
        )
    summary["adjacent_minus_control"] = comparison
    return scored, summary


def _validate_dependencies(panel: PanelName) -> None:
    replay._validate_dependencies(panel)
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
    _validate_dependencies(panel)
    targets = args.targets.resolve()
    if not targets.is_dir():
        raise ValueError(f"organizer-train target directory is absent: {targets}")
    rows = replay.parent._load_rows(panel, smoke_one=bool(args.smoke_one))
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
                "event": "taska_adjacent_tail_all_layouts_frozen",
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
    pair_delta = metrics["adjacent_minus_control"]["satisfied_adjacent_pairs"][
        "mean"
    ]
    gate = {
        "opened_allows_held": panel == "opened32" and pair_delta >= 0.0,
        "held_allows_fresh": panel == "held300" and pair_delta > 0.0,
        "criterion": "opened pair delta >= 0; held pair delta > 0 for fresh",
    }
    report = {
        "schema": REPORT_SCHEMA,
        "status": "completed",
        "panel": panel,
        "case_count": len(rows),
        "candidate": {
            "start": "current fixed four-arm all-bond selected pre-tail layout",
            "control": "original protected tail excluding adjacent swaps",
            "extension": (
                "same global-best tail with exact affected-bond delta for "
                "adjacent free-position swaps"
            ),
            "objective": "original TASKA cost_right/cost_down",
            "max_swaps": TAIL_MAX_SWAPS,
            "minimum_gain": TAIL_MINIMUM_GAIN,
            "stable_global_tie_rule": "row-major first minimum",
            "no_sweep": True,
        },
        "legality": {
            "all_outputs_strict_original_upright_tile_permutations": all(
                row[arm]["strict_permutation"] for row in scored for arm in SCORED_ARMS
            ),
            "initially_realised_harvested_relations_preserved": True,
            "targets_used_only_after_target_free_layout_freeze": True,
            "competition_test_access": False,
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
                "adjacent": metrics["arms"]["adjacent_tail96"],
                "pair_delta": pair_delta,
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

#!/usr/bin/env python3
"""Evaluate one fixed four-layout consensus-component TASKA arm.

All target-free inputs come from already frozen TASKA raw/focal archives.  Raw,
logistic, focal-top5, and nonlinear pre-tail layouts are frozen before their
directed board bonds are counted.  Bonds supported by at least two layouts are
then used as the supply of the unchanged raw-tail component placer, ordered by
support, original TASKA cost, and stable identity.  The result receives the
fixed protected tail-96 continuation.

The consensus layouts and provenance are written before exact references are
recreated.  There is no support threshold, weight, or tail-budget sweep.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np

from aiijc_puzzle.layout_evaluation import evaluate_layout
from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.synthetic_socket_evaluation import make_exact_synthetic_case
from aiijc_puzzle.taska_consensus_component_arm import (
    CONSENSUS_ARM_NAMES,
    solve_taska_consensus_component_arm,
)
from aiijc_puzzle.taska_edge_calibrator import (
    TaskaEdgeCalibrator,
    extract_taska_edge_features,
    solve_prioritized_raw_tail_global,
)
from aiijc_puzzle.taska_layout_portfolio import select_lowest_taska_seam_cost_layout
from aiijc_puzzle.taska_nonlinear_calibrator import TaskaNonlinearCalibrator
from aiijc_puzzle.taska_protected_tail_polish import polish_unprotected_taska_tail

try:
    from scripts import run_taska_focal_verifier_replay as base
except ModuleNotFoundError:  # Direct ``python scripts/run_*.py`` execution.
    import run_taska_focal_verifier_replay as base

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PanelName = Literal["opened32", "held300"]
GRID = 24
COUNT = GRID * GRID
PAIR_DENOMINATOR = 2 * GRID * (GRID - 1)
TAIL_MAX_SWAPS = 96
TAIL_MINIMUM_GAIN = 1e-9
BOOTSTRAP_SEED = 1_407_492_827

DEFAULT_LOGISTIC = PROJECT_ROOT / "outputs/taska-edge-calibrator/train256-v1/calibrator.npz"
DEFAULT_NONLINEAR = (
    PROJECT_ROOT / "outputs/taska-nonlinear-calibrator/train256-v1/calibrator.npz"
)
DEFAULT_OUTPUTS = {
    "opened32": PROJECT_ROOT / "outputs/taska-consensus-component-arm/opened32-v1",
    "held300": PROJECT_ROOT / "outputs/taska-consensus-component-arm/held300-v1",
}
LOGISTIC_SHA256 = "adc76ee87fc112d4ca3eeb676cdec6b7d103c596d62a9848ba65ee5ef384b1ac"
NONLINEAR_SHA256 = "2a5f95bd9d8e08e57b8bd02e242e25ef4661036ed3b1985fda1d70ee1bf9d2a6"
RAW_SOLVER_SHA256 = "97859e1f4ff6ceadf56ebccf429f2944ae7a5726631731d9f776224a667eb486"
FOCAL_ARTIFACTS: dict[PanelName, dict[str, str]] = {
    "opened32": {
        "archive": (
            "outputs/taska-focal-verifier/opened32-train-exact-top5-cpu-v2/"
            "frozen-target-free-eval.npz"
        ),
        "archive_sha256": "60243ab924da96d8bb49b072458c4710c65b8195b8d2c31eff1132b59ee56fd2",
        "metadata": (
            "outputs/taska-focal-verifier/opened32-train-exact-top5-cpu-v2/"
            "frozen-target-free-eval.json"
        ),
        "metadata_sha256": "8e6be1d0f4b2652b784141d7c53d7fb63394e8bda6af3b076a9fd5721f07c9d5",
    },
    "held300": {
        "archive": (
            "outputs/taska-focal-verifier/held300-train-exact-top5-cpu-v2/"
            "frozen-target-free-eval.npz"
        ),
        "archive_sha256": "7d4ad494ab572d1ac3c94ab73a49b54e80b26baba489dfbd56f732a5c43394c5",
        "metadata": (
            "outputs/taska-focal-verifier/held300-train-exact-top5-cpu-v2/"
            "frozen-target-free-eval.json"
        ),
        "metadata_sha256": "301ba535f04b63ff8da48a0a83b5f207521d4b57f1bdbb61ceb58dbee57daff2",
    },
}

FROZEN_SCHEMA = "aiijc-taska-consensus-component-target-free-v1"
FREEZE_SCHEMA = "aiijc-taska-consensus-component-pre-score-freeze-v1"
REPORT_SCHEMA = "aiijc-taska-consensus-component-report-v1"
SCORED_ARMS = ("current_tail96", "consensus_component", "consensus_tail96")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=("opened32", "held300"), required=True)
    parser.add_argument("--targets", type=Path, default=base.DEFAULT_TARGETS)
    parser.add_argument("--logistic", type=Path, default=DEFAULT_LOGISTIC)
    parser.add_argument("--nonlinear", type=Path, default=DEFAULT_NONLINEAR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke-one", action="store_true")
    return parser.parse_args(argv)


def _require_hash(path: Path, expected: str, *, name: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"{name} is absent: {resolved}")
    actual = sha256_file(resolved)
    if actual != expected:
        raise ValueError(f"{name} SHA-256 mismatch: expected {expected}, got {actual}")
    return resolved


def _record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        rendered = str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        rendered = str(resolved)
    return {"path": rendered, "sha256": sha256_file(resolved)}


def _write_json_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {path}") from error


def _write_npz_exclusive(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {path}") from error


def _strict_layout(value: Any) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != (COUNT,) or raw.dtype.kind not in "iu" or raw.dtype.kind == "b":
        raise ValueError("layout must be one integer length-576 vector")
    layout = np.ascontiguousarray(raw, dtype=np.int32)
    if not np.array_equal(np.sort(layout), np.arange(COUNT)):
        raise ValueError("layout must contain every original tile exactly once")
    return layout


def _finite_matrix(archive: Any, key: str) -> np.ndarray:
    matrix = np.asarray(archive[key], dtype=np.float64)
    if matrix.shape != (COUNT, COUNT) or not np.isfinite(matrix).all():
        raise ValueError(f"{key} is not one finite 576x576 matrix")
    return np.ascontiguousarray(matrix)


def _edges(archive: Any, prefix: str) -> tuple[RawTailEdge, ...]:
    source = np.asarray(archive[f"{prefix}__edge_source"], dtype=np.int64)
    target = np.asarray(archive[f"{prefix}__edge_target"], dtype=np.int64)
    axis = np.asarray(archive[f"{prefix}__edge_axis"], dtype=np.uint8)
    if not (source.ndim == target.ndim == axis.ndim == 1):
        raise ValueError("edge arrays must be one-dimensional")
    if not (len(source) == len(target) == len(axis)) or not np.isin(axis, (0, 1)).all():
        raise ValueError("edge arrays are malformed")
    return tuple(
        RawTailEdge(int(left), int(right), "right" if int(direction) == 0 else "down")
        for left, right, direction in zip(source, target, axis, strict=True)
    )


def _focal_paths(panel: PanelName) -> tuple[Path, Path]:
    spec = FOCAL_ARTIFACTS[panel]
    return (
        _require_hash(
            PROJECT_ROOT / spec["archive"],
            spec["archive_sha256"],
            name=f"{panel} focal archive",
        ),
        _require_hash(
            PROJECT_ROOT / spec["metadata"],
            spec["metadata_sha256"],
            name=f"{panel} focal metadata",
        ),
    )


def _validate_focal_rows(
    focal_metadata: Path,
    parent_rows: Sequence[Mapping[str, Any]],
) -> None:
    payload = json.loads(focal_metadata.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if payload.get("contains_exact_references_or_labels") is not False:
        raise ValueError("focal metadata is not target-free")
    if not isinstance(rows, list) or len(rows) != 32:
        raise ValueError("focal metadata must contain 32 rows")
    for parent_row, focal_row in zip(parent_rows, rows, strict=True):
        for field in ("prefix", "case_id", "source_filename", "draw_index", "dirty_sha256"):
            if parent_row[field] != focal_row[field]:
                raise RuntimeError(f"focal and parent row differ at {field}")


def _runtime_sources() -> dict[str, Path]:
    return {
        "runner": Path(__file__).resolve(),
        "consensus_component_arm": (
            PROJECT_ROOT / "src/aiijc_puzzle/taska_consensus_component_arm.py"
        ),
        "edge_calibrator": PROJECT_ROOT / "src/aiijc_puzzle/taska_edge_calibrator.py",
        "nonlinear_calibrator": (
            PROJECT_ROOT / "src/aiijc_puzzle/taska_nonlinear_calibrator.py"
        ),
        "layout_portfolio": PROJECT_ROOT / "src/aiijc_puzzle/taska_layout_portfolio.py",
        "protected_tail": PROJECT_ROOT / "src/aiijc_puzzle/taska_protected_tail_polish.py",
        "frozen_raw_solver": PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py",
    }


def _freeze_target_free(
    *,
    panel: PanelName,
    rows: Sequence[Mapping[str, Any]],
    parent_archive: Path,
    parent_metadata: Path,
    focal_archive: Path,
    focal_metadata: Path,
    logistic_path: Path,
    nonlinear_path: Path,
    output_dir: Path,
) -> tuple[Path, Path, Path, float]:
    output_dir.mkdir(parents=True, exist_ok=False)
    frozen_path = output_dir / "frozen-target-free-eval.npz"
    metadata_path = output_dir / "frozen-target-free-eval.json"
    freeze_path = output_dir / "pre-score-freeze.json"
    logistic = TaskaEdgeCalibrator.load_npz(logistic_path)
    nonlinear = TaskaNonlinearCalibrator.load_npz(nonlinear_path)
    arrays: dict[str, np.ndarray] = {}
    frozen_rows: list[dict[str, Any]] = []
    started = perf_counter()

    with (
        np.load(parent_archive, allow_pickle=False) as parent_data,
        np.load(focal_archive, allow_pickle=False) as focal_data,
    ):
        for index, row in enumerate(rows):
            prefix = str(row["prefix"])
            right = _finite_matrix(parent_data, f"{prefix}__cost_right")
            down = _finite_matrix(parent_data, f"{prefix}__cost_down")
            right_log = _finite_matrix(parent_data, f"{prefix}__right_log")
            down_log = _finite_matrix(parent_data, f"{prefix}__down_log")
            candidate_edges = _edges(parent_data, prefix)
            if len(candidate_edges) != int(row["candidate_edge_count"]):
                raise RuntimeError("candidate edge count changed")
            weights = np.asarray(parent_data[f"{prefix}__edge_weight"], dtype=np.float64)
            votes = np.asarray(
                parent_data[f"{prefix}__edge_vote_count"], dtype=np.float64
            )
            features = extract_taska_edge_features(
                right,
                down,
                right_log,
                down_log,
                candidate_edges,
                weights,
                votes,
                grid=GRID,
            )
            logistic_solved = solve_prioritized_raw_tail_global(
                right,
                down,
                candidate_edges,
                logistic.predict_priorities(features.values),
                border_unary=None,
                grid=GRID,
                config=base.SOLVER_CONFIG,
            )
            nonlinear_solved = solve_prioritized_raw_tail_global(
                right,
                down,
                candidate_edges,
                nonlinear.predict_priorities(features.values),
                border_unary=None,
                grid=GRID,
                config=base.SOLVER_CONFIG,
            )
            layouts = {
                "raw": _strict_layout(parent_data[f"{prefix}__taska_layout"]),
                "logistic": _strict_layout(logistic_solved.layout),
                "focal": _strict_layout(focal_data[f"{prefix}__focal_layout"]),
                "nonlinear": _strict_layout(nonlinear_solved.layout),
            }
            selection = select_lowest_taska_seam_cost_layout(
                layouts,
                right,
                down,
                grid=GRID,
            )
            current_tail = polish_unprotected_taska_tail(
                selection.layout,
                right,
                down,
                candidate_edges,
                grid=GRID,
                max_swaps=TAIL_MAX_SWAPS,
                minimum_gain=TAIL_MINIMUM_GAIN,
            )
            consensus = solve_taska_consensus_component_arm(
                layouts,
                right,
                down,
                grid=GRID,
                solver_config=base.SOLVER_CONFIG,
            )
            consensus_edges = tuple(bond.edge for bond in consensus.bonds)
            consensus_support = np.asarray(
                [bond.support for bond in consensus.bonds], dtype=np.uint8
            )
            arrays[f"{prefix}__current_tail96_layout"] = _strict_layout(
                current_tail.layout
            )
            arrays[f"{prefix}__consensus_component_layout"] = _strict_layout(
                consensus.component.layout
            )
            arrays[f"{prefix}__consensus_tail96_layout"] = _strict_layout(
                consensus.layout
            )
            arrays[f"{prefix}__consensus_edge_source"] = np.asarray(
                [edge.source for edge in consensus_edges], dtype=np.int32
            )
            arrays[f"{prefix}__consensus_edge_target"] = np.asarray(
                [edge.target for edge in consensus_edges], dtype=np.int32
            )
            arrays[f"{prefix}__consensus_edge_axis"] = np.asarray(
                [edge.axis == "down" for edge in consensus_edges], dtype=np.uint8
            )
            arrays[f"{prefix}__consensus_edge_support"] = consensus_support
            frozen_rows.append(
                {
                    "prefix": prefix,
                    "case_id": row["case_id"],
                    "source_filename": row["source_filename"],
                    "draw_index": row["draw_index"],
                    "dirty_sha256": row["dirty_sha256"],
                    "candidate_edge_count": len(candidate_edges),
                    "portfolio_choice": selection.choice,
                    "portfolio_costs": dict(selection.total_costs),
                    "consensus": asdict(consensus.diagnostics),
                    "consensus_component_solver": consensus.component.diagnostics.as_dict(),
                    "consensus_tail": asdict(consensus.tail.diagnostics),
                    "current_tail": asdict(current_tail.diagnostics),
                }
            )
            print(
                json.dumps(
                    {
                        "event": "taska_consensus_component_case_frozen_in_memory",
                        "panel": panel,
                        "case": index + 1,
                        "case_count": len(rows),
                        "consensus_edges": len(consensus_edges),
                        "strict": True,
                    }
                ),
                flush=True,
            )

    _write_npz_exclusive(frozen_path, arrays)
    _write_json_exclusive(
        metadata_path,
        {
            "schema": FROZEN_SCHEMA,
            "panel": panel,
            "contains_exact_references_or_labels": False,
            "contains_target_ids_or_source_grid_coordinates": False,
            "four_layouts_frozen_before_consensus": True,
            "consensus_support_threshold": 2,
            "consensus_priority": ["support_desc", "original_raw_priority_desc", "identity"],
            "original_costs_retained_for_component_placement_and_fill": True,
            "strict_original_tile_layouts": True,
            "rows": frozen_rows,
        },
    )
    _write_json_exclusive(
        freeze_path,
        {
            "schema": FREEZE_SCHEMA,
            "created_before_exact_reference_recreation": True,
            "contains_evaluation_references_or_labels": False,
            "panel": panel,
            "artifacts": {
                "parent_archive": _record(parent_archive),
                "parent_metadata": _record(parent_metadata),
                "focal_archive": _record(focal_archive),
                "focal_metadata": _record(focal_metadata),
                "logistic_calibrator": _record(logistic_path),
                "nonlinear_calibrator": _record(nonlinear_path),
                "frozen_candidate_archive": _record(frozen_path),
                "frozen_candidate_metadata": _record(metadata_path),
                **{name: _record(path) for name, path in _runtime_sources().items()},
            },
        },
    )
    return frozen_path, metadata_path, freeze_path, perf_counter() - started


def _validate_freeze(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("created_before_exact_reference_recreation") is not True:
        raise RuntimeError("pre-score timing contract changed")
    if payload.get("contains_evaluation_references_or_labels") is not False:
        raise RuntimeError("pre-score freeze contains labels")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise RuntimeError("pre-score artifact roster is absent")
    for name, record in artifacts.items():
        if not isinstance(record, Mapping):
            raise RuntimeError(f"malformed artifact record: {name}")
        raw_path = record.get("path")
        expected = record.get("sha256")
        if not isinstance(raw_path, str) or not isinstance(expected, str):
            raise RuntimeError(f"malformed artifact fields: {name}")
        artifact = Path(raw_path)
        if not artifact.is_absolute():
            artifact = PROJECT_ROOT / artifact
        if sha256_file(artifact.resolve()) != expected:
            raise RuntimeError(f"pre-score artifact changed: {name}")


def _layout_metrics(layout: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    result = evaluate_layout(layout, reference, reference_is_exact=True)
    if result.adjacency_total != PAIR_DENOMINATOR:
        raise RuntimeError("pair denominator changed")
    return {
        "satisfied_adjacent_pairs": int(result.adjacency_correct),
        "adjacency_recall": float(result.adjacency),
        "exact_tiles": int(result.correct_tile_count),
        "strict_permutation": True,
    }


def _score_after_freeze(
    *,
    panel: PanelName,
    rows: Sequence[Mapping[str, Any]],
    frozen_path: Path,
    metadata_path: Path,
    freeze_path: Path,
    targets: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _validate_freeze(freeze_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    frozen_rows = metadata.get("rows")
    if not isinstance(frozen_rows, list) or len(frozen_rows) != len(rows):
        raise RuntimeError("frozen row roster changed")
    lookup = base._load_manifest_lookup()
    cache = base.CleanTileCache(targets.resolve())
    scored: list[dict[str, Any]] = []
    with np.load(frozen_path, allow_pickle=False) as candidate:
        for row, frozen_row in zip(rows, frozen_rows, strict=True):
            for field in ("prefix", "case_id", "source_filename", "draw_index", "dirty_sha256"):
                if row[field] != frozen_row[field]:
                    raise RuntimeError(f"parent and frozen row differ at {field}")
            prefix = str(row["prefix"])
            source = str(row["source_filename"])
            draw = int(row["draw_index"])
            dirty, reference = make_exact_synthetic_case(
                cache.load(lookup[source]),
                source_filename=source,
                draw_index=draw,
                seed=base.SYNTHETIC_SEED,
            )
            if (
                dirty.case_id != row["case_id"]
                or reference.case_id != dirty.case_id
                or base._parent_dirty_sha256(panel, dirty.tiles) != row["dirty_sha256"]
            ):
                raise RuntimeError("scoring recreated a different case")
            exact = _strict_layout(reference.tile_at_position)
            scored.append(
                {
                    "prefix": prefix,
                    "case_id": dirty.case_id,
                    "source_filename": source,
                    "draw_index": draw,
                    **{
                        arm: _layout_metrics(
                            _strict_layout(candidate[f"{prefix}__{arm}_layout"]),
                            exact,
                        )
                        for arm in SCORED_ARMS
                    },
                }
            )

    metrics = ("satisfied_adjacent_pairs", "adjacency_recall", "exact_tiles")
    summary: dict[str, Any] = {
        "pair_denominator": PAIR_DENOMINATOR,
        "arms": {
            arm: {
                metric: float(np.mean([float(row[arm][metric]) for row in scored]))
                for metric in metrics
            }
            for arm in SCORED_ARMS
        },
    }
    full = len(scored) == 32
    sources = [str(row["source_filename"]) for row in scored]
    deltas: dict[str, Any] = {}
    for index, metric in enumerate(metrics):
        values = [
            float(row["consensus_tail96"][metric])
            - float(row["current_tail96"][metric])
            for row in scored
        ]
        deltas[metric] = (
            base._source_clustered_delta_ci(
                values,
                sources,
                seed=BOOTSTRAP_SEED + index,
            )
            if full
            else {"mean": float(np.mean(values)), "smoke_only": True}
        )
    summary["consensus_tail96_minus_current_tail96"] = deltas
    return scored, summary


def run(args: argparse.Namespace) -> None:
    panel: PanelName = args.panel
    parent_archive, parent_metadata = base._panel_paths(panel)
    rows = base._validated_rows(parent_metadata, smoke_one=bool(args.smoke_one))
    focal_archive, focal_metadata = _focal_paths(panel)
    _validate_focal_rows(focal_metadata, base._validated_rows(parent_metadata, smoke_one=False))
    logistic = _require_hash(args.logistic, LOGISTIC_SHA256, name="logistic calibrator")
    nonlinear = _require_hash(args.nonlinear, NONLINEAR_SHA256, name="nonlinear calibrator")
    _require_hash(
        PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py",
        RAW_SOLVER_SHA256,
        name="frozen raw solver",
    )
    if not args.targets.resolve().is_dir():
        raise ValueError(f"target directory is absent: {args.targets}")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else DEFAULT_OUTPUTS[panel].resolve()
    )
    started = perf_counter()
    frozen, metadata, freeze, inference_seconds = _freeze_target_free(
        panel=panel,
        rows=rows,
        parent_archive=parent_archive,
        parent_metadata=parent_metadata,
        focal_archive=focal_archive,
        focal_metadata=focal_metadata,
        logistic_path=logistic,
        nonlinear_path=nonlinear,
        output_dir=output_dir,
    )
    print(
        json.dumps(
            {
                "event": "taska_consensus_layouts_frozen_before_references",
                "panel": panel,
                "case_count": len(rows),
                "archive_sha256": sha256_file(frozen),
                "metadata_sha256": sha256_file(metadata),
                "pre_score_freeze_sha256": sha256_file(freeze),
                "exact_reference_persisted": False,
            }
        ),
        flush=True,
    )
    scored_rows, metrics = _score_after_freeze(
        panel=panel,
        rows=rows,
        frozen_path=frozen,
        metadata_path=metadata,
        freeze_path=freeze,
        targets=args.targets,
    )
    pair_delta = metrics["consensus_tail96_minus_current_tail96"][
        "satisfied_adjacent_pairs"
    ]
    strict = all(
        bool(row[arm]["strict_permutation"]) for row in scored_rows for arm in SCORED_ARMS
    )
    report = {
        "schema": REPORT_SCHEMA,
        "status": "smoke-only" if args.smoke_one else "diagnostic-complete",
        "panel": {
            "name": panel,
            "historically_opened": True,
            "fresh_promotion_claimed": False,
            "case_count": len(scored_rows),
            "full_registered_panel": len(scored_rows) == 32,
        },
        "candidate": {
            "arm_names": list(CONSENSUS_ARM_NAMES),
            "minimum_support": 2,
            "priority": ["support_desc", "original_raw_priority_desc", "identity"],
            "consensus_bonds_are_component_supply": True,
            "original_costs_retained_for_placement_and_fill": True,
            "protected_tail_max_swaps": TAIL_MAX_SWAPS,
            "target_free_inference": True,
        },
        "measurement": {
            "all_layouts_strict": strict,
            "valid": len(scored_rows) == 32 and strict,
            "opened_gate_pair_delta_nonnegative": float(pair_delta["mean"]) >= 0.0,
        },
        "frozen_eval": {
            "archive": _record(frozen),
            "metadata": _record(metadata),
            "pre_score_freeze": _record(freeze),
            "contains_exact_references_or_labels": False,
        },
        "metrics": metrics,
        "rows": scored_rows,
        "runtime_seconds": {
            "target_free_solver": inference_seconds,
            "total": perf_counter() - started,
        },
    }
    _write_json_exclusive(output_dir / "report.json", report)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    run(parse_args())

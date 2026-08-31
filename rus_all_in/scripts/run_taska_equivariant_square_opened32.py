#!/usr/bin/env python3
"""Fixed legal TASKA + permutation-equivariant 2x2 rerank follow-up.

This is an isolated comparison to the legal TASKA replay.  Every calibrated
model/view/orientation scorer is reranked before mutual-best voting, and the
raw v3/local pessimistic fusion is reranked before the raw-tail solve.  The
fixed square recipe is k=16, tau=0.5, one centred top-20 round, weight=0.4.

Unlike historical ``quad_rerank``, no row is interpreted as a board boundary:
all rows are treated identically.  Tile ids therefore denote only the current
unordered input bag.  There is no chooser, verifier, structural border prior,
restored-pixel output, or target-derived feature.

All dirty-only scores, harvest edges, and strict layouts are hash-frozen before
the already-opened development references are recreated.  ``--smoke-one`` is
the only intended mode until the isolated arm has been reviewed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np
import torch

from aiijc_puzzle.layout_evaluation import LayoutEvaluation, evaluate_layout
from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.raw_tail_global_solver import RawTailEdge, solve_raw_tail_global
from aiijc_puzzle.taska_equivariant_square import (
    DEFAULT_SQUARE_WEIGHT,
    SQUARE_ROUNDS,
    SQUARE_SHORTLIST,
    SQUARE_SUPPORT_K,
    SQUARE_TEMPERATURE,
    equivariant_square_rerank,
)

try:
    import scripts.run_taska_seam_replay_opened32 as base
except ModuleNotFoundError:  # Direct ``python scripts/run_*.py`` execution.
    import run_taska_seam_replay_opened32 as base  # type: ignore[no-redef]


EXPERIMENT = "taska-equivariant-square-opened32-v1"
REPORT_SCHEMA = "aiijc-taska-equivariant-square-report-v1"
FROZEN_SCHEMA = "aiijc-taska-equivariant-square-frozen-target-free-v1"
DEFAULT_OUTPUT = base.PROJECT_ROOT / "outputs/taska-equivariant-square/smoke1-mps-v1"

SQUARE_WEIGHT = 0.4
VIEWS = ("raw", "median", "bilateral")
ORIENTATIONS = 2
VOTE_TARGET = 350
VOTES_FALLBACK = 10
MARGIN = 0.0

RUNTIME_SOURCE_PATHS = {
    "runner": Path(__file__).resolve(),
    "equivariant_square": (
        base.PROJECT_ROOT / "src/aiijc_puzzle/taska_equivariant_square.py"
    ),
    "taska_seam_matcher": base.RUNTIME_SOURCE_PATHS["taska_seam_matcher"],
    "raw_tail_global_solver": base.RUNTIME_SOURCE_PATHS["raw_tail_global_solver"],
    "base_replay_runner": base.RUNTIME_SOURCE_PATHS["replay_runner"],
    "parent_pilot_runner": base.RUNTIME_SOURCE_PATHS["parent_pilot_runner"],
}


@dataclass(frozen=True)
class RunPaths:
    frozen_eval: Path
    frozen_eval_metadata: Path
    pre_score_freeze: Path
    report: Path


@dataclass(frozen=True)
class SquareMatchResult:
    right_log: np.ndarray
    down_log: np.ndarray
    cost_right: np.ndarray
    cost_down: np.ndarray
    candidate_edges: tuple[RawTailEdge, ...]
    vote_counts: tuple[int, ...]
    minimum_margins: tuple[float, ...]
    chosen_vote_threshold: int
    scorer_count: int
    scorer_audit: tuple[dict[str, Any], ...]
    fused_audit: dict[str, Any]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=base.DEFAULT_CONFIG)
    parser.add_argument("--targets", type=Path, default=base.DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--allow-nondeterministic-mps", action="store_true")
    parser.add_argument(
        "--smoke-one",
        action="store_true",
        help="run one case only; a full opened32 run requires explicit review first",
    )
    return parser.parse_args(argv)


def _prepare_paths(output_dir: Path) -> RunPaths:
    root = output_dir.resolve()
    paths = RunPaths(
        frozen_eval=root / "frozen-target-free-eval.npz",
        frozen_eval_metadata=root / "frozen-target-free-eval.json",
        pre_score_freeze=root / "pre-score-freeze.json",
        report=root / "report.json",
    )
    if any(path.exists() for path in asdict(paths).values()):
        raise FileExistsError("refusing to overwrite a square follow-up run")
    root.mkdir(parents=True, exist_ok=True)
    return paths


def _finite_matrix(value: Any, *, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (base.COUNT, base.COUNT):
        raise ValueError(f"{name} must have shape {(base.COUNT, base.COUNT)}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(matrix)


def _matrix_sha256(value: np.ndarray) -> str:
    matrix = np.ascontiguousarray(value, dtype="<f8")
    return hashlib.sha256(matrix.tobytes()).hexdigest()


def _cost_from_log(value: np.ndarray) -> np.ndarray:
    cost = -np.asarray(value, dtype=np.float64)
    cost -= cost.min()
    np.fill_diagonal(cost, 0.0)
    return np.ascontiguousarray(cost)


def _mutual_edges(
    matrix: np.ndarray,
    axis: Literal["right", "down"],
) -> dict[RawTailEdge, float]:
    """Permutation-equivariant extension of historical mutual-best edges."""

    scores = np.array(matrix, dtype=np.float64, copy=True)
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("mutual scorer must be one square matrix")
    if not np.isfinite(scores).all():
        raise ValueError("mutual scorer must contain only finite values")
    if len(scores) < 3:
        raise ValueError("mutual scorer needs at least three tiles")
    np.fill_diagonal(scores, -np.inf)
    row_maximum = scores.max(axis=1)
    column_maximum = scores.max(axis=0)
    mutual = (scores == row_maximum[:, None]) & (scores == column_maximum[None, :])
    partition = np.partition(scores, -2, axis=1)
    margins = partition[:, -1] - partition[:, -2]
    rows, columns = np.nonzero(mutual)
    return {
        RawTailEdge(int(source), int(target), axis): float(margins[source])
        for source, target in zip(rows, columns, strict=True)
    }


def _vote_threshold(
    scorers: Sequence[Mapping[RawTailEdge, float]],
    *,
    target: int = VOTE_TARGET,
    fallback: int = VOTES_FALLBACK,
) -> int:
    if not scorers:
        raise ValueError("at least one scorer is required")
    if target < 0:
        raise ValueError("vote target must be non-negative")
    if not 1 <= fallback <= len(scorers):
        raise ValueError("fallback threshold must fit the scorer count")
    if not target:
        return fallback
    counts: dict[RawTailEdge, int] = {}
    for scorer in scorers:
        for edge in scorer:
            counts[edge] = counts.get(edge, 0) + 1
    for threshold in range(len(scorers), 0, -1):
        if sum(count >= threshold for count in counts.values()) >= target:
            return threshold
    return 1


def _square_match(
    tiles: np.ndarray,
    matchers: Sequence[Any],
    *,
    device: torch.device,
) -> SquareMatchResult:
    from aiijc_puzzle.taska_seam_matcher import (
        ORIENTATIONS as TASKA_ORIENTATIONS,
    )
    from aiijc_puzzle.taska_seam_matcher import (
        analytic_view,
        calibrated_log_assignments,
        pessimistic_log_assignments,
    )

    if len(matchers) != 2:
        raise ValueError("the fixed square arm requires v3 + local")
    prepared_views = {
        "raw": np.ascontiguousarray(tiles),
        "median": analytic_view("median", tiles),
        "bilateral": analytic_view("bilateral", tiles),
    }
    scorer_sets: list[dict[RawTailEdge, float]] = []
    scorer_audit: list[dict[str, Any]] = []
    for model_index, matcher in enumerate(matchers):
        for view_name in VIEWS:
            for orientation_index, orientation in enumerate(
                TASKA_ORIENTATIONS[:ORIENTATIONS]
            ):
                calibrated_right, calibrated_down = calibrated_log_assignments(
                    matcher,
                    prepared_views[view_name],
                    device=device,
                    orientation=orientation,
                    rounds=base.CYCLE_ROUNDS,
                    cycle_weight=base.CYCLE_WEIGHT,
                    sinkhorn_iterations=base.SINKHORN_ITERATIONS,
                    acyclic_weight=base.ACYCLIC_WEIGHT,
                )
                square_right, square_down = equivariant_square_rerank(
                    calibrated_right,
                    calibrated_down,
                    weight=SQUARE_WEIGHT,
                )
                scorer_sets.append(
                    {
                        **_mutual_edges(square_right, "right"),
                        **_mutual_edges(square_down, "down"),
                    }
                )
                scorer_audit.append(
                    {
                        "model_index": model_index,
                        "view": view_name,
                        "orientation_index": orientation_index,
                        "orientation": list(orientation),
                        "calibrated_right_sha256": _matrix_sha256(calibrated_right),
                        "calibrated_down_sha256": _matrix_sha256(calibrated_down),
                        "square_right_sha256": _matrix_sha256(square_right),
                        "square_down_sha256": _matrix_sha256(square_down),
                        "mutual_edge_count": len(scorer_sets[-1]),
                    }
                )

    fused_right_before, fused_down_before = pessimistic_log_assignments(
        matchers,
        prepared_views["raw"],
        device=device,
        rounds=base.CYCLE_ROUNDS,
        cycle_weight=base.CYCLE_WEIGHT,
        sinkhorn_iterations=base.SINKHORN_ITERATIONS,
        acyclic_weight=base.ACYCLIC_WEIGHT,
    )
    fused_right, fused_down = equivariant_square_rerank(
        fused_right_before,
        fused_down_before,
        weight=SQUARE_WEIGHT,
    )
    threshold = _vote_threshold(scorer_sets)
    records: list[tuple[RawTailEdge, int, float]] = []
    all_edges = set().union(*(set(scorer) for scorer in scorer_sets))
    for edge in all_edges:
        margins = [scorer[edge] for scorer in scorer_sets if edge in scorer]
        if len(margins) >= threshold and min(margins) >= MARGIN:
            records.append((edge, len(margins), float(min(margins))))
    axis_order = {"right": 0, "down": 1}
    records.sort(
        key=lambda record: (
            axis_order[record[0].axis],
            record[0].source,
            record[0].target,
        )
    )
    return SquareMatchResult(
        right_log=_finite_matrix(fused_right, name="fused_right"),
        down_log=_finite_matrix(fused_down, name="fused_down"),
        cost_right=_cost_from_log(fused_right),
        cost_down=_cost_from_log(fused_down),
        candidate_edges=tuple(record[0] for record in records),
        vote_counts=tuple(record[1] for record in records),
        minimum_margins=tuple(record[2] for record in records),
        chosen_vote_threshold=threshold,
        scorer_count=len(scorer_sets),
        scorer_audit=tuple(scorer_audit),
        fused_audit={
            "pessimistic_right_before_square_sha256": _matrix_sha256(fused_right_before),
            "pessimistic_down_before_square_sha256": _matrix_sha256(fused_down_before),
            "pessimistic_right_after_square_sha256": _matrix_sha256(fused_right),
            "pessimistic_down_after_square_sha256": _matrix_sha256(fused_down),
        },
    )


def _freeze_predictions(
    paths: RunPaths,
    specs: Sequence[tuple[Mapping[str, Any], str, int]],
    parent_rows: Sequence[Mapping[str, Any]],
    artifacts: Any,
    *,
    targets: Path,
    device: torch.device,
) -> tuple[float, int]:
    if len(specs) != len(parent_rows):
        raise ValueError("candidate and parent rosters differ")
    matchers = base._load_matchers(artifacts, device=device)
    target_cache = base.CleanTileCache(targets)
    arrays: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    started = perf_counter()
    for index, ((record, source_name, draw), parent_row) in enumerate(
        zip(specs, parent_rows, strict=True)
    ):
        dirty = base._dirty_case(target_cache, record, source_name, draw)
        dirty_sha = base._dirty_sha256(dirty.dirty_tiles)
        if (
            dirty.case_id != parent_row["case_id"]
            or source_name != parent_row["source_filename"]
            or draw != int(parent_row["draw_index"])
            or dirty_sha != parent_row["dirty_sha256"]
        ):
            raise RuntimeError("square replay recreated a different synthetic case")
        matched = _square_match(dirty.dirty_tiles, matchers, device=device)
        solver = solve_raw_tail_global(
            matched.cost_right,
            matched.cost_down,
            matched.candidate_edges,
            border_unary=None,
            grid=base.GRID,
            config=base.SOLVER_CONFIG,
        )
        layout = base._strict_layout(solver.layout)
        prefix = f"case_{index:04d}"
        arrays[f"{prefix}__cost_right"] = matched.cost_right.astype(np.float32)
        arrays[f"{prefix}__cost_down"] = matched.cost_down.astype(np.float32)
        arrays[f"{prefix}__right_log"] = matched.right_log.astype(np.float32)
        arrays[f"{prefix}__down_log"] = matched.down_log.astype(np.float32)
        arrays[f"{prefix}__edge_source"] = np.asarray(
            [edge.source for edge in matched.candidate_edges], dtype=np.int32
        )
        arrays[f"{prefix}__edge_target"] = np.asarray(
            [edge.target for edge in matched.candidate_edges], dtype=np.int32
        )
        arrays[f"{prefix}__edge_axis"] = np.asarray(
            [0 if edge.axis == "right" else 1 for edge in matched.candidate_edges],
            dtype=np.uint8,
        )
        arrays[f"{prefix}__edge_vote_count"] = np.asarray(
            matched.vote_counts, dtype=np.int16
        )
        arrays[f"{prefix}__edge_minimum_margin"] = np.asarray(
            matched.minimum_margins, dtype=np.float32
        )
        arrays[f"{prefix}__taska_square_layout"] = layout
        rows.append(
            {
                "prefix": prefix,
                "case_id": dirty.case_id,
                "source_filename": source_name,
                "draw_index": draw,
                "dirty_sha256": dirty_sha,
                "candidate_edge_count": len(matched.candidate_edges),
                "chosen_vote_threshold": matched.chosen_vote_threshold,
                "scorer_count": matched.scorer_count,
                "scorer_audit": list(matched.scorer_audit),
                "fused_audit": matched.fused_audit,
                "solver_diagnostics": solver.diagnostics.as_dict(),
            }
        )
        print(
            json.dumps(
                {
                    "event": "taska_equivariant_square_case_frozen_in_memory",
                    "case": index + 1,
                    "case_count": len(specs),
                    "harvest_edges": len(matched.candidate_edges),
                    "vote_threshold": matched.chosen_vote_threshold,
                    "strict": True,
                }
            ),
            flush=True,
        )
    base._write_npz_exclusive(paths.frozen_eval, arrays)
    base._write_json_exclusive(
        paths.frozen_eval_metadata,
        {
            "schema": FROZEN_SCHEMA,
            "contains_exact_references_or_labels": False,
            "contains_target_ids_or_source_grid_coordinates": False,
            "contains_dirty_derived_scores": True,
            "contains_frozen_harvest_membership": True,
            "contains_strict_original_tile_layouts": True,
            "square": {
                "support_k": SQUARE_SUPPORT_K,
                "temperature": SQUARE_TEMPERATURE,
                "rounds": SQUARE_ROUNDS,
                "shortlist": SQUARE_SHORTLIST,
                "weight": SQUARE_WEIGHT,
                "all_source_rows": True,
                "index_derived_boundary_masks": False,
                "complete_cutoff_tie_blocks": True,
            },
            "border_prior_used": False,
            "rows": rows,
        },
    )
    return perf_counter() - started, len(rows)


def _artifact_roster(
    paths: RunPaths,
    artifacts: Any,
    *,
    config_path: Path,
    config_sha256: str,
) -> dict[str, dict[str, str]]:
    sidecar = Path(f"{config_path.resolve()}.sha256")
    roster = {
        "base_config": {"path": base._project_path(config_path), "sha256": config_sha256},
        "base_config_sidecar": base._record(sidecar),
        **{
            name: base._record(getattr(artifacts, name))
            for name in base.ARTIFACT_KEYS
        },
        **{name: base._record(path) for name, path in RUNTIME_SOURCE_PATHS.items()},
        "frozen_target_free_eval": base._record(paths.frozen_eval),
        "frozen_target_free_eval_metadata": base._record(paths.frozen_eval_metadata),
    }
    base._write_json_exclusive(
        paths.pre_score_freeze,
        {
            "schema": "aiijc-taska-equivariant-square-pre-score-freeze-v1",
            "created_before_eval_reference_recreation": True,
            "contains_evaluation_references_or_labels": False,
            "already_opened_development_panel": True,
            "freshness_claimed": False,
            "artifacts": roster,
        },
    )
    return roster


def _validate_roster(paths: RunPaths, expected: Mapping[str, Mapping[str, str]]) -> None:
    payload = json.loads(paths.pre_score_freeze.read_text(encoding="utf-8"))
    if payload.get("created_before_eval_reference_recreation") is not True:
        raise RuntimeError("pre-score freeze timing contract changed")
    if payload.get("contains_evaluation_references_or_labels") is not False:
        raise RuntimeError("pre-score freeze contains evaluation labels")
    if payload.get("artifacts") != expected:
        raise RuntimeError("pre-score artifact roster changed")
    for name, record in expected.items():
        path = base._resolve_path(record, name=f"pre_score_{name}")
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"pre-score artifact changed: {name}")


def _layout_metrics(evaluation: LayoutEvaluation) -> dict[str, Any]:
    if evaluation.adjacency_total != base.PAIR_DENOMINATOR:
        raise RuntimeError("adjacency denominator changed")
    pairs = int(evaluation.adjacency_correct)
    recall = float(evaluation.adjacency)
    if pairs != round(recall * evaluation.adjacency_total):
        raise RuntimeError("pair count and adjacency recall disagree")
    return {
        "satisfied_adjacent_pairs": pairs,
        "adjacency_recall": recall,
        "exact_tiles": int(evaluation.correct_tile_count),
        "strict_permutation": True,
    }


def _summary(rows: Sequence[Mapping[str, Any]], *, full_panel: bool) -> dict[str, Any]:
    arms = ("union_v2", "learned_priority", "taska_equivariant_square_raw_tail")
    metrics = ("satisfied_adjacent_pairs", "adjacency_recall", "exact_tiles")
    result: dict[str, Any] = {
        "arms": {
            arm: {
                metric: float(np.mean([float(row[arm][metric]) for row in rows]))
                for metric in metrics
            }
            for arm in arms
        }
    }
    sources = [str(row["source_filename"]) for row in rows]
    result["candidate_deltas"] = {}
    for baseline_index, baseline_name in enumerate(("union_v2", "learned_priority")):
        by_metric: dict[str, Any] = {}
        for metric_index, metric in enumerate(metrics):
            values = [
                float(row["taska_equivariant_square_raw_tail"][metric])
                - float(row[baseline_name][metric])
                for row in rows
            ]
            by_metric[metric] = (
                base.source_clustered_delta_ci(
                    values,
                    sources,
                    seed=base.BOOTSTRAP_SEED + baseline_index * 10 + metric_index,
                )
                if full_panel
                else {
                    "mean": float(np.mean(values)),
                    "ci95_lower": None,
                    "ci95_upper": None,
                    "smoke_only": True,
                }
            )
        result["candidate_deltas"][baseline_name] = by_metric
    return result


def _score_frozen(
    paths: RunPaths,
    artifacts: Any,
    commitment: Mapping[str, Any],
    expected_roster: Mapping[str, Mapping[str, str]],
    *,
    targets: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # No target cache or exact reference may be recreated before this check.
    _validate_roster(paths, expected_roster)
    metadata = json.loads(paths.frozen_eval_metadata.read_text(encoding="utf-8"))
    parent_metadata = json.loads(
        artifacts.parent_frozen_eval_metadata.read_text(encoding="utf-8")
    )
    candidate_rows = metadata["rows"]
    parent_rows = parent_metadata["rows"][: len(candidate_rows)]
    lookup = base._manifest_lookup(commitment)
    target_cache = base.CleanTileCache(targets)
    rows: list[dict[str, Any]] = []
    with (
        np.load(artifacts.parent_frozen_eval) as parent_archive,
        np.load(paths.frozen_eval) as candidate_archive,
    ):
        for candidate_row, parent_row in zip(candidate_rows, parent_rows, strict=True):
            identity = ("prefix", "case_id", "source_filename", "draw_index", "dirty_sha256")
            if any(candidate_row[name] != parent_row[name] for name in identity):
                raise RuntimeError("candidate and parent frozen identities differ")
            source_name = str(candidate_row["source_filename"])
            draw = int(candidate_row["draw_index"])
            case = base.prepare_case(
                target_cache,
                lookup[source_name],
                draw_index=draw,
                seed=base.SYNTHETIC_SEED,
            )
            if (
                str(case.case_id) != candidate_row["case_id"]
                or base._dirty_sha256(case.dirty_tiles) != candidate_row["dirty_sha256"]
            ):
                raise RuntimeError("scoring recreated a different synthetic case")
            reference = base._strict_layout(np.argsort(case.input_tile_to_position))
            prefix = str(candidate_row["prefix"])
            layouts = {
                "union_v2": base._strict_layout(
                    parent_archive[f"{prefix}__union_v2_layout"]
                ),
                "learned_priority": base._strict_layout(
                    parent_archive[f"{prefix}__learned_priority_layout"]
                ),
                "taska_equivariant_square_raw_tail": base._strict_layout(
                    candidate_archive[f"{prefix}__taska_square_layout"]
                ),
            }
            row: dict[str, Any] = {
                "source_filename": source_name,
                "draw_index": draw,
                "case_id": str(case.case_id),
            }
            for arm, layout in layouts.items():
                row[arm] = _layout_metrics(
                    evaluate_layout(layout, reference, reference_is_exact=True)
                )
            rows.append(row)
    return rows, _summary(rows, full_panel=len(rows) == base.EVAL_CASE_COUNT)


def run(args: argparse.Namespace) -> None:
    if not args.smoke_one:
        raise ValueError("full opened32 run is blocked until root reviews the smoke arm")
    if DEFAULT_SQUARE_WEIGHT != SQUARE_WEIGHT:
        raise RuntimeError("fixed candidate square weight drifted")
    config, config_sha, source_names = base._load_preregistration(args.config)
    artifacts = base._validate_artifacts(config)
    commitment, parent_rows = base._validate_parent(artifacts, source_names)
    specs = base._case_specs(source_names, base._manifest_lookup(commitment))[:1]
    parent_rows = parent_rows[:1]
    paths = _prepare_paths(args.output_dir)
    device = base._select_device(
        args.device,
        allow_nondeterministic_mps=args.allow_nondeterministic_mps,
    )
    started = perf_counter()
    inference_seconds, frozen_count = _freeze_predictions(
        paths,
        specs,
        parent_rows,
        artifacts,
        targets=args.targets,
        device=device,
    )
    roster = _artifact_roster(
        paths,
        artifacts,
        config_path=args.config,
        config_sha256=config_sha,
    )
    print(
        json.dumps(
            {
                "event": "taska_equivariant_square_frozen_before_scoring",
                "frozen_eval_sha256": sha256_file(paths.frozen_eval),
                "frozen_metadata_sha256": sha256_file(paths.frozen_eval_metadata),
                "pre_score_freeze_sha256": sha256_file(paths.pre_score_freeze),
                "case_count": frozen_count,
            }
        ),
        flush=True,
    )
    rows, metrics = _score_frozen(
        paths,
        artifacts,
        commitment,
        roster,
        targets=args.targets,
    )
    base._write_json_exclusive(
        paths.report,
        {
            "schema": REPORT_SCHEMA,
            "status": "smoke-only",
            "experiment": EXPERIMENT,
            "panel": {
                "previously_opened": True,
                "freshness_claimed": False,
                "evaluated_case_count": len(rows),
                "full_run_blocked_pending_review": True,
            },
            "candidate": {
                "single_fixed_arm": True,
                "views": list(VIEWS),
                "orientations": ORIENTATIONS,
                "vote_target": VOTE_TARGET,
                "votes_fallback": VOTES_FALLBACK,
                "square_support_k": SQUARE_SUPPORT_K,
                "square_temperature": SQUARE_TEMPERATURE,
                "square_rounds": SQUARE_ROUNDS,
                "square_shortlist": SQUARE_SHORTLIST,
                "square_weight": SQUARE_WEIGHT,
                "square_applied_to_every_vote_scorer": True,
                "square_applied_to_pessimistic_raw_fusion": True,
                "all_source_rows": True,
                "index_derived_boundary_masks": False,
                "chooser_verifier_or_border_prior_used": False,
                "solver": asdict(base.SOLVER_CONFIG),
            },
            "frozen_eval": {
                "archive": base._record(paths.frozen_eval),
                "metadata": base._record(paths.frozen_eval_metadata),
                "pre_score_freeze": base._record(paths.pre_score_freeze),
                "scores_harvest_and_layouts_frozen_before_references": True,
                "contains_exact_references_or_labels": False,
            },
            "metrics": metrics,
            "rows": rows,
            "runtime_seconds": {
                "target_free_inference": inference_seconds,
                "total": perf_counter() - started,
            },
            "legality": {
                "dirty_tiles_only_for_candidate_inference": True,
                "target_ids_or_references_used_by_candidate": False,
                "index_derived_boundary_mask_used": False,
                "original_upright_tile_permutations_only": True,
                "restored_pixels_emitted": False,
                "competition_test_accessed": False,
            },
        },
    )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

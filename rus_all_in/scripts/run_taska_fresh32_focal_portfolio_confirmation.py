#!/usr/bin/env python3
"""Replay the fixed focal and four-arm TASKA leaders on frozen fresh32.

The parent current-disjoint panel already exposed its exact references while
confirming protected-tail budgets.  This follow-up therefore is not a fresh
promotion.  It nevertheless preserves the important target-free ordering:
the signed parent recipe recreates each dirty bag; focal logits, all component
layouts, the all-bond portfolio choice, and its 96-swap protected tail are
hash-frozen before this process reconstructs an exact reference.

There is no parameter sweep.  The two candidates are exactly:

* the recovered focal verifier with its training-exact top-5 features; and
* raw/logistic/focal/nonlinear layouts, selected by minimum original all-bond
  TASKA seam cost, followed by the already confirmed 96-swap protected tail.

All layouts are strict permutations of the 576 original upright dirty tiles.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from aiijc_puzzle.layout_evaluation import evaluate_layout
from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.synthetic_socket_evaluation import make_exact_synthetic_case
from aiijc_puzzle.taska_edge_calibrator import (
    TaskaEdgeCalibrator,
    extract_taska_edge_features,
    solve_prioritized_raw_tail_global,
)
from aiijc_puzzle.taska_focal_verifier import (
    TASKA_FOCAL_FEATURE_TOP_K,
    TASKA_FOCAL_VERIFIER_SHA256,
    load_taska_focal_verifier,
    score_focal_edges,
    solve_focal_raw_tail_global,
)
from aiijc_puzzle.taska_layout_portfolio import select_lowest_taska_seam_cost_layout
from aiijc_puzzle.taska_nonlinear_calibrator import TaskaNonlinearCalibrator
from aiijc_puzzle.taska_protected_tail_polish import polish_unprotected_taska_tail
from aiijc_puzzle.taska_seam_matcher import load_default_taska_ensemble, match_taska_tiles

try:
    from scripts import run_taska_protected_tail_fresh32_confirmation as parent
except ModuleNotFoundError:  # Direct ``python scripts/run_*.py`` execution.
    import run_taska_protected_tail_fresh32_confirmation as parent

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/taska-fresh32-leader-confirmation/fresh-held32-mps-v1"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "artifacts/prior-taska/ckpt/verify_pair_best.pt"
DEFAULT_LOGISTIC = PROJECT_ROOT / "outputs/taska-edge-calibrator/train256-v1/calibrator.npz"
DEFAULT_NONLINEAR = PROJECT_ROOT / "outputs/taska-nonlinear-calibrator/train256-v1/calibrator.npz"
PARENT_ARCHIVE = parent.DEFAULT_OUTPUT / "frozen-target-free-eval.npz"
PARENT_METADATA = parent.DEFAULT_OUTPUT / "frozen-target-free-eval.json"

CONFIG_SHA256 = "9854ef20c479ab358887896b81bf93263a3bdcd7d7014d6a310b7134fb4daad7"
PARENT_ARCHIVE_SHA256 = "d7b156ff1a8cdab702881242e48797b1a18f750a2d6a60f2a7d769dbfa1bffc1"
PARENT_METADATA_SHA256 = "1acb5d0000dd76e48fb6c079827fa2113bb56f541905fc97ced9656b8d7fe53f"
LOGISTIC_SHA256 = "adc76ee87fc112d4ca3eeb676cdec6b7d103c596d62a9848ba65ee5ef384b1ac"
NONLINEAR_SHA256 = "2a5f95bd9d8e08e57b8bd02e242e25ef4661036ed3b1985fda1d70ee1bf9d2a6"
RAW_SOLVER_SHA256 = "97859e1f4ff6ceadf56ebccf429f2944ae7a5726631731d9f776224a667eb486"

FROZEN_SCHEMA = "aiijc-taska-fresh32-leader-target-free-v1"
FREEZE_SCHEMA = "aiijc-taska-fresh32-leader-pre-score-freeze-v1"
REPORT_SCHEMA = "aiijc-taska-fresh32-leader-confirmation-report-v1"
FOCAL_MODE = "train_exact_top5"
PORTFOLIO_ARMS = ("raw", "logistic", "focal", "nonlinear")
SCORED_ARMS = (*PORTFOLIO_ARMS, "portfolio", "portfolio_tail96")
PRIMARY_ARMS = ("focal", "portfolio_tail96")
PAIR_DENOMINATOR = parent.PAIR_DENOMINATOR
CASE_COUNT = parent.CASE_COUNT
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 1_982_331_541
MATRIX_REPLAY_ATOL = 3e-6


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=parent.DEFAULT_CONFIG)
    parser.add_argument("--targets", type=Path, default=parent.DEFAULT_TARGETS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--logistic", type=Path, default=DEFAULT_LOGISTIC)
    parser.add_argument("--nonlinear", type=Path, default=DEFAULT_NONLINEAR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--allow-nondeterministic-mps", action="store_true")
    parser.add_argument("--smoke-one", action="store_true")
    return parser.parse_args(argv)


def _require_hash(path: Path, expected: str, *, name: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"{name} does not exist: {resolved}")
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
    layout = np.ascontiguousarray(value, dtype=np.int32)
    if layout.shape != (parent.COUNT,) or not np.array_equal(
        np.sort(layout), np.arange(parent.COUNT)
    ):
        raise ValueError("layout is not a strict original-tile permutation")
    return layout


def _finite_matrix(archive: Any, key: str) -> np.ndarray:
    matrix = np.asarray(archive[key], dtype=np.float64)
    if matrix.shape != (parent.COUNT, parent.COUNT) or not np.isfinite(matrix).all():
        raise ValueError(f"{key} is not one finite 576x576 matrix")
    return np.ascontiguousarray(matrix)


def _edges_from_archive(archive: Any, prefix: str) -> tuple[RawTailEdge, ...]:
    sources = np.asarray(archive[f"{prefix}__edge_source"], dtype=np.int64)
    targets = np.asarray(archive[f"{prefix}__edge_target"], dtype=np.int64)
    axes = np.asarray(archive[f"{prefix}__edge_axis"], dtype=np.uint8)
    if not (sources.ndim == targets.ndim == axes.ndim == 1):
        raise ValueError("frozen edge arrays must be one-dimensional")
    if not (len(sources) == len(targets) == len(axes)) or not np.isin(axes, (0, 1)).all():
        raise ValueError("frozen edge arrays are not aligned or have invalid axes")
    return tuple(
        RawTailEdge(int(source), int(target), "right" if int(axis) == 0 else "down")
        for source, target, axis in zip(sources, targets, axes, strict=True)
    )


def _runtime_sources() -> dict[str, Path]:
    return {
        "confirmation_runner": Path(__file__).resolve(),
        "signed_parent_runner": Path(parent.__file__).resolve(),
        "focal_verifier": PROJECT_ROOT / "src/aiijc_puzzle/taska_focal_verifier.py",
        "edge_calibrator": PROJECT_ROOT / "src/aiijc_puzzle/taska_edge_calibrator.py",
        "nonlinear_calibrator": PROJECT_ROOT / "src/aiijc_puzzle/taska_nonlinear_calibrator.py",
        "layout_portfolio": PROJECT_ROOT / "src/aiijc_puzzle/taska_layout_portfolio.py",
        "protected_tail": PROJECT_ROOT / "src/aiijc_puzzle/taska_protected_tail_polish.py",
        "frozen_raw_solver": PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py",
    }


def _load_parent_context(
    config_path: Path,
    *,
    smoke_one: bool,
) -> tuple[
    parent.Artifacts,
    list[tuple[Mapping[str, Any], str, int]],
    list[Mapping[str, Any]],
]:
    config, config_sha = parent._load_config(config_path)
    if config_sha != CONFIG_SHA256:
        raise RuntimeError("signed fresh32 config SHA-256 changed")
    artifacts = parent._validate_artifacts(config)
    source_names = parent._validate_preregistration(config, artifacts)
    lookup = parent._load_manifest(artifacts)
    specs = parent._case_specs(source_names, lookup)
    metadata = json.loads(PARENT_METADATA.read_text(encoding="utf-8"))
    rows = metadata.get("rows")
    if metadata.get("contains_exact_references_or_labels") is not False:
        raise ValueError("parent fresh32 metadata is not target-free")
    if not isinstance(rows, list) or len(rows) != CASE_COUNT:
        raise ValueError("parent fresh32 metadata must contain exactly 32 rows")
    for row, (_, source, draw) in zip(rows, specs, strict=True):
        if row.get("source_filename") != source or int(row.get("draw_index", -1)) != draw:
            raise RuntimeError("signed roster and frozen parent row order differ")
    if smoke_one:
        return artifacts, specs[:1], rows[:1]
    return artifacts, specs, rows


def _freeze_target_free(
    *,
    specs: Sequence[tuple[Mapping[str, Any], str, int]],
    parent_rows: Sequence[Mapping[str, Any]],
    artifacts: parent.Artifacts,
    targets: Path,
    checkpoint: Path,
    logistic_path: Path,
    nonlinear_path: Path,
    output_dir: Path,
    device: torch.device,
) -> tuple[Path, Path, Path, float]:
    output_dir.mkdir(parents=True, exist_ok=False)
    frozen_path = output_dir / "frozen-target-free-eval.npz"
    metadata_path = output_dir / "frozen-target-free-eval.json"
    freeze_path = output_dir / "pre-score-freeze.json"

    matchers = load_default_taska_ensemble(artifacts.matcher_v3.parent, device=device)
    focal_model = load_taska_focal_verifier(checkpoint, device=device)
    logistic = TaskaEdgeCalibrator.load_npz(logistic_path)
    nonlinear = TaskaNonlinearCalibrator.load_npz(nonlinear_path)
    cache = parent.CleanTileCache(targets.resolve())
    arrays: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    started = perf_counter()

    with np.load(PARENT_ARCHIVE, allow_pickle=False) as archive:
        for index, (spec, row) in enumerate(zip(specs, parent_rows, strict=True)):
            record, source, draw = spec
            prefix = str(row["prefix"])
            dirty = parent._dirty_case(cache, record, source, draw)
            dirty_sha = parent._dirty_sha256(dirty.dirty_tiles)
            if (
                dirty.case_id != row["case_id"]
                or dirty_sha != row["dirty_sha256"]
                or dirty.source_filename != source
                or dirty.draw_index != draw
            ):
                raise RuntimeError("signed recipe recreated a different dirty bag")

            cost_right = _finite_matrix(archive, f"{prefix}__cost_right")
            cost_down = _finite_matrix(archive, f"{prefix}__cost_down")
            right_log = _finite_matrix(archive, f"{prefix}__right_log")
            down_log = _finite_matrix(archive, f"{prefix}__down_log")
            edges = _edges_from_archive(archive, prefix)
            raw_layout = _strict_layout(archive[f"{prefix}__taska_legal_raw_tail"])

            replay = match_taska_tiles(
                dirty.dirty_tiles,
                matchers,
                config=parent.MATCHER_CONFIG,
                device=device,
            )
            if tuple(replay.candidate_edges) != edges:
                raise RuntimeError("replayed matcher changed frozen candidate membership/order")
            matrix_deltas = {
                "cost_right": float(np.max(np.abs(np.asarray(replay.cost_right) - cost_right))),
                "cost_down": float(np.max(np.abs(np.asarray(replay.cost_down) - cost_down))),
                "right_log": float(np.max(np.abs(np.asarray(replay.right_log) - right_log))),
                "down_log": float(np.max(np.abs(np.asarray(replay.down_log) - down_log))),
            }
            if max(matrix_deltas.values()) > MATRIX_REPLAY_ATOL:
                raise RuntimeError(
                    f"matcher replay exceeded frozen matrix tolerance: {matrix_deltas}"
                )
            if tuple(vote.edge for vote in replay.vote_records) != edges:
                raise RuntimeError("replayed vote records differ from frozen edge order")
            margins = np.asarray(
                [vote.minimum_margin for vote in replay.vote_records], dtype=np.float64
            )
            votes = np.asarray([vote.vote_count for vote in replay.vote_records], dtype=np.float64)
            edge_features = extract_taska_edge_features(
                cost_right,
                cost_down,
                right_log,
                down_log,
                edges,
                margins,
                votes,
                grid=parent.GRID,
            ).values

            logistic_logits = logistic.predict_logits(edge_features)
            logistic_priorities = logistic.predict_priorities(edge_features)
            logistic_solved = solve_prioritized_raw_tail_global(
                cost_right,
                cost_down,
                edges,
                logistic_priorities,
                grid=parent.GRID,
                config=parent.SOLVER_CONFIG,
            )
            logistic_layout = _strict_layout(logistic_solved.layout)

            nonlinear_logits = nonlinear.predict_logits(edge_features)
            nonlinear_priorities = nonlinear.predict_priorities(edge_features)
            nonlinear_solved = solve_prioritized_raw_tail_global(
                cost_right,
                cost_down,
                edges,
                nonlinear_priorities,
                grid=parent.GRID,
                config=parent.SOLVER_CONFIG,
            )
            nonlinear_layout = _strict_layout(nonlinear_solved.layout)

            focal_scores = score_focal_edges(
                focal_model,
                dirty.dirty_tiles,
                cost_right,
                cost_down,
                edges,
                mode=FOCAL_MODE,
                grid=parent.GRID,
            )
            focal_solved = solve_focal_raw_tail_global(
                cost_right,
                cost_down,
                focal_scores,
                grid=parent.GRID,
                config=parent.SOLVER_CONFIG,
            )
            focal_layout = _strict_layout(focal_solved.layout)

            layouts = {
                "raw": raw_layout,
                "logistic": logistic_layout,
                "focal": focal_layout,
                "nonlinear": nonlinear_layout,
            }
            selection = select_lowest_taska_seam_cost_layout(
                layouts,
                cost_right,
                cost_down,
                grid=parent.GRID,
            )
            portfolio_layout = _strict_layout(selection.layout)
            polished = polish_unprotected_taska_tail(
                portfolio_layout,
                cost_right,
                cost_down,
                edges,
                grid=parent.GRID,
                max_swaps=96,
                minimum_gain=1e-9,
            )
            portfolio_tail96 = _strict_layout(polished.layout)

            arrays[f"{prefix}__edge_minimum_margin"] = margins.astype(np.float32)
            arrays[f"{prefix}__edge_vote_count"] = votes.astype(np.uint8)
            arrays[f"{prefix}__edge_features"] = edge_features.astype(np.float32)
            arrays[f"{prefix}__logistic_logits"] = logistic_logits.astype(np.float32)
            arrays[f"{prefix}__nonlinear_logits"] = nonlinear_logits.astype(np.float32)
            arrays[f"{prefix}__focal_logits"] = focal_scores.logits
            arrays[f"{prefix}__focal_features"] = focal_scores.features
            for arm, layout in {
                **layouts,
                "portfolio": portfolio_layout,
                "portfolio_tail96": portfolio_tail96,
            }.items():
                arrays[f"{prefix}__{arm}_layout"] = layout

            rows.append(
                {
                    "prefix": prefix,
                    "case_id": dirty.case_id,
                    "source_filename": source,
                    "draw_index": draw,
                    "dirty_sha256": dirty_sha,
                    "candidate_edge_count": len(edges),
                    "matcher_matrix_replay_max_abs_delta": matrix_deltas,
                    "portfolio_choice": selection.choice,
                    "portfolio_total_costs": dict(selection.total_costs),
                    "protected_tail96_diagnostics": asdict(polished.diagnostics),
                    "solver_diagnostics": {
                        "logistic": logistic_solved.diagnostics.as_dict(),
                        "focal": focal_solved.diagnostics.as_dict(),
                        "nonlinear": nonlinear_solved.diagnostics.as_dict(),
                    },
                }
            )
            print(
                json.dumps(
                    {
                        "event": "fresh32_leader_target_free_case_ready",
                        "case": index + 1,
                        "case_count": len(specs),
                        "source_filename": source,
                        "draw_index": draw,
                        "portfolio_choice": selection.choice,
                        "tail96_swaps": polished.diagnostics.accepted_swap_count,
                        "strict_layouts": len(SCORED_ARMS),
                    }
                ),
                flush=True,
            )

    _write_npz_exclusive(frozen_path, arrays)
    _write_json_exclusive(
        metadata_path,
        {
            "schema": FROZEN_SCHEMA,
            "contains_exact_references_or_labels": False,
            "contains_target_ids_or_source_grid_coordinates": False,
            "parent_panel_targets_previously_opened": True,
            "created_before_reference_reconstruction_in_this_process": True,
            "candidate_membership_unchanged": True,
            "focal_mode": FOCAL_MODE,
            "focal_top_k": TASKA_FOCAL_FEATURE_TOP_K[FOCAL_MODE],
            "portfolio_arm_order": list(PORTFOLIO_ARMS),
            "portfolio_rule": "minimum original TASKA cost over all 1104 board bonds",
            "portfolio_tail_max_swaps": 96,
            "contains_all_strict_original_tile_layouts": True,
            "rows": rows,
        },
    )
    artifacts_to_freeze = {
        "signed_config": _record(parent.DEFAULT_CONFIG),
        "signed_config_sidecar": _record(Path(f"{parent.DEFAULT_CONFIG}.sha256")),
        "parent_archive": _record(PARENT_ARCHIVE),
        "parent_metadata": _record(PARENT_METADATA),
        "focal_checkpoint": _record(checkpoint),
        "logistic_calibrator": _record(logistic_path),
        "nonlinear_calibrator": _record(nonlinear_path),
        "frozen_candidate_archive": _record(frozen_path),
        "frozen_candidate_metadata": _record(metadata_path),
        **{name: _record(path) for name, path in _runtime_sources().items()},
    }
    _write_json_exclusive(
        freeze_path,
        {
            "schema": FREEZE_SCHEMA,
            "created_before_reference_reconstruction_in_this_process": True,
            "parent_panel_targets_previously_opened": True,
            "formal_fresh_promotion_claimed": False,
            "contains_evaluation_references_or_labels": False,
            "device": str(device),
            "artifacts": artifacts_to_freeze,
        },
    )
    return frozen_path, metadata_path, freeze_path, perf_counter() - started


def _validate_freeze(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("created_before_reference_reconstruction_in_this_process") is not True:
        raise RuntimeError("pre-score freeze timing contract changed")
    if payload.get("contains_evaluation_references_or_labels") is not False:
        raise RuntimeError("pre-score freeze unexpectedly contains labels")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise RuntimeError("pre-score artifact roster is missing")
    for name, record in artifacts.items():
        if not isinstance(record, Mapping):
            raise RuntimeError(f"malformed frozen artifact record: {name}")
        raw_path, expected = record.get("path"), record.get("sha256")
        if not isinstance(raw_path, str) or not isinstance(expected, str):
            raise RuntimeError(f"malformed frozen artifact record: {name}")
        artifact = Path(raw_path)
        if not artifact.is_absolute():
            artifact = PROJECT_ROOT / artifact
        if sha256_file(artifact.resolve()) != expected:
            raise RuntimeError(f"pre-score artifact changed: {name}")
    return payload


def _layout_metrics(layout: np.ndarray, exact: np.ndarray) -> dict[str, Any]:
    result = evaluate_layout(layout, exact, reference_is_exact=True)
    if result.adjacency_total != PAIR_DENOMINATOR:
        raise RuntimeError("adjacency denominator changed")
    return {
        "satisfied_adjacent_pairs": int(result.adjacency_correct),
        "adjacency_recall": float(result.adjacency),
        "exact_tiles": int(result.correct_tile_count),
        "strict_permutation": True,
    }


def _clustered_ci(
    values: Sequence[float],
    sources: Sequence[str],
    *,
    seed: int,
) -> dict[str, Any]:
    if len(values) != len(sources) or not values:
        raise ValueError("values and sources must be aligned and non-empty")
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, source in zip(values, sources, strict=True):
        if not math.isfinite(float(value)):
            raise ValueError("clustered values must be finite")
        grouped[source].append(float(value))
    if any(len(group) != 2 for group in grouped.values()):
        raise ValueError("each fresh32 source must contain exactly two draws")
    source_means = np.asarray([np.mean(grouped[name]) for name in sorted(grouped)])
    generator = np.random.default_rng(seed)
    distribution = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_RESAMPLES, 2048):
        stop = min(start + 2048, BOOTSTRAP_RESAMPLES)
        indices = generator.integers(0, len(source_means), size=(stop - start, len(source_means)))
        distribution[start:stop] = source_means[indices].mean(axis=1)
    return {
        "mean": float(np.mean(values)),
        "source_cluster_mean": float(source_means.mean()),
        "ci95_lower": float(np.quantile(distribution, 0.025)),
        "ci95_upper": float(np.quantile(distribution, 0.975)),
        "source_count": len(source_means),
        "case_count": len(values),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": seed,
    }


def _source_wins_ties_losses(values: Sequence[float], sources: Sequence[str]) -> dict[str, int]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, source in zip(values, sources, strict=True):
        grouped[source].append(float(value))
    means = [float(np.mean(grouped[name])) for name in sorted(grouped)]
    return {
        "wins": sum(value > 0 for value in means),
        "ties": sum(value == 0 for value in means),
        "losses": sum(value < 0 for value in means),
    }


def _summarize(rows: Sequence[Mapping[str, Any]], *, full_panel: bool) -> dict[str, Any]:
    metrics = ("satisfied_adjacent_pairs", "adjacency_recall", "exact_tiles")
    summary: dict[str, Any] = {
        "pair_denominator": PAIR_DENOMINATOR,
        "arms": {
            arm: {
                metric: float(np.mean([float(row[arm][metric]) for row in rows]))
                for metric in metrics
            }
            for arm in SCORED_ARMS
        },
        "portfolio_choice_counts": dict(Counter(str(row["portfolio_choice"]) for row in rows)),
        "comparisons_to_raw": {},
    }
    sources = [str(row["source_filename"]) for row in rows]
    for arm_index, arm in enumerate(PRIMARY_ARMS):
        comparison: dict[str, Any] = {}
        for metric_index, metric in enumerate(metrics):
            deltas = [float(row[arm][metric]) - float(row["raw"][metric]) for row in rows]
            if full_panel:
                result = _clustered_ci(
                    deltas,
                    sources,
                    seed=BOOTSTRAP_SEED + 100 * arm_index + metric_index,
                )
                result["source_wins_ties_losses"] = _source_wins_ties_losses(deltas, sources)
            else:
                result = {
                    "mean": float(np.mean(deltas)),
                    "ci95_lower": None,
                    "ci95_upper": None,
                    "smoke_only": True,
                }
            comparison[metric] = result
        summary["comparisons_to_raw"][f"{arm}_minus_raw"] = comparison
    return summary


def _score_after_freeze(
    *,
    specs: Sequence[tuple[Mapping[str, Any], str, int]],
    parent_rows: Sequence[Mapping[str, Any]],
    targets: Path,
    frozen_path: Path,
    metadata_path: Path,
    freeze_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _validate_freeze(freeze_path)
    candidate_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    candidate_rows = candidate_metadata.get("rows")
    if not isinstance(candidate_rows, list) or len(candidate_rows) != len(specs):
        raise RuntimeError("frozen candidate row roster changed")
    cache = parent.CleanTileCache(targets.resolve())
    scored_rows: list[dict[str, Any]] = []
    with np.load(frozen_path, allow_pickle=False) as archive:
        for spec, parent_row, candidate_row in zip(specs, parent_rows, candidate_rows, strict=True):
            record, source, draw = spec
            identity = ("prefix", "case_id", "source_filename", "draw_index", "dirty_sha256")
            if any(parent_row[field] != candidate_row[field] for field in identity):
                raise RuntimeError("parent and candidate frozen row identities differ")
            dirty, reference = make_exact_synthetic_case(
                cache.load(record),
                source_filename=source,
                draw_index=draw,
                seed=parent.SYNTHETIC_SEED,
            )
            if (
                dirty.case_id != parent_row["case_id"]
                or reference.case_id != dirty.case_id
                or parent._dirty_sha256(dirty.tiles) != parent_row["dirty_sha256"]
            ):
                raise RuntimeError("scoring recreated a different synthetic case")
            exact = _strict_layout(reference.tile_at_position)
            prefix = str(parent_row["prefix"])
            row: dict[str, Any] = {
                "prefix": prefix,
                "case_id": dirty.case_id,
                "source_filename": source,
                "draw_index": draw,
                "portfolio_choice": str(candidate_row["portfolio_choice"]),
            }
            for arm in SCORED_ARMS:
                layout = _strict_layout(archive[f"{prefix}__{arm}_layout"])
                row[arm] = _layout_metrics(layout, exact)
            scored_rows.append(row)
    full_panel = len(scored_rows) == CASE_COUNT
    return scored_rows, _summarize(scored_rows, full_panel=full_panel)


def run(args: argparse.Namespace) -> None:
    _require_hash(args.config, CONFIG_SHA256, name="signed fresh32 config")
    _require_hash(PARENT_ARCHIVE, PARENT_ARCHIVE_SHA256, name="parent fresh32 archive")
    _require_hash(PARENT_METADATA, PARENT_METADATA_SHA256, name="parent fresh32 metadata")
    checkpoint = _require_hash(
        args.checkpoint, TASKA_FOCAL_VERIFIER_SHA256, name="focal checkpoint"
    )
    logistic = _require_hash(args.logistic, LOGISTIC_SHA256, name="logistic calibrator")
    nonlinear = _require_hash(args.nonlinear, NONLINEAR_SHA256, name="nonlinear calibrator")
    raw_solver = _require_hash(
        PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py",
        RAW_SOLVER_SHA256,
        name="frozen raw solver",
    )
    artifacts, specs, parent_rows = _load_parent_context(
        args.config, smoke_one=bool(args.smoke_one)
    )
    device = parent._select_device(
        args.device,
        allow_nondeterministic_mps=bool(args.allow_nondeterministic_mps),
    )
    started = perf_counter()
    frozen, metadata, freeze, inference_seconds = _freeze_target_free(
        specs=specs,
        parent_rows=parent_rows,
        artifacts=artifacts,
        targets=args.targets,
        checkpoint=checkpoint,
        logistic_path=logistic,
        nonlinear_path=nonlinear,
        output_dir=args.output_dir.resolve(),
        device=device,
    )
    print(
        json.dumps(
            {
                "event": "fresh32_leader_logits_layouts_and_portfolio_frozen",
                "case_count": len(specs),
                "frozen_archive_sha256": sha256_file(frozen),
                "frozen_metadata_sha256": sha256_file(metadata),
                "pre_score_freeze_sha256": sha256_file(freeze),
                "reference_reconstructed_yet": False,
            }
        ),
        flush=True,
    )
    rows, metrics = _score_after_freeze(
        specs=specs,
        parent_rows=parent_rows,
        targets=args.targets,
        frozen_path=frozen,
        metadata_path=metadata,
        freeze_path=freeze,
    )
    full_panel = len(rows) == CASE_COUNT
    strict = all(row[arm]["strict_permutation"] for row in rows for arm in SCORED_ARMS)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "smoke-only" if args.smoke_one else "confirmation-complete",
        "panel": {
            "current_iteration_source_disjoint_at_parent_creation": True,
            "parent_panel_targets_previously_opened": True,
            "historical_model_selection_exposed": True,
            "formal_fresh_promotion_claimed": False,
            "evaluated_case_count": len(rows),
            "full_registered_panel": full_panel,
        },
        "candidate": {
            "no_parameter_tuning_or_sweep": True,
            "focal_mode": FOCAL_MODE,
            "focal_top_k": TASKA_FOCAL_FEATURE_TOP_K[FOCAL_MODE],
            "portfolio_arm_order": list(PORTFOLIO_ARMS),
            "portfolio_selector": "minimum original TASKA all-bond seam cost",
            "protected_tail_max_swaps": 96,
            "candidate_membership_unchanged": True,
            "original_costs_retained_for_placement_and_fill": True,
            "target_ids_or_exact_references_used_during_candidate_inference": False,
            "solver": asdict(parent.SOLVER_CONFIG),
        },
        "artifacts": {
            "focal_checkpoint": _record(checkpoint),
            "logistic_calibrator": _record(logistic),
            "nonlinear_calibrator": _record(nonlinear),
            "frozen_raw_solver": _record(raw_solver),
            "parent_archive": _record(PARENT_ARCHIVE),
            "parent_metadata": _record(PARENT_METADATA),
        },
        "frozen_eval": {
            "archive": _record(frozen),
            "metadata": _record(metadata),
            "pre_score_freeze": _record(freeze),
            "all_logits_and_layouts_frozen_before_reference_reconstruction": True,
            "contains_exact_references_or_labels": False,
        },
        "metrics": metrics,
        "measurement": {
            "all_six_layouts_strict": strict,
            "valid": full_panel and strict,
        },
        "rows": rows,
        "runtime_seconds": {
            "target_free_inference_and_solver": inference_seconds,
            "total": perf_counter() - started,
        },
        "legality": {
            "organizer_train_sources_only": True,
            "dirty_tiles_only_for_candidate_inference": True,
            "restored_pixels_emitted": False,
            "original_upright_tile_permutations_only": True,
            "competition_test_accessed": False,
        },
    }
    _write_json_exclusive(args.output_dir.resolve() / "report.json", report)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    run(parse_args())

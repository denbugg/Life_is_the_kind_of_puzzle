#!/usr/bin/env python3
"""Fit and gate the one fixed recovered-focal feature-stacking arm.

The experiment is deliberately single-shot: train96 -> disjoint local32 gate,
then the unchanged held32 only after a nonnegative local pair delta, and the
already frozen current-disjoint fresh32 only after a positive held pair delta.
All candidate layouts are written and SHA-frozen before exact references are
reconstructed in each evaluation stage.
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
from aiijc_puzzle.raw_tail_global_solver import RawTailEdge, solve_raw_tail_global
from aiijc_puzzle.taska_edge_calibrator import (
    TaskaEdgeCalibrator,
    extract_taska_edge_features,
    solve_prioritized_raw_tail_global,
)
from aiijc_puzzle.taska_focal_feature_stacker import (
    FOCAL_STACKER_FEATURE_NAMES,
    TaskaFocalFeatureStacker,
    fit_taska_focal_feature_stacker,
    stack_taska_focal_features,
)
from aiijc_puzzle.taska_focal_verifier import score_focal_edges
from aiijc_puzzle.taska_layout_portfolio import select_lowest_taska_seam_cost_layout
from aiijc_puzzle.taska_nonlinear_calibrator import TaskaNonlinearCalibrator
from aiijc_puzzle.taska_pair_pipeline import (
    ARM_NAMES,
    MATCHER_CONFIG,
    SOLVER_CONFIG,
    TaskaPairArtifactPaths,
    load_taska_pair_pipeline_resources,
)
from aiijc_puzzle.taska_protected_tail_polish import polish_unprotected_taska_tail
from aiijc_puzzle.taska_seam_matcher import match_taska_tiles

try:
    from scripts import run_taska_focal_current_finetune as finetune
except ModuleNotFoundError:  # Direct ``python scripts/run_*.py`` execution.
    import run_taska_focal_current_finetune as finetune

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/taska-focal-feature-stacker/train96-v1"
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
TRAIN_EDGE_ARCHIVE = (
    PROJECT_ROOT / "outputs/taska-edge-calibrator/train256-v1/training-features.npz"
)
TRAIN_FOCAL_ARCHIVE = (
    PROJECT_ROOT / "outputs/taska-focal-current-finetune/v1/training-harvest.npz"
)
LOCAL_PARENT_METADATA = (
    PROJECT_ROOT / "outputs/taska-focal-current-finetune/v1/local-gate-target-free.json"
)
HELD_PARENT_ARCHIVE = (
    PROJECT_ROOT / "outputs/taska-seam-replay/held300-diagnostic-mps-v1/"
    "frozen-target-free-eval.npz"
)
HELD_PARENT_METADATA = (
    PROJECT_ROOT / "outputs/taska-seam-replay/held300-diagnostic-mps-v1/"
    "frozen-target-free-eval.json"
)
HELD_FOCAL_ARCHIVE = (
    PROJECT_ROOT / "outputs/taska-focal-verifier/held300-train-exact-top5-cpu-v2/"
    "frozen-target-free-eval.npz"
)
HELD_FOCAL_METADATA = (
    PROJECT_ROOT / "outputs/taska-focal-verifier/held300-train-exact-top5-cpu-v2/"
    "frozen-target-free-eval.json"
)
FRESH_PARENT_ARCHIVE = (
    PROJECT_ROOT / "outputs/taska-protected-tail/fresh-held32-mps-v1/"
    "frozen-target-free-eval.npz"
)
FRESH_PARENT_METADATA = (
    PROJECT_ROOT / "outputs/taska-protected-tail/fresh-held32-mps-v1/"
    "frozen-target-free-eval.json"
)
FRESH_LEADER_ARCHIVE = (
    PROJECT_ROOT / "outputs/taska-fresh32-leader-confirmation/fresh-held32-mps-v1/"
    "frozen-target-free-eval.npz"
)
FRESH_LEADER_METADATA = (
    PROJECT_ROOT / "outputs/taska-fresh32-leader-confirmation/fresh-held32-mps-v1/"
    "frozen-target-free-eval.json"
)

EXPECTED_SHA256 = {
    TRAIN_EDGE_ARCHIVE: "2d1ef6267daab67d74971d625d2d446e7dfb8dc30a6165bd3459ab969e34f373",
    TRAIN_FOCAL_ARCHIVE: "5ee7b100eb213076fc1acbcace1c6d22e17bea99b88266c5c255cd94c85a17a1",
    LOCAL_PARENT_METADATA: "30d6dd6ebe0e4d492bf43fce8436494e93e165066cf08b27b74ce4470aaadd8e",
    HELD_PARENT_ARCHIVE: "0876d7acd6be7ebe863f831de568586c837da52cc603df4ff8d7c5a6b0d441df",
    HELD_PARENT_METADATA: "91710486233f45bda2b8aab019c508d1e6a0f75e282a6ce9a47aee93d9bf0a8d",
    HELD_FOCAL_ARCHIVE: "7d4ad494ab572d1ac3c94ab73a49b54e80b26baba489dfbd56f732a5c43394c5",
    HELD_FOCAL_METADATA: "301ba535f04b63ff8da48a0a83b5f207521d4b57f1bdbb61ceb58dbee57daff2",
    FRESH_PARENT_ARCHIVE: "d7b156ff1a8cdab702881242e48797b1a18f750a2d6a60f2a7d769dbfa1bffc1",
    FRESH_PARENT_METADATA: "1acb5d0000dd76e48fb6c079827fa2113bb56f541905fc97ced9656b8d7fe53f",
    FRESH_LEADER_ARCHIVE: "f3710cc3b00aaf2e75cb4127c280bc95eeeedf237f51a76ca234bac079c6f75f",
    FRESH_LEADER_METADATA: "311a1b3dc42bfb317a2c5cde1cee319de86ceba85622cb376fe4bfb83e2b53b1",
}
GRID = 24
COUNT = GRID * GRID
PAIR_DENOMINATOR = 2 * GRID * (GRID - 1)
TAIL_SWAPS = 96
TRAIN_BOARD_COUNT = 96
FOCAL_MODE = "train_exact_top5"
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 2_026_083_154
STAGE_ARMS = ("stacker", "four_arm_tail96", "five_arm_tail96")
REPORT_SCHEMA = "aiijc-taska-focal-feature-stacker-report-v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--smoke-one", action="store_true")
    return parser.parse_args(argv)


def _require_frozen_inputs() -> None:
    for path, expected in EXPECTED_SHA256.items():
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"frozen input SHA-256 mismatch: {path}")
    if sha256_file(finetune.DEFAULT_CONFIG) != finetune.CONFIG_SHA256:
        raise ValueError("signed fine-tune config changed")


def _record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        rendered = str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        rendered = str(resolved)
    return {"path": rendered, "sha256": sha256_file(resolved)}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {path}") from error


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {path}") from error


def _strict_layout(value: Any) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=np.int32)
    if result.shape != (COUNT,) or not np.array_equal(np.sort(result), np.arange(COUNT)):
        raise ValueError("layout is not a strict 576-tile permutation")
    return result


def _finite_matrix(archive: Any, key: str) -> np.ndarray:
    result = np.asarray(archive[key], dtype=np.float64)
    if result.shape != (COUNT, COUNT) or not np.isfinite(result).all():
        raise ValueError(f"{key} must be one finite 576x576 matrix")
    return np.ascontiguousarray(result)


def _edges_from_archive(archive: Any, prefix: str) -> tuple[RawTailEdge, ...]:
    source = np.asarray(archive[f"{prefix}__edge_source"], dtype=np.int64)
    target = np.asarray(archive[f"{prefix}__edge_target"], dtype=np.int64)
    axis = np.asarray(archive[f"{prefix}__edge_axis"], dtype=np.uint8)
    if not (source.ndim == target.ndim == axis.ndim == 1):
        raise ValueError("edge arrays must be vectors")
    if not (len(source) == len(target) == len(axis)) or not np.isin(axis, (0, 1)).all():
        raise ValueError("edge arrays are misaligned")
    return tuple(
        RawTailEdge(int(s), int(t), "right" if int(a) == 0 else "down")
        for s, t, a in zip(source, target, axis, strict=True)
    )


def _score_cached_training_patches(
    model: torch.nn.Module,
    patches: np.ndarray,
    features: np.ndarray,
    *,
    device: torch.device,
    chunk_size: int = 2048,
) -> np.ndarray:
    if patches.shape[0] != len(features):
        raise ValueError("cached focal patches and features are misaligned")
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(patches), chunk_size):
            stop = min(start + chunk_size, len(patches))
            outputs.append(
                model(
                    torch.from_numpy(patches[start:stop].astype(np.float32)).to(device),
                    torch.from_numpy(features[start:stop]).to(device),
                )
                .detach()
                .cpu()
                .numpy()
            )
    result = np.ascontiguousarray(np.concatenate(outputs), dtype=np.float32)
    if result.shape != (len(features),) or not np.isfinite(result).all():
        raise RuntimeError("recovered focal training logits are malformed")
    return result


def _fit_fixed_stacker(
    resources: Any,
    *,
    output_dir: Path,
) -> tuple[TaskaFocalFeatureStacker, Path, dict[str, Any]]:
    with (
        np.load(TRAIN_EDGE_ARCHIVE, allow_pickle=False) as edge,
        np.load(TRAIN_FOCAL_ARCHIVE, allow_pickle=False) as focal,
    ):
        edge_offsets = np.asarray(edge["offsets"], dtype=np.int64)
        focal_offsets = np.asarray(focal["offsets"], dtype=np.int64)
        if not np.array_equal(edge_offsets[: TRAIN_BOARD_COUNT + 1], focal_offsets):
            raise RuntimeError("train96 cached offsets differ")
        if not np.array_equal(
            np.asarray(edge["source_filenames"][:TRAIN_BOARD_COUNT]),
            np.asarray(focal["source_filenames"]),
        ):
            raise RuntimeError("train96 cached source roster differs")
        stop = int(edge_offsets[TRAIN_BOARD_COUNT])
        edge_features = np.asarray(edge["features"][:stop], dtype=np.float32)
        edge_labels = np.asarray(edge["labels"][:stop], dtype=np.uint8)
        focal_labels = np.asarray(focal["labels"], dtype=np.uint8)
        if not np.array_equal(edge_labels, focal_labels):
            raise RuntimeError("train96 labels differ between independent caches")
        focal_features = np.asarray(focal["features"], dtype=np.float32)
        patches = np.asarray(focal["patches_uint8"], dtype=np.uint8)
    started = perf_counter()
    focal_logits = _score_cached_training_patches(
        resources.focal_verifier,
        patches,
        focal_features,
        device=resources.device,
    )
    stacked = stack_taska_focal_features(edge_features, focal_logits, focal_features)
    stacker = fit_taska_focal_feature_stacker(stacked, edge_labels)
    artifact = output_dir / "stacker.npz"
    stacker.save_npz(artifact)
    reloaded = TaskaFocalFeatureStacker.load_npz(artifact)
    if not np.array_equal(
        reloaded.predict_logits(stacked[:1024]),
        stacker.predict_logits(stacked[:1024]),
    ):
        raise RuntimeError("persisted stacker changed its predictions")
    return stacker, artifact, {
        "single_fixed_arm": True,
        "board_count": TRAIN_BOARD_COUNT,
        "edge_count": len(edge_labels),
        "positive_count": int(edge_labels.sum()),
        "positive_fraction": float(edge_labels.mean()),
        "feature_count": len(FOCAL_STACKER_FEATURE_NAMES),
        "feature_names": list(FOCAL_STACKER_FEATURE_NAMES),
        "estimator": {
            "pipeline": "StandardScaler -> LogisticRegression",
            "C": 1.0,
            "max_iter": 1000,
            "random_state": 0,
            "class_weight": None,
            "feature_selection": False,
            "hyperparameter_sweep": False,
        },
        "cache_alignment": {
            "source_names_equal": True,
            "offsets_equal": True,
            "labels_equal": True,
        },
        "runtime_seconds": perf_counter() - started,
        "artifact": _record(artifact),
    }


def _edge_evidence(matched: Any) -> tuple[np.ndarray, np.ndarray]:
    records = {record.edge: record for record in matched.vote_records}
    if len(records) != len(matched.vote_records) or set(records) != set(matched.candidate_edges):
        raise ValueError("matcher vote records and candidate edges differ")
    margins = np.asarray(
        [records[edge].minimum_margin for edge in matched.candidate_edges], dtype=np.float64
    )
    votes = np.asarray(
        [records[edge].vote_count for edge in matched.candidate_edges], dtype=np.float64
    )
    return margins, votes


def _compose(
    *,
    right: np.ndarray,
    down: np.ndarray,
    edges: tuple[RawTailEdge, ...],
    edge_features: np.ndarray,
    focal_logits: np.ndarray,
    focal_features: np.ndarray,
    raw_layout: np.ndarray,
    focal_layout: np.ndarray,
    logistic: TaskaEdgeCalibrator,
    nonlinear: TaskaNonlinearCalibrator,
    stacker: TaskaFocalFeatureStacker,
    known_logistic_layout: np.ndarray | None = None,
    known_nonlinear_layout: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if known_logistic_layout is None:
        logistic_layout = _strict_layout(
            solve_prioritized_raw_tail_global(
                right,
                down,
                edges,
                logistic.predict_priorities(edge_features),
                grid=GRID,
                config=SOLVER_CONFIG,
            ).layout
        )
    else:
        logistic_layout = _strict_layout(known_logistic_layout)
    if known_nonlinear_layout is None:
        nonlinear_layout = _strict_layout(
            solve_prioritized_raw_tail_global(
                right,
                down,
                edges,
                nonlinear.predict_priorities(edge_features),
                grid=GRID,
                config=SOLVER_CONFIG,
            ).layout
        )
    else:
        nonlinear_layout = _strict_layout(known_nonlinear_layout)
    stack_features = stack_taska_focal_features(
        edge_features,
        focal_logits,
        focal_features,
    )
    stack_logits = stacker.predict_logits(stack_features)
    stack_result = solve_prioritized_raw_tail_global(
        right,
        down,
        edges,
        stacker.predict_priorities(stack_features),
        grid=GRID,
        config=SOLVER_CONFIG,
    )
    stack_layout = _strict_layout(stack_result.layout)
    four = {
        "raw": _strict_layout(raw_layout),
        "logistic": logistic_layout,
        "focal_top5": _strict_layout(focal_layout),
        "nonlinear": nonlinear_layout,
    }
    if tuple(four) != ARM_NAMES:
        raise RuntimeError("four-arm order differs from production pipeline")
    four_selection = select_lowest_taska_seam_cost_layout(four, right, down, grid=GRID)
    four_tail = polish_unprotected_taska_tail(
        four_selection.layout,
        right,
        down,
        edges,
        grid=GRID,
        max_swaps=TAIL_SWAPS,
    )
    five = {**four, "stacker": stack_layout}
    five_selection = select_lowest_taska_seam_cost_layout(five, right, down, grid=GRID)
    five_tail = polish_unprotected_taska_tail(
        five_selection.layout,
        right,
        down,
        edges,
        grid=GRID,
        max_swaps=TAIL_SWAPS,
    )
    layouts = {
        **four,
        "stacker": stack_layout,
        "four_arm_tail96": _strict_layout(four_tail.layout),
        "five_arm_tail96": _strict_layout(five_tail.layout),
    }
    return layouts, {
        "stacker_logits": np.asarray(stack_logits, dtype=np.float32),
        "four_arm_choice": four_selection.choice,
        "five_arm_choice": five_selection.choice,
        "four_arm_costs": dict(four_selection.total_costs),
        "five_arm_costs": dict(five_selection.total_costs),
        "four_arm_tail": asdict(four_tail.diagnostics),
        "five_arm_tail": asdict(five_tail.diagnostics),
        "stacker_solver": stack_result.diagnostics.as_dict(),
    }


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


def _cluster_ci(values: Sequence[float], sources: Sequence[str], *, seed: int) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, source in zip(values, sources, strict=True):
        if not math.isfinite(float(value)):
            raise ValueError("bootstrap values must be finite")
        grouped[source].append(float(value))
    cluster_means = np.asarray([np.mean(grouped[name]) for name in sorted(grouped)])
    generator = np.random.default_rng(seed)
    distribution = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_RESAMPLES, 2048):
        stop = min(start + 2048, BOOTSTRAP_RESAMPLES)
        indices = generator.integers(
            0, len(cluster_means), size=(stop - start, len(cluster_means))
        )
        distribution[start:stop] = cluster_means[indices].mean(axis=1)
    return {
        "mean": float(np.mean(values)),
        "source_cluster_mean": float(cluster_means.mean()),
        "ci95_lower": float(np.quantile(distribution, 0.025)),
        "ci95_upper": float(np.quantile(distribution, 0.975)),
        "source_count": len(cluster_means),
        "case_count": len(values),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": seed,
    }


def _summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = ("satisfied_adjacent_pairs", "adjacency_recall", "exact_tiles")
    summary: dict[str, Any] = {
        "pair_denominator": PAIR_DENOMINATOR,
        "arms": {
            arm: {
                metric: float(np.mean([row["metrics"][arm][metric] for row in rows]))
                for metric in metrics
            }
            for arm in STAGE_ARMS
        },
        "four_arm_choice_counts": dict(Counter(row["four_arm_choice"] for row in rows)),
        "five_arm_choice_counts": dict(Counter(row["five_arm_choice"] for row in rows)),
    }
    sources = [str(row["source_filename"]) for row in rows]
    deltas: dict[str, Any] = {}
    for index, metric in enumerate(metrics):
        values = [
            float(row["metrics"]["five_arm_tail96"][metric])
            - float(row["metrics"]["four_arm_tail96"][metric])
            for row in rows
        ]
        delta = _cluster_ci(values, sources, seed=BOOTSTRAP_SEED + index)
        delta["case_wins_ties_losses"] = {
            "wins": sum(value > 0 for value in values),
            "ties": sum(value == 0 for value in values),
            "losses": sum(value < 0 for value in values),
        }
        deltas[metric] = delta
    summary["five_minus_four"] = deltas
    return summary


def _freeze_stage(
    *,
    name: str,
    output_dir: Path,
    arrays: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
    stacker_path: Path,
    parent_artifacts: Mapping[str, Path],
) -> tuple[Path, Path, Path]:
    stage_dir = output_dir / name
    stage_dir.mkdir(parents=True, exist_ok=False)
    archive = stage_dir / "frozen-target-free-eval.npz"
    metadata = stage_dir / "frozen-target-free-eval.json"
    freeze = stage_dir / "pre-score-freeze.json"
    _write_npz(archive, arrays)
    _write_json(
        metadata,
        {
            "schema": "aiijc-taska-focal-feature-stacker-target-free-v1",
            "stage": name,
            "contains_exact_references_or_labels": False,
            "contains_target_ids_or_source_grid_coordinates": False,
            "candidate_membership_unchanged": True,
            "all_layouts_strict_original_tile_permutations": True,
            "rows": list(rows),
        },
    )
    sources = {
        "stacker_artifact": stacker_path,
        "frozen_archive": archive,
        "frozen_metadata": metadata,
        "runner": Path(__file__).resolve(),
        "stacker_module": (
            PROJECT_ROOT / "src/aiijc_puzzle/taska_focal_feature_stacker.py"
        ),
        "frozen_raw_solver": PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py",
        **parent_artifacts,
    }
    _write_json(
        freeze,
        {
            "schema": "aiijc-taska-focal-feature-stacker-pre-score-freeze-v1",
            "stage": name,
            "created_before_exact_reference_reconstruction": True,
            "contains_evaluation_references_or_labels": False,
            "artifacts": {key: _record(path) for key, path in sources.items()},
        },
    )
    return archive, metadata, freeze


def _validate_freeze(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("created_before_exact_reference_reconstruction") is not True:
        raise RuntimeError("pre-score freeze timing contract differs")
    if payload.get("contains_evaluation_references_or_labels") is not False:
        raise RuntimeError("pre-score freeze contains evaluation labels")
    for name, record in payload["artifacts"].items():
        artifact = Path(record["path"])
        if not artifact.is_absolute():
            artifact = PROJECT_ROOT / artifact
        if sha256_file(artifact) != record["sha256"]:
            raise RuntimeError(f"frozen artifact changed before scoring: {name}")


def _score_stage(
    *,
    archive: Path,
    metadata: Path,
    freeze: Path,
    lookup: Mapping[str, Mapping[str, Any]],
    cache: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _validate_freeze(freeze)
    frozen_rows = json.loads(metadata.read_text(encoding="utf-8"))["rows"]
    scored: list[dict[str, Any]] = []
    with np.load(archive, allow_pickle=False) as candidate:
        for row in frozen_rows:
            prefix = str(row["prefix"])
            source = str(row["source_filename"])
            draw = int(row["draw_index"])
            dirty = finetune._dirty_case(cache, lookup[source], source, draw)
            if finetune._dirty_sha256(dirty.dirty_tiles) != row["dirty_sha256"]:
                raise RuntimeError("scoring recreated different dirty bytes")
            reference = finetune._reference(
                cache, lookup[source], source, draw, dirty.dirty_tiles
            )
            metrics = {
                arm: _layout_metrics(
                    _strict_layout(candidate[f"{prefix}__{arm}_layout"]), reference
                )
                for arm in STAGE_ARMS
            }
            scored.append(
                {
                    "prefix": prefix,
                    "source_filename": source,
                    "draw_index": draw,
                    "metrics": metrics,
                    "four_arm_choice": row["four_arm_choice"],
                    "five_arm_choice": row["five_arm_choice"],
                }
            )
    return scored, _summarize(scored)


def _run_local(
    *,
    output_dir: Path,
    stacker: TaskaFocalFeatureStacker,
    stacker_path: Path,
    resources: Any,
    logistic: TaskaEdgeCalibrator,
    nonlinear: TaskaNonlinearCalibrator,
    lookup: Mapping[str, Mapping[str, Any]],
    cache: Any,
    smoke_one: bool,
) -> dict[str, Any]:
    parent = json.loads(LOCAL_PARENT_METADATA.read_text(encoding="utf-8"))
    parent_rows = parent["rows"][: 1 if smoke_one else None]
    arrays: dict[str, np.ndarray] = {}
    frozen_rows: list[dict[str, Any]] = []
    started = perf_counter()
    for index, parent_row in enumerate(parent_rows):
        source = str(parent_row["source_filename"])
        draw = int(parent_row["draw_index"])
        dirty = finetune._dirty_case(cache, lookup[source], source, draw)
        dirty_sha = finetune._dirty_sha256(dirty.dirty_tiles)
        if dirty_sha != parent_row["dirty_sha256"]:
            raise RuntimeError("local signed recipe recreated different dirty bytes")
        matched = match_taska_tiles(
            dirty.dirty_tiles,
            resources.matchers,
            config=MATCHER_CONFIG,
            device=resources.device,
            require_verified=True,
        )
        margins, votes = _edge_evidence(matched)
        edge_features = extract_taska_edge_features(
            matched.cost_right,
            matched.cost_down,
            matched.right_log,
            matched.down_log,
            matched.candidate_edges,
            margins,
            votes,
            grid=GRID,
        ).values
        focal = score_focal_edges(
            resources.focal_verifier,
            dirty.dirty_tiles,
            matched.cost_right,
            matched.cost_down,
            matched.candidate_edges,
            mode=FOCAL_MODE,
            device=resources.device,
        )
        raw = solve_raw_tail_global(
            matched.cost_right,
            matched.cost_down,
            matched.candidate_edges,
            grid=GRID,
            config=SOLVER_CONFIG,
        )
        focal_layout = solve_prioritized_raw_tail_global(
            matched.cost_right,
            matched.cost_down,
            matched.candidate_edges,
            focal.logits,
            grid=GRID,
            config=SOLVER_CONFIG,
        ).layout
        layouts, diagnostics = _compose(
            right=np.asarray(matched.cost_right),
            down=np.asarray(matched.cost_down),
            edges=tuple(matched.candidate_edges),
            edge_features=edge_features,
            focal_logits=focal.logits,
            focal_features=focal.features,
            raw_layout=raw.layout,
            focal_layout=focal_layout,
            logistic=logistic,
            nonlinear=nonlinear,
            stacker=stacker,
        )
        prefix = f"case_{index:04d}"
        for arm, layout in layouts.items():
            arrays[f"{prefix}__{arm}_layout"] = layout
        arrays[f"{prefix}__cost_right"] = np.asarray(
            matched.cost_right, dtype=np.float32
        )
        arrays[f"{prefix}__cost_down"] = np.asarray(
            matched.cost_down, dtype=np.float32
        )
        arrays[f"{prefix}__edge_source"] = np.asarray(
            [edge.source for edge in matched.candidate_edges], dtype=np.int32
        )
        arrays[f"{prefix}__edge_target"] = np.asarray(
            [edge.target for edge in matched.candidate_edges], dtype=np.int32
        )
        arrays[f"{prefix}__edge_axis"] = np.asarray(
            [edge.axis == "down" for edge in matched.candidate_edges], dtype=np.uint8
        )
        arrays[f"{prefix}__edge_features"] = np.asarray(
            edge_features, dtype=np.float32
        )
        arrays[f"{prefix}__focal_logits"] = focal.logits
        arrays[f"{prefix}__focal_features"] = focal.features
        arrays[f"{prefix}__stacker_logits"] = diagnostics.pop("stacker_logits")
        frozen_rows.append(
            {
                "prefix": prefix,
                "source_filename": source,
                "draw_index": draw,
                "dirty_sha256": dirty_sha,
                "candidate_edge_count": len(matched.candidate_edges),
                **diagnostics,
            }
        )
        print(
            json.dumps(
                {
                    "event": "focal_stacker_local_target_free",
                    "case": index + 1,
                    "case_count": len(parent_rows),
                    "source": source,
                }
            ),
            flush=True,
        )
    archive, metadata, freeze = _freeze_stage(
        name="local32",
        output_dir=output_dir,
        arrays=arrays,
        rows=frozen_rows,
        stacker_path=stacker_path,
        parent_artifacts={"local_parent_metadata": LOCAL_PARENT_METADATA},
    )
    rows, summary = _score_stage(
        archive=archive,
        metadata=metadata,
        freeze=freeze,
        lookup=lookup,
        cache=cache,
    )
    return {
        "status": "smoke-only" if smoke_one else "complete",
        "gate_passed": (
            False
            if smoke_one
            else summary["five_minus_four"]["satisfied_adjacent_pairs"]["mean"] >= 0
        ),
        "summary": summary,
        "rows": rows,
        "runtime_seconds": perf_counter() - started,
        "artifacts": {
            "archive": _record(archive),
            "metadata": _record(metadata),
            "pre_score_freeze": _record(freeze),
        },
    }


def _run_cached_panel(
    *,
    stage: str,
    output_dir: Path,
    stacker: TaskaFocalFeatureStacker,
    stacker_path: Path,
    logistic: TaskaEdgeCalibrator,
    nonlinear: TaskaNonlinearCalibrator,
    lookup: Mapping[str, Mapping[str, Any]],
    cache: Any,
    parent_archive_path: Path,
    parent_metadata_path: Path,
    focal_archive_path: Path,
    focal_metadata_path: Path,
    leader_archive_path: Path | None = None,
) -> dict[str, Any]:
    parent_rows = json.loads(parent_metadata_path.read_text(encoding="utf-8"))["rows"]
    focal_rows = json.loads(focal_metadata_path.read_text(encoding="utf-8"))["rows"]
    if len(parent_rows) != 32 or len(focal_rows) != 32:
        raise ValueError(f"{stage} panel must contain exactly 32 cases")
    arrays: dict[str, np.ndarray] = {}
    frozen_rows: list[dict[str, Any]] = []
    started = perf_counter()
    context = (
        np.load(parent_archive_path, allow_pickle=False),
        np.load(focal_archive_path, allow_pickle=False),
    )
    parent_archive, focal_archive = context
    try:
        for index, (parent_row, focal_row) in enumerate(
            zip(parent_rows, focal_rows, strict=True)
        ):
            identity = ("prefix", "source_filename", "draw_index", "dirty_sha256")
            if any(parent_row[key] != focal_row[key] for key in identity):
                raise RuntimeError(f"{stage} parent and focal row identities differ")
            prefix = str(parent_row["prefix"])
            edges = _edges_from_archive(parent_archive, prefix)
            right = _finite_matrix(parent_archive, f"{prefix}__cost_right")
            down = _finite_matrix(parent_archive, f"{prefix}__cost_down")
            focal_logits = np.asarray(
                focal_archive[f"{prefix}__focal_logits"], dtype=np.float32
            )
            focal_features = np.asarray(
                focal_archive[f"{prefix}__focal_features"], dtype=np.float32
            )
            if leader_archive_path is None:
                edge_features = extract_taska_edge_features(
                    right,
                    down,
                    _finite_matrix(parent_archive, f"{prefix}__right_log"),
                    _finite_matrix(parent_archive, f"{prefix}__down_log"),
                    edges,
                    parent_archive[f"{prefix}__edge_weight"],
                    parent_archive[f"{prefix}__edge_vote_count"],
                    grid=GRID,
                ).values
                raw_layout = _strict_layout(parent_archive[f"{prefix}__taska_layout"])
                focal_layout = _strict_layout(focal_archive[f"{prefix}__focal_layout"])
                known_logistic = None
                known_nonlinear = None
            else:
                with np.load(leader_archive_path, allow_pickle=False) as leader:
                    edge_features = np.asarray(
                        leader[f"{prefix}__edge_features"], dtype=np.float64
                    )
                    raw_layout = _strict_layout(leader[f"{prefix}__raw_layout"])
                    focal_layout = _strict_layout(leader[f"{prefix}__focal_layout"])
                    known_logistic = _strict_layout(
                        leader[f"{prefix}__logistic_layout"]
                    )
                    known_nonlinear = _strict_layout(
                        leader[f"{prefix}__nonlinear_layout"]
                    )
            layouts, diagnostics = _compose(
                right=right,
                down=down,
                edges=edges,
                edge_features=edge_features,
                focal_logits=focal_logits,
                focal_features=focal_features,
                raw_layout=raw_layout,
                focal_layout=focal_layout,
                logistic=logistic,
                nonlinear=nonlinear,
                stacker=stacker,
                known_logistic_layout=known_logistic,
                known_nonlinear_layout=known_nonlinear,
            )
            for arm, layout in layouts.items():
                arrays[f"{prefix}__{arm}_layout"] = layout
            arrays[f"{prefix}__stacker_logits"] = diagnostics.pop("stacker_logits")
            frozen_rows.append(
                {
                    "prefix": prefix,
                    "case_id": parent_row.get("case_id"),
                    "source_filename": parent_row["source_filename"],
                    "draw_index": parent_row["draw_index"],
                    "dirty_sha256": parent_row["dirty_sha256"],
                    "candidate_edge_count": len(edges),
                    **diagnostics,
                }
            )
            print(
                json.dumps(
                    {
                        "event": f"focal_stacker_{stage}_target_free",
                        "case": index + 1,
                        "case_count": len(parent_rows),
                    }
                ),
                flush=True,
            )
    finally:
        parent_archive.close()
        focal_archive.close()
    parent_artifacts = {
        "parent_archive": parent_archive_path,
        "parent_metadata": parent_metadata_path,
        "focal_archive": focal_archive_path,
        "focal_metadata": focal_metadata_path,
    }
    if leader_archive_path is not None:
        parent_artifacts["four_arm_leader_archive"] = leader_archive_path
    archive, metadata, freeze = _freeze_stage(
        name=stage,
        output_dir=output_dir,
        arrays=arrays,
        rows=frozen_rows,
        stacker_path=stacker_path,
        parent_artifacts=parent_artifacts,
    )
    rows, summary = _score_stage(
        archive=archive,
        metadata=metadata,
        freeze=freeze,
        lookup=lookup,
        cache=cache,
    )
    return {
        "status": "complete",
        "summary": summary,
        "rows": rows,
        "runtime_seconds": perf_counter() - started,
        "artifacts": {
            "archive": _record(archive),
            "metadata": _record(metadata),
            "pre_score_freeze": _record(freeze),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    _require_frozen_inputs()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    if args.device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable")
        torch.use_deterministic_algorithms(False)
    else:
        torch.use_deterministic_algorithms(True)
    started = perf_counter()
    paths = TaskaPairArtifactPaths()
    resources = load_taska_pair_pipeline_resources(paths, device=args.device)
    stacker, stacker_path, training = _fit_fixed_stacker(resources, output_dir=output_dir)
    logistic = resources.logistic_calibrator
    nonlinear = resources.nonlinear_calibrator
    config, _, _ = finetune._load_config(finetune.DEFAULT_CONFIG)
    lookup = finetune._manifest_lookup(config)
    cache = finetune.CleanTileCache(args.targets.resolve(), maximum_boards=2)
    local = _run_local(
        output_dir=output_dir,
        stacker=stacker,
        stacker_path=stacker_path,
        resources=resources,
        logistic=logistic,
        nonlinear=nonlinear,
        lookup=lookup,
        cache=cache,
        smoke_one=bool(args.smoke_one),
    )
    held: dict[str, Any] = {"status": "skipped_by_local_gate"}
    fresh: dict[str, Any] = {"status": "skipped_by_local_or_held_gate"}
    if local["gate_passed"]:
        held = _run_cached_panel(
            stage="held32",
            output_dir=output_dir,
            stacker=stacker,
            stacker_path=stacker_path,
            logistic=logistic,
            nonlinear=nonlinear,
            lookup=lookup,
            cache=cache,
            parent_archive_path=HELD_PARENT_ARCHIVE,
            parent_metadata_path=HELD_PARENT_METADATA,
            focal_archive_path=HELD_FOCAL_ARCHIVE,
            focal_metadata_path=HELD_FOCAL_METADATA,
        )
        held_delta = held["summary"]["five_minus_four"]["satisfied_adjacent_pairs"][
            "mean"
        ]
        if held_delta > 0:
            fresh = _run_cached_panel(
                stage="fresh32",
                output_dir=output_dir,
                stacker=stacker,
                stacker_path=stacker_path,
                logistic=logistic,
                nonlinear=nonlinear,
                lookup=lookup,
                cache=cache,
                parent_archive_path=FRESH_PARENT_ARCHIVE,
                parent_metadata_path=FRESH_PARENT_METADATA,
                focal_archive_path=FRESH_LEADER_ARCHIVE,
                focal_metadata_path=FRESH_LEADER_METADATA,
                leader_archive_path=FRESH_LEADER_ARCHIVE,
            )
        else:
            fresh = {"status": "skipped_by_nonpositive_held_pair_delta"}
    report = {
        "schema": REPORT_SCHEMA,
        "protocol": {
            "single_fixed_learned_fusion_arm": True,
            "train_board_count": TRAIN_BOARD_COUNT,
            "local_gate": "five-arm-tail96 minus four-arm-tail96 pairs >= 0",
            "held_opened_only_after_local_gate": True,
            "fresh_opened_only_after_strictly_positive_held_pair_delta": True,
            "fresh_panel_previously_opened": True,
            "no_parameter_sweep": True,
        },
        "training": training,
        "local32": local,
        "held32": held,
        "fresh32": fresh,
        "runtime_seconds": perf_counter() - started,
        "legality": {
            "offline_train_labels_only": True,
            "target_free_features_only_at_inference": True,
            "candidate_membership_unchanged": True,
            "original_costs_retained_for_placement_and_fill": True,
            "all_outputs_strict_original_upright_tile_permutations": True,
            "competition_test_accessed": False,
            "restored_pixels_emitted": False,
        },
        "artifacts": {
            "stacker": _record(stacker_path),
            "recovered_focal_checkpoint": _record(paths.focal_verifier),
            "logistic_calibrator": _record(paths.logistic_calibrator),
            "nonlinear_calibrator": _record(paths.nonlinear_calibrator),
            "train_edge_cache": _record(TRAIN_EDGE_ARCHIVE),
            "train_focal_cache": _record(TRAIN_FOCAL_ARCHIVE),
            "frozen_raw_solver": _record(
                PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py"
            ),
        },
    }
    _write_json(output_dir / "report.json", report)
    print(json.dumps({key: report[key] for key in ("local32", "held32", "fresh32")}, indent=2))
    return report


if __name__ == "__main__":
    run(parse_args())

#!/usr/bin/env python3
"""Run the fixed extension128 context-aware TASKA incidence-GNN experiment.

The model is trained once on train256 indices 128:256, while indices 96:128
remain the untouched local32 gate.  Candidate layouts are SHA-frozen before
any evaluation references are reconstructed.  A failed gate stops the run.
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

from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.taska_edge_calibrator import (
    extract_taska_edge_features,
    solve_prioritized_raw_tail_global,
)
from aiijc_puzzle.taska_focal_feature_stacker import stack_taska_focal_features
from aiijc_puzzle.taska_incidence_gnn import (
    INCIDENCE_GNN_TRAINING,
    TaskaIncidenceGNNBundle,
    load_taska_incidence_gnn_bundle,
    save_taska_incidence_gnn_bundle,
    train_taska_incidence_gnn,
)
from aiijc_puzzle.taska_layout_portfolio import select_lowest_taska_seam_cost_layout
from aiijc_puzzle.taska_pair_pipeline import ARM_NAMES, SOLVER_CONFIG
from aiijc_puzzle.taska_protected_tail_polish import polish_unprotected_taska_tail

try:
    from scripts import run_taska_focal_current_finetune as finetune
    from scripts import run_taska_focal_feature_stacker as baseline
except ModuleNotFoundError:  # Direct ``python scripts/run_*.py`` execution.
    import run_taska_focal_current_finetune as finetune
    import run_taska_focal_feature_stacker as baseline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/taska-incidence-gnn/extension128-v1"
TRAIN_CACHE = (
    PROJECT_ROOT
    / "outputs/taska-focal-feature-stacker/train224-v1/"
    "extension128-focal-harvest.npz"
)
TRAIN_METADATA = TRAIN_CACHE.with_suffix(".json")
TRAIN96_ROOT = PROJECT_ROOT / "outputs/taska-focal-feature-stacker/train96-v1"
LOCAL_ARCHIVE = TRAIN96_ROOT / "local32/frozen-target-free-eval.npz"
LOCAL_METADATA = TRAIN96_ROOT / "local32/frozen-target-free-eval.json"
HELD_LAYOUT_ARCHIVE = TRAIN96_ROOT / "held32/frozen-target-free-eval.npz"
HELD_LAYOUT_METADATA = TRAIN96_ROOT / "held32/frozen-target-free-eval.json"
FRESH_LAYOUT_ARCHIVE = (
    TRAIN96_ROOT / "fresh32-exact-override/frozen-target-free-eval.npz"
)
FRESH_LAYOUT_METADATA = (
    TRAIN96_ROOT / "fresh32-exact-override/frozen-target-free-eval.json"
)
EXPECTED_SHA256 = {
    TRAIN_CACHE: "bf0a6686e8112a841e9a8ea5e133dbed0152ceb346193c59b69dee0959efb87d",
    TRAIN_METADATA: "4133b4d733b21b714c38c0ab2819149e5cf8c61e4283ce0b7e4261558f7584c5",
    LOCAL_ARCHIVE: "e615659915d6bb17710403833b46b78d64916e4910afd7effa834a9f46d98e27",
    LOCAL_METADATA: "78a10b5f3581ef89a1dadae284906cd6f85ec7708250154b4ef4b819cd01a62c",
    HELD_LAYOUT_ARCHIVE: "784a459f6baaaa2d16fd5e1c269c0a50ff456b8c7071ba1cf6035e20be6808f1",
    HELD_LAYOUT_METADATA: "cbe6a774fb3fd3a6095e0124c5f484736ede963c0f6682d93ac49d67fe7c384a",
    FRESH_LAYOUT_ARCHIVE: "61d166fdd5ef275ae0e790951b7d07bb174d66eadbcd4c3b25869a0a587868d2",
    FRESH_LAYOUT_METADATA: "024be2cc842c2c4e7aec8df0a3d10d5f8ea185011f825a9bdb69580f7ee797fb",
    PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py": (
        "97859e1f4ff6ceadf56ebccf429f2944ae7a5726631731d9f776224a667eb486"
    ),
}
GRID = 24
COUNT = GRID * GRID
TAIL_SWAPS = 96
TRAIN_BOARD_COUNT = 128
PAIR_DENOMINATOR = 2 * GRID * (GRID - 1)
PANEL_ARMS = ("incidence_gnn", "four_arm_tail96", "five_arm_tail96")
REPORT_SCHEMA = "aiijc-taska-incidence-gnn-report-v1"
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 2_026_083_184


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=finetune.DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def _record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {
        "path": str(resolved.relative_to(PROJECT_ROOT)),
        "sha256": sha256_file(resolved),
    }


def _write_json(path: Path, payload: Any) -> None:
    baseline._write_json(path, payload)


def _require_inputs() -> None:
    baseline._require_frozen_inputs()
    for path, expected in EXPECTED_SHA256.items():
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"frozen incidence-GNN input changed: {path}")


def _fit_bundle(
    output_dir: Path,
) -> tuple[TaskaIncidenceGNNBundle, Path, dict[str, Any]]:
    started = perf_counter()
    with np.load(TRAIN_CACHE, allow_pickle=False) as archive:
        if str(archive["schema"].item()) != (
            "aiijc-taska-focal-train224-extension128-v1"
        ):
            raise ValueError("extension128 training schema changed")
        features = np.asarray(archive["stacked_features"], dtype=np.float64)
        focal_logits = np.asarray(archive["focal_logits"], dtype=np.float32)
        labels = np.asarray(archive["labels"], dtype=np.uint8)
        offsets = np.asarray(archive["offsets"], dtype=np.int64)
        source = np.asarray(archive["edge_source"], dtype=np.int64)
        target = np.asarray(archive["edge_target"], dtype=np.int64)
        axis = np.asarray(archive["edge_axis"], dtype=np.uint8)
        train_indices = np.asarray(archive["train256_indices"], dtype=np.int16)
        train_sources = tuple(str(value) for value in archive["source_filenames"])
        draws = np.asarray(archive["draw_indices"], dtype=np.uint8)
    if offsets.shape != (TRAIN_BOARD_COUNT + 1,) or len(train_sources) != TRAIN_BOARD_COUNT:
        raise ValueError("extension128 board count changed")
    if not np.array_equal(train_indices, np.arange(128, 256, dtype=np.int16)):
        raise ValueError("extension128 train256 index selection changed")
    if not np.array_equal(draws, np.zeros(TRAIN_BOARD_COUNT, dtype=np.uint8)):
        raise ValueError("extension128 draw selection changed")
    local_rows = json.loads(LOCAL_METADATA.read_text(encoding="utf-8"))["rows"]
    local_sources = {str(row["source_filename"]) for row in local_rows}
    if local_sources.intersection(train_sources):
        raise ValueError("extension128 training sources overlap local32")
    bundle, history = train_taska_incidence_gnn(
        features=features,
        focal_logits=focal_logits,
        labels=labels,
        offsets=offsets,
        source=source,
        target=target,
        axis=axis,
    )
    weights_path, standardizer_path, contract_path = save_taska_incidence_gnn_bundle(
        bundle, output_dir
    )
    contract_sha = sha256_file(contract_path)
    reloaded = load_taska_incidence_gnn_bundle(
        contract_path, expected_contract_sha256=contract_sha
    )
    first_stop = int(offsets[1])
    expected = bundle.predict_logits(
        features[:first_stop],
        focal_logits[:first_stop],
        source[:first_stop],
        target[:first_stop],
        axis[:first_stop],
    )
    actual = reloaded.predict_logits(
        features[:first_stop],
        focal_logits[:first_stop],
        source[:first_stop],
        target[:first_stop],
        axis[:first_stop],
    )
    if not np.array_equal(expected, actual):
        raise RuntimeError("persisted incidence GNN changed its predictions")
    return reloaded, contract_path, {
        "single_fixed_model": True,
        "training_cache": _record(TRAIN_CACHE),
        "training_metadata": _record(TRAIN_METADATA),
        "selected_train256_indices": "128:256",
        "excluded_local32_indices": "96:128",
        "source_disjoint_from_local32": True,
        "hyperparameter_or_epoch_selection": False,
        "training_contract": dict(INCIDENCE_GNN_TRAINING),
        "history": history,
        "artifacts": {
            "weights": _record(weights_path),
            "standardizer": _record(standardizer_path),
            "contract": _record(contract_path),
        },
        "runtime_seconds": perf_counter() - started,
    }


def _strict_layout(value: Any) -> np.ndarray:
    return baseline._strict_layout(value)


def _compose(
    *,
    right: np.ndarray,
    down: np.ndarray,
    edges: Any,
    stacked_features: np.ndarray,
    focal_logits: np.ndarray,
    four_layouts: Mapping[str, np.ndarray],
    bundle: TaskaIncidenceGNNBundle,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if tuple(four_layouts) != ARM_NAMES:
        raise RuntimeError("four-arm order differs from production TASKA")
    source = np.asarray([edge.source for edge in edges], dtype=np.int64)
    target = np.asarray([edge.target for edge in edges], dtype=np.int64)
    axis = np.asarray([edge.axis == "down" for edge in edges], dtype=np.uint8)
    logits = bundle.predict_logits(
        stacked_features,
        focal_logits,
        source,
        target,
        axis,
    )
    priorities = bundle.predict_priorities(
        stacked_features,
        focal_logits,
        source,
        target,
        axis,
    )
    gnn_result = solve_prioritized_raw_tail_global(
        right,
        down,
        edges,
        priorities,
        grid=GRID,
        config=SOLVER_CONFIG,
    )
    gnn_layout = _strict_layout(gnn_result.layout)
    four = {name: _strict_layout(layout) for name, layout in four_layouts.items()}
    four_selection = select_lowest_taska_seam_cost_layout(four, right, down, grid=GRID)
    four_tail = polish_unprotected_taska_tail(
        four_selection.layout,
        right,
        down,
        edges,
        grid=GRID,
        max_swaps=TAIL_SWAPS,
    )
    five = {**four, "incidence_gnn": gnn_layout}
    five_selection = select_lowest_taska_seam_cost_layout(five, right, down, grid=GRID)
    five_tail = polish_unprotected_taska_tail(
        five_selection.layout,
        right,
        down,
        edges,
        grid=GRID,
        max_swaps=TAIL_SWAPS,
    )
    return {
        "incidence_gnn": gnn_layout,
        "four_arm_tail96": _strict_layout(four_tail.layout),
        "five_arm_tail96": _strict_layout(five_tail.layout),
    }, {
        "incidence_logits": np.asarray(logits, dtype=np.float32),
        "four_arm_choice": four_selection.choice,
        "five_arm_choice": five_selection.choice,
        "four_arm_costs": dict(four_selection.total_costs),
        "five_arm_costs": dict(five_selection.total_costs),
        "four_arm_tail": asdict(four_tail.diagnostics),
        "five_arm_tail": asdict(five_tail.diagnostics),
        "incidence_solver": gnn_result.diagnostics.as_dict(),
    }


def _append_case(
    *,
    arrays: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    identity: Mapping[str, Any],
    right: np.ndarray,
    down: np.ndarray,
    edges: Any,
    edge_features: np.ndarray,
    focal_logits: np.ndarray,
    focal_features: np.ndarray,
    layout_archive: Any,
    bundle: TaskaIncidenceGNNBundle,
) -> None:
    prefix = str(identity["prefix"])
    stacked = stack_taska_focal_features(edge_features, focal_logits, focal_features)
    four_layouts = {
        "raw": _strict_layout(layout_archive[f"{prefix}__raw_layout"]),
        "logistic": _strict_layout(layout_archive[f"{prefix}__logistic_layout"]),
        "focal_top5": _strict_layout(
            layout_archive[f"{prefix}__focal_top5_layout"]
        ),
        "nonlinear": _strict_layout(
            layout_archive[f"{prefix}__nonlinear_layout"]
        ),
    }
    layouts, diagnostics = _compose(
        right=right,
        down=down,
        edges=edges,
        stacked_features=stacked,
        focal_logits=focal_logits,
        four_layouts=four_layouts,
        bundle=bundle,
    )
    frozen_four = _strict_layout(layout_archive[f"{prefix}__four_arm_tail96_layout"])
    if not np.array_equal(layouts["four_arm_tail96"], frozen_four):
        raise RuntimeError("recomputed four-arm control differs from frozen control")
    for arm, layout in layouts.items():
        arrays[f"{prefix}__{arm}_layout"] = _strict_layout(layout)
    arrays[f"{prefix}__incidence_logits"] = diagnostics.pop("incidence_logits")
    rows.append(
        {
            "prefix": prefix,
            "case_id": identity.get("case_id"),
            "source_filename": identity["source_filename"],
            "draw_index": identity["draw_index"],
            "dirty_sha256": identity["dirty_sha256"],
            "candidate_edge_count": len(edges),
            **diagnostics,
        }
    )


def _freeze_stage(
    *,
    stage: str,
    output_dir: Path,
    arrays: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
    contract_path: Path,
    parent_artifacts: Mapping[str, Path],
) -> tuple[Path, Path, Path]:
    stage_dir = output_dir / stage
    stage_dir.mkdir(parents=True, exist_ok=False)
    archive = stage_dir / "frozen-target-free-eval.npz"
    metadata = stage_dir / "frozen-target-free-eval.json"
    freeze = stage_dir / "pre-score-freeze.json"
    baseline._write_npz(archive, arrays)
    _write_json(
        metadata,
        {
            "schema": "aiijc-taska-incidence-gnn-target-free-v1",
            "stage": stage,
            "contains_exact_references_or_labels": False,
            "contains_target_ids_or_source_grid_coordinates": False,
            "candidate_membership_unchanged": True,
            "all_layouts_strict_original_tile_permutations": True,
            "rows": list(rows),
        },
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    weights_path = contract_path.parent / contract["artifacts"]["weights"]["filename"]
    standardizer_path = (
        contract_path.parent / contract["artifacts"]["standardizer"]["filename"]
    )
    artifacts = {
        "model_contract": contract_path,
        "model_weights": weights_path,
        "model_standardizer": standardizer_path,
        "frozen_archive": archive,
        "frozen_metadata": metadata,
        "runner": Path(__file__).resolve(),
        "incidence_gnn_module": (
            PROJECT_ROOT / "src/aiijc_puzzle/taska_incidence_gnn.py"
        ),
        "frozen_raw_solver": (
            PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py"
        ),
        **parent_artifacts,
    }
    _write_json(
        freeze,
        {
            "schema": "aiijc-taska-incidence-gnn-pre-score-freeze-v1",
            "stage": stage,
            "created_before_exact_reference_reconstruction": True,
            "contains_evaluation_references_or_labels": False,
            "artifacts": {key: _record(path) for key, path in artifacts.items()},
        },
    )
    return archive, metadata, freeze


def _cluster_ci(
    values: Sequence[float], sources: Sequence[str], *, seed: int
) -> dict[str, Any]:
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
            for arm in PANEL_ARMS
        },
        "four_arm_choice_counts": dict(
            Counter(row["four_arm_choice"] for row in rows)
        ),
        "five_arm_choice_counts": dict(
            Counter(row["five_arm_choice"] for row in rows)
        ),
    }
    sources = [str(row["source_filename"]) for row in rows]
    deltas: dict[str, Any] = {}
    for index, metric in enumerate(metrics):
        values = [
            float(row["metrics"]["five_arm_tail96"][metric])
            - float(row["metrics"]["four_arm_tail96"][metric])
            for row in rows
        ]
        result = _cluster_ci(values, sources, seed=BOOTSTRAP_SEED + index)
        result["case_wins_ties_losses"] = {
            "wins": sum(value > 0 for value in values),
            "ties": sum(value == 0 for value in values),
            "losses": sum(value < 0 for value in values),
        }
        deltas[metric] = result
    summary["five_minus_four"] = deltas
    return summary


def _score_stage(
    *,
    archive: Path,
    metadata: Path,
    freeze: Path,
    lookup: Mapping[str, Mapping[str, Any]],
    cache: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    baseline._validate_freeze(freeze)
    frozen_rows = json.loads(metadata.read_text(encoding="utf-8"))["rows"]
    scored: list[dict[str, Any]] = []
    with np.load(archive, allow_pickle=False) as candidates:
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
                arm: baseline._layout_metrics(
                    _strict_layout(candidates[f"{prefix}__{arm}_layout"]), reference
                )
                for arm in PANEL_ARMS
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


def _finish_stage(
    *,
    stage: str,
    output_dir: Path,
    arrays: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
    contract_path: Path,
    parent_artifacts: Mapping[str, Path],
    lookup: Mapping[str, Mapping[str, Any]],
    cache: Any,
    started: float,
) -> dict[str, Any]:
    archive, metadata, freeze = _freeze_stage(
        stage=stage,
        output_dir=output_dir,
        arrays=arrays,
        rows=rows,
        contract_path=contract_path,
        parent_artifacts=parent_artifacts,
    )
    scored, summary = _score_stage(
        archive=archive,
        metadata=metadata,
        freeze=freeze,
        lookup=lookup,
        cache=cache,
    )
    return {
        "status": "complete",
        "summary": summary,
        "rows": scored,
        "runtime_seconds": perf_counter() - started,
        "artifacts": {
            "archive": _record(archive),
            "metadata": _record(metadata),
            "pre_score_freeze": _record(freeze),
        },
    }


def _run_local(
    *,
    output_dir: Path,
    contract_path: Path,
    bundle: TaskaIncidenceGNNBundle,
    lookup: Mapping[str, Mapping[str, Any]],
    cache: Any,
) -> dict[str, Any]:
    rows = json.loads(LOCAL_METADATA.read_text(encoding="utf-8"))["rows"]
    if len(rows) != 32:
        raise ValueError("local gate must contain exactly 32 cases")
    arrays: dict[str, np.ndarray] = {}
    frozen_rows: list[dict[str, Any]] = []
    started = perf_counter()
    with np.load(LOCAL_ARCHIVE, allow_pickle=False) as parent:
        for index, row in enumerate(rows):
            prefix = str(row["prefix"])
            edges = baseline._edges_from_archive(parent, prefix)
            _append_case(
                arrays=arrays,
                rows=frozen_rows,
                identity=row,
                right=baseline._finite_matrix(parent, f"{prefix}__cost_right"),
                down=baseline._finite_matrix(parent, f"{prefix}__cost_down"),
                edges=edges,
                edge_features=np.asarray(
                    parent[f"{prefix}__edge_features"], dtype=np.float64
                ),
                focal_logits=np.asarray(
                    parent[f"{prefix}__focal_logits"], dtype=np.float32
                ),
                focal_features=np.asarray(
                    parent[f"{prefix}__focal_features"], dtype=np.float32
                ),
                layout_archive=parent,
                bundle=bundle,
            )
            print(
                json.dumps(
                    {
                        "event": "incidence_gnn_local_target_free",
                        "case": index + 1,
                        "case_count": len(rows),
                    }
                ),
                flush=True,
            )
    result = _finish_stage(
        stage="local32",
        output_dir=output_dir,
        arrays=arrays,
        rows=frozen_rows,
        contract_path=contract_path,
        parent_artifacts={
            "local_parent_archive": LOCAL_ARCHIVE,
            "local_parent_metadata": LOCAL_METADATA,
        },
        lookup=lookup,
        cache=cache,
        started=started,
    )
    pair_delta = result["summary"]["five_minus_four"]["satisfied_adjacent_pairs"][
        "mean"
    ]
    result["gate"] = {
        "rule": "five-arm-tail96 minus four-arm-tail96 pairs >= 0",
        "pair_delta": pair_delta,
        "passed": pair_delta >= 0.0,
    }
    return result


def _run_cached_panel(
    *,
    stage: str,
    output_dir: Path,
    contract_path: Path,
    bundle: TaskaIncidenceGNNBundle,
    lookup: Mapping[str, Mapping[str, Any]],
    cache: Any,
    parent_archive_path: Path,
    parent_metadata_path: Path,
    feature_archive_path: Path,
    feature_metadata_path: Path,
    layout_archive_path: Path,
    layout_metadata_path: Path,
    derive_edge_features: bool,
) -> dict[str, Any]:
    parent_rows = json.loads(parent_metadata_path.read_text(encoding="utf-8"))["rows"]
    feature_rows = json.loads(feature_metadata_path.read_text(encoding="utf-8"))["rows"]
    layout_rows = json.loads(layout_metadata_path.read_text(encoding="utf-8"))["rows"]
    if not (len(parent_rows) == len(feature_rows) == len(layout_rows) == 32):
        raise ValueError(f"{stage} inputs must each contain exactly 32 cases")
    arrays: dict[str, np.ndarray] = {}
    frozen_rows: list[dict[str, Any]] = []
    started = perf_counter()
    with (
        np.load(parent_archive_path, allow_pickle=False) as parent,
        np.load(feature_archive_path, allow_pickle=False) as feature_archive,
        np.load(layout_archive_path, allow_pickle=False) as layout_archive,
    ):
        for index, (row, feature_row, layout_row) in enumerate(
            zip(parent_rows, feature_rows, layout_rows, strict=True)
        ):
            identity = ("prefix", "source_filename", "draw_index", "dirty_sha256")
            if any(
                row[key] != other[key]
                for key in identity
                for other in (feature_row, layout_row)
            ):
                raise RuntimeError(f"{stage} frozen identities differ")
            prefix = str(row["prefix"])
            edges = baseline._edges_from_archive(parent, prefix)
            right = baseline._finite_matrix(parent, f"{prefix}__cost_right")
            down = baseline._finite_matrix(parent, f"{prefix}__cost_down")
            if derive_edge_features:
                edge_features = extract_taska_edge_features(
                    right,
                    down,
                    baseline._finite_matrix(parent, f"{prefix}__right_log"),
                    baseline._finite_matrix(parent, f"{prefix}__down_log"),
                    edges,
                    parent[f"{prefix}__edge_weight"],
                    parent[f"{prefix}__edge_vote_count"],
                    grid=GRID,
                ).values
            else:
                edge_features = np.asarray(
                    feature_archive[f"{prefix}__edge_features"], dtype=np.float64
                )
            _append_case(
                arrays=arrays,
                rows=frozen_rows,
                identity=row,
                right=right,
                down=down,
                edges=edges,
                edge_features=edge_features,
                focal_logits=np.asarray(
                    feature_archive[f"{prefix}__focal_logits"], dtype=np.float32
                ),
                focal_features=np.asarray(
                    feature_archive[f"{prefix}__focal_features"], dtype=np.float32
                ),
                layout_archive=layout_archive,
                bundle=bundle,
            )
            print(
                json.dumps(
                    {
                        "event": f"incidence_gnn_{stage}_target_free",
                        "case": index + 1,
                        "case_count": len(parent_rows),
                    }
                ),
                flush=True,
            )
    return _finish_stage(
        stage=stage,
        output_dir=output_dir,
        arrays=arrays,
        rows=frozen_rows,
        contract_path=contract_path,
        parent_artifacts={
            "parent_archive": parent_archive_path,
            "parent_metadata": parent_metadata_path,
            "feature_archive": feature_archive_path,
            "feature_metadata": feature_metadata_path,
            "four_arm_layout_archive": layout_archive_path,
            "four_arm_layout_metadata": layout_metadata_path,
        },
        lookup=lookup,
        cache=cache,
        started=started,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    _require_inputs()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    bundle, contract_path, training = _fit_bundle(output_dir)
    config, _, local_names = finetune._load_config(finetune.DEFAULT_CONFIG)
    lookup = finetune._manifest_lookup(config)
    cache = finetune.CleanTileCache(args.targets.resolve(), maximum_boards=2)
    local = _run_local(
        output_dir=output_dir,
        contract_path=contract_path,
        bundle=bundle,
        lookup=lookup,
        cache=cache,
    )
    if tuple(row["source_filename"] for row in local["rows"]) != local_names:
        raise RuntimeError("local32 differs from excluded train256 indices 96:128")
    held: dict[str, Any] = {"status": "skipped_by_local_gate"}
    fresh: dict[str, Any] = {"status": "skipped_by_local_or_held_gate"}
    if local["gate"]["passed"]:
        held = _run_cached_panel(
            stage="held32",
            output_dir=output_dir,
            contract_path=contract_path,
            bundle=bundle,
            lookup=lookup,
            cache=cache,
            parent_archive_path=baseline.HELD_PARENT_ARCHIVE,
            parent_metadata_path=baseline.HELD_PARENT_METADATA,
            feature_archive_path=baseline.HELD_FOCAL_ARCHIVE,
            feature_metadata_path=baseline.HELD_FOCAL_METADATA,
            layout_archive_path=HELD_LAYOUT_ARCHIVE,
            layout_metadata_path=HELD_LAYOUT_METADATA,
            derive_edge_features=True,
        )
        held_delta = held["summary"]["five_minus_four"][
            "satisfied_adjacent_pairs"
        ]["mean"]
        held["fresh_gate"] = {
            "rule": "five-arm-tail96 minus four-arm-tail96 pairs >= +0.5",
            "pair_delta": held_delta,
            "passed": held_delta >= 0.5,
        }
        if held["fresh_gate"]["passed"]:
            fresh = _run_cached_panel(
                stage="fresh32",
                output_dir=output_dir,
                contract_path=contract_path,
                bundle=bundle,
                lookup=lookup,
                cache=cache,
                parent_archive_path=baseline.FRESH_PARENT_ARCHIVE,
                parent_metadata_path=baseline.FRESH_PARENT_METADATA,
                feature_archive_path=baseline.FRESH_LEADER_ARCHIVE,
                feature_metadata_path=baseline.FRESH_LEADER_METADATA,
                layout_archive_path=FRESH_LAYOUT_ARCHIVE,
                layout_metadata_path=FRESH_LAYOUT_METADATA,
                derive_edge_features=False,
            )
        else:
            fresh = {"status": "skipped_by_held_pair_gate"}
    report = {
        "schema": REPORT_SCHEMA,
        "protocol": {
            "single_fixed_context_aware_arm": True,
            "training_only_train256_indices": "128:256",
            "local32_indices_excluded_from_fit": "96:128",
            "local_gate": "five-minus-four pairs >= 0",
            "held_gate_for_fresh": "five-minus-four pairs >= +0.5",
            "exact_is_secondary": True,
            "no_hyperparameter_or_epoch_selection": True,
            "candidate_membership_unchanged": True,
        },
        "training": training,
        "local32": local,
        "held32": held,
        "fresh32": fresh,
        "runtime_seconds": perf_counter() - started,
        "legality": {
            "offline_training_labels_only": True,
            "target_free_inference_features_only": True,
            "current_taska_candidate_edges_only": True,
            "original_all_1104_bond_selector_objective": True,
            "strict_original_upright_tile_permutations": True,
            "competition_test_accessed": False,
            "pixels_changed_or_emitted": False,
        },
        "distinction": {
            "failed_linear_pairwise_ranker": "independent edge differences",
            "failed_hgb_stacker": "independent edge rows",
            "prior_union_hard_deepsets": "different Union-hard graph and decoder",
            "this_model": (
                "current TASKA board/axis outgoing-source and incoming-target "
                "incidence competition"
            ),
        },
        "artifacts": {
            "model_contract": _record(contract_path),
            "frozen_raw_solver": _record(
                PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py"
            ),
        },
    }
    _write_json(output_dir / "report.json", report)
    print(
        json.dumps(
            {key: report[key] for key in ("local32", "held32", "fresh32")},
            indent=2,
        )
    )
    return report


if __name__ == "__main__":
    run(parse_args())

#!/usr/bin/env python3
"""Fit and gate the fixed 22-feature logistic stacker on train224.

This is a scale-only continuation of the train96 arm: the estimator and all
solver hyperparameters are unchanged.  It first evaluates the excluded
train256 indices 96:128, then opens the unchanged held32 and fresh32 panels only
under the fixed gates documented in the report.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.taska_edge_calibrator import extract_taska_edge_features
from aiijc_puzzle.taska_focal_feature_stacker import (
    FOCAL_STACKER_FEATURE_NAMES,
    TaskaFocalFeatureStacker,
    fit_taska_focal_feature_stacker,
)
from aiijc_puzzle.taska_pair_pipeline import (
    TaskaPairArtifactPaths,
    load_taska_pair_pipeline_resources,
)

try:
    from scripts import run_taska_focal_current_finetune as finetune
    from scripts import run_taska_focal_feature_stacker as baseline
except ModuleNotFoundError:  # Direct ``python scripts/*.py`` execution.
    import run_taska_focal_current_finetune as finetune
    import run_taska_focal_feature_stacker as baseline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/taska-focal-feature-stacker/train224-v1"
TRAINING_CACHE = DEFAULT_OUTPUT / "training-stacked-features.npz"
TRAINING_METADATA = DEFAULT_OUTPUT / "training-stacked-features.json"
TRAIN96_ROOT = PROJECT_ROOT / "outputs/taska-focal-feature-stacker/train96-v1"
TRAIN96_STACKER = TRAIN96_ROOT / "stacker.npz"
TRAIN96_REPORT = TRAIN96_ROOT / "report.json"
TRAIN96_LOCAL_ARCHIVE = TRAIN96_ROOT / "local32/frozen-target-free-eval.npz"
TRAIN96_LOCAL_METADATA = TRAIN96_ROOT / "local32/frozen-target-free-eval.json"
TRAIN96_HELD_ARCHIVE = TRAIN96_ROOT / "held32/frozen-target-free-eval.npz"
TRAIN96_HELD_METADATA = TRAIN96_ROOT / "held32/frozen-target-free-eval.json"
TRAIN96_FRESH_ARCHIVE = (
    TRAIN96_ROOT / "fresh32-exact-override/frozen-target-free-eval.npz"
)
TRAIN96_FRESH_METADATA = (
    TRAIN96_ROOT / "fresh32-exact-override/frozen-target-free-eval.json"
)
TRAIN224_COUNT = 224
TRAIN256_INDICES = np.concatenate(
    (np.arange(96, dtype=np.int16), np.arange(128, 256, dtype=np.int16))
)
PANEL_ARMS = (
    "stacker",
    "four_arm_tail96",
    "train96_five_arm_tail96",
    "train224_five_arm_tail96",
)
DELTA_COMPARISONS = {
    "train224_minus_four": ("train224_five_arm_tail96", "four_arm_tail96"),
    "train224_minus_train96": (
        "train224_five_arm_tail96",
        "train96_five_arm_tail96",
    ),
    "train96_minus_four": ("train96_five_arm_tail96", "four_arm_tail96"),
}
EXPECTED_SHA256 = {
    TRAIN96_STACKER: "adad56de9245ec999741a0e0966c2767992ba362b6fd731a12885588bd13ae4f",
    TRAIN96_REPORT: "cb2fca61e2f715abe096d9e2d10b4951825628c402adbaad6103aae7a054a485",
    TRAIN96_LOCAL_ARCHIVE: "e615659915d6bb17710403833b46b78d64916e4910afd7effa834a9f46d98e27",
    TRAIN96_LOCAL_METADATA: "78a10b5f3581ef89a1dadae284906cd6f85ec7708250154b4ef4b819cd01a62c",
    TRAIN96_HELD_ARCHIVE: "784a459f6baaaa2d16fd5e1c269c0a50ff456b8c7071ba1cf6035e20be6808f1",
    TRAIN96_HELD_METADATA: "cbe6a774fb3fd3a6095e0124c5f484736ede963c0f6682d93ac49d67fe7c384a",
    TRAIN96_FRESH_ARCHIVE: "61d166fdd5ef275ae0e790951b7d07bb174d66eadbcd4c3b25869a0a587868d2",
    TRAIN96_FRESH_METADATA: "024be2cc842c2c4e7aec8df0a3d10d5f8ea185011f825a9bdb69580f7ee797fb",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=finetune.DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    return parser.parse_args(argv)


def _record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {
        "path": str(resolved.relative_to(PROJECT_ROOT)),
        "sha256": sha256_file(resolved),
    }


def _require_frozen_inputs(output_dir: Path) -> tuple[Path, Path]:
    baseline._require_frozen_inputs()
    for path, expected in EXPECTED_SHA256.items():
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"frozen train96 input changed: {path}")
    training_cache = output_dir / TRAINING_CACHE.name
    training_metadata = output_dir / TRAINING_METADATA.name
    if not training_cache.is_file() or not training_metadata.is_file():
        raise ValueError("materialized train224 training cache is absent")
    metadata = json.loads(training_metadata.read_text(encoding="utf-8"))
    if metadata.get("schema") != (
        "aiijc-taska-focal-stacked-training-cache-train224-metadata-v1"
    ):
        raise ValueError("train224 training metadata schema changed")
    recorded = metadata.get("artifacts", {}).get("combined_cache", {})
    if recorded.get("sha256") != sha256_file(training_cache):
        raise ValueError("train224 combined cache differs from its materializer record")
    if metadata.get("selection", {}).get("excluded_local32_absent") is not True:
        raise ValueError("train224 cache does not certify local32 exclusion")
    return training_cache, training_metadata


def _fit_stacker(
    training_cache: Path,
    *,
    output_dir: Path,
) -> tuple[TaskaFocalFeatureStacker, Path, dict[str, Any]]:
    started = perf_counter()
    with np.load(training_cache, allow_pickle=False) as archive:
        if str(archive["schema"].item()) != (
            "aiijc-taska-focal-stacked-training-cache-train224-v1"
        ):
            raise ValueError("train224 cache schema changed")
        features = np.asarray(archive["features"], dtype=np.float64)
        labels = np.asarray(archive["labels"], dtype=np.uint8)
        offsets = np.asarray(archive["offsets"], dtype=np.int64)
        sources = np.asarray(archive["source_filenames"])
        draws = np.asarray(archive["draw_indices"], dtype=np.uint8)
        indices = np.asarray(archive["train256_indices"], dtype=np.int16)
    if len(sources) != TRAIN224_COUNT or offsets.shape != (TRAIN224_COUNT + 1,):
        raise ValueError("train224 board count changed")
    if not np.array_equal(indices, TRAIN256_INDICES):
        raise ValueError("train224 frozen index selection changed")
    if np.any((indices >= 96) & (indices < 128)):
        raise ValueError("excluded local32 leaked into stacker fit")
    if not np.array_equal(draws, np.zeros(TRAIN224_COUNT, dtype=np.uint8)):
        raise ValueError("train224 draw-index contract changed")
    if offsets[0] != 0 or offsets[-1] != len(labels):
        raise ValueError("train224 training offsets are malformed")
    stacker = fit_taska_focal_feature_stacker(features, labels)
    artifact = output_dir / "stacker.npz"
    if artifact.exists():
        raise FileExistsError(f"refusing to overwrite {artifact}")
    stacker.save_npz(artifact)
    reloaded = TaskaFocalFeatureStacker.load_npz(artifact)
    if not np.array_equal(
        reloaded.predict_logits(features[:2048]), stacker.predict_logits(features[:2048])
    ):
        raise RuntimeError("persisted train224 stacker changed predictions")
    train96_stacker = TaskaFocalFeatureStacker.load_npz(TRAIN96_STACKER)
    return stacker, artifact, {
        "single_fixed_arm": True,
        "board_count": len(sources),
        "edge_count": len(labels),
        "positive_count": int(labels.sum()),
        "positive_fraction": float(labels.mean()),
        "feature_count": len(FOCAL_STACKER_FEATURE_NAMES),
        "feature_names": list(FOCAL_STACKER_FEATURE_NAMES),
        "selected_train256_indices": "0:96 + 128:256",
        "excluded_local32_indices": "96:128",
        "excluded_local32_absent": True,
        "estimator": {
            "pipeline": "StandardScaler -> LogisticRegression",
            "C": 1.0,
            "max_iter": 1000,
            "random_state": 0,
            "class_weight": None,
            "feature_selection": False,
            "hyperparameter_sweep": False,
        },
        "coefficient_l2_distance_from_train96": float(
            np.linalg.norm(stacker.coefficients - train96_stacker.coefficients)
        ),
        "runtime_seconds": perf_counter() - started,
        "artifact": _record(artifact),
    }


def _summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = ("satisfied_adjacent_pairs", "adjacency_recall", "exact_tiles")
    summary: dict[str, Any] = {
        "pair_denominator": baseline.PAIR_DENOMINATOR,
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
        "train96_five_arm_choice_counts": dict(
            Counter(row["train96_five_arm_choice"] for row in rows)
        ),
        "train224_five_arm_choice_counts": dict(
            Counter(row["train224_five_arm_choice"] for row in rows)
        ),
    }
    sources = [str(row["source_filename"]) for row in rows]
    deltas: dict[str, Any] = {}
    for comparison_index, (name, (candidate, reference)) in enumerate(
        DELTA_COMPARISONS.items()
    ):
        comparison: dict[str, Any] = {}
        for metric_index, metric in enumerate(metrics):
            values = [
                float(row["metrics"][candidate][metric])
                - float(row["metrics"][reference][metric])
                for row in rows
            ]
            result = baseline._cluster_ci(
                values,
                sources,
                seed=baseline.BOOTSTRAP_SEED
                + comparison_index * len(metrics)
                + metric_index,
            )
            result["case_wins_ties_losses"] = {
                "wins": sum(value > 0 for value in values),
                "ties": sum(value == 0 for value in values),
                "losses": sum(value < 0 for value in values),
            }
            comparison[metric] = result
        deltas[name] = comparison
    summary["deltas"] = deltas
    return summary


def _freeze_stage(
    *,
    stage: str,
    output_dir: Path,
    arrays: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
    stacker_path: Path,
    parent_artifacts: Mapping[str, Path],
) -> tuple[Path, Path, Path]:
    stage_dir = output_dir / stage
    stage_dir.mkdir(parents=True, exist_ok=False)
    archive = stage_dir / "frozen-target-free-eval.npz"
    metadata = stage_dir / "frozen-target-free-eval.json"
    freeze = stage_dir / "pre-score-freeze.json"
    baseline._write_npz(archive, arrays)
    baseline._write_json(
        metadata,
        {
            "schema": "aiijc-taska-focal-feature-stacker-train224-target-free-v1",
            "stage": stage,
            "contains_exact_references_or_labels": False,
            "contains_target_ids_or_source_grid_coordinates": False,
            "candidate_membership_unchanged": True,
            "all_layouts_strict_original_tile_permutations": True,
            "rows": list(rows),
        },
    )
    source_paths = {
        "stacker_artifact": stacker_path,
        "frozen_archive": archive,
        "frozen_metadata": metadata,
        "runner": Path(__file__).resolve(),
        "generic_stacker_module": (
            PROJECT_ROOT / "src/aiijc_puzzle/taska_focal_feature_stacker.py"
        ),
        "frozen_raw_solver": (
            PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py"
        ),
        **parent_artifacts,
    }
    baseline._write_json(
        freeze,
        {
            "schema": "aiijc-taska-focal-feature-stacker-train224-pre-score-freeze-v1",
            "stage": stage,
            "created_before_exact_reference_reconstruction": True,
            "contains_evaluation_references_or_labels": False,
            "artifacts": {key: _record(path) for key, path in source_paths.items()},
        },
    )
    return archive, metadata, freeze


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
                arm: baseline._layout_metrics(
                    baseline._strict_layout(candidate[f"{prefix}__{arm}_layout"]),
                    reference,
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
                    "train96_five_arm_choice": row["train96_five_arm_choice"],
                    "train224_five_arm_choice": row["train224_five_arm_choice"],
                }
            )
    return scored, _summarize(scored)


def _append_case(
    *,
    arrays: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    prefix: str,
    identity: Mapping[str, Any],
    right: np.ndarray,
    down: np.ndarray,
    edges: Any,
    edge_features: np.ndarray,
    focal_logits: np.ndarray,
    focal_features: np.ndarray,
    raw_layout: np.ndarray,
    focal_layout: np.ndarray,
    known_logistic_layout: np.ndarray | None,
    known_nonlinear_layout: np.ndarray | None,
    retained_archive: Any,
    retained_row: Mapping[str, Any],
    stacker: TaskaFocalFeatureStacker,
    resources: Any,
) -> None:
    layouts, diagnostics = baseline._compose(
        right=right,
        down=down,
        edges=edges,
        edge_features=edge_features,
        focal_logits=focal_logits,
        focal_features=focal_features,
        raw_layout=raw_layout,
        focal_layout=focal_layout,
        logistic=resources.logistic_calibrator,
        nonlinear=resources.nonlinear_calibrator,
        stacker=stacker,
        known_logistic_layout=known_logistic_layout,
        known_nonlinear_layout=known_nonlinear_layout,
    )
    four = baseline._strict_layout(layouts["four_arm_tail96"])
    retained_four = baseline._strict_layout(
        retained_archive[f"{prefix}__four_arm_tail96_layout"]
    )
    if not np.array_equal(four, retained_four):
        raise RuntimeError("recomputed four-arm control differs from frozen train96 control")
    retained = baseline._strict_layout(
        retained_archive[f"{prefix}__five_arm_tail96_layout"]
    )
    candidate = baseline._strict_layout(layouts["five_arm_tail96"])
    frozen_layouts = {
        "stacker": baseline._strict_layout(layouts["stacker"]),
        "four_arm_tail96": four,
        "train96_five_arm_tail96": retained,
        "train224_five_arm_tail96": candidate,
    }
    for arm, layout in frozen_layouts.items():
        arrays[f"{prefix}__{arm}_layout"] = layout
    arrays[f"{prefix}__stacker_logits"] = diagnostics.pop("stacker_logits")
    rows.append(
        {
            "prefix": prefix,
            "case_id": identity.get("case_id"),
            "source_filename": identity["source_filename"],
            "draw_index": identity["draw_index"],
            "dirty_sha256": identity["dirty_sha256"],
            "candidate_edge_count": len(edges),
            "four_arm_choice": diagnostics["four_arm_choice"],
            "train96_five_arm_choice": retained_row["five_arm_choice"],
            "train224_five_arm_choice": diagnostics["five_arm_choice"],
            "four_arm_costs": diagnostics["four_arm_costs"],
            "train224_five_arm_costs": diagnostics["five_arm_costs"],
            "train224_stacker_solver": diagnostics["stacker_solver"],
            "four_arm_tail": diagnostics["four_arm_tail"],
            "train224_five_arm_tail": diagnostics["five_arm_tail"],
        }
    )


def _run_local(
    *,
    output_dir: Path,
    stacker: TaskaFocalFeatureStacker,
    stacker_path: Path,
    resources: Any,
    lookup: Mapping[str, Mapping[str, Any]],
    cache: Any,
) -> dict[str, Any]:
    parent_rows = json.loads(TRAIN96_LOCAL_METADATA.read_text(encoding="utf-8"))["rows"]
    if len(parent_rows) != 32:
        raise ValueError("local gate must contain 32 excluded boards")
    arrays: dict[str, np.ndarray] = {}
    frozen_rows: list[dict[str, Any]] = []
    started = perf_counter()
    with np.load(TRAIN96_LOCAL_ARCHIVE, allow_pickle=False) as parent:
        for index, row in enumerate(parent_rows):
            prefix = str(row["prefix"])
            edges = baseline._edges_from_archive(parent, prefix)
            _append_case(
                arrays=arrays,
                rows=frozen_rows,
                prefix=prefix,
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
                raw_layout=baseline._strict_layout(parent[f"{prefix}__raw_layout"]),
                focal_layout=baseline._strict_layout(
                    parent[f"{prefix}__focal_top5_layout"]
                ),
                known_logistic_layout=baseline._strict_layout(
                    parent[f"{prefix}__logistic_layout"]
                ),
                known_nonlinear_layout=baseline._strict_layout(
                    parent[f"{prefix}__nonlinear_layout"]
                ),
                retained_archive=parent,
                retained_row=row,
                stacker=stacker,
                resources=resources,
            )
            print(
                json.dumps(
                    {
                        "event": "train224_local_target_free",
                        "case": index + 1,
                        "case_count": len(parent_rows),
                    }
                ),
                flush=True,
            )
    archive, metadata, freeze = _freeze_stage(
        stage="local32",
        output_dir=output_dir,
        arrays=arrays,
        rows=frozen_rows,
        stacker_path=stacker_path,
        parent_artifacts={
            "train96_local_archive": TRAIN96_LOCAL_ARCHIVE,
            "train96_local_metadata": TRAIN96_LOCAL_METADATA,
        },
    )
    scored, summary = _score_stage(
        archive=archive,
        metadata=metadata,
        freeze=freeze,
        lookup=lookup,
        cache=cache,
    )
    candidate_minus_four = summary["deltas"]["train224_minus_four"]
    candidate_minus_train96 = summary["deltas"]["train224_minus_train96"]
    gate = {
        "train224_minus_four_pairs_at_least_zero": (
            candidate_minus_four["satisfied_adjacent_pairs"]["mean"] >= 0.0
        ),
        "train224_minus_train96_pairs_at_least_minus_0_25": (
            candidate_minus_train96["satisfied_adjacent_pairs"]["mean"] >= -0.25
        ),
    }
    gate["passed"] = all(gate.values())
    return {
        "status": "complete",
        "gate": gate,
        "summary": summary,
        "rows": scored,
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
    resources: Any,
    lookup: Mapping[str, Mapping[str, Any]],
    cache: Any,
    parent_archive_path: Path,
    parent_metadata_path: Path,
    focal_archive_path: Path,
    focal_metadata_path: Path,
    retained_archive_path: Path,
    retained_metadata_path: Path,
    leader_archive_path: Path | None = None,
) -> dict[str, Any]:
    parent_rows = json.loads(parent_metadata_path.read_text(encoding="utf-8"))["rows"]
    focal_rows = json.loads(focal_metadata_path.read_text(encoding="utf-8"))["rows"]
    retained_rows = json.loads(retained_metadata_path.read_text(encoding="utf-8"))["rows"]
    if not (len(parent_rows) == len(focal_rows) == len(retained_rows) == 32):
        raise ValueError(f"{stage} panel inputs must each contain 32 cases")
    arrays: dict[str, np.ndarray] = {}
    frozen_rows: list[dict[str, Any]] = []
    started = perf_counter()
    with (
        np.load(parent_archive_path, allow_pickle=False) as parent,
        np.load(focal_archive_path, allow_pickle=False) as focal,
        np.load(retained_archive_path, allow_pickle=False) as retained,
    ):
        leader_context = (
            np.load(leader_archive_path, allow_pickle=False)
            if leader_archive_path is not None
            else None
        )
        try:
            for index, (row, focal_row, retained_row) in enumerate(
                zip(parent_rows, focal_rows, retained_rows, strict=True)
            ):
                identity_fields = (
                    "prefix",
                    "source_filename",
                    "draw_index",
                    "dirty_sha256",
                )
                if any(
                    row[field] != other[field]
                    for field in identity_fields
                    for other in (focal_row, retained_row)
                ):
                    raise RuntimeError(f"{stage} frozen panel identities differ")
                prefix = str(row["prefix"])
                edges = baseline._edges_from_archive(parent, prefix)
                right = baseline._finite_matrix(parent, f"{prefix}__cost_right")
                down = baseline._finite_matrix(parent, f"{prefix}__cost_down")
                if leader_context is None:
                    edge_features = extract_taska_edge_features(
                        right,
                        down,
                        baseline._finite_matrix(parent, f"{prefix}__right_log"),
                        baseline._finite_matrix(parent, f"{prefix}__down_log"),
                        edges,
                        parent[f"{prefix}__edge_weight"],
                        parent[f"{prefix}__edge_vote_count"],
                        grid=baseline.GRID,
                    ).values
                    raw_layout = baseline._strict_layout(
                        parent[f"{prefix}__taska_layout"]
                    )
                    focal_layout = baseline._strict_layout(
                        focal[f"{prefix}__focal_layout"]
                    )
                    known_logistic = None
                    known_nonlinear = None
                    focal_logits = np.asarray(
                        focal[f"{prefix}__focal_logits"], dtype=np.float32
                    )
                    focal_features = np.asarray(
                        focal[f"{prefix}__focal_features"], dtype=np.float32
                    )
                else:
                    edge_features = np.asarray(
                        leader_context[f"{prefix}__edge_features"], dtype=np.float64
                    )
                    raw_layout = baseline._strict_layout(
                        leader_context[f"{prefix}__raw_layout"]
                    )
                    focal_layout = baseline._strict_layout(
                        leader_context[f"{prefix}__focal_layout"]
                    )
                    known_logistic = baseline._strict_layout(
                        leader_context[f"{prefix}__logistic_layout"]
                    )
                    known_nonlinear = baseline._strict_layout(
                        leader_context[f"{prefix}__nonlinear_layout"]
                    )
                    focal_logits = np.asarray(
                        leader_context[f"{prefix}__focal_logits"], dtype=np.float32
                    )
                    focal_features = np.asarray(
                        leader_context[f"{prefix}__focal_features"], dtype=np.float32
                    )
                _append_case(
                    arrays=arrays,
                    rows=frozen_rows,
                    prefix=prefix,
                    identity=row,
                    right=right,
                    down=down,
                    edges=edges,
                    edge_features=np.asarray(edge_features),
                    focal_logits=focal_logits,
                    focal_features=focal_features,
                    raw_layout=raw_layout,
                    focal_layout=focal_layout,
                    known_logistic_layout=known_logistic,
                    known_nonlinear_layout=known_nonlinear,
                    retained_archive=retained,
                    retained_row=retained_row,
                    stacker=stacker,
                    resources=resources,
                )
                print(
                    json.dumps(
                        {
                            "event": f"train224_{stage}_target_free",
                            "case": index + 1,
                            "case_count": len(parent_rows),
                        }
                    ),
                    flush=True,
                )
        finally:
            if leader_context is not None:
                leader_context.close()
    artifacts = {
        "parent_archive": parent_archive_path,
        "parent_metadata": parent_metadata_path,
        "focal_archive": focal_archive_path,
        "focal_metadata": focal_metadata_path,
        "retained_train96_archive": retained_archive_path,
        "retained_train96_metadata": retained_metadata_path,
    }
    if leader_archive_path is not None:
        artifacts["four_arm_leader_archive"] = leader_archive_path
    archive, metadata, freeze = _freeze_stage(
        stage=stage,
        output_dir=output_dir,
        arrays=arrays,
        rows=frozen_rows,
        stacker_path=stacker_path,
        parent_artifacts=artifacts,
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


def _fresh_gate(held: Mapping[str, Any]) -> dict[str, Any]:
    summary = held["summary"]
    versus_four = summary["deltas"]["train224_minus_four"]
    versus_train96 = summary["deltas"]["train224_minus_train96"]
    pair_nonnegative = versus_four["satisfied_adjacent_pairs"]["mean"] >= 0.0
    exact_improves_train96 = versus_train96["exact_tiles"]["mean"] > 0.0
    pair_noncatastrophic_vs_train96 = (
        versus_train96["satisfied_adjacent_pairs"]["mean"] >= -1.0
    )
    return {
        "train224_minus_four_pairs_at_least_zero": pair_nonnegative,
        "train224_exact_improves_train96": exact_improves_train96,
        "train224_minus_train96_pairs_at_least_minus_one": (
            pair_noncatastrophic_vs_train96
        ),
        "passed": pair_nonnegative
        or (exact_improves_train96 and pair_noncatastrophic_vs_train96),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    training_cache, training_metadata = _require_frozen_inputs(output_dir)
    for name in ("stacker.npz", "report.json", "local32", "held32", "fresh32"):
        if (output_dir / name).exists():
            raise FileExistsError(f"refusing to overwrite {output_dir / name}")
    if args.device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable")
        torch.use_deterministic_algorithms(False)
    else:
        torch.use_deterministic_algorithms(True)
    started = perf_counter()
    resources = load_taska_pair_pipeline_resources(
        TaskaPairArtifactPaths(), device=args.device
    )
    stacker, stacker_path, training = _fit_stacker(
        training_cache, output_dir=output_dir
    )
    config, _, local_names = finetune._load_config(finetune.DEFAULT_CONFIG)
    lookup = finetune._manifest_lookup(config)
    cache = finetune.CleanTileCache(args.targets.resolve(), maximum_boards=2)
    local = _run_local(
        output_dir=output_dir,
        stacker=stacker,
        stacker_path=stacker_path,
        resources=resources,
        lookup=lookup,
        cache=cache,
    )
    local_sources = tuple(row["source_filename"] for row in local["rows"])
    if local_sources != local_names:
        raise RuntimeError("scored local32 differs from excluded train256 indices 96:128")
    held: dict[str, Any] = {"status": "skipped_by_local_gate"}
    fresh: dict[str, Any] = {"status": "skipped_by_local_or_held_gate"}
    if local["gate"]["passed"]:
        held = _run_cached_panel(
            stage="held32",
            output_dir=output_dir,
            stacker=stacker,
            stacker_path=stacker_path,
            resources=resources,
            lookup=lookup,
            cache=cache,
            parent_archive_path=baseline.HELD_PARENT_ARCHIVE,
            parent_metadata_path=baseline.HELD_PARENT_METADATA,
            focal_archive_path=baseline.HELD_FOCAL_ARCHIVE,
            focal_metadata_path=baseline.HELD_FOCAL_METADATA,
            retained_archive_path=TRAIN96_HELD_ARCHIVE,
            retained_metadata_path=TRAIN96_HELD_METADATA,
        )
        held["fresh_gate"] = _fresh_gate(held)
        if held["fresh_gate"]["passed"]:
            fresh = _run_cached_panel(
                stage="fresh32",
                output_dir=output_dir,
                stacker=stacker,
                stacker_path=stacker_path,
                resources=resources,
                lookup=lookup,
                cache=cache,
                parent_archive_path=baseline.FRESH_PARENT_ARCHIVE,
                parent_metadata_path=baseline.FRESH_PARENT_METADATA,
                focal_archive_path=baseline.FRESH_LEADER_ARCHIVE,
                focal_metadata_path=baseline.FRESH_LEADER_METADATA,
                retained_archive_path=TRAIN96_FRESH_ARCHIVE,
                retained_metadata_path=TRAIN96_FRESH_METADATA,
                leader_archive_path=baseline.FRESH_LEADER_ARCHIVE,
            )
        else:
            fresh = {"status": "skipped_by_fixed_held_gate"}
    report = {
        "schema": "aiijc-taska-focal-feature-stacker-train224-report-v1",
        "protocol": {
            "scale_only_continuation_of_train96": True,
            "estimator_or_hyperparameter_change": False,
            "train256_indices": "0:96 + 128:256",
            "local32_indices_excluded_from_fit": "96:128",
            "local_gate": (
                "train224-minus-four pairs >= 0 AND "
                "train224-minus-train96 pairs >= -0.25"
            ),
            "held_opened_only_after_local_gate": True,
            "fresh_gate": (
                "held train224-minus-four pairs >= 0 OR "
                "(held train224 exact improves train96 AND "
                "train224-minus-train96 pairs >= -1.0)"
            ),
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
            "original_costs_retained_for_placement_fill_selection_and_tail": True,
            "all_outputs_strict_original_upright_tile_permutations": True,
            "competition_test_accessed": False,
            "restored_pixels_emitted": False,
        },
        "artifacts": {
            "stacker": _record(stacker_path),
            "training_cache": _record(training_cache),
            "training_metadata": _record(training_metadata),
            "retained_train96_stacker": _record(TRAIN96_STACKER),
            "frozen_raw_solver": _record(
                PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py"
            ),
            "runner": _record(Path(__file__).resolve()),
        },
    }
    baseline._write_json(output_dir / "report.json", report)
    print(
        json.dumps(
            {key: report[key] for key in ("training", "local32", "held32", "fresh32")},
            indent=2,
        ),
        flush=True,
    )
    return report


if __name__ == "__main__":
    run(parse_args())

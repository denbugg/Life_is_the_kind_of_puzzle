#!/usr/bin/env python3
"""Gate one fixed focal-augmented HGB edge-priority arm on local32.

The runner reads the independently frozen 22-feature train96 cache and the
linear stacker experiment's target-free local32 matcher cache.  It fits no
variants: the estimator contract is exactly the existing TASKA nonlinear
calibrator contract.  Candidate layouts are frozen before references are
reconstructed for scoring.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
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
from aiijc_puzzle.taska_edge_calibrator import solve_prioritized_raw_tail_global
from aiijc_puzzle.taska_focal_nonlinear_stacker import (
    FOCAL_NONLINEAR_FEATURE_NAMES,
    FOCAL_NONLINEAR_PARAMETERS,
    TaskaFocalNonlinearStacker,
    fit_taska_focal_nonlinear_stacker,
    stack_taska_focal_nonlinear_features,
)
from aiijc_puzzle.taska_focal_verifier import load_taska_focal_verifier
from aiijc_puzzle.taska_layout_portfolio import select_lowest_taska_seam_cost_layout
from aiijc_puzzle.taska_pair_pipeline import ARM_NAMES, SOLVER_CONFIG
from aiijc_puzzle.taska_protected_tail_polish import polish_unprotected_taska_tail

try:
    from scripts import run_taska_focal_current_finetune as finetune
except ModuleNotFoundError:  # Direct ``python scripts/run_*.py`` execution.
    import run_taska_focal_current_finetune as finetune

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING = (
    PROJECT_ROOT
    / "outputs/taska-focal-nonlinear-stacker/train96-v1/"
    "training-stacked-features.npz"
)
DEFAULT_PARENT_DIR = (
    PROJECT_ROOT / "outputs/taska-focal-feature-stacker/train96-v1/local32"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/taska-focal-nonlinear-stacker/train96-v1"
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
TRAIN_EDGE_ARCHIVE = (
    PROJECT_ROOT / "outputs/taska-edge-calibrator/train256-v1/training-features.npz"
)
TRAIN_FOCAL_ARCHIVE = (
    PROJECT_ROOT / "outputs/taska-focal-current-finetune/v1/training-harvest.npz"
)
FOCAL_CHECKPOINT = PROJECT_ROOT / "artifacts/prior-taska/ckpt/verify_pair_best.pt"
TRAIN_EDGE_SHA256 = "2d1ef6267daab67d74971d625d2d446e7dfb8dc30a6165bd3459ab969e34f373"
TRAIN_FOCAL_SHA256 = "5ee7b100eb213076fc1acbcace1c6d22e17bea99b88266c5c255cd94c85a17a1"
GRID = 24
COUNT = GRID * GRID
PAIR_DENOMINATOR = 2 * GRID * (GRID - 1)
TAIL_SWAPS = 96
CASE_COUNT = 32
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 2_026_083_158
SCORED_ARMS = ("focal_nonlinear", "four_arm_tail96", "five_arm_tail96")
REPORT_SCHEMA = "aiijc-taska-focal-nonlinear-stacker-report-v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--parent-dir", type=Path, default=DEFAULT_PARENT_DIR)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    return parser.parse_args(argv)


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


def _validate_freeze(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("created_before_exact_reference_reconstruction") is not True:
        raise ValueError("artifact cache was not frozen before scoring")
    if payload.get("contains_evaluation_references_or_labels") is not False:
        raise ValueError("pre-score freeze contains evaluation references")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("pre-score freeze artifact map is malformed")
    for name, record in artifacts.items():
        if not isinstance(record, Mapping):
            raise ValueError(f"pre-score freeze record is malformed: {name}")
        artifact = Path(str(record["path"]))
        if not artifact.is_absolute():
            artifact = PROJECT_ROOT / artifact
        if sha256_file(artifact) != record["sha256"]:
            raise ValueError(f"frozen artifact changed before scoring: {name}")


def _load_parent(parent_dir: Path) -> tuple[Path, Path, Path, list[Mapping[str, Any]]]:
    archive = parent_dir / "frozen-target-free-eval.npz"
    metadata = parent_dir / "frozen-target-free-eval.json"
    freeze = parent_dir / "pre-score-freeze.json"
    for path in (archive, metadata, freeze):
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen parent artifact: {path}")
    _validate_freeze(freeze)
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    if payload.get("contains_exact_references_or_labels") is not False:
        raise ValueError("parent metadata is not target-free")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != CASE_COUNT:
        raise ValueError("parent cache must contain exactly 32 rows")
    return archive, metadata, freeze, rows


def _materialize_training_cache(training_path: Path, *, device: torch.device) -> None:
    if sha256_file(TRAIN_EDGE_ARCHIVE) != TRAIN_EDGE_SHA256:
        raise ValueError("train256 TASKA feature archive SHA-256 changed")
    if sha256_file(TRAIN_FOCAL_ARCHIVE) != TRAIN_FOCAL_SHA256:
        raise ValueError("train96 focal harvest archive SHA-256 changed")
    with (
        np.load(TRAIN_EDGE_ARCHIVE, allow_pickle=False) as edge,
        np.load(TRAIN_FOCAL_ARCHIVE, allow_pickle=False) as focal,
    ):
        edge_offsets = np.asarray(edge["offsets"], dtype=np.int64)
        focal_offsets = np.asarray(focal["offsets"], dtype=np.int64)
        if not np.array_equal(edge_offsets[:97], focal_offsets):
            raise RuntimeError("independent train96 offsets differ")
        edge_sources = np.asarray(edge["source_filenames"][:96])
        focal_sources = np.asarray(focal["source_filenames"])
        if not np.array_equal(edge_sources, focal_sources):
            raise RuntimeError("independent train96 source rosters differ")
        stop = int(focal_offsets[-1])
        edge_features = np.asarray(edge["features"][:stop], dtype=np.float32)
        labels = np.asarray(edge["labels"][:stop], dtype=np.uint8)
        focal_labels = np.asarray(focal["labels"], dtype=np.uint8)
        focal_features = np.asarray(focal["features"], dtype=np.float32)
        patches = np.asarray(focal["patches_uint8"], dtype=np.uint8)
    if not np.array_equal(labels, focal_labels):
        raise RuntimeError("independent train96 binary labels differ")
    model = load_taska_focal_verifier(FOCAL_CHECKPOINT, device=device)
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(labels), 2048):
            stop = min(start + 2048, len(labels))
            outputs.append(
                model(
                    torch.from_numpy(patches[start:stop].astype(np.float32)).to(device),
                    torch.from_numpy(focal_features[start:stop]).to(device),
                )
                .detach()
                .cpu()
                .numpy()
            )
    focal_logits = np.ascontiguousarray(np.concatenate(outputs), dtype=np.float32)
    stacked = stack_taska_focal_nonlinear_features(
        edge_features,
        focal_logits,
        focal_features,
    )
    _write_npz(
        training_path,
        {
            "features22": np.asarray(stacked, dtype=np.float32),
            "labels": labels,
            "offsets": focal_offsets.astype(np.int32),
            "source_filenames": focal_sources,
            "focal_logits": focal_logits,
        },
    )


def _fit_fixed_model(
    training_path: Path,
    output_dir: Path,
) -> tuple[TaskaFocalNonlinearStacker, Path, dict[str, Any]]:
    with np.load(training_path, allow_pickle=False) as archive:
        required = {
            "features22",
            "labels",
            "offsets",
            "source_filenames",
            "focal_logits",
        }
        if set(archive.files) != required:
            raise ValueError("training stacked-feature cache contract differs")
        features = np.asarray(archive["features22"], dtype=np.float64)
        labels = np.asarray(archive["labels"], dtype=np.uint8)
        offsets = np.asarray(archive["offsets"], dtype=np.int64)
        sources = np.asarray(archive["source_filenames"])
        focal_logits = np.asarray(archive["focal_logits"], dtype=np.float64)
    if features.shape != (len(labels), len(FOCAL_NONLINEAR_FEATURE_NAMES)):
        raise ValueError("training feature matrix has the wrong shape")
    if focal_logits.shape != (len(labels),) or not np.array_equal(
        features[:, 15], focal_logits
    ):
        raise ValueError("training recovered focal logits are not aligned")
    if offsets.shape != (97,) or offsets[0] != 0 or offsets[-1] != len(labels):
        raise ValueError("training offsets do not describe 96 boards")
    if len(sources) != 96:
        raise ValueError("training cache does not contain 96 sources")
    started = perf_counter()
    model = fit_taska_focal_nonlinear_stacker(features, labels)
    artifact = output_dir / "focal-nonlinear-stacker.npz"
    if artifact.exists():
        raise FileExistsError(f"refusing to overwrite {artifact}")
    model.save_npz(artifact)
    loaded = TaskaFocalNonlinearStacker.load_npz(artifact)
    if not np.array_equal(
        loaded.predict_logits(features[:1024]),
        model.predict_logits(features[:1024]),
    ):
        raise RuntimeError("persisted nonlinear stacker changed predictions")
    return model, artifact, {
        "single_fixed_arm": True,
        "board_count": 96,
        "edge_count": len(labels),
        "positive_count": int(labels.sum()),
        "positive_fraction": float(labels.mean()),
        "feature_count": len(FOCAL_NONLINEAR_FEATURE_NAMES),
        "feature_names": list(FOCAL_NONLINEAR_FEATURE_NAMES),
        "parameters": dict(FOCAL_NONLINEAR_PARAMETERS),
        "hyperparameter_sweep": False,
        "runtime_seconds": perf_counter() - started,
        "input": _record(training_path),
        "artifact": _record(artifact),
    }


def _compose_case(
    archive: Any,
    prefix: str,
    model: TaskaFocalNonlinearStacker,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    right = _finite_matrix(archive, f"{prefix}__cost_right")
    down = _finite_matrix(archive, f"{prefix}__cost_down")
    edges = _edges_from_archive(archive, prefix)
    features = np.asarray(archive[f"{prefix}__edge_features"], dtype=np.float64)
    focal_logits = np.asarray(archive[f"{prefix}__focal_logits"], dtype=np.float64)
    focal_features = np.asarray(archive[f"{prefix}__focal_features"], dtype=np.float64)
    if features.shape != (len(edges), 15):
        raise ValueError("cached TASKA features are malformed")
    stacked = np.ascontiguousarray(np.column_stack((features, focal_logits, focal_features)))
    if stacked.shape != (len(edges), len(FOCAL_NONLINEAR_FEATURE_NAMES)):
        raise ValueError("cached 22-feature matrix is malformed")
    priorities = model.predict_priorities(stacked)
    standalone = solve_prioritized_raw_tail_global(
        right,
        down,
        edges,
        priorities,
        grid=GRID,
        config=SOLVER_CONFIG,
    )
    standalone_layout = _strict_layout(standalone.layout)
    four = {
        arm: _strict_layout(archive[f"{prefix}__{arm}_layout"])
        for arm in ARM_NAMES
    }
    four_selection = select_lowest_taska_seam_cost_layout(four, right, down, grid=GRID)
    four_tail = polish_unprotected_taska_tail(
        four_selection.layout,
        right,
        down,
        edges,
        grid=GRID,
        max_swaps=TAIL_SWAPS,
    )
    known_control = _strict_layout(archive[f"{prefix}__four_arm_tail96_layout"])
    if not np.array_equal(four_tail.layout, known_control):
        raise RuntimeError("replayed four-arm control differs from frozen parent")
    five = {**four, "focal_nonlinear": standalone_layout}
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
        "focal_nonlinear": standalone_layout,
        "four_arm_tail96": known_control,
        "five_arm_tail96": _strict_layout(five_tail.layout),
    }, {
        "candidate_edge_count": len(edges),
        "four_arm_choice": four_selection.choice,
        "five_arm_choice": five_selection.choice,
        "four_arm_total_costs": dict(four_selection.total_costs),
        "five_arm_total_costs": dict(five_selection.total_costs),
        "four_arm_tail96": asdict(four_tail.diagnostics),
        "five_arm_tail96": asdict(five_tail.diagnostics),
        "standalone_solver": standalone.diagnostics.as_dict(),
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


def _bootstrap(values: Sequence[float], *, seed: int) -> dict[str, Any]:
    sample = np.asarray(values, dtype=np.float64)
    if sample.shape != (CASE_COUNT,) or not np.isfinite(sample).all():
        raise ValueError("local comparison must contain 32 finite source values")
    generator = np.random.default_rng(seed)
    distribution = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_RESAMPLES, 2048):
        stop = min(start + 2048, BOOTSTRAP_RESAMPLES)
        indices = generator.integers(0, len(sample), size=(stop - start, len(sample)))
        distribution[start:stop] = sample[indices].mean(axis=1)
    return {
        "mean": float(sample.mean()),
        "ci95_lower": float(np.quantile(distribution, 0.025)),
        "ci95_upper": float(np.quantile(distribution, 0.975)),
        "source_count": CASE_COUNT,
        "case_count": CASE_COUNT,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": seed,
        "source_wins_ties_losses": {
            "wins": int(np.count_nonzero(sample > 0)),
            "ties": int(np.count_nonzero(sample == 0)),
            "losses": int(np.count_nonzero(sample < 0)),
        },
    }


def _summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = ("satisfied_adjacent_pairs", "adjacency_recall", "exact_tiles")
    result: dict[str, Any] = {
        "pair_denominator": PAIR_DENOMINATOR,
        "arms": {
            arm: {
                metric: float(np.mean([row["metrics"][arm][metric] for row in rows]))
                for metric in metrics
            }
            for arm in SCORED_ARMS
        },
        "four_arm_choice_counts": dict(Counter(row["four_arm_choice"] for row in rows)),
        "five_arm_choice_counts": dict(Counter(row["five_arm_choice"] for row in rows)),
    }
    deltas: dict[str, Any] = {}
    for index, metric in enumerate(metrics):
        values = [
            float(row["metrics"]["five_arm_tail96"][metric])
            - float(row["metrics"]["four_arm_tail96"][metric])
            for row in rows
        ]
        deltas[metric] = _bootstrap(values, seed=BOOTSTRAP_SEED + index)
    result["five_minus_four"] = deltas
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    training = args.training.resolve()
    parent_dir = args.parent_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    if args.device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable")
        torch.use_deterministic_algorithms(False)
    else:
        torch.use_deterministic_algorithms(True)
    if not training.is_file():
        _materialize_training_cache(training, device=torch.device(args.device))
    parent_archive, parent_metadata, parent_freeze, parent_rows = _load_parent(parent_dir)
    model, model_path, training_summary = _fit_fixed_model(training, output_dir)

    candidate_arrays: dict[str, np.ndarray] = {}
    frozen_rows: list[dict[str, Any]] = []
    with np.load(parent_archive, allow_pickle=False) as parent:
        for index, row in enumerate(parent_rows):
            prefix = str(row["prefix"])
            layouts, diagnostics = _compose_case(parent, prefix, model)
            for arm, layout in layouts.items():
                candidate_arrays[f"{prefix}__{arm}_layout"] = layout
            frozen_rows.append(
                {
                    "prefix": prefix,
                    "source_filename": str(row["source_filename"]),
                    "draw_index": int(row["draw_index"]),
                    "dirty_sha256": str(row["dirty_sha256"]),
                    **diagnostics,
                }
            )
            print(
                json.dumps(
                    {
                        "event": "focal_nonlinear_local_target_free",
                        "case": index + 1,
                        "case_count": CASE_COUNT,
                    }
                ),
                flush=True,
            )

    candidate_archive = output_dir / "local32-target-free.npz"
    candidate_metadata = output_dir / "local32-target-free.json"
    pre_score_freeze = output_dir / "local32-pre-score-freeze.json"
    _write_npz(candidate_archive, candidate_arrays)
    _write_json(
        candidate_metadata,
        {
            "schema": "aiijc-taska-focal-nonlinear-stacker-target-free-v1",
            "contains_exact_references_or_labels": False,
            "candidate_membership_unchanged": True,
            "all_layouts_strict_original_tile_permutations": True,
            "rows": frozen_rows,
        },
    )
    _write_json(
        pre_score_freeze,
        {
            "schema": "aiijc-taska-focal-nonlinear-stacker-pre-score-freeze-v1",
            "created_before_exact_reference_reconstruction": True,
            "contains_evaluation_references_or_labels": False,
            "artifacts": {
                "training_cache": _record(training),
                "stacker": _record(model_path),
                "candidate_archive": _record(candidate_archive),
                "candidate_metadata": _record(candidate_metadata),
                "parent_archive": _record(parent_archive),
                "parent_metadata": _record(parent_metadata),
                "parent_freeze": _record(parent_freeze),
                "runner": _record(Path(__file__)),
                "runtime": _record(
                    PROJECT_ROOT
                    / "src/aiijc_puzzle/taska_focal_nonlinear_stacker.py"
                ),
                "raw_solver": _record(
                    PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py"
                ),
            },
        },
    )
    _validate_freeze(pre_score_freeze)

    config, _, _ = finetune._load_config(finetune.DEFAULT_CONFIG)
    lookup = finetune._manifest_lookup(config)
    cache = finetune.CleanTileCache(args.targets.resolve(), maximum_boards=2)
    scored: list[dict[str, Any]] = []
    with np.load(candidate_archive, allow_pickle=False) as candidate:
        for row in frozen_rows:
            source = str(row["source_filename"])
            draw = int(row["draw_index"])
            dirty = finetune._dirty_case(cache, lookup[source], source, draw)
            if finetune._dirty_sha256(dirty.dirty_tiles) != row["dirty_sha256"]:
                raise RuntimeError("scoring recreated different dirty bytes")
            reference = finetune._reference(
                cache,
                lookup[source],
                source,
                draw,
                dirty.dirty_tiles,
            )
            prefix = str(row["prefix"])
            metrics = {
                arm: _layout_metrics(
                    _strict_layout(candidate[f"{prefix}__{arm}_layout"]),
                    reference,
                )
                for arm in SCORED_ARMS
            }
            scored.append(
                {
                    "prefix": prefix,
                    "source_filename": source,
                    "draw_index": draw,
                    "four_arm_choice": row["four_arm_choice"],
                    "five_arm_choice": row["five_arm_choice"],
                    "metrics": metrics,
                }
            )
    summary = _summarize(scored)
    pair_delta = summary["five_minus_four"]["satisfied_adjacent_pairs"]["mean"]
    report = {
        "schema": REPORT_SCHEMA,
        "protocol": {
            "single_fixed_learned_arm": True,
            "fixed_estimator_contract": dict(FOCAL_NONLINEAR_PARAMETERS),
            "feature_count": len(FOCAL_NONLINEAR_FEATURE_NAMES),
            "no_hyperparameter_sweep": True,
            "local_gate": "five-arm-tail96 minus four-arm-tail96 pairs >= 0",
            "held_opened_only_after_local_gate": True,
            "fresh_requires_materially_positive_held_result": True,
        },
        "training": training_summary,
        "local32": {
            "status": "complete",
            "gate_passed": pair_delta >= 0.0,
            "summary": summary,
            "rows": scored,
            "artifacts": {
                "archive": _record(candidate_archive),
                "metadata": _record(candidate_metadata),
                "pre_score_freeze": _record(pre_score_freeze),
            },
        },
        "held32": {
            "status": (
                "requires_followup_after_passed_local_gate"
                if pair_delta >= 0.0
                else "skipped_by_negative_local_gate"
            )
        },
        "fresh32": {"status": "not_opened"},
        "runtime_seconds": perf_counter() - started,
        "legality": {
            "offline_train_labels_only": True,
            "target_free_features_only_at_inference": True,
            "candidate_membership_unchanged": True,
            "original_costs_retained_for_placement_and_fill": True,
            "strict_original_upright_tile_permutations": True,
            "competition_test_accessed": False,
            "restored_pixels_emitted": False,
        },
        "artifacts": {
            "model": _record(model_path),
            "training_cache": _record(training),
            "parent_cache": _record(parent_archive),
            "raw_solver": _record(
                PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py"
            ),
        },
    }
    _write_json(output_dir / "report.json", report)
    print(json.dumps(report["local32"]["summary"], indent=2), flush=True)
    return report


if __name__ == "__main__":
    run(parse_args())

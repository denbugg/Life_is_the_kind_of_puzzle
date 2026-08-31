#!/usr/bin/env python3
"""Fit and gate one fixed board/axis focal pairwise TASKA ranker.

The matcher is not rerun.  Training reuses the audited aligned first-96 edge
and focal caches.  Evaluation reuses frozen target-free TASKA evidence and
adds one layout arm whose component order is the portable pairwise linear
score.  Original TASKA costs still place/fill components, select the five-arm
portfolio, and polish the same protected tail with at most 96 swaps.

Candidate layouts are SHA-frozen before exact synthetic references are
reconstructed.  The disjoint local32 gate opens held32 at nonnegative pair
delta.  Fresh32 opens only at a predeclared held gain of at least one satisfied
pair per board.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np
import torch

from aiijc_puzzle.layout_evaluation import evaluate_layout
from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.taska_edge_calibrator import (
    extract_taska_edge_features,
    solve_prioritized_raw_tail_global,
)
from aiijc_puzzle.taska_focal_feature_stacker import stack_taska_focal_features
from aiijc_puzzle.taska_focal_pairwise_ranker import (
    FOCAL_STACKER_FEATURE_NAMES,
    PAIRWISE_RANKER_PARAMETERS,
    TaskaFocalPairwiseRanker,
    fit_taska_focal_pairwise_ranker,
)
from aiijc_puzzle.taska_focal_verifier import load_taska_focal_verifier
from aiijc_puzzle.taska_layout_portfolio import select_lowest_taska_seam_cost_layout
from aiijc_puzzle.taska_pair_pipeline import (
    ARM_NAMES,
    SOLVER_CONFIG,
    TaskaPairArtifactPaths,
)
from aiijc_puzzle.taska_protected_tail_polish import polish_unprotected_taska_tail

try:
    from scripts import run_taska_focal_current_finetune as finetune
    from scripts import run_taska_focal_feature_stacker as stacker_parent
except ModuleNotFoundError:  # Direct ``python scripts/run_*.py`` execution.
    import run_taska_focal_current_finetune as finetune
    import run_taska_focal_feature_stacker as stacker_parent

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/taska-focal-pairwise-ranker/train96-v1"
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
TRAIN_EDGE_ARCHIVE = stacker_parent.TRAIN_EDGE_ARCHIVE
TRAIN_FOCAL_ARCHIVE = stacker_parent.TRAIN_FOCAL_ARCHIVE

STACKER_OUTPUT = PROJECT_ROOT / "outputs/taska-focal-feature-stacker/train96-v1"
LOCAL_ARCHIVE = STACKER_OUTPUT / "local32/frozen-target-free-eval.npz"
LOCAL_METADATA = STACKER_OUTPUT / "local32/frozen-target-free-eval.json"
HELD_PORTFOLIO_ARCHIVE = STACKER_OUTPUT / "held32/frozen-target-free-eval.npz"
HELD_PORTFOLIO_METADATA = STACKER_OUTPUT / "held32/frozen-target-free-eval.json"
HELD_BASE_ARCHIVE = stacker_parent.HELD_PARENT_ARCHIVE
HELD_FOCAL_ARCHIVE = stacker_parent.HELD_FOCAL_ARCHIVE

FRESH_BASE_ARCHIVE = stacker_parent.FRESH_PARENT_ARCHIVE
FRESH_BASE_METADATA = stacker_parent.FRESH_PARENT_METADATA
FRESH_LEADER_ARCHIVE = stacker_parent.FRESH_LEADER_ARCHIVE
FRESH_LEADER_METADATA = stacker_parent.FRESH_LEADER_METADATA

EXPECTED_SHA256 = {
    LOCAL_ARCHIVE: "e615659915d6bb17710403833b46b78d64916e4910afd7effa834a9f46d98e27",
    LOCAL_METADATA: "78a10b5f3581ef89a1dadae284906cd6f85ec7708250154b4ef4b819cd01a62c",
    HELD_PORTFOLIO_ARCHIVE: (
        "784a459f6baaaa2d16fd5e1c269c0a50ff456b8c7071ba1cf6035e20be6808f1"
    ),
    HELD_PORTFOLIO_METADATA: (
        "cbe6a774fb3fd3a6095e0124c5f484736ede963c0f6682d93ac49d67fe7c384a"
    ),
}

GRID = 24
COUNT = GRID * GRID
PAIR_DENOMINATOR = 2 * GRID * (GRID - 1)
TRAIN_BOARD_COUNT = 96
TAIL_MAX_SWAPS = 96
TAIL_MINIMUM_GAIN = 1e-9
FRESH_HELD_PAIR_DELTA_GATE = 1.0
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 2_026_083_162
REPORT_SCHEMA = "aiijc-taska-focal-pairwise-ranker-report-v1"
STAGE_ARMS = ("pairwise_ranker", "four_arm_tail96", "five_arm_tail96")
PanelName = Literal["local32", "held32", "fresh32"]


@dataclass(frozen=True)
class PanelSpec:
    metadata: Path
    base_archive: Path
    evidence_archive: Path
    portfolio_archive: Path


PANEL_SPECS: dict[PanelName, PanelSpec] = {
    "local32": PanelSpec(
        metadata=LOCAL_METADATA,
        base_archive=LOCAL_ARCHIVE,
        evidence_archive=LOCAL_ARCHIVE,
        portfolio_archive=LOCAL_ARCHIVE,
    ),
    "held32": PanelSpec(
        metadata=HELD_PORTFOLIO_METADATA,
        base_archive=HELD_BASE_ARCHIVE,
        evidence_archive=HELD_FOCAL_ARCHIVE,
        portfolio_archive=HELD_PORTFOLIO_ARCHIVE,
    ),
    "fresh32": PanelSpec(
        metadata=FRESH_LEADER_METADATA,
        base_archive=FRESH_BASE_ARCHIVE,
        evidence_archive=FRESH_LEADER_ARCHIVE,
        portfolio_archive=FRESH_LEADER_ARCHIVE,
    ),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--smoke-one", action="store_true")
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


def _require_frozen_inputs() -> None:
    stacker_parent._require_frozen_inputs()
    for path, expected in EXPECTED_SHA256.items():
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"frozen pairwise input SHA-256 mismatch: {path}")


def _strict_layout(value: Any) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=np.int32)
    if result.shape != (COUNT,) or not np.array_equal(np.sort(result), np.arange(COUNT)):
        raise ValueError("layout is not a strict 576-tile permutation")
    result.setflags(write=False)
    return result


def _finite_matrix(archive: Any, key: str) -> np.ndarray:
    result = np.asarray(archive[key], dtype=np.float64)
    if result.shape != (COUNT, COUNT) or not np.isfinite(result).all():
        raise ValueError(f"{key} must be one finite 576x576 matrix")
    return np.ascontiguousarray(result)


def _edges(archive: Any, prefix: str) -> tuple[RawTailEdge, ...]:
    return stacker_parent._edges_from_archive(archive, prefix)


def _score_cached_training_patches(
    model: torch.nn.Module,
    patches: np.ndarray,
    focal_features: np.ndarray,
    *,
    device: torch.device,
    chunk_size: int = 2048,
) -> np.ndarray:
    if len(patches) != len(focal_features):
        raise ValueError("cached focal patches and feature rows are misaligned")
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(patches), chunk_size):
            stop = min(start + chunk_size, len(patches))
            outputs.append(
                model(
                    torch.from_numpy(patches[start:stop].astype(np.float32)).to(device),
                    torch.from_numpy(focal_features[start:stop]).to(device),
                )
                .detach()
                .cpu()
                .numpy()
            )
    result = np.ascontiguousarray(np.concatenate(outputs), dtype=np.float32)
    if result.shape != (len(patches),) or not np.isfinite(result).all():
        raise RuntimeError("recovered focal logits are malformed")
    return result


def _fit_ranker(
    *,
    device: torch.device,
    output_dir: Path,
) -> tuple[TaskaFocalPairwiseRanker, Path, dict[str, Any]]:
    with (
        np.load(TRAIN_EDGE_ARCHIVE, allow_pickle=False) as edge,
        np.load(TRAIN_FOCAL_ARCHIVE, allow_pickle=False) as focal,
    ):
        edge_offsets = np.asarray(edge["offsets"], dtype=np.int64)
        focal_offsets = np.asarray(focal["offsets"], dtype=np.int64)
        if not np.array_equal(edge_offsets[: TRAIN_BOARD_COUNT + 1], focal_offsets):
            raise RuntimeError("train96 cached offsets differ")
        if not np.array_equal(
            edge["source_filenames"][:TRAIN_BOARD_COUNT], focal["source_filenames"]
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
    paths = TaskaPairArtifactPaths()
    focal_model = load_taska_focal_verifier(paths.focal_verifier, device=device)
    started = perf_counter()
    focal_logits = _score_cached_training_patches(
        focal_model,
        patches,
        focal_features,
        device=device,
    )
    features = stack_taska_focal_features(
        edge_features,
        focal_logits,
        focal_features,
    )
    ranker, pair_diagnostics = fit_taska_focal_pairwise_ranker(
        features,
        edge_labels,
        focal_offsets,
    )
    artifact = output_dir / "pairwise-ranker.npz"
    ranker.save_npz(artifact)
    reloaded = TaskaFocalPairwiseRanker.load_npz(artifact)
    if not np.array_equal(
        reloaded.predict_scores(features[:1024]), ranker.predict_scores(features[:1024])
    ):
        raise RuntimeError("persisted pairwise ranker changed its scores")
    return ranker, artifact, {
        "single_fixed_arm": True,
        "board_count": TRAIN_BOARD_COUNT,
        "edge_count": len(edge_labels),
        "positive_count": int(edge_labels.sum()),
        "feature_count": len(FOCAL_STACKER_FEATURE_NAMES),
        "feature_names": list(FOCAL_STACKER_FEATURE_NAMES),
        "estimator": {
            "original_row_scaler": "StandardScaler fitted on all original train rows",
            "pairwise_head": "LogisticRegression",
            **PAIRWISE_RANKER_PARAMETERS,
            "symmetric_sign_reversal": True,
            "hyperparameter_sweep": False,
        },
        "pair_construction": pair_diagnostics,
        "cache_alignment": {
            "source_names_equal": True,
            "offsets_equal": True,
            "labels_equal": True,
        },
        "runtime_seconds": perf_counter() - started,
        "artifact": _record(artifact),
    }


def _case_evidence(
    panel: PanelName,
    prefix: str,
    base: Any,
    evidence: Any,
    portfolio: Any,
) -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[RawTailEdge, ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    np.ndarray,
]:
    right = _finite_matrix(base, f"{prefix}__cost_right")
    down = _finite_matrix(base, f"{prefix}__cost_down")
    edges = _edges(base, prefix)
    if panel == "local32":
        edge_features = np.asarray(base[f"{prefix}__edge_features"], dtype=np.float64)
        focal_logits = np.asarray(base[f"{prefix}__focal_logits"], dtype=np.float64)
        focal_features = np.asarray(base[f"{prefix}__focal_features"], dtype=np.float64)
        four = {
            name: _strict_layout(portfolio[f"{prefix}__{name}_layout"])
            for name in ARM_NAMES
        }
        control = _strict_layout(portfolio[f"{prefix}__four_arm_tail96_layout"])
    elif panel == "held32":
        edge_features = extract_taska_edge_features(
            right,
            down,
            _finite_matrix(base, f"{prefix}__right_log"),
            _finite_matrix(base, f"{prefix}__down_log"),
            edges,
            base[f"{prefix}__edge_weight"],
            base[f"{prefix}__edge_vote_count"],
            grid=GRID,
        ).values
        focal_logits = np.asarray(
            evidence[f"{prefix}__focal_logits"], dtype=np.float64
        )
        focal_features = np.asarray(
            evidence[f"{prefix}__focal_features"], dtype=np.float64
        )
        four = {
            name: _strict_layout(portfolio[f"{prefix}__{name}_layout"])
            for name in ARM_NAMES
        }
        control = _strict_layout(portfolio[f"{prefix}__four_arm_tail96_layout"])
    else:
        edge_features = np.asarray(
            evidence[f"{prefix}__edge_features"], dtype=np.float64
        )
        focal_logits = np.asarray(
            evidence[f"{prefix}__focal_logits"], dtype=np.float64
        )
        focal_features = np.asarray(
            evidence[f"{prefix}__focal_features"], dtype=np.float64
        )
        four = {
            "raw": _strict_layout(portfolio[f"{prefix}__raw_layout"]),
            "logistic": _strict_layout(portfolio[f"{prefix}__logistic_layout"]),
            "focal_top5": _strict_layout(portfolio[f"{prefix}__focal_layout"]),
            "nonlinear": _strict_layout(portfolio[f"{prefix}__nonlinear_layout"]),
        }
        control = _strict_layout(portfolio[f"{prefix}__portfolio_tail96_layout"])
    if tuple(four) != ARM_NAMES:
        raise RuntimeError("four-arm order differs from production pipeline")
    rows = len(edges)
    if not (
        edge_features.shape == (rows, 15)
        and focal_logits.shape == (rows,)
        and focal_features.shape == (rows, 6)
    ):
        raise RuntimeError(f"{panel} cached edge evidence is misaligned")
    return (
        right,
        down,
        edges,
        edge_features,
        focal_logits,
        focal_features,
        four,
        control,
    )


def _compose_case(
    panel: PanelName,
    prefix: str,
    base: Any,
    evidence: Any,
    portfolio: Any,
    ranker: TaskaFocalPairwiseRanker,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    (
        right,
        down,
        edges,
        edge_features,
        focal_logits,
        focal_features,
        four,
        control,
    ) = _case_evidence(panel, prefix, base, evidence, portfolio)
    features = stack_taska_focal_features(
        edge_features,
        focal_logits,
        focal_features,
    )
    scores = ranker.predict_priorities(features)
    solved = solve_prioritized_raw_tail_global(
        right,
        down,
        edges,
        scores,
        grid=GRID,
        config=SOLVER_CONFIG,
    )
    ranker_layout = _strict_layout(solved.layout)
    selection = select_lowest_taska_seam_cost_layout(
        {**four, "pairwise_ranker": ranker_layout},
        right,
        down,
        grid=GRID,
    )
    tail = polish_unprotected_taska_tail(
        selection.layout,
        right,
        down,
        edges,
        grid=GRID,
        max_swaps=TAIL_MAX_SWAPS,
        minimum_gain=TAIL_MINIMUM_GAIN,
    )
    return {
        "pairwise_ranker": ranker_layout,
        "four_arm_tail96": control,
        "five_arm_tail96": _strict_layout(tail.layout),
        "pairwise_scores": np.ascontiguousarray(scores, dtype=np.float32),
    }, {
        "five_arm_choice": selection.choice,
        "five_arm_costs": dict(selection.total_costs),
        "five_arm_tail": asdict(tail.diagnostics),
        "pairwise_solver": solved.diagnostics.as_dict(),
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
            0,
            len(cluster_means),
            size=(stop - start, len(cluster_means)),
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
    panel: PanelName,
    output_dir: Path,
    arrays: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
    ranker_path: Path,
    spec: PanelSpec,
) -> tuple[Path, Path, Path]:
    stage_dir = output_dir / panel
    stage_dir.mkdir(parents=True, exist_ok=False)
    archive = stage_dir / "frozen-target-free-eval.npz"
    metadata = stage_dir / "frozen-target-free-eval.json"
    freeze = stage_dir / "pre-score-freeze.json"
    _write_npz(archive, arrays)
    _write_json(
        metadata,
        {
            "schema": "aiijc-taska-focal-pairwise-ranker-target-free-v1",
            "panel": panel,
            "contains_exact_references_or_labels": False,
            "contains_target_ids_or_source_grid_coordinates": False,
            "candidate_membership_unchanged": True,
            "all_layouts_strict_original_upright_tile_permutations": True,
            "rows": list(rows),
        },
    )
    sources = {
        "pairwise_ranker": ranker_path,
        "frozen_archive": archive,
        "frozen_metadata": metadata,
        "panel_parent_metadata": spec.metadata,
        "panel_base_archive": spec.base_archive,
        "panel_evidence_archive": spec.evidence_archive,
        "panel_portfolio_archive": spec.portfolio_archive,
        "runner": Path(__file__).resolve(),
        "ranker_module": PROJECT_ROOT
        / "src/aiijc_puzzle/taska_focal_pairwise_ranker.py",
        "frozen_raw_solver": PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py",
    }
    _write_json(
        freeze,
        {
            "schema": "aiijc-taska-focal-pairwise-ranker-pre-score-freeze-v1",
            "panel": panel,
            "created_before_exact_reference_reconstruction": True,
            "contains_evaluation_references_or_labels": False,
            "artifacts": {name: _record(path) for name, path in sources.items()},
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
                cache,
                lookup[source],
                source,
                draw,
                dirty.dirty_tiles,
            )
            metrics = {
                arm: _layout_metrics(
                    _strict_layout(candidate[f"{prefix}__{arm}_layout"]),
                    reference,
                )
                for arm in STAGE_ARMS
            }
            scored.append(
                {
                    "prefix": prefix,
                    "source_filename": source,
                    "draw_index": draw,
                    "metrics": metrics,
                    "five_arm_choice": row["five_arm_choice"],
                }
            )
    return scored, _summarize(scored)


def _run_panel(
    *,
    panel: PanelName,
    output_dir: Path,
    ranker: TaskaFocalPairwiseRanker,
    ranker_path: Path,
    lookup: Mapping[str, Mapping[str, Any]],
    cache: Any,
    smoke_one: bool = False,
) -> dict[str, Any]:
    spec = PANEL_SPECS[panel]
    parent_rows = json.loads(spec.metadata.read_text(encoding="utf-8"))["rows"]
    if len(parent_rows) != 32:
        raise ValueError(f"{panel} must contain exactly 32 cases")
    if smoke_one:
        parent_rows = parent_rows[:1]
    arrays: dict[str, np.ndarray] = {}
    frozen_rows: list[dict[str, Any]] = []
    started = perf_counter()
    with ExitStack() as stack:
        loaded: dict[Path, Any] = {}

        def archive(path: Path) -> Any:
            resolved = path.resolve()
            if resolved not in loaded:
                loaded[resolved] = stack.enter_context(
                    np.load(resolved, allow_pickle=False)
                )
            return loaded[resolved]

        base = archive(spec.base_archive)
        evidence = archive(spec.evidence_archive)
        portfolio = archive(spec.portfolio_archive)
        for index, parent_row in enumerate(parent_rows):
            prefix = str(parent_row["prefix"])
            layouts, diagnostics = _compose_case(
                panel,
                prefix,
                base,
                evidence,
                portfolio,
                ranker,
            )
            scores = layouts.pop("pairwise_scores")
            for arm, layout in layouts.items():
                arrays[f"{prefix}__{arm}_layout"] = _strict_layout(layout)
            arrays[f"{prefix}__pairwise_scores"] = scores
            frozen_rows.append(
                {
                    "prefix": prefix,
                    "case_id": parent_row.get("case_id"),
                    "source_filename": parent_row["source_filename"],
                    "draw_index": parent_row["draw_index"],
                    "dirty_sha256": parent_row["dirty_sha256"],
                    "candidate_edge_count": int(len(scores)),
                    **diagnostics,
                }
            )
            print(
                json.dumps(
                    {
                        "event": f"pairwise_ranker_{panel}_target_free",
                        "case": index + 1,
                        "case_count": len(parent_rows),
                    }
                ),
                flush=True,
            )
    candidate, metadata, freeze = _freeze_stage(
        panel=panel,
        output_dir=output_dir,
        arrays=arrays,
        rows=frozen_rows,
        ranker_path=ranker_path,
        spec=spec,
    )
    rows, summary = _score_stage(
        archive=candidate,
        metadata=metadata,
        freeze=freeze,
        lookup=lookup,
        cache=cache,
    )
    return {
        "status": "smoke-only" if smoke_one else "complete",
        "summary": summary,
        "rows": rows,
        "runtime_seconds": perf_counter() - started,
        "artifacts": {
            "archive": _record(candidate),
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
    device = torch.device(args.device)
    started = perf_counter()
    ranker, ranker_path, training = _fit_ranker(device=device, output_dir=output_dir)
    config, _, _ = finetune._load_config(finetune.DEFAULT_CONFIG)
    lookup = finetune._manifest_lookup(config)
    cache = finetune.CleanTileCache(args.targets.resolve(), maximum_boards=2)
    local = _run_panel(
        panel="local32",
        output_dir=output_dir,
        ranker=ranker,
        ranker_path=ranker_path,
        lookup=lookup,
        cache=cache,
        smoke_one=bool(args.smoke_one),
    )
    local_delta = local["summary"]["five_minus_four"]["satisfied_adjacent_pairs"][
        "mean"
    ]
    local_gate = not args.smoke_one and local_delta >= 0.0
    held: dict[str, Any] = {"status": "skipped_by_local_gate"}
    fresh: dict[str, Any] = {"status": "skipped_by_local_or_held_gate"}
    if local_gate:
        held = _run_panel(
            panel="held32",
            output_dir=output_dir,
            ranker=ranker,
            ranker_path=ranker_path,
            lookup=lookup,
            cache=cache,
        )
        held_delta = held["summary"]["five_minus_four"][
            "satisfied_adjacent_pairs"
        ]["mean"]
        if held_delta >= FRESH_HELD_PAIR_DELTA_GATE:
            fresh = _run_panel(
                panel="fresh32",
                output_dir=output_dir,
                ranker=ranker,
                ranker_path=ranker_path,
                lookup=lookup,
                cache=cache,
            )
        else:
            fresh = {"status": "skipped_by_held_material_pair_gate"}
    report = {
        "schema": REPORT_SCHEMA,
        "protocol": {
            "single_fixed_pairwise_arm": True,
            "train_board_count": TRAIN_BOARD_COUNT,
            "local_gate": "five-arm-tail96 minus four-arm-tail96 pairs >= 0",
            "held_opened_only_after_local_gate": True,
            "fresh_held_pair_delta_gate": FRESH_HELD_PAIR_DELTA_GATE,
            "fresh_opened_only_after_material_held_pair_gain": True,
            "no_parameter_or_model_sweep": True,
            "matcher_rerun": False,
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
            "original_costs_retained_for_placement_selection_and_fill": True,
            "all_outputs_strict_original_upright_tile_permutations": True,
            "competition_test_accessed": False,
            "restored_pixels_emitted": False,
        },
        "artifacts": {
            "pairwise_ranker": _record(ranker_path),
            "recovered_focal_checkpoint": _record(
                TaskaPairArtifactPaths().focal_verifier
            ),
            "train_edge_cache": _record(TRAIN_EDGE_ARCHIVE),
            "train_focal_cache": _record(TRAIN_FOCAL_ARCHIVE),
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

#!/usr/bin/env python3
"""Gate one fixed full-resolution-denoised candidate-supply TASKA arm.

The raw/median/bilateral 12-scorer harvest, original dense matrices, and four
production layouts are loaded from SHA-frozen target-free archives.  Only four
restored-view mutual scorers are newly evaluated.  Candidate layouts are
frozen before exact synthetic references are reconstructed.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from aiijc_puzzle.layout_evaluation import evaluate_layout
from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.taska_focal_verifier import score_focal_edges
from aiijc_puzzle.taska_fullres_union_voter import (
    FULLRES_DENOISER_SHA256,
    NEW_EDGE_FOCAL_LOGIT_MINIMUM,
    RESTORED_SCORER_COUNT,
    RESTORED_SUPPORT_MINIMUM,
    accept_focal_proposals,
    compose_fullres_union_focal_arm,
    load_fullres_denoiser,
    restore_fixed_matcher_view,
    restored_mutual_scorer_sets,
    strict_layout,
    supported_absent_edges,
)
from aiijc_puzzle.taska_pair_pipeline import (
    ARM_NAMES,
    TaskaPairArtifactPaths,
    load_taska_pair_pipeline_resources,
)

try:
    from scripts import run_taska_focal_current_finetune as finetune
except ModuleNotFoundError:  # Direct ``python scripts/run_*.py`` execution.
    import run_taska_focal_current_finetune as finetune

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/taska-fullres-union-voter/fixed-v1"
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
FULLRES_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/fullres-boundary-denoiser/pilot-train32-s400-eval16-auto/"
    "fullres_boundary_denoiser.pt"
)
FULLRES_REPORT = (
    PROJECT_ROOT
    / "outputs/fullres-boundary-denoiser/pilot-train32-s400-eval16-auto/report.json"
)

LOCAL_LAYOUT_ARCHIVE = (
    PROJECT_ROOT
    / "outputs/taska-focal-feature-stacker/train96-v1/local32/"
    "frozen-target-free-eval.npz"
)
LOCAL_LAYOUT_METADATA = LOCAL_LAYOUT_ARCHIVE.with_suffix(".json")
HELD_LAYOUT_ARCHIVE = (
    PROJECT_ROOT
    / "outputs/taska-focal-feature-stacker/train96-v1/held32/"
    "frozen-target-free-eval.npz"
)
HELD_LAYOUT_METADATA = HELD_LAYOUT_ARCHIVE.with_suffix(".json")
FRESH_LAYOUT_ARCHIVE = (
    PROJECT_ROOT
    / "outputs/taska-focal-feature-stacker/train96-v1/fresh32-exact-override/"
    "frozen-target-free-eval.npz"
)
FRESH_LAYOUT_METADATA = FRESH_LAYOUT_ARCHIVE.with_suffix(".json")

HELD_BASE_ARCHIVE = (
    PROJECT_ROOT
    / "outputs/taska-seam-replay/held300-diagnostic-mps-v1/"
    "frozen-target-free-eval.npz"
)
HELD_BASE_METADATA = HELD_BASE_ARCHIVE.with_suffix(".json")
HELD_FOCAL_ARCHIVE = (
    PROJECT_ROOT
    / "outputs/taska-focal-verifier/held300-train-exact-top5-cpu-v2/"
    "frozen-target-free-eval.npz"
)
HELD_FOCAL_METADATA = HELD_FOCAL_ARCHIVE.with_suffix(".json")
FRESH_BASE_ARCHIVE = (
    PROJECT_ROOT
    / "outputs/taska-protected-tail/fresh-held32-mps-v1/"
    "frozen-target-free-eval.npz"
)
FRESH_BASE_METADATA = FRESH_BASE_ARCHIVE.with_suffix(".json")
FRESH_FOCAL_ARCHIVE = (
    PROJECT_ROOT
    / "outputs/taska-fresh32-leader-confirmation/fresh-held32-mps-v1/"
    "frozen-target-free-eval.npz"
)
FRESH_FOCAL_METADATA = FRESH_FOCAL_ARCHIVE.with_suffix(".json")

EXPECTED_SHA256 = {
    FULLRES_CHECKPOINT: FULLRES_DENOISER_SHA256,
    LOCAL_LAYOUT_ARCHIVE: "e615659915d6bb17710403833b46b78d64916e4910afd7effa834a9f46d98e27",
    LOCAL_LAYOUT_METADATA: "78a10b5f3581ef89a1dadae284906cd6f85ec7708250154b4ef4b819cd01a62c",
    HELD_LAYOUT_ARCHIVE: "784a459f6baaaa2d16fd5e1c269c0a50ff456b8c7071ba1cf6035e20be6808f1",
    HELD_LAYOUT_METADATA: "cbe6a774fb3fd3a6095e0124c5f484736ede963c0f6682d93ac49d67fe7c384a",
    FRESH_LAYOUT_ARCHIVE: "61d166fdd5ef275ae0e790951b7d07bb174d66eadbcd4c3b25869a0a587868d2",
    FRESH_LAYOUT_METADATA: "024be2cc842c2c4e7aec8df0a3d10d5f8ea185011f825a9bdb69580f7ee797fb",
    HELD_BASE_ARCHIVE: "0876d7acd6be7ebe863f831de568586c837da52cc603df4ff8d7c5a6b0d441df",
    HELD_BASE_METADATA: "91710486233f45bda2b8aab019c508d1e6a0f75e282a6ce9a47aee93d9bf0a8d",
    HELD_FOCAL_ARCHIVE: "7d4ad494ab572d1ac3c94ab73a49b54e80b26baba489dfbd56f732a5c43394c5",
    HELD_FOCAL_METADATA: "301ba535f04b63ff8da48a0a83b5f207521d4b57f1bdbb61ceb58dbee57daff2",
    FRESH_BASE_ARCHIVE: "d7b156ff1a8cdab702881242e48797b1a18f750a2d6a60f2a7d769dbfa1bffc1",
    FRESH_BASE_METADATA: "1acb5d0000dd76e48fb6c079827fa2113bb56f541905fc97ced9656b8d7fe53f",
    FRESH_FOCAL_ARCHIVE: "f3710cc3b00aaf2e75cb4127c280bc95eeeedf237f51a76ca234bac079c6f75f",
    FRESH_FOCAL_METADATA: "311a1b3dc42bfb317a2c5cde1cee319de86ceba85622cb376fe4bfb83e2b53b1",
}

GRID = 24
COUNT = GRID * GRID
PAIR_DENOMINATOR = 2 * GRID * (GRID - 1)
FOCAL_MODE = "train_exact_top5"
LOCAL_CASES = 32
HELD_CASES = 32
FRESH_CASES = 32
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 2_026_083_180
SCORED_ARMS = ("fullres_union_focal", "control_tail96", "five_arm_tail96")
REPORT_SCHEMA = "aiijc-taska-fullres-union-voter-report-v1"


@dataclass(frozen=True)
class PanelSpec:
    name: str
    case_count: int
    layout_archive: Path
    layout_metadata: Path
    base_archive: Path
    base_metadata: Path
    focal_archive: Path
    focal_metadata: Path


PANELS = {
    "local32": PanelSpec(
        "local32",
        LOCAL_CASES,
        LOCAL_LAYOUT_ARCHIVE,
        LOCAL_LAYOUT_METADATA,
        LOCAL_LAYOUT_ARCHIVE,
        LOCAL_LAYOUT_METADATA,
        LOCAL_LAYOUT_ARCHIVE,
        LOCAL_LAYOUT_METADATA,
    ),
    "held32": PanelSpec(
        "held32",
        HELD_CASES,
        HELD_LAYOUT_ARCHIVE,
        HELD_LAYOUT_METADATA,
        HELD_BASE_ARCHIVE,
        HELD_BASE_METADATA,
        HELD_FOCAL_ARCHIVE,
        HELD_FOCAL_METADATA,
    ),
    "fresh32": PanelSpec(
        "fresh32",
        FRESH_CASES,
        FRESH_LAYOUT_ARCHIVE,
        FRESH_LAYOUT_METADATA,
        FRESH_BASE_ARCHIVE,
        FRESH_BASE_METADATA,
        FRESH_FOCAL_ARCHIVE,
        FRESH_FOCAL_METADATA,
    ),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--inference-batch", type=int, default=576)
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


def _require_inputs() -> None:
    for path, expected in EXPECTED_SHA256.items():
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"frozen input SHA-256 mismatch: {path}")


def _rows(path: Path, case_count: int) -> list[Mapping[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    if len(rows) < case_count:
        raise ValueError(f"{path} has fewer than {case_count} rows")
    return rows[:case_count]


def _aligned_rows(spec: PanelSpec) -> list[tuple[Mapping[str, Any], ...]]:
    rosters = [
        _rows(spec.layout_metadata, spec.case_count),
        _rows(spec.base_metadata, spec.case_count),
        _rows(spec.focal_metadata, spec.case_count),
    ]
    result: list[tuple[Mapping[str, Any], ...]] = []
    identity = ("prefix", "source_filename", "draw_index", "dirty_sha256")
    for records in zip(*rosters, strict=True):
        if any(
            records[0][field] != record[field]
            for record in records[1:]
            for field in identity
        ):
            raise RuntimeError(f"{spec.name} frozen input rosters are misaligned")
        result.append(records)
    return result


def _edges(archive: Any, prefix: str) -> tuple[RawTailEdge, ...]:
    source = np.asarray(archive[f"{prefix}__edge_source"], dtype=np.int64)
    target = np.asarray(archive[f"{prefix}__edge_target"], dtype=np.int64)
    axis = np.asarray(archive[f"{prefix}__edge_axis"], dtype=np.uint8)
    if not (source.ndim == target.ndim == axis.ndim == 1):
        raise ValueError("edge arrays must be vectors")
    if not (len(source) == len(target) == len(axis)) or not np.isin(axis, (0, 1)).all():
        raise ValueError("edge arrays are malformed")
    edges = tuple(
        RawTailEdge(int(s), int(t), "right" if int(a) == 0 else "down")
        for s, t, a in zip(source, target, axis, strict=True)
    )
    if len(set(edges)) != len(edges):
        raise ValueError("frozen current candidate edges contain duplicates")
    return edges


def _matrix(archive: Any, key: str) -> np.ndarray:
    value = np.asarray(archive[key], dtype=np.float64)
    if value.shape != (COUNT, COUNT) or not np.isfinite(value).all():
        raise ValueError(f"{key} is not one finite 576x576 matrix")
    return np.ascontiguousarray(value)


def _four_layouts(archive: Any, prefix: str) -> dict[str, np.ndarray]:
    values = {
        "raw": strict_layout(archive[f"{prefix}__raw_layout"]),
        "logistic": strict_layout(archive[f"{prefix}__logistic_layout"]),
        "focal_top5": strict_layout(archive[f"{prefix}__focal_top5_layout"]),
        "nonlinear": strict_layout(archive[f"{prefix}__nonlinear_layout"]),
    }
    if tuple(values) != ARM_NAMES:
        raise RuntimeError("production four-arm order changed")
    return values


def _truth_edges(reference: Any) -> frozenset[RawTailEdge]:
    layout = strict_layout(reference).reshape(GRID, GRID)
    result: set[RawTailEdge] = set()
    for row in range(GRID):
        for column in range(GRID - 1):
            result.add(
                RawTailEdge(
                    int(layout[row, column]), int(layout[row, column + 1]), "right"
                )
            )
    for row in range(GRID - 1):
        for column in range(GRID):
            result.add(
                RawTailEdge(
                    int(layout[row, column]), int(layout[row + 1, column]), "down"
                )
            )
    if len(result) != PAIR_DENOMINATOR:
        raise RuntimeError("truth pair denominator changed")
    return frozenset(result)


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
    means = np.asarray([np.mean(grouped[name]) for name in sorted(grouped)])
    generator = np.random.default_rng(seed)
    distribution = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_RESAMPLES, 2048):
        stop = min(start + 2048, BOOTSTRAP_RESAMPLES)
        indices = generator.integers(0, len(means), size=(stop - start, len(means)))
        distribution[start:stop] = means[indices].mean(axis=1)
    return {
        "mean": float(np.mean(values)),
        "ci95_lower": float(np.quantile(distribution, 0.025)),
        "ci95_upper": float(np.quantile(distribution, 0.975)),
        "source_count": len(means),
        "case_count": len(values),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": seed,
    }


def _summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = ("satisfied_adjacent_pairs", "adjacency_recall", "exact_tiles")
    result: dict[str, Any] = {
        "case_count": len(rows),
        "pair_denominator": PAIR_DENOMINATOR,
        "arms": {
            arm: {
                metric: float(np.mean([row["metrics"][arm][metric] for row in rows]))
                for metric in metrics
            }
            for arm in SCORED_ARMS
        },
        "five_arm_choice_counts": dict(Counter(row["five_arm_choice"] for row in rows)),
    }
    sources = [str(row["source_filename"]) for row in rows]
    deltas: dict[str, Any] = {}
    for index, metric in enumerate(metrics):
        values = [
            float(row["metrics"]["five_arm_tail96"][metric])
            - float(row["metrics"]["control_tail96"][metric])
            for row in rows
        ]
        current = _cluster_ci(values, sources, seed=BOOTSTRAP_SEED + index)
        current["case_wins_ties_losses"] = {
            "wins": sum(value > 0 for value in values),
            "ties": sum(value == 0 for value in values),
            "losses": sum(value < 0 for value in values),
        }
        deltas[metric] = current
    result["five_minus_control"] = deltas
    supply_fields = (
        "current_true_edges",
        "proposed_absent_edges",
        "proposed_absent_true_edges",
        "accepted_new_edges",
        "accepted_new_true_edges",
        "union_true_edges",
    )
    result["candidate_supply_mean_per_board"] = {
        field: float(np.mean([row["candidate_supply"][field] for row in rows]))
        for field in supply_fields
    }
    result["candidate_supply"] = {
        "current_recall": float(
            np.mean([row["candidate_supply"]["current_true_edges"] for row in rows])
            / PAIR_DENOMINATOR
        ),
        "union_recall": float(
            np.mean([row["candidate_supply"]["union_true_edges"] for row in rows])
            / PAIR_DENOMINATOR
        ),
        "proposed_absent_precision": float(
            sum(row["candidate_supply"]["proposed_absent_true_edges"] for row in rows)
            / max(1, sum(row["candidate_supply"]["proposed_absent_edges"] for row in rows))
        ),
        "accepted_new_precision": float(
            sum(row["candidate_supply"]["accepted_new_true_edges"] for row in rows)
            / max(1, sum(row["candidate_supply"]["accepted_new_edges"] for row in rows))
        ),
    }
    return result


def _freeze(
    spec: PanelSpec,
    output_dir: Path,
    arrays: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path, Path]:
    stage = output_dir / spec.name
    stage.mkdir(parents=True, exist_ok=False)
    archive = stage / "frozen-target-free-eval.npz"
    metadata = stage / "frozen-target-free-eval.json"
    freeze = stage / "pre-score-freeze.json"
    _write_npz(archive, arrays)
    _write_json(
        metadata,
        {
            "schema": "aiijc-taska-fullres-union-voter-target-free-v1",
            "stage": spec.name,
            "contains_exact_references_or_candidate_labels": False,
            "restored_pixels_matcher_only": True,
            "raw_dense_cost_matrices_reused_unchanged": True,
            "all_layouts_strict_original_upright_permutations": True,
            "rows": list(rows),
        },
    )
    artifacts = {
        "frozen_archive": archive,
        "frozen_metadata": metadata,
        "denoiser": FULLRES_CHECKPOINT,
        "runner": Path(__file__).resolve(),
        "voter_module": PROJECT_ROOT / "src/aiijc_puzzle/taska_fullres_union_voter.py",
        "raw_solver": PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py",
        "layout_archive": spec.layout_archive,
        "layout_metadata": spec.layout_metadata,
        "base_archive": spec.base_archive,
        "base_metadata": spec.base_metadata,
        "focal_archive": spec.focal_archive,
        "focal_metadata": spec.focal_metadata,
    }
    _write_json(
        freeze,
        {
            "schema": "aiijc-taska-fullres-union-voter-pre-score-freeze-v1",
            "created_before_exact_reference_reconstruction": True,
            "contains_evaluation_references_or_labels": False,
            "artifacts": {name: _record(path) for name, path in artifacts.items()},
        },
    )
    return archive, metadata, freeze


def _validate_freeze(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("created_before_exact_reference_reconstruction") is not True:
        raise RuntimeError("pre-score timing contract differs")
    for name, record in payload["artifacts"].items():
        artifact = Path(record["path"])
        if not artifact.is_absolute():
            artifact = PROJECT_ROOT / artifact
        if sha256_file(artifact) != record["sha256"]:
            raise RuntimeError(f"frozen artifact changed before scoring: {name}")


def _score_panel(
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
            truth = _truth_edges(reference)
            current = set(_edges(candidate, f"{prefix}__current"))
            proposals = set(_edges(candidate, f"{prefix}__proposed"))
            accepted = set(_edges(candidate, f"{prefix}__accepted"))
            union = current | accepted
            metrics = {
                arm: _layout_metrics(candidate[f"{prefix}__{arm}_layout"], reference)
                for arm in SCORED_ARMS
            }
            scored.append(
                {
                    "prefix": prefix,
                    "source_filename": source,
                    "draw_index": draw,
                    "five_arm_choice": row["five_arm_choice"],
                    "metrics": metrics,
                    "candidate_supply": {
                        "current_edges": len(current),
                        "current_true_edges": len(current & truth),
                        "proposed_absent_edges": len(proposals),
                        "proposed_absent_true_edges": len(proposals & truth),
                        "accepted_new_edges": len(accepted),
                        "accepted_new_true_edges": len(accepted & truth),
                        "union_edges": len(union),
                        "union_true_edges": len(union & truth),
                    },
                }
            )
    return scored, _summarize(scored)


def _edge_arrays(prefix: str, name: str, edges: Sequence[RawTailEdge]) -> dict[str, np.ndarray]:
    return {
        f"{prefix}__{name}__edge_source": np.asarray(
            [edge.source for edge in edges], dtype=np.int32
        ),
        f"{prefix}__{name}__edge_target": np.asarray(
            [edge.target for edge in edges], dtype=np.int32
        ),
        f"{prefix}__{name}__edge_axis": np.asarray(
            [edge.axis == "down" for edge in edges], dtype=np.uint8
        ),
    }


def _run_panel(
    spec: PanelSpec,
    *,
    output_dir: Path,
    resources: Any,
    denoiser: Any,
    lookup: Mapping[str, Mapping[str, Any]],
    cache: Any,
    inference_batch: int,
    target_free_only: bool,
) -> dict[str, Any]:
    aligned = _aligned_rows(spec)
    arrays: dict[str, np.ndarray] = {}
    frozen_rows: list[dict[str, Any]] = []
    started = perf_counter()
    with (
        np.load(spec.layout_archive, allow_pickle=False) as layout_archive,
        np.load(spec.base_archive, allow_pickle=False) as base_archive,
        np.load(spec.focal_archive, allow_pickle=False) as focal_archive,
    ):
        for index, records in enumerate(aligned):
            row = records[0]
            prefix = str(row["prefix"])
            source = str(row["source_filename"])
            draw = int(row["draw_index"])
            dirty = finetune._dirty_case(cache, lookup[source], source, draw)
            dirty_sha = finetune._dirty_sha256(dirty.dirty_tiles)
            if dirty_sha != row["dirty_sha256"]:
                raise RuntimeError(f"{spec.name} recreated different dirty bytes")
            right = _matrix(base_archive, f"{prefix}__cost_right")
            down = _matrix(base_archive, f"{prefix}__cost_down")
            current = _edges(base_archive, prefix)
            old_focal = np.asarray(
                focal_archive[f"{prefix}__focal_logits"], dtype=np.float32
            )
            if old_focal.shape != (len(current),) or not np.isfinite(old_focal).all():
                raise RuntimeError("cached focal logits are not edge-aligned")
            restored = restore_fixed_matcher_view(
                denoiser,
                dirty.dirty_tiles,
                device=resources.device,
                batch_size=inference_batch,
            )
            scorer_sets = restored_mutual_scorer_sets(
                restored,
                resources.matchers,
                device=resources.device,
            )
            proposed, support = supported_absent_edges(current, scorer_sets)
            focal = score_focal_edges(
                resources.focal_verifier,
                dirty.dirty_tiles,
                right,
                down,
                proposed,
                mode=FOCAL_MODE,
                grid=GRID,
                device=resources.device,
            )
            accepted, accepted_logits = accept_focal_proposals(proposed, focal.logits)
            four = _four_layouts(layout_archive, prefix)
            composition = compose_fullres_union_focal_arm(
                cost_right=right,
                cost_down=down,
                current_edges=current,
                current_focal_logits=old_focal,
                accepted_new_edges=accepted,
                accepted_new_logits=accepted_logits,
                four_layouts=four,
                grid=GRID,
            )
            control = strict_layout(layout_archive[f"{prefix}__four_arm_tail96_layout"])
            arrays[f"{prefix}__fullres_union_focal_layout"] = composition.fullres_layout
            arrays[f"{prefix}__control_tail96_layout"] = control
            arrays[f"{prefix}__five_arm_tail96_layout"] = composition.layout
            arrays.update(_edge_arrays(prefix, "current", current))
            arrays.update(_edge_arrays(prefix, "proposed", proposed))
            arrays.update(_edge_arrays(prefix, "accepted", accepted))
            arrays[f"{prefix}__proposed_support"] = np.asarray(support, dtype=np.uint8)
            arrays[f"{prefix}__proposed_focal_logits"] = np.asarray(
                focal.logits, dtype=np.float32
            )
            frozen_rows.append(
                {
                    "prefix": prefix,
                    "source_filename": source,
                    "draw_index": draw,
                    "dirty_sha256": dirty_sha,
                    "current_candidate_count": len(current),
                    "restored_scorer_edge_counts": [len(scorer) for scorer in scorer_sets],
                    "restored_supported_absent_count": len(proposed),
                    "focal_accepted_new_count": len(accepted),
                    "five_arm_choice": composition.choice,
                    "five_arm_total_costs": dict(composition.total_costs),
                    **dict(composition.diagnostics),
                }
            )
            print(
                json.dumps(
                    {
                        "event": f"{spec.name}_fullres_target_free",
                        "case": index + 1,
                        "case_count": len(aligned),
                        "proposed": len(proposed),
                        "accepted": len(accepted),
                        "choice": composition.choice,
                    }
                ),
                flush=True,
            )
    archive, metadata, freeze = _freeze(spec, output_dir, arrays, frozen_rows)
    target_free_summary = {
        "case_count": len(frozen_rows),
        "mean_current_candidates": float(
            np.mean([row["current_candidate_count"] for row in frozen_rows])
        ),
        "mean_supported_absent": float(
            np.mean([row["restored_supported_absent_count"] for row in frozen_rows])
        ),
        "mean_focal_accepted_new": float(
            np.mean([row["focal_accepted_new_count"] for row in frozen_rows])
        ),
        "total_supported_absent": int(
            sum(row["restored_supported_absent_count"] for row in frozen_rows)
        ),
        "total_focal_accepted_new": int(
            sum(row["focal_accepted_new_count"] for row in frozen_rows)
        ),
        "five_arm_choice_counts": dict(
            Counter(row["five_arm_choice"] for row in frozen_rows)
        ),
    }
    result: dict[str, Any] = {
        "status": "target-free-smoke" if target_free_only else "complete",
        "target_free_summary": target_free_summary,
        "runtime_seconds": perf_counter() - started,
        "artifacts": {
            "archive": _record(archive),
            "metadata": _record(metadata),
            "pre_score_freeze": _record(freeze),
        },
    }
    if not target_free_only:
        rows, summary = _score_panel(
            archive=archive,
            metadata=metadata,
            freeze=freeze,
            lookup=lookup,
            cache=cache,
        )
        result.update({"rows": rows, "summary": summary})
    return result


def _validate_training_overlap() -> dict[str, Any]:
    denoiser_train = set(
        json.loads(FULLRES_REPORT.read_text(encoding="utf-8"))["selection"][
            "train_filenames"
        ]
    )
    overlaps: dict[str, list[str]] = {}
    for name, spec in PANELS.items():
        panel_names = {
            str(records[0]["source_filename"]) for records in _aligned_rows(spec)
        }
        overlaps[name] = sorted(panel_names & denoiser_train)
    if any(overlaps.values()):
        raise RuntimeError(f"denoiser training overlaps evaluation sources: {overlaps}")
    return {
        "denoiser_train_source_count": len(denoiser_train),
        "evaluation_overlap": {name: len(values) for name, values in overlaps.items()},
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    _require_inputs()
    if args.inference_batch <= 0:
        raise ValueError("inference_batch must be positive")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
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
    denoiser = load_fullres_denoiser(FULLRES_CHECKPOINT, device=resources.device)
    config, _, _ = finetune._load_config(finetune.DEFAULT_CONFIG)
    lookup = finetune._manifest_lookup(config)
    cache = finetune.CleanTileCache(args.targets.resolve(), maximum_boards=2)
    overlap = _validate_training_overlap()

    if args.smoke_one:
        smoke_spec = PanelSpec(
            "smoke1",
            1,
            LOCAL_LAYOUT_ARCHIVE,
            LOCAL_LAYOUT_METADATA,
            LOCAL_LAYOUT_ARCHIVE,
            LOCAL_LAYOUT_METADATA,
            LOCAL_LAYOUT_ARCHIVE,
            LOCAL_LAYOUT_METADATA,
        )
        local = _run_panel(
            smoke_spec,
            output_dir=output_dir,
            resources=resources,
            denoiser=denoiser,
            lookup=lookup,
            cache=cache,
            inference_batch=args.inference_batch,
            target_free_only=True,
        )
        report = {
            "schema": REPORT_SCHEMA,
            "status": "target-free-smoke",
            "local": local,
            "training_overlap": overlap,
            "competition_test_accessed": False,
        }
        _write_json(output_dir / "report.json", report)
        print(json.dumps(report, indent=2))
        return report

    local = _run_panel(
        PANELS["local32"],
        output_dir=output_dir,
        resources=resources,
        denoiser=denoiser,
        lookup=lookup,
        cache=cache,
        inference_batch=args.inference_batch,
        target_free_only=False,
    )
    local_delta = local["summary"]["five_minus_control"][
        "satisfied_adjacent_pairs"
    ]["mean"]
    held: dict[str, Any] = {"status": "skipped_by_negative_local_pair_gate"}
    fresh: dict[str, Any] = {"status": "skipped_by_local_or_held_pair_gate"}
    if local_delta >= 0:
        held = _run_panel(
            PANELS["held32"],
            output_dir=output_dir,
            resources=resources,
            denoiser=denoiser,
            lookup=lookup,
            cache=cache,
            inference_batch=args.inference_batch,
            target_free_only=False,
        )
        held_delta = held["summary"]["five_minus_control"][
            "satisfied_adjacent_pairs"
        ]["mean"]
        if held_delta >= 0.5:
            fresh = _run_panel(
                PANELS["fresh32"],
                output_dir=output_dir,
                resources=resources,
                denoiser=denoiser,
                lookup=lookup,
                cache=cache,
                inference_batch=args.inference_batch,
                target_free_only=False,
            )
        else:
            fresh = {"status": "skipped_by_held_pair_delta_below_0.5"}
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "protocol": {
            "single_fixed_candidate_supply_arm": True,
            "current_12_scorer_harvest_reused_unchanged": True,
            "raw_dense_cost_matrices_reused_unchanged": True,
            "restored_scorers": "v3/local x first two audited orientations",
            "restored_scorer_count": RESTORED_SCORER_COUNT,
            "new_edge_support_minimum": RESTORED_SUPPORT_MINIMUM,
            "new_edge_focal_logit_minimum": NEW_EDGE_FOCAL_LOGIT_MINIMUM,
            "local_gate": "five-arm-tail96 pair delta >= 0",
            "held_gate": "five-arm-tail96 pair delta >= +0.5",
            "local_cases": LOCAL_CASES,
            "held_cases": HELD_CASES,
            "fresh_cases": FRESH_CASES,
            "no_threshold_or_orientation_sweep": True,
        },
        "training_overlap": overlap,
        "local32": local,
        "held32": held,
        "fresh32": fresh,
        "runtime_seconds": perf_counter() - started,
        "legality": {
            "restored_pixels_matcher_only": True,
            "all_outputs_strict_original_upright_tile_permutations": True,
            "targets_used_only_after_candidate_freeze": True,
            "competition_test_accessed": False,
            "postprocessing_used": False,
        },
        "artifacts": {
            "fullres_denoiser": _record(FULLRES_CHECKPOINT),
            "focal_verifier": _record(TaskaPairArtifactPaths().focal_verifier),
            "runner": _record(Path(__file__).resolve()),
            "voter_module": _record(
                PROJECT_ROOT / "src/aiijc_puzzle/taska_fullres_union_voter.py"
            ),
            "raw_solver": _record(
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

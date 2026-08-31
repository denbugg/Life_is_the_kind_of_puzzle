#!/usr/bin/env python3
"""Replay one frozen selective-target500 plus unique-fullres TASKA fusion.

No matcher is rerun.  Frozen accepted supplies are identity-aligned, the
selective final layout is mechanically replayed as the control, and exactly
one combined-union arm is added to the current-four plus selective selector.
Candidate layouts are SHA-frozen before organizer-train references are opened.
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

from aiijc_puzzle.layout_evaluation import evaluate_layout
from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.taska_fullres_union_voter import (
    NEW_EDGE_FOCAL_LOGIT_MINIMUM,
    RESTORED_SCORER_COUNT,
    RESTORED_SUPPORT_MINIMUM,
    accept_focal_proposals,
)
from aiijc_puzzle.taska_pair_pipeline import (
    ARM_NAMES,
    PAIR_DENOMINATOR,
    RAW_TAIL_GLOBAL_SOLVER_SHA256,
)
from aiijc_puzzle.taska_selective_fullres_fusion import (
    FUSION_ARM_NAMES,
    compose_selective_fullres_fusion,
    strict_layout,
)

try:
    from scripts import run_taska_focal_current_finetune as finetune
except ModuleNotFoundError:  # Direct ``python scripts/run_*.py`` execution.
    import run_taska_focal_current_finetune as finetune

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/taska-selective-fullres-union-fusion/fixed-v1"
GRID = 24
COUNT = GRID * GRID
LOCAL_GATE = 0.0
HELD_GATE = 0.5
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 2_026_083_195
REPORT_SCHEMA = "aiijc-taska-selective-fullres-union-fusion-report-v1"
SCORED_ARMS = ("selective_target500_control", "combined_union_candidate")


@dataclass(frozen=True)
class PanelSpec:
    name: str
    case_count: int
    layout_archive: Path
    layout_metadata: Path
    base_archive: Path
    base_metadata: Path
    selective_archive: Path
    selective_metadata: Path
    selective_freeze: Path
    fullres_archive: Path
    fullres_metadata: Path
    fullres_freeze: Path


LAYOUT_ROOT = PROJECT_ROOT / "outputs/taska-focal-feature-stacker/train96-v1"
SELECTIVE_ROOT = PROJECT_ROOT / "outputs/taska-selective-vote500/fixed-v1"
FULLRES_ROOT = PROJECT_ROOT / "outputs/taska-fullres-union-voter/fixed-v1"


def _input_triplet(root: Path, panel: str) -> tuple[Path, Path, Path]:
    archive = root / panel / "frozen-target-free-eval.npz"
    return archive, archive.with_suffix(".json"), archive.parent / "pre-score-freeze.json"


def _panel(
    name: str,
    layout_panel: str,
    base_root: Path,
) -> PanelSpec:
    layout = LAYOUT_ROOT / layout_panel / "frozen-target-free-eval.npz"
    base = base_root / "frozen-target-free-eval.npz"
    selective = _input_triplet(SELECTIVE_ROOT, name)
    fullres = _input_triplet(FULLRES_ROOT, name)
    return PanelSpec(
        name=name,
        case_count=32,
        layout_archive=layout,
        layout_metadata=layout.with_suffix(".json"),
        base_archive=base,
        base_metadata=base.with_suffix(".json"),
        selective_archive=selective[0],
        selective_metadata=selective[1],
        selective_freeze=selective[2],
        fullres_archive=fullres[0],
        fullres_metadata=fullres[1],
        fullres_freeze=fullres[2],
    )


PANELS = {
    "local32": _panel("local32", "local32", LAYOUT_ROOT / "local32"),
    "held32": _panel(
        "held32",
        "held32",
        PROJECT_ROOT / "outputs/taska-seam-replay/held300-diagnostic-mps-v1",
    ),
    "fresh32": _panel(
        "fresh32",
        "fresh32-exact-override",
        PROJECT_ROOT / "outputs/taska-protected-tail/fresh-held32-mps-v1",
    ),
}

_PARENT_SHA256 = {
    "local32": {
        "selective": (
            "fd8977682f7aec1d57bb1afa619fe94d1fe2d61268eb118ae563d5a1df42c2ba",
            "551c42efae81e1fbfb6ffb1523b75aaffd12b5717ffef0acab6bf5267c30ac2c",
            "ff3abdc1014e7b261f83d34290222b4616f2b6c9a6e63c8ffa65d2ad82cf89e6",
        ),
        "fullres": (
            "17dd26e11fbbaf8d79d66c122a1ca7abfbe6e34d6c9a149ed38420381845946d",
            "e5574bbea6ff83bb5caded5152d14ed22d56e9043229ee577cff7bda65f9ea20",
            "dfce253658ff99322cc0c3a08629602f805873def8be23b390e68eea4d7add65",
        ),
    },
    "held32": {
        "selective": (
            "d3ec0c4706cf3db8a704f81e7da5d13663c736f5be5c7986d88c5b6561f50b7c",
            "1eb4b1c47fb0451236aa36c09d37aed7e74fb05a2fc73e80cad700fc18abd870",
            "2e938e39a035fa6ded367293f5a68fb1c7df038cc57de03a73b4252917fb6b00",
        ),
        "fullres": (
            "a6eef323aecacdf6285b5095d004b0c3f141e1a5ab1958cc8639e7c964f4c48e",
            "116b5e77bc01329ffab0f850b4aaa79194d79a0ac6239f0243142c9752724b13",
            "0f3f44e0bd7d36ba49ed19a469d10b7b5ecda9ed4871dd60cec326e048c2deff",
        ),
    },
    "fresh32": {
        "selective": (
            "e04c9350403413c2906df0d2183ad1d2a9cc19c9b48a16a5a46be9360265ea57",
            "63e2b73fcba695ac5078784562dd9374b1e49ae6bf0a604cb7558632923ad55a",
            "025a6aa6b66ae23527db35bc777d404c3a01ba4671a8a3d09f692ab3c8b493a5",
        ),
        "fullres": (
            "06805de4cd0d76b3007f0af620e862d1b062413c9629861d0be1825e1e538af0",
            "af8e1a398bbdbb069870e5224857e758720eb252f2314bd9cf0c527d351c5f3f",
            "69fdd637e7684398a7938442bfb8d808f225c56b3dbe95a69250a3125c7d6547",
        ),
    },
}
_LAYOUT_SHA256 = {
    "local32": (
        "e615659915d6bb17710403833b46b78d64916e4910afd7effa834a9f46d98e27",
        "78a10b5f3581ef89a1dadae284906cd6f85ec7708250154b4ef4b819cd01a62c",
    ),
    "held32": (
        "784a459f6baaaa2d16fd5e1c269c0a50ff456b8c7071ba1cf6035e20be6808f1",
        "cbe6a774fb3fd3a6095e0124c5f484736ede963c0f6682d93ac49d67fe7c384a",
    ),
    "fresh32": (
        "61d166fdd5ef275ae0e790951b7d07bb174d66eadbcd4c3b25869a0a587868d2",
        "024be2cc842c2c4e7aec8df0a3d10d5f8ea185011f825a9bdb69580f7ee797fb",
    ),
}
_BASE_SHA256 = {
    "held32": (
        "0876d7acd6be7ebe863f831de568586c837da52cc603df4ff8d7c5a6b0d441df",
        "91710486233f45bda2b8aab019c508d1e6a0f75e282a6ce9a47aee93d9bf0a8d",
    ),
    "fresh32": (
        "d7b156ff1a8cdab702881242e48797b1a18f750a2d6a60f2a7d769dbfa1bffc1",
        "1acb5d0000dd76e48fb6c079827fa2113bb56f541905fc97ced9656b8d7fe53f",
    ),
}
EXPECTED_SHA256: dict[Path, str] = {
    PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py": RAW_TAIL_GLOBAL_SOLVER_SHA256,
}
for _panel_name, _spec in PANELS.items():
    for _kind in ("selective", "fullres"):
        _paths = (
            getattr(_spec, f"{_kind}_archive"),
            getattr(_spec, f"{_kind}_metadata"),
            getattr(_spec, f"{_kind}_freeze"),
        )
        EXPECTED_SHA256.update(dict(zip(_paths, _PARENT_SHA256[_panel_name][_kind], strict=True)))
    EXPECTED_SHA256[_spec.layout_archive] = _LAYOUT_SHA256[_panel_name][0]
    EXPECTED_SHA256[_spec.layout_metadata] = _LAYOUT_SHA256[_panel_name][1]
    if _panel_name in _BASE_SHA256:
        EXPECTED_SHA256[_spec.base_archive] = _BASE_SHA256[_panel_name][0]
        EXPECTED_SHA256[_spec.base_metadata] = _BASE_SHA256[_panel_name][1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--smoke-one",
        action="store_true",
        help="freeze and replay one local case without reconstructing its reference",
    )
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


def _validate_parent_freeze(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("created_before_exact_reference_reconstruction") is not True:
        raise ValueError(f"parent freeze was not created before scoring: {path}")
    if payload.get("contains_evaluation_references_or_labels") is not False:
        raise ValueError(f"parent freeze contains labels: {path}")


def _require_inputs() -> None:
    for path, expected in EXPECTED_SHA256.items():
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"frozen input SHA-256 mismatch: {path}")
    for spec in PANELS.values():
        _validate_parent_freeze(spec.selective_freeze)
        _validate_parent_freeze(spec.fullres_freeze)


def _rows(path: Path, case_count: int) -> list[Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) < case_count:
        raise ValueError(f"{path} has fewer than {case_count} rows")
    return rows[:case_count]


def _aligned_rows(spec: PanelSpec) -> list[tuple[Mapping[str, Any], ...]]:
    rosters = [
        _rows(spec.layout_metadata, spec.case_count),
        _rows(spec.base_metadata, spec.case_count),
        _rows(spec.selective_metadata, spec.case_count),
        _rows(spec.fullres_metadata, spec.case_count),
    ]
    identity = ("prefix", "source_filename", "draw_index", "dirty_sha256")
    aligned: list[tuple[Mapping[str, Any], ...]] = []
    for records in zip(*rosters, strict=True):
        if any(
            records[0].get(field) != record.get(field)
            for record in records[1:]
            for field in identity
        ):
            raise RuntimeError(f"{spec.name} frozen row identity mismatch")
        aligned.append(records)
    return aligned


def _edge_key(prefix: str, name: str, field: str) -> str:
    middle = f"{name}__" if name else ""
    return f"{prefix}__{middle}edge_{field}"


def _edges(archive: Any, prefix: str, name: str = "") -> tuple[RawTailEdge, ...]:
    source = np.asarray(archive[_edge_key(prefix, name, "source")], dtype=np.int64)
    target = np.asarray(archive[_edge_key(prefix, name, "target")], dtype=np.int64)
    axis = np.asarray(archive[_edge_key(prefix, name, "axis")], dtype=np.uint8)
    if not (source.ndim == target.ndim == axis.ndim == 1):
        raise ValueError("edge arrays must be vectors")
    if not (len(source) == len(target) == len(axis)) or not np.isin(axis, (0, 1)).all():
        raise ValueError("edge arrays are malformed")
    result = tuple(
        RawTailEdge(int(s), int(t), "down" if int(a) else "right")
        for s, t, a in zip(source, target, axis, strict=True)
    )
    if len(set(result)) != len(result):
        raise ValueError("edge arrays contain duplicates")
    return result


def _edge_arrays(prefix: str, name: str, edges: Sequence[RawTailEdge]) -> dict[str, np.ndarray]:
    return {
        _edge_key(prefix, name, "source"): np.asarray(
            [edge.source for edge in edges], dtype=np.int16
        ),
        _edge_key(prefix, name, "target"): np.asarray(
            [edge.target for edge in edges], dtype=np.int16
        ),
        _edge_key(prefix, name, "axis"): np.asarray(
            [edge.axis == "down" for edge in edges], dtype=np.uint8
        ),
    }


def _matrix(archive: Any, key: str) -> np.ndarray:
    value = np.asarray(archive[key], dtype=np.float64)
    if value.shape != (COUNT, COUNT) or not np.isfinite(value).all():
        raise ValueError(f"{key} is not one finite 576x576 matrix")
    return np.ascontiguousarray(value)


def _four_layouts(archive: Any, prefix: str) -> dict[str, np.ndarray]:
    result = {name: strict_layout(archive[f"{prefix}__{name}_layout"]) for name in ARM_NAMES}
    if tuple(result) != ARM_NAMES:
        raise RuntimeError("current four-arm order changed")
    return result


def _fullres_accepted_with_logits(
    archive: Any, prefix: str
) -> tuple[tuple[RawTailEdge, ...], np.ndarray]:
    proposed = _edges(archive, prefix, "proposed")
    frozen_accepted = _edges(archive, prefix, "accepted")
    support = np.asarray(archive[f"{prefix}__proposed_support"], dtype=np.uint8)
    logits = np.asarray(archive[f"{prefix}__proposed_focal_logits"], dtype=np.float32)
    if support.shape != (len(proposed),) or np.any(support < RESTORED_SUPPORT_MINIMUM):
        raise RuntimeError("fullres frozen proposal support contract changed")
    accepted, accepted_logits = accept_focal_proposals(proposed, logits)
    if accepted != frozen_accepted or np.any(accepted_logits < NEW_EDGE_FOCAL_LOGIT_MINIMUM):
        raise RuntimeError("fullres frozen accepted-edge/logit alignment mismatch")
    return accepted, accepted_logits


def _truth_edges(reference: Any) -> frozenset[RawTailEdge]:
    layout = strict_layout(reference).reshape(GRID, GRID)
    result = {
        *(
            RawTailEdge(int(layout[r, c]), int(layout[r, c + 1]), "right")
            for r in range(GRID)
            for c in range(GRID - 1)
        ),
        *(
            RawTailEdge(int(layout[r, c]), int(layout[r + 1, c]), "down")
            for r in range(GRID - 1)
            for c in range(GRID)
        ),
    }
    if len(result) != PAIR_DENOMINATOR:
        raise RuntimeError("truth pair denominator changed")
    return frozenset(result)


def _layout_metrics(layout: Any, reference: Any) -> dict[str, Any]:
    result = evaluate_layout(layout, reference, reference_is_exact=True)
    if result.adjacency_total != PAIR_DENOMINATOR:
        raise RuntimeError("pair denominator changed")
    return {
        "satisfied_adjacent_pairs": int(result.adjacency_correct),
        "adjacency_recall": float(result.adjacency),
        "exact_tiles": int(result.correct_tile_count),
        "strict_original_upright_permutation": True,
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
        "choice_counts": dict(Counter(row["choice"] for row in rows)),
        "control_replay_match_count": sum(
            bool(row["mechanical_control_matches_frozen"]) for row in rows
        ),
    }
    sources = [str(row["source_filename"]) for row in rows]
    deltas: dict[str, Any] = {}
    for index, metric in enumerate(metrics):
        values = [
            float(row["metrics"]["combined_union_candidate"][metric])
            - float(row["metrics"]["selective_target500_control"][metric])
            for row in rows
        ]
        summary = _cluster_ci(values, sources, seed=BOOTSTRAP_SEED + index)
        summary["case_wins_ties_losses"] = {
            "wins": sum(value > 0 for value in values),
            "ties": sum(value == 0 for value in values),
            "losses": sum(value < 0 for value in values),
        }
        deltas[metric] = summary
    result["candidate_minus_control"] = deltas
    supply_fields = (
        "current_edges",
        "current_true_edges",
        "selective_new_edges",
        "selective_new_true_edges",
        "fullres_accepted_edges",
        "fullres_accepted_true_edges",
        "fullres_overlap_current_edges",
        "fullres_overlap_selective_edges",
        "fullres_overlap_selective_true_edges",
        "unique_fullres_edges",
        "unique_fullres_true_edges",
        "combined_union_edges",
        "combined_union_true_edges",
    )
    result["supply_mean_per_board"] = {
        field: float(np.mean([row["supply"][field] for row in rows])) for field in supply_fields
    }
    totals = {field: int(sum(row["supply"][field] for row in rows)) for field in supply_fields}
    result["supply_totals"] = totals
    result["supply_quality"] = {
        "unique_fullres_precision": float(
            totals["unique_fullres_true_edges"] / max(1, totals["unique_fullres_edges"])
        ),
        "overlap_selective_precision": float(
            totals["fullres_overlap_selective_true_edges"]
            / max(1, totals["fullres_overlap_selective_edges"])
        ),
        "combined_union_recall": float(
            result["supply_mean_per_board"]["combined_union_true_edges"] / PAIR_DENOMINATOR
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
            "schema": "aiijc-taska-selective-fullres-union-fusion-target-free-v1",
            "stage": spec.name,
            "contains_exact_references_or_candidate_labels": False,
            "matcher_rerun": False,
            "selector_roster": list(FUSION_ARM_NAMES),
            "standalone_fullres_arm_in_selector": False,
            "fullres_support_minimum_in_parent": RESTORED_SUPPORT_MINIMUM,
            "fullres_scorer_count_in_parent": RESTORED_SCORER_COUNT,
            "fullres_focal_logit_minimum_in_parent": NEW_EDGE_FOCAL_LOGIT_MINIMUM,
            "tail": "focal-gated non-adjacent tail96",
            "all_layouts_strict_original_upright_permutations": True,
            "rows": list(rows),
        },
    )
    artifacts = {
        "frozen_archive": archive,
        "frozen_metadata": metadata,
        "layout_archive": spec.layout_archive,
        "layout_metadata": spec.layout_metadata,
        "base_archive": spec.base_archive,
        "base_metadata": spec.base_metadata,
        "selective_archive": spec.selective_archive,
        "selective_metadata": spec.selective_metadata,
        "selective_parent_freeze": spec.selective_freeze,
        "fullres_archive": spec.fullres_archive,
        "fullres_metadata": spec.fullres_metadata,
        "fullres_parent_freeze": spec.fullres_freeze,
        "runner": Path(__file__).resolve(),
        "fusion_module": (PROJECT_ROOT / "src/aiijc_puzzle/taska_selective_fullres_fusion.py"),
        "raw_solver": PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py",
    }
    _write_json(
        freeze,
        {
            "schema": "aiijc-taska-selective-fullres-union-fusion-pre-score-freeze-v1",
            "created_before_exact_reference_reconstruction": True,
            "contains_evaluation_references_or_labels": False,
            "artifacts": {name: _record(path) for name, path in artifacts.items()},
        },
    )
    return archive, metadata, freeze


def _validate_freeze(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("created_before_exact_reference_reconstruction") is not True:
        raise RuntimeError("pre-score freeze timing contract changed")
    if payload.get("contains_evaluation_references_or_labels") is not False:
        raise RuntimeError("pre-score freeze contains labels")
    for name, record in payload.get("artifacts", {}).items():
        artifact = Path(record["path"])
        if not artifact.is_absolute():
            artifact = PROJECT_ROOT / artifact
        if not artifact.is_file() or sha256_file(artifact) != record["sha256"]:
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
        for frozen in frozen_rows:
            prefix = str(frozen["prefix"])
            source = str(frozen["source_filename"])
            draw = int(frozen["draw_index"])
            dirty = finetune._dirty_case(cache, lookup[source], source, draw)
            if finetune._dirty_sha256(dirty.dirty_tiles) != frozen["dirty_sha256"]:
                raise RuntimeError("scoring recreated different dirty bytes")
            reference = finetune._reference(cache, lookup[source], source, draw, dirty.dirty_tiles)
            truth = _truth_edges(reference)
            current = set(_edges(candidate, prefix, "current"))
            selective = set(_edges(candidate, prefix, "selective_new"))
            fullres = set(_edges(candidate, prefix, "fullres_accepted"))
            unique = set(_edges(candidate, prefix, "unique_fullres"))
            overlap_current = fullres & current
            overlap_selective = fullres & selective
            combined = current | selective | unique
            metrics = {
                arm: _layout_metrics(candidate[f"{prefix}__{arm}_layout"], reference)
                for arm in SCORED_ARMS
            }
            scored.append(
                {
                    "prefix": prefix,
                    "source_filename": source,
                    "draw_index": draw,
                    "choice": frozen["choice"],
                    "mechanical_control_matches_frozen": frozen[
                        "mechanical_control_matches_frozen"
                    ],
                    "metrics": metrics,
                    "supply": {
                        "current_edges": len(current),
                        "current_true_edges": len(current & truth),
                        "selective_new_edges": len(selective),
                        "selective_new_true_edges": len(selective & truth),
                        "fullres_accepted_edges": len(fullres),
                        "fullres_accepted_true_edges": len(fullres & truth),
                        "fullres_overlap_current_edges": len(overlap_current),
                        "fullres_overlap_selective_edges": len(overlap_selective),
                        "fullres_overlap_selective_true_edges": len(overlap_selective & truth),
                        "unique_fullres_edges": len(unique),
                        "unique_fullres_true_edges": len(unique & truth),
                        "combined_union_edges": len(combined),
                        "combined_union_true_edges": len(combined & truth),
                    },
                }
            )
    return scored, _summarize(scored)


def _run_panel(
    spec: PanelSpec,
    *,
    output_dir: Path,
    lookup: Mapping[str, Mapping[str, Any]] | None,
    cache: Any | None,
    target_free_only: bool,
) -> dict[str, Any]:
    aligned = _aligned_rows(spec)
    arrays: dict[str, np.ndarray] = {}
    frozen_rows: list[dict[str, Any]] = []
    started = perf_counter()
    with (
        np.load(spec.layout_archive, allow_pickle=False) as layouts,
        np.load(spec.base_archive, allow_pickle=False) as base,
        np.load(spec.selective_archive, allow_pickle=False) as selective,
        np.load(spec.fullres_archive, allow_pickle=False) as fullres,
    ):
        for index, records in enumerate(aligned):
            row = records[2]
            prefix = str(row["prefix"])
            current = _edges(selective, prefix, "current")
            if current != _edges(base, prefix) or current != _edges(fullres, prefix, "current"):
                raise RuntimeError(f"{spec.name} {prefix} current edge identity mismatch")
            current_logits = np.asarray(
                selective[f"{prefix}__current_focal_logits"], dtype=np.float32
            )
            selective_new = _edges(selective, prefix, "accepted_new")
            selective_logits = np.asarray(
                selective[f"{prefix}__accepted_new_focal_logits"], dtype=np.float32
            )
            fullres_new, fullres_logits = _fullres_accepted_with_logits(fullres, prefix)
            frozen_control = strict_layout(
                selective[f"{prefix}__selective_vote500_focal_gated_layout"]
            )
            result = compose_selective_fullres_fusion(
                cost_right=_matrix(base, f"{prefix}__cost_right"),
                cost_down=_matrix(base, f"{prefix}__cost_down"),
                four_layouts=_four_layouts(layouts, prefix),
                frozen_selective_control=frozen_control,
                current_edges=current,
                current_logits=current_logits,
                selective_new_edges=selective_new,
                selective_new_logits=selective_logits,
                fullres_accepted_edges=fullres_new,
                fullres_accepted_logits=fullres_logits,
            )
            parent_choice = str(row["candidate_choice"])
            if result.selective_choice != parent_choice:
                raise RuntimeError("mechanical selective selector choice mismatch")
            replay_match = bool(np.array_equal(result.mechanical_control_layout, frozen_control))
            if not replay_match:
                raise RuntimeError("mechanical selective final control mismatch")
            arrays[f"{prefix}__selective_target500_control_layout"] = result.control_layout
            arrays[f"{prefix}__combined_union_candidate_layout"] = result.candidate_layout
            arrays[f"{prefix}__mechanical_control_layout"] = result.mechanical_control_layout
            arrays[f"{prefix}__selective_union_layout"] = result.selective_union_layout
            arrays[f"{prefix}__combined_union_layout"] = result.combined_union_layout
            for name, edges in (
                ("current", result.supply.current_edges),
                ("selective_new", result.supply.selective_new_edges),
                ("fullres_accepted", result.supply.fullres_accepted_edges),
                ("unique_fullres", result.supply.unique_fullres_edges),
                ("combined_union", result.supply.combined_union_edges),
            ):
                arrays.update(_edge_arrays(prefix, name, edges))
            arrays[f"{prefix}__current_focal_logits"] = result.supply.current_logits
            arrays[f"{prefix}__selective_new_focal_logits"] = result.supply.selective_new_logits
            arrays[f"{prefix}__fullres_accepted_focal_logits"] = (
                result.supply.fullres_accepted_logits
            )
            arrays[f"{prefix}__unique_fullres_focal_logits"] = result.supply.unique_fullres_logits
            arrays[f"{prefix}__combined_union_focal_logits"] = result.supply.combined_union_logits
            frozen_rows.append(
                {
                    "prefix": prefix,
                    "source_filename": str(row["source_filename"]),
                    "draw_index": int(row["draw_index"]),
                    "dirty_sha256": str(row["dirty_sha256"]),
                    "mechanical_control_matches_frozen": replay_match,
                    "selective_parent_choice": parent_choice,
                    **result.diagnostics(),
                }
            )
            print(
                json.dumps(
                    {
                        "event": f"{spec.name}_selective_fullres_fusion_target_free",
                        "case": index + 1,
                        "case_count": len(aligned),
                        "fullres_accepted": len(fullres_new),
                        "overlap_selective": result.supply.fullres_overlap_selective_count,
                        "unique_fullres": len(result.supply.unique_fullres_edges),
                        "choice": result.choice,
                        "control_replay": replay_match,
                    }
                ),
                flush=True,
            )
    archive, metadata, freeze = _freeze(spec, output_dir, arrays, frozen_rows)
    target_free_summary = {
        "case_count": len(frozen_rows),
        "control_replay_match_count": sum(
            row["mechanical_control_matches_frozen"] for row in frozen_rows
        ),
        "choice_counts": dict(Counter(row["choice"] for row in frozen_rows)),
        "mean_fullres_accepted": float(
            np.mean([row["fullres_accepted_new_count"] for row in frozen_rows])
        ),
        "mean_fullres_overlap_selective": float(
            np.mean([row["fullres_overlap_selective_count"] for row in frozen_rows])
        ),
        "mean_unique_fullres": float(
            np.mean([row["unique_fullres_accepted_count"] for row in frozen_rows])
        ),
    }
    payload: dict[str, Any] = {
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
        if lookup is None or cache is None:
            raise RuntimeError("scoring resources are absent")
        scored, summary = _score_panel(
            archive=archive,
            metadata=metadata,
            freeze=freeze,
            lookup=lookup,
            cache=cache,
        )
        payload.update({"rows": scored, "summary": summary})
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    _require_inputs()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    if args.smoke_one:
        parent = PANELS["local32"]
        smoke = PanelSpec(
            **{
                **parent.__dict__,
                "name": "smoke1",
                "case_count": 1,
            }
        )
        local = _run_panel(
            smoke,
            output_dir=output_dir,
            lookup=None,
            cache=None,
            target_free_only=True,
        )
        report = {
            "schema": REPORT_SCHEMA,
            "status": "target-free-smoke",
            "local32": local,
            "reference_reconstructed": False,
            "competition_test_accessed": False,
        }
        _write_json(output_dir / "report.json", report)
        print(json.dumps(report, indent=2))
        return report

    config, _, _ = finetune._load_config(finetune.DEFAULT_CONFIG)
    lookup = finetune._manifest_lookup(config)
    cache = finetune.CleanTileCache(args.targets.resolve(), maximum_boards=2)
    local = _run_panel(
        PANELS["local32"],
        output_dir=output_dir,
        lookup=lookup,
        cache=cache,
        target_free_only=False,
    )
    local_delta = local["summary"]["candidate_minus_control"]["satisfied_adjacent_pairs"]["mean"]
    held: dict[str, Any] = {"status": "skipped_by_negative_local_pair_gate"}
    fresh: dict[str, Any] = {"status": "skipped_by_local_or_held_pair_gate"}
    if local_delta >= LOCAL_GATE:
        held = _run_panel(
            PANELS["held32"],
            output_dir=output_dir,
            lookup=lookup,
            cache=cache,
            target_free_only=False,
        )
        held_delta = held["summary"]["candidate_minus_control"]["satisfied_adjacent_pairs"]["mean"]
        if held_delta >= HELD_GATE:
            fresh = _run_panel(
                PANELS["fresh32"],
                output_dir=output_dir,
                lookup=lookup,
                cache=cache,
                target_free_only=False,
            )
        else:
            fresh = {"status": "skipped_by_held_pair_delta_below_0.5"}
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "protocol": {
            "matcher_rerun": False,
            "frozen_selective_target500_parent": True,
            "frozen_fullres_accepted_parent": True,
            "fullres_unique_rule": "drop overlap with current or selective accepted",
            "combined_order": "current + selective accepted + unique fullres accepted",
            "selector_roster": list(FUSION_ARM_NAMES),
            "standalone_fullres_arm_in_selector": False,
            "selector": "minimum original TASKA all-1104-bond cost",
            "tail": "focal-gated non-adjacent tail96 on selected candidate set",
            "control": "exact frozen selective target500 final layout",
            "local_pair_gate": LOCAL_GATE,
            "held_pair_gate": HELD_GATE,
            "no_threshold_budget_or_roster_sweep": True,
        },
        "local32": local,
        "held32": held,
        "fresh32": fresh,
        "runtime_seconds": perf_counter() - started,
        "legality": {
            "strict_original_upright_tile_permutations": True,
            "pixels_restored_replaced_rotated_or_warped": False,
            "targets_used_only_after_candidate_freeze": True,
            "competition_test_accessed": False,
            "postprocessing_used": False,
            "production_modified": False,
        },
        "artifacts": {
            "runner": _record(Path(__file__).resolve()),
            "fusion_module": _record(
                PROJECT_ROOT / "src/aiijc_puzzle/taska_selective_fullres_fusion.py"
            ),
            "raw_solver": _record(PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py"),
        },
    }
    _write_json(output_dir / "report.json", report)
    print(
        json.dumps(
            {name: report[name] for name in ("local32", "held32", "fresh32")},
            indent=2,
        )
    )
    return report


if __name__ == "__main__":
    run(parse_args())

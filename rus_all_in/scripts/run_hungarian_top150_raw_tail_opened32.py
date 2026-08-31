#!/usr/bin/env python3
"""Confirm a fixed Hungarian-top150 -> raw-tail solver on opened eval32.

This is a solver-only replay over already frozen, target-free Union-v2
right/down log assignments.  For each axis it solves the complete 576x576
assignment, keeps the 150 highest-scoring assigned edges, and feeds those
identities to the legal raw-tail global solver.  Layouts are written and hashed
before organizer targets are reopened for exact/pair scoring.

The first four cases were used as a small mechanism screen.  The arm was then
fixed and evaluated once on cases 4..31; the report keeps those confirmation
metrics separate from the all-32 descriptive summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from aiijc_puzzle.layout_evaluation import evaluate_layout
from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.raw_tail_global_solver import (
    RawTailEdge,
    RawTailGlobalConfig,
    solve_raw_tail_global,
)

try:
    from scripts.run_component_relation_reranker import CleanTileCache, prepare_case
except ModuleNotFoundError:
    from run_component_relation_reranker import CleanTileCache, prepare_case


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRID = 24
COUNT = GRID * GRID
PAIRS = 2 * GRID * (GRID - 1)
KEEP_PER_AXIS = 150
SCREEN_CASES = 4
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 864_503_191

CACHE_DIR = (
    PROJECT_ROOT
    / "outputs/union-hard-edge-priority/pilot-v1-final/target-free-cache"
)
RIGHT_PATH = CACHE_DIR / "eval-right-assignment.npy"
DOWN_PATH = CACHE_DIR / "eval-down-assignment.npy"
CACHE_METADATA_PATH = CACHE_DIR / "metadata.json"
COMPARATOR_PATH = (
    PROJECT_ROOT
    / "outputs/union-hard-edge-priority/pilot-v1-final/frozen-target-free-eval.npz"
)
SOURCE_CONFIG_PATH = PROJECT_ROOT / "configs/union_hard_edge_priority_pilot_v1.json"
MANIFEST_PATH = PROJECT_ROOT / "data/interim/validation_manifest.json"
TARGETS_PATH = PROJECT_ROOT / "data/raw/train/targets"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/hungarian-top150-raw-tail/opened32-v1"

EXPECTED_INPUT_SHA256 = {
    "right_assignment": "a9fbda51c7b263c3a43da5dc852f976b30e12656cc8e20b3d63013cb9cf1958b",
    "down_assignment": "93a6e62550c59012e8bb060facdb2b9895a31eff68be7c5afc62af4d3ba82a73",
    "cache_metadata": "2aa1d4a747cd08dca60a7bd66f5a347df9511c0f29e0f7be3877a2a2fcb0f6b5",
    "comparator_layouts": "86bf9dfa5f0117e3ea35e3c0806f5909a271c176b90cea24c0f1dc7802e11fcc",
    "source_config": "3cc28b93d88f7e13366740f59a230635a98a528cb11e5e941a0ce3fa9256e7f6",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def _validate_frozen_inputs() -> None:
    paths = {
        "right_assignment": RIGHT_PATH,
        "down_assignment": DOWN_PATH,
        "cache_metadata": CACHE_METADATA_PATH,
        "comparator_layouts": COMPARATOR_PATH,
        "source_config": SOURCE_CONFIG_PATH,
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256_file(path)
        if observed != EXPECTED_INPUT_SHA256[name]:
            raise RuntimeError(f"frozen input drifted: {name} {observed}")


def _strict_layout(value: Any) -> np.ndarray:
    layout = np.asarray(value, dtype=np.int64)
    if layout.shape != (COUNT,) or not np.array_equal(np.sort(layout), np.arange(COUNT)):
        raise ValueError("layout is not a strict 576-tile permutation")
    return np.ascontiguousarray(layout)


def _hungarian_top_edges(
    scores: np.ndarray,
    *,
    axis: str,
    keep: int = KEEP_PER_AXIS,
) -> tuple[RawTailEdge, ...]:
    matrix = np.asarray(scores, dtype=np.float64).copy()
    if matrix.shape[0] != matrix.shape[1] or not np.isfinite(matrix).all():
        raise ValueError("scores must be a finite square matrix")
    count = len(matrix)
    if not 1 <= keep <= count:
        raise ValueError("keep must be in [1, tile_count]")
    np.fill_diagonal(matrix, -np.inf)
    rows, columns = linear_sum_assignment(-matrix)
    order = np.argsort(-matrix[rows, columns], kind="stable")[:keep]
    return tuple(
        RawTailEdge(int(rows[index]), int(columns[index]), axis)  # type: ignore[arg-type]
        for index in order
    )


def _freeze_predictions(output_dir: Path) -> tuple[Path, Path, dict[str, Any]]:
    metadata = json.loads(CACHE_METADATA_PATH.read_text(encoding="utf-8"))
    cases = metadata.get("cases", {}).get("eval")
    if not isinstance(cases, list) or len(cases) != 32:
        raise ValueError("expected exactly 32 frozen eval cases")
    right = np.load(RIGHT_PATH, mmap_mode="r")
    down = np.load(DOWN_PATH, mmap_mode="r")
    if right.shape != (32, COUNT + 1, COUNT + 1) or down.shape != right.shape:
        raise ValueError("assignment cache shape drifted")

    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "frozen-target-free-layouts.npz"
    metadata_path = output_dir / "frozen-target-free-layouts.json"
    arrays: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    started = perf_counter()
    config = RawTailGlobalConfig(
        baseline_quantile=0.15,
        search_rounds=6,
        border_weight=0.0,
        random_seed=0,
        component_cap=0,
        fill_rounds=1,
    )
    for index, case in enumerate(cases):
        horizontal = np.asarray(right[index, :COUNT, :COUNT], dtype=np.float64)
        vertical = np.asarray(down[index, :COUNT, :COUNT], dtype=np.float64)
        edges = _hungarian_top_edges(horizontal, axis="right") + _hungarian_top_edges(
            vertical,
            axis="down",
        )
        result = solve_raw_tail_global(
            -horizontal,
            -vertical,
            edges,
            grid=GRID,
            config=config,
        )
        layout = _strict_layout(result.layout)
        prefix = f"case_{index:04d}"
        arrays[f"{prefix}__layout"] = layout.astype(np.int32)
        rows.append(
            {
                "index": index,
                "prefix": prefix,
                "case_id": str(case["case_id"]),
                "source_filename": str(case["source_filename"]),
                "draw_index": int(case["draw_index"]),
                "dirty_sha256": str(case["dirty_sha256"]),
                "candidate_edges": len(edges),
                "diagnostics": result.diagnostics.as_dict(),
            }
        )
    np.savez_compressed(archive_path, **arrays)
    frozen = {
        "schema": "aiijc-hungarian-top150-raw-tail-frozen-v1",
        "contains_targets_or_exact_references": False,
        "score_semantics": "high-is-good log partial-OT; dustbin sliced away",
        "candidate_rule": {
            "per_axis": "full Hungarian assignment, then highest assigned scores",
            "keep_per_axis": KEEP_PER_AXIS,
            "diagonal_forbidden": True,
        },
        "solver": {
            "name": "raw_tail_global_solver",
            "config": config.__dict__,
            "structural_border_unary": False,
        },
        "input_sha256": EXPECTED_INPUT_SHA256,
        "rows": rows,
        "runtime_seconds": perf_counter() - started,
    }
    metadata_path.write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return archive_path, metadata_path, frozen


def _manifest_records() -> dict[str, Mapping[str, Any]]:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    records: dict[str, Mapping[str, Any]] = {}
    for split in payload["splits"].values():
        for record in split:
            records[str(record["filename"])] = record
    return records


def _arm_metrics(layout: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    result = evaluate_layout(layout, reference, reference_is_exact=True).as_dict()
    return {
        "exact_tiles": int(result["correct_tile_count"]),
        "adjacency_correct": int(result["adjacency_correct"]),
        "adjacency_total": int(result["adjacency_total"]),
        "adjacency_recall": float(result["adjacency"]),
        "strict_permutation": True,
    }


def _summary(rows: list[dict[str, Any]], arm: str, indices: range) -> dict[str, float]:
    selected = [rows[index][arm] for index in indices]
    return {
        "case_count": float(len(selected)),
        "exact_tiles_per_board": float(np.mean([row["exact_tiles"] for row in selected])),
        "satisfied_adjacent_pairs_per_board": float(
            np.mean([row["adjacency_correct"] for row in selected])
        ),
        "adjacency_recall": float(np.mean([row["adjacency_recall"] for row in selected])),
        "strict_fraction": float(np.mean([row["strict_permutation"] for row in selected])),
    }


def _clustered_delta(
    rows: list[dict[str, Any]],
    *,
    candidate: str,
    baseline: str,
    metric: str,
    indices: range,
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    raw: list[float] = []
    for index in indices:
        row = rows[index]
        delta = float(row[candidate][metric]) - float(row[baseline][metric])
        grouped[str(row["source_filename"])].append(delta)
        raw.append(delta)
    source_means = np.asarray([np.mean(grouped[name]) for name in sorted(grouped)])
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    distribution = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_SAMPLES, 2048):
        stop = min(start + 2048, BOOTSTRAP_SAMPLES)
        sample = generator.integers(
            0,
            len(source_means),
            size=(stop - start, len(source_means)),
        )
        distribution[start:stop] = source_means[sample].mean(axis=1)
    return {
        "mean": float(np.mean(raw)),
        "ci95_lower": float(np.quantile(distribution, 0.025)),
        "ci95_upper": float(np.quantile(distribution, 0.975)),
        "source_count": len(source_means),
        "case_count": len(raw),
        "wins": int(np.count_nonzero(source_means > 0)),
        "ties": int(np.count_nonzero(source_means == 0)),
        "losses": int(np.count_nonzero(source_means < 0)),
    }


def _score(
    archive_path: Path,
    metadata_path: Path,
    frozen: Mapping[str, Any],
) -> dict[str, Any]:
    archive_sha = sha256_file(archive_path)
    metadata_sha = sha256_file(metadata_path)
    source_config = json.loads(SOURCE_CONFIG_PATH.read_text(encoding="utf-8"))
    synthetic_seed = int(source_config["selection"]["synthetic_seed"])
    records = _manifest_records()
    cache = CleanTileCache(TARGETS_PATH)
    scored: list[dict[str, Any]] = []
    with np.load(archive_path, allow_pickle=False) as candidate_archive, np.load(
        COMPARATOR_PATH,
        allow_pickle=False,
    ) as comparators:
        for row in frozen["rows"]:
            case = prepare_case(
                cache,
                records[str(row["source_filename"])],
                draw_index=int(row["draw_index"]),
                seed=synthetic_seed,
            )
            dirty_sha = hashlib.sha256(np.ascontiguousarray(case.dirty_tiles).tobytes()).hexdigest()
            if case.case_id != row["case_id"] or dirty_sha != row["dirty_sha256"]:
                raise RuntimeError("exact scoring recreated a different synthetic case")
            reference = _strict_layout(np.argsort(case.input_tile_to_position))
            prefix = str(row["prefix"])
            candidate = _strict_layout(candidate_archive[f"{prefix}__layout"])
            union = _strict_layout(comparators[f"{prefix}__union_v2_layout"])
            learned = _strict_layout(comparators[f"{prefix}__learned_priority_layout"])
            scored.append(
                {
                    "source_filename": str(row["source_filename"]),
                    "draw_index": int(row["draw_index"]),
                    "case_id": str(row["case_id"]),
                    "union_v2": _arm_metrics(union, reference),
                    "learned_priority": _arm_metrics(learned, reference),
                    "hungarian_top150_raw_tail": _arm_metrics(candidate, reference),
                }
            )

    arms = ("union_v2", "learned_priority", "hungarian_top150_raw_tail")
    panels = {
        "screen_first4": range(0, SCREEN_CASES),
        "confirmation_last28": range(SCREEN_CASES, len(scored)),
        "all32_descriptive": range(0, len(scored)),
    }
    summaries = {
        panel: {arm: _summary(scored, arm, indices) for arm in arms}
        for panel, indices in panels.items()
    }
    deltas: dict[str, Any] = {}
    for panel, indices in panels.items():
        deltas[panel] = {}
        for baseline in ("union_v2", "learned_priority"):
            deltas[panel][f"candidate_minus_{baseline}"] = {
                metric: _clustered_delta(
                    scored,
                    candidate="hungarian_top150_raw_tail",
                    baseline=baseline,
                    metric=metric,
                    indices=indices,
                )
                for metric in ("exact_tiles", "adjacency_correct")
            }
    return {
        "schema": "aiijc-hungarian-top150-raw-tail-report-v1",
        "status": "confirmed-over-union-pairs_exact-tradeoff-vs-learned",
        "selection_protocol": {
            "mechanism_screen_indices": [0, 1, 2, 3],
            "confirmation_indices": list(range(4, 32)),
            "fixed_before_confirmation": True,
            "pair_denominator": PAIRS,
            "freshness_caveat": (
                "opened eval32; confirmation means unused by this arm, "
                "not globally untouched"
            ),
        },
        "frozen_predictions": {
            "archive": str(archive_path.relative_to(PROJECT_ROOT)),
            "archive_sha256": archive_sha,
            "metadata": str(metadata_path.relative_to(PROJECT_ROOT)),
            "metadata_sha256": metadata_sha,
            "hash_frozen_before_reference_recreation": True,
        },
        "summaries": summaries,
        "deltas": deltas,
        "rows": scored,
        "legality": {
            "targets_used_for_prediction": False,
            "target_index_or_filename_features": False,
            "input_tiles_rotated_warped_or_replaced": False,
            "output_is_strict_original_upright_tile_permutation": True,
            "pixel_postprocessing": False,
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _validate_frozen_inputs()
    archive_path, metadata_path, frozen = _freeze_predictions(args.output_dir)
    report = _score(archive_path, metadata_path, frozen)
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    all32 = report["summaries"]["all32_descriptive"]
    candidate = all32["hungarian_top150_raw_tail"]
    print(
        json.dumps(
            {
                "report": str(report_path),
                "exact_tiles_per_board": candidate["exact_tiles_per_board"],
                "satisfied_adjacent_pairs_per_board": candidate[
                    "satisfied_adjacent_pairs_per_board"
                ],
                "adjacency_recall": candidate["adjacency_recall"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

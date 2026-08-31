#!/usr/bin/env python3
"""Gate the single fixed TASKA mutual-vote target 500 supply experiment.

The target-500 matcher and its four-arm+tail96 layout are frozen before exact
synthetic references are reconstructed.  The unchanged frozen target-350
four-arm+tail96 layout is the control.  Local32 opens held32 only at a
nonnegative pair delta; held32 opens fresh32 only at pair delta >= +0.5 with
no severe local collapse.  No threshold is swept.
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
from aiijc_puzzle.taska_pair_pipeline import (
    RAW_TAIL_GLOBAL_SOLVER_SHA256,
    load_taska_pair_pipeline_resources,
)
from aiijc_puzzle.taska_vote500 import (
    VOTE500_MATCHER_CONFIG,
    VOTE_TARGET,
    solve_taska_vote_target_pair,
    strict_layout,
)

try:
    from scripts import run_taska_focal_current_finetune as finetune
except ModuleNotFoundError:
    import run_taska_focal_current_finetune as finetune

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/taska-vote500/v1"
GRID = 24
COUNT = GRID * GRID
PAIR_DENOMINATOR = 2 * GRID * (GRID - 1)
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 2_026_083_174
LOCAL_GATE = 0.0
HELD_GATE = 0.5
SEVERE_LOCAL_COLLAPSE = -2.0
CONTROL_EXPECTED = {
    "local32": (314.375, 1.375),
    "held32": (337.5625, 3.0625),
    "fresh32": (346.0625, 1.15625),
}


@dataclass(frozen=True)
class Panel:
    name: str
    control_archive: Path
    control_metadata: Path
    control_edge_archive: Path
    control_layout_key: str = "four_arm_tail96_layout"


PANELS = {
    "local32": Panel(
        "local32",
        PROJECT_ROOT
        / "outputs/taska-focal-feature-stacker/train96-v1/local32/frozen-target-free-eval.npz",
        PROJECT_ROOT
        / "outputs/taska-focal-feature-stacker/train96-v1/local32/frozen-target-free-eval.json",
        PROJECT_ROOT
        / "outputs/taska-focal-feature-stacker/train96-v1/local32/frozen-target-free-eval.npz",
    ),
    "held32": Panel(
        "held32",
        PROJECT_ROOT
        / "outputs/taska-focal-feature-stacker/train96-v1/held32/frozen-target-free-eval.npz",
        PROJECT_ROOT
        / "outputs/taska-focal-feature-stacker/train96-v1/held32/frozen-target-free-eval.json",
        PROJECT_ROOT
        / "outputs/taska-seam-replay/held300-diagnostic-mps-v1/frozen-target-free-eval.npz",
    ),
    "fresh32": Panel(
        "fresh32",
        PROJECT_ROOT
        / (
            "outputs/taska-focal-feature-stacker/train96-v1/"
            "fresh32-exact-override/frozen-target-free-eval.npz"
        ),
        PROJECT_ROOT
        / (
            "outputs/taska-focal-feature-stacker/train96-v1/"
            "fresh32-exact-override/frozen-target-free-eval.json"
        ),
        PROJECT_ROOT
        / (
            "outputs/taska-fresh32-leader-confirmation/"
            "fresh-held32-mps-v1/frozen-target-free-eval.npz"
        ),
    ),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--smoke-one", action="store_true")
    return parser.parse_args(argv)


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


def _record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        name = str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        name = str(resolved)
    return {"path": name, "sha256": sha256_file(resolved)}


def _edges(
    archive: Any,
    prefix: str,
    *,
    arm: str | None = None,
) -> tuple[RawTailEdge, ...]:
    stem = f"{prefix}__{arm}_edge" if arm else f"{prefix}__edge"
    source = np.asarray(archive[f"{stem}_source"], dtype=np.int64)
    target = np.asarray(archive[f"{stem}_target"], dtype=np.int64)
    axis = np.asarray(archive[f"{stem}_axis"], dtype=np.uint8)
    if not (source.ndim == target.ndim == axis.ndim == 1):
        raise ValueError("candidate edge arrays must be vectors")
    if not (len(source) == len(target) == len(axis)) or not np.isin(axis, (0, 1)).all():
        raise ValueError("candidate edge arrays are malformed")
    return tuple(
        RawTailEdge(int(a), int(b), "down" if int(c) else "right")
        for a, b, c in zip(source, target, axis, strict=True)
    )


def _true_edges(reference: np.ndarray) -> set[RawTailEdge]:
    board = strict_layout(reference).reshape(GRID, GRID)
    return {
        *(
            RawTailEdge(int(board[row, col]), int(board[row, col + 1]), "right")
            for row in range(GRID)
            for col in range(GRID - 1)
        ),
        *(
            RawTailEdge(int(board[row, col]), int(board[row + 1, col]), "down")
            for row in range(GRID - 1)
            for col in range(GRID)
        ),
    }


def _layout_edges(layout: np.ndarray) -> set[RawTailEdge]:
    return _true_edges(layout)


def _metrics(
    layout: np.ndarray,
    reference: np.ndarray,
    candidates: tuple[RawTailEdge, ...],
) -> dict[str, Any]:
    score = evaluate_layout(layout, reference, reference_is_exact=True)
    if score.adjacency_total != PAIR_DENOMINATOR:
        raise RuntimeError("pair denominator changed")
    true = _true_edges(reference)
    candidate = set(candidates)
    realised = _layout_edges(layout)
    supplied = candidate & true
    return {
        "satisfied_adjacent_pairs": int(score.adjacency_correct),
        "adjacency_recall": float(score.adjacency),
        "exact_tiles": int(score.correct_tile_count),
        "candidate_count": len(candidate),
        "supplied_true_pairs": len(supplied),
        "candidate_true_pair_recall": len(supplied) / PAIR_DENOMINATOR,
        "realised_supplied_true_pairs": len(realised & supplied),
        "realised_true_noncandidate_pairs": len((realised & true) - candidate),
        "strict_original_tile_permutation": True,
    }


def _cluster_ci(values: Sequence[float], sources: Sequence[str], *, seed: int) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, source in zip(values, sources, strict=True):
        grouped[source].append(float(value))
    cluster = np.asarray([np.mean(grouped[name]) for name in sorted(grouped)])
    generator = np.random.default_rng(seed)
    distribution = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_RESAMPLES, 2048):
        stop = min(start + 2048, BOOTSTRAP_RESAMPLES)
        sampled = generator.integers(0, len(cluster), size=(stop - start, len(cluster)))
        distribution[start:stop] = cluster[sampled].mean(axis=1)
    return {
        "mean": float(np.mean(values)),
        "ci95_lower": float(np.quantile(distribution, 0.025)),
        "ci95_upper": float(np.quantile(distribution, 0.975)),
        "source_count": len(cluster),
        "case_count": len(values),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": seed,
    }


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metric_names = (
        "satisfied_adjacent_pairs",
        "adjacency_recall",
        "exact_tiles",
        "candidate_count",
        "supplied_true_pairs",
        "candidate_true_pair_recall",
        "realised_supplied_true_pairs",
        "realised_true_noncandidate_pairs",
    )
    result: dict[str, Any] = {
        "pair_denominator": PAIR_DENOMINATOR,
        "arms": {
            arm: {
                metric: float(np.mean([row["metrics"][arm][metric] for row in rows]))
                for metric in metric_names
            }
            for arm in ("historical_target350", "target350_control", "target500")
        },
        "target500_vote_threshold_counts": dict(
            Counter(str(row["target500_vote_threshold"]) for row in rows)
        ),
        "target350_vote_threshold_counts": dict(
            Counter(str(row["target350_vote_threshold"]) for row in rows)
        ),
        "target500_four_arm_choice_counts": dict(
            Counter(str(row["target500_choice"]) for row in rows)
        ),
    }
    sources = [str(row["source_filename"]) for row in rows]
    deltas: dict[str, Any] = {}
    for index, metric in enumerate(("satisfied_adjacent_pairs", "adjacency_recall", "exact_tiles")):
        values = [
            float(row["metrics"]["target500"][metric])
            - float(row["metrics"]["target350_control"][metric])
            for row in rows
        ]
        current = _cluster_ci(values, sources, seed=BOOTSTRAP_SEED + index)
        current["case_wins_ties_losses"] = {
            "wins": sum(value > 0 for value in values),
            "ties": sum(value == 0 for value in values),
            "losses": sum(value < 0 for value in values),
        }
        deltas[metric] = current
    result["target500_minus_target350"] = deltas
    result["same_pass_target350_minus_historical"] = {
        metric: float(
            np.mean(
                [
                    float(row["metrics"]["target350_control"][metric])
                    - float(row["metrics"]["historical_target350"][metric])
                    for row in rows
                ]
            )
        )
        for metric in ("satisfied_adjacent_pairs", "adjacency_recall", "exact_tiles")
    }
    return result


def _validate_freeze(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("created_before_exact_reference_reconstruction") is not True:
        raise RuntimeError("freeze timing contract differs")
    for record in payload["artifacts"].values():
        artifact = Path(record["path"])
        if not artifact.is_absolute():
            artifact = PROJECT_ROOT / artifact
        if sha256_file(artifact) != record["sha256"]:
            raise RuntimeError(f"frozen artifact changed: {artifact}")


def _run_panel(
    panel: Panel,
    *,
    output_dir: Path,
    resources: Any,
    lookup: Mapping[str, Mapping[str, Any]],
    cache: Any,
    smoke_one: bool,
) -> dict[str, Any]:
    metadata = json.loads(panel.control_metadata.read_text(encoding="utf-8"))
    parent_rows = metadata["rows"][: 1 if smoke_one else None]
    if not smoke_one and len(parent_rows) != 32:
        raise ValueError(f"{panel.name} must contain exactly 32 cases")
    stage = output_dir / panel.name
    stage.mkdir(parents=True, exist_ok=False)
    arrays: dict[str, np.ndarray] = {}
    frozen_rows: list[dict[str, Any]] = []
    started = perf_counter()
    for index, parent_row in enumerate(parent_rows):
        prefix = str(parent_row["prefix"])
        source = str(parent_row["source_filename"])
        draw = int(parent_row["draw_index"])
        dirty = finetune._dirty_case(cache, lookup[source], source, draw)
        dirty_sha = finetune._dirty_sha256(dirty.dirty_tiles)
        if dirty_sha != parent_row["dirty_sha256"]:
            raise RuntimeError(f"{panel.name} dirty bytes differ from frozen control")
        solved = solve_taska_vote_target_pair(dirty.dirty_tiles, resources)
        for arm, result in (
            ("target350", solved.target350),
            ("target500", solved.target500),
        ):
            arrays[f"{prefix}__{arm}_layout"] = result.layout
            arrays[f"{prefix}__{arm}_edge_source"] = np.asarray(
                [edge.source for edge in result.candidate_edges], dtype=np.int32
            )
            arrays[f"{prefix}__{arm}_edge_target"] = np.asarray(
                [edge.target for edge in result.candidate_edges], dtype=np.int32
            )
            arrays[f"{prefix}__{arm}_edge_axis"] = np.asarray(
                [edge.axis == "down" for edge in result.candidate_edges], dtype=np.uint8
            )
        frozen_rows.append(
            {
                "prefix": prefix,
                "source_filename": source,
                "draw_index": draw,
                "dirty_sha256": dirty_sha,
                "target350_candidate_edge_count": len(solved.target350.candidate_edges),
                "target500_candidate_edge_count": len(solved.target500.candidate_edges),
                "target350_chosen_vote_threshold": (
                    solved.target350.chosen_vote_threshold
                ),
                "target500_chosen_vote_threshold": (
                    solved.target500.chosen_vote_threshold
                ),
                "scorer_count": solved.target500.scorer_count,
                "target350_four_arm_choice": solved.target350.choice,
                "target500_four_arm_choice": solved.target500.choice,
                "target350_four_arm_all_bond_costs": dict(solved.target350.costs),
                "target500_four_arm_all_bond_costs": dict(solved.target500.costs),
            }
        )
        print(
            json.dumps(
                {
                    "event": f"vote500_{panel.name}_target_free",
                    "case": index + 1,
                    "case_count": len(parent_rows),
                    "target350_candidate_edges": len(solved.target350.candidate_edges),
                    "target500_candidate_edges": len(solved.target500.candidate_edges),
                    "target350_vote_threshold": (
                        solved.target350.chosen_vote_threshold
                    ),
                    "target500_vote_threshold": (
                        solved.target500.chosen_vote_threshold
                    ),
                }
            ),
            flush=True,
        )
    archive = stage / "frozen-target-free-eval.npz"
    frozen_metadata = stage / "frozen-target-free-eval.json"
    freeze = stage / "pre-score-freeze.json"
    _write_npz(archive, arrays)
    _write_json(
        frozen_metadata,
        {
            "schema": "aiijc-taska-vote500-target-free-v1",
            "stage": panel.name,
            "contains_exact_references_or_labels": False,
            "contains_target_ids_or_source_grid_coordinates": False,
            "single_fixed_vote_target": VOTE_TARGET,
            "all_layouts_strict_original_upright_tile_permutations": True,
            "rows": frozen_rows,
        },
    )
    sources = {
        "frozen_archive": archive,
        "frozen_metadata": frozen_metadata,
        "runner": Path(__file__).resolve(),
        "vote500_module": PROJECT_ROOT / "src/aiijc_puzzle/taska_vote500.py",
        "raw_solver": PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py",
        "control_archive": panel.control_archive,
        "control_metadata": panel.control_metadata,
        "control_edge_archive": panel.control_edge_archive,
    }
    _write_json(
        freeze,
        {
            "schema": "aiijc-taska-vote500-pre-score-freeze-v1",
            "created_before_exact_reference_reconstruction": True,
            "contains_evaluation_references_or_labels": False,
            "artifacts": {name: _record(path) for name, path in sources.items()},
        },
    )
    _validate_freeze(freeze)
    scored: list[dict[str, Any]] = []
    with (
        np.load(archive, allow_pickle=False) as candidate,
        np.load(panel.control_archive, allow_pickle=False) as control,
        np.load(panel.control_edge_archive, allow_pickle=False) as control_edges,
    ):
        for row in frozen_rows:
            prefix = str(row["prefix"])
            source = str(row["source_filename"])
            draw = int(row["draw_index"])
            dirty = finetune._dirty_case(cache, lookup[source], source, draw)
            reference = finetune._reference(
                cache, lookup[source], source, draw, dirty.dirty_tiles
            )
            target350_edges = _edges(candidate, prefix, arm="target350")
            target500_edges = _edges(candidate, prefix, arm="target500")
            historical_edges = _edges(control_edges, prefix)
            metrics = {
                "historical_target350": _metrics(
                    strict_layout(control[f"{prefix}__{panel.control_layout_key}"]),
                    reference,
                    historical_edges,
                ),
                "target350_control": _metrics(
                    strict_layout(candidate[f"{prefix}__target350_layout"]),
                    reference,
                    target350_edges,
                ),
                "target500": _metrics(
                    strict_layout(candidate[f"{prefix}__target500_layout"]),
                    reference,
                    target500_edges,
                ),
            }
            scored.append(
                {
                    "prefix": prefix,
                    "source_filename": source,
                    "draw_index": draw,
                    "target350_vote_threshold": row[
                        "target350_chosen_vote_threshold"
                    ],
                    "target500_vote_threshold": row[
                        "target500_chosen_vote_threshold"
                    ],
                    "target500_choice": row["target500_four_arm_choice"],
                    "metrics": metrics,
                }
            )
    summary = _summary(scored)
    if not smoke_one:
        expected_pairs, expected_exact = CONTROL_EXPECTED[panel.name]
        control_summary = summary["arms"]["historical_target350"]
        if not math.isclose(control_summary["satisfied_adjacent_pairs"], expected_pairs):
            raise RuntimeError(f"{panel.name} target350 pair control did not replay")
        if not math.isclose(control_summary["exact_tiles"], expected_exact):
            raise RuntimeError(f"{panel.name} target350 exact control did not replay")
    return {
        "status": "smoke-only" if smoke_one else "complete",
        "summary": summary,
        "rows": scored,
        "runtime_seconds": perf_counter() - started,
        "artifacts": {
            "archive": _record(archive),
            "metadata": _record(frozen_metadata),
            "pre_score_freeze": _record(freeze),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if sha256_file(PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py") != (
        RAW_TAIL_GLOBAL_SOLVER_SHA256
    ):
        raise ValueError("frozen raw solver SHA-256 changed")
    if args.device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable")
        torch.use_deterministic_algorithms(False)
    else:
        torch.use_deterministic_algorithms(True)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    config, _, _ = finetune._load_config(finetune.DEFAULT_CONFIG)
    lookup = finetune._manifest_lookup(config)
    cache = finetune.CleanTileCache(args.targets.resolve(), maximum_boards=2)
    resources = load_taska_pair_pipeline_resources(device=args.device)
    started = perf_counter()
    local = _run_panel(
        PANELS["local32"],
        output_dir=output_dir,
        resources=resources,
        lookup=lookup,
        cache=cache,
        smoke_one=bool(args.smoke_one),
    )
    held: dict[str, Any] = {"status": "skipped_by_local_gate"}
    fresh: dict[str, Any] = {"status": "skipped_by_local_or_held_gate"}
    if not args.smoke_one:
        local_delta = local["summary"]["target500_minus_target350"][
            "satisfied_adjacent_pairs"
        ]["mean"]
        if local_delta >= LOCAL_GATE:
            held = _run_panel(
                PANELS["held32"],
                output_dir=output_dir,
                resources=resources,
                lookup=lookup,
                cache=cache,
                smoke_one=False,
            )
            held_delta = held["summary"]["target500_minus_target350"][
                "satisfied_adjacent_pairs"
            ]["mean"]
            if held_delta >= HELD_GATE and local_delta >= SEVERE_LOCAL_COLLAPSE:
                fresh = _run_panel(
                    PANELS["fresh32"],
                    output_dir=output_dir,
                    resources=resources,
                    lookup=lookup,
                    cache=cache,
                    smoke_one=False,
                )
            else:
                fresh = {"status": "skipped_by_held_or_local_gate"}
    report = {
        "schema": "aiijc-taska-vote500-report-v1",
        "protocol": {
            "single_fixed_change": "dynamic mutual-vote target 350 -> 500",
            "no_threshold_or_parameter_sweep": True,
            "matcher_views": list(VOTE500_MATCHER_CONFIG.views),
            "orientations": VOTE500_MATCHER_CONFIG.orientations,
            "scorer_count": 12,
            "local_gate_pair_delta_gte": LOCAL_GATE,
            "held_gate_pair_delta_gte": HELD_GATE,
            "severe_local_collapse_below": SEVERE_LOCAL_COLLAPSE,
            "layouts_and_candidate_edges_frozen_before_references": True,
        },
        "local32": local,
        "held32": held,
        "fresh32": fresh,
        "runtime_seconds": perf_counter() - started,
        "legality": {
            "dirty_original_tiles_only_at_inference": True,
            "denoised_views_used_only_inside_matcher": True,
            "targets_or_exact_references_at_inference": False,
            "all_outputs_strict_original_upright_tile_permutations": True,
            "pixels_emitted_or_modified": False,
            "competition_test_accessed": False,
        },
        "raw_solver_sha256": RAW_TAIL_GLOBAL_SOLVER_SHA256,
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

#!/usr/bin/env python3
"""Gate one fixed selective target-500 supply arm on focal-gated TASKA.

Each board uses one target-500 matcher pass.  The current target-350 subset is
derived from that same pass; only target500-minus-current edges with recovered
focal ``train_exact_top5`` logit at least zero enter one fifth arm.  Original
all-bond costs select among the current four arms and this union-focal arm,
then the already confirmed focal-gated non-adjacent tail96 is applied.

Local32 opens held32 at nonnegative pair delta; held32 opens fresh32 at pair
delta at least +0.5.  No threshold, budget, or arm sweep is exposed.
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
    PAIR_DENOMINATOR,
    RAW_TAIL_GLOBAL_SOLVER_SHA256,
    TaskaPairArtifactPaths,
    load_taska_pair_pipeline_resources,
)
from aiijc_puzzle.taska_selective_vote500 import (
    SELECTIVE_NEW_EDGE_LOGIT_THRESHOLD,
    SELECTIVE_VOTE500_ARM,
    solve_selective_vote500,
)
from aiijc_puzzle.taska_vote500 import VOTE_TARGET, strict_layout

try:
    from scripts import run_taska_focal_current_finetune as finetune
except ModuleNotFoundError:  # Direct ``python scripts/run_*.py`` execution.
    import run_taska_focal_current_finetune as finetune

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/taska-selective-vote500/fixed-v1"
GRID = 24
COUNT = GRID * GRID
LOCAL_GATE = 0.0
HELD_GATE = 0.5
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 2_026_083_190
ARMS = (
    "historical_focal_gated",
    "samepass_control_focal_gated",
    "selective_vote500_focal_gated",
)
REPORT_SCHEMA = "aiijc-taska-selective-vote500-report-v1"


@dataclass(frozen=True)
class PanelSpec:
    name: str
    case_count: int
    historical_archive: Path
    historical_metadata: Path


HISTORICAL_ROOT = (
    PROJECT_ROOT / "outputs/taska-focal-gated-protected-tail/logit0-v1"
)
PANELS = {
    name: PanelSpec(
        name,
        32,
        HISTORICAL_ROOT / name / "frozen-target-free-eval.npz",
        HISTORICAL_ROOT / name / "frozen-target-free-eval.json",
    )
    for name in ("local32", "held32", "fresh32")
}
EXPECTED_SHA256 = {
    PANELS["local32"].historical_archive: (
        "180699bb5b6fdd1e20d1487c43f8c76a96b0e236802cd523d5104631948bab47"
    ),
    PANELS["local32"].historical_metadata: (
        "3f63d9f188c2f7e4b5b5a51b4b0a93ed4316a5c8fa2e1ed60ecc25bf315e4448"
    ),
    PANELS["held32"].historical_archive: (
        "e76a72c57bfc00cd38f7113aa4be3c6f814381ae4ee563aac9831ed83ddd86c6"
    ),
    PANELS["held32"].historical_metadata: (
        "8b50eb17e95948700590ec65e85659f7ce688a7165b7b6a3729da81b5b8c7f96"
    ),
    PANELS["fresh32"].historical_archive: (
        "f4c02c1b30e118e9ce8be583ca1021612bea65950fcc53e2b558164eee1cadb0"
    ),
    PANELS["fresh32"].historical_metadata: (
        "d89e5deb634bfc90b0cfb6c30b0916ce58c8deb769e659f26d3e437f52bafd1f"
    ),
    PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py": (
        RAW_TAIL_GLOBAL_SOLVER_SHA256
    ),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument(
        "--smoke-one",
        action="store_true",
        help="freeze one local target-free case and compare its known control; do not score",
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


def _require_inputs() -> None:
    for path, expected in EXPECTED_SHA256.items():
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"frozen input SHA-256 mismatch: {path}")


def _historical_rows(spec: PanelSpec) -> list[Mapping[str, Any]]:
    payload = json.loads(spec.historical_metadata.read_text(encoding="utf-8"))
    if payload.get("contains_exact_references_or_labels") is not False:
        raise ValueError(f"{spec.name} historical metadata is not target-free")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) < spec.case_count:
        raise ValueError(f"{spec.name} historical row count changed")
    return rows[: spec.case_count]


def _edge_arrays(
    prefix: str, name: str, edges: Sequence[RawTailEdge]
) -> dict[str, np.ndarray]:
    return {
        f"{prefix}__{name}__edge_source": np.asarray(
            [edge.source for edge in edges], dtype=np.int16
        ),
        f"{prefix}__{name}__edge_target": np.asarray(
            [edge.target for edge in edges], dtype=np.int16
        ),
        f"{prefix}__{name}__edge_axis": np.asarray(
            [edge.axis == "down" for edge in edges], dtype=np.uint8
        ),
    }


def _edges(archive: Any, prefix: str, name: str) -> tuple[RawTailEdge, ...]:
    source = np.asarray(archive[f"{prefix}__{name}__edge_source"], dtype=np.int64)
    target = np.asarray(archive[f"{prefix}__{name}__edge_target"], dtype=np.int64)
    axis = np.asarray(archive[f"{prefix}__{name}__edge_axis"], dtype=np.uint8)
    if not (source.ndim == target.ndim == axis.ndim == 1):
        raise ValueError("edge arrays must be one-dimensional")
    if not (len(source) == len(target) == len(axis)) or not np.isin(axis, (0, 1)).all():
        raise ValueError("edge arrays are malformed")
    result = tuple(
        RawTailEdge(int(a), int(b), "down" if int(c) else "right")
        for a, b, c in zip(source, target, axis, strict=True)
    )
    if len(set(result)) != len(result):
        raise ValueError("edge arrays contain duplicates")
    return result


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


def _layout_metrics(layout: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    result = evaluate_layout(layout, reference, reference_is_exact=True)
    if result.adjacency_total != PAIR_DENOMINATOR:
        raise RuntimeError("pair denominator changed")
    return {
        "satisfied_adjacent_pairs": int(result.adjacency_correct),
        "adjacency_recall": float(result.adjacency),
        "exact_tiles": int(result.correct_tile_count),
        "strict_original_upright_permutation": True,
    }


def _cluster_ci(
    values: Sequence[float], sources: Sequence[str], *, seed: int
) -> dict[str, Any]:
    if len(values) != len(sources) or not values:
        raise ValueError("bootstrap inputs must be aligned and non-empty")
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, source in zip(values, sources, strict=True):
        if not math.isfinite(float(value)):
            raise ValueError("bootstrap values must be finite")
        grouped[source].append(float(value))
    source_means = np.asarray([np.mean(grouped[name]) for name in sorted(grouped)])
    generator = np.random.default_rng(seed)
    distribution = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_RESAMPLES, 2048):
        stop = min(start + 2048, BOOTSTRAP_RESAMPLES)
        indices = generator.integers(
            0, len(source_means), size=(stop - start, len(source_means))
        )
        distribution[start:stop] = source_means[indices].mean(axis=1)
    return {
        "mean": float(np.mean(values)),
        "ci95_lower": float(np.quantile(distribution, 0.025)),
        "ci95_upper": float(np.quantile(distribution, 0.975)),
        "source_count": len(source_means),
        "case_count": len(values),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": seed,
        "case_wins_ties_losses": {
            "wins": sum(value > 0 for value in values),
            "ties": sum(value == 0 for value in values),
            "losses": sum(value < 0 for value in values),
        },
        "source_wins_ties_losses": {
            "wins": int(np.sum(source_means > 0)),
            "ties": int(np.sum(source_means == 0)),
            "losses": int(np.sum(source_means < 0)),
        },
    }


def _summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = ("satisfied_adjacent_pairs", "adjacency_recall", "exact_tiles")
    sources = [str(row["source_filename"]) for row in rows]
    result: dict[str, Any] = {
        "case_count": len(rows),
        "pair_denominator": PAIR_DENOMINATOR,
        "arms": {
            arm: {
                metric: float(np.mean([row["metrics"][arm][metric] for row in rows]))
                for metric in metrics
            }
            for arm in ARMS
        },
        "candidate_choice_counts": dict(
            Counter(str(row["candidate_choice"]) for row in rows)
        ),
        "samepass_control_layout_match_count": sum(
            bool(row["mechanical_control_matches_historical"]) for row in rows
        ),
    }
    comparisons: dict[str, Any] = {}
    for comparison_index, (name, candidate, control) in enumerate(
        (
            (
                "candidate_minus_samepass_control",
                "selective_vote500_focal_gated",
                "samepass_control_focal_gated",
            ),
            (
                "samepass_control_minus_historical",
                "samepass_control_focal_gated",
                "historical_focal_gated",
            ),
        )
    ):
        values: dict[str, Any] = {}
        for metric_index, metric in enumerate(metrics):
            deltas = [
                float(row["metrics"][candidate][metric])
                - float(row["metrics"][control][metric])
                for row in rows
            ]
            values[metric] = _cluster_ci(
                deltas,
                sources,
                seed=BOOTSTRAP_SEED + 100 * comparison_index + metric_index,
            )
        comparisons[name] = values
    result["comparisons"] = comparisons
    fields = (
        "current_edges",
        "proposed_new_edges",
        "accepted_new_edges",
        "union_edges",
        "current_true_edges",
        "proposed_new_true_edges",
        "accepted_new_true_edges",
        "union_true_edges",
    )
    means = {
        field: float(np.mean([row["supply"][field] for row in rows]))
        for field in fields
    }
    result["candidate_supply_mean_per_board"] = means
    result["candidate_supply"] = {
        "current_recall": means["current_true_edges"] / PAIR_DENOMINATOR,
        "union_recall": means["union_true_edges"] / PAIR_DENOMINATOR,
        "proposed_new_precision": (
            sum(row["supply"]["proposed_new_true_edges"] for row in rows)
            / max(1, sum(row["supply"]["proposed_new_edges"] for row in rows))
        ),
        "accepted_new_precision": (
            sum(row["supply"]["accepted_new_true_edges"] for row in rows)
            / max(1, sum(row["supply"]["accepted_new_edges"] for row in rows))
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
            "schema": "aiijc-taska-selective-vote500-target-free-v1",
            "stage": spec.name,
            "contains_exact_references_or_candidate_labels": False,
            "one_target500_matcher_pass_per_case": True,
            "same_pass_target350_subset": True,
            "new_edge_focal_logit_threshold": SELECTIVE_NEW_EDGE_LOGIT_THRESHOLD,
            "all_layouts_strict_original_upright_permutations": True,
            "rows": list(rows),
        },
    )
    artifacts = {
        "frozen_archive": archive,
        "frozen_metadata": metadata,
        "historical_archive": spec.historical_archive,
        "historical_metadata": spec.historical_metadata,
        "runner": Path(__file__).resolve(),
        "selective_vote500_module": (
            PROJECT_ROOT / "src/aiijc_puzzle/taska_selective_vote500.py"
        ),
        "focal_gated_tail_module": (
            PROJECT_ROOT / "src/aiijc_puzzle/taska_focal_gated_protected_tail.py"
        ),
        "vote500_module": PROJECT_ROOT / "src/aiijc_puzzle/taska_vote500.py",
        "raw_solver": PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py",
    }
    model_paths = TaskaPairArtifactPaths()
    artifacts.update(
        {
            "matcher_v3": model_paths.matcher_v3,
            "matcher_local": model_paths.matcher_local,
            "logistic_calibrator": model_paths.logistic_calibrator,
            "focal_verifier": model_paths.focal_verifier,
            "nonlinear_calibrator": model_paths.nonlinear_calibrator,
        }
    )
    _write_json(
        freeze,
        {
            "schema": "aiijc-taska-selective-vote500-pre-score-freeze-v1",
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
            reference = finetune._reference(
                cache, lookup[source], source, draw, dirty.dirty_tiles
            )
            truth = _truth_edges(reference)
            current = set(_edges(candidate, prefix, "current"))
            proposed = set(_edges(candidate, prefix, "proposed_new"))
            accepted = set(_edges(candidate, prefix, "accepted_new"))
            union = current | accepted
            metrics = {
                arm: _layout_metrics(candidate[f"{prefix}__{arm}_layout"], reference)
                for arm in ARMS
            }
            scored.append(
                {
                    "prefix": prefix,
                    "source_filename": source,
                    "draw_index": draw,
                    "candidate_choice": frozen["candidate_choice"],
                    "mechanical_control_matches_historical": frozen[
                        "mechanical_control_matches_historical"
                    ],
                    "metrics": metrics,
                    "supply": {
                        "current_edges": len(current),
                        "proposed_new_edges": len(proposed),
                        "accepted_new_edges": len(accepted),
                        "union_edges": len(union),
                        "current_true_edges": len(current & truth),
                        "proposed_new_true_edges": len(proposed & truth),
                        "accepted_new_true_edges": len(accepted & truth),
                        "union_true_edges": len(union & truth),
                    },
                }
            )
    return scored, _summarize(scored)


def _run_panel(
    spec: PanelSpec,
    *,
    output_dir: Path,
    resources: Any,
    lookup: Mapping[str, Mapping[str, Any]],
    cache: Any,
    target_free_only: bool,
) -> dict[str, Any]:
    historical_rows = _historical_rows(spec)
    arrays: dict[str, np.ndarray] = {}
    frozen_rows: list[dict[str, Any]] = []
    started = perf_counter()
    with np.load(spec.historical_archive, allow_pickle=False) as historical:
        for index, row in enumerate(historical_rows):
            prefix = str(row["prefix"])
            source = str(row["source_filename"])
            draw = int(row["draw_index"])
            dirty = finetune._dirty_case(cache, lookup[source], source, draw)
            dirty_sha = finetune._dirty_sha256(dirty.dirty_tiles)
            if dirty_sha != row["dirty_sha256"]:
                raise RuntimeError(f"{spec.name} recreated different dirty bytes")
            result = solve_selective_vote500(dirty.dirty_tiles, resources)
            historical_layout = strict_layout(
                historical[f"{prefix}__focal_gated_tail96_layout"]
            )
            historical_match = bool(
                np.array_equal(result.control_layout, historical_layout)
            )
            arrays[f"{prefix}__historical_focal_gated_layout"] = historical_layout
            arrays[f"{prefix}__samepass_control_focal_gated_layout"] = (
                result.control_layout
            )
            arrays[f"{prefix}__selective_vote500_focal_gated_layout"] = (
                result.candidate_layout
            )
            arrays.update(_edge_arrays(prefix, "current", result.supply.current_edges))
            arrays.update(
                _edge_arrays(prefix, "proposed_new", result.supply.proposed_new_edges)
            )
            arrays.update(
                _edge_arrays(prefix, "accepted_new", result.supply.accepted_new_edges)
            )
            arrays[f"{prefix}__current_focal_logits"] = result.supply.current_logits
            arrays[f"{prefix}__proposed_new_focal_logits"] = (
                result.supply.proposed_new_logits
            )
            arrays[f"{prefix}__accepted_new_focal_logits"] = (
                result.supply.accepted_new_logits
            )
            frozen_rows.append(
                {
                    "prefix": prefix,
                    "source_filename": source,
                    "draw_index": draw,
                    "dirty_sha256": dirty_sha,
                    "mechanical_control_matches_historical": historical_match,
                    "historical_candidate_edge_count": int(row["candidate_edge_count"]),
                    "samepass_candidate_edge_count": len(result.supply.current_edges),
                    **result.diagnostics(),
                }
            )
            print(
                json.dumps(
                    {
                        "event": f"{spec.name}_selective_vote500_target_free",
                        "case": index + 1,
                        "case_count": len(historical_rows),
                        "source_filename": source,
                        "draw_index": draw,
                        "control_matches_historical": historical_match,
                        "current": len(result.supply.current_edges),
                        "proposed_new": len(result.supply.proposed_new_edges),
                        "accepted_new": len(result.supply.accepted_new_edges),
                        "choice": result.candidate_choice,
                    }
                ),
                flush=True,
            )
    archive, metadata, freeze = _freeze(spec, output_dir, arrays, frozen_rows)
    target_free_summary = {
        "case_count": len(frozen_rows),
        "control_layout_match_count": sum(
            row["mechanical_control_matches_historical"] for row in frozen_rows
        ),
        "mean_current_edges": float(
            np.mean([row["current_edge_count"] for row in frozen_rows])
        ),
        "mean_proposed_new_edges": float(
            np.mean([row["proposed_new_edge_count"] for row in frozen_rows])
        ),
        "mean_accepted_new_edges": float(
            np.mean([row["accepted_new_edge_count"] for row in frozen_rows])
        ),
        "candidate_choice_counts": dict(
            Counter(row["candidate_choice"] for row in frozen_rows)
        ),
    }
    result_payload: dict[str, Any] = {
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
        result_payload.update({"rows": rows, "summary": summary})
    return result_payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    _require_inputs()
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
    config, _, _ = finetune._load_config(finetune.DEFAULT_CONFIG)
    lookup = finetune._manifest_lookup(config)
    cache = finetune.CleanTileCache(args.targets.resolve(), maximum_boards=2)

    if args.smoke_one:
        parent = PANELS["local32"]
        smoke = PanelSpec(
            "smoke1",
            1,
            parent.historical_archive,
            parent.historical_metadata,
        )
        local = _run_panel(
            smoke,
            output_dir=output_dir,
            resources=resources,
            lookup=lookup,
            cache=cache,
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

    local = _run_panel(
        PANELS["local32"],
        output_dir=output_dir,
        resources=resources,
        lookup=lookup,
        cache=cache,
        target_free_only=False,
    )
    local_delta = local["summary"]["comparisons"][
        "candidate_minus_samepass_control"
    ]["satisfied_adjacent_pairs"]["mean"]
    held: dict[str, Any] = {"status": "skipped_by_negative_local_pair_gate"}
    fresh: dict[str, Any] = {"status": "skipped_by_local_or_held_pair_gate"}
    if local_delta >= LOCAL_GATE:
        held = _run_panel(
            PANELS["held32"],
            output_dir=output_dir,
            resources=resources,
            lookup=lookup,
            cache=cache,
            target_free_only=False,
        )
        held_delta = held["summary"]["comparisons"][
            "candidate_minus_samepass_control"
        ]["satisfied_adjacent_pairs"]["mean"]
        if held_delta >= HELD_GATE:
            fresh = _run_panel(
                PANELS["fresh32"],
                output_dir=output_dir,
                resources=resources,
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
            "single_fixed_selective_target500_arm": True,
            "one_target500_matcher_pass_per_case": True,
            "same_pass_target350_subset": True,
            "target500_vote_target": VOTE_TARGET,
            "new_edges": "target500 minus same-pass current350",
            "new_edge_focal_mode": "train_exact_top5",
            "new_edge_focal_logit_threshold": SELECTIVE_NEW_EDGE_LOGIT_THRESHOLD,
            "fifth_arm": SELECTIVE_VOTE500_ARM,
            "selector": "minimum original TASKA all-1104-bond cost",
            "tail": "fixed focal-gated non-adjacent tail96",
            "local_pair_gate": LOCAL_GATE,
            "held_pair_gate": HELD_GATE,
            "no_threshold_budget_or_arm_sweep": True,
        },
        "local32": local,
        "held32": held,
        "fresh32": fresh,
        "runtime_seconds": perf_counter() - started,
        "legality": {
            "dirty_tiles_only_for_candidate_inference": True,
            "all_outputs_strict_original_upright_tile_permutations": True,
            "targets_used_only_after_candidate_freeze": True,
            "pixels_restored_replaced_rotated_or_warped": False,
            "competition_test_accessed": False,
            "postprocessing_used": False,
        },
        "artifacts": {
            "runner": _record(Path(__file__).resolve()),
            "selective_vote500_module": _record(
                PROJECT_ROOT / "src/aiijc_puzzle/taska_selective_vote500.py"
            ),
            "raw_solver": _record(
                PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py"
            ),
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

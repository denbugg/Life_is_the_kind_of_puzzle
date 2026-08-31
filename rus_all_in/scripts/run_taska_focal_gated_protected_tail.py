#!/usr/bin/env python3
"""Gate TASKA tail protection with the fixed recovered-focal logit boundary.

This is one bounded replay with no threshold or swap-budget search.  For each
panel, the four-arm all-bond winner is reconstructed from already frozen
target-free archives.  The control runs the current tail96 protecting every
realised harvested edge.  The candidate runs the identical original-cost,
non-adjacent 96-swap tail while protecting only harvested edges whose frozen
``train_exact_top5`` focal logit is at least zero.

Candidate layouts are persisted and hash-frozen before exact references are
reconstructed.  The local32 panel was previously touched by a target-assisted
threshold diagnostic and is reported only as discovery.  A nonnegative local
pair delta opens held32; held pair delta of at least +0.5 opens fresh32.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from aiijc_puzzle.layout_evaluation import evaluate_layout
from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.taska_focal_gated_protected_tail import (
    FOCAL_PROTECTION_LOGIT_THRESHOLD,
    polish_taska_tail_with_focal_gate,
)
from aiijc_puzzle.taska_layout_portfolio import (
    select_lowest_taska_seam_cost_layout,
    total_taska_adjacent_seam_cost,
)
from aiijc_puzzle.taska_protected_tail_polish import polish_unprotected_taska_tail

try:
    from scripts import run_taska_focal_current_finetune as finetune
except ModuleNotFoundError:  # Direct ``python scripts/run_*.py`` execution.
    import run_taska_focal_current_finetune as finetune

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/taska-focal-gated-protected-tail/logit0-v1"
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
GRID = 24
COUNT = GRID * GRID
PAIR_DENOMINATOR = 2 * GRID * (GRID - 1)
TAIL_SWAPS = 96
MINIMUM_GAIN = 1e-9
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 2_026_083_177
ARMS = ("control_all_edges_tail96", "focal_gated_tail96")
FOUR_ARMS = ("raw", "logistic", "focal", "nonlinear")
REPORT_SCHEMA = "aiijc-taska-focal-gated-protected-tail-report-v1"

LOCAL_LAYOUT_ARCHIVE = PROJECT_ROOT / (
    "outputs/taska-focal-feature-stacker/train96-v1/local32/"
    "frozen-target-free-eval.npz"
)
LOCAL_METADATA = LOCAL_LAYOUT_ARCHIVE.with_suffix(".json")
HELD_LAYOUT_ARCHIVE = PROJECT_ROOT / (
    "outputs/taska-focal-feature-stacker/train96-v1/held32/"
    "frozen-target-free-eval.npz"
)
HELD_LAYOUT_METADATA = HELD_LAYOUT_ARCHIVE.with_suffix(".json")
HELD_COST_ARCHIVE = PROJECT_ROOT / (
    "outputs/taska-seam-replay/held300-diagnostic-mps-v1/"
    "frozen-target-free-eval.npz"
)
HELD_COST_METADATA = HELD_COST_ARCHIVE.with_suffix(".json")
HELD_FOCAL_ARCHIVE = PROJECT_ROOT / (
    "outputs/taska-focal-verifier/held300-train-exact-top5-cpu-v2/"
    "frozen-target-free-eval.npz"
)
HELD_FOCAL_METADATA = HELD_FOCAL_ARCHIVE.with_suffix(".json")
FRESH_COST_ARCHIVE = PROJECT_ROOT / (
    "outputs/taska-protected-tail/fresh-held32-mps-v1/"
    "frozen-target-free-eval.npz"
)
FRESH_COST_METADATA = FRESH_COST_ARCHIVE.with_suffix(".json")
FRESH_LAYOUT_ARCHIVE = PROJECT_ROOT / (
    "outputs/taska-fresh32-leader-confirmation/fresh-held32-mps-v1/"
    "frozen-target-free-eval.npz"
)
FRESH_LAYOUT_METADATA = FRESH_LAYOUT_ARCHIVE.with_suffix(".json")

EXPECTED_SHA256 = {
    LOCAL_LAYOUT_ARCHIVE: "e615659915d6bb17710403833b46b78d64916e4910afd7effa834a9f46d98e27",
    LOCAL_METADATA: "78a10b5f3581ef89a1dadae284906cd6f85ec7708250154b4ef4b819cd01a62c",
    HELD_LAYOUT_ARCHIVE: "784a459f6baaaa2d16fd5e1c269c0a50ff456b8c7071ba1cf6035e20be6808f1",
    HELD_LAYOUT_METADATA: "cbe6a774fb3fd3a6095e0124c5f484736ede963c0f6682d93ac49d67fe7c384a",
    HELD_COST_ARCHIVE: "0876d7acd6be7ebe863f831de568586c837da52cc603df4ff8d7c5a6b0d441df",
    HELD_COST_METADATA: "91710486233f45bda2b8aab019c508d1e6a0f75e282a6ce9a47aee93d9bf0a8d",
    HELD_FOCAL_ARCHIVE: "7d4ad494ab572d1ac3c94ab73a49b54e80b26baba489dfbd56f732a5c43394c5",
    HELD_FOCAL_METADATA: "301ba535f04b63ff8da48a0a83b5f207521d4b57f1bdbb61ceb58dbee57daff2",
    FRESH_COST_ARCHIVE: "d7b156ff1a8cdab702881242e48797b1a18f750a2d6a60f2a7d769dbfa1bffc1",
    FRESH_COST_METADATA: "1acb5d0000dd76e48fb6c079827fa2113bb56f541905fc97ced9656b8d7fe53f",
    FRESH_LAYOUT_ARCHIVE: "f3710cc3b00aaf2e75cb4127c280bc95eeeedf237f51a76ca234bac079c6f75f",
    FRESH_LAYOUT_METADATA: "311a1b3dc42bfb317a2c5cde1cee319de86ceba85622cb376fe4bfb83e2b53b1",
    PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py": (
        "97859e1f4ff6ceadf56ebccf429f2944ae7a5726631731d9f776224a667eb486"
    ),
}


@dataclass(frozen=True)
class PanelSpec:
    name: str
    cost_archive: Path
    cost_metadata: Path
    layout_archive: Path
    layout_metadata: Path
    focal_archive: Path
    focal_metadata: Path
    local_key_contract: bool = False


PANEL_SPECS = {
    "local32": PanelSpec(
        "local32",
        LOCAL_LAYOUT_ARCHIVE,
        LOCAL_METADATA,
        LOCAL_LAYOUT_ARCHIVE,
        LOCAL_METADATA,
        LOCAL_LAYOUT_ARCHIVE,
        LOCAL_METADATA,
        local_key_contract=True,
    ),
    "held32": PanelSpec(
        "held32",
        HELD_COST_ARCHIVE,
        HELD_COST_METADATA,
        HELD_LAYOUT_ARCHIVE,
        HELD_LAYOUT_METADATA,
        HELD_FOCAL_ARCHIVE,
        HELD_FOCAL_METADATA,
    ),
    "fresh32": PanelSpec(
        "fresh32",
        FRESH_COST_ARCHIVE,
        FRESH_COST_METADATA,
        FRESH_LAYOUT_ARCHIVE,
        FRESH_LAYOUT_METADATA,
        FRESH_LAYOUT_ARCHIVE,
        FRESH_LAYOUT_METADATA,
    ),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def _record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        rendered = str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        rendered = str(resolved)
    return {"path": rendered, "sha256": sha256_file(resolved)}


def _write_json(path: Path, payload: Any) -> None:
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {path}") from error


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    try:
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {path}") from error


def _require_frozen_inputs() -> None:
    for path, expected in EXPECTED_SHA256.items():
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"frozen input SHA-256 mismatch: {path}")


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


def _edges(archive: Any, prefix: str) -> tuple[RawTailEdge, ...]:
    source = np.asarray(archive[f"{prefix}__edge_source"], dtype=np.int64)
    target = np.asarray(archive[f"{prefix}__edge_target"], dtype=np.int64)
    axis = np.asarray(archive[f"{prefix}__edge_axis"], dtype=np.uint8)
    if not (source.ndim == target.ndim == axis.ndim == 1):
        raise ValueError("edge arrays must be vectors")
    if not (len(source) == len(target) == len(axis)) or not np.isin(axis, (0, 1)).all():
        raise ValueError("edge arrays are malformed")
    return tuple(
        RawTailEdge(int(s), int(t), "right" if int(a) == 0 else "down")
        for s, t, a in zip(source, target, axis, strict=True)
    )


def _identity(row: Mapping[str, Any]) -> tuple[str, str, int, str]:
    return (
        str(row["prefix"]),
        str(row["source_filename"]),
        int(row["draw_index"]),
        str(row["dirty_sha256"]),
    )


def _four_layouts(spec: PanelSpec, archive: Any, prefix: str) -> dict[str, np.ndarray]:
    if spec.name == "fresh32":
        keys = {
            "raw": "raw_layout",
            "logistic": "logistic_layout",
            "focal": "focal_layout",
            "nonlinear": "nonlinear_layout",
        }
    else:
        keys = {
            "raw": "raw_layout",
            "logistic": "logistic_layout",
            "focal": "focal_top5_layout",
            "nonlinear": "nonlinear_layout",
        }
    result = {
        name: _strict_layout(archive[f"{prefix}__{key}"]) for name, key in keys.items()
    }
    if tuple(result) != FOUR_ARMS:
        raise RuntimeError("four-arm insertion order changed")
    return result


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


def _freeze_panel(
    *,
    output_dir: Path,
    spec: PanelSpec,
    arrays: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path, Path]:
    stage_dir = output_dir / spec.name
    stage_dir.mkdir(parents=True, exist_ok=False)
    archive = stage_dir / "frozen-target-free-eval.npz"
    metadata = stage_dir / "frozen-target-free-eval.json"
    freeze = stage_dir / "pre-score-freeze.json"
    _write_npz(archive, arrays)
    _write_json(
        metadata,
        {
            "schema": "aiijc-taska-focal-gated-tail-target-free-v1",
            "panel": spec.name,
            "contains_exact_references_or_labels": False,
            "contains_target_ids_or_source_grid_coordinates": False,
            "focal_mode": "train_exact_top5",
            "focal_logit_threshold": FOCAL_PROTECTION_LOGIT_THRESHOLD,
            "threshold_or_budget_sweep": False,
            "tail_max_swaps": TAIL_SWAPS,
            "tail_minimum_gain": MINIMUM_GAIN,
            "all_layouts_strict_original_tile_permutations": True,
            "rows": list(rows),
        },
    )
    sources = {
        "candidate_archive": archive,
        "candidate_metadata": metadata,
        "runner": Path(__file__).resolve(),
        "focal_gated_tail_module": PROJECT_ROOT
        / "src/aiijc_puzzle/taska_focal_gated_protected_tail.py",
        "protected_tail_module": PROJECT_ROOT
        / "src/aiijc_puzzle/taska_protected_tail_polish.py",
        "raw_solver": PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py",
        "cost_archive": spec.cost_archive,
        "cost_metadata": spec.cost_metadata,
        "layout_archive": spec.layout_archive,
        "layout_metadata": spec.layout_metadata,
        "focal_archive": spec.focal_archive,
        "focal_metadata": spec.focal_metadata,
    }
    _write_json(
        freeze,
        {
            "schema": "aiijc-taska-focal-gated-tail-pre-score-freeze-v1",
            "panel": spec.name,
            "created_before_exact_reference_reconstruction": True,
            "contains_evaluation_references_or_labels": False,
            "artifacts": {name: _record(path) for name, path in sources.items()},
        },
    )
    return archive, metadata, freeze


def _generate_panel(output_dir: Path, spec: PanelSpec) -> tuple[Path, Path, Path]:
    metadata_payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (spec.cost_metadata, spec.layout_metadata, spec.focal_metadata)
    ]
    row_sets = [payload["rows"] for payload in metadata_payloads]
    if any(len(rows) != 32 for rows in row_sets):
        raise ValueError(f"{spec.name} must contain exactly 32 frozen cases")
    arrays: dict[str, np.ndarray] = {}
    frozen_rows: list[dict[str, Any]] = []
    with ExitStack() as stack:
        cost_archive = stack.enter_context(np.load(spec.cost_archive, allow_pickle=False))
        layout_archive = stack.enter_context(
            np.load(spec.layout_archive, allow_pickle=False)
        )
        focal_archive = stack.enter_context(np.load(spec.focal_archive, allow_pickle=False))
        for index, aligned_rows in enumerate(zip(*row_sets, strict=True)):
            identities = [_identity(row) for row in aligned_rows]
            if len(set(identities)) != 1:
                raise RuntimeError(f"{spec.name} frozen row identities differ")
            prefix, source, draw, dirty_sha = identities[0]
            right = _finite_matrix(cost_archive, f"{prefix}__cost_right")
            down = _finite_matrix(cost_archive, f"{prefix}__cost_down")
            edges = _edges(cost_archive, prefix)
            logits = np.asarray(
                focal_archive[f"{prefix}__focal_logits"], dtype=np.float64
            )
            if logits.shape != (len(edges),) or not np.isfinite(logits).all():
                raise ValueError(f"{spec.name} focal logits are not edge-aligned")
            layouts = _four_layouts(spec, layout_archive, prefix)
            selection = select_lowest_taska_seam_cost_layout(
                layouts, right, down, grid=GRID
            )
            pre_tail = _strict_layout(selection.layout)
            if spec.name == "fresh32":
                known_pre_tail = _strict_layout(
                    layout_archive[f"{prefix}__portfolio_layout"]
                )
                if not np.array_equal(pre_tail, known_pre_tail):
                    raise RuntimeError("fresh four-arm all-bond selection changed")
            control = polish_unprotected_taska_tail(
                pre_tail,
                right,
                down,
                edges,
                grid=GRID,
                max_swaps=TAIL_SWAPS,
                minimum_gain=MINIMUM_GAIN,
            )
            candidate = polish_taska_tail_with_focal_gate(
                pre_tail, right, down, edges, logits, grid=GRID
            )
            known_control_key = (
                "portfolio_tail96_layout"
                if spec.name == "fresh32"
                else "four_arm_tail96_layout"
            )
            known_control = _strict_layout(layout_archive[f"{prefix}__{known_control_key}"])
            if not np.array_equal(control.layout, known_control):
                raise RuntimeError(f"{spec.name} current tail96 replay changed")
            if candidate.diagnostics.tail.final_total_cost > (
                candidate.diagnostics.tail.initial_total_cost
                + 1e-9 * max(1.0, abs(candidate.diagnostics.tail.initial_total_cost))
            ):
                raise RuntimeError("focal-gated tail increased original all-bond cost")
            for arm, layout in (
                ("pre_tail", pre_tail),
                (ARMS[0], control.layout),
                (ARMS[1], candidate.layout),
            ):
                arrays[f"{prefix}__{arm}_layout"] = _strict_layout(layout)
            arrays[f"{prefix}__focal_logits"] = np.asarray(logits, dtype=np.float32)
            arrays[f"{prefix}__focal_keep_mask"] = np.asarray(
                logits >= FOCAL_PROTECTION_LOGIT_THRESHOLD, dtype=np.uint8
            )
            frozen_rows.append(
                {
                    "prefix": prefix,
                    "source_filename": source,
                    "draw_index": draw,
                    "dirty_sha256": dirty_sha,
                    "candidate_edge_count": len(edges),
                    "four_arm_choice": selection.choice,
                    "four_arm_total_costs": dict(selection.total_costs),
                    "pre_tail_total_cost": total_taska_adjacent_seam_cost(
                        pre_tail, right, down, grid=GRID
                    ),
                    "control": asdict(control.diagnostics),
                    "candidate": asdict(candidate.diagnostics),
                }
            )
            print(
                json.dumps(
                    {
                        "event": "focal_gated_tail_target_free",
                        "panel": spec.name,
                        "case": index + 1,
                        "case_count": 32,
                    }
                ),
                flush=True,
            )
    return _freeze_panel(
        output_dir=output_dir, spec=spec, arrays=arrays, rows=frozen_rows
    )


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
        "case_wins_ties_losses": {
            "wins": sum(value > 0 for value in values),
            "ties": sum(value == 0 for value in values),
            "losses": sum(value < 0 for value in values),
        },
    }


def _summarize(
    rows: Sequence[Mapping[str, Any]],
    frozen_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    metrics = ("satisfied_adjacent_pairs", "adjacency_recall", "exact_tiles")
    summary: dict[str, Any] = {
        "pair_denominator": PAIR_DENOMINATOR,
        "arms": {
            arm: {
                metric: float(np.mean([row["metrics"][arm][metric] for row in rows]))
                for metric in metrics
            }
            for arm in ARMS
        },
        "four_arm_choice_counts": dict(
            Counter(row["four_arm_choice"] for row in frozen_rows)
        ),
        "target_free_diagnostics": {
            "mean_harvested_edges": float(
                np.mean([row["candidate"]["harvested_edge_count"] for row in frozen_rows])
            ),
            "mean_focal_kept_edges": float(
                np.mean([row["candidate"]["focal_kept_edge_count"] for row in frozen_rows])
            ),
            "mean_control_protected_tiles": float(
                np.mean([row["control"]["protected_tile_count"] for row in frozen_rows])
            ),
            "mean_candidate_protected_tiles": float(
                np.mean(
                    [row["candidate"]["tail"]["protected_tile_count"] for row in frozen_rows]
                )
            ),
            "mean_control_accepted_swaps": float(
                np.mean([row["control"]["accepted_swap_count"] for row in frozen_rows])
            ),
            "mean_candidate_accepted_swaps": float(
                np.mean(
                    [row["candidate"]["tail"]["accepted_swap_count"] for row in frozen_rows]
                )
            ),
        },
    }
    sources = [str(row["source_filename"]) for row in rows]
    summary["candidate_minus_control"] = {
        metric: _cluster_ci(
            [
                float(row["metrics"][ARMS[1]][metric])
                - float(row["metrics"][ARMS[0]][metric])
                for row in rows
            ],
            sources,
            seed=BOOTSTRAP_SEED + index,
        )
        for index, metric in enumerate(metrics)
    }
    return summary


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
            scored.append(
                {
                    "prefix": prefix,
                    "source_filename": source,
                    "draw_index": draw,
                    "metrics": {
                        arm: _layout_metrics(
                            _strict_layout(candidate[f"{prefix}__{arm}_layout"]),
                            reference,
                        )
                        for arm in ARMS
                    },
                }
            )
    return scored, _summarize(scored, frozen_rows)


def _run_panel(
    *,
    output_dir: Path,
    spec: PanelSpec,
    lookup: Mapping[str, Mapping[str, Any]],
    cache: Any,
) -> dict[str, Any]:
    started = perf_counter()
    archive, metadata, freeze = _generate_panel(output_dir, spec)
    rows, summary = _score_panel(
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


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _require_frozen_inputs()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    config, _, _ = finetune._load_config(finetune.DEFAULT_CONFIG)
    lookup = finetune._manifest_lookup(config)
    cache = finetune.CleanTileCache(args.targets.resolve(), maximum_boards=2)
    local = _run_panel(
        output_dir=output_dir,
        spec=PANEL_SPECS["local32"],
        lookup=lookup,
        cache=cache,
    )
    local_delta = local["summary"]["candidate_minus_control"][
        "satisfied_adjacent_pairs"
    ]["mean"]
    local["gate_passed"] = local_delta >= 0.0
    held: dict[str, Any] = {
        "status": "not-run-local-gate-failed",
        "gate_passed": False,
    }
    fresh: dict[str, Any] = {"status": "not-run-held-gate-closed"}
    if local["gate_passed"]:
        held = _run_panel(
            output_dir=output_dir,
            spec=PANEL_SPECS["held32"],
            lookup=lookup,
            cache=cache,
        )
        held_delta = held["summary"]["candidate_minus_control"][
            "satisfied_adjacent_pairs"
        ]["mean"]
        held["gate_passed"] = held_delta >= 0.5 and local_delta >= 0.0
        if held["gate_passed"]:
            fresh = _run_panel(
                output_dir=output_dir,
                spec=PANEL_SPECS["fresh32"],
                lookup=lookup,
                cache=cache,
            )
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "protocol": {
            "single_fixed_threshold": True,
            "focal_mode": "train_exact_top5",
            "focal_logit_threshold": FOCAL_PROTECTION_LOGIT_THRESHOLD,
            "threshold_or_budget_sweep": False,
            "tail_max_swaps": TAIL_SWAPS,
            "tail_minimum_gain": MINIMUM_GAIN,
            "local_panel_touched_by_threshold_diagnostic": True,
            "local_gate": "candidate pair delta >= 0",
            "held_gate": "candidate pair delta >= +0.5 and no local collapse",
            "fresh_open_rule": "only after held gate passes",
        },
        "local32": local,
        "held32": held,
        "fresh32": fresh,
        "runtime_seconds": perf_counter() - started,
        "legality": {
            "inference_uses_targets_or_labels": False,
            "targets_used_only_after_candidate_layout_freeze_for_scoring": True,
            "matcher_rerun": False,
            "candidate_membership_changed": False,
            "original_taska_costs_used_for_all_tail_swaps": True,
            "all_outputs_strict_permutations_of_original_upright_tiles": True,
            "tile_rotation_warp_replacement_or_synthesis": False,
            "competition_test_used": False,
        },
        "artifacts": {
            "runner": _record(Path(__file__).resolve()),
            "module": _record(
                PROJECT_ROOT
                / "src/aiijc_puzzle/taska_focal_gated_protected_tail.py"
            ),
            "raw_solver": _record(
                PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py"
            ),
        },
    }
    _write_json(output_dir / "report.json", report)
    print(json.dumps({key: report[key] for key in ("local32", "held32", "fresh32")}, indent=2))


if __name__ == "__main__":
    main()

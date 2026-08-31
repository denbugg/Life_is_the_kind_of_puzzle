#!/usr/bin/env python3
"""Evaluate one preregistered exact row-phase DP over confirmed TASKA fusion."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from aiijc_puzzle.layout_evaluation import evaluate_layout
from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.taska_pair_pipeline import PAIR_DENOMINATOR
from aiijc_puzzle.taska_row_phase_dp import solve_taska_row_phase_dp

try:
    from scripts import run_taska_focal_current_finetune as finetune
    from scripts import run_taska_selective_fullres_fusion as fusion
except ModuleNotFoundError:
    import run_taska_focal_current_finetune as finetune
    import run_taska_selective_fullres_fusion as fusion

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/taska-row-phase-dp/fixed-v1"
DEFAULT_CONFIG = PROJECT_ROOT / "configs/taska_row_phase_dp_v1.json"
GRID = 24
LOCAL_GATE = 0.0
HELD_GATE = 0.0
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 2_026_083_113
CONTROL_ARM = "confirmed_six_arm_fusion"
CANDIDATE_ARM = "row_phase_dp"
ARMS = (CONTROL_ARM, CANDIDATE_ARM)


@dataclass(frozen=True)
class PanelSpec:
    name: str
    case_count: int
    parent: fusion.PanelSpec
    fusion_archive: Path
    fusion_metadata: Path
    fusion_freeze: Path


FUSION_ROOT = PROJECT_ROOT / "outputs/taska-selective-fullres-union-fusion/fixed-v1"
FUSION_SHA256 = {
    "local32": (
        "1b17c4a52ae80b58f973ee8aaffd20d0e1d9a125c1ac5e3acdc66f31abddf7df",
        "106ac31d166c1b244a498c3cc76f59d4730601e6fba3a35fa6721eb7f18befa1",
        "3b35db324f46a0368cad5c3f6570c08f9631560fe1f5f47f14defc77b8689720",
    ),
    "held32": (
        "6cfb766c1e693a2fec535d683f187a89f2d63632a282ff199e6aa708caafe469",
        "f37d23bd44c1565ae560c46ed6b6f33b4500b52168147ba114ff8debc59f0bf4",
        "aa5b53abbb3fe5b20900a2102e144f024bf563e42dbe37edd1086811515178bc",
    ),
    "fresh32": (
        "75a9359eb3ac798096437c22e269c8374a0a38bb01f8e7f9fa9745bd054180cb",
        "c65d7e332460001d67b2dc2052a2dd3a2e6c62d08f3a936c7723ecec6dac6794",
        "7fca88f9ea4489bf64d73a060127af73e1adaa598089eadfb79c1597627d5e93",
    ),
}


def _panel(name: str) -> PanelSpec:
    root = FUSION_ROOT / name
    return PanelSpec(
        name=name,
        case_count=32,
        parent=fusion.PANELS[name],
        fusion_archive=root / "frozen-target-free-eval.npz",
        fusion_metadata=root / "frozen-target-free-eval.json",
        fusion_freeze=root / "pre-score-freeze.json",
    )


PANELS = {name: _panel(name) for name in ("local32", "held32", "fresh32")}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
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


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    resolved = path.resolve()
    sidecar = resolved.with_suffix(resolved.suffix + ".sha256")
    if not resolved.is_file() or not sidecar.is_file():
        raise FileNotFoundError("signed row-phase preregistration is missing")
    digest = sha256_file(resolved)
    if sidecar.read_text(encoding="utf-8").split()[0] != digest:
        raise ValueError("row-phase preregistration SHA-256 mismatch")
    config = json.loads(resolved.read_text(encoding="utf-8"))
    required = {
        "candidate": "exact Viterbi minimum over 24 cyclic phases of each fixed row",
        "objective": "original TASKA all-1104-bond raw seam cost",
        "control": "exact frozen confirmed six-arm fusion final layout",
        "local_pair_gate": LOCAL_GATE,
        "held_pair_gate": HELD_GATE,
        "no_sweep": True,
    }
    for key, value in required.items():
        if config.get(key) != value:
            raise ValueError(f"row-phase preregistration contract mismatch: {key}")
    for relative, expected in config["fixed_source_sha256"].items():
        target = PROJECT_ROOT / relative
        if not target.is_file() or sha256_file(target) != expected:
            raise ValueError(f"signed candidate source changed: {relative}")
    return config, digest


def _require_inputs() -> None:
    fusion._require_inputs()
    for name, spec in PANELS.items():
        for path, expected in zip(
            (spec.fusion_archive, spec.fusion_metadata, spec.fusion_freeze),
            FUSION_SHA256[name],
            strict=True,
        ):
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError(f"frozen fusion SHA-256 mismatch: {path}")
        fusion._validate_freeze(spec.fusion_freeze)


def _rows(path: Path, count: int) -> list[Mapping[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8")).get("rows")
    if not isinstance(rows, list) or len(rows) < count:
        raise ValueError(f"{path} contains fewer than {count} rows")
    return rows[:count]


def _aligned_rows(spec: PanelSpec) -> list[tuple[Mapping[str, Any], ...]]:
    parent = fusion._aligned_rows(replace(spec.parent, case_count=spec.case_count))
    fused = _rows(spec.fusion_metadata, spec.case_count)
    identity = ("prefix", "source_filename", "draw_index", "dirty_sha256")
    result: list[tuple[Mapping[str, Any], ...]] = []
    for records, final in zip(parent, fused, strict=True):
        if any(records[0].get(field) != final.get(field) for field in identity):
            raise RuntimeError(f"{spec.name} frozen row identity mismatch")
        result.append((*records, final))
    return result


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
        "case_wins_ties_losses": {
            "wins": sum(value > 0 for value in values),
            "ties": sum(value == 0 for value in values),
            "losses": sum(value < 0 for value in values),
        },
    }


def _summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = ("satisfied_adjacent_pairs", "adjacency_recall", "exact_tiles")
    sources = [str(row["source_filename"]) for row in rows]
    arms = {
        arm: {
            metric: float(np.mean([row["metrics"][arm][metric] for row in rows]))
            for metric in metrics
        }
        for arm in ARMS
    }
    deltas = {
        metric: _cluster_ci(
            [
                row["metrics"][CANDIDATE_ARM][metric]
                - row["metrics"][CONTROL_ARM][metric]
                for row in rows
            ],
            sources,
            seed=BOOTSTRAP_SEED + index,
        )
        for index, metric in enumerate(metrics)
    }
    return {
        "case_count": len(rows),
        "pair_denominator": PAIR_DENOMINATOR,
        "arms": arms,
        "candidate_minus_control": deltas,
        "target_free_diagnostics": {
            "mean_changed_rows": float(np.mean([row["changed_row_count"] for row in rows])),
            "mean_raw_cost_improvement": float(
                np.mean([row["raw_cost_improvement"] for row in rows])
            ),
            "changed_layout_count": sum(row["changed_row_count"] > 0 for row in rows),
            "objective_monotone_count": sum(row["objective_monotone"] for row in rows),
        },
    }


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


def _validate_freeze(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("created_before_exact_reference_reconstruction") is not True:
        raise RuntimeError("pre-score timing contract changed")
    if payload.get("contains_evaluation_references_or_labels") is not False:
        raise RuntimeError("pre-score freeze contains labels")
    for name, record in payload["artifacts"].items():
        artifact = Path(record["path"])
        if not artifact.is_absolute():
            artifact = PROJECT_ROOT / artifact
        if not artifact.is_file() or sha256_file(artifact) != record["sha256"]:
            raise RuntimeError(f"pre-score artifact changed: {name}")


def _run_panel(
    spec: PanelSpec,
    *,
    output_dir: Path,
    config_path: Path,
    lookup: Mapping[str, Mapping[str, Any]] | None,
    cache: Any | None,
    target_free_only: bool,
) -> dict[str, Any]:
    aligned = _aligned_rows(spec)
    arrays: dict[str, np.ndarray] = {}
    frozen_rows: list[dict[str, Any]] = []
    stage = output_dir / spec.name
    stage.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    with (
        np.load(spec.parent.base_archive, allow_pickle=False) as base,
        np.load(spec.fusion_archive, allow_pickle=False) as fused,
    ):
        for index, records in enumerate(aligned):
            row = records[-1]
            prefix = str(row["prefix"])
            control = np.asarray(
                fused[f"{prefix}__combined_union_candidate_layout"], dtype=np.int32
            )
            solved = solve_taska_row_phase_dp(
                control,
                fusion._matrix(base, f"{prefix}__cost_right"),
                fusion._matrix(base, f"{prefix}__cost_down"),
            )
            arrays[f"{prefix}__{CONTROL_ARM}_layout"] = control
            arrays[f"{prefix}__{CANDIDATE_ARM}_layout"] = solved.layout
            frozen_rows.append(
                {
                    "prefix": prefix,
                    "source_filename": str(row["source_filename"]),
                    "draw_index": int(row["draw_index"]),
                    "dirty_sha256": str(row["dirty_sha256"]),
                    "row_phases": list(solved.diagnostics.phases),
                    "changed_row_count": solved.diagnostics.changed_row_count,
                    "raw_cost_before": solved.diagnostics.before_total_cost,
                    "raw_cost_after": solved.diagnostics.after_total_cost,
                    "raw_cost_improvement": solved.diagnostics.total_cost_improvement,
                    "objective_monotone": solved.diagnostics.objective_monotone,
                }
            )
            print(
                json.dumps(
                    {
                        "event": f"{spec.name}_row_phase_target_free",
                        "case": index + 1,
                        "case_count": len(aligned),
                        "changed_rows": solved.diagnostics.changed_row_count,
                        "raw_cost_improvement": solved.diagnostics.total_cost_improvement,
                    }
                ),
                flush=True,
            )
    archive = stage / "frozen-target-free-eval.npz"
    metadata = stage / "frozen-target-free-eval.json"
    freeze = stage / "pre-score-freeze.json"
    _write_npz(archive, arrays)
    _write_json(
        metadata,
        {
            "schema": "aiijc-taska-row-phase-dp-target-free-v1",
            "contains_exact_references_or_candidate_labels": False,
            "candidate_family": "independent cyclic phase for each fixed row",
            "exact_dynamic_program": True,
            "objective": "original TASKA all-1104-bond raw seam cost",
            "strict_original_upright_permutations": True,
            "rows": frozen_rows,
        },
    )
    artifacts = {
        "archive": _record(archive),
        "metadata": _record(metadata),
        "preregistration": _record(config_path),
        "runner": _record(Path(__file__).resolve()),
        "solver": _record(PROJECT_ROOT / "src/aiijc_puzzle/taska_row_phase_dp.py"),
        "fusion_archive": _record(spec.fusion_archive),
        "fusion_metadata": _record(spec.fusion_metadata),
        "fusion_parent_freeze": _record(spec.fusion_freeze),
        "base_archive": _record(spec.parent.base_archive),
        "base_metadata": _record(spec.parent.base_metadata),
    }
    _write_json(
        freeze,
        {
            "schema": "aiijc-taska-row-phase-dp-pre-score-freeze-v1",
            "created_before_exact_reference_reconstruction": True,
            "contains_evaluation_references_or_labels": False,
            "artifacts": artifacts,
        },
    )
    payload: dict[str, Any] = {
        "status": "target-free-smoke" if target_free_only else "complete",
        "target_free_summary": {
            "case_count": len(frozen_rows),
            "mean_changed_rows": float(
                np.mean([row["changed_row_count"] for row in frozen_rows])
            ),
            "changed_layout_count": sum(row["changed_row_count"] > 0 for row in frozen_rows),
            "objective_monotone_count": sum(row["objective_monotone"] for row in frozen_rows),
        },
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
        _validate_freeze(freeze)
        scored: list[dict[str, Any]] = []
        with np.load(archive, allow_pickle=False) as frozen:
            for row in frozen_rows:
                prefix = row["prefix"]
                source = row["source_filename"]
                draw = row["draw_index"]
                dirty = finetune._dirty_case(cache, lookup[source], source, draw)
                if finetune._dirty_sha256(dirty.dirty_tiles) != row["dirty_sha256"]:
                    raise RuntimeError("scoring recreated different dirty bytes")
                reference = finetune._reference(
                    cache, lookup[source], source, draw, dirty.dirty_tiles
                )
                scored.append(
                    {
                        **row,
                        "metrics": {
                            arm: _layout_metrics(
                                frozen[f"{prefix}__{arm}_layout"], reference
                            )
                            for arm in ARMS
                        },
                    }
                )
        payload.update({"rows": scored, "summary": _summarize(scored)})
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    _require_inputs()
    config, config_sha256 = _load_config(args.config)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    if args.smoke_one:
        smoke = replace(PANELS["local32"], name="smoke1", case_count=1)
        result = _run_panel(
            smoke,
            output_dir=output_dir,
            config_path=args.config.resolve(),
            lookup=None,
            cache=None,
            target_free_only=True,
        )
        report = {
            "schema": "aiijc-taska-row-phase-dp-report-v1",
            "status": "target-free-smoke",
            "preregistration_sha256": config_sha256,
            "smoke1": result,
            "competition_test_accessed": False,
        }
        _write_json(output_dir / "report.json", report)
        print(json.dumps(report, indent=2))
        return report
    source_config, _, _ = finetune._load_config(finetune.DEFAULT_CONFIG)
    lookup = finetune._manifest_lookup(source_config)
    cache = finetune.CleanTileCache(args.targets.resolve(), maximum_boards=2)
    local = _run_panel(
        PANELS["local32"],
        output_dir=output_dir,
        config_path=args.config.resolve(),
        lookup=lookup,
        cache=cache,
        target_free_only=False,
    )
    local_delta = local["summary"]["candidate_minus_control"][
        "satisfied_adjacent_pairs"
    ]["mean"]
    held: dict[str, Any] = {"status": "skipped_by_local_gate"}
    fresh: dict[str, Any] = {"status": "skipped_by_local_or_held_gate"}
    if local_delta >= LOCAL_GATE:
        held = _run_panel(
            PANELS["held32"],
            output_dir=output_dir,
            config_path=args.config.resolve(),
            lookup=lookup,
            cache=cache,
            target_free_only=False,
        )
        held_delta = held["summary"]["candidate_minus_control"][
            "satisfied_adjacent_pairs"
        ]["mean"]
        if held_delta >= HELD_GATE:
            fresh = _run_panel(
                PANELS["fresh32"],
                output_dir=output_dir,
                config_path=args.config.resolve(),
                lookup=lookup,
                cache=cache,
                target_free_only=False,
            )
        else:
            fresh = {"status": "skipped_by_held_gate"}
    report = {
        "schema": "aiijc-taska-row-phase-dp-report-v1",
        "status": "complete",
        "protocol": config,
        "preregistration_sha256": config_sha256,
        "local32": local,
        "held32": held,
        "fresh32": fresh,
        "legality": {
            "strict_original_upright_tile_permutations": True,
            "pixels_changed_rotated_warped_replaced_or_postprocessed": False,
            "targets_used_only_after_candidate_freeze": True,
            "competition_test_accessed": False,
            "production_modified": False,
        },
        "artifacts": {
            "runner": _record(Path(__file__).resolve()),
            "solver": _record(PROJECT_ROOT / "src/aiijc_puzzle/taska_row_phase_dp.py"),
            "preregistration": _record(args.config.resolve()),
        },
    }
    _write_json(output_dir / "report.json", report)
    print(json.dumps({name: report[name] for name in PANELS}, indent=2))
    return report


if __name__ == "__main__":
    run(parse_args())

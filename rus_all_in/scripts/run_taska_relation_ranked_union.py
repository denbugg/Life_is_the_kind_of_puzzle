#!/usr/bin/env python3
"""Replay one fixed HGB-ranked six-arm relation union on opened TASKA panels."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.taska_pair_pipeline import PAIR_DENOMINATOR, SOLVER_CONFIG
from aiijc_puzzle.taska_relation_ranked_union import solve_relation_ranked_union
from aiijc_puzzle.taska_relation_truth_selector import (
    FEATURE_NAMES,
    MODEL_PARAMETERS,
    RelationFeatureBoard,
    realised_edges,
    select_relation_truth_layout,
)
from aiijc_puzzle.taska_selective_fullres_fusion import FUSION_ARM_NAMES

try:
    from scripts import run_taska_confirmed_arm_portfolio as portfolio
    from scripts import run_taska_focal_current_finetune as finetune
    from scripts import run_taska_relation_truth_selector as relation
except ModuleNotFoundError:
    import run_taska_confirmed_arm_portfolio as portfolio
    import run_taska_focal_current_finetune as finetune
    import run_taska_relation_truth_selector as relation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/taska_relation_ranked_union_v1.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/taska-relation-ranked-union/fixed-v1"
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
RELATION_ROOT = PROJECT_ROOT / "outputs/taska-relation-truth-selector/fixed-v1"
MODEL_PATH = (
    RELATION_ROOT
    / "model-local32-held32/frozen-relation-classifier.pkl"
)
MODEL_SHA256 = "ec4eca99243cdc6be20104d789b9e5d5598b79fa0d1b7e69bc37314375ad8c6b"
CONTROL = "relation_truth_selector"
CANDIDATE = "relation_ranked_all_edge_union"
ARMS = (CONTROL, CANDIDATE)
PANELS = ("local32", "held32", "fresh32")
LOCAL_PAIR_GATE = 0.0
HELD_PAIR_GATE = 0.0
FRESH_PAIR_GATE = 1.0
FRESH_PAIR_CI_LOWER_GATE = 0.0
FRESH_EXACT_GATE = -1.0
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 2_026_083_155
CONFIG_SCHEMA = "aiijc-taska-relation-ranked-union-config-v1"
REPORT_SCHEMA = "aiijc-taska-relation-ranked-union-report-v1"


@dataclass(frozen=True)
class PanelPaths:
    name: str
    relation_archive: Path
    relation_metadata: Path
    relation_freeze: Path
    base_archive: Path
    base_metadata: Path


def _panel(name: str) -> PanelPaths:
    spec = portfolio.PANELS[name]
    root = RELATION_ROOT / name
    return PanelPaths(
        name=name,
        relation_archive=root / "frozen-target-free-relations.npz",
        relation_metadata=root / "frozen-target-free-relations.json",
        relation_freeze=root / "pre-score-freeze.json",
        base_archive=spec.parent.base_archive,
        base_metadata=spec.parent.base_metadata,
    )


PANEL_PATHS = {name: _panel(name) for name in PANELS}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--target-free-smoke", action="store_true")
    return parser.parse_args(argv)


def _project_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {"path": _project_path(resolved), "sha256": sha256_file(resolved)}


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
    sidecar = Path(f"{resolved}.sha256")
    if not resolved.is_file() or not sidecar.is_file():
        raise FileNotFoundError("signed relation-ranked-union config is missing")
    digest = sha256_file(resolved)
    if sidecar.read_text(encoding="utf-8").split()[0] != digest:
        raise ValueError("relation-ranked-union config signature mismatch")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    required = {
        "schema": CONFIG_SCHEMA,
        "model_sha256": MODEL_SHA256,
        "arm_order": list(FUSION_ARM_NAMES),
        "relation_feature_names": list(FEATURE_NAMES),
        "model_parameters": MODEL_PARAMETERS,
        "candidate": {
            "score_occurrences": "all 1104 realised relations in each of six post-tail arms",
            "deduplicate": "maximum positive-class probability per (axis,source,target)",
            "deduplicate_exact_tie": "earliest fixed arm then relation order",
            "order": "all unique edges descending probability; fixed occurrence order ties",
            "threshold": None,
            "top_k": None,
            "solver": "unchanged solve_prioritized_raw_tail_global",
            "solver_config": asdict(SOLVER_CONFIG),
            "cost_matrices_and_fill": "unchanged frozen parent",
        },
        "control": "confirmed whole-arm relation_truth_selector",
        "panel_order": list(PANELS),
        "gates": {
            "local_pair_delta_minimum": LOCAL_PAIR_GATE,
            "held_pair_delta_minimum": HELD_PAIR_GATE,
            "fresh_pair_delta_minimum": FRESH_PAIR_GATE,
            "fresh_pair_source_ci95_lower_minimum": FRESH_PAIR_CI_LOWER_GATE,
            "fresh_exact_delta_minimum": FRESH_EXACT_GATE,
            "all_outputs_strict": True,
        },
        "no_threshold_topk_weight_parameter_or_model_sweep": True,
    }
    for key, value in required.items():
        if payload.get(key) != value:
            raise ValueError(f"signed candidate contract mismatch: {key}")
    for registry in ("fixed_sources", "frozen_inputs"):
        records = payload.get(registry)
        if not isinstance(records, Mapping):
            raise ValueError(f"signed {registry} registry is missing")
        for name, record in records.items():
            if not isinstance(record, Mapping):
                raise ValueError(f"malformed {registry} record: {name}")
            artifact = PROJECT_ROOT / str(record.get("path"))
            if not artifact.is_file() or sha256_file(artifact) != record.get("sha256"):
                raise ValueError(f"signed {registry} artifact changed: {name}")
    return payload, digest


def _rows(path: Path) -> tuple[Mapping[str, Any], ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != 32:
        raise ValueError(f"expected exactly 32 frozen rows: {path}")
    return tuple(rows)


def _load_boards(
    paths: PanelPaths,
) -> tuple[tuple[RelationFeatureBoard, ...], tuple[Mapping[str, Any], ...]]:
    relation._validate_freeze(paths.relation_freeze)
    rows = _rows(paths.relation_metadata)
    spec = portfolio.PANELS[paths.name]
    aligned = portfolio._aligned_rows(spec)
    identity = ("prefix", "source_filename", "draw_index", "dirty_sha256")
    boards: list[RelationFeatureBoard] = []
    with np.load(paths.relation_archive, allow_pickle=False) as archive:
        for index, (row, parent_rows) in enumerate(zip(rows, aligned, strict=True)):
            if any(row[field] != parent_rows[0][field] for field in identity):
                raise RuntimeError(f"{paths.name} row {index} is not parent-aligned")
            prefix = str(row["prefix"])
            layouts = tuple(
                np.asarray(archive[f"{prefix}__{arm}_layout"])
                for arm in FUSION_ARM_NAMES
            )
            boards.append(
                RelationFeatureBoard(
                    layouts=layouts,
                    edges=tuple(realised_edges(layout) for layout in layouts),
                    features=np.asarray(archive[f"{prefix}__features"], dtype=np.float64),
                    control_choice=str(row["control_choice"]),
                )
            )
    return tuple(boards), rows


def _matrix(archive: Any, key: str) -> np.ndarray:
    matrix = np.asarray(archive[key], dtype=np.float64)
    if matrix.shape != (576, 576) or not np.isfinite(matrix).all():
        raise ValueError(f"{key} is not one finite 576x576 cost matrix")
    return np.ascontiguousarray(matrix)


def _freeze_panel(
    paths: PanelPaths,
    *,
    output_dir: Path,
    config_path: Path,
    model: Any,
    case_count: int = 32,
) -> tuple[Path, Path, Path, tuple[Mapping[str, Any], ...]]:
    boards, rows = _load_boards(paths)
    if case_count not in (1, 32):
        raise ValueError("case_count must be 1 or the complete fixed panel")
    stage = output_dir / paths.name
    stage.mkdir(parents=True, exist_ok=False)
    arrays: dict[str, np.ndarray] = {}
    metadata_rows: list[dict[str, Any]] = []
    with np.load(paths.base_archive, allow_pickle=False) as base:
        for index, (board, row) in enumerate(
            zip(boards[:case_count], rows[:case_count], strict=True)
        ):
            prefix = str(row["prefix"])
            right = _matrix(base, f"{prefix}__cost_right")
            down = _matrix(base, f"{prefix}__cost_down")
            control_arm, control_layout, control_scores = select_relation_truth_layout(
                board, model
            )
            candidate = solve_relation_ranked_union(board, model, right, down)
            union = candidate.union
            arrays[f"{prefix}__{CONTROL}_layout"] = control_layout
            arrays[f"{prefix}__{CANDIDATE}_layout"] = candidate.layout
            arrays[f"{prefix}__union_probability"] = union.probabilities
            arrays[f"{prefix}__union_edge_source"] = np.asarray(
                [edge.source for edge in union.edges], dtype=np.int16
            )
            arrays[f"{prefix}__union_edge_target"] = np.asarray(
                [edge.target for edge in union.edges], dtype=np.int16
            )
            arrays[f"{prefix}__union_edge_axis"] = np.asarray(
                [edge.axis == "down" for edge in union.edges], dtype=np.uint8
            )
            arrays[f"{prefix}__winning_arm_index"] = union.winning_arm_indices
            arrays[f"{prefix}__winning_relation_index"] = union.winning_relation_indices
            metadata_rows.append(
                {
                    **row,
                    "control_arm": control_arm,
                    "control_expected_correct_scores": dict(
                        zip(FUSION_ARM_NAMES, control_scores.tolist(), strict=True)
                    ),
                    "candidate_diagnostics": candidate.diagnostics(),
                }
            )
            print(
                json.dumps(
                    {
                        "event": f"{paths.name}_relation_ranked_union_target_free",
                        "case": index + 1,
                        "case_count": case_count,
                        "unique_edges": len(union.edges),
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
            "schema": "aiijc-taska-relation-ranked-union-target-free-v1",
            "panel": paths.name,
            "contains_exact_references_or_labels": False,
            "all_unique_edges_used": True,
            "threshold": None,
            "top_k": None,
            "strict_original_upright_permutations": True,
            "rows": metadata_rows,
        },
    )
    _write_json(
        freeze,
        {
            "schema": "aiijc-taska-relation-ranked-union-pre-score-freeze-v1",
            "created_before_exact_reference_reconstruction": True,
            "contains_evaluation_references_or_labels": False,
            "artifacts": {
                "candidate_archive": _record(archive),
                "candidate_metadata": _record(metadata),
                "signed_config": _record(config_path),
                "relation_model": _record(MODEL_PATH),
                "relation_features": _record(paths.relation_archive),
                "relation_metadata": _record(paths.relation_metadata),
                "relation_freeze": _record(paths.relation_freeze),
                "base_costs": _record(paths.base_archive),
                "base_metadata": _record(paths.base_metadata),
                "candidate_source": _record(
                    PROJECT_ROOT / "src/aiijc_puzzle/taska_relation_ranked_union.py"
                ),
                "runner": _record(Path(__file__).resolve()),
            },
        },
    )
    return archive, metadata, freeze, tuple(metadata_rows)


def _validate_freeze(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("created_before_exact_reference_reconstruction") is not True:
        raise RuntimeError("candidate freeze timing contract changed")
    if payload.get("contains_evaluation_references_or_labels") is not False:
        raise RuntimeError("candidate freeze contains scoring labels")
    for name, record in payload["artifacts"].items():
        path_value = Path(str(record["path"]))
        artifact = path_value if path_value.is_absolute() else PROJECT_ROOT / path_value
        if not artifact.is_file() or sha256_file(artifact) != record["sha256"]:
            raise RuntimeError(f"candidate freeze artifact changed: {name}")


def _cluster_ci(values: Sequence[float], sources: Sequence[str], *, seed: int) -> dict[str, Any]:
    grouped: dict[str, list[float]] = {}
    for value, source in zip(values, sources, strict=True):
        if not math.isfinite(float(value)):
            raise ValueError("metric delta is non-finite")
        grouped.setdefault(source, []).append(float(value))
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
        "case_wins_ties_losses": {
            "wins": sum(value > 0 for value in values),
            "ties": sum(value == 0 for value in values),
            "losses": sum(value < 0 for value in values),
        },
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": seed,
    }


def _score_panel(
    paths: PanelPaths,
    archive: Path,
    freeze: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    lookup: Mapping[str, Mapping[str, Any]],
    cache: Any,
) -> dict[str, Any]:
    _validate_freeze(freeze)
    scored: list[dict[str, Any]] = []
    with np.load(archive, allow_pickle=False) as frozen:
        for row in rows:
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
                arm: portfolio._layout_metrics(
                    frozen[f"{prefix}__{arm}_layout"], reference
                )
                for arm in ARMS
            }
            scored.append(
                {
                    "prefix": prefix,
                    "source_filename": source,
                    "draw_index": draw,
                    "control_arm": row["control_arm"],
                    "metrics": metrics,
                }
            )
    fields = ("satisfied_adjacent_pairs", "adjacency_recall", "exact_tiles")
    sources = [str(row["source_filename"]) for row in scored]
    arms = {
        arm: {
            field: float(np.mean([row["metrics"][arm][field] for row in scored]))
            for field in fields
        }
        for arm in ARMS
    }
    delta = {
        field: _cluster_ci(
            [
                float(row["metrics"][CANDIDATE][field])
                - float(row["metrics"][CONTROL][field])
                for row in scored
            ],
            sources,
            seed=BOOTSTRAP_SEED + index,
        )
        for index, field in enumerate(fields)
    }
    return {
        "panel": paths.name,
        "exposure": (
            "in-sample/mechanical; HGB fit included this panel"
            if paths.name in {"local32", "held32"}
            else "existing opened model-selection-exposed development panel"
        ),
        "case_count": len(scored),
        "pair_denominator": PAIR_DENOMINATOR,
        "arms": arms,
        "candidate_minus_control": delta,
        "all_outputs_strict": all(
            row["metrics"][arm]["strict_original_upright_permutation"]
            for row in scored
            for arm in ARMS
        ),
        "rows": scored,
    }


def _panel_gate(name: str, summary: Mapping[str, Any]) -> bool:
    pair = summary["candidate_minus_control"]["satisfied_adjacent_pairs"]
    exact = summary["candidate_minus_control"]["exact_tiles"]
    strict = bool(summary["all_outputs_strict"])
    if name == "local32":
        return strict and float(pair["mean"]) >= LOCAL_PAIR_GATE
    if name == "held32":
        return strict and float(pair["mean"]) >= HELD_PAIR_GATE
    return bool(
        strict
        and float(pair["mean"]) >= FRESH_PAIR_GATE
        and float(pair["ci95_lower"]) >= FRESH_PAIR_CI_LOWER_GATE
        and float(exact["mean"]) >= FRESH_EXACT_GATE
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    config, config_sha256 = _load_config(args.config)
    if sha256_file(MODEL_PATH) != MODEL_SHA256:
        raise ValueError("frozen relation model changed")
    with MODEL_PATH.open("rb") as stream:
        model = pickle.load(stream)
    for name, value in MODEL_PARAMETERS.items():
        if model.get_params().get(name) != value:
            raise ValueError(f"frozen relation model parameter changed: {name}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    if args.target_free_smoke:
        paths = PANEL_PATHS["local32"]
        archive, metadata, freeze, _ = _freeze_panel(
            paths,
            output_dir=output,
            config_path=args.config.resolve(),
            model=model,
            case_count=1,
        )
        report = {
            "schema": REPORT_SCHEMA,
            "status": "target-free-smoke",
            "config_sha256": config_sha256,
            "artifacts": {
                "archive": _record(archive),
                "metadata": _record(metadata),
                "freeze": _record(freeze),
            },
            "competition_test_accessed": False,
        }
        _write_json(output / "report.json", report)
        return report

    source_config, _, _ = finetune._load_config(finetune.DEFAULT_CONFIG)
    lookup = finetune._manifest_lookup(source_config)
    cache = finetune.CleanTileCache(args.targets.resolve(), maximum_boards=2)
    summaries: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    status = "complete_all_development_gates_passed"
    for name in PANELS:
        paths = PANEL_PATHS[name]
        archive, metadata, freeze, rows = _freeze_panel(
            paths,
            output_dir=output,
            config_path=args.config.resolve(),
            model=model,
        )
        summary = _score_panel(
            paths,
            archive,
            freeze,
            rows,
            lookup=lookup,
            cache=cache,
        )
        passed = _panel_gate(name, summary)
        summary["gate_passed"] = passed
        summaries[name] = summary
        artifacts[name] = {
            "archive": _record(archive),
            "metadata": _record(metadata),
            "pre_score_freeze": _record(freeze),
        }
        if not passed:
            status = f"stopped_after_{name}_gate_failure"
            break
    all_passed = len(summaries) == len(PANELS) and all(
        summary["gate_passed"] for summary in summaries.values()
    )
    report = {
        "schema": REPORT_SCHEMA,
        "status": status,
        "config_sha256": config_sha256,
        "candidate": config["candidate"],
        "control": CONTROL,
        "development_exposure": {
            "local32": "in-sample/mechanical",
            "held32": "in-sample/mechanical",
            "fresh32": "opened/model-selection-exposed",
            "none_are_confirmation": True,
        },
        "panels": summaries,
        "all_development_gates_passed": all_passed,
        "formal_confirmation": {
            "eligible": all_passed,
            "status": (
                "must_sign_wholly_new_source16xdraw2_before_generation_or_scoring"
                if all_passed
                else "not_authorized"
            ),
            "new_roster_generated_or_scored": False,
        },
        "artifacts": artifacts,
        "legality": {
            "strict_original_upright_576_tile_permutations": True,
            "pixels_changed": False,
            "matcher_rerun": False,
            "competition_test_accessed": False,
            "production_or_submission_changed": False,
        },
        "runtime_seconds": perf_counter() - started,
    }
    _write_json(output / "report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    report = run(parse_args(argv))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

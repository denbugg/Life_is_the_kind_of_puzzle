#!/usr/bin/env python3
"""Freeze and score one signed joint-native reciprocal-head arm on FIT64.

``freeze`` reads only target-free fixed-head arrays, frozen raw side sequences,
and the already frozen relation-selector control.  ``score`` first verifies all
freeze hashes and only then materialises FIT ``target_slots`` to reconstruct the
exact reference.  There is no DEV, local, terminal, test, model-inference,
training, whole-arm-selection, sweep, Weco, or submission mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from aiijc_puzzle.joint_native_head_arm import (
    FROZEN_SOLVER_CONFIG,
    frozen_head_edges,
    reference_from_target_slots,
    solve_joint_native_head_arm,
)
from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.structured_decoder_fit_oracle import layout_metrics, strict_layout

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/joint_native_head_arm_fit_v1.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/joint-native-head-arm-fit/fixed-v1"
GRID = 24
COUNT = GRID * GRID
REQUESTED_PER_AXIS = 29
CONFIG_SCHEMA = "aiijc-joint-native-head-arm-fit-config-v1"
HEAD_SCHEMA = "aiijc-joint-reciprocal-target-free-fit-heads-v1"
CONTROL_SCHEMA = "aiijc-structured-decoder-fit-oracle-controls-v1"
FREEZE_SCHEMA = "aiijc-joint-native-head-arm-fit-freeze-v1"
METADATA_SCHEMA = "aiijc-joint-native-head-arm-fit-target-free-layouts-v1"
SCORE_SCHEMA = "aiijc-joint-native-head-arm-fit-score-v1"
ARCHIVE_NAME = "frozen-target-free-layouts.npz"
METADATA_NAME = "frozen-target-free-layouts.json"
FREEZE_NAME = "pre-score-freeze.json"
SCORE_NAME = "score.json"
BOOTSTRAP_RESAMPLES = 100000
BOOTSTRAP_BASE_SEED = 20260831
BOOTSTRAP_METRIC_SEED_STRIDE = 1009
TIE_EPSILON = 1e-15


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("validate", "freeze", "score"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
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


def _path(value: str) -> Path:
    result = Path(value)
    return result.resolve() if result.is_absolute() else (PROJECT_ROOT / result).resolve()


def _write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def _write_npz_exclusive(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def _load_signed_config(path: Path) -> tuple[dict[str, Any], str]:
    resolved = path.resolve()
    sidecar = resolved.with_suffix(resolved.suffix + ".sha256")
    if not resolved.is_file() or not sidecar.is_file():
        raise FileNotFoundError("signed joint-native FIT config is unavailable")
    digest = sha256_file(resolved)
    if sidecar.read_text(encoding="utf-8").split()[0] != digest:
        raise RuntimeError("joint-native FIT config sidecar mismatch")
    config = json.loads(resolved.read_text(encoding="utf-8"))
    if config.get("schema") != CONFIG_SCHEMA or config.get("status") != "signed-fixed-protocol":
        raise RuntimeError("joint-native FIT protocol is not signed/fixed")
    construction = config.get("construction", {})
    expected = {
        "candidate_edges": "exact frozen fixed-5-percent reciprocal head only",
        "edge_priority": (
            "global joint-confidence descending; tie axis right-before-down, source, target"
        ),
        "dense_cost": (
            "mean-square frozen RGB-plus-gradient right-left and bottom-top side sequences"
        ),
        "placement_fill": "existing prioritized raw-tail component placement and Hungarian fill",
        "protected_tail": False,
        "whole_arm_selection": False,
        "requested_per_axis": REQUESTED_PER_AXIS,
        "solver_config": {
            "baseline_quantile": FROZEN_SOLVER_CONFIG.baseline_quantile,
            "search_rounds": FROZEN_SOLVER_CONFIG.search_rounds,
            "border_weight": FROZEN_SOLVER_CONFIG.border_weight,
            "random_seed": FROZEN_SOLVER_CONFIG.random_seed,
            "component_cap": FROZEN_SOLVER_CONFIG.component_cap,
            "fill_rounds": FROZEN_SOLVER_CONFIG.fill_rounds,
        },
    }
    if construction != expected:
        raise RuntimeError("signed joint-native construction changed")
    gate = config.get("gate", {})
    if gate != {
        "pair_mean_strictly_positive": True,
        "pair_source_bootstrap_95pct_lower_nonnegative": True,
        "exact_mean_nonnegative": True,
        "manhattan_benefit_mean_nonnegative": True,
        "radius2_mean_nonnegative": True,
    }:
        raise RuntimeError("signed joint-native gate changed")
    for artifact in config.get("frozen_inputs", {}).values():
        target = _path(str(artifact["path"]))
        if not target.is_file() or sha256_file(target) != artifact["sha256"]:
            raise RuntimeError(f"signed joint-native input changed: {target}")
    return config, digest


def _verify_nested_freeze(
    path: Path,
    *,
    expected_schema: str,
    archive: Path,
    metadata: Path,
) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != expected_schema:
        raise RuntimeError(f"upstream pre-score freeze schema changed: {path}")
    if not any(
        value.get(key) is True
        for key in (
            "created_before_exact_reference_scoring",
            "created_before_fit_head_label_scoring",
        )
    ):
        raise RuntimeError("upstream artifact was not frozen before label scoring")
    forbidden_flags = (
        "contains_exact_references_or_labels",
        "contains_target_slots_truth_or_reference_labels",
    )
    if any(value.get(key) is True for key in forbidden_flags):
        raise RuntimeError("upstream pre-score freeze contains labels")
    for key, target in (("archive", archive), ("metadata", metadata)):
        if value.get("artifacts", {}).get(key, {}).get("sha256") != sha256_file(target):
            raise RuntimeError(f"upstream frozen {key} hash mismatch")


def _inventory(
    config: Mapping[str, Any],
) -> tuple[
    Path,
    Path,
    tuple[dict[str, Any], ...],
    Path,
    tuple[dict[str, Any], ...],
    dict[tuple[str, int], dict[str, Any]],
]:
    inputs = config["frozen_inputs"]
    head_archive = _path(inputs["head_archive"]["path"])
    head_metadata = _path(inputs["head_metadata"]["path"])
    head_freeze = _path(inputs["head_pre_score_freeze"]["path"])
    control_archive = _path(inputs["control_archive"]["path"])
    control_metadata = _path(inputs["control_metadata"]["path"])
    control_freeze = _path(inputs["control_pre_score_freeze"]["path"])
    tri_report_path = _path(inputs["tri_fit_report"]["path"])
    _verify_nested_freeze(
        head_freeze,
        expected_schema="aiijc-joint-reciprocal-fit-heads-pre-score-freeze-v1",
        archive=head_archive,
        metadata=head_metadata,
    )
    _verify_nested_freeze(
        control_freeze,
        expected_schema="aiijc-structured-decoder-fit-oracle-control-freeze-v1",
        archive=control_archive,
        metadata=control_metadata,
    )
    head = json.loads(head_metadata.read_text(encoding="utf-8"))
    control = json.loads(control_metadata.read_text(encoding="utf-8"))
    if head.get("schema") != HEAD_SCHEMA:
        raise RuntimeError("fixed FIT head schema changed")
    if head.get("contains_target_slots_truth_or_reference_labels") is not False:
        raise RuntimeError("fixed FIT head metadata contains labels")
    if control.get("schema") != CONTROL_SCHEMA:
        raise RuntimeError("frozen control schema changed")
    if control.get("contains_exact_references_or_labels") is not False:
        raise RuntimeError("frozen control metadata contains labels")
    head_rows = tuple(head.get("rows", ()))
    control_rows = tuple(control.get("rows", ()))
    if len(head_rows) != 64 or len(control_rows) != 64:
        raise RuntimeError("FIT64 head/control case count changed")
    identity = ("prefix", "case_id", "source_filename", "draw_index", "dirty_sha256")
    if any(
        any(left.get(key) != right.get(key) for key in identity)
        for left, right in zip(head_rows, control_rows, strict=True)
    ):
        raise RuntimeError("FIT head and control sibling identities differ")
    if [row["prefix"] for row in head_rows] != [f"case_{index:04d}" for index in range(64)]:
        raise RuntimeError("FIT64 prefix order changed")

    tri_report = json.loads(tri_report_path.read_text(encoding="utf-8"))
    cache_rows = tuple(tri_report.get("fit_cache", {}).get("rows", ()))
    lookup = {(str(row["source_filename"]), int(row["draw_index"])): row for row in cache_rows}
    if len(cache_rows) != 64 or len(lookup) != 64:
        raise RuntimeError("immutable FIT cache roster changed")
    for row in head_rows:
        key = (str(row["source_filename"]), int(row["draw_index"]))
        cache = lookup.get(key)
        if cache is None or any(
            cache.get(field) != row.get(field) for field in ("case_id", "dirty_sha256")
        ):
            raise RuntimeError("FIT cache/head identity mismatch")
        frozen_cache = row.get("fit_cache", {})
        if frozen_cache.get("path") != cache.get("path") or frozen_cache.get("sha256") != cache.get(
            "sha256"
        ):
            raise RuntimeError("FIT cache/head path or hash mismatch")
        cache_path = _path(str(cache["path"]))
        if not cache_path.is_file() or sha256_file(cache_path) != cache["sha256"]:
            raise RuntimeError("immutable FIT cache bytes changed")

    forbidden = ("target_slots", "truth", "reference", "correct", "label")
    with np.load(head_archive, allow_pickle=False) as archive:
        if any(any(token in name.lower() for token in forbidden) for name in archive.files):
            raise RuntimeError("fixed head archive contains a forbidden label field")
    with np.load(control_archive, allow_pickle=False) as archive:
        if set(archive.files) != {f"{row['prefix']}__control_layout" for row in control_rows}:
            raise RuntimeError("frozen control archive key contract changed")
    return (
        head_archive,
        head_metadata,
        head_rows,
        control_archive,
        control_rows,
        lookup,
    )


def _cache_arrays(path: Path, names: tuple[str, ...]) -> tuple[np.ndarray, ...]:
    """Materialise only explicitly named NPZ members."""

    with np.load(path, allow_pickle=False) as archive:
        missing = set(names) - set(archive.files)
        if missing:
            raise RuntimeError(f"FIT cache omits required members: {sorted(missing)}")
        return tuple(np.array(archive[name], copy=True) for name in names)


def _head_arrays(
    archive: Any, prefix: str
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]:
    names = ("right", "down")
    sources = tuple(np.asarray(archive[f"{prefix}__selected_sources__{axis}"]) for axis in names)
    targets = tuple(np.asarray(archive[f"{prefix}__selected_targets__{axis}"]) for axis in names)
    confidence = tuple(
        np.asarray(archive[f"{prefix}__selected_joint_confidences__{axis}"]) for axis in names
    )
    frozen_head_edges(
        list(sources),
        list(targets),
        list(confidence),
        grid=GRID,
        requested_per_axis=REQUESTED_PER_AXIS,
    )
    return sources, targets, confidence


def _realised_head_edges(
    layout: np.ndarray,
    sources: tuple[np.ndarray, np.ndarray],
    targets: tuple[np.ndarray, np.ndarray],
) -> int:
    positions = np.empty(COUNT, dtype=np.int32)
    positions[layout] = np.arange(COUNT, dtype=np.int32)
    total = 0
    for axis in range(2):
        delta = 1 if axis == 0 else GRID
        source_position = positions[sources[axis]]
        target_position = positions[targets[axis]]
        valid = source_position % GRID != GRID - 1 if axis == 0 else source_position < COUNT - GRID
        total += int(np.count_nonzero(valid & (target_position == source_position + delta)))
    return total


def run_validate(config: Mapping[str, Any]) -> dict[str, Any]:
    _, _, rows, _, _, cache = _inventory(config)
    return {
        "schema": "aiijc-joint-native-head-arm-fit-validation-v1",
        "status": "ready-target-free",
        "case_count": len(rows),
        "cache_count": len(cache),
        "mps_or_model_inference_required": False,
        "fit_references_materialised": False,
    }


def run_freeze(
    config: Mapping[str, Any],
    config_sha: str,
    output_dir: Path,
) -> dict[str, Any]:
    output = output_dir.resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite joint-native FIT output")
    head_path, _, head_rows, control_path, control_rows, cache_lookup = _inventory(config)
    arrays: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    with (
        np.load(head_path, allow_pickle=False) as head_archive,
        np.load(control_path, allow_pickle=False) as control_archive,
    ):
        for index, (head_row, _control_row) in enumerate(zip(head_rows, control_rows, strict=True)):
            prefix = str(head_row["prefix"])
            cache = cache_lookup[(str(head_row["source_filename"]), int(head_row["draw_index"]))]
            cache_path = _path(str(cache["path"]))
            # Strict target-free freeze: target_slots/candidates are not requested here.
            (raw_sides,) = _cache_arrays(cache_path, ("raw_sides",))
            if raw_sides.shape != (4, COUNT, 20, 6) or not np.isfinite(raw_sides).all():
                raise RuntimeError("frozen raw_sides schema changed")
            sources, targets, confidence = _head_arrays(head_archive, prefix)
            solved = solve_joint_native_head_arm(
                raw_sides,
                list(sources),
                list(targets),
                list(confidence),
                grid=GRID,
                requested_per_axis=REQUESTED_PER_AXIS,
            )
            candidate = strict_layout(solved.layout, grid=GRID, name="candidate_layout")
            control = strict_layout(
                control_archive[f"{prefix}__control_layout"],
                grid=GRID,
                name="control_layout",
            )
            arrays[f"{prefix}__candidate_layout"] = candidate
            rows.append(
                {
                    **{
                        key: head_row[key]
                        for key in (
                            "prefix",
                            "case_id",
                            "source_filename",
                            "draw_index",
                            "dirty_sha256",
                        )
                    },
                    "candidate_layout_sha256": solved.layout_sha256,
                    "control_layout_sha256": hashlib.sha256(control.tobytes()).hexdigest(),
                    "changed_from_control": bool(not np.array_equal(candidate, control)),
                    "realised_supplied_head_edges": _realised_head_edges(
                        candidate, sources, targets
                    ),
                    "diagnostics": solved.diagnostics,
                }
            )
            print(
                json.dumps(
                    {
                        "event": "freeze_joint_native_head_arm",
                        "case": index + 1,
                        "count": len(head_rows),
                        "source": head_row["source_filename"],
                        "draw": head_row["draw_index"],
                    }
                ),
                flush=True,
            )
    archive = output / ARCHIVE_NAME
    metadata = output / METADATA_NAME
    freeze = output / FREEZE_NAME
    _write_npz_exclusive(archive, arrays)
    _write_json_exclusive(
        metadata,
        {
            "schema": METADATA_SCHEMA,
            "config_sha256": config_sha,
            "contains_exact_references_or_labels": False,
            "contains_pixels": False,
            "all_layouts_strict_original_upright_permutations": True,
            "case_count": len(rows),
            "changed_case_count": sum(row["changed_from_control"] for row in rows),
            "rows": rows,
        },
    )
    _write_json_exclusive(
        freeze,
        {
            "schema": FREEZE_SCHEMA,
            "created_before_fit_reference_materialisation": True,
            "contains_exact_references_or_labels": False,
            "config_sha256": config_sha,
            "artifacts": {
                "archive": _record(archive),
                "metadata": _record(metadata),
                "config": _record(_path(config["config_path"])),
                "head_archive": config["frozen_inputs"]["head_archive"],
                "control_archive": config["frozen_inputs"]["control_archive"],
                "tri_fit_report": config["frozen_inputs"]["tri_fit_report"],
            },
        },
    )
    return {
        "schema": "aiijc-joint-native-head-arm-fit-freeze-result-v1",
        "status": "target-free-layouts-frozen-fit-reference-scoring-not-run",
        "case_count": len(rows),
        "changed_case_count": sum(row["changed_from_control"] for row in rows),
        "mean_realised_supplied_head_edges": float(
            np.mean([row["realised_supplied_head_edges"] for row in rows])
        ),
        "archive": _record(archive),
        "metadata": _record(metadata),
        "pre_score_freeze": _record(freeze),
        "fit_references_materialised": False,
        "mps_or_model_inference_run": False,
    }


def _verify_own_freeze(
    config_sha: str,
    output_dir: Path,
) -> tuple[Path, tuple[dict[str, Any], ...]]:
    output = output_dir.resolve()
    archive = output / ARCHIVE_NAME
    metadata = output / METADATA_NAME
    freeze_path = output / FREEZE_NAME
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("schema") != FREEZE_SCHEMA:
        raise RuntimeError("joint-native pre-score freeze schema changed")
    if freeze.get("created_before_fit_reference_materialisation") is not True:
        raise RuntimeError("joint-native layouts were not frozen before references")
    if freeze.get("contains_exact_references_or_labels") is not False:
        raise RuntimeError("joint-native freeze unexpectedly contains labels")
    if freeze.get("config_sha256") != config_sha:
        raise RuntimeError("joint-native freeze belongs to another config")
    for key, target in (("archive", archive), ("metadata", metadata)):
        if not target.is_file() or freeze["artifacts"][key]["sha256"] != sha256_file(target):
            raise RuntimeError(f"joint-native frozen {key} changed before scoring")
    value = json.loads(metadata.read_text(encoding="utf-8"))
    if value.get("schema") != METADATA_SCHEMA:
        raise RuntimeError("joint-native target-free metadata schema changed")
    if value.get("contains_exact_references_or_labels") is not False:
        raise RuntimeError("joint-native target-free metadata contains labels")
    return archive, tuple(value["rows"])


def _distribution(values: np.ndarray) -> dict[str, Any]:
    return {
        "mean": float(values.mean()),
        "median": float(np.quantile(values, 0.5, method="linear")),
        "q25": float(np.quantile(values, 0.25, method="linear")),
        "q75": float(np.quantile(values, 0.75, method="linear")),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "wins": int(np.count_nonzero(values > TIE_EPSILON)),
        "ties": int(np.count_nonzero(np.abs(values) <= TIE_EPSILON)),
        "losses": int(np.count_nonzero(values < -TIE_EPSILON)),
    }


def _robust_metric(
    rows: Sequence[Mapping[str, Any]],
    name: str,
    *,
    metric_index: int,
) -> dict[str, Any]:
    case_values = np.asarray([row["benefit_delta"][name] for row in rows], dtype=np.float64)
    grouped: dict[str, list[float]] = defaultdict(list)
    for row, value in zip(rows, case_values, strict=True):
        grouped[str(row["source_filename"])].append(float(value))
    if len(grouped) != 32 or any(len(values) != 2 for values in grouped.values()):
        raise RuntimeError("FIT source-block contract must be 32 sources x two draws")
    source_names = tuple(grouped)
    source_values = np.asarray(
        [np.mean(grouped[source]) for source in source_names], dtype=np.float64
    )
    rng = np.random.default_rng(BOOTSTRAP_BASE_SEED + BOOTSTRAP_METRIC_SEED_STRIDE * metric_index)
    samples = source_values[
        rng.integers(0, len(source_values), size=(BOOTSTRAP_RESAMPLES, len(source_values)))
    ].mean(axis=1)
    positive = np.maximum(source_values, 0.0)
    harm = np.maximum(-source_values, 0.0)
    positive_index = int(np.argmax(positive))
    harm_index = int(np.argmax(harm))
    positive_total = float(positive.sum())
    harm_total = float(harm.sum())
    return {
        "case_distribution": _distribution(case_values),
        "source_mean_distribution": _distribution(source_values),
        "source_bootstrap_mean_95pct_ci": [
            float(np.quantile(samples, 0.025, method="linear")),
            float(np.quantile(samples, 0.975, method="linear")),
        ],
        "tail": {
            "positive_total": positive_total,
            "harm_total_absolute": harm_total,
            "largest_positive": {
                "source_filename": source_names[positive_index],
                "delta": float(source_values[positive_index]),
                "share_of_positive_total": (
                    float(positive[positive_index] / positive_total) if positive_total else 0.0
                ),
            },
            "largest_harm": {
                "source_filename": source_names[harm_index],
                "delta": float(source_values[harm_index]),
                "share_of_harm_total": (
                    float(harm[harm_index] / harm_total) if harm_total else 0.0
                ),
            },
            "leave_largest_positive_source_mean": (
                float((source_values.sum() - source_values[positive_index]) / 31)
                if positive_total
                else float(source_values.mean())
            ),
            "leave_largest_harm_source_mean": (
                float((source_values.sum() - source_values[harm_index]) / 31)
                if harm_total
                else float(source_values.mean())
            ),
        },
    }


def run_score(
    config: Mapping[str, Any],
    config_sha: str,
    output_dir: Path,
) -> dict[str, Any]:
    candidate_path, frozen_rows = _verify_own_freeze(config_sha, output_dir)
    head_path, _, head_rows, control_path, control_rows, cache_lookup = _inventory(config)
    prefixes = [str(row["prefix"]) for row in head_rows]
    if [row["prefix"] for row in control_rows] != prefixes or [
        row["prefix"] for row in frozen_rows
    ] != prefixes:
        raise RuntimeError("score sibling order differs from frozen FIT64 order")
    result_rows: list[dict[str, Any]] = []
    with (
        np.load(candidate_path, allow_pickle=False) as candidate_archive,
        np.load(control_path, allow_pickle=False) as control_archive,
        np.load(head_path, allow_pickle=False) as head_archive,
    ):
        for frozen, head_row in zip(frozen_rows, head_rows, strict=True):
            prefix = str(head_row["prefix"])
            candidate = strict_layout(
                candidate_archive[f"{prefix}__candidate_layout"],
                grid=GRID,
                name="candidate_layout",
            )
            control = strict_layout(
                control_archive[f"{prefix}__control_layout"],
                grid=GRID,
                name="control_layout",
            )
            cache = cache_lookup[(str(head_row["source_filename"]), int(head_row["draw_index"]))]
            # This is the first reference materialisation in this signed runner.
            candidates, target_slots = _cache_arrays(
                _path(str(cache["path"])), ("candidates", "target_slots")
            )
            reference = reference_from_target_slots(
                candidates,
                target_slots,
                grid=GRID,
            )
            control_metrics = layout_metrics(control, reference, grid=GRID)
            candidate_metrics = layout_metrics(candidate, reference, grid=GRID)
            sources, targets, _ = _head_arrays(head_archive, prefix)
            benefit = {
                "satisfied_pairs": (
                    candidate_metrics.satisfied_pairs - control_metrics.satisfied_pairs
                ),
                "exact_tiles": candidate_metrics.exact_tiles - control_metrics.exact_tiles,
                "manhattan": (
                    control_metrics.mean_absolute_manhattan
                    - candidate_metrics.mean_absolute_manhattan
                ),
                "radius2_recall": (
                    candidate_metrics.radius2_recall - control_metrics.radius2_recall
                ),
            }
            result_rows.append(
                {
                    **{
                        key: frozen[key]
                        for key in (
                            "prefix",
                            "case_id",
                            "source_filename",
                            "draw_index",
                        )
                    },
                    "control": control_metrics.as_dict(),
                    "candidate": candidate_metrics.as_dict(),
                    "benefit_delta": benefit,
                    "selected_head_true_edges": sum(
                        int(
                            np.count_nonzero(
                                targets[axis]
                                == _truth_from_reference(reference, axis)[sources[axis]]
                            )
                        )
                        for axis in range(2)
                    ),
                    "candidate_realised_supplied_head_edges": _realised_head_edges(
                        candidate, sources, targets
                    ),
                    "control_realised_supplied_head_edges": _realised_head_edges(
                        control, sources, targets
                    ),
                }
            )
    metric_names = ("satisfied_pairs", "exact_tiles", "manhattan", "radius2_recall")
    robust = {
        name: _robust_metric(result_rows, name, metric_index=index)
        for index, name in enumerate(metric_names)
    }
    control_mean = {
        name: float(np.mean([row["control"][name] for row in result_rows]))
        for name in (
            "satisfied_pairs",
            "exact_tiles",
            "mean_absolute_manhattan",
            "radius2_recall",
        )
    }
    candidate_mean = {
        name: float(np.mean([row["candidate"][name] for row in result_rows]))
        for name in control_mean
    }
    checks = {
        "pair_mean_strictly_positive": robust["satisfied_pairs"]["case_distribution"]["mean"] > 0.0,
        "pair_source_bootstrap_95pct_lower_nonnegative": robust["satisfied_pairs"][
            "source_bootstrap_mean_95pct_ci"
        ][0]
        >= 0.0,
        "exact_mean_nonnegative": robust["exact_tiles"]["case_distribution"]["mean"] >= 0.0,
        "manhattan_benefit_mean_nonnegative": robust["manhattan"]["case_distribution"]["mean"]
        >= 0.0,
        "radius2_mean_nonnegative": robust["radius2_recall"]["case_distribution"]["mean"] >= 0.0,
    }
    report = {
        "schema": SCORE_SCHEMA,
        "status": (
            "pass-new-joint-native-arm" if all(checks.values()) else "fail-stop-do-not-promote"
        ),
        "config_sha256": config_sha,
        "case_count": len(result_rows),
        "source_count": 32,
        "draws_per_source": 2,
        "aggregate": {
            "control_mean": control_mean,
            "candidate_mean": candidate_mean,
            "mean_selected_head_true_edges": float(
                np.mean([row["selected_head_true_edges"] for row in result_rows])
            ),
            "mean_candidate_realised_supplied_head_edges": float(
                np.mean([row["candidate_realised_supplied_head_edges"] for row in result_rows])
            ),
            "mean_control_realised_supplied_head_edges": float(
                np.mean([row["control_realised_supplied_head_edges"] for row in result_rows])
            ),
        },
        "robust_metrics": robust,
        "gate": {"checks": checks, "passed": all(checks.values())},
        "raw_arm_comparator": {
            "available": False,
            "reason": (
                "the frozen FIT64 control archive stores only the relation-selector "
                "winner; reconstructing the historical raw TASKA arm would require "
                "forbidden repeated matcher inference"
            ),
        },
        "rows": result_rows,
        "freeze": {
            "archive_sha256": sha256_file(candidate_path),
            "verified_before_fit_reference_materialisation": True,
        },
        "all_outputs_strict_576_original_upright_permutations": True,
        "fit_only": True,
        "dev_local_terminal_test_or_competition_accessed": False,
        "model_training_or_inference_run": False,
        "whole_arm_reselection_run": False,
        "weco_logged": False,
    }
    _write_json_exclusive(output_dir.resolve() / SCORE_NAME, report)
    return report


def _truth_from_reference(reference: np.ndarray, axis: int) -> np.ndarray:
    board = reference.reshape(GRID, GRID)
    truth = np.full(COUNT, -1, dtype=np.int32)
    if axis == 0:
        truth[board[:, :-1].ravel()] = board[:, 1:].ravel()
    else:
        truth[board[:-1].ravel()] = board[1:].ravel()
    return truth


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config, config_sha = _load_signed_config(args.config)
    if config.get("config_path") != _project_path(args.config):
        raise RuntimeError("runtime config path differs from signed path")
    if args.mode == "validate":
        report = run_validate(config)
    elif args.mode == "freeze":
        report = run_freeze(config, config_sha, args.output_dir)
    else:
        report = run_score(config, config_sha, args.output_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if args.mode == "score" and not report["gate"]["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

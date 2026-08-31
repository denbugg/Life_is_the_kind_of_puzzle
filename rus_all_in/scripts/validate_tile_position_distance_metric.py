#!/usr/bin/env python3
"""Validate tile-position distance against clean, dirty and restored SSIM."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr

from aiijc_puzzle.protocol import assemble_tiles, contest_ssim, sha256_file
from aiijc_puzzle.socket_pixel_tails import historical_rgb_luma_nlm_h20_once
from aiijc_puzzle.taska_selective_fullres_fusion import FUSION_ARM_NAMES
from aiijc_puzzle.tile_position_distance import (
    evaluate_best_cyclic_aligned_tile_position_distance,
    evaluate_tile_position_distance,
)

try:
    from scripts import run_taska_focal_current_finetune as finetune
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    import run_taska_focal_current_finetune as finetune


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/tile_position_distance_metric_validation_v1.json"
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/tile-position-distance-validation/fixed-v1"
ARM_ROOT = PROJECT_ROOT / "outputs/taska-six-arm-learned-selector/fixed-v1/local32"
ARM_ARCHIVE = ARM_ROOT / "frozen-target-free-arms.npz"
ARM_METADATA = ARM_ROOT / "frozen-target-free-arms.json"
TRANSFER_ROOT = PROJECT_ROOT / "outputs/taska-socket-cyclic-origin-transfer/local32-v1"
TRANSFER_ARCHIVE = TRANSFER_ROOT / "frozen-target-free-eval.npz"
TRANSFER_METADATA = TRANSFER_ROOT / "frozen-target-free-eval.json"
GRID = 24
COUNT = GRID * GRID
CONTROL = "confirmed_six_arm_fusion"
TRANSFER = "socket_cyclic_border5_origin"
REFERENCE_VARIANTS = (
    "exact_reference",
    "reference_global_row_plus1",
    "reference_global_row_minus1",
    "reference_global_column_plus1",
    "reference_global_column_minus1",
    "reference_global_diagonal_plus1_plus1",
    "reference_local_horizontal_swaps8",
    "reference_local_horizontal_swaps32",
    "reference_block_swap_4x4",
    "reference_component_shift_irregular21_plus10_plus10",
    "reference_fixed_position_permutation_seed20260831",
)
CONTROL_PERTURBATIONS = (
    "control_global_row_plus1",
    "control_global_column_plus1",
    "control_global_diagonal_minus1_plus1",
    "control_local_horizontal_swaps8",
    "control_component_shift_irregular21_plus10_plus10",
)
SOLVER_VARIANTS = (*FUSION_ARM_NAMES, CONTROL, TRANSFER)
VARIANT_ROSTER = (*SOLVER_VARIANTS, *REFERENCE_VARIANTS, *CONTROL_PERTURBATIONS)
SWAPS8 = (
    (2, 6),
    (5, 15),
    (8, 1),
    (11, 10),
    (14, 19),
    (17, 5),
    (20, 14),
    (22, 20),
)
SWAPS32 = tuple(
    (row, column)
    for row in (1, 4, 7, 10, 13, 16, 19, 22)
    for column in (2, 8, 14, 20)
)
SSIM_NAMES = (
    "layout_only_clean_ssim",
    "production_like_dirty_ssim",
    "production_like_restored_h20_ssim",
)
DISTANCE_NAMES = (
    "mean_manhattan_cells",
    "median_manhattan_cells",
    "p90_manhattan_cells",
    "normalized_mean_l1",
    "mean_euclidean_cells",
)
RECALL_NAMES = (
    "within_radius_0_recall",
    "within_radius_1_recall",
    "within_radius_2_recall",
)
METRIC_SPECS = tuple(
    (f"absolute.{name}", -1.0) for name in DISTANCE_NAMES
) + tuple((f"absolute.{name}", 1.0) for name in RECALL_NAMES) + tuple(
    (f"cyclic_aligned.{name}", -1.0) for name in DISTANCE_NAMES
) + tuple((f"cyclic_aligned.{name}", 1.0) for name in RECALL_NAMES)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
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
        rendered = str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        rendered = str(resolved)
    return {"path": rendered, "sha256": sha256_file(resolved)}


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    resolved = path.resolve()
    sidecar = resolved.with_suffix(resolved.suffix + ".sha256")
    if not resolved.is_file() or not sidecar.is_file():
        raise FileNotFoundError("signed distance-metric preregistration is missing")
    digest = sha256_file(resolved)
    if sidecar.read_text(encoding="utf-8").split()[0] != digest:
        raise ValueError("distance-metric preregistration SHA-256 mismatch")
    config = json.loads(resolved.read_text(encoding="utf-8"))
    required = {
        "source_count": 32,
        "draws_per_source": 1,
        "source_order_digest": "f516f12e8943580ab62e17cd6d4064dc519aa20df6485bf5bca34030beaa2bc3",
    }
    for key, expected in required.items():
        if config.get("panel", {}).get(key) != expected:
            raise ValueError(f"distance-metric panel contract changed: {key}")
    perturbations = config.get("deterministic_perturbation_roster", {})
    if perturbations.get("candidate_count_per_board") != len(VARIANT_ROSTER):
        raise ValueError("distance-metric candidate count changed")
    if perturbations.get("total_candidate_count") != 32 * len(VARIANT_ROSTER):
        raise ValueError("distance-metric total candidate count changed")
    if config.get("frozen_solver_layout_roster") != list(SOLVER_VARIANTS):
        raise ValueError("distance-metric solver roster changed")
    for relative, expected in config["fixed_inputs_sha256"].items():
        source = PROJECT_ROOT / relative
        if not source.is_file() or sha256_file(source) != expected:
            raise ValueError(f"fixed distance-metric input changed: {relative}")
    return config, digest


def _rows(path: Path) -> list[Mapping[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8")).get("rows")
    if not isinstance(rows, list) or len(rows) != 32:
        raise ValueError(f"expected exactly 32 rows in {path}")
    return rows


def _strict(value: Any) -> np.ndarray:
    layout = np.asarray(value, dtype=np.int32)
    if layout.shape != (COUNT,) or not np.array_equal(
        np.sort(layout), np.arange(COUNT, dtype=np.int32)
    ):
        raise ValueError("layout must be a strict 576-tile permutation")
    return np.ascontiguousarray(layout)


def _global_roll(layout: np.ndarray, row: int, column: int) -> np.ndarray:
    return np.ascontiguousarray(
        np.roll(layout.reshape(GRID, GRID), shift=(row, column), axis=(0, 1)).reshape(-1),
        dtype=np.int32,
    )


def _horizontal_swaps(
    layout: np.ndarray, positions: Sequence[tuple[int, int]]
) -> np.ndarray:
    result = layout.copy()
    used: set[int] = set()
    for row, column in positions:
        if not (0 <= row < GRID and 0 <= column < GRID - 1):
            raise ValueError("horizontal swap lies outside the board")
        first = row * GRID + column
        second = first + 1
        if first in used or second in used:
            raise ValueError("horizontal swaps must be disjoint")
        used.update((first, second))
        result[[first, second]] = result[[second, first]]
    return _strict(result)


def _block_swap(layout: np.ndarray) -> np.ndarray:
    board = layout.reshape(GRID, GRID).copy()
    first = board[2:6, 2:6].copy()
    second = board[18:22, 18:22].copy()
    board[2:6, 2:6] = second
    board[18:22, 18:22] = first
    return _strict(board.reshape(-1))


def _component_shift(layout: np.ndarray) -> np.ndarray:
    result = layout.copy()
    offsets = [
        (row, column)
        for row in range(5)
        for column in range(5)
        if (row, column) not in {(0, 0), (0, 4), (4, 0), (4, 4)}
    ]
    source = np.asarray([(3 + row) * GRID + 3 + column for row, column in offsets])
    target = np.asarray([(13 + row) * GRID + 13 + column for row, column in offsets])
    if set(source.tolist()) & set(target.tolist()) or len(source) != 21:
        raise RuntimeError("irregular component recipe changed")
    saved = result[source].copy()
    result[source] = result[target]
    result[target] = saved
    return _strict(result)


def _reference_variants(reference: np.ndarray) -> dict[str, np.ndarray]:
    position_permutation = np.random.default_rng(20260831).permutation(COUNT)
    return {
        "exact_reference": reference,
        "reference_global_row_plus1": _global_roll(reference, 1, 0),
        "reference_global_row_minus1": _global_roll(reference, -1, 0),
        "reference_global_column_plus1": _global_roll(reference, 0, 1),
        "reference_global_column_minus1": _global_roll(reference, 0, -1),
        "reference_global_diagonal_plus1_plus1": _global_roll(reference, 1, 1),
        "reference_local_horizontal_swaps8": _horizontal_swaps(reference, SWAPS8),
        "reference_local_horizontal_swaps32": _horizontal_swaps(reference, SWAPS32),
        "reference_block_swap_4x4": _block_swap(reference),
        "reference_component_shift_irregular21_plus10_plus10": _component_shift(
            reference
        ),
        "reference_fixed_position_permutation_seed20260831": _strict(
            reference[position_permutation]
        ),
    }


def _control_variants(control: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "control_global_row_plus1": _global_roll(control, 1, 0),
        "control_global_column_plus1": _global_roll(control, 0, 1),
        "control_global_diagonal_minus1_plus1": _global_roll(control, -1, 1),
        "control_local_horizontal_swaps8": _horizontal_swaps(control, SWAPS8),
        "control_component_shift_irregular21_plus10_plus10": _component_shift(
            control
        ),
    }


def _variant_family(name: str) -> str:
    if name in SOLVER_VARIANTS:
        return "frozen_solver"
    if name in REFERENCE_VARIANTS:
        return "reference_control"
    if name in CONTROL_PERTURBATIONS:
        return "confirmed_control_perturbation"
    raise KeyError(name)


def _aligned_parent_rows() -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    arms = _rows(ARM_METADATA)
    transfer = _rows(TRANSFER_METADATA)
    identity = ("prefix", "source_filename", "draw_index", "dirty_sha256")
    aligned = []
    for first, second in zip(arms, transfer, strict=True):
        if any(first.get(field) != second.get(field) for field in identity):
            raise RuntimeError("six-arm and Socket-transfer rows do not align")
        aligned.append((first, second))
    return aligned


def _freeze_layouts(
    *, output_dir: Path, config_path: Path, config: Mapping[str, Any], targets_dir: Path
) -> tuple[Path, Path, Path, float]:
    output_dir.mkdir(parents=True, exist_ok=False)
    source_config, _, _ = finetune._load_config(finetune.DEFAULT_CONFIG)
    lookup = finetune._manifest_lookup(source_config)
    cache = finetune.CleanTileCache(targets_dir, maximum_boards=2)
    arrays: dict[str, np.ndarray] = {}
    metadata_rows: list[dict[str, Any]] = []
    started = perf_counter()
    aligned = _aligned_parent_rows()
    configured_names = config["panel"]["source_filenames"]
    observed_names = [str(row[0]["source_filename"]) for row in aligned]
    if observed_names != configured_names:
        raise RuntimeError("configured metric-validation source order changed")

    with (
        np.load(ARM_ARCHIVE, allow_pickle=False) as arms,
        np.load(TRANSFER_ARCHIVE, allow_pickle=False) as transferred,
    ):
        for index, (arm_row, _transfer_row) in enumerate(aligned, start=1):
            prefix = str(arm_row["prefix"])
            source = str(arm_row["source_filename"])
            draw = int(arm_row["draw_index"])
            dirty = finetune._dirty_case(cache, lookup[source], source, draw)
            if finetune._dirty_sha256(dirty.dirty_tiles) != arm_row["dirty_sha256"]:
                raise RuntimeError("metric freeze recreated different dirty bytes")
            reference = finetune._reference(
                cache, lookup[source], source, draw, dirty.dirty_tiles
            )
            variants = {
                name: _strict(arms[f"{prefix}__{name}_layout"])
                for name in FUSION_ARM_NAMES
            }
            variants[CONTROL] = _strict(arms[f"{prefix}__{CONTROL}_layout"])
            variants[TRANSFER] = _strict(
                transferred[f"{prefix}__{TRANSFER}_layout"]
            )
            variants.update(_reference_variants(_strict(reference)))
            variants.update(_control_variants(variants[CONTROL]))
            if tuple(variants) != VARIANT_ROSTER:
                raise RuntimeError("materialized layout variant order changed")
            for name, layout in variants.items():
                arrays[f"{prefix}__{name}_layout"] = _strict(layout)
            metadata_rows.append(
                {
                    "prefix": prefix,
                    "source_filename": source,
                    "draw_index": draw,
                    "dirty_sha256": str(arm_row["dirty_sha256"]),
                    "variant_count": len(variants),
                }
            )
            print(
                json.dumps(
                    {
                        "event": "distance_metric_layout_freeze",
                        "case": index,
                        "variant_count": len(variants),
                    }
                ),
                flush=True,
            )

    archive = output_dir / "frozen-layout-roster.npz"
    metadata = output_dir / "frozen-layout-roster.json"
    freeze = output_dir / "pre-score-freeze.json"
    _write_npz(archive, arrays)
    _write_json(
        metadata,
        {
            "schema": "aiijc-tile-position-distance-frozen-layout-roster-v1",
            "created_before_any_distance_or_ssim_scoring": True,
            "contains_target_assisted_reference_perturbation_controls": True,
            "terminal_held_fresh_or_competition_test_accessed": False,
            "variant_roster": list(VARIANT_ROSTER),
            "variant_families": {
                name: _variant_family(name) for name in VARIANT_ROSTER
            },
            "rows": metadata_rows,
        },
    )
    _write_json(
        freeze,
        {
            "schema": "aiijc-tile-position-distance-pre-score-freeze-v1",
            "created_before_any_distance_or_ssim_scoring": True,
            "candidate_count": len(metadata_rows) * len(VARIANT_ROSTER),
            "terminal_held_fresh_or_competition_test_accessed": False,
            "artifacts": {
                "layout_archive": _record(archive),
                "layout_metadata": _record(metadata),
                "preregistration": _record(config_path),
                "six_arm_archive": _record(ARM_ARCHIVE),
                "six_arm_metadata": _record(ARM_METADATA),
                "socket_transfer_archive": _record(TRANSFER_ARCHIVE),
                "socket_transfer_metadata": _record(TRANSFER_METADATA),
                "module": _record(
                    PROJECT_ROOT / "src/aiijc_puzzle/tile_position_distance.py"
                ),
                "runner": _record(Path(__file__)),
            },
        },
    )
    return archive, metadata, freeze, perf_counter() - started


def _metric_value(row: Mapping[str, Any], name: str) -> float:
    section, metric = name.split(".", 1)
    return float(row["position_distance"][section][metric])


def _safe_correlation(
    x: np.ndarray, y: np.ndarray, *, method: str
) -> float | None:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    result = pearsonr(x, y) if method == "pearson" else spearmanr(x, y)
    value = float(result.statistic)
    return value if np.isfinite(value) else None


def _correlation_views(
    rows: Sequence[Mapping[str, Any]], metric_name: str, ssim_name: str, sign: float
) -> dict[str, Any]:
    x = np.asarray([sign * _metric_value(row, metric_name) for row in rows])
    y = np.asarray([row["ssim"][ssim_name] for row in rows], dtype=np.float64)
    sources = np.asarray([row["source_filename"] for row in rows])
    variants = np.asarray([row["variant"] for row in rows])

    pooled = {
        method: _safe_correlation(x, y, method=method)
        for method in ("pearson", "spearman")
    }

    centered_x = x.copy()
    centered_y = y.copy()
    ranked_x = np.empty_like(x)
    ranked_y = np.empty_like(y)
    per_source: dict[str, list[float]] = {"pearson": [], "spearman": []}
    for source in np.unique(sources):
        mask = sources == source
        centered_x[mask] -= x[mask].mean()
        centered_y[mask] -= y[mask].mean()
        ranked_x[mask] = rankdata(x[mask], method="average")
        ranked_y[mask] = rankdata(y[mask], method="average")
        ranked_x[mask] -= ranked_x[mask].mean()
        ranked_y[mask] -= ranked_y[mask].mean()
        for method in ("pearson", "spearman"):
            value = _safe_correlation(x[mask], y[mask], method=method)
            if value is not None:
                per_source[method].append(value)
    within = {
        "pearson": _safe_correlation(centered_x, centered_y, method="pearson"),
        "spearman": _safe_correlation(ranked_x, ranked_y, method="pearson"),
    }

    family_x = []
    family_y = []
    for variant in VARIANT_ROSTER:
        mask = variants == variant
        if int(mask.sum()) != 32:
            raise RuntimeError("family mean expected 32 source rows per variant")
        family_x.append(float(x[mask].mean()))
        family_y.append(float(y[mask].mean()))
    family_x_array = np.asarray(family_x)
    family_y_array = np.asarray(family_y)
    family = {
        method: _safe_correlation(family_x_array, family_y_array, method=method)
        for method in ("pearson", "spearman")
    }

    per_source_summary = {}
    for method, values in per_source.items():
        array = np.asarray(values, dtype=np.float64)
        per_source_summary[method] = {
            "valid_source_count": len(values),
            "median": float(np.median(array)) if len(array) else None,
            "p25": float(np.quantile(array, 0.25)) if len(array) else None,
            "p75": float(np.quantile(array, 0.75)) if len(array) else None,
        }
    return {
        "quality_orientation": "higher_is_better",
        "raw_metric_sign_multiplier": sign,
        "pooled": pooled,
        "within_source_centered": within,
        "per_source": per_source_summary,
        "family_mean": family,
    }


def _all_correlations(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        metric_name: {
            ssim_name: _correlation_views(rows, metric_name, ssim_name, sign)
            for ssim_name in SSIM_NAMES
        }
        for metric_name, sign in METRIC_SPECS
    }


def _score_layouts(
    *, archive: Path, metadata: Path, targets_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, Any], float]:
    metadata_payload = json.loads(metadata.read_text(encoding="utf-8"))
    if metadata_payload.get("created_before_any_distance_or_ssim_scoring") is not True:
        raise RuntimeError("layout roster was not frozen before scoring")
    rows = metadata_payload["rows"]
    source_config, _, _ = finetune._load_config(finetune.DEFAULT_CONFIG)
    lookup = finetune._manifest_lookup(source_config)
    cache = finetune.CleanTileCache(targets_dir, maximum_boards=2)
    scored: list[dict[str, Any]] = []
    started = perf_counter()
    with np.load(archive, allow_pickle=False) as layouts:
        for case_index, row in enumerate(rows, start=1):
            prefix = str(row["prefix"])
            source = str(row["source_filename"])
            draw = int(row["draw_index"])
            clean_tiles = cache.load(lookup[source])
            dirty = finetune._dirty_case(cache, lookup[source], source, draw)
            if finetune._dirty_sha256(dirty.dirty_tiles) != row["dirty_sha256"]:
                raise RuntimeError("distance scoring recreated different dirty bytes")
            reference = _strict(
                finetune._reference(
                    cache, lookup[source], source, draw, dirty.dirty_tiles
                )
            )
            input_clean_tiles = np.ascontiguousarray(
                clean_tiles[np.argsort(reference)], dtype=np.uint8
            )
            if not np.array_equal(input_clean_tiles[reference], clean_tiles):
                raise RuntimeError("clean shuffled tile identity reconstruction failed")
            clean_target = assemble_tiles(clean_tiles)
            for variant_index, variant in enumerate(VARIANT_ROSTER, start=1):
                layout = _strict(layouts[f"{prefix}__{variant}_layout"])
                absolute = evaluate_tile_position_distance(
                    layout, reference, grid=GRID
                )
                aligned = evaluate_best_cyclic_aligned_tile_position_distance(
                    layout, reference, grid=GRID, minimum_gain=1e-12
                )
                clean_canvas = assemble_tiles(input_clean_tiles[layout])
                dirty_canvas = assemble_tiles(dirty.dirty_tiles[layout])
                restored = historical_rgb_luma_nlm_h20_once(dirty_canvas)
                scored.append(
                    {
                        "prefix": prefix,
                        "source_filename": source,
                        "draw_index": draw,
                        "variant": variant,
                        "variant_family": _variant_family(variant),
                        "position_distance": {
                            "absolute": absolute.as_dict(),
                            "cyclic_aligned": aligned.metrics.as_dict(),
                            "cyclic_alignment": {
                                "selected_row_roll": aligned.selected_row_roll,
                                "selected_column_roll": aligned.selected_column_roll,
                                "changed": aligned.changed,
                                "candidates_evaluated": aligned.candidates_evaluated,
                            },
                        },
                        "ssim": {
                            "layout_only_clean_ssim": contest_ssim(
                                clean_target, clean_canvas
                            ),
                            "production_like_dirty_ssim": contest_ssim(
                                clean_target, dirty_canvas
                            ),
                            "production_like_restored_h20_ssim": contest_ssim(
                                clean_target, restored
                            ),
                        },
                    }
                )
                print(
                    json.dumps(
                        {
                            "event": "distance_metric_score",
                            "case": case_index,
                            "variant": variant_index,
                            "variant_count": len(VARIANT_ROSTER),
                        }
                    ),
                    flush=True,
                )
    if len(scored) != 32 * len(VARIANT_ROSTER):
        raise RuntimeError("distance metric scored row count changed")
    correlations = _all_correlations(scored)
    return scored, correlations, perf_counter() - started


def _variant_means(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["variant"])].append(row)
    output = {}
    for variant in VARIANT_ROSTER:
        values = grouped[variant]
        output[variant] = {
            "variant_family": _variant_family(variant),
            "case_count": len(values),
            "absolute": {
                metric: float(
                    np.mean(
                        [row["position_distance"]["absolute"][metric] for row in values]
                    )
                )
                for metric in (*DISTANCE_NAMES, *RECALL_NAMES)
            },
            "cyclic_aligned": {
                metric: float(
                    np.mean(
                        [
                            row["position_distance"]["cyclic_aligned"][metric]
                            for row in values
                        ]
                    )
                )
                for metric in (*DISTANCE_NAMES, *RECALL_NAMES)
            },
            "ssim": {
                name: float(np.mean([row["ssim"][name] for row in values]))
                for name in SSIM_NAMES
            },
        }
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    config, preregistration_sha256 = _load_config(args.config)
    archive, metadata, freeze, freeze_runtime = _freeze_layouts(
        output_dir=args.output_dir.resolve(),
        config_path=args.config.resolve(),
        config=config,
        targets_dir=args.targets.resolve(),
    )
    rows, correlations, score_runtime = _score_layouts(
        archive=archive, metadata=metadata, targets_dir=args.targets.resolve()
    )
    report = {
        "schema": "aiijc-tile-position-distance-metric-validation-report-v1",
        "status": "complete",
        "preregistration_sha256": preregistration_sha256,
        "protocol": config,
        "scored_row_count": len(rows),
        "variant_means": _variant_means(rows),
        "correlations": correlations,
        "rows": rows,
        "runtime_seconds": {
            "layout_freeze": freeze_runtime,
            "distance_ssim_and_correlation": score_runtime,
        },
        "legality": {
            "strict_original_upright_layouts": True,
            "restoration_is_evaluation_only_single_h20": True,
            "terminal_held_fresh_or_competition_test_accessed": False,
            "model_training_or_threshold_sweep": False,
            "production_or_submission_modified": False,
        },
        "artifacts": {
            "layout_archive": _record(archive),
            "layout_metadata": _record(metadata),
            "pre_score_freeze": _record(freeze),
            "module": _record(
                PROJECT_ROOT / "src/aiijc_puzzle/tile_position_distance.py"
            ),
            "runner": _record(Path(__file__)),
        },
    }
    _write_json(args.output_dir / "report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    selected = (
        "absolute.mean_manhattan_cells",
        "absolute.within_radius_0_recall",
        "absolute.within_radius_1_recall",
        "absolute.within_radius_2_recall",
        "cyclic_aligned.mean_manhattan_cells",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "scored_row_count": report["scored_row_count"],
                "runtime_seconds": report["runtime_seconds"],
                "selected_correlations": {
                    name: report["correlations"][name] for name in selected
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

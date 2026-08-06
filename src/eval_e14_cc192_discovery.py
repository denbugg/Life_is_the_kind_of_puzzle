"""Fixed CPU-only E14 CC192 clean-oracle discovery on opened E12 caches.

E14 compares the exact replayed E12 RR96 baseline with one candidate only:
CC192, built from the existing E12 clean-oracle candidates and scores using
``max_edges=192``, ``min_margin=0``, and ``repair_passes=0``.  It assembles
only the original corrupted upright tiles and applies fixed NLM(10).

The evaluator has a staged fail-closed contract.  It first verifies RR96 and
the label-aware CC192 structural gates.  End-to-end CC192 solve/NLM metrics run
only if mean edge precision is at least 0.95 and mean component coverage is at
least 0.45.  There is no sweep, other budget, transplant, GPU, or orientation
control.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import skimage
from skimage.metrics import structural_similarity as sk_ssim

import e14_cc192_oracle as cc192
import eval_clean_score_oracle as e12
from imgio import assemble
from placement_metrics import neighbour_accuracy, placement_accuracy


class E14ContractError(RuntimeError):
    """The E14 protocol or a byte-pinned E12 input drifted."""


SCHEMA_VERSION = 1
REPORT_SCHEMA = "pazzle-e14-cc192-clean-oracle-discovery-report-v1"
EXPERIMENT = "e14_cc192_clean_oracle_discovery_v1"
EXPECTED_E12_REPORT_SHA256 = (
    "16ceecfea99e006a1126b17d7d58fb5d188ec694c6a5097310dfe021bd2f901a"
)
EXPECTED_RR_MEAN_SOLVE_SSIM = 0.094607964147414
EXPECTED_RR_MEAN_FINAL_SSIM = 0.15930445310452002

STRUCTURAL_RULE: dict[str, float] = {
    "mean_selected_edge_precision_min": 0.95,
    "mean_component_coverage_min": 0.45,
}
END_TO_END_RULE: dict[str, float | int] = {
    "cc192_minus_rr96_mean_solve_ssim_min": 0.010,
    "cc192_minus_rr96_mean_final_ssim_min": 0.015,
    "cc192_minus_rr96_final_wins_min": 6,
    "cc192_minus_rr96_worst_final_delta_min": -0.020,
}
EXPECTED_RUNTIME_PROVENANCE = {
    "python": "3.13.6",
    "numpy": "2.2.6",
    "scikit_image": "0.26.0",
    "opencv": "4.13.0",
    "opencv_build_sha256": "ad2e3bc9bf8eb9d40a90e2f61a2c7667acee8a22e860778ea3378a4ed68f2be7",
    "torch": "2.11.0+cu128",
    "execution": "CPU_only",
}

E14_PROTOCOL: dict[str, Any] = {
    "schema": "pazzle-e14-cc192-clean-oracle-discovery-v1",
    "role": "opened_E12_clean_oracle_discovery_not_production_or_submission",
    "calibration_ids": [10, 11, 12, 13, 14, 15, 16, 17],
    "input_e12_report_sha256": EXPECTED_E12_REPORT_SHA256,
    "arms": {
        "RR96": {
            "candidates": "existing_E12_raw_candidates",
            "scores": "existing_E12_raw_scores",
            "max_edges": 96,
            "min_margin": 0.0,
            "repair_passes": 0,
            "role": "exact_baseline_replay",
        },
        "CC192": {
            "candidates": "existing_E12_clean_oracle_candidates",
            "scores": "existing_E12_clean_oracle_scores",
            "max_edges": 192,
            "min_margin": 0.0,
            "repair_passes": 0,
            "role": "diagnostic_only_not_deployable",
        },
    },
    "structural_measurement": {
        "edge_selection": "solve_buddies._candidate_edges_exact",
        "component_builder": "solve_buddies.build_buddies_components_exact",
        "selected_edge_precision": "mean_per_scene_true_selected_edges_over_selected_edges",
        "selected_edge_count": "exactly_192_per_scene_fail_closed",
        "true_edge": "permutation_coordinate_delta_equals_selected_dy_dx",
        "component_coverage": "mean_per_scene_unique_component_tiles_over_576",
        "gate": dict(STRUCTURAL_RULE),
        "fail_closed_before_end_to_end": True,
    },
    "geometry": {
        "grid": 24,
        "tile_size": 20,
        "orientation": "upright_only",
        "rotation": False,
        "reflection": False,
        "assembly": "original_corrupted_tiles_only",
    },
    "restoration": {
        "name": "opencv_fast_nlm_colored",
        "h": 10,
        "h_color": 10,
        "template_window": 7,
        "search_window": 21,
        "scope": "CC192_once_per_scene_after_assembly",
        "RR96_before": "reuse_exact_pinned_E12_metrics_no_second_NLM_call",
    },
    "rr_reproducibility": {
        "eight_board_hashes": "exact_E12_RR_rows",
        "mean_solve_ssim": EXPECTED_RR_MEAN_SOLVE_SSIM,
        "mean_final_ssim": EXPECTED_RR_MEAN_FINAL_SSIM,
        "absolute_tolerance": 1.0e-12,
    },
    "end_to_end_rule": dict(END_TO_END_RULE),
    "excluded": [
        "budget_sweep",
        "any_budget_other_than_RR96_and_CC192",
        "rank_or_energy_transplant",
        "model_scoring",
        "GPU_execution",
    ],
    "runtime_provenance": dict(EXPECTED_RUNTIME_PROVENANCE),
}

WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_RAW_CACHE_DIR = Path("E:/pazzle_work/edge_confidence/full_graph_cache")
DEFAULT_CALIBRATION_REPORT = (
    WORKSPACE / "artifacts" / "buddies_budget" / "calibration_v1.json"
)
DEFAULT_E12_REPORT = Path(
    "E:/pazzle_work/denoise_oracle/clean_score_oracle_calibration_v1.json"
)
DEFAULT_REPORT = Path(
    "E:/pazzle_work/cc192_oracle_e14/cc192_clean_oracle_discovery_v1.json"
)

BOARD_METRICS = (
    "placement",
    "neighbour",
    "right",
    "down",
    "solve_only_ssim",
    "final_ssim",
)


@dataclass(frozen=True)
class E14Paths:
    raw_cache_dir: Path
    calibration_report: Path
    e12_report: Path
    report: Path


@dataclass(frozen=True)
class _CCState:
    scene: e12.RawScene
    right: np.ndarray
    down: np.ndarray
    cache_sha256: str


def _require_e_drive(path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    if resolved.drive.upper() != "E:":
        raise E14ContractError(f"{label} must stay on E:, got {resolved}")
    return resolved


def _load_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise E14ContractError(f"could not read {label} {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise E14ContractError(f"{label} root is not an object")
    return payload


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    resolved = _require_e_drive(path, label="E14 report")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    temporary = resolved.with_name(f".{resolved.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, resolved)
    finally:
        if temporary.exists():
            temporary.unlink()


def _source_provenance() -> dict[str, str]:
    source = Path(__file__).resolve().parent
    paths = {
        "e14_cc192_oracle.py": source / "e14_cc192_oracle.py",
        "eval_e14_cc192_discovery.py": Path(__file__).resolve(),
    }
    return {name: e12.sha256_file(path) for name, path in sorted(paths.items())}


def _runtime_provenance() -> dict[str, str]:
    import cv2

    observed = {
        "python": platform.python_version(),
        "numpy": str(np.__version__),
        "scikit_image": str(skimage.__version__),
        "opencv": str(cv2.__version__),
        "opencv_build_sha256": hashlib.sha256(
            cv2.getBuildInformation().encode("utf-8")
        ).hexdigest(),
        "torch": str(e12.torch.__version__),
        "execution": "CPU_only",
    }
    if observed != EXPECTED_RUNTIME_PROVENANCE:
        raise E14ContractError(
            f"E14 runtime drifted: expected {EXPECTED_RUNTIME_PROVENANCE}, got {observed}"
        )
    return observed


def _verify_checkpoint_records(records: Mapping[str, Any]) -> None:
    if set(records) != {"ranker", "affinity_primary", "affinity_secondary"}:
        raise E14ContractError("E12 checkpoint records are incomplete")
    for role, raw_record in records.items():
        if not isinstance(raw_record, Mapping):
            raise E14ContractError(f"E12 {role} checkpoint record is malformed")
        path = Path(str(raw_record.get("path", ""))).resolve()
        if not path.is_file():
            raise E14ContractError(f"E12 {role} checkpoint is missing: {path}")
        if int(raw_record.get("size", -1)) != path.stat().st_size:
            raise E14ContractError(f"E12 {role} checkpoint size drifted")
        if str(raw_record.get("sha256", "")) != e12.sha256_file(path):
            raise E14ContractError(f"E12 {role} checkpoint SHA256 drifted")


def load_verified_e12_inputs(paths: E14Paths) -> tuple[
    Mapping[str, Any], Mapping[str, Any], list[e12.RawScene]
]:
    raw_cache_dir = _require_e_drive(paths.raw_cache_dir, label="raw score cache")
    e12_report_path = _require_e_drive(paths.e12_report, label="E12 report")
    digest = e12.sha256_file(e12_report_path)
    if digest != EXPECTED_E12_REPORT_SHA256:
        raise E14ContractError(
            f"E12 report SHA256 mismatch: expected {EXPECTED_E12_REPORT_SHA256}, got {digest}"
        )
    report = _load_json(e12_report_path, label="E12 report")
    if (
        report.get("schema") != e12.REPORT_SCHEMA
        or report.get("experiment") != e12.EXPERIMENT
        or report.get("status") != "complete"
        or report.get("protocol") != e12.ORACLE_PROTOCOL
        or report.get("protocol_sha256") != e12.canonical_digest(e12.ORACLE_PROTOCOL)
    ):
        raise E14ContractError("E12 report protocol/status drifted")
    inputs = report.get("inputs")
    if not isinstance(inputs, Mapping):
        raise E14ContractError("E12 report inputs are malformed")
    if Path(str(inputs.get("cache_dir", ""))).resolve() != raw_cache_dir:
        raise E14ContractError("requested raw score cache differs from E12")
    if Path(str(inputs.get("calibration_report", ""))).resolve() != paths.calibration_report.resolve():
        raise E14ContractError("requested calibration report differs from E12")

    calibration = e12.load_calibration_report(paths.calibration_report.resolve())
    if report.get("code_provenance") != e12.code_provenance():
        raise E14ContractError("source code used by E12 or reused by E14 has drifted")
    if report.get("scoring_code_provenance") != e12.scoring_code_provenance():
        raise E14ContractError("E12 clean score-cache provenance has drifted")
    checkpoint_records = report.get("checkpoints")
    if not isinstance(checkpoint_records, Mapping):
        raise E14ContractError("E12 checkpoint provenance is malformed")
    _verify_checkpoint_records(checkpoint_records)

    scenes = e12.load_raw_scenes(raw_cache_dir, e12.CALIBRATION_IDS)
    observed = e12.validate_scene_replay(scenes, calibration)
    if (
        report.get("scene_provenance") != observed
        or report.get("scene_provenance_digest") != e12.canonical_digest(observed)
    ):
        raise E14ContractError("E12 scene provenance differs from replayed bytes")
    rr_rows = report.get("rows", {}).get("RR") if isinstance(report.get("rows"), Mapping) else None
    if not isinstance(rr_rows, list):
        raise E14ContractError("E12 RR rows are missing")
    e12.verify_rr_replay(rr_rows, calibration)
    return report, calibration, scenes


def _e12_rr_rows(report: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    rows = report.get("rows")
    if not isinstance(rows, Mapping) or not isinstance(rows.get("RR"), list):
        raise E14ContractError("E12 RR rows are missing")
    try:
        return e12._rows_by_calibration_image(rows["RR"], label="E12 RR")
    except e12.OracleContractError as exc:
        raise E14ContractError(str(exc)) from exc


def _clean_cache_records(report: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    raw_records = report.get("score_caches")
    if not isinstance(raw_records, list) or len(raw_records) != len(e12.CALIBRATION_IDS):
        raise E14ContractError("E12 clean score-cache records are incomplete")
    records: dict[int, Mapping[str, Any]] = {}
    for record in raw_records:
        if not isinstance(record, Mapping):
            raise E14ContractError("E12 clean score-cache record is malformed")
        image = int(record.get("image", -1))
        if image in records or image not in e12.CALIBRATION_IDS:
            raise E14ContractError("E12 clean score-cache image IDs drifted")
        path = _require_e_drive(
            Path(str(record.get("path", ""))), label="E12 clean score cache"
        )
        expected = (
            DEFAULT_E12_REPORT.parent
            / "score_cache"
            / f"image_{image:04d}_clean_score_v1.npz"
        ).resolve()
        if path != expected or not path.is_file():
            raise E14ContractError(f"E12 clean cache path drifted for image {image}")
        if str(record.get("sha256", "")) != e12.sha256_file(path):
            raise E14ContractError(f"E12 clean cache SHA256 drifted for image {image}")
        records[image] = record
    if tuple(sorted(records)) != e12.CALIBRATION_IDS:
        raise E14ContractError("E12 clean score-cache records are incomplete")
    return records


def _load_cc_cache(
    scene: e12.RawScene,
    report: Mapping[str, Any],
    record: Mapping[str, Any],
) -> e12.CleanScoreCache:
    path = Path(str(record["path"])).resolve()
    clean_tiles = e12.clean_tiles_input_order(scene.target_uint8, scene.permutation)
    checkpoints = report.get("checkpoints")
    scoring_code = report.get("scoring_code_provenance")
    if not isinstance(checkpoints, Mapping) or not isinstance(scoring_code, Mapping):
        raise E14ContractError("E12 cache provenance is malformed")
    metadata = e12._cache_metadata(scene, clean_tiles, checkpoints, scoring_code)
    try:
        return e12._load_clean_score_cache(path, metadata, scene)
    except e12.OracleContractError as exc:
        raise E14ContractError(str(exc)) from exc


def _replay_rr96(
    scene: e12.RawScene, before: Mapping[str, Any]
) -> tuple[np.ndarray, float, float]:
    right, down = e12.dense_from_graph(
        scene.candidate_ids,
        np.ascontiguousarray(scene.base_scores, dtype=np.float32),
    )
    board, objective, seconds = e12.solve_dense(right, down)
    if e12.array_sha256(board.astype(np.int64, copy=False)) != before.get("board_sha256"):
        raise E14ContractError(f"RR96 board replay drifted for image {scene.image_id}")
    if not math.isclose(
        float(objective), float(before.get("objective", float("nan"))), rel_tol=0.0, abs_tol=1e-12
    ):
        raise E14ContractError(f"RR96 objective replay drifted for image {scene.image_id}")
    solved = np.ascontiguousarray(assemble(scene.tiles_uint8, board), dtype=np.uint8)
    if e12.array_sha256(solved) != before.get("solved_corrupted_canvas_sha256"):
        raise E14ContractError(f"RR96 solved canvas replay drifted for image {scene.image_id}")
    return board, float(objective), float(seconds)


def verify_rr_means(rr_rows: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    ordered = [rr_rows[image] for image in e12.CALIBRATION_IDS]
    solve_mean = float(np.mean([float(row["solve_only_ssim"]) for row in ordered]))
    final_mean = float(np.mean([float(row["final_ssim"]) for row in ordered]))
    if not math.isclose(
        solve_mean, EXPECTED_RR_MEAN_SOLVE_SSIM, rel_tol=0.0, abs_tol=1e-12
    ):
        raise E14ContractError("RR96 mean solve SSIM drifted")
    if not math.isclose(
        final_mean, EXPECTED_RR_MEAN_FINAL_SSIM, rel_tol=0.0, abs_tol=1e-12
    ):
        raise E14ContractError("RR96 mean final SSIM drifted")
    return {
        "expected_mean_solve_ssim": EXPECTED_RR_MEAN_SOLVE_SSIM,
        "observed_mean_solve_ssim": solve_mean,
        "expected_mean_final_ssim": EXPECTED_RR_MEAN_FINAL_SSIM,
        "observed_mean_final_ssim": final_mean,
        "board_hashes": {
            str(image): str(rr_rows[image]["board_sha256"])
            for image in e12.CALIBRATION_IDS
        },
    }


def summarize_structure(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != len(e12.CALIBRATION_IDS):
        raise E14ContractError("CC192 structure requires exactly eight rows")
    precision = np.asarray(
        [float(row["selected_edge_precision"]) for row in rows], dtype=np.float64
    )
    coverage = np.asarray(
        [float(row["component_coverage"]) for row in rows], dtype=np.float64
    )
    return {
        "images": len(rows),
        "mean_selected_edge_precision": float(precision.mean()),
        "worst_selected_edge_precision": float(precision.min()),
        "total_selected_edges": int(sum(int(row["selected_edge_count"]) for row in rows)),
        "total_true_edges": int(sum(int(row["true_edge_count"]) for row in rows)),
        "aggregate_selected_edge_precision": float(
            sum(int(row["true_edge_count"]) for row in rows)
            / max(1, sum(int(row["selected_edge_count"]) for row in rows))
        ),
        "mean_component_coverage": float(coverage.mean()),
        "worst_component_coverage": float(coverage.min()),
        "mean_covered_tiles": float(np.mean([int(row["covered_tiles"]) for row in rows])),
        "mean_largest_component": float(
            np.mean([int(row["largest_component"]) for row in rows])
        ),
    }


def structural_decision(summary: Mapping[str, Any]) -> dict[str, Any]:
    observed = {
        "mean_selected_edge_precision": float(summary["mean_selected_edge_precision"]),
        "mean_component_coverage": float(summary["mean_component_coverage"]),
    }
    checks = {
        "mean_selected_edge_precision": observed["mean_selected_edge_precision"]
        >= STRUCTURAL_RULE["mean_selected_edge_precision_min"],
        "mean_component_coverage": observed["mean_component_coverage"]
        >= STRUCTURAL_RULE["mean_component_coverage_min"],
    }
    passed = all(checks.values())
    return {
        "status": "go_end_to_end" if passed else "kill_before_end_to_end",
        "passed": passed,
        "thresholds": dict(STRUCTURAL_RULE),
        "observed": observed,
        "checks": checks,
    }


def evaluate_cc192_board(
    scene: e12.RawScene,
    board: np.ndarray,
    objective: float,
    *,
    restorer: Callable[[np.ndarray], np.ndarray] = e12.fixed_nlm,
) -> dict[str, Any]:
    board = e12._strict_board(np.asarray(board))
    target = np.asarray(scene.target_uint8)
    tiles = np.asarray(scene.tiles_uint8)
    if target.shape != (e12.IMG, e12.IMG, 3) or target.dtype != np.uint8:
        raise E14ContractError("scene target geometry/dtype drifted")
    if tiles.shape != (e12.NFRAG, e12.FS, e12.FS, 3) or tiles.dtype != np.uint8:
        raise E14ContractError("scene corrupted tile geometry/dtype drifted")
    truth_board = np.argsort(np.asarray(scene.permutation, dtype=np.int64))
    placement = placement_accuracy(board, truth_board)[0]
    neighbour, right, down = neighbour_accuracy(board, truth_board)
    solved = np.ascontiguousarray(assemble(tiles, board), dtype=np.uint8)
    restored = np.asarray(restorer(solved.copy()))
    if restored.shape != target.shape or restored.dtype != np.uint8:
        raise E14ContractError("fixed NLM restorer returned invalid geometry/dtype")
    return {
        "placement": float(placement),
        "neighbour": float(neighbour),
        "right": float(right),
        "down": float(down),
        "solve_only_ssim": float(sk_ssim(target, solved, channel_axis=2, data_range=255)),
        "final_ssim": float(sk_ssim(target, restored, channel_axis=2, data_range=255)),
        "objective": float(objective),
        "board_sha256": e12.array_sha256(board.astype(np.int64, copy=False)),
        "solved_corrupted_canvas_sha256": e12.array_sha256(solved),
        "restored_canvas_sha256": e12.array_sha256(restored),
    }


def paired_summary(
    candidate_rows: Sequence[Mapping[str, Any]], rr_rows: Mapping[int, Mapping[str, Any]]
) -> dict[str, Any]:
    if len(candidate_rows) != len(e12.CALIBRATION_IDS):
        raise E14ContractError("CC192 comparison requires exactly eight candidate rows")
    candidate = {int(row["image"]): row for row in candidate_rows}
    if tuple(sorted(candidate)) != e12.CALIBRATION_IDS:
        raise E14ContractError("CC192 comparison image IDs drifted")
    metrics: dict[str, Any] = {}
    for metric in BOARD_METRICS:
        baseline_values = np.asarray(
            [float(rr_rows[image][metric]) for image in e12.CALIBRATION_IDS], dtype=np.float64
        )
        candidate_values = np.asarray(
            [float(candidate[image][metric]) for image in e12.CALIBRATION_IDS], dtype=np.float64
        )
        delta = candidate_values - baseline_values
        metrics[metric] = {
            "baseline_mean": float(baseline_values.mean()),
            "candidate_mean": float(candidate_values.mean()),
            "mean_delta": float(delta.mean()),
            "median_delta": float(np.median(delta)),
            "best_delta": float(delta.max()),
            "worst_delta": float(delta.min()),
            "wins": int(np.sum(delta > 0.0)),
            "ties": int(np.sum(delta == 0.0)),
            "losses": int(np.sum(delta < 0.0)),
        }
    return {"candidate_arm": "CC192", "baseline_arm": "RR96", "metrics": metrics}


def end_to_end_decision(summary: Mapping[str, Any]) -> dict[str, Any]:
    solve = summary["metrics"]["solve_only_ssim"]
    final = summary["metrics"]["final_ssim"]
    observed = {
        "mean_solve_ssim_delta": float(solve["mean_delta"]),
        "mean_final_ssim_delta": float(final["mean_delta"]),
        "final_wins": int(final["wins"]),
        "worst_final_delta": float(final["worst_delta"]),
    }
    checks = {
        "mean_solve_ssim_delta": observed["mean_solve_ssim_delta"]
        >= float(END_TO_END_RULE["cc192_minus_rr96_mean_solve_ssim_min"]),
        "mean_final_ssim_delta": observed["mean_final_ssim_delta"]
        >= float(END_TO_END_RULE["cc192_minus_rr96_mean_final_ssim_min"]),
        "final_wins": observed["final_wins"]
        >= int(END_TO_END_RULE["cc192_minus_rr96_final_wins_min"]),
        "worst_final_delta": observed["worst_final_delta"]
        >= float(END_TO_END_RULE["cc192_minus_rr96_worst_final_delta_min"]),
    }
    passed = all(checks.values())
    return {
        "status": "go_candidate" if passed else "kill_cc192",
        "passed": passed,
        "thresholds": dict(END_TO_END_RULE),
        "observed": observed,
        "checks": checks,
        "scope": "clean_oracle_diagnostic_only_not_deployable",
    }


def _run_contract(
    paths: E14Paths,
    report: Mapping[str, Any],
    scenes: Sequence[e12.RawScene],
    clean_records: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "protocol_sha256": e12.canonical_digest(E14_PROTOCOL),
        "e12_report": {
            "path": str(paths.e12_report.resolve()),
            "sha256": EXPECTED_E12_REPORT_SHA256,
        },
        "calibration_report": {
            "path": str(paths.calibration_report.resolve()),
            "sha256": e12.CALIBRATION_REPORT_SHA256,
        },
        "raw_cache_dir": str(paths.raw_cache_dir.resolve()),
        "scene_provenance_digest": str(report["scene_provenance_digest"]),
        "raw_score_caches": {
            str(scene.image_id): {"path": str(scene.cache_path), "sha256": scene.cache_sha256}
            for scene in scenes
        },
        "clean_score_caches": {
            str(image): {
                "path": str(Path(str(record["path"])).resolve()),
                "sha256": str(record["sha256"]),
            }
            for image, record in sorted(clean_records.items())
        },
        "e12_code_provenance": report["code_provenance"],
        "e14_code_provenance": _source_provenance(),
        "runtime_provenance": _runtime_provenance(),
    }


def run_discovery(paths: E14Paths) -> Mapping[str, Any]:
    started = time.perf_counter()
    report_path = _require_e_drive(paths.report, label="E14 report")
    if report_path.suffix.lower() != ".json":
        raise E14ContractError("E14 report must be a .json file")
    if report_path in {paths.e12_report.resolve(), paths.calibration_report.resolve()}:
        raise E14ContractError("E14 report must not overwrite an input report")
    raw_cache_dir = paths.raw_cache_dir.resolve()
    clean_cache_dir = (DEFAULT_E12_REPORT.parent / "score_cache").resolve()
    if report_path.is_relative_to(raw_cache_dir) or report_path.is_relative_to(clean_cache_dir):
        raise E14ContractError("E14 report must not be written inside an input cache directory")

    e12_report, _calibration, scenes = load_verified_e12_inputs(paths)
    rr_rows = _e12_rr_rows(e12_report)
    rr_verification = verify_rr_means(rr_rows)
    clean_records = _clean_cache_records(e12_report)
    contract = _run_contract(paths, e12_report, scenes, clean_records)
    contract_digest = e12.canonical_digest(contract)

    if report_path.is_file():
        existing = _load_json(report_path, label="existing E14 report")
        if (
            existing.get("status") == "complete"
            and existing.get("schema") == REPORT_SCHEMA
            and existing.get("experiment") == EXPERIMENT
            and existing.get("run_contract_sha256") == contract_digest
            and existing.get("protocol") == E14_PROTOCOL
            and isinstance(existing.get("decisions"), Mapping)
        ):
            return existing
        if existing.get("run_contract_sha256") != contract_digest:
            raise E14ContractError("existing E14 report belongs to different input/code bytes")

    output: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "schema": REPORT_SCHEMA,
        "experiment": EXPERIMENT,
        "status": "in_progress",
        "stage": "rr_replay_and_structure",
        "protocol": E14_PROTOCOL,
        "protocol_sha256": e12.canonical_digest(E14_PROTOCOL),
        "run_contract": contract,
        "run_contract_sha256": contract_digest,
        "rr_reproducibility": rr_verification,
        "rows": {"RR96": [], "CC192_structure": [], "CC192": []},
        "completed_structure_images": [],
        "completed_end_to_end_images": [],
    }
    _atomic_write_json(report_path, output)

    states: list[_CCState] = []
    try:
        for scene in scenes:
            before = rr_rows[scene.image_id]
            _board, _objective, rr_seconds = _replay_rr96(scene, before)
            output["rows"]["RR96"].append(
                {
                    "image": int(scene.image_id),
                    "validation_name": str(scene.validation_name),
                    **{key: before[key] for key in BOARD_METRICS},
                    "objective": float(before["objective"]),
                    "board_sha256": str(before["board_sha256"]),
                    "solved_corrupted_canvas_sha256": str(
                        before["solved_corrupted_canvas_sha256"]
                    ),
                    "restored_canvas_sha256": str(before["restored_canvas_sha256"]),
                    "solver_replay_seconds": rr_seconds,
                    "source": "exact_E12_RR_record_no_second_NLM_call",
                }
            )

            clean_cache = _load_cc_cache(scene, e12_report, clean_records[scene.image_id])
            right, down = e12.dense_from_graph(clean_cache.cc_candidates, clean_cache.cc_scores)
            structure = cc192.measure_cc192_structure(right, down, scene.permutation)
            if structure.selected_edge_count != cc192.MAX_EDGES:
                raise E14ContractError(
                    f"CC192 selected only {structure.selected_edge_count} edges for "
                    f"image {scene.image_id}; expected exactly {cc192.MAX_EDGES}"
                )
            structure_row = {
                "image": int(scene.image_id),
                "validation_name": str(scene.validation_name),
                **asdict(structure),
            }
            output["rows"]["CC192_structure"].append(structure_row)
            states.append(
                _CCState(
                    scene=scene,
                    right=right,
                    down=down,
                    cache_sha256=clean_cache.sha256,
                )
            )
            output["completed_structure_images"].append(int(scene.image_id))
            output["runtime_seconds"] = float(time.perf_counter() - started)
            _atomic_write_json(report_path, output)

        structure_summary = summarize_structure(output["rows"]["CC192_structure"])
        structure_gate = structural_decision(structure_summary)
        output["structural_summary"] = structure_summary
        output["decisions"] = {"structural": structure_gate, "end_to_end": {"status": "not_run"}}
        if not structure_gate["passed"]:
            output["status"] = "complete"
            output["stage"] = "killed_structural"
            output["runtime_seconds"] = float(time.perf_counter() - started)
            _atomic_write_json(report_path, output)
            return output

        output["stage"] = "end_to_end"
        _atomic_write_json(report_path, output)
        for state in states:
            solve_started = time.perf_counter()
            board, objective = cc192.solve_cc192(state.right, state.down)
            solver_seconds = time.perf_counter() - solve_started
            metrics = evaluate_cc192_board(state.scene, board, objective)
            output["rows"]["CC192"].append(
                {
                    "image": int(state.scene.image_id),
                    "validation_name": str(state.scene.validation_name),
                    "arm": "CC192",
                    **metrics,
                    "solver_seconds": float(solver_seconds),
                    "clean_score_cache_sha256": state.cache_sha256,
                }
            )
            output["completed_end_to_end_images"].append(int(state.scene.image_id))
            output["runtime_seconds"] = float(time.perf_counter() - started)
            _atomic_write_json(report_path, output)

        comparison = paired_summary(output["rows"]["CC192"], rr_rows)
        end_to_end = end_to_end_decision(comparison)
        output["comparison"] = comparison
        output["decisions"]["end_to_end"] = end_to_end
        output["status"] = "complete"
        output["stage"] = end_to_end["status"]
        output["runtime_seconds"] = float(time.perf_counter() - started)
        _atomic_write_json(report_path, output)
        return output
    except Exception as exc:
        output["status"] = "failed"
        output["error"] = f"{type(exc).__name__}: {exc}"
        output["runtime_seconds"] = float(time.perf_counter() - started)
        _atomic_write_json(report_path, output)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run fixed CPU-only E14 CC192 discovery on opened E12 IDs 10..17."
    )
    parser.add_argument("--raw-cache-dir", type=Path, default=DEFAULT_RAW_CACHE_DIR)
    parser.add_argument(
        "--calibration-report", type=Path, default=DEFAULT_CALIBRATION_REPORT
    )
    parser.add_argument("--e12-report", type=Path, default=DEFAULT_E12_REPORT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = run_discovery(
        E14Paths(
            raw_cache_dir=args.raw_cache_dir,
            calibration_report=args.calibration_report,
            e12_report=args.e12_report,
            report=args.report,
        )
    )
    print(
        json.dumps(
            {
                "report": str(args.report.resolve()),
                "stage": result["stage"],
                "structural": result["decisions"]["structural"]["status"],
                "end_to_end": result["decisions"]["end_to_end"]["status"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

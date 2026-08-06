"""CPU-only E13 discovery of a global cyclic row/column origin error.

This evaluator is intentionally restricted to the already-open E12 scenes
10..17 and their byte-pinned score caches.  It replays exactly two Rank96
boards:

* RR96: raw candidates plus raw scores (deployable discovery path);
* CC96: clean-oracle candidates plus clean-oracle scores (diagnostic only).

For each completed upright board, :mod:`e13_torus_origin` selects one global
row/column roll from fixed scaled CIE-Lab depth-1 toroidal seam MSE.  E12's
before metrics are reused; NLM(10) is called exactly once for the rolled board.
No GPU, scoring model, rotation, reflection, tile edit, sweep, or threshold CLI
control exists here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import skimage
from skimage.metrics import structural_similarity as sk_ssim

import e13_torus_origin as torus
import eval_clean_score_oracle as e12
from imgio import assemble
from placement_metrics import neighbour_accuracy, placement_accuracy


class E13ContractError(RuntimeError):
    """The isolated E13 discovery contract or an E12 input drifted."""


SCHEMA_VERSION = 1
REPORT_SCHEMA = "pazzle-e13-torus-origin-discovery-report-v1"
EXPERIMENT = "e13_torus_global_origin_discovery_v1"
ARMS = ("RR96", "CC96")
EXPECTED_E12_REPORT_SHA256 = (
    "16ceecfea99e006a1126b17d7d58fb5d188ec694c6a5097310dfe021bd2f901a"
)
EXPECTED_RUNTIME_PROVENANCE = {
    "python": "3.13.6",
    "numpy": "2.2.6",
    "scikit_image": "0.26.0",
    "opencv": "4.13.0",
    "opencv_build_sha256": "ad2e3bc9bf8eb9d40a90e2f61a2c7667acee8a22e860778ea3378a4ed68f2be7",
    "torch": "2.11.0+cu128",
    "execution": "CPU_only",
}

RR_PROMOTION_RULE: dict[str, float | int] = {
    "mean_solve_delta_min": 0.002,
    "mean_final_delta_min": 0.003,
    "final_wins_min": 5,
    "worst_final_delta_min": -0.015,
}
CC_ORIGIN_DIAGNOSIS_RULE: dict[str, float | int | bool] = {
    "mean_solve_delta_min": 0.0075,
    "mean_final_delta_min": 0.015,
    "final_wins_min": 6,
    "worst_final_delta_min": -0.020,
    "absolute_cc_solve_at_least_rr_baseline": True,
    "absolute_cc_final_at_least_rr_baseline": True,
}

E13_PROTOCOL: dict[str, Any] = {
    "schema": "pazzle-e13-torus-origin-discovery-v1",
    "role": "opened_e12_discovery_not_confirmation_or_submission",
    "calibration_ids": [10, 11, 12, 13, 14, 15, 16, 17],
    "input_e12_report_sha256": EXPECTED_E12_REPORT_SHA256,
    "geometry": {
        "grid": 24,
        "tile_size": 20,
        "num_tiles": 576,
        "orientation": "upright_only",
        "rotation": False,
        "reflection": False,
        "tile_changes": False,
    },
    "selector": {
        "colour_space": "skimage_CIE_Lab_from_float32_RGB_0_1",
        "lab_scale": [100.0, 128.0, 128.0],
        "depth": 1,
        "horizontal_cuts": 24,
        "vertical_cuts": 24,
        "cut_index_k": "boundary_between_k_minus_1_and_k_mod_24",
        "choose": "maximum_energy_seam_to_exclude_per_axis",
        "tie": "numpy_argmax_first_cut0_is_no_roll",
        "transform": "one_global_np_roll_rows_and_columns",
        "score_pixels": "original_corrupted_upright_tiles_for_both_arms",
        "label_free": True,
    },
    "arms": {
        "RR96": {
            "source": "E12_RR_raw_candidates_raw_scores_rank96",
            "role": "deployable_discovery",
        },
        "CC96": {
            "source": "E12_CC_clean_oracle_candidates_scores_rank96",
            "role": "diagnostic_only_not_deployable",
        },
    },
    "baseline": "reuse_exact_E12_before_metrics_and_verify_replayed_board_hash",
    "restoration": {
        "name": "opencv_fast_nlm_colored",
        "h": 10,
        "h_color": 10,
        "template_window": 7,
        "search_window": 21,
        "calls": "once_after_roll_per_arm_per_scene",
    },
    "decision_rules": {
        "rr_promotion_candidate": dict(RR_PROMOTION_RULE),
        "cc_origin_diagnosis": dict(CC_ORIGIN_DIAGNOSIS_RULE),
    },
    "compute": "CPU_only_existing_E12_caches_no_model_scoring",
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
    "E:/pazzle_work/torus_origin_e13/torus_origin_discovery_v1.json"
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
class E13Paths:
    raw_cache_dir: Path
    calibration_report: Path
    e12_report: Path
    report: Path


def _require_e_drive(path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    if resolved.drive.upper() != "E:":
        raise E13ContractError(f"{label} must stay on E:, got {resolved}")
    return resolved


def _load_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise E13ContractError(f"could not read {label} {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise E13ContractError(f"{label} root is not an object")
    return payload


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace one E:-resident JSON report after an fsync."""

    resolved = _require_e_drive(path, label="E13 report")
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
        "e13_torus_origin.py": source / "e13_torus_origin.py",
        "eval_e13_torus_origin.py": Path(__file__).resolve(),
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
        raise E13ContractError(
            "E13 runtime differs from the environment used to open E12: "
            f"expected {EXPECTED_RUNTIME_PROVENANCE}, got {observed}"
        )
    return observed


def _verify_checkpoint_records(records: Mapping[str, Any]) -> None:
    if set(records) != {"ranker", "affinity_primary", "affinity_secondary"}:
        raise E13ContractError("E12 checkpoint records are incomplete")
    for role, raw_record in records.items():
        if not isinstance(raw_record, Mapping):
            raise E13ContractError(f"E12 {role} checkpoint record is malformed")
        path = Path(str(raw_record.get("path", ""))).resolve()
        if not path.is_file():
            raise E13ContractError(f"E12 {role} checkpoint is missing: {path}")
        if int(raw_record.get("size", -1)) != path.stat().st_size:
            raise E13ContractError(f"E12 {role} checkpoint size drifted")
        if str(raw_record.get("sha256", "")) != e12.sha256_file(path):
            raise E13ContractError(f"E12 {role} checkpoint SHA256 drifted")


def load_verified_e12_inputs(paths: E13Paths) -> tuple[
    Mapping[str, Any], Mapping[str, Any], list[e12.RawScene]
]:
    """Load only the exact opened E12 report, scenes, and current code bytes."""

    raw_cache_dir = _require_e_drive(paths.raw_cache_dir, label="raw score cache")
    e12_report_path = _require_e_drive(paths.e12_report, label="E12 report")
    digest = e12.sha256_file(e12_report_path)
    if digest != EXPECTED_E12_REPORT_SHA256:
        raise E13ContractError(
            "E12 report SHA256 mismatch: "
            f"expected {EXPECTED_E12_REPORT_SHA256}, got {digest}"
        )
    report = _load_json(e12_report_path, label="E12 report")
    if (
        report.get("schema") != e12.REPORT_SCHEMA
        or report.get("experiment") != e12.EXPERIMENT
        or report.get("status") != "complete"
        or report.get("protocol") != e12.ORACLE_PROTOCOL
        or report.get("protocol_sha256") != e12.canonical_digest(e12.ORACLE_PROTOCOL)
    ):
        raise E13ContractError("E12 report protocol/status drifted")
    inputs = report.get("inputs")
    if not isinstance(inputs, Mapping):
        raise E13ContractError("E12 report inputs are malformed")
    if Path(str(inputs.get("cache_dir", ""))).resolve() != raw_cache_dir:
        raise E13ContractError("requested raw score cache differs from E12")
    if Path(str(inputs.get("calibration_report", ""))).resolve() != paths.calibration_report.resolve():
        raise E13ContractError("requested calibration report differs from E12")

    calibration = e12.load_calibration_report(paths.calibration_report.resolve())
    if report.get("code_provenance") != e12.code_provenance():
        raise E13ContractError("source code used by E12 or reused by E13 has drifted")
    if report.get("scoring_code_provenance") != e12.scoring_code_provenance():
        raise E13ContractError("E12 score-cache source provenance has drifted")
    checkpoint_records = report.get("checkpoints")
    if not isinstance(checkpoint_records, Mapping):
        raise E13ContractError("E12 checkpoint provenance is malformed")
    _verify_checkpoint_records(checkpoint_records)

    scenes = e12.load_raw_scenes(raw_cache_dir, e12.CALIBRATION_IDS)
    observed = e12.validate_scene_replay(scenes, calibration)
    if (
        report.get("scene_provenance") != observed
        or report.get("scene_provenance_digest") != e12.canonical_digest(observed)
    ):
        raise E13ContractError("E12 scene provenance differs from replayed bytes")
    rr_rows = report.get("rows", {}).get("RR") if isinstance(report.get("rows"), Mapping) else None
    if not isinstance(rr_rows, list):
        raise E13ContractError("E12 RR rows are missing")
    e12.verify_rr_replay(rr_rows, calibration)
    return report, calibration, scenes


def _e12_rows(report: Mapping[str, Any], arm: str) -> dict[int, Mapping[str, Any]]:
    rows = report.get("rows")
    if not isinstance(rows, Mapping) or not isinstance(rows.get(arm), list):
        raise E13ContractError(f"E12 {arm} rows are missing")
    try:
        return e12._rows_by_calibration_image(rows[arm], label=f"E12 {arm}")
    except e12.OracleContractError as exc:
        raise E13ContractError(str(exc)) from exc


def _clean_cache_records(report: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    raw_records = report.get("score_caches")
    if not isinstance(raw_records, list) or len(raw_records) != len(e12.CALIBRATION_IDS):
        raise E13ContractError("E12 clean score-cache records are incomplete")
    records: dict[int, Mapping[str, Any]] = {}
    for record in raw_records:
        if not isinstance(record, Mapping):
            raise E13ContractError("E12 clean score-cache record is malformed")
        image = int(record.get("image", -1))
        if image in records or image not in e12.CALIBRATION_IDS:
            raise E13ContractError("E12 clean score-cache image IDs drifted")
        path = _require_e_drive(
            Path(str(record.get("path", ""))), label="E12 clean score cache"
        )
        expected_path = (
            DEFAULT_E12_REPORT.parent
            / "score_cache"
            / f"image_{image:04d}_clean_score_v1.npz"
        ).resolve()
        if path != expected_path or not path.is_file():
            raise E13ContractError(f"E12 clean cache path drifted for image {image}")
        if str(record.get("sha256", "")) != e12.sha256_file(path):
            raise E13ContractError(f"E12 clean cache SHA256 drifted for image {image}")
        records[image] = record
    if tuple(sorted(records)) != e12.CALIBRATION_IDS:
        raise E13ContractError("E12 clean score-cache records are incomplete")
    return records


def _load_cc_cache(
    scene: e12.RawScene,
    report: Mapping[str, Any],
    record: Mapping[str, Any],
) -> e12.CleanScoreCache:
    path = _require_e_drive(Path(str(record.get("path", ""))), label="E12 clean score cache")
    expected_path = (
        DEFAULT_E12_REPORT.parent
        / "score_cache"
        / f"image_{scene.image_id:04d}_clean_score_v1.npz"
    ).resolve()
    if path != expected_path:
        raise E13ContractError(f"E12 clean cache path drifted for image {scene.image_id}")
    if str(record.get("sha256", "")) != e12.sha256_file(path):
        raise E13ContractError(f"E12 clean cache SHA256 drifted for image {scene.image_id}")
    clean_tiles = e12.clean_tiles_input_order(scene.target_uint8, scene.permutation)
    checkpoints = report.get("checkpoints")
    scoring_code = report.get("scoring_code_provenance")
    if not isinstance(checkpoints, Mapping) or not isinstance(scoring_code, Mapping):
        raise E13ContractError("E12 cache provenance is malformed")
    metadata = e12._cache_metadata(scene, clean_tiles, checkpoints, scoring_code)
    try:
        return e12._load_clean_score_cache(path, metadata, scene)
    except e12.OracleContractError as exc:
        raise E13ContractError(str(exc)) from exc


def _replay_rank96_board(
    scene: e12.RawScene,
    *,
    arm: str,
    cc_cache: e12.CleanScoreCache | None,
    before: Mapping[str, Any],
) -> tuple[np.ndarray, float, float]:
    if arm == "RR96":
        candidates = scene.candidate_ids
        scores = np.ascontiguousarray(scene.base_scores, dtype=np.float32)
    elif arm == "CC96" and cc_cache is not None:
        candidates = cc_cache.cc_candidates
        scores = cc_cache.cc_scores
    else:
        raise E13ContractError(f"invalid E13 arm/cache combination: {arm}")
    right, down = e12.dense_from_graph(candidates, scores)
    board, objective, solver_seconds = e12.solve_dense(right, down)
    board_hash = e12.array_sha256(board.astype(np.int64, copy=False))
    if board_hash != before.get("board_sha256"):
        raise E13ContractError(
            f"{arm} board replay drifted for image {scene.image_id}: "
            f"expected {before.get('board_sha256')}, got {board_hash}"
        )
    if not math.isclose(
        float(objective), float(before.get("objective", float("nan"))), rel_tol=0.0, abs_tol=1e-12
    ):
        raise E13ContractError(f"{arm} objective replay drifted for image {scene.image_id}")
    solved = np.ascontiguousarray(assemble(scene.tiles_uint8, board), dtype=np.uint8)
    if e12.array_sha256(solved) != before.get("solved_corrupted_canvas_sha256"):
        raise E13ContractError(f"{arm} solved canvas replay drifted for image {scene.image_id}")
    return board, float(objective), float(solver_seconds)


def evaluate_rolled_board(
    scene: e12.RawScene,
    board: np.ndarray,
    objective: float,
    *,
    restorer: Callable[[np.ndarray], np.ndarray] = e12.fixed_nlm,
) -> dict[str, Any]:
    """Measure one rolled board, calling the fixed restorer exactly once."""

    board = e12._strict_board(np.asarray(board))
    target = np.asarray(scene.target_uint8)
    tiles = np.asarray(scene.tiles_uint8)
    if target.shape != (e12.IMG, e12.IMG, 3) or target.dtype != np.uint8:
        raise E13ContractError("scene target geometry/dtype drifted")
    if tiles.shape != (e12.NFRAG, e12.FS, e12.FS, 3) or tiles.dtype != np.uint8:
        raise E13ContractError("scene corrupted tile geometry/dtype drifted")
    truth_board = np.argsort(np.asarray(scene.permutation, dtype=np.int64))
    placement = placement_accuracy(board, truth_board)[0]
    neighbour, right, down = neighbour_accuracy(board, truth_board)
    solved = np.ascontiguousarray(assemble(tiles, board), dtype=np.uint8)
    # E13 intentionally has exactly this one restoration call per evaluated
    # rolled arm.  E12's recorded before-final metric is not recomputed.
    restored = np.asarray(restorer(solved.copy()))
    if restored.shape != target.shape or restored.dtype != np.uint8:
        raise E13ContractError("fixed NLM restorer returned invalid geometry/dtype")
    return {
        "placement": float(placement),
        "neighbour": float(neighbour),
        "right": float(right),
        "down": float(down),
        "solve_only_ssim": float(sk_ssim(target, solved, channel_axis=2, data_range=255)),
        "final_ssim": float(sk_ssim(target, restored, channel_axis=2, data_range=255)),
        "pre_roll_solver_objective": float(objective),
        "board_sha256": e12.array_sha256(board.astype(np.int64, copy=False)),
        "solved_corrupted_canvas_sha256": e12.array_sha256(solved),
        "restored_canvas_sha256": e12.array_sha256(restored),
    }


def _selection_record(selection: torus.TorusOriginSelection) -> dict[str, Any]:
    return {
        "row_cut": int(selection.row_cut),
        "column_cut": int(selection.column_cut),
        "row_roll": int(selection.row_roll),
        "column_roll": int(selection.column_roll),
        "horizontal_cut_energies": [float(value) for value in selection.horizontal_cut_energies],
        "vertical_cut_energies": [float(value) for value in selection.vertical_cut_energies],
        "excluded_horizontal_energy": selection.excluded_horizontal_energy,
        "excluded_vertical_energy": selection.excluded_vertical_energy,
        "retained_internal_horizontal_mse": selection.retained_internal_horizontal_mse,
        "retained_internal_vertical_mse": selection.retained_internal_vertical_mse,
        "retained_internal_lab_score": selection.retained_internal_lab_score,
        "tie_rule": "numpy_argmax_first_cut0_is_no_roll",
    }


def _paired_before_after(rows: Sequence[Mapping[str, Any]], *, arm: str) -> dict[str, Any]:
    if len(rows) != len(e12.CALIBRATION_IDS):
        raise E13ContractError(f"{arm} summary requires exactly eight rows")
    images = [int(row.get("image", -1)) for row in rows]
    if tuple(sorted(images)) != e12.CALIBRATION_IDS or len(set(images)) != len(images):
        raise E13ContractError(f"{arm} summary image IDs drifted")
    metrics: dict[str, Any] = {}
    for metric in BOARD_METRICS:
        before = np.asarray([float(row["before"][metric]) for row in rows], dtype=np.float64)
        after = np.asarray([float(row["after"][metric]) for row in rows], dtype=np.float64)
        delta = after - before
        metrics[metric] = {
            "before_mean": float(before.mean()),
            "after_mean": float(after.mean()),
            "mean_delta": float(delta.mean()),
            "median_delta": float(np.median(delta)),
            "best_delta": float(delta.max()),
            "worst_delta": float(delta.min()),
            "wins": int(np.sum(delta > 0.0)),
            "ties": int(np.sum(delta == 0.0)),
            "losses": int(np.sum(delta < 0.0)),
        }
    return {"arm": arm, "images": len(rows), "metrics": metrics}


def rr_promotion_decision(summary: Mapping[str, Any]) -> dict[str, Any]:
    metrics = summary["metrics"]
    solve = metrics["solve_only_ssim"]
    final = metrics["final_ssim"]
    observed = {
        "mean_solve_delta": float(solve["mean_delta"]),
        "mean_final_delta": float(final["mean_delta"]),
        "final_wins": int(final["wins"]),
        "worst_final_delta": float(final["worst_delta"]),
    }
    checks = {
        "mean_solve_delta": observed["mean_solve_delta"]
        >= float(RR_PROMOTION_RULE["mean_solve_delta_min"]),
        "mean_final_delta": observed["mean_final_delta"]
        >= float(RR_PROMOTION_RULE["mean_final_delta_min"]),
        "final_wins": observed["final_wins"] >= int(RR_PROMOTION_RULE["final_wins_min"]),
        "worst_final_delta": observed["worst_final_delta"]
        >= float(RR_PROMOTION_RULE["worst_final_delta_min"]),
    }
    passed = all(checks.values())
    return {
        "status": "promotion_candidate" if passed else "kill_rr_torus_origin",
        "passed": passed,
        "thresholds": dict(RR_PROMOTION_RULE),
        "observed": observed,
        "checks": checks,
        "scope": "opened_E12_discovery_requires_fresh_confirmation_if_passed",
    }


def cc_origin_diagnosis_decision(
    cc_summary: Mapping[str, Any], rr_summary: Mapping[str, Any]
) -> dict[str, Any]:
    cc_metrics = cc_summary["metrics"]
    rr_metrics = rr_summary["metrics"]
    solve = cc_metrics["solve_only_ssim"]
    final = cc_metrics["final_ssim"]
    rr_solve_baseline = float(rr_metrics["solve_only_ssim"]["before_mean"])
    rr_final_baseline = float(rr_metrics["final_ssim"]["before_mean"])
    observed = {
        "mean_solve_delta": float(solve["mean_delta"]),
        "mean_final_delta": float(final["mean_delta"]),
        "final_wins": int(final["wins"]),
        "worst_final_delta": float(final["worst_delta"]),
        "absolute_cc_solve_after": float(solve["after_mean"]),
        "absolute_cc_final_after": float(final["after_mean"]),
        "rr_baseline_solve": rr_solve_baseline,
        "rr_baseline_final": rr_final_baseline,
    }
    checks = {
        "mean_solve_delta": observed["mean_solve_delta"]
        >= float(CC_ORIGIN_DIAGNOSIS_RULE["mean_solve_delta_min"]),
        "mean_final_delta": observed["mean_final_delta"]
        >= float(CC_ORIGIN_DIAGNOSIS_RULE["mean_final_delta_min"]),
        "final_wins": observed["final_wins"]
        >= int(CC_ORIGIN_DIAGNOSIS_RULE["final_wins_min"]),
        "worst_final_delta": observed["worst_final_delta"]
        >= float(CC_ORIGIN_DIAGNOSIS_RULE["worst_final_delta_min"]),
        "absolute_cc_solve_at_least_rr_baseline": observed["absolute_cc_solve_after"]
        >= rr_solve_baseline,
        "absolute_cc_final_at_least_rr_baseline": observed["absolute_cc_final_after"]
        >= rr_final_baseline,
    }
    passed = all(checks.values())
    return {
        "status": "origin_hypothesis_supported" if passed else "origin_hypothesis_insufficient",
        "passed": passed,
        "thresholds": dict(CC_ORIGIN_DIAGNOSIS_RULE),
        "observed": observed,
        "checks": checks,
        "scope": "clean_oracle_diagnostic_only_not_deployable",
    }


def _run_contract(
    paths: E13Paths,
    report: Mapping[str, Any],
    scenes: Sequence[e12.RawScene],
    clean_records: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "protocol_sha256": e12.canonical_digest(E13_PROTOCOL),
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
            str(scene.image_id): {
                "path": str(scene.cache_path),
                "sha256": str(scene.cache_sha256),
            }
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
        "e13_code_provenance": _source_provenance(),
        "runtime_provenance": _runtime_provenance(),
    }


def run_discovery(paths: E13Paths) -> Mapping[str, Any]:
    """Execute the fixed cached CPU discovery and atomically persist progress."""

    started = time.perf_counter()
    report_path = _require_e_drive(paths.report, label="E13 report")
    if report_path.suffix.lower() != ".json":
        raise E13ContractError("E13 report must be a .json file")
    protected = {
        paths.e12_report.resolve(),
        paths.calibration_report.resolve(),
    }
    if report_path in protected:
        raise E13ContractError("E13 report must not overwrite an input report")
    raw_cache_dir = paths.raw_cache_dir.resolve()
    clean_cache_dir = (DEFAULT_E12_REPORT.parent / "score_cache").resolve()
    if report_path.is_relative_to(raw_cache_dir) or report_path.is_relative_to(clean_cache_dir):
        raise E13ContractError("E13 report must not be written inside an input cache directory")
    e12_report, _calibration, scenes = load_verified_e12_inputs(paths)
    rr_before = _e12_rows(e12_report, "RR")
    cc_before = _e12_rows(e12_report, "CC")
    clean_records = _clean_cache_records(e12_report)
    contract = _run_contract(paths, e12_report, scenes, clean_records)
    contract_digest = e12.canonical_digest(contract)

    if report_path.is_file():
        existing = _load_json(report_path, label="existing E13 report")
        if (
            existing.get("status") == "complete"
            and existing.get("schema") == REPORT_SCHEMA
            and existing.get("experiment") == EXPERIMENT
            and existing.get("run_contract_sha256") == contract_digest
            and existing.get("protocol") == E13_PROTOCOL
            and isinstance(existing.get("decisions"), Mapping)
        ):
            return existing
        if existing.get("run_contract_sha256") != contract_digest:
            raise E13ContractError("existing E13 report belongs to different input/code bytes")

    output: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "schema": REPORT_SCHEMA,
        "experiment": EXPERIMENT,
        "status": "in_progress",
        "protocol": E13_PROTOCOL,
        "protocol_sha256": e12.canonical_digest(E13_PROTOCOL),
        "run_contract": contract,
        "run_contract_sha256": contract_digest,
        "rows": {arm: [] for arm in ARMS},
        "completed_images": [],
    }
    _atomic_write_json(report_path, output)

    try:
        for scene in scenes:
            clean_cache = _load_cc_cache(scene, e12_report, clean_records[scene.image_id])
            for arm, before_rows in (("RR96", rr_before), ("CC96", cc_before)):
                before = before_rows[scene.image_id]
                board, objective, solver_seconds = _replay_rank96_board(
                    scene,
                    arm=arm,
                    cc_cache=clean_cache if arm == "CC96" else None,
                    before=before,
                )
                selection = torus.select_torus_origin(scene.tiles_uint8, board)
                after = evaluate_rolled_board(scene, selection.rolled_board, objective)
                before_metrics = {key: before[key] for key in BOARD_METRICS}
                before_metrics.update(
                    {
                        "objective": float(before["objective"]),
                        "board_sha256": str(before["board_sha256"]),
                        "solved_corrupted_canvas_sha256": str(
                            before["solved_corrupted_canvas_sha256"]
                        ),
                        "restored_canvas_sha256": str(before["restored_canvas_sha256"]),
                        "source": "exact_E12_record_not_re-restored",
                    }
                )
                output["rows"][arm].append(
                    {
                        "image": int(scene.image_id),
                        "validation_name": str(scene.validation_name),
                        "arm": arm,
                        "role": "deployable_discovery" if arm == "RR96" else "diagnostic_only",
                        "before": before_metrics,
                        "selection": _selection_record(selection),
                        "after": after,
                        "delta": {
                            key: float(after[key]) - float(before[key]) for key in BOARD_METRICS
                        },
                        "solver_replay_seconds": solver_seconds,
                    }
                )
            output["completed_images"].append(int(scene.image_id))
            output["runtime_seconds"] = float(time.perf_counter() - started)
            _atomic_write_json(report_path, output)
            print(
                json.dumps(
                    {
                        "image": scene.image_id,
                        "completed": len(output["completed_images"]),
                        "total": len(e12.CALIBRATION_IDS),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        rr_summary = _paired_before_after(output["rows"]["RR96"], arm="RR96")
        cc_summary = _paired_before_after(output["rows"]["CC96"], arm="CC96")
        output["summaries"] = {"RR96": rr_summary, "CC96": cc_summary}
        output["decisions"] = {
            "RR96": rr_promotion_decision(rr_summary),
            "CC96": cc_origin_diagnosis_decision(cc_summary, rr_summary),
        }
        output["status"] = "complete"
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
        description="Run fixed CPU-only E13 torus/global-origin discovery on E12 IDs 10..17."
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
        E13Paths(
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
                "rr_decision": result["decisions"]["RR96"]["status"],
                "cc_diagnosis": result["decisions"]["CC96"]["status"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

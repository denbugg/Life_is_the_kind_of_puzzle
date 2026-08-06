"""E16 exact clean-render oracle on the byte-pinned RR96 board.

This diagnostic never changes matching, placement, or orientation.  It maps
pristine target tiles back to the original shuffled input IDs, assembles them
with the exact replayed RR96 board, and compares that non-deployable render to
the stored RR96 NLM result.  No candidate restorer is called and no image is
persisted.
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
from typing import Any, Mapping, Sequence

import numpy as np
import skimage
from skimage.metrics import structural_similarity as sk_ssim

import eval_clean_score_oracle as e12
import eval_e14_cc192_discovery as e14
from imgio import assemble


class E16ContractError(RuntimeError):
    """The frozen E16 protocol, input bytes, or runtime drifted."""


SCHEMA_VERSION = 1
REPORT_SCHEMA = "pazzle-e16-rr96-clean-render-oracle-report-v1"
EXPERIMENT = "e16_exact_clean_render_on_fixed_rr96_board_v1"
EXPECTED_E12_REPORT_SHA256 = e14.EXPECTED_E12_REPORT_SHA256
EXPECTED_RUNTIME_PROVENANCE = dict(e14.EXPECTED_RUNTIME_PROVENANCE)

DECISION_RULE: dict[str, float | int] = {
    "mean_final_ssim_delta_min": 0.050,
    "strict_wins_min": 8,
    "worst_final_delta_min": 0.020,
}

E16_PROTOCOL: dict[str, Any] = {
    "schema": "pazzle-e16-fixed-rr96-content-preserving-clean-render-v1",
    "role": "target_derived_non_deployable_post_assembly_restoration_oracle",
    "calibration_ids": list(e12.CALIBRATION_IDS),
    "baseline": {
        "board": "exact_E12_RR96_replay_and_hash",
        "pixels": "stored_E12_RR96_fixed_NLM_h10_metric_no_second_NLM_call",
    },
    "candidate": {
        "board": "exact_same_RR96_board",
        "clean_tiles": "imgio.to_frags(target_uint8)[permutation]",
        "assembly": "upright_clean_tiles_in_original_input_ID_order",
        "restoration": None,
    },
    "geometry": {
        "grid": 24,
        "tile_size": 20,
        "rotation": False,
        "reflection": False,
        "resize": False,
        "blend": False,
        "inpaint": False,
    },
    "decision": dict(DECISION_RULE),
    "conditional_model_gate": {
        "source_disjoint_mean_final_delta_min": 0.005,
        "oracle_gap_capture_ratio_min": 0.20,
    },
    "excluded": [
        "board_change",
        "matching_on_clean_pixels",
        "candidate_solver",
        "candidate_NLM",
        "diffusion_training",
        "sweep",
        "colour_fit",
        "smoothing",
        "GPU",
    ],
    "runtime_provenance": EXPECTED_RUNTIME_PROVENANCE,
}

WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_RAW_CACHE_DIR = e14.DEFAULT_RAW_CACHE_DIR
DEFAULT_CALIBRATION_REPORT = e14.DEFAULT_CALIBRATION_REPORT
DEFAULT_E12_REPORT = e14.DEFAULT_E12_REPORT
DEFAULT_REPORT = Path(
    "E:/pazzle_work/restoration_ceiling_e16/rr96_clean_render_oracle_v1.json"
)


@dataclass(frozen=True)
class E16Paths:
    raw_cache_dir: Path
    calibration_report: Path
    e12_report: Path
    report: Path


def _require_e_drive(path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    if resolved.drive.upper() != "E:":
        raise E16ContractError(f"{label} must stay on E:, got {resolved}")
    return resolved


def _load_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise E16ContractError(f"could not load {label} {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise E16ContractError(f"{label} root is not an object")
    return payload


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    resolved = _require_e_drive(path, label="E16 report")
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
        raise E16ContractError(
            f"E16 runtime drifted: expected {EXPECTED_RUNTIME_PROVENANCE}, got {observed}"
        )
    return observed


def _source_provenance() -> dict[str, str]:
    source = Path(__file__).resolve().parent
    paths = {
        "eval_e16_clean_render_oracle.py": Path(__file__).resolve(),
        "eval_clean_score_oracle.py": source / "eval_clean_score_oracle.py",
        "eval_e14_cc192_discovery.py": source / "eval_e14_cc192_discovery.py",
        "imgio.py": source / "imgio.py",
    }
    return {name: e12.sha256_file(path) for name, path in sorted(paths.items())}


def clean_render(
    target_uint8: np.ndarray,
    permutation: np.ndarray,
    board: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return pristine input-ID tiles and their unchanged-board canvas."""

    strict_board = e12._strict_board(np.asarray(board))
    clean_tiles = e12.clean_tiles_input_order(target_uint8, permutation)
    rendered = np.ascontiguousarray(assemble(clean_tiles, strict_board), dtype=np.uint8)
    if rendered.shape != (e12.IMG, e12.IMG, 3) or rendered.dtype != np.uint8:
        raise E16ContractError("clean render geometry/dtype drifted")
    return clean_tiles, rendered


def evaluate_scene(
    scene: e12.RawScene,
    board: np.ndarray,
    rr_row: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate one target-derived render without invoking a solver or restorer."""

    strict_board = e12._strict_board(np.asarray(board))
    board_sha = e12.array_sha256(strict_board.astype(np.int64, copy=False))
    if board_sha != str(rr_row.get("board_sha256", "")):
        raise E16ContractError(f"RR96 board hash drifted for image {scene.image_id}")
    clean_tiles, rendered = clean_render(
        scene.target_uint8, scene.permutation, strict_board
    )
    clean_ssim = float(
        sk_ssim(
            np.asarray(scene.target_uint8),
            rendered,
            channel_axis=2,
            data_range=255,
        )
    )
    baseline = float(rr_row["final_ssim"])
    return {
        "image": int(scene.image_id),
        "validation_name": str(scene.validation_name),
        "board_sha256": board_sha,
        "rr96_final_ssim": baseline,
        "clean_render_ssim": clean_ssim,
        "clean_render_minus_rr96_final": float(clean_ssim - baseline),
        "clean_tiles_input_order_sha256": e12.array_sha256(clean_tiles),
        "clean_render_canvas_sha256": e12.array_sha256(rendered),
        "candidate_solver_calls": 0,
        "candidate_restorer_calls": 0,
    }


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != len(e12.CALIBRATION_IDS):
        raise E16ContractError("E16 summary requires exactly eight rows")
    images = tuple(sorted(int(row["image"]) for row in rows))
    if images != e12.CALIBRATION_IDS:
        raise E16ContractError("E16 row image IDs drifted")
    baseline = np.asarray(
        [float(row["rr96_final_ssim"]) for row in rows], dtype=np.float64
    )
    candidate = np.asarray(
        [float(row["clean_render_ssim"]) for row in rows], dtype=np.float64
    )
    delta = candidate - baseline
    return {
        "images": len(rows),
        "rr96_final_mean": float(baseline.mean()),
        "clean_render_mean": float(candidate.mean()),
        "mean_final_ssim_delta": float(delta.mean()),
        "median_final_ssim_delta": float(np.median(delta)),
        "best_final_ssim_delta": float(delta.max()),
        "worst_final_ssim_delta": float(delta.min()),
        "strict_wins": int(np.sum(delta > 0.0)),
        "ties": int(np.sum(delta == 0.0)),
        "losses": int(np.sum(delta < 0.0)),
    }


def decision(summary: Mapping[str, Any]) -> dict[str, Any]:
    observed = {
        "mean_final_ssim_delta": float(summary["mean_final_ssim_delta"]),
        "strict_wins": int(summary["strict_wins"]),
        "worst_final_ssim_delta": float(summary["worst_final_ssim_delta"]),
    }
    checks = {
        "mean_final_ssim_delta": observed["mean_final_ssim_delta"]
        >= float(DECISION_RULE["mean_final_ssim_delta_min"]),
        "strict_wins": observed["strict_wins"]
        >= int(DECISION_RULE["strict_wins_min"]),
        "worst_final_ssim_delta": observed["worst_final_ssim_delta"]
        >= float(DECISION_RULE["worst_final_delta_min"]),
    }
    passed = all(checks.values())
    return {
        "status": "go_bounded_restoration_pilot" if passed else "kill_post_assembly_diffusion",
        "passed": passed,
        "thresholds": dict(DECISION_RULE),
        "observed": observed,
        "checks": checks,
        "scope": "content_preserving_restoration_oracle_not_deployable",
    }


def _validate_complete_report(
    report: Mapping[str, Any],
    *,
    contract_digest: str,
    rr_rows: Mapping[int, Mapping[str, Any]],
) -> None:
    if (
        report.get("status") != "complete"
        or report.get("schema") != REPORT_SCHEMA
        or report.get("experiment") != EXPERIMENT
        or report.get("run_contract_sha256") != contract_digest
        or report.get("protocol") != E16_PROTOCOL
    ):
        raise E16ContractError("existing E16 complete report contract drifted")
    rows = report.get("rows")
    if not isinstance(rows, list):
        raise E16ContractError("existing E16 complete report rows are missing")
    completed = report.get("completed_images")
    if completed != list(e12.CALIBRATION_IDS):
        raise E16ContractError("existing E16 completed image list drifted")
    by_image = {
        int(row.get("image", -1)): row
        for row in rows
        if isinstance(row, Mapping)
    }
    if tuple(sorted(by_image)) != e12.CALIBRATION_IDS or len(by_image) != len(rows):
        raise E16ContractError("existing E16 row IDs are incomplete or duplicated")
    for image in e12.CALIBRATION_IDS:
        if by_image[image].get("board_sha256") != rr_rows[image].get("board_sha256"):
            raise E16ContractError(f"existing E16 RR96 board drifted for image {image}")
    computed_summary = summarize(rows)
    computed_decision = decision(computed_summary)
    if report.get("summary") != computed_summary:
        raise E16ContractError("existing E16 summary does not match its rows")
    if report.get("decision") != computed_decision:
        raise E16ContractError("existing E16 decision does not match its summary")
    if report.get("stage") != computed_decision["status"]:
        raise E16ContractError("existing E16 terminal stage drifted")


def run_oracle(paths: E16Paths) -> Mapping[str, Any]:
    started = time.perf_counter()
    report_path = _require_e_drive(paths.report, label="E16 report")
    if report_path.suffix.lower() != ".json":
        raise E16ContractError("E16 report must be a .json file")
    raw_cache_dir = _require_e_drive(paths.raw_cache_dir, label="raw score cache")
    e12_report_path = _require_e_drive(paths.e12_report, label="E12 report")
    calibration_path = paths.calibration_report.resolve()
    if report_path in {e12_report_path, calibration_path}:
        raise E16ContractError("E16 report must not overwrite an input")
    clean_cache_dir = (DEFAULT_E12_REPORT.parent / "score_cache").resolve()
    if report_path.is_relative_to(raw_cache_dir) or report_path.is_relative_to(
        clean_cache_dir
    ):
        raise E16ContractError("E16 report must not be written inside an input cache")

    e12_report, _calibration, scenes = e14.load_verified_e12_inputs(
        e14.E14Paths(
            raw_cache_dir=raw_cache_dir,
            calibration_report=calibration_path,
            e12_report=e12_report_path,
            report=e14.DEFAULT_REPORT,
        )
    )
    rr_rows = e14._e12_rr_rows(e12_report)
    rr_verification = e14.verify_rr_means(rr_rows)
    contract = {
        "protocol_sha256": e12.canonical_digest(E16_PROTOCOL),
        "report": str(report_path),
        "e12_report": {
            "path": str(e12_report_path),
            "sha256": EXPECTED_E12_REPORT_SHA256,
        },
        "calibration_report": {
            "path": str(calibration_path),
            "sha256": e12.CALIBRATION_REPORT_SHA256,
        },
        "raw_cache_dir": str(raw_cache_dir),
        "scene_provenance_digest": str(e12_report["scene_provenance_digest"]),
        "source_provenance": _source_provenance(),
        "runtime_provenance": _runtime_provenance(),
    }
    contract_digest = e12.canonical_digest(contract)
    if report_path.is_file():
        existing = _load_json(report_path, label="existing E16 report")
        if (
            existing.get("status") == "complete"
            and existing.get("schema") == REPORT_SCHEMA
            and existing.get("experiment") == EXPERIMENT
            and existing.get("run_contract_sha256") == contract_digest
            and existing.get("protocol") == E16_PROTOCOL
        ):
            _validate_complete_report(
                existing,
                contract_digest=contract_digest,
                rr_rows=rr_rows,
            )
            return existing
        if existing.get("run_contract_sha256") != contract_digest:
            raise E16ContractError("existing E16 report belongs to different bytes")

    output: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "schema": REPORT_SCHEMA,
        "experiment": EXPERIMENT,
        "status": "in_progress",
        "stage": "clean_render",
        "protocol": E16_PROTOCOL,
        "protocol_sha256": e12.canonical_digest(E16_PROTOCOL),
        "run_contract": contract,
        "run_contract_sha256": contract_digest,
        "rr_reproducibility": rr_verification,
        "rows": [],
        "completed_images": [],
    }
    _atomic_write_json(report_path, output)
    try:
        for scene in scenes:
            before = rr_rows[scene.image_id]
            board, _objective, replay_seconds = e14._replay_rr96(scene, before)
            row = evaluate_scene(scene, board, before)
            row["rr96_replay_seconds"] = float(replay_seconds)
            output["rows"].append(row)
            output["completed_images"].append(int(scene.image_id))
            output["runtime_seconds"] = float(time.perf_counter() - started)
            _atomic_write_json(report_path, output)

        summary = summarize(output["rows"])
        result = decision(summary)
        output["summary"] = summary
        output["decision"] = result
        output["status"] = "complete"
        output["stage"] = result["status"]
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
        description="Run fixed RR96 exact clean-render restoration oracle."
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
    result = run_oracle(
        E16Paths(
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
                "decision": result["decision"]["status"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

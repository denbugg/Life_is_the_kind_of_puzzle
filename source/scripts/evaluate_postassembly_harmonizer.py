#!/usr/bin/env python3
"""Leakage-safe development evaluation of analytic post-assembly harmonizers."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
from PIL import Image
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from puzzle_assembly.panels import make_exact_panel  # noqa: E402
from puzzle_assembly.postassembly_harmonizer import (  # noqa: E402
    LuminanceGainConfig,
    SeamGraphConfig,
    apply_luminance_gains,
    apply_rgb_offsets,
    bilateral_tile_offsets,
    blend_tiles_uint8,
    image_quality_metrics,
    naive_local_mean_offsets,
    ordered_from_slots,
    paired_bootstrap_ci,
    seam_graph_luminance_gains,
    seam_graph_rgb_offsets,
)
from puzzle_assembly.protocol import per_source_seed, source_names_for_split  # noqa: E402
from puzzle_denoise_v2.inference import load_restorer, restore_tiles_uint8  # noqa: E402
from puzzle_denoise_v2.tiles import merge_tiles_numpy  # noqa: E402


BASELINE = "fixed_alpha_0_5_no_target_tuning"
RGB_CANDIDATE = "seam_graph_rgb_on_blend"
GAIN_CANDIDATE = "seam_graph_rgb_plus_bounded_luminance_gain"
PANELS = ("primary_kornia", "independent_libjpeg")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/postassembly_rgb_offset_v1.json"
    )
    parser.add_argument(
        "--gain-config",
        default=None,
        help="Optional separate luminance-gain protocol; omitted for the RGB-only run.",
    )
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--lln-source-limit", type=int, default=1)
    parser.add_argument("--skip-lln", action="store_true")
    parser.add_argument("--preview-sources", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        default="runs/assembly_v1/postassembly_harmonizer/rgb_offset_v1",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hash(path: Path, expected: str, role: str) -> str:
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"{role} hash mismatch: expected {expected}, got {actual} for {path}"
        )
    return actual


def _names_sha256(names: list[str]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if array.shape != (480, 480, 3):
        raise ValueError(f"expected 480x480 RGB target, got {array.shape}: {path}")
    return array


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validate_protocol(
    config_path: Path,
    gain_config_path: Path | None,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, str]]:
    config = _load_json(config_path)
    if config.get("kind") != "postassembly_rgb_offset_development_protocol":
        raise RuntimeError("unexpected RGB-offset protocol kind")
    if config.get("status") != "precommitted_before_selected_slice_pixel_access":
        raise RuntimeError("RGB-offset protocol is not in its precommitted state")
    if config["scope"].get("development_only") is not True:
        raise RuntimeError("protocol must remain development-only")
    if config["scope"].get("target_pixels_available_to_harmonizer") is not False:
        raise RuntimeError("target-blinding contract was relaxed")

    input_hashes: dict[str, str] = {"config": _sha256(config_path)}
    for role in ("manifest", "quarantine", "audit_exclusion"):
        record = config["authoritative_inputs"][role]
        path = REPO_ROOT / record["path"]
        input_hashes[role] = _require_hash(path, record["sha256"], role)
    for role in ("selected_tilenaf", "production_seam_tilenaf"):
        record = config["authoritative_inputs"][role]
        path = REPO_ROOT / record["path"]
        input_hashes[role] = _require_hash(path, record["sha256"], role)

    manifest_path = REPO_ROOT / config["authoritative_inputs"]["manifest"]["path"]
    quarantine_path = REPO_ROOT / config["authoritative_inputs"]["quarantine"]["path"]
    audit_exclusion_path = REPO_ROOT / config["authoritative_inputs"]["audit_exclusion"]["path"]
    selected = [str(name) for name in config["source_selection"]["names"]]
    if len(selected) != 32 or len(set(selected)) != 32:
        raise RuntimeError("frozen source list must contain 32 unique names")
    if _names_sha256(selected) != config["source_selection"]["names_sha256"]:
        raise RuntimeError("frozen source-name hash mismatch")
    edge_development = source_names_for_split(
        "edge_development",
        manifest_path=manifest_path,
        quarantine_path=quarantine_path,
        audit_exclusion_path=audit_exclusion_path,
    )
    if any(name not in edge_development[160:] for name in selected):
        raise RuntimeError("frozen selection is not contained in edge_development[160:]")
    indices = [edge_development.index(name) for name in selected]
    if indices[0] != 160 or indices[-1] != 199 or indices != sorted(indices):
        raise RuntimeError("frozen selection indices differ from the precommit ledger")
    if set(selected) & set(edge_development[128:160]):
        raise RuntimeError("selection overlaps candidate graph oracle sources")

    disjoint_splits = (
        "assembly_cal",
        "assembly_incremental_gate",
        "assembly_audit_exposed",
        "assembly_final_audit",
    )
    for split in disjoint_splits:
        other = source_names_for_split(
            split,
            manifest_path=manifest_path,
            quarantine_path=quarantine_path,
            audit_exclusion_path=audit_exclusion_path,
        )
        if set(selected) & set(other):
            raise RuntimeError(f"frozen source list overlaps sealed split {split}")
    manifest = _load_json(manifest_path)
    test_names = {path.name for path in (REPO_ROOT / "puzzle/test").glob("*.png")}
    if len(test_names) != 700:
        raise RuntimeError(f"expected 700 test basenames, got {len(test_names)}")
    if set(selected) & test_names:
        raise RuntimeError("frozen source list overlaps test")
    if set(selected) & set(map(str, manifest["excluded_test_overlap"])):
        raise RuntimeError("frozen source list overlaps manifest test-overlap exclusion")

    if tuple(config["synthetic_panels"]["names"]) != PANELS:
        raise RuntimeError("panel order changed")
    if config["renderer_baselines"][BASELINE].get("alpha_was_not_tuned_on_any_target") is not True:
        raise RuntimeError("fixed blend is not explicitly untuned")
    if config["renderer_baselines"][BASELINE]["production_seam_tilenaf_weight"] != 0.5:
        raise RuntimeError("fixed blend alpha changed")
    if config["luminance_gain"].get("enabled") is not False:
        raise RuntimeError("luminance gain may not be folded into the RGB protocol")

    gain_config: dict[str, Any] | None = None
    if gain_config_path is not None:
        gain_config = _load_json(gain_config_path)
        if gain_config.get("kind") != "postassembly_luminance_gain_separate_development_candidate":
            raise RuntimeError("unexpected gain protocol kind")
        base = gain_config["base_protocol"]
        if Path(base["path"]) != config_path.relative_to(REPO_ROOT):
            raise RuntimeError("gain protocol references a different base protocol")
        if input_hashes["config"] != base["sha256"]:
            raise RuntimeError("gain protocol base hash mismatch")
        if gain_config["scope"].get("target_pixels_available_to_gain_estimator") is not False:
            raise RuntimeError("gain estimator target-blinding contract was relaxed")
        input_hashes["gain_config"] = _sha256(gain_config_path)
    return config, gain_config, input_hashes


def _seam_config(config: dict[str, Any]) -> SeamGraphConfig:
    values = config["methods"][RGB_CANDIDATE]
    return SeamGraphConfig(
        extrapolation_band=int(values["extrapolation_band"]),
        confidence_scale=float(values["confidence_scale"]),
        confidence_floor=float(values["confidence_floor"]),
        ridge=float(values["ridge"]),
        huber_delta=float(values["huber_delta"]),
        irls_steps=int(values["irls_steps"]),
        max_abs_offset=float(values["max_abs_offset"]),
    )


def _gain_config(config: dict[str, Any]) -> LuminanceGainConfig:
    values = config["method"]
    return LuminanceGainConfig(
        extrapolation_band=int(values["extrapolation_band"]),
        confidence_scale=float(values["confidence_scale"]),
        confidence_floor=float(values["confidence_floor"]),
        ridge=float(values["ridge"]),
        huber_delta=float(values["huber_delta"]),
        irls_steps=int(values["irls_steps"]),
        max_fractional_gain=float(values["max_fractional_gain"]),
        luminance_floor=float(values["luminance_floor"]),
        luminance_ceiling=float(values["luminance_ceiling"]),
    )


def _summary_by_panel(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for panel in PANELS:
        panel_records = [record for record in records if record["panel"] == panel]
        methods = sorted(panel_records[0]["metrics"])
        result[panel] = {}
        for method in methods:
            result[panel][method] = {
                metric: float(np.mean([record["metrics"][method][metric] for record in panel_records]))
                for metric in (
                    "ssim",
                    "boundary_band_mae",
                    "target_referenced_seam_error",
                    "untargeted_seam_discontinuity",
                    "mae",
                )
            }
    return result


def _comparison(
    records: list[dict[str, Any]],
    *,
    candidate: str,
    baseline: str,
    bootstrap_seed: int,
    resamples: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for panel_index, panel in enumerate(PANELS):
        panel_records = [record for record in records if record["panel"] == panel]
        ssim_delta = np.asarray(
            [
                record["metrics"][candidate]["ssim"]
                - record["metrics"][baseline]["ssim"]
                for record in panel_records
            ],
            dtype=np.float64,
        )
        seam_delta = np.asarray(
            [
                record["metrics"][candidate]["target_referenced_seam_error"]
                - record["metrics"][baseline]["target_referenced_seam_error"]
                for record in panel_records
            ],
            dtype=np.float64,
        )
        low, high = paired_bootstrap_ci(
            ssim_delta,
            seed=bootstrap_seed + panel_index,
            resamples=resamples,
        )
        result[panel] = {
            "source_count": len(panel_records),
            "mean_ssim_delta": float(ssim_delta.mean()),
            "paired_bootstrap_95_ci": [low, high],
            "wins_ties_losses": [
                int(np.sum(ssim_delta > 1e-12)),
                int(np.sum(np.abs(ssim_delta) <= 1e-12)),
                int(np.sum(ssim_delta < -1e-12)),
            ],
            "mean_target_referenced_seam_error_delta": float(seam_delta.mean()),
            "fraction_ssim_regression_gt_0_01": float(np.mean(ssim_delta < -0.01)),
        }
    return result


def _gate(
    comparison: dict[str, Any],
    gate_config: dict[str, Any],
    *,
    full_run: bool,
) -> dict[str, Any]:
    per_panel: dict[str, Any] = {}
    for panel in PANELS:
        values = comparison[panel]
        checks = {
            "mean_ssim_delta_at_least_0_005": values["mean_ssim_delta"]
            >= float(gate_config["per_panel_mean_ssim_delta_minimum"]),
            "bootstrap_lower_above_zero": values["paired_bootstrap_95_ci"][0]
            > float(gate_config["per_panel_paired_bootstrap_lower_must_exceed"]),
            "target_referenced_seam_error_nonregression": values[
                "mean_target_referenced_seam_error_delta"
            ]
            <= float(gate_config["per_panel_mean_target_referenced_seam_error_delta_maximum"]),
        }
        per_panel[panel] = {
            "checks": checks,
            "passed": bool(all(checks.values())),
        }
    return {
        "evaluable": bool(full_run),
        "per_panel": per_panel,
        "passed": bool(full_run and all(values["passed"] for values in per_panel.values())),
        "submission_promotion_allowed": False,
    }


def _save_preview(
    output_dir: Path,
    *,
    source: str,
    panel: str,
    target_tiles: np.ndarray,
    arms: dict[str, np.ndarray],
) -> None:
    preview_dir = output_dir / "previews" / source.removesuffix(".png") / panel
    preview_dir.mkdir(parents=True, exist_ok=False)
    Image.fromarray(merge_tiles_numpy(target_tiles), mode="RGB").save(
        preview_dir / "target.png"
    )
    for name in (
        "raw_corrupted",
        "selected_tilenaf",
        "production_seam_tilenaf",
        BASELINE,
        "naive_5x5_on_blend",
        "bilateral_offset_on_blend",
        RGB_CANDIDATE,
        "shuffled_neighbor_placebo_on_blend",
        GAIN_CANDIDATE,
    ):
        if name in arms:
            Image.fromarray(merge_tiles_numpy(arms[name]), mode="RGB").save(
                preview_dir / f"{name}.png"
            )


def _lln_diagnostic(
    *,
    clean_target: np.ndarray,
    source: str,
    panel: str,
    initial_ordered_raw: np.ndarray,
    target_tiles: np.ndarray,
    master_seed: int,
    k_values: list[int],
) -> dict[str, Any]:
    maximum = max(k_values)
    running = initial_ordered_raw.astype(np.float64)
    metrics: dict[str, Any] = {
        "1": image_quality_metrics(initial_ordered_raw, target_tiles)
    }
    for replica in range(1, maximum):
        seed = per_source_seed(
            master_seed,
            f"postassembly-rgb-offset-{panel}",
            source,
            replica,
        )
        exact = make_exact_panel(clean_target, panel=panel, seed=seed)
        running += ordered_from_slots(exact.slot_tiles, exact.slot_to_target)
        count = replica + 1
        if count in k_values:
            averaged = np.clip(np.rint(running / float(count)), 0, 255).astype(np.uint8)
            metrics[str(count)] = image_quality_metrics(averaged, target_tiles)
    return {
        "source": source,
        "panel": panel,
        "deployable": False,
        "multiple_test_observations_required": True,
        "metrics_by_k": metrics,
    }


def _markdown_summary(report: dict[str, Any]) -> str:
    lines = [
        "# Post-assembly analytic harmonizer",
        "",
        f"Status: `{report['status']}`. This is development-only evidence and cannot promote a submission.",
        "",
        f"Sources: {report['source_count']} whole images x {len(PANELS)} corruption panels.",
        "",
        "## Primary candidate vs fixed 0.5 main/seam blend",
        "",
        "| Panel | Baseline SSIM | Candidate SSIM | Delta | 95% paired CI | Seam-error delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for panel in PANELS:
        summary = report["panel_macro"][panel]
        comp = report["primary_comparison"][panel]
        ci = comp["paired_bootstrap_95_ci"]
        lines.append(
            f"| {panel} | {summary[BASELINE]['ssim']:.6f} | "
            f"{summary[RGB_CANDIDATE]['ssim']:.6f} | {comp['mean_ssim_delta']:+.6f} | "
            f"[{ci[0]:+.6f}, {ci[1]:+.6f}] | "
            f"{comp['mean_target_referenced_seam_error_delta']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "The shuffled-neighbour arm is a topology placebo. The K=1/2/4/8/25 section is an LLN ceiling that uses repeated observations unavailable at test time.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    if not 2 <= args.limit <= 32:
        raise SystemExit("--limit must be between 2 and 32")
    if args.batch_size <= 0 or args.torch_threads <= 0:
        raise SystemExit("--batch-size and --torch-threads must be positive")
    if args.lln_source_limit < 0 or args.lln_source_limit > args.limit:
        raise SystemExit("--lln-source-limit must be between 0 and --limit")
    if args.preview_sources < 0 or args.preview_sources > args.limit:
        raise SystemExit("--preview-sources must be between 0 and --limit")

    config_path = (REPO_ROOT / args.config).resolve()
    gain_config_path = (
        (REPO_ROOT / args.gain_config).resolve() if args.gain_config else None
    )
    config, gain_protocol, input_hashes = _validate_protocol(
        config_path, gain_config_path
    )

    output_dir = (REPO_ROOT / args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    output_dir.mkdir(parents=True)
    _write_json(
        output_dir / "PIXEL_ACCESS_STARTED.json",
        {
            "kind": "postassembly_harmonizer_pixel_access_marker",
            "created_utc": _utc_now(),
            "config_sha256": input_hashes["config"],
            "gain_config_sha256": input_hashes.get("gain_config"),
            "source_count": args.limit,
        },
    )

    selected_names = list(config["source_selection"]["names"][: args.limit])
    torch.set_num_threads(args.torch_threads)
    selected_record = config["authoritative_inputs"]["selected_tilenaf"]
    seam_record = config["authoritative_inputs"]["production_seam_tilenaf"]
    selected_model, device, selected_metadata = load_restorer(
        REPO_ROOT / selected_record["path"], device=args.device, state=selected_record["state"]
    )
    seam_model, seam_device, seam_metadata = load_restorer(
        REPO_ROOT / seam_record["path"], device=str(device), state=seam_record["state"]
    )
    if seam_device != device:
        raise RuntimeError("restorer devices differ")

    rgb_config = _seam_config(config)
    luma_config = _gain_config(gain_protocol) if gain_protocol is not None else None
    naive = config["methods"]["naive_5x5_on_blend"]
    bilateral = config["methods"]["bilateral_offset_on_blend"]
    master_seed = int(config["synthetic_panels"]["master_seed"])
    records: list[dict[str, Any]] = []
    lln_records: list[dict[str, Any]] = []
    started = time.perf_counter()

    for source_index, source in enumerate(selected_names):
        target_path = REPO_ROOT / "puzzle/train/targets" / source
        clean_target = _read_rgb(target_path)
        target_sha256 = _sha256(target_path)
        for panel_index, panel in enumerate(PANELS):
            record_started = time.perf_counter()
            seed = per_source_seed(
                master_seed,
                f"postassembly-rgb-offset-{panel}",
                source,
                0,
            )
            exact = make_exact_panel(clean_target, panel=panel, seed=seed)
            raw = ordered_from_slots(exact.slot_tiles, exact.slot_to_target)
            target_tiles = exact.clean_target_tiles

            selected_tiles = restore_tiles_uint8(
                selected_model, raw, device, batch_size=args.batch_size
            )
            seam_tiles = restore_tiles_uint8(
                seam_model, raw, seam_device, batch_size=args.batch_size
            )
            blend = blend_tiles_uint8(selected_tiles, seam_tiles, auxiliary_weight=0.5)

            naive_offsets = naive_local_mean_offsets(
                blend,
                radius=int(naive["radius_tiles"]),
                strength=float(naive["strength"]),
                max_abs_offset=float(naive["max_abs_rgb_offset"]),
            )
            bilateral_offsets = bilateral_tile_offsets(
                blend,
                radius=int(bilateral["radius_tiles"]),
                sigma_spatial=float(bilateral["sigma_spatial"]),
                sigma_colour=float(bilateral["sigma_colour"]),
                strength=float(bilateral["strength"]),
                max_abs_offset=float(bilateral["max_abs_rgb_offset"]),
            )
            rgb_offsets, rgb_diagnostics = seam_graph_rgb_offsets(blend, rgb_config)
            placebo_seed = per_source_seed(
                master_seed,
                f"postassembly-placebo-{panel}",
                source,
                0,
            )
            placebo_offsets, placebo_diagnostics = seam_graph_rgb_offsets(
                blend, rgb_config, placebo_seed=placebo_seed
            )

            arms = {
                "raw_corrupted": raw,
                "selected_tilenaf": selected_tiles,
                "production_seam_tilenaf": seam_tiles,
                BASELINE: blend,
                "naive_5x5_on_blend": apply_rgb_offsets(blend, naive_offsets),
                "bilateral_offset_on_blend": apply_rgb_offsets(blend, bilateral_offsets),
                RGB_CANDIDATE: apply_rgb_offsets(blend, rgb_offsets),
                "shuffled_neighbor_placebo_on_blend": apply_rgb_offsets(
                    blend, placebo_offsets
                ),
            }
            gain_diagnostics = None
            if luma_config is not None:
                gains, gain_diagnostics = seam_graph_luminance_gains(
                    arms[RGB_CANDIDATE], luma_config
                )
                arms[GAIN_CANDIDATE] = apply_luminance_gains(
                    arms[RGB_CANDIDATE], gains
                )

            # Scoring begins only after every target-blind arm is frozen.
            metrics = {
                name: image_quality_metrics(values, target_tiles)
                for name, values in arms.items()
            }
            records.append(
                {
                    "source": source,
                    "source_index_in_frozen_selection": source_index,
                    "panel": panel,
                    "panel_index": panel_index,
                    "panel_seed": seed,
                    "target_sha256": target_sha256,
                    "metrics": metrics,
                    "diagnostics": {
                        RGB_CANDIDATE: rgb_diagnostics,
                        "shuffled_neighbor_placebo_on_blend": placebo_diagnostics,
                        GAIN_CANDIDATE: gain_diagnostics,
                    },
                    "seconds": float(time.perf_counter() - record_started),
                }
            )
            if source_index < args.preview_sources:
                _save_preview(
                    output_dir,
                    source=source,
                    panel=panel,
                    target_tiles=target_tiles,
                    arms=arms,
                )
            if (
                not args.skip_lln
                and source_index < args.lln_source_limit
            ):
                lln_records.append(
                    _lln_diagnostic(
                        clean_target=clean_target,
                        source=source,
                        panel=panel,
                        initial_ordered_raw=raw,
                        target_tiles=target_tiles,
                        master_seed=master_seed,
                        k_values=[int(value) for value in config["lln_diagnostic"]["k_values"]],
                    )
                )
            print(
                json.dumps(
                    {
                        "completed": len(records),
                        "total": len(selected_names) * len(PANELS),
                        "source": source,
                        "panel": panel,
                        "baseline_ssim": metrics[BASELINE]["ssim"],
                        "candidate_ssim": metrics[RGB_CANDIDATE]["ssim"],
                        "delta": metrics[RGB_CANDIDATE]["ssim"] - metrics[BASELINE]["ssim"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    bootstrap = config["metrics"]
    panel_macro = _summary_by_panel(records)
    primary_comparison = _comparison(
        records,
        candidate=RGB_CANDIDATE,
        baseline=BASELINE,
        bootstrap_seed=int(bootstrap["paired_bootstrap_seed"]),
        resamples=int(bootstrap["paired_bootstrap_resamples"]),
    )
    full_run = args.limit == 32 and selected_names == config["source_selection"]["names"]
    primary_gate = _gate(
        primary_comparison, config["frozen_gate"], full_run=full_run
    )
    gain_comparison = None
    gain_gate = None
    if gain_protocol is not None:
        gain_comparison = _comparison(
            records,
            candidate=GAIN_CANDIDATE,
            baseline=RGB_CANDIDATE,
            bootstrap_seed=int(bootstrap["paired_bootstrap_seed"]) + 100,
            resamples=int(bootstrap["paired_bootstrap_resamples"]),
        )
        gain_gate = _gate(
            gain_comparison, gain_protocol["frozen_gate"], full_run=full_run
        )

    if not full_run:
        status = "partial_smoke_no_gate"
    elif primary_gate["passed"]:
        status = "development_gate_passed_requires_new_confirmation"
    else:
        status = "development_gate_failed_stop_or_redesign"

    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "postassembly_harmonizer_development_report",
        "created_utc": _utc_now(),
        "status": status,
        "development_only": True,
        "submission_promotion_allowed": False,
        "source_count": len(selected_names),
        "panel_count": len(PANELS),
        "record_count": len(records),
        "source_names": selected_names,
        "source_names_sha256": _names_sha256(selected_names),
        "full_frozen_selection_used": full_run,
        "input_hashes": input_hashes,
        "code_hashes": {
            "evaluator": _sha256(Path(__file__).resolve()),
            "harmonizer": _sha256(REPO_ROOT / "src/puzzle_assembly/postassembly_harmonizer.py"),
            "panels": _sha256(REPO_ROOT / "src/puzzle_assembly/panels.py"),
            "degradation": _sha256(REPO_ROOT / "src/puzzle_denoise_v2/degradation.py"),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": str(device),
            "torch_threads": torch.get_num_threads(),
            "pid": os.getpid(),
            "seconds": float(time.perf_counter() - started),
        },
        "model_metadata": {
            "selected_tilenaf": selected_metadata,
            "production_seam_tilenaf": seam_metadata,
        },
        "target_blinding": {
            "all_arms_constructed_before_scoring": True,
            "harmonizer_module_accepts_only_ordered_uint8_tiles_and_frozen_parameters": True,
            "source_names_or_target_arrays_passed_to_harmonizer": False,
            "targets_used_only_for_exact_panel_generation_and_separate_metrics": True,
        },
        "panel_macro": panel_macro,
        "primary_comparison": primary_comparison,
        "primary_gate": primary_gate,
        "gain_comparison": gain_comparison,
        "gain_gate": gain_gate,
        "lln_diagnostic": {
            "deployable": False,
            "requested_source_limit": 0 if args.skip_lln else args.lln_source_limit,
            "complete_for_full_frozen_selection": bool(
                not args.skip_lln and args.lln_source_limit == 32
            ),
            "records": lln_records,
        },
        "records": records,
    }
    report_path = output_dir / "report.json"
    _write_json(report_path, report)
    (output_dir / "SUMMARY.md").write_text(
        _markdown_summary(report), encoding="utf-8"
    )
    result = {
        "status": status,
        "report": str(report_path.relative_to(REPO_ROOT)),
        "report_sha256": _sha256(report_path),
        "source_count": len(selected_names),
        "primary_comparison": primary_comparison,
        "primary_gate": primary_gate,
    }
    _write_json(output_dir / "RESULT.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

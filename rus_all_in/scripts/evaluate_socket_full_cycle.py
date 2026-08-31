#!/usr/bin/env python3
"""Measure one legal Socket sorting/restoration cycle on an existing exact panel.

The source roster, corruption seed, and draw count come from an already opened
Socket exact-synthetic report. Neither clean pixels nor the exact inverse
shuffle are passed to the predictor. Geometry/SSIM metrics are computed only
after a label-free prediction artifact has been atomically written and read
back. No parameter is selected by this runner and no competition input is
accessible through its CLI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from PIL import Image

from aiijc_puzzle.layout_evaluation import evaluate_layout
from aiijc_puzzle.protocol import (
    IMAGE_SIZE,
    assemble_tiles,
    compute_protocol_digest,
    contest_ssim,
    sha256_file,
    split_tiles,
)
from aiijc_puzzle.socket_sorter_production import (
    DECODER_EDGE_BUDGET,
    DECODER_SWAP_STEPS,
    HISTORICAL_RGB_LUMA_NLM_H20_TAIL,
    IDENTITY_PIXEL_TAIL,
    choose_deterministic_device,
    load_socket_checkpoint,
    predict_socket_sorter,
)
from aiijc_puzzle.synthetic_socket_evaluation import make_exact_synthetic_case, names_digest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data/interim/validation_manifest.json"
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
FROZEN_SCHEMA = "aiijc-socket-full-cycle-frozen-v2"
REPORT_EXPERIMENT = "socket-sorter-legal-full-cycle-v2"
IMPLEMENTATION_PATHS = (
    "scripts/evaluate_socket_full_cycle.py",
    "src/aiijc_puzzle/socket_sorter_production.py",
    "src/aiijc_puzzle/socket_pixel_tails.py",
    "src/aiijc_puzzle/socket_matcher.py",
    "src/aiijc_puzzle/socket_decoder.py",
    "src/aiijc_puzzle/socket_translation_placer.py",
    "src/aiijc_puzzle/synthetic_socket_evaluation.py",
    "src/aiijc_puzzle/layout_evaluation.py",
    "src/aiijc_puzzle/postassembly_harmonizer.py",
    "src/aiijc_puzzle/pixel_tails.py",
    "src/aiijc_puzzle/protocol.py",
    "configs/postassembly_rgb_offset_v1.json",
    "configs/postassembly_luminance_gain_v1.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    return parser.parse_args()


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        if (
            image.format != "PNG"
            or image.mode != "RGB"
            or image.size != (IMAGE_SIZE, IMAGE_SIZE)
        ):
            raise ValueError(f"expected RGB {IMAGE_SIZE}x{IMAGE_SIZE}: {path}")
        return np.asarray(image, dtype=np.uint8).copy()


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(encoded).hexdigest()


def _implementation_manifest() -> dict[str, Any]:
    files = {relative: sha256_file(PROJECT_ROOT / relative) for relative in IMPLEMENTATION_PATHS}
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "files": files,
        "files_digest": hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _selection(
    report: dict[str, Any],
) -> tuple[list[str], int, int, dict[str, str]]:
    selection = report.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("source report has no selection object")
    names = selection.get("source_filenames")
    seed = selection.get("seed")
    draws = selection.get("draws_per_source")
    if (
        not isinstance(names, list)
        or not names
        or any(not isinstance(name, str) or not name for name in names)
        or len(names) != len(set(names))
    ):
        raise ValueError("source report has invalid source_filenames")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("source report has invalid seed")
    if isinstance(draws, bool) or not isinstance(draws, int) or draws <= 0:
        raise ValueError("source report has invalid draws_per_source")
    if selection.get("source_limit") != len(names):
        raise ValueError("source report source_limit differs from source_filenames")
    if selection.get("case_count") != len(names) * draws:
        raise ValueError("source report case_count differs from source_count * draws")
    if selection.get("source_digest") != names_digest(names):
        raise ValueError("source report source_digest does not match source_filenames")
    sources = selection.get("sources")
    if not isinstance(sources, list) or len(sources) != len(names):
        raise ValueError("source report has no complete target-hash source roster")
    source_hashes: dict[str, str] = {}
    for expected_name, source in zip(names, sources, strict=True):
        if not isinstance(source, dict) or source.get("filename") != expected_name:
            raise ValueError("source report target-hash roster order differs from source_filenames")
        target_hash = source.get("target_sha256")
        if not _valid_sha256(target_hash):
            raise ValueError(f"source report has invalid target hash for {expected_name}")
        source_hashes[expected_name] = target_hash
    return names, seed, draws, source_hashes


def _validate_source_report_binding(
    report: dict[str, Any],
    *,
    checkpoint: Any,
    checkpoint_sha256: str,
    manifest_digest: str,
    selected_names: list[str],
) -> dict[str, Any]:
    report_checkpoint = report.get("checkpoint")
    if not isinstance(report_checkpoint, dict):
        raise ValueError("source report has no checkpoint object")
    if report_checkpoint.get("sha256") != checkpoint_sha256:
        raise ValueError("checkpoint differs from the matched source report")
    if report_checkpoint.get("architecture_contract") != checkpoint.contract:
        raise ValueError("source report architecture contract differs from checkpoint")
    lineage_names = report_checkpoint.get("lineage_filenames")
    if (
        not isinstance(lineage_names, list)
        or lineage_names != sorted(lineage_names)
        or len(lineage_names) != len(set(lineage_names))
        or any(not isinstance(name, str) or not name for name in lineage_names)
    ):
        raise ValueError("source report checkpoint lineage roster is malformed")
    if tuple(lineage_names) != checkpoint.lineage.exposed_filenames:
        raise ValueError("source report lineage names differ from actual checkpoint exposure")
    if report_checkpoint.get("lineage_digest") != checkpoint.lineage.exposed_digest:
        raise ValueError("source report lineage digest differs from actual checkpoint exposure")
    overlap = sorted(set(selected_names) & set(checkpoint.lineage.exposed_filenames))
    if overlap:
        raise ValueError(f"source roster overlaps actual checkpoint lineage: {overlap}")

    protocol = report.get("protocol")
    if not isinstance(protocol, dict) or protocol.get("manifest_digest") != manifest_digest:
        raise ValueError("source report manifest digest differs from the supplied manifest")
    required_protocol = {
        "manifest_split": "train",
        "checkpoint_lineage_source_disjoint": True,
        "target_hashes_verified_before_use": True,
        "dirty_only_predictions_frozen_before_reference_scoring": True,
        "frozen_artifact_contains_exact_references": False,
    }
    if any(protocol.get(key) != value for key, value in required_protocol.items()):
        raise ValueError("source report does not satisfy the exact-panel protocol contract")

    fixed = report.get("fixed_candidates")
    global_variants = fixed.get("global") if isinstance(fixed, dict) else None
    if (
        not isinstance(fixed, dict)
        or fixed.get("decoder_edge_budget_per_axis") != DECODER_EDGE_BUDGET
        or fixed.get("decoder_swap_steps") != DECODER_SWAP_STEPS
        or not isinstance(global_variants, list)
        or "socket_ot_decoder" not in global_variants
    ):
        raise ValueError("source report decoder contract differs from decoder144")

    frozen = report.get("frozen_predictions")
    if not isinstance(frozen, dict):
        raise ValueError("source report has no frozen-prediction artifact")
    verified_artifacts: dict[str, str] = {}
    for kind in ("arrays", "metadata"):
        raw_path = frozen.get(f"{kind}_path")
        expected_hash = frozen.get(f"{kind}_sha256")
        if not isinstance(raw_path, str) or not _valid_sha256(expected_hash):
            raise ValueError(f"source report frozen {kind} binding is malformed")
        path = Path(raw_path)
        if sha256_file(path) != expected_hash:
            raise ValueError(f"source report frozen {kind} hash mismatch")
        verified_artifacts[f"{kind}_sha256"] = expected_hash
    return {
        "checkpoint_lineage_source_disjoint_recomputed": True,
        "checkpoint_exposed_count": checkpoint.lineage.exposed_count,
        "checkpoint_exposed_digest": checkpoint.lineage.exposed_digest,
        "source_report_frozen_artifacts_verified": True,
        **verified_artifacts,
    }


def _aggregate(rows: list[dict[str, Any]], variant: str) -> dict[str, float | int]:
    metrics = [row[variant] for row in rows]
    return {
        "boards": len(metrics),
        "correct_tile_count_total": sum(int(item["correct_tile_count"]) for item in metrics),
        "correct_tile_count_mean": float(
            np.mean([float(item["correct_tile_count"]) for item in metrics])
        ),
        "direct_placement": float(np.mean([float(item["direct_placement"]) for item in metrics])),
        "row_accuracy": float(np.mean([float(item["row_accuracy"]) for item in metrics])),
        "column_accuracy": float(
            np.mean([float(item["column_accuracy"]) for item in metrics])
        ),
        "translation_aligned_placement": float(
            np.mean([float(item["translation_aligned_placement"]) for item in metrics])
        ),
        "adjacency": float(np.mean([float(item["adjacency"]) for item in metrics])),
        "raw_ssim": float(np.mean([float(item["raw_ssim"]) for item in metrics])),
        "final_ssim": float(np.mean([float(item["final_ssim"]) for item in metrics])),
        "matcher_seconds_mean": float(
            np.mean([float(item["matcher_seconds"]) for item in metrics])
        ),
        "inference_seconds_mean": float(
            np.mean([float(item["inference_seconds"]) for item in metrics])
        ),
        "tail_seconds_mean": float(np.mean([float(item["tail_seconds"]) for item in metrics])),
        "full_cycle_seconds_mean": float(
            np.mean([float(item["full_cycle_seconds"]) for item in metrics])
        ),
    }


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    source_report_path = args.source_report.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory must be absent or empty: {output_dir}")

    source_report = _load_json(source_report_path)
    names, seed, draws, source_target_hashes = _selection(source_report)
    observed_checkpoint_hash = sha256_file(checkpoint_path)

    manifest = _load_json(args.manifest.resolve())
    manifest_digest = compute_protocol_digest(manifest)
    if manifest.get("protocol_digest") != manifest_digest:
        raise ValueError("manifest self-digest is invalid")
    splits = manifest.get("splits")
    train = splits.get("train") if isinstance(splits, dict) else None
    if not isinstance(train, list):
        raise ValueError("manifest has no train split")
    record_rows = [
        record
        for record in train
        if isinstance(record, dict) and isinstance(record.get("filename"), str)
    ]
    records = {
        str(record.get("filename")): record
        for record in record_rows
    }
    if len(records) != len(record_rows):
        raise ValueError("manifest train split has duplicate filenames")
    missing = [name for name in names if name not in records]
    if missing:
        raise ValueError(f"source report contains names outside manifest train: {missing}")

    device = choose_deterministic_device(args.device)
    checkpoint = load_socket_checkpoint(checkpoint_path, device=device)
    source_binding = _validate_source_report_binding(
        source_report,
        checkpoint=checkpoint,
        checkpoint_sha256=observed_checkpoint_hash,
        manifest_digest=manifest_digest,
        selected_names=names,
    )

    frozen: list[dict[str, Any]] = []
    scoring_inputs: list[tuple[np.ndarray, Any, np.ndarray]] = []
    targets_dir = args.targets.resolve()
    for source_index, name in enumerate(names, start=1):
        record = records[name]
        target_path = targets_dir / name
        observed_target_hash = sha256_file(target_path)
        if (
            observed_target_hash != record.get("target_sha256")
            or observed_target_hash != source_target_hashes[name]
        ):
            raise ValueError(f"manifest target hash mismatch: {name}")
        clean = _load_rgb(target_path)
        clean_tiles = split_tiles(clean)
        for draw_index in range(draws):
            synthetic, reference = make_exact_synthetic_case(
                clean_tiles,
                source_filename=name,
                draw_index=draw_index,
                seed=seed,
            )
            dirty_canvas = assemble_tiles(synthetic.tiles)
            baseline = predict_socket_sorter(
                dirty_canvas,
                checkpoint,
                device=device,
                cyclic_border5=False,
                pixel_tail=IDENTITY_PIXEL_TAIL,
            )
            anchored = predict_socket_sorter(
                dirty_canvas,
                checkpoint,
                device=device,
                cyclic_border5=True,
                pixel_tail=IDENTITY_PIXEL_TAIL,
            )
            tail_started = perf_counter()
            final = HISTORICAL_RGB_LUMA_NLM_H20_TAIL.apply(anchored.raw)
            tail_seconds = perf_counter() - tail_started
            frozen.append(
                {
                    "case_id": synthetic.case_id,
                    "source_filename": name,
                    "draw_index": draw_index,
                    "dirty_canvas_sha256": _array_sha256(dirty_canvas),
                    "baseline_layout": baseline.layout.tolist(),
                    "baseline_raw_sha256": _array_sha256(baseline.raw),
                    "baseline_matcher_seconds": baseline.matcher_seconds,
                    "baseline_inference_seconds": baseline.total_seconds,
                    "anchored_layout": anchored.layout.tolist(),
                    "anchored_raw_sha256": _array_sha256(anchored.raw),
                    "anchored_final_sha256": _array_sha256(final),
                    "anchored_matcher_seconds": anchored.matcher_seconds,
                    "anchored_inference_seconds": anchored.total_seconds,
                    "tail_seconds": tail_seconds,
                    "baseline_permutation_audit": baseline.audit.as_dict(),
                    "anchored_permutation_audit": anchored.audit.as_dict(),
                }
            )
            scoring_inputs.append((clean, reference, dirty_canvas))
        print(f"froze {source_index}/{len(names)} {name}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    frozen_path = output_dir / "frozen_predictions.json"
    source_report_sha256 = sha256_file(source_report_path)
    implementation = _implementation_manifest()
    frozen_payload = {
        "schema": FROZEN_SCHEMA,
        "contains_exact_references": False,
        "contains_clean_pixels": False,
        "predictor_input": "dirty RGB canvas only",
        "binding": {
            "matched_source_report_sha256": source_report_sha256,
            "manifest_digest": manifest_digest,
            "checkpoint_sha256": observed_checkpoint_hash,
            "checkpoint_lineage": checkpoint.lineage.as_dict(),
            "source_digest": names_digest(names),
            "source_count": len(names),
            "draws_per_source": draws,
            "case_count": len(frozen),
            "seed": seed,
            "device": str(device),
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "source_report_validation": source_binding,
            "implementation": implementation,
        },
        "tail": HISTORICAL_RGB_LUMA_NLM_H20_TAIL.evidence,
        "boards": frozen,
    }
    # Normalize tuples from PermutationAudit to their exact persisted JSON form
    # before comparing the atomic readback.
    frozen_payload = json.loads(json.dumps(frozen_payload, ensure_ascii=False))
    frozen_sha256 = _atomic_write_json(frozen_path, frozen_payload)
    committed_frozen = _load_json(frozen_path)
    if committed_frozen != frozen_payload or sha256_file(frozen_path) != frozen_sha256:
        raise RuntimeError("frozen prediction artifact failed atomic readback")
    committed_rows = committed_frozen.get("boards")
    if not isinstance(committed_rows, list) or len(committed_rows) != len(scoring_inputs):
        raise RuntimeError("frozen prediction artifact board roster is malformed")
    print(f"frozen artifact committed: {frozen_path}", flush=True)

    boards: list[dict[str, Any]] = []
    for frozen_row, (clean, reference, dirty_canvas) in zip(
        committed_rows, scoring_inputs, strict=True
    ):
        if (
            frozen_row.get("case_id") != reference.case_id
            or frozen_row.get("dirty_canvas_sha256") != _array_sha256(dirty_canvas)
        ):
            raise RuntimeError("committed frozen case binding differs from scoring input")
        dirty_tiles = split_tiles(dirty_canvas)
        baseline_layout = np.asarray(frozen_row["baseline_layout"], dtype=np.int32)
        anchored_layout = np.asarray(frozen_row["anchored_layout"], dtype=np.int32)
        baseline_raw = assemble_tiles(dirty_tiles[baseline_layout])
        anchored_raw = assemble_tiles(dirty_tiles[anchored_layout])
        if (
            _array_sha256(baseline_raw) != frozen_row["baseline_raw_sha256"]
            or _array_sha256(anchored_raw) != frozen_row["anchored_raw_sha256"]
        ):
            raise RuntimeError("committed raw-assembly hash differs during scoring")
        final = HISTORICAL_RGB_LUMA_NLM_H20_TAIL.apply(anchored_raw)
        if _array_sha256(final) != frozen_row["anchored_final_sha256"]:
            raise RuntimeError("committed final-tail hash differs during scoring")
        baseline_metrics = evaluate_layout(
            baseline_layout,
            reference.tile_at_position,
            reference_is_exact=True,
        ).as_dict()
        anchored_metrics = evaluate_layout(
            anchored_layout,
            reference.tile_at_position,
            reference_is_exact=True,
        ).as_dict()
        baseline_metrics.update(
            {
                "raw_ssim": contest_ssim(clean, baseline_raw),
                "final_ssim": contest_ssim(clean, baseline_raw),
                "matcher_seconds": frozen_row["baseline_matcher_seconds"],
                "inference_seconds": frozen_row["baseline_inference_seconds"],
                "tail_seconds": 0.0,
                "full_cycle_seconds": frozen_row["baseline_inference_seconds"],
            }
        )
        anchored_metrics.update(
            {
                "raw_ssim": contest_ssim(clean, anchored_raw),
                "final_ssim": contest_ssim(clean, final),
                "matcher_seconds": frozen_row["anchored_matcher_seconds"],
                "inference_seconds": frozen_row["anchored_inference_seconds"],
                "tail_seconds": frozen_row["tail_seconds"],
                "full_cycle_seconds": (
                    frozen_row["anchored_inference_seconds"] + frozen_row["tail_seconds"]
                ),
            }
        )
        boards.append(
            {
                "case_id": frozen_row["case_id"],
                "source_filename": frozen_row["source_filename"],
                "draw_index": frozen_row["draw_index"],
                "decoder144_identity": baseline_metrics,
                "decoder144_cyclic5_h20": anchored_metrics,
            }
        )

    source_evaluation = source_report.get("evaluation")
    if not isinstance(source_evaluation, dict):
        raise ValueError("source report has no evaluation object")
    local = source_evaluation.get("local_aggregate", {}).get("socket_ot")
    source_global = source_evaluation.get("global_aggregate", {}).get("socket_ot_decoder")
    if not isinstance(local, dict) or not isinstance(source_global, dict):
        raise ValueError("source report lacks matched Socket OT metrics")
    aggregate = {
        "decoder144_identity": _aggregate(boards, "decoder144_identity"),
        "decoder144_cyclic5_h20": _aggregate(boards, "decoder144_cyclic5_h20"),
    }
    baseline_mapping = {
        "correct_tile_count_total": "correct_tile_count_total",
        "correct_tile_count_mean": "correct_tile_count",
        "direct_placement": "direct_placement",
        "row_accuracy": "row_accuracy",
        "column_accuracy": "column_accuracy",
        "translation_aligned_placement": "translation_aligned_placement",
        "adjacency": "adjacency",
    }
    for observed_key, source_key in baseline_mapping.items():
        observed = aggregate["decoder144_identity"][observed_key]
        expected = source_global.get(source_key)
        if not isinstance(expected, int | float) or not np.isclose(
            float(observed), float(expected), rtol=0.0, atol=1e-15
        ):
            raise ValueError(
                f"recomputed decoder144 {observed_key} differs from source report: "
                f"{observed} != {expected}"
            )
    report = {
        "experiment": REPORT_EXPERIMENT,
        "status": "matched-existing-exact-panel-no-selection",
        "protocol": {
            "manifest_digest": manifest_digest,
            "matched_source_report": str(source_report_path),
            "matched_source_report_sha256": source_report_sha256,
            "checkpoint_sha256": observed_checkpoint_hash,
            "source_report_binding": source_binding,
            "clean_pixels_or_exact_reference_supplied_to_predictor": False,
            "geometry_and_ssim_metrics_computed_only_after_atomic_freeze_readback": True,
            "decoder144_geometry_matches_bound_source_report": True,
            "competition_test_opened": False,
            "strict_original_upright_tile_permutation_before_tail": True,
            "tail_is_target_blind_post_layout_only": True,
            "parameter_selection_performed_in_this_runner": False,
            "panel_is_reused_and_not_a_fresh_confirmation": True,
            "implementation": implementation,
            "runtime_definition": {
                "matcher_seconds": "one SocketMatcher forward including device-to-CPU sync",
                "inference_seconds": (
                    "matcher + decoder + optional cyclic anchor + strict raw audit + identity hook"
                ),
                "tail_seconds": "pinned RGB offsets + bounded luminance + one colored NLM h20",
                "full_cycle_seconds": "inference_seconds + tail_seconds",
            },
        },
        "selection": {
            "source_filenames": names,
            "source_digest": names_digest(names),
            "sources": [
                {"filename": name, "target_sha256": source_target_hashes[name]}
                for name in names
            ],
            "source_count": len(names),
            "draws_per_source": draws,
            "case_count": len(boards),
            "seed": seed,
        },
        "local_matched_from_source_report": {
            "pooled_r1": local.get("pooled_r1"),
            "pooled_r5": local.get("pooled_r5"),
        },
        "aggregate": aggregate,
        "boards": boards,
        "frozen_predictions": {
            "path": str(frozen_path),
            "schema": FROZEN_SCHEMA,
            "sha256": frozen_sha256,
        },
    }
    report_path = output_dir / "report.json"
    _atomic_write_json(report_path, report)
    print(json.dumps({"report": str(report_path), **report["aggregate"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

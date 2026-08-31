#!/usr/bin/env python3
"""Replay one frozen DINOv2 boundary-candidate screen on opened local16."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from aiijc_puzzle.dinov2_boundary_matcher import (
    BAND_WIDTH,
    IMAGE_SIZE,
    MODEL_NAME,
    PATCH_GRID,
    TOP_K,
    freeze_topk,
    load_official_dinov2,
    score_dirty_tiles,
)
from aiijc_puzzle.protocol import compute_protocol_digest, sha256_file
from aiijc_puzzle.synthetic_socket_evaluation import (
    ExactSyntheticReference,
    exact_local_retrieval_metrics,
    make_exact_synthetic_case,
    names_digest,
)

try:
    from scripts import run_fullres_boundary_denoiser as parent
    from scripts import run_fullres_retrieval_adapter as adapter
except ModuleNotFoundError:
    import run_fullres_boundary_denoiser as parent
    import run_fullres_retrieval_adapter as adapter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/dinov2_boundary_candidate_screen_v1.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/dinov2-boundary-candidate-screen/opened-local16-v1"
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
COUNT = 24 * 24
KS = (1, 5, 32)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--smoke-one", action="store_true")
    return parser.parse_args(argv)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def _write_npz(path: Path, values: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **values)


def _record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {
        "path": str(resolved.relative_to(PROJECT_ROOT)),
        "sha256": sha256_file(resolved),
    }


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    resolved = path.resolve()
    sidecar = resolved.with_suffix(resolved.suffix + ".sha256")
    if not resolved.is_file() or not sidecar.is_file():
        raise FileNotFoundError("signed DINOv2 screen config is missing")
    digest = sha256_file(resolved)
    if sidecar.read_text(encoding="utf-8").split()[0] != digest:
        raise ValueError("DINOv2 screen preregistration digest mismatch")
    config = json.loads(resolved.read_text(encoding="utf-8"))
    expected = {
        "model": "official DINOv2 ViT-S/14 through timm dynamic_img_size",
        "image_size": IMAGE_SIZE,
        "patch_grid": PATCH_GRID,
        "band_width": BAND_WIDTH,
        "top_k": TOP_K,
        "direct_ks": list(KS),
        "no_sweep": True,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"DINOv2 screen contract mismatch: {key}")
    for item in config["frozen_inputs"].values():
        artifact = PROJECT_ROOT / item["path"]
        if not artifact.is_file() or sha256_file(artifact) != item["sha256"]:
            raise ValueError(f"frozen DINOv2 screen input changed: {artifact}")
    return config, digest


def _device(name: str) -> torch.device:
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    return torch.device(name)


def _reciprocal(scores: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(scores, dtype=np.float64).copy()
    np.fill_diagonal(values, -np.inf)
    target = np.argmax(values, axis=1).astype(np.int32)
    incoming = np.argmax(values, axis=0).astype(np.int32)
    source = np.arange(len(values), dtype=np.int32)
    reciprocal = incoming[target] == source
    partitioned = np.partition(values, kth=len(values) - 2, axis=1)
    second = partitioned[:, -2]
    confidence = values[source, target] - second
    return {
        "target": target,
        "reciprocal": reciprocal,
        "confidence": np.ascontiguousarray(confidence, dtype=np.float32),
    }


def _totals() -> dict[str, int]:
    return {
        f"{scope}_{name}": 0
        for scope in ("right", "down", "pooled")
        for name in ("total", "hits_at_1", "hits_at_5", "hits_at_32")
    }


def _add_metrics(total: dict[str, int], current: Mapping[str, Any]) -> None:
    for key in total:
        total[key] += int(current[key])


def _finish_metrics(total: dict[str, int]) -> dict[str, Any]:
    result: dict[str, Any] = dict(total)
    for scope in ("right", "down", "pooled"):
        denominator = result[f"{scope}_total"]
        for k in KS:
            result[f"{scope}_r{k}"] = result[f"{scope}_hits_at_{k}"] / denominator
    return result


def _score(
    *,
    archive: Path,
    metadata: Path,
    raw_archive: Path,
    raw_metadata: Path,
    references: Mapping[str, ExactSyntheticReference],
) -> dict[str, Any]:
    rows = json.loads(metadata.read_text(encoding="utf-8"))["rows"]
    raw_rows = json.loads(raw_metadata.read_text(encoding="utf-8"))["rows"]
    if len(rows) != len(raw_rows):
        raise RuntimeError("DINO and raw panel lengths differ")
    totals = {"raw_d64_ot": _totals(), "dinov2_boundary": _totals()}
    union = {
        axis: {"total": 0, "raw_hits": 0, "dino_hits": 0, "union_hits": 0}
        for axis in ("right", "down")
    }
    reciprocal_stats = {
        variant: {"admitted": 0, "correct": 0}
        for variant in ("raw_d64_ot", "dinov2_boundary")
    }
    with np.load(archive, allow_pickle=False) as frozen, np.load(
        raw_archive, allow_pickle=False
    ) as raw:
        for row, raw_row in zip(rows, raw_rows, strict=True):
            for field in ("prefix", "case_id", "source_filename", "draw_index", "dirty_sha256"):
                if row[field] != raw_row[field]:
                    raise RuntimeError(f"DINO/raw row mismatch: {field}")
            prefix = row["prefix"]
            reference = references[row["case_id"]].tile_at_position
            candidates = {
                "raw_d64_ot": {
                    axis: raw[f"{prefix}__candidate__raw_d64_ot__{axis}"]
                    for axis in ("right", "down")
                },
                "dinov2_boundary": {
                    axis: frozen[f"{prefix}__candidate__dinov2_boundary__{axis}"]
                    for axis in ("right", "down")
                },
            }
            for variant, axes in candidates.items():
                _add_metrics(
                    totals[variant],
                    exact_local_retrieval_metrics(
                        axes["right"], axes["down"], reference, ks=KS
                    ),
                )
            for axis in ("right", "down"):
                truth = parent._truth_by_anchor(reference, axis=axis)
                valid = truth >= 0
                anchors = np.flatnonzero(valid)
                raw_hit = np.any(
                    candidates["raw_d64_ot"][axis][anchors] == truth[anchors, None], axis=1
                )
                dino_hit = np.any(
                    candidates["dinov2_boundary"][axis][anchors] == truth[anchors, None], axis=1
                )
                union[axis]["total"] += len(anchors)
                union[axis]["raw_hits"] += int(raw_hit.sum())
                union[axis]["dino_hits"] += int(dino_hit.sum())
                union[axis]["union_hits"] += int(np.count_nonzero(raw_hit | dino_hit))
                for variant in reciprocal_stats:
                    evidence = {
                        key: (
                            raw[f"{prefix}__reciprocal__raw_d64_ot__{axis}__{key}"]
                            if variant == "raw_d64_ot"
                            else frozen[f"{prefix}__reciprocal__dinov2_boundary__{axis}__{key}"]
                        )
                        for key in ("target", "reciprocal")
                    }
                    admitted = valid & evidence["reciprocal"]
                    reciprocal_stats[variant]["admitted"] += int(admitted.sum())
                    reciprocal_stats[variant]["correct"] += int(
                        np.count_nonzero(evidence["target"][admitted] == truth[admitted])
                    )
    retrieval = {name: _finish_metrics(value) for name, value in totals.items()}
    valid_total = sum(value["total"] for value in union.values())
    union_total = {
        "pooled_total": valid_total,
        "pooled_raw_hits": sum(value["raw_hits"] for value in union.values()),
        "pooled_dino_hits": sum(value["dino_hits"] for value in union.values()),
        "pooled_union_hits": sum(value["union_hits"] for value in union.values()),
    }
    union_total.update(
        {
            "pooled_raw_coverage": union_total["pooled_raw_hits"] / valid_total,
            "pooled_dino_coverage": union_total["pooled_dino_hits"] / valid_total,
            "pooled_union_coverage": union_total["pooled_union_hits"] / valid_total,
            "pooled_coverage_gain": (
                union_total["pooled_union_hits"] - union_total["pooled_raw_hits"]
            )
            / valid_total,
        }
    )
    reciprocal = {
        name: {
            **value,
            "coverage": value["admitted"] / valid_total,
            "precision": value["correct"] / value["admitted"] if value["admitted"] else 0.0,
        }
        for name, value in reciprocal_stats.items()
    }
    return {
        "retrieval": retrieval,
        "union_top32": {"axes": union, **union_total},
        "reciprocal": reciprocal,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    config, config_sha = _load_config(args.config)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    device = _device(args.device)
    manifest_path = PROJECT_ROOT / config["frozen_inputs"]["manifest"]["path"]
    parent_report_path = PROJECT_ROOT / config["frozen_inputs"]["roster_parent_report"]["path"]
    raw_archive = PROJECT_ROOT / config["frozen_inputs"]["raw_candidate_archive"]["path"]
    raw_metadata = PROJECT_ROOT / config["frozen_inputs"]["raw_candidate_metadata"]["path"]
    checkpoint = PROJECT_ROOT / config["frozen_inputs"]["checkpoint"]["path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_digest") != compute_protocol_digest(manifest):
        raise RuntimeError("manifest protocol digest mismatch")
    parent_report = json.loads(parent_report_path.read_text(encoding="utf-8"))
    names = tuple(parent_report["selection"]["eval_filenames"])
    if names_digest(names) != "25ea956a8514d72cb09b8093f12999534995cf75fb18b383834acf38693ca47f":
        raise RuntimeError("local16 source roster changed")
    if args.smoke_one:
        names = names[:1]
    record_by_name = {
        str(record["filename"]): record for record in manifest["splits"]["train"]
    }
    records = tuple(record_by_name[name] for name in names)
    boards = parent._prepare_boards(records, args.targets.resolve())
    raw_rows = json.loads(raw_metadata.read_text(encoding="utf-8"))["rows"][: len(boards)]
    model = load_official_dinov2(checkpoint, device=device)
    arrays: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    references: dict[str, ExactSyntheticReference] = {}
    started = perf_counter()
    for index, (board, raw_row) in enumerate(zip(boards, raw_rows, strict=True)):
        item, reference = make_exact_synthetic_case(
            board.tiles,
            source_filename=board.filename,
            draw_index=0,
            seed=adapter.EVAL_SEED,
        )
        digest = hashlib.sha256(item.tiles.tobytes()).hexdigest()
        if any(
            raw_row[field] != value
            for field, value in {
                "case_id": item.case_id,
                "source_filename": item.source_filename,
                "draw_index": item.draw_index,
                "dirty_sha256": digest,
            }.items()
        ):
            raise RuntimeError("recreated dirty board does not match frozen raw panel")
        case_started = perf_counter()
        scores = score_dirty_tiles(
            model, item.tiles, device=device, batch_size=args.batch_size
        )
        elapsed = perf_counter() - case_started
        prefix = str(raw_row["prefix"])
        for axis, matrix in (("right", scores.right), ("down", scores.down)):
            arrays[f"{prefix}__candidate__dinov2_boundary__{axis}"] = freeze_topk(matrix)
            for key, value in _reciprocal(matrix).items():
                arrays[f"{prefix}__reciprocal__dinov2_boundary__{axis}__{key}"] = value
        rows.append(
            {
                "prefix": prefix,
                "case_id": item.case_id,
                "source_filename": item.source_filename,
                "draw_index": item.draw_index,
                "dirty_sha256": digest,
                "runtime_seconds": elapsed,
            }
        )
        references[item.case_id] = reference
        print(
            json.dumps(
                {
                    "event": "dinov2_boundary_freeze",
                    "case": index + 1,
                    "count": len(boards),
                    "runtime_seconds": elapsed,
                }
            ),
            flush=True,
        )
    archive = output / "frozen-target-free-candidates.npz"
    metadata = output / "frozen-target-free-candidates.json"
    freeze = output / "pre-score-freeze.json"
    _write_npz(archive, arrays)
    _write_json(
        metadata,
        {
            "schema": "aiijc-dinov2-boundary-candidates-v1",
            "contains_exact_references_or_clean_pixels": False,
            "matcher_only": True,
            "model": MODEL_NAME,
            "rows": rows,
        },
    )
    _write_json(
        freeze,
        {
            "schema": "aiijc-dinov2-boundary-pre-score-freeze-v1",
            "created_before_exact_reference_scoring": True,
            "contains_exact_references_or_labels": False,
            "artifacts": {
                "archive": _record(archive),
                "metadata": _record(metadata),
                "config": _record(args.config),
                "checkpoint": _record(checkpoint),
                "module": _record(PROJECT_ROOT / "src/aiijc_puzzle/dinov2_boundary_matcher.py"),
                "runner": _record(Path(__file__)),
            },
        },
    )
    if args.smoke_one:
        report = {
            "schema": "aiijc-dinov2-boundary-screen-smoke-v1",
            "status": "target-free-smoke",
            "competition_test_accessed": False,
            "artifacts": {
                "archive": _record(archive),
                "metadata": _record(metadata),
                "freeze": _record(freeze),
            },
        }
        _write_json(output / "report.json", report)
        return report
    scored = _score(
        archive=archive,
        metadata=metadata,
        raw_archive=raw_archive,
        raw_metadata=raw_metadata,
        references=references,
    )
    raw_r5 = scored["retrieval"]["raw_d64_ot"]["pooled_r5"]
    dino_r5 = scored["retrieval"]["dinov2_boundary"]["pooled_r5"]
    gain = scored["union_top32"]["pooled_coverage_gain"]
    reciprocal_precision = scored["reciprocal"]["dinov2_boundary"]["precision"]
    gate_spec = config["discovery_gate"]
    passed = gain >= gate_spec["pooled_raw_union_top32_coverage_gain_min"] and (
        dino_r5 - raw_r5 >= gate_spec["either_direct_pooled_r5_gain_vs_raw_min"]
        or reciprocal_precision >= gate_spec["or_reciprocal_precision_min"]
    )
    report = {
        "schema": "aiijc-dinov2-boundary-candidate-screen-report-v1",
        "status": "discovery-gate-passed" if passed else "discovery-gate-failed-stop",
        "preregistration_sha256": config_sha,
        "protocol": config,
        "local16": scored,
        "gate": {
            "passed": passed,
            "pooled_union_top32_gain": gain,
            "direct_pooled_r5_gain": dino_r5 - raw_r5,
            "dinov2_reciprocal_precision": reciprocal_precision,
        },
        "runtime_seconds": perf_counter() - started,
        "legality": {
            "strict_original_upright_tile_ids_only": True,
            "pixels_modified_for_output": False,
            "absolute_or_semantic_position_prior": False,
            "competition_test_accessed": False,
            "production_or_submission_modified": False,
            "target_free_candidates_frozen_before_scoring": True,
        },
        "artifacts": {
            "archive": _record(archive),
            "metadata": _record(metadata),
            "freeze": _record(freeze),
        },
    }
    _write_json(output / "report.json", report)
    print(
        json.dumps(
            {"status": report["status"], "local16": scored, "gate": report["gate"]},
            indent=2,
        )
    )
    return report


if __name__ == "__main__":
    run(parse_args())

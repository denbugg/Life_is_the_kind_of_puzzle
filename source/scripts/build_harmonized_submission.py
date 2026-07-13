#!/usr/bin/env python3
"""Render a deterministic submission from frozen QAP layouts plus RGB harmonization.

The layout is never recomputed here.  Two already-promoted tile restorers are
run on each test input, their outputs are blended with the untuned fixed 0.5
weight, and the confirmed target-blind seam-graph RGB correction is applied
after indexing by the frozen production QAP layout.
"""

from __future__ import annotations

import argparse
from io import BytesIO
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Iterable, Mapping
import zipfile

import numpy as np
from PIL import Image

from puzzle_assembly.postassembly_harmonizer import (
    SeamGraphConfig,
    apply_rgb_offsets,
    blend_tiles_uint8,
    seam_graph_rgb_offsets,
)
from puzzle_denoise_v2.inference import load_restorer, restore_tiles_uint8
from puzzle_denoise_v2.tiles import merge_tiles_numpy, split_tiles_numpy


GRID = 24
TILE_COUNT = GRID * GRID
ARCHIVE_TIMESTAMP = (2026, 7, 12, 0, 0, 0)
SELECTED_DENOISER_SHA256 = (
    "77a2d8607c9bc6b80e7aa99ed03329c1fae6bf94ee1d9a654241ab93a05cc734"
)
SEAM_DENOISER_SHA256 = (
    "f973c7e606a112020c527bb72277b82586df915edc829a22305e587b35aec1b9"
)
EXPECTED_LAYOUT_REPORT_SHA256 = {
    "541e7905dad9373a173c31db068429b20fb614d450f3cfb4439b89a0a45b2e2a",
    "38e8a01de560ef8914f25dbe42b2c43c063bde07e3b537e71c0be28416f8dbc4",
}
EXPECTED_LAYOUT_CONFIGURATION = {
    "soft_cycle": {
        "score": "l1",
        "top_k": 8,
        "keep_per_tile": 1,
        "keep_fraction": 0.5,
        "loop_weight": 1.0,
        "reciprocal_weight": 0.35,
    },
    "qap": {
        "enabled": True,
        "score": "l1w4",
        "iterations": 25,
        "restarts": 2,
        "initial_weight": 0.75,
        "noisy_components": 3,
        "noise_scale": 1.0,
        "boundary_weight": 0.05,
        "refine_swaps": 8,
        "refine_weak_cells": 32,
        "seed_formula": "filename_sha256_first4_le + 7001",
    },
}
HARMONIZER_CONFIG = SeamGraphConfig(
    extrapolation_band=3,
    confidence_scale=12.0,
    confidence_floor=0.05,
    ridge=0.20,
    huber_delta=4.0,
    irls_steps=4,
    max_abs_offset=12.0,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--selected-denoiser", type=Path, required=True)
    parser.add_argument("--seam-denoiser", type=Path, required=True)
    parser.add_argument(
        "--layout-report", type=Path, action="append", required=True,
        help="supply both frozen 350-image QAP shard reports",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--expected-count", type=int, default=700)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        values = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if values.shape != (480, 480, 3):
        raise RuntimeError(f"unexpected input image shape: {path} {values.shape}")
    return np.ascontiguousarray(values)


def _png_bytes(values: np.ndarray) -> bytes:
    if values.shape != (480, 480, 3) or values.dtype != np.uint8:
        raise RuntimeError("output image must be uint8 RGB 480x480")
    buffer = BytesIO()
    Image.fromarray(values, mode="RGB").save(
        buffer, format="PNG", compress_level=6
    )
    return buffer.getvalue()


def _layout_sha256(layout: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(layout, dtype=np.int32).tobytes()).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_layout(layout_value: Any) -> np.ndarray:
    layout = np.asarray(layout_value)
    if layout.shape != (TILE_COUNT,) or layout.dtype.kind not in "iu":
        raise RuntimeError("frozen layout must be an integer vector of length 576")
    layout = layout.astype(np.int32, copy=False)
    if not np.array_equal(np.sort(layout), np.arange(TILE_COUNT, dtype=np.int32)):
        raise RuntimeError("frozen layout is not a permutation")
    return np.ascontiguousarray(layout)


def _require_layout_contract(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != 2:
        raise RuntimeError("layout report schema drift")
    if report.get("kind") != "assembly_v1_submission_report":
        raise RuntimeError("unexpected layout report kind")
    pipeline = report.get("pipeline")
    if not isinstance(pipeline, Mapping) or pipeline.get("mode") != "promoted_directional_qap":
        raise RuntimeError("layout report is not the promoted QAP pipeline")
    if pipeline.get("score_alias") != "l1w4":
        raise RuntimeError("layout report score alias drift")
    configuration = report.get("configuration")
    if not isinstance(configuration, Mapping):
        raise RuntimeError("layout report lacks configuration")
    for key, expected in EXPECTED_LAYOUT_CONFIGURATION.items():
        if configuration.get(key) != expected:
            raise RuntimeError(f"layout report {key} configuration drift")
    if report.get("denoiser_checkpoint_sha256") != SELECTED_DENOISER_SHA256:
        raise RuntimeError("layout scoring denoiser drift")
    anti_leakage = report.get("anti_leakage")
    if not isinstance(anti_leakage, Mapping) or anti_leakage.get("target_paths_or_pixels_read") is not False:
        raise RuntimeError("layout report lacks anti-leakage attestation")


def load_frozen_layouts(
    report_paths: Iterable[Path], *, expected_count: int
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    paths = [path.expanduser().resolve(strict=True) for path in report_paths]
    if len(paths) != 2 or len(set(paths)) != 2:
        raise RuntimeError("exactly two distinct layout shard reports are required")
    report_hashes = {sha256_file(path) for path in paths}
    if report_hashes != EXPECTED_LAYOUT_REPORT_SHA256:
        raise RuntimeError("layout report byte hashes do not match promoted shards")
    layouts: dict[str, dict[str, Any]] = {}
    report_records: list[dict[str, Any]] = []
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            raise RuntimeError("layout report must be a JSON object")
        _require_layout_contract(report)
        sources = report.get("sources")
        names = report.get("source_names")
        if not isinstance(sources, list) or not isinstance(names, list):
            raise RuntimeError("layout report source schema drift")
        if len(sources) != report.get("count") or names != [record.get("source") for record in sources]:
            raise RuntimeError("layout report source order drift")
        for record in sources:
            if not isinstance(record, dict):
                raise RuntimeError("invalid layout source record")
            name = record.get("source")
            if not isinstance(name, str) or Path(name).name != name or not name.endswith(".png"):
                raise RuntimeError("invalid layout source name")
            if name in layouts:
                raise RuntimeError(f"duplicate layout source: {name}")
            layout = _validate_layout(record.get("position_to_slot"))
            if record.get("layout_sha256") != _layout_sha256(layout):
                raise RuntimeError("layout hash mismatch")
            input_hash = record.get("input_pixel_sha256")
            if not _valid_sha256(input_hash):
                raise RuntimeError("input pixel hash is missing from layout record")
            layouts[name] = {
                "layout": layout,
                "layout_sha256": record["layout_sha256"],
                "input_pixel_sha256": input_hash,
            }
        report_records.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "count": len(sources),
            }
        )
    if len(layouts) != expected_count:
        raise RuntimeError(
            f"expected {expected_count} frozen layouts, found {len(layouts)}"
        )
    return layouts, sorted(report_records, key=lambda value: value["sha256"])


def render_harmonized_tiles(
    selected_slot_tiles: np.ndarray,
    seam_slot_tiles: np.ndarray,
    layout: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    permutation = _validate_layout(layout)
    selected_ordered = np.ascontiguousarray(selected_slot_tiles[permutation])
    seam_ordered = np.ascontiguousarray(seam_slot_tiles[permutation])
    blended = blend_tiles_uint8(
        selected_ordered, seam_ordered, auxiliary_weight=0.5
    )
    offsets, diagnostics = seam_graph_rgb_offsets(
        blended, HARMONIZER_CONFIG
    )
    corrected = apply_rgb_offsets(blended, offsets)
    restored = merge_tiles_numpy(corrected)
    return restored, {
        **diagnostics,
        "offset_sha256": hashlib.sha256(
            np.asarray(offsets, dtype=np.float32).tobytes()
        ).hexdigest(),
    }


def build_submission(args: argparse.Namespace) -> dict[str, Any]:
    if args.offset < 0 or args.expected_count <= 0:
        raise RuntimeError("invalid offset or expected count")
    if args.limit is not None and args.limit <= 0:
        raise RuntimeError("limit must be positive")
    if args.batch_size <= 0:
        raise RuntimeError("batch size must be positive")
    input_dir = args.input_dir.expanduser().resolve(strict=True)
    selected_path = args.selected_denoiser.expanduser().resolve(strict=True)
    seam_path = args.seam_denoiser.expanduser().resolve(strict=True)
    if sha256_file(selected_path) != SELECTED_DENOISER_SHA256:
        raise RuntimeError("selected TileNAF checkpoint hash mismatch")
    if sha256_file(seam_path) != SEAM_DENOISER_SHA256:
        raise RuntimeError("production seam TileNAF checkpoint hash mismatch")
    layouts, layout_reports = load_frozen_layouts(
        args.layout_report, expected_count=args.expected_count
    )
    all_paths = sorted(input_dir.glob("*.png"), key=lambda path: path.name)
    if len(all_paths) != args.expected_count:
        raise RuntimeError(
            f"expected {args.expected_count} input PNGs, found {len(all_paths)}"
        )
    all_names = [path.name for path in all_paths]
    if set(all_names) != set(layouts):
        raise RuntimeError("test input names and frozen layout names differ")
    stop = args.expected_count if args.limit is None else args.offset + args.limit
    if args.offset >= args.expected_count or stop > args.expected_count:
        raise RuntimeError("requested shard lies outside the input set")
    paths = all_paths[args.offset:stop]
    output = args.output.expanduser().absolute()
    report_path = (
        args.report.expanduser().absolute()
        if args.report is not None
        else output.with_suffix(".json")
    )
    if output.resolve() == report_path.resolve():
        raise RuntimeError("output and report paths must differ")
    if not args.overwrite and (output.exists() or report_path.exists()):
        raise FileExistsError("output or report already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temporary.exists():
        temporary.unlink()

    selected_model, device, selected_metadata = load_restorer(
        selected_path, device=args.device, state="ema"
    )
    seam_model, seam_device, seam_metadata = load_restorer(
        seam_path, device=str(device), state="ema"
    )
    if seam_device != device:
        raise RuntimeError("restorers resolved to different devices")

    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            for local_index, path in enumerate(paths):
                source_started = time.perf_counter()
                image = _read_rgb(path)
                input_hash = hashlib.sha256(image.tobytes()).hexdigest()
                frozen = layouts[path.name]
                if input_hash != frozen["input_pixel_sha256"]:
                    raise RuntimeError(f"test input pixels drifted for {path.name}")
                raw_tiles = split_tiles_numpy(image)
                denoise_started = time.perf_counter()
                selected = restore_tiles_uint8(
                    selected_model, raw_tiles, device, batch_size=args.batch_size
                )
                seam = restore_tiles_uint8(
                    seam_model, raw_tiles, device, batch_size=args.batch_size
                )
                denoise_seconds = time.perf_counter() - denoise_started
                harmonize_started = time.perf_counter()
                restored, diagnostics = render_harmonized_tiles(
                    selected, seam, frozen["layout"]
                )
                harmonize_seconds = time.perf_counter() - harmonize_started
                png = _png_bytes(restored)
                info = zipfile.ZipInfo(path.name, date_time=ARCHIVE_TIMESTAMP)
                info.create_system = 3
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, png, compresslevel=6)
                record = {
                    "source": path.name,
                    "input_pixel_sha256": input_hash,
                    "layout_sha256": frozen["layout_sha256"],
                    "output_png_sha256": hashlib.sha256(png).hexdigest(),
                    "harmonizer": diagnostics,
                    "denoise_seconds": denoise_seconds,
                    "harmonize_seconds": harmonize_seconds,
                    "total_seconds": time.perf_counter() - source_started,
                }
                records.append(record)
                print(
                    json.dumps(
                        {
                            "event": "harmonized_source_complete",
                            "index": local_index + 1,
                            "count": len(paths),
                            "source": path.name,
                            "layout_sha256": frozen["layout_sha256"],
                            "seconds": record["total_seconds"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        os.replace(temporary, output)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise

    names = [path.name for path in paths]
    with zipfile.ZipFile(output) as archive:
        archive_names = archive.namelist()
        if archive_names != names:
            raise RuntimeError("archive member order drift")
        for info in archive.infolist():
            if (
                info.date_time != ARCHIVE_TIMESTAMP
                or info.create_system != 3
                or info.compress_type != zipfile.ZIP_DEFLATED
                or (info.external_attr >> 16) != 0o100644
            ):
                raise RuntimeError("archive member metadata drift")

    method = {
        "layout": "frozen promoted QAP w4 b0.05 i25 from LB-0.203 artifact",
        "selected_tilenaf_weight": 0.5,
        "production_seam_tilenaf_weight": 0.5,
        "rounding": "blend rounded once to uint8, offsets rounded once to uint8",
        "harmonizer": {
            "extrapolation_band": HARMONIZER_CONFIG.extrapolation_band,
            "confidence_scale": HARMONIZER_CONFIG.confidence_scale,
            "confidence_floor": HARMONIZER_CONFIG.confidence_floor,
            "ridge": HARMONIZER_CONFIG.ridge,
            "huber_delta": HARMONIZER_CONFIG.huber_delta,
            "irls_steps": HARMONIZER_CONFIG.irls_steps,
            "max_abs_offset": HARMONIZER_CONFIG.max_abs_offset,
            "global_gauge": "per-channel median offset equals zero",
        },
    }
    payload = {
        "schema_version": 1,
        "kind": "harmonized_frozen_qap_submission_report",
        "status": "test_only_candidate_not_lb_scored",
        "method": method,
        "method_sha256": canonical_sha256(method),
        "anti_leakage": {
            "target_paths_or_pixels_read": False,
            "layout_recomputed": False,
            "layout_source": "two frozen production submission shard reports",
            "harmonizer_inputs": "test input pixels and restored tiles only",
        },
        "assets": {
            "selected_denoiser": {
                "path": str(selected_path),
                "sha256": SELECTED_DENOISER_SHA256,
                "metadata": selected_metadata,
            },
            "seam_denoiser": {
                "path": str(seam_path),
                "sha256": SEAM_DENOISER_SHA256,
                "metadata": seam_metadata,
            },
            "layout_reports": layout_reports,
        },
        "device": str(device),
        "input_dir": str(input_dir),
        "expected_count": args.expected_count,
        "available_input_count": len(all_paths),
        "offset": args.offset,
        "limit": args.limit,
        "count": len(paths),
        "source_names": names,
        "source_names_sha256": hashlib.sha256(
            "\n".join(names).encode("utf-8")
        ).hexdigest(),
        "sources": records,
        "archive": {
            "path": str(output),
            "sha256": sha256_file(output),
            "bytes": output.stat().st_size,
            "member_order": names,
            "member_timestamp": list(ARCHIVE_TIMESTAMP),
            "compression": "ZIP_DEFLATED",
            "compresslevel": 6,
            "flat_member_names": True,
            "unix_mode": "100644",
        },
        "seconds": time.perf_counter() - started,
    }
    if not math.isfinite(payload["seconds"]):
        raise RuntimeError("non-finite runtime")
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "event": "harmonized_submission_complete",
                "output": str(output),
                "sha256": payload["archive"]["sha256"],
                "count": len(paths),
                "seconds": payload["seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return payload


def main() -> None:
    build_submission(parse_args())


if __name__ == "__main__":
    main()

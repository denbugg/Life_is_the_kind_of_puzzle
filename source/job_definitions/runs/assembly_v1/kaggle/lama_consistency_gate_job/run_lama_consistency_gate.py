#!/usr/bin/env python3
"""Cheap, leakage-separated LaMa large-mask consistency correlation gate.

This is deliberately a correlation gate, not a LaMa-guided search.  Phase A
uses only train inputs, the promoted denoiser, four fixed real16 QAP layouts,
and deterministic seam-guarded block/band moves.  It freezes every layout and
LaMa energy before Phase B opens any target image and recomputes target SSIM.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
import types
from typing import Any, Iterable
import urllib.error
import urllib.request
import zipfile

import numpy as np
from PIL import Image


INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
GRID = 24
TILE = 20
TILE_COUNT = GRID * GRID
IMAGE_SIZE = GRID * TILE
MACRO_GRID = 6
MACRO_SIZE = IMAGE_SIZE // MACRO_GRID
INTERIOR_SIZE = 40
INTERIOR_INSET = (MACRO_SIZE - INTERIOR_SIZE) // 2

OFFICIAL_LAMA_COMMIT = "786f5936b27fb3dacd2b1ad799e4de968ea697e7"
OFFICIAL_LAMA_ARCHIVE_URL = (
    "https://codeload.github.com/advimman/lama/tar.gz/" + OFFICIAL_LAMA_COMMIT
)
OFFICIAL_LAMA_ARCHIVE_SHA256 = (
    "6759af2b68f942c32c52ecfed42d46b414cb1a8c1960a7b1167b88d40828deb7"
)
BIG_LAMA_HF_REVISION = "05cb2be7f8dbe6ca7c6e78f4fc827a4b2baaa4a9"
BIG_LAMA_URL = (
    "https://huggingface.co/smartywu/big-lama/resolve/"
    + BIG_LAMA_HF_REVISION
    + "/big-lama.zip"
)
BIG_LAMA_ZIP_SHA256 = (
    "f1b358ca24093b93a106183b98a3dea6e8ed09f3b43ea7251eb2c81e7b4575f6"
)
BIG_LAMA_XET_HASH = (
    "b2a4ef7f88e28fb6c15f0be152d7265a770b54a719774df975847430fa92a283"
)
LPIPS_VERSION = "0.1.4"
LPIPS_WHEEL_SHA256 = (
    "fd537af5828b69d2e6ffc0a397bd506dbc28ca183543617690844c08e102ec5e"
)

EXPECTED_REPORTS = (
    "qap_l1w4_boundary_real16.json",
    "qap_l1w4_multiseed_real16.json",
    "qap_cross_multiseed_real16.json",
    "qap_l1w4_heavy_real16.json",
)
EXPECTED_LAYOUT_MANIFEST_SHA256 = {
    "qap_l1w4_boundary_real16.json": "ac3d37be678dda139495cdd0dbe008fcdc8f9030b40f43a1f0228a1f01290fba",
    "qap_l1w4_multiseed_real16.json": "aca3c831ab57d3a78615f33f017b6742421fa2dfbea4811e17c41529e17c8c84",
    "qap_cross_multiseed_real16.json": "5974a727578bbf2af010e75c1cc87282ba1275af7cb89f15b360679733e9f10c",
    "qap_l1w4_heavy_real16.json": "098e75dcb31fdfbb0e4ce37b7c08e71d477b0c6d07477a11494645124f8dc7a4",
}
BOUNDARY_REPORT = "qap_l1w4_boundary_real16.json"
FIXED_VARIANT = "qap_softcycle_l1_k8__denoised_render"
EXPECTED_SOURCES = (
    "img_003877.png",
    "img_005080.png",
    "img_004383.png",
    "img_006582.png",
    "img_004810.png",
    "img_006306.png",
    "img_005844.png",
    "img_004191.png",
    "img_001281.png",
    "img_003971.png",
    "img_006070.png",
    "img_005710.png",
    "img_005224.png",
    "img_006489.png",
    "img_002514.png",
    "img_004878.png",
)
AUTHORITATIVE_BOUNDARY_MEAN_SSIM = 0.18281991502795386

DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "max_candidates_per_source": 24,
    "max_move_candidates_per_source": 20,
    "seam_guard_ratio": 1.02,
    "block_sizes": [3, 4],
    "mask_count": 4,
    "macro_grid": 6,
    "masked_macroblocks_per_mask": 9,
    "interior_size": 40,
    "lama_batch_size": 2,
    "max_devices": 2,
    "required_t4_devices": 2,
    "lpips_weight": 1.0,
    "lab_blur_weight": 0.25,
    "gaussian_kernel": 9,
    "gaussian_sigma": 2.0,
    "promotion_spearman": 0.25,
    "promotion_pairwise_accuracy": 0.60,
    "required_source_coverage": 16,
    "quality_tie_tolerance": 1e-12,
    "energy_tie_tolerance": 1e-12,
    "baseline_reproduction_tolerance": 2e-5,
    "hard_runtime_seconds": 1500,
    "soft_runtime_seconds": 1380,
    "denoiser_batch_size": 512,
    "seed": 20260711,
}

PHASE_A_DROP_TOKENS = frozenset(
    {"mae", "metric", "metrics", "oracle", "psnr", "score", "scores", "ssim", "target", "targets"}
)
PHASE_A_FORBIDDEN_FROZEN_TOKENS = frozenset(
    {"oracle", "psnr", "ssim", "target", "targets"}
)


def sanitize_phase_a_metadata(value: Any) -> Any:
    """Drop target-like validation fields from model provenance before freeze."""
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            tokens = set(re.findall(r"[a-z0-9]+", str(key).lower()))
            if tokens & PHASE_A_FORBIDDEN_FROZEN_TOKENS:
                continue
            result[key] = sanitize_phase_a_metadata(child)
        return result
    if isinstance(value, list):
        return [sanitize_phase_a_metadata(child) for child in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", default=str(INPUT))
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--runtime-root", default=None)
    parser.add_argument("--code-root", default=None)
    parser.add_argument(
        "--output", default=str(WORKING / "lama_consistency_gate_report.json")
    )
    parser.add_argument(
        "--frozen-output", default=str(WORKING / "lama_consistency_frozen.json")
    )
    parser.add_argument("--validate-config-only", action="store_true")
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument(
        "--validate-reports-only",
        action="store_true",
        help="validate the four strict report schemas without data/model access",
    )
    return parser.parse_args()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_layout(layout: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(layout, dtype="<i4").tobytes()).hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    if int(config["max_candidates_per_source"]) > 32:
        raise ValueError("candidate cap must not exceed 32")
    if int(config["max_move_candidates_per_source"]) + len(EXPECTED_REPORTS) > int(
        config["max_candidates_per_source"]
    ):
        raise ValueError("fixed plus move caps exceed total candidate cap")
    if int(config["mask_count"]) != 4:
        raise ValueError("this gate requires exactly four masks")
    if int(config["macro_grid"]) != MACRO_GRID:
        raise ValueError("macro grid mismatch")
    if int(config["masked_macroblocks_per_mask"]) != 9:
        raise ValueError("each mask must hide nine macroblocks")
    if int(config["interior_size"]) != INTERIOR_SIZE:
        raise ValueError("eroded interior must be 40x40")
    if float(config["seam_guard_ratio"]) > 1.02:
        raise ValueError("seam guard may not exceed the frozen 2% limit")
    if int(config["required_source_coverage"]) != len(EXPECTED_SOURCES):
        raise ValueError("coverage gate must require all real16 sources")
    if int(config["hard_runtime_seconds"]) > 25 * 60:
        raise ValueError("hard runtime cap exceeds 25 minutes")
    if int(config["soft_runtime_seconds"]) >= int(config["hard_runtime_seconds"]):
        raise ValueError("soft deadline must precede hard deadline")
    masks = make_macro_masks()
    if masks.shape != (4, 1, IMAGE_SIZE, IMAGE_SIZE):
        raise AssertionError(masks.shape)
    macro_coverage = np.zeros((MACRO_GRID, MACRO_GRID), dtype=np.int32)
    for mask_index in range(4):
        occupied = []
        for row in range(MACRO_GRID):
            for column in range(MACRO_GRID):
                cell = masks[
                    mask_index,
                    0,
                    row * MACRO_SIZE : (row + 1) * MACRO_SIZE,
                    column * MACRO_SIZE : (column + 1) * MACRO_SIZE,
                ]
                if np.all(cell == 1):
                    occupied.append((row, column))
                    macro_coverage[row, column] += 1
                elif not np.all(cell == 0):
                    raise AssertionError("mask cuts through a macroblock")
        if len(occupied) != 9:
            raise AssertionError(f"mask {mask_index} has {len(occupied)} blocks")
        occupied_set = set(occupied)
        for row, column in occupied:
            for drow, dcolumn in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if (row + drow, column + dcolumn) in occupied_set:
                    raise AssertionError("same-mask macroblocks are edge-adjacent")
    if not np.all(macro_coverage == 1):
        raise AssertionError("the four masks do not cover every macroblock exactly once")
    return {"config": config, "config_sha256": canonical_json_sha256(config)}


def make_macro_masks() -> np.ndarray:
    masks = np.zeros((4, 1, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    for row in range(MACRO_GRID):
        for column in range(MACRO_GRID):
            mask_index = (row % 2) * 2 + (column % 2)
            masks[
                mask_index,
                0,
                row * MACRO_SIZE : (row + 1) * MACRO_SIZE,
                column * MACRO_SIZE : (column + 1) * MACRO_SIZE,
            ] = 1.0
    return masks


def interior_coordinates(mask_index: int) -> list[tuple[int, int]]:
    result = []
    for row in range(MACRO_GRID):
        for column in range(MACRO_GRID):
            if (row % 2) * 2 + (column % 2) != mask_index:
                continue
            result.append(
                (
                    row * MACRO_SIZE + INTERIOR_INSET,
                    column * MACRO_SIZE + INTERIOR_INSET,
                )
            )
    if len(result) != 9:
        raise AssertionError(result)
    return result


def validate_layout(raw: Any, *, context: str) -> np.ndarray:
    values = np.asarray(raw)
    if values.shape != (TILE_COUNT,):
        raise ValueError(f"{context}: expected 576 entries, found {values.shape}")
    if not np.issubdtype(values.dtype, np.integer):
        rounded = np.rint(values)
        if not np.array_equal(values, rounded):
            raise ValueError(f"{context}: layout contains non-integral values")
        values = rounded
    values = values.astype(np.int32, copy=False)
    if not np.array_equal(np.sort(values), np.arange(TILE_COUNT, dtype=np.int32)):
        raise ValueError(f"{context}: layout is not a [0,575] permutation")
    return values.copy()


def load_phase_a_report(path: Path) -> tuple[dict[str, Any], int]:
    dropped = 0

    def strip_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        nonlocal dropped
        result: dict[str, Any] = {}
        for key, value in pairs:
            tokens = set(re.findall(r"[a-z0-9]+", key.lower()))
            if tokens & PHASE_A_DROP_TOKENS:
                dropped += 1
                continue
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=strip_fields)
    if not isinstance(payload, dict):
        raise ValueError(f"report is not an object: {path}")
    return payload, dropped


def phase_a_report_layout_signature(path: Path) -> str:
    payload, _dropped = load_phase_a_report(path)
    manifest = []
    for source_record in payload.get("sources", []):
        if not isinstance(source_record, dict):
            continue
        source = source_record.get("source")
        variants = source_record.get("variants")
        if not isinstance(source, str) or not isinstance(variants, dict):
            continue
        variant = variants.get(FIXED_VARIANT)
        if not isinstance(variant, dict) or "position_to_slot" not in variant:
            continue
        layout = validate_layout(
            variant["position_to_slot"], context=f"{path.name}:{source}"
        )
        manifest.append({"source": source, "layout_sha256": sha256_layout(layout)})
    if len(manifest) != len(EXPECTED_SOURCES):
        raise RuntimeError(f"{path}: incomplete fixed-layout manifest")
    return canonical_json_sha256(manifest)


def select_report_path(root: Path, basename: str) -> Path:
    candidates = sorted(path for path in root.rglob(basename) if path.is_file())
    if not candidates:
        raise FileNotFoundError(f"missing required report {basename} below {root}")
    hashes: dict[str, list[Path]] = defaultdict(list)
    for path in candidates:
        # Do not hash the target-bearing report bytes in Phase A.  Compare only
        # the sanitized fixed-layout manifests exposed by the stripping loader.
        hashes[phase_a_report_layout_signature(path)].append(path)
    if len(hashes) == 1:
        preferred = [
            path for path in candidates if "qap_tuning_night_output/v2" in str(path)
        ]
        return sorted(preferred or candidates)[0]
    preferred = [
        path for path in candidates if "qap_tuning_night_output/v2" in str(path)
    ]
    preferred_hashes = {phase_a_report_layout_signature(path) for path in preferred}
    if len(preferred_hashes) == 1 and preferred:
        return sorted(preferred)[0]
    raise RuntimeError(
        f"ambiguous distinct copies of {basename}: "
        + ", ".join(str(path) for path in candidates)
    )


def discover_fixed_layouts(
    reports_root: Path,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    report_records: list[dict[str, Any]] = []
    for basename in EXPECTED_REPORTS:
        path = select_report_path(reports_root, basename)
        payload, dropped = load_phase_a_report(path)
        source_names = payload.get("source_names")
        if source_names != list(EXPECTED_SOURCES):
            raise RuntimeError(
                f"{basename}: expected authoritative real16 source order, got {source_names}"
            )
        source_records = payload.get("sources")
        if not isinstance(source_records, list) or len(source_records) != 16:
            raise RuntimeError(f"{basename}: invalid sources collection")
        manifest = []
        seen_sources: set[str] = set()
        for source_record in source_records:
            if not isinstance(source_record, dict):
                raise RuntimeError(f"{basename}: invalid source record")
            source = source_record.get("source")
            variants = source_record.get("variants")
            if source not in EXPECTED_SOURCES or not isinstance(variants, dict):
                raise RuntimeError(f"{basename}: invalid source/variants")
            variant = variants.get(FIXED_VARIANT)
            if not isinstance(variant, dict) or "position_to_slot" not in variant:
                raise RuntimeError(f"{basename}:{source}: missing {FIXED_VARIANT}")
            layout = validate_layout(
                variant["position_to_slot"], context=f"{basename}:{source}"
            )
            layout_hash = sha256_layout(layout)
            slug = basename.removesuffix(".json")
            by_source[source].append(
                {
                    "candidate_id": f"fixed:{slug}",
                    "kind": "fixed_qap",
                    "family": "fixed_qap",
                    "origin": basename,
                    "layout_sha256": layout_hash,
                    "position_to_slot": layout,
                }
            )
            manifest.append({"source": source, "layout_sha256": layout_hash})
            seen_sources.add(source)
        if seen_sources != set(EXPECTED_SOURCES):
            raise RuntimeError(f"{basename}: source coverage mismatch")
        layout_manifest_sha256 = canonical_json_sha256(manifest)
        expected_manifest_sha256 = EXPECTED_LAYOUT_MANIFEST_SHA256[basename]
        if layout_manifest_sha256 != expected_manifest_sha256:
            raise RuntimeError(
                f"{basename}: fixed-layout manifest drifted; expected "
                f"{expected_manifest_sha256}, got {layout_manifest_sha256}"
            )
        report_records.append(
            {
                "basename": basename,
                "path": str(path),
                "layout_manifest_sha256": layout_manifest_sha256,
                "expected_layout_manifest_sha256": expected_manifest_sha256,
                "phase_a_dropped_evaluation_fields": dropped,
            }
        )
    for source in EXPECTED_SOURCES:
        records = by_source[source]
        if len(records) != len(EXPECTED_REPORTS):
            raise RuntimeError(f"{source}: expected four fixed QAP layouts")
        if len({record["layout_sha256"] for record in records}) != 4:
            raise RuntimeError(f"{source}: fixed QAP layouts are not four unique candidates")
    return dict(by_source), report_records


def find_data_root(configured: str | None) -> Path:
    if configured:
        root = Path(configured)
        if not all((root / "train" / "inputs" / name).is_file() for name in EXPECTED_SOURCES):
            raise RuntimeError(f"configured data root lacks real16 inputs: {root}")
        return root
    roots = sorted(
        {
            path.parent.parent
            for path in INPUT.glob("**/train/inputs")
            if path.is_dir()
            and all((path / name).is_file() for name in EXPECTED_SOURCES)
        }
    )
    if len(roots) != 1:
        raise RuntimeError(f"expected one puzzle data root, found {roots}")
    return roots[0]


def find_runtime_root(configured: str | None) -> Path:
    if configured:
        root = Path(configured)
        if not (root / "selected_tilenaf_synth_50k.pt").is_file():
            raise RuntimeError(f"configured runtime root lacks denoiser: {root}")
        return root
    roots = sorted(
        {
            path.parent
            for path in INPUT.glob("**/selected_tilenaf_synth_50k.pt")
            if path.is_file()
        }
    )
    if len(roots) != 1:
        raise RuntimeError(f"expected one assembly runtime root, found {roots}")
    return roots[0]


def find_code_root(configured: str | None) -> Path:
    def has_contract(root: Path) -> bool:
        return (
            (root / "src" / "puzzle_denoise_v2" / "inference.py").is_file()
            and all((root / name).is_file() for name in EXPECTED_REPORTS)
        )

    if configured:
        root = Path(configured)
        if not has_contract(root):
            raise RuntimeError(
                f"configured code root lacks denoiser source or fixed QAP reports: {root}"
            )
        return root
    roots = sorted(
        {
            path.parent.parent.parent
            for path in INPUT.glob("**/src/puzzle_denoise_v2/inference.py")
            if path.is_file()
        }
    )
    contracted = [root for root in roots if has_contract(root)]
    if len(contracted) != 1:
        raise RuntimeError(
            "expected one solver code root containing all fixed QAP reports, "
            f"found {contracted}; all denoiser-source roots were {roots}"
        )
    return contracted[0]


def probe_t4x2(config: dict[str, Any]) -> dict[str, Any]:
    subprocess.run(["nvidia-smi"], check=False)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing CPU fallback")
    count = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(index) for index in range(count)]
    required = int(config["required_t4_devices"])
    if count < required or any("T4" not in names[index] for index in range(required)):
        raise RuntimeError(f"this bounded job requires {required} T4 GPUs, found {names}")
    matmul_means = []
    capabilities = []
    for index in range(required):
        device = torch.device(f"cuda:{index}")
        capabilities.append(list(torch.cuda.get_device_capability(index)))
        left = torch.randn(128, 128, device=device)
        right = torch.randn(128, 128, device=device)
        matmul_means.append(float((left @ right).mean().item()))
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device_count": count,
        "devices": names,
        "capabilities": capabilities,
        "arch_list": torch.cuda.get_arch_list(),
        "matmul_means": matmul_means,
        "devices_used": required,
    }


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        values = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if values.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError(f"unexpected RGB shape {values.shape}: {path}")
    return values


def split_tiles(image: np.ndarray) -> np.ndarray:
    return (
        image.reshape(GRID, TILE, GRID, TILE, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(TILE_COUNT, TILE, TILE, 3)
    )


def merge_tiles(tiles: np.ndarray) -> np.ndarray:
    return (
        tiles.reshape(GRID, GRID, TILE, TILE, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(IMAGE_SIZE, IMAGE_SIZE, 3)
    )


def seam_energy(tiles: np.ndarray, layout: np.ndarray) -> float:
    grid = np.asarray(tiles, dtype=np.int16)[layout].reshape(
        GRID, GRID, TILE, TILE, 3
    )
    right = np.abs(grid[:, :-1, :, -1, :] - grid[:, 1:, :, 0, :])
    down = np.abs(grid[:-1, :, -1, :, :] - grid[1:, :, 0, :, :])
    numerator = float(right.sum(dtype=np.float64) + down.sum(dtype=np.float64))
    denominator = right.size + down.size
    return numerator / float(denominator)


def iter_block_moves(layout: np.ndarray, block: int) -> Iterable[tuple[str, np.ndarray]]:
    grid = layout.reshape(GRID, GRID)
    block_grid = GRID // block
    for row in range(block_grid):
        for column in range(block_grid - 1):
            moved = grid.copy()
            left = moved[
                row * block : (row + 1) * block,
                column * block : (column + 1) * block,
            ].copy()
            right = moved[
                row * block : (row + 1) * block,
                (column + 1) * block : (column + 2) * block,
            ].copy()
            moved[
                row * block : (row + 1) * block,
                column * block : (column + 1) * block,
            ] = right
            moved[
                row * block : (row + 1) * block,
                (column + 1) * block : (column + 2) * block,
            ] = left
            yield f"swap_b{block}_h_r{row}_c{column}", moved.reshape(-1)
    for row in range(block_grid - 1):
        for column in range(block_grid):
            moved = grid.copy()
            top = moved[
                row * block : (row + 1) * block,
                column * block : (column + 1) * block,
            ].copy()
            bottom = moved[
                (row + 1) * block : (row + 2) * block,
                column * block : (column + 1) * block,
            ].copy()
            moved[
                row * block : (row + 1) * block,
                column * block : (column + 1) * block,
            ] = bottom
            moved[
                (row + 1) * block : (row + 2) * block,
                column * block : (column + 1) * block,
            ] = top
            yield f"swap_b{block}_v_r{row}_c{column}", moved.reshape(-1)


def iter_band_moves(layout: np.ndarray, block: int) -> Iterable[tuple[str, np.ndarray]]:
    grid = layout.reshape(GRID, GRID)
    block_grid = GRID // block
    for band in range(block_grid):
        rows = slice(band * block, (band + 1) * block)
        for shift in (-block, block):
            moved = grid.copy()
            moved[rows, :] = np.roll(moved[rows, :], shift=shift, axis=1)
            direction = "m" if shift < 0 else "p"
            yield f"roll_b{block}_rows_{band}_{direction}", moved.reshape(-1)
    for band in range(block_grid):
        columns = slice(band * block, (band + 1) * block)
        for shift in (-block, block):
            moved = grid.copy()
            moved[:, columns] = np.roll(moved[:, columns], shift=shift, axis=0)
            direction = "m" if shift < 0 else "p"
            yield f"roll_b{block}_cols_{band}_{direction}", moved.reshape(-1)


def select_moves(
    *,
    source: str,
    tiles: np.ndarray,
    boundary_layout: np.ndarray,
    existing_hashes: set[str],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    baseline_energy = seam_energy(tiles, boundary_layout)
    guard = float(config["seam_guard_ratio"])
    accepted_by_family: dict[str, list[dict[str, Any]]] = {"block": [], "band": []}
    attempted = {"block": 0, "band": 0}
    rejected_guard = {"block": 0, "band": 0}
    seen = set(existing_hashes)
    for block in [int(value) for value in config["block_sizes"]]:
        for family, iterator in (
            ("block", iter_block_moves(boundary_layout, block)),
            ("band", iter_band_moves(boundary_layout, block)),
        ):
            for label, candidate_layout in iterator:
                attempted[family] += 1
                candidate_layout = validate_layout(
                    candidate_layout, context=f"{source}:{label}"
                )
                layout_hash = sha256_layout(candidate_layout)
                if layout_hash in seen:
                    continue
                candidate_energy = seam_energy(tiles, candidate_layout)
                ratio = candidate_energy / max(baseline_energy, 1e-12)
                if ratio > guard + 1e-12:
                    rejected_guard[family] += 1
                    continue
                accepted_by_family[family].append(
                    {
                        "candidate_id": f"move:{label}",
                        "kind": "conservative_move",
                        "family": family,
                        "origin": "authoritative_boundary_qap",
                        "layout_sha256": layout_hash,
                        "position_to_slot": candidate_layout,
                        "seam_energy": candidate_energy,
                        "seam_ratio_to_boundary": ratio,
                    }
                )
                seen.add(layout_hash)
    for family in accepted_by_family:
        accepted_by_family[family].sort(
            key=lambda item: (
                item["seam_ratio_to_boundary"],
                item["candidate_id"],
                item["layout_sha256"],
            )
        )
    cap = int(config["max_move_candidates_per_source"])
    selected: list[dict[str, Any]] = []
    indexes = {"block": 0, "band": 0}
    while len(selected) < cap:
        progressed = False
        for family in ("block", "band"):
            index = indexes[family]
            values = accepted_by_family[family]
            if index < len(values) and len(selected) < cap:
                selected.append(values[index])
                indexes[family] += 1
                progressed = True
        if not progressed:
            break
    diagnostics = {
        "baseline_seam_energy": baseline_energy,
        "guard_ratio": guard,
        "attempted": attempted,
        "accepted_before_cap": {
            family: len(values) for family, values in accepted_by_family.items()
        },
        "rejected_by_guard": rejected_guard,
        "selected": {
            family: sum(item["family"] == family for item in selected)
            for family in ("block", "band")
        },
    }
    return selected, diagnostics


def cache_denoised_tiles(
    *,
    data_root: Path,
    runtime_root: Path,
    code_root: Path,
    cache_root: Path,
    config: dict[str, Any],
    deadline: float,
) -> tuple[dict[str, Path], dict[str, Any]]:
    import torch

    sys.path.insert(0, str(code_root / "src"))
    from puzzle_denoise_v2.inference import load_restorer, restore_tiles_uint8

    checkpoint = runtime_root / "selected_tilenaf_synth_50k.pt"
    restorer, device, metadata = load_restorer(checkpoint, device="cuda:0")
    cache_root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    records = []
    for index, source in enumerate(EXPECTED_SOURCES):
        if time.time() >= deadline:
            raise TimeoutError("soft deadline reached while denoising Phase-A inputs")
        raw_tiles = split_tiles(read_rgb(data_root / "train" / "inputs" / source))
        denoised = restore_tiles_uint8(
            restorer,
            raw_tiles,
            device,
            batch_size=int(config["denoiser_batch_size"]),
        )
        path = cache_root / f"{Path(source).stem}_denoised.npy"
        np.save(path, denoised, allow_pickle=False)
        paths[source] = path
        records.append(
            {
                "source": source,
                "array_sha256": hashlib.sha256(denoised.tobytes()).hexdigest(),
                "file_sha256": sha256_file(path),
            }
        )
        print(
            json.dumps(
                {
                    "event": "phase_a_denoise_complete",
                    "index": index + 1,
                    "count": len(EXPECTED_SOURCES),
                    "source": source,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    del restorer
    torch.cuda.empty_cache()
    return paths, {"checkpoint": metadata, "cached_sources": records}


def build_candidate_pool(
    *,
    fixed_by_source: dict[str, list[dict[str, Any]]],
    tile_paths: dict[str, Path],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sources: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for source in EXPECTED_SOURCES:
        tiles = np.load(tile_paths[source], allow_pickle=False)
        fixed = []
        for item in fixed_by_source[source]:
            copied = dict(item)
            copied["seam_energy"] = seam_energy(tiles, copied["position_to_slot"])
            fixed.append(copied)
        boundary = next(
            item for item in fixed if item["candidate_id"] == "fixed:qap_l1w4_boundary_real16"
        )
        boundary_energy = float(boundary["seam_energy"])
        for item in fixed:
            item["seam_ratio_to_boundary"] = float(item["seam_energy"]) / max(
                boundary_energy, 1e-12
            )
        moves, move_diagnostics = select_moves(
            source=source,
            tiles=tiles,
            boundary_layout=boundary["position_to_slot"],
            existing_hashes={item["layout_sha256"] for item in fixed},
            config=config,
        )
        candidates = fixed + moves
        if len(candidates) > int(config["max_candidates_per_source"]):
            raise AssertionError(f"{source}: candidate cap violated")
        if len({item["layout_sha256"] for item in candidates}) != len(candidates):
            raise AssertionError(f"{source}: duplicate layout survived")
        serializable_candidates = []
        for item in candidates:
            converted = dict(item)
            converted["position_to_slot"] = item["position_to_slot"].tolist()
            serializable_candidates.append(converted)
        sources.append({"source": source, "candidates": serializable_candidates})
        diagnostics.append(
            {
                "source": source,
                "fixed_candidates": len(fixed),
                "move_candidates": len(moves),
                "total_candidates": len(candidates),
                "move_generation": move_diagnostics,
            }
        )
    return sources, diagnostics


def download_verified(
    *,
    url: str,
    destination: Path,
    expected_sha256: str,
    deadline: float,
    retries: int = 3,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and sha256_file(destination) == expected_sha256:
        return {
            "url": url,
            "sha256": expected_sha256,
            "bytes": destination.stat().st_size,
            "cache_hit": True,
        }
    destination.unlink(missing_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        if time.time() >= deadline:
            raise TimeoutError(f"soft deadline reached before downloading {url}")
        temporary = destination.with_suffix(destination.suffix + f".part{attempt}")
        temporary.unlink(missing_ok=True)
        digest = hashlib.sha256()
        size = 0
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "vsos-lama-consistency-gate/1"}
            )
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open(
                "wb"
            ) as handle:
                while True:
                    if time.time() >= deadline:
                        raise TimeoutError(f"soft deadline reached while downloading {url}")
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            actual = digest.hexdigest()
            if actual != expected_sha256:
                raise RuntimeError(
                    f"download hash mismatch for {url}: expected {expected_sha256}, got {actual}"
                )
            temporary.replace(destination)
            return {
                "url": url,
                "sha256": actual,
                "bytes": size,
                "cache_hit": False,
                "attempt": attempt,
            }
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            last_error = exc
            if attempt < retries:
                time.sleep(min(2**attempt, 5))
    raise RuntimeError(f"failed verified download {url}: {last_error}")


def safe_extract_tar(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            resolved = (destination / member.name).resolve()
            if root != resolved and root not in resolved.parents:
                raise RuntimeError(f"unsafe tar member: {member.name}")
        handle.extractall(destination)


def safe_extract_zip(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            resolved = (destination / member.filename).resolve()
            if root != resolved and root not in resolved.parents:
                raise RuntimeError(f"unsafe zip member: {member.filename}")
        handle.extractall(destination)


def ensure_lpips(deadline: float) -> dict[str, Any]:
    import torch
    import torchvision

    before = {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "lpips": package_version("lpips"),
    }
    installed = False
    if before["lpips"] != LPIPS_VERSION:
        if time.time() >= deadline:
            raise TimeoutError("soft deadline reached before LPIPS install")
        requirements = Path(tempfile.gettempdir()) / "lama_gate_lpips_requirements.txt"
        requirements.write_text(
            f"lpips=={LPIPS_VERSION} --hash=sha256:{LPIPS_WHEEL_SHA256}\n",
            encoding="utf-8",
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--disable-pip-version-check",
                "--no-cache-dir",
                "--no-deps",
                "--only-binary=:all:",
                "--require-hashes",
                "-r",
                str(requirements),
            ],
            check=True,
            timeout=max(1, int(deadline - time.time())),
        )
        installed = True
    after = {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "lpips": package_version("lpips"),
    }
    if after["lpips"] != LPIPS_VERSION:
        raise RuntimeError(f"expected lpips {LPIPS_VERSION}, found {after['lpips']}")
    if after["torch"] != before["torch"] or after["torchvision"] != before["torchvision"]:
        raise RuntimeError("LPIPS installation changed torch or torchvision")
    return {"before": before, "after": after, "installed": installed}


def tensor_state_sha256(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key]
        if not hasattr(tensor, "detach"):
            continue
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(key.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def prepare_lpips_cache(deadline: float) -> dict[str, Any]:
    if time.time() >= deadline:
        raise TimeoutError("soft deadline reached before LPIPS cache preparation")
    import lpips

    model = lpips.LPIPS(
        net="alex", version="0.1", spatial=False, eval_mode=True, verbose=False
    )
    metadata = {
        "package_version": package_version("lpips"),
        "network": "alex",
        "version": "0.1",
        "state_sha256": tensor_state_sha256(model.state_dict()),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
    }
    del model
    return metadata


def prepare_official_lama(cache_root: Path, deadline: float) -> dict[str, Any]:
    cache_root.mkdir(parents=True, exist_ok=True)
    source_archive = cache_root / f"lama-{OFFICIAL_LAMA_COMMIT}.tar.gz"
    model_archive = cache_root / f"big-lama-{BIG_LAMA_HF_REVISION}.zip"
    source_download = download_verified(
        url=OFFICIAL_LAMA_ARCHIVE_URL,
        destination=source_archive,
        expected_sha256=OFFICIAL_LAMA_ARCHIVE_SHA256,
        deadline=deadline,
    )
    model_download = download_verified(
        url=BIG_LAMA_URL,
        destination=model_archive,
        expected_sha256=BIG_LAMA_ZIP_SHA256,
        deadline=deadline,
    )
    source_extract = cache_root / "official_lama_source"
    model_extract = cache_root / "official_lama_model"
    if source_extract.exists():
        shutil.rmtree(source_extract)
    if model_extract.exists():
        shutil.rmtree(model_extract)
    safe_extract_tar(source_archive, source_extract)
    safe_extract_zip(model_archive, model_extract)
    source_roots = sorted(
        path.parent.parent
        for path in source_extract.glob("*/saicinpainting/__init__.py")
    )
    config_paths = sorted(model_extract.glob("**/config.yaml"))
    checkpoint_paths = sorted(model_extract.glob("**/models/best.ckpt"))
    if len(source_roots) != 1 or len(config_paths) != 1 or len(checkpoint_paths) != 1:
        raise RuntimeError(
            "unexpected official LaMa extraction: "
            f"sources={source_roots}, configs={config_paths}, checkpoints={checkpoint_paths}"
        )
    return {
        "source_root": str(source_roots[0]),
        "model_root": str(config_paths[0].parent),
        "config_path": str(config_paths[0]),
        "checkpoint_path": str(checkpoint_paths[0]),
        "source": {
            "repository": "https://github.com/advimman/lama",
            "commit": OFFICIAL_LAMA_COMMIT,
            "archive": source_download,
        },
        "weights": {
            "official_readme_recommended_mirror": "smartywu/big-lama",
            "revision": BIG_LAMA_HF_REVISION,
            "archive": model_download,
            "checkpoint_sha256": sha256_file(checkpoint_paths[0]),
            "config_sha256": sha256_file(config_paths[0]),
        },
    }


def resolve_pinned_generator_config(train_config: dict[str, Any]) -> dict[str, Any]:
    """Resolve full-scalar Hydra references used by the pinned LaMa config.

    The released big-lama ``config.yaml`` is mostly resolved, but three
    generator ratios remain strings such as
    ``${generator.init_conv_kwargs.ratio_gout}``.  Importing OmegaConf would
    pull an unnecessary old runtime into this inference-only job, so resolve
    only exact dotted scalar references and fail on every unresolved generator
    interpolation.
    """

    reference_pattern = re.compile(r"^\$\{([A-Za-z0-9_.]+)\}$")

    def lookup(path: str) -> Any:
        value: Any = train_config
        for part in path.split("."):
            if not isinstance(value, dict) or part not in value:
                raise KeyError(f"unknown pinned LaMa config reference {path!r}")
            value = value[part]
        return value

    def resolve(value: Any, stack: tuple[str, ...]) -> Any:
        if isinstance(value, str):
            match = reference_pattern.fullmatch(value)
            if match is None:
                if value.startswith("${"):
                    raise ValueError(
                        f"unsupported unresolved generator interpolation {value!r}"
                    )
                return value
            path = match.group(1)
            if path in stack:
                raise ValueError(f"cyclic pinned LaMa config reference {stack + (path,)}")
            return resolve(lookup(path), stack + (path,))
        if isinstance(value, dict):
            return {key: resolve(child, stack + (str(key),)) for key, child in value.items()}
        if isinstance(value, list):
            return [resolve(child, stack + (str(index),)) for index, child in enumerate(value)]
        return value

    generator = train_config.get("generator")
    if not isinstance(generator, dict):
        raise ValueError("pinned LaMa config has no generator object")
    resolved = resolve(generator, ("generator",))
    if not isinstance(resolved, dict):
        raise AssertionError("resolved generator config is not a dictionary")
    return resolved


def load_lama_generator(
    source_root: Path, config_path: Path, checkpoint_path: Path, device_index: int
) -> tuple[Any, dict[str, Any]]:
    import torch
    import yaml

    # The official generator imports only seed_everything from Lightning via
    # saicinpainting.utils.  A tiny inference-only compatibility shim avoids
    # installing LaMa's 2021 Lightning stack or replacing Kaggle PyTorch.
    lightning_stub = types.ModuleType("pytorch_lightning")

    def seed_everything(seed: int, *_args: Any, **_kwargs: Any) -> int:
        torch.manual_seed(seed)
        return seed

    lightning_stub.seed_everything = seed_everything
    sys.modules["pytorch_lightning"] = lightning_stub
    sys.path.insert(0, str(source_root))
    from saicinpainting.training.modules import make_generator

    train_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    generator_config = resolve_pinned_generator_config(train_config)
    resolved_train_config = dict(train_config)
    resolved_train_config["generator"] = generator_config
    generator_config = dict(generator_config)
    kind = generator_config.pop("kind")
    generator = make_generator(
        resolved_train_config, kind=kind, **generator_config
    )
    safe_load_error = None
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        load_mode = "weights_only"
    except Exception as exc:
        # This exact archive is SHA-256 pinned before extraction.  Old
        # Lightning checkpoints sometimes contain harmless unsupported config
        # objects, so a pinned-archive compatibility fallback is explicit.
        safe_load_error = f"{type(exc).__name__}: {exc}"[:1000]
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        load_mode = "trusted_sha256_pinned_pickle_fallback"
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict):
        raise RuntimeError("LaMa checkpoint has no state dictionary")
    generator_state = {
        key.removeprefix("generator."): value
        for key, value in state.items()
        if key.startswith("generator.")
    }
    if not generator_state:
        raise RuntimeError("LaMa checkpoint contains no generator.* weights")
    incompatible = generator.load_state_dict(generator_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict LaMa load failed: {incompatible}")
    device = torch.device(f"cuda:{device_index}")
    generator.eval().to(device)
    for parameter in generator.parameters():
        parameter.requires_grad_(False)
    metadata = {
        "kind": kind,
        "config_sha256": sha256_file(config_path),
        "resolved_generator_config_sha256": canonical_json_sha256(
            resolved_train_config["generator"]
        ),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "load_mode": load_mode,
        "weights_only_error": safe_load_error,
        "generator_state_sha256": tensor_state_sha256(generator_state),
        "parameters": int(sum(parameter.numel() for parameter in generator.parameters())),
        "device": str(device),
        "inference_compatibility_shim": "seed_everything_only_no_lightning_install",
    }
    return generator, metadata


def score_worker(
    *,
    device_index: int,
    sources: list[str],
    pool_path: str,
    tile_path_map: dict[str, str],
    lama_preparation: dict[str, Any],
    config: dict[str, Any],
    output_path: str,
    deadline: float,
) -> None:
    try:
        import torch
        import kornia
        import lpips

        torch.manual_seed(int(config["seed"]) + device_index)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        device = torch.device(f"cuda:{device_index}")
        generator, generator_metadata = load_lama_generator(
            Path(lama_preparation["source_root"]),
            Path(lama_preparation["config_path"]),
            Path(lama_preparation["checkpoint_path"]),
            device_index,
        )
        perceptual = lpips.LPIPS(
            net="alex", version="0.1", spatial=False, eval_mode=True, verbose=False
        ).eval().to(device)
        for parameter in perceptual.parameters():
            parameter.requires_grad_(False)
        lpips_metadata = {
            "package_version": package_version("lpips"),
            "state_sha256": tensor_state_sha256(perceptual.state_dict()),
            "parameters": int(sum(p.numel() for p in perceptual.parameters())),
        }
        pool = json.loads(Path(pool_path).read_text(encoding="utf-8"))
        source_lookup = {item["source"]: item for item in pool["sources"]}
        masks_cpu = torch.from_numpy(make_macro_masks())
        kernel = int(config["gaussian_kernel"])
        sigma = float(config["gaussian_sigma"])
        normalizer = torch.tensor([100.0, 128.0, 128.0], device=device).view(
            1, 3, 1, 1
        )
        records: list[dict[str, Any]] = []
        worker_started = time.perf_counter()
        for source_index, source in enumerate(sources):
            if time.time() >= deadline:
                raise TimeoutError("soft deadline reached in LaMa worker")
            source_started = time.perf_counter()
            source_record = source_lookup[source]
            candidates = source_record["candidates"]
            if len(candidates) > 32:
                raise RuntimeError(f"{source}: more than 32 candidates")
            tiles = np.load(tile_path_map[source], allow_pickle=False)
            mosaics = []
            for candidate in candidates:
                layout = validate_layout(
                    candidate["position_to_slot"], context=candidate["candidate_id"]
                )
                mosaic = merge_tiles(tiles[layout])
                mosaics.append(
                    torch.from_numpy(np.ascontiguousarray(mosaic.transpose(2, 0, 1)))
                    .float()
                    .div_(255.0)
                )
            energies = np.full((len(candidates), 4), np.nan, dtype=np.float64)
            lpips_parts = np.full_like(energies, np.nan)
            lab_parts = np.full_like(energies, np.nan)
            pairs = [
                (candidate_index, mask_index)
                for candidate_index in range(len(candidates))
                for mask_index in range(4)
            ]
            batch_size = int(config["lama_batch_size"])
            for start in range(0, len(pairs), batch_size):
                if time.time() >= deadline:
                    raise TimeoutError("soft deadline reached in LaMa forward loop")
                batch_pairs = pairs[start : start + batch_size]
                images = torch.stack(
                    [mosaics[candidate_index] for candidate_index, _ in batch_pairs]
                ).to(device)
                masks = torch.stack(
                    [masks_cpu[mask_index] for _, mask_index in batch_pairs]
                ).to(device)
                masked = images * (1.0 - masks)
                with torch.inference_mode():
                    prediction = generator(torch.cat([masked, masks], dim=1)).clamp(
                        0.0, 1.0
                    )
                    inpainted = prediction * masks + images * (1.0 - masks)
                    output_crops = []
                    reference_crops = []
                    owner_slices = []
                    for local_index, (_, mask_index) in enumerate(batch_pairs):
                        owner_start = len(output_crops)
                        for row, column in interior_coordinates(mask_index):
                            output_crops.append(
                                inpainted[
                                    local_index : local_index + 1,
                                    :,
                                    row : row + INTERIOR_SIZE,
                                    column : column + INTERIOR_SIZE,
                                ]
                            )
                            reference_crops.append(
                                images[
                                    local_index : local_index + 1,
                                    :,
                                    row : row + INTERIOR_SIZE,
                                    column : column + INTERIOR_SIZE,
                                ]
                            )
                        owner_slices.append((owner_start, len(output_crops)))
                    output_crops_tensor = torch.cat(output_crops, dim=0)
                    reference_crops_tensor = torch.cat(reference_crops, dim=0)
                    perceptual_values = perceptual(
                        output_crops_tensor * 2.0 - 1.0,
                        reference_crops_tensor * 2.0 - 1.0,
                    ).reshape(-1)
                    output_blur = kornia.filters.gaussian_blur2d(
                        output_crops_tensor,
                        (kernel, kernel),
                        (sigma, sigma),
                        border_type="reflect",
                    )
                    reference_blur = kornia.filters.gaussian_blur2d(
                        reference_crops_tensor,
                        (kernel, kernel),
                        (sigma, sigma),
                        border_type="reflect",
                    )
                    output_lab = kornia.color.rgb_to_lab(output_blur.clamp(0.0, 1.0))
                    reference_lab = kornia.color.rgb_to_lab(
                        reference_blur.clamp(0.0, 1.0)
                    )
                    lab_values = (
                        ((output_lab - reference_lab).abs() / normalizer)
                        .mean(dim=(1, 2, 3))
                        .reshape(-1)
                    )
                for local_index, (candidate_index, mask_index) in enumerate(batch_pairs):
                    slice_start, slice_end = owner_slices[local_index]
                    lpips_value = float(
                        perceptual_values[slice_start:slice_end].mean().item()
                    )
                    lab_value = float(lab_values[slice_start:slice_end].mean().item())
                    energy = (
                        float(config["lpips_weight"]) * lpips_value
                        + float(config["lab_blur_weight"]) * lab_value
                    )
                    if not all(math.isfinite(value) for value in (lpips_value, lab_value, energy)):
                        raise FloatingPointError("non-finite LaMa energy")
                    lpips_parts[candidate_index, mask_index] = lpips_value
                    lab_parts[candidate_index, mask_index] = lab_value
                    energies[candidate_index, mask_index] = energy
            for candidate_index, candidate in enumerate(candidates):
                records.append(
                    {
                        "source": source,
                        "candidate_id": candidate["candidate_id"],
                        "layout_sha256": candidate["layout_sha256"],
                        "lama_energy": float(energies[candidate_index].mean()),
                        "lama_energy_by_mask": energies[candidate_index].tolist(),
                        "lpips_by_mask": lpips_parts[candidate_index].tolist(),
                        "normalized_blurred_lab_l1_by_mask": lab_parts[
                            candidate_index
                        ].tolist(),
                        "device": str(device),
                    }
                )
            print(
                json.dumps(
                    {
                        "event": "lama_energy_source_complete",
                        "device": str(device),
                        "index": source_index + 1,
                        "count": len(sources),
                        "source": source,
                        "candidates": len(candidates),
                        "seconds": time.perf_counter() - source_started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        result = {
            "device": str(device),
            "sources": sources,
            "seconds": time.perf_counter() - worker_started,
            "generator": generator_metadata,
            "lpips": lpips_metadata,
            "kornia": package_version("kornia"),
            "records": records,
        }
        write_json_atomic(Path(output_path), result)
    except Exception:
        failure = {
            "device_index": device_index,
            "sources": sources,
            "traceback": traceback.format_exc(),
        }
        write_json_atomic(Path(output_path).with_suffix(".failure.json"), failure)
        raise


def run_energy_workers(
    *,
    pool_path: Path,
    tile_paths: dict[str, Path],
    lama_preparation: dict[str, Any],
    config: dict[str, Any],
    cache_root: Path,
    deadline: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    device_count = int(config["max_devices"])
    groups = [list(EXPECTED_SOURCES[index::device_count]) for index in range(device_count)]
    context = mp.get_context("spawn")
    processes = []
    output_paths = []
    for device_index, sources in enumerate(groups):
        output = cache_root / f"lama_worker_{device_index}.json"
        output_paths.append(output)
        process = context.Process(
            target=score_worker,
            kwargs={
                "device_index": device_index,
                "sources": sources,
                "pool_path": str(pool_path),
                "tile_path_map": {
                    source: str(path) for source, path in tile_paths.items()
                },
                "lama_preparation": lama_preparation,
                "config": config,
                "output_path": str(output),
                "deadline": deadline,
            },
        )
        process.start()
        processes.append(process)
    while any(process.is_alive() for process in processes):
        if time.time() >= deadline:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join(timeout=5)
            raise TimeoutError("soft runtime deadline terminated LaMa workers")
        time.sleep(0.25)
    failures = [process.exitcode for process in processes if process.exitcode != 0]
    if failures:
        failure_paths = sorted(cache_root.glob("lama_worker_*.failure.json"))
        details = [path.read_text(encoding="utf-8") for path in failure_paths]
        raise RuntimeError(f"LaMa workers failed {failures}: {details}")
    worker_records = [json.loads(path.read_text(encoding="utf-8")) for path in output_paths]
    generator_hashes = {
        record["generator"]["generator_state_sha256"] for record in worker_records
    }
    lpips_hashes = {record["lpips"]["state_sha256"] for record in worker_records}
    if len(generator_hashes) != 1 or len(lpips_hashes) != 1:
        raise RuntimeError("two GPU workers loaded different frozen model states")
    energies = [item for record in worker_records for item in record["records"]]
    return energies, worker_records


def audit_frozen_keys(value: Any, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            tokens = set(re.findall(r"[a-z0-9]+", str(key).lower()))
            forbidden = tokens & PHASE_A_FORBIDDEN_FROZEN_TOKENS
            if forbidden:
                raise RuntimeError(
                    f"Phase-A frozen artifact contains forbidden key at {path + (str(key),)}"
                )
            audit_frozen_keys(child, path + (str(key),))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            audit_frozen_keys(child, path + (str(index),))


def freeze_phase_a(
    *,
    output: Path,
    pool: dict[str, Any],
    energies: list[dict[str, Any]],
    config: dict[str, Any],
    reports: list[dict[str, Any]],
    denoiser: dict[str, Any],
    lama_preparation: dict[str, Any],
    lpips_preparation: dict[str, Any],
    worker_records: list[dict[str, Any]],
    hardware: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    energy_lookup = {
        (record["source"], record["candidate_id"], record["layout_sha256"]): record
        for record in energies
    }
    frozen_sources = []
    for source_record in pool["sources"]:
        source = source_record["source"]
        candidates = []
        for candidate in source_record["candidates"]:
            key = (source, candidate["candidate_id"], candidate["layout_sha256"])
            energy = energy_lookup.get(key)
            if energy is None:
                raise RuntimeError(f"missing LaMa energy for {key}")
            candidates.append({**candidate, **energy})
        frozen_sources.append({"source": source, "candidates": candidates})
    frozen = {
        "schema_version": 1,
        "kind": "lama_large_mask_input_only_frozen",
        "phase": "A_input_only_complete",
        "anti_leakage": {
            "clean_images_opened": False,
            "evaluation_fields_dropped_during_report_decode": True,
            "candidate_generation_uses_only_denoised_inputs_and_fixed_layouts": True,
            "layout_and_energy_frozen_before_clean_image_access": True,
        },
        "config": config,
        "config_sha256": canonical_json_sha256(config),
        "candidate_pool_sha256": pool["candidate_pool_sha256"],
        "reports": reports,
        "hardware": hardware,
        "denoiser": sanitize_phase_a_metadata(denoiser),
        "official_lama": lama_preparation,
        "lpips_preparation": lpips_preparation,
        "workers": [
            {
                key: value
                for key, value in record.items()
                if key != "records"
            }
            for record in worker_records
        ],
        "sources": frozen_sources,
    }
    audit_frozen_keys(frozen)
    write_json_atomic(output, frozen)
    digest = sha256_file(output)
    reloaded = json.loads(output.read_text(encoding="utf-8"))
    audit_frozen_keys(reloaded)
    if sha256_file(output) != digest:
        raise RuntimeError("frozen Phase-A artifact changed during readback")
    return reloaded, digest


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def spearman(values_a: list[float], values_b: list[float]) -> float | None:
    if len(values_a) < 2:
        return None
    ranks_a = average_ranks(np.asarray(values_a, dtype=np.float64))
    ranks_b = average_ranks(np.asarray(values_b, dtype=np.float64))
    centered_a = ranks_a - ranks_a.mean()
    centered_b = ranks_b - ranks_b.mean()
    denominator = math.sqrt(float(centered_a @ centered_a) * float(centered_b @ centered_b))
    if denominator <= 0:
        return None
    return float((centered_a @ centered_b) / denominator)


def pairwise_counts(
    naturalness: list[float], quality: list[float], config: dict[str, Any]
) -> tuple[float, int]:
    correct = 0.0
    pairs = 0
    target_tolerance = float(config["quality_tie_tolerance"])
    energy_tolerance = float(config["energy_tie_tolerance"])
    for first in range(len(quality)):
        for second in range(first + 1, len(quality)):
            target_delta = quality[first] - quality[second]
            if abs(target_delta) <= target_tolerance:
                continue
            energy_delta = naturalness[first] - naturalness[second]
            pairs += 1
            if abs(energy_delta) <= energy_tolerance:
                correct += 0.5
            elif (energy_delta > 0) == (target_delta > 0):
                correct += 1.0
    return correct, pairs


def attach_phase_b_evaluation(
    *,
    frozen: dict[str, Any],
    frozen_sha256: str,
    frozen_path: Path,
    data_root: Path,
    tile_paths: dict[str, Path],
    config: dict[str, Any],
) -> dict[str, Any]:
    from skimage.metrics import structural_similarity

    if sha256_file(frozen_path) != frozen_sha256:
        raise RuntimeError("Phase-A frozen hash mismatch before Phase B")
    source_reports = []
    fixed_correct_total = 0.0
    fixed_pairs_total = 0
    fixed_correlations = []
    diagnostic_all_correct = 0.0
    diagnostic_all_pairs = 0
    diagnostic_all_correlations = []
    baseline_values = []
    for source_record in frozen["sources"]:
        source = source_record["source"]
        tiles = np.load(tile_paths[source], allow_pickle=False)
        clean_image = read_rgb(data_root / "train" / "targets" / source)
        attached_candidates = []
        naturalness = []
        quality = []
        fixed_naturalness = []
        fixed_quality = []
        for candidate in source_record["candidates"]:
            layout = validate_layout(
                candidate["position_to_slot"], context=candidate["candidate_id"]
            )
            predicted = merge_tiles(tiles[layout])
            image_ssim = float(
                structural_similarity(
                    clean_image, predicted, channel_axis=2, data_range=255
                )
            )
            attached_candidates.append(
                {
                    **candidate,
                    "post_freeze_target_ssim": image_ssim,
                    "naturalness": -float(candidate["lama_energy"]),
                }
            )
            naturalness.append(-float(candidate["lama_energy"]))
            quality.append(image_ssim)
            if candidate["family"] == "fixed_qap":
                fixed_naturalness.append(-float(candidate["lama_energy"]))
                fixed_quality.append(image_ssim)
            if candidate["candidate_id"] == "fixed:qap_l1w4_boundary_real16":
                baseline_values.append(image_ssim)
        if len(fixed_naturalness) != len(EXPECTED_REPORTS):
            raise RuntimeError(
                f"{source}: expected four fixed-QAP candidates, found "
                f"{len(fixed_naturalness)}"
            )
        fixed_correlation = spearman(fixed_naturalness, fixed_quality)
        fixed_correct, fixed_pairs = pairwise_counts(
            fixed_naturalness, fixed_quality, config
        )
        diagnostic_correlation = spearman(naturalness, quality)
        diagnostic_correct, diagnostic_pairs = pairwise_counts(
            naturalness, quality, config
        )
        if fixed_correlation is not None:
            fixed_correlations.append(fixed_correlation)
        if diagnostic_correlation is not None:
            diagnostic_all_correlations.append(diagnostic_correlation)
        fixed_correct_total += fixed_correct
        fixed_pairs_total += fixed_pairs
        diagnostic_all_correct += diagnostic_correct
        diagnostic_all_pairs += diagnostic_pairs
        source_reports.append(
            {
                "source": source,
                "candidate_count": len(attached_candidates),
                "primary_fixed_qap": {
                    "candidate_count": len(fixed_naturalness),
                    "spearman": fixed_correlation,
                    "pairwise_correct_credit": fixed_correct,
                    "pairwise_pairs": fixed_pairs,
                    "pairwise_accuracy": (
                        fixed_correct / fixed_pairs if fixed_pairs else None
                    ),
                },
                "all_candidate_pool_diagnostic": {
                    "candidate_count": len(naturalness),
                    "spearman": diagnostic_correlation,
                    "pairwise_correct_credit": diagnostic_correct,
                    "pairwise_pairs": diagnostic_pairs,
                    "pairwise_accuracy": (
                        diagnostic_correct / diagnostic_pairs
                        if diagnostic_pairs
                        else None
                    ),
                },
                "candidates": attached_candidates,
            }
        )
    coverage = len(
        [
            source
            for source in source_reports
            if source["primary_fixed_qap"]["spearman"] is not None
            and source["primary_fixed_qap"]["pairwise_pairs"] > 0
        ]
    )
    mean_spearman = (
        float(np.mean(fixed_correlations)) if fixed_correlations else None
    )
    micro_pairwise = (
        fixed_correct_total / fixed_pairs_total if fixed_pairs_total else None
    )
    diagnostic_mean_spearman = (
        float(np.mean(diagnostic_all_correlations))
        if diagnostic_all_correlations
        else None
    )
    diagnostic_micro_pairwise = (
        diagnostic_all_correct / diagnostic_all_pairs
        if diagnostic_all_pairs
        else None
    )
    reproduced_baseline = float(np.mean(baseline_values))
    reproduction_delta = reproduced_baseline - AUTHORITATIVE_BOUNDARY_MEAN_SSIM
    coverage_pass = coverage == int(config["required_source_coverage"])
    baseline_pass = abs(reproduction_delta) <= float(
        config["baseline_reproduction_tolerance"]
    )
    correlation_pass = (
        mean_spearman is not None
        and micro_pairwise is not None
        and mean_spearman >= float(config["promotion_spearman"])
        and micro_pairwise >= float(config["promotion_pairwise_accuracy"])
    )
    return {
        "schema_version": 1,
        "kind": "lama_large_mask_consistency_correlation_gate",
        "anti_leakage": {
            "phase_a_frozen_sha256": frozen_sha256,
            "phase_a_readback_verified_before_target_access": True,
            "targets_opened_only_in_phase_b": True,
            "phase_b_cannot_change_candidates_or_energies": True,
        },
        "definition": {
            "candidate_family": "QAP-near only",
            "weak_component_candidates_included": False,
            "search_performed": False,
            "primary_correlation_subset": "four fixed v2 QAP layouts only",
            "generated_moves_role": "diagnostic only; excluded from promotion correlation",
            "masks": "four fixed 2x2-phase checkerboards on a 6x6 grid",
            "hidden_macroblock_size": [80, 80],
            "evaluated_interior_size": [40, 40],
            "energy": "mean_masks(mean_9_crops(LPIPS_alex_v0.1) + 0.25 * normalized_blurred_LabL1)",
            "render_view": "promoted tile denoiser",
        },
        "thresholds": {
            "applied_to": "four fixed v2 QAP layouts per source",
            "mean_within_source_spearman": config["promotion_spearman"],
            "micro_pairwise_accuracy": config["promotion_pairwise_accuracy"],
            "source_coverage": config["required_source_coverage"],
        },
        "authoritative_baseline": {
            "expected_mean_ssim": AUTHORITATIVE_BOUNDARY_MEAN_SSIM,
            "reproduced_mean_ssim": reproduced_baseline,
            "difference": reproduction_delta,
            "tolerance": config["baseline_reproduction_tolerance"],
            "passes": baseline_pass,
            "candidate_id": "fixed:qap_l1w4_boundary_real16",
        },
        "macro": {
            "source_coverage": coverage,
            "mean_within_source_spearman": mean_spearman,
            "micro_pairwise_accuracy": micro_pairwise,
            "pairwise_correct_credit": fixed_correct_total,
            "pairwise_pairs": fixed_pairs_total,
            "primary_fixed_qap_candidates_per_source": len(EXPECTED_REPORTS),
            "all_candidate_pool_diagnostic": {
                "mean_within_source_spearman": diagnostic_mean_spearman,
                "micro_pairwise_accuracy": diagnostic_micro_pairwise,
                "pairwise_correct_credit": diagnostic_all_correct,
                "pairwise_pairs": diagnostic_all_pairs,
            },
            "candidate_count_min": min(item["candidate_count"] for item in source_reports),
            "candidate_count_max": max(item["candidate_count"] for item in source_reports),
            "candidate_count_total": sum(item["candidate_count"] for item in source_reports),
        },
        "gate": {
            "coverage_pass": coverage_pass,
            "baseline_reproduction_pass": baseline_pass,
            "fixed_qap_correlation_thresholds_pass": correlation_pass,
            "passes": coverage_pass and baseline_pass and correlation_pass,
            "next_step_if_passed": "prepare a separate bounded LaMa-guided search; no search is included here",
            "stop_if_failed": "close LaMa and generic no-reference reranking for competitive layouts",
        },
        "sources": source_reports,
    }


def synthetic_smoke(config: dict[str, Any]) -> dict[str, Any]:
    validate_config(config)
    interpolation_fixture = {
        "generator": {
            "init_conv_kwargs": {"ratio_gout": 0},
            "downsample_conv_kwargs": {
                "ratio_gin": "${generator.init_conv_kwargs.ratio_gout}",
                "ratio_gout": "${generator.downsample_conv_kwargs.ratio_gin}",
            },
            "resnet_conv_kwargs": {
                "ratio_gin": 0.75,
                "ratio_gout": "${generator.resnet_conv_kwargs.ratio_gin}",
            },
        }
    }
    resolved_fixture = resolve_pinned_generator_config(interpolation_fixture)
    if resolved_fixture["downsample_conv_kwargs"] != {
        "ratio_gin": 0,
        "ratio_gout": 0,
    } or resolved_fixture["resnet_conv_kwargs"]["ratio_gout"] != 0.75:
        raise AssertionError("pinned LaMa generator interpolation smoke failed")
    rng = np.random.default_rng(int(config["seed"]))
    tiles = rng.integers(0, 256, size=(TILE_COUNT, TILE, TILE, 3), dtype=np.uint8)
    identity = np.arange(TILE_COUNT, dtype=np.int32)
    fixed_hashes = {sha256_layout(identity)}
    moves, diagnostics = select_moves(
        source="synthetic.png",
        tiles=tiles,
        boundary_layout=identity,
        existing_hashes=fixed_hashes,
        config=config,
    )
    for move in moves:
        validate_layout(move["position_to_slot"], context=move["candidate_id"])
        if move["seam_ratio_to_boundary"] > float(config["seam_guard_ratio"]) + 1e-12:
            raise AssertionError("synthetic move bypassed seam guard")
    frozen = {
        "kind": "synthetic_input_only_freeze",
        "anti_leakage": {"clean_images_opened": False},
        "sources": [
            {
                "source": "synthetic.png",
                "candidates": [
                    {
                        "candidate_id": "fixed:synthetic",
                        "layout_sha256": sha256_layout(identity),
                        "position_to_slot": identity.tolist(),
                        "lama_energy": 0.5,
                    }
                ],
            }
        ],
    }
    audit_frozen_keys(frozen)
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "frozen.json"
        write_json_atomic(path, frozen)
        digest = sha256_file(path)
        reloaded = json.loads(path.read_text(encoding="utf-8"))
        audit_frozen_keys(reloaded)
        if sha256_file(path) != digest:
            raise AssertionError("synthetic freeze hash changed")
    return {
        "status": "passed",
        "mask_coverage": "4x9 covers all 36 macroblocks exactly once",
        "pinned_generator_interpolation_resolution": "passed",
        "synthetic_move_count": len(moves),
        "move_diagnostics": diagnostics,
        "freeze_sha256": digest,
    }


def main_impl(args: argparse.Namespace) -> None:
    config = dict(DEFAULT_CONFIG)
    validation = validate_config(config)
    if args.validate_config_only:
        print(json.dumps(validation, indent=2, sort_keys=True))
        return
    if args.synthetic_smoke:
        print(json.dumps(synthetic_smoke(config), indent=2, sort_keys=True))
        return
    reports_root = Path(args.reports_root)
    fixed_by_source, reports = discover_fixed_layouts(reports_root)
    if args.validate_reports_only:
        print(
            json.dumps(
                {
                    "status": "passed",
                    "sources": len(fixed_by_source),
                    "fixed_candidates_per_source": 4,
                    "reports": reports,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    started_wall = time.time()
    soft_deadline = started_wall + int(config["soft_runtime_seconds"])
    data_root = find_data_root(args.data_root)
    runtime_root = find_runtime_root(args.runtime_root)
    code_root = find_code_root(args.code_root)
    hardware = probe_t4x2(config)
    dependency_record = ensure_lpips(soft_deadline)
    lpips_preparation = prepare_lpips_cache(soft_deadline)

    cache_root = Path(tempfile.gettempdir()) / "vsos_lama_consistency_gate"
    if cache_root.exists():
        shutil.rmtree(cache_root)
    cache_root.mkdir(parents=True)
    lama_preparation = prepare_official_lama(cache_root / "models", soft_deadline)
    tile_paths, denoiser_record = cache_denoised_tiles(
        data_root=data_root,
        runtime_root=runtime_root,
        code_root=code_root,
        cache_root=cache_root / "denoised_tiles",
        config=config,
        deadline=soft_deadline,
    )
    pool_sources, move_diagnostics = build_candidate_pool(
        fixed_by_source=fixed_by_source, tile_paths=tile_paths, config=config
    )
    pool = {
        "schema_version": 1,
        "kind": "qap_near_input_only_candidate_pool",
        "config_sha256": canonical_json_sha256(config),
        "sources": pool_sources,
        "move_diagnostics": move_diagnostics,
    }
    pool["candidate_pool_sha256"] = canonical_json_sha256(pool["sources"])
    pool_path = cache_root / "phase_a_candidate_pool.json"
    write_json_atomic(pool_path, pool)
    energies, worker_records = run_energy_workers(
        pool_path=pool_path,
        tile_paths=tile_paths,
        lama_preparation=lama_preparation,
        config=config,
        cache_root=cache_root,
        deadline=soft_deadline,
    )
    if any(
        record["lpips"]["state_sha256"] != lpips_preparation["state_sha256"]
        for record in worker_records
    ):
        raise RuntimeError("worker LPIPS state differs from the parent-frozen state")
    frozen_path = Path(args.frozen_output)
    frozen, frozen_sha256 = freeze_phase_a(
        output=frozen_path,
        pool=pool,
        energies=energies,
        config=config,
        reports=reports,
        denoiser=denoiser_record,
        lama_preparation=lama_preparation,
        lpips_preparation={
            **lpips_preparation,
            "dependency_install": dependency_record,
        },
        worker_records=worker_records,
        hardware=hardware,
    )
    print(
        json.dumps(
            {
                "event": "phase_a_layouts_and_lama_energies_frozen",
                "path": str(frozen_path),
                "sha256": frozen_sha256,
                "targets_opened": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    final_report = attach_phase_b_evaluation(
        frozen=frozen,
        frozen_sha256=frozen_sha256,
        frozen_path=frozen_path,
        data_root=data_root,
        tile_paths=tile_paths,
        config=config,
    )
    final_report["runtime"] = {
        "seconds": time.time() - started_wall,
        "hard_cap_seconds": config["hard_runtime_seconds"],
        "soft_deadline_seconds": config["soft_runtime_seconds"],
    }
    final_report["provenance"] = {
        "runner_sha256": sha256_file(Path(__file__)),
        "config_sha256": canonical_json_sha256(config),
        "phase_a_frozen_sha256": frozen_sha256,
        "candidate_pool_sha256": pool["candidate_pool_sha256"],
        "official_lama_source": "https://github.com/advimman/lama",
        "official_lama_commit": OFFICIAL_LAMA_COMMIT,
        "official_readme_model_url": BIG_LAMA_URL,
        "big_lama_archive_sha256": BIG_LAMA_ZIP_SHA256,
        "big_lama_xet_hash": BIG_LAMA_XET_HASH,
    }
    output = Path(args.output)
    write_json_atomic(output, final_report)
    print(
        json.dumps(
            {
                "event": "lama_consistency_gate_complete",
                "output": str(output),
                "sha256": sha256_file(output),
                "macro": final_report["macro"],
                "gate": final_report["gate"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    hard_seconds = int(DEFAULT_CONFIG["hard_runtime_seconds"])

    def alarm_handler(_signum: int, _frame: Any) -> None:
        raise TimeoutError(f"hard runtime cap reached at {hard_seconds} seconds")

    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, alarm_handler)
        signal.alarm(hard_seconds)
    try:
        main_impl(args)
    except Exception:
        if str(WORKING).startswith("/kaggle") and WORKING.exists():
            write_json_atomic(
                WORKING / "lama_consistency_gate_failure.json",
                {
                    "kind": "lama_consistency_gate_failure",
                    "traceback": traceback.format_exc(),
                    "runner_sha256": sha256_file(Path(__file__)),
                },
            )
        raise
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)


if __name__ == "__main__":
    main()

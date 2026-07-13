#!/usr/bin/env python3
"""Bounded MAE-guided global-layout search with a strict input-only freeze.

This is deliberately a falsification gate.  It starts from the exact frozen
boundary-QAP real16 layouts, searches a small deterministic population of valid
global permutations, and permits MAE to replace the baseline only inside a
strict denoised L1w4 seam guard.  Every searched layout, energy, and input-only
selection is written and hashed before any clean target image or recorded
target metric is opened.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import hashlib
from importlib import metadata as importlib_metadata
import itertools
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image


INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
GRID = 24
TILE = 20
TILE_COUNT = GRID * GRID
IMAGE_SIZE = GRID * TILE
TRANSFORMERS_VERSION = "4.57.1"
MODEL_LOAD_LOCK = threading.Lock()

EXPECTED_QAP_CONFIG: dict[str, Any] = {
    "boundary_weight": 0.05,
    "initial_weight": 0.75,
    "iterations": 25,
    "noise_scale": 1.0,
    "noisy_components": 3,
    "refine_swaps": 8,
    "restarts": 2,
    "score": "l1w4",
    "seeds": ["softcycle_l1_k8"],
}

DEFAULT_CONFIG: dict[str, Any] = {
    "authoritative_report_name": "qap_l1w4_boundary_real16.json",
    "baseline_label": "qap_softcycle_l1_k8",
    "expected_baseline_manifest_sha256": (
        "2a7cc81a95ea03fe339f37032dcb29e5139e386d402e8d1522e7567b94ba4020"
    ),
    "expected_authoritative_baseline_ssim": 0.18281991502795386,
    "baseline_ssim_reproduction_tolerance": 1e-5,
    "expected_denoiser_sha256": (
        "77a2d8607c9bc6b80e7aa99ed03329c1fae6bf94ee1d9a654241ab93a05cc734"
    ),
    "denoiser_name": "selected_tilenaf_synth_50k.pt",
    "model_id": "facebook/vit-mae-base",
    "model_revision": "25b184bea5538bf5c4c852c79d221195fdd2778d",
    "transformers_version": TRANSFORMERS_VERSION,
    "cache_dir": "/tmp/huggingface",
    "seed": 20260711,
    "num_masks": 4,
    "mask_ratio": 0.75,
    "candidate_batch_size": 8,
    "denoiser_batch_size": 512,
    "max_devices": 2,
    "initial_population": 64,
    "max_candidates_per_source": 192,
    "beam_size": 8,
    "max_generation_attempts": 6000,
    "generation_max_seam_loss_fraction": 0.04,
    "selection_max_seam_loss_fraction": 0.02,
    "energy_min_relative_improvement": 0.001,
    "energy_min_mask_win_fraction": 0.75,
    "block_sizes": [2, 3, 4, 6, 8],
    "exact_block_sizes": [4, 8],
    "competitive_target_band_ssim": 0.005,
    "promotion_min_mean_ssim_gain": 0.01,
    "promotion_min_win_rate": 0.6875,
    "promotion_max_mean_seam_loss_fraction": 0.01,
    "promotion_max_source_seam_loss_fraction": 0.02,
    "promotion_min_competitive_spearman": 0.2,
    "promotion_min_competitive_pairwise": 0.6,
    "promotion_min_evaluable_sources": 16,
}


@dataclass
class SearchCandidate:
    candidate_id: str
    layout_sha256: str
    position_to_slot: np.ndarray
    generation: int
    operator: str
    parent_ids: list[str]
    parameters: dict[str, Any]
    changed_positions: int
    seam_cost: float
    seam_ratio: float
    mae_error_by_mask: list[float] = field(default_factory=list)
    mae_error_mean: float | None = None
    mae_error_std: float | None = None

    def frozen_record(self) -> dict[str, Any]:
        if self.mae_error_mean is None or not self.mae_error_by_mask:
            raise RuntimeError(f"candidate has no frozen MAE energy: {self.candidate_id}")
        return {
            "candidate_id": self.candidate_id,
            "layout_sha256": self.layout_sha256,
            "position_to_slot": self.position_to_slot.tolist(),
            "generation": self.generation,
            "operator": self.operator,
            "parent_ids": self.parent_ids,
            "parameters": self.parameters,
            "changed_positions": self.changed_positions,
            "changed_fraction": self.changed_positions / TILE_COUNT,
            "seam_cost": self.seam_cost,
            "seam_ratio": self.seam_ratio,
            "seam_loss_fraction": self.seam_ratio - 1.0,
            "mae_error_by_mask": self.mae_error_by_mask,
            "mae_error_mean": self.mae_error_mean,
            "mae_error_std": self.mae_error_std,
            "naturalness_score": -self.mae_error_mean,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("search_config.json")),
    )
    parser.add_argument(
        "--frozen-output",
        default=str(WORKING / "mae_search_frozen.json"),
    )
    parser.add_argument(
        "--output",
        default=str(WORKING / "mae_search_gate_report.json"),
    )
    parser.add_argument("--validate-config-only", action="store_true")
    parser.add_argument("--synthetic-test", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    # Kaggle script kernels upload only ``code_file``.  Keep the checked-in
    # JSON override for local/repro runs, but fall back to the identical
    # embedded defaults when the companion file is not present in /kaggle/src.
    supplied = (
        json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    )
    if not isinstance(supplied, dict):
        raise ValueError("search config must be a JSON object")
    unknown = sorted(set(supplied) - set(DEFAULT_CONFIG))
    if unknown:
        raise ValueError(f"unknown search config fields: {unknown}")
    config = {**DEFAULT_CONFIG, **supplied}
    integer_positive = (
        "num_masks",
        "candidate_batch_size",
        "denoiser_batch_size",
        "max_devices",
        "initial_population",
        "max_candidates_per_source",
        "beam_size",
        "max_generation_attempts",
        "promotion_min_evaluable_sources",
    )
    for name in integer_positive:
        if isinstance(config[name], bool) or int(config[name]) <= 0:
            raise ValueError(f"{name} must be a positive integer")
        config[name] = int(config[name])
    if not 2 <= config["num_masks"] <= 4:
        raise ValueError("num_masks must remain in the bounded [2, 4] gate range")
    if config["max_candidates_per_source"] > 256:
        raise ValueError("max_candidates_per_source must not exceed 256")
    if config["initial_population"] > config["max_candidates_per_source"]:
        raise ValueError("initial_population exceeds max_candidates_per_source")
    if config["beam_size"] > config["initial_population"]:
        raise ValueError("beam_size exceeds initial_population")
    if not 0.0 < float(config["mask_ratio"]) < 1.0:
        raise ValueError("mask_ratio must lie in (0, 1)")
    fractional = (
        "generation_max_seam_loss_fraction",
        "selection_max_seam_loss_fraction",
        "energy_min_relative_improvement",
        "energy_min_mask_win_fraction",
        "promotion_min_win_rate",
        "promotion_max_mean_seam_loss_fraction",
        "promotion_max_source_seam_loss_fraction",
    )
    for name in fractional:
        if not 0.0 <= float(config[name]) <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1]")
    if float(config["selection_max_seam_loss_fraction"]) > float(
        config["generation_max_seam_loss_fraction"]
    ):
        raise ValueError("selection seam guard must not exceed generation guard")
    for name in (
        "promotion_min_competitive_spearman",
        "promotion_min_competitive_pairwise",
    ):
        if not -1.0 <= float(config[name]) <= 1.0:
            raise ValueError(f"{name} must lie in [-1, 1]")
    for name in (
        "baseline_ssim_reproduction_tolerance",
        "competitive_target_band_ssim",
        "promotion_min_mean_ssim_gain",
    ):
        if float(config[name]) < 0.0:
            raise ValueError(f"{name} must be non-negative")
    for name in ("block_sizes", "exact_block_sizes"):
        values = [int(value) for value in config[name]]
        if not values or any(value < 1 or value > 12 for value in values):
            raise ValueError(f"{name} must contain sizes in [1, 12]")
        config[name] = sorted(set(values))
    if not set(config["exact_block_sizes"]) <= set(config["block_sizes"]):
        raise ValueError("exact_block_sizes must be a subset of block_sizes")
    if config["transformers_version"] != TRANSFORMERS_VERSION:
        raise ValueError(f"transformers_version must be pinned to {TRANSFORMERS_VERSION}")
    for name in ("expected_baseline_manifest_sha256", "expected_denoiser_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", str(config[name])) is None:
            raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_layout(layout: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(layout, dtype="<i4").tobytes()).hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_layout(raw: Any, *, name: str = "position_to_slot") -> np.ndarray:
    values = np.asarray(raw)
    if values.shape != (TILE_COUNT,):
        raise ValueError(f"{name} must have shape {(TILE_COUNT,)}, got {values.shape}")
    if not np.issubdtype(values.dtype, np.integer):
        rounded = np.rint(values)
        if not np.array_equal(values, rounded):
            raise TypeError(f"{name} must contain integers")
        values = rounded
    values = values.astype(np.int32, copy=False)
    if not np.array_equal(np.sort(values), np.arange(TILE_COUNT, dtype=np.int32)):
        raise ValueError(f"{name} is not a permutation of [0, {TILE_COUNT - 1}]")
    return values.copy()


def split_tiles(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or image.dtype != np.uint8:
        raise ValueError("image must be uint8 RGB 480x480")
    return (
        image.reshape(GRID, TILE, GRID, TILE, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(TILE_COUNT, TILE, TILE, 3)
    )


def merge_tiles(tiles: np.ndarray) -> np.ndarray:
    tiles = np.asarray(tiles)
    if tiles.shape != (TILE_COUNT, TILE, TILE, 3):
        raise ValueError(f"unexpected tile shape: {tiles.shape}")
    return (
        tiles.reshape(GRID, GRID, TILE, TILE, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(IMAGE_SIZE, IMAGE_SIZE, 3)
    )


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        values = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if values.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError(f"unexpected image shape {values.shape}: {path}")
    return values


def load_phase_a_json(path: Path) -> tuple[dict[str, Any], int]:
    """Drop real-evaluation fields before the report object reaches Phase A."""

    dropped = 0
    exact_forbidden = {
        "mae",
        "macro",
        "psnr",
        "selector_scores",
    }

    def object_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        nonlocal dropped
        result: dict[str, Any] = {}
        for key, value in pairs:
            lowered = key.lower()
            if lowered in exact_forbidden or lowered.endswith("_ssim"):
                dropped += 1
                continue
            if key in result:
                raise ValueError(f"duplicate JSON key: {key!r}")
            result[key] = value
        return result

    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=object_hook,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"report root is not an object: {path}")
    return payload, dropped


def extract_baseline_manifest(
    payload: dict[str, Any], *, baseline_label: str
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if payload.get("qap") != EXPECTED_QAP_CONFIG:
        raise ValueError(f"authoritative QAP config mismatch: {payload.get('qap')!r}")
    source_names = payload.get("source_names")
    sources = payload.get("sources")
    if not isinstance(source_names, list) or not isinstance(sources, list):
        raise ValueError("authoritative report lacks source_names/sources")
    if len(source_names) != 16 or len(sources) != 16:
        raise ValueError("authoritative gate requires exactly real16")
    layouts: list[dict[str, Any]] = []
    by_source: dict[str, np.ndarray] = {}
    for record in sources:
        if not isinstance(record, dict) or not isinstance(record.get("variants"), dict):
            raise ValueError("malformed authoritative source record")
        source = record.get("source")
        if not isinstance(source, str) or source in by_source:
            raise ValueError(f"invalid/duplicate authoritative source: {source!r}")
        variants = record["variants"]
        denoised = variants.get(f"{baseline_label}__denoised_render")
        raw = variants.get(f"{baseline_label}__raw_render")
        if not isinstance(denoised, dict) or not isinstance(raw, dict):
            raise ValueError(f"baseline variants missing for {source}")
        denoised_layout = validate_layout(
            denoised.get("position_to_slot"), name=f"{source}:denoised baseline"
        )
        raw_layout = validate_layout(
            raw.get("position_to_slot"), name=f"{source}:raw baseline"
        )
        if not np.array_equal(denoised_layout, raw_layout):
            raise ValueError(f"raw and denoised baseline layouts disagree for {source}")
        by_source[source] = denoised_layout
        layouts.append({"source": source, "position_to_slot": denoised_layout.tolist()})
    if source_names != [record["source"] for record in layouts]:
        raise ValueError("source_names order differs from sources order")
    manifest = {
        "qap": payload["qap"],
        "source_names": source_names,
        "baseline_label": baseline_label,
        "layouts": layouts,
    }
    return manifest, by_source


def find_authoritative_baseline(
    config: dict[str, Any],
) -> tuple[Path, dict[str, np.ndarray], dict[str, Any]]:
    paths = sorted(INPUT.glob(f"**/{config['authoritative_report_name']}"))
    if not paths:
        raise RuntimeError(
            f"missing authoritative report {config['authoritative_report_name']} under {INPUT}"
        )
    matches: list[tuple[Path, dict[str, np.ndarray], dict[str, Any], int]] = []
    observed: dict[str, str] = {}
    for path in paths:
        try:
            payload, dropped = load_phase_a_json(path)
            manifest, baselines = extract_baseline_manifest(
                payload, baseline_label=str(config["baseline_label"])
            )
            digest = sha256_json(manifest)
            observed[str(path)] = digest
            if digest == config["expected_baseline_manifest_sha256"]:
                matches.append((path, baselines, manifest, dropped))
        except Exception as exc:
            observed[str(path)] = f"invalid:{exc}"
    if not matches:
        raise RuntimeError(
            "no authoritative report matched the frozen layout manifest; "
            f"observed={observed}"
        )
    matches.sort(key=lambda item: (len(item[0].parts), str(item[0])))
    path, baselines, manifest, dropped = matches[0]
    provenance = {
        "selected_path": str(path),
        "matching_paths": [str(item[0]) for item in matches],
        "baseline_manifest_sha256": sha256_json(manifest),
        "baseline_label": config["baseline_label"],
        "qap": manifest["qap"],
        "source_names": manifest["source_names"],
        "phase_a_dropped_evaluation_fields": dropped,
    }
    return path, baselines, provenance


def find_data_root(sources: Iterable[str]) -> Path:
    source_names = sorted(set(sources))
    roots = sorted(
        {
            path.parent.parent
            for path in INPUT.glob("**/train/inputs")
            if path.is_dir() and all((path / source).is_file() for source in source_names)
        }
    )
    if len(roots) != 1:
        raise RuntimeError(f"expected one puzzle data root, found {roots}")
    return roots[0]


def find_denoiser(config: dict[str, Any]) -> tuple[Path, str]:
    paths = sorted(INPUT.glob(f"**/{config['denoiser_name']}"))
    matching = [(path, sha256_file(path)) for path in paths]
    matching = [item for item in matching if item[1] == config["expected_denoiser_sha256"]]
    if not matching:
        raise RuntimeError("no denoiser checkpoint matched the pinned SHA-256")
    matching.sort(key=lambda item: (len(item[0].parts), str(item[0])))
    return matching[0]


def package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def ensure_transformers() -> dict[str, Any]:
    before = package_version("transformers")
    installed = False
    if before != TRANSFORMERS_VERSION:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--disable-pip-version-check",
                "--no-cache-dir",
                f"transformers=={TRANSFORMERS_VERSION}",
            ],
            check=True,
        )
        installed = True
    after = package_version("transformers")
    if after != TRANSFORMERS_VERSION:
        raise RuntimeError(f"expected transformers {TRANSFORMERS_VERSION}, found {after}")
    return {
        "transformers_before": before,
        "transformers": after,
        "installed_by_runner": installed,
        "huggingface_hub": package_version("huggingface-hub"),
        "safetensors": package_version("safetensors"),
        "numpy": np.__version__,
        "pillow": package_version("Pillow"),
        "scikit_image": package_version("scikit-image"),
    }


def download_model_snapshot(config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    from huggingface_hub import snapshot_download

    cache_dir = Path(str(config["cache_dir"]))
    cache_dir.mkdir(parents=True, exist_ok=True)
    snapshot = Path(
        snapshot_download(
            repo_id=str(config["model_id"]),
            revision=str(config["model_revision"]),
            cache_dir=str(cache_dir),
            allow_patterns=["config.json", "preprocessor_config.json", "*.safetensors"],
        )
    )
    required = [snapshot / "config.json", snapshot / "preprocessor_config.json"]
    weights = sorted(snapshot.glob("*.safetensors"))
    if not all(path.is_file() for path in required) or not weights:
        raise RuntimeError(f"incomplete MAE snapshot: {snapshot}")
    metadata = {
        "model_id": config["model_id"],
        "requested_revision": config["model_revision"],
        "resolved_revision": snapshot.name,
        "configuration_files": [
            {"name": path.name, "sha256": sha256_file(path)} for path in required
        ],
        "weight_files": [
            {"name": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in weights
        ],
        "low_cpu_mem_usage": False,
    }
    if snapshot.name != config["model_revision"]:
        raise RuntimeError(
            f"resolved MAE revision {snapshot.name} != pinned {config['model_revision']}"
        )
    return snapshot, metadata


def probe_hardware(max_devices: int) -> dict[str, Any]:
    subprocess.run(["nvidia-smi"], check=False)
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("MAE search gate requires CUDA")
    count = torch.cuda.device_count()
    result: dict[str, Any] = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": str(torch.version.cuda),
        "device_count": count,
        "devices": [],
        "capabilities": [],
        "arch_list": torch.cuda.get_arch_list(),
        "matmul_means": [],
    }
    for index in range(count):
        result["devices"].append(torch.cuda.get_device_name(index))
        result["capabilities"].append(list(torch.cuda.get_device_capability(index)))
        left = torch.randn(64, 64, device=f"cuda:{index}")
        right = torch.randn(64, 64, device=f"cuda:{index}")
        result["matmul_means"].append(float((left @ right).mean().item()))
    result["devices_used"] = min(count, max_devices)
    return result


def build_tilenaf_model() -> Any:
    """Build the exact promoted TileNAF architecture without external imports."""

    import torch
    from torch import nn

    class LayerNorm2d(nn.Module):
        def __init__(self, channels: int, eps: float = 1e-6) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
            self.eps = eps

        def forward(self, x: Any) -> Any:
            mean = x.mean(dim=1, keepdim=True)
            variance = (x - mean).square().mean(dim=1, keepdim=True)
            return (x - mean) * torch.rsqrt(variance + self.eps) * self.weight + self.bias

    class SimpleGate(nn.Module):
        def forward(self, x: Any) -> Any:
            left, right = x.chunk(2, dim=1)
            return left * right

    class DegradationEncoder(nn.Module):
        def __init__(self, code_dim: int = 32, parameter_dim: int = 5) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 24, 3, padding=1, padding_mode="reflect"),
                nn.GELU(),
                nn.Conv2d(24, 24, 3, stride=2, padding=1, groups=24, padding_mode="reflect"),
                nn.Conv2d(24, 48, 1),
                nn.GELU(),
                nn.Conv2d(48, 48, 3, stride=2, padding=1, groups=48, padding_mode="reflect"),
                nn.Conv2d(48, 64, 1),
                nn.GELU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.to_code = nn.Sequential(nn.Flatten(), nn.Linear(64, code_dim), nn.Tanh())
            self.to_parameters = nn.Linear(code_dim, parameter_dim)

        def forward(self, x: Any) -> tuple[Any, Any]:
            code = self.to_code(self.features(x))
            return code, self.to_parameters(code)

    class FiLMNAFBlock(nn.Module):
        def __init__(
            self,
            channels: int,
            code_dim: int,
            expansion: int = 2,
            ffn_expansion: int = 2,
            dilation: int = 1,
        ) -> None:
            super().__init__()
            expanded = channels * expansion
            ffn_channels = channels * ffn_expansion
            self.norm1 = LayerNorm2d(channels)
            self.film1 = nn.Linear(code_dim, channels * 2)
            self.in_conv = nn.Conv2d(channels, expanded, 1)
            self.depthwise = nn.Conv2d(
                expanded,
                expanded,
                3,
                padding=dilation,
                dilation=dilation,
                groups=expanded,
                padding_mode="reflect",
            )
            self.gate1 = SimpleGate()
            gated = expanded // 2
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Conv2d(gated, gated, 1)
            )
            self.out_conv = nn.Conv2d(gated, channels, 1)
            self.dropout1 = nn.Identity()
            self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
            self.norm2 = LayerNorm2d(channels)
            self.film2 = nn.Linear(code_dim, channels * 2)
            self.ffn_in = nn.Conv2d(channels, ffn_channels, 1)
            self.gate2 = SimpleGate()
            self.ffn_out = nn.Conv2d(ffn_channels // 2, channels, 1)
            self.dropout2 = nn.Identity()
            self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

        @staticmethod
        def film(x: Any, projection: Any, code: Any) -> Any:
            scale, bias = projection(code).chunk(2, dim=1)
            scale = 0.25 * torch.tanh(scale)[:, :, None, None]
            bias = 0.25 * torch.tanh(bias)[:, :, None, None]
            return x * (1.0 + scale) + bias

        def forward(self, x: Any, code: Any) -> Any:
            y = self.film(self.norm1(x), self.film1, code)
            y = self.gate1(self.depthwise(self.in_conv(y)))
            y = y * self.channel_attention(y)
            x = x + self.dropout1(self.out_conv(y)) * self.beta
            y = self.film(self.norm2(x), self.film2, code)
            y = self.ffn_out(self.gate2(self.ffn_in(y)))
            return x + self.dropout2(y) * self.gamma

    class BlockStack(nn.Module):
        def __init__(
            self, channels: int, count: int, code_dim: int, dilations: Sequence[int] = (1,)
        ) -> None:
            super().__init__()
            self.blocks = nn.ModuleList(
                [
                    FiLMNAFBlock(
                        channels,
                        code_dim,
                        dilation=int(dilations[index % len(dilations)]),
                    )
                    for index in range(count)
                ]
            )

        def forward(self, x: Any, code: Any) -> Any:
            for block in self.blocks:
                x = block(x, code)
            return x

    class TileNAFNet(nn.Module):
        def __init__(self, width: int = 48, code_dim: int = 32) -> None:
            super().__init__()
            self.degradation_encoder = DegradationEncoder(code_dim)
            self.stem = nn.Conv2d(3, width, 3, padding=1, padding_mode="reflect")
            self.encoder1 = BlockStack(width, 2, code_dim)
            self.down1 = nn.Conv2d(width, width * 2, 2, stride=2)
            self.encoder2 = BlockStack(width * 2, 4, code_dim)
            self.down2 = nn.Conv2d(width * 2, width * 4, 2, stride=2)
            self.middle = BlockStack(width * 4, 8, code_dim)
            self.up2 = nn.Sequential(nn.Conv2d(width * 4, width * 8, 1), nn.PixelShuffle(2))
            self.decoder2 = BlockStack(width * 2, 4, code_dim)
            self.up1 = nn.Sequential(nn.Conv2d(width * 2, width * 4, 1), nn.PixelShuffle(2))
            self.decoder1 = BlockStack(width, 2, code_dim)
            self.tail = nn.Conv2d(width, 3, 3, padding=1, padding_mode="reflect")
            nn.init.zeros_(self.tail.weight)
            nn.init.zeros_(self.tail.bias)

        def forward(self, x: Any, return_aux: bool = False) -> Any:
            if tuple(x.shape[-2:]) != (TILE, TILE):
                raise ValueError(f"TileNAF requires 20x20 tiles, got {tuple(x.shape[-2:])}")
            code, parameter_prediction = self.degradation_encoder(x)
            base = self.stem(x)
            skip1 = self.encoder1(base, code)
            skip2 = self.encoder2(self.down1(skip1), code)
            features = self.middle(self.down2(skip2), code)
            features = self.decoder2(self.up2(features) + skip2, code)
            features = self.decoder1(self.up1(features) + skip1, code)
            restored = x + self.tail(features)
            return (restored, parameter_prediction) if return_aux else restored

    return TileNAFNet()


def load_denoiser(checkpoint_path: Path, device: Any) -> tuple[Any, dict[str, Any]]:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("denoiser checkpoint root is not a dictionary")
    if checkpoint.get("model_name") != "tile-naf":
        raise ValueError(f"unexpected denoiser model: {checkpoint.get('model_name')!r}")
    if "ema_state" not in checkpoint:
        raise ValueError("denoiser checkpoint lacks ema_state")
    model = build_tilenaf_model()
    model.load_state_dict(checkpoint["ema_state"], strict=True)
    model.to(device).eval()
    metadata = {
        "model_name": checkpoint.get("model_name"),
        "state": "ema_state",
        "step": checkpoint.get("step"),
        "best_ssim": checkpoint.get("best_ssim"),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }
    del checkpoint
    return model, metadata


def restore_tiles_uint8(model: Any, tiles: np.ndarray, device: Any, batch_size: int) -> np.ndarray:
    import torch

    restored_parts: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(tiles), batch_size):
            batch = torch.from_numpy(
                np.ascontiguousarray(tiles[start : start + batch_size].transpose(0, 3, 1, 2))
            ).float().div_(255.0).to(device)
            restored = model(batch)
            restored_parts.append(
                restored.detach()
                .float()
                .cpu()
                .mul(255.0)
                .round()
                .clamp(0, 255)
                .byte()
                .permute(0, 2, 3, 1)
                .numpy()
            )
    return np.concatenate(restored_parts)


def pairwise_l1(left: np.ndarray, right: np.ndarray, chunk_size: int = 32) -> np.ndarray:
    left_values = np.asarray(left, dtype=np.float32).reshape(len(left), -1)
    right_values = np.asarray(right, dtype=np.float32).reshape(len(right), -1)
    result = np.empty((len(left_values), len(right_values)), dtype=np.float32)
    for start in range(0, len(left_values), chunk_size):
        block = left_values[start : start + chunk_size]
        result[start : start + len(block)] = np.mean(
            np.abs(block[:, None, :] - right_values[None, :, :]),
            axis=2,
            dtype=np.float32,
        )
    return result


def l1w4_matrices(denoised_tiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = denoised_tiles.astype(np.float32) / 255.0
    right = pairwise_l1(values[:, :, -4:, :], values[:, :, :4, :])
    down = pairwise_l1(values[:, -4:, :, :], values[:, :4, :, :])
    np.fill_diagonal(right, np.inf)
    np.fill_diagonal(down, np.inf)
    return right, down


def seam_cost(layout: np.ndarray, right: np.ndarray, down: np.ndarray) -> float:
    grid = validate_layout(layout).reshape(GRID, GRID)
    horizontal = right[grid[:, :-1], grid[:, 1:]]
    vertical = down[grid[:-1, :], grid[1:, :]]
    value = 0.5 * (float(np.mean(horizontal)) + float(np.mean(vertical)))
    if not math.isfinite(value) or value <= 0.0:
        raise FloatingPointError(f"invalid seam cost: {value}")
    return value


def rectangles_overlap(
    first: tuple[int, int], second: tuple[int, int], height: int, width: int
) -> bool:
    first_row, first_column = first
    second_row, second_column = second
    return not (
        first_row + height <= second_row
        or second_row + height <= first_row
        or first_column + width <= second_column
        or second_column + width <= first_column
    )


def sample_non_overlapping_positions(
    rng: np.random.Generator,
    *,
    height: int,
    width: int,
    count: int,
    attempts: int = 100,
) -> list[tuple[int, int]] | None:
    positions: list[tuple[int, int]] = []
    for _ in range(attempts):
        candidate = (
            int(rng.integers(0, GRID - height + 1)),
            int(rng.integers(0, GRID - width + 1)),
        )
        if candidate in positions:
            continue
        if any(rectangles_overlap(candidate, other, height, width) for other in positions):
            continue
        positions.append(candidate)
        if len(positions) == count:
            return positions
    return None


def choose_size(rng: np.random.Generator, config: dict[str, Any]) -> int:
    return int(rng.choice(np.asarray(config["block_sizes"], dtype=np.int32)))


def block_swap_mutation(
    layout: np.ndarray,
    rng: np.random.Generator,
    config: dict[str, Any],
    *,
    exact_size: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    height = exact_size if exact_size is not None else choose_size(rng, config)
    width = exact_size if exact_size is not None else choose_size(rng, config)
    positions = sample_non_overlapping_positions(
        rng, height=height, width=width, count=2
    )
    if positions is None:
        return None
    grid = layout.reshape(GRID, GRID).copy()
    (first_row, first_column), (second_row, second_column) = positions
    first = grid[first_row : first_row + height, first_column : first_column + width].copy()
    second = grid[
        second_row : second_row + height, second_column : second_column + width
    ].copy()
    grid[first_row : first_row + height, first_column : first_column + width] = second
    grid[second_row : second_row + height, second_column : second_column + width] = first
    return grid.reshape(-1), {
        "height": height,
        "width": width,
        "positions": [list(position) for position in positions],
    }


def band_swap_mutation(
    layout: np.ndarray,
    rng: np.random.Generator,
    config: dict[str, Any],
    *,
    axis: int,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    width = min(choose_size(rng, config), 8)
    starts = np.arange(0, GRID - width + 1, dtype=np.int32)
    pairs = [
        (int(first), int(second))
        for first in starts
        for second in starts
        if first < second and first + width <= second
    ]
    if not pairs:
        return None
    first, second = pairs[int(rng.integers(0, len(pairs)))]
    grid = layout.reshape(GRID, GRID).copy()
    if axis == 0:
        saved = grid[first : first + width].copy()
        grid[first : first + width] = grid[second : second + width]
        grid[second : second + width] = saved
    else:
        saved = grid[:, first : first + width].copy()
        grid[:, first : first + width] = grid[:, second : second + width]
        grid[:, second : second + width] = saved
    return grid.reshape(-1), {"axis": axis, "band_width": width, "starts": [first, second]}


def cyclic_block_shift_mutation(
    layout: np.ndarray, rng: np.random.Generator, config: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    height = max(2, choose_size(rng, config))
    width = max(2, choose_size(rng, config))
    row = int(rng.integers(0, GRID - height + 1))
    column = int(rng.integers(0, GRID - width + 1))
    row_shift = int(rng.integers(0, height))
    column_shift = int(rng.integers(0, width))
    if row_shift == 0 and column_shift == 0:
        column_shift = 1
    grid = layout.reshape(GRID, GRID).copy()
    block = grid[row : row + height, column : column + width].copy()
    grid[row : row + height, column : column + width] = np.roll(
        block, shift=(row_shift, column_shift), axis=(0, 1)
    )
    return grid.reshape(-1), {
        "height": height,
        "width": width,
        "position": [row, column],
        "shift": [row_shift, column_shift],
    }


def cyclic_band_shift_mutation(
    layout: np.ndarray,
    rng: np.random.Generator,
    config: dict[str, Any],
    *,
    axis: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    band_width = min(choose_size(rng, config), 8)
    start = int(rng.integers(0, GRID - band_width + 1))
    shift = int(rng.integers(1, GRID))
    grid = layout.reshape(GRID, GRID).copy()
    if axis == 0:
        grid[start : start + band_width] = np.roll(
            grid[start : start + band_width], shift=shift, axis=1
        )
    else:
        grid[:, start : start + band_width] = np.roll(
            grid[:, start : start + band_width], shift=shift, axis=0
        )
    return grid.reshape(-1), {
        "axis": axis,
        "band_width": band_width,
        "start": start,
        "shift": shift,
    }


def component_translation_mutation(
    layout: np.ndarray, rng: np.random.Generator, config: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Cut/insert a rectangular component while shifting the intervening strip."""

    height = min(choose_size(rng, config), 8)
    width = min(choose_size(rng, config), 8)
    horizontal = bool(rng.integers(0, 2))
    grid = layout.reshape(GRID, GRID).copy()
    if horizontal:
        row = int(rng.integers(0, GRID - height + 1))
        valid_pairs = [
            (first, second)
            for first in range(GRID - width + 1)
            for second in range(GRID - width + 1)
            if first != second and (first + width <= second or second + width <= first)
        ]
        if not valid_pairs:
            return None
        source_column, destination_column = valid_pairs[
            int(rng.integers(0, len(valid_pairs)))
        ]
        lower = min(source_column, destination_column)
        upper = max(source_column, destination_column) + width
        region = grid[row : row + height, lower:upper].copy()
        grid[row : row + height, lower:upper] = np.roll(
            region, shift=destination_column - source_column, axis=1
        )
        return grid.reshape(-1), {
            "orientation": "horizontal",
            "height": height,
            "width": width,
            "row": row,
            "source": source_column,
            "destination": destination_column,
        }
    column = int(rng.integers(0, GRID - width + 1))
    valid_pairs = [
        (first, second)
        for first in range(GRID - height + 1)
        for second in range(GRID - height + 1)
        if first != second and (first + height <= second or second + height <= first)
    ]
    if not valid_pairs:
        return None
    source_row, destination_row = valid_pairs[int(rng.integers(0, len(valid_pairs)))]
    lower = min(source_row, destination_row)
    upper = max(source_row, destination_row) + height
    region = grid[lower:upper, column : column + width].copy()
    grid[lower:upper, column : column + width] = np.roll(
        region, shift=destination_row - source_row, axis=0
    )
    return grid.reshape(-1), {
        "orientation": "vertical",
        "height": height,
        "width": width,
        "column": column,
        "source": source_row,
        "destination": destination_row,
    }


def block_translation_cycle_mutation(
    layout: np.ndarray, rng: np.random.Generator, config: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]] | None:
    height = min(choose_size(rng, config), 6)
    width = min(choose_size(rng, config), 6)
    positions = sample_non_overlapping_positions(
        rng, height=height, width=width, count=3
    )
    if positions is None:
        return None
    grid = layout.reshape(GRID, GRID).copy()
    blocks = [
        grid[row : row + height, column : column + width].copy()
        for row, column in positions
    ]
    for destination_index, (row, column) in enumerate(positions):
        grid[row : row + height, column : column + width] = blocks[
            (destination_index - 1) % len(blocks)
        ]
    return grid.reshape(-1), {
        "height": height,
        "width": width,
        "positions": [list(position) for position in positions],
        "cycle": [2, 0, 1],
    }


def destroy_repair_mutation(
    layout: np.ndarray,
    rng: np.random.Generator,
    config: dict[str, Any],
    right: np.ndarray,
    down: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Destroy three distant equal blocks, then seam-repair their assignment."""

    repair_sizes = [size for size in config["block_sizes"] if size <= 4]
    size = int(rng.choice(np.asarray(repair_sizes, dtype=np.int32)))
    positions = sample_non_overlapping_positions(rng, height=size, width=size, count=3)
    if positions is None:
        return None
    original = layout.reshape(GRID, GRID)
    blocks = [
        original[row : row + size, column : column + size].copy()
        for row, column in positions
    ]
    scored: list[tuple[float, tuple[int, ...], np.ndarray]] = []
    for permutation in itertools.permutations(range(3)):
        if permutation == (0, 1, 2):
            continue
        grid = original.copy()
        for destination_index, (row, column) in enumerate(positions):
            grid[row : row + size, column : column + size] = blocks[
                permutation[destination_index]
            ]
        candidate = grid.reshape(-1)
        scored.append((seam_cost(candidate, right, down), permutation, candidate.copy()))
    if not scored:
        return None
    score, permutation, candidate = min(scored, key=lambda item: (item[0], item[1]))
    return candidate, {
        "size": size,
        "positions": [list(position) for position in positions],
        "repair_permutation": list(permutation),
        "repair_seam_cost": score,
    }


MUTATION_OPERATORS = (
    "block_swap",
    "block_swap_4",
    "block_swap_8",
    "row_band_swap",
    "column_band_swap",
    "cyclic_block_shift",
    "row_band_cyclic_shift",
    "column_band_cyclic_shift",
    "component_translation",
    "block_translation_cycle",
    "destroy_repair",
)


def mutate_layout(
    layout: np.ndarray,
    operator: str,
    rng: np.random.Generator,
    config: dict[str, Any],
    right: np.ndarray,
    down: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    if operator == "block_swap":
        result = block_swap_mutation(layout, rng, config)
    elif operator == "block_swap_4":
        result = block_swap_mutation(layout, rng, config, exact_size=4)
    elif operator == "block_swap_8":
        result = block_swap_mutation(layout, rng, config, exact_size=8)
    elif operator == "row_band_swap":
        result = band_swap_mutation(layout, rng, config, axis=0)
    elif operator == "column_band_swap":
        result = band_swap_mutation(layout, rng, config, axis=1)
    elif operator == "cyclic_block_shift":
        result = cyclic_block_shift_mutation(layout, rng, config)
    elif operator == "row_band_cyclic_shift":
        result = cyclic_band_shift_mutation(layout, rng, config, axis=0)
    elif operator == "column_band_cyclic_shift":
        result = cyclic_band_shift_mutation(layout, rng, config, axis=1)
    elif operator == "component_translation":
        result = component_translation_mutation(layout, rng, config)
    elif operator == "block_translation_cycle":
        result = block_translation_cycle_mutation(layout, rng, config)
    elif operator == "destroy_repair":
        result = destroy_repair_mutation(layout, rng, config, right, down)
    else:
        raise ValueError(f"unknown mutation operator: {operator}")
    if result is None:
        return None
    candidate, parameters = result
    candidate = validate_layout(candidate, name=f"{operator} result")
    if np.array_equal(candidate, layout):
        return None
    return candidate, parameters


class PopulationBuilder:
    def __init__(
        self,
        *,
        source: str,
        baseline: np.ndarray,
        right: np.ndarray,
        down: np.ndarray,
        config: dict[str, Any],
    ) -> None:
        self.source = source
        self.baseline = validate_layout(baseline, name=f"{source}:baseline")
        self.right = right
        self.down = down
        self.config = config
        self.baseline_seam_cost = seam_cost(self.baseline, right, down)
        self.candidates: dict[str, SearchCandidate] = {}
        self.rejections: dict[str, dict[str, int]] = {}
        self.attempts = 0
        added = self.add(
            self.baseline,
            generation=0,
            operator="authoritative_boundary_qap_baseline",
            parent_ids=[],
            parameters={"retained_unconditionally": True},
            enforce_generation_guard=False,
        )
        if not added:
            raise RuntimeError("failed to insert authoritative baseline")

    def reject(self, operator: str, reason: str) -> None:
        by_reason = self.rejections.setdefault(operator, {})
        by_reason[reason] = by_reason.get(reason, 0) + 1

    def add(
        self,
        layout: np.ndarray,
        *,
        generation: int,
        operator: str,
        parent_ids: list[str],
        parameters: dict[str, Any],
        enforce_generation_guard: bool = True,
    ) -> bool:
        values = validate_layout(layout, name=f"{self.source}:{operator}")
        digest = sha256_layout(values)
        if digest in self.candidates:
            self.reject(operator, "duplicate")
            return False
        cost = seam_cost(values, self.right, self.down)
        ratio = cost / self.baseline_seam_cost
        if enforce_generation_guard and ratio > 1.0 + float(
            self.config["generation_max_seam_loss_fraction"]
        ):
            self.reject(operator, "generation_seam_guard")
            return False
        changed = int(np.count_nonzero(values != self.baseline))
        candidate = SearchCandidate(
            candidate_id=digest,
            layout_sha256=digest,
            position_to_slot=values,
            generation=generation,
            operator=operator,
            parent_ids=list(parent_ids),
            parameters=parameters,
            changed_positions=changed,
            seam_cost=cost,
            seam_ratio=ratio,
        )
        self.candidates[digest] = candidate
        return True

    @property
    def baseline_candidate(self) -> SearchCandidate:
        return self.candidates[sha256_layout(self.baseline)]


def source_rng(seed: int, source: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{source}".encode("utf-8")).digest()
    derived = int.from_bytes(digest[:8], "little", signed=False)
    return np.random.default_rng(derived)


def generate_until(
    builder: PopulationBuilder,
    *,
    parents: Sequence[SearchCandidate],
    target_count: int,
    generation: int,
    rng: np.random.Generator,
) -> None:
    if not parents:
        raise ValueError("candidate generation requires at least one parent")
    operator_offset = generation * 3
    while (
        len(builder.candidates) < target_count
        and builder.attempts < int(builder.config["max_generation_attempts"])
    ):
        attempt = builder.attempts
        builder.attempts += 1
        operator = MUTATION_OPERATORS[(attempt + operator_offset) % len(MUTATION_OPERATORS)]
        parent = parents[attempt % len(parents)]
        result = mutate_layout(
            parent.position_to_slot,
            operator,
            rng,
            builder.config,
            builder.right,
            builder.down,
        )
        if result is None:
            builder.reject(operator, "operator_failed")
            continue
        layout, parameters = result
        builder.add(
            layout,
            generation=generation,
            operator=operator,
            parent_ids=[parent.candidate_id],
            parameters=parameters,
        )


def pareto_beam(
    candidates: Sequence[SearchCandidate], *, beam_size: int
) -> list[SearchCandidate]:
    if any(candidate.mae_error_mean is None for candidate in candidates):
        raise RuntimeError("cannot choose beam before every energy is frozen")
    by_energy = sorted(
        candidates,
        key=lambda item: (float(item.mae_error_mean), item.seam_ratio, item.candidate_id),
    )
    pareto: list[SearchCandidate] = []
    best_seam = math.inf
    for candidate in by_energy:
        if candidate.seam_ratio < best_seam - 1e-12:
            pareto.append(candidate)
            best_seam = candidate.seam_ratio
    ordered: list[SearchCandidate] = []
    seen: set[str] = set()

    def include(items: Iterable[SearchCandidate]) -> None:
        for item in items:
            if len(ordered) >= beam_size:
                return
            if item.candidate_id not in seen:
                ordered.append(item)
                seen.add(item.candidate_id)

    baselines = [item for item in candidates if item.operator.endswith("baseline")]
    include(baselines)
    include(pareto)
    include(by_energy)
    include(sorted(candidates, key=lambda item: (item.seam_ratio, item.candidate_id)))
    return ordered


def configure_mae_mask_ratio(model: Any, mask_ratio: float) -> None:
    model.config.mask_ratio = mask_ratio
    model.vit.config.mask_ratio = mask_ratio
    model.vit.embeddings.config.mask_ratio = mask_ratio


def score_mae_candidates(
    *,
    model: Any,
    processor: Any,
    device: Any,
    raw_tiles: np.ndarray,
    candidates: Sequence[SearchCandidate],
    fixed_noise: np.ndarray,
    config: dict[str, Any],
) -> float:
    import torch

    pending = [candidate for candidate in candidates if candidate.mae_error_mean is None]
    if not pending:
        return 0.0
    num_masks = int(config["num_masks"])
    norm_pix_loss = bool(model.config.norm_pix_loss)
    discrepancies: list[float] = []
    for start in range(0, len(pending), int(config["candidate_batch_size"])):
        batch = pending[start : start + int(config["candidate_batch_size"])]
        images = [
            Image.fromarray(merge_tiles(raw_tiles[candidate.position_to_slot]), mode="RGB")
            for candidate in batch
        ]
        processed = processor(images=images, return_tensors="pt")
        pixel_values = processed["pixel_values"].to(device=device, dtype=torch.float32)
        repeated = pixel_values.repeat_interleave(num_masks, dim=0)
        noise = torch.from_numpy(np.tile(fixed_noise, (len(batch), 1))).to(device)
        with torch.inference_mode():
            outputs = model(pixel_values=repeated, noise=noise)
            target = model.patchify(repeated).float()
            if norm_pix_loss:
                mean = target.mean(dim=-1, keepdim=True)
                variance = target.var(dim=-1, keepdim=True)
                target = (target - mean) / torch.sqrt(variance + 1e-6)
            patch_loss = (outputs.logits.float() - target).square().mean(dim=-1)
            per_sample = (
                (patch_loss * outputs.mask.float()).sum(dim=-1)
                / outputs.mask.float().sum(dim=-1).clamp_min(1.0)
            )
            discrepancies.append(
                abs(float(outputs.loss.float().item()) - float(per_sample.mean().item()))
            )
        errors = per_sample.reshape(len(batch), num_masks).cpu().numpy()
        for candidate, values in zip(batch, errors.tolist(), strict=True):
            candidate.mae_error_by_mask = [float(value) for value in values]
            candidate.mae_error_mean = float(np.mean(values, dtype=np.float64))
            candidate.mae_error_std = float(np.std(values, dtype=np.float64))
            if not math.isfinite(candidate.mae_error_mean):
                raise FloatingPointError(f"non-finite MAE error: {candidate.candidate_id}")
    return max(discrepancies) if discrepancies else 0.0


def conservative_selection(
    builder: PopulationBuilder, config: dict[str, Any]
) -> tuple[SearchCandidate, dict[str, Any]]:
    baseline = builder.baseline_candidate
    if baseline.mae_error_mean is None:
        raise RuntimeError("baseline has no energy")
    required_mask_wins = int(
        math.ceil(float(config["energy_min_mask_win_fraction"]) * int(config["num_masks"]))
    )
    eligible: list[tuple[SearchCandidate, float, int]] = []
    audits: list[dict[str, Any]] = []
    for candidate in builder.candidates.values():
        if candidate.candidate_id == baseline.candidate_id:
            continue
        if candidate.mae_error_mean is None:
            raise RuntimeError("candidate has no energy during selection")
        relative_improvement = (
            float(baseline.mae_error_mean) - float(candidate.mae_error_mean)
        ) / max(float(baseline.mae_error_mean), 1e-12)
        mask_wins = sum(
            candidate_error < baseline_error
            for candidate_error, baseline_error in zip(
                candidate.mae_error_by_mask,
                baseline.mae_error_by_mask,
                strict=True,
            )
        )
        passes_seam = candidate.seam_ratio <= 1.0 + float(
            config["selection_max_seam_loss_fraction"]
        )
        passes_energy = relative_improvement >= float(
            config["energy_min_relative_improvement"]
        )
        passes_masks = mask_wins >= required_mask_wins
        audits.append(
            {
                "candidate_id": candidate.candidate_id,
                "relative_mae_improvement": relative_improvement,
                "mask_wins": mask_wins,
                "passes_seam": passes_seam,
                "passes_energy_margin": passes_energy,
                "passes_mask_consensus": passes_masks,
            }
        )
        if passes_seam and passes_energy and passes_masks:
            eligible.append((candidate, relative_improvement, mask_wins))
    if eligible:
        selected, relative_improvement, mask_wins = min(
            eligible,
            key=lambda item: (
                float(item[0].mae_error_mean),
                item[0].seam_ratio,
                item[0].candidate_id,
            ),
        )
        mode = "conservative_mae_replacement"
    else:
        selected = baseline
        relative_improvement = 0.0
        mask_wins = int(config["num_masks"])
        mode = "baseline_retained_no_candidate_passed_all_input_only_guards"
    audit = {
        "mode": mode,
        "baseline_candidate_id": baseline.candidate_id,
        "selected_candidate_id": selected.candidate_id,
        "eligible_replacements": len(eligible),
        "required_mask_wins": required_mask_wins,
        "selected_relative_mae_improvement": relative_improvement,
        "selected_mask_wins": mask_wins,
        "candidate_audits": audits,
    }
    return selected, audit


def search_source_group(
    *,
    device_index: int,
    sources: list[str],
    baselines: dict[str, np.ndarray],
    data_root: Path,
    denoiser_path: Path,
    model_snapshot: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    import torch
    from transformers import AutoImageProcessor, ViTMAEForPreTraining

    torch.cuda.set_device(device_index)
    torch.manual_seed(int(config["seed"]))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(f"cuda:{device_index}")
    started = time.perf_counter()
    raw_tiles_by_source: dict[str, np.ndarray] = {}
    denoised_tiles_by_source: dict[str, np.ndarray] = {}

    denoiser, denoiser_metadata = load_denoiser(denoiser_path, device)
    for source in sources:
        raw_tiles = split_tiles(read_rgb(data_root / "train" / "inputs" / source))
        raw_tiles_by_source[source] = raw_tiles
        denoised_tiles_by_source[source] = restore_tiles_uint8(
            denoiser,
            raw_tiles,
            device,
            int(config["denoiser_batch_size"]),
        )
    del denoiser
    torch.cuda.empty_cache()

    # Transformers 4.57.1 mutates process-global loading state while resolving
    # low-memory/meta tensors.  Concurrent from_pretrained calls can therefore
    # leave one replica on ``meta`` even with low_cpu_mem_usage=False.  Create
    # and materialize the two replicas serially, then run their forward passes
    # concurrently on separate GPUs.
    with MODEL_LOAD_LOCK:
        processor = AutoImageProcessor.from_pretrained(
            str(model_snapshot), local_files_only=True, use_fast=False
        )
        model = ViTMAEForPreTraining.from_pretrained(
            str(model_snapshot),
            local_files_only=True,
            use_safetensors=True,
            low_cpu_mem_usage=False,
        )
        configure_mae_mask_ratio(model, float(config["mask_ratio"]))
        model.float().eval().to(device)
    processor_metadata = {
        "class": type(processor).__name__,
        "do_resize": getattr(processor, "do_resize", None),
        "size": getattr(processor, "size", None),
        "do_center_crop": getattr(processor, "do_center_crop", None),
        "crop_size": getattr(processor, "crop_size", None),
        "do_rescale": getattr(processor, "do_rescale", None),
        "rescale_factor": getattr(processor, "rescale_factor", None),
        "do_normalize": getattr(processor, "do_normalize", None),
        "image_mean": getattr(processor, "image_mean", None),
        "image_std": getattr(processor, "image_std", None),
    }
    patch_size = int(model.config.patch_size)
    image_size = int(model.config.image_size)
    norm_pix_loss = bool(model.config.norm_pix_loss)
    if image_size % patch_size:
        raise RuntimeError("MAE image size is not divisible by patch size")
    num_patches = (image_size // patch_size) ** 2
    fixed_noise = np.random.default_rng(int(config["seed"])).random(
        (int(config["num_masks"]), num_patches), dtype=np.float32
    )

    source_results: list[dict[str, Any]] = []
    consistency_differences: list[float] = []
    for source_index, source in enumerate(sources):
        source_started = time.perf_counter()
        right, down = l1w4_matrices(denoised_tiles_by_source[source])
        builder = PopulationBuilder(
            source=source,
            baseline=baselines[source],
            right=right,
            down=down,
            config=config,
        )
        rng = source_rng(int(config["seed"]), source)
        generate_until(
            builder,
            parents=[builder.baseline_candidate],
            target_count=int(config["initial_population"]),
            generation=1,
            rng=rng,
        )
        consistency_differences.append(
            score_mae_candidates(
                model=model,
                processor=processor,
                device=device,
                raw_tiles=raw_tiles_by_source[source],
                candidates=list(builder.candidates.values()),
                fixed_noise=fixed_noise,
                config=config,
            )
        )
        beam = pareto_beam(
            list(builder.candidates.values()), beam_size=int(config["beam_size"])
        )
        generate_until(
            builder,
            parents=beam,
            target_count=int(config["max_candidates_per_source"]),
            generation=2,
            rng=rng,
        )
        consistency_differences.append(
            score_mae_candidates(
                model=model,
                processor=processor,
                device=device,
                raw_tiles=raw_tiles_by_source[source],
                candidates=list(builder.candidates.values()),
                fixed_noise=fixed_noise,
                config=config,
            )
        )
        selected, selection_audit = conservative_selection(builder, config)
        operator_counts: dict[str, int] = {}
        generation_counts: dict[str, int] = {}
        for candidate in builder.candidates.values():
            operator_counts[candidate.operator] = operator_counts.get(candidate.operator, 0) + 1
            generation_key = str(candidate.generation)
            generation_counts[generation_key] = generation_counts.get(generation_key, 0) + 1
        source_results.append(
            {
                "source": source,
                "baseline_candidate_id": builder.baseline_candidate.candidate_id,
                "selected_candidate_id": selected.candidate_id,
                "baseline_seam_cost": builder.baseline_seam_cost,
                "selection": selection_audit,
                "operator_counts": operator_counts,
                "generation_counts": generation_counts,
                "generation_attempts": builder.attempts,
                "generation_rejections": builder.rejections,
                "beam_candidate_ids": [candidate.candidate_id for candidate in beam],
                "candidate_objects": list(builder.candidates.values()),
                "seconds": time.perf_counter() - source_started,
                "device": str(device),
            }
        )
        print(
            json.dumps(
                {
                    "event": "mae_search_source_complete",
                    "device": str(device),
                    "source": source,
                    "source_index": source_index + 1,
                    "source_count": len(sources),
                    "candidates": len(builder.candidates),
                    "selection_mode": selection_audit["mode"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    del model
    torch.cuda.empty_cache()
    return {
        "device": str(device),
        "sources": sources,
        "seconds": time.perf_counter() - started,
        "denoiser": denoiser_metadata,
        "processor": processor_metadata,
        "model_image_size": image_size,
        "model_patch_size": patch_size,
        "model_num_patches": num_patches,
        "model_norm_pix_loss": norm_pix_loss,
        "low_cpu_mem_usage": False,
        "max_forward_loss_consistency_abs": max(consistency_differences, default=0.0),
        "source_results": source_results,
        "denoised_tiles_by_source": denoised_tiles_by_source,
    }


def run_parallel_search(
    *,
    baselines: dict[str, np.ndarray],
    data_root: Path,
    denoiser_path: Path,
    model_snapshot: Path,
    device_count: int,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], list[dict[str, Any]]]:
    sources = list(baselines)
    groups = [sources[index::device_count] for index in range(device_count)]
    groups = [group for group in groups if group]
    workers: list[dict[str, Any]] = []
    if len(groups) == 1:
        workers.append(
            search_source_group(
                device_index=0,
                sources=groups[0],
                baselines=baselines,
                data_root=data_root,
                denoiser_path=denoiser_path,
                model_snapshot=model_snapshot,
                config=config,
            )
        )
    else:
        with ThreadPoolExecutor(max_workers=len(groups)) as executor:
            futures = {
                executor.submit(
                    search_source_group,
                    device_index=device_index,
                    sources=group,
                    baselines=baselines,
                    data_root=data_root,
                    denoiser_path=denoiser_path,
                    model_snapshot=model_snapshot,
                    config=config,
                ): device_index
                for device_index, group in enumerate(groups)
            }
            for future in as_completed(futures):
                workers.append(future.result())

    source_results: list[dict[str, Any]] = []
    denoised_tiles: dict[str, np.ndarray] = {}
    worker_metadata: list[dict[str, Any]] = []
    for worker in workers:
        source_results.extend(worker.pop("source_results"))
        cache = worker.pop("denoised_tiles_by_source")
        overlap = set(cache) & set(denoised_tiles)
        if overlap:
            raise RuntimeError(f"duplicate source cache across workers: {sorted(overlap)}")
        denoised_tiles.update(cache)
        worker_metadata.append(worker)
    source_results.sort(key=lambda item: sources.index(item["source"]))
    worker_metadata.sort(key=lambda item: item["device"])
    if [item["source"] for item in source_results] != sources:
        raise RuntimeError("parallel search source order/coverage mismatch")
    if set(denoised_tiles) != set(sources):
        raise RuntimeError("denoised cache source coverage mismatch")
    return source_results, denoised_tiles, worker_metadata


def freeze_search_artifact(
    *,
    output: Path,
    config: dict[str, Any],
    baseline_provenance: dict[str, Any],
    denoiser_metadata: dict[str, Any],
    model_metadata: dict[str, Any],
    hardware: dict[str, Any],
    dependencies: dict[str, Any],
    source_results: list[dict[str, Any]],
    worker_metadata: list[dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    frozen_sources: list[dict[str, Any]] = []
    total_candidates = 0
    for source_result in source_results:
        candidate_objects = source_result["candidate_objects"]
        candidates = sorted(
            (candidate.frozen_record() for candidate in candidate_objects),
            key=lambda item: (item["generation"], item["operator"], item["candidate_id"]),
        )
        total_candidates += len(candidates)
        frozen_sources.append(
            {
                key: value
                for key, value in source_result.items()
                if key != "candidate_objects"
            }
            | {"candidates": candidates}
        )
    payload = {
        "schema_version": 1,
        "kind": "input_only_frozen_mae_global_population_search",
        "contains_target_metrics": False,
        "anti_leakage": {
            "target_images_opened": False,
            "recorded_target_metrics_available_to_phase_a": False,
            "candidate_generation_uses_target": False,
            "candidate_selection_uses_target": False,
            "all_searched_layouts_and_selections_frozen_before_target": True,
            "phase": "A_input_only_search_complete",
        },
        "config": config,
        "authoritative_baseline": baseline_provenance,
        "denoiser": denoiser_metadata,
        "model": model_metadata,
        "hardware": hardware,
        "dependencies": dependencies,
        "workers": worker_metadata,
        "sources": frozen_sources,
        "total_candidates": total_candidates,
    }
    write_json_atomic(output, payload)
    return payload, sha256_file(output)


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * ((start + 1) + end)
        start = end
    return ranks


def spearman_correlation(first: Sequence[float], second: Sequence[float]) -> float | None:
    if len(first) < 3 or len(first) != len(second):
        return None
    first_ranks = average_ranks(np.asarray(first, dtype=np.float64))
    second_ranks = average_ranks(np.asarray(second, dtype=np.float64))
    if np.std(first_ranks) <= 0.0 or np.std(second_ranks) <= 0.0:
        return None
    return float(np.corrcoef(first_ranks, second_ranks)[0, 1])


def pairwise_ranking(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    correct = 0.0
    pairs = 0
    ties = 0
    for first_index in range(len(records)):
        for second_index in range(first_index + 1, len(records)):
            first = records[first_index]
            second = records[second_index]
            target_delta = float(first["denoised_target_ssim"]) - float(
                second["denoised_target_ssim"]
            )
            if abs(target_delta) <= 1e-9:
                continue
            score_delta = float(first["naturalness_score"]) - float(
                second["naturalness_score"]
            )
            pairs += 1
            if abs(score_delta) <= 1e-12:
                ties += 1
                correct += 0.5
            elif (target_delta > 0.0) == (score_delta > 0.0):
                correct += 1.0
    return {
        "accuracy": correct / pairs if pairs else None,
        "correct_weight": correct,
        "pairs": pairs,
        "energy_ties": ties,
    }


def rank_signal(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    spearman = spearman_correlation(
        [float(record["naturalness_score"]) for record in records],
        [float(record["denoised_target_ssim"]) for record in records],
    )
    return {
        "candidate_count": len(records),
        "spearman_naturalness_vs_ssim": spearman,
        "pairwise": pairwise_ranking(records),
    }


def mean_optional(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def read_recorded_authoritative_ssim(
    report_path: Path, *, baseline_label: str
) -> float:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    value = payload["macro"][f"{baseline_label}__denoised_render"][
        "predicted_layout_ssim"
    ]
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("recorded authoritative baseline SSIM is non-finite")
    return value


def evaluate_after_freeze(
    *,
    frozen_output: Path,
    expected_frozen_sha256: str,
    authoritative_report: Path,
    data_root: Path,
    denoised_tiles_by_source: dict[str, np.ndarray],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], list[str]]:
    """Phase B: the first function allowed to open clean real targets."""

    if sha256_file(frozen_output) != expected_frozen_sha256:
        raise RuntimeError("frozen search artifact changed before target evaluation")
    frozen = json.loads(frozen_output.read_text(encoding="utf-8"))
    if frozen.get("contains_target_metrics") is not False:
        raise RuntimeError("frozen search artifact has an invalid anti-leakage marker")
    if frozen.get("anti_leakage", {}).get(
        "all_searched_layouts_and_selections_frozen_before_target"
    ) is not True:
        raise RuntimeError("frozen search artifact lacks the required freeze marker")

    from skimage.metrics import structural_similarity

    targets = data_root / "train" / "targets"
    if not targets.is_dir():
        raise RuntimeError(f"target directory is unavailable in Phase B: {targets}")
    source_reports: list[dict[str, Any]] = []
    warnings: list[str] = []
    total_competitive_correct = 0.0
    total_competitive_pairs = 0
    for source_record in frozen["sources"]:
        source = source_record["source"]
        clean_target = read_rgb(targets / source)
        denoised_tiles = denoised_tiles_by_source[source]
        evaluated: list[dict[str, Any]] = []
        for record in source_record["candidates"]:
            layout = validate_layout(
                record["position_to_slot"], name=f"frozen:{source}:{record['candidate_id']}"
            )
            predicted = merge_tiles(denoised_tiles[layout])
            target_ssim = float(
                structural_similarity(
                    clean_target,
                    predicted,
                    channel_axis=2,
                    data_range=255,
                )
            )
            evaluated.append({**record, "denoised_target_ssim": target_ssim})
        by_id = {record["candidate_id"]: record for record in evaluated}
        baseline = by_id[source_record["baseline_candidate_id"]]
        selected = by_id[source_record["selected_candidate_id"]]
        oracle = max(
            evaluated,
            key=lambda item: (
                item["denoised_target_ssim"],
                item["naturalness_score"],
                item["candidate_id"],
            ),
        )
        baseline_ssim = float(baseline["denoised_target_ssim"])
        competitive = [
            record
            for record in evaluated
            if float(record["denoised_target_ssim"])
            >= baseline_ssim - float(config["competitive_target_band_ssim"])
        ]
        competitive_signal = rank_signal(competitive)
        all_signal = rank_signal(evaluated)
        total_competitive_correct += float(
            competitive_signal["pairwise"]["correct_weight"]
        )
        total_competitive_pairs += int(competitive_signal["pairwise"]["pairs"])
        selected_gain = float(selected["denoised_target_ssim"]) - baseline_ssim

        def compact(record: dict[str, Any]) -> dict[str, Any]:
            return {
                "candidate_id": record["candidate_id"],
                "operator": record["operator"],
                "generation": record["generation"],
                "seam_ratio": record["seam_ratio"],
                "mae_error_mean": record["mae_error_mean"],
                "denoised_target_ssim": record["denoised_target_ssim"],
            }

        source_reports.append(
            {
                "source": source,
                "candidate_count": len(evaluated),
                "baseline": compact(baseline),
                "selected_input_only": compact(selected),
                "target_oracle_post_hoc": compact(oracle),
                "selected_minus_baseline_ssim": selected_gain,
                "selected_wins": selected_gain > 1e-9,
                "selected_seam_loss_fraction": float(selected["seam_ratio"]) - 1.0,
                "all_searched_rank_signal": all_signal,
                "competitive_definition": (
                    "post-hoc candidates with denoised SSIM >= baseline SSIM - "
                    f"{config['competitive_target_band_ssim']}"
                ),
                "competitive_rank_signal": competitive_signal,
                "candidates_post_hoc": evaluated,
            }
        )

    recorded_baseline = read_recorded_authoritative_ssim(
        authoritative_report, baseline_label=str(config["baseline_label"])
    )
    computed_baseline = float(
        np.mean([record["baseline"]["denoised_target_ssim"] for record in source_reports])
    )
    expected_baseline = float(config["expected_authoritative_baseline_ssim"])
    baseline_reproduced = (
        abs(recorded_baseline - expected_baseline)
        <= float(config["baseline_ssim_reproduction_tolerance"])
        and abs(computed_baseline - expected_baseline)
        <= float(config["baseline_ssim_reproduction_tolerance"])
    )
    if not baseline_reproduced:
        warnings.append(
            "authoritative baseline reproduction failed: "
            f"expected={expected_baseline}, recorded={recorded_baseline}, "
            f"computed={computed_baseline}"
        )

    competitive_spearman = [
        record["competitive_rank_signal"]["spearman_naturalness_vs_ssim"]
        for record in source_reports
        if record["competitive_rank_signal"]["spearman_naturalness_vs_ssim"] is not None
    ]
    competitive_pairwise_sources = [
        record
        for record in source_reports
        if record["competitive_rank_signal"]["pairwise"]["accuracy"] is not None
    ]
    gains = [float(record["selected_minus_baseline_ssim"]) for record in source_reports]
    seam_losses = [float(record["selected_seam_loss_fraction"]) for record in source_reports]
    aggregate = {
        "sources": len(source_reports),
        "min_candidates_per_source": min(
            record["candidate_count"] for record in source_reports
        ),
        "mean_candidates_per_source": float(
            np.mean([record["candidate_count"] for record in source_reports])
        ),
        "mean_baseline_ssim": computed_baseline,
        "recorded_authoritative_baseline_ssim": recorded_baseline,
        "expected_authoritative_baseline_ssim": expected_baseline,
        "baseline_reproduced": baseline_reproduced,
        "mean_selected_ssim": float(
            np.mean(
                [record["selected_input_only"]["denoised_target_ssim"] for record in source_reports]
            )
        ),
        "mean_target_oracle_ssim": float(
            np.mean(
                [record["target_oracle_post_hoc"]["denoised_target_ssim"] for record in source_reports]
            )
        ),
        "mean_selected_minus_baseline_ssim": float(np.mean(gains)),
        "median_selected_minus_baseline_ssim": float(np.median(gains)),
        "selected_win_rate": float(np.mean(np.asarray(gains) > 1e-9)),
        "selected_wins": int(np.count_nonzero(np.asarray(gains) > 1e-9)),
        "mean_selected_seam_loss_fraction": float(np.mean(seam_losses)),
        "max_selected_seam_loss_fraction": float(np.max(seam_losses)),
        "competitive_evaluable_spearman_sources": len(competitive_spearman),
        "competitive_evaluable_pairwise_sources": len(competitive_pairwise_sources),
        "mean_competitive_spearman": mean_optional(competitive_spearman),
        "mean_competitive_pairwise_accuracy": mean_optional(
            record["competitive_rank_signal"]["pairwise"]["accuracy"]
            for record in competitive_pairwise_sources
        ),
        "micro_competitive_pairwise_accuracy": (
            total_competitive_correct / total_competitive_pairs
            if total_competitive_pairs
            else None
        ),
        "micro_competitive_pairs": total_competitive_pairs,
        "mean_competitive_candidate_count": float(
            np.mean(
                [record["competitive_rank_signal"]["candidate_count"] for record in source_reports]
            )
        ),
    }

    reasons: list[str] = []
    if not baseline_reproduced:
        reasons.append("authoritative boundary-QAP baseline was not reproduced")
    if len(source_reports) < int(config["promotion_min_evaluable_sources"]):
        reasons.append("insufficient real16 source coverage")
    if aggregate["min_candidates_per_source"] < int(config["initial_population"]):
        reasons.append("generation failed to produce the minimum falsification population")
    if aggregate["mean_selected_minus_baseline_ssim"] < float(
        config["promotion_min_mean_ssim_gain"]
    ):
        reasons.append("mean selected SSIM gain is below +0.01")
    if aggregate["selected_win_rate"] < float(config["promotion_min_win_rate"]):
        reasons.append("selected win rate is below the conservative threshold")
    if aggregate["mean_selected_seam_loss_fraction"] > float(
        config["promotion_max_mean_seam_loss_fraction"]
    ):
        reasons.append("mean selected seam loss exceeds the promotion guard")
    if aggregate["max_selected_seam_loss_fraction"] > float(
        config["promotion_max_source_seam_loss_fraction"]
    ):
        reasons.append("a source exceeds the hard selected seam-loss guard")
    if aggregate["competitive_evaluable_spearman_sources"] < int(
        config["promotion_min_evaluable_sources"]
    ):
        reasons.append("competitive Spearman is not evaluable on all real16 sources")
    if aggregate["competitive_evaluable_pairwise_sources"] < int(
        config["promotion_min_evaluable_sources"]
    ):
        reasons.append("competitive pairwise accuracy is not evaluable on all real16 sources")
    competitive_spearman_value = aggregate["mean_competitive_spearman"]
    if competitive_spearman_value is None or competitive_spearman_value < float(
        config["promotion_min_competitive_spearman"]
    ):
        reasons.append("MAE lacks Spearman rank signal inside the competitive set")
    competitive_pairwise_value = aggregate["micro_competitive_pairwise_accuracy"]
    if competitive_pairwise_value is None or competitive_pairwise_value < float(
        config["promotion_min_competitive_pairwise"]
    ):
        reasons.append("MAE lacks pairwise rank signal inside the competitive set")
    gate = {
        "passed": not reasons,
        "interpretation": (
            "promote only if MAE improves competitive boundary-QAP layouts; "
            "failure falsifies aggressive MAE-guided search on this mutation family"
        ),
        "thresholds": {
            key: config[key]
            for key in (
                "promotion_min_mean_ssim_gain",
                "promotion_min_win_rate",
                "promotion_max_mean_seam_loss_fraction",
                "promotion_max_source_seam_loss_fraction",
                "promotion_min_competitive_spearman",
                "promotion_min_competitive_pairwise",
                "promotion_min_evaluable_sources",
            )
        },
        "observed": aggregate,
        "failure_reasons": reasons,
    }
    return source_reports, aggregate, gate, warnings


def synthetic_generator_test(config: dict[str, Any]) -> dict[str, Any]:
    """Exercise every mutation and bounded population logic without ML/data."""

    rng = np.random.default_rng(12345)
    baseline = np.arange(TILE_COUNT, dtype=np.int32)
    right = rng.uniform(0.05, 0.25, size=(TILE_COUNT, TILE_COUNT)).astype(np.float32)
    down = rng.uniform(0.05, 0.25, size=(TILE_COUNT, TILE_COUNT)).astype(np.float32)
    np.fill_diagonal(right, np.inf)
    np.fill_diagonal(down, np.inf)
    operator_hashes: dict[str, str] = {}
    for operator in MUTATION_OPERATORS:
        result = None
        for _ in range(100):
            result = mutate_layout(baseline, operator, rng, config, right, down)
            if result is not None:
                break
        if result is None:
            raise AssertionError(f"synthetic operator failed repeatedly: {operator}")
        layout, _parameters = result
        validate_layout(layout, name=f"synthetic:{operator}")
        if np.array_equal(layout, baseline):
            raise AssertionError(f"synthetic operator returned identity: {operator}")
        operator_hashes[operator] = sha256_layout(layout)

    test_config = dict(config)
    test_config.update(
        {
            "initial_population": 24,
            "max_candidates_per_source": 64,
            "beam_size": 6,
            "max_generation_attempts": 3000,
            "generation_max_seam_loss_fraction": 1.0,
            "selection_max_seam_loss_fraction": 1.0,
            "energy_min_relative_improvement": 0.0001,
        }
    )
    builder = PopulationBuilder(
        source="synthetic.png",
        baseline=baseline,
        right=right,
        down=down,
        config=test_config,
    )
    generate_until(
        builder,
        parents=[builder.baseline_candidate],
        target_count=test_config["initial_population"],
        generation=1,
        rng=rng,
    )
    if len(builder.candidates) != test_config["initial_population"]:
        raise AssertionError("synthetic initial population did not reach its bound")
    for index, candidate in enumerate(builder.candidates.values()):
        base = 1.0 - index * 1e-4
        candidate.mae_error_by_mask = [base - mask * 1e-5 for mask in range(config["num_masks"])]
        candidate.mae_error_mean = float(np.mean(candidate.mae_error_by_mask))
        candidate.mae_error_std = float(np.std(candidate.mae_error_by_mask))
    beam = pareto_beam(list(builder.candidates.values()), beam_size=test_config["beam_size"])
    generate_until(
        builder,
        parents=beam,
        target_count=test_config["max_candidates_per_source"],
        generation=2,
        rng=rng,
    )
    if len(builder.candidates) != test_config["max_candidates_per_source"]:
        raise AssertionError("synthetic expanded population did not reach its bound")
    pending = [candidate for candidate in builder.candidates.values() if candidate.mae_error_mean is None]
    for index, candidate in enumerate(pending, start=1):
        base = 0.99 - index * 1e-5
        candidate.mae_error_by_mask = [base - mask * 1e-5 for mask in range(config["num_masks"])]
        candidate.mae_error_mean = float(np.mean(candidate.mae_error_by_mask))
        candidate.mae_error_std = float(np.std(candidate.mae_error_by_mask))
    selected, audit = conservative_selection(builder, test_config)
    validate_layout(selected.position_to_slot, name="synthetic selected")
    aligned_records = [
        {"naturalness_score": float(index), "denoised_target_ssim": float(index)}
        for index in range(5)
    ]
    signal = rank_signal(aligned_records)
    if not math.isclose(
        float(signal["spearman_naturalness_vs_ssim"]), 1.0, abs_tol=1e-12
    ):
        raise AssertionError("synthetic Spearman implementation failed")
    if signal["pairwise"]["accuracy"] != 1.0:
        raise AssertionError("synthetic pairwise implementation failed")
    return {
        "operators": operator_hashes,
        "population": len(builder.candidates),
        "beam": len(beam),
        "selected_mode": audit["mode"],
        "rank_signal": signal,
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    config_path = Path(args.config)
    config = load_config(config_path)
    if args.validate_config_only:
        print(json.dumps({"event": "mae_search_config_valid", "config": config}, sort_keys=True))
        return
    if args.synthetic_test:
        result = synthetic_generator_test(config)
        print(json.dumps({"event": "mae_search_synthetic_test_passed", **result}, sort_keys=True))
        return

    os.environ.setdefault("HF_HOME", str(config["cache_dir"]))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Phase A. The report decoder strips real-evaluation fields; only the exact
    # authoritative layouts and QAP input configuration survive discovery.
    authoritative_report, baselines, baseline_provenance = find_authoritative_baseline(
        config
    )
    data_root = find_data_root(baselines)
    denoiser_path, denoiser_sha256 = find_denoiser(config)
    hardware = probe_hardware(int(config["max_devices"]))
    dependencies = ensure_transformers()
    model_snapshot, model_metadata = download_model_snapshot(config)
    source_results, denoised_tiles, worker_metadata = run_parallel_search(
        baselines=baselines,
        data_root=data_root,
        denoiser_path=denoiser_path,
        model_snapshot=model_snapshot,
        device_count=int(hardware["devices_used"]),
        config=config,
    )
    max_consistency = max(
        (float(worker["max_forward_loss_consistency_abs"]) for worker in worker_metadata),
        default=0.0,
    )
    if max_consistency > 1e-4:
        raise RuntimeError(
            f"manual per-sample MAE loss disagrees with model scalar loss: {max_consistency}"
        )
    denoiser_metadata = {
        "path": str(denoiser_path),
        "checkpoint_sha256": denoiser_sha256,
        "expected_checkpoint_sha256": config["expected_denoiser_sha256"],
        "model_name": "tile-naf",
        "state": "ema_state",
        "input_only": True,
    }
    frozen_output = Path(args.frozen_output)
    frozen_payload, frozen_sha256 = freeze_search_artifact(
        output=frozen_output,
        config=config,
        baseline_provenance=baseline_provenance,
        denoiser_metadata=denoiser_metadata,
        model_metadata=model_metadata,
        hardware=hardware,
        dependencies=dependencies,
        source_results=source_results,
        worker_metadata=worker_metadata,
    )
    print(
        json.dumps(
            {
                "event": "mae_search_layouts_and_energies_frozen",
                "output": str(frozen_output),
                "sha256": frozen_sha256,
                "sources": len(frozen_payload["sources"]),
                "total_candidates": frozen_payload["total_candidates"],
            },
            sort_keys=True,
        ),
        flush=True,
    )

    # Phase B begins only after the frozen artifact has been written and hashed.
    source_reports, aggregate, promotion_gate, warnings = evaluate_after_freeze(
        frozen_output=frozen_output,
        expected_frozen_sha256=frozen_sha256,
        authoritative_report=authoritative_report,
        data_root=data_root,
        denoised_tiles_by_source=denoised_tiles,
        config=config,
    )
    operator_coverage: dict[str, int] = {}
    operator_rejections: dict[str, dict[str, int]] = {}
    for frozen_source in frozen_payload["sources"]:
        for operator, count in frozen_source["operator_counts"].items():
            operator_coverage[operator] = operator_coverage.get(operator, 0) + int(count)
        for operator, reasons in frozen_source["generation_rejections"].items():
            aggregate_reasons = operator_rejections.setdefault(operator, {})
            for reason, count in reasons.items():
                aggregate_reasons[reason] = aggregate_reasons.get(reason, 0) + int(count)

    final_report = {
        "schema_version": 1,
        "kind": "post_freeze_mae_search_falsification_gate",
        "anti_leakage": {
            "target_images_opened_only_after_frozen_sha256": True,
            "recorded_authoritative_ssim_read_only_after_freeze": True,
            "candidate_generation_uses_target": False,
            "candidate_selection_uses_target": False,
            "target_oracle_is_post_hoc_only": True,
            "frozen_artifact": str(frozen_output),
            "frozen_sha256": frozen_sha256,
        },
        "config_path": str(config_path) if config_path.is_file() else None,
        "config_source": "file" if config_path.is_file() else "embedded_defaults",
        "config_sha256": (
            sha256_file(config_path) if config_path.is_file() else sha256_json(config)
        ),
        "config": config,
        "authoritative_report": str(authoritative_report),
        "authoritative_baseline": baseline_provenance,
        "data_root": str(data_root),
        "denoiser": denoiser_metadata,
        "model": model_metadata,
        "hardware": hardware,
        "dependencies": dependencies,
        "operator_coverage": operator_coverage,
        "operator_rejections": operator_rejections,
        "sources": source_reports,
        "aggregate": aggregate,
        "promotion_gate": promotion_gate,
        "warnings": warnings,
        "seconds": time.perf_counter() - started,
    }
    output = Path(args.output)
    write_json_atomic(output, final_report)
    print(
        json.dumps(
            {
                "event": "mae_search_falsification_gate_complete",
                "output": str(output),
                "sha256": sha256_file(output),
                "passed": promotion_gate["passed"],
                "aggregate": aggregate,
                "failure_reasons": promotion_gate["failure_reasons"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

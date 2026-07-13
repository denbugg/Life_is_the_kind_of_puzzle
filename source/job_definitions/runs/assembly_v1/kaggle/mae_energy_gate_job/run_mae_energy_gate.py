#!/usr/bin/env python3
"""Leakage-separated frozen-MAE natural-image energy correlation gate.

Phase A discovers input-only layouts in real-assembly reports, reconstructs raw
mosaics, evaluates deterministic masked reconstruction errors, selects the
lowest-error candidate, and freezes those results to disk.  Phase B starts only
after the frozen artifact is hashed; it re-opens the reports and attaches their
already-recorded target SSIM values for correlation analysis.  Target images are
never opened by this program.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Iterable

import numpy as np
from PIL import Image


INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
GRID = 24
TILE = 20
TILE_COUNT = GRID * GRID
IMAGE_SIZE = GRID * TILE
TRANSFORMERS_VERSION = "4.57.1"
PHASE_A_FORBIDDEN_KEY_TOKENS = frozenset(
    {"lpips", "mae", "metric", "mse", "oracle", "psnr", "score", "ssim", "target"}
)


DEFAULT_CONFIG: dict[str, Any] = {
    "reports_root": "/kaggle/input",
    "data_root": None,
    "report_include_regex": r"(?i)(global_real4|qap[^/]*real16|real16[^/]*qap)",
    "report_exclude_regex": r"(?i)(wrapper|mae_energy|frozen)",
    "candidate_include_regex": (
        r"(?i)^(qap_|softcycle_l1_k8$|component_(?:l1|l1fusion|l1w4|cross_l1w4)"
        r"(?:_q50)?$|faithful_multi_phase_rl$|particle_beam_)"
    ),
    "candidate_exclude_regex": r"(?i)^identity$",
    "target_render_view": "denoised",
    "baseline_label_regex": r"(?i)^qap_softcycle_l1_k8$",
    "baseline_report_regex": r"(?i)qap_l1w4_boundary_real16",
    "baseline_fallback_label_regex": r"(?i)^qap_",
    "model_id": "facebook/vit-mae-base",
    "model_revision": "25b184bea5538bf5c4c852c79d221195fdd2778d",
    "cache_dir": "/tmp/huggingface",
    "mask_seed": 20260711,
    "num_masks": 8,
    "mask_ratio": 0.75,
    "candidate_batch_size": 4,
    "dtype": "float32",
    "max_devices": 2,
    "promotion_spearman": 0.30,
    "promotion_pairwise_accuracy": 0.65,
    "min_evaluable_sources": 4,
    "target_tie_tolerance": 1e-9,
    "energy_tie_tolerance": 1e-12,
}


@dataclass(frozen=True)
class CandidateAlias:
    report_path: Path
    report_display: str
    source: str
    label: str


@dataclass
class CandidateLayout:
    candidate_id: str
    source: str
    layout_sha256: str
    position_to_slot: np.ndarray
    aliases: list[CandidateAlias] = field(default_factory=list)

    @property
    def labels(self) -> list[str]:
        return sorted({alias.label for alias in self.aliases})

    @property
    def display_label(self) -> str:
        labels = self.labels
        return labels[0] if labels else self.candidate_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("gate_config.json")),
        help=(
            "optional JSON configuration path; when Kaggle uploads only the "
            "code_file and the default sidecar is absent, embedded defaults are used"
        ),
    )
    parser.add_argument(
        "--output",
        default=str(WORKING / "mae_energy_gate_report.json"),
        help="final target-attached analysis JSON",
    )
    parser.add_argument(
        "--frozen-output",
        default=str(WORKING / "mae_energy_frozen.json"),
        help="target-free Phase-A energy artifact",
    )
    parser.add_argument(
        "--validate-config-only",
        action="store_true",
        help="validate local JSON/regex configuration without Kaggle data or model imports",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    # Kaggle script kernels do not reliably preserve supplementary files from
    # the local job directory.  The checked-in sidecar remains convenient for
    # local review, while the embedded configuration keeps the remote job
    # autonomous.  An explicitly supplied non-default path still fails fast.
    default_sidecar = Path(__file__).with_name("gate_config.json")
    if not path.is_file():
        if path != default_sidecar:
            raise FileNotFoundError(f"configured gate JSON does not exist: {path}")
        supplied: Any = {}
    else:
        supplied = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(supplied, dict):
        raise ValueError("gate config must be a JSON object")
    unknown = sorted(set(supplied) - set(DEFAULT_CONFIG))
    if unknown:
        raise ValueError(f"unknown gate config fields: {unknown}")
    config = {**DEFAULT_CONFIG, **supplied}

    for name in (
        "report_include_regex",
        "report_exclude_regex",
        "candidate_include_regex",
        "candidate_exclude_regex",
        "baseline_label_regex",
        "baseline_report_regex",
        "baseline_fallback_label_regex",
    ):
        re.compile(str(config[name]))
    if config["target_render_view"] not in {"raw", "denoised", "best"}:
        raise ValueError("target_render_view must be raw, denoised, or best")
    if config["dtype"] not in {"float32", "float16"}:
        raise ValueError("dtype must be float32 or float16")
    if int(config["num_masks"]) < 2:
        raise ValueError("num_masks must be at least 2")
    if int(config["candidate_batch_size"]) <= 0:
        raise ValueError("candidate_batch_size must be positive")
    if int(config["max_devices"]) <= 0:
        raise ValueError("max_devices must be positive")
    if int(config["min_evaluable_sources"]) <= 0:
        raise ValueError("min_evaluable_sources must be positive")
    if not 0.0 < float(config["mask_ratio"]) < 1.0:
        raise ValueError("mask_ratio must lie strictly between zero and one")
    for name in ("promotion_spearman", "promotion_pairwise_accuracy"):
        if not -1.0 <= float(config[name]) <= 1.0:
            raise ValueError(f"{name} must lie in [-1, 1]")
    for name in ("target_tie_tolerance", "energy_tie_tolerance"):
        if float(config[name]) < 0.0:
            raise ValueError(f"{name} must be non-negative")
    return config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_layout(layout: np.ndarray) -> str:
    values = np.asarray(layout, dtype="<i4")
    return hashlib.sha256(values.tobytes()).hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def load_phase_a_report(path: Path) -> tuple[dict[str, Any], int]:
    """Load layouts while preventing report metrics from entering Phase A state.

    JSON decoding necessarily reads the report bytes, but the object-pairs hook
    discards every key containing an evaluation token before the surrounding
    object is constructed.  The returned object therefore cannot expose target
    scores to discovery, candidate selection, or energy evaluation.
    """

    dropped_fields = 0

    def strip_evaluation_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        nonlocal dropped_fields
        cleaned: dict[str, Any] = {}
        for key, value in pairs:
            tokens = set(re.findall(r"[a-z0-9]+", key.lower()))
            if tokens & PHASE_A_FORBIDDEN_KEY_TOKENS:
                dropped_fields += 1
                continue
            if key in cleaned:
                raise ValueError(f"duplicate JSON key in Phase-A report: {key!r}")
            cleaned[key] = value
        return cleaned

    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=strip_evaluation_fields,
    )
    if not isinstance(payload, dict):
        raise ValueError("report root must be a JSON object")
    return payload, dropped_fields


def sha256_json(value: Any) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def validate_layout(raw: Any, *, context: str) -> np.ndarray:
    values = np.asarray(raw)
    if values.shape != (TILE_COUNT,):
        raise ValueError(f"{context}: expected {TILE_COUNT} entries, got {values.shape}")
    if not np.issubdtype(values.dtype, np.integer):
        rounded = np.rint(values)
        if not np.array_equal(values, rounded):
            raise ValueError(f"{context}: layout contains non-integral values")
        values = rounded
    values = values.astype(np.int32, copy=False)
    if not np.array_equal(np.sort(values), np.arange(TILE_COUNT, dtype=np.int32)):
        raise ValueError(f"{context}: layout is not a permutation of [0, 575]")
    return values.copy()


def split_real_variant_key(key: str) -> tuple[str, str] | None:
    match = re.fullmatch(r"(.+)__(raw|denoised)_render", key)
    if match is None:
        return None
    return match.group(1), match.group(2)


def discover_layout_candidates(
    config: dict[str, Any],
) -> tuple[list[CandidateLayout], list[dict[str, Any]], list[str]]:
    """Phase A discovery: this function deliberately never reads score fields."""
    reports_root = Path(str(config["reports_root"]))
    include_report = re.compile(str(config["report_include_regex"]))
    exclude_report = re.compile(str(config["report_exclude_regex"]))
    include_candidate = re.compile(str(config["candidate_include_regex"]))
    exclude_candidate = re.compile(str(config["candidate_exclude_regex"]))

    report_paths = sorted(reports_root.rglob("*.json"))
    matched_paths = [
        path
        for path in report_paths
        if include_report.search(display_path(path, reports_root))
        and not exclude_report.search(display_path(path, reports_root))
    ]
    if not matched_paths:
        raise RuntimeError(
            "no report paths matched report_include_regex; "
            f"searched {reports_root} and saw {len(report_paths)} JSON files"
        )

    by_source_and_hash: dict[tuple[str, str], CandidateLayout] = {}
    selected_reports: list[dict[str, Any]] = []
    warnings: list[str] = []
    for report_path in matched_paths:
        try:
            payload, dropped_metric_fields = load_phase_a_report(report_path)
        except Exception as exc:  # malformed unrelated JSON should not kill discovery
            warnings.append(f"skipped unreadable JSON {report_path}: {exc}")
            continue
        sources = payload.get("sources") if isinstance(payload, dict) else None
        if not isinstance(sources, list):
            continue
        if not any(
            isinstance(source, dict) and isinstance(source.get("variants"), dict)
            for source in sources
        ):
            continue

        report_display = display_path(report_path, reports_root)
        report_candidate_count = 0
        report_layout_manifest: list[dict[str, Any]] = []
        for source_record in sources:
            if not isinstance(source_record, dict):
                continue
            source = source_record.get("source")
            variants = source_record.get("variants")
            if not isinstance(source, str) or not isinstance(variants, dict):
                continue
            layouts_by_label: dict[str, np.ndarray] = {}
            for variant_key, variant in sorted(variants.items()):
                parsed = split_real_variant_key(str(variant_key))
                if parsed is None or not isinstance(variant, dict):
                    continue
                label, _render_view = parsed
                if not include_candidate.search(label) or exclude_candidate.search(label):
                    continue
                if "position_to_slot" not in variant:
                    continue
                layout = validate_layout(
                    variant["position_to_slot"],
                    context=f"{report_display}:{source}:{variant_key}",
                )
                previous = layouts_by_label.get(label)
                if previous is not None and not np.array_equal(previous, layout):
                    raise RuntimeError(
                        f"raw/denoised variants disagree on layout: "
                        f"{report_display}:{source}:{label}"
                    )
                layouts_by_label[label] = layout

            for label, layout in sorted(layouts_by_label.items()):
                layout_hash = sha256_layout(layout)
                report_layout_manifest.append(
                    {
                        "source": source,
                        "label": label,
                        "layout_sha256": layout_hash,
                    }
                )
                key = (source, layout_hash)
                candidate = by_source_and_hash.get(key)
                if candidate is None:
                    candidate = CandidateLayout(
                        candidate_id=f"{source}:{layout_hash[:16]}",
                        source=source,
                        layout_sha256=layout_hash,
                        position_to_slot=layout,
                    )
                    by_source_and_hash[key] = candidate
                alias = CandidateAlias(
                    report_path=report_path,
                    report_display=report_display,
                    source=source,
                    label=label,
                )
                if alias not in candidate.aliases:
                    candidate.aliases.append(alias)
                report_candidate_count += 1

        if report_candidate_count:
            selected_reports.append(
                {
                    "path": report_display,
                    "layout_manifest_sha256": sha256_json(report_layout_manifest),
                    "kind": payload.get("kind"),
                    "source_names": payload.get("source_names"),
                    "candidate_aliases": report_candidate_count,
                    "phase_a_dropped_evaluation_fields": dropped_metric_fields,
                }
            )
        del payload

    candidates = sorted(
        by_source_and_hash.values(),
        key=lambda item: (item.source, item.display_label, item.candidate_id),
    )
    if not selected_reports:
        raise RuntimeError(
            "matched JSON paths contained no qualifying real-assembly variants; "
            "check report and candidate regexes"
        )
    if not candidates:
        raise RuntimeError("candidate filters removed every discovered layout")
    for candidate in candidates:
        candidate.aliases.sort(
            key=lambda alias: (alias.report_display, alias.label, alias.source)
        )
    return candidates, selected_reports, warnings


def find_data_root(config: dict[str, Any], sources: Iterable[str]) -> Path:
    configured = config.get("data_root")
    if configured:
        root = Path(str(configured))
        inputs = root / "train" / "inputs"
        if not inputs.is_dir():
            raise RuntimeError(f"configured data_root has no train/inputs: {root}")
        return root

    sources = sorted(set(sources))
    roots = sorted(
        {
            path.parent.parent
            for path in INPUT.glob("**/train/inputs")
            if path.is_dir() and all((path / source).is_file() for source in sources)
        }
    )
    if len(roots) != 1:
        raise RuntimeError(
            f"expected exactly one puzzle input root containing all report sources, found {roots}"
        )
    return roots[0]


def probe_hardware(max_devices: int) -> dict[str, Any]:
    subprocess.run(["nvidia-smi"], check=False)
    import torch

    available = torch.cuda.is_available()
    count = torch.cuda.device_count() if available else 0
    result: dict[str, Any] = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": available,
        "device_count": count,
        "devices": [],
        "capabilities": [],
        "arch_list": torch.cuda.get_arch_list() if available else [],
        "matmul_means": [],
    }
    if count < 1:
        raise RuntimeError(f"MAE gate requires at least one CUDA device: {result}")
    for index in range(count):
        result["devices"].append(torch.cuda.get_device_name(index))
        result["capabilities"].append(list(torch.cuda.get_device_capability(index)))
        left = torch.randn(128, 128, device=f"cuda:{index}")
        right = torch.randn(128, 128, device=f"cuda:{index}")
        result["matmul_means"].append(float((left @ right).mean().item()))
    result["devices_used"] = min(count, max_devices)
    return result


def package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def ensure_transformers() -> dict[str, Any]:
    requirements = Path(__file__).with_name("requirements.txt")
    before = package_version("transformers")
    installed = False
    if before != TRANSFORMERS_VERSION:
        install_target = (
            ["-r", str(requirements)]
            if requirements.is_file()
            else [f"transformers=={TRANSFORMERS_VERSION}"]
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
                *install_target,
            ],
            check=True,
        )
        installed = True
    after = package_version("transformers")
    if after != TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"expected transformers {TRANSFORMERS_VERSION}, found {after}"
        )
    return {
        "transformers_before": before,
        "transformers": after,
        "transformers_installed_by_runner": installed,
        "huggingface_hub": package_version("huggingface-hub"),
        "safetensors": package_version("safetensors"),
        "numpy": np.__version__,
        "pillow": package_version("Pillow"),
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
    if not all(path.is_file() for path in required):
        raise RuntimeError(f"incomplete model snapshot at {snapshot}")
    weights = sorted(snapshot.glob("*.safetensors"))
    if not weights:
        raise RuntimeError(f"model snapshot has no safetensors weights: {snapshot}")
    resolved_revision = snapshot.name
    metadata = {
        "model_id": config["model_id"],
        "requested_revision": config["model_revision"],
        "resolved_revision": resolved_revision,
        "configuration_files": [
            {"name": path.name, "sha256": sha256_file(path)} for path in required
        ],
        "weight_files": [
            {"name": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in weights
        ],
    }
    return snapshot, metadata


def read_input_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        values = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if values.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError(f"unexpected puzzle input shape {values.shape}: {path}")
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


def chunks(values: list[CandidateLayout], size: int) -> Iterable[list[CandidateLayout]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _configure_mask_ratio(model: Any, mask_ratio: float) -> None:
    model.config.mask_ratio = mask_ratio
    model.vit.config.mask_ratio = mask_ratio
    model.vit.embeddings.config.mask_ratio = mask_ratio


def score_source_group(
    *,
    device_index: int,
    sources: list[str],
    candidates_by_source: dict[str, list[CandidateLayout]],
    data_root: Path,
    model_snapshot: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    import torch
    from transformers import AutoImageProcessor, ViTMAEForPreTraining

    torch.manual_seed(int(config["mask_seed"]))
    torch.backends.cudnn.benchmark = False
    device = torch.device(f"cuda:{device_index}")
    processor = AutoImageProcessor.from_pretrained(
        str(model_snapshot), local_files_only=True, use_fast=False
    )
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
    model = ViTMAEForPreTraining.from_pretrained(
        str(model_snapshot),
        local_files_only=True,
        use_safetensors=True,
        # Two worker threads instantiate independent replicas concurrently.
        # Transformers' meta-device low-memory loader is not thread-safe in
        # the current Kaggle torch/transformers combination and can leave one
        # replica with unmaterialized meta parameters.
        low_cpu_mem_usage=False,
    )
    _configure_mask_ratio(model, float(config["mask_ratio"]))
    model.eval().to(device)
    if config["dtype"] == "float16":
        model.half()
        model_dtype = torch.float16
    else:
        model.float()
        model_dtype = torch.float32

    patch_size = int(model.config.patch_size)
    image_size = int(model.config.image_size)
    norm_pix_loss = bool(model.config.norm_pix_loss)
    if image_size % patch_size:
        raise RuntimeError("MAE model image size is not divisible by patch size")
    num_patches = (image_size // patch_size) ** 2
    mask_rng = np.random.default_rng(int(config["mask_seed"]))
    fixed_noise = mask_rng.random(
        (int(config["num_masks"]), num_patches), dtype=np.float32
    )

    energy_records: list[dict[str, Any]] = []
    consistency_differences: list[float] = []
    started = time.perf_counter()
    for source_index, source in enumerate(sources):
        input_path = data_root / "train" / "inputs" / source
        if not input_path.is_file():
            raise FileNotFoundError(f"missing report source input: {input_path}")
        raw_tiles = split_tiles(read_input_rgb(input_path))
        candidates = candidates_by_source[source]
        for candidate_batch in chunks(candidates, int(config["candidate_batch_size"])):
            mosaics = [
                Image.fromarray(
                    merge_tiles(raw_tiles[candidate.position_to_slot]), mode="RGB"
                )
                for candidate in candidate_batch
            ]
            processed = processor(images=mosaics, return_tensors="pt")
            pixel_values = processed["pixel_values"].to(device=device, dtype=model_dtype)
            batch_size = len(candidate_batch)
            num_masks = int(config["num_masks"])
            repeated_pixels = pixel_values.repeat_interleave(num_masks, dim=0)
            noise = torch.from_numpy(np.tile(fixed_noise, (batch_size, 1))).to(device)
            with torch.inference_mode():
                outputs = model(pixel_values=repeated_pixels, noise=noise)
                target = model.patchify(repeated_pixels).float()
                if norm_pix_loss:
                    mean = target.mean(dim=-1, keepdim=True)
                    variance = target.var(dim=-1, keepdim=True)
                    target = (target - mean) / torch.sqrt(variance + 1e-6)
                patch_loss = (outputs.logits.float() - target).square().mean(dim=-1)
                per_sample = (
                    (patch_loss * outputs.mask.float()).sum(dim=-1)
                    / outputs.mask.float().sum(dim=-1).clamp_min(1.0)
                )
                consistency_differences.append(
                    abs(float(outputs.loss.float().item()) - float(per_sample.mean().item()))
                )
            errors = per_sample.reshape(batch_size, num_masks).cpu().numpy()
            for candidate, candidate_errors in zip(
                candidate_batch, errors.tolist(), strict=True
            ):
                mean_error = float(np.mean(candidate_errors, dtype=np.float64))
                std_error = float(np.std(candidate_errors, dtype=np.float64))
                if not math.isfinite(mean_error) or not math.isfinite(std_error):
                    raise FloatingPointError(
                        f"non-finite MAE energy for {candidate.candidate_id}"
                    )
                energy_records.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "source": candidate.source,
                        "label": candidate.display_label,
                        "labels": candidate.labels,
                        "layout_sha256": candidate.layout_sha256,
                        "aliases": [
                            {
                                "report": alias.report_display,
                                "label": alias.label,
                            }
                            for alias in candidate.aliases
                        ],
                        "mae_error_mean": mean_error,
                        "mae_error_std": std_error,
                        "mae_error_by_mask": [float(value) for value in candidate_errors],
                        "naturalness_score": -mean_error,
                        "device": str(device),
                    }
                )
        print(
            json.dumps(
                {
                    "event": "mae_energy_source_complete",
                    "device": str(device),
                    "index": source_index + 1,
                    "count": len(sources),
                    "source": source,
                    "candidates": len(candidates),
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
        "model_image_size": image_size,
        "model_patch_size": patch_size,
        "model_num_patches": num_patches,
        "model_norm_pix_loss": norm_pix_loss,
        "processor": processor_metadata,
        "max_forward_loss_consistency_abs": (
            max(consistency_differences) if consistency_differences else None
        ),
        "energy_records": energy_records,
    }


def evaluate_energies(
    candidates: list[CandidateLayout],
    *,
    data_root: Path,
    model_snapshot: Path,
    device_count: int,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates_by_source: dict[str, list[CandidateLayout]] = {}
    for candidate in candidates:
        candidates_by_source.setdefault(candidate.source, []).append(candidate)
    sources = sorted(candidates_by_source)
    groups = [sources[index::device_count] for index in range(device_count)]
    groups = [group for group in groups if group]
    worker_results: list[dict[str, Any]] = []
    if len(groups) == 1:
        worker_results.append(
            score_source_group(
                device_index=0,
                sources=groups[0],
                candidates_by_source=candidates_by_source,
                data_root=data_root,
                model_snapshot=model_snapshot,
                config=config,
            )
        )
    else:
        with ThreadPoolExecutor(max_workers=len(groups)) as executor:
            futures = {
                executor.submit(
                    score_source_group,
                    device_index=device_index,
                    sources=group,
                    candidates_by_source=candidates_by_source,
                    data_root=data_root,
                    model_snapshot=model_snapshot,
                    config=config,
                ): device_index
                for device_index, group in enumerate(groups)
            }
            for future in as_completed(futures):
                worker_results.append(future.result())

    energy_records = sorted(
        [
            record
            for worker in worker_results
            for record in worker.pop("energy_records")
        ],
        key=lambda record: (
            record["source"],
            record["mae_error_mean"],
            record["candidate_id"],
        ),
    )
    worker_results.sort(key=lambda record: record["device"])
    if len(energy_records) != len(candidates):
        raise RuntimeError(
            f"energy record count mismatch: {len(energy_records)} != {len(candidates)}"
        )
    return energy_records, worker_results


def alias_matches(candidate: dict[str, Any], pattern: re.Pattern[str]) -> bool:
    return any(pattern.search(str(label)) for label in candidate["labels"])


def choose_frozen_selections(
    energy_records: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    by_source: dict[str, list[dict[str, Any]]] = {}
    for record in energy_records:
        by_source.setdefault(record["source"], []).append(record)
    baseline_pattern = re.compile(str(config["baseline_label_regex"]))
    baseline_report_pattern = re.compile(str(config["baseline_report_regex"]))
    fallback_pattern = re.compile(str(config["baseline_fallback_label_regex"]))

    selections: dict[str, dict[str, Any]] = {}
    for source, records in sorted(by_source.items()):
        best = min(
            records,
            key=lambda record: (record["mae_error_mean"], record["candidate_id"]),
        )
        primary_baselines = [
            record
            for record in records
            if alias_matches(record, baseline_pattern)
            and any(
                baseline_report_pattern.search(alias["report"])
                for alias in record["aliases"]
            )
        ]
        baseline_mode = "primary_label_and_report"
        if not primary_baselines:
            primary_baselines = [
                record for record in records if alias_matches(record, baseline_pattern)
            ]
            baseline_mode = "primary_label_any_report"
        if not primary_baselines:
            primary_baselines = [
                record for record in records if alias_matches(record, fallback_pattern)
            ]
            baseline_mode = "fallback_label"
        baseline = (
            min(
                primary_baselines,
                key=lambda record: (record["label"], record["candidate_id"]),
            )
            if primary_baselines
            else None
        )
        selections[source] = {
            "best_by_energy_candidate_id": best["candidate_id"],
            "baseline_candidate_id": baseline["candidate_id"] if baseline else None,
            "baseline_selection_mode": baseline_mode if baseline else "missing",
        }
    return selections


def freeze_energy_artifact(
    *,
    output: Path,
    config: dict[str, Any],
    reports: list[dict[str, Any]],
    model_metadata: dict[str, Any],
    energy_records: list[dict[str, Any]],
    worker_results: list[dict[str, Any]],
    warnings: list[str],
) -> tuple[dict[str, Any], str]:
    selections = choose_frozen_selections(energy_records, config)
    payload = {
        "schema_version": 1,
        "kind": "input_only_frozen_mae_energy",
        "contains_target_metrics": False,
        "anti_leakage": {
            "target_images_opened": False,
            "target_report_fields_accessed_by_phase_a_logic": False,
            "candidate_selection_uses_target": False,
            "phase": "A_frozen_before_target_attachment",
        },
        "config": config,
        "reports": reports,
        "model": model_metadata,
        "workers": worker_results,
        "energy_records": energy_records,
        "frozen_selections": selections,
        "warnings": warnings,
    }
    write_json_atomic(output, payload)
    digest = sha256_file(output)
    return payload, digest


def extract_recorded_ssim(
    payload: dict[str, Any],
    *,
    source: str,
    label: str,
    target_render_view: str,
) -> float | None:
    source_records = [
        item
        for item in payload.get("sources", [])
        if isinstance(item, dict) and item.get("source") == source
    ]
    if len(source_records) != 1:
        return None
    variants = source_records[0].get("variants")
    if not isinstance(variants, dict):
        return None
    if target_render_view in {"raw", "denoised"}:
        keys = [f"{label}__{target_render_view}_render"]
    else:
        keys = [
            f"{label}__raw_render",
            f"{label}__denoised_render",
        ]
    values = []
    for key in keys:
        variant = variants.get(key)
        if not isinstance(variant, dict):
            continue
        value = variant.get("predicted_layout_ssim")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return max(values) if values else None


def attach_targets_after_freeze(
    candidates: list[CandidateLayout],
    energy_records: list[dict[str, Any]],
    *,
    frozen_output: Path,
    expected_frozen_sha256: str,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Phase B: first target-dependent function in the program."""
    if sha256_file(frozen_output) != expected_frozen_sha256:
        raise RuntimeError("frozen energy artifact changed before target attachment")
    frozen = json.loads(frozen_output.read_text(encoding="utf-8"))
    if frozen.get("contains_target_metrics") is not False:
        raise RuntimeError("frozen artifact anti-leakage marker is invalid")

    report_cache: dict[Path, dict[str, Any]] = {}
    target_by_candidate: dict[str, tuple[float | None, list[dict[str, Any]]]] = {}
    warnings: list[str] = []
    for candidate in candidates:
        alias_values: list[dict[str, Any]] = []
        for alias in candidate.aliases:
            payload = report_cache.get(alias.report_path)
            if payload is None:
                payload = json.loads(alias.report_path.read_text(encoding="utf-8"))
                report_cache[alias.report_path] = payload
            value = extract_recorded_ssim(
                payload,
                source=alias.source,
                label=alias.label,
                target_render_view=str(config["target_render_view"]),
            )
            if value is not None:
                alias_values.append(
                    {
                        "report": alias.report_display,
                        "label": alias.label,
                        "predicted_layout_ssim": value,
                    }
                )
        values = [item["predicted_layout_ssim"] for item in alias_values]
        target_value = float(np.median(values)) if values else None
        if values and max(values) - min(values) > 1e-8:
            warnings.append(
                f"target SSIM conflict for duplicate layout {candidate.candidate_id}: "
                f"range={max(values) - min(values):.9g}; median used"
            )
        target_by_candidate[candidate.candidate_id] = (target_value, alias_values)

    attached = []
    for record in energy_records:
        target_value, provenance = target_by_candidate[record["candidate_id"]]
        attached.append(
            {
                **record,
                "recorded_target_ssim": target_value,
                "target_score_provenance": provenance,
            }
        )
    return attached, warnings


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        rank = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = rank
        start = end
    return ranks


def spearman_correlation(first: list[float], second: list[float]) -> float | None:
    if len(first) < 2:
        return None
    first_ranks = average_ranks(np.asarray(first, dtype=np.float64))
    second_ranks = average_ranks(np.asarray(second, dtype=np.float64))
    if np.std(first_ranks) <= 0.0 or np.std(second_ranks) <= 0.0:
        return None
    return float(np.corrcoef(first_ranks, second_ranks)[0, 1])


def pairwise_ranking(
    records: list[dict[str, Any]],
    *,
    target_tolerance: float,
    energy_tolerance: float,
) -> dict[str, Any]:
    correct_weight = 0.0
    pairs = 0
    ties = 0
    for first_index in range(len(records)):
        for second_index in range(first_index + 1, len(records)):
            first = records[first_index]
            second = records[second_index]
            target_delta = float(first["recorded_target_ssim"]) - float(
                second["recorded_target_ssim"]
            )
            if abs(target_delta) <= target_tolerance:
                continue
            score_delta = float(first["naturalness_score"]) - float(
                second["naturalness_score"]
            )
            pairs += 1
            if abs(score_delta) <= energy_tolerance:
                ties += 1
                correct_weight += 0.5
            elif (score_delta > 0.0) == (target_delta > 0.0):
                correct_weight += 1.0
    return {
        "accuracy": correct_weight / pairs if pairs else None,
        "correct_weight": correct_weight,
        "pairs": pairs,
        "energy_ties": ties,
    }


def mean_optional(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def analyze_attached_records(
    attached_records: list[dict[str, Any]],
    frozen_selections: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    by_source: dict[str, list[dict[str, Any]]] = {}
    for record in attached_records:
        by_source.setdefault(record["source"], []).append(record)

    source_reports: list[dict[str, Any]] = []
    total_pair_correct = 0.0
    total_pairs = 0
    for source, all_records in sorted(by_source.items()):
        records = [
            record for record in all_records if record["recorded_target_ssim"] is not None
        ]
        record_by_id = {record["candidate_id"]: record for record in records}
        naturalness = [float(record["naturalness_score"]) for record in records]
        target_ssim = [float(record["recorded_target_ssim"]) for record in records]
        spearman = spearman_correlation(naturalness, target_ssim)
        pairwise = pairwise_ranking(
            records,
            target_tolerance=float(config["target_tie_tolerance"]),
            energy_tolerance=float(config["energy_tie_tolerance"]),
        )
        total_pair_correct += float(pairwise["correct_weight"])
        total_pairs += int(pairwise["pairs"])

        selections = frozen_selections[source]
        best = record_by_id.get(selections["best_by_energy_candidate_id"])
        baseline = record_by_id.get(selections["baseline_candidate_id"])
        oracle = (
            max(
                records,
                key=lambda record: (
                    record["recorded_target_ssim"],
                    -record["mae_error_mean"],
                    record["candidate_id"],
                ),
            )
            if records
            else None
        )

        def summary(record: dict[str, Any] | None) -> dict[str, Any] | None:
            if record is None:
                return None
            return {
                "candidate_id": record["candidate_id"],
                "label": record["label"],
                "labels": record["labels"],
                "mae_error_mean": record["mae_error_mean"],
                "recorded_target_ssim": record["recorded_target_ssim"],
            }

        best_ssim = float(best["recorded_target_ssim"]) if best else None
        baseline_ssim = (
            float(baseline["recorded_target_ssim"]) if baseline else None
        )
        oracle_ssim = float(oracle["recorded_target_ssim"]) if oracle else None
        source_reports.append(
            {
                "source": source,
                "candidate_count_energy": len(all_records),
                "candidate_count_with_target": len(records),
                "spearman_naturalness_vs_ssim": spearman,
                "pairwise_ranking": pairwise,
                "best_by_energy": summary(best),
                "baseline": summary(baseline),
                "baseline_selection_mode": selections["baseline_selection_mode"],
                "oracle": summary(oracle),
                "best_by_energy_minus_baseline_ssim": (
                    best_ssim - baseline_ssim
                    if best_ssim is not None and baseline_ssim is not None
                    else None
                ),
                "oracle_minus_best_by_energy_ssim": (
                    oracle_ssim - best_ssim
                    if oracle_ssim is not None and best_ssim is not None
                    else None
                ),
                "candidates": records,
            }
        )

    valid_spearman = [
        source["spearman_naturalness_vs_ssim"]
        for source in source_reports
        if source["spearman_naturalness_vs_ssim"] is not None
    ]
    valid_pairwise_sources = [
        source
        for source in source_reports
        if source["pairwise_ranking"]["accuracy"] is not None
    ]
    macro = {
        "evaluable_sources": len(
            [
                source
                for source in source_reports
                if source["spearman_naturalness_vs_ssim"] is not None
                and source["pairwise_ranking"]["accuracy"] is not None
            ]
        ),
        "mean_spearman_naturalness_vs_ssim": mean_optional(valid_spearman),
        "mean_pairwise_ranking_accuracy": mean_optional(
            source["pairwise_ranking"]["accuracy"]
            for source in valid_pairwise_sources
        ),
        "micro_pairwise_ranking_accuracy": (
            total_pair_correct / total_pairs if total_pairs else None
        ),
        "micro_pairwise_pairs": total_pairs,
        "mean_best_by_energy_ssim": mean_optional(
            source["best_by_energy"]["recorded_target_ssim"]
            if source["best_by_energy"]
            else None
            for source in source_reports
        ),
        "mean_baseline_ssim": mean_optional(
            source["baseline"]["recorded_target_ssim"]
            if source["baseline"]
            else None
            for source in source_reports
        ),
        "mean_oracle_ssim": mean_optional(
            source["oracle"]["recorded_target_ssim"]
            if source["oracle"]
            else None
            for source in source_reports
        ),
        "mean_best_by_energy_minus_baseline_ssim": mean_optional(
            source["best_by_energy_minus_baseline_ssim"] for source in source_reports
        ),
        "mean_oracle_minus_best_by_energy_ssim": mean_optional(
            source["oracle_minus_best_by_energy_ssim"] for source in source_reports
        ),
    }
    spearman_value = macro["mean_spearman_naturalness_vs_ssim"]
    pairwise_value = macro["micro_pairwise_ranking_accuracy"]
    reasons = []
    if macro["evaluable_sources"] < int(config["min_evaluable_sources"]):
        reasons.append(
            f"evaluable_sources {macro['evaluable_sources']} < "
            f"{config['min_evaluable_sources']}"
        )
    if spearman_value is None or spearman_value < float(config["promotion_spearman"]):
        reasons.append(
            f"mean Spearman {spearman_value} < {config['promotion_spearman']}"
        )
    if pairwise_value is None or pairwise_value < float(
        config["promotion_pairwise_accuracy"]
    ):
        reasons.append(
            f"micro pairwise accuracy {pairwise_value} < "
            f"{config['promotion_pairwise_accuracy']}"
        )
    gate = {
        "passed": not reasons,
        "thresholds": {
            "mean_spearman_naturalness_vs_ssim": config["promotion_spearman"],
            "micro_pairwise_ranking_accuracy": config[
                "promotion_pairwise_accuracy"
            ],
            "min_evaluable_sources": config["min_evaluable_sources"],
        },
        "observed": {
            "mean_spearman_naturalness_vs_ssim": spearman_value,
            "micro_pairwise_ranking_accuracy": pairwise_value,
            "evaluable_sources": macro["evaluable_sources"],
        },
        "failure_reasons": reasons,
    }
    return source_reports, macro, gate


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    config_path = Path(args.config)
    config = load_config(config_path)
    if args.validate_config_only:
        print(
            json.dumps(
                {"event": "mae_energy_config_valid", "config": config},
                sort_keys=True,
            )
        )
        return

    os.environ.setdefault("HF_HOME", str(config["cache_dir"]))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Phase A starts.  Layout discovery intentionally ignores all score fields.
    candidates, selected_reports, discovery_warnings = discover_layout_candidates(config)
    data_root = find_data_root(config, (candidate.source for candidate in candidates))
    hardware = probe_hardware(int(config["max_devices"]))
    dependencies = ensure_transformers()
    model_snapshot, model_metadata = download_model_snapshot(config)
    energy_records, worker_results = evaluate_energies(
        candidates,
        data_root=data_root,
        model_snapshot=model_snapshot,
        device_count=int(hardware["devices_used"]),
        config=config,
    )
    frozen_output = Path(args.frozen_output)
    frozen_payload, frozen_sha256 = freeze_energy_artifact(
        output=frozen_output,
        config=config,
        reports=selected_reports,
        model_metadata=model_metadata,
        energy_records=energy_records,
        worker_results=worker_results,
        warnings=discovery_warnings,
    )
    print(
        json.dumps(
            {
                "event": "mae_energies_frozen",
                "output": str(frozen_output),
                "sha256": frozen_sha256,
                "candidates": len(energy_records),
                "sources": len({record["source"] for record in energy_records}),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    # Phase B starts only after the target-free artifact and input-only choices
    # are immutable and hashed.  Only existing report metrics are read here.
    attached_records, target_warnings = attach_targets_after_freeze(
        candidates,
        energy_records,
        frozen_output=frozen_output,
        expected_frozen_sha256=frozen_sha256,
        config=config,
    )
    source_reports, macro, promotion_gate = analyze_attached_records(
        attached_records,
        frozen_payload["frozen_selections"],
        config,
    )
    final_report = {
        "schema_version": 1,
        "kind": "frozen_mae_energy_target_correlation_gate",
        "anti_leakage": {
            "predictor_accepts_target": False,
            "target_images_opened": False,
            "raw_mosaics_reconstructed_from_train_inputs_only": True,
            "energy_and_candidate_choices_frozen_before_target_metrics": True,
            "frozen_energy_artifact": str(frozen_output),
            "frozen_energy_sha256": frozen_sha256,
            "target_metrics_source": "already-recorded report predicted_layout_ssim",
            "target_metrics_used_for": "post-hoc correlation, pairwise accuracy, and oracle only",
        },
        "config_path": str(config_path),
        "config_sidecar_present": config_path.is_file(),
        "config_sha256": (
            sha256_file(config_path) if config_path.is_file() else sha256_json(config)
        ),
        "config_hash_scope": (
            "sidecar_file" if config_path.is_file() else "embedded_effective_config"
        ),
        "config": config,
        "data_root": str(data_root),
        "hardware": hardware,
        "dependencies": dependencies,
        "model": model_metadata,
        "reports": selected_reports,
        "sources": source_reports,
        "macro": macro,
        "promotion_gate": promotion_gate,
        "warnings": [*discovery_warnings, *target_warnings],
        "seconds": time.perf_counter() - started,
    }
    output = Path(args.output)
    write_json_atomic(output, final_report)
    print(
        json.dumps(
            {
                "event": "mae_energy_gate_complete",
                "output": str(output),
                "sha256": sha256_file(output),
                "passed": promotion_gate["passed"],
                "macro": macro,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

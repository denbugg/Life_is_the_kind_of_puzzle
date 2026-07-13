#!/usr/bin/env python3
"""Build and rigorously verify the final 700-image QAP submission on T4x2.

One image is solved first as a fail-closed end-to-end preflight.  The two fixed
shards (0:350 and 350:700) then run concurrently, one per GPU, and the first
shard must reproduce the preflight image byte-for-byte.  The only configuration
normally edited after validation tuning is DEFAULT_CONFIG below.  It can also
be deep-overridden with VSOS_FINAL_CONFIG_PATH or VSOS_FINAL_CONFIG_JSON without
changing this runner.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
from typing import Any
import zipfile

from PIL import Image


INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
ARCHIVE_TIMESTAMP = (2026, 7, 11, 0, 0, 0)
BUILDER_ARCHIVE_TIMESTAMP = (2026, 7, 10, 0, 0, 0)


# ---------------------------------------------------------------------------
# PROMOTED CONFIG: update this one block after the final validation gate.
# Values under builder_args are translated from snake_case to --kebab-case.
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "expected_total": 700,
    "shard_size": 350,
    "preflight_count": 1,
    "require_gpu_count": 2,
    "require_t4": True,
    "assets": {
        "denoiser_filename": "selected_tilenaf_synth_50k.pt",
        "embedding_checkpoint_filename": "hbt_d320_denoised_rgb_sobel.pt",
        "renderer_denoiser_filename": None,
        # Future promoted context/hyperedge/DINO checkpoints can be wired to
        # builder flags here without changing the shard/preflight machinery:
        # "additional_builder_assets": {"context-checkpoint": "context.pt"}.
        "additional_builder_assets": {},
    },
    # Fail closed if Kaggle mounts a stale or different code-dataset version.
    # Update this contract together with DEFAULT_CONFIG after promoting a new
    # solver implementation.  These hashes match code dataset v7.
    "code_contract": {
        "dataset_slug": "pasha883/vsos-solver-rework-night-code",
        "expected_dataset_version": 7,
        "required_sha256": {
            "scripts/build_assembly_submission.py": "8433c0e545edfeb49f2512208a3ea062fb1a248a64bcde3f87037cdf30d6ac97",
            "src/puzzle_assembly/__init__.py": "09e051b7555471aafca03cd666d789f033aca47f1c82f6e2af9c0cce50afe9d5",
            "src/puzzle_assembly/compatibility.py": "aff2149b161c4fded4e5d91fbea49a8a62967886148d3ad374467331e0416a9f",
            "src/puzzle_assembly/components.py": "53fcc7c4fd23956db884ee45060e47f8e94a931c16e497e426d67549621bd367",
            "src/puzzle_assembly/geometry.py": "1e16bec6fb98a33060558d5d28062334d9114b12424733ef103a40393ef1ba86",
            "src/puzzle_assembly/learned.py": "9e3dba673aa85eaab5698dbeb63b3d94f88e3ea92b5e5979bde4b0273642697b",
            "src/puzzle_assembly/line_seam.py": "56c3065fb36427a96c3fbddda515fc28f49dcfd8e0b3a5a721dd8fd28603305d",
            "src/puzzle_assembly/qap.py": "b8a5e1da67387fd04effd979270ca16925aceab23d37083eda108c6e3e349c32",
            "src/puzzle_assembly/solvers.py": "23f9e32200748349d0da8558b7b44053a758e1c1eb306d8f31ce59feae03fe8e",
            "src/puzzle_denoise_v2/degradation.py": "7e314081c143a1c7846a9777eaea8716092a85595f856769efd3704a2c583a75",
            "src/puzzle_denoise_v2/__init__.py": "30849e0f937ba4a50e85ce2eee0d2b930db06fbcc0b7dff84547e121ef2f30b7",
            "src/puzzle_denoise_v2/inference.py": "20767cc26270cfde7472cf33a0247b1ea6d96e5b5c8ff5d705b785ae710dd6da",
            "src/puzzle_denoise_v2/losses.py": "56776289cd51e49a28ce54bc4762d144d87c7efbf6d4ca56668fc3b019dbbf34",
            "src/puzzle_denoise_v2/metrics.py": "e8275fb096276a63b7114be1a74b24009dc2143dddf299ce5eaceac401a27d36",
            "src/puzzle_denoise_v2/model.py": "37db32fb83ece0f122757bdbec19ffc6a17c5e5e00ef92a26328247d95c55d11",
            "src/puzzle_denoise_v2/tiles.py": "21270e283e50ea0b155ef194de889222fb0c4f6954437eb1526342c006eefaa7",
            "src/puzzle_denoise_v2/training.py": "6719ee6a62434cd8a00fafb92b28f6a10941cdbf5c83573fc6556b33e5eba56e",
        },
    },
    "builder_args": {
        "batch_size": 512,
        "chunk_size": 64,
        "line_seam": False,
        "line_seam_auxiliary_weight": 0.35,
        "line_seam_fusion_weight": 0.5,
        "soft_cycle_score": "l1",
        "soft_cycle_topk": 8,
        "soft_cycle_keep_per_tile": 1,
        "soft_cycle_keep_fraction": 0.5,
        "soft_cycle_loop_weight": 1.0,
        "soft_cycle_reciprocal_weight": 0.35,
        "qap_score": "l1w4",
        "qap_iterations": 25,
        "qap_restarts": 2,
        "qap_initial_weight": 0.75,
        "qap_noisy_components": 3,
        "qap_noise_scale": 1.0,
        "qap_boundary_weight": 0.05,
        "qap_refine_swaps": 8,
        "qap_refine_weak_cells": 32,
    },
}


ENV_BUILDER_OVERRIDES: dict[str, tuple[str, Any]] = {
    "VSOS_QAP_SCORE": ("qap_score", str),
    "VSOS_QAP_ITERATIONS": ("qap_iterations", int),
    "VSOS_QAP_RESTARTS": ("qap_restarts", int),
    "VSOS_QAP_INITIAL_WEIGHT": ("qap_initial_weight", float),
    "VSOS_QAP_NOISY_COMPONENTS": ("qap_noisy_components", int),
    "VSOS_QAP_NOISE_SCALE": ("qap_noise_scale", float),
    "VSOS_QAP_BOUNDARY_WEIGHT": ("qap_boundary_weight", float),
    "VSOS_QAP_REFINE_SWAPS": ("qap_refine_swaps", int),
    "VSOS_QAP_REFINE_WEAK_CELLS": ("qap_refine_weak_cells", int),
    "VSOS_SOFT_CYCLE_SCORE": ("soft_cycle_score", str),
    "VSOS_SOFT_CYCLE_TOPK": ("soft_cycle_topk", int),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_names(names: list[str]) -> str:
    return sha256_bytes(("\n".join(names) + "\n").encode("utf-8"))


def canonical_config_sha256(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(payload)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean: {value!r}")


def load_config() -> tuple[dict[str, Any], dict[str, str]]:
    config = deepcopy(DEFAULT_CONFIG)
    provenance: dict[str, str] = {"base": "embedded DEFAULT_CONFIG"}
    config_path = os.environ.get("VSOS_FINAL_CONFIG_PATH")
    if config_path:
        path = Path(config_path)
        override = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(override, dict):
            raise TypeError("VSOS_FINAL_CONFIG_PATH must contain a JSON object")
        config = deep_merge(config, override)
        provenance["path"] = str(path)
    config_json = os.environ.get("VSOS_FINAL_CONFIG_JSON")
    if config_json:
        override = json.loads(config_json)
        if not isinstance(override, dict):
            raise TypeError("VSOS_FINAL_CONFIG_JSON must be a JSON object")
        config = deep_merge(config, override)
        provenance["json"] = "VSOS_FINAL_CONFIG_JSON"
    builder_args = config.setdefault("builder_args", {})
    for environment_name, (argument_name, converter) in ENV_BUILDER_OVERRIDES.items():
        if environment_name in os.environ:
            builder_args[argument_name] = converter(os.environ[environment_name])
            provenance[environment_name] = "environment override"
    if "VSOS_LINE_SEAM" in os.environ:
        builder_args["line_seam"] = parse_bool(os.environ["VSOS_LINE_SEAM"])
        provenance["VSOS_LINE_SEAM"] = "environment override"
    return config, provenance


def single(candidates: list[Path], label: str) -> Path:
    unique = sorted(set(path.resolve() for path in candidates))
    if len(unique) != 1:
        raise RuntimeError(f"expected exactly one {label}, found {unique}")
    return unique[0]


def require_filename(value: Any, label: str) -> str:
    filename = str(value)
    if Path(filename).name != filename or filename in {"", ".", ".."}:
        raise RuntimeError(f"{label} must be a filename, got {filename!r}")
    return filename


def find_data_root() -> Path:
    return single(
        [
            path.parent.parent
            for path in INPUT.glob("**/train/inputs")
            if path.is_dir()
            and (path.parent / "targets").is_dir()
            and (path.parent.parent / "test").is_dir()
        ],
        "puzzle data root",
    )


def find_runtime_root(denoiser_filename: str, embedding_filename: str) -> Path:
    return single(
        [
            path.parent
            for path in INPUT.glob(f"**/{denoiser_filename}")
            if (path.parent / embedding_filename).is_file()
        ],
        "runtime checkpoint root",
    )


def _is_valid_code_root(path: Path) -> bool:
    return (
        (path / "scripts" / "build_assembly_submission.py").is_file()
        and (path / "src" / "puzzle_assembly" / "qap.py").is_file()
        and (path / "src" / "puzzle_denoise_v2" / "inference.py").is_file()
    )


def find_code_root() -> Path:
    preferred = [
        INPUT / "datasets" / "pasha883" / "vsos-solver-rework-night-code",
        INPUT / "vsos-solver-rework-night-code",
    ]
    for path in preferred:
        if _is_valid_code_root(path):
            return path.resolve()
    candidates = [
        path.parent.parent
        for path in INPUT.glob("**/scripts/build_assembly_submission.py")
        if _is_valid_code_root(path.parent.parent)
    ]
    preferred_candidates = [
        path for path in candidates if "solver-rework-night-code" in str(path)
    ]
    return single(preferred_candidates or candidates, "QAP submission code root")


def validate_code_contract(
    code_root: Path, contract: dict[str, Any]
) -> list[dict[str, Any]]:
    required = contract.get("required_sha256")
    if not isinstance(required, dict) or not required:
        raise RuntimeError("code_contract.required_sha256 must be a non-empty object")
    records: list[dict[str, Any]] = []
    for relative_name, expected_digest in sorted(required.items()):
        if not isinstance(relative_name, str) or not isinstance(expected_digest, str):
            raise RuntimeError("code contract paths and digests must be strings")
        relative = Path(relative_name)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"unsafe code-contract path: {relative_name!r}")
        path = code_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"code-contract file is missing: {path}")
        actual_digest = sha256(path)
        if actual_digest != expected_digest.lower():
            raise RuntimeError(
                f"stale/wrong code dataset file {relative_name}: "
                f"expected {expected_digest}, got {actual_digest}"
            )
        records.append(
            {
                "path": relative_name,
                "bytes": path.stat().st_size,
                "sha256": actual_digest,
            }
        )
    return records


def hardware_probe(config: dict[str, Any]) -> dict[str, Any]:
    nvidia = subprocess.run(
        ["nvidia-smi"], check=False, text=True, capture_output=True
    )
    print(nvidia.stdout, flush=True)
    if nvidia.stderr:
        print(nvidia.stderr, file=sys.stderr, flush=True)

    import torch

    available = torch.cuda.is_available()
    count = torch.cuda.device_count()
    devices = [torch.cuda.get_device_name(index) for index in range(count)]
    result: dict[str, Any] = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": available,
        "device_count": count,
        "devices": devices,
        "capabilities": [
            list(torch.cuda.get_device_capability(index)) for index in range(count)
        ],
        "arch_list": torch.cuda.get_arch_list() if available else [],
        "nvidia_smi_returncode": nvidia.returncode,
        "tensor_probe_means": [],
    }
    required = int(config["require_gpu_count"])
    if not available or count < required:
        raise RuntimeError(f"expected at least {required} CUDA devices, got {result}")
    if bool(config.get("require_t4", False)) and any(
        "T4" not in name.upper() for name in devices[:required]
    ):
        raise RuntimeError(f"expected T4 devices, got {devices[:required]}")
    for index in range(required):
        device = torch.device(f"cuda:{index}")
        generator = torch.Generator(device=device).manual_seed(20260711 + index)
        left = torch.randn(128, 128, device=device, generator=generator)
        right = torch.randn(128, 128, device=device, generator=generator)
        result["tensor_probe_means"].append(float((left @ right).mean().item()))
    return result


def builder_options(builder: Path, code_root: Path) -> set[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(code_root / "src")
    completed = subprocess.run(
        [sys.executable, str(builder), "--help"],
        cwd=code_root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "builder --help failed:\n" + completed.stdout + "\n" + completed.stderr
        )
    options = set(re.findall(r"(?<![\w-])(--[a-z0-9][a-z0-9-]*)", completed.stdout))
    required = {
        "--input-dir",
        "--denoiser",
        "--embedding-checkpoint",
        "--output",
        "--report",
        "--offset",
        "--limit",
        "--expected-count",
        "--device",
        "--overwrite",
        "--qap-score",
        "--qap-iterations",
        "--qap-restarts",
    }
    missing = sorted(required - options)
    if missing:
        raise RuntimeError(f"packaged builder is missing required options: {missing}")
    source = builder.read_text(encoding="utf-8")
    if "directional_qap(" not in source:
        raise RuntimeError(
            "packaged builder does not call directional_qap; refresh the code dataset "
            "instead of silently producing the legacy component submission"
        )
    return options


def append_configured_builder_args(
    command: list[str], builder_args: dict[str, Any], supported: set[str]
) -> None:
    for key in sorted(builder_args):
        value = builder_args[key]
        option = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                if option not in supported:
                    raise RuntimeError(f"configured builder flag is unsupported: {option}")
                command.append(option)
            continue
        if value is None:
            continue
        if option not in supported:
            raise RuntimeError(f"configured builder option is unsupported: {option}")
        command.extend([option, str(value)])


def validate_png(payload: bytes, name: str) -> None:
    with Image.open(BytesIO(payload)) as image:
        image.load()
        if image.format != "PNG":
            raise RuntimeError(f"archive member is not PNG: {name}")
        if image.mode != "RGB" or image.size != (480, 480):
            raise RuntimeError(
                f"invalid archived image {name}: mode={image.mode}, size={image.size}"
            )


def validate_zip(
    path: Path,
    expected_names: list[str],
    *,
    deterministic_timestamp: tuple[int, int, int, int, int, int] | None = None,
) -> list[dict[str, Any]]:
    expected = sorted(expected_names)
    with zipfile.ZipFile(path) as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise RuntimeError(f"corrupt zip member in {path}: {corrupt}")
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError(f"duplicate names in {path}")
        if sorted(names) != expected:
            missing = sorted(set(expected) - set(names))[:10]
            extra = sorted(set(names) - set(expected))[:10]
            raise RuntimeError(
                f"member set mismatch in {path}; missing={missing}, extra={extra}"
            )
        records = []
        for name in sorted(names):
            if Path(name).name != name or not name.endswith(".png"):
                raise RuntimeError(f"non-root or non-PNG archive member: {name}")
            info = archive.getinfo(name)
            if deterministic_timestamp is not None:
                if info.date_time != deterministic_timestamp:
                    raise RuntimeError(f"non-deterministic zip timestamp for {name}")
                if info.compress_type != zipfile.ZIP_DEFLATED:
                    raise RuntimeError(f"unexpected compression type for {name}")
                if info.create_system != 3 or (info.external_attr >> 16) != 0o100644:
                    raise RuntimeError(f"unexpected Unix metadata for {name}")
            payload = archive.read(name)
            validate_png(payload, name)
            records.append(
                {"name": name, "bytes": len(payload), "sha256": sha256_bytes(payload)}
            )
    return records


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if isinstance(expected, float):
        if not isinstance(actual, (int, float)) or isinstance(actual, bool):
            raise RuntimeError(f"{label} must be numeric, got {actual!r}")
        if abs(float(actual) - expected) > 1e-12:
            raise RuntimeError(f"{label} mismatch: expected {expected!r}, got {actual!r}")
        return
    if isinstance(expected, bool) and not isinstance(actual, bool):
        raise RuntimeError(f"{label} must be boolean, got {actual!r}")
    if actual != expected:
        raise RuntimeError(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def validate_builder_report(
    *,
    payload: dict[str, Any],
    report_path: Path,
    output_path: Path,
    expected_names: list[str],
    expected_offset: int,
    builder_args: dict[str, Any],
    member_records: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_count = len(expected_names)
    checks: list[tuple[Any, Any, str]] = [
        (payload.get("kind"), "assembly_v1_submission_report", "report kind"),
        (payload.get("offset"), expected_offset, "report offset"),
        (payload.get("limit"), expected_count, "report limit"),
        (payload.get("expected_count"), expected_count, "report expected_count"),
        (payload.get("count"), expected_count, "report count"),
        (payload.get("available_input_count"), 700, "available input count"),
        (payload.get("source_names"), expected_names, "report source_names"),
        (
            payload.get("pipeline", {}).get("mode"),
            "promoted_directional_qap",
            "pipeline mode",
        ),
        (
            payload.get("pipeline", {}).get("solver"),
            "directional_qap",
            "pipeline solver",
        ),
        (
            payload.get("pipeline", {}).get("seed_solver"),
            "soft_cycle_component_solver",
            "pipeline seed solver",
        ),
        (
            payload.get("anti_leakage", {}).get("target_paths_or_pixels_read"),
            False,
            "anti-leakage target read flag",
        ),
        (
            payload.get("archive", {}).get("member_order"),
            expected_names,
            "archive member order",
        ),
        (
            payload.get("archive", {}).get("flat_member_names"),
            True,
            "archive flat-member flag",
        ),
        (payload.get("output_sha256"), sha256(output_path), "reported output hash"),
        (payload.get("output_bytes"), output_path.stat().st_size, "reported output bytes"),
    ]
    configuration = payload.get("configuration", {})
    qap = configuration.get("qap", {})
    soft_cycle = configuration.get("soft_cycle", {})
    line_seam = configuration.get("line_seam", {})
    configured_fields = [
        (configuration.get("batch_size"), builder_args["batch_size"], "batch_size"),
        (configuration.get("chunk_size"), builder_args["chunk_size"], "chunk_size"),
        (line_seam.get("enabled"), builder_args["line_seam"], "line_seam.enabled"),
        (
            line_seam.get("auxiliary_weight"),
            builder_args["line_seam_auxiliary_weight"],
            "line_seam.auxiliary_weight",
        ),
        (
            line_seam.get("fusion_weight"),
            builder_args["line_seam_fusion_weight"],
            "line_seam.fusion_weight",
        ),
        (soft_cycle.get("score"), builder_args["soft_cycle_score"], "soft_cycle.score"),
        (soft_cycle.get("top_k"), builder_args["soft_cycle_topk"], "soft_cycle.top_k"),
        (
            soft_cycle.get("keep_per_tile"),
            builder_args["soft_cycle_keep_per_tile"],
            "soft_cycle.keep_per_tile",
        ),
        (
            soft_cycle.get("keep_fraction"),
            builder_args["soft_cycle_keep_fraction"],
            "soft_cycle.keep_fraction",
        ),
        (
            soft_cycle.get("loop_weight"),
            builder_args["soft_cycle_loop_weight"],
            "soft_cycle.loop_weight",
        ),
        (
            soft_cycle.get("reciprocal_weight"),
            builder_args["soft_cycle_reciprocal_weight"],
            "soft_cycle.reciprocal_weight",
        ),
        (qap.get("enabled"), True, "qap.enabled"),
        (qap.get("score"), builder_args["qap_score"], "qap.score"),
        (qap.get("iterations"), builder_args["qap_iterations"], "qap.iterations"),
        (qap.get("restarts"), builder_args["qap_restarts"], "qap.restarts"),
        (
            qap.get("initial_weight"),
            builder_args["qap_initial_weight"],
            "qap.initial_weight",
        ),
        (
            qap.get("noisy_components"),
            builder_args["qap_noisy_components"],
            "qap.noisy_components",
        ),
        (qap.get("noise_scale"), builder_args["qap_noise_scale"], "qap.noise_scale"),
        (
            qap.get("boundary_weight"),
            builder_args["qap_boundary_weight"],
            "qap.boundary_weight",
        ),
        (
            qap.get("refine_swaps"),
            builder_args["qap_refine_swaps"],
            "qap.refine_swaps",
        ),
        (
            qap.get("refine_weak_cells"),
            builder_args["qap_refine_weak_cells"],
            "qap.refine_weak_cells",
        ),
    ]
    for actual, expected, label in [*checks, *configured_fields]:
        _require_equal(actual, expected, label)

    sources = payload.get("sources")
    if not isinstance(sources, list) or len(sources) != expected_count:
        raise RuntimeError("builder report must contain one source record per image")
    layout_hashes: list[str] = []
    member_by_name = {record["name"]: record for record in member_records}
    if set(member_by_name) != set(expected_names) or len(member_by_name) != expected_count:
        raise RuntimeError("member records do not match builder report sources")
    for expected_name, source in zip(expected_names, sources, strict=True):
        if not isinstance(source, dict):
            raise RuntimeError("builder source record must be an object")
        _require_equal(source.get("source"), expected_name, "builder source name")
        _require_equal(source.get("pipeline"), "directional_qap", "source pipeline")
        _require_equal(
            source.get("output_png_sha256"),
            member_by_name[expected_name]["sha256"],
            "source/archive PNG hash",
        )
        layout = source.get("position_to_slot")
        if (
            not isinstance(layout, list)
            or len(layout) != 576
            or any(not isinstance(value, int) or isinstance(value, bool) for value in layout)
            or set(layout) != set(range(576))
        ):
            raise RuntimeError(f"invalid 576-tile permutation for {expected_name}")
        layout_hash = source.get("layout_sha256")
        if not isinstance(layout_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", layout_hash):
            raise RuntimeError(f"invalid layout hash for {expected_name}")
        layout_hashes.append(layout_hash)
    return {
        "report": str(report_path),
        "pipeline": "promoted_directional_qap",
        "source_count": expected_count,
        "layout_hashes_sha256": sha256_names(layout_hashes),
        "first_layout_sha256": layout_hashes[0],
    }


def run_shard(
    *,
    phase: str,
    shard_index: int,
    gpu: int,
    offset: int,
    limit: int,
    expected_names: list[str],
    builder: Path,
    code_root: Path,
    input_dir: Path,
    denoiser: Path,
    embedding_checkpoint: Path,
    renderer_denoiser: Path | None,
    additional_builder_assets: dict[str, Path],
    builder_args: dict[str, Any],
    supported: set[str],
) -> dict[str, Any]:
    if phase not in {"preflight", "shard"}:
        raise ValueError(f"unsupported batch phase: {phase!r}")
    stem = f"final_qap_{phase}_{offset:03d}_{offset + limit:03d}"
    output = WORKING / f"{stem}.zip"
    report = WORKING / f"{stem}.json"
    log = WORKING / f"{stem}.log"
    command = [
        sys.executable,
        str(builder),
        "--input-dir",
        str(input_dir),
        "--denoiser",
        str(denoiser),
        "--embedding-checkpoint",
        str(embedding_checkpoint),
        "--output",
        str(output),
        "--report",
        str(report),
        "--offset",
        str(offset),
        "--limit",
        str(limit),
        "--expected-count",
        str(limit),
        "--device",
        "cuda",
        "--overwrite",
    ]
    if renderer_denoiser is not None:
        if "--renderer-denoiser" not in supported:
            raise RuntimeError("configured renderer denoiser is unsupported by builder")
        command.extend(["--renderer-denoiser", str(renderer_denoiser)])
    for argument_name, asset_path in sorted(additional_builder_assets.items()):
        option = "--" + argument_name.replace("_", "-")
        if option not in supported:
            raise RuntimeError(f"configured builder asset is unsupported: {option}")
        command.extend([option, str(asset_path)])
    append_configured_builder_args(command, builder_args, supported)

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["PYTHONPATH"] = str(code_root / "src")
    environment["PYTHONHASHSEED"] = "0"
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    started = time.perf_counter()
    print(
        json.dumps(
            {
                "event": f"{phase}_start",
                "phase": phase,
                "shard": shard_index,
                "gpu": gpu,
                "offset": offset,
                "limit": limit,
                "command": command,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    with log.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=code_root,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while True:
            try:
                returncode = process.wait(timeout=60)
                break
            except subprocess.TimeoutExpired:
                print(
                    json.dumps(
                        {
                            "event": f"{phase}_heartbeat",
                            "phase": phase,
                            "shard": shard_index,
                            "gpu": gpu,
                            "seconds": time.perf_counter() - started,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    seconds = time.perf_counter() - started
    if returncode != 0:
        tail = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-80:])
        raise RuntimeError(
            f"shard {shard_index} failed with return code {returncode}; log tail:\n{tail}"
        )
    if not output.is_file() or not report.is_file():
        raise RuntimeError(f"shard {shard_index} did not produce zip and report")
    members = validate_zip(
        output,
        expected_names,
        deterministic_timestamp=BUILDER_ARCHIVE_TIMESTAMP,
    )
    report_payload = json.loads(report.read_text(encoding="utf-8"))
    report_contract = validate_builder_report(
        payload=report_payload,
        report_path=report,
        output_path=output,
        expected_names=expected_names,
        expected_offset=offset,
        builder_args=builder_args,
        member_records=members,
    )
    record = {
        "phase": phase,
        "shard": shard_index,
        "gpu": gpu,
        "offset": offset,
        "limit": limit,
        "source_names_sha256": sha256_names(sorted(expected_names)),
        "output": str(output),
        "output_sha256": sha256(output),
        "output_bytes": output.stat().st_size,
        "report": str(report),
        "report_sha256": sha256(report),
        "log": str(log),
        "log_sha256": sha256(log),
        "member_count": len(members),
        "members_sha256": sha256_bytes(
            json.dumps(members, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ),
        "first_member": members[0],
        "builder_report_contract": report_contract,
        "seconds": seconds,
    }
    print(
        json.dumps({"event": f"{phase}_complete", **record}, sort_keys=True),
        flush=True,
    )
    return record


def merge_shards(
    shard_paths: list[Path], expected_names: list[str], output: Path
) -> list[dict[str, Any]]:
    owners: dict[str, int] = {}
    archives = [zipfile.ZipFile(path) for path in shard_paths]
    try:
        for archive_index, archive in enumerate(archives):
            for name in archive.namelist():
                if name in owners:
                    raise RuntimeError(f"duplicate member across shards: {name}")
                owners[name] = archive_index
        if sorted(owners) != sorted(expected_names):
            raise RuntimeError("combined shard members differ from the 700 test inputs")
        temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
        if temporary.exists():
            temporary.unlink()
        try:
            with zipfile.ZipFile(
                temporary,
                "w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=6,
            ) as merged:
                for name in sorted(expected_names):
                    payload = archives[owners[name]].read(name)
                    validate_png(payload, name)
                    info = zipfile.ZipInfo(name, date_time=ARCHIVE_TIMESTAMP)
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.create_system = 3
                    info.external_attr = 0o100644 << 16
                    merged.writestr(info, payload, compresslevel=6)
            os.replace(temporary, output)
        finally:
            if temporary.exists():
                temporary.unlink()
    finally:
        for archive in archives:
            archive.close()
    return validate_zip(
        output,
        expected_names,
        deterministic_timestamp=ARCHIVE_TIMESTAMP,
    )


def verify_preflight_replay(
    preflight: dict[str, Any], first_shard: dict[str, Any], expected_name: str
) -> dict[str, Any]:
    preflight_member = preflight["first_member"]
    shard_member = first_shard["first_member"]
    _require_equal(preflight_member.get("name"), expected_name, "preflight member name")
    _require_equal(shard_member.get("name"), expected_name, "shard replay member name")
    _require_equal(
        shard_member.get("sha256"),
        preflight_member.get("sha256"),
        "preflight/shard PNG hash",
    )
    _require_equal(
        shard_member.get("bytes"),
        preflight_member.get("bytes"),
        "preflight/shard PNG byte count",
    )
    preflight_layout = preflight["builder_report_contract"]["first_layout_sha256"]
    shard_layout = first_shard["builder_report_contract"]["first_layout_sha256"]
    _require_equal(shard_layout, preflight_layout, "preflight/shard layout hash")
    return {
        "source": expected_name,
        "png_sha256": preflight_member["sha256"],
        "png_bytes": preflight_member["bytes"],
        "layout_sha256": preflight_layout,
        "byte_identical": True,
        "layout_identical": True,
    }


def deterministic_batch_summary(record: dict[str, Any]) -> dict[str, Any]:
    contract = record["builder_report_contract"]
    return {
        "phase": record["phase"],
        "shard": record["shard"],
        "gpu": record["gpu"],
        "offset": record["offset"],
        "limit": record["limit"],
        "source_names_sha256": record["source_names_sha256"],
        "archive": Path(record["output"]).name,
        "archive_sha256": record["output_sha256"],
        "archive_bytes": record["output_bytes"],
        "member_count": record["member_count"],
        "members_sha256": record["members_sha256"],
        "first_member": record["first_member"],
        "pipeline": contract["pipeline"],
        "layout_hashes_sha256": contract["layout_hashes_sha256"],
        "first_layout_sha256": contract["first_layout_sha256"],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    config, config_provenance = load_config()
    expected_total = int(config["expected_total"])
    shard_size = int(config["shard_size"])
    preflight_count = int(config["preflight_count"])
    if expected_total != 700 or shard_size != 350:
        raise RuntimeError("this final runner intentionally requires 700 images in 350+350 shards")
    if preflight_count != 1:
        raise RuntimeError("this final runner intentionally requires a one-image preflight")
    if int(config["require_gpu_count"]) != 2:
        raise RuntimeError("this final runner intentionally requires two GPUs")

    assets = config["assets"]
    denoiser_filename = require_filename(
        assets["denoiser_filename"], "denoiser_filename"
    )
    embedding_filename = require_filename(
        assets["embedding_checkpoint_filename"], "embedding_checkpoint_filename"
    )
    data_root = find_data_root()
    runtime_root = find_runtime_root(denoiser_filename, embedding_filename)
    code_root = find_code_root()
    input_dir = data_root / "test"
    denoiser = runtime_root / denoiser_filename
    embedding_checkpoint = runtime_root / embedding_filename
    renderer_name = assets.get("renderer_denoiser_filename")
    renderer_denoiser = (
        runtime_root / require_filename(renderer_name, "renderer_denoiser_filename")
        if renderer_name
        else None
    )
    additional_asset_config = assets.get("additional_builder_assets", {})
    if not isinstance(additional_asset_config, dict):
        raise RuntimeError("assets.additional_builder_assets must be an object")
    additional_builder_assets: dict[str, Path] = {}
    reserved_asset_arguments = {
        "input-dir",
        "denoiser",
        "embedding-checkpoint",
        "renderer-denoiser",
        "output",
        "report",
        "offset",
        "limit",
        "expected-count",
        "device",
        "overwrite",
    }
    for raw_argument, raw_filename in sorted(additional_asset_config.items()):
        argument = str(raw_argument).replace("_", "-")
        filename = require_filename(raw_filename, f"additional builder asset --{argument}")
        if not re.fullmatch(r"[a-z][a-z0-9-]*", argument):
            raise RuntimeError(f"invalid additional builder asset argument: {raw_argument!r}")
        if argument in reserved_asset_arguments or argument.replace("-", "_") in config["builder_args"]:
            raise RuntimeError(f"duplicate/reserved builder asset argument: {argument}")
        additional_builder_assets[argument] = runtime_root / filename
    builder = code_root / "scripts" / "build_assembly_submission.py"

    for path, label in [
        (denoiser, "denoiser"),
        (embedding_checkpoint, "embedding checkpoint"),
        (builder, "submission builder"),
        *[
            (path, f"additional builder asset --{argument}")
            for argument, path in sorted(additional_builder_assets.items())
        ],
    ]:
        if not path.is_file():
            raise FileNotFoundError(f"missing {label}: {path}")
    if renderer_denoiser is not None and not renderer_denoiser.is_file():
        raise FileNotFoundError(f"missing renderer denoiser: {renderer_denoiser}")

    source_paths = sorted(input_dir.glob("*.png"))
    source_names = [path.name for path in source_paths]
    if len(source_names) != expected_total or len(set(source_names)) != expected_total:
        raise RuntimeError(
            f"expected exactly {expected_total} unique test PNGs, found {len(source_names)}"
        )
    code_contract_records = validate_code_contract(code_root, config["code_contract"])
    supported = builder_options(builder, code_root)
    probe = hardware_probe(config)
    print(json.dumps({"event": "hardware", **probe}, sort_keys=True), flush=True)

    preflight = run_shard(
        phase="preflight",
        shard_index=-1,
        gpu=0,
        offset=0,
        limit=preflight_count,
        expected_names=source_names[:preflight_count],
        builder=builder,
        code_root=code_root,
        input_dir=input_dir,
        denoiser=denoiser,
        embedding_checkpoint=embedding_checkpoint,
        renderer_denoiser=renderer_denoiser,
        additional_builder_assets=additional_builder_assets,
        builder_args=config["builder_args"],
        supported=supported,
    )

    specs = [
        {
            "phase": "shard",
            "shard_index": 0,
            "gpu": 0,
            "offset": 0,
            "limit": shard_size,
            "expected_names": source_names[:shard_size],
        },
        {
            "phase": "shard",
            "shard_index": 1,
            "gpu": 1,
            "offset": shard_size,
            "limit": shard_size,
            "expected_names": source_names[shard_size:],
        },
    ]
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                run_shard,
                **spec,
                builder=builder,
                code_root=code_root,
                input_dir=input_dir,
                denoiser=denoiser,
                embedding_checkpoint=embedding_checkpoint,
                renderer_denoiser=renderer_denoiser,
                additional_builder_assets=additional_builder_assets,
                builder_args=config["builder_args"],
                supported=supported,
            )
            for spec in specs
        ]
        for future in as_completed(futures):
            records.append(future.result())
    records.sort(key=lambda item: int(item["shard"]))
    replay = verify_preflight_replay(preflight, records[0], source_names[0])

    submission = WORKING / "submission.zip"
    member_records = merge_shards(
        [Path(record["output"]) for record in records], source_names, submission
    )
    manifest = WORKING / "final_submission_manifest.json"
    manifest_payload = {
        "schema_version": 1,
        "kind": "vsos_final_submission_manifest",
        "archive": submission.name,
        "archive_sha256": sha256(submission),
        "archive_bytes": submission.stat().st_size,
        "member_count": len(member_records),
        "source_names_sha256": sha256_names(source_names),
        "members": member_records,
    }
    write_json(manifest, manifest_payload)

    report = WORKING / "final_qap_submission_report.json"
    asset_paths = [denoiser, embedding_checkpoint]
    if renderer_denoiser is not None:
        asset_paths.append(renderer_denoiser)
    asset_paths.extend(additional_builder_assets.values())
    deterministic_assets = [
        {
            "role": (
                "scoring_denoiser"
                if renderer_denoiser is not None
                else "scoring_and_render_denoiser"
            ),
            "filename": denoiser.name,
            "bytes": denoiser.stat().st_size,
            "sha256": sha256(denoiser),
        },
        {
            "role": "side_embedding",
            "filename": embedding_checkpoint.name,
            "bytes": embedding_checkpoint.stat().st_size,
            "sha256": sha256(embedding_checkpoint),
        },
    ]
    if renderer_denoiser is not None:
        deterministic_assets.append(
            {
                "role": "separate_renderer_denoiser",
                "filename": renderer_denoiser.name,
                "bytes": renderer_denoiser.stat().st_size,
                "sha256": sha256(renderer_denoiser),
            }
        )
    deterministic_assets.extend(
        {
            "role": f"builder_asset:--{argument}",
            "filename": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for argument, path in sorted(additional_builder_assets.items())
    )
    deterministic_report_payload = {
        "schema_version": 1,
        "kind": "vsos_final_qap_submission_report",
        "deterministic": True,
        "config": config,
        "config_sha256": canonical_config_sha256(config),
        "code": {
            "dataset_slug": config["code_contract"].get("dataset_slug"),
            "expected_dataset_version": config["code_contract"].get(
                "expected_dataset_version"
            ),
            "contract": code_contract_records,
            "builder": "scripts/build_assembly_submission.py",
            "builder_sha256": sha256(builder),
            "qap_sha256": sha256(code_root / "src" / "puzzle_assembly" / "qap.py"),
        },
        "assets": deterministic_assets,
        "source_count": len(source_names),
        "source_names": source_names,
        "source_names_sha256": sha256_names(source_names),
        "preflight": deterministic_batch_summary(preflight),
        "deterministic_replay": replay,
        "shards": [deterministic_batch_summary(record) for record in records],
        "archive_contract": {
            "member_count": 700,
            "unique_root_level_pngs": True,
            "image_mode": "RGB",
            "image_size": [480, 480],
            "final_timestamp": list(ARCHIVE_TIMESTAMP),
            "builder_batch_timestamp": list(BUILDER_ARCHIVE_TIMESTAMP),
            "compression": "ZIP_DEFLATED",
            "compresslevel": 6,
            "unix_mode": "100644",
        },
        "submission": {
            "path": submission.name,
            "bytes": submission.stat().st_size,
            "sha256": sha256(submission),
        },
        "manifest": {
            "path": manifest.name,
            "bytes": manifest.stat().st_size,
            "sha256": sha256(manifest),
        },
    }
    write_json(report, deterministic_report_payload)

    run_report = WORKING / "final_qap_submission_run.json"
    run_report_payload = {
        "schema_version": 1,
        "kind": "vsos_final_qap_submission_run",
        "config_sha256": canonical_config_sha256(config),
        "config_provenance": config_provenance,
        "hardware": probe,
        "mounts": {
            "data_root": str(data_root),
            "runtime_root": str(runtime_root),
            "code_root": str(code_root),
        },
        "code_contract": code_contract_records,
        "assets": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in asset_paths
        ],
        "preflight": preflight,
        "deterministic_replay": replay,
        "shards": records,
        "submission": {
            "path": str(submission),
            "bytes": submission.stat().st_size,
            "sha256": sha256(submission),
        },
        "manifest": {
            "path": str(manifest),
            "bytes": manifest.stat().st_size,
            "sha256": sha256(manifest),
        },
        "deterministic_report": {
            "path": str(report),
            "bytes": report.stat().st_size,
            "sha256": sha256(report),
        },
        "seconds": time.perf_counter() - started,
    }
    write_json(run_report, run_report_payload)

    hashes = WORKING / "final_artifact_hashes.json"
    deterministic_hash_paths = [
        submission,
        manifest,
        report,
        Path(preflight["output"]),
        *(Path(record["output"]) for record in records),
    ]
    hash_payload = {
        "schema_version": 1,
        "kind": "vsos_final_artifact_hashes",
        "deterministic": True,
        "artifacts": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in deterministic_hash_paths
        ],
    }
    write_json(hashes, hash_payload)
    run_hashes = WORKING / "final_run_artifact_hashes.json"
    run_hash_paths = [
        run_report,
        *(Path(preflight[key]) for key in ("report", "log")),
        *(Path(record[key]) for record in records for key in ("report", "log")),
    ]
    run_hash_payload = {
        "schema_version": 1,
        "kind": "vsos_final_run_artifact_hashes",
        "deterministic": False,
        "artifacts": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in run_hash_paths
        ],
    }
    write_json(run_hashes, run_hash_payload)
    sums_items = [
        *hash_payload["artifacts"],
        *run_hash_payload["artifacts"],
        {
            "path": hashes.name,
            "bytes": hashes.stat().st_size,
            "sha256": sha256(hashes),
        },
        {
            "path": run_hashes.name,
            "bytes": run_hashes.stat().st_size,
            "sha256": sha256(run_hashes),
        },
    ]
    sums = WORKING / "SHA256SUMS.txt"
    sums.write_text(
        "".join(
            f"{item['sha256']}  {item['path']}\n" for item in sums_items
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "event": "final_submission_complete",
                "submission": str(submission),
                "submission_sha256": sha256(submission),
                "manifest_sha256": sha256(manifest),
                "report_sha256": sha256(report),
                "run_report_sha256": sha256(run_report),
                "artifact_hashes_sha256": sha256(hashes),
                "run_artifact_hashes_sha256": sha256(run_hashes),
                "sha256sums_sha256": sha256(sums),
                "member_count": len(member_records),
                "preflight_replay": replay,
                "seconds": time.perf_counter() - started,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        failure = {
            "schema_version": 1,
            "kind": "vsos_final_qap_submission_failure",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        write_json(WORKING / "final_qap_submission_failure.json", failure)
        raise

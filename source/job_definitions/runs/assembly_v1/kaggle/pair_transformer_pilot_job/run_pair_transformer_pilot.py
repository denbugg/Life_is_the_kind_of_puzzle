#!/usr/bin/env python3
"""Fail-closed staging, T4x2 preflight, smoke, and Pair Transformer pilot."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback
import zipfile


INPUT = Path("/kaggle/input")
DATASET_INPUT = INPUT / "datasets" / "pasha883"
WORKING = Path("/kaggle/working")
WRAPPER_PATH = WORKING / "pair_transformer_pilot_wrapper.json"
PUZZLE_INPUT = DATASET_INPUT / "vsos-ai-initiative-pazzle"
RUNTIME_INPUT = DATASET_INPUT / "vsos-assembly-v1-runtime"
BASE_INPUT = DATASET_INPUT / "vsos-solver-rework-night-code"
PSEUDO_INPUT = DATASET_INPUT / "vsos-real-gold-512"
OVERLAY_INPUT = DATASET_INPUT / "vsos-pair-transformer-pilot-code"
EXPECTED_TEST_COUNT = 22
CHECKPOINT_KIND = "puzzle_full_tile_pair_transformer"
ALLOWED_REPORT_STATUS = frozenset({"continue", "stop_or_redesign"})
EXPECTED_GATE_KEYS = frozenset(
    {
        "aggregate_recall_at_1_delta_ge_0.02",
        "aggregate_recall_at_32_delta_ge_minus_0.005",
        "aggregate_softcycle_adjacency_delta_ge_0.01",
        "aggregate_qap_adjacency_delta_vs_no_neural_envelope_ge_0.01",
        "aggregate_qap_ssim_delta_vs_no_neural_envelope_ge_0.005",
        "every_panel_replica_positive_r1",
        "every_panel_replica_positive_qap_ssim_vs_no_neural_envelope",
    }
)
PILOT_MODEL_CONFIG = {
    "model_dim": 512,
    "layers": 8,
    "heads": 8,
    "feedforward_dim": 2048,
    "cnn_channels": 128,
    "patch_grid": 5,
    "side_band": 6,
    "band_tokens": 10,
    "dropout": 0.1,
    "gradient_checkpointing": True,
}
SMOKE_MODEL_CONFIG = {
    **PILOT_MODEL_CONFIG,
    "model_dim": 64,
    "layers": 2,
    "heads": 4,
    "feedforward_dim": 128,
    "cnn_channels": 32,
    "patch_grid": 3,
    "band_tokens": 4,
}
CHECKSUM_ARTIFACTS = (
    "pair_transformer_report.json",
    "pair_transformer_calibrated.pt",
    "pair_transformer_best.pt",
    "pair_transformer_latest.pt",
)
EXPECTED_BASE_ARCHIVE_SHA256 = (
    "a980c158fb349fbc8619e39eb829acdc675e7332d1ec3995c08f38eb49f45d0c"
)
EXPECTED_OVERLAY_ARCHIVE_SHA256 = (
    "f9572a559fd3d536f6a01a51dd46333e55a9d1cc5c1ea53c49ffcb7152dbc6f4"
)
EXPECTED_RUNTIME_ASSET_SHA256 = {
    "pseudo_gold": "70d0b7b3c15fefac62d6d1bf554f0e50a0f3473ddaf01423607b78cf0cde90c2",
    "denoiser": "77a2d8607c9bc6b80e7aa99ed03329c1fae6bf94ee1d9a654241ab93a05cc734",
    "hbt": "c2589bff58573d592227d522954a7ed6c22f6fec7ecbbea2bf0c755d7a720787",
    "manifest": "de4fd2e596efa0d157d2d4480eed5fb84812d358138a1db53c1706bfb580e345",
    "quarantine": "38dfd12f60579d77999c0cdb4a648fb4ff0343a8fb5e4c421f0b29f8b7bd6215",
}
TRANSITIVE_CODE_PATHS = (
    "scripts/train_evaluate_pair_transformer.py",
    "src/puzzle_assembly/__init__.py",
    "src/puzzle_assembly/pair_transformer.py",
    "src/puzzle_assembly/compatibility.py",
    "src/puzzle_assembly/components.py",
    "src/puzzle_assembly/geometry.py",
    "src/puzzle_assembly/learned.py",
    "src/puzzle_assembly/metrics.py",
    "src/puzzle_assembly/panels.py",
    "src/puzzle_assembly/protocol.py",
    "src/puzzle_assembly/qap.py",
    "src/puzzle_assembly/solvers.py",
    "src/puzzle_denoise_v2/__init__.py",
    "src/puzzle_denoise_v2/degradation.py",
    "src/puzzle_denoise_v2/inference.py",
    "src/puzzle_denoise_v2/losses.py",
    "src/puzzle_denoise_v2/metrics.py",
    "src/puzzle_denoise_v2/model.py",
    "src/puzzle_denoise_v2/tiles.py",
    "src/puzzle_denoise_v2/training.py",
)
EXPECTED_TRANSITIVE_CODE_SHA256 = {
    "scripts/train_evaluate_pair_transformer.py": "37f4585ff42eeeaf31a8142f2068d86debe1bc169358cc2f21277fd7aa6f3473",
    "src/puzzle_assembly/__init__.py": "09e051b7555471aafca03cd666d789f033aca47f1c82f6e2af9c0cce50afe9d5",
    "src/puzzle_assembly/pair_transformer.py": "99e14f3741528cf277a5b10fb0a01fac761debc5370fd55fd81f6235a0ae303b",
    "src/puzzle_assembly/compatibility.py": "aff2149b161c4fded4e5d91fbea49a8a62967886148d3ad374467331e0416a9f",
    "src/puzzle_assembly/components.py": "53fcc7c4fd23956db884ee45060e47f8e94a931c16e497e426d67549621bd367",
    "src/puzzle_assembly/geometry.py": "1e16bec6fb98a33060558d5d28062334d9114b12424733ef103a40393ef1ba86",
    "src/puzzle_assembly/learned.py": "9e3dba673aa85eaab5698dbeb63b3d94f88e3ea92b5e5979bde4b0273642697b",
    "src/puzzle_assembly/metrics.py": "84857ef92c382cc0964c21bfec67c13308014a1674aebf8686b17514784dae69",
    "src/puzzle_assembly/panels.py": "783356628517e3a23b8703672bca604c3d879c875f5b5f35f87182425500280f",
    "src/puzzle_assembly/protocol.py": "b711ad6d28a2fe60329e3e8236e58adbfbceea8ca4c8bf85e9a057e7619e24f4",
    "src/puzzle_assembly/qap.py": "b8a5e1da67387fd04effd979270ca16925aceab23d37083eda108c6e3e349c32",
    "src/puzzle_assembly/solvers.py": "23f9e32200748349d0da8558b7b44053a758e1c1eb306d8f31ce59feae03fe8e",
    "src/puzzle_denoise_v2/__init__.py": "30849e0f937ba4a50e85ce2eee0d2b930db06fbcc0b7dff84547e121ef2f30b7",
    "src/puzzle_denoise_v2/degradation.py": "7e314081c143a1c7846a9777eaea8716092a85595f856769efd3704a2c583a75",
    "src/puzzle_denoise_v2/inference.py": "20767cc26270cfde7472cf33a0247b1ea6d96e5b5c8ff5d705b785ae710dd6da",
    "src/puzzle_denoise_v2/losses.py": "56776289cd51e49a28ce54bc4762d144d87c7efbf6d4ca56668fc3b019dbbf34",
    "src/puzzle_denoise_v2/metrics.py": "e8275fb096276a63b7114be1a74b24009dc2143dddf299ce5eaceac401a27d36",
    "src/puzzle_denoise_v2/model.py": "37db32fb83ece0f122757bdbec19ffc6a17c5e5e00ef92a26328247d95c55d11",
    "src/puzzle_denoise_v2/tiles.py": "21270e283e50ea0b155ef194de889222fb0c4f6954437eb1526342c006eefaa7",
    "src/puzzle_denoise_v2/training.py": "6719ee6a62434cd8a00fafb92b28f6a10941cdbf5c83573fc6556b33e5eba56e",
}
EXPECTED_OVERLAY_SHA256 = {
    "src/puzzle_assembly/pair_transformer.py": "99e14f3741528cf277a5b10fb0a01fac761debc5370fd55fd81f6235a0ae303b",
    "scripts/train_evaluate_pair_transformer.py": "37f4585ff42eeeaf31a8142f2068d86debe1bc169358cc2f21277fd7aa6f3473",
    "tests/test_pair_transformer.py": "2a66d5b588b43f8d56b12b4faffeec262ddef746eabf8dab0ece0049e8343b7d",
}


def one(paths: list[Path], label: str) -> Path:
    values = sorted(set(path.resolve() for path in paths))
    if len(values) != 1:
        raise RuntimeError(f"expected exactly one {label}, found {values}")
    return values[0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def require_exact_hashes(
    actual: dict[str, str], expected: dict[str, str], label: str
) -> None:
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        changed = sorted(
            key for key in set(actual) & set(expected) if actual[key] != expected[key]
        )
        changed_hashes = {
            key: {"expected": expected[key], "actual": actual[key]}
            for key in changed
        }
        raise RuntimeError(
            f"{label} SHA256 pin mismatch: missing={missing}, extra={extra}, "
            f"changed={changed_hashes}"
        )


def safe_extract(archive: Path, destination: Path) -> Path:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    with zipfile.ZipFile(archive) as handle:
        names = handle.namelist()
        if any(name.startswith("/") or ".." in Path(name).parts for name in names):
            raise RuntimeError(f"unsafe member in {archive}")
        handle.extractall(destination)
    return destination


def require_mount(path: Path, label: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"missing exact Kaggle {label} mount: {path}")
    return path


def find_data_root() -> Path:
    return one(
        [
            path.parent.parent
            for path in require_mount(PUZZLE_INPUT, "puzzle").glob("**/train/inputs")
            if path.is_dir()
            and (path.parent / "targets").is_dir()
            and len(list((path.parent / "targets").glob("*.png"))) == 7000
        ],
        "puzzle data root",
    )


def find_runtime_root() -> Path:
    return one(
        [
            path.parent
            for path in require_mount(RUNTIME_INPUT, "runtime").glob(
                "**/selected_tilenaf_synth_50k.pt"
            )
            if (path.parent / "hbt_d320_denoised_rgb_sobel.pt").is_file()
        ],
        "assembly runtime asset root",
    )


def find_base_root() -> tuple[Path, Path | None]:
    direct = sorted(
        {
            path.parent.parent.parent
            for path in require_mount(BASE_INPUT, "solver-rework").glob(
                "**/src/puzzle_assembly/qap.py"
            )
            if (
                path.parent.parent.parent
                / "configs"
                / "denoise_splits_seed20260710.json"
            ).is_file()
            and (
                path.parent.parent.parent
                / "src"
                / "puzzle_denoise_v2"
                / "inference.py"
            ).is_file()
        }
    )
    archives = sorted(BASE_INPUT.glob("**/solver_rework_code.zip"))
    archive = one(archives, "solver-rework base archive") if archives else None
    if len(direct) == 1:
        return direct[0], archive
    if direct:
        raise RuntimeError(f"ambiguous direct solver-rework roots: {direct}")
    if archive is None:
        raise RuntimeError("solver-rework base code is neither direct nor archived")
    extracted = safe_extract(archive, WORKING / "pair_base_extracted")
    if not (extracted / "src" / "puzzle_assembly" / "qap.py").is_file():
        raise RuntimeError("solver-rework archive lacks qap.py")
    return extracted, archive


def find_overlay_root() -> tuple[Path, Path | None]:
    direct = sorted(
        {
            path.parent.parent
            for path in require_mount(OVERLAY_INPUT, "pair overlay").glob(
                "**/scripts/train_evaluate_pair_transformer.py"
            )
            if (
                path.parent.parent
                / "src"
                / "puzzle_assembly"
                / "pair_transformer.py"
            ).is_file()
            and (path.parent.parent / "tests" / "test_pair_transformer.py").is_file()
        }
    )
    archives = sorted(OVERLAY_INPUT.glob("**/pair_transformer_code.zip"))
    archive = one(archives, "Pair Transformer overlay archive") if archives else None
    if len(direct) == 1:
        return direct[0], archive
    if direct:
        raise RuntimeError(f"ambiguous direct Pair Transformer overlays: {direct}")
    if archive is None:
        raise RuntimeError("Pair Transformer overlay is neither direct nor archived")
    extracted = safe_extract(archive, WORKING / "pair_overlay_extracted")
    if not (
        extracted / "src" / "puzzle_assembly" / "pair_transformer.py"
    ).is_file():
        raise RuntimeError("Pair Transformer archive lacks model source")
    return extracted, archive


def find_pseudo_gold() -> Path:
    return one(
        list(require_mount(PSEUDO_INPUT, "real-gold-512").glob(
            "**/real_gold_train_512.npz"
        )),
        "real-gold-512 archive",
    )


def copy_code(base: Path, overlay: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    shutil.copytree(base / "src", destination / "src")
    shutil.copytree(base / "configs", destination / "configs")
    for relative in EXPECTED_OVERLAY_SHA256:
        source = overlay / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    runner_target = (
        destination
        / "runs/assembly_v1/kaggle/pair_transformer_pilot_job"
        / "run_pair_transformer_pilot.py"
    )
    runner_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), runner_target)


def verify_overlay(code_root: Path) -> dict[str, str]:
    actual = {
        relative: sha256(code_root / relative)
        for relative in EXPECTED_OVERLAY_SHA256
    }
    require_exact_hashes(actual, EXPECTED_OVERLAY_SHA256, "Pair Transformer overlay")
    return actual


def verify_staged_inputs(
    *,
    code_root: Path,
    base_archive: Path | None,
    overlay_archive: Path | None,
    pseudo_gold: Path,
    denoiser: Path,
    hbt: Path,
    manifest: Path,
    quarantine: Path,
) -> dict[str, object]:
    transitive = {
        relative: sha256(code_root / relative) for relative in TRANSITIVE_CODE_PATHS
    }
    require_exact_hashes(
        transitive,
        EXPECTED_TRANSITIVE_CODE_SHA256,
        "Pair Transformer transitive code",
    )
    assets = {
        "pseudo_gold": sha256(pseudo_gold),
        "denoiser": sha256(denoiser),
        "hbt": sha256(hbt),
        "manifest": sha256(manifest),
        "quarantine": sha256(quarantine),
    }
    require_exact_hashes(assets, EXPECTED_RUNTIME_ASSET_SHA256, "runtime assets")
    archive_hashes: dict[str, str | None] = {
        "base": None if base_archive is None else sha256(base_archive),
        "overlay": None if overlay_archive is None else sha256(overlay_archive),
    }
    if (
        archive_hashes["base"] is not None
        and archive_hashes["base"] != EXPECTED_BASE_ARCHIVE_SHA256
    ):
        raise RuntimeError(
            "solver-rework base archive SHA256 pin mismatch: "
            f"expected={EXPECTED_BASE_ARCHIVE_SHA256}, "
            f"actual={archive_hashes['base']}"
        )
    if (
        archive_hashes["overlay"] is not None
        and archive_hashes["overlay"] != EXPECTED_OVERLAY_ARCHIVE_SHA256
    ):
        raise RuntimeError(
            "Pair Transformer overlay archive SHA256 pin mismatch: "
            f"expected={EXPECTED_OVERLAY_ARCHIVE_SHA256}, "
            f"actual={archive_hashes['overlay']}"
        )
    return {
        "transitive_code_sha256": transitive,
        "runtime_assets_sha256": assets,
        "archive_sha256": archive_hashes,
    }


def hardware_probe() -> dict[str, object]:
    subprocess.run(["nvidia-smi"], check=True)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    count = torch.cuda.device_count()
    if count != 2:
        raise RuntimeError(f"Pair pilot requires exactly two GPUs, found {count}")
    devices: list[dict[str, object]] = []
    for index in range(count):
        device = torch.device(f"cuda:{index}")
        name = torch.cuda.get_device_name(index)
        capability = tuple(torch.cuda.get_device_capability(index))
        if "T4" not in name.upper() or capability != (7, 5):
            raise RuntimeError(
                f"GPU {index} violates NvidiaTeslaT4 pin: {name}, {capability}"
            )
        torch.cuda.reset_peak_memory_stats(device)
        generator = torch.Generator(device=device).manual_seed(20260711 + index)
        left = torch.randn(
            1024,
            1024,
            device=device,
            dtype=torch.float16,
            generator=generator,
            requires_grad=True,
        )
        right = torch.randn(
            1024,
            1024,
            device=device,
            dtype=torch.float16,
            generator=generator,
            requires_grad=True,
        )
        product = left @ right
        loss = product.float().square().mean()
        loss.backward()
        if (
            product.dtype != torch.float16
            or not torch.isfinite(product).all()
            or left.grad is None
            or right.grad is None
            or not torch.isfinite(left.grad).all()
            or not torch.isfinite(right.grad).all()
        ):
            raise RuntimeError(f"GPU {index} failed real fp16 forward/backward")
        torch.cuda.synchronize(device)
        devices.append(
            {
                "index": index,
                "name": name,
                "capability": list(capability),
                "total_memory": int(torch.cuda.get_device_properties(index).total_memory),
                "fp16_matmul_shape": [1024, 1024, 1024],
                "fp16_product_mean": float(product.float().mean().item()),
                "fp16_loss": float(loss.item()),
                "fp16_left_grad_mean": float(left.grad.float().mean().item()),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            }
        )
        del left, right, product, loss
        torch.cuda.empty_cache()
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "arch_list": torch.cuda.get_arch_list(),
        "device_count": count,
        "devices": devices,
    }


def run_checked(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    label: str,
    capture: bool = False,
    telemetry: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            check=False,
            capture_output=capture,
            text=capture,
        )
    except BaseException as error:
        record = {
            "label": label,
            "status": "error",
            "command": command,
            "returncode": None,
            "seconds": time.perf_counter() - started,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        if telemetry is not None:
            telemetry.append(record)
        raise
    seconds = time.perf_counter() - started
    if capture:
        if completed.stdout:
            print(completed.stdout, end="", flush=True)
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr, flush=True)
    record: dict[str, object] = {
        "label": label,
        "status": "passed" if completed.returncode == 0 else "failed",
        "command": command,
        "returncode": completed.returncode,
        "seconds": seconds,
    }
    if capture:
        record["stdout_tail"] = (completed.stdout or "")[-4000:]
        record["stderr_tail"] = (completed.stderr or "")[-4000:]
    if telemetry is not None:
        telemetry.append(record)
    if completed.returncode != 0:
        raise RuntimeError(f"{label} failed with return code {completed.returncode}")
    return record


def pair_transformer_class(model_source: Path):
    if not model_source.is_file():
        raise RuntimeError(f"staged Pair Transformer model source is missing: {model_source}")
    src_root = model_source.parents[1]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    module = importlib.import_module("puzzle_assembly.pair_transformer")
    loaded_source = Path(module.__file__).resolve()
    if loaded_source != model_source.resolve():
        raise RuntimeError(
            f"Pair Transformer class resolved from {loaded_source}, expected {model_source}"
        )
    return module.PairTransformerScorer


def _nonnegative_integer(value: object) -> bool:
    return type(value) is int and int(value) >= 0


def validate_latest_resume_state(
    payload: dict[str, object],
    *,
    label: str,
    expected_world_size: int,
    report_provenance: dict[str, object],
    report_history: object,
    best_checkpoint_sha256: str,
) -> dict[str, object]:
    required = {
        "optimizer_state",
        "scaler_state",
        "scheduler_state",
        "training_state",
    }
    missing = required - set(payload)
    if missing or any(not isinstance(payload.get(key), dict) for key in required):
        raise RuntimeError(f"{label} latest resume bundle is incomplete: {sorted(missing)}")

    optimizer_state = payload["optimizer_state"]
    scaler_state = payload["scaler_state"]
    scheduler_state = payload["scheduler_state"]
    state = payload["training_state"]
    assert isinstance(optimizer_state, dict)
    assert isinstance(scaler_state, dict)
    assert isinstance(scheduler_state, dict)
    assert isinstance(state, dict)
    if not isinstance(optimizer_state.get("state"), dict) or not isinstance(
        optimizer_state.get("param_groups"), list
    ) or not optimizer_state["param_groups"]:
        raise RuntimeError(f"{label} optimizer state schema is invalid")
    if (
        not isinstance(scheduler_state.get("base_lrs"), list)
        or not scheduler_state["base_lrs"]
        or not _nonnegative_integer(scheduler_state.get("last_epoch"))
        or not _nonnegative_integer(scheduler_state.get("_step_count"))
    ):
        raise RuntimeError(f"{label} scheduler state schema is invalid")

    metadata = payload["metadata"]
    assert isinstance(metadata, dict)
    report_resume = report_provenance.get("resume_contract")
    saved_resume = metadata.get("resume_contract")
    if not isinstance(report_resume, dict) or saved_resume != report_resume:
        raise RuntimeError(f"{label} latest resume provenance disagrees with report")
    for key in (
        "schema_version",
        "kind",
        "seed",
        "code",
        "whole_source_splits",
    ):
        if metadata.get(key) != report_provenance.get(key):
            raise RuntimeError(f"{label} latest provenance field {key!r} disagrees")
    saved_assets = metadata.get("assets")
    report_assets = report_provenance.get("assets")
    if not isinstance(saved_assets, dict) or not isinstance(report_assets, dict):
        raise RuntimeError(f"{label} latest asset provenance is missing")
    saved_pseudo = saved_assets.get("pseudo")
    report_pseudo = report_assets.get("pseudo")
    if (
        saved_assets.get("denoiser_sha256")
        != report_assets.get("denoiser_sha256")
        or saved_assets.get("hbt_sha256") != report_assets.get("hbt_sha256")
        or not isinstance(saved_pseudo, dict)
        or not isinstance(report_pseudo, dict)
        or saved_pseudo.get("sha256") != report_pseudo.get("sha256")
    ):
        raise RuntimeError(f"{label} latest asset SHA256 provenance disagrees")
    if (
        metadata.get("schema_version") != 1
        or metadata.get("kind") != "pair_transformer_training_provenance"
    ):
        raise RuntimeError(f"{label} latest training provenance kind/schema is invalid")

    trajectory = report_resume.get("trajectory_arguments")
    runtime_contracts = report_resume.get("runtime_contracts_by_rank")
    if not isinstance(trajectory, dict) or not isinstance(runtime_contracts, list):
        raise RuntimeError(f"{label} resume contract lacks trajectory/runtime provenance")
    if len(runtime_contracts) != expected_world_size:
        raise RuntimeError(f"{label} runtime contract rank count is invalid")
    ranks: list[int] = []
    for contract in runtime_contracts:
        if not isinstance(contract, dict) or not _nonnegative_integer(contract.get("rank")):
            raise RuntimeError(f"{label} per-rank runtime contract is invalid")
        ranks.append(int(contract["rank"]))
        if (
            contract.get("device_type") != "cuda"
            or contract.get("amp") is not True
            or contract.get("amp_dtype") != "torch.float16"
            or contract.get("capability") != [7, 5]
            or "T4" not in str(contract.get("gpu", "")).upper()
        ):
            raise RuntimeError(f"{label} runtime contract is not the pinned fp16 T4 path")
    if sorted(ranks) != list(range(expected_world_size)):
        raise RuntimeError(f"{label} runtime contract ranks are duplicated or missing")

    max_amp_skips = trajectory.get("max_amp_skips")
    amp_init_scale = trajectory.get("amp_init_scale")
    if (
        not _nonnegative_integer(max_amp_skips)
        or not isinstance(amp_init_scale, (int, float))
        or isinstance(amp_init_scale, bool)
        or not math.isfinite(float(amp_init_scale))
        or float(amp_init_scale) <= 0.0
        or trajectory.get("no_amp") is not False
    ):
        raise RuntimeError(f"{label} AMP provenance is invalid")
    required_scaler = {
        "scale",
        "growth_factor",
        "backoff_factor",
        "growth_interval",
        "_growth_tracker",
    }
    if set(scaler_state) != required_scaler or not all(
        isinstance(scaler_state[key], (int, float))
        and not isinstance(scaler_state[key], bool)
        for key in required_scaler
    ):
        raise RuntimeError(f"{label} GradScaler state schema is invalid")
    if (
        float(scaler_state["scale"]) <= 0.0
        or float(scaler_state["growth_factor"]) <= 1.0
        or not 0.0 < float(scaler_state["backoff_factor"]) < 1.0
        or int(scaler_state["growth_interval"]) <= 0
        or int(scaler_state["_growth_tracker"]) < 0
    ):
        raise RuntimeError(f"{label} GradScaler counters are invalid")

    world_size = state.get("world_size")
    successful = state.get("optimizer_steps")
    attempted = state.get("attempted_steps")
    skipped = state.get("amp_skips")
    if (
        world_size != expected_world_size
        or not _nonnegative_integer(successful)
        or not _nonnegative_integer(attempted)
        or not _nonnegative_integer(skipped)
        or int(attempted) != int(successful) + int(skipped)
        or int(skipped) > int(max_amp_skips)
    ):
        raise RuntimeError(f"{label} resume counters/world_size are incoherent")
    cursor = state.get("cursor")
    if not isinstance(cursor, dict):
        raise RuntimeError(f"{label} resume cursor is missing")
    completed_epoch = cursor.get("completed_epoch")
    next_epoch = cursor.get("next_epoch")
    if (
        not _nonnegative_integer(completed_epoch)
        or not _nonnegative_integer(next_epoch)
        or int(next_epoch) != int(completed_epoch) + 1
        or cursor.get("source_index") != 0
        or cursor.get("pseudo_cursor") != 0
        or cursor.get("capture_point") != "epoch_boundary"
        or metadata.get("latest_completed_epoch") != next_epoch
        or trajectory.get("epochs") != next_epoch
    ):
        raise RuntimeError(f"{label} resume epoch/cursor is incoherent")
    history = state.get("history")
    if (
        not isinstance(history, list)
        or len(history) != int(next_epoch)
        or metadata.get("training_history") != history
        or report_history != history
    ):
        raise RuntimeError(f"{label} cumulative history is incoherent")

    rng_states = state.get("rng_states_by_rank")
    generator_states = state.get("generator_states_by_rank")
    if (
        not isinstance(rng_states, list)
        or len(rng_states) != expected_world_size
        or not isinstance(generator_states, list)
        or len(generator_states) != expected_world_size
    ):
        raise RuntimeError(f"{label} per-rank RNG state count is invalid")
    import torch

    for rng_state in rng_states:
        if (
            not isinstance(rng_state, dict)
            or not {"python", "numpy", "torch_cpu", "torch_cuda"} <= set(rng_state)
            or not isinstance(rng_state["torch_cpu"], torch.Tensor)
            or rng_state["torch_cpu"].dtype != torch.uint8
            or rng_state["torch_cpu"].numel() == 0
            or not isinstance(rng_state["torch_cuda"], list)
            or not rng_state["torch_cuda"]
        ):
            raise RuntimeError(f"{label} per-rank RNG payload is invalid")
    if any(
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.uint8
        or value.numel() == 0
        for value in generator_states
    ):
        raise RuntimeError(f"{label} augmentation Generator payload is invalid")
    if state.get("best_checkpoint_sha256") != best_checkpoint_sha256:
        raise RuntimeError(f"{label} latest state disagrees with best checkpoint hash")
    best_delta = state.get("best_delta")
    if (
        not isinstance(best_delta, (int, float))
        or isinstance(best_delta, bool)
        or not math.isfinite(float(best_delta))
    ):
        raise RuntimeError(f"{label} best-selection state is invalid")
    return {
        "world_size": expected_world_size,
        "completed_epoch": int(completed_epoch),
        "next_epoch": int(next_epoch),
        "successful_optimizer_steps": int(successful),
        "attempted_optimizer_steps": int(attempted),
        "amp_skips": int(skipped),
        "best_checkpoint_sha256": best_checkpoint_sha256,
    }


def checkpoint_contract(
    path: Path,
    label: str,
    *,
    model_class,
    expected_model_config: dict[str, object],
    require_resume_state: bool = False,
    expected_world_size: int = 2,
    report_provenance: dict[str, object] | None = None,
    report_history: object = None,
    best_checkpoint_sha256: str = "",
) -> dict[str, object]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("kind") != CHECKPOINT_KIND
        or payload.get("safe_for_submission") is not False
        or not isinstance(payload.get("model_config"), dict)
        or not isinstance(payload.get("model_state"), dict)
        or not isinstance(metadata, dict)
        or metadata.get("safe_for_submission") is not False
    ):
        raise RuntimeError(f"{label} checkpoint violates schema/safety contract")
    if payload["model_config"] != expected_model_config:
        raise RuntimeError(f"{label} checkpoint model config differs from report contract")
    try:
        model = model_class(**expected_model_config)
        model.load_state_dict(payload["model_state"], strict=True)
    except Exception as error:
        raise RuntimeError(f"{label} checkpoint model state is not strictly loadable") from error
    result: dict[str, object] = {
        "schema_version": 1,
        "kind": CHECKPOINT_KIND,
        "safe_for_submission": False,
        "sha256": sha256(path),
        "model_config": expected_model_config,
        "model_state_strictly_loaded": True,
    }
    if require_resume_state:
        if report_provenance is None:
            raise RuntimeError(f"{label} resume validation lacks report provenance")
        result["resume_state"] = validate_latest_resume_state(
            payload,
            label=label,
            expected_world_size=expected_world_size,
            report_provenance=report_provenance,
            report_history=report_history,
            best_checkpoint_sha256=best_checkpoint_sha256,
        )
    return result


def exact_sha256s(path: Path, artifact_paths: dict[str, Path]) -> dict[str, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != len(CHECKSUM_ARTIFACTS):
        raise RuntimeError("SHA256SUMS must contain exactly four artifact entries")
    parsed: dict[str, str] = {}
    order: list[str] = []
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  ([^/\\]+)", line)
        if match is None or match.group(2) in parsed:
            raise RuntimeError("SHA256SUMS has malformed or duplicate entries")
        digest, name = match.groups()
        parsed[name] = digest
        order.append(name)
    if tuple(order) != CHECKSUM_ARTIFACTS:
        raise RuntimeError(
            f"SHA256SUMS artifact order/content mismatch: {order}"
        )
    actual = {name: sha256(artifact_paths[name]) for name in CHECKSUM_ARTIFACTS}
    require_exact_hashes(parsed, actual, "pilot artifact manifest")
    return parsed


def validate_pilot_artifacts(
    output_dir: Path,
    label: str,
    *,
    expected_world_size: int = 2,
    pinned_model_config: dict[str, object] | None = None,
) -> dict[str, object]:
    if expected_world_size not in {1, 2}:
        raise ValueError("artifact validation world_size must be one or two")
    paths = {
        "report": output_dir / "pair_transformer_report.json",
        "calibrated_checkpoint": output_dir / "pair_transformer_calibrated.pt",
        "best_checkpoint": output_dir / "pair_transformer_best.pt",
        "latest_checkpoint": output_dir / "pair_transformer_latest.pt",
        "hashes": output_dir / "SHA256SUMS.txt",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"{label} succeeded without required artifacts: {missing}")
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    if (
        not isinstance(report, dict)
        or type(report.get("schema_version")) is not int
        or report.get("schema_version") != 1
        or report.get("kind") != "pair_transformer_2xt4_pilot"
        or report.get("safe_for_submission") is not False
    ):
        raise RuntimeError(f"{label} report violates fail-closed contract")
    status = report.get("status")
    if status not in ALLOWED_REPORT_STATUS:
        raise RuntimeError(f"{label} report has unsupported status {status!r}")
    evaluation = report.get("evaluation")
    if not isinstance(evaluation, dict):
        raise RuntimeError(f"{label} report lacks evaluation object")
    gates = evaluation.get("continuation_gates")
    if (
        not isinstance(gates, dict)
        or set(gates) != EXPECTED_GATE_KEYS
        or any(type(value) is not bool for value in gates.values())
    ):
        raise RuntimeError(f"{label} continuation gates are incomplete or non-boolean")
    continuation = evaluation.get("continue_to_1024_source_two_seed_run")
    if type(continuation) is not bool or continuation != all(gates.values()):
        raise RuntimeError(f"{label} continuation flag disagrees with gates")
    expected_status = "continue" if continuation else "stop_or_redesign"
    if status != expected_status:
        raise RuntimeError(f"{label} status disagrees with continuation decision")

    provenance = report.get("provenance")
    resume_contract = (
        provenance.get("resume_contract") if isinstance(provenance, dict) else None
    )
    report_model_config = (
        resume_contract.get("model_config")
        if isinstance(resume_contract, dict)
        else None
    )
    code_provenance = provenance.get("code") if isinstance(provenance, dict) else None
    if (
        not isinstance(provenance, dict)
        or not isinstance(report_model_config, dict)
        or not isinstance(code_provenance, dict)
        or not isinstance(code_provenance.get("model"), str)
    ):
        raise RuntimeError(f"{label} report lacks exact model/resume provenance")
    if pinned_model_config is not None and pinned_model_config != report_model_config:
        raise RuntimeError(f"{label} report model config differs from pinned CLI")
    model_source = Path(code_provenance["model"])
    if (
        not model_source.is_file()
        or code_provenance.get("model_sha256") != sha256(model_source)
        or code_provenance.get("model_sha256")
        != EXPECTED_OVERLAY_SHA256["src/puzzle_assembly/pair_transformer.py"]
    ):
        raise RuntimeError(f"{label} report model source provenance mismatch")
    model_class = pair_transformer_class(model_source)

    calibrated_hash = sha256(paths["calibrated_checkpoint"])
    if (
        report.get("checkpoint_sha256") != calibrated_hash
        or not isinstance(report.get("checkpoint"), str)
        or Path(report["checkpoint"]).resolve()
        != paths["calibrated_checkpoint"].resolve()
    ):
        raise RuntimeError(f"{label} calibrated checkpoint provenance mismatch")
    telemetry_record = report.get("training_telemetry")
    latest_hash = sha256(paths["latest_checkpoint"])
    best_hash = sha256(paths["best_checkpoint"])
    if (
        not isinstance(telemetry_record, dict)
        or telemetry_record.get("latest_checkpoint_sha256") != latest_hash
        or not isinstance(telemetry_record.get("latest_checkpoint"), str)
        or Path(telemetry_record["latest_checkpoint"]).resolve()
        != paths["latest_checkpoint"].resolve()
    ):
        raise RuntimeError(f"{label} latest-checkpoint telemetry mismatch")
    if (
        report.get("best_checkpoint_sha256") != best_hash
        or not isinstance(report.get("best_checkpoint"), str)
        or Path(report["best_checkpoint"]).resolve()
        != paths["best_checkpoint"].resolve()
        or telemetry_record.get("best_checkpoint_sha256") != best_hash
        or not isinstance(telemetry_record.get("best_checkpoint"), str)
        or Path(telemetry_record["best_checkpoint"]).resolve()
        != paths["best_checkpoint"].resolve()
    ):
        raise RuntimeError(f"{label} best-checkpoint provenance mismatch")

    checkpoint_contracts = {
        "calibrated_checkpoint": checkpoint_contract(
            paths["calibrated_checkpoint"],
            f"{label} calibrated_checkpoint",
            model_class=model_class,
            expected_model_config=report_model_config,
        ),
        "best_checkpoint": checkpoint_contract(
            paths["best_checkpoint"],
            f"{label} best_checkpoint",
            model_class=model_class,
            expected_model_config=report_model_config,
        ),
        "latest_checkpoint": checkpoint_contract(
            paths["latest_checkpoint"],
            f"{label} latest_checkpoint",
            model_class=model_class,
            expected_model_config=report_model_config,
            require_resume_state=True,
            expected_world_size=expected_world_size,
            report_provenance=provenance,
            report_history=report.get("training_history"),
            best_checkpoint_sha256=best_hash,
        ),
    }
    checksum_paths = {
        "pair_transformer_report.json": paths["report"],
        "pair_transformer_calibrated.pt": paths["calibrated_checkpoint"],
        "pair_transformer_best.pt": paths["best_checkpoint"],
        "pair_transformer_latest.pt": paths["latest_checkpoint"],
    }
    checksum_manifest = exact_sha256s(paths["hashes"], checksum_paths)
    return {
        "status": status,
        "safe_for_submission": False,
        "continuation_gates": gates,
        "continue_to_1024_source_two_seed_run": continuation,
        "training_telemetry": telemetry_record,
        "checkpoint_contracts": checkpoint_contracts,
        "sha256_manifest": checksum_manifest,
        **{
            name: str(path)
            for name, path in paths.items()
        },
        **{
            f"{name}_sha256": sha256(path)
            for name, path in paths.items()
        },
    }


def execute(wrapper: dict[str, object]) -> None:
    commands_value = wrapper.setdefault("commands", [])
    if not isinstance(commands_value, list):
        raise TypeError("wrapper commands telemetry must be a list")
    commands: list[dict[str, object]] = commands_value
    data_root = find_data_root()
    runtime_root = find_runtime_root()
    base_root, base_archive = find_base_root()
    overlay_root, overlay_archive = find_overlay_root()
    pseudo_gold = find_pseudo_gold()
    code_root = WORKING / "pair_transformer_code"
    copy_code(base_root, overlay_root, code_root)
    overlay_hashes = verify_overlay(code_root)

    model = code_root / "src" / "puzzle_assembly" / "pair_transformer.py"
    trainer = code_root / "scripts" / "train_evaluate_pair_transformer.py"
    tests = code_root / "tests" / "test_pair_transformer.py"
    staged_runner = (
        code_root
        / "runs/assembly_v1/kaggle/pair_transformer_pilot_job"
        / "run_pair_transformer_pilot.py"
    )
    manifest = code_root / "configs" / "denoise_splits_seed20260710.json"
    quarantine = code_root / "configs" / "denoise_validation_quarantine_v1.json"
    denoiser = runtime_root / "selected_tilenaf_synth_50k.pt"
    hbt = runtime_root / "hbt_d320_denoised_rgb_sobel.pt"
    for required in (
        model,
        trainer,
        tests,
        staged_runner,
        manifest,
        quarantine,
        denoiser,
        hbt,
        pseudo_gold,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    pinned_inputs = verify_staged_inputs(
        code_root=code_root,
        base_archive=base_archive,
        overlay_archive=overlay_archive,
        pseudo_gold=pseudo_gold,
        denoiser=denoiser,
        hbt=hbt,
        manifest=manifest,
        quarantine=quarantine,
    )

    wrapper["inputs"] = {
        "exact_mounts": {
            "puzzle": str(PUZZLE_INPUT),
            "runtime": str(RUNTIME_INPUT),
            "solver_rework": str(BASE_INPUT),
            "real_gold_512": str(PSEUDO_INPUT),
            "pair_overlay": str(OVERLAY_INPUT),
        },
        "data_root": str(data_root),
        "runtime_root": str(runtime_root),
        "base_root": str(base_root),
        "base_archive": None if base_archive is None else str(base_archive),
        "base_archive_sha256": None
        if base_archive is None
        else sha256(base_archive),
        "overlay_root": str(overlay_root),
        "overlay_archive": None if overlay_archive is None else str(overlay_archive),
        "overlay_archive_sha256": None
        if overlay_archive is None
        else sha256(overlay_archive),
        "overlay_files_sha256": overlay_hashes,
        "pseudo_gold": str(pseudo_gold),
        "pseudo_gold_sha256": sha256(pseudo_gold),
        "denoiser": str(denoiser),
        "denoiser_sha256": sha256(denoiser),
        "hbt": str(hbt),
        "hbt_sha256": sha256(hbt),
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "quarantine": str(quarantine),
        "quarantine_sha256": sha256(quarantine),
        "runner_sha256": sha256(Path(__file__)),
        "verified_pins": pinned_inputs,
    }

    hardware_started = time.perf_counter()
    wrapper["hardware"] = hardware_probe()
    wrapper["hardware_seconds"] = time.perf_counter() - hardware_started
    print(
        json.dumps({"event": "pair_hardware_passed", **wrapper["hardware"]}),
        flush=True,
    )

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(code_root / "src")
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    environment["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    run_checked(
        [
            sys.executable,
            "-m",
            "py_compile",
            str(model),
            str(trainer),
            str(tests),
            str(staged_runner),
        ],
        cwd=code_root,
        environment=environment,
        label="pycompile",
        telemetry=commands,
    )
    pytest_record = run_checked(
        [sys.executable, "-m", "pytest", "-q", str(tests)],
        cwd=code_root,
        environment=environment,
        label=f"pytest_{EXPECTED_TEST_COUNT}",
        capture=True,
        telemetry=commands,
    )
    match = re.search(r"(\d+) passed", str(pytest_record.get("stdout_tail", "")))
    passed_count = int(match.group(1)) if match else -1
    pytest_record["passed_count"] = passed_count
    if passed_count != EXPECTED_TEST_COUNT:
        raise RuntimeError(
            f"expected {EXPECTED_TEST_COUNT} Pair Transformer tests, saw {passed_count}"
        )

    common = [
        "--data-root", str(data_root),
        "--manifest", str(manifest),
        "--quarantine", str(quarantine),
        "--denoiser", str(denoiser),
        "--hbt-checkpoint", str(hbt),
        "--pseudo-gold", str(pseudo_gold),
        "--amp-init-scale", "1024",
        "--max-amp-skips", "4",
    ]
    smoke_dir = WORKING / "pair_transformer_smoke"
    smoke_command = [
        sys.executable,
        str(trainer),
        "--action", "pilot",
        *common,
        "--smoke",
        "--output-dir", str(smoke_dir),
        "--overwrite",
    ]
    smoke_environment = environment.copy()
    smoke_environment["CUDA_VISIBLE_DEVICES"] = "0"
    run_checked(
        smoke_command,
        cwd=code_root,
        environment=smoke_environment,
        label="single_gpu_smoke_actual_assets",
        telemetry=commands,
    )
    wrapper["smoke"] = validate_pilot_artifacts(
        smoke_dir,
        "single-GPU smoke",
        expected_world_size=1,
        pinned_model_config=SMOKE_MODEL_CONFIG,
    )

    output_dir = WORKING / "pair_transformer_pilot"
    pilot_command = [
        sys.executable,
        "-m", "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        str(trainer),
        "--action", "pilot",
        *common,
        "--train-sources", "512",
        "--epochs", "3",
        "--quick-val-sources", "2",
        "--calibration-sources", "4",
        "--validation-sources", "8",
        "--validation-replicas", "2",
        "--solver-sources", "4",
        "--queries-per-source", "48",
        "--negatives", "31",
        "--groups-per-step", "4",
        "--candidate-top-k", "48",
        "--candidate-reverse-top-k", "8",
        "--iterative-passes", "2",
        "--qap-iterations", "12",
        "--qap-restarts", "1",
        "--output-dir", str(output_dir),
        "--overwrite",
    ]
    run_checked(
        pilot_command,
        cwd=code_root,
        environment=environment,
        label="torchrun_t4x2_default_512x3",
        telemetry=commands,
    )
    wrapper["pilot"] = validate_pilot_artifacts(
        output_dir,
        "T4x2 pilot",
        expected_world_size=2,
        pinned_model_config=PILOT_MODEL_CONFIG,
    )
    wrapper["status"] = "complete"


def main() -> None:
    started = time.perf_counter()
    wrapper: dict[str, object] = {
        "schema_version": 1,
        "kind": "pair_transformer_t4x2_pilot_wrapper",
        "status": "running",
        "safe_for_submission": False,
        "network_access_required": False,
        "commands": [],
    }
    exit_code = 0
    try:
        execute(wrapper)
    except BaseException as error:
        exit_code = 1
        wrapper["status"] = "error"
        wrapper["error_type"] = type(error).__name__
        wrapper["error"] = str(error)
        wrapper["traceback"] = traceback.format_exc()
    finally:
        wrapper["seconds"] = time.perf_counter() - started
        atomic_write_json(WRAPPER_PATH, wrapper)
        wrapper["wrapper_sha256"] = sha256(WRAPPER_PATH)
        print(
            json.dumps({"event": "pair_wrapper_final", **wrapper}, default=str),
            flush=True,
        )
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()

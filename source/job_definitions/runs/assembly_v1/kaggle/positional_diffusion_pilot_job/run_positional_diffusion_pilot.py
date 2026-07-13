#!/usr/bin/env python3
"""Fail-closed offline staging and 2xT4 Positional Diffusion pilot runner."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import stat
import subprocess
import sys
import time
import traceback
from typing import Any
import zipfile


INPUT = Path("/kaggle/input")
DATASET_INPUT = INPUT / "datasets" / "pasha883"
WORKING = Path("/kaggle/working")
PUZZLE_INPUT = DATASET_INPUT / "vsos-ai-initiative-pazzle"
RUNTIME_INPUT = DATASET_INPUT / "vsos-assembly-v1-runtime"
BASE_INPUT = DATASET_INPUT / "vsos-solver-rework-night-code"
OVERLAY_INPUT = DATASET_INPUT / "vsos-positional-diffusion-pilot-code"
WRAPPER_PATH = WORKING / "positional_diffusion_pilot_wrapper.json"

EXPECTED_TEST_COUNT = 23
EXPECTED_BASE_ZIP_SHA256 = "a980c158fb349fbc8619e39eb829acdc675e7332d1ec3995c08f38eb49f45d0c"
EXPECTED_OVERLAY_ZIP_SHA256 = "e4987e110ff518b9d7e7b910158709890c181b6b176834c0ec28991d642ea201"
EXPECTED_RUNTIME_SHA256 = {
    "selected_tilenaf_synth_50k.pt": "77a2d8607c9bc6b80e7aa99ed03329c1fae6bf94ee1d9a654241ab93a05cc734",
    "hbt_d320_denoised_rgb_sobel.pt": "c2589bff58573d592227d522954a7ed6c22f6fec7ecbbea2bf0c755d7a720787",
}
EXPECTED_CONFIG_SHA256 = {
    "configs/denoise_splits_seed20260710.json": "de4fd2e596efa0d157d2d4480eed5fb84812d358138a1db53c1706bfb580e345",
    "configs/denoise_validation_quarantine_v1.json": "38dfd12f60579d77999c0cdb4a648fb4ff0343a8fb5e4c421f0b29f8b7bd6215",
}
EXPECTED_OVERLAY_SHA256 = {
    "src/puzzle_assembly/positional_diffusion.py": "25a4ace2f3aaa8e1371ca54a7e65efaddd8db9aafd37a78deb299290a914fae3",
    "scripts/train_evaluate_positional_diffusion.py": "d52ca2665740fff02cbc415c86871fa698f923c1207ca49ab0e9929a874d315d",
    "tests/test_positional_diffusion.py": "4e11218c782a125d55c137b0d7d6d64c51bac1e6e6830fee0fea99ff0d6c5648",
}
EXPECTED_PROVENANCE_CODE_SHA256 = {
    "scripts/train_evaluate_positional_diffusion.py": "d52ca2665740fff02cbc415c86871fa698f923c1207ca49ab0e9929a874d315d",
    "src/puzzle_assembly/compatibility.py": "aff2149b161c4fded4e5d91fbea49a8a62967886148d3ad374467331e0416a9f",
    "src/puzzle_assembly/components.py": "53fcc7c4fd23956db884ee45060e47f8e94a931c16e497e426d67549621bd367",
    "src/puzzle_assembly/geometry.py": "1e16bec6fb98a33060558d5d28062334d9114b12424733ef103a40393ef1ba86",
    "src/puzzle_assembly/learned.py": "9e3dba673aa85eaab5698dbeb63b3d94f88e3ea92b5e5979bde4b0273642697b",
    "src/puzzle_assembly/metrics.py": "84857ef92c382cc0964c21bfec67c13308014a1674aebf8686b17514784dae69",
    "src/puzzle_assembly/panels.py": "783356628517e3a23b8703672bca604c3d879c875f5b5f35f87182425500280f",
    "src/puzzle_assembly/positional_diffusion.py": "25a4ace2f3aaa8e1371ca54a7e65efaddd8db9aafd37a78deb299290a914fae3",
    "src/puzzle_assembly/protocol.py": "b711ad6d28a2fe60329e3e8236e58adbfbceea8ca4c8bf85e9a057e7619e24f4",
    "src/puzzle_assembly/qap.py": "b8a5e1da67387fd04effd979270ca16925aceab23d37083eda108c6e3e349c32",
    "src/puzzle_assembly/solvers.py": "23f9e32200748349d0da8558b7b44053a758e1c1eb306d8f31ce59feae03fe8e",
    "src/puzzle_denoise_v2/degradation.py": "7e314081c143a1c7846a9777eaea8716092a85595f856769efd3704a2c583a75",
    "src/puzzle_denoise_v2/inference.py": "20767cc26270cfde7472cf33a0247b1ea6d96e5b5c8ff5d705b785ae710dd6da",
    "src/puzzle_denoise_v2/losses.py": "56776289cd51e49a28ce54bc4762d144d87c7efbf6d4ca56668fc3b019dbbf34",
    "src/puzzle_denoise_v2/metrics.py": "e8275fb096276a63b7114be1a74b24009dc2143dddf299ce5eaceac401a27d36",
    "src/puzzle_denoise_v2/model.py": "37db32fb83ece0f122757bdbec19ffc6a17c5e5e00ef92a26328247d95c55d11",
    "src/puzzle_denoise_v2/tiles.py": "21270e283e50ea0b155ef194de889222fb0c4f6954437eb1526342c006eefaa7",
    "src/puzzle_denoise_v2/training.py": "6719ee6a62434cd8a00fafb92b28f6a10941cdbf5c83573fc6556b33e5eba56e",
}
EXPECTED_EXECUTED_CODE_SHA256 = {
    **EXPECTED_PROVENANCE_CODE_SHA256,
    "src/puzzle_assembly/__init__.py": "09e051b7555471aafca03cd666d789f033aca47f1c82f6e2af9c0cce50afe9d5",
    "src/puzzle_denoise_v2/__init__.py": "30849e0f937ba4a50e85ce2eee0d2b930db06fbcc0b7dff84547e121ef2f30b7",
}
EXPECTED_MODEL_CONFIG = {
    "model_dim": 384,
    "cnn_channels": 64,
    "layers": 8,
    "heads": 12,
    "feedforward_dim": 1536,
    "dropout": 0.05,
    "diffusion_steps": 300,
    "beta_start": 0.0001,
    "beta_end": 0.02,
    "tile_encode_chunk": 192,
    "activation_checkpointing": True,
}
EXPECTED_MODEL_PARAMETERS = 16_030_530
EXPECTED_MEMORY_ESTIMATE = {
    "attention_logits": 63_700_992,
    "token_activations": 42_467_328,
    "relative_graph": 1_327_104,
    "checkpointed_encoder_chunk": 39_321_600,
    "estimated_peak_activations": 359_153_664,
}
EXPECTED_TRAIN_NAMES_SHA256 = "111363a005aff01f88de0bea497db8bceeeb0afb2833f3c2b3fcea698a164f49"
EXPECTED_TRAIN_DATA_SHA256 = "addfc68707f51665ba15d0b7f47135c66a385540e5e550c27b8f8b30e264a4d4"
EXPECTED_DEV_NAMES = [
    "img_001485.png",
    "img_005748.png",
    "img_003783.png",
    "img_001693.png",
    "img_006659.png",
    "img_004510.png",
    "img_005403.png",
    "img_005200.png",
]
EXPECTED_DEV_NAMES_SHA256 = "bc2f89e49371486ffece5d8ca9881f7de15b22948bab2e0e0749dbfdbffc3581"
EXPECTED_DEV_DATA_SHA256 = "0caaa9133fb181f311a1482872b20a1e442630bb72998bf155beb30b12dbe1fd"
EXPECTED_CORRUPTION_COUNTS = {
    0: (209, 175),
    1: (193, 191),
    2: (193, 191),
    3: (185, 199),
}
EXPECTED_DETERMINISM = {
    "deterministic_algorithms": True,
    "deterministic_warn_only": True,
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
    "cublas_workspace_config": ":4096:8",
}
EXPECTED_PANEL_GATE_KEYS = {
    "adjacency_gain_vs_per_source_best_baseline",
    "ssim_gain_vs_per_source_best_baseline",
    "joint_positive_source_fraction",
    "bootstrap_lower_adjacency_positive",
    "bootstrap_lower_ssim_positive",
}
EXPECTED_MACRO_GATE_KEYS = {"adjacency_gain", "ssim_gain"}
EXPECTED_BOOTSTRAP_RESAMPLES = 2000
EXPECTED_BOOTSTRAP_CONFIDENCE = 0.95
EXPECTED_BOOTSTRAP_UNIT = "whole source after averaging corruption replicas"
STATUS_FOR_GATE = {
    True: "bounded positive signal only; still not submission-ready",
    False: "no trustworthy cross-corruption signal; stop/pivot before larger training",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def safe_extract(archive: Path, destination: Path) -> Path:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    with zipfile.ZipFile(archive) as handle:
        for info in handle.infolist():
            parts = Path(info.filename).parts
            mode = (info.external_attr >> 16) & 0o170000
            if (
                info.filename.startswith("/")
                or ".." in parts
                or mode == stat.S_IFLNK
            ):
                raise RuntimeError(f"unsafe archive member in {archive}: {info.filename}")
        handle.extractall(destination)
    return destination


def require_exact_mounts() -> dict[str, Path]:
    expected = {
        PUZZLE_INPUT.name: PUZZLE_INPUT,
        RUNTIME_INPUT.name: RUNTIME_INPUT,
        BASE_INPUT.name: BASE_INPUT,
        OVERLAY_INPUT.name: OVERLAY_INPUT,
    }
    actual = {path.name for path in DATASET_INPUT.iterdir() if path.is_dir()}
    if actual != set(expected):
        raise RuntimeError(
            f"expected exactly four Kaggle dataset mounts {sorted(expected)}, found {sorted(actual)}"
        )
    for label, path in expected.items():
        if not path.is_dir():
            raise FileNotFoundError(f"missing exact Kaggle mount {label}: {path}")
    return expected


def one(paths: list[Path], label: str) -> Path:
    values = sorted({path.resolve() for path in paths})
    if len(values) != 1:
        raise RuntimeError(f"expected exactly one {label}, found {values}")
    return values[0]


def find_data_root() -> Path:
    return one(
        [
            path.parent.parent
            for path in PUZZLE_INPUT.glob("**/train/inputs")
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
            for path in RUNTIME_INPUT.glob("**/selected_tilenaf_synth_50k.pt")
            if (path.parent / "hbt_d320_denoised_rgb_sobel.pt").is_file()
        ],
        "runtime asset root",
    )


def find_base_root() -> tuple[Path, Path | None]:
    archives = sorted(BASE_INPUT.glob("**/solver_rework_code.zip"))
    archive = one(archives, "solver-rework base archive") if archives else None
    if archive is not None and sha256(archive) != EXPECTED_BASE_ZIP_SHA256:
        raise RuntimeError("solver-rework base ZIP hash mismatch")
    direct = sorted(
        {
            path.parent.parent.parent
            for path in BASE_INPUT.glob("**/src/puzzle_assembly/qap.py")
            if (path.parent.parent.parent / "configs/denoise_splits_seed20260710.json").is_file()
        }
    )
    if len(direct) == 1:
        return direct[0], archive
    if direct:
        raise RuntimeError(f"ambiguous direct solver-rework roots: {direct}")
    if archive is None:
        raise RuntimeError("solver-rework base is neither direct nor archived")
    extracted = safe_extract(archive, WORKING / "positional_diffusion_base")
    if not (extracted / "src/puzzle_assembly/qap.py").is_file():
        raise RuntimeError("solver-rework archive lacks qap.py")
    return extracted, archive


def find_overlay_root() -> tuple[Path, Path | None]:
    archives = sorted(OVERLAY_INPUT.glob("**/positional_diffusion_code.zip"))
    archive = one(archives, "Positional Diffusion overlay archive") if archives else None
    if archive is not None and sha256(archive) != EXPECTED_OVERLAY_ZIP_SHA256:
        raise RuntimeError("Positional Diffusion overlay ZIP hash mismatch")
    direct = sorted(
        {
            path.parent.parent
            for path in OVERLAY_INPUT.glob("**/scripts/train_evaluate_positional_diffusion.py")
            if (path.parent.parent / "src/puzzle_assembly/positional_diffusion.py").is_file()
            and (path.parent.parent / "tests/test_positional_diffusion.py").is_file()
        }
    )
    if len(direct) == 1:
        return direct[0], archive
    if direct:
        raise RuntimeError(f"ambiguous direct Positional Diffusion overlays: {direct}")
    if archive is None:
        raise RuntimeError("Positional Diffusion overlay is neither direct nor archived")
    extracted = safe_extract(archive, WORKING / "positional_diffusion_overlay")
    if not (extracted / "scripts/train_evaluate_positional_diffusion.py").is_file():
        raise RuntimeError("Positional Diffusion archive lacks trainer")
    return extracted, archive


def verify_hashes(root: Path, expected: Mapping[str, str], label: str) -> dict[str, str]:
    actual: dict[str, str] = {}
    for relative, wanted in expected.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        actual[relative] = sha256(path)
        if actual[relative] != wanted:
            raise RuntimeError(
                f"{label} hash mismatch for {relative}: expected {wanted}, found {actual[relative]}"
            )
    return actual


def copy_and_verify_code(base: Path, overlay: Path, destination: Path) -> dict[str, str]:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(base / "src", destination / "src")
    shutil.copytree(base / "configs", destination / "configs")
    (destination / "scripts").mkdir()
    (destination / "tests").mkdir()
    for relative in EXPECTED_OVERLAY_SHA256:
        source = overlay / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    verify_hashes(destination, EXPECTED_OVERLAY_SHA256, "overlay")
    verify_hashes(destination, EXPECTED_CONFIG_SHA256, "config")
    return verify_hashes(destination, EXPECTED_EXECUTED_CODE_SHA256, "executed code")


def dataset_slice_sha256(data_root: Path, names: list[str]) -> str:
    digest = hashlib.sha256()
    for name in names:
        path = data_root / "train" / "targets" / name
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def verify_dataset_slices(
    data_root: Path,
    *,
    code_root: Path,
    manifest: Path,
    quarantine: Path,
) -> dict[str, Any]:
    if str(code_root / "src") not in sys.path:
        sys.path.insert(0, str(code_root / "src"))
    from puzzle_assembly.protocol import source_names_for_split

    train = source_names_for_split(
        "edge_train", manifest_path=manifest, quarantine_path=quarantine
    )[:384]
    development = source_names_for_split(
        "assembly_incremental_gate",
        manifest_path=manifest,
        quarantine_path=quarantine,
    )[:8]
    train_names_hash = hashlib.sha256("\n".join(train).encode("utf-8")).hexdigest()
    dev_names_hash = hashlib.sha256("\n".join(development).encode("utf-8")).hexdigest()
    train_data_hash = dataset_slice_sha256(data_root, train)
    dev_data_hash = dataset_slice_sha256(data_root, development)
    if (
        len(train) != 384
        or train_names_hash != EXPECTED_TRAIN_NAMES_SHA256
        or train_data_hash != EXPECTED_TRAIN_DATA_SHA256
        or development != EXPECTED_DEV_NAMES
        or dev_names_hash != EXPECTED_DEV_NAMES_SHA256
        or dev_data_hash != EXPECTED_DEV_DATA_SHA256
    ):
        raise RuntimeError("pinned train/development dataset slice provenance drifted")
    return {
        "train_source_count": 384,
        "train_names_sha256": train_names_hash,
        "train_data_sha256": train_data_hash,
        "development_names": development,
        "development_names_sha256": dev_names_hash,
        "development_data_sha256": dev_data_hash,
    }


def run_checked(
    command: list[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    label: str,
    telemetry: list[dict[str, Any]],
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=dict(environment),
            check=False,
            capture_output=capture,
            text=True,
        )
    except BaseException as error:
        telemetry.append(
            {
                "label": label,
                "command": command,
                "returncode": None,
                "seconds": time.perf_counter() - started,
                "launch_error": f"{type(error).__name__}: {error}",
            }
        )
        raise
    record: dict[str, Any] = {
        "label": label,
        "command": command,
        "returncode": completed.returncode,
        "seconds": time.perf_counter() - started,
    }
    if capture:
        record["stdout_tail"] = (completed.stdout or "")[-4000:]
        record["stderr_tail"] = (completed.stderr or "")[-4000:]
        if completed.stdout:
            print(completed.stdout, end="", flush=True)
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr, flush=True)
    telemetry.append(record)
    if completed.returncode != 0:
        raise RuntimeError(f"{label} failed with return code {completed.returncode}")
    return completed


def hardware_probe(
    telemetry: list[dict[str, Any]],
    *,
    cwd: Path,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    run_checked(
        ["nvidia-smi"],
        cwd=cwd,
        environment=environment,
        label="nvidia_smi",
        telemetry=telemetry,
    )
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("Positional Diffusion pilot requires exactly two CUDA GPUs")
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_states = torch.cuda.get_rng_state_all()
    if len(cuda_rng_states) != 2:
        raise RuntimeError("Positional Diffusion pilot requires two CUDA RNG generators")
    devices: list[dict[str, Any]] = []
    for index in range(2):
        device = torch.device(f"cuda:{index}")
        name = torch.cuda.get_device_name(index)
        capability = tuple(torch.cuda.get_device_capability(index))
        if "TESLA T4" not in name.upper() or capability != (7, 5):
            raise RuntimeError(f"GPU {index} violates Tesla T4 sm_75 pin: {name}, {capability}")
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
            or not torch.isfinite(loss)
            or left.grad is None
            or right.grad is None
            or not torch.isfinite(left.grad).all()
            or not torch.isfinite(right.grad).all()
        ):
            raise RuntimeError(f"GPU {index} failed deterministic fp16 forward/backward")
        torch.cuda.synchronize(device)
        devices.append(
            {
                "index": index,
                "name": name,
                "capability": list(capability),
                "total_memory_bytes": int(torch.cuda.get_device_properties(index).total_memory),
                "fp16_matmul_shape": [1024, 1024, 1024],
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
        "device_count": 2,
        "devices": devices,
        "rng_state_schema": {
            "torch_cpu": {
                "device": "cpu",
                "dtype": "torch.uint8",
                "ndim": 1,
                "numel": int(cpu_rng_state.numel()),
            },
            "torch_cuda": [
                {
                    "device_index": index,
                    "device": "cpu",
                    "dtype": "torch.uint8",
                    "ndim": 1,
                    "numel": int(state.numel()),
                }
                for index, state in enumerate(cuda_rng_states)
            ],
        },
    }


def expected_arguments(
    *,
    data_root: Path,
    manifest: Path,
    quarantine: Path,
    denoiser: Path,
    hbt: Path,
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "mode": "pilot",
        "output_dir": str(output_dir),
        "checkpoint": "",
        "resume_checkpoint": "",
        "overwrite": True,
        "data_root": str(data_root),
        "manifest": str(manifest),
        "quarantine": str(quarantine),
        "denoiser": str(denoiser),
        "hbt_checkpoint": str(hbt),
        "device": "auto",
        "amp": "fp16",
        "amp_init_scale": 4096.0,
        "amp_max_consecutive_skips": 3,
        "amp_max_total_skips": 8,
        "seed": 20260711,
        "train_offset": 0,
        "train_sources": 384,
        "epochs": 4,
        "gradient_accumulation": 4,
        "max_optimizer_steps": 192,
        "learning_rate": 0.0002,
        "weight_decay": 0.0001,
        "grad_clip": 1.0,
        "structure_weight": 0.20,
        "baseline_condition_dropout": 0.25,
        "warm_start_layout": "softcycle",
        "model_dim": 384,
        "cnn_channels": 64,
        "layers": 8,
        "heads": 12,
        "feedforward_dim": 1536,
        "dropout": 0.05,
        "diffusion_steps": 300,
        "sampling_steps": 30,
        "tile_encode_chunk": 192,
        "activation_checkpointing": True,
        "graph_top_k": 16,
        "graph_temperature": 0.35,
        "denoise_batch_size": 576,
        "dev_offset": 0,
        "dev_split": "assembly_incremental_gate",
        "dev_sources": 8,
        "dev_replicas": 2,
        "gate_bootstrap_resamples": 2000,
        "gate_bootstrap_confidence": 0.95,
        "qap_iterations": 25,
        "qap_restarts": 2,
        "qap_boundary_weight": 0.05,
        "qap_refine_swaps": 8,
        "gate_min_adjacency_gain": 0.002,
        "gate_min_ssim_gain": 0.001,
        "gate_min_positive_source_fraction": 0.50,
        "log_every": 8,
    }


def pilot_command(script: Path, arguments: Mapping[str, Any]) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        str(script),
        "--mode",
        "pilot",
        "--output-dir",
        str(arguments["output_dir"]),
        "--overwrite",
        "--data-root",
        str(arguments["data_root"]),
        "--manifest",
        str(arguments["manifest"]),
        "--quarantine",
        str(arguments["quarantine"]),
        "--denoiser",
        str(arguments["denoiser"]),
        "--hbt-checkpoint",
        str(arguments["hbt_checkpoint"]),
        "--device",
        "auto",
        "--amp",
        "fp16",
        "--amp-init-scale",
        "4096",
        "--amp-max-consecutive-skips",
        "3",
        "--amp-max-total-skips",
        "8",
        "--seed",
        "20260711",
        "--train-offset",
        "0",
        "--train-sources",
        "384",
        "--epochs",
        "4",
        "--gradient-accumulation",
        "4",
        "--max-optimizer-steps",
        "192",
        "--learning-rate",
        "0.0002",
        "--weight-decay",
        "0.0001",
        "--grad-clip",
        "1.0",
        "--structure-weight",
        "0.20",
        "--baseline-condition-dropout",
        "0.25",
        "--warm-start-layout",
        "softcycle",
        "--model-dim",
        "384",
        "--cnn-channels",
        "64",
        "--layers",
        "8",
        "--heads",
        "12",
        "--feedforward-dim",
        "1536",
        "--dropout",
        "0.05",
        "--diffusion-steps",
        "300",
        "--sampling-steps",
        "30",
        "--tile-encode-chunk",
        "192",
        "--activation-checkpointing",
        "--graph-top-k",
        "16",
        "--graph-temperature",
        "0.35",
        "--denoise-batch-size",
        "576",
        "--dev-offset",
        "0",
        "--dev-split",
        "assembly_incremental_gate",
        "--dev-sources",
        "8",
        "--dev-replicas",
        "2",
        "--gate-bootstrap-resamples",
        "2000",
        "--gate-bootstrap-confidence",
        "0.95",
        "--qap-iterations",
        "25",
        "--qap-restarts",
        "2",
        "--qap-boundary-weight",
        "0.05",
        "--qap-refine-swaps",
        "8",
        "--gate-min-adjacency-gain",
        "0.002",
        "--gate-min-ssim-gain",
        "0.001",
        "--gate-min-positive-source-fraction",
        "0.50",
        "--log-every",
        "8",
    ]


def assert_recursive_safety(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        if "safe_for_submission" in value and value["safe_for_submission"] is not False:
            raise RuntimeError(f"{path}.safe_for_submission must be false")
        if "submission_ready" in value and value["submission_ready"] is not False:
            raise RuntimeError(f"{path}.submission_ready must be false")
        for key, item in value.items():
            assert_recursive_safety(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_recursive_safety(item, f"{path}[{index}]")


def assert_recursive_finite(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            assert_recursive_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_recursive_finite(item, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"{path} must be finite")


def finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{label} must be finite")
    return result


def validate_gate(record: Any, label: str, *, strict: bool) -> bool:
    if not isinstance(record, Mapping) or not isinstance(record.get("passed"), bool):
        raise RuntimeError(f"{label} must contain boolean passed")
    value = finite_number(record.get("value"), f"{label}.value")
    minimum = finite_number(record.get("minimum"), f"{label}.minimum")
    expected = value > minimum if strict else value >= minimum
    if record["passed"] is not expected:
        raise RuntimeError(f"{label}.passed is inconsistent with value/minimum")
    return expected


def close(left: float, right: float, *, tolerance: float = 1e-10) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def bootstrap_seed(master: int, stage: str, source: str, replica: int = 0) -> int:
    digest = hashlib.sha256(
        f"{master}:{stage}:{source}:{replica}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def recompute_bootstrap_ci(
    values: Any,
    *,
    seed: int,
) -> dict[str, Any]:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) != 8 or not np.isfinite(array).all():
        raise RuntimeError("bootstrap requires exactly eight finite source-level values")
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0,
        len(array),
        size=(EXPECTED_BOOTSTRAP_RESAMPLES, len(array)),
    )
    means = array[indices].mean(axis=1)
    alpha = (1.0 - EXPECTED_BOOTSTRAP_CONFIDENCE) / 2.0
    return {
        "mean": float(array.mean()),
        "lower": float(np.quantile(means, alpha)),
        "upper": float(np.quantile(means, 1.0 - alpha)),
        "confidence": EXPECTED_BOOTSTRAP_CONFIDENCE,
        "resamples": EXPECTED_BOOTSTRAP_RESAMPLES,
        "unit": EXPECTED_BOOTSTRAP_UNIT,
    }


def validate_bootstrap_ci(
    recorded: Any,
    expected: Mapping[str, Any],
    label: str,
) -> None:
    expected_fields = {"mean", "lower", "upper", "confidence", "resamples", "unit"}
    if not isinstance(recorded, Mapping) or set(recorded) != expected_fields:
        raise RuntimeError(f"{label} bootstrap CI schema is invalid")
    if (
        not isinstance(recorded.get("resamples"), int)
        or isinstance(recorded.get("resamples"), bool)
        or recorded.get("resamples") != EXPECTED_BOOTSTRAP_RESAMPLES
        or recorded.get("unit") != EXPECTED_BOOTSTRAP_UNIT
    ):
        raise RuntimeError(f"{label} bootstrap CI contract drifted")
    for name in ("mean", "lower", "upper", "confidence"):
        if not close(
            finite_number(recorded.get(name), f"{label}.{name}"),
            finite_number(expected.get(name), f"{label}.expected.{name}"),
        ):
            raise RuntimeError(f"{label}.{name} is not the deterministic recomputation")


def validate_development(development: Any) -> tuple[bool, str]:
    if not isinstance(development, Mapping):
        raise RuntimeError("development report must be a mapping")
    assert_recursive_safety(development, "development")
    if (
        development.get("safe_for_submission") is not False
        or development.get("submission_ready") is not False
        or not isinstance(development.get("development_gate_passed"), bool)
    ):
        raise RuntimeError("development safety/overall gate schema is invalid")
    if development.get("source_names") != EXPECTED_DEV_NAMES:
        raise RuntimeError("development source names drifted")
    if development.get("source_names_sha256") != EXPECTED_DEV_NAMES_SHA256:
        raise RuntimeError("development source-name hash drifted")
    panels = development.get("panels")
    if not isinstance(panels, Mapping) or set(panels) != {
        "primary_kornia",
        "independent_libjpeg",
    }:
        raise RuntimeError("development must contain exactly two corruption panels")

    all_records: list[Mapping[str, Any]] = []
    panel_passes: list[bool] = []
    for panel_name in ("primary_kornia", "independent_libjpeg"):
        panel = panels[panel_name]
        if not isinstance(panel, Mapping) or not isinstance(panel.get("gate_passed"), bool):
            raise RuntimeError(f"{panel_name} panel gate schema is invalid")
        if (
            panel.get("source_count") != 8
            or panel.get("replicas_per_source") != 2
            or panel.get("cell_count") != 16
        ):
            raise RuntimeError(f"{panel_name} panel cardinality drifted")
        records = panel.get("per_source")
        if not isinstance(records, list) or len(records) != 16:
            raise RuntimeError(f"{panel_name} must contain exactly 16 cells")
        cells: set[tuple[str, int]] = set()
        source_deltas: dict[str, list[tuple[float, float]]] = {}
        for record in records:
            if not isinstance(record, Mapping):
                raise RuntimeError(f"{panel_name} cell is not a mapping")
            source = str(record.get("source"))
            replica = record.get("replica")
            if (
                source not in EXPECTED_DEV_NAMES
                or not isinstance(replica, int)
                or isinstance(replica, bool)
                or replica not in {0, 1}
                or record.get("panel") != panel_name
            ):
                raise RuntimeError(f"{panel_name} cell identity is invalid")
            cells.add((source, replica))
            if (
                record.get("truth_derived_confidence_used") is not False
                or record.get("target_selected_candidate_used") is not False
            ):
                raise RuntimeError(f"{panel_name} cell violates anti-shortcut contract")
            candidate = record.get("candidate")
            w1 = record.get("qap_w1_baseline")
            w4 = record.get("w4_qap_baseline")
            hbt = record.get("pure_hbt_qap_baseline")
            envelope = record.get("baseline_envelope")
            delta = record.get("paired_delta")
            if not all(isinstance(item, Mapping) for item in (candidate, w1, w4, hbt, envelope, delta)):
                raise RuntimeError(f"{panel_name} cell metric schema is invalid")
            computed: list[float] = []
            for metric in ("combined_adjacency", "predicted_layout_ssim"):
                envelope_value = max(
                    finite_number(w1.get(metric), f"{panel_name}.{metric}.w1"),
                    finite_number(w4.get(metric), f"{panel_name}.{metric}.w4"),
                    finite_number(hbt.get(metric), f"{panel_name}.{metric}.hbt"),
                )
                recorded_envelope = finite_number(
                    envelope.get(metric), f"{panel_name}.{metric}.envelope"
                )
                recorded_delta = finite_number(delta.get(metric), f"{panel_name}.{metric}.delta")
                candidate_value = finite_number(
                    candidate.get(metric), f"{panel_name}.{metric}.candidate"
                )
                if not close(recorded_envelope, envelope_value) or not close(
                    recorded_delta, candidate_value - envelope_value
                ):
                    raise RuntimeError(f"{panel_name} envelope/delta is inconsistent")
                computed.append(recorded_delta)
            source_deltas.setdefault(source, []).append((computed[0], computed[1]))
        if len(cells) != 16 or set(source_deltas) != set(EXPECTED_DEV_NAMES):
            raise RuntimeError(f"{panel_name} cells are duplicated or incomplete")
        import numpy as np

        ordered_sources = sorted(source_deltas)
        adjacency_values = np.asarray(
            [
                np.mean([item[0] for item in source_deltas[source]])
                for source in ordered_sources
            ],
            dtype=np.float64,
        )
        ssim_values = np.asarray(
            [
                np.mean([item[1] for item in source_deltas[source]])
                for source in ordered_sources
            ],
            dtype=np.float64,
        )
        mean_adjacency = float(adjacency_values.mean())
        mean_ssim = float(ssim_values.mean())
        positive_fraction = float(
            np.mean((adjacency_values > 0.0) & (ssim_values > 0.0))
        )
        mean_record = panel.get("mean_paired_delta_vs_envelope")
        if not isinstance(mean_record, Mapping) or not close(
            finite_number(mean_record.get("combined_adjacency"), f"{panel_name}.mean_adj"),
            mean_adjacency,
        ) or not close(
            finite_number(mean_record.get("predicted_layout_ssim"), f"{panel_name}.mean_ssim"),
            mean_ssim,
        ):
            raise RuntimeError(f"{panel_name} aggregate deltas are inconsistent")
        expected_ci = {
            "combined_adjacency": recompute_bootstrap_ci(
                adjacency_values,
                seed=bootstrap_seed(
                    20260711,
                    "posdiff:bootstrap-adjacency",
                    panel_name,
                ),
            ),
            "predicted_layout_ssim": recompute_bootstrap_ci(
                ssim_values,
                seed=bootstrap_seed(
                    20260711,
                    "posdiff:bootstrap-ssim",
                    panel_name,
                ),
            ),
        }
        recorded_ci = panel.get("source_bootstrap_ci")
        if not isinstance(recorded_ci, Mapping) or set(recorded_ci) != set(expected_ci):
            raise RuntimeError(f"{panel_name} source_bootstrap_ci schema is invalid")
        for metric, expected in expected_ci.items():
            validate_bootstrap_ci(
                recorded_ci[metric],
                expected,
                f"{panel_name}.source_bootstrap_ci.{metric}",
            )
        gates = panel.get("gates")
        if not isinstance(gates, Mapping) or set(gates) != EXPECTED_PANEL_GATE_KEYS:
            raise RuntimeError(f"{panel_name} gate set drifted")
        expected_gate_values = {
            "adjacency_gain_vs_per_source_best_baseline": mean_adjacency,
            "ssim_gain_vs_per_source_best_baseline": mean_ssim,
            "joint_positive_source_fraction": positive_fraction,
            "bootstrap_lower_adjacency_positive": expected_ci["combined_adjacency"]["lower"],
            "bootstrap_lower_ssim_positive": expected_ci["predicted_layout_ssim"]["lower"],
        }
        expected_gate_minimums = {
            "adjacency_gain_vs_per_source_best_baseline": 0.002,
            "ssim_gain_vs_per_source_best_baseline": 0.001,
            "joint_positive_source_fraction": 0.50,
            "bootstrap_lower_adjacency_positive": 0.0,
            "bootstrap_lower_ssim_positive": 0.0,
        }
        passed: list[bool] = []
        for name, record in gates.items():
            strict = name.startswith("bootstrap_lower_")
            passed.append(validate_gate(record, f"{panel_name}.{name}", strict=strict))
            if not close(
                finite_number(record["value"], f"{panel_name}.{name}.value"),
                expected_gate_values[name],
            ) or not close(
                finite_number(record["minimum"], f"{panel_name}.{name}.minimum"),
                expected_gate_minimums[name],
            ):
                raise RuntimeError(f"{panel_name}.{name} value is not recomputed aggregate")
        panel_expected = all(passed)
        if panel["gate_passed"] is not panel_expected:
            raise RuntimeError(f"{panel_name} overall gate is incoherent")
        panel_passes.append(panel_expected)
        all_records.extend(records)

    macro_gates = development.get("macro_gates")
    if not isinstance(macro_gates, Mapping) or set(macro_gates) != EXPECTED_MACRO_GATE_KEYS:
        raise RuntimeError("macro gate set drifted")
    by_source: dict[str, list[tuple[float, float]]] = {}
    for record in all_records:
        delta = record["paired_delta"]
        by_source.setdefault(str(record["source"]), []).append(
            (float(delta["combined_adjacency"]), float(delta["predicted_layout_ssim"]))
        )
    macro_adjacency = sum(
        sum(item[0] for item in values) / len(values) for values in by_source.values()
    ) / 8.0
    macro_ssim = sum(
        sum(item[1] for item in values) / len(values) for values in by_source.values()
    ) / 8.0
    macro_values = {"adjacency_gain": macro_adjacency, "ssim_gain": macro_ssim}
    macro_passes: list[bool] = []
    for name, record in macro_gates.items():
        macro_passes.append(validate_gate(record, f"macro.{name}", strict=False))
        if not close(finite_number(record["value"], f"macro.{name}.value"), macro_values[name]):
            raise RuntimeError(f"macro.{name} value is incoherent")
    overall = all(panel_passes) and all(macro_passes)
    if development["development_gate_passed"] is not overall:
        raise RuntimeError("development overall gate is incoherent")
    assessment = STATUS_FOR_GATE[overall]
    if development.get("assessment") != assessment:
        raise RuntimeError("development assessment is incoherent")
    return overall, assessment


def validate_report_payload(
    report: Any,
    *,
    expected_args: Mapping[str, Any],
    hardware: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(report, Mapping):
        raise RuntimeError("pilot report must be a mapping")
    assert_recursive_safety(report, "report")
    assert_recursive_finite(report, "report")
    if (
        report.get("schema_version") != 1
        or report.get("kind") != "positional_diffusion_bounded_signal_report"
        or report.get("mode") != "pilot"
        or report.get("safe_for_submission") is not False
        or report.get("submission_ready") is not False
    ):
        raise RuntimeError("pilot report top-level schema/safety contract failed")
    if report.get("arguments") != dict(expected_args):
        raise RuntimeError("pilot report arguments differ from the fully explicit command")
    if report.get("model_config") != EXPECTED_MODEL_CONFIG:
        raise RuntimeError("pilot report model config drifted")
    if report.get("model_parameters") != EXPECTED_MODEL_PARAMETERS:
        raise RuntimeError("pilot report parameter count drifted")
    memory = report.get("memory_estimate")
    if not isinstance(memory, Mapping) or any(
        memory.get(name) != value for name, value in EXPECTED_MEMORY_ESTIMATE.items()
    ):
        raise RuntimeError("pilot report memory estimate drifted")
    split = report.get("split_provenance")
    if not isinstance(split, Mapping):
        raise RuntimeError("pilot report split provenance is missing")
    if (
        split.get("manifest_sha256") != EXPECTED_CONFIG_SHA256[
            "configs/denoise_splits_seed20260710.json"
        ]
        or split.get("quarantine_sha256") != EXPECTED_CONFIG_SHA256[
            "configs/denoise_validation_quarantine_v1.json"
        ]
        or split.get("train_source_names_sha256") != EXPECTED_TRAIN_NAMES_SHA256
        or split.get("train_data_sha256") != EXPECTED_TRAIN_DATA_SHA256
        or split.get("development_source_names") != EXPECTED_DEV_NAMES
        or split.get("development_source_names_sha256") != EXPECTED_DEV_NAMES_SHA256
        or split.get("whole_source_disjoint") is not True
        or split.get("development_split") != "assembly_incremental_gate"
    ):
        raise RuntimeError("pilot report split provenance drifted")
    exposure = split.get("upstream_exposure_audit")
    if not isinstance(exposure, Mapping) or (
        exposure.get("development_source_count") != 8
        or exposure.get("denoiser_exposure_source_count") != 4993
        or exposure.get("hbt_exposure_source_count") != 2080
        or exposure.get("denoiser_overlap_count") != 0
        or exposure.get("hbt_overlap_count") != 0
        or exposure.get("zero_upstream_exposure_asserted") is not True
    ):
        raise RuntimeError("pilot report upstream exposure audit drifted")
    overall, status = validate_development(report.get("development"))
    if report.get("method_status") not in set(STATUS_FOR_GATE.values()):
        raise RuntimeError("pilot report method_status is not allowed")
    if report.get("method_status") != status:
        raise RuntimeError("pilot report method_status is incoherent with gates")
    runtime = report.get("runtime")
    if not isinstance(runtime, Mapping) or (
        runtime.get("world_size") != 2
        or runtime.get("amp_enabled") is not True
        or runtime.get("amp_dtype") != "torch.float16"
        or runtime.get("determinism") != EXPECTED_DETERMINISM
    ):
        raise RuntimeError("pilot report runtime contract drifted")
    if hardware is not None and len(runtime.get("hardware_by_rank", [])) != 2:
        raise RuntimeError("pilot report lacks two-rank hardware provenance")
    return {"development_gate_passed": overall, "method_status": status}


def tree_hash(value: Any) -> str:
    import numpy as np
    import torch

    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode())
            digest.update(json.dumps(list(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b"ndarray\0")
            digest.update(str(array.dtype).encode())
            digest.update(json.dumps(list(array.shape)).encode())
            digest.update(array.tobytes())
        elif isinstance(item, Mapping):
            digest.update(b"mapping\0")
            for key in sorted(item, key=lambda value: str(value)):
                update(str(key))
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(b"sequence\0")
            for child in item:
                update(child)
        elif isinstance(item, bytes):
            digest.update(b"bytes\0" + item)
        elif item is None:
            digest.update(b"none\0")
        else:
            digest.update(type(item).__name__.encode() + b"\0")
            digest.update(repr(item).encode())
    update(value)
    return digest.hexdigest()


def tensors_are_finite(value: Any) -> bool:
    import torch

    if isinstance(value, torch.Tensor):
        return not value.is_floating_point() or bool(torch.isfinite(value).all())
    if isinstance(value, Mapping):
        return all(tensors_are_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(tensors_are_finite(item) for item in value)
    return True


def validate_runtime_contracts(value: Any, hardware: Mapping[str, Any]) -> str:
    if not isinstance(value, list) or len(value) != 2:
        raise RuntimeError("checkpoint must contain two runtime contracts")
    ranks: set[int] = set()
    for expected_rank, contract in enumerate(value):
        if not isinstance(contract, Mapping):
            raise RuntimeError("runtime contract must be a mapping")
        rank = contract.get("rank")
        if not isinstance(rank, int) or isinstance(rank, bool):
            raise RuntimeError("runtime rank must be an integer")
        ranks.add(rank)
        if (
            rank != expected_rank
            or contract.get("device_type") != "cuda"
            or contract.get("amp_enabled") is not True
            or contract.get("amp_dtype") != "torch.float16"
            or "TESLA T4" not in str(contract.get("device_name", "")).upper()
            or contract.get("device_capability") != [7, 5]
            or contract.get("determinism") != EXPECTED_DETERMINISM
            or contract.get("torch_version") != hardware.get("torch")
            or contract.get("cuda_version") != hardware.get("cuda")
        ):
            raise RuntimeError("checkpoint runtime provenance drifted")
    if ranks != {0, 1}:
        raise RuntimeError("checkpoint runtime ranks must be exactly {0,1}")
    return tree_hash(value)


def validate_rng_states(value: Any, *, hardware: Mapping[str, Any]) -> str:
    import numpy as np
    import torch

    if not isinstance(value, list) or len(value) != 2:
        raise RuntimeError("checkpoint must contain two rank RNG states")
    if hardware.get("device_count") != 2:
        raise RuntimeError("RNG validation requires the pinned two-device hardware contract")
    schema = hardware.get("rng_state_schema")
    if not isinstance(schema, Mapping) or set(schema) != {"torch_cpu", "torch_cuda"}:
        raise RuntimeError("hardware RNG schema is missing or malformed")
    cpu_schema = schema["torch_cpu"]
    cuda_schema = schema["torch_cuda"]
    expected_tensor_fields = {"device", "dtype", "ndim", "numel"}
    if (
        not isinstance(cpu_schema, Mapping)
        or set(cpu_schema) != expected_tensor_fields
        or cpu_schema.get("device") != "cpu"
        or cpu_schema.get("dtype") != "torch.uint8"
        or cpu_schema.get("ndim") != 1
        or not isinstance(cpu_schema.get("numel"), int)
        or isinstance(cpu_schema.get("numel"), bool)
        or cpu_schema.get("numel", 0) <= 1
        or not isinstance(cuda_schema, list)
        or len(cuda_schema) != 2
    ):
        raise RuntimeError("hardware RNG tensor schema is invalid")
    expected_cuda_numel: list[int] = []
    for index, item in enumerate(cuda_schema):
        if (
            not isinstance(item, Mapping)
            or set(item) != expected_tensor_fields | {"device_index"}
            or item.get("device_index") != index
            or item.get("device") != "cpu"
            or item.get("dtype") != "torch.uint8"
            or item.get("ndim") != 1
            or not isinstance(item.get("numel"), int)
            or isinstance(item.get("numel"), bool)
            or item.get("numel", 0) <= 1
        ):
            raise RuntimeError("hardware CUDA RNG tensor schema is invalid")
        expected_cuda_numel.append(int(item["numel"]))

    live_cpu = torch.get_rng_state().clone()
    if (
        live_cpu.dtype != torch.uint8
        or live_cpu.device.type != "cpu"
        or live_cpu.ndim != 1
        or live_cpu.numel() != cpu_schema["numel"]
    ):
        raise RuntimeError("hardware CPU RNG schema disagrees with the live framework")
    cuda_available = torch.cuda.is_available()
    live_cuda: list[Any] = []
    if cuda_available:
        if torch.cuda.device_count() != 2:
            raise RuntimeError("live CUDA device count disagrees with RNG hardware contract")
        live_cuda = [state.clone() for state in torch.cuda.get_rng_state_all()]
        if len(live_cuda) != 2 or any(
            state.dtype != torch.uint8
            or state.device.type != "cpu"
            or state.ndim != 1
            or state.numel() != expected_cuda_numel[index]
            for index, state in enumerate(live_cuda)
        ):
            raise RuntimeError("hardware CUDA RNG schema disagrees with the live framework")

    def validate_tensor(
        tensor: Any,
        *,
        expected_numel: int,
        label: str,
    ) -> None:
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.dtype != torch.uint8
            or tensor.device.type != "cpu"
            or tensor.ndim != 1
            or not tensor.is_contiguous()
            or tensor.numel() != expected_numel
        ):
            raise RuntimeError(f"{label} is not an exact restorable RNG byte tensor")

    saved_python = random.getstate()
    saved_numpy = np.random.get_state()
    saved_cpu = live_cpu
    saved_cuda = live_cuda
    try:
        for rank, state in enumerate(value):
            if not isinstance(state, Mapping) or set(state) != {
                "python",
                "numpy",
                "torch_cpu",
                "torch_cuda",
            }:
                raise RuntimeError(f"rank {rank} checkpoint RNG state schema is invalid")
            python_state = state["python"]
            numpy_state = state["numpy"]
            cpu_state = state["torch_cpu"]
            cuda_states = state["torch_cuda"]
            if not isinstance(python_state, tuple) or not isinstance(numpy_state, tuple):
                raise RuntimeError(f"rank {rank} Python/NumPy RNG state is invalid")
            validate_tensor(
                cpu_state,
                expected_numel=int(cpu_schema["numel"]),
                label=f"rank {rank} CPU RNG state",
            )
            if not isinstance(cuda_states, list) or len(cuda_states) != 2:
                raise RuntimeError(f"rank {rank} CUDA RNG state count is not two")
            for device_index, cuda_state in enumerate(cuda_states):
                validate_tensor(
                    cuda_state,
                    expected_numel=expected_cuda_numel[device_index],
                    label=f"rank {rank} CUDA:{device_index} RNG state",
                )

            try:
                random.setstate(python_state)
                if tree_hash(random.getstate()) != tree_hash(python_state):
                    raise RuntimeError("Python RNG state did not round-trip")
                np.random.set_state(numpy_state)
                if tree_hash(np.random.get_state()) != tree_hash(numpy_state):
                    raise RuntimeError("NumPy RNG state did not round-trip")
                torch.set_rng_state(cpu_state)
                if not torch.equal(torch.get_rng_state(), cpu_state):
                    raise RuntimeError("torch CPU RNG state did not round-trip")
                if cuda_available:
                    torch.cuda.set_rng_state_all(cuda_states)
                    restored_cuda = torch.cuda.get_rng_state_all()
                    if len(restored_cuda) != 2 or any(
                        not torch.equal(restored, expected)
                        for restored, expected in zip(restored_cuda, cuda_states, strict=True)
                    ):
                        raise RuntimeError("torch CUDA RNG states did not round-trip")
            except BaseException as error:
                raise RuntimeError(f"rank {rank} RNG state is not restorable") from error
    finally:
        random.setstate(saved_python)
        np.random.set_state(saved_numpy)
        torch.set_rng_state(saved_cpu)
        if cuda_available:
            torch.cuda.set_rng_state_all(saved_cuda)
    return tree_hash(value)


def validate_scaler_state(value: Any) -> str:
    if not isinstance(value, Mapping):
        raise RuntimeError("checkpoint scaler_state must be a mapping")
    required = {"scale", "growth_factor", "backoff_factor", "growth_interval", "_growth_tracker"}
    if not required.issubset(value):
        raise RuntimeError("checkpoint scaler_state is incomplete")
    if (
        finite_number(value["scale"], "scaler.scale") <= 0
        or finite_number(value["growth_factor"], "scaler.growth_factor") <= 1
        or not 0 < finite_number(value["backoff_factor"], "scaler.backoff_factor") < 1
        or int(value["growth_interval"]) <= 0
        or int(value["_growth_tracker"]) < 0
    ):
        raise RuntimeError("checkpoint scaler_state values are invalid")
    return tree_hash(value)


def validate_checkpoint_payload(
    payload: Any,
    *,
    label: str,
    expected_args: Mapping[str, Any],
    expected_epoch: int,
    expected_attempted: int,
    expected_capture_point: str,
    hardware: Mapping[str, Any],
    code_root: Path,
) -> dict[str, Any]:
    import torch

    if not isinstance(payload, Mapping):
        raise RuntimeError(f"{label} checkpoint must be a mapping")
    assert_recursive_safety(payload, label)
    assert_recursive_finite(payload, label)
    required = {
        "schema_version",
        "kind",
        "safe_for_submission",
        "model_config",
        "model_state",
        "metadata",
        "optimizer_state",
        "scaler_state",
        "training_state",
    }
    if not required.issubset(payload):
        raise RuntimeError(f"{label} checkpoint is missing {sorted(required - set(payload))}")
    if set(payload) != required:
        raise RuntimeError(f"{label} checkpoint has unexpected fields {sorted(set(payload) - required)}")
    if (
        payload.get("schema_version") != 1
        or payload.get("kind") != "puzzle_positional_diffusion"
        or payload.get("safe_for_submission") is not False
        or payload.get("model_config") != EXPECTED_MODEL_CONFIG
    ):
        raise RuntimeError(f"{label} checkpoint top-level schema drifted")
    model_state = payload["model_state"]
    if not isinstance(model_state, Mapping) or not model_state or not tensors_are_finite(model_state):
        raise RuntimeError(f"{label} model_state is invalid")
    if str(code_root / "src") not in sys.path:
        sys.path.insert(0, str(code_root / "src"))
    from puzzle_assembly.positional_diffusion import (
        PositionalDiffusionConfig,
        PositionalDiffusionNet,
    )
    model = PositionalDiffusionNet(PositionalDiffusionConfig(**payload["model_config"]))
    model.load_state_dict(model_state, strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_MODEL_PARAMETERS:
        raise RuntimeError(f"{label} model parameter count drifted")

    optimizer_state = payload["optimizer_state"]
    if (
        not isinstance(optimizer_state, Mapping)
        or not isinstance(optimizer_state.get("state"), Mapping)
        or not optimizer_state["state"]
        or not isinstance(optimizer_state.get("param_groups"), list)
        or not tensors_are_finite(optimizer_state)
    ):
        raise RuntimeError(f"{label} optimizer_state is invalid")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(expected_args["learning_rate"]),
        weight_decay=float(expected_args["weight_decay"]),
    )
    optimizer.load_state_dict(optimizer_state)
    scaler_hash = validate_scaler_state(payload["scaler_state"])

    metadata = payload["metadata"]
    if not isinstance(metadata, Mapping):
        raise RuntimeError(f"{label} metadata is invalid")
    if (
        metadata.get("schema_version") != 1
        or metadata.get("experiment") != "bounded_positional_diffusion_signal_pilot"
        or metadata.get("epoch") != expected_epoch
        or metadata.get("seed") != 20260711
        or metadata.get("train_source_count") != 384
        or not isinstance(metadata.get("train_source_names"), list)
        or len(metadata["train_source_names"]) != 384
        or metadata.get("train_source_names_sha256") != EXPECTED_TRAIN_NAMES_SHA256
        or hashlib.sha256(
            "\n".join(metadata["train_source_names"]).encode("utf-8")
        ).hexdigest()
        != EXPECTED_TRAIN_NAMES_SHA256
        or metadata.get("train_data_sha256") != EXPECTED_TRAIN_DATA_SHA256
        or metadata.get("training_arguments") != dict(expected_args)
        or metadata.get("warm_start_layout") != "softcycle"
        or metadata.get("development_targets_opened_during_training") is not False
        or metadata.get("competition_test_targets_opened") is not False
        or metadata.get("manifest_sha256") != EXPECTED_CONFIG_SHA256[
            "configs/denoise_splits_seed20260710.json"
        ]
        or metadata.get("quarantine_sha256") != EXPECTED_CONFIG_SHA256[
            "configs/denoise_validation_quarantine_v1.json"
        ]
        or metadata.get("code_sha256") != EXPECTED_PROVENANCE_CODE_SHA256
        or metadata.get("determinism") != EXPECTED_DETERMINISM
        or metadata.get("safe_for_submission") is not False
        or metadata.get("submission_ready") is not False
    ):
        raise RuntimeError(f"{label} checkpoint metadata provenance drifted")
    for asset_name, key in (("denoiser", "selected_tilenaf_synth_50k.pt"), ("hbt", "hbt_d320_denoised_rgb_sobel.pt")):
        asset = metadata.get(asset_name)
        if not isinstance(asset, Mapping) or asset.get("checkpoint_sha256") != EXPECTED_RUNTIME_SHA256[key]:
            raise RuntimeError(f"{label} {asset_name} provenance drifted")

    state = payload["training_state"]
    if not isinstance(state, Mapping):
        raise RuntimeError(f"{label} training_state is invalid")
    expected_state_keys = {
        "world_size",
        "gradient_accumulation",
        "completed_epoch",
        "next_epoch",
        "attempted_optimizer_steps",
        "successful_optimizer_steps",
        "skipped_optimizer_updates",
        "consecutive_skipped_optimizer_updates",
        "epoch_history",
        "rng_states_by_rank",
        "runtime_contracts_by_rank",
        "capture_point",
    }
    if set(state) != expected_state_keys:
        raise RuntimeError(f"{label} training_state field set drifted")
    counters = {
        name: state.get(name)
        for name in (
            "attempted_optimizer_steps",
            "successful_optimizer_steps",
            "skipped_optimizer_updates",
            "consecutive_skipped_optimizer_updates",
        )
    }
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in counters.values()
    ):
        raise RuntimeError(f"{label} training counters are invalid")
    attempted = counters["attempted_optimizer_steps"]
    successful = counters["successful_optimizer_steps"]
    skipped = counters["skipped_optimizer_updates"]
    consecutive = counters["consecutive_skipped_optimizer_updates"]
    if (
        state.get("world_size") != 2
        or state.get("gradient_accumulation") != 4
        or state.get("completed_epoch") != expected_epoch
        or state.get("next_epoch") != expected_epoch + 1
        or attempted != expected_attempted
        or successful + skipped != attempted
        or skipped > 8
        or consecutive > 3
        or state.get("capture_point") != expected_capture_point
        or metadata.get("attempted_optimizer_steps") != attempted
        or metadata.get("successful_optimizer_steps") != successful
        or metadata.get("optimizer_steps") != successful
        or metadata.get("skipped_optimizer_updates") != skipped
    ):
        raise RuntimeError(f"{label} epoch/counter cursor is incoherent")
    history = state.get("epoch_history")
    if not isinstance(history, list) or len(history) != expected_epoch + 1:
        raise RuntimeError(f"{label} epoch history length is incoherent")
    for epoch, record in enumerate(history):
        if not isinstance(record, Mapping):
            raise RuntimeError(f"{label} epoch history record is invalid")
        primary, independent = EXPECTED_CORRUPTION_COUNTS[epoch]
        record_attempted = 48 * (epoch + 1)
        if (
            record.get("epoch") != epoch
            or record.get("attempted_optimizer_steps") != record_attempted
            or not isinstance(record.get("successful_optimizer_steps"), int)
            or not isinstance(record.get("skipped_optimizer_updates"), int)
            or record["successful_optimizer_steps"] + record["skipped_optimizer_updates"]
            != record_attempted
            or record.get("global_examples") != 384
            or record.get("primary_examples") != primary
            or record.get("independent_examples") != independent
            or record.get("softcycle_warm_examples") != 384
            or record.get("w4_qap_warm_examples") != 0
        ):
            raise RuntimeError(f"{label} epoch {epoch} history is incoherent")
    if (
        history[-1]["attempted_optimizer_steps"] != attempted
        or history[-1]["successful_optimizer_steps"] != successful
        or history[-1]["skipped_optimizer_updates"] != skipped
    ):
        raise RuntimeError(f"{label} history tail disagrees with counters")
    runtime_hash = validate_runtime_contracts(state.get("runtime_contracts_by_rank"), hardware)
    rng_hash = validate_rng_states(state.get("rng_states_by_rank"), hardware=hardware)
    summary = {
        "label": label,
        "epoch": expected_epoch,
        "next_epoch": expected_epoch + 1,
        "attempted": attempted,
        "successful": successful,
        "skipped": skipped,
        "capture_point": expected_capture_point,
        "model_state_sha256": tree_hash(model_state),
        "optimizer_state_sha256": tree_hash(optimizer_state),
        "scaler_state_sha256": scaler_hash,
        "metadata_sha256": tree_hash(metadata),
        "history_sha256": tree_hash(history),
        "history_prefix_sha256": tree_hash(history[:-1]),
        "rng_sha256": rng_hash,
        "runtime_sha256": runtime_hash,
        "training_state_without_capture_sha256": tree_hash(
            {key: value for key, value in state.items() if key != "capture_point"}
        ),
    }
    del optimizer, model
    return summary


def load_and_validate_checkpoint(
    path: Path,
    **kwargs: Any,
) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    return validate_checkpoint_payload(payload, **kwargs)


def validate_checkpoint_relations(
    final: Mapping[str, Any],
    latest: Mapping[str, Any],
    previous: Mapping[str, Any],
) -> None:
    for key in (
        "epoch",
        "next_epoch",
        "attempted",
        "successful",
        "skipped",
        "model_state_sha256",
        "optimizer_state_sha256",
        "scaler_state_sha256",
        "metadata_sha256",
        "history_sha256",
        "rng_sha256",
        "runtime_sha256",
        "training_state_without_capture_sha256",
    ):
        if final.get(key) != latest.get(key):
            raise RuntimeError(f"final/latest checkpoint relation failed for {key}")
    if (
        previous.get("epoch") != latest.get("epoch", -1) - 1
        or previous.get("next_epoch") != latest.get("epoch")
        or previous.get("attempted", -1) + 48 != latest.get("attempted")
        or previous.get("successful", -1) > latest.get("successful", -1)
        or previous.get("skipped", -1) > latest.get("skipped", -1)
        or previous.get("history_sha256") != latest.get("history_prefix_sha256")
    ):
        raise RuntimeError("previous/latest checkpoint epoch chain is incoherent")


def verify_hash_manifest(path: Path, expected: Mapping[str, Path]) -> dict[str, str]:
    records: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if (
            not separator
            or not is_sha256(digest)
            or not name
            or name in records
        ):
            raise RuntimeError(f"malformed or duplicate SHA256SUMS line: {line!r}")
        records[name] = digest
    wanted = {name: sha256(value) for name, value in expected.items()}
    if records != wanted:
        raise RuntimeError(f"SHA256SUMS mismatch: expected {wanted}, found {records}")
    return records


def validate_artifacts(
    output_dir: Path,
    *,
    expected_args: Mapping[str, Any],
    hardware: Mapping[str, Any],
    code_root: Path,
) -> dict[str, Any]:
    report_path = output_dir / "positional_diffusion_report.json"
    final_path = output_dir / "positional_diffusion.pt"
    latest_path = output_dir / "positional_diffusion_latest.pt"
    previous_path = output_dir / "positional_diffusion_latest.pt.previous"
    hashes_path = output_dir / "SHA256SUMS.txt"
    paths = {
        report_path.name: report_path,
        final_path.name: final_path,
        latest_path.name: latest_path,
        previous_path.name: previous_path,
    }
    missing = [str(path) for path in (*paths.values(), hashes_path) if not path.is_file()]
    if missing:
        raise RuntimeError(f"successful pilot lacks required artifacts: {missing}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report_summary = validate_report_payload(
        report,
        expected_args=expected_args,
        hardware=hardware,
    )
    final = load_and_validate_checkpoint(
        final_path,
        label="final",
        expected_args=expected_args,
        expected_epoch=3,
        expected_attempted=192,
        expected_capture_point="final epoch boundary before final checkpoint save",
        hardware=hardware,
        code_root=code_root,
    )
    latest = load_and_validate_checkpoint(
        latest_path,
        label="latest",
        expected_args=expected_args,
        expected_epoch=3,
        expected_attempted=192,
        expected_capture_point="epoch boundary after optimizer update and before checkpoint save",
        hardware=hardware,
        code_root=code_root,
    )
    previous = load_and_validate_checkpoint(
        previous_path,
        label="previous",
        expected_args=expected_args,
        expected_epoch=2,
        expected_attempted=144,
        expected_capture_point="epoch boundary after optimizer update and before checkpoint save",
        hardware=hardware,
        code_root=code_root,
    )
    validate_checkpoint_relations(final, latest, previous)
    training = report.get("training")
    if not isinstance(training, Mapping) or (
        training.get("attempted_optimizer_steps") != final["attempted"]
        or training.get("successful_optimizer_steps") != final["successful"]
        or training.get("skipped_optimizer_updates") != final["skipped"]
        or training.get("start_epoch") != 0
        or training.get("resumed_from") is not None
        or training.get("resume_used_previous_fallback") is not False
        or training.get("epochs") is None
        or tree_hash(training.get("epochs")) != final["history_sha256"]
        or training.get("resume_cursor")
        != {"next_epoch": 4, "attempted_optimizer_steps": 192}
    ):
        raise RuntimeError("report training summary disagrees with final checkpoint")
    manifest = verify_hash_manifest(hashes_path, paths)
    return {
        "report": str(report_path),
        "report_sha256": sha256(report_path),
        "development_gate_passed": report_summary["development_gate_passed"],
        "method_status": report_summary["method_status"],
        "final_checkpoint": final,
        "latest_checkpoint": latest,
        "previous_checkpoint": previous,
        "artifact_sha256": manifest,
        "hashes_path": str(hashes_path),
        "hashes_sha256": sha256(hashes_path),
    }


def execute(wrapper: dict[str, Any]) -> None:
    commands = wrapper["commands"]
    assert isinstance(commands, list)
    mounts = require_exact_mounts()
    data_root = find_data_root()
    runtime_root = find_runtime_root()
    base_root, base_archive = find_base_root()
    overlay_root, overlay_archive = find_overlay_root()
    code_root = WORKING / "positional_diffusion_code"
    code_hashes = copy_and_verify_code(base_root, overlay_root, code_root)
    denoiser = runtime_root / "selected_tilenaf_synth_50k.pt"
    hbt = runtime_root / "hbt_d320_denoised_rgb_sobel.pt"
    runtime_hashes = verify_hashes(runtime_root, EXPECTED_RUNTIME_SHA256, "runtime asset")
    manifest = code_root / "configs/denoise_splits_seed20260710.json"
    quarantine = code_root / "configs/denoise_validation_quarantine_v1.json"
    dataset_provenance = verify_dataset_slices(
        data_root,
        code_root=code_root,
        manifest=manifest,
        quarantine=quarantine,
    )
    output_dir = WORKING / "positional_diffusion_pilot"
    args = expected_arguments(
        data_root=data_root,
        manifest=manifest,
        quarantine=quarantine,
        denoiser=denoiser,
        hbt=hbt,
        output_dir=output_dir,
    )
    wrapper["inputs"] = {
        "exact_mounts": {name: str(path) for name, path in mounts.items()},
        "data_root": str(data_root),
        "runtime_root": str(runtime_root),
        "base_root": str(base_root),
        "base_archive": None if base_archive is None else str(base_archive),
        "base_archive_sha256": None if base_archive is None else sha256(base_archive),
        "overlay_root": str(overlay_root),
        "overlay_archive": None if overlay_archive is None else str(overlay_archive),
        "overlay_archive_sha256": None
        if overlay_archive is None
        else sha256(overlay_archive),
        "runtime_sha256": runtime_hashes,
        "executed_code_sha256": code_hashes,
        "config_sha256": verify_hashes(code_root, EXPECTED_CONFIG_SHA256, "config"),
        "dataset_provenance": dataset_provenance,
        "runner_sha256": sha256(Path(__file__)),
    }
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(code_root / "src")
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    wrapper["hardware"] = hardware_probe(
        commands,
        cwd=code_root,
        environment=environment,
    )
    model = code_root / "src/puzzle_assembly/positional_diffusion.py"
    trainer = code_root / "scripts/train_evaluate_positional_diffusion.py"
    tests = code_root / "tests/test_positional_diffusion.py"
    run_checked(
        [sys.executable, "-m", "py_compile", str(model), str(trainer), str(tests)],
        cwd=code_root,
        environment=environment,
        label="pycompile",
        telemetry=commands,
    )
    pytest_result = run_checked(
        [sys.executable, "-m", "pytest", "-q", str(tests)],
        cwd=code_root,
        environment=environment,
        label="pytest_exact_23",
        telemetry=commands,
        capture=True,
    )
    passed = [
        int(value)
        for value in re.findall(
            r"\b(\d+) passed\b",
            f"{pytest_result.stdout}\n{pytest_result.stderr}",
        )
    ]
    if passed != [EXPECTED_TEST_COUNT]:
        raise RuntimeError(f"expected exactly 23 tests, saw {passed}")
    dry_dir = WORKING / "positional_diffusion_dry_run"
    dry_command = [
        sys.executable,
        str(trainer),
        "--mode",
        "dry-run",
        "--device",
        "cuda:0",
        "--output-dir",
        str(dry_dir),
        "--model-dim",
        "384",
        "--cnn-channels",
        "64",
        "--layers",
        "8",
        "--heads",
        "12",
        "--feedforward-dim",
        "1536",
        "--dropout",
        "0.05",
        "--diffusion-steps",
        "300",
        "--sampling-steps",
        "30",
        "--tile-encode-chunk",
        "192",
        "--activation-checkpointing",
    ]
    run_checked(
        dry_command,
        cwd=code_root,
        environment=environment,
        label="cuda_dry_run",
        telemetry=commands,
    )
    command = pilot_command(trainer, args)
    run_checked(
        command,
        cwd=code_root,
        environment=environment,
        label="torchrun_t4x2_full_explicit",
        telemetry=commands,
    )
    wrapper["pilot_command"] = command
    wrapper["validation"] = validate_artifacts(
        output_dir,
        expected_args=args,
        hardware=wrapper["hardware"],
        code_root=code_root,
    )
    wrapper["status"] = "complete"


def main() -> None:
    started = time.perf_counter()
    wrapper: dict[str, Any] = {
        "schema_version": 1,
        "kind": "positional_diffusion_t4x2_pilot_wrapper",
        "status": "running",
        "safe_for_submission": False,
        "submission_ready": False,
        "commands": [],
    }
    exit_code = 0
    atomic_json(WRAPPER_PATH, wrapper)
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
        atomic_json(WRAPPER_PATH, wrapper)
        wrapper["wrapper_sha256"] = sha256(WRAPPER_PATH)
        print(json.dumps({"event": "wrapper_final", **wrapper}, default=str), flush=True)
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run two pinned 5x5 TileNAF ablations on T4x2 and open one frozen gate."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import zipfile


WORKING = Path("/kaggle/working")
INPUT_OWNER = Path("/kaggle/input/datasets/pasha883")
DATA_MOUNT = INPUT_OWNER / "vsos-ai-initiative-pazzle"
RUNTIME_MOUNT = INPUT_OWNER / "vsos-assembly-v1-runtime"
CODE_MOUNT = INPUT_OWNER / "vsos-denoise-block5x5-code"
CODE_ARCHIVE_SHA256 = "839e5c11023a4ff02a49478c522762366089ac542b3dd71f15b39f58dc393e46"
CODE_MANIFEST_SHA256 = "89263beabf5ca58c93da0a4c6f8260ffa8ee07b52f3fc09855ca2f079a0c982d"
PROTOCOL_SHA256 = "6f6fd7f1fd0e35661cb804f40261ae301163f746861574d032b9ff92a3f0a01b"
INIT_SHA256 = "77a2d8607c9bc6b80e7aa99ed03329c1fae6bf94ee1d9a654241ab93a05cc734"
WRAPPER_PATH = WORKING / "block5x5_wrapper.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def exactly_one(paths: list[Path], label: str) -> Path:
    values = sorted({path.resolve() for path in paths if path.is_file()})
    if len(values) != 1:
        raise RuntimeError(f"expected exactly one {label}, found {values}")
    return values[0]


def verify_code_tree(destination: Path) -> Path:
    manifest_path = destination / "MANIFEST.json"
    if sha256(manifest_path) != CODE_MANIFEST_SHA256:
        raise RuntimeError("code bundle manifest SHA mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "denoise_block5x5_kaggle_code_bundle":
        raise RuntimeError("unexpected code bundle manifest")
    expected_files = {record["path"] for record in manifest["files"]} | {"MANIFEST.json"}
    actual_files = {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        raise RuntimeError(
            f"code bundle tree mismatch: missing={sorted(expected_files-actual_files)}, "
            f"extra={sorted(actual_files-expected_files)}"
        )
    for record in manifest["files"]:
        path = destination / record["path"]
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise RuntimeError(f"code bundle file mismatch: {record['path']}")
    return destination


def safe_extract(archive: Path, destination: Path) -> Path:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    with zipfile.ZipFile(archive) as handle:
        members = handle.infolist()
        for member in members:
            path = Path(member.filename)
            mode = (member.external_attr >> 16) & 0o170000
            if member.filename.startswith("/") or ".." in path.parts or mode == 0o120000:
                raise RuntimeError(f"unsafe code archive member: {member.filename}")
        handle.extractall(destination)
    return verify_code_tree(destination)


def find_code() -> tuple[Path, dict]:
    if not CODE_MOUNT.is_dir():
        raise FileNotFoundError(f"missing exact code mount {CODE_MOUNT}")
    archives = list(CODE_MOUNT.rglob("denoise_block5x5_code.zip"))
    if archives:
        archive = exactly_one(archives, "code archive")
        actual = sha256(archive)
        if actual != CODE_ARCHIVE_SHA256:
            raise RuntimeError(f"code archive SHA mismatch: {actual}")
        root = safe_extract(archive, WORKING / "block5x5_code")
        record = {
            "mode": "archive",
            "mount": str(CODE_MOUNT),
            "archive": str(archive),
            "archive_sha256": actual,
        }
    else:
        manifest = exactly_one(list(CODE_MOUNT.rglob("MANIFEST.json")), "extracted manifest")
        root = verify_code_tree(manifest.parent)
        record = {
            "mode": "kaggle_auto_extracted_zip",
            "mount": str(CODE_MOUNT),
            "root": str(root),
            "manifest_sha256": sha256(manifest),
            "source_archive_sha256": CODE_ARCHIVE_SHA256,
        }
    protocol = root / "configs" / "denoise_block5x5_v1.json"
    if sha256(protocol) != PROTOCOL_SHA256:
        raise RuntimeError("protocol SHA mismatch after code staging")
    return root, record


def find_data_root() -> Path:
    if not DATA_MOUNT.is_dir():
        raise FileNotFoundError(f"missing exact puzzle mount {DATA_MOUNT}")
    candidates = []
    for targets in DATA_MOUNT.rglob("train/targets"):
        inputs = targets.parent / "inputs"
        test = targets.parents[1] / "test"
        if (
            targets.is_dir()
            and inputs.is_dir()
            and test.is_dir()
            and len(list(targets.glob("*.png"))) == 7000
            and len(list(inputs.glob("*.png"))) == 7000
            and len(list(test.glob("*.png"))) == 700
        ):
            candidates.append(targets.parents[1])
    values = sorted({path.resolve() for path in candidates})
    if len(values) != 1:
        raise RuntimeError(f"expected one complete puzzle root, found {values}")
    return values[0]


def find_init_checkpoint() -> Path:
    if not RUNTIME_MOUNT.is_dir():
        raise FileNotFoundError(f"missing exact runtime mount {RUNTIME_MOUNT}")
    path = exactly_one(
        list(RUNTIME_MOUNT.rglob("selected_tilenaf_synth_50k.pt")),
        "selected TileNAF initialization",
    )
    actual = sha256(path)
    if actual != INIT_SHA256:
        raise RuntimeError(f"initial checkpoint SHA mismatch: {actual}")
    return path


def gpu_preflight() -> dict:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError(
            f"block5x5 job requires T4x2, cuda={torch.cuda.is_available()} "
            f"count={torch.cuda.device_count()}"
        )
    devices = []
    for index in range(2):
        name = torch.cuda.get_device_name(index)
        if "T4" not in name.upper():
            raise RuntimeError(f"expected Tesla T4 at index {index}, got {name}")
        device = torch.device("cuda", index)
        left = torch.randn(512, 512, device=device)
        right = torch.randn(512, 512, device=device)
        devices.append(
            {
                "index": index,
                "name": name,
                "capability": list(torch.cuda.get_device_capability(index)),
                "total_memory": int(torch.cuda.get_device_properties(index).total_memory),
                "matmul_mean": float((left @ right).mean().cpu()),
            }
        )
    smi = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    ).stdout.strip().splitlines()
    return {
        "torch": torch.__version__,
        "compiled_cuda": torch.version.cuda,
        "arch_list": torch.cuda.get_arch_list(),
        "device_count": 2,
        "devices": devices,
        "nvidia_smi": smi,
    }


def run_logged(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    timeout: int,
) -> dict:
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=timeout,
        )
    return {
        "command": command,
        "returncode": completed.returncode,
        "seconds": time.perf_counter() - started,
        "log": str(log_path),
        "log_sha256": sha256(log_path),
    }


def train_command(
    *,
    code_root: Path,
    data_root: Path,
    init_checkpoint: Path,
    output: Path,
    variant: str,
    config: dict,
) -> list[str]:
    return [
        sys.executable,
        str(code_root / "scripts" / "train_denoise_block5x5.py"),
        "--data-root",
        str(data_root),
        "--manifest",
        str(code_root / "configs" / "denoise_splits_seed20260710.json"),
        "--protocol",
        str(code_root / "configs" / "denoise_block5x5_v1.json"),
        "--init-checkpoint",
        str(init_checkpoint),
        "--output",
        str(output),
        "--variant",
        variant,
        "--train-images",
        str(config["train_images"]),
        "--steps",
        str(config["steps"]),
        "--block-batch-size",
        str(config["block_batch_size"]),
        "--eval-batch-size",
        "512",
        "--learning-rate",
        str(config["learning_rate"]),
        "--weight-decay",
        str(config["weight_decay"]),
        "--ema-decay",
        str(config["ema_decay"]),
        "--eval-interval",
        str(config["eval_interval"]),
        "--log-interval",
        "50",
        "--seed",
        str(config["variants"][variant]["seed"]),
        "--device",
        "cuda",
        "--tile-ssim",
        str(config["variants"][variant]["tile_ssim"]),
        "--tile-gradient",
        str(config["variants"][variant]["tile_gradient"]),
        "--tile-boundary-extra",
        str(config["variants"][variant]["tile_boundary_extra"]),
        "--block-ssim",
        str(config["variants"][variant]["block_ssim"]),
        "--block-gradient",
        str(config["variants"][variant]["block_gradient"]),
        "--seam-gradient",
        str(config["variants"][variant]["seam_gradient"]),
        "--neighbour-mean",
        str(config["variants"][variant]["neighbour_mean"]),
    ]


def launch_parallel_training(
    code_root: Path,
    data_root: Path,
    init_checkpoint: Path,
    protocol: dict,
    base_env: dict[str, str],
) -> dict[str, dict]:
    processes: dict[str, subprocess.Popen] = {}
    logs: dict[str, object] = {}
    handles = {}
    started = {}
    outputs = {
        "moderate": WORKING / "block5x5_moderate.pt",
        "strong": WORKING / "block5x5_strong.pt",
    }
    try:
        for gpu, variant in enumerate(("moderate", "strong")):
            env = dict(base_env)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["PYTHONHASHSEED"] = str(protocol["training"]["variants"][variant]["seed"])
            env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            env["OMP_NUM_THREADS"] = "2"
            log_path = WORKING / f"block5x5_{variant}.log"
            handle = log_path.open("w", encoding="utf-8")
            handles[variant] = handle
            command = train_command(
                code_root=code_root,
                data_root=data_root,
                init_checkpoint=init_checkpoint,
                output=outputs[variant],
                variant=variant,
                config=protocol["training"],
            )
            started[variant] = time.perf_counter()
            processes[variant] = subprocess.Popen(
                command,
                cwd=code_root,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            logs[variant] = {"command": command, "log": str(log_path), "gpu": gpu}

        deadline = time.monotonic() + 9000
        while True:
            statuses = {variant: process.poll() for variant, process in processes.items()}
            failed = [variant for variant, status in statuses.items() if status not in (None, 0)]
            if failed:
                for process in processes.values():
                    if process.poll() is None:
                        process.terminate()
                raise RuntimeError(f"parallel training failed: {statuses}")
            if all(status == 0 for status in statuses.values()):
                break
            if time.monotonic() > deadline:
                for process in processes.values():
                    if process.poll() is None:
                        process.kill()
                raise TimeoutError("parallel block5x5 training exceeded 9000 seconds")
            time.sleep(5)
    finally:
        for handle in handles.values():
            handle.close()
    for variant, record in logs.items():
        log_path = Path(record["log"])
        output = outputs[variant]
        if not output.is_file():
            raise FileNotFoundError(f"training did not produce {output}")
        record.update(
            {
                "returncode": processes[variant].returncode,
                "seconds": time.perf_counter() - started[variant],
                "log_sha256": sha256(log_path),
                "checkpoint": str(output),
                "checkpoint_sha256": sha256(output),
            }
        )
    return logs


def main() -> int:
    started = time.perf_counter()
    state: dict[str, object] = {
        "schema_version": 1,
        "kind": "denoise_block5x5_t4x2_wrapper",
        "status": "running",
        "kernel_slug": "pasha883/vsos-denoise-true-block5x5-t4x2",
        "dataset_slugs": {
            "data": "pasha883/vsos-ai-initiative-pazzle",
            "runtime": "pasha883/vsos-assembly-v1-runtime",
            "code": "pasha883/vsos-denoise-block5x5-code",
        },
        "code_archive_sha256": CODE_ARCHIVE_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "init_checkpoint_sha256": INIT_SHA256,
    }
    try:
        code_root, code_record = find_code()
        data_root = find_data_root()
        init_checkpoint = find_init_checkpoint()
        protocol = json.loads(
            (code_root / "configs" / "denoise_block5x5_v1.json").read_text(encoding="utf-8")
        )
        hardware = gpu_preflight()
        state.update(
            {
                "code": code_record,
                "data_root": str(data_root),
                "init_checkpoint": str(init_checkpoint),
                "hardware": hardware,
            }
        )
        base_env = dict(os.environ)
        base_env["PYTHONPATH"] = str(code_root / "src")
        tests = run_logged(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/test_denoise_block5x5.py",
            ],
            cwd=code_root,
            env=base_env,
            log_path=WORKING / "block5x5_tests.log",
            timeout=180,
        )
        state["tests"] = tests
        if tests["returncode"] != 0:
            raise RuntimeError("focused block5x5 tests failed")

        training = launch_parallel_training(
            code_root, data_root, init_checkpoint, protocol, base_env
        )
        state["training"] = training
        selection_path = WORKING / "block5x5_selection.json"
        selection_log = WORKING / "block5x5_selection.log"
        selection_run = run_logged(
            [
                sys.executable,
                str(code_root / "scripts" / "select_denoise_block5x5_candidate.py"),
                "--protocol",
                str(code_root / "configs" / "denoise_block5x5_v1.json"),
                "--candidate",
                str(WORKING / "block5x5_moderate.pt"),
                "--candidate",
                str(WORKING / "block5x5_strong.pt"),
                "--output",
                str(selection_path),
            ],
            cwd=code_root,
            env=base_env,
            log_path=selection_log,
            timeout=300,
        )
        if selection_run["returncode"] != 0:
            raise RuntimeError("development selector failed")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        state["selection"] = {
            **selection_run,
            "path": str(selection_path),
            "sha256": sha256(selection_path),
            "decision": selection["decision"],
            "selected_variant": selection["selected_variant"],
            "selected_checkpoint_sha256": selection["selected_checkpoint_sha256"],
        }
        if selection["decision"] != "open_frozen_gate":
            state["status"] = "stop_no_development_signal"
            state["frozen_gate_accessed"] = False
            state["seconds"] = time.perf_counter() - started
            atomic_json(WRAPPER_PATH, state)
            print(json.dumps({"event": "block5x5_wrapper_complete", **state}, sort_keys=True))
            return 0

        gate_path = WORKING / "block5x5_frozen_gate.json"
        gate_env = dict(base_env)
        gate_env["CUDA_VISIBLE_DEVICES"] = "0"
        gate_run = run_logged(
            [
                sys.executable,
                str(code_root / "scripts" / "evaluate_denoise_block5x5.py"),
                "--data-root",
                str(data_root),
                "--manifest",
                str(code_root / "configs" / "denoise_splits_seed20260710.json"),
                "--protocol",
                str(code_root / "configs" / "denoise_block5x5_v1.json"),
                "--selection",
                str(selection_path),
                "--current-checkpoint",
                str(init_checkpoint),
                "--candidate-checkpoint",
                str(selection["selected_checkpoint"]),
                "--output",
                str(gate_path),
                "--device",
                "cuda",
                "--batch-size",
                "512",
                "--torch-threads",
                "4",
            ],
            cwd=code_root,
            env=gate_env,
            log_path=WORKING / "block5x5_frozen_gate.log",
            timeout=9000,
        )
        if gate_run["returncode"] != 0:
            raise RuntimeError("frozen gate evaluator failed")
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        state["gate"] = {
            **gate_run,
            "path": str(gate_path),
            "sha256": sha256(gate_path),
            "verdict": gate["verdict"],
            "candidate_minus_current": gate["candidate_minus_current"],
            "checks": gate["checks"],
            "retrieval_recall_at_1_delta": gate["retrieval_recall_at_1_delta"],
            "qap_diagnostic": gate["qap_diagnostic"],
        }
        state["frozen_gate_accessed"] = True
        state["status"] = "promoted" if gate["verdict"] == "promote" else "rejected_keep_current"
        if gate["verdict"] == "promote":
            promoted = WORKING / "promoted_block5x5.pt"
            shutil.copy2(selection["selected_checkpoint"], promoted)
            state["promoted_checkpoint"] = {
                "path": str(promoted),
                "sha256": sha256(promoted),
            }
        state["seconds"] = time.perf_counter() - started
        atomic_json(WRAPPER_PATH, state)
        print(json.dumps({"event": "block5x5_wrapper_complete", **state}, sort_keys=True))
        return 0
    except Exception as error:
        state["status"] = "error"
        state["error"] = {"type": type(error).__name__, "message": str(error)}
        state["traceback"] = traceback.format_exc()
        state["seconds"] = time.perf_counter() - started
        atomic_json(WRAPPER_PATH, state)
        print(json.dumps({"event": "block5x5_wrapper_error", **state}, sort_keys=True))
        raise


if __name__ == "__main__":
    raise SystemExit(main())

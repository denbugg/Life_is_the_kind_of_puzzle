#!/usr/bin/env python3
"""Hash-pinned 2xT4 runner for the TileNAF-latent Stage-1 gate."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
STAGING = WORKING / "latent_edge_staging"
WRAPPER = WORKING / "latent_edge_embedding_wrapper.json"
BASE_MANIFEST_SHA256 = "1583053a4276ba9b30368dcc6af00f24cd0fe091bfaff93090c41c01b0c3675b"
ASSET_HASHES = {
    "selected_tilenaf_synth_50k.pt": "77a2d8607c9bc6b80e7aa99ed03329c1fae6bf94ee1d9a654241ab93a05cc734",
    "hbt_d320_denoised_rgb_sobel.pt": "c2589bff58573d592227d522954a7ed6c22f6fec7ecbbea2bf0c755d7a720787",
}
OVERLAY_HASHES = {
    "src/puzzle_denoise_v2/model.py": "d111d106352715cdebbbf24fd6c99facd0b3c9520951abf588a6c6bc03c7feba",
    "src/puzzle_assembly/latent_edge_embedding.py": "55230824629c36781305211ea642724b6ae99adaf445d0942e4b8b326cebf04e",
    "scripts/train_evaluate_latent_edge_embedding.py": "2d8e7ee2a3ae2014ed5eb3818328e6262c2facefeb158949780b87fe0290b19a",
    "tests/test_latent_edge_embedding.py": "418b7ca5cee17232f5e4ea93c0ae247674384278493da418c3d6ad67bc21e5bb",
    "tests/test_latent_edge_trainer.py": "5afe50b66c4e915c96030ecb93a24530eb65484a46d0df467069d56ae2d31de6",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exactly_one(paths: list[Path], label: str) -> Path:
    values = sorted({path.resolve() for path in paths})
    if len(values) != 1:
        raise RuntimeError(f"expected exactly one {label}, got {values}")
    return values[0]


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def find_hash_pinned(filename: str, expected: str) -> Path:
    return exactly_one(
        [path for path in INPUT.rglob(filename) if path.is_file() and sha256(path) == expected],
        f"hash-pinned {filename}",
    )


def verify_base() -> tuple[Path, dict]:
    manifest_path = find_hash_pinned(
        "masked_gap_code_manifest_v1.json", BASE_MANIFEST_SHA256
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "masked_gap_recursive_code_manifest_v1":
        raise RuntimeError("base code manifest kind mismatch")
    files = manifest.get("file_sha256")
    if not isinstance(files, dict) or not files:
        raise RuntimeError("base manifest has no executable closure")
    root = manifest_path.parent
    for relative, expected in files.items():
        path = (root / relative).resolve(strict=True)
        path.relative_to(root.resolve())
        if sha256(path) != expected:
            raise RuntimeError(f"base closure hash mismatch: {relative}")
    return root, manifest


def stage_code() -> dict:
    base_root, manifest = verify_base()
    # Kaggle expands uploaded ZIP datasets before mounting them.  Resolve the
    # unpacked root by the complete allowlisted file closure, not by the local
    # upload-container filename.
    overlay_roots = []
    for marker in INPUT.rglob("src/puzzle_assembly/latent_edge_embedding.py"):
        root = marker.parents[2]
        if all(
            (root / relative).is_file()
            and sha256(root / relative) == expected
            for relative, expected in OVERLAY_HASHES.items()
        ):
            overlay_roots.append(root)
    overlay_root = exactly_one(overlay_roots, "hash-pinned unpacked overlay root")
    if STAGING.exists():
        shutil.rmtree(STAGING)
    shutil.copytree(base_root, STAGING)
    for relative, expected in OVERLAY_HASHES.items():
        source = overlay_root / relative
        if sha256(source) != expected:
            raise RuntimeError(f"overlay member hash mismatch: {relative}")
        destination = STAGING / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    staged = {relative: sha256(STAGING / relative) for relative in OVERLAY_HASHES}
    if staged != OVERLAY_HASHES:
        raise RuntimeError("staged overlay hashes drifted")
    return {
        "base_root": str(base_root),
        "base_manifest_sha256": BASE_MANIFEST_SHA256,
        "base_file_count": len(manifest["file_sha256"]),
        "overlay_root": str(overlay_root),
        "overlay_hashes": staged,
        "staging": str(STAGING),
    }


def find_data_root() -> Path:
    candidates = []
    for targets in INPUT.rglob("train/targets"):
        inputs = targets.parent / "inputs"
        if (
            targets.is_dir()
            and inputs.is_dir()
            and len(list(targets.glob("*.png"))) == 7000
            and len(list(inputs.glob("*.png"))) == 7000
        ):
            candidates.append(targets.parent.parent)
    return exactly_one(candidates, "7000-source puzzle root")


def gpu_preflight() -> dict:
    code = r'''import json, torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
    raise RuntimeError("requires exactly two CUDA devices")
devices=[]
for index in range(2):
    name=torch.cuda.get_device_name(index)
    capability=list(torch.cuda.get_device_capability(index))
    if "T4" not in name.upper() or capability != [7,5]:
        raise RuntimeError(f"device {index} is not T4 sm75: {name}, {capability}")
    value=torch.randn((128,128), device=f"cuda:{index}")
    devices.append({"index":index,"name":name,"capability":capability,
                    "total_memory":int(torch.cuda.get_device_properties(index).total_memory),
                    "tensor_probe":float((value@value).mean().cpu())})
print(json.dumps({"torch":torch.__version__,"cuda":torch.version.cuda,"devices":devices}))'''
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )
    result = json.loads(completed.stdout)
    result["nvidia_smi"] = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    ).stdout.strip().splitlines()
    return result


def run_logged(
    command: list[str],
    *,
    env: dict[str, str],
    label: str,
    log_path: Path,
    timeout_seconds: int,
) -> dict:
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=STAGING,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=timeout_seconds)
            timed_out = False
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                returncode = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                returncode = process.wait(timeout=30)
            returncode = None
            timed_out = True
    return {
        "label": label,
        "command": command,
        "returncode": returncode,
        "timed_out": timed_out,
        "timeout_seconds": timeout_seconds,
        "seconds": time.perf_counter() - started,
        "log": str(log_path),
        "log_sha256": sha256(log_path),
    }


def require_success(record: dict) -> None:
    if record["returncode"] == 0 and record["timed_out"] is False:
        return
    log = Path(record["log"]).read_text(encoding="utf-8", errors="replace")
    raise RuntimeError(
        f"{record['label']} failed returncode={record['returncode']} "
        f"timed_out={record['timed_out']}:\n" + "\n".join(log.splitlines()[-100:])
    )


def validate_report(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("kind") != "tilenaf_latent_edge_stage1_report":
        raise RuntimeError("unexpected latent-edge report kind")
    if report.get("safe_for_submission") is not False or report.get("qap_run") is not False:
        raise RuntimeError("Stage-1 report is not fail-closed")
    allowed = {
        "stop_selection_retrieval",
        "stop_holdout_retrieval",
        "retrieval_gate_passed_qap_not_run",
    }
    if report.get("status") not in allowed:
        raise RuntimeError(f"unexpected Stage-1 status: {report.get('status')}")
    selection = report.get("selection", {}).get("gate", {})
    holdout_opened = report.get("partitions", {}).get("holdout", {}).get("opened")
    if bool(selection.get("passed")) != bool(holdout_opened):
        raise RuntimeError("selection decision and holdout opening disagree")
    holdout = report.get("holdout")
    status = report.get("status")
    if status == "stop_selection_retrieval":
        if selection.get("passed") is not False or holdout is not None:
            raise RuntimeError("selection-stop report opened or recorded holdout")
    elif status == "stop_holdout_retrieval":
        if selection.get("passed") is not True or not isinstance(holdout, dict):
            raise RuntimeError("holdout-stop report lacks a passed selection")
        if holdout.get("gate", {}).get("passed") is not False:
            raise RuntimeError("holdout-stop report does not contain a failed holdout")
    elif status == "retrieval_gate_passed_qap_not_run":
        if (
            selection.get("passed") is not True
            or not isinstance(holdout, dict)
            or holdout.get("gate", {}).get("passed") is not True
        ):
            raise RuntimeError("retrieval-pass status is inconsistent with its gates")
    checkpoint = Path(str(report.get("checkpoint", "")))
    if not checkpoint.is_file():
        raise RuntimeError("reported latent-edge checkpoint is missing")
    checkpoint_sha256 = sha256(checkpoint)
    if checkpoint_sha256 != report.get("checkpoint_sha256"):
        raise RuntimeError("reported checkpoint SHA256 does not match the artifact")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "checkpoint_sha256": checkpoint_sha256,
        "status": status,
        "selected_alpha": selection.get("selected_alpha"),
        "selection_passed": selection.get("passed"),
        "holdout_opened": holdout_opened,
    }


def main() -> None:
    wrapper = {
        "schema_version": 1,
        "kind": "tilenaf_latent_edge_kaggle_wrapper",
        "status": "starting",
        "safe_for_submission": False,
        "qap_run": False,
        "steps": [],
        "started_unix": time.time(),
    }
    atomic_json(WRAPPER, wrapper)
    try:
        staging = stage_code()
        data_root = find_data_root()
        denoiser = find_hash_pinned(
            "selected_tilenaf_synth_50k.pt",
            ASSET_HASHES["selected_tilenaf_synth_50k.pt"],
        )
        hbt = find_hash_pinned(
            "hbt_d320_denoised_rgb_sobel.pt",
            ASSET_HASHES["hbt_d320_denoised_rgb_sobel.pt"],
        )
        hardware = gpu_preflight()
        wrapper.update(
            {
                "status": "staged",
                "staging": staging,
                "data_root": str(data_root),
                "assets": {
                    "denoiser": {"path": str(denoiser), "sha256": sha256(denoiser)},
                    "hbt": {"path": str(hbt), "sha256": sha256(hbt)},
                },
                "hardware": hardware,
            }
        )
        atomic_json(WRAPPER, wrapper)

        env = dict(os.environ)
        env.update(
            {
                "PYTHONPATH": str(STAGING / "src"),
                "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2",
                "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
                "PYTHONUNBUFFERED": "1",
            }
        )
        tests = run_logged(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/test_latent_edge_embedding.py",
                "tests/test_latent_edge_trainer.py",
            ],
            env=env,
            label="overlay unit tests",
            log_path=WORKING / "latent_edge_tests.log",
            timeout_seconds=900,
        )
        wrapper["steps"].append(tests)
        atomic_json(WRAPPER, wrapper)
        require_success(tests)

        common = [
            "--data-root",
            str(data_root),
            "--denoiser",
            str(denoiser),
            "--hbt-checkpoint",
            str(hbt),
            "--manifest",
            str(STAGING / "configs/denoise_splits_seed20260710.json"),
            "--quarantine",
            str(STAGING / "configs/denoise_validation_quarantine_v1.json"),
            # T4 fp16 produced non-finite Transformer gradients in the bounded
            # smoke.  The 1.68M-parameter model fits comfortably in fp32.
            "--no-amp",
        ]
        smoke_output = WORKING / "latent_edge_smoke"
        smoke = run_logged(
            [
                "torchrun",
                "--standalone",
                "--nproc_per_node=2",
                "scripts/train_evaluate_latent_edge_embedding.py",
                "--output-dir",
                str(smoke_output),
                "--smoke",
                "--selection-split",
                "edge_development",
                "--selection-offset",
                "96",
                "--holdout-offset",
                "112",
                *common,
            ],
            env=env,
            label="2xT4 end-to-end smoke",
            log_path=WORKING / "latent_edge_smoke.log",
            timeout_seconds=3600,
        )
        wrapper["steps"].append(smoke)
        atomic_json(WRAPPER, wrapper)
        require_success(smoke)
        smoke["report"] = validate_report(smoke_output / "latent_edge_embedding_report.json")
        atomic_json(WRAPPER, wrapper)

        pilot_output = WORKING / "latent_edge_pilot"
        pilot = run_logged(
            [
                "torchrun",
                "--standalone",
                "--nproc_per_node=2",
                "scripts/train_evaluate_latent_edge_embedding.py",
                "--output-dir",
                str(pilot_output),
                "--train-offset",
                "4096",
                "--train-sources",
                "256",
                "--epochs",
                "2",
                "--selection-sources",
                "16",
                "--holdout-sources",
                "16",
                "--selection-split",
                "assembly_incremental_gate",
                "--selection-offset",
                "192",
                "--holdout-offset",
                "208",
                *common,
            ],
            env=env,
            label="bounded latent-edge Stage-1 pilot",
            log_path=WORKING / "latent_edge_pilot.log",
            timeout_seconds=18000,
        )
        wrapper["steps"].append(pilot)
        atomic_json(WRAPPER, wrapper)
        require_success(pilot)
        pilot["report"] = validate_report(pilot_output / "latent_edge_embedding_report.json")
        wrapper.update(
            {
                "status": "complete",
                "result": pilot["report"],
                "seconds": time.time() - wrapper["started_unix"],
            }
        )
        atomic_json(WRAPPER, wrapper)
    except Exception as error:
        wrapper.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "seconds": time.time() - wrapper["started_unix"],
            }
        )
        atomic_json(WRAPPER, wrapper)
        raise


if __name__ == "__main__":
    main()

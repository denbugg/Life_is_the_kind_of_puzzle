#!/usr/bin/env python3
"""Fail-closed staging, T4x2 smoke, and dense all-pairs residual pilot."""

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


INPUT = Path("/kaggle/input/datasets/pasha883")
WORKING = Path("/kaggle/working")
PUZZLE_INPUT = INPUT / "vsos-ai-initiative-pazzle"
RUNTIME_INPUT = INPUT / "vsos-assembly-v1-runtime"
BASE_INPUT = INPUT / "vsos-solver-rework-night-code"
CODE_INPUT = INPUT / "vsos-dense-pair-residual-code"
WRAPPER = WORKING / "dense_pair_residual_pilot_wrapper.json"
EXPECTED_CODE_ARCHIVE_SHA256 = "3bb34eced0f289f77ed547b4dd41bc3c98e8ead8bfc8c310a87ec6889ef1a1c5"
EXPECTED_BASE_ARCHIVE_SHA256 = "a980c158fb349fbc8619e39eb829acdc675e7332d1ec3995c08f38eb49f45d0c"
EXPECTED_OVERLAY_HASHES = {
    "scripts/train_evaluate_dense_pair_residual.py": "3141acc7ac4d58a29e76bfa558f070504f0d352cf5fd480243362b6807d00d7d",
    "src/puzzle_assembly/dense_pair_residual.py": "38ef49942500a4d044c6faeb4807e207f5ee28ce7c73af0a82c3302d06fecbfc",
    "src/puzzle_assembly/protocol.py": "7651d4405ce4dd35203a0cae7bfdd591621044f9e90dc522a314262727c86eca",
    "configs/assembly_audit_exclusion_v1.json": "772e89ad4f633d2050f8ad3806cd24bffed132bcd8914951b7b8edff3f608ab6",
    "tests/test_dense_pair_residual.py": "7b4729c54652c47e0fd1955fd0796814901c304941293d848999c645fdf0e20d",
    "tests/test_dense_pair_trainer.py": "50e47e1c0c5c684535658f7b93f3300d01676689a553129dbf92538a6d211078",
}
EXPECTED_BASE_HASHES = {
    "src/puzzle_assembly/__init__.py": "09e051b7555471aafca03cd666d789f033aca47f1c82f6e2af9c0cce50afe9d5",
    "src/puzzle_assembly/compatibility.py": "aff2149b161c4fded4e5d91fbea49a8a62967886148d3ad374467331e0416a9f",
    "src/puzzle_assembly/components.py": "53fcc7c4fd23956db884ee45060e47f8e94a931c16e497e426d67549621bd367",
    "src/puzzle_assembly/geometry.py": "1e16bec6fb98a33060558d5d28062334d9114b12424733ef103a40393ef1ba86",
    "src/puzzle_assembly/learned.py": "9e3dba673aa85eaab5698dbeb63b3d94f88e3ea92b5e5979bde4b0273642697b",
    "src/puzzle_assembly/metrics.py": "84857ef92c382cc0964c21bfec67c13308014a1674aebf8686b17514784dae69",
    "src/puzzle_assembly/panels.py": "783356628517e3a23b8703672bca604c3d879c875f5b5f35f87182425500280f",
    "src/puzzle_assembly/protocol.py": "b711ad6d28a2fe60329e3e8236e58adbfbceea8ca4c8bf85e9a057e7619e24f4",
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
    "configs/denoise_splits_seed20260710.json": "de4fd2e596efa0d157d2d4480eed5fb84812d358138a1db53c1706bfb580e345",
    "configs/denoise_validation_quarantine_v1.json": "38dfd12f60579d77999c0cdb4a648fb4ff0343a8fb5e4c421f0b29f8b7bd6215",
}
EXPECTED_RUNTIME_HASHES = {
    "selected_tilenaf_synth_50k.pt": "77a2d8607c9bc6b80e7aa99ed03329c1fae6bf94ee1d9a654241ab93a05cc734",
    "hbt_d320_denoised_rgb_sobel.pt": "c2589bff58573d592227d522954a7ed6c22f6fec7ecbbea2bf0c755d7a720787",
}
ALLOWED_STATUS = {
    "stop_cheap_selection_retrieval",
    "stop_synthetic_transfer_retrieval",
    "stop_synthetic_transfer_qap",
    "stop_original_real_input_gate",
    "continue_candidate_only",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def require_mount(path: Path, label: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"missing exact Kaggle {label} mount: {path}")
    return path


def exactly_one(paths: list[Path], label: str) -> Path:
    values = sorted(set(path.resolve() for path in paths))
    if len(values) != 1:
        raise RuntimeError(f"expected exactly one {label}, found {values}")
    return values[0]


def safe_extract(archive: Path, destination: Path) -> Path:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    with zipfile.ZipFile(archive) as handle:
        members = handle.infolist()
        for member in members:
            name = member.filename
            mode = (member.external_attr >> 16) & 0o170000
            if name.startswith("/") or ".." in Path(name).parts or mode == 0o120000:
                raise RuntimeError(f"unsafe archive member: {name}")
        handle.extractall(destination)
    return destination


def exact_hashes(root: Path, expected: dict[str, str], label: str) -> dict[str, str]:
    actual: dict[str, str] = {}
    for relative, expected_hash in expected.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"{label} missing {relative}")
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"{label} hash mismatch for {relative}: {actual_hash} != {expected_hash}"
            )
        actual[relative] = actual_hash
    return actual


def require_exact_overlay_tree(root: Path) -> None:
    files = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"overlay contains symlink: {path}")
        if path.is_file():
            files.add(path.relative_to(root).as_posix())
    expected = set(EXPECTED_OVERLAY_HASHES)
    if files != expected:
        raise RuntimeError(
            f"overlay allowlist mismatch: missing={sorted(expected - files)}, "
            f"extra={sorted(files - expected)}"
        )


def find_base_root() -> tuple[Path, dict[str, object]]:
    mount = require_mount(BASE_INPUT, "solver base")
    direct = sorted(
        path.parents[2]
        for path in mount.glob("**/src/puzzle_assembly/qap.py")
        if path.is_file()
    )
    source_root: Path
    mode: dict[str, object]
    if len(set(direct)) == 1:
        source_root = direct[0]
        mode = {"mode": "direct", "source_root": str(source_root)}
    if direct:
        if len(set(direct)) != 1:
            raise RuntimeError(f"ambiguous direct solver roots: {direct}")
    else:
        archive = exactly_one(list(mount.glob("**/solver_rework_code.zip")), "solver archive")
        if sha256(archive) != EXPECTED_BASE_ARCHIVE_SHA256:
            raise RuntimeError("solver base archive SHA256 mismatch")
        extracted = safe_extract(archive, WORKING / "dense_pair_base_extracted")
        roots = {
            path.parents[2]
            for path in extracted.glob("**/src/puzzle_assembly/qap.py")
            if path.is_file()
        }
        if len(roots) != 1:
            raise RuntimeError(f"ambiguous archived solver roots: {sorted(roots)}")
        source_root = roots.pop()
        mode = {"mode": "archive", "archive": str(archive), "sha256": sha256(archive)}

    # Kaggle input mounts are immutable.  Always construct a small writable
    # working copy before applying the experiment overlay.
    root = WORKING / "dense_pair_base"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    for directory in ("src", "configs"):
        source = source_root / directory
        if not source.is_dir():
            raise FileNotFoundError(f"solver base lacks {directory}/")
        shutil.copytree(source, root / directory)
    mode["working_root"] = str(root)
    return root, mode


def overlay_code(base_root: Path) -> dict[str, object]:
    mount = require_mount(CODE_INPUT, "dense-pair code")
    direct_scripts = list(mount.glob("**/scripts/train_evaluate_dense_pair_residual.py"))
    if direct_scripts:
        script = exactly_one(direct_scripts, "direct dense-pair trainer")
        overlay_root = script.parents[1]
        require_exact_overlay_tree(overlay_root)
        exact_hashes(overlay_root, EXPECTED_OVERLAY_HASHES, "direct overlay")
        mode: dict[str, object] = {"mode": "direct", "root": str(overlay_root)}
    else:
        archive = exactly_one(list(mount.glob("**/dense_pair_residual_code.zip")), "code archive")
        archive_hash = sha256(archive)
        if archive_hash != EXPECTED_CODE_ARCHIVE_SHA256:
            raise RuntimeError("dense-pair code archive SHA256 mismatch")
        overlay_root = safe_extract(archive, WORKING / "dense_pair_overlay")
        require_exact_overlay_tree(overlay_root)
        exact_hashes(overlay_root, EXPECTED_OVERLAY_HASHES, "archive overlay")
        mode = {"mode": "archive", "archive": str(archive), "sha256": archive_hash}
    for relative in EXPECTED_OVERLAY_HASHES:
        source = overlay_root / relative
        destination = base_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    mode["staged_hashes"] = exact_hashes(base_root, EXPECTED_OVERLAY_HASHES, "staged overlay")
    return mode


def find_data_root() -> Path:
    mount = require_mount(PUZZLE_INPUT, "puzzle data")
    candidates = []
    for inputs in mount.glob("**/train/inputs"):
        targets = inputs.parent / "targets"
        if inputs.is_dir() and targets.is_dir():
            if len(list(targets.glob("*.png"))) == 7000 and len(list(inputs.glob("*.png"))) == 7000:
                candidates.append(inputs.parent.parent)
    return exactly_one(candidates, "7000-source puzzle root")


def find_runtime_asset(filename: str) -> Path:
    mount = require_mount(RUNTIME_INPUT, "assembly runtime")
    path = exactly_one([value for value in mount.glob(f"**/{filename}") if value.is_file()], filename)
    actual = sha256(path)
    if actual != EXPECTED_RUNTIME_HASHES[filename]:
        raise RuntimeError(f"runtime asset hash mismatch for {filename}")
    return path


def gpu_preflight() -> dict[str, object]:
    # Probe CUDA in a child that exits before torchrun, so the wrapper process
    # cannot retain primary contexts and steal memory from the serious model.
    code = r'''import json, torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
    raise RuntimeError("pilot requires exactly two visible CUDA devices")
devices = []
for index in range(2):
    name = torch.cuda.get_device_name(index)
    if "T4" not in name.upper():
        raise RuntimeError(f"device {index} is not a Tesla T4: {name}")
    device = torch.device("cuda", index)
    value = torch.arange(4096, device=device, dtype=torch.float32)
    devices.append({
        "index": index,
        "name": name,
        "capability": list(torch.cuda.get_device_capability(index)),
        "total_memory": int(torch.cuda.get_device_properties(index).total_memory),
        "tensor_probe": float((value.sin().square().mean()).item()),
    })
print(json.dumps({
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "device_count": 2,
    "devices": devices,
}))'''
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    result = json.loads(completed.stdout)
    result["nvidia_smi"] = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    ).stdout.strip().splitlines()
    return result


def run_logged(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    label: str,
    timeout_seconds: int,
) -> dict[str, object]:
    started = time.perf_counter()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        try:
            completed = subprocess.run(
                command,
                cwd=cwd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                timeout=timeout_seconds,
            )
            returncode: int | None = completed.returncode
            timed_out = False
        except subprocess.TimeoutExpired:
            returncode = None
            timed_out = True
    record = {
        "label": label,
        "command": command,
        "returncode": returncode,
        "timed_out": timed_out,
        "timeout_seconds": timeout_seconds,
        "seconds": time.perf_counter() - started,
        "log": str(log_path),
        "log_sha256": sha256(log_path),
    }
    return record


def require_success(record: dict[str, object]) -> None:
    if record.get("returncode") == 0 and record.get("timed_out") is False:
        return
    log_path = Path(str(record["log"]))
    tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
    raise RuntimeError(
        f"{record['label']} failed with returncode={record.get('returncode')} "
        f"timed_out={record.get('timed_out')}:\n" + "\n".join(tail)
    )


def validate_report(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("schema_version") != 1 or report.get("kind") != "dense_all_pairs_residual_pilot_report":
        raise RuntimeError("unexpected dense-pair report schema")
    if report.get("safe_for_submission") is not False:
        raise RuntimeError("report is not fail-closed")
    status = report.get("status")
    if status not in ALLOWED_STATUS:
        raise RuntimeError(f"unexpected report status: {status}")
    provenance = report.get("provenance", {})
    if provenance.get("safe_for_submission") is not False:
        raise RuntimeError("provenance is not fail-closed")
    if "575 valid alternatives" not in provenance.get("all_negatives_contract", ""):
        raise RuntimeError("report does not attest the all-575-negative contract")
    selection = report.get("selection")
    if not isinstance(selection, dict) or not isinstance(selection.get("retrieval_gate"), dict):
        raise RuntimeError("report lacks selection retrieval gate")
    opened = report.get("gate_opened")
    if not isinstance(opened, dict):
        raise RuntimeError("report lacks gate-open map")
    if opened.get("true_final_audit") is not False or opened.get("true_confirmation") is not False:
        raise RuntimeError("pilot illegally opened the sealed audit")
    if status == "stop_cheap_selection_retrieval" and opened.get("synthetic_transfer"):
        raise RuntimeError("synthetic holdout opened before cheap selection passed")
    if status in {"stop_synthetic_transfer_retrieval", "stop_synthetic_transfer_qap"}:
        if opened.get("synthetic_transfer") is not True or opened.get("original_real_input"):
            raise RuntimeError("synthetic-transfer status disagrees with gate-open map")
    if status in {"stop_original_real_input_gate", "continue_candidate_only"}:
        if opened.get("original_real_input") is not True:
            raise RuntimeError("real-gate status lacks frozen original-input evidence")
        real_gate = report.get("real_gate")
        if not isinstance(real_gate, dict) or real_gate.get(
            "target_opened_after_predictions_frozen"
        ) is not True:
            raise RuntimeError("real gate did not prove input-only freeze before target attach")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "status": status,
        "gate_opened": opened,
    }


def main() -> None:
    wrapper: dict[str, object] = {
        "schema_version": 1,
        "kind": "dense_pair_residual_kaggle_wrapper",
        "status": "starting",
        "safe_for_submission": False,
        "started_unix": time.time(),
        "steps": [],
    }
    atomic_json(WRAPPER, wrapper)
    try:
        base_root, base_mode = find_base_root()
        base_hashes = exact_hashes(base_root, EXPECTED_BASE_HASHES, "solver base")
        overlay_mode = overlay_code(base_root)
        data_root = find_data_root()
        denoiser = find_runtime_asset("selected_tilenaf_synth_50k.pt")
        hbt = find_runtime_asset("hbt_d320_denoised_rgb_sobel.pt")
        manifest = base_root / "configs/denoise_splits_seed20260710.json"
        quarantine = base_root / "configs/denoise_validation_quarantine_v1.json"
        audit_exclusion = base_root / "configs/assembly_audit_exclusion_v1.json"
        hardware = gpu_preflight()
        wrapper.update(
            {
                "status": "staged",
                "base": base_mode,
                "base_hashes": base_hashes,
                "overlay": overlay_mode,
                "data_root": str(data_root),
                "assets": {
                    "denoiser": {"path": str(denoiser), "sha256": sha256(denoiser)},
                    "hbt": {"path": str(hbt), "sha256": sha256(hbt)},
                    "manifest": {"path": str(manifest), "sha256": sha256(manifest)},
                    "quarantine": {"path": str(quarantine), "sha256": sha256(quarantine)},
                    "audit_exclusion": {
                        "path": str(audit_exclusion),
                        "sha256": sha256(audit_exclusion),
                    },
                },
                "hardware": hardware,
            }
        )
        atomic_json(WRAPPER, wrapper)

        env = dict(os.environ)
        env["PYTHONPATH"] = str(base_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
        tests = run_logged(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/test_dense_pair_residual.py",
                "tests/test_dense_pair_trainer.py",
            ],
            cwd=base_root,
            env=env,
            log_path=WORKING / "dense_pair_residual_tests.log",
            label="dense-pair unit tests",
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
            str(manifest),
            "--quarantine",
            str(quarantine),
            "--audit-exclusion",
            str(audit_exclusion),
            "--selection-offset",
            "96",
            "--holdout-offset",
            "112",
            "--real-gate-offset",
            "128",
            "--final-audit-offset",
            "0",
            "--confirmation-offset",
            "64",
            "--panels",
            "primary_kornia,independent_libjpeg",
        ]
        smoke_output = WORKING / "dense_pair_residual_smoke"
        if smoke_output.exists():
            shutil.rmtree(smoke_output)
        smoke = run_logged(
            [
                "torchrun",
                "--standalone",
                "--nproc_per_node=2",
                "scripts/train_evaluate_dense_pair_residual.py",
                "--action",
                "pilot",
                "--output-dir",
                str(smoke_output),
                "--smoke",
                "--full-model-smoke",
                *common,
            ],
            cwd=base_root,
            env=env,
            log_path=WORKING / "dense_pair_residual_smoke.log",
            label="2xT4 full-model one-step smoke",
            timeout_seconds=3600,
        )
        wrapper["steps"].append(smoke)
        atomic_json(WRAPPER, wrapper)
        require_success(smoke)
        smoke["report"] = validate_report(smoke_output / "dense_pair_residual_report.json")
        atomic_json(WRAPPER, wrapper)

        pilot_output = WORKING / "dense_pair_residual_pilot"
        if pilot_output.exists():
            shutil.rmtree(pilot_output)
        pilot = run_logged(
            [
                "torchrun",
                "--standalone",
                "--nproc_per_node=2",
                "scripts/train_evaluate_dense_pair_residual.py",
                "--action",
                "pilot",
                "--output-dir",
                str(pilot_output),
                "--train-sources",
                "256",
                "--train-offset",
                "4096",
                "--epochs",
                "2",
                "--queries-per-source",
                "48",
                "--selection-sources",
                "32",
                "--holdout-sources",
                "16",
                "--real-gate-sources",
                "64",
                "--final-audit-sources",
                "64",
                "--confirmation-sources",
                "64",
                "--quick-sources",
                "32",
                "--evaluation-replicas",
                "1",
                *common,
            ],
            cwd=base_root,
            env=env,
            log_path=WORKING / "dense_pair_residual_pilot.log",
            label="bounded dense-pair residual pilot",
            timeout_seconds=10800,
        )
        wrapper["steps"].append(pilot)
        atomic_json(WRAPPER, wrapper)
        require_success(pilot)
        pilot["report"] = validate_report(pilot_output / "dense_pair_residual_report.json")
        wrapper.update(
            {
                "status": "complete",
                "completed_unix": time.time(),
                "pilot_report": pilot["report"],
                "safe_for_submission": False,
            }
        )
        atomic_json(WRAPPER, wrapper)
        print(json.dumps(wrapper, indent=2, sort_keys=True, default=str))
    except BaseException as error:
        wrapper.update(
            {
                "status": "failed",
                "failed_unix": time.time(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "safe_for_submission": False,
            }
        )
        atomic_json(WRAPPER, wrapper)
        raise


if __name__ == "__main__":
    main()

"""Bounded weights-only continuation of the frozen best HBT side encoder.

This job deliberately stops after training and a same-selection diagnostic.
The untouched primary/libjpeg comparison on edge_development[96:128] is a
separate gate; this checkpoint is never submission-safe by construction.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile


INPUT = Path("/kaggle/input/datasets/pasha883")
WORKING = Path("/kaggle/working")
RUNTIME = INPUT / "vsos-assembly-v1-runtime"
DATA = INPUT / "vsos-ai-initiative-pazzle"
OUTPUT = WORKING / "hbt_continuation"
WRAPPER = WORKING / "hbt_continuation_wrapper.json"

EXPECTED = {
    "baseline": "c2589bff58573d592227d522954a7ed6c22f6fec7ecbbea2bf0c755d7a720787",
    "denoiser": "77a2d8607c9bc6b80e7aa99ed03329c1fae6bf94ee1d9a654241ab93a05cc734",
    "code_zip": "726a512fca9df5003e37575181cd877abfa0f47eada478e90f9a7fc481887cf2",
    "trainer": "20cde60a1f67e5f61d7c043f54ee72452c708551831bda88a09f6bd038565081",
    "learned": "a415aae32b3f38aae1f4fe36d91343ead3099d448b5490c4f6eeecf6ea6337d7",
    "manifest": "de4fd2e596efa0d157d2d4480eed5fb84812d358138a1db53c1706bfb580e345",
    "quarantine": "38dfd12f60579d77999c0cdb4a648fb4ff0343a8fb5e4c421f0b29f8b7bd6215",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def names_sha256(names: list[str]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def exact_one(paths: list[Path], label: str) -> Path:
    resolved = sorted({path.resolve() for path in paths if path.is_file()})
    if len(resolved) != 1:
        raise RuntimeError(f"expected exactly one {label}, found {resolved}")
    return resolved[0]


def exact_directory(paths: list[Path], label: str) -> Path:
    resolved = sorted({path.resolve() for path in paths if path.is_dir()})
    if len(resolved) != 1:
        raise RuntimeError(f"expected exactly one {label}, found {resolved}")
    return resolved[0]


def safe_extract(archive: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=False)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        members = handle.infolist()
        if not members:
            raise RuntimeError("code archive is empty")
        for member in members:
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"unsafe archive member: {member.filename}")
            mode = (member.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                raise RuntimeError(f"symlink archive member rejected: {member.filename}")
        handle.extractall(destination)
    return destination


def resolve_code_root() -> tuple[Path, dict[str, str]]:
    direct = sorted(RUNTIME.glob("**/src/puzzle_assembly/__init__.py"))
    archives = sorted(RUNTIME.glob("**/assembly_v1_code.zip"))
    if len(direct) > 1 or len(archives) > 1:
        raise RuntimeError(f"ambiguous code payload: direct={direct}, zip={archives}")
    if len(direct) == 1:
        root = direct[0].parents[2]
        provenance = {"mode": "direct_tree"}
        if archives:
            if len(archives) != 1 or sha256(archives[0]) != EXPECTED["code_zip"]:
                raise RuntimeError(f"unexpected archive alongside direct code: {archives}")
            provenance["archive_sha256"] = EXPECTED["code_zip"]
    elif len(archives) == 1:
        archive = archives[0]
        actual = sha256(archive)
        if actual != EXPECTED["code_zip"]:
            raise RuntimeError(f"code archive hash mismatch: {actual}")
        root = safe_extract(archive, WORKING / "assembly_v1_code")
        provenance = {"mode": "verified_zip", "archive_sha256": actual}
    else:
        raise RuntimeError(f"unable to resolve unique code payload: direct={direct}, zip={archives}")
    required = {
        "trainer": root / "scripts/train_side_embeddings.py",
        "learned": root / "src/puzzle_assembly/learned.py",
        "manifest": root / "configs/denoise_splits_seed20260710.json",
        "quarantine": root / "configs/denoise_validation_quarantine_v1.json",
    }
    for label, path in required.items():
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"missing or symlinked required code file: {path}")
        actual = sha256(path)
        if actual != EXPECTED[label]:
            raise RuntimeError(f"{label} hash mismatch: {actual}")
        provenance[f"{label}_sha256"] = actual
    return root, provenance


def hardware_probe() -> dict[str, object]:
    import numpy
    import PIL
    import scipy
    import skimage
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA GPU is required")
    devices = []
    for index in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(index)
        capability = list(torch.cuda.get_device_capability(index))
        left = torch.randn((256, 256), device=f"cuda:{index}")
        right = torch.randn((256, 256), device=f"cuda:{index}")
        mean = float((left @ right).mean().cpu())
        if not "T4" in name.upper():
            raise RuntimeError(f"expected a T4 allocation, got {name!r}")
        devices.append({"index": index, "name": name, "capability": capability, "matmul_mean": mean})
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "numpy": numpy.__version__,
        "pillow": PIL.__version__,
        "scipy": scipy.__version__,
        "skimage": skimage.__version__,
        "device_count": torch.cuda.device_count(),
        "devices": devices,
    }


def stream_command(command: list[str], *, environment: dict[str, str], log: Path) -> None:
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=environment,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()
        return_code = process.wait()
    if return_code:
        raise RuntimeError(f"training failed with exit code {return_code}")


def main() -> None:
    started = time.time()
    if not RUNTIME.is_dir() or not DATA.is_dir():
        raise RuntimeError(f"required Kaggle mounts are absent: runtime={RUNTIME}, data={DATA}")
    if OUTPUT.exists():
        raise RuntimeError(f"refusing to reuse non-empty output path: {OUTPUT}")
    OUTPUT.mkdir(parents=True)

    code_root, code_provenance = resolve_code_root()
    baseline = exact_one(list(RUNTIME.glob("**/hbt_d320_denoised_rgb_sobel.pt")), "baseline HBT")
    denoiser = exact_one(list(RUNTIME.glob("**/selected_tilenaf_synth_50k.pt")), "TileNAF")
    targets = exact_directory(list(DATA.glob("**/train/targets")), "train targets directory")
    data_root = targets.parent.parent
    for label, path in (("baseline", baseline), ("denoiser", denoiser)):
        actual = sha256(path)
        if actual != EXPECTED[label]:
            raise RuntimeError(f"{label} hash mismatch: {actual}")

    probe = hardware_probe()
    sys.path.insert(0, str(code_root / "src"))
    import torch
    from puzzle_assembly.protocol import source_names_for_split

    manifest = code_root / "configs/denoise_splits_seed20260710.json"
    quarantine = code_root / "configs/denoise_validation_quarantine_v1.json"
    edge_train = source_names_for_split(
        "edge_train", manifest_path=manifest, quarantine_path=quarantine
    )
    edge_development = source_names_for_split(
        "edge_development", manifest_path=manifest, quarantine_path=quarantine
    )
    payload = torch.load(baseline, map_location="cpu", weights_only=False)
    baseline_train = list(payload.get("metadata", {}).get("train_names", []))
    expected_baseline_train = edge_train[:2048]
    continuation_train = edge_train[2048:4096]
    selection_names = edge_development[:32]
    untouched_gate_names = edge_development[96:128]
    if baseline_train != expected_baseline_train:
        raise RuntimeError("baseline training-name provenance is not the expected edge_train prefix")
    partitions = [set(baseline_train), set(continuation_train), set(selection_names), set(untouched_gate_names)]
    for left in range(len(partitions)):
        for right in range(left + 1, len(partitions)):
            if partitions[left] & partitions[right]:
                raise RuntimeError(f"source partition overlap at {left},{right}")

    checkpoint = OUTPUT / "hbt_d320_denoised_rgb_sobel_cont.pt"
    report = checkpoint.with_suffix(".json")
    log = OUTPUT / "training.log"
    command = [
        sys.executable,
        str(code_root / "scripts/train_side_embeddings.py"),
        "--data-root", str(data_root),
        "--denoiser", str(denoiser),
        "--init-checkpoint", str(baseline),
        "--manifest", str(manifest),
        "--quarantine", str(quarantine),
        "--panel", "primary_kornia",
        "--view", "denoised",
        "--train-offset", "2048",
        "--train-sources", "2048",
        "--val-offset", "0",
        "--val-sources", "32",
        "--epochs", "2",
        "--replica-offset", "2",
        "--seed", "20260710",
        "--loss", "hard_triplet",
        "--triplet-margin", "0.2",
        "--cross-entropy-weight", "0.25",
        "--embedding-l2-weight", "0.0001",
        "--outside-weight", "0.2",
        "--learning-rate", "0.0001",
        "--weight-decay", "0.0001",
        "--grad-clip", "1.0",
        "--device", "cuda",
        "--output", str(checkpoint),
    ]
    if int(probe["device_count"]) > 1:
        command.append("--data-parallel")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(code_root / "src")
    print(json.dumps({"event": "hbt_continuation_start", "command": command, "probe": probe}, sort_keys=True), flush=True)
    stream_command(command, environment=environment, log=log)

    if not checkpoint.is_file() or not report.is_file():
        raise RuntimeError("trainer returned without checkpoint/report")
    candidate = torch.load(checkpoint, map_location="cpu", weights_only=False)
    candidate_metadata = dict(candidate.get("metadata", {}))
    if list(candidate_metadata.get("train_names", [])) != continuation_train:
        raise RuntimeError("candidate checkpoint training-name provenance mismatch")
    if list(candidate_metadata.get("val_names", [])) != selection_names:
        raise RuntimeError("candidate checkpoint selection-name provenance mismatch")
    if candidate_metadata.get("init_checkpoint_sha256") != EXPECTED["baseline"]:
        raise RuntimeError("candidate init-checkpoint hash provenance mismatch")

    training_report = json.loads(report.read_text(encoding="utf-8"))
    result = {
        "schema_version": 1,
        "kind": "hbt_weights_only_continuation_t4x2",
        "status": "training_complete_untouched_gate_pending",
        "safe_for_submission": False,
        "probe": probe,
        "code_provenance": code_provenance,
        "baseline": {
            "path": str(baseline),
            "sha256": EXPECTED["baseline"],
            "reported_selection_r1": payload.get("metadata", {}).get("best_validation_recall_at_1"),
            "train_names_sha256": names_sha256(baseline_train),
        },
        "candidate": {
            "path": str(checkpoint),
            "sha256": sha256(checkpoint),
            "report": str(report),
            "report_sha256": sha256(report),
            "best_epoch": training_report.get("best_epoch"),
            "reported_selection_r1": training_report.get("best_validation_recall_at_1"),
            "train_names_sha256": names_sha256(continuation_train),
            "selection_names_sha256": names_sha256(selection_names),
        },
        "training_contract": {
            "kind": "weights_only_finetune_optimizer_reset",
            "train_slice": [2048, 4096],
            "replicas": [2, 3],
            "epochs": 2,
            "learning_rate": 0.0001,
            "loss": "hard_triplet_all_575_negatives",
        },
        "next_gate": {
            "split": "edge_development",
            "slice": [96, 128],
            "names_sha256": names_sha256(untouched_gate_names),
            "engines": ["primary_kornia", "independent_libjpeg"],
            "compare_epoch0_baseline": True,
            "note": "dev[0:64] was touched by historical selection and dev[64:72] by exact8 gates; dev[96:128] is the frozen clean comparator",
        },
        "elapsed_seconds": time.time() - started,
    }
    WRAPPER.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    hashes = OUTPUT / "SHA256SUMS.txt"
    artifacts = [checkpoint, report, log, WRAPPER]
    hashes.write_text(
        "".join(f"{sha256(path)}  {path.relative_to(WORKING)}\n" for path in artifacts),
        encoding="utf-8",
    )
    print(json.dumps({"event": "hbt_continuation_complete", "wrapper": str(WRAPPER), "wrapper_sha256": sha256(WRAPPER), "hashes": str(hashes), "candidate_sha256": sha256(checkpoint)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

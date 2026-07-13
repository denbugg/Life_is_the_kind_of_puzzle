#!/usr/bin/env python3
"""Run the frozen-layout harmonized submission renderer on two Kaggle T4 GPUs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping
import zipfile

from PIL import Image
import torch


INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
EXPECTED_TOTAL = 700
SHARD_SIZE = 350
ARCHIVE_TIMESTAMP = (2026, 7, 12, 0, 0, 0)
EXPECTED_BUNDLE_MANIFEST_SHA256 = (
    "fadfd1df59a7a28d417f9b3de33784f5a5fe76f362197f8bdc4e4bbe1eeda3cc"
)
EXPECTED_BUILDER_SHA256 = (
    "5777a774b8be7d718b469b32cd7a678c15a7075e84057bd79285e9a53e6cd6f9"
)
EXPECTED_LAYOUT_REPORT_SHA256 = {
    "541e7905dad9373a173c31db068429b20fb614d450f3cfb4439b89a0a45b2e2a",
    "38e8a01de560ef8914f25dbe42b2c43c063bde07e3b537e71c0be28416f8dbc4",
}
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RuntimeError(f"expected regular unlinked file: {path}")
        for chunk in iter(lambda: os.read(descriptor, 1024 * 1024), b""):
            digest.update(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def single(paths: list[Path], label: str) -> Path:
    unique = sorted({path.resolve() for path in paths})
    if len(unique) != 1:
        raise RuntimeError(f"expected exactly one {label}, found {unique}")
    return unique[0]


def find_test_dir() -> Path:
    candidates = []
    for path in INPUT.glob("**/test"):
        if path.is_dir() and len(list(path.glob("*.png"))) == EXPECTED_TOTAL:
            candidates.append(path)
    return single(candidates, "700-image puzzle test directory")


def find_bundle_root() -> Path:
    return single(
        [path.parent for path in INPUT.glob("**/BUNDLE_MANIFEST.json")],
        "harmonized runtime bundle",
    )


def validate_bundle(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = root / "BUNDLE_MANIFEST.json"
    if sha256(manifest_path) != EXPECTED_BUNDLE_MANIFEST_SHA256:
        raise RuntimeError("runtime bundle manifest hash mismatch")
    manifest = load_json(manifest_path)
    if set(manifest) != {
        "schema_version",
        "kind",
        "safe_for_submission",
        "member_count",
        "members",
        "assets",
    }:
        raise RuntimeError("runtime bundle manifest schema drift")
    if (
        manifest["schema_version"] != 1
        or manifest["kind"] != "harmonized_submission_runtime_bundle_manifest"
        or manifest["safe_for_submission"] is not False
    ):
        raise RuntimeError("runtime bundle manifest identity drift")
    members = manifest.get("members")
    if not isinstance(members, list) or len(members) != manifest.get("member_count"):
        raise RuntimeError("runtime bundle member count drift")
    expected_paths: set[str] = {"BUNDLE_MANIFEST.json"}
    validated: list[dict[str, Any]] = []
    for record in members:
        if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}:
            raise RuntimeError("invalid runtime bundle member record")
        relative = record["path"]
        if (
            not isinstance(relative, str)
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or relative in expected_paths
        ):
            raise RuntimeError("invalid or duplicate runtime bundle member path")
        if not isinstance(record["bytes"], int) or record["bytes"] <= 0:
            raise RuntimeError("invalid runtime bundle member size")
        if not isinstance(record["sha256"], str) or not SHA_RE.fullmatch(record["sha256"]):
            raise RuntimeError("invalid runtime bundle member hash")
        path = root.joinpath(*Path(relative).parts)
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"runtime bundle member missing: {relative}")
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise RuntimeError(f"runtime bundle member drift: {relative}")
        expected_paths.add(relative)
        validated.append(record)
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual_paths != expected_paths:
        raise RuntimeError(
            f"runtime bundle file tree drift: missing={sorted(expected_paths-actual_paths)} "
            f"extra={sorted(actual_paths-expected_paths)}"
        )
    by_path = {record["path"]: record for record in validated}
    if by_path["scripts/build_harmonized_submission.py"]["sha256"] != EXPECTED_BUILDER_SHA256:
        raise RuntimeError("harmonized builder hash mismatch")
    layout_hashes = {
        by_path["layouts/final_qap_shard_000_350.json"]["sha256"],
        by_path["layouts/final_qap_shard_350_700.json"]["sha256"],
    }
    if layout_hashes != EXPECTED_LAYOUT_REPORT_SHA256:
        raise RuntimeError("frozen layout report hashes drifted")
    return manifest, validated


def gpu_fingerprint() -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("exactly two CUDA GPUs are required")
    devices = []
    for index in range(2):
        name = torch.cuda.get_device_name(index)
        capability = list(torch.cuda.get_device_capability(index))
        if "T4" not in name or capability != [7, 5]:
            raise RuntimeError(f"GPU {index} is not a Tesla T4 sm_75: {name} {capability}")
        left = torch.arange(256, device=f"cuda:{index}", dtype=torch.float32).reshape(16, 16)
        checksum = float((left @ left.T).sum().item())
        if not checksum > 0:
            raise RuntimeError("CUDA tensor smoke failed")
        devices.append(
            {
                "index": index,
                "name": name,
                "capability": capability,
                "tensor_checksum": checksum,
            }
        )
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device_count": 2,
        "devices": devices,
    }


def run_builder(
    *,
    bundle_root: Path,
    test_dir: Path,
    gpu: int,
    offset: int,
    limit: int,
    label: str,
) -> dict[str, Any]:
    output = WORKING / f"{label}.zip"
    report = WORKING / f"{label}.json"
    log = WORKING / f"{label}.log"
    for path in (output, report, log):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"builder output must be fresh: {path}")
    command = [
        sys.executable,
        str(bundle_root / "scripts/build_harmonized_submission.py"),
        "--input-dir",
        str(test_dir),
        "--selected-denoiser",
        str(bundle_root / "assets/selected_tilenaf_synth_50k.pt"),
        "--seam-denoiser",
        str(bundle_root / "assets/seam_denoiser_gpu.pt"),
        "--layout-report",
        str(bundle_root / "layouts/final_qap_shard_000_350.json"),
        "--layout-report",
        str(bundle_root / "layouts/final_qap_shard_350_700.json"),
        "--output",
        str(output),
        "--report",
        str(report),
        "--offset",
        str(offset),
        "--limit",
        str(limit),
        "--expected-count",
        str(EXPECTED_TOTAL),
        "--device",
        "cuda",
        "--batch-size",
        "512",
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["PYTHONPATH"] = str(bundle_root / "src")
    started = time.perf_counter()
    with log.open("wb") as handle:
        completed = subprocess.run(
            command,
            env=environment,
            cwd=str(bundle_root),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"harmonized builder failed for {label} on GPU {gpu}: rc={completed.returncode}"
        )
    payload = load_json(report)
    if (
        payload.get("kind") != "harmonized_frozen_qap_submission_report"
        or payload.get("status") != "test_only_candidate_not_lb_scored"
        or payload.get("offset") != offset
        or payload.get("limit") != limit
        or payload.get("count") != limit
        or payload.get("anti_leakage", {}).get("target_paths_or_pixels_read") is not False
        or payload.get("anti_leakage", {}).get("layout_recomputed") is not False
        or payload.get("archive", {}).get("sha256") != sha256(output)
    ):
        raise RuntimeError(f"harmonized builder report contract failed: {label}")
    report_layout_hashes = {
        record.get("sha256")
        for record in payload.get("assets", {}).get("layout_reports", [])
    }
    if report_layout_hashes != EXPECTED_LAYOUT_REPORT_SHA256:
        raise RuntimeError("builder report layout provenance drift")
    return {
        "label": label,
        "gpu": gpu,
        "offset": offset,
        "limit": limit,
        "output": str(output),
        "output_sha256": sha256(output),
        "output_bytes": output.stat().st_size,
        "report": str(report),
        "report_sha256": sha256(report),
        "log": str(log),
        "log_sha256": sha256(log),
        "method_sha256": payload.get("method_sha256"),
        "source_names_sha256": payload.get("source_names_sha256"),
        "seconds": time.perf_counter() - started,
    }


def read_member(archive_path: Path, name: str) -> bytes:
    with zipfile.ZipFile(archive_path) as archive:
        info = archive.getinfo(name)
        if (
            Path(info.filename).name != info.filename
            or info.date_time != ARCHIVE_TIMESTAMP
            or info.create_system != 3
            or info.compress_type != zipfile.ZIP_DEFLATED
            or (info.external_attr >> 16) != 0o100644
        ):
            raise RuntimeError(f"invalid shard archive member metadata: {name}")
        payload = archive.read(name)
    with Image.open(__import__("io").BytesIO(payload)) as image:
        if image.mode != "RGB" or image.size != (480, 480):
            raise RuntimeError(f"invalid output PNG: {name}")
        image.load()
    return payload


def merge_shards(
    *, test_names: list[str], first: dict[str, Any], second: dict[str, Any]
) -> tuple[Path, dict[str, Any]]:
    paths = [Path(first["output"]), Path(second["output"])]
    ownership: dict[str, Path] = {}
    for path in paths:
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if name in ownership:
                    raise RuntimeError(f"duplicate shard member: {name}")
                ownership[name] = path
    if set(ownership) != set(test_names) or len(ownership) != EXPECTED_TOTAL:
        raise RuntimeError("shard union does not exactly match the 700 test names")
    destination = WORKING / "submission.zip"
    if destination.exists() or destination.is_symlink():
        raise RuntimeError("final submission path must be fresh")
    members = []
    with zipfile.ZipFile(
        destination,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as output:
        for name in test_names:
            payload = read_member(ownership[name], name)
            info = zipfile.ZipInfo(name, date_time=ARCHIVE_TIMESTAMP)
            info.create_system = 3
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            output.writestr(info, payload, compresslevel=6)
            members.append(
                {
                    "name": name,
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
    manifest = {
        "schema_version": 1,
        "kind": "harmonized_submission_manifest",
        "status": "test_only_candidate_not_lb_scored",
        "archive": {
            "path": destination.name,
            "bytes": destination.stat().st_size,
            "sha256": sha256(destination),
        },
        "member_count": len(members),
        "members": members,
        "members_sha256": canonical_sha256({"members": members}),
    }
    manifest_path = WORKING / "HARMONIZED_SUBMISSION_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return destination, manifest


def main() -> None:
    started = time.perf_counter()
    try:
        bundle_root = find_bundle_root()
        test_dir = find_test_dir()
        manifest, bundle_members = validate_bundle(bundle_root)
        gpu = gpu_fingerprint()
        test_names = sorted(path.name for path in test_dir.glob("*.png"))
        if len(test_names) != EXPECTED_TOTAL or len(set(test_names)) != EXPECTED_TOTAL:
            raise RuntimeError("test filename contract failed")

        preflight = run_builder(
            bundle_root=bundle_root,
            test_dir=test_dir,
            gpu=0,
            offset=0,
            limit=1,
            label="harmonized_preflight_000_001",
        )
        jobs = [
            dict(gpu=0, offset=0, limit=SHARD_SIZE, label="harmonized_shard_000_350"),
            dict(gpu=1, offset=SHARD_SIZE, limit=SHARD_SIZE, label="harmonized_shard_350_700"),
        ]
        shards: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(
                    run_builder,
                    bundle_root=bundle_root,
                    test_dir=test_dir,
                    **job,
                ): job
                for job in jobs
            }
            for future in as_completed(futures):
                shards.append(future.result())
        shards.sort(key=lambda value: value["offset"])
        if len({record["method_sha256"] for record in [preflight, *shards]}) != 1:
            raise RuntimeError("method hash differs between preflight and shards")
        first_name = test_names[0]
        if read_member(Path(preflight["output"]), first_name) != read_member(
            Path(shards[0]["output"]), first_name
        ):
            raise RuntimeError("preflight image is not byte-identical in shard replay")

        submission, submission_manifest = merge_shards(
            test_names=test_names, first=shards[0], second=shards[1]
        )
        report = {
            "schema_version": 1,
            "kind": "harmonized_submission_kaggle_run",
            "status": "candidate_ready_not_lb_scored",
            "created_utc": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
            "anti_leakage": {
                "target_paths_or_pixels_read": False,
                "layout_recomputed": False,
                "only_test_pixels_and_frozen_layout_reports_used": True,
            },
            "environment": gpu,
            "bundle": {
                "root": str(bundle_root),
                "manifest_sha256": EXPECTED_BUNDLE_MANIFEST_SHA256,
                "manifest": manifest,
                "validated_member_count": len(bundle_members),
            },
            "test_dir": str(test_dir),
            "source_names_sha256": hashlib.sha256(
                "\n".join(test_names).encode("utf-8")
            ).hexdigest(),
            "preflight": preflight,
            "shards": shards,
            "preflight_replay_byte_identical": True,
            "submission": submission_manifest["archive"],
            "submission_manifest_sha256": sha256(
                WORKING / "HARMONIZED_SUBMISSION_MANIFEST.json"
            ),
            "safe_for_submission": True,
            "leaderboard_score": None,
            "seconds": time.perf_counter() - started,
        }
        report_path = WORKING / "HARMONIZED_SUBMISSION_RUN.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "event": "harmonized_submission_ready",
                    "submission": str(submission),
                    "sha256": sha256(submission),
                    "member_count": EXPECTED_TOTAL,
                    "seconds": report["seconds"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    except BaseException as error:
        invalid = {
            "schema_version": 1,
            "kind": "harmonized_submission_invalid_no_result",
            "status": "invalid_no_submission",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "safe_for_submission": False,
            "leaderboard_score": None,
        }
        (WORKING / "HARMONIZED_INVALID_NO_RESULT.json").write_text(
            json.dumps(invalid, indent=2, sort_keys=True), encoding="utf-8"
        )
        raise


if __name__ == "__main__":
    main()

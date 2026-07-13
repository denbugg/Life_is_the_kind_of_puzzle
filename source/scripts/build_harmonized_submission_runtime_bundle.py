#!/usr/bin/env python3
"""Build the deterministic private Kaggle runtime bundle for harmonized inference."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping
import zipfile


ARCHIVE_TIMESTAMP = (2026, 7, 12, 0, 0, 0)
SELECTED_SHA256 = "77a2d8607c9bc6b80e7aa99ed03329c1fae6bf94ee1d9a654241ab93a05cc734"
SEAM_SHA256 = "f973c7e606a112020c527bb72277b82586df915edc829a22305e587b35aec1b9"
LAYOUT_REPORTS = {
    "layouts/final_qap_shard_000_350.json": (
        "runs/assembly_v1/kaggle/final_qap_submission_output/v1/"
        "final_qap_shard_000_350.json",
        "541e7905dad9373a173c31db068429b20fb614d450f3cfb4439b89a0a45b2e2a",
    ),
    "layouts/final_qap_shard_350_700.json": (
        "runs/assembly_v1/kaggle/final_qap_submission_output/v1/"
        "final_qap_shard_350_700.json",
        "38e8a01de560ef8914f25dbe42b2c43c063bde07e3b537e71c0be28416f8dbc4",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--dataset-id",
        default="pasha883/vsos-postassembly-harmonizer-runtime",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _copy(source: Path, destination: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise RuntimeError(f"bundle source must be a regular non-symlink file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    os.chmod(destination, 0o644)


def _payload_sources(repo_root: Path) -> dict[str, Path]:
    sources: dict[str, Path] = {
        "scripts/build_harmonized_submission.py": repo_root
        / "scripts/build_harmonized_submission.py",
        "assets/selected_tilenaf_synth_50k.pt": repo_root
        / "runs/denoise_v2/release/selected_tilenaf_synth_50k.pt",
        "assets/seam_denoiser_gpu.pt": repo_root
        / "runs/assembly_v1/kaggle/seam_denoiser_gpu/seam_denoiser_gpu.pt",
    }
    for package in ("puzzle_assembly", "puzzle_denoise_v2"):
        root = repo_root / "src" / package
        for path in sorted(root.glob("*.py"), key=lambda value: value.name):
            sources[f"src/{package}/{path.name}"] = path
    for destination, (source, expected_hash) in LAYOUT_REPORTS.items():
        path = repo_root / source
        if _sha256(path) != expected_hash:
            raise RuntimeError(f"frozen layout report drift: {path}")
        sources[destination] = path
    if _sha256(sources["assets/selected_tilenaf_synth_50k.pt"]) != SELECTED_SHA256:
        raise RuntimeError("selected TileNAF checkpoint drift")
    if _sha256(sources["assets/seam_denoiser_gpu.pt"]) != SEAM_SHA256:
        raise RuntimeError("seam TileNAF checkpoint drift")
    return sources


def build_bundle(repo_root: Path, output_root: Path, dataset_id: str) -> dict[str, Any]:
    repo_root = repo_root.expanduser().resolve(strict=True)
    output_root = output_root.expanduser().absolute()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"output root must be fresh: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    if not dataset_id.startswith("pasha883/") or dataset_id.count("/") != 1:
        raise RuntimeError("unexpected private Kaggle dataset id")
    sources = _payload_sources(repo_root)
    with tempfile.TemporaryDirectory(
        prefix="harmonized-bundle-", dir=output_root.parent
    ) as temporary:
        staging = Path(temporary)
        payload_root = staging / "payload"
        for destination, source in sorted(sources.items()):
            _copy(source, payload_root / destination)
        members = []
        for destination in sorted(sources):
            path = payload_root / destination
            members.append(
                {
                    "path": destination,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
        manifest = {
            "schema_version": 1,
            "kind": "harmonized_submission_runtime_bundle_manifest",
            "safe_for_submission": False,
            "member_count": len(members),
            "members": members,
            "assets": {
                "selected_denoiser_sha256": SELECTED_SHA256,
                "seam_denoiser_sha256": SEAM_SHA256,
                "layout_report_sha256": sorted(
                    expected for _, expected in LAYOUT_REPORTS.values()
                ),
            },
        }
        manifest_path = payload_root / "BUNDLE_MANIFEST.json"
        manifest_path.write_bytes(_canonical_bytes(manifest))
        archive_path = staging / "harmonized_submission_runtime_v1.zip"
        with zipfile.ZipFile(
            archive_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            for path in sorted(payload_root.rglob("*")):
                if not path.is_file():
                    continue
                relative = path.relative_to(payload_root).as_posix()
                info = zipfile.ZipInfo(relative, date_time=ARCHIVE_TIMESTAMP)
                info.create_system = 3
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, path.read_bytes(), compresslevel=6)
        metadata = {
            "id": dataset_id,
            "title": "VSOS Postassembly Harmonizer Runtime",
            "isPrivate": True,
            "licenses": [{"name": "other"}],
        }
        metadata_path = staging / "dataset-metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
        receipt = {
            "schema_version": 1,
            "kind": "harmonized_submission_runtime_bundle_receipt",
            "created_utc": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
            "dataset_id": dataset_id,
            "archive": {
                "path": archive_path.name,
                "bytes": archive_path.stat().st_size,
                "sha256": _sha256(archive_path),
            },
            "manifest": {
                "path": "BUNDLE_MANIFEST.json",
                "bytes": manifest_path.stat().st_size,
                "sha256": _sha256(manifest_path),
            },
            "member_count": len(members),
            "members_sha256": hashlib.sha256(_canonical_bytes({"members": members})).hexdigest(),
            "upload_performed": False,
            "safe_for_submission": False,
        }
        receipt_path = staging / "BUNDLE_BUILD_RECEIPT.json"
        receipt_path.write_bytes(_canonical_bytes(receipt))
        os.replace(staging, output_root)
    return {
        "status": "harmonized_submission_runtime_bundle_built",
        "output_root": str(output_root),
        "archive_sha256": receipt["archive"]["sha256"],
        "manifest_sha256": receipt["manifest"]["sha256"],
        "member_count": len(members),
        "safe_for_submission": False,
    }


def main() -> None:
    args = parse_args()
    result = build_bundle(args.repo_root, args.output_root, args.dataset_id)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

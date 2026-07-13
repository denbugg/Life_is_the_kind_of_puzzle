#!/usr/bin/env python3
"""Materialize the exact code-v2 snapshot plus seven frozen verifier dependencies."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import zipfile

from scripts import verify_candidate_graph_oracle_v3_phase_a_composite as composite


def _atomic_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            composite._require(written > 0, f"short write: {path}")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def materialize(
    *, source_root: Path, archive: Path, destination: Path, receipt: Path
) -> dict[str, object]:
    for label, path in (("source_root", source_root), ("archive", archive)):
        composite._require(path.is_absolute(), f"{label} must be absolute")
        path.resolve(strict=True)
    for label, path in (("destination", destination), ("receipt", receipt)):
        composite._require(path.is_absolute(), f"{label} must be absolute")
        composite._require(not path.exists(), f"{label} already exists")
    source_root = source_root.resolve(strict=True)
    archive = archive.resolve(strict=True)
    composite._require(
        composite._sha(archive) == composite.SNAPSHOT_ARCHIVE_SHA256,
        "code-v2 archive SHA drift",
    )
    archive_members = composite._snapshot_archive_members(archive)
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as bundle:
        for relative in sorted(archive_members):
            data = bundle.read(relative)
            composite._require(
                hashlib.sha256(data).hexdigest() == archive_members[relative],
                f"archive member changed during extraction: {relative}",
            )
            _atomic_exclusive(destination / relative, data)

    supplements: dict[str, dict[str, object]] = {}
    for relative, expected_sha in sorted(composite.SUPPLEMENT_SHA256S.items()):
        source = source_root / relative
        data = composite._read_regular(source)
        actual_sha = hashlib.sha256(data).hexdigest()
        composite._require(
            actual_sha == expected_sha, f"supplement source SHA drift: {relative}"
        )
        _atomic_exclusive(destination / relative, data)
        supplements[relative] = {
            "expected_sha256": expected_sha,
            "source_actual_sha256": actual_sha,
            "bound_actual_sha256": composite._sha(destination / relative),
            "bytes": len(data),
        }
    closure = composite.verify_snapshot_exact(destination, archive)
    payload = {
        "schema_version": 1,
        "kind": "candidate_graph_oracle_v3_bound_verifier_repository_closure",
        "created_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "protocol_instance_id": composite.INSTANCE,
        "source_archive": {
            "path": str(archive),
            "sha256": composite.SNAPSHOT_ARCHIVE_SHA256,
            "member_count": len(archive_members),
        },
        "bound_repository": {
            "path": str(destination),
            **closure,
        },
        "supplements": supplements,
        "current_source_modules_copied": False,
        "supplement_count": len(supplements),
        "fixture_paths_constructed": False,
        "fixture_files_opened": False,
        "label_paths_constructed": False,
        "label_files_opened": False,
        "safe_for_submission": False,
    }
    envelope = {
        "payload": payload,
        "payload_sha256": hashlib.sha256(composite._canonical_object(payload)).hexdigest(),
    }
    receipt_sha = composite._write_exclusive(receipt, envelope)
    return {
        "status": "materialized_and_verified",
        "bound_repository": str(destination),
        "closure_receipt": str(receipt),
        "closure_receipt_sha256": receipt_sha,
        "archive_members": len(archive_members),
        "supplements": len(supplements),
        "total_files": len(archive_members) + len(supplements),
        "label_paths_constructed_or_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            materialize(
                source_root=args.source_root,
                archive=args.archive,
                destination=args.destination,
                receipt=args.receipt,
            ),
            sort_keys=True,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()

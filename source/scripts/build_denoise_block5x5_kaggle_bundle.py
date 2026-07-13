#!/usr/bin/env python3
"""Build the small deterministic code archive for the 5x5 Kaggle job."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile


FILES = (
    "configs/denoise_block5x5_v1.json",
    "configs/denoise_splits_seed20260710.json",
    "configs/denoise_validation_quarantine_v1.json",
    "scripts/evaluate_denoise_block5x5.py",
    "scripts/select_denoise_block5x5_candidate.py",
    "scripts/train_denoise_block5x5.py",
    "src/puzzle_assembly/__init__.py",
    "src/puzzle_assembly/compatibility.py",
    "src/puzzle_assembly/components.py",
    "src/puzzle_assembly/geometry.py",
    "src/puzzle_assembly/metrics.py",
    "src/puzzle_assembly/qap.py",
    "src/puzzle_assembly/solvers.py",
    "src/puzzle_denoise_v2/__init__.py",
    "src/puzzle_denoise_v2/block5x5.py",
    "src/puzzle_denoise_v2/degradation.py",
    "src/puzzle_denoise_v2/losses.py",
    "src/puzzle_denoise_v2/metrics.py",
    "src/puzzle_denoise_v2/model.py",
    "src/puzzle_denoise_v2/tiles.py",
    "src/puzzle_denoise_v2/training.py",
    "tests/test_denoise_block5x5.py",
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    info.create_system = 3
    return info


def build(root: Path, output: Path) -> dict:
    records = []
    contents: dict[str, bytes] = {}
    for relative in sorted(FILES):
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"missing regular bundle file: {relative}")
        payload = path.read_bytes()
        contents[relative] = payload
        records.append(
            {"path": relative, "bytes": len(payload), "sha256": sha256_bytes(payload)}
        )
    manifest = {
        "schema_version": 1,
        "kind": "denoise_block5x5_kaggle_code_bundle",
        "fixed_zip_timestamp": "1980-01-01T00:00:00",
        "files": records,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            archive.writestr(zip_info("MANIFEST.json"), manifest_bytes)
            for relative in sorted(contents):
                archive.writestr(zip_info(relative), contents[relative])
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "output": str(output.resolve()),
        "bytes": output.stat().st_size,
        "sha256": sha256_bytes(output.read_bytes()),
        "file_count": len(FILES),
        "manifest_sha256": sha256_bytes(manifest_bytes),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument(
        "--output",
        default="runs/denoise_v2/kaggle_block5x5_bundle/denoise_block5x5_code.zip",
    )
    args = parser.parse_args()
    result = build(Path(args.root).resolve(), Path(args.output))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Independently verify the completed Rank96 submission directory and ZIP."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(*, input_dir: Path, output_dir: Path, zip_path: Path) -> dict[str, object]:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    zip_path = zip_path.resolve()
    expected = sorted(
        path.name
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    )
    outputs = sorted(
        path.name
        for path in output_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    )
    if len(expected) != 700 or outputs != expected:
        raise ValueError("input/output PNG basename contract failed")

    bad_images: list[object] = []
    byte_mismatches: list[str] = []
    zip_hashes: dict[str, str] = {}
    with zipfile.ZipFile(zip_path, "r") as archive:
        names = archive.namelist()
        if names != expected or len(names) != len(set(names)) or len(names) != 700:
            raise ValueError("ZIP basename/order/duplicate contract failed")
        if any("/" in name or "\\" in name for name in names):
            raise ValueError("ZIP contains a directory or nested entry")
        corrupt = archive.testzip()
        if corrupt is not None:
            raise ValueError(f"ZIP CRC failure: {corrupt}")
        for name in names:
            data = archive.read(name)
            output_data = (output_dir / name).read_bytes()
            zip_hashes[name] = hashlib.sha256(data).hexdigest()
            if hashlib.sha256(data).digest() != hashlib.sha256(output_data).digest():
                byte_mismatches.append(name)
            with Image.open(io.BytesIO(data)) as image:
                image.load()
                if image.format != "PNG" or image.mode != "RGB" or image.size != (480, 480):
                    bad_images.append((name, image.format, image.mode, image.size))
    if bad_images or byte_mismatches:
        raise ValueError(
            f"ZIP image contract failed: bad={bad_images[:3]}, mismatches={byte_mismatches[:3]}"
        )

    manifest = json.loads((output_dir / "rank96_manifest.json").read_text(encoding="utf-8"))
    report = json.loads((output_dir / "rank96_report.json").read_text(encoding="utf-8"))
    zip_sha = sha256_file(zip_path)
    if report.get("status") != "completed":
        raise ValueError("inference report is not complete")
    if report.get("input_count") != 700 or report.get("completed_count") != 700:
        raise ValueError("inference report count mismatch")
    if report.get("generic_count") != 682 or report.get("override_count") != 18:
        raise ValueError("inference report source count mismatch")
    if report.get("output_zip_sha256") != zip_sha:
        raise ValueError("inference report ZIP SHA mismatch")

    entries = manifest.get(
        "completed", manifest.get("entries", manifest.get("images", manifest.get("outputs")))
    )
    if isinstance(entries, dict):
        rows = list(entries.values())
    elif isinstance(entries, list):
        rows = entries
    else:
        raise ValueError("manifest output entries are missing")
    if isinstance(entries, dict):
        if set(entries) != set(expected):
            raise ValueError("manifest basename set differs from the input set")
        for name, row in entries.items():
            if row.get("output_sha256") != zip_hashes[name]:
                raise ValueError(f"manifest output SHA mismatch: {name}")
    sources = Counter(row.get("source") for row in rows)
    if sources != Counter({"rank96": 682, "verified_source_override": 18}):
        raise ValueError(f"manifest source counts differ: {dict(sources)}")

    return {
        "status": "independent_validation_pass",
        "input_pngs": len(expected),
        "output_pngs": len(outputs),
        "zip_entries": 700,
        "zip_testzip": None,
        "bad_images": 0,
        "byte_mismatches": 0,
        "source_counts": dict(sources),
        "zip_size_bytes": zip_path.stat().st_size,
        "zip_sha256": zip_sha,
        "manifest_schema": manifest.get("schema"),
        "report_schema": report.get("schema"),
    }


def main() -> int:
    workspace = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("E:/pazzle_data/test"))
    parser.add_argument(
        "--output-dir", type=Path, default=workspace / "artifacts" / "rank96_submission_v1"
    )
    parser.add_argument(
        "--zip", dest="zip_path", type=Path, default=workspace / "artifacts" / "submission_rank96_v1.zip"
    )
    args = parser.parse_args()
    print(
        json.dumps(
            verify(input_dir=args.input_dir, output_dir=args.output_dir, zip_path=args.zip_path),
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

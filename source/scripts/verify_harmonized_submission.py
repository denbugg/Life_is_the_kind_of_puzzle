#!/usr/bin/env python3
"""Independently verify the 700-image frozen-layout harmonized submission."""

from __future__ import annotations

import argparse
import hashlib
import json
from io import BytesIO
from pathlib import Path
import re
import zipfile

import numpy as np
from PIL import Image

from scripts.build_harmonized_submission import (
    ARCHIVE_TIMESTAMP,
    load_frozen_layouts,
    sha256_file,
)


EXPECTED_OLD_SUBMISSION_SHA256 = (
    "1eeae828dd893198c07ac502d29aa5eeebd54bf6b818293d3b7e3f67ecb59607"
)
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--builder-report", type=Path, required=True)
    parser.add_argument("--old-submission", type=Path, required=True)
    parser.add_argument("--layout-report", type=Path, action="append", required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _decode(payload: bytes, name: str) -> np.ndarray:
    with Image.open(BytesIO(payload)) as image:
        if image.mode != "RGB" or image.size != (480, 480):
            raise RuntimeError(f"invalid PNG contract: {name}")
        values = np.asarray(image, dtype=np.uint8)
    return np.ascontiguousarray(values)


def _seam_discontinuity(image: np.ndarray) -> float:
    values = image.astype(np.float32)
    vertical = [
        np.abs(values[:, column - 1] - values[:, column]).mean()
        for column in range(20, 480, 20)
    ]
    horizontal = [
        np.abs(values[row - 1] - values[row]).mean()
        for row in range(20, 480, 20)
    ]
    return float(np.mean([*vertical, *horizontal]) / 255.0)


def verify(args: argparse.Namespace) -> dict:
    submission = args.submission.expanduser().resolve(strict=True)
    builder_report_path = args.builder_report.expanduser().resolve(strict=True)
    old_submission = args.old_submission.expanduser().resolve(strict=True)
    test_dir = args.test_dir.expanduser().resolve(strict=True)
    output = args.output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"verification output must be fresh: {output}")
    if sha256_file(old_submission) != EXPECTED_OLD_SUBMISSION_SHA256:
        raise RuntimeError("old LB-0.203 submission archive drift")
    report = json.loads(builder_report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise RuntimeError("builder report is not a JSON object")
    submission_sha = sha256_file(submission)
    if (
        report.get("kind") != "harmonized_frozen_qap_submission_report"
        or report.get("status") != "test_only_candidate_not_lb_scored"
        or report.get("anti_leakage", {}).get("target_paths_or_pixels_read") is not False
        or report.get("anti_leakage", {}).get("layout_recomputed") is not False
        or report.get("archive", {}).get("sha256") != submission_sha
        or report.get("count") != 700
    ):
        raise RuntimeError("builder report contract failed")
    layouts, layout_reports = load_frozen_layouts(
        args.layout_report, expected_count=700
    )
    names = sorted(path.name for path in test_dir.glob("*.png"))
    if len(names) != 700 or len(set(names)) != 700 or set(names) != set(layouts):
        raise RuntimeError("test filename contract failed")
    source_records = report.get("sources")
    if not isinstance(source_records, list) or len(source_records) != 700:
        raise RuntimeError("builder source record count drift")
    by_name = {record.get("source"): record for record in source_records}
    if set(by_name) != set(names) or len(by_name) != 700:
        raise RuntimeError("builder source record names drift")

    changed_images = 0
    mean_absolute_differences: list[float] = []
    changed_pixel_fractions: list[float] = []
    old_seams: list[float] = []
    new_seams: list[float] = []
    member_hashes: list[dict[str, object]] = []
    with zipfile.ZipFile(submission) as new_archive, zipfile.ZipFile(
        old_submission
    ) as old_archive:
        if new_archive.namelist() != names or old_archive.namelist() != names:
            raise RuntimeError("archive member names/order differ from test inputs")
        for name in names:
            info = new_archive.getinfo(name)
            if (
                Path(info.filename).name != info.filename
                or info.date_time != ARCHIVE_TIMESTAMP
                or info.create_system != 3
                or info.compress_type != zipfile.ZIP_DEFLATED
                or (info.external_attr >> 16) != 0o100644
            ):
                raise RuntimeError(f"new archive member metadata drift: {name}")
            new_payload = new_archive.read(name)
            old_payload = old_archive.read(name)
            new_hash = hashlib.sha256(new_payload).hexdigest()
            record = by_name[name]
            frozen = layouts[name]
            if (
                not SHA_RE.fullmatch(new_hash)
                or record.get("output_png_sha256") != new_hash
                or record.get("layout_sha256") != frozen["layout_sha256"]
                or record.get("input_pixel_sha256") != frozen["input_pixel_sha256"]
            ):
                raise RuntimeError(f"source provenance/hash drift: {name}")
            new_image = _decode(new_payload, name)
            old_image = _decode(old_payload, name)
            difference = np.abs(
                new_image.astype(np.int16) - old_image.astype(np.int16)
            )
            if np.any(difference):
                changed_images += 1
            mean_absolute_differences.append(float(difference.mean()))
            changed_pixel_fractions.append(float(np.mean(np.any(difference != 0, axis=2))))
            old_seams.append(_seam_discontinuity(old_image))
            new_seams.append(_seam_discontinuity(new_image))
            member_hashes.append(
                {"name": name, "bytes": len(new_payload), "sha256": new_hash}
            )
    if changed_images != 700:
        raise RuntimeError(
            f"expected the confirmed renderer to change all 700 images, changed {changed_images}"
        )
    result = {
        "schema_version": 1,
        "kind": "harmonized_submission_independent_verification",
        "status": "verified_candidate_ready_not_lb_scored",
        "submission": {
            "path": str(submission),
            "bytes": submission.stat().st_size,
            "sha256": submission_sha,
            "member_count": 700,
        },
        "builder_report": {
            "path": str(builder_report_path),
            "sha256": sha256_file(builder_report_path),
        },
        "old_submission": {
            "path": str(old_submission),
            "sha256": EXPECTED_OLD_SUBMISSION_SHA256,
            "known_user_lb_score": 0.203,
        },
        "layout_reports": layout_reports,
        "checks": {
            "all_700_pngs_decode_rgb_480x480": True,
            "all_member_hashes_match_builder_report": True,
            "all_input_and_layout_hashes_match_frozen_lb_artifact": True,
            "archive_metadata_canonical": True,
            "target_paths_or_pixels_read": False,
            "layout_recomputed": False,
        },
        "target_free_comparison_to_old_renderer": {
            "changed_images": changed_images,
            "mean_absolute_pixel_delta": float(np.mean(mean_absolute_differences)),
            "median_absolute_pixel_delta": float(np.median(mean_absolute_differences)),
            "mean_changed_pixel_fraction": float(np.mean(changed_pixel_fractions)),
            "old_mean_untargeted_seam_discontinuity": float(np.mean(old_seams)),
            "new_mean_untargeted_seam_discontinuity": float(np.mean(new_seams)),
            "mean_untargeted_seam_delta": float(
                np.mean(np.asarray(new_seams) - np.asarray(old_seams))
            ),
        },
        "members_sha256": hashlib.sha256(
            json.dumps(
                {"members": member_hashes},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "safe_for_submission": True,
        "leaderboard_score": None,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    result = verify(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

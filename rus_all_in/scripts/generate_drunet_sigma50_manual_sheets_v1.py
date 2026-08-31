#!/usr/bin/env python3
"""Create target-free all-700 manual sheets bound to a frozen B commitment."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import stat
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUTS = PROJECT_ROOT / "data/raw/train/inputs"
OUTPUT_ROOT = PROJECT_ROOT / "outputs/drunet-sigma50-protected/all700-measurement-v1"
STAGE_ROOTS = {
    "calibration": OUTPUT_ROOT / "calibration700",
    "holdout": OUTPUT_ROOT / "holdout700",
}
MANUAL_DIRECTORY_NAME = "manual-review-target-free-v1"
REPORT_BINDING_NAME = "manual-review-target-free-v1.report-binding.json"
SOURCE_SIDECAR = Path(f"{Path(__file__).resolve()}.sha256")
EXPECTED_CONFIG_SHA256 = "a402fc682b0db96b60004fa2c33ea70baf06035cb4971b8ee0778ceb1b7f05ac"
BOARD_COUNT = 700
PAGE_SIZE = 100
WORST_COUNT = 24
WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("generate", "bind-report"), required=True)
    parser.add_argument("--stage", choices=tuple(STAGE_ROOTS), required=True)
    parser.add_argument("--run", action="store_true")
    return parser.parse_args()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def require_readonly(path: Path) -> None:
    if path.stat().st_mode & WRITE_BITS:
        raise PermissionError(f"integrity-bound path is writable: {path}")


def write_json_exclusive_readonly(path: Path, payload: Mapping[str, Any]) -> str:
    encoded = canonical_json_bytes(payload)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(path, 0o444)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    require_readonly(path)
    return sha256_bytes(encoded)


def verify_own_source() -> str:
    source = Path(__file__).resolve()
    require_readonly(source)
    require_readonly(SOURCE_SIDECAR)
    observed = sha256_file(source)
    expected = SOURCE_SIDECAR.read_text(encoding="utf-8").split()[0]
    if observed != expected:
        raise ValueError("manual-sheet generator source differs from readonly sidecar")
    return observed


def load_rgb_png(path: Path, *, expected_png_sha256: str) -> Image.Image:
    payload = path.read_bytes()
    if sha256_bytes(payload) != expected_png_sha256:
        raise ValueError(f"PNG hash mismatch: {path}")
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        if image.format != "PNG" or image.mode != "RGB" or image.size != (480, 480):
            raise ValueError(f"expected strict RGB 480x480 PNG: {path}")
        return image.copy()


def pixel_sha256(image: Image.Image) -> str:
    return sha256_bytes(np.asarray(image, dtype=np.uint8).tobytes())


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def competition_badness(values: Sequence[float]) -> np.ndarray:
    """Return tied target-free high-is-bad empirical ranks in [0, 1]."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all():
        raise ValueError("badness inputs must be a finite vector with at least two values")
    if float(array.max()) == float(array.min()):
        return np.zeros_like(array)
    return np.asarray([np.sum(array < value) / (len(array) - 1) for value in array])


def load_committed_boards(stage: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = STAGE_ROOTS[stage]
    commitment_path = root / "prediction-commitment.json"
    receipt_path = root / "commitment-receipt.json"
    for path in (commitment_path, receipt_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing stage commitment artifact: {path}")
        require_readonly(path)
    commitment = json.loads(commitment_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    commitment_sha256 = sha256_file(commitment_path)
    if (
        commitment.get("stage") != stage
        or commitment.get("count") != BOARD_COUNT
        or commitment.get("config_sha256") != EXPECTED_CONFIG_SHA256
        or receipt.get("commitment_sha256") != commitment_sha256
        or receipt.get("candidate_roster_sha256") != commitment.get("candidate_roster_sha256")
        or receipt.get("targets_decoded_before_receipt") is not False
    ):
        raise ValueError("stage commitment or receipt binding changed")

    boards: list[dict[str, Any]] = []
    for ranked_index, item in enumerate(commitment["boards"]):
        filename = str(item["filename"])
        metadata_path = root / item["record_relative_path"]
        require_readonly(metadata_path)
        if sha256_file(metadata_path) != item["record_sha256"]:
            raise ValueError(f"committed record hash changed: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("filename") != filename
            or metadata.get("config_sha256") != EXPECTED_CONFIG_SHA256
            or metadata.get("stage") != stage
            or metadata["images"]["candidate"]["pixel_sha256"] != item["candidate_pixel_sha256"]
        ):
            raise ValueError(f"committed board binding changed: {metadata_path}")
        input_path = INPUTS / filename
        input_png_sha256 = sha256_file(input_path)
        if input_png_sha256 != metadata["input_sha256"]:
            raise ValueError(f"input hash changed: {input_path}")
        candidate_record = metadata["images"]["candidate"]
        candidate_path = metadata_path.parent / candidate_record["filename"]
        require_readonly(candidate_path)
        candidate = load_rgb_png(
            candidate_path,
            expected_png_sha256=candidate_record["png_sha256"],
        )
        if pixel_sha256(candidate) != candidate_record["pixel_sha256"]:
            raise ValueError(f"candidate pixel hash changed: {candidate_path}")
        safety = metadata["safety"]
        grid_ratio = float(
            safety["candidate_structure"]["grid_ratio"]
            / max(float(safety["reference_structure"]["grid_ratio"]), 1e-12)
        )
        flatness = safety["candidate_tile_flatness"]
        boards.append(
            {
                "ranked_index": ranked_index,
                "filename": filename,
                "input_path": input_path,
                "input_png_sha256": input_png_sha256,
                "candidate_path": candidate_path,
                "candidate_png_sha256": candidate_record["png_sha256"],
                "candidate_pixel_sha256": candidate_record["pixel_sha256"],
                "record_sha256": item["record_sha256"],
                "grid_ratio_relative_to_h28": grid_ratio,
                "candidate_near_flat_std_lt_2_count": int(
                    flatness["near_flat_tiles_global_std_lt_2"]
                ),
                "candidate_exact_constant_count": int(
                    flatness["exact_spatially_constant_rgb_tiles"]
                ),
                "maximum_abs_rgb_mean_shift_vs_h28": float(
                    safety["maximum_abs_rgb_mean_shift_vs_reference"]
                ),
            }
        )
    if len(boards) != BOARD_COUNT or len({row["filename"] for row in boards}) != BOARD_COUNT:
        raise ValueError("commitment does not contain 700 unique boards")
    return commitment, boards


def label_for(row: Mapping[str, Any], *, sorted_index: int | None = None) -> str:
    prefix = f"rank {int(row['ranked_index']):03d}"
    if sorted_index is not None:
        prefix = f"sorted {sorted_index:03d} | {prefix}"
    return f"{prefix} | {row['filename']} | dirty / B"


def paired_thumbnail_cell(row: Mapping[str, Any], label: str) -> Image.Image:
    font = load_font(14)
    thumbnail_size = 128
    gap = 4
    label_height = 22
    cell = Image.new("RGB", (thumbnail_size * 2 + gap, thumbnail_size + label_height), "white")
    dirty = load_rgb_png(Path(row["input_path"]), expected_png_sha256=row["input_png_sha256"])
    candidate = load_rgb_png(
        Path(row["candidate_path"]), expected_png_sha256=row["candidate_png_sha256"]
    )
    dirty.thumbnail((thumbnail_size, thumbnail_size), Image.Resampling.LANCZOS)
    candidate.thumbnail((thumbnail_size, thumbnail_size), Image.Resampling.LANCZOS)
    cell.paste(dirty, (0, label_height))
    cell.paste(candidate, (thumbnail_size + gap, label_height))
    ImageDraw.Draw(cell).text((2, 2), label, fill="black", font=font)
    return cell


def render_sorted_page(rows: Sequence[Mapping[str, Any]], page_index: int) -> Image.Image:
    if len(rows) != PAGE_SIZE:
        raise ValueError("each sorted sheet must contain exactly 100 boards")
    columns = 10
    rows_per_page = 10
    cell_width = 260
    cell_height = 150
    header_height = 30
    sheet = Image.new(
        "RGB",
        (columns * cell_width, header_height + rows_per_page * cell_height),
        "white",
    )
    ImageDraw.Draw(sheet).text(
        (8, 6),
        f"Target-free dirty / fixed-B — sorted page {page_index + 1}/7",
        fill="black",
        font=load_font(16),
    )
    for offset, row in enumerate(rows):
        sorted_index = page_index * PAGE_SIZE + offset
        cell = paired_thumbnail_cell(row, label_for(row, sorted_index=sorted_index))
        x = (offset % columns) * cell_width
        y = header_height + (offset // columns) * cell_height
        sheet.paste(cell, (x, y))
    return sheet


def select_target_free_worst(boards: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_names = (
        "grid_ratio_relative_to_h28",
        "candidate_near_flat_std_lt_2_count",
        "candidate_exact_constant_count",
        "maximum_abs_rgb_mean_shift_vs_h28",
    )
    percentiles = {
        name: competition_badness([float(row[name]) for row in boards]) for name in metric_names
    }
    ranked: list[dict[str, Any]] = []
    for index, source in enumerate(boards):
        row = dict(source)
        components = {name: float(percentiles[name][index]) for name in metric_names}
        row["badness_percentiles"] = components
        row["composite_badness"] = float(
            max(components.values()) + 0.25 * np.mean(list(components.values()))
        )
        ranked.append(row)
    ranked.sort(key=lambda row: (-float(row["composite_badness"]), str(row["filename"])))
    return ranked[:WORST_COUNT]


def render_worst_sheet(rows: Sequence[Mapping[str, Any]]) -> Image.Image:
    if len(rows) != WORST_COUNT:
        raise ValueError("worst sheet must contain exactly 24 boards")
    columns = 4
    rows_per_page = 6
    image_size = 480
    gap = 6
    label_height = 28
    cell_width = image_size * 2 + gap
    cell_height = image_size + label_height
    header_height = 40
    sheet = Image.new(
        "RGB",
        (columns * cell_width, header_height + rows_per_page * cell_height),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (8, 8),
        "Target-free worst 24: grid / near-flat / exact-constant / RGB-shift — dirty / B",
        fill="black",
        font=load_font(20),
    )
    font = load_font(16)
    for offset, row in enumerate(rows):
        dirty = load_rgb_png(Path(row["input_path"]), expected_png_sha256=row["input_png_sha256"])
        candidate = load_rgb_png(
            Path(row["candidate_path"]), expected_png_sha256=row["candidate_png_sha256"]
        )
        x = (offset % columns) * cell_width
        y = header_height + (offset // columns) * cell_height
        sheet.paste(dirty, (x, y + label_height))
        sheet.paste(candidate, (x + image_size + gap, y + label_height))
        label = (
            f"#{offset + 1:02d} rank {int(row['ranked_index']):03d} {row['filename']} "
            f"grid {float(row['grid_ratio_relative_to_h28']):.4f} "
            f"nf2 {int(row['candidate_near_flat_std_lt_2_count'])} "
            f"const {int(row['candidate_exact_constant_count'])} "
            f"shift {float(row['maximum_abs_rgb_mean_shift_vs_h28']):.2f}"
        )
        ImageDraw.Draw(sheet).text((x + 2, y + 3), label, fill="black", font=font)
    return sheet


def save_png_readonly(path: Path, image: Image.Image) -> dict[str, Any]:
    image.save(path, format="PNG", compress_level=6)
    os.chmod(path, 0o444)
    require_readonly(path)
    return {
        "filename": path.name,
        "png_sha256": sha256_file(path),
        "width": image.width,
        "height": image.height,
    }


def generate(stage: str) -> None:
    source_sha256 = verify_own_source()
    root = STAGE_ROOTS[stage]
    if (root / "report.json").exists() or (root / "targets-opened-receipt.json").exists():
        raise RuntimeError("manual sheets must be generated after commitment and before score")
    final_directory = root / MANUAL_DIRECTORY_NAME
    if final_directory.exists():
        raise FileExistsError(f"manual artifact directory already exists: {final_directory}")
    commitment, boards = load_committed_boards(stage)
    sorted_boards = sorted(boards, key=lambda row: str(row["filename"]))
    worst = select_target_free_worst(boards)
    temporary = Path(tempfile.mkdtemp(prefix=f".{MANUAL_DIRECTORY_NAME}.", dir=root))
    try:
        sheets: list[dict[str, Any]] = []
        for page_index in range(7):
            start = page_index * PAGE_SIZE
            page_rows = sorted_boards[start : start + PAGE_SIZE]
            sheet_path = temporary / f"all700-sorted-{start + 1:03d}-{start + PAGE_SIZE:03d}.png"
            record = save_png_readonly(sheet_path, render_sorted_page(page_rows, page_index))
            record["count"] = len(page_rows)
            record["filenames"] = [row["filename"] for row in page_rows]
            sheets.append(record)
        worst_path = temporary / "worst24-target-free-high-resolution.png"
        worst_sheet = save_png_readonly(worst_path, render_worst_sheet(worst))
        roster_lines = [
            f"{row['filename']} {row['input_png_sha256']} {row['candidate_pixel_sha256']}"
            for row in sorted_boards
        ]
        manifest = {
            "schema": "aiijc-drunet-sigma50-target-free-manual-sheets-v1",
            "status": (
                "immutable_target_free_manual_artifacts_created_after_commitment_before_score"
            ),
            "stage": stage,
            "count": len(boards),
            "config_sha256": EXPECTED_CONFIG_SHA256,
            "generator_relative_path": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
            "generator_sha256": source_sha256,
            "commitment_relative_path": str(
                (root / "prediction-commitment.json").relative_to(PROJECT_ROOT)
            ),
            "commitment_sha256": sha256_file(root / "prediction-commitment.json"),
            "commitment_receipt_sha256": sha256_file(root / "commitment-receipt.json"),
            "candidate_roster_sha256": commitment["candidate_roster_sha256"],
            "sorted_visual_roster_sha256": sha256_bytes("\n".join(roster_lines).encode()),
            "all_700_sorted_contact_sheets": sheets,
            "worst24_selection": {
                "target_free": True,
                "metrics": [
                    "grid_ratio_relative_to_same_drunet50_h28_reference",
                    "candidate_near_flat_std_lt_2_tile_count",
                    "candidate_exact_constant_tile_count",
                    "maximum_abs_rgb_mean_shift_vs_same_drunet50_h28_reference",
                ],
                "badness": (
                    "tied high-is-bad empirical percentile per metric; "
                    "composite=max(percentiles)+0.25*mean(percentiles); "
                    "descending composite then filename"
                ),
                "sheet": worst_sheet,
                "boards": [
                    {
                        key: row[key]
                        for key in (
                            "ranked_index",
                            "filename",
                            "candidate_pixel_sha256",
                            "grid_ratio_relative_to_h28",
                            "candidate_near_flat_std_lt_2_count",
                            "candidate_exact_constant_count",
                            "maximum_abs_rgb_mean_shift_vs_h28",
                            "badness_percentiles",
                            "composite_badness",
                        )
                    }
                    for row in worst
                ],
            },
            "visual_contract": {
                "only_dirty_input_and_committed_B_candidate_pixels": True,
                "targets_opened_or_rendered": False,
                "seven_pages_of_exactly_100_sorted_unique_boards": True,
                "all_700_unique_boards_covered_exactly_once": True,
                "labels_include_filename_and_commitment_ranked_index": True,
                "manual_review_status": (
                    "not_performed; root reviewer must create a separate bound review JSON"
                ),
            },
            "report_binding_status": "pending_separate_readonly_binding_after_report_exists",
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_bytes(canonical_json_bytes(manifest))
        os.chmod(manifest_path, 0o444)
        for path in temporary.iterdir():
            require_readonly(path)
        os.chmod(temporary, 0o555)
        os.rename(temporary, final_directory)
    finally:
        if temporary.exists():
            os.chmod(temporary, 0o755)
            shutil.rmtree(temporary)
    print(
        json.dumps(
            {
                "manual_directory": str(final_directory),
                "manifest_sha256": sha256_file(final_directory / "manifest.json"),
                "sheet_count": 8,
                "stage": stage,
            },
            indent=2,
        )
    )


def bind_report(stage: str) -> None:
    source_sha256 = verify_own_source()
    root = STAGE_ROOTS[stage]
    manual_root = root / MANUAL_DIRECTORY_NAME
    manifest_path = manual_root / "manifest.json"
    report_path = root / "report.json"
    commitment_path = root / "prediction-commitment.json"
    for path in (manifest_path, report_path, commitment_path):
        if not path.is_file():
            raise FileNotFoundError(f"binding input missing: {path}")
        require_readonly(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        manifest.get("stage") != stage
        or manifest.get("generator_sha256") != source_sha256
        or manifest.get("commitment_sha256") != sha256_file(commitment_path)
        or report.get("stage") != stage
        or report.get("config_sha256") != EXPECTED_CONFIG_SHA256
        or report.get("commitment_sha256") != sha256_file(commitment_path)
        or report.get("candidate_roster_sha256") != manifest.get("candidate_roster_sha256")
    ):
        raise ValueError("manual manifest, commitment, or report binding changed")
    sheets = [*manifest["all_700_sorted_contact_sheets"], manifest["worst24_selection"]["sheet"]]
    for sheet in sheets:
        path = manual_root / sheet["filename"]
        require_readonly(path)
        if sha256_file(path) != sheet["png_sha256"]:
            raise ValueError(f"manual sheet changed: {path}")
    binding = {
        "schema": "aiijc-drunet-sigma50-manual-sheets-report-binding-v1",
        "status": "immutable_cryptographic_binding_only_not_a_manual_pass_or_certification",
        "stage": stage,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "generator_sha256": source_sha256,
        "commitment_sha256": sha256_file(commitment_path),
        "manual_manifest_sha256": sha256_file(manifest_path),
        "report_sha256": sha256_file(report_path),
        "candidate_roster_sha256": report["candidate_roster_sha256"],
        "sheet_png_sha256": {sheet["filename"]: sheet["png_sha256"] for sheet in sheets},
        "manual_review_status": "not_performed; no visual conclusion is claimed by this binding",
    }
    path = root / REPORT_BINDING_NAME
    digest = write_json_exclusive_readonly(path, binding)
    print(json.dumps({"report_binding": str(path), "sha256": digest}, indent=2))


def main() -> None:
    args = parse_args()
    if not args.run:
        raise SystemExit("Refusing to run without --run")
    if args.phase == "generate":
        generate(args.stage)
    else:
        bind_report(args.stage)


if __name__ == "__main__":
    main()

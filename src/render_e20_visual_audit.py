"""Render the completed E20 sparse relative clusters without inventing a board.

The renderer is deliberately post-run and read-only with respect to the E20
experiment.  Relative tile coordinates come exclusively from
``rows[].core.selected.relative_entries``.  Ground-truth targets,
permutations, and label-derived metrics never influence placement.  The
second panel is a clearly labelled tile-wise fixed-NLM preview; missing cells
remain neutral and are never inferred.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np

import eval_clean_score_oracle as e12
import eval_e17_cc192_rigid_viability as e17
import eval_e20_triangle_potential_viability as e20


TILE_SIZE = 20
GRID = 24
NEUTRAL_RGB = (52, 52, 52)
PANEL_BACKGROUND_RGB = (22, 22, 22)
TEXT_RGB = (245, 245, 245)
MUTED_TEXT_RGB = (185, 185, 185)
PANEL_HEADER = 42
GLOBAL_HEADER = 78
PANEL_MIN_WIDTH = 500
PANEL_GAP = 12

DEFAULT_REPORT = Path(
    "E:/pazzle_work/triangle_pose_e20/cc192_triangle_potential_viability_v1.json"
)
DEFAULT_OUT_DIR = Path("E:/pazzle_work/visual_audit_e20")


class E20VisualAuditError(RuntimeError):
    """The completed report, sparse layout, source tiles, or E: path drifted."""


def _require_e_drive(path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    if resolved.drive.upper() != "E:":
        raise E20VisualAuditError(f"{label} must stay on E:, got {resolved}")
    return resolved


def _strict_int(value: object, *, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise E20VisualAuditError(f"{label} is not an integer")
    return int(value)


def _sha256_file(path: Path) -> str:
    return e12.sha256_file(path.resolve())


def _verify_complete_report_read_only(
    report_path: Path,
) -> tuple[Mapping[str, Any], str]:
    """Authenticate and replay-validate an already-complete report, never resume it."""

    resolved = _require_e_drive(report_path, label="E20 visual-audit report")
    if not resolved.is_file():
        raise E20VisualAuditError(f"completed E20 report is missing: {resolved}")
    before_digest = _sha256_file(resolved)
    try:
        preflight = e20._load_json(resolved, label="E20 visual-audit report")
    except e20.E20ContractError as exc:
        raise E20VisualAuditError(str(exc)) from exc
    if preflight.get("status") != "complete":
        raise E20VisualAuditError(
            "visual audit requires an already-complete E20 report; "
            "it will not resume or modify an incomplete run"
        )
    if preflight.get("completed_images") != list(e12.CALIBRATION_IDS):
        raise E20VisualAuditError("completed E20 image IDs are not exactly 10..17")

    paths = e20.E20Paths(
        raw_cache_dir=e20.DEFAULT_RAW_CACHE_DIR,
        calibration_report=e20.DEFAULT_CALIBRATION_REPORT,
        e12_report=e20.DEFAULT_E12_REPORT,
        e19_report=e20.DEFAULT_E19_REPORT,
        report=resolved,
    )
    try:
        validated = e20.run_gate(paths)
    except Exception as exc:
        raise E20VisualAuditError(
            f"completed E20 report failed exact read-only replay: {exc}"
        ) from exc
    after_digest = _sha256_file(resolved)
    if before_digest != after_digest:
        raise E20VisualAuditError("E20 report bytes changed during read-only validation")
    if validated != preflight:
        raise E20VisualAuditError("E20 report payload changed during validation")
    return validated, before_digest


def _rows_by_image(report: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    rows = report.get("rows")
    if not isinstance(rows, list) or len(rows) != len(e12.CALIBRATION_IDS):
        raise E20VisualAuditError("E20 visual audit requires exactly eight rows")
    output: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise E20VisualAuditError("E20 report contains a non-object row")
        image = _strict_int(row.get("image"), label="E20 row image")
        if image in output or image not in e12.CALIBRATION_IDS:
            raise E20VisualAuditError("E20 row image IDs are duplicated or invalid")
        output[image] = row
    if tuple(sorted(output)) != e12.CALIBRATION_IDS:
        raise E20VisualAuditError("E20 report rows are not exactly images 10..17")
    return output


def _selected_entries(
    row: Mapping[str, Any],
) -> tuple[tuple[int, int, int], ...]:
    core = row.get("core")
    if not isinstance(core, Mapping):
        raise E20VisualAuditError("E20 row has no core payload")
    selected = core.get("selected")
    if not isinstance(selected, Mapping):
        raise E20VisualAuditError("E20 row has no selected sparse cluster")
    raw_entries = selected.get("relative_entries")
    if not isinstance(raw_entries, (list, tuple)) or not raw_entries:
        raise E20VisualAuditError("selected sparse cluster has no relative entries")
    entries: list[tuple[int, int, int]] = []
    for raw in raw_entries:
        if not isinstance(raw, (list, tuple)) or len(raw) != 3:
            raise E20VisualAuditError("selected relative entry is malformed")
        tile, row_value, col_value = (
            _strict_int(raw[0], label="relative tile"),
            _strict_int(raw[1], label="relative row"),
            _strict_int(raw[2], label="relative column"),
        )
        if not 0 <= tile < e12.NFRAG:
            raise E20VisualAuditError("relative tile ID is outside 0..575")
        if not 0 <= row_value < GRID or not 0 <= col_value < GRID:
            raise E20VisualAuditError("relative coordinate is outside the 24-cell span")
        entries.append((tile, row_value, col_value))
    value = tuple(entries)
    if value != tuple(sorted(value)):
        raise E20VisualAuditError("relative entries are not canonical")
    if len({tile for tile, _row, _col in value}) != len(value):
        raise E20VisualAuditError("relative entries repeat a tile")
    if len({(row_value, col_value) for _tile, row_value, col_value in value}) != len(
        value
    ):
        raise E20VisualAuditError("relative entries collide")
    if min(row_value for _tile, row_value, _col in value) != 0 or min(
        col_value for _tile, _row, col_value in value
    ) != 0:
        raise E20VisualAuditError("relative entries are not min-coordinate normalized")
    height = max(row_value for _tile, row_value, _col in value) + 1
    width = max(col_value for _tile, _row, col_value in value) + 1
    if (
        _strict_int(selected.get("rigid_tiles"), label="selected rigid tiles")
        != len(value)
        or _strict_int(selected.get("bbox_height"), label="selected bbox height")
        != height
        or _strict_int(selected.get("bbox_width"), label="selected bbox width")
        != width
    ):
        raise E20VisualAuditError("selected sparse count/bbox disagrees with entries")
    return value


def _validate_tiles(tiles: np.ndarray) -> np.ndarray:
    value = np.asarray(tiles)
    if value.shape != (e12.NFRAG, TILE_SIZE, TILE_SIZE, 3) or value.dtype != np.uint8:
        raise E20VisualAuditError(
            "source tiles must be upright uint8 RGB 576x20x20x3"
        )
    return np.ascontiguousarray(value)


def _sparse_canvas(
    tiles: np.ndarray,
    entries: Sequence[tuple[int, int, int]],
    *,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Place upright source tiles in the exact reported relative coordinates."""

    source = _validate_tiles(tiles)
    values = tuple(entries)
    if not values:
        raise E20VisualAuditError("cannot render an empty sparse cluster")
    height = max(row for _tile, row, _col in values) + 1
    width = max(col for _tile, _row, col in values) + 1
    canvas = np.full(
        (height * TILE_SIZE, width * TILE_SIZE, 3),
        NEUTRAL_RGB,
        dtype=np.uint8,
    )
    seen_tiles: set[int] = set()
    seen_coordinates: set[tuple[int, int]] = set()
    for tile, row, col in values:
        if tile in seen_tiles or (row, col) in seen_coordinates:
            raise E20VisualAuditError("sparse render received duplicate tile/coordinate")
        if not 0 <= tile < e12.NFRAG or not 0 <= row < GRID or not 0 <= col < GRID:
            raise E20VisualAuditError("sparse render entry is outside the frozen geometry")
        seen_tiles.add(tile)
        seen_coordinates.add((row, col))
        pixels = source[tile]
        if transform is not None:
            pixels = np.asarray(transform(np.ascontiguousarray(pixels)))
            if pixels.shape != (TILE_SIZE, TILE_SIZE, 3) or pixels.dtype != np.uint8:
                raise E20VisualAuditError("tile preview transform changed shape/dtype")
        y0, x0 = row * TILE_SIZE, col * TILE_SIZE
        canvas[y0 : y0 + TILE_SIZE, x0 : x0 + TILE_SIZE] = pixels
    return canvas


def _put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float,
    colour: tuple[int, int, int] = TEXT_RGB,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        colour,
        1,
        cv2.LINE_AA,
    )


def _panel(canvas: np.ndarray, title: str) -> np.ndarray:
    width = max(PANEL_MIN_WIDTH, int(canvas.shape[1]))
    panel = np.full(
        (PANEL_HEADER + canvas.shape[0], width, 3),
        PANEL_BACKGROUND_RGB,
        dtype=np.uint8,
    )
    _put_text(panel, title, (8, 27), scale=0.50)
    offset = (width - canvas.shape[1]) // 2
    panel[PANEL_HEADER:, offset : offset + canvas.shape[1]] = canvas
    return panel


def _finite_metric(metrics: Mapping[str, Any], key: str) -> float:
    try:
        value = float(metrics[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise E20VisualAuditError(f"missing evaluation-only metric {key}") from exc
    if not math.isfinite(value):
        raise E20VisualAuditError(f"evaluation-only metric {key} is not finite")
    return value


def _sheet(
    raw: np.ndarray,
    preview: np.ndarray,
    *,
    image: int,
    validation_name: str,
    tiles: int,
    bbox_height: int,
    bbox_width: int,
    metrics: Mapping[str, Any],
) -> np.ndarray:
    raw_panel = _panel(raw, "RAW distorted upright tiles | exact sparse relative entries")
    preview_panel = _panel(
        preview,
        "PREVIEW ONLY | tile-wise NLM h=10 | blanks unchanged",
    )
    height = max(raw_panel.shape[0], preview_panel.shape[0])
    width = raw_panel.shape[1] + PANEL_GAP + preview_panel.shape[1]
    body = np.full((height, width, 3), PANEL_BACKGROUND_RGB, dtype=np.uint8)
    body[: raw_panel.shape[0], : raw_panel.shape[1]] = raw_panel
    body[
        : preview_panel.shape[0],
        raw_panel.shape[1] + PANEL_GAP :,
    ] = preview_panel
    header = np.full((GLOBAL_HEADER, width, 3), PANEL_BACKGROUND_RGB, dtype=np.uint8)
    _put_text(
        header,
        (
            f"E20 sparse RELATIVE cluster | image={image} {validation_name} | "
            f"tiles={tiles} bbox={bbox_height}x{bbox_width} | NO ABSOLUTE ORIGIN"
        ),
        (8, 20),
        scale=0.48,
    )
    _put_text(
        header,
        (
            "EVALUATION-ONLY labels: "
            f"pose_precision={_finite_metric(metrics, 'exact_relative_pose_precision'):.3f} "
            f"pose_coverage={_finite_metric(metrics, 'exact_pose_coverage'):.3f} "
            f"relation_precision={_finite_metric(metrics, 'accepted_relation_precision'):.3f} "
            f"seam_precision={_finite_metric(metrics, 'accepted_cross_seam_precision'):.3f}"
        ),
        (8, 43),
        scale=0.43,
        colour=MUTED_TEXT_RGB,
    )
    _put_text(
        header,
        (
            "Placement source: core.selected.relative_entries ONLY; upright/no rotation; "
            "neutral cells are unfilled (no board completion)."
        ),
        (8, 66),
        scale=0.43,
        colour=MUTED_TEXT_RGB,
    )
    return np.vstack((header, body))


def _atomic_write_png(path: Path, rgb: np.ndarray) -> None:
    resolved = _require_e_drive(path, label="E20 visual-audit PNG")
    temporary = resolved.with_name(f".{resolved.stem}.{os.getpid()}.tmp.png")
    try:
        ok = cv2.imwrite(
            str(temporary),
            cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_PNG_COMPRESSION, 9],
        )
        if not ok:
            raise E20VisualAuditError(f"could not write temporary PNG {temporary}")
        os.replace(temporary, resolved)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    resolved = _require_e_drive(path, label="E20 visual-audit manifest")
    encoded = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    temporary = resolved.with_name(f".{resolved.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, resolved)
    finally:
        if temporary.exists():
            temporary.unlink()


def render(report_path: Path, out_dir: Path) -> Mapping[str, Any]:
    report, report_digest = _verify_complete_report_read_only(report_path)
    output_dir = _require_e_drive(out_dir, label="E20 visual-audit output")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        _e12_report, _calibration, scenes = e17._load_verified_structure_inputs(
            e20.DEFAULT_RAW_CACHE_DIR,
            e20.DEFAULT_CALIBRATION_REPORT.resolve(),
            e20.DEFAULT_E12_REPORT,
        )
    except Exception as exc:
        raise E20VisualAuditError(f"could not replay verified E12 scenes: {exc}") from exc
    rows = _rows_by_image(report)
    scene_by_image = {int(scene.image_id): scene for scene in scenes}
    if tuple(sorted(scene_by_image)) != e12.CALIBRATION_IDS:
        raise E20VisualAuditError("verified E12 scenes are not exactly images 10..17")

    panels: list[dict[str, Any]] = []
    for image in e12.CALIBRATION_IDS:
        scene = scene_by_image[image]
        row = rows[image]
        if row.get("validation_name") != str(scene.validation_name):
            raise E20VisualAuditError(f"scene/report name drifted for image {image}")
        entries = _selected_entries(row)
        raw = _sparse_canvas(scene.tiles_uint8, entries)
        preview = _sparse_canvas(
            scene.tiles_uint8,
            entries,
            transform=e12.fixed_nlm,
        )
        core = row["core"]
        selected = core["selected"]
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping):
            raise E20VisualAuditError(f"E20 row metrics are missing for image {image}")
        bbox_height = _strict_int(selected.get("bbox_height"), label="bbox height")
        bbox_width = _strict_int(selected.get("bbox_width"), label="bbox width")
        sheet = _sheet(
            raw,
            preview,
            image=image,
            validation_name=str(scene.validation_name),
            tiles=len(entries),
            bbox_height=bbox_height,
            bbox_width=bbox_width,
            metrics=metrics,
        )
        output = output_dir / f"image_{image:04d}_e20_sparse_raw_nlm.png"
        _atomic_write_png(output, sheet)
        record = {
            "image": image,
            "validation_name": str(scene.validation_name),
            "panel": str(output.resolve()),
            "panel_sha256": _sha256_file(output),
            "relative_entries_sha256": e12.canonical_digest(entries),
            "selected_tiles": len(entries),
            "bbox_height": bbox_height,
            "bbox_width": bbox_width,
            "placement_source": "core.selected.relative_entries_only",
            "absolute_origin_inferred": False,
            "missing_board_cells_inferred": False,
            "rotation": False,
            "preview": "tilewise_fixed_NLM_h10_evaluation_preview_only",
            "evaluation_only_metrics": {
                "exact_relative_pose_precision": _finite_metric(
                    metrics, "exact_relative_pose_precision"
                ),
                "exact_pose_coverage": _finite_metric(metrics, "exact_pose_coverage"),
                "accepted_relation_precision": _finite_metric(
                    metrics, "accepted_relation_precision"
                ),
                "accepted_cross_seam_precision": _finite_metric(
                    metrics, "accepted_cross_seam_precision"
                ),
            },
        }
        panels.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    manifest: dict[str, Any] = {
        "schema": "pazzle-e20-sparse-visual-audit-v1",
        "source_report": str(Path(report_path).resolve()),
        "source_report_sha256": report_digest,
        "source_report_status": str(report["status"]),
        "source_report_stage": str(report["stage"]),
        "source_report_passed": bool(report["decision"]["passed"]),
        "placement_source": "rows[].core.selected.relative_entries_only",
        "truth_target_permutation_used_for_placement": False,
        "absolute_origin_inferred": False,
        "missing_board_cells_inferred": False,
        "rotation": False,
        "panels": panels,
    }
    manifest_path = output_dir / "manifest.json"
    _atomic_write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path.resolve()),
                "manifest_sha256": _sha256_file(manifest_path),
                "panels": len(panels),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    render(args.report, args.out_dir)


if __name__ == "__main__":
    main()

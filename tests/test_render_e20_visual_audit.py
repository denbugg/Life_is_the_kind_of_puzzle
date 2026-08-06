from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import render_e20_visual_audit as render  # noqa: E402


def _selected(
    entries: list[list[object]] | None = None,
) -> dict[str, object]:
    if entries is None:
        entries = [[0, 0, 1], [1, 1, 0]]
    return {
        "relative_entries": entries,
        "rigid_tiles": len(entries),
        "bbox_height": max(int(value[1]) for value in entries) + 1,
        "bbox_width": max(int(value[2]) for value in entries) + 1,
    }


def _row(entries: list[list[object]] | None = None) -> dict[str, object]:
    return {
        "image": 10,
        "validation_name": "img_006710.png",
        "core": {"selected": _selected(entries)},
        "metrics": {
            "exact_relative_pose_precision": 0.25,
            "exact_pose_coverage": 0.05,
            "accepted_relation_precision": 0.20,
            "accepted_cross_seam_precision": 0.30,
        },
    }


class SparseLayoutTests(unittest.TestCase):
    def test_entries_are_taken_exactly_from_selected_sparse_payload(self) -> None:
        self.assertEqual(render._selected_entries(_row()), ((0, 0, 1), (1, 1, 0)))

    def test_entries_fail_closed_on_noninteger_duplicate_and_bbox_drift(self) -> None:
        cases = (
            [[False, 0, 0]],
            [[0.5, 0, 0]],
            [[0, 0, 0], [0, 0, 1]],
            [[0, 0, 0], [1, 0, 0]],
            [[0, 1, 0]],
            [[0, 0, 24]],
        )
        for entries in cases:
            with self.subTest(entries=entries), self.assertRaises(
                render.E20VisualAuditError
            ):
                render._selected_entries(_row(entries))

        row = _row()
        row["core"]["selected"]["bbox_width"] = 3
        with self.assertRaises(render.E20VisualAuditError):
            render._selected_entries(row)

    def test_raw_sparse_canvas_preserves_upright_tile_bytes_and_neutral_holes(self) -> None:
        tiles = np.zeros((576, 20, 20, 3), dtype=np.uint8)
        tile_zero = np.arange(20 * 20 * 3, dtype=np.uint16).reshape(20, 20, 3)
        tiles[0] = (tile_zero % 251).astype(np.uint8)
        tiles[1] = 173
        entries = ((0, 0, 1), (1, 1, 0))
        canvas = render._sparse_canvas(tiles, entries)
        self.assertEqual(canvas.shape, (40, 40, 3))
        np.testing.assert_array_equal(canvas[0:20, 20:40], tiles[0])
        np.testing.assert_array_equal(canvas[20:40, 0:20], tiles[1])
        np.testing.assert_array_equal(
            canvas[0:20, 0:20],
            np.full((20, 20, 3), render.NEUTRAL_RGB, dtype=np.uint8),
        )
        self.assertFalse(np.array_equal(canvas[0:20, 20:40], np.rot90(tiles[0])))

    def test_preview_transform_runs_per_selected_tile_without_filling_holes(self) -> None:
        tiles = np.zeros((576, 20, 20, 3), dtype=np.uint8)
        tiles[0] = 10
        tiles[1] = 20
        calls: list[int] = []

        def preview(tile: np.ndarray) -> np.ndarray:
            calls.append(int(tile[0, 0, 0]))
            return np.full_like(tile, 200 + len(calls))

        canvas = render._sparse_canvas(
            tiles,
            ((0, 0, 1), (1, 1, 0)),
            transform=preview,
        )
        self.assertEqual(calls, [10, 20])
        self.assertTrue(np.all(canvas[0:20, 20:40] == 201))
        self.assertTrue(np.all(canvas[20:40, 0:20] == 202))
        self.assertTrue(np.all(canvas[0:20, 0:20] == render.NEUTRAL_RGB))

    def test_sheet_captions_are_outside_unchanged_sparse_canvases(self) -> None:
        raw = np.full((40, 60, 3), 11, dtype=np.uint8)
        preview = np.full_like(raw, 22)
        sheet = render._sheet(
            raw,
            preview,
            image=10,
            validation_name="img_006710.png",
            tiles=2,
            bbox_height=2,
            bbox_width=3,
            metrics=_row()["metrics"],
        )
        first_offset = (render.PANEL_MIN_WIDTH - raw.shape[1]) // 2
        y0 = render.GLOBAL_HEADER + render.PANEL_HEADER
        np.testing.assert_array_equal(
            sheet[y0 : y0 + raw.shape[0], first_offset : first_offset + raw.shape[1]],
            raw,
        )


class ContractTests(unittest.TestCase):
    def test_e_drive_guard_rejects_c(self) -> None:
        with self.assertRaises(render.E20VisualAuditError):
            render._require_e_drive(Path("C:/tmp/e20.png"), label="test")

    def test_incomplete_report_fails_before_gate_can_resume_or_write(self) -> None:
        with mock.patch.object(Path, "is_file", return_value=True), mock.patch.object(
            render, "_sha256_file", return_value="a" * 64
        ), mock.patch.object(
            render.e20,
            "_load_json",
            return_value={"status": "in_progress", "completed_images": []},
        ), mock.patch.object(render.e20, "run_gate") as gate:
            with self.assertRaisesRegex(
                render.E20VisualAuditError, "already-complete"
            ):
                render._verify_complete_report_read_only(
                    Path("E:/pazzle_work/e20_incomplete.json")
                )
        gate.assert_not_called()

    def test_report_rows_require_exact_unique_calibration_ids(self) -> None:
        rows = [
            {"image": image, "core": {"selected": {}}, "metrics": {}}
            for image in render.e12.CALIBRATION_IDS
        ]
        mapping = render._rows_by_image({"rows": rows})
        self.assertEqual(tuple(sorted(mapping)), render.e12.CALIBRATION_IDS)
        duplicated = {"rows": [*rows[:-1], dict(rows[0])]}
        with self.assertRaises(render.E20VisualAuditError):
            render._rows_by_image(duplicated)

    def test_render_layout_path_never_accepts_target_or_permutation(self) -> None:
        parameters = set(
            __import__("inspect").signature(render._sparse_canvas).parameters
        )
        self.assertEqual(parameters, {"tiles", "entries", "transform"})
        self.assertNotIn("target", parameters)
        self.assertNotIn("permutation", parameters)


if __name__ == "__main__":
    unittest.main()

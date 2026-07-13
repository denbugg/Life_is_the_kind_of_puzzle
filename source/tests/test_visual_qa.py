from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from puzzle_denoise_v2.visual_qa import (
    _checkpoint_provenance,
    _validate_selected_pixel_file_identity,
    render_visual_qa_contact_sheet,
    select_visual_qa_sources,
    selection_sha256,
    validate_calibration_only_selection,
    write_visual_qa_outputs,
)


def _rows(count: int) -> list[dict]:
    return [
        {
            "source_index": index,
            "source_name": f"img_{index:06d}.png",
            "input_slot": 10 + index,
            "clean_tile_index": 20 + index,
            "confidence": 1.5 + index / 10,
            "pair_row": 100 + index,
        }
        for index in range(count)
    ]


def test_visual_source_selection_is_name_stable_and_calibration_only() -> None:
    names = tuple(f"img_{index:06d}.png" for index in range(12))
    calibration = np.asarray([1, 2, 4, 5, 7, 9, 11], dtype=np.int64)
    selected_a = select_visual_qa_sources(
        names,
        calibration,
        pair_count=4,
        seed=20260710,
    )
    selected_b = select_visual_qa_sources(
        names,
        calibration,
        pair_count=4,
        seed=20260710,
    )
    assert np.array_equal(selected_a, selected_b)
    assert len(np.unique(selected_a)) == 4
    assert set(selected_a.tolist()) <= set(calibration.tolist())

    reversed_names = tuple(reversed(names))
    reversed_calibration = np.asarray(
        [reversed_names.index(names[int(index)]) for index in calibration],
        dtype=np.int64,
    )
    selected_reversed = select_visual_qa_sources(
        reversed_names,
        reversed_calibration,
        pair_count=4,
        seed=20260710,
    )
    assert [names[int(index)] for index in selected_a] == [
        reversed_names[int(index)] for index in selected_reversed
    ]

    validate_calibration_only_selection(
        selected_a,
        calibration,
        frozen_gate_source_indices=[0, 3, 6],
        quarantine_source_indices=[8, 10],
    )
    with pytest.raises(ValueError, match="non-calibration"):
        validate_calibration_only_selection(
            [int(selected_a[0]), 3],
            calibration,
            frozen_gate_source_indices=[0, 3, 6],
            quarantine_source_indices=[8, 10],
        )


def test_contact_sheet_and_provenance_are_deterministic_temp_artifacts(
    tmp_path: Path,
) -> None:
    corrupt = np.zeros((2, 20, 20, 3), dtype=np.uint8)
    corrupt[1, ..., 0] = 40
    restored = np.full_like(corrupt, 128)
    clean = np.full_like(corrupt, 255)
    rows = _rows(2)
    checkpoint_sha256 = "a" * 64

    first = render_visual_qa_contact_sheet(
        corrupt,
        restored,
        clean,
        rows,
        checkpoint_sha256=checkpoint_sha256,
        selection_seed=17,
        tile_scale=3,
    )
    second = render_visual_qa_contact_sheet(
        corrupt,
        restored,
        clean,
        rows,
        checkpoint_sha256=checkpoint_sha256,
        selection_seed=17,
        tile_scale=3,
    )
    assert first.mode == "RGB"
    assert first.size == second.size
    assert first.tobytes() == second.tobytes()

    output_png = tmp_path / "qa" / "contact.png"
    report_json = tmp_path / "qa" / "contact.json"
    report = write_visual_qa_outputs(
        first,
        {
            "schema_version": 1,
            "kind": "unit_test_visual_qa",
            "selection": {"selection_sha256": selection_sha256(rows)},
        },
        output_png=output_png,
        report_json=report_json,
    )
    assert output_png.is_file() and report_json.is_file()
    assert report["outputs"]["contact_sheet_png_sha256"] == hashlib.sha256(
        output_png.read_bytes()
    ).hexdigest()
    assert json.loads(report_json.read_text(encoding="utf-8")) == report
    with Image.open(output_png) as reopened:
        assert reopened.mode == "RGB"
        assert reopened.size == first.size

    with pytest.raises(FileExistsError, match="output exists"):
        write_visual_qa_outputs(
            second,
            report,
            output_png=output_png,
            report_json=report_json,
        )


def test_renderer_rejects_shape_dtype_and_label_mismatches() -> None:
    valid = np.zeros((1, 20, 20, 3), dtype=np.uint8)
    kwargs = {
        "checkpoint_sha256": "b" * 64,
        "selection_seed": 3,
    }
    with pytest.raises(TypeError, match="restored tiles must be uint8"):
        render_visual_qa_contact_sheet(
            valid,
            valid.astype(np.float32),
            valid,
            _rows(1),
            **kwargs,
        )
    with pytest.raises(ValueError, match="rows length"):
        render_visual_qa_contact_sheet(valid, valid, valid, [], **kwargs)

    incomplete = [{"source_name": "img_000000.png"}]
    with pytest.raises(ValueError, match="missing labels"):
        render_visual_qa_contact_sheet(valid, valid, valid, incomplete, **kwargs)


def test_checkpoint_provenance_never_copies_frozen_gate_details() -> None:
    sanitized = _checkpoint_provenance(
        {
            "checkpoint_sha256": "c" * 64,
            "model_name": "tile-naf",
            "promotion_status": "promoted",
            "best_ssim": 0.80828,
            "best_real_ssim": 0.81,
            "source_split": {"gate_source_names": ["do-not-disclose.png"]},
            "gate_validation": {"source_metrics": [0.9]},
            "best_validation": {"panel": "calibration"},
        }
    )
    assert sanitized == {
        "checkpoint_sha256": "c" * 64,
        "model_name": "tile-naf",
        "promotion_status": "promoted",
    }


def test_pixel_identity_guard_rejects_forbidden_aliases_and_symlinks(
    tmp_path: Path,
) -> None:
    selected = tmp_path / "selected.png"
    second = tmp_path / "second.png"
    forbidden = tmp_path / "forbidden.png"
    selected.write_bytes(b"selected")
    second.write_bytes(b"second")
    forbidden.write_bytes(b"forbidden")
    _validate_selected_pixel_file_identity([selected, second], [forbidden])

    alias = tmp_path / "alias.png"
    alias.hardlink_to(forbidden)
    with pytest.raises(ValueError, match="aliases a quarantine/frozen-gate"):
        _validate_selected_pixel_file_identity([alias], [forbidden])

    symlink = tmp_path / "symlink.png"
    symlink.symlink_to(selected)
    with pytest.raises(ValueError, match="must not be a symlink"):
        _validate_selected_pixel_file_identity([symlink], [forbidden])


def test_writer_rejects_hardlinked_output_pair(tmp_path: Path) -> None:
    output_png = tmp_path / "contact.png"
    report_json = tmp_path / "contact.json"
    output_png.write_bytes(b"old")
    report_json.hardlink_to(output_png)
    with pytest.raises(ValueError, match="hard links to the same file"):
        write_visual_qa_outputs(
            Image.new("RGB", (2, 2)),
            {"schema_version": 1},
            output_png=output_png,
            report_json=report_json,
            overwrite=True,
        )

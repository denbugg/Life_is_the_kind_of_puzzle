from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import build_harmonized_submission as harmonized


REPO_ROOT = Path(__file__).resolve().parents[1]
LAYOUT_REPORTS = [
    REPO_ROOT
    / "runs/assembly_v1/kaggle/final_qap_submission_output/v1/final_qap_shard_000_350.json",
    REPO_ROOT
    / "runs/assembly_v1/kaggle/final_qap_submission_output/v1/final_qap_shard_350_700.json",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_loads_exact_promoted_700_layouts() -> None:
    layouts, reports = harmonized.load_frozen_layouts(
        LAYOUT_REPORTS, expected_count=700
    )
    assert len(layouts) == 700
    assert len(reports) == 2
    assert {record["sha256"] for record in reports} == harmonized.EXPECTED_LAYOUT_REPORT_SHA256
    assert set(layouts) == {
        path.name for path in (REPO_ROOT / "puzzle/test").glob("*.png")
    }
    assert all(
        np.array_equal(
            np.sort(record["layout"]),
            np.arange(harmonized.TILE_COUNT, dtype=np.int32),
        )
        for record in layouts.values()
    )


def test_rejects_layout_tamper_even_if_outer_file_hash_is_rebound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = json.loads(LAYOUT_REPORTS[0].read_text(encoding="utf-8"))
    first["sources"][0]["position_to_slot"][0], first["sources"][0][
        "position_to_slot"
    ][1] = (
        first["sources"][0]["position_to_slot"][1],
        first["sources"][0]["position_to_slot"][0],
    )
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(first, sort_keys=True), encoding="utf-8")
    monkeypatch.setattr(
        harmonized,
        "EXPECTED_LAYOUT_REPORT_SHA256",
        {_sha256(tampered), _sha256(LAYOUT_REPORTS[1])},
    )
    with pytest.raises(RuntimeError, match="layout hash mismatch"):
        harmonized.load_frozen_layouts(
            [tampered, LAYOUT_REPORTS[1]], expected_count=700
        )


def test_rejects_layout_configuration_tamper_before_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = json.loads(LAYOUT_REPORTS[0].read_text(encoding="utf-8"))
    first["configuration"]["qap"]["boundary_weight"] = 0.0
    tampered = tmp_path / "tampered-config.json"
    tampered.write_text(json.dumps(first, sort_keys=True), encoding="utf-8")
    monkeypatch.setattr(
        harmonized,
        "EXPECTED_LAYOUT_REPORT_SHA256",
        {_sha256(tampered), _sha256(LAYOUT_REPORTS[1])},
    )
    with pytest.raises(RuntimeError, match="qap configuration drift"):
        harmonized.load_frozen_layouts(
            [tampered, LAYOUT_REPORTS[1]], expected_count=700
        )


def test_render_is_deterministic_and_layout_sensitive() -> None:
    grid_y, grid_x = np.mgrid[:20, :20]
    base = np.empty((harmonized.TILE_COUNT, 20, 20, 3), dtype=np.uint8)
    for index in range(harmonized.TILE_COUNT):
        row, column = divmod(index, 24)
        base[index, :, :, 0] = np.clip(20 + 5 * column + grid_x, 0, 255)
        base[index, :, :, 1] = np.clip(15 + 5 * row + grid_y, 0, 255)
        base[index, :, :, 2] = np.clip(40 + 2 * row + 2 * column, 0, 255)
    seam = np.clip(
        base.astype(np.int16)
        + ((np.arange(harmonized.TILE_COUNT) % 7) - 3)[:, None, None, None],
        0,
        255,
    ).astype(np.uint8)
    identity = np.arange(harmonized.TILE_COUNT, dtype=np.int32)
    first, first_diag = harmonized.render_harmonized_tiles(base, seam, identity)
    second, second_diag = harmonized.render_harmonized_tiles(base, seam, identity)
    assert np.array_equal(first, second)
    assert first_diag == second_diag
    assert first.shape == (480, 480, 3)
    assert first.dtype == np.uint8
    assert first_diag["placebo"] is False
    assert first_diag["edge_count"] == 1104.0

    swapped = identity.copy()
    swapped[0], swapped[-1] = swapped[-1], swapped[0]
    different, _ = harmonized.render_harmonized_tiles(base, seam, swapped)
    assert not np.array_equal(first, different)


def test_invalid_layout_is_rejected() -> None:
    duplicate = np.zeros(harmonized.TILE_COUNT, dtype=np.int32)
    with pytest.raises(RuntimeError, match="not a permutation"):
        harmonized._validate_layout(duplicate)


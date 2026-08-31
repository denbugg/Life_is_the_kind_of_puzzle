from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_direct_rank_delta_component_selector_opened64 import (
    _component_evidence,
    _report_path,
    _win_tie_loss,
)


def test_component_evidence_uses_consistent_count_and_largest_partition() -> None:
    evidence = _component_evidence(
        {"consistent": 7, "added": 3},
        ({0: (0, 0), 1: (0, 1), 2: (1, 0)}, {3: (0, 0)}),
        tile_count=4,
    )
    assert evidence.lexicographic_key == (7, 3)


def test_component_evidence_requires_complete_partition() -> None:
    with pytest.raises(ValueError, match="partition"):
        _component_evidence(
            {"consistent": 0},
            ({0: (0, 0)},),
            tile_count=2,
        )


def test_win_tie_loss_is_board_level() -> None:
    assert _win_tie_loss([2.0, 0.0, -1.0, 0.5]) == {
        "wins": 2,
        "ties": 1,
        "losses": 1,
    }


def test_report_path_supports_project_and_external_smoke_outputs(tmp_path) -> None:
    project_path = _report_path(Path("outputs/component-selector/test.json"))
    external_path = _report_path(tmp_path / "test.json")
    assert project_path == "outputs/component-selector/test.json"
    assert external_path == str((tmp_path / "test.json").resolve())

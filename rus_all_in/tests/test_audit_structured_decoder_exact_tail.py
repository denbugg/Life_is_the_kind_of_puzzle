from __future__ import annotations

import pytest

from scripts.audit_structured_decoder_exact_tail import compute_exact_tail


def _row(index: int, control: int, delta: int) -> dict:
    return {
        "prefix": f"case_{index:04d}",
        "case_id": f"case-{index}",
        "control": {"exact_tiles": control},
        "ceiling": {"exact_tiles": control + delta},
        "delta": {"exact_tiles": delta},
    }


def test_exact_tail_metrics_and_positive_concentration() -> None:
    rows = (
        _row(0, 1, -1),
        _row(1, 0, 0),
        _row(2, 1, 2),
        _row(3, 2, 6),
    )
    result = compute_exact_tail(rows)
    assert result["exact_delta"] == {
        "mean": pytest.approx(1.75),
        "median": pytest.approx(1.0),
        "q25": pytest.approx(-0.25),
        "q75": pytest.approx(3.0),
        "quantile_method": "linear",
    }
    assert result["win_tie_loss"]["win_count"] == 2
    assert result["win_tie_loss"]["tie_count"] == 1
    assert result["win_tie_loss"]["loss_count"] == 1
    assert result["absolute_exact"]["control"]["zero_count"] == 1
    assert result["absolute_exact"]["pair_safe_ceiling"]["at_most_one_count"] == 2
    concentration = result["positive_concentration"]
    assert concentration["largest_positive_share"] == pytest.approx(0.75)
    assert concentration["leave_largest_positive_mean_exact_delta"] == pytest.approx(
        1 / 3
    )
    assert concentration["removed_largest_positive"]["prefix"] == "case_0003"


def test_exact_tail_rejects_inconsistent_absolute_counts() -> None:
    row = _row(0, 1, 2)
    row["ceiling"]["exact_tiles"] = 2
    with pytest.raises(RuntimeError, match="do not match"):
        compute_exact_tail((row,))

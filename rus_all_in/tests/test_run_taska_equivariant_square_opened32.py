from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"


def _load_runner() -> ModuleType:
    sys.path.insert(0, str(SCRIPTS))
    path = SCRIPTS / "run_taska_equivariant_square_opened32.py"
    specification = importlib.util.spec_from_file_location(
        "run_taska_equivariant_square_opened32_test",
        path,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


runner = _load_runner()


def test_fixed_arm_is_the_isolated_legal_square_recipe() -> None:
    assert runner.SQUARE_SUPPORT_K == 16
    assert runner.SQUARE_TEMPERATURE == 0.5
    assert runner.SQUARE_ROUNDS == 1
    assert runner.SQUARE_SHORTLIST == 20
    assert runner.SQUARE_WEIGHT == 0.4
    assert runner.VIEWS == ("raw", "median", "bilateral")
    assert runner.ORIENTATIONS == 2
    assert runner.VOTE_TARGET == 350
    assert runner.base.SOLVER_CONFIG.border_weight == 0.0


def test_mutual_harvest_is_permutation_equivariant_even_with_ties() -> None:
    rng = np.random.default_rng(7)
    scores = np.round(rng.normal(size=(24, 24)), 0)
    permutation = rng.permutation(len(scores))
    original = runner._mutual_edges(scores, "right")
    relabelled = runner._mutual_edges(
        scores[np.ix_(permutation, permutation)],
        "right",
    )
    old_to_new = np.empty(len(permutation), dtype=np.int64)
    old_to_new[permutation] = np.arange(len(permutation))
    expected = {
        RawTailEdge(
            int(old_to_new[edge.source]),
            int(old_to_new[edge.target]),
            edge.axis,
        ): margin
        for edge, margin in original.items()
    }
    assert relabelled.keys() == expected.keys()
    for edge in expected:
        assert relabelled[edge] == pytest.approx(expected[edge])


def test_dynamic_vote_threshold_reaches_target_or_one() -> None:
    common = [RawTailEdge(index, index + 1, "right") for index in range(5)]
    scorers = [{edge: 1.0 for edge in common} for _ in range(12)]
    assert runner._vote_threshold(scorers, target=5, fallback=10) == 12
    assert runner._vote_threshold(scorers, target=6, fallback=10) == 1
    assert runner._vote_threshold(scorers, target=0, fallback=10) == 10


def test_square_is_applied_to_all_scorers_and_fused_raw_before_consumption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import aiijc_puzzle.taska_seam_matcher as seam

    count = runner.base.COUNT
    calibrated = np.full((count, count), -3.0, dtype=np.float64)
    pessimistic = np.full((count, count), -7.0, dtype=np.float64)
    square_inputs: list[tuple[float, float]] = []
    consumed: list[float] = []

    monkeypatch.setattr(seam, "analytic_view", lambda _name, tiles: tiles)
    monkeypatch.setattr(
        seam,
        "calibrated_log_assignments",
        lambda *_args, **_kwargs: (calibrated.copy(), (calibrated - 1).copy()),
    )
    monkeypatch.setattr(
        seam,
        "pessimistic_log_assignments",
        lambda *_args, **_kwargs: (pessimistic.copy(), (pessimistic - 1).copy()),
    )

    def square(
        right: np.ndarray,
        down: np.ndarray,
        *,
        weight: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert weight == 0.4
        square_inputs.append((float(right[0, 1]), float(down[0, 1])))
        return right + 100.0, down + 100.0

    def consume(matrix: np.ndarray, _axis: str) -> dict[RawTailEdge, float]:
        consumed.append(float(matrix[0, 1]))
        return {}

    monkeypatch.setattr(runner, "equivariant_square_rerank", square)
    monkeypatch.setattr(runner, "_mutual_edges", consume)
    result = runner._square_match(
        np.zeros((count, 20, 20, 3), dtype=np.uint8),
        (object(), object()),
        device=torch.device("cpu"),
    )
    # 2 models x 3 views x 2 orientations, plus one pessimistic fused pair.
    assert len(square_inputs) == 13
    assert square_inputs[:12] == [(-3.0, -4.0)] * 12
    assert square_inputs[-1] == (-7.0, -8.0)
    assert consumed == [97.0, 96.0] * 12
    assert result.right_log[0, 1] == 93.0
    assert result.down_log[0, 1] == 92.0
    assert result.scorer_count == 12


def test_scoring_validates_frozen_roster_before_opening_targets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    opened_target = False

    def reject_roster(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("frozen roster changed")

    def open_target(_path: Path) -> None:
        nonlocal opened_target
        opened_target = True
        raise AssertionError("target must stay closed")

    monkeypatch.setattr(runner, "_validate_roster", reject_roster)
    monkeypatch.setattr(runner.base, "CleanTileCache", open_target)
    paths = runner.RunPaths(
        frozen_eval=tmp_path / "frozen.npz",
        frozen_eval_metadata=tmp_path / "frozen.json",
        pre_score_freeze=tmp_path / "freeze.json",
        report=tmp_path / "report.json",
    )
    with pytest.raises(RuntimeError, match="frozen roster"):
        runner._score_frozen(
            paths,
            object(),
            {},
            {},
            targets=tmp_path / "targets",
        )
    assert opened_target is False


def test_non_smoke_full_run_is_blocked_before_touching_config(tmp_path: Path) -> None:
    arguments = runner.parse_args(
        [
            "--config",
            str(tmp_path / "absent.json"),
            "--output-dir",
            str(tmp_path / "output"),
            "--device",
            "cpu",
        ]
    )
    with pytest.raises(ValueError, match="blocked"):
        runner.run(arguments)

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"


def _load_runner() -> ModuleType:
    sys.path.insert(0, str(SCRIPTS))
    path = SCRIPTS / "run_direct_hard_edge_fresh64.py"
    specification = importlib.util.spec_from_file_location(
        "run_direct_hard_edge_fresh64_test",
        path,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


runner = _load_runner()


def test_source_clustered_ci_is_deterministic_and_contains_constant() -> None:
    first = runner.source_clustered_ci([2.0] * 64, seed=17, resamples=1000)
    second = runner.source_clustered_ci([2.0] * 64, seed=17, resamples=1000)
    assert first == second
    assert first["mean"] == 2.0
    assert first["ci95_lower"] == 2.0
    assert first["ci95_upper"] == 2.0


def _metrics(correct: float, adjacency: float, aligned: float, exact: float = 0.0) -> dict:
    row = lambda value: {"mean": value, "ci95_lower": value - 0.1}  # noqa: E731
    return {
        "hard_edge_correct_gain": row(correct),
        "adjacency_delta": row(adjacency),
        "translation_aligned_tiles_delta": row(aligned),
        "exact_tiles_delta": row(exact),
    }


def test_confirmation_gate_requires_both_primary_and_secondary_safety() -> None:
    passed = runner.evaluate_confirmation_gate(_metrics(1.0, 0.0001, 0.0))
    assert passed["pass"]
    assert not passed["promotion_authorized"]
    assert not runner.evaluate_confirmation_gate(_metrics(0.99, 0.1, 1.0))["pass"]
    assert not runner.evaluate_confirmation_gate(_metrics(2.0, 0.0, 1.0))["pass"]
    assert not runner.evaluate_confirmation_gate(_metrics(2.0, 0.1, -0.01))["pass"]


def test_fresh64_commitment_is_hash_locked_before_access() -> None:
    config, digest = runner.load_confirmation_config(
        PROJECT_ROOT / "configs/direct_hard_edge_fresh64_confirmation_v1.json"
    )
    assert digest == "6056fcc57898935c59d9575e7fa2371f9b003fb4843decb115d69e41f0e1735e"
    assert len(config["selection"]["source_filenames"]) == 64
    assert config["selection"]["selected_exclusion_overlap"] == []
    assert not config["legality"]["competition_test_opened"]

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
CONFIG_PATH = PROJECT_ROOT / "configs/frame_side_origin_preregistered_v1.json"


def _load_runner() -> ModuleType:
    sys.path.insert(0, str(SCRIPTS))
    path = SCRIPTS / "run_frame_side_origin.py"
    specification = importlib.util.spec_from_file_location(
        "run_frame_side_origin_test",
        path,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


runner = _load_runner()


def _summary(*, f1_gain: float, exact_gain: float, adjacency_delta: float) -> dict:
    return {
        "frame_socket_v2": {"macro_f1": 0.1},
        "frame_candidate": {"macro_f1": 0.1 + f1_gain},
        "comparator_layout": {"correct_tile_count": 10.0, "adjacency": 0.1},
        "candidate_layout": {
            "correct_tile_count": 10.0 + exact_gain,
            "adjacency": 0.1 + adjacency_delta,
        },
        "strict_permutation_count": 32,
    }


def test_low_d1_gate_preserves_adjacency_and_strict_permutation_safety() -> None:
    config, _ = runner._load_config(CONFIG_PATH)  # noqa: SLF001
    assert runner._gate(  # noqa: SLF001
        _summary(f1_gain=0.02, exact_gain=0.0, adjacency_delta=-0.002),
        config,
    )["pass"]
    assert runner._gate(  # noqa: SLF001
        _summary(f1_gain=0.0, exact_gain=0.1, adjacency_delta=-0.002),
        config,
    )["pass"]
    assert not runner._gate(  # noqa: SLF001
        _summary(f1_gain=0.5, exact_gain=1.0, adjacency_delta=-0.0021),
        config,
    )["pass"]
    unsafe = _summary(f1_gain=0.5, exact_gain=1.0, adjacency_delta=0.0)
    unsafe["strict_permutation_count"] = 31
    assert not runner._gate(unsafe, config)["pass"]  # noqa: SLF001


def test_preregistration_and_source_disjoint_rosters_are_immutable() -> None:
    config, digest = runner._load_config(CONFIG_PATH)  # noqa: SLF001
    manifest = json.loads(
        (PROJECT_ROOT / "data/interim/validation_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    fit, evaluation, selection, selection_digest = runner._load_rosters(  # noqa: SLF001
        config,
        manifest,
    )
    assert digest == "1483ea867782f955e650555e71c6626eb02379cfe0124a7006a2c07becdd1a78"
    assert selection_digest == config["selection_commitment"]["sha256"]
    assert len(fit) == 256
    assert len(evaluation) == 32
    assert not ({row["filename"] for row in fit} & {row["filename"] for row in evaluation})
    assert selection["written_before_any_selected_target_access"]
    assert not selection["holdout_opened"]
    assert not selection["competition_test_opened"]

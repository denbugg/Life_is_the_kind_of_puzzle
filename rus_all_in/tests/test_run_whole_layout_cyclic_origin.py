from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"


def _load_runner() -> ModuleType:
    sys.path.insert(0, str(SCRIPTS))
    path = SCRIPTS / "run_whole_layout_cyclic_origin.py"
    specification = importlib.util.spec_from_file_location(
        "run_whole_layout_cyclic_origin_test",
        path,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


runner = _load_runner()


def _gate_contract() -> dict:
    return {
        "minimum_exact_gain_per_board": 0.1,
        "minimum_r1_gain_over_uniform": 0.05,
        "minimum_r5_gain_over_uniform": 0.10,
        "minimum_nll_gain_over_uniform": 0.10,
        "minimum_adjacency_delta": -0.002,
        "strict_original_permutations_required": 16,
    }


def _summary(*, exact: float, adjacency: float, r1: float = 0.0) -> dict:
    return {
        "delta": {"exact_tiles_per_board": exact, "adjacency": adjacency},
        "roll_diagnostics": {
            "r1_gain_over_uniform": r1,
            "r5_gain_over_uniform": 0.0,
            "nll_gain_over_uniform": 0.0,
        },
        "strict_original_permutations": 16,
    }


def test_low_d1_gate_has_primary_and_auxiliary_paths_but_keeps_safety() -> None:
    primary = runner.evaluate_discovery_gate(
        _summary(exact=0.1, adjacency=-0.002),
        _gate_contract(),
    )
    assert primary["pass"]
    auxiliary = runner.evaluate_discovery_gate(
        _summary(exact=0.0, adjacency=-0.002, r1=0.05),
        _gate_contract(),
    )
    assert auxiliary["pass"]
    assert not auxiliary["promotion_authorized"]
    assert not auxiliary["competition_test_authorized"]
    assert not runner.evaluate_discovery_gate(
        _summary(exact=-0.01, adjacency=0.0, r1=0.5),
        _gate_contract(),
    )["pass"]
    assert not runner.evaluate_discovery_gate(
        _summary(exact=1.0, adjacency=-0.0021),
        _gate_contract(),
    )["pass"]


def test_curriculum_roll_updates_the_exact_count_target_consistently() -> None:
    grid = runner.GRID
    count = grid * grid
    features = np.arange(count * 2, dtype=np.float32).reshape(count, 2)
    reference = np.arange(count, dtype=np.int32)
    decoder = np.roll(reference.reshape(grid, grid), (3, 5), (0, 1)).reshape(-1)
    case = runner.FrozenFeatureCase(
        source_filename="img_000001.png",
        case_id="case",
        tile_features=features,
        reference_layout=reference,
        decoder_layout=decoder,
        decoder_counts=runner.cyclic_exact_counts(decoder, reference, grid=grid),
    )
    feature_grid, counts = runner._rolled_training_example(  # noqa: SLF001
        case,
        exact_stage=False,
        row_roll=7,
        column_roll=11,
    )
    rolled_layout = np.roll(decoder.reshape(grid, grid), (7, 11), (0, 1)).reshape(-1)
    observed = runner.cyclic_exact_counts(rolled_layout, reference, grid=grid)
    assert np.array_equal(counts, observed)
    assert np.array_equal(
        feature_grid,
        np.roll(
            runner.assemble_feature_grid(features, decoder, grid=grid),
            (7, 11),
            (1, 2),
        ),
    )


def test_recursive_filename_collection_is_fail_closed_for_duplicates() -> None:
    value = {
        "outer": [
            {"fit_filenames": ["a/img_000001.png", "img_000002.png"]},
            {"source_filename": "img_000003.png"},
        ]
    }
    assert runner.collect_declared_filenames(value) == {
        "img_000001.png",
        "img_000002.png",
        "img_000003.png",
    }
    with pytest.raises(ValueError, match="duplicate"):
        runner.collect_declared_filenames(
            {"bad_filenames": ["img_000001.png", "img_000001.png"]}
        )


def test_preregistration_and_selection_are_frozen_and_train_only() -> None:
    config, digest = runner.load_frozen_config(
        PROJECT_ROOT / "configs/whole_layout_cyclic_origin_preregistered_v1.json"
    )
    selection, selection_digest = runner.load_selection_commitment(config)
    manifest = json.loads(
        (PROJECT_ROOT / "data/interim/validation_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    fit, evaluation = runner.validate_rosters(selection, manifest)
    assert digest == "51f962a4b6ca18cc98ab1255d604a0e78c9947409828ffe06f193df9d5e4e1a1"
    assert selection_digest == config["selection_commitment"]["sha256"]
    assert len(fit) == 256
    assert len(evaluation) == 16
    fit_names = {record["filename"] for record in fit}
    evaluation_names = {record["filename"] for record in evaluation}
    assert not (fit_names & evaluation_names)
    assert not config["evaluation"]["holdout_opened"]
    assert not config["evaluation"]["competition_test_opened"]

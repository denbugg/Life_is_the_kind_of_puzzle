from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from aiijc_puzzle.protocol import sha256_file
from scripts.run_fullres_twin_side_matcher import (
    DEFAULT_CONFIG,
    _json_panel_names,
    _load_config,
    _procedural_capacity_tiles,
    _two_view_case,
)


def test_frozen_config_and_sidecar_match() -> None:
    config = _load_config(DEFAULT_CONFIG)
    assert config["selection"]["fit_sources"] == 256
    assert config["selection"]["evaluation_sources"] == 24
    assert config["training"]["maximum_updates"] == 600
    assert config["architecture"]["field"].startswith("20x20x48")
    sidecar = Path(f"{DEFAULT_CONFIG}.sha256").read_text(encoding="utf-8").split()[0]
    assert sidecar == sha256_file(DEFAULT_CONFIG)


def test_panel_registry_excludes_eval_but_not_historical_fit() -> None:
    payload = {
        "selection": {
            "fit_filenames": ["fit.png"],
            "train_filenames": ["train.png"],
            "evaluation_filenames": ["eval.png"],
            "source_filenames": ["decoder-source.png"],
        },
        "training_history": [{"source_filename": "history-fit.png"}],
        "cases": [{"source_filename": "frozen-eval.png"}],
    }
    assert _json_panel_names(payload) == {
        "eval.png",
        "decoder-source.png",
        "frozen-eval.png",
    }


def test_two_corruptions_share_shuffle_but_are_independent_and_deterministic() -> None:
    clean = _procedural_capacity_tiles()
    first, second, layout = _two_view_case(
        clean,
        first_seed=11,
        second_seed=12,
        permutation_seed=13,
    )
    repeat = _two_view_case(
        clean,
        first_seed=11,
        second_seed=12,
        permutation_seed=13,
    )
    assert first.shape == second.shape == (16, 20, 20, 3)
    assert np.array_equal(np.sort(layout), np.arange(16))
    assert not np.array_equal(first, second)
    assert all(
        np.array_equal(value, other)
        for value, other in zip((first, second, layout), repeat, strict=True)
    )


def test_config_is_valid_json_without_decoder_or_rgb_objective() -> None:
    payload = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    assert payload["architecture"]["decoder"] is False
    assert payload["architecture"]["pixel_prediction_head"] is False
    assert payload["legality"]["target_available_at_inference"] is False

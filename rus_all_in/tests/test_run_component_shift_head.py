from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_runner() -> ModuleType:
    path = PROJECT_ROOT / "scripts/run_component_shift_head.py"
    specification = importlib.util.spec_from_file_location(
        "run_component_shift_head_test",
        path,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


runner = _load_runner()


def _arguments(**overrides: object) -> Namespace:
    values: dict[str, object] = {
        "train_limit": 32,
        "steps": 100,
        "log_every": 25,
        "learning_rate": 3e-4,
        "weight_decay": 2e-4,
    }
    values.update(overrides)
    return Namespace(**values)


def test_filename_collector_catches_arbitrary_nested_lineage_keys() -> None:
    payload = {
        "selection": {
            "lineage_train_filenames": ["a.png"],
            "lineage_exposed_filenames": ["a.png", "b.png", "c.png"],
        },
        "history": [
            {"future_panel_filenames": ["b.png"]},
            {"another_filenames": ("c.png",)},
        ],
    }
    assert runner.collect_filename_lists(payload) == {"a.png", "b.png", "c.png"}

    with pytest.raises(ValueError, match="duplicate"):
        runner.collect_filename_lists({"bad_filenames": ["a.png", "a.png"]})
    with pytest.raises(ValueError, match="non-empty"):
        runner.collect_filename_lists({"bad_filenames": [""]})


def test_training_selector_excludes_checkpoint_and_extra_report(tmp_path: Path) -> None:
    manifest = {
        "splits": {
            "train": [
                {"filename": f"{letter}.png"}
                for letter in ("a", "b", "c", "d", "e", "f")
            ]
        }
    }
    checkpoint = {
        "selection": {
            "lineage_exposed_filenames": ["a.png"],
            "historical_panel_filenames": ["b.png"],
        }
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps({"nested": {"future_filenames": ["c.png"]}}),
        encoding="utf-8",
    )
    records, forbidden, counts = runner.select_training_records(
        manifest,
        checkpoint,
        [report_path],
        limit=2,
        checkpoint_sha256="frozen-parent",
    )
    selected = {record["filename"] for record in records}
    assert len(selected) == 2
    assert selected.isdisjoint({"a.png", "b.png", "c.png"})
    assert forbidden == {"a.png", "b.png", "c.png"}
    assert counts["absolute_checkpoint"] == 2
    assert counts[str(report_path.resolve())] == 1


def test_predeclared_training_only_gate_requires_support_and_both_axes() -> None:
    metrics = {
        "row_accuracy": 0.20,
        "row_chance_accuracy": 0.10,
        "row_nll_gain_vs_uniform": 0.08,
        "column_accuracy": 0.22,
        "column_chance_accuracy": 0.10,
        "column_nll_gain_vs_uniform": 0.07,
    }
    support = {
        "predicted_supported_tiles_per_board": 20.0,
        "chance_expected_supported_tiles_per_board": 5.0,
        "centre_supported_tiles_per_board": 10.0,
        "dominant_oracle_supported_tiles_per_board": 30.0,
    }
    passed = runner.evaluate_training_only_gate(metrics, support)
    assert passed["pass"]
    assert passed["status"] == "pass-await-root-review"
    assert not passed["quality_panel_authorized"]

    weak_column = metrics | {"column_nll_gain_vs_uniform": 0.019}
    failed = runner.evaluate_training_only_gate(weak_column, support)
    assert not failed["pass"]
    assert failed["status"] == "stop"
    assert not failed["column"]["pass"]


def test_training_metrics_cover_accuracy_nll_bins_and_supported_tiles() -> None:
    components = tuple(
        runner.ComponentDescriptor((tile,), (0,), (0,), 0.0) for tile in range(4)
    )
    positions = np.arange(4)
    targets = runner.dominant_component_shift_targets(components, positions, grid=2)
    row_logits = torch.full((4, 2), -8.0)
    column_logits = torch.full((4, 2), -8.0)
    for index, target in enumerate(targets):
        row_logits[index, target.target_row_shift] = 8.0
        column_logits[index, target.target_column_shift] = 8.0
    output = runner.ComponentShiftOutput(
        row_logits=row_logits,
        column_logits=column_logits,
        feasible_row_shifts=(2, 2, 2, 2),
        feasible_column_shifts=(2, 2, 2, 2),
    )
    observations = runner.component_observations(
        output,
        components,
        targets,
        positions,
        grid=2,
    )
    metrics = runner.aggregate_component_observations(observations)
    support = runner.board_support_summary(observations)
    assert metrics["row_accuracy"] == metrics["column_accuracy"] == 1.0
    assert metrics["joint_accuracy"] == 1.0
    assert metrics["row_chance_accuracy"] == metrics["column_chance_accuracy"] == 0.5
    assert metrics["joint_chance_accuracy"] == 0.25
    assert metrics["row_chance_normalized_nll"] < 1e-5
    assert metrics["column_chance_normalized_nll"] < 1e-5
    assert support == {
        "predicted_supported_tiles": 4.0,
        "chance_expected_supported_tiles": 1.0,
        "centre_supported_tiles": 1.0,
        "dominant_oracle_supported_tiles": 4.0,
    }
    assert set(runner.aggregate_bins(observations, "purity_bin")) == {"pure_1"}
    assert set(runner.aggregate_bins(observations, "size_bin")) == {"singleton_1"}

    with pytest.raises(ValueError, match="exact permutation"):
        runner.component_observations(
            output,
            components,
            targets,
            np.zeros(4),
            grid=2,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("train_limit", 0),
        ("train_limit", 2049),
        ("steps", 0),
        ("steps", 801),
        ("log_every", 0),
        ("learning_rate", 0.0),
        ("weight_decay", -1.0),
    ),
)
def test_runner_enforces_bounded_training_contract(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        runner.validate_args(_arguments(**{field: value}))


def test_runner_fixes_exact_60k_head_contract() -> None:
    head = runner.ComponentShiftHead(
        32,
        grid=runner.GRID,
        hidden_dimension=runner.HEAD_HIDDEN_DIMENSION,
    )
    assert sum(parameter.numel() for parameter in head.parameters()) == 60_208
    assert runner.EXPECTED_HEAD_PARAMETERS == 60_208

from __future__ import annotations

import inspect
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_path_cover_gate as gate
from puzzle_assembly.geometry import GRID, TILE_COUNT, true_neighbour_slots
from puzzle_assembly.path_cover import solve_path_cover
from puzzle_assembly.protocol import source_names_for_split


def test_frozen_source_slice_and_hash_are_exact() -> None:
    names = source_names_for_split(
        gate.FROZEN_SPLIT,
        manifest_path=ROOT / "configs/denoise_splits_seed20260710.json",
        quarantine_path=ROOT / "configs/denoise_validation_quarantine_v1.json",
        audit_exclusion_path=ROOT / "configs/assembly_audit_exclusion_v1.json",
    )[gate.FROZEN_SOURCE_OFFSET : gate.FROZEN_SOURCE_OFFSET + gate.FROZEN_SOURCE_COUNT]
    assert names == [
        "img_004149.png",
        "img_001707.png",
        "img_000385.png",
        "img_005134.png",
        "img_001005.png",
        "img_003422.png",
        "img_000277.png",
        "img_005032.png",
    ]
    assert gate.names_sha256(names) == gate.FROZEN_SOURCE_NAMES_SHA256


def test_reference_axes_and_metrics_follow_geometry_conventions() -> None:
    identity = np.arange(TILE_COUNT, dtype=np.int32)
    right = gate._reference_paths(identity, "right")
    down = gate._reference_paths(identity, "down")
    assert right[0] == tuple(range(GRID))
    assert down[0] == tuple(range(0, TILE_COUNT, GRID))

    true_right, true_down = true_neighbour_slots(identity)
    assert gate._axis_accuracy(right, true_right) == 1.0
    assert gate._axis_accuracy(down, true_down) == 1.0
    assert gate._path_purity(right, identity, "right") == 1.0
    assert gate._path_purity(down, identity, "down") == 1.0


def test_panel_gate_is_frozen_at_strict_four_cell_thresholds() -> None:
    passing = {
        axis: {
            "mean_adjacency_delta": 0.02,
            "min_adjacency_delta": -0.02,
            "adjacency_wins": 6,
            "mean_path_purity_delta": 0.0,
            "fallbacks": 1,
            "valid_covers": 8,
            "max_selected_rescue_only_fraction": 0.10,
        }
        for axis in ("right", "down")
    }
    assert gate._panel_gate(passing)["passed"] is True
    for axis in ("right", "down"):
        failed = {name: dict(values) for name, values in passing.items()}
        failed[axis]["mean_adjacency_delta"] = np.nextafter(0.02, 0.0)
        assert gate._panel_gate(failed)["passed"] is False


def test_solver_api_has_no_truth_target_or_layout_parameter() -> None:
    parameters = set(inspect.signature(solve_path_cover).parameters)
    assert not parameters.intersection(
        {"truth", "target", "target_pixels", "clean", "slot_to_target", "layout"}
    )
    assert {
        "cost_matrix",
        "path_count",
        "path_length",
        "outgoing_top_k",
        "incoming_top_k",
        "rescue_edges",
        "reference_paths",
    } <= parameters

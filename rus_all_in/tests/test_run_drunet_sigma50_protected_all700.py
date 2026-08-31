from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import torch


def load_runner() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts/run_drunet_sigma50_protected_all700.py"
    spec = importlib.util.spec_from_file_location("run_drunet_sigma50_protected_all700", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class FakeSolve:
    layout: np.ndarray
    solver: str = "fake_buddies96"
    objective: float = 1.25


@dataclass
class FakeAudit:
    passed: bool = True

    def as_dict(self) -> dict[str, object]:
        return {
            "passed": True,
            "input_shape": (480, 480, 3),
            "tile_multiset_equal": True,
        }


def test_infer_uses_sigma50_renderer_contract_and_json_normalizes_audit(monkeypatch) -> None:
    runner = load_runner()
    dirty = np.arange(480 * 480 * 3, dtype=np.uint32).reshape(480, 480, 3).astype(np.uint8)
    layout = np.arange(576, dtype=np.int32)
    renderer_calls: list[np.ndarray] = []

    monkeypatch.setattr(
        runner,
        "directional_scores",
        lambda _tiles, *, views: {"bilateral": (np.zeros((1, 1)), np.zeros((1, 1)))},
    )
    monkeypatch.setattr(
        runner,
        "solve_buddies",
        lambda _right, _down, *, max_edges: FakeSolve(layout=layout.copy()),
    )
    monkeypatch.setattr(runner, "audit_raw_permutation", lambda *args, **kwargs: FakeAudit())
    monkeypatch.setattr(runner, "apply_rgb_luma", lambda tiles: tiles.copy())

    reference = np.full((480, 480, 3), 80, dtype=np.uint8)
    candidate = np.full((480, 480, 3), 81, dtype=np.uint8)

    def fake_render(model, harmonized, *, device):
        del model, device
        renderer_calls.append(harmonized.copy())
        return reference, candidate, {"contract": "sigma50_protected"}

    monkeypatch.setattr(runner, "render_sigma50_protected", fake_render)
    result = runner.infer_board(dirty, torch.nn.Identity(), torch.device("cpu"))

    assert len(renderer_calls) == 1
    assert renderer_calls[0].shape == (576, 20, 20, 3)
    np.testing.assert_array_equal(result["reference"], reference)
    np.testing.assert_array_equal(result["candidate"], candidate)
    assert result["diagnostics"] == {"contract": "sigma50_protected"}
    assert result["audit"]["input_shape"] == [480, 480, 3]
    assert json.loads(json.dumps(result["audit"], sort_keys=True)) == result["audit"]


def test_safety_gate_binds_established_h28_and_flatness_thresholds() -> None:
    runner = load_runner()
    summary = {
        "mean_luminance_gradient_retention": 0.80,
        "minimum_luminance_gradient_retention": 0.70,
        "mean_chroma_gradient_retention": 0.80,
        "minimum_chroma_gradient_retention": 0.70,
        "mean_laplacian_retention": 0.72,
        "minimum_laplacian_retention": 0.60,
        "mean_grid_ratio_relative_to_baseline": 1.05,
        "maximum_grid_ratio_relative_to_baseline": 1.12,
        "protected_fraction_mean_min_max": [0.50, 0.30, 0.85],
        "maximum_clipping_increase": 0.01,
        "candidate_pixel_distinct_from_reference_on_every_board": True,
        "tile_flatness": {
            "reference_exact_constant_total": 10,
            "candidate_exact_constant_total": 10,
            "near_flat_std_lt_2_delta_mean_max": [2.0, 6],
        },
    }
    gate = runner.safety_gate(summary, all_provenance_pass=True)
    assert gate["passed"] is True
    summary["tile_flatness"]["near_flat_std_lt_2_delta_mean_max"] = [2.0, 7]
    assert runner.safety_gate(summary, all_provenance_pass=True)["passed"] is False

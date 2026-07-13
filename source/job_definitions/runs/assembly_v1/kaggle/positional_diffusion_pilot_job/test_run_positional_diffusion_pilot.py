from __future__ import annotations

import copy
import random

import numpy as np
import pytest
import torch

import run_positional_diffusion_pilot as runner


def _rng_hardware() -> dict[str, object]:
    size = int(torch.get_rng_state().numel())
    return {
        "device_count": 2,
        "rng_state_schema": {
            "torch_cpu": {
                "device": "cpu",
                "dtype": "torch.uint8",
                "ndim": 1,
                "numel": size,
            },
            "torch_cuda": [
                {
                    "device_index": index,
                    "device": "cpu",
                    "dtype": "torch.uint8",
                    "ndim": 1,
                    "numel": size,
                }
                for index in range(2)
            ],
        },
    }


def _valid_rng_states() -> list[dict[str, object]]:
    states = []
    for rank in range(2):
        python_generator = random.Random(20260711 + rank)
        numpy_generator = np.random.RandomState(20260711 + rank)
        torch_generator = torch.Generator().manual_seed(20260711 + rank)
        torch_state = torch_generator.get_state()
        states.append(
            {
                "python": python_generator.getstate(),
                "numpy": numpy_generator.get_state(),
                "torch_cpu": torch_state.clone(),
                "torch_cuda": [torch_state.clone(), torch_state.clone()],
            }
        )
    return states


def _valid_development() -> dict[str, object]:
    panels: dict[str, object] = {}
    all_records: list[dict[str, object]] = []
    for panel_index, panel_name in enumerate(
        ("primary_kornia", "independent_libjpeg")
    ):
        records: list[dict[str, object]] = []
        for replica in range(2):
            for source_index, source in enumerate(runner.EXPECTED_DEV_NAMES):
                adjacency_delta = 0.010 + 0.0002 * source_index + 0.0001 * replica
                ssim_delta = 0.006 + 0.0001 * source_index + 0.00005 * replica
                adjacency_delta += 0.0003 * panel_index
                ssim_delta += 0.0002 * panel_index
                adjacency_envelope = 0.45
                ssim_envelope = 0.60
                records.append(
                    {
                        "source": source,
                        "panel": panel_name,
                        "replica": replica,
                        "candidate": {
                            "combined_adjacency": adjacency_envelope + adjacency_delta,
                            "predicted_layout_ssim": ssim_envelope + ssim_delta,
                        },
                        "qap_w1_baseline": {
                            "combined_adjacency": 0.40,
                            "predicted_layout_ssim": 0.55,
                        },
                        "w4_qap_baseline": {
                            "combined_adjacency": adjacency_envelope,
                            "predicted_layout_ssim": ssim_envelope,
                        },
                        "pure_hbt_qap_baseline": {
                            "combined_adjacency": 0.43,
                            "predicted_layout_ssim": 0.58,
                        },
                        "baseline_envelope": {
                            "combined_adjacency": adjacency_envelope,
                            "predicted_layout_ssim": ssim_envelope,
                        },
                        "paired_delta": {
                            "combined_adjacency": adjacency_delta,
                            "predicted_layout_ssim": ssim_delta,
                        },
                        "truth_derived_confidence_used": False,
                        "target_selected_candidate_used": False,
                    }
                )
        adjacency_by_source = np.asarray(
            [
                np.mean(
                    [
                        record["paired_delta"]["combined_adjacency"]
                        for record in records
                        if record["source"] == source
                    ]
                )
                for source in sorted(runner.EXPECTED_DEV_NAMES)
            ],
            dtype=np.float64,
        )
        ssim_by_source = np.asarray(
            [
                np.mean(
                    [
                        record["paired_delta"]["predicted_layout_ssim"]
                        for record in records
                        if record["source"] == source
                    ]
                )
                for source in sorted(runner.EXPECTED_DEV_NAMES)
            ],
            dtype=np.float64,
        )
        adjacency_ci = runner.recompute_bootstrap_ci(
            adjacency_by_source,
            seed=runner.bootstrap_seed(
                20260711, "posdiff:bootstrap-adjacency", panel_name
            ),
        )
        ssim_ci = runner.recompute_bootstrap_ci(
            ssim_by_source,
            seed=runner.bootstrap_seed(20260711, "posdiff:bootstrap-ssim", panel_name),
        )
        mean_adjacency = float(adjacency_by_source.mean())
        mean_ssim = float(ssim_by_source.mean())
        gates = {
            "adjacency_gain_vs_per_source_best_baseline": {
                "value": mean_adjacency,
                "minimum": 0.002,
                "passed": mean_adjacency >= 0.002,
            },
            "ssim_gain_vs_per_source_best_baseline": {
                "value": mean_ssim,
                "minimum": 0.001,
                "passed": mean_ssim >= 0.001,
            },
            "joint_positive_source_fraction": {
                "value": 1.0,
                "minimum": 0.50,
                "passed": True,
            },
            "bootstrap_lower_adjacency_positive": {
                "value": adjacency_ci["lower"],
                "minimum": 0.0,
                "passed": adjacency_ci["lower"] > 0.0,
            },
            "bootstrap_lower_ssim_positive": {
                "value": ssim_ci["lower"],
                "minimum": 0.0,
                "passed": ssim_ci["lower"] > 0.0,
            },
        }
        panels[panel_name] = {
            "source_count": 8,
            "replicas_per_source": 2,
            "cell_count": 16,
            "mean_paired_delta_vs_envelope": {
                "combined_adjacency": mean_adjacency,
                "predicted_layout_ssim": mean_ssim,
            },
            "source_bootstrap_ci": {
                "combined_adjacency": adjacency_ci,
                "predicted_layout_ssim": ssim_ci,
            },
            "gates": gates,
            "gate_passed": all(item["passed"] for item in gates.values()),
            "per_source": records,
        }
        all_records.extend(records)

    macro_adjacency = float(
        np.mean(
            [
                np.mean(
                    [
                        record["paired_delta"]["combined_adjacency"]
                        for record in all_records
                        if record["source"] == source
                    ]
                )
                for source in sorted(runner.EXPECTED_DEV_NAMES)
            ]
        )
    )
    macro_ssim = float(
        np.mean(
            [
                np.mean(
                    [
                        record["paired_delta"]["predicted_layout_ssim"]
                        for record in all_records
                        if record["source"] == source
                    ]
                )
                for source in sorted(runner.EXPECTED_DEV_NAMES)
            ]
        )
    )
    macro_gates = {
        "adjacency_gain": {
            "value": macro_adjacency,
            "minimum": 0.002,
            "passed": macro_adjacency >= 0.002,
        },
        "ssim_gain": {
            "value": macro_ssim,
            "minimum": 0.001,
            "passed": macro_ssim >= 0.001,
        },
    }
    return {
        "source_names": runner.EXPECTED_DEV_NAMES,
        "source_names_sha256": runner.EXPECTED_DEV_NAMES_SHA256,
        "panels": panels,
        "macro_delta_vs_envelope": {
            "combined_adjacency": macro_adjacency,
            "predicted_layout_ssim": macro_ssim,
        },
        "macro_gates": macro_gates,
        "development_gate_passed": True,
        "assessment": runner.STATUS_FOR_GATE[True],
        "safe_for_submission": False,
        "submission_ready": False,
    }


def test_rng_states_are_restorable_and_validation_preserves_global_state() -> None:
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    states = _valid_rng_states()

    assert len(runner.validate_rng_states(states, hardware=_rng_hardware())) == 64
    assert runner.tree_hash(random.getstate()) == runner.tree_hash(python_before)
    assert runner.tree_hash(np.random.get_state()) == runner.tree_hash(numpy_before)
    assert torch.equal(torch.get_rng_state(), torch_before)


def test_rng_states_round_trip_and_restore_two_live_cuda_generators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states = _valid_rng_states()
    cuda_global = [
        torch.Generator().manual_seed(101).get_state(),
        torch.Generator().manual_seed(102).get_state(),
    ]
    cuda_before = [state.clone() for state in cuda_global]

    def get_rng_state_all() -> list[torch.Tensor]:
        return [state.clone() for state in cuda_global]

    def set_rng_state_all(values: list[torch.Tensor]) -> None:
        assert len(values) == 2
        cuda_global[:] = [state.clone() for state in values]

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", get_rng_state_all)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", set_rng_state_all)

    assert len(runner.validate_rng_states(states, hardware=_rng_hardware())) == 64
    assert all(
        torch.equal(after, before)
        for after, before in zip(cuda_global, cuda_before, strict=True)
    )


def test_rng_restore_finally_runs_after_partially_applied_invalid_state() -> None:
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    states = _valid_rng_states()
    states[0]["numpy"] = (
        "MT19937",
        np.zeros(1, dtype=np.uint32),
        0,
        0,
        0.0,
    )

    with pytest.raises(RuntimeError):
        runner.validate_rng_states(states, hardware=_rng_hardware())
    assert runner.tree_hash(random.getstate()) == runner.tree_hash(python_before)
    assert runner.tree_hash(np.random.get_state()) == runner.tree_hash(numpy_before)
    assert torch.equal(torch.get_rng_state(), torch_before)


@pytest.mark.parametrize(
    ("field", "garbage"),
    [
        ("python", None),
        ("numpy", None),
        ("torch_cpu", torch.zeros(1, dtype=torch.uint8)),
        ("torch_cuda", [torch.zeros(1, dtype=torch.uint8)] * 2),
    ],
)
def test_rng_states_reject_none_and_one_byte_garbage(
    field: str, garbage: object
) -> None:
    states = _valid_rng_states()
    states[0][field] = garbage
    with pytest.raises(RuntimeError):
        runner.validate_rng_states(states, hardware=_rng_hardware())


def test_development_requires_exact_deterministic_bootstrap_ci() -> None:
    development = _valid_development()
    assert runner.validate_development(development) == (
        True,
        runner.STATUS_FOR_GATE[True],
    )

    missing = copy.deepcopy(development)
    del missing["panels"]["primary_kornia"]["source_bootstrap_ci"]
    with pytest.raises(RuntimeError):
        runner.validate_development(missing)

    fabricated = copy.deepcopy(development)
    panel = fabricated["panels"]["primary_kornia"]
    panel["source_bootstrap_ci"]["combined_adjacency"]["lower"] = 0.123
    panel["gates"]["bootstrap_lower_adjacency_positive"] = {
        "value": 0.123,
        "minimum": 0.0,
        "passed": True,
    }
    with pytest.raises(RuntimeError):
        runner.validate_development(fabricated)

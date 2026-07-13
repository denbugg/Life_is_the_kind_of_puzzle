from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from puzzle_assembly.compatibility import CompatibilityMatrices
from puzzle_assembly.protocol import source_names_for_split

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "evaluate_d4_consensus_gate",
    ROOT / "scripts/evaluate_d4_consensus_gate.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _record(adjacency: float, ssim: float, seconds: float, changed: bool) -> dict:
    return {
        "delta": {
            "combined_adjacency": adjacency,
            "harmonized_ssim": ssim,
        },
        "phase_a": {
            "candidate_seconds": seconds,
            "layout_changed": changed,
        },
    }


def test_frozen_source_fingerprint_and_protocol_slice() -> None:
    names = source_names_for_split(
        "edge_development",
        manifest_path=ROOT / "configs/denoise_splits_seed20260710.json",
        quarantine_path=ROOT / "configs/denoise_validation_quarantine_v1.json",
        audit_exclusion_path=ROOT / "configs/assembly_audit_exclusion_v1.json",
    )[340:348]
    digest = hashlib.sha256(("\n".join(names) + "\n").encode()).hexdigest()
    assert MODULE.FROZEN_SPLIT == "edge_development"
    assert MODULE.FROZEN_SOURCE_OFFSET == 340
    assert MODULE.FROZEN_SOURCE_COUNT == 8
    assert MODULE.FROZEN_SOURCE_NAMES_SHA256 == digest


def test_layout_solver_and_renderer_have_no_target_bearing_parameters() -> None:
    assert tuple(inspect.signature(MODULE._solve_layout).parameters) == (
        "seed_score",
        "qap_score",
        "name",
    )
    assert tuple(inspect.signature(MODULE._render_promoted).parameters) == (
        "selected_slot_tiles",
        "seam_slot_tiles",
        "layout",
    )
    protocol = MODULE._protocol(["opaque"])
    assert protocol["production_solver"].startswith("softcycle_l1_")
    assert protocol["authoritative_score_builder_sha256"] == MODULE.AUTHORITATIVE_BUILDER_SHA256


def test_phase_b_requires_global_two_panel_authorization(tmp_path: Path) -> None:
    phase_a_report = tmp_path / "phase_a.json"
    artifact = tmp_path / "phase_a.npz"
    authorization = tmp_path / "authorization.json"
    phase_a_report.write_text(
        json.dumps(
            {
                "kind": "d4_compatibility_consensus_phase_a",
                "panel": "primary_kornia",
                "phase_a": {"passed": True, "target_metrics_opened": False},
            }
        )
    )
    artifact.write_bytes(b"sealed")
    authorization.write_text(json.dumps({"authorized": False}))
    args = type(
        "Args",
        (),
        {"panel": "primary_kornia", "data_root": str(tmp_path)},
    )()
    with pytest.raises(RuntimeError, match="valid global two-panel authorization"):
        MODULE._run_phase_b(
            args,
            tmp_path / "output.json",
            artifact,
            phase_a_report,
            authorization,
            {},
            ["opaque.png"],
        )


def test_phase_a_artifact_is_hash_sealed_and_round_trips(tmp_path: Path) -> None:
    artifact = tmp_path / "sealed.npz"
    record = {
        "name": "opaque.png",
        "panel_seed": 17,
        "raw_tiles": np.zeros((1, 1), dtype=np.uint8),
        "selected_tiles": np.ones((1, 1), dtype=np.uint8),
        "seam_tiles": np.full((1, 1), 2, dtype=np.uint8),
        "baseline_layout": np.asarray([0], dtype=np.int32),
        "candidate_layout": np.asarray([0], dtype=np.int32),
        "baseline_render": np.full((1, 1), 3, dtype=np.uint8),
        "candidate_render": np.full((1, 1), 4, dtype=np.uint8),
    }
    digest = MODULE._atomic_write_phase_a_artifact(artifact, [record])
    loaded = MODULE._load_phase_a_artifact(
        artifact,
        expected_sha256=digest,
        expected_names=["opaque.png"],
    )
    assert loaded["panel_seeds"].tolist() == [17]
    assert loaded["candidate_render"].tolist() == [[[4]]]
    artifact.write_bytes(artifact.read_bytes() + b"tamper")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        MODULE._load_phase_a_artifact(
            artifact,
            expected_sha256=digest,
            expected_names=["opaque.png"],
        )


def test_fast_score_equivalence_requires_name_dtype_and_exact_values() -> None:
    values = np.zeros((576, 576), dtype=np.float32)
    np.fill_diagonal(values, np.inf)
    down = values.copy()
    down[1, 2] = 1.0
    first = CompatibilityMatrices("frozen", values.copy(), down.copy())
    same = CompatibilityMatrices("frozen", values.copy(), down.copy())
    assert MODULE._scores_bit_exact(first, same)
    changed = values.copy()
    changed[0, 1] = np.nextafter(np.float32(0.0), np.float32(1.0))
    assert not MODULE._scores_bit_exact(
        first, CompatibilityMatrices("frozen", changed, down.copy())
    )
    assert not MODULE._scores_bit_exact(
        first, CompatibilityMatrices("renamed", values.copy(), down.copy())
    )
    assert not MODULE._scores_bit_exact(
        first, CompatibilityMatrices("frozen", down.copy(), values.copy())
    )


@pytest.mark.parametrize("defect", ["dtype", "shape", "nan", "diagonal", "signed_zero"])
def test_fast_score_equivalence_rejects_structural_defects(defect: str) -> None:
    values = np.zeros((576, 576), dtype=np.float32)
    np.fill_diagonal(values, np.inf)
    first = CompatibilityMatrices("frozen", values.copy(), values.copy())
    second = CompatibilityMatrices("frozen", values.copy(), values.copy())
    if defect == "dtype":
        object.__setattr__(second, "right", second.right.astype(np.float64))
    elif defect == "shape":
        object.__setattr__(second, "right", second.right[:-1])
    elif defect == "nan":
        second.right[0, 1] = np.nan
    elif defect == "diagonal":
        second.right[0, 0] = -np.inf
    else:
        second.right[0, 1] = -0.0
        assert np.array_equal(first.right, second.right)
    assert not MODULE._scores_bit_exact(first, second)


def test_fast_builder_uses_exact_production_primitives(monkeypatch: pytest.MonkeyPatch) -> None:
    values = np.zeros((576, 576), dtype=np.float32)
    np.fill_diagonal(values, np.inf)
    bank = {
        "denoised_z_c2": CompatibilityMatrices("z_c2", values, values),
        "denoised_b": CompatibilityMatrices("b", values, values),
        "denoised_a": CompatibilityMatrices("a", values, values),
    }
    calls = []

    def fake_bank(tiles, *, prefix, chunk_size):
        assert prefix == "denoised"
        assert chunk_size == 64
        return bank

    def fake_fuse(score_bank, *, names, weights=None, name):
        calls.append((list(names), weights, name))
        return CompatibilityMatrices(name, values.copy(), values.copy())

    def fake_learned(model, tiles, *, device, name):
        assert name == "denoised_l1_embedding"
        return CompatibilityMatrices(name, values.copy(), values.copy()), np.zeros(1)

    monkeypatch.setattr(MODULE, "build_classical_score_bank", fake_bank)
    monkeypatch.setattr(MODULE, "fuse_ranked_scores", fake_fuse)
    monkeypatch.setattr(MODULE, "learned_compatibility", fake_learned)
    l1, l1w4 = MODULE._build_fast_equivalent_scores(
        np.zeros((576, 20, 20, 3), dtype=np.uint8), object(), device="cpu"
    )
    assert l1.name == "denoised_l1_embedding"
    assert l1w4.name == "denoised_C1_L1w4_rank_fusion"
    assert calls == [
        (
            ["denoised_a", "denoised_b"],
            None,
            "denoised_C1_equal_rank_fusion",
        ),
        (
            ["denoised_C1_equal_rank_fusion", "denoised_l1_embedding"],
            {"denoised_l1_embedding": 4.0},
            "denoised_C1_L1w4_rank_fusion",
        ),
    ]


def test_panel_gate_exactly_matches_frozen_thresholds() -> None:
    records = [_record(0.003, 0.002, 20.0, True) for _ in range(8)]
    summary = MODULE._panel_summary(records)
    gate = MODULE._panel_gate(summary)
    assert summary["ssim_wins"] == 8
    assert summary["different_layouts"] == 8
    assert gate["passed"] is True
    assert set(gate["checks"]) == {
        "mean_combined_adjacency_delta_ge_0.003",
        "mean_harmonized_ssim_delta_ge_0.002",
        "ssim_wins_ge_6_of_8",
        "worst_harmonized_ssim_delta_ge_-0.010",
        "max_candidate_seconds_le_20",
    }


def test_panel_gate_fails_each_regression_boundary() -> None:
    cases = [
        [_record(0.0029, 0.002, 19.0, True) for _ in range(8)],
        [_record(0.003, 0.0019, 19.0, True) for _ in range(8)],
        [_record(0.003, 0.003 if index < 5 else -0.001, 19.0, True) for index in range(8)],
        [_record(0.003, -0.0101 if index == 0 else 0.004, 19.0, True) for index in range(8)],
        [_record(0.003, 0.002, 20.01 if index == 0 else 19.0, True) for index in range(8)],
    ]
    for records in cases:
        assert MODULE._panel_gate(MODULE._panel_summary(records))["passed"] is False

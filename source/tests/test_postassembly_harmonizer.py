from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from puzzle_assembly.postassembly_harmonizer import (
    LuminanceGainConfig,
    SeamGraphConfig,
    apply_luminance_gains,
    apply_rgb_offsets,
    bilateral_tile_offsets,
    blend_tiles_uint8,
    image_quality_metrics,
    naive_local_mean_offsets,
    ordered_from_slots,
    paired_bootstrap_ci,
    seam_graph_luminance_gains,
    seam_graph_rgb_offsets,
)
from puzzle_assembly.protocol import source_names_for_split
from puzzle_denoise_v2.tiles import split_tiles_numpy


ROOT = Path(__file__).resolve().parents[1]


def _smooth_target() -> np.ndarray:
    y, x = np.mgrid[:480, :480]
    return np.stack(
        (
            30.0 + 0.25 * x + 0.05 * y,
            40.0 + 0.10 * x + 0.20 * y,
            50.0 + 0.15 * x + 0.10 * y,
        ),
        axis=-1,
    ).clip(0, 255).astype(np.uint8)


def test_ordered_from_slots_exactly_inverts_permutation() -> None:
    target = split_tiles_numpy(_smooth_target())
    permutation = np.random.default_rng(11).permutation(576).astype(np.int32)
    slots = target[permutation]
    assert np.array_equal(ordered_from_slots(slots, permutation), target)
    with pytest.raises(ValueError, match="not a permutation"):
        ordered_from_slots(slots, np.zeros(576, dtype=np.int32))


def test_seam_graph_recovers_synthetic_additive_tile_bias() -> None:
    target = split_tiles_numpy(_smooth_target())
    rng = np.random.default_rng(7)
    bias = rng.uniform(-8.0, 8.0, size=(576, 3))
    observed = np.clip(
        np.rint(target.astype(np.float64) + bias[:, None, None, :]), 0, 255
    ).astype(np.uint8)

    offsets, diagnostics = seam_graph_rgb_offsets(observed, SeamGraphConfig())
    expected = -bias + np.median(bias, axis=0, keepdims=True)
    corrected = apply_rgb_offsets(observed, offsets)

    assert offsets.shape == (576, 3)
    assert np.median(np.abs(offsets - expected)) < 0.8
    assert diagnostics["edge_count"] == 1104.0
    assert image_quality_metrics(corrected, target)["ssim"] > 0.995
    assert image_quality_metrics(corrected, target)["ssim"] > image_quality_metrics(
        observed, target
    )["ssim"]


def test_shuffled_neighbour_placebo_is_deterministic_and_fails_falsification() -> None:
    target = split_tiles_numpy(_smooth_target())
    rng = np.random.default_rng(19)
    bias = rng.uniform(-8.0, 8.0, size=(576, 3))
    observed = np.clip(
        np.rint(target.astype(np.float64) + bias[:, None, None, :]), 0, 255
    ).astype(np.uint8)
    real_offsets, _ = seam_graph_rgb_offsets(observed)
    placebo_a, placebo_diagnostics = seam_graph_rgb_offsets(observed, placebo_seed=123)
    placebo_b, _ = seam_graph_rgb_offsets(observed, placebo_seed=123)

    assert np.array_equal(placebo_a, placebo_b)
    assert placebo_diagnostics["placebo"] is True
    real_ssim = image_quality_metrics(apply_rgb_offsets(observed, real_offsets), target)[
        "ssim"
    ]
    placebo_ssim = image_quality_metrics(apply_rgb_offsets(observed, placebo_a), target)[
        "ssim"
    ]
    assert real_ssim > placebo_ssim + 0.02


def test_local_controls_and_blend_are_bounded() -> None:
    rng = np.random.default_rng(3)
    first = rng.integers(0, 256, size=(576, 20, 20, 3), dtype=np.uint8)
    second = rng.integers(0, 256, size=(576, 20, 20, 3), dtype=np.uint8)
    blend = blend_tiles_uint8(first, second, auxiliary_weight=0.5)
    expected = np.clip(
        np.rint(0.5 * first.astype(np.float32) + 0.5 * second.astype(np.float32)),
        0,
        255,
    ).astype(np.uint8)
    assert np.array_equal(blend, expected)

    naive = naive_local_mean_offsets(blend, max_abs_offset=7.0)
    bilateral = bilateral_tile_offsets(blend, max_abs_offset=7.0)
    assert naive.shape == bilateral.shape == (576, 3)
    assert np.max(np.abs(naive)) <= 7.0
    assert np.max(np.abs(bilateral)) <= 7.0


def test_bounded_luminance_gain_recovers_synthetic_gain() -> None:
    target_image = (_smooth_target().astype(np.float64) * 0.65 + 40.0).clip(20, 220).astype(
        np.uint8
    )
    target = split_tiles_numpy(target_image)
    rng = np.random.default_rng(9)
    nuisance_gain = rng.uniform(0.97, 1.03, size=576)
    observed = np.clip(
        np.rint(target.astype(np.float64) * nuisance_gain[:, None, None, None]),
        0,
        255,
    ).astype(np.uint8)
    gains, diagnostics = seam_graph_luminance_gains(observed, LuminanceGainConfig())
    corrected = apply_luminance_gains(observed, gains)

    assert gains.shape == (576,)
    assert gains.min() >= 0.96 - 1e-6
    assert gains.max() <= 1.04 + 1e-6
    assert diagnostics["gain_min"] >= 0.96 - 1e-6
    assert image_quality_metrics(corrected, target)["ssim"] > image_quality_metrics(
        observed, target
    )["ssim"]


def test_metrics_identity_and_boundary_error_order() -> None:
    target = split_tiles_numpy(_smooth_target())
    identity = image_quality_metrics(target, target)
    assert identity["ssim"] == pytest.approx(1.0)
    assert identity["boundary_band_mae"] == 0.0
    assert identity["target_referenced_seam_error"] == 0.0

    damaged = target.copy()
    damaged[1::2] = np.clip(damaged[1::2].astype(np.int16) + 10, 0, 255).astype(
        np.uint8
    )
    metrics = image_quality_metrics(damaged, target)
    assert metrics["ssim"] < 1.0
    assert metrics["boundary_band_mae"] > 0.0
    assert metrics["target_referenced_seam_error"] > 0.0


def test_paired_bootstrap_is_source_level_and_deterministic() -> None:
    deltas = np.asarray([0.004, 0.006, 0.008, 0.010], dtype=np.float64)
    first = paired_bootstrap_ci(deltas, seed=17, resamples=5000)
    second = paired_bootstrap_ci(deltas, seed=17, resamples=5000)
    assert first == second
    assert first[0] > 0
    assert first[0] <= deltas.mean() <= first[1]


def test_harmonizer_public_api_has_no_target_or_source_argument() -> None:
    for function in (
        seam_graph_rgb_offsets,
        naive_local_mean_offsets,
        bilateral_tile_offsets,
        seam_graph_luminance_gains,
    ):
        parameters = set(inspect.signature(function).parameters)
        assert "target" not in parameters
        assert "source" not in parameters
        assert "source_name" not in parameters


def test_protocol_selection_and_separate_gain_hash_are_consistent() -> None:
    base_path = ROOT / "configs/postassembly_rgb_offset_v1.json"
    gain_path = ROOT / "configs/postassembly_luminance_gain_v1.json"
    base = json.loads(base_path.read_text(encoding="utf-8"))
    gain = json.loads(gain_path.read_text(encoding="utf-8"))
    selected = base["source_selection"]["names"]

    assert len(selected) == len(set(selected)) == 32
    assert hashlib.sha256("\n".join(selected).encode()).hexdigest() == base[
        "source_selection"
    ]["names_sha256"]
    assert hashlib.sha256(base_path.read_bytes()).hexdigest() == gain["base_protocol"][
        "sha256"
    ]
    edge_development = source_names_for_split(
        "edge_development",
        manifest_path=ROOT / "configs/denoise_splits_seed20260710.json",
        quarantine_path=ROOT / "configs/denoise_validation_quarantine_v1.json",
    )
    assert not set(selected) & set(edge_development[128:160])
    assert not set(selected) & {path.name for path in (ROOT / "puzzle/test").glob("*.png")}
    manifest = json.loads(
        (ROOT / "configs/denoise_splits_seed20260710.json").read_text(encoding="utf-8")
    )
    assert not set(selected) & set(manifest["excluded_test_overlap"])
    for split in (
        "assembly_cal",
        "assembly_incremental_gate",
        "assembly_audit_exposed",
        "assembly_final_audit",
    ):
        assert not set(selected) & set(
            source_names_for_split(
                split,
                manifest_path=ROOT / "configs/denoise_splits_seed20260710.json",
                quarantine_path=ROOT / "configs/denoise_validation_quarantine_v1.json",
                audit_exclusion_path=ROOT / "configs/assembly_audit_exclusion_v1.json",
            )
        )


def _load_actual_layout_runner():
    path = ROOT / "scripts/evaluate_postassembly_actual_layout.py"
    spec = importlib.util.spec_from_file_location("postassembly_actual_layout_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_actual_layout_protocol_is_frozen_and_reuses_base_without_retuning() -> None:
    actual_path = ROOT / "configs/postassembly_actual_qap_layout_v1.json"
    actual = json.loads(actual_path.read_text(encoding="utf-8"))
    base_path = ROOT / actual["base_harmonizer_protocol"]["path"]
    base = json.loads(base_path.read_text(encoding="utf-8"))
    names = base["source_selection"]["names"]

    assert actual["status"] == "precommitted_before_actual_layout_target_metrics"
    assert actual["base_harmonizer_protocol"]["reuse_without_retuning"] is True
    assert hashlib.sha256(base_path.read_bytes()).hexdigest() == actual[
        "base_harmonizer_protocol"
    ]["sha256"]
    assert hashlib.sha256("\n".join(names).encode()).hexdigest() == actual[
        "source_selection"
    ]["names_sha256"]
    assert hashlib.sha256("\n".join(names[:8]).encode()).hexdigest() == actual[
        "source_selection"
    ]["small_smoke_names_sha256"]
    assert actual["scope"]["layout_refinement_forbidden"] is True
    assert actual["scope"]["same_frozen_qap_w4_layout_for_every_render_arm"] is True


def test_actual_layout_predictor_api_is_label_blind_and_layout_metrics_are_exact() -> None:
    runner = _load_actual_layout_runner()
    parameters = set(inspect.signature(runner._predict_qap_w4).parameters)
    assert "target" not in parameters
    assert "slot_to_target" not in parameters
    assert "clean" not in parameters
    identity = np.arange(576, dtype=np.int32)
    metrics = runner._layout_metrics(identity, identity)
    assert metrics == {
        "valid_permutation": True,
        "strict_position_accuracy": 1.0,
        "right_down_adjacency_recall": 1.0,
    }
    assert runner.ARMS == (
        "raw_on_frozen_qap_w4",
        "selected_tilenaf_on_frozen_qap_w4",
        "production_seam_tilenaf_on_frozen_qap_w4",
        "fixed_alpha_0_5_on_frozen_qap_w4",
        "seam_graph_rgb_on_frozen_qap_w4",
        "shuffled_neighbor_placebo_on_frozen_qap_w4",
        "naive_5x5_on_frozen_qap_w4",
        "bilateral_offset_on_frozen_qap_w4",
    )


def test_source_disjoint_confirmation_is_precommitted_and_runner_pinned() -> None:
    path = ROOT / "scripts/run_postassembly_actual_confirmation.py"
    spec = importlib.util.spec_from_file_location("postassembly_confirmation_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config_path = ROOT / "configs/postassembly_actual_qap_confirmation_v1.json"
    confirmation, effective, names, _runner, runner_path = module._validate(config_path)

    assert confirmation["status"] == "precommitted_before_confirmation_pixel_or_metric_access"
    assert len(names) == len(set(names)) == 32
    assert set(names).isdisjoint(
        json.loads(
            (ROOT / "configs/postassembly_rgb_offset_v1.json").read_text(encoding="utf-8")
        )["source_selection"]["names"]
    )
    assert hashlib.sha256(runner_path.read_bytes()).hexdigest() == confirmation[
        "implementation"
    ]["frozen_runner_sha256"]
    assert effective["scope"]["layout_refinement_forbidden"] is True

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
from PIL import Image

from puzzle_assembly.layout_energy_hybrid import (
    DirectionalSeam,
    SwapProposal,
    apply_swap,
    local_seam_costs,
    raw_border_l1_seam,
    seam_objective,
    seam_select_or_noop,
    sha256_array,
    sha256_file,
    swap_seam_delta,
    top_delta_swaps,
)
from scripts.evaluate_layout_energy_hybrid import (
    _predict_source,
    _score_frozen_predictions,
    paired_bootstrap,
)
from scripts.export_frozen_real16_layouts import extract_layout_only_manifest


def _random_seam(count: int, seed: int = 0) -> DirectionalSeam:
    rng = np.random.default_rng(seed)
    right = rng.uniform(0.01, 1.0, size=(count, count)).astype(np.float32)
    down = rng.uniform(0.01, 1.0, size=(count, count)).astype(np.float32)
    np.fill_diagonal(right, np.inf)
    np.fill_diagonal(down, np.inf)
    return DirectionalSeam(right=right, down=down)


def test_swap_delta_matches_full_objective() -> None:
    seam = _random_seam(16, seed=3)
    layout = np.random.default_rng(4).permutation(16).astype(np.int32)
    for first, second in ((0, 1), (0, 15), (5, 10), (7, 8)):
        proposal = SwapProposal(
            delta=swap_seam_delta(layout, seam, first, second),
            first=first,
            second=second,
        )
        actual = seam_objective(apply_swap(layout, proposal), seam) - seam_objective(
            layout, seam
        )
        assert np.isclose(proposal.delta, actual, atol=1e-6)


def test_top_delta_swaps_is_exact_deterministic_and_suspect_bounded() -> None:
    seam = _random_seam(16, seed=5)
    layout = np.arange(16, dtype=np.int32)
    first = top_delta_swaps(layout, seam, [1, 6], budget=7)
    second = top_delta_swaps(layout, seam, [6, 1, 1], budget=7)
    assert first == second
    assert len(first) == 7
    assert list(first) == sorted(first)
    assert all(1 in (item.first, item.second) or 6 in (item.first, item.second) for item in first)


def test_seam_selector_never_accepts_non_improving_move() -> None:
    layout = np.arange(16, dtype=np.int32)
    rejected, selected = seam_select_or_noop(
        layout,
        [SwapProposal(delta=0.1, first=0, second=1)],
    )
    assert selected is None
    assert np.array_equal(rejected, layout)
    accepted, selected = seam_select_or_noop(
        layout,
        [
            SwapProposal(delta=-0.1, first=0, second=1),
            SwapProposal(delta=-0.2, first=2, second=3),
        ],
    )
    assert selected == SwapProposal(delta=-0.2, first=2, second=3)
    assert accepted[2] == 3 and accepted[3] == 2


def test_local_costs_and_raw_border_seam_are_finite() -> None:
    rng = np.random.default_rng(9)
    tiles = rng.integers(0, 256, size=(16, 4, 4, 3), dtype=np.uint8)
    seam = raw_border_l1_seam(tiles, strip=2, chunk_size=5)
    assert seam.right.shape == (16, 16)
    assert np.isinf(np.diag(seam.right)).all()
    costs = local_seam_costs(np.arange(16, dtype=np.int32), seam)
    assert costs.shape == (16,)
    assert np.isfinite(costs).all()


def _fake_solver_report(name: str) -> dict:
    layout = np.arange(576, dtype=np.int32).tolist()
    variants = {}
    for prefix in ("softcycle_l1_k8", "qap_softcycle_l1_k8"):
        for render in ("raw", "denoised"):
            variants[f"{prefix}__{render}_render"] = {
                "position_to_slot": layout,
                "predicted_layout_ssim": 0.999,
                "mae": 1.0,
            }
    return {
        "schema_version": 1,
        "kind": "real_input_only_assembly_target_only_score",
        "split": "assembly_cal",
        "anti_leakage": {
            "predictor_accepts_target": False,
            "target_opened_after_layouts_frozen": True,
            "pseudo_mapping_used": False,
        },
        "qap": {
            "boundary_weight": 0.05,
            "initial_weight": 0.75,
            "iterations": 25,
            "noise_scale": 1.0,
            "noisy_components": 3,
            "refine_swaps": 8,
            "restarts": 2,
            "score": "l1w4",
            "seeds": ["softcycle_l1_k8"],
        },
        "source_names": [name],
        "sources": [{"source": name, "variants": variants}],
    }


def test_layout_export_whitelists_layouts_and_never_exports_metrics(tmp_path: Path) -> None:
    name = "img_test.png"
    input_path = tmp_path / "train" / "inputs" / name
    input_path.parent.mkdir(parents=True)
    Image.fromarray(np.zeros((480, 480, 3), dtype=np.uint8)).save(input_path)
    manifest = extract_layout_only_manifest(
        _fake_solver_report(name),
        source_report_name="source.json",
        source_report_sha256="a" * 64,
        data_root=tmp_path,
        expected_source_count=1,
    )
    serialized = json.dumps(manifest)
    assert "predicted_layout_ssim" not in serialized
    assert '"mae"' not in serialized
    assert manifest["export_contract"]["target_paths_accessed"] is False
    assert manifest["export_contract"]["target_metrics_exported"] is False
    assert set(manifest["sources"][0]["layouts"]) == {"hbt", "qap"}


def test_predictor_api_cannot_receive_target() -> None:
    parameters = inspect.signature(_predict_source).parameters
    assert "target" not in parameters
    assert "target_path" not in parameters


def test_paired_bootstrap_is_deterministic() -> None:
    values = np.asarray([0.01, -0.002, 0.005, 0.007], dtype=np.float64)
    first = paired_bootstrap(values, seed=17, samples=1000)
    second = paired_bootstrap(values, seed=17, samples=1000)
    assert first == second
    assert np.isclose(first["mean"], values.mean())
    assert first["lower_95"] <= first["mean"] <= first["upper_95"]


def test_target_scoring_reads_an_immutable_prediction_artifact(tmp_path: Path) -> None:
    name = "img_test.png"
    inputs = tmp_path / "train" / "inputs"
    targets = tmp_path / "train" / "targets"
    inputs.mkdir(parents=True)
    targets.mkdir(parents=True)
    image = np.zeros((480, 480, 3), dtype=np.uint8)
    image[:, :, 1] = np.arange(480, dtype=np.uint16)[None, :] % 256
    Image.fromarray(image).save(inputs / name)
    Image.fromarray(image).save(targets / name)
    layout = np.arange(576, dtype=np.int32)
    layout_list = layout.tolist()
    layout_hash = sha256_array(layout)

    def variant() -> dict:
        return {
            "position_to_slot": layout_list,
            "position_to_slot_sha256": layout_hash,
            "accepted_move": False,
            "selected_swap": None,
            "proposal_budget": 4,
        }

    frozen = {
        "prediction_stage": {"targets_accessed": False},
        "configuration": {"suspect_k": [8]},
        "sources": [
            {
                "source": name,
                "raw_input": f"train/inputs/{name}",
                "raw_input_sha256": sha256_file(inputs / name),
                "bases": [
                    {
                        "base": "qap",
                        "base_position_to_slot": layout_list,
                        "base_position_to_slot_sha256": layout_hash,
                        "base_error_probabilities": [0.5] * 576,
                        "configurations": [
                            {
                                "suspect_k": 8,
                                "variants": {
                                    "critic_heatmap_seam": variant(),
                                    "critic_heatmap_energy_rerank": variant(),
                                    "seam_only_control": variant(),
                                    "no_op_budget_matched": variant(),
                                },
                            }
                        ],
                    }
                ],
            }
        ],
    }
    prediction_path = tmp_path / "frozen.json"
    prediction_path.write_text(json.dumps(frozen), encoding="utf-8")
    before = sha256_file(prediction_path)
    report = _score_frozen_predictions(
        prediction_path,
        data_root=tmp_path,
        bootstrap_samples=100,
        seed=23,
    )
    assert sha256_file(prediction_path) == before
    assert report["safe_for_submission"] is False
    assert report["anti_leakage"]["prediction_artifact_unchanged"] is True
    assert report["status"] == "no_actionable_signal"

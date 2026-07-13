from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from puzzle_assembly.compatibility import CompatibilityMatrices
from puzzle_assembly.pair_transformer import (
    DOWN,
    RIGHT,
    PairCandidates,
    PairTransformerScorer,
    fit_binary_temperature,
    load_pair_transformer_checkpoint,
    load_pair_transformer_checkpoint_payload,
    multistage_candidates,
    save_pair_transformer_checkpoint,
    score_pairs,
)


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/train_evaluate_pair_transformer.py"
_SPEC = importlib.util.spec_from_file_location("pair_transformer_pilot", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
pilot = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = pilot
_SPEC.loader.exec_module(pilot)

_RUNNER_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "runs/assembly_v1/kaggle/pair_transformer_pilot_job/run_pair_transformer_pilot.py"
)
_RUNNER_SPEC = importlib.util.spec_from_file_location(
    "pair_transformer_kaggle_runner", _RUNNER_SCRIPT
)
assert _RUNNER_SPEC is not None and _RUNNER_SPEC.loader is not None
runner = importlib.util.module_from_spec(_RUNNER_SPEC)
sys.modules[_RUNNER_SPEC.name] = runner
_RUNNER_SPEC.loader.exec_module(runner)


def _tiny_model() -> PairTransformerScorer:
    return PairTransformerScorer(
        model_dim=32,
        layers=1,
        heads=4,
        feedforward_dim=64,
        cnn_channels=16,
        patch_grid=2,
        side_band=3,
        band_tokens=3,
        dropout=0.0,
        gradient_checkpointing=False,
    )


def _coarse() -> CompatibilityMatrices:
    rng = np.random.default_rng(7)
    right = rng.random((576, 576), dtype=np.float32)
    down = rng.random((576, 576), dtype=np.float32)
    np.fill_diagonal(right, np.inf)
    np.fill_diagonal(down, np.inf)
    return CompatibilityMatrices("hbt", right, down)


def test_full_tile_pair_forward_is_directional_and_backpropagates() -> None:
    torch.manual_seed(11)
    model = _tiny_model().train()
    first = torch.rand(3, 6, 20, 20, requires_grad=True)
    second = torch.rand(3, 6, 20, 20, requires_grad=True)
    directions = torch.as_tensor([RIGHT, DOWN, RIGHT])
    output = model(first, second, directions)
    assert set(output) == {
        "logits",
        "calibrated_logits",
        "probability",
        "confidence",
        "pair_embedding",
    }
    assert output["logits"].shape == (3,)
    assert output["pair_embedding"].shape == (3, 64)
    assert torch.isfinite(output["logits"]).all()
    output["logits"].sum().backward()
    assert first.grad is not None and torch.isfinite(first.grad).all()
    assert second.grad is not None and torch.isfinite(second.grad).all()


def test_interior_pixels_affect_score_not_only_explicit_band() -> None:
    torch.manual_seed(13)
    model = _tiny_model().eval()
    first = torch.rand(1, 6, 20, 20)
    second = torch.rand(1, 6, 20, 20)
    with torch.inference_mode():
        reference = model(first, second, torch.as_tensor([RIGHT]))["logits"]
        changed = first.clone()
        changed[:, :, 8:12, 8:12] = 1.0 - changed[:, :, 8:12, 8:12]
        modified = model(changed, second, torch.as_tensor([RIGHT]))["logits"]
    assert not torch.allclose(reference, modified, atol=1e-7, rtol=1e-7)


def test_multistage_candidates_include_coarse_and_current_layout_edges() -> None:
    coarse = _coarse()
    layout = np.arange(576, dtype=np.int32)[::-1]
    candidates = multistage_candidates(
        coarse, top_k=2, reverse_top_k=1, layouts=[layout]
    )
    triples = set(
        zip(
            candidates.direction.tolist(),
            candidates.first.tolist(),
            candidates.second.tolist(),
            strict=True,
        )
    )
    grid = layout.reshape(24, 24)
    assert (RIGHT, int(grid[0, 0]), int(grid[0, 1])) in triples
    assert (DOWN, int(grid[0, 0]), int(grid[1, 0])) in triples
    assert all(first != second for _, first, second in triples)
    assert candidates.counts()["right"] >= 576 * 2
    assert candidates.counts()["down"] >= 576 * 2


def test_sparse_pair_scoring_contract() -> None:
    rng = np.random.default_rng(17)
    raw = rng.integers(0, 256, size=(576, 20, 20, 3), dtype=np.uint8)
    denoised = rng.integers(0, 256, size=(576, 20, 20, 3), dtype=np.uint8)
    candidates = PairCandidates(
        first=np.asarray([0, 0, 1, 2], dtype=np.int32),
        second=np.asarray([1, 2, 3, 4], dtype=np.int32),
        direction=np.asarray([RIGHT, DOWN, RIGHT, DOWN], dtype=np.int8),
    )
    telemetry: dict[str, object] = {}
    logits, probability, confidence = score_pairs(
        _tiny_model(),
        raw,
        denoised,
        candidates,
        device="cpu",
        batch_size=2,
        telemetry=telemetry,
    )
    assert logits.shape == probability.shape == confidence.shape == (4,)
    assert np.isfinite(logits).all()
    assert np.all((probability >= 0.0) & (probability <= 1.0))
    assert np.all((confidence >= 0.0) & (confidence <= 1.0))
    assert telemetry["cached_tile_count"] == 576
    assert telemetry["tile_encoder_passes"] == 1
    assert telemetry["candidate_pairs"] == 4
    assert float(telemetry["pairs_per_second"]) > 0.0


def test_cached_tile_bank_matches_uncached_pair_forward() -> None:
    torch.manual_seed(18)
    model = _tiny_model().eval()
    tiles = torch.rand(5, 6, 20, 20)
    first = torch.as_tensor([0, 0, 3, 4])
    second = torch.as_tensor([1, 2, 4, 1])
    directions = torch.as_tensor([RIGHT, DOWN, RIGHT, DOWN])
    with torch.inference_mode():
        direct = model(tiles[first], tiles[second], directions)
        encoded = model.encode_tile_bank(tiles)
        cached = model.forward_from_encoded(
            encoded, encoded, first, second, directions
        )
    for key in ("logits", "probability", "confidence", "pair_embedding"):
        torch.testing.assert_close(cached[key], direct[key], atol=2e-6, rtol=2e-6)


def test_coordinate_coded_directional_bands_select_physical_sides() -> None:
    model = _tiny_model()
    x = torch.arange(20, dtype=torch.float32).view(1, 1, 20).expand(1, 20, 20)
    y = torch.arange(20, dtype=torch.float32).view(1, 20, 1).expand(1, 20, 20)
    first = torch.cat([x, y], dim=0).unsqueeze(0).repeat(2, 1, 1, 1)
    second = first + 100.0
    first_band, second_band = model._canonical_bands(
        first, second, torch.as_tensor([RIGHT, DOWN])
    )
    torch.testing.assert_close(first_band[0], first[0, :, :, -3:])
    torch.testing.assert_close(second_band[0], second[0, :, :, :3])
    torch.testing.assert_close(
        first_band[1], first[1, :, -3:, :].transpose(1, 2)
    )
    torch.testing.assert_close(
        second_band[1], second[1, :, :3, :].transpose(1, 2)
    )
    assert first_band[0, 0, 7].tolist() == [17.0, 18.0, 19.0]
    assert first_band[1, 1, 7].tolist() == [17.0, 18.0, 19.0]


def test_temperature_fit_improves_miscalibrated_binary_nll() -> None:
    logits = np.asarray([-8.0, -5.0, -2.0, 2.0, 5.0, 8.0], dtype=np.float32)
    labels = np.asarray([0, 0, 1, 0, 1, 1], dtype=np.float32)
    temperature, bias, metrics = fit_binary_temperature(logits, labels)
    assert temperature > 0.0 and np.isfinite(bias)
    assert metrics["after"]["nll"] < metrics["before"]["nll"]


def test_checkpoint_roundtrip_is_explicitly_not_submission_safe(tmp_path: Path) -> None:
    path = tmp_path / "pair.pt"
    model = _tiny_model()
    model.set_calibration(1.7, -0.3)
    save_pair_transformer_checkpoint(path, model, metadata={"experiment": "unit"})
    loaded, metadata = load_pair_transformer_checkpoint(path)
    assert metadata["safe_for_submission"] is False
    assert loaded.config() == model.config()
    assert float(loaded.calibration_temperature) == pytest.approx(1.7)
    assert float(loaded.calibration_bias) == pytest.approx(-0.3)


def test_atomic_checkpoint_fallback_survives_corrupt_latest_and_next_save(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pair_latest.pt"
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4)
    scheduler = pilot._scheduler(optimizer, total_steps=4, warmup_fraction=0.25)
    rng_state = pilot._capture_rng_state()
    generator_state = torch.Generator().manual_seed(47).get_state()
    contract = {"fixture": "initial_epoch_boundary"}
    resume_bundle = {
        "optimizer_state": pilot._to_cpu_tree(optimizer.state_dict()),
        "scaler_state": {},
        "scheduler_state": pilot._to_cpu_tree(scheduler.state_dict()),
        "training_state": {
            "world_size": 1,
            "cursor": {
                "completed_epoch": -1,
                "next_epoch": 0,
                "source_index": 0,
                "pseudo_cursor": 0,
                "capture_point": "epoch_boundary",
            },
            "optimizer_steps": 0,
            "attempted_steps": 0,
            "amp_skips": 0,
            "rng_states_by_rank": [rng_state],
            "generator_states_by_rank": [generator_state],
            "history": [],
        },
    }
    save_pair_transformer_checkpoint(
        path,
        model,
        metadata={"generation": 1, "resume_contract": contract},
        preserve_previous=True,
        **resume_bundle,
    )
    save_pair_transformer_checkpoint(
        path,
        model,
        metadata={"generation": 2, "resume_contract": contract},
        preserve_previous=True,
        **resume_bundle,
    )
    previous = path.with_name(f"{path.name}.previous")
    assert load_pair_transformer_checkpoint_payload(previous)["metadata"][
        "generation"
    ] == 1

    save_pair_transformer_checkpoint(
        path,
        model,
        metadata={"generation": 99, "resume_contract": contract},
    )
    resume_recovered = load_pair_transformer_checkpoint_payload(
        path, require_training_state=True
    )
    assert resume_recovered["metadata"]["generation"] == 1
    assert resume_recovered["used_previous_fallback"] is True

    torch.save(
        {
            "schema_version": 1,
            "kind": "puzzle_full_tile_pair_transformer",
            "safe_for_submission": False,
            "model_config": {},
            "model_state": {},
            "metadata": {"safe_for_submission": False, "generation": 100},
            **resume_bundle,
        },
        path,
    )
    recovered = load_pair_transformer_checkpoint_payload(path)
    assert recovered["metadata"]["generation"] == 1
    assert recovered["used_previous_fallback"] is True

    save_pair_transformer_checkpoint(
        path,
        model,
        metadata={"generation": 3, "resume_contract": contract},
        preserve_previous=True,
        **resume_bundle,
    )
    latest = load_pair_transformer_checkpoint_payload(path)
    retained = load_pair_transformer_checkpoint_payload(previous)
    assert latest["metadata"]["generation"] == 3
    assert latest["safe_for_submission"] is False
    assert retained["metadata"]["generation"] == 1
    assert not list(tmp_path.glob("*.tmp-*"))


def test_pilot_defaults_are_material_2xt4_and_strictly_held_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_evaluate_pair_transformer.py", "--output-dir", "unused"],
    )
    args = pilot.parse_args()
    assert 256 <= args.train_sources <= 1024
    assert args.epochs >= 3
    assert args.model_dim >= 384 and args.layers >= 6
    assert args.negatives >= 15 and args.candidate_top_k >= 32
    assert args.validation_replicas >= 2
    assert args.calibration_sources > 0 and args.validation_sources > 0
    assert args.affine_probability > 0
    assert args.extra_noise_probability > 0
    assert args.blur_probability > 0
    assert args.jpeg_probability > 0
    assert args.erosion_probability > 0
    assert args.view_dropout > 0


def test_mining_uses_fixed_group_count_for_ddp_and_both_hard_sources() -> None:
    coarse = _coarse()
    positives = pilot.PositiveEdges(
        first=np.asarray([0, 1, 2, 3], dtype=np.int32),
        second=np.asarray([4, 5, 6, 7], dtype=np.int32),
        direction=np.asarray([RIGHT, RIGHT, DOWN, DOWN], dtype=np.int8),
        weight=np.ones(4, dtype=np.float32),
    )
    groups = pilot._mine_groups(
        positives,
        coarse,
        CompatibilityMatrices("visual", coarse.right[::-1].copy(), coarse.down[::-1].copy()),
        rng=np.random.default_rng(19),
        queries=8,
        negatives=7,
        hbt_fraction=0.5,
        visual_fraction=0.25,
    )
    assert groups.first.shape == (8, 8)
    assert groups.second.shape == (8, 8)
    assert np.all(groups.first[:, 0] == groups.first[:, 1])
    assert np.all(groups.second[:, 0] != groups.first[:, 0])
    assert set(groups.direction[:, 0].tolist()) == {RIGHT, DOWN}


def test_task_augmentation_keeps_views_finite_and_supports_both_directions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_evaluate_pair_transformer.py", "--output-dir", "unused"],
    )
    args = pilot.parse_args()
    values = torch.randint(0, 256, (4, 6, 20, 20), dtype=torch.int32).float()
    directions = torch.as_tensor([RIGHT, DOWN, RIGHT, DOWN])
    first = pilot._augment_views(
        values,
        directions,
        endpoint="first",
        args=args,
        generator=torch.Generator().manual_seed(23),
    )
    second = pilot._augment_views(
        values,
        directions,
        endpoint="second",
        args=args,
        generator=torch.Generator().manual_seed(29),
    )
    assert first.shape == second.shape == values.shape
    assert torch.isfinite(first).all() and torch.isfinite(second).all()
    assert 0.0 <= float(first.min()) <= float(first.max()) <= 1.0
    assert 0.0 <= float(second.min()) <= float(second.max()) <= 1.0


def test_repeated_query_is_augmented_once_per_ranking_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_evaluate_pair_transformer.py", "--output-dir", "unused"],
    )
    args = pilot.parse_args()
    bank = torch.randint(0, 256, (8, 6, 20, 20), dtype=torch.int32).float()
    groups = pilot.TrainingGroups(
        first=np.asarray([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.int32),
        second=np.asarray([[2, 3, 4, 5], [2, 3, 6, 7]], dtype=np.int32),
        direction=np.asarray(
            [[RIGHT, RIGHT, RIGHT, RIGHT], [DOWN, DOWN, DOWN, DOWN]],
            dtype=np.int8,
        ),
        weight=np.ones(2, dtype=np.float32),
    )
    first, second, directions = pilot._augment_training_group_batch(
        bank,
        groups,
        0,
        2,
        args=args,
        generator=torch.Generator().manual_seed(31),
    )
    grouped = first.reshape(2, 4, 6, 20, 20)
    torch.testing.assert_close(grouped[:, 1:], grouped[:, :1].expand(-1, 3, -1, -1, -1))
    assert second.shape == first.shape == (8, 6, 20, 20)
    assert directions.tolist() == [RIGHT] * 4 + [DOWN] * 4


def test_training_group_microstep_reports_finite_throughput(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_evaluate_pair_transformer.py", "--output-dir", "unused"],
    )
    args = pilot.parse_args()
    args.groups_per_step = 2
    model = _tiny_model().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = pilot._scheduler(optimizer, total_steps=1, warmup_fraction=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    groups = pilot.TrainingGroups(
        first=np.asarray([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.int32),
        second=np.asarray([[2, 3, 4, 5], [2, 3, 6, 7]], dtype=np.int32),
        direction=np.asarray(
            [[RIGHT, RIGHT, RIGHT, RIGHT], [DOWN, DOWN, DOWN, DOWN]],
            dtype=np.int8,
        ),
        weight=np.ones(2, dtype=np.float32),
    )
    raw = np.zeros((576, 20, 20, 3), dtype=np.uint8)
    denoised = raw.copy()
    metrics = pilot._train_groups(
        model,
        groups,
        raw,
        denoised,
        args=args,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        runtime=pilot.Runtime(0, 1, 0, torch.device("cpu")),
        generator=torch.Generator().manual_seed(33),
        source_weight=1.0,
        amp_skip_state={"count": 0},
    )
    assert np.isfinite(metrics["loss"])
    assert metrics["steps"] == 1.0
    assert metrics["skipped_steps"] == 0.0
    assert metrics["pairs"] == 8.0
    assert metrics["pairs_per_second"] > 0.0


def test_authoritative_eval_partitions_are_upstream_disjoint() -> None:
    manifest_path = Path("configs/denoise_splits_seed20260710.json")
    quarantine_path = Path("configs/denoise_validation_quarantine_v1.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assembly_cal = pilot.source_names_for_split(
        "assembly_cal",
        manifest_path=manifest_path,
        quarantine_path=quarantine_path,
    )
    incremental = pilot.source_names_for_split(
        "assembly_incremental_gate",
        manifest_path=manifest_path,
        quarantine_path=quarantine_path,
    )
    manifest_hash = pilot._sha256(manifest_path)
    quarantine_hash = pilot._sha256(quarantine_path)
    denoiser_metadata = {
        "manifest_sha256": manifest_hash,
        "training_data_sha256": "train-data",
        "validation_data_sha256": "val-data",
    }
    hbt_metadata = {
        "manifest_sha256": manifest_hash,
        "quarantine_sha256": quarantine_hash,
        "train_names": manifest["splits"]["train"][:32],
    }
    audit = pilot._upstream_disjoint_audit(
        quick_names=assembly_cal[:2],
        calibration_names=assembly_cal[2:6],
        validation_names=incremental[:8],
        assembly_cal=assembly_cal,
        assembly_incremental_gate=incremental,
        manifest_path=manifest_path,
        quarantine_path=quarantine_path,
        denoiser_metadata=denoiser_metadata,
        hbt_metadata=hbt_metadata,
    )
    assert audit["all_upstream_overlaps_zero"] is True
    assert set(audit["overlap_counts"].values()) == {0}
    assert audit["quick_partition"] == "assembly_cal"
    assert audit["holdout_partition"] == "assembly_incremental_gate"

    leaked_hbt = {**hbt_metadata, "train_names": [assembly_cal[0]]}
    with pytest.raises(RuntimeError, match="HBT train provenance|overlaps frozen upstream"):
        pilot._upstream_disjoint_audit(
            quick_names=assembly_cal[:2],
            calibration_names=assembly_cal[2:6],
            validation_names=incremental[:8],
            assembly_cal=assembly_cal,
            assembly_incremental_gate=incremental,
            manifest_path=manifest_path,
            quarantine_path=quarantine_path,
            denoiser_metadata=denoiser_metadata,
            hbt_metadata=leaked_hbt,
        )


def test_equal_budget_control_runs_one_qap_per_neural_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_evaluate_pair_transformer.py", "--output-dir", "unused"],
    )
    args = pilot.parse_args()
    args.iterative_passes = 3
    calls: list[tuple[int, int]] = []

    def fake_qap(
        score: CompatibilityMatrices,
        initial: np.ndarray,
        *,
        seed: int,
        args: object,
        iterations: int | None = None,
        restarts: int | None = None,
    ) -> np.ndarray:
        calls.append((seed, int(iterations or 0)))
        return np.asarray(initial, dtype=np.int32).copy()

    monkeypatch.setattr(pilot, "_qap_layout", fake_qap)
    layout = np.arange(576, dtype=np.int32)
    output = pilot._equal_budget_no_neural_control(
        _coarse(), layout, seed=37, args=args
    )
    np.testing.assert_array_equal(output, layout)
    assert calls == [(37, 0), (37, 0), (37, 0)]
    assert pilot.PROMOTED_QAP_ITERATIONS == 25
    assert pilot.PROMOTED_QAP_RESTARTS == 2
    source = _SCRIPT.read_text(encoding="utf-8")
    assert '"promoted_w1_i25_ssim"' in source
    assert '"promoted_w4_i25_ssim"' in source
    assert "qap_w1_b0.05_i25" in source
    assert "strongest_known_promoted_comparator" in source


def _random_full_resume_step(
    model: PairTransformerScorer,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    generator: torch.Generator,
) -> torch.Tensor:
    optimizer.zero_grad(set_to_none=True)
    generator_draw = torch.rand((), generator=generator)
    loss = torch.zeros(())
    for parameter in model.parameters():
        loss = loss + (parameter * (torch.randn_like(parameter) + generator_draw)).mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    return generator_draw


def test_full_resume_restores_optimizer_scaler_scheduler_generator_and_rng(
    tmp_path: Path,
) -> None:
    pilot.random.seed(43)
    np.random.seed(43)
    torch.manual_seed(43)
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = pilot._scheduler(optimizer, total_steps=4, warmup_fraction=0.25)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    generator = torch.Generator().manual_seed(47)
    _random_full_resume_step(model, optimizer, scheduler, generator)
    rng_state = pilot._capture_rng_state()
    generator_state = generator.get_state().cpu()
    contract = {"test": "exact"}
    path = tmp_path / "latest.pt"
    save_pair_transformer_checkpoint(
        path,
        model,
        metadata={"resume_contract": contract},
        optimizer_state=pilot._to_cpu_tree(optimizer.state_dict()),
        scaler_state=pilot._to_cpu_tree(scaler.state_dict()),
        scheduler_state=pilot._to_cpu_tree(scheduler.state_dict()),
        training_state={
            "world_size": 1,
            "cursor": {
                "completed_epoch": 0,
                "next_epoch": 1,
                "source_index": 0,
                "pseudo_cursor": 0,
                "capture_point": "epoch_boundary",
            },
            "optimizer_steps": 1,
            "attempted_steps": 1,
            "amp_skips": 0,
            "rng_states_by_rank": [rng_state],
            "generator_states_by_rank": [generator_state],
        },
    )
    expected_draw = _random_full_resume_step(model, optimizer, scheduler, generator)
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
    expected_scheduler = scheduler.state_dict()

    payload = load_pair_transformer_checkpoint_payload(
        path, require_training_state=True
    )
    runtime = pilot.Runtime(0, 1, 0, torch.device("cpu"))
    state = pilot._validate_resume_payload(
        payload, expected_contract=contract, runtime=runtime
    )
    restored = _tiny_model()
    restored.load_state_dict(payload["model_state"], strict=True)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=3e-4)
    restored_optimizer.load_state_dict(payload["optimizer_state"])
    restored_scheduler = pilot._scheduler(
        restored_optimizer, total_steps=4, warmup_fraction=0.25
    )
    restored_scheduler.load_state_dict(payload["scheduler_state"])
    restored_scaler = torch.amp.GradScaler("cuda", enabled=False)
    restored_scaler.load_state_dict(payload["scaler_state"])
    restored_generator = torch.Generator()
    restored_generator.set_state(state["generator_states_by_rank"][0])
    pilot._restore_rng_state(state["rng_states_by_rank"][0])
    restored_draw = _random_full_resume_step(
        restored, restored_optimizer, restored_scheduler, restored_generator
    )
    torch.testing.assert_close(restored_draw, expected_draw, atol=0.0, rtol=0.0)
    for name, value in restored.state_dict().items():
        torch.testing.assert_close(value, expected[name], atol=0.0, rtol=0.0)
    assert restored_scheduler.state_dict() == expected_scheduler

    with pytest.raises(ValueError, match="config/hash contract"):
        pilot._validate_resume_payload(
            payload, expected_contract={"test": "drift"}, runtime=runtime
        )


def test_t4_preflight_and_rank0_eval_teardown_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_evaluate_pair_transformer.py", "--output-dir", "unused"],
    )
    args = pilot.parse_args()
    runtime = pilot.Runtime(0, 1, 0, torch.device("cpu"))
    preflight = pilot._bounded_t4_preflight(_tiny_model(), args, runtime)
    assert preflight["pairs_per_microstep"] == args.groups_per_step * (
        args.negatives + 1
    )
    assert preflight["fits_bounded_t4_envelope"] is True
    real = pilot._real_cuda_microstep_preflight(_tiny_model(), args, runtime)
    assert real == {"executed": False, "reason": "CUDA unavailable"}
    source = _SCRIPT.read_text(encoding="utf-8")
    teardown = source.index("dist.destroy_process_group()")
    calibration = source.index("calibration = _calibrate(")
    assert teardown < calibration
    assert "assembly_incremental_gate" in source
    assert "delta_ssim_vs_no_neural_envelope" in source


def test_report_contract_keeps_research_checkpoint_unsafe() -> None:
    source = _SCRIPT.read_text(encoding="utf-8")
    assert '"safe_for_submission": False' in source
    assert "freeze real16 layouts before opening targets" in source
    assert "global layout-energy ViT" in source
    assert "qap_like_negative_recipe" in source


def _write_runner_artifacts(
    output: Path, *, continuation: bool = False, world_size: int = 2
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    model_source = Path(__file__).resolve().parents[1] / "src/puzzle_assembly/pair_transformer.py"
    model = _tiny_model()
    runtime_contracts = [
        {
            "rank": rank,
            "device_type": "cuda",
            "torch": torch.__version__,
            "cuda_runtime": "test",
            "amp": True,
            "amp_dtype": "torch.float16",
            "gpu": "Tesla T4",
            "capability": [7, 5],
            "total_memory": 16 * 1024**3,
        }
        for rank in range(world_size)
    ]
    resume_contract = {
        "model_config": model.config(),
        "trajectory_arguments": {
            "epochs": 1,
            "learning_rate": 3.0e-4,
            "weight_decay": 1.0e-4,
            "amp_init_scale": 1024.0,
            "max_amp_skips": 4,
            "no_amp": False,
        },
        "runtime_contracts_by_rank": runtime_contracts,
    }
    provenance = {
        "schema_version": 1,
        "kind": "pair_transformer_training_provenance",
        "args": dict(resume_contract["trajectory_arguments"]),
        "seed": 20260711,
        "code": {
            "model": str(model_source.resolve()),
            "model_sha256": runner.sha256(model_source),
        },
        "assets": {
            "denoiser_sha256": "1" * 64,
            "hbt_sha256": "2" * 64,
            "pseudo": {"sha256": "3" * 64},
        },
        "whole_source_splits": {"pairwise_disjoint": True},
        "resume_contract": resume_contract,
    }
    history = [{"epoch": 1, "optimizer_steps": 1, "attempted_steps": 1}]
    best = output / "pair_transformer_best.pt"
    calibrated = output / "pair_transformer_calibrated.pt"
    save_pair_transformer_checkpoint(
        best,
        model,
        metadata={
            **provenance,
            "training_history": history,
            "best_epoch": 1,
        },
    )
    save_pair_transformer_checkpoint(
        calibrated,
        model,
        metadata={**provenance, "calibration": {"fixture": True}},
    )
    best_hash = runner.sha256(best)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    loss = sum(parameter.square().mean() for parameter in model.parameters())
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    rng_state = pilot._capture_rng_state()
    rng_state["torch_cuda"] = [torch.arange(8, dtype=torch.uint8)]
    generator_state = torch.Generator().manual_seed(47).get_state()
    training_state = {
        "world_size": world_size,
        "cursor": {
            "completed_epoch": 0,
            "next_epoch": 1,
            "source_index": 0,
            "pseudo_cursor": 0,
            "capture_point": "epoch_boundary",
        },
        "optimizer_steps": 1,
        "attempted_steps": 1,
        "amp_skips": 0,
        "rng_states_by_rank": [pilot._to_cpu_tree(rng_state) for _ in range(world_size)],
        "generator_states_by_rank": [generator_state.clone() for _ in range(world_size)],
        "history": history,
        "best_delta": 0.1,
        "best_checkpoint_sha256": best_hash,
    }
    latest = output / "pair_transformer_latest.pt"
    save_pair_transformer_checkpoint(
        latest,
        model,
        metadata={
            **provenance,
            "training_history": history,
            "latest_completed_epoch": 1,
        },
        optimizer_state=pilot._to_cpu_tree(optimizer.state_dict()),
        scaler_state={
            "scale": 1024.0,
            "growth_factor": 2.0,
            "backoff_factor": 0.5,
            "growth_interval": 2000,
            "_growth_tracker": 1,
        },
        scheduler_state=pilot._to_cpu_tree(scheduler.state_dict()),
        training_state=training_state,
    )
    gates = {key: continuation for key in runner.EXPECTED_GATE_KEYS}
    report = {
        "schema_version": 1,
        "kind": "pair_transformer_2xt4_pilot",
        "status": "continue" if continuation else "stop_or_redesign",
        "safe_for_submission": False,
        "provenance": provenance,
        "training_history": history,
        "best_checkpoint": str(best.resolve()),
        "best_checkpoint_sha256": best_hash,
        "checkpoint": str(calibrated.resolve()),
        "checkpoint_sha256": runner.sha256(calibrated),
        "training_telemetry": {
            "best_checkpoint": str(best.resolve()),
            "best_checkpoint_sha256": best_hash,
            "latest_checkpoint": str(latest.resolve()),
            "latest_checkpoint_sha256": runner.sha256(latest),
        },
        "evaluation": {
            "continuation_gates": gates,
            "continue_to_1024_source_two_seed_run": continuation,
        },
    }
    report_path = output / "pair_transformer_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifact_paths = {
        "pair_transformer_report.json": report_path,
        "pair_transformer_calibrated.pt": calibrated,
        "pair_transformer_best.pt": output / "pair_transformer_best.pt",
        "pair_transformer_latest.pt": latest,
    }
    (output / "SHA256SUMS.txt").write_text(
        "".join(
            f"{runner.sha256(artifact_paths[name])}  {name}\n"
            for name in runner.CHECKSUM_ARTIFACTS
        ),
        encoding="utf-8",
    )


def test_runner_accepts_complete_fail_closed_artifact_contract(tmp_path: Path) -> None:
    _write_runner_artifacts(tmp_path, continuation=True)
    audit = runner.validate_pilot_artifacts(
        tmp_path,
        "unit",
        pinned_model_config=_tiny_model().config(),
    )
    assert audit["status"] == "continue"
    assert audit["continue_to_1024_source_two_seed_run"] is True
    assert set(audit["checkpoint_contracts"]) == {
        "calibrated_checkpoint",
        "best_checkpoint",
        "latest_checkpoint",
    }
    assert tuple(audit["sha256_manifest"]) == runner.CHECKSUM_ARTIFACTS


def test_runner_rejects_adversarial_artifact_contracts(tmp_path: Path) -> None:
    mutations = (
        ("unsupported_status", lambda report: report.__setitem__("status", "complete")),
        (
            "non_boolean_gate",
            lambda report: report["evaluation"]["continuation_gates"].__setitem__(
                next(iter(runner.EXPECTED_GATE_KEYS)), 1
            ),
        ),
        (
            "continuation_disagrees",
            lambda report: report["evaluation"].__setitem__(
                "continue_to_1024_source_two_seed_run", True
            ),
        ),
        ("status_disagrees", lambda report: report.__setitem__("status", "continue")),
        (
            "calibrated_hash_disagrees",
            lambda report: report.__setitem__("checkpoint_sha256", "0" * 64),
        ),
        (
            "latest_hash_disagrees",
            lambda report: report["training_telemetry"].__setitem__(
                "latest_checkpoint_sha256", "0" * 64
            ),
        ),
        (
            "best_hash_disagrees",
            lambda report: report.__setitem__("best_checkpoint_sha256", "0" * 64),
        ),
    )
    for name, mutate in mutations:
        output = tmp_path / name
        _write_runner_artifacts(output, continuation=False)
        report_path = output / "pair_transformer_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        mutate(report)
        report_path.write_text(json.dumps(report) + "\n", encoding="utf-8")
        with pytest.raises(RuntimeError):
            runner.validate_pilot_artifacts(output, name)

    checksum_output = tmp_path / "bad_checksum"
    _write_runner_artifacts(checksum_output)
    hashes = checksum_output / "SHA256SUMS.txt"
    hashes.write_text(hashes.read_text(encoding="utf-8") + "0" * 64 + "  extra.pt\n")
    with pytest.raises(RuntimeError, match="exactly four"):
        runner.validate_pilot_artifacts(checksum_output, "bad checksum")

    unsafe_output = tmp_path / "unsafe_checkpoint"
    _write_runner_artifacts(unsafe_output)
    unsafe_path = unsafe_output / "pair_transformer_calibrated.pt"
    payload = torch.load(unsafe_path, map_location="cpu", weights_only=False)
    payload["safe_for_submission"] = True
    torch.save(payload, unsafe_path)
    report_path = unsafe_output / "pair_transformer_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["checkpoint_sha256"] = runner.sha256(unsafe_path)
    report_path.write_text(json.dumps(report) + "\n", encoding="utf-8")
    artifact_paths = {
        "pair_transformer_report.json": report_path,
        "pair_transformer_calibrated.pt": unsafe_path,
        "pair_transformer_best.pt": unsafe_output / "pair_transformer_best.pt",
        "pair_transformer_latest.pt": unsafe_output / "pair_transformer_latest.pt",
    }
    (unsafe_output / "SHA256SUMS.txt").write_text(
        "".join(
            f"{runner.sha256(artifact_paths[name])}  {name}\n"
            for name in runner.CHECKSUM_ARTIFACTS
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="schema/safety"):
        runner.validate_pilot_artifacts(unsafe_output, "unsafe checkpoint")

    def refresh(output: Path, report: dict[str, object]) -> None:
        report_path = output / "pair_transformer_report.json"
        report_path.write_text(json.dumps(report) + "\n", encoding="utf-8")
        artifact_paths = {
            "pair_transformer_report.json": report_path,
            "pair_transformer_calibrated.pt": output
            / "pair_transformer_calibrated.pt",
            "pair_transformer_best.pt": output / "pair_transformer_best.pt",
            "pair_transformer_latest.pt": output / "pair_transformer_latest.pt",
        }
        (output / "SHA256SUMS.txt").write_text(
            "".join(
                f"{runner.sha256(artifact_paths[name])}  {name}\n"
                for name in runner.CHECKSUM_ARTIFACTS
            ),
            encoding="utf-8",
        )

    empty_state_output = tmp_path / "empty_model_state"
    _write_runner_artifacts(empty_state_output)
    calibrated = empty_state_output / "pair_transformer_calibrated.pt"
    payload = torch.load(calibrated, map_location="cpu", weights_only=False)
    payload["model_state"] = {}
    torch.save(payload, calibrated)
    report = json.loads(
        (empty_state_output / "pair_transformer_report.json").read_text()
    )
    report["checkpoint_sha256"] = runner.sha256(calibrated)
    refresh(empty_state_output, report)
    with pytest.raises(RuntimeError, match="strictly loadable"):
        runner.validate_pilot_artifacts(empty_state_output, "empty model state")

    empty_config_output = tmp_path / "empty_model_config"
    _write_runner_artifacts(empty_config_output)
    calibrated = empty_config_output / "pair_transformer_calibrated.pt"
    payload = torch.load(calibrated, map_location="cpu", weights_only=False)
    payload["model_config"] = {}
    torch.save(payload, calibrated)
    report = json.loads(
        (empty_config_output / "pair_transformer_report.json").read_text()
    )
    report["checkpoint_sha256"] = runner.sha256(calibrated)
    refresh(empty_config_output, report)
    with pytest.raises(RuntimeError, match="model config differs"):
        runner.validate_pilot_artifacts(empty_config_output, "empty model config")

    missing_resume_output = tmp_path / "missing_resume_bundle"
    _write_runner_artifacts(missing_resume_output)
    latest = missing_resume_output / "pair_transformer_latest.pt"
    payload = torch.load(latest, map_location="cpu", weights_only=False)
    payload.pop("scheduler_state")
    torch.save(payload, latest)
    report = json.loads(
        (missing_resume_output / "pair_transformer_report.json").read_text()
    )
    report["training_telemetry"]["latest_checkpoint_sha256"] = runner.sha256(
        latest
    )
    refresh(missing_resume_output, report)
    with pytest.raises(RuntimeError, match="resume bundle is incomplete"):
        runner.validate_pilot_artifacts(missing_resume_output, "missing resume")


def test_runner_records_failed_command_and_atomically_writes_wrapper(
    tmp_path: Path,
) -> None:
    telemetry: list[dict[str, object]] = []
    with pytest.raises(RuntimeError, match="return code 7"):
        runner.run_checked(
            [sys.executable, "-c", "raise SystemExit(7)"],
            cwd=tmp_path,
            environment=dict(os.environ),
            label="expected_failure",
            capture=True,
            telemetry=telemetry,
        )
    assert len(telemetry) == 1
    assert telemetry[0]["status"] == "failed"
    assert telemetry[0]["returncode"] == 7

    wrapper_path = tmp_path / "wrapper.json"
    runner.atomic_write_json(
        wrapper_path,
        {"status": "error", "safe_for_submission": False, "commands": telemetry},
    )
    assert json.loads(wrapper_path.read_text(encoding="utf-8"))["commands"] == telemetry
    assert not list(tmp_path.glob("*.tmp-*"))

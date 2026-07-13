from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


_PILOT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/train_evaluate_positional_diffusion.py"
)
_PILOT_SPEC = importlib.util.spec_from_file_location(
    "positional_diffusion_pilot", _PILOT_PATH
)
assert _PILOT_SPEC is not None and _PILOT_SPEC.loader is not None
pilot = importlib.util.module_from_spec(_PILOT_SPEC)
sys.modules[_PILOT_SPEC.name] = pilot
_PILOT_SPEC.loader.exec_module(pilot)

from puzzle_assembly.positional_diffusion import (
    GaussianPositionDiffusion,
    PositionalDiffusionConfig,
    PositionalDiffusionNet,
    compatibility_to_relative_graph,
    estimate_peak_memory_bytes,
    layout_to_tile_positions,
    load_positional_diffusion_checkpoint,
    load_positional_diffusion_checkpoint_payload,
    normalized_grid_positions,
    project_positions_hungarian,
    save_positional_diffusion_checkpoint,
)


def _tiny_config() -> PositionalDiffusionConfig:
    return PositionalDiffusionConfig(
        model_dim=32,
        cnn_channels=8,
        layers=1,
        heads=4,
        feedforward_dim=64,
        dropout=0.0,
        diffusion_steps=8,
        tile_encode_chunk=4,
        activation_checkpointing=False,
    )


def _random_graph(size: int, seed: int = 7) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    graph = torch.rand(1, 2, size, size, generator=generator)
    diagonal = torch.arange(size)
    graph[:, :, diagonal, diagonal] = 0.0
    return graph / graph.sum(dim=-1, keepdim=True)


def test_grid_layout_round_trip_uses_xy_row_major_convention() -> None:
    grid = normalized_grid_positions(2, 3)
    expected = torch.tensor(
        [
            [-1.0, -1.0],
            [0.0, -1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    torch.testing.assert_close(grid, expected)
    layout = np.asarray([4, 0, 5, 2, 1, 3], dtype=np.int32)
    tile_positions = layout_to_tile_positions(layout, 2, 3)
    projected = project_positions_hungarian(tile_positions, 2, 3)
    np.testing.assert_array_equal(projected.position_to_tile, layout)
    assert projected.squared_assignment_cost == pytest.approx(0.0)


def test_hungarian_projection_is_global_and_returns_both_permutations() -> None:
    layout = np.asarray([2, 0, 3, 1], dtype=np.int32)
    positions = layout_to_tile_positions(layout, 2, 2)
    positions = positions + torch.tensor(
        [[0.04, -0.02], [-0.03, 0.02], [0.01, 0.03], [-0.02, -0.04]]
    )
    result = project_positions_hungarian(positions, 2, 2)
    np.testing.assert_array_equal(result.position_to_tile, layout)
    np.testing.assert_array_equal(
        result.tile_to_position[result.position_to_tile], np.arange(4)
    )


def test_hbt_cost_graph_is_sparse_stochastic_and_permutation_equivariant() -> None:
    rng = np.random.default_rng(11)
    right = rng.normal(size=(7, 7)).astype(np.float32)
    down = rng.normal(size=(7, 7)).astype(np.float32)
    np.fill_diagonal(right, np.inf)
    np.fill_diagonal(down, np.inf)
    graph = compatibility_to_relative_graph(right, down, top_k=3, temperature=0.4)
    assert graph.shape == (2, 7, 7)
    np.testing.assert_allclose(graph.sum(axis=2), 1.0, atol=1e-6)
    assert np.all(np.count_nonzero(graph, axis=2) == 3)
    order = np.asarray([3, 0, 6, 1, 5, 2, 4])
    permuted = compatibility_to_relative_graph(
        right[order][:, order], down[order][:, order], top_k=3, temperature=0.4
    )
    np.testing.assert_allclose(
        permuted, graph[:, order][:, :, order], atol=1e-7
    )


def test_attention_gnn_is_permutation_equivariant_with_graph_and_warm_layout() -> None:
    torch.manual_seed(13)
    model = PositionalDiffusionNet(_tiny_config()).eval()
    size = 6
    raw = torch.rand(1, size, 3, 8, 8)
    restored = torch.rand(1, size, 3, 8, 8)
    noisy = torch.randn(1, size, 2)
    baseline = torch.randn(1, size, 2)
    graph = _random_graph(size)
    timesteps = torch.tensor([5])
    order = torch.as_tensor([3, 0, 5, 1, 4, 2])
    with torch.inference_mode():
        reference = model(
            raw,
            restored,
            noisy,
            timesteps,
            relative_graph=graph,
            baseline_positions=baseline,
        )
        permuted = model(
            raw[:, order],
            restored[:, order],
            noisy[:, order],
            timesteps,
            relative_graph=graph[:, :, order][:, :, :, order],
            baseline_positions=baseline[:, order],
        )
    torch.testing.assert_close(permuted, reference[:, order], atol=2e-5, rtol=2e-5)


def test_training_loss_is_finite_and_backpropagates_on_small_grid() -> None:
    torch.manual_seed(17)
    config = _tiny_config()
    model = PositionalDiffusionNet(config).train()
    diffusion = GaussianPositionDiffusion(config)
    size = 9
    raw = torch.rand(1, size, 3, 8, 8)
    restored = torch.rand(1, size, 3, 8, 8)
    permutation = torch.randperm(size)
    target = normalized_grid_positions(3, 3)[permutation].unsqueeze(0)
    baseline = normalized_grid_positions(3, 3)[torch.randperm(size)].unsqueeze(0)
    graph = _random_graph(size)
    noise = torch.randn(1, size, 2)
    losses = diffusion.training_loss(
        model,
        raw,
        restored,
        target,
        rows=3,
        columns=3,
        relative_graph=graph,
        baseline_positions=baseline,
        timesteps=torch.tensor([4]),
        noise=noise,
        structure_weight=0.2,
    )
    assert torch.isfinite(losses["loss"])
    assert torch.isfinite(losses["position_loss"])
    assert torch.isfinite(losses["structure_loss"])
    losses["loss"].backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_activation_checkpointed_encoder_and_transformer_backpropagate() -> None:
    torch.manual_seed(18)
    config = PositionalDiffusionConfig(
        model_dim=32,
        cnn_channels=8,
        layers=1,
        heads=4,
        feedforward_dim=64,
        dropout=0.0,
        diffusion_steps=8,
        tile_encode_chunk=2,
        activation_checkpointing=True,
    )
    model = PositionalDiffusionNet(config).train()
    output = model(
        torch.rand(1, 4, 3, 8, 8),
        torch.rand(1, 4, 3, 8, 8),
        torch.randn(1, 4, 2),
        torch.tensor([3]),
        relative_graph=_random_graph(4),
        baseline_positions=torch.zeros(1, 4, 2),
    )
    output.square().mean().backward()
    assert model.tile_encoder.stem[0].weight.grad is not None
    assert model.layers[0].self_attn.in_proj_weight.grad is not None


def test_ddim_input_layout_start_is_deterministic_and_projects_validly() -> None:
    torch.manual_seed(19)
    config = _tiny_config()
    model = PositionalDiffusionNet(config).eval()
    diffusion = GaussianPositionDiffusion(config)
    raw = torch.rand(1, 4, 3, 8, 8)
    restored = torch.rand(1, 4, 3, 8, 8)
    graph = _random_graph(4)
    layout = np.asarray([2, 0, 3, 1], dtype=np.int32)
    baseline = layout_to_tile_positions(layout, 2, 2).unsqueeze(0)
    first = diffusion.ddim_sample(
        model,
        raw,
        restored,
        rows=2,
        columns=2,
        relative_graph=graph,
        baseline_positions=baseline,
        sampling_steps=4,
        initialization="input_layout",
        seed=23,
    )
    second = diffusion.ddim_sample(
        model,
        raw,
        restored,
        rows=2,
        columns=2,
        relative_graph=graph,
        baseline_positions=baseline,
        sampling_steps=4,
        initialization="input_layout",
        seed=23,
    )
    generator = torch.Generator(device="cpu").manual_seed(23)
    epsilon = torch.randn(
        baseline.shape, generator=generator, dtype=baseline.dtype
    )
    start_t = first.sampling_timesteps[0]
    expected_initial = (
        diffusion.sqrt_alpha_bars[start_t] * baseline
        + diffusion.sqrt_one_minus_alpha_bars[start_t] * epsilon
    )
    torch.testing.assert_close(first.initial_positions, expected_initial)
    assert float(first.initial_positions.std()) > 0.25
    torch.testing.assert_close(first.positions, second.positions)
    torch.testing.assert_close(first.initial_positions, second.initial_positions)
    assert torch.isfinite(first.positions).all()
    np.testing.assert_array_equal(
        np.sort(first.projections[0].position_to_tile), np.arange(4)
    )
    assert first.initialization == "input_layout"
    assert first.sampling_timesteps[0] == config.diffusion_steps - 1
    assert first.sampling_timesteps[-1] == 0


def test_checkpoint_round_trip_preserves_config_weights_and_metadata(tmp_path: Path) -> None:
    torch.manual_seed(29)
    model = PositionalDiffusionNet(_tiny_config()).eval()
    path = tmp_path / "posdiff.pt"
    save_positional_diffusion_checkpoint(
        path,
        model,
        metadata={"targets_opened": False, "seed": 29},
    )
    loaded, diffusion, metadata = load_positional_diffusion_checkpoint(path)
    assert loaded.config == model.config
    assert diffusion.config == model.config
    assert metadata == {"targets_opened": False, "seed": 29}
    for name, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value)


def test_atomic_checkpoint_retains_and_recovers_previous(tmp_path: Path) -> None:
    model = PositionalDiffusionNet(_tiny_config()).eval()
    path = tmp_path / "latest.pt"
    save_positional_diffusion_checkpoint(
        path, model, metadata={"generation": 1}, preserve_previous=True
    )
    save_positional_diffusion_checkpoint(
        path, model, metadata={"generation": 2}, preserve_previous=True
    )
    previous = path.with_name(f"{path.name}.previous")
    assert previous.is_file()
    assert load_positional_diffusion_checkpoint_payload(previous)["metadata"]["generation"] == 1
    path.write_bytes(b"truncated")
    recovered = load_positional_diffusion_checkpoint_payload(path)
    assert recovered["metadata"]["generation"] == 1
    assert recovered["used_previous_fallback"] is True
    assert recovered["safe_for_submission"] is False
    previous_bytes = previous.read_bytes()

    save_positional_diffusion_checkpoint(
        path, model, metadata={"generation": 3}, preserve_previous=True
    )
    assert previous.read_bytes() == previous_bytes
    assert (
        load_positional_diffusion_checkpoint_payload(previous)["metadata"]["generation"]
        == 1
    )
    latest = load_positional_diffusion_checkpoint_payload(path)
    assert latest["metadata"]["generation"] == 3
    assert latest["used_previous_fallback"] is False


def _random_optimizer_step(
    model: PositionalDiffusionNet, optimizer: torch.optim.Optimizer
) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss = torch.zeros(())
    for parameter in model.parameters():
        loss = loss + (parameter * torch.randn_like(parameter)).mean()
    loss.backward()
    optimizer.step()


def test_full_checkpoint_rng_optimizer_continuation_is_exact(tmp_path: Path) -> None:
    pilot._seed_everything(101, 0)
    model = PositionalDiffusionNet(_tiny_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    _random_optimizer_step(model, optimizer)
    rng_state = pilot._capture_rng_state()
    runtime = pilot.Runtime(torch.device("cpu"), 0, 0, 1, False)
    runtime_contracts = pilot._gather_runtime_resume_contracts(
        runtime,
        amp_enabled=False,
        amp_dtype=torch.float32,
    )
    path = tmp_path / "resume.pt"
    save_positional_diffusion_checkpoint(
        path,
        model,
        metadata={"seed": 101},
        optimizer_state=pilot._to_cpu_tree(optimizer.state_dict()),
        scaler_state=pilot._to_cpu_tree(scaler.state_dict()),
        training_state={
            "world_size": 1,
            "gradient_accumulation": 1,
            "completed_epoch": 0,
            "next_epoch": 1,
            "optimizer_steps": 1,
            "rng_states_by_rank": [rng_state],
            "runtime_contracts_by_rank": runtime_contracts,
            "capture_point": "epoch boundary after optimizer update and before checkpoint save",
        },
    )

    _random_optimizer_step(model, optimizer)
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}

    payload = load_positional_diffusion_checkpoint_payload(path)
    restored = PositionalDiffusionNet(_tiny_config())
    restored.load_state_dict(payload["model_state"], strict=True)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=3e-4)
    restored_optimizer.load_state_dict(payload["optimizer_state"])
    restored_scaler = torch.amp.GradScaler("cuda", enabled=False)
    restored_scaler.load_state_dict(payload["scaler_state"])
    pilot._restore_rng_state(payload["training_state"]["rng_states_by_rank"][0])
    _random_optimizer_step(restored, restored_optimizer)

    for name, value in restored.state_dict().items():
        torch.testing.assert_close(value, expected[name], atol=0.0, rtol=0.0)


def test_memory_estimate_scales_quadratically_with_tile_count() -> None:
    config = _tiny_config()
    small = estimate_peak_memory_bytes(config, batch_size=1, tile_count=16)
    large = estimate_peak_memory_bytes(config, batch_size=1, tile_count=64)
    assert large["attention_logits"] == 16 * small["attention_logits"]
    assert large["estimated_peak_activations"] > small["estimated_peak_activations"]


def test_pilot_defaults_are_2xt4_bounded_and_gate_nonzero_cross_corruption() -> None:
    args = pilot._parser().parse_args(["--output-dir", "unused"])
    assert args.model_dim >= 256
    assert args.layers >= 6
    assert args.train_sources >= 384
    assert args.gradient_accumulation >= 4
    assert args.max_optimizer_steps <= 256
    assert args.diffusion_steps == 300
    assert args.sampling_steps == 30
    assert args.amp == "fp16"
    assert args.amp_init_scale <= 4096.0
    assert args.amp_max_consecutive_skips >= 1
    assert args.warm_start_layout == "softcycle"
    assert args.dev_split == "assembly_incremental_gate"
    assert args.dev_sources >= 8
    assert args.dev_replicas >= 2
    assert pilot.DEFAULT_HBT.endswith("hbt_d320_denoised_rgb_sobel.pt")
    assert args.gate_min_adjacency_gain > 0.0
    assert args.gate_min_ssim_gain > 0.0
    assert "target" not in inspect.signature(pilot._input_only_evidence).parameters


def test_early_seeding_reproduces_model_initialization_and_precedes_main_model() -> None:
    pilot._seed_everything(73, 0)
    first = PositionalDiffusionNet(_tiny_config())
    pilot._seed_everything(73, 0)
    second = PositionalDiffusionNet(_tiny_config())
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name], atol=0.0, rtol=0.0)
    source = inspect.getsource(pilot.main)
    assert source.index("_seed_everything") < source.index("PositionalDiffusionNet(config)")


def test_authoritative_filename_qap_seed_and_warm_layout_ablation() -> None:
    name = "img_000123.png"
    expected = int.from_bytes(
        __import__("hashlib").sha256(name.encode("utf-8")).digest()[:4], "little"
    ) + 7001
    assert pilot._filename_qap_seed(name) == expected
    identity = np.arange(576, dtype=np.int32)
    reverse = identity[::-1].copy()
    evidence = pilot.InputOnlyEvidence(
        graph=np.zeros((2, 576, 576), dtype=np.float32),
        hbt_score=None,
        soft_layout=identity,
        w4_qap_layout=reverse,
    )
    np.testing.assert_array_equal(
        pilot._select_warm_layout(evidence, "softcycle"), identity
    )
    np.testing.assert_array_equal(
        pilot._select_warm_layout(evidence, "w4-qap"), reverse
    )


def test_training_qap_mode_runs_only_selected_warm_solver(monkeypatch: pytest.MonkeyPatch) -> None:
    identity = np.arange(576, dtype=np.int32)
    hbt = SimpleNamespace(
        name="input_only_hbt",
        right=np.zeros((576, 576), dtype=np.float32),
        down=np.zeros((576, 576), dtype=np.float32),
    )
    soft = SimpleNamespace(
        position_to_slot=identity,
        accepted_edges=0,
        proposed_edges=0,
        component_sizes=[576],
    )
    monkeypatch.setattr(pilot, "learned_compatibility", lambda *a, **k: (hbt, None))
    monkeypatch.setattr(
        pilot,
        "compatibility_to_relative_graph",
        lambda *a, **k: np.zeros((2, 576, 576), dtype=np.float32),
    )
    monkeypatch.setattr(pilot, "soft_cycle_component_solver", lambda *a, **k: soft)
    c1 = SimpleNamespace(name="denoised_C1_equal_rank")
    monkeypatch.setattr(
        pilot,
        "build_classical_score_bank",
        lambda *a, **k: {"denoised_rgb": SimpleNamespace(name="denoised_rgb")},
    )

    def fake_fuse(*args, name: str, **kwargs):
        return c1 if name == c1.name else SimpleNamespace(name=name)

    monkeypatch.setattr(pilot, "fuse_ranked_scores", fake_fuse)
    calls: list[str] = []

    def fake_qap(score, **kwargs):
        calls.append(score.name)
        return SimpleNamespace(objective=0.0, restart=0, position_to_slot=identity)

    monkeypatch.setattr(pilot, "directional_qap", fake_qap)
    args = pilot._parser().parse_args(["--output-dir", "unused"])
    pilot._input_only_evidence(
        np.zeros((576, 20, 20, 3), dtype=np.uint8),
        hbt_model=torch.nn.Identity(),
        device=torch.device("cpu"),
        args=args,
        source_name="img_000001.png",
        qap_mode="w4",
    )
    assert calls == ["input_only_C1_HBTw4"]
    calls.clear()
    evidence = pilot._input_only_evidence(
        np.zeros((576, 20, 20, 3), dtype=np.uint8),
        hbt_model=torch.nn.Identity(),
        device=torch.device("cpu"),
        args=args,
        source_name="img_000001.png",
        qap_mode="comparators",
    )
    assert calls == ["input_only_C1_HBTw4", "input_only_C1_HBTw1", "input_only_hbt"]
    assert evidence.w1_qap_layout is not None


def test_finite_guard_and_transitive_provenance_contract() -> None:
    runtime = pilot.Runtime(torch.device("cpu"), 0, 0, 1, False)
    assert pilot._all_ranks_finite(torch.tensor(1.0), runtime)
    assert not pilot._all_ranks_finite(torch.tensor(float("nan")), runtime)
    names = {path.name for path in pilot.CODE_PATHS}
    assert {"geometry.py", "solvers.py", "degradation.py", "model.py", "tiles.py"} <= names


def test_bounded_amp_skip_recovery_fails_only_after_limit() -> None:
    total = consecutive = 0
    for _ in range(3):
        total, consecutive, exceeded = pilot._bounded_skip_state(
            total_skips=total,
            consecutive_skips=consecutive,
            skipped=True,
            max_total=8,
            max_consecutive=3,
        )
        assert not exceeded
    total, consecutive, exceeded = pilot._bounded_skip_state(
        total_skips=total,
        consecutive_skips=consecutive,
        skipped=True,
        max_total=8,
        max_consecutive=3,
    )
    assert exceeded
    total, consecutive, exceeded = pilot._bounded_skip_state(
        total_skips=total,
        consecutive_skips=consecutive,
        skipped=False,
        max_total=8,
        max_consecutive=3,
    )
    assert consecutive == 0 and not exceeded


def test_dataset_hash_covers_ordered_actual_image_bytes(tmp_path: Path) -> None:
    targets = tmp_path / "train" / "targets"
    targets.mkdir(parents=True)
    (targets / "a.png").write_bytes(b"first")
    (targets / "b.png").write_bytes(b"second")
    original = pilot._dataset_slice_sha256(tmp_path, ["a.png", "b.png"])
    assert original != pilot._dataset_slice_sha256(tmp_path, ["b.png", "a.png"])
    (targets / "a.png").write_bytes(b"changed")
    assert original != pilot._dataset_slice_sha256(tmp_path, ["a.png", "b.png"])


def test_clean_development_exposure_audit_is_fail_closed(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    quarantine = tmp_path / "quarantine.json"
    manifest.write_text(json.dumps({"splits": {"train": ["train.png"]}}))
    quarantine.write_text(json.dumps({"quarantine_names": ["denoise_val.png"]}))
    audit = pilot._upstream_exposure_audit(
        ["clean.png"],
        manifest=manifest,
        quarantine=quarantine,
        hbt_metadata={"train_names": ["hbt_train.png"], "val_names": ["hbt_val.png"]},
    )
    assert audit["zero_upstream_exposure_asserted"] is True
    with pytest.raises(RuntimeError, match="denoiser"):
        pilot._upstream_exposure_audit(
            ["denoise_val.png"],
            manifest=manifest,
            quarantine=quarantine,
            hbt_metadata={},
        )
    with pytest.raises(RuntimeError, match="hbt"):
        pilot._upstream_exposure_audit(
            ["hbt_val.png"],
            manifest=manifest,
            quarantine=quarantine,
            hbt_metadata={"val_names": ["hbt_val.png"]},
        )


def test_bootstrap_is_deterministic_and_source_level() -> None:
    values = np.asarray([0.01, 0.02, 0.03, 0.04])
    first = pilot._bootstrap_mean_ci(values, seed=9, resamples=500, confidence=0.95)
    second = pilot._bootstrap_mean_ci(values, seed=9, resamples=500, confidence=0.95)
    assert first == second
    assert first["lower"] > 0.0
    assert first["unit"].startswith("whole source")


def test_resume_contract_accepts_exact_state_and_rejects_training_drift() -> None:
    args = pilot._parser().parse_args(["--output-dir", "unused"])
    train_names = ["image_a.png", "image_b.png"]
    restorer_metadata = {"checkpoint_sha256": "denoiser-hash"}
    hbt_metadata = {"checkpoint_sha256": "hbt-hash"}
    metadata = pilot._checkpoint_metadata(
        args=args,
        train_names=train_names,
        epoch=0,
        optimizer_steps=1,
        restorer_metadata=restorer_metadata,
        hbt_metadata=hbt_metadata,
    )
    runtime = pilot.Runtime(torch.device("cpu"), 0, 0, 1, False)
    runtime_contracts = pilot._gather_runtime_resume_contracts(
        runtime,
        amp_enabled=False,
        amp_dtype=torch.float32,
    )
    payload = {
        "metadata": metadata,
        "training_state": {"runtime_contracts_by_rank": runtime_contracts},
    }
    pilot._validate_resume_contract(
        payload,
        args=args,
        train_names=train_names,
        restorer_metadata=restorer_metadata,
        hbt_metadata=hbt_metadata,
        runtime_contracts=runtime_contracts,
    )

    changed = argparse.Namespace(**vars(args))
    changed.learning_rate *= 2.0
    with pytest.raises(ValueError, match="learning_rate"):
        pilot._validate_resume_contract(
            payload,
            args=changed,
            train_names=train_names,
            restorer_metadata=restorer_metadata,
            hbt_metadata=hbt_metadata,
            runtime_contracts=runtime_contracts,
        )

    qap_drift = {
        "qap_iterations": args.qap_iterations + 1,
        "qap_restarts": args.qap_restarts + 1,
        "qap_boundary_weight": args.qap_boundary_weight + 0.01,
        "qap_refine_swaps": args.qap_refine_swaps + 1,
    }
    for name, value in qap_drift.items():
        changed = argparse.Namespace(**vars(args))
        setattr(changed, name, value)
        with pytest.raises(ValueError, match=name):
            pilot._validate_resume_contract(
                payload,
                args=changed,
                train_names=train_names,
                restorer_metadata=restorer_metadata,
                hbt_metadata=hbt_metadata,
                runtime_contracts=runtime_contracts,
            )


def test_standalone_evaluation_contract_rejects_warm_family_drift() -> None:
    pilot._configure_determinism()
    args = pilot._parser().parse_args(["--output-dir", "unused"])
    restorer = {"checkpoint_sha256": "denoiser-hash"}
    hbt = {"checkpoint_sha256": "hbt-hash"}
    metadata = pilot._checkpoint_metadata(
        args=args,
        train_names=["train.png"],
        epoch=0,
        optimizer_steps=1,
        restorer_metadata=restorer,
        hbt_metadata=hbt,
    )
    config = pilot._config(args)
    payload = {
        "safe_for_submission": False,
        "model_config": vars(config),
        "metadata": metadata,
    }
    manifest = (pilot.REPO_ROOT / args.manifest).resolve()
    quarantine = (pilot.REPO_ROOT / args.quarantine).resolve()
    audit = pilot._validate_evaluation_contract(
        payload,
        args=args,
        config=config,
        restorer_metadata=restorer,
        hbt_metadata=hbt,
        manifest=manifest,
        quarantine=quarantine,
    )
    assert audit["validated"] is True
    changed = argparse.Namespace(**vars(args))
    changed.warm_start_layout = "w4-qap"
    with pytest.raises(ValueError, match="warm"):
        pilot._validate_evaluation_contract(
            payload,
            args=changed,
            config=config,
            restorer_metadata=restorer,
            hbt_metadata=hbt,
            manifest=manifest,
            quarantine=quarantine,
        )


def test_strong_corruption_is_deterministic_input_only_uint8() -> None:
    tiles = np.arange(5 * 20 * 20 * 3, dtype=np.uint32).reshape(5, 20, 20, 3)
    tiles = (tiles % 256).astype(np.uint8)
    first = pilot._strong_corrupt_tiles(
        tiles, rng=np.random.default_rng(41), severity=0.8
    )
    second = pilot._strong_corrupt_tiles(
        tiles, rng=np.random.default_rng(41), severity=0.8
    )
    assert first.shape == tiles.shape
    assert first.dtype == np.uint8
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, tiles)

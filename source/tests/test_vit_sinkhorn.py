from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

_PILOT_PATH = Path(__file__).resolve().parents[1] / "scripts/train_evaluate_vit_sinkhorn.py"
_PILOT_SPEC = importlib.util.spec_from_file_location("vit_sinkhorn_pilot", _PILOT_PATH)
assert _PILOT_SPEC is not None and _PILOT_SPEC.loader is not None
pilot = importlib.util.module_from_spec(_PILOT_SPEC)
sys.modules[_PILOT_SPEC.name] = pilot
_PILOT_SPEC.loader.exec_module(pilot)

from puzzle_assembly.vit_sinkhorn import (
    ViTSinkhorn,
    ViTSinkhornConfig,
    hungarian_position_to_tile,
    log_sinkhorn,
    make_synthetic_smoke_batch,
    permutation_metrics_from_logits,
    vit_sinkhorn_losses,
)


def _tiny_config(*, qap_prior_dropout: float = 0.0) -> ViTSinkhornConfig:
    return ViTSinkhornConfig(
        grid_size=3,
        tile_size=8,
        d_model=32,
        num_layers=1,
        num_heads=4,
        feedforward_dim=64,
        cnn_channels=8,
        edge_channels=8,
        edge_dim=8,
        edge_band=2,
        edge_bins=4,
        dropout=0.0,
        qap_prior_dropout=qap_prior_dropout,
        sinkhorn_iterations=16,
        sinkhorn_temperature=0.25,
        activation_checkpointing=False,
    )


def test_log_sinkhorn_is_doubly_stochastic_and_differentiable() -> None:
    generator = torch.Generator().manual_seed(17)
    logits = torch.randn(2, 7, 7, generator=generator, requires_grad=True)
    log_assignment = log_sinkhorn(logits, iterations=30, temperature=0.4)
    assignment = log_assignment.exp()
    torch.testing.assert_close(
        assignment.sum(dim=2), torch.ones(2, 7), atol=2e-5, rtol=2e-5
    )
    torch.testing.assert_close(
        assignment.sum(dim=1), torch.ones(2, 7), atol=2e-5, rtol=2e-5
    )
    (assignment.square().sum()).backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_hungarian_returns_position_to_tile() -> None:
    tile_to_position = np.asarray([2, 0, 3, 1], dtype=np.int32)
    logits = np.full((4, 4), -10.0, dtype=np.float32)
    logits[np.arange(4), tile_to_position] = 10.0
    position_to_tile = hungarian_position_to_tile(logits)
    np.testing.assert_array_equal(position_to_tile, np.asarray([1, 3, 0, 2]))


def test_model_is_input_permutation_equivariant_without_prior() -> None:
    torch.manual_seed(23)
    config = _tiny_config()
    model = ViTSinkhorn(config).eval()
    raw = torch.rand(1, 9, 3, 8, 8)
    restored = torch.rand(1, 9, 3, 8, 8)
    order = torch.as_tensor([3, 8, 1, 5, 0, 7, 2, 6, 4])
    with torch.inference_mode():
        reference = model(raw, restored)
        permuted = model(raw[:, order], restored[:, order])
    torch.testing.assert_close(
        permuted.logits,
        reference.logits[:, order],
        atol=2e-5,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        permuted.log_assignment,
        reference.log_assignment[:, order],
        atol=2e-5,
        rtol=2e-5,
    )


def test_partial_gold_multitask_loss_is_finite_and_backpropagates() -> None:
    torch.manual_seed(29)
    model = ViTSinkhorn(_tiny_config()).train()
    raw = torch.rand(1, 9, 3, 8, 8)
    restored = torch.rand(1, 9, 3, 8, 8)
    full_target = torch.randperm(9).unsqueeze(0)
    partial_target = full_target.clone()
    partial_target[:, 4:] = -1
    confidence = torch.zeros(1, 9)
    confidence[:, :4] = torch.as_tensor([0.9, 0.8, 0.7, 0.6])
    output = model(raw, restored)
    losses = vit_sinkhorn_losses(
        output,
        partial_target,
        confidence=confidence,
        grid_size=3,
        consistency_topk=3,
    )
    assert set(losses) == {
        "total",
        "assignment",
        "directional_contrast",
        "neighbor_consistency",
    }
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_qap_prior_dropout_one_removes_prior_during_training() -> None:
    torch.manual_seed(31)
    model = ViTSinkhorn(_tiny_config(qap_prior_dropout=1.0)).train()
    raw = torch.rand(1, 9, 3, 8, 8)
    restored = torch.rand(1, 9, 3, 8, 8)
    qap = torch.randperm(9).unsqueeze(0)
    confidence = torch.ones(1, 9)
    without = model(raw, restored)
    with_prior = model(
        raw,
        restored,
        qap_tile_to_position=qap,
        qap_confidence=confidence,
    )
    assert with_prior.prior_keep_mask is not None
    assert not with_prior.prior_keep_mask.any()
    torch.testing.assert_close(with_prior.logits, without.logits)


def test_synthetic_batch_and_metrics_contract() -> None:
    batch = make_synthetic_smoke_batch(
        grid_size=3, tile_size=8, batch_size=1, seed=37
    )
    assert batch["raw_tiles"].shape == (1, 9, 3, 8, 8)
    assert "qap_tile_to_position" not in batch
    assert "qap_confidence" not in batch
    target = batch["target_tile_to_position"][0].numpy()
    logits = np.full((9, 9), -5.0, dtype=np.float32)
    logits[np.arange(9), target] = 5.0
    metrics = permutation_metrics_from_logits(logits, target, grid_size=3)
    assert metrics["valid_permutation"] is True
    assert metrics["position_accuracy"] == pytest.approx(1.0)
    assert metrics["combined_adjacency"] == pytest.approx(1.0)


def test_pilot_defaults_are_bounded_but_material_and_have_holdout() -> None:
    args = pilot._build_arg_parser().parse_args(["--output-dir", "unused"])
    assert args.train_sources >= 256
    assert args.epochs >= 3
    assert args.dev_sources > 0
    assert args.holdout_sources > 0
    assert args.gate_min_position_accuracy >= 0.01
    assert args.gate_min_combined_adjacency >= 0.02
    assert args.gate_min_classical_manhattan_reduction > 0
    assert args.gate_min_ssim_delta_vs_classical > 0
    assert args.qap_prior_probability == 0.0
    pilot._validate_args(args)


def test_enabling_qap_probability_requires_explicit_asset() -> None:
    args = pilot._build_arg_parser().parse_args(
        ["--output-dir", "unused", "--qap-prior-probability", "0.5"]
    )
    with pytest.raises(ValueError, match="explicit --qap-priors"):
        pilot._validate_args(args)


def test_synthetic_training_example_never_contains_truth_derived_prior() -> None:
    args = pilot._build_arg_parser().parse_args(
        ["--output-dir", "unused", "--disable-denoiser"]
    )
    name = pilot.source_names_for_split(
        "edge_train", manifest_path=args.manifest, quarantine_path=args.quarantine
    )[0]
    example = pilot._prepare_synthetic(
        name,
        args=args,
        epoch=0,
        stage="train",
        restorer=None,
        device=torch.device("cpu"),
    )
    assert example.qap_tile_to_position is None
    assert example.qap_confidence is None
    assert example.panel in {"primary_kornia", "independent_libjpeg"}


def _write_qap_asset(path: Path, *, targets_opened: bool | None) -> None:
    arrays: dict[str, np.ndarray] = {
        "source_names": np.asarray(["img_000001.png"]),
        "tile_to_position": np.arange(576, dtype=np.int64)[None],
    }
    if targets_opened is not None:
        arrays["meta"] = np.asarray(
            json.dumps({"kind": "input_only_qap", "targets_opened": targets_opened})
        )
    np.savez(path, **arrays)


@pytest.mark.parametrize("targets_opened", [None, True])
def test_qap_asset_fails_closed_without_explicit_input_only_metadata(
    tmp_path: Path, targets_opened: bool | None
) -> None:
    path = tmp_path / "qap.npz"
    _write_qap_asset(path, targets_opened=targets_opened)
    with pytest.raises(ValueError, match="targets_opened=false|must contain meta"):
        pilot._load_qap_priors(path)


def test_qap_asset_accepts_explicit_unopened_targets_metadata(tmp_path: Path) -> None:
    path = tmp_path / "qap.npz"
    _write_qap_asset(path, targets_opened=False)
    table, provenance = pilot._load_qap_priors(path)
    assert set(table) == {"img_000001.png"}
    assert provenance["metadata"]["targets_opened"] is False


def _write_pseudo_gold_asset(
    path: Path,
    *,
    manifest_path: Path,
    metadata_updates: dict[str, object] | None = None,
) -> None:
    metadata: dict[str, object] = {
        "schema_version": 1,
        "kind": "high_purity_real_tile_pairs",
        "split": "train",
        "old_q90_used_as_ground_truth": False,
        "source_name_encoding": "source_names[source_index]",
        "manifest_sha256": pilot._sha256(manifest_path),
        "test_overlap_excluded": 1,
        "source_count": 1,
        "selected_pairs": 1,
        "joint_confidence_definition": "normalized bidirectional margin",
        "selection_rule": "two input-only descriptors agree",
    }
    if metadata_updates:
        metadata.update(metadata_updates)
    np.savez(
        path,
        meta=np.asarray(json.dumps(metadata)),
        source_names=np.asarray(["train_a.png"]),
        source_index=np.asarray([0], dtype=np.int64),
        input_slot=np.asarray([3], dtype=np.int64),
        clean_tile_index=np.asarray([7], dtype=np.int64),
        joint_confidence=np.asarray([1.5], dtype=np.float32),
    )


@pytest.mark.parametrize(
    "metadata_updates",
    [
        {"kind": "ground_truth"},
        {"split": "test"},
        {"old_q90_used_as_ground_truth": True},
        {"old_q90_used_as_ground_truth": 0},
        {"source_name_encoding": "filenames_are_targets"},
        {"manifest_sha256": "0" * 64},
        {"test_overlap_excluded": 0},
        {"test_overlap_excluded": "1"},
    ],
)
def test_pseudo_gold_metadata_is_validated_fail_closed(
    tmp_path: Path, metadata_updates: dict[str, object]
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    data_root = tmp_path / "data"
    (data_root / "test").mkdir(parents=True)
    (data_root / "test" / "test_a.png").write_bytes(b"placeholder")
    archive = tmp_path / "pseudo.npz"
    _write_pseudo_gold_asset(
        archive, manifest_path=manifest, metadata_updates=metadata_updates
    )
    with pytest.raises(ValueError, match="pseudo-gold"):
        pilot._load_pseudo_gold(
            archive,
            allowed_train_names={"train_a.png"},
            forbidden_dev_names=set(),
            manifest_path=manifest,
            data_root=data_root,
        )


def test_pseudo_gold_uses_documented_raw_clipped_confidence(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    data_root = tmp_path / "data"
    (data_root / "test").mkdir(parents=True)
    (data_root / "test" / "test_a.png").write_bytes(b"placeholder")
    archive = tmp_path / "pseudo.npz"
    _write_pseudo_gold_asset(archive, manifest_path=manifest)
    table, provenance = pilot._load_pseudo_gold(
        archive,
        allowed_train_names={"train_a.png"},
        forbidden_dev_names=set(),
        manifest_path=manifest,
        data_root=data_root,
    )
    assert table["train_a.png"].confidence[3] == pytest.approx(1.0)
    assert provenance["label_kind"] == "partial_real_pseudo_gold_not_ground_truth"
    assert provenance["confidence_transform"] == "clip(joint_confidence,0,1)"


def test_rng_capture_contains_resume_state() -> None:
    state = pilot._capture_rng_state()
    assert set(state) == {"python", "numpy", "torch_cpu", "torch_cuda"}
    assert isinstance(state["torch_cpu"], torch.Tensor)
    gathered = pilot._gather_rank_rng_states(
        pilot.Runtime(torch.device("cpu"), 0, 0, 1, False)
    )
    assert gathered is not None and len(gathered) == 1


def test_training_snapshot_is_resume_complete_and_epoch_bound() -> None:
    model = ViTSinkhorn(_tiny_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    snapshot = pilot._capture_training_state(
        model, optimizer, None, selected_epoch=2
    )
    assert set(snapshot) == {
        "model_state",
        "optimizer_state",
        "scaler_state",
        "rng_state",
        "selected_epoch",
    }
    assert snapshot["selected_epoch"] == 2
    assert snapshot["model_state"]
    assert "param_groups" in snapshot["optimizer_state"]


def test_amp_overflow_policy_is_bounded_and_t4_safe_by_default() -> None:
    args = pilot._build_arg_parser().parse_args(["--output-dir", "unused"])
    assert args.amp_init_scale == pytest.approx(1024.0)
    assert args.max_consecutive_amp_skips == 8
    assert args.amp_init_scale < 65536.0


def test_code_provenance_hashes_model_script_and_imported_sources() -> None:
    provenance = pilot._code_provenance()
    assert len(provenance["combined_sha256"]) == 64
    files = provenance["files"]
    assert "src/puzzle_assembly/vit_sinkhorn.py" in files
    assert "scripts/train_evaluate_vit_sinkhorn.py" in files
    assert "src/puzzle_denoise_v2/degradation.py" in files
    assert all(len(digest) == 64 for digest in files.values())


def test_selection_and_independent_holdout_are_disjoint_whole_sources() -> None:
    args = pilot._build_arg_parser().parse_args(
        [
            "--output-dir",
            "unused",
            "--train-sources",
            "2",
            "--dev-sources",
            "2",
            "--holdout-sources",
            "2",
        ]
    )
    train, selection, holdout, audit = pilot._complete_source_split(args, gold={})
    assert set(train).isdisjoint(selection)
    assert set(train).isdisjoint(holdout)
    assert set(selection).isdisjoint(holdout)
    assert audit["selection_holdout_overlap_count"] == 0
    assert audit["selected_train_names"] == train
    assert audit["selected_development_names"] == selection
    assert audit["selected_holdout_names"] == holdout

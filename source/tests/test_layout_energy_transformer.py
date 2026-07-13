from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch import nn

from puzzle_assembly.layout_energy_transformer import (
    NEGATIVE_FAMILIES,
    LayoutEnergyConfig,
    LayoutEnergyOutput,
    LayoutEnergyTransformer,
    classical_seam_energy,
    iterative_refine_layout,
    layout_energy_losses,
    make_negative_layout,
    score_candidate_layouts,
)


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts/train_evaluate_layout_energy.py"
_SPEC = importlib.util.spec_from_file_location("layout_energy_pilot", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
pilot = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = pilot
_SPEC.loader.exec_module(pilot)


def _tiny_config() -> LayoutEnergyConfig:
    return LayoutEnergyConfig(
        grid_size=4,
        tile_size=8,
        d_model=32,
        num_heads=4,
        local_layers=2,
        window_size=2,
        global_layers=1,
        global_tokens=2,
        feedforward_dim=64,
        cnn_channels=8,
        edge_dim=8,
        edge_band=2,
        move_dim=8,
        dropout=0.0,
    )


def test_forward_shapes_and_candidate_encoding_reuse_contract() -> None:
    torch.manual_seed(3)
    config = _tiny_config()
    model = LayoutEnergyTransformer(config).eval()
    tiles = torch.rand(1, 16, 3, 8, 8)
    identity = torch.arange(16).view(1, 1, 16)
    with torch.inference_mode():
        direct = model(tiles)
        reused = model(tiles, candidate_layouts=identity)
    assert direct.energy.shape == (1,)
    assert direct.local_error_logits.shape == (1, 16)
    assert direct.move_vectors.shape == (1, 16, 2)
    assert direct.move_queries.shape == (1, 16, 8)
    assert direct.global_features.shape == (1, 2, 32)
    torch.testing.assert_close(direct.energy, reused.energy, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        direct.local_error_logits, reused.local_error_logits, atol=1e-6, rtol=1e-6
    )


def test_candidate_forward_rejects_non_permutations() -> None:
    model = LayoutEnergyTransformer(_tiny_config())
    tiles = torch.rand(1, 16, 3, 8, 8)
    repeated = torch.zeros(1, 1, 16, dtype=torch.long)
    with pytest.raises(ValueError, match="permutation"):
        model(tiles, candidate_layouts=repeated)


def test_ranking_local_and_move_losses_are_finite_and_backpropagate() -> None:
    torch.manual_seed(5)
    config = _tiny_config()
    model = LayoutEnergyTransformer(config).train()
    tiles = torch.rand(1, 16, 3, 8, 8)
    negative = make_negative_layout(
        grid_size=4,
        family="block_swap",
        rng=np.random.default_rng(5),
        severity=0.2,
    )
    layouts = torch.from_numpy(
        np.stack([np.arange(16, dtype=np.int32), negative.position_to_tile])
    ).long().unsqueeze(0)
    errors = torch.from_numpy(
        np.stack([np.zeros(16, dtype=np.float32), negative.error_mask])
    )
    moves = torch.from_numpy(
        np.stack([np.zeros((16, 2), dtype=np.float32), negative.move_targets])
    )
    output = model(tiles, candidate_layouts=layouts)
    losses = layout_energy_losses(
        output, errors, moves, candidates_per_source=2
    )
    assert set(losses) == {
        "total",
        "ranking",
        "listwise",
        "local_error",
        "move",
        "move_matching",
        "graded_monotonic",
        "energy_regularization",
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


def test_loss_requires_positive_first_and_nonempty_negatives() -> None:
    model = LayoutEnergyTransformer(_tiny_config()).eval()
    tiles = torch.rand(1, 16, 3, 8, 8)
    identity = torch.arange(16).view(1, 1, 16).expand(1, 2, 16)
    output = model(tiles, candidate_layouts=identity)
    errors = torch.zeros(2, 16)
    moves = torch.zeros(2, 16, 2)
    errors[0, 0] = 1.0
    with pytest.raises(ValueError, match="candidate zero"):
        layout_energy_losses(
            output, errors, moves, candidates_per_source=2
        )
    errors.zero_()
    with pytest.raises(ValueError, match="negative candidate"):
        layout_energy_losses(
            output, errors, moves, candidates_per_source=2
        )


def test_graded_monotonic_loss_prefers_energy_ordered_by_wrongness() -> None:
    tile_count = 4
    errors = torch.zeros(3, tile_count)
    errors[1, :2] = 1
    errors[2, :] = 1
    moves = torch.zeros(3, tile_count, 2)
    severity = torch.tensor([0.0, 0.4, 0.9])

    def output(energy: torch.Tensor) -> LayoutEnergyOutput:
        return LayoutEnergyOutput(
            energy=energy,
            local_error_logits=torch.zeros(3, tile_count),
            move_vectors=torch.zeros(3, tile_count, 2),
            move_queries=torch.nn.functional.normalize(torch.rand(3, tile_count, 4), dim=2),
            move_keys=torch.nn.functional.normalize(torch.rand(3, tile_count, 4), dim=2),
            tile_features=torch.zeros(3, tile_count, 4),
            global_features=torch.zeros(3, 1, 4),
        )

    ordered = layout_energy_losses(
        output(torch.tensor([0.0, 1.0, 2.0])),
        errors,
        moves,
        candidates_per_source=3,
        severity_targets=severity,
    )["graded_monotonic"]
    reversed_loss = layout_energy_losses(
        output(torch.tensor([2.0, 1.0, 0.0])),
        errors,
        moves,
        candidates_per_source=3,
        severity_targets=severity,
    )["graded_monotonic"]
    assert ordered < reversed_loss


@pytest.mark.parametrize("family", NEGATIVE_FAMILIES)
def test_every_hard_negative_family_is_a_supervised_permutation(family: str) -> None:
    rng = np.random.default_rng(11)
    features = rng.normal(size=(36, 12)).astype(np.float32)
    result = make_negative_layout(
        grid_size=6,
        family=family,
        rng=rng,
        tile_features=features,
        severity=0.08,
    )
    np.testing.assert_array_equal(np.sort(result.position_to_tile), np.arange(36))
    assert not np.array_equal(result.position_to_tile, np.arange(36))
    np.testing.assert_array_equal(
        result.error_mask.astype(bool), result.position_to_tile != np.arange(36)
    )
    positions = np.arange(36)
    expected_move = np.stack(
        [
            (result.position_to_tile // 6 - positions // 6) / 5.0,
            (result.position_to_tile % 6 - positions % 6) / 5.0,
        ],
        axis=1,
    )
    np.testing.assert_allclose(result.move_targets, expected_move, atol=1e-7)


def test_positive_and_negatives_share_exact_dirty_tile_multiset() -> None:
    rng = np.random.default_rng(13)
    dirty_tiles = rng.integers(0, 256, size=(36, 8, 8, 3), dtype=np.uint8)
    layouts = [np.arange(36, dtype=np.int32)]
    for family in NEGATIVE_FAMILIES:
        layouts.append(
            make_negative_layout(
                grid_size=6,
                family=family,
                rng=np.random.default_rng(100 + len(layouts)),
                tile_features=rng.normal(size=(36, 8)),
                severity=0.08,
            ).position_to_tile
        )
    reference_hashes = sorted(
        hashlib_sha(tile) for tile in dirty_tiles
    )
    for layout in layouts:
        candidate = dirty_tiles[layout]
        assert sorted(hashlib_sha(tile) for tile in candidate) == reference_hashes


def test_actual_panel_candidate_builder_never_recorrupts_negatives() -> None:
    args = pilot._build_parser().parse_args(
        ["--output-dir", "unused", "--epochs", "2"]
    )
    name = pilot.source_names_for_split(
        "edge_train", manifest_path=args.manifest, quarantine_path=args.quarantine
    )[0]
    panel = pilot._panel_for_source(
        name, args=args, epoch=0, stage="train"
    )
    layouts, _, _, severities, labels, semantic = pilot._candidate_set(
        panel,
        source_name=name,
        args=args,
        epoch=0,
        stage="train",
        count=3,
    )
    reference = pilot._tile_multiset_sha256(panel.raw_tiles)
    assert panel.metadata["corrupted_tile_multiset_sha256"] == reference
    assert "raw_seam_first_pass" in labels
    assert severities[0] == 0.0
    np.testing.assert_array_equal(semantic[0], np.arange(576))
    np.testing.assert_array_equal(
        np.sort(panel.first_pass_position_to_slot), np.arange(576)
    )
    chain = [
        severities[index]
        for index, label in enumerate(labels)
        if label == "raw_seam_first_pass" or label.startswith("raw_seam_residual_repair_")
    ]
    assert all(later < earlier for earlier, later in zip(chain, chain[1:]))
    for layout in layouts:
        assert pilot._tile_multiset_sha256(panel.raw_tiles[layout]) == reference


def hashlib_sha(values: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def test_classical_raw_seam_energy_prefers_coherent_grid() -> None:
    grid, tile = 2, 4
    side = grid * tile
    y, x = np.mgrid[:side, :side]
    image = np.stack([x * 20, y * 20, (x + y) * 10], axis=2).astype(np.uint8)
    tiles = (
        image.reshape(grid, tile, grid, tile, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(grid * grid, tile, tile, 3)
    )
    swapped = tiles[np.asarray([3, 1, 2, 0])]
    assert float(classical_seam_energy(tiles)) < float(classical_seam_energy(swapped))


class _ToyEnergy(nn.Module):
    """Exact 2x2 energy/move oracle used only to test target-free search API."""

    def __init__(self) -> None:
        super().__init__()
        self.config = LayoutEnergyConfig(
            grid_size=2,
            tile_size=4,
            d_model=4,
            num_heads=2,
            local_layers=1,
            window_size=1,
            global_layers=1,
            global_tokens=1,
            feedforward_dim=8,
            cnn_channels=4,
            edge_dim=2,
            edge_band=1,
            move_dim=4,
            dropout=0.0,
        )
        self.anchor = nn.Parameter(torch.zeros(()))

    def encode_tiles(self, tiles: torch.Tensor) -> torch.Tensor:
        identity = torch.round(tiles.mean(dim=(2, 3, 4)) * 3.0) / 3.0
        encoded = torch.zeros(len(tiles), 4, 4, device=tiles.device)
        encoded[:, :, 0] = identity
        return encoded + self.anchor * 0.0

    def score_encoded_tiles(self, tokens: torch.Tensor) -> LayoutEnergyOutput:
        batch = len(tokens)
        tile_ids = torch.round(tokens[:, :, 0] * 3.0).long()
        positions = torch.arange(4, device=tokens.device).view(1, 4)
        wrong = tile_ids != positions
        row_delta = tile_ids // 2 - positions // 2
        col_delta = tile_ids % 2 - positions % 2
        moves = torch.stack([row_delta, col_delta], dim=2).float()
        queries = torch.nn.functional.one_hot(tile_ids, num_classes=4).float()
        keys = torch.eye(4, device=tokens.device).unsqueeze(0).expand(batch, -1, -1)
        return LayoutEnergyOutput(
            energy=wrong.float().sum(dim=1) + self.anchor * 0.0,
            local_error_logits=torch.where(wrong, 8.0, -8.0),
            move_vectors=moves,
            move_queries=queries,
            move_keys=keys,
            tile_features=tokens,
            global_features=tokens[:, :1],
        )


def _toy_raw_tiles() -> np.ndarray:
    values = np.asarray([0, 85, 170, 255], dtype=np.uint8)
    return np.stack(
        [np.full((4, 4, 3), value, dtype=np.uint8) for value in values]
    )


def test_scoring_and_iterative_refinement_need_no_target() -> None:
    model = _ToyEnergy().eval()
    raw = _toy_raw_tiles()
    identity = np.arange(4, dtype=np.int32)
    swapped = np.asarray([1, 0, 2, 3], dtype=np.int32)
    scores = score_candidate_layouts(model, raw, np.stack([identity, swapped]))
    np.testing.assert_allclose(scores.energies, [0.0, 2.0])
    result = iterative_refine_layout(
        model,
        raw,
        swapped,
        steps=2,
        beam_width=2,
        hot_positions=2,
        proposals_per_layout=4,
        score_batch_size=4,
        min_improvement=0.0,
    )
    np.testing.assert_array_equal(result.position_to_slot, identity)
    assert result.final_energy < result.initial_energy
    assert result.steps[0].accepted_improvement is True


def test_curriculum_and_defaults_are_serious_but_bounded() -> None:
    args = pilot._build_parser().parse_args(["--output-dir", "unused"])
    assert args.train_sources >= 512
    assert args.epochs >= 4
    assert args.negatives_per_source >= 4
    assert args.d_model >= 256 and args.local_layers >= 6
    assert args.selection_sources > 0 and args.holdout_sources > 0
    assert args.gate_min_ranking_accuracy > 0.5
    assert args.gate_min_delta_vs_classical > 0
    assert args.gate_min_relative_repair_error_reduction >= 0.25
    assert args.repair_steps >= 6
    assert args.amp_init_scale <= 1024
    assert args.max_consecutive_amp_skips > 1
    assert args.eval_replicas >= 2
    early, early_severity = pilot.curriculum_families(0, args.epochs)
    late, late_severity = pilot.curriculum_families(args.epochs - 1, args.epochs)
    assert "row_column" in early
    assert "similar_swap" in late and "solver_like_sparse" in late
    assert late_severity < early_severity
    pilot._validate_args(args)


def test_repair_gate_is_rejected_when_theoretically_impossible() -> None:
    args = pilot._build_parser().parse_args(
        [
            "--output-dir",
            "unused",
            "--repair-steps",
            "6",
            "--repair-hot-positions",
            "2",
            "--gate-min-relative-repair-error-reduction",
            "0.25",
        ]
    )
    with pytest.raises(ValueError, match="theoretical"):
        pilot._validate_args(args)


def test_whole_source_selection_and_holdout_are_disjoint() -> None:
    args = pilot._build_parser().parse_args(
        [
            "--output-dir",
            "unused",
            "--train-sources",
            "2",
            "--selection-sources",
            "2",
            "--holdout-sources",
            "2",
        ]
    )
    train, selection, holdout, audit = pilot._split_sources(args)
    assert set(train).isdisjoint(selection)
    assert set(train).isdisjoint(holdout)
    assert set(selection).isdisjoint(holdout)
    assert audit["test_targets_opened"] is False


def test_binary_auc_and_code_provenance_contracts() -> None:
    assert pilot._binary_auc(np.asarray([0, 0, 1, 1]), np.asarray([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    provenance = pilot._code_provenance()
    assert len(provenance["combined_sha256"]) == 64
    assert "src/puzzle_assembly/layout_energy_transformer.py" in provenance["files"]
    assert "scripts/train_evaluate_layout_energy.py" in provenance["files"]


def test_resume_contract_binds_args_split_data_and_code() -> None:
    args = pilot._build_parser().parse_args(["--output-dir", "unused"])
    source = {"combined_sha256": "a" * 64}
    first = pilot._resume_contract(
        args,
        split_audit={"train_names_sha256": "b" * 64},
        data_provenance={"train_targets_sha256": "c" * 64},
        source_code=source,
    )
    second = pilot._resume_contract(
        args,
        split_audit={"train_names_sha256": "different"},
        data_provenance={"train_targets_sha256": "c" * 64},
        source_code=source,
    )
    assert first["contract_sha256"] != second["contract_sha256"]
    runtime = pilot.Runtime(torch.device("cpu"), 0, 0, 1, False)
    assert pilot._synchronized_all_finite(runtime, True) is True
    assert pilot._synchronized_all_finite(runtime, False) is False


def test_amp_skip_policy_halves_scale_and_is_bounded() -> None:
    scale, consecutive, total = pilot._bounded_amp_skip_update(
        scale_before=1024.0,
        consecutive_skips=0,
        total_skips=0,
        max_consecutive=2,
        max_total=4,
    )
    assert (scale, consecutive, total) == (512.0, 1, 1)
    with pytest.raises(RuntimeError, match="skip budget exhausted"):
        pilot._bounded_amp_skip_update(
            scale_before=512.0,
            consecutive_skips=2,
            total_skips=2,
            max_consecutive=2,
            max_total=4,
        )


def test_rank_preflight_is_explicitly_fail_closed_without_t4() -> None:
    runtime = pilot.Runtime(torch.device("cpu"), 0, 0, 1, False)
    probe = pilot._rank_hardware_probe(runtime, require_t4=False)
    assert probe["cuda"] is False
    with pytest.raises(RuntimeError, match="requires CUDA T4"):
        pilot._rank_hardware_probe(runtime, require_t4=True)
    assert pilot._gather_objects(runtime, {"rank": 0}) == [{"rank": 0}]


def test_raw_only_branch_does_not_import_denoiser_inference() -> None:
    source = _SCRIPT_PATH.read_text(encoding="utf-8")
    assert "load_restorer" not in source
    assert "restore_tiles_uint8" not in source
    assert '"denoiser_used": False' in source

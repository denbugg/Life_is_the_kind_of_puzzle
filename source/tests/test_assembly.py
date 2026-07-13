from __future__ import annotations

import numpy as np
import torch

from puzzle_assembly.compatibility import (
    CompatibilityMatrices,
    build_edge_filter_score_bank,
    fuse_ranked_scores,
)
from puzzle_assembly.components import (
    mutual_topk_component_solver,
    reciprocal_component_solver,
    weighted_l1_component_solver,
)
from puzzle_assembly.geometry import GRID, TILE_COUNT, inverse_permutation
from puzzle_assembly.learned import (
    ContextPositionTransformer,
    GlobalSuccessorMatcher,
    PositionPriorHead,
    RankFeatureNet,
    SeamPairNet,
    SideEmbeddingNet,
    SideSequenceEmbeddingNet,
    candidate_rank_features,
    candidate_union,
    direction_labels,
    embedding_hard_triplet_loss,
    global_matching_loss,
    load_context_position_checkpoint,
    load_global_matcher_checkpoint,
    load_embedding_checkpoint,
    load_pair_checkpoint,
    load_position_prior_checkpoint,
    load_rank_feature_checkpoint,
    save_context_position_checkpoint,
    save_global_matcher_checkpoint,
    save_embedding_checkpoint,
    save_pair_checkpoint,
    save_position_prior_checkpoint,
    save_rank_feature_checkpoint,
    seam_pair_patches,
)
from puzzle_assembly.metrics import layout_metrics, retrieval_metrics
from puzzle_assembly.protocol import source_names_for_split
from puzzle_assembly.solvers import (
    beam_row_major,
    greedy_row_major,
    identity_layout,
    random_layout,
    relaxation_labeling_solver,
    outside_logits_placement_unary,
    position_logits_placement_unary,
    segment_block_refine,
    simulated_anneal_swaps,
)


def oracle_compatibility(slot_to_target: np.ndarray) -> CompatibilityMatrices:
    position_to_slot = inverse_permutation(slot_to_target)
    right = np.ones((TILE_COUNT, TILE_COUNT), dtype=np.float32)
    down = np.ones((TILE_COUNT, TILE_COUNT), dtype=np.float32)
    np.fill_diagonal(right, np.inf)
    np.fill_diagonal(down, np.inf)
    for slot, target_position in enumerate(slot_to_target.tolist()):
        row, column = divmod(target_position, GRID)
        if column + 1 < GRID:
            right[slot, position_to_slot[target_position + 1]] = 0.0
        if row + 1 < GRID:
            down[slot, position_to_slot[target_position + GRID]] = 0.0
    return CompatibilityMatrices("oracle", right, down)


def test_permutation_inverse_and_layout_metrics() -> None:
    slot_to_target = np.random.default_rng(17).permutation(TILE_COUNT).astype(np.int32)
    position_to_slot = inverse_permutation(slot_to_target)
    assert np.array_equal(slot_to_target[position_to_slot], np.arange(TILE_COUNT))
    metrics = layout_metrics(position_to_slot, slot_to_target)
    assert metrics["valid_permutation"] is True
    assert metrics["position_accuracy"] == 1.0
    assert metrics["combined_adjacency"] == 1.0
    assert metrics["largest_correct_component"] == TILE_COUNT


def test_oracle_retrieval_and_solvers() -> None:
    slot_to_target = np.random.default_rng(20260710).permutation(TILE_COUNT).astype(np.int32)
    compatibility = oracle_compatibility(slot_to_target)
    retrieval = retrieval_metrics(compatibility, slot_to_target)
    assert retrieval["combined"]["recall_at_1"] == 1.0
    expected = inverse_permutation(slot_to_target)
    greedy = greedy_row_major(compatibility)
    beam = beam_row_major(compatibility, width=4, candidate_pool=2)
    assert np.array_equal(greedy, expected)
    assert np.array_equal(beam, expected)
    assert np.array_equal(segment_block_refine(greedy, compatibility), expected)
    assert np.array_equal(
        simulated_anneal_swaps(greedy, compatibility, seed=4, evaluations=20), expected
    )
    assert np.array_equal(
        relaxation_labeling_solver(
            compatibility, initial=greedy, iterations=1, sinkhorn_iterations=2
        ),
        expected,
    )


def test_oracle_reciprocal_components_recover_exact_layout() -> None:
    slot_to_target = np.random.default_rng(824).permutation(TILE_COUNT).astype(np.int32)
    compatibility = oracle_compatibility(slot_to_target)
    result = reciprocal_component_solver(compatibility, refine=False)
    expected = inverse_permutation(slot_to_target)
    assert np.array_equal(result.position_to_slot, expected)
    assert result.component_sizes[0] == TILE_COUNT
    assert result.placed_component_tiles == TILE_COUNT
    assert result.unresolved_tiles_before_assignment == 0
    beam_result = reciprocal_component_solver(
        compatibility,
        refine=False,
        placement_beam_width=3,
        placement_beam_components=2,
    )
    assert np.array_equal(beam_result.position_to_slot, expected)
    mutual_result = mutual_topk_component_solver(compatibility, top_k=2)
    assert np.array_equal(mutual_result.position_to_slot, expected)
    loop_only = reciprocal_component_solver(
        compatibility, only_verified_loops=True, refine=False
    )
    assert np.array_equal(loop_only.position_to_slot, expected)
    lp_result = weighted_l1_component_solver(compatibility)
    assert np.array_equal(lp_result.position_to_slot, expected)
    assert lp_result.lp_failures == 0


def test_rank_fusion_keeps_oracle_first() -> None:
    slot_to_target = np.random.default_rng(4).permutation(TILE_COUNT).astype(np.int32)
    oracle = oracle_compatibility(slot_to_target)
    fused = fuse_ranked_scores({"first": oracle, "second": oracle})
    assert retrieval_metrics(fused, slot_to_target)["combined"]["recall_at_1"] == 1.0


def test_edge_filter_score_bank_shapes_and_self_exclusion() -> None:
    tiles = np.random.default_rng(37).integers(
        0, 256, size=(TILE_COUNT, 20, 20, 3), dtype=np.uint8
    )
    bank = build_edge_filter_score_bank(tiles, prefix="raw", chunk_size=64)
    assert set(bank) == {"raw_sobel_l1_w2", "raw_binary_edge_hamming_w2"}
    for score in bank.values():
        assert score.right.shape == score.down.shape == (TILE_COUNT, TILE_COUNT)
        assert np.all(np.isinf(np.diag(score.right)))
        assert np.all(np.isinf(np.diag(score.down)))
        assert np.all(np.isfinite(score.right[~np.eye(TILE_COUNT, dtype=bool)]))


def test_baseline_layouts_are_permutations() -> None:
    assert np.array_equal(identity_layout(), np.arange(TILE_COUNT))
    random = random_layout(3)
    assert len(np.unique(random)) == TILE_COUNT


def test_authoritative_split_derivation_counts() -> None:
    kwargs = {
        "manifest_path": "configs/denoise_splits_seed20260710.json",
        "quarantine_path": "configs/denoise_validation_quarantine_v1.json",
    }
    assert len(source_names_for_split("edge_train", **kwargs)) == 4500
    assert len(source_names_for_split("edge_development", **kwargs)) == 400
    assert len(source_names_for_split("assembly_cal", **kwargs)) == 257
    assert len(source_names_for_split("assembly_incremental_gate", **kwargs)) == 350


def test_learned_direction_labels_and_model_shapes() -> None:
    slot_to_target = np.random.default_rng(91).permutation(TILE_COUNT).astype(np.int32)
    labels = direction_labels(slot_to_target)
    assert len(labels.right_queries) == GRID * (GRID - 1)
    assert len(labels.down_queries) == GRID * (GRID - 1)
    assert labels.outside.sum() == 4 * GRID
    assert np.all(
        slot_to_target[labels.right_targets] - slot_to_target[labels.right_queries] == 1
    )
    assert np.all(
        slot_to_target[labels.down_targets] - slot_to_target[labels.down_queries] == GRID
    )
    model = SideEmbeddingNet(channels=16, embedding_dim=12, tangent_bins=5)
    outputs = model(torch.rand(7, 3, 20, 20))
    assert model.config()["embedding_dim"] == 12
    assert outputs["q_right"].shape == (7, 12)
    assert outputs["k_left"].shape == (7, 12)
    assert outputs["q_down"].shape == (7, 12)
    assert outputs["k_up"].shape == (7, 12)
    assert outputs["outside_logits"].shape == (7, 4)
    for input_mode in ("rgb_sobel", "sobel_only", "binary_edges"):
        edge_model = SideEmbeddingNet(
            channels=16,
            embedding_dim=12,
            tangent_bins=5,
            input_mode=input_mode,
        )
        edge_outputs = edge_model(torch.rand(7, 3, 20, 20))
        assert edge_outputs["q_right"].shape == (7, 12)
        assert edge_model.config()["input_mode"] == input_mode
    binary_model = SideEmbeddingNet(
        channels=16, embedding_dim=12, tangent_bins=5, input_mode="binary_edges"
    )
    binary_features = binary_model._input_features(torch.rand(7, 3, 20, 20))
    assert set(torch.unique(binary_features).tolist()).issubset({0.0, 1.0})
    full_outputs = model(torch.rand(TILE_COUNT, 3, 20, 20))
    hard_loss, hard_metrics = embedding_hard_triplet_loss(
        full_outputs,
        labels,
        temperature=model.temperature,
        margin=0.2,
        cross_entropy_weight=0.25,
        embedding_l2_weight=1e-4,
        outside_weight=0.2,
    )
    assert torch.isfinite(hard_loss)
    assert hard_metrics["triplet_loss"] >= 0.0
    assert 0.0 <= hard_metrics["recall_at_1"] <= 1.0
    sequence_model = SideSequenceEmbeddingNet(
        channels=16, embedding_dim=8, side_band=3
    )
    sequence_outputs = sequence_model(torch.rand(7, 3, 20, 20))
    assert sequence_outputs["q_right"].shape == (7, 20, 8)
    assert sequence_outputs["k_left"].shape == (7, 20, 8)
    tiles = torch.rand(7, 3, 20, 20)
    first = torch.tensor([0, 1, 2, 3])
    second = torch.tensor([4, 5, 6, 0])
    directions = torch.tensor([0, 1, 0, 1])
    pair_model = SeamPairNet(channels=16, side_band=4)
    patches = seam_pair_patches(
        tiles, first, second, directions, side_band=pair_model.side_band
    )
    assert patches.shape == (4, 3, 20, 8)
    assert pair_model(patches).shape == (4,)
    context = ContextPositionTransformer(
        model_dim=32, layers=1, heads=4, feedforward_dim=64
    ).eval()
    context_tiles = torch.rand(1, 7, 3, 20, 20)
    context_rows, context_columns = context(context_tiles)
    assert context_rows.shape == context_columns.shape == (1, 7, GRID)
    permutation = torch.tensor([3, 0, 6, 1, 5, 2, 4])
    permuted_rows, _ = context(context_tiles[:, permutation])
    assert torch.allclose(permuted_rows, context_rows[:, permutation], atol=1e-5)


def test_global_successor_matcher_assignment_and_loss() -> None:
    slot_to_target = np.random.default_rng(113).permutation(TILE_COUNT).astype(np.int32)
    labels = direction_labels(slot_to_target)
    embeddings = {
        name: torch.nn.functional.normalize(torch.randn(TILE_COUNT, 16), dim=1)
        for name in ("q_right", "k_left", "q_down", "k_up")
    }
    embeddings["outside_logits"] = torch.randn(TILE_COUNT, 4)
    model = GlobalSuccessorMatcher(
        embedding_dim=16,
        model_dim=16,
        layers=1,
        heads=4,
        feedforward_dim=32,
        sinkhorn_iterations=3,
    )
    outputs = model(embeddings)
    assert outputs["scores"].shape == (2, TILE_COUNT, TILE_COUNT)
    assert outputs["log_assignment"].shape == (2, TILE_COUNT + 1, TILE_COUNT + 1)
    loss, metrics = global_matching_loss(outputs, labels)
    assert torch.isfinite(loss)
    loss.backward()
    assert 0.0 <= metrics["recall_at_1"] <= 1.0
    assert 0.0 <= metrics["mutual_precision"] <= 1.0


def test_candidate_union_shape_and_self_exclusion() -> None:
    slot_to_target = np.random.default_rng(31).permutation(TILE_COUNT).astype(np.int32)
    oracle = oracle_compatibility(slot_to_target)
    right, down = candidate_union(
        {"oracle": oracle}, names=["oracle"], per_score_top_k=4, cap=8
    )
    assert right.shape == down.shape == (TILE_COUNT, 8)
    assert np.all(right != np.arange(TILE_COUNT)[:, None])
    assert np.all(down != np.arange(TILE_COUNT)[:, None])
    features = candidate_rank_features(
        {"oracle": oracle}, (right, down), names=["oracle"]
    )
    assert features.shape == (2, TILE_COUNT, 8, 8)
    assert np.isfinite(features).all()


def test_learned_outside_unary_prefers_correct_boundary_class() -> None:
    logits = np.full((TILE_COUNT, 4), -10.0, dtype=np.float32)
    logits[0, [0, 2]] = 10.0
    unary = outside_logits_placement_unary(logits)
    assert unary.shape == (TILE_COUNT, TILE_COUNT)
    assert unary[0, 0] < unary[GRID + 1, 0]
    assert unary[GRID + 1, 1] < unary[0, 1]
    row_logits = np.full((TILE_COUNT, GRID), -10.0, dtype=np.float32)
    column_logits = np.full((TILE_COUNT, GRID), -10.0, dtype=np.float32)
    row_logits[0, 0] = 10.0
    column_logits[0, 0] = 10.0
    position_unary = position_logits_placement_unary(row_logits, column_logits)
    assert position_unary[0, 0] < position_unary[GRID + 1, 0]


def test_learned_checkpoint_roundtrips(tmp_path) -> None:
    embedding = SideEmbeddingNet(channels=16, embedding_dim=12, tangent_bins=5)
    embedding_path = tmp_path / "embedding.pt"
    save_embedding_checkpoint(embedding_path, embedding, metadata={"test": True})
    loaded_embedding, embedding_metadata = load_embedding_checkpoint(embedding_path)
    assert loaded_embedding.config() == embedding.config()
    assert embedding_metadata == {"test": True}

    sequence = SideSequenceEmbeddingNet(channels=16, embedding_dim=8, side_band=3)
    sequence_path = tmp_path / "sequence.pt"
    save_embedding_checkpoint(sequence_path, sequence, metadata={"sequence": True})
    loaded_sequence, sequence_metadata = load_embedding_checkpoint(sequence_path)
    assert isinstance(loaded_sequence, SideSequenceEmbeddingNet)
    assert loaded_sequence.config() == sequence.config()
    assert sequence_metadata == {"sequence": True}

    pair = SeamPairNet(channels=16, side_band=4)
    pair_path = tmp_path / "pair.pt"
    save_pair_checkpoint(pair_path, pair, metadata={"test": True})
    loaded_pair, pair_metadata = load_pair_checkpoint(pair_path)
    assert loaded_pair.config() == pair.config()
    assert pair_metadata == {"test": True}

    rank_feature = RankFeatureNet(feature_dim=24, hidden_dim=16)
    assert rank_feature(torch.rand(3, 7, 24)).shape == (3, 7)
    rank_path = tmp_path / "rank.pt"
    save_rank_feature_checkpoint(
        rank_path,
        rank_feature,
        feature_names=["a", "b", "c", "d", "e"],
        metadata={"test": True},
    )
    loaded_rank, feature_names, rank_metadata = load_rank_feature_checkpoint(rank_path)
    assert loaded_rank.config() == rank_feature.config()
    assert feature_names == ["a", "b", "c", "d", "e"]
    assert rank_metadata == {"test": True}

    position = PositionPriorHead(feature_dim=28, hidden_dim=16)
    row_logits, column_logits = position(torch.rand(5, 28))
    assert row_logits.shape == column_logits.shape == (5, GRID)
    position_path = tmp_path / "position.pt"
    save_position_prior_checkpoint(position_path, position, metadata={"test": True})
    loaded_position, position_metadata = load_position_prior_checkpoint(position_path)
    assert loaded_position.config() == position.config()
    assert position_metadata == {"test": True}

    context = ContextPositionTransformer(
        model_dim=32, layers=1, heads=4, feedforward_dim=64
    )
    context_path = tmp_path / "context.pt"
    save_context_position_checkpoint(context_path, context, metadata={"test": True})
    loaded_context, context_metadata = load_context_position_checkpoint(context_path)
    assert loaded_context.config() == context.config()
    assert context_metadata == {"test": True}

    global_matcher = GlobalSuccessorMatcher(
        embedding_dim=12,
        model_dim=16,
        layers=1,
        heads=4,
        feedforward_dim=32,
        sinkhorn_iterations=3,
    )
    global_path = tmp_path / "global_matcher.pt"
    save_global_matcher_checkpoint(
        global_path, global_matcher, metadata={"test": True}
    )
    loaded_global, global_metadata = load_global_matcher_checkpoint(global_path)
    assert loaded_global.config() == global_matcher.config()
    assert global_metadata == {"test": True}
    load_position_prior_checkpoint,
    save_position_prior_checkpoint,
    load_context_position_checkpoint,
    save_context_position_checkpoint,

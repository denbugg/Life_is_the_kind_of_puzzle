from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from e26_contextual_edge_net import (  # noqa: E402
    DOWN,
    LEFT,
    RIGHT,
    UP,
    ContextualDirectionalEdgeNet,
    ContextualEdgeConfig,
    directional_neighbour_labels,
    listwise_directional_ce,
)


def tiny_config() -> ContextualEdgeConfig:
    return ContextualEdgeConfig(
        grid_height=2,
        grid_width=3,
        cnn_width=16,
        d_model=32,
        local_dim=24,
        match_dim=16,
        transformer_layers=2,
        attention_heads=4,
        ff_multiplier=1.5,
        dropout=0.0,
        boundary_band=1,
        boundary_bins=3,
        reconstruction_samples=4,
        encoder_chunk_size=4,
    )


class ContextualEdgeForwardContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(26)
        cls.model = ContextualDirectionalEdgeNet(tiny_config()).eval()
        cls.tiles = torch.rand(1, 6, 3, 8, 8)

    def test_smoke_shapes_none_class_and_finite_diagonal(self) -> None:
        with torch.no_grad():
            output = self.model(self.tiles)
        self.assertEqual(tuple(output["pair_logits"].shape), (1, 4, 6, 6))
        self.assertEqual(tuple(output["none_logits"].shape), (1, 4, 6))
        self.assertEqual(tuple(output["logits"].shape), (1, 4, 6, 7))
        self.assertEqual(tuple(output["tile_tokens"].shape), (1, 6, 32))
        self.assertEqual(tuple(output["side_tokens"].shape), (1, 6, 4, 16))
        self.assertEqual(tuple(output["boundary_reconstruction"].shape), (1, 6, 4, 3, 4))
        self.assertTrue(torch.isfinite(output["logits"]).all())
        diagonal = output["pair_logits"].diagonal(dim1=-2, dim2=-1)
        self.assertTrue(
            torch.equal(diagonal, torch.full_like(diagonal, tiny_config().diagonal_mask_value))
        )
        self.assertTrue(torch.equal(output["logits"][..., -1], output["none_logits"]))

    def test_inverse_symmetry_is_exact_by_construction(self) -> None:
        with torch.no_grad():
            pair = self.model(self.tiles)["pair_logits"]
        self.assertTrue(torch.equal(pair[:, LEFT], pair[:, RIGHT].transpose(-1, -2)))
        self.assertTrue(torch.equal(pair[:, UP], pair[:, DOWN].transpose(-1, -2)))

    def test_input_permutation_equivariance(self) -> None:
        permutation = torch.tensor([3, 0, 5, 2, 1, 4])
        with torch.no_grad():
            original = self.model(self.tiles)
            shuffled = self.model(self.tiles[:, permutation])
        expected_pair = original["pair_logits"][:, :, permutation][:, :, :, permutation]
        expected_none = original["none_logits"][:, :, permutation]
        expected_reconstruction = original["boundary_reconstruction"][:, permutation]
        torch.testing.assert_close(shuffled["pair_logits"], expected_pair, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(shuffled["none_logits"], expected_none, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(
            shuffled["boundary_reconstruction"],
            expected_reconstruction,
            atol=2e-5,
            rtol=2e-5,
        )


class DirectionalLabelContracts(unittest.TestCase):
    def test_identity_labels_have_exact_neighbours_and_none_borders(self) -> None:
        labels = directional_neighbour_labels(torch.arange(6), 2, 3)[0]
        none = 6
        self.assertEqual(labels[:, 0].tolist(), [none, 3, none, 1])
        self.assertEqual(labels[:, 4].tolist(), [1, none, 3, 5])
        self.assertEqual(int(labels[UP].eq(none).sum()), 3)
        self.assertEqual(int(labels[DOWN].eq(none).sum()), 3)
        self.assertEqual(int(labels[LEFT].eq(none).sum()), 2)
        self.assertEqual(int(labels[RIGHT].eq(none).sum()), 2)

    def test_shuffled_labels_refer_to_input_tile_indices(self) -> None:
        # Input tile 0 came from clean position 2.  Its down neighbour is clean
        # position 5, which appears at input index 2 in this permutation.
        permutation = torch.tensor([[2, 0, 5, 1, 4, 3]])
        labels = directional_neighbour_labels(permutation, 2, 3)
        self.assertEqual(int(labels[0, DOWN, 0]), 2)
        self.assertEqual(int(labels[0, RIGHT, 0]), 6)
        # Exact inverse relation in the labels: 0 down->2 means 2 up->0.
        self.assertEqual(int(labels[0, UP, 2]), 0)

    def test_listwise_ce_uses_all_neighbours_plus_none(self) -> None:
        labels = directional_neighbour_labels(torch.arange(6), 2, 3)
        logits = torch.full((1, 4, 6, 7), -9.0)
        diagonal = torch.eye(6, dtype=torch.bool).view(1, 1, 6, 6)
        logits[..., :6] = logits[..., :6].masked_fill(diagonal, -10_000.0)
        logits.scatter_(-1, labels.unsqueeze(-1), 9.0)
        good = listwise_directional_ce(logits, labels)
        smoothed = listwise_directional_ce(logits, labels, label_smoothing=0.02)
        bad = listwise_directional_ce(torch.zeros_like(logits), labels)
        self.assertLess(float(good), 1.0e-5)
        self.assertLess(float(smoothed), 1.0)
        self.assertGreater(float(bad), 1.0)

    def test_invalid_permutation_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "exact permutation"):
            directional_neighbour_labels(torch.tensor([0, 1, 2, 3, 4, 4]), 2, 3)


if __name__ == "__main__":
    unittest.main()

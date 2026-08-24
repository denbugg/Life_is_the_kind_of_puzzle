from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from m144_dct_where import (  # noqa: E402
    DCT_OUTPUT_DIM,
    M144WhereModel,
    RGB_OUTPUT_DIM,
    bicubic_interpolation_matrix,
    dct_2d,
    encode_dct_residual,
    encode_rgb_residual,
    evaluate_cal_gates,
    evaluate_dev_gates,
    flat_rgb_from_tiles,
    fixed_bicubic_resize,
    idct_2d,
    one_sided_bootstrap_lower,
    paired_lift,
    render_dct_residual,
    render_rgb_residual,
    skimage_ssim_parity,
    summarize_arm_metrics,
    uniform_ssim,
    zigzag_indices,
)


class DCTRepresentationTests(unittest.TestCase):
    def test_zigzag_prefix_is_canonical_and_complete(self) -> None:
        expected = (
            (0, 0),
            (0, 1),
            (1, 0),
            (2, 0),
            (1, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (2, 1),
            (3, 0),
        )
        self.assertEqual(zigzag_indices(4, 10), expected)
        complete = zigzag_indices(8)
        self.assertEqual(len(complete), 64)
        self.assertEqual(len(set(complete)), 64)
        self.assertTrue(all(0 <= row < 8 and 0 <= column < 8 for row, column in complete))

    def test_dct_idct_round_trip_rectangular_batch(self) -> None:
        torch.manual_seed(144)
        value = torch.randn(2, 3, 7, 9, dtype=torch.float64)
        recovered = idct_2d(dct_2d(value))
        self.assertLess(float((recovered - value).abs().max()), 2.0e-14)

    def test_fixed_bicubic_matrix_matches_production_forward_and_has_gradients(self) -> None:
        torch.manual_seed(144_480)
        for side in (8, 16):
            value = torch.randn(2, 3, side, side, dtype=torch.float32, requires_grad=True)
            reference = F.interpolate(
                value.detach(), size=(480, 480), mode="bicubic", align_corners=False
            )
            rendered = fixed_bicubic_resize(value, (480, 480))
            self.assertEqual(tuple(rendered.shape), (2, 3, 480, 480))
            self.assertLessEqual(float((rendered.detach() - reference).abs().max()), 2.0e-6)

            matrix = bicubic_interpolation_matrix(
                side, 480, device=torch.device("cpu"), dtype=torch.float32
            )
            self.assertEqual(tuple(matrix.shape), (480, side))
            self.assertFalse(matrix.requires_grad)
            self.assertLessEqual(float((matrix.sum(dim=1) - 1.0).abs().max()), 1.0e-6)
            rendered.square().mean().backward()
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())

    def test_zero_dct_residual_renders_the_exact_flat(self) -> None:
        flat = torch.tensor([[0.125, 0.5, 0.875], [0.2, 0.3, 0.4]], dtype=torch.float32)
        coefficients = torch.zeros(2, 3, 32)
        rendered = render_dct_residual(coefficients, flat, size=(31, 29))
        expected = flat[:, :, None, None].expand(-1, -1, 31, 29)
        self.assertTrue(torch.equal(rendered, expected))

        target = flat[:, :, None, None].expand(-1, -1, 23, 27).clone()
        encoded = encode_dct_residual(target, flat)
        self.assertTrue(torch.equal(encoded, torch.zeros_like(encoded)))

    def test_all_coefficients_reconstruct_the_reduced_target(self) -> None:
        torch.manual_seed(145)
        target = torch.rand(3, 3, 8, 8, dtype=torch.float64) * 0.8 + 0.1
        flat = torch.rand(3, 3, dtype=torch.float64) * 0.4 + 0.3
        coefficients = encode_dct_residual(target, flat, side=8, count=64)
        rendered = render_dct_residual(
            coefficients, flat, size=(8, 8), side=8, count=64, clamp=False
        )
        self.assertLess(float((rendered - target).abs().max()), 2.0e-14)

    def test_rgb_comparator_round_trip_and_flat_contract(self) -> None:
        torch.manual_seed(146)
        target = torch.rand(2, 3, 8, 8) * 0.8 + 0.1
        flat = torch.tensor([[0.4, 0.5, 0.6], [0.2, 0.3, 0.4]])
        residual = encode_rgb_residual(target, flat, side=8)
        rendered = render_rgb_residual(residual, flat, size=8, side=8, clamp=False)
        self.assertTrue(torch.allclose(rendered, target, atol=1.0e-7, rtol=0.0))

        zero = torch.zeros(2, RGB_OUTPUT_DIM)
        zero_render = render_rgb_residual(zero, flat, size=(13, 11))
        expected = flat[:, :, None, None].expand(-1, -1, 13, 11)
        self.assertTrue(torch.equal(zero_render, expected))

    def test_flat_rgb_from_tiles_has_no_hidden_rounding(self) -> None:
        tiles = torch.arange(2 * 5 * 3 * 4 * 4, dtype=torch.float64).reshape(2, 5, 3, 4, 4)
        observed = flat_rgb_from_tiles(tiles)
        expected = tiles.mean(dim=(1, 3, 4))
        self.assertTrue(torch.equal(observed, expected))


class SlotModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(144_001)
        self.model = M144WhereModel(
            output_dim=DCT_OUTPUT_DIM,
            embedding_dim=32,
            num_slots=16,
            num_heads=4,
            self_layers=2,
            ffn_dim=48,
            hidden_dim=64,
        ).eval()
        self.embeddings = torch.randn(3, 37, 32)
        self.flat = torch.rand(3, 3) * 0.5 + 0.25

    def _activate_zero_head(self) -> None:
        final = self.model.head[-1]
        assert isinstance(final, torch.nn.Linear)
        with torch.no_grad():
            final.weight.normal_(mean=0.0, std=0.02)
            final.bias.normal_(mean=0.0, std=0.01)

    def test_zero_initialised_head_emits_exact_zero_in_both_modes(self) -> None:
        with torch.no_grad():
            full = self.model.forward_full(self.embeddings, self.flat)
            blind = self.model.forward_blind(self.embeddings, self.flat)
        self.assertEqual(tuple(full.shape), (3, DCT_OUTPUT_DIM))
        self.assertTrue(torch.equal(full, torch.zeros_like(full)))
        self.assertTrue(torch.equal(blind, torch.zeros_like(blind)))

    def test_full_forward_is_invariant_to_tile_permutation(self) -> None:
        self._activate_zero_head()
        permutation = torch.randperm(self.embeddings.shape[1])
        with torch.no_grad():
            original = self.model.forward_full(self.embeddings, self.flat)
            shuffled = self.model.forward_full(self.embeddings[:, permutation], self.flat)
        self.assertTrue(torch.allclose(original, shuffled, atol=2.0e-6, rtol=2.0e-6))

    def test_blind_forward_ignores_every_embedding_value(self) -> None:
        self._activate_zero_head()
        replacement = torch.randn_like(self.embeddings) * 100.0
        with torch.no_grad():
            first = self.model.forward_blind(self.embeddings, self.flat)
            second = self.model.forward_blind(replacement, self.flat)
        self.assertTrue(torch.equal(first, second))

    def test_model_and_dct_render_receive_finite_gradients(self) -> None:
        self.model.train()
        self._activate_zero_head()
        embeddings = self.embeddings.clone().requires_grad_(True)
        prediction = self.model.forward_full(embeddings, self.flat)
        rendered = render_dct_residual(prediction, self.flat, size=(15, 15))
        target = torch.rand_like(rendered)
        loss = 1.0 - uniform_ssim(rendered.float(), target.float()).mean()
        loss.backward()
        self.assertIsNotNone(embeddings.grad)
        assert embeddings.grad is not None
        self.assertTrue(torch.isfinite(embeddings.grad).all())
        self.assertGreater(float(embeddings.grad.abs().sum()), 0.0)
        self.assertTrue(
            any(
                parameter.grad is not None
                and torch.isfinite(parameter.grad).all()
                and float(parameter.grad.abs().sum()) > 0.0
                for parameter in self.model.parameters()
            )
        )


class SSIMParityTests(unittest.TestCase):
    def test_identical_images_score_exactly_one(self) -> None:
        torch.manual_seed(147)
        image = torch.rand(2, 3, 17, 19, dtype=torch.float64)
        score = uniform_ssim(image, image)
        self.assertTrue(torch.allclose(score, torch.ones_like(score), atol=1.0e-13, rtol=0.0))

    def test_proxy_matches_skimage_uniform_window(self) -> None:
        torch.manual_seed(148)
        first = torch.rand(3, 3, 19, 21, dtype=torch.float64)
        second = (first * 0.83 + torch.rand_like(first) * 0.17).clamp(0.0, 1.0)
        proxy = uniform_ssim(first, second).detach().cpu().numpy()
        reference = skimage_ssim_parity(first, second)
        self.assertLess(float(np.max(np.abs(proxy - reference))), 2.0e-12)

    def test_ssim_is_differentiable(self) -> None:
        torch.manual_seed(149)
        first = torch.rand(2, 3, 15, 15, requires_grad=True)
        second = torch.rand_like(first)
        loss = 1.0 - uniform_ssim(first, second).mean()
        loss.backward()
        self.assertIsNotNone(first.grad)
        assert first.grad is not None
        self.assertTrue(torch.isfinite(first.grad).all())
        self.assertGreater(float(first.grad.abs().sum()), 0.0)


class MetricAndGateTests(unittest.TestCase):
    def _passing_arrays(self, count: int = 12) -> dict[str, np.ndarray]:
        flat = np.full(count, 0.350, dtype=np.float64)
        return {
            "flat_ssim": flat,
            "target_oracle_dct_ssim": flat + 0.050,
            "dct_full_ssim": flat + 0.020,
            "dct_blind_ssim": flat + 0.010,
            "dct_swapped_ssim": flat + 0.010,
            "rgb8_full_ssim": flat + 0.009,
            "rgb8_blind_ssim": flat + 0.004,
        }

    def test_paired_lift_validates_and_does_not_alias_inputs(self) -> None:
        first = np.array([0.4, 0.5, 0.6])
        second = np.array([0.1, 0.2, 0.3])
        lift = paired_lift(first, second)
        self.assertTrue(np.allclose(lift, [0.3, 0.3, 0.3]))
        lift[0] = 99.0
        self.assertEqual(first[0], 0.4)

    def test_cluster_bootstrap_is_deterministic_and_preserves_constant_lift(self) -> None:
        values = np.full(10, 0.0075)
        groups = np.repeat(np.arange(5), 2)
        first = one_sided_bootstrap_lower(values, groups=groups, samples=257, seed=9)
        second = one_sided_bootstrap_lower(values, groups=groups, samples=257, seed=9)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first, 0.0075, places=15)

    def test_summary_uses_source_groups_and_swap_cycles(self) -> None:
        arrays = self._passing_arrays()
        source_groups = np.repeat(np.arange(6), 2)
        swap_groups = np.repeat(np.arange(4), 3)
        summary = summarize_arm_metrics(
            **arrays,
            source_groups=source_groups,
            swap_groups=swap_groups,
            bootstrap_samples=321,
            alpha=0.05,
        )
        self.assertEqual(summary["n_boards"], 12)
        self.assertEqual(summary["n_source_groups"], 6)
        self.assertEqual(summary["n_swap_cycles"], 4)
        self.assertAlmostEqual(summary["gains"]["dct_full"], 0.020)
        self.assertAlmostEqual(summary["contrasts"]["full_minus_blind"]["mean"], 0.010)
        self.assertAlmostEqual(summary["contrasts"]["full_minus_swapped"]["mean"], 0.010)
        self.assertAlmostEqual(summary["contrasts"]["representation_delta"]["mean"], 0.005)
        self.assertEqual(summary["contrasts"]["full_minus_blind"]["win_fraction"], 1.0)
        self.assertTrue(evaluate_dev_gates(summary)["passed"])

    def test_cal_gate_requires_90_percent_summary_and_oracle(self) -> None:
        arrays = self._passing_arrays()
        summary = summarize_arm_metrics(
            **arrays,
            bootstrap_samples=257,
            alpha=0.10,
        )
        decision = evaluate_cal_gates(summary)
        self.assertTrue(decision["passed"])
        self.assertIn("oracle_gain", decision["checks"])

        wrong_confidence = summarize_arm_metrics(
            **arrays,
            bootstrap_samples=257,
            alpha=0.05,
        )
        with self.assertRaisesRegex(ValueError, "90%"):
            evaluate_cal_gates(wrong_confidence)

    def test_dev_gate_fails_a_nonpositive_lower_bound(self) -> None:
        arrays = self._passing_arrays(count=10)
        # Keep a strong point mean but create clustered uncertainty whose lower
        # tail crosses zero.  A point estimate alone must not authorize DEV.
        alternating = np.array([0.05] * 5 + [-0.03] * 5)
        arrays["dct_full_ssim"] = arrays["dct_blind_ssim"] + alternating
        summary = summarize_arm_metrics(
            **arrays,
            source_groups=np.arange(10),
            swap_groups=np.arange(10),
            bootstrap_samples=2_000,
            alpha=0.05,
        )
        decision = evaluate_dev_gates(summary)
        self.assertFalse(decision["checks"]["full_blind"]["passed"])
        self.assertLessEqual(decision["checks"]["full_blind"]["lower"], 0.0)
        self.assertFalse(decision["passed"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import m144_dct_where as core  # noqa: E402
import verify_m144_dct_where as verify  # noqa: E402


def synthetic_feature60(count: int, offset: float = 0.0) -> np.ndarray:
    """Make structurally valid dimension-major palette features, no image data."""

    base = np.linspace(0.15 + offset, 0.85 + offset, count, dtype=np.float64)
    result = np.empty((count, 60), dtype=np.float64)
    quantile_offsets = np.linspace(-0.01, 0.01, 13, dtype=np.float64)
    for column in range(4):
        center = np.clip(base + 0.005 * column, 0.02, 0.98)
        result[:, column * 13 : (column + 1) * 13] = np.clip(
            center[:, None] + quantile_offsets[None, :], 0.0, 1.0
        )
        result[:, 52 + column] = center
        result[:, 56 + column] = 0.02 + 0.001 * column
    return result


class VerifierMathTests(unittest.TestCase):
    def test_independent_renderer_is_byte_exact_with_core(self) -> None:
        generator = torch.Generator().manual_seed(144032)
        flat = torch.rand(3, 3, generator=generator)
        dct = torch.randn(3, 96, generator=generator)
        rgb = torch.randn(3, 192, generator=generator)
        self.assertTrue(
            torch.equal(
                verify._render_dct_coefficients(dct, flat),
                core.render_dct_residual(dct, flat),
            )
        )
        self.assertTrue(
            torch.equal(
                verify._render_rgb8_residual(rgb, flat),
                core.render_rgb_residual(rgb, flat),
            )
        )

    def test_independent_oracle_encoding_matches_core(self) -> None:
        generator = torch.Generator().manual_seed(144033)
        target = torch.rand(2, 3, 480, 480, generator=generator)
        flat = torch.rand(2, 3, generator=generator)
        expected = core.encode_dct_residual(target, flat).reshape(2, 96)
        observed = verify._encode_oracle_coefficients(target, flat)
        self.assertTrue(torch.equal(expected, observed))

    def test_official_uint8_prediction_evidence_and_tamper(self) -> None:
        temp_parent = os.environ.get("M144_TEST_TMP")
        if temp_parent:
            Path(temp_parent).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_parent) as directory:
            root = Path(directory)
            targets: list[Path] = []
            arrays: list[np.ndarray] = []
            for index, value in enumerate((64, 192)):
                image = np.full((480, 480, 3), value, dtype=np.uint8)
                image[:, :, 1] = np.uint8(value // 2)
                path = root / f"img_{index:06d}.png"
                Image.fromarray(image, mode="RGB").save(path)
                targets.append(path)
                arrays.append(image)
            flat = np.asarray(
                [[64 / 255.0, 32 / 255.0, 64 / 255.0],
                 [192 / 255.0, 96 / 255.0, 192 / 255.0]],
                dtype=np.float32,
            )
            target_tensor = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).float() / 255.0
            flat_tensor = torch.from_numpy(flat)
            oracle = verify._encode_oracle_coefficients(target_tensor, flat_tensor).numpy()
            rendered_flat = flat_tensor[:, :, None, None].expand(-1, -1, 480, 480).numpy()
            rendered_oracle = verify._render_dct_coefficients(
                torch.from_numpy(oracle), flat_tensor
            ).numpy()
            stored = {
                "flat_ssim": np.asarray(
                    [
                        verify._official_uint8_ssim(
                            verify._quantize_prediction(rendered_flat[i]), arrays[i]
                        )
                        for i in range(2)
                    ],
                    dtype=np.float64,
                ),
                "target_oracle_dct_ssim": np.asarray(
                    [
                        verify._official_uint8_ssim(
                            verify._quantize_prediction(rendered_oracle[i]), arrays[i]
                        )
                        for i in range(2)
                    ],
                    dtype=np.float64,
                ),
            }
            with mock.patch.object(verify, "PARTITION_COUNT", 2):
                recomputed, diagnostic = verify.verify_prediction_evidence(
                    flat_rgb=flat,
                    target_paths=targets,
                    stored_scores=stored,
                    oracle_coeff_reference=oracle.astype(np.float32),
                    batch_size=2,
                )
                self.assertTrue(np.array_equal(recomputed["flat_ssim"], stored["flat_ssim"]))
                self.assertEqual(
                    diagnostic["stored_score_error"]["target_oracle_dct_ssim"]["max_abs"],
                    0.0,
                )
                tampered = {key: value.copy() for key, value in stored.items()}
                tampered["flat_ssim"][0] += 2.0e-12
                with self.assertRaisesRegex(verify.VerificationError, "official uint8 SSIM"):
                    verify.verify_prediction_evidence(
                        flat_rgb=flat,
                        target_paths=targets,
                        stored_scores=tampered,
                        oracle_coeff_reference=oracle.astype(np.float32),
                        batch_size=2,
                    )

    def test_canonical_arbitrary_cycles(self) -> None:
        board = np.arange(10, 17, dtype=np.int64)
        # index cycles (0 1), (2 3 4), (5 6)
        donor_index = np.asarray([1, 0, 3, 4, 2, 6, 5], dtype=np.int64)
        donor_board = board[donor_index]
        expected = np.asarray([0, 0, 1, 1, 1, 2, 2], dtype=np.int64)
        self.assertTrue(
            np.array_equal(verify.canonical_cycle_ids(board, donor_board), expected)
        )

    def test_fixed_point_and_non_bijection_rejected(self) -> None:
        board = np.arange(4, dtype=np.int64)
        with self.assertRaisesRegex(verify.VerificationError, "fixed point"):
            verify.canonical_cycle_ids(board, board.copy())
        with self.assertRaisesRegex(verify.VerificationError, "bijection"):
            verify.canonical_cycle_ids(
                board, np.asarray([1, 0, 1, 2], dtype=np.int64)
            )

    def test_hungarian_is_bijective_and_group_safe(self) -> None:
        count = 12
        features = synthetic_feature60(count)
        groups = np.repeat(np.arange(count // 2, dtype=np.int64), 2)
        fit = synthetic_feature60(40, offset=0.01)
        mean = fit.mean(axis=0, dtype=np.float64)
        scale = np.maximum(fit.std(axis=0, dtype=np.float64), 1.0e-6)
        donor = verify.recompute_palette_assignment(features, groups, mean, scale)
        self.assertEqual(len(np.unique(donor)), count)
        self.assertTrue(np.all(donor != np.arange(count)))
        self.assertTrue(np.all(groups[donor] != groups))

    def test_cycle_cluster_minimum_is_fail_closed(self) -> None:
        count = verify.PARTITION_COUNT
        board = np.arange(10_000, 10_000 + count, dtype=np.int64)
        donor_index = np.roll(np.arange(count, dtype=np.int64), -1)
        donor = board[donor_index]
        groups = np.arange(count, dtype=np.int64)
        swap = {
            "board_id": board,
            "source_group_id": groups,
            "donor_board_id": donor,
            "donor_source_group_id": groups[donor_index],
            "swap_cycle_id": verify.canonical_cycle_ids(board, donor),
            "dirty_feature60": synthetic_feature60(count),
        }
        fit_features = synthetic_feature60(20)
        fit = {
            "mean60": fit_features.mean(axis=0, dtype=np.float64),
            "scale60": np.maximum(
                fit_features.std(axis=0, dtype=np.float64), 1.0e-6
            ),
        }
        with mock.patch.object(
            verify, "recompute_palette_assignment", return_value=donor_index
        ):
            with self.assertRaisesRegex(verify.VerificationError, "under-clustered"):
                verify.verify_swap_semantics(swap, fit)

    def test_cluster_bootstrap_uses_declared_clusters(self) -> None:
        values = np.asarray([0.01, 0.01, -0.005, -0.005, 0.02, 0.02], dtype=np.float64)
        groups = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
        with mock.patch.object(verify, "PARTITION_COUNT", 6), mock.patch.object(
            verify, "BOOTSTRAP_SAMPLES", 2_000
        ):
            first = verify.one_sided_cluster_bootstrap_lower(
                values, groups, confidence=0.95, seed=144032
            )
            second = verify.one_sided_cluster_bootstrap_lower(
                values, groups, confidence=0.95, seed=144032
            )
        self.assertEqual(first, second)
        self.assertLess(first, float(values.mean()))

    def test_representation_delta_blocks_generic_prior(self) -> None:
        count = 6
        base = np.full(count, 0.30, dtype=np.float64)
        arrays = {
            "flat_ssim": base,
            "target_oracle_dct_ssim": base + 0.05,
            "dct_full_ssim": base + 0.02,
            "dct_blind_ssim": base + 0.01,
            "dct_swapped_ssim": base + 0.012,
            # Same conditional lift as DCT: representation delta is exactly 0.
            "rgb8_full_ssim": base + 0.015,
            "rgb8_blind_ssim": base + 0.005,
            "source_group_id": np.arange(count, dtype=np.int64),
            "swap_cycle_id": np.repeat(np.arange(3, dtype=np.int64), 2),
        }
        with mock.patch.object(verify, "PARTITION_COUNT", count), mock.patch.object(
            verify, "BOOTSTRAP_SAMPLES", 500
        ):
            summary = verify.summarize_arrays(arrays, confidence=verify.DEV_CONFIDENCE)
            gates = verify.evaluate_dev_gates(summary)
        self.assertAlmostEqual(summary["contrasts"]["representation_delta"]["mean"], 0.0)
        self.assertFalse(gates["checks"]["representation_delta"]["passed"])
        self.assertFalse(gates["passed"])

    def test_dev_win_fraction_is_strict_and_required(self) -> None:
        summary = {
            "gains": {"dct_full": 0.02},
            "contrasts": {
                "full_minus_blind": {
                    "mean": 0.01, "lower": 0.005, "win_fraction": 0.59
                },
                "full_minus_swapped": {
                    "mean": 0.01, "lower": 0.005, "win_fraction": 0.60
                },
                "representation_delta": {"mean": 0.005, "lower": 0.002},
            },
        }
        gates = verify.evaluate_dev_gates(summary)
        self.assertFalse(gates["checks"]["full_blind_win"]["passed"])
        self.assertTrue(gates["checks"]["full_swapped_win"]["passed"])
        self.assertFalse(gates["passed"])

    def test_create_once_output_is_idempotent_not_overwritable(self) -> None:
        temp_parent = os.environ.get("M144_TEST_TMP")
        if temp_parent:
            Path(temp_parent).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_parent) as directory:
            output = Path(directory) / "verification.json"
            verify.write_create_once(output, {"valid": True, "decision": "KILL"})
            first = output.read_bytes()
            verify.write_create_once(output, {"valid": True, "decision": "KILL"})
            self.assertEqual(output.read_bytes(), first)
            with self.assertRaisesRegex(verify.VerificationError, "not byte-exact"):
                verify.write_create_once(output, {"valid": True, "decision": "PROMOTE"})


if __name__ == "__main__":
    unittest.main()


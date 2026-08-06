from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import eval_cross_stage_buddies as cross  # noqa: E402
from rank_transplant import row_zscore  # noqa: E402


def _metric_row(image: int, variant: str, solve_ssim: float) -> dict[str, float | str]:
    row: dict[str, float | str] = {
        "image": float(image),
        "variant": variant,
        "component_source": "raw",
        "packing_source": "raw",
    }
    for metric in cross.MEAN_METRICS:
        row[metric] = float(solve_ssim if metric == "solve_only_ssim" else 0.1)
    return row


def _frozen_config(variant_name: str = "raw_components_i21_pack") -> dict[str, object]:
    variant = cross.VARIANT_BY_NAME[variant_name]
    return {
        "schema_version": cross.SCHEMA_VERSION,
        "experiment": cross.EXPERIMENT,
        "status": "frozen",
        "calibration_images": list(cross.CALIBRATION_IDS),
        "confirmation_images": list(cross.CONFIRMATION_IDS),
        "replay_group": cross.REPLAY_GROUP_TEXT,
        "replay_seed": cross.REPLAY_SEED,
        "cache_tag": cross.CACHE_TAG,
        "spatial_checkpoint": "unused.pt",
        "spatial_sha256": "a" * 64,
        "spatial_step": 6000,
        "i21_alpha": cross.I21_ALPHA,
        "budget": cross.BUDGET,
        "repair_passes": cross.REPAIR_PASSES,
        "selection": {
            "variant": variant.name,
            "configuration": {
                "name": variant.name,
                "component_source": variant.component_source,
                "packing_source": variant.packing_source,
            },
        },
        "code": cross._code_provenance(),
    }


class CrossStageContracts(unittest.TestCase):
    def test_split_and_variants_are_frozen(self) -> None:
        self.assertEqual(cross.CALIBRATION_IDS, tuple(range(10, 18)))
        self.assertEqual(cross.CONFIRMATION_IDS, tuple(range(18, 22)))
        self.assertEqual(cross.REPLAY_GROUPS, ((10, 12),))
        self.assertEqual(cross.I21_ALPHA, 1.25)
        self.assertEqual(cross.BUDGET, 512)
        self.assertEqual(cross.REPAIR_PASSES, 0)
        self.assertEqual(
            [(row.component_source, row.packing_source) for row in cross.VARIANTS],
            [("raw", "raw"), ("raw", "i21"), ("i21", "raw"), ("i21", "i21")],
        )

    def test_i21_fusion_is_exact_row_z_sum_and_preserves_mask(self) -> None:
        rng = np.random.default_rng(7)
        raw = rng.normal(size=(4, 5, 6)).astype(np.float32)
        spatial = rng.normal(size=raw.shape).astype(np.float32)
        raw[0, 0, -1] = -np.inf
        spatial[0, 0, -1] = -np.inf
        frozen_raw = raw.copy()
        frozen_spatial = spatial.copy()
        actual = cross.i21_scores(raw, spatial)
        valid = np.isfinite(raw)
        expected = np.full(raw.shape, -np.inf, dtype=np.float32)
        expected[valid] = (
            row_zscore(raw, valid)[valid]
            + np.float32(cross.I21_ALPHA) * row_zscore(spatial, valid)[valid]
        )
        self.assertTrue(np.array_equal(np.isfinite(actual), valid))
        self.assertTrue(np.allclose(actual[valid], expected[valid], atol=0.0, rtol=0.0))
        self.assertTrue(np.array_equal(raw, frozen_raw))
        self.assertTrue(np.array_equal(spatial, frozen_spatial))

    def test_cross_stage_mapping_keeps_component_and_packing_inputs_separate(self) -> None:
        raw_right, raw_down = np.asarray([[1.0]]), np.asarray([[2.0]])
        i21_right, i21_down = np.asarray([[3.0]]), np.asarray([[4.0]])
        raw_components = [{0: (0, 0)}]
        i21_components = [{1: (0, 0)}]
        prepared = cross.PreparedScene(
            scene=None,  # type: ignore[arg-type]
            scores={},
            dense={"raw": (raw_right, raw_down), "i21": (i21_right, i21_down)},
            components={"raw": raw_components, "i21": i21_components},
            edge={},
        )
        components, right, down = cross.cross_stage_inputs(prepared, "raw_components_i21_pack")
        self.assertIs(components, raw_components)
        self.assertIs(right, i21_right)
        self.assertIs(down, i21_down)
        components, right, down = cross.cross_stage_inputs(prepared, "i21_components_raw_pack")
        self.assertIs(components, i21_components)
        self.assertIs(right, raw_right)
        self.assertIs(down, raw_down)
        with self.assertRaisesRegex(ValueError, "unknown"):
            cross.cross_stage_inputs(prepared, "invented")

    def test_paired_delta_matches_by_image_not_row_order(self) -> None:
        baseline = [_metric_row(10, "raw_to_raw", 0.10), _metric_row(11, "raw_to_raw", 0.20)]
        candidate = [
            _metric_row(11, "raw_components_i21_pack", 0.24),
            _metric_row(10, "raw_components_i21_pack", 0.12),
        ]
        delta = cross.paired_deltas(candidate, baseline)
        self.assertAlmostEqual(delta["solve_only_ssim"]["mean_delta"], 0.03)
        self.assertEqual(delta["solve_only_ssim"]["wins"], 2)
        duplicate = candidate + [candidate[0]]
        with self.assertRaisesRegex(ValueError, "same unique image IDs"):
            cross.paired_deltas(duplicate, baseline)

    def test_calibration_freezes_only_strictly_positive_ssim(self) -> None:
        def row(ssim: float, neighbour: float = 0.0) -> dict[str, object]:
            return {
                "paired_delta": {
                    metric: {"mean_delta": ssim if metric == "solve_only_ssim" else neighbour}
                    for metric in cross.PRIMARY_METRICS
                }
            }

        none = {
            "raw_to_raw": row(0.0),
            "raw_components_i21_pack": row(0.0, 1.0),
            "i21_components_raw_pack": row(-0.01),
            "i21_to_i21": row(-0.02),
        }
        self.assertIsNone(cross.select_calibration_variant(none))
        grid = dict(none)
        grid["raw_components_i21_pack"] = row(0.002, 0.5)
        grid["i21_components_raw_pack"] = row(0.004, -0.5)
        selected = cross.select_calibration_variant(grid)
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected[0], "i21_components_raw_pack")

    def test_frozen_config_rejects_any_changed_protocol_or_baseline(self) -> None:
        frozen = _frozen_config()
        variant = cross.validate_frozen_config(frozen)
        self.assertEqual(variant.name, "raw_components_i21_pack")
        changed = copy.deepcopy(frozen)
        changed["i21_alpha"] = 1.0
        with self.assertRaisesRegex(RuntimeError, "i21_alpha"):
            cross.validate_frozen_config(changed)
        baseline = _frozen_config()
        baseline_variant = cross.VARIANT_BY_NAME[cross.BASELINE_VARIANT]
        baseline["selection"] = {
            "variant": cross.BASELINE_VARIANT,
            "configuration": {
                "name": baseline_variant.name,
                "component_source": baseline_variant.component_source,
                "packing_source": baseline_variant.packing_source,
            },
        }
        with self.assertRaisesRegex(RuntimeError, "nonbaseline"):
            cross.validate_frozen_config(baseline)

    def test_confirmation_evaluates_only_baseline_and_frozen_choice(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frozen_path = root / "frozen.json"
            frozen_path.write_text(json.dumps(_frozen_config()), encoding="utf-8")
            args = argparse.Namespace(
                frozen_config=frozen_path,
                cache_dir=root / "cache",
                spatial=root / "unused.pt",
                device="cpu",
                report=root / "confirmation.json",
            )
            prepared = list(cross.CONFIRMATION_IDS)

            def evaluate(image: int, variant: str) -> dict[str, float | str]:
                solve = 0.10 if variant == cross.BASELINE_VARIANT else 0.11
                return _metric_row(image, variant, solve)

            with (
                patch.object(
                    cross,
                    "_load_phase_scenes",
                    return_value=(list(cross.CONFIRMATION_IDS), "a" * 64, 6000),
                ),
                patch.object(cross, "prepare_scene", side_effect=lambda scene: scene),
                patch.object(cross, "evaluate_variant", side_effect=evaluate) as evaluated,
                patch.object(cross, "scene_provenance", return_value=[]),
                patch("builtins.print"),
            ):
                report = cross.run_confirmation(args)
            self.assertEqual(report["status"], "pass")
            self.assertEqual(evaluated.call_count, 2 * len(cross.CONFIRMATION_IDS))
            called_variants = [call.args[1] for call in evaluated.call_args_list]
            self.assertEqual(
                called_variants,
                [cross.BASELINE_VARIANT] * 4 + ["raw_components_i21_pack"] * 4,
            )


if __name__ == "__main__":
    unittest.main()

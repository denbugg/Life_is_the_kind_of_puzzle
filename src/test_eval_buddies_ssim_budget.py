"""Focused contracts for the raw buddies solve-SSIM budget gate."""
from __future__ import annotations

import unittest

import numpy as np

from eval_buddies_ssim_budget import (
    BASELINE_BUDGET,
    BUDGETS,
    CACHE_TAG,
    CALIBRATION_IDS,
    CONFIRMATION_IDS,
    DATASET_SEED,
    EXPERIMENT,
    MIN_MARGIN,
    REPAIR_PASSES,
    REPLAY_GROUP,
    REPLAY_SEED,
    SCHEMA_VERSION,
    _fixed_contract,
    array_sha256,
    build_parser,
    canonical_digest,
    paired_confirmation_summary,
    select_budget,
    validate_frozen_config,
)


def _summary(ssim: float, neighbour: float) -> dict[str, float]:
    return {"solve_only_ssim": ssim, "neighbour": neighbour}


class FixedContractTests(unittest.TestCase):
    def test_split_budget_and_solver_contract_are_hard_coded(self) -> None:
        self.assertEqual(REPLAY_GROUP, (10, 12))
        self.assertEqual(CALIBRATION_IDS, tuple(range(10, 18)))
        self.assertEqual(CONFIRMATION_IDS, tuple(range(18, 22)))
        self.assertFalse(set(CALIBRATION_IDS) & set(CONFIRMATION_IDS))
        self.assertEqual(
            BUDGETS, (64, 96, 128, 192, 256, 384, 512, 768, 900)
        )
        self.assertEqual(BASELINE_BUDGET, 512)
        self.assertEqual((REPAIR_PASSES, MIN_MARGIN), (0, 0.0))
        self.assertEqual(CACHE_TAG, "k64")
        self.assertEqual(DATASET_SEED, REPLAY_SEED + 400_000)

    def test_cli_exposes_no_images_budgets_repairs_or_device(self) -> None:
        args = build_parser().parse_args(
            ["calibrate", "--report", "report.json", "--frozen-config", "frozen.json"]
        )
        self.assertEqual(args.phase, "calibrate")
        for forbidden in ("images", "budgets", "repair_passes", "min_margin", "device"):
            self.assertFalse(hasattr(args, forbidden))


class SelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = {budget: _summary(0.10, 0.10) for budget in BUDGETS}

    def test_mean_ssim_is_primary(self) -> None:
        self.rows[64] = _summary(0.11, 0.01)
        self.rows[900] = _summary(0.10, 0.99)
        self.assertEqual(select_budget(self.rows), 64)

    def test_neighbour_breaks_exact_ssim_tie(self) -> None:
        self.rows[96] = _summary(0.12, 0.20)
        self.rows[128] = _summary(0.12, 0.30)
        self.assertEqual(select_budget(self.rows), 128)

    def test_smaller_budget_breaks_exact_metric_tie(self) -> None:
        self.rows[96] = _summary(0.12, 0.30)
        self.rows[128] = _summary(0.12, 0.30)
        self.assertEqual(select_budget(self.rows), 96)

    def test_incomplete_or_nonfinite_grid_fails_closed(self) -> None:
        incomplete = dict(self.rows)
        del incomplete[64]
        with self.assertRaises(ValueError):
            select_budget(incomplete)
        invalid = dict(self.rows)
        invalid[64] = _summary(float("nan"), 0.1)
        with self.assertRaises(ValueError):
            select_budget(invalid)


class ConfirmationAndProvenanceTests(unittest.TestCase):
    def test_paired_delta_aligns_by_image_not_input_order(self) -> None:
        candidate = [
            {"image": 18, "solve_only_ssim": 0.12, "neighbour": 0.20, "placement": 0.02},
            {"image": 19, "solve_only_ssim": 0.11, "neighbour": 0.18, "placement": 0.01},
        ]
        baseline = [
            {"image": 19, "solve_only_ssim": 0.10, "neighbour": 0.17, "placement": 0.01},
            {"image": 18, "solve_only_ssim": 0.10, "neighbour": 0.19, "placement": 0.01},
        ]
        paired = paired_confirmation_summary(candidate, baseline)
        self.assertAlmostEqual(paired["mean_solve_only_ssim"], 0.015)
        self.assertAlmostEqual(paired["mean_neighbour"], 0.01)
        self.assertEqual(paired["positive_ssim_scenes"], 2)

    def test_selected_512_has_exact_zero_paired_delta(self) -> None:
        row = {"image": 18, "solve_only_ssim": 0.1, "neighbour": 0.2, "placement": 0.0}
        paired = paired_confirmation_summary([row], [dict(row)])
        self.assertEqual(paired["mean_solve_only_ssim"], 0.0)

    def test_hashes_include_array_dtype_shape_and_payload_structure(self) -> None:
        values = np.arange(4, dtype=np.int64)
        self.assertEqual(array_sha256(values), array_sha256(values.copy()))
        self.assertNotEqual(array_sha256(values), array_sha256(values.astype(np.int32)))
        self.assertNotEqual(array_sha256(values), array_sha256(values.reshape(2, 2)))
        self.assertEqual(canonical_digest({"b": 2, "a": 1}), canonical_digest({"a": 1, "b": 2}))

    def test_frozen_config_rejects_contract_or_code_drift(self) -> None:
        code = {"solver.py": "abc"}
        frozen = {
            "schema_version": SCHEMA_VERSION,
            "experiment": EXPERIMENT,
            "status": "frozen",
            "calibration_ids": list(CALIBRATION_IDS),
            "confirmation_ids_reserved": list(CONFIRMATION_IDS),
            "contract": _fixed_contract(),
            "selected_budget": 192,
            "code_provenance": code,
            "calibration_scene_provenance_digest": "scene-hash",
            "calibration_report_sha256": "report-hash",
        }
        self.assertEqual(validate_frozen_config(frozen, code), 192)
        changed_contract = dict(frozen)
        changed_contract["contract"] = {**_fixed_contract(), "repair_passes": 1}
        with self.assertRaises(RuntimeError):
            validate_frozen_config(changed_contract, code)
        with self.assertRaises(RuntimeError):
            validate_frozen_config(frozen, {"solver.py": "changed"})


if __name__ == "__main__":
    unittest.main(verbosity=2)

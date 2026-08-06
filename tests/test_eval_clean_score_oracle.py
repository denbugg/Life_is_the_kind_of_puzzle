"""Focused data-free CPU contracts for the frozen clean-score oracle."""
from __future__ import annotations

import ast
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import eval_clean_score_oracle as oracle  # noqa: E402
from imgio import assemble, from_frags  # noqa: E402


def _perfect_graph() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidates = np.empty(
        (oracle.NFRAG, oracle.CANDIDATE_STORAGE_WIDTH), dtype=np.int64
    )
    valid = np.ones_like(candidates, dtype=np.bool_)
    scores = np.full(
        (4, oracle.NFRAG, oracle.CANDIDATE_STORAGE_WIDTH),
        -10.0,
        dtype=np.float32,
    )
    deltas = (-oracle.GRID, oracle.GRID, -1, 1)
    for anchor in range(oracle.NFRAG):
        row, col = divmod(anchor, oracle.GRID)
        exists = (row > 0, row < oracle.GRID - 1, col > 0, col < oracle.GRID - 1)
        targets = [
            anchor + deltas[direction] if exists[direction] else None
            for direction in range(4)
        ]
        ordered: list[int] = []
        for target in targets:
            if target is not None and target not in ordered:
                ordered.append(target)
        for candidate in range(oracle.NFRAG):
            if candidate != anchor and candidate not in ordered:
                ordered.append(candidate)
            if len(ordered) == oracle.CANDIDATE_STORAGE_WIDTH:
                break
        candidates[anchor] = np.asarray(ordered, dtype=np.int64)
        for direction, target in enumerate(targets):
            if target is not None:
                slot = ordered.index(target)
                scores[direction, anchor, slot] = 10.0
    return candidates, valid, scores


def _calibration_payload() -> tuple[dict[str, object], str]:
    provenance = [
        {
            "image": image,
            "validation_name": name,
            "cache": f"E:\\pazzle_work\\image_{image:04d}.npz",
            "cache_sha256": f"{image:064x}",
        }
        for image, name in zip(oracle.CALIBRATION_IDS, oracle.CALIBRATION_NAMES)
    ]
    digest = oracle.canonical_digest(provenance)
    rows = [
        {
            "image": image,
            "board_sha256": f"{image + 100:064x}",
            "solve_only_ssim": oracle.EXPECTED_RR_MEAN_SOLVE_SSIM,
        }
        for image in oracle.CALIBRATION_IDS
    ]
    payload: dict[str, object] = {
        "schema_version": 1,
        "experiment": "raw_buddies_solve_ssim_budget",
        "phase": "calibration",
        "status": "frozen",
        "calibration_ids": list(oracle.CALIBRATION_IDS),
        "confirmation_ids_reserved": [18, 19, 20, 21],
        "contract": copy.deepcopy(oracle.CALIBRATION_CONTRACT),
        "selected_budget": 96,
        "scene_provenance": provenance,
        "scene_provenance_digest": digest,
        "selected_metrics": {
            "solve_only_ssim": oracle.EXPECTED_RR_MEAN_SOLVE_SSIM
        },
        "grid": {
            "96": {"solve_only_ssim": oracle.EXPECTED_RR_MEAN_SOLVE_SSIM}
        },
        "grid_per_image": {"96": rows},
    }
    return payload, digest


def _metric_rows(
    solve_deltas: list[float], final_deltas: list[float]
) -> tuple[list[dict[str, float | int]], list[dict[str, float | int]]]:
    baseline: list[dict[str, float | int]] = []
    candidate: list[dict[str, float | int]] = []
    for index, image in enumerate(oracle.CALIBRATION_IDS):
        base: dict[str, float | int] = {"image": image}
        cand: dict[str, float | int] = {"image": image}
        for metric in oracle.METRICS:
            base[metric] = 1.0
            cand[metric] = 1.0
        cand["solve_only_ssim"] = 1.0 + solve_deltas[index]
        cand["final_ssim"] = 1.0 + final_deltas[index]
        baseline.append(base)
        candidate.append(cand)
    return candidate, list(reversed(baseline))


class FrozenContractTests(unittest.TestCase):
    def test_protocol_is_literal_exact_and_has_no_orientation_search(self) -> None:
        source = (SRC / "eval_clean_score_oracle.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        assignment = next(
            node
            for node in tree.body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "ORACLE_PROTOCOL"
        )
        literal = ast.literal_eval(assignment.value)
        self.assertEqual(literal, oracle.ORACLE_PROTOCOL)
        self.assertEqual(oracle.CALIBRATION_IDS, tuple(range(10, 18)))
        self.assertEqual(
            oracle.CALIBRATION_NAMES,
            tuple(f"img_{6700 + image:06d}.png" for image in oracle.CALIBRATION_IDS),
        )
        self.assertEqual(literal["orientation"], "fixed")
        self.assertFalse(literal["corruption"]["rotation"])
        self.assertFalse(literal["corruption"]["reflection"])
        self.assertEqual(literal["clean_tile_mapping"], "imgio.to_frags(target_uint8)[permutation]")
        self.assertEqual(
            literal["solver"],
            {
                "name": "solve_buddies.solve_buddies_from_scores",
                "max_edges": 96,
                "min_margin": 0.0,
                "repair_passes": 0,
            },
        )
        self.assertEqual(literal["restoration"]["h"], 10)
        self.assertEqual(literal["assembly"], "original_corrupted_upright_tiles_only")

    def test_checkpoint_hashes_and_kill_rule_are_pinned(self) -> None:
        self.assertEqual(
            oracle.EXPECTED_CHECKPOINT_SHA256,
            {
                "ranker": "42685373b1a450a4cb3d7a9b22370dfcfaa2335e9e8ada609f21b7cc64abbfbc",
                "affinity_primary": "708565329c7661a965215d98e85f462a90930071f36a0f75b4813c0c5797ec4f",
                "affinity_secondary": "0fceafdb110bde59149fe1ad1e800a69d116041bc627af369aaecd60be53b6c8",
            },
        )
        self.assertEqual(
            oracle.KILL_RULE,
            {
                "cc_minus_rr_mean_solve_ssim_min": 0.010,
                "cc_minus_rr_mean_final_ssim_min": 0.015,
                "cc_minus_rr_final_wins_min": 6,
                "cc_minus_rr_worst_final_delta_min": -0.020,
            },
        )
        self.assertEqual(
            oracle.CALIBRATION_REPORT_SHA256,
            "3b76d6bed59df13eb98af049c3a756151b4485c2e50b1da88ec50fb7a1dfe305",
        )
        self.assertEqual(
            oracle.SCENE_PROVENANCE_DIGEST,
            "00cd2fdd9189d6453e7c1b215e4ee067b843bc51cdcd0122fa66fdc076779c98",
        )
        provenance = oracle.code_provenance()
        for required in ("candidate_rank.py", "macro_affinity.py", "config.py"):
            self.assertIn(required, provenance)

    def test_cli_exposes_paths_only(self) -> None:
        parser = oracle.build_parser()
        actions = [action for action in parser._actions if action.dest != "help"]
        self.assertEqual(
            {action.dest for action in actions},
            {
                "cache_dir",
                "calibration_report",
                "ranker_checkpoint",
                "affinity_primary",
                "affinity_secondary",
                "output_dir",
                "report",
            },
        )
        self.assertTrue(all(action.type is Path for action in actions))
        args = parser.parse_args([])
        self.assertEqual(args.output_dir, Path("E:/pazzle_work/denoise_oracle"))
        self.assertEqual(
            args.report,
            Path("E:/pazzle_work/denoise_oracle/clean_score_oracle_calibration_v1.json"),
        )
        for forbidden in (
            "ids",
            "device",
            "candidate_k",
            "pair_batch",
            "max_edges",
            "min_margin",
            "repair_passes",
            "threshold",
            "nlm_h",
        ):
            self.assertFalse(hasattr(args, forbidden))


class MappingAndGraphTests(unittest.TestCase):
    def test_clean_tiles_preserve_corrupted_input_tile_ids(self) -> None:
        row_major = np.empty(
            (oracle.NFRAG, oracle.FS, oracle.FS, 3), dtype=np.uint8
        )
        for cell in range(oracle.NFRAG):
            row_major[cell, ..., 0] = cell % 251
            row_major[cell, ..., 1] = (cell // 251) % 251
            row_major[cell, ..., 2] = (3 * cell) % 251
        target = from_frags(row_major)
        permutation = np.roll(np.arange(oracle.NFRAG, dtype=np.int64), 137)
        clean = oracle.clean_tiles_input_order(target, permutation)
        self.assertTrue(np.array_equal(clean, row_major[permutation]))
        tensor = oracle.model_tiles(clean, torch.device("cpu"))
        self.assertEqual(tuple(tensor.shape), (576, 3, 20, 20))
        self.assertEqual(tensor.dtype, torch.float32)

    def test_raw_valid_mask_is_common_and_never_empty(self) -> None:
        scores = np.zeros((4, oracle.NFRAG, 3), dtype=np.float32)
        scores[:, :, 2] = -np.inf
        valid = oracle.raw_common_valid_mask(scores)
        self.assertTrue(np.array_equal(valid[:, :2], np.ones((576, 2), dtype=np.bool_)))
        self.assertFalse(valid[:, 2].any())

        drifted = scores.copy()
        drifted[1, 0, 1] = -np.inf
        with self.assertRaisesRegex(oracle.OracleContractError, "differs by direction"):
            oracle.raw_common_valid_mask(drifted)

        empty = scores.copy()
        empty[:, 7, :] = -np.inf
        with self.assertRaisesRegex(oracle.OracleContractError, "no valid candidate"):
            oracle.raw_common_valid_mask(empty)

    def test_perfect_graph_has_exact_recall_and_rank1(self) -> None:
        candidates, valid, scores = _perfect_graph()
        metrics = oracle.directed_graph_metrics(
            candidates,
            valid,
            scores,
            np.arange(oracle.NFRAG, dtype=np.int64),
        )
        self.assertEqual(metrics["directed_true_edges"], 2208)
        self.assertEqual(metrics["candidate_hits"], 2208)
        self.assertEqual(metrics["rank1_hits"], 2208)
        self.assertEqual(metrics["candidate_recall"], 1.0)
        self.assertEqual(metrics["edge_r1"], 1.0)

    def test_graph_validation_fails_on_duplicate_or_masked_finite_score(self) -> None:
        candidates, valid, scores = _perfect_graph()
        oracle.validate_graph_arrays(candidates, valid, scores, label="test")

        duplicate = candidates.copy()
        duplicate[0, 1] = duplicate[0, 0]
        with self.assertRaisesRegex(oracle.OracleContractError, "duplicate"):
            oracle.validate_graph_arrays(duplicate, valid, scores, label="test")

        masked_valid = valid.copy()
        masked_valid[0, 0] = False
        with self.assertRaisesRegex(oracle.OracleContractError, "finite score in an invalid"):
            oracle.validate_graph_arrays(candidates, masked_valid, scores, label="test")

    def test_dense_conversion_is_cpu_float32(self) -> None:
        candidates = np.zeros((oracle.NFRAG, 2), dtype=np.int64)
        scores = np.zeros((4, oracle.NFRAG, 2), dtype=np.float32)
        observed: dict[str, object] = {}

        def fake_dense(candidates_t: torch.Tensor, scores_t: torch.Tensor):
            observed["candidate_device"] = candidates_t.device.type
            observed["candidate_dtype"] = candidates_t.dtype
            observed["score_device"] = scores_t.device.type
            observed["score_dtype"] = scores_t.dtype
            zeros = torch.zeros((oracle.NFRAG, oracle.NFRAG), dtype=torch.float32)
            return zeros, zeros.clone()

        with patch.object(oracle, "dense_rd", side_effect=fake_dense):
            right, down = oracle.dense_from_graph(candidates, scores)
        self.assertEqual(
            observed,
            {
                "candidate_device": "cpu",
                "candidate_dtype": torch.int64,
                "score_device": "cpu",
                "score_dtype": torch.float32,
            },
        )
        self.assertEqual(right.dtype, np.float32)
        self.assertEqual(down.dtype, np.float32)


class SolverAndCanvasTests(unittest.TestCase):
    def test_solver_contract_is_exact_and_bad_board_fails_closed(self) -> None:
        right = np.zeros((oracle.NFRAG, oracle.NFRAG), dtype=np.float64)
        down = np.zeros_like(right)
        observed: dict[str, object] = {}

        def solver(r: np.ndarray, d: np.ndarray, **kwargs: object):
            observed["r_dtype"] = r.dtype
            observed["d_dtype"] = d.dtype
            observed["kwargs"] = kwargs
            return np.arange(oracle.NFRAG, dtype=np.int64), 12.5

        board, objective, elapsed = oracle.solve_dense(right, down, solver=solver)
        self.assertTrue(np.array_equal(board, np.arange(oracle.NFRAG)))
        self.assertEqual(objective, 12.5)
        self.assertGreaterEqual(elapsed, 0.0)
        self.assertEqual(observed["r_dtype"], np.dtype(np.float32))
        self.assertEqual(observed["d_dtype"], np.dtype(np.float32))
        self.assertEqual(
            observed["kwargs"],
            {"max_edges": 96, "min_margin": 0.0, "repair_passes": 0},
        )

        def bad_solver(r: np.ndarray, d: np.ndarray, **kwargs: object):
            return np.zeros(oracle.NFRAG, dtype=np.int64), 0.0

        with self.assertRaisesRegex(oracle.OracleContractError, "strict tile permutation"):
            oracle.solve_dense(right, down, solver=bad_solver)

    def test_board_metrics_assemble_corrupted_tiles_not_clean_target(self) -> None:
        corrupted_tiles = np.full(
            (oracle.NFRAG, oracle.FS, oracle.FS, 3), 7, dtype=np.uint8
        )
        target = np.full((oracle.IMG, oracle.IMG, 3), 193, dtype=np.uint8)
        scene = SimpleNamespace(
            tiles_uint8=corrupted_tiles,
            target_uint8=target,
            permutation=np.arange(oracle.NFRAG, dtype=np.int64),
        )
        observed: dict[str, np.ndarray] = {}

        def identity_restorer(image: np.ndarray) -> np.ndarray:
            observed["input"] = image.copy()
            return image

        board = np.arange(oracle.NFRAG, dtype=np.int64)
        metrics = oracle.board_metrics(
            scene, board, 1.0, restorer=identity_restorer
        )
        expected = assemble(corrupted_tiles, board)
        self.assertTrue(np.array_equal(observed["input"], expected))
        self.assertFalse(np.array_equal(observed["input"], target))
        self.assertEqual(metrics["solve_only_ssim"], metrics["final_ssim"])
        self.assertEqual(metrics["placement"], 1.0)
        self.assertEqual(metrics["neighbour"], 1.0)


class ProvenanceAndReplayTests(unittest.TestCase):
    def test_calibration_payload_rejects_contract_and_board_drift(self) -> None:
        payload, digest = _calibration_payload()
        with patch.object(oracle, "SCENE_PROVENANCE_DIGEST", digest):
            hashes = oracle.validate_calibration_payload(payload)
            self.assertEqual(tuple(sorted(hashes)), oracle.CALIBRATION_IDS)

            changed = copy.deepcopy(payload)
            changed["calibration_ids"] = list(range(11, 19))
            with self.assertRaisesRegex(oracle.OracleContractError, "IDs"):
                oracle.validate_calibration_payload(changed)

            changed = copy.deepcopy(payload)
            changed["contract"]["repair_passes"] = 1
            with self.assertRaisesRegex(oracle.OracleContractError, "contract"):
                oracle.validate_calibration_payload(changed)

            changed = copy.deepcopy(payload)
            changed["grid_per_image"]["96"][0]["board_sha256"] = "bad"
            with self.assertRaisesRegex(oracle.OracleContractError, "board hashes"):
                oracle.validate_calibration_payload(changed)

    def test_report_sha_is_checked_before_payload_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            path.write_text("{}", encoding="utf-8")
            with patch.object(oracle, "sha256_file", return_value="0" * 64):
                with self.assertRaisesRegex(oracle.OracleContractError, "SHA256 mismatch"):
                    oracle.load_calibration_report(path)

    def test_rr_replay_requires_every_board_hash_and_exact_mean(self) -> None:
        payload, digest = _calibration_payload()
        expected_rows = payload["grid_per_image"]["96"]
        rr_rows = [
            {
                "image": row["image"],
                "board_sha256": row["board_sha256"],
                "solve_only_ssim": oracle.EXPECTED_RR_MEAN_SOLVE_SSIM,
            }
            for row in expected_rows
        ]
        with patch.object(oracle, "SCENE_PROVENANCE_DIGEST", digest):
            result = oracle.verify_rr_replay(list(reversed(rr_rows)), payload)
            self.assertTrue(result["passed"])
            self.assertEqual(
                result["observed_mean_solve_ssim"],
                oracle.EXPECTED_RR_MEAN_SOLVE_SSIM,
            )

            bad_hash = copy.deepcopy(rr_rows)
            bad_hash[2]["board_sha256"] = "f" * 64
            with self.assertRaisesRegex(oracle.OracleContractError, "board hash mismatch"):
                oracle.verify_rr_replay(bad_hash, payload)

            bad_mean = copy.deepcopy(rr_rows)
            bad_mean[0]["solve_only_ssim"] += 1.0e-6
            with self.assertRaisesRegex(oracle.OracleContractError, "mean solve SSIM mismatch"):
                oracle.verify_rr_replay(bad_mean, payload)


class ScoreCacheTests(unittest.TestCase):
    def test_cache_metadata_binds_scoring_code_bundle(self) -> None:
        candidates, _, scores = _perfect_graph()
        scene = SimpleNamespace(
            image_id=10,
            validation_name=oracle.CALIBRATION_NAMES[0],
            cache_path=Path("E:/pazzle_work/image_0010_k64.npz").resolve(),
            cache_sha256="1" * 64,
            candidate_ids=candidates,
            base_scores=scores,
            permutation=np.arange(oracle.NFRAG, dtype=np.int64),
            tiles_uint8=np.zeros(
                (oracle.NFRAG, oracle.FS, oracle.FS, 3), dtype=np.uint8
            ),
            target_uint8=np.zeros((oracle.IMG, oracle.IMG, 3), dtype=np.uint8),
        )
        checkpoints = {
            role: {"sha256": digest}
            for role, digest in oracle.EXPECTED_CHECKPOINT_SHA256.items()
        }
        scoring_code = {"candidate_rank.py": "a" * 64, "macro_affinity.py": "b" * 64}
        metadata = oracle._cache_metadata(
            scene,
            scene.tiles_uint8,
            checkpoints,
            scoring_code,
        )
        self.assertEqual(metadata["scoring_code_sha256"], scoring_code)
        self.assertEqual(
            metadata["scoring_code_digest"], oracle.canonical_digest(scoring_code)
        )

    def test_cache_requires_exact_metadata_shapes_and_dtypes(self) -> None:
        candidates, valid, scores = _perfect_graph()
        scene = SimpleNamespace(candidate_ids=candidates, base_scores=scores)
        metadata = {
            "schema": oracle.SCORE_CACHE_SCHEMA,
            "image": 10,
            "scoring_code_sha256": {"candidate_rank.py": "a" * 64},
            "scoring_code_digest": "b" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"

            def write(rc_scores: np.ndarray) -> None:
                np.savez_compressed(
                    path,
                    metadata_json=np.asarray(
                        json.dumps(metadata, sort_keys=True, separators=(",", ":"))
                    ),
                    rc_scores=rc_scores,
                    cc_candidates=candidates,
                    cc_valid=valid,
                    cc_scores=scores,
                )

            created = oracle._write_score_cache(
                path,
                metadata,
                scene,
                scores,
                candidates,
                valid,
                scores,
            )
            self.assertEqual(created.status, "created")
            cached = oracle._load_clean_score_cache(path, metadata, scene)
            self.assertEqual(cached.status, "reused")
            self.assertEqual(cached.rc_scores.dtype, np.float32)
            with self.assertRaisesRegex(oracle.OracleContractError, "metadata drifted"):
                oracle._load_clean_score_cache(path, {**metadata, "image": 11}, scene)
            changed_code = copy.deepcopy(metadata)
            changed_code["scoring_code_sha256"]["candidate_rank.py"] = "c" * 64
            with self.assertRaisesRegex(oracle.OracleContractError, "metadata drifted"):
                oracle._load_clean_score_cache(path, changed_code, scene)

            write(scores.astype(np.float64))
            with self.assertRaisesRegex(oracle.OracleContractError, "dtype is not float32"):
                oracle._load_clean_score_cache(path, metadata, scene)

            np.savez_compressed(
                path,
                metadata_json=np.asarray(json.dumps(metadata)),
                rc_scores=scores,
                cc_candidates=candidates,
                cc_scores=scores,
            )
            with self.assertRaisesRegex(oracle.OracleContractError, "missing score-cache fields"):
                oracle._load_clean_score_cache(path, metadata, scene)


class DecisionTests(unittest.TestCase):
    def test_paired_summary_aligns_by_image_and_counts_strict_wins(self) -> None:
        solve = [0.010] * 8
        final = [0.025] * 6 + [-0.010] * 2
        candidate, reversed_baseline = _metric_rows(solve, final)
        summary = oracle.paired_summary(
            candidate,
            reversed_baseline,
            candidate_arm="CC",
            baseline_arm="RR",
        )
        self.assertAlmostEqual(
            summary["metrics"]["solve_only_ssim"]["mean_delta"], 0.010
        )
        self.assertEqual(summary["metrics"]["final_ssim"]["wins"], 6)
        self.assertEqual(summary["metrics"]["final_ssim"]["losses"], 2)
        self.assertAlmostEqual(
            summary["metrics"]["final_ssim"]["worst_delta"], -0.010
        )
        with self.assertRaisesRegex(oracle.OracleContractError, "exactly eight"):
            oracle.paired_summary(
                candidate + [dict(candidate[0])],
                reversed_baseline,
                candidate_arm="CC",
                baseline_arm="RR",
            )

    def test_kill_rule_is_inclusive_and_each_condition_is_required(self) -> None:
        passing = {
            "metrics": {
                "solve_only_ssim": {"mean_delta": 0.010},
                "final_ssim": {
                    "mean_delta": 0.015,
                    "wins": 6,
                    "worst_delta": -0.020,
                },
            }
        }
        self.assertTrue(oracle.passes_kill_rule(passing)["passed"])
        mutations = (
            ("solve_only_ssim", "mean_delta", 0.009999),
            ("final_ssim", "mean_delta", 0.014999),
            ("final_ssim", "wins", 5),
            ("final_ssim", "worst_delta", -0.020001),
        )
        for metric, field, value in mutations:
            with self.subTest(metric=metric, field=field):
                changed = copy.deepcopy(passing)
                changed["metrics"][metric][field] = value
                self.assertFalse(oracle.passes_kill_rule(changed)["passed"])

        failing_rc = copy.deepcopy(passing)
        failing_rc["metrics"]["solve_only_ssim"]["mean_delta"] = 0.0
        route = oracle.routing_decision(passing, failing_rc)
        self.assertEqual(route["status"], "pass_headroom")
        self.assertEqual(route["route"], "pursue_learned_pre_denoiser")
        self.assertEqual(
            route["rc_diagnostic"]["suggested_future_input"],
            "denoiser_feeds_both_affinity_encoders_and_ranker",
        )
        self.assertNotIn("status", route["rc_diagnostic"])
        self.assertIn("mean_solve_ssim_delta", route["rc_diagnostic"]["observed_rc_minus_rr"])
        self.assertNotIn(
            "cc_minus_rr_mean_solve_ssim",
            route["rc_diagnostic"]["observed_rc_minus_rr"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

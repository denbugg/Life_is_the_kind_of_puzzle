from __future__ import annotations

import contextlib
import inspect
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


E_TEST_ROOT = Path("E:/pazzle_work/posegraph_e24_selector/test_tmp")
E_TEST_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = str(E_TEST_ROOT)
os.environ["TMP"] = str(E_TEST_ROOT)
os.environ["TMPDIR"] = str(E_TEST_ROOT)
if sys.pycache_prefix is None or Path(sys.pycache_prefix).drive.upper() != "E:":
    sys.pycache_prefix = str(
        Path("E:/pazzle_work/posegraph_e24_selector/test_pycache")
    )

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import eval_e24_context_relation_selector as gate  # noqa: E402
import e24_context_relation_selector as selector  # noqa: E402
import run_e24_context_relation_selector as runner  # noqa: E402


def _raw_arrays() -> tuple[np.ndarray, np.ndarray]:
    ids = np.empty((576, 128), dtype=np.int64)
    for source in range(576):
        ids[source] = (source + 1 + np.arange(128, dtype=np.int64)) % 576
    scores = np.zeros((4, 576, 128), dtype=np.float32)
    return ids, scores


def _prediction_rows(
    fold: int, counts: dict[int, int], *, score_value: float = 0.25
) -> gate.PredictionRows:
    scene_parts: list[np.ndarray] = []
    row_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []
    for image in gate.OOF_FOLDS[fold]:
        count = counts[image]
        scene_parts.append(np.full(count, image, dtype=np.int16))
        row_parts.append(np.arange(count, dtype=np.int64))
        score_parts.append(np.full(count, score_value, dtype=np.float64))
    return gate.PredictionRows(
        scene_ids=np.concatenate(scene_parts),
        row_indices=np.concatenate(row_parts),
        scores=np.concatenate(score_parts),
    )


def _feature_table(
    *, query_sizes: tuple[int, ...] = (2, 3)
) -> selector.RelationFeatureTable:
    rows = sum(query_sizes)
    features = np.zeros((rows, len(selector.FEATURE_NAMES)), dtype=np.float32)
    hypothesis_ids = np.full(rows, selector.NONE_HYPOTHESIS_ID, dtype=np.int64)
    relation_ids = np.full(rows, selector.NONE_RELATION_ID, dtype=np.int64)
    relations = np.zeros((rows, 4), dtype=np.int64)
    row_kind = np.full(rows, selector.ROW_NONE, dtype=np.uint8)
    support = np.zeros(rows, dtype=np.int64)
    offsets = [0]
    cursor = 0
    identity = 0
    for query, size in enumerate(query_sizes):
        u, v = 2 * query, 2 * query + 1
        for local in range(size - 1):
            hypothesis_ids[cursor] = identity
            relation_ids[cursor] = identity
            relations[cursor] = (u, v, local, local + 1)
            row_kind[cursor] = selector.ROW_OFFSET
            support[cursor] = local + 1
            identity += 1
            cursor += 1
        relations[cursor] = (u, v, 0, 0)
        features[cursor, selector.FEATURE_INDEX["is_none"]] = 1.0
        cursor += 1
        offsets.append(cursor)
    return selector.RelationFeatureTable(
        features=features,
        hypothesis_ids=hypothesis_ids,
        relation_ids=relation_ids,
        relations=relations,
        row_kind=row_kind,
        support=support,
        query_offsets=np.asarray(offsets, dtype=np.int64),
        scene_offsets=np.asarray((0, rows), dtype=np.int64),
    )


def _relevance(table: selector.RelationFeatureTable) -> np.ndarray:
    labels = np.zeros(table.rows, dtype=np.int8)
    first_start = int(table.query_offsets[0])
    labels[first_start] = 1
    second_none = int(table.query_offsets[2] - 1)
    labels[second_none] = 1
    return labels


def _run_provenance(fold: int, *, ledger_digit: str = "1") -> dict[str, object]:
    boundary = gate.fold_boundary(fold)
    return {
        "ledger_sha256": ledger_digit * 64,
        "run_contract_sha256": "2" * 64,
        "core_source_sha256": "3" * 64,
        "ordered_feature_schema_sha256": "4" * 64,
        "lightgbm_contract_sha256": "5" * 64,
        "canary_gate_sha256": "6" * 64,
        "train_feature_sha256": {
            image: f"{1000 + image:064x}" for image in boundary.train_ids
        },
        "train_label_manifest_sha256": {
            image: f"{2000 + image:064x}" for image in boundary.train_ids
        },
    }


class FrozenProtocolTests(unittest.TestCase):
    def test_folds_model_gates_storage_and_e25_are_exact(self) -> None:
        self.assertEqual(
            dict(gate.OOF_FOLDS),
            {0: (10, 14), 1: (11, 15), 2: (12, 16), 3: (13, 17)},
        )
        self.assertEqual(gate.CALIBRATION_IDS, tuple(range(10, 18)))
        self.assertEqual(gate.STORAGE_ROOT.drive.upper(), "E:")
        self.assertEqual(gate.FEATURE_CACHE_BYTES_MAX, 4 * 1024**3)
        self.assertEqual(gate.ALL_ARTIFACT_BYTES_MAX, 8 * 1024**3)
        self.assertEqual(gate.GEOMETRY_HYPOTHESES_MAX_EACH, 450_000)
        self.assertEqual(
            gate.STRUCTURAL_GATES["proposed_precision_mean_min"], 0.70
        )
        self.assertEqual(
            gate.STRUCTURAL_GATES["true_relation_recall_worst_min"], 0.50
        )
        self.assertEqual(
            gate.STRUCTURAL_GATES["exact_connected_coverage_worst_min"], 0.35
        )
        self.assertEqual(gate.END_TO_END_GATES["final_wins_min"], 5)
        self.assertEqual(len(gate.E25_SEALED_IDS), 48)
        self.assertEqual(len(set(gate.E25_SEALED_IDS)), 48)
        self.assertEqual(
            gate.E25_NEWLINE_LIST_SHA256,
            "407a6326ceeec2e8cc78106b74c2f10c46a55143ea488a30f7bac66e2b373caa",
        )

    def test_frozen_lightgbm_configuration_has_no_selection_callback(self) -> None:
        for fold in range(4):
            config = gate.frozen_lightgbm_config(fold)
            self.assertEqual(config["objective"], "lambdarank")
            self.assertEqual(config["label_gain"], [0, 1])
            self.assertEqual(config["eval_at"], [1])
            self.assertEqual(config["n_estimators"], 256)
            self.assertEqual(config["learning_rate"], 0.05)
            self.assertEqual(config["num_leaves"], 31)
            self.assertEqual(config["min_child_samples"], 200)
            self.assertEqual(config["lambdarank_truncation_level"], 30)
            self.assertEqual(config["random_state"], 1234 + fold)
            self.assertEqual(config["data_random_seed"], 1234 + fold)
            self.assertEqual(config["feature_fraction_seed"], 1234 + fold)
            self.assertNotIn("early_stopping", config)
            self.assertNotIn("callbacks", config)
            self.assertNotIn("valid_set", config)
        self.assertEqual(gate.validate_lightgbm_runtime_version("4.6.0"), "4.6.0")
        with self.assertRaisesRegex(gate.E24EvaluatorContractError, "exactly 4.6.0"):
            gate.validate_lightgbm_runtime_version("4.5.0")

    def test_source_orders_commit_verification_before_label_access(self) -> None:
        self.assertFalse(hasattr(gate, "evaluate_heldout_after_commit"))
        source = inspect.getsource(gate.evaluate_oof_after_all_commits)
        self.assertLess(
            source.index("verify_all_oof_commits"),
            source.index("permutation_loader(image)"),
        )
        worker_signature = tuple(
            inspect.signature(gate.load_feature_worker_raw_npz).parameters
        )
        self.assertEqual(worker_signature, ("path",))
        lowered = inspect.getsource(gate.load_feature_worker_raw_npz).lower()
        for forbidden in ("rawscene", "permutation", "target", "report", "label"):
            self.assertNotIn(f'archive["{forbidden}"]', lowered)

    def test_runtime_paths_must_all_resolve_under_frozen_e24_root(self) -> None:
        valid = {
            "PYTHONPYCACHEPREFIX": str(gate.STORAGE_ROOT / "pycache"),
            "TEMP": str(gate.STORAGE_ROOT / "tmp"),
            "TMP": str(gate.STORAGE_ROOT / "tmp"),
            "TMPDIR": str(gate.STORAGE_ROOT / "tmp"),
        }
        self.assertEqual(set(gate.validate_e24_runtime_paths(valid)), set(valid))
        for key in valid:
            bad = dict(valid)
            bad[key] = "C:/temp"
            with self.subTest(key=key), self.assertRaisesRegex(
                gate.E24EvaluatorContractError, "E:"
            ):
                gate.validate_e24_runtime_paths(bad)


class CoreAdversarialBoundaryTests(unittest.TestCase):
    def test_core_lightgbm_contract_has_every_frozen_parameter(self) -> None:
        config = selector.LIGHTGBM_CONFIG
        expected = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "eval_at": [1],
            "label_gain": [0, 1],
            "n_estimators": 256,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 200,
            "max_bin": 255,
            "feature_fraction": 1.0,
            "bagging_fraction": 1.0,
            "lambda_l2": 1.0,
            "lambda_l1": 0.0,
            "lambdarank_truncation_level": 30,
            "lambdarank_norm": True,
            "n_jobs": 8,
            "deterministic": True,
            "force_col_wise": True,
        }
        for key, value in expected.items():
            self.assertIn(key, config)
            self.assertEqual(config[key], value)

    def test_core_fit_uses_all_three_fold_seeds_and_no_validation_callback(self) -> None:
        table = _feature_table()
        labels = _relevance(table)
        weights = selector.balanced_query_row_weights(table, labels)
        captured: dict[str, object] = {}

        class FakeRanker:
            def __init__(self, **config: object) -> None:
                captured["config"] = config

            def fit(self, *args: object, **kwargs: object) -> "FakeRanker":
                captured["fit_args"] = args
                captured["fit_kwargs"] = kwargs
                return self

        fake_lightgbm = SimpleNamespace(LGBMRanker=FakeRanker)
        with mock.patch.dict(sys.modules, {"lightgbm": fake_lightgbm}):
            selector.fit_lambdarank(
                table, labels, fold=3, row_weights=weights
            )
        config = captured["config"]
        self.assertEqual(config["random_state"], 1237)
        self.assertEqual(config["data_random_seed"], 1237)
        self.assertEqual(config["feature_fraction_seed"], 1237)
        fit_kwargs = captured["fit_kwargs"]
        self.assertEqual(set(fit_kwargs), {"group", "sample_weight"})
        self.assertNotIn("eval_set", fit_kwargs)
        self.assertNotIn("callbacks", fit_kwargs)

    def test_width_one_seam_uses_physical_19_and_0_boundary_pixels(self) -> None:
        rgb = np.zeros((2, 20, 20, 3), dtype=np.float32)
        lab = np.zeros_like(rgb)
        # Physical right/left boundary matches exactly.  The one-pixel inset
        # (18/1) intentionally differs, so an off-by-one implementation fails.
        rgb[0, :, 19, :] = 4.0
        rgb[1, :, 0, :] = 4.0
        rgb[0, :, 18, :] = 1.0
        rgb[1, :, 1, :] = 2.0
        lab[0, :, 19, :] = 7.0
        lab[1, :, 0, :] = 7.0
        lab[0, :, 18, :] = 1.0
        lab[1, :, 1, :] = 3.0
        claim = SimpleNamespace(first=0, second=1, dx=1, dy=0)
        rgb_mse, _gradient_mse, ncc, lab_mse = selector._seam_values(
            rgb, lab, claim, 1
        )
        self.assertEqual(rgb_mse, 0.0)
        self.assertEqual(lab_mse, 0.0)
        self.assertAlmostEqual(ncc, 0.0)

        down_rgb = np.transpose(rgb, (0, 2, 1, 3)).copy()
        down_lab = np.transpose(lab, (0, 2, 1, 3)).copy()
        down = SimpleNamespace(first=0, second=1, dx=0, dy=1)
        down_mse, _down_gradient, _down_ncc, down_lab_mse = selector._seam_values(
            down_rgb, down_lab, down, 1
        )
        self.assertEqual(down_mse, 0.0)
        self.assertEqual(down_lab_mse, 0.0)

    def test_rejected_dsu_operation_does_not_path_compress_or_mutate_delta(self) -> None:
        dsu = object.__new__(selector._PotentialDSU)
        dsu.parent = {0: 0, 1: 0, 2: 1}
        dsu.delta = {0: (0, 0), 1: (1, 0), 2: (1, 0)}
        before_parent = dict(dsu.parent)
        before_delta = dict(dsu.delta)
        selection = selector.SelectedRelation(
            hypothesis_id=0,
            relation_id=0,
            u=2,
            v=0,
            dr=999,
            dc=999,
            score=1.0,
            none_score=0.0,
            margin=1.0,
            support=1,
        )
        accepted, reason, tree, cycle = dsu.try_accept(selection)
        self.assertEqual((accepted, reason, tree, cycle), (False, "conflict", False, False))
        self.assertEqual(dsu.parent, before_parent)
        self.assertEqual(dsu.delta, before_delta)

    def test_equal_score_ties_ignore_support_and_use_canonical_relation_only(self) -> None:
        table = _feature_table(query_sizes=(3, 2))
        # Query one offset rows have supports 1 and 2 but identical scores.
        # Canonical (dr,dc)=(0,1) must win despite lower support.
        scores = np.asarray((1.0, 1.0, 0.0, 1.0, 0.0), dtype=np.float64)
        selected, attempted, _cap = selector.select_pair_winners(
            table, scores, component_count=4
        )
        self.assertEqual(selected[0].relation, (0, 1, 0, 1))
        self.assertEqual(tuple(item.relation for item in attempted), tuple(item.relation for item in selected))


class RawFeatureWorkerBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(dir=E_TEST_ROOT)
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write(self, **arrays: np.ndarray) -> Path:
        path = self.root / "raw.npz"
        np.savez(path, **arrays)
        return path

    def test_exact_whitelist_loads_detached_read_only_arrays(self) -> None:
        ids, scores = _raw_arrays()
        path = self._write(candidate_ids=ids, candidate_scores=scores)
        loaded = gate.load_feature_worker_raw_npz(path)
        self.assertEqual(loaded.candidate_ids.dtype, np.int64)
        self.assertEqual(loaded.candidate_scores.dtype, np.float32)
        self.assertFalse(loaded.candidate_ids.flags.writeable)
        self.assertFalse(loaded.candidate_scores.flags.writeable)
        ids[0, 0] = 500
        scores[0, 0, 0] = 500.0
        self.assertNotEqual(loaded.candidate_ids[0, 0], 500)
        self.assertNotEqual(loaded.candidate_scores[0, 0, 0], 500.0)
        self.assertRegex(loaded.source_sha256, r"^[0-9a-f]{64}$")

    def test_rejects_every_forbidden_extra_npz_field(self) -> None:
        ids, scores = _raw_arrays()
        for field in (
            "labels",
            "permutation",
            "target",
            "clean",
            "pure",
            "shift",
            "board",
            "report",
            "oracle",
        ):
            with self.subTest(field=field):
                path = self.root / f"bad_{field}.npz"
                np.savez(
                    path,
                    candidate_ids=ids,
                    candidate_scores=scores,
                    **{field: np.zeros(1, dtype=np.int64)},
                )
                with self.assertRaisesRegex(
                    gate.E24EvaluatorContractError, "whitelist mismatch"
                ):
                    gate.load_feature_worker_raw_npz(path)

    def test_rejects_missing_wrong_shape_dtype_and_out_of_range(self) -> None:
        ids, scores = _raw_arrays()
        cases = (
            {"candidate_ids": ids},
            {
                "candidate_ids": ids.astype(np.int32),
                "candidate_scores": scores,
            },
            {
                "candidate_ids": ids,
                "candidate_scores": scores.astype(np.float64),
            },
            {
                "candidate_ids": ids[:, :-1],
                "candidate_scores": scores,
            },
        )
        for index, values in enumerate(cases):
            path = self.root / f"bad_shape_{index}.npz"
            np.savez(path, **values)
            with self.subTest(index=index), self.assertRaises(
                gate.E24EvaluatorContractError
            ):
                gate.load_feature_worker_raw_npz(path)
        ids[0, 0] = 576
        path = self.root / "bad_range.npz"
        np.savez(path, candidate_ids=ids, candidate_scores=scores)
        with self.assertRaisesRegex(gate.E24EvaluatorContractError, "0..575"):
            gate.load_feature_worker_raw_npz(path)

    def test_rejects_nan_positive_infinity_and_direction_mask_drift(self) -> None:
        ids, scores = _raw_arrays()
        for index, value in enumerate((np.nan, np.inf)):
            bad = scores.copy()
            bad[0, 0, 0] = value
            path = self.root / f"nonfinite_{index}.npz"
            np.savez(path, candidate_ids=ids, candidate_scores=bad)
            with self.assertRaisesRegex(
                gate.E24EvaluatorContractError, "finite values or -inf"
            ):
                gate.load_feature_worker_raw_npz(path)
        drift = scores.copy()
        drift[0, 0, 0] = -np.inf
        path = self.root / "mask_drift.npz"
        np.savez(path, candidate_ids=ids, candidate_scores=drift)
        with self.assertRaisesRegex(gate.E24EvaluatorContractError, "masks differ"):
            gate.load_feature_worker_raw_npz(path)

    def test_rejects_c_drive_before_open(self) -> None:
        with mock.patch("numpy.load") as loader:
            with self.assertRaisesRegex(gate.E24EvaluatorContractError, "E:"):
                gate.load_feature_worker_raw_npz("C:/forbidden/raw.npz")
            loader.assert_not_called()

    def test_authenticated_sanitizer_never_parses_original_forbidden_archive(self) -> None:
        ids, scores = _raw_arrays()
        original = self.root / "original_raw_with_labels.npz"
        np.savez(
            original,
            features=np.zeros((1, 2), dtype=np.float32),
            labels=np.zeros(1, dtype=np.int64),
            anchors=np.zeros(1, dtype=np.int64),
            directions=np.zeros(1, dtype=np.int64),
            predicted=np.zeros(1, dtype=np.int64),
            permutation=np.arange(576, dtype=np.int64),
            candidate_ids=ids,
            candidate_scores=scores,
        )
        original_sha = hashlib.sha256(original.read_bytes()).hexdigest()
        sanitized = self.root / "sanitized.npz"
        manifest = self.root / "sanitized.json"
        real_load = np.load
        opened: list[Path] = []

        def tracking_load(path: object, *args: object, **kwargs: object) -> object:
            opened.append(Path(path))
            return real_load(path, *args, **kwargs)

        with mock.patch.object(gate.np, "load", side_effect=tracking_load):
            artifact = gate.sanitize_raw_candidate_cache(
                scene_id=10,
                original_raw_cache_path=original,
                expected_original_sha256=original_sha,
                source_scene_contract_sha256="c" * 64,
                candidate_ids=ids,
                candidate_scores=scores,
                sanitized_npz_path=sanitized,
                manifest_path=manifest,
            )
        self.assertNotIn(original, opened)
        self.assertTrue(opened)
        self.assertTrue(all(path == sanitized for path in opened))
        self.assertEqual(artifact.original_sha256, original_sha)
        self.assertEqual(artifact.arrays.candidate_ids.shape, (576, 128))
        with np.load(sanitized, allow_pickle=False) as stored:
            self.assertEqual(set(stored.files), {"candidate_ids", "candidate_scores"})
        payload = json.loads(manifest.read_text(encoding="ascii"))
        self.assertFalse(payload["original_raw_cache"]["parsed_by_sanitizer"])
        self.assertEqual(payload["original_raw_cache"]["sha256"], original_sha)

    def test_original_archive_readers_open_only_literal_capability_members(self) -> None:
        ids, scores = _raw_arrays()
        flat_scores = np.ascontiguousarray(
            scores.transpose(1, 0, 2).reshape(576 * 4, 128), dtype=np.float32
        )
        permutation = np.arange(576, dtype=np.int64)[::-1].copy()
        original = self.root / "original_literal_capabilities.npz"
        np.savez(
            original,
            features=np.array([object()], dtype=object),
            labels=np.array([object()], dtype=object),
            target=np.array([object()], dtype=object),
            permutation=permutation,
            candidate_ids=ids,
            candidate_scores=flat_scores,
        )
        original_sha = hashlib.sha256(original.read_bytes()).hexdigest()
        real_open = gate.zipfile.ZipFile.open
        opened: list[str] = []

        def tracking_open(
            archive: object, member: object, *args: object, **kwargs: object
        ) -> object:
            opened.append(getattr(member, "filename", str(member)))
            return real_open(archive, member, *args, **kwargs)

        with mock.patch.object(gate.zipfile.ZipFile, "open", new=tracking_open):
            raw = gate.load_original_raw_candidate_members(
                original, expected_sha256=original_sha
            )
        self.assertEqual(opened, ["candidate_ids.npy", "candidate_scores.npy"])
        np.testing.assert_array_equal(raw.candidate_ids, ids)
        np.testing.assert_array_equal(raw.candidate_scores, scores)

        opened.clear()
        with mock.patch.object(gate.zipfile.ZipFile, "open", new=tracking_open):
            loaded_permutation = gate.load_original_permutation_member(
                original, expected_sha256=original_sha
            )
        self.assertEqual(opened, ["permutation.npy"])
        np.testing.assert_array_equal(loaded_permutation, permutation)
        self.assertFalse(loaded_permutation.flags.writeable)

    def test_sanitized_npz_is_deterministic_resumable_and_source_bound(self) -> None:
        ids, scores = _raw_arrays()
        original = self.root / "original.npz"
        np.savez(original, candidate_ids=ids, candidate_scores=scores, labels=np.zeros(1))
        original_sha = hashlib.sha256(original.read_bytes()).hexdigest()

        hashes: list[str] = []
        for index in range(2):
            artifact = gate.sanitize_raw_candidate_cache(
                scene_id=11,
                original_raw_cache_path=original,
                expected_original_sha256=original_sha,
                source_scene_contract_sha256="d" * 64,
                candidate_ids=ids,
                candidate_scores=scores,
                sanitized_npz_path=self.root / f"sanitized_{index}.npz",
                manifest_path=self.root / f"sanitized_{index}.json",
            )
            hashes.append(artifact.npz_sha256)
        self.assertEqual(hashes[0], hashes[1])

        resumed = gate.sanitize_raw_candidate_cache(
            scene_id=11,
            original_raw_cache_path=original,
            expected_original_sha256=original_sha,
            source_scene_contract_sha256="d" * 64,
            candidate_ids=ids,
            candidate_scores=scores,
            sanitized_npz_path=self.root / "sanitized_0.npz",
            manifest_path=self.root / "sanitized_0.json",
        )
        self.assertEqual(resumed.npz_sha256, hashes[0])

        original.write_bytes(b"tampered-original")
        with self.assertRaisesRegex(
            gate.E24EvaluatorContractError, "original raw-cache provenance mismatch"
        ):
            gate.verify_sanitized_raw_artifact(self.root / "sanitized_1.json")

    def test_source_sha_failure_creates_no_sanitized_output(self) -> None:
        ids, scores = _raw_arrays()
        original = self.root / "original_bad_sha.npz"
        np.savez(original, candidate_ids=ids, candidate_scores=scores)
        sanitized = self.root / "never.npz"
        manifest = self.root / "never.json"
        with self.assertRaisesRegex(gate.E24EvaluatorContractError, "SHA mismatch"):
            gate.sanitize_raw_candidate_cache(
                scene_id=12,
                original_raw_cache_path=original,
                expected_original_sha256="0" * 64,
                source_scene_contract_sha256="e" * 64,
                candidate_ids=ids,
                candidate_scores=scores,
                sanitized_npz_path=sanitized,
                manifest_path=manifest,
            )
        self.assertFalse(sanitized.exists())
        self.assertFalse(manifest.exists())


class FoldIsolationTests(unittest.TestCase):
    def test_every_fold_has_six_training_and_two_heldout_scenes(self) -> None:
        observed: list[int] = []
        for fold in range(4):
            boundary = gate.fold_boundary(fold)
            self.assertEqual(len(boundary.train_ids), 6)
            self.assertEqual(len(boundary.heldout_ids), 2)
            self.assertFalse(set(boundary.train_ids) & set(boundary.heldout_ids))
            observed.extend(boundary.heldout_ids)
        self.assertEqual(sorted(observed), list(gate.CALIBRATION_IDS))

    def test_training_partition_accepts_only_exact_six_label_scenes(self) -> None:
        for fold in range(4):
            boundary = gate.fold_boundary(fold)
            result = gate.validate_fold_training_partition(
                fold,
                feature_scene_ids=gate.CALIBRATION_IDS,
                label_scene_ids=boundary.train_ids,
            )
            self.assertEqual(result, boundary)

    def test_heldout_label_injection_fails_closed(self) -> None:
        boundary = gate.fold_boundary(0)
        labels = boundary.train_ids[:-1] + (boundary.heldout_ids[0],)
        with self.assertRaisesRegex(gate.E24EvaluatorContractError, "exactly the six"):
            gate.validate_fold_training_partition(
                0,
                feature_scene_ids=gate.CALIBRATION_IDS,
                label_scene_ids=labels,
            )

    def test_missing_duplicate_or_e25_scene_fails_closed(self) -> None:
        boundary = gate.fold_boundary(1)
        bad_features = gate.CALIBRATION_IDS[:-1] + (gate.CALIBRATION_IDS[-2],)
        with self.assertRaises(gate.E24EvaluatorContractError):
            gate.validate_fold_training_partition(
                1,
                feature_scene_ids=bad_features,
                label_scene_ids=boundary.train_ids,
            )
        with self.assertRaises(gate.E24EvaluatorContractError):
            gate.validate_fold_training_partition(
                1,
                feature_scene_ids=gate.CALIBRATION_IDS,
                label_scene_ids=boundary.train_ids[:-1] + (gate.E25_SEALED_IDS[0],),
            )

    def test_training_batch_balances_categories_inside_each_scene_then_scenes(self) -> None:
        tables = {
            image: _feature_table(query_sizes=(2, 2 + image % 4))
            for image in gate.CALIBRATION_IDS
        }
        boundary = gate.fold_boundary(0)
        labels = {image: _relevance(tables[image]) for image in boundary.train_ids}
        batch = gate.build_fold_training_batch(
            0, tables_by_scene=tables, relevance_by_scene=labels
        )
        masses: list[float] = []
        raw_weights: list[np.ndarray] = []
        for image in boundary.train_ids:
            start, stop = batch.scene_row_offsets[image]
            scene_weights = batch.row_weights[start:stop]
            masses.append(float(scene_weights.sum(dtype=np.float64)))
            table = tables[image]
            scene_labels = labels[image]
            _validated, categories = gate._validate_scene_relevance(
                table, scene_labels, image=image
            )
            raw_weights.append(
                gate._per_scene_balanced_weights(table, categories)
            )
            category_masses = {False: 0.0, True: 0.0}
            local_cursor = 0
            for query_start, query_stop in zip(
                table.query_offsets[:-1], table.query_offsets[1:]
            ):
                size = int(query_stop - query_start)
                none_positive = bool(scene_labels[int(query_stop) - 1] == 1)
                category_masses[none_positive] += float(
                    scene_weights[local_cursor : local_cursor + size].sum(
                        dtype=np.float64
                    )
                )
                local_cursor += size
            self.assertAlmostEqual(category_masses[False], category_masses[True], places=5)
        self.assertTrue(
            np.array_equal(
                batch.row_weights,
                gate._canonical_float32_fold_weights(raw_weights),
            )
        )
        self.assertAlmostEqual(float(batch.row_weights.mean()), 1.0, places=6)

    def test_large_float32_reduction_is_exact_and_one_ulp_drift_fails(self) -> None:
        raw = (
            np.full(2_000_000, 2.7e-6, dtype=np.float64),
            np.full(223_009, 1.3e-5, dtype=np.float64),
        )
        expected = gate._canonical_float32_fold_weights(raw)
        ideal64 = np.concatenate(raw)
        ideal64 /= float(ideal64.mean())
        self.assertFalse(
            np.allclose(
                expected.astype(np.float64),
                ideal64,
                rtol=2.0e-7,
                atol=2.0e-7,
            )
        )
        self.assertTrue(
            np.array_equal(expected, gate._canonical_float32_fold_weights(raw))
        )
        drifted = expected.copy()
        drifted[0] = np.nextafter(drifted[0], np.float32(np.inf))
        self.assertFalse(np.array_equal(drifted, expected))

    def test_training_batch_rejects_scene_without_both_query_categories(self) -> None:
        tables = {image: _feature_table() for image in gate.CALIBRATION_IDS}
        boundary = gate.fold_boundary(0)
        labels = {image: _relevance(tables[image]) for image in boundary.train_ids}
        first = boundary.train_ids[0]
        labels[first] = labels[first].copy()
        labels[first][:] = 0
        for stop in tables[first].query_offsets[1:]:
            labels[first][int(stop) - 1] = 1
        with self.assertRaisesRegex(gate.E24EvaluatorContractError, "both"):
            gate.build_fold_training_batch(
                0, tables_by_scene=tables, relevance_by_scene=labels
            )

    def test_fit_wrapper_passes_only_six_scene_labels_and_frozen_weights(self) -> None:
        tables = {image: _feature_table() for image in gate.CALIBRATION_IDS}
        boundary = gate.fold_boundary(2)
        labels = {image: _relevance(tables[image]) for image in boundary.train_ids}
        sentinel = object()
        with mock.patch.object(
            gate.selector, "fit_lambdarank", return_value=sentinel
        ) as fit:
            model, batch = gate.fit_oof_fold(
                2, tables_by_scene=tables, relevance_by_scene=labels
            )
        self.assertIs(model, sentinel)
        self.assertEqual(batch.boundary.train_ids, boundary.train_ids)
        fit.assert_called_once()
        args, kwargs = fit.call_args
        self.assertIs(args[0], batch.table)
        self.assertIs(args[1], batch.relevance)
        self.assertEqual(kwargs["fold"], 2)
        self.assertIs(kwargs["row_weights"], batch.row_weights)


class AtomicPredictionBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(dir=E_TEST_ROOT)
        self.root = Path(self.temp.name)
        self.model = self.root / "model.txt"
        self.model.write_bytes(b"synthetic-fixed-model\n")
        self.prediction = self.root / "predictions.npz"
        self.commit = self.root / "commit.json"
        self.fold = 0
        self.counts = {10: 3, 14: 2}
        self.features = {10: "a" * 64, 14: "b" * 64}

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _commit(self, rows: gate.PredictionRows | None = None) -> dict[str, object]:
        return gate.commit_fold_predictions(
            fold=self.fold,
            model_path=self.model,
            prediction_path=self.prediction,
            commit_path=self.commit,
            run_provenance=_run_provenance(self.fold),
            feature_sha256=self.features,
            row_counts=self.counts,
            rows=rows or _prediction_rows(self.fold, self.counts),
        )

    def test_valid_transaction_roundtrips_and_is_bound_to_model(self) -> None:
        payload = self._commit()
        verified = gate.verify_prediction_commit(self.commit, expected_fold=0)
        self.assertEqual(payload["status"], "complete")
        self.assertEqual(verified.heldout_ids, (10, 14))
        self.assertEqual(verified.row_counts, self.counts)
        self.assertEqual(len(verified.predictions.scores), 5)
        self.assertEqual(verified.model_sha256, payload["model"]["sha256"])
        self.assertEqual(
            dict(verified.run_provenance),
            gate.normalize_fold_run_provenance(0, _run_provenance(0)),
        )
        self.assertFalse(verified.predictions.scores.flags.writeable)

    def test_prediction_npz_is_deterministic_and_rejects_timestamped_equivalent(self) -> None:
        rows = _prediction_rows(self.fold, self.counts)
        model_sha = hashlib.sha256(self.model.read_bytes()).hexdigest()
        first = gate._prediction_npz_bytes(
            fold=self.fold, model_sha256=model_sha, rows=rows
        )
        second = gate._prediction_npz_bytes(
            fold=self.fold, model_sha256=model_sha, rows=rows
        )
        self.assertEqual(first, second)
        noncanonical = self.root / "timestamped_predictions.npz"
        np.savez(
            noncanonical,
            schema=np.asarray(gate.PREDICTION_SCHEMA),
            fold=np.asarray(self.fold, dtype=np.int8),
            model_sha256=np.asarray(model_sha),
            scene_ids=rows.scene_ids,
            row_indices=rows.row_indices,
            scores=rows.scores,
        )
        with self.assertRaisesRegex(
            gate.E24EvaluatorContractError, "not canonical"
        ):
            gate._load_prediction_npz(noncanonical)

    def test_stale_run_provenance_blocks_label_access(self) -> None:
        self._commit()
        with self.assertRaisesRegex(
            gate.E24EvaluatorContractError, "stale/different run provenance"
        ):
            gate.verify_prediction_commit(
                self.commit,
                expected_fold=0,
                expected_run_provenance=_run_provenance(0, ledger_digit="9"),
            )

    def test_transaction_exact_orphans_are_resumable_but_drift_is_rejected(self) -> None:
        first = self._commit()
        second = self._commit()
        self.assertEqual(first, second)
        self.commit.unlink()
        with self.prediction.open("ab") as stream:
            stream.write(b"drift")
        with self.assertRaisesRegex(
            gate.E24EvaluatorContractError, "existing deterministic artifact differs"
        ):
            self._commit()

    def test_nonfinite_or_incomplete_rows_fail_before_artifact_commit(self) -> None:
        for name, rows in (
            (
                "nan",
                _prediction_rows(self.fold, self.counts, score_value=float("nan")),
            ),
            (
                "inf",
                _prediction_rows(self.fold, self.counts, score_value=float("inf")),
            ),
        ):
            with self.subTest(name=name), self.assertRaisesRegex(
                gate.E24EvaluatorContractError, "non-finite"
            ):
                self._commit(rows)
            self.assertFalse(self.prediction.exists())
            self.assertFalse(self.commit.exists())
        incomplete = _prediction_rows(self.fold, self.counts)
        incomplete = gate.PredictionRows(
            scene_ids=incomplete.scene_ids[:-1],
            row_indices=incomplete.row_indices[:-1],
            scores=incomplete.scores[:-1],
        )
        with self.assertRaisesRegex(gate.E24EvaluatorContractError, "incomplete"):
            self._commit(incomplete)
        self.assertFalse(self.prediction.exists())
        self.assertFalse(self.commit.exists())
    def test_missing_duplicate_or_noncanonical_row_fails(self) -> None:
        rows = _prediction_rows(self.fold, self.counts)
        indices = rows.row_indices.copy()
        indices[1] = 0
        bad = gate.PredictionRows(rows.scene_ids, indices, rows.scores)
        with self.assertRaisesRegex(
            gate.E24EvaluatorContractError, "complete contiguous held-out blocks"
        ):
            self._commit(bad)

        # Per-scene row indices remain locally canonical, but the two scene
        # blocks are interleaved. Literal artifact identity must reject this.
        order = np.asarray((0, 3, 1, 4, 2), dtype=np.int64)
        interleaved = gate.PredictionRows(
            np.ascontiguousarray(rows.scene_ids[order]),
            np.ascontiguousarray(rows.row_indices[order]),
            np.ascontiguousarray(rows.scores[order]),
        )
        with self.assertRaisesRegex(
            gate.E24EvaluatorContractError, "complete contiguous held-out blocks"
        ):
            self._commit(interleaved)

    def test_single_fold_verifier_has_no_label_opening_capability(self) -> None:
        self.assertFalse(hasattr(gate, "evaluate_heldout_after_commit"))
        self.assertNotIn(
            "permutation_loader",
            inspect.signature(gate.verify_prediction_commit).parameters,
        )
        with self.assertRaisesRegex(gate.E24EvaluatorContractError, "absent"):
            gate.verify_prediction_commit(self.commit, expected_fold=0)
        self._commit()
        verified = gate.verify_prediction_commit(self.commit, expected_fold=0)
        self.assertEqual(verified.heldout_ids, (10, 14))

    def test_model_tamper_blocks_label_access(self) -> None:
        self._commit()
        self.model.write_bytes(b"tampered-model\n")
        with self.assertRaisesRegex(gate.E24EvaluatorContractError, "model .* mismatch"):
            gate.verify_prediction_commit(self.commit, expected_fold=0)

    def test_prediction_tamper_blocks_label_access(self) -> None:
        self._commit()
        with self.prediction.open("ab") as stream:
            stream.write(b"tamper")
        with self.assertRaisesRegex(
            gate.E24EvaluatorContractError, "prediction (size|SHA) mismatch"
        ):
            gate.verify_prediction_commit(self.commit, expected_fold=0)

    def test_noncanonical_or_incomplete_commit_blocks_label_access(self) -> None:
        self._commit()
        payload = json.loads(self.commit.read_text(encoding="ascii"))
        self.commit.unlink()
        self.commit.write_text(json.dumps(payload, indent=2), encoding="ascii")
        with self.assertRaisesRegex(gate.E24EvaluatorContractError, "canonical JSON"):
            gate.verify_prediction_commit(self.commit, expected_fold=0)

        payload["status"] = "partial"
        canonical = gate._canonical_json_bytes(payload)
        self.commit.write_bytes(canonical)
        with self.assertRaisesRegex(gate.E24EvaluatorContractError, "identity drifted"):
            gate.verify_prediction_commit(self.commit, expected_fold=0)

    def test_wrong_fold_blocks_label_access(self) -> None:
        self._commit()
        with self.assertRaisesRegex(gate.E24EvaluatorContractError, "wrong fold"):
            gate.verify_prediction_commit(self.commit, expected_fold=1)

    def test_c_drive_artifact_paths_fail_before_prediction_write(self) -> None:
        with self.assertRaisesRegex(gate.E24EvaluatorContractError, "E:"):
            gate.commit_fold_predictions(
                fold=0,
                model_path=self.model,
                prediction_path="C:/forbidden/predictions.npz",
                commit_path=self.commit,
                run_provenance=_run_provenance(0),
                feature_sha256=self.features,
                row_counts=self.counts,
                rows=_prediction_rows(self.fold, self.counts),
            )
        self.assertFalse(self.prediction.exists())
        self.assertFalse(self.commit.exists())


class GlobalOOFBarrierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(dir=E_TEST_ROOT)
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _commit_all(self) -> dict[int, Path]:
        commits: dict[int, Path] = {}
        for fold, heldout in gate.OOF_FOLDS.items():
            directory = self.root / f"fold_{fold}"
            directory.mkdir()
            model = directory / "model.txt"
            model.write_bytes(f"synthetic-model-{fold}\n".encode("ascii"))
            counts = {image: fold + 2 for image in heldout}
            features = {
                heldout[0]: f"{2 * fold + 1:064x}",
                heldout[1]: f"{2 * fold + 2:064x}",
            }
            prediction = directory / "predictions.npz"
            commit = directory / "commit.json"
            gate.commit_fold_predictions(
                fold=fold,
                model_path=model,
                prediction_path=prediction,
                commit_path=commit,
                run_provenance=_run_provenance(fold),
                feature_sha256=features,
                row_counts=counts,
                rows=_prediction_rows(fold, counts),
            )
            commits[fold] = commit
        return commits

    def test_all_four_commits_verify_before_first_oof_label_read(self) -> None:
        commits = self._commit_all()
        calls: list[int] = []

        def load(image: int) -> np.ndarray:
            calls.append(image)
            return np.arange(576, dtype=np.int64)

        result = gate.evaluate_oof_after_all_commits(
            commits,
            permutation_loader=load,
            evaluator=lambda verified, permutations: (
                tuple(verified.commits),
                tuple(permutations),
            ),
        )
        self.assertEqual(result, ((0, 1, 2, 3), gate.CALIBRATION_IDS))
        self.assertEqual(tuple(calls), gate.CALIBRATION_IDS)

    def test_invalid_permutation_is_rejected_only_after_global_barrier(self) -> None:
        commits = self._commit_all()
        invalid = np.zeros(576, dtype=np.int64)
        evaluator = mock.Mock()
        calls: list[int] = []

        def load(image: int) -> np.ndarray:
            calls.append(image)
            return invalid

        with self.assertRaisesRegex(gate.E24EvaluatorContractError, "not a bijection"):
            gate.evaluate_oof_after_all_commits(
                commits,
                permutation_loader=load,
                evaluator=evaluator,
            )
        self.assertEqual(calls, [gate.CALIBRATION_IDS[0]])
        evaluator.assert_not_called()

    def test_missing_or_tampered_last_fold_blocks_every_label_read(self) -> None:
        commits = self._commit_all()
        loader = mock.Mock(return_value=np.arange(576, dtype=np.int64))
        missing = dict(commits)
        del missing[3]
        with self.assertRaisesRegex(gate.E24EvaluatorContractError, "exactly folds"):
            gate.evaluate_oof_after_all_commits(
                missing,
                permutation_loader=loader,
                evaluator=mock.Mock(),
            )
        loader.assert_not_called()

        payload = json.loads(commits[3].read_text(encoding="ascii"))
        model = Path(payload["model"]["path"])
        model.write_bytes(b"tampered-last-fold\n")
        with self.assertRaisesRegex(gate.E24EvaluatorContractError, "model .* mismatch"):
            gate.evaluate_oof_after_all_commits(
                commits,
                permutation_loader=loader,
                evaluator=mock.Mock(),
            )
        loader.assert_not_called()

    def test_stale_fold_provenance_blocks_global_label_boundary(self) -> None:
        commits = self._commit_all()
        expected = {fold: _run_provenance(fold) for fold in gate.OOF_FOLDS}
        expected[3] = _run_provenance(3, ledger_digit="9")
        loader = mock.Mock(return_value=np.arange(576, dtype=np.int64))
        with self.assertRaisesRegex(
            gate.E24EvaluatorContractError, "stale/different run provenance"
        ):
            gate.evaluate_oof_after_all_commits(
                commits,
                expected_run_provenance=expected,
                permutation_loader=loader,
                evaluator=mock.Mock(),
            )
        loader.assert_not_called()


class RunnerProcessBoundaryTests(unittest.TestCase):
    LEDGER = Path("E:/pazzle_work/posegraph_e24_selector/test_tmp/ledger.json")
    LEDGER_SHA = "a" * 64

    def test_cli_exposes_every_separate_process_mode_and_routes_predictor(self) -> None:
        common = ["--ledger", str(self.LEDGER), "--ledger-sha256", self.LEDGER_SHA]
        parser = runner.build_parser()
        self.assertEqual(parser.parse_args(["preflight"]).mode, "preflight")
        for mode, extra in (
            ("prepare-tile-bytes", ["--image", "17"]),
            ("prepare-inputs", ["--image", "17"]),
            ("feature-worker", ["--image", "10"]),
            ("prepare-fold-labels", ["--fold", "0"]),
            ("train-fold", ["--fold", "0"]),
            ("predict-fold", ["--fold", "0"]),
            ("structural-eval", []),
            ("orchestrate", []),
        ):
            with self.subTest(mode=mode):
                self.assertEqual(
                    parser.parse_args([mode, *extra, *common]).mode, mode
                )
        with mock.patch.object(runner, "predict_commit_fold") as predict:
            with contextlib.redirect_stdout(io.StringIO()):
                runner.main(
                    ["predict-fold", "--fold", "2", *common]
                )
        predict.assert_called_once_with(2, self.LEDGER, self.LEDGER_SHA)

    def test_every_target_entrypoint_authenticates_ledger_before_work(self) -> None:
        stop = runner.E24RunnerError("synthetic ledger stop")
        calls = (
            lambda: runner.prepare_upstream_tile_bytes(
                17, self.LEDGER, self.LEDGER_SHA
            ),
            lambda: runner.prepare_label_free_input(
                17, self.LEDGER, self.LEDGER_SHA
            ),
            lambda: runner.run_feature_worker(10, self.LEDGER, self.LEDGER_SHA),
            lambda: runner.prepare_fold_train_labels(
                0, self.LEDGER, self.LEDGER_SHA
            ),
            lambda: runner.train_fold_model(0, self.LEDGER, self.LEDGER_SHA),
            lambda: runner.predict_commit_fold(0, self.LEDGER, self.LEDGER_SHA),
            lambda: runner.run_structural_evaluation(
                self.LEDGER, self.LEDGER_SHA
            ),
            lambda: runner.orchestrate(self.LEDGER, self.LEDGER_SHA),
        )
        for index, call in enumerate(calls):
            with self.subTest(index=index), mock.patch.object(
                runner, "verify_preflight_ledger", side_effect=stop
            ) as verify, self.assertRaisesRegex(
                runner.E24RunnerError, "synthetic ledger stop"
            ):
                call()
            verify.assert_called_once_with(self.LEDGER, self.LEDGER_SHA)

    def test_source_enforces_feature_train_predict_and_label_barriers(self) -> None:
        module_prefix = inspect.getsource(runner).split("class E24RunnerError", 1)[0]
        self.assertNotIn("import eval_clean_score_oracle", module_prefix)
        self.assertNotIn("import eval_e14_cc192_discovery", module_prefix)
        self.assertNotIn("import eval_e23_i21_residual_candidate_ceiling", module_prefix)

        feature = inspect.getsource(runner.run_feature_worker)
        self.assertNotIn("_replay_detached_tiles", feature)
        self.assertNotIn(".permutation", feature)
        self.assertNotIn("build_label_only_relation_truth", feature)

        tile_lineage = inspect.getsource(runner.prepare_upstream_tile_bytes)
        self.assertIn("_replay_detached_tiles", tile_lineage)
        self.assertNotIn(".permutation", tile_lineage)
        self.assertNotIn("target_uint8", tile_lineage)
        replay = inspect.getsource(runner._replay_detached_tiles)
        self.assertIn("CanvasDataset", replay)
        self.assertNotIn("_e23_report_rows", replay)
        self.assertNotIn("DEFAULT_E23_REPORT", replay)
        self.assertNotIn("RawScene", replay)
        self.assertNotIn("candidate_ids", replay)
        self.assertNotIn("candidate_scores", replay)

        input_broker = inspect.getsource(runner.prepare_label_free_input)
        self.assertIn("load_original_raw_candidate_members", input_broker)
        self.assertIn("_load_upstream_tile_bytes", input_broker)
        self.assertNotIn("_replay_detached_tiles", input_broker)
        self.assertNotIn("RawScene", input_broker)
        self.assertNotIn(".permutation", input_broker)
        self.assertNotIn("target_uint8", input_broker)

        raw_reader = inspect.getsource(gate.load_original_raw_candidate_members)
        self.assertNotIn(".files", raw_reader)
        self.assertLess(
            raw_reader.index('"candidate_ids.npy"'),
            raw_reader.index('"candidate_scores.npy"'),
        )

        train = inspect.getsource(runner.train_fold_model)
        self.assertIn("fit_oof_fold", train)
        self.assertNotIn("predict_scores", train)
        self.assertNotIn("commit_fold_predictions", train)

        predict = inspect.getsource(runner.predict_commit_fold)
        self.assertIn("_reload_committed_predictor", predict)
        self.assertLess(
            predict.index("_reload_committed_predictor"),
            predict.index("predict_scores"),
        )
        self.assertNotIn("_load_fold_label(", predict)
        self.assertNotIn("fit_oof_fold", predict)
        self.assertNotIn(".permutation", predict)

        structural = inspect.getsource(runner.run_structural_evaluation)
        self.assertLess(
            structural.index("verify_all_oof_commits"),
            structural.index("_load_projected_permutation"),
        )
        self.assertNotIn(".permutation", structural)
        self.assertNotIn("candidate_pool_provenance_ok=True", structural)
        self.assertIn("candidate_pool_provenance_ok = bool", structural)

        broker = inspect.getsource(runner.prepare_fold_train_labels)
        self.assertIn("for image in boundary.train_ids", broker)
        self.assertIn("_load_projected_permutation", broker)
        self.assertNotIn("_replay_detached_tiles", broker)
        self.assertNotIn("target_uint8", broker)
        self.assertIn("_load_projected_permutation", structural)
        self.assertNotIn("_replay_detached_tiles", structural)

    def test_tile_lineage_handoff_is_resumable_exact_key_and_tamper_evident(self) -> None:
        with tempfile.TemporaryDirectory(dir=E_TEST_ROOT) as temporary:
            root = Path(temporary)
            input_root = root / "inputs"
            raw_path = root / "source_raw.npz"
            raw_path.write_bytes(b"authenticated-source")
            raw_sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()
            tiles = np.zeros((576, 20, 20, 3), dtype=np.uint8)
            tiles[17, 19, 19] = (1, 2, 3)
            tile_sha = runner._array_sha256(tiles)
            projection_sha = "c" * 64
            ledger = {
                "run_contract_sha256": "b" * 64,
                "upstream": {
                    "label_free_input_projection": {
                        "records_sha256": projection_sha
                    }
                },
            }
            source_record = {
                "validation_name": "synthetic_scene.png",
                "raw_cache": {
                    "path": str(raw_path.resolve()),
                    "bytes": raw_path.stat().st_size,
                    "file_sha256": raw_sha,
                },
                "tiles_uint8_sha256": tile_sha,
            }
            with mock.patch.object(
                runner, "INPUT_ROOT", input_root
            ), mock.patch.object(
                runner, "verify_preflight_ledger", return_value=ledger
            ), mock.patch.object(
                runner,
                "_validated_upstream_projection",
                return_value=source_record,
            ), mock.patch.object(
                runner,
                "_replay_detached_tiles",
                return_value=("synthetic_scene.png", tiles),
            ), mock.patch.object(
                runner, "enforce_aggregate_artifact_caps", return_value={}
            ):
                runner.prepare_upstream_tile_bytes(
                    17, self.LEDGER, self.LEDGER_SHA
                )
                tiles_path, lineage_path = runner._tile_lineage_paths(17)
                first_tiles = tiles_path.read_bytes()
                first_lineage = lineage_path.read_bytes()

                # A deterministic restart verifies the existing transaction.
                runner.prepare_upstream_tile_bytes(
                    17, self.LEDGER, self.LEDGER_SHA
                )
                self.assertEqual(tiles_path.read_bytes(), first_tiles)
                self.assertEqual(lineage_path.read_bytes(), first_lineage)

                payload, loaded = runner._load_upstream_tile_bytes(
                    17,
                    self.LEDGER_SHA,
                    ledger["run_contract_sha256"],
                    source_record,
                    projection_sha,
                )
                np.testing.assert_array_equal(loaded, tiles)
                self.assertEqual(payload["output_capability"], ["tiles_uint8"])
                self.assertFalse(payload["sealed_fields_exported"])
                forbidden = {
                    "permutation",
                    "target",
                    "clean",
                    "label",
                    "board",
                    "metric",
                }
                self.assertTrue(forbidden.isdisjoint(payload))

                forged = dict(payload)
                forged["target"] = "forbidden"
                lineage_path.write_bytes(runner._canonical_json_bytes(forged))
                with self.assertRaisesRegex(
                    runner.E24RunnerError, "capability identity"
                ):
                    runner._load_upstream_tile_bytes(
                        17,
                        self.LEDGER_SHA,
                        ledger["run_contract_sha256"],
                        source_record,
                        projection_sha,
                    )

                lineage_path.write_bytes(first_lineage)
                corrupted = bytearray(first_tiles)
                corrupted[-1] ^= 1
                tiles_path.write_bytes(corrupted)
                with self.assertRaisesRegex(
                    runner.E24RunnerError, "file provenance mismatch"
                ):
                    runner._load_upstream_tile_bytes(
                        17,
                        self.LEDGER_SHA,
                        ledger["run_contract_sha256"],
                        source_record,
                        projection_sha,
                    )

    def test_noncanary_input_is_blocked_before_upstream_scene_replay(self) -> None:
        ledger = {"run_contract_sha256": "b" * 64}
        stop = runner.E24RunnerError("canary stop")
        with mock.patch.object(
            runner, "verify_preflight_ledger", return_value=ledger
        ), mock.patch.object(
            runner, "enforce_aggregate_artifact_caps", return_value={}
        ), mock.patch.object(
            runner, "verify_feature_canary", side_effect=stop
        ), mock.patch.object(
            runner, "_validated_upstream_projection"
        ) as projection, self.assertRaisesRegex(runner.E24RunnerError, "canary stop"):
            runner.prepare_label_free_input(10, self.LEDGER, self.LEDGER_SHA)
        projection.assert_not_called()

        with mock.patch.object(
            runner, "verify_preflight_ledger", return_value=ledger
        ), mock.patch.object(
            runner, "enforce_aggregate_artifact_caps", return_value={}
        ), mock.patch.object(
            runner, "verify_feature_canary", side_effect=stop
        ), mock.patch.object(
            runner, "_replay_detached_tiles"
        ) as replay, self.assertRaisesRegex(runner.E24RunnerError, "canary stop"):
            runner.prepare_upstream_tile_bytes(10, self.LEDGER, self.LEDGER_SHA)
        replay.assert_not_called()

    def test_target_ledger_verification_does_not_reparse_e23_report(self) -> None:
        source = inspect.getsource(runner.verify_preflight_ledger)
        self.assertIn("_build_preflight_base_payload", source)
        self.assertIn("_validated_upstream_projection", source)
        self.assertNotIn("build_preflight_payload()", source)
        self.assertNotIn("_e23_report_rows", source)

    def test_preflight_base_payload_survives_canonical_json_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory(dir=E_TEST_ROOT) as temporary:
            report = Path(temporary) / "frozen_e23.json"
            report.write_bytes(b"synthetic-pinned-report")
            report_sha = hashlib.sha256(report.read_bytes()).hexdigest()
            runtime_paths = {
                "PYTHONPYCACHEPREFIX": str(E_TEST_ROOT / "pycache"),
                "TEMP": str(E_TEST_ROOT),
                "TMP": str(E_TEST_ROOT),
                "TMPDIR": str(E_TEST_ROOT),
            }
            with mock.patch.object(
                runner, "DEFAULT_E23_REPORT", report
            ), mock.patch.object(
                runner, "EXPECTED_E23_REPORT_SHA256", report_sha
            ), mock.patch.object(
                runner.evaluator,
                "validate_e24_runtime_paths",
                return_value=runtime_paths,
            ), mock.patch.object(
                runner.evaluator,
                "validate_lightgbm_runtime_version",
                return_value=runner.evaluator.EXPECTED_LIGHTGBM_VERSION,
            ), mock.patch.object(
                runner, "_source_hashes", return_value={"synthetic": "a" * 64}
            ):
                payload = runner._build_preflight_base_payload()
            reloaded = json.loads(
                runner._canonical_json_bytes(payload).decode("ascii")
            )
            self.assertEqual(payload, reloaded)
            self.assertIsInstance(
                payload["core_protocol"]["forbidden_features"], list
            )

    def test_forged_canary_checks_cannot_hide_oversized_receipt(self) -> None:
        with tempfile.TemporaryDirectory(dir=E_TEST_ROOT) as temporary:
            root = Path(temporary)
            input_root = root / "inputs"
            feature_root = root / "features"
            fold_root = root / "folds"
            gate_path = root / "canary" / "gate.json"
            structural = root / "report.json"
            input_dir = input_root / "image_0017"
            input_dir.mkdir(parents=True)
            (input_dir / "input.bin").write_bytes(b"i")
            feature_root.mkdir()
            feature_path = feature_root / "image_0017_features.npz"
            manifest_path = feature_root / "image_0017_features.json"
            receipt_path = feature_root / "image_0017_receipt.json"
            feature_path.write_bytes(b"f")
            manifest_path.write_bytes(b"m")
            receipt_path.write_bytes(b"r")
            oversized_receipt = {
                "hypotheses": runner.CANARY_EXPECTED_HYPOTHESES,
                "feature_file_bytes": 1,
                "resource": {
                    "wall_seconds": runner.CANARY_WALL_SECONDS_MAX + 1.0,
                    "process_cpu_seconds": 1.0,
                    "peak_rss_bytes": 1,
                },
            }
            forged_receipt = {
                **oversized_receipt,
                "resource": {
                    **oversized_receipt["resource"],
                    "wall_seconds": 1.0,
                },
            }
            observed, checks = runner._canary_observed_and_checks(
                forged_receipt,
                input_bytes=1,
                feature_bundle_bytes=3,
                aggregate_total_bytes_at_gate=100,
                first_target_state=True,
            )
            gate_payload = {
                "schema": "pazzle-e24-crs-v1-feature-canary-gate-v1",
                "status": "pass",
                "passed": True,
                "ledger_sha256": self.LEDGER_SHA,
                "run_contract_sha256": "b" * 64,
                "receipt_sha256": hashlib.sha256(b"r").hexdigest(),
                "thresholds": runner._canary_thresholds(),
                "observed": observed,
                "checks": checks,
                "labels_or_metrics_opened": False,
            }
            gate_path.parent.mkdir()
            gate_path.write_bytes(gate._canonical_json_bytes(gate_payload))
            with mock.patch.multiple(
                runner,
                INPUT_ROOT=input_root,
                FEATURE_ROOT=feature_root,
                FOLD_ROOT=fold_root,
                STRUCTURAL_REPORT=structural,
                CANARY_GATE_PATH=gate_path,
            ), mock.patch.object(
                runner,
                "verify_preflight_ledger",
                return_value={"run_contract_sha256": "b" * 64},
            ), mock.patch.object(
                runner, "_load_feature_receipt", return_value=oversized_receipt
            ), mock.patch.object(
                runner,
                "enforce_aggregate_artifact_caps",
                return_value={"feature_bytes": 3, "total_bytes": 104},
            ), self.assertRaisesRegex(
                runner.E24RunnerError, "canary did not pass"
            ):
                runner.verify_feature_canary(self.LEDGER, self.LEDGER_SHA)

    def test_orchestrator_uses_fresh_predictor_after_model_commit(self) -> None:
        source = inspect.getsource(runner.orchestrate)
        self.assertLess(
            source.index('invoke("prepare-tile-bytes", "--image", str(CANARY_IMAGE))'),
            source.index('invoke("prepare-inputs", "--image", str(CANARY_IMAGE))'),
        )
        self.assertLess(
            source.index('invoke("prepare-inputs", "--image", str(CANARY_IMAGE))'),
            source.index('invoke("feature-worker", "--image", str(CANARY_IMAGE))'),
        )
        self.assertLess(
            source.index('invoke("feature-worker", "--image", str(CANARY_IMAGE))'),
            source.index("for image in evaluator.CALIBRATION_IDS"),
        )
        self.assertLess(
            source.index('invoke("train-fold"'),
            source.index('invoke("predict-fold"'),
        )
        self.assertLess(
            source.index('invoke("predict-fold"'),
            source.index('invoke("structural-eval"'),
        )
        self.assertIn("subprocess.run", source)
        worker_output = json.dumps(
            {
                "resource": {
                    "wall_seconds": 1.0,
                    "process_cpu_seconds": 1.0,
                    "peak_rss_bytes": 1024,
                }
            }
        )
        completed = SimpleNamespace(returncode=0, stdout=worker_output, stderr="")
        with tempfile.TemporaryDirectory(dir=E_TEST_ROOT) as temporary, mock.patch.object(
            runner,
            "ORCHESTRATION_RECEIPT_PATH",
            Path(temporary) / "orchestration.json",
        ), mock.patch.object(
            runner,
            "verify_preflight_ledger",
            return_value={"run_contract_sha256": "b" * 64},
        ), mock.patch.object(
            runner.subprocess, "run", return_value=completed
        ) as launch, mock.patch.object(
            runner, "_sha256_file", return_value="c" * 64
        ), mock.patch.object(
            runner,
            "enforce_aggregate_artifact_caps",
            return_value={"feature_bytes": 0, "total_bytes": 0},
        ), mock.patch.object(runner.evaluator, "_atomic_write_create"):
            resource = runner.orchestrate(self.LEDGER, self.LEDGER_SHA)
        commands = [item.args[0] for item in launch.call_args_list]
        self.assertEqual(
            commands[0][2:5], ["prepare-tile-bytes", "--image", "17"]
        )
        self.assertEqual(
            commands[1][2:5], ["prepare-inputs", "--image", "17"]
        )
        self.assertEqual(
            commands[2][2:5], ["feature-worker", "--image", "17"]
        )
        self.assertEqual(
            [command[2] for command in commands].count("prepare-tile-bytes"), 8
        )
        self.assertEqual(
            [command[2] for command in commands].count("feature-worker"), 8
        )
        self.assertEqual(commands[-1][2], "structural-eval")
        self.assertEqual(resource["child_process_cpu_seconds"], float(len(commands)))

    def test_source_contract_excludes_mutable_board_and_includes_dependencies(self) -> None:
        names = {path.name for path in runner.SOURCE_FILES}
        self.assertNotIn("board.jsonl", names)
        self.assertTrue(
            {
                "e24_context_relation_selector.py",
                "eval_e24_context_relation_selector.py",
                "run_e24_context_relation_selector.py",
                "e23_i21_residual_candidate_oracle.py",
                "e22_rcce4_candidate_oracle.py",
                "e21_posegraph_candidate_oracle.py",
                "rank96_lab_selector.py",
                "solve_buddies.py",
                "eval_seeded_qap.py",
                "eval_clean_score_oracle.py",
                "eval_e14_cc192_discovery.py",
                "canvas_data.py",
                "distort.py",
                "imgio.py",
                "E24_CONTEXT_RELATION_SELECTOR.md",
                "PLAN.md",
                "BUDGET.md",
            }.issubset(names)
        )
        preflight_source = "\n".join(
            (
                inspect.getsource(runner._build_preflight_base_payload),
                inspect.getsource(runner.build_preflight_payload),
            )
        )
        self.assertNotIn("BOARD", preflight_source)
        self.assertIn("runtime_versions", preflight_source)
        self.assertIn("external append-only preflight event", preflight_source)

    def test_aggregate_feature_and_all_artifact_caps_are_independent(self) -> None:
        class FakeFile:
            def __init__(self, size: int) -> None:
                self.size = size

            def stat(self) -> SimpleNamespace:
                return SimpleNamespace(st_size=self.size)

        feature = FakeFile(gate.FEATURE_CACHE_BYTES_MAX)
        other = FakeFile(gate.ALL_ARTIFACT_BYTES_MAX)
        with mock.patch.object(
            runner, "_artifact_files", side_effect=({feature}, {feature})
        ), self.assertRaisesRegex(runner.E24RunnerError, "4 GiB"):
            runner.enforce_aggregate_artifact_caps(
                ledger_path=self.LEDGER, additional_feature_bytes=1
            )
        with mock.patch.object(
            runner, "_artifact_files", side_effect=(set(), {other})
        ), self.assertRaisesRegex(runner.E24RunnerError, "8 GiB"):
            runner.enforce_aggregate_artifact_caps(
                ledger_path=self.LEDGER, additional_total_bytes=1
            )
        accounting_source = inspect.getsource(runner.enforce_aggregate_artifact_caps)
        self.assertIn("STORAGE_ROOT", accounting_source)

    def test_malformed_input_and_feature_manifests_fail_before_array_load(self) -> None:
        with mock.patch.object(
            runner, "_load_canonical_json", return_value={"unexpected": True}
        ), mock.patch.object(
            runner.evaluator, "verify_sanitized_raw_artifact"
        ) as raw_loader, self.assertRaisesRegex(
            runner.E24RunnerError, "input bundle identity"
        ):
            runner._load_input_bundle(10, "a" * 64, "b" * 64, "c" * 64)
        raw_loader.assert_not_called()

        with mock.patch.object(
            runner, "_load_canonical_json", return_value={"unexpected": True}
        ), mock.patch.object(
            runner.selector, "load_feature_table_npz"
        ) as feature_loader, self.assertRaisesRegex(
            runner.E24RunnerError, "feature manifest identity"
        ):
            runner._load_feature_artifact(10, "a" * 64, "b" * 64)
        feature_loader.assert_not_called()

    def test_fold_run_provenance_binds_core_config_and_train_artifacts(self) -> None:
        fold = 0
        boundary = gate.fold_boundary(fold)
        ledger = {
            "run_contract_sha256": "b" * 64,
            "ordered_feature_names_sha256": "c" * 64,
            "lightgbm": {"version": gate.EXPECTED_LIGHTGBM_VERSION},
            "sources": {
                str((ROOT / "src/e24_context_relation_selector.py").resolve()): "d"
                * 64
            },
        }
        manifests = {
            image: {"feature_file": {"sha256": f"{3000 + image:064x}"}}
            for image in gate.CALIBRATION_IDS
        }
        labels = {
            image: f"{4000 + image:064x}" for image in boundary.train_ids
        }
        with mock.patch.object(runner, "_sha256_file", return_value="e" * 64):
            value = runner._fold_run_provenance(
                fold=fold,
                ledger=ledger,
                ledger_sha256="a" * 64,
                feature_manifests=manifests,
                label_manifest_sha256=labels,
            )
        self.assertEqual(value["ledger_sha256"], "a" * 64)
        self.assertEqual(value["core_source_sha256"], "d" * 64)
        self.assertEqual(value["canary_gate_sha256"], "e" * 64)
        self.assertEqual(set(map(int, value["train_feature_sha256"])), set(boundary.train_ids))
        self.assertEqual(set(map(int, value["train_label_manifest_sha256"])), set(boundary.train_ids))
        self.assertRegex(value["lightgbm_contract_sha256"], r"^[0-9a-f]{64}$")

    def test_committed_predictor_is_reloaded_and_contract_checked(self) -> None:
        fake = mock.Mock()
        fake.num_trees.return_value = 256
        fake.num_feature.return_value = len(selector.FEATURE_NAMES)
        model_path = Path(
            "E:/pazzle_work/posegraph_e24_selector/test_tmp/fold_model.txt"
        )
        manifest = {"model": {"sha256": "e" * 64}}
        with mock.patch.object(
            runner, "_load_model_manifest", return_value=(model_path, manifest)
        ), mock.patch("lightgbm.Booster", return_value=fake) as constructor:
            observed, observed_path, observed_manifest = (
                runner._reload_committed_predictor(0, _run_provenance(0))
            )
        self.assertIs(observed, fake)
        self.assertEqual(observed_path, model_path)
        self.assertIs(observed_manifest, manifest)
        constructor.assert_called_once_with(model_file=str(model_path))


class StructuralDecisionTests(unittest.TestCase):
    @staticmethod
    def _passing_row(image: int) -> gate.StructuralSceneCounts:
        fold = next(fold for fold, ids in gate.OOF_FOLDS.items() if image in ids)
        return gate.StructuralSceneCounts(
            image=image,
            fold=fold,
            provenance_ok=True,
            query_canonical_onehot=True,
            orientation_ok=True,
            fold_isolated=True,
            finite_output=True,
            dsu_legal=True,
            legal_origin=True,
            component_count=100,
            geometry_hypotheses=1_000,
            proposed_relations=100,
            true_proposed_relations=70,
            true_relations=100,
            accepted_relations=80,
            true_accepted_relations=65,
            exact_connected_tiles=288,
            accepted_graph_vertices=50,
            accepted_graph_components=1,
        )

    def test_exact_structural_summary_and_inclusive_pass(self) -> None:
        rows = [self._passing_row(image) for image in gate.CALIBRATION_IDS]
        summary = gate.summarize_structural(rows)
        decision = gate.structural_decision(summary)
        self.assertEqual(summary["completed_scenes"], 8)
        self.assertEqual(summary["mean_proposed_precision"], 0.70)
        self.assertEqual(summary["worst_proposed_precision"], 0.70)
        self.assertEqual(summary["mean_true_relation_recall"], 0.70)
        self.assertEqual(summary["mean_exact_connected_coverage"], 0.50)
        self.assertTrue(decision["passed"])
        self.assertEqual(decision["stage"], "go_staged_end_to_end")
        self.assertTrue(all(decision["checks"].values()))

    def test_true_relation_denominator_uses_physical_seams_not_all_pure_pairs(self) -> None:
        owner = np.arange(576, dtype=np.int64)
        permutation = np.arange(576, dtype=np.int64)
        shifts = {
            tile: (tile // 24, tile % 24)
            for tile in range(576)
        }
        seam_relations = gate._ground_truth_seam_relation_set(
            owner, permutation, shifts
        )
        adjacent = (0, 1, 0, 1)
        distant = (0, 575, 23, 23)
        self.assertIn(adjacent, seam_relations)
        self.assertNotIn(distant, seam_relations)
        # A selector may have arbitrarily many pure/distant component-pair
        # queries, but only relations induced by one of the 1104 upright GT
        # seams enter precision/recall truth.
        query_truth = {adjacent, distant, (1, 574, 23, 21)}
        self.assertEqual(query_truth.intersection(seam_relations), {adjacent})
        self.assertEqual(len(seam_relations), 24 * 23 * 2)

    def test_integrity_failure_cannot_be_rescued_by_high_diagnostics(self) -> None:
        rows = [self._passing_row(image) for image in gate.CALIBRATION_IDS]
        rows[0] = replace(rows[0], fold_isolated=False)
        summary = gate.summarize_structural(rows)
        decision = gate.structural_decision(summary)
        self.assertFalse(decision["passed"])
        self.assertEqual(decision["stage"], "kill_crs_v1")
        self.assertFalse(decision["checks"]["complete_integrity_legal_scenes"])

    def test_zero_denominator_wrong_fold_cap_and_duplicate_scene_fail_closed(self) -> None:
        passing = self._passing_row(10)
        for changes, message in (
            ({"proposed_relations": 0}, "proposed relations"),
            ({"true_relations": 0}, "true relations"),
            ({"accepted_relations": 0}, "accepted relations"),
            ({"fold": 1}, "wrong OOF fold"),
            ({"geometry_hypotheses": 450_001}, "cap"),
        ):
            values = {
                field: getattr(passing, field)
                for field in passing.__dataclass_fields__
            }
            values.update(changes)
            with self.subTest(changes=changes), self.assertRaisesRegex(
                gate.E24EvaluatorContractError, message
            ):
                gate.structural_scene_metrics(gate.StructuralSceneCounts(**values))
        rows = [self._passing_row(image) for image in gate.CALIBRATION_IDS]
        rows[-1] = self._passing_row(16)
        with self.assertRaisesRegex(gate.E24EvaluatorContractError, "each E24 scene"):
            gate.summarize_structural(rows)


if __name__ == "__main__":
    unittest.main()

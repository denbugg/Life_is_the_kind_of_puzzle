from __future__ import annotations

import copy
import inspect
import io
import json
import os
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


E_TEST_ROOT = Path("E:/pazzle_work/posegraph_e24_selector/test_final_all8")
os.environ["PYTHONPYCACHEPREFIX"] = str(E_TEST_ROOT / "pycache")
os.environ["TEMP"] = str(E_TEST_ROOT / "tmp")
os.environ["TMP"] = str(E_TEST_ROOT / "tmp")
os.environ["TMPDIR"] = str(E_TEST_ROOT / "tmp")
sys.pycache_prefix = str(E_TEST_ROOT / "pycache")

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import e24_context_relation_selector as selector
import eval_e24_context_relation_selector as e24_eval
import eval_e24_staged_ssim_nlm as staged
import run_e24_final_all8 as runner


def sha(value: int | str) -> str:
    if isinstance(value, int):
        return f"{value:064x}"
    return value * 64


def feature_table() -> selector.RelationFeatureTable:
    rows = 4
    features = np.zeros((rows, len(selector.FEATURE_NAMES)), dtype=np.float32)
    hypothesis_ids = np.asarray((0, -1, 1, -1), dtype=np.int64)
    relation_ids = np.asarray((0, -1, 1, -1), dtype=np.int64)
    relations = np.asarray(
        ((0, 1, 0, 1), (0, 1, 0, 0), (2, 3, 1, 0), (2, 3, 0, 0)),
        dtype=np.int64,
    )
    row_kind = np.asarray(
        (selector.ROW_OFFSET, selector.ROW_NONE, selector.ROW_OFFSET, selector.ROW_NONE),
        dtype=np.uint8,
    )
    support = np.asarray((1, 0, 1, 0), dtype=np.int64)
    return selector.RelationFeatureTable(
        features=features,
        hypothesis_ids=hypothesis_ids,
        relation_ids=relation_ids,
        relations=relations,
        row_kind=row_kind,
        support=support,
        query_offsets=np.asarray((0, 2, 4), dtype=np.int64),
        scene_offsets=np.asarray((0, 4), dtype=np.int64),
    )


def relevance() -> np.ndarray:
    return np.asarray((1, 0, 0, 1), dtype=np.int8)


def staged_rows() -> list[dict]:
    rows: list[dict] = []
    for image in e24_eval.CALIBRATION_IDS:
        fold = next(fold for fold, heldout in e24_eval.OOF_FOLDS.items() if image in heldout)
        rr96 = {
            "objective": 10.0,
            "placement": 0.5,
            "neighbour": 0.50,
            "right": 0.50,
            "down": 0.50,
            "solve_only_ssim": staged.PINNED_RR96_MEAN_SOLVE_SSIM,
            "final_ssim": staged.PINNED_RR96_MEAN_FINAL_SSIM,
            "board_sha256": sha(1000 + image),
            "solved_corrupted_canvas_sha256": sha(2000 + image),
            "restored_canvas_sha256": sha(3000 + image),
        }
        candidate = {
            "objective": 9.0,
            "placement": 0.6,
            "neighbour": 0.52,
            "right": 0.52,
            "down": 0.52,
            "solve_only_ssim": staged.PINNED_RR96_MEAN_SOLVE_SSIM + 0.01,
            "final_ssim": staged.PINNED_RR96_MEAN_FINAL_SSIM + 0.01,
            "board_sha256": sha(4000 + image),
            "solved_corrupted_canvas_sha256": sha(5000 + image),
            "restored_canvas_sha256": sha(6000 + image),
        }
        rows.append(
            {
                "image": image,
                "fold": fold,
                "validation_name": f"img_{image:06d}.png",
                "orientation_degrees": 0,
                "reflection": False,
                "provenance": {"board_commit_sha256": sha(7000 + image)},
                "permutation_sha256": sha(8000 + image),
                "target_sha256": sha(9000 + image),
                "rr96": rr96,
                "candidate": candidate,
                "delta": {
                    "solve_only_ssim": candidate["solve_only_ssim"]
                    - rr96["solve_only_ssim"],
                    "final_ssim": candidate["final_ssim"] - rr96["final_ssim"],
                    "neighbour": candidate["neighbour"] - rr96["neighbour"],
                },
            }
        )
    return rows


def staged_payload() -> tuple[dict, dict[int, str]]:
    rows = staged_rows()
    summary = staged.summarize_staged(rows)
    decision = staged.staged_decision(summary)
    board_hashes = {image: sha(7000 + image) for image in e24_eval.CALIBRATION_IDS}
    payload = {
        "schema": runner.STAGED_REPORT_SCHEMA,
        "schema_version": 1,
        "status": "complete",
        "stage": "go_final_all8_fit",
        "staged_protocol_sha256": staged.PROTOCOL_SHA256,
        "metric_broker_contract_sha256": staged.METRIC_BROKER_CONTRACT_SHA256,
        "ledger_sha256": sha("1"),
        "run_contract_sha256": sha("2"),
        "premetric_seal_sha256": sha("3"),
        "structural_report_sha256": sha("4"),
        "orchestration_receipt_sha256": sha("5"),
        "board_barrier_sha256": sha("6"),
        "board_commit_sha256": {
            str(image): board_hashes[image] for image in e24_eval.CALIBRATION_IDS
        },
        "rr96_verification": staged.rr96_verification_for_rows(rows),
        "rows": rows,
        "summary": summary,
        "decision": decision,
        "e25_opened": False,
    }
    return payload, board_hashes


def manifest_fixture() -> tuple[dict, dict, dict, dict, dict, dict]:
    authority = {"ledger": {"path": "E:/x", "sha256": sha("1")}}
    sources = {str(ROOT / "source.py"): sha("2")}
    features = {
        str(image): {"feature_sha256": sha(image), "rows": 4, "queries": 2}
        for image in e24_eval.CALIBRATION_IDS
    }
    labels = {str(image): {"label_sha256": sha(100 + image)} for image in e24_eval.CALIBRATION_IDS}
    model = {
        "path": str(runner.MODEL_PATH.resolve()),
        "bytes": 123,
        "sha256": sha("3"),
        "num_trees": 256,
        "current_iteration": 256,
        "num_features": 227,
        "num_model_per_iteration": 1,
        "objective": "lambdarank",
        "canonical_reload_equal": True,
    }
    offsets = {
        str(image): [4 * index, 4 * (index + 1)]
        for index, image in enumerate(e24_eval.CALIBRATION_IDS)
    }
    training = {
        "scene_ids": list(e24_eval.CALIBRATION_IDS),
        "feature_count": 227,
        "ordered_feature_names_sha256": runner._feature_names_sha256(),
        "learner_contract": runner.FINAL_LEARNER_CONTRACT,
        "learner_contract_sha256": runner.FINAL_LEARNER_CONTRACT_SHA256,
        "lightgbm_version": e24_eval.EXPECTED_LIGHTGBM_VERSION,
        "rows": 32,
        "queries": 16,
        "scene_row_offsets": offsets,
        "feature_provenance": features,
        "label_provenance": labels,
        "relevance_sha256": sha("4"),
        "row_weights_sha256": sha("5"),
    }
    resource = {
        "fit_cpu_seconds": 100.0,
        "cpu_seconds_max": e24_eval.FINAL_FIT_CPU_SECONDS_MAX,
        "fit_wall_seconds": 20.0,
        "peak_rss_bytes": 1024,
        "peak_rss_bytes_max": e24_eval.PEAK_RAM_BYTES_MAX,
    }
    checks = {
        "exact_8_scenes": True,
        "exact_227_features": True,
        "exact_256_trees": True,
        "seed_1234": True,
        "no_validation_or_early_stopping": True,
        "fit_cpu_at_most_2h": True,
        "peak_rss_at_most_16gib": True,
        "aggregate_artifacts_at_most_8gib": True,
        "reloaded_model_canonical": True,
    }
    payload = {
        "schema": runner.FINAL_MANIFEST_SCHEMA,
        "schema_version": 1,
        "status": "complete_pass_only_final_all8",
        "authority": authority,
        "sources_sha256": sources,
        "training": training,
        "resource": resource,
        "model": model,
        "checks": checks,
        "e25_opened": False,
    }
    return payload, authority, sources, features, labels, model


class FinalAll8SyntheticTests(unittest.TestCase):
    def test_prefit_source_seal_rejects_tamper_before_label_or_fit(self) -> None:
        own = {str(path.resolve()): sha(index + 1) for index, path in enumerate(runner.FINAL_OWN_SOURCE_FILES)}
        full = dict(own)
        with (
            mock.patch.object(runner, "_sha", side_effect=lambda path: own[str(path.resolve())]),
            mock.patch.object(runner, "_source_hashes", return_value=full),
        ):
            self.assertEqual(
                runner.authenticate_prefit_source_snapshot({"sources": own}), full
            )
            tampered = {"sources": dict(own)}
            tampered["sources"][str(runner.FINAL_OWN_SOURCE_FILES[0])] = sha("f")
            with self.assertRaisesRegex(runner.E24FinalFitError, "changed after staged"):
                runner.authenticate_prefit_source_snapshot(tampered)

        order: list[str] = []
        authority = runner.FinalFitAuthority(
            upstream=SimpleNamespace(
                ledger_path=Path("E:/pazzle_work/posegraph_e24_selector/preflight/x.json")
            ),
            staged_report_path=runner.STAGED_REPORT_PATH,
            staged_report_sha256=sha("1"),
            staged_report={},
            premetric_seal_sha256=sha("2"),
            board_barrier_sha256=sha("3"),
            board_commit_sha256={},
            metric_broker_contract_sha256=sha("4"),
            prefit_sources_sha256={"sealed": sha("5")},
        )
        with (
            mock.patch.object(runner, "_source_hashes", side_effect=lambda: order.append("sources") or {"drift": sha("6")}),
            mock.patch.object(runner, "load_consensus_all8_labels", side_effect=lambda *_: order.append("labels")) as labels,
            mock.patch.object(runner, "_fit_model", side_effect=lambda *_: order.append("fit")) as fit,
        ):
            with self.assertRaisesRegex(runner.E24FinalFitError, "before label access/fit"):
                runner.run_final_fit(authority)
        self.assertEqual(order, ["sources"])
        labels.assert_not_called()
        fit.assert_not_called()

    def test_upstream_authority_failure_precedes_report_or_label_access(self) -> None:
        with (
            mock.patch.object(
                runner.staged_runner,
                "authenticate_authority",
                side_effect=RuntimeError("bad structural receipt"),
            ) as upstream,
            mock.patch.object(runner, "_load_json") as report,
            mock.patch.object(runner, "load_consensus_all8_labels") as labels,
        ):
            with self.assertRaisesRegex(runner.E24FinalFitError, "structural/orchestration"):
                runner.authenticate_final_fit_authority()
        upstream.assert_called_once()
        report.assert_not_called()
        labels.assert_not_called()
        source = inspect.getsource(runner.authenticate_final_fit_authority)
        for required in (
            "expected_ledger_sha256",
            "expected_run_contract_sha256",
            "expected_premetric_seal_sha256",
            "expected_structural_report_sha256",
            "expected_orchestration_receipt_sha256",
            "expected_board_barrier_sha256",
            "expected_board_commit_sha256",
        ):
            self.assertIn(required, source)

    def test_learner_contract_is_literal_final_crs_v1(self) -> None:
        contract = runner.final_learner_contract()
        self.assertEqual(len(selector.FEATURE_NAMES), 227)
        self.assertEqual(contract["feature_count"], 227)
        self.assertEqual(contract["trees"], 256)
        self.assertEqual(contract["seed"], 1234)
        self.assertEqual(contract["config"]["n_estimators"], 256)
        self.assertEqual(contract["config"]["random_state"], 1234)
        self.assertEqual(contract["config"]["data_random_seed"], 1234)
        self.assertEqual(contract["config"]["feature_fraction_seed"], 1234)
        self.assertFalse(contract["validation"])
        self.assertFalse(contract["early_stopping"])
        self.assertNotIn("callbacks", contract["config"])

    def test_staged_pass_is_recomputed_and_hash_bound(self) -> None:
        payload, board_hashes = staged_payload()
        observed = runner.validate_staged_pass_payload(
            payload,
            ledger_sha256=sha("1"),
            run_contract_sha256=sha("2"),
            structural_report_sha256=sha("4"),
            orchestration_receipt_sha256=sha("5"),
            premetric_seal_sha256=sha("3"),
            board_barrier_sha256=sha("6"),
            board_commit_sha256=board_hashes,
        )
        self.assertTrue(observed["decision"]["passed"])
        forged = copy.deepcopy(payload)
        forged["summary"]["mean_final_ssim_delta"] = 1.0
        with self.assertRaises(runner.E24FinalFitError):
            runner.validate_staged_pass_payload(
                forged,
                ledger_sha256=sha("1"),
                run_contract_sha256=sha("2"),
                structural_report_sha256=sha("4"),
                orchestration_receipt_sha256=sha("5"),
                premetric_seal_sha256=sha("3"),
                board_barrier_sha256=sha("6"),
                board_commit_sha256=board_hashes,
            )
        wrong = dict(board_hashes)
        wrong[10] = sha("f")
        with self.assertRaises(runner.E24FinalFitError):
            runner.validate_staged_pass_payload(
                payload,
                ledger_sha256=sha("1"),
                run_contract_sha256=sha("2"),
                structural_report_sha256=sha("4"),
                orchestration_receipt_sha256=sha("5"),
                premetric_seal_sha256=sha("3"),
                board_barrier_sha256=sha("6"),
                board_commit_sha256=wrong,
            )

    def test_failed_staged_decision_cannot_open_final_fit(self) -> None:
        payload, board_hashes = staged_payload()
        failed = copy.deepcopy(payload)
        for row in failed["rows"]:
            row["candidate"]["final_ssim"] = row["rr96"]["final_ssim"] - 0.1
            row["delta"]["final_ssim"] = (
                row["candidate"]["final_ssim"] - row["rr96"]["final_ssim"]
            )
        failed["summary"] = staged.summarize_staged(failed["rows"])
        failed["decision"] = staged.staged_decision(failed["summary"])
        failed["stage"] = failed["decision"]["stage"]
        self.assertFalse(failed["decision"]["passed"])
        with self.assertRaises(runner.E24FinalFitError):
            runner.validate_staged_pass_payload(
                failed,
                ledger_sha256=sha("1"),
                run_contract_sha256=sha("2"),
                structural_report_sha256=sha("4"),
                orchestration_receipt_sha256=sha("5"),
                premetric_seal_sha256=sha("3"),
                board_barrier_sha256=sha("6"),
                board_commit_sha256=board_hashes,
            )

    def test_final_batch_uses_all8_and_exact_frozen_weights(self) -> None:
        tables = {image: feature_table() for image in e24_eval.CALIBRATION_IDS}
        labels = {image: relevance() for image in e24_eval.CALIBRATION_IDS}
        batch = runner.build_final_training_batch(
            tables_by_scene=tables, relevance_by_scene=labels
        )
        self.assertEqual(batch.table.features.shape, (32, 227))
        self.assertEqual(batch.table.queries, 16)
        self.assertEqual(batch.scene_row_offsets[10], (0, 4))
        self.assertEqual(batch.scene_row_offsets[17], (28, 32))
        expected = selector.balanced_query_row_weights(batch.table, batch.relevance)
        self.assertTrue(np.array_equal(batch.row_weights, expected))
        masses = [
            float(batch.row_weights[start:stop].sum(dtype=np.float64))
            for start, stop in batch.scene_row_offsets.values()
        ]
        self.assertTrue(all(value == masses[0] for value in masses))
        missing = dict(labels)
        missing.pop(17)
        with self.assertRaises(runner.E24FinalFitError):
            runner.build_final_training_batch(
                tables_by_scene=tables, relevance_by_scene=missing
            )

    def test_fit_calls_frozen_core_as_fold0_with_no_validation_surface(self) -> None:
        batch = runner.build_final_training_batch(
            tables_by_scene={image: feature_table() for image in e24_eval.CALIBRATION_IDS},
            relevance_by_scene={image: relevance() for image in e24_eval.CALIBRATION_IDS},
        )
        sentinel = object()
        with (
            mock.patch.object(runner.e24_eval, "validate_lightgbm_runtime_version"),
            mock.patch.object(runner.selector, "fit_lambdarank", return_value=sentinel) as fit,
        ):
            self.assertIs(runner._fit_model(batch), sentinel)
        fit.assert_called_once()
        args, kwargs = fit.call_args
        self.assertIs(args[0], batch.table)
        self.assertIs(args[1], batch.relevance)
        self.assertEqual(kwargs["fold"], 0)
        self.assertTrue(np.array_equal(kwargs["row_weights"], batch.row_weights))
        self.assertEqual(set(kwargs), {"fold", "row_weights"})

    def test_reloaded_model_requires_exact_tree_feature_and_canonical_bytes(self) -> None:
        raw = "synthetic model\n".encode("utf-8")

        class FakeBooster:
            trees = 256

            def __init__(self, *, model_file: str) -> None:
                self.model_file = model_file

            def num_trees(self) -> int:
                return self.trees

            def current_iteration(self) -> int:
                return 256

            def num_feature(self) -> int:
                return 227

            def num_model_per_iteration(self) -> int:
                return 1

            def model_to_string(self, *, num_iteration: int) -> str:
                self.assert_iteration = num_iteration
                return raw.decode("utf-8")

            def dump_model(self) -> dict:
                return {
                    "max_feature_idx": 226,
                    "num_tree_per_iteration": 1,
                    "objective": "lambdarank sigmoid:1",
                }

        fake_module = SimpleNamespace(Booster=FakeBooster)
        with (
            mock.patch.dict(sys.modules, {"lightgbm": fake_module}),
            mock.patch.object(runner.e24_eval, "validate_lightgbm_runtime_version"),
            mock.patch.object(runner, "_sha", return_value=sha("a")),
            mock.patch.object(Path, "is_file", return_value=True),
            mock.patch.object(Path, "stat", return_value=SimpleNamespace(st_size=len(raw))),
            mock.patch.object(Path, "read_bytes", return_value=raw),
        ):
            predictor, record = runner._reload_model(
                runner.MODEL_PATH,
                expected_sha256=sha("a"),
                expected_bytes=len(raw),
            )
            self.assertEqual(predictor.num_trees(), 256)
            self.assertTrue(record["canonical_reload_equal"])
            FakeBooster.trees = 255
            with self.assertRaisesRegex(runner.E24FinalFitError, "contract drifted"):
                runner._reload_model(runner.MODEL_PATH)
        FakeBooster.trees = 256

    def test_three_label_copies_must_be_identical(self) -> None:
        feature_manifests = {
            image: {
                "rows": 4,
                "queries": 2,
                "feature_file": {"sha256": sha(100 + image)},
            }
            for image in e24_eval.CALIBRATION_IDS
        }
        authority = runner.FinalFitAuthority(
            upstream=SimpleNamespace(
                ledger_sha256=sha("1"),
                ledger={"run_contract_sha256": sha("2")},
                feature_manifests=feature_manifests,
            ),
            staged_report_path=runner.STAGED_REPORT_PATH,
            staged_report_sha256=sha("3"),
            staged_report={},
            premetric_seal_sha256=sha("4"),
            board_barrier_sha256=sha("5"),
            board_commit_sha256={},
            metric_broker_contract_sha256=sha("6"),
            prefit_sources_sha256={},
        )

        def verify(fold: int, image: int, **_kwargs: object) -> tuple[dict, str]:
            path = Path(f"E:/pazzle_work/posegraph_e24_selector/test/f{fold}_i{image}.npy")
            return (
                {"label_file": {"path": str(path), "sha256": sha(1000 + image)}},
                sha(2000 + fold * 100 + image),
            )

        def load(path: Path, **_kwargs: object) -> np.ndarray:
            value = relevance().copy()
            if "f2_i10" in str(path):
                value[:] = (0, 1, 0, 1)
            return value

        with (
            mock.patch.object(runner.e24_runner, "_verify_fold_label_manifest", side_effect=verify),
            mock.patch.object(runner.e24_runner, "_load_exact_npy", side_effect=load),
            mock.patch.object(
                runner.e24_runner,
                "_label_paths",
                side_effect=lambda fold, image: (
                    Path(f"E:/x/f{fold}_i{image}.npy"),
                    Path(f"E:/x/f{fold}_i{image}.json"),
                ),
            ),
        ):
            with self.assertRaisesRegex(runner.E24FinalFitError, "three byte-equivalent"):
                runner.load_consensus_all8_labels(authority)

    def test_manifest_recomputes_resource_and_contract_checks(self) -> None:
        payload, authority, sources, features, labels, model = manifest_fixture()
        observed = runner.validate_final_manifest_payload(
            payload,
            expected_authority=authority,
            expected_sources=sources,
            expected_feature_provenance=features,
            expected_label_provenance=labels,
            expected_model=model,
        )
        self.assertEqual(observed, payload)
        over = copy.deepcopy(payload)
        over["resource"]["fit_cpu_seconds"] = e24_eval.FINAL_FIT_CPU_SECONDS_MAX + 1.0
        with self.assertRaises(runner.E24FinalFitError):
            runner.validate_final_manifest_payload(
                over,
                expected_authority=authority,
                expected_sources=sources,
                expected_feature_provenance=features,
                expected_label_provenance=labels,
                expected_model=model,
            )
        forged = copy.deepcopy(payload)
        forged["training"]["learner_contract"]["seed"] = 999
        with self.assertRaises(runner.E24FinalFitError):
            runner.validate_final_manifest_payload(
                forged,
                expected_authority=authority,
                expected_sources=sources,
                expected_feature_provenance=features,
                expected_label_provenance=labels,
                expected_model=model,
            )

    def test_cpu_cap_fails_before_model_or_manifest_write(self) -> None:
        authority = runner.FinalFitAuthority(
            upstream=SimpleNamespace(
                ledger_path=Path("E:/pazzle_work/posegraph_e24_selector/preflight/x.json")
            ),
            staged_report_path=runner.STAGED_REPORT_PATH,
            staged_report_sha256=sha("1"),
            staged_report={},
            premetric_seal_sha256=sha("2"),
            board_barrier_sha256=sha("3"),
            board_commit_sha256={},
            metric_broker_contract_sha256=sha("4"),
            prefit_sources_sha256={},
        )
        fake_batch = SimpleNamespace()
        fake_manifest = Path("E:/pazzle_work/posegraph_e24_selector/test/no_manifest.json")
        fake_model = Path("E:/pazzle_work/posegraph_e24_selector/test/no_model.txt")
        order: list[str] = []
        with (
            mock.patch.object(runner, "MANIFEST_PATH", fake_manifest),
            mock.patch.object(runner, "MODEL_PATH", fake_model),
            mock.patch.object(
                runner,
                "_source_hashes",
                side_effect=lambda: order.append("sources") or {},
            ),
            mock.patch.object(runner.e24_eval, "validate_e24_runtime_paths"),
            mock.patch.object(runner.e24_eval, "validate_lightgbm_runtime_version"),
            mock.patch.object(runner.e24_runner, "enforce_aggregate_artifact_caps"),
            mock.patch.object(
                runner,
                "load_consensus_all8_labels",
                side_effect=lambda *_: order.append("labels") or ({}, {}),
            ),
            mock.patch.object(runner, "_feature_provenance", return_value={}),
            mock.patch.object(runner, "_load_feature_tables", return_value={}),
            mock.patch.object(
                runner,
                "build_final_training_batch",
                side_effect=lambda **_: order.append("batch") or fake_batch,
            ),
            mock.patch.object(
                runner,
                "_fit_model",
                side_effect=lambda *_: order.append("fit") or object(),
            ),
            mock.patch.object(runner.e24_runner, "_serialize_model_bytes", return_value=b"model"),
            mock.patch.object(runner.time, "process_time", side_effect=(0.0, 7200.1)),
            mock.patch.object(runner.time, "monotonic", side_effect=(0.0, 1.0)),
            mock.patch.object(runner, "_peak_rss_bytes", return_value=1024),
            mock.patch.object(runner.e24_eval, "_atomic_write_create_or_verify") as write,
        ):
            with self.assertRaisesRegex(runner.E24FinalFitError, "2 CPU-hour"):
                runner.run_final_fit(authority)
        self.assertEqual(
            order,
            ["sources", "labels", "batch", "sources", "fit", "sources"],
        )
        write.assert_not_called()

    def test_smoke_is_data_free_and_all_outputs_are_on_e(self) -> None:
        output = io.StringIO()
        with (
            redirect_stdout(output),
            mock.patch.object(runner, "authenticate_final_fit_authority") as authority,
            mock.patch.object(runner, "load_consensus_all8_labels") as labels,
        ):
            runner.main(["smoke"])
        authority.assert_not_called()
        labels.assert_not_called()
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["status"], "data_free")
        self.assertEqual(payload["seed"], 1234)
        self.assertFalse(payload["e25_opened"])
        for path in (
            runner.STORAGE_ROOT,
            runner.FINAL_ROOT,
            runner.MODEL_PATH,
            runner.MANIFEST_PATH,
            runner.STAGED_REPORT_PATH,
        ):
            resolved = path.resolve(strict=False)
            self.assertEqual(resolved.drive.upper(), "E:")
            self.assertTrue(str(resolved).startswith(str(runner.STORAGE_ROOT.resolve(strict=False))))


if __name__ == "__main__":
    unittest.main()

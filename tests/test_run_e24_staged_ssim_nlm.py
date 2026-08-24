from __future__ import annotations

import copy
import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import eval_e24_context_relation_selector as e24_eval
import eval_e24_staged_ssim_nlm as staged
import run_e24_context_relation_selector as e24_runner
import run_e24_staged_ssim_nlm as runner


def sha(char: str) -> str:
    return char * 64


def structural_payload() -> tuple[dict, dict[int, str]]:
    rows = []
    for image in e24_eval.CALIBRATION_IDS:
        fold = next(fold for fold, ids in e24_eval.OOF_FOLDS.items() if image in ids)
        rows.append(
            e24_eval.StructuralSceneCounts(
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
                geometry_hypotheses=1000,
                proposed_relations=10,
                true_proposed_relations=8,
                true_relations=10,
                accepted_relations=10,
                true_accepted_relations=8,
                exact_connected_tiles=400,
                accepted_graph_vertices=10,
                accepted_graph_components=1,
            )
        )
    summary = e24_eval.summarize_structural(rows)
    decision = e24_eval.structural_decision(summary)
    fold_hashes = {fold: sha(str(fold + 1)) for fold in e24_eval.OOF_FOLDS}
    payload = {
        "schema": e24_runner.STRUCTURAL_REPORT_SCHEMA,
        "status": "complete",
        "stage": "go_staged_end_to_end",
        "ledger_sha256": sha("a"),
        "run_contract_sha256": sha("b"),
        "fold_commit_sha256": {
            str(fold): fold_hashes[fold] for fold in e24_eval.OOF_FOLDS
        },
        "rows": [row.__dict__ for row in rows],
        "summary": summary,
        "decision": decision,
        "staged_board_ssim_nlm": "sealed_not_run",
        "e25_opened": False,
    }
    return payload, fold_hashes


def receipt_payload() -> dict:
    return {
        "schema": e24_runner.ORCHESTRATION_RECEIPT_SCHEMA,
        "status": "pass",
        "ledger_sha256": sha("a"),
        "run_contract_sha256": sha("b"),
        "canary_gate_sha256": sha("c"),
        "structural_report_sha256": sha("d"),
        "resource": {
            "child_process_cpu_seconds": 100.0,
            "maximum_child_peak_rss_bytes": 1024,
            "cpu_seconds_max": e24_eval.OOF_CPU_SECONDS_MAX,
            "peak_rss_bytes_max": e24_eval.PEAK_RAM_BYTES_MAX,
        },
        "checks": {
            "oof_cpu_at_most_8h": True,
            "peak_rss_at_most_16gib": True,
            "aggregate_artifacts_at_most_8gib": True,
        },
    }


class E24StagedRunnerSyntheticTests(unittest.TestCase):
    def test_structural_pass_is_recomputed_not_trusted(self) -> None:
        payload, fold_hashes = structural_payload()
        observed = runner.validate_structural_pass_payload(
            payload,
            ledger_sha256=sha("a"),
            run_contract_sha256=sha("b"),
            fold_commit_sha256=fold_hashes,
        )
        self.assertTrue(observed["decision"]["passed"])
        forged = copy.deepcopy(payload)
        forged["summary"]["mean_proposed_precision"] = 1.0
        forged["decision"]["passed"] = True
        with self.assertRaises(runner.E24StagedRunnerError):
            runner.validate_structural_pass_payload(
                forged,
                ledger_sha256=sha("a"),
                run_contract_sha256=sha("b"),
                fold_commit_sha256=fold_hashes,
            )

    def test_structural_fold_commit_hash_mismatch_fails(self) -> None:
        payload, fold_hashes = structural_payload()
        drifted = dict(fold_hashes)
        drifted[0] = sha("e")
        with self.assertRaises(runner.E24StagedRunnerError):
            runner.validate_structural_pass_payload(
                payload,
                ledger_sha256=sha("a"),
                run_contract_sha256=sha("b"),
                fold_commit_sha256=drifted,
            )

    def test_receipt_resource_values_and_hash_bindings_are_recomputed(self) -> None:
        payload = receipt_payload()
        observed = runner.validate_orchestration_receipt_payload(
            payload,
            ledger_sha256=sha("a"),
            run_contract_sha256=sha("b"),
            canary_gate_sha256=sha("c"),
            structural_report_sha256=sha("d"),
        )
        self.assertEqual(observed, payload)
        over = copy.deepcopy(payload)
        over["resource"]["child_process_cpu_seconds"] = (
            e24_eval.OOF_CPU_SECONDS_MAX + 1.0
        )
        # A forged stored True cannot rescue the recomputed false check.
        with self.assertRaises(runner.E24StagedRunnerError):
            runner.validate_orchestration_receipt_payload(
                over,
                ledger_sha256=sha("a"),
                run_contract_sha256=sha("b"),
                canary_gate_sha256=sha("c"),
                structural_report_sha256=sha("d"),
            )

    def test_ledger_failure_stops_before_canary_or_any_downstream_capability(self) -> None:
        with (
            mock.patch.object(
                runner.e24_runner,
                "verify_preflight_ledger",
                side_effect=e24_runner.E24RunnerError("bad ledger"),
            ) as ledger,
            mock.patch.object(runner.e24_runner, "verify_feature_canary") as canary,
        ):
            with self.assertRaises(runner.E24StagedRunnerError):
                runner.authenticate_authority(
                    e24_runner.DEFAULT_LEDGER,
                    runner.EXPECTED_GENERATION3_LEDGER_SHA256,
                )
        ledger.assert_called_once()
        canary.assert_not_called()

    def test_metric_entrypoint_requires_caller_pinned_barrier_sha(self) -> None:
        authority = mock.sentinel.authority
        with (
            mock.patch.object(runner, "verify_board_barrier", return_value={}) as barrier,
            mock.patch.object(runner, "_sha", return_value=sha("e")),
        ):
            with self.assertRaisesRegex(runner.E24StagedRunnerError, "caller-pinned"):
                runner.verify_expected_board_barrier(authority, sha("f"), sha("d"))
        barrier.assert_called_once_with(authority, sha("f"))
        self.assertFalse(hasattr(runner, "load_target"))
        self.assertFalse(hasattr(runner, "load_raw_scene"))

    def test_metric_request_is_exact_and_rejects_tamper_extra_or_reorder(self) -> None:
        authority = SimpleNamespace(
            ledger_sha256=sha("a"),
            ledger={"run_contract_sha256": sha("b")},
            structural_report_sha256=sha("c"),
            orchestration_receipt_sha256=sha("d"),
        )
        raw = {
            "validation_name": "synthetic.png",
            "raw_cache": {
                "path": "E:/pazzle_work/edge_confidence/full_graph_cache/image_0010_k64.npz",
                "bytes": 1,
                "file_sha256": sha("e"),
            },
        }
        paths = (
            Path("E:/pazzle_work/posegraph_e24_selector/staged_v1/image_0010/decode.npz"),
            Path("E:/pazzle_work/posegraph_e24_selector/staged_v1/image_0010/decode.commit.json"),
            Path("E:/pazzle_work/posegraph_e24_selector/staged_v1/image_0010/board.npz"),
            Path("E:/pazzle_work/posegraph_e24_selector/staged_v1/image_0010/board.commit.json"),
        )
        with (
            mock.patch.object(runner.e24_runner, "_validated_upstream_projection", return_value=raw),
            mock.patch.object(runner, "_scene_paths", return_value=paths),
            mock.patch.object(runner, "_sha", return_value=sha("f")),
        ):
            request = runner._build_metric_request(
                authority,
                image=10,
                seal_sha256=sha("1"),
                barrier_sha256=sha("2"),
                previous_response_sha256=runner.METRIC_CHAIN_GENESIS_SHA256,
            )
            self.assertEqual(
                runner._validate_metric_request(
                    authority,
                    payload=request,
                    seal_sha256=sha("1"),
                    barrier_sha256=sha("2"),
                ),
                request,
            )
            extra = dict(request)
            extra["target"] = "forbidden"
            with self.assertRaises(runner.E24StagedRunnerError):
                runner._validate_metric_request(
                    authority,
                    payload=extra,
                    seal_sha256=sha("1"),
                    barrier_sha256=sha("2"),
                )
            tampered = dict(request)
            tampered["raw_archive_path"] = "E:/foreign.npz"
            with self.assertRaises(runner.E24StagedRunnerError):
                runner._validate_metric_request(
                    authority,
                    payload=tampered,
                    seal_sha256=sha("1"),
                    barrier_sha256=sha("2"),
                )
        reordered = dict(request)
        reordered["image"] = 11
        reordered["sequence_index"] = 1
        with self.assertRaises(runner.E24StagedRunnerError):
            runner._validate_metric_request(
                authority,
                payload=reordered,
                seal_sha256=sha("1"),
                barrier_sha256=sha("2"),
            )

    def test_subprocess_argv_contains_only_narrow_paths_and_authority(self) -> None:
        completed = SimpleNamespace(returncode=0, stdout="", stderr="")
        with mock.patch.object(runner.subprocess, "run", return_value=completed) as call:
            runner._spawn_metric_worker(
                request_path=Path("E:/pazzle_work/posegraph_e24_selector/staged_v1/metric_broker/image_0010/request.json"),
                response_path=Path("E:/pazzle_work/posegraph_e24_selector/staged_v1/metric_broker/image_0010/response.json"),
                ledger_path=e24_runner.DEFAULT_LEDGER,
                ledger_sha256=sha("a"),
                seal_sha256=sha("b"),
                barrier_sha256=sha("c"),
            )
        argv = call.call_args.args[0]
        self.assertIn("_metric-worker", argv)
        self.assertNotIn("permutation", " ".join(argv).lower())
        self.assertNotIn("target", " ".join(argv).lower())

    def test_smoke_is_data_free_and_reports_sealed_broker(self) -> None:
        output = io.StringIO()
        with (
            redirect_stdout(output),
            mock.patch.object(runner, "authenticate_authority") as authority,
        ):
            runner.main(["smoke"])
        authority.assert_not_called()
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["status"], "data_free")
        self.assertEqual(payload["metric_broker"], "frozen_data_free_not_invoked")
        self.assertEqual(
            payload["metric_broker_contract_sha256"],
            staged.METRIC_BROKER_CONTRACT_SHA256,
        )
        self.assertFalse(payload["target_or_e25_opened"])

    def test_all_generated_paths_are_under_e24_on_e(self) -> None:
        for path in (
            runner.STAGED_ROOT,
            runner.SEAL_PATH,
            runner.BARRIER_PATH,
            runner.REPORT_PATH,
            runner.METRIC_ROOT,
        ):
            resolved = path.resolve(strict=False)
            self.assertEqual(resolved.drive.upper(), "E:")
            self.assertTrue(str(resolved).startswith(str(runner.STORAGE_ROOT.resolve(strict=False))))


if __name__ == "__main__":
    unittest.main()

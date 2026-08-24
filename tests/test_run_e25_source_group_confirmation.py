from __future__ import annotations

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

import eval_e25_source_group_confirmation as e25
import run_e25_source_group_confirmation as runner


def sha(char: str) -> str:
    return char * 64


def authority_payload() -> dict:
    return {
        "schema": runner.UPSTREAM_AUTHORITY_SCHEMA,
        "status": "authenticated_e24_structural_staged_final_all8",
        "ledger": {
            "path": str(runner.E24_LEDGER_PATH.resolve()),
            "sha256": sha("a"),
            "run_contract_sha256": sha("b"),
        },
        "structural": {
            "path": str(runner.E24_STRUCTURAL_REPORT_PATH.resolve()),
            "sha256": sha("c"),
            "passed": True,
            "stage": "go_staged_end_to_end",
        },
        "orchestration": {
            "path": str(runner.E24_ORCHESTRATION_RECEIPT_PATH.resolve()),
            "sha256": sha("d"),
            "status": "pass",
        },
        "staged": {
            "path": str(runner.E24_STAGED_REPORT_PATH.resolve()),
            "sha256": sha("e"),
            "passed": True,
            "stage": "go_final_all8_fit",
            "premetric_seal_sha256": sha("f"),
            "board_barrier_sha256": sha("1"),
            "metric_broker_contract_sha256": sha("2"),
        },
        "final_all8": {
            "manifest_path": str(runner.E24_FINAL_MANIFEST_PATH.resolve()),
            "manifest_sha256": sha("3"),
            "model_path": str(runner.E24_FINAL_MODEL_PATH.resolve()),
            "model_sha256": sha("4"),
            "status": "complete_pass_only_final_all8",
            "e25_opened": False,
        },
        "e25_pixels_logits_features_predictions_labels_metrics_opened": False,
    }


class E25RunnerSyntheticTests(unittest.TestCase):
    def test_upstream_requires_every_pass_not_status_text_only(self) -> None:
        payload = authority_payload()
        self.assertEqual(runner.validate_upstream_authority_payload(payload), payload)
        for section, key, value in (
            ("structural", "passed", False),
            ("orchestration", "status", "fail"),
            ("staged", "stage", "kill_crs_v1"),
            ("final_all8", "e25_opened", True),
        ):
            bad = json.loads(json.dumps(payload))
            bad[section][key] = value
            with self.assertRaises(runner.E25RunnerError):
                runner.validate_upstream_authority_payload(bad)

    def test_upstream_rejects_wrong_literal_path_and_hash(self) -> None:
        payload = authority_payload()
        bad_path = json.loads(json.dumps(payload))
        bad_path["staged"]["path"] = "E:/wrong/report.json"
        with self.assertRaises(runner.E25RunnerError):
            runner.validate_upstream_authority_payload(bad_path)
        bad_hash = json.loads(json.dumps(payload))
        bad_hash["final_all8"]["model_sha256"] = "A" * 64
        with self.assertRaises(runner.E25RunnerError):
            runner.validate_upstream_authority_payload(bad_hash)

    def test_final_authenticator_projection_binds_all_layers(self) -> None:
        payload = authority_payload()
        upstream = SimpleNamespace(
            ledger_path=runner.E24_LEDGER_PATH,
            ledger_sha256=payload["ledger"]["sha256"],
            ledger={"run_contract_sha256": payload["ledger"]["run_contract_sha256"]},
            structural_report_sha256=payload["structural"]["sha256"],
            structural_report={
                "decision": {"passed": True},
                "stage": "go_staged_end_to_end",
            },
            orchestration_receipt_sha256=payload["orchestration"]["sha256"],
            orchestration_receipt={"status": "pass"},
        )
        authority = SimpleNamespace(
            upstream=upstream,
            staged_report_path=runner.E24_STAGED_REPORT_PATH,
            staged_report_sha256=payload["staged"]["sha256"],
            staged_report={"decision": {"passed": True}, "stage": "go_final_all8_fit"},
            premetric_seal_sha256=payload["staged"]["premetric_seal_sha256"],
            board_barrier_sha256=payload["staged"]["board_barrier_sha256"],
            metric_broker_contract_sha256=payload["staged"][
                "metric_broker_contract_sha256"
            ],
        )
        authenticated = SimpleNamespace(
            authority=authority,
            manifest_path=runner.E24_FINAL_MANIFEST_PATH,
            manifest_sha256=payload["final_all8"]["manifest_sha256"],
            manifest={
                "status": "complete_pass_only_final_all8",
                "e25_opened": False,
            },
            model_path=runner.E24_FINAL_MODEL_PATH,
            model_sha256=payload["final_all8"]["model_sha256"],
        )
        self.assertEqual(runner.build_upstream_authority_payload(authenticated), payload)

    def test_smoke_is_data_free_and_makes_no_directory(self) -> None:
        output = io.StringIO()
        with mock.patch.object(Path, "mkdir") as mkdir, redirect_stdout(output):
            runner.main(["smoke"])
        mkdir.assert_not_called()
        result = json.loads(output.getvalue())
        self.assertEqual(result["status"], "data_free_fail_closed")
        self.assertFalse(result["artifact_root_created"])

    def test_every_real_mode_refuses_before_authority_or_manifest_access(self) -> None:
        modes = (
            "run-canary",
            "prepare-label-free",
            "verify-label-free-barrier",
            "metric-broker",
            "evaluate",
        )
        for mode in modes:
            with self.subTest(mode=mode), mock.patch.object(
                runner, "authenticate_upstream"
            ) as authority, mock.patch.object(
                e25, "load_and_validate_source_manifest"
            ) as manifest:
                with self.assertRaisesRegex(runner.E25RunnerError, "sealed before data access"):
                    runner.main([mode])
                authority.assert_not_called()
                manifest.assert_not_called()

    def test_source_seal_freeze_authenticates_before_manifest(self) -> None:
        order: list[str] = []
        upstream_payload = authority_payload()
        authority = runner.AuthenticatedUpstream(
            payload=upstream_payload,
            sha256=sha("5"),
            final_model_sha256=sha("4"),
        )

        def auth():
            order.append("authority")
            return authority

        def manifest():
            order.append("manifest")
            return []

        with mock.patch.object(runner, "authenticate_upstream", side_effect=auth), mock.patch.object(
            e25, "load_and_validate_source_manifest", side_effect=manifest
        ), mock.patch.object(
            runner, "build_source_seal_payload", return_value={"synthetic": True}
        ), mock.patch.object(
            runner, "_prepare_runtime_for_seal_write"
        ) as prepare, mock.patch.object(
            runner, "_commit_create_or_verify", return_value=sha("6")
        ):
            payload, digest = runner.freeze_source_seal()
        self.assertEqual(order, ["authority", "manifest"])
        prepare.assert_called_once_with()
        self.assertEqual(payload, {"synthetic": True})
        self.assertEqual(digest, sha("6"))

    def test_source_seal_carries_frozen_protocol_gates_and_unopened_flag(self) -> None:
        # Patch only the synthetic record digest; no metadata/data file is read.
        records = [
            {
                "name": name,
                "source_group": f"group_{index}",
                "target_sha256": sha("a"),
            }
            for index, name in enumerate(e25.E25_NAMES)
        ]
        digest = e25.sha256_bytes(
            e25.canonical_json_bytes(records, newline=False)
        )
        authority = runner.AuthenticatedUpstream(
            payload=authority_payload(),
            sha256=sha("5"),
            final_model_sha256=sha("4"),
        )
        with mock.patch.object(
            e25, "E25_CANONICAL_RECORDS_SHA256", digest
        ), mock.patch.object(runner, "_source_hashes", return_value={"source": sha("7")}):
            seal = runner.build_source_seal_payload(authority, records)
        self.assertEqual(seal["protocol"]["staged_gates"]["strict_positive_final_wins_min"], 30)
        self.assertFalse(
            seal[
                "pixels_logits_features_predictions_permutations_targets_labels_metrics_opened"
            ]
        )
        self.assertEqual(seal["real_worker_implementation"], "sealed_pending_separate_review")


if __name__ == "__main__":
    unittest.main()

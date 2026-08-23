"""Fail-closed premetric runner for frozen E25 confirmation.

Only the upstream-authority and metadata/source-seal transaction is
implemented.  Every command that could open an E25 pixel, logit, feature,
permutation, target, label, board or metric refuses before resolving a dataset
path.  Import and ``smoke`` perform no filesystem write.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


# Assignment alone creates nothing.  It keeps any later lazy import bytecode
# off C: when this file is not invoked with the documented ``python -B``.
_E25_ROOT_LITERAL = Path("E:/pazzle_work/posegraph_e25_confirmation")
sys.dont_write_bytecode = True
sys.pycache_prefix = str(_E25_ROOT_LITERAL / "pycache")
for _runtime_key in (
    "TEMP",
    "TMP",
    "TMPDIR",
    "JOBLIB_TEMP_FOLDER",
    "LIGHTGBM_TMPDIR",
    "PYTHONPYCACHEPREFIX",
):
    os.environ[_runtime_key] = str(
        _E25_ROOT_LITERAL / ("pycache" if _runtime_key == "PYTHONPYCACHEPREFIX" else "tmp")
    )

import eval_e25_source_group_confirmation as e25


class E25RunnerError(RuntimeError):
    """E25 authority, provenance or process separation failed."""


ROOT = Path(__file__).resolve().parents[1]
STORAGE_ROOT = e25.STORAGE_ROOT
RUNTIME_ROOT = STORAGE_ROOT / "tmp"
PYCACHE_ROOT = STORAGE_ROOT / "pycache"
SEAL_PATH = STORAGE_ROOT / "premetric" / "e25_source_seal.json"
CANARY_ROOT = STORAGE_ROOT / "canary"
LABEL_FREE_ROOT = STORAGE_ROOT / "label_free_v1"
LABEL_FREE_BARRIER_PATH = LABEL_FREE_ROOT / "label_free_48_barrier.json"
METRIC_ROOT = STORAGE_ROOT / "metric_broker_v1"
REPORT_PATH = STORAGE_ROOT / "e25_source_group_confirmation_v1.json"

E24_LEDGER_PATH = Path(
    "E:/pazzle_work/posegraph_e24_selector/preflight/e24_crs_v1_preflight.json"
)
E24_STRUCTURAL_REPORT_PATH = Path(
    "E:/pazzle_work/posegraph_e24_selector/contextual_relation_selector_oof_v1.json"
)
E24_ORCHESTRATION_RECEIPT_PATH = Path(
    "E:/pazzle_work/posegraph_e24_selector/oof_orchestration_receipt.json"
)
E24_STAGED_REPORT_PATH = Path(
    "E:/pazzle_work/posegraph_e24_selector/staged_v1/staged_ssim_nlm_report.json"
)
E24_FINAL_MANIFEST_PATH = Path(
    "E:/pazzle_work/posegraph_e24_selector/final/final_all8_manifest.json"
)
E24_FINAL_MODEL_PATH = Path(
    "E:/pazzle_work/posegraph_e24_selector/final/model.txt"
)

PROTOCOL_DOCUMENT = ROOT / "E25_SOURCE_GROUP_DISJOINT_CONFIRMATION.md"
PLAN_DOCUMENT = ROOT / "autoresearch-runs/pazzle-solution-20260806/PLAN.md"
SOURCE_FILES = (
    Path(__file__).resolve(),
    ROOT / "src/eval_e25_source_group_confirmation.py",
    ROOT / "tests/test_e25_source_group_confirmation.py",
    ROOT / "tests/test_run_e25_source_group_confirmation.py",
    PROTOCOL_DOCUMENT,
    ROOT / "E24_CONTEXT_RELATION_SELECTOR.md",
    PLAN_DOCUMENT,
)

UPSTREAM_AUTHORITY_SCHEMA = "pazzle-e25-crs-v1-upstream-authority-v1"


@dataclass(frozen=True)
class AuthenticatedUpstream:
    payload: Mapping[str, Any]
    sha256: str
    final_model_sha256: str


def _lower_sha(value: object, *, label: str) -> str:
    try:
        return e25.lower_sha256(value, label=label)
    except e25.E25ContractError as exc:
        raise E25RunnerError(str(exc)) from exc


def _sha(path: Path) -> str:
    try:
        return e25.sha256_file(path)
    except OSError as exc:
        raise E25RunnerError(f"cannot hash {path}") from exc


def _canonical_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(e25.canonical_json_bytes(dict(value)).decode("ascii"))
    except e25.E25ContractError as exc:
        raise E25RunnerError(str(exc)) from exc


def _require_literal_file(path: Path, expected: Path, *, label: str) -> Path:
    if path.resolve(strict=False) != expected.resolve(strict=False) or not path.is_file():
        raise E25RunnerError(f"{label} must exist at its literal frozen path")
    return path.resolve()


def build_upstream_authority_payload(authenticated_final: Any) -> dict[str, Any]:
    """Project only authenticated E24 authority hashes, never E25 data."""

    try:
        final_authority = authenticated_final.authority
        upstream = final_authority.upstream
        staged_report = dict(final_authority.staged_report)
        final_manifest = dict(authenticated_final.manifest)
        payload = {
            "schema": UPSTREAM_AUTHORITY_SCHEMA,
            "status": "authenticated_e24_structural_staged_final_all8",
            "ledger": {
                "path": str(upstream.ledger_path.resolve()),
                "sha256": upstream.ledger_sha256,
                "run_contract_sha256": upstream.ledger["run_contract_sha256"],
            },
            "structural": {
                "path": str(E24_STRUCTURAL_REPORT_PATH.resolve()),
                "sha256": upstream.structural_report_sha256,
                "passed": upstream.structural_report["decision"]["passed"],
                "stage": upstream.structural_report["stage"],
            },
            "orchestration": {
                "path": str(E24_ORCHESTRATION_RECEIPT_PATH.resolve()),
                "sha256": upstream.orchestration_receipt_sha256,
                "status": upstream.orchestration_receipt["status"],
            },
            "staged": {
                "path": str(final_authority.staged_report_path.resolve()),
                "sha256": final_authority.staged_report_sha256,
                "passed": staged_report["decision"]["passed"],
                "stage": staged_report["stage"],
                "premetric_seal_sha256": final_authority.premetric_seal_sha256,
                "board_barrier_sha256": final_authority.board_barrier_sha256,
                "metric_broker_contract_sha256": (
                    final_authority.metric_broker_contract_sha256
                ),
            },
            "final_all8": {
                "manifest_path": str(authenticated_final.manifest_path.resolve()),
                "manifest_sha256": authenticated_final.manifest_sha256,
                "model_path": str(authenticated_final.model_path.resolve()),
                "model_sha256": authenticated_final.model_sha256,
                "status": final_manifest["status"],
                "e25_opened": final_manifest["e25_opened"],
            },
            "e25_pixels_logits_features_predictions_labels_metrics_opened": False,
        }
    except (AttributeError, KeyError, TypeError) as exc:
        raise E25RunnerError("final authenticator returned an incomplete authority") from exc
    return validate_upstream_authority_payload(payload)


def validate_upstream_authority_payload(payload: object) -> dict[str, Any]:
    """Strict synthetic-testable validator for the complete authority chain."""

    expected_top = {
        "schema",
        "status",
        "ledger",
        "structural",
        "orchestration",
        "staged",
        "final_all8",
        "e25_pixels_logits_features_predictions_labels_metrics_opened",
    }
    if type(payload) is not dict or set(payload) != expected_top:
        raise E25RunnerError("E25 upstream authority field set drifted")
    expected_nested = {
        "ledger": {"path", "sha256", "run_contract_sha256"},
        "structural": {"path", "sha256", "passed", "stage"},
        "orchestration": {"path", "sha256", "status"},
        "staged": {
            "path",
            "sha256",
            "passed",
            "stage",
            "premetric_seal_sha256",
            "board_barrier_sha256",
            "metric_broker_contract_sha256",
        },
        "final_all8": {
            "manifest_path",
            "manifest_sha256",
            "model_path",
            "model_sha256",
            "status",
            "e25_opened",
        },
    }
    for key, fields in expected_nested.items():
        if type(payload[key]) is not dict or set(payload[key]) != fields:
            raise E25RunnerError(f"E25 upstream {key} field set drifted")
    for container, keys in {
        "ledger": ("sha256", "run_contract_sha256"),
        "structural": ("sha256",),
        "orchestration": ("sha256",),
        "staged": (
            "sha256",
            "premetric_seal_sha256",
            "board_barrier_sha256",
            "metric_broker_contract_sha256",
        ),
        "final_all8": ("manifest_sha256", "model_sha256"),
    }.items():
        for key in keys:
            _lower_sha(payload[container][key], label=f"{container} {key}")
    expected_paths = {
        ("ledger", "path"): E24_LEDGER_PATH,
        ("structural", "path"): E24_STRUCTURAL_REPORT_PATH,
        ("orchestration", "path"): E24_ORCHESTRATION_RECEIPT_PATH,
        ("staged", "path"): E24_STAGED_REPORT_PATH,
        ("final_all8", "manifest_path"): E24_FINAL_MANIFEST_PATH,
        ("final_all8", "model_path"): E24_FINAL_MODEL_PATH,
    }
    for (section, key), expected in expected_paths.items():
        if Path(payload[section][key]).resolve(strict=False) != expected.resolve(strict=False):
            raise E25RunnerError(f"E25 upstream {section} path drifted")
    if (
        payload["schema"] != UPSTREAM_AUTHORITY_SCHEMA
        or payload["status"] != "authenticated_e24_structural_staged_final_all8"
        or payload["structural"]["passed"] is not True
        or payload["structural"]["stage"] != "go_staged_end_to_end"
        or payload["orchestration"]["status"] != "pass"
        or payload["staged"]["passed"] is not True
        or payload["staged"]["stage"] != "go_final_all8_fit"
        or payload["final_all8"]["status"] != "complete_pass_only_final_all8"
        or payload["final_all8"]["e25_opened"] is not False
        or payload[
            "e25_pixels_logits_features_predictions_labels_metrics_opened"
        ]
        is not False
    ):
        raise E25RunnerError("E25 upstream chain is not the exact unopened PASS")
    return _canonical_mapping(payload)


def authenticate_upstream() -> AuthenticatedUpstream:
    """Invoke the final model owner's full validator; open no E25 artifact."""

    try:
        import run_e24_final_all8 as final_runner

        if (
            final_runner.MANIFEST_PATH.resolve(strict=False)
            != E24_FINAL_MANIFEST_PATH.resolve(strict=False)
            or final_runner.MODEL_PATH.resolve(strict=False)
            != E24_FINAL_MODEL_PATH.resolve(strict=False)
        ):
            raise E25RunnerError("final-all8 owner path constants drifted")
        authenticated_final = final_runner.authenticate_final_model()
    except Exception as exc:
        raise E25RunnerError("E24 structural/staged/final authority failed") from exc
    payload = build_upstream_authority_payload(authenticated_final)
    digest = e25.sha256_bytes(e25.canonical_json_bytes(payload))
    return AuthenticatedUpstream(
        payload=payload,
        sha256=digest,
        final_model_sha256=payload["final_all8"]["model_sha256"],
    )


def _source_hashes() -> dict[str, str]:
    output: dict[str, str] = {}
    for path in SOURCE_FILES:
        resolved = path.resolve()
        if not resolved.is_file():
            raise E25RunnerError(f"E25 source/protocol file is absent: {resolved}")
        output[str(resolved)] = _sha(resolved)
    return dict(sorted(output.items()))


def build_source_seal_payload(
    authority: AuthenticatedUpstream,
    records: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    try:
        normalized = e25.validate_sealed_records(list(records))
    except e25.E25ContractError as exc:
        raise E25RunnerError(str(exc)) from exc
    if type(authority) is not AuthenticatedUpstream:
        raise E25RunnerError("E25 source seal requires exact authenticated authority")
    return {
        "schema": e25.SOURCE_SEAL_SCHEMA,
        "schema_version": e25.SCHEMA_VERSION,
        "status": "frozen_premetric_manifest_only",
        "protocol": _canonical_mapping(e25.E25_PROTOCOL),
        "protocol_sha256": e25.PROTOCOL_SHA256,
        "metric_broker_contract": _canonical_mapping(e25.METRIC_BROKER_CONTRACT),
        "metric_broker_contract_sha256": e25.METRIC_BROKER_CONTRACT_SHA256,
        "sources_sha256": _source_hashes(),
        "upstream_authority": dict(authority.payload),
        "upstream_authority_sha256": authority.sha256,
        "source_manifest": {
            "path": str(e25.SOURCE_GROUP_MANIFEST_PATH.resolve()),
            "sha256": e25.SOURCE_GROUP_MANIFEST_SHA256,
            "read_scope": "metadata_json_only_no_target_member",
        },
        "records": normalized,
        "records_sha256": e25.E25_CANONICAL_RECORDS_SHA256,
        "newline_list_sha256": e25.E25_NEWLINE_LIST_SHA256,
        "source_group_disjoint": {
            "count": 48,
            "unique_groups": 48,
            "against_training_0_6699": True,
            "against_validation_relative_0_99": True,
            "against_e24_10_17": True,
        },
        "canary": {
            "image": e25.E25_CANARY_ID,
            "selection": "first_sealed_manifest_id_without_data_access",
            "opened": False,
        },
        "authorized_outputs": [
            "separately_reviewed_scene_226_label_free_canary",
            "separately_reviewed_48_scene_label_free_commits",
            "global_label_free_barrier_before_metric_broker",
        ],
        "real_worker_implementation": "sealed_pending_separate_review",
        "pixels_logits_features_predictions_permutations_targets_labels_metrics_opened": False,
    }


def _prepare_runtime_for_seal_write() -> None:
    """Create only E25 premetric/runtime directories, after full authority."""

    for key, value in {
        "TEMP": RUNTIME_ROOT,
        "TMP": RUNTIME_ROOT,
        "TMPDIR": RUNTIME_ROOT,
        "JOBLIB_TEMP_FOLDER": RUNTIME_ROOT,
        "LIGHTGBM_TMPDIR": RUNTIME_ROOT,
        "PYTHONPYCACHEPREFIX": PYCACHE_ROOT,
    }.items():
        candidate = Path(value)
        if candidate.drive.upper() != "E:":
            raise E25RunnerError(f"unsafe E25 runtime path for {key}")
        os.environ[key] = str(candidate)
    sys.pycache_prefix = str(PYCACHE_ROOT)
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    PYCACHE_ROOT.mkdir(parents=True, exist_ok=True)
    SEAL_PATH.parent.mkdir(parents=True, exist_ok=True)


def _commit_create_or_verify(path: Path, payload: Mapping[str, Any]) -> str:
    if path.resolve(strict=False) != SEAL_PATH.resolve(strict=False):
        raise E25RunnerError("only the literal E25 premetric seal may be committed")
    body = e25.canonical_json_bytes(dict(payload))
    digest = e25.sha256_bytes(body)
    if path.exists():
        try:
            observed = path.read_bytes()
        except OSError as exc:
            raise E25RunnerError("existing E25 source seal is unreadable") from exc
        if observed != body:
            raise E25RunnerError("existing E25 source seal drifted; append-only stop")
        return digest
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != body:
                raise E25RunnerError("concurrent E25 source seal drifted")
    except E25RunnerError:
        raise
    except OSError as exc:
        raise E25RunnerError("atomic E25 source seal commit failed") from exc
    finally:
        if temporary.exists():
            temporary.unlink()
    return digest


def freeze_source_seal() -> tuple[dict[str, Any], str]:
    """Authenticate E24 first, then read only pinned metadata and seal it."""

    authority = authenticate_upstream()
    try:
        records = e25.load_and_validate_source_manifest()
    except e25.E25ContractError as exc:
        raise E25RunnerError("E25 metadata manifest verification failed") from exc
    payload = build_source_seal_payload(authority, records)
    _prepare_runtime_for_seal_write()
    return payload, _commit_create_or_verify(SEAL_PATH, payload)


def verify_source_seal() -> tuple[dict[str, Any], str]:
    """Rebuild the expected seal from authenticated metadata and compare bytes."""

    authority = authenticate_upstream()
    try:
        records = e25.load_and_validate_source_manifest()
    except e25.E25ContractError as exc:
        raise E25RunnerError("E25 metadata manifest verification failed") from exc
    expected = build_source_seal_payload(authority, records)
    if not SEAL_PATH.is_file():
        raise E25RunnerError("E25 premetric source seal is absent")
    body = e25.canonical_json_bytes(expected)
    if SEAL_PATH.read_bytes() != body:
        raise E25RunnerError("E25 premetric source seal no longer matches authority")
    return expected, e25.sha256_bytes(body)


def refuse_real_data_mode(mode: str) -> None:
    """Hard stop before any E25 data path, archive or member is resolved."""

    raise E25RunnerError(
        f"E25 real mode {mode!r} is sealed before data access: the concrete "
        "trusted lineage, label-free worker and metric-broker adapters require "
        "separate review; do not bypass this guard"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "smoke",
            "verify-upstream",
            "freeze-source-seal",
            "verify-source-seal",
            "run-canary",
            "prepare-label-free",
            "verify-label-free-barrier",
            "metric-broker",
            "evaluate",
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.mode == "smoke":
        print(
            e25.canonical_json_bytes(
                {
                    "status": "data_free_fail_closed",
                    "protocol_sha256": e25.PROTOCOL_SHA256,
                    "metric_broker_contract_sha256": e25.METRIC_BROKER_CONTRACT_SHA256,
                    "sealed_scenes": 48,
                    "canary": e25.E25_CANARY_ID,
                    "artifact_root_created": False,
                    "real_data_modes": "sealed_pending_separate_review",
                    "python": platform.python_version(),
                }
            ).decode("ascii"),
            end="",
        )
        return
    if args.mode == "verify-upstream":
        authority = authenticate_upstream()
        if STORAGE_ROOT.exists():
            raise E25RunnerError(
                "verify-upstream unexpectedly observed an E25 artifact root"
            )
        print(
            e25.canonical_json_bytes(
                {
                    "status": "pass",
                    "upstream_authority_sha256": authority.sha256,
                    "final_model_sha256": authority.final_model_sha256,
                    "e25_opened": False,
                }
            ).decode("ascii"),
            end="",
        )
        return
    if args.mode == "freeze-source-seal":
        _payload, digest = freeze_source_seal()
        print(
            e25.canonical_json_bytes(
                {"status": "frozen", "path": str(SEAL_PATH), "sha256": digest}
            ).decode("ascii"),
            end="",
        )
        return
    if args.mode == "verify-source-seal":
        _payload, digest = verify_source_seal()
        print(
            e25.canonical_json_bytes(
                {"status": "pass", "path": str(SEAL_PATH), "sha256": digest}
            ).decode("ascii"),
            end="",
        )
        return
    refuse_real_data_mode(args.mode)


if __name__ == "__main__":
    main()


__all__ = (
    "AuthenticatedUpstream",
    "E25RunnerError",
    "LABEL_FREE_BARRIER_PATH",
    "REPORT_PATH",
    "SEAL_PATH",
    "UPSTREAM_AUTHORITY_SCHEMA",
    "authenticate_upstream",
    "build_source_seal_payload",
    "build_upstream_authority_payload",
    "freeze_source_seal",
    "main",
    "refuse_real_data_mode",
    "validate_upstream_authority_payload",
    "verify_source_seal",
)

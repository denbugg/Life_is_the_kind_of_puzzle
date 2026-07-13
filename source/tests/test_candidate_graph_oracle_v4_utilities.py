from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest


REPO = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


audit_util = _load(
    "candidate_graph_oracle_v4_launch_audit_utility_test",
    "scripts/audit_candidate_graph_oracle_v4_launch_closure.py",
)
download_util = _load(
    "candidate_graph_oracle_v4_download_utility_test",
    "scripts/download_candidate_graph_oracle_v4_phase_a_files.py",
)
materialize_util = _load(
    "candidate_graph_oracle_v4_materialize_utility_test",
    "scripts/materialize_candidate_graph_oracle_v4_bound_verifier_repo.py",
)


def _canonical_hash(value: dict[str, Any]) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, value: dict[str, Any], *, repository: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if repository:
        raw = (json.dumps(value, ensure_ascii=True, indent=2) + "\n").encode("utf-8")
    else:
        raw = (
            json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            + "\n"
        ).encode("utf-8")
    path.write_bytes(raw)


def _kernel_metadata(
    *, kernel_id: int = -1, reservation_sha256: str | None = None
) -> dict[str, Any]:
    datasets = {
        label: {"slug": slug, "version": audit_util.KERNEL_VERSION}
        for label, slug in audit_util.DATASETS.items()
    }
    return {
        "id": audit_util.KERNEL_SLUG,
        "id_no": kernel_id,
        "reservation_receipt_sha256": reservation_sha256,
        "is_private": True,
        "enable_gpu": True,
        "machine_shape": "NvidiaTeslaT4",
        "enable_internet": False,
        "dataset_sources": [
            f"{slug}/{audit_util.KERNEL_VERSION}"
            for slug in audit_util.DATASETS.values()
        ],
        "oracle_launch_expectation": {
            "kernel_id": kernel_id,
            "kernel_slug": audit_util.KERNEL_SLUG,
            "kernel_version": audit_util.KERNEL_VERSION,
            "reservation_receipt_sha256": reservation_sha256,
            "dataset_versions": datasets,
        },
    }


def _draft_repo(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    root = tmp_path / "repo"
    root.mkdir()
    for index, (_, _, relative) in enumerate(materialize_util.CODE_PIN_PAIRS):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == audit_util.KERNEL_METADATA_RELATIVE:
            _write_json(path, _kernel_metadata())
        elif relative.endswith("candidate_graph_oracle_v4_phase_a_job/run_phase_a.py"):
            path.write_text(
                "KERNEL_ID = -1\nRESERVATION_RECEIPT_SHA256 = None\n",
                encoding="utf-8",
            )
        elif relative == "scripts/push_candidate_graph_oracle_v4_phase_a.py":
            path.write_text(
                "EXPECTED_KERNEL_ID = -1\nRESERVATION_RECEIPT_SHA256 = None\n",
                encoding="utf-8",
            )
        else:
            path.write_bytes(f"v4-source-{index}\n".encode("ascii"))
    live_config = json.loads(
        (REPO / materialize_util.CONFIG_RELATIVE).read_text(encoding="utf-8")
    )
    frozen = copy.deepcopy(live_config["frozen_contract"])
    assert _canonical_hash(frozen) == materialize_util.FROZEN_CONTRACT_SHA256
    assert audit_util.FROZEN_CONTRACT_SHA256 == materialize_util.FROZEN_CONTRACT_SHA256
    assert download_util.FROZEN_CONTRACT_SHA256 == materialize_util.FROZEN_CONTRACT_SHA256
    for known_relative, expected_sha in frozen["assets"]["known_code_sha256"].items():
        source = REPO / known_relative
        destination = root / known_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        assert hashlib.sha256(destination.read_bytes()).hexdigest() == expected_sha
    code_policy = [
        {"path_field": path_field, "sha256_field": sha_field}
        for path_field, sha_field, _ in materialize_util.CODE_PIN_PAIRS
    ]
    fixture_policy = [
        {"path_field": path_field, "sha256_field": sha_field}
        for path_field, sha_field, _ in audit_util.FIXTURE_PIN_PAIRS
    ]
    runtime_pins: dict[str, Any] = {}
    for path_field, sha_field, relative in materialize_util.CODE_PIN_PAIRS:
        runtime_pins[path_field] = relative
        runtime_pins[sha_field] = None
    for path_field, sha_field, relative in audit_util.FIXTURE_PIN_PAIRS:
        runtime_pins[sha_field] = None
        runtime_pins[path_field] = relative
    config = {
        "schema_version": 1,
        "kind": "candidate_graph_oracle_ceiling",
        "status": "local_pre_reservation_source_closure_no_claims",
        "created_utc": "2026-07-12T00:00:00Z",
        "protocol_instance_id": materialize_util.INSTANCE,
        "decision_basis": {},
        "frozen_contract": frozen,
        "frozen_contract_sha256": _canonical_hash(frozen),
        "runtime_pins": runtime_pins,
        "runtime_pin_mutation_policy": {
            "transition_ledger_root": (
                "runs/assembly_v1/protocol_ledgers/candidate_graph_oracle/"
                + materialize_util.INSTANCE
            ),
            "code_pin_fields": code_policy,
            "fixture_pin_fields": fixture_policy,
        },
        "safe_for_submission": False,
    }
    _write_json(root / materialize_util.CONFIG_RELATIVE, config)
    return root, config


def _rewrite_config(root: Path, config: dict[str, Any]) -> None:
    _write_json(root / materialize_util.CONFIG_RELATIVE, config)


def test_launch_audit_recognizes_but_never_promotes_pre_reservation_draft(
    tmp_path: Path,
) -> None:
    root, _ = _draft_repo(tmp_path)
    result = audit_util.audit(repo_root=root)
    assert result["status"] == "not_launchable"
    assert result["stage"] == "pre_reservation_draft"
    assert result["launch_ready"] is False
    assert result["blockers"] == [
        "kernel_identity_not_reserved",
        "code_pins_are_null",
        "fixture_pins_are_null",
    ]
    assert result["kernel"]["id"] == -1
    assert result["remote_api_called"] is False
    assert result["label_paths_constructed"] is False


def test_launch_audit_fails_closed_on_frozen_hash_or_partial_pins(
    tmp_path: Path,
) -> None:
    root, config = _draft_repo(tmp_path)
    drift = copy.deepcopy(config)
    drift["frozen_contract_sha256"] = "0" * 64
    _rewrite_config(root, drift)
    with pytest.raises(RuntimeError, match="frozen contract SHA"):
        audit_util.audit(repo_root=root)

    partial = copy.deepcopy(config)
    partial["runtime_pins"]["evaluator_sha256"] = "a" * 64
    _rewrite_config(root, partial)
    with pytest.raises(RuntimeError, match="partial code pin"):
        audit_util.audit(repo_root=root)


def test_launch_audit_rejects_zero_kernel_id_as_neither_draft_nor_reserved(
    tmp_path: Path,
) -> None:
    root, _ = _draft_repo(tmp_path)
    _write_json(root / audit_util.KERNEL_METADATA_RELATIVE, _kernel_metadata(kernel_id=0))
    (root / "runs/assembly_v1/kaggle/candidate_graph_oracle_v4_phase_a_job/run_phase_a.py").write_text(
        "KERNEL_ID = 0\nRESERVATION_RECEIPT_SHA256 = None\n",
        encoding="utf-8",
    )
    (root / "scripts/push_candidate_graph_oracle_v4_phase_a.py").write_text(
        "EXPECTED_KERNEL_ID = 0\nRESERVATION_RECEIPT_SHA256 = None\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="exactly -1.*positive integer"):
        audit_util.audit(repo_root=root)


def test_phase_a_runner_rejects_zero_kernel_id_before_mount_access(
    tmp_path: Path,
) -> None:
    source_path = REPO / (
        "runs/assembly_v1/kaggle/"
        "candidate_graph_oracle_v4_phase_a_job/run_phase_a.py"
    )
    source = source_path.read_text(encoding="utf-8")
    lines = source.splitlines()
    assignments = [line for line in lines if line.startswith("KERNEL_ID = ")]
    assert len(assignments) == 1
    zero_runner = tmp_path / "run_phase_a_zero.py"
    zero_runner.write_text(
        "\n".join(
            "KERNEL_ID = 0" if line.startswith("KERNEL_ID = ") else line
            for line in lines
        )
        + "\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, str(zero_runner)],
        cwd=tmp_path,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert completed.returncode != 0
    assert "unresolved or non-positive kernel id" in completed.stdout


def test_v4_closure_uses_instance_suffixed_fixture_root() -> None:
    assert audit_util.FIXTURE_ROOT_RELATIVE == (
        "runs/assembly_v1/"
        "candidate_graph_oracle_fixtures_v4_6c0fe4e8524ce39d830d9a5bee118d8b"
    )


def test_fully_pinned_local_evidence_reads_only_instance_suffixed_fixture_root(
    tmp_path: Path,
) -> None:
    root, config = _draft_repo(tmp_path)
    pins = config["runtime_pins"]
    for index, (_, sha_field, _) in enumerate(audit_util.CODE_PIN_PAIRS, start=1):
        pins[sha_field] = f"{index:064x}"
    pins["fixture_label_manifest_sha256"] = "b" * 64

    common = {
        "protocol_instance_id": audit_util.INSTANCE,
        "frozen_contract_sha256": config["frozen_contract_sha256"],
        "record_count": 64,
        **{
            sha_field: pins[sha_field]
            for _, sha_field, _ in audit_util.CODE_PIN_PAIRS
        },
    }
    fixture_root = root / audit_util.FIXTURE_ROOT_RELATIVE
    input_path = fixture_root / pins["fixture_input_manifest_relative_path"]
    _write_json(input_path, common)
    pins["fixture_input_manifest_sha256"] = hashlib.sha256(
        input_path.read_bytes()
    ).hexdigest()
    lock = {
        **common,
        "fixture_input_manifest_sha256": pins["fixture_input_manifest_sha256"],
        "fixture_label_manifest_sha256": pins["fixture_label_manifest_sha256"],
        "phase_a_may_receive_label_root": False,
        "phase_a_may_receive_master_secret": False,
    }
    lock_path = fixture_root / pins["fixture_lock_relative_path"]
    _write_json(lock_path, lock)
    pins["fixture_lock_sha256"] = hashlib.sha256(lock_path.read_bytes()).hexdigest()

    legacy_root = root / "runs/assembly_v1/candidate_graph_oracle_fixtures_v4"
    _write_json(
        legacy_root / pins["fixture_input_manifest_relative_path"],
        {"protocol_instance_id": "0" * 32},
    )
    _write_json(
        legacy_root / pins["fixture_lock_relative_path"],
        {"protocol_instance_id": "0" * 32},
    )

    with pytest.raises(RuntimeError, match="lifecycle ledger root missing"):
        audit_util._validate_final_local_evidence(
            root,
            config,
            "c" * 64,
            config["frozen_contract_sha256"],
            123456,
        )


def test_launch_audit_validates_reserved_identity_but_keeps_null_pins_blocked(
    tmp_path: Path,
) -> None:
    root, _ = _draft_repo(tmp_path)
    reservation_runner = root / audit_util.RESERVATION_RUNNER_RELATIVE
    reservation_runner.parent.mkdir(parents=True, exist_ok=True)
    reservation_runner.write_bytes((REPO / audit_util.RESERVATION_RUNNER_RELATIVE).read_bytes())
    assert hashlib.sha256(reservation_runner.read_bytes()).hexdigest() == (
        audit_util.RESERVATION_RUNNER_SHA256
    )
    kernel_id = 123456
    payload = {
        "schema_version": 1,
        "kind": "candidate_graph_oracle_v4_kaggle_reservation_receipt",
        "created_utc": "2026-07-12T00:00:00Z",
        "protocol_instance_id": audit_util.INSTANCE,
        "reservation_orchestrator_sha256": audit_util.RESERVATION_ORCHESTRATOR_SHA256,
        "local_validation": {
            "schema_version": 1,
            "kind": "candidate_graph_oracle_v4_local_reservation_validation",
            "protocol_instance_id": audit_util.INSTANCE,
            "reservation_orchestrator_sha256": audit_util.RESERVATION_ORCHESTRATOR_SHA256,
            "contains_fixture_pixels": False,
            "gpu_requested": False,
            "safe_for_submission": False,
        },
        "contains_fixture_pixels": False,
        "gpu_requested": False,
        "dataset_v2_uploaded": False,
        "phase_a_push_performed": False,
        "safe_for_submission": False,
        "kernel": {
            "slug": audit_util.KERNEL_SLUG,
            "kernel_id": kernel_id,
            "reserved_version": 1,
            "is_private": True,
            "enable_gpu": False,
            "enable_tpu": False,
            "enable_internet": False,
            "dataset_sources": [],
            "kernel_sources": [],
            "competition_sources": [],
            "model_sources": [],
            "status": "complete",
            "reservation_runner_sha256": audit_util.RESERVATION_RUNNER_SHA256,
        },
        "datasets": {
            label: {
                "slug": slug,
                "reserved_version": 1,
                "is_private": True,
                "status": "ready",
            }
            for label, slug in audit_util.DATASETS.items()
        },
    }
    envelope = {"payload": payload, "payload_sha256": _canonical_hash(payload)}
    receipt = root / audit_util.RESERVATION_RECEIPT_RELATIVE
    _write_json(receipt, envelope, repository=False)
    receipt_sha = hashlib.sha256(receipt.read_bytes()).hexdigest()
    _write_json(
        root / audit_util.KERNEL_METADATA_RELATIVE,
        _kernel_metadata(kernel_id=kernel_id, reservation_sha256=receipt_sha),
    )
    (root / "runs/assembly_v1/kaggle/candidate_graph_oracle_v4_phase_a_job/run_phase_a.py").write_text(
        f"KERNEL_ID = {kernel_id}\nRESERVATION_RECEIPT_SHA256 = {receipt_sha!r}\n",
        encoding="utf-8",
    )
    (root / "scripts/push_candidate_graph_oracle_v4_phase_a.py").write_text(
        f"EXPECTED_KERNEL_ID = {kernel_id}\nRESERVATION_RECEIPT_SHA256 = {receipt_sha!r}\n",
        encoding="utf-8",
    )
    result = audit_util.audit(repo_root=root)
    assert result["stage"] == "reserved_unpinned"
    assert result["launch_ready"] is False
    assert result["blockers"] == ["code_pins_are_null", "fixture_pins_are_null"]
    assert result["reservation_receipt_sha256"] == receipt_sha


def test_materializer_validate_only_accepts_all_null_draft_without_writes(
    tmp_path: Path,
) -> None:
    root, config = _draft_repo(tmp_path)
    result = materialize_util.validate_source_closure(root)
    assert result["status"] == "valid_pre_reservation_source_closure"
    assert result["code_pin_state"] == "null"
    known_count = len(
        result["source_files"]
    ) - len(materialize_util.CODE_PIN_PAIRS) - 1
    assert known_count == 18
    assert result["files_written"] == 0
    assert result["label_paths_constructed"] is False
    config["runtime_pins"]["evaluator_sha256"] = "a" * 64
    _rewrite_config(root, config)
    with pytest.raises(RuntimeError, match="partial code-pin transition"):
        materialize_util.validate_source_closure(root)


def test_materializer_requires_atomic_code_pins_and_materializes_exact_tree(
    tmp_path: Path,
) -> None:
    root, config = _draft_repo(tmp_path)
    for path_field, sha_field, _ in materialize_util.CODE_PIN_PAIRS:
        config["runtime_pins"][sha_field] = hashlib.sha256(
            (root / config["runtime_pins"][path_field]).read_bytes()
        ).hexdigest()
    _rewrite_config(root, config)
    destination = tmp_path / "bound"
    receipt = tmp_path / "receipt.json"
    result = materialize_util.materialize(
        source_root=root, destination=destination, receipt=receipt
    )
    assert result["status"] == "materialized_and_verified_v4_source_closure"
    assert result["total_files"] == len(materialize_util.CODE_PIN_PAIRS) + 18 + 1
    assert result["label_paths_constructed_or_opened"] is False
    envelope = json.loads(receipt.read_text(encoding="utf-8"))
    assert envelope["payload_sha256"] == _canonical_hash(envelope["payload"])
    expected_files = set(envelope["payload"]["files"])
    actual_files = {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file()
    }
    assert actual_files == expected_files
    with pytest.raises(RuntimeError, match="destination already exists"):
        materialize_util.materialize(
            source_root=root, destination=destination, receipt=tmp_path / "other.json"
        )


def test_materializer_rejects_sensitive_known_code_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, config = _draft_repo(tmp_path)
    forbidden = "private_labels/verifier.py"
    path = root / forbidden
    path.parent.mkdir(parents=True)
    path.write_bytes(b"forbidden\n")
    config["frozen_contract"]["assets"]["known_code_sha256"][forbidden] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    config["frozen_contract_sha256"] = _canonical_hash(config["frozen_contract"])
    monkeypatch.setattr(
        materialize_util,
        "FROZEN_CONTRACT_SHA256",
        config["frozen_contract_sha256"],
    )
    _rewrite_config(root, config)
    with pytest.raises(RuntimeError, match="source-only authority"):
        materialize_util.validate_source_closure(root)


def _phase_a_readback(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    root = tmp_path / "readback"
    finalized = root / download_util.PREFIX
    (finalized / "artifacts").mkdir(parents=True)
    (finalized / "renders").mkdir()
    records: list[dict[str, Any]] = []
    for index in range(64):
        opaque_id = f"{index:032x}"
        graph_relative = f"artifacts/{opaque_id}.graph.npz"
        graph = f"graph-{index}\n".encode("ascii")
        (finalized / graph_relative).write_bytes(graph)
        renders: dict[str, Any] = {}
        for label in download_util.RENDER_LABELS:
            relative = f"renders/{opaque_id}__{label}.png"
            raw = f"render-{index}-{label}\n".encode("ascii")
            (finalized / relative).write_bytes(raw)
            renders[label] = {
                "path": relative,
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        records.append(
            {
                "opaque_id": opaque_id,
                "graph_artifact": graph_relative,
                "graph_artifact_sha256": hashlib.sha256(graph).hexdigest(),
                "renders": renders,
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "frozen_candidate_graph_input_only",
        "protocol_instance_id": download_util.INSTANCE,
        "config_sha256": "1" * 64,
        "fixture_manifest_sha256": "2" * 64,
        "frozen_contract_sha256": download_util.FROZEN_CONTRACT_SHA256,
        "phase_a_lifecycle_sha256": "4" * 64,
        "script_sha256": "5" * 64,
        "record_count": 64,
        "records": records,
        "safe_for_submission": False,
        "target_files_opened": False,
        "target_paths_constructed": False,
    }
    manifest["self_sha256"] = _canonical_hash(manifest)
    _write_json(finalized / "FROZEN_CANDIDATE_GRAPH_MANIFEST.json", manifest)
    return root, manifest


def test_download_validate_only_cli_never_constructs_kaggle_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root, _ = _phase_a_readback(tmp_path)

    def forbidden_api():
        raise AssertionError("Kaggle API must not be constructed")

    def forbidden_write(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("validate-only must not write files")

    monkeypatch.setattr(download_util, "_new_api", forbidden_api)
    monkeypatch.setattr(download_util, "_atomic", forbidden_write)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_candidate_graph_oracle_v4_phase_a_files.py",
            "--readback-root",
            str(root),
            "--validate-only",
        ],
    )
    download_util.main()
    result = json.loads(capsys.readouterr().out)
    assert result["verified_files"] == 256
    assert result["validate_only"] is True
    assert result["remote_api_called"] is False
    assert result["label_fixture_accessed"] is False


def test_download_validate_only_rejects_tamper_and_wrong_instance(
    tmp_path: Path,
) -> None:
    root, manifest = _phase_a_readback(tmp_path)
    first = manifest["records"][0]["graph_artifact"]
    (root / download_util.PREFIX / first).write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="local Phase-A SHA drift"):
        download_util.validate_local(root)

    root2, manifest2 = _phase_a_readback(tmp_path / "second")
    manifest2["frozen_contract_sha256"] = "f" * 64
    manifest2.pop("self_sha256")
    manifest2["self_sha256"] = _canonical_hash(manifest2)
    _write_json(
        root2 / download_util.MANIFEST_RELATIVE,
        manifest2,
    )
    with pytest.raises(RuntimeError, match="frozen contract drift"):
        download_util.validate_local(root2)

    root3, _ = _phase_a_readback(tmp_path / "third")
    with pytest.raises(RuntimeError, match="manifest instance drift"):
        download_util.validate_local(root3, expected_instance_id="f" * 32)

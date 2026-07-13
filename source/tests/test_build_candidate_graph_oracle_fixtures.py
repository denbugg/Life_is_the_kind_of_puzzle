from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import cv2
import kornia
import scipy
import skimage
import torch
from PIL import __version__ as pillow_version

from puzzle_assembly.protocol import source_names_for_split
from scripts import build_candidate_graph_oracle_fixtures as fixtures
from scripts import evaluate_candidate_graph_oracle as evaluator
from scripts import update_candidate_graph_oracle_ledger as lifecycle


def _write(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, payload: dict) -> str:
    encoded = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
    return _write(path, encoded)


def _write_code_transition_receipts(
    repo: Path, config_path: Path, config: dict
) -> None:
    transition = (
        repo
        / config["runtime_pin_mutation_policy"]["transition_ledger_root"]
        / lifecycle.TRANSITION_DIRECTORY
    )
    transition.mkdir(parents=True, exist_ok=True)
    pin_values = {
        pair["sha256_field"]: config["runtime_pins"][pair["sha256_field"]]
        for pair in config["runtime_pin_mutation_policy"]["code_pin_fields"]
    }
    final_config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    intent = {
        "schema_version": 1,
        "kind": lifecycle.TRANSITION_INTENT_KIND,
        "stage": "code",
        "stage_index": 0,
        "protocol_instance_id": config["protocol_instance_id"],
        "frozen_contract_sha256": config["frozen_contract_sha256"],
        "config_relative_path": "configs/protocol.json",
        "previous_config_sha256": "0" * 64,
        "intended_config_sha256": final_config_sha256,
        "pin_sha256_values": pin_values,
        "created_utc": "2026-07-12T00:00:00Z",
    }
    intent_bytes = fixtures._canonical_json_bytes(intent)
    (transition / "00_code_pins.intent.json").write_bytes(intent_bytes)
    completion = {
        "schema_version": 1,
        "kind": lifecycle.TRANSITION_COMPLETION_KIND,
        "stage": "code",
        "stage_index": 0,
        "protocol_instance_id": config["protocol_instance_id"],
        "frozen_contract_sha256": config["frozen_contract_sha256"],
        "config_relative_path": "configs/protocol.json",
        "previous_config_sha256": "0" * 64,
        "final_config_sha256": final_config_sha256,
        "pin_sha256_values": pin_values,
        "intent_sha256": hashlib.sha256(intent_bytes).hexdigest(),
        "completed_utc": "2026-07-12T00:00:01Z",
    }
    (transition / "00_code_pins.complete.json").write_bytes(
        fixtures._canonical_json_bytes(completion)
    )


def _fake_repo(tmp_path: Path, *, source_count: int = 1) -> tuple[Path, Path, list[str]]:
    repo = tmp_path / "repo"
    repo.mkdir()
    train = [f"train_{index:04d}.png" for index in range(4900)]
    validation = [f"val_{index:04d}.png" for index in range(700)]
    audit = [f"audit_{index:04d}.png" for index in range(700)]
    manifest = {"splits": {"train": train, "val": validation, "audit": audit}}
    quarantine = {"quarantine_names": validation[:93]}
    manifest_sha = _write_json(repo / "configs/manifest.json", manifest)
    quarantine_sha = _write_json(repo / "configs/quarantine.json", quarantine)
    ledger_sha = _write_json(
        repo / "configs/audit_ledger.json", {"excluded_names": audit[:32]}
    )
    selected = source_names_for_split(
        "edge_development",
        manifest_path=repo / "configs/manifest.json",
        quarantine_path=repo / "configs/quarantine.json",
        audit_exclusion_path=repo / "configs/audit_ledger.json",
    )[128 : 128 + source_count]

    evaluator_sha = _write(repo / "scripts/evaluator.py", b"# evaluator\n")
    tests_sha = _write(repo / "tests/test_evaluator.py", b"# tests\n")
    builder_sha = _write(repo / "scripts/builder.py", b"# builder\n")
    builder_tests_sha = _write(
        repo / "tests/test_builder.py", b"# builder integration tests\n"
    )
    finalizer_sha = _write(repo / "scripts/finalizer.py", b"# finalizer\n")
    lifecycle_sha = _write(repo / "scripts/lifecycle.py", b"# lifecycle\n")
    verifier_sha = _write(repo / "scripts/verifier.py", b"# verifier\n")
    phase_a_runner_sha = _write(repo / "jobs/run_phase_a.py", b"# phase a\n")
    kernel_metadata_sha = _write_json(
        repo / "jobs/kernel-metadata.json", {"kind": "test metadata"}
    )
    launcher_sha = _write(repo / "scripts/launch.py", b"# launcher\n")
    phase_b_runner_sha = _write(repo / "scripts/run_phase_b.py", b"# phase b\n")
    denoiser_sha = _write(repo / "assets/denoiser.pt", b"denoiser")
    hbt_sha = _write(repo / "assets/hbt.pt", b"hbt")
    code_sha = _write(repo / "src/dummy.py", b"# pinned code\n")
    components_sha = _write(
        repo / "src/puzzle_assembly/components.py", b"# private module pin\n"
    )
    confirmation_config_sha = _write_json(
        repo / "configs/qap_confirmation.json", {"kind": "confirmation"}
    )
    confirmation_report_sha = _write_json(
        repo / "runs/qap_report.json", {"status": "confirmed"}
    )
    environment_sha = _write_json(
        repo / "configs/environment.json",
        {
            "fixture_preparation_and_phase_b": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "packages": {
                    "numpy": np.__version__,
                    "opencv": cv2.__version__,
                    "pillow": pillow_version,
                    "kornia": kornia.__version__,
                    "scikit_image": skimage.__version__,
                    "scipy": scipy.__version__,
                    "torch": torch.__version__,
                },
            }
        },
    )

    for name in selected:
        _write(repo / "puzzle/train/targets" / name, f"target:{name}".encode())

    selection = {
        "authoritative_manifest": "configs/manifest.json",
        "authoritative_manifest_sha256": manifest_sha,
        "quarantine": "configs/quarantine.json",
        "quarantine_sha256": quarantine_sha,
        "split": "edge_development",
        "offset": 128,
        "count": source_count,
        "source_names_sha256": hashlib.sha256(
            "\n".join(selected).encode("utf-8")
        ).hexdigest(),
        "source_count_must_equal": source_count,
        "panels_in_label_order": ["primary_kornia", "independent_libjpeg"],
        "total_fixture_records": source_count * 2,
    }
    contract = {
        "protocol_instance_id": "0123456789abcdef0123456789abcdef",
        "source_selection": selection,
        "fixture_preparation": {
            "exact_common_manifest_binding_field_names": [
                "protocol_instance_id",
                "frozen_contract_sha256",
                "evaluator_sha256",
                "tests_sha256",
                "fixture_builder_sha256",
                "fixture_builder_tests_sha256",
                "pin_finalizer_sha256",
                "lifecycle_tool_sha256",
                "result_verifier_sha256",
                "environment_lock_sha256",
                "phase_a_runner_sha256",
                "phase_a_kernel_metadata_sha256",
                "phase_a_launcher_sha256",
                "phase_b_runner_sha256",
            ]
        },
        "runtime_environment": {
            "fixture_preparation_and_phase_b": {"environment": sys.prefix}
        },
        "assets": {
            "denoiser": {
                "path": "assets/denoiser.pt",
                "sha256": denoiser_sha,
            },
            "hbt": {"path": "assets/hbt.pt", "sha256": hbt_sha},
            "known_code_sha256": {"src/dummy.py": code_sha},
            "private_function_contract": {
                "module": "src/puzzle_assembly/components.py",
                "module_sha256": components_sha,
                "required_symbols": [
                    "grow_components",
                    "_place_components_beam",
                    "_complete_with_hungarian",
                ],
            },
        },
        "synthetic_corruption": {"master_seed": 20260711},
        "sealed_sets": {
            "audit_exclusion_ledger": "configs/audit_ledger.json",
            "audit_exclusion_ledger_sha256": ledger_sha,
        },
    }
    config = {
        "schema_version": 1,
        "kind": fixtures.EXPECTED_KIND,
        "protocol_instance_id": contract["protocol_instance_id"],
        "decision_basis": {
            "qap_confirmation_config": "configs/qap_confirmation.json",
            "qap_confirmation_config_sha256": confirmation_config_sha,
            "qap_confirmation_report": "runs/qap_report.json",
            "qap_confirmation_report_sha256": confirmation_report_sha,
        },
        "frozen_contract": contract,
        "frozen_contract_sha256": fixtures._canonical_object_sha256(contract),
        "runtime_pins": {
            "evaluator_path": "scripts/evaluator.py",
            "evaluator_sha256": evaluator_sha,
            "tests_path": "tests/test_evaluator.py",
            "tests_sha256": tests_sha,
            "fixture_builder_path": "scripts/builder.py",
            "fixture_builder_sha256": builder_sha,
            "fixture_builder_tests_path": "tests/test_builder.py",
            "fixture_builder_tests_sha256": builder_tests_sha,
            "pin_finalizer_path": "scripts/finalizer.py",
            "pin_finalizer_sha256": finalizer_sha,
            "lifecycle_tool_path": "scripts/lifecycle.py",
            "lifecycle_tool_sha256": lifecycle_sha,
            "result_verifier_path": "scripts/verifier.py",
            "result_verifier_sha256": verifier_sha,
            "environment_lock_path": "configs/environment.json",
            "environment_lock_sha256": environment_sha,
            "phase_a_runner_path": "jobs/run_phase_a.py",
            "phase_a_runner_sha256": phase_a_runner_sha,
            "phase_a_kernel_metadata_path": "jobs/kernel-metadata.json",
            "phase_a_kernel_metadata_sha256": kernel_metadata_sha,
            "phase_a_launcher_path": "scripts/launch.py",
            "phase_a_launcher_sha256": launcher_sha,
            "phase_b_runner_path": "scripts/run_phase_b.py",
            "phase_b_runner_sha256": phase_b_runner_sha,
            "fixture_input_manifest_relative_path": "fixture_input/fixture_input_manifest.json",
            "fixture_input_manifest_sha256": None,
            "fixture_label_manifest_relative_path": "fixture_label/fixture_label_manifest.json",
            "fixture_label_manifest_sha256": None,
            "fixture_lock_relative_path": "fixture_control/fixture_lock.json",
            "fixture_lock_sha256": None,
            "fixture_prep_marker_relative_path": (
                "fixture_control/FIXTURE_PIXEL_ACCESS_STARTED.json"
            ),
        },
        "runtime_pin_mutation_policy": {
            "transition_ledger_root": "ledger/0123456789abcdef0123456789abcdef",
            "code_pin_fields": [
                {"path_field": "evaluator_path", "sha256_field": "evaluator_sha256"},
                {"path_field": "tests_path", "sha256_field": "tests_sha256"},
                {
                    "path_field": "fixture_builder_path",
                    "sha256_field": "fixture_builder_sha256",
                },
                {
                    "path_field": "fixture_builder_tests_path",
                    "sha256_field": "fixture_builder_tests_sha256",
                },
                {
                    "path_field": "pin_finalizer_path",
                    "sha256_field": "pin_finalizer_sha256",
                },
                {
                    "path_field": "lifecycle_tool_path",
                    "sha256_field": "lifecycle_tool_sha256",
                },
                {
                    "path_field": "result_verifier_path",
                    "sha256_field": "result_verifier_sha256",
                },
                {
                    "path_field": "environment_lock_path",
                    "sha256_field": "environment_lock_sha256",
                },
                {
                    "path_field": "phase_a_runner_path",
                    "sha256_field": "phase_a_runner_sha256",
                },
                {
                    "path_field": "phase_a_kernel_metadata_path",
                    "sha256_field": "phase_a_kernel_metadata_sha256",
                },
                {
                    "path_field": "phase_a_launcher_path",
                    "sha256_field": "phase_a_launcher_sha256",
                },
                {
                    "path_field": "phase_b_runner_path",
                    "sha256_field": "phase_b_runner_sha256",
                },
            ],
            "fixture_pin_fields": [
                {
                    "path_field": "fixture_input_manifest_relative_path",
                    "sha256_field": "fixture_input_manifest_sha256",
                },
                {
                    "path_field": "fixture_label_manifest_relative_path",
                    "sha256_field": "fixture_label_manifest_sha256",
                },
                {
                    "path_field": "fixture_lock_relative_path",
                    "sha256_field": "fixture_lock_sha256",
                },
            ],
        },
    }
    config_path = repo / "configs/protocol.json"
    _write_json(config_path, config)
    _write_code_transition_receipts(repo, config_path, config)
    return repo, config_path, selected


def _fake_loader(calls: list[str], *, must_exist_before_read: tuple[Path, ...] = ()):
    def load(path: Path) -> np.ndarray:
        assert all(required.is_file() for required in must_exist_before_read)
        calls.append(path.name)
        values = np.zeros((480, 480, 3), dtype=np.uint8)
        values[0, 0, 0] = len(calls)
        return values

    return load


def _fake_panel(clean_target: np.ndarray, *, panel: str, seed: int):
    rng = np.random.default_rng(seed)
    slot_to_target = rng.permutation(576).astype(np.int32)
    tiles = np.zeros((576, 20, 20, 3), dtype=np.uint8)
    tiles[:, 0, 0, 0] = np.arange(576, dtype=np.uint16) % 251
    if panel == "independent_libjpeg":
        tiles[:, 0, 0, 1] = 1
    return SimpleNamespace(slot_tiles=tiles, slot_to_target=slot_to_target)


def _run(tmp_path: Path, *, secret: bytes = bytes(range(32))):
    repo, config, selected = _fake_repo(tmp_path)
    calls: list[str] = []
    root = tmp_path / "published"
    prep_path = repo / "ledger/0123456789abcdef0123456789abcdef/PREP.json"
    marker_path = root / "fixture_control/FIXTURE_PIXEL_ACCESS_STARTED.json"
    summary = fixtures.prepare_fixtures(
        config_path=config,
        data_root=repo / "puzzle",
        input_root=root / "fixture_input",
        label_root=root / "fixture_label",
        lock_path=root / "fixture_control/fixture_lock.json",
        marker_path=marker_path,
        lifecycle_ledger_root=repo
        / "ledger/0123456789abcdef0123456789abcdef",
        repo_root=repo,
        image_loader=_fake_loader(
            calls, must_exist_before_read=(prep_path, marker_path)
        ),
        panel_builder=_fake_panel,
        master_secret=secret,
        executing_builder_path=repo / "scripts/builder.py",
    )
    return repo, root, selected, calls, summary


def test_preparation_physically_separates_and_blinds_inputs(tmp_path: Path) -> None:
    repo, root, selected, calls, summary = _run(tmp_path)
    assert calls == selected
    assert summary["record_count"] == 2
    assert summary["status"] == "fixtures_prepared_runtime_pins_required"

    input_manifest_path = root / "fixture_input" / fixtures.INPUT_MANIFEST_NAME
    label_manifest_path = root / "fixture_label" / fixtures.LABEL_MANIFEST_NAME
    input_manifest = json.loads(input_manifest_path.read_text())
    labels = json.loads(label_manifest_path.read_text())
    lock = json.loads((root / "fixture_control/fixture_lock.json").read_text())
    assert input_manifest["kind"] == fixtures.INPUT_KIND
    assert labels["kind"] == fixtures.LABEL_KIND
    assert lock["kind"] == fixtures.LOCK_KIND
    assert summary["fixture_input_manifest_sha256"] == fixtures._sha256_file(
        input_manifest_path
    )
    assert summary["fixture_label_manifest_sha256"] == fixtures._sha256_file(
        label_manifest_path
    )
    assert summary["fixture_lock_sha256"] == fixtures._sha256_file(
        root / "fixture_control/fixture_lock.json"
    )
    assert labels["fixture_input_manifest_sha256"] == summary[
        "fixture_input_manifest_sha256"
    ]
    assert lock["fixture_input_manifest_sha256"] == summary[
        "fixture_input_manifest_sha256"
    ]
    assert lock["fixture_label_manifest_sha256"] == summary[
        "fixture_label_manifest_sha256"
    ]
    prep_path = (
        repo / "ledger/0123456789abcdef0123456789abcdef/PREP.json"
    )
    prep = json.loads(prep_path.read_text())
    assert prep["state"] == "PREP"
    assert prep["predecessor_sha256"] is None
    assert prep_path.read_bytes() == fixtures._canonical_json_bytes(prep)

    serialized_inputs = json.dumps(input_manifest, sort_keys=True)
    serialized_lock = json.dumps(lock, sort_keys=True)
    assert selected[0] not in serialized_inputs
    assert "primary_kornia" not in serialized_inputs
    assert "independent_libjpeg" not in serialized_inputs
    secret_sha256 = labels["master_secret"]["sha256"]
    assert secret_sha256 not in serialized_inputs
    assert secret_sha256 not in serialized_lock
    assert all(
        set(record) == {"opaque_id", "artifact", "arrays"}
        for record in input_manifest["records"]
    )

    assert {record["source_name"] for record in labels["records"]} == set(selected)
    assert {record["panel"] for record in labels["records"]} == {
        "primary_kornia",
        "independent_libjpeg",
    }
    assert os.stat(root / "fixture_label" / fixtures.SECRET_NAME).st_mode & 0o777 == 0o600
    assert os.stat(label_manifest_path).st_mode & 0o777 == 0o600
    assert os.stat(root / "fixture_label").st_mode & 0o777 == 0o700
    assert os.stat(root / "fixture_label/records").st_mode & 0o777 == 0o700

    input_ids = [record["opaque_id"] for record in input_manifest["records"]]
    label_ids = [record["opaque_id"] for record in labels["records"]]
    assert input_ids == label_ids == sorted(input_ids)
    seeds = []
    for record in input_manifest["records"]:
        assert record["arrays"]["qap_seed"]["dtype"] == "uint64"
        assert record["arrays"]["qap_seed"]["shape"] == []
        artifact = root / "fixture_input" / record["artifact"]["path"]
        with np.load(artifact, allow_pickle=False) as payload:
            assert payload["slot_tiles"].shape == (576, 20, 20, 3)
            assert payload["qap_seed"].dtype == np.uint64
            assert payload["qap_seed"].shape == ()
            qap_seed = int(payload["qap_seed"])
        seeds.append(qap_seed)
        assert qap_seed == fixtures._qap_seed(record["opaque_id"])
    assert len(set(seeds)) == 2

    for record in labels["records"]:
        artifact = root / "fixture_label" / record["artifact"]["path"]
        with np.load(artifact, allow_pickle=False) as payload:
            assert sorted(payload.files) == [
                "clean_target_rgb",
                "composed_slot_to_target",
                "opaque_slot_permutation",
            ]
            assert np.array_equal(
                np.sort(payload["composed_slot_to_target"]), np.arange(576)
            )
            assert np.array_equal(
                np.sort(payload["opaque_slot_permutation"]), np.arange(576)
            )


def test_real_builder_output_is_consumed_by_evaluator_loader_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the exact builder->manifest->NPZ loader seam that v1 missed."""

    repo, config_path, selected = _fake_repo(tmp_path, source_count=32)
    calls: list[str] = []
    root = tmp_path / "published_e2e"
    ledger = repo / "ledger/0123456789abcdef0123456789abcdef"
    summary = fixtures.prepare_fixtures(
        config_path=config_path,
        data_root=repo / "puzzle",
        input_root=root / "fixture_input",
        label_root=root / "fixture_label",
        lock_path=root / "fixture_control/fixture_lock.json",
        marker_path=root / "fixture_control/FIXTURE_PIXEL_ACCESS_STARTED.json",
        lifecycle_ledger_root=ledger,
        repo_root=repo,
        image_loader=_fake_loader(
            calls,
            must_exist_before_read=(
                ledger / "PREP.json",
                root / "fixture_control/FIXTURE_PIXEL_ACCESS_STARTED.json",
            ),
        ),
        panel_builder=_fake_panel,
        master_secret=bytes(range(32)),
        executing_builder_path=repo / "scripts/builder.py",
    )
    assert calls == selected
    assert summary["record_count"] == 64

    protocol = json.loads(config_path.read_text())
    protocol["runtime_pins"]["fixture_input_manifest_sha256"] = summary[
        "fixture_input_manifest_sha256"
    ]
    monkeypatch.setattr(
        evaluator,
        "EXPECTED_PROTOCOL_INSTANCE_ID",
        protocol["protocol_instance_id"],
    )
    monkeypatch.setattr(
        evaluator,
        "EXPECTED_FROZEN_CONTRACT_SHA256",
        protocol["frozen_contract_sha256"],
    )
    args = SimpleNamespace(
        fixture_manifest=str(
            root / "fixture_input" / fixtures.INPUT_MANIFEST_NAME
        ),
        fixture_manifest_sha256=summary["fixture_input_manifest_sha256"],
        fixture_root=str(root / "fixture_input"),
    )
    bindings, snapshots, artifact_hashes = evaluator._load_input_fixture_bindings(
        args, protocol
    )
    assert len(bindings) == len(artifact_hashes) == 64
    assert len(snapshots) == 65
    assert all(values["qap_seed"].shape == () for values in bindings.values())
    assert all(values["qap_seed"].dtype == np.uint64 for values in bindings.values())

    manifest = json.loads(
        (root / "fixture_input" / fixtures.INPUT_MANIFEST_NAME).read_text()
    )
    manifest["records"][0]["arrays"]["qap_seed"]["shape"] = [1]
    with pytest.raises(RuntimeError, match="opaque fixture array dtype/shape drift"):
        evaluator._opaque_input_records(manifest, protocol)


def test_missing_runtime_pin_refuses_before_marker_or_pixels(tmp_path: Path) -> None:
    repo, config_path, _ = _fake_repo(tmp_path)
    config = json.loads(config_path.read_text())
    config["runtime_pins"]["evaluator_sha256"] = None
    config_path.write_text(json.dumps(config), encoding="utf-8")
    calls: list[str] = []
    root = tmp_path / "outputs"
    with pytest.raises(RuntimeError, match="runtime pin"):
        fixtures.prepare_fixtures(
            config_path=config_path,
            data_root=repo / "puzzle",
            input_root=root / "inputs",
            label_root=root / "labels",
            lock_path=root / "control/lock.json",
            marker_path=root / "control/marker.json",
            lifecycle_ledger_root=repo
            / "ledger/0123456789abcdef0123456789abcdef",
            repo_root=repo,
            image_loader=_fake_loader(calls),
            panel_builder=_fake_panel,
            master_secret=bytes(32),
            executing_builder_path=repo / "scripts/builder.py",
        )
    assert calls == []
    assert not (root / "control/marker.json").exists()


def test_tampered_evaluator_refuses_before_pixels(tmp_path: Path) -> None:
    repo, config_path, _ = _fake_repo(tmp_path)
    (repo / "scripts/evaluator.py").write_text("tampered", encoding="utf-8")
    calls: list[str] = []
    root = tmp_path / "outputs"
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        fixtures.prepare_fixtures(
            config_path=config_path,
            data_root=repo / "puzzle",
            input_root=root / "inputs",
            label_root=root / "labels",
            lock_path=root / "control/lock.json",
            marker_path=root / "control/marker.json",
            lifecycle_ledger_root=repo
            / "ledger/0123456789abcdef0123456789abcdef",
            repo_root=repo,
            image_loader=_fake_loader(calls),
            panel_builder=_fake_panel,
            master_secret=bytes(32),
            executing_builder_path=repo / "scripts/builder.py",
        )
    assert calls == []
    assert not (root / "control/marker.json").exists()


def test_output_roots_must_match_immutable_protocol_paths(tmp_path: Path) -> None:
    repo, config_path, _ = _fake_repo(tmp_path)
    calls: list[str] = []
    root = tmp_path / "outputs"
    with pytest.raises(
        RuntimeError,
        match="do not share one bundle root|immutable fixture output path mismatch",
    ):
        fixtures.prepare_fixtures(
            config_path=config_path,
            data_root=repo / "puzzle",
            input_root=root / "fixtures",
            label_root=root / "fixtures/labels",
            lock_path=root / "control/lock.json",
            marker_path=root / "control/marker.json",
            lifecycle_ledger_root=repo
            / "ledger/0123456789abcdef0123456789abcdef",
            repo_root=repo,
            image_loader=_fake_loader(calls),
            panel_builder=_fake_panel,
            master_secret=bytes(32),
            executing_builder_path=repo / "scripts/builder.py",
        )
    assert calls == []


def test_opaque_material_is_deterministic_but_panel_unlinkable() -> None:
    secret = bytes(range(32))
    source = "img_000001.png"
    first = fixtures._opaque_id(secret, source, "primary_kornia")
    repeated = fixtures._opaque_id(secret, source, "primary_kornia")
    second_panel = fixtures._opaque_id(secret, source, "independent_libjpeg")
    assert first == repeated
    assert fixtures.OPAQUE_ID_RE.fullmatch(first)
    assert first != second_panel
    assert fixtures._qap_seed(first) != fixtures._qap_seed(second_panel)
    permutation = fixtures._opaque_permutation(secret, source, "primary_kornia")
    assert np.array_equal(np.sort(permutation), np.arange(576))


def test_phase_runners_keep_labels_off_kaggle() -> None:
    metadata_path = (
        fixtures.REPO_ROOT
        / "runs/assembly_v1/kaggle/candidate_graph_oracle_v3_phase_a_job/kernel-metadata.json"
    )
    metadata = json.loads(metadata_path.read_text())
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["enable_internet"] is False
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["dataset_sources"] == [
        "pasha883/vsos-candidate-graph-oracle-v3-code/2",
        "pasha883/vsos-candidate-graph-oracle-v3-inputs/2",
        "pasha883/vsos-candidate-graph-oracle-v3-runtime/2",
    ]
    assert not any(
        token in source.lower()
        for source in metadata["dataset_sources"]
        for token in ("label", "target", "puzzle")
    )
    phase_a_source = (
        fixtures.REPO_ROOT
        / "runs/assembly_v1/kaggle/candidate_graph_oracle_v3_phase_a_job/run_phase_a.py"
    ).read_text()
    assert '"--action",\n        "phase-a"' in phase_a_source
    assert '"--world-size",\n        "2"' in phase_a_source
    assert "labels-manifest" not in phase_a_source
    phase_b_source = (
        fixtures.REPO_ROOT / "scripts/run_candidate_graph_oracle_phase_b.py"
    ).read_text()
    assert 'EXPECTED_PYTHON = REPO_ROOT / ".conda/bin/python"' in phase_b_source
    assert '"--action",\n        "phase-b"' in phase_b_source
    assert "--label-secret" not in phase_b_source
    assert '"--fixture-manifest"' in phase_b_source
    assert "--labels-manifest" not in phase_b_source
    assert "--labels-root" not in phase_b_source
    assert '"--fixture-bundle-root"' in phase_b_source

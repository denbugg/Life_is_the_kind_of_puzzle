from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from scripts import finalize_candidate_graph_oracle_protocol as finalizer


INSTANCE = "0123456789abcdef0123456789abcdef"


def _bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, data: bytes | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.encode("utf-8") if isinstance(data, str) else data)


def _fake_contract() -> dict[str, Any]:
    common = ["protocol_instance_id", "frozen_contract_sha256"] + [
        sha_field for _, sha_field, _ in finalizer.EXPECTED_CODE_PIN_PAIRS
    ]
    return {
        "protocol_instance_id": INSTANCE,
        "immutable": True,
        "fixture_preparation": {
            "exact_common_manifest_binding_field_names": common,
            "exact_crosslink_field_names": {
                "label_to_input": "fixture_input_manifest_sha256",
                "lock_to_input": "fixture_input_manifest_sha256",
                "lock_to_label": "fixture_label_manifest_sha256",
                "lock_to_prep_marker": "prep_marker_sha256",
            },
        },
    }


def _policy() -> dict[str, Any]:
    values: dict[str, Any] = {}
    for key in finalizer.EXPECTED_POLICY_KEYS:
        if key == "top_level_immutable_fields":
            value: Any = list(finalizer.EXPECTED_TOP_LEVEL_IMMUTABLE_FIELDS)
        elif key == "transition_ledger_root":
            value = (
                "runs/assembly_v1/protocol_ledgers/candidate_graph_oracle/"
                + INSTANCE
            )
        elif key == "code_pin_fields":
            value = finalizer._pair_objects(finalizer.EXPECTED_CODE_PIN_PAIRS)
        elif key == "fixture_pin_fields":
            value = finalizer._pair_objects(finalizer.EXPECTED_FIXTURE_PIN_PAIRS)
        elif key in {
            "frozen_contract_is_immutable",
            "protocol_instance_id_is_immutable",
            "runtime_pin_paths_are_immutable",
            "runtime_pins_schema_and_key_order_are_immutable",
            "only_null_sha256_values_may_transition_once_to_lowercase_64_hex",
            "partial_pin_transition_forbidden",
            "all_code_test_runner_kernel_metadata_and_environment_hashes_must_be_pinned_before_fixture_pixel_access",
            "fixture_manifest_and_lock_hashes_may_be_pinned_once_after_fixture_preparation_but_before_phase_a",
            "every_pin_transition_requires_recomputing_and_recording_the_whole_config_file_sha256",
            "no_pin_may_change_after_phase_a_starts",
        }:
            value = True
        else:
            value = f"frozen:{key}"
        values[key] = value
    return values


def _runtime_pins() -> dict[str, Any]:
    pins: dict[str, Any] = {}
    for path_field, sha_field, path in finalizer.EXPECTED_CODE_PIN_PAIRS:
        pins[path_field] = path
        pins[sha_field] = None
    for path_field, sha_field, path in finalizer.EXPECTED_FIXTURE_PIN_PAIRS:
        pins[sha_field] = None
        pins[path_field] = path
    pins["fixture_prep_marker_relative_path"] = finalizer.EXPECTED_PREP_MARKER_PATH
    pins["phase_a_must_not_start_until_all_non_path_values_are_non_null"] = True
    assert tuple(pins) == finalizer.EXPECTED_RUNTIME_PIN_KEYS
    return pins


def _config_payload(contract: dict[str, Any]) -> dict[str, Any]:
    frozen_hash = finalizer._canonical_object_sha256(contract)
    payload = {
        "schema_version": 1,
        "kind": finalizer.PROTOCOL_KIND,
        "status": "semantic_protocol_precommitted_before_fixture_pixel_access",
        "created_utc": "2026-07-12T00:00:00Z",
        "protocol_instance_id": INSTANCE,
        "decision_basis": {"immutable": True},
        "frozen_contract": contract,
        "frozen_contract_sha256": frozen_hash,
        "runtime_pins": _runtime_pins(),
        "runtime_pin_mutation_policy": _policy(),
        "safe_for_submission": False,
    }
    assert tuple(payload) == finalizer.EXPECTED_TOP_LEVEL_KEYS
    return payload


def _fake_repo(tmp_path: Path) -> tuple[Path, Path, str, str]:
    repo = tmp_path / "repo"
    contract = _fake_contract()
    payload = _config_payload(contract)
    config = repo / "configs/candidate_graph_oracle_ceiling_v3.json"
    _write(config, finalizer._canonical_config_bytes(payload))
    for index, (_, _, relative) in enumerate(finalizer.EXPECTED_CODE_PIN_PAIRS):
        _write(repo / relative, f"artifact-{index}\n")
    return repo, config, _sha(config), payload["frozen_contract_sha256"]


def _run_code(repo: Path, config: Path, expected: str, frozen_hash: str) -> dict[str, Any]:
    return finalizer.finalize_runtime_pins(
        config_path=config,
        expected_config_sha256=expected,
        stage="code",
        repo_root=repo,
        expected_protocol_instance_id=INSTANCE,
        expected_frozen_contract_sha256=frozen_hash,
    )


def _common_bindings(config: Path) -> dict[str, str]:
    payload = json.loads(config.read_text(encoding="utf-8"))
    pins = payload["runtime_pins"]
    return {
        "protocol_instance_id": payload["protocol_instance_id"],
        "frozen_contract_sha256": payload["frozen_contract_sha256"],
        **{
            sha_field: pins[sha_field]
            for _, sha_field, _ in finalizer.EXPECTED_CODE_PIN_PAIRS
        },
    }


def _fixture_bundle(repo: Path, config: Path) -> Path:
    bundle = repo / "fixture_bundle"
    common = _common_bindings(config)
    marker = bundle / finalizer.EXPECTED_PREP_MARKER_PATH
    _write(marker, _bytes({"kind": "pixel_access_started", **common}))
    input_payload = {
        "schema_version": 1,
        "kind": "candidate_graph_oracle_fixture_inputs",
        **common,
        "records": [],
    }
    input_path = bundle / finalizer.EXPECTED_FIXTURE_PIN_PAIRS[0][2]
    _write(input_path, _bytes(input_payload))
    input_hash = _sha(input_path)
    label_payload = {
        "schema_version": 1,
        "kind": "candidate_graph_oracle_fixture_labels",
        **common,
        "fixture_input_manifest_sha256": input_hash,
        "records": [],
    }
    label_path = bundle / finalizer.EXPECTED_FIXTURE_PIN_PAIRS[1][2]
    _write(label_path, _bytes(label_payload))
    lock_payload = {
        "schema_version": 1,
        "kind": "candidate_graph_oracle_fixture_lock",
        **common,
        "fixture_input_manifest_sha256": input_hash,
        "fixture_label_manifest_sha256": _sha(label_path),
        "prep_marker_sha256": _sha(marker),
    }
    _write(bundle / finalizer.EXPECTED_FIXTURE_PIN_PAIRS[2][2], _bytes(lock_payload))
    return bundle


def _run_fixtures(
    repo: Path,
    config: Path,
    expected: str,
    frozen_hash: str,
    bundle: Path,
) -> dict[str, Any]:
    return finalizer.finalize_runtime_pins(
        config_path=config,
        expected_config_sha256=expected,
        stage="fixtures",
        fixture_bundle_root=bundle,
        repo_root=repo,
        expected_protocol_instance_id=INSTANCE,
        expected_frozen_contract_sha256=frozen_hash,
    )


def _transition_dir(repo: Path) -> Path:
    return (
        repo
        / "runs/assembly_v1/protocol_ledgers/candidate_graph_oracle"
        / INSTANCE
        / finalizer.TRANSITION_DIR_NAME
    )


def _install_frozen_environment_contract(
    repo: Path, config: Path, *, tamper_numpy: bool = False
) -> tuple[str, str]:
    payload = json.loads(config.read_text(encoding="utf-8"))
    local = {
        "environment": "/pinned/env",
        "python": "3.11.15",
        "torch": "2.12.1",
        "numpy": "2.4.6",
        "scipy": "1.17.1",
        "scikit_image": "0.26.0",
        "pillow": "12.3.0",
        "opencv": "5.0.0",
        "kornia": "0.8.3",
    }
    kaggle = {
        "python": "3.12.13",
        "torch": "2.10.0+cu128",
        "cuda_runtime": "12.8",
        "numpy": "2.0.2",
        "scipy": "1.16.3",
        "scikit_image": "0.25.2",
        "pillow": "11.3.0",
        "opencv": "4.13.0",
        "kornia": "0.8.3",
        "device": "2x Tesla T4 sm_75",
    }
    payload["frozen_contract"]["runtime_environment"] = {
        "fixture_preparation_and_phase_b": local,
        "kaggle_phase_a": kaggle,
    }
    payload["frozen_contract_sha256"] = finalizer._canonical_object_sha256(
        payload["frozen_contract"]
    )
    _write(config, finalizer._canonical_config_bytes(payload))

    package_names = (
        "numpy",
        "opencv",
        "pillow",
        "kornia",
        "scikit_image",
        "scipy",
        "torch",
    )
    local_packages = {name: local[name] for name in package_names}
    if tamper_numpy:
        local_packages["numpy"] = "0.0.0"
    lock = {
        "schema_version": 1,
        "kind": "candidate_graph_oracle_environment_lock",
        "fixture_preparation_and_phase_b": {
            "python": local["python"],
            "packages": local_packages,
            "execution": "repo-owned /pinned/env only",
            "exact_match_required_before_fixture_pixel_access": True,
            "phase_b_runs_in_a_fresh_local_process_with_the_same_exact_environment": True,
        },
        "kaggle_phase_a": {
            "python": kaggle["python"],
            "packages": {name: kaggle[name] for name in package_names},
            "cuda_runtime": kaggle["cuda_runtime"],
            "device_count": 2,
            "devices": [
                {"index": 0, "name": "Tesla T4", "capability": [7, 5]},
                {"index": 1, "name": "Tesla T4", "capability": [7, 5]},
            ],
            "real_tensor_probe_required_on_each_device": True,
            "phase_a_exact_library_match_required": True,
        },
    }
    environment_relative = next(
        relative
        for path_field, _, relative in finalizer.EXPECTED_CODE_PIN_PAIRS
        if path_field == "environment_lock_path"
    )
    _write(repo / environment_relative, _bytes(lock))
    return _sha(config), payload["frozen_contract_sha256"]


def test_code_pin_crosschecks_environment_lock_against_frozen_contract(
    tmp_path: Path,
) -> None:
    repo, config, _, _ = _fake_repo(tmp_path)
    expected, frozen_hash = _install_frozen_environment_contract(repo, config)
    result = _run_code(repo, config, expected, frozen_hash)
    assert result["status"] == "completed"


def test_code_pin_rejects_environment_lock_version_drift(tmp_path: Path) -> None:
    repo, config, _, _ = _fake_repo(tmp_path)
    expected, frozen_hash = _install_frozen_environment_contract(
        repo, config, tamper_numpy=True
    )
    with pytest.raises(RuntimeError, match="contradicts frozen local runtime"):
        _run_code(repo, config, expected, frozen_hash)
    transition = _transition_dir(repo)
    assert transition.is_dir()
    assert list(transition.iterdir()) == []


def test_exact_two_stage_transition_preserves_all_immutable_content(tmp_path: Path) -> None:
    repo, config, initial_sha, frozen_hash = _fake_repo(tmp_path)
    before = json.loads(config.read_text(encoding="utf-8"))

    code = _run_code(repo, config, initial_sha, frozen_hash)
    assert code["status"] == "completed"
    assert code["previous_config_sha256"] == initial_sha
    assert code["final_config_sha256"] == _sha(config)
    after_code = json.loads(config.read_text(encoding="utf-8"))
    for key in finalizer.EXPECTED_TOP_LEVEL_IMMUTABLE_FIELDS:
        assert after_code[key] == before[key]
    assert tuple(after_code["runtime_pins"]) == finalizer.EXPECTED_RUNTIME_PIN_KEYS
    for _, sha_field, _ in finalizer.EXPECTED_CODE_PIN_PAIRS:
        assert finalizer.SHA_RE.fullmatch(after_code["runtime_pins"][sha_field])
    for _, sha_field, _ in finalizer.EXPECTED_FIXTURE_PIN_PAIRS:
        assert after_code["runtime_pins"][sha_field] is None
    assert config.read_bytes() == finalizer._canonical_config_bytes(after_code)

    bundle = _fixture_bundle(repo, config)
    fixture = _run_fixtures(repo, config, _sha(config), frozen_hash, bundle)
    assert fixture["status"] == "completed"
    final = json.loads(config.read_text(encoding="utf-8"))
    for key in finalizer.EXPECTED_TOP_LEVEL_IMMUTABLE_FIELDS:
        assert final[key] == before[key]
    for _, sha_field, _ in (
        *finalizer.EXPECTED_CODE_PIN_PAIRS,
        *finalizer.EXPECTED_FIXTURE_PIN_PAIRS,
    ):
        assert finalizer.SHA_RE.fullmatch(final["runtime_pins"][sha_field])
    assert fixture["final_config_sha256"] == _sha(config)
    assert config.read_bytes() == finalizer._canonical_config_bytes(final)

    transition = _transition_dir(repo)
    assert sorted(path.name for path in transition.iterdir()) == [
        "00_code_pins.complete.json",
        "00_code_pins.intent.json",
        "01_fixtures_pins.complete.json",
        "01_fixtures_pins.intent.json",
    ]
    for path in transition.iterdir():
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert path.read_bytes() == finalizer._canonical_ledger_bytes(payload)


def test_wrong_out_of_band_whole_config_hash_has_no_side_effect(tmp_path: Path) -> None:
    repo, config, initial_sha, frozen_hash = _fake_repo(tmp_path)
    before = config.read_bytes()
    with pytest.raises(RuntimeError, match="out-of-band expectation"):
        _run_code(repo, config, "0" * 64, frozen_hash)
    assert config.read_bytes() == before
    assert not _transition_dir(repo).exists()
    assert initial_sha == _sha(config)


@pytest.mark.parametrize("mutation", ["extra_top", "extra_pin", "path", "pair_order"])
def test_schema_path_or_extra_key_drift_is_rejected(
    tmp_path: Path, mutation: str
) -> None:
    repo, config, _, frozen_hash = _fake_repo(tmp_path)
    payload = json.loads(config.read_text(encoding="utf-8"))
    if mutation == "extra_top":
        payload["extra"] = True
    elif mutation == "extra_pin":
        payload["runtime_pins"]["extra"] = None
    elif mutation == "path":
        payload["runtime_pins"]["evaluator_path"] = "scripts/other.py"
        _write(repo / "scripts/other.py", "other")
    else:
        pairs = payload["runtime_pin_mutation_policy"]["code_pin_fields"]
        pairs[0], pairs[1] = pairs[1], pairs[0]
    _write(config, finalizer._canonical_config_bytes(payload))
    with pytest.raises(RuntimeError, match="drift|extra key"):
        _run_code(repo, config, _sha(config), frozen_hash)
    assert not _transition_dir(repo).exists()


def test_frozen_contract_and_instance_are_verified_outside_config(tmp_path: Path) -> None:
    repo, config, _, frozen_hash = _fake_repo(tmp_path)
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["frozen_contract"]["immutable"] = False
    payload["frozen_contract_sha256"] = finalizer._canonical_object_sha256(
        payload["frozen_contract"]
    )
    _write(config, finalizer._canonical_config_bytes(payload))
    with pytest.raises(RuntimeError, match="immutable expected contract"):
        _run_code(repo, config, _sha(config), frozen_hash)

    payload = _config_payload(_fake_contract())
    payload["protocol_instance_id"] = "f" * 32
    _write(config, finalizer._canonical_config_bytes(payload))
    with pytest.raises(RuntimeError, match="immutable expected value"):
        _run_code(repo, config, _sha(config), frozen_hash)


def test_partial_or_preexisting_pin_without_ledger_is_rejected(tmp_path: Path) -> None:
    repo, config, _, frozen_hash = _fake_repo(tmp_path)
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["runtime_pins"]["evaluator_sha256"] = "a" * 64
    _write(config, finalizer._canonical_config_bytes(payload))
    with pytest.raises(RuntimeError, match="partial code"):
        _run_code(repo, config, _sha(config), frozen_hash)

    payload = _config_payload(_fake_contract())
    for _, sha_field, path in finalizer.EXPECTED_CODE_PIN_PAIRS:
        payload["runtime_pins"][sha_field] = _sha(repo / path)
    _write(config, finalizer._canonical_config_bytes(payload))
    with pytest.raises(RuntimeError, match="no append-only transition intent"):
        _run_code(repo, config, _sha(config), frozen_hash)


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_code_pin_symlink_or_hardlink_is_rejected(
    tmp_path: Path, link_kind: str
) -> None:
    repo, config, initial_sha, frozen_hash = _fake_repo(tmp_path)
    path = repo / finalizer.EXPECTED_CODE_PIN_PAIRS[0][2]
    original = path.with_name("original.py")
    path.rename(original)
    if link_kind == "symlink":
        path.symlink_to(original.name)
        match = "symlink"
    else:
        os.link(original, path)
        match = "st_nlink"
    with pytest.raises(RuntimeError, match=match):
        _run_code(repo, config, initial_sha, frozen_hash)
    transition = _transition_dir(repo)
    assert not transition.exists() or not list(transition.iterdir())


def test_config_as_pinned_file_hash_cycle_is_rejected_by_immutable_path(tmp_path: Path) -> None:
    repo, config, _, frozen_hash = _fake_repo(tmp_path)
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["runtime_pins"]["evaluator_path"] = (
        "configs/candidate_graph_oracle_ceiling_v3.json"
    )
    _write(config, finalizer._canonical_config_bytes(payload))
    with pytest.raises(RuntimeError, match="immutable runtime path drift"):
        _run_code(repo, config, _sha(config), frozen_hash)


def test_crash_after_intent_recovers_without_second_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, config, initial_sha, frozen_hash = _fake_repo(tmp_path)
    original_replace = finalizer._atomic_replace_config

    def crash(*args: Any, **kwargs: Any) -> finalizer.FileSnapshot:
        raise RuntimeError("simulated crash before replace")

    monkeypatch.setattr(finalizer, "_atomic_replace_config", crash)
    with pytest.raises(RuntimeError, match="simulated crash"):
        _run_code(repo, config, initial_sha, frozen_hash)
    transition = _transition_dir(repo)
    intent = transition / "00_code_pins.intent.json"
    assert intent.is_file()
    intent_bytes = intent.read_bytes()
    assert not (transition / "00_code_pins.complete.json").exists()
    assert _sha(config) == initial_sha

    monkeypatch.setattr(finalizer, "_atomic_replace_config", original_replace)
    result = _run_code(repo, config, initial_sha, frozen_hash)
    assert result["status"] == "completed"
    assert intent.read_bytes() == intent_bytes
    assert (transition / "00_code_pins.complete.json").is_file()


def test_crash_after_replace_recovers_only_missing_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, config, initial_sha, frozen_hash = _fake_repo(tmp_path)
    original_completion = finalizer._write_completion

    def crash(*args: Any, **kwargs: Any) -> str:
        raise RuntimeError("simulated crash before completion")

    monkeypatch.setattr(finalizer, "_write_completion", crash)
    with pytest.raises(RuntimeError, match="simulated crash"):
        _run_code(repo, config, initial_sha, frozen_hash)
    after_sha = _sha(config)
    assert after_sha != initial_sha
    transition = _transition_dir(repo)
    assert (transition / "00_code_pins.intent.json").is_file()
    assert not (transition / "00_code_pins.complete.json").exists()

    monkeypatch.setattr(finalizer, "_write_completion", original_completion)
    result = _run_code(repo, config, after_sha, frozen_hash)
    assert result["status"] == "recovered_completion"
    assert result["final_config_sha256"] == after_sha
    again = _run_code(repo, config, after_sha, frozen_hash)
    assert again["status"] == "already_completed"


def test_artifact_tamper_after_intent_is_not_recovered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, config, initial_sha, frozen_hash = _fake_repo(tmp_path)

    def crash(*args: Any, **kwargs: Any) -> finalizer.FileSnapshot:
        raise RuntimeError("stop after intent")

    monkeypatch.setattr(finalizer, "_atomic_replace_config", crash)
    with pytest.raises(RuntimeError, match="stop after intent"):
        _run_code(repo, config, initial_sha, frozen_hash)
    path = repo / finalizer.EXPECTED_CODE_PIN_PAIRS[0][2]
    path.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="differ from the append-only intent"):
        _run_code(repo, config, initial_sha, frozen_hash)
    assert _sha(config) == initial_sha


def test_tampered_or_extra_transition_ledger_key_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, config, initial_sha, frozen_hash = _fake_repo(tmp_path)

    def crash(*args: Any, **kwargs: Any) -> finalizer.FileSnapshot:
        raise RuntimeError("stop after intent")

    monkeypatch.setattr(finalizer, "_atomic_replace_config", crash)
    with pytest.raises(RuntimeError):
        _run_code(repo, config, initial_sha, frozen_hash)
    intent = _transition_dir(repo) / "00_code_pins.intent.json"
    payload = json.loads(intent.read_text(encoding="utf-8"))
    payload["extra"] = True
    intent.write_bytes(_bytes(payload))
    with pytest.raises(RuntimeError, match="schema drift or extra key"):
        _run_code(repo, config, initial_sha, frozen_hash)


def test_fixture_stage_requires_code_completion_and_exact_crosslinks(tmp_path: Path) -> None:
    repo, config, initial_sha, frozen_hash = _fake_repo(tmp_path)
    bundle = repo / "fixture_bundle"
    bundle.mkdir()
    with pytest.raises(RuntimeError, match="completed code-pin transition"):
        _run_fixtures(repo, config, initial_sha, frozen_hash, bundle)

    code = _run_code(repo, config, initial_sha, frozen_hash)
    bundle = _fixture_bundle(repo, config)
    label_path = bundle / finalizer.EXPECTED_FIXTURE_PIN_PAIRS[1][2]
    label = json.loads(label_path.read_text(encoding="utf-8"))
    label["fixture_input_manifest_sha256"] = "f" * 64
    label_path.write_bytes(_bytes(label))
    lock_path = bundle / finalizer.EXPECTED_FIXTURE_PIN_PAIRS[2][2]
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["fixture_label_manifest_sha256"] = _sha(label_path)
    lock_path.write_bytes(_bytes(lock))
    with pytest.raises(RuntimeError, match="does not bind the exact input"):
        _run_fixtures(repo, config, code["final_config_sha256"], frozen_hash, bundle)


@pytest.mark.parametrize("bad_location", ["input", "label", "lock"])
def test_fixture_whole_config_hash_binding_is_rejected(
    tmp_path: Path, bad_location: str
) -> None:
    repo, config, initial_sha, frozen_hash = _fake_repo(tmp_path)
    code = _run_code(repo, config, initial_sha, frozen_hash)
    bundle = _fixture_bundle(repo, config)
    index = {"input": 0, "label": 1, "lock": 2}[bad_location]
    path = bundle / finalizer.EXPECTED_FIXTURE_PIN_PAIRS[index][2]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["final_config_sha256"] = code["final_config_sha256"]
    path.write_bytes(_bytes(payload))
    if bad_location == "input":
        label_path = bundle / finalizer.EXPECTED_FIXTURE_PIN_PAIRS[1][2]
        label = json.loads(label_path.read_text(encoding="utf-8"))
        label["fixture_input_manifest_sha256"] = _sha(path)
        label_path.write_bytes(_bytes(label))
        lock_path = bundle / finalizer.EXPECTED_FIXTURE_PIN_PAIRS[2][2]
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        lock["fixture_input_manifest_sha256"] = _sha(path)
        lock["fixture_label_manifest_sha256"] = _sha(label_path)
        lock_path.write_bytes(_bytes(lock))
    elif bad_location == "label":
        lock_path = bundle / finalizer.EXPECTED_FIXTURE_PIN_PAIRS[2][2]
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        lock["fixture_label_manifest_sha256"] = _sha(path)
        lock_path.write_bytes(_bytes(lock))
    with pytest.raises(RuntimeError, match="whole-config hash field"):
        _run_fixtures(repo, config, code["final_config_sha256"], frozen_hash, bundle)


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_fixture_manifest_symlink_or_hardlink_is_rejected(
    tmp_path: Path, link_kind: str
) -> None:
    repo, config, initial_sha, frozen_hash = _fake_repo(tmp_path)
    code = _run_code(repo, config, initial_sha, frozen_hash)
    bundle = _fixture_bundle(repo, config)
    path = bundle / finalizer.EXPECTED_FIXTURE_PIN_PAIRS[0][2]
    original = path.with_name("original.json")
    path.rename(original)
    if link_kind == "symlink":
        path.symlink_to(original.name)
        match = "symlink"
    else:
        os.link(original, path)
        match = "st_nlink"
    with pytest.raises(RuntimeError, match=match):
        _run_fixtures(repo, config, code["final_config_sha256"], frozen_hash, bundle)


def test_fixture_crosslink_cannot_point_backward_from_input(tmp_path: Path) -> None:
    repo, config, initial_sha, frozen_hash = _fake_repo(tmp_path)
    code = _run_code(repo, config, initial_sha, frozen_hash)
    bundle = _fixture_bundle(repo, config)
    input_path = bundle / finalizer.EXPECTED_FIXTURE_PIN_PAIRS[0][2]
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    payload["fixture_label_manifest_sha256"] = "a" * 64
    input_path.write_bytes(_bytes(payload))
    label_path = bundle / finalizer.EXPECTED_FIXTURE_PIN_PAIRS[1][2]
    label = json.loads(label_path.read_text(encoding="utf-8"))
    label["fixture_input_manifest_sha256"] = _sha(input_path)
    label_path.write_bytes(_bytes(label))
    lock_path = bundle / finalizer.EXPECTED_FIXTURE_PIN_PAIRS[2][2]
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["fixture_input_manifest_sha256"] = _sha(input_path)
    lock["fixture_label_manifest_sha256"] = _sha(label_path)
    lock_path.write_bytes(_bytes(lock))
    with pytest.raises(RuntimeError, match="one-way fixture crosslink"):
        _run_fixtures(repo, config, code["final_config_sha256"], frozen_hash, bundle)


def test_code_stage_never_accepts_fixture_root(tmp_path: Path) -> None:
    repo, config, initial_sha, frozen_hash = _fake_repo(tmp_path)
    with pytest.raises(RuntimeError, match="must not receive"):
        finalizer.finalize_runtime_pins(
            config_path=config,
            expected_config_sha256=initial_sha,
            stage="code",
            fixture_bundle_root=repo / "anything",
            repo_root=repo,
            expected_protocol_instance_id=INSTANCE,
            expected_frozen_contract_sha256=frozen_hash,
        )
    assert not _transition_dir(repo).exists()

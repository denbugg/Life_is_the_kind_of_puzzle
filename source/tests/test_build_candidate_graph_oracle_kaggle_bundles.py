from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import stat
import zipfile

import pytest

from scripts import build_candidate_graph_oracle_kaggle_bundles as bundles


INSTANCE = "0123456789abcdef0123456789abcdef"


def _object_sha(payload: dict) -> str:
    return hashlib.sha256(bundles._canonical_object_bytes(payload)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _write_json(path: Path, payload: dict, *, repository_style: bool = False) -> None:
    if repository_style:
        data = (json.dumps(payload, ensure_ascii=True, indent=2) + "\n").encode()
    else:
        data = bundles._canonical_json_bytes(payload)
    _write(path, data)


def _runner_source() -> bytes:
    lifecycle = ",\n        ".join(repr(item) for item in bundles.LIFECYCLE_MEMBERS)
    return f'''from pathlib import Path
OWNER = "pasha883"
CODE_SLUG = "vsos-candidate-graph-oracle-v3-code"
INPUT_SLUG = "vsos-candidate-graph-oracle-v3-inputs"
RUNTIME_SLUG = "vsos-candidate-graph-oracle-v3-runtime"
CONFIG_RELATIVE = Path("configs/candidate_graph_oracle_ceiling_v3.json")

def _assert_exact_code_mount(root, config):
    expected_hashes = {{CONFIG_RELATIVE.as_posix(): "hash"}}
    lifecycle_files = {{
        {lifecycle}
    }}
    expected_files = set(expected_hashes) | lifecycle_files
    return expected_files
'''.encode()


def _transition_payloads(
    *,
    stage: str,
    index: int,
    frozen_sha: str,
    previous: str,
    final: str,
    pin_values: dict[str, str],
) -> tuple[dict, dict]:
    common = {
        "schema_version": 1,
        "stage": stage,
        "stage_index": index,
        "protocol_instance_id": INSTANCE,
        "frozen_contract_sha256": frozen_sha,
        "config_relative_path": bundles.CONFIG_MEMBER,
        "previous_config_sha256": previous,
        "pin_sha256_values": pin_values,
    }
    intent = {
        **common,
        "kind": "candidate_graph_oracle_runtime_pin_transition_intent",
        "intended_config_sha256": final,
        "created_utc": "2026-07-12T00:00:00Z",
    }
    intent_sha = hashlib.sha256(bundles._canonical_json_bytes(intent)).hexdigest()
    completion = {
        **common,
        "kind": "candidate_graph_oracle_runtime_pin_transition_completion",
        "final_config_sha256": final,
        "intent_sha256": intent_sha,
        "completed_utc": "2026-07-12T00:00:01Z",
    }
    return intent, completion


def _fixture(tmp_path: Path) -> dict[str, Path | str | dict]:
    repo = tmp_path / "repo"
    input_root = tmp_path / "physically_separate" / "fixture_input"
    config_path = repo / bundles.CONFIG_MEMBER
    ledger_relative = f"ledger/{INSTANCE}"
    ledger = repo / ledger_relative

    runner = repo / "jobs/run_phase_a.py"
    helper = repo / "scripts/evaluator.py"
    known = repo / "src/pkg/core.py"
    denoiser = repo / "models/denoiser.pt"
    hbt = repo / "models/hbt.pt"
    _write(runner, _runner_source())
    _write(helper, b"def evaluate():\n    return 1\n")
    _write(known, b"VALUE = 7\n")
    _write(denoiser, b"deterministic-denoiser-checkpoint\x00\x01")
    _write(hbt, b"deterministic-hbt-checkpoint\x02\x03")

    code_pairs = [
        {"path_field": "phase_a_runner_path", "sha256_field": "phase_a_runner_sha256"},
        {"path_field": "evaluator_path", "sha256_field": "evaluator_sha256"},
    ]
    fixture_pairs = [
        {
            "path_field": "fixture_input_manifest_relative_path",
            "sha256_field": "fixture_input_manifest_sha256",
        },
        {
            "path_field": "private_fixture_manifest_relative_path",
            "sha256_field": "private_fixture_manifest_sha256",
        },
        {
            "path_field": "fixture_lock_relative_path",
            "sha256_field": "fixture_lock_sha256",
        },
    ]
    code_hashes = {
        "phase_a_runner_sha256": _file_sha(runner),
        "evaluator_sha256": _file_sha(helper),
    }
    opaque_ids = ["0" * 31 + "1", "0" * 31 + "2"]
    records = []
    for index, opaque_id in enumerate(opaque_ids):
        artifact = input_root / "records" / f"{opaque_id}.npz"
        data = b"opaque-npz-container-bytes-" + bytes([index])
        _write(artifact, data)
        records.append(
            {
                "opaque_id": opaque_id,
                "artifact": {
                    "path": f"records/{opaque_id}.npz",
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                },
                "arrays": {},
            }
        )

    frozen = {
        "source_selection": {"total_fixture_records": 2},
        "assets": {
            "known_code_sha256": {"src/pkg/core.py": _file_sha(known)},
            "denoiser": {
                "path": "models/denoiser.pt",
                "sha256": _file_sha(denoiser),
            },
            "hbt": {"path": "models/hbt.pt", "sha256": _file_sha(hbt)},
        },
    }
    frozen_sha = _object_sha(frozen)
    input_manifest = {
        "schema_version": 1,
        "created_utc": "2026-07-12T00:00:00Z",
        "protocol_instance_id": INSTANCE,
        "frozen_contract_sha256": frozen_sha,
        **code_hashes,
        "record_count": 2,
        "opaque_ids_sha256": hashlib.sha256("\n".join(opaque_ids).encode()).hexdigest(),
        "canonical_record_order": "ascending opaque_id",
        "kind": "candidate_graph_oracle_fixture_inputs",
        "allowed_record_metadata": ["opaque_id", "artifact", "arrays"],
        "records": records,
    }
    input_manifest_path = input_root / bundles.INPUT_MANIFEST_NAME
    _write_json(input_manifest_path, input_manifest)

    pins = {
        "phase_a_runner_path": "jobs/run_phase_a.py",
        "phase_a_runner_sha256": code_hashes["phase_a_runner_sha256"],
        "evaluator_path": "scripts/evaluator.py",
        "evaluator_sha256": code_hashes["evaluator_sha256"],
        "fixture_input_manifest_relative_path": "fixture_input/fixture_input_manifest.json",
        "fixture_input_manifest_sha256": _file_sha(input_manifest_path),
        "private_fixture_manifest_relative_path": "private/manifest.json",
        "private_fixture_manifest_sha256": "b" * 64,
        "fixture_lock_relative_path": "control/lock.json",
        "fixture_lock_sha256": "c" * 64,
        "phase_a_must_not_start_until_all_non_path_values_are_non_null": True,
    }
    policy = {
        "transition_ledger_root": ledger_relative,
        "code_pin_fields": code_pairs,
        "fixture_pin_fields": fixture_pairs,
    }
    config = {
        "schema_version": 1,
        "kind": bundles.PROTOCOL_KIND,
        "protocol_instance_id": INSTANCE,
        "frozen_contract": frozen,
        "frozen_contract_sha256": frozen_sha,
        "runtime_pins": pins,
        "runtime_pin_mutation_policy": policy,
        "safe_for_submission": False,
    }
    pre_fixture = json.loads(json.dumps(config))
    for pair in fixture_pairs:
        pre_fixture["runtime_pins"][pair["sha256_field"]] = None
    prep_config_sha = hashlib.sha256(
        (json.dumps(pre_fixture, ensure_ascii=True, indent=2) + "\n").encode()
    ).hexdigest()
    _write_json(config_path, config, repository_style=True)
    config_sha = _file_sha(config_path)

    code_values = {pair["sha256_field"]: pins[pair["sha256_field"]] for pair in code_pairs}
    fixture_values = {
        pair["sha256_field"]: pins[pair["sha256_field"]] for pair in fixture_pairs
    }
    code_intent, code_complete = _transition_payloads(
        stage="code",
        index=0,
        frozen_sha=frozen_sha,
        previous="a" * 64,
        final=prep_config_sha,
        pin_values=code_values,
    )
    fixture_intent, fixture_complete = _transition_payloads(
        stage="fixtures",
        index=1,
        frozen_sha=frozen_sha,
        previous=prep_config_sha,
        final=config_sha,
        pin_values=fixture_values,
    )
    transition = ledger / bundles.TRANSITION_DIRECTORY
    _write_json(transition / "00_code_pins.intent.json", code_intent)
    _write_json(transition / "00_code_pins.complete.json", code_complete)
    _write_json(transition / "01_fixtures_pins.intent.json", fixture_intent)
    _write_json(transition / "01_fixtures_pins.complete.json", fixture_complete)

    previous = None
    for state, binding in (
        ("PREP", prep_config_sha),
        ("SEALED", config_sha),
        ("PHASE_A", config_sha),
    ):
        claim = {
            "schema_version": 1,
            "kind": "candidate_graph_oracle_lifecycle",
            "protocol_instance_id": INSTANCE,
            "state": state,
            "frozen_contract_sha256": frozen_sha,
            "config_sha256_or_null": binding,
            "predecessor_sha256": previous,
        }
        path = ledger / f"{state}.json"
        _write_json(path, claim)
        previous = _file_sha(path)
    return {
        "repo": repo,
        "input": input_root,
        "config": config_path,
        "ledger": ledger,
        "config_sha": config_sha,
        "pins": pins,
        "policy": policy,
        "known": known,
    }


def _build(fixture: dict, output: Path) -> dict:
    return bundles.build_bundles(
        repo_root=fixture["repo"],
        config_path=fixture["config"],
        lifecycle_ledger_root=fixture["ledger"],
        fixture_input_root=fixture["input"],
        output_root=output,
    )


def test_builds_exact_reproducible_v2_archives_and_canonical_receipt(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    first = tmp_path / "build_one"
    second = tmp_path / "build_two"
    result_one = _build(fixture, first)
    result_two = _build(fixture, second)
    assert result_one["config_sha256"] == fixture["config_sha"]
    assert result_one["dataset_archive_sha256"] == result_two["dataset_archive_sha256"]
    assert (first / bundles.RECEIPT_NAME).read_bytes() == (
        second / bundles.RECEIPT_NAME
    ).read_bytes()

    receipt_raw = (first / bundles.RECEIPT_NAME).read_bytes()
    receipt = json.loads(receipt_raw)
    assert receipt_raw == bundles._canonical_json_bytes(receipt)
    assert receipt["payload_sha256"] == hashlib.sha256(
        bundles._canonical_object_bytes(receipt["payload"])
    ).hexdigest()
    payload = receipt["payload"]
    assert payload["fully_pinned_config_sha256"] == fixture["config_sha"]
    assert payload["lifecycle_terminal_state"] == "PHASE_A"
    assert payload["upload_performed"] is False
    assert payload["input_payload_decoded"] is False

    expected_code = {
        bundles.CONFIG_MEMBER,
        "jobs/run_phase_a.py",
        "scripts/evaluator.py",
        "src/pkg/core.py",
        *bundles.LIFECYCLE_MEMBERS,
    }
    expected_input = {
        bundles.INPUT_MANIFEST_NAME,
        "records/" + "0" * 31 + "1.npz",
        "records/" + "0" * 31 + "2.npz",
    }
    expected_runtime = {"denoiser.pt", "hbt.pt"}
    for key, expected in (
        ("code", expected_code),
        ("input", expected_input),
        ("runtime", expected_runtime),
    ):
        dataset = payload["datasets"][key]
        assert dataset["expected_version"] == 2
        assert dataset["must_remain_private"] is True
        assert dataset["slug"] == bundles.DATASET_BY_KEY[key].slug
        archive_path = first / dataset["archive"]["path"]
        with zipfile.ZipFile(archive_path) as archive:
            assert set(archive.namelist()) == expected
            assert all(info.compress_type == zipfile.ZIP_STORED for info in archive.infolist())
            assert all(info.date_time == bundles.ZIP_TIMESTAMP for info in archive.infolist())
            assert "dataset-metadata.json" not in archive.namelist()
        metadata_path = first / dataset["dataset_metadata"]["path"]
        metadata = json.loads(metadata_path.read_text())
        assert metadata == {
            "id": bundles.DATASET_BY_KEY[key].slug,
            "isPrivate": True,
            "licenses": [{"name": "other"}],
            "title": bundles.DATASET_BY_KEY[key].title,
        }


def test_refuses_before_exact_phase_a_terminal_claim(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    Path(fixture["ledger"], "PHASE_A.json").unlink()
    with pytest.raises(RuntimeError, match="terminate exactly at PHASE_A"):
        _build(fixture, tmp_path / "refused")
    assert not (tmp_path / "refused").exists()


def test_refuses_null_runtime_pin_before_writing(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    config_path = Path(fixture["config"])
    config = json.loads(config_path.read_text())
    config["runtime_pins"]["evaluator_sha256"] = None
    _write_json(config_path, config, repository_style=True)
    with pytest.raises(RuntimeError, match="not a populated"):
        _build(fixture, tmp_path / "refused")
    assert not (tmp_path / "refused").exists()


def test_input_tree_extra_file_is_rejected_without_decoding_npz(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture(tmp_path)
    input_root = Path(fixture["input"])
    _write(input_root / "records/extra.npz", b"not-decoded")
    original = zipfile.ZipFile.open

    def guarded_open(self, name, *args, **kwargs):
        if isinstance(name, str) and name.endswith(".npz") and self.mode == "r":
            raise AssertionError("input arrays must not be decoded/read through ZIP")
        return original(self, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", guarded_open)
    with pytest.raises(RuntimeError, match="exact tree drift"):
        _build(fixture, tmp_path / "refused")


def test_packager_has_no_hidden_fixture_path_argument_and_never_touches_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    sentinel = Path(fixture["input"]).parent / "DO_NOT_TOUCH"
    sentinel.mkdir()
    (sentinel / "manifest.json").write_text("forbidden", encoding="utf-8")
    original_lstat = Path.lstat

    def guarded_lstat(path: Path, *args, **kwargs):
        if "DO_NOT_TOUCH" in path.parts:
            raise AssertionError("non-input fixture sibling was touched")
        return original_lstat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", guarded_lstat)
    _build(fixture, tmp_path / "safe")
    parameters = set(inspect.signature(bundles.build_bundles).parameters)
    assert parameters == {
        "repo_root",
        "config_path",
        "lifecycle_ledger_root",
        "fixture_input_root",
        "output_root",
    }


def test_runner_static_contract_drift_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    runner = Path(fixture["repo"]) / "jobs/run_phase_a.py"
    runner.write_text(runner.read_text().replace(
        'CODE_SLUG = "vsos-candidate-graph-oracle-v3-code"',
        'CODE_SLUG = "wrong-slug"',
    ))
    config_path = Path(fixture["config"])
    config = json.loads(config_path.read_text())
    config["runtime_pins"]["phase_a_runner_sha256"] = _file_sha(runner)
    # This test targets the static runner check, so update all earlier bindings
    # consistently rather than relying on a simpler runtime-pin mismatch.
    input_path = Path(fixture["input"]) / bundles.INPUT_MANIFEST_NAME
    manifest = json.loads(input_path.read_text())
    manifest["phase_a_runner_sha256"] = _file_sha(runner)
    _write_json(input_path, manifest)
    config["runtime_pins"]["fixture_input_manifest_sha256"] = _file_sha(input_path)
    # Reconstructing the transition/config hash cycle here adds no coverage;
    # call the focused static checker on an independently verified snapshot.
    snapshot, _ = bundles._snapshot_regular(runner, member="jobs/run_phase_a.py")
    with pytest.raises(RuntimeError, match="constants differ"):
        bundles._assert_runner_mount_contract(snapshot)

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import update_candidate_graph_oracle_ledger as ledger


INSTANCE = "0123456789abcdef0123456789abcdef"


def _canonical_bytes(payload: dict) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _write_transition_pair(
    root: Path,
    *,
    stage: str,
    stage_index: int,
    frozen_hash: str,
    previous_config_sha256: str,
    final_config_sha256: str,
    pin_values: dict[str, str],
) -> None:
    transition = root / ledger.TRANSITION_DIRECTORY
    transition.mkdir(parents=True, exist_ok=True)
    prefix = f"{stage_index:02d}_{stage}_pins"
    intent = {
        "schema_version": 1,
        "kind": ledger.TRANSITION_INTENT_KIND,
        "stage": stage,
        "stage_index": stage_index,
        "protocol_instance_id": INSTANCE,
        "frozen_contract_sha256": frozen_hash,
        "config_relative_path": "configs/protocol.json",
        "previous_config_sha256": previous_config_sha256,
        "intended_config_sha256": final_config_sha256,
        "pin_sha256_values": pin_values,
        "created_utc": "2026-07-12T00:00:00Z",
    }
    intent_bytes = _canonical_bytes(intent)
    (transition / f"{prefix}.intent.json").write_bytes(intent_bytes)
    completion = {
        "schema_version": 1,
        "kind": ledger.TRANSITION_COMPLETION_KIND,
        "stage": stage,
        "stage_index": stage_index,
        "protocol_instance_id": INSTANCE,
        "frozen_contract_sha256": frozen_hash,
        "config_relative_path": "configs/protocol.json",
        "previous_config_sha256": previous_config_sha256,
        "final_config_sha256": final_config_sha256,
        "pin_sha256_values": pin_values,
        "intent_sha256": hashlib.sha256(intent_bytes).hexdigest(),
        "completed_utc": "2026-07-12T00:00:01Z",
    }
    (transition / f"{prefix}.complete.json").write_bytes(
        _canonical_bytes(completion)
    )


def _write_config(path: Path, *, code_pinned: bool, fixtures_pinned: bool) -> Path:
    contract = {"immutable": True}
    code_sha = "b" * 64 if code_pinned else None
    fixture_sha = "a" * 64 if fixtures_pinned else None
    payload = {
        "schema_version": 1,
        "kind": ledger.PROTOCOL_KIND,
        "protocol_instance_id": INSTANCE,
        "frozen_contract": contract,
        "frozen_contract_sha256": ledger._canonical_object_sha256(contract),
        "runtime_pins": {
            "code_path": "scripts/code.py",
            "code_sha256": code_sha,
            "fixture_path": "fixture/input.json",
            "fixture_sha256": fixture_sha,
        },
        "runtime_pin_mutation_policy": {
            "transition_ledger_root": f"ledger/{INSTANCE}",
            "code_pin_fields": [
                {"path_field": "code_path", "sha256_field": "code_sha256"}
            ],
            "fixture_pin_fields": [
                {
                    "path_field": "fixture_path",
                    "sha256_field": "fixture_sha256",
                }
            ],
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    root = path.parent.parent / "ledger" / INSTANCE
    if code_pinned:
        config_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        transition = root / ledger.TRANSITION_DIRECTORY
        code_completion = transition / "00_code_pins.complete.json"
        if not code_completion.exists():
            _write_transition_pair(
                root,
                stage="code",
                stage_index=0,
                frozen_hash=payload["frozen_contract_sha256"],
                previous_config_sha256="0" * 64,
                final_config_sha256=config_sha256,
                pin_values={"code_sha256": code_sha},
            )
        if fixtures_pinned:
            code_final = json.loads(code_completion.read_text())["final_config_sha256"]
            _write_transition_pair(
                root,
                stage="fixtures",
                stage_index=1,
                frozen_hash=payload["frozen_contract_sha256"],
                previous_config_sha256=code_final,
                final_config_sha256=config_sha256,
                pin_values={"fixture_sha256": fixture_sha},
            )
    return root


def test_exact_irreversible_lifecycle(tmp_path: Path) -> None:
    config = tmp_path / "configs/protocol.json"
    root = _write_config(config, code_pinned=True, fixtures_pinned=False)
    prep = ledger.advance_state(config_path=config, ledger_root=root, state="PREP")
    assert prep["state"] == "PREP"
    with pytest.raises(RuntimeError, match="already been consumed"):
        ledger.advance_state(config_path=config, ledger_root=root, state="PREP")

    _write_config(config, code_pinned=True, fixtures_pinned=True)
    sealed = ledger.advance_state(config_path=config, ledger_root=root, state="SEALED")
    phase_a = ledger.advance_state(config_path=config, ledger_root=root, state="PHASE_A")
    label = ledger.advance_state(
        config_path=config, ledger_root=root, state="LABEL_ACCESS"
    )
    assert [sealed["state"], phase_a["state"], label["state"]] == [
        "SEALED",
        "PHASE_A",
        "LABEL_ACCESS",
    ]
    assert sorted(path.name for path in root.iterdir()) == [
        "LABEL_ACCESS.json",
        "PHASE_A.json",
        "PREP.json",
        "SEALED.json",
        "runtime_pin_transitions",
    ]
    previous = None
    for state in ledger.STATES:
        path = root / f"{state}.json"
        payload = json.loads(path.read_text())
        assert set(payload) == ledger.EXACT_PAYLOAD_KEYS
        assert payload["state"] == state
        assert payload["predecessor_sha256"] == previous
        assert path.read_bytes() == ledger._canonical_bytes(payload)
        previous = ledger._sha256_file(path)


def test_code_pins_required_before_prep(tmp_path: Path) -> None:
    config = tmp_path / "configs/protocol.json"
    root = _write_config(config, code_pinned=False, fixtures_pinned=False)
    with pytest.raises(RuntimeError, match="code/environment/runner pin"):
        ledger.advance_state(config_path=config, ledger_root=root, state="PREP")
    assert not root.exists()


def test_fixture_pins_and_strict_order_required(tmp_path: Path) -> None:
    config = tmp_path / "configs/protocol.json"
    root = _write_config(config, code_pinned=True, fixtures_pinned=False)
    ledger.advance_state(config_path=config, ledger_root=root, state="PREP")
    with pytest.raises(RuntimeError, match="fixture SHA pin"):
        ledger.advance_state(config_path=config, ledger_root=root, state="SEALED")
    _write_config(config, code_pinned=True, fixtures_pinned=True)
    with pytest.raises(RuntimeError, match="lifecycle prefix"):
        ledger.advance_state(config_path=config, ledger_root=root, state="PHASE_A")


def test_final_config_cannot_change_after_sealed(tmp_path: Path) -> None:
    config = tmp_path / "configs/protocol.json"
    root = _write_config(config, code_pinned=True, fixtures_pinned=False)
    ledger.advance_state(config_path=config, ledger_root=root, state="PREP")
    _write_config(config, code_pinned=True, fixtures_pinned=True)
    ledger.advance_state(config_path=config, ledger_root=root, state="SEALED")
    payload = json.loads(config.read_text())
    payload["extra_tamper"] = True
    config.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        RuntimeError,
        match="changed after SEALED|differs from fixture-pin completion",
    ):
        ledger.advance_state(config_path=config, ledger_root=root, state="PHASE_A")


def test_out_of_band_hash_and_root_are_enforced(tmp_path: Path) -> None:
    config = tmp_path / "configs/protocol.json"
    root = _write_config(config, code_pinned=True, fixtures_pinned=False)
    with pytest.raises(RuntimeError, match="out-of-band"):
        ledger.advance_state(
            config_path=config,
            ledger_root=root,
            state="PREP",
            expected_config_sha256="0" * 64,
        )
    with pytest.raises(RuntimeError, match="immutable protocol path"):
        ledger.advance_state(
            config_path=config, ledger_root=tmp_path / "wrong", state="PREP"
        )


def test_partial_fixture_pin_transition_is_rejected(tmp_path: Path) -> None:
    config = tmp_path / "configs/protocol.json"
    root = _write_config(config, code_pinned=True, fixtures_pinned=False)
    payload = json.loads(config.read_text())
    payload["runtime_pins"]["fixture_sha256"] = "a" * 64
    payload["runtime_pins"]["second_fixture_path"] = "fixture/second.json"
    payload["runtime_pins"]["second_fixture_sha256"] = None
    payload["runtime_pin_mutation_policy"]["fixture_pin_fields"].append(
        {
            "path_field": "second_fixture_path",
            "sha256_field": "second_fixture_sha256",
        }
    )
    config.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="partial runtime pin"):
        ledger.advance_state(config_path=config, ledger_root=root, state="PREP")


def test_prep_rejects_config_changed_after_code_pin_completion(tmp_path: Path) -> None:
    config = tmp_path / "configs/protocol.json"
    root = _write_config(config, code_pinned=True, fixtures_pinned=False)
    payload = json.loads(config.read_text())
    payload["runtime_pin_mutation_policy"]["tampered_after_pin"] = True
    config.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="differs from code-pin completion"):
        ledger.advance_state(config_path=config, ledger_root=root, state="PREP")
    assert not (root / "PREP.json").exists()

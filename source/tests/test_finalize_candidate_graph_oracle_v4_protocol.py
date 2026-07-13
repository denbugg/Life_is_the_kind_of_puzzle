from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from scripts import finalize_candidate_graph_oracle_v4_protocol as finalizer


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/candidate_graph_oracle_ceiling_v4.json"


def _copy_placeholder_guard_inputs(root: Path, config: dict) -> Path:
    config_path = root / "configs/candidate_graph_oracle_ceiling_v4.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(config, ensure_ascii=True, indent=2) + "\n")
    pins = config["runtime_pins"]
    for field in (
        "phase_a_kernel_metadata_path",
        "phase_a_runner_path",
        "phase_a_launcher_path",
    ):
        relative = pins[field]
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, target)
    metadata_path = root / pins["phase_a_kernel_metadata_path"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["id_no"] = -1
    metadata["reservation_receipt_sha256"] = None
    metadata["oracle_launch_expectation"]["kernel_id"] = -1
    metadata["oracle_launch_expectation"]["reservation_receipt_sha256"] = None
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2) + "\n")
    for field, id_name in (
        ("phase_a_runner_path", "KERNEL_ID"),
        ("phase_a_launcher_path", "EXPECTED_KERNEL_ID"),
    ):
        path = root / pins[field]
        lines = path.read_text(encoding="utf-8").splitlines()
        lines = [
            f"{id_name} = -1" if line.startswith(f"{id_name} = ") else
            "RESERVATION_RECEIPT_SHA256: str | None = None"
            if line.startswith("RESERVATION_RECEIPT_SHA256: str | None = ")
            else line
            for line in lines
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config_path


def test_v4_finalizer_has_exact_twelve_runtime_code_pins() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    pairs = finalizer.EXPECTED_CODE_PIN_PAIRS
    assert len(pairs) == 12
    assert len(config["runtime_pin_mutation_policy"]["code_pin_fields"]) == 12
    assert pairs[2] == (
        "fixture_builder_path",
        "fixture_builder_sha256",
        "scripts/build_candidate_graph_oracle_v4_fixtures.py",
    )
    assert pairs[5] == (
        "lifecycle_tool_path",
        "lifecycle_tool_sha256",
        "scripts/update_candidate_graph_oracle_v4_ledger.py",
    )
    assert finalizer.EXPECTED_PROTOCOL_INSTANCE_ID == config["protocol_instance_id"]
    assert finalizer.EXPECTED_FROZEN_CONTRACT_SHA256 == config[
        "frozen_contract_sha256"
    ]


def test_v4_ledger_is_self_contained_equivalent_and_has_no_v3_docstring() -> None:
    generic_path = REPO / "scripts/update_candidate_graph_oracle_ledger.py"
    v4_path = REPO / "scripts/update_candidate_graph_oracle_v4_ledger.py"
    generic_tree = ast.parse(generic_path.read_text(encoding="utf-8"))
    v4_tree = ast.parse(v4_path.read_text(encoding="utf-8"))
    assert ast.get_docstring(generic_tree) is not None
    assert ast.get_docstring(v4_tree) is not None
    assert "ceiling_v3.json" in ast.get_docstring(generic_tree)
    assert "v4" in ast.get_docstring(v4_tree).lower()
    assert "v3" not in ast.get_docstring(v4_tree).lower()
    del generic_tree.body[0]
    del v4_tree.body[0]
    assert ast.dump(v4_tree, include_attributes=False) == ast.dump(
        generic_tree, include_attributes=False
    )


def test_code_finalization_refuses_placeholder_before_ledger_creation(
    tmp_path: Path,
) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config_path = _copy_placeholder_guard_inputs(tmp_path, config)
    expected_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    ledger = tmp_path / config["runtime_pin_mutation_policy"]["transition_ledger_root"]
    with pytest.raises(RuntimeError, match="reservation id is unresolved"):
        finalizer.finalize_runtime_pins(
            config_path=config_path,
            expected_config_sha256=expected_sha,
            stage="code",
            repo_root=tmp_path,
        )
    assert not ledger.exists()


def test_finalizer_hash_binds_the_exact_reservation_validator() -> None:
    orchestrator = REPO / finalizer.RESERVATION_ORCHESTRATOR_RELATIVE
    assert hashlib.sha256(orchestrator.read_bytes()).hexdigest() == (
        finalizer.EXPECTED_RESERVATION_ORCHESTRATOR_SHA256
    )
    source = orchestrator.read_text(encoding="utf-8")
    assert "def _validate_existing_receipt(" in source
    assert "reservation journal contains unbound JSON evidence" in source

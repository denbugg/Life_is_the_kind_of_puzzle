#!/usr/bin/env python3
"""Non-accepting full v3 Phase-A diagnostic with the one-key verifier fix."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
from typing import Any, Mapping
import zipfile

from scripts import verify_candidate_graph_oracle_v3_phase_a_composite as composite


DIAGNOSTIC_VERIFIER_SHA256 = (
    "415bd15850172ba7cd0446d40df65732cd03a83bcd247b0a0af987ef467c460e"
)
RETIREMENT_SHA256 = "bc8154087f9a24eadbc6ffe795f0e2113e63fe93d18fa27546ad9b2764725251"
PREPATCH_CLOSURE_SHA256 = (
    "7297bb55686ca689da52f9eb24396609503a395b29ddd6ed2824528df6e34706"
)
VERIFIER_RELATIVE = "scripts/verify_candidate_graph_oracle_result.py"
ORIGINAL_LINE = (
    b'    _require_exact_keys(diagnostics, {"hbt", "softcycle", "qap"}, '
    b'label="phase_a.derivation_diagnostics")\n'
)
PATCHED_LINE = (
    b'    _require_exact_keys(diagnostics, {"hbt_outside_logits", "softcycle", '
    b'"qap"}, label="phase_a.derivation_diagnostics")\n'
)
FORBIDDEN_COMPONENTS = {
    "fixture_label",
    "label",
    "labels",
    "target",
    "targets",
    "fixture_master_secret.bin",
    "master_secret",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _guard_path(path: Path, *, label: str) -> Path:
    _require(path.is_absolute(), f"{label} must be absolute")
    for candidate in (path, path.resolve(strict=True)):
        lowered = {part.lower() for part in candidate.parts}
        _require(
            not lowered.intersection(FORBIDDEN_COMPONENTS),
            f"{label} enters forbidden namespace",
        )
    return path.resolve(strict=True)


def _tree_hashes(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for candidate in root.rglob("*"):
        if candidate.is_dir():
            _require(not candidate.is_symlink(), f"directory symlink: {candidate}")
            continue
        _guard_path(candidate, label=f"tree file below {root.name}")
        relative = candidate.relative_to(root).as_posix()
        result[relative] = composite._sha(candidate)
    return dict(sorted(result.items()))


def _read_snapshot(paths: Mapping[str, Path]) -> dict[str, Any]:
    trees = {
        "diagnostic_repository": _tree_hashes(paths["diagnostic_repository"]),
        "fixture_input_root": _tree_hashes(paths["fixture_input_root"]),
        "phase_a_root": _tree_hashes(paths["phase_a_root"]),
        "ledger_root": _tree_hashes(paths["ledger_root"]),
    }
    files = {
        key: composite._sha(paths[key])
        for key in (
            "snapshot_archive",
            "wrapper",
            "launch_receipt",
            "recovered_verification",
            "recovered_verifier",
            "retirement",
            "prepatch_closure",
        )
    }
    return {"trees": trees, "files": files}


def _verify_diagnostic_repository(root: Path, archive: Path) -> dict[str, Any]:
    _require(
        composite._sha(archive) == composite.SNAPSHOT_ARCHIVE_SHA256,
        "code-v2 archive SHA drift",
    )
    archive_members = composite._snapshot_archive_members(archive)
    with zipfile.ZipFile(archive) as bundle:
        original_verifier = bundle.read(VERIFIER_RELATIVE)
    _require(original_verifier.count(ORIGINAL_LINE) == 1, "original verifier patch site drift")
    expected_patched = original_verifier.replace(ORIGINAL_LINE, PATCHED_LINE)
    actual_patched = composite._read_regular(root / VERIFIER_RELATIVE)
    _require(actual_patched == expected_patched, "diagnostic verifier has non-one-line drift")
    _require(
        hashlib.sha256(actual_patched).hexdigest() == DIAGNOSTIC_VERIFIER_SHA256,
        "diagnostic verifier SHA drift",
    )
    expected = {**archive_members, **composite.SUPPLEMENT_SHA256S}
    expected[VERIFIER_RELATIVE] = DIAGNOSTIC_VERIFIER_SHA256
    observed = _tree_hashes(root)
    _require(observed == dict(sorted(expected.items())), "diagnostic repository closure drift")
    return {
        "archive_sha256": composite.SNAPSHOT_ARCHIVE_SHA256,
        "archive_member_count": len(archive_members),
        "supplement_count": len(composite.SUPPLEMENT_SHA256S),
        "total_file_count": len(expected),
        "original_verifier_sha256": composite.FROZEN_VERIFIER_SHA256,
        "diagnostic_verifier_sha256": DIAGNOSTIC_VERIFIER_SHA256,
        "exact_one_line_patch": True,
        "tree_sha256": hashlib.sha256(composite._canonical_object(observed)).hexdigest(),
    }


def _load_diagnostic_verifier(root: Path) -> Any:
    _require(
        not any(name == "puzzle_assembly" or name.startswith("puzzle_assembly.") for name in sys.modules),
        "puzzle_assembly was preloaded before isolated diagnostic import",
    )
    path = root / VERIFIER_RELATIVE
    name = "_candidate_graph_oracle_v3_diagnostic_verifier_415b"
    spec = importlib.util.spec_from_file_location(name, path)
    _require(spec is not None and spec.loader is not None, "cannot load diagnostic verifier")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _diagnostic_context(verifier: Any, repository: Path) -> Any:
    config_path = repository / "configs/candidate_graph_oracle_ceiling_v3.json"
    raw, _ = verifier._secure_absolute_file(config_path)
    _require(verifier._sha256_bytes(raw) == composite.CONFIG_SHA256, "config SHA drift")
    config = verifier._require_object(
        verifier._parse_json(raw, label="protocol config", canonical_file=False),
        label="protocol config",
    )
    _require(
        config.get("protocol_instance_id") == composite.INSTANCE
        and config.get("frozen_contract_sha256") == composite.FROZEN_CONTRACT_SHA256
        and config.get("safe_for_submission") is False,
        "diagnostic config header drift",
    )
    pins = config["runtime_pins"]
    for pair in config["runtime_pin_mutation_policy"]["code_pin_fields"]:
        relative = pins[pair["path_field"]]
        expected = pins[pair["sha256_field"]]
        actual = composite._sha(repository / relative)
        if pair["sha256_field"] == "result_verifier_sha256":
            _require(
                expected == composite.FROZEN_VERIFIER_SHA256
                and actual == DIAGNOSTIC_VERIFIER_SHA256,
                "diagnostic verifier pin exception drift",
            )
        else:
            _require(actual == expected, f"runtime code pin drift: {relative}")
    context = verifier.ProtocolContext(
        config=config,
        config_path=config_path,
        config_sha256=composite.CONFIG_SHA256,
        repository=repository,
    )
    verifier._verify_frozen_static_bindings(context)
    known = config["frozen_contract"]["assets"]["known_code_sha256"]
    imported: dict[str, str] = {}
    for module_name in (
        "puzzle_assembly",
        "puzzle_assembly.geometry",
        "puzzle_assembly.metrics",
        "puzzle_assembly.panels",
    ):
        loaded = sys.modules.get(module_name)
        _require(loaded is not None and getattr(loaded, "__file__", None), f"missing module {module_name}")
        module_path = Path(loaded.__file__).resolve(strict=True)
        _require(repository == module_path or repository in module_path.parents, f"module escaped repository: {module_name}")
        relative = module_path.relative_to(repository).as_posix()
        _require(relative in known and composite._sha(module_path) == known[relative], f"module hash drift: {module_name}")
        imported[module_name] = relative
    return context, imported


def _load_canonical(path: Path) -> tuple[dict[str, Any], bytes, str]:
    raw = composite._read_regular(path)
    payload = json.loads(raw.decode("utf-8"))
    _require(isinstance(payload, dict), f"not JSON object: {path}")
    _require(raw == composite._canonical_file(payload), f"noncanonical JSON: {path}")
    return payload, raw, hashlib.sha256(raw).hexdigest()


def _verify_transition_pair(
    *, ledger: Path, stage: str, index: int, previous: str, final: str
) -> dict[str, str]:
    prefix = f"{index:02d}_{stage}_pins"
    intent_path = ledger / "runtime_pin_transitions" / f"{prefix}.intent.json"
    complete_path = ledger / "runtime_pin_transitions" / f"{prefix}.complete.json"
    intent, intent_raw, intent_sha = _load_canonical(intent_path)
    complete, _, complete_sha = _load_canonical(complete_path)
    _require(
        set(intent)
        == {
            "schema_version", "kind", "created_utc", "protocol_instance_id",
            "frozen_contract_sha256", "stage", "stage_index", "config_relative_path",
            "previous_config_sha256", "intended_config_sha256", "pin_sha256_values",
        },
        f"transition intent schema drift: {stage}",
    )
    _require(
        set(complete)
        == {
            "schema_version", "kind", "completed_utc", "protocol_instance_id",
            "frozen_contract_sha256", "stage", "stage_index", "config_relative_path",
            "previous_config_sha256", "final_config_sha256", "pin_sha256_values",
            "intent_sha256",
        },
        f"transition completion schema drift: {stage}",
    )
    common = {
        "schema_version": 1,
        "protocol_instance_id": composite.INSTANCE,
        "frozen_contract_sha256": composite.FROZEN_CONTRACT_SHA256,
        "stage": stage,
        "stage_index": index,
        "config_relative_path": "configs/candidate_graph_oracle_ceiling_v3.json",
        "previous_config_sha256": previous,
    }
    for key, value in common.items():
        _require(intent.get(key) == value and complete.get(key) == value, f"transition common drift: {stage}.{key}")
    _require(
        intent["kind"] == "candidate_graph_oracle_runtime_pin_transition_intent"
        and complete["kind"] == "candidate_graph_oracle_runtime_pin_transition_completion"
        and intent["intended_config_sha256"] == final
        and complete["final_config_sha256"] == final
        and intent["pin_sha256_values"] == complete["pin_sha256_values"]
        and complete["intent_sha256"] == intent_sha,
        f"transition crosslink drift: {stage}",
    )
    return {"intent": intent_sha, "complete": complete_sha, "final_config": final}


def _verify_phase_a_only_ledger(ledger: Path, manifest_phase_a_sha: str | None = None) -> dict[str, Any]:
    _require(
        {item.name for item in ledger.iterdir()}
        == {"PREP.json", "SEALED.json", "PHASE_A.json", "runtime_pin_transitions"},
        "ledger top-level closure drift",
    )
    transitions = ledger / "runtime_pin_transitions"
    _require(transitions.is_dir() and not transitions.is_symlink(), "transition directory drift")
    _require(
        {item.name for item in transitions.iterdir()}
        == {
            "00_code_pins.intent.json", "00_code_pins.complete.json",
            "01_fixtures_pins.intent.json", "01_fixtures_pins.complete.json",
        },
        "transition tree closure drift",
    )
    for candidate in ledger.rglob("*"):
        _require("label_access" not in candidate.name.lower(), "LABEL_ACCESS hidden in ledger tree")
    lifecycle: dict[str, tuple[dict[str, Any], bytes, str]] = {
        state: _load_canonical(ledger / f"{state}.json")
        for state in ("PREP", "SEALED", "PHASE_A")
    }
    expected_keys = {
        "schema_version", "kind", "protocol_instance_id", "state",
        "frozen_contract_sha256", "config_sha256_or_null", "predecessor_sha256",
    }
    for state, (payload, _, _) in lifecycle.items():
        _require(set(payload) == expected_keys, f"lifecycle schema drift: {state}")
        _require(
            payload["schema_version"] == 1
            and payload["kind"] == "candidate_graph_oracle_lifecycle"
            and payload["protocol_instance_id"] == composite.INSTANCE
            and payload["frozen_contract_sha256"] == composite.FROZEN_CONTRACT_SHA256
            and payload["state"] == state,
            f"lifecycle header drift: {state}",
        )
    prep_sha = lifecycle["PREP"][2]
    sealed_sha = lifecycle["SEALED"][2]
    phase_sha = lifecycle["PHASE_A"][2]
    code_final = lifecycle["PREP"][0]["config_sha256_or_null"]
    _require(
        lifecycle["PREP"][0]["predecessor_sha256"] is None
        and lifecycle["SEALED"][0]["predecessor_sha256"] == prep_sha
        and lifecycle["PHASE_A"][0]["predecessor_sha256"] == sealed_sha
        and lifecycle["SEALED"][0]["config_sha256_or_null"] == composite.CONFIG_SHA256
        and lifecycle["PHASE_A"][0]["config_sha256_or_null"] == composite.CONFIG_SHA256,
        "lifecycle predecessor/config chain drift",
    )
    code = _verify_transition_pair(
        ledger=ledger, stage="code", index=0,
        previous="788487ad7b9321a0f164915158edeec9c88937da5eb25c5470eae60e0600a22b",
        final=code_final,
    )
    fixtures = _verify_transition_pair(
        ledger=ledger, stage="fixtures", index=1,
        previous=code_final, final=composite.CONFIG_SHA256,
    )
    if manifest_phase_a_sha is not None:
        _require(manifest_phase_a_sha == phase_sha, "Phase-A manifest/lifecycle hash drift")
    hashes = _tree_hashes(ledger)
    return {
        "terminal_state": "PHASE_A",
        "label_access_present": False,
        "lifecycle_sha256s": {"PREP": prep_sha, "SEALED": sealed_sha, "PHASE_A": phase_sha},
        "transitions": {"code": code, "fixtures": fixtures},
        "tree_sha256": hashlib.sha256(composite._canonical_object(hashes)).hexdigest(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    path_names = (
        "diagnostic_repository", "snapshot_archive", "fixture_input_root", "phase_a_root",
        "wrapper", "launch_receipt", "recovered_verification", "recovered_verifier",
        "ledger_root", "retirement", "prepatch_closure",
    )
    paths = {name: _guard_path(getattr(args, name), label=name) for name in path_names}
    _require(args.output.is_absolute() and not args.output.exists(), "output path invalid or exists")
    _require(composite._sha(paths["retirement"]) == RETIREMENT_SHA256, "retirement SHA drift")
    _require(composite._sha(paths["prepatch_closure"]) == PREPATCH_CLOSURE_SHA256, "prepatch closure SHA drift")
    repository = _verify_diagnostic_repository(paths["diagnostic_repository"], paths["snapshot_archive"])
    ledger_before = _verify_phase_a_only_ledger(paths["ledger_root"])
    read_before = _read_snapshot(paths)

    verifier = _load_diagnostic_verifier(paths["diagnostic_repository"])
    context, imported_modules = _diagnostic_context(verifier, paths["diagnostic_repository"])
    input_evidence = verifier.verify_input_fixture(
        context,
        fixture_root=paths["fixture_input_root"],
        expected_manifest_sha256=composite.INPUT_MANIFEST_SHA256,
    )
    phase_a = verifier.verify_phase_a(
        context,
        phase_a_root=paths["phase_a_root"],
        expected_envelope_sha256=composite.PHASE_A_MANIFEST_SHA256,
        shard_anchors=composite.SHARD_SHA256S,
        input_evidence=input_evidence,
    )
    _require(phase_a.kaggle_attestation is None, "normal launch attestation unexpectedly set")
    ledger_bound = _verify_phase_a_only_ledger(
        paths["ledger_root"], manifest_phase_a_sha=phase_a.payload["phase_a_lifecycle_sha256"]
    )
    wrapper = composite._verify_wrapper(
        verifier, context=context, phase_a=phase_a,
        input_evidence=input_evidence, wrapper_path=paths["wrapper"],
    )
    recovered = composite._verify_recovered_evidence(
        recovered_verification_path=paths["recovered_verification"],
        recovered_verifier_path=paths["recovered_verifier"],
        launch_receipt_path=paths["launch_receipt"],
    )
    # Diagnostic equivalent of the frozen postcondition: every read input is
    # rehashed, including all lifecycle/transition files and imported modules.
    read_after = _read_snapshot(paths)
    _require(read_after == read_before, "read-input TOCTOU drift")
    repository_after = _verify_diagnostic_repository(paths["diagnostic_repository"], paths["snapshot_archive"])
    _require(repository_after == repository, "diagnostic repository changed")
    ledger_after = _verify_phase_a_only_ledger(
        paths["ledger_root"], manifest_phase_a_sha=phase_a.payload["phase_a_lifecycle_sha256"]
    )
    _require(ledger_after == ledger_before == ledger_bound, "ledger changed or binding drifted")

    render_count = sum(len(record.manifest["renders"]) for record in phase_a.records.values())
    _require(len(phase_a.records) == 64 and render_count == 192, "full Phase-A coverage drift")
    payload = {
        "schema_version": 1,
        "kind": "candidate_graph_oracle_v3_nonaccepting_schema_fix_diagnostic",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "status": "diagnostic_green_nonaccepting_v3",
        "protocol_instance_id": composite.INSTANCE,
        "v3_disposition": "INVALID_NO_RESULT",
        "v3_retirement_sha256": RETIREMENT_SHA256,
        "accepts_or_rehabilitates_v3": False,
        "diagnostic_patch": {
            "from_key": "hbt",
            "to_key": "hbt_outside_logits",
            "exact_one_line_patch": True,
            "original_verifier_sha256": composite.FROZEN_VERIFIER_SHA256,
            "diagnostic_verifier_sha256": DIAGNOSTIC_VERIFIER_SHA256,
        },
        "repository": repository,
        "imported_snapshot_modules": imported_modules,
        "phase_a": {
            "manifest_sha256": phase_a.envelope_sha256,
            "records": len(phase_a.records),
            "graphs": len(phase_a.records),
            "renders": render_count,
            "all_array_descriptors_verified": True,
            "all_candidate_unions_reconstructed": True,
            "all_render_pixels_reconstructed": True,
        },
        "wrapper": wrapper,
        "recovered_launch": recovered,
        "ledger": ledger_after,
        "pre_post_all_read_input_hashes_equal": True,
        "normal_schema_launch_attestation_claimed": False,
        "accepted_aggregate_metrics": None,
        "continuation_gate_passed": False,
        "label_paths_constructed": False,
        "label_files_opened": False,
        "label_access_performed": False,
        "safe_for_submission": False,
    }
    envelope = {
        "payload": payload,
        "payload_sha256": hashlib.sha256(composite._canonical_object(payload)).hexdigest(),
    }
    output_sha = composite._write_exclusive(args.output, envelope)
    return {
        "status": payload["status"],
        "output": str(args.output),
        "output_sha256": output_sha,
        "payload_sha256": envelope["payload_sha256"],
        "records": 64,
        "renders": 192,
        "accepting_v3": False,
        "labels_constructed_or_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "diagnostic_repository", "snapshot_archive", "fixture_input_root", "phase_a_root",
        "wrapper", "launch_receipt", "recovered_verification", "recovered_verifier",
        "ledger_root", "retirement", "prepatch_closure", "output",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, type=Path, required=True)
    print(json.dumps(run(parser.parse_args()), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()

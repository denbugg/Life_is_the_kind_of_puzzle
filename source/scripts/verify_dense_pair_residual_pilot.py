#!/usr/bin/env python3
"""Verify downloaded dense-pair residual Kaggle pilot artifacts read-only.

The verifier deliberately treats the JSON reports as claims, not evidence.  It
re-hashes every locally downloaded artifact, reconstructs every named source
slice from the authoritative split files, checks the sequential gate state
machine, and (when the real-input gate opened) validates the immutable Phase-A
prediction envelope before accepting the Phase-B target-access event.

Example::

    python scripts/verify_dense_pair_residual_pilot.py \
      runs/assembly_v1/kaggle/dense_pair_residual_pilot_output/v1
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
from typing import Any, Callable


REPORT_NAME = "dense_pair_residual_report.json"
BEST_CHECKPOINT = "dense_pair_residual_best.pt"
LATEST_CHECKPOINT = "dense_pair_residual_latest.pt"
HASHES_NAME = "SHA256SUMS.txt"
WRAPPER_NAME = "dense_pair_residual_pilot_wrapper.json"
PHASE_A_NAME = "FROZEN_INPUT_ONLY_MANIFEST.json"
TARGET_EVENT_NAME = "TARGET_ACCESS_STARTED.json"

ALLOWED_STATUS = {
    "stop_cheap_selection_retrieval",
    "stop_synthetic_transfer_retrieval",
    "stop_synthetic_transfer_qap",
    "stop_original_real_input_gate",
    "continue_candidate_only",
}
GATE_ORDER = [
    "cheap_synthetic_selection_retrieval",
    "synthetic_transfer_holdout_retrieval_QAP",
    "frozen_original_real_input_QAP_SSIM",
]
CHECKSUM_ARTIFACTS = (REPORT_NAME, BEST_CHECKPOINT, LATEST_CHECKPOINT)
SLICE_SPECS = {
    "train": "edge_train",
    "selection": "edge_development",
    "holdout": "assembly_cal",
    "real_gate": "assembly_incremental_gate",
    "final_audit": "assembly_final_audit",
    "confirmation": "assembly_final_audit",
}
PARTITION_RE = re.compile(r"([a-z_]+)\[(\d+):(\d+)\]")
HEX64_RE = re.compile(r"[0-9a-f]{64}")


class VerificationError(RuntimeError):
    """The downloaded artifact contract is incomplete or inconsistent."""


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def names_sha256(names: list[str]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fail(message: str) -> None:
    raise VerificationError(message)


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{label} must be a JSON object")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(f"{label} must be a JSON array")
    return value


def _bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        _fail(f"{label} must be boolean")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{label} must be finite")
    return result


def _hex64(value: Any, label: str) -> str:
    if not isinstance(value, str) or HEX64_RE.fullmatch(value) is None:
        _fail(f"{label} must be a lowercase SHA-256 digest")
    return value


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VerificationError(f"cannot read {label}: {path}") from error
    return _object(value, label)


def _find_single(root: Path, name: str, label: str) -> Path:
    matches = sorted(path for path in root.rglob(name) if path.is_file())
    if len(matches) != 1:
        _fail(f"expected exactly one {label} below {root}, found {matches}")
    return matches[0]


def _find_by_hash(root: Path, name: str, digest: str, label: str) -> Path:
    _hex64(digest, f"{label} claimed hash")
    matches = [
        path
        for path in sorted(root.rglob(name))
        if path.is_file() and sha256(path) == digest
    ]
    if len(matches) != 1:
        _fail(
            f"expected exactly one locally downloaded {label} named {name} "
            f"with hash {digest}, found {matches}"
        )
    return matches[0]


def _parse_sha256s(path: Path, artifact_dir: Path) -> dict[str, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != len(CHECKSUM_ARTIFACTS):
        _fail("SHA256SUMS.txt must contain exactly report, best, and latest")
    parsed: dict[str, str] = {}
    order: list[str] = []
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  ([^/\\]+)", line)
        if match is None or match.group(2) in parsed:
            _fail("SHA256SUMS.txt contains a malformed or duplicate entry")
        digest, name = match.groups()
        parsed[name] = digest
        order.append(name)
    if tuple(order) != CHECKSUM_ARTIFACTS:
        _fail(f"SHA256SUMS.txt order/content mismatch: {order}")
    for name, digest in parsed.items():
        artifact = artifact_dir / name
        if not artifact.is_file():
            _fail(f"SHA256SUMS.txt names a missing artifact: {artifact}")
        actual = sha256(artifact)
        if actual != digest:
            _fail(f"SHA256SUMS mismatch for {name}: {actual} != {digest}")
    return parsed


def _load_protocol(repo_root: Path) -> Callable[..., list[str]]:
    path = repo_root / "src/puzzle_assembly/protocol.py"
    if not path.is_file():
        _fail(f"missing authoritative split implementation: {path}")
    spec = importlib.util.spec_from_file_location("_dense_pair_verify_protocol", path)
    if spec is None or spec.loader is None:
        _fail(f"cannot import authoritative split implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, "source_names_for_split", None)
    if not callable(function):
        _fail("protocol.py does not expose source_names_for_split")
    return function


def _verify_source_slices(
    provenance: dict[str, Any], repo_root: Path
) -> dict[str, Any]:
    if provenance.get("safe_for_submission") is not False:
        _fail("provenance must be safe_for_submission=false")

    files = {
        "manifest": repo_root / "configs/denoise_splits_seed20260710.json",
        "quarantine": repo_root / "configs/denoise_validation_quarantine_v1.json",
        "audit_exclusion": repo_root / "configs/assembly_audit_exclusion_v1.json",
    }
    for label, path in files.items():
        if not path.is_file():
            _fail(f"missing local authoritative {label}: {path}")
        claim = _object(provenance.get(label), f"provenance.{label}")
        claimed_hash = _hex64(claim.get("sha256"), f"provenance.{label}.sha256")
        actual_hash = sha256(path)
        if claimed_hash != actual_hash:
            _fail(
                f"provenance.{label} hash differs from local authoritative file: "
                f"{claimed_hash} != {actual_hash}"
            )

    source_names_for_split = _load_protocol(repo_root)
    split_cache: dict[str, list[str]] = {}
    verified: dict[str, Any] = {}
    sets: dict[str, set[str]] = {}
    for label, expected_split in SLICE_SPECS.items():
        partition = provenance.get(f"{label}_partition")
        names = provenance.get(f"{label}_names")
        digest = provenance.get(f"{label}_names_sha256")
        if not isinstance(partition, str):
            _fail(f"provenance.{label}_partition must be a string")
        match = PARTITION_RE.fullmatch(partition)
        if match is None:
            _fail(f"malformed {label} partition: {partition!r}")
        split, start_text, stop_text = match.groups()
        if split != expected_split:
            _fail(f"{label} partition uses {split}, expected {expected_split}")
        typed_names = _list(names, f"provenance.{label}_names")
        if any(not isinstance(name, str) for name in typed_names):
            _fail(f"provenance.{label}_names contains a non-string")
        typed_names = list(typed_names)
        if len(set(typed_names)) != len(typed_names):
            _fail(f"provenance.{label}_names contains duplicates")
        start, stop = int(start_text), int(stop_text)
        if stop < start or stop - start != len(typed_names):
            _fail(f"{label} partition bounds disagree with its name count")
        if split not in split_cache:
            split_cache[split] = source_names_for_split(
                split,
                manifest_path=files["manifest"],
                quarantine_path=files["quarantine"],
                audit_exclusion_path=files["audit_exclusion"],
            )
        expected_names = split_cache[split][start:stop]
        if typed_names != expected_names or len(expected_names) != stop - start:
            _fail(f"{label} names are not the exact authoritative {partition} slice")
        expected_digest = names_sha256(typed_names)
        if digest != expected_digest:
            _fail(f"{label} source-name SHA-256 mismatch")
        sets[label] = set(typed_names)
        verified[label] = {
            "partition": partition,
            "count": len(typed_names),
            "names_sha256": expected_digest,
        }

    labels = list(sets)
    for index, first in enumerate(labels):
        for second in labels[index + 1 :]:
            overlap = sets[first] & sets[second]
            if overlap:
                _fail(f"source partitions {first} and {second} overlap: {sorted(overlap)[:3]}")

    selection_names = list(provenance["selection_names"])
    quick = _list(
        provenance.get("quick_selection_names"),
        "provenance.quick_selection_names",
    )
    if any(not isinstance(name, str) for name in quick):
        _fail("quick-selection names contain a non-string")
    if quick != selection_names[: len(quick)]:
        _fail("quick-selection names are not an exact selection prefix")
    if provenance.get("quick_selection_names_sha256") != names_sha256(list(quick)):
        _fail("quick-selection source-name SHA-256 mismatch")
    verified["quick_selection"] = {
        "count": len(quick),
        "names_sha256": names_sha256(list(quick)),
    }
    return verified


def _gate(
    value: Any,
    expected_checks: dict[str, bool],
    label: str,
) -> bool:
    gate = _object(value, label)
    checks = _object(gate.get("checks"), f"{label}.checks")
    if set(checks) != set(expected_checks):
        _fail(f"{label} check keys differ from the frozen gate contract")
    for key, expected in expected_checks.items():
        actual = _bool(checks[key], f"{label}.checks.{key}")
        if actual != expected:
            _fail(f"{label}.checks.{key} disagrees with its aggregate")
    passed = _bool(gate.get("passed"), f"{label}.passed")
    if passed != all(expected_checks.values()):
        _fail(f"{label}.passed disagrees with its checks")
    return passed


def _retrieval_pass(split: dict[str, Any], label: str) -> bool:
    retrieval = _object(split.get("retrieval"), f"{label}.retrieval")
    aggregate = _object(retrieval.get("aggregate"), f"{label}.retrieval.aggregate")
    panels = _object(aggregate.get("panels"), f"{label}.retrieval.aggregate.panels")
    if not panels:
        _fail(f"{label} retrieval aggregate has no panels")
    panel_positive = all(
        _number(
            _object(values, f"{label}.panels.{panel}").get(
                "mean_delta_recall_at_1"
            ),
            f"{label}.panels.{panel}.mean_delta_recall_at_1",
        )
        > 0.0
        for panel, values in panels.items()
    )
    interval = _list(
        aggregate.get("bootstrap_95_delta_recall_at_1"),
        f"{label}.bootstrap_95_delta_recall_at_1",
    )
    if len(interval) != 2:
        _fail(f"{label} retrieval bootstrap interval must have two endpoints")
    expected = {
        "mean_recall_at_1_delta_ge_0.01": _number(
            aggregate.get("mean_delta_recall_at_1"), f"{label}.mean_delta_recall_at_1"
        )
        >= 0.01,
        "mean_mrr_delta_ge_0.01": _number(
            aggregate.get("mean_delta_mrr"), f"{label}.mean_delta_mrr"
        )
        >= 0.01,
        "mean_recall_at_32_delta_ge_minus_0.005": _number(
            aggregate.get("mean_delta_recall_at_32"), f"{label}.mean_delta_recall_at_32"
        )
        >= -0.005,
        "bootstrap_recall_at_1_lower_gt_0": _number(
            interval[0], f"{label}.bootstrap_95_delta_recall_at_1[0]"
        )
        > 0.0,
        "every_panel_recall_at_1_positive": panel_positive,
    }
    return _gate(split.get("retrieval_gate"), expected, f"{label}.retrieval_gate")


def _qap_pass(split: dict[str, Any], label: str) -> bool:
    qap = _object(split.get("qap"), f"{label}.qap")
    aggregate = _object(qap.get("aggregate"), f"{label}.qap.aggregate")
    panels = _object(aggregate.get("panels"), f"{label}.qap.aggregate.panels")
    if not panels:
        _fail(f"{label} QAP aggregate has no panels")
    panel_positive = all(
        _number(
            _object(values, f"{label}.panels.{panel}").get("mean_delta_ssim"),
            f"{label}.panels.{panel}.mean_delta_ssim",
        )
        > 0.0
        for panel, values in panels.items()
    )
    interval = _list(
        aggregate.get("bootstrap_95_delta_ssim"),
        f"{label}.bootstrap_95_delta_ssim",
    )
    if len(interval) != 2:
        _fail(f"{label} QAP bootstrap interval must have two endpoints")
    expected = {
        "mean_qap_ssim_delta_ge_0.005": _number(
            aggregate.get("mean_delta_ssim"), f"{label}.mean_delta_ssim"
        )
        >= 0.005,
        "mean_qap_adjacency_delta_ge_0.01": _number(
            aggregate.get("mean_delta_adjacency"), f"{label}.mean_delta_adjacency"
        )
        >= 0.01,
        "bootstrap_qap_ssim_lower_gt_0": _number(
            interval[0], f"{label}.bootstrap_95_delta_ssim[0]"
        )
        > 0.0,
        "every_panel_qap_ssim_positive": panel_positive,
    }
    return _gate(split.get("qap_gate"), expected, f"{label}.qap_gate")


def _verify_split(
    value: Any,
    *,
    label: str,
    expected_split_label: str,
    expected_names: list[str],
    qap_required: bool,
) -> tuple[bool, bool | None]:
    split = _object(value, label)
    if split.get("split") != expected_split_label:
        _fail(f"{label}.split differs from the frozen protocol")
    names = _list(split.get("names"), f"{label}.names")
    if names != expected_names or split.get("names_sha256") != names_sha256(expected_names):
        _fail(f"{label} names/hash differ from provenance")
    if split.get("synthetic_target_files_opened") is not True:
        _fail(f"{label} must attest synthetic target access")
    retrieval_passed = _retrieval_pass(split, label)
    qap_computed = _bool(split.get("qap_metrics_computed"), f"{label}.qap_metrics_computed")
    if qap_computed != qap_required:
        _fail(f"{label} QAP-computed flag violates gate order")
    if qap_required:
        return retrieval_passed, _qap_pass(split, label)
    if "qap" in split or "qap_gate" in split:
        _fail(f"{label} contains QAP evidence before its gate opened")
    return retrieval_passed, None


def _checkpoint_contract(path: Path, label: str) -> dict[str, Any]:
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise VerificationError(f"cannot load {label} checkpoint: {path}") from error
    checkpoint = _object(payload, f"{label} checkpoint")
    if (
        type(checkpoint.get("schema_version")) is not int
        or checkpoint.get("schema_version") != 1
        or checkpoint.get("kind") != "puzzle_dense_pair_residual"
        or checkpoint.get("safe_for_submission") is not False
    ):
        _fail(f"{label} checkpoint violates schema/fail-closed contract")
    metadata = _object(checkpoint.get("metadata"), f"{label}.metadata")
    if metadata.get("safe_for_submission") is not False:
        _fail(f"{label} checkpoint metadata is not fail-closed")
    if not isinstance(checkpoint.get("model_config"), dict):
        _fail(f"{label} checkpoint lacks model_config")
    model_state = checkpoint.get("model_state")
    if not isinstance(model_state, dict) or not model_state:
        _fail(f"{label} checkpoint lacks a non-empty model_state")
    return checkpoint


def _json_normalized(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def _verify_real_phase_a(
    report_dir: Path,
    real_gate: dict[str, Any] | None,
    provenance: dict[str, Any],
    candidate_checkpoint_sha256: str,
) -> dict[str, Any] | None:
    manifests = sorted(report_dir.rglob(PHASE_A_NAME))
    events = sorted(report_dir.rglob(TARGET_EVENT_NAME))
    if real_gate is None:
        if manifests or events:
            _fail("real-input freeze/event artifacts exist although the gate was not opened")
        return None
    if len(manifests) != 1 or len(events) != 1:
        _fail("opened real gate requires exactly one Phase-A manifest and target event")

    names = list(provenance["real_gate_names"])
    names_digest = names_sha256(names)
    if real_gate.get("split") != "frozen_original_real_input_gate":
        _fail("real_gate.split differs from the frozen protocol")
    if real_gate.get("source_names") != names or real_gate.get("source_names_sha256") != names_digest:
        _fail("real-gate names/hash differ from provenance")
    if real_gate.get("target_opened_after_predictions_frozen") is not True:
        _fail("real gate lacks the Phase-A-before-Phase-B attestation")
    claimed_manifest = real_gate.get("phase_a_manifest")
    claimed_event = real_gate.get("target_access_event")
    if (
        not isinstance(claimed_manifest, str)
        or Path(claimed_manifest).name != PHASE_A_NAME
        or not isinstance(claimed_event, str)
        or Path(claimed_event).name != TARGET_EVENT_NAME
    ):
        _fail("real-gate manifest/event path schema is invalid")

    manifest_path = manifests[0]
    if events[0].parent != manifest_path.parent:
        _fail("Phase-A manifest and target event are not colocated")
    manifest_hash = sha256(manifest_path)
    if real_gate.get("phase_a_manifest_sha256") != manifest_hash:
        _fail("real-gate report Phase-A manifest hash mismatch")
    envelope = _load_json(manifest_path, "Phase-A manifest envelope")
    if set(envelope) != {"payload", "payload_sha256"}:
        _fail("Phase-A manifest envelope has unexpected keys")
    payload = _object(envelope.get("payload"), "Phase-A payload")
    payload_hash = canonical_json_sha256(payload)
    if envelope.get("payload_sha256") != payload_hash:
        _fail("Phase-A canonical payload hash mismatch")
    if real_gate.get("phase_a_payload_sha256") != payload_hash:
        _fail("real-gate report Phase-A payload hash mismatch")
    if (
        payload.get("schema_version") != 1
        or payload.get("kind") != "dense_pair_input_only_frozen_predictions"
        or payload.get("split") != "frozen_original_real_input_gate"
        or payload.get("target_files_opened") is not False
    ):
        _fail("Phase-A payload violates schema/target-sealed contract")
    if payload.get("source_names") != names or payload.get("source_names_sha256") != names_digest:
        _fail("Phase-A source names/hash differ from provenance")
    if payload.get("candidate_checkpoint_sha256") != candidate_checkpoint_sha256:
        _fail("Phase-A checkpoint hash differs from the report candidate")

    try:
        import numpy as np
        from PIL import Image
    except ImportError as error:
        raise VerificationError("NumPy and Pillow are required for Phase-A verification") from error

    records = _list(payload.get("records"), "Phase-A records")
    if len(records) != len(names):
        _fail("Phase-A record count differs from the frozen source slice")
    if [record.get("name") for record in records if isinstance(record, dict)] != names:
        _fail("Phase-A record order/names differ from the frozen source slice")
    manifest_records: dict[str, dict[str, Any]] = {}
    for index, raw_record in enumerate(records):
        record = _object(raw_record, f"Phase-A record {index}")
        name = str(record.get("name"))
        manifest_records[name] = record
        _hex64(record.get("input_pixel_sha256"), f"Phase-A {name} input hash")
        expected_seed = int.from_bytes(
            hashlib.sha256(name.encode("utf-8")).digest()[:4], "little"
        ) + 7001
        if record.get("qap_seed") != expected_seed:
            _fail(f"Phase-A {name} QAP seed mismatch")
        for side in ("base", "candidate"):
            layout = record.get(f"{side}_layout")
            if (
                not isinstance(layout, list)
                or len(layout) != 576
                or any(type(value) is not int for value in layout)
                or set(layout) != set(range(576))
            ):
                _fail(f"Phase-A {name} {side} layout is not a 576-slot permutation")
            layout_array = np.asarray(layout, dtype=np.int32)
            layout_hash = hashlib.sha256(layout_array.tobytes()).hexdigest()
            if record.get(f"{side}_layout_sha256") != layout_hash:
                _fail(f"Phase-A {name} {side} layout hash mismatch")

            expected_filename = f"{Path(name).stem}.{side}.png"
            claimed_path = record.get(f"{side}_render")
            if not isinstance(claimed_path, str) or Path(claimed_path).name != expected_filename:
                _fail(f"Phase-A {name} {side} render path/filename mismatch")
            render_path = manifest_path.parent / expected_filename
            if not render_path.is_file():
                _fail(f"missing frozen render: {render_path}")
            render_hash = sha256(render_path)
            if record.get(f"{side}_render_sha256") != render_hash:
                _fail(f"Phase-A {name} {side} render hash mismatch")
            try:
                with Image.open(render_path) as image:
                    if image.mode != "RGB" or image.size != (480, 480):
                        _fail(f"frozen render has wrong mode/shape: {render_path}")
                    image.verify()
            except OSError as error:
                raise VerificationError(f"invalid frozen PNG: {render_path}") from error

    event_path = events[0]
    event_hash = sha256(event_path)
    if real_gate.get("target_access_event_sha256") != event_hash:
        _fail("real-gate report target-event hash mismatch")
    event = _load_json(event_path, "target-access event")
    expected_event = {
        "schema_version": 1,
        "kind": "dense_pair_target_access_event",
        "split": "frozen_original_real_input_gate",
        "phase_a_manifest_sha256": manifest_hash,
        "phase_a_payload_sha256": payload_hash,
        "candidate_checkpoint_sha256": candidate_checkpoint_sha256,
        "source_names_sha256": names_digest,
        "target_access_started": True,
        "target_files_may_have_been_opened": True,
    }
    if event != expected_event:
        _fail("target-access event differs from the immutable Phase-A anchor")

    score_records = _list(real_gate.get("records"), "real_gate.records")
    if len(score_records) != len(names) or [
        record.get("name") for record in score_records if isinstance(record, dict)
    ] != names:
        _fail("real-gate score record order/names differ from Phase A")
    for raw_record in score_records:
        record = _object(raw_record, "real-gate score record")
        frozen = manifest_records[str(record["name"])]
        for side in ("base", "candidate"):
            if record.get(f"{side}_layout_sha256") != frozen.get(
                f"{side}_layout_sha256"
            ):
                _fail("real-gate score record layout hash differs from Phase A")

    aggregate = _object(real_gate.get("aggregate"), "real_gate.aggregate")
    interval = _list(
        aggregate.get("bootstrap_95_delta_ssim"),
        "real_gate.aggregate.bootstrap_95_delta_ssim",
    )
    if len(interval) != 2:
        _fail("real-gate bootstrap interval must have two endpoints")
    checks = {
        "mean_real_ssim_delta_ge_0.005": _number(
            aggregate.get("mean_delta_ssim"), "real_gate.mean_delta_ssim"
        )
        >= 0.005,
        "bootstrap_real_ssim_lower_gt_0": _number(
            interval[0], "real_gate.bootstrap_95_delta_ssim[0]"
        )
        > 0.0,
        "real_ssim_win_rate_ge_0.60": _number(
            aggregate.get("win_rate"), "real_gate.win_rate"
        )
        >= 0.60,
    }
    passed = _gate(real_gate.get("gate"), checks, "real_gate.gate")
    return {
        "manifest_sha256": manifest_hash,
        "payload_sha256": payload_hash,
        "target_event_sha256": event_hash,
        "source_count": len(names),
        "render_count": 2 * len(names),
        "passed": passed,
    }


def verify_report_dir(
    report_path: Path,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    report_dir = report_path.parent
    report = _load_json(report_path, "dense-pair report")
    if (
        type(report.get("schema_version")) is not int
        or report.get("schema_version") != 1
        or report.get("kind") != "dense_all_pairs_residual_pilot_report"
        or report.get("safe_for_submission") is not False
    ):
        _fail("dense-pair report violates schema/fail-closed contract")
    status = report.get("status")
    if status not in ALLOWED_STATUS:
        _fail(f"unsupported dense-pair report status: {status!r}")
    if not isinstance(report.get("model_config"), dict):
        _fail("report lacks model_config")
    if not isinstance(report.get("training"), dict):
        _fail("pilot report lacks training telemetry")

    provenance = _object(report.get("provenance"), "report.provenance")
    if provenance.get("kind") != "dense_all_pairs_residual_pilot":
        _fail("report provenance kind mismatch")
    if "575 valid alternatives" not in str(provenance.get("all_negatives_contract", "")):
        _fail("report lacks the all-575-negative contract")
    gate_contract = _object(provenance.get("gate_contract"), "provenance.gate_contract")
    if gate_contract.get("selection_order") != GATE_ORDER:
        _fail("provenance gate order differs from the frozen sequential contract")
    slices = _verify_source_slices(provenance, repo_root)

    hashes_path = report_dir / HASHES_NAME
    if not hashes_path.is_file():
        _fail(f"missing {HASHES_NAME} beside report")
    sums = _parse_sha256s(hashes_path, report_dir)
    if sums[REPORT_NAME] != sha256(report_path):
        _fail("report digest differs from SHA256SUMS.txt")
    candidate_hash = _hex64(
        report.get("candidate_checkpoint_sha256"),
        "report.candidate_checkpoint_sha256",
    )
    if candidate_hash != sums[BEST_CHECKPOINT]:
        _fail("candidate checkpoint hash is not the covered best checkpoint")

    best = _checkpoint_contract(report_dir / BEST_CHECKPOINT, "best")
    latest = _checkpoint_contract(report_dir / LATEST_CHECKPOINT, "latest")
    if best.get("model_config") != report.get("model_config"):
        _fail("best checkpoint model_config differs from report")
    if _json_normalized(best.get("metadata")) != _json_normalized(
        report.get("checkpoint_metadata")
    ):
        _fail("best checkpoint metadata differs from report checkpoint_metadata")
    for checkpoint_label, checkpoint in (("best", best), ("latest", latest)):
        metadata = _object(checkpoint.get("metadata"), f"{checkpoint_label}.metadata")
        for label in SLICE_SPECS:
            for suffix in ("partition", "names", "names_sha256"):
                key = f"{label}_{suffix}"
                if metadata.get(key) != provenance.get(key):
                    _fail(f"{checkpoint_label} checkpoint provenance differs at {key}")

    opened = _object(report.get("gate_opened"), "report.gate_opened")
    if set(opened) != {
        "synthetic_transfer",
        "original_real_input",
        "true_final_audit",
        "true_confirmation",
    } or any(type(value) is not bool for value in opened.values()):
        _fail("report gate-open map is incomplete or non-boolean")
    if opened["true_final_audit"] or opened["true_confirmation"]:
        _fail("pilot illegally opened a true audit/confirmation target")
    audit_policy = report.get("audit_policy")
    if not isinstance(audit_policy, str) or "remains sealed" not in audit_policy:
        _fail("report does not attest the sealed true audit")

    selection_names = list(provenance["selection_names"])
    selection_passed, _ = _verify_split(
        report.get("selection"),
        label="selection",
        expected_split_label="cheap_selection_edge_development",
        expected_names=selection_names,
        qap_required=False,
    )
    holdout = report.get("holdout")
    real_value = report.get("real_gate")

    holdout_passed: bool | None = None
    qap_passed: bool | None = None
    if holdout is not None:
        holdout_object = _object(holdout, "holdout")
        qap_required = bool(holdout_object.get("qap_metrics_computed"))
        holdout_passed, qap_passed = _verify_split(
            holdout_object,
            label="holdout",
            expected_split_label="synthetic_transfer_assembly_cal",
            expected_names=list(provenance["holdout_names"]),
            qap_required=qap_required,
        )
        if qap_required and not holdout_passed:
            _fail("holdout QAP opened although holdout retrieval failed")

    real_object = None if real_value is None else _object(real_value, "real_gate")
    phase_a = _verify_real_phase_a(
        report_dir, real_object, provenance, candidate_hash
    )
    real_passed = None if phase_a is None else bool(phase_a["passed"])

    expected_state = {
        "stop_cheap_selection_retrieval": (False, None, None, None),
        "stop_synthetic_transfer_retrieval": (True, False, None, None),
        "stop_synthetic_transfer_qap": (True, True, False, None),
        "stop_original_real_input_gate": (True, True, True, False),
        "continue_candidate_only": (True, True, True, True),
    }[str(status)]
    actual_state = (selection_passed, holdout_passed, qap_passed, real_passed)
    if actual_state != expected_state:
        _fail(
            f"report status {status} disagrees with sequential gates: "
            f"{actual_state} != {expected_state}"
        )
    expected_opened = {
        "synthetic_transfer": holdout is not None,
        "original_real_input": real_object is not None,
        "true_final_audit": False,
        "true_confirmation": False,
    }
    if opened != expected_opened:
        _fail("gate-open map disagrees with materialized gate evidence")

    audit_names = set(provenance["final_audit_names"]) | set(
        provenance["confirmation_names"]
    )
    opened_names = set(selection_names)
    if holdout is not None:
        opened_names.update(provenance["holdout_names"])
    if real_object is not None:
        opened_names.update(provenance["real_gate_names"])
    if audit_names & opened_names:
        _fail("sealed audit names leaked into an opened evaluation stage")

    return {
        "path": str(report_path),
        "sha256": sums[REPORT_NAME],
        "status": status,
        "safe_for_submission": False,
        "checkpoint_sha256": candidate_hash,
        "sha256s_sha256": sha256(hashes_path),
        "source_slices": slices,
        "gate_opened": opened,
        "phase_a": phase_a,
        "audit_unopened": True,
    }


def _verify_wrapper_step_log(root: Path, step: dict[str, Any], label: str) -> str:
    if step.get("label") != label:
        _fail(f"wrapper step label mismatch: expected {label!r}")
    if type(step.get("returncode")) is not int or step.get("returncode") != 0:
        _fail(f"wrapper step {label!r} did not return zero")
    if step.get("timed_out") is not False:
        _fail(f"wrapper step {label!r} timed out")
    command = step.get("command")
    if not isinstance(command, list) or not command or any(
        not isinstance(value, str) for value in command
    ):
        _fail(f"wrapper step {label!r} has malformed command provenance")
    claimed_log = step.get("log")
    if not isinstance(claimed_log, str):
        _fail(f"wrapper step {label!r} lacks a log path")
    digest = _hex64(step.get("log_sha256"), f"wrapper step {label!r} log hash")
    log = _find_by_hash(root, Path(claimed_log).name, digest, f"{label} log")
    return str(log)


def _verify_wrapper_staging(wrapper: dict[str, Any], repo_root: Path) -> None:
    """Validate the durable staging/preflight portion of the wrapper schema."""

    _object(wrapper.get("base"), "wrapper.base")
    base_hashes = _object(wrapper.get("base_hashes"), "wrapper.base_hashes")
    if not base_hashes:
        _fail("wrapper.base_hashes must be non-empty")
    for name, digest in base_hashes.items():
        if not isinstance(name, str):
            _fail("wrapper.base_hashes contains a non-string path")
        _hex64(digest, f"wrapper.base_hashes.{name}")
    overlay = _object(wrapper.get("overlay"), "wrapper.overlay")
    staged_hashes = _object(overlay.get("staged_hashes"), "wrapper.overlay.staged_hashes")
    if not staged_hashes:
        _fail("wrapper overlay lacks staged hashes")
    for name, digest in staged_hashes.items():
        if not isinstance(name, str):
            _fail("wrapper overlay contains a non-string path")
        _hex64(digest, f"wrapper.overlay.staged_hashes.{name}")
    if not isinstance(wrapper.get("data_root"), str):
        _fail("wrapper.data_root must be a path string")

    assets = _object(wrapper.get("assets"), "wrapper.assets")
    expected_assets = {
        "denoiser",
        "hbt",
        "manifest",
        "quarantine",
        "audit_exclusion",
    }
    if set(assets) != expected_assets:
        _fail("wrapper.assets differs from the frozen five-asset contract")
    local_configs = {
        "manifest": repo_root / "configs/denoise_splits_seed20260710.json",
        "quarantine": repo_root / "configs/denoise_validation_quarantine_v1.json",
        "audit_exclusion": repo_root / "configs/assembly_audit_exclusion_v1.json",
    }
    for label in sorted(expected_assets):
        asset = _object(assets[label], f"wrapper.assets.{label}")
        if not isinstance(asset.get("path"), str):
            _fail(f"wrapper.assets.{label}.path must be a string")
        digest = _hex64(asset.get("sha256"), f"wrapper.assets.{label}.sha256")
        if label in local_configs and digest != sha256(local_configs[label]):
            _fail(f"wrapper {label} hash differs from the local authoritative config")

    hardware = _object(wrapper.get("hardware"), "wrapper.hardware")
    if hardware.get("device_count") != 2:
        _fail("wrapper hardware must attest exactly two GPUs")
    devices = _list(hardware.get("devices"), "wrapper.hardware.devices")
    if len(devices) != 2:
        _fail("wrapper hardware must contain exactly two device records")
    for index, value in enumerate(devices):
        device = _object(value, f"wrapper.hardware.devices[{index}]")
        if device.get("index") != index:
            _fail("wrapper hardware device indices are not exact")
        if not isinstance(device.get("name"), str) or "T4" not in device["name"].upper():
            _fail("wrapper hardware is not the precommitted 2xT4 environment")
        capability = _list(
            device.get("capability"), f"wrapper.hardware.devices[{index}].capability"
        )
        if capability != [7, 5]:
            _fail("wrapper T4 compute capability must be exactly 7.5")
        if type(device.get("total_memory")) is not int or device["total_memory"] <= 0:
            _fail("wrapper hardware total-memory record is invalid")
        _number(device.get("tensor_probe"), f"wrapper.hardware.devices[{index}].tensor_probe")


def verify_dense_pair_residual_pilot(
    output_root: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Verify a complete downloaded wrapper, smoke, and bounded pilot tree."""

    root = Path(output_root).resolve()
    if not root.is_dir():
        _fail(f"output root is not a directory: {root}")
    repository = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[1]
    )
    wrapper_path = _find_single(root, WRAPPER_NAME, "Kaggle wrapper")
    wrapper = _load_json(wrapper_path, "Kaggle wrapper")
    if (
        type(wrapper.get("schema_version")) is not int
        or wrapper.get("schema_version") != 1
        or wrapper.get("kind") != "dense_pair_residual_kaggle_wrapper"
        or wrapper.get("status") != "complete"
        or wrapper.get("safe_for_submission") is not False
    ):
        _fail("Kaggle wrapper violates complete/fail-closed schema")
    _number(wrapper.get("started_unix"), "wrapper.started_unix")
    _number(wrapper.get("completed_unix"), "wrapper.completed_unix")
    if float(wrapper["completed_unix"]) < float(wrapper["started_unix"]):
        _fail("wrapper completion time precedes start time")
    _verify_wrapper_staging(wrapper, repository)

    steps = _list(wrapper.get("steps"), "wrapper.steps")
    labels = [
        "dense-pair unit tests",
        "2xT4 full-model one-step smoke",
        "bounded dense-pair residual pilot",
    ]
    if len(steps) != len(labels):
        _fail("complete wrapper must contain exactly tests, smoke, and pilot steps")
    logs: dict[str, str] = {}
    reports: dict[str, dict[str, Any]] = {}
    for index, label in enumerate(labels):
        step = _object(steps[index], f"wrapper.steps[{index}]")
        logs[label] = _verify_wrapper_step_log(root, step, label)
        if index == 0:
            if "report" in step:
                _fail("unit-test wrapper step unexpectedly contains a pilot report")
            continue
        report_ref = _object(step.get("report"), f"wrapper {label} report reference")
        claimed_report_path = report_ref.get("path")
        if (
            not isinstance(claimed_report_path, str)
            or Path(claimed_report_path).name != REPORT_NAME
        ):
            _fail(f"wrapper {label} report path schema is invalid")
        report_hash = _hex64(report_ref.get("sha256"), f"wrapper {label} report hash")
        report_path = _find_by_hash(root, REPORT_NAME, report_hash, f"{label} report")
        verified = verify_report_dir(report_path, repo_root=repository)
        if report_ref.get("status") != verified["status"]:
            _fail(f"wrapper {label} report status disagrees with the report")
        if report_ref.get("gate_opened") != verified["gate_opened"]:
            _fail(f"wrapper {label} gate map disagrees with the report")
        reports["smoke" if index == 1 else "pilot"] = verified

    pilot_ref = _object(wrapper.get("pilot_report"), "wrapper.pilot_report")
    if (
        not isinstance(pilot_ref.get("path"), str)
        or Path(pilot_ref["path"]).name != REPORT_NAME
    ):
        _fail("wrapper.pilot_report path schema is invalid")
    pilot = reports["pilot"]
    if (
        pilot_ref.get("sha256") != pilot["sha256"]
        or pilot_ref.get("status") != pilot["status"]
        or pilot_ref.get("gate_opened") != pilot["gate_opened"]
    ):
        _fail("wrapper.pilot_report differs from the verified pilot step")

    return {
        "schema_version": 1,
        "kind": "dense_pair_residual_download_verification",
        "output_root": str(root),
        "wrapper": {"path": str(wrapper_path), "sha256": sha256(wrapper_path)},
        "reports": reports,
        "logs": logs,
        "safe_for_submission": False,
        "verified": True,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", help="downloaded Kaggle output directory")
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="repository containing authoritative configs and protocol.py",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = verify_dense_pair_residual_pilot(
        args.output_root, repo_root=args.repo_root
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Leakage-safe oracle-ceiling diagnostic for the frozen tile candidate graph.

The program has deliberately separate input-only and label-reading phases.
``phase-a`` only accepts opaque corrupted slot tiles plus a non-identifying
nuisance seed, derives and freezes a sparse candidate graph in two GPU shards,
and ``finalize-phase-a`` merges them without labels.  ``phase-b`` first revalidates that complete frozen
envelope, durably writes ``TARGET_ACCESS_STARTED.json``, and only then is
allowed to construct a label path.  The labels are synthetic known-permutation
panels; this diagnostic never opens real puzzle targets.

The gate-driving oracle is intentionally bounded.  Ground truth may only say
which already-proposed candidate edges are true.  It may not provide absolute
tile positions to the beam/Hungarian/QAP packer.  A separate translation
ceiling is reported for diagnosis, is visibly labelled target-assisted, never
places singleton components with truth, and is never used by a continuation
gate.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from io import BytesIO
import hmac
import hashlib
import json
import os
from pathlib import Path
import platform as platform_module
import re
import stat
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
FROZEN_SRC_ROOT = (
    REPO_ROOT
    / "runs/assembly_v1/kaggle/candidate_graph_oracle_v4_source_snapshot/src"
)
if str(FROZEN_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(FROZEN_SRC_ROOT))

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from skimage.metrics import structural_similarity

from puzzle_assembly.compatibility import (
    CompatibilityMatrices,
    build_classical_score_bank,
    fuse_ranked_scores,
)
from puzzle_assembly.components import (
    ProposedEdge,
    _complete_with_hungarian,
    _place_components_beam,
    grow_components,
    soft_cycle_component_solver,
)
from puzzle_assembly.geometry import (
    GRID,
    TILE,
    TILE_COUNT,
    inverse_permutation,
    true_neighbour_slots,
    validate_permutation,
)
from puzzle_assembly.metrics import layout_metrics
from puzzle_assembly.learned import learned_compatibility, load_embedding_checkpoint
from puzzle_assembly.panels import make_exact_panel
from puzzle_assembly.protocol import per_source_seed, source_names_for_split
from puzzle_assembly.qap import directional_qap
from puzzle_assembly.solvers import placement_unary
from puzzle_denoise_v2.inference import load_restorer, restore_tiles_uint8
from puzzle_denoise_v2.tiles import merge_tiles_numpy


# Updated only together with the reviewed immutable protocol.  Keeping this a
# literal makes a stale evaluator fail before it can consume any fixture.
EXPECTED_FROZEN_CONTRACT_SHA256 = (
    "2070c1b4ff0a3ff42c5ffdd6d611c214c02dbb99b2c51a88f38a862bb1f8a05c"
)
EXPECTED_PROTOCOL_INSTANCE_ID = "6c0fe4e8524ce39d830d9a5bee118d8b"
EXPECTED_NAMES_SHA256 = (
    "149ca83873e5e2e79e6458098c5c758b935af5d9131e093f5eb34fef82b76634"
)
PANELS = ("primary_kornia", "independent_libjpeg")
FIXTURE_MANIFEST = "INPUT_ONLY_FIXTURE_MANIFEST.json"
PHASE_A_MANIFEST = "FROZEN_CANDIDATE_GRAPH_MANIFEST.json"
PHASE_A_SHARD_MANIFEST = "FROZEN_CANDIDATE_GRAPH_SHARD_MANIFEST.json"
TARGET_MARKER = "TARGET_ACCESS_STARTED.json"
REPORT_NAME = "candidate_graph_oracle_ceiling_report.json"

ORIGIN_BITS: dict[str, int] = {
    "c1_out32": 1,
    "hbt_out32": 2,
    "c1_in8": 4,
    "hbt_in8": 8,
    "softcycle": 16,
    "qap_w4": 32,
    "qap_w1": 64,
}
ALL_ORIGINS = sum(ORIGIN_BITS.values())
EXPECTED_PRE_DEDUP_COUNTS = {
    "c1_out32": 36864,
    "hbt_out32": 36864,
    "c1_in8": 9216,
    "hbt_in8": 9216,
    "softcycle": 1104,
    "qap_w4": 1104,
    "qap_w1": 1104,
}
FORBIDDEN_INPUT_COMPONENTS = {"target", "targets", "label", "labels"}
REQUIRED_DERIVED_ARRAYS = {
    "c1_right",
    "c1_down",
    "hbt_right",
    "hbt_down",
    "w1_right",
    "w1_down",
    "w4_right",
    "w4_down",
    "softcycle_layout",
    "qap_w4_layout",
    "qap_w1_layout",
    "denoised_tiles",
}
REQUIRED_INPUT_ARRAYS = {"slot_tiles", "qap_seed"}
REQUIRED_LABEL_ARRAYS = {
    "opaque_slot_permutation",
    "composed_slot_to_target",
    "clean_target_rgb",
}
_SAFE_ID = re.compile(r"[A-Za-z0-9_.-]+\Z")
COMMON_MANIFEST_SHA_FIELDS = (
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
)
FROZEN_RECORD_KEYS = {
    "opaque_id",
    "qap_seed",
    "input_fixture_sha256",
    "input_slot_tiles_c_sha256",
    "graph_artifact",
    "graph_artifact_byte_size",
    "graph_artifact_sha256",
    "candidate_edge_count",
    "candidate_origin_mask_sha256",
    "origin_pre_dedup_counts",
    "arrays",
    "renders",
    "derivation_diagnostics",
}


@dataclass(frozen=True)
class CandidateGraph:
    direction: np.ndarray
    source: np.ndarray
    destination: np.ndarray
    origin_mask: np.ndarray
    c1_cost: np.ndarray
    hbt_cost: np.ndarray
    w1_cost: np.ndarray
    w4_cost: np.ndarray

    def __post_init__(self) -> None:
        arrays = (
            np.asarray(self.direction),
            np.asarray(self.source),
            np.asarray(self.destination),
            np.asarray(self.origin_mask),
            np.asarray(self.c1_cost),
            np.asarray(self.hbt_cost),
            np.asarray(self.w1_cost),
            np.asarray(self.w4_cost),
        )
        if any(value.ndim != 1 for value in arrays):
            raise ValueError("candidate graph arrays must be one-dimensional")
        if len({len(value) for value in arrays}) != 1:
            raise ValueError("candidate graph arrays have different lengths")
        if np.any((arrays[0] < 0) | (arrays[0] > 1)):
            raise ValueError("candidate direction must be right=0 or down=1")
        if np.any((arrays[1] < 0) | (arrays[1] >= TILE_COUNT)):
            raise ValueError("candidate source is outside tile range")
        if np.any((arrays[2] < 0) | (arrays[2] >= TILE_COUNT)):
            raise ValueError("candidate destination is outside tile range")
        if np.any(arrays[1] == arrays[2]):
            raise ValueError("candidate self edges are forbidden")
        if np.any((arrays[3] <= 0) | ((arrays[3].astype(np.int64) & ~ALL_ORIGINS) != 0)):
            raise ValueError("candidate origin mask is invalid")
        for index, name in zip(range(4, 8), ("c1", "hbt", "w1", "w4"), strict=True):
            if arrays[index].dtype != np.float32 or not np.all(np.isfinite(arrays[index])):
                raise ValueError(f"candidate {name} costs must be finite float32")
        keys = np.stack([arrays[0], arrays[1], arrays[2]], axis=1).astype(np.int64)
        if len(np.unique(keys, axis=0)) != len(keys):
            raise ValueError("candidate graph contains duplicate directed edges")
        order = np.lexsort((arrays[2], arrays[1], arrays[0]))
        if not np.array_equal(order, np.arange(len(order))):
            raise ValueError("candidate graph is not in canonical order")


@dataclass(frozen=True)
class OracleComponent:
    members: tuple[int, ...]
    coordinates: tuple[tuple[int, int], ...]

    def as_dict(self) -> dict[int, tuple[int, int]]:
        return dict(zip(self.members, self.coordinates, strict=True))


@dataclass(frozen=True)
class BeamState:
    occupancy: np.ndarray
    score: float
    split_tiles: tuple[int, ...]


@dataclass(frozen=True)
class DerivedFixture:
    arrays: dict[str, np.ndarray]
    diagnostics: dict[str, Any]


FixtureBuilder = Callable[[str, np.ndarray, int], DerivedFixture]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action",
        required=True,
        choices=("phase-a", "finalize-phase-a", "phase-b"),
    )
    parser.add_argument(
        "--config", default="configs/candidate_graph_oracle_ceiling_v4.json"
    )
    parser.add_argument("--config-sha256")
    parser.add_argument("--fixture-manifest")
    parser.add_argument("--fixture-manifest-sha256")
    parser.add_argument("--fixture-root")
    parser.add_argument("--phase-a-dir")
    parser.add_argument("--phase-a-envelope-sha256")
    parser.add_argument("--phase-a-dirs", nargs="+")
    parser.add_argument("--phase-a-envelope-sha256s", nargs="+")
    parser.add_argument("--finalized-phase-a-dir")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--fixture-bundle-root")
    parser.add_argument("--lifecycle-ledger")
    parser.add_argument("--output")
    parser.add_argument("--denoiser")
    parser.add_argument("--hbt-checkpoint")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--denoise-batch-size", type=int, default=512)
    parser.add_argument("--classical-chunk-size", type=int, default=64)
    return parser.parse_args(argv)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _open_anchored_directory(path: Path) -> int:
    """Open every absolute directory component with O_NOFOLLOW."""

    resolved_text = os.path.abspath(os.fspath(path.expanduser()))
    parts = Path(resolved_text).parts
    descriptor = os.open(parts[0], os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in parts[1:]:
            next_descriptor = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _read_fd_bytes(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _secure_absolute_bytes(path: Path) -> tuple[bytes, os.stat_result]:
    parent_descriptor = _open_anchored_directory(path.parent)
    descriptor = -1
    try:
        descriptor = os.open(
            path.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise RuntimeError(f"secure file must be regular with nlink==1: {path.name}")
        payload = _read_fd_bytes(descriptor)
        if len(payload) != metadata.st_size:
            raise RuntimeError(f"secure file changed while reading: {path.name}")
        return payload, metadata
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_descriptor)


def _secure_relative_bytes(
    root: Path, relative_value: str, *, expected_parent: str
) -> tuple[bytes, os.stat_result, Path]:
    relative = Path(relative_value)
    if (
        relative.is_absolute()
        or relative.parts != (expected_parent, relative.name)
        or ".." in relative.parts
        or not _SAFE_ID.fullmatch(relative.name)
    ):
        raise RuntimeError(f"secure relative path is not canonical: {relative_value}")
    root_descriptor = _open_anchored_directory(root)
    parent_descriptor = -1
    descriptor = -1
    try:
        root_metadata = os.fstat(root_descriptor)
        parent_descriptor = os.open(
            expected_parent,
            os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=root_descriptor,
        )
        descriptor = os.open(
            relative.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise RuntimeError(f"secure artifact must be regular with nlink==1: {relative.name}")
        if os.fstat(parent_descriptor).st_dev != root_metadata.st_dev or metadata.st_dev != root_metadata.st_dev:
            raise RuntimeError(f"secure artifact crosses a mount boundary: {relative.name}")
        payload = _read_fd_bytes(descriptor)
        if len(payload) != metadata.st_size:
            raise RuntimeError(f"secure artifact changed while reading: {relative.name}")
        return payload, metadata, root / relative
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if parent_descriptor >= 0:
            os.close(parent_descriptor)
        os.close(root_descriptor)


def _secure_direct_bytes(root: Path, name: str) -> tuple[bytes, os.stat_result, Path]:
    if not _SAFE_ID.fullmatch(name) or "/" in name:
        raise RuntimeError(f"invalid direct secure filename: {name}")
    root_descriptor = _open_anchored_directory(root)
    descriptor = -1
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=root_descriptor,
        )
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise RuntimeError(f"secure direct file must be regular with nlink==1: {name}")
        payload = _read_fd_bytes(descriptor)
        if len(payload) != metadata.st_size:
            raise RuntimeError(f"secure direct file changed while reading: {name}")
        return payload, metadata, root / name
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(root_descriptor)


def _assert_exact_directory_entries(path: Path, expected: set[str]) -> None:
    descriptor = _open_anchored_directory(path)
    try:
        actual = set(os.listdir(descriptor))
    finally:
        os.close(descriptor)
    if actual != expected:
        raise RuntimeError(
            f"unlisted directory entry detected in {path.name}: "
            f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
        )


def _verify_lifecycle_chain(
    ledger_value: str | None,
    *,
    protocol_instance_id: str,
    config_sha256: str,
    required_last_state: str,
    protocol: Mapping[str, Any] | None = None,
) -> tuple[Path, dict[str, str]]:
    if not ledger_value:
        raise RuntimeError("lifecycle ledger is required")
    root = Path(ledger_value).expanduser().absolute()
    states = ("PREP", "SEALED", "PHASE_A", "LABEL_ACCESS")
    if required_last_state not in states:
        raise RuntimeError("invalid required lifecycle terminal state")
    expected_states = list(states[: states.index(required_last_state) + 1])
    allowed_entries = {f"{state}.json" for state in expected_states}
    transition_directory = root / "runtime_pin_transitions"
    if protocol is not None:
        allowed_entries.add("runtime_pin_transitions")
    _assert_exact_directory_entries(root, allowed_entries)
    hashes: dict[str, str] = {}
    previous_state: str | None = None
    previous_hash: str | None = None
    prep_config_sha256: str | None = None
    expected_payload_keys = {
        "schema_version",
        "kind",
        "protocol_instance_id",
        "state",
        "frozen_contract_sha256",
        "config_sha256_or_null",
        "predecessor_sha256",
    }
    for state in expected_states:
        filename = f"{state}.json"
        file_bytes, _, _ = _secure_direct_bytes(root, filename)
        try:
            payload = json.loads(file_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"invalid lifecycle JSON: {state}") from error
        if file_bytes != _ledger_canonical_bytes(payload) or set(payload) != expected_payload_keys:
            raise RuntimeError(f"lifecycle schema/canonical bytes drift: {state}")
        if (
            payload["schema_version"] != 1
            or payload["kind"] != "candidate_graph_oracle_lifecycle"
            or payload["protocol_instance_id"] != protocol_instance_id
            or payload["frozen_contract_sha256"] != EXPECTED_FROZEN_CONTRACT_SHA256
            or payload["state"] != state
            or payload["predecessor_sha256"] != previous_hash
            or not isinstance(payload["config_sha256_or_null"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", payload["config_sha256_or_null"])
        ):
            raise RuntimeError(f"lifecycle chain content drift: {state}")
        if state in {"SEALED", "PHASE_A", "LABEL_ACCESS"} and payload["config_sha256_or_null"] != config_sha256:
            raise RuntimeError(f"{state} does not bind final config SHA256")
        if state == "PREP":
            prep_config_sha256 = payload["config_sha256_or_null"]
        hashes[state] = _bytes_sha256(file_bytes)
        previous_state = state
        previous_hash = hashes[state]
    del previous_state
    if protocol is not None:
        if prep_config_sha256 is None:
            raise RuntimeError("lifecycle PREP config binding is missing")
        _verify_runtime_pin_transition_receipts(
            root=transition_directory,
            protocol=protocol,
            prep_config_sha256=prep_config_sha256,
            final_config_sha256=config_sha256,
        )
    return root, hashes


def _verify_runtime_pin_transition_receipts(
    *,
    root: Path,
    protocol: Mapping[str, Any],
    prep_config_sha256: str,
    final_config_sha256: str,
) -> None:
    expected_names = {
        "00_code_pins.intent.json",
        "00_code_pins.complete.json",
        "01_fixtures_pins.intent.json",
        "01_fixtures_pins.complete.json",
    }
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("runtime pin transition receipt directory is missing")
    _assert_exact_directory_entries(root, expected_names)
    intent_keys = {
        "schema_version",
        "kind",
        "stage",
        "stage_index",
        "protocol_instance_id",
        "frozen_contract_sha256",
        "config_relative_path",
        "previous_config_sha256",
        "intended_config_sha256",
        "pin_sha256_values",
        "created_utc",
    }
    completion_keys = {
        "schema_version",
        "kind",
        "stage",
        "stage_index",
        "protocol_instance_id",
        "frozen_contract_sha256",
        "config_relative_path",
        "previous_config_sha256",
        "final_config_sha256",
        "pin_sha256_values",
        "intent_sha256",
        "completed_utc",
    }
    pins = protocol.get("runtime_pins")
    policy = protocol.get("runtime_pin_mutation_policy")
    if not isinstance(pins, dict) or not isinstance(policy, dict):
        raise RuntimeError("runtime pin receipt protocol closure is missing")
    previous_final: str | None = None
    for stage, index, prefix, pair_key, expected_final in (
        ("code", 0, "00_code_pins", "code_pin_fields", prep_config_sha256),
        (
            "fixtures",
            1,
            "01_fixtures_pins",
            "fixture_pin_fields",
            final_config_sha256,
        ),
    ):
        intent_raw, _, _ = _secure_direct_bytes(root, f"{prefix}.intent.json")
        completion_raw, _, _ = _secure_direct_bytes(
            root, f"{prefix}.complete.json"
        )
        try:
            intent = json.loads(intent_raw.decode("utf-8"))
            completion = json.loads(completion_raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"invalid {stage} transition receipt JSON") from error
        if (
            not isinstance(intent, dict)
            or not isinstance(completion, dict)
            or set(intent) != intent_keys
            or set(completion) != completion_keys
            or intent_raw != _ledger_canonical_bytes(intent)
            or completion_raw != _ledger_canonical_bytes(completion)
        ):
            raise RuntimeError(f"{stage} transition receipt schema drift")
        common = {
            "schema_version": 1,
            "stage": stage,
            "stage_index": index,
            "protocol_instance_id": protocol.get("protocol_instance_id"),
            "frozen_contract_sha256": EXPECTED_FROZEN_CONTRACT_SHA256,
            "config_relative_path": (
                "configs/candidate_graph_oracle_ceiling_v4.json"
            ),
        }
        for payload, kind in (
            (
                intent,
                "candidate_graph_oracle_runtime_pin_transition_intent",
            ),
            (
                completion,
                "candidate_graph_oracle_runtime_pin_transition_completion",
            ),
        ):
            if payload.get("kind") != kind or any(
                payload.get(key) != value for key, value in common.items()
            ):
                raise RuntimeError(f"{stage} transition receipt binding drift")
        pairs = policy.get(pair_key)
        if not isinstance(pairs, list) or not pairs:
            raise RuntimeError(f"{stage} transition pin policy is missing")
        fields: list[str] = []
        for pair in pairs:
            if not isinstance(pair, dict) or set(pair) != {
                "path_field",
                "sha256_field",
            }:
                raise RuntimeError(f"{stage} transition pin pair drift")
            fields.append(str(pair["sha256_field"]))
        expected_values = {field: pins.get(field) for field in fields}
        if (
            intent.get("pin_sha256_values") != expected_values
            or completion.get("pin_sha256_values") != expected_values
        ):
            raise RuntimeError(f"{stage} transition pin values drift")
        if (
            completion.get("previous_config_sha256")
            != intent.get("previous_config_sha256")
            or completion.get("final_config_sha256")
            != intent.get("intended_config_sha256")
            or completion.get("intent_sha256") != _bytes_sha256(intent_raw)
            or completion.get("final_config_sha256") != expected_final
        ):
            raise RuntimeError(f"{stage} transition receipt chain mismatch")
        if previous_final is not None and intent.get(
            "previous_config_sha256"
        ) != previous_final:
            raise RuntimeError("fixture transition does not follow code transition")
        previous_final = str(completion["final_config_sha256"])


def _ledger_canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _canonical_sha256(payload: Any) -> str:
    return _bytes_sha256(_canonical_bytes(payload))


def _bind_self_sha256(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "self_sha256" in payload:
        raise RuntimeError("self_sha256 must not be pre-populated")
    result = dict(payload)
    result["self_sha256"] = _canonical_sha256(result)
    return result


def _verify_self_sha256(payload: Mapping[str, Any]) -> None:
    expected = payload.get("self_sha256")
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise RuntimeError("manifest self_sha256 is missing or malformed")
    base = {key: value for key, value in payload.items() if key != "self_sha256"}
    if _canonical_sha256(base) != expected:
        raise RuntimeError("manifest self_sha256 mismatch")


def _write_plain_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_bytes(path, _ledger_canonical_bytes(payload))


def _load_self_manifest(path: Path, expected_file_sha256: str) -> dict[str, Any]:
    file_bytes, _ = _secure_absolute_bytes(path)
    if _bytes_sha256(file_bytes) != expected_file_sha256:
        raise RuntimeError(f"manifest SHA256 mismatch: {path.name}")
    try:
        payload = json.loads(file_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid manifest JSON: {path.name}") from error
    if not isinstance(payload, dict) or file_bytes != _ledger_canonical_bytes(payload):
        raise RuntimeError(f"manifest is not canonical JSON: {path.name}")
    _verify_self_sha256(payload)
    return payload


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _write_envelope(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    envelope = {"payload": payload, "payload_sha256": _canonical_sha256(payload)}
    _atomic_bytes(path, _canonical_bytes(envelope) + b"\n")
    return envelope


def _load_envelope(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    payload, _ = _secure_absolute_bytes(path)
    return _load_envelope_bytes(payload, expected_sha256, label=str(path))


def _load_envelope_bytes(
    file_bytes: bytes, expected_sha256: str | None, *, label: str
) -> dict[str, Any]:
    if expected_sha256 is not None and _bytes_sha256(file_bytes) != expected_sha256:
        raise RuntimeError(f"envelope SHA256 mismatch: {label}")
    try:
        envelope = json.loads(file_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid envelope JSON: {label}") from error
    if not isinstance(envelope, dict):
        raise RuntimeError(f"envelope must be an object: {label}")
    if set(envelope) != {"payload", "payload_sha256"}:
        raise RuntimeError(f"non-canonical envelope keys: {label}")
    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        raise RuntimeError(f"envelope payload is not an object: {label}")
    if envelope["payload_sha256"] != _canonical_sha256(payload):
        raise RuntimeError(f"envelope payload hash mismatch: {label}")
    if file_bytes != _canonical_bytes(envelope) + b"\n":
        raise RuntimeError(f"envelope is not canonical JSON: {label}")
    return envelope


def _npy_archive_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    output = BytesIO()
    np.savez(output, **{key: np.asarray(arrays[key]) for key in sorted(arrays)})
    return output.getvalue()


def _npy_bytes(array: np.ndarray) -> bytes:
    output = BytesIO()
    np.save(output, np.asarray(array), allow_pickle=False)
    return output.getvalue()


def _png_bytes(values: np.ndarray) -> bytes:
    image = np.asarray(values)
    if image.dtype != np.uint8 or image.shape != (GRID * TILE, GRID * TILE, 3):
        raise ValueError("render must be uint8 RGB 480x480")
    output = BytesIO()
    Image.fromarray(image, mode="RGB").save(output, format="PNG", compress_level=6)
    return output.getvalue()


def _names_sha256(names: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def _layout_sha256(layout: np.ndarray) -> str:
    layout = validate_permutation(layout)
    return hashlib.sha256(layout.astype(np.int32, copy=False).tobytes()).hexdigest()


def _safe_relative(root: Path, relative_value: str, *, expected_parent: str) -> Path:
    relative = Path(relative_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"artifact path is not safe-relative: {relative_value}")
    if len(relative.parts) != 2 or relative.parts[0] != expected_parent:
        raise RuntimeError(f"artifact path is not canonical: {relative_value}")
    if not _SAFE_ID.fullmatch(relative.name):
        raise RuntimeError(f"artifact name is unsafe: {relative.name}")
    expected_root = (root.resolve() / expected_parent).resolve()
    resolved = (root.resolve() / relative).resolve()
    if resolved.parent != expected_root:
        raise RuntimeError(f"artifact escaped canonical root: {relative_value}")
    return resolved


def _assert_input_only_path(path: Path) -> None:
    lowered = {part.lower() for part in path.expanduser().absolute().parts}
    overlap = lowered.intersection(FORBIDDEN_INPUT_COMPONENTS)
    if overlap:
        raise RuntimeError(f"input-only path contains forbidden component: {sorted(overlap)}")


def _assert_empty_dir(path: Path) -> Path:
    path = path.expanduser().absolute()
    if path.is_symlink():
        raise RuntimeError(f"output directory may not be a symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    parent_descriptor = _open_anchored_directory(path.parent)
    try:
        try:
            os.mkdir(path.name, mode=0o755, dir_fd=parent_descriptor)
        except FileExistsError:
            pass
        descriptor = os.open(
            path.name,
            os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
        try:
            entries = os.listdir(descriptor)
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_descriptor)
    if entries:
        raise RuntimeError(f"output directory must be empty: {path}")
    return path


def _validate_score(matrix: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(matrix)
    if value.shape != (TILE_COUNT, TILE_COUNT):
        raise RuntimeError(f"{name} must be 576x576")
    if value.dtype != np.float32:
        raise RuntimeError(f"{name} must be float32")
    diagonal = np.diag(value)
    off_diagonal = value[~np.eye(TILE_COUNT, dtype=bool)]
    if not np.all(np.isfinite(off_diagonal)):
        raise RuntimeError(f"{name} has non-finite off-diagonal entries")
    if not np.all(np.isposinf(diagonal)):
        raise RuntimeError(f"{name} diagonal must be +inf")
    return value


def _validate_tiles(tiles: np.ndarray) -> np.ndarray:
    value = np.asarray(tiles)
    if value.dtype != np.uint8 or value.shape != (TILE_COUNT, TILE, TILE, 3):
        raise RuntimeError("denoised_tiles must be uint8 576x20x20x3")
    return value


def _input_fixture_arrays(path_or_bytes: Path | bytes) -> dict[str, np.ndarray]:
    source: Path | BytesIO = (
        BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else path_or_bytes
    )
    try:
        with np.load(source, allow_pickle=False) as archive:
            keys = set(archive.files)
            if keys != REQUIRED_INPUT_ARRAYS:
                raise RuntimeError(
                    f"fixture keys drift: {sorted(keys)}"
                )
            values = {key: np.asarray(archive[key]) for key in sorted(keys)}
    except (OSError, ValueError) as error:
        raise RuntimeError("invalid input-only fixture") from error
    values["slot_tiles"] = _validate_tiles(values["slot_tiles"])
    qap_seed = values["qap_seed"]
    if qap_seed.shape != () or qap_seed.dtype != np.uint64:
        raise RuntimeError("qap_seed must be a scalar uint64")
    return values


def _validate_derived_arrays(values: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    if set(values) != REQUIRED_DERIVED_ARRAYS:
        raise RuntimeError(f"derived fixture keys drift: {sorted(values)}")
    result = {key: np.asarray(values[key]) for key in sorted(values)}
    for key in (
        "c1_right",
        "c1_down",
        "hbt_right",
        "hbt_down",
        "w1_right",
        "w1_down",
        "w4_right",
        "w4_down",
    ):
        result[key] = _validate_score(result[key], name=key)
    for key in ("softcycle_layout", "qap_w4_layout", "qap_w1_layout"):
        result[key] = validate_permutation(result[key], name=key)
    result["denoised_tiles"] = _validate_tiles(result["denoised_tiles"])
    return result


def _label_arrays(path_or_bytes: Path | bytes) -> dict[str, np.ndarray]:
    source: Path | BytesIO = (
        BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else path_or_bytes
    )
    try:
        with np.load(source, allow_pickle=False) as archive:
            keys = set(archive.files)
            if keys != REQUIRED_LABEL_ARRAYS:
                raise RuntimeError(f"label keys drift: {sorted(keys)}")
            values = {key: np.asarray(archive[key]) for key in sorted(keys)}
    except (OSError, ValueError) as error:
        raise RuntimeError("invalid exact-panel label") from error
    values["opaque_slot_permutation"] = validate_permutation(
        values["opaque_slot_permutation"], name="opaque_slot_permutation"
    )
    values["composed_slot_to_target"] = validate_permutation(
        values["composed_slot_to_target"], name="composed_slot_to_target"
    )
    clean = values["clean_target_rgb"]
    if clean.dtype != np.uint8 or clean.shape != (GRID * TILE, GRID * TILE, 3):
        raise RuntimeError("clean_target_rgb must be uint8 RGB 480x480")
    return values


def _stable_candidates(matrix: np.ndarray, *, outgoing: int, incoming: int) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix)
    row_order = np.argsort(matrix, axis=1, kind="stable")
    column_order = np.argsort(matrix, axis=0, kind="stable")
    rows: list[tuple[int, int]] = []
    columns: list[tuple[int, int]] = []
    for source in range(TILE_COUNT):
        selected = [int(value) for value in row_order[source] if int(value) != source]
        if len(selected) < outgoing:
            raise RuntimeError("not enough finite outgoing candidates")
        rows.extend((source, destination) for destination in selected[:outgoing])
    for destination in range(TILE_COUNT):
        selected = [
            int(value) for value in column_order[:, destination] if int(value) != destination
        ]
        if len(selected) < incoming:
            raise RuntimeError("not enough finite incoming candidates")
        columns.extend((source, destination) for source in selected[:incoming])
    return np.asarray(rows, dtype=np.int32), np.asarray(columns, dtype=np.int32)


def _layout_edges(layout: np.ndarray, direction: int) -> np.ndarray:
    layout = validate_permutation(layout)
    grid = layout.reshape(GRID, GRID)
    if direction == 0:
        return np.stack([grid[:, :-1].ravel(), grid[:, 1:].ravel()], axis=1).astype(
            np.int32
        )
    if direction == 1:
        return np.stack([grid[:-1, :].ravel(), grid[1:, :].ravel()], axis=1).astype(
            np.int32
        )
    raise ValueError("direction must be 0 or 1")


def build_candidate_graph(arrays: Mapping[str, np.ndarray]) -> CandidateGraph:
    """Build the exact deduplicated C1/HBT/layout candidate union."""

    edges: dict[tuple[int, int, int], int] = {}
    pre_dedup_counts = {key: 0 for key in ORIGIN_BITS}

    def add(direction: int, pairs: np.ndarray, origin: str) -> None:
        bit = ORIGIN_BITS[origin]
        pre_dedup_counts[origin] += len(pairs)
        for source, destination in np.asarray(pairs).tolist():
            key = (direction, int(source), int(destination))
            if source == destination:
                raise RuntimeError("self edge reached candidate union")
            edges[key] = edges.get(key, 0) | bit

    for direction, suffix in ((0, "right"), (1, "down")):
        for prefix in ("c1", "hbt"):
            outgoing, incoming = _stable_candidates(
                arrays[f"{prefix}_{suffix}"], outgoing=32, incoming=8
            )
            add(direction, outgoing, f"{prefix}_out32")
            add(direction, incoming, f"{prefix}_in8")
        for layout_key, origin_key in (
            ("softcycle_layout", "softcycle"),
            ("qap_w4_layout", "qap_w4"),
            ("qap_w1_layout", "qap_w1"),
        ):
            add(direction, _layout_edges(arrays[layout_key], direction), origin_key)
    if pre_dedup_counts != EXPECTED_PRE_DEDUP_COUNTS:
        raise RuntimeError("candidate origin pre-dedup count drift")
    ordered = sorted(edges)
    c1 = CompatibilityMatrices("c1", arrays["c1_right"], arrays["c1_down"])
    hbt = CompatibilityMatrices("hbt", arrays["hbt_right"], arrays["hbt_down"])
    expected_w1 = fuse_ranked_scores(
        {"c1": c1, "hbt": hbt}, names=["c1", "hbt"], weights={"hbt": 1.0}, name="w1"
    )
    expected_w4 = fuse_ranked_scores(
        {"c1": c1, "hbt": hbt}, names=["c1", "hbt"], weights={"hbt": 4.0}, name="w4"
    )
    direction = np.asarray([key[0] for key in ordered], dtype=np.uint8)
    source = np.asarray([key[1] for key in ordered], dtype=np.uint16)
    destination = np.asarray([key[2] for key in ordered], dtype=np.uint16)
    w1 = CompatibilityMatrices("w1", arrays["w1_right"], arrays["w1_down"])
    w4 = CompatibilityMatrices("w4", arrays["w4_right"], arrays["w4_down"])
    for frozen, expected, label in (
        (w1, expected_w1, "w1"),
        (w4, expected_w4, "w4"),
    ):
        if not np.array_equal(frozen.right, expected.right) or not np.array_equal(
            frozen.down, expected.down
        ):
            raise RuntimeError(f"frozen {label} matrix differs from exact rank fusion")
    cost_sources = {"c1": c1, "hbt": hbt, "w1": w1, "w4": w4}
    costs: dict[str, np.ndarray] = {}
    for name, score in cost_sources.items():
        costs[name] = np.asarray(
            [
                (score.right if int(d) == 0 else score.down)[int(first), int(second)]
                for d, first, second in zip(direction, source, destination, strict=True)
            ],
            dtype=np.float32,
        )
    return CandidateGraph(
        direction=direction,
        source=source,
        destination=destination,
        origin_mask=np.asarray([edges[key] for key in ordered], dtype=np.uint8),
        c1_cost=costs["c1"],
        hbt_cost=costs["hbt"],
        w1_cost=costs["w1"],
        w4_cost=costs["w4"],
    )


def _graph_arrays(graph: CandidateGraph, fixture: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    values = {
        "candidate_direction": graph.direction,
        "candidate_source": graph.source,
        "candidate_destination": graph.destination,
        "candidate_origin_mask": graph.origin_mask,
        "candidate_c1_cost": graph.c1_cost,
        "candidate_hbt_cost": graph.hbt_cost,
        "candidate_w1_cost": graph.w1_cost,
        "candidate_w4_cost": graph.w4_cost,
    }
    values.update({key: np.asarray(fixture[key]) for key in sorted(REQUIRED_DERIVED_ARRAYS)})
    return values


def _graph_from_archive(values: Mapping[str, np.ndarray]) -> CandidateGraph:
    return CandidateGraph(
        direction=values["candidate_direction"],
        source=values["candidate_source"],
        destination=values["candidate_destination"],
        origin_mask=values["candidate_origin_mask"],
        c1_cost=values["candidate_c1_cost"],
        hbt_cost=values["candidate_hbt_cost"],
        w1_cost=values["candidate_w1_cost"],
        w4_cost=values["candidate_w4_cost"],
    )


def _load_graph_artifact(path_or_bytes: Path | bytes) -> tuple[CandidateGraph, dict[str, np.ndarray]]:
    expected = REQUIRED_DERIVED_ARRAYS | {
        "candidate_direction",
        "candidate_source",
        "candidate_destination",
        "candidate_origin_mask",
        "candidate_c1_cost",
        "candidate_hbt_cost",
        "candidate_w1_cost",
        "candidate_w4_cost",
    }
    source: Path | BytesIO = (
        BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else path_or_bytes
    )
    try:
        with np.load(source, allow_pickle=False) as archive:
            if set(archive.files) != expected:
                raise RuntimeError("graph artifact keys drift")
            values = {key: np.asarray(archive[key]) for key in sorted(archive.files)}
    except (OSError, ValueError) as error:
        raise RuntimeError("invalid graph artifact") from error
    derived = _validate_derived_arrays(
        {key: values[key] for key in REQUIRED_DERIVED_ARRAYS}
    )
    values.update(derived)
    return _graph_from_archive(values), values


def _truth_masks(graph: CandidateGraph, slot_to_target: np.ndarray) -> np.ndarray:
    right_truth, down_truth = true_neighbour_slots(slot_to_target)
    truth = (right_truth, down_truth)
    expected = np.asarray(
        [truth[int(direction)][int(source)] for direction, source in zip(graph.direction, graph.source, strict=True)],
        dtype=np.int32,
    )
    return graph.destination.astype(np.int32) == expected


def validate_truth_geometry(slot_to_target: np.ndarray) -> dict[str, int]:
    truth = validate_permutation(slot_to_target, name="composed_slot_to_target")
    right, down = true_neighbour_slots(truth)
    if int(np.count_nonzero(right >= 0)) != 552 or int(np.count_nonzero(down >= 0)) != 552:
        raise RuntimeError("truth must contain exactly 552 right and 552 down edges")
    if int(np.count_nonzero(right < 0)) != GRID or int(np.count_nonzero(down < 0)) != GRID:
        raise RuntimeError("truth boundary cardinality drift")
    right_sources = np.flatnonzero(right >= 0)
    down_sources = np.flatnonzero(down >= 0)
    if len(np.unique(right[right_sources])) != 552 or len(np.unique(down[down_sources])) != 552:
        raise RuntimeError("truth directed neighbours are not one-to-one")
    left = np.full(TILE_COUNT, -1, dtype=np.int32)
    up = np.full(TILE_COUNT, -1, dtype=np.int32)
    left[right[right_sources]] = right_sources
    up[down[down_sources]] = down_sources
    if int(np.count_nonzero(left >= 0)) != 552 or int(np.count_nonzero(up >= 0)) != 552:
        raise RuntimeError("truth inverse-side cardinality drift")
    for source in right_sources.tolist():
        destination = int(right[source])
        if int(left[destination]) != source:
            raise RuntimeError("right/left truth inverse mismatch")
        first_target, second_target = int(truth[source]), int(truth[destination])
        if first_target // GRID != second_target // GRID or second_target != first_target + 1:
            raise RuntimeError("right truth crosses row boundary")
    for source in down_sources.tolist():
        destination = int(down[source])
        if int(up[destination]) != source or int(truth[destination]) != int(truth[source]) + GRID:
            raise RuntimeError("down/up truth inverse mismatch")
    return {"right": 552, "left": 552, "down": 552, "up": 552, "unique": 1104}


def candidate_recall_metrics(
    graph: CandidateGraph, slot_to_target: np.ndarray
) -> dict[str, Any]:
    slot_to_target = validate_permutation(slot_to_target, name="slot_to_target")
    validate_truth_geometry(slot_to_target)
    right_truth, down_truth = true_neighbour_slots(slot_to_target)
    lookup = {
        (int(direction), int(source), int(destination)): int(mask)
        for direction, source, destination, mask in zip(
            graph.direction,
            graph.source,
            graph.destination,
            graph.origin_mask,
            strict=True,
        )
    }
    side_hits: dict[str, list[bool]] = {key: [] for key in ("right", "left", "down", "up")}
    unique_hits: list[bool] = []
    origin_unique_hits = {key: [] for key in ORIGIN_BITS}
    for direction, truth, outgoing_side, incoming_side in (
        (0, right_truth, "right", "left"),
        (1, down_truth, "down", "up"),
    ):
        for source in np.flatnonzero(truth >= 0).tolist():
            destination = int(truth[source])
            mask = lookup.get((direction, int(source), destination), 0)
            hit = mask != 0
            side_hits[outgoing_side].append(hit)
            side_hits[incoming_side].append(hit)
            unique_hits.append(hit)
            for origin, bit in ORIGIN_BITS.items():
                origin_unique_hits[origin].append(bool(mask & bit))
    if len(unique_hits) != 2 * GRID * (GRID - 1):
        raise RuntimeError("truth edge denominator drift")
    result: dict[str, Any] = {
        "truth_unique_edges": len(unique_hits),
        "truth_four_side_queries": 2 * len(unique_hits),
        "unique_true_edge_recall": float(np.mean(unique_hits)),
        "candidate_restricted_attainable_adjacency_fraction": float(
            np.mean(unique_hits)
        ),
        "four_side_recall": float(
            np.mean([value for values in side_hits.values() for value in values])
        ),
        "side_recall": {key: float(np.mean(values)) for key, values in side_hits.items()},
        "origin_unique_true_edge_recall": {
            key: float(np.mean(values)) for key, values in origin_unique_hits.items()
        },
    }
    return result


class _RelativeUnion:
    def __init__(self) -> None:
        self.parent = np.arange(TILE_COUNT, dtype=np.int32)
        # coord[node] - coord[parent[node]]
        self.dx = np.zeros(TILE_COUNT, dtype=np.int32)
        self.dy = np.zeros(TILE_COUNT, dtype=np.int32)
        self.size = np.ones(TILE_COUNT, dtype=np.int32)

    def find(self, node: int) -> tuple[int, tuple[int, int]]:
        parent = int(self.parent[node])
        if parent == node:
            return node, (0, 0)
        root, offset = self.find(parent)
        x = int(self.dx[node]) + offset[0]
        y = int(self.dy[node]) + offset[1]
        self.parent[node] = root
        self.dx[node] = x
        self.dy[node] = y
        return root, (x, y)

    def union(self, first: int, second: int, delta: tuple[int, int]) -> None:
        first_root, first_offset = self.find(first)
        second_root, second_offset = self.find(second)
        # coord(second) == coord(first) + delta
        desired = (
            first_offset[0] + delta[0] - second_offset[0],
            first_offset[1] + delta[1] - second_offset[1],
        )
        if first_root == second_root:
            if desired != (0, 0):
                raise RuntimeError("oracle truth graph has inconsistent coordinates")
            return
        if self.size[first_root] < self.size[second_root]:
            self.parent[first_root] = second_root
            self.dx[first_root] = -desired[0]
            self.dy[first_root] = -desired[1]
            self.size[second_root] += self.size[first_root]
        else:
            self.parent[second_root] = first_root
            self.dx[second_root] = desired[0]
            self.dy[second_root] = desired[1]
            self.size[first_root] += self.size[second_root]


def oracle_components(
    graph: CandidateGraph, slot_to_target: np.ndarray
) -> tuple[list[OracleComponent], int]:
    truth_mask = _truth_masks(graph, validate_permutation(slot_to_target))
    union = _RelativeUnion()
    true_edges = 0
    for direction, source, destination, keep in zip(
        graph.direction, graph.source, graph.destination, truth_mask, strict=True
    ):
        if not bool(keep):
            continue
        delta = (1, 0) if int(direction) == 0 else (0, 1)
        union.union(int(source), int(destination), delta)
        true_edges += 1
    grouped: dict[int, list[tuple[int, tuple[int, int]]]] = {}
    for tile in range(TILE_COUNT):
        root, coordinate = union.find(tile)
        grouped.setdefault(root, []).append((tile, coordinate))
    components: list[OracleComponent] = []
    for entries in grouped.values():
        entries.sort()
        min_x = min(value[1][0] for value in entries)
        min_y = min(value[1][1] for value in entries)
        components.append(
            OracleComponent(
                members=tuple(value[0] for value in entries),
                coordinates=tuple(
                    (value[1][0] - min_x, value[1][1] - min_y) for value in entries
                ),
            )
        )
    components.sort(key=lambda value: (-len(value.members), value.members))
    return components, true_edges


def truth_filtered_components(
    graph: CandidateGraph, slot_to_target: np.ndarray
) -> tuple[list[dict[int, tuple[int, int]]], dict[str, Any]]:
    """Run the production component grower on C intersect truth.

    ``grow_components`` counts every accepted coordinate-consistent edge,
    including edges that close an already-connected cycle.  The returned count
    is therefore intentionally named ``accepted_consistent_edges`` and never
    reported as a merge count.  An independent weighted-union implementation
    verifies the complete partition and every relative offset.
    """

    truth = validate_permutation(slot_to_target, name="slot_to_target")
    keep = _truth_masks(graph, truth)
    selected_edges = [
        (
            float(w4_cost),
            -int(int(origin_mask).bit_count()),
            int(direction),
            int(source),
            int(destination),
        )
        for direction, source, destination, origin_mask, w4_cost, selected in zip(
            graph.direction,
            graph.source,
            graph.destination,
            graph.origin_mask,
            graph.w4_cost,
            keep,
            strict=True,
        )
        if bool(selected)
    ]
    selected_edges.sort()
    proposals = [
        ProposedEdge(
            first=source,
            second=destination,
            dx=1 if direction == 0 else 0,
            dy=0 if direction == 0 else 1,
            cost=w4_cost,
            margin=0.0,
            reciprocal=False,
            in_loop=False,
        )
        for w4_cost, _, direction, source, destination in selected_edges
    ]
    grown, accepted = grow_components(proposals)
    independent, independent_edges = oracle_components(graph, truth)

    def canonical_component(members: Mapping[int, tuple[int, int]]) -> tuple[Any, ...]:
        min_x = min(value[0] for value in members.values())
        min_y = min(value[1] for value in members.values())
        return tuple(
            (int(tile), int(x - min_x), int(y - min_y))
            for tile, (x, y) in sorted(members.items())
        )

    grown_canonical = sorted(
        (canonical_component(component) for component in grown),
        key=lambda value: (-len(value), value),
    )
    independent_canonical = [
        tuple(
            (tile, coordinate[0], coordinate[1])
            for tile, coordinate in zip(
                component.members, component.coordinates, strict=True
            )
        )
        for component in independent
    ]
    if grown_canonical != independent_canonical:
        raise RuntimeError("grow_components partition/relative-offset audit failed")
    if independent_edges != len(proposals):
        raise RuntimeError("truth-filter edge count audit failed")
    if sorted(tile for component in grown for tile in component) != list(range(TILE_COUNT)):
        raise RuntimeError("oracle component partition is incomplete")
    return grown, {
        "truth_filtered_candidate_edges": len(proposals),
        "accepted_consistent_edges": int(accepted),
        "connected_component_count": len(grown),
        "component_sizes": [len(component) for component in grown],
        "largest_connected_component": max(len(component) for component in grown),
        "non_singleton_covered_tile_fraction": float(
            sum(len(component) for component in grown if len(component) >= 2)
            / TILE_COUNT
        ),
        "partition_and_relative_offsets_independently_verified": True,
    }


def _w4_compatibility(arrays: Mapping[str, np.ndarray]) -> CompatibilityMatrices:
    return CompatibilityMatrices(
        "frozen_C1_HBTw4_rank_fusion",
        arrays["w4_right"],
        arrays["w4_down"],
    )


def _translation_options(component: OracleComponent) -> Iterable[tuple[int, np.ndarray]]:
    coordinates = np.asarray(component.coordinates, dtype=np.int32)
    width = int(coordinates[:, 0].max()) + 1
    height = int(coordinates[:, 1].max()) + 1
    members = np.asarray(component.members, dtype=np.int32)
    for origin_y in range(GRID - height + 1):
        for origin_x in range(GRID - width + 1):
            positions = (coordinates[:, 1] + origin_y) * GRID + coordinates[:, 0] + origin_x
            yield origin_y * GRID + origin_x, np.stack([positions, members], axis=1)


def _partial_increment(
    occupancy: np.ndarray,
    placements: np.ndarray,
    compatibility: CompatibilityMatrices,
    unary: np.ndarray,
    *,
    boundary_weight: float,
) -> float:
    placed_by_position = {int(position): int(tile) for position, tile in placements.tolist()}
    score = 0.0
    for position, tile in placed_by_position.items():
        score += boundary_weight * float(unary[tile, position])
        row, column = divmod(position, GRID)
        for neighbour_position, kind in (
            (position - 1, "left") if column > 0 else (-1, "none"),
            (position + 1, "right") if column + 1 < GRID else (-1, "none"),
            (position - GRID, "up") if row > 0 else (-1, "none"),
            (position + GRID, "down") if row + 1 < GRID else (-1, "none"),
        ):
            if neighbour_position < 0:
                continue
            neighbour = placed_by_position.get(neighbour_position)
            internal = neighbour is not None
            if neighbour is None:
                value = int(occupancy[neighbour_position])
                if value >= 0:
                    neighbour = value
            if neighbour is None:
                continue
            # Internal contacts are counted only from left/up to avoid doubling.
            if internal and kind in {"right", "down"}:
                continue
            if kind == "left":
                score += float(compatibility.right[neighbour, tile])
            elif kind == "right":
                score += float(compatibility.right[tile, neighbour])
            elif kind == "up":
                score += float(compatibility.down[neighbour, tile])
            elif kind == "down":
                score += float(compatibility.down[tile, neighbour])
    return score


def _hungarian_complete(
    occupancy: np.ndarray,
    compatibility: CompatibilityMatrices,
    unary: np.ndarray,
    *,
    boundary_weight: float,
) -> np.ndarray:
    occupancy = np.asarray(occupancy, dtype=np.int32).copy()
    missing_tiles = np.setdiff1d(
        np.arange(TILE_COUNT, dtype=np.int32), occupancy[occupancy >= 0], assume_unique=True
    )
    holes = np.flatnonzero(occupancy < 0).astype(np.int32)
    if len(missing_tiles) != len(holes):
        raise RuntimeError("partial layout tile/hole counts disagree")
    if len(holes) == 0:
        return validate_permutation(occupancy)
    cost = np.empty((len(missing_tiles), len(holes)), dtype=np.float64)
    for tile_index, tile in enumerate(missing_tiles.tolist()):
        for hole_index, position in enumerate(holes.tolist()):
            row, column = divmod(position, GRID)
            value = boundary_weight * float(unary[tile, position])
            contacts = 0
            if column > 0 and occupancy[position - 1] >= 0:
                value += float(compatibility.right[int(occupancy[position - 1]), tile])
                contacts += 1
            if column + 1 < GRID and occupancy[position + 1] >= 0:
                value += float(compatibility.right[tile, int(occupancy[position + 1])])
                contacts += 1
            if row > 0 and occupancy[position - GRID] >= 0:
                value += float(compatibility.down[int(occupancy[position - GRID]), tile])
                contacts += 1
            if row + 1 < GRID and occupancy[position + GRID] >= 0:
                value += float(compatibility.down[tile, int(occupancy[position + GRID])])
                contacts += 1
            if contacts:
                value /= contacts
            # Exact deterministic perturbation only breaks a mathematical tie.
            value += np.finfo(np.float64).eps * (tile_index * len(holes) + hole_index)
            cost[tile_index, hole_index] = value
    row_indices, column_indices = linear_sum_assignment(cost)
    if not np.array_equal(row_indices, np.arange(len(missing_tiles))):
        raise RuntimeError("Hungarian assignment did not cover every missing tile")
    occupancy[holes[column_indices]] = missing_tiles[row_indices]
    return validate_permutation(occupancy, name="hungarian_position_to_slot")


def oracle_filter_beam_hungarian_qap(
    components: Sequence[dict[int, tuple[int, int]]],
    compatibility: CompatibilityMatrices,
    *,
    qap_seed: int,
    beam_width: int = 8,
    boundary_weight: float = 0.05,
    qap_iterations: int = 25,
    qap_restarts: int = 2,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Pack oracle-filtered relative components without absolute truth positions."""

    if beam_width != 8 or qap_iterations != 25 or qap_restarts != 2:
        raise RuntimeError("oracle packer contract drift")
    if boundary_weight != 0.05:
        raise RuntimeError("oracle packer boundary weight drift")
    ordered = [dict(component) for component in components]
    expected_order = sorted(ordered, key=lambda members: (-len(members), min(members)))
    if ordered != expected_order:
        raise RuntimeError("oracle components are not in deterministic grow order")
    grid, placed_tiles = _place_components_beam(
        ordered,
        compatibility,
        boundary_weight=boundary_weight,
        beam_width=beam_width,
        beam_components=8,
        translations_per_state=8,
        placement_costs=None,
    )
    grid_before_hungarian = np.asarray(grid, dtype=np.int32).copy()
    pre_qap, unresolved = _complete_with_hungarian(
        grid.copy(),
        compatibility,
        boundary_weight=boundary_weight,
        placement_costs=None,
    )
    if not np.array_equal(grid, grid_before_hungarian):
        raise RuntimeError("Hungarian completion mutated the frozen beam grid")
    pre_qap = validate_permutation(pre_qap, name="oracle_pre_qap_layout")
    qap = directional_qap(
        compatibility,
        initial=pre_qap,
        iterations=qap_iterations,
        restarts=qap_restarts,
        seed=qap_seed,
        boundary_weight=boundary_weight,
        initial_weight=0.75,
        noisy_components=3,
        noise_scale=1.0,
        refine_swaps=8,
        refine_weak_cells=32,
    )
    layout = validate_permutation(qap.position_to_slot, name="oracle_qap_layout")
    return layout, {
        "beam_width": beam_width,
        "beam_components": 8,
        "translations_per_state": 8,
        "placement_costs": None,
        "multi_tile_components": int(sum(len(value) >= 2 for value in ordered)),
        "beam_placed_tiles": int(placed_tiles),
        "unresolved_before_hungarian": int(unresolved),
        "hungarian_received_grid_copy": True,
        "pre_qap_layout_sha256": _layout_sha256(pre_qap),
        "qap_iterations": qap_iterations,
        "qap_restarts": qap_restarts,
        "qap_seed": qap_seed,
        "qap_objective": float(qap.objective),
        "qap_relaxed_objective": float(qap.relaxed_objective),
        "qap_restart": int(qap.restart),
        "layout_sha256": _layout_sha256(layout),
    }


def target_assisted_translation_ceiling(
    components: Sequence[Mapping[int, tuple[int, int]]],
    compatibility: CompatibilityMatrices,
    slot_to_target: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Diagnostic-only ceiling using true translations of non-singleton components.

    Singletons are deliberately left to the same frozen-w4 Hungarian completion
    used by the ordinary component solver.  This prevents the diagnostic from
    silently becoming a perfect target-derived permutation.
    """

    truth = validate_permutation(slot_to_target, name="slot_to_target")
    true_position_to_slot = inverse_permutation(truth)
    occupancy = np.full(TILE_COUNT, -1, dtype=np.int32)
    assisted_tiles: set[int] = set()
    for component in components:
        if len(component) <= 1:
            continue
        translations = {
            (
                int(truth[int(tile)] % GRID - coordinate[0]),
                int(truth[int(tile)] // GRID - coordinate[1]),
            )
            for tile, coordinate in component.items()
        }
        if len(translations) != 1:
            raise RuntimeError("oracle component has no single true translation")
        tx, ty = next(iter(translations))
        for tile, (x, y) in component.items():
            column, row = x + tx, y + ty
            if not (0 <= column < GRID and 0 <= row < GRID):
                raise RuntimeError("target-assisted translation left grid bounds")
            position = row * GRID + column
            if position != int(truth[int(tile)]):
                raise RuntimeError("target-assisted translation does not match truth")
            if occupancy[position] >= 0:
                raise RuntimeError("target-assisted component translations overlap")
            occupancy[position] = tile
            assisted_tiles.add(int(tile))
    grid = occupancy.reshape(GRID, GRID)
    frozen_grid = grid.copy()
    layout, unresolved = _complete_with_hungarian(
        grid.copy(),
        compatibility,
        boundary_weight=0.05,
        placement_costs=None,
    )
    if not np.array_equal(grid, frozen_grid):
        raise RuntimeError("translation diagnostic Hungarian mutated anchored grid")
    layout = validate_permutation(layout, name="target_assisted_translation_layout")
    if len(assisted_tiles) < TILE_COUNT and np.array_equal(layout, true_position_to_slot):
        # Perfect can occur by chance (e.g. baseline already perfect), but the
        # report must make clear it was not produced by placing singleton truth.
        accidental_perfect = True
    else:
        accidental_perfect = False
    return layout, {
        "diagnostic_only": True,
        "eligible_for_gate": False,
        "non_singleton_assisted_tiles": len(assisted_tiles),
        "singleton_truth_placements": 0,
        "unresolved_before_w4_hungarian": int(unresolved),
        "post_completion_qap": False,
        "accidentally_exact_after_baseline_repair": accidental_perfect,
        "layout_sha256": _layout_sha256(layout),
    }


def _render_metrics(
    layout: np.ndarray, tiles: np.ndarray, clean_target: np.ndarray
) -> dict[str, float]:
    render = merge_tiles_numpy(_validate_tiles(tiles)[validate_permutation(layout)])
    return {
        "rgb_ssim": float(
            structural_similarity(
                clean_target, render, channel_axis=2, data_range=255
            )
        )
    }


def _opaque_qap_seed(opaque_id: str) -> int:
    if not re.fullmatch(r"[0-9a-f]{32}", opaque_id):
        raise RuntimeError("invalid opaque id for QAP seed")
    return int.from_bytes(
        hashlib.sha256(f"qap:{opaque_id}".encode("utf-8")).digest()[:8], "big"
    )


def _resolve_repo_path(config_path: Path, configured: str, override: str | None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    path = Path(configured).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (config_path.parent.parent / path).resolve()


def _validated_runtime_assets(
    protocol: Mapping[str, Any], config_path: Path, args: argparse.Namespace
) -> dict[str, dict[str, str]]:
    contract = protocol["frozen_contract"]
    assets = contract["assets"]
    result: dict[str, dict[str, str]] = {}
    for key, override in (("denoiser", args.denoiser), ("hbt", args.hbt_checkpoint)):
        spec = assets[key]
        path = _resolve_repo_path(config_path, str(spec["path"]), override)
        payload, _ = _secure_absolute_bytes(path)
        actual = _bytes_sha256(payload)
        if actual != spec["sha256"]:
            raise RuntimeError(f"pinned {key} asset SHA256 mismatch")
        result[key] = {"path": str(path), "sha256": actual}
    repository = config_path.parent.parent
    known_code = assets["known_code_sha256"]
    if not isinstance(known_code, dict) or not known_code:
        raise RuntimeError("protocol has no known-code hash closure")
    for relative, expected in known_code.items():
        path = _safe_repo_relative(repository, str(relative))
        payload, _ = _secure_absolute_bytes(path)
        if _bytes_sha256(payload) != expected:
            raise RuntimeError(f"pinned code SHA256 mismatch: {relative}")
    private = assets["private_function_contract"]
    module_path = _safe_repo_relative(repository, str(private["module"]))
    module_payload, _ = _secure_absolute_bytes(module_path)
    if _bytes_sha256(module_payload) != private["module_sha256"]:
        raise RuntimeError("private component-function module hash mismatch")
    for symbol in private["required_symbols"]:
        if symbol not in {"grow_components", "_place_components_beam", "_complete_with_hungarian"}:
            raise RuntimeError("private function symbol contract drift")
    return result


def _validated_runtime_pin_closure(
    protocol: Mapping[str, Any],
    config_path: Path,
    *,
    include_label_fixture_pin: bool = True,
) -> dict[str, str]:
    pins = protocol["runtime_pins"]
    pairs = (
        ("evaluator_path", "evaluator_sha256"),
        ("tests_path", "tests_sha256"),
        ("fixture_builder_path", "fixture_builder_sha256"),
        ("fixture_builder_tests_path", "fixture_builder_tests_sha256"),
        ("pin_finalizer_path", "pin_finalizer_sha256"),
        ("lifecycle_tool_path", "lifecycle_tool_sha256"),
        ("result_verifier_path", "result_verifier_sha256"),
        ("environment_lock_path", "environment_lock_sha256"),
        ("phase_a_runner_path", "phase_a_runner_sha256"),
        ("phase_a_kernel_metadata_path", "phase_a_kernel_metadata_sha256"),
        ("phase_a_launcher_path", "phase_a_launcher_sha256"),
        ("phase_b_runner_path", "phase_b_runner_sha256"),
    )
    repository = config_path.parent.parent
    result: dict[str, str] = {}
    for path_key, hash_key in pairs:
        relative, expected = pins.get(path_key), pins.get(hash_key)
        if not isinstance(relative, str) or not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise RuntimeError(f"runtime pin is null or invalid: {hash_key}")
        path = _safe_repo_relative(repository, relative)
        payload, _ = _secure_absolute_bytes(path)
        if _bytes_sha256(payload) != expected:
            raise RuntimeError(f"runtime pin SHA256 mismatch: {path_key}")
        result[path_key] = str(path)
    fixture_keys = ["fixture_input_manifest_sha256", "fixture_lock_sha256"]
    if include_label_fixture_pin:
        fixture_keys.append("fixture_label_manifest_sha256")
    for key in fixture_keys:
        value = pins.get(key)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise RuntimeError(f"fixture runtime pin is null or invalid: {key}")
    return result


def _validate_environment_lock(
    protocol: Mapping[str, Any], config_path: Path, *, phase: str
) -> None:
    pins = protocol["runtime_pins"]
    path = _safe_repo_relative(config_path.parent.parent, pins["environment_lock_path"])
    payload, _ = _secure_absolute_bytes(path)
    if _bytes_sha256(payload) != pins["environment_lock_sha256"]:
        raise RuntimeError("environment lock SHA256 mismatch")
    try:
        lock = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("invalid environment lock JSON") from error
    if lock.get("kind") != "candidate_graph_oracle_environment_lock":
        raise RuntimeError("environment lock identity drift")
    import cv2
    import kornia
    import scipy
    import skimage
    import torch
    from PIL import __version__ as pillow_version

    actual_packages = {
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "pillow": pillow_version,
        "kornia": kornia.__version__,
        "scikit_image": skimage.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
    }
    if phase == "phase-b":
        expected = lock["fixture_preparation_and_phase_b"]
        if sys.prefix != "/Users/rusyalain/Documents/test/.conda":
            raise RuntimeError("Phase B is not running in the repo-owned .conda")
        if expected["python"] != platform_module.python_version() or expected["platform"] != platform_module.platform() or expected["packages"] != actual_packages:
            raise RuntimeError("Phase-B exact environment lock mismatch")
    elif phase == "phase-a":
        expected = lock["kaggle_phase_a"]
        if platform_module.python_version() != str(expected["python"]) or expected["packages"] != actual_packages:
            raise RuntimeError("Phase-A Kaggle package lock mismatch")
        if not torch.cuda.is_available():
            raise RuntimeError("Phase-A CUDA is unavailable")
        visible = torch.cuda.device_count()
        if visible not in {1, int(expected["device_count"])}:
            raise RuntimeError("Phase-A visible CUDA device count is invalid")
        if torch.version.cuda != str(expected["cuda_runtime"]):
            raise RuntimeError("Phase-A CUDA runtime lock mismatch")
        allowed_devices = {
            (str(value["name"]), tuple(value["capability"]))
            for value in expected["devices"]
        }
        for index in range(visible):
            identity = (
                torch.cuda.get_device_name(index),
                tuple(torch.cuda.get_device_capability(index)),
            )
            if identity not in allowed_devices:
                raise RuntimeError("Phase-A CUDA device identity drift")
            probe = torch.ones((64, 64), device=f"cuda:{index}", dtype=torch.float16)
            if not torch.isfinite(probe @ probe).all().item():
                raise RuntimeError("Phase-A CUDA tensor probe failed")
    else:
        raise RuntimeError("unknown environment-lock phase")


def _safe_repo_relative(repository: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"unsafe repository-relative path: {value}")
    path = (repository.resolve() / relative).resolve()
    try:
        path.relative_to(repository.resolve())
    except ValueError as error:
        raise RuntimeError(f"repository path escaped root: {value}") from error
    return path


def _build_default_fixture_builder(
    protocol: Mapping[str, Any],
    assets: Mapping[str, Mapping[str, str]],
    args: argparse.Namespace,
) -> FixtureBuilder:
    restorer, device, _ = load_restorer(
        assets["denoiser"]["path"], device=args.device, state="ema"
    )
    embedding, _ = load_embedding_checkpoint(assets["hbt"]["path"], device=device)
    for model in (restorer, embedding):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    common = protocol["frozen_contract"]["common_solver"]
    soft = common["soft_cycle"]
    qap_spec = common["qap"]

    def build(opaque_id: str, slot_tiles: np.ndarray, qap_seed: int) -> DerivedFixture:
        del opaque_id
        raw = _validate_tiles(slot_tiles)
        denoised = restore_tiles_uint8(
            restorer, raw, device, batch_size=args.denoise_batch_size
        )
        bank = build_classical_score_bank(
            denoised, prefix="denoised", chunk_size=args.classical_chunk_size
        )
        names = [
            key
            for key in sorted(bank)
            if key.startswith("denoised_") and not key.endswith("_c2")
        ]
        expected_names = protocol["frozen_contract"]["scores"][
            "denoised_C1_equal_rank_fusion"
        ]["component_names_sorted"]
        if names != expected_names:
            raise RuntimeError("C1 component-name closure drift")
        c1 = fuse_ranked_scores(
            bank,
            names=names,
            name="denoised_C1_equal_rank_fusion",
        )
        hbt, hbt_diagnostics = learned_compatibility(
            embedding, denoised, device=device, name="denoised_hbt_l1"
        )
        seed_result = soft_cycle_component_solver(
            hbt,
            top_k=int(soft["top_k"]),
            keep_per_tile=int(soft["keep_per_tile"]),
            proposal_keep_fraction=float(soft["proposal_keep_fraction"]),
            loop_weight=float(soft["loop_weight"]),
            reciprocal_weight=float(soft["reciprocal_weight"]),
        )
        initial = validate_permutation(
            seed_result.position_to_slot, name="softcycle_layout"
        )
        layouts: dict[str, np.ndarray] = {"softcycle_layout": initial.copy()}
        qap_diagnostics: dict[str, Any] = {}
        fused_scores: dict[str, CompatibilityMatrices] = {}
        for label, weight in (("qap_w4", 4.0), ("qap_w1", 1.0)):
            score = fuse_ranked_scores(
                {c1.name: c1, hbt.name: hbt},
                names=[c1.name, hbt.name],
                weights={hbt.name: weight},
                name=f"denoised_C1_HBTw{int(weight)}_rank_fusion",
            )
            fused_scores[label] = score
            result = directional_qap(
                score,
                initial=initial.copy(),
                iterations=int(qap_spec["iterations"]),
                restarts=int(qap_spec["restarts"]),
                seed=int(qap_seed),
                boundary_weight=float(qap_spec["boundary_weight"]),
                initial_weight=float(qap_spec["initial_weight"]),
                noisy_components=int(qap_spec["noisy_components"]),
                noise_scale=float(qap_spec["noise_scale"]),
                refine_swaps=int(qap_spec["refine_swaps"]),
                refine_weak_cells=int(qap_spec["refine_weak_cells"]),
            )
            layouts[f"{label}_layout"] = validate_permutation(
                result.position_to_slot, name=f"{label}_layout"
            ).copy()
            qap_diagnostics[label] = {
                "objective": float(result.objective),
                "relaxed_objective": float(result.relaxed_objective),
                "restart": int(result.restart),
                "iterations": int(result.iterations),
                "converged": bool(result.converged),
            }
        arrays = _validate_derived_arrays(
            {
                "c1_right": c1.right.copy(),
                "c1_down": c1.down.copy(),
                "hbt_right": hbt.right.copy(),
                "hbt_down": hbt.down.copy(),
                "w1_right": fused_scores["qap_w1"].right.copy(),
                "w1_down": fused_scores["qap_w1"].down.copy(),
                "w4_right": fused_scores["qap_w4"].right.copy(),
                "w4_down": fused_scores["qap_w4"].down.copy(),
                "denoised_tiles": denoised.copy(),
                **layouts,
            }
        )
        return DerivedFixture(
            arrays=arrays,
            diagnostics={
                "hbt_outside_logits": {
                    "dtype": str(np.asarray(hbt_diagnostics).dtype),
                    "shape": list(np.asarray(hbt_diagnostics).shape),
                    "c_order_sha256": _array_c_sha256(
                        np.asarray(hbt_diagnostics)
                    ),
                },
                "softcycle": {
                    "accepted_edges": int(seed_result.accepted_edges),
                    "component_sizes": [int(value) for value in seed_result.component_sizes],
                },
                "qap": qap_diagnostics,
            },
        )

    return build


def _validated_protocol(
    config_value: str | Path, expected_config_sha256: str | None
) -> tuple[dict[str, Any], Path, str]:
    config_path = Path(config_value).expanduser().absolute()
    config_bytes, _ = _secure_absolute_bytes(config_path)
    actual_hash = _bytes_sha256(config_bytes)
    if not expected_config_sha256 or actual_hash != expected_config_sha256:
        raise RuntimeError("candidate-graph oracle protocol SHA256 mismatch")
    try:
        protocol = json.loads(config_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("invalid candidate-graph protocol JSON") from error
    if not isinstance(protocol, dict):
        raise RuntimeError("candidate-graph protocol must be an object")
    if protocol.get("kind") != "candidate_graph_oracle_ceiling":
        raise RuntimeError("unexpected candidate-graph protocol kind")
    if protocol.get("safe_for_submission") is not False:
        raise RuntimeError("oracle diagnostic is not fail-closed")
    protocol_instance_id = protocol.get("protocol_instance_id")
    if protocol_instance_id != EXPECTED_PROTOCOL_INSTANCE_ID:
        raise RuntimeError("protocol_instance_id is absent or invalid")
    frozen = protocol.get("frozen_contract")
    if not isinstance(frozen, dict):
        raise RuntimeError("protocol has no frozen contract")
    if protocol.get("frozen_contract_sha256") != EXPECTED_FROZEN_CONTRACT_SHA256:
        raise RuntimeError("embedded frozen-contract SHA256 drift")
    if _canonical_sha256(frozen) != EXPECTED_FROZEN_CONTRACT_SHA256:
        raise RuntimeError("frozen-contract semantic SHA256 mismatch")
    selection = frozen["source_selection"]
    expected_selection = {
        "split": "edge_development",
        "offset": 128,
        "count": 32,
        "source_names_sha256": EXPECTED_NAMES_SHA256,
        "source_count_must_equal": 32,
        "panels_in_label_order": list(PANELS),
        "records_per_panel": 32,
        "total_fixture_records": 64,
    }
    for key, expected in expected_selection.items():
        if selection.get(key) != expected:
            raise RuntimeError(f"source-selection contract drift: {key}")
    origins = frozen["candidate_graph"]["origins"]
    origin_map = {str(value["name"]): int(value["mask"]) for value in origins}
    expected_origin_map = {
        "C1_OUT32": 1,
        "HBT_OUT32": 2,
        "C1_IN8": 4,
        "HBT_IN8": 8,
        "SOFTCYCLE_LAYOUT": 16,
        "QAP_W4_LAYOUT": 32,
        "QAP_W1_LAYOUT": 64,
    }
    if origin_map != expected_origin_map or [int(value["bit"]) for value in origins] != list(range(7)):
        raise RuntimeError("candidate origin-bit contract drift")
    packer = frozen["phase_b_label_scoring"]["gate_driving_oracle_packer"]
    expected_packer = {
        "truth_use": "truth filters candidate edges only",
        "absolute_truth_coordinates_forbidden": True,
        "compatibility": "frozen denoised_C1_HBTw4_rank_fusion",
        "beam_width": 8,
        "beam_components": 8,
        "translations_per_state": 8,
        "boundary_weight": 0.05,
        "placement_costs": None,
        "complete_boundary_weight": 0.05,
        "complete_placement_costs": None,
        "qap_iterations": 25,
        "qap_restarts": 2,
        "qap_boundary_weight": 0.05,
        "qap_initial_weight": 0.75,
        "qap_noisy_components": 3,
        "qap_noise_scale": 1.0,
        "qap_refine_swaps": 8,
        "qap_refine_weak_cells": 32,
        "contributes_to_gate": True,
    }
    for key, expected in expected_packer.items():
        if packer.get(key) != expected:
            raise RuntimeError(f"oracle-filter packer contract drift: {key}")
    translation = frozen["phase_b_label_scoring"][
        "target_assisted_translation_diagnostic"
    ]
    if translation.get("singleton_placement_from_truth_forbidden") is not True or translation.get("contributes_to_gate") is not False:
        raise RuntimeError("translation-ceiling non-gating contract drift")
    gate = frozen["gate"]
    each_panel = gate["each_panel_all_of"]
    macro = gate["balanced_macro_any_of"]
    expected_gate_values = {
        "mean_candidate_union_recall_min": 0.65,
        "median_oracle_largest_connected_component_min_tiles": 128,
        "mean_oracle_packer_combined_adjacency_delta_min": 0.0,
        "mean_oracle_packer_denoised_render_ssim_delta_min": 0.0,
    }
    for key, expected in expected_gate_values.items():
        if each_panel.get(key) != expected:
            raise RuntimeError(f"per-panel gate contract drift: {key}")
    if macro.get("mean_oracle_packer_combined_adjacency_delta_min") != 0.1 or macro.get("mean_oracle_packer_denoised_render_ssim_delta_min") != 0.02:
        raise RuntimeError("balanced-macro gate contract drift")
    if frozen["post_result_policy"].get("qap_weight_reopening_forbidden") is not True:
        raise RuntimeError("protocol permits forbidden QAP-weight reopening")
    pins = protocol.get("runtime_pins")
    if not isinstance(pins, dict):
        raise RuntimeError("protocol has no runtime pins")
    expected_paths = {
        "evaluator_path": "scripts/evaluate_candidate_graph_oracle_v4.py",
        "tests_path": "tests/test_candidate_graph_oracle_v4.py",
    }
    for key, expected in expected_paths.items():
        if pins.get(key) != expected:
            raise RuntimeError(f"runtime pin path drift: {key}")
    repository = config_path.parent.parent
    evaluator_path = _safe_repo_relative(repository, pins["evaluator_path"])
    evaluator_bytes, _ = _secure_absolute_bytes(evaluator_path)
    if pins.get("evaluator_sha256") is None or _bytes_sha256(evaluator_bytes) != pins["evaluator_sha256"]:
        raise RuntimeError("evaluator runtime pin is null or mismatched")
    test_path = _safe_repo_relative(repository, pins["tests_path"])
    test_bytes, _ = _secure_absolute_bytes(test_path)
    if pins.get("tests_sha256") is None or _bytes_sha256(test_bytes) != pins["tests_sha256"]:
        raise RuntimeError("test runtime pin is null or mismatched")
    return protocol, config_path, actual_hash


def _reject_forbidden_input_metadata(value: Any, *, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in ("source", "panel", "target", "label", "secret", "shuffle", "permutation")):
                raise RuntimeError(f"forbidden input-only metadata key at {path}.{key}")
            _reject_forbidden_input_metadata(nested, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_forbidden_input_metadata(nested, path=f"{path}[{index}]")


def _opaque_input_records(
    manifest: dict[str, Any], protocol: Mapping[str, Any] | None = None
) -> list[dict[str, Any]]:
    expected_top = {
        "schema_version",
        "created_utc",
        "kind",
        *COMMON_MANIFEST_SHA_FIELDS,
        "record_count",
        "opaque_ids_sha256",
        "canonical_record_order",
        "allowed_record_metadata",
        "records",
    }
    if set(manifest) != expected_top:
        raise RuntimeError("input fixture manifest has opaque/schema-drift fields")
    if manifest.get("schema_version") != 1 or manifest.get("kind") != "candidate_graph_oracle_fixture_inputs":
        raise RuntimeError("unexpected input fixture manifest identity")
    if manifest.get("frozen_contract_sha256") != EXPECTED_FROZEN_CONTRACT_SHA256:
        raise RuntimeError("input fixture frozen-contract binding mismatch")
    if manifest.get("record_count") != 64:
        raise RuntimeError("input fixture record count drift")
    if manifest.get("protocol_instance_id") != EXPECTED_PROTOCOL_INSTANCE_ID:
        raise RuntimeError("input fixture protocol instance mismatch")
    if manifest.get("canonical_record_order") != "ascending opaque_id" or manifest.get("allowed_record_metadata") != ["opaque_id", "artifact", "arrays"]:
        raise RuntimeError("input fixture opaque metadata contract drift")
    if protocol is not None:
        pins = protocol["runtime_pins"]
        for manifest_key, pin_key in (
            ("evaluator_sha256", "evaluator_sha256"),
            ("tests_sha256", "tests_sha256"),
            ("environment_lock_sha256", "environment_lock_sha256"),
            ("fixture_builder_sha256", "fixture_builder_sha256"),
            ("fixture_builder_tests_sha256", "fixture_builder_tests_sha256"),
            ("pin_finalizer_sha256", "pin_finalizer_sha256"),
            ("lifecycle_tool_sha256", "lifecycle_tool_sha256"),
            ("result_verifier_sha256", "result_verifier_sha256"),
            ("phase_a_runner_sha256", "phase_a_runner_sha256"),
            ("phase_a_kernel_metadata_sha256", "phase_a_kernel_metadata_sha256"),
            ("phase_a_launcher_sha256", "phase_a_launcher_sha256"),
            ("phase_b_runner_sha256", "phase_b_runner_sha256"),
        ):
            if manifest.get(manifest_key) != pins.get(pin_key):
                raise RuntimeError(f"input fixture runtime provenance drift: {manifest_key}")
        if manifest.get("protocol_instance_id") != protocol.get("protocol_instance_id") or manifest.get("frozen_contract_sha256") != protocol.get("frozen_contract_sha256"):
            raise RuntimeError("input fixture protocol binding drift")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != 64:
        raise RuntimeError("input fixtures require exactly 64 opaque records")
    expected_record_keys = {
        "opaque_id",
        "artifact",
        "arrays",
    }
    ids: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != expected_record_keys:
            raise RuntimeError(f"input fixture record schema drift at {index}")
        opaque_id = record["opaque_id"]
        if not isinstance(opaque_id, str) or not re.fullmatch(r"[0-9a-f]{32}", opaque_id):
            raise RuntimeError("invalid opaque fixture id")
        ids.append(opaque_id)
        artifact = record["artifact"]
        if not isinstance(artifact, dict) or set(artifact) != {"path", "bytes", "sha256"}:
            raise RuntimeError("invalid opaque fixture artifact schema")
        if not isinstance(artifact["path"], str) or not isinstance(artifact["bytes"], int) or artifact["bytes"] <= 0:
            raise RuntimeError("invalid opaque fixture artifact metadata")
        if not isinstance(artifact["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"]):
            raise RuntimeError("invalid opaque fixture artifact SHA256")
        arrays = record["arrays"]
        expected_arrays = {
            "slot_tiles": {
                "dtype": "uint8",
                "shape": [576, 20, 20, 3],
            },
            "qap_seed": {"dtype": "uint64", "shape": []},
        }
        if not isinstance(arrays, dict) or set(arrays) != set(expected_arrays):
            raise RuntimeError("invalid opaque fixture array schema")
        for key, base in expected_arrays.items():
            value = arrays[key]
            if not isinstance(value, dict) or set(value) != {"semantic", "dtype", "shape", "c_order_sha256"}:
                raise RuntimeError("invalid opaque fixture array metadata")
            if value["dtype"] != base["dtype"] or value["shape"] != base["shape"]:
                raise RuntimeError("opaque fixture array dtype/shape drift")
            if not isinstance(value["semantic"], str) or not isinstance(value["c_order_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", value["c_order_sha256"]):
                raise RuntimeError("opaque fixture C-order hash is invalid")
    if ids != sorted(ids) or len(set(ids)) != 64:
        raise RuntimeError("opaque fixture ids must be unique canonical lexicographic order")
    if manifest.get("opaque_ids_sha256") != hashlib.sha256("\n".join(ids).encode("ascii")).hexdigest():
        raise RuntimeError("opaque fixture id-list hash mismatch")
    _reject_forbidden_input_metadata(manifest)
    return records


def _fixture_manifest(
    args: argparse.Namespace, protocol: Mapping[str, Any]
) -> tuple[Path, Path, dict[str, Any], list[dict[str, Any]], str]:
    if not args.fixture_manifest or not args.fixture_manifest_sha256 or not args.fixture_root:
        raise RuntimeError(
            "phase-a requires --fixture-manifest, --fixture-manifest-sha256, and --fixture-root"
        )
    manifest_path = Path(args.fixture_manifest).expanduser().absolute()
    root = Path(args.fixture_root).expanduser().absolute()
    _assert_input_only_path(manifest_path)
    _assert_input_only_path(root)
    if manifest_path.parent != root:
        raise RuntimeError("input manifest must be directly anchored in fixture root")
    manifest_bytes, _ = _secure_absolute_bytes(manifest_path)
    if _bytes_sha256(manifest_bytes) != args.fixture_manifest_sha256:
        raise RuntimeError("input-only fixture manifest SHA256 anchor mismatch")
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("invalid input fixture manifest JSON") from error
    if not isinstance(manifest, dict) or manifest_bytes != _ledger_canonical_bytes(manifest):
        raise RuntimeError("input fixture manifest is not canonical JSON")
    records = _opaque_input_records(manifest, protocol)
    pinned = protocol["runtime_pins"].get("fixture_input_manifest_sha256")
    if pinned is None or pinned != args.fixture_manifest_sha256:
        raise RuntimeError("input fixture manifest runtime pin is null or mismatched")
    fixture_names = {Path(str(record["artifact"]["path"])).name for record in records}
    _assert_exact_directory_entries(root, {manifest_path.name, "records"})
    _assert_exact_directory_entries(root / "records", fixture_names)
    return manifest_path, root, manifest, records, args.fixture_manifest_sha256


def _load_input_fixture_bindings(
    args: argparse.Namespace, protocol: Mapping[str, Any]
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, str], dict[str, str]]:
    manifest_path, root, _, records, manifest_hash = _fixture_manifest(args, protocol)
    bindings: dict[str, dict[str, np.ndarray]] = {}
    snapshots = {str(manifest_path): manifest_hash}
    artifact_hashes: dict[str, str] = {}
    for record in records:
        artifact = record["artifact"]
        file_bytes, metadata, path = _secure_relative_bytes(
            root, str(artifact["path"]), expected_parent="records"
        )
        if metadata.st_size != artifact["bytes"] or _bytes_sha256(file_bytes) != artifact["sha256"]:
            raise RuntimeError("input fixture artifact hash/size mismatch")
        arrays = _input_fixture_arrays(file_bytes)
        for key in REQUIRED_INPUT_ARRAYS:
            if _array_c_sha256(arrays[key]) != record["arrays"][key]["c_order_sha256"]:
                raise RuntimeError(f"input fixture array hash mismatch: {key}")
        opaque_id = str(record["opaque_id"])
        if int(arrays["qap_seed"]) != _opaque_qap_seed(opaque_id):
            raise RuntimeError("input fixture opaque seed mismatch")
        bindings[opaque_id] = arrays
        artifact_hashes[opaque_id] = str(artifact["sha256"])
        snapshots[str(path)] = str(artifact["sha256"])
    if len(bindings) != 64:
        raise RuntimeError("input fixture binding coverage drift")
    if len({int(value["qap_seed"]) for value in bindings.values()}) != 64:
        raise RuntimeError("opaque nuisance seeds are not unique")
    return bindings, snapshots, artifact_hashes


def run_phase_a(
    args: argparse.Namespace, *, fixture_builder: FixtureBuilder | None = None
) -> dict[str, Any]:
    if not args.phase_a_dir:
        raise RuntimeError("phase-a requires --phase-a-dir")
    if args.world_size != 2 or args.rank not in {0, 1}:
        raise RuntimeError("phase-a requires rank 0/1 with world-size 2")
    if any((args.fixture_bundle_root, args.output)):
        raise RuntimeError("phase-a refuses every label/target/output argument")
    protocol, config_path, config_sha = _validated_protocol(
        args.config, args.config_sha256
    )
    _validated_runtime_pin_closure(protocol, config_path)
    _validate_environment_lock(protocol, config_path, phase="phase-a")
    assets = _validated_runtime_assets(protocol, config_path, args)
    manifest_path, fixture_root, _, records, fixture_manifest_sha = _fixture_manifest(
        args, protocol
    )
    pins = protocol["runtime_pins"]
    if any(pins.get(key) is None for key in (
        "evaluator_sha256",
        "tests_sha256",
        "fixture_input_manifest_sha256",
        "fixture_label_manifest_sha256",
        "fixture_lock_sha256",
    )):
        raise RuntimeError("Phase A refuses null runtime/fixture pins")
    protocol_instance_id = str(protocol["protocol_instance_id"])
    ledger_root, lifecycle_hashes = _verify_lifecycle_chain(
        args.lifecycle_ledger,
        protocol_instance_id=protocol_instance_id,
        config_sha256=config_sha,
        required_last_state="PHASE_A",
        protocol=protocol,
    )
    del ledger_root
    phase_a_lifecycle_sha = lifecycle_hashes["PHASE_A"]
    if fixture_builder is None:
        fixture_builder = _build_default_fixture_builder(protocol, assets, args)
    output_root = _assert_empty_dir(Path(args.phase_a_dir))
    artifacts_root = output_root / "artifacts"
    renders_root = output_root / "renders"
    artifacts_root.mkdir(parents=True, exist_ok=False)
    renders_root.mkdir(parents=True, exist_ok=False)
    _fsync_directory(output_root)
    frozen_records: list[dict[str, Any]] = []
    shard_records = [
        record for index, record in enumerate(records) if index % args.world_size == args.rank
    ]
    if len(shard_records) != 32:
        raise RuntimeError("Phase-A shard coverage drift")
    for record in shard_records:
        artifact_record = record["artifact"]
        fixture_bytes, fixture_stat, fixture_path = _secure_relative_bytes(
            fixture_root, str(artifact_record["path"]), expected_parent="records"
        )
        _assert_input_only_path(fixture_path)
        if _bytes_sha256(fixture_bytes) != artifact_record["sha256"]:
            raise RuntimeError(f"input-only fixture SHA256 mismatch: {fixture_path.name}")
        if fixture_stat.st_size != artifact_record["bytes"]:
            raise RuntimeError("input-only fixture byte-size mismatch")
        input_arrays = _input_fixture_arrays(fixture_bytes)
        if hashlib.sha256(input_arrays["slot_tiles"].tobytes(order="C")).hexdigest() != record["arrays"]["slot_tiles"]["c_order_sha256"]:
            raise RuntimeError("input-only slot tile C-order hash mismatch")
        opaque_id = str(record["opaque_id"])
        qap_seed = int(input_arrays["qap_seed"])
        if hashlib.sha256(input_arrays["qap_seed"].tobytes(order="C")).hexdigest() != record["arrays"]["qap_seed"]["c_order_sha256"]:
            raise RuntimeError("input-only QAP seed C-order hash mismatch")
        if qap_seed != _opaque_qap_seed(opaque_id):
            raise RuntimeError("opaque QAP seed derivation mismatch")
        derived = fixture_builder(
            opaque_id, input_arrays["slot_tiles"], qap_seed
        )
        fixture = _validate_derived_arrays(derived.arrays)
        graph = build_candidate_graph(fixture)
        stem = opaque_id
        artifact_name = f"{stem}.graph.npz"
        artifact_relative = f"artifacts/{artifact_name}"
        artifact_path = _safe_relative(
            output_root, artifact_relative, expected_parent="artifacts"
        )
        graph_arrays = _graph_arrays(graph, fixture)
        _atomic_bytes(artifact_path, _npy_archive_bytes(graph_arrays))
        render_records: dict[str, dict[str, str]] = {}
        for label, layout_key in (
            ("softcycle", "softcycle_layout"),
            ("qap_w4", "qap_w4_layout"),
            ("qap_w1", "qap_w1_layout"),
        ):
            render_name = f"{stem}__{label}.png"
            render_relative = f"renders/{render_name}"
            render_path = _safe_relative(
                output_root, render_relative, expected_parent="renders"
            )
            render = merge_tiles_numpy(fixture["denoised_tiles"][fixture[layout_key]])
            _atomic_bytes(render_path, _png_bytes(render))
            render_records[label] = {
                "path": render_relative,
                "sha256": _sha256(render_path),
                "layout_sha256": _layout_sha256(fixture[layout_key]),
            }
        frozen_records.append(
            {
                "opaque_id": opaque_id,
                "qap_seed": qap_seed,
                "input_fixture_sha256": str(artifact_record["sha256"]),
                "input_slot_tiles_c_sha256": str(
                    record["arrays"]["slot_tiles"]["c_order_sha256"]
                ),
                "graph_artifact": artifact_relative,
                "graph_artifact_byte_size": artifact_path.stat().st_size,
                "graph_artifact_sha256": _sha256(artifact_path),
                "candidate_edge_count": len(graph.direction),
                "candidate_origin_mask_sha256": hashlib.sha256(
                    graph.origin_mask.tobytes()
                ).hexdigest(),
                "origin_pre_dedup_counts": dict(EXPECTED_PRE_DEDUP_COUNTS),
                "arrays": {
                    key: _array_descriptor(value, key)
                    for key, value in sorted(graph_arrays.items())
                },
                "renders": render_records,
                "derivation_diagnostics": derived.diagnostics,
            }
        )
    # Phase-A postcondition: re-open the immutable input tree and every runtime
    # binding after inference, before the shard manifest becomes durable.
    config_after, _ = _secure_absolute_bytes(config_path)
    script_after, _ = _secure_absolute_bytes(Path(__file__).resolve())
    if _bytes_sha256(config_after) != config_sha or _bytes_sha256(script_after) != protocol["runtime_pins"]["evaluator_sha256"]:
        raise RuntimeError("Phase-A config/evaluator TOCTOU mismatch")
    _validated_runtime_pin_closure(protocol, config_path)
    post_assets = _validated_runtime_assets(protocol, config_path, args)
    if {
        key: value["sha256"] for key, value in sorted(post_assets.items())
    } != {key: value["sha256"] for key, value in sorted(assets.items())}:
        raise RuntimeError("Phase-A runtime asset TOCTOU mismatch")
    _, _, post_input_hashes = _load_input_fixture_bindings(args, protocol)
    for record in frozen_records:
        if post_input_hashes[record["opaque_id"]] != record["input_fixture_sha256"]:
            raise RuntimeError("Phase-A input fixture TOCTOU mismatch")
        _verify_shard_record(output_root, record)
    script_path = Path(__file__).resolve()
    payload = {
        "schema_version": 1,
        "kind": "frozen_candidate_graph_input_only_shard",
        "rank": args.rank,
        "world_size": args.world_size,
        "global_record_count": len(records),
        "config_sha256": config_sha,
        "protocol_instance_id": protocol_instance_id,
        "frozen_contract_sha256": EXPECTED_FROZEN_CONTRACT_SHA256,
        "phase_a_lifecycle_sha256": phase_a_lifecycle_sha,
        "script_sha256": _sha256(script_path),
        "fixture_manifest_sha256": fixture_manifest_sha,
        "fixture_manifest_name": manifest_path.name,
        "runtime_asset_sha256": {
            key: value["sha256"] for key, value in sorted(assets.items())
        },
        "runtime_pin_sha256": {
            key: value
            for key, value in sorted(protocol["runtime_pins"].items())
            if key.endswith("_sha256")
        },
        "record_count": len(frozen_records),
        "records": frozen_records,
        "target_paths_constructed": False,
        "target_files_opened": False,
        "safe_for_submission": False,
    }
    payload = _bind_self_sha256(payload)
    manifest_path_out = output_root / PHASE_A_SHARD_MANIFEST
    _write_plain_json(manifest_path_out, payload)
    _assert_exact_directory_entries(
        output_root, {PHASE_A_SHARD_MANIFEST, "artifacts", "renders"}
    )
    _assert_exact_directory_entries(
        output_root / "artifacts",
        {Path(value["graph_artifact"]).name for value in frozen_records},
    )
    _assert_exact_directory_entries(
        output_root / "renders",
        {
            Path(descriptor["path"]).name
            for value in frozen_records
            for descriptor in value["renders"].values()
        },
    )
    return {
        "phase_a_manifest": str(manifest_path_out),
        "phase_a_envelope_sha256": _sha256(manifest_path_out),
        "record_count": len(frozen_records),
        "target_paths_or_pixels_read": False,
    }


def _validate_derivation_diagnostics(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != {
        "hbt_outside_logits",
        "softcycle",
        "qap",
    }:
        raise RuntimeError("Phase-A derivation diagnostics schema drift")
    hbt = value["hbt_outside_logits"]
    if (
        not isinstance(hbt, dict)
        or set(hbt) != {"dtype", "shape", "c_order_sha256"}
        or hbt.get("dtype") != "float32"
        or hbt.get("shape") != [TILE_COUNT, 4]
        or not isinstance(hbt.get("c_order_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", hbt["c_order_sha256"]) is None
    ):
        raise RuntimeError("Phase-A HBT outside-logit diagnostics drift")
    softcycle = value["softcycle"]
    if (
        not isinstance(softcycle, dict)
        or set(softcycle) != {"accepted_edges", "component_sizes"}
        or type(softcycle.get("accepted_edges")) is not int
        or not isinstance(softcycle.get("component_sizes"), list)
    ):
        raise RuntimeError("Phase-A softcycle diagnostics schema/type drift")
    accepted_edges = softcycle["accepted_edges"]
    component_sizes = softcycle["component_sizes"]
    if (
        accepted_edges < 0
        or accepted_edges > 1104
        or not component_sizes
        or any(type(item) is not int or item < 1 for item in component_sizes)
        or sum(component_sizes) != TILE_COUNT
        or component_sizes != sorted(component_sizes, reverse=True)
        or accepted_edges < TILE_COUNT - len(component_sizes)
    ):
        raise RuntimeError("Phase-A softcycle diagnostics semantic drift")
    qap = value["qap"]
    if not isinstance(qap, dict) or set(qap) != {"qap_w1", "qap_w4"}:
        raise RuntimeError("Phase-A QAP diagnostics coverage drift")
    for label in ("qap_w1", "qap_w4"):
        item = qap[label]
        if (
            not isinstance(item, dict)
            or set(item)
            != {"objective", "relaxed_objective", "restart", "iterations", "converged"}
            or type(item.get("objective")) not in (int, float)
            or type(item.get("relaxed_objective")) not in (int, float)
            or not bool(np.isfinite(float(item["objective"])))
            or not bool(np.isfinite(float(item["relaxed_objective"])))
            or type(item.get("restart")) is not int
            or item["restart"] not in (0, 1)
            or type(item.get("iterations")) is not int
            or item["iterations"] != 25
            or type(item.get("converged")) is not bool
        ):
            raise RuntimeError(f"Phase-A QAP diagnostics drift: {label}")


def _verify_shard_record(
    root: Path, record: Mapping[str, Any]
) -> tuple[dict[str, Any], bytes, dict[str, bytes]]:
    if set(record) != FROZEN_RECORD_KEYS:
        raise RuntimeError("Phase-A shard record schema drift")
    opaque_id = record["opaque_id"]
    if not isinstance(opaque_id, str) or not re.fullmatch(r"[0-9a-f]{32}", opaque_id):
        raise RuntimeError("Phase-A shard opaque id drift")
    if record["qap_seed"] != _opaque_qap_seed(opaque_id):
        raise RuntimeError("Phase-A shard opaque QAP seed drift")
    graph_bytes, graph_stat, _ = _secure_relative_bytes(
        root, str(record["graph_artifact"]), expected_parent="artifacts"
    )
    if _bytes_sha256(graph_bytes) != record["graph_artifact_sha256"] or graph_stat.st_size != record["graph_artifact_byte_size"]:
        raise RuntimeError("Phase-A shard graph hash/size mismatch")
    graph, values = _load_graph_artifact(graph_bytes)
    if len(graph.direction) != record["candidate_edge_count"] or hashlib.sha256(graph.origin_mask.tobytes()).hexdigest() != record["candidate_origin_mask_sha256"]:
        raise RuntimeError("Phase-A shard graph semantic hash mismatch")
    if record.get("origin_pre_dedup_counts") != EXPECTED_PRE_DEDUP_COUNTS:
        raise RuntimeError("Phase-A shard origin pre-dedup counts drift")
    descriptors = record.get("arrays")
    if not isinstance(descriptors, dict) or set(descriptors) != set(values):
        raise RuntimeError("Phase-A shard array descriptor coverage drift")
    for key, value in values.items():
        descriptor = descriptors[key]
        if not isinstance(descriptor, dict) or set(descriptor) != {
            "semantic",
            "dtype",
            "shape",
            "c_order_sha256",
        }:
            raise RuntimeError("Phase-A shard array descriptor schema drift")
        if descriptor != _array_descriptor(value, key):
            raise RuntimeError(f"Phase-A shard array descriptor mismatch: {key}")
    renders = record.get("renders")
    if not isinstance(renders, dict) or set(renders) != {"softcycle", "qap_w4", "qap_w1"}:
        raise RuntimeError("Phase-A shard render coverage drift")
    render_bytes: dict[str, bytes] = {}
    for label, descriptor in renders.items():
        if not isinstance(descriptor, dict) or set(descriptor) != {"path", "sha256", "layout_sha256"}:
            raise RuntimeError("Phase-A shard render schema drift")
        payload, _, _ = _secure_relative_bytes(
            root, str(descriptor["path"]), expected_parent="renders"
        )
        if _bytes_sha256(payload) != descriptor["sha256"]:
            raise RuntimeError("Phase-A shard render hash mismatch")
        layout_key = {
            "softcycle": "softcycle_layout",
            "qap_w4": "qap_w4_layout",
            "qap_w1": "qap_w1_layout",
        }[label]
        layout = values[layout_key]
        if descriptor["layout_sha256"] != _layout_sha256(layout):
            raise RuntimeError("Phase-A shard render layout binding mismatch")
        expected_render = merge_tiles_numpy(values["denoised_tiles"][layout])
        if payload != _png_bytes(expected_render):
            raise RuntimeError("Phase-A shard frozen render semantic mismatch")
        render_bytes[label] = payload
    _validate_derivation_diagnostics(record.get("derivation_diagnostics"))
    return dict(record), graph_bytes, render_bytes


def _load_phase_a_shard(
    root: Path,
    anchor: str,
    *,
    config_sha: str,
    protocol_instance_id: str,
) -> tuple[dict[str, Any], list[tuple[dict[str, Any], bytes, dict[str, bytes]]]]:
    manifest_path = root / PHASE_A_SHARD_MANIFEST
    payload = _load_self_manifest(manifest_path, anchor)
    expected_top = {
        "schema_version",
        "kind",
        "rank",
        "world_size",
        "global_record_count",
        "config_sha256",
        "protocol_instance_id",
        "frozen_contract_sha256",
        "phase_a_lifecycle_sha256",
        "script_sha256",
        "fixture_manifest_sha256",
        "fixture_manifest_name",
        "runtime_asset_sha256",
        "runtime_pin_sha256",
        "record_count",
        "records",
        "target_paths_constructed",
        "target_files_opened",
        "safe_for_submission",
        "self_sha256",
    }
    if set(payload) != expected_top:
        raise RuntimeError("Phase-A shard manifest schema drift")
    _verify_self_sha256(payload)
    rank = payload.get("rank")
    expected_values = {
        "schema_version": 1,
        "kind": "frozen_candidate_graph_input_only_shard",
        "world_size": 2,
        "global_record_count": 64,
        "config_sha256": config_sha,
        "protocol_instance_id": protocol_instance_id,
        "frozen_contract_sha256": EXPECTED_FROZEN_CONTRACT_SHA256,
        "record_count": 32,
        "target_paths_constructed": False,
        "target_files_opened": False,
        "safe_for_submission": False,
    }
    for key, expected in expected_values.items():
        if payload.get(key) != expected:
            raise RuntimeError(f"Phase-A shard invariant drift: {key}")
    if rank not in {0, 1}:
        raise RuntimeError("Phase-A shard rank drift")
    script_bytes, _ = _secure_absolute_bytes(Path(__file__).resolve())
    if payload.get("script_sha256") != _bytes_sha256(script_bytes):
        raise RuntimeError("Phase-A shard evaluator hash drift")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != 32:
        raise RuntimeError("Phase-A shard record count drift")
    verified = [_verify_shard_record(root, record) for record in records]
    ids = [value[0]["opaque_id"] for value in verified]
    if ids != sorted(ids) or len(set(ids)) != 32:
        raise RuntimeError("Phase-A shard opaque order/uniqueness drift")
    _assert_exact_directory_entries(
        root, {PHASE_A_SHARD_MANIFEST, "artifacts", "renders"}
    )
    _assert_exact_directory_entries(
        root / "artifacts", {Path(value[0]["graph_artifact"]).name for value in verified}
    )
    _assert_exact_directory_entries(
        root / "renders",
        {
            Path(descriptor["path"]).name
            for value in verified
            for descriptor in value[0]["renders"].values()
        },
    )
    return payload, verified


def _validate_shard_coverage(
    shards: Sequence[tuple[dict[str, Any], Sequence[tuple[dict[str, Any], bytes, dict[str, bytes]]]]]
) -> list[tuple[dict[str, Any], bytes, dict[str, bytes]]]:
    if len(shards) != 2 or {int(value[0]["rank"]) for value in shards} != {0, 1}:
        raise RuntimeError("finalization requires exactly rank-0 and rank-1 shards")
    by_id: dict[str, tuple[int, tuple[dict[str, Any], bytes, dict[str, bytes]]]] = {}
    for payload, records in shards:
        rank = int(payload["rank"])
        for record in records:
            opaque_id = str(record[0]["opaque_id"])
            if opaque_id in by_id:
                raise RuntimeError("duplicate opaque id across Phase-A shards")
            by_id[opaque_id] = (rank, record)
    if len(by_id) != 64:
        raise RuntimeError("missing Phase-A shard coverage")
    ordered = sorted(by_id)
    for index, opaque_id in enumerate(ordered):
        if by_id[opaque_id][0] != index % 2:
            raise RuntimeError("Phase-A shard modulo assignment drift")
    if len({int(by_id[opaque_id][1][0]["qap_seed"]) for opaque_id in ordered}) != 64:
        raise RuntimeError("Phase-A opaque nuisance seeds are not unique")
    return [by_id[opaque_id][1] for opaque_id in ordered]


def _validate_shard_common_bindings(
    shards: Sequence[tuple[Mapping[str, Any], Any]]
) -> None:
    common_keys = (
        "config_sha256",
        "protocol_instance_id",
        "frozen_contract_sha256",
        "phase_a_lifecycle_sha256",
        "script_sha256",
        "fixture_manifest_sha256",
        "fixture_manifest_name",
        "runtime_asset_sha256",
        "runtime_pin_sha256",
    )
    if len(shards) != 2:
        raise RuntimeError("two Phase-A shard bindings are required")
    for key in common_keys:
        if len({_canonical_sha256(value[0][key]) for value in shards}) != 1:
            raise RuntimeError(f"mixed Phase-A shard binding: {key}")


def run_finalize_phase_a(args: argparse.Namespace) -> dict[str, Any]:
    if not args.finalized_phase_a_dir:
        raise RuntimeError("finalize-phase-a requires --finalized-phase-a-dir")
    if not args.phase_a_dirs or not args.phase_a_envelope_sha256s or len(args.phase_a_dirs) != 2 or len(args.phase_a_envelope_sha256s) != 2:
        raise RuntimeError("finalize-phase-a requires two shard dirs and two anchors")
    if any((args.fixture_manifest, args.fixture_root, args.fixture_bundle_root, args.output)):
        raise RuntimeError("finalize-phase-a refuses fixture/label/output roots")
    protocol, config_path, config_sha = _validated_protocol(
        args.config, args.config_sha256
    )
    _validated_runtime_pin_closure(protocol, config_path)
    _validate_environment_lock(protocol, config_path, phase="phase-a")
    instance = str(protocol["protocol_instance_id"])
    _, lifecycle = _verify_lifecycle_chain(
        args.lifecycle_ledger,
        protocol_instance_id=instance,
        config_sha256=config_sha,
        required_last_state="PHASE_A",
        protocol=protocol,
    )
    shards = [
        _load_phase_a_shard(
            Path(directory).expanduser().absolute(),
            anchor,
            config_sha=config_sha,
            protocol_instance_id=instance,
        )
        for directory, anchor in zip(
            args.phase_a_dirs, args.phase_a_envelope_sha256s, strict=True
        )
    ]
    if [int(value[0]["rank"]) for value in shards] != [0, 1]:
        raise RuntimeError("Phase-A shard arguments must be in canonical rank 0,1 order")
    _validate_shard_common_bindings(shards)
    if shards[0][0]["phase_a_lifecycle_sha256"] != lifecycle["PHASE_A"]:
        raise RuntimeError("Phase-A shard lifecycle binding mismatch")
    records = _validate_shard_coverage(shards)
    output_root = _assert_empty_dir(Path(args.finalized_phase_a_dir))
    (output_root / "artifacts").mkdir(parents=True, exist_ok=False)
    (output_root / "renders").mkdir(parents=True, exist_ok=False)
    finalized_records: list[dict[str, Any]] = []
    for record, graph_bytes, render_payloads in records:
        graph_path = _safe_relative(
            output_root, str(record["graph_artifact"]), expected_parent="artifacts"
        )
        _atomic_bytes(graph_path, graph_bytes)
        for label, payload in render_payloads.items():
            render_path = _safe_relative(
                output_root,
                str(record["renders"][label]["path"]),
                expected_parent="renders",
            )
            _atomic_bytes(render_path, payload)
        finalized_records.append(record)
        _verify_shard_record(output_root, record)
    script_bytes, _ = _secure_absolute_bytes(Path(__file__).resolve())
    payload = {
        "schema_version": 1,
        "kind": "frozen_candidate_graph_input_only",
        "config_sha256": config_sha,
        "protocol_instance_id": instance,
        "frozen_contract_sha256": EXPECTED_FROZEN_CONTRACT_SHA256,
        "phase_a_lifecycle_sha256": lifecycle["PHASE_A"],
        "script_sha256": _bytes_sha256(script_bytes),
        "fixture_manifest_sha256": shards[0][0]["fixture_manifest_sha256"],
        "fixture_manifest_name": shards[0][0]["fixture_manifest_name"],
        "runtime_asset_sha256": shards[0][0]["runtime_asset_sha256"],
        "runtime_pin_sha256": shards[0][0]["runtime_pin_sha256"],
        "shard_envelope_sha256s": list(args.phase_a_envelope_sha256s),
        "record_count": 64,
        "records": finalized_records,
        "target_paths_constructed": False,
        "target_files_opened": False,
        "safe_for_submission": False,
    }
    payload = _bind_self_sha256(payload)
    manifest_path = output_root / PHASE_A_MANIFEST
    _write_plain_json(manifest_path, payload)
    _assert_exact_directory_entries(
        output_root, {PHASE_A_MANIFEST, "artifacts", "renders"}
    )
    _assert_exact_directory_entries(
        output_root / "artifacts",
        {Path(value["graph_artifact"]).name for value in finalized_records},
    )
    _assert_exact_directory_entries(
        output_root / "renders",
        {
            Path(descriptor["path"]).name
            for value in finalized_records
            for descriptor in value["renders"].values()
        },
    )
    return {
        "phase_a_manifest": str(manifest_path),
        "phase_a_envelope_sha256": _sha256(manifest_path),
        "record_count": 64,
        "target_paths_or_pixels_read": False,
    }


def _verify_phase_a(
    args: argparse.Namespace, config_sha: str, protocol_instance_id: str
) -> tuple[Path, dict[str, Any], dict[str, str]]:
    if not args.phase_a_dir or not args.phase_a_envelope_sha256:
        raise RuntimeError("phase-b requires Phase-A directory and SHA256 anchor")
    root = Path(args.phase_a_dir).expanduser().absolute()
    manifest_path = root / PHASE_A_MANIFEST
    payload = _load_self_manifest(manifest_path, args.phase_a_envelope_sha256)
    expected_top = {
        "schema_version",
        "kind",
        "config_sha256",
        "protocol_instance_id",
        "frozen_contract_sha256",
        "phase_a_lifecycle_sha256",
        "script_sha256",
        "fixture_manifest_sha256",
        "fixture_manifest_name",
        "runtime_asset_sha256",
        "runtime_pin_sha256",
        "shard_envelope_sha256s",
        "record_count",
        "records",
        "target_paths_constructed",
        "target_files_opened",
        "safe_for_submission",
        "self_sha256",
    }
    if set(payload) != expected_top:
        raise RuntimeError("Phase-A manifest has opaque/schema-drift fields")
    _verify_self_sha256(payload)
    expected_values = {
        "schema_version": 1,
        "kind": "frozen_candidate_graph_input_only",
        "config_sha256": config_sha,
        "protocol_instance_id": protocol_instance_id,
        "frozen_contract_sha256": EXPECTED_FROZEN_CONTRACT_SHA256,
        "record_count": 64,
        "target_paths_constructed": False,
        "target_files_opened": False,
        "safe_for_submission": False,
    }
    for key, expected in expected_values.items():
        if payload.get(key) != expected:
            raise RuntimeError(f"Phase-A invariant drift: {key}")
    anchors = payload.get("shard_envelope_sha256s")
    if not isinstance(anchors, list) or len(anchors) != 2 or any(
        not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
        for value in anchors
    ):
        raise RuntimeError("finalized Phase-A shard anchor coverage drift")
    script_bytes, _ = _secure_absolute_bytes(Path(__file__).resolve())
    if payload.get("script_sha256") != _bytes_sha256(script_bytes):
        raise RuntimeError("Phase-A evaluator code SHA256 drift")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != 64:
        raise RuntimeError("Phase-A record coverage drift")
    snapshots: dict[str, str] = {
        str(manifest_path): str(args.phase_a_envelope_sha256)
    }
    expected_record_keys = FROZEN_RECORD_KEYS
    ids: list[str] = []
    artifact_names: set[str] = set()
    render_names: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != expected_record_keys:
            raise RuntimeError(f"Phase-A record schema drift at {index}")
        opaque_id = record["opaque_id"]
        if not isinstance(opaque_id, str) or not re.fullmatch(r"[0-9a-f]{32}", opaque_id):
            raise RuntimeError("Phase-A opaque id drift")
        ids.append(opaque_id)
        if record["qap_seed"] != _opaque_qap_seed(opaque_id):
            raise RuntimeError("Phase-A opaque QAP seed drift")
        artifact_bytes, artifact_stat, artifact_path = _secure_relative_bytes(
            root, str(record["graph_artifact"]), expected_parent="artifacts"
        )
        artifact_names.add(artifact_path.name)
        if _bytes_sha256(artifact_bytes) != record["graph_artifact_sha256"]:
            raise RuntimeError("Phase-A graph artifact tamper detected")
        if artifact_stat.st_size != record["graph_artifact_byte_size"]:
            raise RuntimeError("Phase-A graph artifact size mismatch")
        snapshots[str(artifact_path)] = str(record["graph_artifact_sha256"])
        graph, graph_values = _load_graph_artifact(artifact_bytes)
        if len(graph.direction) != record["candidate_edge_count"]:
            raise RuntimeError("Phase-A candidate edge count mismatch")
        if hashlib.sha256(graph.origin_mask.tobytes()).hexdigest() != record["candidate_origin_mask_sha256"]:
            raise RuntimeError("Phase-A origin mask hash mismatch")
        if record.get("origin_pre_dedup_counts") != EXPECTED_PRE_DEDUP_COUNTS:
            raise RuntimeError("Phase-A origin pre-dedup counts drift")
        descriptors = record.get("arrays")
        if not isinstance(descriptors, dict) or set(descriptors) != set(graph_values):
            raise RuntimeError("Phase-A array descriptor coverage drift")
        for key, value in graph_values.items():
            if descriptors.get(key) != _array_descriptor(value, key):
                raise RuntimeError(f"Phase-A array descriptor mismatch: {key}")
        renders = record.get("renders")
        if not isinstance(renders, dict) or set(renders) != {"softcycle", "qap_w4", "qap_w1"}:
            raise RuntimeError("Phase-A render record drift")
        for render_label, render_record in renders.items():
            if not isinstance(render_record, dict) or set(render_record) != {
                "path",
                "sha256",
                "layout_sha256",
            }:
                raise RuntimeError("Phase-A render schema drift")
            render_bytes, _, render_path = _secure_relative_bytes(
                root, str(render_record["path"]), expected_parent="renders"
            )
            render_names.add(render_path.name)
            if _bytes_sha256(render_bytes) != render_record["sha256"]:
                raise RuntimeError("Phase-A render tamper detected")
            layout_key = {
                "softcycle": "softcycle_layout",
                "qap_w4": "qap_w4_layout",
                "qap_w1": "qap_w1_layout",
            }[render_label]
            layout = graph_values[layout_key]
            if render_record["layout_sha256"] != _layout_sha256(layout) or render_bytes != _png_bytes(
                merge_tiles_numpy(graph_values["denoised_tiles"][layout])
            ):
                raise RuntimeError("Phase-A render semantic binding mismatch")
            snapshots[str(render_path)] = str(render_record["sha256"])
    if ids != sorted(ids) or len(set(ids)) != 64:
        raise RuntimeError("Phase-A opaque-id coverage/order drift")
    _assert_exact_directory_entries(root, {PHASE_A_MANIFEST, "artifacts", "renders"})
    _assert_exact_directory_entries(root / "artifacts", artifact_names)
    _assert_exact_directory_entries(root / "renders", render_names)
    return root, payload, snapshots


def evaluate_continuation_gate(panel_summaries: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    if set(panel_summaries) != set(PANELS):
        raise RuntimeError("continuation gate requires both exact panels")
    numeric_keys = {
        "mean_union_true_edge_recall",
        "median_largest_connected_component",
        "mean_beam_qap_adjacency_delta",
        "mean_beam_qap_ssim_delta",
    }
    guards: dict[str, dict[str, bool]] = {}
    for panel in PANELS:
        summary = panel_summaries[panel]
        if not numeric_keys.issubset(summary):
            raise RuntimeError(f"panel summary is incomplete: {panel}")
        values = {key: float(summary[key]) for key in numeric_keys}
        if not all(np.isfinite(value) for value in values.values()):
            raise RuntimeError(f"non-finite gate metric: {panel}")
        guards[panel] = {
            "union_true_edge_recall_ge_0.65": values["mean_union_true_edge_recall"] >= 0.65,
            "median_lcc_ge_128": values["median_largest_connected_component"] >= 128.0,
            "beam_qap_adjacency_delta_nonnegative": values["mean_beam_qap_adjacency_delta"] >= 0.0,
            "beam_qap_ssim_delta_nonnegative": values["mean_beam_qap_ssim_delta"] >= 0.0,
        }
    macro_adjacency = float(
        np.mean([float(panel_summaries[panel]["mean_beam_qap_adjacency_delta"]) for panel in PANELS])
    )
    macro_ssim = float(
        np.mean([float(panel_summaries[panel]["mean_beam_qap_ssim_delta"]) for panel in PANELS])
    )
    major = macro_adjacency >= 0.10 or macro_ssim >= 0.02
    all_panel_guards = all(all(value.values()) for value in guards.values())
    passed = bool(all_panel_guards and major)
    return {
        "panel_guards": guards,
        "all_panel_guards_passed": all_panel_guards,
        "balanced_panel_macro_adjacency_delta": macro_adjacency,
        "balanced_panel_macro_ssim_delta": macro_ssim,
        "major_gain_adjacency_ge_0.10": macro_adjacency >= 0.10,
        "major_gain_ssim_ge_0.02": macro_ssim >= 0.02,
        "major_gain_or_passed": major,
        "continue_to_cycle_factor_synchronizer": passed,
        "safe_for_submission": False,
    }


def _label_records_after_marker(
    args: argparse.Namespace,
    protocol: Mapping[str, Any],
    marker_path: Path,
) -> tuple[Path, Path, list[dict[str, Any]], dict[str, str], bytes]:
    if not marker_path.is_file() or _sha256(marker_path) == "":
        raise RuntimeError("durable target-access marker is absent")
    # These are intentionally the first reads of label-relative pins and the
    # first Path constructions involving the physically separated label root.
    bundle_root = Path(args.fixture_bundle_root).expanduser().absolute()
    pins = protocol["runtime_pins"]
    relative_manifest = Path(str(pins["fixture_label_manifest_relative_path"]))
    if (
        relative_manifest.is_absolute()
        or relative_manifest.parts
        != ("fixture_label", "fixture_label_manifest.json")
    ):
        raise RuntimeError("pinned label-manifest relative path drift")
    label_root = bundle_root / relative_manifest.parts[0]
    label_manifest_path = bundle_root / relative_manifest
    secret_path = label_root / "FIXTURE_MASTER_SECRET.bin"
    if label_manifest_path.parent != label_root:
        raise RuntimeError("label manifest/secret must be directly anchored in label root")
    manifest_bytes, _ = _secure_absolute_bytes(label_manifest_path)
    pinned_label_sha256 = pins.get("fixture_label_manifest_sha256")
    if _bytes_sha256(manifest_bytes) != pinned_label_sha256:
        raise RuntimeError("label manifest SHA256 anchor mismatch")
    if not isinstance(pinned_label_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", pinned_label_sha256
    ):
        raise RuntimeError("label manifest runtime pin is null or mismatched")
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("invalid label manifest JSON") from error
    if not isinstance(manifest, dict) or manifest_bytes != _ledger_canonical_bytes(manifest):
        raise RuntimeError("label manifest is not canonical JSON")
    expected_top = {
        "schema_version",
        "created_utc",
        "kind",
        *COMMON_MANIFEST_SHA_FIELDS,
        "fixture_input_manifest_sha256",
        "record_count",
        "opaque_ids_sha256",
        "canonical_record_order",
        "hidden_panel_counts",
        "master_secret",
        "records",
    }
    if set(manifest) != expected_top:
        raise RuntimeError("label manifest schema drift")
    if manifest.get("schema_version") != 1 or manifest.get("kind") != "candidate_graph_oracle_fixture_labels":
        raise RuntimeError("unexpected label manifest identity")
    if manifest.get("frozen_contract_sha256") != EXPECTED_FROZEN_CONTRACT_SHA256:
        raise RuntimeError("label manifest frozen-contract binding mismatch")
    if manifest.get("record_count") != 64:
        raise RuntimeError("label record count drift")
    if manifest.get("protocol_instance_id") != protocol["protocol_instance_id"] or manifest.get("canonical_record_order") != "ascending opaque_id":
        raise RuntimeError("label protocol/order drift")
    if manifest.get("hidden_panel_counts") != {panel: 32 for panel in PANELS}:
        raise RuntimeError("label hidden-panel count drift")
    for manifest_key, pin_key in (
        ("evaluator_sha256", "evaluator_sha256"),
        ("tests_sha256", "tests_sha256"),
        ("environment_lock_sha256", "environment_lock_sha256"),
        ("fixture_builder_sha256", "fixture_builder_sha256"),
        ("fixture_builder_tests_sha256", "fixture_builder_tests_sha256"),
        ("pin_finalizer_sha256", "pin_finalizer_sha256"),
        ("lifecycle_tool_sha256", "lifecycle_tool_sha256"),
        ("result_verifier_sha256", "result_verifier_sha256"),
        ("phase_a_runner_sha256", "phase_a_runner_sha256"),
        ("phase_a_kernel_metadata_sha256", "phase_a_kernel_metadata_sha256"),
        ("phase_a_launcher_sha256", "phase_a_launcher_sha256"),
        ("phase_b_runner_sha256", "phase_b_runner_sha256"),
    ):
        if manifest.get(manifest_key) != pins.get(pin_key):
            raise RuntimeError(f"label runtime provenance drift: {manifest_key}")
    if manifest.get("fixture_input_manifest_sha256") != pins.get("fixture_input_manifest_sha256"):
        raise RuntimeError("label-to-input manifest crosslink mismatch")
    selection = protocol["frozen_contract"]["source_selection"]
    repository = Path(args.config).expanduser().resolve().parent.parent
    all_names = source_names_for_split(
        str(selection["split"]),
        manifest_path=repository / str(selection["authoritative_manifest"]),
        quarantine_path=repository / str(selection["quarantine"]),
        audit_exclusion_path=repository
        / str(protocol["frozen_contract"]["sealed_sets"]["audit_exclusion_ledger"]),
    )
    names = all_names[int(selection["offset"]) : int(selection["offset"]) + int(selection["count"])]
    if len(names) != 32 or _names_sha256(names) != EXPECTED_NAMES_SHA256:
        raise RuntimeError("authoritative label source-name slice drift")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != 64:
        raise RuntimeError("label manifest requires 64 records")
    expected_record_keys = {
        "opaque_id",
        "source_name",
        "panel",
        "panel_seed",
        "target_file_sha256",
        "artifact",
        "arrays",
    }
    ids: list[str] = []
    coverage: set[tuple[int, str]] = set()
    snapshots = {str(label_manifest_path): pinned_label_sha256}
    for record in records:
        if not isinstance(record, dict) or set(record) != expected_record_keys:
            raise RuntimeError("label record schema drift")
        opaque_id = record["opaque_id"]
        if not isinstance(opaque_id, str) or not re.fullmatch(r"[0-9a-f]{32}", opaque_id):
            raise RuntimeError("label opaque id is invalid")
        ids.append(opaque_id)
        source_name = record["source_name"]
        panel = record["panel"]
        if source_name not in names or panel not in PANELS:
            raise RuntimeError("label source/panel identity drift")
        source_index = names.index(source_name)
        if not isinstance(record["panel_seed"], int) or not 0 <= record["panel_seed"] < 2**64:
            raise RuntimeError("label panel seed is invalid")
        coverage.add((source_index, panel))
        if not isinstance(record["target_file_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", record["target_file_sha256"]):
            raise RuntimeError("label target-file hash metadata is invalid")
        artifact = record["artifact"]
        if not isinstance(artifact, dict) or set(artifact) != {"path", "bytes", "sha256"}:
            raise RuntimeError("label artifact descriptor drift")
        label_bytes, label_stat, label_path = _secure_relative_bytes(
            label_root, str(artifact["path"]), expected_parent="records"
        )
        if _bytes_sha256(label_bytes) != artifact["sha256"] or label_stat.st_size != artifact["bytes"]:
            raise RuntimeError("label artifact hash/size mismatch")
        arrays = record["arrays"]
        expected_arrays = {
            "opaque_slot_permutation": ("int32", [576]),
            "composed_slot_to_target": ("int32", [576]),
            "clean_target_rgb": ("uint8", [480, 480, 3]),
        }
        if not isinstance(arrays, dict) or set(arrays) != set(expected_arrays):
            raise RuntimeError("label array descriptor coverage drift")
        for key, (dtype, shape) in expected_arrays.items():
            descriptor = arrays[key]
            if not isinstance(descriptor, dict) or set(descriptor) != {"semantic", "dtype", "shape", "c_order_sha256"}:
                raise RuntimeError("label array descriptor schema drift")
            if descriptor["dtype"] != dtype or descriptor["shape"] != shape or not re.fullmatch(r"[0-9a-f]{64}", str(descriptor["c_order_sha256"])):
                raise RuntimeError("label array descriptor value drift")
        snapshots[str(label_path)] = str(artifact["sha256"])
    if ids != sorted(ids) or len(set(ids)) != 64:
        raise RuntimeError("label opaque ids must be unique canonical order")
    if coverage != {(index, panel) for index in range(32) for panel in PANELS}:
        raise RuntimeError("label source-panel coverage is incomplete")
    ids_sha = hashlib.sha256("\n".join(ids).encode("ascii")).hexdigest()
    if manifest.get("opaque_ids_sha256") != ids_sha:
        raise RuntimeError("label opaque-id list hash mismatch")
    secret_descriptor = manifest.get("master_secret")
    if not isinstance(secret_descriptor, dict) or set(secret_descriptor) != {"path", "bytes", "sha256", "mode"}:
        raise RuntimeError("label master-secret descriptor drift")
    if secret_descriptor["path"] != secret_path.name or secret_descriptor["bytes"] != 32 or secret_descriptor["mode"] != "0600":
        raise RuntimeError("label master-secret metadata drift")
    try:
        secret_bytes, secret_stat = _secure_absolute_bytes(secret_path)
    except Exception:
        raise RuntimeError("label-only secret integrity failure") from None
    if len(secret_bytes) != 32 or _bytes_sha256(secret_bytes) != secret_descriptor["sha256"]:
        raise RuntimeError("label master-secret integrity mismatch")
    if stat.S_IMODE(secret_stat.st_mode) != 0o600:
        raise RuntimeError("label master-secret mode must be 0600")
    snapshots[str(secret_path)] = str(secret_descriptor["sha256"])
    record_names = {Path(str(record["artifact"]["path"])).name for record in records}
    _assert_exact_directory_entries(
        label_root, {label_manifest_path.name, "records", secret_path.name}
    )
    _assert_exact_directory_entries(label_root / "records", record_names)
    enriched_records = [
        {**record, "_source_index": names.index(str(record["source_name"]))}
        for record in records
    ]
    return label_manifest_path, label_root, enriched_records, snapshots, secret_bytes


def _mean(values: Sequence[float], *, name: str) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) != 32 or not np.all(np.isfinite(array)):
        raise RuntimeError(f"invalid panel aggregate values: {name}")
    return float(np.mean(array))


def _panel_summaries(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for panel in PANELS:
        selected = [record for record in records if record["panel"] == panel]
        if len(selected) != 32:
            raise RuntimeError(f"panel record count drift: {panel}")
        result[panel] = {
            "record_count": 32,
            "mean_union_true_edge_recall": _mean(
                [float(value["candidate_recall"]["unique_true_edge_recall"]) for value in selected],
                name=f"{panel}.recall",
            ),
            "median_largest_connected_component": float(
                np.median(
                    [int(value["components"]["largest_connected_component"]) for value in selected]
                )
            ),
            "mean_beam_qap_adjacency_delta": _mean(
                [float(value["paired_delta"]["combined_adjacency"]) for value in selected],
                name=f"{panel}.adjacency_delta",
            ),
            "mean_beam_qap_ssim_delta": _mean(
                [float(value["paired_delta"]["rgb_ssim"]) for value in selected],
                name=f"{panel}.ssim_delta",
            ),
        }
    return result


def _origin_histogram(graph: CandidateGraph) -> dict[str, int]:
    return {
        key: int(np.count_nonzero(graph.origin_mask.astype(np.int64) & bit))
        for key, bit in ORIGIN_BITS.items()
    }


def _array_c_sha256(values: np.ndarray) -> str:
    array = np.asarray(values)
    contiguous = array if array.flags.c_contiguous else np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def _array_descriptor(values: np.ndarray, semantic: str) -> dict[str, Any]:
    array = np.asarray(values)
    contiguous = array if array.flags.c_contiguous else np.ascontiguousarray(array)
    return {
        "semantic": semantic,
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
        "c_order_sha256": _array_c_sha256(contiguous),
    }


def recompute_fixture_binding_after_marker(
    *,
    secret: bytes,
    opaque_id: str,
    label_record: Mapping[str, Any],
    labels: Mapping[str, np.ndarray],
    input_arrays: Mapping[str, np.ndarray],
    phase_record: Mapping[str, Any],
    master_seed: int,
) -> dict[str, Any]:
    """Independently reconstruct the exact opaque fixture from label-only data."""

    if len(secret) != 32:
        raise RuntimeError("label-only secret length drift")
    source_name = str(label_record["source_name"])
    panel = str(label_record["panel"])

    def material(prefix: str) -> bytes:
        message = f"{prefix}:{source_name}:{panel}".encode("utf-8")
        return hmac.new(secret, message, hashlib.sha256).digest()

    recomputed_id = material("id")[:16].hex()
    if not hmac.compare_digest(recomputed_id, opaque_id):
        raise RuntimeError("label-only opaque-id recomputation mismatch")
    shuffle_seed = int.from_bytes(material("shuffle")[:8], "big", signed=False)
    permutation = (
        np.random.Generator(np.random.PCG64(shuffle_seed))
        .permutation(TILE_COUNT)
        .astype(np.int32)
    )
    if not np.array_equal(permutation, labels["opaque_slot_permutation"]):
        raise RuntimeError("label-only opaque permutation recomputation mismatch")
    expected_panel_seed = per_source_seed(
        master_seed,
        f"candidate-graph-oracle-{panel}",
        source_name,
        0,
    )
    if int(label_record["panel_seed"]) != expected_panel_seed:
        raise RuntimeError("label panel seed recomputation mismatch")
    exact = make_exact_panel(
        np.ascontiguousarray(labels["clean_target_rgb"]),
        panel=panel,
        seed=expected_panel_seed,
    )
    exact_tiles = _validate_tiles(np.asarray(exact.slot_tiles))
    exact_truth = validate_permutation(
        np.asarray(exact.slot_to_target), name="recomputed_exact_slot_to_target"
    )
    opaque_tiles = np.ascontiguousarray(exact_tiles[permutation])
    composed_truth = validate_permutation(
        exact_truth[permutation], name="recomputed_composed_slot_to_target"
    )
    if not np.array_equal(opaque_tiles, input_arrays["slot_tiles"]):
        raise RuntimeError("label-only decoded input tile recomputation mismatch")
    if _array_c_sha256(opaque_tiles) != phase_record["input_slot_tiles_c_sha256"]:
        raise RuntimeError("label-only opaque tile recomputation mismatch")
    if not np.array_equal(composed_truth, labels["composed_slot_to_target"]):
        raise RuntimeError("label-only composed truth recomputation mismatch")
    for key in REQUIRED_LABEL_ARRAYS:
        expected_hash = label_record["arrays"][key]["c_order_sha256"]
        if _array_c_sha256(labels[key]) != expected_hash:
            raise RuntimeError(f"label array C-order hash mismatch: {key}")
    if int(input_arrays["qap_seed"]) != _opaque_qap_seed(opaque_id) or int(phase_record["qap_seed"]) != _opaque_qap_seed(opaque_id):
        raise RuntimeError("opaque nuisance seed recomputation mismatch")
    validate_truth_geometry(composed_truth)
    return {
        "opaque_id_recomputed": True,
        "opaque_permutation_recomputed": True,
        "panel_seed_recomputed": True,
        "opaque_slot_tiles_recomputed": True,
        "composed_truth_recomputed": True,
        "opaque_qap_seed_recomputed": True,
        "truth_geometry_verified": True,
    }


def run_phase_b(args: argparse.Namespace) -> dict[str, Any]:
    if not args.output:
        raise RuntimeError("phase-b requires --output")
    if not args.fixture_bundle_root:
        raise RuntimeError("phase-b requires one opaque fixture-bundle-root string")
    if not args.fixture_manifest or not args.fixture_manifest_sha256 or not args.fixture_root:
        raise RuntimeError("phase-b requires immutable input fixture manifest/hash/root")
    protocol, config_path, config_sha = _validated_protocol(
        args.config, args.config_sha256
    )
    _validated_runtime_pin_closure(
        protocol, config_path, include_label_fixture_pin=False
    )
    _validate_environment_lock(protocol, config_path, phase="phase-b")
    _validated_runtime_assets(protocol, config_path, args)
    protocol_instance_id = str(protocol["protocol_instance_id"])
    phase_a_root, phase_a_payload, phase_a_snapshots = _verify_phase_a(
        args, config_sha, protocol_instance_id
    )
    expected_runtime_assets = {
        key: str(protocol["frozen_contract"]["assets"][key]["sha256"])
        for key in ("denoiser", "hbt")
    }
    if phase_a_payload["runtime_asset_sha256"] != expected_runtime_assets:
        raise RuntimeError("Phase-A runtime asset binding drift")
    input_bindings, input_snapshots, input_artifact_hashes = _load_input_fixture_bindings(
        args, protocol
    )
    ledger_root, lifecycle_hashes = _verify_lifecycle_chain(
        args.lifecycle_ledger,
        protocol_instance_id=protocol_instance_id,
        config_sha256=config_sha,
        required_last_state="LABEL_ACCESS",
        protocol=protocol,
    )
    if lifecycle_hashes["PHASE_A"] != phase_a_payload["phase_a_lifecycle_sha256"]:
        raise RuntimeError("Phase-A lifecycle marker binding mismatch")
    del ledger_root
    label_access_lifecycle_sha = lifecycle_hashes["LABEL_ACCESS"]
    output_root = _assert_empty_dir(Path(args.output))
    artifacts_root = output_root / "artifacts"
    renders_root = output_root / "renders"
    artifacts_root.mkdir(parents=True, exist_ok=False)
    renders_root.mkdir(parents=True, exist_ok=False)
    _fsync_directory(output_root)
    marker_payload = {
        "schema_version": 1,
        "kind": "candidate_graph_target_access_started",
        "config_sha256": config_sha,
        "protocol_instance_id": protocol_instance_id,
        "frozen_contract_sha256": EXPECTED_FROZEN_CONTRACT_SHA256,
        "phase_a_envelope_sha256": args.phase_a_envelope_sha256,
        "script_sha256": _sha256(Path(__file__).resolve()),
        "label_access_lifecycle_sha256": label_access_lifecycle_sha,
        "label_paths_constructed_before_marker": False,
        "label_files_opened_before_marker": False,
    }
    marker_path = output_root / TARGET_MARKER
    _write_plain_json(marker_path, marker_payload)
    marker_sha = _sha256(marker_path)

    _, label_root, label_records, label_snapshots, label_secret = _label_records_after_marker(
        args, protocol, marker_path
    )
    expected_runtime_pins = {
        key: value
        for key, value in sorted(protocol["runtime_pins"].items())
        if key.endswith("_sha256")
    }
    if phase_a_payload["runtime_pin_sha256"] != expected_runtime_pins:
        raise RuntimeError("Phase-A runtime pin closure drift")
    phase_records = phase_a_payload["records"]
    phase_by_id = {str(value["opaque_id"]): value for value in phase_records}
    label_by_id = {str(value["opaque_id"]): value for value in label_records}
    if set(phase_by_id) != set(label_by_id):
        raise RuntimeError("opaque Phase-A/label join mismatch")
    if set(phase_by_id) != set(input_bindings):
        raise RuntimeError("opaque input/Phase-A join mismatch")
    for opaque_id, phase_record in phase_by_id.items():
        if phase_record["input_fixture_sha256"] != input_artifact_hashes[opaque_id]:
            raise RuntimeError("Phase-A/input serialized artifact binding mismatch")
    # The frozen protocol requires every one of the 64 fixtures to be
    # independently reconstructed before the first metric is evaluated.  Keep
    # this as a complete first pass rather than interleaving validation and
    # scoring record by record.
    verified_labels: dict[str, tuple[dict[str, np.ndarray], dict[str, Any]]] = {}
    for opaque_id in sorted(phase_by_id):
        phase_record = phase_by_id[opaque_id]
        label_record = label_by_id[opaque_id]
        label_bytes, _, _ = _secure_relative_bytes(
            label_root,
            str(label_record["artifact"]["path"]),
            expected_parent="records",
        )
        if _bytes_sha256(label_bytes) != label_record["artifact"]["sha256"]:
            raise RuntimeError("label artifact changed before recomposition")
        labels = _label_arrays(label_bytes)
        recomposition = recompute_fixture_binding_after_marker(
            secret=label_secret,
            opaque_id=opaque_id,
            label_record=label_record,
            labels=labels,
            input_arrays=input_bindings[opaque_id],
            phase_record=phase_record,
            master_seed=int(
                protocol["frozen_contract"]["synthetic_corruption"]["master_seed"]
            ),
        )
        verified_labels[opaque_id] = (labels, recomposition)
    if len(verified_labels) != 64:
        raise RuntimeError("fixture recomposition coverage drift before metrics")

    record_reports: list[dict[str, Any]] = []
    for opaque_id in sorted(phase_by_id):
        phase_record = phase_by_id[opaque_id]
        label_record = label_by_id[opaque_id]
        graph_bytes, _, graph_path = _secure_relative_bytes(
            phase_a_root,
            str(phase_record["graph_artifact"]),
            expected_parent="artifacts",
        )
        if _bytes_sha256(graph_bytes) != phase_record["graph_artifact_sha256"]:
            raise RuntimeError("Phase-A graph changed before scoring")
        graph, arrays = _load_graph_artifact(graph_bytes)
        labels, recomposition = verified_labels[opaque_id]
        truth = labels["composed_slot_to_target"]
        clean_target = labels["clean_target_rgb"]
        recall = candidate_recall_metrics(graph, truth)
        components, component_diagnostics = truth_filtered_components(graph, truth)
        w4 = _w4_compatibility(arrays)
        oracle_layout, oracle_diagnostics = oracle_filter_beam_hungarian_qap(
            components,
            w4,
            qap_seed=int(phase_record["qap_seed"]),
        )
        translation_layout, translation_diagnostics = target_assisted_translation_ceiling(
            components, w4, truth
        )
        layouts = {
            "softcycle_l1_k8": arrays["softcycle_layout"],
            "qap_w4_b0.05_i25": arrays["qap_w4_layout"],
            "qap_w1_b0.05_i25": arrays["qap_w1_layout"],
            "oracle_filter_beam8_hungarian_qap25": oracle_layout,
            "absolute_true_component_translation_ceiling": translation_layout,
        }
        metrics: dict[str, dict[str, Any]] = {}
        for label, layout in layouts.items():
            metrics[label] = {
                "layout_sha256": _layout_sha256(layout),
                "layout": layout_metrics(layout, truth),
                "render": _render_metrics(
                    layout, arrays["denoised_tiles"], clean_target
                ),
            }
        baseline = metrics["qap_w4_b0.05_i25"]
        oracle = metrics["oracle_filter_beam8_hungarian_qap25"]
        paired_delta = {
            "combined_adjacency": float(
                oracle["layout"]["combined_adjacency"]
                - baseline["layout"]["combined_adjacency"]
            ),
            "rgb_ssim": float(
                oracle["render"]["rgb_ssim"] - baseline["render"]["rgb_ssim"]
            ),
        }
        layout_artifact = f"artifacts/{opaque_id}__oracle_layout.npy"
        layout_path = _safe_relative(
            output_root, layout_artifact, expected_parent="artifacts"
        )
        _atomic_bytes(layout_path, _npy_bytes(oracle_layout.astype(np.int32)))
        oracle_render = merge_tiles_numpy(
            arrays["denoised_tiles"][oracle_layout]
        )
        render_artifact = f"renders/{opaque_id}__oracle.png"
        render_path = _safe_relative(
            output_root, render_artifact, expected_parent="renders"
        )
        _atomic_bytes(render_path, _png_bytes(oracle_render))
        report = {
            "opaque_id": opaque_id,
            "source_index": int(label_record["_source_index"]),
            "name": str(label_record["source_name"]),
            "panel": str(label_record["panel"]),
            "panel_seed": int(label_record["panel_seed"]),
            "candidate_edge_count": len(graph.direction),
            "candidate_origin_counts": _origin_histogram(graph),
            "candidate_recall": recall,
            "fixture_recomposition": recomposition,
            "components": component_diagnostics,
            "layouts": metrics,
            "oracle_filter_diagnostics": oracle_diagnostics,
            "target_assisted_translation_diagnostics": translation_diagnostics,
            "paired_delta": paired_delta,
            "artifacts": {
                "oracle_layout": {
                    "path": layout_artifact,
                    "sha256": _sha256(layout_path),
                },
                "oracle_render": {
                    "path": render_artifact,
                    "sha256": _sha256(render_path),
                },
            },
        }
        if translation_diagnostics["eligible_for_gate"] is not False:
            raise RuntimeError("target-assisted diagnostic leaked into gate")
        record_reports.append(report)

    panel_summaries = _panel_summaries(record_reports)
    gate = evaluate_continuation_gate(panel_summaries)
    if len(record_reports) != 64 or any(
        not value["layouts"]["qap_w4_b0.05_i25"]["layout"]["valid_permutation"]
        or not value["layouts"]["oracle_filter_beam8_hungarian_qap25"]["layout"]["valid_permutation"]
        for value in record_reports
    ):
        raise RuntimeError("record/permutation integrity gate failed")

    # Full post-score TOCTOU audit.  No report is accepted until all snapshots
    # and runtime code/assets still match their pre-access bindings.
    config_after, _ = _secure_absolute_bytes(config_path)
    script_after, _ = _secure_absolute_bytes(Path(__file__).resolve())
    if _bytes_sha256(config_after) != config_sha or _bytes_sha256(script_after) != phase_a_payload["script_sha256"]:
        raise RuntimeError("config/evaluator TOCTOU mismatch")
    _validated_runtime_pin_closure(protocol, config_path)
    _validated_runtime_assets(protocol, config_path, args)
    for path_value, expected in {
        **phase_a_snapshots,
        **input_snapshots,
        **label_snapshots,
    }.items():
        try:
            snapshot_bytes, _ = _secure_absolute_bytes(Path(path_value))
        except Exception:
            raise RuntimeError("artifact TOCTOU mismatch") from None
        if _bytes_sha256(snapshot_bytes) != expected:
            raise RuntimeError("artifact TOCTOU mismatch")
    marker_after, _ = _secure_absolute_bytes(marker_path)
    if _bytes_sha256(marker_after) != marker_sha:
        raise RuntimeError("target-access marker TOCTOU mismatch")
    payload = {
        "schema_version": 1,
        "kind": "candidate_graph_oracle_ceiling_report",
        "status": "continue" if gate["continue_to_cycle_factor_synchronizer"] else "stop_or_pivot",
        "config_sha256": config_sha,
        "protocol_instance_id": protocol_instance_id,
        "frozen_contract_sha256": EXPECTED_FROZEN_CONTRACT_SHA256,
        "phase_a_envelope_sha256": args.phase_a_envelope_sha256,
        "target_access_marker_sha256": marker_sha,
        "lifecycle_sha256": dict(lifecycle_hashes),
        "fixture_input_manifest_sha256": args.fixture_manifest_sha256,
        "fixture_label_manifest_sha256": protocol["runtime_pins"][
            "fixture_label_manifest_sha256"
        ],
        "runtime_asset_sha256": expected_runtime_assets,
        "records": record_reports,
        "panel_summaries": panel_summaries,
        "continuation_gate": gate,
        "integrity": {
            "fixture_record_count": 64,
            "records_per_panel": 32,
            "candidate_graph_count": 64,
            "valid_baseline_permutation_count": 64,
            "valid_oracle_packer_permutation_count": 64,
            "artifact_hash_or_toctou_failures": 0,
            "opaque_id_join_errors": 0,
            "post_score_toctou_verified": True,
            "all_64_fixtures_recomposed_before_first_metric": True,
        },
        "target_assisted_translation_contributes_to_gate": False,
        "qap_weight_reopened": False,
        "safe_for_submission": False,
    }
    report_path = output_root / REPORT_NAME
    # The target-access marker is intentionally plain canonical JSON, while
    # the scored Phase-B result is an authenticated canonical envelope.  Keep
    # this distinction explicit because the independent result verifier must
    # reject an unwrapped report even when its payload happens to be valid.
    _write_envelope(report_path, payload)
    _assert_exact_directory_entries(
        output_root, {TARGET_MARKER, REPORT_NAME, "artifacts", "renders"}
    )
    _assert_exact_directory_entries(
        output_root / "artifacts",
        {f"{value['opaque_id']}__oracle_layout.npy" for value in record_reports},
    )
    _assert_exact_directory_entries(
        output_root / "renders",
        {f"{value['opaque_id']}__oracle.png" for value in record_reports},
    )
    return {
        "report": str(report_path),
        "report_sha256": _sha256(report_path),
        "status": payload["status"],
        "continue_to_cycle_factor_synchronizer": gate[
            "continue_to_cycle_factor_synchronizer"
        ],
        "safe_for_submission": False,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        if args.action == "phase-a":
            result = run_phase_a(args)
        elif args.action == "finalize-phase-a":
            result = run_finalize_phase_a(args)
        else:
            result = run_phase_b(args)
    except Exception as error:
        # If an output directory was authorized, preserve a fail-closed error
        # record.  Its absence can never be interpreted as a passing result.
        destination_value = (
            args.phase_a_dir
            if args.action == "phase-a"
            else args.finalized_phase_a_dir
            if args.action == "finalize-phase-a"
            else args.output
        )
        if destination_value:
            destination = Path(destination_value).expanduser().resolve()
            destination.mkdir(parents=True, exist_ok=True)
            error_path = destination / "INVALID_NO_RESULT.json"
            payload = {
                "schema_version": 1,
                "kind": "candidate_graph_oracle_invalid_no_result",
                "error_type": type(error).__name__,
                "error": str(error),
                "gate_passed": False,
                "accepted_aggregate_metrics": None,
                "safe_for_submission": False,
            }
            _write_envelope(error_path, payload)
        raise
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

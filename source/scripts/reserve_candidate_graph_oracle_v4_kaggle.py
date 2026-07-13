#!/usr/bin/env python3
"""Reserve the private Kaggle version-1 identities for oracle v4.

The default ``--validate-only`` path is deliberately standard-library-only: it
validates the four pixel-free exact trees and never imports, authenticates, or
calls Kaggle.  ``--execute`` is an explicit remote mutation mode.  Every write
is preceded by an O_EXCL/fsync intent and dispatch guard, every returned SDK
object is journaled before semantic parsing, and an ambiguous kernel commit is
never retried against the same slug.

This file only reserves empty version-1 identities.  It must never upload the
fixture, a version-2 dataset, the Phase-A runner, or a GPU-enabled kernel.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
import time
from typing import Any, Callable, Mapping, Protocol, Sequence


SCHEMA_VERSION = 1
PROTOCOL_INSTANCE_ID = "6c0fe4e8524ce39d830d9a5bee118d8b"
MARKER_NAME = "RESERVED_VERSION_1.txt"
RESERVATION_RUNNER_SHA256 = (
    "adf4d61a528f91ce5a4c282b0f3999f8bcdbe8c18d5429a98210dc6b991ab460"
)
RECOVERABLE_PREDECESSOR_ORCHESTRATOR_SHA256 = (
    "96da193cbe3e208c0044ab3f937b73f885c68bbf7572a38776cc028475279ff4",
    "93b0edc9a870965a62443489214dee3b9e582ff75c43cecc2bc05f6558e9dcdd",
    "e5240001d50195782cb8609e85cea1a68ae0d053ba274f62690df11a5d3c1e3d",
)
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESERVATION_ROOT = (
    REPO_ROOT
    / "runs/assembly_v1/kaggle/candidate_graph_oracle_v4_reservations"
)
DEFAULT_RECEIPT = DEFAULT_RESERVATION_ROOT / "RESERVATION_RECEIPT.json"
ENV_PREFIX = "/Users/rusyalain/Documents/test/.conda"
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
KAGGLE_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*/[A-Za-z0-9][A-Za-z0-9_-]*$")
MAX_MINE_DATASET_LIST_PAGES = 1000
MAX_MINE_KERNEL_LIST_PAGES = 1000


@dataclass(frozen=True)
class DatasetSpec:
    role: str
    slug: str
    title: str


DATASET_SPECS = (
    DatasetSpec(
        "code",
        "pasha883/vsos-candidate-graph-oracle-v4-code",
        "VSOS Candidate Graph Oracle V4 Code",
    ),
    DatasetSpec(
        "input",
        "pasha883/vsos-candidate-graph-oracle-v4-inputs",
        "VSOS Candidate Graph Oracle V4 Inputs",
    ),
    DatasetSpec(
        "runtime",
        "pasha883/vsos-candidate-graph-oracle-v4-runtime",
        "VSOS Candidate Graph Oracle V4 Runtime",
    ),
)
KERNEL_SLUG = "pasha883/vsos-candidate-graph-oracle-v4-phase-a-t4x2"
KERNEL_TITLE = "VSOS Candidate Graph Oracle V4 Phase A T4x2"

DATASET_RAW_FIELDS = ("ref", "url", "status", "error", "invalid_tags")
KERNEL_RAW_FIELDS = (
    "ref",
    "url",
    "version_number",
    "error",
    "invalid_tags",
    "invalid_dataset_sources",
    "invalid_competition_sources",
    "invalid_kernel_sources",
    "invalid_model_sources",
    "kernel_id",
)
PENDING_DATASET_STATUSES = {"creating", "pending", "queued", "running"}
PENDING_KERNEL_STATUSES = {"pending", "queued", "running"}


class ReservationAPI(Protocol):
    """Small injectable boundary used by production and fake tests."""

    def get_dataset_snapshot(self, slug: str, marker_name: str) -> dict[str, Any] | None:
        ...

    def create_dataset(self, directory: Path) -> Any:
        ...

    def get_kernel_snapshot(self, slug: str) -> dict[str, Any] | None:
        ...

    def create_kernel(self, directory: Path) -> Any:
        ...


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _canonical_object_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _canonical_file_bytes(value: Mapping[str, Any]) -> bytes:
    return _canonical_object_bytes(value) + b"\n"


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_object_bytes(value)).hexdigest()


def _require_directory(path: Path, *, label: str) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise RuntimeError(f"missing {label}: {path}") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise RuntimeError(f"{label} must be a real directory: {path}")


def _sha256_file(path: Path) -> str:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    digest = hashlib.sha256()
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RuntimeError(f"artifact must be a regular nlink==1 file: {path}")
        for chunk in iter(lambda: os.read(descriptor, 1024 * 1024), b""):
            digest.update(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _read_regular_bytes(path: Path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    chunks: list[bytes] = []
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RuntimeError(f"artifact must be a regular nlink==1 file: {path}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(_read_regular_bytes(path).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid JSON object: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> str:
    _require_directory(path.parent, label="evidence parent")
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    encoded = _canonical_file_bytes(payload)
    try:
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise RuntimeError(f"short write while committing {path}")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)
    return _sha256_file(path)


def _load_canonical(
    path: Path, *, expected_kind: str, envelope: bool = False
) -> tuple[dict[str, Any], str]:
    raw = _read_regular_bytes(path)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid canonical evidence: {path}") from error
    if not isinstance(value, dict) or raw != _canonical_file_bytes(value):
        raise RuntimeError(f"non-canonical evidence: {path}")
    if envelope:
        if set(value) != {"payload", "payload_sha256"}:
            raise RuntimeError(f"receipt envelope drift: {path}")
        payload = value.get("payload")
        if not isinstance(payload, dict) or value.get("payload_sha256") != _canonical_sha256(payload):
            raise RuntimeError(f"receipt self-hash mismatch: {path}")
        if payload.get("kind") != expected_kind:
            raise RuntimeError(f"receipt kind drift: {path}")
    elif value.get("schema_version") != SCHEMA_VERSION or value.get("kind") != expected_kind:
        raise RuntimeError(f"evidence schema/kind drift: {path}")
    return value, hashlib.sha256(raw).hexdigest()


def _write_receipt(path: Path, payload: Mapping[str, Any]) -> str:
    envelope = {"payload": dict(payload), "payload_sha256": _canonical_sha256(payload)}
    return _write_exclusive(path, envelope)


def _marker_bytes(role: str) -> bytes:
    return (
        "Candidate-graph oracle v4 private dataset version-1 reservation only.\n"
        f"protocol_instance_id={PROTOCOL_INSTANCE_ID}\n"
        f"role={role}\n"
        "contains_fixture_pixels=false\n"
        "safe_for_submission=false\n"
    ).encode("ascii")


def _expected_dataset_metadata(spec: DatasetSpec) -> dict[str, Any]:
    return {
        "id": spec.slug,
        "title": spec.title,
        "isPrivate": True,
        "licenses": [{"name": "other"}],
        "description": (
            "Pixel-free private version-1 reservation for candidate-graph "
            f"oracle v4 role {spec.role}."
        ),
    }


def _expected_kernel_metadata() -> dict[str, Any]:
    return {
        "id": KERNEL_SLUG,
        "title": KERNEL_TITLE,
        "code_file": "reservation_runner.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def _exact_tree(path: Path, expected_names: set[str]) -> dict[str, Any]:
    _require_directory(path, label="reservation tree")
    entries = sorted(path.iterdir(), key=lambda value: value.name)
    names = {entry.name for entry in entries}
    if names != expected_names:
        raise RuntimeError(
            f"reservation exact-tree drift at {path}: expected "
            f"{sorted(expected_names)}, got {sorted(names)}"
        )
    files: list[dict[str, Any]] = []
    for entry in entries:
        raw = _read_regular_bytes(entry)
        files.append(
            {
                "path": entry.name,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    manifest = {"files": files}
    return {"manifest": manifest, "exact_tree_sha256": _canonical_sha256(manifest)}


def validate_local_templates(reservation_root: Path) -> dict[str, Any]:
    """Validate all local reservation templates without importing Kaggle."""

    root = reservation_root.absolute()
    _require_directory(root, label="reservation root")
    _require_directory(root / "journal", label="reservation journal")
    datasets: dict[str, Any] = {}
    scanned: list[Path] = []
    for spec in DATASET_SPECS:
        directory = root / spec.role
        tree = _exact_tree(directory, {"dataset-metadata.json", MARKER_NAME})
        metadata_path = directory / "dataset-metadata.json"
        marker_path = directory / MARKER_NAME
        if _load_json_object(metadata_path) != _expected_dataset_metadata(spec):
            raise RuntimeError(f"dataset metadata drift: {spec.role}")
        marker = _read_regular_bytes(marker_path)
        if marker != _marker_bytes(spec.role):
            raise RuntimeError(f"dataset reservation marker drift: {spec.role}")
        scanned.extend((metadata_path, marker_path))
        datasets[spec.role] = {
            "slug": spec.slug,
            "title": spec.title,
            "reserved_version": 1,
            "expected_private": True,
            "metadata_sha256": _sha256_file(metadata_path),
            "marker_sha256": hashlib.sha256(marker).hexdigest(),
            "marker_bytes": len(marker),
            **tree,
        }

    kernel_dir = root / "kernel"
    kernel_tree = _exact_tree(
        kernel_dir, {"kernel-metadata.json", "reservation_runner.py"}
    )
    metadata_path = kernel_dir / "kernel-metadata.json"
    runner_path = kernel_dir / "reservation_runner.py"
    if _load_json_object(metadata_path) != _expected_kernel_metadata():
        raise RuntimeError("kernel reservation metadata drift")
    runner_sha256 = _sha256_file(runner_path)
    if runner_sha256 != RESERVATION_RUNNER_SHA256:
        raise RuntimeError("kernel reservation runner SHA-256 drift")
    scanned.extend((metadata_path, runner_path))

    forbidden = (b"INSERT_", b"CHANGEME", b"<owner>", b"<slug>", b"TODO")
    for path in scanned:
        raw = _read_regular_bytes(path)
        if any(token in raw for token in forbidden):
            raise RuntimeError(f"unresolved reservation placeholder: {path}")

    orchestrator_sha256 = _sha256_file(Path(__file__).resolve())
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "candidate_graph_oracle_v4_local_reservation_validation",
        "protocol_instance_id": PROTOCOL_INSTANCE_ID,
        "reservation_root": str(root),
        "reservation_orchestrator_sha256": orchestrator_sha256,
        "datasets": datasets,
        "kernel": {
            "slug": KERNEL_SLUG,
            "title": KERNEL_TITLE,
            "reserved_version": 1,
            "expected_private": True,
            "expected_enable_gpu": False,
            "expected_enable_tpu": False,
            "expected_enable_internet": False,
            "metadata_sha256": _sha256_file(metadata_path),
            "reservation_runner_sha256": runner_sha256,
            **kernel_tree,
        },
        "contains_fixture_pixels": False,
        "gpu_requested": False,
        "safe_for_submission": False,
    }


def _raw_json_value(value: Any, *, seen: set[int] | None = None) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {"__python_type__": "float", "value": repr(value)}
    if isinstance(value, bytes):
        return {
            "__python_type__": "bytes",
            "base64": base64.b64encode(value).decode("ascii"),
        }
    if isinstance(value, Path):
        return {"__python_type__": "pathlib.Path", "value": str(value)}
    if isinstance(value, Enum):
        return {
            "__python_type__": f"{type(value).__module__}.{type(value).__qualname__}",
            "name": value.name,
            "value": _raw_json_value(value.value, seen=seen),
        }
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return {"__python_cycle__": f"{type(value).__module__}.{type(value).__qualname__}"}
    seen.add(identity)
    try:
        if isinstance(value, Mapping):
            if not all(isinstance(key, str) for key in value):
                raise RuntimeError("raw Kaggle evidence contains a non-string mapping key")
            return {str(key): _raw_json_value(item, seen=seen) for key, item in value.items()}
        if isinstance(value, list):
            return [_raw_json_value(item, seen=seen) for item in value]
        if isinstance(value, tuple):
            return {"__python_type__": "tuple", "items": [_raw_json_value(item, seen=seen) for item in value]}
        if isinstance(value, Sequence):
            return [_raw_json_value(item, seen=seen) for item in value]
        try:
            state = vars(value)
        except TypeError:
            state = None
        if state is not None:
            return {
                "__python_type__": f"{type(value).__module__}.{type(value).__qualname__}",
                "state": _raw_json_value(state, seen=seen),
            }
        return {
            "__python_type__": f"{type(value).__module__}.{type(value).__qualname__}",
            "repr": repr(value),
            "representation_only": True,
        }
    finally:
        seen.remove(identity)


def _field(value: Any, name: str) -> Any:
    aliases = {"invalid_tags": "invalidTags"}
    if isinstance(value, Mapping):
        if name in value:
            return value[name]
        alias = aliases.get(name)
        if alias and alias in value:
            return value[alias]
        raise RuntimeError(f"Kaggle object is missing field {name}")
    try:
        return getattr(value, name)
    except AttributeError:
        alias = aliases.get(name)
        if alias is None:
            raise RuntimeError(f"Kaggle object is missing field {name}") from None
        try:
            return getattr(value, alias)
        except AttributeError:
            raise RuntimeError(f"Kaggle object is missing field {name}") from None


def _raw_response_payload(
    response: Any, *, kind: str, fields: Sequence[str], now: Callable[[], str]
) -> dict[str, Any]:
    public_fields: dict[str, Any] = {}
    for name in fields:
        try:
            public_fields[name] = _raw_json_value(_field(response, name))
        except Exception as error:
            public_fields[name] = {
                "__attribute_error__": {
                    "type": f"{type(error).__module__}.{type(error).__qualname__}",
                    "message": str(error),
                }
            }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "recorded_utc": now(),
        "response_type": {
            "module": type(response).__module__,
            "qualname": type(response).__qualname__,
        },
        "public_fields": public_fields,
        "object_state": _raw_json_value(vars(response) if hasattr(response, "__dict__") else None),
    }


def _validate_raw_response_envelope(
    raw: Mapping[str, Any], *, kind: str, fields: Sequence[str]
) -> None:
    if (
        set(raw)
        != {
            "schema_version",
            "kind",
            "recorded_utc",
            "response_type",
            "public_fields",
            "object_state",
        }
        or raw.get("schema_version") != SCHEMA_VERSION
        or raw.get("kind") != kind
        or not isinstance(raw.get("recorded_utc"), str)
        or not UTC_RE.fullmatch(str(raw["recorded_utc"]))
        or not isinstance(raw.get("response_type"), dict)
        or set(raw["response_type"]) != {"module", "qualname"}
        or not all(
            isinstance(raw["response_type"].get(name), str)
            for name in ("module", "qualname")
        )
        or not isinstance(raw.get("public_fields"), dict)
        or set(raw["public_fields"]) != set(fields)
    ):
        raise RuntimeError(f"raw Kaggle response envelope drift: {kind}")


def _strict_string_list(value: Any, *, field: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise RuntimeError(f"Kaggle response field {field} must be list[str]")
    return list(value)


def _dataset_response_payload(
    raw: Mapping[str, Any], *, spec: DatasetSpec, raw_file: str, raw_sha256: str, now: Callable[[], str]
) -> dict[str, Any]:
    _validate_raw_response_envelope(
        raw,
        kind="candidate_graph_oracle_v4_dataset_raw_create_response",
        fields=DATASET_RAW_FIELDS,
    )
    fields = raw.get("public_fields")
    if not isinstance(fields, dict) or set(fields) != set(DATASET_RAW_FIELDS):
        raise RuntimeError("raw dataset-create response schema drift")
    ref = fields["ref"]
    url = fields["url"]
    status_value = fields["status"]
    error = fields["error"]
    invalid_tags = _strict_string_list(fields["invalid_tags"], field="invalid_tags")
    if not all(isinstance(value, str) for value in (ref, url, status_value)):
        raise RuntimeError("dataset-create response has non-string identity fields")
    status_text = status_value.lower()
    accepted_refs = {spec.slug, f"/datasets/{spec.slug}"}
    if (
        ref not in accepted_refs
        or not url.startswith("https://www.kaggle.com/")
        or status_text not in {"ok", "pending"}
        or error not in (None, "")
        or invalid_tags
    ):
        raise RuntimeError("dataset-create response violates reservation expectation")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "candidate_graph_oracle_v4_dataset_reservation_response",
        "recorded_utc": now(),
        "protocol_instance_id": PROTOCOL_INSTANCE_ID,
        "role": spec.role,
        "ref": spec.slug,
        "url": url,
        "status": status_text,
        "error": error,
        "invalid_tags": invalid_tags,
        "raw_response_file": raw_file,
        "raw_response_sha256": raw_sha256,
    }


def _kernel_response_payload(
    raw: Mapping[str, Any], *, raw_file: str, raw_sha256: str, now: Callable[[], str]
) -> dict[str, Any]:
    _validate_raw_response_envelope(
        raw,
        kind="candidate_graph_oracle_v4_kernel_raw_create_response",
        fields=KERNEL_RAW_FIELDS,
    )
    fields = raw.get("public_fields")
    if not isinstance(fields, dict) or set(fields) != set(KERNEL_RAW_FIELDS):
        raise RuntimeError("raw kernel-create response schema drift")
    raw_ref = fields["ref"]
    if raw_ref == KERNEL_SLUG:
        ref = raw_ref
    elif raw_ref == f"/code/{KERNEL_SLUG}":
        ref = KERNEL_SLUG
    else:
        raise RuntimeError("kernel-create response ref drift")
    kernel_id = fields["kernel_id"]
    version = fields["version_number"]
    url = fields["url"]
    if isinstance(kernel_id, bool) or not isinstance(kernel_id, int) or kernel_id <= 0:
        raise RuntimeError("kernel-create response kernel_id must be positive int")
    if isinstance(version, bool) or version != 1:
        raise RuntimeError("kernel-create response must prove version 1")
    if not isinstance(url, str) or not url.startswith("https://www.kaggle.com/"):
        raise RuntimeError("kernel-create response URL drift")
    for field in (
        "invalid_tags",
        "invalid_dataset_sources",
        "invalid_competition_sources",
        "invalid_kernel_sources",
        "invalid_model_sources",
    ):
        if _strict_string_list(fields[field], field=field):
            raise RuntimeError(f"kernel-create response contains {field}")
    if fields["error"] not in (None, ""):
        raise RuntimeError("kernel-create response contains an error")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "candidate_graph_oracle_v4_kernel_reservation_response",
        "recorded_utc": now(),
        "protocol_instance_id": PROTOCOL_INSTANCE_ID,
        "ref": ref,
        "raw_ref": raw_ref,
        "url": url,
        "kernel_id": kernel_id,
        "version_number": version,
        "error": fields["error"],
        "invalid_tags": [],
        "invalid_dataset_sources": [],
        "invalid_competition_sources": [],
        "invalid_kernel_sources": [],
        "invalid_model_sources": [],
        "raw_response_file": raw_file,
        "raw_response_sha256": raw_sha256,
    }


def _status_text(value: Any) -> str:
    if isinstance(value, str):
        text = value
    elif hasattr(value, "name"):
        text = str(value.name)
    else:
        text = str(value)
    return text.rsplit(".", 1)[-1].lower()


def _decode_b64(value: Any, *, label: str) -> bytes:
    if not isinstance(value, str):
        raise RuntimeError(f"{label} must be base64 text")
    try:
        return base64.b64decode(value, validate=True)
    except ValueError as error:
        raise RuntimeError(f"{label} is invalid base64") from error


def _project_dataset_snapshot(
    snapshot: Mapping[str, Any], *, spec: DatasetSpec, local: Mapping[str, Any]
) -> dict[str, Any]:
    dataset = snapshot.get("dataset")
    status = snapshot.get("status")
    file_list = snapshot.get("file_list")
    marker = snapshot.get("marker")
    if not all(isinstance(value, dict) for value in (dataset, status, file_list, marker)):
        raise RuntimeError(f"dataset readback schema incomplete: {spec.role}")
    assert isinstance(dataset, dict) and isinstance(status, dict)
    assert isinstance(file_list, dict) and isinstance(marker, dict)
    files = file_list.get("files")
    if not isinstance(files, list) or len(files) != 1 or not isinstance(files[0], dict):
        raise RuntimeError(f"dataset must contain exactly one remote file: {spec.role}")
    remote_file = files[0]
    marker_bytes = _decode_b64(marker.get("base64"), label="downloaded marker")
    if (
        dataset.get("ref") != spec.slug
        or dataset.get("title") != spec.title
        or dataset.get("is_private") is not True
        or dataset.get("current_version_number") != 1
        or status.get("current_version_number") != 1
        or _status_text(status.get("status")) != "ready"
        or file_list.get("next_page_token") not in (None, "")
        or file_list.get("error_message") not in (None, "")
        or remote_file.get("name") != MARKER_NAME
        or remote_file.get("total_bytes") != len(marker_bytes)
        or marker.get("name") != MARKER_NAME
        or marker_bytes != _marker_bytes(spec.role)
        or hashlib.sha256(marker_bytes).hexdigest() != local["marker_sha256"]
    ):
        raise RuntimeError(f"dataset readback violates exact private v1 reservation: {spec.role}")
    dataset_id = dataset.get("id")
    if isinstance(dataset_id, bool) or not isinstance(dataset_id, int) or dataset_id <= 0:
        raise RuntimeError(f"dataset readback id must be positive: {spec.role}")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "candidate_graph_oracle_v4_dataset_reservation_readback",
        "protocol_instance_id": PROTOCOL_INSTANCE_ID,
        "role": spec.role,
        "slug": spec.slug,
        "dataset_id": dataset_id,
        "title": spec.title,
        "reserved_version": 1,
        "is_private": True,
        "status": "ready",
        "remote_files": [{"name": MARKER_NAME, "bytes": len(marker_bytes)}],
        "next_page_token": None,
        "downloaded_marker_sha256": hashlib.sha256(marker_bytes).hexdigest(),
        "contains_fixture_pixels": False,
        "safe_for_submission": False,
    }


def _project_kernel_snapshot(
    snapshot: Mapping[str, Any], *, response_kernel_id: int | None = None
) -> dict[str, Any]:
    metadata = snapshot.get("metadata")
    source = snapshot.get("source")
    status = snapshot.get("status")
    if not isinstance(metadata, dict) or not isinstance(source, dict) or not isinstance(status, dict):
        raise RuntimeError("kernel readback schema incomplete")
    source_bytes = _decode_b64(source.get("base64"), label="kernel source")
    kernel_id = metadata.get("id")
    expected_empty = (
        metadata.get("dataset_sources") == []
        and metadata.get("kernel_sources") == []
        and metadata.get("competition_sources") == []
        and metadata.get("model_sources") == []
    )
    if (
        isinstance(kernel_id, bool)
        or not isinstance(kernel_id, int)
        or kernel_id <= 0
        or (response_kernel_id is not None and kernel_id != response_kernel_id)
        or metadata.get("ref") != KERNEL_SLUG
        or metadata.get("slug") != KERNEL_SLUG.split("/", 1)[1]
        or metadata.get("title") != KERNEL_TITLE
        or metadata.get("language") != "python"
        or metadata.get("kernel_type") != "script"
        or metadata.get("is_private") is not True
        or metadata.get("enable_gpu") is not False
        or metadata.get("enable_tpu") is not False
        or metadata.get("enable_internet") is not False
        or not expected_empty
        or metadata.get("current_version_number") != 1
        or _status_text(status.get("status")) != "complete"
        or hashlib.sha256(source_bytes).hexdigest() != RESERVATION_RUNNER_SHA256
    ):
        raise RuntimeError("kernel readback violates exact private CPU-only v1 reservation")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "candidate_graph_oracle_v4_kernel_reservation_readback",
        "protocol_instance_id": PROTOCOL_INSTANCE_ID,
        "slug": KERNEL_SLUG,
        "kernel_id": kernel_id,
        "title": KERNEL_TITLE,
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
        "failure_message": status.get("failure_message") or "",
        "reservation_runner_sha256": RESERVATION_RUNNER_SHA256,
        "gpu_requested": False,
        "safe_for_submission": False,
    }


@dataclass(frozen=True)
class JournalPaths:
    intent: Path
    dispatch: Path
    raw_response: Path
    response: Path
    raw_readback: Path
    readback: Path


def _journal_paths(journal_dir: Path, name: str) -> JournalPaths:
    return JournalPaths(
        intent=journal_dir / f"{name}.00_intent.json",
        dispatch=journal_dir / f"{name}.01_dispatch.json",
        raw_response=journal_dir / f"{name}.02_raw_response.json",
        response=journal_dir / f"{name}.03_response.json",
        raw_readback=journal_dir / f"{name}.04_raw_readback.json",
        readback=journal_dir / f"{name}.05_readback.json",
    )


def _dataset_intent(
    *, spec: DatasetSpec, local: Mapping[str, Any], orchestrator_sha256: str, root: Path, created_utc: str
) -> dict[str, Any]:
    directory = root / spec.role
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "candidate_graph_oracle_v4_dataset_reservation_intent",
        "created_utc": created_utc,
        "protocol_instance_id": PROTOCOL_INSTANCE_ID,
        "role": spec.role,
        "slug": spec.slug,
        "expected_version": 1,
        "expected_private": True,
        "metadata_sha256": local["metadata_sha256"],
        "marker_name": MARKER_NAME,
        "marker_sha256": local["marker_sha256"],
        "marker_bytes": local["marker_bytes"],
        "exact_tree_sha256": local["exact_tree_sha256"],
        "reservation_orchestrator_sha256": orchestrator_sha256,
        "sdk_operation": {
            "method": "KaggleApi.dataset_create_new",
            "directory": str(directory),
            "public": False,
            "quiet": True,
            "convert_to_csv": False,
            "dir_mode": "skip",
        },
        "equivalent_cli_argv": [
            "conda", "run", "-p", ENV_PREFIX, "kaggle", "datasets", "create",
            "-p", str(directory), "--dir-mode", "skip",
        ],
        "contains_fixture_pixels": False,
        "safe_for_submission": False,
    }


def _kernel_intent(
    *, local: Mapping[str, Any], orchestrator_sha256: str, root: Path, created_utc: str
) -> dict[str, Any]:
    directory = root / "kernel"
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "candidate_graph_oracle_v4_kernel_reservation_intent",
        "created_utc": created_utc,
        "protocol_instance_id": PROTOCOL_INSTANCE_ID,
        "slug": KERNEL_SLUG,
        "expected_version": 1,
        "expected_private": True,
        "expected_enable_gpu": False,
        "expected_enable_tpu": False,
        "expected_enable_internet": False,
        "expected_dataset_sources": [],
        "expected_kernel_sources": [],
        "expected_competition_sources": [],
        "expected_model_sources": [],
        "metadata_sha256": local["metadata_sha256"],
        "reservation_runner_sha256": local["reservation_runner_sha256"],
        "exact_tree_sha256": local["exact_tree_sha256"],
        "reservation_orchestrator_sha256": orchestrator_sha256,
        "sdk_operation": {
            "method": "KaggleApi.kernels_push",
            "directory": str(directory),
            "timeout": None,
            "accelerator_override": None,
        },
        "equivalent_cli_argv": [
            "conda", "run", "-p", ENV_PREFIX, "kaggle", "kernels", "push", "-p", str(directory),
        ],
        "contains_fixture_pixels": False,
        "gpu_requested": False,
        "safe_for_submission": False,
    }


def _ensure_intent(
    path: Path,
    expected: Mapping[str, Any],
    *,
    predecessor_recovery_proof: Sequence[Path] = (),
) -> tuple[dict[str, Any], str, bool]:
    if path.exists() or path.is_symlink():
        value, sha256 = _load_canonical(path, expected_kind=str(expected["kind"]))
        created = value.get("created_utc")
        if not isinstance(created, str) or not UTC_RE.fullmatch(created):
            raise RuntimeError(f"intent timestamp drift: {path}")
        comparison = dict(expected)
        comparison["created_utc"] = created
        if value != comparison:
            predecessor_matches = any(
                value
                == {
                    **comparison,
                    "reservation_orchestrator_sha256": predecessor_sha,
                }
                for predecessor_sha in RECOVERABLE_PREDECESSOR_ORCHESTRATOR_SHA256
            )
            if not predecessor_matches or not predecessor_recovery_proof:
                raise RuntimeError(f"intent no longer matches local reservation closure: {path}")
            for proof in predecessor_recovery_proof:
                _read_regular_bytes(proof)
        return value, sha256, True
    value = dict(expected)
    sha256 = _write_exclusive(path, value)
    return value, sha256, False


def _dispatch_payload(
    *, resource_type: str, resource: str, intent_sha256: str, now: Callable[[], str]
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "candidate_graph_oracle_v4_remote_write_dispatch_guard",
        "created_utc": now(),
        "protocol_instance_id": PROTOCOL_INSTANCE_ID,
        "resource_type": resource_type,
        "resource": resource,
        "intent_sha256": intent_sha256,
        "single_remote_write_authorized": True,
        "same_slug_retry_after_unknown_commit_forbidden": True,
        "safe_for_submission": False,
    }


def _raw_readback_payload(
    *, kind: str, resource: str, slug: str, snapshot: Mapping[str, Any], now: Callable[[], str]
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "recorded_utc": now(),
        "protocol_instance_id": PROTOCOL_INSTANCE_ID,
        "resource": resource,
        "slug": slug,
        "snapshot": dict(snapshot),
    }


def _journal_ref(path: Path, *, root: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    return {"file": str(path.relative_to(root)), "sha256": _sha256_file(path)}


def _load_or_project_dataset_readback(
    *, paths: JournalPaths, spec: DatasetSpec, local: Mapping[str, Any], root: Path, now: Callable[[], str]
) -> dict[str, Any] | None:
    if paths.readback.exists() or paths.readback.is_symlink():
        value, _ = _load_canonical(
            paths.readback, expected_kind="candidate_graph_oracle_v4_dataset_reservation_readback"
        )
        raw, raw_sha = _load_canonical(
            paths.raw_readback,
            expected_kind="candidate_graph_oracle_v4_dataset_raw_readback",
        )
        snapshot = raw.get("snapshot")
        if not isinstance(snapshot, dict):
            raise RuntimeError(f"raw dataset readback snapshot drift: {spec.role}")
        recorded = value.get("recorded_utc")
        if not isinstance(recorded, str) or not UTC_RE.fullmatch(recorded):
            raise RuntimeError(f"dataset readback timestamp drift: {spec.role}")
        expected = _project_dataset_snapshot(snapshot, spec=spec, local=local)
        expected.update(
            {
                "recorded_utc": recorded,
                "raw_readback_file": str(paths.raw_readback.relative_to(root)),
                "raw_readback_sha256": raw_sha,
            }
        )
        if value != expected:
            raise RuntimeError(f"persisted dataset readback drift: {spec.role}")
        return value
    if not paths.raw_readback.exists() and not paths.raw_readback.is_symlink():
        return None
    raw, raw_sha = _load_canonical(
        paths.raw_readback, expected_kind="candidate_graph_oracle_v4_dataset_raw_readback"
    )
    snapshot = raw.get("snapshot")
    if not isinstance(snapshot, dict):
        raise RuntimeError(f"raw dataset readback snapshot drift: {spec.role}")
    value = _project_dataset_snapshot(snapshot, spec=spec, local=local)
    value.update(
        {
            "recorded_utc": now(),
            "raw_readback_file": str(paths.raw_readback.relative_to(root)),
            "raw_readback_sha256": raw_sha,
        }
    )
    _write_exclusive(paths.readback, value)
    return value


def _load_or_project_kernel_readback(
    *, paths: JournalPaths, response_kernel_id: int | None, root: Path, now: Callable[[], str]
) -> dict[str, Any] | None:
    if paths.readback.exists() or paths.readback.is_symlink():
        value, _ = _load_canonical(
            paths.readback, expected_kind="candidate_graph_oracle_v4_kernel_reservation_readback"
        )
        raw, raw_sha = _load_canonical(
            paths.raw_readback, expected_kind="candidate_graph_oracle_v4_kernel_raw_readback"
        )
        snapshot = raw.get("snapshot")
        if not isinstance(snapshot, dict):
            raise RuntimeError("raw kernel readback snapshot drift")
        projected = _project_kernel_snapshot(snapshot, response_kernel_id=response_kernel_id)
        recorded = value.get("recorded_utc")
        if not isinstance(recorded, str) or not UTC_RE.fullmatch(recorded):
            raise RuntimeError("kernel readback timestamp drift")
        projected.update(
            {
                "recorded_utc": recorded,
                "raw_readback_file": str(paths.raw_readback.relative_to(root)),
                "raw_readback_sha256": raw_sha,
            }
        )
        if value != projected:
            raise RuntimeError("persisted kernel readback drift")
        return value
    if not paths.raw_readback.exists() and not paths.raw_readback.is_symlink():
        return None
    raw, raw_sha = _load_canonical(
        paths.raw_readback, expected_kind="candidate_graph_oracle_v4_kernel_raw_readback"
    )
    snapshot = raw.get("snapshot")
    if not isinstance(snapshot, dict):
        raise RuntimeError("raw kernel readback snapshot drift")
    value = _project_kernel_snapshot(snapshot, response_kernel_id=response_kernel_id)
    value.update(
        {
            "recorded_utc": now(),
            "raw_readback_file": str(paths.raw_readback.relative_to(root)),
            "raw_readback_sha256": raw_sha,
        }
    )
    _write_exclusive(paths.readback, value)
    return value


def _finalize_dataset_snapshot(
    *, snapshot: Mapping[str, Any], paths: JournalPaths, spec: DatasetSpec, local: Mapping[str, Any], root: Path, now: Callable[[], str]
) -> dict[str, Any]:
    raw_payload = _raw_readback_payload(
        kind="candidate_graph_oracle_v4_dataset_raw_readback",
        resource=spec.role,
        slug=spec.slug,
        snapshot=snapshot,
        now=now,
    )
    raw_sha = _write_exclusive(paths.raw_readback, raw_payload)
    projected = _project_dataset_snapshot(snapshot, spec=spec, local=local)
    projected.update(
        {
            "recorded_utc": now(),
            "raw_readback_file": str(paths.raw_readback.relative_to(root)),
            "raw_readback_sha256": raw_sha,
        }
    )
    _write_exclusive(paths.readback, projected)
    return projected


def _finalize_kernel_snapshot(
    *, snapshot: Mapping[str, Any], paths: JournalPaths, response_kernel_id: int | None, root: Path, now: Callable[[], str]
) -> dict[str, Any]:
    raw_payload = _raw_readback_payload(
        kind="candidate_graph_oracle_v4_kernel_raw_readback",
        resource="kernel",
        slug=KERNEL_SLUG,
        snapshot=snapshot,
        now=now,
    )
    raw_sha = _write_exclusive(paths.raw_readback, raw_payload)
    projected = _project_kernel_snapshot(snapshot, response_kernel_id=response_kernel_id)
    projected.update(
        {
            "recorded_utc": now(),
            "raw_readback_file": str(paths.raw_readback.relative_to(root)),
            "raw_readback_sha256": raw_sha,
        }
    )
    _write_exclusive(paths.readback, projected)
    return projected


def _wait_dataset_ready(
    api: ReservationAPI, *, spec: DatasetSpec, attempts: int, sleep: Callable[[float], None]
) -> dict[str, Any] | None:
    last: dict[str, Any] | None = None
    for attempt in range(attempts):
        snapshot = api.get_dataset_snapshot(spec.slug, MARKER_NAME)
        if snapshot is None:
            last = None
        else:
            status = snapshot.get("status")
            status_value = status.get("status") if isinstance(status, dict) else None
            text = _status_text(status_value)
            if text == "ready":
                return snapshot
            if text not in PENDING_DATASET_STATUSES:
                raise RuntimeError(f"dataset reservation entered terminal status {text}: {spec.slug}")
            last = snapshot
        if attempt + 1 < attempts:
            sleep(2.0)
    return last if last is not None and _status_text(last.get("status", {}).get("status")) == "ready" else None


def _wait_kernel_complete(
    api: ReservationAPI, *, attempts: int, sleep: Callable[[float], None]
) -> dict[str, Any] | None:
    last: dict[str, Any] | None = None
    for attempt in range(attempts):
        snapshot = api.get_kernel_snapshot(KERNEL_SLUG)
        if snapshot is None:
            last = None
        else:
            status = snapshot.get("status")
            status_value = status.get("status") if isinstance(status, dict) else None
            text = _status_text(status_value)
            if text == "complete":
                return snapshot
            if text not in PENDING_KERNEL_STATUSES:
                raise RuntimeError(f"kernel reservation entered terminal status {text}: {KERNEL_SLUG}")
            last = snapshot
        if attempt + 1 < attempts:
            sleep(2.0)
    return last if last is not None and _status_text(last.get("status", {}).get("status")) == "complete" else None


def _adopt_existing_dataset_if_present(
    api: ReservationAPI,
    *,
    spec: DatasetSpec,
    attempts: int,
    sleep: Callable[[float], None],
) -> dict[str, Any] | None:
    """Return an exact ready snapshot, or prove the slug was absent before create.

    Once a same-slug dataset has been observed, absence and pending exhaustion
    are both fail-closed: a create call could otherwise publish version 2.
    """

    snapshot = api.get_dataset_snapshot(spec.slug, MARKER_NAME)
    if snapshot is None:
        # Repeat the exhaustive absence proof immediately before dispatch.  The
        # Kaggle API has no atomic create-if-absent primitive, so a single stale
        # listing is not sufficient protection against accidental version 2.
        snapshot = api.get_dataset_snapshot(spec.slug, MARKER_NAME)
        if snapshot is None:
            return None
    for attempt in range(attempts):
        status = snapshot.get("status")
        status_value = status.get("status") if isinstance(status, dict) else None
        text = _status_text(status_value)
        if text == "ready":
            return snapshot
        if text not in PENDING_DATASET_STATUSES:
            raise RuntimeError(
                f"existing dataset reservation entered terminal status {text}: {spec.slug}"
            )
        if attempt + 1 < attempts:
            sleep(2.0)
            snapshot = api.get_dataset_snapshot(spec.slug, MARKER_NAME)
            if snapshot is None:
                raise RuntimeError(
                    "same-slug dataset disappeared while pending; refusing create "
                    f"because it could create version 2: {spec.slug}"
                )
    raise RuntimeError(
        "same-slug dataset already exists but did not become exact ready private v1; "
        f"refusing create because it could create version 2: {spec.slug}"
    )


def _adopt_existing_kernel_if_present(
    api: ReservationAPI,
    *,
    attempts: int,
    sleep: Callable[[float], None],
) -> dict[str, Any] | None:
    """Return an exact complete snapshot, or prove the slug was absent before push."""

    snapshot = api.get_kernel_snapshot(KERNEL_SLUG)
    if snapshot is None:
        snapshot = api.get_kernel_snapshot(KERNEL_SLUG)
        if snapshot is None:
            return None
    for attempt in range(attempts):
        status = snapshot.get("status")
        status_value = status.get("status") if isinstance(status, dict) else None
        text = _status_text(status_value)
        if text == "complete":
            return snapshot
        if text not in PENDING_KERNEL_STATUSES:
            raise RuntimeError(
                f"existing kernel reservation entered terminal status {text}: {KERNEL_SLUG}"
            )
        if attempt + 1 < attempts:
            sleep(2.0)
            snapshot = api.get_kernel_snapshot(KERNEL_SLUG)
            if snapshot is None:
                raise RuntimeError(
                    "same-slug kernel disappeared while pending; refusing push "
                    "because it could create version 2"
                )
    raise RuntimeError(
        "same-slug kernel already exists but did not become exact complete private v1; "
        "refusing push because it could create version 2"
    )


def _reserve_dataset(
    api: ReservationAPI,
    *,
    spec: DatasetSpec,
    local: Mapping[str, Any],
    root: Path,
    journal_dir: Path,
    orchestrator_sha256: str,
    attempts: int,
    sleep: Callable[[float], None],
    now: Callable[[], str],
) -> dict[str, Any]:
    paths = _journal_paths(journal_dir, f"dataset_{spec.role}")
    intent_expected = _dataset_intent(
        spec=spec,
        local=local,
        orchestrator_sha256=orchestrator_sha256,
        root=root,
        created_utc=now(),
    )
    _, intent_sha, _ = _ensure_intent(
        paths.intent,
        intent_expected,
        predecessor_recovery_proof=(paths.dispatch, paths.raw_response),
    )

    replay = _load_or_project_dataset_readback(
        paths=paths, spec=spec, local=local, root=root, now=now
    )
    if replay is not None:
        mode = "journal_replay"
        return {
            "mode": mode,
            "readback": replay,
            "paths": paths,
            "remote_write_performed": False,
        }

    if paths.response.exists() and not paths.raw_response.exists():
        raise RuntimeError(f"dataset response exists without raw response: {spec.role}")
    response: dict[str, Any] | None = None
    if paths.raw_response.exists() or paths.raw_response.is_symlink():
        raw, raw_sha = _load_canonical(
            paths.raw_response,
            expected_kind="candidate_graph_oracle_v4_dataset_raw_create_response",
        )
        if paths.response.exists() or paths.response.is_symlink():
            response, _ = _load_canonical(
                paths.response,
                expected_kind="candidate_graph_oracle_v4_dataset_reservation_response",
            )
            expected = _dataset_response_payload(
                raw,
                spec=spec,
                raw_file=str(paths.raw_response.relative_to(root)),
                raw_sha256=raw_sha,
                now=lambda: str(response.get("recorded_utc")),
            )
            if response != expected:
                raise RuntimeError(f"dataset normalized response drift: {spec.role}")
        else:
            response = _dataset_response_payload(
                raw,
                spec=spec,
                raw_file=str(paths.raw_response.relative_to(root)),
                raw_sha256=raw_sha,
                now=now,
            )
            _write_exclusive(paths.response, response)
        snapshot = _wait_dataset_ready(api, spec=spec, attempts=attempts, sleep=sleep)
        if snapshot is None:
            raise RuntimeError(f"dataset create response exists but exact ready v1 is not readable: {spec.slug}")
        readback = _finalize_dataset_snapshot(
            snapshot=snapshot, paths=paths, spec=spec, local=local, root=root, now=now
        )
        return {
            "mode": "recovered_raw_response",
            "readback": readback,
            "paths": paths,
            "remote_write_performed": False,
        }

    if paths.dispatch.exists() or paths.dispatch.is_symlink():
        snapshot = _wait_dataset_ready(api, spec=spec, attempts=attempts, sleep=sleep)
        if snapshot is None:
            raise RuntimeError(
                f"dataset dispatch may have committed but no raw response exists; refusing same-slug retry: {spec.slug}"
            )
        readback = _finalize_dataset_snapshot(
            snapshot=snapshot, paths=paths, spec=spec, local=local, root=root, now=now
        )
        return {
            "mode": "recovered_unknown_commit",
            "readback": readback,
            "paths": paths,
            "remote_write_performed": False,
        }

    existing = _adopt_existing_dataset_if_present(
        api, spec=spec, attempts=attempts, sleep=sleep
    )
    if existing is not None:
        readback = _finalize_dataset_snapshot(
            snapshot=existing, paths=paths, spec=spec, local=local, root=root, now=now
        )
        return {
            "mode": "adopted_existing",
            "readback": readback,
            "paths": paths,
            "remote_write_performed": False,
        }

    _write_exclusive(
        paths.dispatch,
        _dispatch_payload(
            resource_type="dataset", resource=spec.role, intent_sha256=intent_sha, now=now
        ),
    )
    try:
        sdk_response = api.create_dataset(root / spec.role)
    except Exception as error:
        snapshot = _wait_dataset_ready(api, spec=spec, attempts=attempts, sleep=sleep)
        if snapshot is None:
            raise RuntimeError(
                f"dataset create has unknown commit state; same-slug retry is forbidden: {spec.slug}"
            ) from error
        readback = _finalize_dataset_snapshot(
            snapshot=snapshot, paths=paths, spec=spec, local=local, root=root, now=now
        )
        return {
            "mode": "recovered_unknown_commit",
            "readback": readback,
            "paths": paths,
            "remote_write_performed": True,
        }

    raw = _raw_response_payload(
        sdk_response,
        kind="candidate_graph_oracle_v4_dataset_raw_create_response",
        fields=DATASET_RAW_FIELDS,
        now=now,
    )
    raw_sha = _write_exclusive(paths.raw_response, raw)
    response = _dataset_response_payload(
        raw,
        spec=spec,
        raw_file=str(paths.raw_response.relative_to(root)),
        raw_sha256=raw_sha,
        now=now,
    )
    _write_exclusive(paths.response, response)
    snapshot = _wait_dataset_ready(api, spec=spec, attempts=attempts, sleep=sleep)
    if snapshot is None:
        raise RuntimeError(f"new dataset did not become exact ready private v1: {spec.slug}")
    readback = _finalize_dataset_snapshot(
        snapshot=snapshot, paths=paths, spec=spec, local=local, root=root, now=now
    )
    return {
        "mode": "created",
        "readback": readback,
        "paths": paths,
        "remote_write_performed": True,
    }


def _reserve_kernel(
    api: ReservationAPI,
    *,
    local: Mapping[str, Any],
    root: Path,
    journal_dir: Path,
    orchestrator_sha256: str,
    attempts: int,
    sleep: Callable[[float], None],
    now: Callable[[], str],
) -> dict[str, Any]:
    paths = _journal_paths(journal_dir, "kernel")
    intent_expected = _kernel_intent(
        local=local,
        orchestrator_sha256=orchestrator_sha256,
        root=root,
        created_utc=now(),
    )
    _, intent_sha, _ = _ensure_intent(
        paths.intent,
        intent_expected,
        predecessor_recovery_proof=(paths.dispatch, paths.raw_response),
    )

    response: dict[str, Any] | None = None
    if paths.raw_response.exists() or paths.raw_response.is_symlink():
        raw, raw_sha = _load_canonical(
            paths.raw_response,
            expected_kind="candidate_graph_oracle_v4_kernel_raw_create_response",
        )
        if paths.response.exists() or paths.response.is_symlink():
            response, _ = _load_canonical(
                paths.response,
                expected_kind="candidate_graph_oracle_v4_kernel_reservation_response",
            )
            expected = _kernel_response_payload(
                raw,
                raw_file=str(paths.raw_response.relative_to(root)),
                raw_sha256=raw_sha,
                now=lambda: str(response.get("recorded_utc")),
            )
            if response != expected:
                raise RuntimeError("kernel normalized response drift")
        else:
            response = _kernel_response_payload(
                raw,
                raw_file=str(paths.raw_response.relative_to(root)),
                raw_sha256=raw_sha,
                now=now,
            )
            _write_exclusive(paths.response, response)
    elif paths.response.exists() or paths.response.is_symlink():
        raise RuntimeError("kernel response exists without raw response")

    response_kernel_id = response["kernel_id"] if response is not None else None
    replay = _load_or_project_kernel_readback(
        paths=paths,
        response_kernel_id=response_kernel_id,
        root=root,
        now=now,
    )
    if replay is not None:
        return {
            "mode": "journal_replay",
            "readback": replay,
            "paths": paths,
            "remote_write_performed": False,
        }

    if response is not None:
        snapshot = _wait_kernel_complete(api, attempts=attempts, sleep=sleep)
        if snapshot is None:
            raise RuntimeError("kernel create response exists but exact complete private v1 is not readable")
        readback = _finalize_kernel_snapshot(
            snapshot=snapshot,
            paths=paths,
            response_kernel_id=response_kernel_id,
            root=root,
            now=now,
        )
        return {
            "mode": "recovered_raw_response",
            "readback": readback,
            "paths": paths,
            "remote_write_performed": False,
        }

    if paths.dispatch.exists() or paths.dispatch.is_symlink():
        snapshot = _wait_kernel_complete(api, attempts=attempts, sleep=sleep)
        if snapshot is None:
            raise RuntimeError(
                "kernel dispatch may have committed but no raw response exists; "
                "refusing same-slug push because it could create version 2"
            )
        readback = _finalize_kernel_snapshot(
            snapshot=snapshot,
            paths=paths,
            response_kernel_id=None,
            root=root,
            now=now,
        )
        return {
            "mode": "recovered_unknown_commit",
            "readback": readback,
            "paths": paths,
            "remote_write_performed": False,
        }

    existing = _adopt_existing_kernel_if_present(
        api, attempts=attempts, sleep=sleep
    )
    if existing is not None:
        readback = _finalize_kernel_snapshot(
            snapshot=existing,
            paths=paths,
            response_kernel_id=None,
            root=root,
            now=now,
        )
        return {
            "mode": "adopted_existing",
            "readback": readback,
            "paths": paths,
            "remote_write_performed": False,
        }

    _write_exclusive(
        paths.dispatch,
        _dispatch_payload(
            resource_type="kernel", resource=KERNEL_SLUG, intent_sha256=intent_sha, now=now
        ),
    )
    try:
        sdk_response = api.create_kernel(root / "kernel")
    except Exception as error:
        snapshot = _wait_kernel_complete(api, attempts=attempts, sleep=sleep)
        if snapshot is None:
            raise RuntimeError(
                "kernel create has unknown commit state; refusing same-slug push "
                "because it could create version 2"
            ) from error
        readback = _finalize_kernel_snapshot(
            snapshot=snapshot,
            paths=paths,
            response_kernel_id=None,
            root=root,
            now=now,
        )
        return {
            "mode": "recovered_unknown_commit",
            "readback": readback,
            "paths": paths,
            "remote_write_performed": True,
        }

    raw = _raw_response_payload(
        sdk_response,
        kind="candidate_graph_oracle_v4_kernel_raw_create_response",
        fields=KERNEL_RAW_FIELDS,
        now=now,
    )
    raw_sha = _write_exclusive(paths.raw_response, raw)
    response = _kernel_response_payload(
        raw,
        raw_file=str(paths.raw_response.relative_to(root)),
        raw_sha256=raw_sha,
        now=now,
    )
    _write_exclusive(paths.response, response)
    snapshot = _wait_kernel_complete(api, attempts=attempts, sleep=sleep)
    if snapshot is None:
        raise RuntimeError("new kernel did not become exact complete CPU-only private v1")
    readback = _finalize_kernel_snapshot(
        snapshot=snapshot,
        paths=paths,
        response_kernel_id=response["kernel_id"],
        root=root,
        now=now,
    )
    return {
        "mode": "created",
        "readback": readback,
        "paths": paths,
        "remote_write_performed": True,
    }


def _resource_journal(paths: JournalPaths, *, root: Path) -> dict[str, Any]:
    return {
        "intent": _journal_ref(paths.intent, root=root),
        "dispatch": _journal_ref(paths.dispatch, root=root),
        "raw_create_response": _journal_ref(paths.raw_response, root=root),
        "create_response": _journal_ref(paths.response, root=root),
        "raw_readback": _journal_ref(paths.raw_readback, root=root),
        "readback": _journal_ref(paths.readback, root=root),
    }


def _verify_receipt_journal(
    value: Any, *, paths: JournalPaths, root: Path
) -> dict[str, bool]:
    expected_paths = {
        "intent": paths.intent,
        "dispatch": paths.dispatch,
        "raw_create_response": paths.raw_response,
        "create_response": paths.response,
        "raw_readback": paths.raw_readback,
        "readback": paths.readback,
    }
    if not isinstance(value, dict) or set(value) != set(expected_paths):
        raise RuntimeError("reservation receipt journal schema drift")
    present: dict[str, bool] = {}
    for field, expected_path in expected_paths.items():
        exists = expected_path.exists() or expected_path.is_symlink()
        present[field] = exists
        reference = value[field]
        if not exists:
            if reference is not None:
                raise RuntimeError(f"journal receipt claims missing file: {field}")
            continue
        if not isinstance(reference, dict) or set(reference) != {"file", "sha256"}:
            raise RuntimeError(f"journal receipt reference drift: {field}")
        expected_relative = str(expected_path.relative_to(root))
        if (
            reference.get("file") != expected_relative
            or reference.get("sha256") != _sha256_file(expected_path)
        ):
            raise RuntimeError(f"journal receipt hash/path mismatch: {field}")
    if not present["intent"] or not present["raw_readback"] or not present["readback"]:
        raise RuntimeError("reservation receipt is missing mandatory journal evidence")
    if present["raw_create_response"] != present["create_response"]:
        raise RuntimeError("raw and normalized create-response journals must be paired")
    if present["raw_create_response"] and not present["dispatch"]:
        raise RuntimeError("create response exists without a dispatch guard")
    return present


def _verify_raw_readback_outer(
    value: Mapping[str, Any], *, kind: str, resource: str, slug: str
) -> Mapping[str, Any]:
    if (
        set(value)
        != {
            "schema_version",
            "kind",
            "recorded_utc",
            "protocol_instance_id",
            "resource",
            "slug",
            "snapshot",
        }
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("kind") != kind
        or not isinstance(value.get("recorded_utc"), str)
        or not UTC_RE.fullmatch(str(value["recorded_utc"]))
        or value.get("protocol_instance_id") != PROTOCOL_INSTANCE_ID
        or value.get("resource") != resource
        or value.get("slug") != slug
        or not isinstance(value.get("snapshot"), dict)
    ):
        raise RuntimeError(f"raw readback envelope drift: {resource}")
    return value["snapshot"]


def _verify_mode_against_journal(mode: Any, present: Mapping[str, bool]) -> None:
    allowed = {
        "created",
        "adopted_existing",
        "recovered_unknown_commit",
        "recovered_raw_response",
        "journal_replay",
    }
    if mode not in allowed:
        raise RuntimeError("reservation receipt mode drift")
    if mode == "adopted_existing" and any(
        present[field]
        for field in ("dispatch", "raw_create_response", "create_response")
    ):
        raise RuntimeError("adopted reservation unexpectedly has remote-write evidence")
    if mode in {"created", "recovered_raw_response"} and not all(
        present[field]
        for field in ("dispatch", "raw_create_response", "create_response")
    ):
        raise RuntimeError("created reservation lacks complete response journal")
    if mode == "recovered_unknown_commit" and (
        not present["dispatch"]
        or present["raw_create_response"]
        or present["create_response"]
    ):
        raise RuntimeError("unknown-commit recovery journal is inconsistent")


def _verify_dataset_receipt_record(
    *,
    record: Any,
    spec: DatasetSpec,
    local: Mapping[str, Any],
    root: Path,
    orchestrator_sha256: str,
) -> set[Path]:
    expected_keys = {
        "slug",
        "dataset_id",
        "reserved_version",
        "is_private",
        "status",
        "marker_sha256",
        "exact_tree_sha256",
        "mode",
        "journal",
    }
    if not isinstance(record, dict) or set(record) != expected_keys:
        raise RuntimeError(f"dataset receipt schema drift: {spec.role}")
    dataset_id = record.get("dataset_id")
    if (
        record.get("slug") != spec.slug
        or isinstance(dataset_id, bool)
        or not isinstance(dataset_id, int)
        or dataset_id <= 0
        or record.get("reserved_version") != 1
        or record.get("is_private") is not True
        or record.get("status") != "ready"
        or record.get("marker_sha256") != local["marker_sha256"]
        or record.get("exact_tree_sha256") != local["exact_tree_sha256"]
    ):
        raise RuntimeError(f"dataset receipt identity drift: {spec.role}")

    paths = _journal_paths(root / "journal", f"dataset_{spec.role}")
    present = _verify_receipt_journal(record["journal"], paths=paths, root=root)
    _verify_mode_against_journal(record["mode"], present)

    intent, intent_sha = _load_canonical(
        paths.intent,
        expected_kind="candidate_graph_oracle_v4_dataset_reservation_intent",
    )
    created = intent.get("created_utc")
    if not isinstance(created, str) or not UTC_RE.fullmatch(created):
        raise RuntimeError(f"dataset intent timestamp drift: {spec.role}")
    expected_intent = _dataset_intent(
        spec=spec,
        local=local,
        orchestrator_sha256=orchestrator_sha256,
        root=root,
        created_utc=created,
    )
    if intent != expected_intent:
        predecessor_matches = any(
            intent
            == {
                **expected_intent,
                "reservation_orchestrator_sha256": predecessor_sha,
            }
            for predecessor_sha in RECOVERABLE_PREDECESSOR_ORCHESTRATOR_SHA256
        )
        if (
            not predecessor_matches
            or not present["dispatch"]
            or not present["raw_create_response"]
        ):
            raise RuntimeError(f"dataset intent closure drift: {spec.role}")

    if present["dispatch"]:
        dispatch, _ = _load_canonical(
            paths.dispatch,
            expected_kind="candidate_graph_oracle_v4_remote_write_dispatch_guard",
        )
        dispatch_created = dispatch.get("created_utc")
        if not isinstance(dispatch_created, str) or not UTC_RE.fullmatch(dispatch_created):
            raise RuntimeError(f"dataset dispatch timestamp drift: {spec.role}")
        expected_dispatch = _dispatch_payload(
            resource_type="dataset",
            resource=spec.role,
            intent_sha256=intent_sha,
            now=lambda: dispatch_created,
        )
        if dispatch != expected_dispatch:
            raise RuntimeError(f"dataset dispatch closure drift: {spec.role}")

    if present["raw_create_response"]:
        raw_response, raw_response_sha = _load_canonical(
            paths.raw_response,
            expected_kind="candidate_graph_oracle_v4_dataset_raw_create_response",
        )
        response, _ = _load_canonical(
            paths.response,
            expected_kind="candidate_graph_oracle_v4_dataset_reservation_response",
        )
        response_created = response.get("recorded_utc")
        if not isinstance(response_created, str) or not UTC_RE.fullmatch(response_created):
            raise RuntimeError(f"dataset response timestamp drift: {spec.role}")
        expected_response = _dataset_response_payload(
            raw_response,
            spec=spec,
            raw_file=str(paths.raw_response.relative_to(root)),
            raw_sha256=raw_response_sha,
            now=lambda: response_created,
        )
        if response != expected_response:
            raise RuntimeError(f"dataset response closure drift: {spec.role}")

    raw_readback, raw_readback_sha = _load_canonical(
        paths.raw_readback,
        expected_kind="candidate_graph_oracle_v4_dataset_raw_readback",
    )
    snapshot = _verify_raw_readback_outer(
        raw_readback,
        kind="candidate_graph_oracle_v4_dataset_raw_readback",
        resource=spec.role,
        slug=spec.slug,
    )
    readback, _ = _load_canonical(
        paths.readback,
        expected_kind="candidate_graph_oracle_v4_dataset_reservation_readback",
    )
    readback_created = readback.get("recorded_utc")
    if not isinstance(readback_created, str) or not UTC_RE.fullmatch(readback_created):
        raise RuntimeError(f"dataset readback timestamp drift: {spec.role}")
    expected_readback = _project_dataset_snapshot(snapshot, spec=spec, local=local)
    expected_readback.update(
        {
            "recorded_utc": readback_created,
            "raw_readback_file": str(paths.raw_readback.relative_to(root)),
            "raw_readback_sha256": raw_readback_sha,
        }
    )
    if readback != expected_readback or readback["dataset_id"] != dataset_id:
        raise RuntimeError(f"dataset receipt/readback crosslink drift: {spec.role}")
    return {path for path in paths.__dict__.values() if path.exists() or path.is_symlink()}


def _verify_kernel_receipt_record(
    *,
    record: Any,
    local: Mapping[str, Any],
    root: Path,
    orchestrator_sha256: str,
) -> set[Path]:
    expected_keys = {
        "slug",
        "kernel_id",
        "reserved_version",
        "is_private",
        "enable_gpu",
        "enable_tpu",
        "enable_internet",
        "dataset_sources",
        "kernel_sources",
        "competition_sources",
        "model_sources",
        "status",
        "reservation_runner_sha256",
        "exact_tree_sha256",
        "mode",
        "journal",
    }
    if not isinstance(record, dict) or set(record) != expected_keys:
        raise RuntimeError("kernel receipt schema drift")
    kernel_id = record.get("kernel_id")
    if (
        record.get("slug") != KERNEL_SLUG
        or isinstance(kernel_id, bool)
        or not isinstance(kernel_id, int)
        or kernel_id <= 0
        or record.get("reserved_version") != 1
        or record.get("is_private") is not True
        or record.get("enable_gpu") is not False
        or record.get("enable_tpu") is not False
        or record.get("enable_internet") is not False
        or record.get("dataset_sources") != []
        or record.get("kernel_sources") != []
        or record.get("competition_sources") != []
        or record.get("model_sources") != []
        or record.get("status") != "complete"
        or record.get("reservation_runner_sha256") != RESERVATION_RUNNER_SHA256
        or record.get("exact_tree_sha256") != local["exact_tree_sha256"]
    ):
        raise RuntimeError("kernel receipt identity drift")

    paths = _journal_paths(root / "journal", "kernel")
    present = _verify_receipt_journal(record["journal"], paths=paths, root=root)
    _verify_mode_against_journal(record["mode"], present)

    intent, intent_sha = _load_canonical(
        paths.intent,
        expected_kind="candidate_graph_oracle_v4_kernel_reservation_intent",
    )
    created = intent.get("created_utc")
    if not isinstance(created, str) or not UTC_RE.fullmatch(created):
        raise RuntimeError("kernel intent timestamp drift")
    expected_intent = _kernel_intent(
        local=local,
        orchestrator_sha256=orchestrator_sha256,
        root=root,
        created_utc=created,
    )
    if intent != expected_intent:
        predecessor_matches = any(
            intent
            == {
                **expected_intent,
                "reservation_orchestrator_sha256": predecessor_sha,
            }
            for predecessor_sha in RECOVERABLE_PREDECESSOR_ORCHESTRATOR_SHA256
        )
        if (
            not predecessor_matches
            or not present["dispatch"]
            or not present["raw_create_response"]
        ):
            raise RuntimeError("kernel intent closure drift")

    if present["dispatch"]:
        dispatch, _ = _load_canonical(
            paths.dispatch,
            expected_kind="candidate_graph_oracle_v4_remote_write_dispatch_guard",
        )
        dispatch_created = dispatch.get("created_utc")
        if not isinstance(dispatch_created, str) or not UTC_RE.fullmatch(dispatch_created):
            raise RuntimeError("kernel dispatch timestamp drift")
        expected_dispatch = _dispatch_payload(
            resource_type="kernel",
            resource=KERNEL_SLUG,
            intent_sha256=intent_sha,
            now=lambda: dispatch_created,
        )
        if dispatch != expected_dispatch:
            raise RuntimeError("kernel dispatch closure drift")

    response_kernel_id: int | None = None
    if present["raw_create_response"]:
        raw_response, raw_response_sha = _load_canonical(
            paths.raw_response,
            expected_kind="candidate_graph_oracle_v4_kernel_raw_create_response",
        )
        response, _ = _load_canonical(
            paths.response,
            expected_kind="candidate_graph_oracle_v4_kernel_reservation_response",
        )
        response_created = response.get("recorded_utc")
        if not isinstance(response_created, str) or not UTC_RE.fullmatch(response_created):
            raise RuntimeError("kernel response timestamp drift")
        expected_response = _kernel_response_payload(
            raw_response,
            raw_file=str(paths.raw_response.relative_to(root)),
            raw_sha256=raw_response_sha,
            now=lambda: response_created,
        )
        if response != expected_response:
            raise RuntimeError("kernel response closure drift")
        response_kernel_id = response["kernel_id"]

    raw_readback, raw_readback_sha = _load_canonical(
        paths.raw_readback,
        expected_kind="candidate_graph_oracle_v4_kernel_raw_readback",
    )
    snapshot = _verify_raw_readback_outer(
        raw_readback,
        kind="candidate_graph_oracle_v4_kernel_raw_readback",
        resource="kernel",
        slug=KERNEL_SLUG,
    )
    readback, _ = _load_canonical(
        paths.readback,
        expected_kind="candidate_graph_oracle_v4_kernel_reservation_readback",
    )
    readback_created = readback.get("recorded_utc")
    if not isinstance(readback_created, str) or not UTC_RE.fullmatch(readback_created):
        raise RuntimeError("kernel readback timestamp drift")
    expected_readback = _project_kernel_snapshot(
        snapshot, response_kernel_id=response_kernel_id
    )
    expected_readback.update(
        {
            "recorded_utc": readback_created,
            "raw_readback_file": str(paths.raw_readback.relative_to(root)),
            "raw_readback_sha256": raw_readback_sha,
        }
    )
    if readback != expected_readback or readback["kernel_id"] != kernel_id:
        raise RuntimeError("kernel receipt/readback crosslink drift")
    return {path for path in paths.__dict__.values() if path.exists() or path.is_symlink()}


def _validate_receipt_payload(
    payload: Mapping[str, Any], *, root: Path
) -> None:
    expected_payload_keys = {
        "schema_version",
        "kind",
        "created_utc",
        "protocol_instance_id",
        "reservation_orchestrator_sha256",
        "local_validation",
        "datasets",
        "kernel",
        "contains_fixture_pixels",
        "gpu_requested",
        "dataset_v2_uploaded",
        "phase_a_push_performed",
        "safe_for_submission",
    }
    if set(payload) != expected_payload_keys:
        raise RuntimeError("existing reservation receipt top-level schema drift")
    created = payload.get("created_utc")
    if not isinstance(created, str) or not UTC_RE.fullmatch(created):
        raise RuntimeError("existing reservation receipt timestamp drift")
    local = validate_local_templates(root)
    if payload.get("local_validation") != local:
        raise RuntimeError("reservation receipt local template closure drift")
    orchestrator_sha256 = local["reservation_orchestrator_sha256"]
    datasets = payload.get("datasets")
    kernel = payload.get("kernel")
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("kind")
        != "candidate_graph_oracle_v4_kaggle_reservation_receipt"
        or payload.get("protocol_instance_id") != PROTOCOL_INSTANCE_ID
        or payload.get("reservation_orchestrator_sha256") != orchestrator_sha256
        or not isinstance(datasets, dict)
        or set(datasets) != {spec.role for spec in DATASET_SPECS}
        or not isinstance(kernel, dict)
        or payload.get("contains_fixture_pixels") is not False
        or payload.get("gpu_requested") is not False
        or payload.get("dataset_v2_uploaded") is not False
        or payload.get("phase_a_push_performed") is not False
        or payload.get("safe_for_submission") is not False
    ):
        raise RuntimeError("existing reservation receipt violates v4 fail-closed flags")
    expected_journal_files: set[Path] = set()
    for spec in DATASET_SPECS:
        expected_journal_files.update(
            _verify_dataset_receipt_record(
                record=datasets[spec.role],
                spec=spec,
                local=local["datasets"][spec.role],
                root=root,
                orchestrator_sha256=orchestrator_sha256,
            )
        )
    expected_journal_files.update(
        _verify_kernel_receipt_record(
            record=kernel,
            local=local["kernel"],
            root=root,
            orchestrator_sha256=orchestrator_sha256,
        )
    )
    actual_json_files = {
        value
        for value in (root / "journal").iterdir()
        if value.name.endswith(".json")
    }
    if actual_json_files != expected_journal_files:
        raise RuntimeError("reservation journal contains unbound JSON evidence")


def _validate_existing_receipt(path: Path) -> dict[str, Any]:
    path = path.absolute()
    root = path.parent
    if path != root / "RESERVATION_RECEIPT.json":
        raise RuntimeError("reservation receipt has a non-canonical path")
    envelope, _ = _load_canonical(
        path,
        expected_kind="candidate_graph_oracle_v4_kaggle_reservation_receipt",
        envelope=True,
    )
    _validate_receipt_payload(envelope["payload"], root=root)
    return envelope


def reserve_and_record(
    *,
    reservation_root: Path,
    receipt_path: Path,
    api: ReservationAPI,
    attempts: int = 10,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], str] = _utc_now,
) -> dict[str, Any]:
    if attempts < 1:
        raise RuntimeError("readback attempts must be positive")
    root = reservation_root.absolute()
    local = validate_local_templates(root)
    if receipt_path.absolute() != root / "RESERVATION_RECEIPT.json":
        raise RuntimeError("v4 reservation receipt path must be inside the exact reservation root")
    if receipt_path.exists() or receipt_path.is_symlink():
        envelope = _validate_existing_receipt(receipt_path)
        return {
            "status": "reservation_receipt_already_exists",
            "receipt_path": str(receipt_path),
            "payload_sha256": envelope["payload_sha256"],
            "remote_calls_performed": False,
        }

    journal_dir = root / "journal"
    datasets: dict[str, Any] = {}
    dataset_outcomes: dict[str, dict[str, Any]] = {}
    for spec in DATASET_SPECS:
        outcome = _reserve_dataset(
            api,
            spec=spec,
            local=local["datasets"][spec.role],
            root=root,
            journal_dir=journal_dir,
            orchestrator_sha256=local["reservation_orchestrator_sha256"],
            attempts=attempts,
            sleep=sleep,
            now=now,
        )
        dataset_outcomes[spec.role] = outcome
        readback = outcome["readback"]
        paths = outcome["paths"]
        datasets[spec.role] = {
            "slug": spec.slug,
            "dataset_id": readback["dataset_id"],
            "reserved_version": 1,
            "is_private": True,
            "status": "ready",
            "marker_sha256": local["datasets"][spec.role]["marker_sha256"],
            "exact_tree_sha256": local["datasets"][spec.role]["exact_tree_sha256"],
            "mode": outcome["mode"],
            "journal": _resource_journal(paths, root=root),
        }

    kernel_outcome = _reserve_kernel(
        api,
        local=local["kernel"],
        root=root,
        journal_dir=journal_dir,
        orchestrator_sha256=local["reservation_orchestrator_sha256"],
        attempts=attempts,
        sleep=sleep,
        now=now,
    )
    kernel_readback = kernel_outcome["readback"]
    kernel_paths = kernel_outcome["paths"]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "candidate_graph_oracle_v4_kaggle_reservation_receipt",
        "created_utc": now(),
        "protocol_instance_id": PROTOCOL_INSTANCE_ID,
        "reservation_orchestrator_sha256": local["reservation_orchestrator_sha256"],
        "local_validation": local,
        "datasets": datasets,
        "kernel": {
            "slug": KERNEL_SLUG,
            "kernel_id": kernel_readback["kernel_id"],
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
            "reservation_runner_sha256": RESERVATION_RUNNER_SHA256,
            "exact_tree_sha256": local["kernel"]["exact_tree_sha256"],
            "mode": kernel_outcome["mode"],
            "journal": _resource_journal(kernel_paths, root=root),
        },
        "contains_fixture_pixels": False,
        "gpu_requested": False,
        "dataset_v2_uploaded": False,
        "phase_a_push_performed": False,
        "safe_for_submission": False,
    }
    _validate_receipt_payload(payload, root=root)
    receipt_sha = _write_receipt(receipt_path, payload)
    write_performed_now = any(
        outcome["remote_write_performed"] for outcome in dataset_outcomes.values()
    ) or kernel_outcome["remote_write_performed"]
    return {
        "status": "private_version_1_reservations_attested",
        "receipt_path": str(receipt_path),
        "receipt_sha256": receipt_sha,
        "payload_sha256": _canonical_sha256(payload),
        "kernel_id": kernel_readback["kernel_id"],
        "remote_write_calls_performed_in_this_process": write_performed_now,
        "safe_for_submission": False,
    }


def _http_status(error: BaseException) -> int | None:
    response = getattr(error, "response", None)
    response_status = getattr(response, "status_code", None)
    direct_status = getattr(error, "status", None)
    for value in (response_status, direct_status):
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _is_not_found(error: BaseException) -> bool:
    return _http_status(error) == 404


def _authenticated_mine_dataset_ref_exists(api: Any, slug: str) -> bool:
    """Exhaustively check the authenticated account's dataset refs.

    Kaggle currently returns HTTP 403, not 404, for ``GetDataset`` against an
    unused private slug.  Treating every 403 as absence would be unsafe because
    it could also conceal an existing dataset and turn a create into version 2.
    The authenticated ``mine`` listing is therefore paged to exhaustion.  Any
    malformed or repeating page fails closed instead of claiming absence.
    """

    owner = slug.split("/", 1)[0].casefold()
    seen_pages: set[tuple[str, ...]] = set()
    exact_count = 0
    intended_owner_seen = False
    for page in range(1, MAX_MINE_DATASET_LIST_PAGES + 1):
        values = api.dataset_list(mine=True, page=page)
        if values is None:
            values = []
        if not isinstance(values, list):
            raise RuntimeError("authenticated mine dataset listing is not a list")
        refs: list[str] = []
        for value in values:
            ref = getattr(value, "ref", None)
            if not isinstance(ref, str) or KAGGLE_REF_RE.fullmatch(ref) is None:
                raise RuntimeError("authenticated mine dataset listing contains an invalid ref")
            refs.append(ref)
        exact_count += sum(ref.casefold() == slug.casefold() for ref in refs)
        intended_owner_seen = intended_owner_seen or any(
            ref.split("/", 1)[0].casefold() == owner for ref in refs
        )
        if not refs:
            break
        fingerprint = tuple(refs)
        if fingerprint in seen_pages:
            raise RuntimeError("authenticated mine dataset listing repeated a page")
        seen_pages.add(fingerprint)
    else:
        raise RuntimeError("authenticated mine dataset listing did not terminate")
    if exact_count > 1:
        raise RuntimeError("authenticated mine dataset listing contains duplicate exact refs")
    if exact_count == 1:
        return True
    if not intended_owner_seen:
        raise RuntimeError(
            "authenticated mine dataset listing did not establish the intended owner"
        )
    return False


def _authenticated_mine_kernel_ref_exists(api: Any, slug: str) -> bool:
    """Exhaustively exact-match a kernel in the authenticated profile list."""

    owner = slug.split("/", 1)[0].casefold()
    seen_pages: set[tuple[str, ...]] = set()
    exact_count = 0
    intended_owner_seen = False
    for page in range(1, MAX_MINE_KERNEL_LIST_PAGES + 1):
        values = api.kernels_list(mine=True, page=page, page_size=100)
        if values is None:
            values = []
        if not isinstance(values, list):
            raise RuntimeError("authenticated mine kernel listing is not a list")
        refs: list[str] = []
        for value in values:
            ref = getattr(value, "ref", None)
            if (
                ref == ""
                and getattr(value, "slug", None) == ""
                and getattr(value, "author", None) == ""
                and getattr(value, "title", None) == "[Private Notebook]"
            ):
                # Kaggle's profile endpoint includes an opaque redacted row for
                # a private notebook that cannot identify or occupy any slug.
                continue
            if not isinstance(ref, str) or KAGGLE_REF_RE.fullmatch(ref) is None:
                raise RuntimeError("authenticated mine kernel listing contains an invalid ref")
            refs.append(ref)
        exact_count += sum(ref.casefold() == slug.casefold() for ref in refs)
        intended_owner_seen = intended_owner_seen or any(
            ref.split("/", 1)[0].casefold() == owner for ref in refs
        )
        if not refs:
            break
        fingerprint = tuple(refs)
        if fingerprint in seen_pages:
            raise RuntimeError("authenticated mine kernel listing repeated a page")
        seen_pages.add(fingerprint)
    else:
        raise RuntimeError("authenticated mine kernel listing did not terminate")
    if exact_count > 1:
        raise RuntimeError("authenticated mine kernel listing contains duplicate exact refs")
    if exact_count == 1:
        return True
    if not intended_owner_seen:
        raise RuntimeError(
            "authenticated mine kernel listing did not establish the intended owner"
        )
    return False


class KaggleSdkReservationAPI:
    """Lazily imported production adapter; construction authenticates Kaggle."""

    def __init__(self) -> None:
        from kaggle.api.kaggle_api_extended import KaggleApi

        self._api = KaggleApi()
        self._api.authenticate()

    def create_dataset(self, directory: Path) -> Any:
        return self._api.dataset_create_new(
            str(directory),
            public=False,
            quiet=True,
            convert_to_csv=False,
            dir_mode="skip",
        )

    def create_kernel(self, directory: Path) -> Any:
        return self._api.kernels_push(str(directory), timeout=None, acc=None)

    def get_dataset_snapshot(self, slug: str, marker_name: str) -> dict[str, Any] | None:
        from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest

        owner, dataset_slug = slug.split("/", 1)
        try:
            with self._api.build_kaggle_client() as client:
                request = ApiGetDatasetRequest()
                request.owner_slug = owner
                request.dataset_slug = dataset_slug
                dataset = client.datasets.dataset_api_client.get_dataset(request)
        except Exception as error:
            if _is_not_found(error):
                return None
            if _http_status(error) == 403:
                try:
                    exact_ref_exists = _authenticated_mine_dataset_ref_exists(
                        self._api, slug
                    )
                except Exception as listing_error:
                    raise RuntimeError(
                        "unable to establish exact dataset absence after forbidden "
                        f"direct read; refusing create: {slug}"
                    ) from listing_error
                if exact_ref_exists:
                    return {
                        "status": {
                            "status": "pending",
                            "current_version_number": 1,
                        },
                        "sdk_objects": {
                            "get_dataset_forbidden_while_exact_mine_ref_visible": True
                        },
                    }
                return None
            raise
        status = json.loads(
            self._api.dataset_status(
                slug, format="json(status,current_version_number)"
            )
        )
        files_response = self._api.dataset_list_files(f"{slug}/1", page_size=20)
        files = list(files_response.files or [])
        marker_bytes: bytes | None = None
        if _status_text(status.get("status")) == "ready":
            with tempfile.TemporaryDirectory(prefix="oracle-v4-reservation-readback-") as temporary:
                self._api.dataset_download_file(
                    f"{slug}/1",
                    marker_name,
                    path=temporary,
                    force=True,
                    quiet=True,
                )
                marker_path = Path(temporary) / marker_name
                marker_bytes = marker_path.read_bytes()
        return {
            "dataset": {
                "id": int(dataset.id),
                "ref": str(dataset.ref),
                "title": str(dataset.title),
                "is_private": dataset.is_private,
                "current_version_number": int(dataset.current_version_number),
                "total_bytes": int(dataset.total_bytes),
            },
            "status": {
                "status": _status_text(status.get("status")),
                "current_version_number": int(status["current_version_number"]),
            },
            "file_list": {
                "files": [
                    {"name": str(item.name), "total_bytes": int(item.total_bytes)}
                    for item in files
                ],
                "next_page_token": files_response.next_page_token or None,
                "error_message": files_response.error_message or None,
            },
            "marker": {
                "name": marker_name,
                "base64": base64.b64encode(marker_bytes).decode("ascii") if marker_bytes is not None else None,
            },
            "sdk_objects": {
                "get_dataset": _raw_json_value(dataset),
                "list_dataset_files": _raw_json_value(files_response),
                "status": _raw_json_value(status),
            },
        }

    def get_kernel_snapshot(self, slug: str) -> dict[str, Any] | None:
        from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

        owner, kernel_slug = slug.split("/", 1)
        try:
            with self._api.build_kaggle_client() as client:
                request = ApiGetKernelRequest()
                request.user_name = owner
                request.kernel_slug = kernel_slug
                response = client.kernels.kernels_api_client.get_kernel(request)
        except Exception as error:
            if _is_not_found(error):
                return None
            if _http_status(error) == 403:
                try:
                    exact_ref_exists = _authenticated_mine_kernel_ref_exists(
                        self._api, slug
                    )
                except Exception as listing_error:
                    raise RuntimeError(
                        "unable to establish exact kernel absence after forbidden "
                        f"direct read; refusing push: {slug}"
                    ) from listing_error
                if exact_ref_exists:
                    return {
                        "status": {"status": "pending", "failure_message": ""},
                        "sdk_objects": {
                            "get_kernel_forbidden_while_exact_mine_ref_visible": True
                        },
                    }
                return None
            raise
        metadata = response.metadata
        blob = response.blob
        if metadata is None or blob is None or not isinstance(blob.source, str):
            raise RuntimeError("Kaggle kernel readback is incomplete")
        status_response = self._api.kernels_status(slug)
        return {
            "metadata": {
                "id": int(metadata.id),
                "ref": str(metadata.ref),
                "slug": str(metadata.slug),
                "title": str(metadata.title),
                "language": metadata.language,
                "kernel_type": metadata.kernel_type,
                "is_private": metadata.is_private,
                "enable_gpu": metadata.enable_gpu,
                "enable_tpu": metadata.enable_tpu,
                "enable_internet": metadata.enable_internet,
                "dataset_sources": list(metadata.dataset_data_sources or []),
                "kernel_sources": list(metadata.kernel_data_sources or []),
                "competition_sources": list(metadata.competition_data_sources or []),
                "model_sources": list(metadata.model_data_sources or []),
                "current_version_number": int(metadata.current_version_number),
            },
            "source": {
                "base64": base64.b64encode(blob.source.encode("utf-8")).decode("ascii")
            },
            "status": {
                "status": _status_text(status_response.status),
                "failure_message": status_response.failure_message or "",
            },
            "sdk_objects": {
                "get_kernel": _raw_json_value(response),
                "status": _raw_json_value(status_response),
            },
        }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true", default=True)
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--instance-id", default=PROTOCOL_INSTANCE_ID)
    parser.add_argument("--reservation-root", type=Path, default=DEFAULT_RESERVATION_ROOT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--readback-attempts", type=int, default=10)
    parser.add_argument(
        "--confirm-private-v1-reservations",
        action="store_true",
        help="Required with --execute; authorizes only the four pixel-free private v1 writes.",
    )
    args = parser.parse_args(argv)
    if args.execute:
        args.validate_only = False
    return args


def main(
    argv: Sequence[str] | None = None,
    *,
    api_factory: Callable[[], ReservationAPI] | None = None,
) -> int:
    args = parse_args(argv)
    if args.instance_id != PROTOCOL_INSTANCE_ID:
        raise RuntimeError("protocol instance id differs from the frozen v4 reservation")
    local = validate_local_templates(args.reservation_root)
    if args.validate_only:
        print(json.dumps(local, sort_keys=True, indent=2))
        return 0
    if not args.confirm_private_v1_reservations:
        raise RuntimeError("--execute requires --confirm-private-v1-reservations")
    factory = KaggleSdkReservationAPI if api_factory is None else api_factory
    api = factory()
    result = reserve_and_record(
        reservation_root=args.reservation_root,
        receipt_path=args.receipt,
        api=api,
        attempts=args.readback_attempts,
    )
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

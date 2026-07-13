#!/usr/bin/env python3
"""Launch a Phase-A kernel with a crash-safe evidence journal.

Kaggle does not permit version-qualified pulls for this private kernel: both
``slug/1`` and ``slug/2`` return HTTP 403.  Its unversioned GetKernel response
does expose ``currentVersionNumber`` and the current source, but normalizes
dataset sources by dropping ``/2`` and may report GPU/machine metadata as
false/None even when the executed runner later proves two T4s.  This launcher
therefore separates the evidence:

* dataset status calls independently prove all three private datasets are
  READY at exactly version 2;
* the exact SaveKernel response proves kernel id/version 2;
* an fsync'd three-file journal preserves the launch intent, the raw SDK push
  response, and its separately validated semantic projection;
* unversioned GetKernel proves the current version, normalized source set, and
  exact runner source hash;
* the separately verified Phase-A wrapper proves the actual two-T4 runtime,
  environment, and mounted-file hashes.

The raw SDK response is committed before any field coercion or semantic
validation.  Thus an unexpected-but-returned response can be diagnosed and a
fixed parser can resume from the durable raw journal without another push.  If
a process dies after Kaggle creates the intended version but before even the
raw response is durable, recovery still fails closed and never pushes again.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import time
from typing import Any, Mapping, Sequence

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest


SCHEMA_VERSION = 2
EXPECTED_PROTOCOL_INSTANCE_ID = "4f3da49d17e8adba46b1359d2cc81a19"
EXPECTED_KERNEL_SLUG = (
    "pasha883/vsos-candidate-graph-oracle-v3-phase-a-t4x2"
)
EXPECTED_KERNEL_ID = 126846203
RESERVATION_KERNEL_VERSION = 1
EXPECTED_KERNEL_VERSION = 2
RESERVATION_RUNNER_SHA256 = (
    "a30a195e4e761a4d9bbdd21367dc4c0f3be45c9b5373bf11ae40b2f3c6e2a24d"
)
EXPECTED_DATASETS = {
    "code": "pasha883/vsos-candidate-graph-oracle-v3-code",
    "input": "pasha883/vsos-candidate-graph-oracle-v3-inputs",
    "runtime": "pasha883/vsos-candidate-graph-oracle-v3-runtime",
}
INTENT_NAME = "00_launch.intent.json"
RAW_RESPONSE_NAME = "01_push.raw_response.json"
RESPONSE_NAME = "02_push.response.json"
RAW_RESPONSE_SCHEMA_VERSION = 1
RAW_RESPONSE_KIND = "candidate_graph_oracle_kaggle_raw_push_response"
RAW_RESPONSE_FIELDS = (
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
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _canonical_object_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    digest = hashlib.sha256()
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RuntimeError(f"launch artifact must be regular with nlink==1: {path}")
        for chunk in iter(lambda: os.read(descriptor, 1024 * 1024), b""):
            digest.update(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        encoded = _canonical_bytes(payload)
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise RuntimeError("short write to launch evidence")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)
    return _sha256(path)


def _load_canonical(
    path: Path,
    *,
    expected_kind: str,
    expected_schema_version: int = SCHEMA_VERSION,
) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != expected_schema_version
        or payload.get("kind") != expected_kind
        or raw != _canonical_bytes(payload)
    ):
        raise RuntimeError(f"launch evidence schema/canonical drift: {path.name}")
    return payload, hashlib.sha256(raw).hexdigest()


def _exclusive_receipt(path: Path, payload: Mapping[str, Any]) -> str:
    envelope_payload = dict(payload)
    payload_sha256 = hashlib.sha256(
        _canonical_object_bytes(envelope_payload)
    ).hexdigest()
    return _write_exclusive(
        path, {"payload": envelope_payload, "payload_sha256": payload_sha256}
    )


def _dataset_versions(api: KaggleApi) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for label, slug in EXPECTED_DATASETS.items():
        response = json.loads(
            api.dataset_status(
                slug, format="json(status,current_version_number)"
            )
        )
        if response != {"status": "ready", "current_version_number": 2}:
            raise RuntimeError(f"dataset is not frozen at ready version 2: {slug}")
        result[label] = {"slug": slug, "version": 2, "status": "ready"}
    return result


def _sdk_current_kernel(api: KaggleApi) -> tuple[dict[str, Any], str]:
    """Return stable unversioned metadata and source hash without writing files."""

    test_hook = getattr(api, "candidate_graph_oracle_current_readback", None)
    if callable(test_hook):
        metadata, source_sha256 = test_hook()
        return dict(metadata), str(source_sha256)

    owner, slug = EXPECTED_KERNEL_SLUG.split("/", 1)
    with api.build_kaggle_client() as client:
        request = ApiGetKernelRequest()
        request.user_name = owner
        request.kernel_slug = slug
        response = client.kernels.kernels_api_client.get_kernel(request)
    metadata = response.metadata
    blob = response.blob
    if metadata is None or blob is None or not isinstance(blob.source, str):
        raise RuntimeError("Kaggle unversioned GetKernel response is incomplete")
    normalized = {
        "id": int(metadata.id),
        "ref": str(metadata.ref),
        "title": str(metadata.title),
        "slug": str(metadata.slug),
        "language": metadata.language,
        "kernel_type": metadata.kernel_type,
        "is_private": metadata.is_private,
        "enable_gpu_observation": metadata.enable_gpu,
        "enable_internet": metadata.enable_internet,
        "enable_tpu_observation": metadata.enable_tpu,
        "dataset_sources": list(metadata.dataset_data_sources or []),
        "kernel_sources": list(metadata.kernel_data_sources or []),
        "competition_sources": list(metadata.competition_data_sources or []),
        "model_sources": list(metadata.model_data_sources or []),
        "current_version_number": int(metadata.current_version_number),
        "docker_image": metadata.docker_image,
        "machine_shape_observation": metadata.machine_shape,
    }
    source_sha256 = hashlib.sha256(blob.source.encode("utf-8")).hexdigest()
    return normalized, source_sha256


def _validate_current_readback(
    api: KaggleApi,
    *,
    expected_version: int,
    expected_source_sha256: str,
    expected_title: str,
) -> dict[str, Any]:
    metadata, source_sha256 = _sdk_current_kernel(api)
    exact_keys = {
        "id",
        "ref",
        "title",
        "slug",
        "language",
        "kernel_type",
        "is_private",
        "enable_gpu_observation",
        "enable_internet",
        "enable_tpu_observation",
        "dataset_sources",
        "kernel_sources",
        "competition_sources",
        "model_sources",
        "current_version_number",
        "docker_image",
        "machine_shape_observation",
    }
    if set(metadata) != exact_keys:
        raise RuntimeError("Kaggle unversioned metadata schema drift")
    expected_sources = (
        []
        if expected_version == RESERVATION_KERNEL_VERSION
        else list(EXPECTED_DATASETS.values())
    )
    if (
        metadata["id"] != EXPECTED_KERNEL_ID
        or metadata["ref"] != EXPECTED_KERNEL_SLUG
        or metadata["slug"] != EXPECTED_KERNEL_SLUG.split("/", 1)[1]
        or metadata["title"] != expected_title
        or metadata["language"] != "python"
        or metadata["kernel_type"] != "script"
        or metadata["is_private"] is not True
        or metadata["enable_internet"] is not False
        or metadata["dataset_sources"] != expected_sources
        or metadata["kernel_sources"] != []
        or metadata["competition_sources"] != []
        or metadata["model_sources"] != []
        or metadata["current_version_number"] != expected_version
        or source_sha256 != expected_source_sha256
    ):
        raise RuntimeError("Kaggle unversioned current kernel differs from expectation")
    # These two fields are observations only.  Kaggle has returned false/None
    # after a genuine 2xT4 execution, so the executed wrapper is authoritative.
    if metadata["enable_gpu_observation"] not in (True, False, None):
        raise RuntimeError("unexpected enable_gpu observation type")
    if metadata["machine_shape_observation"] is not None and not isinstance(
        metadata["machine_shape_observation"], str
    ):
        raise RuntimeError("unexpected machine_shape observation type")
    return {
        "access_mode": "unversioned_private_get_kernel",
        "version_qualified_pull_used": False,
        "metadata": metadata,
        "metadata_sha256": hashlib.sha256(
            _canonical_object_bytes(metadata)
        ).hexdigest(),
        "source_sha256": source_sha256,
    }


def _readback_with_retry(
    api: KaggleApi,
    *,
    runner_sha256: str,
    title: str,
    attempts: int,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return _validate_current_readback(
                api,
                expected_version=EXPECTED_KERNEL_VERSION,
                expected_source_sha256=runner_sha256,
                expected_title=title,
            )
        except Exception as error:  # SDK transports expose several exception types
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(2.0)
    raise RuntimeError(
        "Kaggle intended kernel version did not become verifiable"
    ) from last_error


def _raw_json_value(value: Any, *, seen: set[int] | None = None) -> Any:
    """Return a canonical-JSON-safe, representational snapshot of ``value``.

    Kaggle SDK response objects currently contain only primitive/list state,
    but this encoder deliberately handles a broader set of Python values.  It
    never relies on SDK semantic field expectations, so raw evidence can be
    committed even when a future SDK adds fields or changes a field type.
    Tagged forms preserve distinctions that plain JSON would otherwise lose.
    """

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {
            "__python_type__": "float",
            "value": "nan"
            if math.isnan(value)
            else ("inf" if value > 0 else "-inf"),
        }
    if isinstance(value, bytes):
        return {
            "__python_type__": "bytes",
            "base64": base64.b64encode(value).decode("ascii"),
        }
    if isinstance(value, bytearray):
        return {
            "__python_type__": "bytearray",
            "base64": base64.b64encode(bytes(value)).decode("ascii"),
        }
    if isinstance(value, Path):
        return {"__python_type__": "pathlib.Path", "value": str(value)}
    if isinstance(value, Enum):
        return {
            "__python_type__": (
                f"{type(value).__module__}.{type(value).__qualname__}"
            ),
            "name": value.name,
            "value": _raw_json_value(value.value, seen=seen),
        }

    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return {
            "__python_cycle__": (
                f"{type(value).__module__}.{type(value).__qualname__}"
            )
        }
    seen.add(identity)
    try:
        if isinstance(value, Mapping):
            if all(isinstance(key, str) for key in value):
                return {
                    str(key): _raw_json_value(item, seen=seen)
                    for key, item in value.items()
                }
            items = [
                [
                    _raw_json_value(key, seen=seen),
                    _raw_json_value(item, seen=seen),
                ]
                for key, item in value.items()
            ]
            items.sort(key=lambda item: _canonical_object_bytes({"key": item[0]}))
            return {"__python_type__": "mapping", "items": items}
        if isinstance(value, list):
            return [_raw_json_value(item, seen=seen) for item in value]
        if isinstance(value, tuple):
            return {
                "__python_type__": "tuple",
                "items": [_raw_json_value(item, seen=seen) for item in value],
            }
        if isinstance(value, (set, frozenset)):
            items = [_raw_json_value(item, seen=seen) for item in value]
            items.sort(key=lambda item: _canonical_object_bytes({"item": item}))
            return {
                "__python_type__": "frozenset"
                if isinstance(value, frozenset)
                else "set",
                "items": items,
            }
        if isinstance(value, Sequence):
            return {
                "__python_type__": (
                    f"{type(value).__module__}.{type(value).__qualname__}"
                ),
                "items": [_raw_json_value(item, seen=seen) for item in value],
            }
        try:
            state = vars(value)
        except TypeError:
            state = None
        if state is not None:
            return {
                "__python_type__": (
                    f"{type(value).__module__}.{type(value).__qualname__}"
                ),
                "state": _raw_json_value(state, seen=seen),
            }
        try:
            rendered = repr(value)
        except Exception as error:  # pragma: no cover - pathological fallback
            rendered = f"<repr failed: {type(error).__module__}.{type(error).__qualname__}>"
        return {
            "__python_type__": (
                f"{type(value).__module__}.{type(value).__qualname__}"
            ),
            "repr": rendered,
            "representation_only": True,
        }
    finally:
        seen.remove(identity)


def _raw_response_payload(response: Any) -> dict[str, Any]:
    response_type = type(response)
    public_fields: dict[str, Any] = {}
    for name in RAW_RESPONSE_FIELDS:
        try:
            value = getattr(response, name)
        except Exception as error:
            public_fields[name] = {
                "__attribute_error__": {
                    "type": f"{type(error).__module__}.{type(error).__qualname__}",
                    "message": str(error),
                }
            }
        else:
            public_fields[name] = _raw_json_value(value)
    try:
        state = vars(response)
    except TypeError:
        state = None
    return {
        "schema_version": RAW_RESPONSE_SCHEMA_VERSION,
        "kind": RAW_RESPONSE_KIND,
        "recorded_utc": _utc_now(),
        "response_type": {
            "module": response_type.__module__,
            "qualname": response_type.__qualname__,
        },
        "public_fields": public_fields,
        "object_state": _raw_json_value(state),
    }


def _validate_raw_response_payload(payload: Mapping[str, Any]) -> None:
    if (
        set(payload)
        != {
            "schema_version",
            "kind",
            "recorded_utc",
            "response_type",
            "public_fields",
            "object_state",
        }
        or payload.get("schema_version") != RAW_RESPONSE_SCHEMA_VERSION
        or payload.get("kind") != RAW_RESPONSE_KIND
        or not isinstance(payload.get("recorded_utc"), str)
        or not isinstance(payload.get("response_type"), dict)
        or set(payload["response_type"]) != {"module", "qualname"}
        or not all(
            isinstance(payload["response_type"].get(key), str)
            for key in ("module", "qualname")
        )
        or not isinstance(payload.get("public_fields"), dict)
        or set(payload["public_fields"]) != set(RAW_RESPONSE_FIELDS)
    ):
        raise RuntimeError("persisted raw Kaggle push response drift")


def _strict_string_list(value: Any, *, field: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise RuntimeError(f"Kaggle push response field {field} is not list[str]")
    return list(value)


def _push_response_payload(
    raw_payload: Mapping[str, Any], *, raw_response_sha256: str
) -> dict[str, Any]:
    _validate_raw_response_payload(raw_payload)
    if not SHA_RE.fullmatch(raw_response_sha256):
        raise RuntimeError("raw Kaggle push response SHA-256 is invalid")
    fields = raw_payload["public_fields"]
    if not isinstance(fields, dict):  # narrowed by validation; keeps mypy honest
        raise RuntimeError("raw Kaggle push response fields are invalid")
    ref = fields["ref"]
    kernel_id = fields["kernel_id"]
    version_number = fields["version_number"]
    url = fields["url"]
    error = fields["error"]
    if not isinstance(ref, str):
        raise RuntimeError("Kaggle push response field ref is not str")
    if isinstance(kernel_id, bool) or not isinstance(kernel_id, int):
        raise RuntimeError("Kaggle push response field kernel_id is not int")
    if isinstance(version_number, bool) or not isinstance(version_number, int):
        raise RuntimeError("Kaggle push response field version_number is not int")
    if not isinstance(url, str):
        raise RuntimeError("Kaggle push response field url is not str")
    if error is not None and not isinstance(error, str):
        raise RuntimeError("Kaggle push response field error is not str/null")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "candidate_graph_oracle_kaggle_push_response",
        "ref": ref,
        "kernel_id": kernel_id,
        "version_number": version_number,
        "url": url,
        "error": error,
        "invalid_dataset_sources": _strict_string_list(
            fields["invalid_dataset_sources"], field="invalid_dataset_sources"
        ),
        "invalid_competition_sources": _strict_string_list(
            fields["invalid_competition_sources"],
            field="invalid_competition_sources",
        ),
        "invalid_kernel_sources": _strict_string_list(
            fields["invalid_kernel_sources"], field="invalid_kernel_sources"
        ),
        "invalid_model_sources": _strict_string_list(
            fields["invalid_model_sources"], field="invalid_model_sources"
        ),
        "raw_response_file": RAW_RESPONSE_NAME,
        "raw_response_sha256": raw_response_sha256,
        "recorded_utc": _utc_now(),
    }
    if (
        payload["ref"] != EXPECTED_KERNEL_SLUG
        or payload["kernel_id"] != EXPECTED_KERNEL_ID
        or payload["version_number"] != EXPECTED_KERNEL_VERSION
        or not payload["url"].startswith("https://www.kaggle.com/")
        or payload["error"] not in (None, "")
        or any(
            payload[key]
            for key in (
                "invalid_dataset_sources",
                "invalid_competition_sources",
                "invalid_kernel_sources",
                "invalid_model_sources",
            )
        )
    ):
        raise RuntimeError("Kaggle push response violates frozen launch expectation")
    return payload


def _validate_response_payload(payload: Mapping[str, Any]) -> None:
    exact_keys = {
        "schema_version",
        "kind",
        "ref",
        "kernel_id",
        "version_number",
        "url",
        "error",
        "invalid_dataset_sources",
        "invalid_competition_sources",
        "invalid_kernel_sources",
        "invalid_model_sources",
        "raw_response_file",
        "raw_response_sha256",
        "recorded_utc",
    }
    if (
        set(payload) != exact_keys
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("kind") != "candidate_graph_oracle_kaggle_push_response"
        or payload.get("ref") != EXPECTED_KERNEL_SLUG
        or payload.get("kernel_id") != EXPECTED_KERNEL_ID
        or payload.get("version_number") != EXPECTED_KERNEL_VERSION
        or not isinstance(payload.get("url"), str)
        or not str(payload["url"]).startswith("https://www.kaggle.com/")
        or payload.get("error") not in (None, "")
        or payload.get("raw_response_file") != RAW_RESPONSE_NAME
        or not isinstance(payload.get("raw_response_sha256"), str)
        or not SHA_RE.fullmatch(str(payload["raw_response_sha256"]))
        or any(
            payload.get(key) != []
            for key in (
                "invalid_dataset_sources",
                "invalid_competition_sources",
                "invalid_kernel_sources",
                "invalid_model_sources",
            )
        )
    ):
        raise RuntimeError("persisted Kaggle push response drift")


def _intent_payload(
    *,
    metadata_sha256: str,
    runner_sha256: str,
    launcher_sha256: str,
    dataset_versions: Mapping[str, Any],
    reservation_readback: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "candidate_graph_oracle_kaggle_launch_intent",
        "created_utc": _utc_now(),
        "protocol_instance_id": EXPECTED_PROTOCOL_INSTANCE_ID,
        "kernel": {
            "slug": EXPECTED_KERNEL_SLUG,
            "kernel_id": EXPECTED_KERNEL_ID,
            "reserved_version": RESERVATION_KERNEL_VERSION,
            "intended_version": EXPECTED_KERNEL_VERSION,
        },
        "dataset_versions": dict(dataset_versions),
        "local_kernel_metadata_sha256": metadata_sha256,
        "local_runner_sha256": runner_sha256,
        "local_launcher_sha256": launcher_sha256,
        "reservation_readback": dict(reservation_readback),
        "safe_for_submission": False,
    }


def _validate_intent(
    payload: Mapping[str, Any],
    *,
    metadata_sha256: str,
    runner_sha256: str,
    launcher_sha256: str,
    dataset_versions: Mapping[str, Any],
) -> None:
    if (
        set(payload)
        != {
            "schema_version",
            "kind",
            "created_utc",
            "protocol_instance_id",
            "kernel",
            "dataset_versions",
            "local_kernel_metadata_sha256",
            "local_runner_sha256",
            "local_launcher_sha256",
            "reservation_readback",
            "safe_for_submission",
        }
        or payload.get("protocol_instance_id") != EXPECTED_PROTOCOL_INSTANCE_ID
        or payload.get("kernel")
        != {
            "slug": EXPECTED_KERNEL_SLUG,
            "kernel_id": EXPECTED_KERNEL_ID,
            "reserved_version": RESERVATION_KERNEL_VERSION,
            "intended_version": EXPECTED_KERNEL_VERSION,
        }
        or payload.get("dataset_versions") != dataset_versions
        or payload.get("local_kernel_metadata_sha256") != metadata_sha256
        or payload.get("local_runner_sha256") != runner_sha256
        or payload.get("local_launcher_sha256") != launcher_sha256
        or payload.get("safe_for_submission") is not False
    ):
        raise RuntimeError("persisted Kaggle launch intent drift")


def push_and_record(
    *,
    job_dir: Path,
    receipt_path: Path,
    state_dir: Path | None = None,
    api: KaggleApi | None = None,
) -> dict[str, Any]:
    job_dir = job_dir.expanduser().resolve(strict=True)
    receipt_path = receipt_path.expanduser().absolute()
    state_dir = (
        receipt_path.parent / "candidate_graph_oracle_v3_launch_state"
        if state_dir is None
        else state_dir.expanduser().absolute()
    )
    if receipt_path.exists() or receipt_path.is_symlink():
        raise RuntimeError("launch receipt path must be fresh")
    if state_dir.is_symlink():
        raise RuntimeError("launch state directory may not be a symlink")
    state_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = job_dir / "kernel-metadata.json"
    metadata = _load_json(metadata_path)
    code_file = metadata.get("code_file")
    if not isinstance(code_file, str):
        raise RuntimeError("kernel metadata code_file is missing")
    runner_path = job_dir / code_file
    expectation = metadata.get("oracle_launch_expectation")
    expected_sources = [f"{slug}/2" for slug in EXPECTED_DATASETS.values()]
    expected_expectation = {
        "kernel_id": EXPECTED_KERNEL_ID,
        "kernel_slug": EXPECTED_KERNEL_SLUG,
        "kernel_version": EXPECTED_KERNEL_VERSION,
        "dataset_versions": {
            label: {"slug": slug, "version": 2}
            for label, slug in EXPECTED_DATASETS.items()
        },
    }
    if (
        metadata.get("id") != EXPECTED_KERNEL_SLUG
        or metadata.get("id_no") != EXPECTED_KERNEL_ID
        or metadata.get("dataset_sources") != expected_sources
        or metadata.get("is_private") is not True
        or metadata.get("enable_gpu") is not True
        or metadata.get("enable_internet") is not False
        or metadata.get("machine_shape") != "NvidiaTeslaT4"
        or expectation != expected_expectation
    ):
        raise RuntimeError("kernel metadata does not match frozen launch expectation")
    metadata_sha256 = _sha256(metadata_path)
    runner_sha256 = _sha256(runner_path)
    launcher_sha256 = _sha256(Path(__file__).resolve())

    kaggle = KaggleApi() if api is None else api
    if api is None:
        kaggle.authenticate()
    datasets_before = _dataset_versions(kaggle)
    intent_path = state_dir / INTENT_NAME
    raw_response_path = state_dir / RAW_RESPONSE_NAME
    response_path = state_dir / RESPONSE_NAME

    if intent_path.exists() or intent_path.is_symlink():
        if intent_path.is_symlink():
            raise RuntimeError("launch intent may not be a symlink")
        intent, intent_sha256 = _load_canonical(
            intent_path,
            expected_kind="candidate_graph_oracle_kaggle_launch_intent",
        )
        _validate_intent(
            intent,
            metadata_sha256=metadata_sha256,
            runner_sha256=runner_sha256,
            launcher_sha256=launcher_sha256,
            dataset_versions=datasets_before,
        )
    else:
        reservation = _validate_current_readback(
            kaggle,
            expected_version=RESERVATION_KERNEL_VERSION,
            expected_source_sha256=RESERVATION_RUNNER_SHA256,
            expected_title="VSOS Candidate Graph Oracle V3 Phase A T4x2",
        )
        intent = _intent_payload(
            metadata_sha256=metadata_sha256,
            runner_sha256=runner_sha256,
            launcher_sha256=launcher_sha256,
            dataset_versions=datasets_before,
            reservation_readback=reservation,
        )
        intent_sha256 = _write_exclusive(intent_path, intent)

    if response_path.exists() or response_path.is_symlink():
        if response_path.is_symlink():
            raise RuntimeError("push response journal may not be a symlink")
        if not raw_response_path.exists() or raw_response_path.is_symlink():
            raise RuntimeError(
                "validated push response exists without its regular raw journal"
            )
        raw_response_payload, raw_response_sha256 = _load_canonical(
            raw_response_path,
            expected_kind=RAW_RESPONSE_KIND,
            expected_schema_version=RAW_RESPONSE_SCHEMA_VERSION,
        )
        _validate_raw_response_payload(raw_response_payload)
        response_payload, response_sha256 = _load_canonical(
            response_path,
            expected_kind="candidate_graph_oracle_kaggle_push_response",
        )
        _validate_response_payload(response_payload)
        if response_payload["raw_response_sha256"] != raw_response_sha256:
            raise RuntimeError("validated push response does not bind raw journal")
        push_performed_now = False
        response_recovered_from_raw = False
    elif raw_response_path.exists() or raw_response_path.is_symlink():
        if raw_response_path.is_symlink():
            raise RuntimeError("raw push response journal may not be a symlink")
        raw_response_payload, raw_response_sha256 = _load_canonical(
            raw_response_path,
            expected_kind=RAW_RESPONSE_KIND,
            expected_schema_version=RAW_RESPONSE_SCHEMA_VERSION,
        )
        _validate_raw_response_payload(raw_response_payload)
        # This is the deliberate recovery path for a parser/validation crash.
        # It never calls Kaggle's push endpoint again.
        response_payload = _push_response_payload(
            raw_response_payload,
            raw_response_sha256=raw_response_sha256,
        )
        response_sha256 = _write_exclusive(response_path, response_payload)
        push_performed_now = False
        response_recovered_from_raw = True
    else:
        # A retry may push only while the exact reservation v1 is still current.
        # If the intended remote version already exists without a durable raw
        # response record, the launch is intentionally unrecoverable and must
        # never advance to another version.
        try:
            _validate_current_readback(
                kaggle,
                expected_version=RESERVATION_KERNEL_VERSION,
                expected_source_sha256=RESERVATION_RUNNER_SHA256,
                expected_title="VSOS Candidate Graph Oracle V3 Phase A T4x2",
            )
        except Exception as error:
            raise RuntimeError(
                "kernel is no longer the exact version-1 reservation but no "
                "durable raw push response exists; refusing any retry that could "
                "create another kernel version"
            ) from error
        response = kaggle.kernels_push(str(job_dir), timeout=None, acc=None)
        # Commit the raw SDK object before parsing or checking any response
        # field.  A semantic rejection below therefore remains diagnosable and
        # recoverable without a second remote write.
        raw_response_payload = _raw_response_payload(response)
        raw_response_sha256 = _write_exclusive(
            raw_response_path, raw_response_payload
        )
        response_payload = _push_response_payload(
            raw_response_payload,
            raw_response_sha256=raw_response_sha256,
        )
        response_sha256 = _write_exclusive(response_path, response_payload)
        push_performed_now = True
        response_recovered_from_raw = False

    current = _readback_with_retry(
        kaggle,
        runner_sha256=runner_sha256,
        title=str(metadata["title"]),
        attempts=10,
    )
    datasets_after = _dataset_versions(kaggle)
    if datasets_after != datasets_before:
        raise RuntimeError("dataset status/version changed across kernel launch")
    if (
        _sha256(intent_path) != intent_sha256
        or _sha256(raw_response_path) != raw_response_sha256
        or _sha256(response_path) != response_sha256
    ):
        raise RuntimeError("launch journal changed before receipt commit")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "candidate_graph_oracle_kaggle_launch_receipt",
        "created_utc": _utc_now(),
        "protocol_instance_id": EXPECTED_PROTOCOL_INSTANCE_ID,
        "kernel": {
            "slug": response_payload["ref"],
            "kernel_id": response_payload["kernel_id"],
            "version": response_payload["version_number"],
            "url": response_payload["url"],
        },
        "dataset_versions_before_push": datasets_before,
        "dataset_versions_after_push": datasets_after,
        "local_kernel_metadata_sha256": metadata_sha256,
        "local_runner_sha256": runner_sha256,
        "local_launcher_sha256": launcher_sha256,
        "launch_journal": {
            "intent_file": INTENT_NAME,
            "intent_sha256": intent_sha256,
            "raw_push_response_file": RAW_RESPONSE_NAME,
            "raw_push_response_sha256": raw_response_sha256,
            "push_response_file": RESPONSE_NAME,
            "push_response_sha256": response_sha256,
        },
        "launch_intent": intent,
        "raw_push_response": raw_response_payload,
        "push_response": response_payload,
        "server_readback": current,
        "gpu_and_machine_metadata_authority": (
            "executed_phase_a_wrapper_hardware_not_normalized_get_kernel_metadata"
        ),
        "push_performed_in_this_process": push_performed_now,
        "push_response_recovered_from_raw_journal": response_recovered_from_raw,
        "safe_for_submission": False,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_sha256 = _exclusive_receipt(receipt_path, payload)
    return {
        "status": "kernel_version_2_pushed_and_attested",
        "receipt_path": str(receipt_path),
        "receipt_sha256": receipt_sha256,
        "state_dir": str(state_dir),
        "intent_sha256": intent_sha256,
        "raw_push_response_sha256": raw_response_sha256,
        "push_response_sha256": response_sha256,
        "kernel_id": EXPECTED_KERNEL_ID,
        "kernel_version": EXPECTED_KERNEL_VERSION,
        "safe_for_submission": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = push_and_record(
        job_dir=args.job_dir,
        receipt_path=args.receipt,
        state_dir=args.state_dir,
    )
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()

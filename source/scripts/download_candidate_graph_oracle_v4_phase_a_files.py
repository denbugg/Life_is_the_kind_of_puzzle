#!/usr/bin/env python3
"""Download or locally validate every manifest-listed v4 Phase-A file.

``--validate-only`` is deliberately API-free: Kaggle modules are imported only
inside the remote download path, after the local manifest has been validated.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import threading
import time
from typing import Any, Callable, Mapping, Sequence


KERNEL = "pasha883/vsos-candidate-graph-oracle-v4-phase-a-t4x2"
VERSION = 2
INSTANCE = "6c0fe4e8524ce39d830d9a5bee118d8b"
FROZEN_CONTRACT_SHA256 = (
    "2070c1b4ff0a3ff42c5ffdd6d611c214c02dbb99b2c51a88f38a862bb1f8a05c"
)
PREFIX = "candidate_graph_oracle_v4_phase_a/finalized/"
MANIFEST_RELATIVE = PREFIX + "FROZEN_CANDIDATE_GRAPH_MANIFEST.json"
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
OPAQUE_RE = re.compile(r"^[0-9a-f]{32}$")
RENDER_LABELS = ("qap_w1", "qap_w4", "softcycle")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_regular(path: Path) -> tuple[bytes, os.stat_result]:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    chunks: list[bytes] = []
    try:
        info = os.fstat(descriptor)
        _require(
            stat.S_ISREG(info.st_mode) and info.st_nlink == 1,
            f"not a one-link regular file: {path}",
        )
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    return b"".join(chunks), info


def _sha(path: Path) -> str:
    raw, _ = _read_regular(path)
    return hashlib.sha256(raw).hexdigest()


def _decode_manifest(raw: bytes, *, path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_pairs
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid UTF-8 manifest JSON: {path}") from error
    _require(isinstance(value, dict), "Phase-A manifest must be an object")
    return value


def _canonical_object(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _require_sha(value: Any, *, label: str) -> str:
    _require(
        isinstance(value, str) and SHA_RE.fullmatch(value) is not None,
        f"{label} is not a lowercase SHA-256",
    )
    return value


def _relative(value: Any, *, label: str) -> str:
    _require(isinstance(value, str) and value, f"{label} must be a relative path")
    _require("\x00" not in value and "\\" not in value, f"{label} has a forbidden separator")
    pure = PurePosixPath(value)
    _require(
        not pure.is_absolute()
        and pure.parts
        and all(part not in ("", ".", "..") for part in pure.parts)
        and pure.as_posix() == value,
        f"{label} is not a canonical relative POSIX path",
    )
    _require(
        not any(token in value.lower() for token in ("label", "target", "secret")),
        f"{label} contains a forbidden Phase-A token",
    )
    return value


def _validate_manifest_header(
    manifest: Mapping[str, Any], *, expected_instance_id: str
) -> None:
    _require(
        isinstance(expected_instance_id, str)
        and OPAQUE_RE.fullmatch(expected_instance_id) is not None,
        "expected instance id must be 32 lowercase hex characters",
    )
    _require(
        manifest.get("schema_version") == 1
        and manifest.get("kind") == "frozen_candidate_graph_input_only",
        "wrong Phase-A manifest schema",
    )
    _require(
        manifest.get("protocol_instance_id") == expected_instance_id,
        "Phase-A manifest instance drift",
    )
    _require(manifest.get("record_count") == 64, "Phase-A record count is not 64")
    _require(
        manifest.get("target_files_opened") is False
        and manifest.get("target_paths_constructed") is False
        and manifest.get("safe_for_submission") is False,
        "Phase-A input-only safety header drift",
    )
    for field in (
        "config_sha256",
        "fixture_manifest_sha256",
        "frozen_contract_sha256",
        "phase_a_lifecycle_sha256",
        "script_sha256",
        "self_sha256",
    ):
        _require_sha(manifest.get(field), label=field)
    _require(
        manifest.get("frozen_contract_sha256") == FROZEN_CONTRACT_SHA256,
        "Phase-A manifest frozen contract drift",
    )
    expected_self = _require_sha(manifest.get("self_sha256"), label="self_sha256")
    base = {key: value for key, value in manifest.items() if key != "self_sha256"}
    _require(
        hashlib.sha256(_canonical_object(base)).hexdigest() == expected_self,
        "Phase-A manifest self_sha256 mismatch",
    )


def manifest_tasks(manifest: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Return the exact 64 graph plus 192 render closure."""

    records = manifest.get("records")
    _require(isinstance(records, list) and len(records) == 64, "manifest records drift")
    tasks: list[tuple[str, str]] = []
    opaque_ids: set[str] = set()
    for index, record in enumerate(records):
        _require(isinstance(record, dict), f"record {index} is not an object")
        opaque_id = record.get("opaque_id")
        _require(
            isinstance(opaque_id, str)
            and OPAQUE_RE.fullmatch(opaque_id) is not None
            and opaque_id not in opaque_ids,
            f"record {index} opaque id drift",
        )
        opaque_ids.add(opaque_id)
        artifact = _relative(record.get("graph_artifact"), label=f"record {index} graph")
        _require(
            artifact == f"artifacts/{opaque_id}.graph.npz",
            f"record {index} graph filename is not bound to opaque id",
        )
        tasks.append(
            (artifact, _require_sha(record.get("graph_artifact_sha256"), label="graph SHA"))
        )
        renders = record.get("renders")
        _require(
            isinstance(renders, dict) and set(renders) == set(RENDER_LABELS),
            f"record {index} render closure drift",
        )
        for label in RENDER_LABELS:
            descriptor = renders[label]
            _require(isinstance(descriptor, dict), f"record {index} render descriptor drift")
            relative = _relative(
                descriptor.get("path"), label=f"record {index} render {label}"
            )
            _require(
                relative == f"renders/{opaque_id}__{label}.png",
                f"record {index} render filename is not bound to opaque id",
            )
            tasks.append(
                (relative, _require_sha(descriptor.get("sha256"), label="render SHA"))
            )
    tasks.sort()
    _require(
        len(tasks) == 64 * 4 and len({path for path, _ in tasks}) == len(tasks),
        "manifest file closure is not 64 graphs plus 192 renders",
    )
    return tasks


def load_manifest(
    readback_root: Path, *, expected_instance_id: str = INSTANCE
) -> tuple[dict[str, Any], list[tuple[str, str]], str]:
    root = readback_root.expanduser().resolve(strict=True)
    finalized = root / PREFIX
    _require(
        finalized.is_dir() and not finalized.is_symlink(),
        "finalized Phase-A path is not a real directory",
    )
    manifest_path = root / MANIFEST_RELATIVE
    raw, _ = _read_regular(manifest_path)
    manifest = _decode_manifest(raw, path=manifest_path)
    _validate_manifest_header(manifest, expected_instance_id=expected_instance_id)
    tasks = manifest_tasks(manifest)
    return manifest, tasks, hashlib.sha256(raw).hexdigest()


def _assert_exact_local_tree(root: Path, tasks: Sequence[tuple[str, str]]) -> None:
    finalized = root / PREFIX
    expected_top = {"FROZEN_CANDIDATE_GRAPH_MANIFEST.json", "artifacts", "renders"}
    _require(finalized.is_dir(), "finalized Phase-A directory is missing")
    actual_top = {entry.name for entry in os.scandir(finalized)}
    _require(actual_top == expected_top, "finalized Phase-A top-level closure drift")
    for directory in ("artifacts", "renders"):
        path = finalized / directory
        _require(path.is_dir() and not path.is_symlink(), f"{directory} is not a real directory")
        expected = {
            PurePosixPath(relative).name
            for relative, _ in tasks
            if PurePosixPath(relative).parent.as_posix() == directory
        }
        actual = {entry.name for entry in os.scandir(path)}
        _require(actual == expected, f"{directory} file closure drift")


def validate_local(
    readback_root: Path,
    *,
    expected_instance_id: str = INSTANCE,
    exact_tree: bool = True,
) -> dict[str, Any]:
    """Validate an already-downloaded closure without importing Kaggle."""

    root = readback_root.expanduser().resolve(strict=True)
    _, tasks, manifest_file_sha = load_manifest(
        root, expected_instance_id=expected_instance_id
    )
    for directory in ("artifacts", "renders"):
        path = root / PREFIX / directory
        _require(
            path.is_dir() and not path.is_symlink(),
            f"{directory} is not a real directory",
        )
    total_bytes = 0
    for relative, expected_sha in tasks:
        path = root / PREFIX / relative
        raw, _ = _read_regular(path)
        actual_sha = hashlib.sha256(raw).hexdigest()
        _require(actual_sha == expected_sha, f"local Phase-A SHA drift: {relative}")
        total_bytes += len(raw)
    if exact_tree:
        _assert_exact_local_tree(root, tasks)
    return {
        "status": "local_v4_phase_a_closure_validated",
        "protocol_instance_id": expected_instance_id,
        "kernel": KERNEL,
        "kernel_version": VERSION,
        "manifest_file_sha256": manifest_file_sha,
        "verified_files": len(tasks),
        "verified_bytes": total_bytes,
        "validate_only": True,
        "remote_api_called": False,
        "target_files_opened": False,
        "target_paths_constructed": False,
        "label_fixture_accessed": False,
    }


def _atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(data):
            written = os.write(descriptor, data[offset:])
            _require(written > 0, "short output-file write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    parent = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def _new_api() -> Any:
    # The import belongs here so --validate-only has no Kaggle dependency or call.
    from kaggle.api.kaggle_api_extended import KaggleApi

    return KaggleApi()


def _download_request_type() -> type[Any]:
    from kagglesdk.kernels.types.kernels_api_service import (
        ApiDownloadKernelOutputRequest,
    )

    return ApiDownloadKernelOutputRequest


def _download_group(
    *,
    root: Path,
    group: Sequence[tuple[str, str]],
    counter: list[int],
    lock: threading.Lock,
    request_delay_seconds: float,
    max_429_retries: int,
    api_factory: Callable[[], Any],
) -> tuple[int, int]:
    from requests import HTTPError

    owner, slug = KERNEL.split("/", 1)
    api = api_factory()
    api.authenticate()
    request_type = _download_request_type()
    downloaded = 0
    downloaded_bytes = 0
    with api.build_kaggle_client() as client:
        for relative, expected_sha in group:
            local = root / PREFIX / relative
            try:
                if _sha(local) == expected_sha:
                    with lock:
                        counter[0] += 1
                    continue
            except FileNotFoundError:
                pass
            request = request_type()
            request.owner_slug = owner
            request.kernel_slug = slug
            request.file_path = PREFIX + relative
            request.version_number = VERSION
            response = None
            for attempt in range(max_429_retries + 1):
                try:
                    response = client.kernels.kernels_api_client.download_kernel_output(
                        request
                    )
                    break
                except HTTPError as error:
                    if (
                        error.response is None
                        or error.response.status_code != 429
                        or attempt == max_429_retries
                    ):
                        raise
                    retry_after = error.response.headers.get("Retry-After")
                    try:
                        retry_seconds = float(retry_after) if retry_after else 0.0
                    except ValueError:
                        retry_seconds = 0.0
                    time.sleep(max(retry_seconds, min(60.0, 5.0 * 2**attempt)))
            _require(response is not None, f"download response missing: {relative}")
            _require(response.status_code == 200, f"download failed: {relative}")
            data = bytes(response.content)
            _require(
                hashlib.sha256(data).hexdigest() == expected_sha,
                f"downloaded SHA drift: {relative}",
            )
            _atomic(local, data)
            downloaded += 1
            downloaded_bytes += len(data)
            with lock:
                counter[0] += 1
            if request_delay_seconds:
                time.sleep(request_delay_seconds)
    return downloaded, downloaded_bytes


def download(
    readback_root: Path,
    *,
    expected_instance_id: str = INSTANCE,
    workers: int = 4,
    request_delay_seconds: float = 0.0,
    max_429_retries: int = 8,
    api_factory: Callable[[], Any] = _new_api,
) -> dict[str, Any]:
    _require(1 <= workers <= 8, "workers must be in [1, 8]")
    _require(0.0 <= request_delay_seconds <= 60.0, "request delay must be in [0, 60]")
    _require(0 <= max_429_retries <= 16, "max 429 retries must be in [0, 16]")
    root = readback_root.expanduser().resolve(strict=True)
    _, tasks, _ = load_manifest(root, expected_instance_id=expected_instance_id)
    groups = [tasks[index::workers] for index in range(workers)]
    counter = [0]
    counter_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _download_group,
                root=root,
                group=group,
                counter=counter,
                lock=counter_lock,
                request_delay_seconds=request_delay_seconds,
                max_429_retries=max_429_retries,
                api_factory=api_factory,
            )
            for group in groups
            if group
        ]
        results = [future.result() for future in futures]
    validated = validate_local(
        root, expected_instance_id=expected_instance_id, exact_tree=True
    )
    validated.update(
        {
            "status": "downloaded_and_validated_v4_phase_a_closure",
            "validate_only": False,
            "remote_api_called": True,
            "new_files": sum(item[0] for item in results),
            "new_bytes": sum(item[1] for item in results),
        }
    )
    return validated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readback-root", type=Path, required=True)
    parser.add_argument("--expected-instance-id", default=INSTANCE)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--allow-extra-files", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--request-delay-seconds", type=float, default=0.0)
    parser.add_argument("--max-429-retries", type=int, default=8)
    args = parser.parse_args()
    if args.validate_only:
        result = validate_local(
            args.readback_root,
            expected_instance_id=args.expected_instance_id,
            exact_tree=not args.allow_extra_files,
        )
    else:
        _require(not args.allow_extra_files, "--allow-extra-files is validate-only")
        result = download(
            args.readback_root,
            expected_instance_id=args.expected_instance_id,
            workers=args.workers,
            request_delay_seconds=args.request_delay_seconds,
            max_429_retries=args.max_429_retries,
        )
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()

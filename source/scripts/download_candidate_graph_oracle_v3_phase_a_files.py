#!/usr/bin/env python3
"""Download every manifest-listed v3 Phase-A file through the per-file API."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from typing import Any

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest
from requests import HTTPError


KERNEL = "pasha883/vsos-candidate-graph-oracle-v3-phase-a-t4x2"
VERSION = 2
PREFIX = "candidate_graph_oracle_v3_phase_a/finalized/"
MANIFEST_RELATIVE = "candidate_graph_oracle_v3_phase_a/finalized/FROZEN_CANDIDATE_GRAPH_MANIFEST.json"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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
            if written <= 0:
                raise RuntimeError("short output-file write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def _tasks(manifest: dict[str, Any]) -> list[tuple[str, str]]:
    tasks: list[tuple[str, str]] = []
    for record in manifest["records"]:
        artifact = record["graph_artifact"]
        tasks.append((artifact, record["graph_artifact_sha256"]))
        for render in record["renders"].values():
            tasks.append((render["path"], render["sha256"]))
    tasks.sort()
    if len(tasks) != 64 * 4 or len({path for path, _ in tasks}) != len(tasks):
        raise RuntimeError("manifest file closure is not 64 graphs plus 192 renders")
    if any(
        path.startswith("/")
        or ".." in Path(path).parts
        or any(token in path.lower() for token in ("label", "target", "secret"))
        for path, _ in tasks
    ):
        raise RuntimeError("forbidden or non-relative Phase-A output path")
    return tasks


def _download_group(
    *,
    root: Path,
    group: list[tuple[str, str]],
    counter: list[int],
    lock: threading.Lock,
    request_delay_seconds: float,
    max_429_retries: int,
) -> tuple[int, int]:
    owner, slug = KERNEL.split("/", 1)
    api = KaggleApi()
    api.authenticate()
    downloaded = 0
    downloaded_bytes = 0
    with api.build_kaggle_client() as client:
        for relative, expected_sha in group:
            local = root / PREFIX / relative
            if local.is_file() and _sha(local) == expected_sha:
                with lock:
                    counter[0] += 1
                continue
            request = ApiDownloadKernelOutputRequest()
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
                        retry_after_seconds = float(retry_after) if retry_after else 0.0
                    except ValueError:
                        retry_after_seconds = 0.0
                    backoff = max(retry_after_seconds, min(60.0, 5.0 * 2**attempt))
                    print(
                        f"rate_limited file={relative} retry={attempt + 1}/"
                        f"{max_429_retries} backoff_seconds={backoff:g}",
                        flush=True,
                    )
                    time.sleep(backoff)
            if response is None:
                raise RuntimeError(f"download response missing for {relative}")
            if response.status_code != 200:
                raise RuntimeError(
                    f"download failed for {relative}: HTTP {response.status_code}"
                )
            data = bytes(response.content)
            actual_sha = hashlib.sha256(data).hexdigest()
            if actual_sha != expected_sha:
                raise RuntimeError(
                    f"downloaded SHA drift for {relative}: {actual_sha} != {expected_sha}"
                )
            _atomic(local, data)
            downloaded += 1
            downloaded_bytes += len(data)
            with lock:
                counter[0] += 1
                if counter[0] % 16 == 0 or counter[0] == 256:
                    print(f"verified_files={counter[0]}/256", flush=True)
            if request_delay_seconds:
                time.sleep(request_delay_seconds)
    return downloaded, downloaded_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readback-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--request-delay-seconds", type=float, default=0.0)
    parser.add_argument("--max-429-retries", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 8:
        raise RuntimeError("workers must be in [1, 8]")
    if args.request_delay_seconds < 0.0 or args.request_delay_seconds > 60.0:
        raise RuntimeError("request delay must be in [0, 60] seconds")
    if args.max_429_retries < 0 or args.max_429_retries > 16:
        raise RuntimeError("max 429 retries must be in [0, 16]")
    root = args.readback_root.expanduser().resolve(strict=True)
    manifest_path = root / MANIFEST_RELATIVE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("protocol_instance_id")
        != "4f3da49d17e8adba46b1359d2cc81a19"
        or manifest.get("record_count") != 64
        or manifest.get("target_files_opened") is not False
        or manifest.get("target_paths_constructed") is not False
    ):
        raise RuntimeError("input-only Phase-A manifest header drift")
    tasks = _tasks(manifest)
    groups = [tasks[index :: args.workers] for index in range(args.workers)]
    counter = [0]
    counter_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(
            executor.map(
                lambda group: _download_group(
                    root=root,
                    group=group,
                    counter=counter,
                    lock=counter_lock,
                    request_delay_seconds=args.request_delay_seconds,
                    max_429_retries=args.max_429_retries,
                ),
                groups,
            )
        )
    for relative, expected_sha in tasks:
        path = root / PREFIX / relative
        if not path.is_file() or _sha(path) != expected_sha:
            raise RuntimeError(f"post-download verification failed: {relative}")
    print(
        json.dumps(
            {
                "status": "all_manifest_listed_phase_a_files_downloaded",
                "verified_files": len(tasks),
                "new_files": sum(item[0] for item in results),
                "new_bytes": sum(item[1] for item in results),
                "target_files_opened": False,
                "target_paths_constructed": False,
                "label_fixture_accessed": False,
            },
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

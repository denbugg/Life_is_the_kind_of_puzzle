"""Fail-closed launch handshake for an autonomous E26 stage.

The parent starts this tiny process with stdin connected to a private pipe.  It
authenticates a canonical launch request, then blocks.  Only after the parent
has durably recorded this PID does it send ``GO <request-sha256>``.  If the
parent disappears anywhere in the Popen-to-ledger window, stdin reaches EOF and
the scientific target is never executed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


SCHEMA = "pazzle-e26-stage-launch-request-v1"


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--request-sha256", required=True)
    args = parser.parse_args()
    request_path = args.request.resolve()
    if sha256_file(request_path) != args.request_sha256:
        print("launch request SHA-256 mismatch", file=sys.stderr, flush=True)
        return 90
    raw = request_path.read_bytes()
    try:
        request = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(f"invalid launch request: {exc}", file=sys.stderr, flush=True)
        return 91
    if raw != canonical_json(request) or not isinstance(request, dict):
        print("launch request is not a canonical object", file=sys.stderr, flush=True)
        return 92
    if request.get("schema") != SCHEMA:
        print("launch request schema mismatch", file=sys.stderr, flush=True)
        return 93
    if set(request) != {
        "schema", "argv", "argv_sha256", "environment_sha256",
        "working_directory", "execution_contract_sha256",
    }:
        print("launch request keys mismatch", file=sys.stderr, flush=True)
        return 93
    argv = request.get("argv")
    if (
        not isinstance(argv, list)
        or len(argv) < 3
        or any(not isinstance(value, str) or not value for value in argv)
    ):
        print("launch argv is invalid", file=sys.stderr, flush=True)
        return 94
    expected_argv_sha = hashlib.sha256(canonical_json(argv)).hexdigest()
    if request.get("argv_sha256") != expected_argv_sha:
        print("launch argv digest mismatch", file=sys.stderr, flush=True)
        return 95
    observed_environment = dict(os.environ)
    observed_environment_sha = hashlib.sha256(
        canonical_json(observed_environment)
    ).hexdigest()
    if request.get("environment_sha256") != observed_environment_sha:
        print("launch environment digest mismatch", file=sys.stderr, flush=True)
        return 97
    contract_sha = request.get("execution_contract_sha256")
    if (
        not isinstance(contract_sha, str)
        or len(contract_sha) != 64
        or any(character not in "0123456789abcdef" for character in contract_sha)
    ):
        print("launch execution-contract digest is invalid", file=sys.stderr, flush=True)
        return 99
    working_directory = request.get("working_directory")
    if (
        not isinstance(working_directory, str)
        or Path(working_directory).resolve() != Path.cwd().resolve()
    ):
        print("launch working-directory mismatch", file=sys.stderr, flush=True)
        return 98
    token = sys.stdin.buffer.readline(256)
    expected = f"GO {args.request_sha256}\n".encode("ascii")
    if token != expected:
        print("launch authorization missing; target not executed", file=sys.stderr, flush=True)
        return 96
    # Keep the authenticated bootstrap as the process-tree root.  This is
    # essential on Windows, where CRT exec may briefly detach a replacement
    # process from the parent handle.  The supervisor monitors/kills the whole
    # bootstrap tree and sees the target's exact exit code.
    completed = subprocess.run(
        argv,
        stdin=subprocess.DEVNULL,
        check=False,
        cwd=working_directory,
        env=dict(os.environ),
        close_fds=True,
    )
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())

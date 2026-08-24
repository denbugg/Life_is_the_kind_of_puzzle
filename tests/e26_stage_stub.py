"""Tiny subprocess fixture for autonomous E26 runner tests."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def increment(path: Path) -> int:
    value = int(path.read_text("ascii")) if path.exists() else 0
    value += 1
    atomic_write(path, str(value).encode("ascii"))
    return value


def write_command(args: argparse.Namespace) -> int:
    count = increment(args.counter) if args.counter else 1
    print("step=1/2", flush=True)
    if args.sleep:
        time.sleep(args.sleep)
    if args.fail_first and count == 1:
        args.fail_first.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(args.fail_first, b"failed-once")
        print("intentional first-attempt failure", file=sys.stderr, flush=True)
        return 17
    payload = {
        "schema": "e26-stage-stub-v1",
        "payload": args.payload,
        "decision": args.decision,
        "attempt_count": count,
        "environment": {
            name: os.environ.get(name)
            for name in ("TEMP", "TMP", "PYTHONPYCACHEPREFIX", "TORCH_HOME")
        },
        "environment_sha256": hashlib.sha256(canonical(dict(os.environ))).hexdigest(),
        "ambient_sentinel": os.environ.get("E26_AMBIENT_SENTINEL"),
        "working_directory": str(Path.cwd().resolve()),
    }
    atomic_write(args.output, canonical(payload))
    print("step=2/2", flush=True)
    return 0


def verify_command(args: argparse.Namespace) -> int:
    raw = args.path.read_bytes()
    value = json.loads(raw)
    if raw != canonical(value):
        raise SystemExit("not canonical")
    if value.get("payload") != args.expected:
        raise SystemExit("payload mismatch")
    print("verified", flush=True)
    return 0


def mutate_command(args: argparse.Namespace) -> int:
    atomic_write(args.path, args.payload.encode("utf-8"))
    print("mutated", flush=True)
    return 0


def delayed_marker_command(args: argparse.Namespace) -> int:
    time.sleep(args.delay)
    atomic_write(args.marker, b"descendant-survived")
    return 0


def spawn_child_command(args: argparse.Namespace) -> int:
    subprocess.Popen(
        [
            sys.executable, "-B", str(Path(__file__).resolve()),
            "delayed-marker", "--marker", str(args.marker),
            "--delay", str(args.delay),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
    )
    atomic_write(args.ready, b"ready")
    time.sleep(60.0)
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command", required=True)
    write = sub.add_parser("write")
    write.add_argument("--output", type=Path, required=True)
    write.add_argument("--payload", required=True)
    write.add_argument("--decision", default="PASS")
    write.add_argument("--counter", type=Path)
    write.add_argument("--fail-first", type=Path)
    write.add_argument("--sleep", type=float, default=0.0)
    verify = sub.add_parser("verify")
    verify.add_argument("--path", type=Path, required=True)
    verify.add_argument("--expected", required=True)
    mutate = sub.add_parser("mutate")
    mutate.add_argument("--path", type=Path, required=True)
    mutate.add_argument("--payload", default="mutated")
    delayed = sub.add_parser("delayed-marker")
    delayed.add_argument("--marker", type=Path, required=True)
    delayed.add_argument("--delay", type=float, default=1.0)
    spawn = sub.add_parser("spawn-child")
    spawn.add_argument("--ready", type=Path, required=True)
    spawn.add_argument("--marker", type=Path, required=True)
    spawn.add_argument("--delay", type=float, default=1.0)
    return root


def main() -> int:
    args = parser().parse_args()
    if args.command == "write":
        return write_command(args)
    if args.command == "verify":
        return verify_command(args)
    if args.command == "mutate":
        return mutate_command(args)
    if args.command == "delayed-marker":
        return delayed_marker_command(args)
    return spawn_child_command(args)


if __name__ == "__main__":
    raise SystemExit(main())

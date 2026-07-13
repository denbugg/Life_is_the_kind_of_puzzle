#!/usr/bin/env python3
"""Run the target-free group-switch synchronization synthetic gate."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback


INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
WRAPPER = WORKING / "group_switch_synth_wrapper.json"
EXPECTED_CODE_TREE_SHA256 = "95e92e39aa0f0028fc3df13231d8699c2201228f9ea5f53c0ac7bbffe0f9342e"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(root: Path) -> str:
    entries = [
        [path.relative_to(root).as_posix(), sha256(path)]
        for path in sorted(value for value in root.rglob("*") if value.is_file())
    ]
    payload = json.dumps(entries, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def unique(paths: list[Path], label: str) -> Path:
    values = sorted(set(path.resolve() for path in paths))
    if len(values) != 1:
        raise RuntimeError(f"expected one {label}, got {values}")
    return values[0]


def write(payload: dict) -> None:
    temporary = WRAPPER.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(WRAPPER)


def main() -> None:
    started = time.time()
    wrapper = {
        "schema_version": 1,
        "kind": "group_switch_synthetic_kaggle_wrapper",
        "status": "starting",
        "safe_for_submission": False,
        "targets_opened": False,
        "started_unix": started,
    }
    write(wrapper)
    try:
        test_file = unique(
            list(INPUT.rglob("tests/test_group_switch_sync.py")),
            "group-switch tests",
        )
        code_root = test_file.parents[1]
        code_tree_hash = tree_sha256(code_root)
        if code_tree_hash != EXPECTED_CODE_TREE_SHA256:
            raise RuntimeError(f"code tree hash mismatch: {code_tree_hash}")
        environment = dict(os.environ)
        environment.update(
            {
                "PYTHONPATH": str(code_root / "src"),
                "PYTHONHASHSEED": "20260713",
                "PYTHONUNBUFFERED": "1",
                "OPENBLAS_NUM_THREADS": "4",
                "OMP_NUM_THREADS": "4",
                "MKL_NUM_THREADS": "4",
            }
        )
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_group_switch_sync.py",
        ]
        completed = subprocess.run(
            command,
            cwd=code_root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        wrapper.update(
            {
                "status": "complete" if completed.returncode == 0 else "failed",
                "scientific_status": (
                    "go_exposed_panel_screen_only"
                    if completed.returncode == 0
                    else "stop_synthetic_gate_failed"
                ),
                "code_root": str(code_root),
                "code_tree_sha256": code_tree_hash,
                "command": command,
                "tests": {
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                },
                "environment": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                },
                "safe_for_submission": False,
                "targets_opened": False,
                "seconds": time.time() - started,
            }
        )
        write(wrapper)
        if completed.returncode:
            raise RuntimeError("group-switch synthetic gate failed")
        print(json.dumps(wrapper, sort_keys=True), flush=True)
    except Exception as error:
        wrapper.update(
            {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "safe_for_submission": False,
                "targets_opened": False,
                "seconds": time.time() - started,
            }
        )
        write(wrapper)
        raise


if __name__ == "__main__":
    main()

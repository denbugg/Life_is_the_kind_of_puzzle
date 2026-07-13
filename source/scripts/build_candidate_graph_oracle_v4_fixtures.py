#!/usr/bin/env python3
"""Versioned v4 fixture-builder entry point bound to the frozen source snapshot.

The reusable historical implementation is loaded only after this process has
proved that no project module was imported from another tree and has placed the
v4 snapshot first on ``sys.path``.  The implementation source itself is also
hash-bound here, so the pinned wrapper transitively closes every executed
fixture-builder byte without modifying the historical file.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
from types import ModuleType
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/candidate_graph_oracle_ceiling_v4.json"
SNAPSHOT_RELATIVE = Path(
    "runs/assembly_v1/kaggle/candidate_graph_oracle_v4_source_snapshot/src"
)
SNAPSHOT_ROOT = REPO_ROOT / SNAPSHOT_RELATIVE
HISTORICAL_BUILDER = REPO_ROOT / "scripts/build_candidate_graph_oracle_fixtures.py"
HISTORICAL_BUILDER_SHA256 = (
    "68845ed93db08928f16d5b4d3c16f8905cedc6e48f673528a14db7f8f692ecb8"
)
EXPECTED_INSTANCE = "6c0fe4e8524ce39d830d9a5bee118d8b"
PROJECT_PREFIXES = ("puzzle_assembly", "puzzle_denoise_v2")


def _read_regular(path: Path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    chunks: list[bytes] = []
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RuntimeError(f"not a one-link regular file: {path}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def _sha256(path: Path) -> str:
    return hashlib.sha256(_read_regular(path)).hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(_read_regular(path).decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("v4 protocol config must be a JSON object")
    if payload.get("protocol_instance_id") != EXPECTED_INSTANCE:
        raise RuntimeError("v4 protocol instance drift")
    return payload


def _known_code(config: Mapping[str, Any]) -> dict[str, str]:
    value = (
        config.get("frozen_contract", {})
        .get("assets", {})
        .get("known_code_sha256")
    )
    if not isinstance(value, dict) or not value:
        raise RuntimeError("v4 frozen known-code closure is missing")
    result: dict[str, str] = {}
    for relative, digest in value.items():
        if (
            not isinstance(relative, str)
            or not relative.startswith(SNAPSHOT_RELATIVE.as_posix() + "/")
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            raise RuntimeError("v4 frozen known-code closure is malformed")
        path = REPO_ROOT / relative
        if _sha256(path) != digest:
            raise RuntimeError(f"v4 source snapshot hash drift: {relative}")
        result[relative] = digest
    return result


def _is_project_module(name: str) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in PROJECT_PREFIXES)


def _assert_clean_import_state() -> None:
    loaded = sorted(name for name in sys.modules if _is_project_module(name))
    if loaded:
        raise RuntimeError(
            "v4 fixture builder requires an isolated process before snapshot imports: "
            + ",".join(loaded)
        )


def _module_source(module: ModuleType) -> Path:
    spec = getattr(module, "__spec__", None)
    origin = getattr(spec, "origin", None)
    if not isinstance(origin, str) or origin in {"built-in", "frozen"}:
        raise RuntimeError(f"project module has no regular source origin: {module.__name__}")
    return Path(origin).resolve(strict=True)


def _assert_loaded_origins(known: Mapping[str, str]) -> None:
    loaded = {
        name: module
        for name, module in sys.modules.items()
        if _is_project_module(name) and isinstance(module, ModuleType)
    }
    if not loaded:
        raise RuntimeError("historical fixture builder imported no project modules")
    for name, module in sorted(loaded.items()):
        source = _module_source(module)
        try:
            relative = source.relative_to(REPO_ROOT).as_posix()
        except ValueError as error:
            raise RuntimeError(f"project module escaped v4 repository: {name}") from error
        expected = known.get(relative)
        if expected is None or _sha256(source) != expected:
            raise RuntimeError(f"project module did not originate in frozen v4 snapshot: {name}")


def _load_historical_builder(config: Mapping[str, Any]) -> ModuleType:
    if _sha256(HISTORICAL_BUILDER) != HISTORICAL_BUILDER_SHA256:
        raise RuntimeError("historical generic fixture-builder source drift")
    known = _known_code(config)
    _assert_clean_import_state()
    sys.path.insert(0, str(SNAPSHOT_ROOT))
    spec = importlib.util.spec_from_file_location(
        "_candidate_graph_oracle_v4_fixture_builder_base_68845e",
        HISTORICAL_BUILDER,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load historical fixture-builder implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _assert_loaded_origins(known)
    return module


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "puzzle")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--lock-path", type=Path, required=True)
    parser.add_argument("--prep-marker-path", type=Path)
    parser.add_argument("--lifecycle-ledger-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config_path = args.config.expanduser().resolve(strict=True)
    if config_path != CONFIG_PATH.resolve(strict=True):
        raise RuntimeError("v4 fixture builder accepts only the v4 protocol config")
    config = _load_config(config_path)
    builder = _load_historical_builder(config)
    marker = args.prep_marker_path
    if marker is None:
        marker = args.lock_path.with_name("FIXTURE_PIXEL_ACCESS_STARTED.json")
    summary = builder.prepare_fixtures(
        config_path=config_path,
        data_root=args.data_root,
        input_root=args.input_root,
        label_root=args.label_root,
        lock_path=args.lock_path,
        marker_path=marker,
        lifecycle_ledger_root=args.lifecycle_ledger_root,
        repo_root=REPO_ROOT,
        executing_builder_path=Path(__file__).resolve(strict=True),
    )
    print(json.dumps(summary, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()

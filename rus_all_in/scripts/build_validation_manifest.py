"""Build the frozen train/calibration/holdout validation manifest."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from aiijc_puzzle.protocol import (
    SplitCounts,
    build_validation_manifest,
    write_validation_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "validation.yaml"


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a YAML mapping")
    return value


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _project_path(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path string")
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_config(path: Path) -> tuple[Path, Path, Path, int, int, SplitCounts]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = _mapping(raw, "config")
    splits = _mapping(config.get("splits"), "splits")
    counts = SplitCounts(
        train=_integer(splits.get("train"), "splits.train"),
        calibration=_integer(splits.get("calibration"), "splits.calibration"),
        holdout=_integer(splits.get("holdout"), "splits.holdout"),
    )
    return (
        _project_path(config.get("inputs_dir"), "inputs_dir"),
        _project_path(config.get("targets_dir"), "targets_dir"),
        _project_path(config.get("manifest_path"), "manifest_path"),
        _integer(config.get("seed"), "seed"),
        _integer(config.get("expected_pairs"), "expected_pairs"),
        counts,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="YAML protocol config (default: configs/validation.yaml)",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="validate, hash, and report the manifest without writing it",
    )
    mode.add_argument("--run", action="store_true", help="build and atomically write the manifest")
    return parser


def main() -> None:
    args = _parser().parse_args()
    inputs_dir, targets_dir, output_path, seed, expected_pairs, counts = _load_config(
        args.config.resolve()
    )
    manifest = build_validation_manifest(
        inputs_dir,
        targets_dir,
        seed=seed,
        counts=counts,
        expected_pairs=expected_pairs,
    )
    if args.run:
        write_validation_manifest(manifest, output_path)

    summary = {
        "mode": "run" if args.run else "dry-run",
        "pairs": expected_pairs,
        "counts": counts.as_dict(),
        "protocol_digest": manifest["protocol_digest"],
        "manifest_path": str(output_path),
        "written": bool(args.run),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

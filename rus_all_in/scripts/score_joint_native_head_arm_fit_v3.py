#!/usr/bin/env python3
"""One-field metadata overlay for the immutable joint-native FIT64 scorer.

The signed v2 scoring binding contains one transcribed SHA-256 typo in its
target-free source roster.  This wrapper proves that the correction comes from
the already frozen validation manifest and that applying it changes exactly one
JSON leaf.  It then delegates validation/scoring to the unchanged v2 scorer.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from aiijc_puzzle.protocol import sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import score_joint_native_head_arm_fit_v2 as v2  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "configs/joint_native_head_arm_fit_score_v3.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/joint-native-head-arm-fit/fixed-v1"
DEFAULT_MANIFEST = PROJECT_ROOT / "data/interim/validation_manifest.json"
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
CONFIG_SCHEMA = "aiijc-joint-native-head-arm-fit-score-binding-v3"
EXPECTED_CORRECTION_PATH = (
    "repair_only",
    "source_roster",
    17,
    "target_sha256",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("validate", "score"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    return parser.parse_args(argv)


def _project_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {"path": _project_path(resolved), "sha256": sha256_file(resolved)}


def _json_leaf_differences(
    left: Any,
    right: Any,
    path: tuple[str | int, ...] = (),
) -> list[tuple[tuple[str | int, ...], Any, Any]]:
    """Return exact changed JSON leaves, including missing keys/list items."""

    if isinstance(left, Mapping) and isinstance(right, Mapping):
        differences: list[tuple[tuple[str | int, ...], Any, Any]] = []
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                differences.append((path + (str(key),), left.get(key), right.get(key)))
            else:
                differences.extend(
                    _json_leaf_differences(left[key], right[key], path + (str(key),))
                )
        return differences
    if isinstance(left, list) and isinstance(right, list):
        differences = []
        common = min(len(left), len(right))
        for index in range(common):
            differences.extend(
                _json_leaf_differences(left[index], right[index], path + (index,))
            )
        for index in range(common, max(len(left), len(right))):
            differences.append(
                (
                    path + (index,),
                    left[index] if index < len(left) else None,
                    right[index] if index < len(right) else None,
                )
            )
        return differences
    return [] if left == right else [(path, left, right)]


def _load_signed_overlay(path: Path) -> tuple[dict[str, Any], str]:
    resolved = path.resolve()
    sidecar = resolved.with_suffix(resolved.suffix + ".sha256")
    if not resolved.is_file() or not sidecar.is_file():
        raise FileNotFoundError("signed v3 one-field overlay is unavailable")
    digest = sha256_file(resolved)
    if sidecar.read_text(encoding="utf-8").split()[0] != digest:
        raise RuntimeError("v3 overlay sidecar mismatch")
    config = json.loads(resolved.read_text(encoding="utf-8"))
    if config.get("schema") != CONFIG_SCHEMA or config.get("status") != "signed-one-field-overlay":
        raise RuntimeError("v3 overlay is not signed/fixed")
    if config.get("config_path") != _project_path(resolved):
        raise RuntimeError("runtime v3 config path differs from signed path")
    if config.get("allowed_change_count") != 1:
        raise RuntimeError("v3 overlay allows more than one metadata change")
    for artifact in config.get("bound_inputs", {}).values():
        target = _path(str(artifact["path"]))
        if not target.is_file() or _record(target) != artifact:
            raise RuntimeError(f"v3 bound input changed: {target}")
    return config, digest


def _apply_one_field_overlay(
    base: Mapping[str, Any],
    overlay: Mapping[str, Any],
    manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Patch one roster SHA and prove the exact source and JSON diff."""

    correction = overlay.get("correction", {})
    expected_keys = {
        "filename",
        "field",
        "roster_index",
        "from_sha256",
        "to_sha256",
        "json_path",
        "authoritative_source",
        "derived_from_target_pixels",
    }
    if set(correction) != expected_keys:
        raise RuntimeError("v3 correction schema changed")
    if (
        correction["filename"] != "img_001111.png"
        or correction["field"] != "target_sha256"
        or correction["roster_index"] != 17
        or tuple(correction["json_path"]) != EXPECTED_CORRECTION_PATH
        or correction["authoritative_source"] != "frozen-validation-manifest-metadata"
        or correction["derived_from_target_pixels"] is not False
    ):
        raise RuntimeError("v3 correction is not the preregistered one-field repair")

    manifest_record = _record(manifest_path)
    if manifest_record != overlay["bound_inputs"]["manifest"]:
        raise RuntimeError("runtime manifest differs from signed v3 authority")
    if manifest_record != base["frozen_inputs"]["manifest"]:
        raise RuntimeError("v3 authority differs from the signed v2 manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    matches = [
        row
        for row in manifest["splits"]["train"]
        if row.get("filename") == correction["filename"]
    ]
    if len(matches) != 1 or matches[0].get("target_sha256") != correction["to_sha256"]:
        raise RuntimeError("v3 corrected SHA is not the immutable manifest declaration")

    repaired = copy.deepcopy(dict(base))
    roster = repaired["repair_only"]["source_roster"]
    index = int(correction["roster_index"])
    if (
        roster[index].get("filename") != correction["filename"]
        or roster[index].get("target_sha256") != correction["from_sha256"]
    ):
        raise RuntimeError("signed v2 typo no longer matches the v3 overlay precondition")
    roster[index]["target_sha256"] = correction["to_sha256"]
    differences = _json_leaf_differences(base, repaired)
    expected_difference = [
        (
            EXPECTED_CORRECTION_PATH,
            correction["from_sha256"],
            correction["to_sha256"],
        )
    ]
    if differences != expected_difference:
        raise RuntimeError(f"v3 overlay changed more than one JSON leaf: {differences!r}")
    proof = {
        "changed_leaf_count": len(differences),
        "changed_json_path": list(differences[0][0]),
        "filename": correction["filename"],
        "from_sha256": correction["from_sha256"],
        "to_sha256": correction["to_sha256"],
        "authority": manifest_record,
        "derived_from_target_pixels": False,
    }
    return repaired, proof


def _prepare(
    overlay_path: Path,
    manifest_path: Path,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    overlay, overlay_sha = _load_signed_overlay(overlay_path)
    base_path = _path(overlay["bound_inputs"]["v2_config"]["path"])
    base, base_sha = v2._load_signed_config(base_path)
    if base_sha != overlay["bound_inputs"]["v2_config"]["sha256"]:
        raise RuntimeError("loaded v2 binding differs from v3 commitment")
    repaired, proof = _apply_one_field_overlay(base, overlay, manifest_path)
    return repaired, overlay_sha, proof


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config, overlay_sha, proof = _prepare(args.config, args.manifest)
    if args.mode == "validate":
        validation = v2.run_validate(config, args.output_dir, args.manifest)
        report = {
            "schema": "aiijc-joint-native-head-arm-fit-score-v3-validation-v1",
            "status": validation["status"],
            "v3_config_sha256": overlay_sha,
            "one_field_overlay_proof": proof,
            "unchanged_v2_validation": validation,
        }
    else:
        report = v2.run_score(
            config,
            overlay_sha,
            args.output_dir,
            args.manifest,
            args.targets,
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if args.mode == "score" and not report["gate"]["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

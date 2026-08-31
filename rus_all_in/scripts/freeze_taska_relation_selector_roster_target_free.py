#!/usr/bin/env python3
"""Freeze the existing TASKA six-arm relation-selector roster for joint DEV.

This is a target-free sibling of the joint DEV freeze.  It reconstructs the
same dirty shuffled tile bags, runs only the already SHA-gated production
relation-selector pipeline, and stores the six whole layouts, relation features,
frozen HGB scores and incumbent choice.  It never reconstructs an exact layout,
loads labels, scores a metric, or creates a new solver arm.

The checked-in config is unsigned and blocked.  Do not run this script until a
reviewer creates a separate signed protocol before any DEV pixel is opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from aiijc_puzzle.joint_relation_selector_consumer import (
    RELATION_ROSTER_CASE_KEYS,
    FrozenSixArmRoster,
    reject_target_bearing_array_names,
)
from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.synthetic_socket_evaluation import names_digest
from aiijc_puzzle.taska_relation_selector_pipeline import (
    load_taska_relation_selector_resources,
    verify_taska_relation_selector_solver,
)
from aiijc_puzzle.taska_relation_truth_selector import FEATURE_NAMES
from aiijc_puzzle.taska_selective_fullres_fusion import FUSION_ARM_NAMES

try:
    from scripts import run_fullres_boundary_denoiser as boundary
    from scripts import run_joint_reciprocal_tri_emitter_real as joint
    from scripts import run_taska_relation_truth_selector_confirmation as relation
except ModuleNotFoundError:
    import run_fullres_boundary_denoiser as boundary
    import run_joint_reciprocal_tri_emitter_real as joint
    import run_taska_relation_truth_selector_confirmation as relation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/taska_relation_selector_roster_v2_dev32_unsigned_template_v1.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs/taska-relation-selector-roster/v2-dev32-draw0-v1"
)
DEFAULT_MANIFEST = joint.prior.roster.DEFAULT_MANIFEST
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"

CONFIG_SCHEMA = "aiijc-taska-relation-selector-roster-protocol-v1"
SIGNED_STATUS = "signed-fixed-protocol"
BLOCKED_STATUS = "unsigned-template-blocked-before-dev-pixel-access"
METADATA_SCHEMA = "aiijc-taska-relation-selector-target-free-roster-v1"
FREEZE_SCHEMA = "aiijc-taska-relation-selector-target-free-roster-freeze-v1"
RESULT_SCHEMA = "aiijc-taska-relation-selector-target-free-roster-result-v1"
ARCHIVE_NAME = "frozen-target-free-roster.npz"
METADATA_NAME = "frozen-target-free-roster.json"
FREEZE_NAME = "pre-score-freeze.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    return parser.parse_args(argv)


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        label = str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        label = str(resolved)
    return {"path": label, "sha256": sha256_file(resolved)}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def rule_commitment_sha256(config: Mapping[str, Any]) -> str:
    """Hash every inference decision fixed before DEV pixels are opened."""

    payload = {
        "schema": config.get("schema"),
        "joint_protocol_sha256": config.get("joint_protocol_sha256"),
        "source_protocol": config.get("source_protocol"),
        "runtime": config.get("runtime"),
        "output_contract": config.get("output_contract"),
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _require_exact_contract(config: Mapping[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA:
        raise RuntimeError("relation-roster protocol schema changed")
    source = config.get("source_protocol", {})
    names = source.get("dev_filenames")
    if not isinstance(names, list) or len(names) != 32 or len(set(names)) != 32:
        raise RuntimeError("relation-roster DEV source32 is not explicit and unique")
    if names_digest(names) != source.get("dev_digest"):
        raise RuntimeError("relation-roster DEV digest changed")
    if int(source.get("dev_draw_index", -1)) != 0:
        raise RuntimeError("relation-roster DEV draw changed")
    if int(source.get("dev_case_seed", -1)) != 20260908:
        raise RuntimeError("relation-roster corruption seed changed")
    if config.get("runtime") != {
        "device": "mps",
        "inference_batch": 576,
        "model_or_threshold_tuning": False,
        "existing_sha_gated_pipeline_only": True,
    }:
        raise RuntimeError("relation-roster fixed runtime changed")
    if config.get("output_contract") != {
        "arm_names": list(FUSION_ARM_NAMES),
        "relation_feature_names": list(FEATURE_NAMES),
        "relation_rows_per_arm": 1104,
        "strict_original_upright_permutations": True,
        "contains_pixels": False,
        "contains_exact_references_or_labels": False,
        "normalized_case_key_allowlist": sorted(RELATION_ROSTER_CASE_KEYS),
    }:
        raise RuntimeError("relation-roster normalized output contract changed")
    if config.get("rule_commitment_sha256") != rule_commitment_sha256(config):
        raise RuntimeError("relation-roster pre-pixel rule commitment changed")


def _verified_record(config: Mapping[str, Any], name: str) -> Path:
    artifact = config.get("frozen_inputs", {}).get(name)
    if not isinstance(artifact, Mapping):
        raise RuntimeError(f"relation-roster frozen input missing: {name}")
    path = _project_path(str(artifact.get("path", "")))
    if not path.is_file() or sha256_file(path) != artifact.get("sha256"):
        raise RuntimeError(f"relation-roster frozen input changed: {name}")
    return path


def _load_signed_config(path: Path) -> tuple[dict[str, Any], str]:
    resolved = path.resolve()
    config = json.loads(resolved.read_text(encoding="utf-8"))
    _require_exact_contract(config)
    if config.get("status") == BLOCKED_STATUS:
        raise RuntimeError("relation-roster template is intentionally blocked")
    sidecar = resolved.with_suffix(resolved.suffix + ".sha256")
    digest = sha256_file(resolved)
    if (
        config.get("status") != SIGNED_STATUS
        or not sidecar.is_file()
        or sidecar.read_text(encoding="utf-8").split()[0] != digest
    ):
        raise RuntimeError("relation-roster protocol is not signed/fixed")
    for name in config.get("frozen_inputs", {}):
        _verified_record(config, name)
    joint_config_path = _verified_record(config, "joint_protocol")
    if sha256_file(joint_config_path) != config.get("joint_protocol_sha256"):
        raise RuntimeError("relation-roster joint protocol digest changed")
    joint_config = json.loads(joint_config_path.read_text(encoding="utf-8"))
    for key in ("dev_filenames", "dev_digest", "dev_draw_index", "dev_case_seed"):
        if config["source_protocol"][key] != joint_config.get("source_protocol", {}).get(key):
            raise RuntimeError(f"relation-roster differs from joint protocol: {key}")
    return config, digest


def normalize_relation_case(
    arrays: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    *,
    grid_size: int = 24,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Keep the exact label-free allowlist and revalidate the whole-arm roster."""

    reject_target_bearing_array_names(tuple(arrays))
    missing = RELATION_ROSTER_CASE_KEYS - set(arrays)
    if missing:
        raise RuntimeError(f"relation-selector output omits normalized keys: {sorted(missing)}")
    normalized = {
        key: np.ascontiguousarray(arrays[key]) for key in RELATION_ROSTER_CASE_KEYS
    }
    choice = diagnostics.get("choice")
    row = {"choice": choice, "arm_names": list(FUSION_ARM_NAMES)}
    roster = FrozenSixArmRoster.from_case_arrays(
        normalized, row, grid_size=grid_size
    )
    if roster.incumbent_index != FUSION_ARM_NAMES.index(str(choice)):
        raise RuntimeError("normalized incumbent arm changed")
    return normalized, {
        "choice": choice,
        "control_choice": diagnostics.get("control_choice"),
        "changed_from_control": bool(diagnostics.get("changed_from_control")),
        "arm_names": list(FUSION_ARM_NAMES),
    }


def freeze_target_free_roster(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    config_sha256: str,
) -> dict[str, Any]:
    """Run the unchanged pipeline and freeze no more than the normalized roster."""

    verified_solver = verify_taska_relation_selector_solver()
    device = torch.device(str(config["runtime"]["device"]))
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("signed relation-roster protocol requires MPS")
    resources = load_taska_relation_selector_resources(device=device)
    if resources.confirmed_sha256 != verified_solver:
        raise RuntimeError("loaded relation-selector resources differ from SHA gate")
    names = tuple(config["source_protocol"]["dev_filenames"])
    manifest = _verified_record(config, "validation_manifest")
    if args.manifest.resolve() != manifest:
        raise RuntimeError("relation-roster manifest path differs from signed protocol")
    records = joint._manifest_records(args.manifest, names)
    boards = boundary._prepare_boards(records, args.targets)
    if tuple(board.filename for board in boards) != names:
        raise RuntimeError("prepared relation-roster boards differ from signed order")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    arrays: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    draw_index = int(config["source_protocol"]["dev_draw_index"])
    seed = int(config["source_protocol"]["dev_case_seed"])
    inference_batch = int(config["runtime"]["inference_batch"])
    for index, board in enumerate(boards):
        item = joint.make_target_free_synthetic_case(
            board.tiles,
            source_filename=board.filename,
            draw_index=draw_index,
            seed=seed,
        )
        parent_arrays, parent_diagnostics = relation.parent._target_free_case(
            item.tiles,
            resources=resources.parent.pair,
            denoiser=resources.parent.denoiser,
            inference_batch=inference_batch,
        )
        relation_arrays, relation_diagnostics = relation._relation_case(
            parent_arrays, parent_diagnostics, resources.relation_model
        )
        normalized, normalized_row = normalize_relation_case(
            relation_arrays, relation_diagnostics
        )
        prefix = f"case_{index:04d}"
        arrays.update(
            {f"{prefix}__{key}": value for key, value in normalized.items()}
        )
        rows.append(
            {
                "prefix": prefix,
                "case_id": item.case_id,
                "source_filename": item.source_filename,
                "draw_index": item.draw_index,
                "dirty_sha256": hashlib.sha256(item.tiles.tobytes()).hexdigest(),
                **normalized_row,
            }
        )
        print(
            json.dumps(
                {
                    "event": "relation_selector_target_free_roster_case",
                    "case": index + 1,
                    "case_count": len(boards),
                    "source_filename": board.filename,
                    "choice": normalized_row["choice"],
                }
            ),
            flush=True,
        )

    archive_path = output / ARCHIVE_NAME
    metadata_path = output / METADATA_NAME
    freeze_path = output / FREEZE_NAME
    _write_npz(archive_path, arrays)
    _write_json(
        metadata_path,
        {
            "schema": METADATA_SCHEMA,
            "config_sha256": config_sha256,
            "contains_exact_references_or_labels": False,
            "contains_pixels": False,
            "all_layouts_strict_original_upright_tile_permutations": True,
            "arm_names": list(FUSION_ARM_NAMES),
            "relation_feature_names": list(FEATURE_NAMES),
            "relation_rows_per_arm": 1104,
            "normalized_case_key_allowlist": sorted(RELATION_ROSTER_CASE_KEYS),
            "rows": rows,
        },
    )
    _write_json(
        freeze_path,
        {
            "schema": FREEZE_SCHEMA,
            "created_before_exact_reference_reconstruction": True,
            "contains_evaluation_references_or_labels": False,
            "config_sha256": config_sha256,
            "source_case_count": len(rows),
            "verified_relation_selector_sha256": dict(verified_solver),
            "artifacts": {
                "archive": _record(archive_path),
                "metadata": _record(metadata_path),
                "preregistration": _record(args.config),
                "joint_protocol": _record(_verified_record(config, "joint_protocol")),
                "production_pipeline": _record(
                    PROJECT_ROOT / "src/aiijc_puzzle/taska_relation_selector_pipeline.py"
                ),
                "freezer": _record(Path(__file__)),
            },
        },
    )
    return {
        "schema": RESULT_SCHEMA,
        "status": "target-free-six-arm-roster-frozen-no-reference-scoring",
        "case_count": len(rows),
        "archive": _record(archive_path),
        "metadata": _record(metadata_path),
        "pre_score_freeze": _record(freeze_path),
        "dev_pixels_opened_only_after_signed_protocol": True,
        "dev_labels_or_exact_references_opened": False,
        "local_terminal_or_competition_test_accessed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config, config_sha = _load_signed_config(args.config)
    result = freeze_target_free_roster(args, config, config_sha)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

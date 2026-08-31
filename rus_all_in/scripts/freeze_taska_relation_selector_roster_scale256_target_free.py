#!/usr/bin/env python3
"""Freeze the unchanged TASKA six-arm roster on the scale256 DEV64 roster.

This wrapper owns only the DEV64 protocol bridge.  It derives the explicit
source roster from the signed scale-cache protocol, checks its digest/draw/seed
against the signed scale256 protocol, and delegates inference and exclusive
output writing to the unchanged target-free roster freezer.

The checked-in config is blocked.  Do not run this script until a reviewer has
created and signed a separate fixed protocol before any DEV64 pixel is opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.synthetic_socket_evaluation import names_digest
from aiijc_puzzle.taska_relation_truth_selector import FEATURE_NAMES
from aiijc_puzzle.taska_selective_fullres_fusion import FUSION_ARM_NAMES

try:
    from scripts import freeze_taska_relation_selector_roster_target_free as base
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    import freeze_taska_relation_selector_roster_target_free as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/taska_relation_selector_roster_scale256_dev64_unsigned_template_v1.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs/taska-relation-selector-roster/scale256-dev64-draw0-v1"
)

CONFIG_SCHEMA = "aiijc-taska-relation-selector-scale256-dev64-roster-protocol-v1"
SIGNED_STATUS = "signed-fixed-protocol"
BLOCKED_STATUS = "unsigned-template-blocked-before-dev64-pixel-access"
SCALE_SCHEMA = "aiijc-joint-reciprocal-scale256-real-protocol-v1"
SCALE_CACHE_SCHEMA = "aiijc-joint-reciprocal-scale-fit-cache-materialization-v1"
DEV_SOURCE_COUNT = 64
DEV_DIGEST = "5c6cb5b9b204a38c78e79936ff34235dae9896cfc13d6edaf12dfad635bcdb8e"
DEV_DRAW_INDEX = 0
DEV_CASE_SEED = 20260908
SELECTION_NAMESPACE = "aiijc-joint-reciprocal-scale256-fit256-dev64-v1"
SELECTION_SEED = 20260913

EXISTING_FROZEN_INPUTS: dict[str, tuple[str, str]] = {
    "joint_protocol": (
        "configs/joint_reciprocal_scale256_real_preregistered_v1.json",
        "e3fe4aa5c594b149c4dab93960aabf220754d82bf938edef852658356fc9bf3b",
    ),
    "scale_cache_protocol": (
        "configs/joint_reciprocal_scale256_fit_cache_preregistered_v1.json",
        "3e397e7ff3a565de2b1ab412f71f8e5b7d500649b40a51be2ed97805ecc7344e",
    ),
    "joint_runner": (
        "scripts/run_joint_reciprocal_tri_emitter_real.py",
        "7b6f760155f86c9c3c465a8833c9c71687123b0ac248a64d55b91d0f8c1ad82c",
    ),
    "validation_manifest": (
        "data/interim/validation_manifest.json",
        "4781e370e092ad272c63e6d5165b25951aaf93fae5fde74c75d534a9e8efc9da",
    ),
    "board_loader": (
        "scripts/run_fullres_boundary_denoiser.py",
        "3afa0340ecc2f857e501fc86e866feb32bd4a8789cd445fc66442dd00c5ac274",
    ),
    "production_pipeline": (
        "src/aiijc_puzzle/taska_relation_selector_pipeline.py",
        "1020ebc28777ba02872a82613bbb433d802e9e2b3e6fc04a5cbd2b81e49e7976",
    ),
    "relation_model": (
        "outputs/taska-relation-truth-selector/fixed-v1/model-local32-held32/"
        "frozen-relation-classifier.pkl",
        "ec4eca99243cdc6be20104d789b9e5d5598b79fa0d1b7e69bc37314375ad8c6b",
    ),
    "relation_confirmation_config": (
        "configs/taska_relation_truth_selector_confirmation_v1.json",
        "3d903eb595d1c0d152a8b53c7c9fa578b5b012227eeb03ab629a7dd24d5ce4e9",
    ),
    "relation_confirmation_report": (
        "outputs/taska-relation-truth-selector/formal-confirmation-v1/report.json",
        "d260872251077e1515251b6c7afc316af25df75045c8119112dff4f36c68ea23",
    ),
    "relation_confirmation_runner": (
        "scripts/run_taska_relation_truth_selector_confirmation.py",
        "d04299b0e69a12abb0e063a919407deb692254860ab53c084255ad1a97e5330e",
    ),
    "fusion_confirmation_runner": (
        "scripts/run_taska_selective_fullres_union_fusion_fresh32_confirmation.py",
        "f38356446da7283d42fb8c14ce2c024a74c5f57a1c9133ca3f107f54bdd5a654",
    ),
    "base_freezer": (
        "scripts/freeze_taska_relation_selector_roster_target_free.py",
        "3b98381675d9becb6fe494126006a93083caae8621e057e38afe6837b67e1f27",
    ),
}
WRAPPER_PATH = "scripts/freeze_taska_relation_selector_roster_scale256_target_free.py"
REQUIRED_FROZEN_INPUTS = frozenset(EXISTING_FROZEN_INPUTS) | {"wrapper_freezer"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=base.DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=base.DEFAULT_TARGETS)
    return parser.parse_args(argv)


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def rule_commitment_sha256(config: Mapping[str, Any]) -> str:
    """Hash every inference decision fixed before DEV64 pixels are opened."""

    payload = {
        "schema": config.get("schema"),
        "joint_protocol_sha256": config.get("joint_protocol_sha256"),
        "scale_cache_protocol_sha256": config.get("scale_cache_protocol_sha256"),
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
        raise RuntimeError("scale256 relation-roster protocol schema changed")
    source = config.get("source_protocol", {})
    names = source.get("dev_filenames")
    if not isinstance(names, list) or len(names) != DEV_SOURCE_COUNT:
        raise RuntimeError("scale256 relation-roster DEV64 is not explicit")
    if len(set(names)) != DEV_SOURCE_COUNT or names_digest(names) != DEV_DIGEST:
        raise RuntimeError("scale256 relation-roster DEV64 roster changed")
    if source.get("dev_digest") != DEV_DIGEST:
        raise RuntimeError("scale256 relation-roster DEV64 digest changed")
    if source.get("dev_draw_index") != DEV_DRAW_INDEX:
        raise RuntimeError("scale256 relation-roster DEV draw changed")
    if source.get("dev_case_seed") != DEV_CASE_SEED:
        raise RuntimeError("scale256 relation-roster DEV seed changed")
    if config.get("runtime") != {
        "device": "mps",
        "inference_batch": 576,
        "model_or_threshold_tuning": False,
        "existing_sha_gated_pipeline_only": True,
    }:
        raise RuntimeError("scale256 relation-roster runtime changed")
    if config.get("output_contract") != {
        "arm_names": list(FUSION_ARM_NAMES),
        "relation_feature_names": list(FEATURE_NAMES),
        "relation_rows_per_arm": 1104,
        "strict_original_upright_permutations": True,
        "contains_pixels": False,
        "contains_exact_references_or_labels": False,
        "normalized_case_key_allowlist": sorted(base.RELATION_ROSTER_CASE_KEYS),
    }:
        raise RuntimeError("scale256 relation-roster output contract changed")
    frozen = config.get("frozen_inputs", {})
    if not isinstance(frozen, Mapping) or set(frozen) != REQUIRED_FROZEN_INPUTS:
        raise RuntimeError("scale256 relation-roster frozen input inventory changed")
    for name, (path, digest) in EXISTING_FROZEN_INPUTS.items():
        if frozen.get(name) != {"path": path, "sha256": digest}:
            raise RuntimeError(f"scale256 relation-roster input changed: {name}")
    wrapper = frozen.get("wrapper_freezer", {})
    wrapper_digest = wrapper.get("sha256")
    if (
        wrapper.get("path") != WRAPPER_PATH
        or not isinstance(wrapper_digest, str)
        or len(wrapper_digest) != 64
    ):
        raise RuntimeError("scale256 relation-roster wrapper record is invalid")
    if config.get("joint_protocol_sha256") != frozen["joint_protocol"]["sha256"]:
        raise RuntimeError("scale256 relation-roster scale protocol digest changed")
    if (
        config.get("scale_cache_protocol_sha256")
        != frozen["scale_cache_protocol"]["sha256"]
    ):
        raise RuntimeError("scale256 relation-roster scale-cache digest changed")
    if config.get("rule_commitment_sha256") != rule_commitment_sha256(config):
        raise RuntimeError("scale256 relation-roster rule commitment changed")


def _verified_record(config: Mapping[str, Any], name: str) -> Path:
    artifact = config.get("frozen_inputs", {}).get(name)
    if not isinstance(artifact, Mapping):
        raise RuntimeError(f"scale256 relation-roster frozen input missing: {name}")
    path = _project_path(str(artifact.get("path", "")))
    if not path.is_file() or sha256_file(path) != artifact.get("sha256"):
        raise RuntimeError(f"scale256 relation-roster frozen input changed: {name}")
    return path


def derive_source_protocol(
    config: Mapping[str, Any],
    scale_config: Mapping[str, Any],
    scale_cache_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconcile scale ``source_contract`` with cache ``source_protocol``."""

    if scale_config.get("schema") != SCALE_SCHEMA:
        raise RuntimeError("scale256 upstream protocol schema changed")
    if scale_cache_config.get("schema") != SCALE_CACHE_SCHEMA:
        raise RuntimeError("scale256 cache protocol schema changed")
    if (
        scale_config.get("status") != SIGNED_STATUS
        or scale_cache_config.get("status") != SIGNED_STATUS
    ):
        raise RuntimeError("scale256 upstream protocols are not signed/fixed")

    scale_source = scale_config.get("source_contract", {})
    cache_source = scale_cache_config.get("source_protocol", {})
    if (
        scale_source.get("selection_namespace") != SELECTION_NAMESPACE
        or cache_source.get("selection_namespace") != SELECTION_NAMESPACE
        or scale_source.get("selection_seed") != SELECTION_SEED
        or cache_source.get("selection_seed") != SELECTION_SEED
    ):
        raise RuntimeError("scale256 source selection lineage changed")
    cache_names = cache_source.get("reserved_dev_filenames")
    if not isinstance(cache_names, list) or len(cache_names) != DEV_SOURCE_COUNT:
        raise RuntimeError("scale-cache explicit DEV64 roster is missing")
    if len(set(cache_names)) != DEV_SOURCE_COUNT or names_digest(cache_names) != DEV_DIGEST:
        raise RuntimeError("scale-cache explicit DEV64 roster changed")
    if (
        scale_source.get("reserved_dev_source_count") != DEV_SOURCE_COUNT
        or cache_source.get("reserved_dev_source_count") != DEV_SOURCE_COUNT
        or scale_source.get("reserved_dev_digest") != DEV_DIGEST
        or cache_source.get("reserved_dev_digest") != DEV_DIGEST
        or scale_source.get("dev_draw_index") != DEV_DRAW_INDEX
        or scale_source.get("dev_case_seed") != DEV_CASE_SEED
    ):
        raise RuntimeError("scale256 DEV64 digest/draw/seed contract changed")
    if (
        scale_source.get("fit_source_count") != cache_source.get("fit_source_count")
        or scale_source.get("fit_digest") != cache_source.get("fit_digest")
        or scale_source.get("fit_draw_indices") != cache_source.get("fit_draw_indices")
        or scale_source.get("fit_case_seed") != cache_source.get("fit_case_seed")
    ):
        raise RuntimeError("scale256 FIT lineage differs from scale-cache protocol")
    if set(cache_names) & set(cache_source.get("fit_filenames", ())):
        raise RuntimeError("scale256 FIT and DEV64 rosters overlap")
    expected = {
        "dev_filenames": list(cache_names),
        "dev_digest": DEV_DIGEST,
        "dev_draw_index": DEV_DRAW_INDEX,
        "dev_case_seed": DEV_CASE_SEED,
    }
    if config.get("source_protocol") != expected:
        raise RuntimeError("wrapper DEV64 source protocol differs from upstream")
    expected_cache_record = scale_config.get("frozen_inputs", {}).get(
        "scale_cache_config"
    )
    if expected_cache_record != config.get("frozen_inputs", {}).get(
        "scale_cache_protocol"
    ):
        raise RuntimeError("scale protocol points to a different scale-cache config")
    return expected


def _load_frozen_signed_json(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    path = _verified_record(config, name)
    digest = sha256_file(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if (
        not sidecar.is_file()
        or sidecar.read_text(encoding="utf-8").split()[0] != digest
    ):
        raise RuntimeError(f"scale256 upstream sidecar changed: {name}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_signed_config(path: Path) -> tuple[dict[str, Any], str]:
    resolved = path.resolve()
    config = json.loads(resolved.read_text(encoding="utf-8"))
    _require_exact_contract(config)
    if config.get("status") == BLOCKED_STATUS:
        raise RuntimeError("scale256 relation-roster template is intentionally blocked")
    digest = sha256_file(resolved)
    sidecar = resolved.with_suffix(resolved.suffix + ".sha256")
    if (
        config.get("status") != SIGNED_STATUS
        or not sidecar.is_file()
        or sidecar.read_text(encoding="utf-8").split()[0] != digest
    ):
        raise RuntimeError("scale256 relation-roster protocol is not signed/fixed")
    for name in config["frozen_inputs"]:
        _verified_record(config, name)
    scale_config = _load_frozen_signed_json(config, "joint_protocol")
    scale_cache_config = _load_frozen_signed_json(config, "scale_cache_protocol")
    derive_source_protocol(config, scale_config, scale_cache_config)
    return config, digest


def _validate_runtime_paths(
    args: argparse.Namespace, config: Mapping[str, Any]
) -> None:
    expected_manifest = _project_path(
        config["frozen_inputs"]["validation_manifest"]["path"]
    )
    if args.manifest.resolve() != expected_manifest:
        raise RuntimeError("runtime manifest differs from frozen DEV64 protocol")
    if args.targets.resolve() != base.DEFAULT_TARGETS.resolve():
        raise RuntimeError("runtime targets directory differs from frozen DEV64 protocol")


def run(args: argparse.Namespace) -> dict[str, Any]:
    config, config_sha256 = _load_signed_config(args.config)
    _validate_runtime_paths(args, config)
    return base.freeze_target_free_roster(args, config, config_sha256)


def main(argv: Sequence[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

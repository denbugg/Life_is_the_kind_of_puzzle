"""Fail-closed production packaging for the frozen DRUNet protected stack.

This is deliberately separate from :mod:`aiijc_puzzle.compliant_submission`.
The historical h20x1 fallback, its schema and its published artifacts remain
immutable.  Production is blocked until an immutable promotion authorization
binds the preregistration, both quantitative reports, commitments, receipts and
both root manual reviews.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
import torch
from jsonschema import Draft202012Validator

from aiijc_puzzle.compliant_atlas_decoder import audit_raw_permutation
from aiijc_puzzle.compliant_submission import (
    EXPECTED_TEST_FILES,
    METHOD_STATUS,
    OFFICIAL_FILENAMES_SHA256,
    OFFICIAL_TEST_ARCHIVE_SHA256,
    InputSnapshot,
    _absolute_without_resolving,
    _mkdir_without_symlink_ancestors,
    _require_regular_file,
    array_sha256,
    atomic_write_json,
    build_official_input_snapshot,
    guard_artifact_paths,
    load_rgb_png,
)
from aiijc_puzzle.edge_protected_nlm import colored_nlm, protected_masks
from aiijc_puzzle.legacy_upgrade import (
    atomic_write_png,
    deterministic_submission_zip,
    directional_scores,
    layout_digest,
    solve_buddies,
)
from aiijc_puzzle.postassembly_harmonizer import (
    DEFAULT_LUMINANCE_GAIN_CONFIG,
    DEFAULT_SEAM_GRAPH_CONFIG,
    apply_luminance_gains,
    apply_rgb_offsets,
    seam_graph_luminance_gains,
    seam_graph_rgb_offsets,
)
from aiijc_puzzle.pretrained_tile_denoiser import load_drunet_color, render_drunet_tiles
from aiijc_puzzle.protocol import (
    IMAGE_SIZE,
    RGB_CHANNELS,
    TILE_COUNT,
    TILE_SIZE,
    assemble_tiles,
    sha256_file,
    split_tiles,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "outputs/compliant-drunet-protected-submission-v1"
DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / "predictions"
DEFAULT_OUTPUT_ZIP = OUTPUT_ROOT / "submission.zip"
DEFAULT_ATTESTATION = OUTPUT_ROOT / "compliance-attestation.json"
DEFAULT_VALIDATION_REPORT = OUTPUT_ROOT / "independent-validation.json"
DEFAULT_PROMOTION_CONFIG = PROJECT_ROOT / "configs/compliant_drunet_protected_submission_v1.json"
DEFAULT_SCHEMA_PATH = PROJECT_ROOT / "configs/compliant-drunet-protected-submission-v3.schema.json"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "artifacts/pretrained-denoisers/kair-fc1732f/drunet_color.pth"

SCHEMA_NAME = "aiijc-puzzle-drunet-protected-submission-compliance-v3"
PROOF_SCOPE = "provenance_bijection_geometry_frozen_drunet_and_protected_tail_only"
PROOF_LIMITATION = (
    "PASS proves corresponding-input provenance, upright 20x20 strict bijection, raw "
    "geometry, the frozen bilateral buddies96 solver, independent-tile official DRUNet40 "
    "and the frozen protected NLM tail only; it does not prove the hidden ground-truth "
    "permutation, reconstruction accuracy, or manual acceptance."
)
RESTORATION_NAME = (
    "rgb-seam-offsets_then-bounded-luminance-gains_then-independent-tile-official-"
    "color-DRUNet-sigma40-then-independent-colored-NLM-h20-h28-h40-then-t40-"
    "protected-h28-flat-h40-blend"
)
PROMOTED_ARM = "D_drunet_sigma40_protected_h28_h40_t40"
EDGE_BUDGET = 96
DRUNET_SIGMA_255 = 40.0
DRUNET_BATCH_SIZE = 144
SOBEL_THRESHOLD = 40.0
NLM_STRENGTHS = (20, 28, 40)
NLM_TEMPLATE_WINDOW = 7
NLM_SEARCH_WINDOW = 21
CANONICAL_DEVICE = "mps"
CHECKPOINT_URL = "https://github.com/cszn/KAIR/releases/download/v1.0/drunet_color.pth"
CHECKPOINT_SHA256 = "479abe3c5327dfd10ff54a80ec7d4098ca80752a5c9492cdff31cee430bec4b4"
KAIR_COMMIT = "fc1732f4a4514e42ce15e5b3a1e18c828af47a1e"
MODEL_PARAMETER_COUNT = 32_640_960

# Filled with the hash of the new parallel schema.  It is intentionally not the
# historical h20x1 schema hash.
PINNED_SCHEMA_SHA256 = "a5581b56604671cee44747ede095a2666f2375ebd74962b8ffaa5484fcc5bf69"

EXPECTED_ASSET_SHA256 = {
    "artifacts/pretrained-denoisers/kair-fc1732f/LICENSE": (
        "448e69b705d64f21bf8cb86562301e0edd99ac79026064ddd75af8242b067be5"
    ),
    "artifacts/pretrained-denoisers/kair-fc1732f/drunet_color.pth": CHECKPOINT_SHA256,
    "artifacts/pretrained-denoisers/kair-fc1732f/models/basicblock.py": (
        "48406db8867394ac5ae233ebeec7711ac10acfc3a6bbf0072c33aa77d659b6fd"
    ),
    "artifacts/pretrained-denoisers/kair-fc1732f/models/network_unet.py": (
        "8043b6350f1589d5f08892e3be0b4d12c5a502058014285107b7360696d12bf5"
    ),
}
EXPECTED_HARMONIZER_SHA256 = {
    "configs/postassembly_rgb_offset_v1.json": (
        "4adfd9b614e8556b7de5c1f527d759d15d29c0f74e20aa26ff87900dd773ec9a"
    ),
    "configs/postassembly_luminance_gain_v1.json": (
        "7488cad2ae7cc75792d6ff0ff2ea0a38fa778979083ffd5c161c857b68fd550f"
    ),
}
EXPECTED_POLICY = {
    "output_derived_only_from_corresponding_input": True,
    "all_576_input_tiles_used_exactly_once": True,
    "tile_identity_preserved_before_restoration": True,
    "tile_geometry_preserved_before_restoration": True,
    "restoration_after_layout_only": True,
    "drunet_each_tile_independent": True,
    "drunet_cross_tile_context_used": False,
    "targets_used": False,
    "reference_images_used": False,
    "source_lookup_used": False,
    "external_templates_used": False,
    "cross_board_pixels_used": False,
    "tile_substitution_used": False,
    "rotation_or_flip_used": False,
    "resize_or_warp_used": False,
    "filename_or_board_overrides_used": False,
}

RUNTIME_FILE_RELATIVE_PATHS = (
    "src/aiijc_puzzle/compliant_drunet_protected_submission.py",
    "src/aiijc_puzzle/compliant_drunet_protected_validation.py",
    "src/aiijc_puzzle/compliant_submission.py",
    "src/aiijc_puzzle/compliant_atlas_decoder.py",
    "src/aiijc_puzzle/legacy_upgrade.py",
    "src/aiijc_puzzle/postassembly_harmonizer.py",
    "src/aiijc_puzzle/protocol.py",
    "src/aiijc_puzzle/pretrained_tile_denoiser.py",
    "src/aiijc_puzzle/edge_protected_nlm.py",
    "scripts/run_compliant_drunet_protected_submission.py",
    "scripts/validate_compliant_drunet_protected_submission.py",
    "configs/compliant-drunet-protected-submission-v3.schema.json",
    "configs/compliant_drunet_protected_submission_v1.json",
    "configs/postassembly_rgb_offset_v1.json",
    "configs/postassembly_luminance_gain_v1.json",
    "configs/pretrained_drunet_protected_stack_preregistered_v1.json",
    "uv.lock",
)

PROMOTION_EVIDENCE_PATHS = {
    "preregistration": "configs/pretrained_drunet_protected_stack_preregistered_v1.json",
    "primary_commitment": (
        "outputs/pretrained-drunet-protected-stack/primary-calibration-offset264-count120/"
        "prediction-commitment.json"
    ),
    "primary_commitment_receipt": (
        "outputs/pretrained-drunet-protected-stack/"
        "primary-calibration-offset264-count120.commitment-receipt.json"
    ),
    "primary_report": (
        "outputs/pretrained-drunet-protected-stack/primary-calibration-offset264-count120/"
        "report.json"
    ),
    "primary_manual_review": (
        "outputs/pretrained-drunet-protected-stack/primary-calibration-offset264-count120/"
        "manual-review.json"
    ),
    "confirmation_commitment": (
        "outputs/pretrained-drunet-protected-stack/"
        "confirmation-calibration-offset408-count120/prediction-commitment.json"
    ),
    "confirmation_commitment_receipt": (
        "outputs/pretrained-drunet-protected-stack/"
        "confirmation-calibration-offset408-count120.commitment-receipt.json"
    ),
    "confirmation_report": (
        "outputs/pretrained-drunet-protected-stack/"
        "confirmation-calibration-offset408-count120/report.json"
    ),
    "confirmation_manual_review": (
        "outputs/pretrained-drunet-protected-stack/"
        "confirmation-calibration-offset408-count120/manual-review.json"
    ),
}


@dataclass(frozen=True)
class PromotionEvidence:
    """Immutable authorization and the exact evidence hashes it binds."""

    config_sha256: str
    artifacts: dict[str, dict[str, str]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "config_sha256": self.config_sha256,
            "artifacts": json.loads(json.dumps(self.artifacts, sort_keys=True)),
            "promoted_arm": PROMOTED_ARM,
            "production_authorized": True,
        }


@dataclass(frozen=True)
class ProtectedSubmissionPrediction:
    """One strict-layout production output and its target-free evidence."""

    layout: np.ndarray
    raw: np.ndarray
    harmonized: np.ndarray
    restored: np.ndarray
    audit: dict[str, Any]
    tile_multiset_sha256: str
    restoration: dict[str, Any]
    score_seconds: float
    solve_seconds: float
    restoration_seconds: float


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_readonly(path: Path) -> None:
    if path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise PermissionError(f"integrity-bound artifact is writable: {path}")


def _load_json_object(path: Path) -> dict[str, Any]:
    checked = _require_regular_file(path)
    payload = json.loads(checked.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {checked}")
    return payload


def _check_exact_keys(payload: Mapping[str, Any], expected: set[str], context: str) -> None:
    if set(payload) != expected:
        raise ValueError(f"{context} keys changed: {sorted(payload)}")


def _validate_quantitative_report(
    report: Mapping[str, Any],
    *,
    stage: str,
    preregistration_sha256: str,
) -> None:
    if report.get("schema") != "aiijc-pretrained-drunet-protected-stack-report-v1":
        raise ValueError(f"{stage} report schema changed")
    if report.get("status") != "scored_from_frozen_predictions":
        raise ValueError(f"{stage} report is not a completed frozen score")
    if report.get("stage") != stage or report.get("count") != 120:
        raise ValueError(f"{stage} report panel changed")
    expected_offset = 264 if stage == "primary" else 408
    if report.get("offset") != expected_offset:
        raise ValueError(f"{stage} report offset changed")
    if report.get("preregistration_sha256") != preregistration_sha256:
        raise ValueError(f"{stage} report is not bound to the preregistration")
    if report.get("quantitative_pass") is not True:
        raise ValueError(f"{stage} quantitative gate did not pass")
    if report.get("selected_passing_winner") != PROMOTED_ARM:
        raise ValueError(f"{stage} did not select the frozen combined arm")
    if report.get("competition_test_access") is not False:
        raise ValueError(f"{stage} report declares competition-test access")
    if report.get("holdout_access") is not False:
        raise ValueError(f"{stage} report declares holdout access")
    checks = report.get("quantitative_checks")
    if not isinstance(checks, dict) or not checks or not all(checks.values()):
        raise ValueError(f"{stage} quantitative checks are incomplete")


def _validate_manual_review(
    review: Mapping[str, Any],
    *,
    stage: str,
    preregistration_sha256: str,
    report_sha256: str,
) -> None:
    if review.get("reviewer") != "root" or review.get("reviewed_arm") != PROMOTED_ARM:
        raise ValueError(f"{stage} review has the wrong reviewer or arm")
    if review.get("reviewed_board_count") != 120:
        raise ValueError(f"{stage} review does not cover all 120 boards")
    if review.get("reviewed_all_full_canvas_triplets") is not True:
        raise ValueError(f"{stage} review did not inspect every full-canvas triplet")
    if review.get("passed") is not True or review.get("severe_artifacts") != 0:
        raise ValueError(f"{stage} manual review did not pass with zero severe artifacts")
    if review.get("material_face_text_or_object_loss") is not False:
        raise ValueError(f"{stage} review found material structure loss")
    if review.get("mask_halo_or_boundary_damage") is not False:
        raise ValueError(f"{stage} review found mask or boundary damage")
    if review.get("preregistration_sha256") != preregistration_sha256:
        raise ValueError(f"{stage} review preregistration binding changed")
    report_key = f"{stage}_report_sha256"
    if review.get(report_key) != report_sha256:
        raise ValueError(f"{stage} review report binding changed")


def load_promotion_evidence(
    config_path: Path = DEFAULT_PROMOTION_CONFIG,
    *,
    project_root: Path = PROJECT_ROOT,
) -> PromotionEvidence:
    """Validate the final root authorization and every hash-bound gate artifact."""

    config_path = _require_regular_file(config_path)
    _require_readonly(config_path)
    config = _load_json_object(config_path)
    _check_exact_keys(
        config,
        {
            "schema",
            "status",
            "production_authorized_by_root",
            "canonical_device",
            "promoted_arm",
            "pipeline",
            "evidence",
        },
        "promotion config",
    )
    if config["schema"] != "aiijc-drunet-protected-production-authorization-v1":
        raise ValueError("promotion authorization schema changed")
    if config["status"] != "PRIMARY_AND_CONFIRMATION_NUMERIC_AND_MANUAL_PASS":
        raise ValueError("promotion authorization does not declare both panels passed")
    if config["production_authorized_by_root"] is not True:
        raise ValueError("root did not authorize production")
    if config["canonical_device"] != CANONICAL_DEVICE:
        raise ValueError("promotion authorization canonical device changed")
    if config["promoted_arm"] != PROMOTED_ARM:
        raise ValueError("promotion authorization arm changed")
    if config["pipeline"] != frozen_pipeline_record():
        raise ValueError("promotion authorization pipeline changed")
    evidence = config["evidence"]
    if not isinstance(evidence, dict) or set(evidence) != set(PROMOTION_EVIDENCE_PATHS):
        raise ValueError("promotion evidence roster changed")

    verified: dict[str, dict[str, str]] = {}
    loaded: dict[str, dict[str, Any]] = {}
    root = project_root.resolve()
    for name, expected_relative in PROMOTION_EVIDENCE_PATHS.items():
        record = evidence[name]
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise ValueError(f"invalid promotion evidence record: {name}")
        if record["path"] != expected_relative:
            raise ValueError(f"promotion evidence path changed: {name}")
        artifact_path = root / expected_relative
        checked = _require_regular_file(artifact_path)
        _require_readonly(checked)
        actual_sha256 = sha256_file(checked)
        if record["sha256"] != actual_sha256:
            raise ValueError(f"promotion evidence hash changed: {name}")
        verified[name] = {"path": expected_relative, "sha256": actual_sha256}
        loaded[name] = _load_json_object(checked)

    preregistration_sha256 = verified["preregistration"]["sha256"]
    preregistration = loaded["preregistration"]
    if preregistration.get("schema") != (
        "aiijc-pretrained-drunet-protected-stack-preregistration-v1"
    ):
        raise ValueError("combined-stack preregistration schema changed")
    if preregistration.get("arm_names", [])[-1:] != [PROMOTED_ARM]:
        raise ValueError("combined-stack preregistration candidate changed")
    geometry = preregistration.get("geometry_contract", {})
    required_geometry = {
        "strict_shared_layout": True,
        "all_576_upright_tiles_preserved_one_to_one": True,
        "same_board_pixels_only": True,
        "cross_tile_neural_context": False,
        "cross_board_context": False,
        "resize": False,
        "warp": False,
        "rotation": False,
        "flip": False,
        "external_reference_or_template_pixels": False,
        "generation_or_substitution": False,
    }
    if any(geometry.get(key) != value for key, value in required_geometry.items()):
        raise ValueError("combined-stack geometry contract changed")

    for stage in ("primary", "confirmation"):
        report_name = f"{stage}_report"
        report_sha256 = verified[report_name]["sha256"]
        report = loaded[report_name]
        _validate_quantitative_report(
            report,
            stage=stage,
            preregistration_sha256=preregistration_sha256,
        )
        if report.get("commitment_sha256") != verified[f"{stage}_commitment"]["sha256"]:
            raise ValueError(f"{stage} report commitment binding changed")
        if (
            report.get("commitment_receipt_sha256")
            != verified[f"{stage}_commitment_receipt"]["sha256"]
        ):
            raise ValueError(f"{stage} report receipt binding changed")
        _validate_manual_review(
            loaded[f"{stage}_manual_review"],
            stage=stage,
            preregistration_sha256=preregistration_sha256,
            report_sha256=report_sha256,
        )

    return PromotionEvidence(
        config_sha256=sha256_file(config_path),
        artifacts=verified,
    )


def verify_official_assets() -> dict[str, str]:
    """Verify the official MIT KAIR sources and checkpoint without decoding targets."""

    observed = {
        relative: sha256_file(_require_regular_file(PROJECT_ROOT / relative))
        for relative in EXPECTED_ASSET_SHA256
    }
    if observed != EXPECTED_ASSET_SHA256:
        raise ValueError("official KAIR assets changed")
    license_text = (PROJECT_ROOT / next(iter(EXPECTED_ASSET_SHA256))).read_text(encoding="utf-8")
    if "MIT License" not in license_text or "Copyright (c) 2019 Kai Zhang" not in license_text:
        raise ValueError("official KAIR license text changed")
    return observed


def verify_harmonizer_configs() -> dict[str, str]:
    observed = {
        relative: sha256_file(_require_regular_file(PROJECT_ROOT / relative))
        for relative in EXPECTED_HARMONIZER_SHA256
    }
    if observed != EXPECTED_HARMONIZER_SHA256:
        raise ValueError("frozen harmonizer configs changed")
    return observed


def tile_multiset_sha256(image: np.ndarray) -> str:
    """Hash the sorted roster of all exact upright tile bytes."""

    value = np.asarray(image)
    if value.shape != (IMAGE_SIZE, IMAGE_SIZE, RGB_CHANNELS) or value.dtype != np.uint8:
        raise ValueError("tile multiset requires one uint8 RGB 480x480 board")
    hashes = sorted(hashlib.sha256(tile.tobytes()).digest() for tile in split_tiles(value))
    return hashlib.sha256(b"".join(hashes)).hexdigest()


def _apply_frozen_harmonizers(raw: np.ndarray) -> np.ndarray:
    ordered = split_tiles(raw)
    offsets, _ = seam_graph_rgb_offsets(ordered, DEFAULT_SEAM_GRAPH_CONFIG)
    rgb_corrected = apply_rgb_offsets(ordered, offsets)
    gains, _ = seam_graph_luminance_gains(
        rgb_corrected,
        DEFAULT_LUMINANCE_GAIN_CONFIG,
    )
    return assemble_tiles(apply_luminance_gains(rgb_corrected, gains))


def frozen_pipeline_record() -> dict[str, Any]:
    return {
        "layout": "bilateral directional scores -> solve_buddies(max_edges=96)",
        "strict_upright_tile_bijection": True,
        "harmonizer": "RGB seam offsets -> bounded luminance gains",
        "drunet": {
            "architecture": "official colour DRUNet UNetRes",
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "sigma_255": DRUNET_SIGMA_255,
            "tile_input": [20, 20, 3],
            "same_tile_reflect_padding_right_bottom": [4, 4],
            "exact_crop": [20, 20],
            "batch_size": DRUNET_BATCH_SIZE,
            "cross_tile_context": False,
            "cross_board_context": False,
        },
        "nlm": {
            "independent_single_pass_strengths": list(NLM_STRENGTHS),
            "h_color_equals_h": True,
            "template_window": NLM_TEMPLATE_WINDOW,
            "search_window": NLM_SEARCH_WINDOW,
        },
        "mask": {
            "source": "DRUNet then independent h20",
            "sobel_threshold": SOBEL_THRESHOLD,
            "all_20px_grid_boundaries_protected": True,
            "dilation": "3x3 one iteration",
            "softening": "Gaussian sigma1",
        },
        "blend": "rint(soft*h28 + (1-soft)*h40), clip uint8",
    }


def predict_drunet_protected(
    input_image: np.ndarray,
    model: torch.nn.Module,
    *,
    device: torch.device,
) -> ProtectedSubmissionPrediction:
    """Run the fixed legal candidate on one corresponding dirty board only."""

    value = np.asarray(input_image)
    if value.shape != (IMAGE_SIZE, IMAGE_SIZE, RGB_CHANNELS) or value.dtype != np.uint8:
        raise ValueError("production input must be uint8 RGB 480x480")
    input_tiles = split_tiles(value)
    score_started = perf_counter()
    right, down = directional_scores(input_tiles, views=("bilateral",))["bilateral"]
    score_seconds = perf_counter() - score_started
    solved = solve_buddies(right, down, max_edges=EDGE_BUDGET)
    layout = np.asarray(solved.layout, dtype=np.int32)
    raw = assemble_tiles(input_tiles[layout])
    audit_object = audit_raw_permutation(
        value,
        raw,
        layout,
        restoration_applied_after_audit=True,
    )
    if not audit_object.passed:
        raise RuntimeError("strict raw permutation and pixel multiset audit failed")
    input_multiset = tile_multiset_sha256(value)
    if tile_multiset_sha256(raw) != input_multiset:
        raise RuntimeError("independent raw tile-multiset audit failed")

    restoration_started = perf_counter()
    harmonized = _apply_frozen_harmonizers(raw)
    harmonized_tiles = split_tiles(harmonized)
    drunet_tiles, drunet_diagnostics = render_drunet_tiles(
        model,
        harmonized_tiles,
        sigma_255=DRUNET_SIGMA_255,
        device=device,
        batch_size=DRUNET_BATCH_SIZE,
    )
    if drunet_tiles.shape != harmonized_tiles.shape:
        raise RuntimeError("DRUNet changed the exact tile roster shape")
    drunet_canvas = assemble_tiles(drunet_tiles)
    h20 = colored_nlm(drunet_canvas, 20)
    h28 = colored_nlm(drunet_canvas, 28)
    h40 = colored_nlm(drunet_canvas, 40)
    binary, soft, protected_fraction = protected_masks(
        h20,
        sobel_threshold=SOBEL_THRESHOLD,
    )
    mixed = np.rint(
        soft[..., None] * h28.astype(np.float32) + (1.0 - soft[..., None]) * h40.astype(np.float32)
    )
    restored = np.ascontiguousarray(mixed.clip(0, 255).astype(np.uint8))
    restoration_seconds = perf_counter() - restoration_started
    restoration = {
        "harmonized_array_sha256": array_sha256(harmonized),
        "drunet_tiles_array_sha256": array_sha256(drunet_tiles),
        "drunet_canvas_array_sha256": array_sha256(drunet_canvas),
        "drunet_diagnostics": drunet_diagnostics.as_dict(),
        "nlm_h20_array_sha256": array_sha256(h20),
        "nlm_h28_array_sha256": array_sha256(h28),
        "nlm_h40_array_sha256": array_sha256(h40),
        "binary_mask_array_sha256": array_sha256(binary),
        "soft_mask_array_sha256": array_sha256(soft),
        "protected_fraction": protected_fraction,
        "output_array_sha256": array_sha256(restored),
    }
    return ProtectedSubmissionPrediction(
        layout=layout,
        raw=raw,
        harmonized=harmonized,
        restored=restored,
        audit=audit_object.as_dict(),
        tile_multiset_sha256=input_multiset,
        restoration=restoration,
        score_seconds=score_seconds,
        solve_seconds=solved.runtime_seconds,
        restoration_seconds=restoration_seconds,
    )


def build_runtime_manifest() -> dict[str, Any]:
    files = {
        relative: sha256_file(_require_regular_file(PROJECT_ROOT / relative))
        for relative in RUNTIME_FILE_RELATIVE_PATHS
    }
    assets = verify_official_assets()
    harmonizers = verify_harmonizer_configs()
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "torch": torch.__version__,
    }
    content: dict[str, Any] = {
        "files": files,
        "assets": assets,
        "harmonizers": harmonizers,
        "versions": versions,
        "canonical_device": CANONICAL_DEVICE,
    }
    return {**content, "digest_sha256": _canonical_json_sha256(content)}


def _restoration_attestation(prediction: ProtectedSubmissionPrediction) -> dict[str, Any]:
    diagnostics = prediction.restoration["drunet_diagnostics"]
    return {
        "name": RESTORATION_NAME,
        "input_is_raw_assembly": True,
        "pixel_restoration_only": True,
        "layout_changed": False,
        "spatial_warp_used": False,
        "external_or_cross_board_pixels_used": False,
        "harmonizers": {
            "order": ["rgb_seam_offsets", "bounded_luminance_gains"],
            "config_sha256": dict(EXPECTED_HARMONIZER_SHA256),
        },
        "harmonized_array_sha256": prediction.restoration["harmonized_array_sha256"],
        "drunet": {
            "official_repository": "https://github.com/cszn/KAIR",
            "official_commit": KAIR_COMMIT,
            "license": "MIT",
            "checkpoint_url": CHECKPOINT_URL,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "parameter_count": MODEL_PARAMETER_COUNT,
            "sigma_255": DRUNET_SIGMA_255,
            "tile_count": TILE_COUNT,
            "tile_input_size": [TILE_SIZE, TILE_SIZE],
            "same_tile_reflect_padding_right_bottom": [4, 4],
            "exact_crop_size": [TILE_SIZE, TILE_SIZE],
            "batch_size": DRUNET_BATCH_SIZE,
            "cross_tile_context": False,
            "cross_board_context": False,
            "restored_tiles_array_sha256": prediction.restoration["drunet_tiles_array_sha256"],
            "restored_canvas_array_sha256": prediction.restoration["drunet_canvas_array_sha256"],
            "runtime_diagnostics": diagnostics,
        },
        "nlm": {
            "proper_rgb_bgr_roundtrip": True,
            "independent_single_pass_strengths": list(NLM_STRENGTHS),
            "h_color_equals_h": True,
            "template_window_size": NLM_TEMPLATE_WINDOW,
            "search_window_size": NLM_SEARCH_WINDOW,
            "h20_array_sha256": prediction.restoration["nlm_h20_array_sha256"],
            "h28_array_sha256": prediction.restoration["nlm_h28_array_sha256"],
            "h40_array_sha256": prediction.restoration["nlm_h40_array_sha256"],
        },
        "protected_mask": {
            "source": "independent DRUNet+h20",
            "sobel_threshold": SOBEL_THRESHOLD,
            "grid_period": TILE_SIZE,
            "dilation": "3x3 one iteration",
            "softening": "Gaussian sigma1",
            "binary_array_sha256": prediction.restoration["binary_mask_array_sha256"],
            "soft_array_sha256": prediction.restoration["soft_mask_array_sha256"],
            "protected_fraction": prediction.restoration["protected_fraction"],
        },
        "blend": "rint(soft*h28 + (1-soft)*h40), clip uint8",
        "output_array_sha256": prediction.restoration["output_array_sha256"],
    }


def board_attestation(
    *,
    filename: str,
    input_sha256: str,
    prediction: ProtectedSubmissionPrediction,
    output_png_sha256: str,
) -> dict[str, Any]:
    layout = np.asarray(prediction.layout, dtype=np.int32)
    if layout.shape != (TILE_COUNT,) or not np.array_equal(
        np.sort(layout), np.arange(TILE_COUNT, dtype=np.int32)
    ):
        raise ValueError("production layout is not a strict 0..575 permutation")
    if prediction.audit.get("passed") is not True:
        raise ValueError("production board attestation requires a passed raw audit")
    return {
        "filename": filename,
        "input_sha256": input_sha256,
        "tile_at_position": layout.tolist(),
        "layout_sha256": layout_digest(layout),
        "raw_assembly_sha256": array_sha256(prediction.raw),
        "input_and_raw_tile_multiset_sha256": prediction.tile_multiset_sha256,
        "raw_permutation_audit": prediction.audit,
        "restoration": _restoration_attestation(prediction),
        "output_png_sha256": output_png_sha256,
    }


def _method_declaration(
    promotion: PromotionEvidence,
    runtime_manifest: Mapping[str, Any],
) -> str:
    return (
        "corresponding-input-only; upright-20x20-tiles; bilateral-directional-scores; "
        "strict-buddies96-permutation; raw-pixel-and-tile-multiset-audit; "
        "rgb-luma-harmonize; official-independent-tile-DRUNet-sigma40; "
        "independent-NLM-h20-h28-h40; exact-t40-protected-h28-flat-h40; "
        f"promotion-config-sha256={promotion.config_sha256}; "
        f"runtime-manifest-sha256={runtime_manifest['digest_sha256']}"
    )


def build_attestation(
    *,
    snapshot: InputSnapshot,
    archive_sha256: str,
    per_board: Sequence[Mapping[str, Any]],
    promotion: PromotionEvidence,
    runtime_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    records = [dict(record) for record in per_board]
    if [record.get("filename") for record in records] != list(snapshot.filenames):
        raise ValueError("per-board evidence must follow the exact official roster")
    return {
        "schema": SCHEMA_NAME,
        "status": METHOD_STATUS,
        "scope": PROOF_SCOPE,
        "correct_hidden_layout_proven": False,
        "proof_limitation": PROOF_LIMITATION,
        "method": _method_declaration(promotion, runtime_manifest),
        "policy": dict(EXPECTED_POLICY),
        "promotion_evidence": promotion.as_dict(),
        "runtime_manifest": json.loads(json.dumps(runtime_manifest)),
        "input_snapshot": snapshot.attestation_record(),
        "archive": {
            "sha256": archive_sha256,
            "file_count": snapshot.file_count,
            "root_only": True,
            "filenames_match_input_snapshot": True,
            "format": "PNG",
            "mode": "RGB",
            "width": IMAGE_SIZE,
            "height": IMAGE_SIZE,
            "filenames": list(snapshot.filenames),
        },
        "canonical_execution": {
            "device": CANONICAL_DEVICE,
            "cuda_or_cpu_reproduction_is_noncanonical": True,
            "noncanonical_backend_may_differ_by_one_lsb": True,
        },
        "per_board": records,
    }


def _load_schema(path: Path = DEFAULT_SCHEMA_PATH) -> dict[str, Any]:
    checked = _require_regular_file(path)
    actual = sha256_file(checked)
    if actual != PINNED_SCHEMA_SHA256:
        raise ValueError(
            f"DRUNet protected schema hash changed: expected {PINNED_SCHEMA_SHA256}, got {actual}"
        )
    schema = json.loads(checked.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    if schema.get("properties", {}).get("schema", {}).get("const") != SCHEMA_NAME:
        raise ValueError("DRUNet protected schema name changed")
    return schema


def load_and_validate_attestation(path: Path) -> dict[str, Any]:
    payload = _load_json_object(path)
    Draft202012Validator(_load_schema()).validate(payload)
    return payload


def _guard_validation_report(path: Path, artifacts: Sequence[Path]) -> Path:
    report = _absolute_without_resolving(path)
    if report.exists() or report.is_symlink():
        raise FileExistsError(f"refusing to overwrite validation report: {report}")
    report_resolved = report.resolve(strict=False)
    artifact_resolved = [artifact.resolve(strict=False) for artifact in artifacts]
    if any(
        report_resolved == artifact
        or report_resolved in artifact.parents
        or artifact in report_resolved.parents
        for artifact in artifact_resolved
    ):
        raise ValueError("validation report path overlaps another output artifact")
    _mkdir_without_symlink_ancestors(report.parent)
    return report


def _require_canonical_mps() -> torch.device:
    if not torch.backends.mps.is_available():
        raise RuntimeError("canonical production requires Apple MPS")
    return torch.device("mps")


def _unused_temporary_path(*, prefix: str, parent: Path) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=prefix, dir=parent)
    os.close(descriptor)
    path = Path(name)
    path.unlink()
    return path


def run_production_submission(
    *,
    inputs_dir: Path,
    source_archive: Path,
) -> dict[str, Any]:
    """Generate, independently recompute, and atomically publish the 700-board bundle."""

    # Imported lazily so the validation implementation remains a separate code path.
    from aiijc_puzzle.compliant_drunet_protected_validation import (
        validate_against_snapshot,
    )

    promotion = load_promotion_evidence()
    _load_schema()
    snapshot = build_official_input_snapshot(inputs_dir, source_archive)
    runtime_manifest = build_runtime_manifest()
    device = _require_canonical_mps()
    model = load_drunet_color(DEFAULT_CHECKPOINT, device)
    if sum(parameter.numel() for parameter in model.parameters()) != MODEL_PARAMETER_COUNT:
        raise ValueError("official DRUNet parameter count changed")
    inputs, source, output, archive, attestation = guard_artifact_paths(
        inputs_dir=inputs_dir,
        source_archive=source_archive,
        output_dir=DEFAULT_OUTPUT_DIR,
        output_zip=DEFAULT_OUTPUT_ZIP,
        attestation_path=DEFAULT_ATTESTATION,
    )
    validation_report = _guard_validation_report(
        DEFAULT_VALIDATION_REPORT,
        (output, archive, attestation),
    )

    staging_dir = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    zip_descriptor, zip_name = tempfile.mkstemp(prefix=f".{archive.name}.", dir=archive.parent)
    os.close(zip_descriptor)
    temporary_zip = Path(zip_name)
    temporary_attestation = _unused_temporary_path(
        prefix=f".{attestation.name}.",
        parent=attestation.parent,
    )
    temporary_validation = _unused_temporary_path(
        prefix=f".{validation_report.name}.",
        parent=validation_report.parent,
    )
    published = False
    board_records: list[dict[str, Any]] = []
    score_seconds = 0.0
    solve_seconds = 0.0
    restoration_seconds = 0.0
    started = perf_counter()
    try:
        for index, name in enumerate(snapshot.filenames, start=1):
            image = load_rgb_png(inputs / name, expected_sha256=snapshot.hashes_by_name[name])
            prediction = predict_drunet_protected(image, model, device=device)
            output_png_sha256 = atomic_write_png(staging_dir / name, prediction.restored)
            board_records.append(
                board_attestation(
                    filename=name,
                    input_sha256=snapshot.hashes_by_name[name],
                    prediction=prediction,
                    output_png_sha256=output_png_sha256,
                )
            )
            score_seconds += prediction.score_seconds
            solve_seconds += prediction.solve_seconds
            restoration_seconds += prediction.restoration_seconds
            print(f"[{index:03d}/{snapshot.file_count}] {name}", flush=True)

        if build_official_input_snapshot(inputs, source) != snapshot:
            raise RuntimeError("official input snapshot changed during production")
        archive_sha256 = deterministic_submission_zip(
            staging_dir,
            list(snapshot.filenames),
            temporary_zip,
        )
        attestation_payload = build_attestation(
            snapshot=snapshot,
            archive_sha256=archive_sha256,
            per_board=board_records,
            promotion=promotion,
            runtime_manifest=runtime_manifest,
        )
        atomic_write_json(temporary_attestation, attestation_payload)

        # This loads a second official model and recomputes every one of the 700
        # boards from the official input archive before anything is published.
        del model
        torch.mps.empty_cache()
        validation = validate_against_snapshot(
            snapshot=snapshot,
            inputs_dir=inputs,
            submission_zip=temporary_zip,
            attestation_path=temporary_attestation,
            promotion=promotion,
            runtime_manifest=runtime_manifest,
            device=device,
        )
        atomic_write_json(temporary_validation, validation)
        os.replace(staging_dir, output)
        os.replace(temporary_zip, archive)
        os.replace(temporary_attestation, attestation)
        os.replace(temporary_validation, validation_report)
        published = True
    finally:
        if not published:
            shutil.rmtree(staging_dir, ignore_errors=True)
            temporary_zip.unlink(missing_ok=True)
            temporary_attestation.unlink(missing_ok=True)
            temporary_validation.unlink(missing_ok=True)

    return {
        "status": METHOD_STATUS,
        "scope": PROOF_SCOPE,
        "correct_hidden_layout_proven": False,
        "proof_limitation": PROOF_LIMITATION,
        "file_count": snapshot.file_count,
        "output_dir": str(output),
        "output_zip": str(archive),
        "attestation": str(attestation),
        "independent_validation_report": str(validation_report),
        "submission_zip_sha256": validation["submission_zip_sha256"],
        "promotion_config_sha256": promotion.config_sha256,
        "runtime_manifest_sha256": runtime_manifest["digest_sha256"],
        "canonical_device": CANONICAL_DEVICE,
        "score_seconds": score_seconds,
        "solve_seconds": solve_seconds,
        "restoration_seconds": restoration_seconds,
        "elapsed_seconds": perf_counter() - started,
        "independent_validation": validation,
    }


def dry_run_status() -> dict[str, Any]:
    """Read-only readiness check; never decodes any competition board."""

    promotion_exists = DEFAULT_PROMOTION_CONFIG.is_file()
    result: dict[str, Any] = {
        "status": (
            "READY_FOR_EXPLICIT_RUN"
            if promotion_exists
            else "BLOCKED_AWAITING_IMMUTABLE_PROMOTION_AUTHORIZATION"
        ),
        "production_executed": False,
        "output_root": str(OUTPUT_ROOT),
        "canonical_device": CANONICAL_DEVICE,
        "promotion_config": str(DEFAULT_PROMOTION_CONFIG),
        "promotion_config_exists": promotion_exists,
        "official_test_archive_sha256": OFFICIAL_TEST_ARCHIVE_SHA256,
        "official_filenames_sha256": OFFICIAL_FILENAMES_SHA256,
        "expected_file_count": EXPECTED_TEST_FILES,
    }
    if promotion_exists:
        result["promotion"] = load_promotion_evidence().as_dict()
    return result


__all__ = [
    "CANONICAL_DEVICE",
    "CHECKPOINT_SHA256",
    "CHECKPOINT_URL",
    "DEFAULT_ATTESTATION",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_OUTPUT_ZIP",
    "DEFAULT_VALIDATION_REPORT",
    "EXPECTED_POLICY",
    "METHOD_STATUS",
    "OUTPUT_ROOT",
    "PROOF_LIMITATION",
    "PROOF_SCOPE",
    "PROMOTED_ARM",
    "ProtectedSubmissionPrediction",
    "PromotionEvidence",
    "RESTORATION_NAME",
    "SCHEMA_NAME",
    "board_attestation",
    "build_attestation",
    "build_runtime_manifest",
    "dry_run_status",
    "frozen_pipeline_record",
    "load_and_validate_attestation",
    "load_promotion_evidence",
    "predict_drunet_protected",
    "run_production_submission",
    "tile_multiset_sha256",
]

"""Fail-closed production scaffold for the fixed-B standard DRUNet50 arm.

The module is intentionally parallel to both existing production paths.  It
cannot inspect competition-test inputs until a read-only root authorization
binds passing calibration-700 and unchanged holdout-700 evidence, including
numeric, provenance, safety, flatness, and manual-review gates.
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
from contextlib import suppress
from dataclasses import dataclass
from importlib.metadata import version as distribution_version
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
import scipy
import skimage
import sklearn
import torch
from jsonschema import Draft202012Validator
from PIL import __version__ as PILLOW_VERSION

from aiijc_puzzle.compliant_atlas_decoder import audit_raw_permutation
from aiijc_puzzle.compliant_submission import (
    METHOD_STATUS,
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
from aiijc_puzzle.drunet_goal_cycle2 import DIRECT_SIGMA, MODEL_BATCH_SIZE
from aiijc_puzzle.drunet_sigma50_protected_broad import render_sigma50_protected
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
from aiijc_puzzle.pretrained_tile_denoiser import load_drunet_color
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
OUTPUT_ROOT = PROJECT_ROOT / "outputs/compliant-fixed-b-standard-submission-v1"
DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / "predictions"
DEFAULT_OUTPUT_ZIP = OUTPUT_ROOT / "submission.zip"
DEFAULT_ATTESTATION = OUTPUT_ROOT / "compliance-attestation.json"
DEFAULT_VALIDATION_REPORT = OUTPUT_ROOT / "independent-validation.json"
DEFAULT_PROMOTION_CONFIG = PROJECT_ROOT / "configs/compliant_fixed_b_standard_submission_v1.json"
DEFAULT_SCHEMA_PATH = PROJECT_ROOT / "configs/compliant-fixed-b-standard-submission-v1.schema.json"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "artifacts/pretrained-denoisers/kair-fc1732f/drunet_color.pth"
RUNTIME_PREFLIGHT_RELATIVE_PATH = (
    "outputs/compliant-fixed-b-standard-preflight-v1/runtime-manifest.json"
)
DEFAULT_RUNTIME_PREFLIGHT = PROJECT_ROOT / RUNTIME_PREFLIGHT_RELATIVE_PATH

SCHEMA_NAME = "aiijc-puzzle-fixed-b-standard-submission-compliance-v1"
PROOF_SCOPE = "provenance_bijection_geometry_fixed_b_standard_restoration_only"
PROOF_LIMITATION = (
    "PASS proves corresponding-input provenance, an upright 20x20 strict bijection, "
    "exact raw reassembly, the frozen bilateral buddies96 solver, independent-tile "
    "official DRUNet50 and the fixed t60 protected h28/h50 tail only; it does not "
    "prove the hidden ground-truth permutation, reconstruction accuracy, or manual "
    "acceptance of the competition-test scenes."
)
# These identifiers match the immutable broad-measurement config and commitment
# exactly. Similar-looking later aliases must fail closed.
PROMOTED_ARM = "B_drunet50_protected_h28_h50_t60"
SAFETY_REFERENCE = "R_drunet50_h28_safety_reference"
RESTORATION_NAME = (
    "rgb-seam-offsets_then-bounded-luminance-gains_then-independent-tile-official-"
    "color-DRUNet-sigma50-then-independent-colored-NLM-h20-h28-h50-then-t60-"
    "protected-h28-flat-h50-blend"
)
EDGE_BUDGET = 96
DRUNET_SIGMA_255 = float(DIRECT_SIGMA)
DRUNET_BATCH_SIZE = int(MODEL_BATCH_SIZE)
SOBEL_THRESHOLD = 60.0
NLM_STRENGTHS = (20, 28, 50)
NLM_TEMPLATE_WINDOW = 7
NLM_SEARCH_WINDOW = 21
CANONICAL_DEVICE = "mps"
CHECKPOINT_URL = "https://github.com/cszn/KAIR/releases/download/v1.0/drunet_color.pth"
CHECKPOINT_SHA256 = "479abe3c5327dfd10ff54a80ec7d4098ca80752a5c9492cdff31cee430bec4b4"
KAIR_COMMIT = "fc1732f4a4514e42ce15e5b3a1e18c828af47a1e"
MODEL_PARAMETER_COUNT = 32_640_960
MEASUREMENT_CONFIG_PATH = "configs/drunet_sigma50_protected_all700_measurement_v1.json"
MEASUREMENT_CONFIG_SHA256 = "a402fc682b0db96b60004fa2c33ea70baf06035cb4971b8ee0778ceb1b7f05ac"
# Immutable binding for the parallel fixed-B attestation schema.
PINNED_SCHEMA_SHA256 = "cd529f425b29150594d8d73ce735dc4872d21cbd69165565c684918d4df688c3"

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
# Exact maps copied byte-for-byte from the immutable calibration700 measurement
# config and prediction commitment.  Production must refuse if the config,
# either stage commitment, or the current broad implementation/assets differ.
EXPECTED_BROAD_SOURCE_SHA256 = {
    "scripts/run_drunet_sigma50_protected_all700.py": (
        "64563820beec56fccf00fc11f899411c175274d46c50a124154ca37bd857bde6"
    ),
    "src/aiijc_puzzle/compliant_atlas_decoder.py": (
        "86704849112b66fa2355c1a07c89c409f0f77ea48e175099ae54337f74a0196d"
    ),
    "src/aiijc_puzzle/drunet_goal_cycle2.py": (
        "b49913044fcec261b29d1f788f1fb48e1e5f972aba6706942543c017b8462838"
    ),
    "src/aiijc_puzzle/drunet_sigma50_protected_broad.py": (
        "b0b73fb30787394f839a8449a7e7c898e92269a65cef613b5152c90b53db5a9f"
    ),
    "src/aiijc_puzzle/edge_protected_nlm.py": (
        "e817068e5ef9ffa1cce82a09cc5d8c7763adf0ff680cacfde7b4b1224ff0259f"
    ),
    "src/aiijc_puzzle/legacy_upgrade.py": (
        "3908f7c192a0f9b43e288b17f7614b66456b2bf20e4539e09aa9f5c5ae2a4586"
    ),
    "src/aiijc_puzzle/nlm_luma_chroma.py": (
        "6d743edfedaf287e49944e56fcb094714214890b9623ceb7036e98a66ec7dbd3"
    ),
    "src/aiijc_puzzle/postassembly_harmonizer.py": (
        "4f7a44ab0781d6da8cee88ed956c34e92d756417162855ee4b1cd84b017dcb41"
    ),
    "src/aiijc_puzzle/pretrained_tile_denoiser.py": (
        "9b79f27d6b8570d65f64e21f703e20bb880ee9d000adb7675cb89c673f4a196e"
    ),
    "src/aiijc_puzzle/protocol.py": (
        "c97fcb5e6eb07abe6a91480c525f63b72a73bfb0aec7df4d51971e82b2f481be"
    ),
}
EXPECTED_BROAD_ASSET_SHA256 = {
    "LICENSE": "448e69b705d64f21bf8cb86562301e0edd99ac79026064ddd75af8242b067be5",
    "drunet_color.pth": CHECKPOINT_SHA256,
    "models/basicblock.py": (
        "48406db8867394ac5ae233ebeec7711ac10acfc3a6bbf0072c33aa77d659b6fd"
    ),
    "models/network_unet.py": (
        "8043b6350f1589d5f08892e3be0b4d12c5a502058014285107b7360696d12bf5"
    ),
}
KAIR_ASSET_ROOT_RELATIVE = "artifacts/pretrained-denoisers/kair-fc1732f"
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
    "constant_or_near_flat_tile_substitution_used": False,
    "tile_substitution_used": False,
    "rotation_or_flip_used": False,
    "resize_or_warp_used": False,
    "filename_or_board_overrides_used": False,
}
EXPECTED_BROAD_SAFETY_CHECK_NAMES = frozenset(
    {
        "all_700_strict_raw_provenance_audits_pass",
        "candidate_pixel_distinct_from_reference_on_every_board",
        "chroma_gradient_mean_at_least_0_80",
        "chroma_gradient_min_at_least_0_70",
        "clipping_increase_at_most_0_01",
        "grid_ratio_max_at_most_1_12",
        "grid_ratio_mean_at_most_1_05",
        "laplacian_mean_at_least_0_72",
        "laplacian_min_at_least_0_60",
        "luma_gradient_mean_at_least_0_80",
        "luma_gradient_min_at_least_0_70",
        "near_flat_std_lt_2_maximum_board_increase_at_most_6_tiles",
        "near_flat_std_lt_2_mean_increase_at_most_2_tiles",
        "no_total_exact_constant_tile_increase",
        "protected_fraction_every_board_between_0_30_and_0_85",
        "protected_fraction_mean_between_0_40_and_0_75",
    }
)

EVIDENCE_NAMES = (
    "measurement_config",
    "production_runtime_manifest",
    "calibration_commitment",
    "calibration_commitment_receipt",
    "calibration_target_access_receipt",
    "calibration_report",
    "calibration_manual_review",
    "holdout_commitment",
    "holdout_commitment_receipt",
    "holdout_target_access_receipt",
    "holdout_report",
    "holdout_manual_review",
)

RUNTIME_FILE_RELATIVE_PATHS = (
    "src/aiijc_puzzle/__init__.py",
    "src/aiijc_puzzle/candidate_supply.py",
    "src/aiijc_puzzle/compliant_fixed_b_standard_submission.py",
    "src/aiijc_puzzle/compliant_fixed_b_standard_validation.py",
    "src/aiijc_puzzle/compliant_submission.py",
    "src/aiijc_puzzle/compliant_atlas_decoder.py",
    "src/aiijc_puzzle/drunet_goal_cycle2.py",
    "src/aiijc_puzzle/drunet_sigma50_protected_broad.py",
    "src/aiijc_puzzle/legacy_upgrade.py",
    "src/aiijc_puzzle/nlm_luma_chroma.py",
    "src/aiijc_puzzle/novel_analog_layout.py",
    "src/aiijc_puzzle/pixel_tails.py",
    "src/aiijc_puzzle/postassembly_harmonizer.py",
    "src/aiijc_puzzle/pretrained_drunet_protected_stack.py",
    "src/aiijc_puzzle/protocol.py",
    "src/aiijc_puzzle/pretrained_tile_denoiser.py",
    "src/aiijc_puzzle/edge_protected_nlm.py",
    "scripts/run_compliant_fixed_b_standard_submission.py",
    "scripts/validate_compliant_fixed_b_standard_submission.py",
    "configs/compliant-fixed-b-standard-submission-v1.schema.json",
    "configs/postassembly_rgb_offset_v1.json",
    "configs/postassembly_luminance_gain_v1.json",
    MEASUREMENT_CONFIG_PATH,
    "uv.lock",
)


@dataclass(frozen=True)
class PromotionEvidence:
    """The root authorization and exact future evidence paths/hashes it binds."""

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
class FixedBSubmissionPrediction:
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
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _require_readonly(path: Path) -> None:
    if path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise PermissionError(f"integrity-bound artifact is writable: {path}")


def _load_json_object(path: Path) -> dict[str, Any]:
    checked = _require_regular_file(path)
    payload = json.loads(checked.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {checked}")
    return payload


def verify_broad_measurement_integrity(
    *, project_root: Path = PROJECT_ROOT
) -> dict[str, dict[str, str]]:
    """Hash the current frozen broad sources/assets against calibration700."""

    observed_sources = {
        relative: sha256_file(_require_regular_file(project_root / relative))
        for relative in EXPECTED_BROAD_SOURCE_SHA256
    }
    if observed_sources != EXPECTED_BROAD_SOURCE_SHA256:
        changed = sorted(
            relative
            for relative, expected in EXPECTED_BROAD_SOURCE_SHA256.items()
            if observed_sources.get(relative) != expected
        )
        raise ValueError(f"current frozen broad source hash drift: {changed}")
    asset_root = project_root / KAIR_ASSET_ROOT_RELATIVE
    observed_assets = {
        relative: sha256_file(_require_regular_file(asset_root / relative))
        for relative in EXPECTED_BROAD_ASSET_SHA256
    }
    if observed_assets != EXPECTED_BROAD_ASSET_SHA256:
        changed = sorted(
            relative
            for relative, expected in EXPECTED_BROAD_ASSET_SHA256.items()
            if observed_assets.get(relative) != expected
        )
        raise ValueError(f"current frozen broad asset hash drift: {changed}")
    return {"source_sha256": observed_sources, "asset_sha256": observed_assets}


def _safe_evidence_path(project_root: Path, relative: str, name: str) -> Path:
    candidate = Path(relative)
    if (
        candidate.is_absolute()
        or not candidate.parts
        or ".." in candidate.parts
        or candidate.suffix != ".json"
    ):
        raise ValueError(f"unsafe promotion evidence path: {name}")
    if name == "measurement_config":
        if relative != MEASUREMENT_CONFIG_PATH:
            raise ValueError("measurement config path changed")
    elif name == "production_runtime_manifest":
        if relative != RUNTIME_PREFLIGHT_RELATIVE_PATH:
            raise ValueError("production runtime preflight path changed")
    elif candidate.parts[0] != "outputs":
        raise ValueError(f"future promotion evidence must be under outputs/: {name}")
    root = _absolute_without_resolving(project_root)
    unresolved = _absolute_without_resolving(root / candidate)
    resolved_root = root.resolve(strict=False)
    resolved = unresolved.resolve(strict=False)
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"promotion evidence escapes project root: {name}")
    # Return the unresolved absolute spelling.  `_require_regular_file` then
    # sees and rejects a symlink leaf or any symlink ancestor instead of being
    # handed an already-resolved alias.
    return unresolved


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
            "source": "DRUNet50 then independent h20",
            "sobel_threshold": SOBEL_THRESHOLD,
            "all_20px_grid_boundaries_protected": True,
            "dilation": "3x3 one iteration",
            "softening": "Gaussian sigma1",
        },
        "blend": "rint(soft*h28 + (1-soft)*h50), clip uint8",
        "constant_or_near_flat_tile_substitution": False,
    }


def _validate_commitment(
    commitment: Mapping[str, Any],
    *,
    stage: str,
    measurement_sha256: str,
) -> None:
    if commitment.get("schema") != "aiijc-drunet-sigma50-protected-all700-commitment-v1":
        raise ValueError(f"{stage} commitment schema changed")
    if commitment.get("stage") != stage or commitment.get("count") != 700:
        raise ValueError(f"{stage} commitment panel changed")
    if commitment.get("config_sha256") != measurement_sha256:
        raise ValueError(f"{stage} commitment measurement binding changed")
    if commitment.get("fixed_candidate") != PROMOTED_ARM:
        raise ValueError(f"{stage} commitment candidate changed")
    if commitment.get("target_free_safety_reference_only") != SAFETY_REFERENCE:
        raise ValueError(f"{stage} safety reference changed")
    if commitment.get("all_700_strict_raw_permutation_audits_pass") is not True:
        raise ValueError(f"{stage} strict provenance did not pass")
    safety = commitment.get("target_free_safety", {})
    if not isinstance(safety, Mapping) or safety.get("passed") is not True:
        raise ValueError(f"{stage} safety/flatness commitment gate did not pass")
    checks = safety.get("checks")
    if (
        not isinstance(checks, Mapping)
        or set(checks) != EXPECTED_BROAD_SAFETY_CHECK_NAMES
        or not all(value is True for value in checks.values())
    ):
        raise ValueError(f"{stage} commitment safety check roster did not fully pass")
    if commitment.get("source_sha256") != EXPECTED_BROAD_SOURCE_SHA256:
        raise ValueError(f"{stage} commitment frozen broad source map changed")
    if commitment.get("asset_sha256") != EXPECTED_BROAD_ASSET_SHA256:
        raise ValueError(f"{stage} commitment frozen broad asset map changed")
    model = commitment.get("model", {})
    if (
        model.get("sigma_255") != DRUNET_SIGMA_255
        or model.get("batch_size") != DRUNET_BATCH_SIZE
        or model.get("parameter_count") != MODEL_PARAMETER_COUNT
    ):
        raise ValueError(f"{stage} model contract changed")
    roster_sha256 = commitment.get("candidate_roster_sha256")
    if (
        not isinstance(roster_sha256, str)
        or len(roster_sha256) != 64
        or any(character not in "0123456789abcdef" for character in roster_sha256)
    ):
        raise ValueError(f"{stage} commitment candidate roster digest changed")
    boards = commitment.get("boards")
    if not isinstance(boards, list) or len(boards) != 700:
        raise ValueError(f"{stage} commitment board roster changed")
    filenames: list[str] = []
    for board in boards:
        if not isinstance(board, Mapping):
            raise ValueError(f"{stage} commitment board record changed")
        filename = board.get("filename")
        if (
            not isinstance(filename, str)
            or not filename.endswith(".png")
            or "/" in filename
            or "\\" in filename
            or board.get("raw_permutation_audit_passed") is not True
        ):
            raise ValueError(f"{stage} commitment board provenance changed")
        for digest_name in ("layout_sha256", "candidate_pixel_sha256"):
            digest = board.get(digest_name)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"{stage} commitment board {digest_name} changed")
        filenames.append(filename)
    if len(set(filenames)) != 700:
        raise ValueError(f"{stage} commitment filenames are not unique")


def _validate_report(
    report: Mapping[str, Any],
    *,
    commitment: Mapping[str, Any],
    stage: str,
    measurement_sha256: str,
    commitment_sha256: str,
    receipt_sha256: str,
    target_access_sha256: str,
) -> None:
    if report.get("schema") != "aiijc-drunet-sigma50-protected-all700-report-v1":
        raise ValueError(f"{stage} report schema changed")
    if report.get("status") != "exact_all700_measurement_from_precommitted_predictions":
        raise ValueError(f"{stage} report is incomplete")
    if report.get("stage") != stage or report.get("count") != 700:
        raise ValueError(f"{stage} report panel changed")
    if report.get("fixed_candidate") != PROMOTED_ARM:
        raise ValueError(f"{stage} report candidate changed")
    mean = report.get("mean_ssim")
    if isinstance(mean, bool) or not isinstance(mean, (int, float)) or not 0.27 <= mean <= 0.28:
        raise ValueError(f"{stage} mean SSIM is outside [0.27,0.28]")
    if report.get("config_sha256") != measurement_sha256:
        raise ValueError(f"{stage} report measurement binding changed")
    if report.get("commitment_sha256") != commitment_sha256:
        raise ValueError(f"{stage} report commitment binding changed")
    if report.get("commitment_receipt_sha256") != receipt_sha256:
        raise ValueError(f"{stage} report receipt binding changed")
    if report.get("target_access_receipt_sha256") != target_access_sha256:
        raise ValueError(f"{stage} report target-access binding changed")
    if report.get("candidate_roster_sha256") != commitment.get("candidate_roster_sha256"):
        raise ValueError(f"{stage} report candidate roster binding changed")
    committed_boards = commitment["boards"]
    rows = report.get("rows")
    if not isinstance(rows, list) or len(rows) != 700:
        raise ValueError(f"{stage} report row roster changed")
    scores: list[float] = []
    filenames: list[str] = []
    for index, (row, board) in enumerate(zip(rows, committed_boards, strict=True)):
        if not isinstance(row, Mapping):
            raise ValueError(f"{stage} report row changed")
        expected = {
            "filename": board["filename"],
            "layout_sha256": board["layout_sha256"],
            "candidate_pixel_sha256": board["candidate_pixel_sha256"],
        }
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError(f"{stage} report row/commitment mismatch at index {index}")
        score = row.get("ssim")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not np.isfinite(score)
            or not 0.0 <= score <= 1.0
        ):
            raise ValueError(f"{stage} report SSIM is invalid at index {index}")
        scores.append(float(score))
        filenames.append(str(row["filename"]))
    if len(set(filenames)) != 700:
        raise ValueError(f"{stage} report filenames are not unique")
    recomputed_mean = float(np.asarray(scores, dtype=np.float64).mean())
    if not np.isclose(float(mean), recomputed_mean, rtol=0.0, atol=1e-15):
        raise ValueError(f"{stage} report mean does not match its 700 committed rows")
    if report.get("strict_provenance", {}).get("all_700_pass") is not True:
        raise ValueError(f"{stage} report provenance gate did not pass")
    if report.get("broad_completion_gate", {}).get("passed") is not True:
        raise ValueError(f"{stage} broad completion gate did not pass")
    safety = report.get("target_free_safety")
    if safety != commitment.get("target_free_safety"):
        raise ValueError(f"{stage} report safety differs from committed safety")
    if report.get("competition_test_access") is not False:
        raise ValueError(f"{stage} report declares competition-test access")


def _validate_manual_review(
    review: Mapping[str, Any],
    *,
    stage: str,
    report_sha256: str,
    commitment_sha256: str,
) -> None:
    expected = {
        "schema": "aiijc-fixed-b-standard-all700-manual-review-v1",
        "reviewer": "root",
        "stage": stage,
        "reviewed_arm": PROMOTED_ARM,
        "reviewed_board_count": 700,
        "reviewed_all_700_outputs": True,
        "severe_artifacts": 0,
        "material_face_text_or_object_loss": False,
        "mask_halo_or_boundary_damage": False,
        "constant_or_near_flat_tile_substitution": False,
        "passed": True,
        "report_sha256": report_sha256,
        "commitment_sha256": commitment_sha256,
    }
    for key, value in expected.items():
        if review.get(key) != value:
            raise ValueError(f"{stage} manual review binding failed: {key}")


def _validate_production_runtime_preflight(
    manifest: Mapping[str, Any],
    *,
    expected_digest: str,
) -> None:
    expected_keys = {
        "files",
        "assets",
        "harmonizers",
        "versions",
        "host",
        "canonical_device",
        "digest_sha256",
    }
    if set(manifest) != expected_keys:
        raise ValueError("production runtime preflight keys changed")
    content = {key: manifest[key] for key in manifest if key != "digest_sha256"}
    internal_digest = _canonical_json_sha256(content)
    if manifest.get("digest_sha256") != internal_digest or expected_digest != internal_digest:
        raise ValueError("production runtime preflight internal digest changed")
    if manifest.get("host", {}).get("mps_available") is not True:
        raise ValueError("production runtime preflight requires an available MPS backend")
    current = build_runtime_manifest()
    if manifest != current:
        changed = sorted(
            key for key in expected_keys if manifest.get(key) != current.get(key)
        )
        raise ValueError(f"production runtime differs from frozen preflight: {changed}")


def load_promotion_evidence(
    config_path: Path = DEFAULT_PROMOTION_CONFIG,
    *,
    project_root: Path = PROJECT_ROOT,
) -> PromotionEvidence:
    """Validate explicit root authorization before any test input is inspected."""

    if not config_path.is_file():
        raise FileNotFoundError(
            "BLOCKED_AWAITING_FIXED_B_CALIBRATION700_HOLDOUT700_AND_MANUAL_AUTHORIZATION"
        )
    checked_config = _require_regular_file(config_path)
    _require_readonly(checked_config)
    config = _load_json_object(checked_config)
    expected_keys = {
        "schema",
        "status",
        "production_authorized_by_root",
        "canonical_device",
        "promoted_arm",
        "pipeline",
        "evidence",
    }
    if set(config) != expected_keys:
        raise ValueError("promotion authorization keys changed")
    if config["schema"] != "aiijc-fixed-b-standard-production-authorization-v1":
        raise ValueError("promotion authorization schema changed")
    if config["status"] != (
        "CALIBRATION700_AND_UNCHANGED_HOLDOUT700_NUMERIC_PROVENANCE_SAFETY_FLATNESS_MANUAL_PASS"
    ):
        raise ValueError("promotion authorization does not declare every required gate")
    if config["production_authorized_by_root"] is not True:
        raise ValueError("root did not authorize production")
    if config["canonical_device"] != CANONICAL_DEVICE:
        raise ValueError("canonical device changed")
    if config["promoted_arm"] != PROMOTED_ARM:
        raise ValueError("promoted arm changed")
    if config["pipeline"] != frozen_pipeline_record():
        raise ValueError("promotion pipeline changed")
    evidence = config["evidence"]
    if not isinstance(evidence, dict) or set(evidence) != set(EVIDENCE_NAMES):
        raise ValueError("promotion evidence roster changed")

    verified: dict[str, dict[str, str]] = {}
    loaded: dict[str, dict[str, Any]] = {}
    observed_paths: set[Path] = set()
    for name in EVIDENCE_NAMES:
        record = evidence[name]
        expected_record_keys = (
            {"path", "sha256", "digest_sha256"}
            if name == "production_runtime_manifest"
            else {"path", "sha256"}
        )
        if not isinstance(record, dict) or set(record) != expected_record_keys:
            raise ValueError(f"invalid evidence record: {name}")
        path = _safe_evidence_path(project_root, record["path"], name)
        if path in observed_paths:
            raise ValueError("promotion evidence paths must be distinct")
        observed_paths.add(path)
        checked = _require_regular_file(path)
        _require_readonly(checked)
        actual = sha256_file(checked)
        if record["sha256"] != actual:
            raise ValueError(f"promotion evidence hash changed: {name}")
        verified[name] = {"path": record["path"], "sha256": actual}
        if name == "production_runtime_manifest":
            verified[name]["digest_sha256"] = record["digest_sha256"]
        loaded[name] = _load_json_object(checked)

    # This scaffold intentionally predates the two future passing reports.  The
    # root authorization must provide the already-frozen measurement protocol
    # plus the exact future stage artifacts it eventually produced.
    measurement_sha256 = verified["measurement_config"]["sha256"]
    if measurement_sha256 != MEASUREMENT_CONFIG_SHA256:
        raise ValueError("fixed-B measurement config hash changed")
    measurement = loaded["measurement_config"]
    if (
        measurement.get("schema") != "aiijc-drunet-sigma50-protected-all700-measurement-v1"
        or measurement.get("fixed_pipeline", {}).get("single_candidate_only") != PROMOTED_ARM
        or measurement.get("fixed_pipeline", {}).get("target_free_safety_reference_only")
        != SAFETY_REFERENCE
    ):
        raise ValueError("fixed-B measurement method changed")
    if measurement.get("source_sha256") != EXPECTED_BROAD_SOURCE_SHA256:
        raise ValueError("fixed-B measurement frozen broad source map changed")
    if measurement.get("asset_sha256") != EXPECTED_BROAD_ASSET_SHA256:
        raise ValueError("fixed-B measurement frozen broad asset map changed")
    verify_broad_measurement_integrity()
    _validate_production_runtime_preflight(
        loaded["production_runtime_manifest"],
        expected_digest=verified["production_runtime_manifest"]["digest_sha256"],
    )

    for stage in ("calibration", "holdout"):
        commitment_name = f"{stage}_commitment"
        receipt_name = f"{stage}_commitment_receipt"
        target_name = f"{stage}_target_access_receipt"
        report_name = f"{stage}_report"
        review_name = f"{stage}_manual_review"
        commitment_sha = verified[commitment_name]["sha256"]
        receipt_sha = verified[receipt_name]["sha256"]
        target_sha = verified[target_name]["sha256"]
        report_sha = verified[report_name]["sha256"]
        commitment = loaded[commitment_name]
        _validate_commitment(
            commitment,
            stage=stage,
            measurement_sha256=measurement_sha256,
        )
        receipt = loaded[receipt_name]
        if (
            receipt.get("schema") != "aiijc-drunet-sigma50-protected-all700-receipt-v1"
            or receipt.get("status")
            != "commitment_created_before_any_target_decode_in_this_measurement_stage"
            or receipt.get("stage") != stage
            or receipt.get("count") != 700
            or receipt.get("config_sha256") != measurement_sha256
            or receipt.get("commitment_sha256") != commitment_sha
            or receipt.get("candidate_roster_sha256")
            != commitment.get("candidate_roster_sha256")
            or receipt.get("targets_decoded_before_receipt") is not False
            or receipt.get("competition_test_access") is not False
        ):
            raise ValueError(f"{stage} commitment receipt binding failed")
        target_access = loaded[target_name]
        if (
            target_access.get("schema")
            != "aiijc-drunet-sigma50-protected-all700-target-access-v1"
            or target_access.get("status")
            != "written_after_full_prediction_verification_and_immediately_before_target_decode"
            or target_access.get("stage") != stage
            or target_access.get("count") != 700
            or target_access.get("config_sha256") != measurement_sha256
            or target_access.get("commitment_sha256") != commitment_sha
            or target_access.get("commitment_receipt_sha256") != receipt_sha
            or target_access.get("candidate_roster_sha256")
            != commitment.get("candidate_roster_sha256")
            or target_access.get("predictions_were_committed_before_current_target_decode")
            is not True
            or target_access.get("historical_workspace_target_exposure_acknowledged")
            is not True
            or target_access.get("freshness_claim") is not False
        ):
            raise ValueError(f"{stage} target-access receipt binding failed")
        _validate_report(
            loaded[report_name],
            commitment=commitment,
            stage=stage,
            measurement_sha256=measurement_sha256,
            commitment_sha256=commitment_sha,
            receipt_sha256=receipt_sha,
            target_access_sha256=target_sha,
        )
        _validate_manual_review(
            loaded[review_name],
            stage=stage,
            report_sha256=report_sha,
            commitment_sha256=commitment_sha,
        )

    calibration = loaded["calibration_commitment"]
    holdout = loaded["holdout_commitment"]
    for key in ("source_sha256", "asset_sha256", "model", "fixed_candidate"):
        if calibration.get(key) != holdout.get(key):
            raise ValueError(f"holdout is not unchanged from calibration: {key}")

    return PromotionEvidence(
        config_sha256=sha256_file(checked_config),
        artifacts=verified,
    )


def verify_official_assets() -> dict[str, str]:
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
    value = np.asarray(image)
    if value.shape != (IMAGE_SIZE, IMAGE_SIZE, RGB_CHANNELS) or value.dtype != np.uint8:
        raise ValueError("tile multiset requires uint8 RGB 480x480")
    hashes = sorted(hashlib.sha256(tile.tobytes()).digest() for tile in split_tiles(value))
    return hashlib.sha256(b"".join(hashes)).hexdigest()


def _apply_frozen_harmonizers(raw: np.ndarray) -> np.ndarray:
    ordered = split_tiles(raw)
    offsets, _ = seam_graph_rgb_offsets(ordered, DEFAULT_SEAM_GRAPH_CONFIG)
    rgb = apply_rgb_offsets(ordered, offsets)
    gains, _ = seam_graph_luminance_gains(rgb, DEFAULT_LUMINANCE_GAIN_CONFIG)
    return assemble_tiles(apply_luminance_gains(rgb, gains))


def predict_fixed_b_standard(
    input_image: np.ndarray,
    model: torch.nn.Module,
    *,
    device: torch.device,
) -> FixedBSubmissionPrediction:
    """Run the one fixed legal method on one corresponding dirty board."""

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
    audit_object = audit_raw_permutation(value, raw, layout, restoration_applied_after_audit=True)
    if not audit_object.passed:
        raise RuntimeError("strict raw permutation audit failed")
    input_multiset = tile_multiset_sha256(value)
    if tile_multiset_sha256(raw) != input_multiset:
        raise RuntimeError("independent raw tile multiset audit failed")

    restoration_started = perf_counter()
    harmonized = _apply_frozen_harmonizers(raw)
    reference, restored, diagnostics = render_sigma50_protected(
        model, split_tiles(harmonized), device=device
    )
    restoration_seconds = perf_counter() - restoration_started
    intermediates = diagnostics["neural_intermediate_pixel_sha256"]
    restoration = {
        "harmonized_array_sha256": array_sha256(harmonized),
        "safety_reference_h28_array_sha256": array_sha256(reference),
        "drunet50_canvas_pixel_sha256": intermediates["drunet50_canvas"],
        "nlm_h20_pixel_sha256": intermediates["drunet50_then_h20_mask_source"],
        "nlm_h28_pixel_sha256": intermediates["drunet50_then_h28_reference_and_safe"],
        "nlm_h50_pixel_sha256": intermediates["drunet50_then_h50_flat"],
        "binary_mask_array_sha256": diagnostics["mask"]["binary_mask_sha256"],
        "soft_mask_array_sha256": diagnostics["mask"]["soft_mask_sha256"],
        "protected_fraction": diagnostics["mask"]["binary_dilated_protected_fraction"],
        "drunet_diagnostics": diagnostics["drunet"],
        "output_array_sha256": array_sha256(restored),
    }
    return FixedBSubmissionPrediction(
        layout=layout,
        raw=raw,
        harmonized=harmonized,
        restored=restored,
        audit=json.loads(json.dumps(audit_object.as_dict(), sort_keys=True)),
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
    mac_release, mac_version_info, mac_machine = platform.mac_ver()
    content: dict[str, Any] = {
        "files": files,
        "assets": verify_official_assets(),
        "harmonizers": verify_harmonizer_configs(),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "torch": torch.__version__,
            "scipy": scipy.__version__,
            "pillow": PILLOW_VERSION,
            "scikit_image": skimage.__version__,
            "scikit_learn": sklearn.__version__,
            "jsonschema": distribution_version("jsonschema"),
        },
        "host": {
            "platform": platform.platform(),
            "mac_ver": [mac_release, list(mac_version_info), mac_machine],
            "machine": platform.machine(),
            "mps_available": bool(torch.backends.mps.is_available()),
        },
        "canonical_device": CANONICAL_DEVICE,
    }
    return {**content, "digest_sha256": _canonical_json_sha256(content)}


def freeze_production_runtime_preflight(*, path: Path | None = None) -> dict[str, Any]:
    """Write the final source/environment manifest once, before authorization."""

    if DEFAULT_PROMOTION_CONFIG.exists() or DEFAULT_PROMOTION_CONFIG.is_symlink():
        raise FileExistsError("runtime preflight must be frozen before promotion authorization")
    destination = _absolute_without_resolving(path or DEFAULT_RUNTIME_PREFLIGHT)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite runtime preflight: {destination}")
    manifest = build_runtime_manifest()
    if manifest["host"]["mps_available"] is not True:
        raise RuntimeError("canonical fixed-B runtime requires an available MPS backend")
    _mkdir_without_symlink_ancestors(destination.parent)
    atomic_write_json(destination, manifest)
    destination.chmod(0o444)
    checked = _require_regular_file(destination)
    _require_readonly(checked)
    observed = _load_json_object(checked)
    _validate_production_runtime_preflight(
        observed,
        expected_digest=manifest["digest_sha256"],
    )
    return {
        "status": "IMMUTABLE_PRODUCTION_RUNTIME_PREFLIGHT_FROZEN_BEFORE_AUTHORIZATION",
        "path": str(destination),
        "relative_path": (
            str(destination.relative_to(PROJECT_ROOT))
            if destination.is_relative_to(PROJECT_ROOT)
            else None
        ),
        "sha256": sha256_file(destination),
        "digest_sha256": manifest["digest_sha256"],
        "read_only": True,
        "competition_test_access": False,
    }


def _restoration_attestation(prediction: FixedBSubmissionPrediction) -> dict[str, Any]:
    return {
        "name": RESTORATION_NAME,
        "input_is_raw_assembly": True,
        "pixel_restoration_only": True,
        "layout_changed": False,
        "spatial_warp_used": False,
        "constant_or_near_flat_tile_substitution_used": False,
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
            "restored_canvas_pixel_sha256": prediction.restoration["drunet50_canvas_pixel_sha256"],
            "runtime_diagnostics": prediction.restoration["drunet_diagnostics"],
        },
        "nlm": {
            "proper_rgb_bgr_roundtrip": True,
            "independent_single_pass_strengths": list(NLM_STRENGTHS),
            "h_color_equals_h": True,
            "template_window_size": NLM_TEMPLATE_WINDOW,
            "search_window_size": NLM_SEARCH_WINDOW,
            "h20_pixel_sha256": prediction.restoration["nlm_h20_pixel_sha256"],
            "h28_pixel_sha256": prediction.restoration["nlm_h28_pixel_sha256"],
            "h50_pixel_sha256": prediction.restoration["nlm_h50_pixel_sha256"],
        },
        "protected_mask": {
            "source": "independent DRUNet50+h20",
            "sobel_threshold": SOBEL_THRESHOLD,
            "grid_period": TILE_SIZE,
            "dilation": "3x3 one iteration",
            "softening": "Gaussian sigma1",
            "binary_array_sha256": prediction.restoration["binary_mask_array_sha256"],
            "soft_array_sha256": prediction.restoration["soft_mask_array_sha256"],
            "protected_fraction": prediction.restoration["protected_fraction"],
        },
        "blend": "rint(soft*h28 + (1-soft)*h50), clip uint8",
        "output_array_sha256": prediction.restoration["output_array_sha256"],
    }


def board_attestation(
    *,
    filename: str,
    input_sha256: str,
    prediction: FixedBSubmissionPrediction,
    output_png_sha256: str,
) -> dict[str, Any]:
    layout = np.asarray(prediction.layout, dtype=np.int32)
    if layout.shape != (TILE_COUNT,) or not np.array_equal(
        np.sort(layout), np.arange(TILE_COUNT, dtype=np.int32)
    ):
        raise ValueError("layout is not a strict 0..575 permutation")
    if prediction.audit.get("passed") is not True:
        raise ValueError("board attestation requires a passed raw audit")
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


def _method_declaration(promotion: PromotionEvidence, runtime_manifest: Mapping[str, Any]) -> str:
    return (
        "corresponding-input-only; upright-20x20-tiles; bilateral-directional-scores; "
        "strict-buddies96-permutation; raw-pixel-and-tile-multiset-audit; "
        "rgb-luma-harmonize; official-independent-tile-DRUNet-sigma50; "
        "independent-NLM-h20-h28-h50; exact-t60-protected-h28-flat-h50; "
        "no-constant-or-near-flat-tile-substitution; "
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
        raise ValueError("per-board evidence must follow official roster")
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
            f"fixed-B schema hash changed: expected {PINNED_SCHEMA_SHA256}, got {actual}"
        )
    schema = json.loads(checked.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    if schema.get("properties", {}).get("schema", {}).get("const") != SCHEMA_NAME:
        raise ValueError("fixed-B schema name changed")
    return schema


def load_and_validate_attestation(path: Path) -> dict[str, Any]:
    payload = _load_json_object(path)
    Draft202012Validator(_load_schema()).validate(payload)
    return payload


def _guard_validation_report(path: Path, artifacts: Sequence[Path]) -> Path:
    report = _absolute_without_resolving(path)
    if report.exists() or report.is_symlink():
        raise FileExistsError(f"refusing to overwrite validation report: {report}")
    resolved = report.resolve(strict=False)
    artifact_resolved = [artifact.resolve(strict=False) for artifact in artifacts]
    if any(
        resolved == artifact or resolved in artifact.parents or artifact in resolved.parents
        for artifact in artifact_resolved
    ):
        raise ValueError("validation report path overlaps an output artifact")
    _mkdir_without_symlink_ancestors(report.parent)
    return report


def _require_canonical_mps() -> torch.device:
    if not torch.backends.mps.is_available():
        raise RuntimeError("canonical fixed-B production requires Apple MPS")
    return torch.device("mps")


def _unused_temporary_path(*, prefix: str, parent: Path) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=prefix, dir=parent)
    os.close(descriptor)
    path = Path(name)
    path.unlink()
    return path


def _publish_validated_artifacts(pairs: Sequence[tuple[Path, Path]]) -> None:
    """Publish validated artifacts and roll back every destination on failure."""

    moved: list[Path] = []
    try:
        for source, destination in pairs:
            os.replace(source, destination)
            moved.append(destination)
    except BaseException:
        for destination in reversed(moved):
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink(missing_ok=True)
        raise


def dry_run_status() -> dict[str, Any]:
    """Report gate state without reading the competition-test directory/archive."""

    if not DEFAULT_PROMOTION_CONFIG.is_file():
        return {
            "status": "BLOCKED_AWAITING_FIXED_B_CALIBRATION700_HOLDOUT700_AND_MANUAL_AUTHORIZATION",
            "production_authorized": False,
            "promotion_config": str(DEFAULT_PROMOTION_CONFIG),
            "promotion_config_exists": False,
            "competition_test_access": False,
            "output_root": str(OUTPUT_ROOT),
            "output_root_exists": OUTPUT_ROOT.exists(),
            "existing_h20_and_drunet40_artifacts_modified": False,
        }
    promotion = load_promotion_evidence()
    return {
        "status": "AUTHORIZED_NOT_RUN",
        "production_authorized": True,
        "promotion_config": str(DEFAULT_PROMOTION_CONFIG),
        "promotion_config_sha256": promotion.config_sha256,
        "competition_test_access": False,
        "output_root": str(OUTPUT_ROOT),
        "output_root_exists": OUTPUT_ROOT.exists(),
        "existing_h20_and_drunet40_artifacts_modified": False,
    }


def run_production_submission(*, inputs_dir: Path, source_archive: Path) -> dict[str, Any]:
    """Generate and independently recompute all 700 outputs before publication."""

    from aiijc_puzzle.compliant_fixed_b_standard_validation import (
        validate_against_snapshot,
    )

    # Authorization and schema checks deliberately precede all test access.
    promotion = load_promotion_evidence()
    _load_schema()
    runtime_manifest = build_runtime_manifest()
    snapshot = build_official_input_snapshot(inputs_dir, source_archive)
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
        DEFAULT_VALIDATION_REPORT, (output, archive, attestation)
    )
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    descriptor, zip_name = tempfile.mkstemp(prefix=f".{archive.name}.", dir=archive.parent)
    os.close(descriptor)
    temporary_zip = Path(zip_name)
    temporary_attestation = _unused_temporary_path(
        prefix=f".{attestation.name}.", parent=attestation.parent
    )
    temporary_validation = _unused_temporary_path(
        prefix=f".{validation_report.name}.", parent=validation_report.parent
    )
    published = False
    board_records: list[dict[str, Any]] = []
    score_seconds = solve_seconds = restoration_seconds = 0.0
    started = perf_counter()
    try:
        for index, name in enumerate(snapshot.filenames, start=1):
            image = load_rgb_png(inputs / name, expected_sha256=snapshot.hashes_by_name[name])
            prediction = predict_fixed_b_standard(image, model, device=device)
            output_hash = atomic_write_png(staging / name, prediction.restored)
            board_records.append(
                board_attestation(
                    filename=name,
                    input_sha256=snapshot.hashes_by_name[name],
                    prediction=prediction,
                    output_png_sha256=output_hash,
                )
            )
            score_seconds += prediction.score_seconds
            solve_seconds += prediction.solve_seconds
            restoration_seconds += prediction.restoration_seconds
            print(f"[{index:03d}/{snapshot.file_count}] {name}", flush=True)

        if build_official_input_snapshot(inputs, source) != snapshot:
            raise RuntimeError("official test snapshot changed during production")
        archive_hash = deterministic_submission_zip(
            staging, list(snapshot.filenames), temporary_zip
        )
        atomic_write_json(
            temporary_attestation,
            build_attestation(
                snapshot=snapshot,
                archive_sha256=archive_hash,
                per_board=board_records,
                promotion=promotion,
                runtime_manifest=runtime_manifest,
            ),
        )
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
        _publish_validated_artifacts(
            (
                (staging, output),
                (temporary_zip, archive),
                (temporary_attestation, attestation),
                (temporary_validation, validation_report),
            )
        )
        published = True
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)
            temporary_zip.unlink(missing_ok=True)
            temporary_attestation.unlink(missing_ok=True)
            temporary_validation.unlink(missing_ok=True)
            with suppress(OSError):
                OUTPUT_ROOT.rmdir()
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
    }


__all__ = [
    "DEFAULT_ATTESTATION",
    "DEFAULT_OUTPUT_ZIP",
    "DEFAULT_PROMOTION_CONFIG",
    "FixedBSubmissionPrediction",
    "PromotionEvidence",
    "board_attestation",
    "build_attestation",
    "build_runtime_manifest",
    "dry_run_status",
    "freeze_production_runtime_preflight",
    "frozen_pipeline_record",
    "load_and_validate_attestation",
    "load_promotion_evidence",
    "predict_fixed_b_standard",
    "run_production_submission",
    "tile_multiset_sha256",
]

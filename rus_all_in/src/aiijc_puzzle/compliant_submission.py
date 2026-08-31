"""Fail-closed production packaging for the compliant puzzle pipeline.

The production path is intentionally narrow: one official dirty board is split
into its 576 upright 20x20 tiles, the no-atlas bilateral/buddies decoder returns
a strict permutation, the exact raw reassembly is audited, then the frozen RGB
seam offsets, bounded luminance gains and one colored-NLM h=20 pass are applied.
This module also contains the independent archive validator used before publish.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import shutil
import stat
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
import scipy
import skimage
import sklearn
from jsonschema import Draft202012Validator
from PIL import Image
from PIL import __version__ as PILLOW_VERSION

from aiijc_puzzle.compliant_atlas_decoder import (
    PRODUCTION_EDGE_BUDGET,
    PermutationAudit,
    audit_raw_permutation,
)
from aiijc_puzzle.legacy_upgrade import (
    atomic_write_png,
    deterministic_submission_zip,
    directional_scores,
    layout_digest,
    solve_buddies,
)
from aiijc_puzzle.pixel_tails import (
    NLM_SEARCH_WINDOW,
    NLM_TEMPLATE_WINDOW,
)
from aiijc_puzzle.postassembly_harmonizer import (
    DEFAULT_LUMINANCE_GAIN_CONFIG,
    DEFAULT_SEAM_GRAPH_CONFIG,
    apply_luminance_gains,
    apply_rgb_offsets,
    seam_graph_luminance_gains,
    seam_graph_rgb_offsets,
)
from aiijc_puzzle.protocol import (
    IMAGE_SIZE,
    RGB_CHANNELS,
    TILE_COUNT,
    TILE_SIZE,
    assemble_tiles,
    sha256_file,
    split_tiles,
)

EXPECTED_TEST_FILES = 700
SCHEMA_NAME = "aiijc-puzzle-submission-compliance-v2"
METHOD_STATUS = "METHOD_COMPLIANT_LAYOUT_ACCURACY_UNPROVEN"
PROOF_SCOPE = "provenance_bijection_geometry_and_tail_only"
PROOF_LIMITATION = (
    "PASS proves corresponding-input provenance, upright 20x20 strict bijection, "
    "raw geometry, the frozen solver implementation, and the frozen restoration tail only; "
    "it does not prove the hidden ground-truth permutation, reconstruction accuracy, "
    "or manual acceptance."
)
OFFICIAL_TEST_ARCHIVE_SHA256 = "62d365c45fe85c3da06e96f83390e7bb056935036a9b5dee7a99d32f11483c89"
OFFICIAL_FILENAMES_SHA256 = "312e8c46b2ccfa27e525d607d046d0e3676688f8c71533b8498c377d71805376"
FINAL_NLM_H = 20
FINAL_NLM_H_COLOR = 20
FINAL_NLM_PASSES = 1
RESTORATION_NAME = (
    "historical-rgb-seam-offsets_then-bounded-luminance-gains_then-"
    "opencv-fastNlMeansDenoisingColored-rgb-h20-hColor20-template7-search21-passes1"
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHEMA_PATH = PROJECT_ROOT / "configs/submission-compliance.schema.json"
PINNED_SCHEMA_SHA256 = "9e1b046a7484b20c6883a8b0322500e8230cb66a8b4ca8edd7370af05584a8ac"
DEFAULT_RGB_CONFIG_PATH = PROJECT_ROOT / "configs/postassembly_rgb_offset_v1.json"
DEFAULT_LUMA_CONFIG_PATH = PROJECT_ROOT / "configs/postassembly_luminance_gain_v1.json"
DEFAULT_PIPELINE_CONFIG_PATH = PROJECT_ROOT / "configs/frozen_submission_h20x1_fallback_v1.json"
RUNTIME_FILE_RELATIVE_PATHS = (
    "src/aiijc_puzzle/compliant_submission.py",
    "src/aiijc_puzzle/compliant_atlas_decoder.py",
    "src/aiijc_puzzle/legacy_upgrade.py",
    "src/aiijc_puzzle/postassembly_harmonizer.py",
    "src/aiijc_puzzle/protocol.py",
    "src/aiijc_puzzle/pixel_tails.py",
    "src/aiijc_puzzle/candidate_supply.py",
    "src/aiijc_puzzle/novel_analog_layout.py",
    "scripts/run_compliant_submission.py",
    "scripts/validate_compliant_submission.py",
    "configs/submission-compliance.schema.json",
    "configs/postassembly_rgb_offset_v1.json",
    "configs/postassembly_luminance_gain_v1.json",
    "configs/frozen_submission_h20x1_fallback_v1.json",
    "uv.lock",
)
EXPECTED_POLICY = {
    "output_derived_only_from_corresponding_input": True,
    "all_576_input_tiles_used_exactly_once": True,
    "tile_identity_preserved": True,
    "tile_geometry_preserved_before_restoration": True,
    "restoration_after_layout_only": True,
    "targets_used": False,
    "reference_images_used": False,
    "source_lookup_used": False,
    "external_templates_used": False,
    "cross_board_pixels_used": False,
    "tile_substitution_used": False,
    "filename_or_board_overrides_used": False,
}


@dataclass(frozen=True)
class InputSnapshot:
    """Content-addressed view of an official archive and its extraction."""

    source_archive_sha256: str
    filenames: tuple[str, ...]
    filenames_sha256: str
    input_sha256: tuple[tuple[str, str], ...]

    @property
    def file_count(self) -> int:
        return len(self.filenames)

    @property
    def hashes_by_name(self) -> dict[str, str]:
        return dict(self.input_sha256)

    def attestation_record(self) -> dict[str, Any]:
        return {
            "source_archive_sha256": self.source_archive_sha256,
            "filenames_sha256": self.filenames_sha256,
            "file_count": self.file_count,
            "regular_files_only": True,
            "symlinks_rejected": True,
        }


@dataclass(frozen=True)
class FrozenSubmissionPrediction:
    """One compliant prediction plus the exact restoration configuration."""

    layout: np.ndarray
    raw: np.ndarray
    harmonized: np.ndarray
    restored: np.ndarray
    audit: PermutationAudit
    score_seconds: float
    solve_seconds: float
    restoration_seconds: float


@dataclass(frozen=True)
class FrozenTailEvidence:
    """Hashes of the validated, target-blind final-tail configuration files."""

    rgb_config_sha256: str
    luma_config_sha256: str
    pipeline_config_sha256: str

    def harmonizers_record(self) -> dict[str, Any]:
        return {
            "order": ["rgb_seam_offsets", "bounded_luminance_gains"],
            "rgb_seam_offsets": {
                "name": "historical_additive_rgb_seam_graph_offsets",
                "config_sha256": self.rgb_config_sha256,
                "target_blind": True,
            },
            "bounded_luminance_gains": {
                "name": "historical_bounded_luminance_seam_graph_gains",
                "config_sha256": self.luma_config_sha256,
                "target_blind": True,
            },
        }


def _absolute_without_resolving(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _reject_symlink_components(path: Path, *, require_leaf: bool = True) -> Path:
    """Reject symlinks in every existing component without following the leaf."""

    absolute = _absolute_without_resolving(path)
    components = absolute.parts
    current = Path(components[0])
    for component in components[1:]:
        current /= component
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            if require_leaf or current != absolute:
                raise
            break
        if stat.S_ISLNK(mode):
            raise ValueError(f"symlink paths are forbidden: {current}")
    return absolute


def _require_regular_file(path: Path) -> Path:
    absolute = _reject_symlink_components(path)
    mode = os.lstat(absolute).st_mode
    if not stat.S_ISREG(mode):
        raise ValueError(f"expected a regular file: {absolute}")
    return absolute


def _require_directory(path: Path) -> Path:
    absolute = _reject_symlink_components(path)
    mode = os.lstat(absolute).st_mode
    if not stat.S_ISDIR(mode):
        raise ValueError(f"expected a directory: {absolute}")
    return absolute


def _mkdir_without_symlink_ancestors(path: Path) -> Path:
    """Create a directory only after validating its nearest existing ancestor."""

    absolute = _absolute_without_resolving(path)
    existing = absolute
    while True:
        try:
            os.lstat(existing)
        except FileNotFoundError:
            if existing == existing.parent:
                raise
            existing = existing.parent
            continue
        break
    _reject_symlink_components(existing)
    absolute.mkdir(parents=True, exist_ok=True)
    return _require_directory(absolute)


def _valid_png_basename(name: str) -> bool:
    return (
        bool(name)
        and "/" not in name
        and "\\" not in name
        and Path(name).name == name
        and name.endswith(".png")
    )


def filenames_digest(filenames: Sequence[str]) -> str:
    """Hash an ordered filename roster with unambiguous NUL separators."""

    names = tuple(filenames)
    if names != tuple(sorted(names)) or len(names) != len(set(names)):
        raise ValueError("filename roster must be sorted and unique")
    if any(not _valid_png_basename(name) for name in names):
        raise ValueError("filename roster must contain PNG basenames only")
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    """Return the SHA-256 of a contiguous array's bytes."""

    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _sha256_stream(stream: Any, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    while payload := stream.read(chunk_size):
        digest.update(payload)
    return digest.hexdigest()


def _scan_flat_png_directory(inputs_dir: Path, *, expected_count: int) -> tuple[str, ...]:
    inputs_dir = _require_directory(inputs_dir)
    names: list[str] = []
    with os.scandir(inputs_dir) as entries:
        for entry in entries:
            if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                raise ValueError(f"input snapshot contains a non-regular entry: {entry.path}")
            if not _valid_png_basename(entry.name):
                raise ValueError(f"input snapshot contains a non-PNG basename: {entry.name}")
            names.append(entry.name)
    names.sort()
    if len(names) != expected_count:
        raise ValueError(f"input snapshot has {len(names)} files, expected {expected_count}")
    return tuple(names)


def _safe_archive_members(
    archive: zipfile.ZipFile,
    *,
    expected_count: int,
) -> dict[str, zipfile.ZipInfo]:
    infos = archive.infolist()
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        raise ValueError("source archive contains duplicate member names")
    if len(infos) != expected_count:
        raise ValueError(f"source archive has {len(infos)} members, expected {expected_count}")
    result: dict[str, zipfile.ZipInfo] = {}
    for info in infos:
        unix_mode = (info.external_attr >> 16) & 0xFFFF
        if (
            info.is_dir()
            or not _valid_png_basename(info.filename)
            or bool(info.flag_bits & 0x1)
            or (unix_mode and stat.S_ISLNK(unix_mode))
        ):
            raise ValueError(f"unsafe source archive member: {info.filename!r}")
        result[info.filename] = info
    return result


def build_input_snapshot(
    inputs_dir: Path,
    source_archive: Path,
    *,
    expected_count: int = EXPECTED_TEST_FILES,
) -> InputSnapshot:
    """Verify and hash an extracted test directory against the organizer ZIP."""

    if isinstance(expected_count, bool) or expected_count <= 0:
        raise ValueError("expected_count must be a positive integer")
    inputs_dir = _require_directory(inputs_dir)
    source_archive = _require_regular_file(source_archive)
    filenames = _scan_flat_png_directory(inputs_dir, expected_count=expected_count)
    file_hashes: list[tuple[str, str]] = []
    archive_hash_before = sha256_file(source_archive)
    with zipfile.ZipFile(source_archive) as archive:
        members = _safe_archive_members(archive, expected_count=expected_count)
        if set(members) != set(filenames):
            raise ValueError("source archive and extracted input filename rosters differ")
        for name in filenames:
            extracted = _require_regular_file(inputs_dir / name)
            extracted_hash = sha256_file(extracted)
            with archive.open(members[name], "r") as stream:
                archived_hash = _sha256_stream(stream)
            if archived_hash != extracted_hash:
                raise ValueError(f"extracted input differs from source archive: {name}")
            file_hashes.append((name, extracted_hash))
    archive_hash_after = sha256_file(source_archive)
    if archive_hash_after != archive_hash_before:
        raise RuntimeError("source archive changed while its snapshot was being built")
    return InputSnapshot(
        source_archive_sha256=archive_hash_before,
        filenames=filenames,
        filenames_sha256=filenames_digest(filenames),
        input_sha256=tuple(file_hashes),
    )


def build_official_input_snapshot(
    inputs_dir: Path,
    source_archive: Path,
) -> InputSnapshot:
    """Bind production and public validation to the exact organizer test set."""

    snapshot = build_input_snapshot(
        inputs_dir,
        source_archive,
        expected_count=EXPECTED_TEST_FILES,
    )
    if snapshot.source_archive_sha256 != OFFICIAL_TEST_ARCHIVE_SHA256:
        raise ValueError(
            "source archive is not the pinned official test.zip: "
            f"expected {OFFICIAL_TEST_ARCHIVE_SHA256}, "
            f"got {snapshot.source_archive_sha256}"
        )
    if snapshot.filenames_sha256 != OFFICIAL_FILENAMES_SHA256:
        raise ValueError(
            "input extraction does not have the pinned official filename roster: "
            f"expected {OFFICIAL_FILENAMES_SHA256}, got {snapshot.filenames_sha256}"
        )
    return snapshot


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_runtime_manifest() -> dict[str, Any]:
    """Content-address the exact checked-in runtime and dependency versions."""

    files = {
        relative: sha256_file(_require_regular_file(PROJECT_ROOT / relative))
        for relative in RUNTIME_FILE_RELATIVE_PATHS
    }
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "scipy": scipy.__version__,
        "pillow": PILLOW_VERSION,
        "scikit_image": skimage.__version__,
        "scikit_learn": sklearn.__version__,
    }
    content: dict[str, Any] = {"files": files, "versions": versions}
    return {**content, "digest_sha256": _canonical_json_sha256(content)}


def load_rgb_png(path: Path, *, expected_sha256: str | None = None) -> np.ndarray:
    """Read one regular PNG once, validate it, and return a detached RGB array."""

    path = _require_regular_file(path)
    payload = path.read_bytes()
    actual_hash = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and actual_hash != expected_sha256:
        raise ValueError(f"input changed after snapshot: {path.name}")
    return decode_rgb_png(payload, context=str(path))


def decode_rgb_png(payload: bytes, *, context: str) -> np.ndarray:
    """Decode a strict, single-frame 480x480 RGB PNG from bytes."""

    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        if (
            image.format != "PNG"
            or image.mode != "RGB"
            or image.size != (IMAGE_SIZE, IMAGE_SIZE)
            or int(getattr(image, "n_frames", 1)) != 1
        ):
            raise ValueError(
                f"invalid PNG {context}: format={image.format}, mode={image.mode}, "
                f"size={image.size}, frames={getattr(image, 'n_frames', 1)}"
            )
        result = np.asarray(image, dtype=np.uint8).copy()
    expected = (IMAGE_SIZE, IMAGE_SIZE, RGB_CHANNELS)
    if result.shape != expected:
        raise ValueError(f"decoded image has shape {result.shape}, expected {expected}")
    return result


def restoration_name() -> str:
    """Return the only restoration declaration accepted for production."""

    return RESTORATION_NAME


def _load_json_object(path: Path) -> tuple[Path, dict[str, Any]]:
    checked = _require_regular_file(path)
    payload = json.loads(checked.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must contain a JSON object: {checked}")
    return checked, payload


def load_frozen_tail_evidence(
    *,
    rgb_config_path: Path = DEFAULT_RGB_CONFIG_PATH,
    luma_config_path: Path = DEFAULT_LUMA_CONFIG_PATH,
    pipeline_config_path: Path = DEFAULT_PIPELINE_CONFIG_PATH,
) -> FrozenTailEvidence:
    """Validate the checked-in no-atlas/harmonizer/h20x1 contract and hash it."""

    rgb_path, rgb = _load_json_object(rgb_config_path)
    luma_path, luma = _load_json_object(luma_config_path)
    pipeline_path, pipeline = _load_json_object(pipeline_config_path)
    expected_rgb_method = {
        **asdict(DEFAULT_SEAM_GRAPH_CONFIG),
        "global_gauge": "per-channel median offset equals zero",
    }
    expected_luma_method = {
        **asdict(DEFAULT_LUMINANCE_GAIN_CONFIG),
        "global_gauge": "median log gain equals zero",
    }
    if rgb.get("target_access") is not False or rgb.get("method") != expected_rgb_method:
        raise ValueError("RGB seam-offset config is not the frozen target-blind method")
    if luma.get("target_access") is not False or luma.get("method") != expected_luma_method:
        raise ValueError("luminance-gain config is not the frozen target-blind method")
    frozen = pipeline.get("pipeline")
    if not isinstance(frozen, dict):
        raise ValueError("frozen final config has no pipeline object")
    if pipeline.get("purpose") != (
        "unchanged legal final pipeline under the user-authorized >=0.25 fallback "
        "after more than ten failed architecture/layout attempts"
    ):
        raise ValueError("pipeline evidence is not the frozen user-authorized fallback")
    if pipeline.get("decision") != {
        "aspirational_gate_result": (
            "configs/frozen_final_h20x1_v1.json failed its preregistered 0.28 "
            "absolute calibration gate and remains immutable"
        ),
        "fallback_authority": (
            "user explicitly allowed completion at >=0.25 after 10-15 unsuccessful attempts"
        ),
        "retuning_after_aspirational_failure": False,
        "pipeline_changed_from_aspirational_config": False,
    }:
        raise ValueError("fallback decision provenance is missing or changed")
    if frozen.get("layout") != {
        "edge_view": "bilateral",
        "solver": "best_buddies",
        "max_edges": PRODUCTION_EDGE_BUDGET,
        "atlas_weight": 0.0,
        "strict_permutation": True,
    }:
        raise ValueError("final pipeline config is not strict no-atlas buddies96")
    if frozen.get("postassembly") != {
        "rgb_seam_offsets": "configs/postassembly_rgb_offset_v1.json",
        "bounded_luminance_gains": "configs/postassembly_luminance_gain_v1.json",
        "order": "rgb_offsets_then_luminance_then_nlm",
    }:
        raise ValueError("final pipeline config has an unsupported harmonizer order")
    if frozen.get("restoration") != {
        "name": "opencv_fast_nl_means_colored",
        "h": FINAL_NLM_H,
        "h_color": FINAL_NLM_H_COLOR,
        "template_window_size": NLM_TEMPLATE_WINDOW,
        "search_window_size": NLM_SEARCH_WINDOW,
        "passes": FINAL_NLM_PASSES,
    }:
        raise ValueError("final pipeline config is not colored NLM h20/hColor20 x1")
    forbidden = pipeline.get("forbidden")
    if not isinstance(forbidden, list) or "multipass_nlm" not in forbidden:
        raise ValueError("final pipeline config does not explicitly forbid multi-pass NLM")
    return FrozenTailEvidence(
        rgb_config_sha256=sha256_file(rgb_path),
        luma_config_sha256=sha256_file(luma_path),
        pipeline_config_sha256=sha256_file(pipeline_path),
    )


def _proper_rgb_nlm_h20_once(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image)
    expected = (IMAGE_SIZE, IMAGE_SIZE, RGB_CHANNELS)
    if value.shape != expected or value.dtype != np.uint8:
        raise ValueError(f"expected uint8 RGB image {expected}, got {value.shape}")
    bgr = cv2.cvtColor(value, cv2.COLOR_RGB2BGR)
    filtered = cv2.fastNlMeansDenoisingColored(
        bgr,
        None,
        FINAL_NLM_H,
        FINAL_NLM_H_COLOR,
        NLM_TEMPLATE_WINDOW,
        NLM_SEARCH_WINDOW,
    )
    return cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)


def _apply_frozen_harmonizers(raw: np.ndarray) -> np.ndarray:
    ordered = split_tiles(raw)
    offsets, _ = seam_graph_rgb_offsets(ordered, DEFAULT_SEAM_GRAPH_CONFIG)
    rgb_corrected = apply_rgb_offsets(ordered, offsets)
    gains, _ = seam_graph_luminance_gains(
        rgb_corrected,
        DEFAULT_LUMINANCE_GAIN_CONFIG,
    )
    return assemble_tiles(apply_luminance_gains(rgb_corrected, gains))


def predict_frozen_submission(input_image: np.ndarray) -> FrozenSubmissionPrediction:
    """Run the sole production path: no-atlas buddies96 and frozen h20x1 tail."""

    tiles = split_tiles(input_image)
    score_started = perf_counter()
    right, down = directional_scores(tiles, views=("bilateral",))["bilateral"]
    score_seconds = perf_counter() - score_started
    solved = solve_buddies(right, down, max_edges=PRODUCTION_EDGE_BUDGET)
    raw = assemble_tiles(tiles[solved.layout])
    audit = audit_raw_permutation(
        input_image,
        raw,
        solved.layout,
        restoration_applied_after_audit=True,
    )
    if not audit.passed:
        raise RuntimeError("plain buddies raw tile-permutation audit failed")
    restoration_started = perf_counter()
    harmonized = _apply_frozen_harmonizers(raw)
    restored = _proper_rgb_nlm_h20_once(harmonized)
    return FrozenSubmissionPrediction(
        layout=solved.layout,
        raw=raw,
        harmonized=harmonized,
        restored=restored,
        audit=audit,
        score_seconds=score_seconds,
        solve_seconds=solved.runtime_seconds,
        restoration_seconds=perf_counter() - restoration_started,
    )


def _nlm_attestation_record() -> dict[str, Any]:
    return {
        "name": "opencv_fast_nl_means_colored",
        "proper_rgb_bgr_roundtrip": True,
        "h": FINAL_NLM_H,
        "h_color": FINAL_NLM_H_COLOR,
        "template_window_size": NLM_TEMPLATE_WINDOW,
        "search_window_size": NLM_SEARCH_WINDOW,
        "passes": FINAL_NLM_PASSES,
    }


def _restoration_attestation(
    output: np.ndarray,
    harmonized: np.ndarray,
    evidence: FrozenTailEvidence,
) -> dict[str, Any]:
    return {
        "name": RESTORATION_NAME,
        "input_is_raw_assembly": True,
        "pixel_restoration_only": True,
        "layout_changed": False,
        "spatial_warp_used": False,
        "external_or_cross_board_pixels_used": False,
        "pipeline_config_sha256": evidence.pipeline_config_sha256,
        "harmonizers": evidence.harmonizers_record(),
        "harmonized_array_sha256": array_sha256(harmonized),
        "nlm": _nlm_attestation_record(),
        "output_array_sha256": array_sha256(output),
    }


def board_attestation(
    *,
    filename: str,
    input_sha256: str,
    layout: np.ndarray,
    raw: np.ndarray,
    harmonized: np.ndarray,
    restored: np.ndarray,
    output_png_sha256: str,
    tail_evidence: FrozenTailEvidence,
) -> dict[str, Any]:
    """Build one schema-compatible board record from audited arrays."""

    layout_array = np.asarray(layout, dtype=np.int32)
    if layout_array.shape != (TILE_COUNT,) or not np.array_equal(
        np.sort(layout_array), np.arange(TILE_COUNT, dtype=np.int32)
    ):
        raise ValueError("tile_at_position must be an exact permutation of 0..575")
    if not _valid_png_basename(filename):
        raise ValueError(f"invalid board filename: {filename!r}")
    return {
        "filename": filename,
        "input_sha256": input_sha256,
        "tile_at_position": [int(value) for value in layout_array],
        "layout_sha256": layout_digest(layout_array),
        "raw_assembly_sha256": array_sha256(raw),
        "restoration": _restoration_attestation(restored, harmonized, tail_evidence),
        "output_png_sha256": output_png_sha256,
    }


def build_attestation(
    *,
    snapshot: InputSnapshot,
    archive_sha256: str,
    per_board: Sequence[Mapping[str, Any]],
    method: str,
    runtime_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the strict compliance declaration; schema validation is separate."""

    records = [dict(record) for record in per_board]
    record_names = [record.get("filename") for record in records]
    if record_names != list(snapshot.filenames):
        raise ValueError("per-board evidence must follow the exact sorted input roster")
    if not method:
        raise ValueError("method declaration must be non-empty")
    manifest = json.loads(json.dumps(runtime_manifest))
    if manifest != build_runtime_manifest():
        raise ValueError("runtime manifest does not match the current checked-in runtime")
    return {
        "schema": SCHEMA_NAME,
        "status": METHOD_STATUS,
        "scope": PROOF_SCOPE,
        "correct_hidden_layout_proven": False,
        "proof_limitation": PROOF_LIMITATION,
        "method": method,
        "policy": dict(EXPECTED_POLICY),
        "runtime_manifest": manifest,
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
        "per_board": records,
    }


def _load_compliance_schema(
    schema_path: Path = DEFAULT_SCHEMA_PATH,
) -> dict[str, Any]:
    schema_path = _require_regular_file(schema_path)
    actual_hash = sha256_file(schema_path)
    if actual_hash != PINNED_SCHEMA_SHA256:
        raise ValueError(
            "compliance schema hash differs from the pinned checked-in contract: "
            f"expected {PINNED_SCHEMA_SHA256}, got {actual_hash}"
        )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    if schema.get("properties", {}).get("schema", {}).get("const") != SCHEMA_NAME:
        raise ValueError("compliance schema does not declare the frozen v2 contract")
    return schema


def load_and_validate_attestation(path: Path) -> dict[str, Any]:
    """Load JSON and validate it against the checked-in Draft 2020-12 schema."""

    path = _require_regular_file(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema = _load_compliance_schema()
    Draft202012Validator(schema).validate(payload)
    return payload


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> str:
    """Atomically write canonical human-readable JSON and return its hash."""

    path = _absolute_without_resolving(path)
    _mkdir_without_symlink_ancestors(path.parent)
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(encoded).hexdigest()


def _tree_overlap(left: Path, right: Path) -> bool:
    left_resolved = left.resolve(strict=False)
    right_resolved = right.resolve(strict=False)
    return (
        left_resolved == right_resolved
        or left_resolved in right_resolved.parents
        or right_resolved in left_resolved.parents
    )


def guard_artifact_paths(
    *,
    inputs_dir: Path,
    source_archive: Path,
    output_dir: Path,
    output_zip: Path,
    attestation_path: Path,
) -> tuple[Path, Path, Path, Path, Path]:
    """Fail before writing if inputs and artifacts overlap or outputs exist."""

    inputs = _require_directory(inputs_dir)
    source = _require_regular_file(source_archive)
    output = _absolute_without_resolving(output_dir)
    archive = _absolute_without_resolving(output_zip)
    attestation = _absolute_without_resolving(attestation_path)
    artifacts = (output, archive, attestation)
    if any(_tree_overlap(inputs, artifact) for artifact in artifacts):
        raise ValueError("input and output paths must be disjoint")
    if any(artifact.resolve(strict=False) == source.resolve() for artifact in artifacts):
        raise ValueError("source archive cannot be used as an output path")
    if _tree_overlap(output, archive) or _tree_overlap(output, attestation):
        raise ValueError("ZIP and attestation must be outside the prediction directory")
    if archive.resolve(strict=False) == attestation.resolve(strict=False):
        raise ValueError("ZIP and attestation paths must differ")
    for artifact in artifacts:
        if artifact.exists() or artifact.is_symlink():
            raise FileExistsError(f"refusing to overwrite output artifact: {artifact}")
        _mkdir_without_symlink_ancestors(artifact.parent)
    return inputs, source, output, archive, attestation


def _independent_raw_assembly(input_image: np.ndarray, layout: Sequence[int]) -> np.ndarray:
    """Independently reconstruct raw output without using the production audit."""

    layout_array = np.asarray(layout, dtype=np.int32)
    if layout_array.shape != (TILE_COUNT,) or not np.array_equal(
        np.sort(layout_array), np.arange(TILE_COUNT, dtype=np.int32)
    ):
        raise ValueError("declared tile_at_position is not a strict permutation")
    tiles = (
        input_image.reshape(24, TILE_SIZE, 24, TILE_SIZE, RGB_CHANNELS)
        .transpose(0, 2, 1, 3, 4)
        .reshape(TILE_COUNT, TILE_SIZE, TILE_SIZE, RGB_CHANNELS)
    )
    selected = tiles[layout_array]
    return (
        selected.reshape(24, 24, TILE_SIZE, TILE_SIZE, RGB_CHANNELS)
        .transpose(0, 2, 1, 3, 4)
        .reshape(IMAGE_SIZE, IMAGE_SIZE, RGB_CHANNELS)
    )


def _independent_layout_digest(layout: Sequence[int]) -> str:
    layout_array = np.asarray(layout, dtype=np.int32)
    if layout_array.shape != (TILE_COUNT,) or not np.array_equal(
        np.sort(layout_array), np.arange(TILE_COUNT, dtype=np.int32)
    ):
        raise ValueError("declared tile_at_position is not a strict permutation")
    return hashlib.sha256(layout_array.astype("<i4", copy=False).tobytes()).hexdigest()


def _independent_no_atlas_buddies96_layout(input_image: np.ndarray) -> np.ndarray:
    """Re-run the frozen target-blind solver from one corresponding input."""

    value = np.asarray(input_image)
    expected = (IMAGE_SIZE, IMAGE_SIZE, RGB_CHANNELS)
    if value.shape != expected or value.dtype != np.uint8:
        raise ValueError(f"expected uint8 RGB input image {expected}, got {value.shape}")
    tiles = (
        value.reshape(24, TILE_SIZE, 24, TILE_SIZE, RGB_CHANNELS)
        .transpose(0, 2, 1, 3, 4)
        .reshape(TILE_COUNT, TILE_SIZE, TILE_SIZE, RGB_CHANNELS)
    )
    right, down = directional_scores(tiles, views=("bilateral",))["bilateral"]
    solved = solve_buddies(right, down, max_edges=PRODUCTION_EDGE_BUDGET)
    layout = np.asarray(solved.layout, dtype=np.int32)
    _independent_layout_digest(layout)
    return layout


def _independent_frozen_harmonizers(raw: np.ndarray) -> np.ndarray:
    """Recompute both historical harmonizers from the declared raw assembly."""

    current = np.asarray(raw)
    expected = (IMAGE_SIZE, IMAGE_SIZE, RGB_CHANNELS)
    if current.shape != expected or current.dtype != np.uint8:
        raise ValueError(f"expected uint8 RGB raw assembly {expected}, got {current.shape}")
    tiles = (
        current.reshape(24, TILE_SIZE, 24, TILE_SIZE, RGB_CHANNELS)
        .transpose(0, 2, 1, 3, 4)
        .reshape(TILE_COUNT, TILE_SIZE, TILE_SIZE, RGB_CHANNELS)
    )
    offsets, _ = seam_graph_rgb_offsets(tiles, DEFAULT_SEAM_GRAPH_CONFIG)
    rgb_corrected = apply_rgb_offsets(tiles, offsets)
    gains, _ = seam_graph_luminance_gains(
        rgb_corrected,
        DEFAULT_LUMINANCE_GAIN_CONFIG,
    )
    corrected = apply_luminance_gains(rgb_corrected, gains)
    return (
        corrected.reshape(24, 24, TILE_SIZE, TILE_SIZE, RGB_CHANNELS)
        .transpose(0, 2, 1, 3, 4)
        .reshape(IMAGE_SIZE, IMAGE_SIZE, RGB_CHANNELS)
    )


def _independent_nlm_h20_once(harmonized: np.ndarray) -> np.ndarray:
    """Recompute exactly one h=20/hColor=20 OpenCV pass."""

    value = np.asarray(harmonized)
    expected = (IMAGE_SIZE, IMAGE_SIZE, RGB_CHANNELS)
    if value.shape != expected or value.dtype != np.uint8:
        raise ValueError(f"expected uint8 RGB harmonized image {expected}, got {value.shape}")
    bgr = cv2.cvtColor(value, cv2.COLOR_RGB2BGR)
    filtered_bgr = cv2.fastNlMeansDenoisingColored(
        bgr,
        None,
        FINAL_NLM_H,
        FINAL_NLM_H_COLOR,
        NLM_TEMPLATE_WINDOW,
        NLM_SEARCH_WINDOW,
    )
    return cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2RGB)


def _inspect_submission_members(
    archive: zipfile.ZipFile,
    *,
    expected_names: Sequence[str],
) -> dict[str, zipfile.ZipInfo]:
    infos = archive.infolist()
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        raise ValueError("submission ZIP contains duplicate member names")
    if names != list(expected_names):
        raise ValueError("submission ZIP roster/order differs from attestation")
    result: dict[str, zipfile.ZipInfo] = {}
    for info in infos:
        unix_mode = (info.external_attr >> 16) & 0xFFFF
        if (
            info.is_dir()
            or not _valid_png_basename(info.filename)
            or bool(info.flag_bits & 0x1)
            or unix_mode != 0o100644
            or info.compress_type != zipfile.ZIP_DEFLATED
            or info.file_size <= 0
        ):
            raise ValueError(f"unsafe submission member: {info.filename!r}")
        result[info.filename] = info
    return result


def _validate_submission_against_snapshot(
    *,
    snapshot: InputSnapshot,
    inputs_dir: Path,
    submission_zip: Path,
    attestation_path: Path,
) -> dict[str, Any]:
    """Validate against an already content-verified 700-board snapshot."""

    submission_zip = _require_regular_file(submission_zip)
    inputs_dir = _require_directory(inputs_dir)
    attestation = load_and_validate_attestation(attestation_path)
    tail_evidence = load_frozen_tail_evidence()
    runtime_manifest = build_runtime_manifest()
    if snapshot.file_count != EXPECTED_TEST_FILES:
        raise ValueError(f"validator requires exactly {EXPECTED_TEST_FILES} input boards")
    if attestation["status"] != METHOD_STATUS:
        raise ValueError("attestation status overclaims or differs from the frozen method status")
    if attestation["scope"] != PROOF_SCOPE:
        raise ValueError("attestation proof scope differs from the frozen limited scope")
    if attestation["correct_hidden_layout_proven"] is not False:
        raise ValueError("attestation must not claim the hidden correct layout was proven")
    if attestation["policy"] != EXPECTED_POLICY:
        raise ValueError("attested policy booleans differ from the full frozen policy object")
    if attestation["runtime_manifest"] != runtime_manifest:
        raise ValueError("attested runtime manifest differs from the current runtime")
    if attestation["input_snapshot"] != snapshot.attestation_record():
        raise ValueError("attested input snapshot does not match organizer archive/extraction")
    archive_record = attestation["archive"]
    if archive_record["filenames"] != list(snapshot.filenames):
        raise ValueError("attested output roster differs from verified input roster")
    actual_archive_hash = sha256_file(submission_zip)
    if archive_record["sha256"] != actual_archive_hash:
        raise ValueError("submission ZIP hash differs from attestation")

    board_records = attestation["per_board"]
    if [record["filename"] for record in board_records] != list(snapshot.filenames):
        raise ValueError("per-board evidence roster/order differs from input snapshot")
    input_hashes = snapshot.hashes_by_name
    restoration_declarations = {record["restoration"]["name"] for record in board_records}
    if restoration_declarations != {RESTORATION_NAME}:
        raise ValueError("all boards must use the sole frozen h20x1 restoration")
    if attestation["method"] != _method_declaration(tail_evidence, runtime_manifest):
        raise ValueError("method declaration differs from the frozen final contract")

    derived_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    with zipfile.ZipFile(submission_zip) as archive:
        members = _inspect_submission_members(archive, expected_names=snapshot.filenames)
        for record in board_records:
            name = record["filename"]
            expected_input_hash = input_hashes[name]
            if record["input_sha256"] != expected_input_hash:
                raise ValueError(f"input hash mismatch in board evidence: {name}")
            input_image = load_rgb_png(
                _absolute_without_resolving(inputs_dir) / name,
                expected_sha256=expected_input_hash,
            )
            layout = np.asarray(record["tile_at_position"], dtype=np.int32)
            if _independent_layout_digest(layout) != record["layout_sha256"]:
                raise ValueError(f"layout hash mismatch: {name}")
            derived = derived_cache.get(expected_input_hash)
            if derived is None:
                solver_layout = _independent_no_atlas_buddies96_layout(input_image)
                raw = _independent_raw_assembly(input_image, solver_layout)
                harmonized = _independent_frozen_harmonizers(raw)
                recomputed = _independent_nlm_h20_once(harmonized)
                derived = (solver_layout, raw, harmonized, recomputed)
                derived_cache[expected_input_hash] = derived
            solver_layout, raw, harmonized, recomputed = derived
            if not np.array_equal(layout, solver_layout):
                raise ValueError(f"attested layout differs from frozen solver layout: {name}")
            if array_sha256(raw) != record["raw_assembly_sha256"]:
                raise ValueError(f"raw reassembly hash mismatch: {name}")

            restoration = record["restoration"]
            if restoration["pipeline_config_sha256"] != tail_evidence.pipeline_config_sha256:
                raise ValueError(f"frozen pipeline config hash mismatch: {name}")
            if restoration["harmonizers"] != tail_evidence.harmonizers_record():
                raise ValueError(f"harmonizer declaration/config hash mismatch: {name}")
            if restoration["nlm"] != _nlm_attestation_record():
                raise ValueError(f"NLM declaration is not frozen h20/hColor20 x1: {name}")

            payload = archive.read(members[name])
            if hashlib.sha256(payload).hexdigest() != record["output_png_sha256"]:
                raise ValueError(f"output PNG hash mismatch: {name}")
            output = decode_rgb_png(payload, context=f"{submission_zip}:{name}")
            output_hash = array_sha256(output)
            if output_hash != restoration["output_array_sha256"]:
                raise ValueError(f"decoded output array hash mismatch: {name}")
            if array_sha256(harmonized) != restoration["harmonized_array_sha256"]:
                raise ValueError(f"harmonized array hash mismatch: {name}")
            if not np.array_equal(recomputed, output):
                raise ValueError(f"output is not frozen harmonizers then h20x1 restoration: {name}")

    return {
        "status": METHOD_STATUS,
        "scope": PROOF_SCOPE,
        "correct_hidden_layout_proven": False,
        "proof_limitation": PROOF_LIMITATION,
        "file_count": snapshot.file_count,
        "source_archive_sha256": snapshot.source_archive_sha256,
        "submission_zip_sha256": actual_archive_hash,
        "filenames_sha256": snapshot.filenames_sha256,
        "restoration": RESTORATION_NAME,
        "restoration_recomputed": True,
        "rgb_config_sha256": tail_evidence.rgb_config_sha256,
        "luma_config_sha256": tail_evidence.luma_config_sha256,
        "pipeline_config_sha256": tail_evidence.pipeline_config_sha256,
        "schema_sha256": PINNED_SCHEMA_SHA256,
        "runtime_manifest_sha256": runtime_manifest["digest_sha256"],
        "all_raw_assemblies_recomputed": True,
        "all_solver_layouts_recomputed": True,
        "all_layouts_are_strict_permutations": True,
        "unique_input_derivations_recomputed": len(derived_cache),
    }


def validate_submission(
    *,
    inputs_dir: Path,
    source_archive: Path,
    submission_zip: Path,
    attestation_path: Path,
) -> dict[str, Any]:
    """Public fail-closed validation bound to the exact official 700-board set."""

    snapshot = build_official_input_snapshot(inputs_dir, source_archive)
    return _validate_submission_against_snapshot(
        snapshot=snapshot,
        inputs_dir=inputs_dir,
        submission_zip=submission_zip,
        attestation_path=attestation_path,
    )


def _method_declaration(
    evidence: FrozenTailEvidence,
    runtime_manifest: Mapping[str, Any],
) -> str:
    runtime_digest = runtime_manifest.get("digest_sha256")
    if not isinstance(runtime_digest, str) or len(runtime_digest) != 64:
        raise ValueError("runtime manifest has no SHA-256 digest")
    return (
        "corresponding-input-only; upright-20x20-tiles; "
        "absolute-position-unary=none; bilateral-directional-scores; "
        f"strict-buddies{PRODUCTION_EDGE_BUDGET}-permutation; raw-pixel-audit; "
        f"rgb-config-sha256={evidence.rgb_config_sha256}; "
        f"luma-config-sha256={evidence.luma_config_sha256}; "
        f"pipeline-config-sha256={evidence.pipeline_config_sha256}; "
        f"runtime-manifest-sha256={runtime_digest}; "
        f"{RESTORATION_NAME}"
    )


def run_production_submission(
    *,
    inputs_dir: Path,
    source_archive: Path,
    output_dir: Path,
    output_zip: Path,
    attestation_path: Path,
) -> dict[str, Any]:
    """Generate, fully validate, and atomically publish one 700-image bundle."""

    inputs, source, output, archive_path, attestation_final = guard_artifact_paths(
        inputs_dir=inputs_dir,
        source_archive=source_archive,
        output_dir=output_dir,
        output_zip=output_zip,
        attestation_path=attestation_path,
    )
    _load_compliance_schema()
    snapshot = build_official_input_snapshot(inputs, source)
    tail_evidence = load_frozen_tail_evidence()
    runtime_manifest = build_runtime_manifest()

    staging_dir = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    zip_descriptor, zip_name = tempfile.mkstemp(
        prefix=f".{archive_path.name}.", dir=archive_path.parent
    )
    os.close(zip_descriptor)
    temporary_zip = Path(zip_name)
    json_descriptor, json_name = tempfile.mkstemp(
        prefix=f".{attestation_final.name}.", dir=attestation_final.parent
    )
    os.close(json_descriptor)
    temporary_attestation = Path(json_name)
    temporary_attestation.unlink()
    published = False
    started = perf_counter()
    board_records: list[dict[str, Any]] = []
    total_score_seconds = 0.0
    total_solve_seconds = 0.0
    total_restoration_seconds = 0.0
    try:
        for index, name in enumerate(snapshot.filenames, start=1):
            image = load_rgb_png(inputs / name, expected_sha256=snapshot.hashes_by_name[name])
            prediction = predict_frozen_submission(image)
            if not prediction.audit.passed:
                raise RuntimeError(f"pre-restoration permutation audit failed: {name}")
            output_png_hash = atomic_write_png(staging_dir / name, prediction.restored)
            board_records.append(
                board_attestation(
                    filename=name,
                    input_sha256=snapshot.hashes_by_name[name],
                    layout=prediction.layout,
                    raw=prediction.raw,
                    harmonized=prediction.harmonized,
                    restored=prediction.restored,
                    output_png_sha256=output_png_hash,
                    tail_evidence=tail_evidence,
                )
            )
            total_score_seconds += prediction.score_seconds
            total_solve_seconds += prediction.solve_seconds
            total_restoration_seconds += prediction.restoration_seconds
            print(f"[{index:03d}/{snapshot.file_count}] {name}", flush=True)

        if build_official_input_snapshot(inputs, source) != snapshot:
            raise RuntimeError("official input snapshot changed during inference")
        archive_sha256 = deterministic_submission_zip(
            staging_dir,
            list(snapshot.filenames),
            temporary_zip,
        )
        attestation_payload = build_attestation(
            snapshot=snapshot,
            archive_sha256=archive_sha256,
            per_board=board_records,
            method=_method_declaration(tail_evidence, runtime_manifest),
            runtime_manifest=runtime_manifest,
        )
        atomic_write_json(temporary_attestation, attestation_payload)
        validation = validate_submission(
            inputs_dir=inputs,
            source_archive=source,
            submission_zip=temporary_zip,
            attestation_path=temporary_attestation,
        )

        os.replace(staging_dir, output)
        os.replace(temporary_zip, archive_path)
        os.replace(temporary_attestation, attestation_final)
        published = True
    finally:
        if not published:
            shutil.rmtree(staging_dir, ignore_errors=True)
            temporary_zip.unlink(missing_ok=True)
            temporary_attestation.unlink(missing_ok=True)

    return {
        "status": METHOD_STATUS,
        "scope": PROOF_SCOPE,
        "correct_hidden_layout_proven": False,
        "proof_limitation": PROOF_LIMITATION,
        "file_count": snapshot.file_count,
        "output_dir": str(output),
        "output_zip": str(archive_path),
        "attestation": str(attestation_final),
        "source_archive_sha256": snapshot.source_archive_sha256,
        "submission_zip_sha256": validation["submission_zip_sha256"],
        "layout": "no-atlas bilateral buddies96",
        "restoration": RESTORATION_NAME,
        "rgb_config_sha256": tail_evidence.rgb_config_sha256,
        "luma_config_sha256": tail_evidence.luma_config_sha256,
        "pipeline_config_sha256": tail_evidence.pipeline_config_sha256,
        "schema_sha256": PINNED_SCHEMA_SHA256,
        "runtime_manifest_sha256": runtime_manifest["digest_sha256"],
        "score_seconds": total_score_seconds,
        "solve_seconds": total_solve_seconds,
        "restoration_seconds": total_restoration_seconds,
        "elapsed_seconds": perf_counter() - started,
        "independent_validation": validation,
    }

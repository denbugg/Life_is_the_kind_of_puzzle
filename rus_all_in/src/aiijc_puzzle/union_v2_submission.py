"""Fail-closed resumable production packager for the frozen Union-v2 solver.

The module has no target, manifest, source-retrieval, template, filename-rule or
cross-board-pixel input.  One corresponding official RGB board is split into
its 576 original upright 20x20 tiles.  The SHA-locked raw/twin/Union adapter
returns one strict layout, that raw permutation is audited, and only then the
historical RGB-offset -> bounded-luminance -> colored-NLM h20 tail is applied.

Dry-run inspection deliberately reads archive/directory metadata only.  The
official PNG payloads are opened only by :func:`run_union_v2_submission` after
an explicit CLI ``--run``.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import stat
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
import torch
from PIL import __version__ as PILLOW_VERSION

from aiijc_puzzle.compliant_submission import (
    EXPECTED_TEST_FILES,
    OFFICIAL_FILENAMES_SHA256,
    OFFICIAL_TEST_ARCHIVE_SHA256,
    InputSnapshot,
    array_sha256,
    atomic_write_json,
    build_official_input_snapshot,
    filenames_digest,
    load_rgb_png,
)
from aiijc_puzzle.legacy_upgrade import (
    atomic_write_png,
    deterministic_submission_zip,
    layout_digest,
)
from aiijc_puzzle.protocol import IMAGE_SIZE, TILE_COUNT, sha256_file
from aiijc_puzzle.socket_pixel_tails import (
    historical_rgb_luma_nlm_h20_contract,
    historical_rgb_luma_nlm_h20_once,
)
from aiijc_puzzle.socket_sorter_production import (
    assemble_audited_original_tiles,
    load_socket_checkpoint,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/union_v2_submission_production_v1.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/union-v2-submission/predictions"
DEFAULT_OUTPUT_ZIP = PROJECT_ROOT / "outputs/union-v2-submission/submission-union-v2.zip"
DEFAULT_ATTESTATION = PROJECT_ROOT / "outputs/union-v2-submission/attestation.json"
DEFAULT_VALIDATION_STATE = (
    PROJECT_ROOT / "outputs/union-v2-submission/validation-progress.json"
)
PACKAGER_SCHEMA = "aiijc-union-v2-submission-packager-v1"
RECORD_SCHEMA = "aiijc-union-v2-submission-board-record-v1"
ATTESTATION_SCHEMA = "aiijc-union-v2-submission-attestation-v1"
METHOD_STATUS = "METHOD_COMPLIANT_LAYOUT_ACCURACY_UNPROVEN"
PROOF_SCOPE = "provenance_strict_bijection_geometry_model_and_tail_reexecution"
PROOF_LIMITATION = (
    "PASS proves exact official-input provenance, the frozen target-free model and "
    "configuration lineage, one strict upright 20x20 tile permutation, raw-pixel "
    "bijection before restoration, and the frozen per-board restoration tail; it "
    "does not prove the hidden ground-truth layout, score, or manual acceptance."
)
FROZEN_PRODUCTION_CONFIG_SHA256 = (
    "0d58b59915a5797db0ec4ac956fb2180fea2aed4df8d8f19bc02795924311aad"
)
RUNTIME_SOURCE_ALLOWLIST = (
    "src/aiijc_puzzle/union_v2_submission.py",
    "src/aiijc_puzzle/union_v2_submission_validation.py",
    "src/aiijc_puzzle/raw_twin_union_production.py",
    "src/aiijc_puzzle/raw_twin_union_reranker.py",
    "src/aiijc_puzzle/fullres_twin_side_matcher.py",
    "src/aiijc_puzzle/component_relation_reranker.py",
    "src/aiijc_puzzle/socket_sorter_production.py",
    "src/aiijc_puzzle/socket_matcher.py",
    "src/aiijc_puzzle/socket_decoder.py",
    "src/aiijc_puzzle/socket_translation_placer.py",
    "src/aiijc_puzzle/socket_pixel_tails.py",
    "src/aiijc_puzzle/postassembly_harmonizer.py",
    "src/aiijc_puzzle/pixel_tails.py",
    "src/aiijc_puzzle/protocol.py",
    "src/aiijc_puzzle/legacy_upgrade.py",
    "src/aiijc_puzzle/compliant_submission.py",
    "scripts/run_union_v2_submission.py",
    "scripts/validate_union_v2_submission.py",
)
EXPECTED_POLICY: dict[str, bool] = {
    "corresponding_input_only": True,
    "targets_used": False,
    "reference_images_used": False,
    "source_lookup_used": False,
    "external_templates_used": False,
    "filename_or_board_overrides_used": False,
    "tile_substitution_used": False,
    "constant_or_near_flat_tile_substitution_used": False,
    "tile_rotation_used": False,
    "tile_resize_used": False,
    "tile_warp_used": False,
    "cross_board_pixels_used": False,
    "external_pixels_used": False,
    "all_576_input_tiles_used_exactly_once_before_restoration": True,
    "original_upright_tiles_only": True,
    "raw_permutation_audited_before_restoration": True,
    "restoration_changes_layout": False,
}


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _valid_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        raise ValueError(f"{name} must be one lowercase SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be one lowercase SHA-256 digest") from error
    return value


def _reject_symlink_components(path: Path, *, require_leaf: bool = True) -> Path:
    absolute = _absolute(path)
    current = Path(absolute.parts[0])
    for component in absolute.parts[1:]:
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
    if not stat.S_ISREG(os.lstat(absolute).st_mode):
        raise ValueError(f"expected a regular file: {absolute}")
    return absolute


def _require_directory(path: Path) -> Path:
    absolute = _reject_symlink_components(path)
    if not stat.S_ISDIR(os.lstat(absolute).st_mode):
        raise ValueError(f"expected a directory: {absolute}")
    return absolute


def _mkdir_safe(path: Path) -> Path:
    absolute = _absolute(path)
    ancestor = absolute
    while not ancestor.exists():
        if ancestor == ancestor.parent:
            raise ValueError(f"cannot find a safe output ancestor: {absolute}")
        ancestor = ancestor.parent
    _reject_symlink_components(ancestor)
    absolute.mkdir(parents=True, exist_ok=True)
    return _require_directory(absolute)


def _paths_overlap(first: Path, second: Path) -> bool:
    left = first.resolve(strict=False)
    right = second.resolve(strict=False)
    return left == right or left in right.parents or right in left.parents


def _valid_png_name(name: str) -> bool:
    return (
        bool(name)
        and Path(name).name == name
        and "/" not in name
        and "\\" not in name
        and name.endswith(".png")
    )


def _metadata_roster(source_dir: Path, source_archive: Path) -> tuple[Path, Path, tuple[str, ...]]:
    """Verify official archive and flat rosters without reading PNG payloads."""

    source = _require_directory(source_dir)
    archive_path = _require_regular_file(source_archive)
    archive_sha256 = sha256_file(archive_path)
    if archive_sha256 != OFFICIAL_TEST_ARCHIVE_SHA256:
        raise ValueError("source archive is not the pinned official test.zip")
    directory_names: list[str] = []
    with os.scandir(source) as entries:
        for entry in entries:
            if (
                entry.is_symlink()
                or not entry.is_file(follow_symlinks=False)
                or not _valid_png_name(entry.name)
            ):
                raise ValueError(f"source directory has a foreign entry: {entry.name}")
            directory_names.append(entry.name)
    directory_names.sort()
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        archive_names = [info.filename for info in infos]
        if len(archive_names) != len(set(archive_names)):
            raise ValueError("official archive metadata has duplicate members")
        if any(info.is_dir() or not _valid_png_name(info.filename) for info in infos):
            raise ValueError("official archive metadata is not a flat PNG roster")
    archive_names.sort()
    names = tuple(directory_names)
    if (
        len(names) != EXPECTED_TEST_FILES
        or names != tuple(archive_names)
        or filenames_digest(names) != OFFICIAL_FILENAMES_SHA256
    ):
        raise ValueError("source/archive metadata differs from the pinned official roster")
    return source, archive_path, names


@dataclass(frozen=True)
class FrozenUnionSubmissionConfig:
    path: Path
    sha256: str
    payload: dict[str, Any]
    artifacts: dict[str, Path]
    source_hashes: dict[str, str]


def load_union_submission_config(path: Path = DEFAULT_CONFIG) -> FrozenUnionSubmissionConfig:
    """Load one sidecar-pinned production config and verify every artifact/source."""

    config_path = _require_regular_file(path)
    config_sha = sha256_file(config_path)
    if config_sha != FROZEN_PRODUCTION_CONFIG_SHA256:
        raise ValueError("Union-v2 production config SHA-256 mismatch")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != PACKAGER_SCHEMA:
        raise ValueError("unsupported Union-v2 production config schema")
    official = payload.get("official_input")
    if official != {
        "archive_sha256": OFFICIAL_TEST_ARCHIVE_SHA256,
        "filenames_sha256": OFFICIAL_FILENAMES_SHA256,
        "file_count": EXPECTED_TEST_FILES,
    }:
        raise ValueError("production config does not pin the exact official roster")
    if payload.get("policy") != EXPECTED_POLICY:
        raise ValueError("production config policy differs from the fail-closed policy")
    raw_artifacts = payload.get("artifacts")
    if not isinstance(raw_artifacts, Mapping):
        raise ValueError("production config has no artifact registry")
    artifacts: dict[str, Path] = {}
    for name in (
        "socket_checkpoint",
        "twin_checkpoint",
        "union_checkpoint",
        "union_preregistration",
        "union_selection",
        "attestation_schema",
    ):
        record = raw_artifacts.get(name)
        if not isinstance(record, Mapping):
            raise ValueError(f"production artifact record is missing: {name}")
        relative = record.get("path")
        expected = _valid_sha256(record.get("sha256"), name=f"artifacts.{name}.sha256")
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise ValueError(f"production artifact path must be project-relative: {name}")
        resolved = _require_regular_file(PROJECT_ROOT / relative)
        if sha256_file(resolved) != expected:
            raise ValueError(f"production artifact SHA-256 mismatch: {name}")
        artifacts[name] = resolved
    if payload.get("runtime_source_allowlist") != list(RUNTIME_SOURCE_ALLOWLIST):
        raise ValueError("production config runtime source allowlist changed")
    source_hashes = {
        relative: sha256_file(_require_regular_file(PROJECT_ROOT / relative))
        for relative in RUNTIME_SOURCE_ALLOWLIST
    }
    tail = historical_rgb_luma_nlm_h20_contract()
    if payload.get("pixel_tail") != tail:
        raise ValueError("production config tail differs from the frozen historical tail")
    return FrozenUnionSubmissionConfig(
        path=config_path,
        sha256=config_sha,
        payload=payload,
        artifacts=artifacts,
        source_hashes=source_hashes,
    )


@dataclass(frozen=True)
class LoadedUnionV2Engine:
    device: torch.device
    socket: Any
    twin: Any
    union: Any
    evidence: dict[str, Any]

    @torch.inference_mode()
    def predict_layout(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        from aiijc_puzzle.raw_twin_union_production import (
            predict_raw_twin_union_variants,
        )

        variants = predict_raw_twin_union_variants(
            image,
            self.socket,
            device=self.device,
            twin=self.twin,
            union=self.union,
        )
        layout = np.ascontiguousarray(variants.selected.layout, dtype=np.int32)
        if layout.shape != (TILE_COUNT,) or not np.array_equal(
            np.sort(layout), np.arange(TILE_COUNT, dtype=np.int32)
        ):
            raise RuntimeError("Union-v2 adapter returned a non-strict layout")
        report = variants.report()
        if not isinstance(report, dict):
            raise RuntimeError("Union-v2 adapter report is not a mapping")
        if (
            report.get("selected_variant") != "raw-twin-union-v2"
            or report.get("fallback_reason") is not None
        ):
            raise RuntimeError("Union-v2 production adapter silently fell back")
        # Canonical roundtrip also rejects tensors/arrays accidentally leaking
        # into resumable board records.
        report = json.loads(json.dumps(report, allow_nan=False))
        return layout, report


def _device(name: str, *, allow_nondeterministic_mps: bool) -> torch.device:
    if name not in {"cpu", "mps"}:
        raise ValueError("device must be explicitly cpu or mps")
    if name == "mps":
        if not torch.backends.mps.is_available() or not allow_nondeterministic_mps:
            raise ValueError(
                "MPS requires availability and explicit nondeterminism acknowledgement"
            )
        torch.use_deterministic_algorithms(False)
    elif allow_nondeterministic_mps:
        raise ValueError("MPS acknowledgement cannot be supplied for CPU")
    else:
        torch.use_deterministic_algorithms(True)
    return torch.device(name)


def load_union_v2_engine(
    config: FrozenUnionSubmissionConfig,
    *,
    device_name: str,
    allow_nondeterministic_mps: bool = False,
) -> LoadedUnionV2Engine:
    """Load only the three artifacts pinned by the production config."""

    from aiijc_puzzle.raw_twin_union_production import (
        load_fullres_twin_checkpoint,
        load_raw_twin_union_checkpoint,
    )

    device = _device(
        device_name,
        allow_nondeterministic_mps=allow_nondeterministic_mps,
    )
    socket = load_socket_checkpoint(config.artifacts["socket_checkpoint"], device=device)
    twin = load_fullres_twin_checkpoint(config.artifacts["twin_checkpoint"], device=device)
    union = load_raw_twin_union_checkpoint(
        config.artifacts["union_checkpoint"],
        config_path=config.artifacts["union_preregistration"],
        selection_path=config.artifacts["union_selection"],
        device=device,
    )
    configured = config.payload["artifacts"]
    evidence = {
        name: {
            "path": str(configured[name]["path"]),
            "sha256": str(configured[name]["sha256"]),
        }
        for name in (
            "socket_checkpoint",
            "twin_checkpoint",
            "union_checkpoint",
            "union_preregistration",
            "union_selection",
        )
    }
    return LoadedUnionV2Engine(device, socket, twin, union, evidence)


def _runtime_manifest(config: FrozenUnionSubmissionConfig) -> dict[str, Any]:
    content: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "pillow": PILLOW_VERSION,
        "source_sha256": dict(config.source_hashes),
    }
    return {**content, "digest_sha256": _digest_json(content)}


def build_pipeline_contract(
    config: FrozenUnionSubmissionConfig,
    engine: LoadedUnionV2Engine,
    *,
    allow_nondeterministic_mps: bool,
) -> dict[str, Any]:
    """Build the immutable model, policy, source and tail identity."""

    runtime = _runtime_manifest(config)
    payload: dict[str, Any] = {
        "schema": PACKAGER_SCHEMA,
        "production_config_sha256": config.sha256,
        "artifacts": engine.evidence,
        "layout": dict(config.payload["layout"]),
        "pixel_tail": historical_rgb_luma_nlm_h20_contract(),
        "device": str(engine.device),
        "mps_nondeterminism_acknowledged": bool(allow_nondeterministic_mps),
        "policy": dict(EXPECTED_POLICY),
        "runtime": runtime,
    }
    payload["pipeline_digest"] = _digest_json(payload)
    return payload


def inspect_union_v2_submission(
    *,
    source_dir: Path,
    source_archive: Path,
    output_dir: Path,
    output_zip: Path,
    attestation_path: Path,
    config_path: Path = DEFAULT_CONFIG,
    device_name: str = "cpu",
    allow_nondeterministic_mps: bool = False,
) -> tuple[FrozenUnionSubmissionConfig, LoadedUnionV2Engine, dict[str, Any]]:
    """Validate metadata/models and return a no-test-pixel, no-write plan."""

    source, archive, names = _metadata_roster(source_dir, source_archive)
    output = _absolute(output_dir)
    zip_path = _absolute(output_zip)
    attestation = _absolute(attestation_path)
    if any(_paths_overlap(source, item) for item in (output, zip_path, attestation)):
        raise ValueError("official input and output paths must be disjoint")
    if _paths_overlap(output, zip_path) or _paths_overlap(output, attestation):
        raise ValueError("ZIP and attestation must remain outside the prediction directory")
    if zip_path in (archive, attestation) or attestation == archive:
        raise ValueError("archive and output artifact paths must be distinct")
    config = load_union_submission_config(config_path)
    engine = load_union_v2_engine(
        config,
        device_name=device_name,
        allow_nondeterministic_mps=allow_nondeterministic_mps,
    )
    pipeline = build_pipeline_contract(
        config,
        engine,
        allow_nondeterministic_mps=allow_nondeterministic_mps,
    )
    return config, engine, {
        "status": "DRY_RUN_METADATA_ONLY_NO_TEST_PIXELS_OPENED",
        "source_dir": str(source),
        "source_archive": str(archive),
        "output_dir": str(output),
        "output_zip": str(zip_path),
        "attestation": str(attestation),
        "file_count": len(names),
        "filenames_sha256": filenames_digest(names),
        "source_archive_sha256": OFFICIAL_TEST_ARCHIVE_SHA256,
        "pipeline": pipeline,
        "test_pixels_opened": False,
        "writes_performed": False,
    }


def _output_state(
    output_root: Path,
    records_dir: Path,
    filenames: Sequence[str],
    *,
    require_complete: bool,
) -> None:
    expected_outputs = set(filenames)
    expected_records = {f"{name}.json" for name in filenames}
    observed_outputs: set[str] = set()
    with os.scandir(output_root) as entries:
        for entry in entries:
            if entry.name == "records":
                if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
                    raise ValueError("records is not a regular directory")
                continue
            if entry.name == "run.json":
                if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                    raise ValueError("run.json is not a regular file")
                continue
            if (
                entry.name not in expected_outputs
                or entry.is_symlink()
                or not entry.is_file(follow_symlinks=False)
            ):
                raise ValueError(f"prediction directory contains a foreign file: {entry.name}")
            observed_outputs.add(entry.name)
    observed_records: set[str] = set()
    with os.scandir(records_dir) as entries:
        for entry in entries:
            if (
                entry.name not in expected_records
                or entry.is_symlink()
                or not entry.is_file(follow_symlinks=False)
            ):
                raise ValueError(f"records directory contains a foreign file: {entry.name}")
            observed_records.add(entry.name)
    if require_complete and (
        observed_outputs != expected_outputs or observed_records != expected_records
    ):
        raise ValueError("completed prediction/record rosters are incomplete")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(_require_regular_file(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def _board_record(
    *,
    filename: str,
    input_sha256: str,
    input_image: np.ndarray,
    layout: np.ndarray,
    raw: np.ndarray,
    audit: Any,
    output: np.ndarray,
    output_png_sha256: str,
    pipeline: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": RECORD_SCHEMA,
        "filename": filename,
        "input": {
            "file_sha256": input_sha256,
            "decoded_rgb_sha256": array_sha256(input_image),
        },
        "lineage": {"pipeline_digest": pipeline["pipeline_digest"]},
        "layout": {
            "tile_at_position": layout.tolist(),
            "sha256_int32_le": layout_digest(layout),
            "strict_permutation": True,
            "all_576_original_upright_tiles_used_once": True,
        },
        "raw_assembly": {
            "array_sha256": array_sha256(raw),
            "audit": audit.as_dict(),
            "audited_before_restoration": True,
        },
        "pixel_tail": {
            "contract": historical_rgb_luma_nlm_h20_contract(),
            "layout_changed": False,
            "output_array_sha256": array_sha256(output),
        },
        "output_png_sha256": output_png_sha256,
        "diagnostics": dict(diagnostics),
    }


def _stable_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: record[key]
        for key in (
            "schema",
            "filename",
            "input",
            "lineage",
            "layout",
            "raw_assembly",
            "pixel_tail",
            "output_png_sha256",
        )
    }


def _record_layout(record: Mapping[str, Any], *, filename: str) -> np.ndarray:
    layout_record = record.get("layout")
    if not isinstance(layout_record, Mapping):
        raise ValueError(f"resume layout record is malformed: {filename}")
    raw = np.asarray(layout_record.get("tile_at_position"))
    if raw.dtype.kind not in {"i", "u"}:
        raise ValueError(f"resume layout is not integer-valued: {filename}")
    layout = np.ascontiguousarray(raw, dtype=np.int32)
    if layout.shape != (TILE_COUNT,) or not np.array_equal(
        np.sort(layout), np.arange(TILE_COUNT, dtype=np.int32)
    ):
        raise ValueError(f"resume layout is not one strict permutation: {filename}")
    return layout


def _predict_board(
    image: np.ndarray,
    engine: LoadedUnionV2Engine,
) -> tuple[np.ndarray, np.ndarray, Any, np.ndarray, dict[str, Any]]:
    layout, diagnostics = engine.predict_layout(image)
    raw, audit = assemble_audited_original_tiles(
        image,
        layout,
        restoration_applied_after_audit=True,
    )
    if not audit.passed:
        raise RuntimeError("Union-v2 raw permutation audit failed before restoration")
    output = historical_rgb_luma_nlm_h20_once(raw)
    return layout, raw, audit, output, diagnostics


def _validate_material(
    *,
    record: Mapping[str, Any],
    filename: str,
    input_sha256: str,
    input_image: np.ndarray,
    layout: np.ndarray,
    raw: np.ndarray,
    output: np.ndarray,
    output_path: Path,
    pipeline: Mapping[str, Any],
) -> None:
    if record.get("schema") != RECORD_SCHEMA or record.get("filename") != filename:
        raise ValueError(f"resume record identity mismatch: {filename}")
    if record.get("input") != {
        "file_sha256": input_sha256,
        "decoded_rgb_sha256": array_sha256(input_image),
    }:
        raise ValueError(f"resume input mismatch: {filename}")
    if record.get("lineage") != {"pipeline_digest": pipeline["pipeline_digest"]}:
        raise ValueError(f"resume pipeline mismatch: {filename}")
    layout_record = record.get("layout")
    raw_record = record.get("raw_assembly")
    tail_record = record.get("pixel_tail")
    if not all(isinstance(item, Mapping) for item in (layout_record, raw_record, tail_record)):
        raise ValueError(f"resume record structure mismatch: {filename}")
    if (
        layout_record.get("tile_at_position") != layout.tolist()
        or layout_record.get("sha256_int32_le") != layout_digest(layout)
        or raw_record.get("array_sha256") != array_sha256(raw)
        or tail_record.get("contract") != historical_rgb_luma_nlm_h20_contract()
        or tail_record.get("output_array_sha256") != array_sha256(output)
    ):
        raise ValueError(f"resume model/raw/tail mismatch: {filename}")
    observed = load_rgb_png(output_path)
    png_sha = sha256_file(output_path)
    if record.get("output_png_sha256") != png_sha or not np.array_equal(observed, output):
        raise ValueError(f"resume output PNG mismatch: {filename}")


def _run_identity(
    snapshot: InputSnapshot,
    pipeline: Mapping[str, Any],
    source_dir: Path,
) -> dict[str, Any]:
    roster = [
        {"filename": name, "input_file_sha256": digest}
        for name, digest in snapshot.input_sha256
    ]
    payload: dict[str, Any] = {
        "schema": PACKAGER_SCHEMA,
        "pipeline_digest": pipeline["pipeline_digest"],
        "source_dir": str(source_dir),
        "source_archive_sha256": snapshot.source_archive_sha256,
        "filenames_sha256": snapshot.filenames_sha256,
        "roster": roster,
        "roster_digest": _digest_json({"roster": roster}),
    }
    payload["run_digest"] = _digest_json(payload)
    return payload


def _build_attestation(
    *,
    snapshot: InputSnapshot,
    pipeline: Mapping[str, Any],
    archive_sha256: str,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    compact = [_stable_record(record) for record in records]
    if [record["filename"] for record in compact] != list(snapshot.filenames):
        raise ValueError("attestation record roster/order differs from official input")
    return {
        "schema": ATTESTATION_SCHEMA,
        "status": METHOD_STATUS,
        "scope": PROOF_SCOPE,
        "correct_hidden_layout_proven": False,
        "proof_limitation": PROOF_LIMITATION,
        "policy": dict(EXPECTED_POLICY),
        "input_snapshot": snapshot.attestation_record(),
        "pipeline": dict(pipeline),
        "archive": {
            "sha256": archive_sha256,
            "file_count": snapshot.file_count,
            "root_only": True,
            "format": "PNG",
            "mode": "RGB",
            "width": IMAGE_SIZE,
            "height": IMAGE_SIZE,
            "filenames": list(snapshot.filenames),
        },
        "per_board": compact,
    }


def run_union_v2_submission(
    *,
    source_dir: Path,
    source_archive: Path,
    output_dir: Path = DEFAULT_OUTPUT_ROOT,
    output_zip: Path = DEFAULT_OUTPUT_ZIP,
    attestation_path: Path = DEFAULT_ATTESTATION,
    validation_state_path: Path = DEFAULT_VALIDATION_STATE,
    config_path: Path = DEFAULT_CONFIG,
    device_name: str = "cpu",
    allow_nondeterministic_mps: bool = False,
) -> dict[str, Any]:
    """Run/resume all 700 boards, package, independently validate and publish."""

    config, engine, plan = inspect_union_v2_submission(
        source_dir=source_dir,
        source_archive=source_archive,
        output_dir=output_dir,
        output_zip=output_zip,
        attestation_path=attestation_path,
        config_path=config_path,
        device_name=device_name,
        allow_nondeterministic_mps=allow_nondeterministic_mps,
    )
    source = Path(plan["source_dir"])
    archive_source = Path(plan["source_archive"])
    output_root = _absolute(output_dir)
    zip_final = _absolute(output_zip)
    attestation_final = _absolute(attestation_path)
    state_path = _absolute(validation_state_path)
    forbidden_state_paths = {zip_final, attestation_final, archive_source}
    if (
        _paths_overlap(source, state_path)
        or _paths_overlap(output_root, state_path)
        or state_path in forbidden_state_paths
    ):
        raise ValueError("validation state must be a distinct artifact outside predictions")
    snapshot = build_official_input_snapshot(source, archive_source)
    pipeline = plan["pipeline"]
    identity = _run_identity(snapshot, pipeline, source)

    if zip_final.exists() != attestation_final.exists():
        raise ValueError("one-sided final ZIP/attestation publication is forbidden")
    if zip_final.exists():
        from aiijc_puzzle.union_v2_submission_validation import (
            validate_union_v2_submission,
        )

        validation = validate_union_v2_submission(
            source_dir=source,
            source_archive=archive_source,
            output_dir=output_root,
            submission_zip=zip_final,
            attestation_path=attestation_final,
            validation_state_path=state_path,
            config_path=config.path,
            device_name=device_name,
            allow_nondeterministic_mps=allow_nondeterministic_mps,
        )
        return {
            "status": "COMPLETE_RESUMED_FINAL_VALIDATED",
            "run_digest": identity["run_digest"],
            "validation": validation,
        }

    output_root = _mkdir_safe(output_root)
    records_dir = _mkdir_safe(output_root / "records")
    run_path = output_root / "run.json"
    _output_state(output_root, records_dir, snapshot.filenames, require_complete=False)
    stable_run = dict(identity)
    if run_path.exists():
        existing = _load_json(run_path)
        if any(existing.get(key) != value for key, value in stable_run.items()):
            raise ValueError("existing run.json belongs to another pipeline/input snapshot")
    elif any(path.name != "records" for path in output_root.iterdir()) or any(
        records_dir.iterdir()
    ):
        raise ValueError("non-empty prediction directory has no matching run.json")
    progress: dict[str, Any] = {
        **stable_run,
        "status": "IN_PROGRESS",
        "pipeline": pipeline,
        "completed_filenames": [],
    }
    atomic_write_json(run_path, progress)
    processed = 0
    resumed = 0
    started = perf_counter()
    for index, (name, expected_input_hash) in enumerate(snapshot.input_sha256, start=1):
        image = load_rgb_png(source / name, expected_sha256=expected_input_hash)
        output_path = output_root / name
        record_path = records_dir / f"{name}.json"
        output_exists = output_path.exists()
        record_exists = record_path.exists()
        if output_exists != record_exists:
            raise ValueError(f"one-sided board output/record is forbidden: {name}")
        if output_exists:
            record = _load_json(record_path)
            layout = _record_layout(record, filename=name)
            raw, audit = assemble_audited_original_tiles(
                image,
                layout,
                restoration_applied_after_audit=True,
            )
            if not audit.passed:
                raise RuntimeError(f"resume raw permutation audit failed: {name}")
            output = historical_rgb_luma_nlm_h20_once(raw)
            resumed += 1
        else:
            layout, raw, audit, output, diagnostics = _predict_board(image, engine)
            output_png_sha = atomic_write_png(output_path, output)
            record = _board_record(
                filename=name,
                input_sha256=expected_input_hash,
                input_image=image,
                layout=layout,
                raw=raw,
                audit=audit,
                output=output,
                output_png_sha256=output_png_sha,
                pipeline=pipeline,
                diagnostics=diagnostics,
            )
            atomic_write_json(record_path, record)
            processed += 1
        _validate_material(
            record=record,
            filename=name,
            input_sha256=expected_input_hash,
            input_image=image,
            layout=layout,
            raw=raw,
            output=output,
            output_path=output_path,
            pipeline=pipeline,
        )
        progress["completed_filenames"].append(name)
        atomic_write_json(run_path, progress)
        print(
            json.dumps(
                {
                    "event": "union_v2_submission_board",
                    "index": index,
                    "count": snapshot.file_count,
                    "filename": name,
                    "resumed": output_exists and record_exists,
                }
            ),
            flush=True,
        )
    _output_state(output_root, records_dir, snapshot.filenames, require_complete=True)
    records = [_load_json(records_dir / f"{name}.json") for name in snapshot.filenames]
    if build_official_input_snapshot(source, archive_source) != snapshot:
        raise RuntimeError("official input snapshot changed during Union-v2 inference")
    _mkdir_safe(zip_final.parent)
    _mkdir_safe(attestation_final.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{zip_final.name}.",
        dir=zip_final.parent,
    )
    os.close(descriptor)
    temporary_zip = Path(temporary_name)
    temporary_zip.unlink()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{attestation_final.name}.",
        dir=attestation_final.parent,
    )
    os.close(descriptor)
    temporary_attestation = Path(temporary_name)
    temporary_attestation.unlink()
    try:
        archive_sha = deterministic_submission_zip(
            output_root,
            list(snapshot.filenames),
            temporary_zip,
        )
        attestation = _build_attestation(
            snapshot=snapshot,
            pipeline=pipeline,
            archive_sha256=archive_sha,
            records=records,
        )
        atomic_write_json(temporary_attestation, attestation)
        from aiijc_puzzle.union_v2_submission_validation import (
            validate_union_v2_submission,
        )

        validation = validate_union_v2_submission(
            source_dir=source,
            source_archive=archive_source,
            output_dir=output_root,
            submission_zip=temporary_zip,
            attestation_path=temporary_attestation,
            validation_state_path=state_path,
            config_path=config.path,
            device_name=device_name,
            allow_nondeterministic_mps=allow_nondeterministic_mps,
        )
        if zip_final.exists() or attestation_final.exists():
            raise FileExistsError("final Union-v2 artifacts appeared during validation")
        os.replace(temporary_zip, zip_final)
        os.replace(temporary_attestation, attestation_final)
    finally:
        temporary_zip.unlink(missing_ok=True)
        temporary_attestation.unlink(missing_ok=True)
    progress["status"] = "COMPLETE"
    progress["processed_this_invocation"] = processed
    progress["resumed_this_invocation"] = resumed
    progress["submission_zip_sha256"] = sha256_file(zip_final)
    progress["attestation_sha256"] = sha256_file(attestation_final)
    atomic_write_json(run_path, progress)
    return {
        "status": "COMPLETE_VALIDATED",
        "file_count": snapshot.file_count,
        "processed": processed,
        "resumed": resumed,
        "run_digest": identity["run_digest"],
        "pipeline_digest": pipeline["pipeline_digest"],
        "output_dir": str(output_root),
        "submission_zip": str(zip_final),
        "submission_zip_sha256": sha256_file(zip_final),
        "attestation": str(attestation_final),
        "attestation_sha256": sha256_file(attestation_final),
        "elapsed_seconds": perf_counter() - started,
        "validation": validation,
    }


__all__ = [
    "ATTESTATION_SCHEMA",
    "DEFAULT_ATTESTATION",
    "DEFAULT_CONFIG",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_OUTPUT_ZIP",
    "DEFAULT_VALIDATION_STATE",
    "EXPECTED_POLICY",
    "FrozenUnionSubmissionConfig",
    "LoadedUnionV2Engine",
    "METHOD_STATUS",
    "PROOF_LIMITATION",
    "PROOF_SCOPE",
    "build_pipeline_contract",
    "inspect_union_v2_submission",
    "load_union_submission_config",
    "load_union_v2_engine",
    "run_union_v2_submission",
]

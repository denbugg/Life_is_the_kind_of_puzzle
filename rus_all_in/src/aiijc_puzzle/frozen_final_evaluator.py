"""Fail-closed evaluator for the frozen compliant h20x1 pipeline.

The evaluator has two deliberately separate phases:

1. every input is decoded and all three preregistered predictions are frozen;
2. only after a content-addressed commitment exists may paired targets be read.

For the holdout split an exclusive, write-once receipt is created after phase 1
and before phase 2.  A pre-existing receipt always refuses another holdout run.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import stat
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
from PIL import Image

from aiijc_puzzle.compliant_atlas_decoder import audit_raw_permutation
from aiijc_puzzle.legacy_upgrade import directional_scores, layout_digest, solve_buddies
from aiijc_puzzle.pixel_tails import apply_nlm_color
from aiijc_puzzle.postassembly_harmonizer import (
    LuminanceGainConfig,
    SeamGraphConfig,
    apply_luminance_gains,
    apply_rgb_offsets,
    seam_graph_luminance_gains,
    seam_graph_rgb_offsets,
)
from aiijc_puzzle.protocol import (
    EXPERIMENT_SUBSET_NAMESPACE,
    EXPERIMENT_SUBSET_SEED,
    GRID_SIZE,
    IMAGE_SIZE,
    SPLIT_ALGORITHM,
    TILE_COUNT,
    TILE_SIZE,
    assemble_tiles,
    compute_protocol_digest,
    contest_ssim,
    select_manifest_records,
    sha256_file,
    split_tiles,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FROZEN_CONFIG_PATH = PROJECT_ROOT / "configs" / "frozen_final_h20x1_v1.json"
RGB_CONFIG_PATH = PROJECT_ROOT / "configs" / "postassembly_rgb_offset_v1.json"
LUMA_CONFIG_PATH = PROJECT_ROOT / "configs" / "postassembly_luminance_gain_v1.json"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "interim" / "validation_manifest.json"
DEFAULT_INPUTS_DIR = PROJECT_ROOT / "data" / "raw" / "train" / "inputs"
DEFAULT_TARGETS_DIR = PROJECT_ROOT / "data" / "raw" / "train" / "targets"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "frozen-final-evaluations"
V1_CONFIG_SHA256 = "83443d7500ae98b6e4f33dd1ef26c2300e2a5228117d6d9cc7d6536c70c2e5e8"
V1_OUTPUT_ROOT = OUTPUT_ROOT / V1_CONFIG_SHA256
DEFAULT_CALIBRATION_REPORT = V1_OUTPUT_ROOT / "calibration-report.json"
DEFAULT_HOLDOUT_REPORT = V1_OUTPUT_ROOT / "holdout-report.json"
CALIBRATION_COMMITMENT = V1_OUTPUT_ROOT / "calibration-prediction-commitment.json"
HOLDOUT_COMMITMENT = V1_OUTPUT_ROOT / "holdout-prediction-commitment.json"
HOLDOUT_RECEIPT = V1_OUTPUT_ROOT / "HOLDOUT_OPENED.receipt.json"

MANIFEST_PROTOCOL_DIGEST = "2a9e3b74f7defa8c00846a05eb598fd263fd16c2787c70e77d3b7a4b585bfbf4"
RGB_CONFIG_SHA256 = "4adfd9b614e8556b7de5c1f527d759d15d29c0f74e20aa26ff87900dd773ec9a"
LUMA_CONFIG_SHA256 = "7488cad2ae7cc75792d6ff0ff2ea0a38fa778979083ffd5c161c857b68fd550f"

ARMS = (
    "raw_strict_assembly",
    "colored_nlm_h20x1_control",
    "rgb_luma_then_colored_nlm_h20x1_final",
)
FINAL_ARM = ARMS[2]
CONTROL_ARM = ARMS[1]
RAW_ARM = ARMS[0]
BOOTSTRAP_REPLICATES = 20_000
REPORT_SCHEMA = "aiijc-frozen-final-h20x1-evaluation-v1"
COMMITMENT_SCHEMA = "aiijc-frozen-final-h20x1-prediction-commitment-v1"
RECEIPT_SCHEMA = "aiijc-single-use-holdout-open-receipt-v1"


@dataclass(frozen=True)
class EvaluationContext:
    """Validated immutable inputs to one calibration or holdout evaluation."""

    mode: str
    config_path: Path
    config: Mapping[str, Any]
    config_sha256: str
    manifest_path: Path
    manifest: Mapping[str, Any]
    manifest_file_sha256: str
    records: tuple[Mapping[str, Any], ...]
    selection_digest: str
    source_sha256: Mapping[str, str]
    method_config_sha256: Mapping[str, str]


@dataclass(frozen=True)
class EvaluationArtifactPaths:
    """Config-addressed paths that cannot be redirected for holdout access."""

    root: Path
    calibration_report: Path
    calibration_commitment: Path
    holdout_report: Path
    holdout_commitment: Path
    holdout_receipt: Path


@dataclass(frozen=True)
class FrozenBoard:
    """One fully inferred board whose arrays cannot be mutated in phase 2."""

    record: Mapping[str, Any]
    layout: np.ndarray
    audit: Mapping[str, Any]
    predictions: Mapping[str, np.ndarray]
    prediction_sha256: Mapping[str, str]
    layout_sha256: str
    objective: float
    solver: str
    harmonizer_diagnostics: Mapping[str, Any]
    runtime_seconds: float


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _now_utc() -> str:
    return datetime.now(UTC).isoformat()


def names_digest(records: Sequence[Mapping[str, Any]]) -> str:
    """Digest an ordered record roster exactly as preregistered."""

    return hashlib.sha256(
        "\n".join(str(record["filename"]) for record in records).encode("utf-8")
    ).hexdigest()


def artifact_paths(config_sha256: str) -> EvaluationArtifactPaths:
    """Return fixed artifact locations derived only from the frozen config hash."""

    if (
        not isinstance(config_sha256, str)
        or len(config_sha256) != 64
        or any(character not in "0123456789abcdef" for character in config_sha256)
    ):
        raise ValueError("config_sha256 must be a lowercase SHA-256 digest")
    root = OUTPUT_ROOT / config_sha256
    return EvaluationArtifactPaths(
        root=root,
        calibration_report=root / "calibration-report.json",
        calibration_commitment=root / "calibration-prediction-commitment.json",
        holdout_report=root / "holdout-report.json",
        holdout_commitment=root / "holdout-prediction-commitment.json",
        holdout_receipt=root / "HOLDOUT_OPENED.receipt.json",
    )


def array_digest(value: np.ndarray) -> str:
    """Hash the exact uint8 RGB pixels of a frozen prediction."""

    array = np.asarray(value)
    if array.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or array.dtype != np.uint8:
        raise ValueError(f"invalid prediction for hashing: {array.dtype} {array.shape}")
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace a JSON artifact and fsync its contents."""

    resolved = path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{resolved.name}.", dir=resolved.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, resolved)
        directory_descriptor = os.open(resolved.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def create_immutable_receipt(path: Path, payload: Mapping[str, Any]) -> str:
    """Create a fixed single-use receipt without any overwrite code path.

    ``O_EXCL`` serialises concurrent attempts.  The file is never subsequently
    edited and is chmod'ed read-only after its bytes have reached stable storage.
    Any crash after creation intentionally burns the holdout rather than making
    partial target access repeatable.
    """

    resolved = path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(resolved, flags, 0o444)
    except FileExistsError as error:
        raise RuntimeError(
            f"holdout already opened; immutable receipt exists: {resolved}"
        ) from error
    contents = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(resolved, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        directory_descriptor = os.open(resolved.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        # Deliberately do not remove a partial receipt: fail closed after an
        # uncertain single-use transition.
        raise
    return hashlib.sha256(contents).hexdigest()


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{label} keys drifted: missing={missing}, extra={extra}")


def _validate_panel(section: Mapping[str, Any], expected_split: str, label: str) -> None:
    if section.get("split") != expected_split:
        raise ValueError(f"{label} must use the {expected_split} split")
    for key, allow_zero in (("offset", True), ("count", False)):
        value = section.get(key)
        minimum = 0 if allow_zero else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{label} {key} must be an integer >= {minimum}")
    if section["offset"] + section["count"] > 700:
        raise ValueError(f"{label} panel exceeds its frozen 700-record split")
    digest = section.get("filenames_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{label} filenames_sha256 is malformed")


def _validate_thresholds(
    values: Mapping[str, Any],
    expected_keys: set[str],
    label: str,
) -> None:
    _require_exact_keys(values, expected_keys, label)
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} {name} must be numeric")
        if not math.isfinite(float(value)):
            raise ValueError(f"{label} {name} must be finite")


def validate_frozen_config(config: Mapping[str, Any]) -> None:
    """Validate a frozen panel around the only supported h20x1 pipeline."""

    common_keys = {
        "schema_version",
        "frozen_at",
        "purpose",
        "pipeline",
        "calibration_confirmation",
        "single_use_holdout",
        "forbidden",
    }
    decision_keys = {"selection_basis", "decision"} & set(config)
    if len(decision_keys) != 1:
        raise ValueError("frozen config needs exactly one selection_basis or decision block")
    _require_exact_keys(config, common_keys | decision_keys, "frozen config")
    if config.get("schema_version") != 1:
        raise ValueError("only frozen config schema_version=1 is supported")
    if not isinstance(config.get("frozen_at"), str) or not config["frozen_at"]:
        raise ValueError("frozen_at must be a non-empty timestamp string")
    if not isinstance(config.get("purpose"), str) or not config["purpose"]:
        raise ValueError("purpose must be a non-empty string")
    decision_name = next(iter(decision_keys))
    decision = config.get(decision_name)
    if not isinstance(decision, Mapping) or not decision:
        raise ValueError(f"{decision_name} must be a non-empty mapping")
    if decision_name == "decision":
        _require_exact_keys(
            decision,
            {
                "aspirational_gate_result",
                "fallback_authority",
                "retuning_after_aspirational_failure",
                "pipeline_changed_from_aspirational_config",
            },
            "fallback decision",
        )
        if (
            decision["retuning_after_aspirational_failure"] is not False
            or decision["pipeline_changed_from_aspirational_config"] is not False
        ):
            raise ValueError("fallback config must preserve the aspirational pipeline")
        for key in ("aspirational_gate_result", "fallback_authority"):
            if not isinstance(decision[key], str) or not decision[key]:
                raise ValueError(f"fallback decision {key} must be a non-empty string")
    pipeline = config.get("pipeline")
    if not isinstance(pipeline, Mapping):
        raise ValueError("pipeline must be a mapping")
    _require_exact_keys(pipeline, {"layout", "postassembly", "restoration"}, "pipeline")
    expected_layout = {
        "edge_view": "bilateral",
        "solver": "best_buddies",
        "max_edges": 96,
        "atlas_weight": 0.0,
        "strict_permutation": True,
    }
    if pipeline.get("layout") != expected_layout:
        raise ValueError("frozen layout semantics drifted")
    expected_postassembly = {
        "rgb_seam_offsets": "configs/postassembly_rgb_offset_v1.json",
        "bounded_luminance_gains": "configs/postassembly_luminance_gain_v1.json",
        "order": "rgb_offsets_then_luminance_then_nlm",
    }
    if pipeline.get("postassembly") != expected_postassembly:
        raise ValueError("frozen postassembly semantics drifted")
    expected_restoration = {
        "name": "opencv_fast_nl_means_colored",
        "h": 20,
        "h_color": 20,
        "template_window_size": 7,
        "search_window_size": 21,
        "passes": 1,
    }
    if pipeline.get("restoration") != expected_restoration:
        raise ValueError("frozen restoration semantics drifted")

    calibration = config.get("calibration_confirmation")
    holdout = config.get("single_use_holdout")
    if not isinstance(calibration, Mapping) or not isinstance(holdout, Mapping):
        raise ValueError("calibration and holdout specifications must be mappings")
    calibration_base_keys = {"split", "offset", "count", "filenames_sha256", "gate"}
    calibration_evidence_keys = {
        "existing_report",
        "observed_before_this_config_was_frozen",
    }
    calibration_extra = set(calibration) - calibration_base_keys
    if calibration_extra not in (set(), calibration_evidence_keys):
        raise ValueError("calibration evidence fields are partial or unknown")
    _require_exact_keys(calibration, calibration_base_keys | calibration_extra, "calibration")
    holdout_keys = {
        "split",
        "offset",
        "count",
        "filenames_sha256",
        "arms",
        "success",
        "no_retuning_after_open",
    }
    if "fresh_relative_to_current_workspace_holdout_runs" in holdout:
        holdout_keys.add("fresh_relative_to_current_workspace_holdout_runs")
    _require_exact_keys(holdout, holdout_keys, "holdout")
    _validate_panel(calibration, "calibration", "calibration")
    _validate_panel(holdout, "holdout", "holdout")
    if calibration_extra:
        report_name = calibration["existing_report"]
        observed = calibration["observed_before_this_config_was_frozen"]
        if (
            not isinstance(report_name, str)
            or Path(report_name).is_absolute()
            or ".." in Path(report_name).parts
            or Path(report_name).suffix != ".json"
        ):
            raise ValueError("existing calibration report must be a safe relative JSON path")
        if not isinstance(observed, Mapping):
            raise ValueError("frozen observed calibration evidence must be a mapping")
        _require_exact_keys(
            observed,
            {
                "raw_mean_ssim",
                "same_h_without_harmonizer_mean_ssim",
                "final_mean_ssim",
                "final_gain_vs_control",
                "gain_ci95",
                "wins_ties_losses",
            },
            "observed calibration evidence",
        )
        scalar_names = (
            "raw_mean_ssim",
            "same_h_without_harmonizer_mean_ssim",
            "final_mean_ssim",
            "final_gain_vs_control",
        )
        if any(
            isinstance(observed[name], bool)
            or not isinstance(observed[name], (int, float))
            or not math.isfinite(float(observed[name]))
            for name in scalar_names
        ):
            raise ValueError("observed calibration scalar evidence is malformed")
        if not all(0 <= observed[name] <= 1 for name in scalar_names[:3]):
            raise ValueError("observed calibration SSIM evidence must be in [0, 1]")
        if not math.isclose(
            observed["final_mean_ssim"] - observed["same_h_without_harmonizer_mean_ssim"],
            observed["final_gain_vs_control"],
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("observed calibration gain does not reproduce")
        interval = observed["gain_ci95"]
        roster = observed["wins_ties_losses"]
        if (
            not isinstance(interval, list)
            or len(interval) != 2
            or not all(
                isinstance(value, (int, float)) and math.isfinite(value) for value in interval
            )
            or interval[0] > interval[1]
        ):
            raise ValueError("observed calibration CI is malformed")
        if (
            not isinstance(roster, list)
            or len(roster) != 3
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in roster
            )
            or sum(roster) != calibration["count"]
        ):
            raise ValueError("observed wins/ties/losses do not match calibration count")
    calibration_gate = calibration.get("gate")
    if not isinstance(calibration_gate, Mapping):
        raise ValueError("calibration gate must be a mapping")
    _validate_thresholds(
        calibration_gate,
        {
            "final_mean_ssim_min",
            "gain_vs_same_h_without_harmonizer_min",
            "gain_ci95_lower_min",
            "wins_min",
        },
        "calibration gate",
    )
    if not 0 <= calibration_gate["final_mean_ssim_min"] <= 1:
        raise ValueError("calibration final SSIM threshold must be in [0, 1]")
    if (
        isinstance(calibration_gate["wins_min"], bool)
        or not isinstance(calibration_gate["wins_min"], int)
        or not 0 <= calibration_gate["wins_min"] <= calibration["count"]
    ):
        raise ValueError("calibration wins_min must fit the selected count")
    if holdout["arms"] != list(ARMS) or holdout["no_retuning_after_open"] is not True:
        raise ValueError("holdout arms or no-retuning contract drifted")
    if (
        "fresh_relative_to_current_workspace_holdout_runs" in holdout
        and holdout["fresh_relative_to_current_workspace_holdout_runs"] is not True
    ):
        raise ValueError("a declared fresh holdout panel must be marked true")
    holdout_gate = holdout.get("success")
    if not isinstance(holdout_gate, Mapping):
        raise ValueError("holdout success gate must be a mapping")
    _validate_thresholds(
        holdout_gate,
        {"final_mean_ssim_min", "gain_vs_control_ci95_lower_min"},
        "holdout success gate",
    )
    if not 0 <= holdout_gate["final_mean_ssim_min"] <= 1:
        raise ValueError("holdout final SSIM threshold must be in [0, 1]")
    required_forbidden = {
        "targets_or_reference_images_during_inference",
        "test_reference_overrides",
        "population_or_constant_canvas_rendering",
        "tile_substitution_or_duplicate_use",
        "tile_warping_or_rotation",
        "multipass_nlm",
        "nlm_h_at_least_30",
    }
    if set(config.get("forbidden", ())) != required_forbidden:
        raise ValueError("forbidden-method roster drifted")


def _validate_method_configs() -> tuple[SeamGraphConfig, LuminanceGainConfig, dict[str, str]]:
    if sha256_file(RGB_CONFIG_PATH) != RGB_CONFIG_SHA256:
        raise ValueError("RGB harmonizer config hash mismatch")
    if sha256_file(LUMA_CONFIG_PATH) != LUMA_CONFIG_SHA256:
        raise ValueError("luminance harmonizer config hash mismatch")
    rgb = json.loads(RGB_CONFIG_PATH.read_text(encoding="utf-8"))
    luma = json.loads(LUMA_CONFIG_PATH.read_text(encoding="utf-8"))
    expected_origin = {
        "repository": "/Users/rusyalain/Documents/GitHub/pazzle_will_be_killed",
        "branch": "origin/таска-говно",
        "commit": "d6a82f82ceefa109ef706402712d03805bc9e880",
        "source_path": "source/src/puzzle_assembly/postassembly_harmonizer.py",
        "source_blob": "9d8d01c0f48d0e1473c1ff48285b06ab786a5dd8",
    }
    for label, value in (("rgb", rgb), ("luminance", luma)):
        if value.get("schema_version") != 1 or value.get("target_access") is not False:
            raise ValueError(f"{label} method provenance contract drifted")
        if value.get("origin") != expected_origin:
            raise ValueError(f"{label} historical source provenance drifted")
    rgb_method = dict(rgb["method"])
    luma_method = dict(luma["method"])
    if rgb_method.pop("global_gauge", None) != "per-channel median offset equals zero":
        raise ValueError("RGB global gauge drifted")
    if luma_method.pop("global_gauge", None) != "median log gain equals zero":
        raise ValueError("luminance global gauge drifted")
    rgb_config = SeamGraphConfig(**rgb_method)
    luma_config = LuminanceGainConfig(**luma_method)
    rgb_config.validate()
    luma_config.validate()
    return (
        rgb_config,
        luma_config,
        {
            str(RGB_CONFIG_PATH.relative_to(PROJECT_ROOT)): RGB_CONFIG_SHA256,
            str(LUMA_CONFIG_PATH.relative_to(PROJECT_ROOT)): LUMA_CONFIG_SHA256,
        },
    )


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate both digest and the exact frozen generic-only protocol."""

    if manifest.get("schema_version") != 1:
        raise ValueError("manifest schema_version mismatch")
    if manifest.get("protocol_digest") != compute_protocol_digest(manifest):
        raise ValueError("manifest self-digest mismatch")
    if manifest.get("protocol_digest") != MANIFEST_PROTOCOL_DIGEST:
        raise ValueError("manifest is not the preregistered validation manifest")
    protocol = manifest.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("manifest protocol must be a mapping")
    required_protocol = {
        "seed": 20260829,
        "expected_pairs": 7000,
        "counts": {"train": 5600, "calibration": 700, "holdout": 700},
        "split_algorithm": SPLIT_ALGORITHM,
        "metric": {
            "name": "skimage.metrics.structural_similarity",
            "channel_axis": 2,
            "data_range": 255,
            "win_size": 7,
        },
        "tiling": {
            "grid_rows": GRID_SIZE,
            "grid_columns": GRID_SIZE,
            "tile_height": TILE_SIZE,
            "tile_width": TILE_SIZE,
            "order": "row-major",
        },
        "digest": {
            "algorithm": "sha256",
            "scope": "canonical JSON of this manifest excluding protocol_digest",
        },
    }
    if protocol != required_protocol:
        raise ValueError("manifest protocol semantics drifted")
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != {"train", "calibration", "holdout"}:
        raise ValueError("manifest split roster drifted")
    seen: set[str] = set()
    for split, expected_count in required_protocol["counts"].items():
        records = splits[split]
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            raise ValueError(f"manifest split {split} is malformed")
        if len(records) != expected_count:
            raise ValueError(f"manifest split {split} has the wrong count")
        for record in records:
            if not isinstance(record, Mapping) or set(record) != {
                "filename",
                "input_sha256",
                "target_sha256",
            }:
                raise ValueError(f"malformed record in manifest split {split}")
            filename = record["filename"]
            if (
                not isinstance(filename, str)
                or Path(filename).name != filename
                or not filename.endswith(".png")
                or filename in seen
            ):
                raise ValueError(f"invalid or duplicate manifest filename: {filename!r}")
            seen.add(filename)
            for hash_key in ("input_sha256", "target_sha256"):
                digest = record[hash_key]
                if (
                    not isinstance(digest, str)
                    or len(digest) != 64
                    or any(character not in "0123456789abcdef" for character in digest)
                ):
                    raise ValueError(f"invalid {hash_key} for {filename}")
    if len(seen) != 7000:
        raise ValueError("manifest filenames are not globally disjoint")


def source_hashes() -> dict[str, str]:
    paths = (
        PROJECT_ROOT / "src" / "aiijc_puzzle" / "frozen_final_evaluator.py",
        PROJECT_ROOT / "scripts" / "run_frozen_final_evaluation.py",
        PROJECT_ROOT / "src" / "aiijc_puzzle" / "protocol.py",
        PROJECT_ROOT / "src" / "aiijc_puzzle" / "legacy_upgrade.py",
        PROJECT_ROOT / "src" / "aiijc_puzzle" / "compliant_atlas_decoder.py",
        PROJECT_ROOT / "src" / "aiijc_puzzle" / "pixel_tails.py",
        PROJECT_ROOT / "src" / "aiijc_puzzle" / "postassembly_harmonizer.py",
    )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"evaluation source is incomplete: {missing}")
    return {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in paths}


def load_context(
    mode: str,
    *,
    config_path: Path = FROZEN_CONFIG_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> EvaluationContext:
    """Load, pin and semantically validate every preregistered control."""

    if mode not in {"calibration", "holdout"}:
        raise ValueError("mode must be calibration or holdout")
    config_path = config_path.resolve()
    manifest_path = manifest_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"frozen config does not exist: {config_path}")
    config_hash = sha256_file(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_frozen_config(config)
    _, _, method_hashes = _validate_method_configs()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    manifest_hash = sha256_file(manifest_path)
    section_name = "calibration_confirmation" if mode == "calibration" else "single_use_holdout"
    section = config[section_name]
    panel = select_manifest_records(
        manifest,
        section["split"],
        limit=int(section["offset"]) + int(section["count"]),
        seed=EXPERIMENT_SUBSET_SEED,
        namespace=EXPERIMENT_SUBSET_NAMESPACE,
    )
    records = tuple(panel[int(section["offset"]) :])
    digest = names_digest(records)
    if len(records) != section["count"] or digest != section["filenames_sha256"]:
        raise ValueError("selected filenames do not match the frozen panel digest")
    return EvaluationContext(
        mode=mode,
        config_path=config_path,
        config=config,
        config_sha256=config_hash,
        manifest_path=manifest_path,
        manifest=manifest,
        manifest_file_sha256=manifest_hash,
        records=records,
        selection_digest=digest,
        source_sha256=source_hashes(),
        method_config_sha256=method_hashes,
    )


def load_rgb_verified(path: Path, expected_hash: str) -> np.ndarray:
    """Decode one content-addressed strict RGB 480x480 PNG."""

    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_hash:
        raise ValueError(f"content hash mismatch: {path}")
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        if image.format != "PNG" or image.mode != "RGB" or image.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"expected strict RGB 480x480 PNG: {path}")
        return np.asarray(image, dtype=np.uint8)


def infer_three_arms(dirty: np.ndarray) -> dict[str, Any]:
    """Infer exactly the three preregistered target-blind prediction arms."""

    started = perf_counter()
    rgb_config, luma_config, _ = _validate_method_configs()
    input_tiles = split_tiles(dirty)
    right, down = directional_scores(input_tiles, views=("bilateral",))["bilateral"]
    solved = solve_buddies(right, down, max_edges=96)
    layout = np.asarray(solved.layout, dtype=np.int32)
    raw = assemble_tiles(input_tiles[layout])
    audit = audit_raw_permutation(
        dirty,
        raw,
        layout,
        restoration_applied_after_audit=True,
    )
    if not audit.passed:
        raise RuntimeError(f"strict permutation audit failed: {audit.as_dict()}")

    ordered = split_tiles(raw)
    offsets, rgb_diagnostics = seam_graph_rgb_offsets(ordered, rgb_config)
    rgb_tiles = apply_rgb_offsets(ordered, offsets)
    gains, luma_diagnostics = seam_graph_luminance_gains(rgb_tiles, luma_config)
    rgb_luma = assemble_tiles(apply_luminance_gains(rgb_tiles, gains))
    control = apply_nlm_color(raw, h=20).image
    final = apply_nlm_color(rgb_luma, h=20).image
    predictions = {
        RAW_ARM: raw,
        CONTROL_ARM: control,
        FINAL_ARM: final,
    }
    if tuple(predictions) != ARMS:
        raise RuntimeError("prediction arm roster drifted")
    return {
        "layout": layout,
        "audit": audit.as_dict(),
        "predictions": predictions,
        "objective": float(solved.objective),
        "solver": solved.solver,
        "harmonizer_diagnostics": {
            "rgb_seam_offsets": rgb_diagnostics,
            "bounded_luminance_gains": luma_diagnostics,
        },
        "runtime_seconds": perf_counter() - started,
    }


def freeze_all_predictions(
    context: EvaluationContext,
    inputs_dir: Path,
    *,
    image_loader: Callable[[Path, str], np.ndarray] = load_rgb_verified,
    infer: Callable[[np.ndarray], Mapping[str, Any]] = infer_three_arms,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[FrozenBoard, ...]:
    """Finish all inference and audits before returning to the scoring phase."""

    boards: list[FrozenBoard] = []
    for index, record in enumerate(context.records, start=1):
        filename = str(record["filename"])
        dirty = image_loader(inputs_dir / filename, str(record["input_sha256"]))
        inference = infer(dirty)
        raw_predictions = inference.get("predictions")
        if not isinstance(raw_predictions, Mapping) or tuple(raw_predictions) != ARMS:
            raise RuntimeError("inference returned anything other than the three frozen arms")
        if (
            not isinstance(inference.get("audit"), Mapping)
            or inference["audit"].get("passed") is not True
        ):
            raise RuntimeError(f"permutation audit failed for {filename}")
        raw_layout = np.asarray(inference["layout"])
        if not np.issubdtype(raw_layout.dtype, np.integer):
            raise RuntimeError(f"layout dtype is not integral for {filename}")
        layout = raw_layout.astype(np.int32, copy=True)
        if layout.shape != (TILE_COUNT,) or not np.array_equal(
            np.sort(layout), np.arange(TILE_COUNT)
        ):
            raise RuntimeError(f"layout is not a strict permutation for {filename}")
        layout.setflags(write=False)
        predictions: dict[str, np.ndarray] = {}
        hashes: dict[str, str] = {}
        for arm in ARMS:
            source_prediction = np.asarray(raw_predictions[arm])
            if (
                source_prediction.shape != (IMAGE_SIZE, IMAGE_SIZE, 3)
                or source_prediction.dtype != np.uint8
            ):
                raise RuntimeError(
                    f"arm {arm} returned invalid pixels for {filename}: "
                    f"{source_prediction.dtype} {source_prediction.shape}"
                )
            prediction = source_prediction.copy()
            hashes[arm] = array_digest(prediction)
            prediction.setflags(write=False)
            predictions[arm] = prediction
        board = FrozenBoard(
            record=dict(record),
            layout=layout,
            audit=dict(inference["audit"]),
            predictions=predictions,
            prediction_sha256=hashes,
            layout_sha256=layout_digest(layout),
            objective=float(inference["objective"]),
            solver=str(inference["solver"]),
            harmonizer_diagnostics=dict(inference.get("harmonizer_diagnostics", {})),
            runtime_seconds=float(inference.get("runtime_seconds", 0.0)),
        )
        boards.append(board)
        if progress is not None:
            progress(
                {
                    "phase": "target_blind_freeze",
                    "done": index,
                    "total": len(context.records),
                    "filename": filename,
                }
            )
    return tuple(boards)


def build_commitment(context: EvaluationContext, boards: Sequence[FrozenBoard]) -> dict[str, Any]:
    """Build a target-free commitment to every layout, audit and prediction."""

    if len(boards) != len(context.records):
        raise ValueError("prediction commitment board count mismatch")
    per_board = []
    for record, board in zip(context.records, boards, strict=True):
        if board.record["filename"] != record["filename"]:
            raise ValueError("prediction commitment order drifted")
        per_board.append(
            {
                "filename": record["filename"],
                "input_sha256": record["input_sha256"],
                "tile_at_position": board.layout.tolist(),
                "layout_sha256": board.layout_sha256,
                "permutation_audit": board.audit,
                "prediction_sha256": board.prediction_sha256,
                "solver": board.solver,
                "objective": board.objective,
                "harmonizer_diagnostics": board.harmonizer_diagnostics,
                "inference_runtime_seconds": board.runtime_seconds,
            }
        )
    aggregate_digest = hashlib.sha256(
        "\n".join(
            " ".join(board.prediction_sha256[arm] for arm in ARMS) for board in boards
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema": COMMITMENT_SCHEMA,
        "created_at_utc": _now_utc(),
        "mode": context.mode,
        "count": len(boards),
        "config_sha256": context.config_sha256,
        "manifest_protocol_digest": context.manifest["protocol_digest"],
        "manifest_file_sha256": context.manifest_file_sha256,
        "selection_namespace": EXPERIMENT_SUBSET_NAMESPACE,
        "selection_seed": EXPERIMENT_SUBSET_SEED,
        "selection_digest": context.selection_digest,
        "arms": list(ARMS),
        "source_sha256": dict(context.source_sha256),
        "method_config_sha256": dict(context.method_config_sha256),
        "contract": {
            "target_paths_opened": False,
            "all_predictions_frozen": True,
            "all_permutation_audits_passed": all(
                board.audit.get("passed") is True for board in boards
            ),
            "frozen_prediction_aggregate_sha256": aggregate_digest,
        },
        "per_board": per_board,
    }


def paired_bootstrap(values: Sequence[float]) -> tuple[float, float]:
    """Return the frozen deterministic percentile interval for a paired mean."""

    differences = np.asarray(values, dtype=np.float64)
    if differences.ndim != 1 or len(differences) == 0 or not np.isfinite(differences).all():
        raise ValueError("paired bootstrap needs a non-empty finite vector")
    rng = np.random.default_rng(EXPERIMENT_SUBSET_SEED)
    chunks: list[np.ndarray] = []
    remaining = BOOTSTRAP_REPLICATES
    while remaining:
        count = min(remaining, 4096)
        indices = rng.integers(0, len(differences), size=(count, len(differences)))
        chunks.append(differences[indices].mean(axis=1))
        remaining -= count
    low, high = np.quantile(np.concatenate(chunks), (0.025, 0.975))
    return float(low), float(high)


def aggregate_scores(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Aggregate arm scores and every preregistered paired contrast."""

    if not rows:
        raise ValueError("cannot aggregate an empty evaluation")
    summary: dict[str, Any] = {}
    for arm in ARMS:
        scores = np.asarray([row["ssim"][arm] for row in rows], dtype=np.float64)
        if not np.isfinite(scores).all():
            raise ValueError(f"non-finite score in arm {arm}")
        summary[arm] = {
            "mean_ssim": float(scores.mean()),
            "std_ssim": float(scores.std()),
            "min_ssim": float(scores.min()),
            "max_ssim": float(scores.max()),
            "count": len(scores),
        }
    comparisons: dict[str, Any] = {}
    for numerator, denominator in (
        (CONTROL_ARM, RAW_ARM),
        (FINAL_ARM, RAW_ARM),
        (FINAL_ARM, CONTROL_ARM),
    ):
        difference = np.asarray(
            [row["ssim"][numerator] - row["ssim"][denominator] for row in rows],
            dtype=np.float64,
        )
        comparisons[f"{numerator}__minus__{denominator}"] = {
            "numerator": numerator,
            "denominator": denominator,
            "mean_gain": float(difference.mean()),
            "gain_ci95": list(paired_bootstrap(difference)),
            "wins_ties_losses": [
                int(np.sum(difference > 0)),
                int(np.sum(difference == 0)),
                int(np.sum(difference < 0)),
            ],
            "count": len(difference),
        }
    return summary, comparisons


def evaluate_gate(
    mode: str,
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    comparisons: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate only the gate that was frozen for the selected split."""

    comparison = comparisons[f"{FINAL_ARM}__minus__{CONTROL_ARM}"]
    final_mean = float(summary[FINAL_ARM]["mean_ssim"])
    mean_gain = float(comparison["mean_gain"])
    ci_low = float(comparison["gain_ci95"][0])
    wins = int(comparison["wins_ties_losses"][0])
    if mode == "calibration":
        thresholds = config["calibration_confirmation"]["gate"]
        actuals = {
            "final_mean_ssim_min": final_mean,
            "gain_vs_same_h_without_harmonizer_min": mean_gain,
            "gain_ci95_lower_min": ci_low,
            "wins_min": wins,
        }
    elif mode == "holdout":
        thresholds = config["single_use_holdout"]["success"]
        actuals = {
            "final_mean_ssim_min": final_mean,
            "gain_vs_control_ci95_lower_min": ci_low,
        }
    else:
        raise ValueError("unknown gate mode")
    checks = {
        name: {
            "actual": actuals[name],
            "minimum": minimum,
            "passed": bool(actuals[name] >= minimum),
        }
        for name, minimum in thresholds.items()
    }
    return {"checks": checks, "all_passed": all(check["passed"] for check in checks.values())}


def score_frozen_predictions(
    context: EvaluationContext,
    boards: Sequence[FrozenBoard],
    targets_dir: Path,
    *,
    image_loader: Callable[[Path, str], np.ndarray] = load_rgb_verified,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Open paired targets only after the caller has committed all boards."""

    if len(boards) != len(context.records):
        raise ValueError("frozen board count mismatch")
    rows: list[dict[str, Any]] = []
    for index, (record, board) in enumerate(zip(context.records, boards, strict=True), start=1):
        if board.record["filename"] != record["filename"]:
            raise ValueError("frozen board order mismatch")
        filename = str(record["filename"])
        target = image_loader(targets_dir / filename, str(record["target_sha256"]))
        scores = {arm: contest_ssim(target, board.predictions[arm]) for arm in ARMS}
        if tuple(scores) != ARMS or any(not math.isfinite(score) for score in scores.values()):
            raise RuntimeError("score roster is incomplete or non-finite")
        rows.append(
            {
                "filename": filename,
                "input_sha256": record["input_sha256"],
                "target_sha256": record["target_sha256"],
                "all_predictions_frozen_before_any_target_decode": True,
                "tile_at_position": board.layout.tolist(),
                "layout_sha256": board.layout_sha256,
                "permutation_audit": board.audit,
                "prediction_sha256": board.prediction_sha256,
                "solver": board.solver,
                "objective": board.objective,
                "harmonizer_diagnostics": board.harmonizer_diagnostics,
                "ssim": scores,
            }
        )
        if progress is not None:
            progress(
                {
                    "phase": "posthoc_target_score",
                    "done": index,
                    "total": len(context.records),
                    "filename": filename,
                    "final_ssim": scores[FINAL_ARM],
                }
            )
    return rows


def validate_calibration_prerequisite(
    path: Path,
    calibration_context: EvaluationContext,
) -> dict[str, Any]:
    """Recompute and validate a same-code passing calibration report."""

    if not path.is_file():
        raise RuntimeError(f"holdout requires a completed calibration report: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    expected_scalars = {
        "schema": REPORT_SCHEMA,
        "status": "completed_gate_passed",
        "mode": "calibration",
        "count": len(calibration_context.records),
        "config": str(calibration_context.config_path),
        "config_sha256": calibration_context.config_sha256,
        "manifest": str(calibration_context.manifest_path),
        "manifest_protocol_digest": calibration_context.manifest["protocol_digest"],
        "manifest_file_sha256": calibration_context.manifest_file_sha256,
        "selection_digest": calibration_context.selection_digest,
        "arms": list(ARMS),
    }
    for key, expected in expected_scalars.items():
        if report.get(key) != expected:
            raise RuntimeError(f"calibration prerequisite mismatch for {key}")
    if report.get("source_sha256") != dict(calibration_context.source_sha256):
        raise RuntimeError("calibration was not produced by the current evaluation source")
    if report.get("method_config_sha256") != dict(calibration_context.method_config_sha256):
        raise RuntimeError("calibration method config hashes drifted")
    if report.get("configuration") != calibration_context.config:
        raise RuntimeError("calibration embedded config drifted")
    paths = artifact_paths(calibration_context.config_sha256)
    prediction_contract = report.get("prediction_contract")
    if not isinstance(prediction_contract, Mapping):
        raise RuntimeError("calibration prediction contract is absent")
    expected_commitment = paths.calibration_commitment.resolve()
    if (
        prediction_contract.get("inference_target_access") is not False
        or prediction_contract.get("all_predictions_and_audits_frozen_before_any_target_decode")
        is not True
        or prediction_contract.get("holdout_access") is not False
        or prediction_contract.get("test_access") is not False
        or prediction_contract.get("prediction_commitment") != str(expected_commitment)
    ):
        raise RuntimeError("calibration prediction contract drifted")
    commitment_hash = prediction_contract.get("prediction_commitment_sha256")
    if not expected_commitment.is_file() or sha256_file(expected_commitment) != commitment_hash:
        raise RuntimeError("calibration prediction commitment is absent or changed")
    commitment = json.loads(expected_commitment.read_text(encoding="utf-8"))
    expected_commitment_scalars = {
        "schema": COMMITMENT_SCHEMA,
        "mode": "calibration",
        "count": len(calibration_context.records),
        "config_sha256": calibration_context.config_sha256,
        "manifest_protocol_digest": calibration_context.manifest["protocol_digest"],
        "manifest_file_sha256": calibration_context.manifest_file_sha256,
        "selection_digest": calibration_context.selection_digest,
        "arms": list(ARMS),
    }
    if any(commitment.get(key) != value for key, value in expected_commitment_scalars.items()):
        raise RuntimeError("calibration prediction commitment metadata drifted")
    if commitment.get("source_sha256") != dict(calibration_context.source_sha256):
        raise RuntimeError("calibration commitment source hashes drifted")
    if commitment.get("method_config_sha256") != dict(calibration_context.method_config_sha256):
        raise RuntimeError("calibration commitment method hashes drifted")
    commitment_contract = commitment.get("contract")
    if (
        not isinstance(commitment_contract, Mapping)
        or commitment_contract.get("target_paths_opened") is not False
        or commitment_contract.get("all_predictions_frozen") is not True
        or commitment_contract.get("all_permutation_audits_passed") is not True
    ):
        raise RuntimeError("calibration commitment phase contract drifted")
    rows = report.get("per_board")
    if not isinstance(rows, list) or len(rows) != len(calibration_context.records):
        raise RuntimeError("calibration per-board evidence is incomplete")
    expected_names = [str(record["filename"]) for record in calibration_context.records]
    if [row.get("filename") for row in rows] != expected_names:
        raise RuntimeError("calibration per-board roster drifted")
    commitment_rows = commitment.get("per_board")
    if not isinstance(commitment_rows, list) or len(commitment_rows) != len(rows):
        raise RuntimeError("calibration commitment per-board evidence is incomplete")
    for record, row, committed in zip(
        calibration_context.records,
        rows,
        commitment_rows,
        strict=True,
    ):
        if (
            row.get("input_sha256") != record["input_sha256"]
            or row.get("target_sha256") != record["target_sha256"]
        ):
            raise RuntimeError("calibration record hashes drifted")
        if row.get("all_predictions_frozen_before_any_target_decode") is not True:
            raise RuntimeError("calibration prediction-freeze evidence is absent")
        if (
            not isinstance(row.get("permutation_audit"), Mapping)
            or row["permutation_audit"].get("passed") is not True
        ):
            raise RuntimeError("calibration permutation audit is absent or failed")
        layout = row.get("tile_at_position")
        if not isinstance(layout, list) or sorted(layout) != list(range(TILE_COUNT)):
            raise RuntimeError("calibration layout is not a strict permutation")
        layout_array = np.asarray(layout, dtype=np.int32)
        if row.get("layout_sha256") != layout_digest(layout_array):
            raise RuntimeError("calibration layout digest does not reproduce")
        hashes = row.get("prediction_sha256")
        if not isinstance(hashes, Mapping) or set(hashes) != set(ARMS):
            raise RuntimeError("calibration prediction hash roster drifted")
        if any(
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for digest in hashes.values()
        ):
            raise RuntimeError("calibration prediction hash is malformed")
        if (
            committed.get("filename") != row["filename"]
            or committed.get("input_sha256") != row["input_sha256"]
            or committed.get("tile_at_position") != layout
            or committed.get("layout_sha256") != row["layout_sha256"]
            or committed.get("permutation_audit") != row["permutation_audit"]
            or committed.get("prediction_sha256") != hashes
        ):
            raise RuntimeError("calibration report differs from its target-blind commitment")
        scores = row.get("ssim")
        if not isinstance(scores, Mapping) or set(scores) != set(ARMS):
            raise RuntimeError("calibration score arm roster drifted")
        if any(
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or not -1 <= score <= 1
            for score in scores.values()
        ):
            raise RuntimeError("calibration score is malformed")
    summary, comparisons = aggregate_scores(rows)
    gate = evaluate_gate("calibration", calibration_context.config, summary, comparisons)
    if report.get("summary") != summary or report.get("paired_comparisons") != comparisons:
        raise RuntimeError("calibration aggregates do not reproduce from per-board scores")
    if report.get("preregistered_gate") != gate:
        raise RuntimeError("calibration gate does not reproduce")
    if gate["all_passed"] is not True:
        raise RuntimeError("calibration preregistered gate failed; holdout remains sealed")
    return report


def _contract_snapshot(context: EvaluationContext) -> dict[str, str]:
    return {
        "config": sha256_file(context.config_path),
        "manifest": sha256_file(context.manifest_path),
        **dict(context.method_config_sha256),
        **source_hashes(),
    }


def _assert_contract_unchanged(context: EvaluationContext, initial: Mapping[str, str]) -> None:
    current = _contract_snapshot(context)
    if current != initial:
        changed = sorted(
            key for key in set(initial) | set(current) if initial.get(key) != current.get(key)
        )
        raise RuntimeError(f"evaluation contract changed during the run: {changed}")


def run_evaluation(
    context: EvaluationContext,
    *,
    inputs_dir: Path = DEFAULT_INPUTS_DIR,
    targets_dir: Path = DEFAULT_TARGETS_DIR,
    report_path: Path | None = None,
    commitment_path: Path | None = None,
    allow_holdout: bool = False,
    calibration_report_path: Path | None = None,
    holdout_receipt_path: Path | None = None,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Execute the two-phase evaluation with a fail-closed holdout transition."""

    paths = artifact_paths(context.config_sha256)
    if context.mode == "calibration":
        report_path = report_path or paths.calibration_report
        commitment_path = commitment_path or paths.calibration_commitment
    else:
        report_path = report_path or paths.holdout_report
        commitment_path = commitment_path or paths.holdout_commitment
    calibration_report_path = calibration_report_path or paths.calibration_report
    holdout_receipt_path = holdout_receipt_path or paths.holdout_receipt

    if context.mode == "holdout":
        if not allow_holdout:
            raise RuntimeError("holdout requires explicit --allow-holdout")
        if holdout_receipt_path.resolve() != paths.holdout_receipt.resolve():
            raise RuntimeError(f"holdout receipt path is fixed: {paths.holdout_receipt}")
        if report_path.resolve() != paths.holdout_report.resolve():
            raise RuntimeError(f"holdout report path is fixed: {paths.holdout_report}")
        if commitment_path.resolve() != paths.holdout_commitment.resolve():
            raise RuntimeError(f"holdout commitment path is fixed: {paths.holdout_commitment}")
        if calibration_report_path.resolve() != paths.calibration_report.resolve():
            raise RuntimeError(
                f"holdout calibration prerequisite path is fixed: {paths.calibration_report}"
            )
        if holdout_receipt_path.exists():
            raise RuntimeError(
                f"holdout already opened; immutable receipt exists: {holdout_receipt_path}"
            )
        if report_path.exists():
            raise RuntimeError("holdout report exists without its required immutable receipt")
        calibration_context = load_context(
            "calibration",
            config_path=context.config_path,
            manifest_path=context.manifest_path,
        )
        calibration_report_sha256 = sha256_file(calibration_report_path)
        calibration_report = validate_calibration_prerequisite(
            calibration_report_path,
            calibration_context,
        )
        if sha256_file(calibration_report_path) != calibration_report_sha256:
            raise RuntimeError("calibration report changed while it was being validated")
    else:
        if allow_holdout:
            raise RuntimeError("--allow-holdout is invalid during calibration")
        calibration_report = None
        calibration_report_sha256 = None

    initial_contract = _contract_snapshot(context)
    started = perf_counter()
    boards = freeze_all_predictions(context, inputs_dir, progress=progress)
    commitment = build_commitment(context, boards)
    atomic_json(commitment_path, commitment)
    commitment_sha256 = sha256_file(commitment_path)
    _assert_contract_unchanged(context, initial_contract)

    receipt_sha256: str | None = None
    if context.mode == "holdout":
        if sha256_file(calibration_report_path) != calibration_report_sha256:
            raise RuntimeError("calibration report changed before holdout transition")
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "created_at_utc": _now_utc(),
            "single_use_transition": "created before first holdout target decode",
            "config_sha256": context.config_sha256,
            "manifest_protocol_digest": context.manifest["protocol_digest"],
            "selection_digest": context.selection_digest,
            "arms": list(ARMS),
            "prediction_commitment": str(commitment_path.resolve()),
            "prediction_commitment_sha256": commitment_sha256,
            "calibration_report": str(calibration_report_path.resolve()),
            "calibration_report_sha256": calibration_report_sha256,
            "calibration_gate_passed": calibration_report["preregistered_gate"]["all_passed"],
        }
        receipt_sha256 = create_immutable_receipt(holdout_receipt_path, receipt)

    rows = score_frozen_predictions(context, boards, targets_dir, progress=progress)
    _assert_contract_unchanged(context, initial_contract)
    if sha256_file(commitment_path) != commitment_sha256:
        raise RuntimeError("prediction commitment changed after target access")
    if context.mode == "holdout" and sha256_file(holdout_receipt_path) != receipt_sha256:
        raise RuntimeError("holdout receipt changed after creation")
    summary, comparisons = aggregate_scores(rows)
    gate = evaluate_gate(context.mode, context.config, summary, comparisons)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "completed_gate_passed" if gate["all_passed"] else "completed_gate_failed",
        "created_at_utc": _now_utc(),
        "mode": context.mode,
        "count": len(rows),
        "config": str(context.config_path),
        "config_sha256": context.config_sha256,
        "manifest": str(context.manifest_path),
        "manifest_protocol_digest": context.manifest["protocol_digest"],
        "manifest_file_sha256": context.manifest_file_sha256,
        "selection_namespace": EXPERIMENT_SUBSET_NAMESPACE,
        "selection_seed": EXPERIMENT_SUBSET_SEED,
        "selection_digest": context.selection_digest,
        "arms": list(ARMS),
        "prediction_contract": {
            "inference_target_access": False,
            "all_predictions_and_audits_frozen_before_any_target_decode": True,
            "prediction_commitment": str(commitment_path.resolve()),
            "prediction_commitment_sha256": commitment_sha256,
            "holdout_receipt": (
                str(holdout_receipt_path.resolve()) if context.mode == "holdout" else None
            ),
            "holdout_receipt_sha256": receipt_sha256,
            "holdout_access": context.mode == "holdout",
            "test_access": False,
        },
        "configuration": context.config,
        "source_sha256": dict(context.source_sha256),
        "method_config_sha256": dict(context.method_config_sha256),
        "runtime": {
            "python_pid": os.getpid(),
            "opencv_version": cv2.__version__,
            "numpy_version": np.__version__,
            "seconds": perf_counter() - started,
        },
        "compliance": {
            "exactly_three_preregistered_arms": True,
            "all_576_tiles_used_exactly_once": all(
                row["permutation_audit"]["passed"] for row in rows
            ),
            "raw_assembly_pixel_preserving": True,
            "restoration_after_layout_only": True,
            "harmonizers_target_blind": True,
            "spatial_warp_rotation_or_tile_substitution": False,
            "nlm_passes": 1,
            "nlm_h": 20,
        },
        "summary": summary,
        "paired_comparisons": comparisons,
        "preregistered_gate": gate,
        "per_board": rows,
    }
    atomic_json(report_path, report)
    return report


__all__ = [
    "ARMS",
    "CALIBRATION_COMMITMENT",
    "CONTROL_ARM",
    "DEFAULT_CALIBRATION_REPORT",
    "DEFAULT_HOLDOUT_REPORT",
    "FINAL_ARM",
    "FROZEN_CONFIG_PATH",
    "HOLDOUT_COMMITMENT",
    "HOLDOUT_RECEIPT",
    "RAW_ARM",
    "EvaluationArtifactPaths",
    "EvaluationContext",
    "FrozenBoard",
    "aggregate_scores",
    "artifact_paths",
    "build_commitment",
    "create_immutable_receipt",
    "evaluate_gate",
    "freeze_all_predictions",
    "infer_three_arms",
    "load_context",
    "names_digest",
    "run_evaluation",
    "score_frozen_predictions",
    "validate_calibration_prerequisite",
    "validate_frozen_config",
    "validate_manifest",
]

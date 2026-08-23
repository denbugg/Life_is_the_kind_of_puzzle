"""Data-free contracts for the frozen E25 CRS-v1 confirmation.

This module does not contain a dataset, archive, image, permutation, target,
feature or prediction loader.  It validates the already-public manifest
identity seal, the process-separated label-free barrier, and synthetic metric
rows.  Importing it performs no I/O and creates no directory.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from decimal import Decimal
from pathlib import Path, PureWindowsPath
from types import MappingProxyType
from typing import Any, Mapping, Sequence


# Importing this contract must not create either C: bytecode or the E25 root.
sys.dont_write_bytecode = True


class E25ContractError(RuntimeError):
    """The frozen E25 protocol or an immutable hand-off was violated."""


SCHEMA_VERSION = 1
PROTOCOL_SCHEMA = "pazzle-e25-crs-v1-source-group-confirmation-protocol-v1"
SOURCE_SEAL_SCHEMA = "pazzle-e25-crs-v1-source-manifest-premetric-seal-v1"
LABEL_FREE_COMMIT_SCHEMA = "pazzle-e25-crs-v1-label-free-scene-commit-v1"
LABEL_FREE_BARRIER_SCHEMA = "pazzle-e25-crs-v1-label-free-48-barrier-v1"
METRIC_BROKER_CONTRACT_SCHEMA = "pazzle-e25-crs-v1-metric-broker-contract-v1"
STRUCTURAL_REPORT_SCHEMA = "pazzle-e25-crs-v1-structural-report-v1"
CONFIRMATION_REPORT_SCHEMA = "pazzle-e25-crs-v1-confirmation-report-v1"

E25_IDS = (
    226, 262, 242, 123, 103, 231, 286, 296, 230, 134, 118, 110,
    239, 269, 146, 187, 183, 151, 148, 247, 191, 186, 193, 106,
    220, 274, 125, 117, 115, 265, 165, 257, 210, 213, 132, 143,
    152, 137, 177, 225, 113, 259, 101, 178, 202, 141, 273, 111,
)
E25_NAMES = tuple(f"img_{6700 + image:06d}.png" for image in E25_IDS)
E24_IDS = tuple(range(10, 18))
E25_CANARY_ID = 226
E25_NEWLINE_LIST_SHA256 = (
    "407a6326ceeec2e8cc78106b74c2f10c46a55143ea488a30f7bac66e2b373caa"
)
E25_CANONICAL_RECORDS_SHA256 = (
    "76e6b9431de41388e4aebef525ff4a5fd8354f789cf0a5913c1e29d8db148e2e"
)
SOURCE_GROUP_MANIFEST_PATH = Path(
    "E:/pazzle_work/rank96_e11_v4/source_groups_v4.json"
)
SOURCE_GROUP_MANIFEST_SHA256 = (
    "fa142c5f9c4fa17671b60d72b9acedff0eafcad4e77afac2b17a9649adfbfbd9"
)

STORAGE_ROOT = Path("E:/pazzle_work/posegraph_e25_confirmation")
FEATURE_CACHE_BYTES_MAX = 24 * 1024**3
ALL_ARTIFACT_BYTES_MAX = 48 * 1024**3
PEAK_RAM_BYTES_MAX = 16 * 1024**3
TOTAL_CPU_SECONDS_MAX = 48 * 60 * 60
GEOMETRY_HYPOTHESES_MAX_EACH = 450_000

STRUCTURAL_GATES: Mapping[str, float | int] = MappingProxyType(
    {
        "completed_integrity_scenes": 48,
        "proposed_precision_mean_min": 0.70,
        "proposed_precision_worst_min": 0.60,
        "true_relation_recall_mean_min": 0.65,
        "true_relation_recall_worst_min": 0.50,
        "exact_connected_coverage_mean_min": 0.50,
        "exact_connected_coverage_worst_min": 0.35,
        "mean_cycle_rank_ratio_min": 0.05,
        "geometry_hypotheses_max_each": GEOMETRY_HYPOTHESES_MAX_EACH,
    }
)
STAGED_GATES: Mapping[str, float | int] = MappingProxyType(
    {
        "solve_ssim_delta_mean_min": 0.003,
        "final_ssim_delta_mean_min": 0.002,
        "strict_positive_final_wins_min": 30,
        "worst_final_delta_min": -0.020,
        "neighbour_delta_mean_min": 0.005,
    }
)
RESOURCE_LIMITS: Mapping[str, int] = MappingProxyType(
    {
        "feature_cache_bytes_max": FEATURE_CACHE_BYTES_MAX,
        "all_artifact_bytes_max": ALL_ARTIFACT_BYTES_MAX,
        "peak_ram_bytes_max": PEAK_RAM_BYTES_MAX,
        "total_cpu_seconds_max": TOTAL_CPU_SECONDS_MAX,
    }
)

METRIC_BROKER_CONTRACT: Mapping[str, Any] = MappingProxyType(
    {
        "schema": METRIC_BROKER_CONTRACT_SCHEMA,
        "scene_ids": E25_IDS,
        "structural_label_phase": {
            "opens_only_after": "authenticated_complete_48_of_48_label_free_barrier",
            "allowed_scene_members": ("permutation",),
            "clean_target_opened": False,
            "output": "atomic_structural_report",
        },
        "staged_image_phase": {
            "opens_only_after": "authenticated_E25_structural_PASS",
            "allowed_scene_members": ("permutation", "clean_target"),
            "reads_predictions_and_boards": "already_committed_only",
            "output": "atomic_confirmation_report",
        },
        "training_or_rescoring": False,
        "artifact_selection": False,
        "writeback_to_label_free_cache": False,
        "structural_gates": dict(STRUCTURAL_GATES),
        "staged_gates": dict(STAGED_GATES),
        "diagnostics_can_rescue": False,
    }
)

E25_PROTOCOL: Mapping[str, Any] = MappingProxyType(
    {
        "schema": PROTOCOL_SCHEMA,
        "role": "one_shot_source_group_disjoint_confirmation",
        "ids": E25_IDS,
        "names": E25_NAMES,
        "newline_list_sha256": E25_NEWLINE_LIST_SHA256,
        "canonical_records_sha256": E25_CANONICAL_RECORDS_SHA256,
        "canary_id": E25_CANARY_ID,
        "canary_selection": "first_sealed_manifest_id_without_data_access",
        "orientation_degrees": (0,),
        "reflection": False,
        "feature_count": 227,
        "checkpoint": "exact_authenticated_e24_final_all8",
        "decoder": "exact_frozen_e24_crs_v1",
        "packer": "solve_components_from_scores_raw_rd_repair0",
        "restoration": "NLM10",
        "baseline": "exact_RR96",
        "structural_gates": dict(STRUCTURAL_GATES),
        "staged_gates": dict(STAGED_GATES),
        "resource_limits": dict(RESOURCE_LIMITS),
        "process_order": (
            "source_manifest_broker",
            "label_free_feature_inference_workers",
            "global_48_of_48_label_free_barrier",
            "trusted_structural_label_broker",
            "structural_pass_barrier",
            "trusted_staged_image_metric_broker",
        ),
        "real_workers": "sealed_pending_separate_review",
        "sweep": False,
        "rotation_or_reflection_search": False,
    }
)


def canonical_json_bytes(value: Any, *, newline: bool = True) -> bytes:
    """Encode exact finite canonical ASCII JSON."""

    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise E25ContractError("value is not canonical finite JSON") from exc
    if newline:
        text += "\n"
    return text.encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


PROTOCOL_SHA256 = sha256_bytes(canonical_json_bytes(dict(E25_PROTOCOL)))
METRIC_BROKER_CONTRACT_SHA256 = sha256_bytes(
    canonical_json_bytes(dict(METRIC_BROKER_CONTRACT))
)


def lower_sha256(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise E25ContractError(f"{label} must be a lowercase SHA256")
    return value


def _strict_e25_path(path: object, *, label: str) -> str:
    if type(path) is not str:
        raise E25ContractError(f"{label} path must be text")
    candidate = PureWindowsPath(path.replace("/", "\\"))
    root = PureWindowsPath(str(STORAGE_ROOT).replace("/", "\\"))
    if (
        candidate.drive.upper() != "E:"
        or not candidate.is_absolute()
        or candidate == root
        or root not in candidate.parents
        or ".." in candidate.parts
    ):
        raise E25ContractError(f"{label} must be a child of the frozen E25 root")
    return str(candidate)


def expected_name(image: int) -> str:
    if type(image) is not int or image not in E25_IDS:
        raise E25ContractError("image is outside the ordered E25 seal")
    return f"img_{6700 + image:06d}.png"


def _canonical_record_sha(records: Sequence[Mapping[str, str]]) -> str:
    return sha256_bytes(canonical_json_bytes(list(records), newline=False))


def validate_sealed_records(records: object) -> list[dict[str, str]]:
    """Validate only the 48 allowed manifest identity records."""

    if type(records) is not list or len(records) != 48:
        raise E25ContractError("E25 manifest projection must contain exactly 48 records")
    normalized: list[dict[str, str]] = []
    for index, item in enumerate(records):
        if type(item) is not dict or set(item) != {
            "name", "source_group", "target_sha256"
        }:
            raise E25ContractError("E25 manifest record field set drifted")
        if item["name"] != E25_NAMES[index]:
            raise E25ContractError("E25 manifest identity order drifted")
        if type(item["source_group"]) is not str or not item["source_group"]:
            raise E25ContractError("E25 source group is absent")
        normalized.append(
            {
                "name": item["name"],
                "source_group": item["source_group"],
                "target_sha256": lower_sha256(
                    item["target_sha256"], label="E25 target manifest SHA"
                ),
            }
        )
    if len({item["source_group"] for item in normalized}) != 48:
        raise E25ContractError("E25 records are not source-group distinct")
    names_body = "".join(f"{name}\n" for name in E25_NAMES).encode("ascii")
    if sha256_bytes(names_body) != E25_NEWLINE_LIST_SHA256:
        raise E25ContractError("frozen E25 newline-list literal is inconsistent")
    if _canonical_record_sha(normalized) != E25_CANONICAL_RECORDS_SHA256:
        raise E25ContractError("E25 canonical record SHA mismatch")
    return normalized


def project_and_validate_source_manifest(payload: object) -> list[dict[str, str]]:
    """Project the exact metadata manifest without opening any image member."""

    if type(payload) is not dict:
        raise E25ContractError("source-group manifest must be an object")
    files = payload.get("files")
    stats = payload.get("stats")
    split = payload.get("split")
    if (
        payload.get("schema_version") != 2
        or type(files) is not dict
        or len(files) != 7000
        or type(stats) is not dict
        or stats.get("files") != 7000
        or type(split) is not dict
        or split.get("train_count") != 6700
        or split.get("val_count") != 300
    ):
        raise E25ContractError("source-group manifest coverage/schema drifted")
    required_names = tuple(f"img_{index:06d}.png" for index in range(7000))
    if set(files) != set(required_names):
        raise E25ContractError("source-group manifest is not exact full coverage")

    def group(name: str) -> str:
        row = files.get(name)
        value = row.get("source_group") if type(row) is dict else None
        if type(value) is not str or not value:
            raise E25ContractError("source-group manifest has an invalid group")
        return value

    records: list[dict[str, str]] = []
    for name in E25_NAMES:
        row = files[name]
        digest = row.get("sha256") if type(row) is dict else None
        records.append(
            {
                "name": name,
                "source_group": group(name),
                "target_sha256": lower_sha256(
                    digest, label="source-manifest target SHA"
                ),
            }
        )
    normalized = validate_sealed_records(records)
    e25_groups = {item["source_group"] for item in normalized}
    train_groups = {group(name) for name in required_names[:6700]}
    tune_groups = {group(name) for name in required_names[6700:6800]}
    e24_groups = {group(f"img_{6700 + image:06d}.png") for image in E24_IDS}
    if e25_groups & train_groups:
        raise E25ContractError("E25 source group overlaps training")
    if e25_groups & tune_groups:
        raise E25ContractError("E25 source group overlaps validation IDs 0..99")
    if e25_groups & e24_groups:
        raise E25ContractError("E25 source group overlaps E24")
    eligible = split.get("eligible_confirmation")
    if type(eligible) is not list or not set(E25_NAMES).issubset(set(eligible)):
        raise E25ContractError("E25 identities are outside the frozen eligible pool")
    return normalized


def load_and_validate_source_manifest(
    path: str | os.PathLike[str] = SOURCE_GROUP_MANIFEST_PATH,
) -> list[dict[str, str]]:
    """Read the one pinned metadata JSON; never enumerate a data directory."""

    source = Path(path)
    if source.resolve(strict=False) != SOURCE_GROUP_MANIFEST_PATH.resolve(strict=False):
        raise E25ContractError("only the literal pinned source manifest path is allowed")
    if not source.is_file() or sha256_file(source) != SOURCE_GROUP_MANIFEST_SHA256:
        raise E25ContractError("pinned source-group manifest is absent or changed")
    try:
        payload = json.loads(source.read_bytes().decode("utf-8"))
    except Exception as exc:
        raise E25ContractError("pinned source-group manifest is unreadable") from exc
    return project_and_validate_source_manifest(payload)


_STRUCTURAL_ROW_KEYS = {
    "image",
    "provenance_ok",
    "input_ok",
    "query_canonical_onehot",
    "finite_output",
    "dsu_legal",
    "legal_origin",
    "orientation_degrees",
    "reflection",
    "proposal_denominator",
    "accepted_denominator",
    "true_relation_denominator",
    "proposed_precision",
    "true_relation_recall",
    "exact_connected_coverage",
    "cycle_rank_ratio",
    "geometry_hypotheses",
}


def _finite_number(value: object, *, label: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        raise E25ContractError(f"{label} must be finite numeric")
    return float(value)


def _validate_structural_row(row: object, index: int) -> dict[str, Any]:
    if type(row) is not dict or set(row) != _STRUCTURAL_ROW_KEYS:
        raise E25ContractError("E25 structural row field set drifted")
    if (
        row["image"] != E25_IDS[index]
        or row["orientation_degrees"] != 0
        or row["reflection"] is not False
    ):
        raise E25ContractError("E25 structural identity/integrity/orientation failed")
    for key in (
        "provenance_ok",
        "input_ok",
        "query_canonical_onehot",
        "finite_output",
        "dsu_legal",
        "legal_origin",
    ):
        if row[key] is not True:
            raise E25ContractError(f"E25 structural integrity check {key} failed")
    for key in ("proposal_denominator", "accepted_denominator", "true_relation_denominator"):
        if type(row[key]) is not int or row[key] <= 0:
            raise E25ContractError("E25 required structural denominator is not positive")
    if (
        type(row["geometry_hypotheses"]) is not int
        or row["geometry_hypotheses"] < 0
    ):
        raise E25ContractError("E25 geometry count is invalid")
    normalized = dict(row)
    for key in (
        "proposed_precision",
        "true_relation_recall",
        "exact_connected_coverage",
        "cycle_rank_ratio",
    ):
        normalized[key] = _finite_number(row[key], label=key)
        if normalized[key] < 0.0 or (
            key != "cycle_rank_ratio" and normalized[key] > 1.0
        ):
            raise E25ContractError(f"E25 {key} is outside its legal range")
    return normalized


def summarize_structural(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != 48:
        raise E25ContractError("E25 structural summary requires 48 rows")
    values = [_validate_structural_row(dict(row), index) for index, row in enumerate(rows)]

    def mean(key: str) -> float:
        # Canonical reports serialize finite decimals.  Average those exact
        # serialized values so an inclusive literal boundary such as 48 copies
        # of 0.70 cannot fail only because binary summation lands one ULP low.
        return float(sum((Decimal(str(row[key])) for row in values), Decimal(0)) / Decimal(48))

    return {
        "completed_integrity_scenes": 48,
        "positive_denominator_scenes": 48,
        "proposed_precision_mean": mean("proposed_precision"),
        "proposed_precision_worst": min(row["proposed_precision"] for row in values),
        "true_relation_recall_mean": mean("true_relation_recall"),
        "true_relation_recall_worst": min(row["true_relation_recall"] for row in values),
        "exact_connected_coverage_mean": mean("exact_connected_coverage"),
        "exact_connected_coverage_worst": min(row["exact_connected_coverage"] for row in values),
        "mean_cycle_rank_ratio": mean("cycle_rank_ratio"),
        "maximum_geometry_hypotheses": max(row["geometry_hypotheses"] for row in values),
    }


def structural_decision(summary: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "completed_integrity_scenes",
        "positive_denominator_scenes",
        "proposed_precision_mean",
        "proposed_precision_worst",
        "true_relation_recall_mean",
        "true_relation_recall_worst",
        "exact_connected_coverage_mean",
        "exact_connected_coverage_worst",
        "mean_cycle_rank_ratio",
        "maximum_geometry_hypotheses",
    }
    if type(summary) is not dict or set(summary) != expected:
        raise E25ContractError("E25 structural summary field set drifted")
    if (
        type(summary["completed_integrity_scenes"]) is not int
        or type(summary["positive_denominator_scenes"]) is not int
        or type(summary["maximum_geometry_hypotheses"]) is not int
    ):
        raise E25ContractError("E25 structural summary counts must be exact integers")
    numeric = {key: _finite_number(summary[key], label=key) for key in expected}
    checks = {
        "completed_integrity_scenes": summary["completed_integrity_scenes"] == 48,
        "positive_denominator_scenes": summary["positive_denominator_scenes"] == 48,
        "proposed_precision_mean": numeric["proposed_precision_mean"] >= 0.70,
        "proposed_precision_worst": numeric["proposed_precision_worst"] >= 0.60,
        "true_relation_recall_mean": numeric["true_relation_recall_mean"] >= 0.65,
        "true_relation_recall_worst": numeric["true_relation_recall_worst"] >= 0.50,
        "exact_connected_coverage_mean": numeric["exact_connected_coverage_mean"] >= 0.50,
        "exact_connected_coverage_worst": numeric["exact_connected_coverage_worst"] >= 0.35,
        "mean_cycle_rank_ratio": numeric["mean_cycle_rank_ratio"] >= 0.05,
        "geometry_hypotheses_cap": summary["maximum_geometry_hypotheses"] <= 450_000,
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "stage": "go_e25_staged_metrics" if passed else "kill_crs_v1",
        "checks": checks,
        "thresholds": dict(STRUCTURAL_GATES),
    }


_STAGED_ROW_KEYS = {
    "image",
    "provenance_ok",
    "paired_identity_ok",
    "rr96_verified",
    "frozen_candidate_ok",
    "orientation_degrees",
    "reflection",
    "solve_only_ssim_delta",
    "final_ssim_delta",
    "neighbour_delta",
}


def _validate_staged_row(row: object, index: int) -> dict[str, Any]:
    if type(row) is not dict or set(row) != _STAGED_ROW_KEYS:
        raise E25ContractError("E25 staged row field set drifted")
    if (
        row["image"] != E25_IDS[index]
        or row["orientation_degrees"] != 0
        or row["reflection"] is not False
    ):
        raise E25ContractError("E25 staged identity/integrity/orientation failed")
    for key in (
        "provenance_ok",
        "paired_identity_ok",
        "rr96_verified",
        "frozen_candidate_ok",
    ):
        if row[key] is not True:
            raise E25ContractError(f"E25 staged integrity check {key} failed")
    normalized = dict(row)
    for key in ("solve_only_ssim_delta", "final_ssim_delta", "neighbour_delta"):
        normalized[key] = _finite_number(row[key], label=key)
        if not -1.0 <= normalized[key] <= 1.0:
            raise E25ContractError(f"E25 {key} is outside the legal paired range")
    return normalized


def summarize_staged(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != 48:
        raise E25ContractError("E25 staged summary requires 48 rows")
    values = [_validate_staged_row(dict(row), index) for index, row in enumerate(rows)]

    def mean(key: str) -> float:
        return float(sum((Decimal(str(row[key])) for row in values), Decimal(0)) / Decimal(48))

    return {
        "completed_integrity_scenes": 48,
        "mean_solve_only_ssim_delta": mean("solve_only_ssim_delta"),
        "mean_final_ssim_delta": mean("final_ssim_delta"),
        "strict_positive_final_ssim_wins": sum(
            row["final_ssim_delta"] > 0.0 for row in values
        ),
        "worst_final_ssim_delta": min(row["final_ssim_delta"] for row in values),
        "mean_neighbour_delta": mean("neighbour_delta"),
    }


def staged_decision(summary: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "completed_integrity_scenes",
        "mean_solve_only_ssim_delta",
        "mean_final_ssim_delta",
        "strict_positive_final_ssim_wins",
        "worst_final_ssim_delta",
        "mean_neighbour_delta",
    }
    if type(summary) is not dict or set(summary) != expected:
        raise E25ContractError("E25 staged summary field set drifted")
    if (
        type(summary["completed_integrity_scenes"]) is not int
        or type(summary["strict_positive_final_ssim_wins"]) is not int
        or not 0 <= summary["strict_positive_final_ssim_wins"] <= 48
    ):
        raise E25ContractError("E25 staged summary counts must be exact integers")
    numeric = {key: _finite_number(summary[key], label=key) for key in expected}
    checks = {
        "completed_integrity_scenes": summary["completed_integrity_scenes"] == 48,
        "mean_solve_only_ssim_delta": numeric["mean_solve_only_ssim_delta"] >= 0.003,
        "mean_final_ssim_delta": numeric["mean_final_ssim_delta"] >= 0.002,
        "strict_positive_final_ssim_wins": summary["strict_positive_final_ssim_wins"] >= 30,
        "worst_final_ssim_delta": numeric["worst_final_ssim_delta"] >= -0.020,
        "mean_neighbour_delta": numeric["mean_neighbour_delta"] >= 0.005,
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "stage": "go_production_parity_replay" if passed else "kill_crs_v1",
        "checks": checks,
        "thresholds": dict(STAGED_GATES),
    }


_COMMIT_KEYS = {
    "schema",
    "schema_version",
    "image",
    "name",
    "source_group",
    "source_seal_sha256",
    "authority_sha256",
    "final_model_sha256",
    "input_receipt_path",
    "input_receipt_sha256",
    "feature_path",
    "feature_sha256",
    "feature_bytes",
    "prediction_path",
    "prediction_sha256",
    "prediction_bytes",
    "worker_receipt_path",
    "worker_receipt_sha256",
    "labels_targets_metrics_opened",
    "orientation_degrees",
    "reflection",
}


def validate_label_free_commit(
    record: object,
    *,
    index: int,
    sealed_record: Mapping[str, str],
    source_seal_sha256: str,
    authority_sha256: str,
    final_model_sha256: str,
) -> dict[str, Any]:
    if type(record) is not dict or set(record) != _COMMIT_KEYS:
        raise E25ContractError("E25 label-free commit field set drifted")
    image = E25_IDS[index]
    if (
        record["schema"] != LABEL_FREE_COMMIT_SCHEMA
        or record["schema_version"] != SCHEMA_VERSION
        or record["image"] != image
        or record["name"] != E25_NAMES[index]
        or record["source_group"] != sealed_record["source_group"]
        or record["source_seal_sha256"] != source_seal_sha256
        or record["authority_sha256"] != authority_sha256
        or record["final_model_sha256"] != final_model_sha256
        or record["labels_targets_metrics_opened"] is not False
        or record["orientation_degrees"] != 0
        or record["reflection"] is not False
    ):
        raise E25ContractError("E25 label-free commit identity/authority drifted")
    for key in (
        "source_seal_sha256", "authority_sha256", "final_model_sha256",
        "input_receipt_sha256", "feature_sha256", "prediction_sha256",
        "worker_receipt_sha256",
    ):
        lower_sha256(record[key], label=key)
    for key in ("input_receipt_path", "feature_path", "prediction_path", "worker_receipt_path"):
        _strict_e25_path(record[key], label=key)
    for key in ("feature_bytes", "prediction_bytes"):
        if type(record[key]) is not int or record[key] <= 0:
            raise E25ContractError(f"E25 {key} is not a positive integer")
    return dict(record)


def build_label_free_barrier(
    *,
    records: Sequence[Mapping[str, Any]],
    sealed_records: Sequence[Mapping[str, str]],
    source_seal_sha256: str,
    authority_sha256: str,
    final_model_sha256: str,
    canary_receipt_sha256: str,
    child_cpu_seconds: float,
    peak_rss_bytes: int,
    aggregate_artifact_bytes: int,
) -> dict[str, Any]:
    """Build a pure 48/48 barrier payload; this function performs no I/O."""

    source_sha = lower_sha256(source_seal_sha256, label="source seal SHA")
    authority_sha = lower_sha256(authority_sha256, label="authority SHA")
    model_sha = lower_sha256(final_model_sha256, label="final model SHA")
    canary_sha = lower_sha256(canary_receipt_sha256, label="canary receipt SHA")
    normalized_seal = validate_sealed_records(list(sealed_records))
    if len(records) != 48:
        raise E25ContractError("E25 label-free barrier requires 48 commits")
    commits = [
        validate_label_free_commit(
            dict(record),
            index=index,
            sealed_record=normalized_seal[index],
            source_seal_sha256=source_sha,
            authority_sha256=authority_sha,
            final_model_sha256=model_sha,
        )
        for index, record in enumerate(records)
    ]
    cpu = _finite_number(child_cpu_seconds, label="E25 child CPU seconds")
    if cpu < 0.0 or type(peak_rss_bytes) is not int or peak_rss_bytes < 0:
        raise E25ContractError("E25 resource observation is invalid")
    if type(aggregate_artifact_bytes) is not int or aggregate_artifact_bytes < 0:
        raise E25ContractError("E25 aggregate artifact bytes is invalid")
    feature_bytes = sum(record["feature_bytes"] for record in commits)
    checks = {
        "complete_ordered_48_of_48": True,
        "canary_226_passed": True,
        "labels_targets_metrics_unopened": all(
            record["labels_targets_metrics_opened"] is False for record in commits
        ),
        "feature_cache_at_most_24gib": feature_bytes <= FEATURE_CACHE_BYTES_MAX,
        "all_artifacts_at_most_48gib": aggregate_artifact_bytes <= ALL_ARTIFACT_BYTES_MAX,
        "peak_ram_at_most_16gib": peak_rss_bytes <= PEAK_RAM_BYTES_MAX,
        "total_cpu_at_most_48h": cpu <= TOTAL_CPU_SECONDS_MAX,
    }
    if not all(checks.values()):
        raise E25ContractError("E25 label-free barrier resource/integrity gate failed")
    return {
        "schema": LABEL_FREE_BARRIER_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "complete_48_of_48_label_free",
        "protocol_sha256": PROTOCOL_SHA256,
        "metric_broker_contract_sha256": METRIC_BROKER_CONTRACT_SHA256,
        "source_seal_sha256": source_sha,
        "authority_sha256": authority_sha,
        "final_model_sha256": model_sha,
        "canary": {"image": E25_CANARY_ID, "receipt_sha256": canary_sha, "status": "pass"},
        "completed_images": list(E25_IDS),
        "commits": commits,
        "resource": {
            "feature_bytes": feature_bytes,
            "aggregate_artifact_bytes": aggregate_artifact_bytes,
            "peak_rss_bytes": peak_rss_bytes,
            "child_cpu_seconds": cpu,
            "limits": dict(RESOURCE_LIMITS),
        },
        "checks": checks,
        "metric_broker_authorized": True,
        "labels_targets_metrics_opened": False,
    }


__all__ = (
    "ALL_ARTIFACT_BYTES_MAX",
    "CONFIRMATION_REPORT_SCHEMA",
    "E25_CANARY_ID",
    "E25_CANONICAL_RECORDS_SHA256",
    "E25ContractError",
    "E25_IDS",
    "E25_NAMES",
    "E25_NEWLINE_LIST_SHA256",
    "E25_PROTOCOL",
    "FEATURE_CACHE_BYTES_MAX",
    "LABEL_FREE_BARRIER_SCHEMA",
    "LABEL_FREE_COMMIT_SCHEMA",
    "METRIC_BROKER_CONTRACT",
    "METRIC_BROKER_CONTRACT_SHA256",
    "PEAK_RAM_BYTES_MAX",
    "PROTOCOL_SHA256",
    "RESOURCE_LIMITS",
    "SOURCE_GROUP_MANIFEST_PATH",
    "SOURCE_GROUP_MANIFEST_SHA256",
    "SOURCE_SEAL_SCHEMA",
    "STAGED_GATES",
    "STORAGE_ROOT",
    "STRUCTURAL_GATES",
    "TOTAL_CPU_SECONDS_MAX",
    "build_label_free_barrier",
    "canonical_json_bytes",
    "expected_name",
    "load_and_validate_source_manifest",
    "lower_sha256",
    "project_and_validate_source_manifest",
    "sha256_bytes",
    "sha256_file",
    "staged_decision",
    "structural_decision",
    "summarize_staged",
    "summarize_structural",
    "validate_label_free_commit",
    "validate_sealed_records",
)

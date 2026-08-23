"""PASS-only final all-eight CRS-v1 fit and immutable model authority.

This module is deliberately downstream of both E24 gates.  It opens no labels
until it has authenticated the generation-3 structural report, orchestration
receipt, and the exact staged SSIM/NLM PASS report.  It then fits exactly one
LightGBM LambdaRank model on scenes 10..17 using the frozen 227-column feature
schema, frozen balanced query-row weights, 256 trees, seed 1234, and no
validation or early stopping.

All generated files and runtime scratch paths are below the E24 root on E:.
Import and ``smoke`` are data-free.  The writer is create-or-byte-verify only;
it never overwrites an existing model or manifest.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence


STORAGE_ROOT = Path("E:/pazzle_work/posegraph_e24_selector")
RUNTIME_ROOT = STORAGE_ROOT / "tmp"
PYCACHE_ROOT = STORAGE_ROOT / "pycache"
sys.pycache_prefix = str(PYCACHE_ROOT)
for _key in (
    "PYTHONPYCACHEPREFIX",
    "TEMP",
    "TMP",
    "TMPDIR",
    "JOBLIB_TEMP_FOLDER",
    "LIGHTGBM_TMPDIR",
):
    os.environ[_key] = str(PYCACHE_ROOT if _key == "PYTHONPYCACHEPREFIX" else RUNTIME_ROOT)

import numpy as np

import e24_context_relation_selector as selector
import eval_e24_context_relation_selector as e24_eval
import eval_e24_staged_ssim_nlm as staged
import run_e24_context_relation_selector as e24_runner
import run_e24_staged_ssim_nlm as staged_runner


class E24FinalFitError(RuntimeError):
    """A final-fit authority, data, resource, or model invariant failed."""


ROOT = Path(__file__).resolve().parents[1]
FINAL_ROOT = STORAGE_ROOT / "final"
MODEL_PATH = FINAL_ROOT / "model.txt"
MANIFEST_PATH = FINAL_ROOT / "final_all8_manifest.json"
STAGED_REPORT_PATH = STORAGE_ROOT / "staged_v1" / "staged_ssim_nlm_report.json"

SCHEMA_VERSION = 1
STAGED_REPORT_SCHEMA = "pazzle-e24-crs-v1-staged-ssim-nlm-report-v1"
FINAL_MANIFEST_SCHEMA = "pazzle-e24-crs-v1-final-all8-model-v1"
FINAL_SEED = 1234
FINAL_TREES = 256
FINAL_FEATURES = 227

SOURCE_FILES = (
    Path(__file__).resolve(),
    ROOT / "tests/test_run_e24_final_all8.py",
    ROOT / "src/e24_context_relation_selector.py",
    ROOT / "src/eval_e24_context_relation_selector.py",
    ROOT / "src/run_e24_context_relation_selector.py",
    ROOT / "src/eval_e24_staged_ssim_nlm.py",
    ROOT / "src/run_e24_staged_ssim_nlm.py",
    ROOT / "tests/test_e24_context_relation_selector.py",
    ROOT / "tests/test_e24_context_relation_evaluator.py",
    ROOT / "tests/test_e24_staged_ssim_nlm.py",
    ROOT / "tests/test_run_e24_staged_ssim_nlm.py",
    ROOT / "E24_CONTEXT_RELATION_SELECTOR.md",
    ROOT / "autoresearch-runs/pazzle-solution-20260806/PLAN.md",
)
FINAL_OWN_SOURCE_FILES = (
    Path(__file__).resolve(),
    (ROOT / "tests/test_run_e24_final_all8.py").resolve(),
)


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return staged.canonical_json_bytes(dict(value))
    except Exception as exc:
        raise E24FinalFitError("canonical JSON serialization failed") from exc


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise E24FinalFitError(f"cannot hash required file: {path}") from exc
    return digest.hexdigest()


def _array_sha(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _lower_sha(value: object, *, label: str) -> str:
    try:
        return e24_eval._validate_lower_hex_sha256(value, label=label)
    except e24_eval.E24EvaluatorContractError as exc:
        raise E24FinalFitError(str(exc)) from exc


def _require_storage(path: Path, *, label: str) -> Path:
    try:
        return e24_eval._require_e24_storage_path(path, label=label)
    except e24_eval.E24EvaluatorContractError as exc:
        raise E24FinalFitError(str(exc)) from exc


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    target = _require_storage(path, label=label)
    try:
        raw = target.read_bytes()
        payload = json.loads(raw.decode("ascii"))
    except Exception as exc:
        raise E24FinalFitError(f"{label} is unreadable") from exc
    if type(payload) is not dict or raw != _canonical_bytes(payload):
        raise E24FinalFitError(f"{label} is not canonical JSON")
    return payload


def _source_hashes() -> dict[str, str]:
    records: dict[str, str] = {}
    for path in SOURCE_FILES:
        resolved = path.resolve()
        if not resolved.is_file():
            raise E24FinalFitError(f"final-fit source/protocol file is absent: {resolved}")
        records[str(resolved)] = _sha(resolved)
    return dict(sorted(records.items()))


def authenticate_prefit_source_snapshot(
    premetric_seal: Mapping[str, Any],
) -> dict[str, str]:
    """Bind current final-fit code/tests to the append-only staged source seal.

    This is intentionally a tiny independent check over the already-authenticated
    premetric seal.  It must run before a final trainer opens a label or builds a
    training batch.  The full source inventory is snapshotted at the same point
    and is later copied verbatim into the final manifest.
    """

    if type(premetric_seal) is not dict:
        raise E24FinalFitError("premetric seal payload is not an exact mapping")
    sealed = premetric_seal.get("sources")
    if type(sealed) is not dict:
        raise E24FinalFitError("premetric seal has no source SHA map")
    for path in FINAL_OWN_SOURCE_FILES:
        key = str(path.resolve())
        observed = _sha(path.resolve())
        expected = sealed.get(key)
        if (
            _lower_sha(expected, label=f"sealed source SHA for {path.name}")
            != observed
        ):
            raise E24FinalFitError(
                f"final-fit source changed after staged premetric seal: {path.name}"
            )
    snapshot = _source_hashes()
    for key, observed in snapshot.items():
        expected = sealed.get(key)
        if (
            expected is None
            or _lower_sha(expected, label=f"sealed source SHA for {Path(key).name}")
            != observed
        ):
            raise E24FinalFitError(
                f"final-fit dependency changed after staged premetric seal: {Path(key).name}"
            )
    return snapshot


def _verify_prefit_source_snapshot(authority: "FinalFitAuthority") -> dict[str, str]:
    """Rehash the exact sealed inventory at a final-fit capability boundary."""

    current = _source_hashes()
    if current != dict(authority.prefit_sources_sha256):
        raise E24FinalFitError("final-fit source snapshot drifted before label access/fit")
    return current


def _feature_names_sha256() -> str:
    return hashlib.sha256(
        _canonical_bytes({"feature_names": list(selector.FEATURE_NAMES)})
    ).hexdigest()


def final_learner_contract() -> dict[str, Any]:
    """Return the sole legal final learner configuration."""

    try:
        config = e24_eval.frozen_lightgbm_config(0)
    except e24_eval.E24EvaluatorContractError as exc:
        raise E24FinalFitError("frozen LightGBM configuration is unavailable") from exc
    contract = {
        "library": "lightgbm",
        "version": e24_eval.EXPECTED_LIGHTGBM_VERSION,
        "api": "selector.fit_lambdarank_fold_0_seed_projection",
        "config": config,
        "seed": FINAL_SEED,
        "trees": FINAL_TREES,
        "feature_count": FINAL_FEATURES,
        "ordered_feature_names_sha256": _feature_names_sha256(),
        "training_scenes": list(e24_eval.CALIBRATION_IDS),
        "weights": "scene_then_positive_offset_vs_NONE_category_then_query_balanced",
        "validation": False,
        "early_stopping": False,
        "sampling_or_mining": False,
    }
    if (
        len(selector.FEATURE_NAMES) != FINAL_FEATURES
        or config.get("n_estimators") != FINAL_TREES
        or config.get("random_state") != FINAL_SEED
        or config.get("data_random_seed") != FINAL_SEED
        or config.get("feature_fraction_seed") != FINAL_SEED
        or "early_stopping" in config
        or "callbacks" in config
    ):
        raise E24FinalFitError("final learner contract drifted from frozen CRS-v1")
    return json.loads(_canonical_bytes(contract))


FINAL_LEARNER_CONTRACT = final_learner_contract()
FINAL_LEARNER_CONTRACT_SHA256 = hashlib.sha256(
    _canonical_bytes(FINAL_LEARNER_CONTRACT)
).hexdigest()


@dataclass(frozen=True)
class FinalFitAuthority:
    upstream: staged_runner.AuthenticatedAuthority
    staged_report_path: Path
    staged_report_sha256: str
    staged_report: Mapping[str, Any]
    premetric_seal_sha256: str
    board_barrier_sha256: str
    board_commit_sha256: Mapping[int, str]
    metric_broker_contract_sha256: str
    prefit_sources_sha256: Mapping[str, str]


@dataclass(frozen=True)
class FinalTrainingBatch:
    table: selector.RelationFeatureTable
    relevance: np.ndarray
    row_weights: np.ndarray
    scene_row_offsets: Mapping[int, tuple[int, int]]


@dataclass(frozen=True)
class AuthenticatedFinalModel:
    authority: FinalFitAuthority
    manifest_path: Path
    manifest_sha256: str
    manifest: Mapping[str, Any]
    model_path: Path
    model_sha256: str
    predictor: Any


_STAGED_REPORT_KEYS = {
    "schema",
    "schema_version",
    "status",
    "stage",
    "staged_protocol_sha256",
    "metric_broker_contract_sha256",
    "ledger_sha256",
    "run_contract_sha256",
    "premetric_seal_sha256",
    "structural_report_sha256",
    "orchestration_receipt_sha256",
    "board_barrier_sha256",
    "board_commit_sha256",
    "rr96_verification",
    "rows",
    "summary",
    "decision",
    "e25_opened",
}


def validate_staged_pass_payload(
    payload: Mapping[str, Any],
    *,
    ledger_sha256: str,
    run_contract_sha256: str,
    structural_report_sha256: str,
    orchestration_receipt_sha256: str,
    premetric_seal_sha256: str,
    board_barrier_sha256: str,
    board_commit_sha256: Mapping[int, str],
) -> dict[str, Any]:
    """Recompute the staged decision and authenticate all routing hashes."""

    if type(payload) is not dict or set(payload) != _STAGED_REPORT_KEYS:
        raise E24FinalFitError("staged report field set drifted")
    expected_board = {
        str(image): _lower_sha(board_commit_sha256[image], label=f"scene {image} board SHA")
        for image in e24_eval.CALIBRATION_IDS
    }
    module_schema = getattr(staged, "STAGED_REPORT_SCHEMA", STAGED_REPORT_SCHEMA)
    if module_schema != STAGED_REPORT_SCHEMA:
        raise E24FinalFitError("staged report schema constant drifted")
    if (
        payload["schema"] != STAGED_REPORT_SCHEMA
        or payload["schema_version"] != SCHEMA_VERSION
        or payload["status"] != "complete"
        or payload["stage"] != "go_final_all8_fit"
        or payload["staged_protocol_sha256"] != staged.PROTOCOL_SHA256
        or payload["ledger_sha256"] != _lower_sha(ledger_sha256, label="ledger SHA")
        or payload["run_contract_sha256"]
        != _lower_sha(run_contract_sha256, label="run-contract SHA")
        or payload["structural_report_sha256"]
        != _lower_sha(structural_report_sha256, label="structural report SHA")
        or payload["orchestration_receipt_sha256"]
        != _lower_sha(orchestration_receipt_sha256, label="orchestration receipt SHA")
        or payload["premetric_seal_sha256"]
        != _lower_sha(premetric_seal_sha256, label="premetric seal SHA")
        or payload["board_barrier_sha256"]
        != _lower_sha(board_barrier_sha256, label="board barrier SHA")
        or payload["board_commit_sha256"] != expected_board
        or payload["e25_opened"] is not False
    ):
        raise E24FinalFitError("staged report is not the exact PASS authority")
    if (
        _lower_sha(
            payload["metric_broker_contract_sha256"],
            label="metric broker contract SHA",
        )
        != staged.METRIC_BROKER_CONTRACT_SHA256
    ):
        raise E24FinalFitError("staged metric-broker contract SHA drifted")
    rows = payload["rows"]
    if type(rows) is not list:
        raise E24FinalFitError("staged report rows are not a list")
    try:
        expected_summary = staged.summarize_staged(rows)
        expected_decision = staged.staged_decision(expected_summary)
        expected_rr96 = staged.rr96_verification_for_rows(rows)
    except staged.E24StagedContractError as exc:
        raise E24FinalFitError("staged report rows failed exact re-evaluation") from exc
    if (
        payload["summary"] != expected_summary
        or payload["decision"] != expected_decision
        or payload["rr96_verification"] != expected_rr96
        or expected_decision.get("passed") is not True
        or expected_decision.get("stage") != "go_final_all8_fit"
        or payload["stage"] != expected_decision["stage"]
        or type(expected_decision.get("checks")) is not dict
        or not all(value is True for value in expected_decision["checks"].values())
    ):
        raise E24FinalFitError("forged, failed, or drifted staged PASS")
    return dict(payload)


def _actual_board_commit_hashes() -> dict[int, str]:
    records: dict[int, str] = {}
    for image in e24_eval.CALIBRATION_IDS:
        try:
            _decode_artifact, _decode_commit, _board_artifact, board_commit = (
                staged_runner._scene_paths(image)
            )
        except Exception as exc:
            raise E24FinalFitError("staged board path API is unavailable") from exc
        expected = _require_storage(board_commit, label=f"scene {image} board commit")
        if not expected.is_file():
            raise E24FinalFitError(f"scene {image} board commit is absent")
        records[image] = _sha(expected)
    return records


def authenticate_final_fit_authority(
    ledger_path: Path = e24_runner.DEFAULT_LEDGER,
    ledger_sha256: str = staged_runner.EXPECTED_GENERATION3_LEDGER_SHA256,
) -> FinalFitAuthority:
    """Authenticate both E24 PASS gates before any all-eight label access."""

    try:
        upstream = staged_runner.authenticate_authority(ledger_path, ledger_sha256)
    except Exception as exc:
        raise E24FinalFitError("structural/orchestration authority failed") from exc
    report_path = _require_storage(STAGED_REPORT_PATH, label="staged SSIM/NLM report")
    if report_path.resolve() != staged_runner.REPORT_PATH.resolve() or not report_path.is_file():
        raise E24FinalFitError("final fit requires the literal staged report path")
    report_sha = _sha(report_path)
    payload = _load_json(report_path, label="staged SSIM/NLM report")
    premetric_sha = _lower_sha(payload.get("premetric_seal_sha256"), label="premetric seal SHA")
    try:
        premetric_seal = staged_runner.verify_premetric_seal(upstream, premetric_sha)
        staged_runner.verify_board_barrier(upstream, premetric_sha)
    except Exception as exc:
        raise E24FinalFitError("premetric seal/board barrier authentication failed") from exc
    barrier_path = _require_storage(staged_runner.BARRIER_PATH, label="board barrier")
    if not barrier_path.is_file():
        raise E24FinalFitError("board barrier is absent")
    barrier_sha = _sha(barrier_path)
    board_hashes = _actual_board_commit_hashes()
    try:
        validated_by_owner = staged.validate_staged_report(
            report_path,
            expected_ledger_sha256=upstream.ledger_sha256,
            expected_run_contract_sha256=upstream.ledger["run_contract_sha256"],
            expected_premetric_seal_sha256=premetric_sha,
            expected_structural_report_sha256=upstream.structural_report_sha256,
            expected_orchestration_receipt_sha256=upstream.orchestration_receipt_sha256,
            expected_board_barrier_sha256=barrier_sha,
            expected_board_commit_sha256={
                str(image): board_hashes[image]
                for image in e24_eval.CALIBRATION_IDS
            },
        )
    except Exception as exc:
        raise E24FinalFitError("owner staged-report validator rejected authority") from exc
    validated = validate_staged_pass_payload(
        payload,
        ledger_sha256=upstream.ledger_sha256,
        run_contract_sha256=upstream.ledger["run_contract_sha256"],
        structural_report_sha256=upstream.structural_report_sha256,
        orchestration_receipt_sha256=upstream.orchestration_receipt_sha256,
        premetric_seal_sha256=premetric_sha,
        board_barrier_sha256=barrier_sha,
        board_commit_sha256=board_hashes,
    )
    if validated_by_owner != validated:
        raise E24FinalFitError("independent and owner staged-report validators disagree")
    # This is the final authority operation: no label or training capability has
    # been called above.  Its immutable result is carried into fit/manifest.
    source_snapshot = authenticate_prefit_source_snapshot(premetric_seal)
    return FinalFitAuthority(
        upstream=upstream,
        staged_report_path=report_path,
        staged_report_sha256=report_sha,
        staged_report=validated,
        premetric_seal_sha256=premetric_sha,
        board_barrier_sha256=barrier_sha,
        board_commit_sha256=board_hashes,
        metric_broker_contract_sha256=validated["metric_broker_contract_sha256"],
        prefit_sources_sha256=source_snapshot,
    )


def _authority_record(authority: FinalFitAuthority) -> dict[str, Any]:
    upstream = authority.upstream
    return {
        "ledger": {"path": str(upstream.ledger_path), "sha256": upstream.ledger_sha256},
        "run_contract_sha256": upstream.ledger["run_contract_sha256"],
        "structural_report": {
            "path": str(e24_runner.STRUCTURAL_REPORT.resolve()),
            "sha256": upstream.structural_report_sha256,
        },
        "orchestration_receipt": {
            "path": str(e24_runner.ORCHESTRATION_RECEIPT_PATH.resolve()),
            "sha256": upstream.orchestration_receipt_sha256,
        },
        "premetric_seal": {
            "path": str(staged_runner.SEAL_PATH.resolve()),
            "sha256": authority.premetric_seal_sha256,
        },
        "board_barrier": {
            "path": str(staged_runner.BARRIER_PATH.resolve()),
            "sha256": authority.board_barrier_sha256,
        },
        "staged_report": {
            "path": str(authority.staged_report_path.resolve()),
            "sha256": authority.staged_report_sha256,
            "schema": STAGED_REPORT_SCHEMA,
        },
        "metric_broker_contract_sha256": authority.metric_broker_contract_sha256,
        "board_commit_sha256": {
            str(image): authority.board_commit_sha256[image]
            for image in e24_eval.CALIBRATION_IDS
        },
    }


def _feature_provenance(authority: FinalFitAuthority) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for image in e24_eval.CALIBRATION_IDS:
        manifest = authority.upstream.feature_manifests[image]
        feature_path, manifest_path = e24_runner._feature_paths(image)
        records[str(image)] = {
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": _sha(manifest_path),
            "feature_path": str(feature_path.resolve()),
            "feature_sha256": manifest["feature_file"]["sha256"],
            "rows": manifest["rows"],
            "queries": manifest["queries"],
        }
    return records


def load_consensus_all8_labels(
    authority: FinalFitAuthority,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Authenticate all three immutable fold-label copies for every scene."""

    labels: dict[int, np.ndarray] = {}
    provenance: dict[str, Any] = {}
    ledger_sha = authority.upstream.ledger_sha256
    run_sha = authority.upstream.ledger["run_contract_sha256"]
    for image in e24_eval.CALIBRATION_IDS:
        feature_manifest = authority.upstream.feature_manifests[image]
        copies: list[dict[str, Any]] = []
        values: list[np.ndarray] = []
        for fold in sorted(e24_eval.OOF_FOLDS):
            boundary = e24_eval.fold_boundary(fold)
            if image not in boundary.train_ids:
                continue
            try:
                manifest, manifest_sha = e24_runner._verify_fold_label_manifest(
                    fold,
                    image,
                    ledger_sha256=ledger_sha,
                    run_contract_sha256=run_sha,
                    rows=feature_manifest["rows"],
                    queries=feature_manifest["queries"],
                    feature_sha256=feature_manifest["feature_file"]["sha256"],
                    verify_label_file=True,
                )
                label_file = Path(manifest["label_file"]["path"]).resolve()
                one = e24_runner._load_exact_npy(
                    label_file,
                    shape=(feature_manifest["rows"],),
                    dtype=np.dtype(np.int8),
                    file_sha256=manifest["label_file"]["sha256"],
                )
            except Exception as exc:
                raise E24FinalFitError(
                    f"scene {image} fold {fold} label authentication failed"
                ) from exc
            values.append(one)
            _label_path, manifest_path = e24_runner._label_paths(fold, image)
            copies.append(
                {
                    "fold": fold,
                    "manifest_path": str(manifest_path.resolve()),
                    "manifest_sha256": manifest_sha,
                    "label_path": str(label_file),
                    "label_file_sha256": manifest["label_file"]["sha256"],
                    "label_array_sha256": _array_sha(one),
                }
            )
        if len(values) != 3 or any(not np.array_equal(values[0], item) for item in values[1:]):
            raise E24FinalFitError(
                f"scene {image} does not have three byte-equivalent fold-label copies"
            )
        frozen = np.ascontiguousarray(values[0], dtype=np.int8)
        frozen.setflags(write=False)
        labels[image] = frozen
        provenance[str(image)] = {
            "consensus_array_sha256": _array_sha(frozen),
            "copies": copies,
        }
    return labels, provenance


def build_final_training_batch(
    *,
    tables_by_scene: Mapping[int, selector.RelationFeatureTable],
    relevance_by_scene: Mapping[int, np.ndarray],
) -> FinalTrainingBatch:
    """Build exact all-eight data with the frozen float32 weight algebra."""

    expected_ids = set(e24_eval.CALIBRATION_IDS)
    if (
        set(tables_by_scene) != expected_ids
        or set(relevance_by_scene) != expected_ids
        or expected_ids.intersection(e24_eval.E25_SEALED_IDS)
    ):
        raise E24FinalFitError("final fit requires exactly E24 scenes 10..17")
    tables: list[selector.RelationFeatureTable] = []
    labels: list[np.ndarray] = []
    raw_weights: list[np.ndarray] = []
    scene_offsets: dict[int, tuple[int, int]] = {}
    cursor = 0
    try:
        for image in e24_eval.CALIBRATION_IDS:
            table = tables_by_scene[image]
            one_labels, categories = e24_eval._validate_scene_relevance(
                table, relevance_by_scene[image], image=image
            )
            tables.append(table)
            labels.append(one_labels)
            raw_weights.append(e24_eval._per_scene_balanced_weights(table, categories))
            scene_offsets[image] = (cursor, cursor + table.rows)
            cursor += table.rows
        combined = selector.concatenate_feature_tables(tables)
        combined_labels = np.ascontiguousarray(np.concatenate(labels), dtype=np.int8)
        independent = e24_eval._canonical_float32_fold_weights(raw_weights)
        core_weights = selector.balanced_query_row_weights(combined, combined_labels)
    except (e24_eval.E24EvaluatorContractError, selector.ContextRelationSelectorError) as exc:
        raise E24FinalFitError("final all-eight batch construction failed") from exc
    if (
        combined.features.shape[1] != FINAL_FEATURES
        or combined_labels.shape != (combined.rows,)
        or not np.array_equal(independent, core_weights)
        or not np.array_equal(
            combined.scene_offsets,
            np.asarray([0] + [scene_offsets[i][1] for i in e24_eval.CALIBRATION_IDS], dtype=np.int64),
        )
    ):
        raise E24FinalFitError("final all-eight feature/weight contract drifted")
    combined_labels.setflags(write=False)
    return FinalTrainingBatch(
        table=combined,
        relevance=combined_labels,
        row_weights=core_weights,
        scene_row_offsets=MappingProxyType(scene_offsets),
    )


def _load_feature_tables(
    authority: FinalFitAuthority,
) -> dict[int, selector.RelationFeatureTable]:
    tables: dict[int, selector.RelationFeatureTable] = {}
    for image in e24_eval.CALIBRATION_IDS:
        try:
            table, manifest = e24_runner._load_feature_artifact(
                image,
                authority.upstream.ledger_sha256,
                authority.upstream.ledger["run_contract_sha256"],
            )
        except Exception as exc:
            raise E24FinalFitError(f"scene {image} feature authentication failed") from exc
        if manifest != authority.upstream.feature_manifests[image]:
            raise E24FinalFitError(f"scene {image} feature manifest changed after PASS")
        tables[image] = table
    return tables


def _fit_model(batch: FinalTrainingBatch) -> Any:
    try:
        e24_eval.validate_lightgbm_runtime_version()
        return selector.fit_lambdarank(
            batch.table,
            batch.relevance,
            fold=0,
            row_weights=batch.row_weights,
        )
    except Exception as exc:
        raise E24FinalFitError("exact final LambdaRank fit failed") from exc


def _reload_model(
    path: Path,
    *,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
) -> tuple[Any, dict[str, Any]]:
    target = _require_storage(path, label="final all-eight model")
    if target.resolve() != MODEL_PATH.resolve() or not target.is_file():
        raise E24FinalFitError("final model is absent or at a non-canonical path")
    observed_sha = _sha(target)
    observed_bytes = target.stat().st_size
    if expected_sha256 is not None and observed_sha != _lower_sha(
        expected_sha256, label="final model SHA"
    ):
        raise E24FinalFitError("final model SHA mismatch")
    if expected_bytes is not None and observed_bytes != expected_bytes:
        raise E24FinalFitError("final model byte count mismatch")
    try:
        e24_eval.validate_lightgbm_runtime_version()
        from lightgbm import Booster

        predictor = Booster(model_file=str(target))
        canonical = predictor.model_to_string(num_iteration=FINAL_TREES).encode("utf-8")
        dump = predictor.dump_model()
    except Exception as exc:
        raise E24FinalFitError("committed final LightGBM model cannot be reloaded") from exc
    if (
        predictor.num_trees() != FINAL_TREES
        or predictor.current_iteration() != FINAL_TREES
        or predictor.num_feature() != FINAL_FEATURES
        or predictor.num_model_per_iteration() != 1
        or canonical != target.read_bytes()
        or type(dump) is not dict
        or dump.get("max_feature_idx") != FINAL_FEATURES - 1
        or dump.get("num_tree_per_iteration") != 1
        or not str(dump.get("objective", "")).startswith("lambdarank")
    ):
        raise E24FinalFitError("reloaded final model contract drifted")
    record = {
        "path": str(target.resolve()),
        "bytes": observed_bytes,
        "sha256": observed_sha,
        "num_trees": predictor.num_trees(),
        "current_iteration": predictor.current_iteration(),
        "num_features": predictor.num_feature(),
        "num_model_per_iteration": predictor.num_model_per_iteration(),
        "objective": "lambdarank",
        "canonical_reload_equal": True,
    }
    return predictor, record


_MANIFEST_KEYS = {
    "schema",
    "schema_version",
    "status",
    "authority",
    "sources_sha256",
    "training",
    "resource",
    "model",
    "checks",
    "e25_opened",
}


def validate_final_manifest_payload(
    payload: Mapping[str, Any],
    *,
    expected_authority: Mapping[str, Any],
    expected_sources: Mapping[str, str],
    expected_feature_provenance: Mapping[str, Any],
    expected_label_provenance: Mapping[str, Any],
    expected_model: Mapping[str, Any],
) -> dict[str, Any]:
    """Pure strict validator used by the E25 authority and synthetic tests."""

    if type(payload) is not dict or set(payload) != _MANIFEST_KEYS:
        raise E24FinalFitError("final model manifest field set drifted")
    training = payload.get("training")
    resource = payload.get("resource")
    checks = payload.get("checks")
    expected_training_keys = {
        "scene_ids",
        "feature_count",
        "ordered_feature_names_sha256",
        "learner_contract",
        "learner_contract_sha256",
        "lightgbm_version",
        "rows",
        "queries",
        "scene_row_offsets",
        "feature_provenance",
        "label_provenance",
        "relevance_sha256",
        "row_weights_sha256",
    }
    expected_resource_keys = {
        "fit_cpu_seconds",
        "cpu_seconds_max",
        "fit_wall_seconds",
        "peak_rss_bytes",
        "peak_rss_bytes_max",
    }
    expected_check_keys = {
        "exact_8_scenes",
        "exact_227_features",
        "exact_256_trees",
        "seed_1234",
        "no_validation_or_early_stopping",
        "fit_cpu_at_most_2h",
        "peak_rss_at_most_16gib",
        "aggregate_artifacts_at_most_8gib",
        "reloaded_model_canonical",
    }
    if (
        type(training) is not dict
        or set(training) != expected_training_keys
        or type(resource) is not dict
        or set(resource) != expected_resource_keys
        or type(checks) is not dict
        or set(checks) != expected_check_keys
    ):
        raise E24FinalFitError("final training/resource/check field set drifted")
    cpu = resource["fit_cpu_seconds"]
    wall = resource["fit_wall_seconds"]
    peak = resource["peak_rss_bytes"]
    if (
        type(cpu) not in {int, float}
        or type(wall) not in {int, float}
        or not math.isfinite(float(cpu))
        or not math.isfinite(float(wall))
        or float(cpu) < 0.0
        or float(wall) < 0.0
        or type(peak) is not int
        or peak < 0
        or resource["cpu_seconds_max"] != e24_eval.FINAL_FIT_CPU_SECONDS_MAX
        or resource["peak_rss_bytes_max"] != e24_eval.PEAK_RAM_BYTES_MAX
    ):
        raise E24FinalFitError("final fit resource record drifted")
    recomputed_checks = {
        "exact_8_scenes": training["scene_ids"] == list(e24_eval.CALIBRATION_IDS),
        "exact_227_features": training["feature_count"] == FINAL_FEATURES,
        "exact_256_trees": expected_model.get("num_trees") == FINAL_TREES,
        "seed_1234": training["learner_contract"] == FINAL_LEARNER_CONTRACT
        and training["learner_contract"].get("seed") == FINAL_SEED,
        "no_validation_or_early_stopping": training["learner_contract"].get("validation")
        is False
        and training["learner_contract"].get("early_stopping") is False,
        "fit_cpu_at_most_2h": float(cpu) <= e24_eval.FINAL_FIT_CPU_SECONDS_MAX,
        "peak_rss_at_most_16gib": peak <= e24_eval.PEAK_RAM_BYTES_MAX,
        "aggregate_artifacts_at_most_8gib": True,
        "reloaded_model_canonical": expected_model.get("canonical_reload_equal") is True,
    }
    if (
        payload["schema"] != FINAL_MANIFEST_SCHEMA
        or payload["schema_version"] != SCHEMA_VERSION
        or payload["status"] != "complete_pass_only_final_all8"
        or payload["authority"] != dict(expected_authority)
        or payload["sources_sha256"] != dict(expected_sources)
        or training["feature_count"] != len(selector.FEATURE_NAMES)
        or training["ordered_feature_names_sha256"] != _feature_names_sha256()
        or training["learner_contract"] != FINAL_LEARNER_CONTRACT
        or training["learner_contract_sha256"] != FINAL_LEARNER_CONTRACT_SHA256
        or training["lightgbm_version"] != e24_eval.EXPECTED_LIGHTGBM_VERSION
        or training["feature_provenance"] != dict(expected_feature_provenance)
        or training["label_provenance"] != dict(expected_label_provenance)
        or payload["model"] != dict(expected_model)
        or checks != recomputed_checks
        or not all(recomputed_checks.values())
        or payload["e25_opened"] is not False
    ):
        raise E24FinalFitError("final model manifest is not exact PASS-only authority")
    for key in ("relevance_sha256", "row_weights_sha256"):
        _lower_sha(training[key], label=key)
    if type(training["rows"]) is not int or training["rows"] <= 0:
        raise E24FinalFitError("final training row count is invalid")
    if type(training["queries"]) is not int or training["queries"] <= 0:
        raise E24FinalFitError("final training query count is invalid")
    offsets = training["scene_row_offsets"]
    if (
        type(offsets) is not dict
        or set(offsets) != {str(image) for image in e24_eval.CALIBRATION_IDS}
        or any(
            type(offsets[str(image)]) is not list
            or len(offsets[str(image)]) != 2
            or any(type(value) is not int for value in offsets[str(image)])
            for image in e24_eval.CALIBRATION_IDS
        )
    ):
        raise E24FinalFitError("final scene-row offsets drifted")
    ordered = [offsets[str(image)] for image in e24_eval.CALIBRATION_IDS]
    try:
        provenance_rows = [
            expected_feature_provenance[str(image)]["rows"]
            for image in e24_eval.CALIBRATION_IDS
        ]
        provenance_queries = [
            expected_feature_provenance[str(image)]["queries"]
            for image in e24_eval.CALIBRATION_IDS
        ]
    except Exception as exc:
        raise E24FinalFitError("feature provenance lacks exact row/query counts") from exc
    if (
        ordered[0][0] != 0
        or ordered[-1][1] != training["rows"]
        or any(a >= b for a, b in ordered)
        or any(ordered[index][1] != ordered[index + 1][0] for index in range(7))
        or any(type(value) is not int or value <= 0 for value in provenance_rows)
        or any(type(value) is not int or value <= 0 for value in provenance_queries)
        or training["rows"] != sum(provenance_rows)
        or training["queries"] != sum(provenance_queries)
        or any(
            stop - start != rows
            for (start, stop), rows in zip(ordered, provenance_rows)
        )
    ):
        raise E24FinalFitError("final scene-row offsets are not a contiguous partition")
    return dict(payload)


def _peak_rss_bytes() -> int:
    try:
        return int(e24_runner._peak_rss_bytes())
    except Exception as exc:
        raise E24FinalFitError("peak-RSS measurement failed") from exc


def run_final_fit(authority: FinalFitAuthority) -> dict[str, Any]:
    """Fit and commit the sole final model after authenticated staged PASS."""

    if type(authority) is not FinalFitAuthority:
        raise E24FinalFitError("final fit requires exact authenticated authority")
    # Rehash before any label access.  The exact result must equal the source
    # snapshot created by ``authenticate_final_fit_authority``.
    current_sources = _verify_prefit_source_snapshot(authority)
    _require_storage(MODEL_PATH, label="final model")
    _require_storage(MANIFEST_PATH, label="final model manifest")
    try:
        e24_eval.validate_e24_runtime_paths()
        e24_eval.validate_lightgbm_runtime_version()
        e24_runner.enforce_aggregate_artifact_caps(
            ledger_path=authority.upstream.ledger_path
        )
    except Exception as exc:
        raise E24FinalFitError("final fit runtime/storage preflight failed") from exc

    # A complete existing transaction is verification-only; never retrain it.
    if MANIFEST_PATH.exists():
        return dict(authenticate_final_model(authority=authority).manifest)

    started_cpu = time.process_time()
    started_wall = time.monotonic()
    labels, label_provenance = load_consensus_all8_labels(authority)
    feature_provenance = _feature_provenance(authority)
    tables = _load_feature_tables(authority)
    batch = build_final_training_batch(
        tables_by_scene=tables, relevance_by_scene=labels
    )
    del tables, labels
    gc.collect()
    # Labels/batch construction can be long.  Close the TOCTOU window by
    # authenticating source bytes again immediately before model.fit.
    _verify_prefit_source_snapshot(authority)
    model = _fit_model(batch)
    try:
        model_bytes = e24_runner._serialize_model_bytes(model)
    except Exception as exc:
        raise E24FinalFitError("final model serialization failed") from exc
    # A source mutation during fit cannot be published under the sealed hash.
    _verify_prefit_source_snapshot(authority)
    fit_cpu = float(time.process_time() - started_cpu)
    fit_wall = float(time.monotonic() - started_wall)
    peak = _peak_rss_bytes()
    if fit_cpu > e24_eval.FINAL_FIT_CPU_SECONDS_MAX:
        raise E24FinalFitError("final fit exceeded the frozen 2 CPU-hour cap")
    if peak > e24_eval.PEAK_RAM_BYTES_MAX:
        raise E24FinalFitError("final fit exceeded the frozen 16 GiB peak-RSS cap")
    try:
        e24_runner.enforce_aggregate_artifact_caps(
            ledger_path=authority.upstream.ledger_path,
            additional_total_bytes=0 if MODEL_PATH.exists() else len(model_bytes),
        )
        e24_eval._atomic_write_create_or_verify(MODEL_PATH, model_bytes)
    except Exception as exc:
        raise E24FinalFitError("final model create-once commit failed") from exc
    predictor, model_record = _reload_model(MODEL_PATH)
    del predictor, model
    gc.collect()

    training = {
        "scene_ids": list(e24_eval.CALIBRATION_IDS),
        "feature_count": FINAL_FEATURES,
        "ordered_feature_names_sha256": _feature_names_sha256(),
        "learner_contract": FINAL_LEARNER_CONTRACT,
        "learner_contract_sha256": FINAL_LEARNER_CONTRACT_SHA256,
        "lightgbm_version": e24_eval.EXPECTED_LIGHTGBM_VERSION,
        "rows": batch.table.rows,
        "queries": batch.table.queries,
        "scene_row_offsets": {
            str(image): list(batch.scene_row_offsets[image])
            for image in e24_eval.CALIBRATION_IDS
        },
        "feature_provenance": feature_provenance,
        "label_provenance": label_provenance,
        "relevance_sha256": _array_sha(batch.relevance),
        "row_weights_sha256": _array_sha(batch.row_weights),
    }
    resource = {
        "fit_cpu_seconds": fit_cpu,
        "cpu_seconds_max": e24_eval.FINAL_FIT_CPU_SECONDS_MAX,
        "fit_wall_seconds": fit_wall,
        "peak_rss_bytes": peak,
        "peak_rss_bytes_max": e24_eval.PEAK_RAM_BYTES_MAX,
    }
    checks = {
        "exact_8_scenes": True,
        "exact_227_features": True,
        "exact_256_trees": True,
        "seed_1234": True,
        "no_validation_or_early_stopping": True,
        "fit_cpu_at_most_2h": True,
        "peak_rss_at_most_16gib": True,
        "aggregate_artifacts_at_most_8gib": True,
        "reloaded_model_canonical": True,
    }
    payload = {
        "schema": FINAL_MANIFEST_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "complete_pass_only_final_all8",
        "authority": _authority_record(authority),
        "sources_sha256": current_sources,
        "training": training,
        "resource": resource,
        "model": model_record,
        "checks": checks,
        "e25_opened": False,
    }
    validate_final_manifest_payload(
        payload,
        expected_authority=_authority_record(authority),
        expected_sources=current_sources,
        expected_feature_provenance=feature_provenance,
        expected_label_provenance=label_provenance,
        expected_model=model_record,
    )
    manifest_bytes = _canonical_bytes(payload)
    try:
        e24_runner.enforce_aggregate_artifact_caps(
            ledger_path=authority.upstream.ledger_path,
            additional_total_bytes=0 if MANIFEST_PATH.exists() else len(manifest_bytes),
        )
        e24_eval._atomic_write_create_or_verify(MANIFEST_PATH, manifest_bytes)
        e24_runner.enforce_aggregate_artifact_caps(
            ledger_path=authority.upstream.ledger_path
        )
    except Exception as exc:
        raise E24FinalFitError("final manifest create-once commit failed") from exc
    return dict(payload)


def authenticate_final_model(
    ledger_path: Path = e24_runner.DEFAULT_LEDGER,
    ledger_sha256: str = staged_runner.EXPECTED_GENERATION3_LEDGER_SHA256,
    *,
    authority: FinalFitAuthority | None = None,
) -> AuthenticatedFinalModel:
    """Verify the immutable final authority without opening any E25 artifact."""

    trusted = (
        authenticate_final_fit_authority(ledger_path, ledger_sha256)
        if authority is None
        else authority
    )
    if type(trusted) is not FinalFitAuthority:
        raise E24FinalFitError("final model verifier requires exact authority")
    manifest_path = _require_storage(MANIFEST_PATH, label="final model manifest")
    if not manifest_path.is_file():
        raise E24FinalFitError("final model manifest is absent")
    payload = _load_json(manifest_path, label="final model manifest")
    model_record = payload.get("model")
    if type(model_record) is not dict or set(model_record) != {
        "path",
        "bytes",
        "sha256",
        "num_trees",
        "current_iteration",
        "num_features",
        "num_model_per_iteration",
        "objective",
        "canonical_reload_equal",
    }:
        raise E24FinalFitError("final model record field set drifted")
    predictor, observed_model = _reload_model(
        MODEL_PATH,
        expected_sha256=model_record.get("sha256"),
        expected_bytes=model_record.get("bytes"),
    )
    _labels, label_provenance = load_consensus_all8_labels(trusted)
    del _labels
    feature_provenance = _feature_provenance(trusted)
    try:
        e24_runner.enforce_aggregate_artifact_caps(
            ledger_path=trusted.upstream.ledger_path
        )
    except Exception as exc:
        raise E24FinalFitError("aggregate artifact cap failed during final verification") from exc
    validated = validate_final_manifest_payload(
        payload,
        expected_authority=_authority_record(trusted),
        expected_sources=_source_hashes(),
        expected_feature_provenance=feature_provenance,
        expected_label_provenance=label_provenance,
        expected_model=observed_model,
    )
    return AuthenticatedFinalModel(
        authority=trusted,
        manifest_path=manifest_path,
        manifest_sha256=_sha(manifest_path),
        manifest=validated,
        model_path=MODEL_PATH.resolve(),
        model_sha256=observed_model["sha256"],
        predictor=predictor,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("smoke", "verify-authority", "fit", "verify-model")
    )
    parser.add_argument("--ledger", type=Path, default=e24_runner.DEFAULT_LEDGER)
    parser.add_argument(
        "--ledger-sha256", default=staged_runner.EXPECTED_GENERATION3_LEDGER_SHA256
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.mode == "smoke":
        print(
            _canonical_bytes(
                {
                    "status": "data_free",
                    "model_path": str(MODEL_PATH.resolve()),
                    "manifest_path": str(MANIFEST_PATH.resolve()),
                    "learner_contract_sha256": FINAL_LEARNER_CONTRACT_SHA256,
                    "scenes": list(e24_eval.CALIBRATION_IDS),
                    "feature_count": FINAL_FEATURES,
                    "trees": FINAL_TREES,
                    "seed": FINAL_SEED,
                    "validation": False,
                    "e25_opened": False,
                }
            ).decode("ascii"),
            end="",
        )
        return
    authority = authenticate_final_fit_authority(args.ledger, args.ledger_sha256)
    if args.mode == "verify-authority":
        print(
            _canonical_bytes(
                {
                    "status": "pass",
                    "staged_report_sha256": authority.staged_report_sha256,
                    "stage": authority.staged_report["stage"],
                    "e25_opened": False,
                }
            ).decode("ascii"),
            end="",
        )
    elif args.mode == "fit":
        result = run_final_fit(authority)
        print(
            _canonical_bytes(
                {
                    "status": result["status"],
                    "manifest_path": str(MANIFEST_PATH.resolve()),
                    "manifest_sha256": _sha(MANIFEST_PATH),
                    "model_sha256": result["model"]["sha256"],
                    "e25_opened": False,
                }
            ).decode("ascii"),
            end="",
        )
    elif args.mode == "verify-model":
        result = authenticate_final_model(authority=authority)
        print(
            _canonical_bytes(
                {
                    "status": "pass",
                    "manifest_sha256": result.manifest_sha256,
                    "model_sha256": result.model_sha256,
                    "e25_opened": False,
                }
            ).decode("ascii"),
            end="",
        )


if __name__ == "__main__":
    main()


__all__ = (
    "AuthenticatedFinalModel",
    "E24FinalFitError",
    "FINAL_LEARNER_CONTRACT",
    "FINAL_LEARNER_CONTRACT_SHA256",
    "FINAL_MANIFEST_SCHEMA",
    "FINAL_OWN_SOURCE_FILES",
    "FINAL_ROOT",
    "FinalFitAuthority",
    "FinalTrainingBatch",
    "MANIFEST_PATH",
    "MODEL_PATH",
    "STAGED_REPORT_PATH",
    "STAGED_REPORT_SCHEMA",
    "authenticate_final_fit_authority",
    "authenticate_final_model",
    "authenticate_prefit_source_snapshot",
    "build_final_training_batch",
    "final_learner_contract",
    "load_consensus_all8_labels",
    "run_final_fit",
    "validate_final_manifest_payload",
    "validate_staged_pass_payload",
)

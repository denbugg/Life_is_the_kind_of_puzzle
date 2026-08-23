"""Fail-closed staging and trusted metric runner for the E24 SSIM/NLM decision.

The runner first completes the label-free transaction:

1. authenticate the generation-3 ledger, canary, structural PASS, resource
   receipt, all fold models, and all OOF prediction commits;
2. create one append-only source/protocol seal for the new staged code;
3. commit all eight label-free decode artifacts;
4. commit all eight raw-R/D, board, upright-canvas, and NLM10 artifacts;
5. publish a global 8/8 board barrier.

Only then, and only when the caller supplies the exact expected barrier SHA,
``staged-eval`` starts one authenticated subprocess per scene.  That worker may
read exactly ``permutation.npy`` from the ledger-pinned raw archive and the
``clean`` field from the pinned CanvasDataset replay.  It emits one canonical
metric row and never exports either array.  RR96 metrics reuse the exact pinned
E12 record after byte checks, matching the frozen E14 choice.  Importing this
module is data-free; E25 and test data are never in scope.
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


STORAGE_ROOT = Path("E:/pazzle_work/posegraph_e24_selector")
RUNTIME_ROOT = STORAGE_ROOT / "tmp"
PYCACHE_ROOT = STORAGE_ROOT / "pycache"
sys.pycache_prefix = str(PYCACHE_ROOT)
for _key in ("TEMP", "TMP", "TMPDIR", "JOBLIB_TEMP_FOLDER", "LIGHTGBM_TMPDIR"):
    os.environ[_key] = str(RUNTIME_ROOT)

import numpy as np

import e24_context_relation_selector as selector
import eval_e24_context_relation_selector as e24_eval
import eval_e24_staged_ssim_nlm as staged
import run_e24_context_relation_selector as e24_runner


# The trusted metric worker must stay narrow.  Broad historical oracle/dataset
# modules expose RawScene, directory enumeration, and unrestricted replay and
# are therefore forbidden in that subprocess even if a transitive dependency
# later attempts to import them.
FORBIDDEN_METRIC_WORKER_MODULES = frozenset(
    {
        "eval_clean_score_oracle",
        "eval_e14_cc192_discovery",
        "eval_buddies_ssim_budget",
        "canvas_data",
    }
)
FORBIDDEN_METRIC_WORKER_SYMBOLS = frozenset(
    {"RawScene", "load_raw_scenes", "train_val_split", "list_train", "list_test"}
)


class E24StagedRunnerError(RuntimeError):
    """An authority, provenance, or pre-metric staged transaction failed."""


ROOT = Path(__file__).resolve().parents[1]
STAGED_ROOT = STORAGE_ROOT / "staged_v1"
SEAL_PATH = STAGED_ROOT / "premetric_seal.json"
BARRIER_PATH = STAGED_ROOT / "decode_board_barrier.json"
REPORT_PATH = STAGED_ROOT / "staged_ssim_nlm_report.json"
METRIC_ROOT = STAGED_ROOT / "metric_broker"
BARRIER_SCHEMA = "pazzle-e24-crs-v1-staged-decode-board-barrier-v1"
METRIC_CHAIN_GENESIS_SHA256 = staged.METRIC_CHAIN_GENESIS_SHA256

# Literal calibration-only lineage copied from the byte-pinned E12 report.
# The metric worker does not import or execute that broad historical oracle.
PINNED_METRIC_SCENES: Mapping[int, Mapping[str, Any]] = {
    10: {"validation_name":"img_006710.png","cache_sha256":"6da35bbb4257c1011fa7558318e117688823f80f5f338033bdc9d8dc4ac1e56c","permutation_sha256":"844457d42c6cd0be2d294dbd42c4196bdfd44551ce7e2bc54bf891299f46ca54","target_sha256":"448d005c21899d9d69ac5cdf3142864b01bc237473d9dd2840c724abff549ec6","placement":0.001736111111111111,"neighbour":0.11775362318840579,"right":0.1322463768115942,"down":0.10326086956521739,"solve_only_ssim":0.07901157342227812,"final_ssim":0.14792033808788543,"objective":123.48388593626032,"board_sha256":"4393ab546aa4199c377b047d27b7e2f9c0d9c483ad6650f8dcd4d06d331e322f","solved_corrupted_canvas_sha256":"ed4e1db74c523b13a3809a0b956aa5dd5d9286315a4bb2da5d8628c543f7a85a","restored_canvas_sha256":"a768748bb3999750e4944ed6da7d38e37258c5b1ce0ed498edb00f5afac85a1c"},
    11: {"validation_name":"img_006711.png","cache_sha256":"49bf2864f55dc8c2e043e4d1a9debf975808ceb951ae248a008a84600acc14ee","permutation_sha256":"1ede93527b156b86492e8920fd1e7f601d9b70f5074aef7583e61d7ad5a9e07c","target_sha256":"45490a4a40ff951d18524d0daede72ffce935605b8b3b5dc7dad148a0cbd4e95","placement":0.005208333333333333,"neighbour":0.10326086956521739,"right":0.06702898550724638,"down":0.13949275362318841,"solve_only_ssim":0.12531929072235068,"final_ssim":0.19582124834406925,"objective":97.52050641085702,"board_sha256":"613f0b6a9f840429ef4ad292cb75639ca4baee155994e22d58e5d873ae4fc737","solved_corrupted_canvas_sha256":"6202563d2b41d077d70571b8e2786e29452ea257fe3f9b729b3fd65140758381","restored_canvas_sha256":"e8e1b842dd0ee6e20053e65e3afee456f1976efe8ee1cf5b6c07fce07849740d"},
    12: {"validation_name":"img_006712.png","cache_sha256":"cd9cbab2843d8285d887ddc5a240b4a72818582c774407f6e60b9160edaa40c2","permutation_sha256":"0bde19797ac6a0a0404b9cb1ec6d11b1764be3b9e33f339a9c756def7e1ebebe","target_sha256":"71451fd207204e2d953f531c8aca0fb82bfad38cb2aa9e1c1df2b84ed7ea5bac","placement":0.010416666666666666,"neighbour":0.13858695652173914,"right":0.14492753623188406,"down":0.1322463768115942,"solve_only_ssim":0.10647774682097837,"final_ssim":0.1877059619330476,"objective":136.37888660158504,"board_sha256":"03cb9756b6226e65e56f1ce8acd46f8485ae958791db64ed01f34c8d77166632","solved_corrupted_canvas_sha256":"693b5588a953878b93795fb6553bff53e0191ccaeed4b979a436f928ae70d4d","restored_canvas_sha256":"621a63b4e75696591c5b0a2874281dad3a91ab6aebd507baea934cce25acb270"},
    13: {"validation_name":"img_006713.png","cache_sha256":"7926ed7a52becf4064243157b8d741c5f538aad3a98c7deeaa24e5081aa69100","permutation_sha256":"c6b05f1a13037be8265af42a6316457a12f24b9e91bec3fbfe18cf080551c3f3","target_sha256":"1feb1f707699923ec718ee52cb97dbd907275d8628eafd78c506323b6881437a","placement":0.003472222222222222,"neighbour":0.13043478260869565,"right":0.16304347826086957,"down":0.09782608695652174,"solve_only_ssim":0.08961915516939149,"final_ssim":0.14227160420236515,"objective":130.92928356438188,"board_sha256":"3795a03744ccb815256b679a11eaa4b65fb7d797685a8a7b6d215c93cfb03d65","solved_corrupted_canvas_sha256":"6654a84a9127e7b1167c416f74d12eb2fd248edef44e011e373fb386b36f87e2","restored_canvas_sha256":"c42562f34ac2aeb3a3e223f858882c0d359a58c731b464f29ad787f6819d2e82"},
    14: {"validation_name":"img_006714.png","cache_sha256":"4a615d76e9f90481a702b9a4cafb0d9551cade50b003aff7b1f360e6c07c947f","permutation_sha256":"32f63c250a7d909ccc17df13f6d0b0f35a53d77c45473022a6f1ec8f0330df04","target_sha256":"b359bd49b500c439a3d16ad3ea5eb6bbd6e73fbdd85db93d4484c46790ab3a1d","placement":0.0,"neighbour":0.10054347826086957,"right":0.14855072463768115,"down":0.05253623188405797,"solve_only_ssim":0.08531096200858113,"final_ssim":0.13762968886192767,"objective":97.87761797961093,"board_sha256":"f9cb50135cc492c4e1d07e72e304278b9a295818a175812c4a97ff96ab1bd264","solved_corrupted_canvas_sha256":"87f39b6dd59b9cee4ae2598e2e68eed05c0af1cb35c71f1bd02615bd8174dd4e","restored_canvas_sha256":"888124e50f05d171eb39388099683639f1cd7ea2966e334a1affb586109ae6ed"},
    15: {"validation_name":"img_006715.png","cache_sha256":"ba0a7213cea9b3dd52d65226bf0b95d37ecdb86601de68ea64ed69509b750381","permutation_sha256":"cb31efc21388b2e684250c87b71757384d11df2525bc888876b5d2058a237714","target_sha256":"677a0cb5b53b2804101b7b5c5dfeda7dce6fb80e61d98b43aca60b88f08205b6","placement":0.001736111111111111,"neighbour":0.09510869565217392,"right":0.06884057971014493,"down":0.1213768115942029,"solve_only_ssim":0.13953348343964708,"final_ssim":0.255155966007237,"objective":96.08115629004351,"board_sha256":"643dada48bc51f889bb06c446e61a1a71b5bde1ff6ed27be9bce7b571e3dfd96","solved_corrupted_canvas_sha256":"ba6107399600f80822f1cf9ad5b5f475d30e40284346749c5d7650aa5b2b09d5","restored_canvas_sha256":"d0475e06242d414ec1e587dd516a94c91c06babb41f9316bd41a7e054bc96943"},
    16: {"validation_name":"img_006716.png","cache_sha256":"0f0e554b79bba825f36d8f8bfcdc399f15656e3e130e976cedb839dfc470af30","permutation_sha256":"66a68d6ca81373d0afb54ac79eb430d80b98de91f625532be692e1d137865e9c","target_sha256":"ebce46ead4c51ec06af38b3f484434c1464c3f77409332908fdb680af4527674","placement":0.0,"neighbour":0.125,"right":0.09057971014492754,"down":0.15942028985507245,"solve_only_ssim":0.06323343259250945,"final_ssim":0.09473940225470993,"objective":119.35319747999233,"board_sha256":"4e3dc3597195e8e397749e7ca6d705e73edce40f9c3495e2cf4e90a54573b1c0","solved_corrupted_canvas_sha256":"2ffc8d16b490bfb4ea5464bd38e07ac35b23c18f89537738e9b718d78b9838af","restored_canvas_sha256":"9904dcca3c83ae9415cecf0b8d38616a267321ae58eee596186350059784d690"},
    17: {"validation_name":"img_006717.png","cache_sha256":"94ad5820219f2d08e2bad2bfbb832519d586efc3bc72bd5ee650fc886c838c40","permutation_sha256":"242446f740cec709d72a771326bd72f3e8fa85889cbd6396c006c58b76890f97","target_sha256":"446ab71ce5565e235275ce078505ff160b6de30937875b0d9469f72f08960ce1","placement":0.001736111111111111,"neighbour":0.12590579710144928,"right":0.11594202898550725,"down":0.1358695652173913,"solve_only_ssim":0.06835806900357573,"final_ssim":0.11319141514491816,"objective":133.8746766626079,"board_sha256":"61a0b6d2912d5368a67f67697703f7b32c23dc8b43e3266bdcec3769a224535f","solved_corrupted_canvas_sha256":"a7fd4f84bf6a3984074ca709607e8f8f29f1ebf1b7f53a3206d2d6347c3dc5f6","restored_canvas_sha256":"c941f7188c191ec590602aebcce9c79e3131b245294c5cf6d01068b8eb67d0f8"},
}

EXPECTED_GENERATION3_LEDGER_SHA256 = (
    "e859edfaff913329429115ad171571b8f5a40a3698a1c4a847f0abef1a5a4bf5"
)
EXPECTED_GENERATION3_RUN_CONTRACT_SHA256 = (
    "6fe34603e714776dff53a763eab63abadedd85be0333b4d0861f0ad37f4fcbcc"
)

PROTOCOL_DOCUMENT = ROOT / "E24_CONTEXT_RELATION_SELECTOR.md"
PLAN_DOCUMENT = ROOT / "autoresearch-runs/pazzle-solution-20260806/PLAN.md"

SOURCE_FILES = (
    Path(__file__).resolve(),
    ROOT / "src/eval_e24_staged_ssim_nlm.py",
    ROOT / "tests/test_e24_staged_ssim_nlm.py",
    ROOT / "tests/test_run_e24_staged_ssim_nlm.py",
    ROOT / "src/e24_context_relation_selector.py",
    ROOT / "src/eval_e24_context_relation_selector.py",
    ROOT / "src/run_e24_context_relation_selector.py",
    ROOT / "tests/test_e24_context_relation_selector.py",
    ROOT / "tests/test_e24_context_relation_evaluator.py",
    ROOT / "src/e23_i21_residual_candidate_oracle.py",
    ROOT / "src/solve_buddies.py",
    ROOT / "src/eval_seeded_qap.py",
    ROOT / "src/eval_clean_score_oracle.py",
    ROOT / "src/eval_buddies_ssim_budget.py",
    ROOT / "src/eval_e14_cc192_discovery.py",
    ROOT / "src/imgio.py",
    ROOT / "src/pipeline.py",
    ROOT / "src/placement_metrics.py",
    ROOT / "src/canvas_data.py",
    ROOT / "src/distort.py",
    ROOT / "src/config.py",
    # Freeze the complete authorized downstream chain before the first E24
    # image metric.  These are hashed dynamically into the create-once seal.
    ROOT / "src/run_e24_final_all8.py",
    ROOT / "tests/test_run_e24_final_all8.py",
    ROOT / "E25_SOURCE_GROUP_DISJOINT_CONFIRMATION.md",
    ROOT / "src/eval_e25_source_group_confirmation.py",
    ROOT / "src/run_e25_source_group_confirmation.py",
    ROOT / "tests/test_e25_source_group_confirmation.py",
    ROOT / "tests/test_run_e25_source_group_confirmation.py",
    ROOT / "src/infer_e24.py",
    ROOT / "tests/test_infer_e24.py",
    PROTOCOL_DOCUMENT,
    PLAN_DOCUMENT,
)


@dataclass(frozen=True)
class AuthenticatedAuthority:
    ledger_path: Path
    ledger_sha256: str
    ledger: Mapping[str, Any]
    structural_report_sha256: str
    structural_report: Mapping[str, Any]
    orchestration_receipt_sha256: str
    orchestration_receipt: Mapping[str, Any]
    canary_gate_sha256: str
    fold_commit_paths: Mapping[int, Path]
    fold_commit_sha256: Mapping[int, str]
    feature_manifests: Mapping[int, Mapping[str, Any]]
    verified_oof: Any


def _sha(path: Path) -> str:
    return staged.sha256_file(path)


def _lower_sha(value: object, *, label: str) -> str:
    try:
        return e24_eval._validate_lower_hex_sha256(value, label=label)
    except e24_eval.E24EvaluatorContractError as exc:
        raise E24StagedRunnerError(str(exc)) from exc


def _require_storage(path: Path, *, label: str) -> Path:
    try:
        return e24_eval._require_e24_storage_path(path, label=label)
    except e24_eval.E24EvaluatorContractError as exc:
        raise E24StagedRunnerError(str(exc)) from exc


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        return staged.load_canonical_json(path, label=label)
    except staged.E24StagedContractError as exc:
        raise E24StagedRunnerError(str(exc)) from exc


def _structural_rows(payload: Mapping[str, Any]) -> list[e24_eval.StructuralSceneCounts]:
    fields = {field.name for field in dataclasses.fields(e24_eval.StructuralSceneCounts)}
    raw_rows = payload.get("rows")
    if type(raw_rows) is not list or len(raw_rows) != len(e24_eval.CALIBRATION_IDS):
        raise E24StagedRunnerError("structural report does not contain exactly eight rows")
    rows: list[e24_eval.StructuralSceneCounts] = []
    try:
        for raw in raw_rows:
            if type(raw) is not dict or set(raw) != fields:
                raise E24StagedRunnerError("structural row field set drifted")
            rows.append(e24_eval.StructuralSceneCounts(**raw))
    except E24StagedRunnerError:
        raise
    except Exception as exc:
        raise E24StagedRunnerError("structural row cannot be reconstructed") from exc
    return rows


def validate_structural_pass_payload(
    payload: Mapping[str, Any],
    *,
    ledger_sha256: str,
    run_contract_sha256: str,
    fold_commit_sha256: Mapping[int, str],
) -> dict[str, Any]:
    """Recompute every structural summary/check instead of trusting PASS text."""

    expected_keys = {
        "schema",
        "status",
        "stage",
        "ledger_sha256",
        "run_contract_sha256",
        "fold_commit_sha256",
        "rows",
        "summary",
        "decision",
        "staged_board_ssim_nlm",
        "e25_opened",
    }
    if type(payload) is not dict or set(payload) != expected_keys:
        raise E24StagedRunnerError("structural report field set drifted")
    if (
        payload["schema"] != e24_runner.STRUCTURAL_REPORT_SCHEMA
        or payload["status"] != "complete"
        or payload["stage"] != "go_staged_end_to_end"
        or payload["ledger_sha256"] != ledger_sha256
        or payload["run_contract_sha256"] != run_contract_sha256
        or payload["staged_board_ssim_nlm"] != "sealed_not_run"
        or payload["e25_opened"] is not False
    ):
        raise E24StagedRunnerError("structural report is not the sealed PASS authority")
    normalized_fold_hashes = {
        str(fold): _lower_sha(fold_commit_sha256[fold], label=f"fold {fold} commit SHA")
        for fold in sorted(e24_eval.OOF_FOLDS)
    }
    if payload["fold_commit_sha256"] != normalized_fold_hashes:
        raise E24StagedRunnerError("structural report fold-commit hashes drifted")
    try:
        expected_summary = e24_eval.summarize_structural(_structural_rows(payload))
        expected_decision = e24_eval.structural_decision(expected_summary)
    except e24_eval.E24EvaluatorContractError as exc:
        raise E24StagedRunnerError("structural report failed exact re-evaluation") from exc
    if (
        payload["summary"] != expected_summary
        or payload["decision"] != expected_decision
        or expected_decision["passed"] is not True
        or expected_decision["stage"] != "go_staged_end_to_end"
        or payload["stage"] != expected_decision["stage"]
    ):
        raise E24StagedRunnerError("forged/drifted structural PASS")
    return dict(payload)


def validate_orchestration_receipt_payload(
    payload: Mapping[str, Any],
    *,
    ledger_sha256: str,
    run_contract_sha256: str,
    canary_gate_sha256: str,
    structural_report_sha256: str,
) -> dict[str, Any]:
    expected_keys = {
        "schema",
        "status",
        "ledger_sha256",
        "run_contract_sha256",
        "canary_gate_sha256",
        "structural_report_sha256",
        "resource",
        "checks",
    }
    if type(payload) is not dict or set(payload) != expected_keys:
        raise E24StagedRunnerError("orchestration receipt field set drifted")
    resource = payload.get("resource")
    if type(resource) is not dict or set(resource) != {
        "child_process_cpu_seconds",
        "maximum_child_peak_rss_bytes",
        "cpu_seconds_max",
        "peak_rss_bytes_max",
    }:
        raise E24StagedRunnerError("orchestration resource record drifted")
    cpu = resource["child_process_cpu_seconds"]
    peak = resource["maximum_child_peak_rss_bytes"]
    if (
        type(cpu) not in {int, float}
        or not math.isfinite(float(cpu))
        or float(cpu) < 0.0
        or type(peak) is not int
        or peak < 0
        or resource["cpu_seconds_max"] != e24_eval.OOF_CPU_SECONDS_MAX
        or resource["peak_rss_bytes_max"] != e24_eval.PEAK_RAM_BYTES_MAX
    ):
        raise E24StagedRunnerError("orchestration resource values drifted")
    expected_checks = {
        "oof_cpu_at_most_8h": float(cpu) <= e24_eval.OOF_CPU_SECONDS_MAX,
        "peak_rss_at_most_16gib": peak <= e24_eval.PEAK_RAM_BYTES_MAX,
        "aggregate_artifacts_at_most_8gib": True,
    }
    if (
        payload["schema"] != e24_runner.ORCHESTRATION_RECEIPT_SCHEMA
        or payload["status"] != "pass"
        or payload["ledger_sha256"] != ledger_sha256
        or payload["run_contract_sha256"] != run_contract_sha256
        or payload["canary_gate_sha256"] != canary_gate_sha256
        or payload["structural_report_sha256"] != structural_report_sha256
        or payload["checks"] != expected_checks
        or not all(expected_checks.values())
    ):
        raise E24StagedRunnerError("orchestration receipt is not an exact resource PASS")
    return dict(payload)


def _expected_oof_provenance(
    ledger: Mapping[str, Any],
    ledger_sha256: str,
    feature_manifests: Mapping[int, Mapping[str, Any]],
) -> dict[int, dict[str, Any]]:
    output: dict[int, dict[str, Any]] = {}
    for fold in e24_eval.OOF_FOLDS:
        boundary = e24_eval.fold_boundary(fold)
        label_manifest_sha256: dict[int, str] = {}
        for image in boundary.train_ids:
            manifest = feature_manifests[image]
            _payload, label_manifest_sha256[image] = e24_runner._verify_fold_label_manifest(
                fold,
                image,
                ledger_sha256=ledger_sha256,
                run_contract_sha256=ledger["run_contract_sha256"],
                rows=manifest["rows"],
                queries=manifest["queries"],
                feature_sha256=manifest["feature_file"]["sha256"],
                verify_label_file=False,
            )
        output[fold] = e24_runner._fold_run_provenance(
            fold=fold,
            ledger=ledger,
            ledger_sha256=ledger_sha256,
            feature_manifests=feature_manifests,
            label_manifest_sha256=label_manifest_sha256,
        )
    return output


def authenticate_authority(
    ledger_path: Path, ledger_sha256: str
) -> AuthenticatedAuthority:
    """Read-only authentication; never call ``e24_runner.orchestrate`` here."""

    expected_ledger = _require_storage(e24_runner.DEFAULT_LEDGER, label="generation-3 ledger")
    supplied_ledger = _require_storage(ledger_path, label="generation-3 ledger")
    if supplied_ledger != expected_ledger:
        raise E24StagedRunnerError("staged authority requires the literal E24 ledger path")
    try:
        ledger = e24_runner.verify_preflight_ledger(supplied_ledger, ledger_sha256)
        e24_runner.verify_feature_canary(supplied_ledger, ledger_sha256)
        e24_runner.enforce_aggregate_artifact_caps(ledger_path=supplied_ledger)
    except (e24_runner.E24RunnerError, e24_eval.E24EvaluatorContractError) as exc:
        raise E24StagedRunnerError("generation-3 ledger/canary/cap authentication failed") from exc
    if ledger["run_contract_sha256"] != EXPECTED_GENERATION3_RUN_CONTRACT_SHA256:
        raise E24StagedRunnerError("generation-3 run-contract SHA is not the audited value")
    if ledger_sha256 != EXPECTED_GENERATION3_LEDGER_SHA256:
        raise E24StagedRunnerError("generation-3 ledger SHA is not the audited value")

    fold_commit_paths = {
        fold: e24_runner._fold_paths(fold)[2].resolve()
        for fold in e24_eval.OOF_FOLDS
    }
    if any(not path.is_file() for path in fold_commit_paths.values()):
        raise E24StagedRunnerError("all four fold commits must exist before staged authority")
    fold_commit_sha256 = {fold: _sha(path) for fold, path in fold_commit_paths.items()}

    structural_path = _require_storage(
        e24_runner.STRUCTURAL_REPORT, label="structural report"
    )
    if not structural_path.is_file():
        raise E24StagedRunnerError("structural report is absent")
    structural_sha = _sha(structural_path)
    structural = validate_structural_pass_payload(
        _load_json(structural_path, label="structural report"),
        ledger_sha256=ledger_sha256,
        run_contract_sha256=ledger["run_contract_sha256"],
        fold_commit_sha256=fold_commit_sha256,
    )

    receipt_path = _require_storage(
        e24_runner.ORCHESTRATION_RECEIPT_PATH, label="orchestration receipt"
    )
    if not receipt_path.is_file():
        raise E24StagedRunnerError("orchestration receipt is absent")
    receipt_sha = _sha(receipt_path)
    canary_sha = _sha(e24_runner.CANARY_GATE_PATH)
    receipt = validate_orchestration_receipt_payload(
        _load_json(receipt_path, label="orchestration receipt"),
        ledger_sha256=ledger_sha256,
        run_contract_sha256=ledger["run_contract_sha256"],
        canary_gate_sha256=canary_sha,
        structural_report_sha256=structural_sha,
    )

    feature_manifests: dict[int, Mapping[str, Any]] = {}
    for image in e24_eval.CALIBRATION_IDS:
        try:
            table, manifest = e24_runner._load_feature_artifact(
                image, ledger_sha256, ledger["run_contract_sha256"]
            )
        except e24_runner.E24RunnerError as exc:
            raise E24StagedRunnerError(f"scene {image} feature authentication failed") from exc
        feature_manifests[image] = manifest
        del table
        gc.collect()
    expected_provenance = _expected_oof_provenance(
        ledger, ledger_sha256, feature_manifests
    )
    model_records: dict[int, tuple[Path, str]] = {}
    for fold in e24_eval.OOF_FOLDS:
        try:
            predictor, model_path, model_manifest = e24_runner._reload_committed_predictor(
                fold, expected_provenance[fold]
            )
        except e24_runner.E24RunnerError as exc:
            raise E24StagedRunnerError(f"fold {fold} model authentication failed") from exc
        model_records[fold] = (model_path, model_manifest["model"]["sha256"])
        del predictor
    try:
        verified = e24_eval.verify_all_oof_commits(
            fold_commit_paths, expected_run_provenance=expected_provenance
        )
    except e24_eval.E24EvaluatorContractError as exc:
        raise E24StagedRunnerError("global OOF prediction barrier failed") from exc
    for fold, commit in verified.commits.items():
        if (
            commit.model_path != model_records[fold][0]
            or commit.model_sha256 != model_records[fold][1]
        ):
            raise E24StagedRunnerError("OOF commit/model-manifest binding drifted")

    return AuthenticatedAuthority(
        ledger_path=supplied_ledger,
        ledger_sha256=ledger_sha256,
        ledger=ledger,
        structural_report_sha256=structural_sha,
        structural_report=structural,
        orchestration_receipt_sha256=receipt_sha,
        orchestration_receipt=receipt,
        canary_gate_sha256=canary_sha,
        fold_commit_paths=fold_commit_paths,
        fold_commit_sha256=fold_commit_sha256,
        feature_manifests=feature_manifests,
        verified_oof=verified,
    )


def _source_hashes() -> dict[str, str]:
    output: dict[str, str] = {}
    for path in SOURCE_FILES:
        resolved = path.resolve()
        if not resolved.is_file():
            raise E24StagedRunnerError(f"staged source/protocol file is absent: {resolved}")
        output[str(resolved)] = _sha(resolved)
    return dict(sorted(output.items()))


def _runtime_provenance() -> dict[str, str]:
    try:
        import eval_e14_cc192_discovery as e14

        return dict(e14._runtime_provenance())
    except Exception as exc:
        raise E24StagedRunnerError("exact E12/E14 OpenCV runtime provenance failed") from exc


def build_premetric_seal_payload(authority: AuthenticatedAuthority) -> dict[str, Any]:
    import eval_clean_score_oracle as e12
    import eval_e14_cc192_discovery as e14

    fold_records = {}
    for fold, commit in authority.verified_oof.commits.items():
        fold_records[str(fold)] = {
            "commit_path": str(authority.fold_commit_paths[fold]),
            "commit_sha256": authority.fold_commit_sha256[fold],
            "model_path": str(commit.model_path),
            "model_sha256": commit.model_sha256,
            "prediction_path": str(commit.prediction_path),
            "prediction_sha256": commit.prediction_sha256,
            "feature_sha256": {
                str(image): commit.feature_sha256[image]
                for image in sorted(commit.feature_sha256)
            },
        }
    return {
        "schema": staged.PREMETRIC_SEAL_SCHEMA,
        "schema_version": staged.SCHEMA_VERSION,
        "status": "frozen_staging_and_post_barrier_metric_broker",
        "staged_protocol": json.loads(staged.canonical_json_bytes(dict(staged.STAGED_PROTOCOL))),
        "staged_protocol_sha256": staged.PROTOCOL_SHA256,
        "authority": {
            "ledger_path": str(authority.ledger_path),
            "ledger_sha256": authority.ledger_sha256,
            "run_contract_sha256": authority.ledger["run_contract_sha256"],
            "canary_gate_path": str(e24_runner.CANARY_GATE_PATH.resolve()),
            "canary_gate_sha256": authority.canary_gate_sha256,
            "structural_report_path": str(e24_runner.STRUCTURAL_REPORT.resolve()),
            "structural_report_sha256": authority.structural_report_sha256,
            "orchestration_receipt_path": str(
                e24_runner.ORCHESTRATION_RECEIPT_PATH.resolve()
            ),
            "orchestration_receipt_sha256": authority.orchestration_receipt_sha256,
            "folds": fold_records,
        },
        "sources": _source_hashes(),
        "runtime": _runtime_provenance(),
        "rr96_reference_pins": {
            "e12_report_sha256": e14.EXPECTED_E12_REPORT_SHA256,
            "calibration_report_path": str(e12.DEFAULT_CALIBRATION_REPORT.resolve()),
            "calibration_report_sha256": e12.CALIBRATION_REPORT_SHA256,
            "scene_provenance_digest": e12.SCENE_PROVENANCE_DIGEST,
            "mean_solve_only_ssim": e14.EXPECTED_RR_MEAN_SOLVE_SSIM,
            "mean_final_ssim": e14.EXPECTED_RR_MEAN_FINAL_SSIM,
            "metric_choice": "exact_E12_RR_record_no_second_NLM_call",
        },
        "metric_broker_contract": json.loads(
            staged.canonical_json_bytes(dict(staged.METRIC_BROKER_CONTRACT))
        ),
        "metric_broker_contract_sha256": staged.METRIC_BROKER_CONTRACT_SHA256,
        "authorized_outputs": [
            "eight_label_free_decode_commits",
            "eight_label_free_raw_rd_board_upright_nlm10_commits",
            "global_decode_board_barrier",
            "eight_sha_chained_metric_requests_and_row_only_responses_after_exact_barrier",
            "one_canonical_staged_ssim_nlm_report",
        ],
        "still_sealed": [
            "final_all8_fit",
            "e25",
            "test",
        ],
        "clean_target_metric_broker": "frozen_narrow_subprocess_after_exact_barrier_only",
        "metrics_opened": False,
        "e25_opened": False,
    }


def freeze_premetric_seal(authority: AuthenticatedAuthority) -> tuple[dict[str, Any], str]:
    destination = _require_storage(SEAL_PATH, label="staged premetric seal")
    if BARRIER_PATH.exists() or REPORT_PATH.exists():
        raise E24StagedRunnerError("cannot create a premetric seal after downstream artifacts")
    payload = build_premetric_seal_payload(authority)
    try:
        digest = staged.commit_canonical_create_once(destination, payload)
    except staged.E24StagedContractError as exc:
        raise E24StagedRunnerError(str(exc)) from exc
    return payload, digest


def verify_premetric_seal(
    authority: AuthenticatedAuthority, expected_sha256: str
) -> dict[str, Any]:
    expected = _lower_sha(expected_sha256, label="premetric seal SHA")
    seal = _require_storage(SEAL_PATH, label="staged premetric seal")
    if not seal.is_file() or _sha(seal) != expected:
        raise E24StagedRunnerError("premetric seal SHA mismatch/absence")
    observed = _load_json(seal, label="staged premetric seal")
    expected_payload = build_premetric_seal_payload(authority)
    if observed != expected_payload:
        raise E24StagedRunnerError("premetric seal no longer matches sources/authority")
    return observed


def _scene_paths(image: int) -> tuple[Path, Path, Path, Path]:
    if image not in staged.CALIBRATION_IDS:
        raise E24StagedRunnerError("staged scene ID is outside 10..17")
    root = STAGED_ROOT / f"image_{image:04d}"
    return (
        root / "decode.npz",
        root / "decode.commit.json",
        root / "board_nlm.npz",
        root / "board_nlm.commit.json",
    )


def _scene_prediction(
    authority: AuthenticatedAuthority, image: int
) -> tuple[int, Any, np.ndarray, np.ndarray]:
    fold = next(fold for fold, ids in e24_eval.OOF_FOLDS.items() if image in ids)
    commit = authority.verified_oof.commits[fold]
    mask = commit.predictions.scene_ids == image
    rows = commit.predictions.row_indices[mask]
    scores = np.ascontiguousarray(commit.predictions.scores[mask], dtype=np.float64)
    return fold, commit, np.ascontiguousarray(rows, dtype=np.int64), scores


def _expected_decode_provenance(
    authority: AuthenticatedAuthority,
    *,
    image: int,
    seal_sha256: str,
    input_payload: Mapping[str, Any],
    right: np.ndarray,
    down: np.ndarray,
) -> dict[str, str]:
    fold = next(fold for fold, ids in e24_eval.OOF_FOLDS.items() if image in ids)
    prediction_commit = authority.verified_oof.commits[fold]
    feature_manifest = authority.feature_manifests[image]
    return {
        "ledger_sha256": authority.ledger_sha256,
        "run_contract_sha256": authority.ledger["run_contract_sha256"],
        "premetric_seal_sha256": seal_sha256,
        "structural_report_sha256": authority.structural_report_sha256,
        "orchestration_receipt_sha256": authority.orchestration_receipt_sha256,
        "fold_commit_sha256": authority.fold_commit_sha256[fold],
        "model_sha256": prediction_commit.model_sha256,
        "prediction_sha256": prediction_commit.prediction_sha256,
        "feature_sha256": feature_manifest["feature_file"]["sha256"],
        "input_manifest_sha256": _sha(e24_runner._input_manifest_path(image)),
        "source_scene_contract_sha256": input_payload["source_scene_contract_sha256"],
        "right_sha256": staged.array_sha256(right),
        "down_sha256": staged.array_sha256(down),
    }


def _expected_board_provenance(
    authority: AuthenticatedAuthority,
    *,
    image: int,
    seal_sha256: str,
    input_payload: Mapping[str, Any],
    raw: Any,
    decode_manifest: Mapping[str, Any],
) -> dict[str, str]:
    _decode_artifact, decode_commit, _board_artifact, _board_commit = _scene_paths(
        image
    )
    tiles_record = input_payload["tiles"]
    return {
        "ledger_sha256": authority.ledger_sha256,
        "run_contract_sha256": authority.ledger["run_contract_sha256"],
        "premetric_seal_sha256": seal_sha256,
        "structural_report_sha256": authority.structural_report_sha256,
        "orchestration_receipt_sha256": authority.orchestration_receipt_sha256,
        "decode_commit_sha256": _sha(decode_commit),
        "decode_artifact_sha256": decode_manifest["artifact"]["sha256"],
        "raw_manifest_sha256": raw.manifest_sha256,
        "tiles_file_sha256": tiles_record["file_sha256"],
        "tiles_array_sha256": tiles_record["array_sha256"],
    }


def _commit_or_verify_decode(
    authority: AuthenticatedAuthority,
    *,
    image: int,
    seal_sha256: str,
    input_payload: Mapping[str, Any],
    raw: Any,
    result: Any,
    right: np.ndarray,
    down: np.ndarray,
) -> tuple[staged.FrozenDecode, dict[str, Any]]:
    table, feature_manifest = e24_runner._load_feature_artifact(
        image, authority.ledger_sha256, authority.ledger["run_contract_sha256"]
    )
    fold, prediction_commit, row_indices, scores = _scene_prediction(authority, image)
    if (
        prediction_commit.feature_sha256[image]
        != feature_manifest["feature_file"]["sha256"]
        or not np.array_equal(row_indices, np.arange(table.rows, dtype=np.int64))
        or scores.shape != (table.rows,)
        or not bool(np.isfinite(scores).all())
    ):
        raise E24StagedRunnerError("prediction/feature table binding drifted")
    try:
        e24_eval._validate_result_table_binding(result, table)
        decoded = selector.decode_relation_scores(result, table, scores)
        frozen = staged.freeze_decode_result(
            image=image,
            base_component_count=len(result.components),
            decoded=decoded,
        )
    except Exception as exc:
        raise E24StagedRunnerError("exact relation decode/freeze failed") from exc
    decode_artifact, decode_commit, _board_artifact, _board_commit = _scene_paths(image)
    provenance = _expected_decode_provenance(
        authority,
        image=image,
        seal_sha256=seal_sha256,
        input_payload=input_payload,
        right=right,
        down=down,
    )
    if decode_commit.is_file():
        observed, manifest = staged.load_decode(
            decode_commit, expected_image=image, expected_provenance=provenance
        )
        if staged.decode_npz_bytes(observed) != staged.decode_npz_bytes(frozen):
            raise E24StagedRunnerError("existing decode differs from exact replay")
        return observed, manifest
    body = staged.decode_npz_bytes(frozen)
    e24_runner.enforce_aggregate_artifact_caps(
        ledger_path=authority.ledger_path,
        additional_total_bytes=(0 if decode_artifact.exists() else len(body)) + 64 * 1024,
    )
    try:
        manifest = staged.commit_decode(
            artifact_path=decode_artifact,
            commit_path=decode_commit,
            value=frozen,
            provenance=provenance,
        )
    except staged.E24StagedContractError as exc:
        raise E24StagedRunnerError(str(exc)) from exc
    return frozen, manifest


def _commit_or_verify_board(
    authority: AuthenticatedAuthority,
    *,
    image: int,
    seal_sha256: str,
    input_payload: Mapping[str, Any],
    raw: Any,
    tiles: np.ndarray,
    right: np.ndarray,
    down: np.ndarray,
    decode: staged.FrozenDecode,
    decode_manifest: Mapping[str, Any],
) -> tuple[staged.FrozenBoardPair, dict[str, Any]]:
    _decode_artifact, _decode_commit, board_artifact, board_commit = _scene_paths(image)
    provenance = _expected_board_provenance(
        authority,
        image=image,
        seal_sha256=seal_sha256,
        input_payload=input_payload,
        raw=raw,
        decode_manifest=decode_manifest,
    )
    if board_commit.is_file():
        return staged.load_board_pair(
            board_commit, expected_image=image, expected_provenance=provenance
        )

    def frozen_dense(_ids: np.ndarray, _scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return right, down

    try:
        boards = staged.build_board_pair(
            image=image,
            candidate_ids=raw.arrays.candidate_ids,
            raw_logits=raw.arrays.candidate_scores,
            tiles=tiles,
            decode=decode,
            dense_builder=frozen_dense,
        )
    except staged.E24StagedContractError as exc:
        raise E24StagedRunnerError("exact RR96/CRS packer or NLM10 failed") from exc
    body = staged.board_npz_bytes(boards)
    e24_runner.enforce_aggregate_artifact_caps(
        ledger_path=authority.ledger_path,
        additional_total_bytes=(0 if board_artifact.exists() else len(body)) + 64 * 1024,
    )
    try:
        manifest = staged.commit_board_pair(
            artifact_path=board_artifact,
            commit_path=board_commit,
            value=boards,
            provenance=provenance,
        )
    except staged.E24StagedContractError as exc:
        raise E24StagedRunnerError(str(exc)) from exc
    return boards, manifest


def prepare_label_free_staged(
    authority: AuthenticatedAuthority, seal_sha256: str
) -> dict[str, Any]:
    verify_premetric_seal(authority, seal_sha256)
    if REPORT_PATH.exists():
        raise E24StagedRunnerError("staged report already exists; label-free replay is closed")
    records: list[dict[str, Any]] = []
    projection_sha = authority.ledger["upstream"]["label_free_input_projection"][
        "records_sha256"
    ]
    for image in staged.CALIBRATION_IDS:
        try:
            input_payload, raw, tiles, _spatial, result = e24_runner._recompute_candidate_pool(
                image,
                authority.ledger_sha256,
                authority.ledger["run_contract_sha256"],
                projection_sha,
            )
            right, down = staged.dense_rd_from_raw(
                raw.arrays.candidate_ids, raw.arrays.candidate_scores
            )
            decode, decode_manifest = _commit_or_verify_decode(
                authority,
                image=image,
                seal_sha256=seal_sha256,
                input_payload=input_payload,
                raw=raw,
                result=result,
                right=right,
                down=down,
            )
            _boards, board_manifest = _commit_or_verify_board(
                authority,
                image=image,
                seal_sha256=seal_sha256,
                input_payload=input_payload,
                raw=raw,
                tiles=tiles,
                right=right,
                down=down,
                decode=decode,
                decode_manifest=decode_manifest,
            )
        finally:
            gc.collect()
        decode_artifact, decode_commit, board_artifact, board_commit = _scene_paths(image)
        records.append(
            {
                "image": image,
                "decode_commit_path": str(decode_commit.resolve()),
                "decode_commit_sha256": _sha(decode_commit),
                "decode_artifact_sha256": _sha(decode_artifact),
                "board_commit_path": str(board_commit.resolve()),
                "board_commit_sha256": _sha(board_commit),
                "board_artifact_sha256": _sha(board_artifact),
            }
        )
    payload = {
        "schema": BARRIER_SCHEMA,
        "schema_version": staged.SCHEMA_VERSION,
        "status": "complete_8_of_8_label_free",
        "staged_protocol_sha256": staged.PROTOCOL_SHA256,
        "premetric_seal_sha256": seal_sha256,
        "ledger_sha256": authority.ledger_sha256,
        "run_contract_sha256": authority.ledger["run_contract_sha256"],
        "structural_report_sha256": authority.structural_report_sha256,
        "orchestration_receipt_sha256": authority.orchestration_receipt_sha256,
        "records": records,
        "completed_images": list(staged.CALIBRATION_IDS),
        "permutation_target_ssim_or_neighbour_opened": False,
        "e25_opened": False,
        "metric_broker": "frozen_narrow_subprocess_requires_exact_barrier_sha",
    }
    barrier = _require_storage(BARRIER_PATH, label="decode/board barrier")
    if barrier.exists():
        if _load_json(barrier, label="decode/board barrier") != payload:
            raise E24StagedRunnerError("existing decode/board barrier drifted")
    else:
        try:
            staged.commit_canonical_create_once(barrier, payload)
        except staged.E24StagedContractError as exc:
            raise E24StagedRunnerError(str(exc)) from exc
    e24_runner.enforce_aggregate_artifact_caps(ledger_path=authority.ledger_path)
    return verify_board_barrier(authority, seal_sha256)


def verify_board_barrier(
    authority: AuthenticatedAuthority, seal_sha256: str
) -> dict[str, Any]:
    verify_premetric_seal(authority, seal_sha256)
    payload = _load_json(BARRIER_PATH, label="decode/board barrier")
    expected_keys = {
        "schema",
        "schema_version",
        "status",
        "staged_protocol_sha256",
        "premetric_seal_sha256",
        "ledger_sha256",
        "run_contract_sha256",
        "structural_report_sha256",
        "orchestration_receipt_sha256",
        "records",
        "completed_images",
        "permutation_target_ssim_or_neighbour_opened",
        "e25_opened",
        "metric_broker",
    }
    if (
        set(payload) != expected_keys
        or payload["schema"] != BARRIER_SCHEMA
        or payload["schema_version"] != staged.SCHEMA_VERSION
        or payload["status"] != "complete_8_of_8_label_free"
        or payload["staged_protocol_sha256"] != staged.PROTOCOL_SHA256
        or payload["premetric_seal_sha256"] != seal_sha256
        or payload["ledger_sha256"] != authority.ledger_sha256
        or payload["run_contract_sha256"] != authority.ledger["run_contract_sha256"]
        or payload["structural_report_sha256"] != authority.structural_report_sha256
        or payload["orchestration_receipt_sha256"]
        != authority.orchestration_receipt_sha256
        or payload["completed_images"] != list(staged.CALIBRATION_IDS)
        or payload["permutation_target_ssim_or_neighbour_opened"] is not False
        or payload["e25_opened"] is not False
        or payload["metric_broker"]
        != "frozen_narrow_subprocess_requires_exact_barrier_sha"
    ):
        raise E24StagedRunnerError("decode/board barrier identity drifted")
    records = payload["records"]
    if type(records) is not list or [item.get("image") for item in records if type(item) is dict] != list(
        staged.CALIBRATION_IDS
    ):
        raise E24StagedRunnerError("decode/board barrier scene set drifted")
    expected_record_keys = {
        "image",
        "decode_commit_path",
        "decode_commit_sha256",
        "decode_artifact_sha256",
        "board_commit_path",
        "board_commit_sha256",
        "board_artifact_sha256",
    }
    projection_sha = authority.ledger["upstream"]["label_free_input_projection"][
        "records_sha256"
    ]
    for record in records:
        image = record["image"]
        if set(record) != expected_record_keys:
            raise E24StagedRunnerError("decode/board barrier record field set drifted")
        decode_artifact, decode_commit, board_artifact, board_commit = _scene_paths(image)
        expected_paths = {
            "decode_commit_path": decode_commit.resolve(),
            "board_commit_path": board_commit.resolve(),
        }
        if any(Path(record[key]).resolve() != value for key, value in expected_paths.items()):
            raise E24StagedRunnerError("decode/board barrier path drifted")
        for path, key in (
            (decode_commit, "decode_commit_sha256"),
            (decode_artifact, "decode_artifact_sha256"),
            (board_commit, "board_commit_sha256"),
            (board_artifact, "board_artifact_sha256"),
        ):
            if not path.is_file() or _sha(path) != _lower_sha(record[key], label=key):
                raise E24StagedRunnerError("decode/board barrier artifact SHA drifted")
        try:
            input_payload, raw, _tiles, _spatial = e24_runner._load_input_bundle(
                image,
                authority.ledger_sha256,
                authority.ledger["run_contract_sha256"],
                projection_sha,
            )
            right, down = staged.dense_rd_from_raw(
                raw.arrays.candidate_ids, raw.arrays.candidate_scores
            )
            decode_provenance = _expected_decode_provenance(
                authority,
                image=image,
                seal_sha256=seal_sha256,
                input_payload=input_payload,
                right=right,
                down=down,
            )
            _decode, decode_manifest = staged.load_decode(
                decode_commit,
                expected_image=image,
                expected_provenance=decode_provenance,
            )
            board_provenance = _expected_board_provenance(
                authority,
                image=image,
                seal_sha256=seal_sha256,
                input_payload=input_payload,
                raw=raw,
                decode_manifest=decode_manifest,
            )
            boards, _board_manifest = staged.load_board_pair(
                board_commit,
                expected_image=image,
                expected_provenance=board_provenance,
            )
        except (e24_runner.E24RunnerError, staged.E24StagedContractError) as exc:
            raise E24StagedRunnerError(
                f"scene {image} inner decode/board provenance barrier failed"
            ) from exc
        if (
            staged.array_sha256(boards.right) != staged.array_sha256(right)
            or staged.array_sha256(boards.down) != staged.array_sha256(down)
        ):
            raise E24StagedRunnerError(
                f"scene {image} board artifact does not contain exact raw R/D"
            )
    return payload


def verify_expected_board_barrier(
    authority: AuthenticatedAuthority,
    seal_sha256: str,
    expected_board_barrier_sha256: str,
) -> dict[str, Any]:
    expected = _lower_sha(
        expected_board_barrier_sha256, label="expected decode/board barrier SHA"
    )
    payload = verify_board_barrier(authority, seal_sha256)
    if _sha(BARRIER_PATH) != expected:
        raise E24StagedRunnerError("decode/board barrier differs from caller-pinned SHA")
    return payload


def _metric_paths(image: int) -> tuple[Path, Path]:
    if image not in staged.CALIBRATION_IDS:
        raise E24StagedRunnerError("metric scene ID is outside 10..17")
    root = METRIC_ROOT / f"image_{image:04d}"
    return root / "request.json", root / "response.json"


def _pinned_rr96_inputs(image: int, validation_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read only pinned JSON lineage/RR rows; never construct a RawScene."""

    import eval_clean_score_oracle as e12
    import eval_e14_cc192_discovery as e14

    try:
        report_path = e24_eval._require_e_drive(
            e14.DEFAULT_E12_REPORT, label="pinned E12 report"
        )
    except e24_eval.E24EvaluatorContractError as exc:
        raise E24StagedRunnerError(str(exc)) from exc
    if not report_path.is_file() or _sha(report_path) != staged.PINNED_E12_REPORT_SHA256:
        raise E24StagedRunnerError("pinned E12 report SHA/path drifted")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise E24StagedRunnerError("pinned E12 report is unreadable") from exc
    if type(report) is not dict:
        raise E24StagedRunnerError("pinned E12 report root is not an object")
    if (
        report.get("schema") != e12.REPORT_SCHEMA
        or report.get("experiment") != e12.EXPERIMENT
        or report.get("status") != "complete"
        or report.get("protocol") != e12.ORACLE_PROTOCOL
        or report.get("protocol_sha256") != e12.canonical_digest(e12.ORACLE_PROTOCOL)
        or report.get("code_provenance") != e12.code_provenance()
        or report.get("scene_provenance_digest")
        != staged.PINNED_SCENE_PROVENANCE_DIGEST
    ):
        raise E24StagedRunnerError("pinned E12 report contract drifted")

    calibration_path = e12.DEFAULT_CALIBRATION_REPORT.resolve()
    if (
        not calibration_path.is_file()
        or _sha(calibration_path) != staged.PINNED_CALIBRATION_REPORT_SHA256
    ):
        raise E24StagedRunnerError("pinned RR96 calibration report drifted")
    try:
        calibration = e12.load_calibration_report(calibration_path)
        rr_rows_raw = report.get("rows", {}).get("RR")
        if type(rr_rows_raw) is not list:
            raise E24StagedRunnerError("pinned E12 RR rows are absent")
        e12.verify_rr_replay(rr_rows_raw, calibration)
        rr_rows = e12._rows_by_calibration_image(rr_rows_raw, label="pinned E12 RR")
        e14.verify_rr_means(rr_rows)
    except (e12.OracleContractError, e14.E14ContractError) as exc:
        raise E24StagedRunnerError("pinned RR96 replay verification failed") from exc

    provenance_rows = report.get("scene_provenance")
    if (
        type(provenance_rows) is not list
        or e12.canonical_digest(provenance_rows) != staged.PINNED_SCENE_PROVENANCE_DIGEST
    ):
        raise E24StagedRunnerError("pinned target/permutation lineage digest drifted")
    by_image = {
        row.get("image"): row for row in provenance_rows if type(row) is dict
    }
    lineage = by_image.get(image)
    rr = rr_rows.get(image)
    if (
        type(lineage) is not dict
        or type(rr) is not dict
        or lineage.get("validation_name") != validation_name
        or rr.get("validation_name") != validation_name
    ):
        raise E24StagedRunnerError("pinned scene identity/order drifted")
    arm = {key: rr[key] for key in staged._ARM_METRIC_KEYS}
    return dict(lineage), arm


def _replay_clean_target_only(
    image: int, *, expected_validation_name: str, expected_target_sha256: str
) -> np.ndarray:
    """Trusted target-lineage capability; retain only one detached clean image."""

    import random

    import torch
    from canvas_data import CanvasDataset
    from imgio import train_val_split

    if image not in staged.CALIBRATION_IDS:
        raise E24StagedRunnerError("target lineage is restricted to E24 IDs 10..17")
    _train_names, validation_names = train_val_split()
    if validation_names[image] != expected_validation_name:
        raise E24StagedRunnerError("target lineage validation name drifted")
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    dataset = CanvasDataset(
        validation_names[10:22], real_prob=0.0, seed=401234
    )
    requested = None
    for local in range(image - 10 + 1):
        sample = dataset[local]
        if local == image - 10:
            # The broker's only allowed sample field is clean.  Earlier samples
            # are replayed solely to preserve the frozen global RNG sequence.
            requested = sample["clean"]
        del sample
    if requested is None:
        raise E24StagedRunnerError("target lineage replay returned no sample")
    target = requested.permute(1, 2, 0).numpy()
    detached = np.ascontiguousarray(
        np.rint(target * 255.0).clip(0, 255).astype(np.uint8)
    )
    del requested, target, dataset
    if (
        detached.shape != (staged.IMAGE_SIZE, staged.IMAGE_SIZE, 3)
        or staged.array_sha256(detached)
        != _lower_sha(expected_target_sha256, label="expected target lineage SHA")
    ):
        raise E24StagedRunnerError("target lineage bytes differ from pinned E12")
    return detached


_METRIC_REQUEST_KEYS = {
    "schema",
    "schema_version",
    "image",
    "sequence_index",
    "previous_response_path",
    "previous_response_sha256",
    "ledger_sha256",
    "run_contract_sha256",
    "premetric_seal_sha256",
    "structural_report_sha256",
    "orchestration_receipt_sha256",
    "board_barrier_sha256",
    "board_commit_path",
    "board_commit_sha256",
    "raw_archive_path",
    "raw_archive_sha256",
    "validation_name",
    "metric_broker_contract_sha256",
    "e12_report_sha256",
    "calibration_report_sha256",
    "scene_provenance_digest",
    "e25_opened",
}


def _build_metric_request(
    authority: AuthenticatedAuthority,
    *,
    image: int,
    seal_sha256: str,
    barrier_sha256: str,
    previous_response_sha256: str,
) -> dict[str, Any]:
    sequence = image - staged.CALIBRATION_IDS[0]
    _decode_artifact, _decode_commit, _board_artifact, board_commit = _scene_paths(image)
    source = e24_runner._validated_upstream_projection(authority.ledger, image)
    raw = source["raw_cache"]
    previous_path = ""
    if sequence:
        previous_path = str(_metric_paths(image - 1)[1].resolve())
    return {
        "schema": staged.METRIC_REQUEST_SCHEMA,
        "schema_version": staged.SCHEMA_VERSION,
        "image": image,
        "sequence_index": sequence,
        "previous_response_path": previous_path,
        "previous_response_sha256": previous_response_sha256,
        "ledger_sha256": authority.ledger_sha256,
        "run_contract_sha256": authority.ledger["run_contract_sha256"],
        "premetric_seal_sha256": seal_sha256,
        "structural_report_sha256": authority.structural_report_sha256,
        "orchestration_receipt_sha256": authority.orchestration_receipt_sha256,
        "board_barrier_sha256": barrier_sha256,
        "board_commit_path": str(board_commit.resolve()),
        "board_commit_sha256": _sha(board_commit),
        "raw_archive_path": str(Path(raw["path"]).resolve()),
        "raw_archive_sha256": raw["file_sha256"],
        "validation_name": source["validation_name"],
        "metric_broker_contract_sha256": staged.METRIC_BROKER_CONTRACT_SHA256,
        "e12_report_sha256": staged.PINNED_E12_REPORT_SHA256,
        "calibration_report_sha256": staged.PINNED_CALIBRATION_REPORT_SHA256,
        "scene_provenance_digest": staged.PINNED_SCENE_PROVENANCE_DIGEST,
        "e25_opened": False,
    }


def _validate_metric_response(
    payload: Mapping[str, Any],
    *,
    request_sha256: str,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    expected_keys = {
        "schema",
        "schema_version",
        "status",
        "image",
        "sequence_index",
        "request_sha256",
        "previous_response_sha256",
        "row",
        "arrays_exported",
        "e25_opened",
    }
    if (
        type(payload) is not dict
        or set(payload) != expected_keys
        or payload["schema"] != staged.METRIC_RESPONSE_SCHEMA
        or payload["schema_version"] != staged.SCHEMA_VERSION
        or payload["status"] != "complete_row_only"
        or payload["image"] != request["image"]
        or payload["sequence_index"] != request["sequence_index"]
        or payload["request_sha256"] != request_sha256
        or payload["previous_response_sha256"]
        != request["previous_response_sha256"]
        or payload["arrays_exported"] is not False
        or payload["e25_opened"] is not False
    ):
        raise E24StagedRunnerError("metric response identity/order drifted")
    try:
        row = staged._validate_scene_row(payload["row"])
    except staged.E24StagedContractError as exc:
        raise E24StagedRunnerError("metric response row is malformed") from exc
    if (
        row["image"] != request["image"]
        or row["validation_name"] != request["validation_name"]
        or row["provenance"].get("metric_request_sha256") != request_sha256
        or row["provenance"].get("board_barrier_sha256")
        != request["board_barrier_sha256"]
        or row["provenance"].get("board_commit_sha256")
        != request["board_commit_sha256"]
    ):
        raise E24StagedRunnerError("metric response row binding drifted")
    return dict(payload)


def _validate_metric_request(
    authority: AuthenticatedAuthority,
    *,
    payload: Mapping[str, Any],
    seal_sha256: str,
    barrier_sha256: str,
) -> dict[str, Any]:
    if type(payload) is not dict or set(payload) != _METRIC_REQUEST_KEYS:
        raise E24StagedRunnerError("metric request field set drifted")
    image = payload.get("image")
    if type(image) is not int or image not in staged.CALIBRATION_IDS:
        raise E24StagedRunnerError("metric request scene is outside E24")
    sequence = image - staged.CALIBRATION_IDS[0]
    expected_previous = METRIC_CHAIN_GENESIS_SHA256
    expected_previous_path = ""
    if sequence:
        previous_path = _metric_paths(image - 1)[1].resolve()
        expected_previous_path = str(previous_path)
        if not previous_path.is_file():
            raise E24StagedRunnerError("metric response chain is truncated/reordered")
        expected_previous = _sha(previous_path)
    expected = _build_metric_request(
        authority,
        image=image,
        seal_sha256=seal_sha256,
        barrier_sha256=barrier_sha256,
        previous_response_sha256=expected_previous,
    )
    if dict(payload) != expected or payload["previous_response_path"] != expected_previous_path:
        raise E24StagedRunnerError("metric request authority/order/replay binding drifted")
    if sequence:
        prior_request_path, prior_response_path = _metric_paths(image - 1)
        prior_request = _load_json(prior_request_path, label="prior metric request")
        prior_response = _validate_metric_response(
            _load_json(prior_response_path, label="prior metric response"),
            request_sha256=_sha(prior_request_path),
            request=prior_request,
        )
        if prior_response["image"] != image - 1:
            raise E24StagedRunnerError("metric response chain order drifted")
    return dict(payload)


def metric_worker_once(
    authority: AuthenticatedAuthority,
    *,
    request_path: Path,
    response_path: Path,
    seal_sha256: str,
    expected_board_barrier_sha256: str,
) -> dict[str, Any]:
    """One-scene trusted capability.  The response contains metrics, never arrays."""

    barrier = verify_expected_board_barrier(
        authority, seal_sha256, expected_board_barrier_sha256
    )
    barrier_sha = _sha(BARRIER_PATH)
    request_file = _require_storage(request_path, label="metric request")
    expected_request, expected_response = _metric_paths(
        _load_json(request_file, label="metric request").get("image", -1)
    )
    if request_file != expected_request.resolve() or response_path.resolve() != expected_response.resolve():
        raise E24StagedRunnerError("metric request/response path is not canonical")
    request = _validate_metric_request(
        authority,
        payload=_load_json(request_file, label="metric request"),
        seal_sha256=seal_sha256,
        barrier_sha256=barrier_sha,
    )
    if response_path.exists():
        raise E24StagedRunnerError("metric response is create-once; replay refused")
    image = request["image"]
    record = next(item for item in barrier["records"] if item["image"] == image)
    if record["board_commit_sha256"] != request["board_commit_sha256"]:
        raise E24StagedRunnerError("request board is outside authenticated barrier")
    try:
        boards, _manifest = staged.load_board_pair(
            request["board_commit_path"], expected_image=image
        )
        permutation = e24_eval.load_original_permutation_member(
            request["raw_archive_path"],
            expected_sha256=request["raw_archive_sha256"],
        )
    except (staged.E24StagedContractError, e24_eval.E24EvaluatorContractError) as exc:
        raise E24StagedRunnerError("metric board/permutation capability failed") from exc

    lineage, pinned_rr96 = _pinned_rr96_inputs(image, request["validation_name"])
    if (
        Path(str(lineage.get("cache", ""))).resolve()
        != Path(request["raw_archive_path"]).resolve()
        or lineage.get("cache_sha256") != request["raw_archive_sha256"]
        or lineage.get("permutation_sha256") != staged.array_sha256(permutation)
    ):
        raise E24StagedRunnerError("raw permutation differs from pinned E12 lineage")
    target = _replay_clean_target_only(
        image,
        expected_validation_name=request["validation_name"],
        expected_target_sha256=lineage["target_sha256"],
    )
    request_sha = _sha(request_file)
    provenance = {
        "board_barrier_sha256": barrier_sha,
        "board_commit_sha256": request["board_commit_sha256"],
        "metric_broker_contract_sha256": staged.METRIC_BROKER_CONTRACT_SHA256,
        "metric_request_sha256": request_sha,
        "premetric_seal_sha256": seal_sha256,
        "raw_archive_sha256": request["raw_archive_sha256"],
        "e12_report_sha256": staged.PINNED_E12_REPORT_SHA256,
        "calibration_report_sha256": staged.PINNED_CALIBRATION_REPORT_SHA256,
        "scene_provenance_digest": staged.PINNED_SCENE_PROVENANCE_DIGEST,
    }
    try:
        row = staged.measure_scene_with_pinned_rr96(
            boards=boards,
            permutation=permutation,
            target=target,
            validation_name=request["validation_name"],
            provenance=provenance,
            pinned_rr96=pinned_rr96,
            expected_permutation_sha256=lineage["permutation_sha256"],
            expected_target_sha256=lineage["target_sha256"],
        )
    except staged.E24StagedContractError as exc:
        raise E24StagedRunnerError("trusted metric calculation failed") from exc
    finally:
        del permutation, target
        gc.collect()
    response = {
        "schema": staged.METRIC_RESPONSE_SCHEMA,
        "schema_version": staged.SCHEMA_VERSION,
        "status": "complete_row_only",
        "image": image,
        "sequence_index": request["sequence_index"],
        "request_sha256": request_sha,
        "previous_response_sha256": request["previous_response_sha256"],
        "row": row,
        "arrays_exported": False,
        "e25_opened": False,
    }
    try:
        staged.commit_canonical_create_once(response_path, response)
    except staged.E24StagedContractError as exc:
        raise E24StagedRunnerError(str(exc)) from exc
    return _validate_metric_response(response, request_sha256=request_sha, request=request)


def _spawn_metric_worker(
    *,
    request_path: Path,
    response_path: Path,
    ledger_path: Path,
    ledger_sha256: str,
    seal_sha256: str,
    barrier_sha256: str,
) -> None:
    env = os.environ.copy()
    for key in ("TEMP", "TMP", "TMPDIR", "JOBLIB_TEMP_FOLDER", "LIGHTGBM_TMPDIR"):
        env[key] = str(RUNTIME_ROOT)
    env["PYTHONPYCACHEPREFIX"] = str(PYCACHE_ROOT)
    command = [
        sys.executable,
        "-B",
        str(Path(__file__).resolve()),
        "_metric-worker",
        "--ledger",
        str(ledger_path.resolve()),
        "--ledger-sha256",
        ledger_sha256,
        "--seal-sha256",
        seal_sha256,
        "--board-barrier-sha256",
        barrier_sha256,
        "--request",
        str(request_path.resolve()),
        "--response",
        str(response_path.resolve()),
    ]
    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    if completed.returncode != 0:
        raise E24StagedRunnerError(
            "trusted metric subprocess failed without opening another scene: "
            + completed.stderr[-2000:]
        )


def run_staged_metric_phase(
    authority: AuthenticatedAuthority,
    seal_sha256: str,
    expected_board_barrier_sha256: str,
) -> dict[str, Any]:
    barrier = verify_expected_board_barrier(
        authority, seal_sha256, expected_board_barrier_sha256
    )
    barrier_sha = _sha(BARRIER_PATH)
    board_hashes = {
        str(record["image"]): record["board_commit_sha256"]
        for record in barrier["records"]
    }
    if REPORT_PATH.exists():
        try:
            return staged.validate_staged_report(
                REPORT_PATH,
                expected_ledger_sha256=authority.ledger_sha256,
                expected_run_contract_sha256=authority.ledger["run_contract_sha256"],
                expected_premetric_seal_sha256=seal_sha256,
                expected_structural_report_sha256=authority.structural_report_sha256,
                expected_orchestration_receipt_sha256=authority.orchestration_receipt_sha256,
                expected_board_barrier_sha256=barrier_sha,
                expected_board_commit_sha256=board_hashes,
            )
        except staged.E24StagedContractError as exc:
            raise E24StagedRunnerError("existing staged report failed authentication") from exc

    rows: list[dict[str, Any]] = []
    previous_sha = METRIC_CHAIN_GENESIS_SHA256
    for image in staged.CALIBRATION_IDS:
        request_path, response_path = _metric_paths(image)
        request = _build_metric_request(
            authority,
            image=image,
            seal_sha256=seal_sha256,
            barrier_sha256=barrier_sha,
            previous_response_sha256=previous_sha,
        )
        try:
            staged.commit_canonical_or_verify(request_path, request)
        except staged.E24StagedContractError as exc:
            raise E24StagedRunnerError("metric request create/replay failed") from exc
        request_sha = _sha(request_path)
        if not response_path.exists():
            _spawn_metric_worker(
                request_path=request_path,
                response_path=response_path,
                ledger_path=authority.ledger_path,
                ledger_sha256=authority.ledger_sha256,
                seal_sha256=seal_sha256,
                barrier_sha256=barrier_sha,
            )
        response = _validate_metric_response(
            _load_json(response_path, label="metric response"),
            request_sha256=request_sha,
            request=request,
        )
        rows.append(response["row"])
        previous_sha = _sha(response_path)

    rr96 = staged.rr96_verification_for_rows(rows)
    try:
        report = staged.build_staged_report(
            ledger_sha256=authority.ledger_sha256,
            run_contract_sha256=authority.ledger["run_contract_sha256"],
            premetric_seal_sha256=seal_sha256,
            structural_report_sha256=authority.structural_report_sha256,
            orchestration_receipt_sha256=authority.orchestration_receipt_sha256,
            board_barrier_sha256=barrier_sha,
            board_commit_sha256=board_hashes,
            rows=rows,
            rr96_verification=rr96,
        )
        staged.commit_canonical_create_once(REPORT_PATH, report)
        return staged.validate_staged_report(
            REPORT_PATH,
            expected_ledger_sha256=authority.ledger_sha256,
            expected_run_contract_sha256=authority.ledger["run_contract_sha256"],
            expected_premetric_seal_sha256=seal_sha256,
            expected_structural_report_sha256=authority.structural_report_sha256,
            expected_orchestration_receipt_sha256=authority.orchestration_receipt_sha256,
            expected_board_barrier_sha256=barrier_sha,
            expected_board_commit_sha256=board_hashes,
        )
    except staged.E24StagedContractError as exc:
        raise E24StagedRunnerError("staged report create/validation failed") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "smoke",
            "verify-authority",
            "freeze",
            "prepare-label-free",
            "verify-board-barrier",
            "staged-eval",
            "verify-report",
            "_metric-worker",
        ),
    )
    parser.add_argument("--ledger", type=Path, default=e24_runner.DEFAULT_LEDGER)
    parser.add_argument("--ledger-sha256", default=EXPECTED_GENERATION3_LEDGER_SHA256)
    parser.add_argument("--seal-sha256", default="")
    parser.add_argument("--board-barrier-sha256", default="")
    parser.add_argument("--request", type=Path)
    parser.add_argument("--response", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.mode == "smoke":
        print(
            staged.canonical_json_bytes(
                {
                    "status": "data_free",
                    "protocol_sha256": staged.PROTOCOL_SHA256,
                    "storage_root": str(STAGED_ROOT),
                    "metric_broker": "frozen_data_free_not_invoked",
                    "metric_broker_contract_sha256": staged.METRIC_BROKER_CONTRACT_SHA256,
                    "target_or_e25_opened": False,
                }
            ).decode("ascii"),
            end="",
        )
        return
    authority = authenticate_authority(args.ledger, args.ledger_sha256)
    if args.mode == "verify-authority":
        print(
            staged.canonical_json_bytes(
                {
                    "status": "pass",
                    "ledger_sha256": authority.ledger_sha256,
                    "structural_report_sha256": authority.structural_report_sha256,
                    "orchestration_receipt_sha256": authority.orchestration_receipt_sha256,
                    "target_or_e25_opened": False,
                }
            ).decode("ascii"),
            end="",
        )
        return
    if args.mode == "freeze":
        _payload, digest = freeze_premetric_seal(authority)
        print(
            staged.canonical_json_bytes(
                {"status": "frozen", "path": str(SEAL_PATH.resolve()), "sha256": digest}
            ).decode("ascii"),
            end="",
        )
        return
    if not args.seal_sha256:
        raise E24StagedRunnerError("--seal-sha256 is mandatory after freeze")
    if args.mode == "prepare-label-free":
        result = prepare_label_free_staged(authority, args.seal_sha256)
        print(
            staged.canonical_json_bytes(
                {
                    "status": result["status"],
                    "barrier_path": str(BARRIER_PATH.resolve()),
                    "barrier_sha256": _sha(BARRIER_PATH),
                    "target_or_e25_opened": False,
                }
            ).decode("ascii"),
            end="",
        )
    elif args.mode == "verify-board-barrier":
        result = verify_board_barrier(authority, args.seal_sha256)
        print(
            staged.canonical_json_bytes(
                {"status": result["status"], "target_or_e25_opened": False}
            ).decode("ascii"),
            end="",
        )
    elif args.mode == "staged-eval":
        if not args.board_barrier_sha256:
            raise E24StagedRunnerError(
                "--board-barrier-sha256 is mandatory before any metric capability"
            )
        result = run_staged_metric_phase(
            authority, args.seal_sha256, args.board_barrier_sha256
        )
        print(
            staged.canonical_json_bytes(
                {
                    "status": result["status"],
                    "stage": result["stage"],
                    "passed": result["decision"]["passed"],
                    "report_path": str(REPORT_PATH.resolve()),
                    "report_sha256": _sha(REPORT_PATH),
                    "e25_opened": False,
                }
            ).decode("ascii"),
            end="",
        )
    elif args.mode == "verify-report":
        if not args.board_barrier_sha256:
            raise E24StagedRunnerError("--board-barrier-sha256 is mandatory")
        barrier = verify_expected_board_barrier(
            authority, args.seal_sha256, args.board_barrier_sha256
        )
        board_hashes = {
            str(record["image"]): record["board_commit_sha256"]
            for record in barrier["records"]
        }
        try:
            result = staged.validate_staged_report(
                REPORT_PATH,
                expected_ledger_sha256=authority.ledger_sha256,
                expected_run_contract_sha256=authority.ledger["run_contract_sha256"],
                expected_premetric_seal_sha256=args.seal_sha256,
                expected_structural_report_sha256=authority.structural_report_sha256,
                expected_orchestration_receipt_sha256=authority.orchestration_receipt_sha256,
                expected_board_barrier_sha256=args.board_barrier_sha256,
                expected_board_commit_sha256=board_hashes,
            )
        except staged.E24StagedContractError as exc:
            raise E24StagedRunnerError("staged report authentication failed") from exc
        print(
            staged.canonical_json_bytes(
                {
                    "status": "pass" if result["decision"]["passed"] else "fail",
                    "stage": result["stage"],
                    "report_sha256": _sha(REPORT_PATH),
                    "e25_opened": False,
                }
            ).decode("ascii"),
            end="",
        )
    elif args.mode == "_metric-worker":
        if (
            not args.board_barrier_sha256
            or args.request is None
            or args.response is None
        ):
            raise E24StagedRunnerError(
                "metric worker requires barrier SHA and canonical request/response paths"
            )
        result = metric_worker_once(
            authority,
            request_path=args.request,
            response_path=args.response,
            seal_sha256=args.seal_sha256,
            expected_board_barrier_sha256=args.board_barrier_sha256,
        )
        print(
            staged.canonical_json_bytes(
                {
                    "status": result["status"],
                    "image": result["image"],
                    "arrays_exported": False,
                    "e25_opened": False,
                }
            ).decode("ascii"),
            end="",
        )


if __name__ == "__main__":
    main()

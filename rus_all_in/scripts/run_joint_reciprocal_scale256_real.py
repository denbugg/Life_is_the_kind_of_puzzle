#!/usr/bin/env python3
"""Run one signed FIT256/DEV64 scale transition through the frozen v2 engine.

The wrapper owns only protocol validation and stage separation.  Training,
target-free DEV freezing, and post-freeze DEV scoring are delegated to the
unchanged v2 runner.  FIT may read the signed FIT caches (including their known
synthetic target slots), but it cannot open reserved DEV pixels or labels.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.synthetic_socket_evaluation import names_digest

try:
    from scripts import materialize_joint_reciprocal_scale_fit_cache as materializer
    from scripts import run_joint_reciprocal_tri_emitter_real as base
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    import materialize_joint_reciprocal_scale_fit_cache as materializer
    import run_joint_reciprocal_tri_emitter_real as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/joint_reciprocal_scale256_real_unsigned_template_v1.json"
)
DEFAULT_EXPERIMENT_DIR = (
    PROJECT_ROOT
    / "outputs/joint-reciprocal-tri-emitter-verifier/"
    "scale-fit256-draw2-dev64-real-v1"
)

CONFIG_SCHEMA = "aiijc-joint-reciprocal-scale256-real-protocol-v1"
SIGNED_STATUS = "signed-fixed-protocol"
BLOCKED_STATUS = "unsigned-template-blocked-awaiting-final-review"
FIT_SOURCE_COUNT = 256
DEV_SOURCE_COUNT = 64
FIT_CASE_COUNT = 512
FIT_DRAWS = (0, 1)
SELECTION_SEED = 20260913
SELECTION_NAMESPACE = "aiijc-joint-reciprocal-scale256-fit256-dev64-v1"
FIT_DIGEST = "5b3d9c56d8bc2eaf0fec6cf54b1dcc3c888ca5647d7fee4e2860928339708370"
DEV_DIGEST = "5c6cb5b9b204a38c78e79936ff34235dae9896cfc13d6edaf12dfad635bcdb8e"
PARENT_EXCLUSION_COUNT = 1120
PARENT_EXCLUSION_DIGEST = (
    "d93311aa39c3c4ccc349928a3e6269103f540affde585e889e3980d2f21227e2"
)
FIT_CACHE_REPORT_SHA256 = (
    "04c0bf7edebc8809f854a95bf97b361ad28fec5b12eb26e54d4c780b2c8d823d"
)

# These are deliberately literal.  The scale protocol must fail if any frozen
# v2 engine or upstream lineage bytes drift, even if a config is re-signed.
IMMUTABLE_BASE_ARTIFACTS: dict[str, tuple[str, str]] = {
    "fit_cache_report": (
        "outputs/joint-reciprocal-tri-emitter-verifier/"
        "scale-fit256-draw2-cache-v1/report.json",
        FIT_CACHE_REPORT_SHA256,
    ),
    "scale_cache_config": (
        "configs/joint_reciprocal_scale256_fit_cache_preregistered_v1.json",
        "3e397e7ff3a565de2b1ab412f71f8e5b7d500649b40a51be2ed97805ecc7344e",
    ),
    "manifest": (
        "data/interim/validation_manifest.json",
        "4781e370e092ad272c63e6d5165b25951aaf93fae5fde74c75d534a9e8efc9da",
    ),
    "socket_checkpoint": (
        "outputs/socket-matcher/v2-d64-train1024-s1600-r400-dev32/"
        "socket_matcher.pt",
        "0e9df49a503c65aac7f1468e9acd6a074a5e658ae8b61f8954be086272c49670",
    ),
    "adapter1600_checkpoint": (
        "outputs/fullres-retrieval-adapter/scale1600-local16-v1/"
        "adapter_step1600.pt",
        "51beee8dea615e00440f90737ee537244dcf26934e9e292ac7a33bea235e6a48",
    ),
    "dino_checkpoint": (
        "artifacts/foundation-semantics/dinov2-vits14-official/"
        "dinov2_vits14_pretrain.pth",
        "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9",
    ),
    "socket_parent_report": (
        "outputs/socket-matcher/v2-d64-train1024-s1600-r400-dev32/report.json",
        "ff461744ed214848e65bfac5f006bc543776a057c8df7ab68c021dabca3942e5",
    ),
    "adapter_parent_report": (
        "outputs/fullres-retrieval-adapter/scale1600-local16-v1/report.json",
        "47ce8b176d2da5b6c278af6bc66be27464d87465cca6507a94e57e569f0ec796",
    ),
    "capacity_report": (
        "outputs/joint-reciprocal-tri-emitter-verifier/"
        "capacity4x4-collision-v2/report.json",
        "94b769da114553ec98212abca4de758a587dc52661c54c66fd4eda03e8b8ed7c",
    ),
    "base_module": (
        "src/aiijc_puzzle/joint_reciprocal_tri_emitter_verifier.py",
        "13a5a649815c1dd48d9db69ca30d3303e3f70237688657156f4c19d7ae196ad3",
    ),
    "base_runner": (
        "scripts/run_joint_reciprocal_tri_emitter_real.py",
        "7b6f760155f86c9c3c465a8833c9c71687123b0ac248a64d55b91d0f8c1ad82c",
    ),
    "base_runner_test": (
        "tests/test_run_joint_reciprocal_tri_emitter_real.py",
        "1475fbf62b35302ca2582e52ab8d1c0f8c6da832ad93ec734334ec7d88813146",
    ),
    "materializer_runner": (
        "scripts/materialize_joint_reciprocal_scale_fit_cache.py",
        "aed6817da76d6a9d307e3931461040c89c25793dec72063884b6ad91bfa060b3",
    ),
}

NEW_ARTIFACT_PATHS = {
    "scale_wrapper": "scripts/run_joint_reciprocal_scale256_real.py",
    "generic_target_free_freezer": (
        "scripts/freeze_joint_reciprocal_fit_heads_target_free_generic.py"
    ),
    "scale_wrapper_test": "tests/test_run_joint_reciprocal_scale256_real.py",
    "generic_target_free_freezer_test": (
        "tests/test_freeze_joint_reciprocal_fit_heads_target_free_generic.py"
    ),
    "unsigned_template": (
        "configs/joint_reciprocal_scale256_real_unsigned_template_v1.json"
    ),
}
REQUIRED_FROZEN_INPUTS = frozenset(IMMUTABLE_BASE_ARTIFACTS) | frozenset(
    NEW_ARTIFACT_PATHS
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", required=True, choices=("fit", "freeze-dev", "score-dev")
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR
    )
    parser.add_argument("--manifest", type=Path, default=base.prior.roster.DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=base.prior.roster.DEFAULT_TARGETS)
    parser.add_argument(
        "--socket-checkpoint", type=Path, default=base.prior.SOCKET_CHECKPOINT
    )
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--allow-nondeterministic-mps", action="store_true")
    return parser.parse_args(argv)


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _require_float(
    values: Mapping[str, Any], key: str, expected: float, *, section: str
) -> None:
    if not math.isclose(
        float(values.get(key, math.nan)), expected, rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError(f"fixed scale {section} changed: {key}")


def require_exact_contract(config: Mapping[str, Any]) -> None:
    """Validate the signed scale contract without opening any source pixels."""

    if config.get("schema") != CONFIG_SCHEMA:
        raise RuntimeError("scale real protocol schema changed")
    model = config.get("fixed_model", {})
    expected_model = {
        "architecture": "joint-reciprocal-tri-emitter-verifier-v1",
        "candidate_roster": "raw+adapter1600+DINO-top32-stable-union",
        "dino_projection_dim": 16,
        "auxiliary_dim": 19,
        "width": 32,
        "hidden": 96,
        "one_endpoint_one_seed_no_checkpoint_selection": True,
        "capacity_checkpoint_reuse": False,
    }
    if model != expected_model:
        raise RuntimeError("fixed scale architecture/endpoint contract changed")

    objective = config.get("objective", {})
    expected_objective = {
        "row_cross_entropy_weight": 1.0,
        "column_cross_entropy_weight": 1.0,
        "confidence_bce_weight": 0.25,
        "delta_l2_weight": 0.001,
        "softmin_tau": 0.25,
        "fixed_reciprocal_fraction_per_axis_per_board": 0.05,
    }
    for key, expected in expected_objective.items():
        _require_float(objective, key, expected, section="objective")
    if objective.get("learned_none") is not True:
        raise RuntimeError("fixed scale learned-NONE contract changed")

    training = config.get("training", {})
    expected_training = {
        "from_scratch": True,
        "seed": 20260913,
        "epochs": 3,
        "optimizer_updates": 1752,
        "learning_rate": 0.0003,
        "weight_decay": 0.0001,
        "gradient_clip": 1.0,
        "torch_threads": 6,
        "checkpoint_selection": "single-final-endpoint-no-selection",
    }
    if training != expected_training:
        raise RuntimeError("fixed scale from-scratch training contract changed")

    source = config.get("source_contract", {})
    expected_source = {
        "selection_namespace": SELECTION_NAMESPACE,
        "selection_seed": SELECTION_SEED,
        "fit_source_count": FIT_SOURCE_COUNT,
        "fit_digest": FIT_DIGEST,
        "fit_draw_indices": list(FIT_DRAWS),
        "fit_case_seed": 20260914,
        "fit_case_count": FIT_CASE_COUNT,
        "reserved_dev_source_count": DEV_SOURCE_COUNT,
        "reserved_dev_digest": DEV_DIGEST,
        "dev_draw_index": 0,
        "dev_case_seed": 20260908,
        "parent_exclusion_count": PARENT_EXCLUSION_COUNT,
        "parent_exclusion_digest": PARENT_EXCLUSION_DIGEST,
    }
    if source != expected_source:
        raise RuntimeError("fixed FIT256/DEV64 source contract changed")

    gate = config.get("dev_gate", {})
    for key, expected in {
        "pooled_r1_gain_minimum": 0.005,
        "pooled_r5_gain_minimum": 0.0,
        "pooled_fixed_head_precision_gain_minimum": 0.02,
        "per_axis_gain_minimum": 0.0,
    }.items():
        _require_float(gate, key, expected, section="DEV gate")
    if not isinstance(gate.get("logic"), str):
        raise RuntimeError("scale DEV gate logic is missing")

    stages = config.get("stage_contract", {})
    expected_stages = {
        "fit": {
            "fit_cache_pixels_opened": False,
            "known_synthetic_target_slots_opened": True,
            "reserved_dev_metadata_only": True,
            "reserved_dev_pixels_opened": False,
            "reserved_dev_labels_opened": False,
        },
        "freeze_dev": {
            "requires_completed_fit_endpoint": True,
            "reserved_dev_pixels_opened_for_target_free_synthetic_cases": True,
            "reserved_dev_labels_opened": False,
        },
        "score_dev": {
            "requires_verified_target_free_freeze": True,
            "reserved_dev_labels_opened_only_after_freeze_hashes": True,
        },
        "terminal_or_competition_test_mode": False,
        "warmstart_or_resume": False,
        "weco_run": False,
    }
    if stages != expected_stages:
        raise RuntimeError("scale stage access contract changed")

    frozen = config.get("frozen_inputs", {})
    if not isinstance(frozen, Mapping) or set(frozen) != REQUIRED_FROZEN_INPUTS:
        raise RuntimeError("scale frozen input inventory changed")
    for key, (path, digest) in IMMUTABLE_BASE_ARTIFACTS.items():
        if frozen.get(key) != {"path": path, "sha256": digest}:
            raise RuntimeError(f"immutable base artifact contract changed: {key}")
    for key, path in NEW_ARTIFACT_PATHS.items():
        artifact = frozen.get(key, {})
        digest = artifact.get("sha256")
        if artifact.get("path") != path or not isinstance(digest, str) or len(digest) != 64:
            raise RuntimeError(f"new scale artifact record is invalid: {key}")


def _load_json_artifact(config: Mapping[str, Any], key: str) -> dict[str, Any]:
    path = _project_path(config["frozen_inputs"][key]["path"])
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_cache_report(
    config: Mapping[str, Any], fit: tuple[str, ...], dev: tuple[str, ...]
) -> None:
    report = _load_json_artifact(config, "fit_cache_report")
    if report.get("schema") != "aiijc-tri-emitter-edge-verifier-report-v1":
        raise RuntimeError("scale FIT cache compatible schema changed")
    if report.get("producer_schema") != "aiijc-joint-reciprocal-scale-fit-cache-v1":
        raise RuntimeError("scale FIT cache producer schema changed")
    if report.get("status") != "complete-cache-only-ready-for-separate-fit-preregistration":
        raise RuntimeError("scale FIT cache is not complete")
    protocol = report.get("protocol", {})
    if tuple(protocol.get("fit_filenames", ())) != fit:
        raise RuntimeError("scale cache report FIT roster changed")
    if protocol.get("fit_digest") != FIT_DIGEST:
        raise RuntimeError("scale cache report FIT digest changed")
    if protocol.get("reserved_dev_source_count") != DEV_SOURCE_COUNT:
        raise RuntimeError("scale cache report reserved DEV count changed")
    if protocol.get("reserved_dev_digest") != DEV_DIGEST:
        raise RuntimeError("scale cache report reserved DEV digest changed")
    if protocol.get("reserved_dev_opened") is not False:
        raise RuntimeError("scale cache report already opened reserved DEV")
    rows = tuple(report.get("fit_cache", {}).get("rows", ()))
    observed = tuple((row.get("source_filename"), row.get("draw_index")) for row in rows)
    expected = tuple((name, draw) for name in fit for draw in FIT_DRAWS)
    if observed != expected or len(rows) != FIT_CASE_COUNT:
        raise RuntimeError("scale FIT cache row roster changed")
    scope = report.get("scope", {})
    if scope.get("reserved_dev_pixels_or_labels_opened") is not False:
        raise RuntimeError("cache materialization touched reserved DEV")
    if any(name in set(fit) for name in dev):
        raise RuntimeError("scale FIT and reserved DEV rosters overlap")


def _derive_runtime_config(
    config: Mapping[str, Any], scale_cache_config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, tuple[str, ...]]]:
    manifest = _load_json_artifact(config, "manifest")
    socket_report = _load_json_artifact(config, "socket_parent_report")
    adapter_report = _load_json_artifact(config, "adapter_parent_report")
    _fit_records, audit = materializer.validate_metadata_rosters(
        scale_cache_config, manifest, socket_report, adapter_report
    )
    source = scale_cache_config["source_protocol"]
    fit = tuple(source["fit_filenames"])
    dev = tuple(source["reserved_dev_filenames"])
    groups = materializer._parent_source_groups(socket_report, adapter_report)
    excluded = tuple(sorted(set().union(*(set(values) for values in groups.values()))))
    opened = tuple(groups["adapter_opened_local"])
    terminal = tuple(groups["adapter_terminal_owned"])
    audit_excluded = tuple(sorted(set(excluded) - set(opened) - set(terminal)))
    if (
        len(fit) != FIT_SOURCE_COUNT
        or names_digest(fit) != FIT_DIGEST
        or len(dev) != DEV_SOURCE_COUNT
        or names_digest(dev) != DEV_DIGEST
    ):
        raise RuntimeError("derived FIT256/DEV64 roster changed")
    if (
        audit.get("parent_exclusion_count") != PARENT_EXCLUSION_COUNT
        or audit.get("parent_exclusion_digest") != PARENT_EXCLUSION_DIGEST
        or (set(fit) | set(dev)) & set(excluded)
    ):
        raise RuntimeError("derived parent-lineage exclusions changed")
    _validate_cache_report(config, fit, dev)
    runtime = {
        "fixed_model": dict(config["fixed_model"]),
        "objective": dict(config["objective"]),
        "training": dict(config["training"]),
        "source_protocol": {
            "fit_filenames": list(fit),
            "fit_digest": FIT_DIGEST,
            "fit_draw_indices": list(FIT_DRAWS),
            "dev_filenames": list(dev),
            "dev_digest": DEV_DIGEST,
            "dev_draw_index": 0,
            "dev_case_seed": 20260908,
            "opened_local16_owned_filenames": list(opened),
            "terminal16_owned_filenames": list(terminal),
            "source_audit_excluded_filenames": list(audit_excluded),
        },
        "dev_gate": dict(config["dev_gate"]),
        "frozen_inputs": dict(config["frozen_inputs"]),
    }
    rosters = base.validate_source_rosters(runtime)
    return runtime, rosters


def load_signed_runtime_config(
    path: Path,
) -> tuple[dict[str, Any], str, dict[str, tuple[str, ...]]]:
    """Verify every byte and derive rosters using metadata only."""

    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"scale real protocol is missing: {resolved}")
    config = json.loads(resolved.read_text(encoding="utf-8"))
    if config.get("status") == BLOCKED_STATUS:
        raise RuntimeError("scale real unsigned template is intentionally blocked")
    sidecar = resolved.with_suffix(resolved.suffix + ".sha256")
    if config.get("status") != SIGNED_STATUS or not sidecar.is_file():
        raise RuntimeError("scale real protocol is not signed/fixed")
    digest = sha256_file(resolved)
    if sidecar.read_text(encoding="utf-8").split()[0] != digest:
        raise RuntimeError("scale real protocol sidecar mismatch")
    require_exact_contract(config)
    for key, artifact in config["frozen_inputs"].items():
        target = _project_path(artifact["path"])
        if not target.is_file() or sha256_file(target) != artifact["sha256"]:
            raise RuntimeError(f"frozen scale input changed: {key} ({target})")
    scale_cache_path = _project_path(
        config["frozen_inputs"]["scale_cache_config"]["path"]
    )
    scale_cache_config, cache_config_sha = materializer.load_signed_config(
        scale_cache_path
    )
    if cache_config_sha != IMMUTABLE_BASE_ARTIFACTS["scale_cache_config"][1]:
        raise RuntimeError("scale cache preregistration config changed")
    runtime, rosters = _derive_runtime_config(config, scale_cache_config)
    return runtime, digest, rosters


def _validate_runtime_paths(args: argparse.Namespace, mode: str) -> None:
    expected_manifest = _project_path(IMMUTABLE_BASE_ARTIFACTS["manifest"][0])
    expected_socket = _project_path(IMMUTABLE_BASE_ARTIFACTS["socket_checkpoint"][0])
    if args.manifest.resolve() != expected_manifest:
        raise RuntimeError("runtime manifest differs from signed scale protocol")
    if args.socket_checkpoint.resolve() != expected_socket:
        raise RuntimeError("runtime socket checkpoint differs from signed scale protocol")
    if args.device == "mps" and not args.allow_nondeterministic_mps:
        raise ValueError("MPS requires explicit nondeterminism consent")
    if args.device == "cpu" and args.allow_nondeterministic_mps:
        raise ValueError("MPS consent is incompatible with CPU")
    experiment = args.experiment_dir.resolve()
    if mode == "fit" and experiment.exists():
        raise FileExistsError("scale experiment already exists; resume/overwrite forbidden")
    if mode != "fit" and not experiment.is_dir():
        raise FileNotFoundError("completed scale FIT experiment is missing")


def run_mode(args: argparse.Namespace) -> dict[str, Any]:
    _validate_runtime_paths(args, args.mode)
    config, config_sha, rosters = load_signed_runtime_config(args.config)
    if args.mode == "fit":
        return base.run_fit(args, config, config_sha, rosters)
    if args.mode == "freeze-dev":
        return base.run_freeze_dev(args, config, config_sha, rosters)
    return base.run_score_dev(args, config, config_sha, rosters)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_mode(args)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.mode == "score-dev" and not report["gate"]["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

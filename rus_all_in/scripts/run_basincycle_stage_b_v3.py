#!/usr/bin/env python3
"""Run the separately bound BasinCycle Stage-B MPS-reduction v3 retry.

This CLI retains the v2 staged proposal transfer and additionally installs
finite sentinels for the three padded all-masked action-feature reductions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import aiijc_puzzle.basincycle_stage_b_runner as stage_b_runner
from aiijc_puzzle.basincycle_stage_b_mps_reductions_v3 import (
    install_mps_reductions_v3,
)
from aiijc_puzzle.basincycle_stage_b_runner import (
    atomic_write_json,
    audit_protocol,
    choose_device,
    fit_model,
    freeze_eval_predictions,
    load_final_model,
    load_json,
    score_frozen_predictions,
    sha256_file,
    validate_execution_acknowledgement,
    validate_freeze_bundle,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/basincycle_stage_b_6x6_preregistered_v1.json"
DEFAULT_BINDING = PROJECT_ROOT / "configs/basincycle_stage_b_execution_binding_v3.json"
EXECUTION_REVISION = "mps-finite-masked-reductions-v3"
BASE_EXECUTION_BINDING_SCHEMAS = {
    "aiijc-basincycle-stage-b-execution-binding-v1",
    "aiijc-basincycle-stage-b-execution-binding-v2",
}
EXECUTION_BINDING_SCHEMA = "aiijc-basincycle-stage-b-execution-binding-v3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("audit", "fit", "freeze", "score"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--binding", type=Path, default=DEFAULT_BINDING)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--freeze-dir", type=Path)
    parser.add_argument("--review-acknowledgement", default="")
    return parser.parse_args()


def _bound_path(
    value: Path | None,
    *,
    binding: dict,
    key: str,
    option: str,
) -> Path:
    expected = (PROJECT_ROOT / binding["artifacts"][key]).resolve()
    observed = expected if value is None else value.resolve()
    if observed != expected:
        raise ValueError(f"{option} must equal the one-run bound v3 path: {expected}")
    return expected


def _validate_v3_boundary(config_path: Path, binding_path: Path) -> dict:
    if config_path != DEFAULT_CONFIG.resolve():
        raise ValueError("v3 retry requires the exact signed Stage-B scientific config path")
    if binding_path != DEFAULT_BINDING.resolve():
        raise ValueError("v3 retry requires the exact MPS-reduction v3 binding path")
    binding = load_json(binding_path)
    if binding.get("schema") != EXECUTION_BINDING_SCHEMA:
        raise ValueError("wrong Stage-B v3 execution-binding schema")
    if binding.get("execution_revision") != EXECUTION_REVISION:
        raise ValueError("wrong Stage-B v3 execution revision")
    reduction = binding.get("mps_reduction_fix", {})
    expected = {
        "replacement": "finite-dtype-sentinel-plus-explicit-has-any-zero",
        "scientific_semantics_changed": False,
        "all_masked_reduction_count_fixed": 3,
    }
    for key, value in expected.items():
        if reduction.get(key) != value:
            raise ValueError(f"Stage-B v3 reduction binding differs at {key}")
    return binding


def _install_v3_runtime() -> None:
    current_schema = stage_b_runner.EXECUTION_BINDING_SCHEMA
    allowed = BASE_EXECUTION_BINDING_SCHEMAS | {EXECUTION_BINDING_SCHEMA}
    if current_schema not in allowed:
        raise RuntimeError("Stage-B execution-binding validator was already replaced")
    stage_b_runner.EXECUTION_BINDING_SCHEMA = EXECUTION_BINDING_SCHEMA
    install_mps_reductions_v3()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    binding_path = args.binding.resolve()
    binding = _validate_v3_boundary(config_path, binding_path)
    _install_v3_runtime()
    audit = audit_protocol(
        project_root=PROJECT_ROOT,
        config_path=config_path,
        binding_path=binding_path,
    )
    if args.mode == "audit":
        print(json.dumps(audit, indent=2, sort_keys=True))
        return

    config = load_json(config_path)
    validate_execution_acknowledgement(binding, args.review_acknowledgement)
    config_sha256 = str(audit["scientific_config_sha256"])
    binding_sha256 = str(audit["execution_binding_sha256"])
    device = choose_device(args.device)
    targets_root = (PROJECT_ROOT / binding["data"]["organizer_train_targets"]).resolve()
    socket_checkpoint = (
        PROJECT_ROOT / config["frozen_inputs"]["socket_v2_checkpoint"]["path"]
    ).resolve()

    if args.mode == "fit":
        output_dir = _bound_path(
            args.output_dir,
            binding=binding,
            key="fit_output_dir",
            option="--output-dir",
        )
        report = fit_model(
            config=config,
            binding=binding,
            config_sha256=config_sha256,
            binding_sha256=binding_sha256,
            targets_root=targets_root,
            socket_checkpoint=socket_checkpoint,
            output_dir=output_dir,
            device=device,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    checkpoint = _bound_path(
        args.checkpoint,
        binding=binding,
        key="final_checkpoint",
        option="--checkpoint",
    )
    model, _ = load_final_model(
        checkpoint,
        config=config,
        config_sha256=config_sha256,
        binding_sha256=binding_sha256,
        device=device,
    )
    if args.mode == "freeze":
        output_dir = _bound_path(
            args.output_dir,
            binding=binding,
            key="freeze_output_dir",
            option="--output-dir",
        )
        receipt = freeze_eval_predictions(
            model=model,
            config=config,
            binding=binding,
            config_sha256=config_sha256,
            binding_sha256=binding_sha256,
            checkpoint_path=checkpoint,
            targets_root=targets_root,
            socket_checkpoint=socket_checkpoint,
            output_dir=output_dir,
            device=device,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return

    freeze_dir = _bound_path(
        args.freeze_dir,
        binding=binding,
        key="freeze_output_dir",
        option="--freeze-dir",
    )
    output_dir = _bound_path(
        args.output_dir,
        binding=binding,
        key="score_output_dir",
        option="--output-dir",
    )
    if output_dir.exists():
        raise FileExistsError("v3 score output directory exists; overwrite is forbidden")
    output_dir.mkdir(parents=True)
    receipt_path = freeze_dir / "freeze_receipt.json"
    receipt = load_json(receipt_path)
    arrays = validate_freeze_bundle(
        bundle_path=freeze_dir / "target_free_predictions.npz",
        receipt=receipt,
        config_sha256=config_sha256,
        binding_sha256=binding_sha256,
    )
    if receipt.get("model_sha256") != sha256_file(checkpoint):
        raise ValueError("score checkpoint differs from the target-free freeze receipt")
    report = score_frozen_predictions(
        arrays=arrays,
        receipt=receipt,
        config=config,
        config_sha256=config_sha256,
        binding=binding,
        binding_sha256=binding_sha256,
    )
    report["freeze_receipt_file_sha256"] = sha256_file(receipt_path)
    atomic_write_json(output_dir / "score_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

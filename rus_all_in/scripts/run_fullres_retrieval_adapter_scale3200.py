#!/usr/bin/env python3
"""Run the signed fixed 1600-to-3200 retrieval-adapter scale test."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from aiijc_puzzle.fullres_retrieval_adapter import retrieval_adapter_contract
from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.socket_sorter_production import load_socket_checkpoint

try:
    from scripts import run_fullres_retrieval_adapter as pilot
    from scripts import run_fullres_retrieval_adapter_scale1600 as scale
except ModuleNotFoundError:  # Direct ``python scripts/run_*.py`` execution.
    import run_fullres_retrieval_adapter as pilot
    import run_fullres_retrieval_adapter_scale1600 as scale

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs/fullres_retrieval_adapter_scale3200_preregistered_v1.json"
CONFIG_SHA256 = "792dc3304bcd173fd954bf3f0484338c7c690afc28e3b597c9c1d6d362a91a82"
PREVIOUS_REPORT = (
    PROJECT_ROOT / "outputs/fullres-retrieval-adapter/scale1600-local16-v1/report.json"
)
PREVIOUS_REPORT_SHA256 = "47ce8b176d2da5b6c278af6bc66be27464d87465cca6507a94e57e569f0ec796"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/fullres-retrieval-adapter/scale3200-local16-v1"
CHECKPOINT_STEPS = (1600, 3200)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=pilot.DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=pilot.DEFAULT_TARGETS)
    parser.add_argument("--socket-checkpoint", type=Path, default=pilot.DEFAULT_SOCKET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--allow-nondeterministic-mps", action="store_true")
    return parser.parse_args(argv)


def _configure_shared_runner() -> None:
    """Bind the unchanged implementation to the signed 3200-step protocol."""
    scale.CONFIG = CONFIG
    scale.CONFIG_SHA256 = CONFIG_SHA256
    scale.PILOT_REPORT = PREVIOUS_REPORT
    scale.PILOT_REPORT_SHA256 = PREVIOUS_REPORT_SHA256
    scale.CHECKPOINT_STEPS = CHECKPOINT_STEPS


def _record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        shown = str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        shown = str(resolved)
    return {"path": shown, "sha256": sha256_file(resolved)}


def _checkpoint_comparison(
    cases: list[pilot.FrozenCase],
    references: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    rows1600 = scale._reciprocal_rows(cases, references, "adapter_step1600")
    rows3200 = scale._reciprocal_rows(cases, references, "adapter_step3200")
    count = min(len(rows1600), len(rows3200))
    valid = int(metrics["retrieval"]["raw_d64_ot"]["pooled_total"])
    precision1600 = scale._precision_at(rows1600, count)
    precision3200 = scale._precision_at(rows3200, count)
    return {
        "matched_query_count": count,
        "matched_coverage": count / valid,
        "adapter_step1600_precision": precision1600,
        "adapter_step3200_precision": precision3200,
        "step3200_minus_step1600_precision": precision3200 - precision1600,
    }


def _scaling(metrics: dict[str, Any]) -> dict[str, Any]:
    retrieval1600 = metrics["retrieval"]["adapter_step1600"]
    retrieval3200 = metrics["retrieval"]["adapter_step3200"]
    supply1600 = metrics["supply"]["adapter_step1600"]
    supply3200 = metrics["supply"]["adapter_step3200"]
    return {
        "retrieval": {
            key: retrieval3200[key] - retrieval1600[key]
            for key in (
                "pooled_r1",
                "pooled_r5",
                "pooled_r32",
                "right_r1",
                "right_r5",
                "down_r1",
                "down_r5",
            )
        },
        "raw_union_top32": {
            "pooled_coverage": (
                supply3200["pooled_union_coverage"]
                - supply1600["pooled_union_coverage"]
            ),
            "right_coverage": (
                supply3200["axes"]["right"]["union_coverage"]
                - supply1600["axes"]["right"]["union_coverage"]
            ),
            "down_coverage": (
                supply3200["axes"]["down"]["union_coverage"]
                - supply1600["axes"]["down"]["union_coverage"]
            ),
        },
        "matched_reciprocal": metrics["checkpoint_matched_reciprocal"],
    }


def _gate(metrics: dict[str, Any], scaling: dict[str, Any], *, terminal: bool) -> dict[str, Any]:
    raw = metrics["retrieval"]["raw_d64_ot"]
    candidate = metrics["retrieval"]["adapter_step3200"]
    precision = metrics["reciprocal"]["matched_vs_raw"]["adapter_step3200"]
    r1_gain = candidate["pooled_r1"] - raw["pooled_r1"]
    r5_gain = candidate["pooled_r5"] - raw["pooled_r5"]
    if terminal:
        passed = bool(
            r1_gain >= 0.0
            and r5_gain >= 0.0
            and precision["precision_gain"] >= 0.0
            and precision["matched_coverage"] >= 0.03
        )
    else:
        passed = bool(
            r1_gain >= 0.005
            and r5_gain >= 0.0
            and precision["precision_gain"] >= 0.002
            and precision["matched_coverage"] >= 0.03
            and scaling["retrieval"]["pooled_r5"] >= 0.0
            and scaling["raw_union_top32"]["pooled_coverage"] >= 0.0
        )
    return {
        "r1_gain": r1_gain,
        "r5_gain": r5_gain,
        "matched_reciprocal_precision_gain": precision["precision_gain"],
        "matched_reciprocal_coverage": precision["matched_coverage"],
        "matched_step1600_to_step3200_r5_gain": (
            None if terminal else scaling["retrieval"]["pooled_r5"]
        ),
        "matched_step1600_to_step3200_union_gain": (
            None if terminal else scaling["raw_union_top32"]["pooled_coverage"]
        ),
        "transfer_passed" if terminal else "terminal_open_gate_passed": passed,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    _configure_shared_runner()
    scale._require_signed_inputs(args)
    scale._verify_stream_prefix()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    protocol, fit_boards, local_records, terminal_records = pilot._load_protocol(args)
    device = pilot._device(args.device)
    checkpoints, history, runtime = scale._train(
        args,
        protocol,
        fit_boards,
        output_dir,
        device=device,
        checkpoint_callback=None,
    )
    socket = load_socket_checkpoint(args.socket_checkpoint, device=device)
    adapters = scale._load_adapters(checkpoints, device=device)
    local_cases, local_references = scale._freeze_panel(
        local_records,
        panel_name="local16",
        targets=args.targets,
        socket=socket,
        adapters=adapters,
        device=device,
        output_dir=output_dir,
    )
    local = pilot._score_panel(local_cases, local_references)
    local["checkpoint_matched_reciprocal"] = _checkpoint_comparison(
        local_cases, local_references, local
    )
    scaling = _scaling(local)
    local_gate = _gate(local, scaling, terminal=False)

    terminal: dict[str, Any] = {"status": "skipped_by_local_gate"}
    terminal_gate: dict[str, Any] | None = None
    if local_gate["terminal_open_gate_passed"]:
        terminal_cases, terminal_references = scale._freeze_panel(
            terminal_records,
            panel_name="terminal16",
            targets=args.targets,
            socket=socket,
            adapters={"adapter_step3200": adapters["adapter_step3200"]},
            device=device,
            output_dir=output_dir,
        )
        terminal = pilot._score_panel(terminal_cases, terminal_references)
        terminal["status"] = "complete"
        terminal_gate = _gate(terminal, {}, terminal=True)

    report = {
        "schema": "aiijc-fullres-retrieval-adapter-scale3200-report-v1",
        "status": (
            "terminal-transfer-passed-decoder-separately-eligible"
            if terminal_gate is not None and terminal_gate["transfer_passed"]
            else "local-gate-passed-terminal-transfer-failed"
            if terminal_gate is not None
            else "local-gate-failed-stop-no-terminal-no-decoder"
        ),
        "protocol": protocol,
        "contract": retrieval_adapter_contract(adapters["adapter_step3200"]),
        "configuration": {
            "config_sha256": CONFIG_SHA256,
            "device": str(device),
            "steps": 3200,
            "fixed_checkpoints": list(CHECKPOINT_STEPS),
            "training_seed": scale.TRAIN_SEED,
            "evaluation_seed": scale.EVAL_SEED,
            "learning_rate": scale.LEARNING_RATE,
            "weight_decay": scale.WEIGHT_DECAY,
            "scheduler_t_max": 3200,
        },
        "training": {
            "runtime": runtime,
            "history": history,
            "checkpoints": {
                str(step): _record(path) for step, path in checkpoints.items()
            },
        },
        "local16": local,
        "scaling_step3200_minus_step1600": scaling,
        "local_terminal_open_gate": local_gate,
        "terminal16": terminal,
        "terminal_transfer_gate": terminal_gate,
        "decoder": {
            "run": False,
            "eligible": bool(
                terminal_gate is not None and terminal_gate["transfer_passed"]
            ),
        },
        "legality": {
            "organizer_train_only": True,
            "raw_d64_evidence_immutable": True,
            "adapter_pixels_matcher_only": True,
            "competition_test_accessed": False,
            "submission_output_or_production_modified": False,
        },
        "artifacts": {
            "config": _record(CONFIG),
            "module": _record(
                PROJECT_ROOT / "src/aiijc_puzzle/fullres_retrieval_adapter.py"
            ),
            "runner": _record(Path(__file__)),
            "socket": _record(args.socket_checkpoint),
            "previous_scale_report": _record(PREVIOUS_REPORT),
        },
    }
    scale._write_json(output_dir / "report.json", report)
    return report


def main() -> None:
    args = parse_args()
    random.seed(scale.TRAIN_SEED)
    np.random.seed(scale.TRAIN_SEED)
    torch.manual_seed(scale.TRAIN_SEED)
    torch.use_deterministic_algorithms(
        True,
        warn_only=args.allow_nondeterministic_mps,
    )
    report = run(args)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

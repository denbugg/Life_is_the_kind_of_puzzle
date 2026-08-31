#!/usr/bin/env python3
"""Open decoder40 only after the preregistered v1.1 confirm gate passes."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from run_component_relation_confidence import (
    CleanTileCache,
    filename_digest,
    frozen_case_forward,
    load_preregistration,
    manifest_record_lookup,
    prepare_case,
)

from aiijc_puzzle.component_relation_confidence import (
    FEATURE_NAMES,
    LogisticConfidenceCalibrator,
    aggregate_confidence_observations,
    build_query_confidence_features,
    calibrated_component_edge_priorities,
    confidence_query_observations,
)
from aiijc_puzzle.component_relation_reranker import (
    ComponentRelationReranker,
    aggregate_relation_observations,
    relation_query_observations,
)
from aiijc_puzzle.layout_evaluation import evaluate_layout
from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.socket_decoder import SocketDecoderConfig, decode_socket_assignments
from aiijc_puzzle.socket_sorter_production import (
    choose_deterministic_device,
    load_socket_checkpoint,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREREGISTRATION = (
    PROJECT_ROOT / "configs/component_relation_confidence_preregistered_v1_1.json"
)
DEFAULT_CONFIRM_REPORT = (
    PROJECT_ROOT
    / "outputs/component-relation-confidence/v1_1-local32-confirm24/report.json"
)
DEFAULT_MANIFEST = PROJECT_ROOT / "data/interim/validation_manifest.json"
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
GRID = 24
EXPECTED_RELATION_PARAMETERS = 131_665
EXPECTED_CALIBRATOR_PARAMETERS = 68
BOOTSTRAP_SAMPLES = 100_000
BOOTSTRAP_SEED = 20290910


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confirm-report", type=Path, default=DEFAULT_CONFIRM_REPORT)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    return parser.parse_args()


def paired_source_bootstrap(values: Any) -> dict[str, Any]:
    """One case per source, so paired board bootstrap is source-cluster bootstrap."""

    difference = np.asarray(values, dtype=np.float64)
    if difference.ndim != 1 or len(difference) == 0 or not np.isfinite(difference).all():
        raise ValueError("paired bootstrap requires one finite non-empty vector")
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    chunks: list[np.ndarray] = []
    remaining = BOOTSTRAP_SAMPLES
    while remaining:
        size = min(remaining, 4096)
        indices = generator.integers(0, len(difference), size=(size, len(difference)))
        chunks.append(difference[indices].mean(axis=1))
        remaining -= size
    samples = np.concatenate(chunks)
    return {
        "source_count": len(difference),
        "case_count": len(difference),
        "mean_delta_per_board": float(difference.mean()),
        "source_cluster_bootstrap_ci95": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED,
    }


def evaluate_promotion_gate(
    exact_bootstrap: Mapping[str, Any],
    *,
    adjacency_delta: float,
    strict_permutation_all_boards: bool,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    mean_exact = float(exact_bootstrap["mean_delta_per_board"])
    lower = float(exact_bootstrap["source_cluster_bootstrap_ci95"][0])
    checks = {
        "mean_exact_tiles_gain": {
            "observed": mean_exact,
            "required": contract["minimum_mean_exact_tiles_gain_per_board"],
            "pass": mean_exact >= contract["minimum_mean_exact_tiles_gain_per_board"],
        },
        "exact_gain_source_cluster_ci95_lower_strictly_positive": {
            "observed": lower,
            "required_strictly_greater_than": contract[
                "minimum_source_cluster_bootstrap_95ci_lower_exact_gain"
            ],
            "pass": lower
            > contract["minimum_source_cluster_bootstrap_95ci_lower_exact_gain"],
        },
        "adjacency_non_regression": {
            "observed_delta": adjacency_delta,
            "minimum_delta": -float(contract["maximum_adjacency_regression_fraction"]),
            "pass": adjacency_delta
            >= -float(contract["maximum_adjacency_regression_fraction"]),
        },
        "strict_original_tile_permutation_all_boards": {
            "observed": strict_permutation_all_boards,
            "required": True,
            "pass": strict_permutation_all_boards,
        },
    }
    passed = all(bool(value["pass"]) for value in checks.values())
    return {
        "status": "pass-promotion-eligible" if passed else "stop-research-only",
        "pass": passed,
        "promotion_eligible": passed,
        "competition_test_authorized": False,
        "checks": checks,
    }


def _summary(rows: list[dict[str, Any]], arm: str) -> dict[str, float]:
    fields = (
        "correct_tile_count",
        "direct_placement",
        "translation_aligned_count",
        "translation_aligned_placement",
        "adjacency_correct",
        "adjacency",
    )
    return {
        field: float(np.mean([float(row[arm]["metrics"][field]) for row in rows]))
        for field in fields
    }


def main() -> None:
    args = parse_args()
    random.seed(20290910)
    np.random.seed(20290910)
    torch.manual_seed(20290910)
    device = choose_deterministic_device(args.device)
    print(
        json.dumps(
            {"event": "start", "pid": os.getpid(), "device": str(device), "stage": "decoder40"}
        ),
        flush=True,
    )
    prereg, prereg_hash = load_preregistration(args.preregistration)
    confirm_report_path = args.confirm_report.resolve()
    confirm_report = json.loads(confirm_report_path.read_text(encoding="utf-8"))
    if confirm_report["preregistration"]["sha256"] != prereg_hash:
        raise ValueError("confirm24 report used a different preregistration")
    if not (
        confirm_report["gate"]["pass"]
        and confirm_report["gate"]["decoder40_authorized"]
        and not confirm_report["decoder40_opened"]
    ):
        raise PermissionError("confirm24 did not authorize opening decoder40")
    calibrator_path = Path(confirm_report["artifacts"]["calibrator"])
    if sha256_file(calibrator_path) != confirm_report["artifacts"]["calibrator_sha256"]:
        raise ValueError("confirmed calibrator artifact hash mismatch")
    calibrator = LogisticConfidenceCalibrator.from_dict(
        json.loads(calibrator_path.read_text(encoding="utf-8"))
    )
    if calibrator.parameter_count != EXPECTED_CALIBRATOR_PARAMETERS or tuple(
        calibrator.feature_names
    ) != FEATURE_NAMES:
        raise ValueError("confirmed calibrator architecture changed")

    frozen = prereg["frozen_inputs"]
    relation_path = PROJECT_ROOT / frozen["relation_checkpoint"]
    if sha256_file(relation_path) != frozen["relation_checkpoint_sha256"]:
        raise ValueError("frozen v1 relation checkpoint hash mismatch")
    relation_payload = torch.load(relation_path, map_location="cpu", weights_only=True)
    relation_contract = relation_payload["contract"]
    head = ComponentRelationReranker(
        int(relation_contract["tile_dimension"]),
        grid=int(relation_contract["grid"]),
        hidden_dimension=int(relation_contract["hidden_dimension"]),
    ).to(device)
    head.load_state_dict(relation_payload["state_dict"], strict=True)
    head.eval()
    if sum(parameter.numel() for parameter in head.parameters()) != EXPECTED_RELATION_PARAMETERS:
        raise ValueError("frozen relation parameter count changed")
    socket = load_socket_checkpoint(PROJECT_ROOT / frozen["socket_checkpoint"], device=device)
    if socket.sha256 != relation_payload["socket_checkpoint"]["sha256"]:
        raise ValueError("Socket lineage differs from frozen v1")

    decoder_contract = prereg["decoder40"]
    decoder_names = list(prereg["reserved64_split"]["decoder40"]["filenames"])
    if filename_digest(decoder_names) != prereg["reserved64_split"]["decoder40"]["digest"]:
        raise ValueError("decoder40 roster digest mismatch")
    if decoder_names != confirm_report["selection"]["decoder_reserved_filenames"]:
        raise ValueError("confirm report reserved a different decoder40 roster")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    lookup = manifest_record_lookup(manifest)
    if set(decoder_names) - set(lookup):
        raise ValueError("decoder40 contains non-train or missing manifest sources")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    if report_path.exists():
        raise FileExistsError("refusing to overwrite an existing decoder40 report")
    cache = CleanTileCache(args.targets)
    decoder_config = SocketDecoderConfig(
        component_edge_budget_per_axis=144,
        max_swap_steps=24,
    )
    board_rows: list[dict[str, Any]] = []
    relation_observations: list[dict[str, Any]] = []
    confidence_observations: list[dict[str, Any]] = []
    exact_delta: list[float] = []
    adjacency_delta: list[float] = []
    strict_all = True
    started = perf_counter()
    for index, filename in enumerate(decoder_names):
        case = prepare_case(
            cache,
            lookup[filename],
            draw_index=int(decoder_contract["draw_index"]),
            seed=int(decoder_contract["synthetic_seed"]),
        )
        output = frozen_case_forward(case, socket=socket, head=head, device=device)
        rows = build_query_confidence_features(
            output.logits,
            output.candidates,
            output.components,
            board_id=case.case_id,
            grid=GRID,
        )
        probabilities = calibrator.predict_probabilities([row.values for row in rows])
        priority, priority_diagnostics = calibrated_component_edge_priorities(
            output.socket_output.right_log_assignment,
            output.socket_output.down_log_assignment,
            rows,
            probabilities,
            output.candidates,
            grid=GRID,
            top_cap=32,
            bonus_scale=0.25,
        )
        baseline = decode_socket_assignments(
            output.socket_output.right_log_assignment,
            output.socket_output.down_log_assignment,
            grid=GRID,
            config=decoder_config,
        )
        treatment = decode_socket_assignments(
            output.socket_output.right_log_assignment,
            output.socket_output.down_log_assignment,
            grid=GRID,
            config=decoder_config,
            component_edge_priority=priority,
        )
        reference = np.argsort(case.input_tile_to_position)
        baseline_metrics = evaluate_layout(
            baseline.layout,
            reference,
            reference_is_exact=True,
        )
        treatment_metrics = evaluate_layout(
            treatment.layout,
            reference,
            reference_is_exact=True,
        )
        baseline_strict = bool(
            np.array_equal(np.sort(baseline.layout), np.arange(GRID * GRID))
        )
        treatment_strict = bool(
            np.array_equal(np.sort(treatment.layout), np.arange(GRID * GRID))
        )
        strict_all &= baseline_strict and treatment_strict
        exact_delta.append(
            treatment_metrics.correct_tile_count - baseline_metrics.correct_tile_count
        )
        adjacency_delta.append(treatment_metrics.adjacency - baseline_metrics.adjacency)
        relation_observations.extend(
            relation_query_observations(
                output.logits,
                output.candidates,
                output.labels,
                output.oracle_relations,
                output.profiles,
                board_id=case.case_id,
            )
        )
        confidence_observations.extend(
            confidence_query_observations(
                rows,
                calibrator,
                output.candidates,
                output.labels,
                output.oracle_relations,
                output.profiles,
            )
        )
        board_rows.append(
            {
                "case_id": case.case_id,
                "source_filename": case.source_filename,
                "baseline": {
                    "metrics": baseline_metrics.as_dict(),
                    "decoder": baseline.report(),
                    "strict_original_tile_permutation": baseline_strict,
                },
                "calibrated_relation_priority": {
                    "metrics": treatment_metrics.as_dict(),
                    "decoder": treatment.report(),
                    "strict_original_tile_permutation": treatment_strict,
                    "priority": priority_diagnostics,
                },
            }
        )
        print(
            json.dumps(
                {
                    "event": "decoder40",
                    "done": index + 1,
                    "total": len(decoder_names),
                    "exact_delta": exact_delta[-1],
                    "elapsed_seconds": perf_counter() - started,
                }
            ),
            flush=True,
        )

    baseline_summary = _summary(board_rows, "baseline")
    treatment_summary = _summary(board_rows, "calibrated_relation_priority")
    bootstrap = paired_source_bootstrap(exact_delta)
    mean_adjacency_delta = float(np.mean(adjacency_delta))
    gate = evaluate_promotion_gate(
        bootstrap,
        adjacency_delta=mean_adjacency_delta,
        strict_permutation_all_boards=strict_all,
        contract=decoder_contract["promotion_gate"],
    )
    report = {
        "experiment": "d64-component-relation-confidence-v1.1-decoder40",
        "status": gate["status"],
        "competition_test_opened": False,
        "strict_original_tiles_only": True,
        "preregistration": {
            "path": str(args.preregistration.resolve()),
            "sha256": prereg_hash,
        },
        "confirm24": {
            "report": str(confirm_report_path),
            "report_sha256": sha256_file(confirm_report_path),
            "gate": confirm_report["gate"],
            "calibrator_sha256": confirm_report["artifacts"]["calibrator_sha256"],
        },
        "selection": {
            "decoder_filenames": decoder_names,
            "decoder_digest": filename_digest(decoder_names),
        },
        "decoder_contract": {
            "baseline": decoder_contract["baseline"],
            "treatment": decoder_contract["treatment"],
            "component_edge_budget_per_axis": 144,
            "max_swap_steps": 24,
            "top_calibrated_queries": 32,
            "bonus_scale": 0.25,
        },
        "summary": {
            "baseline": baseline_summary,
            "calibrated_relation_priority": treatment_summary,
            "delta": {
                key: treatment_summary[key] - baseline_summary[key]
                for key in baseline_summary
            },
            "exact_source_cluster_bootstrap": bootstrap,
            "mean_adjacency_delta": mean_adjacency_delta,
        },
        "decoder40_relation_metrics": aggregate_relation_observations(
            relation_observations,
            high_confidence_caps=(32, 144),
        ),
        "decoder40_confidence_metrics": aggregate_confidence_observations(
            confidence_observations,
            caps=(32, 144),
        ),
        "promotion_gate": gate,
        "runtime_seconds": perf_counter() - started,
        "boards": board_rows,
        "artifacts": {"report": str(report_path)},
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "event": "complete",
                "status": gate["status"],
                "promotion_eligible": gate["promotion_eligible"],
                "report": str(report_path),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

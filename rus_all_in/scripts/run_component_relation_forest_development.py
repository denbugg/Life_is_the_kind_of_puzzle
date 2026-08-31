#!/usr/bin/env python3
"""Development-only relation-forest score substitution on opened decoder40."""

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
    build_query_confidence_features,
    calibrated_component_edge_priorities,
    relation_forest_score_substitution,
)
from aiijc_puzzle.component_relation_reranker import ComponentRelationReranker
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
DEFAULT_V1_1_REPORT = (
    PROJECT_ROOT / "outputs/component-relation-confidence/v1_1-decoder40/report.json"
)
DEFAULT_V1_2_REPORT = (
    PROJECT_ROOT
    / "outputs/component-relation-confidence/v1_2-relation-forest-development/report.json"
)
DEFAULT_MANIFEST = PROJECT_ROOT / "data/interim/validation_manifest.json"
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
GRID = 24
CAPS = (16, 32, 64)
ARM_NAMES = tuple(f"relation_forest_rank_substitution_top{cap}" for cap in CAPS)
HYBRID_ARM = "relation_forest_top16_plus_calibrated_hard_order_top32"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--v1-1-report", type=Path, default=DEFAULT_V1_1_REPORT)
    parser.add_argument("--v1-2-report", type=Path, default=DEFAULT_V1_2_REPORT)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument(
        "--hybrid-only",
        action="store_true",
        help="development follow-up selected after the first opened-panel v1.2 report",
    )
    return parser.parse_args()


def _arm_summary(rows: list[dict[str, Any]], arm: str) -> dict[str, float]:
    metric_fields = (
        "correct_tile_count",
        "direct_placement",
        "translation_aligned_count",
        "translation_aligned_placement",
        "adjacency_correct",
        "adjacency",
    )
    diagnostic_fields = ("component_count", "largest_component")
    result = {
        field: float(np.mean([float(row[arm]["metrics"][field]) for row in rows]))
        for field in metric_fields
    }
    result.update(
        {
            field: float(
                np.mean(
                    [float(row[arm]["decoder"]["diagnostics"][field]) for row in rows]
                )
            )
            for field in diagnostic_fields
        }
    )
    return result


def _paired_development_delta(
    rows: list[dict[str, Any]],
    arm: str,
    *,
    baseline: str = "raw_decoder144",
) -> dict[str, Any]:
    fields = (
        "correct_tile_count",
        "translation_aligned_count",
        "adjacency",
    )
    result: dict[str, Any] = {}
    for field in fields:
        difference = np.asarray(
            [
                float(row[arm]["metrics"][field])
                - float(row[baseline]["metrics"][field])
                for row in rows
            ],
            dtype=np.float64,
        )
        result[field] = {
            "mean": float(difference.mean()),
            "wins_ties_losses": [
                int(np.sum(difference > 0)),
                int(np.sum(difference == 0)),
                int(np.sum(difference < 0)),
            ],
        }
    exact = result["correct_tile_count"]["mean"]
    adjacency = result["adjacency"]["mean"]
    aligned = result["translation_aligned_count"]["mean"]
    broad_regression = exact < 0 and adjacency < -0.002 and aligned < -0.5
    return {
        "metrics": result,
        "development_classification": (
            "descriptive-positive"
            if exact > 0 and adjacency >= 0
            else "broad-negative"
            if broad_regression
            else "mixed"
        ),
        "broad_regression": broad_regression,
        "not_confirmation": True,
    }


def _copy_prior_arm(prior: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "metrics": dict(prior["metrics"]),
        "decoder": dict(prior["decoder"]),
        "strict_original_tile_permutation": bool(
            prior["strict_original_tile_permutation"]
        ),
        "source": "frozen v1.1 decoder40 report; not recomputed",
    }


def main() -> None:
    args = parse_args()
    random.seed(20290911)
    np.random.seed(20290911)
    torch.manual_seed(20290911)
    device = choose_deterministic_device(args.device)
    print(
        json.dumps(
            {
                "event": "start",
                "pid": os.getpid(),
                "device": str(device),
                "stage": (
                    "opened-decoder40-development-v1.2-hybrid"
                    if args.hybrid_only
                    else "opened-decoder40-development-v1.2"
                ),
            }
        ),
        flush=True,
    )
    prereg, prereg_hash = load_preregistration(args.preregistration)
    prior_path = args.v1_1_report.resolve()
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    if prior["preregistration"]["sha256"] != prereg_hash:
        raise ValueError("v1.1 decoder report used a different preregistration")
    if prior["status"] != "stop-research-only" or len(prior["boards"]) != 40:
        raise ValueError("v1.1 decoder40 must be complete before development reuse")
    development_parent: dict[str, Any] | None = None
    if args.hybrid_only:
        development_parent = json.loads(args.v1_2_report.read_text(encoding="utf-8"))
        if (
            development_parent.get("development_selected_arm")
            != "relation_forest_rank_substitution_top16"
            or development_parent.get("status") != "development-only-no-promotion"
        ):
            raise ValueError("hybrid requires the completed top16-selected v1.2 report")
    decoder_names = list(prior["selection"]["decoder_filenames"])
    if filename_digest(decoder_names) != prior["selection"]["decoder_digest"]:
        raise ValueError("opened decoder40 roster digest mismatch")

    calibrator_path = Path(prior["confirm24"]["report"]).parent / (
        "component_relation_confidence.json"
    )
    if sha256_file(calibrator_path) != prior["confirm24"]["calibrator_sha256"]:
        raise ValueError("v1.1 calibrator hash mismatch")
    calibrator = LogisticConfidenceCalibrator.from_dict(
        json.loads(calibrator_path.read_text(encoding="utf-8"))
    )
    if calibrator.parameter_count != 68 or tuple(calibrator.feature_names) != FEATURE_NAMES:
        raise ValueError("v1.1 calibrator contract changed")

    frozen = prereg["frozen_inputs"]
    relation_path = PROJECT_ROOT / frozen["relation_checkpoint"]
    if sha256_file(relation_path) != frozen["relation_checkpoint_sha256"]:
        raise ValueError("frozen relation checkpoint hash mismatch")
    relation_payload = torch.load(relation_path, map_location="cpu", weights_only=True)
    relation_contract = relation_payload["contract"]
    head = ComponentRelationReranker(
        int(relation_contract["tile_dimension"]),
        grid=int(relation_contract["grid"]),
        hidden_dimension=int(relation_contract["hidden_dimension"]),
    ).to(device)
    head.load_state_dict(relation_payload["state_dict"], strict=True)
    head.eval()
    socket = load_socket_checkpoint(PROJECT_ROOT / frozen["socket_checkpoint"], device=device)
    if socket.sha256 != relation_payload["socket_checkpoint"]["sha256"]:
        raise ValueError("Socket lineage differs from frozen relation v1")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    lookup = manifest_record_lookup(manifest)
    if set(decoder_names) - set(lookup):
        raise ValueError("opened decoder40 is not a subset of manifest train")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    if report_path.exists():
        raise FileExistsError("refusing to overwrite v1.2 development report")

    decoder_contract = prereg["decoder40"]
    decoder_config = SocketDecoderConfig(
        component_edge_budget_per_axis=144,
        max_swap_steps=24,
    )
    cache = CleanTileCache(args.targets)
    prior_by_source = {row["source_filename"]: row for row in prior["boards"]}
    board_rows: list[dict[str, Any]] = []
    development_arms = (HYBRID_ARM,) if args.hybrid_only else ARM_NAMES
    started = perf_counter()
    for index, filename in enumerate(decoder_names):
        case = prepare_case(
            cache,
            lookup[filename],
            draw_index=int(decoder_contract["draw_index"]),
            seed=int(decoder_contract["synthetic_seed"]),
        )
        prior_row = prior_by_source[filename]
        if case.case_id != prior_row["case_id"]:
            raise ValueError("development case does not exactly replay v1.1 decoder40")
        output = frozen_case_forward(case, socket=socket, head=head, device=device)
        confidence_rows = build_query_confidence_features(
            output.logits,
            output.candidates,
            output.components,
            board_id=case.case_id,
            grid=GRID,
        )
        probabilities = calibrator.predict_probabilities(
            [row.values for row in confidence_rows]
        )
        reference = np.argsort(case.input_tile_to_position)
        board: dict[str, Any] = {
            "case_id": case.case_id,
            "source_filename": filename,
            "raw_decoder144": _copy_prior_arm(prior_row["baseline"]),
            "v1_1_existing_hard_edge_priority": _copy_prior_arm(
                prior_row["calibrated_relation_priority"]
            ),
        }
        cap_arms = ((16, HYBRID_ARM),) if args.hybrid_only else tuple(
            zip(CAPS, ARM_NAMES, strict=True)
        )
        for cap, arm in cap_arms:
            assignments, forest = relation_forest_score_substitution(
                output.socket_output.right_log_assignment,
                output.socket_output.down_log_assignment,
                confidence_rows,
                probabilities,
                output.candidates,
                grid=GRID,
                top_cap=cap,
                component_edge_budget_per_axis=144,
            )
            hard_priority = None
            hard_priority_diagnostics = None
            if args.hybrid_only:
                hard_priority, hard_priority_diagnostics = (
                    calibrated_component_edge_priorities(
                        assignments["right"],
                        assignments["down"],
                        confidence_rows,
                        probabilities,
                        output.candidates,
                        grid=GRID,
                        top_cap=32,
                        bonus_scale=0.25,
                    )
                )
            decoded = decode_socket_assignments(
                assignments["right"],
                assignments["down"],
                grid=GRID,
                config=decoder_config,
                component_edge_priority=hard_priority,
            )
            metrics = evaluate_layout(
                decoded.layout,
                reference,
                reference_is_exact=True,
            )
            strict = bool(
                np.array_equal(np.sort(decoded.layout), np.arange(GRID * GRID))
            )
            if not strict:
                raise RuntimeError("relation forest decoder violated strict permutation")
            board[arm] = {
                "metrics": metrics.as_dict(),
                "decoder": decoded.report(),
                "strict_original_tile_permutation": strict,
                "relation_forest": forest,
                "calibrated_hard_priority": hard_priority_diagnostics,
            }
        board_rows.append(board)
        print(
            json.dumps(
                {
                    "event": "development-board",
                    "done": index + 1,
                    "total": len(decoder_names),
                    "exact_deltas": {
                        arm: board[arm]["metrics"]["correct_tile_count"]
                        - board["raw_decoder144"]["metrics"]["correct_tile_count"]
                        for arm in development_arms
                    },
                    "elapsed_seconds": perf_counter() - started,
                }
            ),
            flush=True,
        )

    all_arms = (
        "raw_decoder144",
        "v1_1_existing_hard_edge_priority",
        *development_arms,
    )
    summaries = {arm: _arm_summary(board_rows, arm) for arm in all_arms}
    comparisons = {
        arm: _paired_development_delta(board_rows, arm) for arm in all_arms[1:]
    }
    best = max(
        development_arms,
        key=lambda arm: (
            comparisons[arm]["metrics"]["correct_tile_count"]["mean"],
            comparisons[arm]["metrics"]["adjacency"]["mean"],
            comparisons[arm]["metrics"]["translation_aligned_count"]["mean"],
        ),
    )
    report = {
        "experiment": (
            "component-relation-forest-hybrid-development-v1.2h"
            if args.hybrid_only
            else "component-relation-forest-development-v1.2"
        ),
        "status": "development-only-no-promotion",
        "panel": {
            "kind": "already-opened decoder40 reuse",
            "fresh_confirmation": False,
            "model_selection_exposed": True,
            "source_report": str(prior_path),
            "source_report_sha256": sha256_file(prior_path),
            "development_parent_report": (
                str(args.v1_2_report.resolve()) if args.hybrid_only else None
            ),
            "development_parent_report_sha256": (
                sha256_file(args.v1_2_report) if args.hybrid_only else None
            ),
        },
        "competition_test_opened": False,
        "promotion_authorized": False,
        "strict_original_tiles_only": True,
        "selection": {
            "decoder_filenames": decoder_names,
            "decoder_digest": filename_digest(decoder_names),
        },
        "arms": {
            "raw_decoder144": "frozen report replay",
            "v1_1_existing_hard_edge_priority": "frozen report replay",
            "relation_forest": (
                "calibrated top16/32/64 atomic relations; per-axis out/in capacity; "
                "baseline-component coordinate cycle/collision/span checks; accepted "
                "new contacts promoted to row/column-best score before normal decoder144"
            ),
            "hybrid_followup": (
                "top16 relation-forest score substitution selected on the opened v1.2 "
                "panel, followed by v1.1 top32 calibrated ordering on the resulting "
                "hard matching"
                if args.hybrid_only
                else None
            ),
        },
        "summaries": summaries,
        "paired_development_comparisons_vs_raw": comparisons,
        "development_selected_arm": best,
        "selection_caveat": (
            "Selected descriptively on this already-opened panel; cannot support CI, "
            "promotion or competition-test access."
        ),
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
                "selected_arm": best,
                "classification": comparisons[best]["development_classification"],
                "report": str(report_path),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compose frozen cyclic-border5 with relation decoders on opened decoder40."""

from __future__ import annotations

import argparse
import json
import os
import random
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
from aiijc_puzzle.socket_translation_placer import (
    CyclicTranslationConfig,
    select_global_cyclic_translation,
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
DEFAULT_HYBRID_REPORT = (
    PROJECT_ROOT
    / "outputs/component-relation-confidence/v1_2h-relation-forest-hybrid-development/report.json"
)
DEFAULT_MANIFEST = PROJECT_ROOT / "data/interim/validation_manifest.json"
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
GRID = 24
BASE_ARMS = (
    "raw_decoder144",
    "v1_1_existing_hard_edge_priority",
    "v1_2_relation_forest_top16",
    "v1_2h_relation_forest_top16_plus_hard_order",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--v1-1-report", type=Path, default=DEFAULT_V1_1_REPORT)
    parser.add_argument("--v1-2-report", type=Path, default=DEFAULT_V1_2_REPORT)
    parser.add_argument("--hybrid-report", type=Path, default=DEFAULT_HYBRID_REPORT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    return parser.parse_args()


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


def _paired(rows: list[dict[str, Any]], arm: str, baseline: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in ("correct_tile_count", "translation_aligned_count", "adjacency"):
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
    return {
        "metrics": result,
        "descriptive_positive": exact > 0 and adjacency >= 0,
        "not_confirmation": True,
    }


def _arm_payload(decoded: Any, reference: np.ndarray) -> dict[str, Any]:
    metrics = evaluate_layout(decoded.layout, reference, reference_is_exact=True)
    return {
        "metrics": metrics.as_dict(),
        "decoder": decoded.report(),
        "strict_original_tile_permutation": bool(
            np.array_equal(np.sort(decoded.layout), np.arange(GRID * GRID))
        ),
    }


def main() -> None:
    args = parse_args()
    random.seed(20290912)
    np.random.seed(20290912)
    torch.manual_seed(20290912)
    device = choose_deterministic_device(args.device)
    print(
        json.dumps(
            {
                "event": "start",
                "pid": os.getpid(),
                "device": str(device),
                "stage": "opened-decoder40-relation-plus-cyclic-border5",
            }
        ),
        flush=True,
    )
    prereg, prereg_hash = load_preregistration(args.preregistration)
    prior = json.loads(args.v1_1_report.read_text(encoding="utf-8"))
    v1_2 = json.loads(args.v1_2_report.read_text(encoding="utf-8"))
    hybrid = json.loads(args.hybrid_report.read_text(encoding="utf-8"))
    if prior["preregistration"]["sha256"] != prereg_hash:
        raise ValueError("v1.1 report used a different preregistration")
    if (
        v1_2["development_selected_arm"]
        != "relation_forest_rank_substitution_top16"
        or hybrid["development_selected_arm"]
        != "relation_forest_top16_plus_calibrated_hard_order_top32"
    ):
        raise ValueError("cyclic composition requires completed v1.2 development")
    decoder_names = list(prior["selection"]["decoder_filenames"])
    if not all(
        report["selection"]["decoder_filenames"] == decoder_names
        for report in (v1_2, hybrid)
    ):
        raise ValueError("relation development reports use different panels")
    if filename_digest(decoder_names) != prior["selection"]["decoder_digest"]:
        raise ValueError("opened decoder40 digest mismatch")

    calibrator_path = Path(prior["confirm24"]["report"]).parent / (
        "component_relation_confidence.json"
    )
    if sha256_file(calibrator_path) != prior["confirm24"]["calibrator_sha256"]:
        raise ValueError("confirmed calibrator hash mismatch")
    calibrator = LogisticConfidenceCalibrator.from_dict(
        json.loads(calibrator_path.read_text(encoding="utf-8"))
    )
    if calibrator.parameter_count != 68 or tuple(calibrator.feature_names) != FEATURE_NAMES:
        raise ValueError("calibrator contract changed")

    frozen = prereg["frozen_inputs"]
    relation_path = PROJECT_ROOT / frozen["relation_checkpoint"]
    if sha256_file(relation_path) != frozen["relation_checkpoint_sha256"]:
        raise ValueError("frozen relation checkpoint hash mismatch")
    relation_payload = torch.load(relation_path, map_location="cpu", weights_only=True)
    contract = relation_payload["contract"]
    head = ComponentRelationReranker(
        int(contract["tile_dimension"]),
        grid=int(contract["grid"]),
        hidden_dimension=int(contract["hidden_dimension"]),
    ).to(device)
    head.load_state_dict(relation_payload["state_dict"], strict=True)
    head.eval()
    socket = load_socket_checkpoint(PROJECT_ROOT / frozen["socket_checkpoint"], device=device)
    if socket.sha256 != relation_payload["socket_checkpoint"]["sha256"]:
        raise ValueError("Socket lineage differs from relation v1")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    lookup = manifest_record_lookup(manifest)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    if report_path.exists():
        raise FileExistsError("refusing to overwrite cyclic development report")
    decoder_config = SocketDecoderConfig(
        component_edge_budget_per_axis=144,
        max_swap_steps=24,
    )
    cyclic_config = CyclicTranslationConfig(border_weight=5.0)
    decoder_contract = prereg["decoder40"]
    cache = CleanTileCache(args.targets)
    prior_by_source = {row["source_filename"]: row for row in prior["boards"]}
    v1_2_by_source = {row["source_filename"]: row for row in v1_2["boards"]}
    hybrid_by_source = {row["source_filename"]: row for row in hybrid["boards"]}
    boards: list[dict[str, Any]] = []
    started = perf_counter()
    for index, filename in enumerate(decoder_names):
        case = prepare_case(
            cache,
            lookup[filename],
            draw_index=int(decoder_contract["draw_index"]),
            seed=int(decoder_contract["synthetic_seed"]),
        )
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
        original_right = output.socket_output.right_log_assignment
        original_down = output.socket_output.down_log_assignment
        raw = decode_socket_assignments(
            original_right,
            original_down,
            grid=GRID,
            config=decoder_config,
        )
        v1_priority, _ = calibrated_component_edge_priorities(
            original_right,
            original_down,
            confidence_rows,
            probabilities,
            output.candidates,
            grid=GRID,
            top_cap=32,
            bonus_scale=0.25,
        )
        v1_decoded = decode_socket_assignments(
            original_right,
            original_down,
            grid=GRID,
            config=decoder_config,
            component_edge_priority=v1_priority,
        )
        substituted, _ = relation_forest_score_substitution(
            original_right,
            original_down,
            confidence_rows,
            probabilities,
            output.candidates,
            grid=GRID,
            top_cap=16,
            component_edge_budget_per_axis=144,
        )
        forest_decoded = decode_socket_assignments(
            substituted["right"],
            substituted["down"],
            grid=GRID,
            config=decoder_config,
        )
        hybrid_priority, _ = calibrated_component_edge_priorities(
            substituted["right"],
            substituted["down"],
            confidence_rows,
            probabilities,
            output.candidates,
            grid=GRID,
            top_cap=32,
            bonus_scale=0.25,
        )
        hybrid_decoded = decode_socket_assignments(
            substituted["right"],
            substituted["down"],
            grid=GRID,
            config=decoder_config,
            component_edge_priority=hybrid_priority,
        )
        base_decoders = dict(
            zip(
                BASE_ARMS,
                (raw, v1_decoded, forest_decoded, hybrid_decoded),
                strict=True,
            )
        )
        reference = np.argsort(case.input_tile_to_position)
        board: dict[str, Any] = {"case_id": case.case_id, "source_filename": filename}
        expected_hashes = {
            "raw_decoder144": prior_by_source[filename]["baseline"]["decoder"][
                "layout_sha256"
            ],
            "v1_1_existing_hard_edge_priority": prior_by_source[filename][
                "calibrated_relation_priority"
            ]["decoder"]["layout_sha256"],
            "v1_2_relation_forest_top16": v1_2_by_source[filename][
                "relation_forest_rank_substitution_top16"
            ]["decoder"]["layout_sha256"],
            "v1_2h_relation_forest_top16_plus_hard_order": hybrid_by_source[filename][
                "relation_forest_top16_plus_calibrated_hard_order_top32"
            ]["decoder"]["layout_sha256"],
        }
        for arm, decoded in base_decoders.items():
            base_payload = _arm_payload(decoded, reference)
            if base_payload["decoder"]["layout_sha256"] != expected_hashes[arm]:
                raise RuntimeError(f"{arm} did not replay its frozen development layout")
            board[arm] = base_payload
            cyclic = select_global_cyclic_translation(
                decoded.layout,
                original_right,
                original_down,
                grid=GRID,
                config=cyclic_config,
            )
            cyclic_metrics = evaluate_layout(
                cyclic.layout,
                reference,
                reference_is_exact=True,
            )
            board[f"{arm}_cyclic_border5"] = {
                "metrics": cyclic_metrics.as_dict(),
                "cyclic": cyclic.report(),
                "strict_original_tile_permutation": bool(
                    np.array_equal(np.sort(cyclic.layout), np.arange(GRID * GRID))
                ),
            }
        if not all(
            bool(value["strict_original_tile_permutation"])
            for name, value in board.items()
            if name not in {"case_id", "source_filename"}
        ):
            raise RuntimeError("a cyclic composition violated strict permutation")
        boards.append(board)
        print(
            json.dumps(
                {
                    "event": "cyclic-development",
                    "done": index + 1,
                    "total": len(decoder_names),
                    "cyclic_exact": {
                        arm: board[f"{arm}_cyclic_border5"]["metrics"][
                            "correct_tile_count"
                        ]
                        for arm in BASE_ARMS
                    },
                    "elapsed_seconds": perf_counter() - started,
                }
            ),
            flush=True,
        )

    cyclic_arms = tuple(f"{arm}_cyclic_border5" for arm in BASE_ARMS)
    all_arms = (*BASE_ARMS, *cyclic_arms)
    summaries = {arm: _summary(boards, arm) for arm in all_arms}
    cyclic_baseline = "raw_decoder144_cyclic_border5"
    paired = {
        arm: _paired(boards, arm, cyclic_baseline)
        for arm in cyclic_arms
        if arm != cyclic_baseline
    }
    best = max(
        paired,
        key=lambda arm: (
            paired[arm]["metrics"]["correct_tile_count"]["mean"],
            paired[arm]["metrics"]["adjacency"]["mean"],
            paired[arm]["metrics"]["translation_aligned_count"]["mean"],
        ),
    )
    report = {
        "experiment": "component-relation-cyclic-border5-development-v1.3",
        "status": "development-only-no-promotion",
        "panel": {
            "kind": "already-opened decoder40 reuse",
            "fresh_confirmation": False,
            "model_selection_exposed": True,
            "source_reports": {
                "v1_1": {
                    "path": str(args.v1_1_report.resolve()),
                    "sha256": sha256_file(args.v1_1_report),
                },
                "v1_2": {
                    "path": str(args.v1_2_report.resolve()),
                    "sha256": sha256_file(args.v1_2_report),
                },
                "hybrid": {
                    "path": str(args.hybrid_report.resolve()),
                    "sha256": sha256_file(args.hybrid_report),
                },
            },
        },
        "competition_test_opened": False,
        "promotion_authorized": False,
        "strict_original_tiles_only": True,
        "cyclic_contract": {
            "primitive": "already-confirmed socket-global-cyclic-translation-v1",
            "border_weight": 5.0,
            "same_original_socket_assignments_for_every_arm": True,
            "targets_or_pixels_used_for_origin": False,
        },
        "selection": {
            "decoder_filenames": decoder_names,
            "decoder_digest": filename_digest(decoder_names),
        },
        "summaries": summaries,
        "paired_cyclic_relation_arms_vs_raw_cyclic": paired,
        "development_selected_arm": best,
        "fresh_gate_candidate": bool(paired[best]["descriptive_positive"]),
        "selection_caveat": (
            "The arm is selected descriptively on an already-opened panel; it is only "
            "a candidate for a separately preregistered fresh gate."
        ),
        "runtime_seconds": perf_counter() - started,
        "boards": boards,
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
                "fresh_gate_candidate": report["fresh_gate_candidate"],
                "report": str(report_path),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

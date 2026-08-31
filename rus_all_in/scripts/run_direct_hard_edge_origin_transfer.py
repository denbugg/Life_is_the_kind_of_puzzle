#!/usr/bin/env python3
"""Same-opened-D1 target-free baseline-origin transfer diagnostic.

For the learned hard-edge layout before cyclic placement, enumerate only the
576 global rolls and choose the one with maximum tilewise overlap with the
frozen raw decoder144+cyclic5 prediction.  The raw prediction is inference
visible; exact references are not used until all layouts are persisted.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from run_direct_hard_edge_priority import (
    GRID,
    TILE_COUNT,
    CleanTileCache,
    _dirty_sha256,
    _forward_board,
    _make_case,
    _record_lookup,
    _resolve,
    learned_priority_matrices,
    load_frozen_config,
)

from aiijc_puzzle.direct_hard_edge_priority import (
    DirectHardEdgePriority,
    transfer_cyclic_origin_by_baseline_overlap,
)
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
from aiijc_puzzle.synthetic_socket_evaluation import names_digest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/direct_hard_edge_board_priority_preregistered_v1.json"
DEFAULT_PRIOR_REPORT = (
    PROJECT_ROOT
    / "outputs/direct-hard-edge-priority/v1-fit256-s600-d1-32-cpu/report.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs/direct-hard-edge-priority/v1-fit256-s600-d1-32-cpu/origin-transfer"
)
DEFAULT_MANIFEST = PROJECT_ROOT / "data/interim/validation_manifest.json"
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--prior-report", type=Path, default=DEFAULT_PRIOR_REPORT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    return parser.parse_args()


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric in ("exact_tiles", "adjacency", "translation_aligned_tiles"):
        arms: dict[str, np.ndarray] = {
            arm: np.asarray([row[f"{arm}_{metric}"] for row in rows], dtype=np.float64)
            for arm in ("baseline", "independent", "transferred")
        }
        result[metric] = {
            **{f"{arm}_mean": float(value.mean()) for arm, value in arms.items()},
            "independent_delta_vs_baseline": float(
                (arms["independent"] - arms["baseline"]).mean()
            ),
            "transferred_delta_vs_baseline": float(
                (arms["transferred"] - arms["baseline"]).mean()
            ),
            "transferred_delta_vs_independent": float(
                (arms["transferred"] - arms["independent"]).mean()
            ),
        }
    overlap = np.asarray([row["baseline_overlap_count"] for row in rows], dtype=np.float64)
    result["target_free_baseline_overlap"] = {
        "mean_tiles": float(overlap.mean()),
        "minimum_tiles": int(overlap.min()),
        "maximum_tiles": int(overlap.max()),
    }
    return result


def _assert_reproduces_prior(
    aggregate: Mapping[str, Any],
    prior_report: Mapping[str, Any],
) -> None:
    prior = prior_report["d1"]["same_panel_decoder144_cyclic5"]["aggregate"]
    for metric in ("exact_tiles", "adjacency", "translation_aligned_tiles"):
        for arm, previous_arm in (("baseline", "raw"), ("independent", "learned")):
            observed = float(aggregate[metric][f"{arm}_mean"])
            expected = float(prior[metric][f"{previous_arm}_mean"])
            if not np.isclose(observed, expected, rtol=0.0, atol=1e-12):
                raise RuntimeError(
                    f"deterministic regeneration changed {metric}/{arm}: "
                    f"{observed} != {expected}"
                )


def main() -> None:
    args = parse_args()
    config, config_sha = load_frozen_config(args.config)
    prior_report = json.loads(args.prior_report.read_text(encoding="utf-8"))
    prior_sha = sha256_file(args.prior_report)
    if prior_report.get("config_sha256") != config_sha:
        raise ValueError("prior report and preregistered config differ")
    checkpoint_path = Path(prior_report["checkpoint"]["path"])
    if sha256_file(checkpoint_path) != prior_report["checkpoint"]["sha256"]:
        raise ValueError("direct hard-edge checkpoint SHA mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("config_sha256") != config_sha:
        raise ValueError("checkpoint/config lineage mismatch")
    socket_path = _resolve(str(config["frozen_inputs"]["socket_checkpoint"]))
    if sha256_file(socket_path) != config["frozen_inputs"]["socket_checkpoint_sha256"]:
        raise ValueError("Socket checkpoint SHA mismatch")

    device = choose_deterministic_device(args.device)
    socket = load_socket_checkpoint(socket_path, device=device)
    head = DirectHardEdgePriority(
        int(config["model"]["input_dimension"]),
        hidden_dimension=int(config["model"]["hidden_dimension"]),
    ).to(device)
    head.load_state_dict(checkpoint["state_dict"], strict=True)
    head.eval()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    source_names = list(config["selection"]["d1_source_filenames"])
    if names_digest(source_names) != config["selection"]["d1_source_order_digest"]:
        raise ValueError("D1 roster digest mismatch")
    records = _record_lookup(manifest, source_names)
    cache = CleanTileCache(args.targets)
    seed = int(config["training"]["synthetic_seed"])
    decoder_config = SocketDecoderConfig(
        component_edge_budget_per_axis=144,
        max_swap_steps=24,
    )
    cyclic_config = CyclicTranslationConfig(border_weight=5.0)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "frozen_predictions.json"
    report_path = output_dir / "report.json"
    if prediction_path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite the origin-transfer diagnostic")
    print(
        json.dumps(
            {
                "event": "start",
                "pid": os.getpid(),
                "device": str(device),
                "sources": len(records),
                "same_opened_d1_only": True,
            }
        ),
        flush=True,
    )
    started = perf_counter()
    predictions: list[dict[str, Any]] = []
    with torch.no_grad():
        for index, record in enumerate(records):
            dirty, _ = _make_case(cache, record, draw_index=0, seed=seed + 10_000)
            board, _, scores, output = _forward_board(
                socket,
                head,
                dirty.tiles,
                device=device,
            )
            priorities = learned_priority_matrices(board, scores, grid=GRID)
            baseline_precyclic = decode_socket_assignments(
                output.right_log_assignment,
                output.down_log_assignment,
                grid=GRID,
                config=decoder_config,
            )
            learned_precyclic = decode_socket_assignments(
                output.right_log_assignment,
                output.down_log_assignment,
                grid=GRID,
                config=decoder_config,
                component_edge_priority=priorities,
            )
            baseline = select_global_cyclic_translation(
                baseline_precyclic.layout,
                output.right_log_assignment,
                output.down_log_assignment,
                grid=GRID,
                config=cyclic_config,
            )
            independent = select_global_cyclic_translation(
                learned_precyclic.layout,
                output.right_log_assignment,
                output.down_log_assignment,
                grid=GRID,
                config=cyclic_config,
            )
            transferred = transfer_cyclic_origin_by_baseline_overlap(
                learned_precyclic.layout,
                baseline.layout,
                grid=GRID,
            )
            layouts = (baseline.layout, independent.layout, transferred.layout)
            if any(
                not np.array_equal(np.sort(layout), np.arange(TILE_COUNT))
                for layout in layouts
            ):
                raise RuntimeError("origin-transfer output is not a strict permutation")
            predictions.append(
                {
                    "source_filename": str(record["filename"]),
                    "case_id": dirty.case_id,
                    "dirty_sha256": _dirty_sha256(dirty.tiles),
                    "baseline_final_layout": baseline.layout.tolist(),
                    "learned_precyclic_layout": learned_precyclic.layout.tolist(),
                    "learned_independent_cyclic5_layout": independent.layout.tolist(),
                    "learned_baseline_transferred_layout": transferred.layout.tolist(),
                    "transfer_row_roll": transferred.row_roll,
                    "transfer_column_roll": transferred.column_roll,
                    "baseline_overlap_count": transferred.overlap_count,
                }
            )
            print(
                json.dumps(
                    {"event": "freeze-prediction", "done": index + 1, "total": len(records)}
                ),
                flush=True,
            )
    artifact = {
        "schema": "aiijc-direct-hard-edge-baseline-origin-transfer-predictions-v1",
        "method": (
            "single stable row-major argmax of tilewise overlap between learned "
            "precyclic global rolls and raw decoder144+cyclic5 final layout"
        ),
        "target_free_selection": True,
        "same_opened_d1_only": True,
        "config_sha256": config_sha,
        "prior_report_sha256": prior_sha,
        "source_filenames": source_names,
        "source_order_digest": names_digest(source_names),
        "predictions_frozen_before_reference_scoring": True,
        "competition_test_opened": False,
        "predictions": predictions,
    }
    prediction_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    prediction_sha = sha256_file(prediction_path)
    print(
        json.dumps(
            {
                "event": "predictions-frozen",
                "path": str(prediction_path),
                "sha256": prediction_sha,
            }
        ),
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    for index, (record, prediction) in enumerate(zip(records, predictions, strict=True)):
        dirty, reference = _make_case(cache, record, draw_index=0, seed=seed + 10_000)
        if dirty.case_id != prediction["case_id"] or _dirty_sha256(dirty.tiles) != prediction[
            "dirty_sha256"
        ]:
            raise RuntimeError("scoring input differs from frozen origin-transfer input")
        metrics = {
            "baseline": evaluate_layout(
                prediction["baseline_final_layout"],
                reference.tile_at_position,
                reference_is_exact=True,
            ),
            "independent": evaluate_layout(
                prediction["learned_independent_cyclic5_layout"],
                reference.tile_at_position,
                reference_is_exact=True,
            ),
            "transferred": evaluate_layout(
                prediction["learned_baseline_transferred_layout"],
                reference.tile_at_position,
                reference_is_exact=True,
            ),
        }
        row: dict[str, Any] = {
            "source_filename": str(record["filename"]),
            "baseline_overlap_count": int(prediction["baseline_overlap_count"]),
        }
        for arm, metric in metrics.items():
            row[f"{arm}_exact_tiles"] = metric.correct_tile_count
            row[f"{arm}_adjacency"] = metric.adjacency
            row[f"{arm}_translation_aligned_tiles"] = metric.translation_aligned_count
        rows.append(row)
        print(
            json.dumps({"event": "score", "done": index + 1, "total": len(records)}),
            flush=True,
        )
    aggregate = _aggregate(rows)
    _assert_reproduces_prior(aggregate, prior_report)
    report = {
        "schema": "aiijc-direct-hard-edge-baseline-origin-transfer-report-v1",
        "status": "same-opened-d1-development-only",
        "target_free_selection": True,
        "no_weight_or_arm_sweep": True,
        "config_sha256": config_sha,
        "prior_report": str(args.prior_report.resolve()),
        "prior_report_sha256": prior_sha,
        "frozen_predictions": str(prediction_path),
        "frozen_predictions_sha256": prediction_sha,
        "source_filenames": source_names,
        "source_order_digest": names_digest(source_names),
        "rows": rows,
        "aggregate": aggregate,
        "runtime_seconds": perf_counter() - started,
        "strict_original_permutations": 3 * len(records),
        "promotion_authorized": False,
        "competition_test_opened": False,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "event": "complete",
                "report": str(report_path),
                "report_sha256": sha256_file(report_path),
                "aggregate": aggregate,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

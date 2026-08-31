#!/usr/bin/env python3
"""Target-assisted bottleneck audit on the already-open frame-side eval32.

This diagnostic never selects or opens another source.  It compares frozen
learned/Socket frame sets with exact frame memberships and measures the cyclic
roll ceiling of the unchanged raw d64 decoder144 layout.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from aiijc_puzzle.component_relation_reranker import extract_frozen_socket_context
from aiijc_puzzle.frame_side_origin import (
    SIDES,
    frame_topk_metrics,
    select_frame_cyclic_translation,
)
from aiijc_puzzle.layout_evaluation import evaluate_layout
from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.socket_sorter_production import load_socket_checkpoint

try:
    from scripts.run_component_relation_reranker import (
        GRID,
        CleanTileCache,
        _tile_tensor,
        prepare_case,
    )
    from scripts.run_frame_side_origin import (
        DEFAULT_CONFIG,
        DEFAULT_MANIFEST,
        DEFAULT_TARGETS,
        SOCKET_CHECKPOINT,
        SOCKET_SHA256,
        _load_config,
        _load_json,
        _load_rosters,
        _write_json,
    )
except ModuleNotFoundError:  # Direct ``python scripts/analyze_*.py`` execution.
    from run_component_relation_reranker import (
        GRID,
        CleanTileCache,
        _tile_tensor,
        prepare_case,
    )
    from run_frame_side_origin import (
        DEFAULT_CONFIG,
        DEFAULT_MANIFEST,
        DEFAULT_TARGETS,
        SOCKET_CHECKPOINT,
        SOCKET_SHA256,
        _load_config,
        _load_json,
        _load_rosters,
        _write_json,
    )

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FROZEN = (
    PROJECT_ROOT
    / "outputs/frame-side-origin/v1-fit256-s600-eval32/frozen_predictions.json"
)
DEFAULT_REPORT = PROJECT_ROOT / "outputs/frame-side-origin/v1-fit256-s600-eval32/report.json"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs/frame-side-origin/v1-fit256-s600-eval32/bottleneck_diagnostic_v2.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--frozen", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--allow-nondeterministic-mps", action="store_true")
    return parser.parse_args()


def _true_frame_sets(tile_to_position: np.ndarray) -> np.ndarray:
    row, column = divmod(np.asarray(tile_to_position, dtype=np.int64), GRID)
    masks = (row == 0, row == GRID - 1, column == 0, column == GRID - 1)
    return np.stack([np.flatnonzero(mask) for mask in masks]).astype(np.int32)


def _best_cyclic_oracle(
    layout: np.ndarray,
    reference: np.ndarray,
) -> dict[str, Any]:
    board = np.asarray(layout, dtype=np.int32).reshape(GRID, GRID)
    best: tuple[int, float, int, int] | None = None
    for row_roll in range(GRID):
        for column_roll in range(GRID):
            candidate = np.roll(board, (row_roll, column_roll), (0, 1)).reshape(-1)
            metrics = evaluate_layout(candidate, reference, reference_is_exact=True)
            key = (
                metrics.correct_tile_count,
                metrics.adjacency,
                -row_roll,
                -column_roll,
            )
            if best is None or key > best:
                best = key
    assert best is not None
    return {
        "correct_tile_count": best[0],
        "adjacency": best[1],
        "row_roll": -best[2],
        "column_roll": -best[3],
    }


def _frozen_base_layout(row: Mapping[str, Any]) -> np.ndarray:
    candidate = np.asarray(row["candidate_tile_at_position"], dtype=np.int32).reshape(
        GRID, GRID
    )
    diagnostics = row["candidate"]["diagnostics"]
    base = np.roll(
        candidate,
        (-int(diagnostics["selected_row_roll"]), -int(diagnostics["selected_column_roll"])),
        (0, 1),
    )
    comparator = np.asarray(
        row["comparator_tile_at_position"], dtype=np.int32
    ).reshape(GRID, GRID)
    comparator_diagnostics = row["comparator"]["diagnostics"]
    comparator_base = np.roll(
        comparator,
        (
            -int(comparator_diagnostics["selected_row_roll"]),
            -int(comparator_diagnostics["selected_column_roll"]),
        ),
        (0, 1),
    )
    if not np.array_equal(base, comparator_base):
        raise RuntimeError("frozen candidate/comparator do not share one decoder layout")
    return np.ascontiguousarray(base.reshape(-1))


def _mean_layout(rows: list[Mapping[str, Any]], key: str) -> dict[str, float]:
    return {
        field: float(np.mean([row[key][field] for row in rows]))
        for field in ("correct_tile_count", "adjacency")
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite frame-side diagnostic")
    if args.device == "mps":
        if not torch.backends.mps.is_available() or not args.allow_nondeterministic_mps:
            raise ValueError("MPS diagnostic requires explicit acknowledgement")
        torch.use_deterministic_algorithms(False)
    elif args.allow_nondeterministic_mps:
        raise ValueError("MPS acknowledgement supplied for CPU")
    else:
        torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    config, config_sha256 = _load_config(args.config)
    manifest = _load_json(args.manifest)
    _, eval_records, selection, _ = _load_rosters(config, manifest)
    frozen = _load_json(args.frozen)
    report = _load_json(args.report)
    if frozen["evaluation_digest"] != selection["evaluation_order_digest"]:
        raise ValueError("frozen predictions do not match the opened eval32")
    if report["freeze"]["sha256"] != sha256_file(args.frozen):
        raise ValueError("frozen prediction receipt changed")
    if sha256_file(SOCKET_CHECKPOINT) != SOCKET_SHA256:
        raise ValueError("Socket checkpoint changed")
    frozen_lookup = {row["source_filename"]: row for row in frozen["rows"]}
    if list(frozen_lookup) != list(selection["evaluation_filenames"]):
        raise ValueError("frozen row order does not match the preregistered eval32")
    socket = load_socket_checkpoint(SOCKET_CHECKPOINT, device=device)
    cache = CleanTileCache(args.targets)
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(eval_records):
        case = prepare_case(
            cache,
            record,
            draw_index=int(config["evaluation"]["draw_index"]),
            seed=int(config["evaluation"]["seed"]),
        )
        with torch.inference_mode():
            _, output = extract_frozen_socket_context(
                socket.model,
                _tile_tensor(case.dirty_tiles, device=device),
                grid=GRID,
            )
        frozen_row = frozen_lookup[case.source_filename]
        base_layout = _frozen_base_layout(frozen_row)
        learned_sets = np.asarray(frozen_row["candidate_top24_sets"], dtype=np.int32)
        socket_sets = np.asarray(frozen_row["socket_top24_sets"], dtype=np.int32)
        truth_sets = _true_frame_sets(case.input_tile_to_position)
        reference = np.argsort(case.input_tile_to_position).astype(np.int32)
        oracle_arm_sets = []
        union_correct: dict[str, int] = {}
        for side, name in enumerate(SIDES):
            truth = set(truth_sets[side].tolist())
            learned_correct = len(truth & set(learned_sets[side].tolist()))
            socket_correct = len(truth & set(socket_sets[side].tolist()))
            oracle_arm_sets.append(
                learned_sets[side] if learned_correct > socket_correct else socket_sets[side]
            )
            union_correct[name] = len(
                truth & (set(learned_sets[side].tolist()) | set(socket_sets[side].tolist()))
            )
        placements = {
            "socket_top24_frame_roll": select_frame_cyclic_translation(
                base_layout,
                socket_sets,
                output.right_log_assignment,
                output.down_log_assignment,
                grid=GRID,
            ).layout,
            "frozen_learned_top24_frame_roll": np.asarray(
                frozen_row["candidate_tile_at_position"], dtype=np.int32
            ),
            "frozen_cyclic_border5_comparator": np.asarray(
                frozen_row["comparator_tile_at_position"], dtype=np.int32
            ),
            "oracle_arm_per_side_frame_roll": select_frame_cyclic_translation(
                base_layout,
                np.stack(oracle_arm_sets),
                output.right_log_assignment,
                output.down_log_assignment,
                grid=GRID,
            ).layout,
            "true_frame_membership_roll": select_frame_cyclic_translation(
                base_layout,
                truth_sets,
                output.right_log_assignment,
                output.down_log_assignment,
                grid=GRID,
            ).layout,
        }
        row: dict[str, Any] = {
            "case_id": case.case_id,
            "source_filename": case.source_filename,
            "learned_frame": frame_topk_metrics(
                learned_sets, case.input_tile_to_position, grid=GRID
            ),
            "socket_frame": frame_topk_metrics(
                socket_sets, case.input_tile_to_position, grid=GRID
            ),
            "union_true_recall": {
                name: union_correct[name] / GRID for name in SIDES
            },
            "best_exact_cyclic_roll": _best_cyclic_oracle(base_layout, reference),
        }
        for key, layout in placements.items():
            metrics = evaluate_layout(layout, reference, reference_is_exact=True)
            row[key] = {
                "correct_tile_count": metrics.correct_tile_count,
                "adjacency": metrics.adjacency,
            }
        rows.append(row)
        if (index + 1) % 8 == 0:
            print(json.dumps({"event": "diagnostic", "sources": index + 1}), flush=True)
    summary = {
        "source_count": len(rows),
        "learned_frame_macro_f1": float(
            np.mean([row["learned_frame"]["macro_f1"] for row in rows])
        ),
        "socket_frame_macro_f1": float(
            np.mean([row["socket_frame"]["macro_f1"] for row in rows])
        ),
        "union_true_recall": {
            name: float(np.mean([row["union_true_recall"][name] for row in rows]))
            for name in SIDES
        },
        "layouts": {
            key: _mean_layout(rows, key)
            for key in (
                "socket_top24_frame_roll",
                "frozen_learned_top24_frame_roll",
                "frozen_cyclic_border5_comparator",
                "oracle_arm_per_side_frame_roll",
                "true_frame_membership_roll",
                "best_exact_cyclic_roll",
            )
        },
    }
    payload = {
        "experiment": "frame-side-origin-opened-eval32-bottleneck-diagnostic-v2",
        "status": "target-assisted-diagnostic-only-not-deployable",
        "same_opened_eval32_only": True,
        "new_source_opened": False,
        "config_sha256": config_sha256,
        "frozen_predictions_sha256": sha256_file(args.frozen),
        "preregistered_report_sha256": sha256_file(args.report),
        "summary": summary,
        "fresh64_opened": False,
        "holdout_opened": False,
        "competition_test_opened": False,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.output, payload)
    print(
        json.dumps(
            {
                "event": "complete",
                "output": str(args.output),
                "sha256": sha256_file(args.output),
                "summary": summary,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Engineering-only full-score fragment synchronization on opened frozen64.

This runner reuses the established source-disjoint Union-v2 frozen64 roster as
an already-opened D1 panel.  It is suitable for bounded mechanism debugging,
not for a fresh promotion claim.  Candidate layouts and reports are written
before the synthetic reference permutations are scored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from aiijc_puzzle.protocol import compute_protocol_digest, sha256_file
from aiijc_puzzle.raw_twin_union_production import (
    CYCLIC_BORDER_WEIGHT,
    FROZEN_UNION_CHECKPOINT_SHA256,
    infer_raw_twin_union_assignments,
    load_fullres_twin_checkpoint,
    load_raw_twin_union_checkpoint,
)
from aiijc_puzzle.socket_decoder import SocketDecoderConfig, decode_socket_assignments
from aiijc_puzzle.socket_sorter_production import (
    DECODER_EDGE_BUDGET,
    DECODER_SWAP_STEPS,
    load_socket_checkpoint,
)
from aiijc_puzzle.socket_translation_placer import (
    CyclicTranslationConfig,
    select_global_cyclic_translation,
)
from aiijc_puzzle.union_fragment_synchronizer import (
    UnionFragmentSynchronizerConfig,
    decode_union_fragment_layout,
)

try:
    from scripts.run_fullres_twin_side_matcher import (
        _atomic_json,
        _prepare_boards,
        _two_view_case,
    )
    from scripts.run_raw_twin_union_reranker_fresh64 import (
        DEFAULT_CONFIG,
        DEFAULT_MANIFEST,
        DEFAULT_TARGETS,
        SOCKET_CHECKPOINT,
        TWIN_CHECKPOINT,
        UNION_CHECKPOINT,
        UNION_SELECTION,
        _case_seeds,
        load_config,
        source_clustered_ci,
    )
    from scripts.run_raw_twin_union_reranker_v2 import _adjacency_fraction
except ModuleNotFoundError:  # Direct ``python scripts/run_*.py`` execution.
    from run_fullres_twin_side_matcher import _atomic_json, _prepare_boards, _two_view_case
    from run_raw_twin_union_reranker_fresh64 import (
        DEFAULT_CONFIG,
        DEFAULT_MANIFEST,
        DEFAULT_TARGETS,
        SOCKET_CHECKPOINT,
        TWIN_CHECKPOINT,
        UNION_CHECKPOINT,
        UNION_SELECTION,
        _case_seeds,
        load_config,
        source_clustered_ci,
    )
    from run_raw_twin_union_reranker_v2 import _adjacency_fraction


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UNION_CONFIG = PROJECT_ROOT / "configs/raw_twin_union_reranker_v2_preregistered.json"
FROZEN_PANEL = (
    PROJECT_ROOT / "outputs/raw-twin-union-reranker/frozen-v2-fresh64-draw0"
)
FROZEN_PREDICTIONS = FROZEN_PANEL / "frozen-target-free-predictions.npz"
FROZEN_METADATA = FROZEN_PANEL / "frozen-target-free-predictions.json"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs/union-fragment-synchronizer/d1-opened-frozen64-v1"
)
GRID = 24
COUNT = GRID * GRID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=64)
    return parser.parse_args()


def _cached_layout(archive: Any, prefix: str) -> np.ndarray:
    key = f"{prefix}__learned_union__layout"
    if key not in archive:
        raise KeyError(f"frozen archive is missing {key}")
    layout = np.ascontiguousarray(archive[key], dtype=np.int32)
    if layout.shape != (COUNT,) or not np.array_equal(np.sort(layout), np.arange(COUNT)):
        raise ValueError("cached Union layout is not a strict permutation")
    return layout


def _redecode_union_layout(inference: Any) -> np.ndarray:
    decoded = decode_socket_assignments(
        inference.learned_right_log_assignment,
        inference.learned_down_log_assignment,
        grid=GRID,
        config=SocketDecoderConfig(
            component_edge_budget_per_axis=DECODER_EDGE_BUDGET,
            swap_edge_budget_per_axis=DECODER_EDGE_BUDGET,
            max_swap_steps=DECODER_SWAP_STEPS,
        ),
    )
    return np.ascontiguousarray(
        select_global_cyclic_translation(
            decoded.layout,
            inference.learned_right_log_assignment,
            inference.learned_down_log_assignment,
            grid=GRID,
            config=CyclicTranslationConfig(border_weight=CYCLIC_BORDER_WEIGHT),
        ).layout,
        dtype=np.int32,
    )


def _mean(rows: list[dict[str, Any]], arm: str, metric: str) -> float:
    return float(np.mean([float(row[arm][metric]) for row in rows]))


def run(args: argparse.Namespace) -> None:
    if not 1 <= args.limit <= 64:
        raise ValueError("limit must be in [1, 64]")
    config, config_sha = load_config(args.config)
    metadata = json.loads(FROZEN_METADATA.read_text(encoding="utf-8"))
    cases = metadata.get("cases")
    if not isinstance(cases, list) or len(cases) != 64:
        raise ValueError("expected the established frozen64 case metadata")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("protocol_digest") != compute_protocol_digest(manifest):
        raise ValueError("manifest protocol digest mismatch")
    names = tuple(config["selection"]["source_filenames"][: args.limit])
    case_names = tuple(str(case["source_filename"]) for case in cases[: args.limit])
    if names != case_names:
        raise ValueError("config and frozen metadata source order differ")
    lookup = {str(record["filename"]): dict(record) for record in manifest["splits"]["train"]}
    records = tuple(lookup[name] for name in names)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "frozen-target-free-layouts.npz"
    metadata_path = output_dir / "frozen-target-free-layouts.json"
    report_path = output_dir / "report.json"
    if any(path.exists() for path in (prediction_path, metadata_path, report_path)):
        raise FileExistsError("refusing to overwrite a fragment-synchronizer run")

    boards = _prepare_boards(records, args.targets)
    device = torch.device("cpu")
    socket = load_socket_checkpoint(SOCKET_CHECKPOINT, device=device)
    twin = load_fullres_twin_checkpoint(TWIN_CHECKPOINT, device=device)
    union = load_raw_twin_union_checkpoint(
        UNION_CHECKPOINT,
        config_path=UNION_CONFIG,
        selection_path=UNION_SELECTION,
        device=device,
    )
    if union.sha256 != FROZEN_UNION_CHECKPOINT_SHA256:
        raise ValueError("Union checkpoint identity changed")

    arrays: dict[str, np.ndarray] = {}
    frozen_rows: list[dict[str, Any]] = []
    references: list[np.ndarray] = []
    started = perf_counter()
    synthetic_seed = int(config["selection"]["synthetic_seed"])
    synchronizer_config = UnionFragmentSynchronizerConfig()
    with np.load(FROZEN_PREDICTIONS) as frozen_archive, torch.inference_mode():
        for index, (case, board) in enumerate(
            zip(cases[: args.limit], boards, strict=True)
        ):
            corruption_seed, permutation_seed = _case_seeds(
                synthetic_seed,
                board.filename,
            )
            dirty, _, reference = _two_view_case(
                board.tiles,
                first_seed=corruption_seed,
                second_seed=corruption_seed + 1,
                permutation_seed=permutation_seed,
            )
            inference = infer_raw_twin_union_assignments(
                dirty,
                socket,
                twin,
                union,
                device=device,
            )
            prefix = str(case["prefix"])
            fallback = _cached_layout(frozen_archive, prefix)
            replay = _redecode_union_layout(inference)
            if not np.array_equal(replay, fallback):
                raise RuntimeError("regenerated Union-v2 baseline differs from frozen layout")
            solver_started = perf_counter()
            result = decode_union_fragment_layout(
                inference.learned_right_log_assignment,
                inference.learned_down_log_assignment,
                inference.candidate_snapshot,
                fallback,
                config=synchronizer_config,
            )
            solver_seconds = perf_counter() - solver_started
            if not np.array_equal(np.sort(result.layout), np.arange(COUNT)):
                raise RuntimeError("fragment synchronizer emitted a non-permutation")
            arrays[f"{prefix}__layout"] = np.asarray(result.layout, dtype=np.int32)
            frozen_rows.append(
                {
                    "prefix": prefix,
                    "source_filename": board.filename,
                    "candidate_snapshot_sha256": inference.candidate_snapshot.sha256,
                    "solver_seconds": solver_seconds,
                    "used_fallback": result.used_fallback,
                    "fallback_reason": result.fallback_reason,
                    "solver_report": result.report(),
                }
            )
            references.append(np.ascontiguousarray(reference, dtype=np.int32))
            print(
                json.dumps(
                    {
                        "event": "freeze",
                        "done": index + 1,
                        "total": args.limit,
                        "fallback": result.used_fallback,
                        "solver_seconds": solver_seconds,
                    }
                ),
                flush=True,
            )

    np.savez_compressed(prediction_path, **arrays)
    _atomic_json(
        metadata_path,
        {
            "schema": "aiijc-union-fragment-synchronizer-opened-d1-predictions-v1",
            "panel_role": "already-opened engineering D1; not promotion evidence",
            "contains_exact_references": False,
            "contains_dirty_or_clean_pixels": False,
            "contains_target_free_strict_layouts": True,
            "config": {
                "hard_edge_budget_per_axis": synchronizer_config.hard_edge_budget_per_axis,
                "synchronization_passes": synchronizer_config.synchronization_passes,
                "milp_time_limit_seconds": synchronizer_config.milp_time_limit_seconds,
                "milp_relative_gap": synchronizer_config.milp_relative_gap,
                "cyclic_border_weight": synchronizer_config.cyclic_border_weight,
            },
            "cases": frozen_rows,
        },
    )
    prediction_sha = sha256_file(prediction_path)
    metadata_sha = sha256_file(metadata_path)
    print(
        json.dumps(
            {
                "event": "predictions_frozen_before_scoring",
                "sha256": prediction_sha,
            }
        ),
        flush=True,
    )

    scored_rows: list[dict[str, Any]] = []
    with np.load(FROZEN_PREDICTIONS) as baseline_archive, np.load(
        prediction_path
    ) as candidate_archive:
        for case, reference in zip(cases[: args.limit], references, strict=True):
            prefix = str(case["prefix"])
            baseline = _cached_layout(baseline_archive, prefix)
            candidate = np.asarray(candidate_archive[f"{prefix}__layout"], dtype=np.int32)
            scored_rows.append(
                {
                    "source_filename": str(case["source_filename"]),
                    "union_v2": {
                        "exact_tiles": int(np.count_nonzero(baseline == reference)),
                        "adjacency": float(_adjacency_fraction(baseline, reference)),
                    },
                    "fragment_sync": {
                        "exact_tiles": int(np.count_nonzero(candidate == reference)),
                        "adjacency": float(_adjacency_fraction(candidate, reference)),
                    },
                }
            )

    exact_deltas = [
        row["fragment_sync"]["exact_tiles"] - row["union_v2"]["exact_tiles"]
        for row in scored_rows
    ]
    adjacency_deltas = [
        row["fragment_sync"]["adjacency"] - row["union_v2"]["adjacency"]
        for row in scored_rows
    ]
    metrics = {
        "arms": {
            "union_v2": {
                "exact_tiles_per_board": _mean(scored_rows, "union_v2", "exact_tiles"),
                "adjacency": _mean(scored_rows, "union_v2", "adjacency"),
            },
            "fragment_sync": {
                "exact_tiles_per_board": _mean(
                    scored_rows,
                    "fragment_sync",
                    "exact_tiles",
                ),
                "adjacency": _mean(scored_rows, "fragment_sync", "adjacency"),
            },
        },
        "exact_delta": source_clustered_ci(exact_deltas, seed=20330991),
        "adjacency_delta": source_clustered_ci(adjacency_deltas, seed=20330992),
        "strict_boards": args.limit,
        "rigid_boards": sum(
            int(row["solver_report"]["audit"]["rigidity_preserved"])
            for row in frozen_rows
        ),
        "fallback_boards": sum(int(row["used_fallback"]) for row in frozen_rows),
        "mean_solver_seconds": float(
            np.mean([float(row["solver_seconds"]) for row in frozen_rows])
        ),
    }
    gate = {
        "exact_delta_at_least_quarter_tile": float(metrics["exact_delta"]["mean"])
        >= 0.25,
        "adjacency_nonnegative": float(metrics["adjacency_delta"]["mean"]) >= 0.0,
        "all_strict": metrics["strict_boards"] == args.limit,
        "passed": False,
    }
    gate["passed"] = all(
        gate[key]
        for key in (
            "exact_delta_at_least_quarter_tile",
            "adjacency_nonnegative",
            "all_strict",
        )
    )
    _atomic_json(
        report_path,
        {
            "schema": "aiijc-union-fragment-synchronizer-opened-d1-report-v1",
            "status": "engineering-gate-pass" if gate["passed"] else "engineering-gate-fail",
            "panel_role": "already-opened engineering D1; not promotion evidence",
            "frozen_union_config_sha256": config_sha,
            "frozen_union_checkpoint_sha256": union.sha256,
            "predictions": {
                "path": str(prediction_path.relative_to(PROJECT_ROOT)),
                "sha256": prediction_sha,
                "metadata_path": str(metadata_path.relative_to(PROJECT_ROOT)),
                "metadata_sha256": metadata_sha,
                "frozen_before_reference_scoring": True,
                "contains_exact_references": False,
            },
            "metrics": metrics,
            "gate": gate,
            "rows": scored_rows,
            "runtime_seconds": perf_counter() - started,
            "organizer_holdout_or_test_opened": False,
            "original_upright_tile_permutations_only": True,
        },
    )
    print(
        json.dumps(
            {
                "event": "complete",
                "report": str(report_path),
                "report_sha256": sha256_file(report_path),
                "metrics": metrics,
                "gate": gate,
            }
        ),
        flush=True,
    )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

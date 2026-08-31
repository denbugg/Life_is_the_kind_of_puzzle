#!/usr/bin/env python3
"""Opened-panel pilot: fullres pre-denoise fusion into rigid global solver.

The frozen full-resolution denoiser and learned raw+restored relation selector
are used only as target-blind matcher views.  Their component relations are
expanded to canonical tile contacts and consumed by the reversible fragment
synchronizer.  Final layouts remain strict permutations of the original
upright dirty tiles.  The reused D2 source40 panel is engineering evidence,
not a fresh promotion panel.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from aiijc_puzzle.fullres_fusion_snapshot import build_fullres_fusion_snapshot
from aiijc_puzzle.layout_evaluation import evaluate_layout
from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.raw_twin_union_production import (
    CYCLIC_BORDER_WEIGHT,
    infer_raw_twin_union_assignments,
    load_fullres_twin_checkpoint,
    load_raw_twin_union_checkpoint,
)
from aiijc_puzzle.socket_decoder import SocketDecoderConfig, decode_socket_assignments
from aiijc_puzzle.socket_sorter_production import (
    DECODER_EDGE_BUDGET,
    DECODER_SWAP_STEPS,
)
from aiijc_puzzle.socket_translation_placer import (
    CyclicTranslationConfig,
    select_global_cyclic_translation,
)
from aiijc_puzzle.union_fragment_synchronizer import decode_union_fragment_layout

try:
    from scripts.run_component_relation_reranker import CleanTileCache, prepare_case
    from scripts.run_fullres_relation_fusion import _load_config, _load_models, prepare_fusion_board
    from scripts.run_fullres_relation_fusion_decoder_d2 import (
        DEFAULT_CONFIG,
        DEFAULT_MANIFEST,
        DEFAULT_TARGETS,
        PROJECT_ROOT,
        _load_fusion,
        load_d2_config,
        selected_records,
        validate_frozen_inputs,
    )
except ModuleNotFoundError:  # Direct ``python scripts/run_*.py`` execution.
    from run_component_relation_reranker import CleanTileCache, prepare_case
    from run_fullres_relation_fusion import _load_config, _load_models, prepare_fusion_board
    from run_fullres_relation_fusion_decoder_d2 import (
        DEFAULT_CONFIG,
        DEFAULT_MANIFEST,
        DEFAULT_TARGETS,
        PROJECT_ROOT,
        _load_fusion,
        load_d2_config,
        selected_records,
        validate_frozen_inputs,
    )


UNION_DIR = PROJECT_ROOT / "outputs/raw-twin-union-reranker/v2-fit256-s400-eval24"
UNION_CHECKPOINT = UNION_DIR / "raw-twin-union-reranker-v2.pt"
UNION_SELECTION = UNION_DIR / "selection-commitment.json"
UNION_CONFIG = PROJECT_ROOT / "configs/raw_twin_union_reranker_v2_preregistered.json"
TWIN_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/fullres-twin-side-matcher/v1-fit256-s400-eval24"
    / "fullres-twin-side-matcher.pt"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs/fullres-fusion-fragment-solver/d1-opened-source40-v1"
)
GRID = 24
COUNT = GRID * GRID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--allow-nondeterministic-mps", action="store_true")
    parser.add_argument("--inference-batch", type=int, default=576)
    parser.add_argument("--limit", type=int, default=40)
    return parser.parse_args()


def _strict_layout(value: Any) -> np.ndarray:
    layout = np.asarray(value, dtype=np.int32)
    if layout.shape != (COUNT,) or not np.array_equal(np.sort(layout), np.arange(COUNT)):
        raise ValueError("layout is not a strict original-tile permutation")
    return np.ascontiguousarray(layout)


def _union_fallback(inference: Any) -> np.ndarray:
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
    cyclic = select_global_cyclic_translation(
        decoded.layout,
        inference.learned_right_log_assignment,
        inference.learned_down_log_assignment,
        grid=GRID,
        config=CyclicTranslationConfig(border_weight=CYCLIC_BORDER_WEIGHT),
    )
    return _strict_layout(cyclic.layout)


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _delta_ci(values: list[float], *, seed: int) -> dict[str, Any]:
    difference = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    chunks: list[np.ndarray] = []
    remaining = 20_000
    while remaining:
        batch = min(remaining, 2048)
        indices = generator.integers(0, len(difference), size=(batch, len(difference)))
        chunks.append(difference[indices].mean(axis=1))
        remaining -= batch
    distribution = np.concatenate(chunks)
    return {
        "mean": float(difference.mean()),
        "ci95_lower": float(np.quantile(distribution, 0.025)),
        "ci95_upper": float(np.quantile(distribution, 0.975)),
        "wins": int(np.count_nonzero(difference > 0)),
        "ties": int(np.count_nonzero(difference == 0)),
        "losses": int(np.count_nonzero(difference < 0)),
        "source_count": len(difference),
    }


def run(args: argparse.Namespace) -> None:
    if not 1 <= args.limit <= 40:
        raise ValueError("limit must be in [1, 40]")
    if args.inference_batch <= 0:
        raise ValueError("inference-batch must be positive")
    config, config_sha = load_d2_config(args.config)
    validate_frozen_inputs(config)
    seed = int(config["protocol"]["synthetic_seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.device == "mps":
        if not args.allow_nondeterministic_mps:
            raise ValueError("MPS requires --allow-nondeterministic-mps")
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable")
        torch.use_deterministic_algorithms(False)
        device = torch.device("mps")
    else:
        if args.allow_nondeterministic_mps:
            raise ValueError("allow-nondeterministic-mps requires MPS")
        torch.use_deterministic_algorithms(True)
        device = torch.device("cpu")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    all_records, _ = selected_records(config, manifest)
    records = all_records[: args.limit]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "frozen-target-free-layouts.npz"
    metadata_path = output_dir / "frozen-target-free-layouts.json"
    report_path = output_dir / "report.json"
    if any(path.exists() for path in (prediction_path, metadata_path, report_path)):
        raise FileExistsError("refusing to overwrite a fusion-fragment run")

    d1_config, _ = _load_config(
        PROJECT_ROOT / str(config["frozen_inputs"]["fusion_preregistration"])
    )
    socket, relation, denoiser, _ = _load_models(d1_config, device=device)
    fusion = _load_fusion(config, device=device)
    twin = load_fullres_twin_checkpoint(TWIN_CHECKPOINT, device=device)
    union = load_raw_twin_union_checkpoint(
        UNION_CHECKPOINT,
        config_path=UNION_CONFIG,
        selection_path=UNION_SELECTION,
        device=device,
    )
    candidate_contract = config["candidate_and_decoder"]
    cache = CleanTileCache(args.targets)
    arrays: dict[str, np.ndarray] = {}
    frozen_rows: list[dict[str, Any]] = []
    started = perf_counter()
    with torch.inference_mode():
        for index, record in enumerate(records):
            case_started = perf_counter()
            case = prepare_case(
                cache,
                record,
                draw_index=int(config["protocol"]["draw_index"]),
                seed=seed,
            )
            board = prepare_fusion_board(
                case,
                socket=socket,
                relation=relation,
                denoiser=denoiser,
                device=device,
                inference_batch=args.inference_batch,
                raw_topk=int(candidate_contract["raw_proposal_topk_per_exposed_member"]),
                raw_cap=int(candidate_contract["raw_candidate_cap_per_query"]),
                union_cap=int(candidate_contract["union_candidate_cap_per_query"]),
                attach_exact_labels=False,
            )
            if board.union_labels or board.oracle_relations or board.profiles:
                raise RuntimeError("exact labels entered target-blind fusion inference")
            features = torch.from_numpy(board.features).to(device)
            relation_scores = torch.from_numpy(board.frozen_relation_scores).to(device)
            fusion_output = fusion(features, relation_scores)
            snapshot, snapshot_diagnostics = build_fullres_fusion_snapshot(
                board.union_candidates,
                fusion_output.scores,
                grid=GRID,
            )
            union_inference = infer_raw_twin_union_assignments(
                case.dirty_tiles,
                socket,
                twin,
                union,
                device=device,
            )
            fallback = _union_fallback(union_inference)
            solver_started = perf_counter()
            result = decode_union_fragment_layout(
                union_inference.learned_right_log_assignment,
                union_inference.learned_down_log_assignment,
                snapshot,
                fallback,
            )
            solver_seconds = perf_counter() - solver_started
            layout = _strict_layout(result.layout)
            prefix = f"case_{index:04d}"
            arrays[f"{prefix}__union_v2_layout"] = fallback
            arrays[f"{prefix}__fusion_fragment_layout"] = layout
            frozen_rows.append(
                {
                    "prefix": prefix,
                    "case_id": case.case_id,
                    "source_filename": case.source_filename,
                    "fusion_snapshot_sha256": snapshot.sha256,
                    "fusion_snapshot": asdict(snapshot_diagnostics),
                    "solver": result.report(),
                    "used_fallback": result.used_fallback,
                    "runtime_seconds": {
                        **board.runtime_seconds,
                        "fragment_solver": solver_seconds,
                        "case_total": perf_counter() - case_started,
                    },
                }
            )
            print(
                json.dumps(
                    {
                        "event": "freeze",
                        "done": index + 1,
                        "total": args.limit,
                        "relations": len(board.union_candidates),
                        "tile_edges": snapshot.count,
                        "fallback": result.used_fallback,
                        "case_seconds": perf_counter() - case_started,
                    }
                ),
                flush=True,
            )

    np.savez_compressed(prediction_path, **arrays)
    _write_json(
        metadata_path,
        {
            "schema": "aiijc-fullres-fusion-fragment-opened-d1-predictions-v1",
            "panel_role": "reused opened source40; engineering evidence only",
            "contains_exact_references": False,
            "contains_pixels": False,
            "restored_pixels_matcher_view_only": True,
            "strict_original_upright_tile_layouts": True,
            "rows": frozen_rows,
        },
    )
    prediction_sha = sha256_file(prediction_path)
    metadata_sha = sha256_file(metadata_path)
    print(
        json.dumps(
            {
                "event": "layouts-frozen-before-scoring",
                "sha256": prediction_sha,
            }
        ),
        flush=True,
    )

    scored_rows: list[dict[str, Any]] = []
    scoring_cache = CleanTileCache(args.targets)
    with np.load(prediction_path) as archive:
        for record, frozen in zip(records, frozen_rows, strict=True):
            case = prepare_case(
                scoring_cache,
                record,
                draw_index=int(config["protocol"]["draw_index"]),
                seed=seed,
            )
            if case.case_id != frozen["case_id"]:
                raise RuntimeError("scoring phase recreated a different synthetic case")
            reference = np.argsort(case.input_tile_to_position).astype(np.int32)
            prefix = str(frozen["prefix"])
            baseline = _strict_layout(archive[f"{prefix}__union_v2_layout"])
            candidate = _strict_layout(archive[f"{prefix}__fusion_fragment_layout"])
            baseline_metrics = evaluate_layout(
                baseline,
                reference,
                reference_is_exact=True,
            ).as_dict()
            candidate_metrics = evaluate_layout(
                candidate,
                reference,
                reference_is_exact=True,
            ).as_dict()
            scored_rows.append(
                {
                    "source_filename": frozen["source_filename"],
                    "case_id": frozen["case_id"],
                    "union_v2": baseline_metrics,
                    "fusion_fragment": candidate_metrics,
                    "exact_delta": int(
                        candidate_metrics["correct_tile_count"]
                        - baseline_metrics["correct_tile_count"]
                    ),
                    "adjacency_delta": float(
                        candidate_metrics["adjacency"] - baseline_metrics["adjacency"]
                    ),
                }
            )
    exact_delta = [float(row["exact_delta"]) for row in scored_rows]
    adjacency_delta = [float(row["adjacency_delta"]) for row in scored_rows]
    metrics = {
        "arms": {
            arm: {
                "exact_tiles_per_board": float(
                    np.mean([row[arm]["correct_tile_count"] for row in scored_rows])
                ),
                "adjacency": float(
                    np.mean([row[arm]["adjacency"] for row in scored_rows])
                ),
                "translation_aligned_tiles_per_board": float(
                    np.mean(
                        [row[arm]["translation_aligned_count"] for row in scored_rows]
                    )
                ),
            }
            for arm in ("union_v2", "fusion_fragment")
        },
        "exact_delta": _delta_ci(exact_delta, seed=20321031),
        "adjacency_delta": _delta_ci(adjacency_delta, seed=20321032),
        "fallback_boards": sum(int(row["used_fallback"]) for row in frozen_rows),
        "strict_boards": args.limit,
        "mean_case_seconds": float(
            np.mean([row["runtime_seconds"]["case_total"] for row in frozen_rows])
        ),
    }
    gate = {
        "exact_delta_at_least_quarter_tile": metrics["exact_delta"]["mean"] >= 0.25,
        "adjacency_nonnegative": metrics["adjacency_delta"]["mean"] >= 0.0,
        "all_strict": metrics["strict_boards"] == args.limit,
    }
    gate["passed"] = all(gate.values())
    _write_json(
        report_path,
        {
            "schema": "aiijc-fullres-fusion-fragment-opened-d1-report-v1",
            "status": "engineering-gate-pass" if gate["passed"] else "engineering-gate-fail",
            "panel_role": "reused opened source40; engineering evidence only",
            "config_sha256": config_sha,
            "predictions": {
                "path": str(prediction_path.relative_to(PROJECT_ROOT)),
                "sha256": prediction_sha,
                "metadata": str(metadata_path.relative_to(PROJECT_ROOT)),
                "metadata_sha256": metadata_sha,
                "frozen_before_exact_scoring": True,
            },
            "metrics": metrics,
            "gate": gate,
            "rows": scored_rows,
            "runtime_seconds": perf_counter() - started,
            "organizer_holdout_or_test_opened": False,
            "restored_pixels_emitted": False,
            "original_upright_tile_permutations_only": True,
            "mps_determinism_claimed": device.type != "mps",
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

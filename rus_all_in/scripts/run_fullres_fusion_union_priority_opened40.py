#!/usr/bin/env python3
"""Evaluate denoise-aware Union hard-edge ordering on opened D2 source40.

This is an engineering conversion test, not fresh promotion evidence.  The
frozen full-resolution denoiser and relation-fusion head are matcher-only
views.  Their top32-query/top5-candidate evidence may reprioritise an edge only
when the identical directed tile edge already survives the frozen Union-v2
hard projection.  Both strict original-tile layouts and all 1,104 baseline and
treatment hard priorities are persisted before the synthetic references are
recreated and scored.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from aiijc_puzzle.fullres_fusion_union_priority import (
    FusionUnionPriorityConfig,
    build_fullres_fusion_union_priority,
)
from aiijc_puzzle.layout_evaluation import evaluate_layout
from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.raw_twin_union_production import (
    CYCLIC_BORDER_WEIGHT,
    FROZEN_TWIN_SHA256,
    FROZEN_UNION_CHECKPOINT_SHA256,
    infer_raw_twin_union_assignments,
    load_fullres_twin_checkpoint,
    load_raw_twin_union_checkpoint,
)
from aiijc_puzzle.socket_decoder import (
    SocketDecoderConfig,
    decode_socket_assignments,
    hard_partial_axis_matching,
)
from aiijc_puzzle.socket_sorter_production import (
    DECODER_EDGE_BUDGET,
    DECODER_SWAP_STEPS,
)
from aiijc_puzzle.socket_translation_placer import (
    CyclicTranslationConfig,
    select_global_cyclic_translation,
)

try:
    from scripts.run_component_relation_reranker import CleanTileCache, prepare_case
    from scripts.run_fullres_relation_fusion import (
        _load_config,
        _load_models,
        prepare_fusion_board,
    )
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
    from run_fullres_relation_fusion import (
        _load_config,
        _load_models,
        prepare_fusion_board,
    )
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
    PROJECT_ROOT / "outputs/fullres-fusion-union-priority/opened-source40-v1"
)
GRID = 24
COUNT = GRID * GRID
HARD_EDGES_PER_AXIS = GRID * (GRID - 1)
FIXED_TOP_EDGE_COUNT = 2 * DECODER_EDGE_BUDGET


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


def _validate_limit(limit: int) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 40:
        raise ValueError("limit must be in [1, 40]")
    return limit


def _select_device(name: str, *, allow_nondeterministic_mps: bool) -> torch.device:
    if name == "mps":
        if not allow_nondeterministic_mps:
            raise ValueError("MPS requires --allow-nondeterministic-mps")
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable")
        torch.use_deterministic_algorithms(False)
        return torch.device("mps")
    if name != "cpu":
        raise ValueError("device must be cpu or mps")
    if allow_nondeterministic_mps:
        raise ValueError("allow-nondeterministic-mps requires MPS")
    torch.use_deterministic_algorithms(True)
    return torch.device("cpu")


def _strict_layout(value: Any) -> np.ndarray:
    layout = np.asarray(value, dtype=np.int32)
    if layout.shape != (COUNT,) or not np.array_equal(np.sort(layout), np.arange(COUNT)):
        raise ValueError("layout is not a strict original-tile permutation")
    return np.ascontiguousarray(layout)


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _prepare_output_paths(output_dir: Path) -> tuple[Path, Path, Path]:
    resolved = output_dir.resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    paths = (
        resolved / "frozen-target-free-layouts-and-priorities.npz",
        resolved / "frozen-target-free-layouts-and-priorities.json",
        resolved / "report.json",
    )
    if any(path.exists() for path in paths):
        raise FileExistsError("refusing to overwrite a fusion-priority run")
    return paths


def _decode_layout(
    right: np.ndarray,
    down: np.ndarray,
    *,
    component_edge_priority: dict[str, np.ndarray] | None,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    decoder = decode_socket_assignments(
        right,
        down,
        grid=GRID,
        config=SocketDecoderConfig(
            component_edge_budget_per_axis=DECODER_EDGE_BUDGET,
            swap_edge_budget_per_axis=DECODER_EDGE_BUDGET,
            max_swap_steps=DECODER_SWAP_STEPS,
        ),
        component_edge_priority=component_edge_priority,
    )
    cyclic = select_global_cyclic_translation(
        decoder.layout,
        right,
        down,
        grid=GRID,
        config=CyclicTranslationConfig(border_weight=CYCLIC_BORDER_WEIGHT),
    )
    return _strict_layout(cyclic.layout), decoder.report(), cyclic.report()


def _edge_arrays(
    right: np.ndarray,
    down: np.ndarray,
    priorities: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for axis_index, (axis, assignment) in enumerate((('right', right), ('down', down))):
        matching = hard_partial_axis_matching(assignment, grid=GRID, axis=axis)
        source = np.asarray([edge.source for edge in matching.edges], dtype=np.int32)
        target = np.asarray([edge.target for edge in matching.edges], dtype=np.int32)
        baseline = np.asarray([edge.confidence for edge in matching.edges], dtype=np.float64)
        treatment = np.asarray(priorities[axis][source, target], dtype=np.float64)
        if (
            source.shape != (HARD_EDGES_PER_AXIS,)
            or target.shape != source.shape
            or baseline.shape != source.shape
            or treatment.shape != source.shape
            or not np.isfinite(baseline).all()
            or not np.isfinite(treatment).all()
        ):
            raise RuntimeError("hard-edge priority freeze contract changed")
        result[f"axis_{axis_index}_source"] = source
        result[f"axis_{axis_index}_target"] = target
        result[f"axis_{axis_index}_baseline_priority"] = baseline
        result[f"axis_{axis_index}_treatment_priority"] = treatment
    return result


def _edge_is_correct(
    source: np.ndarray,
    target: np.ndarray,
    *,
    axis: int,
    reference: np.ndarray,
) -> np.ndarray:
    position = np.empty(COUNT, dtype=np.int32)
    position[reference] = np.arange(COUNT, dtype=np.int32)
    source_position = position[source]
    target_position = position[target]
    if axis == 0:
        return (target_position == source_position + 1) & (
            source_position % GRID != GRID - 1
        )
    if axis == 1:
        return target_position == source_position + GRID
    raise ValueError("axis must be 0 or 1")


def _fixed_top288_correct(
    archive: Any,
    prefix: str,
    reference: np.ndarray,
    *,
    arm: str,
) -> int:
    if arm not in {"baseline", "treatment"}:
        raise ValueError("arm must be baseline or treatment")
    total = 0
    for axis in (0, 1):
        source = np.asarray(archive[f"{prefix}__axis_{axis}_source"], dtype=np.int32)
        target = np.asarray(archive[f"{prefix}__axis_{axis}_target"], dtype=np.int32)
        priority = np.asarray(
            archive[f"{prefix}__axis_{axis}_{arm}_priority"],
            dtype=np.float64,
        )
        if (
            source.shape != (HARD_EDGES_PER_AXIS,)
            or target.shape != source.shape
            or priority.shape != source.shape
            or not np.isfinite(priority).all()
        ):
            raise ValueError("frozen hard-edge arrays violate the top288 contract")
        order = np.argsort(-priority, kind="stable")[:DECODER_EDGE_BUDGET]
        truth = _edge_is_correct(source, target, axis=axis, reference=reference)
        total += int(np.count_nonzero(truth[order]))
    return total


def _delta_ci(values: list[float], *, seed: int) -> dict[str, Any]:
    difference = np.asarray(values, dtype=np.float64)
    if difference.ndim != 1 or len(difference) == 0 or not np.isfinite(difference).all():
        raise ValueError("delta CI requires one non-empty finite vector")
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


def _mean(rows: list[dict[str, Any]], arm: str, metric: str) -> float:
    return float(np.mean([float(row[arm][metric]) for row in rows]))


def run(args: argparse.Namespace) -> None:
    limit = _validate_limit(args.limit)
    if args.inference_batch <= 0:
        raise ValueError("inference-batch must be positive")
    config, config_sha = load_d2_config(args.config)
    validate_frozen_inputs(config)
    seed = int(config["protocol"]["synthetic_seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = _select_device(
        args.device,
        allow_nondeterministic_mps=args.allow_nondeterministic_mps,
    )

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    all_records, names = selected_records(config, manifest)
    records = all_records[:limit]
    prediction_path, metadata_path, report_path = _prepare_output_paths(args.output_dir)

    d1_config_path = PROJECT_ROOT / str(
        config["frozen_inputs"]["fusion_preregistration"]
    )
    d1_config, d1_config_sha = _load_config(d1_config_path)
    socket, relation, denoiser, model_metadata = _load_models(d1_config, device=device)
    fusion = _load_fusion(config, device=device)
    twin = load_fullres_twin_checkpoint(TWIN_CHECKPOINT, device=device)
    union = load_raw_twin_union_checkpoint(
        UNION_CHECKPOINT,
        config_path=UNION_CONFIG,
        selection_path=UNION_SELECTION,
        device=device,
    )
    if twin.sha256 != FROZEN_TWIN_SHA256:
        raise ValueError("Twin checkpoint identity changed")
    if union.sha256 != FROZEN_UNION_CHECKPOINT_SHA256:
        raise ValueError("Union-v2 checkpoint identity changed")

    candidate_contract = config["candidate_and_decoder"]
    priority_config = FusionUnionPriorityConfig()
    arrays: dict[str, np.ndarray] = {}
    frozen_rows: list[dict[str, Any]] = []
    cache = CleanTileCache(args.targets)
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
                raw_topk=int(
                    candidate_contract["raw_proposal_topk_per_exposed_member"]
                ),
                raw_cap=int(candidate_contract["raw_candidate_cap_per_query"]),
                union_cap=int(candidate_contract["union_candidate_cap_per_query"]),
                attach_exact_labels=False,
            )
            if board.union_labels or board.oracle_relations or board.profiles:
                raise RuntimeError("exact labels entered target-blind fusion inference")
            feature_tensor = torch.from_numpy(board.features).to(device)
            relation_tensor = torch.from_numpy(board.frozen_relation_scores).to(device)
            fusion_output = fusion(feature_tensor, relation_tensor)
            union_inference = infer_raw_twin_union_assignments(
                case.dirty_tiles,
                socket,
                twin,
                union,
                device=device,
            )
            priority = build_fullres_fusion_union_priority(
                union_inference.learned_right_log_assignment,
                union_inference.learned_down_log_assignment,
                board.union_candidates,
                fusion_output.scores,
                fusion_output.confidence_logits,
                grid=GRID,
                config=priority_config,
            )
            baseline, baseline_decoder, baseline_cyclic = _decode_layout(
                union_inference.learned_right_log_assignment,
                union_inference.learned_down_log_assignment,
                component_edge_priority=None,
            )
            treatment, treatment_decoder, treatment_cyclic = _decode_layout(
                union_inference.learned_right_log_assignment,
                union_inference.learned_down_log_assignment,
                component_edge_priority=priority.component_edge_priority,
            )
            prefix = f"case_{index:04d}"
            arrays[f"{prefix}__union_v2_layout"] = baseline
            arrays[f"{prefix}__fusion_priority_layout"] = treatment
            for key, value in _edge_arrays(
                union_inference.learned_right_log_assignment,
                union_inference.learned_down_log_assignment,
                priority.component_edge_priority,
            ).items():
                arrays[f"{prefix}__{key}"] = value
            frozen_rows.append(
                {
                    "prefix": prefix,
                    "case_id": case.case_id,
                    "source_filename": case.source_filename,
                    "priority": priority.report(),
                    "union_inference": union_inference.report(),
                    "fusion_candidate_count": len(board.union_candidates),
                    "restored_only_candidate_count": len(board.union_candidates)
                    - len(board.raw_candidates),
                    "baseline_decoder": baseline_decoder,
                    "treatment_decoder": treatment_decoder,
                    "baseline_cyclic": baseline_cyclic,
                    "treatment_cyclic": treatment_cyclic,
                    "runtime_seconds": {
                        **board.runtime_seconds,
                        "case_total": perf_counter() - case_started,
                    },
                }
            )
            print(
                json.dumps(
                    {
                        "event": "freeze",
                        "done": index + 1,
                        "total": limit,
                        "supported_union_hard_edges": sum(
                            priority.diagnostics.supported_hard_edges_per_axis.values()
                        ),
                        "case_seconds": perf_counter() - case_started,
                    }
                ),
                flush=True,
            )

    strict_count = sum(
        int(
            np.array_equal(
                np.sort(arrays[f"case_{index:04d}__union_v2_layout"]),
                np.arange(COUNT),
            )
            and np.array_equal(
                np.sort(arrays[f"case_{index:04d}__fusion_priority_layout"]),
                np.arange(COUNT),
            )
        )
        for index in range(limit)
    )
    if strict_count != limit:
        raise RuntimeError("strict original-tile layout freeze invariant failed")
    np.savez_compressed(prediction_path, **arrays)
    _write_json(
        metadata_path,
        {
            "schema": "aiijc-fullres-fusion-union-priority-opened40-predictions-v1",
            "panel_role": "reused opened D2 source40; engineering evidence only",
            "source_order": list(names[:limit]),
            "source_count": limit,
            "contains_exact_references": False,
            "contains_dirty_clean_or_restored_pixels": False,
            "contains_target_free_hard_priorities": True,
            "contains_strict_original_upright_tile_layouts": True,
            "strict_layout_count": strict_count,
            "priority_config": {
                "query_cap": priority_config.query_cap,
                "candidate_rank_cap": priority_config.candidate_rank_cap,
                "boost_scale": priority_config.boost_scale,
            },
            "rows": frozen_rows,
        },
    )
    prediction_sha = sha256_file(prediction_path)
    metadata_sha = sha256_file(metadata_path)
    print(
        json.dumps(
            {
                "event": "layouts_and_priorities_frozen_before_scoring",
                "prediction_sha256": prediction_sha,
                "metadata_sha256": metadata_sha,
            }
        ),
        flush=True,
    )

    # Only now recreate the synthetic references and score immutable artifacts.
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
            reference = _strict_layout(np.argsort(case.input_tile_to_position))
            prefix = str(frozen["prefix"])
            baseline = _strict_layout(archive[f"{prefix}__union_v2_layout"])
            treatment = _strict_layout(archive[f"{prefix}__fusion_priority_layout"])
            baseline_metrics = evaluate_layout(
                baseline,
                reference,
                reference_is_exact=True,
            ).as_dict()
            treatment_metrics = evaluate_layout(
                treatment,
                reference,
                reference_is_exact=True,
            ).as_dict()
            baseline_top288 = _fixed_top288_correct(
                archive,
                prefix,
                reference,
                arm="baseline",
            )
            treatment_top288 = _fixed_top288_correct(
                archive,
                prefix,
                reference,
                arm="treatment",
            )
            scored_rows.append(
                {
                    "source_filename": frozen["source_filename"],
                    "case_id": frozen["case_id"],
                    "union_v2": {
                        "exact_tiles": int(baseline_metrics["correct_tile_count"]),
                        "adjacency": float(baseline_metrics["adjacency"]),
                        "fixed_top288_correct": baseline_top288,
                    },
                    "fusion_priority": {
                        "exact_tiles": int(treatment_metrics["correct_tile_count"]),
                        "adjacency": float(treatment_metrics["adjacency"]),
                        "fixed_top288_correct": treatment_top288,
                    },
                }
            )

    def deltas(metric: str) -> list[float]:
        return [
            float(row["fusion_priority"][metric]) - float(row["union_v2"][metric])
            for row in scored_rows
        ]

    metrics = {
        "arms": {
            arm: {
                metric: _mean(scored_rows, arm, metric)
                for metric in ("exact_tiles", "adjacency", "fixed_top288_correct")
            }
            for arm in ("union_v2", "fusion_priority")
        },
        "exact_delta": _delta_ci(deltas("exact_tiles"), seed=20331011),
        "adjacency_delta": _delta_ci(deltas("adjacency"), seed=20331012),
        "fixed_top288_correct_delta": _delta_ci(
            deltas("fixed_top288_correct"),
            seed=20331013,
        ),
        "strict_boards": strict_count,
        "fixed_top_edge_count": FIXED_TOP_EDGE_COUNT,
        "mean_supported_union_hard_edges": float(
            np.mean(
                [
                    sum(
                        row["priority"]["diagnostics"][
                            "supported_hard_edges_per_axis"
                        ].values()
                    )
                    for row in frozen_rows
                ]
            )
        ),
        "mean_matched_fusion_contacts": float(
            np.mean(
                [
                    sum(
                        row["priority"]["diagnostics"][
                            "matched_contacts_per_axis"
                        ].values()
                    )
                    for row in frozen_rows
                ]
            )
        ),
        "mean_case_seconds": float(
            np.mean([row["runtime_seconds"]["case_total"] for row in frozen_rows])
        ),
    }
    gate = {
        "fixed_top288_gain_at_least_half_edge": float(
            metrics["fixed_top288_correct_delta"]["mean"]
        )
        >= 0.5,
        "exact_nonnegative": float(metrics["exact_delta"]["mean"]) >= 0.0,
        "adjacency_nonnegative": float(metrics["adjacency_delta"]["mean"]) >= 0.0,
        "all_strict": strict_count == limit,
    }
    gate["passed"] = all(gate.values())
    _write_json(
        report_path,
        {
            "schema": "aiijc-fullres-fusion-union-priority-opened40-report-v1",
            "status": "engineering-gate-pass" if gate["passed"] else "engineering-gate-fail",
            "panel_role": "reused opened D2 source40; engineering evidence only",
            "device": {
                "value": str(device),
                "nondeterministic_mps_explicitly_allowed": bool(
                    args.allow_nondeterministic_mps
                ),
                "determinism_claimed": device.type != "mps",
            },
            "frozen_inputs": {
                "d2_config_sha256": config_sha,
                "d1_fusion_config_sha256": d1_config_sha,
                "loaded_fusion_models": model_metadata,
                "twin_checkpoint_sha256": twin.sha256,
                "union_checkpoint_sha256": union.sha256,
            },
            "predictions": {
                "path": str(prediction_path.relative_to(PROJECT_ROOT)),
                "sha256": prediction_sha,
                "metadata_path": str(metadata_path.relative_to(PROJECT_ROOT)),
                "metadata_sha256": metadata_sha,
                "layouts_and_priorities_frozen_before_reference_recreation": True,
                "contains_exact_references": False,
            },
            "metrics": metrics,
            "gate": gate,
            "rows": scored_rows,
            "runtime_seconds": perf_counter() - started,
            "organizer_holdout_or_test_opened": False,
            "restored_pixels_matcher_only": True,
            "restored_pixels_emitted": False,
            "new_hard_edges_introduced": False,
            "original_upright_tile_permutations_only": True,
            "weight_or_budget_sweep": False,
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

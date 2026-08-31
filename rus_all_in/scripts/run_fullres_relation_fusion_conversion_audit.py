#!/usr/bin/env python3
"""Post-hoc target-assisted conversion audit on the already-opened D2 source40.

This script is diagnostic-only.  It reuses exactly the opened D2 roster and
decomposes candidate supply, learned query selection, hard matching, component
geometry/packing, and cyclic origin.  Oracle arms consume exact synthetic truth
and therefore can never be deployed or used as a submission method.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from aiijc_puzzle import socket_decoder
from aiijc_puzzle.component_relation_confidence import (
    QueryConfidenceFeatures,
    relation_forest_score_substitution,
)
from aiijc_puzzle.component_relation_reranker import DIRECTION_TO_INDEX
from aiijc_puzzle.fullres_relation_decoder import build_fusion_forest_inputs
from aiijc_puzzle.layout_evaluation import evaluate_layout
from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.socket_decoder import (
    SocketDecoderConfig,
    decode_socket_assignments,
    hard_partial_axis_matching,
    prioritise_component_edges,
)
from aiijc_puzzle.socket_translation_placer import (
    CyclicTranslationConfig,
    select_global_cyclic_translation,
)

try:
    from scripts.run_component_relation_reranker import CleanTileCache, prepare_case
    from scripts.run_fullres_relation_fusion import _load_config, _load_models, prepare_fusion_board
    from scripts.run_fullres_relation_fusion_decoder_d2 import (
        DEFAULT_CONFIG,
        DEFAULT_MANIFEST,
        DEFAULT_TARGETS,
        GRID,
        PROJECT_ROOT,
        _load_fusion,
        load_d2_config,
        selected_records,
    )
except ModuleNotFoundError:  # Direct ``python scripts/run_*.py`` execution.
    from run_component_relation_reranker import CleanTileCache, prepare_case
    from run_fullres_relation_fusion import _load_config, _load_models, prepare_fusion_board
    from run_fullres_relation_fusion_decoder_d2 import (
        DEFAULT_CONFIG,
        DEFAULT_MANIFEST,
        DEFAULT_TARGETS,
        GRID,
        PROJECT_ROOT,
        _load_fusion,
        load_d2_config,
        selected_records,
    )

TILE_COUNT = GRID * GRID
DEFAULT_D2_OUTPUT = PROJECT_ROOT / "outputs/fullres-relation-fusion/decoder-d2-source40-draw1"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs/fullres-relation-fusion/conversion-audit-opened-source40-v2"
)
ARMS = ("baseline", "learned_forest", "oracle_hard_priority", "oracle_forest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--d2-output-dir", type=Path, default=DEFAULT_D2_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--allow-nondeterministic-mps", action="store_true")
    parser.add_argument("--inference-batch", type=int, default=576)
    parser.add_argument("--log-every", type=int, default=1)
    return parser.parse_args()


def _strict(value: Any) -> np.ndarray:
    layout = np.asarray(value, dtype=np.int32)
    if layout.shape != (TILE_COUNT,) or not np.array_equal(
        np.sort(layout), np.arange(TILE_COUNT)
    ):
        raise ValueError("layout is not a strict original-tile permutation")
    return np.ascontiguousarray(layout)


def _positions(layout: Any) -> np.ndarray:
    value = _strict(layout)
    result = np.empty((TILE_COUNT, 2), dtype=np.int32)
    result[value, 0], result[value, 1] = divmod(np.arange(TILE_COUNT), GRID)
    return result


def _edge_correct(axis: str, source: int, target: int, tile_to_position: np.ndarray) -> bool:
    source_row, source_column = divmod(int(tile_to_position[source]), GRID)
    target_row, target_column = divmod(int(tile_to_position[target]), GRID)
    return bool(
        (axis == "right" and target_row == source_row and target_column == source_column + 1)
        or (axis == "down" and target_row == source_row + 1 and target_column == source_column)
    )


def hard_edge_keys(right: Any, down: Any) -> set[tuple[str, int, int]]:
    result: set[tuple[str, int, int]] = set()
    for axis, value in (("right", right), ("down", down)):
        matching = hard_partial_axis_matching(value, grid=GRID, axis=axis)
        result.update((axis, edge.source, edge.target) for edge in matching.edges)
    return result


def edge_stats(keys: set[tuple[str, int, int]], tile_to_position: np.ndarray) -> dict[str, Any]:
    correct = sum(
        _edge_correct(axis, source, target, tile_to_position)
        for axis, source, target in keys
    )
    return {
        "edge_count": len(keys),
        "correct_edges": correct,
        "precision": correct / len(keys) if keys else None,
    }


def oracle_priority(tile_to_position: np.ndarray) -> dict[str, np.ndarray]:
    priority = {
        "right": np.zeros((TILE_COUNT, TILE_COUNT), dtype=np.float64),
        "down": np.zeros((TILE_COUNT, TILE_COUNT), dtype=np.float64),
    }
    for axis in priority:
        for source in range(TILE_COUNT):
            for target in range(TILE_COUNT):
                priority[axis][source, target] = float(
                    _edge_correct(axis, source, target, tile_to_position)
                )
    return priority


def build_components(
    right: Any,
    down: Any,
    *,
    budget: int,
    priority: Mapping[str, Any] | None = None,
) -> tuple[dict[int, tuple[int, int]], ...]:
    right_matching = hard_partial_axis_matching(right, grid=GRID, axis="right")
    down_matching = hard_partial_axis_matching(down, grid=GRID, axis="down")
    edges = prioritise_component_edges(
        right_matching,
        down_matching,
        edge_budget_per_axis=budget,
        tile_count=TILE_COUNT,
        component_edge_priority=priority,
    )
    builder = socket_decoder._TranslationComponents(count=TILE_COUNT, grid=GRID)  # noqa: SLF001
    for edge in edges:
        builder.add(edge)
    components = builder.complete_components()
    return tuple(sorted(components, key=lambda value: (-len(value), min(value))))


def component_stats(
    components: Sequence[Mapping[int, tuple[int, int]]],
    tile_to_position: np.ndarray,
    layout: Any,
) -> dict[str, Any]:
    true_positions = np.column_stack(divmod(tile_to_position, GRID)).astype(np.int32)
    predicted_positions = _positions(layout)
    weighted_support = 0
    exact_component_tiles = 0
    anchor_correct_support = 0
    preserved_support = 0
    pair_correct = 0
    pair_total = 0
    purities: list[float] = []
    sizes: list[int] = []
    largest_purity = 0.0
    for index, component in enumerate(components):
        sizes.append(len(component))
        true_shifts = Counter(
            (
                int(true_positions[tile, 0]) - coordinate[0],
                int(true_positions[tile, 1]) - coordinate[1],
            )
            for tile, coordinate in component.items()
        )
        predicted_shifts = Counter(
            (
                int(predicted_positions[tile, 0]) - coordinate[0],
                int(predicted_positions[tile, 1]) - coordinate[1],
            )
            for tile, coordinate in component.items()
        )
        true_shift, true_support = min(true_shifts.items(), key=lambda item: (-item[1], item[0]))
        predicted_shift, predicted_support = min(
            predicted_shifts.items(), key=lambda item: (-item[1], item[0])
        )
        purity = true_support / len(component)
        purities.append(purity)
        weighted_support += true_support
        preserved_support += predicted_support
        if true_support == len(component):
            exact_component_tiles += len(component)
        if predicted_shift == true_shift:
            anchor_correct_support += true_support
        pair_correct += sum(value * (value - 1) // 2 for value in true_shifts.values())
        pair_total += len(component) * (len(component) - 1) // 2
        if index == 0:
            largest_purity = purity
    return {
        "component_count": len(components),
        "largest_component": max(sizes),
        "largest_component_truth_purity": largest_purity,
        "mean_component_size": float(np.mean(sizes)),
        "mean_unweighted_truth_purity": float(np.mean(purities)),
        "tile_weighted_truth_translation_purity": weighted_support / TILE_COUNT,
        "pairwise_relative_accuracy": pair_correct / pair_total if pair_total else 1.0,
        "tiles_in_internally_exact_components": exact_component_tiles,
        "predicted_layout_component_preservation": preserved_support / TILE_COUNT,
        "truth_mode_anchor_correct_support": anchor_correct_support,
        "truth_mode_anchor_correct_fraction": anchor_correct_support / TILE_COUNT,
    }


def oracle_best_cyclic(layout: Any, reference: Any) -> dict[str, Any]:
    initial = _strict(layout).reshape(GRID, GRID)
    best: tuple[int, int, int, int, np.ndarray, dict[str, Any]] | None = None
    for row_roll in range(GRID):
        for column_roll in range(GRID):
            candidate = np.roll(initial, shift=(row_roll, column_roll), axis=(0, 1)).reshape(-1)
            metrics = evaluate_layout(candidate, reference, reference_is_exact=True).as_dict()
            key = (
                int(metrics["correct_tile_count"]),
                int(metrics["adjacency_correct"]),
                -row_roll,
                -column_roll,
            )
            if best is None or key > best[:4]:
                best = (*key, np.ascontiguousarray(candidate), metrics)
    if best is None:
        raise RuntimeError("oracle cyclic search returned no layout")
    layout_value = best[4]
    return {
        "row_roll": -best[2],
        "column_roll": -best[3],
        "layout_sha256": hashlib.sha256(layout_value.astype("<i4").tobytes()).hexdigest(),
        "metrics": best[5],
        "target_assisted_not_deployable": True,
    }


def oracle_forest_inputs(
    board: Any,
    fusion_scores: np.ndarray,
) -> tuple[tuple[QueryConfidenceFeatures, ...], np.ndarray]:
    grouped: dict[tuple[int, str], list[int]] = defaultdict(list)
    for index, candidate in enumerate(board.union_candidates):
        if board.union_labels[index].positive:
            grouped[candidate.query_key].append(index)
    rows: list[QueryConfidenceFeatures] = []
    correct_precision: list[float] = []
    correct_count: list[int] = []
    for query in sorted(grouped, key=lambda item: (item[0], DIRECTION_TO_INDEX[item[1]])):
        positives = sorted(
            grouped[query],
            key=lambda index: (
                -board.union_labels[index].correct_contacts
                / board.union_labels[index].contact_count,
                -board.union_labels[index].correct_contacts,
                -float(fusion_scores[index]),
                board.union_candidates[index].relation_key,
            ),
        )
        winner = positives[0]
        rows.append(
            QueryConfidenceFeatures(
                board_id=board.case_id,
                source_component=query[0],
                direction=query[1],
                learned_top_candidate=winner,
                raw_top_candidate=winner,
                learned_margin=0.0,
                raw_margin=0.0,
                values=(),
            )
        )
        label = board.union_labels[winner]
        correct_precision.append(label.correct_contacts / label.contact_count)
        correct_count.append(label.correct_contacts)
    maximum = max(correct_count, default=1)
    probabilities = np.asarray(
        [
            0.5 + 0.4 * precision + 0.09 * count / maximum
            for precision, count in zip(correct_precision, correct_count, strict=True)
        ],
        dtype=np.float64,
    )
    return tuple(rows), probabilities


def score_layout(layout: Any, reference: Any) -> dict[str, Any]:
    return evaluate_layout(_strict(layout), reference, reference_is_exact=True).as_dict()


def aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in ARMS:
        result[arm] = {}
        for stage in ("precyclic", "dirty_cyclic", "oracle_cyclic"):
            result[arm][stage] = {
                field: float(np.mean([row["arms"][arm][stage]["metrics"][field] for row in rows]))
                for field in (
                    "correct_tile_count",
                    "translation_aligned_count",
                    "adjacency_correct",
                    "adjacency",
                )
            }
    supply_fields = (
        "oracle_query_count",
        "oracle_relation_count",
        "raw_supplied_query_count",
        "raw_supplied_relation_count",
        "union_supplied_query_count",
        "union_supplied_relation_count",
        "candidate_query_count",
        "learned_top1_correct_queries",
        "learned_top8_correct_queries",
        "learned_top8_restored_only_selected_queries",
        "learned_top8_restored_only_correct_queries",
    )
    result["supply_and_rank_mean_per_board"] = {
        field: float(np.mean([row["supply_and_rank"][field] for row in rows]))
        for field in supply_fields
    }
    for arm in ARMS:
        result[arm]["component_mean"] = {
            field: float(np.mean([row["arms"][arm]["components"][field] for row in rows]))
            for field in (
                "component_count",
                "largest_component",
                "largest_component_truth_purity",
                "tile_weighted_truth_translation_purity",
                "pairwise_relative_accuracy",
                "tiles_in_internally_exact_components",
                "predicted_layout_component_preservation",
                "truth_mode_anchor_correct_support",
            )
        }
        result[arm]["hard_matching_mean"] = {
            field: float(
                np.mean([row["arms"][arm]["hard_matching"][field] for row in rows])
            )
            for field in ("edge_count", "correct_edges", "precision")
        }
    result["forest_mean_per_board"] = {
        method: {
            field: float(np.mean([row[method][field] for row in rows]))
            for field in (
                "selected_queries",
                "accepted_relations",
                "accepted_contacts",
                "new_contacts_absent_from_original_hard_matching",
                "accepted_contacts_surviving_new_hard_matching",
            )
        }
        for method in ("learned_forest", "oracle_forest")
    }
    result["hard_matching_delta_mean_per_board"] = {
        kind: {
            field: float(
                np.mean([row["hard_matching_delta"][kind][field] for row in rows])
            )
            for field in ("edge_count", "correct_edges")
        }
        for kind in ("learned_new", "learned_removed", "oracle_new", "oracle_removed")
    }
    return result


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.inference_batch <= 0 or args.log_every <= 0:
        raise ValueError("inference-batch and log-every must be positive")
    config, config_sha256 = load_d2_config(args.config)
    d2_report_path = args.d2_output_dir / "report.json"
    d2_frozen_path = args.d2_output_dir / "frozen_predictions.json"
    d2_report = json.loads(d2_report_path.read_text(encoding="utf-8"))
    if d2_report["preregistration"]["sha256"] != config_sha256:
        raise ValueError("D2 report belongs to another config")
    if sha256_file(d2_frozen_path) != d2_report["layout_freeze"]["sha256"]:
        raise ValueError("D2 frozen layouts changed")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records, names = selected_records(config, manifest)
    if list(names) != list(d2_report["selection"]["source_filenames"]):
        raise ValueError("audit roster differs from opened D2 source40")

    if args.device == "mps":
        if not args.allow_nondeterministic_mps or not torch.backends.mps.is_available():
            raise ValueError("MPS requires available backend and explicit acknowledgement")
        torch.use_deterministic_algorithms(False)
        device = torch.device("mps")
    else:
        if args.allow_nondeterministic_mps:
            raise ValueError("MPS acknowledgement supplied for CPU")
        torch.use_deterministic_algorithms(True)
        device = torch.device("cpu")
    d1_config, _ = _load_config(
        PROJECT_ROOT / str(config["frozen_inputs"]["fusion_preregistration"])
    )
    socket, relation, denoiser, _ = _load_models(d1_config, device=device)
    fusion = _load_fusion(config, device=device)
    contract = config["candidate_and_decoder"]
    budget = int(contract["component_edge_budget_per_axis"])
    decoder_config = SocketDecoderConfig(
        component_edge_budget_per_axis=budget,
        max_swap_steps=int(contract["decoder_max_swap_steps"]),
    )
    cyclic_config = CyclicTranslationConfig(border_weight=float(contract["cyclic_border_weight"]))
    cache = CleanTileCache(args.targets)
    rows: list[dict[str, Any]] = []
    started = perf_counter()
    with torch.inference_mode():
        for index, record in enumerate(records):
            case = prepare_case(
                cache,
                record,
                draw_index=int(config["protocol"]["draw_index"]),
                seed=int(config["protocol"]["synthetic_seed"]),
            )
            board = prepare_fusion_board(
                case,
                socket=socket,
                relation=relation,
                denoiser=denoiser,
                device=device,
                inference_batch=args.inference_batch,
                raw_topk=int(contract["raw_proposal_topk_per_exposed_member"]),
                raw_cap=int(contract["raw_candidate_cap_per_query"]),
                union_cap=int(contract["union_candidate_cap_per_query"]),
                attach_exact_labels=True,
            )
            output = fusion(
                torch.from_numpy(board.features).to(device),
                torch.from_numpy(board.frozen_relation_scores).to(device),
            )
            fusion_scores = output.scores.float().cpu().numpy()
            adapter = build_fusion_forest_inputs(
                board.union_candidates,
                output.scores,
                output.confidence_logits,
                raw_candidate_keys=frozenset(
                    candidate.relation_key for candidate in board.raw_candidates
                ),
                board_id=case.case_id,
            )
            selected = sorted(
                range(len(adapter.rows)),
                key=lambda item: (
                    -float(adapter.probabilities[item]),
                    adapter.rows[item].source_component,
                    DIRECTION_TO_INDEX[adapter.rows[item].direction],
                ),
            )[: int(contract["forest_top_query_cap"])]
            learned_top1_correct = sum(
                board.union_labels[row.learned_top_candidate].positive for row in adapter.rows
            )
            learned_top8_correct = sum(
                board.union_labels[adapter.rows[item].learned_top_candidate].positive
                for item in selected
            )
            raw_keys = {candidate.relation_key for candidate in board.raw_candidates}
            learned_top8_restored_correct = sum(
                board.union_labels[adapter.rows[item].learned_top_candidate].positive
                and board.union_candidates[adapter.rows[item].learned_top_candidate].relation_key
                not in raw_keys
                for item in selected
            )
            learned_top8_restored_selected = sum(
                board.union_candidates[adapter.rows[item].learned_top_candidate].relation_key
                not in raw_keys
                for item in selected
            )
            oracle_queries = {
                (relation_key[0], relation_key[1])
                for relation_key in board.oracle_relations
            }
            raw_supplied_relations = raw_keys & board.oracle_relations
            raw_supplied_queries = {
                (relation_key[0], relation_key[1])
                for relation_key in raw_supplied_relations
            }
            union_keys = {candidate.relation_key for candidate in board.union_candidates}
            union_supplied_relations = union_keys & board.oracle_relations
            union_supplied_queries = {
                (relation_key[0], relation_key[1])
                for relation_key in union_supplied_relations
            }

            raw = board.raw_socket_output
            learned_matrices, learned_forest_diagnostics = relation_forest_score_substitution(
                raw.right_log_assignment,
                raw.down_log_assignment,
                adapter.rows,
                adapter.probabilities,
                board.union_candidates,
                grid=GRID,
                top_cap=int(contract["forest_top_query_cap"]),
                component_edge_budget_per_axis=budget,
            )
            oracle_rows, oracle_probabilities = oracle_forest_inputs(board, fusion_scores)
            oracle_matrices, oracle_forest_diagnostics = relation_forest_score_substitution(
                raw.right_log_assignment,
                raw.down_log_assignment,
                oracle_rows,
                oracle_probabilities,
                board.union_candidates,
                grid=GRID,
                top_cap=int(contract["forest_top_query_cap"]),
                component_edge_budget_per_axis=budget,
            )
            exact_priority = oracle_priority(case.input_tile_to_position)
            decodes = {
                "baseline": decode_socket_assignments(
                    raw.right_log_assignment,
                    raw.down_log_assignment,
                    grid=GRID,
                    config=decoder_config,
                ),
                "learned_forest": decode_socket_assignments(
                    learned_matrices["right"],
                    learned_matrices["down"],
                    grid=GRID,
                    config=decoder_config,
                ),
                "oracle_hard_priority": decode_socket_assignments(
                    raw.right_log_assignment,
                    raw.down_log_assignment,
                    grid=GRID,
                    config=decoder_config,
                    component_edge_priority=exact_priority,
                ),
                "oracle_forest": decode_socket_assignments(
                    oracle_matrices["right"],
                    oracle_matrices["down"],
                    grid=GRID,
                    config=decoder_config,
                ),
            }
            matrices = {
                "baseline": (raw.right_log_assignment, raw.down_log_assignment, None),
                "learned_forest": (
                    learned_matrices["right"],
                    learned_matrices["down"],
                    None,
                ),
                "oracle_hard_priority": (
                    raw.right_log_assignment,
                    raw.down_log_assignment,
                    exact_priority,
                ),
                "oracle_forest": (
                    oracle_matrices["right"],
                    oracle_matrices["down"],
                    None,
                ),
            }
            reference = np.argsort(case.input_tile_to_position).astype(np.int32)
            arm_rows: dict[str, Any] = {}
            for arm in ARMS:
                right, down, priority = matrices[arm]
                precyclic = _strict(decodes[arm].layout)
                dirty_cyclic = select_global_cyclic_translation(
                    precyclic,
                    raw.right_log_assignment,
                    raw.down_log_assignment,
                    grid=GRID,
                    config=cyclic_config,
                ).layout
                components = build_components(
                    right, down, budget=budget, priority=priority
                )
                arm_rows[arm] = {
                    "precyclic": {"metrics": score_layout(precyclic, reference)},
                    "dirty_cyclic": {"metrics": score_layout(dirty_cyclic, reference)},
                    "oracle_cyclic": oracle_best_cyclic(precyclic, reference),
                    "components": component_stats(
                        components, case.input_tile_to_position, precyclic
                    ),
                    "hard_matching": edge_stats(
                        hard_edge_keys(right, down), case.input_tile_to_position
                    ),
                    "decoder": decodes[arm].report(),
                }
            baseline_keys = hard_edge_keys(
                raw.right_log_assignment, raw.down_log_assignment
            )
            learned_keys = hard_edge_keys(
                learned_matrices["right"], learned_matrices["down"]
            )
            oracle_keys = hard_edge_keys(
                oracle_matrices["right"], oracle_matrices["down"]
            )
            rows.append(
                {
                    "source_filename": case.source_filename,
                    "case_id": case.case_id,
                    "supply_and_rank": {
                        "oracle_query_count": len(oracle_queries),
                        "oracle_relation_count": len(board.oracle_relations),
                        "raw_supplied_query_count": len(raw_supplied_queries),
                        "raw_supplied_relation_count": len(raw_supplied_relations),
                        "union_supplied_query_count": len(union_supplied_queries),
                        "union_supplied_relation_count": len(union_supplied_relations),
                        "candidate_query_count": len(adapter.rows),
                        "learned_top1_correct_queries": learned_top1_correct,
                        "learned_top8_correct_queries": learned_top8_correct,
                        "learned_top8_restored_only_selected_queries": (
                            learned_top8_restored_selected
                        ),
                        "learned_top8_restored_only_correct_queries": learned_top8_restored_correct,
                        "union_candidate_count": len(board.union_candidates),
                        "restored_only_candidate_count": len(board.union_candidates)
                        - len(board.raw_candidates),
                    },
                    "learned_forest": learned_forest_diagnostics,
                    "oracle_forest": oracle_forest_diagnostics,
                    "hard_matching_delta": {
                        "learned_new": edge_stats(
                            learned_keys - baseline_keys, case.input_tile_to_position
                        ),
                        "learned_removed": edge_stats(
                            baseline_keys - learned_keys, case.input_tile_to_position
                        ),
                        "oracle_new": edge_stats(
                            oracle_keys - baseline_keys, case.input_tile_to_position
                        ),
                        "oracle_removed": edge_stats(
                            baseline_keys - oracle_keys, case.input_tile_to_position
                        ),
                    },
                    "arms": arm_rows,
                }
            )
            if (index + 1) % args.log_every == 0 or index + 1 == len(records):
                print(
                    json.dumps(
                        {
                            "event": "audit",
                            "index": index + 1,
                            "sources": len(records),
                            "case_id": case.case_id,
                            "elapsed_seconds": perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
    report = {
        "experiment": "fullres-relation-fusion-conversion-audit-opened-source40-v2",
        "status": "post-hoc-target-assisted-diagnostic-only",
        "deployable": False,
        "new_target_panel_opened": False,
        "competition_test_opened": False,
        "source_roster": {
            "identical_to_opened_d2": True,
            "source_count": len(names),
            "source_order_digest": config["selection"]["source_order_digest"],
        },
        "lineage": {
            "d2_config_sha256": config_sha256,
            "d2_report": str(d2_report_path.resolve()),
            "d2_report_sha256": sha256_file(d2_report_path),
            "d2_frozen_predictions_sha256": sha256_file(d2_frozen_path),
        },
        "oracle_contract": (
            "Exact synthetic labels are used for analysis, correct hard-edge priority, "
            "correct supplied relation selection, and best cyclic roll. No oracle arm "
            "is an inference method or promotion candidate."
        ),
        "summary": aggregate(rows),
        "runtime_seconds": perf_counter() - started,
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    if report_path.exists():
        raise FileExistsError("refusing to overwrite conversion audit")
    _write_json(report_path, report)
    print(
        json.dumps(
            {
                "event": "complete",
                "report": str(report_path.resolve()),
                "report_sha256": sha256_file(report_path),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

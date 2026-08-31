#!/usr/bin/env python3
"""Replay one frozen learned-priority no-QAP arm on opened fresh64.

This is a bounded engineering replay on the already-opened rank-delta source64
roster.  It is not fresh promotion evidence.  ``freeze`` produces all
target-free priorities and strict layouts before ``score`` can recreate exact
synthetic references.  The treatment changes exactly one decoder constant:
the learned-priority arm uses ``max_swap_steps=0`` instead of the frozen
standard value 24.  Both arms use the same learned top-144 priority, immutable
Union hard identities, component budget 144 and cyclic-border weight 5.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from aiijc_puzzle.direct_hard_edge_production import CYCLIC_BORDER_WEIGHT
from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.socket_decoder import SocketDecoderConfig, decode_socket_assignments
from aiijc_puzzle.socket_sorter_production import (
    DECODER_EDGE_BUDGET,
    DECODER_SWAP_STEPS,
)
from aiijc_puzzle.socket_translation_placer import (
    CyclicTranslationConfig,
    select_global_cyclic_translation,
)
from aiijc_puzzle.synthetic_socket_evaluation import names_digest
from aiijc_puzzle.union_hard_edge_priority import union_hard_edge_priority_matrices

try:
    import scripts.run_learned_membership_rank_delta_composition_opened64 as upstream
except ModuleNotFoundError:  # Direct ``python scripts/run_*.py`` execution.
    import run_learned_membership_rank_delta_composition_opened64 as upstream


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRID = upstream.GRID
COUNT = upstream.COUNT
HARD_EDGES_PER_AXIS = upstream.HARD_EDGES_PER_AXIS
HARD_EDGE_COUNT = upstream.HARD_EDGE_COUNT
EXPECTED_SOURCES = upstream.EXPECTED_SOURCES
BOOTSTRAP_SEED = 306_947_117
BOOTSTRAP_RESAMPLES = upstream.BOOTSTRAP_RESAMPLES
STANDARD_SWAP_STEPS = DECODER_SWAP_STEPS
NO_QAP_SWAP_STEPS = 0
EXPECTED_STANDARD_SWAP_STEPS = 24
EXPECTED_EDGE_BUDGET = 144
EXPECTED_CYCLIC_BORDER_WEIGHT = 5.0

RANK_CONFIG = upstream.RANK_CONFIG
LEARNED_CONFIG = upstream.LEARNED_PILOT_CONFIG
LEARNED_OUTPUT = upstream.LEARNED_OUTPUT
DEFAULT_MANIFEST = upstream.DEFAULT_MANIFEST
DEFAULT_TARGETS = upstream.DEFAULT_TARGETS
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/learned-no-qap/opened-fresh64-v1"
UPSTREAM_RUNNER = Path(upstream.__file__).resolve()

ARM_NAMES = (
    "union_v2",
    "rank_delta_transfer",
    "learned_standard",
    "learned_no_qap",
)
METRIC_NAMES = (
    "exact_tiles",
    "adjacency",
    "satisfied_pairs",
    "fixed_top288_correct",
)


@dataclass(frozen=True)
class ReplayPaths:
    predictions: Path
    metadata: Path
    freeze_commitment: Path
    report: Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "score"))
    parser.add_argument("--rank-config", type=Path, default=RANK_CONFIG)
    parser.add_argument("--learned-config", type=Path, default=LEARNED_CONFIG)
    parser.add_argument("--learned-output", type=Path, default=LEARNED_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--allow-nondeterministic-mps", action="store_true")
    parser.add_argument("--inference-batch", type=int, default=upstream.INFERENCE_BATCH)
    parser.add_argument("--limit", type=int, default=EXPECTED_SOURCES)
    return parser.parse_args(argv)


def _paths(output_dir: Path) -> ReplayPaths:
    root = output_dir.resolve()
    return ReplayPaths(
        predictions=root / "frozen-target-free-priorities-layouts.npz",
        metadata=root / "frozen-target-free-priorities-layouts.json",
        freeze_commitment=root / "freeze-commitment.json",
        report=root / "report.json",
    )


def _report_path(path: Path) -> str:
    return upstream._report_path(path)


def _runtime_input_records(
    *,
    rank_config: Path,
    learned_config: Path,
    learned_output: Path,
    manifest: Path,
) -> dict[str, dict[str, str]]:
    paths = {
        "runner": Path(__file__).resolve(),
        "upstream_composition_runner": UPSTREAM_RUNNER,
        "rank_confirmation_config": rank_config.resolve(),
        "learned_priority_config": learned_config.resolve(),
        "learned_priority_commitment": learned_output.resolve()
        / "selection-commitment.json",
        "learned_priority_checkpoint": learned_output.resolve()
        / "union-hard-edge-priority.pt",
        "manifest": manifest.resolve(),
        "socket_decoder": PROJECT_ROOT / "src/aiijc_puzzle/socket_decoder.py",
        "cyclic_translation": PROJECT_ROOT
        / "src/aiijc_puzzle/socket_translation_placer.py",
        "learned_priority_implementation": PROJECT_ROOT
        / "src/aiijc_puzzle/union_hard_edge_priority.py",
    }
    return {
        name: {"path": _report_path(path), "sha256": sha256_file(path)}
        for name, path in paths.items()
    }


def _strict_layout(value: Any) -> np.ndarray:
    return upstream._strict_layout(value)


def _validate_decoder_contract() -> None:
    observed = {
        "standard_swap_steps": STANDARD_SWAP_STEPS,
        "edge_budget_per_axis": DECODER_EDGE_BUDGET,
        "cyclic_border_weight": CYCLIC_BORDER_WEIGHT,
    }
    expected = {
        "standard_swap_steps": EXPECTED_STANDARD_SWAP_STEPS,
        "edge_budget_per_axis": EXPECTED_EDGE_BUDGET,
        "cyclic_border_weight": EXPECTED_CYCLIC_BORDER_WEIGHT,
    }
    if observed != expected:
        raise ValueError(f"frozen learned no-QAP decoder contract changed: {observed}")


def _decode_learned_no_qap(
    right: np.ndarray,
    down: np.ndarray,
    *,
    component_edge_priority: Mapping[str, Any],
) -> np.ndarray:
    """Decode learned top-144 components with QAP swaps disabled."""

    decoder = decode_socket_assignments(
        right,
        down,
        grid=GRID,
        config=SocketDecoderConfig(
            component_edge_budget_per_axis=DECODER_EDGE_BUDGET,
            swap_edge_budget_per_axis=DECODER_EDGE_BUDGET,
            max_swap_steps=NO_QAP_SWAP_STEPS,
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
    return _strict_layout(cyclic.layout)


def _decoder_order(
    source: np.ndarray,
    target: np.ndarray,
    base_priority: np.ndarray,
    priority: np.ndarray,
) -> np.ndarray:
    return np.lexsort((target, source, -base_priority, -priority))


def _layout_positions(layout: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    position = np.empty(COUNT, dtype=np.int32)
    position[_strict_layout(layout)] = np.arange(COUNT, dtype=np.int32)
    return np.divmod(position, GRID)


def _own_top288_satisfied(
    source: np.ndarray,
    target: np.ndarray,
    axis: np.ndarray,
    base_priority: np.ndarray,
    priority: np.ndarray,
    layout: np.ndarray,
) -> int:
    """Count an arm's own selected hard edges realised by its target-free layout."""

    row, column = _layout_positions(layout)
    total = 0
    for axis_index in (0, 1):
        selected = np.flatnonzero(axis == axis_index)
        if len(selected) != HARD_EDGES_PER_AXIS:
            raise ValueError("hard-edge axis cardinality changed")
        order = _decoder_order(
            source[selected],
            target[selected],
            base_priority[selected],
            priority[selected],
        )[:DECODER_EDGE_BUDGET]
        edge_indices = selected[order]
        edge_source = source[edge_indices]
        edge_target = target[edge_indices]
        if axis_index == 0:
            realised = (row[edge_target] == row[edge_source]) & (
                column[edge_target] == column[edge_source] + 1
            )
        else:
            realised = (column[edge_target] == column[edge_source]) & (
                row[edge_target] == row[edge_source] + 1
            )
        total += int(np.count_nonzero(realised))
    return total


def _edge_truth(
    source: np.ndarray,
    target: np.ndarray,
    *,
    axis: int,
    reference: np.ndarray,
) -> np.ndarray:
    position = np.empty(COUNT, dtype=np.int32)
    position[_strict_layout(reference)] = np.arange(COUNT, dtype=np.int32)
    source_position = position[source]
    target_position = position[target]
    if axis == 0:
        return (target_position == source_position + 1) & (
            source_position % GRID != GRID - 1
        )
    if axis == 1:
        return target_position == source_position + GRID
    raise ValueError("axis must be zero or one")


def _fixed_top288_correct(
    archive: Mapping[str, Any],
    prefix: str,
    reference: np.ndarray,
    *,
    arm: str,
) -> int:
    if arm not in ARM_NAMES:
        raise ValueError(f"unknown no-QAP replay arm: {arm}")
    total = 0
    for axis in (0, 1):
        source = np.asarray(archive[f"{prefix}__axis_{axis}_source"], dtype=np.int32)
        target = np.asarray(archive[f"{prefix}__axis_{axis}_target"], dtype=np.int32)
        base = np.asarray(
            archive[f"{prefix}__axis_{axis}_union_v2_priority"],
            dtype=np.float64,
        )
        priority = np.asarray(
            archive[f"{prefix}__axis_{axis}_{arm}_priority"],
            dtype=np.float64,
        )
        if (
            source.shape != (HARD_EDGES_PER_AXIS,)
            or target.shape != source.shape
            or base.shape != source.shape
            or priority.shape != source.shape
            or not np.isfinite(base).all()
            or not np.isfinite(priority).all()
        ):
            raise ValueError("frozen arrays violate the fixed-top288 contract")
        order = _decoder_order(source, target, base, priority)[:DECODER_EDGE_BUDGET]
        total += int(
            np.count_nonzero(
                _edge_truth(source, target, axis=axis, reference=reference)[order]
            )
        )
    return total


def _satisfied_pairs(adjacency: float) -> int:
    value = float(adjacency) * HARD_EDGE_COUNT
    rounded = int(round(value))
    if not np.isfinite(value) or abs(value - rounded) > 1e-9:
        raise ValueError("adjacency is not an exact satisfied-pair fraction")
    return rounded


def _win_tie_loss(values: Sequence[float]) -> dict[str, int]:
    return upstream._win_tie_loss(values)


def _comparison(
    rows: Sequence[Mapping[str, Any]],
    *,
    treatment: str,
    baseline: str,
    seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for offset, metric in enumerate(METRIC_NAMES):
        values = [
            float(row[treatment][metric]) - float(row[baseline][metric])
            for row in rows
        ]
        result[f"{metric}_delta"] = upstream.source_clustered_ci(
            values,
            seed=seed + offset,
            resamples=BOOTSTRAP_RESAMPLES,
        )
        result[f"{metric}_win_tie_loss"] = _win_tie_loss(values)
    return result


def evaluate_gate(
    metrics: Mapping[str, Any],
    *,
    strict_layouts: int,
    case_count: int,
) -> dict[str, Any]:
    versus_standard = metrics["learned_no_qap_vs_learned_standard"]
    versus_rank = metrics["learned_no_qap_vs_rank_delta_transfer"]
    exact = float(versus_standard["exact_tiles_delta"]["mean"])
    adjacency = float(versus_standard["adjacency_delta"]["mean"])
    pairs = float(versus_standard["satisfied_pairs_delta"]["mean"])
    rank_exact = float(versus_rank["exact_tiles_delta"]["mean"])
    checks = {
        "satisfied_pairs_strictly_positive_vs_learned_standard": {
            "observed": pairs,
            "required": ">0",
            "pass": pairs > 0.0,
        },
        "adjacency_strictly_positive_vs_learned_standard": {
            "observed": adjacency,
            "required": ">0",
            "pass": adjacency > 0.0,
        },
        "exact_nonnegative_vs_learned_standard": {
            "observed": exact,
            "required": ">=0",
            "pass": exact >= 0.0,
        },
        "secondary_exact_nonnegative_vs_rank_delta": {
            "observed": rank_exact,
            "required": ">=0",
            "pass": rank_exact >= 0.0,
        },
        "all_four_arms_strict": {
            "observed": strict_layouts,
            "required": 4 * case_count,
            "pass": strict_layouts == 4 * case_count,
        },
        "complete_opened_fresh64_roster": {
            "observed": case_count,
            "required": EXPECTED_SOURCES,
            "pass": case_count == EXPECTED_SOURCES,
        },
    }
    passed = all(bool(check["pass"]) for check in checks.values())
    return {
        "pass": passed,
        "status": (
            "opened-engineering-no-qap-gate-pass"
            if passed
            else "opened-engineering-no-qap-gate-fail"
        ),
        "checks": checks,
        "primary_comparison": "learned_no_qap vs learned_standard",
        "secondary_comparison": "learned_no_qap vs rank_delta_transfer",
        "fresh_promotion_evidence": False,
    }


def freeze(args: argparse.Namespace) -> None:
    paths = _paths(args.output_dir)
    paths.predictions.parent.mkdir(parents=True, exist_ok=True)
    if any(path.exists() for path in asdict(paths).values()):
        raise FileExistsError("refusing to overwrite an opened no-QAP replay")
    _validate_decoder_contract()
    upstream._validate_pinned_learned_artifacts(
        args.learned_config,
        args.learned_output,
    )
    rank_config, rank_config_sha, _, records = upstream._load_roster(
        args.rank_config,
        args.manifest,
        limit=args.limit,
    )
    learned_commitment = upstream._load_commitment(
        args.learned_output,
        args.learned_config,
        args.manifest,
    )
    learned_commitment_sha = sha256_file(
        args.learned_output / "selection-commitment.json"
    )
    if learned_commitment_sha != upstream.LEARNED_COMMITMENT_SHA256:
        raise ValueError("learned commitment differs from the pinned pilot")

    device = upstream._select_device(
        args.device,
        allow_nondeterministic_mps=args.allow_nondeterministic_mps,
    )
    models = upstream._load_models_from_commitment(
        learned_commitment,
        device=device,
    )
    upstream._validate_cross_lineage(rank_config, models.metadata)
    learned_model = upstream._load_learned_model(
        args.learned_output / "union-hard-edge-priority.pt",
        args.learned_output / "selection-commitment.json",
        device=device,
    )
    boards = upstream._prepare_boards(records, args.targets)
    arrays: dict[str, np.ndarray] = {}
    frozen_rows: list[dict[str, Any]] = []
    strict_layouts = 0
    synthetic_seed = int(rank_config["selection"]["synthetic_seed"])
    started = perf_counter()
    with torch.inference_mode():
        for index, clean_board in enumerate(boards):
            corruption_seed, permutation_seed = upstream._case_seeds(
                synthetic_seed,
                clean_board.filename,
            )
            dirty, unused_second, unused_reference = upstream._two_view_case(
                clean_board.tiles,
                first_seed=corruption_seed,
                second_seed=corruption_seed + 1,
                permutation_seed=permutation_seed,
            )
            del unused_second, unused_reference
            dirty_sha = upstream._dirty_sha256(dirty)
            target_free = upstream.TargetFreeCase(
                case_id=f"opened-no-qap-{index:04d}-{dirty_sha[:16]}",
                source_filename=clean_board.filename,
                dirty_tiles=dirty,
            )
            board, right, down, preparation = upstream._prepare_target_free_board(
                target_free,
                models,
                device=device,
                inference_batch=args.inference_batch,
                assert_production_parity=False,
            )
            learned_output = learned_model(board)
            union_base = upstream._exact_union_base_priority(board, right, down)
            rank_delta = upstream._rank_delta_priority_from_board(
                board,
                union_base_priority=union_base,
            )
            learned_scores = upstream._aligned_learned_scores(
                board,
                learned_output.scores,
                rank_delta.source,
                rank_delta.target,
                rank_delta.axis,
            )
            learned_matrices = union_hard_edge_priority_matrices(
                board,
                learned_output.scores,
            )
            priorities = {
                "union_v2": rank_delta.base_priority,
                "rank_delta_transfer": rank_delta.scores,
                "learned_standard": learned_scores,
                "learned_no_qap": learned_scores,
            }
            layouts = {
                "union_v2": upstream._decode_layout(
                    right,
                    down,
                    component_edge_priority=None,
                ),
                "rank_delta_transfer": upstream._decode_layout(
                    right,
                    down,
                    component_edge_priority=rank_delta.component_edge_priority,
                ),
                "learned_standard": upstream._decode_layout(
                    right,
                    down,
                    component_edge_priority=learned_matrices,
                ),
                "learned_no_qap": _decode_learned_no_qap(
                    right,
                    down,
                    component_edge_priority=learned_matrices,
                ),
            }
            prefix = f"case_{index:04d}"
            own_satisfied: dict[str, int] = {}
            for arm, layout in layouts.items():
                strict = _strict_layout(layout)
                arrays[f"{prefix}__{arm}_layout"] = strict
                strict_layouts += 1
                own_satisfied[arm] = _own_top288_satisfied(
                    rank_delta.source,
                    rank_delta.target,
                    rank_delta.axis,
                    rank_delta.base_priority,
                    priorities[arm],
                    strict,
                )
            for axis_index in (0, 1):
                selected = rank_delta.axis == axis_index
                arrays[f"{prefix}__axis_{axis_index}_source"] = rank_delta.source[
                    selected
                ]
                arrays[f"{prefix}__axis_{axis_index}_target"] = rank_delta.target[
                    selected
                ]
                for arm, priority in priorities.items():
                    arrays[f"{prefix}__axis_{axis_index}_{arm}_priority"] = np.asarray(
                        priority[selected],
                        dtype=np.float32,
                    )
            frozen_rows.append(
                {
                    "prefix": prefix,
                    "source_filename": clean_board.filename,
                    "draw_index": 0,
                    "corruption_seed": corruption_seed,
                    "permutation_seed": permutation_seed,
                    "dirty_sha256": dirty_sha,
                    "target_free_preparation": preparation,
                    "rank_delta": rank_delta.report(),
                    "own_top288_satisfied": own_satisfied,
                    "learned_layout_same_slots": int(
                        np.count_nonzero(
                            layouts["learned_standard"] == layouts["learned_no_qap"]
                        )
                    ),
                    "decoder_contract": {
                        "learned_standard_max_swap_steps": STANDARD_SWAP_STEPS,
                        "learned_no_qap_max_swap_steps": NO_QAP_SWAP_STEPS,
                        "component_edge_budget_per_axis": DECODER_EDGE_BUDGET,
                        "swap_edge_budget_per_axis": DECODER_EDGE_BUDGET,
                        "cyclic_border_weight": CYCLIC_BORDER_WEIGHT,
                        "same_learned_priority_vector": True,
                    },
                }
            )
            print(
                json.dumps(
                    {"event": "freeze", "done": index + 1, "total": len(boards)}
                ),
                flush=True,
            )

    if strict_layouts != len(records) * len(ARM_NAMES):
        raise RuntimeError("freeze did not produce four strict layouts per case")
    np.savez_compressed(paths.predictions, **arrays)
    runtime_inputs = _runtime_input_records(
        rank_config=args.rank_config,
        learned_config=args.learned_config,
        learned_output=args.learned_output,
        manifest=args.manifest,
    )
    metadata = {
        "schema": "aiijc-learned-no-qap-opened-freeze-v1",
        "panel_role": "already-opened rank-delta fresh64 engineering replay",
        "fresh_promotion_evidence": False,
        "contains_exact_references_or_labels": False,
        "contains_clean_or_dirty_pixels": False,
        "contains_target_free_priorities_and_strict_layouts": True,
        "single_frozen_treatment": "learned priority with max_swap_steps=0",
        "weight_threshold_budget_seed_or_arm_sweep": False,
        "arm_names": list(ARM_NAMES),
        "case_count": len(records),
        "complete_roster": len(records) == EXPECTED_SOURCES,
        "source_filenames": [row["source_filename"] for row in frozen_rows],
        "source_order_digest": names_digest(
            tuple(row["source_filename"] for row in frozen_rows)
        ),
        "synthetic_seed": synthetic_seed,
        "draw_indices": [0],
        "rank_confirmation_config_sha256": rank_config_sha,
        "learned_priority_commitment_sha256": learned_commitment_sha,
        "learned_priority_checkpoint_sha256": sha256_file(
            args.learned_output / "union-hard-edge-priority.pt"
        ),
        "runtime_inputs": runtime_inputs,
        "device": {
            "value": str(device),
            "nondeterministic_mps_explicitly_allowed": bool(
                args.allow_nondeterministic_mps
            ),
            "determinism_claimed": args.device == "cpu",
            "role": "engineering replay only",
        },
        "cases": frozen_rows,
        "runtime_seconds": perf_counter() - started,
    }
    upstream._atomic_json(paths.metadata, metadata)
    freeze_commitment = {
        "schema": "aiijc-learned-no-qap-opened-freeze-commitment-v1",
        "created_after_all_target_free_layouts": True,
        "created_before_exact_reference_scoring": True,
        "predictions": {
            "path": _report_path(paths.predictions),
            "sha256": sha256_file(paths.predictions),
        },
        "metadata": {
            "path": _report_path(paths.metadata),
            "sha256": sha256_file(paths.metadata),
        },
        "runtime_inputs": runtime_inputs,
        "rank_confirmation_config_sha256": rank_config_sha,
        "learned_priority_commitment_sha256": learned_commitment_sha,
        "learned_priority_checkpoint_sha256": sha256_file(
            args.learned_output / "union-hard-edge-priority.pt"
        ),
        "case_count": len(records),
        "source_order_digest": metadata["source_order_digest"],
        "exact_reference_scored": False,
    }
    upstream._atomic_json(paths.freeze_commitment, freeze_commitment)
    print(
        json.dumps(
            {
                "event": "target_free_freeze_complete",
                "predictions_sha256": freeze_commitment["predictions"]["sha256"],
                "metadata_sha256": freeze_commitment["metadata"]["sha256"],
                "freeze_commitment_sha256": sha256_file(paths.freeze_commitment),
            }
        ),
        flush=True,
    )


def _validate_freeze(
    args: argparse.Namespace,
    paths: ReplayPaths,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if paths.report.exists():
        raise FileExistsError("refusing to overwrite an opened no-QAP report")
    if not all(
        path.is_file()
        for path in (paths.predictions, paths.metadata, paths.freeze_commitment)
    ):
        raise FileNotFoundError("score requires a complete prior target-free freeze")
    commitment = json.loads(paths.freeze_commitment.read_text(encoding="utf-8"))
    metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
    if commitment.get("schema") != (
        "aiijc-learned-no-qap-opened-freeze-commitment-v1"
    ):
        raise ValueError("unsupported no-QAP freeze commitment schema")
    if commitment.get("created_before_exact_reference_scoring") is not True:
        raise ValueError("freeze commitment lacks pre-score timing")
    if commitment["predictions"]["sha256"] != sha256_file(paths.predictions):
        raise ValueError("frozen predictions changed before scoring")
    if commitment["metadata"]["sha256"] != sha256_file(paths.metadata):
        raise ValueError("frozen metadata changed before scoring")
    if metadata.get("contains_exact_references_or_labels") is not False:
        raise ValueError("target-free metadata claims exact evidence")
    if metadata.get("arm_names") != list(ARM_NAMES):
        raise ValueError("frozen no-QAP arm roster changed")
    expected_inputs = _runtime_input_records(
        rank_config=args.rank_config,
        learned_config=args.learned_config,
        learned_output=args.learned_output,
        manifest=args.manifest,
    )
    if commitment.get("runtime_inputs") != expected_inputs:
        raise ValueError("runtime input identity changed after freeze")
    if commitment.get("rank_confirmation_config_sha256") != sha256_file(
        args.rank_config
    ):
        raise ValueError("rank confirmation config changed after freeze")
    if commitment.get("learned_priority_commitment_sha256") != sha256_file(
        args.learned_output / "selection-commitment.json"
    ):
        raise ValueError("learned commitment changed after freeze")
    if commitment.get("learned_priority_checkpoint_sha256") != sha256_file(
        args.learned_output / "union-hard-edge-priority.pt"
    ):
        raise ValueError("learned checkpoint changed after freeze")
    return commitment, metadata


def score(args: argparse.Namespace) -> None:
    paths = _paths(args.output_dir)
    freeze_commitment, frozen_metadata = _validate_freeze(args, paths)
    rank_config, rank_config_sha, _, records = upstream._load_roster(
        args.rank_config,
        args.manifest,
        limit=int(freeze_commitment["case_count"]),
    )
    source_names = tuple(record["filename"] for record in records)
    if names_digest(source_names) != freeze_commitment["source_order_digest"]:
        raise ValueError("score roster differs from frozen roster")
    frozen_rows = frozen_metadata.get("cases")
    if not isinstance(frozen_rows, list) or len(frozen_rows) != len(records):
        raise ValueError("frozen case metadata cardinality changed")
    boards = upstream._prepare_boards(records, args.targets)
    synthetic_seed = int(rank_config["selection"]["synthetic_seed"])
    scored_rows: list[dict[str, Any]] = []
    strict_layouts = 0
    started = perf_counter()
    with np.load(paths.predictions, allow_pickle=False) as archive:
        for index, (clean_board, frozen) in enumerate(
            zip(boards, frozen_rows, strict=True)
        ):
            corruption_seed, permutation_seed = upstream._case_seeds(
                synthetic_seed,
                clean_board.filename,
            )
            dirty, unused_second, reference = upstream._two_view_case(
                clean_board.tiles,
                first_seed=corruption_seed,
                second_seed=corruption_seed + 1,
                permutation_seed=permutation_seed,
            )
            del unused_second
            if (
                frozen.get("source_filename") != clean_board.filename
                or frozen.get("corruption_seed") != corruption_seed
                or frozen.get("permutation_seed") != permutation_seed
                or frozen.get("dirty_sha256") != upstream._dirty_sha256(dirty)
            ):
                raise RuntimeError("exact scoring recreated a different synthetic case")
            reference = _strict_layout(reference)
            prefix = f"case_{index:04d}"
            row: dict[str, Any] = {
                "source_filename": clean_board.filename,
                "draw_index": 0,
            }
            for arm in ARM_NAMES:
                layout = _strict_layout(archive[f"{prefix}__{arm}_layout"])
                strict_layouts += 1
                adjacency = float(upstream._adjacency_fraction(layout, reference))
                row[arm] = {
                    "exact_tiles": int(np.count_nonzero(layout == reference)),
                    "adjacency": adjacency,
                    "satisfied_pairs": _satisfied_pairs(adjacency),
                    "fixed_top288_correct": _fixed_top288_correct(
                        archive,
                        prefix,
                        reference,
                        arm=arm,
                    ),
                }
            scored_rows.append(row)

    arms = {
        arm: {
            metric: float(
                np.mean([float(row[arm][metric]) for row in scored_rows])
            )
            for metric in METRIC_NAMES
        }
        for arm in ARM_NAMES
    }
    comparisons = {
        "learned_no_qap_vs_learned_standard": _comparison(
            scored_rows,
            treatment="learned_no_qap",
            baseline="learned_standard",
            seed=BOOTSTRAP_SEED,
        ),
        "learned_no_qap_vs_rank_delta_transfer": _comparison(
            scored_rows,
            treatment="learned_no_qap",
            baseline="rank_delta_transfer",
            seed=BOOTSTRAP_SEED + 100,
        ),
        "learned_standard_vs_rank_delta_transfer": _comparison(
            scored_rows,
            treatment="learned_standard",
            baseline="rank_delta_transfer",
            seed=BOOTSTRAP_SEED + 200,
        ),
    }
    metrics = {
        "arms": arms,
        **comparisons,
        "strict_layouts": strict_layouts,
        "case_count": len(scored_rows),
    }
    gate = evaluate_gate(
        metrics,
        strict_layouts=strict_layouts,
        case_count=len(scored_rows),
    )
    report = {
        "schema": "aiijc-learned-no-qap-opened-report-v1",
        "status": gate["status"],
        "panel_role": "already-opened rank-delta fresh64 engineering replay",
        "fresh_promotion_evidence": False,
        "single_frozen_treatment": "learned priority with max_swap_steps=0",
        "rank_confirmation_config": {
            "path": _report_path(args.rank_config),
            "sha256": rank_config_sha,
        },
        "learned_priority_selection_commitment": {
            "path": _report_path(
                args.learned_output / "selection-commitment.json"
            ),
            "sha256": sha256_file(
                args.learned_output / "selection-commitment.json"
            ),
        },
        "learned_priority_checkpoint": {
            "path": _report_path(
                args.learned_output / "union-hard-edge-priority.pt"
            ),
            "sha256": sha256_file(
                args.learned_output / "union-hard-edge-priority.pt"
            ),
        },
        "freeze": {
            "commitment_path": _report_path(paths.freeze_commitment),
            "commitment_sha256": sha256_file(paths.freeze_commitment),
            "predictions_path": _report_path(paths.predictions),
            "predictions_sha256": sha256_file(paths.predictions),
            "metadata_path": _report_path(paths.metadata),
            "metadata_sha256": sha256_file(paths.metadata),
            "all_priorities_and_layouts_frozen_before_exact_scoring": True,
        },
        "decoder_contract": {
            "learned_standard_max_swap_steps": STANDARD_SWAP_STEPS,
            "learned_no_qap_max_swap_steps": NO_QAP_SWAP_STEPS,
            "component_edge_budget_per_axis": DECODER_EDGE_BUDGET,
            "swap_edge_budget_per_axis": DECODER_EDGE_BUDGET,
            "cyclic_border_weight": CYCLIC_BORDER_WEIGHT,
            "same_learned_top144_priority": True,
        },
        "metrics": metrics,
        "gate": gate,
        "rows": scored_rows,
        "runtime_seconds": perf_counter() - started,
        "legality": {
            "organizer_train_only": True,
            "competition_test_or_holdout_opened": False,
            "original_upright_tile_permutations_only": True,
            "restored_pixels_matcher_only": True,
            "new_hard_edges_introduced": False,
            "target_available_to_priority_or_decoder": False,
        },
        "weight_threshold_budget_seed_or_arm_sweep": False,
    }
    upstream._atomic_json(paths.report, report)
    print(
        json.dumps(
            {
                "event": "score_complete",
                "report": _report_path(paths.report),
                "report_sha256": sha256_file(paths.report),
                "gate": gate,
            }
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.mode == "freeze":
        freeze(args)
    else:
        score(args)


if __name__ == "__main__":
    main()

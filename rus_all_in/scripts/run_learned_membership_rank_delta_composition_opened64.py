#!/usr/bin/env python3
"""Replay learned-membership + rank-delta composition on opened fresh64.

This is an engineering replay on the already-opened rank-delta confirmation
roster, not fresh promotion evidence.  ``freeze`` builds all four target-free
priority/layout arms and writes a hash commitment.  ``score`` refuses to run
unless those artifacts and every pinned runtime input still match, then and
only then recreates exact synthetic references.

The four arms share one immutable Union-v2 hard-edge supply:

* ``union_v2``: native Union ordering;
* ``rank_delta_transfer``: Direct learned-minus-raw rank displacement already
  embedded in the learned-priority feature board;
* ``learned_priority``: the frozen Union hard-edge learned residual head;
* ``membership_rank_composition``: learned top-144 membership per axis with
  rank-delta ordering inside/outside that cutoff.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from aiijc_puzzle.direct_hard_edge_production import infer_direct_hard_edge_priorities
from aiijc_puzzle.direct_residual_union_priority import (
    _descending_midrank_quality,
    build_direct_rank_delta_union_priority,
)
from aiijc_puzzle.learned_membership_rank_delta_priority import (
    compose_learned_membership_rank_delta_priority,
)
from aiijc_puzzle.protocol import compute_protocol_digest, sha256_file
from aiijc_puzzle.socket_decoder import hard_partial_axis_matching
from aiijc_puzzle.socket_sorter_production import DECODER_EDGE_BUDGET
from aiijc_puzzle.synthetic_socket_evaluation import names_digest
from aiijc_puzzle.union_hard_edge_priority import (
    FEATURE_NAMES,
    UnionHardEdgeBoard,
    UnionHardEdgePriority,
    union_hard_edge_priority_matrices,
    validate_union_hard_edge_board,
)

try:
    from scripts.run_direct_rank_delta_component_selector_fresh64 import (
        _validated_run_roster,
        load_confirmation_config,
    )
    from scripts.run_fullres_twin_side_matcher import (
        _atomic_json,
        _prepare_boards,
        _two_view_case,
    )
    from scripts.run_raw_twin_union_reranker_fresh64 import (
        DEFAULT_MANIFEST,
        DEFAULT_TARGETS,
        _case_seeds,
        source_clustered_ci,
    )
    from scripts.run_raw_twin_union_reranker_v2 import _adjacency_fraction
    from scripts.run_union_hard_edge_priority_pilot import (
        DEFAULT_CONFIG as LEARNED_PILOT_CONFIG,
    )
    from scripts.run_union_hard_edge_priority_pilot import (
        INFERENCE_BATCH,
        TargetFreeCase,
        _decode_layout,
        _load_commitment,
        _load_models_from_commitment,
        _prepare_target_free_board,
        _select_device,
        _strict_layout,
    )
except ModuleNotFoundError:  # Direct ``python scripts/run_*.py`` execution.
    from run_direct_rank_delta_component_selector_fresh64 import (
        _validated_run_roster,
        load_confirmation_config,
    )
    from run_fullres_twin_side_matcher import _atomic_json, _prepare_boards, _two_view_case
    from run_raw_twin_union_reranker_fresh64 import (
        DEFAULT_MANIFEST,
        DEFAULT_TARGETS,
        _case_seeds,
        source_clustered_ci,
    )
    from run_raw_twin_union_reranker_v2 import _adjacency_fraction
    from run_union_hard_edge_priority_pilot import (
        DEFAULT_CONFIG as LEARNED_PILOT_CONFIG,
    )
    from run_union_hard_edge_priority_pilot import (
        INFERENCE_BATCH,
        TargetFreeCase,
        _decode_layout,
        _load_commitment,
        _load_models_from_commitment,
        _prepare_target_free_board,
        _select_device,
        _strict_layout,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRID = 24
COUNT = GRID * GRID
HARD_EDGES_PER_AXIS = GRID * (GRID - 1)
HARD_EDGE_COUNT = 2 * HARD_EDGES_PER_AXIS
EXPECTED_SOURCES = 64
BOOTSTRAP_SEED = 955_314_071
BOOTSTRAP_RESAMPLES = 20_000

RANK_CONFIG = (
    PROJECT_ROOT / "configs/direct_rank_delta_component_selector_fresh64_confirmation_v1.json"
)
LEARNED_OUTPUT = PROJECT_ROOT / "outputs/union-hard-edge-priority/pilot-v1-final"
COMPOSITION_IMPLEMENTATION = (
    PROJECT_ROOT / "src/aiijc_puzzle/learned_membership_rank_delta_priority.py"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs/learned-membership-rank-delta-composition/opened-fresh64-v1"
)

LEARNED_COMMITMENT_SHA256 = "575bb43d850ec3276b61aef616cfa9f2f5fa6f31db35f417d6852b9a38dac540"
LEARNED_CHECKPOINT_SHA256 = "472c2770e8960125359c44afdafa6cd31fbb6517d3db33e514b94aa56905efd5"
LEARNED_PILOT_CONFIG_SHA256 = "3cc28b93d88f7e13366740f59a230635a98a528cb11e5e941a0ce3fa9256e7f6"

ARM_NAMES = (
    "union_v2",
    "rank_delta_transfer",
    "learned_priority",
    "membership_rank_composition",
)
METRIC_NAMES = ("exact_tiles", "adjacency", "fixed_top288_correct")
DIRECT_DELTA_FEATURE = "direct_rank_quality_delta_axis"
DIRECT_PRESENT_FEATURE = "direct_identity_present"
DIRECT_RAW_QUALITY_FEATURE = "direct_raw_rank_quality_axis"
DIRECT_LEARNED_QUALITY_FEATURE = "direct_learned_rank_quality_axis"
DIRECT_RANK_QUANTUM = 0.5 / (HARD_EDGES_PER_AXIS - 1)


@dataclass(frozen=True)
class ReplayPaths:
    predictions: Path
    metadata: Path
    freeze_commitment: Path
    report: Path


@dataclass(frozen=True)
class EmbeddedRankDeltaPriority:
    """Scale-preserving rank-delta priority reconstructed from board features."""

    source: np.ndarray
    target: np.ndarray
    axis: np.ndarray
    base_priority: np.ndarray
    scores: np.ndarray
    component_edge_priority: dict[str, np.ndarray]
    matched_per_axis: tuple[int, int]
    changed_membership_per_axis: tuple[int, int]

    def report(self) -> dict[str, Any]:
        return {
            "schema": "aiijc-embedded-direct-rank-delta-union-priority-v1",
            "method": (
                "Union base rank plus FEATURE_NAMES direct_rank_quality_delta_axis; "
                "original Union confidence multiset reassigned per axis"
            ),
            "direct_feature": DIRECT_DELTA_FEATURE,
            "float32_rank_features_restored_to_exact_midrank_lattice": {
                "raw": DIRECT_RAW_QUALITY_FEATURE,
                "learned": DIRECT_LEARNED_QUALITY_FEATURE,
                "quantum": DIRECT_RANK_QUANTUM,
            },
            "matched_per_axis": list(self.matched_per_axis),
            "changed_top144_membership_per_axis": list(self.changed_membership_per_axis),
            "confidence_multiset_preserved_per_axis": [True, True],
            "new_hard_edges_introduced": False,
        }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "score"))
    parser.add_argument("--rank-config", type=Path, default=RANK_CONFIG)
    parser.add_argument("--learned-config", type=Path, default=LEARNED_PILOT_CONFIG)
    parser.add_argument("--learned-output", type=Path, default=LEARNED_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--allow-nondeterministic-mps", action="store_true")
    parser.add_argument("--inference-batch", type=int, default=INFERENCE_BATCH)
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
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _dirty_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _runtime_input_records(
    *,
    rank_config: Path,
    learned_config: Path,
    learned_output: Path,
    manifest: Path,
) -> dict[str, dict[str, str]]:
    paths = {
        "runner": Path(__file__).resolve(),
        "rank_confirmation_config": rank_config.resolve(),
        "learned_priority_config": learned_config.resolve(),
        "learned_priority_commitment": learned_output.resolve() / "selection-commitment.json",
        "learned_priority_checkpoint": learned_output.resolve() / "union-hard-edge-priority.pt",
        "composition_implementation": COMPOSITION_IMPLEMENTATION.resolve(),
        "manifest": manifest.resolve(),
    }
    return {
        name: {"path": _report_path(path), "sha256": sha256_file(path)}
        for name, path in paths.items()
    }


def _validate_pinned_learned_artifacts(
    learned_config: Path,
    learned_output: Path,
) -> None:
    observed = {
        "config": sha256_file(learned_config),
        "commitment": sha256_file(learned_output / "selection-commitment.json"),
        "checkpoint": sha256_file(learned_output / "union-hard-edge-priority.pt"),
    }
    expected = {
        "config": LEARNED_PILOT_CONFIG_SHA256,
        "commitment": LEARNED_COMMITMENT_SHA256,
        "checkpoint": LEARNED_CHECKPOINT_SHA256,
    }
    if observed != expected:
        raise ValueError(f"learned-priority artifact identity changed: {observed}")


def _load_learned_model(
    checkpoint_path: Path,
    commitment_path: Path,
    *,
    device: torch.device,
) -> UnionHardEdgePriority:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "aiijc-union-hard-edge-priority-checkpoint-v1":
        raise ValueError("unsupported learned-priority checkpoint schema")
    contract = payload.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("learned-priority checkpoint lacks a model contract")
    pinned = {
        "architecture": "union-hard-edge-deepsets-bounded-residual-v1",
        "feature_dimension": len(FEATURE_NAMES),
        "hard_edge_count": HARD_EDGE_COUNT,
        "edge_budget_per_axis": DECODER_EDGE_BUDGET,
    }
    if any(contract.get(name) != value for name, value in pinned.items()):
        raise ValueError("learned-priority checkpoint contract changed")
    if tuple(contract.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("learned-priority checkpoint feature names changed")
    if payload.get("selection_commitment_sha256") != sha256_file(commitment_path):
        raise ValueError("checkpoint points to a different learned selection commitment")
    model = UnionHardEdgePriority(
        feature_dimension=int(contract["feature_dimension"]),
        hidden_dimension=int(contract["hidden_dimension"]),
        residual_limit=float(contract["residual_limit"]),
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model.to(device)


def _deterministic_order(
    primary: np.ndarray,
    base: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    return np.lexsort((target, source, -base, -primary))


def _exact_union_base_priority(
    board: UnionHardEdgeBoard,
    right: np.ndarray,
    down: np.ndarray,
) -> np.ndarray:
    """Align decoder-native float64 Union confidences to board identities."""

    board_axis = np.asarray(board.axis.detach().cpu().numpy(), dtype=np.int8)
    result = np.empty(HARD_EDGE_COUNT, dtype=np.float64)
    for axis_index, (name, assignment) in enumerate((("right", right), ("down", down))):
        matching = hard_partial_axis_matching(assignment, grid=GRID, axis=name)
        by_identity = {
            (edge.source, edge.target): float(edge.confidence) for edge in matching.edges
        }
        selected = np.flatnonzero(board_axis == axis_index)
        identities = [
            (int(board.source[index]), int(board.target[index])) for index in selected
        ]
        if len(by_identity) != HARD_EDGES_PER_AXIS or set(identities) != set(by_identity):
            raise ValueError("Union assignment and learned board hard identities differ")
        result[selected] = [by_identity[identity] for identity in identities]
    return np.ascontiguousarray(result)


def _rank_delta_priority_from_board(
    board: UnionHardEdgeBoard,
    *,
    union_base_priority: Any | None = None,
) -> EmbeddedRankDeltaPriority:
    """Reconstruct the established rank-delta arm without a second Direct pass."""

    validate_union_hard_edge_board(board)
    source = np.asarray(board.source, dtype=np.int32)
    target = np.asarray(board.target, dtype=np.int32)
    axis = np.asarray(board.axis.detach().cpu().numpy(), dtype=np.int8)
    if union_base_priority is None:
        base = np.asarray(
            board.base_priority.detach().float().cpu().numpy(), dtype=np.float64
        )
    else:
        base = np.asarray(union_base_priority, dtype=np.float64)
    values = np.asarray(board.values.detach().float().cpu().numpy(), dtype=np.float64)
    encoded_delta = values[:, FEATURE_NAMES.index(DIRECT_DELTA_FEATURE)]
    encoded_raw_quality = values[:, FEATURE_NAMES.index(DIRECT_RAW_QUALITY_FEATURE)]
    encoded_learned_quality = values[
        :, FEATURE_NAMES.index(DIRECT_LEARNED_QUALITY_FEATURE)
    ]
    present = values[:, FEATURE_NAMES.index(DIRECT_PRESENT_FEATURE)] > 0.5
    if (
        source.shape != (HARD_EDGE_COUNT,)
        or target.shape != source.shape
        or axis.shape != source.shape
        or base.shape != source.shape
        or values.shape != (HARD_EDGE_COUNT, len(FEATURE_NAMES))
        or not np.isfinite(base).all()
        or not np.isfinite(encoded_delta).all()
        or not np.isfinite(encoded_raw_quality).all()
        or not np.isfinite(encoded_learned_quality).all()
        or np.any((~present) & (encoded_delta != 0.0))
    ):
        raise ValueError("cached Union board violates embedded rank-delta contract")

    def restore_midrank_quality(encoded: np.ndarray) -> np.ndarray:
        twice_midrank = np.rint(
            (1.0 - encoded) * (2 * (HARD_EDGES_PER_AXIS - 1))
        ).astype(np.int64)
        if np.any((twice_midrank < 0) | (twice_midrank > 2 * (HARD_EDGES_PER_AXIS - 1))):
            raise ValueError("embedded Direct quality is outside its midrank lattice")
        return 1.0 - (0.5 * twice_midrank) / (HARD_EDGES_PER_AXIS - 1)

    # Recover both canonical float64 operations, not merely their float32
    # difference.  This reproduces even exact adjusted-rank ties from the
    # original Direct transfer without repeating Direct production inference.
    raw_quality = restore_midrank_quality(encoded_raw_quality)
    learned_quality = restore_midrank_quality(encoded_learned_quality)
    delta = np.where(present, learned_quality - raw_quality, 0.0)
    if np.max(np.abs(delta - encoded_delta)) > 1e-6:
        raise ValueError("embedded direct rank delta is off its exact midrank lattice")

    scores = np.empty(HARD_EDGE_COUNT, dtype=np.float64)
    matrices = {
        "right": np.zeros((COUNT, COUNT), dtype=np.float64),
        "down": np.zeros((COUNT, COUNT), dtype=np.float64),
    }
    matched: list[int] = []
    membership_changes: list[int] = []
    for axis_index, name in ((0, "right"), (1, "down")):
        indices = np.flatnonzero(axis == axis_index)
        if len(indices) != HARD_EDGES_PER_AXIS:
            raise ValueError("embedded rank-delta axis cardinality changed")
        axis_base = base[indices]
        base_quality = _descending_midrank_quality(axis_base)
        adjusted = base_quality + delta[indices]
        adjusted_order = _deterministic_order(
            adjusted,
            axis_base,
            source[indices],
            target[indices],
        )
        base_order = _deterministic_order(
            axis_base,
            axis_base,
            source[indices],
            target[indices],
        )
        descending_base = np.sort(axis_base)[::-1]
        axis_scores = np.empty(HARD_EDGES_PER_AXIS, dtype=np.float64)
        axis_scores[adjusted_order] = descending_base
        if not np.array_equal(np.sort(axis_scores), np.sort(axis_base)):
            raise RuntimeError("embedded rank-delta changed the Union confidence multiset")
        scores[indices] = axis_scores
        matrices[name][source[indices], target[indices]] = axis_scores
        matched.append(int(np.count_nonzero(present[indices])))
        membership_changes.append(
            len(
                set(base_order[:DECODER_EDGE_BUDGET].tolist())
                ^ set(adjusted_order[:DECODER_EDGE_BUDGET].tolist())
            )
            // 2
        )
    return EmbeddedRankDeltaPriority(
        source=source.copy(),
        target=target.copy(),
        axis=axis.copy(),
        base_priority=np.ascontiguousarray(base),
        scores=np.ascontiguousarray(scores),
        component_edge_priority={
            name: np.ascontiguousarray(value) for name, value in matrices.items()
        },
        matched_per_axis=(matched[0], matched[1]),
        changed_membership_per_axis=(membership_changes[0], membership_changes[1]),
    )


def _assert_first_case_rank_delta_parity(
    embedded: EmbeddedRankDeltaPriority,
    right: np.ndarray,
    down: np.ndarray,
    dirty: np.ndarray,
    models: Any,
    *,
    device: torch.device,
) -> dict[str, Any]:
    direct = infer_direct_hard_edge_priorities(
        dirty,
        models.socket,
        models.direct,
        device=device,
    )
    canonical = build_direct_rank_delta_union_priority(
        right,
        down,
        direct_source=direct.source,
        direct_target=direct.target,
        direct_axis=direct.axis,
        direct_raw_scores=direct.raw_scores,
        direct_learned_scores=direct.learned_scores,
        grid=GRID,
    )
    axis_reports: dict[str, Any] = {}
    for axis_index, name in ((0, "right"), (1, "down")):
        selected = embedded.axis == axis_index
        canonical_scores = canonical.component_edge_priority[name][
            embedded.source[selected], embedded.target[selected]
        ]
        embedded_scores = embedded.scores[selected]
        canonical_order = _deterministic_order(
            canonical_scores,
            embedded.base_priority[selected],
            embedded.source[selected],
            embedded.target[selected],
        )
        embedded_order = _deterministic_order(
            embedded_scores,
            embedded.base_priority[selected],
            embedded.source[selected],
            embedded.target[selected],
        )
        if not np.array_equal(canonical_order, embedded_order):
            raise RuntimeError(f"embedded rank-delta order differs from canonical {name}")
        axis_reports[name] = {
            "complete_order_equal": True,
            "top144_membership_equal": bool(
                set(canonical_order[:DECODER_EDGE_BUDGET].tolist())
                == set(embedded_order[:DECODER_EDGE_BUDGET].tolist())
            ),
        }
    return {
        "checked": True,
        "case_index": 0,
        "canonical_direct_inference": direct.report(),
        "axes": axis_reports,
    }


def _aligned_learned_scores(
    board: UnionHardEdgeBoard,
    scores: Any,
    source: np.ndarray,
    target: np.ndarray,
    axis: np.ndarray,
) -> np.ndarray:
    learned = np.asarray(scores.detach().float().cpu().numpy(), dtype=np.float64)
    board_axis = np.asarray(board.axis.detach().cpu().numpy(), dtype=np.int8)
    if learned.shape != (HARD_EDGE_COUNT,):
        raise ValueError("learned scores violate hard-edge cardinality")
    by_identity = {
        (int(a), int(s), int(t)): float(value)
        for a, s, t, value in zip(
            board_axis,
            board.source,
            board.target,
            learned,
            strict=True,
        )
    }
    identities = [(int(a), int(s), int(t)) for a, s, t in zip(axis, source, target, strict=True)]
    if len(by_identity) != HARD_EDGE_COUNT or set(identities) != set(by_identity):
        raise ValueError("learned and rank-delta hard-edge identity rosters differ")
    return np.asarray([by_identity[identity] for identity in identities], dtype=np.float64)


def _priority_vectors(
    result: EmbeddedRankDeltaPriority,
    learned_scores: np.ndarray,
    composed_scores: np.ndarray,
) -> dict[str, np.ndarray]:
    vectors = {
        "union_v2": result.base_priority,
        "rank_delta_transfer": result.scores,
        "learned_priority": learned_scores,
        "membership_rank_composition": np.asarray(composed_scores, dtype=np.float64),
    }
    if any(
        value.shape != (HARD_EDGE_COUNT,) or not np.isfinite(value).all()
        for value in vectors.values()
    ):
        raise ValueError("one replay arm has an invalid priority vector")
    return {name: np.ascontiguousarray(value) for name, value in vectors.items()}


def _edge_truth(
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
        return (target_position == source_position + 1) & (source_position % GRID != GRID - 1)
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
        raise ValueError(f"unknown replay arm: {arm}")
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
            raise ValueError("frozen arrays violate the fixed-top288 contract")
        order = np.argsort(-priority, kind="stable")[:DECODER_EDGE_BUDGET]
        total += int(
            np.count_nonzero(_edge_truth(source, target, axis=axis, reference=reference)[order])
        )
    return total


def _win_tie_loss(values: Sequence[float]) -> dict[str, int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("delta values must be a finite non-empty vector")
    return {
        "wins": int(np.count_nonzero(array > 0)),
        "ties": int(np.count_nonzero(array == 0)),
        "losses": int(np.count_nonzero(array < 0)),
    }


def _comparison(
    rows: Sequence[Mapping[str, Any]],
    *,
    treatment: str,
    baseline: str,
    seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for offset, metric in enumerate(METRIC_NAMES):
        values = [float(row[treatment][metric]) - float(row[baseline][metric]) for row in rows]
        result[f"{metric}_delta"] = source_clustered_ci(
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
    versus_rank = metrics["membership_rank_composition_vs_rank_delta_transfer"]
    exact = float(versus_rank["exact_tiles_delta"]["mean"])
    adjacency = float(versus_rank["adjacency_delta"]["mean"])
    fixed = float(versus_rank["fixed_top288_correct_delta"]["mean"])
    checks = {
        "exact_nonnegative_vs_rank_delta": {
            "observed": exact,
            "required": ">=0",
            "pass": exact >= 0.0,
        },
        "adjacency_strictly_positive_vs_rank_delta": {
            "observed": adjacency,
            "required": ">0",
            "pass": adjacency > 0.0,
        },
        "fixed_top288_nonnegative_vs_rank_delta": {
            "observed": fixed,
            "required": ">=0",
            "pass": fixed >= 0.0,
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
            "opened-engineering-replay-gate-pass"
            if passed
            else "opened-engineering-replay-gate-fail"
        ),
        "checks": checks,
        "comparison": "membership_rank_composition vs rank_delta_transfer",
        "fresh_promotion_evidence": False,
    }


def _load_roster(
    rank_config_path: Path,
    manifest_path: Path,
    *,
    limit: int,
) -> tuple[dict[str, Any], str, dict[str, Any], tuple[dict[str, Any], ...]]:
    if not 1 <= limit <= EXPECTED_SOURCES:
        raise ValueError(f"limit must be in [1, {EXPECTED_SOURCES}]")
    rank_config, rank_config_sha = load_confirmation_config(rank_config_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_digest") != compute_protocol_digest(manifest):
        raise ValueError("manifest protocol digest mismatch")
    manifest_record = rank_config["frozen_inputs"]["manifest"]
    if sha256_file(manifest_path) != manifest_record["sha256"]:
        raise ValueError("replay manifest differs from rank-delta confirmation")
    audit_path = _resolve_project_path(str(rank_config["frozen_inputs"]["roster_audit"]["path"]))
    names = _validated_run_roster(rank_config, audit_path, manifest)
    if rank_config["selection"].get("draw_indices") != [0]:
        raise ValueError("rank-delta replay requires exactly draw zero")
    lookup = {str(record["filename"]): dict(record) for record in manifest["splits"]["train"]}
    records = tuple(lookup[name] for name in names[:limit])
    return rank_config, rank_config_sha, manifest, records


def _validate_cross_lineage(
    rank_config: Mapping[str, Any],
    model_metadata: Mapping[str, Any],
) -> None:
    frozen = rank_config["frozen_inputs"]
    observed = {
        "socket_checkpoint": model_metadata["fusion_dependencies"]["socket"]["sha256"],
        "twin_checkpoint": model_metadata["twin_checkpoint_sha256"],
        "union_checkpoint": model_metadata["union_checkpoint_sha256"],
        "direct_checkpoint": model_metadata["direct_checkpoint_sha256"],
    }
    expected = {name: frozen[name]["sha256"] for name in observed}
    if observed != expected:
        raise ValueError("rank-delta and learned-priority frozen model lineages differ")


def freeze(args: argparse.Namespace) -> None:
    paths = _paths(args.output_dir)
    paths.predictions.parent.mkdir(parents=True, exist_ok=True)
    if any(path.exists() for path in asdict(paths).values()):
        raise FileExistsError("refusing to overwrite an opened composition replay")
    _validate_pinned_learned_artifacts(args.learned_config, args.learned_output)
    rank_config, rank_config_sha, _, records = _load_roster(
        args.rank_config,
        args.manifest,
        limit=args.limit,
    )
    learned_commitment = _load_commitment(
        args.learned_output,
        args.learned_config,
        args.manifest,
    )
    learned_commitment_sha = sha256_file(args.learned_output / "selection-commitment.json")
    if learned_commitment_sha != LEARNED_COMMITMENT_SHA256:
        raise ValueError("learned commitment differs from the pinned pilot")

    device = _select_device(
        args.device,
        allow_nondeterministic_mps=args.allow_nondeterministic_mps,
    )
    models = _load_models_from_commitment(learned_commitment, device=device)
    _validate_cross_lineage(rank_config, models.metadata)
    learned_model = _load_learned_model(
        args.learned_output / "union-hard-edge-priority.pt",
        args.learned_output / "selection-commitment.json",
        device=device,
    )
    boards = _prepare_boards(records, args.targets)
    arrays: dict[str, np.ndarray] = {}
    frozen_rows: list[dict[str, Any]] = []
    strict_layouts = 0
    parity_report: dict[str, Any] | None = None
    synthetic_seed = int(rank_config["selection"]["synthetic_seed"])
    started = perf_counter()
    with torch.inference_mode():
        for index, clean_board in enumerate(boards):
            corruption_seed, permutation_seed = _case_seeds(
                synthetic_seed,
                clean_board.filename,
            )
            dirty, unused_second, unused_reference = _two_view_case(
                clean_board.tiles,
                first_seed=corruption_seed,
                second_seed=corruption_seed + 1,
                permutation_seed=permutation_seed,
            )
            del unused_second, unused_reference
            dirty_sha = _dirty_sha256(dirty)
            target_free = TargetFreeCase(
                case_id=f"opened-replay-{index:04d}-{dirty_sha[:16]}",
                source_filename=clean_board.filename,
                dirty_tiles=dirty,
            )
            board, right, down, preparation = _prepare_target_free_board(
                target_free,
                models,
                device=device,
                inference_batch=args.inference_batch,
                assert_production_parity=False,
            )
            learned_output = learned_model(board)
            union_base = _exact_union_base_priority(board, right, down)
            rank_delta = _rank_delta_priority_from_board(
                board,
                union_base_priority=union_base,
            )
            if index == 0:
                socket_device = next(models.socket.model.parameters()).device
                direct_device = next(models.direct.model.parameters()).device
                if direct_device != socket_device:
                    raise RuntimeError(
                        "Socket and Direct models occupy different devices during parity"
                    )
                parity_report = _assert_first_case_rank_delta_parity(
                    rank_delta,
                    right,
                    down,
                    dirty,
                    models,
                    device=socket_device,
                )
            learned_scores = _aligned_learned_scores(
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
            composition = compose_learned_membership_rank_delta_priority(
                rank_delta.source,
                rank_delta.target,
                rank_delta.axis,
                rank_delta.base_priority,
                learned_scores,
                rank_delta.scores,
                grid=GRID,
                edge_budget_per_axis=DECODER_EDGE_BUDGET,
            )
            priorities = _priority_vectors(
                rank_delta,
                learned_scores,
                composition.scores,
            )
            layouts = {
                "union_v2": _decode_layout(right, down, component_edge_priority=None),
                "rank_delta_transfer": _decode_layout(
                    right,
                    down,
                    component_edge_priority=rank_delta.component_edge_priority,
                ),
                "learned_priority": _decode_layout(
                    right,
                    down,
                    component_edge_priority=learned_matrices,
                ),
                "membership_rank_composition": _decode_layout(
                    right,
                    down,
                    component_edge_priority=composition.component_edge_priority,
                ),
            }
            prefix = f"case_{index:04d}"
            for arm, layout in layouts.items():
                arrays[f"{prefix}__{arm}_layout"] = _strict_layout(layout)
                strict_layouts += 1
            for axis_index in (0, 1):
                selected = rank_delta.axis == axis_index
                arrays[f"{prefix}__axis_{axis_index}_source"] = rank_delta.source[selected]
                arrays[f"{prefix}__axis_{axis_index}_target"] = rank_delta.target[selected]
                for arm, priority in priorities.items():
                    arrays[f"{prefix}__axis_{axis_index}_{arm}_priority"] = np.asarray(
                        priority[selected], dtype=np.float32
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
                    "composition": composition.report(),
                }
            )
            print(
                json.dumps({"event": "freeze", "done": index + 1, "total": len(boards)}),
                flush=True,
            )

    if strict_layouts != len(records) * len(ARM_NAMES):
        raise RuntimeError("freeze did not produce four strict layouts per case")
    if parity_report is None:
        raise RuntimeError("first-case rank-delta parity was not checked")
    np.savez_compressed(paths.predictions, **arrays)
    runtime_inputs = _runtime_input_records(
        rank_config=args.rank_config,
        learned_config=args.learned_config,
        learned_output=args.learned_output,
        manifest=args.manifest,
    )
    metadata = {
        "schema": "aiijc-learned-membership-rank-delta-opened-freeze-v1",
        "panel_role": "already-opened rank-delta fresh64 engineering replay",
        "contains_exact_references_or_labels": False,
        "contains_clean_or_dirty_pixels": False,
        "contains_target_free_priorities_and_strict_layouts": True,
        "arm_names": list(ARM_NAMES),
        "case_count": len(records),
        "complete_roster": len(records) == EXPECTED_SOURCES,
        "source_filenames": [row["source_filename"] for row in frozen_rows],
        "source_order_digest": names_digest(tuple(row["source_filename"] for row in frozen_rows)),
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
            "nondeterministic_mps_explicitly_allowed": bool(args.allow_nondeterministic_mps),
            "determinism_claimed": args.device == "cpu",
        },
        "first_case_rank_delta_parity": parity_report,
        "cases": frozen_rows,
        "runtime_seconds": perf_counter() - started,
    }
    _atomic_json(paths.metadata, metadata)
    freeze_commitment = {
        "schema": "aiijc-learned-membership-rank-delta-opened-freeze-commitment-v1",
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
    _atomic_json(paths.freeze_commitment, freeze_commitment)
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
        raise FileExistsError("refusing to overwrite an opened composition report")
    if not all(
        path.is_file() for path in (paths.predictions, paths.metadata, paths.freeze_commitment)
    ):
        raise FileNotFoundError("score requires a complete prior target-free freeze")
    commitment = json.loads(paths.freeze_commitment.read_text(encoding="utf-8"))
    metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
    if commitment.get("schema") != (
        "aiijc-learned-membership-rank-delta-opened-freeze-commitment-v1"
    ):
        raise ValueError("unsupported freeze commitment schema")
    if commitment.get("created_before_exact_reference_scoring") is not True:
        raise ValueError("freeze commitment lacks pre-score timing")
    if commitment["predictions"]["sha256"] != sha256_file(paths.predictions):
        raise ValueError("frozen predictions changed before scoring")
    if commitment["metadata"]["sha256"] != sha256_file(paths.metadata):
        raise ValueError("frozen metadata changed before scoring")
    if metadata.get("contains_exact_references_or_labels") is not False:
        raise ValueError("target-free metadata claims exact evidence")
    if metadata.get("arm_names") != list(ARM_NAMES):
        raise ValueError("frozen arm roster changed")
    for name, record in commitment.get("runtime_inputs", {}).items():
        path = _resolve_project_path(str(record.get("path", "")))
        if sha256_file(path) != record.get("sha256"):
            raise ValueError(f"runtime input changed after freeze: {name}")
    if commitment.get("rank_confirmation_config_sha256") != sha256_file(args.rank_config):
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
    rank_config, rank_config_sha, _, records = _load_roster(
        args.rank_config,
        args.manifest,
        limit=int(freeze_commitment["case_count"]),
    )
    names = tuple(record["filename"] for record in records)
    if names_digest(names) != freeze_commitment["source_order_digest"]:
        raise ValueError("score roster differs from frozen roster")
    frozen_rows = frozen_metadata.get("cases")
    if not isinstance(frozen_rows, list) or len(frozen_rows) != len(records):
        raise ValueError("frozen case metadata cardinality changed")
    boards = _prepare_boards(records, args.targets)
    synthetic_seed = int(rank_config["selection"]["synthetic_seed"])
    scored_rows: list[dict[str, Any]] = []
    strict_layouts = 0
    started = perf_counter()
    with np.load(paths.predictions, allow_pickle=False) as archive:
        for index, (clean_board, frozen) in enumerate(zip(boards, frozen_rows, strict=True)):
            corruption_seed, permutation_seed = _case_seeds(
                synthetic_seed,
                clean_board.filename,
            )
            dirty, unused_second, reference = _two_view_case(
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
                or frozen.get("dirty_sha256") != _dirty_sha256(dirty)
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
                row[arm] = {
                    "exact_tiles": int(np.count_nonzero(layout == reference)),
                    "adjacency": float(_adjacency_fraction(layout, reference)),
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
            metric: float(np.mean([float(row[arm][metric]) for row in scored_rows]))
            for metric in METRIC_NAMES
        }
        for arm in ARM_NAMES
    }
    comparisons = {
        f"membership_rank_composition_vs_{baseline}": _comparison(
            scored_rows,
            treatment="membership_rank_composition",
            baseline=baseline,
            seed=BOOTSTRAP_SEED + 10 * index,
        )
        for index, baseline in enumerate(("union_v2", "rank_delta_transfer", "learned_priority"))
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
        "schema": "aiijc-learned-membership-rank-delta-opened-report-v1",
        "status": gate["status"],
        "panel_role": "already-opened rank-delta fresh64 engineering replay",
        "fresh_promotion_evidence": False,
        "rank_confirmation_config": {
            "path": _report_path(args.rank_config),
            "sha256": rank_config_sha,
        },
        "learned_priority_selection_commitment": {
            "path": _report_path(args.learned_output / "selection-commitment.json"),
            "sha256": sha256_file(args.learned_output / "selection-commitment.json"),
        },
        "learned_priority_checkpoint": {
            "path": _report_path(args.learned_output / "union-hard-edge-priority.pt"),
            "sha256": sha256_file(args.learned_output / "union-hard-edge-priority.pt"),
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
    _atomic_json(paths.report, report)
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

#!/usr/bin/env python3
"""Oracle-only exact-placement decomposition for frozen Union-v2 fresh64.

The diagnostic never opens organizer target pixels.  Synthetic permutation
truth is reconstructed solely from the immutable case seed and filename, then
used after the saved target-free layouts/hard projections have been loaded.
It measures ceilings; none of the target-assisted choices is an inference
method or a production candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from aiijc_puzzle import socket_decoder
from aiijc_puzzle.socket_decoder import SocketEdge

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANEL = (
    PROJECT_ROOT / "outputs/raw-twin-union-reranker/frozen-v2-fresh64-draw0"
)
DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs/raw_twin_union_reranker_fresh64_confirmation_v1.json"
)
DEFAULT_OUTPUT = DEFAULT_PANEL / "exact-bottleneck-oracle-v1.json"
GRID = 24
COUNT = GRID * GRID
CURRENT_EDGE_BUDGET = 144
EDGE_BUDGETS = (16, 32, 48, 64, 96, 144, 192, 288, 400, 552)
VARIANT = "learned_union"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _case_seeds(seed: int, filename: str) -> tuple[int, int]:
    digest = hashlib.sha256(f"{seed}\0{filename}\0raw-twin-v2".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63), int.from_bytes(
        digest[8:16], "little"
    ) % (2**63)


def _reference_layout(seed: int, filename: str) -> np.ndarray:
    _, permutation_seed = _case_seeds(seed, filename)
    permutation = np.random.default_rng(permutation_seed).permutation(COUNT)
    return np.ascontiguousarray(np.argsort(permutation), dtype=np.int32)


def _positions(layout: Any) -> np.ndarray:
    value = np.asarray(layout, dtype=np.int32)
    if value.shape != (COUNT,) or not np.array_equal(np.sort(value), np.arange(COUNT)):
        raise ValueError("layout is not a strict 0..575 tile-at-position permutation")
    positions = np.empty((COUNT, 2), dtype=np.int32)
    positions[value, 0], positions[value, 1] = divmod(np.arange(COUNT), GRID)
    return positions


def _normalise_component(
    component: Mapping[int, tuple[int, int]],
) -> dict[int, tuple[int, int]]:
    minimum_row = min(row for row, _ in component.values())
    minimum_column = min(column for _, column in component.values())
    return {
        int(tile): (int(row - minimum_row), int(column - minimum_column))
        for tile, (row, column) in component.items()
    }


def _array(
    archive: Mapping[str, np.ndarray], prefix: str, axis: str, field: str
) -> np.ndarray:
    key = f"{prefix}__{VARIANT}__{axis}__{field}"
    if key not in archive:
        raise KeyError(f"frozen prediction archive is missing {key}")
    return np.asarray(archive[key])


def _decoder_components(
    archive: Mapping[str, np.ndarray], prefix: str, *, edge_budget: int
) -> tuple[tuple[dict[int, tuple[int, int]], ...], Counter[str]]:
    if not 1 <= edge_budget <= COUNT - GRID:
        raise ValueError("edge budget is outside the frozen hard-projection range")
    edges: list[SocketEdge] = []
    for axis, delta_row, delta_column in (("right", 0, 1), ("down", 1, 0)):
        sources = _array(archive, prefix, axis, "sources")
        targets = _array(archive, prefix, axis, "targets")
        confidence = _array(archive, prefix, axis, "confidence")
        if sources.shape != (COUNT - GRID,) or (
            targets.shape != sources.shape or confidence.shape != sources.shape
        ):
            raise ValueError("saved hard projection has an unexpected shape")
        edges.extend(
            SocketEdge(
                source=int(source),
                target=int(target),
                delta_row=delta_row,
                delta_column=delta_column,
                confidence=float(score),
                axis=axis,
            )
            for source, target, score in zip(
                sources[:edge_budget],
                targets[:edge_budget],
                confidence[:edge_budget],
                strict=True,
            )
        )
    edges.sort(
        key=lambda edge: (-edge.confidence, edge.axis, edge.source, edge.target)
    )
    # This intentionally mirrors the production decoder's private component
    # builder.  The saved float32 confidences can only affect a true confidence
    # tie; all scored metric identities are checked against the frozen report.
    builder = socket_decoder._TranslationComponents(count=COUNT, grid=GRID)
    statuses: Counter[str] = Counter()
    for edge in edges:
        statuses[builder.add(edge)] += 1
    components = tuple(
        sorted(
            (_normalise_component(value) for value in builder.complete_components()),
            key=lambda value: (-len(value), min(value)),
        )
    )
    if sorted(tile for component in components for tile in component) != list(
        range(COUNT)
    ):
        raise RuntimeError("rebuilt components do not partition all tile identities")
    return components, statuses


Transform = Callable[[int, int], tuple[int, int]]
D4_TRANSFORMS: dict[str, Transform] = {
    "identity": lambda row, column: (row, column),
    "rotate_90": lambda row, column: (column, -row),
    "rotate_180": lambda row, column: (-row, -column),
    "rotate_270": lambda row, column: (-column, row),
    "reflect_vertical": lambda row, column: (row, -column),
    "reflect_horizontal": lambda row, column: (-row, column),
    "reflect_main_diagonal": lambda row, column: (column, row),
    "reflect_anti_diagonal": lambda row, column: (-column, -row),
}


def _mode(values: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], int]:
    counts = Counter(values)
    if not counts:
        raise ValueError("cannot take a mode of an empty sequence")
    return min(counts.items(), key=lambda item: (-item[1], item[0]))


def _component_modes(
    component: Mapping[int, tuple[int, int]],
    reference_positions: np.ndarray,
) -> tuple[tuple[int, int], tuple[int, ...], str, int]:
    identity_groups: dict[tuple[int, int], list[int]] = {}
    for tile, (row, column) in component.items():
        shift = (
            int(reference_positions[tile, 0] - row),
            int(reference_positions[tile, 1] - column),
        )
        identity_groups.setdefault(shift, []).append(tile)
    identity_shift, identity_members_list = min(
        identity_groups.items(), key=lambda item: (-len(item[1]), item[0])
    )
    identity_members = tuple(sorted(identity_members_list))
    best_name = "identity"
    best_support = len(identity_members)
    for name, transform in D4_TRANSFORMS.items():
        shifts = (
            (
                int(reference_positions[tile, 0] - transform(row, column)[0]),
                int(reference_positions[tile, 1] - transform(row, column)[1]),
            )
            for tile, (row, column) in component.items()
        )
        _, support = _mode(shifts)
        if support > best_support:
            best_name = name
            best_support = support
    return identity_shift, identity_members, best_name, best_support


def _board_d4(board: np.ndarray, name: str) -> np.ndarray:
    transforms: dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "identity": lambda value: value,
        "rotate_90": lambda value: np.rot90(value, -1),
        "rotate_180": lambda value: np.rot90(value, 2),
        "rotate_270": lambda value: np.rot90(value, 1),
        "reflect_vertical": np.fliplr,
        "reflect_horizontal": np.flipud,
        "reflect_main_diagonal": lambda value: value.T,
        "reflect_anti_diagonal": lambda value: np.rot90(value, 2).T,
    }
    return np.ascontiguousarray(transforms[name](board))


def _best_cyclic(
    layout: np.ndarray, reference: np.ndarray, *, allow_d4: bool
) -> tuple[int, str, int, int]:
    board = layout.reshape(GRID, GRID)
    names = tuple(D4_TRANSFORMS) if allow_d4 else ("identity",)
    best = (-1, "identity", 0, 0)
    for name in names:
        transformed = _board_d4(board, name)
        for row_shift in range(GRID):
            for column_shift in range(GRID):
                exact = int(
                    np.count_nonzero(
                        np.roll(
                            transformed,
                            (row_shift, column_shift),
                            axis=(0, 1),
                        ).ravel()
                        == reference
                    )
                )
                candidate = (exact, name, row_shift, column_shift)
                if candidate > best:
                    best = candidate
    return best


class _UnionFind:
    def __init__(self) -> None:
        self.parent = np.arange(COUNT, dtype=np.int32)
        self.size = np.ones(COUNT, dtype=np.int32)

    def find(self, value: int) -> int:
        root = value
        while int(self.parent[root]) != root:
            root = int(self.parent[root])
        while value != root:
            parent = int(self.parent[value])
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: int, right: int) -> None:
        left = self.find(left)
        right = self.find(right)
        if left == right:
            return
        if int(self.size[left]) < int(self.size[right]):
            left, right = right, left
        self.parent[right] = left
        self.size[left] += self.size[right]

    def statistics(self) -> dict[str, int]:
        sizes = Counter(self.find(tile) for tile in range(COUNT)).values()
        ordered = sorted(sizes, reverse=True)
        return {
            "component_count": len(ordered),
            "largest_component_tiles": ordered[0],
            "tiles_in_nontrivial_components": sum(size for size in ordered if size > 1),
            "nontrivial_component_count": sum(size > 1 for size in ordered),
        }


def _truth_by_anchor(reference: np.ndarray, axis: str) -> np.ndarray:
    positions = np.arange(COUNT)
    valid = positions % GRID != GRID - 1 if axis == "right" else positions < COUNT - GRID
    delta = 1 if axis == "right" else GRID
    truth = np.full(COUNT, -1, dtype=np.int32)
    truth[reference[positions[valid]]] = reference[positions[valid] + delta]
    return truth


def _oracle_candidate_graphs(
    archive: Mapping[str, np.ndarray], prefix: str, reference: np.ndarray
) -> dict[str, dict[str, int]]:
    forests = {name: _UnionFind() for name in ("hard_top48", "hard_all552", "top5")}
    correct = Counter()
    for axis in ("right", "down"):
        truth = _truth_by_anchor(reference, axis)
        sources = _array(archive, prefix, axis, "sources")
        targets = _array(archive, prefix, axis, "targets")
        for budget, name in ((48, "hard_top48"), (COUNT - GRID, "hard_all552")):
            for source, target in zip(
                sources[:budget], targets[:budget], strict=True
            ):
                if int(truth[int(source)]) == int(target):
                    forests[name].union(int(source), int(target))
                    correct[name] += 1
        candidates = _array(archive, prefix, axis, "candidates")
        if candidates.shape != (COUNT, 5):
            raise ValueError("saved local candidate array is not 576x5")
        for source in range(COUNT):
            target = int(truth[source])
            if target >= 0 and target in candidates[source]:
                forests["top5"].union(source, target)
                correct["top5"] += 1
    return {
        name: forests[name].statistics() | {"oracle_correct_edges": correct[name]}
        for name in forests
    }


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _metric_summary(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, float | int]:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return {
        "mean_tiles_per_board": float(values.mean()),
        "mean_fraction": float(values.mean() / COUNT),
        "total_tiles": int(values.sum()),
        "minimum": int(values.min()),
        "maximum": int(values.max()),
    }


def _averaged_graphs(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for name in ("hard_top48", "hard_all552", "top5"):
        values = [row["oracle_candidate_graphs"][name] for row in rows]
        output[name] = {
            key: float(np.mean([float(value[key]) for value in values]))
            for key in values[0]
        }
    return output


def run(args: argparse.Namespace) -> None:
    panel = args.panel_dir.resolve()
    config_path = args.config.resolve()
    prediction_path = panel / "frozen-target-free-predictions.npz"
    metadata_path = panel / "frozen-target-free-predictions.json"
    frozen_report_path = panel / "report.json"
    paths = (config_path, prediction_path, metadata_path, frozen_report_path)
    if not all(path.is_file() for path in paths):
        missing = [str(path) for path in paths if not path.is_file()]
        raise FileNotFoundError(f"missing frozen inputs: {missing}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    frozen_report = json.loads(frozen_report_path.read_text(encoding="utf-8"))
    cases = metadata.get("cases")
    if not isinstance(cases, list) or len(cases) != 64:
        raise ValueError("expected the frozen fresh64 case roster")
    if metadata.get("contains_exact_references") is not False:
        raise ValueError("frozen metadata reference contract changed")
    seed = int(config["selection"]["synthetic_seed"])
    board_rows: list[dict[str, Any]] = []
    sweep_rows: dict[int, list[dict[str, Any]]] = {
        budget: [] for budget in EDGE_BUDGETS
    }
    with np.load(prediction_path) as archive:
        for case in cases:
            prefix = str(case["prefix"])
            filename = str(case["source_filename"])
            reference = _reference_layout(seed, filename)
            layout = np.ascontiguousarray(
                archive[f"{prefix}__{VARIANT}__layout"], dtype=np.int32
            )
            reference_positions = _positions(reference)
            predicted_positions = _positions(layout)
            current_exact = int(np.count_nonzero(layout == reference))
            current_components, statuses = _decoder_components(
                archive, prefix, edge_budget=CURRENT_EDGE_BUDGET
            )
            modal_tiles: list[int] = []
            identity_support = 0
            d4_support = 0
            pure_tiles = 0
            pure_nontrivial_tiles = 0
            pure_nontrivial_components = 0
            d4_component_gains = 0
            best_transform_by_component: Counter[str] = Counter()
            for component in current_components:
                _, members, best_transform, best_support = _component_modes(
                    component, reference_positions
                )
                modal_tiles.extend(members)
                identity_support += len(members)
                d4_support += best_support
                d4_component_gains += best_support - len(members)
                best_transform_by_component[best_transform] += 1
                if len(members) == len(component):
                    pure_tiles += len(component)
                    if len(component) > 1:
                        pure_nontrivial_tiles += len(component)
                        pure_nontrivial_components += 1
            modal_errors = Counter(
                (
                    int(predicted_positions[tile, 0] - reference_positions[tile, 0])
                    % GRID,
                    int(predicted_positions[tile, 1] - reference_positions[tile, 1])
                    % GRID,
                )
                for tile in modal_tiles
            )
            best_modal_error, best_modal_global = min(
                modal_errors.items(), key=lambda item: (-item[1], item[0])
            )
            best_cyclic = _best_cyclic(layout, reference, allow_d4=False)
            best_d4_cyclic = _best_cyclic(layout, reference, allow_d4=True)
            graph_rows = _oracle_candidate_graphs(archive, prefix, reference)
            board_rows.append(
                {
                    "source_filename": filename,
                    "current_exact_tiles": current_exact,
                    "component_count": len(current_components),
                    "nontrivial_component_count": sum(
                        len(component) > 1 for component in current_components
                    ),
                    "largest_component_tiles": max(map(len, current_components)),
                    "component_builder_statuses": dict(statuses),
                    "identity_rigid_translation_oracle_tiles": identity_support,
                    "d4_rigid_translation_oracle_tiles": d4_support,
                    "d4_gain_tiles": d4_component_gains,
                    "best_d4_transform_component_counts": dict(
                        best_transform_by_component
                    ),
                    "pure_component_tiles_including_singletons": pure_tiles,
                    "pure_nontrivial_component_tiles": pure_nontrivial_tiles,
                    "pure_nontrivial_component_count": pure_nontrivial_components,
                    "modal_tiles_currently_exact": modal_errors[(0, 0)],
                    "modal_tiles_best_single_cyclic_origin": best_modal_global,
                    "best_modal_position_error": list(best_modal_error),
                    "correcting_modal_cyclic_shift": [
                        (-best_modal_error[0]) % GRID,
                        (-best_modal_error[1]) % GRID,
                    ],
                    "whole_layout_best_cyclic_exact": best_cyclic[0],
                    "whole_layout_best_cyclic_shift": [best_cyclic[2], best_cyclic[3]],
                    "whole_layout_best_d4_cyclic_exact": best_d4_cyclic[0],
                    "whole_layout_best_d4_transform": best_d4_cyclic[1],
                    "whole_layout_best_d4_cyclic_shift": [
                        best_d4_cyclic[2],
                        best_d4_cyclic[3],
                    ],
                    "oracle_candidate_graphs": graph_rows,
                }
            )
            for budget in EDGE_BUDGETS:
                components, _ = _decoder_components(
                    archive, prefix, edge_budget=budget
                )
                modal_support = 0
                budget_pure_tiles = 0
                budget_pure_nontrivial_tiles = 0
                budget_pure_nontrivial_components = 0
                for component in components:
                    _, members, _, _ = _component_modes(component, reference_positions)
                    modal_support += len(members)
                    if len(members) == len(component):
                        budget_pure_tiles += len(component)
                        if len(component) > 1:
                            budget_pure_nontrivial_tiles += len(component)
                            budget_pure_nontrivial_components += 1
                correct_edges = 0
                for axis in ("right", "down"):
                    truth = _truth_by_anchor(reference, axis)
                    sources = _array(archive, prefix, axis, "sources")[:budget]
                    targets = _array(archive, prefix, axis, "targets")[:budget]
                    correct_edges += int(
                        np.count_nonzero(truth[sources.astype(np.int64)] == targets)
                    )
                sweep_rows[budget].append(
                    {
                        "component_count": len(components),
                        "nontrivial_component_count": sum(
                            len(component) > 1 for component in components
                        ),
                        "largest_component_tiles": max(map(len, components)),
                        "identity_rigid_translation_oracle_tiles": modal_support,
                        "pure_component_tiles_including_singletons": budget_pure_tiles,
                        "pure_nontrivial_component_tiles": budget_pure_nontrivial_tiles,
                        "pure_nontrivial_component_count": (
                            budget_pure_nontrivial_components
                        ),
                        "correct_hard_edges": correct_edges,
                    }
                )

    frozen_rows = frozen_report["rows"]
    expected_exact = float(
        frozen_report["metrics"]["arms"]["exact_tiles"]["learned_union_mean"]
    )
    observed_exact = _mean(board_rows, "current_exact_tiles")
    report_row_exact = float(
        np.mean([row[VARIANT]["exact_tiles"] for row in frozen_rows])
    )
    if observed_exact != expected_exact or observed_exact != report_row_exact:
        raise RuntimeError("reconstructed seed truth does not reproduce frozen exact metric")

    identity_mean = _mean(board_rows, "identity_rigid_translation_oracle_tiles")
    d4_mean = _mean(board_rows, "d4_rigid_translation_oracle_tiles")
    modal_current_mean = _mean(board_rows, "modal_tiles_currently_exact")
    modal_global_mean = _mean(
        board_rows, "modal_tiles_best_single_cyclic_origin"
    )
    budget_sweep: list[dict[str, Any]] = []
    for budget in EDGE_BUDGETS:
        rows = sweep_rows[budget]
        correct = _mean(rows, "correct_hard_edges")
        budget_sweep.append(
            {
                "hard_edges_per_axis": budget,
                "hard_edge_precision": correct / (2 * budget),
                "correct_hard_edges_per_board": correct,
                "component_count": _mean(rows, "component_count"),
                "nontrivial_component_count": _mean(
                    rows, "nontrivial_component_count"
                ),
                "largest_component_tiles_mean": _mean(
                    rows, "largest_component_tiles"
                ),
                "largest_component_tiles_maximum": max(
                    int(row["largest_component_tiles"]) for row in rows
                ),
                "identity_rigid_translation_oracle_tiles": _mean(
                    rows, "identity_rigid_translation_oracle_tiles"
                ),
                "identity_rigid_translation_oracle_fraction": _mean(
                    rows, "identity_rigid_translation_oracle_tiles"
                )
                / COUNT,
                "pure_component_tiles_including_singletons": _mean(
                    rows, "pure_component_tiles_including_singletons"
                ),
                "pure_nontrivial_component_tiles": _mean(
                    rows, "pure_nontrivial_component_tiles"
                ),
                "pure_nontrivial_component_count": _mean(
                    rows, "pure_nontrivial_component_count"
                ),
            }
        )
    knee48 = next(
        row for row in budget_sweep if row["hard_edges_per_axis"] == 48
    )
    payload = {
        "schema": "aiijc-union-v2-fresh64-exact-bottleneck-oracle-v1",
        "status": "diagnostic-complete-not-an-inference-candidate",
        "provenance": {
            "config": str(config_path.relative_to(PROJECT_ROOT)),
            "config_sha256": _sha256(config_path),
            "frozen_prediction_npz": str(prediction_path.relative_to(PROJECT_ROOT)),
            "frozen_prediction_npz_sha256": _sha256(prediction_path),
            "frozen_metadata": str(metadata_path.relative_to(PROJECT_ROOT)),
            "frozen_metadata_sha256": _sha256(metadata_path),
            "frozen_report": str(frozen_report_path.relative_to(PROJECT_ROOT)),
            "frozen_report_sha256": _sha256(frozen_report_path),
            "source_count": len(board_rows),
            "synthetic_reference_recovery": (
                "argsort(default_rng(frozen permutation seed).permutation(576)); "
                "no image pixels"
            ),
            "organizer_target_pixels_opened": False,
            "competition_test_opened": False,
            "target_assisted_results_are_oracle_diagnostics_only": True,
        },
        "definitions": {
            "internal_rigid_translation_oracle": (
                "For each decoder component independently, retain the largest tile "
                "subset sharing one exact translation from predicted relative "
                "coordinates to truth. Components may overlap after their separate "
                "oracle shifts, so this is an optimistic ceiling, not a strict layout."
            ),
            "d4_rigid_translation_oracle": (
                "The same optimistic component-wise ceiling after allowing one of "
                "eight coordinate D4 transforms before translation. Tiles themselves "
                "are not rotated; this only diagnoses arrangement orientation."
            ),
            "global_origin_oracle": (
                "Choose one toroidal row/column shift for the already-decoded whole "
                "layout. Modal-only counts restrict evaluation to tiles recoverable by "
                "the internal rigid-translation oracle."
            ),
            "relative_component_placement_gap": (
                "Internal rigid-translation ceiling minus the best modal support "
                "recoverable by one shared cyclic origin. It is the dominant gap left "
                "after internal geometry and global origin are separately granted."
            ),
        },
        "validation": {
            "reconstructed_current_exact_mean": observed_exact,
            "frozen_report_current_exact_mean": expected_exact,
            "exact_metric_identity_pass": True,
            "all_saved_layouts_strict": True,
        },
        "current_decoder144": {
            "current_exact": _metric_summary(board_rows, "current_exact_tiles"),
            "component_count_mean": _mean(board_rows, "component_count"),
            "nontrivial_component_count_mean": _mean(
                board_rows, "nontrivial_component_count"
            ),
            "largest_component_tiles_mean": _mean(
                board_rows, "largest_component_tiles"
            ),
            "identity_rigid_translation_oracle": _metric_summary(
                board_rows, "identity_rigid_translation_oracle_tiles"
            ),
            "internal_geometry_loss_tiles_per_board": COUNT - identity_mean,
            "internal_geometry_loss_fraction": (COUNT - identity_mean) / COUNT,
            "d4_rigid_translation_oracle": _metric_summary(
                board_rows, "d4_rigid_translation_oracle_tiles"
            ),
            "rotation_reflection_gain_tiles_per_board": d4_mean - identity_mean,
            "rotation_reflection_gain_fraction": (d4_mean - identity_mean) / COUNT,
            "boards_with_positive_component_d4_gain": sum(
                int(row["d4_gain_tiles"]) > 0 for row in board_rows
            ),
            "pure_component_tiles_including_singletons": _metric_summary(
                board_rows, "pure_component_tiles_including_singletons"
            ),
            "pure_nontrivial_component_tiles": _metric_summary(
                board_rows, "pure_nontrivial_component_tiles"
            ),
            "pure_nontrivial_component_count_mean": _mean(
                board_rows, "pure_nontrivial_component_count"
            ),
            "modal_tiles_currently_exact_per_board": modal_current_mean,
            "modal_tiles_best_single_cyclic_origin_per_board": modal_global_mean,
            "global_origin_recoverable_gain_tiles_per_board": (
                modal_global_mean - modal_current_mean
            ),
            "global_origin_recoverable_gain_fraction": (
                modal_global_mean - modal_current_mean
            )
            / COUNT,
            "relative_component_placement_gap_tiles_per_board": (
                identity_mean - modal_global_mean
            ),
            "relative_component_placement_gap_fraction": (
                identity_mean - modal_global_mean
            )
            / COUNT,
            "current_exact_outside_dominant_component_modes_per_board": (
                observed_exact - modal_current_mean
            ),
            "whole_layout_best_cyclic_exact": _metric_summary(
                board_rows, "whole_layout_best_cyclic_exact"
            ),
            "whole_layout_best_d4_cyclic_exact": _metric_summary(
                board_rows, "whole_layout_best_d4_cyclic_exact"
            ),
            "whole_layout_best_d4_transform_counts": dict(
                Counter(
                    str(row["whole_layout_best_d4_transform"]) for row in board_rows
                )
            ),
        },
        "hard_component_budget_sweep": budget_sweep,
        "oracle_candidate_connectivity": _averaged_graphs(board_rows),
        "next_high_signal_placer_experiment": {
            "name": "top48-fragment robust 2D coordinate synchronization",
            "why_top48": {
                "hard_edge_precision": knee48["hard_edge_precision"],
                "identity_rigid_translation_oracle_tiles": knee48[
                    "identity_rigid_translation_oracle_tiles"
                ],
                "identity_rigid_translation_oracle_fraction": knee48[
                    "identity_rigid_translation_oracle_fraction"
                ],
                "pure_nontrivial_component_tiles": knee48[
                    "pure_nontrivial_component_tiles"
                ],
                "component_count": knee48["component_count"],
                "interpretation": (
                    "This is the sweep knee: many more internally recoverable tiles "
                    "than budget144 while retaining the maximum pure nontrivial tile "
                    "support among tested practical budgets."
                ),
            },
            "algorithm": [
                (
                    "Use only the top48 projected edges per axis as irreversible "
                    "rigid-fragment constraints."
                ),
                (
                    "Convert every remaining raw32/twin32 Union candidate into a "
                    "soft equation between two fragment translations; aggregate "
                    "duplicate component-pair displacement hypotheses."
                ),
                (
                    "Solve the two integer coordinates jointly on Z_24 with robust "
                    "max-consensus/cycle consistency; do not greedily merge a relation "
                    "and do not predict an absolute coordinate from isolated pixels."
                ),
                (
                    "Turn synchronized component-offset hypotheses into tile-to-slot "
                    "unaries, enforce one tile per slot with global linear assignment, "
                    "then apply the existing bounded pair-energy polish."
                ),
                "Choose only the final shared origin with the already-frozen border5 scorer.",
            ],
            "distinction_from_closed_lines": (
                "Unlike the failed absolute component head it learns/solves only "
                "relative translations. Unlike the failed relation forest, edges "
                "below top48 remain reversible soft factors and are decided jointly "
                "through cycles before any strict packing."
            ),
            "first_bounded_gate": (
                "Freeze one no-sweep implementation on a source-disjoint train64. "
                "Continue if mean exact improves by at least 0.25 tile/board and "
                "adjacency is nonnegative versus the identical Union-v2 decoder144 "
                "arm; otherwise stop this formulation."
            ),
        },
        "board_rows": board_rows,
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(output),
                "sha256": _sha256(output),
                "current_exact": observed_exact,
                "internal_oracle": identity_mean,
                "best_global_modal": modal_global_mean,
                "relative_placement_gap": identity_mean - modal_global_mean,
                "top48_internal_oracle": knee48[
                    "identity_rigid_translation_oracle_tiles"
                ],
            }
        )
    )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

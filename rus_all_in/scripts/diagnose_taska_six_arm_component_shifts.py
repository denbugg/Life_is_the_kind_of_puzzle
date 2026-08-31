#!/usr/bin/env python3
"""Measure exact-placement headroom in frozen six-arm TASKA layouts.

This is an explicitly target-assisted organizer-train diagnostic.  It never
produces or selects a deployable layout.  Components are defined only from
target-blind focal-kept candidate edges that the frozen six-arm final layout
actually realises.  Exact synthetic references are then reconstructed to
measure global-translation and independent-component translation ceilings.
Competition-test inputs are not accepted by this script.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from aiijc_puzzle.layout_evaluation import evaluate_layout
from aiijc_puzzle.protocol import sha256_file

try:
    from scripts import run_taska_focal_current_finetune as finetune
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    import run_taska_focal_current_finetune as finetune


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRID = 24
COUNT = GRID * GRID
PAIR_DENOMINATOR = 2 * GRID * (GRID - 1)
DEFAULT_ARCHIVE = PROJECT_ROOT / (
    "outputs/taska-selective-fullres-union-fusion/fixed-v1/local32/"
    "frozen-target-free-eval.npz"
)
DEFAULT_METADATA = DEFAULT_ARCHIVE.with_suffix(".json")
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
DEFAULT_OUTPUT = PROJECT_ROOT / (
    "outputs/taska-six-arm-component-shift-diagnostic/local32-v1/report.json"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def _strict_layout(value: Any) -> np.ndarray:
    layout = np.ascontiguousarray(value, dtype=np.int32)
    if layout.shape != (COUNT,) or not np.array_equal(np.sort(layout), np.arange(COUNT)):
        raise ValueError("layout must be a strict 576-tile permutation")
    return layout


def _edge_arrays(
    archive: Any,
    prefix: str,
    name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source = np.asarray(archive[f"{prefix}__{name}__edge_source"], dtype=np.int32)
    target = np.asarray(archive[f"{prefix}__{name}__edge_target"], dtype=np.int32)
    axis = np.asarray(archive[f"{prefix}__{name}__edge_axis"], dtype=np.uint8)
    logits = np.asarray(archive[f"{prefix}__{name}_focal_logits"], dtype=np.float64)
    if not (source.shape == target.shape == axis.shape == logits.shape):
        raise ValueError("edge identity/logit arrays are not aligned")
    if source.ndim != 1 or not np.isin(axis, (0, 1)).all() or not np.isfinite(logits).all():
        raise ValueError("edge arrays are malformed")
    return source, target, axis, logits


def _selected_kept_edges(
    archive: Any,
    prefix: str,
    choice: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if choice == "combined_union_focal":
        names = ("combined_union",)
    elif choice == "selective_vote500_focal":
        names = ("current", "selective_new")
    else:
        names = ("current",)
    arrays = [_edge_arrays(archive, prefix, name) for name in names]
    source = np.concatenate([item[0] for item in arrays])
    target = np.concatenate([item[1] for item in arrays])
    axis = np.concatenate([item[2] for item in arrays])
    logits = np.concatenate([item[3] for item in arrays])
    keep = logits >= 0.0
    return source[keep], target[keep], axis[keep], logits[keep]


class _DisjointSet:
    def __init__(self, count: int) -> None:
        self.parent = np.arange(count, dtype=np.int32)
        self.size = np.ones(count, dtype=np.int32)

    def find(self, value: int) -> int:
        root = value
        while int(self.parent[root]) != root:
            root = int(self.parent[root])
        while int(self.parent[value]) != value:
            parent = int(self.parent[value])
            self.parent[value] = root
            value = parent
        return root

    def union(self, first: int, second: int) -> None:
        a, b = self.find(first), self.find(second)
        if a == b:
            return
        if int(self.size[a]) < int(self.size[b]):
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]


def _visible_components(
    layout: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    axis: np.ndarray,
    logits: np.ndarray,
) -> tuple[tuple[int, ...], ...]:
    position = np.empty(COUNT, dtype=np.int32)
    position[layout] = np.arange(COUNT, dtype=np.int32)
    rows, columns = divmod(position, GRID)
    realised = np.where(
        (axis == 0)
        & (rows[source] == rows[target])
        & (columns[target] == columns[source] + 1)
        | (axis == 1)
        & (columns[source] == columns[target])
        & (rows[target] == rows[source] + 1)
    )[0]
    graph = _DisjointSet(COUNT)
    for index in realised:
        graph.union(int(source[index]), int(target[index]))
    groups: dict[int, list[int]] = defaultdict(list)
    for tile in range(COUNT):
        groups[graph.find(tile)].append(tile)
    return tuple(
        tuple(values)
        for values in sorted(groups.values(), key=lambda values: (-len(values), values[0]))
    )


def _component_shifts(
    layout: np.ndarray,
    reference: np.ndarray,
    component: Sequence[int],
) -> tuple[Counter[tuple[int, int]], tuple[int, int, int, int]]:
    predicted_position = np.empty(COUNT, dtype=np.int32)
    reference_position = np.empty(COUNT, dtype=np.int32)
    predicted_position[layout] = np.arange(COUNT, dtype=np.int32)
    reference_position[reference] = np.arange(COUNT, dtype=np.int32)
    tiles = np.asarray(component, dtype=np.int32)
    predicted_rows, predicted_columns = divmod(predicted_position[tiles], GRID)
    reference_rows, reference_columns = divmod(reference_position[tiles], GRID)
    shifts = Counter(
        zip(
            (reference_rows - predicted_rows).tolist(),
            (reference_columns - predicted_columns).tolist(),
            strict=True,
        )
    )
    bounds = (
        int(predicted_rows.min()),
        int(predicted_rows.max()),
        int(predicted_columns.min()),
        int(predicted_columns.max()),
    )
    return shifts, bounds


def _feasible_shifts(bounds: tuple[int, int, int, int]) -> tuple[tuple[int, int], ...]:
    minimum_row, maximum_row, minimum_column, maximum_column = bounds
    return tuple(
        (row_shift, column_shift)
        for row_shift in range(-minimum_row, GRID - maximum_row)
        for column_shift in range(-minimum_column, GRID - maximum_column)
    )


def _translate_component_with_local_fill(
    layout: np.ndarray,
    component: Sequence[int],
    shift: tuple[int, int],
) -> np.ndarray:
    """Rigidly move one component and relocate only directly displaced tiles."""

    row_shift, column_shift = shift
    position = np.empty(COUNT, dtype=np.int32)
    position[layout] = np.arange(COUNT, dtype=np.int32)
    tiles = np.asarray(component, dtype=np.int32)
    old_positions = position[tiles]
    old_rows, old_columns = divmod(old_positions, GRID)
    new_rows = old_rows + row_shift
    new_columns = old_columns + column_shift
    if (
        np.any(new_rows < 0)
        or np.any(new_rows >= GRID)
        or np.any(new_columns < 0)
        or np.any(new_columns >= GRID)
    ):
        raise ValueError("component shift is not feasible")
    new_positions = new_rows * GRID + new_columns
    if len(np.unique(new_positions)) != len(tiles):
        raise RuntimeError("rigid shift created an internal collision")
    old_set = set(int(value) for value in old_positions)
    new_set = set(int(value) for value in new_positions)
    vacated = sorted(old_set - new_set)
    entered = sorted(new_set - old_set)
    displaced = [int(layout[value]) for value in entered]
    result = layout.copy()
    for tile, new_position in zip(tiles, new_positions, strict=True):
        result[int(new_position)] = int(tile)
    for destination, tile in zip(vacated, displaced, strict=True):
        result[destination] = tile
    return _strict_layout(result)


def _metrics(layout: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    result = evaluate_layout(layout, reference, reference_is_exact=True)
    return {
        "exact_tiles": int(result.correct_tile_count),
        "satisfied_adjacent_pairs": int(result.adjacency_correct),
        "adjacency_recall": float(result.adjacency),
        "translation_aligned_tiles": int(result.translation_aligned_count),
    }


def _best_cyclic(layout: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    board = layout.reshape(GRID, GRID)
    best_layout = layout
    best_roll = (0, 0)
    best_key = (-1, -1)
    for row_roll in range(GRID):
        for column_roll in range(GRID):
            candidate = np.roll(board, (row_roll, column_roll), axis=(0, 1)).reshape(-1)
            metrics = evaluate_layout(candidate, reference, reference_is_exact=True)
            key = (metrics.correct_tile_count, metrics.adjacency_correct)
            if key > best_key:
                best_key = key
                best_layout = np.ascontiguousarray(candidate, dtype=np.int32)
                best_roll = (row_roll, column_roll)
    return best_layout, best_roll


def _case_diagnostic(
    *,
    layout: np.ndarray,
    reference: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    axis: np.ndarray,
    logits: np.ndarray,
) -> dict[str, Any]:
    baseline = _metrics(layout, reference)
    cyclic_layout, cyclic_roll = _best_cyclic(layout, reference)
    components = _visible_components(layout, source, target, axis, logits)
    component_rows: list[dict[str, Any]] = []
    best_move_layout = layout
    best_move_key = (baseline["exact_tiles"], baseline["satisfied_adjacent_pairs"])
    best_move: dict[str, Any] | None = None
    pure_tile_ceiling = 0
    dominant_feasible_ceiling = 0
    nontrivial_pure_tile_ceiling = 0
    nontrivial_dominant_feasible_ceiling = 0
    for component_index, component in enumerate(components):
        shifts, bounds = _component_shifts(layout, reference, component)
        feasible = _feasible_shifts(bounds)
        best_support = max((shifts.get(shift, 0) for shift in feasible), default=0)
        dominant_feasible_ceiling += best_support
        if best_support == len(component):
            pure_tile_ceiling += len(component)
        if len(component) >= 2:
            nontrivial_dominant_feasible_ceiling += best_support
            if best_support == len(component):
                nontrivial_pure_tile_ceiling += len(component)
        component_best_key = best_move_key
        component_best_shift = (0, 0)
        component_best_metrics = baseline
        if len(component) >= 2:
            for shift in feasible:
                if shift == (0, 0):
                    candidate = layout
                    metrics = baseline
                else:
                    candidate = _translate_component_with_local_fill(layout, component, shift)
                    metrics = _metrics(candidate, reference)
                key = (metrics["exact_tiles"], metrics["satisfied_adjacent_pairs"])
                if key > component_best_key:
                    component_best_key = key
                    component_best_shift = shift
                    component_best_metrics = metrics
                    if key > best_move_key:
                        best_move_key = key
                        best_move_layout = candidate
                        best_move = {
                            "component_index": component_index,
                            "component_size": len(component),
                            "shift": list(shift),
                        }
        component_rows.append(
            {
                "component_index": component_index,
                "size": len(component),
                "bounds": list(bounds),
                "feasible_shift_count": len(feasible),
                "dominant_feasible_support": int(best_support),
                "dominant_feasible_purity": float(best_support / len(component)),
                "zero_shift_support": int(shifts.get((0, 0), 0)),
                "oracle_local_fill_shift": list(component_best_shift),
                "oracle_local_fill_exact": int(component_best_metrics["exact_tiles"]),
                "oracle_local_fill_pairs": int(
                    component_best_metrics["satisfied_adjacent_pairs"]
                ),
            }
        )
    return {
        "baseline": baseline,
        "visible_component_count": len(components),
        "nontrivial_component_count": sum(len(component) >= 2 for component in components),
        "largest_component_size": len(components[0]),
        "pure_component_tile_ceiling": pure_tile_ceiling,
        "independent_dominant_shift_tile_ceiling": dominant_feasible_ceiling,
        "nontrivial_pure_component_tile_ceiling": nontrivial_pure_tile_ceiling,
        "nontrivial_independent_dominant_shift_tile_ceiling": (
            nontrivial_dominant_feasible_ceiling
        ),
        "oracle_best_cyclic": {
            **_metrics(cyclic_layout, reference),
            "roll": list(cyclic_roll),
        },
        "oracle_best_single_component_local_fill": {
            **_metrics(best_move_layout, reference),
            "move": best_move,
        },
        "components": component_rows,
    }


def _mean(rows: Sequence[Mapping[str, Any]], path: Sequence[str]) -> float:
    values: list[float] = []
    for row in rows:
        value: Any = row
        for key in path:
            value = value[key]
        values.append(float(value))
    return float(np.mean(values))


def run(args: argparse.Namespace) -> Path:
    archive_path = args.archive.resolve()
    metadata_path = args.metadata.resolve()
    targets = args.targets.resolve()
    output = args.output.resolve()
    if not archive_path.is_file() or not metadata_path.is_file():
        raise ValueError("frozen six-arm archive/metadata is absent")
    if not targets.is_dir() or targets.name != "targets" or targets.parent.name != "train":
        raise ValueError("only organizer-train targets are accepted")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    frozen_rows = payload.get("rows")
    if not isinstance(frozen_rows, list) or len(frozen_rows) != 32:
        raise ValueError("expected the already-opened local32 roster")
    config, _, _ = finetune._load_config(finetune.DEFAULT_CONFIG)
    lookup = finetune._manifest_lookup(config)
    cache = finetune.CleanTileCache(targets, maximum_boards=2)
    rows: list[dict[str, Any]] = []
    with np.load(archive_path, allow_pickle=False) as archive:
        for index, frozen in enumerate(frozen_rows):
            prefix = str(frozen["prefix"])
            source_name = str(frozen["source_filename"])
            draw = int(frozen["draw_index"])
            dirty = finetune._dirty_case(cache, lookup[source_name], source_name, draw)
            if finetune._dirty_sha256(dirty.dirty_tiles) != frozen["dirty_sha256"]:
                raise RuntimeError("dirty case recreation differs from frozen metadata")
            reference = finetune._reference(
                cache,
                lookup[source_name],
                source_name,
                draw,
                dirty.dirty_tiles,
            )
            layout = _strict_layout(archive[f"{prefix}__combined_union_candidate_layout"])
            edge_arrays = _selected_kept_edges(archive, prefix, str(frozen["choice"]))
            diagnostic = _case_diagnostic(
                layout=layout,
                reference=reference,
                source=edge_arrays[0],
                target=edge_arrays[1],
                axis=edge_arrays[2],
                logits=edge_arrays[3],
            )
            rows.append(
                {
                    "prefix": prefix,
                    "source_filename": source_name,
                    "draw_index": draw,
                    "choice": frozen["choice"],
                    **diagnostic,
                }
            )
            print(
                json.dumps(
                    {
                        "case": index + 1,
                        "baseline_exact": diagnostic["baseline"]["exact_tiles"],
                        "cyclic_exact": diagnostic["oracle_best_cyclic"]["exact_tiles"],
                        "single_component_exact": diagnostic[
                            "oracle_best_single_component_local_fill"
                        ]["exact_tiles"],
                    }
                ),
                flush=True,
            )
    report = {
        "schema": "aiijc-taska-six-arm-component-shift-diagnostic-v1",
        "status": "target-assisted-opened-local32-diagnostic-only",
        "panel": "already-opened-local32",
        "case_count": len(rows),
        "component_definition": (
            "connected components of focal-logit>=0 selected-supply edges realised "
            "by the frozen six-arm final layout; singleton tiles retained"
        ),
        "summary": {
            "baseline_exact_tiles_per_board": _mean(rows, ("baseline", "exact_tiles")),
            "baseline_pairs_per_board": _mean(
                rows, ("baseline", "satisfied_adjacent_pairs")
            ),
            "baseline_translation_aligned_tiles_per_board": _mean(
                rows, ("baseline", "translation_aligned_tiles")
            ),
            "oracle_best_cyclic_exact_tiles_per_board": _mean(
                rows, ("oracle_best_cyclic", "exact_tiles")
            ),
            "oracle_best_cyclic_pairs_per_board": _mean(
                rows, ("oracle_best_cyclic", "satisfied_adjacent_pairs")
            ),
            "oracle_best_single_component_local_fill_exact_tiles_per_board": _mean(
                rows,
                ("oracle_best_single_component_local_fill", "exact_tiles"),
            ),
            "oracle_best_single_component_local_fill_pairs_per_board": _mean(
                rows,
                ("oracle_best_single_component_local_fill", "satisfied_adjacent_pairs"),
            ),
            "pure_component_tile_ceiling_per_board": _mean(
                rows, ("pure_component_tile_ceiling",)
            ),
            "independent_dominant_shift_tile_ceiling_per_board": _mean(
                rows, ("independent_dominant_shift_tile_ceiling",)
            ),
            "nontrivial_pure_component_tile_ceiling_per_board": _mean(
                rows, ("nontrivial_pure_component_tile_ceiling",)
            ),
            "nontrivial_independent_dominant_shift_tile_ceiling_per_board": _mean(
                rows, ("nontrivial_independent_dominant_shift_tile_ceiling",)
            ),
            "visible_component_count_per_board": _mean(rows, ("visible_component_count",)),
            "nontrivial_component_count_per_board": _mean(
                rows, ("nontrivial_component_count",)
            ),
            "largest_component_size_per_board": _mean(rows, ("largest_component_size",)),
        },
        "interpretation_boundary": {
            "targets_used_only_for_offline_diagnostic": True,
            "deployable_selector_or_layout_produced": False,
            "competition_test_accessed": False,
            "all_input_layouts_strict_original_upright_permutations": True,
        },
        "inputs": {
            "archive": {"path": str(archive_path), "sha256": sha256_file(archive_path)},
            "metadata": {"path": str(metadata_path), "sha256": sha256_file(metadata_path)},
        },
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {output}") from error
    print(json.dumps(report["summary"], indent=2))
    return output


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

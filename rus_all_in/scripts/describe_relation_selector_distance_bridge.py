#!/usr/bin/env python3
"""Describe position-distance and SSIM for the frozen relation confirmation.

This is a post-hoc measurement bridge, not a selector, gate, or new inference
run.  It reuses the two layouts already frozen for every formal-confirmation
case and reconstructs exact organizer-train references only for evaluation.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from aiijc_puzzle.layout_evaluation import evaluate_layout
from aiijc_puzzle.protocol import assemble_tiles, contest_ssim, sha256_file
from aiijc_puzzle.socket_pixel_tails import historical_rgb_luma_nlm_h20_once
from aiijc_puzzle.synthetic_socket_evaluation import make_exact_synthetic_case
from aiijc_puzzle.taska_selective_fullres_fusion import strict_layout
from aiijc_puzzle.tile_position_distance import evaluate_tile_position_distance

try:
    from scripts import run_taska_protected_tail_fresh32_confirmation as synthetic
    from scripts import run_taska_relation_truth_selector_confirmation as relation
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    import run_taska_protected_tail_fresh32_confirmation as synthetic
    import run_taska_relation_truth_selector_confirmation as relation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = (
    PROJECT_ROOT
    / "outputs/taska-relation-truth-selector/formal-confirmation-v1/"
    "frozen-target-free-eval.npz"
)
DEFAULT_METADATA = DEFAULT_ARCHIVE.with_suffix(".json")
DEFAULT_PARENT_REPORT = DEFAULT_ARCHIVE.parent / "report.json"
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs/tile-position-distance-validation/relation-selector-bridge-v1"
)
EXPECTED_SHA256 = {
    "archive": "4cd0346333813cea3576f6db40ea517dcc45fdd5aa81a432a351cf4afdd73131",
    "metadata": "4ae4b8f27d3d6abef21581b189e84c456768c3935d975cec389952b40fdba64c",
    "parent_report": "d260872251077e1515251b6c7afc316af25df75045c8119112dff4f36c68ea23",
}
CONTROL = relation.CONTROL
CANDIDATE = relation.CANDIDATE
ARMS = (CONTROL, CANDIDATE)
GRID = relation.GRID_SIZE
COUNT = GRID * GRID
PAIR_DENOMINATOR = relation.PAIR_DENOMINATOR
SSIM_NAMES = (
    "layout_only_clean_ssim",
    "production_like_dirty_ssim",
    "production_like_restored_h20_ssim",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--parent-report", type=Path, default=DEFAULT_PARENT_REPORT)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def _project_record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        rendered = str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        rendered = str(resolved)
    return {"path": rendered, "sha256": sha256_file(resolved)}


def _verify_frozen_inputs(
    archive: Path, metadata: Path, parent_report: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = {
        "archive": archive.resolve(),
        "metadata": metadata.resolve(),
        "parent_report": parent_report.resolve(),
    }
    for name, path in paths.items():
        if not path.is_file() or sha256_file(path) != EXPECTED_SHA256[name]:
            raise ValueError(f"frozen relation-confirmation {name} changed: {path}")
    metadata_payload = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    report_payload = json.loads(paths["parent_report"].read_text(encoding="utf-8"))
    if (
        metadata_payload.get("contains_exact_references_or_labels") is not False
        or metadata_payload.get("all_layouts_strict_original_upright_tile_permutations")
        is not True
        or len(metadata_payload.get("rows", ())) != relation.CASE_COUNT
    ):
        raise RuntimeError("frozen relation metadata contract changed")
    if (
        report_payload.get("schema") != relation.REPORT_SCHEMA
        or report_payload.get("status") != "confirmed"
        or len(report_payload.get("rows", ())) != relation.CASE_COUNT
    ):
        raise RuntimeError("formal relation report contract changed")
    return metadata_payload, report_payload


def _strict(value: Any) -> np.ndarray:
    return strict_layout(value, grid=GRID)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=False)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {path}") from error


def _summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["arm"])].append(row)
    output: dict[str, Any] = {}
    metrics = (
        "satisfied_adjacent_pairs",
        "exact_tiles",
        "mean_manhattan_cells",
        "within_radius_0_recall",
        "within_radius_2_recall",
        *SSIM_NAMES,
    )
    for arm in ARMS:
        arm_rows = grouped[arm]
        if len(arm_rows) != relation.CASE_COUNT:
            raise RuntimeError("bridge arm case count changed")
        output[arm] = {
            "case_count": len(arm_rows),
            **{
                metric: float(np.mean([float(row[metric]) for row in arm_rows]))
                for metric in metrics
            },
        }
    delta = {
        metric: output[CANDIDATE][metric] - output[CONTROL][metric]
        for metric in metrics
    }
    changed = [row for row in rows if bool(row["changed_from_control"])]
    changed_grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in changed:
        changed_grouped[str(row["arm"])].append(row)
    changed_delta = {
        metric: float(
            np.mean([float(row[metric]) for row in changed_grouped[CANDIDATE]])
            - np.mean([float(row[metric]) for row in changed_grouped[CONTROL]])
        )
        for metric in metrics
    }
    return {
        "arms": output,
        "candidate_minus_control": delta,
        "changed_case_count": len(changed_grouped[CONTROL]),
        "changed_cases_candidate_minus_control": changed_delta,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    archive_path = args.archive.resolve()
    metadata_path = args.metadata.resolve()
    parent_report_path = args.parent_report.resolve()
    targets_path = args.targets.resolve()
    output_dir = args.output_dir.resolve()
    metadata, parent_report = _verify_frozen_inputs(
        archive_path, metadata_path, parent_report_path
    )

    config = relation._load_signed_json(  # noqa: SLF001
        relation.DEFAULT_CONFIG, schema=relation.CONFIG_SCHEMA
    )
    roster, manifest = relation._validate_preregistration(  # noqa: SLF001
        relation.DEFAULT_CONFIG, config
    )
    specs = [(manifest[name], name, draw) for name in roster for draw in relation.DRAWS]
    frozen_rows = metadata["rows"]
    report_rows = parent_report["rows"]
    cache = synthetic.CleanTileCache(targets_path, maximum_boards=2)
    scored: list[dict[str, Any]] = []
    report_replay_count = 0
    strict_count = 0
    started = perf_counter()

    with np.load(archive_path, allow_pickle=False) as archive:
        for (record, source, draw), frozen, reported in zip(
            specs, frozen_rows, report_rows, strict=True
        ):
            dirty, reference = make_exact_synthetic_case(
                cache.load(record),
                source_filename=source,
                draw_index=draw,
                seed=synthetic.SYNTHETIC_SEED,
            )
            if (
                frozen["source_filename"] != source
                or int(frozen["draw_index"]) != draw
                or frozen["case_id"] != dirty.case_id
                or frozen["dirty_sha256"] != synthetic._dirty_sha256(dirty.tiles)  # noqa: SLF001
                or reported["prefix"] != frozen["prefix"]
            ):
                raise RuntimeError("bridge reconstructed a different frozen case")
            prefix = str(frozen["prefix"])
            exact = _strict(reference.tile_at_position)
            clean_tiles = cache.load(record)
            shuffled_clean_tiles = np.ascontiguousarray(
                clean_tiles[np.argsort(exact)], dtype=np.uint8
            )
            if not np.array_equal(shuffled_clean_tiles[exact], clean_tiles):
                raise RuntimeError("clean shuffled identity reconstruction failed")
            clean_target = assemble_tiles(clean_tiles)

            for arm in ARMS:
                layout = _strict(archive[f"{prefix}__{arm}_layout"])
                strict_count += 1
                classic = evaluate_layout(layout, exact, reference_is_exact=True)
                if classic.adjacency_total != PAIR_DENOMINATOR:
                    raise RuntimeError("bridge pair denominator changed")
                expected = reported[arm]
                if (
                    int(classic.adjacency_correct)
                    != int(expected["satisfied_adjacent_pairs"])
                    or int(classic.correct_tile_count) != int(expected["exact_tiles"])
                    or expected.get("strict_permutation") is not True
                ):
                    raise RuntimeError("bridge does not replay the formal report")
                report_replay_count += 1
                distance = evaluate_tile_position_distance(layout, exact, grid=GRID)
                if (
                    distance.exact_tile_count != int(classic.correct_tile_count)
                    or distance.within_radius_0_recall
                    != int(classic.correct_tile_count) / COUNT
                ):
                    raise RuntimeError("distance radius0 does not equal exact")
                clean_canvas = assemble_tiles(shuffled_clean_tiles[layout])
                dirty_canvas = assemble_tiles(dirty.tiles[layout])
                restored = historical_rgb_luma_nlm_h20_once(dirty_canvas)
                scored.append(
                    {
                        "prefix": prefix,
                        "source_filename": source,
                        "draw_index": draw,
                        "arm": arm,
                        "changed_from_control": bool(frozen["changed_from_control"]),
                        "satisfied_adjacent_pairs": int(classic.adjacency_correct),
                        "exact_tiles": int(classic.correct_tile_count),
                        "mean_manhattan_cells": distance.mean_manhattan_cells,
                        "within_radius_0_recall": distance.within_radius_0_recall,
                        "within_radius_2_recall": distance.within_radius_2_recall,
                        "layout_only_clean_ssim": contest_ssim(
                            clean_target, clean_canvas
                        ),
                        "production_like_dirty_ssim": contest_ssim(
                            clean_target, dirty_canvas
                        ),
                        "production_like_restored_h20_ssim": contest_ssim(
                            clean_target, restored
                        ),
                    }
                )

    if len(scored) != relation.CASE_COUNT * len(ARMS):
        raise RuntimeError("bridge scored row count changed")
    summary = _summarize(scored)
    parent_metrics = parent_report["metrics"]["arms"]
    for arm in ARMS:
        if (
            summary["arms"][arm]["satisfied_adjacent_pairs"]
            != float(parent_metrics[arm]["satisfied_adjacent_pairs"])
            or summary["arms"][arm]["exact_tiles"]
            != float(parent_metrics[arm]["exact_tiles"])
        ):
            raise RuntimeError("bridge aggregate does not replay formal report")

    report = {
        "schema": "aiijc-relation-selector-distance-bridge-report-v1",
        "status": "post_hoc_descriptive_complete",
        "scope": {
            "comparison": list(ARMS),
            "existing_formal_confirmation_cases_only": True,
            "new_inference_model_selection_gate_or_threshold_sweep": False,
            "absolute_position_metrics_only": True,
            "ssim_is_evaluation_only": True,
        },
        "verification": {
            "strict_layout_count": strict_count,
            "formal_report_replay_count": report_replay_count,
            "radius0_equals_exact_count": len(scored),
            "expected_case_count": relation.CASE_COUNT,
        },
        "metrics": summary,
        "rows": scored,
        "runtime_seconds": perf_counter() - started,
        "legality": {
            "organizer_train_targets_only_for_post_freeze_evaluation": True,
            "competition_test_terminal_held_or_fresh_accessed": False,
            "production_or_submission_modified": False,
            "output_layouts_are_original_upright_strict_permutations": True,
            "restoration_is_evaluation_only_single_h20": True,
        },
        "artifacts": {
            "frozen_relation_archive": _project_record(archive_path),
            "frozen_relation_metadata": _project_record(metadata_path),
            "formal_parent_report": _project_record(parent_report_path),
            "distance_module": _project_record(
                PROJECT_ROOT / "src/aiijc_puzzle/tile_position_distance.py"
            ),
            "runner": _project_record(Path(__file__)),
        },
    }
    _write_json(output_dir / "report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    report = run(parse_args(argv))
    print(
        json.dumps(
            {
                "status": report["status"],
                "runtime_seconds": report["runtime_seconds"],
                "verification": report["verification"],
                "metrics": report["metrics"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

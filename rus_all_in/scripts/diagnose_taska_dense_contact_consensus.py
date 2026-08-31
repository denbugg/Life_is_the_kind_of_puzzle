#!/usr/bin/env python3
"""Run one frozen dense-top8 component-consensus feasibility diagnostic."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.taska_dense_contact_consensus import (
    DenseConsensusBoard,
    build_dense_contact_consensus,
)

try:
    from scripts import run_taska_focal_current_finetune as finetune
    from scripts import run_taska_joint_component_pose as joint_pose
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    import run_taska_focal_current_finetune as finetune
    import run_taska_joint_component_pose as joint_pose


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs/taska_dense_contact_consensus_feasibility_v1.json"
)
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs/taska-dense-contact-consensus-feasibility/fixed-v1"
)
JOINT_CACHE = (
    PROJECT_ROOT
    / "outputs/taska-joint-component-pose/pilot-v1/cache/dirty-visible-inputs.npz"
)
JOINT_METADATA = (
    PROJECT_ROOT / "outputs/taska-joint-component-pose/pilot-v1/cache/metadata.json"
)
GRID = 24
COUNT = GRID * GRID


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=("freeze", "score", "all"), default="all")
    return parser.parse_args(argv)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {path}") from error


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {path}") from error


def _record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        rendered = str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        rendered = str(resolved)
    return {"path": rendered, "sha256": sha256_file(resolved)}


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    resolved = path.resolve()
    sidecar = resolved.with_suffix(resolved.suffix + ".sha256")
    if not resolved.is_file() or not sidecar.is_file():
        raise FileNotFoundError("signed feasibility preregistration is missing")
    digest = sha256_file(resolved)
    if sidecar.read_text(encoding="utf-8").split()[0] != digest:
        raise ValueError("feasibility preregistration SHA-256 mismatch")
    config = json.loads(resolved.read_text(encoding="utf-8"))
    required = {
        "grid_size": 24,
        "dense_contact_topk": 8,
        "minimum_independent_contacts": 2,
        "require_every_counted_contact_reciprocal": True,
        "require_axis_consistency": True,
        "exclude_contacts_already_realised_by_control": True,
        "no_sweep": True,
        "competition_test_accessed": False,
    }
    for key, value in required.items():
        if config.get(key) != value:
            raise ValueError(f"fixed feasibility preregistration mismatch: {key}")
    for relative, expected in config["fixed_inputs_sha256"].items():
        source = PROJECT_ROOT / relative
        if not source.is_file() or sha256_file(source) != expected:
            raise ValueError(f"fixed input changed: {relative}")
    return config, digest


def _board_arrays(key: str, board: DenseConsensusBoard) -> dict[str, np.ndarray]:
    return {
        f"{key}__edge_source": board.edge_source,
        f"{key}__edge_target": board.edge_target,
        f"{key}__edge_axis": board.edge_axis,
        f"{key}__edge_group": board.edge_group,
        f"{key}__group_component_low": board.group_component_low,
        f"{key}__group_component_high": board.group_component_high,
        f"{key}__group_relative_translation": board.group_relative_translation,
        f"{key}__group_support": board.group_support,
        f"{key}__group_right_support": board.group_right_support,
        f"{key}__group_down_support": board.group_down_support,
    }


def _freeze_target_free(
    *,
    output_dir: Path,
    config: Mapping[str, Any],
    config_path: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    metadata = json.loads(JOINT_METADATA.read_text(encoding="utf-8"))
    cache_rows = {str(row["key"]): row for row in metadata["rows"]}
    arrays: dict[str, np.ndarray] = {}
    rows_out: list[dict[str, Any]] = []
    started = perf_counter()
    with np.load(JOINT_CACHE, allow_pickle=False) as cached:
        for panel in joint_pose.PANELS:
            rows = joint_pose._rows(panel)
            with np.load(panel.base_archive, allow_pickle=False) as base:
                for index, row in enumerate(rows):
                    key = f"{panel.name}_{index:03d}"
                    cached_row = cache_rows[key]
                    if (
                        cached_row["source_filename"] != row["source_filename"]
                        or int(cached_row["draw_index"]) != int(row["draw_index"])
                        or cached_row["dirty_sha256"] != row["dirty_sha256"]
                    ):
                        raise RuntimeError("joint cache row no longer matches panel row")
                    board = joint_pose._load_board(cached, key)
                    prefix = str(row["prefix"])
                    consensus = build_dense_contact_consensus(
                        layout=board.layout,
                        component_of_tile=board.component_of_tile,
                        component_relative_coordinates=(
                            board.component_relative_coordinates
                        ),
                        cost_right=base[f"{prefix}__cost_right"],
                        cost_down=base[f"{prefix}__cost_down"],
                        grid=GRID,
                        dense_topk=int(config["dense_contact_topk"]),
                        minimum_support=int(config["minimum_independent_contacts"]),
                    )
                    arrays.update(_board_arrays(key, consensus))
                    rows_out.append(
                        {
                            "key": key,
                            "panel": panel.name,
                            "prefix": prefix,
                            "source_filename": str(row["source_filename"]),
                            "draw_index": int(row["draw_index"]),
                            "dirty_sha256": str(row["dirty_sha256"]),
                            "control_layout_sha256": __import__("hashlib")
                            .sha256(board.layout.tobytes())
                            .hexdigest(),
                            "component_count": int(len(board.component_sizes)),
                            "outgoing_physical_contact_count": int(
                                consensus.outgoing_physical_contact_count
                            ),
                            "incoming_physical_contact_count": int(
                                consensus.incoming_physical_contact_count
                            ),
                            "reciprocal_missing_contact_count": int(
                                consensus.reciprocal_missing_contact_count
                            ),
                            "qualifying_group_count": int(
                                len(consensus.group_support)
                            ),
                            "emitted_edge_count": int(len(consensus.edge_source)),
                        }
                    )
                    print(
                        json.dumps(
                            {
                                "event": "dense_consensus_freeze",
                                "panel": panel.name,
                                "case": index + 1,
                                "groups": len(consensus.group_support),
                                "edges": len(consensus.edge_source),
                            }
                        ),
                        flush=True,
                    )
    archive = output_dir / "frozen-target-free-consensus.npz"
    metadata_path = output_dir / "frozen-target-free-consensus.json"
    _write_npz(archive, arrays)
    _write_json(
        metadata_path,
        {
            "schema": "aiijc-taska-dense-contact-consensus-freeze-v1",
            "target_or_reference_accessed_before_freeze": False,
            "competition_test_accessed": False,
            "fixed_rule": {
                "topk": 8,
                "minimum_support": 2,
                "all_contacts_reciprocal": True,
                "both_axes_required": True,
                "already_realised_contacts_excluded": True,
            },
            "rows": rows_out,
        },
    )
    pre_score = output_dir / "pre-score-freeze.json"
    _write_json(
        pre_score,
        {
            "schema": "aiijc-taska-dense-contact-consensus-pre-score-v1",
            "target_or_reference_accessed": False,
            "competition_test_accessed": False,
            "artifacts": {
                "preregistration": _record(config_path),
                "target_free_archive": _record(archive),
                "target_free_metadata": _record(metadata_path),
                "joint_input_cache": _record(JOINT_CACHE),
                "joint_metadata": _record(JOINT_METADATA),
            },
        },
    )
    return {
        "runtime_seconds": perf_counter() - started,
        "archive": _record(archive),
        "metadata": _record(metadata_path),
        "pre_score_freeze": _record(pre_score),
    }


def _edge_set(layout: np.ndarray) -> set[tuple[int, int, int]]:
    board = np.asarray(layout, dtype=np.int32).reshape(GRID, GRID)
    edges = {
        (int(board[row, column]), int(board[row, column + 1]), 0)
        for row in range(GRID)
        for column in range(GRID - 1)
    }
    edges.update(
        (int(board[row, column]), int(board[row + 1, column]), 1)
        for row in range(GRID - 1)
        for column in range(GRID)
    )
    return edges


def _panel_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    board_count = len(rows)
    emitted = sum(int(row["emitted_edge_count"]) for row in rows)
    true = sum(int(row["true_emitted_edge_count"]) for row in rows)
    missing = sum(int(row["true_missing_edge_count"]) for row in rows)
    groups = sum(int(row["qualifying_group_count"]) for row in rows)
    all_true_groups = sum(int(row["all_true_group_count"]) for row in rows)
    touched = sum(int(row["emitted_edge_count"]) > 0 for row in rows)
    return {
        "board_count": board_count,
        "boards_with_signal": touched,
        "board_signal_rate": touched / board_count,
        "qualifying_group_count": groups,
        "mean_qualifying_groups_per_board": groups / board_count,
        "emitted_edge_count": emitted,
        "mean_emitted_edges_per_board": emitted / board_count,
        "true_emitted_edge_count": true,
        "pooled_edge_precision": true / emitted if emitted else 0.0,
        "true_missing_edge_count": missing,
        "true_missing_coverage": true / missing if missing else 0.0,
        "all_true_group_count": all_true_groups,
        "all_true_group_precision": all_true_groups / groups if groups else 0.0,
    }


def _score(
    *,
    output_dir: Path,
    targets_dir: Path,
    config: Mapping[str, Any],
    config_path: Path,
    config_sha256: str,
) -> dict[str, Any]:
    archive_path = output_dir / "frozen-target-free-consensus.npz"
    metadata_path = output_dir / "frozen-target-free-consensus.json"
    pre_score = output_dir / "pre-score-freeze.json"
    for path in (archive_path, metadata_path, pre_score):
        if not path.is_file():
            raise FileNotFoundError(f"missing target-free freeze: {path}")
    frozen_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    frozen_rows = {str(row["key"]): row for row in frozen_metadata["rows"]}

    # Exact organizer-train references are reconstructed only after the
    # target-free archive and pre-score manifest exist on disk.
    source_config, _, _ = finetune._load_config(finetune.DEFAULT_CONFIG)
    lookup = finetune._manifest_lookup(source_config)
    tile_cache = finetune.CleanTileCache(targets_dir, maximum_boards=2)
    rows_scored: list[dict[str, Any]] = []
    started = perf_counter()
    with (
        np.load(archive_path, allow_pickle=False) as frozen,
        np.load(JOINT_CACHE, allow_pickle=False) as joint_cached,
    ):
        for panel in joint_pose.PANELS:
            for index, row in enumerate(joint_pose._rows(panel)):
                key = f"{panel.name}_{index:03d}"
                frozen_row = frozen_rows[key]
                source = str(row["source_filename"])
                draw = int(row["draw_index"])
                dirty = finetune._dirty_case(tile_cache, lookup[source], source, draw)
                if finetune._dirty_sha256(dirty.dirty_tiles) != row["dirty_sha256"]:
                    raise RuntimeError("exact reconstruction changed frozen dirty bytes")
                reference = finetune._reference(
                    tile_cache,
                    lookup[source],
                    source,
                    draw,
                    dirty.dirty_tiles,
                )
                board = joint_pose._load_board(joint_cached, key)
                truth = _edge_set(reference)
                realised = _edge_set(board.layout)
                missing = truth - realised
                sources = np.asarray(frozen[f"{key}__edge_source"], dtype=np.int32)
                targets = np.asarray(frozen[f"{key}__edge_target"], dtype=np.int32)
                axes = np.asarray(frozen[f"{key}__edge_axis"], dtype=np.uint8)
                group_index = np.asarray(frozen[f"{key}__edge_group"], dtype=np.int32)
                emitted_edges = [
                    (int(source_value), int(target_value), int(axis_value))
                    for source_value, target_value, axis_value in zip(
                        sources, targets, axes, strict=True
                    )
                ]
                if len(emitted_edges) != len(set(emitted_edges)):
                    raise RuntimeError("consensus emitter produced duplicate physical edges")
                if any(edge in realised for edge in emitted_edges):
                    raise RuntimeError("consensus emitter retained an already realised edge")
                correct = np.asarray(
                    [edge in truth for edge in emitted_edges], dtype=bool
                )
                group_count = int(frozen_row["qualifying_group_count"])
                all_true_groups = sum(
                    bool(correct[group_index == group].all())
                    for group in range(group_count)
                )
                rows_scored.append(
                    {
                        **frozen_row,
                        "true_emitted_edge_count": int(correct.sum()),
                        "edge_precision": (
                            float(correct.mean()) if len(correct) else 0.0
                        ),
                        "true_missing_edge_count": int(len(missing)),
                        "true_missing_coverage": (
                            float(correct.sum() / len(missing)) if missing else 0.0
                        ),
                        "all_true_group_count": int(all_true_groups),
                    }
                )
    panels = {
        panel.name: _panel_summary(
            [row for row in rows_scored if row["panel"] == panel.name]
        )
        for panel in joint_pose.PANELS
    }
    frequency = config["frequency_gate_per_panel"]
    precision = config["precision_gate_per_panel"]
    for summary in panels.values():
        summary["frequency_gate_pass"] = bool(
            summary["boards_with_signal"]
            >= int(frequency["minimum_boards_with_signal"])
            and summary["mean_emitted_edges_per_board"]
            >= float(frequency["minimum_mean_emitted_edges_per_board"])
        )
        # Parent preregistration required strictly above 60%, not a tie.
        summary["precision_gate_pass"] = bool(
            summary["pooled_edge_precision"]
            > float(precision["minimum_pooled_edge_precision"])
        )
        summary["panel_gate_pass"] = bool(
            summary["frequency_gate_pass"] and summary["precision_gate_pass"]
        )
    passed = all(summary["panel_gate_pass"] for summary in panels.values())
    report = {
        "schema": "aiijc-taska-dense-contact-consensus-feasibility-report-v1",
        "status": (
            "feasibility-pass-preregister-one-solver-arm"
            if passed
            else "feasibility-fail-stop-without-solver"
        ),
        "decision": {
            "all_panels_pass": passed,
            "solver_constructed_or_evaluated": False,
            "next_action": config["pass_action"] if passed else config["fail_action"],
            "threshold_or_topk_sweep_performed": False,
        },
        "protocol": {
            "preregistration": _record(config_path),
            "preregistration_sha256": config_sha256,
            "target_free_freeze_preceded_reference_access": True,
            "targets": "organizer train only",
            "fit_and_local_only": True,
            "fresh_or_competition_test_accessed": False,
            "strict_upright_layout_contract": True,
        },
        "fixed_rule": {
            "dense_topk": 8,
            "minimum_independent_contacts": 2,
            "every_contact_reciprocal": True,
            "right_and_down_axis_support_required": True,
            "already_realised_contacts_excluded": True,
        },
        "panels": panels,
        "rows": rows_scored,
        "runtime_seconds": perf_counter() - started,
        "artifacts": {
            "target_free_archive": _record(archive_path),
            "target_free_metadata": _record(metadata_path),
            "pre_score_freeze": _record(pre_score),
            "module": _record(
                PROJECT_ROOT
                / "src/aiijc_puzzle/taska_dense_contact_consensus.py"
            ),
            "runner": _record(Path(__file__)),
        },
    }
    report_path = output_dir / "report.json"
    _write_json(report_path, report)
    return {**report, "report": _record(report_path)}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config, digest = _load_config(args.config)
    if args.mode in {"freeze", "all"}:
        summary = _freeze_target_free(
            output_dir=args.output_dir,
            config=config,
            config_path=args.config,
        )
        print(json.dumps({"event": "target_free_frozen", **summary}), flush=True)
    if args.mode in {"score", "all"}:
        report = _score(
            output_dir=args.output_dir,
            targets_dir=args.targets,
            config=config,
            config_path=args.config,
            config_sha256=digest,
        )
        print(
            json.dumps(
                {
                    "event": "dense_consensus_scored",
                    "status": report["status"],
                    "panels": report["panels"],
                    "report": report["report"],
                },
                indent=2,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

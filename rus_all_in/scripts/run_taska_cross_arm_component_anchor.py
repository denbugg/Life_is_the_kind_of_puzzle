#!/usr/bin/env python3
"""Run one preregistered local-only cross-arm component anchor."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from aiijc_puzzle.layout_evaluation import evaluate_layout
from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.taska_cross_arm_component_anchor import (
    anchor_one_component_from_cross_arm_agreement,
)
from aiijc_puzzle.taska_pair_pipeline import PAIR_DENOMINATOR
from aiijc_puzzle.taska_selective_fullres_fusion import FUSION_ARM_NAMES

try:
    from scripts import run_taska_component_relation_anchor as relation_anchor
    from scripts import run_taska_focal_current_finetune as finetune
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    import run_taska_component_relation_anchor as relation_anchor
    import run_taska_focal_current_finetune as finetune


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/taska_cross_arm_component_anchor_v1.json"
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/taska-cross-arm-component-anchor/local32-v1"
ARM_ROOT = PROJECT_ROOT / "outputs/taska-six-arm-learned-selector/fixed-v1/local32"
ARM_ARCHIVE = ARM_ROOT / "frozen-target-free-arms.npz"
ARM_METADATA = ARM_ROOT / "frozen-target-free-arms.json"
ARM_FREEZE = ARM_ROOT / "pre-arm-score-freeze.json"
FUSION_ROOT = (
    PROJECT_ROOT / "outputs/taska-selective-fullres-union-fusion/fixed-v1/local32"
)
FUSION_ARCHIVE = FUSION_ROOT / "frozen-target-free-eval.npz"
FUSION_METADATA = FUSION_ROOT / "frozen-target-free-eval.json"
FUSION_FREEZE = FUSION_ROOT / "pre-score-freeze.json"
GRID = 24
CONTROL = "confirmed_six_arm_fusion"
CANDIDATE = "cross_arm_component_anchor"
LOCAL_EXACT_GATE = "strictly_positive"
LOCAL_PAIR_GATE = -1.0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-free-only", action="store_true")
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
        raise FileNotFoundError("signed cross-arm anchor preregistration is missing")
    digest = sha256_file(resolved)
    if sidecar.read_text(encoding="utf-8").split()[0] != digest:
        raise ValueError("cross-arm anchor preregistration SHA-256 mismatch")
    config = json.loads(resolved.read_text(encoding="utf-8"))
    required = {
        "arm_roster": list(FUSION_ARM_NAMES),
        "component_focal_threshold": 0.0,
        "whole_component_rigid_agreement_required": True,
        "minimum_distinct_arm_support": 2,
        "maximum_moved_components": 1,
        "raw_seam_veto_or_score_used": False,
        "local_exact_gate": LOCAL_EXACT_GATE,
        "local_pair_gate": LOCAL_PAIR_GATE,
        "terminal_or_fresh_opened_before_local_pass": False,
        "no_sweep": True,
    }
    for key, value in required.items():
        if config.get(key) != value:
            raise ValueError(f"cross-arm anchor preregistration mismatch: {key}")
    for relative, expected in config["fixed_source_and_input_sha256"].items():
        source = PROJECT_ROOT / relative
        if not source.is_file() or sha256_file(source) != expected:
            raise ValueError(f"fixed cross-arm source/input changed: {relative}")
    return config, digest


def _rows(path: Path) -> list[Mapping[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8")).get("rows")
    if not isinstance(rows, list) or len(rows) != 32:
        raise ValueError(f"{path} must contain exactly 32 rows")
    return rows


def _aligned_rows() -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    arm_rows = _rows(ARM_METADATA)
    fusion_rows = _rows(FUSION_METADATA)
    identity = ("prefix", "source_filename", "draw_index", "dirty_sha256")
    aligned = []
    for arm, fused in zip(arm_rows, fusion_rows, strict=True):
        if any(arm.get(field) != fused.get(field) for field in identity):
            raise RuntimeError("arm and fusion local32 rows do not align")
        aligned.append((arm, fused))
    return aligned


def _validate_parent_freezes() -> None:
    for path in (ARM_FREEZE, FUSION_FREEZE):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("created_before_exact_reference_reconstruction") is not True:
            raise RuntimeError(f"parent target-free timing changed: {path}")
        if payload.get("contains_evaluation_references_or_labels") is not False:
            raise RuntimeError(f"parent freeze unexpectedly contains labels: {path}")


def _freeze_candidates(
    *, output_dir: Path, config_path: Path
) -> tuple[Path, Path, Path, float]:
    output_dir.mkdir(parents=True, exist_ok=False)
    _validate_parent_freezes()
    arrays: dict[str, np.ndarray] = {}
    rows_out: list[dict[str, Any]] = []
    started = perf_counter()
    with (
        np.load(ARM_ARCHIVE, allow_pickle=False) as arms,
        np.load(FUSION_ARCHIVE, allow_pickle=False) as fused,
    ):
        for index, (arm_row, fused_row) in enumerate(_aligned_rows()):
            prefix = str(arm_row["prefix"])
            control = np.asarray(
                arms[f"{prefix}__{CONTROL}_layout"], dtype=np.int32
            )
            if not np.array_equal(
                control,
                fused[f"{prefix}__combined_union_candidate_layout"],
            ):
                raise RuntimeError("six-arm control differs between frozen parents")
            arm_layouts = {
                name: np.asarray(arms[f"{prefix}__{name}_layout"], dtype=np.int32)
                for name in FUSION_ARM_NAMES
            }
            edges, logits, selected_family = relation_anchor._selected_supply(
                fused, prefix, str(fused_row["choice"])
            )
            anchored = anchor_one_component_from_cross_arm_agreement(
                control,
                arm_layouts,
                edges,
                logits,
                grid=GRID,
                focal_threshold=0.0,
                minimum_distinct_arm_support=2,
            )
            arrays[f"{prefix}__{CONTROL}_layout"] = control
            arrays[f"{prefix}__{CANDIDATE}_layout"] = anchored.layout
            rows_out.append(
                {
                    "prefix": prefix,
                    "source_filename": str(arm_row["source_filename"]),
                    "draw_index": int(arm_row["draw_index"]),
                    "dirty_sha256": str(arm_row["dirty_sha256"]),
                    "control_choice": str(arm_row["control_choice"]),
                    "selected_supply_family": selected_family,
                    "diagnostics": asdict(anchored.diagnostics),
                }
            )
            print(
                json.dumps(
                    {
                        "event": "cross_arm_component_anchor_freeze",
                        "case": index + 1,
                        "changed": anchored.diagnostics.changed,
                        "hypotheses": (
                            anchored.diagnostics.consensus_hypothesis_count
                        ),
                        "selected_size": (
                            anchored.diagnostics.selected_component_size
                        ),
                        "support": (
                            anchored.diagnostics.selected_distinct_arm_support
                        ),
                    }
                ),
                flush=True,
            )
    archive = output_dir / "frozen-target-free-eval.npz"
    metadata = output_dir / "frozen-target-free-eval.json"
    freeze = output_dir / "pre-score-freeze.json"
    _write_npz(archive, arrays)
    _write_json(
        metadata,
        {
            "schema": "aiijc-taska-cross-arm-component-anchor-target-free-v1",
            "created_before_exact_reference_reconstruction": True,
            "contains_evaluation_references_or_labels": False,
            "competition_test_accessed": False,
            "arm_roster": list(FUSION_ARM_NAMES),
            "rows": rows_out,
        },
    )
    _write_json(
        freeze,
        {
            "schema": "aiijc-taska-cross-arm-component-anchor-pre-score-v1",
            "created_before_exact_reference_reconstruction": True,
            "contains_evaluation_references_or_labels": False,
            "terminal_or_fresh_accessed": False,
            "competition_test_accessed": False,
            "artifacts": {
                "candidate_archive": _record(archive),
                "candidate_metadata": _record(metadata),
                "preregistration": _record(config_path),
                "six_arm_archive": _record(ARM_ARCHIVE),
                "six_arm_metadata": _record(ARM_METADATA),
                "six_arm_freeze": _record(ARM_FREEZE),
                "fusion_archive": _record(FUSION_ARCHIVE),
                "fusion_metadata": _record(FUSION_METADATA),
                "fusion_freeze": _record(FUSION_FREEZE),
            },
        },
    )
    return archive, metadata, freeze, perf_counter() - started


def _layout_metrics(layout: Any, reference: Any) -> dict[str, Any]:
    result = evaluate_layout(layout, reference, reference_is_exact=True)
    if result.adjacency_total != PAIR_DENOMINATOR:
        raise RuntimeError("pair denominator changed")
    return {
        "exact_tiles": int(result.correct_tile_count),
        "satisfied_adjacent_pairs": int(result.adjacency_correct),
        "adjacency_recall": float(result.adjacency),
        "strict_original_upright_permutation": True,
    }


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = ("exact_tiles", "satisfied_adjacent_pairs", "adjacency_recall")
    arms = {
        arm: {
            metric: float(np.mean([row["metrics"][arm][metric] for row in rows]))
            for metric in metrics
        }
        for arm in (CONTROL, CANDIDATE)
    }
    deltas: dict[str, Any] = {}
    for metric in metrics:
        values = np.asarray(
            [
                row["metrics"][CANDIDATE][metric]
                - row["metrics"][CONTROL][metric]
                for row in rows
            ],
            dtype=np.float64,
        )
        deltas[metric] = {
            "mean": float(values.mean()),
            "wins": int(np.count_nonzero(values > 0)),
            "ties": int(np.count_nonzero(values == 0)),
            "losses": int(np.count_nonzero(values < 0)),
        }
    diagnostics = [row["diagnostics"] for row in rows]
    return {
        "case_count": len(rows),
        "pair_denominator": PAIR_DENOMINATOR,
        "arms": arms,
        "candidate_minus_control": deltas,
        "target_free_diagnostics": {
            "changed_layout_count": sum(row["diagnostics"]["changed"] for row in rows),
            "mean_consensus_hypothesis_count": float(
                np.mean([row["consensus_hypothesis_count"] for row in diagnostics])
            ),
            "mean_selected_component_size": float(
                np.mean([row["selected_component_size"] for row in diagnostics])
            ),
            "mean_selected_distinct_arm_support": float(
                np.mean(
                    [row["selected_distinct_arm_support"] for row in diagnostics]
                )
            ),
            "supporting_arm_combinations": dict(
                Counter(
                    "+".join(row["selected_supporting_arms"])
                    if row["selected_supporting_arms"]
                    else "fallback"
                    for row in diagnostics
                )
            ),
        },
    }


def _score_local(
    *,
    archive: Path,
    metadata: Path,
    targets_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], float]:
    frozen_rows = _rows(metadata)
    source_config, _, _ = finetune._load_config(finetune.DEFAULT_CONFIG)
    lookup = finetune._manifest_lookup(source_config)
    cache = finetune.CleanTileCache(targets_dir, maximum_boards=2)
    scored: list[dict[str, Any]] = []
    started = perf_counter()
    with np.load(archive, allow_pickle=False) as frozen:
        for row in frozen_rows:
            prefix = str(row["prefix"])
            source = str(row["source_filename"])
            draw = int(row["draw_index"])
            dirty = finetune._dirty_case(cache, lookup[source], source, draw)
            if finetune._dirty_sha256(dirty.dirty_tiles) != row["dirty_sha256"]:
                raise RuntimeError("scoring recreated different dirty bytes")
            reference = finetune._reference(
                cache, lookup[source], source, draw, dirty.dirty_tiles
            )
            scored.append(
                {
                    **row,
                    "metrics": {
                        arm: _layout_metrics(
                            frozen[f"{prefix}__{arm}_layout"], reference
                        )
                        for arm in (CONTROL, CANDIDATE)
                    },
                }
            )
    return scored, _summary(scored), perf_counter() - started


def run(args: argparse.Namespace) -> dict[str, Any]:
    config, config_sha256 = _load_config(args.config)
    archive, metadata, freeze, freeze_runtime = _freeze_candidates(
        output_dir=args.output_dir.resolve(),
        config_path=args.config.resolve(),
    )
    if args.target_free_only:
        report = {
            "schema": "aiijc-taska-cross-arm-component-anchor-report-v1",
            "status": "target-free-only",
            "terminal_or_fresh_accessed": False,
            "competition_test_accessed": False,
            "artifacts": {
                "archive": _record(archive),
                "metadata": _record(metadata),
                "freeze": _record(freeze),
            },
        }
        _write_json(args.output_dir / "report.json", report)
        return report
    rows, summary, score_runtime = _score_local(
        archive=archive,
        metadata=metadata,
        targets_dir=args.targets.resolve(),
    )
    exact_delta = summary["candidate_minus_control"]["exact_tiles"]["mean"]
    pair_delta = summary["candidate_minus_control"]["satisfied_adjacent_pairs"][
        "mean"
    ]
    passed = exact_delta > 0.0 and pair_delta >= LOCAL_PAIR_GATE
    report = {
        "schema": "aiijc-taska-cross-arm-component-anchor-report-v1",
        "status": "local-gate-pass-await-root" if passed else "local-gate-fail-stop",
        "protocol": config,
        "preregistration_sha256": config_sha256,
        "local32": {"rows": rows, "summary": summary},
        "decision": {
            "local_gate_pass": passed,
            "exact_delta_must_be_strictly_positive": True,
            "pair_delta_minimum": LOCAL_PAIR_GATE,
            "terminal_or_fresh_opened": False,
            "next_action": (
                "root review before any terminal/fresh panel"
                if passed
                else "stop this fixed candidate without nearby sweep"
            ),
        },
        "runtime_seconds": {
            "target_free_freeze": freeze_runtime,
            "local_scoring": score_runtime,
        },
        "legality": {
            "strict_original_upright_permutations": True,
            "pixels_changed_rotated_warped_replaced_or_postprocessed": False,
            "terminal_or_fresh_accessed": False,
            "competition_test_accessed": False,
            "production_or_submission_modified": False,
        },
        "artifacts": {
            "archive": _record(archive),
            "metadata": _record(metadata),
            "freeze": _record(freeze),
            "module": _record(
                PROJECT_ROOT / "src/aiijc_puzzle/taska_cross_arm_component_anchor.py"
            ),
            "runner": _record(Path(__file__)),
        },
    }
    _write_json(args.output_dir / "report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "local_summary": report.get("local32", {}).get("summary"),
                "decision": report.get("decision"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the preregistered Socket border5 origin transfer on TASKA local32."""

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
import torch

from aiijc_puzzle.layout_evaluation import evaluate_layout
from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.socket_sorter_production import (
    choose_deterministic_device,
    load_socket_checkpoint,
)
from aiijc_puzzle.taska_pair_pipeline import PAIR_DENOMINATOR
from aiijc_puzzle.taska_socket_cyclic_origin_transfer import (
    transfer_socket_cyclic_origin,
)

try:
    from scripts import run_taska_focal_current_finetune as finetune
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    import run_taska_focal_current_finetune as finetune


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/taska_socket_cyclic_origin_transfer_v1.json"
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/socket-matcher/v2-d64-train1024-s1600-r400-dev32/socket_matcher.pt"
)
ARM_ROOT = PROJECT_ROOT / "outputs/taska-six-arm-learned-selector/fixed-v1/local32"
ARM_ARCHIVE = ARM_ROOT / "frozen-target-free-arms.npz"
ARM_METADATA = ARM_ROOT / "frozen-target-free-arms.json"
ARM_FREEZE = ARM_ROOT / "pre-arm-score-freeze.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/taska-socket-cyclic-origin-transfer/local32-v1"
GRID = 24
COUNT = GRID * GRID
CONTROL = "confirmed_six_arm_fusion"
CANDIDATE = "socket_cyclic_border5_origin"
EXACT_GATE = 0.0
PAIR_GATE = -2.0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
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
        raise FileNotFoundError("signed Socket origin-transfer preregistration is missing")
    digest = sha256_file(resolved)
    if sidecar.read_text(encoding="utf-8").split()[0] != digest:
        raise ValueError("Socket origin-transfer preregistration SHA-256 mismatch")
    config = json.loads(resolved.read_text(encoding="utf-8"))
    required = {
        "development_panel": "already-opened local32 only",
        "control_layout": "confirmed_six_arm_fusion_layout",
        "candidate_layout": (
            "socket_cyclic_border5_roll_of_confirmed_six_arm_fusion_layout"
        ),
        "socket_checkpoint_sha256": sha256_file(DEFAULT_CHECKPOINT),
        "socket_checkpoint_recursive_lineage_count": 1056,
        "local32_source_count": 32,
        "local32_socket_lineage_overlap_count": 6,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f"Socket origin-transfer preregistration mismatch: {key}")
    rule = config.get("frozen_rule", {})
    if rule.get("border_weight") != 5.0 or rule.get("minimum_gain") != 1e-9:
        raise ValueError("Socket cyclic primitive weights changed")
    gate = config.get("local_gate", {})
    if gate.get("exact_delta_must_be_strictly_positive") is not True:
        raise ValueError("exact gate changed")
    if gate.get("minimum_pair_delta_per_board") != PAIR_GATE:
        raise ValueError("pair gate changed")
    if config.get("decision", {}).get("no_sweep") is not True:
        raise ValueError("no-sweep commitment changed")
    for relative, expected in config["fixed_source_and_input_sha256"].items():
        source = PROJECT_ROOT / relative
        if not source.is_file() or sha256_file(source) != expected:
            raise ValueError(f"fixed Socket transfer source/input changed: {relative}")
    return config, digest


def _rows(path: Path) -> list[Mapping[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8")).get("rows")
    if not isinstance(rows, list) or len(rows) != 32:
        raise ValueError(f"{path} must contain exactly 32 rows")
    return rows


def _strict(layout: Any) -> np.ndarray:
    value = np.asarray(layout, dtype=np.int32)
    if value.shape != (COUNT,) or not np.array_equal(
        np.sort(value), np.arange(COUNT, dtype=np.int32)
    ):
        raise ValueError("layout is not a strict 576-tile permutation")
    return np.ascontiguousarray(value)


def _validate_parent_freeze() -> None:
    payload = json.loads(ARM_FREEZE.read_text(encoding="utf-8"))
    if payload.get("created_before_exact_reference_reconstruction") is not True:
        raise RuntimeError("six-arm parent was not frozen before exact scoring")
    if payload.get("contains_evaluation_references_or_labels") is not False:
        raise RuntimeError("six-arm parent freeze unexpectedly contains labels")


@torch.inference_mode()
def _freeze_candidates(
    *,
    output_dir: Path,
    config_path: Path,
    config: Mapping[str, Any],
    checkpoint_path: Path,
    targets_dir: Path,
    device_name: str,
) -> tuple[Path, Path, Path, float]:
    output_dir.mkdir(parents=True, exist_ok=False)
    _validate_parent_freeze()
    device = choose_deterministic_device(device_name)
    checkpoint = load_socket_checkpoint(checkpoint_path, device=device)
    if checkpoint.sha256 != config["socket_checkpoint_sha256"]:
        raise ValueError("loaded Socket checkpoint differs from preregistration")
    if checkpoint.lineage.exposed_count != config["socket_checkpoint_recursive_lineage_count"]:
        raise ValueError("loaded Socket checkpoint lineage count changed")

    rows = _rows(ARM_METADATA)
    source_names = {str(row["source_filename"]) for row in rows}
    overlap = sorted(source_names & set(checkpoint.lineage.exposed_filenames))
    if overlap != config["local32_socket_lineage_overlap"]:
        raise ValueError("local32/Socket lineage overlap changed")

    source_config, _, _ = finetune._load_config(finetune.DEFAULT_CONFIG)
    lookup = finetune._manifest_lookup(source_config)
    cache = finetune.CleanTileCache(targets_dir, maximum_boards=2)
    arrays: dict[str, np.ndarray] = {}
    metadata_rows: list[dict[str, Any]] = []
    started = perf_counter()
    with np.load(ARM_ARCHIVE, allow_pickle=False) as arms:
        for index, row in enumerate(rows, start=1):
            prefix = str(row["prefix"])
            source = str(row["source_filename"])
            draw = int(row["draw_index"])
            dirty = finetune._dirty_case(cache, lookup[source], source, draw)
            if finetune._dirty_sha256(dirty.dirty_tiles) != row["dirty_sha256"]:
                raise RuntimeError("target-free phase recreated different dirty bytes")
            tiles = torch.from_numpy(dirty.dirty_tiles.astype(np.float32)).permute(
                0, 3, 1, 2
            ) / 255.0
            output = checkpoint.model(tiles.unsqueeze(0).to(device), grid=GRID)
            right = output.right_log_assignment[0].float().cpu().numpy()
            down = output.down_log_assignment[0].float().cpu().numpy()
            control = _strict(arms[f"{prefix}__{CONTROL}_layout"])
            candidate = transfer_socket_cyclic_origin(control, right, down, grid=GRID)
            arrays[f"{prefix}__{CONTROL}_layout"] = control
            arrays[f"{prefix}__{CANDIDATE}_layout"] = candidate.layout
            metadata_rows.append(
                {
                    "prefix": prefix,
                    "source_filename": source,
                    "draw_index": draw,
                    "dirty_sha256": str(row["dirty_sha256"]),
                    "checkpoint_lineage_overlap": source in overlap,
                    "diagnostics": asdict(candidate.diagnostics),
                }
            )
            print(
                json.dumps(
                    {
                        "event": "taska_socket_origin_freeze",
                        "case": index,
                        "changed": candidate.diagnostics.changed,
                        "roll": [
                            candidate.diagnostics.selected_row_roll,
                            candidate.diagnostics.selected_column_roll,
                        ],
                        "objective_gain": candidate.diagnostics.objective_gain,
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
            "schema": "aiijc-taska-socket-cyclic-origin-transfer-target-free-v1",
            "created_before_exact_reference_reconstruction": True,
            "contains_evaluation_references_or_labels": False,
            "competition_test_accessed": False,
            "device": device_name,
            "checkpoint": {
                "sha256": checkpoint.sha256,
                "train_lineage_count": checkpoint.lineage.train_count,
                "exposed_lineage_count": checkpoint.lineage.exposed_count,
            },
            "rows": metadata_rows,
        },
    )
    _write_json(
        freeze,
        {
            "schema": "aiijc-taska-socket-cyclic-origin-transfer-pre-score-v1",
            "created_before_exact_reference_reconstruction": True,
            "contains_evaluation_references_or_labels": False,
            "terminal_or_fresh_accessed": False,
            "competition_test_accessed": False,
            "artifacts": {
                "candidate_archive": _record(archive),
                "candidate_metadata": _record(metadata),
                "preregistration": _record(config_path),
                "checkpoint": _record(checkpoint_path),
                "six_arm_archive": _record(ARM_ARCHIVE),
                "six_arm_metadata": _record(ARM_METADATA),
                "six_arm_freeze": _record(ARM_FREEZE),
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
            "changed_layout_count": int(sum(row["changed"] for row in diagnostics)),
            "mean_objective_gain": float(
                np.mean([row["objective_gain"] for row in diagnostics])
            ),
            "selected_roll_histogram": dict(
                Counter(
                    f"{row['selected_row_roll']},{row['selected_column_roll']}"
                    for row in diagnostics
                )
            ),
        },
    }


def _score_local(
    *, archive: Path, metadata: Path, targets_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], float]:
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
    disjoint = [row for row in scored if not row["checkpoint_lineage_overlap"]]
    if len(disjoint) != 26:
        raise RuntimeError("expected 26 Socket-lineage-disjoint local rows")
    return scored, _summary(scored), _summary(disjoint), perf_counter() - started


def run(args: argparse.Namespace) -> dict[str, Any]:
    config, preregistration_sha256 = _load_config(args.config)
    archive, metadata, freeze, freeze_runtime = _freeze_candidates(
        output_dir=args.output_dir.resolve(),
        config_path=args.config.resolve(),
        config=config,
        checkpoint_path=args.checkpoint.resolve(),
        targets_dir=args.targets.resolve(),
        device_name=args.device,
    )
    rows, all32, disjoint26, score_runtime = _score_local(
        archive=archive, metadata=metadata, targets_dir=args.targets.resolve()
    )
    exact_delta = all32["candidate_minus_control"]["exact_tiles"]["mean"]
    pair_delta = all32["candidate_minus_control"]["satisfied_adjacent_pairs"][
        "mean"
    ]
    passed = exact_delta > EXACT_GATE and pair_delta >= PAIR_GATE
    report = {
        "schema": "aiijc-taska-socket-cyclic-origin-transfer-report-v1",
        "status": "local-gate-pass-await-root" if passed else "local-gate-fail-stop",
        "protocol": config,
        "preregistration_sha256": preregistration_sha256,
        "local32": {
            "rows": rows,
            "all32_summary": all32,
            "socket_lineage_disjoint26_summary": disjoint26,
        },
        "decision": {
            "local_gate_pass": passed,
            "gate_uses": "all32 preregistered opened development panel",
            "exact_delta_must_be_strictly_positive": True,
            "pair_delta_minimum": PAIR_GATE,
            "terminal_or_fresh_opened": False,
            "weco_step": 147,
            "next_action": (
                "stop and request root review before any terminal/fresh panel"
                if passed
                else "stop fixed transfer without nearby sweep"
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
                PROJECT_ROOT
                / "src/aiijc_puzzle/taska_socket_cyclic_origin_transfer.py"
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
                "all32_summary": report["local32"]["all32_summary"],
                "disjoint26_summary": report["local32"][
                    "socket_lineage_disjoint26_summary"
                ],
                "decision": report["decision"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

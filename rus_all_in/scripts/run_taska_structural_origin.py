#!/usr/bin/env python3
"""Evaluate one fixed structural-border cyclic origin on focal top-5 layouts.

Dirty tiles, audited v3+local matcher checkpoints, and the already-frozen
focal top-5 layout are the only inference inputs.  For every case the runner
builds the historical structural border unary with slack=6 and 20 Sinkhorn
iterations, enumerates all 24x24 whole-layout rolls, and freezes the selected
strict layout before reconstructing an exact reference for measurement.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np
import torch

from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.synthetic_socket_evaluation import make_exact_synthetic_case
from aiijc_puzzle.taska_seam_matcher import load_default_taska_ensemble
from aiijc_puzzle.taska_structural_border import structural_border_unary
from aiijc_puzzle.taska_structural_origin import select_structural_border_cyclic_origin

try:
    from scripts.run_taska_focal_verifier_replay import (
        BOOTSTRAP_RESAMPLES,
        BOOTSTRAP_SEED,
        DEFAULT_TARGETS,
        GRID,
        MANIFEST,
        PAIR_DENOMINATOR,
        PROJECT_ROOT,
        SYNTHETIC_SEED,
        CleanTileCache,
        _dirty_case,
        _layout_metrics,
        _load_manifest_lookup,
        _parent_dirty_sha256,
        _source_clustered_delta_ci,
        _strict_layout,
        _validated_rows,
    )
except ModuleNotFoundError:  # Direct ``python scripts/run_*.py`` execution.
    from run_taska_focal_verifier_replay import (
        BOOTSTRAP_RESAMPLES,
        BOOTSTRAP_SEED,
        DEFAULT_TARGETS,
        GRID,
        MANIFEST,
        PAIR_DENOMINATOR,
        PROJECT_ROOT,
        SYNTHETIC_SEED,
        CleanTileCache,
        _dirty_case,
        _layout_metrics,
        _load_manifest_lookup,
        _parent_dirty_sha256,
        _source_clustered_delta_ci,
        _strict_layout,
        _validated_rows,
    )

PanelName = Literal["opened32", "held300"]

V3_SHA256 = "6f0917d66d908f6cc0f4c1fcb949d3bcbadcba2490a6f7b5a12596e61de9730e"
LOCAL_SHA256 = "5932853a73961d261b494368a4db04633fecc5996771c14d64f49ef00c7cfe73"
FOCAL_MODE = "train_exact_top5"
SLACK = 6.0
SINKHORN_ITERATIONS = 20

PANEL_SPECS: dict[PanelName, dict[str, str]] = {
    "opened32": {
        "archive": (
            "outputs/taska-focal-verifier/opened32-train-exact-top5-cpu-v2/"
            "frozen-target-free-eval.npz"
        ),
        "archive_sha256": "60243ab924da96d8bb49b072458c4710c65b8195b8d2c31eff1132b59ee56fd2",
        "metadata": (
            "outputs/taska-focal-verifier/opened32-train-exact-top5-cpu-v2/"
            "frozen-target-free-eval.json"
        ),
        "metadata_sha256": "8e6be1d0f4b2652b784141d7c53d7fb63394e8bda6af3b076a9fd5721f07c9d5",
    },
    "held300": {
        "archive": (
            "outputs/taska-focal-verifier/held300-train-exact-top5-cpu-v2/"
            "frozen-target-free-eval.npz"
        ),
        "archive_sha256": "7d4ad494ab572d1ac3c94ab73a49b54e80b26baba489dfbd56f732a5c43394c5",
        "metadata": (
            "outputs/taska-focal-verifier/held300-train-exact-top5-cpu-v2/"
            "frozen-target-free-eval.json"
        ),
        "metadata_sha256": "301ba535f04b63ff8da48a0a83b5f207521d4b57f1bdbb61ceb58dbee57daff2",
    },
}

FROZEN_SCHEMA = "aiijc-taska-structural-origin-frozen-target-free-v1"
FREEZE_SCHEMA = "aiijc-taska-structural-origin-pre-score-freeze-v1"
REPORT_SCHEMA = "aiijc-taska-structural-origin-report-v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=tuple(PANEL_SPECS), required=True)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--smoke-one", action="store_true")
    return parser.parse_args(argv)


def _project_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {"path": _project_path(resolved), "sha256": sha256_file(resolved)}


def _require_hash(path: Path, expected: str, *, name: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"{name} is absent: {resolved}")
    actual = sha256_file(resolved)
    if actual != expected:
        raise ValueError(f"{name} SHA-256 mismatch: expected {expected}, got {actual}")
    return resolved


def _write_json_exclusive(path: Path, payload: Any) -> None:
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {path}") from error


def _write_npz_exclusive(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    try:
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {path}") from error


def _panel_paths(panel: PanelName) -> tuple[Path, Path]:
    spec = PANEL_SPECS[panel]
    return (
        _require_hash(PROJECT_ROOT / spec["archive"], spec["archive_sha256"], name="focal archive"),
        _require_hash(
            PROJECT_ROOT / spec["metadata"],
            spec["metadata_sha256"],
            name="focal metadata",
        ),
    )


def _runtime_sources() -> dict[str, Path]:
    return {
        "runner": Path(__file__).resolve(),
        "selector": PROJECT_ROOT / "src/aiijc_puzzle/taska_structural_origin.py",
        "structural_border": PROJECT_ROOT / "src/aiijc_puzzle/taska_structural_border.py",
        "taska_matcher": PROJECT_ROOT / "src/aiijc_puzzle/taska_seam_matcher.py",
        "dirty_case_source": PROJECT_ROOT / "scripts/run_taska_seam_held300_diagnostic.py",
        "focal_replay_source": PROJECT_ROOT / "scripts/run_taska_focal_verifier_replay.py",
    }


def _freeze_candidate(
    *,
    panel: PanelName,
    rows: Sequence[Mapping[str, Any]],
    focal_archive: Path,
    focal_metadata: Path,
    targets: Path,
    output_dir: Path,
    device: torch.device,
) -> tuple[Path, Path, Path, float]:
    output_dir.mkdir(parents=True, exist_ok=False)
    frozen_path = output_dir / "frozen-target-free-eval.npz"
    metadata_path = output_dir / "frozen-target-free-eval.json"
    freeze_path = output_dir / "pre-score-freeze.json"
    v3 = _require_hash(
        PROJECT_ROOT / "artifacts/prior-taska/ckpt/seam_embed_v3.pt",
        V3_SHA256,
        name="TASKA v3 checkpoint",
    )
    local = _require_hash(
        PROJECT_ROOT / "artifacts/prior-taska/ckpt/seam_embed_local.pt",
        LOCAL_SHA256,
        name="TASKA local checkpoint",
    )
    matchers = load_default_taska_ensemble(v3.parent, device=device)
    lookup = _load_manifest_lookup()
    cache = CleanTileCache(targets.resolve())
    arrays: dict[str, np.ndarray] = {}
    frozen_rows: list[dict[str, Any]] = []
    started = perf_counter()
    with np.load(focal_archive, allow_pickle=False) as focal:
        for index, row in enumerate(rows):
            prefix = str(row["prefix"])
            source = str(row["source_filename"])
            draw = int(row["draw_index"])
            dirty = _dirty_case(cache, lookup[source], source, draw)
            dirty_sha = _parent_dirty_sha256(panel, dirty.dirty_tiles)
            if dirty.case_id != row["case_id"] or dirty_sha != row["dirty_sha256"]:
                raise RuntimeError("dirty-only recreation differs from frozen focal row")
            focal_layout = _strict_layout(focal[f"{prefix}__focal_layout"])
            unary = structural_border_unary(
                matchers,
                dirty.dirty_tiles,
                device=device,
                grid=GRID,
                slack=SLACK,
                sinkhorn_iterations=SINKHORN_ITERATIONS,
            )
            selected = select_structural_border_cyclic_origin(
                focal_layout,
                unary,
                grid=GRID,
            )
            layout = _strict_layout(selected.layout)
            arrays[f"{prefix}__selected_layout"] = layout
            frozen_rows.append(
                {
                    "prefix": prefix,
                    "case_id": dirty.case_id,
                    "source_filename": source,
                    "draw_index": draw,
                    "dirty_sha256": dirty_sha,
                    "selected_row_roll": selected.selected_row_roll,
                    "selected_column_roll": selected.selected_column_roll,
                    "selected_score": selected.selected_score,
                    "unchanged_score": selected.unchanged_score,
                    "changed": selected.changed,
                    "strict_original_tile_permutation": True,
                }
            )
            print(
                json.dumps(
                    {
                        "event": "taska_structural_origin_case_frozen_in_memory",
                        "panel": panel,
                        "case": index + 1,
                        "case_count": len(rows),
                        "roll": [selected.selected_row_roll, selected.selected_column_roll],
                        "changed": selected.changed,
                    }
                ),
                flush=True,
            )
    _write_npz_exclusive(frozen_path, arrays)
    _write_json_exclusive(
        metadata_path,
        {
            "schema": FROZEN_SCHEMA,
            "panel": panel,
            "feature_mode": FOCAL_MODE,
            "contains_exact_references_or_labels": False,
            "contains_target_ids_or_source_grid_coordinates": False,
            "contains_strict_original_tile_layouts": True,
            "relative_layout_is_unchanged_by_global_roll": True,
            "rolls_enumerated": GRID * GRID,
            "selection_objective": "sum structural_border_unary on physical border positions",
            "stable_tie_order": "row roll then column roll, ascending",
            "structural_border": {
                "matchers": ["v3", "local"],
                "slack": SLACK,
                "sinkhorn_iterations": SINKHORN_ITERATIONS,
            },
            "rows": frozen_rows,
        },
    )
    artifacts = {
        "focal_archive": _record(focal_archive),
        "focal_metadata": _record(focal_metadata),
        "validation_manifest": _record(MANIFEST),
        "matcher_v3": _record(v3),
        "matcher_local": _record(local),
        "frozen_candidate_archive": _record(frozen_path),
        "frozen_candidate_metadata": _record(metadata_path),
        **{name: _record(path) for name, path in _runtime_sources().items()},
    }
    _write_json_exclusive(
        freeze_path,
        {
            "schema": FREEZE_SCHEMA,
            "created_before_exact_reference_recreation": True,
            "contains_evaluation_references_or_labels": False,
            "panel": panel,
            "device": str(device),
            "artifacts": artifacts,
        },
    )
    return frozen_path, metadata_path, freeze_path, perf_counter() - started


def _validate_freeze(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("created_before_exact_reference_recreation") is not True:
        raise RuntimeError("candidate was not frozen before exact reconstruction")
    if payload.get("contains_evaluation_references_or_labels") is not False:
        raise RuntimeError("candidate freeze contains labels")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise RuntimeError("candidate freeze artifact roster is absent")
    for name, record in artifacts.items():
        if not isinstance(record, Mapping):
            raise RuntimeError(f"malformed artifact record: {name}")
        raw_path, expected = record.get("path"), record.get("sha256")
        if not isinstance(raw_path, str) or not isinstance(expected, str):
            raise RuntimeError(f"malformed artifact record: {name}")
        artifact = Path(raw_path)
        if not artifact.is_absolute():
            artifact = PROJECT_ROOT / artifact
        if sha256_file(artifact.resolve()) != expected:
            raise RuntimeError(f"pre-score artifact changed: {name}")
    return payload


def _summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = ("satisfied_adjacent_pairs", "adjacency_recall", "exact_tiles")
    arms = ("focal", "structural_origin")
    summary: dict[str, Any] = {
        "pair_denominator": PAIR_DENOMINATOR,
        "arms": {
            arm: {
                metric: float(np.mean([float(row[arm][metric]) for row in rows]))
                for metric in metrics
            }
            for arm in arms
        },
        "changed_case_count": sum(bool(row["changed"]) for row in rows),
    }
    full_panel = len(rows) == 32
    sources = [str(row["source_filename"]) for row in rows]
    deltas: dict[str, Any] = {}
    for index, metric in enumerate(metrics):
        values = [
            float(row["structural_origin"][metric]) - float(row["focal"][metric])
            for row in rows
        ]
        deltas[metric] = (
            _source_clustered_delta_ci(values, sources, seed=BOOTSTRAP_SEED + 90 + index)
            if full_panel
            else {"mean": float(np.mean(values)), "ci95_lower": None, "ci95_upper": None}
        )
    summary["structural_origin_minus_focal"] = deltas
    return summary


def _score_after_freeze(
    *,
    panel: PanelName,
    rows: Sequence[Mapping[str, Any]],
    focal_archive: Path,
    frozen_path: Path,
    metadata_path: Path,
    freeze_path: Path,
    targets: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _validate_freeze(freeze_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    frozen_rows = metadata.get("rows")
    if not isinstance(frozen_rows, list) or len(frozen_rows) != len(rows):
        raise RuntimeError("frozen candidate row roster changed")
    lookup = _load_manifest_lookup()
    cache = CleanTileCache(targets.resolve())
    scored: list[dict[str, Any]] = []
    with (
        np.load(focal_archive, allow_pickle=False) as focal,
        np.load(frozen_path, allow_pickle=False) as candidate,
    ):
        for parent_row, frozen_row in zip(rows, frozen_rows, strict=True):
            fields = ("prefix", "case_id", "source_filename", "draw_index", "dirty_sha256")
            if any(parent_row[field] != frozen_row[field] for field in fields):
                raise RuntimeError("frozen candidate identity changed")
            prefix = str(parent_row["prefix"])
            source = str(parent_row["source_filename"])
            draw = int(parent_row["draw_index"])
            dirty, reference = make_exact_synthetic_case(
                cache.load(lookup[source]),
                source_filename=source,
                draw_index=draw,
                seed=SYNTHETIC_SEED,
            )
            if (
                dirty.case_id != parent_row["case_id"]
                or reference.case_id != dirty.case_id
                or _parent_dirty_sha256(panel, dirty.tiles) != parent_row["dirty_sha256"]
            ):
                raise RuntimeError("exact scoring recreated a different synthetic case")
            exact = _strict_layout(reference.tile_at_position)
            focal_layout = _strict_layout(focal[f"{prefix}__focal_layout"])
            selected_layout = _strict_layout(candidate[f"{prefix}__selected_layout"])
            scored.append(
                {
                    "prefix": prefix,
                    "case_id": dirty.case_id,
                    "source_filename": source,
                    "draw_index": draw,
                    "selected_row_roll": int(frozen_row["selected_row_roll"]),
                    "selected_column_roll": int(frozen_row["selected_column_roll"]),
                    "changed": bool(frozen_row["changed"]),
                    "focal": _layout_metrics(focal_layout, exact),
                    "structural_origin": _layout_metrics(selected_layout, exact),
                }
            )
    return scored, _summarize(scored)


def run(args: argparse.Namespace) -> None:
    panel: PanelName = args.panel
    focal_archive, focal_metadata = _panel_paths(panel)
    rows = _validated_rows(focal_metadata, smoke_one=bool(args.smoke_one))
    targets = args.targets.resolve()
    if not targets.is_dir():
        raise ValueError(f"organizer targets are absent: {targets}")
    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable")
    output_dir = args.output_dir.resolve()
    started = perf_counter()
    frozen, metadata, freeze, inference_seconds = _freeze_candidate(
        panel=panel,
        rows=rows,
        focal_archive=focal_archive,
        focal_metadata=focal_metadata,
        targets=targets,
        output_dir=output_dir,
        device=device,
    )
    print(
        json.dumps(
            {
                "event": "taska_structural_origin_layouts_frozen_before_references",
                "panel": panel,
                "case_count": len(rows),
                "frozen_eval_sha256": sha256_file(frozen),
                "frozen_metadata_sha256": sha256_file(metadata),
                "pre_score_freeze_sha256": sha256_file(freeze),
            }
        ),
        flush=True,
    )
    scored_rows, metrics = _score_after_freeze(
        panel=panel,
        rows=rows,
        focal_archive=focal_archive,
        frozen_path=frozen,
        metadata_path=metadata,
        freeze_path=freeze,
        targets=targets,
    )
    delta = metrics["structural_origin_minus_focal"]
    exact_delta = float(delta["exact_tiles"]["mean"])
    pair_delta = float(delta["satisfied_adjacent_pairs"]["mean"])
    opened_gate = exact_delta > 0.0 and pair_delta >= -2.0
    report = {
        "schema": REPORT_SCHEMA,
        "status": "smoke-only" if args.smoke_one else "diagnostic-complete",
        "panel": {
            "name": panel,
            "historically_opened": True,
            "fresh_promotion_claimed": False,
            "case_count": len(scored_rows),
            "full_registered_panel": len(scored_rows) == 32,
        },
        "candidate": {
            "starting_layout": "recovered focal train_exact_top5 only",
            "rolls_enumerated": GRID * GRID,
            "objective": "maximum structural border unary on border positions",
            "seam_or_border_weight_mixing": False,
            "stable_row_major_tie": True,
            "target_or_reference_used_during_inference": False,
            "structural_border": {
                "matchers": ["v3", "local"],
                "slack": SLACK,
                "sinkhorn_iterations": SINKHORN_ITERATIONS,
            },
        },
        "frozen_eval": {
            "archive": _record(frozen),
            "metadata": _record(metadata),
            "pre_score_freeze": _record(freeze),
            "layouts_frozen_before_exact_reference_recreation": True,
            "contains_exact_references_or_labels": False,
        },
        "runtime_sources": {name: _record(path) for name, path in _runtime_sources().items()},
        "measurement": {
            "all_layouts_strict": all(
                row[arm]["strict_permutation"]
                for row in scored_rows
                for arm in ("focal", "structural_origin")
            ),
            "opened_gate_definition": "exact delta > 0 and pair delta >= -2 per board",
            "opened_gate_passed": opened_gate if panel == "opened32" else None,
        },
        "metrics": metrics,
        "rows": scored_rows,
        "runtime_seconds": {
            "target_free_inference": inference_seconds,
            "total": perf_counter() - started,
        },
        "bootstrap": {
            "seed_base": BOOTSTRAP_SEED + 90,
            "resamples": BOOTSTRAP_RESAMPLES,
            "source_clustered": True,
        },
    }
    _write_json_exclusive(output_dir / "report.json", report)
    print(json.dumps(report["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    run(parse_args())

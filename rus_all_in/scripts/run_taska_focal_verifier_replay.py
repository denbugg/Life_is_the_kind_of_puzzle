#!/usr/bin/env python3
"""Replay one fixed historical focal-verifier contract on a frozen TASKA panel.

The parent archive supplies matcher costs and frozen candidate membership.  The
dirty board is recreated independently and hash-checked before the recovered
verifier sees it.  Verifier logits affect only component-build order; original
matcher costs remain untouched for component placement and Hungarian fill.

The logits and strict layouts are written and hash-rostered before any exact
reference is constructed.  This script does not tune a threshold, add/drop an
edge, restore pixels, or consume a target-derived id during candidate inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np
import torch

from aiijc_puzzle.layout_evaluation import evaluate_layout
from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.raw_tail_global_solver import RawTailEdge, RawTailGlobalConfig
from aiijc_puzzle.synthetic_socket_evaluation import make_exact_synthetic_case
from aiijc_puzzle.taska_focal_verifier import (
    TASKA_FOCAL_FEATURE_TOP_K,
    TASKA_FOCAL_VERIFIER_SHA256,
    FocalFeatureMode,
    load_taska_focal_verifier,
    score_focal_edges,
    solve_focal_raw_tail_global,
)

try:
    from scripts.run_taska_seam_held300_diagnostic import (
        SYNTHETIC_SEED,
        CleanTileCache,
        _dirty_case,
        _dirty_sha256,
    )
except ModuleNotFoundError:  # Direct ``python scripts/run_*.py`` execution.
    from run_taska_seam_held300_diagnostic import (
        SYNTHETIC_SEED,
        CleanTileCache,
        _dirty_case,
        _dirty_sha256,
    )

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRID = 24
COUNT = GRID * GRID
PAIR_DENOMINATOR = 2 * GRID * (GRID - 1)
BOOTSTRAP_SEED = 934_711_727
BOOTSTRAP_RESAMPLES = 20_000
MANIFEST = PROJECT_ROOT / "data/interim/validation_manifest.json"
MANIFEST_SHA256 = "4781e370e092ad272c63e6d5165b25951aaf93fae5fde74c75d534a9e8efc9da"
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "artifacts/prior-taska/ckpt/verify_pair_best.pt"

PanelName = Literal["opened32", "held300"]

PANEL_ARTIFACTS: dict[PanelName, dict[str, str]] = {
    "opened32": {
        "archive": "outputs/taska-seam-replay/opened32-mps-v1/frozen-target-free-eval.npz",
        "archive_sha256": "1880940897caeec6b87631d53e1aede1f809955a7acd3e56da9bcf432939e994",
        "metadata": "outputs/taska-seam-replay/opened32-mps-v1/frozen-target-free-eval.json",
        "metadata_sha256": "f327664bc9db353b53b8f05738f94a5baaf8eefec1c708ae92f5032c37ce6eaf",
    },
    "held300": {
        "archive": (
            "outputs/taska-seam-replay/held300-diagnostic-mps-v1/"
            "frozen-target-free-eval.npz"
        ),
        "archive_sha256": "0876d7acd6be7ebe863f831de568586c837da52cc603df4ff8d7c5a6b0d441df",
        "metadata": (
            "outputs/taska-seam-replay/held300-diagnostic-mps-v1/"
            "frozen-target-free-eval.json"
        ),
        "metadata_sha256": "91710486233f45bda2b8aab019c508d1e6a0f75e282a6ce9a47aee93d9bf0a8d",
    },
}

SOLVER_CONFIG = RawTailGlobalConfig(
    baseline_quantile=0.15,
    search_rounds=6,
    border_weight=0.0,
    random_seed=0,
    component_cap=0,
    fill_rounds=1,
)

FROZEN_SCHEMA = "aiijc-taska-focal-verifier-frozen-target-free-v1"
FREEZE_SCHEMA = "aiijc-taska-focal-verifier-pre-score-freeze-v1"
REPORT_SCHEMA = "aiijc-taska-focal-verifier-replay-report-v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=tuple(PANEL_ARTIFACTS), required=True)
    parser.add_argument(
        "--mode",
        choices=tuple(TASKA_FOCAL_FEATURE_TOP_K),
        required=True,
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument(
        "--smoke-one",
        action="store_true",
        help="evaluate only the first row; no panel-level claim is emitted",
    )
    return parser.parse_args(argv)


def _write_json_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {path}") from error


def _write_npz_exclusive(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
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


def _require_hash(path: Path, expected: str, *, name: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"{name} does not exist: {resolved}")
    actual = sha256_file(resolved)
    if actual != expected:
        raise ValueError(f"{name} SHA-256 mismatch: expected {expected}, got {actual}")
    return resolved


def _strict_layout(value: Any) -> np.ndarray:
    layout = np.ascontiguousarray(value, dtype=np.int32)
    if layout.shape != (COUNT,) or not np.array_equal(np.sort(layout), np.arange(COUNT)):
        raise ValueError("layout is not a strict original-tile permutation")
    return layout


def _parent_dirty_sha256(panel: PanelName, tiles: np.ndarray) -> str:
    """Match each already-frozen parent's historical byte-hash convention."""

    value = np.ascontiguousarray(tiles)
    if panel == "opened32":
        return hashlib.sha256(value.tobytes()).hexdigest()
    if panel == "held300":
        return _dirty_sha256(value)
    raise ValueError(f"unsupported panel: {panel}")


def _load_manifest_lookup() -> dict[str, Mapping[str, Any]]:
    _require_hash(MANIFEST, MANIFEST_SHA256, name="validation manifest")
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    splits = payload.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("validation manifest has no split mapping")
    lookup: dict[str, Mapping[str, Any]] = {}
    for records in splits.values():
        if not isinstance(records, list):
            raise ValueError("validation manifest split is not a list")
        for record in records:
            if not isinstance(record, Mapping):
                raise ValueError("validation manifest record is malformed")
            name = record.get("filename")
            if not isinstance(name, str) or name in lookup:
                raise ValueError("validation manifest filenames are malformed or duplicated")
            lookup[name] = record
    if len(lookup) != 7000:
        raise ValueError("validation manifest must cover exactly 7000 organizer-train sources")
    return lookup


def _panel_paths(panel: PanelName) -> tuple[Path, Path]:
    spec = PANEL_ARTIFACTS[panel]
    archive = _require_hash(
        PROJECT_ROOT / spec["archive"],
        spec["archive_sha256"],
        name=f"{panel} frozen archive",
    )
    metadata = _require_hash(
        PROJECT_ROOT / spec["metadata"],
        spec["metadata_sha256"],
        name=f"{panel} frozen metadata",
    )
    return archive, metadata


def _validated_rows(metadata_path: Path, *, smoke_one: bool) -> list[Mapping[str, Any]]:
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if payload.get("contains_exact_references_or_labels") is not False:
        raise ValueError("parent metadata is not target-free")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != 32:
        raise ValueError("parent metadata must contain the registered 32 cases")
    required = {
        "prefix",
        "case_id",
        "source_filename",
        "draw_index",
        "dirty_sha256",
        "candidate_edge_count",
    }
    for row in rows:
        if not isinstance(row, Mapping) or not required <= set(row):
            raise ValueError("parent metadata row is malformed")
    selected = rows[:1] if smoke_one else rows
    if not smoke_one:
        groups: dict[str, set[int]] = defaultdict(set)
        for row in selected:
            groups[str(row["source_filename"])].add(int(row["draw_index"]))
        if len(groups) != 16 or any(draws != {0, 1} for draws in groups.values()):
            raise ValueError("panel is not the registered 16-source x 2-draw roster")
    return selected


def _edges_from_archive(archive: Any, prefix: str) -> tuple[RawTailEdge, ...]:
    sources = np.asarray(archive[f"{prefix}__edge_source"], dtype=np.int64)
    targets = np.asarray(archive[f"{prefix}__edge_target"], dtype=np.int64)
    axes = np.asarray(archive[f"{prefix}__edge_axis"], dtype=np.uint8)
    if not (sources.ndim == targets.ndim == axes.ndim == 1):
        raise ValueError("frozen edge arrays must be one-dimensional")
    if not (len(sources) == len(targets) == len(axes)):
        raise ValueError("frozen edge arrays are not aligned")
    if not np.isin(axes, (0, 1)).all():
        raise ValueError("frozen edge axis encoding is malformed")
    return tuple(
        RawTailEdge(int(source), int(target), "right" if int(axis) == 0 else "down")
        for source, target, axis in zip(sources, targets, axes, strict=True)
    )


def _finite_cost(archive: Any, key: str) -> np.ndarray:
    value = np.asarray(archive[key], dtype=np.float64)
    if value.shape != (COUNT, COUNT) or not np.isfinite(value).all():
        raise ValueError(f"{key} is not one finite 576x576 matrix")
    return np.ascontiguousarray(value)


def _runtime_sources() -> dict[str, Path]:
    return {
        "replay_runner": Path(__file__).resolve(),
        "focal_verifier": PROJECT_ROOT / "src/aiijc_puzzle/taska_focal_verifier.py",
        "prioritized_tail": PROJECT_ROOT / "src/aiijc_puzzle/taska_edge_calibrator.py",
        "frozen_raw_tail_solver": PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py",
        "dirty_case_source": PROJECT_ROOT / "scripts/run_taska_seam_held300_diagnostic.py",
    }


def _freeze_candidate(
    *,
    panel: PanelName,
    mode: FocalFeatureMode,
    checkpoint: Path,
    targets: Path,
    parent_archive_path: Path,
    parent_metadata_path: Path,
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    device: torch.device,
) -> tuple[Path, Path, Path, float]:
    output_dir.mkdir(parents=True, exist_ok=False)
    frozen_path = output_dir / "frozen-target-free-eval.npz"
    metadata_path = output_dir / "frozen-target-free-eval.json"
    freeze_path = output_dir / "pre-score-freeze.json"
    lookup = _load_manifest_lookup()
    cache = CleanTileCache(targets.resolve())
    model = load_taska_focal_verifier(checkpoint, device=device)
    arrays: dict[str, np.ndarray] = {}
    frozen_rows: list[dict[str, Any]] = []
    started = perf_counter()
    with np.load(parent_archive_path, allow_pickle=False) as parent:
        for index, row in enumerate(rows):
            prefix = str(row["prefix"])
            source = str(row["source_filename"])
            draw = int(row["draw_index"])
            if source not in lookup:
                raise ValueError(f"source is absent from validation manifest: {source}")
            dirty = _dirty_case(cache, lookup[source], source, draw)
            dirty_sha = _parent_dirty_sha256(panel, dirty.dirty_tiles)
            if (
                dirty.case_id != row["case_id"]
                or dirty_sha != row["dirty_sha256"]
                or dirty.source_filename != source
                or dirty.draw_index != draw
            ):
                raise RuntimeError("dirty-only recreation differs from the frozen parent row")
            cost_right = _finite_cost(parent, f"{prefix}__cost_right")
            cost_down = _finite_cost(parent, f"{prefix}__cost_down")
            edges = _edges_from_archive(parent, prefix)
            if len(edges) != int(row["candidate_edge_count"]):
                raise RuntimeError("frozen candidate membership count changed")
            scored = score_focal_edges(
                model,
                dirty.dirty_tiles,
                cost_right,
                cost_down,
                edges,
                mode=mode,
                grid=GRID,
                device=device,
            )
            solved = solve_focal_raw_tail_global(
                cost_right,
                cost_down,
                scored,
                border_unary=None,
                grid=GRID,
                config=SOLVER_CONFIG,
            )
            layout = _strict_layout(solved.layout)
            arrays[f"{prefix}__focal_logits"] = scored.logits
            arrays[f"{prefix}__focal_features"] = scored.features
            arrays[f"{prefix}__focal_layout"] = layout
            frozen_rows.append(
                {
                    "prefix": prefix,
                    "case_id": dirty.case_id,
                    "source_filename": source,
                    "draw_index": draw,
                    "dirty_sha256": dirty_sha,
                    "candidate_edge_count": len(edges),
                    "feature_mode": mode,
                    "feature_top_k": TASKA_FOCAL_FEATURE_TOP_K[mode],
                    "checkpoint_sha256": TASKA_FOCAL_VERIFIER_SHA256,
                    "logit_min": float(np.min(scored.logits)),
                    "logit_max": float(np.max(scored.logits)),
                    "solver_diagnostics": solved.diagnostics.as_dict(),
                }
            )
            print(
                json.dumps(
                    {
                        "event": "taska_focal_target_free_case_frozen_in_memory",
                        "panel": panel,
                        "mode": mode,
                        "case": index + 1,
                        "case_count": len(rows),
                        "source_filename": source,
                        "draw_index": draw,
                        "candidate_edge_count": len(edges),
                        "strict": True,
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
            "feature_mode": mode,
            "feature_top_k": TASKA_FOCAL_FEATURE_TOP_K[mode],
            "contains_exact_references_or_labels": False,
            "contains_target_ids_or_source_grid_coordinates": False,
            "contains_dirty_derived_logits_and_features": True,
            "contains_frozen_harvest_membership": False,
            "parent_archive_supplies_frozen_harvest_membership": True,
            "verifier_adds_or_drops_edges": False,
            "contains_strict_original_tile_layouts": True,
            "original_costs_retained_for_placement_and_fill": True,
            "rows": frozen_rows,
        },
    )
    artifacts = {
        "parent_frozen_archive": _record(parent_archive_path),
        "parent_frozen_metadata": _record(parent_metadata_path),
        "validation_manifest": _record(MANIFEST),
        "recovered_focal_checkpoint": _record(checkpoint),
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
            "feature_mode": mode,
            "device": str(device),
            "checkpoint_was_not_copied_into_workspace": True,
            "artifacts": artifacts,
        },
    )
    return frozen_path, metadata_path, freeze_path, perf_counter() - started


def _validate_freeze(freeze_path: Path) -> Mapping[str, Any]:
    payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    if payload.get("created_before_exact_reference_recreation") is not True:
        raise RuntimeError("pre-score freeze timing contract changed")
    if payload.get("contains_evaluation_references_or_labels") is not False:
        raise RuntimeError("pre-score freeze unexpectedly contains evaluation labels")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise RuntimeError("pre-score artifact roster is missing")
    for name, record in artifacts.items():
        if not isinstance(record, Mapping):
            raise RuntimeError(f"pre-score artifact record is malformed: {name}")
        raw_path = record.get("path")
        expected = record.get("sha256")
        if not isinstance(raw_path, str) or not isinstance(expected, str):
            raise RuntimeError(f"pre-score artifact record is malformed: {name}")
        path = Path(raw_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if sha256_file(path.resolve()) != expected:
            raise RuntimeError(f"pre-score artifact changed: {name}")
    return payload


def _layout_metrics(layout: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    result = evaluate_layout(layout, reference, reference_is_exact=True)
    if result.adjacency_total != PAIR_DENOMINATOR:
        raise RuntimeError("adjacency denominator changed")
    return {
        "satisfied_adjacent_pairs": int(result.adjacency_correct),
        "adjacency_recall": float(result.adjacency),
        "exact_tiles": int(result.correct_tile_count),
        "strict_permutation": True,
    }


def _source_clustered_delta_ci(
    values: Sequence[float],
    sources: Sequence[str],
    *,
    seed: int,
) -> dict[str, Any]:
    if len(values) != len(sources) or not values:
        raise ValueError("delta values and sources must be aligned and non-empty")
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, source in zip(values, sources, strict=True):
        if not math.isfinite(float(value)):
            raise ValueError("delta values must be finite")
        grouped[str(source)].append(float(value))
    if any(len(group) != 2 for group in grouped.values()):
        raise ValueError("every source cluster must contain exactly two draws")
    source_means = np.asarray([np.mean(grouped[name]) for name in sorted(grouped)])
    generator = np.random.default_rng(seed)
    distribution = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_RESAMPLES, 2048):
        stop = min(start + 2048, BOOTSTRAP_RESAMPLES)
        indices = generator.integers(
            0,
            len(source_means),
            size=(stop - start, len(source_means)),
        )
        distribution[start:stop] = source_means[indices].mean(axis=1)
    return {
        "mean": float(np.mean(values)),
        "source_cluster_mean": float(source_means.mean()),
        "ci95_lower": float(np.quantile(distribution, 0.025)),
        "ci95_upper": float(np.quantile(distribution, 0.975)),
        "source_count": len(source_means),
        "case_count": len(values),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": seed,
    }


def _summarize(scored_rows: Sequence[Mapping[str, Any]], *, full_panel: bool) -> dict[str, Any]:
    metrics = ("satisfied_adjacent_pairs", "adjacency_recall", "exact_tiles")
    arms = ("raw", "focal")
    summary: dict[str, Any] = {
        "pair_denominator": PAIR_DENOMINATOR,
        "arms": {
            arm: {
                metric: float(np.mean([float(row[arm][metric]) for row in scored_rows]))
                for metric in metrics
            }
            for arm in arms
        },
    }
    sources = [str(row["source_filename"]) for row in scored_rows]
    deltas: dict[str, Any] = {}
    for index, metric in enumerate(metrics):
        values = [
            float(row["focal"][metric]) - float(row["raw"][metric])
            for row in scored_rows
        ]
        deltas[metric] = (
            _source_clustered_delta_ci(values, sources, seed=BOOTSTRAP_SEED + index)
            if full_panel
            else {
                "mean": float(np.mean(values)),
                "ci95_lower": None,
                "ci95_upper": None,
                "smoke_only": True,
            }
        )
    summary["focal_minus_raw"] = deltas
    return summary


def _score_after_freeze(
    *,
    panel: PanelName,
    parent_archive_path: Path,
    rows: Sequence[Mapping[str, Any]],
    frozen_path: Path,
    metadata_path: Path,
    freeze_path: Path,
    targets: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _validate_freeze(freeze_path)
    candidate_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    candidate_rows = candidate_metadata.get("rows")
    if not isinstance(candidate_rows, list) or len(candidate_rows) != len(rows):
        raise RuntimeError("frozen candidate row roster changed")
    lookup = _load_manifest_lookup()
    cache = CleanTileCache(targets.resolve())
    scored_rows: list[dict[str, Any]] = []
    with (
        np.load(parent_archive_path, allow_pickle=False) as parent,
        np.load(frozen_path, allow_pickle=False) as candidate,
    ):
        for parent_row, candidate_row in zip(rows, candidate_rows, strict=True):
            identity_fields = (
                "prefix",
                "case_id",
                "source_filename",
                "draw_index",
                "dirty_sha256",
            )
            if any(parent_row[field] != candidate_row[field] for field in identity_fields):
                raise RuntimeError("parent and candidate frozen row identities differ")
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
                raise RuntimeError("scoring recreated a different synthetic case")
            exact = _strict_layout(reference.tile_at_position)
            raw_layout = _strict_layout(parent[f"{prefix}__taska_layout"])
            focal_layout = _strict_layout(candidate[f"{prefix}__focal_layout"])
            scored_rows.append(
                {
                    "prefix": prefix,
                    "case_id": dirty.case_id,
                    "source_filename": source,
                    "draw_index": draw,
                    "raw": _layout_metrics(raw_layout, exact),
                    "focal": _layout_metrics(focal_layout, exact),
                }
            )
    full_panel = len(scored_rows) == 32
    return scored_rows, _summarize(scored_rows, full_panel=full_panel)


def run(args: argparse.Namespace) -> None:
    panel: PanelName = args.panel
    mode: FocalFeatureMode = args.mode
    parent_archive, parent_metadata = _panel_paths(panel)
    rows = _validated_rows(parent_metadata, smoke_one=bool(args.smoke_one))
    checkpoint = _require_hash(
        args.checkpoint,
        TASKA_FOCAL_VERIFIER_SHA256,
        name="recovered focal checkpoint",
    )
    if not args.targets.resolve().is_dir():
        raise ValueError(f"organizer-train target directory is absent: {args.targets}")
    output_dir = args.output_dir.resolve()
    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable")
    torch.use_deterministic_algorithms(device.type == "cpu")
    started = perf_counter()
    frozen, metadata, freeze, inference_seconds = _freeze_candidate(
        panel=panel,
        mode=mode,
        checkpoint=checkpoint,
        targets=args.targets,
        parent_archive_path=parent_archive,
        parent_metadata_path=parent_metadata,
        rows=rows,
        output_dir=output_dir,
        device=device,
    )
    print(
        json.dumps(
            {
                "event": "taska_focal_logits_and_layouts_frozen_before_references",
                "panel": panel,
                "mode": mode,
                "case_count": len(rows),
                "frozen_eval_sha256": sha256_file(frozen),
                "frozen_metadata_sha256": sha256_file(metadata),
                "pre_score_freeze_sha256": sha256_file(freeze),
                "exact_reference_persisted": False,
            }
        ),
        flush=True,
    )
    scored_rows, metrics = _score_after_freeze(
        panel=panel,
        parent_archive_path=parent_archive,
        rows=rows,
        frozen_path=frozen,
        metadata_path=metadata,
        freeze_path=freeze,
        targets=args.targets,
    )
    full_panel = len(scored_rows) == 32
    strict = all(
        bool(row[arm]["strict_permutation"])
        for row in scored_rows
        for arm in ("raw", "focal")
    )
    pair_delta = metrics["focal_minus_raw"]["satisfied_adjacent_pairs"]
    report = {
        "schema": REPORT_SCHEMA,
        "status": "smoke-only" if args.smoke_one else "diagnostic-complete",
        "panel": {
            "name": panel,
            "historically_opened": True,
            "fresh_promotion_claimed": False,
            "case_count": len(scored_rows),
            "full_registered_panel": full_panel,
        },
        "candidate": {
            "feature_mode": mode,
            "feature_top_k": TASKA_FOCAL_FEATURE_TOP_K[mode],
            "checkpoint": _record(checkpoint),
            "candidate_membership_unchanged": True,
            "verifier_adds_or_drops_edges": False,
            "verifier_logits_used_only_for_component_order": True,
            "original_costs_retained_for_placement_and_fill": True,
            "target_ids_or_exact_references_used_during_inference": False,
            "solver": asdict(SOLVER_CONFIG),
        },
        "parent": {
            "archive": _record(parent_archive),
            "metadata": _record(parent_metadata),
        },
        "frozen_eval": {
            "archive": _record(frozen),
            "metadata": _record(metadata),
            "pre_score_freeze": _record(freeze),
            "logits_and_layouts_frozen_before_references": True,
            "contains_exact_references_or_labels": False,
        },
        "runtime_sources": {name: _record(path) for name, path in _runtime_sources().items()},
        "measurement": {
            "all_layouts_strict": strict,
            "valid": full_panel and strict,
            "pair_delta_nonnegative": float(pair_delta["mean"]) >= 0.0,
            "pair_delta_ci_excludes_zero": (
                pair_delta["ci95_lower"] is not None
                and float(pair_delta["ci95_lower"]) > 0.0
            ),
        },
        "metrics": metrics,
        "rows": scored_rows,
        "runtime_seconds": {
            "target_free_inference_and_solver": inference_seconds,
            "total": perf_counter() - started,
        },
    }
    _write_json_exclusive(output_dir / "report.json", report)
    print(json.dumps(report["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    run(parse_args())

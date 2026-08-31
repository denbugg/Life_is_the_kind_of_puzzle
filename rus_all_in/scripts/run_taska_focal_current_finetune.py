#!/usr/bin/env python3
"""Run the one preregistered current-harvest TASKA focal fine-tune.

The fixed sequence is 96 supervised train boards, 32 disjoint local-gate
boards, then the unchanged historical held32 panel only when fine-tuning does
not reduce local-gate satisfied pairs relative to the recovered top-5 verifier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from aiijc_puzzle.layout_evaluation import evaluate_layout
from aiijc_puzzle.protocol import compute_protocol_digest, sha256_file
from aiijc_puzzle.raw_tail_global_solver import (
    RawTailEdge,
    RawTailGlobalConfig,
    solve_raw_tail_global,
)
from aiijc_puzzle.synthetic_socket_evaluation import make_exact_synthetic_case
from aiijc_puzzle.taska_focal_current_finetune import (
    FocalTrainingBoard,
    load_finetuned_checkpoint,
    make_focal_training_board,
    prepare_finetune_model,
    save_finetuned_checkpoint,
    score_current_focal_edges,
    solve_current_focal_edges,
    train_fixed_focal_model,
)
from aiijc_puzzle.taska_focal_verifier import load_taska_focal_verifier
from aiijc_puzzle.taska_seam_matcher import (
    load_default_taska_ensemble,
    match_taska_tiles,
)

try:
    from scripts.run_taska_seam_held300_diagnostic import (
        MATCHER_CONFIG,
        SYNTHETIC_SEED,
        CleanTileCache,
        _dirty_case,
        _dirty_sha256,
    )
except ModuleNotFoundError:
    from run_taska_seam_held300_diagnostic import (
        MATCHER_CONFIG,
        SYNTHETIC_SEED,
        CleanTileCache,
        _dirty_case,
        _dirty_sha256,
    )

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/taska_focal_current_finetune_v1.json"
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/taska-focal-current-finetune/v1"
GRID = 24
COUNT = GRID * GRID
PAIR_DENOMINATOR = 2 * GRID * (GRID - 1)
CONFIG_SCHEMA = "aiijc-taska-focal-current-finetune-v1"
REPORT_SCHEMA = "aiijc-taska-focal-current-finetune-report-v1"
CONFIG_SHA256 = "94dce4a73410f9d40cf52b136af27116ddabf79dd5948c68307839e1e7bc6a23"
TRAIN_DIGEST = "7b3ff55d8e73097fccfe2aeae45528c13734c7793a9fd0f8ee1dfdf4893cd7fe"
GATE_DIGEST = "f516f12e8943580ab62e17cd6d4064dc519aa20df6485bf5bca34030beaa2bc3"
PARENT_ROSTER_SHA256 = "e940944865f0a4f93e6f6a9782c33c2da1566ffb8ef1253e88bec369d30c630c"
HELD_ARCHIVE = PROJECT_ROOT / (
    "outputs/taska-seam-replay/held300-diagnostic-mps-v1/frozen-target-free-eval.npz"
)
HELD_ARCHIVE_SHA256 = "0876d7acd6be7ebe863f831de568586c837da52cc603df4ff8d7c5a6b0d441df"
HELD_METADATA = PROJECT_ROOT / (
    "outputs/taska-seam-replay/held300-diagnostic-mps-v1/frozen-target-free-eval.json"
)
HELD_METADATA_SHA256 = "91710486233f45bda2b8aab019c508d1e6a0f75e282a6ce9a47aee93d9bf0a8d"
SOLVER_CONFIG = RawTailGlobalConfig(
    baseline_quantile=0.15,
    search_rounds=6,
    border_weight=0.0,
    random_seed=0,
    component_cap=0,
    fill_rounds=1,
)


def _digest_names(names: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {path}") from error


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    try:
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {path}") from error


def _strict_layout(value: Any) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=np.int32)
    if result.shape != (COUNT,) or not np.array_equal(np.sort(result), np.arange(COUNT)):
        raise ValueError("solver output is not a strict 576-tile permutation")
    return result


def _edges_from_arrays(source: Any, target: Any, axis: Any) -> tuple[RawTailEdge, ...]:
    sources = np.asarray(source, dtype=np.int64)
    targets = np.asarray(target, dtype=np.int64)
    axes = np.asarray(axis, dtype=np.uint8)
    if not (sources.ndim == targets.ndim == axes.ndim == 1):
        raise ValueError("edge arrays must be vectors")
    if not (len(sources) == len(targets) == len(axes)) or not np.isin(axes, (0, 1)).all():
        raise ValueError("edge arrays are malformed")
    return tuple(
        RawTailEdge(int(s), int(t), "right" if int(a) == 0 else "down")
        for s, t, a in zip(sources, targets, axes, strict=True)
    )


def _load_config(path: Path) -> tuple[dict[str, Any], tuple[str, ...], tuple[str, ...]]:
    path = path.resolve()
    if sha256_file(path) != CONFIG_SHA256:
        raise ValueError("fine-tune preregistration SHA-256 mismatch")
    sidecar = Path(f"{path}.sha256")
    if not sidecar.is_file() or sidecar.read_text().split()[0] != CONFIG_SHA256:
        raise ValueError("fine-tune preregistration sidecar mismatch")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("fine-tune preregistration schema mismatch")
    if config.get("protocol", {}).get("single_fixed_arm") is not True:
        raise ValueError("fine-tune is not preregistered as a single fixed arm")
    if config.get("training") != {
        "initial_checkpoint": {
            "path": "artifacts/prior-taska/ckpt/verify_pair_best.pt",
            "sha256": "3bcc89a12e7b539304484b441688b4b9fb1c3711e918befed9cdef7c17f776e7",
        },
        "feature_mode": "train_exact_top5",
        "candidate_distribution": "only current TASKA harvested edges",
        "loss": (
            "mean softplus(-(positive_logit-negative_logit)) over every true-false "
            "harvested-edge pair within each board"
        ),
        "optimizer": "AdamW",
        "learning_rate": 0.00003,
        "weight_decay": 0.01,
        "epochs": 2,
        "board_batch_size": 1,
        "gradient_clip_norm": 1.0,
        "seed": 2026083103,
        "freeze_raw_score_prior": True,
        "checkpoint_selection": "final epoch only; no epoch or arm selection",
    }:
        raise ValueError("fixed training arm changed")
    roster_spec = config["selection"]["parent_roster"]
    roster_path = PROJECT_ROOT / roster_spec["path"]
    if sha256_file(roster_path) != PARENT_ROSTER_SHA256:
        raise ValueError("parent train256 roster changed")
    roster = json.loads(roster_path.read_text(encoding="utf-8"))["source_filenames"]
    if len(roster) != 256 or _digest_names(roster) != roster_spec["source_order_digest"]:
        raise ValueError("parent train256 roster contract changed")
    train_names = tuple(roster[:96])
    gate_names = tuple(roster[96:128])
    if _digest_names(train_names) != TRAIN_DIGEST or _digest_names(gate_names) != GATE_DIGEST:
        raise ValueError("fixed fine-tune train/gate roster changed")
    if set(train_names) & set(gate_names):
        raise ValueError("fine-tune train and gate rosters overlap")
    exposed: set[str] = set()
    for key in ("opened32_recipe", "held300_recipe", "fresh32_recipe"):
        spec = config["artifacts"][key]
        artifact = PROJECT_ROOT / spec["path"]
        if sha256_file(artifact) != spec["sha256"]:
            raise ValueError(f"{key} changed")
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        names = payload.get("panel", {}).get("source_filenames", ())
        if not isinstance(names, list):
            raise ValueError(f"{key} source roster is malformed")
        exposed.update(str(name) for name in names)
    if (set(train_names) | set(gate_names)) & exposed:
        raise ValueError("fine-tune roster overlaps an excluded evaluation panel")
    for spec in config["artifacts"].values():
        artifact = PROJECT_ROOT / spec["path"]
        if sha256_file(artifact) != spec["sha256"]:
            raise ValueError(f"artifact changed: {artifact}")
    return config, train_names, gate_names


def _manifest_lookup(config: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    path = PROJECT_ROOT / config["artifacts"]["manifest"]["path"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol_digest") != compute_protocol_digest(payload):
        raise ValueError("manifest protocol digest mismatch")
    rows = [row for split in payload["splits"].values() for row in split]
    lookup = {str(row["filename"]): row for row in rows}
    if len(rows) != 7000 or len(lookup) != 7000:
        raise ValueError("manifest must contain 7000 unique organizer-train rows")
    return lookup


def _reference(
    cache: CleanTileCache,
    record: Mapping[str, Any],
    source_name: str,
    draw: int,
    dirty: np.ndarray,
) -> np.ndarray:
    synthetic, reference = make_exact_synthetic_case(
        cache.load(record),
        source_filename=source_name,
        draw_index=draw,
        seed=SYNTHETIC_SEED,
    )
    if not np.array_equal(synthetic.tiles, dirty):
        raise RuntimeError("dirty/reference synthetic construction diverged")
    return np.ascontiguousarray(reference.tile_at_position, dtype=np.int32)


def _build_training_archive(
    names: Sequence[str],
    lookup: Mapping[str, Mapping[str, Any]],
    cache: CleanTileCache,
    *,
    matchers: Mapping[str, torch.nn.Module],
    device: torch.device,
    output: Path,
) -> tuple[list[FocalTrainingBoard], dict[str, Any]]:
    boards: list[FocalTrainingBoard] = []
    offsets = [0]
    patches: list[np.ndarray] = []
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    edge_counts: list[int] = []
    positive_counts: list[int] = []
    started = perf_counter()
    for index, name in enumerate(names):
        record = lookup[name]
        dirty_case = _dirty_case(cache, record, name, 0)
        matched = match_taska_tiles(
            dirty_case.dirty_tiles,
            matchers,
            config=MATCHER_CONFIG,
            device=device,
        )
        reference = _reference(cache, record, name, 0, dirty_case.dirty_tiles)
        board = make_focal_training_board(
            dirty_case.dirty_tiles,
            matched.cost_right,
            matched.cost_down,
            matched.candidate_edges,
            reference,
            source_filename=name,
        )
        boards.append(board)
        patches.append(board.patches.astype(np.uint8))
        features.append(board.features)
        labels.append(board.labels)
        edge_counts.append(len(board.labels))
        positive_counts.append(int(board.labels.sum()))
        offsets.append(offsets[-1] + len(board.labels))
        print(
            json.dumps(
                {
                    "event": "fine_tune_train_case_built",
                    "case": index + 1,
                    "case_count": len(names),
                    "source": name,
                    "edges": len(board.labels),
                    "positive": int(board.labels.sum()),
                }
            ),
            flush=True,
        )
    _write_npz(
        output,
        {
            "patches_uint8": np.concatenate(patches),
            "features": np.concatenate(features),
            "labels": np.concatenate(labels),
            "offsets": np.asarray(offsets, dtype=np.int32),
            "source_filenames": np.asarray(names),
        },
    )
    return boards, {
        "source_count": len(names),
        "edge_count": int(sum(edge_counts)),
        "positive_count": int(sum(positive_counts)),
        "positive_fraction": float(sum(positive_counts) / sum(edge_counts)),
        "runtime_seconds": perf_counter() - started,
        "artifact": {"path": str(output.relative_to(PROJECT_ROOT)), "sha256": sha256_file(output)},
    }


def _metric_row(layouts: Mapping[str, np.ndarray], reference: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm, layout in layouts.items():
        metric = evaluate_layout(layout, reference, reference_is_exact=True)
        result[arm] = {
            "pairs": metric.adjacency_correct,
            "recall": metric.adjacency,
            "exact": metric.correct_tile_count,
        }
    return result


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    arms = tuple(rows[0]["metrics"])
    recovered_pairs = np.asarray(
        [row["metrics"]["recovered_top5"]["pairs"] for row in rows]
    )
    summary: dict[str, Any] = {}
    for arm in arms:
        pairs = np.asarray([row["metrics"][arm]["pairs"] for row in rows], dtype=float)
        exact = np.asarray([row["metrics"][arm]["exact"] for row in rows], dtype=float)
        summary[arm] = {
            "satisfied_pairs_per_board": float(pairs.mean()),
            "adjacency_recall": float(pairs.mean() / PAIR_DENOMINATOR),
            "exact_tiles_per_board": float(exact.mean()),
            "source_wins_ties_losses_vs_recovered": (
                None
                if arm == "recovered_top5"
                else [
                    int(np.count_nonzero(pairs > recovered_pairs)),
                    int(np.count_nonzero(pairs == recovered_pairs)),
                    int(np.count_nonzero(pairs < recovered_pairs)),
                ]
            ),
        }
    summary["deltas"] = {
        "finetuned_minus_recovered_pairs": (
            summary["finetuned"]["satisfied_pairs_per_board"]
            - summary["recovered_top5"]["satisfied_pairs_per_board"]
        ),
        "finetuned_minus_raw_pairs": (
            summary["finetuned"]["satisfied_pairs_per_board"]
            - summary["raw_taska"]["satisfied_pairs_per_board"]
        ),
        "finetuned_minus_recovered_exact": (
            summary["finetuned"]["exact_tiles_per_board"]
            - summary["recovered_top5"]["exact_tiles_per_board"]
        ),
    }
    return summary


def _run_gate(
    names: Sequence[str],
    lookup: Mapping[str, Mapping[str, Any]],
    cache: CleanTileCache,
    *,
    matchers: Mapping[str, torch.nn.Module],
    recovered: torch.nn.Module,
    finetuned: torch.nn.Module,
    device: torch.device,
    output_dir: Path,
) -> tuple[dict[str, Any], bool]:
    target_free_arrays: dict[str, np.ndarray] = {}
    target_free_rows: list[dict[str, Any]] = []
    deferred: list[tuple[Mapping[str, Any], str, np.ndarray, dict[str, np.ndarray]]] = []
    started = perf_counter()
    for index, name in enumerate(names):
        record = lookup[name]
        dirty_case = _dirty_case(cache, record, name, 0)
        matched = match_taska_tiles(
            dirty_case.dirty_tiles,
            matchers,
            config=MATCHER_CONFIG,
            device=device,
        )
        edges = tuple(matched.candidate_edges)
        raw = solve_raw_tail_global(
            matched.cost_right,
            matched.cost_down,
            edges,
            border_unary=None,
            grid=GRID,
            config=SOLVER_CONFIG,
        )
        recovered_logits = score_current_focal_edges(
            recovered,
            dirty_case.dirty_tiles,
            matched.cost_right,
            matched.cost_down,
            edges,
            device=device,
        )
        finetuned_logits = score_current_focal_edges(
            finetuned,
            dirty_case.dirty_tiles,
            matched.cost_right,
            matched.cost_down,
            edges,
            device=device,
        )
        recovered_result = solve_current_focal_edges(
            matched.cost_right,
            matched.cost_down,
            edges,
            recovered_logits,
            grid=GRID,
            config=SOLVER_CONFIG,
        )
        finetuned_result = solve_current_focal_edges(
            matched.cost_right,
            matched.cost_down,
            edges,
            finetuned_logits,
            grid=GRID,
            config=SOLVER_CONFIG,
        )
        layouts = {
            "raw_taska": _strict_layout(raw.layout),
            "recovered_top5": _strict_layout(recovered_result.layout),
            "finetuned": _strict_layout(finetuned_result.layout),
        }
        prefix = f"case_{index:04d}"
        for arm, layout in layouts.items():
            target_free_arrays[f"{prefix}__{arm}_layout"] = layout
        target_free_arrays[f"{prefix}__recovered_logits"] = recovered_logits
        target_free_arrays[f"{prefix}__finetuned_logits"] = finetuned_logits
        target_free_rows.append(
            {
                "prefix": prefix,
                "source_filename": name,
                "draw_index": 0,
                "dirty_sha256": _dirty_sha256(dirty_case.dirty_tiles),
                "candidate_edge_count": len(edges),
            }
        )
        deferred.append((record, name, dirty_case.dirty_tiles, layouts))
        print(
            json.dumps(
                {
                    "event": "local_gate_target_free",
                    "case": index + 1,
                    "case_count": len(names),
                    "source": name,
                }
            ),
            flush=True,
        )
    frozen_npz = output_dir / "local-gate-target-free.npz"
    frozen_json = output_dir / "local-gate-target-free.json"
    _write_npz(frozen_npz, target_free_arrays)
    _write_json(
        frozen_json,
        {
            "schema": "aiijc-taska-focal-current-local-gate-target-free-v1",
            "contains_exact_references_or_labels": False,
            "strict_layout_count_per_arm": len(names),
            "rows": target_free_rows,
        },
    )
    scored_rows: list[dict[str, Any]] = []
    for record, name, dirty, layouts in deferred:
        reference = _reference(cache, record, name, 0, dirty)
        scored_rows.append(
            {
                "source_filename": name,
                "draw_index": 0,
                "metrics": _metric_row(layouts, reference),
            }
        )
    summary = _aggregate(scored_rows)
    passed = summary["deltas"]["finetuned_minus_recovered_pairs"] >= 0.0
    return {
        "source_count": len(names),
        "rows": scored_rows,
        "summary": summary,
        "gate_passed": passed,
        "runtime_seconds": perf_counter() - started,
        "target_free_artifacts": {
            "npz": {
                "path": str(frozen_npz.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(frozen_npz),
            },
            "metadata": {
                "path": str(frozen_json.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(frozen_json),
            },
        },
    }, passed


def _run_held(
    lookup: Mapping[str, Mapping[str, Any]],
    cache: CleanTileCache,
    *,
    recovered: torch.nn.Module,
    finetuned: torch.nn.Module,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    if (
        sha256_file(HELD_ARCHIVE) != HELD_ARCHIVE_SHA256
        or sha256_file(HELD_METADATA) != HELD_METADATA_SHA256
    ):
        raise ValueError("unchanged held32 parent artifacts changed")
    metadata = json.loads(HELD_METADATA.read_text(encoding="utf-8"))
    parent_rows = metadata["rows"]
    target_free_arrays: dict[str, np.ndarray] = {}
    target_free_rows: list[dict[str, Any]] = []
    deferred: list[tuple[Mapping[str, Any], str, int, np.ndarray, dict[str, np.ndarray]]] = []
    started = perf_counter()
    with np.load(HELD_ARCHIVE, allow_pickle=False) as archive:
        for index, row in enumerate(parent_rows):
            prefix = str(row["prefix"])
            name = str(row["source_filename"])
            draw = int(row["draw_index"])
            record = lookup[name]
            dirty_case = _dirty_case(cache, record, name, draw)
            if _dirty_sha256(dirty_case.dirty_tiles) != row["dirty_sha256"]:
                raise ValueError("held dirty bytes differ from frozen parent")
            right = np.asarray(archive[f"{prefix}__cost_right"], dtype=np.float32)
            down = np.asarray(archive[f"{prefix}__cost_down"], dtype=np.float32)
            edges = _edges_from_arrays(
                archive[f"{prefix}__edge_source"],
                archive[f"{prefix}__edge_target"],
                archive[f"{prefix}__edge_axis"],
            )
            raw_layout = _strict_layout(archive[f"{prefix}__taska_layout"])
            recovered_logits = score_current_focal_edges(
                recovered, dirty_case.dirty_tiles, right, down, edges, device=device
            )
            finetuned_logits = score_current_focal_edges(
                finetuned, dirty_case.dirty_tiles, right, down, edges, device=device
            )
            recovered_layout = _strict_layout(
                solve_current_focal_edges(
                    right,
                    down,
                    edges,
                    recovered_logits,
                    grid=GRID,
                    config=SOLVER_CONFIG,
                ).layout
            )
            finetuned_layout = _strict_layout(
                solve_current_focal_edges(
                    right,
                    down,
                    edges,
                    finetuned_logits,
                    grid=GRID,
                    config=SOLVER_CONFIG,
                ).layout
            )
            layouts = {
                "raw_taska": raw_layout,
                "recovered_top5": recovered_layout,
                "finetuned": finetuned_layout,
            }
            for arm, layout in layouts.items():
                target_free_arrays[f"{prefix}__{arm}_layout"] = layout
            target_free_arrays[f"{prefix}__recovered_logits"] = recovered_logits
            target_free_arrays[f"{prefix}__finetuned_logits"] = finetuned_logits
            target_free_rows.append(
                {
                    "prefix": prefix,
                    "source_filename": name,
                    "draw_index": draw,
                    "dirty_sha256": row["dirty_sha256"],
                    "candidate_edge_count": len(edges),
                }
            )
            deferred.append((record, name, draw, dirty_case.dirty_tiles, layouts))
            print(
                json.dumps(
                    {
                        "event": "held_target_free",
                        "case": index + 1,
                        "case_count": len(parent_rows),
                        "source": name,
                        "draw": draw,
                    }
                ),
                flush=True,
            )
    frozen_npz = output_dir / "held32-target-free.npz"
    frozen_json = output_dir / "held32-target-free.json"
    _write_npz(frozen_npz, target_free_arrays)
    _write_json(
        frozen_json,
        {
            "schema": "aiijc-taska-focal-current-held32-target-free-v1",
            "contains_exact_references_or_labels": False,
            "parent_archive_sha256": HELD_ARCHIVE_SHA256,
            "rows": target_free_rows,
        },
    )
    scored_rows: list[dict[str, Any]] = []
    for record, name, draw, dirty, layouts in deferred:
        reference = _reference(cache, record, name, draw, dirty)
        scored_rows.append(
            {
                "source_filename": name,
                "draw_index": draw,
                "metrics": _metric_row(layouts, reference),
            }
        )
    return {
        "case_count": len(scored_rows),
        "source_count": len({row["source_filename"] for row in scored_rows}),
        "rows": scored_rows,
        "summary": _aggregate(scored_rows),
        "runtime_seconds": perf_counter() - started,
        "target_free_artifacts": {
            "npz": {
                "path": str(frozen_npz.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(frozen_npz),
            },
            "metadata": {
                "path": str(frozen_json.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(frozen_json),
            },
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument(
        "--resume-after-training",
        action="store_true",
        help="reuse the already completed fixed train archive/checkpoint and start at gate",
    )
    args = parser.parse_args(argv)
    if args.device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable")
        torch.use_deterministic_algorithms(False)
    else:
        torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    output_dir = args.output_dir.resolve()
    if args.resume_after_training:
        if not output_dir.is_dir():
            raise ValueError("resume output directory does not exist")
        for name in (
            "local-gate-target-free.npz",
            "local-gate-target-free.json",
            "held32-target-free.npz",
            "held32-target-free.json",
            "report.json",
        ):
            if (output_dir / name).exists():
                raise ValueError("resume refuses partially or fully completed gate artifacts")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    config, train_names, gate_names = _load_config(args.config)
    lookup = _manifest_lookup(config)
    if not (set(train_names) | set(gate_names)) <= set(lookup):
        raise ValueError("fine-tune roster escaped organizer train manifest")
    cache = CleanTileCache(args.targets.resolve(), maximum_boards=2)
    checkpoint_path = PROJECT_ROOT / config["training"]["initial_checkpoint"]["path"]
    matcher_dir = PROJECT_ROOT / "artifacts/prior-taska/ckpt"
    training_archive = output_dir / "training-harvest.npz"
    fine_checkpoint = output_dir / "taska-focal-current-finetuned.pt"
    if args.resume_after_training:
        if not training_archive.is_file() or not fine_checkpoint.is_file():
            raise ValueError("completed train archive/checkpoint is absent")
        with np.load(training_archive, allow_pickle=False) as training:
            labels = np.asarray(training["labels"], dtype=np.uint8)
            sources = np.asarray(training["source_filenames"])
            if len(sources) != 96 or tuple(sources.tolist()) != train_names:
                raise ValueError("resumed train archive roster differs")
            training_summary = {
                "status": "resumed_after_completed_fixed_training",
                "source_count": len(sources),
                "edge_count": len(labels),
                "positive_count": int(labels.sum()),
                "positive_fraction": float(labels.mean()),
                "artifact": {
                    "path": str(training_archive.relative_to(PROJECT_ROOT)),
                    "sha256": sha256_file(training_archive),
                },
            }
        payload = torch.load(fine_checkpoint, map_location="cpu", weights_only=True)
        metadata = payload["metadata"]
        training_summary["history"] = metadata["history"]
        training_summary["frozen_prior"] = metadata["frozen_prior"]
        fine_sha = sha256_file(fine_checkpoint)
    else:
        matchers = load_default_taska_ensemble(matcher_dir, device=device)
        boards, training_summary = _build_training_archive(
            train_names,
            lookup,
            cache,
            matchers=matchers,
            device=device,
            output=training_archive,
        )
        del matchers
        if device.type == "mps":
            torch.mps.empty_cache()
        model, frozen_prior = prepare_finetune_model(checkpoint_path, device=device)
        training_started = perf_counter()
        history = train_fixed_focal_model(model, boards, device=device)
        training_summary["optimization_runtime_seconds"] = (
            perf_counter() - training_started
        )
        training_summary["history"] = history
        training_summary["frozen_prior"] = frozen_prior
        fine_sha = save_finetuned_checkpoint(
            fine_checkpoint,
            model,
            config_sha256=CONFIG_SHA256,
            train_source_digest=TRAIN_DIGEST,
            history=history,
            frozen_prior=frozen_prior,
        )
        del model, boards
    if device.type == "mps":
        torch.mps.empty_cache()
    finetuned = load_finetuned_checkpoint(
        fine_checkpoint,
        expected_sha256=fine_sha,
        expected_config_sha256=CONFIG_SHA256,
        device=device,
    )
    recovered = load_taska_focal_verifier(checkpoint_path, device=device)
    matchers = load_default_taska_ensemble(matcher_dir, device=device)
    gate, passed = _run_gate(
        gate_names,
        lookup,
        cache,
        matchers=matchers,
        recovered=recovered,
        finetuned=finetuned,
        device=device,
        output_dir=output_dir,
    )
    del matchers
    if device.type == "mps":
        torch.mps.empty_cache()
    held = (
        _run_held(
            lookup,
            cache,
            recovered=recovered,
            finetuned=finetuned,
            device=device,
            output_dir=output_dir,
        )
        if passed
        else {"status": "skipped_by_preregistered_nonnegative_local_gate"}
    )
    report = {
        "schema": REPORT_SCHEMA,
        "config": {
            "path": str(args.config.resolve().relative_to(PROJECT_ROOT)),
            "sha256": CONFIG_SHA256,
        },
        "device": str(device),
        "rosters": {
            "train": {"count": len(train_names), "digest": TRAIN_DIGEST},
            "local_gate": {"count": len(gate_names), "digest": GATE_DIGEST},
            "overlap": 0,
        },
        "training": training_summary,
        "checkpoint": {"path": str(fine_checkpoint.relative_to(PROJECT_ROOT)), "sha256": fine_sha},
        "local_gate": gate,
        "held32": held,
        "raw_solver_sha256": sha256_file(
            PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py"
        ),
        "legality": {
            "strict_original_tile_permutations": True,
            "candidate_membership_unchanged": True,
            "targets_or_exact_references_at_inference": False,
            "raw_score_prior_frozen": True,
            "competition_test_access": False,
        },
        "runtime_seconds": perf_counter() - started,
    }
    report_path = output_dir / "report.json"
    _write_json(report_path, report)
    print(
        json.dumps(
            {
                "event": "fine_tune_complete",
                "report": str(report_path),
                "local_gate": gate["summary"],
                "held32": held.get("summary"),
                "runtime_seconds": report["runtime_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

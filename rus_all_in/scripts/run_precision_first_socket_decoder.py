#!/usr/bin/env python3
"""Select, freeze, and confirm one precision-first Socket OT decoder.

``curve`` reads the already-open offset-2304 diagnostic only and reports edge
precision versus rank/confidence.  ``freeze`` applies one fixed JSON config to
a fresh panel using dirty inputs only.  ``evaluate`` verifies frozen hashes and
then opens targets exactly once for direct placement, adjacency, SSIM, and
component-purity diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from aiijc_puzzle.candidate_supply import recover_layout
from aiijc_puzzle.component_anchor_diagnostic import (
    ComponentTranslationDiagnostic,
    diagnose_component_translation,
    rebuild_decoder_components,
)
from aiijc_puzzle.layout_evaluation import RECOVERED_REFERENCE_CAVEAT, evaluate_layout
from aiijc_puzzle.precision_first_socket_decoder import (
    PrecisionFirstDecoderConfig,
    PrecisionFirstDecodeResult,
    decode_precision_first,
    precision_edge_evidence,
)
from aiijc_puzzle.protocol import (
    IMAGE_SIZE,
    assemble_tiles,
    compute_protocol_digest,
    contest_ssim,
    select_manifest_records,
    sha256_file,
    split_tiles,
)
from aiijc_puzzle.socket_decoder import SocketDecoderConfig, decode_socket_assignments
from aiijc_puzzle.socket_matcher import BORDER_HEAD_EMBEDDING_V2, SocketMatcher

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRID = 24
TILE_COUNT = GRID * GRID
NAMESPACE = "aiijc-socket-matcher-v1"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "interim" / "validation_manifest.json"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/socket-matcher/v2-border-train512-s300-r100-dev24/socket_matcher.pt"
)
DEFAULT_PRIMARY_FROZEN = (
    PROJECT_ROOT
    / "outputs/socket-matcher/component-anchor-diagnostic-offset2304-dev24"
    / "frozen_predictions.npz"
)
DEFAULT_CURVE_REPORT = (
    PROJECT_ROOT / "outputs/socket-matcher/precision-first-selection-curve-offset2304.json"
)
DEFAULT_CONFIG = PROJECT_ROOT / "configs/socket_precision_first_v1.json"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs/socket-matcher/precision-first-confirm-offset2816-dev24"
)
FROZEN_FILENAME = "frozen_predictions.npz"
FREEZE_METADATA_FILENAME = "freeze_metadata.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("curve", "freeze", "evaluate"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--inputs", type=Path, default=Path("data/raw/train/inputs"))
    parser.add_argument("--targets", type=Path, default=Path("data/raw/train/targets"))
    parser.add_argument("--prior-output-root", type=Path, default=Path("outputs/socket-matcher"))
    parser.add_argument("--primary-frozen", type=Path, default=DEFAULT_PRIMARY_FROZEN)
    parser.add_argument("--curve-report", type=Path, default=DEFAULT_CURVE_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--offset", type=int, default=2816)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    return parser.parse_args()


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        if image.mode != "RGB" or image.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"expected RGB 480x480 image: {path}")
        return np.asarray(image, dtype=np.uint8)


def _names_digest(names: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(names).encode()).hexdigest()


def _filename_lists(value: Any) -> Iterable[list[str]]:
    if isinstance(value, dict):
        for key, child in value.items():
            if key.endswith("filenames") and isinstance(child, list) and all(
                isinstance(item, str) for item in child
            ):
                yield child
            else:
                yield from _filename_lists(child)
    elif isinstance(value, list):
        for child in value:
            yield from _filename_lists(child)


def _known_exposed_names(root: Path, *, ignore: Path) -> tuple[set[str], int]:
    names: set[str] = set()
    reports = 0
    if root.exists():
        for path in sorted(root.glob("*/report.json")):
            if path.parent.resolve() == ignore.resolve():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            for values in _filename_lists(payload.get("selection", {})):
                names.update(values)
            reports += 1
    return names, reports


def _load_manifest_records(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], tuple[Any, ...]]:
    if args.offset < 2700 or args.offset in {3072}:
        raise ValueError("confirm offset must be >=2700 and must not be 3072")
    if args.offset in {2304, 2560}:
        raise ValueError("confirm offset reuses an explicitly forbidden diagnostic panel")
    if args.limit <= 0:
        raise ValueError("limit must be positive")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    panel = select_manifest_records(
        manifest,
        "train",
        limit=args.offset + args.limit,
        namespace=NAMESPACE,
    )
    records = tuple(panel[args.offset : args.offset + args.limit])
    if len(records) != args.limit:
        raise ValueError("confirmation panel is incomplete")
    return manifest, records


def _load_model(
    checkpoint: Path,
    *,
    device: torch.device,
) -> tuple[SocketMatcher, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    contract = payload.get("contract", {})
    if contract.get("architecture") != "board-conditioned-partial-socket-matcher-v2":
        raise ValueError("precision-first confirm requires a SocketMatcher v2 checkpoint")
    model = SocketMatcher(
        dimension=int(contract["dimension"]),
        heads=int(contract["heads"]),
        board_layers=int(contract["board_layers"]),
        socket_layers=int(contract["socket_layers"]),
        sinkhorn_iterations=int(contract["sinkhorn_iterations"]),
        border_head_version=str(
            contract.get("border_head_version", BORDER_HEAD_EMBEDDING_V2)
        ),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def _load_precision_config(path: Path) -> tuple[PrecisionFirstDecoderConfig, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("decoder") != "socket-precision-first-components-v1":
        raise ValueError("unexpected precision-first config contract")
    values = payload.get("config", {})
    config = PrecisionFirstDecoderConfig(**values)
    config.validate(tile_count=TILE_COUNT)
    return config, payload


def _positions(reference: np.ndarray) -> np.ndarray:
    position = np.empty(TILE_COUNT, dtype=np.int32)
    position[reference] = np.arange(TILE_COUNT, dtype=np.int32)
    return position


def _edge_correct(edge: Any, position: np.ndarray) -> bool:
    source_row, source_column = divmod(int(position[edge.source]), GRID)
    target_row, target_column = divmod(int(position[edge.target]), GRID)
    return bool(
        target_row - source_row == edge.delta_row
        and target_column - source_column == edge.delta_column
    )


def _curve_summary(rows: list[dict[str, Any]], predicate: Any) -> dict[str, float]:
    dirty = [row for row in rows if predicate(row)]
    trusted = [row for row in dirty if row["trusted"]]
    correct = sum(row["correct"] for row in trusted)
    return {
        "dirty_selected_edges_per_board": len(dirty) / 24,
        "trusted_selected_edges_per_board": len(trusted) / 24,
        "trusted_exact_edge_rate": correct / len(trusted) if trusted else math.nan,
        "trusted_correct_edges_per_board": correct / 24,
    }


def curve(args: argparse.Namespace) -> None:
    frozen = np.load(args.primary_frozen, allow_pickle=False)
    names = tuple(str(value) for value in frozen["filenames"].tolist())
    if len(names) != 24:
        raise ValueError("selection curve is fixed to the opened 24-board panel")
    permissive = PrecisionFirstDecoderConfig(
        minimum_edge_confidence=-1e6,
        minimum_real_row_margin=-1e6,
        minimum_real_column_margin=-1e6,
        minimum_dustbin_margin=-1e6,
    )
    rows: list[dict[str, Any]] = []
    for board_index, filename in enumerate(names):
        dirty = split_tiles(load_rgb(args.inputs / filename))
        clean = split_tiles(load_rgb(args.targets / filename))
        recovered = recover_layout(dirty, clean)
        position = _positions(recovered.dirty_at_position)
        trusted_position = recovered.margin_at_position >= np.median(
            recovered.margin_at_position
        )
        trusted_tile = np.zeros(TILE_COUNT, dtype=bool)
        trusted_tile[recovered.dirty_at_position] = trusted_position
        for axis, key in (
            ("right", "right_log_assignment"),
            ("down", "down_log_assignment"),
        ):
            matrix = frozen[key][board_index]
            matching = socket_decoder_matching(matrix, axis=axis)
            for rank, edge in enumerate(matching, start=1):
                evidence = precision_edge_evidence(
                    matrix,
                    edge,
                    grid=GRID,
                    config=permissive,
                )
                rows.append(
                    {
                        "board": board_index,
                        "axis": axis,
                        "rank_per_axis": rank,
                        "confidence": edge.confidence,
                        "real_row_margin": evidence.real_row_margin,
                        "real_column_margin": evidence.real_column_margin,
                        "dustbin_margin": evidence.dustbin_margin,
                        "trusted": bool(
                            trusted_tile[edge.source] and trusted_tile[edge.target]
                        ),
                        "correct": _edge_correct(edge, position),
                    }
                )
    rank_curve = {
        str(budget): _curve_summary(
            rows,
            lambda row, budget=budget: row["rank_per_axis"] <= budget,
        )
        for budget in (8, 16, 24, 32, 48, 64, 96, 144)
    }
    confidence_curve = {
        str(threshold): _curve_summary(
            rows,
            lambda row, threshold=threshold: row["confidence"] >= threshold,
        )
        for threshold in (-1.5, -1.25, -1.0, -0.75, -0.5, 0.0)
    }
    fixed = PrecisionFirstDecoderConfig()
    fixed_policy = _curve_summary(
        rows,
        lambda row: (
            row["confidence"] >= fixed.minimum_edge_confidence
            and row["real_row_margin"] >= fixed.minimum_real_row_margin
            and row["real_column_margin"] >= fixed.minimum_real_column_margin
            and row["dustbin_margin"] >= fixed.minimum_dustbin_margin
        ),
    )
    report = {
        "diagnostic": "socket-precision-selection-curve-v1",
        "selection": {
            "source": str(args.primary_frozen),
            "offset": 2304,
            "limit": 24,
            "filenames": list(names),
            "filenames_digest": _names_digest(names),
            "targets_previously_opened": True,
        },
        "reference_caveat": RECOVERED_REFERENCE_CAVEAT,
        "trusted_policy": "both endpoints in per-board top 50% recovered-position margin",
        "rank_budget_curve": rank_curve,
        "confidence_threshold_curve": confidence_curve,
        "fixed_policy_preview": {
            "config": as_config_dict(fixed),
            "edge_metrics": fixed_policy,
            "component_cap_selected_from_prior_diagnostic": 8,
            "rationale": (
                "confidence -1 retains about 18 trusted-correct edges/board at ~78% "
                "precision after mutual-real and dustbin margins; cap 8 exceeds the prior "
                "largest exactly rigid component size 6 but blocks giant weak bridges"
            ),
        },
    }
    args.curve_report.parent.mkdir(parents=True, exist_ok=True)
    args.curve_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["fixed_policy_preview"], indent=2), flush=True)


def socket_decoder_matching(matrix: Any, *, axis: str) -> tuple[Any, ...]:
    from aiijc_puzzle.socket_decoder import hard_partial_axis_matching

    return hard_partial_axis_matching(matrix, grid=GRID, axis=axis).edges


def as_config_dict(config: PrecisionFirstDecoderConfig) -> dict[str, Any]:
    return {
        "minimum_edge_confidence": config.minimum_edge_confidence,
        "minimum_real_row_margin": config.minimum_real_row_margin,
        "minimum_real_column_margin": config.minimum_real_column_margin,
        "minimum_dustbin_margin": config.minimum_dustbin_margin,
        "maximum_component_size": config.maximum_component_size,
        "border_weight": config.border_weight,
    }


@torch.no_grad()
def freeze(args: argparse.Namespace) -> None:
    manifest, records = _load_manifest_records(args)
    names = tuple(str(record["filename"]) for record in records)
    known_names, report_count = _known_exposed_names(
        args.prior_output_root,
        ignore=args.output_dir,
    )
    overlap = sorted(set(names) & known_names)
    if overlap:
        raise ValueError(f"confirmation panel overlaps prior reports: {overlap}")
    config, config_payload = _load_precision_config(args.config)
    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    model, checkpoint_payload = _load_model(args.checkpoint, device=device)
    checkpoint_names: set[str] = set()
    for values in _filename_lists(checkpoint_payload.get("selection", {})):
        checkpoint_names.update(values)
    checkpoint_overlap = sorted(set(names) & checkpoint_names)
    if checkpoint_overlap:
        raise ValueError(f"confirmation panel overlaps checkpoint lineage: {checkpoint_overlap}")

    right_assignments: list[np.ndarray] = []
    down_assignments: list[np.ndarray] = []
    default_layouts: list[np.ndarray] = []
    precision_layouts: list[np.ndarray] = []
    input_hashes: dict[str, str] = {}
    dirty_diagnostics: list[dict[str, Any]] = []
    default_config = SocketDecoderConfig(
        component_edge_budget_per_axis=144,
        swap_edge_budget_per_axis=144,
        max_swap_steps=24,
    )
    for index, filename in enumerate(names, start=1):
        input_path = args.inputs / filename
        dirty = split_tiles(load_rgb(input_path))
        input_hashes[filename] = sha256_file(input_path)
        tensor = torch.from_numpy(dirty.astype(np.float32)).permute(0, 3, 1, 2)
        output = model((tensor / 255.0).unsqueeze(0).to(device), grid=GRID)
        right = output.right_log_assignment[0].float().cpu().numpy()
        down = output.down_log_assignment[0].float().cpu().numpy()
        default = decode_socket_assignments(
            right,
            down,
            grid=GRID,
            config=default_config,
        )
        precision = decode_precision_first(right, down, grid=GRID, config=config)
        right_assignments.append(right.astype(np.float32))
        down_assignments.append(down.astype(np.float32))
        default_layouts.append(default.layout.astype(np.int16))
        precision_layouts.append(precision.layout.astype(np.int16))
        dirty_diagnostics.append(
            {
                "filename": filename,
                "default": default.diagnostics.as_dict(),
                "precision_first": precision.diagnostics.as_dict(),
            }
        )
        print(
            f"froze {index}/{len(names)} {filename} "
            f"selected={precision.diagnostics.attempted_constraints} "
            f"largest={precision.diagnostics.largest_component}",
            flush=True,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frozen_path = args.output_dir / FROZEN_FILENAME
    np.savez(
        frozen_path,
        filenames=np.asarray(names),
        right_log_assignment=np.stack(right_assignments),
        down_log_assignment=np.stack(down_assignments),
        default_decoder_layout=np.stack(default_layouts),
        precision_first_layout=np.stack(precision_layouts),
    )
    metadata = {
        "experiment": "socket-precision-first-confirm-v1",
        "phase": "dirty_predictions_frozen_before_target_access",
        "target_access_in_this_phase": False,
        "selection": {
            "split": "train",
            "namespace": NAMESPACE,
            "offset": args.offset,
            "limit": args.limit,
            "filenames": list(names),
            "filenames_digest": _names_digest(names),
            "source_disjoint_from_checkpoint": True,
            "source_disjoint_from_prior_reports": True,
            "prior_report_count_checked": report_count,
        },
        "protocol_digest": compute_protocol_digest(manifest),
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": sha256_file(args.checkpoint),
            "contract": checkpoint_payload["contract"],
        },
        "fixed_config": {
            "path": str(args.config),
            "sha256": sha256_file(args.config),
            "payload": config_payload,
        },
        "selection_curve": {
            "path": str(args.curve_report),
            "sha256": sha256_file(args.curve_report),
        },
        "input_sha256": input_hashes,
        "frozen_predictions": {
            "path": FROZEN_FILENAME,
            "sha256": sha256_file(frozen_path),
        },
        "dirty_diagnostics": dirty_diagnostics,
    }
    (args.output_dir / FREEZE_METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {frozen_path}", flush=True)


def _mean(values: Iterable[float]) -> float:
    array = np.asarray(tuple(values), dtype=np.float64)
    return float(np.mean(array)) if len(array) else math.nan


def _component_purity(
    components: Iterable[dict[int, tuple[int, int]]],
    reference: np.ndarray,
    trusted_tile: np.ndarray,
    layout: np.ndarray,
) -> dict[str, Any]:
    all_rows: list[ComponentTranslationDiagnostic] = []
    trusted_rows: list[ComponentTranslationDiagnostic] = []
    for component_id, component in enumerate(components):
        if len(component) < 2:
            continue
        all_row = diagnose_component_translation(
            component,
            reference,
            {"layout": layout},
            grid=GRID,
            component_id=component_id,
        )
        all_rows.append(all_row)
        evidence = np.asarray(
            [tile for tile in component if trusted_tile[tile]],
            dtype=np.int64,
        )
        if len(evidence) >= 2:
            trusted_rows.append(
                diagnose_component_translation(
                    component,
                    reference,
                    {"layout": layout},
                    grid=GRID,
                    component_id=component_id,
                    evidence_tiles=evidence,
                )
            )
    largest = max(trusted_rows, key=lambda row: row.size, default=None)
    full_exact = [row for row in all_rows if row.internally_exact]
    return {
        "nontrivial_component_count": len(all_rows),
        "largest_component_size": max((row.size for row in all_rows), default=1),
        "largest_component_trusted_purity": (
            largest.translation_purity if largest is not None else math.nan
        ),
        "trusted_translation_purity_weighted": (
            sum(row.true_shift_support for row in trusted_rows)
            / sum(row.evidence_size for row in trusted_rows)
            if trusted_rows
            else math.nan
        ),
        "all_reference_exact_component_count": len(full_exact),
        "all_reference_exact_component_tile_count": sum(row.size for row in full_exact),
        "largest_all_reference_exact_component": max(
            (row.size for row in full_exact), default=1
        ),
    }


def _selected_edge_precision(
    result: PrecisionFirstDecodeResult,
    reference: np.ndarray,
    trusted_tile: np.ndarray,
) -> dict[str, Any]:
    position = _positions(reference)
    all_edges = [item.edge for item in result.selected_edges]
    trusted = [
        edge
        for edge in all_edges
        if trusted_tile[edge.source] and trusted_tile[edge.target]
    ]
    correct = sum(_edge_correct(edge, position) for edge in trusted)
    return {
        "selected_edges": len(all_edges),
        "trusted_selected_edges": len(trusted),
        "trusted_exact_edges": correct,
        "trusted_exact_edge_rate": correct / len(trusted) if trusted else math.nan,
    }


def evaluate(args: argparse.Namespace) -> None:
    metadata_path = args.output_dir / FREEZE_METADATA_FILENAME
    frozen_path = args.output_dir / FROZEN_FILENAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("phase") != "dirty_predictions_frozen_before_target_access":
        raise ValueError("missing dirty-only freeze declaration")
    for path, expected in (
        (frozen_path, metadata["frozen_predictions"]["sha256"]),
        (args.checkpoint, metadata["checkpoint"]["sha256"]),
        (args.config, metadata["fixed_config"]["sha256"]),
        (args.curve_report, metadata["selection_curve"]["sha256"]),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"frozen dependency changed: {path}")
    manifest, records = _load_manifest_records(args)
    names = tuple(str(record["filename"]) for record in records)
    if list(names) != metadata["selection"]["filenames"]:
        raise ValueError("evaluation panel differs from frozen panel")
    config, _ = _load_precision_config(args.config)
    frozen = np.load(frozen_path, allow_pickle=False)
    if tuple(frozen["filenames"].tolist()) != names:
        raise ValueError("frozen filenames differ from metadata")

    boards: list[dict[str, Any]] = []
    for index, filename in enumerate(names):
        input_path = args.inputs / filename
        if sha256_file(input_path) != metadata["input_sha256"][filename]:
            raise ValueError(f"dirty input changed after freeze: {filename}")
        dirty_image = load_rgb(input_path)
        dirty = split_tiles(dirty_image)
        clean_image = load_rgb(args.targets / filename)
        clean = split_tiles(clean_image)
        recovered = recover_layout(dirty, clean)
        reference = recovered.dirty_at_position
        trusted_position = recovered.margin_at_position >= np.median(
            recovered.margin_at_position
        )
        trusted_tile = np.zeros(TILE_COUNT, dtype=bool)
        trusted_tile[reference] = trusted_position
        right = frozen["right_log_assignment"][index]
        down = frozen["down_log_assignment"][index]
        default_layout = frozen["default_decoder_layout"][index].astype(np.int64)
        precision_layout = frozen["precision_first_layout"][index].astype(np.int64)
        precision = decode_precision_first(right, down, grid=GRID, config=config)
        if not np.array_equal(precision.layout, precision_layout):
            raise RuntimeError("precision-first decode differs from frozen dirty-only layout")
        default_build = rebuild_decoder_components(
            right,
            down,
            grid=GRID,
            edge_budget_per_axis=144,
        )
        variants: dict[str, Any] = {}
        for name, layout, components in (
            ("default_decoder144", default_layout, default_build.components),
            ("precision_first", precision_layout, precision.components),
        ):
            geometry = evaluate_layout(layout, reference).as_dict()
            prediction = assemble_tiles(dirty[layout])
            variants[name] = {
                "geometry": geometry,
                "raw_ssim": contest_ssim(clean_image, prediction),
                "component_purity": _component_purity(
                    components,
                    reference,
                    trusted_tile,
                    layout,
                ),
            }
        boards.append(
            {
                "filename": filename,
                "primary_direct_placement": {
                    name: variants[name]["geometry"]["direct_placement"]
                    for name in variants
                },
                "variants": variants,
                "precision_selected_edges": _selected_edge_precision(
                    precision,
                    reference,
                    trusted_tile,
                ),
                "precision_dirty_diagnostics": precision.diagnostics.as_dict(),
            }
        )
        print(
            f"evaluated {index + 1}/{len(names)} {filename} "
            f"direct={variants['precision_first']['geometry']['correct_tile_count']} "
            f"adj={variants['precision_first']['geometry']['adjacency']:.4f}",
            flush=True,
        )

    metrics = (
        "correct_tile_count",
        "direct_placement",
        "translation_aligned_count",
        "translation_aligned_placement",
        "adjacency",
    )
    variant_mean: dict[str, Any] = {}
    for name in ("default_decoder144", "precision_first"):
        variant_mean[name] = {
            "geometry": {
                metric: _mean(board["variants"][name]["geometry"][metric] for board in boards)
                for metric in metrics
            },
            "raw_ssim": _mean(board["variants"][name]["raw_ssim"] for board in boards),
            "component_purity": {
                key: _mean(
                    board["variants"][name]["component_purity"][key]
                    for board in boards
                )
                for key in (
                    "nontrivial_component_count",
                    "largest_component_size",
                    "largest_component_trusted_purity",
                    "trusted_translation_purity_weighted",
                    "all_reference_exact_component_count",
                    "all_reference_exact_component_tile_count",
                    "largest_all_reference_exact_component",
                )
            },
        }
    default = variant_mean["default_decoder144"]
    candidate = variant_mean["precision_first"]
    delta = {
        "geometry": {
            key: candidate["geometry"][key] - default["geometry"][key]
            for key in metrics
        },
        "raw_ssim": candidate["raw_ssim"] - default["raw_ssim"],
        "component_purity": {
            key: candidate["component_purity"][key] - default["component_purity"][key]
            for key in candidate["component_purity"]
        },
    }
    report = {
        "experiment": "socket-precision-first-confirm-v1",
        "status": "single frozen source-disjoint confirmation; no sweep",
        "selection": metadata["selection"],
        "protocol_digest": compute_protocol_digest(manifest),
        "checkpoint": metadata["checkpoint"],
        "fixed_config": metadata["fixed_config"],
        "selection_curve": metadata["selection_curve"],
        "frozen_predictions": metadata["frozen_predictions"],
        "reference_caveat": RECOVERED_REFERENCE_CAVEAT,
        "primary_metric": "direct_placement against target-assisted recovered permutation",
        "aggregate": {
            "primary_direct_placement": {
                name: variant_mean[name]["geometry"]["direct_placement"]
                for name in variant_mean
            },
            "variant_mean": variant_mean,
            "delta_precision_first_minus_default": delta,
            "selected_edge_evidence": {
                "mean_selected_edges": _mean(
                    board["precision_selected_edges"]["selected_edges"] for board in boards
                ),
                "mean_trusted_selected_edges": _mean(
                    board["precision_selected_edges"]["trusted_selected_edges"]
                    for board in boards
                ),
                "trusted_exact_edge_rate": (
                    sum(
                        board["precision_selected_edges"]["trusted_exact_edges"]
                        for board in boards
                    )
                    / sum(
                        board["precision_selected_edges"]["trusted_selected_edges"]
                        for board in boards
                    )
                ),
            },
        },
        "boards": boards,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2), flush=True)
    print(f"wrote {report_path}", flush=True)


def main() -> None:
    args = parse_args()
    if args.phase == "curve":
        curve(args)
    elif args.phase == "freeze":
        freeze(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()

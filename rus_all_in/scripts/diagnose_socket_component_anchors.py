#!/usr/bin/env python3
"""Freeze and diagnose SocketGlue component placement on a fresh train panel.

Run ``freeze`` first.  That phase reads dirty inputs only and writes immutable
socket assignments plus two target-blind decoder layouts.  Run ``evaluate`` in
a separate process to open clean targets and quantify internal component
geometry versus absolute translation error.
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
    constraint_is_reference_correct,
    diagnose_component_translation,
    rebuild_decoder_components,
)
from aiijc_puzzle.layout_evaluation import RECOVERED_REFERENCE_CAVEAT, evaluate_layout
from aiijc_puzzle.protocol import (
    IMAGE_SIZE,
    compute_protocol_digest,
    select_manifest_records,
    sha256_file,
    split_tiles,
)
from aiijc_puzzle.socket_decoder import (
    SocketDecoderConfig,
    decode_socket_assignments,
    texture_centrality_unary,
)
from aiijc_puzzle.socket_matcher import SocketMatcher

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "interim" / "validation_manifest.json"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "socket-matcher"
    / "v2-border-train512-s300-r100-dev24"
    / "socket_matcher.pt"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "socket-matcher"
    / "component-anchor-diagnostic-offset2304-dev24"
)
NAMESPACE = "aiijc-socket-matcher-v1"
GRID = 24
TILE_COUNT = GRID * GRID
FROZEN_FILENAME = "frozen_predictions.npz"
FREEZE_METADATA_FILENAME = "freeze_metadata.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "evaluate"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--inputs", type=Path, default=Path("data/raw/train/inputs"))
    parser.add_argument("--targets", type=Path, default=Path("data/raw/train/targets"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prior-output-root", type=Path, default=Path("outputs/socket-matcher"))
    parser.add_argument("--offset", type=int, default=2304)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--component-budget", type=int, default=144)
    parser.add_argument("--texture-prior-weight", type=float, default=0.05)
    parser.add_argument("--centre-radius", type=float, default=4.0)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    return parser.parse_args()


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        if image.mode != "RGB" or image.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"expected RGB 480x480 image: {path}")
        return np.asarray(image, dtype=np.uint8)


def _names_digest(names: Iterable[str]) -> str:
    value = "\n".join(names).encode()
    return hashlib.sha256(value).hexdigest()


def _select_records(args: argparse.Namespace) -> tuple[dict[str, Any], tuple[Any, ...]]:
    if args.offset < 2300:
        raise ValueError("this fresh diagnostic is preregistered for an offset >= 2300")
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
        raise ValueError("fresh evaluation panel is incomplete")
    return manifest, records


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


def _load_model(
    checkpoint: Path,
    *,
    device: torch.device,
) -> tuple[SocketMatcher, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    contract = payload.get("contract", {})
    if contract.get("architecture") != "board-conditioned-partial-socket-matcher-v2":
        raise ValueError("diagnostic requires a SocketMatcher v2 checkpoint")
    model = SocketMatcher(
        dimension=int(contract["dimension"]),
        heads=int(contract["heads"]),
        board_layers=int(contract["board_layers"]),
        socket_layers=int(contract["socket_layers"]),
        sinkhorn_iterations=int(contract["sinkhorn_iterations"]),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def _decoder_config(args: argparse.Namespace, *, texture: bool) -> SocketDecoderConfig:
    return SocketDecoderConfig(
        component_edge_budget_per_axis=args.component_budget,
        swap_edge_budget_per_axis=args.component_budget,
        max_swap_steps=24,
        component_shift_unary_weight=args.texture_prior_weight if texture else 0.0,
    )


@torch.no_grad()
def freeze(args: argparse.Namespace) -> None:
    manifest, records = _select_records(args)
    names = tuple(str(record["filename"]) for record in records)
    known_names, report_count = _known_exposed_names(
        args.prior_output_root,
        ignore=args.output_dir,
    )
    selected = set(names)
    overlap = sorted(selected & known_names)
    if overlap:
        raise ValueError(f"fresh panel overlaps prior reports: {overlap}")

    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    model, checkpoint_payload = _load_model(args.checkpoint, device=device)
    checkpoint_names: set[str] = set()
    for values in _filename_lists(checkpoint_payload.get("selection", {})):
        checkpoint_names.update(values)
    checkpoint_overlap = sorted(selected & checkpoint_names)
    if checkpoint_overlap:
        raise ValueError(f"fresh panel overlaps checkpoint lineage: {checkpoint_overlap}")

    right_assignments: list[np.ndarray] = []
    down_assignments: list[np.ndarray] = []
    border_layouts: list[np.ndarray] = []
    texture_layouts: list[np.ndarray] = []
    component_sizes: list[list[int]] = []
    input_hashes: dict[str, str] = {}
    for index, filename in enumerate(names, start=1):
        input_path = args.inputs / filename
        dirty = split_tiles(load_rgb(input_path))
        input_hashes[filename] = sha256_file(input_path)
        tensor = torch.from_numpy(dirty.astype(np.float32)).permute(0, 3, 1, 2)
        output = model((tensor / 255.0).unsqueeze(0).to(device), grid=GRID)
        right = output.right_log_assignment[0].float().cpu().numpy()
        down = output.down_log_assignment[0].float().cpu().numpy()
        border = decode_socket_assignments(
            right,
            down,
            grid=GRID,
            config=_decoder_config(args, texture=False),
        )
        texture = decode_socket_assignments(
            right,
            down,
            grid=GRID,
            config=_decoder_config(args, texture=True),
            component_shift_unary=texture_centrality_unary(dirty, grid=GRID),
        )
        rebuilt = rebuild_decoder_components(
            right,
            down,
            grid=GRID,
            edge_budget_per_axis=args.component_budget,
        )
        sizes = sorted((len(component) for component in rebuilt.components), reverse=True)
        if tuple(sizes) != border.diagnostics.component_sizes:
            raise RuntimeError("diagnostic component rebuild diverged from decoder")
        right_assignments.append(right.astype(np.float32))
        down_assignments.append(down.astype(np.float32))
        border_layouts.append(border.layout.astype(np.int16))
        texture_layouts.append(texture.layout.astype(np.int16))
        component_sizes.append(sizes)
        print(
            f"froze {index}/{len(names)} {filename} "
            f"largest={sizes[0]} components={len(sizes)}",
            flush=True,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frozen_path = args.output_dir / FROZEN_FILENAME
    np.savez(
        frozen_path,
        filenames=np.asarray(names),
        right_log_assignment=np.stack(right_assignments),
        down_log_assignment=np.stack(down_assignments),
        decoder_border_layout=np.stack(border_layouts),
        decoder_texture_centre_layout=np.stack(texture_layouts),
    )
    metadata = {
        "diagnostic": "socket-component-absolute-anchor-v1",
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
        "input_sha256": input_hashes,
        "frozen_predictions": {
            "path": FROZEN_FILENAME,
            "sha256": sha256_file(frozen_path),
        },
        "preregistered_analysis": {
            "component_edge_budget_per_axis": args.component_budget,
            "decoder_border_weight": 0.2,
            "decoder_qap_swaps": 24,
            "texture_prior_weight": args.texture_prior_weight,
            "trusted_reference_policy": "per-board top 50% recovered-position margin",
            "component_internal_metric": (
                "mode of target coordinate minus predicted relative coordinate over trusted tiles"
            ),
            "near_centre_radius_cells": args.centre_radius,
            "near_anchor_threshold": "Manhattan translation error <= 2 cells",
            "component_sizes": component_sizes,
        },
    }
    (args.output_dir / FREEZE_METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {frozen_path}", flush=True)


def _safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else math.nan


def _mean(values: Iterable[float]) -> float:
    array = np.asarray(tuple(values), dtype=np.float64)
    return float(np.mean(array)) if len(array) else math.nan


def _median(values: Iterable[float]) -> float:
    array = np.asarray(tuple(values), dtype=np.float64)
    return float(np.median(array)) if len(array) else math.nan


def _anchor_summary(
    rows: list[tuple[str, ComponentTranslationDiagnostic]],
) -> dict[str, Any]:
    if not rows:
        return {"component_count": 0, "boards_covered": 0}
    diagnostics = [row for _, row in rows]
    result: dict[str, Any] = {
        "component_count": len(rows),
        "boards_covered": len({filename for filename, _ in rows}),
        "tile_count": int(sum(row.size for row in diagnostics)),
        "trusted_evidence_tile_count": int(sum(row.evidence_size for row in diagnostics)),
        "mean_size": _mean(row.size for row in diagnostics),
        "mean_trusted_evidence_size": _mean(row.evidence_size for row in diagnostics),
        "mean_translation_purity": _mean(row.translation_purity for row in diagnostics),
        "mean_pairwise_relative_accuracy": _mean(
            row.pairwise_relative_accuracy for row in diagnostics
        ),
        "internally_exact_rate": _mean(float(row.internally_exact) for row in diagnostics),
        "true_centroid_near_board_centre_rate": _mean(
            float(row.true_centroid_near_board_centre) for row in diagnostics
        ),
        "mean_true_centroid_distance_cells": _mean(
            row.true_centroid_distance_from_board_centre for row in diagnostics
        ),
        "mean_texture_strength": _mean(row.texture_strength for row in diagnostics),
    }
    anchor_names = tuple(diagnostics[0].anchors)
    result["anchors"] = {}
    for name in anchor_names:
        observations = [row.anchors[name] for row in diagnostics]
        result["anchors"][name] = {
            "exact_translation_rate": _mean(float(item.exact) for item in observations),
            "within_two_cells_rate": _mean(
                float(item.within_two_cells) for item in observations
            ),
            "mean_l1_error_cells": _mean(item.l1_error for item in observations),
            "median_l1_error_cells": _median(item.l1_error for item in observations),
            "mean_euclidean_error_cells": _mean(
                item.euclidean_error for item in observations
            ),
            "mean_component_rigidity_after_qap": _mean(item.rigidity for item in observations),
        }
    border_errors = [row.anchors["decoder_border"].l1_error for row in diagnostics]
    texture_errors = [
        row.anchors["decoder_texture_centre"].l1_error for row in diagnostics
    ]
    centre_errors = [row.anchors["geometric_centre"].l1_error for row in diagnostics]
    result["comparisons"] = {
        "texture_better_than_border_rate": _mean(
            float(texture < border)
            for texture, border in zip(texture_errors, border_errors, strict=True)
        ),
        "texture_equal_to_border_rate": _mean(
            float(texture == border)
            for texture, border in zip(texture_errors, border_errors, strict=True)
        ),
        "geometric_centre_better_than_border_rate": _mean(
            float(centre < border)
            for centre, border in zip(centre_errors, border_errors, strict=True)
        ),
    }
    return result


def _size_histogram(rows: list[ComponentTranslationDiagnostic]) -> dict[str, int]:
    bins = {"2": 0, "3-4": 0, "5-8": 0, "9-16": 0, "17-32": 0, "33+": 0}
    for row in rows:
        if row.size == 2:
            bins["2"] += 1
        elif row.size <= 4:
            bins["3-4"] += 1
        elif row.size <= 8:
            bins["5-8"] += 1
        elif row.size <= 16:
            bins["9-16"] += 1
        elif row.size <= 32:
            bins["17-32"] += 1
        else:
            bins["33+"] += 1
    return bins


def evaluate(args: argparse.Namespace) -> None:
    metadata_path = args.output_dir / FREEZE_METADATA_FILENAME
    frozen_path = args.output_dir / FROZEN_FILENAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("phase") != "dirty_predictions_frozen_before_target_access":
        raise ValueError("freeze metadata does not declare a clean dirty-only phase")
    if sha256_file(frozen_path) != metadata["frozen_predictions"]["sha256"]:
        raise ValueError("frozen predictions changed after the dirty-only phase")
    if sha256_file(args.checkpoint) != metadata["checkpoint"]["sha256"]:
        raise ValueError("checkpoint changed after the dirty-only phase")
    manifest, records = _select_records(args)
    names = tuple(str(record["filename"]) for record in records)
    if list(names) != metadata["selection"]["filenames"]:
        raise ValueError("evaluation selection differs from frozen selection")

    frozen = np.load(frozen_path, allow_pickle=False)
    if tuple(frozen["filenames"].tolist()) != names:
        raise ValueError("frozen prediction filenames differ from metadata")

    boards: list[dict[str, Any]] = []
    aggregate_rows: list[tuple[str, ComponentTranslationDiagnostic]] = []
    aggregate_all_reference_rows: list[tuple[str, ComponentTranslationDiagnostic]] = []
    for index, filename in enumerate(names):
        input_path = args.inputs / filename
        if sha256_file(input_path) != metadata["input_sha256"][filename]:
            raise ValueError(f"dirty input changed after freezing: {filename}")
        dirty = split_tiles(load_rgb(input_path))
        right = frozen["right_log_assignment"][index]
        down = frozen["down_log_assignment"][index]
        border_layout = frozen["decoder_border_layout"][index].astype(np.int64)
        texture_layout = frozen["decoder_texture_centre_layout"][index].astype(np.int64)
        build = rebuild_decoder_components(
            right,
            down,
            grid=GRID,
            edge_budget_per_axis=args.component_budget,
        )

        # Target access begins only here, after all model outputs and decoder
        # variants were persisted and hash-locked by the separate freeze run.
        clean = split_tiles(load_rgb(args.targets / filename))
        recovered = recover_layout(dirty, clean)
        reference = recovered.dirty_at_position
        trusted_position = recovered.margin_at_position >= np.median(
            recovered.margin_at_position
        )
        trusted_tile = np.zeros(TILE_COUNT, dtype=bool)
        trusted_tile[reference] = trusted_position
        trusted_tiles = np.flatnonzero(trusted_tile)
        texture_unary = texture_centrality_unary(dirty, grid=GRID)
        component_rows: list[ComponentTranslationDiagnostic] = []
        all_reference_by_component: dict[int, ComponentTranslationDiagnostic] = {}
        for component_id, component in enumerate(build.components):
            evidence = np.asarray(
                [tile for tile in component if trusted_tile[tile]],
                dtype=np.int64,
            )
            all_reference = diagnose_component_translation(
                component,
                reference,
                {
                    "decoder_border": border_layout,
                    "decoder_texture_centre": texture_layout,
                },
                grid=GRID,
                component_id=component_id,
                texture_unary=texture_unary,
                centre_radius=args.centre_radius,
            )
            all_reference_by_component[component_id] = all_reference
            if all_reference.size >= 2:
                aggregate_all_reference_rows.append((filename, all_reference))
            if not len(evidence):
                continue
            row = diagnose_component_translation(
                component,
                reference,
                {
                    "decoder_border": border_layout,
                    "decoder_texture_centre": texture_layout,
                },
                grid=GRID,
                component_id=component_id,
                texture_unary=texture_unary,
                evidence_tiles=evidence,
                centre_radius=args.centre_radius,
            )
            component_rows.append(row)
            if row.size >= 2 and row.evidence_size >= 2:
                aggregate_rows.append((filename, row))

        nontrivial = [row for row in component_rows if row.size >= 2 and row.evidence_size >= 2]
        exact = [row for row in nontrivial if row.internally_exact]
        largest = max(nontrivial, key=lambda row: (row.size, -row.component_id), default=None)
        largest_exact = max(exact, key=lambda row: (row.size, -row.component_id), default=None)
        all_constraints = list(build.constraints)
        accepted_constraints = [
            item for item in all_constraints if item.status in {"added", "consistent"}
        ]
        added_constraints = [item for item in all_constraints if item.status == "added"]
        trusted_constraints = [
            item
            for item in all_constraints
            if trusted_tile[item.edge.source] and trusted_tile[item.edge.target]
        ]
        trusted_added_constraints = [
            item for item in trusted_constraints if item.status == "added"
        ]
        correct_all = sum(
            constraint_is_reference_correct(item, reference, grid=GRID)
            for item in all_constraints
        )
        correct_accepted = sum(
            constraint_is_reference_correct(item, reference, grid=GRID)
            for item in accepted_constraints
        )
        correct_added = sum(
            constraint_is_reference_correct(item, reference, grid=GRID)
            for item in added_constraints
        )
        correct_trusted = sum(
            constraint_is_reference_correct(item, reference, grid=GRID)
            for item in trusted_constraints
        )
        correct_trusted_added = sum(
            constraint_is_reference_correct(item, reference, grid=GRID)
            for item in trusted_added_constraints
        )
        all_reference_nontrivial = [
            row for row in all_reference_by_component.values() if row.size >= 2
        ]
        all_reference_exact = [
            row for row in all_reference_nontrivial if row.internally_exact
        ]
        board = {
            "filename": filename,
            "reference": {
                "kind": "target-assisted recovered permutation",
                "trusted_tile_count": int(len(trusted_tiles)),
            },
            "components": {
                "count": len(build.components),
                "nontrivial_with_two_trusted_tiles": len(nontrivial),
                "size_histogram": _size_histogram(nontrivial),
                "largest": largest.as_dict() if largest is not None else None,
                "largest_internally_exact": (
                    largest_exact.as_dict() if largest_exact is not None else None
                ),
                "trusted_internally_consistent_count": len(exact),
                "trusted_internally_consistent_component_tile_count": int(
                    sum(row.size for row in exact)
                ),
                "trusted_translation_purity_weighted": _safe_rate(
                    sum(row.true_shift_support for row in nontrivial),
                    sum(row.evidence_size for row in nontrivial),
                ),
                "all_recovered_reference_exact_count": len(all_reference_exact),
                "all_recovered_reference_exact_tile_count": int(
                    sum(row.size for row in all_reference_exact)
                ),
                "all_recovered_reference_translation_purity_weighted": _safe_rate(
                    sum(row.true_shift_support for row in all_reference_nontrivial),
                    sum(row.evidence_size for row in all_reference_nontrivial),
                ),
                "details": [
                    row.as_dict()
                    | {
                        "all_recovered_reference_internal_geometry": {
                            "translation_purity": all_reference_by_component[
                                row.component_id
                            ].translation_purity,
                            "pairwise_relative_accuracy": all_reference_by_component[
                                row.component_id
                            ].pairwise_relative_accuracy,
                            "internally_exact": all_reference_by_component[
                                row.component_id
                            ].internally_exact,
                        }
                    }
                    for row in nontrivial
                ],
            },
            "constraints": {
                "attempted": len(all_constraints),
                "attempted_exact_edge_rate": _safe_rate(correct_all, len(all_constraints)),
                "accepted": len(accepted_constraints),
                "accepted_exact_edge_rate": _safe_rate(
                    correct_accepted, len(accepted_constraints)
                ),
                "added": len(added_constraints),
                "added_exact_edge_rate": _safe_rate(correct_added, len(added_constraints)),
                "trusted_attempted": len(trusted_constraints),
                "trusted_attempted_exact_edge_rate": _safe_rate(
                    correct_trusted, len(trusted_constraints)
                ),
                "trusted_added": len(trusted_added_constraints),
                "trusted_added_exact_edge_rate": _safe_rate(
                    correct_trusted_added, len(trusted_added_constraints)
                ),
                "status_counts": build.status_counts,
            },
            "layouts": {
                "decoder_border": evaluate_layout(border_layout, reference).as_dict(),
                "decoder_texture_centre": evaluate_layout(texture_layout, reference).as_dict(),
            },
        }
        boards.append(board)
        print(
            f"evaluated {index + 1}/{len(names)} {filename} "
            f"largest={largest.size if largest else 0} "
            f"purity={largest.translation_purity if largest else math.nan:.3f}",
            flush=True,
        )

    largest_rows = [
        (board["filename"], max(
            (
                row
                for filename, row in aggregate_rows
                if filename == board["filename"]
            ),
            key=lambda row: (row.size, -row.component_id),
        ))
        for board in boards
        if any(filename == board["filename"] for filename, _ in aggregate_rows)
    ]
    exact_rows = [(filename, row) for filename, row in aggregate_rows if row.internally_exact]
    largest_exact_rows: list[tuple[str, ComponentTranslationDiagnostic]] = []
    for board in boards:
        candidates = [
            row
            for filename, row in exact_rows
            if filename == board["filename"]
        ]
        if candidates:
            largest_exact_rows.append(
                (
                    board["filename"],
                    max(candidates, key=lambda row: (row.size, -row.component_id)),
                )
            )
    high_purity_rows = [
        (filename, row)
        for filename, row in aggregate_rows
        if row.translation_purity >= 0.8
    ]
    component_count = sum(len(board["components"]["details"]) for board in boards)
    exact_tile_count = sum(
        board["components"]["trusted_internally_consistent_component_tile_count"]
        for board in boards
    )
    all_reference_exact_rows = [
        (filename, row)
        for filename, row in aggregate_all_reference_rows
        if row.internally_exact
    ]
    all_reference_exact_tile_count = sum(row.size for _, row in all_reference_exact_rows)
    trusted_support = sum(row.true_shift_support for _, row in aggregate_rows)
    trusted_evidence = sum(row.evidence_size for _, row in aggregate_rows)
    all_reference_support = sum(
        row.true_shift_support for _, row in aggregate_all_reference_rows
    )
    all_reference_evidence = sum(
        row.evidence_size for _, row in aggregate_all_reference_rows
    )
    report = {
        "diagnostic": "socket-component-absolute-anchor-v1",
        "phase": "post-freeze target-assisted diagnostic",
        "selection": metadata["selection"],
        "protocol_digest": compute_protocol_digest(manifest),
        "checkpoint": metadata["checkpoint"],
        "frozen_predictions": metadata["frozen_predictions"],
        "reference_caveat": RECOVERED_REFERENCE_CAVEAT,
        "evidence_policy": (
            "component truth uses only the per-board top 50% recovered-position margins; "
            "component size and predicted anchors remain target-blind"
        ),
        "aggregate": {
            "board_count": len(boards),
            "nontrivial_component_count": component_count,
            "mean_decoder_component_count": _mean(
                board["components"]["count"] for board in boards
            ),
            "mean_largest_component_size": _mean(
                row.size for _, row in largest_rows
            ),
            "mean_largest_component_trusted_purity": _mean(
                row.translation_purity for _, row in largest_rows
            ),
            "boards_with_trusted_internally_consistent_largest_component_rate": _mean(
                float(row.internally_exact) for _, row in largest_rows
            ),
            "trusted_translation_purity_weighted": _safe_rate(
                trusted_support, trusted_evidence
            ),
            "all_recovered_reference_translation_purity_weighted": _safe_rate(
                all_reference_support, all_reference_evidence
            ),
            "trusted_internally_consistent_component_tile_count": exact_tile_count,
            "trusted_internally_consistent_component_tile_rate": exact_tile_count
            / (len(boards) * TILE_COUNT),
            "all_recovered_reference_exact_component_tile_count": (
                all_reference_exact_tile_count
            ),
            "all_recovered_reference_exact_component_tile_rate": (
                all_reference_exact_tile_count / (len(boards) * TILE_COUNT)
            ),
            "constraint_attempted_exact_edge_rate": _mean(
                board["constraints"]["attempted_exact_edge_rate"] for board in boards
            ),
            "constraint_added_exact_edge_rate": _mean(
                board["constraints"]["added_exact_edge_rate"] for board in boards
            ),
            "trusted_constraint_attempted_exact_edge_rate": _mean(
                board["constraints"]["trusted_attempted_exact_edge_rate"]
                for board in boards
            ),
            "trusted_constraint_added_exact_edge_rate": _mean(
                board["constraints"]["trusted_added_exact_edge_rate"]
                for board in boards
            ),
            "decoder_layout_mean": {
                name: {
                    metric: _mean(
                        board["layouts"][name][metric] for board in boards
                    )
                    for metric in (
                        "correct_tile_count",
                        "direct_placement",
                        "translation_aligned_count",
                        "translation_aligned_placement",
                        "adjacency",
                    )
                }
                for name in ("decoder_border", "decoder_texture_centre")
            },
            "anchor_slices": {
                "largest_high_confidence_component_per_board": _anchor_summary(largest_rows),
                "largest_trusted_internally_consistent_component_per_board": _anchor_summary(
                    largest_exact_rows
                ),
                "all_trusted_internally_consistent_nontrivial_components": _anchor_summary(
                    exact_rows
                ),
                "all_at_least_80pct_pure_components": _anchor_summary(high_purity_rows),
            },
        },
        "boards": boards,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {report_path}", flush=True)


def main() -> None:
    args = parse_args()
    if not 1 <= args.component_budget <= TILE_COUNT - GRID:
        raise ValueError("component-budget is outside the exact partial-matching range")
    if not np.isfinite(args.texture_prior_weight) or args.texture_prior_weight < 0:
        raise ValueError("texture-prior-weight must be finite and non-negative")
    if not np.isfinite(args.centre_radius) or args.centre_radius < 0:
        raise ValueError("centre-radius must be finite and non-negative")
    if args.phase == "freeze":
        freeze(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Freeze/evaluate one fixed SocketMatcher + edge-ranker rank fusion.

This is deliberately an evaluation-only runner.  It loads the retained d32
SocketMatcher-v2 and k5 raw edge-ranker checkpoints, checks their declared
contracts and source lineages, and evaluates exactly three decoder144 arms on
a fixed fresh manifest-train panel: each model alone and an equal-rank fusion.
All dirty-only layouts are committed before any target image is opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from PIL import Image

from aiijc_puzzle.candidate_supply import recover_layout
from aiijc_puzzle.edge_ranker import PairwiseEdgeRanker, build_inference_board, score_board
from aiijc_puzzle.layout_evaluation import evaluate_layout
from aiijc_puzzle.legacy_upgrade import validate_layout
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
from aiijc_puzzle.socket_matcher import SocketMatcher
from aiijc_puzzle.socket_ranker_fusion import (
    analytic_border_logits,
    calibrated_partial_assignment,
    equal_rank_fusion,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "data" / "interim" / "validation_manifest.json"
INPUTS = PROJECT_ROOT / "data" / "raw" / "train" / "inputs"
TARGETS = PROJECT_ROOT / "data" / "raw" / "train" / "targets"
SOCKET_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "socket-matcher"
    / "v2-border-train512-s300-r100-dev24"
    / "socket_matcher.pt"
)
RANKER_CHECKPOINT = (
    PROJECT_ROOT / "outputs" / "edge-ranker" / "scale-raw-train64-cal12" / "edge_ranker.pt"
)
SOCKET_SHA256 = "7ccb14042e50432bf450018d4ebb32b78866d3755d8387cb1534f67155fd1c19"
RANKER_SHA256 = "d18ff864c63170d5fcdb868d672a60515d10ac600afa2ed0424000921ecbb21a"
SELECTION_NAMESPACE = "aiijc-socket-matcher-v1"
# Ranks 2573 and 2583 were excluded before target access: the former belongs
# to an available k16 ranker training lineage, while the latter appeared in a
# pre-existing experiment report.  The remaining fixed 24 are globally fresh
# in the repository's documented experiment ledger at preregistration time.
PANEL_RANKS = tuple(rank for rank in range(2560, 2586) if rank not in {2573, 2583})
GRID = 24
COUNT = GRID * GRID
VARIANTS = ("socket_decoder144", "ranker_k5_decoder144", "equal_rank_fusion_decoder144")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/socket-ranker-fusion/fresh-train24-ranks2560-2585-k5"),
    )
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        if image.format != "PNG" or image.mode != "RGB" or image.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"expected strict RGB 480x480 PNG: {path}")
        return np.asarray(image, dtype=np.uint8)


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def names_digest(names: list[str]) -> str:
    return hashlib.sha256("\n".join(names).encode()).hexdigest()


def _filenames_in_payload(payload: Any) -> set[str]:
    names: set[str] = set()

    def visit(value: Any, key: str = "") -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                if child_key == "filename" and isinstance(child, str):
                    names.add(child)
                elif "filename" in child_key and isinstance(child, list):
                    names.update(item for item in child if isinstance(item, str))
                visit(child, child_key)
        elif isinstance(value, list):
            for child in value:
                visit(child, key)

    visit(payload)
    return names


def preexisting_report_filenames(output_dir: Path) -> set[str]:
    names: set[str] = set()
    resolved_output = output_dir.resolve()
    for path in (PROJECT_ROOT / "outputs").glob("**/report.json"):
        resolved = path.resolve()
        if resolved == resolved_output or resolved_output in resolved.parents:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        names.update(_filenames_in_payload(payload))
    return names


def select_fresh_panel(
    manifest: dict[str, Any],
    *,
    output_dir: Path,
    socket_payload: dict[str, Any],
    ranker_payload: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    ranked = select_manifest_records(
        manifest,
        "train",
        limit=max(PANEL_RANKS) + 1,
        namespace=SELECTION_NAMESPACE,
    )
    selected = tuple(ranked[index] for index in PANEL_RANKS)
    names = {str(record["filename"]) for record in selected}
    socket_selection = socket_payload.get("selection", {})
    socket_lineage = set(
        socket_selection.get(
            "lineage_train_filenames", socket_selection.get("train_filenames", [])
        )
    )
    ranker_lineage = set(ranker_payload.get("contract", {}).get("train_filenames", []))
    documented = preexisting_report_filenames(output_dir)
    overlaps = {
        "socket_checkpoint_lineage": sorted(names & socket_lineage),
        "ranker_checkpoint_lineage": sorted(names & ranker_lineage),
        "preexisting_reports": sorted(names & documented),
    }
    if any(overlaps.values()):
        raise ValueError(f"fixed evaluation panel is no longer fresh: {overlaps}")
    return selected, {
        "namespace": SELECTION_NAMESPACE,
        "rank_indices": list(PANEL_RANKS),
        "filenames": [str(record["filename"]) for record in selected],
        "filenames_digest": names_digest([str(record["filename"]) for record in selected]),
        "freshness_overlaps": overlaps,
    }


def load_models(
    device: torch.device,
) -> tuple[SocketMatcher, PairwiseEdgeRanker, dict[str, Any], dict[str, Any]]:
    if sha256_file(SOCKET_CHECKPOINT) != SOCKET_SHA256:
        raise ValueError("SocketMatcher checkpoint digest changed")
    if sha256_file(RANKER_CHECKPOINT) != RANKER_SHA256:
        raise ValueError("edge-ranker checkpoint digest changed")
    socket_payload = torch.load(SOCKET_CHECKPOINT, map_location="cpu", weights_only=False)
    socket_contract = socket_payload.get("contract", {})
    expected_socket = {
        "architecture": "board-conditioned-partial-socket-matcher-v2",
        "dimension": 32,
        "heads": 4,
        "board_layers": 1,
        "socket_layers": 1,
        "sinkhorn_iterations": 10,
    }
    if any(socket_contract.get(key) != value for key, value in expected_socket.items()):
        raise ValueError("SocketMatcher checkpoint contract is not the retained d32 v2 model")
    socket = SocketMatcher(
        dimension=32,
        heads=4,
        board_layers=1,
        socket_layers=1,
        sinkhorn_iterations=10,
    ).to(device)
    socket.load_state_dict(socket_payload["state_dict"])
    socket.eval()

    ranker_payload = torch.load(RANKER_CHECKPOINT, map_location="cpu", weights_only=False)
    ranker_contract = ranker_payload.get("contract", {})
    expected_ranker = {
        "architecture": "joint-seam-context-cnn-v1",
        "candidate_k": 5,
        "view_mode": "raw",
        "feature_dim": 12,
        "width": 24,
        "hidden": 48,
    }
    if any(ranker_contract.get(key) != value for key, value in expected_ranker.items()):
        raise ValueError("edge-ranker checkpoint is not the retained raw-k5 auxiliary")
    semantic_paths = {
        "edge_ranker": PROJECT_ROOT / "src" / "aiijc_puzzle" / "edge_ranker.py",
        "candidate_supply": PROJECT_ROOT / "src" / "aiijc_puzzle" / "candidate_supply.py",
        "legacy_upgrade": PROJECT_ROOT / "src" / "aiijc_puzzle" / "legacy_upgrade.py",
        "protocol": PROJECT_ROOT / "src" / "aiijc_puzzle" / "protocol.py",
    }
    declared_hashes = ranker_contract.get("semantic_code_sha256", {})
    mismatches = {
        name: (declared_hashes.get(name), sha256_file(path))
        for name, path in semantic_paths.items()
        if declared_hashes.get(name) != sha256_file(path)
    }
    if mismatches:
        raise ValueError(f"edge-ranker semantic source differs from its contract: {mismatches}")
    ranker = PairwiseEdgeRanker(
        feature_dim=12,
        view_mode="raw",
        width=24,
        hidden=48,
    ).to(device)
    ranker.load_state_dict(ranker_payload["state_dict"])
    ranker.eval()
    return socket, ranker, socket_payload, ranker_payload


def _decode(right: np.ndarray, down: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    decoded = decode_socket_assignments(
        right,
        down,
        grid=GRID,
        config=SocketDecoderConfig(
            component_edge_budget_per_axis=144,
            swap_edge_budget_per_axis=144,
            max_swap_steps=24,
        ),
    )
    return validate_layout(decoded.layout), decoded.report()


@torch.no_grad()
def freeze_dirty_predictions(
    socket: SocketMatcher,
    ranker: PairwiseEdgeRanker,
    records: tuple[Any, ...],
    *,
    device: torch.device,
    output_dir: Path,
    selection: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], float]:
    started = perf_counter()
    frozen: list[dict[str, Any]] = []
    commitment_boards: list[dict[str, Any]] = []
    for board_index, record in enumerate(records, start=1):
        filename = str(record["filename"])
        dirty = split_tiles(load_rgb(INPUTS / filename))
        tensor = torch.from_numpy(dirty.astype(np.float32)).permute(0, 3, 1, 2)
        socket_output = socket((tensor / 255.0).unsqueeze(0).to(device), grid=GRID)
        socket_right_raw = socket_output.right_raw[0].float().cpu().numpy()
        socket_down_raw = socket_output.down_raw[0].float().cpu().numpy()
        socket_right_assignment = (
            socket_output.right_log_assignment[0].float().cpu().numpy()
        )
        socket_down_assignment = socket_output.down_log_assignment[0].float().cpu().numpy()

        ranker_board = build_inference_board(
            dirty,
            filename=filename,
            views=tuple(ranker_payload_contract_views(ranker)),
            candidate_k=5,
        )
        ranker_right, ranker_down, ranker_diagnostics = score_board(
            ranker,
            ranker_board,
            device=device,
            pair_batch=2048,
        )
        ranker_right_out, ranker_left_in = analytic_border_logits(ranker_right)
        ranker_bottom_out, ranker_top_in = analytic_border_logits(ranker_down)
        ranker_right_assignment = calibrated_partial_assignment(
            ranker_right,
            grid=GRID,
            outgoing_border=ranker_right_out,
            incoming_border=ranker_left_in,
            iterations=10,
        )
        ranker_down_assignment = calibrated_partial_assignment(
            ranker_down,
            grid=GRID,
            outgoing_border=ranker_bottom_out,
            incoming_border=ranker_top_in,
            iterations=10,
        )
        fused_right_scores, fused_right_assignment = equal_rank_fusion(
            socket_right_raw,
            ranker_right,
            first_outgoing_border=socket_output.right_out_border_logits[0].float().cpu().numpy(),
            first_incoming_border=socket_output.left_in_border_logits[0].float().cpu().numpy(),
            second_outgoing_border=ranker_right_out,
            second_incoming_border=ranker_left_in,
            grid=GRID,
            iterations=10,
        )
        fused_down_scores, fused_down_assignment = equal_rank_fusion(
            socket_down_raw,
            ranker_down,
            first_outgoing_border=socket_output.bottom_out_border_logits[0].float().cpu().numpy(),
            first_incoming_border=socket_output.top_in_border_logits[0].float().cpu().numpy(),
            second_outgoing_border=ranker_bottom_out,
            second_incoming_border=ranker_top_in,
            grid=GRID,
            iterations=10,
        )
        assignments = {
            "socket_decoder144": (socket_right_assignment, socket_down_assignment),
            "ranker_k5_decoder144": (ranker_right_assignment, ranker_down_assignment),
            "equal_rank_fusion_decoder144": (
                fused_right_assignment,
                fused_down_assignment,
            ),
        }
        layouts: dict[str, np.ndarray] = {}
        decoder_reports: dict[str, Any] = {}
        commitment_variants: dict[str, Any] = {}
        for name in VARIANTS:
            layout, decoder_report = _decode(*assignments[name])
            prediction = assemble_tiles(dirty[layout])
            layouts[name] = layout
            decoder_reports[name] = decoder_report
            commitment_variants[name] = {
                "layout_tile_at_position": layout.tolist(),
                "layout_sha256": array_sha256(layout.astype(np.int32)),
                "assembled_dirty_sha256": array_sha256(prediction),
            }
        frozen.append(
            {
                "filename": filename,
                "dirty": dirty,
                "layouts": layouts,
                "decoder_reports": decoder_reports,
                "retrieval_scores": {
                    "socket": (
                        socket_right_assignment[:COUNT, :COUNT],
                        socket_down_assignment[:COUNT, :COUNT],
                    ),
                    "ranker_k5": (
                        ranker_right_assignment[:COUNT, :COUNT],
                        ranker_down_assignment[:COUNT, :COUNT],
                    ),
                    "equal_rank_fusion": (
                        fused_right_assignment[:COUNT, :COUNT],
                        fused_down_assignment[:COUNT, :COUNT],
                    ),
                },
                "ranker_diagnostics": ranker_diagnostics,
                "fused_score_sha256": {
                    "right": array_sha256(fused_right_scores),
                    "down": array_sha256(fused_down_scores),
                },
            }
        )
        commitment_boards.append(
            {"filename": filename, "variants": commitment_variants}
        )
        print(f"froze dirty board {board_index}/{len(records)} {filename}", flush=True)

    commitment = {
        "experiment": "socket-ranker-equal-rank-fusion-v1",
        "targets_opened_during_freeze": False,
        "selection": selection,
        "checkpoints": {
            "socket": {"path": str(SOCKET_CHECKPOINT), "sha256": SOCKET_SHA256},
            "ranker": {"path": str(RANKER_CHECKPOINT), "sha256": RANKER_SHA256},
        },
        "fixed_roster": list(VARIANTS),
        "fusion": {
            "real_edge_weight_socket": 0.5,
            "real_edge_weight_ranker": 0.5,
            "border_weight_socket": 0.5,
            "border_weight_ranker": 0.5,
            "calibration": "per-anchor inverse-normal rank; no target calibration",
            "partial_ot_unmatched_per_side": GRID,
        },
        "decoder": {
            "component_edge_budget_per_axis": 144,
            "swap_edge_budget_per_axis": 144,
            "max_swap_steps": 24,
        },
        "boards": commitment_boards,
    }
    commitment_path = output_dir / "prediction-commitment.json"
    atomic_json(commitment_path, commitment)
    return frozen, commitment, perf_counter() - started


def ranker_payload_contract_views(model: PairwiseEdgeRanker) -> tuple[str, ...]:
    # The retained checkpoint contract fixes this exact dirty-only candidate
    # supply.  Keeping it explicit prevents accidental use of target metadata.
    if model.view_mode != "raw":
        raise ValueError("this evaluation is fixed to the retained raw ranker")
    return ("raw", "tile_z", "bilateral", "gray")


def directional_retrieval(
    right: np.ndarray,
    down: np.ndarray,
    reference: np.ndarray,
    *,
    prefix: str,
) -> dict[str, float]:
    position_of_tile = np.empty(COUNT, dtype=np.int64)
    position_of_tile[reference] = np.arange(COUNT)
    metrics: dict[str, float] = {}
    for axis, scores, delta in (("right", right, 1), ("down", down, GRID)):
        position = np.arange(COUNT)
        legal = position % GRID != GRID - 1 if axis == "right" else position < COUNT - GRID
        position = position[legal]
        anchors = reference[position]
        truth = reference[position + delta]
        order = np.argsort(-np.asarray(scores)[anchors], axis=1, kind="stable")
        for k in (1, 5, 16, 32):
            metrics[f"{prefix}_{axis}_r{k}"] = float(
                np.mean(np.any(order[:, :k] == truth[:, None], axis=1))
            )
    for k in (1, 5, 16, 32):
        metrics[f"{prefix}_pooled_r{k}"] = 0.5 * (
            metrics[f"{prefix}_right_r{k}"] + metrics[f"{prefix}_down_r{k}"]
        )
    # This check is intentionally target-assisted and runs only after the
    # prediction commitment has been persisted.
    if not np.array_equal(np.sort(reference), np.arange(COUNT)):
        raise ValueError("recovered reference is not a strict permutation")
    if not np.array_equal(position_of_tile[reference], np.arange(COUNT)):
        raise RuntimeError("reference inverse-permutation invariant failed")
    return metrics


def mean_numeric(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = [
        key
        for key, value in rows[0].items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    return {key: float(np.mean([float(row[key]) for row in rows])) for key in keys}


def evaluate_after_commitment(frozen: list[dict[str, Any]]) -> tuple[dict[str, Any], float]:
    started = perf_counter()
    boards: list[dict[str, Any]] = []
    for board_index, item in enumerate(frozen, start=1):
        clean_image = load_rgb(TARGETS / item["filename"])
        recovered = recover_layout(item["dirty"], split_tiles(clean_image))
        local: dict[str, float] = {}
        for name, (right, down) in item["retrieval_scores"].items():
            local.update(
                directional_retrieval(
                    right,
                    down,
                    recovered.dirty_at_position,
                    prefix=name,
                )
            )
        variants: dict[str, Any] = {}
        for name, layout in item["layouts"].items():
            geometry = evaluate_layout(
                layout,
                recovered.dirty_at_position,
                reference_is_exact=False,
            ).as_dict()
            prediction = assemble_tiles(item["dirty"][layout])
            variants[name] = geometry | {"raw_ssim": contest_ssim(clean_image, prediction)}
        boards.append(
            {
                "filename": item["filename"],
                "local": local,
                "variants": variants,
                "decoder_reports": item["decoder_reports"],
                "ranker_diagnostics": item["ranker_diagnostics"],
            }
        )
        print(f"evaluated target {board_index}/{len(frozen)} {item['filename']}", flush=True)

    local_mean = mean_numeric([board["local"] for board in boards])
    variants_mean = {
        name: mean_numeric([board["variants"][name] for board in boards])
        for name in VARIANTS
    }
    fusion = variants_mean["equal_rank_fusion_decoder144"]
    deltas = {
        control: {
            key: fusion[key] - metrics[key]
            for key in (
                "correct_tile_count",
                "direct_placement",
                "translation_aligned_placement",
                "adjacency",
                "raw_ssim",
            )
        }
        for control, metrics in variants_mean.items()
        if control != "equal_rank_fusion_decoder144"
    }
    return {
        "reference": "target-assisted recovered permutation; organizer labels unavailable",
        "primary_metric": "direct_placement",
        "boards": boards,
        "local_mean": local_mean,
        "variants_mean": variants_mean,
        "fusion_deltas_vs_controls": deltas,
    }, perf_counter() - started


def main() -> None:
    args = parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    device = torch.device(args.device)
    torch.manual_seed(20260830)
    np.random.seed(20260830)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    socket, ranker, socket_payload, ranker_payload = load_models(device)
    if ranker_payload["contract"].get("protocol_digest") != compute_protocol_digest(manifest):
        raise ValueError("ranker checkpoint and current manifest protocol differ")
    records, selection = select_fresh_panel(
        manifest,
        output_dir=output_dir,
        socket_payload=socket_payload,
        ranker_payload=ranker_payload,
    )
    frozen, _, freeze_seconds = freeze_dirty_predictions(
        socket,
        ranker,
        records,
        device=device,
        output_dir=output_dir,
        selection=selection,
    )
    commitment_path = output_dir / "prediction-commitment.json"
    commitment_sha256 = sha256_file(commitment_path)
    evaluation, evaluation_seconds = evaluate_after_commitment(frozen)
    report = {
        "experiment": "socket-ranker-equal-rank-fusion-v1",
        "status": "diagnostic-not-production",
        "hypothesis": (
            "the retained pairwise ranker contributes true-neighbour ordering not already "
            "captured by board-conditioned SocketMatcher, improving direct coordinates after "
            "one fixed decoder"
        ),
        "protocol": {
            "manifest_digest": compute_protocol_digest(manifest),
            "split": "train",
            "calibration_opened": False,
            "holdout_opened": False,
            "competition_test_opened": False,
            "all_dirty_only_layouts_committed_before_target_access": True,
            "prediction_commitment": str(commitment_path),
            "prediction_commitment_sha256": commitment_sha256,
        },
        "selection": selection,
        "models": {
            "socket": {
                "checkpoint": str(SOCKET_CHECKPOINT),
                "sha256": SOCKET_SHA256,
                "contract": socket_payload["contract"],
            },
            "ranker": {
                "checkpoint": str(RANKER_CHECKPOINT),
                "sha256": RANKER_SHA256,
                "contract": ranker_payload["contract"],
                "selection_reason": (
                    "retained k5 raw checkpoint: unlike k16 it passed its original local gate "
                    "and remained documented as a reusable layout auxiliary"
                ),
            },
        },
        "fixed_roster": list(VARIANTS),
        "fusion": {
            "calibration": "per-anchor inverse-normal ranks",
            "weights": {"socket": 0.5, "ranker": 0.5},
            "border_weights": {"socket": 0.5, "ranker": 0.5},
            "partial_ot_unmatched_per_axis_side": GRID,
            "target_tuned": False,
        },
        "decoder": {
            "name": "socket-translation-components-qap-v1",
            "component_edge_budget_per_axis": 144,
            "swap_edge_budget_per_axis": 144,
            "max_swap_steps": 24,
        },
        "runtime_seconds": {
            "dirty_only_freeze": freeze_seconds,
            "target_assisted_evaluation": evaluation_seconds,
        },
        "evaluation": evaluation,
    }
    report_path = output_dir / "report.json"
    atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "event": "complete",
                "report": str(report_path),
                "variants": evaluation["variants_mean"],
                "local": evaluation["local_mean"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

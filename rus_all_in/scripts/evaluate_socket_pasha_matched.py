#!/usr/bin/env python3
"""Matched frozen-v2 Socket diagnostic on Pasha883's four exposed boards.

The script discovers filenames only from Pasha's already-frozen dirty-only
score artifacts.  It freezes Socket raw/partial-OT scores and the single
preregistered 50/50 row-rank-percentile fusion before opening the recovered
permutation cache, Pasha's scored report, or clean targets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from PIL import Image

from aiijc_puzzle.layout_evaluation import evaluate_layout
from aiijc_puzzle.legacy_upgrade import layout_digest, solve_buddies
from aiijc_puzzle.pasha883_pairwise import pasha883_directional_retrieval_metrics
from aiijc_puzzle.protocol import IMAGE_SIZE, assemble_tiles, contest_ssim, sha256_file, split_tiles
from aiijc_puzzle.socket_pasha_matched import (
    fuse_pasha_socket_ot_rank_percentiles,
    mean_numeric_metrics,
    validate_directional_scores,
)
from aiijc_puzzle.socket_sorter_production import (
    choose_deterministic_device,
    load_socket_checkpoint,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/socket-matcher/v2-d64-train1024-s1600-r400-dev32/socket_matcher.pt"
)
DEFAULT_PASHA_DIR = PROJECT_ROOT / "outputs/pasha883-pairwise-audit/last4-full576"
DEFAULT_INPUTS = PROJECT_ROOT / "data/raw/train/inputs"
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
DEFAULT_PERMUTATIONS = PROJECT_ROOT / "artifacts/prior-pasha883/perms.npz"
FROZEN_CHECKPOINT_SHA256 = "0e9df49a503c65aac7f1468e9acd6a074a5e658ae8b61f8954be086272c49670"
INDICES = (6996, 6997, 6998, 6999)
GRID = 24
TILE_COUNT = GRID * GRID
KS = (1, 5, 25)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--pasha-dir", type=Path, default=DEFAULT_PASHA_DIR)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--permutations", type=Path, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    return parser.parse_args()


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        if image.mode != "RGB" or image.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"expected RGB 480x480 image: {path}")
        return np.asarray(image, dtype=np.uint8)


def _load_pasha_dirty_score(path: Path, index: int) -> dict[str, Any]:
    expected_keys = {"index", "filename", "right", "down", "inference_seconds"}
    with np.load(path) as archive:
        if set(archive.files) != expected_keys:
            raise ValueError(f"Pasha score artifact is not proven dirty-only: {path}")
        observed_index = int(archive["index"])
        filename = str(archive["filename"])
        right, down = validate_directional_scores(archive["right"], archive["down"])
        inference_seconds = float(archive["inference_seconds"])
    if observed_index != index:
        raise ValueError(f"Pasha score index mismatch: {path}")
    return {
        "index": index,
        "filename": filename,
        "right": right,
        "down": down,
        "inference_seconds": inference_seconds,
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
    }


@torch.inference_mode()
def _socket_scores(
    image: np.ndarray,
    *,
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    tiles = split_tiles(image)
    tensor = torch.from_numpy(tiles.astype(np.float32)).permute(0, 3, 1, 2).div_(255.0)
    started = perf_counter()
    output = model(tensor.unsqueeze(0).to(device), grid=GRID)
    elapsed = perf_counter() - started
    raw = validate_directional_scores(
        output.right_raw[0].float().cpu().numpy(),
        output.down_raw[0].float().cpu().numpy(),
    )
    normaliser = np.log(float(TILE_COUNT + GRID))
    partial_ot = validate_directional_scores(
        output.right_log_assignment[0, :TILE_COUNT, :TILE_COUNT].float().cpu().numpy()
        + normaliser,
        output.down_log_assignment[0, :TILE_COUNT, :TILE_COUNT].float().cpu().numpy()
        + normaliser,
    )
    return raw[0], raw[1], partial_ot[0], partial_ot[1], elapsed


def _freeze_one(
    pasha: dict[str, Any],
    *,
    inputs: Path,
    output_dir: Path,
    model: torch.nn.Module,
    device: torch.device,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    index = int(pasha["index"])
    filename = str(pasha["filename"])
    input_path = inputs / filename
    input_sha256 = sha256_file(input_path)
    output_path = output_dir / f"scores-{index:04d}.npz"
    if output_path.exists():
        with np.load(output_path) as archive:
            if (
                int(archive["index"]) != index
                or str(archive["filename"]) != filename
                or str(archive["input_sha256"]) != input_sha256
                or str(archive["checkpoint_sha256"]) != checkpoint_sha256
                or str(archive["pasha_score_sha256"]) != pasha["sha256"]
            ):
                raise ValueError(f"frozen Socket score identity mismatch: {output_path}")
            for right_key, down_key in (
                ("socket_raw_right", "socket_raw_down"),
                ("socket_ot_right", "socket_ot_down"),
                ("fusion_right", "fusion_down"),
            ):
                validate_directional_scores(archive[right_key], archive[down_key])
            layout = archive["fusion_buddies96_layout"].astype(np.int32)
            if layout.shape != (TILE_COUNT,) or not np.array_equal(
                np.sort(layout), np.arange(TILE_COUNT)
            ):
                raise ValueError(f"invalid frozen fusion layout: {output_path}")
            elapsed = float(archive["socket_inference_seconds"])
        print(f"reused {output_path.name} {filename}", flush=True)
        return {
            "index": index,
            "filename": filename,
            "path": str(output_path.resolve()),
            "sha256": sha256_file(output_path),
            "input_sha256": input_sha256,
            "pasha_score_sha256": pasha["sha256"],
            "socket_inference_seconds": elapsed,
            "reused": True,
        }

    raw_right, raw_down, ot_right, ot_down, elapsed = _socket_scores(
        _load_rgb(input_path),
        model=model,
        device=device,
    )
    fusion_right, fusion_down = fuse_pasha_socket_ot_rank_percentiles(
        pasha["right"],
        pasha["down"],
        ot_right,
        ot_down,
    )
    solve = solve_buddies(fusion_right, fusion_down, max_edges=96)
    np.savez_compressed(
        output_path,
        index=np.asarray(index),
        filename=np.asarray(filename),
        input_sha256=np.asarray(input_sha256),
        checkpoint_sha256=np.asarray(checkpoint_sha256),
        pasha_score_sha256=np.asarray(pasha["sha256"]),
        socket_raw_right=raw_right,
        socket_raw_down=raw_down,
        socket_ot_right=ot_right,
        socket_ot_down=ot_down,
        fusion_right=fusion_right,
        fusion_down=fusion_down,
        fusion_buddies96_layout=solve.layout,
        fusion_buddies96_objective=np.asarray(solve.objective),
        socket_inference_seconds=np.asarray(elapsed),
    )
    print(f"froze {output_path.name} {filename} in {elapsed:.2f}s", flush=True)
    return {
        "index": index,
        "filename": filename,
        "path": str(output_path.resolve()),
        "sha256": sha256_file(output_path),
        "input_sha256": input_sha256,
        "pasha_score_sha256": pasha["sha256"],
        "socket_inference_seconds": elapsed,
        "reused": False,
    }


def _local_deltas(
    left: dict[str, float],
    right: dict[str, float],
) -> dict[str, float]:
    return {
        f"{side}_r{k}": left[f"{side}_r{k}"] - right[f"{side}_r{k}"]
        for side in ("right", "down", "pooled")
        for k in KS
    }


def _global_mean(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = (
        "direct_placement",
        "row_accuracy",
        "column_accuracy",
        "translation_aligned_placement",
        "right_adjacency",
        "down_adjacency",
        "adjacency",
        "raw_ssim",
    )
    return {key: float(np.mean([row["fusion_buddies96"][key] for row in rows])) for key in keys}


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint.resolve()
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if checkpoint_sha256 != FROZEN_CHECKPOINT_SHA256:
        raise ValueError("this diagnostic is frozen to the reviewed v2 d64 checkpoint SHA-256")
    device = choose_deterministic_device(args.device)
    checkpoint = load_socket_checkpoint(checkpoint_path, device=device)
    if checkpoint.contract.get("architecture") != "board-conditioned-partial-socket-matcher-v2":
        raise ValueError("this diagnostic requires SocketMatcher v2")

    # Only the five-field dirty-only Pasha artifacts may reveal the matched filenames here.
    pasha_dirty = [
        _load_pasha_dirty_score(args.pasha_dir / f"scores-{index:04d}.npz", index)
        for index in INDICES
    ]
    filenames = [str(item["filename"]) for item in pasha_dirty]
    if len(set(filenames)) != len(filenames):
        raise ValueError("Pasha dirty-only artifacts contain duplicate filenames")
    if any(not (args.inputs / filename).is_file() for filename in filenames):
        raise FileNotFoundError("one or more matched dirty inputs are missing")

    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    selection = checkpoint_payload.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("checkpoint selection metadata is missing")
    socket_train_names = set(selection.get("lineage_train_filenames", ()))
    socket_exposed_names = set(selection.get("lineage_exposed_filenames", ()))
    if len(socket_train_names) != checkpoint.lineage.train_count or len(
        socket_exposed_names
    ) != checkpoint.lineage.exposed_count:
        raise ValueError("checkpoint lineage membership lists disagree with strict loader")
    socket_eval_names = socket_exposed_names - socket_train_names
    lineage_membership = {
        filename: {
            "socket_train1024": filename in socket_train_names,
            "socket_eval32": filename in socket_eval_names,
            "socket_any_checkpoint_exposure": filename in socket_exposed_names,
            "pasha_historical_validation_and_model_selection": 6700 <= index < 7000,
        }
        for index, filename in zip(INDICES, filenames, strict=True)
    }
    if any(
        membership["socket_any_checkpoint_exposure"]
        for membership in lineage_membership.values()
    ):
        raise ValueError("matched boards unexpectedly overlap frozen Socket checkpoint lineage")

    freeze_started = perf_counter()
    frozen = [
        _freeze_one(
            item,
            inputs=args.inputs,
            output_dir=output_dir,
            model=checkpoint.model,
            device=device,
            checkpoint_sha256=checkpoint_sha256,
        )
        for item in pasha_dirty
    ]
    freeze_seconds = perf_counter() - freeze_started
    freeze_manifest = {
        "schema": "aiijc-socket-pasha-matched-dirty-freeze-v1",
        "contains_reference_layouts": False,
        "contains_target_pixels": False,
        "checkpoint_sha256": checkpoint_sha256,
        "indices": list(INDICES),
        "filenames": filenames,
        "fusion": (
            "exactly 0.5*row_rank_percentile(Pasha raw) + "
            "0.5*row_rank_percentile(Socket partial-OT real block); self excluded"
        ),
        "decoder": "frozen ORBIT buddies96",
        "artifacts": frozen,
    }
    freeze_manifest_path = output_dir / "dirty_freeze.json"
    freeze_manifest_path.write_text(
        json.dumps(freeze_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Reference-assisted phase: every Socket/fusion score and layout is now on disk.
    pasha_report_path = args.pasha_dir / "report.json"
    pasha_report = json.loads(pasha_report_path.read_text(encoding="utf-8"))
    pasha_boards = {int(row["index"]): row for row in pasha_report["boards"]}
    if set(pasha_boards) != set(INDICES):
        raise ValueError("Pasha report does not contain exactly the four matched indices")

    evaluation_started = perf_counter()
    rows: list[dict[str, Any]] = []
    with np.load(args.permutations) as reference_archive:
        names = reference_archive["names"]
        inverse = reference_archive["inv"]
        confidence = reference_archive["conf"]
        for item, pasha in zip(frozen, pasha_dirty, strict=True):
            index = int(item["index"])
            filename = str(item["filename"])
            if str(names[index]) != filename:
                raise ValueError("recovered reference identity mismatches frozen score identity")
            reference = inverse[index].astype(np.int64)
            with np.load(item["path"]) as socket_archive:
                raw_right = socket_archive["socket_raw_right"]
                raw_down = socket_archive["socket_raw_down"]
                ot_right = socket_archive["socket_ot_right"]
                ot_down = socket_archive["socket_ot_down"]
                fusion_right = socket_archive["fusion_right"]
                fusion_down = socket_archive["fusion_down"]
                fusion_layout = socket_archive["fusion_buddies96_layout"].astype(np.int32)
                fusion_objective = float(socket_archive["fusion_buddies96_objective"])

            local = {
                "pasha883_raw": pasha883_directional_retrieval_metrics(
                    pasha["right"], pasha["down"], reference, ks=KS
                ),
                "socket_raw": pasha883_directional_retrieval_metrics(
                    raw_right, raw_down, reference, ks=KS
                ),
                "socket_partial_ot": pasha883_directional_retrieval_metrics(
                    ot_right, ot_down, reference, ks=KS
                ),
                "pasha_socket_ot_rank50": pasha883_directional_retrieval_metrics(
                    fusion_right, fusion_down, reference, ks=KS
                ),
            }
            reported = pasha_boards[index]
            if reported["filename"] != filename:
                raise ValueError("Pasha report filename mismatches dirty-only artifact")
            pasha_differences = {
                key: local["pasha883_raw"][key] - float(reported["pasha883_local"][key])
                for key in local["pasha883_raw"]
            }
            if max(abs(value) for value in pasha_differences.values()) > 1e-12:
                raise ValueError("recomputed Pasha retrieval differs from its frozen report")

            geometry = evaluate_layout(
                fusion_layout,
                reference,
                reference_is_exact=False,
            ).as_dict()
            dirty = split_tiles(_load_rgb(args.inputs / filename))
            target = _load_rgb(args.targets / filename)
            fusion_global = geometry | {
                "raw_ssim": contest_ssim(target, assemble_tiles(dirty[fusion_layout])),
                "solver": "buddies_96",
                "objective": fusion_objective,
                "layout_digest": layout_digest(fusion_layout),
            }
            rows.append(
                {
                    "index": index,
                    "filename": filename,
                    "lineage_membership": lineage_membership[filename],
                    "reference_confidence_mean": float(confidence[index].mean()),
                    "reference_confidence_below_0_5_fraction": float(
                        np.mean(confidence[index] < 0.5)
                    ),
                    "local": local,
                    "pasha_report_reproduction_max_abs_error": max(
                        abs(value) for value in pasha_differences.values()
                    ),
                    "global": {
                        "pasha883_buddies96_from_report": reported["global"][
                            "pasha883_buddies"
                        ],
                        "fusion_buddies96": fusion_global,
                    },
                }
            )
            print(
                json.dumps(
                    {
                        "event": "evaluated",
                        "index": index,
                        "socket_ot_pooled_r1": local["socket_partial_ot"]["pooled_r1"],
                        "fusion_pooled_r1": local["pasha_socket_ot_rank50"]["pooled_r1"],
                        "fusion_adjacency": fusion_global["adjacency"],
                    }
                ),
                flush=True,
            )
    evaluation_seconds = perf_counter() - evaluation_started

    local_aggregate = {
        name: mean_numeric_metrics(
            [{"metrics": row["local"][name]} for row in rows],
            "metrics",
        )
        for name in rows[0]["local"]
    }
    fusion_global_rows = [
        {"fusion_buddies96": row["global"]["fusion_buddies96"]} for row in rows
    ]
    fusion_global = _global_mean(fusion_global_rows)
    pasha_global = pasha_report["aggregate"]["global"]["pasha883_buddies"]
    global_deltas = {
        key: fusion_global[key] - float(pasha_global[key])
        for key in (
            "direct_placement",
            "row_accuracy",
            "column_accuracy",
            "translation_aligned_placement",
            "adjacency",
            "raw_ssim",
        )
    }
    report = {
        "experiment": "socket-v2-d64-vs-pasha883-matched-last4-v1",
        "status": "source-exposed-exploratory-diagnostic-do-not-promote",
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
            "architecture": checkpoint.contract["architecture"],
            "dimension": checkpoint.contract["dimension"],
            "lineage_train_count": checkpoint.lineage.train_count,
            "lineage_exposed_count": checkpoint.lineage.exposed_count,
            "matched_board_lineage_membership": lineage_membership,
        },
        "pasha_report": {
            "path": str(pasha_report_path.resolve()),
            "sha256": sha256_file(pasha_report_path),
        },
        "reference_cache": {
            "path": str(args.permutations.resolve()),
            "sha256": sha256_file(args.permutations),
            "contract": "target-assisted recovered permutation; inv[position] = input tile",
        },
        "protocol": {
            "indices": list(INDICES),
            "filenames_discovered_from": "Pasha dirty-only score artifacts",
            "candidate_pool": "all 576 input tiles; query self masked; 552 queries per axis",
            "rank_ties": "one plus strictly-greater score count; exact ties share best rank",
            "socket_variants": ["raw", "partial-OT real 576x576 block"],
            "single_fusion": (
                "0.5*stable descending row-rank percentile(Pasha raw) + "
                "0.5*stable descending row-rank percentile(Socket partial-OT); "
                "self excluded, highest=1, lowest=0, diagonal=-1"
            ),
            "fusion_decoder": "frozen ORBIT buddies96",
            "all_socket_scores_and_fusion_layouts_frozen_before_reference_or_targets": True,
            "source_exposure": (
                "all four boards are source-disjoint from Socket d64 train1024 and eval32, "
                "but all four belong to Pasha's historical validation/model-selection roster"
            ),
            "interpretation_limit": (
                "matched pixels/corruptions make Socket-vs-Pasha descriptive comparison fairer, "
                "but Pasha/fusion are source-exposed and the reference is target-assisted "
                "recovery rather than organizer ground truth"
            ),
            "competition_test_opened": False,
            "promotion_allowed": False,
        },
        "dirty_freeze_manifest": {
            "path": str(freeze_manifest_path.resolve()),
            "sha256": sha256_file(freeze_manifest_path),
        },
        "frozen_scores": frozen,
        "runtime_seconds": {
            "dirty_only_freeze_wall": freeze_seconds,
            "dirty_only_socket_model_sum": float(
                sum(item["socket_inference_seconds"] for item in frozen)
            ),
            "reference_assisted_evaluation": evaluation_seconds,
        },
        "boards": rows,
        "aggregate": {
            "local": local_aggregate,
            "socket_raw_minus_pasha_raw": _local_deltas(
                local_aggregate["socket_raw"], local_aggregate["pasha883_raw"]
            ),
            "socket_partial_ot_minus_pasha_raw": _local_deltas(
                local_aggregate["socket_partial_ot"], local_aggregate["pasha883_raw"]
            ),
            "fusion_minus_pasha_raw": _local_deltas(
                local_aggregate["pasha_socket_ot_rank50"],
                local_aggregate["pasha883_raw"],
            ),
            "global": {
                "pasha883_buddies96_from_report": pasha_global,
                "fusion_buddies96": fusion_global,
                "fusion_minus_pasha883": global_deltas,
            },
        },
        "verdict": (
            "Matched source-exposed diagnostic only. Do not promote either Socket or fusion "
            "from these four historically exposed recovered-reference boards."
        ),
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "event": "complete",
                "report": str(report_path),
                "local": local_aggregate,
                "global": report["aggregate"]["global"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
